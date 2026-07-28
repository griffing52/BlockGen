"""Attachment-op linearization of a build (implementation_plan.md §2).

A build is emitted not as absolute coordinates but as a sequence of **attachment
ops**, each op a decision about one *open face* (a port, in ``graph_data`` terms)
popped from a frontier:

    SEED(piece)                -- place the first piece at the anchor
    ATTACH(piece, direction)   -- fill the target face's neighbour with ``piece``
    CLOSE                      -- the target face is a boundary (the air replacement)

Pose is **derived**: the attached voxel sits at ``parent + PORT_DIRECTIONS[dir]``.
No X/Y/Z is ever emitted, so a non-connecting placement is inexpressible.

**Ordering is a parameter, not a constant.** ``implementation_plan.md`` §2 proposes
one canonical order (bottom-center seed, BFS the bottom plane, then climb). That
order is untested here and there is contrary in-repo evidence: notes.md §8/T11 found
BFS-from-ground token order was the *worst* non-broken arm for raster AR
(``ar_bfs`` val_nn 0.337; ``bfs_constrained`` 0.397 < ``raster_constrained`` 0.428).
The plan argues that finding does not transfer because here the order *is* the
generative process rather than a scan pattern to invert -- plausible, untested, and
load-bearing for the whole architecture. So every ordering is a strategy object and
selecting one is an experiment (``scripts/ordering_bakeoff.py``), not a decision
baked into the tokenizer.

**The decode-availability constraint (found empirically, 2026-07-21).** An ordering
must be a pure function of information the *decoder* also has: the face coordinates
and the partial structure built so far. It may NOT read the ground-truth occupancy.
A first draft included ``support_first`` ("prefer children resting on occupied
cells") and ``shell_first`` ("exterior before interior"); both scored the frontier
against the *final* build at encode time but against the *partial* build at decode
time, so the heap popped in a different order and parent assignment desynced --
round-trip IoU 0.23 instead of 1.0. This constraint is not stated in
``implementation_plan.md`` and it rules out any ordering defined over properties of
the finished structure. State-dependent orderings are still possible, but they need
a frontier that re-evaluates priorities at pop time (stale-key problem), which is
deferred; everything below is *static* -- computable from coordinates alone.

Orderings implemented (see ``ORDERINGS``), all static and all round-tripping:
  * ``bfs_bottom_center``  -- the plan's canonical order, §2: exhaust the current y
    (horizontal faces before +Y) before climbing.
  * ``layered_raster``     -- bottom-up by y-layer, raster *within* each layer.
    Keeps the gravity prior while restoring the raster regularity T11 measured as
    easier to learn than BFS.
  * ``radial``             -- outward from the seed by squared distance. True
    radial growth, the "grow the blob" prior.
  * ``raster``             -- pure (y, z, x) raster control, matching the ordering
    that won T11.

Priorities are shift-invariant (they compare coordinates, and encode/decode frames
differ only by a constant translation), which is what lets the decoder reproduce the
encoder's frontier exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import heapq

import numpy as np

from blockgen.utils.data import Structure
from blockgen.utils.graph_data import OPPOSITE_PORT, PORT_DIRECTIONS

Coord = Tuple[int, int, int]

# Op kinds.
OP_SEED = 0
OP_ATTACH = 1
OP_CLOSE = 2

_OP_NAMES = {OP_SEED: "SEED", OP_ATTACH: "ATTACH", OP_CLOSE: "CLOSE"}


@dataclass(frozen=True)
class Op:
    """One attachment decision.

    ``kind`` is one of OP_SEED / OP_ATTACH / OP_CLOSE.
    ``piece`` is the block token (SEED, ATTACH) or -1 (CLOSE).
    ``direction`` is the ``PORT_DIRECTIONS`` index of the target face being
    decided (ATTACH, CLOSE) or -1 (SEED). The *parent* voxel is implicit: it is
    whatever the frontier pops next, which the decoder reconstructs by replaying
    the same ordering. That is what makes the stream coordinate-free.
    """

    kind: int
    piece: int = -1
    direction: int = -1

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        name = _OP_NAMES[self.kind]
        if self.kind == OP_SEED:
            return f"SEED({self.piece})"
        if self.kind == OP_CLOSE:
            return f"CLOSE(d{self.direction})"
        return f"ATTACH({self.piece},d{self.direction})"


# --------------------------------------------------------------------------
# Piece tokens
# --------------------------------------------------------------------------

def _pack_token(block_id: int, block_data: int, oriented: bool) -> int:
    """Pack (block_id, data) into one int. Mirrors serialize's token identity."""
    if not oriented:
        return int(block_id)
    return int(block_id) * 16 + (int(block_data) & 0xF)


def _unpack_token(token: int, oriented: bool) -> Tuple[int, int]:
    if not oriented:
        return int(token), 0
    return int(token) // 16, int(token) % 16


# --------------------------------------------------------------------------
# Ordering strategies
# --------------------------------------------------------------------------
#
# An ordering is a *priority function* over open faces. The frontier is a heap
# keyed by (priority_tuple, tiebreak_counter), so every ordering is deterministic
# and a pure function of occupancy -- the property cluster_bpe relies on.
#
# priority_fn(parent, direction, child, ctx) -> tuple
#   parent    : the placed voxel owning this face
#   direction : PORT_DIRECTIONS index of the face
#   child     : parent + PORT_DIRECTIONS[direction] (the cell being decided)
#   ctx       : OrderingContext with occupancy + derived fields

@dataclass
class OrderingContext:
    """Decode-available context. Deliberately holds NO ground-truth occupancy.

    ``seed`` is the anchor voxel of the current component -- known to the decoder
    because it places the seed itself. Anything added here must be reproducible at
    decode time or round-trip breaks (see the module docstring).
    """

    seed: Coord


PriorityFn = Callable[[Coord, int, Coord, OrderingContext], tuple]


def _prio_bfs_bottom_center(parent: Coord, direction: int, child: Coord,
                            ctx: OrderingContext) -> tuple:
    """implementation_plan.md §2: exhaust the current y before any face that raises y.

    Horizontal faces before +Y, so the bottom plane finishes before climbing.
    """
    is_vertical = 1 if direction in (2, 3) else 0
    return (child[1] - ctx.seed[1], is_vertical, direction)


def _prio_layered_raster(parent: Coord, direction: int, child: Coord,
                         ctx: OrderingContext) -> tuple:
    """Bottom-up by layer, raster (z, x) within the layer."""
    return (child[1] - ctx.seed[1], child[2] - ctx.seed[2], child[0] - ctx.seed[0])


def _prio_radial(parent: Coord, direction: int, child: Coord,
                 ctx: OrderingContext) -> tuple:
    """Outward from the seed by squared distance -- true radial growth."""
    dx = child[0] - ctx.seed[0]
    dy = child[1] - ctx.seed[1]
    dz = child[2] - ctx.seed[2]
    return (dx * dx + dy * dy + dz * dz, dy, dz, dx)


def _prio_dfs(parent: Coord, direction: int, child: Coord,
              ctx: OrderingContext) -> tuple:
    """Depth-first: no coordinate priority; ordered purely LIFO (see ``LIFO_ORDERINGS``).

    Grows a tendril as deep as it can before backtracking -- the natural contrast
    to the breadth-first family, and the order under which a long run of ATTACHes
    is most locally predictable.
    """
    return ()


ORDERINGS: Dict[str, PriorityFn] = {
    "bfs_bottom_center": _prio_bfs_bottom_center,
    "layered_raster": _prio_layered_raster,
    "radial": _prio_radial,
    "dfs": _prio_dfs,
}

# Orderings whose frontier is LIFO rather than FIFO. The tie-break counter is
# negated so the most recently pushed face pops first.
LIFO_ORDERINGS = {"dfs"}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _in_bounds(c: Coord, arr: np.ndarray) -> bool:
    return (0 <= c[0] < arr.shape[0] and 0 <= c[1] < arr.shape[1]
            and 0 <= c[2] < arr.shape[2])


def _surface_mask(occ: np.ndarray) -> np.ndarray:
    """Occupied voxels exposed to empty space (or the grid edge) on >=1 face."""
    exposed = np.zeros_like(occ, dtype=bool)
    for dx, dy, dz in PORT_DIRECTIONS:
        shifted = np.ones_like(occ, dtype=bool)  # out-of-bounds counts as empty
        src = np.roll(occ, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        sl = [slice(None)] * 3
        for axis, d in enumerate((dx, dy, dz)):
            if d > 0:
                sl[axis] = slice(0, occ.shape[axis] - d)
            elif d < 0:
                sl[axis] = slice(-d, occ.shape[axis])
        shifted[tuple(sl)] = src[tuple(sl)]
        exposed |= occ & ~shifted
    return exposed


def choose_seed(occ: np.ndarray) -> Coord:
    """Bottom-center seed (implementation_plan.md §2.1).

    Minimum y, then the voxel closest to the (x, z) centroid of that bottom
    layer, ties broken lexicographically by (x, z).
    """
    coords = np.argwhere(occ)
    if coords.size == 0:
        raise ValueError("empty structure has no seed")
    y_min = int(coords[:, 1].min())
    bottom = coords[coords[:, 1] == y_min]
    cx, cz = bottom[:, 0].mean(), bottom[:, 2].mean()
    d2 = (bottom[:, 0] - cx) ** 2 + (bottom[:, 2] - cz) ** 2
    best = np.lexsort((bottom[:, 2], bottom[:, 0], d2))[0]
    return tuple(int(v) for v in bottom[best])


# --------------------------------------------------------------------------
# Encode
# --------------------------------------------------------------------------

def structure_to_attach_ops(
    structure: Structure,
    *,
    ordering: str = "bfs_bottom_center",
    oriented: bool = False,
    max_ops: Optional[int] = None,
    multi_component: str = "largest",
) -> Tuple[List[Op], Dict[str, object]]:
    """Linearize a build into attachment ops under ``ordering``.

    ``multi_component`` decides the §9 open question -- disconnected builds cannot
    be grown from a single seed, since the growth model reaches only what is
    6-connected to the seed:
      * ``"largest"``  -- keep the largest connected component, drop the rest.
      * ``"reseed"``   -- emit a fresh SEED per component, largest-first.
      * ``"reject"``   -- raise if the build is not fully connected.

    Returns ``(ops, info)``; ``info`` records the coverage actually achieved so a
    lossy multi-component policy can never be silently mistaken for a clean
    round-trip.
    """
    if ordering not in ORDERINGS:
        raise ValueError(f"unknown ordering {ordering!r}; have {sorted(ORDERINGS)}")
    prio = ORDERINGS[ordering]

    s = structure.crop_to_non_air()
    occ = s.occupied_mask
    total_occupied = int(occ.sum())
    if total_occupied == 0:
        raise ValueError("cannot linearize an empty structure")

    components = _connected_components(occ)
    components.sort(key=len, reverse=True)
    if multi_component == "reject" and len(components) > 1:
        raise ValueError(f"structure has {len(components)} components")
    if multi_component == "largest":
        components = components[:1]
    elif multi_component not in ("reseed", "largest"):
        raise ValueError(f"unknown multi_component {multi_component!r}")

    lifo = ordering in LIFO_ORDERINGS

    ops: List[Op] = []
    placed: set = set()
    truncated = False

    for comp in components:
        comp_occ = np.zeros_like(occ)
        for c in comp:
            comp_occ[c] = True
        seed = choose_seed(comp_occ)
        ctx = OrderingContext(seed=seed)

        token = _pack_token(s.block_ids[seed], s.block_data[seed], oriented)
        ops.append(Op(OP_SEED, piece=token))
        placed.add(seed)

        # Frontier: heap of (priority, counter, parent, direction).
        frontier: List[tuple] = []
        counter = 0
        visited_faces: set = set()

        def push_faces(voxel: Coord) -> None:
            nonlocal counter
            for d, (dx, dy, dz) in enumerate(PORT_DIRECTIONS):
                child = (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)
                face = (voxel, d)
                if face in visited_faces:
                    continue
                # A face pointing at an already-placed voxel is an internal,
                # already-satisfied edge -- it is never emitted (§2.4).
                if child in placed:
                    continue
                visited_faces.add(face)
                tie = -counter if lifo else counter
                heapq.heappush(frontier, (prio(voxel, d, child, ctx), tie, voxel, d))
                counter += 1

        push_faces(seed)

        while frontier:
            if max_ops is not None and len(ops) >= max_ops:
                truncated = True
                break
            _p, _c, parent, d = heapq.heappop(frontier)
            dx, dy, dz = PORT_DIRECTIONS[d]
            child = (parent[0] + dx, parent[1] + dy, parent[2] + dz)

            # Resolved in the meantime by another face's attachment.
            if child in placed:
                continue

            if _in_bounds(child, occ) and occ[child]:
                token = _pack_token(s.block_ids[child], s.block_data[child], oriented)
                ops.append(Op(OP_ATTACH, piece=token, direction=d))
                placed.add(child)
                push_faces(child)
            else:
                ops.append(Op(OP_CLOSE, direction=d))

    info = {
        "ordering": ordering,
        "n_ops": len(ops),
        "n_placed": len(placed),
        "n_occupied": total_occupied,
        "coverage": len(placed) / total_occupied,
        "n_components": len(_connected_components(occ)),
        "truncated": truncated,
        "shape": tuple(int(v) for v in s.shape),
        "n_attach": sum(1 for o in ops if o.kind == OP_ATTACH),
        "n_close": sum(1 for o in ops if o.kind == OP_CLOSE),
    }
    return ops, info


def _connected_components(occ: np.ndarray) -> List[List[Coord]]:
    """6-connected components of the occupancy, as coordinate lists."""
    from collections import deque

    remaining = {tuple(int(v) for v in c) for c in np.argwhere(occ)}
    comps: List[List[Coord]] = []
    while remaining:
        seed = min(remaining)
        q = deque([seed])
        remaining.discard(seed)
        comp = []
        while q:
            v = q.popleft()
            comp.append(v)
            for dx, dy, dz in PORT_DIRECTIONS:
                n = (v[0] + dx, v[1] + dy, v[2] + dz)
                if n in remaining:
                    remaining.discard(n)
                    q.append(n)
        comps.append(comp)
    return comps


# --------------------------------------------------------------------------
# Decode
# --------------------------------------------------------------------------

def attach_ops_to_structure(
    ops: Sequence[Op],
    *,
    ordering: str = "bfs_bottom_center",
    oriented: bool = False,
    pad: int = 2,
    grid: int = 256,
) -> Structure:
    """Replay ops into a Structure. Inverse of ``structure_to_attach_ops``.

    The decoder rebuilds the frontier with the *same* priority function, which is
    what lets the op stream omit coordinates entirely: parent identity is implied
    by frontier position. ``grid`` sizes the working volume; the result is cropped.
    """
    if ordering not in ORDERINGS:
        raise ValueError(f"unknown ordering {ordering!r}")
    prio = ORDERINGS[ordering]

    lifo = ordering in LIFO_ORDERINGS

    # Sparse placement. A free-running model can grow past ANY preallocated grid
    # (it emitted index -257 into a 256^3 array before this was sparse), and a
    # fixed working volume would quietly reintroduce the box this representation
    # exists to remove. Coordinates are unbounded here and materialized at the end.
    cells: Dict[Coord, Tuple[int, int]] = {}

    placed: set = set()
    frontier: List[tuple] = []
    counter = 0
    visited_faces: set = set()
    origin = (grid // 2, pad, grid // 2)
    ctx = OrderingContext(seed=origin)

    def push_faces(voxel: Coord) -> None:
        nonlocal counter
        for d, (dx, dy, dz) in enumerate(PORT_DIRECTIONS):
            child = (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)
            face = (voxel, d)
            if face in visited_faces or child in placed:
                continue
            visited_faces.add(face)
            tie = -counter if lifo else counter
            heapq.heappush(frontier, (prio(voxel, d, child, ctx), tie, voxel, d))
            counter += 1

    def place(voxel: Coord, token: int) -> None:
        bid, bdata = _unpack_token(token, oriented)
        cells[voxel] = (bid, bdata)
        placed.add(voxel)

    for op in ops:
        if op.kind == OP_SEED:
            # Each SEED starts a fresh component. Offset far enough that separate
            # components cannot touch, and reset the frontier + ordering context so
            # the decoder mirrors the encoder's per-component state exactly.
            # NOTE: the op stream encodes no *inter-component* offset -- only
            # intra-component connectivity. So "reseed" can never reproduce the
            # original relative placement of disconnected components; it is
            # geometry-lossy BY CONSTRUCTION, and roundtrip_iou will show that.
            # "largest" is the policy used for corpus runs for exactly this reason.
            n_seeded = sum(1 for o in ops[:ops.index(op) + 1] if o.kind == OP_SEED)
            anchor = (origin[0], origin[1], origin[2] + (n_seeded - 1) * (grid // 16))
            frontier.clear()
            visited_faces.clear()
            counter = 0
            ctx = OrderingContext(seed=anchor)
            place(anchor, op.piece)
            push_faces(anchor)
            continue

        while frontier:
            _p, _c, parent, d = heapq.heappop(frontier)
            dx, dy, dz = PORT_DIRECTIONS[d]
            child = (parent[0] + dx, parent[1] + dy, parent[2] + dz)
            if child in placed:
                continue
            if op.kind == OP_ATTACH:
                place(child, op.piece)
                push_faces(child)
            break

    if not cells:
        return Structure(block_ids=np.zeros((1, 1, 1), dtype=np.int32),
                         block_data=np.zeros((1, 1, 1), dtype=np.int32))

    coords = np.array(list(cells.keys()))
    lo = coords.min(axis=0)
    shape = tuple(int(v) for v in (coords.max(axis=0) - lo + 1))
    block_ids = np.zeros(shape, dtype=np.int32)
    block_data = np.zeros(shape, dtype=np.int32)
    for (x, y, z), (bid, bdata) in cells.items():
        idx = (x - lo[0], y - lo[1], z - lo[2])
        block_ids[idx] = bid
        block_data[idx] = bdata
    return Structure(block_ids=block_ids, block_data=block_data)


def roundtrip_iou(structure: Structure, *, ordering: str = "bfs_bottom_center",
                  oriented: bool = False, multi_component: str = "largest") -> float:
    """Encode → decode → occupancy IoU against the input. The Phase-0 gate."""
    ops, _info = structure_to_attach_ops(
        structure, ordering=ordering, oriented=oriented,
        multi_component=multi_component)
    rebuilt = attach_ops_to_structure(ops, ordering=ordering, oriented=oriented)

    a = structure.crop_to_non_air().occupied_mask
    b = rebuilt.occupied_mask
    if multi_component == "largest":
        # Compare against the component actually encoded, not the whole build.
        comps = _connected_components(a)
        comps.sort(key=len, reverse=True)
        keep = np.zeros_like(a)
        for c in comps[0]:
            keep[c] = True
        a = keep
        # Recrop to the kept component's bounding box.
        idx = np.argwhere(a)
        lo, hi = idx.min(0), idx.max(0) + 1
        a = a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    shape = tuple(max(x, y) for x, y in zip(a.shape, b.shape))
    pa = np.zeros(shape, dtype=bool)
    pb = np.zeros(shape, dtype=bool)
    pa[:a.shape[0], :a.shape[1], :a.shape[2]] = a
    pb[:b.shape[0], :b.shape[1], :b.shape[2]] = b
    union = (pa | pb).sum()
    return float((pa & pb).sum() / union) if union else 1.0
