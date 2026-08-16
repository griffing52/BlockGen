"""Growth sequences: a build as an ordered list of (piece, where-it-attached).

This is the representation the pick-and-place model consumes. It is a sibling of
`attach_order.py`, and the difference is the whole point of the model:

* `attach_order` makes placement **implicit** — the parent is whatever a fixed
  frontier heap pops next, so the stream is coordinate-free and the model only
  chooses the piece. That is what T21 built, and T21's verdict was *blindness*:
  the model learned op frequencies but "cannot read its own op history as
  geometry", and shutting the frontier down faster the more real structure it
  was shown.
* `growth_order` makes placement **explicit** — every step records the parent
  node and the face used, so a model can *learn where to attach* rather than
  inheriting it from a hand-written priority function. `implementation_plan.md`
  §3 names exactly this (a state encoder over the placed structure, predicting
  per-face) as the prerequisite for the track working at all.

A `GrowthSequence` is nodes in placement order. Node 0 is the seed; every later
node names an already-placed parent and one of the six face directions. That
invariant is what makes the sequence replayable, and `growth_to_structure`
+ `roundtrip_iou` prove it — Phase 0 of T21 was exactly this check, and it is
the first thing to run before trusting any model built on top.

**Legality, precomputed.** A face `(i, d)` is available at step `t` iff node `i`
is placed and the cell it points at is empty. Since a cell is occupied exactly
when its node index is `<= t`, storing the node index that eventually occupies
each neighbour cell (`neighbor_node`, `-1` for never-occupied) makes the whole
time-varying legality mask a comparison against `t`:

    legal(i, d, t)  ==  (i <= t) and not (0 <= neighbor_node[i, d] <= t)

which vectorizes over all steps at once. No per-step set membership, no Python
loop in the training path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from blockgen.utils.data import Structure
from blockgen.utils.graph_data import PORT_DIRECTIONS

Coord = Tuple[int, int, int]

N_DIR = len(PORT_DIRECTIONS)
_DIRS = np.array(PORT_DIRECTIONS, dtype=np.int32)          # [6, 3]

ORDERINGS = ("bfs", "dfs", "layered")


def pack_piece(block_id: int, block_data: int, oriented: bool = False) -> int:
    """Piece token identity. Mirrors `attach_order._pack_token`."""
    if not oriented:
        return int(block_id)
    return int(block_id) * 16 + (int(block_data) & 0xF)


def unpack_piece(token: int, oriented: bool = False) -> Tuple[int, int]:
    if not oriented:
        return int(token), 0
    return int(token) // 16, int(token) % 16


@dataclass
class GrowthSequence:
    """One build, in placement order.

    All arrays are length ``N`` (number of placed voxels) and indexed by
    placement step. Node 0 is the seed.
    """

    coords: np.ndarray          # [N, 3] int32, absolute (seed-relative after norm)
    pieces: np.ndarray          # [N]    int64, piece token per node
    parent: np.ndarray          # [N]    int64, index of the node it attached to (-1 seed)
    direction: np.ndarray       # [N]    int64, PORT_DIRECTIONS index used   (-1 seed)
    neighbor_node: np.ndarray   # [N, 6] int64, node index occupying that cell, else -1
    source_path: Optional[str] = None

    def __post_init__(self) -> None:
        n = len(self.coords)
        for name, arr, shape in (("pieces", self.pieces, (n,)),
                                 ("parent", self.parent, (n,)),
                                 ("direction", self.direction, (n,)),
                                 ("neighbor_node", self.neighbor_node, (n, N_DIR))):
            if tuple(arr.shape) != shape:
                raise ValueError(f"{name} has shape {arr.shape}, expected {shape}")
        if n and (self.parent[0] != -1 or self.direction[0] != -1):
            raise ValueError("node 0 must be the seed (parent=-1, direction=-1)")
        if n > 1 and not np.all(self.parent[1:] < np.arange(1, n)):
            raise ValueError("every node must attach to an EARLIER node")

    def __len__(self) -> int:
        return len(self.coords)

    @property
    def n_nodes(self) -> int:
        return len(self.coords)

    def placement_masks(self) -> np.ndarray:
        """``[N, N, 6]`` bool: entry ``[t, i, d]`` = may node ``t`` attach to face
        ``(i, d)``?

        Indexed by *the node being placed*, so row ``t`` describes the world with
        nodes ``0..t-1`` present. Row 0 (the seed) is all-False by construction --
        the seed has no parent. Keeping the index on "who is being placed" rather
        than "how many are placed" removes the off-by-one that would silently let
        a node attach to itself.
        """
        n = self.n_nodes
        t = np.arange(n, dtype=np.int64)[:, None, None]            # [N,1,1]
        idx = np.arange(n, dtype=np.int64)[None, :, None]          # [1,N,1]
        nbr = self.neighbor_node[None, :, :]                       # [1,N,6]
        placed = idx < t                                           # parent precedes t
        cell_taken = (nbr >= 0) & (nbr < t)                        # cell already filled
        return placed & ~cell_taken

    def target_ports(self) -> np.ndarray:
        """``[N]`` flattened port index ``parent*6 + direction`` (-1 for the seed)."""
        out = np.where(self.parent >= 0, self.parent * N_DIR + self.direction, -1)
        return out.astype(np.int64)


def _neighbor_table(coords: np.ndarray) -> np.ndarray:
    """``[N, 6]`` index of the node occupying each neighbour cell, else -1."""
    lookup: Dict[Coord, int] = {tuple(int(v) for v in c): i
                                for i, c in enumerate(coords)}
    out = np.full((len(coords), N_DIR), -1, dtype=np.int64)
    for i, c in enumerate(coords):
        for d in range(N_DIR):
            nb = (int(c[0] + _DIRS[d, 0]), int(c[1] + _DIRS[d, 1]),
                  int(c[2] + _DIRS[d, 2]))
            j = lookup.get(nb)
            if j is not None:
                out[i, d] = j
    return out


def _largest_component(occ: np.ndarray) -> np.ndarray:
    """Boolean mask of the largest 6-connected component."""
    from scipy import ndimage
    lab, n = ndimage.label(occ, structure=ndimage.generate_binary_structure(3, 1))
    if n <= 1:
        return occ
    sizes = np.bincount(lab.ravel())[1:]
    return lab == (int(np.argmax(sizes)) + 1)


def _seed_index(coords: np.ndarray) -> int:
    """Lowest, then most central — a deterministic, decoder-reproducible anchor."""
    y = coords[:, 1]
    cand = np.flatnonzero(y == y.min())
    sub = coords[cand]
    centre = sub.mean(axis=0)
    d = ((sub[:, 0] - centre[0]) ** 2 + (sub[:, 2] - centre[2]) ** 2)
    # Break ties on raw coordinate so the choice never depends on array order.
    best = cand[np.lexsort((sub[:, 2], sub[:, 0], d))[0]]
    return int(best)


def _priority(ordering: str, coord: np.ndarray, seed: np.ndarray, step: int) -> tuple:
    if ordering == "bfs":
        return (step,)
    if ordering == "layered":            # fill low layers before climbing
        return (int(coord[1]), step)
    if ordering == "dfs":
        return (-step,)
    raise ValueError(f"ordering must be one of {ORDERINGS}, got {ordering!r}")


def structure_to_growth(
    structure: Structure,
    *,
    ordering: str = "bfs",
    oriented: bool = False,
    largest_component_only: bool = True,
    max_nodes: Optional[int] = None,
) -> Optional[GrowthSequence]:
    """Extract a growth sequence. Returns None for an empty structure.

    Growth is **connected by construction**: node 0 is the seed and every later
    node attaches to an already-placed one, which is what lets a decoder replay
    the sequence without coordinates.

    ``largest_component_only`` keeps the build connected. Roughly a third of real
    curated houses are multi-component, so this drops material — `growth_stats`
    reports how much rather than hiding it.
    """
    import heapq

    c = structure.crop_to_non_air()
    occ = c.occupied_mask
    if not occ.any():
        return None
    if largest_component_only:
        occ = _largest_component(occ)

    coords_all = np.argwhere(occ).astype(np.int32)
    if len(coords_all) == 0:
        return None
    lookup = {tuple(int(v) for v in p): k for k, p in enumerate(coords_all)}

    seed_i = _seed_index(coords_all)
    seed = coords_all[seed_i]

    order: List[int] = [seed_i]
    parents: List[int] = [-1]
    dirs: List[int] = [-1]
    seen = np.zeros(len(coords_all), dtype=bool)
    seen[seed_i] = True

    # (priority, tiebreak, node, parent_step, direction)
    heap: List[tuple] = []
    counter = 0
    step = 0

    def push_faces(node_idx: int, parent_step: int) -> None:
        nonlocal counter, step
        base = coords_all[node_idx]
        for d in range(N_DIR):
            nb = (int(base[0] + _DIRS[d, 0]), int(base[1] + _DIRS[d, 1]),
                  int(base[2] + _DIRS[d, 2]))
            j = lookup.get(nb)
            if j is None or seen[j]:
                continue
            step += 1
            heapq.heappush(heap, (_priority(ordering, coords_all[j], seed, step),
                                  counter, j, parent_step, d))
            counter += 1

    push_faces(seed_i, 0)
    while heap and (max_nodes is None or len(order) < max_nodes):
        _, _, node, parent_step, d = heapq.heappop(heap)
        if seen[node]:
            continue
        seen[node] = True
        parents.append(parent_step)
        dirs.append(d)
        order.append(node)
        push_faces(node, len(order) - 1)

    coords = coords_all[order]
    pieces = np.array([pack_piece(int(c.block_ids[tuple(p)]),
                                  int(c.block_data[tuple(p)]), oriented)
                       for p in coords], dtype=np.int64)
    return GrowthSequence(
        coords=coords.astype(np.int32), pieces=pieces,
        parent=np.array(parents, dtype=np.int64),
        direction=np.array(dirs, dtype=np.int64),
        neighbor_node=_neighbor_table(coords),
        source_path=structure.source_path)


def growth_to_structure(seq: GrowthSequence, *, oriented: bool = False,
                        n_steps: Optional[int] = None) -> Structure:
    """Replay a growth sequence into a Structure.

    Rebuilds coordinates by walking parent/direction from the seed rather than
    reading `seq.coords`, so a mismatch between the two -- the failure mode that
    would make the representation a lie -- shows up as a broken round-trip.
    """
    n = seq.n_nodes if n_steps is None else min(n_steps, seq.n_nodes)
    if n == 0:
        z = np.zeros((1, 1, 1), dtype=np.int32)
        return Structure(block_ids=z, block_data=z.copy())

    pos = np.zeros((n, 3), dtype=np.int64)
    for i in range(1, n):
        pos[i] = pos[seq.parent[i]] + _DIRS[seq.direction[i]]

    lo = pos.min(axis=0)
    pos -= lo
    dims = pos.max(axis=0) + 1
    ids = np.zeros(tuple(int(x) for x in dims), dtype=np.int32)
    data = np.zeros_like(ids)
    for i in range(n):
        bid, bdata = unpack_piece(int(seq.pieces[i]), oriented)
        ids[tuple(pos[i])] = bid
        data[tuple(pos[i])] = bdata
    return Structure(block_ids=ids, block_data=data, source_path=seq.source_path)


def sequence_to_reference(seq: GrowthSequence, *, oriented: bool = False) -> Structure:
    """The structure a sequence *claims* to encode, built from its own coords.

    This is the correct comparison target for a round-trip: encoding drops
    everything outside the largest component on purpose, so measuring the replay
    against the *original* structure conflates "the replay is wrong" with "we
    deliberately kept only one component". Those need separate numbers.
    """
    pos = seq.coords.astype(np.int64) - seq.coords.min(axis=0)
    dims = pos.max(axis=0) + 1
    ids = np.zeros(tuple(int(x) for x in dims), dtype=np.int32)
    data = np.zeros_like(ids)
    for i in range(seq.n_nodes):
        bid, bdata = unpack_piece(int(seq.pieces[i]), oriented)
        ids[tuple(pos[i])] = bid
        data[tuple(pos[i])] = bdata
    return Structure(block_ids=ids, block_data=data)


def roundtrip_iou(structure: Structure, *, ordering: str = "bfs",
                  oriented: bool = False) -> float:
    """Occupancy IoU of encode->replay. Must be exactly 1.0.

    Replay walks parent/direction from the seed and never reads `seq.coords`, so
    this is a real test that the coordinate-free stream reconstructs the geometry
    -- the property the whole representation rests on.

    The Phase 0 gate: T21 validated `attach_order` with exactly this check across
    4 orderings x 1,200 builds before trusting anything downstream. Component
    coverage is reported separately by `growth_stats["block_retention"]`.
    """
    seq = structure_to_growth(structure, ordering=ordering, oriented=oriented)
    if seq is None:
        return float("nan")
    a = sequence_to_reference(seq, oriented=oriented).occupied_mask
    b = growth_to_structure(seq, oriented=oriented).occupied_mask
    if a.shape != b.shape:
        return 0.0
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union else 1.0


def growth_stats(structures: Sequence[Structure], *, ordering: str = "bfs",
                 oriented: bool = False) -> dict:
    """Coverage/fidelity summary over a corpus. Report this before training."""
    kept, total, sizes, ious = 0, 0, [], []
    n_empty = 0
    for s in structures:
        occ = int(s.occupied_mask.sum())
        total += occ
        seq = structure_to_growth(s, ordering=ordering, oriented=oriented)
        if seq is None:
            n_empty += 1
            continue
        kept += seq.n_nodes
        sizes.append(seq.n_nodes)
        ious.append(roundtrip_iou(s, ordering=ordering, oriented=oriented))
    return {
        "n_structures": len(structures), "n_empty": n_empty,
        "block_retention": kept / total if total else 0.0,
        "median_nodes": float(np.median(sizes)) if sizes else 0.0,
        "max_nodes": int(max(sizes)) if sizes else 0,
        "mean_roundtrip_iou": float(np.mean(ious)) if ious else float("nan"),
        "min_roundtrip_iou": float(np.min(ious)) if ious else float("nan"),
    }
