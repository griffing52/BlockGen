"""Traversability: can a player actually walk into and through this build?

The bench's interior measure (`geometry.interior_volumes`) says whether a build
*has* rooms, counting air that becomes enclosed once doors and windows up to two
blocks wide are closed. It does not say whether anyone can get into them. A
house with a 1-block doorway, a 2-block ceiling and a staircase with no headroom
has the same interior ratio as a house a player can live in, and the agentic
pilots (notes.md section 38) produce exactly that doll's-house failure: an
interior that reads fine in the isometric view and is unusable inside.

This module answers the question a Minecraft player would ask, with the rules a
Minecraft player obeys. A player occupies one column two blocks tall, stands on
a solid block, steps up one block (that covers stairs, slabs and a jump), falls
any distance, climbs ladders and vines, and walks through doors, trapdoors,
gates, carpets, torches and plants, which the renderer draws but no body is
stopped by. The build stands on flat ground one block below its lowest layer,
and the player starts anywhere on that ground outside the footprint. Everything
reachable from there is the *reach set*; everything in the reach set that is
also interior air is interior a player can visit.

Three things fall out of one flood fill:

* **interior_standing_reached** -- the share of interior cells a player could
  stand in that a player can actually reach from outside. 1.0 is a house you
  can walk every room of; 0.0 is a sealed shell or a doll's house. This is the
  number the bench's interior ratio cannot give.
* **floors_reached / floors** -- storeys with a reachable floor patch against
  storeys with any floor patch. A staircase that does not work shows up here
  and nowhere else.
* **entrances** -- doorways actually used: clusters of reached interior
  standing cells one step from a reached exterior one.

Two probes make the measure falsifiable, in the style of `probes.solidify`:
`seal_openings` bricks up every aperture (interior volume unchanged, reach goes
to zero) and `lower_ceilings` drops every room's ceiling by one block (silhouette
and interior ratio nearly unchanged, a player no longer fits where the room was
exactly two high). The bench's own `solidify` is the third: no interior, no
reach.

Everything here is integer geometry on a padded boolean grid and runs in
milliseconds per build. Nothing in `blockgen.eval.bench` is modified; the interior
masks are recomputed with the same closing recipe and pinned to
`geometry.interior_volumes` by test.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from blockgen.utils.data import Structure

Pos = Tuple[int, int, int]

# --- what a body passes through ----------------------------------------------
#: Legacy numeric ids a player walks through. Doors, trapdoors and fence gates
#: open; carpets, snow layers, pressure plates, rails, torches, signs, buttons,
#: levers, tripwire, banners, plants and crops have no collision box a body
#: notices. Water is swimmable. Fences, walls, panes, bars, beds, chests, slabs
#: and stairs stay solid: a player cannot walk through them and can stand on
#: them, which is what "solid" means here.
PASSABLE_IDS = frozenset({
    6, 8, 9, 27, 28, 30, 31, 32, 37, 38, 39, 40, 50, 51, 55, 59, 63, 64, 65,
    66, 68, 69, 70, 71, 72, 75, 76, 77, 78, 83, 96, 104, 105, 106, 107, 111,
    115, 127, 131, 132, 141, 142, 143, 147, 148, 157, 167, 171, 175, 176, 177,
    183, 184, 185, 186, 187, 193, 194, 195, 196, 197, 198, 207,
})
#: Cells a player can move straight up and down in: ladders, vines, water.
CLIMBABLE_IDS = frozenset({8, 9, 65, 106})
#: Door blocks (wooden and iron). A door is where the builder meant you to
#: come in, and in the world it was reachable from the ground; a crop that cut
#: the ground away leaves it two blocks up a sheer foundation. The outside walk
#: therefore also starts in every doorway that holds a door.
DOOR_IDS = frozenset({64, 71, 193, 194, 195, 196, 197})
#: Everything that opens: doors, trapdoors, fence gates. `seal_openings` turns
#: these to wall along with the empty doorways and windows.
OPENABLE_IDS = DOOR_IDS | frozenset({96, 167, 107, 183, 184, 185, 186, 187})
#: Half-height steps: stairs and single slabs. Walking up one of these is a
#: half-block rise and needs no jump, so a staircase under a two-block ceiling
#: works, as it does in the game. Double slabs (43, 125, 181, 204) are full.
STEP_IDS = frozenset({53, 67, 108, 109, 114, 128, 134, 135, 136, 156, 163, 164,
                      180, 203, 44, 126, 182, 205})
#: Ground, as opposed to building: stone, grass, dirt, bedrock, sand, gravel,
#: ores, snow, clay, mycelium, grass path. Many corpus crops carry a slab of
#: terrain under the house, so the outside ground is not the crop's lowest layer
#: but the top of this material wherever it meets the crop's boundary. A player
#: on the ring may step onto a boundary cell that stands on terrain at any
#: height; that is walking up the hill the house sits on, not climbing the house.
NATURAL_IDS = frozenset({1, 2, 3, 7, 12, 13, 14, 15, 16, 21, 56, 73, 80, 82,
                         110, 208})

#: Marker blocks for the walkthrough frames (wool colours). Only floors are
#: recoloured, so rooms stay open to the camera.
_WOOL = 35
TRAIL_BLOCK = (_WOOL, 5)       # lime: interior floor the player has reached
PLAYER_BLOCK = (_WOOL, 14)     # red: the two-block stand-in
UNREACHED_BLOCK = (_WOOL, 3)   # light blue: interior floor still to reach


def solid_mask(s: Structure) -> np.ndarray:
    """Cells a body cannot occupy: every non-air block that is not passable."""
    ids = s.block_ids
    solid = ids != s.air_block_id
    if solid.any():
        solid &= ~np.isin(ids, list(PASSABLE_IDS))
    return solid


def climbable_mask(s: Structure) -> np.ndarray:
    return np.isin(s.block_ids, list(CLIMBABLE_IDS))


# --- interior air, the bench's recipe ---------------------------------------
_STRUCT6 = ndimage.generate_binary_structure(3, 1)
_STRUCT26 = ndimage.generate_binary_structure(3, 3)
_PAD = 3


def _with_ground(occ: np.ndarray) -> np.ndarray:
    """The build standing on a solid ground layer (prepended along y)."""
    g = np.ones((occ.shape[0], 1, occ.shape[2]), dtype=bool)
    return np.concatenate([g, occ], axis=1)


def interior_masks(occ: np.ndarray, ground: bool = False
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """`(sealed, open)` interior air masks.

    Sealed air is unreachable from the bounding box without crossing solid.
    Open interior is air that becomes sealed once apertures up to two blocks
    wide are closed: rooms with a door or a window. With `ground=False` the
    counts equal `geometry.interior_volumes` exactly (pinned by
    `tests/eval/test_traversal.py`); the masks are recomputed here because the
    bench function returns counts only.

    `ground=True` sets the build on a solid ground layer first. A corpus crop
    often takes the house and not the ground it stood on, so a floorless
    ground-floor room is open to the outside through the bottom of the box and
    counts as no room at all under the bench definition. For walkability the
    ground is there, as it is in the game, and this module uses `ground=True`
    throughout.
    """
    occ = np.asarray(occ, dtype=bool)
    if not occ.any():
        z = np.zeros(occ.shape, dtype=bool)
        return z, z.copy()
    if ground:
        s, o = interior_masks(_with_ground(occ), ground=False)
        return s[:, 1:, :], o[:, 1:, :]
    filled = ndimage.binary_fill_holes(occ, structure=_STRUCT6)
    sealed = filled & ~occ
    closed = ndimage.binary_closing(np.pad(occ, _PAD), structure=_STRUCT26)
    filled_c = ndimage.binary_fill_holes(closed, structure=_STRUCT6)
    open_int = (filled_c & ~closed)[_PAD:-_PAD, _PAD:-_PAD, _PAD:-_PAD] & ~sealed & ~occ
    return sealed, open_int


def aperture_mask(occ: np.ndarray, ground: bool = False) -> np.ndarray:
    """Air cells the closing fills: the doorways and windows themselves."""
    occ = np.asarray(occ, dtype=bool)
    if not occ.any():
        return np.zeros(occ.shape, dtype=bool)
    if ground:
        return aperture_mask(_with_ground(occ), ground=False)[:, 1:, :]
    closed = ndimage.binary_closing(np.pad(occ, _PAD), structure=_STRUCT26)
    return closed[_PAD:-_PAD, _PAD:-_PAD, _PAD:-_PAD] & ~occ


# --- the player -------------------------------------------------------------
#: Padding around the build: one cell of outside on each side in x and z, a
#: ground layer below, two cells of sky above so a head on the roof fits.
_PADS = ((1, 1), (1, 2), (1, 1))


def _unpad(a: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Padded grid -> build coordinates."""
    nx, ny, nz = shape
    return a[1:1 + nx, 1:1 + ny, 1:1 + nz]


@dataclass
class _Grids:
    """The boolean grids the walk reads, all in padded coordinates."""

    solid: np.ndarray
    climb: np.ndarray
    free: np.ndarray
    head_free: np.ndarray
    standing: np.ndarray       # free feet, free head, solid below (or climbable)


def _grids(s: Structure, blind: bool = False) -> _Grids:
    """`blind=True` is the occupancy-only player: every block is a wall, nothing
    opens, nothing climbs, no half steps. Exactly blind to material, which is
    what the geometric descriptor requires of anything it carries."""
    if blind:
        S = np.pad(s.occupied_mask, _PADS)
        C = np.zeros_like(S)
    else:
        S = np.pad(solid_mask(s), _PADS)
        C = np.pad(climbable_mask(s), _PADS)
    S[:, 0, :] = True              # the ground
    free = ~S
    below = np.zeros_like(S)
    below[:, 1:, :] = S[:, :-1, :]
    head_free = np.zeros_like(free)
    head_free[:, :-1, :] = free[:, 1:, :]
    standing = free & head_free & (below | C)
    return _Grids(S, C, free, head_free, standing)


@dataclass
class Walk:
    """One flood fill, in padded coordinates (`_unpad` returns to the build's).
    Feet cells may lie one outside the footprint, on the surrounding ground."""

    reach: np.ndarray            # bool, feet cells a player can stand in
    dist: np.ndarray             # BFS distance per feet cell (-1 = unreached)
    shape: Tuple[int, int, int]  # the cropped build's shape

    def feet(self) -> np.ndarray:
        """Reached feet cells in build coordinates (cells outside are dropped)."""
        return _unpad(self.reach, self.shape)

    def body(self) -> np.ndarray:
        """Cells a reached body occupies (feet and head), build coordinates."""
        f = self.feet()
        head = np.zeros_like(f)
        head[:, 1:, :] = f[:, :-1, :]
        return f | head

    def frontier(self, d: int) -> np.ndarray:
        """Feet cells at exactly BFS distance `d`, build coordinates."""
        return _unpad(self.dist == d, self.shape)

    def reached_by(self, d: int) -> np.ndarray:
        """Feet cells at BFS distance at most `d`, build coordinates."""
        return _unpad((self.dist >= 0) & (self.dist <= d), self.shape)

    @property
    def max_dist(self) -> int:
        return int(self.dist.max())


def exterior_air(occ: np.ndarray, ground: bool = False) -> np.ndarray:
    """Air 6-connected to the outside of the bounding box (`probes.exterior_air`)."""
    occ = np.asarray(occ, dtype=bool)
    if ground:
        return exterior_air(_with_ground(occ), ground=False)[:, 1:, :]
    padded = np.pad(occ, 1)
    lab, _ = ndimage.label(~padded, structure=_STRUCT6)
    return (lab == lab[0, 0, 0])[1:-1, 1:-1, 1:-1]


def walk(s: Structure, start: str = "outside", blind: bool = False) -> Walk:
    """Flood fill of every feet cell a player can stand in, from `start`.

    `blind=True` reads occupancy only (see `_grids`): doors are walls, stairs
    need a jump like any block, ladders do not climb, and the walk does not
    start in doorways. It is the variant the geometric descriptor carries.

    `start="outside"` (the default) seeds every standing cell in exterior air
    that is neither interior nor an open doorway or window cell, plus every
    doorway holding a door block: a player set down anywhere outside the
    building, on the ground, a porch, an outside stair or a balcony, or in the
    front door. This tests the building, not the site: a corpus crop often
    cuts the hill a house stands on, leaves a yard wall with its gate outside
    the crop, starts an outside staircase in mid-air or leaves the front door
    two blocks up a sheer foundation, and none of those is a fact about the
    house. `start="ground"` is the stricter site reading: seeds
    on the ring of ground around the footprint, plus the crop's edge where it
    stands on natural terrain. `start="anywhere"` seeds every standing cell,
    inside or out, which answers "is the interior connected" rather than "can
    you get in".

    Movement, per BFS step, from feet cell (x, y, z):

    * walk to a 4-neighbour if its feet and head cells are free, then fall to
      the first solid below (any height);
    * step up to a neighbour one higher if the target's feet and head are free
      and either the current head has a block of jump room above it or the
      step is a half-block rise (stairs or a slab under the source or the
      target, `STEP_IDS`), which needs no jump;
    * in a climbable cell, or with one directly overhead, move straight up or
      down one cell when the destination is free.

    "Free" means not solid; "standing" means free feet, free head, solid below
    (or in a climbable cell, which holds a body up).
    """
    g = _grids(s, blind)
    S, C, free, head_free, standing = g.solid, g.climb, g.free, g.head_free, g.standing
    nx, ny, nz = S.shape
    step = (np.zeros_like(S) if blind
            else np.pad(np.isin(s.block_ids, list(STEP_IDS)), _PADS))

    seed = np.zeros_like(standing)
    if start == "outside":
        occ = s.occupied_mask
        sealed, open_int = interior_masks(occ, ground=True)
        outside = (exterior_air(occ, ground=True) & ~sealed & ~open_int
                   & ~aperture_mask(occ, ground=True))
        doors = (np.zeros(s.shape, dtype=bool) if blind
                 else np.isin(s.block_ids, list(DOOR_IDS)))
        ring = np.zeros_like(standing)
        ring[0, :, :] = ring[-1, :, :] = True
        ring[:, :, 0] = ring[:, :, -1] = True
        seed = standing & (ring | np.pad(outside | doors, _PADS))
    elif start == "ground":
        ring = np.zeros_like(standing)
        ring[0, :, :] = ring[-1, :, :] = True
        ring[:, :, 0] = ring[:, :, -1] = True
        # the crop's outermost columns, where they stand on terrain
        natural = np.pad(np.isin(s.block_ids, list(NATURAL_IDS)), _PADS)
        on_terrain = np.zeros_like(standing)
        on_terrain[:, 1:, :] = natural[:, :-1, :]
        edge = np.zeros_like(standing)
        edge[1, :, :] = edge[-2, :, :] = True
        edge[:, :, 1] = edge[:, :, -2] = True
        seed = standing & (ring | (edge & on_terrain))
    elif start == "anywhere":
        seed = standing
    else:
        raise ValueError(f"unknown start {start!r}")

    dist = np.full(S.shape, -1, dtype=np.int32)
    seeds = np.argwhere(seed)
    for x, y, z in seeds:
        dist[x, y, z] = 0
    frontier: List[Pos] = [tuple(int(v) for v in p) for p in seeds]
    d = 0

    def land(x: int, y: int, z: int) -> Optional[Pos]:
        """Where a body entering free cell (x, y, z) comes to rest."""
        while y > 0 and not (S[x, y - 1, z] or C[x, y, z]):
            y -= 1
        return (x, y, z) if standing[x, y, z] else None

    while frontier:
        d += 1
        nxt: List[Pos] = []
        for x, y, z in frontier:
            cands: List[Optional[Pos]] = []
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                X, Z = x + dx, z + dz
                if not (0 <= X < nx and 0 <= Z < nz):
                    continue
                if free[X, y, Z] and head_free[X, y, Z]:
                    cands.append(land(X, y, Z))
                # step up: a free standing cell one higher, reached with a
                # jump (room over the current head) or up a half-block step
                if y + 1 < ny and standing[X, y + 1, Z]:
                    half = step[x, y - 1, z] or step[X, y, Z]
                    if half or (y + 2 < ny and free[x, y + 2, z]):
                        cands.append((X, y + 1, Z))
            # climb
            if C[x, y, z] or (y + 1 < ny and C[x, y + 1, z]):
                if y + 1 < ny and standing[x, y + 1, z]:
                    cands.append((x, y + 1, z))
            if C[x, y, z] and y - 1 >= 0 and standing[x, y - 1, z]:
                cands.append((x, y - 1, z))
            for c in cands:
                if c is not None and dist[c] < 0:
                    dist[c] = d
                    nxt.append(c)
        frontier = nxt
    return Walk(reach=dist >= 0, dist=dist, shape=tuple(s.shape))


# --- the metrics ------------------------------------------------------------
#: A floor patch is a 4-connected set of interior standing cells on one y level
#: at least this big and at least two wide each way. The inside of a pitched
#: roof is a staircase of one-wide ledges, a windowsill is a strip, a table is
#: a cell; none of those is a storey, and all of them are standing cells.
FLOOR_MIN_CELLS = 6
FLOOR_MIN_SIDE = 2

TRAVERSAL_METRICS = (
    "interior_standing_reached",
    "floors",
    "floors_reached",
    "entrances",
    "headroom3",
    "enterable",
)


@dataclass
class TraversalStats:
    interior_cells: int
    interior_standing: int             # interior cells a player could stand in
    interior_standing_reached: float   # share of those reachable from outside
    floors: int                        # y levels with a floor patch
    floors_reached: int                # of those, levels with a reached patch cell
    entrances: int                     # doorways used: clusters of reached interior cells beside reached exterior ones
    headroom3: float                   # share of interior standing cells with a third free block above
    enterable: bool
    reach_cells: int                   # every reached feet cell, inside and out

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def floor_levels(int_stand: np.ndarray) -> Dict[int, np.ndarray]:
    """`{y: mask}` of floor patches per level (see `FLOOR_MIN_CELLS`)."""
    out: Dict[int, np.ndarray] = {}
    struct = ndimage.generate_binary_structure(2, 1)
    for y in range(int_stand.shape[1]):
        layer = int_stand[:, y, :]
        if layer.sum() < FLOOR_MIN_CELLS:
            continue
        lab, n = ndimage.label(layer, structure=struct)
        keep = np.zeros_like(layer)
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None:
                continue
            comp = lab[sl] == i
            if (comp.sum() >= FLOOR_MIN_CELLS
                    and min(comp.shape) >= FLOOR_MIN_SIDE):
                keep[sl] |= comp
        if keep.any():
            out[y] = keep
    return out


def standing_mask(s: Structure, blind: bool = False) -> np.ndarray:
    """Cells a player could stand in (feet cell), build coordinates, reachable or not."""
    return _unpad(_grids(s, blind).standing, tuple(s.shape))


def traversal_stats(s: Structure, w: Optional[Walk] = None,
                    blind: bool = False) -> TraversalStats:
    """All traversal scalars for one build (cropped first)."""
    c = s.crop_to_non_air()
    occ = c.occupied_mask
    if w is None:
        w = walk(c, blind=blind)
    sealed, open_int = interior_masks(occ, ground=True)
    interior = sealed | open_int
    n_int = int(interior.sum())
    feet = w.feet()
    standing = standing_mask(c, blind)
    int_stand = standing & interior
    n_stand = int(int_stand.sum())
    reached_int_stand = feet & int_stand

    # storeys: y levels with a floor patch, reached if any patch cell is
    levels = floor_levels(int_stand)
    floors_reached = int(sum(1 for y, m in levels.items()
                             if (m & reached_int_stand[:, y, :]).any()))

    # entrances: clusters of reached interior standing cells with a reached
    # exterior feet neighbour (same level or one step down/up). The doorway
    # cell itself is aperture air, not interior, so it is the exterior side.
    ext_feet = feet & ~interior
    ext_pad = np.pad(ext_feet, 1)
    near = np.zeros_like(ext_feet)
    for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for dy in (-1, 0, 1):
            near |= ext_pad[1 + dx:ext_feet.shape[0] + 1 + dx,
                            1 + dy:ext_feet.shape[1] + 1 + dy,
                            1 + dz:ext_feet.shape[2] + 1 + dz]
    _, entrances = ndimage.label(reached_int_stand & near, structure=_STRUCT26)

    # headroom: a third free cell over an interior standing cell
    solid = occ if blind else solid_mask(c)
    free3 = np.zeros_like(solid)
    free3[:, :-2, :] = ~solid[:, 2:, :]
    head3 = float((int_stand & free3).sum() / n_stand) if n_stand else float("nan")

    return TraversalStats(
        interior_cells=n_int,
        interior_standing=n_stand,
        # A build with no interior standing cell has no walkable interior: 0,
        # not NaN, so a sealed shell and a doll's house score alike and the
        # summary's `no_interior_frac` tells them apart.
        interior_standing_reached=float(reached_int_stand.sum() / n_stand) if n_stand else 0.0,
        floors=int(len(levels)),
        floors_reached=floors_reached,
        entrances=int(entrances),
        headroom3=head3,
        enterable=bool(reached_int_stand.any()),
        reach_cells=int(w.reach.sum()),
    )


def summarize(structures: Sequence[Structure]) -> Dict[str, Dict[str, float]]:
    """Distribution summary over builds: median, mean, and the share of builds
    that are enterable and that have every floor reachable. NaNs (no interior)
    are excluded from the ratios and counted in `no_interior_frac`."""
    rows = [traversal_stats(s) for s in structures]
    out: Dict[str, Dict[str, float]] = {}
    for k in ("interior_standing_reached", "headroom3"):
        v = np.array([getattr(r, k) for r in rows], dtype=float)
        v = v[~np.isnan(v)]
        out[k] = {"median": float(np.median(v)) if v.size else float("nan"),
                  "mean": float(v.mean()) if v.size else float("nan"),
                  "n": int(v.size)}
    n = max(len(rows), 1)
    with_floors = [r for r in rows if r.floors > 0]
    out["builds"] = {
        "n": len(rows),
        "no_interior_frac": float(sum(1 for r in rows if r.interior_standing == 0) / n),
        "enterable_frac": float(sum(1 for r in rows if r.enterable) / n),
        "all_floors_reached_frac": (float(sum(1 for r in with_floors if r.floors_reached == r.floors)
                                          / len(with_floors)) if with_floors else float("nan")),
        "floors_mean": float(np.mean([r.floors for r in rows])) if rows else float("nan"),
        "floors_reached_mean": float(np.mean([r.floors_reached for r in rows])) if rows else float("nan"),
        "entrances_mean": float(np.mean([r.entrances for r in rows])) if rows else float("nan"),
    }
    return out


# --- what the bench carries -------------------------------------------------
#: What the geometric descriptor carries: the player's *room*, not the player's
#: *route*. `walk_standing` is the share of interior air a two-block body can
#: stand in, `walk_floors` the number of storeys with a floor patch, and
#: `walk_headroom3` the share of interior standing cells with a third free block
#: above. All three read occupancy only, so the descriptor keeps its exact
#: blindness to material; a doll's house (rooms nobody fits in) and a stick
#: (no rooms) are both zero on all three. Reachability is deliberately *not*
#: here: whether a body can get in depends on which blocks are doors, and an
#: occupancy-only walk reads 70% of real houses as locked. That reading lives in
#: the material-aware `traversal` block and pillar. Undefined values are 0.
DESCRIPTOR_WALK = ("walk_standing", "walk_floors", "walk_headroom3")


def walk_scalars(s: Structure) -> Dict[str, float]:
    """`DESCRIPTOR_WALK` values for one build, from the blind player's grids.

    Exactly D4-invariant (every rule is symmetric under rotations and mirrors
    of the horizontal plane) and exactly blind to material (`blind=True` reads
    occupancy only).
    """
    r = traversal_stats(s, blind=True)
    return {
        "walk_standing": float(r.interior_standing / r.interior_cells) if r.interior_cells else 0.0,
        "walk_floors": float(r.floors),
        "walk_headroom3": float(r.headroom3) if np.isfinite(r.headroom3) else 0.0,
    }


#: The player's reading, reported per arm against the real distribution and
#: combined into the `traversal` BlockScore pillar. Material-aware: doors open,
#: stairs climb. `headroom3` is NaN where a build has no interior standing room
#: and the W1 distance skips it, as it does for every undefined scalar.
TRAVERSAL_REPORT = ("interior_standing_reached", "enterable", "floors_reached",
                    "headroom3")


def traversal_table(structures: Sequence[Structure]) -> Dict[str, np.ndarray]:
    """Per-metric arrays over a set of builds (column-major `traversal_stats`)."""
    rows = [traversal_stats(s) for s in structures]
    return {k: np.array([float(getattr(r, k)) for r in rows], dtype=float)
            for k in TRAVERSAL_REPORT}


def traversal_report(generated: Sequence[Structure], real: Sequence[Structure], *,
                     n_boot: int = 1000, alpha: float = 0.05,
                     rng: Optional[np.random.Generator] = None,
                     real_table: Optional[Dict[str, np.ndarray]] = None
                     ) -> Dict[str, dict]:
    """Generated-vs-real traversability in the shape `geometry.geometry_report`
    emits (`gen`, `real`, `w1`, `w1_norm`, `real_spread`, `direction`), so it
    renders through the same distance-to-real table and feeds `composite` the
    same way. Distance to real, never "higher is better": a generator that is
    *more* enterable than the corpus is also not building what the corpus has."""
    from blockgen.eval.bench import stats

    rng = rng or np.random.default_rng(0)
    gen = traversal_table(generated)
    ref = real_table if real_table is not None else traversal_table(real)
    out: Dict[str, dict] = {}
    for key in TRAVERSAL_REPORT:
        g, r = gen[key], ref[key]
        g, r = g[np.isfinite(g)], r[np.isfinite(r)]
        gp, glo, ghi = stats.bootstrap_ci(g, n_boot=n_boot, alpha=alpha, rng=rng)
        rp, rlo, rhi = stats.bootstrap_ci(r, n_boot=n_boot, alpha=alpha, rng=rng)
        d = stats.w1(g, r)
        spread = stats.robust_spread(r)
        out[key] = {"gen": {"mean": gp, "ci": [glo, ghi]},
                    "real": {"mean": rp, "ci": [rlo, rhi]},
                    "w1": d,
                    "w1_norm": (d / spread) if np.isfinite(spread) and spread > 0 else float("nan"),
                    "real_spread": spread,
                    "direction": "distance_to_real"}
    return out


# --- probes -----------------------------------------------------------------
def _dominant(s: Structure) -> Tuple[int, int]:
    occ = s.occupied_mask
    if not occ.any():
        return (4, 0)
    pairs = np.stack([s.block_ids[occ], s.block_data[occ]], axis=1)
    vals, counts = np.unique(pairs, axis=0, return_counts=True)
    a, b = vals[counts.argmax()]
    return int(a), int(b)


def seal_openings(s: Structure) -> Structure:
    """Brick up every doorway and window with the build's dominant material.

    Empty doorways and windows are filled, and every door, trapdoor and gate
    becomes wall. Interior volume is unchanged (open interior becomes sealed
    interior), the silhouette changes by the few cells of the apertures, and
    nobody can get in. A traversal measure must go to zero here; the interior
    ratio must not move. A build with nothing to seal is returned cropped and
    otherwise as is.
    """
    c = s.crop_to_non_air()
    occ = c.occupied_mask
    _, open_int = interior_masks(occ, ground=True)
    # Only the aperture cells that touch a room. The closing also fills nooks in
    # an ornate facade, and bricking those would change the exterior for no
    # reason; one layer of door or window is what seals the room.
    ap = aperture_mask(occ, ground=True) & ndimage.binary_dilation(open_int, _STRUCT6)
    ap |= np.isin(c.block_ids, list(OPENABLE_IDS))
    if not ap.any():
        return c
    bi, bd = c.block_ids.copy(), c.block_data.copy()
    a, b = _dominant(c)
    bi[ap], bd[ap] = a, b
    return Structure(block_ids=bi, block_data=bd, air_block_id=c.air_block_id)


def lower_ceilings(s: Structure) -> Structure:
    """Fill the top layer of every room: the doll's-house probe.

    For each interior air cell whose cell above is solid (the room's ceiling
    layer), place the dominant material. A room three blocks high becomes two
    (a player still fits), a room two high becomes one (nobody fits). Interior
    ratio drops by roughly one layer per room and the exterior is untouched, so
    the exterior-view metrics see nothing and the interior ratio sees little;
    the traversal measure sees rooms vanish exactly where they were already
    cramped. Builds without interior are returned cropped and unchanged.
    """
    c = s.crop_to_non_air()
    occ = c.occupied_mask
    sealed, open_int = interior_masks(occ, ground=True)
    interior = sealed | open_int
    if not interior.any():
        return c
    above_solid = np.zeros_like(occ)
    above_solid[:, :-1, :] = occ[:, 1:, :]
    top = interior & above_solid
    bi, bd = c.block_ids.copy(), c.block_data.copy()
    a, b = _dominant(c)
    bi[top], bd[top] = a, b
    return Structure(block_ids=bi, block_data=bd, air_block_id=c.air_block_id)


# --- walkthrough frames -----------------------------------------------------
def _under(feet: np.ndarray) -> np.ndarray:
    """The cell below each marked feet cell."""
    under = np.zeros_like(feet)
    under[:, :-1, :] = feet[:, 1:, :]
    return under


def walkthrough_frames(s: Structure, step: int = 1, mark_interior: bool = True,
                       trail: bool = True) -> Iterator[Structure]:
    """Structures for an animation of the flood fill, one per BFS distance.

    Frame `d` recolours the floor under every interior standing cell: lime
    where a player has stood by distance `d`, light blue where not yet (if
    `mark_interior`), and places a red two-block stand-in at every feet cell
    *at* distance `d`, the wavefront, inside or out. Only floors change, so
    the rooms stay open to the camera. Render these with a cutaway
    (`features.apply_cut` with a quadrant or roof-off spec) and stack the
    images into a GIF.
    """
    c = s.crop_to_non_air()
    w = walk(c)
    sealed, open_int = interior_masks(c.occupied_mask, ground=True)
    int_stand = standing_mask(c) & (sealed | open_int)
    solid = c.block_ids != c.air_block_id
    nx, ny, nz = c.shape
    for d in range(0, w.max_dist + 1, max(1, step)):
        bi, bd = c.block_ids.copy(), c.block_data.copy()
        feet = w.reached_by(d)
        if mark_interior:
            left = _under(int_stand & ~feet) & solid
            bi[left], bd[left] = UNREACHED_BLOCK
        if trail:
            paint = _under(int_stand & feet) & solid
            bi[paint], bd[paint] = TRAIL_BLOCK
        front = w.frontier(d)
        xs, ys, zs = np.nonzero(front)
        for x, y, z in zip(xs, ys, zs):
            for yy in (y, y + 1):
                if 0 <= yy < ny:
                    bi[x, yy, z], bd[x, yy, z] = PLAYER_BLOCK
        yield Structure(block_ids=bi, block_data=bd, air_block_id=c.air_block_id)
