"""Scorer ``interior``: does the build have rooms, and is there anything in them?

Why a separate, voxel-only metric: the staged methods (blueprint, floors) spend
their budget on interiors, which no exterior render shows, so the in-loop image
scores cannot tell a furnished house from a hollow shell. ``interior`` is
measured AFTER a run for every arm alike (``rescore <run> --scorers interior``)
and is never part of a recipe's ``score`` / ``final_score``: a metric the loop
optimizes stops measuring anything (Goodhart), and an offline one is uniform
across arms that do and do not look inside.

Definitions (pure numpy/scipy on the cropped build, fast at 32^3):

* **enclosed air**: air that is cut off from the outside. Two tests, either is
  enough. (a) *sealed*: not 6-connected to the canvas border, the floor of the
  canvas excluded (the build stands on the ground, so a house without a floor
  slab still holds its rooms). (b) *covered*: a ray along each of the six axis
  directions hits a block before leaving the canvas (downwards the ground
  counts). Why (b): a doorway left as air (common in LLM builds) connects every
  room to the outside, and with (a) alone a single missing door block would zero
  the metric; (b) still drops the cells lined up with the opening, which is the
  right answer for them.
* **rooms**: 6-connected components of enclosed air with >= ``min_room`` voxels.
* **furnish**: blocks inside the build's x/z bbox that touch enclosed air and are
  not the outer hull (touching outside air), not a separator (touching two
  different air pockets: a partition or a door), not part of a wall (a vertical
  run of >= ``wall_run`` blocks: a partition whose doorway was left open joins
  the two pockets, so the separator test alone misses it) and not on a storey
  slab plane (a y level whose blocks cover >= ``slab_frac`` of the bbox interior).

value = 0.5 * min(1, rooms / 2) + 0.5 * min(1, furnish / (``furnish_frac`` *
enclosed air)); a solid block and an open platform both score 0.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from blockgen.agentic.core import Attempt, ScoreTerm, register

#: 6-neighbourhood (face-adjacent), for flood fill and "touches".
_SIX = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                 [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                 [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], bool)


def _covered(occ: np.ndarray) -> np.ndarray:
    """Air whose rays along +-x, +y, +-z hit a block (downwards: the ground always does)."""
    def hit(axis: int, forward: bool) -> np.ndarray:
        a = occ if forward else np.flip(occ, axis)
        # strictly after the cell: cumulative OR, shifted by one along the axis
        c = np.maximum.accumulate(np.flip(a, axis), axis=axis)
        c = np.flip(c, axis)
        sh = np.zeros_like(c)
        sl_dst = [slice(None)] * 3
        sl_src = [slice(None)] * 3
        sl_dst[axis] = slice(0, -1)
        sl_src[axis] = slice(1, None)
        sh[tuple(sl_dst)] = c[tuple(sl_src)]
        return sh if forward else np.flip(sh, axis)
    out = ~occ
    for axis, forward in ((0, True), (0, False), (1, True), (2, True), (2, False)):
        out &= hit(axis, forward)
    return out


def _vertical_runs(occ: np.ndarray) -> np.ndarray:
    """Length of the contiguous vertical (y) run of blocks each block belongs to."""
    up = np.zeros(occ.shape, np.int32)
    down = np.zeros(occ.shape, np.int32)
    for y in range(occ.shape[1]):
        up[:, y] = np.where(occ[:, y], (up[:, y - 1] if y else 0) + 1, 0)
    for y in range(occ.shape[1] - 1, -1, -1):
        down[:, y] = np.where(occ[:, y], (down[:, y + 1] if y + 1 < occ.shape[1] else 0) + 1, 0)
    return np.where(occ, up + down - 1, 0)


def interior_stats(occ: np.ndarray, *, min_room: int = 8, slab_frac: float = 0.5,
                   wall_run: int = 4) -> Dict[str, int]:
    """Raw counts for an occupancy grid (x, y-up, z): enclosed air, rooms, furnish."""
    from scipy import ndimage

    occ = np.asarray(occ, bool)
    if not occ.any():
        return {"enclosed_air": 0, "rooms": 0, "furnish": 0, "blocks": 0}
    # Pad one air layer on every side except the bottom (y = 0 is the ground).
    padded = np.pad(occ, ((1, 1), (0, 1), (1, 1)))
    air = ~padded
    seed = np.zeros_like(air)
    seed[0], seed[-1] = air[0], air[-1]
    seed[:, -1] |= air[:, -1]
    seed[:, :, 0] |= air[:, :, 0]
    seed[:, :, -1] |= air[:, :, -1]
    outside = ndimage.binary_propagation(seed, structure=_SIX, mask=air)
    enclosed = (air & ~outside) | _covered(padded)
    outside = air & ~enclosed
    lab, n = ndimage.label(enclosed, structure=_SIX)
    sizes = np.bincount(lab.ravel())[1:] if n else np.zeros(0, int)
    rooms = int((sizes >= int(min_room)).sum())
    touch_in = ndimage.binary_dilation(enclosed, structure=_SIX) & padded
    hull = ndimage.binary_dilation(outside, structure=_SIX) & padded
    # A block touching two different air pockets separates them: a partition wall
    # or a door, not furniture.
    big = np.where(lab > 0, lab, np.iinfo(lab.dtype).max)
    sep = (ndimage.maximum_filter(lab, footprint=_SIX, mode="constant", cval=0)
           != ndimage.minimum_filter(big, footprint=_SIX, mode="constant",
                                     cval=np.iinfo(lab.dtype).max)) & padded
    sep &= ndimage.maximum_filter(lab, footprint=_SIX, mode="constant", cval=0) > 0
    xs, _, zs = np.nonzero(padded)
    x0, x1, z0, z1 = xs.min(), xs.max(), zs.min(), zs.max()
    # Slab planes are judged on the bbox interior (shrunk by 1 where it can be):
    # on a small footprint the perimeter walls alone cover half the bbox.
    ix0, ix1 = (x0 + 1, x1 - 1) if x1 - x0 >= 2 else (x0, x1)
    iz0, iz1 = (z0 + 1, z1 - 1) if z1 - z0 >= 2 else (z0, z1)
    area = float((ix1 - ix0 + 1) * (iz1 - iz0 + 1))
    per_y = padded[ix0:ix1 + 1, :, iz0:iz1 + 1].sum(axis=(0, 2))
    slab = np.zeros_like(padded)
    slab[:, per_y >= slab_frac * area, :] = True
    inbox = np.zeros_like(padded)
    inbox[x0:x1 + 1, :, z0:z1 + 1] = True
    wall = _vertical_runs(padded) >= int(wall_run)
    furnish = int((touch_in & ~hull & ~slab & ~sep & ~wall & inbox).sum())
    return {"enclosed_air": int(enclosed.sum()), "rooms": rooms, "furnish": furnish,
            "blocks": int(occ.sum())}


def interior_value(stats: Dict[str, int], *, furnish_frac: float = 0.06) -> float:
    rooms_term = min(1.0, stats["rooms"] / 2.0)
    air = stats["enclosed_air"]
    furn_term = min(1.0, stats["furnish"] / (furnish_frac * air)) if air > 0 else 0.0
    return 0.5 * rooms_term + 0.5 * furn_term


@register("scorer", "interior", needs_dream=False, needs_renders=False)
def interior(ep: Any, att: Attempt, *, min_room: int = 8, furnish_frac: float = 0.06,
             slab_frac: float = 0.5, wall_run: int = 4) -> ScoreTerm:
    """Rooms and furnishing from voxels (see the module docstring). Measure-only:
    rescore runs with it, do not put it in a recipe's score."""
    w = getattr(att, "world", None)
    if w is None or att.blocks <= 0:
        return ScoreTerm(0.0, note="interior: empty build")
    s = w.to_structure(crop=True)
    st = interior_stats(s.occupied_mask, min_room=min_room, slab_frac=slab_frac,
                        wall_run=wall_run)
    v = interior_value(st, furnish_frac=furnish_frac)
    return ScoreTerm(float(v), note=(f"interior: {st['rooms']} room(s), "
                                     f"{st['enclosed_air']} enclosed air, "
                                     f"{st['furnish']} furnishing block(s)"))


__all__ = ["interior", "interior_stats", "interior_value"]
