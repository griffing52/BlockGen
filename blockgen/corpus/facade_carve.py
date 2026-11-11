"""Space-carving fusion of a face set: the classical alternative to Cross-Face decoding.

For a face set the visual hull keeps every voxel that no face's ray passes through (a
see-through cell empties its whole ray; a hit at depth d empties the d voxels in front of
it). The carved build is the visible shell of that hull, each surface voxel taking the
majority block of the faces whose ray first hits it there, or failing that of every face
whose ray passes through it. Unlike the overwrite `facade.lift`, carving never lets a face
add a voxel another face saw through, so disagreements remove content instead of adding
fragments. Used as the baseline of the faces paper (results: faces paper, carving section).
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Sequence, Tuple

import numpy as np

from blockgen.utils.data import Structure

from . import facade as fc


def _dist_grid(face: str, shape: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every voxel, its depth along `face`'s ray (0 = nearest the face), and (u, v) index grids."""
    axis, sign = fc._parse(face)
    ua, va = fc._uv_axes(axis)
    idx = np.indices(tuple(shape))
    t = idx[axis]
    d = t if sign > 0 else shape[axis] - 1 - t
    return d, idx[ua], idx[va]


def carve(faces: Dict[str, fc.Facade], shape: Sequence[int]) -> Structure:
    """The carved build of a face set (may be empty when one see-through cell clears every ray)."""
    shape = tuple(int(s) for s in shape)
    empty = np.zeros(shape, bool)
    for f, F in faces.items():
        d, u, v = _dist_grid(f, shape)
        dep = F.depth[u, v]
        empty |= (dep < 0) | (d < dep)
    hull = ~empty
    hull_s = Structure(block_ids=hull.astype(np.int32), block_data=np.zeros(shape, np.int32))
    surf = fc.lift(fc.project_all(hull_s, tuple(faces)), shape).block_ids > 0
    votes_hit: Dict[tuple, Counter] = {}
    votes_any: Dict[tuple, Counter] = {}
    for f, F in faces.items():
        d, u, v = _dist_grid(f, shape)
        dep = F.depth[u, v]
        ids, dat = F.ids[u, v], F.data[u, v]
        hit = surf & (dep >= 0) & (d == dep)
        through = surf & (dep >= 0) & (d >= dep)
        for m, store in ((hit, votes_hit), (through, votes_any)):
            for p, b, bd in zip(zip(*np.nonzero(m)), ids[m], dat[m]):
                store.setdefault(p, Counter())[(int(b), int(bd))] += 1
    out_ids = np.zeros(shape, np.int32)
    out_dat = np.zeros(shape, np.int32)
    for p in zip(*np.nonzero(surf)):
        c = votes_hit.get(p) or votes_any.get(p)
        if c:
            (b, bd), _ = c.most_common(1)[0]
            out_ids[p], out_dat[p] = b, bd
    return Structure(block_ids=out_ids, block_data=out_dat)


def kept_fraction(faces: Dict[str, fc.Facade], s: Structure) -> float:
    """Share of the face cells that the build `s` reproduces exactly under projection."""
    tot = same_n = 0
    for f, F in faces.items():
        B = fc.project(s, f)
        same = (F.mask == B.mask) & ((~F.mask) | ((F.depth == B.depth) & (F.ids == B.ids) & (F.data == B.data)))
        tot += same.size
        same_n += int(same.sum())
    return same_n / max(tot, 1)
