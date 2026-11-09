"""Per-build face and fragmentation statistics, shared by the facade sheets and the rewards.

Moved out of ``scripts/facade_sheets.py`` so a reward can score the same numbers
the analysis tables report. The arithmetic is unchanged: 2D components use
8-connectivity on each face mask, 3D components 26-connectivity, and every
fraction is over the face's filled cells or the build's occupied voxels.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from blockgen.corpus import facade as fc
from blockgen.utils.data import Structure

#: Components with fewer blocks than this are dust.
DUST_MAX = 5

_FOUR = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])


def face_stats(s: Structure) -> dict:
    """Face fragmentation and 3D connectivity of one build.

    Keys: ``face_comps`` (8-connected components per face), ``face_holes``
    (see-through cells inside the face's filled hull, per filled cell),
    ``face_jumps`` (adjacent filled cells whose depth differs by two or more, per
    adjacent pair), ``face_lone`` (filled cells with no filled 4-neighbour, per
    filled cell), ``comps3d`` (26-connected components), ``lcc`` (largest
    component's share of the blocks), ``dust`` (components under ``DUST_MAX``
    blocks) and ``dust_blocks`` (their share of the blocks). The face means are
    over the faces that hold a block; a build with none gives NaN there.
    """
    s = s.crop_to_non_air()
    faces = fc.project_all(s, fc.ALL_FACES)
    comps, holes, jumps, lone = [], [], [], []
    for f in faces.values():
        m = f.mask
        if not m.any():
            continue
        _, n = ndi.label(m, structure=np.ones((3, 3)))
        comps.append(n)
        # holes: see-through cells inside the face's filled hull (filled 4-neighbourhood)
        filled = ndi.binary_fill_holes(m)
        holes.append(int((filled & ~m).sum()) / max(m.sum(), 1))
        # relief jumps between horizontally/vertically adjacent filled cells
        d = f.depth.astype(int)
        pairs = []
        for ax in (0, 1):
            a, b = np.moveaxis(m, ax, 0)[:-1], np.moveaxis(m, ax, 0)[1:]
            da, db = np.moveaxis(d, ax, 0)[:-1], np.moveaxis(d, ax, 0)[1:]
            both = a & b
            pairs.append((np.abs(da - db)[both] >= 2))
        pairs = np.concatenate(pairs)
        jumps.append(pairs.mean() if len(pairs) else 0.0)
        # lone cells: filled with no filled 4-neighbour
        nb = ndi.convolve(m.astype(int), _FOUR, mode='constant')
        lone.append(int(((nb == 0) & m).sum()) / max(m.sum(), 1))
    occ = s.occupied_mask
    lab, n3 = ndi.label(occ, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel())[1:]
    sizes.sort()
    big = sizes[-1] if len(sizes) else 0
    dust = int((sizes < DUST_MAX).sum())
    return dict(face_comps=np.mean(comps), face_holes=np.mean(holes), face_jumps=np.mean(jumps),
                face_lone=np.mean(lone), comps3d=n3, lcc=big / max(occ.sum(), 1), dust=dust,
                dust_blocks=int(sizes[sizes < DUST_MAX].sum()) / max(occ.sum(), 1))


def one_face_stats(f: fc.Facade) -> dict:
    """The face part of :func:`face_stats` for a single face: ``comps``, ``holes``,
    ``jumps``, ``lone`` as above, plus ``filled`` (cells) and ``fill_frac`` (filled cells
    over the face's rectangle), ``n_depths`` and ``n_blocks`` (distinct values)."""
    m = f.mask
    if not m.any():
        return dict(comps=0.0, holes=0.0, jumps=0.0, lone=0.0, filled=0.0, fill_frac=0.0,
                    n_depths=0.0, n_blocks=0.0)
    _, n = ndi.label(m, structure=np.ones((3, 3)))
    filled = ndi.binary_fill_holes(m)
    holes = int((filled & ~m).sum()) / max(m.sum(), 1)
    d = f.depth.astype(int)
    pairs = []
    for ax in (0, 1):
        a, b = np.moveaxis(m, ax, 0)[:-1], np.moveaxis(m, ax, 0)[1:]
        da, db = np.moveaxis(d, ax, 0)[:-1], np.moveaxis(d, ax, 0)[1:]
        both = a & b
        pairs.append(np.abs(da - db)[both] >= 2)
    pairs = np.concatenate(pairs)
    nb = ndi.convolve(m.astype(int), _FOUR, mode="constant")
    key = (f.ids.astype(int) * 16 + f.data.astype(int))[m]
    return dict(comps=float(n), holes=float(holes), jumps=float(pairs.mean()) if len(pairs) else 0.0,
                lone=float(((nb == 0) & m).sum() / max(m.sum(), 1)), filled=float(m.sum()),
                fill_frac=float(m.mean()), n_depths=float(len(np.unique(d[m]))),
                n_blocks=float(len(np.unique(key))))


def face_agreement(ref: fc.Facade, other: fc.Facade) -> dict:
    """How much of ``ref`` a same-shape face ``other`` reproduces: mask IoU, precision
    and recall; depth exactness and mean absolute error and block agreement on the
    cells filled in both."""
    a, b = ref.mask, other.mask
    both = a & b
    inter, union = both.sum(), (a | b).sum()
    out = dict(mask_iou=inter / max(union, 1), mask_precision=inter / max(b.sum(), 1),
               mask_recall=inter / max(a.sum(), 1))
    if both.any():
        dd = np.abs(ref.depth[both].astype(int) - other.depth[both].astype(int))
        out["depth_exact"] = float((dd == 0).mean())
        out["depth_mae"] = float(dd.mean())
        out["block_agree"] = float(((ref.ids == other.ids) & (ref.data == other.data))[both].mean())
    else:
        out.update(depth_exact=0.0, depth_mae=float("nan"), block_agree=0.0)
    return out
