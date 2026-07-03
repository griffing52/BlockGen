"""Data augmentation for block structures.

Small labeled subsets (a few hundred to ~1k builds per class) are the binding
constraint on generation quality. The cheapest, most semantically-safe multiplier
is the **horizontal dihedral group D4**: the four 90° rotations about the vertical
axis, optionally composed with a horizontal mirror — up to **8×** the data.

We deliberately do *not* flip vertically or rotate off-axis: buildings and
vehicles are gravity-oriented (roof up, wheels down), so only the horizontal
symmetries preserve semantics.

Caveat: block *orientation* is encoded in ``block_data`` for a handful of blocks
(stairs, logs, doors). We rotate the geometry (the arrays) but do **not** remap
those orientation values, so a rotated stair points a slightly-wrong way. This is
fine for shape learning (our headline metrics are occupancy-IoU based) and is
the standard trade-off; a future pass can canonicalize orientation `data`.

Augment the **training set only** — never the held-out val set used for novelty.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from blockgen.utils.data import Structure


# XYZ arrays with the vertical axis = Y (axis 1). Rotations about vertical act on
# the X-Z plane (axes 0 and 2); a horizontal mirror flips X (axis 0).
def _d4_arrays(block_ids: np.ndarray, block_data: np.ndarray):
    """Yield the (block_ids, block_data) arrays for each of the 8 D4 elements."""
    for mirror in (False, True):
        bi = block_ids[::-1] if mirror else block_ids
        bd = block_data[::-1] if mirror else block_data
        for k in range(4):
            yield np.rot90(bi, k, axes=(0, 2)).copy(), np.rot90(bd, k, axes=(0, 2)).copy()


def augment_structure(s: Structure, dedupe: bool = True) -> List[Structure]:
    """Return the D4 orbit of ``s`` (≤ 8 structures, identity first).

    With ``dedupe`` (default) rotationally-symmetric structures that map to
    themselves are collapsed, so a symmetric build isn't over-represented.
    """
    out: List[Structure] = []
    seen: set = set()
    for bi, bd in _d4_arrays(s.block_ids, s.block_data):
        if dedupe:
            key = bi.shape + (bi.tobytes(), bd.tobytes())
            if key in seen:
                continue
            seen.add(key)
        out.append(Structure(block_ids=bi, block_data=bd,
                             source_path=s.source_path, air_block_id=s.air_block_id,
                             metadata=dict(s.metadata)))
    return out


def augment_with_labels(structs: Sequence[Structure], labels: Sequence[str],
                        dedupe: bool = True) -> Tuple[List[Structure], List[str]]:
    """D4-augment a labeled list, expanding each label across its structure's orbit."""
    aug_s: List[Structure] = []
    aug_l: List[str] = []
    for s, lab in zip(structs, labels):
        orbit = augment_structure(s, dedupe=dedupe)
        aug_s.extend(orbit)
        aug_l.extend([lab] * len(orbit))
    return aug_s, aug_l


def augment(structs: Sequence[Structure], dedupe: bool = True) -> List[Structure]:
    """D4-augment an unlabeled list."""
    out: List[Structure] = []
    for s in structs:
        out.extend(augment_structure(s, dedupe=dedupe))
    return out
