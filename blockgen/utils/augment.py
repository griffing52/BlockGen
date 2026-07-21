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


# --- orientation-aware block_data remapping under D4 -------------------------
# When the geometry rotates, orientation-bearing blocks must have their facing/axis
# data remapped or the augmented copy shows wrong-facing stairs/logs (the §17 caveat).
# np.rot90(k=1, axes=(0,2)) rotates +x(axis0) -> +z(axis2), i.e. R(dx,dz) = (-dz, dx);
# the horizontal mirror flips x (dx -> -dx). We remap facing VALUES to match.
#
# Legacy (pre-flattening) conventions:
#   stairs  data & 3 = facing 0:east(+x) 1:west(-x) 2:south(+z) 3:north(-z); bit4 = upside-down.
#   logs    (data >> 2) & 3 = axis 0:y(up-down) 1:x 2:z 3:bark; data & 3 = species.
_STAIR_ROT1 = {0: 2, 1: 3, 2: 1, 3: 0}   # one 90° rot: E->S, W->N, S->W, N->E
_STAIR_MIRROR = {0: 1, 1: 0, 2: 2, 3: 3}  # flip x: E<->W, S/N unchanged
_LOG_ROT1 = {0: 0, 1: 2, 2: 1, 3: 3}      # x<->z under 90°; y/bark unchanged; mirror is identity
_STAIR_IDS = frozenset({53, 67, 108, 109, 114, 128, 134, 135, 136, 156, 163, 164, 180, 203})
_LOG_IDS = frozenset({17, 162})


def _remap_stair(d: int, mirror: bool, k: int) -> int:
    up, f = d & 4, d & 3
    if mirror:
        f = _STAIR_MIRROR[f]
    for _ in range(k % 4):
        f = _STAIR_ROT1[f]
    return up | f


def _remap_log(d: int, mirror: bool, k: int) -> int:
    species, axis = d & 3, (d >> 2) & 3
    for _ in range(k % 4):      # mirror leaves axis unchanged
        axis = _LOG_ROT1[axis]
    return species | (axis << 2)


# Precompute 16-entry lookup tables per (mirror, k) so the remap is a vectorized numpy
# gather, not a Python per-voxel loop (which made D4-augmenting the whole corpus hang).
_STAIR_IDS_ARR = np.array(sorted(_STAIR_IDS))
_LOG_IDS_ARR = np.array(sorted(_LOG_IDS))
_STAIR_LUT = {(m, k): np.array([_remap_stair(d, m, k) for d in range(16)], dtype=np.int32)
              for m in (False, True) for k in range(4)}
_LOG_LUT = {(m, k): np.array([_remap_log(d, m, k) for d in range(16)], dtype=np.int32)
            for m in (False, True) for k in range(4)}


def _rotate_orientation_data(bi: np.ndarray, bd: np.ndarray, mirror: bool, k: int) -> np.ndarray:
    """Return bd with stair/log facing values remapped for a (mirror, k·90°) transform.

    Vectorized (LUT gather over the ≤15 data values). Doors/trapdoors also encode facing
    but with tangled open/hinge bits; they are left unrotated (rare, and forced-merged
    into single pieces) — a small, documented gap.
    """
    k %= 4
    if not mirror and k == 0:
        return bd
    out = bd.copy()
    for ids, lut in ((_STAIR_IDS_ARR, _STAIR_LUT[(mirror, k)]),
                     (_LOG_IDS_ARR, _LOG_LUT[(mirror, k)])):
        mask = np.isin(bi, ids)
        if mask.any():
            vals = bd[mask]
            in_range = vals < lut.shape[0]        # orientation data is 0..15; guard anyway
            new = vals.copy()
            new[in_range] = lut[vals[in_range]]
            out[mask] = new
    return out


# XYZ arrays with the vertical axis = Y (axis 1). Rotations about vertical act on
# the X-Z plane (axes 0 and 2); a horizontal mirror flips X (axis 0).
def _d4_arrays(block_ids: np.ndarray, block_data: np.ndarray):
    """Yield the (block_ids, block_data) arrays for each of the 8 D4 elements.

    Orientation ``block_data`` (stairs/logs) is remapped so facings stay correct under
    rotation/mirror — required when the vocab is orientation-aware (notes §17/§19).
    """
    for mirror in (False, True):
        bi0 = block_ids[::-1] if mirror else block_ids
        bd0 = block_data[::-1] if mirror else block_data
        for k in range(4):
            bi = np.rot90(bi0, k, axes=(0, 2)).copy()
            bd = np.rot90(bd0, k, axes=(0, 2)).copy()
            yield bi, _rotate_orientation_data(bi, bd, mirror, k)


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
