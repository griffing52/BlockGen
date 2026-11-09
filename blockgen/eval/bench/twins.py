"""Within-set near-twins: the duplication the train-split gates cannot see.

The two existing novelty gates both miss a set that repeats *itself*.
`voxel_dup_rate` compares each build to the TRAIN split, so a generator that
reproduces pieces of some other corpus (vanilla village templates, in the
Dream-Cubed audit of 2026-11-04) reads as novel. `voxel_diversity` is a
within-set MEAN of pairwise distance, so 160 tight clusters of near-copies still
average out to a diverse-looking set (0.866 against real's 0.898). A generator
whose output is 59% near-copies of other outputs passed both.

The statistic here asks the direct question per build: does it have a twin
elsewhere in the same set? Similarity is occupancy IoU at the best of the 8
yaw/mirror variants (y is up) and *every* integer translation, computed exactly
by FFT cross-correlation, so a copy that was rotated, mirrored or shifted inside
its crop is still a copy. `twin_rate` is the fraction of builds whose nearest
other build reaches `threshold`.

Held-out real houses have almost no twins (0.8% at IoU >= 0.8, n=256), so the
gate in `composite.GATES` reads this the way it reads the train-split duplicate
rate: z spreads above the real control scored in the same run.

Pairs are pruned by the size bound IoU <= min(|a|,|b|) / max(|a|,|b|): a pair
that cannot reach `threshold` is never correlated, so the nearest-neighbour IoU
returned is exact whenever it is >= `threshold` and is 0 for builds with no
size-compatible partner. CPU only (scipy FFT); about a minute for n=256.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy import fft as sfft

#: IoU at which two builds count as twins. The same bar the Dream-Cubed audit
#: reported, chosen as "the same building up to a few blocks", not tuned to it.
TWIN_THRESHOLD = 0.8


def _occ(s) -> np.ndarray:
    """Cropped occupancy of a Structure or a raw id array."""
    ids = np.asarray(getattr(s, "block_ids", s))
    occ = ids != 0
    if not occ.any():
        return np.zeros((0, 0, 0), bool)
    xs, ys, zs = np.where(occ)
    return occ[xs.min():xs.max() + 1, ys.min():ys.max() + 1, zs.min():zs.max() + 1]


def d4(occ: np.ndarray) -> List[np.ndarray]:
    """The 8 yaw/mirror variants about the vertical (axis 1)."""
    out = []
    for m in (occ, occ[::-1]):
        for k in range(4):
            out.append(np.ascontiguousarray(np.rot90(m, k, axes=(0, 2))))
    return out


@dataclass
class TwinReport:
    nn_iou: np.ndarray        # per build; exact when >= threshold, else a lower bound
    twin: np.ndarray          # per build, bool
    threshold: float
    nn_index: np.ndarray = None  # per build, its nearest other build (-1: none checked)

    @property
    def rate(self) -> float:
        return float(self.twin.mean()) if self.twin.size else float("nan")


def within_set_twins(structures: Sequence, threshold: float = TWIN_THRESHOLD,
                     workers: int = -1) -> TwinReport:
    occs = [_occ(s) for s in structures]
    n = len(occs)
    sizes = np.array([int(o.sum()) for o in occs])
    nn = np.zeros(n)
    idx = np.full(n, -1)
    if n < 2:
        return TwinReport(nn, nn >= threshold, threshold, idx)
    dxz = max(max(o.shape[0], o.shape[2]) for o in occs)
    dy = max(o.shape[1] for o in occs)
    # Two crops of extent <= d never wrap in a grid of 2d - 1.
    gxz, gy = sfft.next_fast_len(2 * dxz), sfft.next_fast_len(2 * dy)
    shape = (gxz, gy, gxz)

    def spec(a: np.ndarray) -> np.ndarray:
        g = np.zeros(shape, np.float32)
        g[:a.shape[0], :a.shape[1], :a.shape[2]] = a
        return sfft.rfftn(g, workers=workers)

    # Exact copies up to D4 and translation share a canonical key: IoU 1 without
    # an FFT. A pair whose members both already read 1.0 can change nothing, so
    # a set of repeats costs O(n) rather than O(n^2) correlations.
    keys = [min(v.tobytes() + bytes(str(v.shape), "ascii") for v in d4(o)) if s else None
            for o, s in zip(occs, sizes)]
    groups: dict = {}
    for i, k in enumerate(keys):
        if k is not None:
            groups.setdefault(k, []).append(i)
    for members in groups.values():
        if len(members) > 1:
            nn[members] = 1.0
            for k, m in enumerate(members):
                idx[m] = members[(k + 1) % len(members)]

    plain = [spec(o) if s else None for o, s in zip(occs, sizes)]
    for i in range(n):
        if sizes[i] == 0:
            continue
        lo, hi = np.minimum(sizes, sizes[i]), np.maximum(sizes, sizes[i])
        bound = np.where(hi > 0, lo / np.maximum(hi, 1), 0.0)
        cand = [j for j in range(i + 1, n) if sizes[j] and bound[j] >= threshold
                and not (nn[i] >= 1.0 and nn[j] >= 1.0) and keys[j] != keys[i]]
        if not cand:
            continue
        variants = np.stack([spec(v) for v in d4(occs[i])])          # (8, ...)
        for j in cand:
            corr = sfft.irfftn(variants * np.conj(plain[j])[None], s=shape,
                               axes=(1, 2, 3), workers=workers)
            inter = float(np.round(corr.max()))
            iou = inter / (sizes[i] + sizes[j] - inter)
            if iou > nn[i]:
                nn[i], idx[i] = iou, j
            if iou > nn[j]:
                nn[j], idx[j] = iou, i
    return TwinReport(nn, nn >= threshold, threshold, idx)
