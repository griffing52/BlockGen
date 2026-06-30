"""Novelty / nearest-neighbor evaluation shared by all generative tracks.

The question we want to answer: *are generated structures genuinely new, or is the
model memorizing the training set?* Every structure (generated or training) is
voxelized into a common translation-tolerant canonical grid (crop + center-pad).
For each generated sample we find its nearest training neighbors by occupancy IoU,
then report a small panel of metrics and render side-by-side comparison grids.

Interpretation:
- **NN-IoU near 1.0** for many samples  -> memorization / duplicates.
- **NN-IoU well below 1.0** with non-trivial **diversity** -> genuinely new shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from blockgen.utils.data import Structure
from blockgen.utils.serialize import BlockVocab, structure_to_grid


# --- voxelization ----------------------------------------------------------
def voxelize_occupancy(
    structures: Sequence[Structure], grid: int, vocab: BlockVocab
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (occupancy bool [N,V], class int [N,V]) flattened per structure."""
    occ = np.zeros((len(structures), grid ** 3), dtype=bool)
    cls = np.zeros((len(structures), grid ** 3), dtype=np.int32)
    for i, s in enumerate(structures):
        g = structure_to_grid(s, grid, vocab).reshape(-1)
        occ[i] = g > 0
        cls[i] = g
    return occ, cls


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise occupancy IoU between rows of a [Na,V] and b [Nb,V] (bool)."""
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    inter = af @ bf.T                       # [Na, Nb]
    asum = af.sum(1, keepdims=True)         # [Na, 1]
    bsum = bf.sum(1, keepdims=True).T       # [1, Nb]
    union = asum + bsum - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


# --- nearest neighbors -----------------------------------------------------
@dataclass
class NoveltyReport:
    nn_iou: np.ndarray            # [N] best occupancy IoU to the training set
    nn_index: np.ndarray         # [N, k] training indices of top-k neighbors
    nn_iou_topk: np.ndarray      # [N, k] IoUs of those neighbors
    duplicate_rate: float        # fraction with NN-IoU >= dup_threshold
    mean_nn_iou: float
    diversity: float             # 1 - mean pairwise IoU among generated samples
    validity_rate: float         # fraction non-empty & single connected component
    block_agreement: np.ndarray  # [N] block-class agreement with the NN-1 neighbor


def _connected_single_component(occ_row: np.ndarray, grid: int) -> bool:
    """True if occupied voxels form exactly one 6-connected component (non-empty)."""
    vol = occ_row.reshape(grid, grid, grid)
    occupied = np.argwhere(vol)
    if occupied.shape[0] == 0:
        return False
    occ_set = set(map(tuple, occupied.tolist()))
    start = tuple(occupied[0].tolist())
    seen = {start}
    stack = [start]
    nbrs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    while stack:
        x, y, z = stack.pop()
        for dx, dy, dz in nbrs:
            p = (x + dx, y + dy, z + dz)
            if p in occ_set and p not in seen:
                seen.add(p)
                stack.append(p)
    return len(seen) == len(occ_set)


def evaluate_novelty(
    generated: Sequence[Structure],
    training: Sequence[Structure],
    vocab: BlockVocab,
    grid: int = 24,
    k: int = 3,
    dup_threshold: float = 0.95,
) -> NoveltyReport:
    gen_occ, gen_cls = voxelize_occupancy(generated, grid, vocab)
    train_occ, train_cls = voxelize_occupancy(training, grid, vocab)

    iou = _iou_matrix(gen_occ, train_occ)  # [Ngen, Ntrain]
    k = min(k, train_occ.shape[0])
    nn_index = np.argsort(-iou, axis=1)[:, :k]
    nn_iou_topk = np.take_along_axis(iou, nn_index, axis=1)
    nn_iou = nn_iou_topk[:, 0] if k > 0 else np.zeros(len(generated))

    # Block-class agreement with the closest neighbor (over union of occupied).
    block_agreement = np.zeros(len(generated))
    for i in range(len(generated)):
        j = nn_index[i, 0]
        a, b = gen_cls[i], train_cls[j]
        union = (a > 0) | (b > 0)
        if union.sum() > 0:
            block_agreement[i] = float(((a == b) & union).sum()) / float(union.sum())

    # Diversity among generated samples: 1 - mean off-diagonal pairwise IoU.
    if len(generated) > 1:
        gg = _iou_matrix(gen_occ, gen_occ)
        n = len(generated)
        off = (gg.sum() - np.trace(gg)) / (n * (n - 1))
        diversity = float(1.0 - off)
    else:
        diversity = float("nan")

    validity = np.mean([_connected_single_component(gen_occ[i], grid) for i in range(len(generated))])

    return NoveltyReport(
        nn_iou=nn_iou,
        nn_index=nn_index,
        nn_iou_topk=nn_iou_topk,
        duplicate_rate=float(np.mean(nn_iou >= dup_threshold)),
        mean_nn_iou=float(np.mean(nn_iou)),
        diversity=diversity,
        validity_rate=float(validity),
        block_agreement=block_agreement,
    )


def summary_row(name: str, report: NoveltyReport) -> dict:
    """One row for a cross-track comparison table."""
    return {
        "track": name,
        "mean_nn_iou": round(report.mean_nn_iou, 3),
        "duplicate_rate": round(report.duplicate_rate, 3),
        "diversity": round(report.diversity, 3),
        "validity_rate": round(report.validity_rate, 3),
        "mean_block_agreement": round(float(np.mean(report.block_agreement)), 3),
    }


# --- visualization ---------------------------------------------------------
def render_comparison_grid(
    generated: Sequence[Structure],
    training: Sequence[Structure],
    report: NoveltyReport,
    num_rows: int = 4,
    k: int = 3,
    title: Optional[str] = None,
    max_dim: int = 24,
):
    """Render rows of [generated sample | top-k nearest training neighbors].

    Returns the matplotlib Figure (does not call plt.show()).
    """
    import matplotlib.pyplot as plt
    from blockgen.renderer.render import render_schem

    num_rows = min(num_rows, len(generated))
    k = min(k, report.nn_index.shape[1])
    ncols = 1 + k
    fig = plt.figure(figsize=(3.2 * ncols, 3.2 * num_rows))

    for r in range(num_rows):
        ax = fig.add_subplot(num_rows, ncols, r * ncols + 1, projection="3d")
        render_schem(generated[r], ax=ax, max_dim=max_dim, show=False)
        ax.set_title(f"GEN #{r}\nNN-IoU={report.nn_iou[r]:.2f}", fontsize=9)
        for c in range(k):
            j = int(report.nn_index[r, c])
            ax2 = fig.add_subplot(num_rows, ncols, r * ncols + 2 + c, projection="3d")
            render_schem(training[j], ax=ax2, max_dim=max_dim, show=False)
            ax2.set_title(f"NN{c+1} IoU={report.nn_iou_topk[r, c]:.2f}", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig
