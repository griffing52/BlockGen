"""Trained edge probes on frozen per-node hidden states.

If no single head exposes adjacency, the information may still be *in* the residual
stream. Two low-rank probes, both symmetric in (i, j):

* ``distance`` (Hewitt and Manning, 2019): ``s = b - ||B (h_i - h_j)||^2``.
* ``bilinear``: ``s = (U h_i).(V h_j) + (V h_i).(U h_j) + b``.

Trained with class-weighted BCE over all pairs of the training scenes, scored by AUROC
on held-out scenes (different builds).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from .adjacency import score_matrix


class EdgeProbe(nn.Module):
    def __init__(self, dim: int, rank: int = 32, kind: str = "distance"):
        super().__init__()
        self.kind = kind
        self.B = nn.Linear(dim, rank, bias=False)
        self.V = nn.Linear(dim, rank, bias=False) if kind == "bilinear" else None
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """``h [N, D] -> [N, N]`` symmetric edge logits."""
        u = self.B(h)
        if self.kind == "distance":
            return self.b - torch.cdist(u, u).pow(2)
        v = self.V(h)
        return u @ v.T + v @ u.T + self.b


def _pairs(mat: torch.Tensor) -> torch.Tensor:
    il = torch.tril_indices(mat.shape[0], mat.shape[0], -1)
    return mat[il[0], il[1]]


def train_probe(feats: Sequence[np.ndarray], adjs: Sequence[np.ndarray], kind: str = "distance",
                rank: int = 32, steps: int = 400, lr: float = 3e-3, weight_decay: float = 1e-3,
                seed: int = 0):
    """Fit a probe; returns ``(probe, mean, std)`` (features are standardised)."""
    torch.manual_seed(seed)
    allf = np.concatenate(feats)
    mean, std = allf.mean(0), allf.std(0) + 1e-6
    xs = [torch.tensor((f - mean) / std, dtype=torch.float32) for f in feats]
    ys = [_pairs(torch.tensor(a, dtype=torch.float32)) for a in adjs]
    pos = sum(float(y.sum()) for y in ys)
    total = sum(len(y) for y in ys)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((total - pos) / max(pos, 1.0)))
    probe = EdgeProbe(allf.shape[1], rank, kind)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        opt.zero_grad()
        loss = sum(loss_fn(_pairs(probe(x)), y) for x, y in zip(xs, ys)) / len(xs)
        loss.backward()
        opt.step()
    return probe, mean, std


@torch.no_grad()
def probe_scores(probe, mean, std, feats: Sequence[np.ndarray]) -> List[np.ndarray]:
    return [probe(torch.tensor((f - mean) / std, dtype=torch.float32)).numpy() for f in feats]


def evaluate(scores: Sequence[np.ndarray], adjs: Sequence[np.ndarray],
             coords: Optional[Sequence[np.ndarray]] = None) -> Dict[str, float]:
    """Macro-averaged causal-pair metrics (``auc``, ``strat``, ``ap``, ``hard``) over scenes."""
    rows = [score_matrix(s[None], a, None if coords is None else coords[k])
            for k, (s, a) in enumerate(zip(scores, adjs))]
    return {m: float(np.nanmean([r[m][0] for r in rows])) for m in rows[0]}
