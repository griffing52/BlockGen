"""Score every head of a model against the true graph, with the T28 baselines.

An *arm* is one condition (model x format x order x query slot). For each item in the
arm we keep per-head metrics (AUROC, distance-stratified AUROC, AP, hard-pair AUROC),
rollout metrics, the same metrics for every model-free baseline and, optionally,
per-node hidden states for the trained probe. The best head is *selected* on the
even-indexed items and *reported* on the odd ones (T28 did the same with halves).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .adjacency import baseline_scores, causal_pairs, parent_top1, score_matrix
from .probe import evaluate, probe_scores, train_probe
from .scenes import Scene

METRICS = ("auc", "strat", "ap", "hard")


@dataclass
class Arm:
    name: str
    heads: Dict[str, List[np.ndarray]] = field(default_factory=dict)      # metric -> [L, H] per item
    rollout: Dict[str, List[np.ndarray]] = field(default_factory=dict)    # metric -> [L] per item
    baselines: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    hidden: Dict[int, List[np.ndarray]] = field(default_factory=dict)
    scenes: List[Scene] = field(default_factory=list)
    parents: List[Optional[List[int]]] = field(default_factory=list)

    def add(self, scene: Scene, heads: np.ndarray, roll: np.ndarray,
            parents: Optional[List[int]] = None, hidden: Optional[dict] = None) -> None:
        adj, c = scene.adjacency, scene.coords
        _append(self.heads, score_matrix(heads, adj, c))
        _append(self.rollout, score_matrix(roll, adj, c, with_ap=False))
        for name, mat in baseline_scores(c, scene.names, parents).items():
            for k, v in score_matrix(mat[None], adj, c).items():
                self.baselines.setdefault(name, {}).setdefault(k, []).append(float(v[0]))
        if parents is not None:
            _append(self.heads, _relative_metrics(heads, adj, parents))
            for name, mat in baseline_scores(c, scene.names, parents).items():
                for k, v in _relative_metrics(mat[None], adj, parents).items():
                    self.baselines[name].setdefault(k, []).append(float(v[0]))
        for l, f in (hidden or {}).items():
            self.hidden.setdefault(l, []).append(f)
        self.scenes.append(scene)
        self.parents.append(parents)


def _append(store: Dict[str, list], metrics: Dict[str, np.ndarray]) -> None:
    for k, v in metrics.items():
        store.setdefault(k, []).append(v)


def _relative_metrics(S: np.ndarray, adj: np.ndarray, parents: List[int]) -> Dict[str, np.ndarray]:
    """For the coordinate-free format: does attention find the *stated* parent (a pointer
    the text spells out) and the *unstated* contacts (non-tree edges)?"""
    n = adj.shape[0]
    ii, jj = causal_pairs(n)
    par = np.zeros((n, n), dtype=bool)
    for i, p in enumerate(parents):
        if p >= 0:
            par[i, p] = par[p, i] = True
    is_par = par[ii, jj]
    out = {"parent_auc": score_matrix(S, par, with_ap=False)["auc"],
           "parent_top1": parent_top1(S, parents)}
    unstated = score_matrix(S, adj & ~par, with_ap=False, pair_mask=~is_par)
    out["unstated_auc"], out["unstated_strat"] = unstated["auc"], unstated["strat"]
    return out


# ---------------------------------------------------------------- summaries

def best_head(per_item: List[np.ndarray]) -> dict:
    """Select on even items, report on odd items; also the in-sample best."""
    a = np.stack(per_item)                                    # [S, L, H]
    sel = np.nanmean(a[0::2], 0)
    l, h = np.unravel_index(np.nanargmax(sel), sel.shape)
    full = np.nanmean(a, 0)
    lf, hf = np.unravel_index(np.nanargmax(full), full.shape)
    return {"head": [int(l), int(h)], "heldout": float(np.nanmean(a[1::2, l, h])),
            "in_sample_head": [int(lf), int(hf)], "in_sample": float(full[lf, hf]),
            "mean_all_heads": float(np.nanmean(a))}


def summarize(arm: Arm) -> dict:
    out = {"name": arm.name, "n_items": len(arm.scenes),
           "median_nodes": int(np.median([len(s) for s in arm.scenes])),
           "edge_rate": float(np.mean([s.adjacency[causal_pairs(len(s))].mean() for s in arm.scenes])),
           "heads": {k: best_head(v) for k, v in arm.heads.items()},
           "rollout": {k: float(np.nanmax(np.nanmean(np.stack(v), 0))) for k, v in arm.rollout.items()},
           # Baselines on the odd (report) items, like the selected head.
           "baselines": {n: {k: float(np.nanmean(v[1::2])) for k, v in d.items()}
                         for n, d in arm.baselines.items()}}
    return out


def run_probes(arm: Arm, train_frac: float = 0.6, kinds=("distance", "bilinear"),
               rank: int = 32, steps: int = 400) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Train on the first ``train_frac`` of items, test on the rest (different builds)."""
    n_tr = int(round(train_frac * len(arm.scenes)))
    adjs = [s.adjacency for s in arm.scenes]
    out: Dict[str, Dict[int, Dict[str, float]]] = {}
    for kind in kinds:
        out[kind] = {}
        for l, feats in sorted(arm.hidden.items()):
            probe, mu, sd = train_probe(feats[:n_tr], adjs[:n_tr], kind=kind, rank=rank, steps=steps)
            scores = probe_scores(probe, mu, sd, feats[n_tr:])
            out[kind][l] = evaluate(scores, adjs[n_tr:], [s.coords for s in arm.scenes[n_tr:]])
    return out


def probe_test_baselines(arm: Arm, train_frac: float = 0.6) -> Dict[str, Dict[str, float]]:
    """Baselines on the probe's test items, for a like-for-like comparison."""
    n_tr = int(round(train_frac * len(arm.scenes)))
    return {n: {k: float(np.nanmean(v[n_tr:])) for k, v in d.items()}
            for n, d in arm.baselines.items()}


def write_head_csv(arms: Sequence[Arm], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        keys = sorted({k for a in arms for k in a.heads})
        w.writerow(["arm", "layer", "head"] + [f"{k}_mean" for k in keys])
        for a in arms:
            means = {k: np.nanmean(np.stack(a.heads[k]), 0) for k in keys if k in a.heads}
            L, H = next(iter(means.values())).shape
            for l in range(L):
                for h in range(H):
                    w.writerow([a.name, l, h] + [f"{means[k][l, h]:.4f}" if k in means else ""
                                                 for k in keys])
            for l, v in enumerate(np.nanmean(np.stack(a.rollout["auc"]), 0)):
                w.writerow([a.name, l, "rollout"] + [f"{v:.4f}" if k == "auc" else ""
                                                     for k in keys])
