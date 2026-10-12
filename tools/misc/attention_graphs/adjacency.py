"""Ground-truth graphs, text baselines and ranking metrics.

Scores are evaluated the way T28 (``scripts/probe_llm_attention.py``) did, so the
numbers are comparable: every *causal candidate* pair ``(i, j)`` with ``j < i`` is
ranked by ``S[i, j]`` (the attention line ``i`` pays to an earlier line ``j``), and we
report AUROC, distance-stratified AUROC (AUROC within bins of ``i - j``, pooled, which
removes the recency shortcut) and average precision. The Mann-Whitney and stratified
AUROC are imported from T28's script rather than re-derived.

One extra view is new here: **hard pairs**, the candidate pairs that share exactly two
of x/y/z. A head that matches identical coordinate tokens cannot separate them; only
knowing that the third coordinate differs by one can.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Sequence

import numpy as np

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, os.path.abspath(_SCRIPTS))
from probe_llm_attention import auc_rows, auc_rows_stratified  # noqa: E402  (T28)


# ---------------------------------------------------------------- ground truth

def face_adjacency(coords: np.ndarray) -> np.ndarray:
    """6-neighbour adjacency: blocks whose integer coordinates differ by 1 on one axis."""
    c = np.asarray(coords, dtype=int)
    return np.abs(c[:, None, :] - c[None, :, :]).sum(-1) == 1


def stud_adjacency(boxes: Sequence[tuple]) -> np.ndarray:
    """LEGO stud connectivity for bricks ``(x0, layer, z0, w, d)``.

    Two bricks connect when they sit in consecutive layers and their footprints overlap.
    """
    n = len(boxes)
    adj = np.zeros((n, n), dtype=bool)
    for i, (xi, yi, zi, wi, di) in enumerate(boxes):
        for j, (xj, yj, zj, wj, dj) in enumerate(boxes):
            if i != j and abs(yi - yj) == 1:
                overlap_x = min(xi + wi, xj + wj) - max(xi, xj)
                overlap_z = min(zi + di, zj + dj) - max(zi, zj)
                adj[i, j] = overlap_x > 0 and overlap_z > 0
    return adj


def tree_hops(parents: Sequence[int]) -> np.ndarray:
    """Path length between every pair in the tree given by ``parents`` (root = -1)."""
    n = len(parents)
    depth = np.zeros(n, dtype=int)
    for i in range(n):                       # parents always precede children
        depth[i] = 0 if parents[i] < 0 else depth[parents[i]] + 1
    anc = [[i] for i in range(n)]
    for i in range(n):
        if parents[i] >= 0:
            anc[i] = [i] + anc[parents[i]]
    hops = np.zeros((n, n), dtype=int)
    for i in range(n):
        si = set(anc[i])
        for j in range(i):
            lca = next(a for a in anc[j] if a in si)
            hops[i, j] = hops[j, i] = depth[i] + depth[j] - 2 * depth[lca]
    return hops


# ---------------------------------------------------------------- baselines

def baseline_scores(coords: np.ndarray, names: Sequence[str],
                    parents: Optional[Sequence[int]] = None) -> Dict[str, np.ndarray]:
    """Model-free ``[N, N]`` scores in the serialised order (higher = more likely an edge).

    * ``recency``: ``-|i - j|``, pure position in the text.
    * ``coord_match``: how many of x/y/z are the identical number (T28's copy signature).
    * ``coord_L1``: ``-L1`` between the coordinates (T28's 0.980 subtraction). It is a
      surface baseline only when coordinates are printed; otherwise an oracle.
    * ``same_name``: same block type.
    * with ``parents`` (relative format): ``tree_hops`` (``-`` path length in the stated
      parent tree, readable from the text) and ``parent`` (``j`` is ``i``'s stated parent).
    """
    n = len(names)
    idx = np.arange(n)
    c = np.asarray(coords, dtype=float)
    out = {
        "recency": -np.abs(idx[:, None] - idx[None, :]).astype(float),
        "coord_match": (c[:, None, :] == c[None, :, :]).sum(-1).astype(float),
        "coord_L1": -np.abs(c[:, None, :] - c[None, :, :]).sum(-1),
        "same_name": np.array([[a == b for b in names] for a in names], dtype=float),
    }
    if parents is not None:
        out["tree_hops"] = -tree_hops(parents).astype(float)
        p = np.zeros((n, n))
        for i, k in enumerate(parents):
            if k >= 0:
                p[i, k] = p[k, i] = 1.0
        out["parent"] = p
    return out


# ---------------------------------------------------------------- metrics

def causal_pairs(n: int):
    """``(ii, jj)`` for all candidates ``j < i``."""
    return np.tril_indices(n, k=-1)


def hard_mask(coords: np.ndarray) -> np.ndarray:
    """Causal candidates that share exactly two coordinates."""
    c = np.asarray(coords)
    ii, jj = causal_pairs(len(c))
    return (c[ii] == c[jj]).sum(-1) == 2


def average_precision(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """AP along the last axis of ``scores [K, P]`` (ties broken pessimistically)."""
    labels = np.asarray(labels, dtype=bool)
    pos = labels.sum()
    if pos == 0:
        return np.full(scores.shape[:-1], np.nan)
    flat = scores.reshape(-1, scores.shape[-1])
    out = np.empty(len(flat))
    for k, row in enumerate(flat):
        hits = labels[np.lexsort((labels, -row))]      # score desc, negatives first on ties
        precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
        out[k] = precision[hits].sum() / pos
    return out.reshape(scores.shape[:-1])


def auroc(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """AUROC along the last axis, any leading shape (T28's ``auc_rows``)."""
    lead = scores.shape[:-1]
    return auc_rows(scores.reshape(-1, scores.shape[-1]).astype(np.float64),
                    np.asarray(labels, dtype=bool)).reshape(lead)


def score_matrix(S: np.ndarray, adj: np.ndarray, coords: Optional[np.ndarray] = None,
                 with_ap: bool = True, pair_mask: Optional[np.ndarray] = None
                 ) -> Dict[str, np.ndarray]:
    """Metrics for score matrices ``S [..., N, N]`` over causal candidates ``j < i``.

    Returns arrays of shape ``S.shape[:-2]``: ``auc``, ``strat`` (distance-stratified),
    ``ap`` and, given ``coords``, ``hard`` (AUROC on pairs sharing two coordinates) and
    ``hard_strat`` (the same pairs, stratified by sequence distance). ``pair_mask`` (bool over the causal pairs) restricts every metric to a subset.
    """
    n = S.shape[-1]
    ii, jj = causal_pairs(n)
    keep = np.ones(len(ii), dtype=bool) if pair_mask is None else pair_mask
    hard = hard_mask(coords)[keep] if coords is not None else None
    ii, jj = ii[keep], jj[keep]
    lab = adj[ii, jj].astype(bool)
    lead = S.shape[:-2]
    flat = S[..., ii, jj].reshape(-1, len(ii)).astype(np.float64)
    out = {"auc": auc_rows(flat, lab).reshape(lead),
           "strat": auc_rows_stratified(flat, lab, (ii - jj).astype(np.int64)).reshape(lead)}
    if with_ap:
        out["ap"] = average_precision(flat, lab).reshape(lead)
    if hard is not None:
        nan = np.full(lead, np.nan)
        out["hard"] = auc_rows(flat[:, hard], lab[hard]).reshape(lead) if hard.any() else nan
        # both shortcuts removed: hard pairs, AUROC within sequence-distance bins
        out["hard_strat"] = (auc_rows_stratified(flat[:, hard], lab[hard],
                                                 (ii - jj)[hard].astype(np.int64)).reshape(lead)
                             if hard.any() else nan)
    return out


def parent_top1(S: np.ndarray, parents: Sequence[int]) -> np.ndarray:
    """Fraction of non-root lines whose highest-scoring earlier line is the stated parent."""
    rows = [i for i, p in enumerate(parents) if p >= 0 and i > 1]   # i=1 has one candidate
    if not rows:
        return np.full(S.shape[:-2], np.nan)
    hits = [np.argmax(S[..., i, :i], axis=-1) == parents[i] for i in rows]
    return np.mean(hits, axis=0)
