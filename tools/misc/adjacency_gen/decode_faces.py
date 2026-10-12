"""Decoders for the face model.

(A) Sequential open-face builder (:func:`sequential_decode`). The seed head places one
block at an absolute window cell. Then, one model call per step, every attachment
``(open face d of placed block p, unplaced block j)`` is scored by the glue matrices and
one is taken (greedy, or sampled at a temperature); j lands at ``pos[p] + e_d``.

* ``check=True``: an attachment is legal only if the target cell is empty and inside the
  window. Every face touching the new cell is glued automatically, so the realized
  build is always consistent. Decoding stops early if no legal attachment is left
  (those blocks are unplaced).
* ``check=False``: no occupancy or bounds test. A face is open until it is used by an
  explicit attachment (no automatic gluing), so the result is the model's spanning tree;
  a block attached into an occupied cell is a collision and is dropped from the build.

(B) One-shot matching (:func:`oneshot_decode`). One call with nothing placed gives the
three axis matrices and the exposed logits. Per axis, a log-domain Sinkhorn with a dust
bin (per-face bin score = the exposed logit) normalizes the scores, a Hungarian
assignment picks at most one partner per face, and the expected number of glued faces
(the plan's non-bin mass) of its most confident pairs are kept (:func:`match_axis`). The matches are
realized by BFS: with the check, Kruskal order by matching confidence and a merge that
would collide is rejected (:class:`decode.Placement`); without it, :func:`decode.realize`.
One-shot output has no absolute frame, so the largest component is translated to the
shift that best matches the coarse condition (:func:`align_to_condition`).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .coarse import Coarse, coarsen
from .data import OffsetSpace
from .decode import Placement, realize
from .faces import DIRS, OPP, action_scores
from .train_faces import collate_states


# --------------------------------------------------------------------------- #
# (A) sequential open-face builder
# --------------------------------------------------------------------------- #
def _choose(logits: np.ndarray, legal: np.ndarray, temperature: float, rng) -> Optional[int]:
    flat = np.where(legal, logits, -np.inf).ravel()
    if not np.isfinite(flat).any():
        return None
    if temperature <= 0:
        return int(np.argmax(flat))
    z = flat / temperature
    z = z - z[np.isfinite(z)].max()
    p = np.exp(z)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


@torch.no_grad()
def sequential_decode(model, items: Sequence[dict], device, check: bool = True,
                      temperature: float = 0.0, seed: int = 0, bs: int = 32, policy: str = "model",
                      cmask: bool = False, size: int = 8) -> List[dict]:
    """Returns per item ``pos`` [N,3] (window frame), ``placed`` [N], ``conflict`` [N].

    ``policy="random"`` replaces the model by a uniform choice over the legal moves (the
    model is not called), which isolates what the check and the mask do on their own.
    ``cmask=True`` also uses the coarse condition as a hard mask: the seed must land in an
    occupied coarse cell and an attachment must target a coarse cell that still has fewer
    placed blocks than its condition count (if that leaves nothing legal, the unmasked
    legal set is used)."""
    cfg = model.cfg if model is not None else None
    S = cfg.size if cfg is not None else size
    out = []
    for b0 in range(0, len(items), bs):
        chunk = list(items[b0:b0 + bs])
        rngs = [np.random.default_rng(seed + b0 + k) for k in range(len(chunk))]
        st = []
        for it in chunk:
            n = len(it["types"])
            st.append({"placed": np.zeros(n, bool), "pos": np.zeros((n, 3), np.int64),
                       "conflict": np.zeros(n, bool), "occ": {}, "used": np.zeros((n, 6), bool),
                       "done": False, "steps": 0})
        while True:
            act = [k for k, s in enumerate(st) if not s["done"]]
            if not act:
                break
            sub = [chunk[k] for k in act]
            fst = []
            for k in act:
                s = st[k]
                from .faces import buried_state
                f = buried_state(s["pos"], s["placed"] & ~s["conflict"])
                if not check:            # open = not used by an explicit attachment
                    f = np.where(s["placed"][:, None], np.where(s["used"], 2, 1), 0)
                fst.append(f)
            if policy == "model":
                batch = collate_states(sub, [st[k]["placed"] for k in act], [st[k]["pos"] for k in act],
                                       device, cfg, fstate_list=fst)
                o = model(**batch)
                X = o["X"].float().cpu().numpy()
                seedl = o["seed"].float().cpu().numpy()
            else:
                Nm = max(len(it["types"]) for it in sub)
                X = np.zeros((len(sub), 3, Nm, Nm))
                seedl = np.zeros((len(sub), Nm, S ** 3))
            temp = temperature if policy == "model" else 1.0
            for r, k in enumerate(act):
                s, it = st[k], chunk[k]
                n = len(it["types"])
                if cmask:
                    cond = it["coarse"]
                    fine = np.stack(np.meshgrid(*[np.arange(S)] * 3, indexing="ij"), -1).reshape(-1, 3)
                    cc = fine // cond.f
                    resid = cond.count.copy()
                    for q in {tuple(int(v) for v in s["pos"][i]) for i in np.nonzero(s["placed"])[0]}:
                        if all(0 <= v < S for v in q):
                            resid[tuple(v // cond.f for v in q)] -= 1
                if not s["placed"].any():
                    sl = np.ones((n, S ** 3), bool)
                    if cmask:
                        sl &= (cond.count[cc[:, 0], cc[:, 1], cc[:, 2]] > 0)[None, :]
                    c = _choose(seedl[r, :n], sl, temp, rngs[k])
                    j, cell = divmod(c, S ** 3)
                    s["pos"][j] = (cell // (S * S), (cell // S) % S, cell % S)
                    s["placed"][j] = True
                    s["occ"][tuple(int(v) for v in s["pos"][j])] = j
                    s["steps"] += 1
                    s["done"] = n == 1
                    continue
                A = action_scores(X[r, :, :n, :n])                         # [6,N,N]
                placed = s["placed"]
                legal = placed[None, :, None] & ~placed[None, None, :] & np.ones((6, 1, 1), bool)
                if check:
                    tgt = s["pos"][None, :, :] + DIRS[:, None, :]          # [6,N,3]
                    ok = np.all((tgt >= 0) & (tgt < S), -1)
                    for d, p in zip(*np.nonzero(ok & placed[None, :])):
                        if tuple(int(v) for v in tgt[d, p]) in s["occ"]:
                            ok[d, p] = False
                    legal &= ok[:, :, None]
                else:
                    legal &= ~s["used"].T[:, :, None]
                if cmask:
                    tgt = s["pos"][None, :, :] + DIRS[:, None, :]
                    ins = np.all((tgt >= 0) & (tgt < S), -1)
                    tc = np.clip(tgt, 0, S - 1) // cond.f
                    short = ins & (resid[tc[..., 0], tc[..., 1], tc[..., 2]] > 0)
                    lm = legal & short[:, :, None]
                    if lm.any():
                        legal = lm
                a = _choose(A, legal, temp, rngs[k])
                if a is None:
                    s["done"] = True
                    continue
                d, rem = divmod(a, n * n)
                p, j = divmod(rem, n)
                s["pos"][j] = s["pos"][p] + DIRS[d]
                s["placed"][j] = True
                s["used"][p, d] = s["used"][j, OPP[d]] = True
                cell = tuple(int(v) for v in s["pos"][j])
                if cell in s["occ"]:
                    s["conflict"][j] = True
                else:
                    s["occ"][cell] = j
                s["steps"] += 1
                if s["placed"].all():
                    s["done"] = True
        for s in st:
            out.append({"pos": s["pos"], "placed": s["placed"], "conflict": s["conflict"]})
    return out


# --------------------------------------------------------------------------- #
# (B) one-shot matching
# --------------------------------------------------------------------------- #
def sinkhorn_dustbin(S: np.ndarray, bin_row: np.ndarray, bin_col: np.ndarray, iters: int = 30) -> np.ndarray:
    """Log-domain Sinkhorn on the (N+1)x(N+1) augmented score matrix (SuperGlue-style):
    each real row/column has mass 1, the bins have mass N. ``S`` has -inf where forbidden.
    Returns the log transport plan (augmented)."""
    N = len(S)
    Z = np.full((N + 1, N + 1), 0.0)
    Z[:N, :N] = S
    Z[:N, N] = bin_row
    Z[N, :N] = bin_col
    log_mu = np.concatenate([np.zeros(N), [np.log(N)]])
    log_nu = log_mu.copy()
    u = np.zeros(N + 1)
    v = np.zeros(N + 1)
    from scipy.special import logsumexp
    for _ in range(iters):
        u = log_mu - logsumexp(Z + v[None, :], axis=1)
        v = log_nu - logsumexp(Z + u[:, None], axis=0)
    return Z + u[:, None] + v[None, :]


def match_axis(S: np.ndarray, bin_row: np.ndarray, bin_col: np.ndarray, sinkhorn: bool = True):
    """At most one partner per face. Hungarian on the real block of the (Sinkhorn) log
    plan gives a one-to-one assignment; the ``K`` most confident pairs are kept, with
    ``K`` = the plan's expected number of glued +a faces (non-bin mass, rounded).

    Keeping the MAP of the augmented problem instead (bins included) leaves almost every
    face in its bin: under a multimodal posterior P(exposed) beats P(glued to any one
    specific face), so the MAP matching is nearly empty (seen in the first smoke run).
    Returns a list of (i, j, confidence)."""
    from scipy.optimize import linear_sum_assignment
    from scipy.special import expit
    N = len(S)
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    if sinkhorn:
        P = sinkhorn_dustbin(S, bin_row, bin_col)
        real = P[:N, :N]
        K = int(round(float(np.exp(real).sum())))
    else:                                    # raw scores; K from the exposed head
        real = S
        K = int(round(float((1 - expit(bin_row)).sum() + (1 - expit(bin_col)).sum()) / 2))
    BIG = 1e6
    C = np.where(np.isfinite(real), -real, BIG)
    r, c = linear_sum_assignment(C)
    pairs = sorted(((float(real[i, j]), int(i), int(j)) for i, j in zip(r, c) if C[i, j] < BIG),
                   reverse=True)[:K]
    return [(i, j, float(np.exp(v)) if sinkhorn else float(v)) for v, i, j in pairs]


@torch.no_grad()
def oneshot_scores(model, items: Sequence[dict], device, bs: int = 32) -> List[dict]:
    cfg = model.cfg
    res = []
    for b0 in range(0, len(items), bs):
        chunk = list(items[b0:b0 + bs])
        pl = [np.zeros(len(it["types"]), bool) for it in chunk]
        pos = [np.zeros((len(it["types"]), 3), np.int64) for it in chunk]
        if cfg.merge:
            from .merge import collate_merge
            o = model(**collate_merge(chunk, [np.arange(len(it["types"])) for it in chunk], pos, device, cfg))
        else:
            o = model(**collate_states(chunk, pl, pos, device, cfg))
        X = o["X"].float().cpu().numpy()
        ex = o["ex"].float().cpu().numpy()
        for r, it in enumerate(chunk):
            n = len(it["types"])
            res.append({"X": X[r, :, :n, :n], "ex": ex[r, :n]})
    return res


def oneshot_realize(X: np.ndarray, ex: np.ndarray, check: bool = True, sinkhorn: bool = True) -> dict:
    """Matches per axis, then BFS realization. Returns pos (relative frame), keep (largest
    component, collision-free), conflict, matches [(a, i, j, conf)], rejected merges."""
    N = X.shape[1]
    matches = []
    for a in range(3):
        for i, j, c in match_axis(X[a], ex[:, 2 * a], ex[:, 2 * a + 1], sinkhorn=sinkhorn):
            matches.append((a, i, j, c))
    shapes = [np.zeros((1, 3), np.int64)] * N
    sp = OffsetSpace(1)
    rejected = 0
    if check:
        pl = Placement(shapes)
        for a, i, j, c in sorted(matches, key=lambda m: -m[3]):
            if pl.comp[i] == pl.comp[j]:
                continue
            if not pl.try_merge(i, j, DIRS[2 * a]):
                rejected += 1
        comp = np.array(pl.comp)
        pos = pl.pos.copy()
        conflict = np.zeros(N, bool)
    else:
        labels = np.zeros((N, N), np.int64)
        conf = np.zeros((N, N))
        for a, i, j, c in matches:
            labels[i, j] = sp.cls(DIRS[2 * a])
            labels[j, i] = sp.cls(DIRS[2 * a + 1])
            conf[i, j] = conf[j, i] = c
        r = realize(labels, conf, shapes, sp)
        comp, pos, conflict = r["comp"], r["pos"], r["conflict"]
    vals, cnts = np.unique(comp, return_counts=True)
    lc = vals[np.argmax(cnts)]
    keep = (comp == lc) & ~conflict
    return {"pos": pos, "keep": keep, "conflict": conflict, "matches": matches, "rejected": rejected,
            "largest": int(cnts.max())}


# --------------------------------------------------------------------------- #
# frame alignment for frame-free outputs
# --------------------------------------------------------------------------- #
def align_to_condition(pos: np.ndarray, keep: np.ndarray, cond: Coarse, size: int = 8) -> np.ndarray:
    """Translate the kept blocks to the shift that maximizes coarse-occupancy IoU with
    the condition (ties: smallest coarse-count L1). Shifts keep the bounding box inside
    the window along every axis where it fits."""
    if not keep.any():
        return pos
    P = pos[keep]
    mn, mx = P.min(0), P.max(0)
    ext = mx - mn + 1
    rngs = [range(-mn[a], size - ext[a] - mn[a] + 1) if ext[a] <= size
            else range(size - ext[a] - mn[a], -mn[a] + 1) for a in range(3)]
    occ = cond.occ.ravel()
    G, f = cond.G, cond.f
    best, best_t = (-1.0, 0.0), np.zeros(3, np.int64)
    for tx in rngs[0]:
        for ty in rngs[1]:
            for tz in rngs[2]:
                t = np.array([tx, ty, tz])
                Q = P + t
                ins = np.all((Q >= 0) & (Q < size), 1)
                c = Q[ins] // f
                cnt = np.bincount(c[:, 0] * G * G + c[:, 1] * G + c[:, 2], minlength=G ** 3)
                o = cnt > 0
                iou = (o & occ).sum() / max((o | occ).sum(), 1)
                score = (iou, -np.abs(cnt - cond.count.ravel()).sum())
                if score > best:
                    best, best_t = score, t
    return pos + best_t
