"""Merge formulation: all blocks known up front, one glued face pair per step.

State: a partition of the inventory into rigid components (initially every block alone),
each with relative block positions. A *connection* glues face (i, +a) to face (j, -a)
with i and j in different components; it fixes the relative pose of the two components
(j lands at pos_i + e_a). With the check, a merge whose rigid bodies would overlap is
rejected (that pair is banned and the step re-samples), and so is a merge whose bounding
box would exceed the window (``extent``; the first evaluation ran without it). Every
face contact implied by the merge (blocks now in face-adjacent cells) is glued automatically, because positions are
what the model sees. Generation ends after exactly N - 1 merging connections (one
component); there is no stop token. Clusters can form independently and merge later.

Model input per state (``model_faces.FaceModel`` with ``merge=True``): component id as a
random per-component vector, log component size, the block's offset to its component
anchor (the component's lowest-index block, shifted by S/2), the multi-block flag and the
per-face buried state inside the component. Scores are the three axis matrices; the
candidate set at a step is every complementary face pair across two components.

Training (teacher forcing over random merge orders): a random Kruskal order of the real
contact edges is cut after k merges (k uniform in 0..N-2); the target is the set of ALL
true connections between two current components (a real contact edge across components),
and the loss is ``logsumexp(all candidates) - logsumexp(true candidates)``. Singletons of
the same type are exchangeable, so a connection that uses a same-type singleton in place
of the real one is also counted as true.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from .faces import DIRS, action_scores


# --------------------------------------------------------------------------- #
# states and targets
# --------------------------------------------------------------------------- #
def _find(par, a):
    while par[a] != a:
        par[a] = par[par[a]]
        a = par[a]
    return a


def random_merge_state(coords: np.ndarray, G: np.ndarray, rng: np.random.Generator,
                       k: Optional[int] = None) -> np.ndarray:
    """Component label per block after k merges of a random Kruskal order of the real
    contact edges (``G`` = :func:`faces.glue_matrices`)."""
    N = len(coords)
    edges = np.argwhere(G)                                   # [E,3] (a, i, j)
    edges = edges[rng.permutation(len(edges))]
    if k is None:
        k = int(rng.integers(0, max(N - 1, 1)))
    par = list(range(N))
    done = 0
    for _, i, j in edges:
        if done >= k:
            break
        ri, rj = _find(par, int(i)), _find(par, int(j))
        if ri != rj:
            par[ri] = rj
            done += 1
    comp = np.array([_find(par, i) for i in range(N)])
    _, comp = np.unique(comp, return_inverse=True)
    return comp


def merge_targets(G: np.ndarray, types: np.ndarray, comp: np.ndarray) -> np.ndarray:
    """[3, N, N] bool: true connections across components (singleton type exchange)."""
    N = len(types)
    size = np.bincount(comp, minlength=N)
    single = size[comp] == 1
    C = np.zeros_like(G)
    for a, i, j in np.argwhere(G):
        if comp[i] == comp[j]:
            continue
        I = np.nonzero(single & (types == types[i]))[0] if single[i] else np.array([i])
        J = np.nonzero(single & (types == types[j]))[0] if single[j] else np.array([j])
        C[a][np.ix_(I, J)] = True
    for a in range(3):
        np.fill_diagonal(C[a], False)
    return C


def comp_positions(pos: np.ndarray, comp: np.ndarray, size: int = 8) -> np.ndarray:
    """Offsets to each component's anchor (lowest-index member), shifted by size // 2."""
    out = np.zeros_like(pos)
    for c in np.unique(comp):
        m = np.nonzero(comp == c)[0]
        out[m] = pos[m] - pos[m[0]] + size // 2
    return out


def comp_fstate(pos: np.ndarray, comp: np.ndarray) -> np.ndarray:
    """[N, 6]: 1 open, 2 facing a block of the same component (singletons are all open)."""
    N = len(pos)
    st = np.ones((N, 6), np.int64)
    key = {(int(comp[i]), *map(int, pos[i])) for i in range(N)}
    for i in range(N):
        for d in range(6):
            q = pos[i] + DIRS[d]
            if (int(comp[i]), int(q[0]), int(q[1]), int(q[2])) in key:
                st[i, d] = 2
    return st


def collate_merge(items: Sequence[dict], comps: Sequence[np.ndarray], poss: Sequence[np.ndarray],
                  device, cfg) -> dict:
    """Model inputs for merge states (``poss`` = positions consistent within components)."""
    from .train_faces import collate_states
    S = cfg.size
    rel = [comp_positions(p, c, S) for p, c in zip(poss, comps)]
    multi = [np.bincount(c)[c] > 1 for c in comps]
    fst = [comp_fstate(p, c) for p, c in zip(poss, comps)]
    batch = collate_states(items, multi, rel, device, cfg, fstate_list=fst)
    N = batch["types"].shape[1]
    cp = np.zeros((len(items), N), np.int64)
    for b, c in enumerate(comps):
        cp[b, :len(c)] = c
        cp[b, len(c):] = np.arange(len(c), N)              # padding: own components
    batch["comp"] = torch.as_tensor(cp, device=device)
    return batch


def merge_losses(model, items, rng: np.random.Generator, device, cfg) -> dict:
    """Pointer loss over merge states plus the glue/exposed losses of train_faces."""
    import torch.nn.functional as F
    comps, tgts = [], []
    for it in items:
        c = random_merge_state(it["coords"], it["glue"], rng)
        comps.append(c)
        tgts.append(merge_targets(it["glue"], it["types"], c))
    batch = collate_merge(items, comps, [it["coords"] for it in items], device, cfg)
    out = model(**batch)
    X, ex = out["X"].float(), out["ex"].float()
    Bsz, _, N, _ = X.shape
    valid = ~batch["pad"]
    comp = batch["comp"]
    T = np.zeros((Bsz, 3, N, N), bool)
    GG = np.zeros((Bsz, 3, N, N), bool)
    EX = np.zeros((Bsz, N, 6), bool)
    for b, it in enumerate(items):
        n = len(it["types"])
        T[b, :, :n, :n] = tgts[b]
        GG[b, :, :n, :n] = it["glue"]
        EX[b, :n] = it["exposed"]
    T, GG, EX = (torch.as_tensor(a, device=device) for a in (T, GG, EX))
    legal = (valid[:, :, None] & valid[:, None, :] & (comp[:, :, None] != comp[:, None, :]))[:, None].expand_as(X)
    res = {}
    has = T.flatten(1).any(1)
    lse_all = X.masked_fill(~legal, -1e9).flatten(1).logsumexp(1)
    lse_c = X.masked_fill(~T, -1e9).flatten(1).logsumexp(1)
    res["ptr"] = (lse_all - lse_c)[has].mean()
    am = X.masked_fill(~legal, -1e9).flatten(1).argmax(1)
    res["ptr_acc"] = T.flatten(1).gather(1, am[:, None]).squeeze(1)[has].float().mean().detach()
    # glue dual softmax with dust bin + exposed BCE (as in train_faces)
    eye = torch.eye(N, dtype=torch.bool, device=device)
    pv = (valid[:, :, None] & valid[:, None, :] & ~eye)[:, None].expand_as(X)
    Xm = X.masked_fill(~pv, -1e9)
    ex_p = ex[:, :, 0::2].permute(0, 2, 1)
    ex_m = ex[:, :, 1::2].permute(0, 2, 1)
    vr = valid[:, None, :].expand(Bsz, 3, N)
    full = torch.full_like(GG[..., 0], N, dtype=torch.long)
    t_r = torch.where(GG.any(-1), GG.float().argmax(-1), full)
    t_c = torch.where(GG.any(-2), GG.float().argmax(-2), full)
    res["glue"] = 0.5 * (F.cross_entropy(torch.cat([Xm, ex_p[..., None]], -1)[vr], t_r[vr])
                         + F.cross_entropy(torch.cat([Xm.transpose(-1, -2), ex_m[..., None]], -1)[vr], t_c[vr]))
    res["exposed"] = F.binary_cross_entropy_with_logits(ex[valid[..., None].expand_as(ex)],
                                                        EX[valid[..., None].expand_as(ex)].float())
    res["loss"] = res["ptr"] + res["glue"] + res["exposed"]
    return res


# --------------------------------------------------------------------------- #
# decoder
# --------------------------------------------------------------------------- #
class _State:
    def __init__(self, n: int):
        self.comp = np.arange(n)
        self.pos = np.zeros((n, 3), np.int64)
        self.ban = np.zeros((3, n, n), bool)
        self.conflict = np.zeros(n, bool)
        self.merges = 0
        self.rejected = 0
        self.done = n <= 1
        self.history: List[tuple] = []          # accepted connections (a, i, j)


def fits_condition(P: np.ndarray, cond, size: int) -> bool:
    """True if some translation puts the rigid block set P inside the window with every
    coarse cell holding at most its condition count (so only occupied cells are used)."""
    mn, mx = P.min(0), P.max(0)
    ext = mx - mn + 1
    if np.any(ext > size):
        return False
    G, f = cond.G, cond.f
    cap = cond.count.ravel()
    rng = [np.arange(-mn[a], size - ext[a] - mn[a] + 1) for a in range(3)]
    T = np.stack(np.meshgrid(*rng, indexing="ij"), -1).reshape(-1, 3)
    Q = (P[None] + T[:, None]) // f                                   # [K,m,3]
    idx = Q[..., 0] * G * G + Q[..., 1] * G + Q[..., 2]
    K = len(T)
    cnt = np.zeros((K, G ** 3), np.int64)
    np.add.at(cnt, (np.repeat(np.arange(K), P.shape[0]), idx.ravel()), 1)
    return bool((cnt <= cap[None]).all(1).any())


def _apply_merge(s: _State, i: int, j: int, a: int, check: bool, extent: Optional[int] = None,
                 cond=None) -> bool:
    ci, cj = s.comp[i], s.comp[j]
    shift = s.pos[i] + DIRS[2 * a] - s.pos[j]
    mi = s.comp == ci
    A = {tuple(v) for v in s.pos[mi].tolist()}
    mj = np.nonzero(s.comp == cj)[0]
    Bc = [tuple(v) for v in (s.pos[mj] + shift).tolist()]
    hit = [k for k, c in zip(mj, Bc) if c in A]
    if hit and check:
        return False
    if check and extent is not None:             # merged rigid body must fit the window
        P = np.concatenate([s.pos[mi], s.pos[mj] + shift])
        if np.any(P.max(0) - P.min(0) + 1 > extent):
            return False
        if cond is not None and not fits_condition(P, cond, extent):
            return False
    s.pos[mj] += shift
    s.comp[mj] = ci
    s.conflict[hit] = True                 # no-check: colliding blocks are dropped later
    s.history.append((a, i, j))
    return True


@torch.no_grad()
def merge_decode(model, items: Sequence[dict], device, check: bool = True, temperature: float = 0.0,
                 seed: int = 0, bs: int = 32, policy: str = "model", size: int = 8,
                 extent: bool = True, cmask: bool = False) -> List[dict]:
    """Returns per item pos [N,3] (relative frame of the final component), comp, conflict,
    merges, rejected. ``policy="random"``: uniform over the legal connections (no model).
    With the check, ``extent=True`` also rejects a merge whose bounding box exceeds the
    window (the sequential builder's in-window test, in a frame-free form). ``cmask=True``
    (with the check) also rejects a merge after which the component cannot be translated
    into the window without exceeding some coarse cell's condition count
    (:func:`fits_condition`), the merge analogue of the grow decoder's coarse mask."""
    from .decode_faces import _choose
    cfg = model.cfg if model is not None else None
    out = []
    for b0 in range(0, len(items), bs):
        chunk = list(items[b0:b0 + bs])
        rngs = [np.random.default_rng(seed + b0 + k) for k in range(len(chunk))]
        st = [_State(len(it["types"])) for it in chunk]
        while True:
            act = [k for k, s in enumerate(st) if not s.done]
            if not act:
                break
            sub = [chunk[k] for k in act]
            if policy == "model":
                batch = collate_merge(sub, [st[k].comp for k in act], [st[k].pos for k in act], device, cfg)
                X = model(**batch)["X"].float().cpu().numpy()
            else:
                Nm = max(len(it["types"]) for it in sub)
                X = np.zeros((len(sub), 3, Nm, Nm))
            temp = temperature if policy == "model" else 1.0
            for r, k in enumerate(act):
                s = st[k]
                n = len(chunk[k]["types"])
                x = X[r, :, :n, :n]
                while True:
                    legal = (s.comp[:, None] != s.comp[None, :])[None] & ~s.ban
                    c = _choose(x, legal, temp, rngs[k])
                    if c is None:
                        s.done = True
                        break
                    a, rem = divmod(c, n * n)
                    i, j = divmod(rem, n)
                    S_ = cfg.size if cfg is not None else size
                    if _apply_merge(s, i, j, a, check, S_ if (extent or cmask) else None,
                                    chunk[k]["coarse"] if cmask else None):
                        s.merges += 1
                        s.done = s.merges >= n - 1
                        break
                    s.ban[a, i, j] = True
                    s.rejected += 1
        for s in st:
            out.append({"pos": s.pos, "comp": s.comp, "conflict": s.conflict, "merges": s.merges,
                        "rejected": s.rejected, "history": s.history})
    return out
