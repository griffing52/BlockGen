"""MaskGIT-style edge decoding with an optional realizability check, and BFS realization.

Decoding state is a committed label matrix ``C`` (``-1`` = undecided). Each step runs the
scorer on the committed edges and commits the most confident undecided *edges* (the
best non-"none" label of each pair, ranked by its log-probability). Committing pairs
by plain argmax does not work: a given pair is an edge with probability ~3/N, so
"none" wins every pair and nothing is ever connected (seen in the first smoke run).

* stopping: with the check, until the nodes form one component (training chunks are
  connected by construction, and once connected every remaining pair is fixed by
  geometry); without it, until the model's expected edge count at full mask,
  ``round(sum P(edge))`` (at least N-1), has been committed. Leftover pairs are "none".

* ``check=True``: nodes live in components with known relative anchors. Committing an
  offset edge between two components translates one onto the other; if any voxel would
  share a cell the label is *banned* for that pair (stays undecided, re-scored next
  step). After a merge, every pair spanning the two components is fixed by geometry
  (touching -> its offset class, else "none"). Geometry wins over earlier "none"
  commits that end up touching; those are counted (``none_overridden``).
* ``grow=True`` (with the check): after the first committed edge, only pairs with
  exactly one end in the grown component are candidates, so decoding is a single
  inventory-conditioned growth process (an autoregressive spanning tree, several
  attachments per forward pass) instead of merging many components.
* ``check=False``: labels are committed as predicted (antisymmetric pairs). At the end,
  :func:`realize` BFS-places nodes from committed edges, highest confidence first, and
  counts collisions and edges the placement contradicts.

The scorer interface is ``scorer(types, obs, ban) -> log-probs [N,N,K+1]`` with
``obs`` the pair codes of :func:`model.obs_codes`; the trained model and the
type-pair prior baseline both fit it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .data import NodeVocab, OffsetSpace, ValidTable, contact_offsets, touching

Scorer = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def valid_matrix(types: np.ndarray, vt: ValidTable) -> np.ndarray:
    N = len(types)
    out = np.zeros((N, N, vt.space.K + 1), dtype=bool)
    for i in range(N):
        for j in range(N):
            if i != j:
                out[i, j] = vt.mask(types[i], types[j])
    return out


def model_scorer(model, vt: ValidTable, device) -> Scorer:
    cache = {}

    @torch.no_grad()
    def score(types, obs, ban):
        key = np.asarray(types).tobytes()
        if key not in cache:
            cache.clear()
            cache[key] = valid_matrix(types, vt)
        t = torch.as_tensor(types, device=device)[None]
        pad = torch.zeros_like(t, dtype=torch.bool)
        valid = torch.as_tensor(cache[key] & ~ban, device=device)[None]
        z = model(t, pad, torch.as_tensor(obs, device=device)[None], valid)[0]
        return torch.log_softmax(z.float(), -1).cpu().numpy()
    return score


class Placement:
    """Union of rigid components with relative anchors (``pos``) and occupied cells."""

    def __init__(self, shapes: Sequence[np.ndarray]):
        self.shapes = [np.asarray(s) for s in shapes]
        N = len(shapes)
        self.comp = list(range(N))
        self.members: Dict[int, List[int]] = {i: [i] for i in range(N)}
        self.pos = np.zeros((N, 3), dtype=np.int64)

    def cells(self, c: int, shift=(0, 0, 0)) -> Dict[tuple, int]:
        out = {}
        for u in self.members[c]:
            for v in (self.shapes[u] + self.pos[u] + np.asarray(shift)).tolist():
                out[tuple(v)] = u
        return out

    def try_merge(self, i: int, j: int, d) -> bool:
        """Place j at pos_i + d by translating j's component; False on a collision."""
        ci, cj = self.comp[i], self.comp[j]
        assert ci != cj
        shift = self.pos[i] + np.asarray(d) - self.pos[j]
        a = self.cells(ci)
        b = self.cells(cj, shift)
        if a.keys() & b.keys():
            return False
        if len(self.members[ci]) < len(self.members[cj]):       # keep the big one fixed
            ci, cj, shift = cj, ci, -shift
        for u in self.members[cj]:
            self.pos[u] += shift
            self.comp[u] = ci
        self.members[ci] += self.members.pop(cj)
        return True


def decode(types: np.ndarray, vocab: NodeVocab, scorer: Scorer, steps: int = 24,
           check: bool = True, temperature: float = 0.0, seed: int = 0, grow: bool = False) -> dict:
    """Generate committed labels for a node multiset. Returns a result dict with
    ``labels`` [N,N], ``pos`` [N,3] (check=True only), ``pred`` [N,N] (the scorer's
    last argmax for each pair while it was undecided) and counters."""
    rng = np.random.default_rng(seed)
    space = vocab.space
    opp = space.opp
    offsets = space.offsets
    N = len(types)
    shapes = [vocab.shapes[int(t)] for t in types]
    C = -np.ones((N, N), dtype=np.int64)
    np.fill_diagonal(C, 0)
    pred = np.zeros((N, N), dtype=np.int64)
    lastlp = np.full((N, N, space.K + 1), -np.inf, dtype=np.float32)
    conf_at = np.zeros((N, N))
    ban = np.zeros((N, N, space.K + 1), dtype=bool)
    pl = Placement(shapes) if check else None
    iu = np.triu_indices(N, 1)
    P = len(iu[0])
    stats = defaultdict(int)

    def set_pair(i, j, k):
        C[i, j] = k
        C[j, i] = opp[k]

    def propagate(c):
        """All pairs inside component c are determined by geometry."""
        mem = pl.members[c]
        adj = touching(pl.pos[mem], [shapes[u] for u in mem])
        for a in range(len(mem)):
            for b in range(a + 1, len(mem)):
                u, v = mem[a], mem[b]
                k = space.cls(pl.pos[v] - pl.pos[u]) if adj[a, b] else 0
                k = max(k, 0)
                if C[u, v] == 0 and k > 0:
                    stats["none_overridden"] += 1
                if C[u, v] < 0 or C[u, v] != k:
                    set_pair(u, v, k)

    def gumbel(shape):
        return -np.log(-np.log(rng.random(shape) + 1e-12) + 1e-12)

    E_target = None
    n_edges = 0
    for step in range(1, 4 * steps + 1):
        left = max(1, steps - step + 1)
        if check:
            n_comp = len(pl.members)
            if n_comp == 1:
                break
            n_commit = int(math.ceil((n_comp - 1) / left))
        elif E_target is not None:
            if n_edges >= E_target:
                break
            n_commit = int(math.ceil((E_target - n_edges) / left))
        und = C[iu] < 0
        if not und.any():
            break
        obs = np.where(C >= 0, C + 1, 0)
        logp = scorer(types, obs, ban)
        ii, jj = iu[0][und], iu[1][und]
        if grow and stats.get("merges", 0) > 0:
            # single-tree growth: only pairs with exactly one end in the grown component
            main = max(pl.members, key=lambda c: len(pl.members[c]))
            ci = np.array(pl.comp)[ii] == main
            cj = np.array(pl.comp)[jj] == main
            sel = ci ^ cj
            if not sel.any():
                break
            ii, jj = ii[sel], jj[sel]
        elif grow:
            n_commit = 1                              # one seed edge, then grow from it
        lp = logp[ii, jj]                                           # [M,K+1]
        full = np.argmax(lp, -1)
        pred[ii, jj] = full
        pred[jj, ii] = opp[full]
        lastlp[ii, jj] = lp
        lastlp[jj, ii] = lp[:, opp]
        if E_target is None and not check:
            # expected edge count under the model at full mask, at least a spanning tree
            p_edge = 1.0 - np.exp(lp[:, 0])
            E_target = max(N - 1, int(round(p_edge.sum())))
            n_commit = int(math.ceil(E_target / left))
        el = lp[:, 1:]                                              # edge labels only
        if temperature > 0:
            choice = np.argmax(el / temperature + gumbel(el.shape), -1) + 1
        else:
            choice = np.argmax(el, -1) + 1
        conf = lp[np.arange(len(ii)), choice]
        ok = np.isfinite(conf)
        if temperature > 0:   # MaskGIT: noisy confidence so the commit order is sampled
            conf = conf + temperature * gumbel(conf.shape)
        conf = np.where(ok, conf, -np.inf)
        order = np.argsort(-conf)
        done = 0
        for o in order:
            if done >= n_commit or not np.isfinite(conf[o]):
                break
            i, j, k = int(ii[o]), int(jj[o]), int(choice[o])
            if C[i, j] >= 0:                         # fixed meanwhile by propagation
                continue
            conf_at[i, j] = conf_at[j, i] = conf[o]
            if not check:
                set_pair(i, j, k)
                n_edges += 1
                done += 1
                continue
            if pl.comp[i] == pl.comp[j]:             # cannot happen: propagated
                continue
            if pl.try_merge(i, j, offsets[k - 1]):
                stats["merges"] += 1
                propagate(pl.comp[i])
                done += 1
            else:
                stats["rejected"] += 1
                ban[i, j, k] = True
                ban[j, i, opp[k]] = True
                if not ((logp[i, j, 1:] > -np.inf) & ~ban[i, j, 1:]).any():
                    set_pair(i, j, 0)                # nothing left but "none"
        if done == 0 and check and not stats.get("_stalled"):
            stats["stalled_steps"] += 1
    stats["forwards"] = step
    # anything still undecided after the schedule: none
    C[C < 0] = 0
    res = {"labels": C, "pred": lastlp, "conf": conf_at, "stats": dict(stats)}
    if check:
        res["pos"] = pl.pos.copy()
        res["comp"] = np.array(pl.comp)
    return res


def realize(labels: np.ndarray, conf: np.ndarray, shapes: Sequence[np.ndarray],
            space: OffsetSpace) -> dict:
    """BFS placement from committed offset edges (no check). Spanning forest by
    confidence (Kruskal order); each tree placed independently from its root.

    Returns ``pos``, ``comp`` (tree id), ``conflict`` [N] bool (node's voxels hit an
    already-placed node of the same tree), ``edge_ok`` fraction of committed offset
    edges whose endpoints the placement satisfies."""
    N = len(labels)
    offsets = space.offsets
    parent = list(range(N))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    edges = [(conf[i, j], i, j) for i in range(N) for j in range(i + 1, N) if labels[i, j] > 0]
    edges.sort(reverse=True)
    tree = defaultdict(list)
    for _, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            tree[i].append((j, offsets[labels[i, j] - 1]))
            tree[j].append((i, offsets[labels[j, i] - 1]))
    pos = np.zeros((N, 3), dtype=np.int64)
    placed = np.zeros(N, bool)
    conflict = np.zeros(N, bool)
    comp = -np.ones(N, dtype=np.int64)
    for root in range(N):
        if placed[root]:
            continue
        occ = {}
        queue = [root]
        placed[root] = True
        comp[root] = root
        while queue:
            u = queue.pop(0)
            cells = [tuple(v) for v in (np.asarray(shapes[u]) + pos[u]).tolist()]
            if any(c in occ for c in cells):
                conflict[u] = True
            else:
                for c in cells:
                    occ[c] = u
            for v, d in tree[u]:
                if not placed[v]:
                    placed[v] = True
                    comp[v] = root
                    pos[v] = pos[u] + d
                    queue.append(v)
    ok = tot = 0
    for _, i, j in edges:
        tot += 1
        ok += int(comp[i] == comp[j] and np.array_equal(pos[j] - pos[i], offsets[labels[i, j] - 1]))
    return {"pos": pos, "comp": comp, "conflict": conflict,
            "edge_ok": ok / max(tot, 1), "n_edges": tot}


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def random_tree(types: np.ndarray, vocab: NodeVocab, vt: ValidTable, check: bool,
                seed: int = 0, tries: int = 50) -> dict:
    """Random spanning tree with random (shape-valid) offsets. ``check=True`` retries
    parent/offset until the new node does not collide (random growth)."""
    rng = np.random.default_rng(seed)
    N = len(types)
    shapes = [vocab.shapes[int(t)] for t in types]
    order = rng.permutation(N)
    pos = np.zeros((N, 3), dtype=np.int64)
    occ = {tuple(v): order[0] for v in shapes[order[0]].tolist()}
    conflict = np.zeros(N, bool)
    offs = vocab.space.offsets
    for k in range(1, N):
        u = order[k]
        for t in range(tries if check else 1):
            p = order[rng.integers(0, k)]
            m = vt.mask(types[p], types[u])[1:]
            d = offs[rng.choice(np.nonzero(m)[0])]
            cand = pos[p] + d
            cells = [tuple(v) for v in (shapes[u] + cand).tolist()]
            if not any(c in occ for c in cells):
                break
        pos[u] = cand
        if any(c in occ for c in cells):
            conflict[u] = True
        else:
            for c in cells:
                occ[c] = u
    return {"pos": pos, "comp": np.zeros(N, dtype=np.int64), "conflict": conflict}


class TypePrior:
    """Block-type-only heuristic: P(label | type_i, type_j) from train counts, backed off
    to P(label | shape_i, shape_j) and masked to shape-valid labels. Ignores observed
    edges, so its MaskGIT decode is driven by the check alone."""

    def __init__(self, graphs, vocab: NodeVocab, vt: ValidTable, alpha: float = 0.5):
        from .data import edge_labels
        K = vocab.space.K
        self.vt = vt
        self.K = K
        self.alpha = alpha
        self.counts: Dict[tuple, np.ndarray] = {}
        self.marg = np.zeros(K + 1)
        for g in graphs:
            lab, _ = edge_labels(g.coords, [vocab.shapes[t] for t in g.types], vocab.space)
            N = len(g)
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    key = (int(g.types[i]), int(g.types[j]))
                    c = self.counts.get(key)
                    if c is None:
                        c = self.counts[key] = np.zeros(K + 1)
                    c[lab[i, j]] += 1
                    self.marg[lab[i, j]] += 1

    def logp(self, ta, tb) -> np.ndarray:
        m = self.vt.mask(ta, tb)
        base = self.marg * m
        base = base / max(base.sum(), 1e-9)
        c = self.counts.get((int(ta), int(tb)), np.zeros(self.K + 1)) * m
        p = (c + self.alpha * 20 * base) + 1e-6 * m
        p = p / p.sum()
        with np.errstate(divide="ignore"):
            return np.where(m, np.log(p), -np.inf)

    def scorer(self) -> Scorer:
        cache: Dict[tuple, np.ndarray] = {}

        def score(types, obs, ban):
            N = len(types)
            out = np.full((N, N, self.K + 1), -np.inf)
            for i in range(N):
                for j in range(N):
                    if i != j:
                        key = (int(types[i]), int(types[j]))
                        if key not in cache:
                            cache[key] = self.logp(*key)
                        out[i, j] = cache[key]
            out[ban] = -np.inf
            return out
        return score
