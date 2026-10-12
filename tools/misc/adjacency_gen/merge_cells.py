"""Stage 2, second arm: the merge formulation inside cell frames.

All blocks are known from the start: the cell assignment instantiates ``A[c, t]`` blocks
of type t in coarse cell c. Every block starts as its own cluster; one step glues block
i's +a face to block j's -a face across two clusters, which fixes their relative pose.
The cell frame is what T52's merge decoder lacked:

* every cluster has a *translation box*: the set of translations that keep each member in
  its own coarse cell. A singleton's box is its cell (f^3 positions); a merge intersects
  the two boxes, and an empty intersection rejects the merge. Clusters therefore have a
  position from step one, and the coarse condition constrains every step;
* a candidate (i, j, a) must have j's cell equal to i's cell or to the next cell along +a
  (static candidate list);
* checks: no overlap inside the merged cluster; a box that shrinks to one point fixes the
  cluster in the grid, and a fixed cluster may not overlap another fixed cluster. Every
  check only gets stricter as clusters grow, so a rejected candidate is banned for good.

Type slots: singletons of one (cell, type) slot are interchangeable and get identical
features (no per-block random identity), so only one representative per slot (its
lowest-index singleton) enters the candidate set. Clusters get a random per-cluster
vector (fresh every forward).

Model (:class:`MergeNet`): per-block features (type, cell position, estimated absolute
position = box centre + offset, box extent, log cluster size, cluster vector, face state
inside the cluster, log number of singletons in the slot), then rounds of
[MLP; mean over the cluster; mean over the coarse cell -> 3x3x3 conv over the coarse grid
(the 26 neighbouring cells) + a transformer layer over cells with the condition tokens ->
back to the blocks]. Score of (i, j, a) = <q_a(h_i), k_a(h_j)> / sqrt(dk) + b_a.

Targets (:func:`targets`): a multi-block cluster is *consistent* if some translation in
its box puts every member on a real block of its type; the first such translation maps its
members to real blocks. For every real contact (u +a-> v), u is realized by the cluster
member mapped to it or, if no consistent cluster claims u, by the representative singleton
of u's slot; the same for v. If both ends are realized in different clusters that
candidate is correct. On teacher-forced states (random Kruskal prefixes of the real
contact graph) this is "the real cross-cluster contacts, singletons exchangeable within
their slot". On the decoder's own states (``dagger``) it still gives the merges that
extend consistent clusters, so the model learns to continue from its own mistakes.

Decoding: up to ``K = ceil((N - 1) / steps)`` merges per model call, tried in order of
the (Gumbel-perturbed at T > 0) scores and re-checked for legality at accept time. When
no legal candidate is left, or one cluster remains, each cluster is laid out (fixed
clusters at their point; the others, largest first, at a free translation in their box,
dropping colliding blocks if none is free).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cells import S, House, assignment, cell_of, house_coarse
from .model_faces import Fourier

DIRS = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], np.int64)
E = np.eye(3, dtype=np.int64)


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #
def blocks_from_assignment(A: np.ndarray):
    """Instantiate the slots: (types [N], flat cells [N]) sorted by cell, then type."""
    c, t = np.nonzero(A)
    reps = A[c, t]
    return np.repeat(t, reps).astype(np.int64), np.repeat(c, reps).astype(np.int64)


class MState:
    """Clusters with offsets, translation boxes, occupancy and the fixed-cluster grid."""

    def __init__(self, types: np.ndarray, cells: np.ndarray, f: int):
        self.f, self.G = f, S // f
        G = self.G
        self.t = np.asarray(types, np.int64)
        self.cell = np.asarray(cells, np.int64)
        N = len(self.t)
        self.N = N
        self.cxyz = np.stack([self.cell // (G * G), (self.cell // G) % G, self.cell % G], 1)
        self.slot = self.cell * 100_000 + self.t
        self.root = np.arange(N)
        self.off = np.zeros((N, 3), np.int64)
        self.lo = self.cxyz * f                       # per root
        self.hi = self.cxyz * f + f - 1
        self.members: Dict[int, List[int]] = {i: [i] for i in range(N)}
        self.occ: Dict[int, Dict[tuple, int]] = {i: {(0, 0, 0): i} for i in range(N)}
        self.fixed_occ: Dict[tuple, int] = {}
        self.merges = 0
        self.rejected = 0
        self.history: List[tuple] = []
        if f == 1:                                    # degenerate: every singleton fixed
            for i in range(N):
                self.fixed_occ[tuple(self.lo[i])] = i

    # ---- queries
    def n_clusters(self) -> int:
        return len(self.members)

    def size(self) -> np.ndarray:
        return np.array([len(self.members[r]) for r in self.root])

    def face_open(self) -> np.ndarray:
        """[N, 6] bool: the cell the face points at is free inside the block's cluster."""
        enc = lambda r, o: ((r * 128 + o[..., 0] + 64) * 128 + o[..., 1] + 64) * 128 + o[..., 2] + 64
        keys = enc(self.root, self.off)
        nb = enc(self.root[:, None], self.off[:, None, :] + DIRS[None])
        return ~np.isin(nb, keys)

    def reps(self) -> np.ndarray:
        """[N] bool: multi-block cluster members, plus the lowest-index singleton of each slot."""
        single = self.size() == 1
        rep = ~single
        idx = np.nonzero(single)[0]
        if len(idx):
            _, first = np.unique(self.slot[idx], return_index=True)
            rep[idx[first]] = True
        return rep

    def singles_in_slot(self) -> np.ndarray:
        """[N]: for a singleton, how many singletons its slot still has; 1 otherwise."""
        single = self.size() == 1
        out = np.ones(self.N, np.int64)
        if single.any():
            _, inv, cnt = np.unique(self.slot[single], return_inverse=True, return_counts=True)
            out[single] = cnt[inv]
        return out

    # ---- merging
    def try_merge(self, i: int, j: int, a: int) -> bool:
        """Glue i's +a face to j's -a face. Returns False (state unchanged) if illegal."""
        ri, rj = int(self.root[i]), int(self.root[j])
        if ri == rj:
            return False
        shift = self.off[i] + E[a] - self.off[j]          # j's cluster offsets -> i's frame
        # move the smaller cluster into the larger one's frame
        if len(self.members[ri]) >= len(self.members[rj]):
            keep, move, sh = ri, rj, shift
        else:
            keep, move, sh = rj, ri, -shift
        lo = np.maximum(self.lo[keep], self.lo[move] - sh)
        hi = np.minimum(self.hi[keep], self.hi[move] - sh)
        if np.any(lo > hi):
            return False
        ok = self.occ[keep]
        newpos = [tuple(int(v) for v in self.off[m] + sh) for m in self.members[move]]
        if any(p in ok for p in newpos):
            return False
        fixed = bool(np.all(lo == hi))
        if fixed:
            allp = list(ok.keys()) + newpos
            for p in allp:
                q = (p[0] + int(lo[0]), p[1] + int(lo[1]), p[2] + int(lo[2]))
                o = self.fixed_occ.get(q)
                if o is not None and o not in (keep, move):
                    return False
        # commit
        if fixed:
            for q in [k for k, v in self.fixed_occ.items() if v in (keep, move)]:
                del self.fixed_occ[q]
            for p in list(ok.keys()) + newpos:
                self.fixed_occ[(p[0] + int(lo[0]), p[1] + int(lo[1]), p[2] + int(lo[2]))] = keep
        for m, p in zip(self.members[move], newpos):
            self.off[m] = p
            self.root[m] = keep
            ok[p] = m
        self.members[keep] += self.members.pop(move)
        del self.occ[move]
        self.lo[keep], self.hi[keep] = lo, hi
        self.merges += 1
        self.history.append((i, j, a))
        return True

    def layout(self, rng: np.random.Generator):
        """Absolute positions: fixed clusters at their point, the others (largest first) at a
        free translation of their box. Returns (coords, types, dropped)."""
        occ = dict(self.fixed_occ)
        pos = np.zeros((self.N, 3), np.int64)
        keep = np.ones(self.N, bool)
        order = sorted(self.members, key=lambda r: -len(self.members[r]))
        for r in order:
            mem = self.members[r]
            lo, hi = self.lo[r], self.hi[r]
            if np.all(lo == hi):
                pos[mem] = self.off[mem] + lo
                continue
            g = np.stack(np.meshgrid(*[np.arange(lo[k], hi[k] + 1) for k in range(3)], indexing="ij"), -1).reshape(-1, 3)
            g = g[rng.permutation(len(g))]
            best, bt = None, None
            for tr in g:
                P = self.off[mem] + tr
                hit = sum((int(p[0]), int(p[1]), int(p[2])) in occ for p in P)
                if best is None or hit < best:
                    best, bt = hit, tr
                if hit == 0:
                    break
            P = self.off[mem] + bt
            for m, p in zip(mem, P):
                key = (int(p[0]), int(p[1]), int(p[2]))
                if key in occ:
                    keep[m] = False
                else:
                    occ[key] = r
            pos[mem] = P
        return pos[keep], self.t[keep], int((~keep).sum())


def state_from_partition(types, cells, coords, comp, f) -> MState:
    """Teacher-forcing state: clusters = ``comp`` labels of real blocks at real positions."""
    st = MState(types, cells, f)
    coords = np.asarray(coords, np.int64)
    st.members, st.occ = {}, {}
    for c in np.unique(comp):
        mem = np.nonzero(comp == c)[0]
        r = int(mem[0])
        st.root[mem] = r
        st.off[mem] = coords[mem] - coords[r]
        lo = np.max(st.cxyz[mem] * f - st.off[mem], 0)
        hi = np.min(st.cxyz[mem] * f + f - 1 - st.off[mem], 0)
        st.lo[r], st.hi[r] = lo, hi
        st.members[r] = [int(m) for m in mem]
        st.occ[r] = {tuple(int(v) for v in st.off[m]): int(m) for m in mem}
        if np.all(lo == hi):
            for m in mem:
                st.fixed_occ[tuple(int(v) for v in st.off[m] + lo)] = r
    st.merges = len(types) - len(st.members)
    return st


# --------------------------------------------------------------------------- #
# candidates and targets
# --------------------------------------------------------------------------- #
def static_candidates(cells: np.ndarray, f: int) -> np.ndarray:
    """[C, 3] (i, j, a): j's cell is i's cell or the next one along +a, i != j."""
    G = S // f
    cx = np.stack([cells // (G * G), (cells // G) % G, cells % G], 1)
    by = {}
    for k, c in enumerate(cells.tolist()):
        by.setdefault(c, []).append(k)
    by = {c: np.array(v) for c, v in by.items()}
    out = []
    for c, I in by.items():
        x = np.array([c // (G * G), (c // G) % G, c % G])
        for a in range(3):
            for J in (I, by.get(int(((x + E[a]) * np.array([G * G, G, 1])).sum()) if x[a] + 1 < G else None)):
                if J is None:
                    continue
                ii, jj = np.meshgrid(I, J, indexing="ij")
                m = ii != jj
                out.append(np.stack([ii[m], jj[m], np.full(int(m.sum()), a)], 1))
    return np.concatenate(out) if out else np.zeros((0, 3), np.int64)


def legal_mask(st: MState, cand: np.ndarray, ban: np.ndarray, fo=None, rep=None) -> np.ndarray:
    fo = st.face_open() if fo is None else fo
    rep = st.reps() if rep is None else rep
    i, j, a = cand[:, 0], cand[:, 1], cand[:, 2]
    return (st.root[i] != st.root[j]) & fo[i, 2 * a] & fo[j, 2 * a + 1] & rep[i] & rep[j] & ~ban


def real_contacts(coords: np.ndarray) -> np.ndarray:
    """[E, 3] (u, v, a) with v at u + e_a."""
    c = np.asarray(coords, np.int64)
    idx = -np.ones((S + 1, S + 1, S + 1), np.int64)
    idx[c[:, 0], c[:, 1], c[:, 2]] = np.arange(len(c))
    out = []
    for a in range(3):
        n = c + E[a]
        v = idx[n[:, 0], n[:, 1], n[:, 2]]
        m = v >= 0
        out.append(np.stack([np.nonzero(m)[0], v[m], np.full(int(m.sum()), a)], 1))
    return np.concatenate(out)


def targets(st: MState, real_coords: np.ndarray, real_types: np.ndarray, contacts: np.ndarray,
            cand: np.ndarray, legal: np.ndarray, rep=None) -> np.ndarray:
    """[C] bool correct candidates (see module docstring). ``real_*`` index the real house;
    the state's blocks need not correspond to real blocks one to one."""
    rc = np.asarray(real_coords, np.int64)
    rt = np.asarray(real_types, np.int64)
    rgrid = -np.ones((S, S, S), np.int64)
    rgrid[rc[:, 0], rc[:, 1], rc[:, 2]] = np.arange(len(rc))
    claim = -np.ones(len(rc), np.int64)                 # real block -> state block
    for r, mem in st.members.items():
        if len(mem) == 1:
            continue
        mem = np.array(mem)
        lo, hi = st.lo[r], st.hi[r]
        g = np.stack(np.meshgrid(*[np.arange(lo[k], hi[k] + 1) for k in range(3)], indexing="ij"), -1).reshape(-1, 3)
        P = st.off[mem][None] + g[:, None]                                  # [K,m,3]
        inb = np.all((P >= 0) & (P < S), -1)
        Pc = np.clip(P, 0, S - 1)
        rid = rgrid[Pc[..., 0], Pc[..., 1], Pc[..., 2]]
        ok = inb & (rid >= 0) & (rt[np.maximum(rid, 0)] == st.t[mem][None])
        good = np.nonzero(ok.all(1))[0]
        if len(good):
            claim[rid[good[0]]] = mem
    rep = st.reps() if rep is None else rep
    single = st.size() == 1
    slot_rep = {}
    for k in np.nonzero(rep & single)[0]:
        slot_rep[int(st.slot[k])] = int(k)
    rslot = (np.asarray(cell_of(rc, st.f)) * 100_000 + rt)

    def who(u):
        x = claim[u]
        if x >= 0:
            return int(x)
        return slot_rep.get(int(rslot[u]), -1)

    pos = set()
    for u, v, a in contacts.tolist():
        x, y = who(u), who(v)
        if x >= 0 and y >= 0 and st.root[x] != st.root[y]:
            pos.add((x, y, a))
    if not pos:
        return np.zeros(len(cand), bool)
    key = cand[:, 0] * (8 * 4096) + cand[:, 1] * 4 + cand[:, 2]
    pk = np.array([x * (8 * 4096) + y * 4 + a for x, y, a in pos])
    return np.isin(key, pk) & legal


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
@dataclass
class MergeConfig:
    n_types: int
    f: int = 4
    d: int = 192
    rounds: int = 4
    heads: int = 6
    dk: int = 64
    rid: int = 8

    def to_dict(self):
        return asdict(self)

    @property
    def G(self) -> int:
        return S // self.f


def state_features(st: MState) -> dict:
    fo = st.face_open()
    size = st.size()
    lo, hi = st.lo[st.root], st.hi[st.root]
    return {"types": st.t, "cell": st.cell, "est": (lo + hi) / 2.0 + st.off, "ext": (hi - lo + 1) / st.f,
            "lsize": np.log(size), "multi": size > 1, "root": st.root, "fstate": fo,
            "lslot": np.log(st.singles_in_slot().clip(min=1))}


class MergeNet(nn.Module):
    def __init__(self, cfg: MergeConfig):
        super().__init__()
        self.cfg = cfg
        d, G, T = cfg.d, cfg.G, cfg.n_types
        self.type_emb = nn.Embedding(T, d)
        self.pos = Fourier(S, 4, d)
        self.cellpos = Fourier(S, 4, d)
        self.feat = nn.Linear(3 + 1 + 1 + 6 + 1 + cfg.rid, d)
        c = torch.stack(torch.meshgrid(*[torch.arange(G)] * 3, indexing="ij"), -1).reshape(-1, 3)
        self.register_buffer("centre", (c.float() + 0.5) * cfg.f - 0.5, persistent=False)
        self.cnt_lin = nn.Linear(2, d)
        self.dom_emb = nn.Embedding(T + 1, d)
        self.mlp = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
                                  for _ in range(cfg.rounds)])
        self.clu = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d)) for _ in range(cfg.rounds)])
        self.cin = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d)) for _ in range(cfg.rounds)])
        self.conv = nn.ModuleList([nn.Conv3d(d, d, 3, padding=1) for _ in range(cfg.rounds)])
        self.ctf = nn.ModuleList([nn.TransformerEncoderLayer(d, cfg.heads, 2 * d, dropout=0.0, activation="gelu",
                                                             batch_first=True, norm_first=True)
                                  for _ in range(cfg.rounds)])
        self.cout = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d)) for _ in range(cfg.rounds)])
        self.ln = nn.LayerNorm(d)
        self.q = nn.Linear(d, 3 * cfg.dk)
        self.k = nn.Linear(d, 3 * cfg.dk)
        self.bias = nn.Parameter(torch.full((3,), -2.0))

    def forward(self, feats: Sequence[dict], conds, device):
        """Per-block embeddings for a list of states -> list of [N_b, d]."""
        cfg = self.cfg
        G3 = cfg.G ** 3
        Ns = [len(x["types"]) for x in feats]
        B = len(feats)
        boff = np.repeat(np.arange(B), Ns)
        ints = np.concatenate([np.stack([x["types"], x["cell"], x["root"]], 1) for x in feats])
        ints = ints + np.stack([np.zeros_like(boff), boff * G3, boff * 100_000], 1)
        flt = np.concatenate([np.concatenate([x["ext"], x["lsize"][:, None], x["multi"][:, None],
                                              x["fstate"], x["lslot"][:, None], x["est"]], 1) for x in feats])
        ints = torch.as_tensor(ints, dtype=torch.long, device=device)
        flt = torch.as_tensor(flt, dtype=torch.float32, device=device)
        types, cell = ints[:, 0], ints[:, 1]
        _, root = torch.unique(ints[:, 2], return_inverse=True)
        nr = int(root.max()) + 1
        multi = flt[:, 4]
        rid = torch.randn(nr, cfg.rid, device=device)[root] * multi[:, None]
        cellc = self.centre[cell % G3]
        x = torch.cat([flt[:, :12], rid], 1)
        h = self.type_emb(types) + self.feat(x) + self.pos(flt[:, 12:15]) + self.cellpos(cellc)
        cd = torch.as_tensor(np.stack([np.stack(c.flat()) for c in conds]), device=device)
        cnt, dom = cd[:, 0].float(), cd[:, 1]
        ctok = (self.cnt_lin(torch.stack([cnt / cfg.f ** 3, torch.log1p(cnt)], -1)) + self.dom_emb(dom)
                + self.pos(self.centre)[None])                                   # [B,G3,d]
        ones = torch.ones(len(types), 1, device=device)
        csize = torch.zeros(nr, 1, device=device).index_add(0, root, ones).clamp(min=1)
        csz = torch.zeros(B * G3, 1, device=device).index_add(0, cell, ones).clamp(min=1)
        for r in range(cfg.rounds):
            h = h + self.mlp[r](h)
            m = torch.zeros(nr, h.shape[1], device=device).index_add(0, root, h.float()) / csize
            h = h + self.clu[r](m[root])
            cm = torch.zeros(B * G3, h.shape[1], device=device).index_add(0, cell, self.cin[r](h).float()) / csz
            cm = cm.view(B, G3, -1) + ctok
            g = cm.transpose(1, 2).reshape(B, -1, cfg.G, cfg.G, cfg.G)
            g = g + self.conv[r](g)
            cm = self.ctf[r](g.flatten(2).transpose(1, 2))
            h = h + self.cout[r](cm.reshape(B * G3, -1)[cell])
        h = self.ln(h)
        return list(torch.split(h, Ns))

    def pair_scores(self, h: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        dk = self.cfg.dk
        q = self.q(h).view(-1, 3, dk)
        k = self.k(h).view(-1, 3, dk)
        a = cand[:, 2]
        return (q[cand[:, 0], a] * k[cand[:, 1], a]).sum(-1) / math.sqrt(dk) + self.bias[a]


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def kruskal_partition(N: int, contacts: np.ndarray, rng, k: Optional[int] = None) -> np.ndarray:
    par = np.arange(N)

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    if k is None:
        k = int(rng.integers(0, max(N - 1, 1)))
    done = 0
    for u, v, _ in contacts[rng.permutation(len(contacts))]:
        if done >= k:
            break
        a, b = find(int(u)), find(int(v))
        if a != b:
            par[a] = b
            done += 1
    return np.array([find(i) for i in range(N)])


def make_example(st: MState, h: House, contacts: np.ndarray, cand: np.ndarray, cond, ban=None) -> dict:
    fo = st.face_open()
    rep = st.reps()
    ban = np.zeros(len(cand), bool) if ban is None else ban
    legal = legal_mask(st, cand, ban, fo, rep)
    tg = targets(st, h.coords, h.types, contacts, cand, legal, rep)
    lc = cand[legal]
    return {"feat": state_features(st), "cand": lc, "pos": tg[legal], "cond": cond}


def tf_example(h: House, f: int, T: int, rng) -> dict:
    """Teacher-forced example: a random Kruskal prefix of the real contact graph."""
    c = np.asarray(h.coords, np.int64)
    t = np.asarray(h.types, np.int64)
    cells = cell_of(c, f)
    con = real_contacts(c)
    comp = kruskal_partition(len(t), con, rng)
    st = state_from_partition(t, cells, c, comp, f)
    return make_example(st, h, con, static_candidates(cells, f), house_coarse(h, f, T))


class TFStream(torch.utils.data.IterableDataset):
    def __init__(self, houses, f, T, bs, seed=0):
        self.houses, self.f, self.T, self.bs, self.seed = houses, f, T, bs, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0) * 7919)
        while True:
            yield [tf_example(self.houses[i], self.f, self.T, rng)
                   for i in rng.choice(len(self.houses), self.bs, replace=False)]


def merge_loss(model, exs, device):
    hs = model([e["feat"] for e in exs], [e["cond"] for e in exs], device)
    losses, accs = [], []
    for h, e in zip(hs, exs):
        if not e["pos"].any():
            continue
        cand = torch.as_tensor(e["cand"], device=device)
        s = model.pair_scores(h, cand).float()
        p = torch.as_tensor(e["pos"], device=device)
        losses.append(torch.logsumexp(s, 0) - torch.logsumexp(s[p], 0))
        accs.append(p[s.argmax()].float())
    if not losses:
        z = sum(x.sum() for x in hs) * 0
        return {"loss": z, "acc": torch.tensor(0.0)}
    return {"loss": torch.stack(losses).mean(), "acc": torch.stack(accs).mean().detach()}


def rollout_examples(model, houses, f, T, device, rng, n_snap: int = 3, steps: int = 64) -> List[dict]:
    """DAgger states: run the decoder (T = 1, oracle assignment) on training houses and
    snapshot its own states at random rounds."""
    out = []
    A = [assignment(h, f, T) for h in houses]
    snaps = [sorted(rng.choice(steps, n_snap, replace=False).tolist()) for _ in houses]
    res = merge_cells_decode(model, houses, A, device, f, temperature=1.0, steps=steps,
                             seed=int(rng.integers(1 << 30)), snapshots=snaps)
    for h, r in zip(houses, res):
        cond = house_coarse(h, f, T)
        con = real_contacts(np.asarray(h.coords, np.int64))
        for st, ban in r["snaps"]:
            ex = make_example(st, h, con, r["cand"], cond, ban)
            if ex["pos"].any():
                out.append(ex)
    return out


def train_merge(train_houses, val_houses, cfg: MergeConfig, run: Path, device, minutes: float = 40,
                bs: int = 8, lr: float = 3e-4, seed: int = 0, dagger: bool = False, init: Optional[str] = None,
                workers: int = 8, refresh_every: int = 150, n_roll: int = 8, max_n: int = 1024) -> MergeNet:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_houses = [h for h in train_houses if len(h) <= max_n]
    val_houses = [h for h in val_houses if len(h) <= max_n]
    model = MergeNet(cfg).to(device)
    if init:
        model.load_state_dict(torch.load(Path(init) / "merge.pt", map_location=device)["state"])
    print(f"[merge] {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
          f"{len(train_houses)} train houses (N <= {max_n})", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    tf_bs = bs // 2 if dagger else bs
    dl = torch.utils.data.DataLoader(TFStream(train_houses, cfg.f, cfg.n_types, tf_bs, seed), batch_size=None,
                                     num_workers=workers, prefetch_factor=4, persistent_workers=True,
                                     collate_fn=lambda x: x)
    vrng = np.random.default_rng(123)
    vex = [tf_example(h, cfg.f, cfg.n_types, vrng) for h in val_houses[:64]]
    buf: List[dict] = []

    def val():
        model.eval()
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            r = [merge_loss(model, vex[k:k + 8], device) for k in range(0, len(vex), 8)]
        model.train()
        return {"val_loss": float(np.mean([float(x["loss"]) for x in r])),
                "val_acc": float(np.mean([float(x["acc"]) for x in r]))}

    log = open(run / "train_log.jsonl", "w")
    t0, step, troll = time.time(), 0, 0.0
    for batch in dl:
        frac = (time.time() - t0) / (minutes * 60)
        if frac >= 1:
            break
        if dagger and (step % refresh_every == 0):
            tr0 = time.time()
            model.eval()
            hs = [train_houses[i] for i in rng.choice(len(train_houses), n_roll, replace=False)]
            buf = (buf + rollout_examples(model, hs, cfg.f, cfg.n_types, device, rng))[-256:]
            model.train()
            troll += time.time() - tr0
        if dagger and buf:
            batch = batch + [buf[i] for i in rng.choice(len(buf), min(bs - len(batch), len(buf)), replace=False)]
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 300) * 0.5 * (1 + math.cos(math.pi * frac))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            r = merge_loss(model, batch, device)
        opt.zero_grad(set_to_none=True)
        r["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 50 == 0:
            rec = {"step": step, "min": round((time.time() - t0) / 60, 2), "loss": round(float(r["loss"]), 4),
                   "acc": round(float(r["acc"]), 3), "rollout_min": round(troll / 60, 2), "buf": len(buf)}
            if step % 500 == 0:
                rec.update(val())
                torch.save({"cfg": cfg.to_dict(), "state": model.state_dict(), "step": step}, run / "merge.pt")
            if device.type == "cuda":
                rec["max_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
    rec = {"step": step, "final": val(), "train_min": round((time.time() - t0) / 60, 2),
           "rollout_min": round(troll / 60, 2)}
    print(json.dumps(rec), flush=True)
    log.write(json.dumps(rec) + "\n"); log.close()
    torch.save({"cfg": cfg.to_dict(), "state": model.state_dict(), "step": step}, run / "merge.pt")
    model.eval()
    return model


def load_merge(path: Path, device) -> MergeNet:
    ck = torch.load(path, map_location=device)
    m = MergeNet(MergeConfig(**ck["cfg"])).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #
@torch.no_grad()
def merge_cells_decode(model, houses: Sequence[House], assigns: Sequence[np.ndarray], device, f: int,
                       temperature: float = 0.0, steps: int = 256, seed: int = 0, bs: int = 16,
                       policy: str = "model", snapshots=None, max_tries: Optional[int] = None) -> List[dict]:
    """Returns per house coords, types, complete (one cluster), n_clusters, dropped,
    rejected, calls (and ``snaps`` = [(state copy, ban)] when ``snapshots`` gives rounds)."""
    import copy
    T = assigns[0].shape[1]
    out = []
    for b0 in range(0, len(houses), bs):
        hs = list(houses[b0:b0 + bs])
        rng = np.random.default_rng(seed + b0)
        sts, cands, bans, conds = [], [], [], []
        for h, A in zip(hs, assigns[b0:b0 + bs]):
            t, c = blocks_from_assignment(A)
            st = MState(t, c, f)
            sts.append(st)
            cd = static_candidates(c, f)
            cands.append(cd)
            bans.append(np.zeros(len(cd), bool))
            conds.append(house_coarse(h, f, T))
        done = [st.n_clusters() <= 1 for st in sts]
        boost = [1] * len(hs)                      # grows when a round accepts nothing
        snaps = [[] for _ in hs]
        rnd = 0
        while not all(done):
            act = [k for k in range(len(hs)) if not done[k]]
            if snapshots is not None:
                for k in act:
                    if rnd in snapshots[b0 + k]:
                        snaps[k].append((copy.deepcopy(sts[k]), bans[k].copy()))
                if all(rnd > max(snapshots[b0 + k]) for k in range(len(hs))):
                    break
            fos = {k: sts[k].face_open() for k in act}
            reps = {k: sts[k].reps() for k in act}
            legal = {k: legal_mask(sts[k], cands[k], bans[k], fos[k], reps[k]) for k in act}
            if policy == "model":
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    hh = model([state_features(sts[k]) for k in act], [conds[k] for k in act], device)
            lis = {k: np.nonzero(legal[k])[0] for k in act}
            if policy == "model":
                cc = [cands[k][lis[k]] for k in act]
                offs = np.cumsum([0] + [len(x) for x in hh])[:-1]
                allc = np.concatenate([c + np.array([o, o, 0]) for c, o in zip(cc, offs)])
                allsc = model.pair_scores(torch.cat(hh), torch.as_tensor(allc, device=device)).float().cpu().numpy()
                bnd = np.cumsum([0] + [len(c) for c in cc])
            for r_, k in enumerate(act):
                st = sts[k]
                li = lis[k]
                if len(li) == 0:
                    done[k] = True
                    continue
                if policy == "model":
                    sc = allsc[bnd[r_]:bnd[r_ + 1]]
                    tmp = temperature
                else:
                    sc = np.zeros(len(li))
                    tmp = 1.0
                if tmp > 0:
                    sc = sc / tmp + rng.gumbel(size=len(sc))
                K = max(1, math.ceil((st.N - 1) / steps))
                tries = (max_tries or (4 * K + 64)) * boost[k]
                order = np.argsort(-sc)[:tries]
                got = 0
                for o in order:
                    ci = li[o]
                    i, j, a = (int(v) for v in cands[k][ci])
                    if st.root[i] == st.root[j]:
                        continue
                    # faces may have closed earlier in this round
                    oi = st.off[i] + E[a]
                    if (int(oi[0]), int(oi[1]), int(oi[2])) in st.occ[int(st.root[i])]:
                        continue
                    oj = st.off[j] - E[a]
                    if (int(oj[0]), int(oj[1]), int(oj[2])) in st.occ[int(st.root[j])]:
                        continue
                    if st.try_merge(i, j, a):
                        got += 1
                        if got >= K or st.n_clusters() <= 1:
                            break
                    else:
                        bans[k][ci] = True
                        st.rejected += 1
                boost[k] = 1 if got else boost[k] * 2
                if st.n_clusters() <= 1:
                    done[k] = True
            rnd += 1
        for k, st in enumerate(sts):
            coords, types, dropped = st.layout(rng)
            o = {"coords": coords, "types": types, "complete": st.n_clusters() == 1,
                 "n_clusters": st.n_clusters(), "dropped": dropped, "rejected": st.rejected, "calls": rnd,
                 "merges": st.merges}
            if snapshots is not None:
                o["snaps"] = snaps[k]
                o["cand"] = cands[k]
            out.append(o)
    return out
