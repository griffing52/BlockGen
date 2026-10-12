"""Stage 2, main arm: grow a full house inside the frame of a cell assignment.

The state is the partial build on the 32^3 grid plus the remaining type slots of every
coarse cell (``rem[c, t]`` = assigned blocks of type t in cell c not yet placed). One
action attaches one slot to the build: a type t and an empty fine cell q that is face
adjacent to a placed block (so it glues to every placed neighbour) and lies in a coarse
cell c(q) with ``rem[c(q), t] > 0``. This is the T52 grow decoder with the oracle cell
index, rewritten over type slots: the attachment "open face of p, unplaced block j" of
T52 is replaced by "free fine cell next to the build, slot type", which merges every face
pointing at the same cell (T24e's cell-marginalized target) and every interchangeable
block of the slot (same type, same cell) into one action. The check (target cell empty
and in the grid) and the coarse mask (only cells with slots left) hold by construction.
The first block of a build, and a new seed when no legal attachment is left while slots
remain (counted as ``n_seeds``), may go to any free fine cell of a coarse cell with slots.

Model (:class:`GrowNet`): full houses have up to 7,673 blocks (46k face tokens), so the
encoder is local at fine resolution and global only at the coarse level. A 3D U-Net
over the 32^3 grid of placed types (receptive field of the two fine levels: about one
coarse cell plus its 26 neighbours) meets, at the 8^3 bottleneck, a transformer over the
coarse cells whose tokens carry the condition (count, dominant type), the remaining slot
counts per type and the cell position. The slot counts are also broadcast back to the
fine grid. Score of an action ``(q, t) = <u(h_q), e_t> + b_t``.

Training: teacher forcing on prefixes of random growth orders of real houses (Dijkstra
order from a random seed with i.i.d. exponential edge weights, i.e. first-passage
percolation, which is the same law as T52's "uniform random frontier edge" order), with
the true assignment. Loss: ``-log sum_{correct} p`` over the legal set, where an action
is correct iff q holds a real block of type t (any face and any slot instance counts).

Decoding: one model call places up to ``K = ceil(N / steps)`` blocks at distinct
frontier cells (top-K of the Gumbel-perturbed scores at temperature T, or of the raw
scores at T = 0), skipping a pick whose slot ran out within the call; seeds are placed one
per call. ``policy="random"`` draws uniformly over the legal (q, t) pairs.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cells import S, House, assignment, cell_of, house_coarse
from .model_faces import Fourier


@dataclass
class GrowConfig:
    n_types: int
    f: int = 4
    e: int = 32          # fine type embedding
    c1: int = 48         # 32^3 channels
    c2: int = 96         # 16^3 channels
    d: int = 192         # 8^3 / coarse token width
    layers: int = 4
    heads: int = 6
    dk: int = 64

    def to_dict(self):
        return asdict(self)

    @property
    def G(self) -> int:
        return S // self.f


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, c), nn.Conv3d(c, c, 3, padding=1)
        self.n2, self.c2 = nn.GroupNorm(8, c), nn.Conv3d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.gelu(self.n2(self.c1(F.gelu(self.n1(x))))))


class GrowNet(nn.Module):
    def __init__(self, cfg: GrowConfig):
        super().__init__()
        self.cfg = cfg
        T, G, d = cfg.n_types, cfg.G, cfg.d
        self.type_in = nn.Embedding(T + 1, cfg.e)                     # 0 = empty
        self.rem_lin = nn.Linear(T, d)
        self.cnt_lin = nn.Linear(4, d)
        self.dom_emb = nn.Embedding(T + 1, d)
        self.cpos = Fourier(S, 4, d)
        c = torch.stack(torch.meshgrid(*[torch.arange(G)] * 3, indexing="ij"), -1).reshape(-1, 3)
        self.register_buffer("centre", (c.float() + 0.5) * cfg.f - 0.5, persistent=False)
        b = torch.stack(torch.meshgrid(*[torch.arange(8)] * 3, indexing="ij"), -1).reshape(-1, 3)
        self.register_buffer("bcentre", (b.float() + 0.5) * 4 - 0.5, persistent=False)
        self.cell_to_fine = nn.Linear(d, cfg.e)
        self.conv_in = nn.Conv3d(cfg.e, cfg.c1, 3, padding=1)
        self.r1 = Res(cfg.c1)
        self.down1 = nn.Conv3d(cfg.c1, cfg.c2, 2, stride=2)
        self.r2 = Res(cfg.c2)
        self.down2 = nn.Conv3d(cfg.c2, d, 2, stride=2)
        self.cell_to_b = nn.Linear(d, d)
        layer = nn.TransformerEncoderLayer(d, cfg.heads, 4 * d, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.up2 = nn.ConvTranspose3d(d, cfg.c2, 2, stride=2)
        self.r3 = Res(cfg.c2)
        self.up1 = nn.ConvTranspose3d(cfg.c2, cfg.c1, 2, stride=2)
        self.r4 = Res(cfg.c1)
        self.head = nn.Sequential(nn.Linear(cfg.c1, 2 * cfg.c1), nn.GELU(), nn.Linear(2 * cfg.c1, cfg.dk))
        self.type_out = nn.Embedding(T, cfg.dk)
        self.type_bias = nn.Parameter(torch.zeros(T))

    def _grid(self, x, n):                                   # [B, n^3, C] -> [B, C, n, n, n]
        B, _, C = x.shape
        return x.transpose(1, 2).reshape(B, C, n, n, n)

    def forward(self, grid_t, rem, count, dom):
        """grid_t [B,S,S,S] long (0 empty, t+1 placed); rem [B,G^3,T] float remaining slots;
        count/dom [B,G^3] long condition. Returns fine features [B,c1,S,S,S]."""
        cfg = self.cfg
        f3 = cfg.f ** 3
        remt = rem.sum(-1)
        cf = count.float()
        cell = (self.rem_lin(rem / f3) + self.dom_emb(dom) + self.cpos(self.centre)[None]
                + self.cnt_lin(torch.stack([cf / f3, remt / f3, torch.log1p(cf), torch.log1p(remt)], -1)))
        G = cfg.G
        cf_fine = self._grid(self.cell_to_fine(cell), G)
        cf_fine = cf_fine.repeat_interleave(cfg.f, 2).repeat_interleave(cfg.f, 3).repeat_interleave(cfg.f, 4)
        x = self.type_in(grid_t).permute(0, 4, 1, 2, 3) + cf_fine
        s1 = self.r1(self.conv_in(x))
        s2 = self.r2(self.down1(s1))
        x = self.down2(s2)                                    # [B,d,8,8,8]
        cb = self._grid(self.cell_to_b(cell), G)
        if G < 8:
            cb = F.interpolate(cb, size=(8, 8, 8), mode="nearest")
        elif G > 8:
            cb = F.adaptive_avg_pool3d(cb, 8)
        x = x + cb
        B = x.shape[0]
        tok = x.flatten(2).transpose(1, 2) + self.cpos(self.bcentre)[None]
        x = self._grid(self.ln(self.enc(tok)), 8)
        x = self.r3(self.up2(x) + s2)
        return self.r4(self.up1(x) + s1)

    def scores(self, h, b, q):
        """Scores [M, T] of fine cells (b, q = [M,3])."""
        hq = h.permute(0, 2, 3, 4, 1)[b, q[:, 0], q[:, 1], q[:, 2]]
        u = self.head(hq.float())
        return u @ self.type_out.weight.t() / math.sqrt(self.cfg.dk) + self.type_bias


# --------------------------------------------------------------------------- #
# legal sets
# --------------------------------------------------------------------------- #
def face_dilate(occ: torch.Tensor) -> torch.Tensor:
    """[B,S,S,S] bool -> cells with a face-adjacent occupied cell."""
    p = F.pad(occ.float()[:, None], (1, 1, 1, 1, 1, 1))[:, 0] > 0
    return (p[:, 2:, 1:-1, 1:-1] | p[:, :-2, 1:-1, 1:-1] | p[:, 1:-1, 2:, 1:-1] | p[:, 1:-1, :-2, 1:-1]
            | p[:, 1:-1, 1:-1, 2:] | p[:, 1:-1, 1:-1, :-2])


def fine_cell_index(f: int, device) -> torch.Tensor:
    G = S // f
    g = torch.arange(S, device=device) // f
    return (g[:, None, None] * G * G + g[None, :, None] * G + g[None, None, :])        # [S,S,S]


def frontier(grid_t, rem, f: int, reseed: bool = False):
    """Legal fine cells [B,S,S,S] bool and per-house seed flag [B]. A house with nothing
    placed (or, with ``reseed``, no legal attachment while slots remain) may use any free
    fine cell of a coarse cell with slots left."""
    occ = grid_t > 0
    fc = fine_cell_index(f, grid_t.device)
    has_slot = (rem.sum(-1) > 0)[:, fc]                                # [B,S,S,S]
    free = ~occ & has_slot
    front = free & face_dilate(occ)
    empty = ~occ.flatten(1).any(1)
    seed = empty.clone()
    if reseed:
        seed |= ~front.flatten(1).any(1) & free.flatten(1).any(1)
    front = torch.where(seed[:, None, None, None], free, front)
    return front, seed


def legal_scores(model, grid_t, rem, count, dom, f, reseed=False, h=None):
    """(b, q, legal [M,T], scores [M,T] or None, seed flags)."""
    front, seed = frontier(grid_t, rem, f, reseed)
    idx = front.nonzero()
    b, q = idx[:, 0], idx[:, 1:]
    fc = fine_cell_index(f, grid_t.device)[q[:, 0], q[:, 1], q[:, 2]]
    legal = rem[b, fc] > 0
    sc = None
    if model is not None:
        if h is None:
            h = model(grid_t, rem, count, dom)
        sc = model.scores(h, b, q)
    return b, q, legal, sc, seed


# --------------------------------------------------------------------------- #
# training data
# --------------------------------------------------------------------------- #
def growth_order(coords: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random attachment order: Dijkstra from a random seed with Exp(1) edge weights."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    c = np.asarray(coords, np.int64)
    N = len(c)
    idx = -np.ones((S + 1, S + 1, S + 1), np.int64)
    idx[c[:, 0], c[:, 1], c[:, 2]] = np.arange(N)
    I, J = [], []
    for a in range(3):
        n = c.copy()
        n[:, a] += 1
        j = idx[n[:, 0], n[:, 1], n[:, 2]]
        m = j >= 0
        I.append(np.nonzero(m)[0]); J.append(j[m])
    I, J = np.concatenate(I), np.concatenate(J)
    w = rng.exponential(size=len(I)) + 1e-9
    M = coo_matrix((w, (I, J)), shape=(N, N)).tocsr()
    d = dijkstra(M, directed=False, indices=int(rng.integers(N)))
    d[~np.isfinite(d)] = 1e18 + rng.random(int((~np.isfinite(d)).sum()))
    return np.argsort(d, kind="stable")


def make_state(h: House, A: np.ndarray, f: int, n_types: int, rng, p0: float = 0.05,
               k: Optional[int] = None) -> dict:
    """One teacher-forcing example: a random growth prefix of a real house."""
    c = np.asarray(h.coords, np.int64)
    t = np.asarray(h.types, np.int64)
    N = len(t)
    if k is None:
        k = 0 if rng.random() < p0 else int(rng.integers(1, N))
    order = growth_order(c, rng) if k > 0 else np.zeros(0, np.int64)
    pre = order[:k]
    grid = np.zeros((S, S, S), np.int16)
    grid[c[pre, 0], c[pre, 1], c[pre, 2]] = t[pre] + 1
    real = np.zeros((S, S, S), np.int16)
    real[c[:, 0], c[:, 1], c[:, 2]] = t + 1
    rem = A.astype(np.float32).copy()
    np.add.at(rem, (cell_of(c[pre], f), t[pre]), -1)
    cond = house_coarse(h, f, n_types)
    cnt, dom = cond.flat()
    return {"grid": grid, "real": real, "rem": rem, "count": cnt, "dom": dom}


class StateStream(torch.utils.data.IterableDataset):
    def __init__(self, houses: Sequence[House], f: int, n_types: int, bs: int, seed: int = 0):
        self.houses, self.f, self.T, self.bs, self.seed = houses, f, n_types, bs, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0) * 7919)
        while True:
            hs = [self.houses[i] for i in rng.choice(len(self.houses), self.bs, replace=False)]
            st = [make_state(h, assignment(h, self.f, self.T), self.f, self.T, rng) for h in hs]
            yield {k: torch.as_tensor(np.stack([s[k] for s in st]).astype(np.int64 if k != "rem" else np.float32))
                   for k in st[0]}


def seg_lse(x: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    """Per-segment logsumexp of x [M] (segments 0..n-1; empty -> -inf)."""
    mx = torch.full((n,), float("-inf"), device=x.device).scatter_reduce(0, seg, x, "amax")
    mxs = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    s = torch.zeros(n, device=x.device).index_add(0, seg, torch.exp(x - mxs[seg]))
    return torch.log(s) + mxs


def grow_loss(model, batch, device, f):
    grid = batch["grid"].to(device, non_blocking=True)
    real = batch["real"].to(device, non_blocking=True)
    rem = batch["rem"].to(device, non_blocking=True)
    count, dom = batch["count"].to(device), batch["dom"].to(device)
    b, q, legal, sc, seed = legal_scores(model, grid, rem, count, dom, f)
    B = grid.shape[0]
    sc = sc.float()
    rt = real[b, q[:, 0], q[:, 1], q[:, 2]] - 1                        # real type at q, -1 if empty
    neg = torch.finfo(sc.dtype).min / 4
    sl = sc.masked_fill(~legal, neg)
    row_all = torch.logsumexp(sl, -1)
    ok = rt >= 0
    row_pos = torch.where(ok, sc.gather(1, rt.clamp(min=0)[:, None])[:, 0], torch.full_like(row_all, neg))
    lse_all = seg_lse(row_all, b, B)
    lse_pos = seg_lse(row_pos, b, B)
    has = lse_pos > neg / 2
    loss = (lse_all - lse_pos)[has].mean()
    # top-1 accuracy: the best legal action of each house is a correct one
    best = torch.full((B,), float("-inf"), device=device).scatter_reduce(0, b, sl.max(-1).values, "amax")
    hit = torch.zeros(B, device=device).index_add(0, b, (ok & (row_pos >= best[b] - 1e-6)).float())
    acc = (hit[has] > 0).float().mean()
    return {"loss": loss, "acc": acc.detach(), "seed_frac": seed.float().mean().detach()}


def train_grow(train_houses, val_houses, cfg: GrowConfig, run: Path, device, minutes: float = 40,
               bs: int = 16, lr: float = 3e-4, seed: int = 0, workers: int = 6) -> GrowNet:
    torch.manual_seed(seed)
    model = GrowNet(cfg).to(device)
    print(f"[grow] {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    dl = torch.utils.data.DataLoader(StateStream(train_houses, cfg.f, cfg.n_types, bs, seed), batch_size=None,
                                     num_workers=workers, pin_memory=True, prefetch_factor=4,
                                     persistent_workers=True)
    vrng = np.random.default_rng(123)
    vb = []
    for k in range(0, min(len(val_houses), 128), bs):
        st = [make_state(h, assignment(h, cfg.f, cfg.n_types), cfg.f, cfg.n_types, vrng)
              for h in val_houses[k:k + bs]]
        vb.append({kk: torch.as_tensor(np.stack([s[kk] for s in st]).astype(np.int64 if kk != "rem" else np.float32))
                   for kk in st[0]})

    def val():
        model.eval()
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            r = [grow_loss(model, x, device, cfg.f) for x in vb]
        model.train()
        return {"val_loss": float(np.mean([float(x["loss"]) for x in r])),
                "val_acc": float(np.mean([float(x["acc"]) for x in r]))}

    log = open(run / "train_log.jsonl", "w")
    t0, step = time.time(), 0
    for batch in dl:
        frac = (time.time() - t0) / (minutes * 60)
        if frac >= 1:
            break
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 500) * 0.5 * (1 + math.cos(math.pi * frac))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            r = grow_loss(model, batch, device, cfg.f)
        opt.zero_grad(set_to_none=True)
        r["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 100 == 0:
            rec = {"step": step, "min": round((time.time() - t0) / 60, 2),
                   **{k: round(float(v), 4) for k, v in r.items()}}
            if step % 1000 == 0:
                rec.update(val())
                torch.save({"cfg": cfg.to_dict(), "state": model.state_dict(), "step": step}, run / "grow.pt")
            if device.type == "cuda":
                rec["max_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
    rec = {"step": step, "final": val(), "train_min": round((time.time() - t0) / 60, 2)}
    print(json.dumps(rec), flush=True)
    log.write(json.dumps(rec) + "\n"); log.close()
    torch.save({"cfg": cfg.to_dict(), "state": model.state_dict(), "step": step}, run / "grow.pt")
    model.eval()
    return model


def load_grow(path: Path, device) -> GrowNet:
    ck = torch.load(path, map_location=device)
    m = GrowNet(GrowConfig(**ck["cfg"])).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #
@torch.no_grad()
def grow_decode(model, houses: Sequence[House], assigns: Sequence[np.ndarray], conds, device, f: int,
                temperature: float = 0.0, steps: int = 256, seed: int = 0, bs: int = 16,
                policy: str = "model") -> List[dict]:
    """Build every house inside its assignment. Returns per house ``coords`` [n,3],
    ``types`` [n], ``n_seeds``, ``calls``."""
    T = assigns[0].shape[1]
    out = []
    for b0 in range(0, len(houses), bs):
        hs = list(houses[b0:b0 + bs])
        B = len(hs)
        gen = torch.Generator(device=device).manual_seed(seed + b0)
        grid = torch.zeros((B, S, S, S), dtype=torch.long, device=device)
        rem = torch.as_tensor(np.stack(assigns[b0:b0 + B]).astype(np.float32), device=device)
        cnt = torch.as_tensor(np.stack([c.flat()[0] for c in conds[b0:b0 + B]]), device=device)
        dom = torch.as_tensor(np.stack([c.flat()[1] for c in conds[b0:b0 + B]]), device=device)
        K = [max(1, math.ceil(len(h) / steps)) for h in hs]
        placed = [[] for _ in range(B)]
        n_seeds = np.zeros(B, np.int64)
        calls = 0
        fc_all = fine_cell_index(f, device)
        while True:
            active = rem.flatten(1).sum(1) > 0
            if not bool(active.any()):
                break
            ab = active.nonzero()[:, 0]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                b, q, legal, sc, seedf = legal_scores(model if policy == "model" else None, grid[ab], rem[ab],
                                                     cnt[ab], dom[ab], f, reseed=True)
            calls += 1
            if sc is None:
                z = torch.zeros(legal.shape, device=device)
                tmp = 1.0
            else:
                z = sc.float()
                tmp = temperature
            if tmp > 0:
                u = torch.rand(z.shape, device=device, generator=gen).clamp(1e-10, 1 - 1e-10)
                z = z / tmp - torch.log(-torch.log(u))
            z = z.masked_fill(~legal, float("-inf"))
            bc = b.cpu().numpy()
            bounds = np.searchsorted(bc, np.arange(len(ab) + 1))
            upd_b, upd_q, upd_t = [], [], []
            for r, hb in enumerate(ab.tolist()):
                lo, hi = bounds[r], bounds[r + 1]
                if hi <= lo:
                    continue
                k = 1 if bool(seedf[r]) else K[hb]
                n_seeds[hb] += int(bool(seedf[r]))
                zz = z[lo:hi].flatten()
                top = torch.topk(zz, min(len(zz), 4 * k + 8))
                m_ = top.indices // T
                t_ = top.indices % T
                qq = q[lo:hi][m_]
                ci_ = fc_all[qq[:, 0], qq[:, 1], qq[:, 2]]
                left_ = rem[hb, ci_, t_]
                pack = torch.cat([top.values[:, None].float(), qq.float(), t_[:, None].float(), ci_[:, None].float(),
                                  left_[:, None].float()], 1).cpu().numpy()
                used = set()
                remc = {}
                for v, x, y, zc, t, ci, left in pack:
                    if not np.isfinite(v) or len(used) >= k:
                        break
                    cell = (int(x), int(y), int(zc))
                    if cell in used:
                        continue
                    key = (int(ci), int(t))
                    left = remc.get(key, int(left))
                    if left <= 0:
                        continue
                    remc[key] = left - 1
                    used.add(cell)
                    upd_b.append(hb); upd_q.append(cell); upd_t.append(int(t))
                    placed[hb].append((*cell, int(t)))
            if not upd_b:
                break
            ub = torch.as_tensor(upd_b, device=device)
            uq = torch.as_tensor(np.array(upd_q), device=device)
            ut = torch.as_tensor(upd_t, device=device)
            grid[ub, uq[:, 0], uq[:, 1], uq[:, 2]] = ut + 1
            rem.index_put_((ub, fc_all[uq[:, 0], uq[:, 1], uq[:, 2]], ut),
                           torch.full_like(ut, -1, dtype=rem.dtype), accumulate=True)
        for r in range(B):
            P = np.array(placed[r], np.int64).reshape(-1, 4)
            out.append({"coords": P[:, :3], "types": P[:, 3], "n_seeds": int(n_seeds[r]), "calls": calls})
    return out
