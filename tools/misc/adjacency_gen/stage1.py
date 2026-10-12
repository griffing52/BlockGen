"""Stage 1: cell assignment. Given a coarse condition and an inventory, a distribution over
how many blocks of each type go in each coarse cell.

Model (:class:`CellAssignModel`): a transformer over the occupied coarse cells plus one
inventory token. A cell token is its condition (count, dominant type) and the Fourier
position of its centre; the inventory token is a linear map of ``log1p(count of t)``.
Output per cell: logits over block types, with types absent from the inventory masked.
Loss: cross-entropy of every real block's type under its cell's distribution (the
per-cell type histogram is the target).

Sampling (:func:`cells.sample_assignment`): the per-cell softmax (at a temperature) is
scaled by IPF so that its expected table has the condition counts as row sums and the
inventory as column sums, then integer counts are drawn slot by slot. Every sample
matches both margins exactly, and one (condition, inventory) pair gives many different
assignments.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cells import S, House, assignment, house_coarse, inventory, sample_assignment
from .model_faces import Fourier


@dataclass
class Stage1Config:
    n_types: int
    f: int = 4
    d: int = 192
    layers: int = 4
    heads: int = 6
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)

    @property
    def G(self) -> int:
        return S // self.f


class CellAssignModel(nn.Module):
    def __init__(self, cfg: Stage1Config):
        super().__init__()
        self.cfg = cfg
        d, G = cfg.d, cfg.G
        self.count_lin = nn.Linear(2, d)
        self.dom_emb = nn.Embedding(cfg.n_types + 1, d)
        self.pos = Fourier(S, 4, d)
        self.inv_lin = nn.Linear(cfg.n_types, d)
        self.inv_tok = nn.Parameter(torch.zeros(d))
        c = torch.stack(torch.meshgrid(*[torch.arange(G)] * 3, indexing="ij"), -1).reshape(-1, 3)
        self.register_buffer("centre", (c.float() + 0.5) * cfg.f - 0.5, persistent=False)
        layer = nn.TransformerEncoderLayer(d, cfg.heads, 4 * d, dropout=cfg.dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, cfg.n_types)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.log_alpha = nn.Parameter(torch.tensor(float(np.log(4.0))))

    def forward(self, count, dom, inv):
        """count/dom [B, G^3] long, inv [B, T] (counts). Returns logits [B, G^3, T]
        (types absent from the inventory at -inf; empty cells are ignored by callers)."""
        f3 = self.cfg.f ** 3
        cf = count.float()
        x = (self.count_lin(torch.stack([cf / f3, torch.log1p(cf)], -1)) + self.dom_emb(dom)
             + self.pos(self.centre)[None])
        g = self.inv_lin(torch.log1p(inv.float())) + self.inv_tok
        x = torch.cat([g[:, None], x], 1)
        pad = torch.cat([torch.zeros_like(count[:, :1], dtype=torch.bool), count == 0], 1)
        h = self.ln(self.enc(x, src_key_padding_mask=pad))[:, 1:]
        # residual over the heuristic prior (cells.prior_cell_dist): global inventory
        # frequencies plus a learned bonus alpha for the cell's dominant type
        invf = inv.float()
        prior = (invf / invf.sum(-1, keepdim=True).clamp(min=1))[:, None, :].expand(-1, count.shape[1], -1)
        onehot = F.one_hot(dom.clamp(max=self.cfg.n_types - 1), self.cfg.n_types).float() * (dom < self.cfg.n_types)[..., None]
        prior = prior + self.log_alpha.exp() * onehot
        lg = self.out(h).float() + torch.log(prior.clamp(min=1e-12))
        return lg.masked_fill((inv == 0)[:, None, :], float("-inf"))


def stage1_items(houses: Sequence[House], f: int, n_types: int) -> List[dict]:
    out = []
    for h in houses:
        c = house_coarse(h, f, n_types)
        cnt, dom = c.flat()
        out.append({"count": cnt, "dom": dom, "inv": inventory(h.types, n_types),
                    "A": assignment(h, f, n_types), "coarse": c})
    return out


def _batch(items, device):
    t = lambda k: torch.as_tensor(np.stack([it[k] for it in items]), device=device)
    return t("count"), t("dom"), t("inv")


def stage1_loss(model, items, device):
    count, dom, inv = _batch(items, device)
    lg = model(count, dom, inv)
    A = torch.as_tensor(np.stack([it["A"] for it in items]), device=device).float()
    logp = torch.log_softmax(lg, -1).masked_fill(A == 0, 0.0)
    return -(A * logp).sum() / A.sum()


@torch.no_grad()
def stage1_val(model, items, device, bs=32):
    model.eval()
    tot, n = 0.0, 0
    for k in range(0, len(items), bs):
        ch = items[k:k + bs]
        tot += float(stage1_loss(model, ch, device)) * sum(int(it["A"].sum()) for it in ch)
        n += sum(int(it["A"].sum()) for it in ch)
    model.train()
    return tot / max(n, 1)


def train_stage1(train_items, val_items, cfg: Stage1Config, run: Path, device, minutes: float = 10,
                 bs: int = 32, lr: float = 3e-4, seed: int = 0, wd: float = 0.01,
                 eval_every: int = 1000) -> CellAssignModel:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CellAssignModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    log = open(run / "train_log.jsonl", "w")
    t0, step = time.time(), 0
    best = float("inf")
    while True:
        frac = (time.time() - t0) / (minutes * 60)
        if frac >= 1:
            break
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 300) * 0.5 * (1 + math.cos(math.pi * frac))
        ch = [train_items[i] for i in rng.choice(len(train_items), bs, replace=False)]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = stage1_loss(model, ch, device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % eval_every == 0:
            rec = {"step": step, "min": round((time.time() - t0) / 60, 2), "loss": round(float(loss), 4)}
            if True:
                rec["val_nll"] = round(stage1_val(model, val_items, device), 4)
                if rec["val_nll"] < best:
                    best = rec["val_nll"]
                    torch.save({"cfg": cfg.to_dict(), "state": model.state_dict(), "step": step}, run / "stage1.pt")
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
    v = stage1_val(model, val_items, device)
    if v < best:
        best = v
        torch.save({"cfg": cfg.to_dict(), "state": model.state_dict(), "step": step}, run / "stage1.pt")
    ck = torch.load(run / "stage1.pt", map_location=device)
    model.load_state_dict(ck["state"])
    rec = {"step": step, "final_val_nll": v, "best_val_nll": best, "best_step": ck["step"],
           "train_min": round((time.time() - t0) / 60, 2)}
    print(json.dumps(rec), flush=True)
    log.write(json.dumps(rec) + "\n"); log.close()
    model.eval()
    return model


def load_stage1(path: Path, device) -> CellAssignModel:
    ck = torch.load(path, map_location=device)
    m = CellAssignModel(Stage1Config(**ck["cfg"])).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m


@torch.no_grad()
def stage1_probs(model, items, device, temperature: float = 1.0, bs: int = 32) -> List[np.ndarray]:
    out = []
    for k in range(0, len(items), bs):
        ch = items[k:k + bs]
        lg = model(*_batch(ch, device)) / max(temperature, 1e-6)
        out += list(torch.softmax(lg, -1).cpu().numpy())
    return out


def nll_of(P: np.ndarray, A: np.ndarray) -> float:
    """Per-block NLL (nats) of the true assignment under per-cell distributions P."""
    m = A > 0
    return float(-(A[m] * np.log(np.maximum(P[m], 1e-12))).sum() / max(A.sum(), 1))


def sample_many(P: np.ndarray, it: dict, k: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [sample_assignment(P, it["count"], it["inv"], rng) for _ in range(k)]
