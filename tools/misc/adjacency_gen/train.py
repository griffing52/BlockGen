"""Masked edge modelling: train on random pair-mask rates, CE on the masked pairs."""

from __future__ import annotations

import json
import math
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .data import Graph, NodeVocab, ValidTable, edge_labels
from .model import AdjacencyModel, ModelConfig, obs_codes


# --------------------------------------------------------------------------- #
# device: the GPU is shared, take at most ~4 GB
# --------------------------------------------------------------------------- #
def pick_device(max_gb: float = 4.0, need_free_gb: float = 5.0) -> torch.device:
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        if free / 2**30 >= need_free_gb:
            torch.cuda.set_per_process_memory_fraction(min(1.0, max_gb * 2**30 / total))
            return torch.device("cuda")
        print(f"[device] only {free / 2**30:.1f} GB free on GPU, using CPU", flush=True)
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# shape-pair validity table (shared by all graphs of a vocab)
# --------------------------------------------------------------------------- #
class ShapeIndex:
    """type -> shape id; (shape_a, shape_b) -> row of a [U, K+1] validity table."""

    def __init__(self, vocab: NodeVocab):
        self.vocab = vocab
        key_to_sid: Dict[bytes, int] = {}
        self.type_sid = {}
        rep = {}
        for t, sh in vocab.shapes.items():
            k = np.asarray(sh, dtype=np.int64).tobytes()
            sid = key_to_sid.setdefault(k, len(key_to_sid))
            self.type_sid[t] = sid
            rep.setdefault(sid, t)
        self.vt = ValidTable({sid: vocab.shapes[t] for sid, t in rep.items()}, vocab.space)
        self.rows: Dict[tuple, int] = {}
        self.table: List[np.ndarray] = []
        self._gpu = None

    def type_valid(self) -> ValidTable:
        """A ValidTable keyed by node type (for decoding)."""
        idx = self

        class _TV:
            space = self.vocab.space

            def mask(self_, ta, tb):
                return idx.vt.mask(idx.type_sid[int(ta)], idx.type_sid[int(tb)])
        return _TV()

    def pair_rows(self, types: np.ndarray) -> np.ndarray:
        s = np.array([self.type_sid[int(t)] for t in types])
        N = len(s)
        out = np.zeros((N, N), dtype=np.int64)
        for i in range(N):
            for j in range(N):
                key = (s[i], s[j])
                r = self.rows.get(key)
                if r is None:
                    r = self.rows[key] = len(self.table)
                    self.table.append(self.vt.mask(*key))
                    self._gpu = None
                out[i, j] = r
        return out

    def gpu_table(self, device) -> torch.Tensor:
        if self._gpu is None or self._gpu.device != device or len(self._gpu) != len(self.table):
            self._gpu = torch.as_tensor(np.stack(self.table), device=device)
        return self._gpu


def prepare(graphs: Sequence[Graph], vocab: NodeVocab, sidx: ShapeIndex) -> List[dict]:
    out = []
    for g in graphs:
        lab, _ = edge_labels(g.coords, [vocab.shapes[int(t)] for t in g.types], vocab.space)
        out.append({"types": g.types.astype(np.int64), "labels": lab, "rows": sidx.pair_rows(g.types)})
    return out


def prefix_mask(lab: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Growth-state mask: a random connected prefix (BFS from a random root, random
    length) is fully observed, every pair touching the rest is masked."""
    n = len(lab)
    adj = lab > 0
    root = int(rng.integers(n))
    order, seen = [root], {root}
    k = 0
    while k < len(order):
        nb = [int(v) for v in np.nonzero(adj[order[k]])[0] if v not in seen]
        rng.shuffle(nb)
        for v in nb:
            seen.add(v); order.append(v)
        k += 1
    order += [v for v in range(n) if v not in seen]
    m = int(rng.integers(1, n))
    placed = np.zeros(n, bool)
    placed[order[:m]] = True
    return ~(placed[:, None] & placed[None, :])


def collate(items: Sequence[dict], device, mask_rates: Optional[Sequence[float]] = None,
            rng: Optional[np.random.Generator] = None, prefix_frac: float = 0.0):
    """Pad a batch; sample a symmetric pair mask per graph (rate given or random).
    With ``prefix_frac`` > 0 that share of training graphs gets a growth-state mask
    (:func:`prefix_mask`) instead, for the grow decoder."""
    rng = rng or np.random.default_rng()
    Bsz = len(items)
    N = max(len(it["types"]) for it in items)
    types = np.zeros((Bsz, N), np.int64)
    pad = np.ones((Bsz, N), bool)
    labels = np.full((Bsz, N, N), -100, np.int64)
    rows = np.zeros((Bsz, N, N), np.int64)
    masked = np.zeros((Bsz, N, N), bool)
    for b, it in enumerate(items):
        n = len(it["types"])
        types[b, :n] = it["types"]
        pad[b, :n] = False
        labels[b, :n, :n] = it["labels"]
        rows[b, :n, :n] = it["rows"]
        if mask_rates is None and prefix_frac > 0 and rng.random() < prefix_frac:
            pm = prefix_mask(it["labels"], rng)
            np.fill_diagonal(pm, False)
            masked[b, :n, :n] = pm
            continue
        if mask_rates is not None:
            r = mask_rates[b]
        else:
            r = 1.0 if rng.random() < 0.2 else rng.uniform(0.05, 1.0)
        m = np.triu(rng.random((n, n)) < r, 1)
        masked[b, :n, :n] = m | m.T
    t = lambda a: torch.as_tensor(a, device=device)
    return t(types), t(pad), t(labels), t(rows), t(masked)


def forward_batch(model, sidx: ShapeIndex, batch, device):
    types, pad, labels, rows, masked = batch
    valid = sidx.gpu_table(device)[rows]                                   # [B,N,N,K+1]
    observed = ~masked & (labels >= 0)
    obs = obs_codes(labels.clamp(min=0), observed)
    z = model(types, pad, obs, valid)
    return z, labels, masked


def masked_loss(z, labels, masked):
    N = z.shape[1]
    upper = torch.triu(torch.ones(N, N, dtype=torch.bool, device=z.device), 1)
    sel = masked & upper & (labels >= 0)
    return F.cross_entropy(z[sel].float(), labels[sel])


@torch.no_grad()
def masked_metrics(model, sidx, items, device, rate: float, bs: int = 32, seed: int = 0) -> dict:
    """Single-pass prediction of masked pairs at a fixed mask rate."""
    model.eval()
    rng = np.random.default_rng(seed)
    tp = fp = fn = 0
    dir_ok = dir_n = 0
    loss_sum = n_pairs = 0
    scores, ys = [], []
    for k in range(0, len(items), bs):
        chunk = items[k:k + bs]
        batch = collate(chunk, device, [rate] * len(chunk), rng)
        z, labels, masked = forward_batch(model, sidx, batch, device)
        N = z.shape[1]
        upper = torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), 1)
        sel = masked & upper & (labels >= 0)
        zz, yy = z[sel].float(), labels[sel]
        loss_sum += float(F.cross_entropy(zz, yy, reduction="sum"))
        n_pairs += int(sel.sum())
        p = zz.argmax(-1)
        scores.append((1 - zz.softmax(-1)[:, 0]).cpu().numpy()); ys.append((yy > 0).cpu().numpy())
        tp += int(((p > 0) & (yy > 0)).sum()); fp += int(((p > 0) & (yy == 0)).sum())
        fn += int(((p == 0) & (yy > 0)).sum())
        both = (p > 0) & (yy > 0)
        dir_ok += int((p[both] == yy[both]).sum()); dir_n += int(both.sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {"rate": rate, "edge_f1": 2 * prec * rec / max(prec + rec, 1e-9), "edge_prec": prec,
            "edge_rec": rec, "dir_acc": dir_ok / max(dir_n, 1), "nll": loss_sum / max(n_pairs, 1),
            "edge_ap": average_precision(np.concatenate(scores), np.concatenate(ys)),
            "edge_rate": float(np.concatenate(ys).mean())}


def average_precision(score: np.ndarray, y: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-score, kind="stable")
    y = y[o].astype(float)
    prec = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def train(train_items, val_items, vocab: NodeVocab, sidx: ShapeIndex, run: Path, device,
          cfg: ModelConfig, minutes: float = 30.0, bs: int = 32, lr: float = 3e-4,
          max_steps: int = 100000, log_every: int = 200, eval_every: int = 1500,
          free_run_fn=None, seed: int = 0, prefix_frac: float = 0.0) -> AdjacencyModel:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = AdjacencyModel(cfg).to(device)
    print(f"[model] {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    t0 = time.time()
    # LR schedule against the time budget: warmup then cosine on elapsed fraction
    log = open(run / "train_log.jsonl", "w")
    step = 0
    amp = device.type == "cuda"
    while step < max_steps:
        frac = (time.time() - t0) / (minutes * 60)
        if frac >= 1.0:
            break
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 500) * (0.5 * (1 + math.cos(math.pi * frac)))
        idx = rng.choice(len(train_items), bs, replace=False)
        batch = collate([train_items[i] for i in idx], device, None, rng, prefix_frac)
        model.train()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            z, labels, masked = forward_batch(model, sidx, batch, device)
        loss = masked_loss(z, labels, masked)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % log_every == 0:
            rec = {"step": step, "loss": float(loss.detach()), "min": (time.time() - t0) / 60,
                   "lr": opt.param_groups[0]["lr"]}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
        if step % eval_every == 0:
            rec = {"step": step, "val": masked_metrics(model, sidx, val_items[:256], device, 1.0)}
            rec["val_r50"] = masked_metrics(model, sidx, val_items[:256], device, 0.5)
            if free_run_fn is not None:            # T24b rule: one free-running number
                rec["free_run"] = free_run_fn(model)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            torch.save({"cfg": cfg.to_dict(), "state": model.state_dict(), "step": step}, run / "model.pt")
    torch.save({"cfg": cfg.to_dict(), "state": model.state_dict(), "step": step}, run / "model.pt")
    log.close()
    print(f"[train] {step} steps in {(time.time() - t0) / 60:.1f} min", flush=True)
    return model


def load_model(run: Path, device) -> AdjacencyModel:
    ck = torch.load(run / "model.pt", map_location=device)
    m = AdjacencyModel(ModelConfig(**ck["cfg"])).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m


def save_dataset(ds: dict, path: Path):
    with open(path, "wb") as f:
        pickle.dump({k: v for k, v in ds.items() if k != "vocab"} | {"vocab": ds["vocab"].to_json()}, f)


def load_dataset(path: Path) -> dict:
    with open(path, "rb") as f:
        ds = pickle.load(f)
    ds["vocab"] = NodeVocab.from_json(ds["vocab"])
    return ds
