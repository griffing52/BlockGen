"""Teacher-forced training of the face model over random attachment orders.

Each training example is a real chunk in a random *growth state*: a prefix of a random
valid attachment order (:func:`faces.random_attachment_order`) is placed at its real
positions, the rest is unplaced. With probability ``p0`` the prefix is empty.

Losses (summed):

* pointer (prefix size >= 1): softmax over every attachment ``(d, p, j)`` with p placed
  and j unplaced (including faces that point into occupied or out-of-window cells: those
  are always wrong, and the check only removes them at decode time, so the no-check
  decoder is a fair test of what the model learned). The target is the *set* of correct
  attachments (:func:`faces.correct_actions`): ``-log sum_{C} p``, i.e. the loss is
  marginalized over every face that points at a correct cell and every unplaced block of
  the right type (T24e's cell-marginalized loss, plus type exchangeability).
* seed (empty prefix): softmax over (block, fine cell), target = every (j, c) whose real
  cell c holds a block of j's type.
* glue: per face, a softmax over the complementary faces of the other blocks plus a dust
  bin scored by the face's exposed logit (rows and columns of each axis matrix).
* exposed: BCE on the faces of unplaced blocks.
"""

from __future__ import annotations

import json
import math
import pickle
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .coarse import coarse_cell, graph_coarse
from .data import Graph, NodeVocab
from .faces import (attach_targets, buried_state, exposed_faces, glue_matrices,
                    random_attachment_order)
from .model_faces import FaceConfig, FaceModel


def prepare_faces(graphs: Sequence[Graph], vocab: NodeVocab, f: int, size: int = 8,
                  oracle_f: int = 2) -> List[dict]:
    """Per chunk: types, coords, glue [3,N,N], exposed [N,6], fine cell types [S^3], the
    coarse condition at factor f (f = 0: the f = 2 condition is still stored for eval) and
    the oracle coarse-cell index (at ``oracle_f``)."""
    out = []
    fc = f or 2
    for g in graphs:
        c = graph_coarse(g, fc, vocab, size)
        ct = -np.ones(size ** 3, np.int64)
        flat = g.coords[:, 0] * size * size + g.coords[:, 1] * size + g.coords[:, 2]
        ct[flat] = g.types
        cnt, dom = c.flat()
        out.append({"types": g.types.astype(np.int64), "coords": g.coords.astype(np.int64),
                    "glue": glue_matrices(g.coords), "exposed": exposed_faces(g.coords),
                    "cell_type": ct, "cond_count": cnt, "cond_dom": dom, "coarse": c,
                    "ocell": coarse_cell(g.coords, oracle_f, size // oracle_f)})
    return out


def collate_states(items: Sequence[dict], placed_list: Sequence[np.ndarray],
                   pos_list: Optional[Sequence[np.ndarray]], device, cfg: FaceConfig,
                   fstate_list: Optional[Sequence[np.ndarray]] = None) -> dict:
    """Pad a batch of (chunk, state) pairs into model inputs. ``pos_list`` defaults to the
    real coordinates (teacher forcing)."""
    Bsz = len(items)
    N = max(len(it["types"]) for it in items)
    types = np.zeros((Bsz, N), np.int64)
    pad = np.ones((Bsz, N), bool)
    placed = np.zeros((Bsz, N), bool)
    pos = np.zeros((Bsz, N, 3), np.int64)
    fst = np.zeros((Bsz, N, 6), np.int64)
    ocell = np.zeros((Bsz, N), np.int64)
    for b, it in enumerate(items):
        n = len(it["types"])
        types[b, :n] = it["types"]
        pad[b, :n] = False
        placed[b, :n] = placed_list[b]
        p = it["coords"] if pos_list is None else pos_list[b]
        pos[b, :n] = np.where(placed_list[b][:, None], p, 0)
        fst[b, :n] = buried_state(p, placed_list[b]) if fstate_list is None else fstate_list[b]
        ocell[b, :n] = it["ocell"]
    t = lambda a: torch.as_tensor(a, device=device)
    batch = {"types": t(types), "pad": t(pad), "placed": t(placed), "pos": t(pos), "fstate": t(fst)}
    if cfg.f:
        batch["cond_count"] = t(np.stack([it["cond_count"] for it in items]))
        batch["cond_dom"] = t(np.stack([it["cond_dom"] for it in items]))
    if cfg.oracle:
        batch["ocell"] = t(ocell)
    if cfg.f and cfg.residual:
        G = cfg.G
        cp = np.zeros((Bsz, G ** 3), np.int64)
        for b, it in enumerate(items):
            p = it["coords"] if pos_list is None else pos_list[b]
            q = p[placed_list[b]]
            q = np.unique(q[np.all((q >= 0) & (q < cfg.size), 1)], axis=0)   # colliding blocks count once
            q = q // cfg.f
            np.add.at(cp[b], q[:, 0] * G * G + q[:, 1] * G + q[:, 2], 1)
        batch["cond_placed"] = t(cp)
    return batch


def sample_states(items: Sequence[dict], rng: np.random.Generator, p0: float = 0.2):
    placed_list, tgts = [], []
    for it in items:
        n = len(it["types"])
        order = random_attachment_order(it["coords"], rng)
        k = 0 if rng.random() < p0 else int(rng.integers(1, n))
        pl = np.zeros(n, bool)
        pl[order[:k]] = True
        placed_list.append(pl)
        tgts.append(attach_targets(it["coords"], it["types"], pl))
    return placed_list, tgts


def losses(model: FaceModel, items, placed_list, tgts, device, cfg: FaceConfig) -> dict:
    batch = collate_states(items, placed_list, None, device, cfg)
    out = model(**batch)
    X, ex, seed = out["X"].float(), out["ex"].float(), out["seed"].float()
    Bsz, _, N, _ = X.shape
    types, pad, placed = batch["types"], batch["pad"], batch["placed"]
    valid = ~pad
    unpl = valid & ~placed
    eye = torch.eye(N, dtype=torch.bool, device=device)
    # targets padded
    G = np.zeros((Bsz, 3, N, N), bool)
    EX = np.zeros((Bsz, N, 6), bool)
    T = -np.ones((Bsz, 6, N), np.int64)
    CT = np.stack([it["cell_type"] for it in items])
    for b, it in enumerate(items):
        n = len(it["types"])
        G[b, :, :n, :n] = it["glue"]
        EX[b, :n] = it["exposed"]
        T[b, :, :n] = tgts[b]
    G, EX, T, CT = (torch.as_tensor(a, device=device) for a in (G, EX, T, CT))
    res = {}
    # pointer
    from .faces import action_scores
    A = action_scores(X)                                                   # [B,6,N,N]
    legal = (placed[:, None, :, None] & unpl[:, None, None, :]).expand_as(A)
    corr = legal & (T[:, :, :, None] == types[:, None, None, :])
    has = corr.flatten(1).any(1)
    if has.any():
        lse_all = A.masked_fill(~legal, -1e9).flatten(1).logsumexp(1)
        lse_c = A.masked_fill(~corr, -1e9).flatten(1).logsumexp(1)
        res["ptr"] = (lse_all - lse_c)[has].mean()
        am = A.masked_fill(~legal, -1e9).flatten(1).argmax(1)
        res["ptr_acc"] = corr.flatten(1).gather(1, am[:, None]).squeeze(1)[has].float().mean().detach()
    # seed
    k0 = (placed.sum(1) == 0)
    if k0.any():
        sc = seed[k0]                                                      # [b,N,S^3]
        sv = valid[k0][:, :, None].expand_as(sc)
        sc_c = sv & (types[k0][:, :, None] == CT[k0][:, None, :])
        lse_all = sc.masked_fill(~sv, -1e9).flatten(1).logsumexp(1)
        lse_c = sc.masked_fill(~sc_c, -1e9).flatten(1).logsumexp(1)
        res["seed"] = (lse_all - lse_c).mean()
    # glue: dual softmax with a dust bin (SuperGlue-style). Each +a face picks one -a face
    # or the bin (exposed), and each -a face picks one +a face or the bin; the bin score is
    # the face's exposed logit, which is what the one-shot Sinkhorn decoder uses.
    pv = (valid[:, :, None] & valid[:, None, :] & ~eye)[:, None].expand_as(X)
    Xm = X.masked_fill(~pv, -1e9)
    ex_p = ex[:, :, 0::2].permute(0, 2, 1)                                 # [B,3,N]
    ex_m = ex[:, :, 1::2].permute(0, 2, 1)
    vr = valid[:, None, :].expand(Bsz, 3, N)
    t_r = torch.where(G.any(-1), G.float().argmax(-1), torch.full_like(G[..., 0], N, dtype=torch.long))
    t_c = torch.where(G.any(-2), G.float().argmax(-2), torch.full_like(G[..., 0], N, dtype=torch.long))
    lr_ = torch.cat([Xm, ex_p[..., None]], -1)
    lc_ = torch.cat([Xm.transpose(-1, -2), ex_m[..., None]], -1)
    res["glue"] = 0.5 * (F.cross_entropy(lr_[vr], t_r[vr]) + F.cross_entropy(lc_[vr], t_c[vr]))
    em = unpl[:, :, None].expand_as(ex)
    if em.any():
        res["exposed"] = F.binary_cross_entropy_with_logits(ex[em], EX[em].float())
    res["loss"] = sum(v for k, v in res.items() if k in ("ptr", "seed", "glue", "exposed"))
    return res


def _step_losses(model, chunk, rng, device, cfg):
    if cfg.merge:
        from .merge import merge_losses
        return merge_losses(model, chunk, rng, device, cfg)
    pl, tg = sample_states(chunk, rng)
    return losses(model, chunk, pl, tg, device, cfg)


@torch.no_grad()
def val_metrics(model, items, device, cfg, n_batches: int = 8, bs: int = 16, seed: int = 0) -> dict:
    model.eval()
    rng = np.random.default_rng(seed)
    acc = {}
    for k in range(n_batches):
        chunk = items[k * bs:(k + 1) * bs]
        if not chunk:
            break
        r = _step_losses(model, chunk, rng, device, cfg)
        for kk, v in r.items():
            acc.setdefault(kk, []).append(float(v))
    model.train()
    return {k: float(np.mean(v)) for k, v in acc.items()}


def train_faces(train_items, val_items, cfg: FaceConfig, run: Path, device, minutes: float = 30,
                bs: int = 16, lr: float = 3e-4, log_every: int = 200, eval_every: int = 1500,
                seed: int = 0) -> FaceModel:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = FaceModel(cfg).to(device)
    print(f"[model] {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    amp = device.type == "cuda"
    log = open(run / "train_log.jsonl", "w")
    t0 = time.time()
    step = 0
    while True:
        frac = (time.time() - t0) / (minutes * 60)
        if frac >= 1.0:
            break
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 500) * (0.5 * (1 + math.cos(math.pi * frac)))
        idx = rng.choice(len(train_items), bs, replace=False)
        chunk = [train_items[i] for i in idx]
        model.train()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            r = _step_losses(model, chunk, rng, device, cfg)
        opt.zero_grad(set_to_none=True)
        r["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % log_every == 0:
            rec = {"step": step, "min": round((time.time() - t0) / 60, 2), "lr": opt.param_groups[0]["lr"],
                   **{k: round(float(v.detach()), 4) for k, v in r.items()}}
            if device.type == "cuda":
                rec["max_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
        if step % eval_every == 0:
            rec = {"step": step, "val": val_metrics(model, val_items, device, cfg, bs=min(16, bs),
                                                     n_batches=max(8, 128 // min(16, bs)))}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            torch.save({"cfg": cfg.to_dict(), "state": model.state_dict(), "step": step}, run / "model.pt")
    rec = {"step": step, "final_val": val_metrics(model, val_items, device, cfg, bs=min(16, bs),
                                                   n_batches=max(16, 256 // min(16, bs)))}
    print(json.dumps(rec), flush=True)
    log.write(json.dumps(rec) + "\n")
    log.close()
    torch.save({"cfg": cfg.to_dict(), "state": model.state_dict(), "step": step}, run / "model.pt")
    print(f"[train] {step} steps in {(time.time() - t0) / 60:.1f} min", flush=True)
    model.eval()
    return model


def load_face_model(run: Path, device) -> FaceModel:
    ck = torch.load(run / "model.pt", map_location=device)
    m = FaceModel(FaceConfig(**ck["cfg"])).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m


def save_window_dataset(ds: dict, path: Path):
    with open(path, "wb") as fh:
        pickle.dump({k: v for k, v in ds.items() if k != "vocab"} | {"vocab": ds["vocab"].to_json()}, fh)


def load_window_dataset(path: Path) -> dict:
    with open(path, "rb") as fh:
        ds = pickle.load(fh)
    ds["vocab"] = NodeVocab.from_json(ds["vocab"])
    return ds
