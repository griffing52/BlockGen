"""Conditioned AR training on our stack, LegoACE-style (results.md T15).

Backbone: ``CondVoxelAR2`` (T11-winning phase4 PE) over 3D-BPE piece sequences
(T14-winning representation). Conditioning: precomputed frozen embeddings
(``embed_conditions.py``) projected to prefix tokens, 10% cond-dropout to train
the null branch for classifier-free guidance — the LegoACE recipe on our model.

Trains on ALL houses_32 (2,661; sequences capped at --max-seq-len, default 5480
= p95 of piece lengths). Deterministic val split matches the MinecraftACE
exports (same md5(source_path) hash, seed 0).

Usage:
    python -m blockgen.training.train_conditioned --cond image \
        --out outputs/cond/image_run
    python -m blockgen.training.train_conditioned --cond text \
        --out outputs/cond/text_run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from blockgen.curation.houses import load_house_structures
from blockgen.experiments_gen import canonicalize
from blockgen.export.minecraftace import (load_piece_vocab, split_indices,
                                          structure_to_piece_records)
from blockgen.models.voxel_transformer_cond import CondVoxelAR2
from blockgen.utils.serialize import build_block_vocab, structure_to_tokens

PIECE_VOCAB = "data/minecraftace/houses_32_bpe/houses_32_bpe_piece_vocab.json"
EMBEDS = "data/minecraft/labels/houses_32_cond_embeds.npz"

# Token layout (matches serialize.py convention): PAD=0, BOS=1, EOS=2,
# coords 3..3+32, pieces 35..35+678.
PAD, BOS, EOS, NUM_SPECIAL = 0, 1, 2, 3


def build_sequences(structures, cv, max_seq_len):
    """Per-structure token sequence [BOS,(X,Y,Z,PIECE)*,EOS] or None if too long."""
    coord_off = NUM_SPECIAL
    piece_off = NUM_SPECIAL + cv.max_dim
    seqs = []
    for s in structures:
        rec = structure_to_piece_records(s, cv)  # coords 1-based, piece raw
        if len(rec) * 4 + 2 > max_seq_len:
            seqs.append(None)
            continue
        toks = np.empty((len(rec), 4), dtype=np.int64)
        toks[:, :3] = rec[:, :3] - 1 + coord_off
        toks[:, 3] = rec[:, 3] + piece_off
        seqs.append(np.concatenate([[BOS], toks.reshape(-1), [EOS]]))
    return seqs, NUM_SPECIAL + cv.max_dim + cv.n_pieces  # vocab_size


def build_sequences_voxel(structures, vocab, max_seq_len):
    """T12-regime per-voxel sequences via serialize.structure_to_tokens."""
    seqs = []
    for s in structures:
        toks = structure_to_tokens(s, vocab)
        seqs.append(np.asarray(toks, dtype=np.int64)
                    if len(toks) <= max_seq_len else None)
    return seqs, vocab.vocab_size


class CondSeqDataset(torch.utils.data.Dataset):
    """(tokens, cond) pairs; text picks one of 4 caption embeds per fetch."""

    def __init__(self, seqs, embeds, cond_mode, indices):
        self.items = [(i, seqs[i]) for i in indices if seqs[i] is not None]
        self.embeds = embeds
        self.cond_mode = cond_mode

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, seq = self.items[k]
        if self.cond_mode == "image":
            cond = self.embeds[i]                      # (4, 768)
        else:
            j = np.random.randint(self.embeds.shape[1])
            cond = self.embeds[i, j][None]             # (1, 512)
        return torch.from_numpy(seq), torch.from_numpy(cond.copy())


def collate(batch):
    seqs, conds = zip(*batch)
    L = max(len(s) for s in seqs)
    tok = torch.full((len(seqs), L), PAD, dtype=torch.long)
    pad = torch.ones(len(seqs), L, dtype=torch.bool)
    for i, s in enumerate(seqs):
        tok[i, : len(s)] = s
        pad[i, : len(s)] = False
    return tok, pad, torch.stack(conds)


def run_epoch(model, loader, device, optimizer=None, scheduler=None,
              cond_dropout=0.1, use_amp=True):
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for tok, pad, cond in loader:
            tok, pad, cond = tok.to(device), pad.to(device), cond.to(device)
            if training and cond_dropout > 0:
                drop = torch.rand(tok.size(0), device=device) < cond_dropout
                cond = torch.where(drop[:, None, None],
                                   model.null_cond[None].expand_as(cond), cond)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(tok, cond=cond, pad_mask=pad)
                tgt = tok[:, 1:].clone()
                tgt[pad[:, 1:]] = -100
                loss = F.cross_entropy(
                    logits[:, :-1].flatten(0, 1).float(), tgt.flatten(), ignore_index=-100)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total += loss.item()
            n += 1
    return total / max(1, n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cond", choices=["image", "text"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repr", choices=["bpe", "voxel"], default="bpe",
                        help="bpe: 3D-BPE pieces (T14); voxel: per-voxel tokens "
                             "as in the T11/T12 runs")
    parser.add_argument("--canon-dim", type=int, default=None,
                        help="downsample builds to fit this box first "
                             "(T11/T12 used 16)")
    parser.add_argument("--max-seq-len", type=int, default=5480)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--pe", default="phase4")
    parser.add_argument("--cond-dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    os.makedirs(args.out, exist_ok=True)

    structures, manifest = load_house_structures(max_dim=32)
    if args.canon_dim:
        structures = canonicalize(structures, args.canon_dim)
    if args.repr == "voxel":
        vocab = build_block_vocab(structures, max_dim=args.canon_dim or 32)
        seqs, vocab_size = build_sequences_voxel(structures, vocab, args.max_seq_len)
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "block_vocab.json"), "w") as f:
            json.dump({"max_dim": vocab.max_dim,
                       "block_token_to_id": vocab.block_token_to_id,
                       "block_index_to_pair": [list(p) for p in vocab.block_index_to_pair]},
                      f)
    else:
        cv = load_piece_vocab(PIECE_VOCAB)
        seqs, vocab_size = build_sequences(structures, cv, args.max_seq_len)
    kept = sum(s is not None for s in seqs)
    print(f"{kept}/{len(seqs)} structures fit max_seq_len {args.max_seq_len}; "
          f"vocab {vocab_size}", flush=True)

    blob = np.load(EMBEDS)
    embeds = blob["image_embeds" if args.cond == "image" else "text_embeds"]
    cond_dim = embeds.shape[-1]
    n_prefix = 4 if args.cond == "image" else 1

    assignment = split_indices(manifest["items"], seed=0)
    train_idx = [i for i in range(len(seqs)) if assignment[i] != "val"]
    val_idx = [i for i in range(len(seqs)) if assignment[i] == "val"]
    train_ds = CondSeqDataset(seqs, embeds, args.cond, train_idx)
    val_ds = CondSeqDataset(seqs, embeds, args.cond, val_idx)
    print(f"train {len(train_ds)}, val {len(val_ds)}", flush=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = CondVoxelAR2(
        cond_dim=cond_dim, n_prefix=n_prefix, vocab_size=vocab_size,
        max_seq_len=args.max_seq_len, d_model=args.d_model, nhead=8,
        num_layers=args.layers, dim_feedforward=4 * args.d_model,
        pe=args.pe).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CondVoxelAR2 {n_params/1e6:.1f}M cond={args.cond} "
          f"cond_dim={cond_dim} n_prefix={n_prefix}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, s / 200) * 0.5 * (
            1 + math.cos(math.pi * min(1.0, s / total_steps))))

    cfg = vars(args) | {"vocab_size": vocab_size, "cond_dim": cond_dim,
                        "n_prefix": n_prefix, "n_params": n_params}
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    best = float("inf")
    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device, optimizer, scheduler,
                       cond_dropout=args.cond_dropout)
        vl = run_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "train": tr, "val": vl})
        print(f"epoch {epoch}: train {tr:.4f} val_loss {vl:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))
        if vl < best:
            best = vl
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(history, f, indent=1)
    print(f"done. best val {best:.4f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
