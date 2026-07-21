"""Idea #6: train the text-conditioned piece AR with a resampled conditioning channel.

``ResampledCondVoxelAR2`` replaces ``CondVoxelAR2``'s single pooled-CLIP prefix with a
Q-Former-lite resampler that cross-attends the *full* CLIP text token sequence (idea #6,
ideas.md). Uses the sequence-level embeddings from ``embed_conditions`` (``--no-text-
sequence`` off): ``text_token_embeds`` [N,4,L,512] + ``text_token_mask`` [N,4,L].

Everything else mirrors ``train_conditioned.py`` (same piece vocab, split, cond-dropout
+ CFG, cosine schedule) so the two are comparable — the only change is the conditioning
front-end. Per project convention, it renders sample builds into the run folder.

    python scripts/train_cond_resampler.py --out outputs/cond/text_resampler \
        --epochs 40 --n-prefix 16
    python scripts/train_cond_resampler.py --out outputs/cond/text_resampler_smoke \
        --epochs 2 --n-prefix 8 --limit 128 --num-samples 8
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

from blockgen.curation.houses import load_house_structures
from blockgen.eval.cond_render import render_cond_run
from blockgen.export.minecraftace import load_piece_vocab, split_indices
from blockgen.models.condition_resampler import ResampledCondVoxelAR2
from blockgen.training.train_conditioned import BOS, EOS, PAD, PIECE_VOCAB, build_sequences

EMBEDS_SEQ = "data/minecraft/labels/houses_32_cond_embeds_seq.npz"


class SeqCondDataset(torch.utils.data.Dataset):
    """(piece tokens, one caption's token-embedding sequence) pairs."""

    def __init__(self, seqs, tok_embeds, tok_mask, indices):
        self.items = [i for i in indices if seqs[i] is not None]
        self.seqs, self.emb, self.mask = seqs, tok_embeds, tok_mask

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i = self.items[k]
        n_caps = int((self.mask[i].sum(-1) > 0).sum())  # captions that exist
        j = np.random.randint(max(1, n_caps))
        length = int(self.mask[i, j].sum())
        cond = self.emb[i, j, :length].astype(np.float32)         # (length, 512)
        return torch.from_numpy(self.seqs[i]), torch.from_numpy(cond)


def collate(batch):
    seqs, conds = zip(*batch)
    L = max(len(s) for s in seqs)
    tok = torch.full((len(seqs), L), PAD, dtype=torch.long)
    pad = torch.ones(len(seqs), L, dtype=torch.bool)
    for i, s in enumerate(seqs):
        tok[i, : len(s)] = s
        pad[i, : len(s)] = False
    S = max(c.shape[0] for c in conds)
    cond = torch.zeros(len(conds), S, conds[0].shape[1])
    cmask = torch.ones(len(conds), S, dtype=torch.bool)           # True = pad
    for i, c in enumerate(conds):
        cond[i, : c.shape[0]] = c
        cmask[i, : c.shape[0]] = False
    return tok, pad, cond, cmask


def run_epoch(model, loader, device, optimizer=None, scheduler=None, cond_dropout=0.1):
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    with (torch.enable_grad() if training else torch.no_grad()):
        for tok, pad, cond, cmask in loader:
            tok, pad = tok.to(device), pad.to(device)
            cond, cmask = cond.to(device), cmask.to(device)
            drop = (torch.rand(tok.size(0), device=device) < cond_dropout) \
                if training and cond_dropout > 0 else None
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(tok, cond=cond, pad_mask=pad, cond_mask=cmask,
                               cond_drop=drop)
                tgt = tok[:, 1:].clone()
                tgt[pad[:, 1:]] = -100
                loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(),
                                       tgt.flatten(), ignore_index=-100)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total += loss.item(); n += 1
    return total / max(1, n)


@torch.no_grad()
def sample_and_render(model, cv, val_ds, captions, out, device, num=16, cfg=3.0,
                      max_new=2500):
    model.eval()
    # Standard: textured render titled with the input prompt (blockgen.eval.cond_render).
    picks = [i for i in val_ds.items
             if captions.get(f"h{i:05d}") and int(val_ds.mask[i, 0].sum()) > 0][:num]
    if not picks:
        print("WARNING: no val conditions with captions to sample from; skipping render",
              flush=True)
        return
    render_cond_run(model, cv, picks, val_ds.emb, val_ds.mask, captions,
                    os.path.join(out, "samples.png"), cfg_scale=cfg, caption_idx=0,
                    device=device, max_new=max_new,
                    suptitle="text_resampler (val conds) — textured, prompt -> output")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="cond_text_resampler",
                    help="experiment name -> outputs/run_<timestamp>_<name>")
    ap.add_argument("--out", default=None,
                    help="explicit output dir (overrides the run_<timestamp>_<name> path)")
    ap.add_argument("--embeds", default=EMBEDS_SEQ)
    ap.add_argument("--cache", default=None,
                    help="structure cache .npz (default: houses_32); e.g. the corpus "
                         "data/minecraft/cache/all_32.npz")
    ap.add_argument("--piece-vocab", default=PIECE_VOCAB,
                    help="ClusterVocab json matching --cache (corpus vocab for all_32)")
    ap.add_argument("--captions", default="data/minecraft/labels/all_32_captions.json",
                    help="captions json {h#####:[...]} for the textured prompt->output grid")
    ap.add_argument("--max-seq-len", type=int, default=5480)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--n-prefix", type=int, default=16, help="resampler query tokens")
    ap.add_argument("--resampler-layers", type=int, default=2)
    ap.add_argument("--cond-dropout", type=float, default=0.1)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="cap #structures (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from blockgen.utils.runs import new_run_dir
    out = args.out or str(new_run_dir(args.name))
    os.makedirs(out, exist_ok=True)
    args.out = out
    print(f"run dir: {out}", flush=True)

    if args.cache:
        from blockgen.curation.houses import load_structures_from_cache
        structures, manifest = load_structures_from_cache(args.cache)
    else:
        structures, manifest = load_house_structures(max_dim=32)
    cv = load_piece_vocab(args.piece_vocab)
    seqs, vocab_size = build_sequences(structures, cv, args.max_seq_len)
    blob = np.load(args.embeds)
    tok_embeds, tok_mask = blob["text_token_embeds"], blob["text_token_mask"]

    assignment = split_indices(manifest["items"], seed=0)
    train_idx = [i for i in range(len(seqs)) if assignment[i] != "val"]
    val_idx = [i for i in range(len(seqs)) if assignment[i] == "val"]
    if args.limit:  # smoke: keep the pipeline honest but small
        train_idx, val_idx = train_idx[:args.limit], val_idx[:max(8, args.limit // 8)]
    train_ds = SeqCondDataset(seqs, tok_embeds, tok_mask, train_idx)
    val_ds = SeqCondDataset(seqs, tok_embeds, tok_mask, val_idx)
    print(f"train {len(train_ds)}, val {len(val_ds)}, vocab {vocab_size}, "
          f"cond_dim {tok_embeds.shape[-1]}", flush=True)

    tl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                     collate_fn=collate, drop_last=True)
    vl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                     collate_fn=collate)

    model = ResampledCondVoxelAR2(
        cond_dim=tok_embeds.shape[-1], n_prefix=args.n_prefix, vocab_size=vocab_size,
        max_seq_len=args.max_seq_len, d_model=args.d_model, nhead=8,
        num_layers=args.layers, dim_feedforward=4 * args.d_model, pe="phase4",
        resampler_layers=args.resampler_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ResampledCondVoxelAR2 {n_params/1e6:.1f}M n_prefix={args.n_prefix}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, len(tl) * args.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))

    cfg = vars(args) | {"vocab_size": vocab_size, "n_params": n_params,
                        "cond_dim": int(tok_embeds.shape[-1])}
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)

    best, history = float("inf"), []
    for epoch in range(args.epochs):
        t0 = time.time()
        tr = run_epoch(model, tl, device, opt, sched, cond_dropout=args.cond_dropout)
        va = run_epoch(model, vl, device) if len(val_ds) else float("nan")
        history.append({"epoch": epoch, "train": tr, "val": va})
        print(f"epoch {epoch}: train {tr:.4f} val {va:.4f} ({time.time()-t0:.0f}s)",
              flush=True)
        torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))
        if va < best:
            best = va
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
        json.dump(history, open(os.path.join(args.out, "history.json"), "w"), indent=1)

    # Render samples into the run folder (project convention: textured + prompts).
    captions = json.load(open(args.captions)) if os.path.exists(args.captions) else {}
    sample_and_render(model, cv, val_ds, captions, args.out, device,
                      num=args.num_samples, cfg=args.cfg,
                      max_new=min(2500, args.max_seq_len - 1))
    print(f"done. best val {best:.4f}; samples -> {args.out}/samples.png", flush=True)


if __name__ == "__main__":
    main()
