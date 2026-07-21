"""Idea #7 A/B: does a semantic block-embedding prior beat a learned table?

Trains the canon-16 flat AR (T18 arm A regime) twice on the IDENTICAL split, config,
and seed — once with the default learned ``nn.Embedding``, once with a
``SemanticTokenEmbedding`` (block rows = projection of frozen CLIP text embeddings of
block names). The hypothesis (ideas.md #7): semantically-near blocks starting near each
other regularizes material choice, so we watch **block-agreement** and **palette**
coherence, plus val-nn/validity for overall quality.

Everything except the embedding is held fixed, so any delta is the prior. Mirrors
``experiments_native.arm_flat`` exactly for comparability.

    python scripts/run_semantic_prior.py --epochs 60
    python scripts/run_semantic_prior.py --epochs 3 --samples 8   # quick signal
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from blockgen.experiments_gen import canonicalize
from blockgen.experiments_overnight import _eval_row
from blockgen.experiments_transfer import _split_houses
from blockgen.models.semantic_embedding import build_semantic_embedding
from blockgen.training.train_ar import ARTrainConfig, build_sequences, sample_structures
from blockgen.training.train_ar_ext import train_from_sequences
from blockgen.utils.augment import augment_with_labels
from blockgen.utils.serialize import build_block_vocab, save_block_vocab


def train_arm(name, fit, ref, val, vocab, seq, cfg, out: Path, semantic):
    run = out / name
    run.mkdir(parents=True, exist_ok=True)
    seqs = build_sequences(fit, vocab, seq)
    t = time.time()
    model, hist = train_from_sequences(seqs, vocab.vocab_size, cfg, pe="phase4",
                                       semantic_embedding=semantic)
    torch.save(model.state_dict(), run / "model.pt")
    samples = sample_structures(model, vocab, num_samples=cfg._samples,
                                temperature=1.0, top_k=40)
    row = _eval_row(name, samples, ref, val, vocab, 16, run)
    row.update(final_loss=round(hist["loss"][-1], 4), n_train=len(seqs),
               train_min=round((time.time() - t) / 60, 1))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--encoder", default="clip")
    ap.add_argument("--stamp", default=None)
    args = ap.parse_args()

    from blockgen.utils.runs import new_run_dir
    out = new_run_dir("semantic_prior", stamp=args.stamp or None)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cmd.txt").write_text(json.dumps(vars(args), indent=2))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # arm_flat prep, verbatim: split -> D4 augment -> canonicalize to 16.
    h_train, h_val = _split_houses(args.seed, args.val_frac)
    h_aug, _ = augment_with_labels(h_train, ["h"] * len(h_train))
    fit = canonicalize(h_aug, 16)
    ref = canonicalize(h_train, 16)
    val = canonicalize(h_val, 16)
    vocab = build_block_vocab(ref + val, max_dim=16)
    save_block_vocab(vocab, str(out / "block_vocab.json"))
    print(f"split {len(h_train)}/{len(h_val)}, D4 -> {len(fit)}, vocab {vocab.vocab_size}",
          flush=True)

    cfg = ARTrainConfig(max_seq_len=args.seq, epochs=args.epochs, batch_size=args.batch,
                        device=device)
    cfg._samples = args.samples  # stash for train_arm

    torch.manual_seed(args.seed)
    baseline = train_arm("baseline_learned", fit, ref, val, vocab, args.seq, cfg, out,
                         semantic=None)
    print("baseline:", json.dumps(baseline), flush=True)

    semantic = build_semantic_embedding(
        vocab, d_model=cfg.d_model, encoder=args.encoder, device=device,
        cache_path=str(out / f"semantic_{args.encoder}.npy"))
    torch.manual_seed(args.seed)
    sem_row = train_arm("semantic_prior", fit, ref, val, vocab, args.seq, cfg, out,
                        semantic=semantic)
    print("semantic:", json.dumps(sem_row), flush=True)

    rows = [baseline, sem_row]
    (out / "rows.json").write_text(json.dumps(rows, indent=2))
    print("\n" + "=" * 74)
    print(f"{'arm':<20}{'val_nn':>9}{'valid':>8}{'blk_agree':>11}{'dup':>7}{'loss':>8}")
    for r in rows:
        print(f"{r['track']:<20}{r.get('val_nn_iou', float('nan')):>9.3f}"
              f"{r['validity_rate']:>8.3f}{r.get('mean_block_agreement', float('nan')):>11.3f}"
              f"{r['duplicate_rate']:>7.3f}{r['final_loss']:>8.3f}")
    print("=" * 74)
    print(f"artifacts -> {out}")


if __name__ == "__main__":
    main()
