"""§9.0 — the gating run: does native resolution fix unconditioned generation?

Every good uncond number we have (T11 val_nn 0.405, T12 scratch 0.405) is a
**canon-16 miniature**. T17 measured what that regime does to the training target
itself: strided decimation destroys 86.5% of blocks and halves connectivity on the
63% of houses_32 it touches. So the model may be faithfully reproducing rubble, and
"the unconditioned model isn't good" has never been tested against a model that was
allowed to see a house.

The premise for downsampling — sequence budget — is false. 3D-BPE fits 94.8% of
houses_32 natively at max_seq_len 5480 (a length T15 already trained at).

Two arms, same protocol, same epochs:

  A. canon16_flat  — canon-16 + flat tokens + phase4, seq 1600, grid 16.
     HARNESS CHECK: must land near T12 scratch (val_nn 0.405 / baseline 0.48 = 84%).
  B. native_bpe    — native 32 + 3D-BPE + phase4, seq 5480, grid 32. The intervention.

**Compare ratios, not raw val_nn.** nn_iou is grid-dependent, so a grid-32 number is
not commensurable with a grid-16 one. `val_baseline_nn_iou` (how close a real held-out
build is to train, same grid) is the per-arm normalizer.

Protocol mirrors T12 exactly: split houses_32 by source -> D4-augment TRAIN ONLY ->
eval vs distinct real builds, novelty measured against train (dup = memorization) and
against the held-out val split (val_nn = generalization).

Deliberately NOT changed (T15 changed four variables at once and T16 cost a rerun to
unwind it): the cache, batch size, lr, epochs, sampler. Factored embeddings
(notes §9.1) stay off — they are the next ablation, not part of this one.

Not included: adjacency-constrained decoding. `sample_constrained_structures` takes a
BlockVocab and cannot gate ClusterVocab piece tokens; that is a follow-up.

Usage::

    python -m blockgen.experiments_native --quick    # smoke, ~5 min
    python -m blockgen.experiments_native            # full
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from blockgen.eval.novelty import evaluate_novelty
from blockgen.experiments_gen import canonicalize
from blockgen.experiments_overnight import _eval_row, _log
from blockgen.experiments_transfer import _split_houses
from blockgen.tokenizers.cluster_bpe import build_cluster_sequences, learn_clusters
from blockgen.training.train_ar import ARTrainConfig, build_sequences
from blockgen.training.train_ar_ext import (sample_cluster_structures,
                                            train_from_sequences)
from blockgen.training.train_ar_ext import generate_from_prefix  # noqa: F401
from blockgen.export.minecraftace import save_piece_vocab
from blockgen.utils.augment import augment_with_labels
from blockgen.utils.serialize import build_block_vocab, save_block_vocab
from blockgen.utils.data import Structure

OUTPUTS = Path("outputs")
DS = "native"


def _occ(s: Structure) -> int:
    return int((s.block_ids != s.air_block_id).sum())


def _baseline(val_c, ref_c, vocab, grid: int) -> float:
    """How close a REAL held-out build is to train, at this grid. The normalizer."""
    return round(float(np.mean(
        evaluate_novelty(val_c, ref_c, vocab, grid=grid, k=1).nn_iou)), 3)


def arm_flat(fit, ref, val, out: Path, dim: int, seq: int, epochs: int,
             samples: int, batch: int) -> Optional[dict]:
    """A: canon-16 + flat + phase4 — reproduces the T11/T12 baseline regime."""
    from blockgen.training.train_ar import sample_structures  # AR2 has .generate too
    run = out / "canon16_flat"; run.mkdir(parents=True, exist_ok=True)
    vocab = build_block_vocab(ref + val, max_dim=dim)
    # Save the vocab BEFORE training: the checkpoint is unloadable without it, and
    # saving it up front means a crashed or killed run still leaves a usable one.
    save_block_vocab(vocab, str(run / "block_vocab.json"))
    seqs = build_sequences(fit, vocab, seq)
    _log(DS, f"[A] canon{dim}+flat: {len(seqs)}/{len(fit)} fit, vocab={vocab.vocab_size}, "
             f"median_len={int(np.median([len(s) for s in seqs]))}")
    cfg = ARTrainConfig(max_seq_len=seq, epochs=epochs, batch_size=batch, device="cuda")
    t = time.time()
    model, hist = train_from_sequences(seqs, vocab.vocab_size, cfg, pe="phase4")
    torch.save(model.state_dict(), run / "model.pt")
    s = sample_structures(model, vocab, num_samples=samples, temperature=1.0, top_k=40)
    row = _eval_row("canon16_flat", s, ref, val, vocab, dim, run)
    row.update(final_loss=round(hist["loss"][-1], 4), n_train=len(seqs),
               train_min=round((time.time() - t) / 60, 1),
               val_baseline_nn_iou=_baseline(val, ref, vocab, dim))
    return row


def arm_bpe(fit, ref, val, out: Path, dim: int, seq: int, epochs: int,
            samples: int, batch: int, n_merges: int) -> Optional[dict]:
    """B: native 32 + 3D-BPE + phase4 — the intervention."""
    run = out / "native_bpe"; run.mkdir(parents=True, exist_ok=True)
    vocab = build_block_vocab(ref + val, max_dim=dim)   # novelty grid only
    cv = learn_clusters(fit, max_dim=dim, n_merges=n_merges, max_corpus=400)
    # Save the piece vocab BEFORE training. A piece token id means nothing without the
    # patterns it expands to, so the checkpoint is dead weight on its own; cluster_meta
    # below records only counts. Re-deriving it costs a full replay of the seeded
    # pipeline (notes.md §18). ~7 min of BPE learning vs 3.5 h of training -- save it.
    save_piece_vocab(cv, str(run / "piece_vocab.json"))
    seqs, kept = build_cluster_sequences(fit, cv, seq)
    _log(DS, f"[B] native{dim}+bpe: {len(kept)}/{len(fit)} fit, pieces={cv.n_pieces}, "
             f"vocab={cv.vocab_size}, median_len="
             f"{int(np.median([len(s) for s in seqs])) if seqs else 0}")
    if len(seqs) < 16:
        (run / "SKIP.txt").write_text("too few fit"); return None
    cfg = ARTrainConfig(max_seq_len=seq, epochs=epochs, batch_size=batch, device="cuda")
    t = time.time()
    model, hist = train_from_sequences(seqs, cv.vocab_size, cfg, pe="phase4")
    torch.save(model.state_dict(), run / "model.pt")
    (run / "cluster_meta.json").write_text(json.dumps(
        {"n_pieces": cv.n_pieces, "num_blocks": cv.num_blocks,
         "n_merges": len(cv.merges), "n_fit": len(kept),
         "median_seq_len": int(np.median([len(s) for s in seqs]))}, indent=2))
    s = sample_cluster_structures(model, cv, num_samples=samples, temperature=1.0, top_k=40)
    row = _eval_row("native_bpe", s, ref, val, vocab, dim, run)
    row.update(final_loss=round(hist["loss"][-1], 4), n_train=len(seqs),
               train_min=round((time.time() - t) / 60, 1),
               val_baseline_nn_iou=_baseline(val, ref, vocab, dim))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--merges", type=int, default=256)
    ap.add_argument("--seq-flat", type=int, default=1600)
    ap.add_argument("--seq-bpe", type=int, default=5480)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--stamp", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--arms", nargs="+", default=["flat", "bpe"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.samples, args.merges = 2, 4, 24

    torch.manual_seed(args.seed)
    out = OUTPUTS / f"run_{args.stamp}_native"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cmd.txt").write_text(json.dumps(vars(args), indent=2))

    h_train, h_val = _split_houses(args.seed, args.val_frac)
    _log(DS, f"houses_32 split: {len(h_train)} train / {len(h_val)} val")
    h_aug, _ = augment_with_labels(h_train, ["h"] * len(h_train))
    _log(DS, f"D4-augmented train: {len(h_train)} -> {len(h_aug)}")

    rows: List[dict] = []
    if "flat" in args.arms:
        try:
            r = arm_flat(canonicalize(h_aug, 16), canonicalize(h_train, 16),
                         canonicalize(h_val, 16), out, 16, args.seq_flat,
                         args.epochs, args.samples, args.batch)
            if r: rows.append(r)
        except Exception:  # noqa: BLE001
            (out / "flat_ERROR.txt").write_text(traceback.format_exc())
            _log(DS, "[A] FAILED")
    if "bpe" in args.arms:
        try:
            # canonicalize(_, 32) on houses_32 crops but never downsamples: native.
            r = arm_bpe(canonicalize(h_aug, 32), canonicalize(h_train, 32),
                        canonicalize(h_val, 32), out, 32, args.seq_bpe,
                        args.epochs, args.samples, args.batch, args.merges)
            if r: rows.append(r)
        except Exception:  # noqa: BLE001
            (out / "bpe_ERROR.txt").write_text(traceback.format_exc())
            _log(DS, "[B] FAILED")

    (out / "rows.json").write_text(json.dumps(rows, indent=2))
    print("\n" + "=" * 78)
    print(f"{'arm':<16}{'val_nn':>9}{'baseline':>10}{'ratio':>8}{'valid':>8}"
          f"{'v/train':>9}{'dup':>7}{'occ':>8}")
    for r in rows:
        vb = r.get("val_baseline_nn_iou") or float("nan")
        vn = r.get("val_nn_iou") or float("nan")
        print(f"{r['track']:<16}{vn:>9.3f}{vb:>10.3f}{vn/vb:>8.1%}"
              f"{r['validity_rate']:>8.3f}{r.get('validity_vs_train', float('nan')):>9.3f}"
              f"{r['duplicate_rate']:>7.3f}{r['median_sample_occ']:>8.0f}")
    print("=" * 78)
    print("T12 scratch reference: val_nn 0.405 / baseline 0.48 = 84.4% (canon-16, flat)")
    print(f"artifacts -> {out}")


if __name__ == "__main__":
    main()
