"""Recover the vocabularies ``experiments_native`` trained against but never saved.

Neither arm of that experiment saves its vocabulary: ``arm_bpe`` writes ``model.pt``
and a ``cluster_meta.json`` of *counts* but not the ``ClusterVocab``, and ``arm_flat``
builds its ``BlockVocab`` inline and drops it. A token id means nothing without the
vocabulary that assigns it meaning, so neither checkpoint is loadable on its own.

The vocabulary is recoverable because every input to it is seeded and
deterministic: ``_split_houses(seed, val_frac)`` uses a seeded permutation,
``augment_with_labels`` enumerates the full D4 orbit (no sampling), and
``learn_clusters`` seeds its own corpus subsample and breaks merge ties by
insertion order. Re-running that exact prefix reproduces the vocab bit-for-bit.

This script re-derives them, checks each against the checkpoint's actual ``lm_head``
shape, and writes the vocab next to its checkpoint so the server can load it.

The size checks are necessary but not sufficient: for the BPE arm ``n_pieces`` is
structurally ``num_blocks + n_merges``, so it would match even if the *patterns*
diverged. The real proof is downstream sample quality — compare against the run's
``rows.json`` (native_bpe: median_sample_occ ~942, mean_nn_iou ~0.329). A wrong
vocab decodes to noise, so eyeball a sample before trusting a rebuilt run.

Usage:
    python scripts/rebuild_native_vocab.py --run outputs/run_20260715_065938_native
    python scripts/rebuild_native_vocab.py --run ... --arm canon16_flat
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from blockgen.experiments_gen import canonicalize
from blockgen.experiments_transfer import _split_houses
from blockgen.export.minecraftace import save_piece_vocab
from blockgen.tokenizers.cluster_bpe import learn_clusters
from blockgen.utils.augment import augment_with_labels
from blockgen.utils.serialize import build_block_vocab, save_block_vocab


def rebuild_bpe(run: Path, arm: str, cmd: dict, out: Path) -> None:
    """Mirror experiments_native.main() -> arm_bpe exactly."""
    meta = json.loads((run / arm / "cluster_meta.json").read_text())
    h_train, h_val = _split_houses(cmd["seed"], cmd["val_frac"])
    print(f"split: {len(h_train)} train / {len(h_val)} val")
    h_aug, _ = augment_with_labels(h_train, ["h"] * len(h_train))
    print(f"D4-augmented: {len(h_train)} -> {len(h_aug)}")
    cv = learn_clusters(canonicalize(h_aug, 32), max_dim=32,
                        n_merges=cmd["merges"], max_corpus=400)

    problems = []
    if cv.n_pieces != meta["n_pieces"]:
        problems.append(f"n_pieces {cv.n_pieces} != recorded {meta['n_pieces']}")
    if cv.num_blocks != meta["num_blocks"]:
        problems.append(f"num_blocks {cv.num_blocks} != recorded {meta['num_blocks']}")
    _finish(run, arm, cv.vocab_size, problems,
            f"n_pieces={cv.n_pieces} num_blocks={cv.num_blocks}")
    save_piece_vocab(cv, str(out))
    print(f"-> {out}")


def rebuild_flat(run: Path, arm: str, cmd: dict, out: Path) -> None:
    """Mirror experiments_native.main() -> arm_flat exactly."""
    h_train, h_val = _split_houses(cmd["seed"], cmd["val_frac"])
    # arm_flat builds the vocab from ref + val = canonicalize(h_train,16) + (h_val,16).
    vocab = build_block_vocab(canonicalize(h_train, 16) + canonicalize(h_val, 16),
                              max_dim=16)
    _finish(run, arm, vocab.vocab_size, [], f"num_blocks={vocab.num_blocks}")
    save_block_vocab(vocab, str(out))
    print(f"-> {out}")


def _finish(run: Path, arm: str, vocab_size: int, problems: list, summary: str) -> None:
    """Cross-check the rebuilt vocab against the checkpoint before writing it."""
    sd = torch.load(run / arm / "model.pt", map_location="cpu", weights_only=True)
    lm = sd["lm_head.weight"].shape[0]
    if lm != vocab_size:
        problems.append(f"checkpoint lm_head={lm} != rebuilt vocab_size={vocab_size}")
    print(f"rebuilt: {summary} vocab_size={vocab_size} (checkpoint lm_head={lm})")
    if problems:
        for p in problems:
            print(f"  ! {p}")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="outputs/run_..._native")
    ap.add_argument("--arm", default="native_bpe", choices=["native_bpe", "canon16_flat"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    cmd = json.loads((run / "cmd.txt").read_text())
    print(f"run={run} arm={args.arm} seed={cmd['seed']} val_frac={cmd['val_frac']}")

    if args.arm == "native_bpe":
        out = Path(args.out) if args.out else run / args.arm / "piece_vocab.json"
        rebuild_bpe(run, args.arm, cmd, out)
        print("NOTE: sizes match, which does NOT prove the patterns match. Sample and "
              "compare to rows.json before trusting this run.")
    else:
        out = Path(args.out) if args.out else run / args.arm / "block_vocab.json"
        rebuild_flat(run, args.arm, cmd, out)


if __name__ == "__main__":
    main()
