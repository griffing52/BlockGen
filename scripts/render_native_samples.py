"""Textured sample sheets for the §9.0 arms (results.md T20).

`_eval_row` only writes matplotlib voxel plots and does not persist the sampled
Structures, so this reloads each checkpoint and re-samples, then renders with the
real-texture pyrender pipeline -- same look as the F16 dataset sheets, so generated
output can be compared to real houses in the same visual language.

Rebuilds each arm's exact vocab the way the run did (seed 0 split -> D4 augment ->
canonicalize), so token ids line up with the saved weights.

Usage::
    .venv/bin/python scripts/render_native_samples.py --run outputs/run_20260715_065938_native
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from blockgen.curation.houses import load_house_structures  # noqa: F401
from blockgen.experiments_gen import canonicalize
from blockgen.experiments_transfer import _split_houses
from blockgen.models.voxel_transformer_ar2 import VoxelTransformerAR2
from blockgen.renderer.grid import save_grid
from blockgen.tokenizers.cluster_bpe import (build_cluster_sequences, learn_clusters,
                                             cluster_tokens_to_structure)
from blockgen.utils.augment import augment_with_labels
from blockgen.utils.serialize import (BOS_TOKEN, EOS_TOKEN, build_block_vocab,
                                      tokens_to_structure)


def occ(s):
    return int((s.block_ids != s.air_block_id).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--merges", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--outdir", default="outputs/figures")
    args = ap.parse_args()

    run = Path(args.run)
    torch.manual_seed(args.seed)
    h_train, h_val = _split_houses(args.seed, args.val_frac)
    h_aug, _ = augment_with_labels(h_train, ["h"] * len(h_train))
    os.makedirs(args.outdir, exist_ok=True)

    # ---- arm A: canon-16 + flat ------------------------------------------------
    ckpt = run / "canon16_flat" / "model.pt"
    if ckpt.exists():
        ref, val = canonicalize(h_train, 16), canonicalize(h_val, 16)
        vocab = build_block_vocab(ref + val, max_dim=16)
        m = VoxelTransformerAR2(vocab_size=vocab.vocab_size, max_seq_len=1600,
                                pe="phase4").cuda()
        m.load_state_dict(torch.load(ckpt)); m.eval()
        out = []
        for i in range(args.samples):
            t = m.generate(bos_token_id=BOS_TOKEN, eos_token_id=EOS_TOKEN,
                           max_new_tokens=1599, temperature=1.0, top_k=40)
            s = tokens_to_structure(t[0].tolist(), vocab)
            out.append(s); print(f"  A #{i}: occ={occ(s)}", flush=True)
        p = f"{args.outdir}/samples_native_canon16_flat.png"
        save_grid(out, p, cols=6, tile_px=256)
        print(f"arm A -> {p}")

    # ---- arm B: native 32 + 3D-BPE ---------------------------------------------
    ckpt = run / "native_bpe" / "model.pt"
    if ckpt.exists():
        fit = canonicalize(h_aug, 32)
        cv = learn_clusters(fit, max_dim=32, n_merges=args.merges, max_corpus=400,
                            verbose=False)
        m = VoxelTransformerAR2(vocab_size=cv.vocab_size, max_seq_len=5480,
                                pe="phase4").cuda()
        m.load_state_dict(torch.load(ckpt)); m.eval()
        out = []
        for i in range(args.samples):
            t = m.generate(bos_token_id=BOS_TOKEN, eos_token_id=EOS_TOKEN,
                           max_new_tokens=5479, temperature=1.0, top_k=40)
            s = cluster_tokens_to_structure(t[0].tolist(), cv)
            out.append(s); print(f"  B #{i}: occ={occ(s)}", flush=True)
        p = f"{args.outdir}/samples_native_bpe.png"
        save_grid(out, p, cols=6, tile_px=256)
        print(f"arm B -> {p}")


if __name__ == "__main__":
    main()
