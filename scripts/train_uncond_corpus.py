"""Unconditioned native_bpe on the FULL corpus with the new changes — the diagnostic.

Trains an unconditioned ``VoxelTransformerAR2`` (phase4) over 3D-BPE piece tokens of the
whole-corpus cache using the ORIENTED, forced-multi-block-merge vocab, so we can see
whether the native_bpe cohesion survives the changes (bigger oriented vocab + corpus
diversity) independent of conditioning/CFG. If this is cohesive, the conditioned run's
blobbiness is a conditioning/CFG problem; if it's also blobby, it's the corpus/vocab.

**No D4 augmentation:** the augmenter does not rotate ``block_data``, so augmenting the
oriented vocab would corrupt stair/log facings (notes §17). Trains on the raw corpus.

    python scripts/train_uncond_corpus.py --epochs 60
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from blockgen.eval.cond_render import textured_prompt_grid
from blockgen.experiments_gen import canonicalize
from blockgen.export.minecraftace import load_piece_vocab, save_piece_vocab
from blockgen.tokenizers.cluster_bpe import build_cluster_sequences
from blockgen.training.train_ar import ARTrainConfig
from blockgen.training.train_ar_ext import sample_cluster_structures, train_from_sequences
from blockgen.utils.runs import new_run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="data/minecraft/cache/all_32.npz")
    ap.add_argument("--piece-vocab", default="outputs/corpus/piece_vocab_oriented.json")
    ap.add_argument("--name", default="uncond_corpus_oriented")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=5480)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = new_run_dir(args.name)
    print(f"run dir: {out}", flush=True)

    from blockgen.curation.houses import load_structures_from_cache
    structs, _ = load_structures_from_cache(args.cache)
    cv = load_piece_vocab(args.piece_vocab)
    fit = canonicalize(structs, 32)                       # crop only (native 32), no downsample
    seqs, kept = build_cluster_sequences(fit, cv, args.max_seq_len)
    med = int(np.median([len(s) for s in seqs])) if seqs else 0
    print(f"{len(kept)}/{len(fit)} fit max_seq_len {args.max_seq_len}; vocab {cv.vocab_size}, "
          f"oriented={cv.oriented}, median_len {med}", flush=True)
    save_piece_vocab(cv, str(out / "piece_vocab.json"))   # co-locate vocab with the run

    cfg = ARTrainConfig(max_seq_len=args.max_seq_len, epochs=args.epochs,
                        batch_size=args.batch, device=device)
    t = time.time()
    model, hist = train_from_sequences(seqs, cv.vocab_size, cfg, pe="phase4")
    torch.save(model.state_dict(), out / "model.pt")
    json.dump({"cache": args.cache, "piece_vocab": args.piece_vocab,
               "vocab_size": cv.vocab_size, "oriented": cv.oriented, "n_train": len(kept),
               "epochs": args.epochs, "final_loss": round(hist["loss"][-1], 4),
               "median_seq_len": med, "train_min": round((time.time() - t) / 60, 1)},
              open(out / "config.json", "w"), indent=1)

    # Textured samples (project standard), titled with block counts (no prompts here).
    samples = sample_cluster_structures(model, cv, num_samples=args.samples,
                                        temperature=1.0, top_k=40)
    textured_prompt_grid(samples, [f"{int(s.occupied_mask.sum())} blocks" for s in samples],
                         out / "samples.png",
                         suptitle=f"{out.name} — uncond, native_bpe + oriented vocab")
    print(f"done. final_loss {hist['loss'][-1]:.4f}; samples -> {out}/samples.png",
          flush=True)


if __name__ == "__main__":
    main()
