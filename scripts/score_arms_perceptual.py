"""Adjudicate the T20 arms with the perceptual metric (results.md T21).

nn_iou said native_bpe (75.4%) < canon16_flat (83.3%), while native_bpe produced the
first samples in the project with pitched roofs and window openings. That disagreement
is why blockgen/eval/perceptual.py exists. It passed its sanity check
(scripts/validate_perceptual.py): CMMD floor 0.056 on held-out real natives, 0.266 on
canon-16, 0.748 on canon-8 -- monotone in known damage -- and 0.520 on
materials-shuffled builds that nn_iou would score at a perfect IoU of 1.0.

**Both arms are scored against the SAME real-native reference**, rendered identically at
224px. That is the whole point: a 16³ build and a 32³ build are both just images, so the
metric does not inherit nn_iou's grid dependence. It does handicap the canon-16 arm --
deliberately. The question is "does it make good Minecraft houses?", not "did it learn
its own degraded training data?".

Anchors printed alongside so the arm numbers are readable rather than free-floating:
real-native (floor) and real-@-canon-16 (what a *perfect* canon-16 model could hope for).

Usage::
    .venv/bin/python scripts/score_arms_perceptual.py --run outputs/run_20260715_065938_native
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from blockgen.eval.perceptual import (DEFAULT_PROMPT, DEFAULT_VIEWS, clip_image_features,
                                      clip_text_features, cmmd, render_views)
from blockgen.experiments_gen import canonicalize
from blockgen.experiments_transfer import _split_houses
from blockgen.models.voxel_transformer_ar2 import VoxelTransformerAR2
from blockgen.tokenizers.cluster_bpe import cluster_tokens_to_structure, learn_clusters
from blockgen.utils.augment import augment_with_labels
from blockgen.utils.serialize import (BOS_TOKEN, EOS_TOKEN, build_block_vocab,
                                      tokens_to_structure)


def occ(s):
    return int((s.block_ids != s.air_block_id).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--ref", type=int, default=32)
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--px", type=int, default=224)
    ap.add_argument("--merges", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/analysis/perceptual_arms.json")
    args = ap.parse_args()

    views = DEFAULT_VIEWS[: args.views]
    torch.manual_seed(args.seed)
    h_train, h_val = _split_houses(args.seed, 0.15)
    h_aug, _ = augment_with_labels(h_train, ["h"] * len(h_train))

    # --- the shared real-native reference (held-out val builds, never trained on) ---
    ref = [s.crop_to_non_air() for s in h_val][: args.ref]
    print(f"reference: {len(ref)} held-out real native builds, {len(views)} views")
    f_ref = clip_image_features(render_views(ref, views, args.px, verbose=False))
    t = clip_text_features([DEFAULT_PROMPT])

    rows = {}

    def score(tag, structs):
        f = clip_image_features(render_views(structs, views, args.px, verbose=False))
        rows[tag] = {"cmmd": round(cmmd(f, f_ref), 3),
                     "clip_text": round(float((f @ t.T).mean()), 4),
                     "median_occ": float(np.median([occ(s) for s in structs])),
                     "n": len(structs)}
        print(f"  {tag:<34} CMMD {rows[tag]['cmmd']:>7.3f}   occ {rows[tag]['median_occ']:>6.0f}")

    # --- anchors: what real data scores, so the arms are readable ------------------
    extra = [s.crop_to_non_air() for s in h_train][: args.ref]
    print("\nanchors (real data):")
    score("real native (train, held-out ref)", extra)
    score("real @ canon-16 (arm A's ceiling)", canonicalize(extra, 16))

    # --- arm A: canon-16 + flat -----------------------------------------------------
    print("\narms:")
    ck = os.path.join(args.run, "canon16_flat", "model.pt")
    if os.path.exists(ck):
        v = build_block_vocab(canonicalize(h_train, 16) + canonicalize(h_val, 16), max_dim=16)
        m = VoxelTransformerAR2(vocab_size=v.vocab_size, max_seq_len=1600, pe="phase4").cuda()
        m.load_state_dict(torch.load(ck)); m.eval()
        out = [tokens_to_structure(
            m.generate(bos_token_id=BOS_TOKEN, eos_token_id=EOS_TOKEN,
                       max_new_tokens=1599, temperature=1.0, top_k=40)[0].tolist(), v)
            for _ in range(args.samples)]
        score("canon16_flat  (nn_iou 83.3%)", out)

    # --- arm B: native 32 + 3D-BPE --------------------------------------------------
    ck = os.path.join(args.run, "native_bpe", "model.pt")
    if os.path.exists(ck):
        cv = learn_clusters(canonicalize(h_aug, 32), max_dim=32, n_merges=args.merges,
                            max_corpus=400, verbose=False)
        m = VoxelTransformerAR2(vocab_size=cv.vocab_size, max_seq_len=5480, pe="phase4").cuda()
        m.load_state_dict(torch.load(ck)); m.eval()
        out = [cluster_tokens_to_structure(
            m.generate(bos_token_id=BOS_TOKEN, eos_token_id=EOS_TOKEN,
                       max_new_tokens=5479, temperature=1.0, top_k=40)[0].tolist(), cv)
            for _ in range(args.samples)]
        score("native_bpe    (nn_iou 75.4%)", out)

    print(f"\n{'':<36}{'CMMD':>8}{'CLIP txt':>10}{'occ':>8}")
    print("-" * 62)
    for k, v in rows.items():
        print(f"{k:<36}{v['cmmd']:>8.3f}{v['clip_text']:>10.4f}{v['median_occ']:>8.0f}")
    print("-" * 62)
    print("CMMD: LOWER = looks more like a real native Minecraft house.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"run": args.run, "views": len(views), "px": args.px,
               "samples": args.samples, "rows": rows}, open(args.out, "w"), indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
