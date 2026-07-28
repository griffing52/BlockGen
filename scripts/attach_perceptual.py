"""Perceptual scoring of attachment/growth arms — the metric that actually adjudicates.

notes.md T20 is blunt: `nn_iou` disagreed with the eye three times in one day and is
grid-dependent, so "nothing downstream is adjudicable" without a perceptual metric.
`eval/perceptual.py` (CMMD in CLIP space + CLIP text score) is that metric. Everything
measured overnight — bits/op, thickness, occupancy — is a cheap proxy; this is the one
that scores how the builds *look*.

Arms are scored against a **single shared real reference** (cached and reused), because
scoring each arm against its own training slice would reintroduce the normalization
problem the module exists to escape.

The reference is drawn from builds the arm could actually train on (seq <= max_seq_len)
when `--match-filter` is set, since that filter systematically removes large builds and
an unfiltered reference would penalize every arm for a data-selection effect.

Usage:
    python -m scripts.attach_perceptual --run outputs/run_..._attach_growth_ordering --temp 0.6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from blockgen.curation.houses import load_structures_from_cache
from blockgen.eval.perceptual import (DEFAULT_PROMPT, DEFAULT_VIEWS,
                                      clip_image_features, evaluate_perceptual,
                                      render_views)
from blockgen.utils.attach_order import attach_ops_to_structure, structure_to_attach_ops
from blockgen.utils.attach_vocab import BOS_ID, EOS_ID, AttachVocab


def load_model(arm: Path, device: str):
    blob = torch.load(arm / "model.pt", map_location=device, weights_only=False)
    cfg = blob["config"]
    from blockgen.models.voxel_transformer_ar2 import VoxelTransformerAR2
    m = VoxelTransformerAR2(
        vocab_size=blob["vocab_size"], max_seq_len=cfg["max_seq_len"],
        d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"], dropout=cfg["dropout"], pe="sin",
    ).to(device)
    m.load_state_dict(blob["state_dict"])
    m.eval()
    return m, cfg, blob["ordering"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--n-ref", type=int, default=64)
    ap.add_argument("--cache", default="data/minecraft/cache/all_32.npz")
    ap.add_argument("--match-filter", action="store_true", default=True,
                    help="draw the reference from builds that fit the arm's seq cap")
    ap.add_argument("--seed", type=int, default=None,
                    help="torch seed; also tags the output file so replicates coexist")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    run = Path(args.run)
    arms = sorted(p for p in run.iterdir() if p.is_dir() and (p / "model.pt").exists())
    if not arms:
        raise SystemExit(f"no arms with model.pt under {run}")

    structures, _ = load_structures_from_cache(args.cache)

    # Shared reference, built once and reused across every arm.
    _m, cfg0, ord0 = load_model(arms[0], args.device)
    max_len = cfg0["max_seq_len"]
    del _m
    ref: list = []
    for s in structures:
        if len(ref) >= args.n_ref:
            break
        if args.match_filter:
            try:
                _ops, info = structure_to_attach_ops(s, ordering=ord0)
            except Exception:
                continue
            if info["n_ops"] + 2 > max_len:
                continue
        ref.append(s.crop_to_non_air())
    print(f"[perceptual] reference: {len(ref)} real builds "
          f"({'seq<=%d' % max_len if args.match_filter else 'unfiltered'})", flush=True)
    ref_feats = clip_image_features(render_views(ref, DEFAULT_VIEWS, 224, True), args.device)

    rows = []
    for arm in arms:
        model, cfg, ordering = load_model(arm, args.device)
        vocab = AttachVocab.load(arm / "attach_vocab.json")
        samples = []
        for _ in range(args.n):
            with torch.no_grad():
                toks = model.generate(bos_token_id=BOS_ID, eos_token_id=EOS_ID,
                                      max_new_tokens=cfg["max_seq_len"],
                                      temperature=args.temp, device=args.device)
            if isinstance(toks, torch.Tensor):
                toks = toks.flatten().tolist()
            ops = vocab.ids_to_ops(toks)
            if not ops:
                continue
            s = attach_ops_to_structure(ops, ordering=ordering, oriented=vocab.oriented)
            if int(s.occupied_mask.sum()) > 0:
                samples.append(s)
        if not samples:
            print(f"[perceptual] {ordering}: no samples", flush=True)
            continue
        rep = evaluate_perceptual(samples, ref, ref_feats=ref_feats, device=args.device,
                                  verbose=False)
        row = rep.row(ordering)
        row["temp"] = args.temp
        rows.append(row)
        print(f"[perceptual] {ordering:20s} CMMD={rep.cmmd:8.3f} (lower better)  "
              f"CLIP={rep.clip_text:.4f} (higher better)  n={len(samples)}", flush=True)
        del model
        torch.cuda.empty_cache()

    # Real-vs-real floor: how low can CMMD go given sample size? Without this the
    # arm numbers have no scale.
    half = len(ref) // 2
    if half >= 4:
        floor = evaluate_perceptual(ref[:half], ref[half:],
                                    ref_feats=clip_image_features(
                                        render_views(ref[half:], DEFAULT_VIEWS, 224, False),
                                        args.device),
                                    device=args.device, verbose=False)
        print(f"\n[perceptual] real-vs-real FLOOR  CMMD={floor.cmmd:8.3f}  "
              f"CLIP={floor.clip_text:.4f}  <-- the best achievable at this n", flush=True)
        rows.append({**floor.row("_real_vs_real_floor"), "temp": None})

    tag = f"perceptual_T{args.temp:g}".replace(".", "")
    if args.seed is not None:
        tag += f"_s{args.seed}"
    out = run / (tag + ".json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
