"""Render a conditioned resampler run's samples with the TEXTURED renderer + prompts.

Regenerates samples from a trained ``ResampledCondVoxelAR2`` run, conditioned on
held-out val captions, renders each build with the real-texture EGL renderer
(``blockgen.renderer.textured``) rather than the matplotlib voxel grid, and titles
each with the exact input caption so prompt -> output is visible. Writes
``samples_textured.png`` into the run dir.

    python scripts/render_cond_samples.py --run outputs/run_..._all32_text_resampler \
        --num 12 --cfg 3.0 --caption-idx 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from blockgen.eval.cond_render import render_cond_run
from blockgen.export.minecraftace import load_piece_vocab, split_indices
from blockgen.models.condition_resampler import ResampledCondVoxelAR2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run dir with config.json + best.pt")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--cfg", type=float, default=None, help="override config cfg")
    ap.add_argument("--caption-idx", type=int, default=0,
                    help="which of the 4 captions to condition on / show (0 = short tag)")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--px", type=int, default=384)
    ap.add_argument("--max-new", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    cv = load_piece_vocab(cfg["piece_vocab"])
    model = ResampledCondVoxelAR2(
        cond_dim=cfg["cond_dim"], n_prefix=cfg["n_prefix"], vocab_size=cfg["vocab_size"],
        max_seq_len=cfg["max_seq_len"], d_model=cfg["d_model"], nhead=8,
        num_layers=cfg["layers"], dim_feedforward=4 * cfg["d_model"], pe="phase4",
        resampler_layers=cfg["resampler_layers"]).to(device).eval()
    model.load_state_dict(torch.load(run / args.ckpt, map_location=device))

    # Held-out val split (same seed as training), + captions to display.
    from blockgen.curation.houses import load_structures_from_cache
    _, manifest = load_structures_from_cache(cfg["cache"])
    assignment = split_indices(manifest["items"], seed=0)
    val_idx = [i for i in range(len(manifest["items"])) if assignment[i] == "val"]
    captions = json.load(open("data/minecraft/labels/all_32_captions.json"))

    blob = np.load(cfg["embeds"])
    tok_embeds, tok_mask = blob["text_token_embeds"], blob["text_token_mask"]

    # Keep only val structures that have a caption at caption-idx with real tokens.
    picks = []
    for i in val_idx:
        if int(tok_mask[i, args.caption_idx].sum()) > 0 and captions.get(f"h{i:05d}"):
            picks.append(i)
        if len(picks) >= args.num:
            break

    cfg_scale = args.cfg if args.cfg is not None else cfg.get("cfg", 3.0)
    print(f"generating {len(picks)} samples (cfg {cfg_scale})...", flush=True)
    out = render_cond_run(
        model, cv, picks, tok_embeds, tok_mask, captions, run / "samples_textured.png",
        cfg_scale=cfg_scale, caption_idx=args.caption_idx, device=device,
        max_new=args.max_new, cols=args.cols, px=args.px,
        suptitle=f"{run.name} — textured, prompt -> output (cfg {cfg_scale})")
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
