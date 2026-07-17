"""Input → output showcase grids for the conditioned runs (T15).

For each of N held-out val conditions, renders a row:
  image arm:  [conditioning view PNG] [sample 1] [sample 2]
  text arm:   [caption tile]          [sample 1] [sample 2]

Two samples per condition show diversity under the same input.
Output: outputs/cond/showcase_{image,text}.png

Usage:
    python scripts/render_cond_showcase.py --run outputs/cond/image_run --num 6
    python scripts/render_cond_showcase.py --run outputs/cond/text_run --num 6
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from blockgen.curation.houses import load_house_structures
from blockgen.export.minecraftace import load_piece_vocab, split_indices
from blockgen.models.voxel_transformer_cond import CondVoxelAR2
from blockgen.renderer.textured import render_structure
from blockgen.training.train_conditioned import (BOS, EOS, EMBEDS, PIECE_VOCAB,
                                                 build_sequences)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_conditioned import load_run_data  # noqa: E402

PX = 384
SAMPLES_PER_COND = 2


def caption_tile(text: str, px: int) -> Image.Image:
    img = Image.new("RGB", (px, px), (245, 243, 238))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    lines = textwrap.wrap(f"“{text}”", width=30)
    y = px // 2 - len(lines) * 14
    for line in lines:
        draw.text((20, y), line, fill=(40, 40, 40), font=font)
        y += 30
    return img


def struct_tile(s, px: int) -> Image.Image:
    arr = render_structure(s, px=px, bg=(1.0, 1.0, 1.0))
    return Image.fromarray(arr).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--num", type=int, default=6)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--max-new", type=int, default=2000)
    parser.add_argument("--out-dir", default="outputs/cond")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text())
    device = "cuda"

    structures, manifest = load_house_structures(max_dim=32)
    structures, seqs, decode, _bv = load_run_data(run, cfg, structures)
    assignment = split_indices(manifest["items"], seed=0)
    val_idx = [i for i in range(len(seqs))
               if assignment[i] == "val" and seqs[i] is not None][: args.num]

    blob = np.load(EMBEDS)
    embeds = blob["image_embeds" if cfg["cond"] == "image" else "text_embeds"]
    if cfg["cond"] == "image":
        cond1 = torch.tensor(embeds[val_idx], device=device)
    else:
        cond1 = torch.tensor(embeds[val_idx, 0][:, None], device=device)
    # repeat each condition SAMPLES_PER_COND times for diversity
    cond = cond1.repeat_interleave(SAMPLES_PER_COND, dim=0)

    model = CondVoxelAR2(
        cond_dim=cfg["cond_dim"], n_prefix=cfg["n_prefix"],
        vocab_size=cfg["vocab_size"], max_seq_len=cfg["max_seq_len"],
        d_model=cfg["d_model"], nhead=8, num_layers=cfg["layers"],
        dim_feedforward=4 * cfg["d_model"], pe=cfg["pe"]).to(device).eval()
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))

    toks = model.generate_cond(
        cond=cond, bos_token_id=BOS, eos_token_id=EOS,
        max_new_tokens=args.max_new, cfg_scale=args.cfg)
    samples = [decode(row.cpu().numpy()) for row in toks]

    captions = None
    if cfg["cond"] == "text":
        captions = json.load(open("data/minecraft/labels/houses_32_captions.json"))

    cols = 1 + SAMPLES_PER_COND
    sheet = Image.new("RGB", (cols * PX + (cols + 1) * 8,
                              len(val_idx) * PX + (len(val_idx) + 1) * 8),
                      (255, 255, 255))
    for r, i in enumerate(val_idx):
        if cfg["cond"] == "image":
            tile = Image.open(f"outputs/renders/houses_32/h{i:05d}_view0.png") \
                        .convert("RGB").resize((PX, PX))
        else:
            tile = caption_tile(captions[f"h{i:05d}"][0], PX)
        row_tiles = [tile] + [
            struct_tile(samples[r * SAMPLES_PER_COND + k], PX)
            for k in range(SAMPLES_PER_COND)]
        for c, t in enumerate(row_tiles):
            sheet.paste(t, (8 + c * (PX + 8), 8 + r * (PX + 8)))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"showcase_{cfg['cond']}.png"
    sheet.save(path)
    print(f"saved {path} ({len(val_idx)} rows: input | "
          f"{SAMPLES_PER_COND} samples)")


if __name__ == "__main__":
    main()
