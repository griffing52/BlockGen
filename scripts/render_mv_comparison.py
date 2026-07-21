"""Input-vs-output sheet for MinecraftACE multi-view conditioned samples.

The Minecraft analog of the LegoACE-MV reconstruction figure: for each held-out
structure, show the 4 rendered views the model was conditioned on, then the
structures it regenerated from them.

Usage:
    python scripts/render_mv_comparison.py \
        --samples-dir outputs/minecraftace/mv-samples \
        --dataset-dir data/minecraftace/houses_32_bpe \
        --out outputs/figures/minecraftace/mv-sheet.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from blockgen.curation.houses import load_house_structures
from blockgen.eval.validity import n_components
from blockgen.export.minecraftace import load_piece_vocab, piece_records_to_structure
from blockgen.renderer.textured import render_structure

CELL, PAD, LABEL_W, HDR = 300, 10, 250, 74


def font(sz):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", sz)
    except OSError:
        return ImageFont.load_default()


def render_rgb(structure, px):
    """Textured render on white; returns a PIL RGB image (blank if empty)."""
    if structure is None or not structure.occupied_mask.any():
        return Image.new("RGB", (px, px), "white")
    rgba = render_structure(structure, px=px, azim_deg=45.0, elev_deg=30.0, bg=(1, 1, 1))
    return Image.fromarray(np.asarray(rgba)[..., :3], "RGB")


def four_view_tile(paths, size):
    tile = Image.new("RGB", (size, size), "white")
    half = size // 2
    for i, p in enumerate(paths[:4]):
        if Path(p).exists():
            tile.paste(Image.open(p).convert("RGB").resize((half, half), Image.LANCZOS),
                       ((i % 2) * half, (i // 2) * half))
    d = ImageDraw.Draw(tile)
    d.line([(half, 0), (half, size)], fill=(225, 225, 225))
    d.line([(0, half), (size, half)], fill=(225, 225, 225))
    return tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--px", type=int, default=512)
    ap.add_argument("--max-rows", type=int, default=8)
    ap.add_argument("--title", default="MinecraftACE-MV: 4 rendered views -> regenerated structure")
    args = ap.parse_args()

    samples_dir, ds_dir = Path(args.samples_dir), Path(args.dataset_dir)
    manifest = json.loads((samples_dir / "manifest.json").read_text())

    pv = list(ds_dir.glob("*_piece_vocab.json"))
    if not pv:
        raise FileNotFoundError(f"no *_piece_vocab.json in {ds_dir} (BPE arm expected)")
    cv = load_piece_vocab(str(pv[0]))
    # entry['n_blocks'] counts BPE pieces, not voxels; use the cache for a like-for-like count
    gt_all, _ = load_house_structures(max_dim=32)

    rows = list(manifest)[: args.max_rows]
    n_samp = max(len(manifest[r]["samples"]) for r in rows)

    W = LABEL_W + (1 + n_samp) * (CELL + PAD) + PAD
    H = HDR + len(rows) * (CELL + PAD) + PAD
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    d.text((PAD, 12), args.title, fill="black", font=font(21))

    heads = ["INPUT: 4 conditioning views"] + [f"OUTPUT: sample {i}" for i in range(n_samp)]
    for c, h in enumerate(heads):
        d.text((LABEL_W + c * (CELL + PAD) + 4, 48), h, fill=(70, 70, 70), font=font(15))

    f_lbl, f_sm = font(15), font(13)
    for r, sid in enumerate(rows):
        y = HDR + r * (CELL + PAD)
        entry = manifest[sid]
        d.text((PAD, y + 8), sid, fill="black", font=f_lbl)
        gt_vox = int(gt_all[int(sid[1:])].occupied_mask.sum())
        d.text((PAD, y + 28), f"GT {gt_vox} blocks", fill=(90, 90, 90), font=f_sm)

        sheet.paste(four_view_tile(entry["images"], CELL), (LABEL_W, y))
        d.rectangle([LABEL_W, y, LABEL_W + CELL - 1, y + CELL - 1], outline=(40, 120, 200), width=2)

        for c, s in enumerate(entry["samples"]):
            x = LABEL_W + (c + 1) * (CELL + PAD)
            rec = np.load(samples_dir / "npy" / s["file"])
            struct = piece_records_to_structure(rec, cv) if len(rec) else None
            sheet.paste(render_rgb(struct, args.px).resize((CELL, CELL), Image.LANCZOS), (x, y))
            d.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline=(215, 215, 215))

            nb = 0 if struct is None else int(struct.occupied_mask.sum())
            comp = n_components(struct) if nb else 0
            note = f"{nb} blocks, {comp} parts" + ("" if s["terminated"] else "  (no EOS)")
            d.text((x + 6, y + CELL - 20), note, fill=(90, 90, 90), font=f_sm)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"saved {args.out} {sheet.size}")


if __name__ == "__main__":
    main()
