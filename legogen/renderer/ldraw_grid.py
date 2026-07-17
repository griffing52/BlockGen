"""Render a dense grid of LDraw models from a LEGO corpus, à la the Minecraft
``blockgen/renderer/grid.py`` dataset sheets.

Selects models from an OMR-style folder of ``.mpd`` / ``.ldr`` files (filtered by
part-reference count so tiles are legible), renders each to a transparent PNG
tile with Blender + ImportLDraw (``blender_render.py``, one batched session), then
composites the tiles onto a white canvas.

    # dense sheet of small/medium official sets
    legogen/renderer/ldraw_grid.py --rows 8 --cols 12 --band 15 160 \
        --out outputs/figures/lego_data/omr_grid.png

    # showcase sheet of larger, part-diverse sets
    legogen/renderer/ldraw_grid.py --rows 4 --cols 6 --band 160 600 --tile-px 384 \
        --render-res 640 --out outputs/figures/lego_data/omr_showcase.png

Blender is invoked as a subprocess, so run this with a normal Python (it only
needs numpy + pillow); Blender itself needs no extra packages.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
BLENDER = Path("/home/griffing52/downloads/blender-4.2.3-linux-x64/blender")
LDRAW_LIB = REPO / "data" / "lego" / "ldraw" / "ldraw"
RENDER_SCRIPT = REPO / "legogen" / "renderer" / "blender_render.py"


def part_ref_count(path: Path) -> int:
    try:
        return sum(1 for ln in path.read_text(errors="ignore").splitlines()
                   if ln.strip().startswith("1 "))
    except Exception:  # noqa: BLE001
        return -1


def select_models(src_dir: Path, band: tuple[int, int], n: int, seed: int,
                  spread: bool) -> list[Path]:
    """Files whose part-ref count falls in ``band``, sampled to ``n``."""
    files = sorted(p for p in src_dir.glob("*.mpd"))
    files += sorted(p for p in src_dir.glob("*.ldr"))
    lo, hi = band
    keep = [p for p in sorted(set(files)) if lo <= part_ref_count(p) <= hi]
    if len(keep) <= n:
        return keep
    if spread:  # evenly spaced across the sorted (≈ set-number / theme) order
        idx = np.linspace(0, len(keep) - 1, n).round().astype(int)
        return [keep[i] for i in sorted(set(idx.tolist()))]
    rng = np.random.default_rng(seed)
    return [keep[i] for i in sorted(rng.permutation(len(keep))[:n].tolist())]


def run_blender(job_path: Path, log_path: Path) -> None:
    cmd = [str(BLENDER), "--background", "--python", str(RENDER_SCRIPT), "--",
           str(job_path)]
    print("[ldraw_grid] launching Blender:", " ".join(cmd), flush=True)
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"[ldraw_grid] Blender exit {proc.returncode}; see {log_path}", flush=True)


def compose_grid(tiles: list[dict], rows: int, cols: int, tile_px: int,
                 pad: int, out_path: Path) -> int:
    W = cols * tile_px + (cols + 1) * pad
    H = rows * tile_px + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    placed = 0
    for k, tile in enumerate(tiles):
        if k >= rows * cols:
            break
        p = Path(tile["out"])
        if not p.exists():
            continue
        img = Image.open(p).convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
        img.thumbnail((tile_px, tile_px), Image.LANCZOS)
        r, c = divmod(k, cols)
        x = pad + c * (tile_px + pad) + (tile_px - img.width) // 2
        y = pad + r * (tile_px + pad) + (tile_px - img.height) // 2
        canvas.paste(img, (x, y))
        placed += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return placed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "data" / "lego" / "omr" / "files"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--cols", type=int, default=12)
    ap.add_argument("--band", type=int, nargs=2, default=[15, 160],
                    metavar=("LO", "HI"), help="part-reference-count band")
    ap.add_argument("--tile-px", type=int, default=256)
    ap.add_argument("--pad", type=int, default=6)
    ap.add_argument("--render-res", type=int, default=512)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--engine", choices=["CYCLES", "EEVEE"], default="CYCLES")
    ap.add_argument("--azim", type=float, default=40.0)
    ap.add_argument("--elev", type=float, default=28.0)
    ap.add_argument("--margin", type=float, default=1.12)
    ap.add_argument("--colour-scheme", default="lgeo")
    ap.add_argument("--look", default="normal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random", action="store_true",
                    help="random sample instead of evenly-spread")
    ap.add_argument("--skip-render", action="store_true",
                    help="reuse existing tiles, only recompose the grid")
    args = ap.parse_args()

    out_path = Path(args.out)
    work = out_path.parent / (out_path.stem + "_tiles")
    work.mkdir(parents=True, exist_ok=True)
    n = args.rows * args.cols

    models = select_models(Path(args.src), tuple(args.band), n, args.seed,
                           spread=not args.random)
    print(f"[ldraw_grid] selected {len(models)} models (band {args.band}, want {n})",
          flush=True)
    tiles = [{"src": str(p.resolve()), "out": str((work / (p.stem + ".png")).resolve())}
             for p in models]

    if not args.skip_render:
        job = {"ldraw_lib": str(LDRAW_LIB.resolve()), "res": args.render_res,
               "samples": args.samples, "engine": args.engine, "azim": args.azim,
               "elev": args.elev, "margin": args.margin,
               "colour_scheme": args.colour_scheme, "look": args.look, "tiles": tiles}
        job_path = work / "job.json"
        job_path.write_text(json.dumps(job, indent=1))
        run_blender(job_path, work / "blender.log")

    placed = compose_grid(tiles, args.rows, args.cols, args.tile_px, args.pad, out_path)
    print(f"[ldraw_grid] composed {placed}/{len(tiles)} tiles -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
