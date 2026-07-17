"""Dense mosaic grids of many structures — dataset showcase figures.

Tiles per-structure renders into one big image. The tile renderer is
pluggable: uses the textured renderer (real Minecraft textures, pyrender)
when available, else falls back to the matplotlib voxel plot.

CLI::

    python -m blockgen.renderer.grid --houses 32 --rows 8 --cols 12 \
        --out outputs/figures/houses_32_grid.png
"""

from __future__ import annotations

import argparse
import os
from typing import Callable, List, Optional, Sequence

import numpy as np

from blockgen.utils.data import Structure

TileFn = Callable[[Structure, int], np.ndarray]  # (structure, px) -> RGB(A) uint8


_FACE_TEXTURES = None


def _tile_textured(structure: Structure, px: int) -> np.ndarray:
    global _FACE_TEXTURES
    from blockgen.renderer.textured import render_structure
    from blockgen.renderer.textures import load_face_textures
    if _FACE_TEXTURES is None:
        _FACE_TEXTURES = load_face_textures()
    return render_structure(structure, px=px, face_textures=_FACE_TEXTURES)


def _tile_matplotlib(structure: Structure, px: int) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from blockgen.renderer.render import render_schem
    dpi = 100
    fig = plt.figure(figsize=(px / dpi, px / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    render_schem(structure, ax=ax, show=False)
    ax.set_title("")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4).copy()
    plt.close(fig)
    return img


def default_tile_fn() -> TileFn:
    try:
        import pyrender  # noqa: F401
        from blockgen.renderer import textured  # noqa: F401
        return _tile_textured
    except Exception:  # noqa: BLE001
        return _tile_matplotlib


def render_grid(structures: Sequence[Structure], cols: int = 10,
                tile_px: int = 192, pad: int = 2,
                bg: tuple = (255, 255, 255), tile_fn: Optional[TileFn] = None,
                verbose: bool = True) -> np.ndarray:
    """Render structures into a dense (rows x cols) RGB mosaic."""
    tile_fn = tile_fn or default_tile_fn()
    n = len(structures)
    rows = (n + cols - 1) // cols
    H = rows * tile_px + (rows + 1) * pad
    W = cols * tile_px + (cols + 1) * pad
    canvas = np.empty((H, W, 3), dtype=np.uint8)
    canvas[:] = np.asarray(bg, dtype=np.uint8)
    for k, s in enumerate(structures):
        img = tile_fn(s, tile_px)
        if img.shape[-1] == 4:  # alpha-composite onto bg
            a = img[..., 3:4].astype(np.float32) / 255.0
            rgb = (img[..., :3].astype(np.float32) * a
                   + np.asarray(bg, np.float32) * (1 - a)).astype(np.uint8)
        else:
            rgb = img[..., :3]
        if rgb.shape[0] != tile_px:  # defensive resize (nearest)
            yi = (np.arange(tile_px) * rgb.shape[0] // tile_px)
            xi = (np.arange(tile_px) * rgb.shape[1] // tile_px)
            rgb = rgb[yi][:, xi]
        r, c = divmod(k, cols)
        y0 = pad + r * (tile_px + pad)
        x0 = pad + c * (tile_px + pad)
        canvas[y0:y0 + tile_px, x0:x0 + tile_px] = rgb
        if verbose and (k + 1) % 25 == 0:
            print(f"  rendered {k + 1}/{n} tiles", flush=True)
    return canvas


def save_grid(structures: Sequence[Structure], out_path: str, cols: int = 10,
              tile_px: int = 192, **kwargs) -> str:
    from PIL import Image
    img = render_grid(structures, cols=cols, tile_px=tile_px, **kwargs)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    Image.fromarray(img).save(out_path)
    print(f"saved {img.shape[1]}x{img.shape[0]} grid ({len(structures)} tiles) -> {out_path}")
    return out_path


def _pick(structures: List[Structure], n: int, seed: int, order: str) -> List[Structure]:
    if order == "largest":
        return sorted(structures, key=lambda s: -int(s.occupied_mask.sum()))[:n]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(structures))[:n]
    return [structures[i] for i in idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a dense sample grid.")
    parser.add_argument("--houses", type=int, metavar="MAX_DIM",
                        help="use the curated house cache houses_<MAX_DIM>.npz")
    parser.add_argument("--cache", help="or: path to any structure .npz cache")
    parser.add_argument("--corpus", help="filter house cache to one corpus "
                        "(grabcraft|3dcraft|text2mc)")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--cols", type=int, default=12)
    parser.add_argument("--tile-px", type=int, default=192)
    parser.add_argument("--order", choices=["random", "largest"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--matplotlib", action="store_true",
                        help="force the matplotlib tile renderer")
    args = parser.parse_args()

    if args.houses:
        from blockgen.curation.houses import load_house_structures
        structures, _ = load_house_structures(max_dim=args.houses)
        if args.corpus:
            structures = [s for s in structures
                          if s.metadata.get("corpus") == args.corpus]
    elif args.cache:
        blob = np.load(args.cache, allow_pickle=True)
        structures = [Structure(block_ids=blob["block_ids"][i],
                                block_data=blob["block_data"][i])
                      for i in range(len(blob["block_ids"]))]
    else:
        parser.error("pass --houses MAX_DIM or --cache PATH")

    n = min(args.rows * args.cols, len(structures))
    picked = _pick(structures, n, args.seed, args.order)
    tile_fn = _tile_matplotlib if args.matplotlib else None
    save_grid(picked, args.out, cols=args.cols, tile_px=args.tile_px, tile_fn=tile_fn)


if __name__ == "__main__":
    main()
