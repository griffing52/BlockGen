"""Multi-view render export for auto-labeling and image conditioning.

Renders each structure from 4 fixed viewpoints with the textured EGL renderer
(``blockgen.renderer.textured``) on a white background — the format both VLM
captioners and DINOv2 image conditioning consume.

File naming matches ``blockgen.export.minecraftace`` ids:
``<out_dir>/h00042_view{0..3}.png`` where the index is the position in the
houses cache / export index.json.

Usage:
    python -m blockgen.labeling.render_views --houses 32 \
        --out outputs/renders/houses_32 --px 512 --workers 6
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from blockgen.utils.data import Structure

DEFAULT_VIEWS: Tuple[Tuple[float, float], ...] = (
    (45.0, 30.0), (135.0, 30.0), (225.0, 30.0), (315.0, 30.0))  # (azim, elev)


def render_structure_views(structure: Structure, px: int = 512,
                           views: Sequence[Tuple[float, float]] = DEFAULT_VIEWS,
                           face_textures=None,
                           bg: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                           ) -> List[np.ndarray]:
    """Render one structure from each viewpoint. Returns RGBA arrays."""
    from blockgen.renderer.textured import render_structure
    return [render_structure(structure, px=px, azim_deg=azim, elev_deg=elev,
                             bg=bg, face_textures=face_textures)
            for azim, elev in views]


def _render_range(index_structure_pairs, out_dir: str, px: int,
                  views: Sequence[Tuple[float, float]], overwrite: bool) -> int:
    """Worker body: render a list of (index, Structure). Returns tiles written."""
    from PIL import Image

    from blockgen.renderer.textured import load_face_textures

    face_textures = load_face_textures()  # shared across all renders in this worker
    written = 0
    for idx, structure in index_structure_pairs:
        sid = f"h{idx:05d}"
        paths = [os.path.join(out_dir, f"{sid}_view{v}.png") for v in range(len(views))]
        if not overwrite and all(os.path.exists(p) for p in paths):
            continue
        try:
            images = render_structure_views(structure, px=px, views=views,
                                            face_textures=face_textures)
        except Exception as exc:  # keep the sweep alive on one bad structure
            print(f"render failed for {sid}: {exc}", flush=True)
            continue
        for path, img in zip(paths, images):
            Image.fromarray(img).save(path)
            written += 1
    return written


def export_views(structures: Sequence[Structure], out_dir: str, px: int = 512,
                 views: Sequence[Tuple[float, float]] = DEFAULT_VIEWS,
                 overwrite: bool = False, workers: int = 1,
                 indices: Optional[Sequence[int]] = None) -> int:
    """Render all structures to ``<out_dir>/h<idx>_view<v>.png``. Resumable."""
    os.makedirs(out_dir, exist_ok=True)
    if indices is None:
        indices = range(len(structures))
    pairs = list(zip(indices, structures))

    if workers <= 1:
        return _render_range(pairs, out_dir, px, views, overwrite)

    # One EGL context per process; spawn (not fork) keeps GL state clean.
    import multiprocessing as mp
    chunks = [pairs[i::workers] for i in range(workers)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        counts = pool.starmap(
            _render_range,
            [(chunk, out_dir, px, views, overwrite) for chunk in chunks if chunk])
    return sum(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export multi-view renders.")
    parser.add_argument("--houses", type=int, default=None, metavar="MAX_DIM",
                        help="render the curated houses_<MAX_DIM> cache")
    parser.add_argument("--cache", default=None, metavar="NPZ",
                        help="render any structure cache .npz (e.g. a whole-corpus "
                             "cache); mutually exclusive with --houses")
    parser.add_argument("--out", required=True)
    parser.add_argument("--px", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="render only the first N structures (debug)")
    args = parser.parse_args()

    if (args.houses is None) == (args.cache is None):
        parser.error("pass exactly one of --houses MAX_DIM or --cache NPZ")
    if args.cache is not None:
        from blockgen.curation.houses import load_structures_from_cache
        structures, _ = load_structures_from_cache(args.cache)
    else:
        from blockgen.curation.houses import load_house_structures
        structures, _ = load_house_structures(max_dim=args.houses)
    indices = range(len(structures))
    if args.limit is not None:
        structures = structures[: args.limit]
        indices = range(args.limit)

    written = export_views(structures, args.out, px=args.px, overwrite=args.overwrite,
                           workers=args.workers, indices=indices)
    print(f"wrote {written} tiles -> {args.out}")


if __name__ == "__main__":
    main()
