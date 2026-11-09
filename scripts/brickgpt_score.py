"""BrickGPT outputs -> bench arm npz + samples.png (project venv).

    python scripts/brickgpt_score.py --run outputs/run_<stamp>_brickgpt_lego

Reads `<run>/gen/*.json` (from `scripts/brickgpt_generate.py`), voxelizes each
brick list with the SAME encoder the LEGO corpus uses (`bench.lego`: z-up ->
y-up, wool keyed by brick footprint), and writes

* `<run>/brickgpt_20.npz` (+ manifest): the arm the bench scores;
* `<run>/heldout_matched_20.npz`: the held-out structure each prompt came from,
  in prompt order (a paired reference for the figure; not the bench reference);
* `<run>/samples.png`: generated (odd rows) above the held-out structure of the
  same prompt (even rows), textured renderer.
* `<run>/gen_summary.json`: throughput, rejection and rollback statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from blockgen.curation.houses import save_house_cache
from blockgen.eval.bench import lego


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--fig-pairs", type=int, default=24)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    run = Path(args.run)

    import pyarrow.parquet as pq
    rows = {r["structure_id"]: r for r in pq.read_table(
        lego.DEFAULT_ROOT / "test-00000-of-00001.parquet",
        columns=["structure_id", "object_id", "category_id", "bricks"]).to_pylist()}

    recs = [json.loads(f.read_text()) for f in sorted((run / "gen").glob("*.json"))]
    gen, ref, bad = [], [], 0
    for g in recs:
        meta = {"corpus": "brickgpt", "category": g["category"],
                "title": g["caption"][:120], "url": "hf://AvaLovelace/BrickGPT"}
        try:
            s = lego.bricks_to_structure(lego.parse_bricks(g["bricks"]),
                                         source_path=f"brickgpt:{g['idx']:04d}",
                                         metadata=meta)
        except ValueError:
            bad += 1
            continue
        if not s.occupied_mask.any():
            bad += 1
            continue
        gen.append(s.crop_to_non_air())
        ref.append(lego.row_structure(rows[g["structure_id"]], "test").crop_to_non_air())

    save_house_cache(gen, max_dim=lego.GRID, cache_dir=str(run), name="brickgpt",
                     report={"n_records": len(recs), "unparseable_or_empty": bad})
    save_house_cache(ref, max_dim=lego.GRID, cache_dir=str(run), name="heldout_matched",
                     report={"note": "held-out ST2B structure each prompt was drawn from"})

    sec = np.array([g["seconds"] for g in recs])
    nb = np.array([g["n_bricks"] for g in recs])
    rej = np.array([g["n_rejections"] for g in recs])
    regen = np.array([g["n_regenerations"] for g in recs if not g.get("timed_out")])
    reasons: dict = {}
    for g in recs:
        for k, v in g["rejection_reasons"].items():
            reasons[k] = reasons.get(k, 0) + v
    summary = {
        "n": len(recs), "n_scored": len(gen), "unparseable_or_empty": bad,
        "n_timed_out": int(sum(bool(g.get("timed_out")) for g in recs)),
        "seconds_median": float(np.median(sec)), "seconds_mean": float(sec.mean()),
        "seconds_total": float(sec.sum()),
        "structures_per_hour": float(3600 * len(recs) / sec.sum()),
        "bricks_median": float(np.median(nb)), "bricks_mean": float(nb.mean()),
        "bricks_per_second": float(nb.sum() / sec.sum()),
        "rejections_per_structure_mean": float(rej.mean()),
        "rejections_per_structure_median": float(np.median(rej)),
        "rejections_per_accepted_brick": float(rej.sum() / max(nb.sum(), 1)),
        "rejection_reasons": reasons,
        "rollbacks_median": float(np.median(regen)), "rollbacks_mean": float(regen.mean()),
        "frac_any_rollback": float((regen > 0).mean()),
        "frac_hit_rollback_cap": float((regen >= 100).mean()),
        "peak_vram_gb_max": float(max(g.get("peak_vram_gb", 0) for g in recs)),
    }
    (run / "gen_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if not args.no_render:
        from PIL import Image, ImageDraw
        from blockgen.renderer.grid import render_grid
        k, cols = min(args.fig_pairs, len(gen)), 8
        tiles = []
        for r0 in range(0, k, cols):
            tiles += gen[r0:r0 + cols] + ref[r0:r0 + cols]
        img = render_grid(tiles, cols=cols, tile_px=192, verbose=False)
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        for r in range(0, (len(tiles) + cols - 1) // cols):
            d.text((6, 4 + r * 194), "BrickGPT" if r % 2 == 0 else "held-out real",
                   fill=(200, 0, 0))
        pil.save(run / "samples.png")
        print(f"-> {run / 'samples.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
