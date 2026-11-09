"""The pooled-face probe: what each level of a fixed coarse-to-fine hierarchy keeps.

Step 1 of the 2D next-scale plan (research.md H.7): before any model, pool real
face images to coarser grids and look. A face (`blockgen.corpus.facade`) is a
(U, V) grid of cells, each see-through or `(depth, block)`. For every pooling
factor f in {2, 4, 8, 16} the face is reduced to ceil(U/f) x ceil(V/f) cells under
a rule, upsampled back by repetition, and compared with the original: mask IoU,
precision and recall; depth exactness and error on cells filled in both; block
agreement; and the face statistics the sheets report (relief jump rate, 2D
components, holes) on the upsampled face, so a level can be read as "a facade"
or "noise" by number as well as by eye.

The three rules (``any``, ``majority``, ``center``) are `facade.pool`'s: dilate,
vote, or decimate.

Sheets: ``sheet_<rule>.png`` has one real test house per row and the levels
f = 1, 2, 4, 8, 16 as columns, drawn at the face's own size so the eye sees what
a model at that level would be asked to draw. ``stats.json`` and ``stats.md``
hold the per-face medians and means per rule and level.

    PYTHONPATH=. python scripts/facade_pool_probe.py                 # roof plan, test split
    PYTHONPATH=. python scripts/facade_pool_probe.py --face -z       # the front
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw

from blockgen.corpus import facade as fc
from blockgen.eval.bench import splits
from blockgen.eval.facade_stats import face_agreement as compare, one_face_stats as single_face_stats
from blockgen.utils.runs import new_run_dir

RULES = fc.POOL_RULES
FACTORS = (1, 2, 4, 8, 16)


def pool_face(f: fc.Facade, factor: int, rule: str) -> fc.Facade:
    """`f` at level `factor` under `rule` (`facade.pool`), repeated back to its own size."""
    if factor == 1:
        return f
    return fc.upsample(fc.pool(f, factor, rule), factor, f.depth.shape)


def draw_sheet(rows: List[List[fc.Facade]], labels: List[str], scale: int, gap: int = 6,
               header: int = 14) -> np.ndarray:
    pics = [[np.repeat(np.repeat(fc.face_picture(f), scale, 0), scale, 1) for f in r] for r in rows]
    col_w = [max(p[c].shape[1] for p in pics) for c in range(len(labels))]
    row_h = [max(p.shape[0] for p in r) for r in pics]
    W = sum(col_w) + gap * (len(labels) + 1)
    H = header + sum(row_h) + gap * (len(rows) + 1)
    out = np.full((H, W, 3), 255, np.uint8)
    y = header + gap
    for r, h in zip(pics, row_h):
        x = gap
        for c, p in enumerate(r):
            out[y:y + p.shape[0], x:x + p.shape[1]] = p
            x += col_w[c] + gap
        y += h + gap
    img = Image.fromarray(out)
    dr = ImageDraw.Draw(img)
    x = gap
    for c, lab in enumerate(labels):
        dr.text((x, 1), lab, fill=(0, 0, 0))
        x += col_w[c] + gap
    return np.asarray(img)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", default="houses_32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--face", default="+y", help="face to probe: +y roof plan, -z front, ...")
    ap.add_argument("--rows", type=int, default=12, help="houses per sheet")
    ap.add_argument("--scale", type=int, default=5)
    ap.add_argument("--out", default=None, help="run dir; default outputs/run_<stamp>_facade_pool_probe")
    args = ap.parse_args()

    split = splits.load_split(args.corpus, args.seed)
    houses = [s for s in splits.split_structures(split, args.split) if s.occupied_mask.any()]
    faces = [fc.project(s, args.face) for s in houses]
    faces = [f for f in faces if f.mask.any()]
    out = Path(args.out) if args.out else new_run_dir("facade_pool_probe")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[probe] {len(faces)} {args.face} faces from {args.corpus}:{args.split}; writing {out}", flush=True)

    stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    real_stats = [single_face_stats(f) for f in faces]
    stats["real"] = {k: float(np.mean([r[k] for r in real_stats])) for k in real_stats[0]}
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(faces), min(args.rows, len(faces)), replace=False)
    for rule in RULES:
        stats[rule] = {}
        rows = []
        for fac in FACTORS:
            pooled = [pool_face(f, fac, rule) for f in faces]
            cmp = [compare(o, p) for o, p in zip(faces, pooled)]
            fs = [single_face_stats(p) for p in pooled]
            agg = {k: float(np.nanmean([c[k] for c in cmp])) for k in cmp[0]}
            agg.update({k: float(np.mean([s[k] for s in fs])) for k in fs[0]})
            agg["empty_faces"] = float(np.mean([not p.mask.any() for p in pooled]))
            stats[rule][f"f{fac}"] = agg
        for i in pick:
            rows.append([pool_face(faces[i], fac, rule) for fac in FACTORS])
        img = draw_sheet(rows, [f"f={fac}" if fac > 1 else "real" for fac in FACTORS], args.scale)
        Image.fromarray(img).save(out / f"sheet_{rule}.png")

    (out / "stats.json").write_text(json.dumps(
        {"corpus": args.corpus, "split": args.split, "seed": args.seed, "face": args.face,
         "n_faces": len(faces), "rules": RULES, "factors": FACTORS, "stats": stats}, indent=2))

    cols = ["mask_iou", "mask_recall", "mask_precision", "depth_exact", "depth_mae", "block_agree",
            "jumps", "comps", "holes", "lone", "empty_faces"]
    lines = [f"# Pooled {args.face} faces, {args.corpus}:{args.split}, n={len(faces)} (means over faces)", "",
             "| rule | f | " + " | ".join(cols) + " |", "|---|--:|" + "--:|" * len(cols)]
    r = stats["real"]
    lines.append("| real | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00 | 1.000 | "
                 f"{r['jumps']:.3f} | {r['comps']:.2f} | {r['holes']:.3f} | {r['lone']:.4f} | 0.000 |")
    for rule in RULES:
        for fac in FACTORS[1:]:
            a = stats[rule][f"f{fac}"]
            lines.append(f"| {rule} | {fac} | " + " | ".join(
                f"{a[c]:.3f}" if c not in ("depth_mae", "comps") else f"{a[c]:.2f}" for c in cols) + " |")
    md = "\n".join(lines) + "\n"
    (out / "stats.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, ".")
    sys.exit(main())
