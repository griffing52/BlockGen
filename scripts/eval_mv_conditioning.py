"""Does MV conditioning carry structure-SPECIFIC information?

For each generated sample, score it against (a) the ground-truth structure it was
conditioned on ["paired"] and (b) the other ground-truth structures ["shuffled"].
If conditioning works, paired must beat shuffled. The paired/shuffled ratio separates
"the model learned generic houses" from "the model used THIS structure's views".

Two channels, reported separately because T15 found they dissociate:
  palette  -- cosine similarity of block-type histograms (material transfer)
  geometry -- IoU of occupancy masks, aligned at min-corner (shape transfer)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from blockgen.curation.houses import load_house_structures
from blockgen.eval.validity import n_components
from blockgen.export.minecraftace import load_piece_vocab, piece_records_to_structure


def palette_vec(s, n_types=512):
    ids = s.block_ids[s.occupied_mask]
    v = np.bincount(ids.ravel(), minlength=n_types).astype(np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cos(a, b):
    return float(np.dot(a, b))


def occ_iou(a, b):
    """IoU of occupancy, both cropped to content and aligned at the min corner."""
    A, B = a.crop_to_non_air().occupied_mask, b.crop_to_non_air().occupied_mask
    shape = np.maximum(A.shape, B.shape)
    pa = np.zeros(shape, bool); pa[:A.shape[0], :A.shape[1], :A.shape[2]] = A
    pb = np.zeros(shape, bool); pb[:B.shape[0], :B.shape[1], :B.shape[2]] = B
    union = (pa | pb).sum()
    return float((pa & pb).sum() / union) if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--max-dim", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sdir, ddir = Path(args.samples_dir), Path(args.dataset_dir)
    manifest = json.loads((sdir / "manifest.json").read_text())
    cv = load_piece_vocab(str(next(ddir.glob("*_piece_vocab.json"))))

    # ground truth: houses cache row i == id h{i:05d}
    gt_all, _ = load_house_structures(max_dim=args.max_dim)
    gt = {sid: gt_all[int(sid[1:])] for sid in manifest}

    rows = []
    for sid, entry in manifest.items():
        for s in entry["samples"]:
            rec = np.load(sdir / "npy" / s["file"])
            if not len(rec):
                continue
            gen = piece_records_to_structure(rec, cv)
            gv = palette_vec(gen)

            paired_pal = cos(gv, palette_vec(gt[sid]))
            paired_geo = occ_iou(gen, gt[sid])
            others = [o for o in manifest if o != sid]
            shuf_pal = float(np.mean([cos(gv, palette_vec(gt[o])) for o in others]))
            shuf_geo = float(np.mean([occ_iou(gen, gt[o]) for o in others]))

            rows.append({
                "id": sid, "file": s["file"], "terminated": s["terminated"],
                "n_blocks": int(gen.occupied_mask.sum()),
                "gt_n_blocks": int(gt[sid].occupied_mask.sum()),
                "components": n_components(gen),
                "palette_paired": paired_pal, "palette_shuffled": shuf_pal,
                "geom_paired": paired_geo, "geom_shuffled": shuf_geo,
            })

    def m(k):
        return float(np.mean([r[k] for r in rows]))

    summary = {
        "n_samples": len(rows),
        "terminated_frac": float(np.mean([r["terminated"] for r in rows])),
        "components_median": float(np.median([r["components"] for r in rows])),
        "palette_paired": m("palette_paired"), "palette_shuffled": m("palette_shuffled"),
        "geom_paired": m("geom_paired"), "geom_shuffled": m("geom_shuffled"),
    }
    summary["palette_lift"] = summary["palette_paired"] / max(summary["palette_shuffled"], 1e-9)
    summary["geom_lift"] = summary["geom_paired"] / max(summary["geom_shuffled"], 1e-9)

    print(json.dumps(summary, indent=2))
    print("\nper-sample:")
    for r in rows:
        print(f"  {r['file']:<18} blocks {r['n_blocks']:>5} (gt {r['gt_n_blocks']:>5})  "
              f"comp {r['components']:>3}  pal {r['palette_paired']:.3f}/{r['palette_shuffled']:.3f}  "
              f"geo {r['geom_paired']:.3f}/{r['geom_shuffled']:.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
