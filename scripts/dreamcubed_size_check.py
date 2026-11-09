"""How much of Dream-Cubed's V-KID gap is build size? (notes §69, results T85)

Dream-Cubed's extracted builds are about a third the size of a houses_32
house, and the 77-d descriptor carries one explicit size term,
`log1p(n_blocks)`. Two checks, both on the n=256 leaderboard of record's own
reference (the 399 validation builds) and the v1 descriptor that run pins:

* refit the standardiser without the size column and rescore (raw geom_kid);
* score against only the reference builds inside Dream-Cubed's own block-count
  interquartile range (standardiser unchanged).

The full-descriptor score is recomputed first and must equal the scorecard's
value, so the two variants are measured on exactly the scored features.

    .venv/bin/python scripts/dreamcubed_size_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUN = ROOT / "outputs" / "run_20261103_090315_bench_n_256_leaderboard_of_record"
OUT = ROOT / "outputs" / "figstage" / "dreamcubed_size.json"
ARMS = ("dreamcubed", "canon_oriented", "native_oriented")


def main() -> None:
    from blockgen.curation.houses import load_structures_from_cache
    from blockgen.eval.bench import geometry as G, splits

    card = json.loads((RUN / "scorecard.json").read_text())
    G.set_descriptor_version(card["context"]["descriptor_version"])
    split = splits.load_split(card["context"]["corpus"], int(card["context"]["seed"]))
    val = [s.crop_to_non_air() for s in splits.split_structures(split, "val")]
    ref_d = G.descriptor(val)
    saved = np.load(RUN / "features.npz")
    std = G.Standardizer.fit(ref_d)
    assert np.allclose(std(ref_d), saved["ref::geom"], atol=1e-9), "reference mismatch"
    size_col = ref_d.shape[1] - 1                     # log1p(n_blocks), last scalar
    ref_n = np.expm1(ref_d[:, size_col])

    out = {"run": RUN.name, "descriptor_dim": int(ref_d.shape[1]), "arms": {}}
    for arm in ARMS:
        src = ROOT / card["arms"][arm]["meta"]["source"]
        structs = [s.crop_to_non_air() for s in load_structures_from_cache(str(src))[0]]
        d = G.descriptor(structs)
        full = G.geom_kid(std(d), std(ref_d))
        recorded = card["arms"][arm]["realism"]["geom_kid"]["value"]
        assert abs(full - recorded) < 1e-6, (arm, full, recorded)
        keep = [j for j in range(ref_d.shape[1]) if j != size_col]
        std_ns = G.Standardizer.fit(ref_d[:, keep])
        no_size = G.geom_kid(std_ns(d[:, keep]), std_ns(ref_d[:, keep]))
        n_blocks = np.expm1(d[:, size_col])
        q1, q3 = np.percentile(n_blocks, [25, 75])
        inside = (ref_n >= q1) & (ref_n <= q3)
        matched = G.geom_kid(std(d), std(ref_d[inside]))
        out["arms"][arm] = {
            "n": len(structs), "geom_kid_full": full, "geom_kid_no_size": no_size,
            "size_share_upper_bound": float(max(0.0, 1.0 - no_size / full)),
            "median_blocks": float(np.median(n_blocks)),
            "iqr_blocks": [float(q1), float(q3)],
            "ref_inside_iqr": int(inside.sum()),
            "geom_kid_size_matched_ref": matched,
        }
        print(arm, json.dumps(out["arms"][arm]))
    out["ref_median_blocks"] = float(np.median(ref_n))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
