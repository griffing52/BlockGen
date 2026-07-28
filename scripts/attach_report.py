"""Morning report over attachment/growth runs.

Adds the metric the per-arm summaries cannot express: **bits per build**.

`bits/op` is NOT comparable across representations -- an ordering or tokenizer that
emits more, cheaper ops can win bits/op while describing the same build less
efficiently. Total description length (bits/op x ops/build) is representation-
agnostic and is the honest way to compare the growth model against raster AR, which
uses a different vocabulary and a different sequence length for the same structure.

Usage:
    python -m scripts.attach_report                    # newest attach_growth* runs
    python -m scripts.attach_report --runs outputs/run_X outputs/run_Y
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List


def load_runs(paths: List[Path]) -> List[dict]:
    rows = []
    for p in paths:
        summ = p / "summary.json"
        if not summ.exists():
            continue
        blob = json.loads(summ.read_text())
        for r in blob.get("results", []):
            if "val_bits_per_op" not in r:
                rows.append({"run": p.name, "ordering": r.get("ordering"),
                             "error": True})
                continue
            r = dict(r)
            r["run"] = p.name
            r["max_seq_len"] = blob.get("args", {}).get("max_seq_len")
            # Description length for a median build under this arm.
            r["bits_per_build"] = r["val_bits_per_op"] * r["seq_p50"]
            rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.runs:
        paths = [Path(p) for p in args.runs]
    else:
        paths = sorted(Path("outputs").glob("run_*attach_growth*"))
    rows = load_runs(paths)
    ok = [r for r in rows if not r.get("error")]
    if not ok:
        print("no completed arms found")
        return

    print(f"{'run':38s} {'ordering':20s} {'seq':>5s} {'bits/op':>8s} "
          f"{'bits/build':>11s} {'fit':>6s} {'occ_p50':>8s} {'valid':>6s} {'comps':>6s}")
    print("-" * 118)
    for r in sorted(ok, key=lambda r: r["val_bits_per_op"]):
        print(f"{r['run'][:38]:38s} {str(r['ordering'])[:20]:20s} "
              f"{str(r.get('max_seq_len','')):>5s} "
              f"{r['val_bits_per_op']:8.4f} {r['bits_per_build']:11.0f} "
              f"{100*r.get('frac_fit',0):5.1f}% {r.get('sample_occ_p50',0):8.0f} "
              f"{r.get('validity_learned_unmasked', r.get('validity_by_construction',0)):6.2f} "
              f"{r.get('sample_components_mean',0):6.2f}")

    for r in rows:
        if r.get("error"):
            print(f"  ERROR: {r['run']} / {r['ordering']}")

    print()
    print("bits/op    = held-out per-op NLL. NOT comparable across representations.")
    print("bits/build = bits/op x median ops/build. Representation-agnostic; this is")
    print("             the number to compare against raster AR (native_bpe).")
    print("valid      = LEARNED validity, sampled UNMASKED (SEED not banned after pos 0).")
    print("             comps > 1 means the model emitted extra SEEDs. Masking SEED would")
    print("             force valid=1.00 as a decode-time filter -- the artifact confound.")

    if args.out:
        Path(args.out).write_text(json.dumps(ok, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
