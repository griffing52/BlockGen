"""Two-seed summary of a probe3d pair of runs (notes §65).

Reads results.json of two runs (same script, different `--seed`), prints a
markdown table with both seeds' z-scores per (row, arm), the raw difference on
KID/MMD rows, and the agreement statistics (`probe3d.agreement`), and writes
it to `seeds.md` in both run dirs.

    .venv/bin/python scripts/summarize_probe3d.py RUN_SEED0 RUN_SEED1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blockgen.eval.bench import probe3d as p3


def main() -> None:
    a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
    A = json.loads((a_dir / "results.json").read_text())
    B = json.loads((b_dir / "results.json").read_text())
    ra, rb = A["results"], B["results"]
    rungs = [k for k in next(iter(ra.values()))["sep"]]
    ag = p3.agreement(ra, rb, rungs)
    ag_3d = p3.agreement({k: v for k, v in ra.items() if k.split("_")[0] in
                          ("uni3d", "ulip2", "cd", "emd")}, rb, rungs)
    lines = [f"# Two seeds: {a_dir.name} (seed {A['seed']}) vs {b_dir.name} (seed {B['seed']})",
             "", "Cells: z seed A / z seed B; (raw difference, seed A) on KID/MMD rows.", "",
             f"Agreement over all {ag['cells']} cells: Pearson {ag['pearson']:.3f}, Spearman "
             f"{ag['spearman']:.3f}, median |dz| {ag['median_abs_dz']:.2f}, blind/not-blind "
             f"(|z|<2) verdict agrees on {100 * ag['verdict_agree']:.0f}%.",
             f"3D-native rows only ({ag_3d['cells']} cells): Pearson {ag_3d['pearson']:.3f}, "
             f"median |dz| {ag_3d['median_abs_dz']:.2f}, verdict agrees "
             f"{100 * ag_3d['verdict_agree']:.0f}%.", "",
             "| metric | " + " | ".join(rungs) + " |", "|" + "---|" * (len(rungs) + 1)]
    for m in ra:
        if m not in rb:
            continue
        raw = any(t in m for t in ("_kid", "_mmd"))
        cells = []
        for v in rungs:
            c = f"{ra[m]['sep'][v]:+.1f} / {rb[m]['sep'][v]:+.1f}"
            if raw:
                c += f" ({ra[m]['raw_delta'][v]:+.3g})"
            cells.append(c)
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    for d in (a_dir, b_dir):
        (d / "seeds.md").write_text(text)
    out = {"agreement_all": ag, "agreement_3d": ag_3d}
    for d in (a_dir, b_dir):
        (d / "seeds.json").write_text(json.dumps(out, indent=2) + "\n")
    print(text)


if __name__ == "__main__":
    main()
