"""The VLM judge's system ranking against the n=256 leaderboard of record.

The judge run (outputs/run_20261102_214026_vlm_judge, notes §67) correlated its
Bradley-Terry strengths with the n=128 board of 0925. The leaderboard of record
is now the n=256 rerun, which scores the same eleven arms. This recomputes the
Spearman correlations against it without touching the judge run (no API calls,
no re-analysis of its trials). Arms the n=256 board lacks are dropped and named.

    .venv/bin/python scripts/vlm_vs_leaderboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

JUDGE = ROOT / "outputs" / "run_20261102_214026_vlm_judge" / "analysis.json"
BOARDS = {
    "n128_0925": ROOT / "outputs" / "run_20260925_002631_bench_canonical-split_fixed_transparency_cull",
    "n256_1103": ROOT / "outputs" / "run_20261103_090315_bench_n_256_leaderboard_of_record",
}
OUT = ROOT / "outputs" / "figstage" / "vlm_vs_leaderboard.json"


def main() -> None:
    from blockgen.eval.bench.vlm_judge import leaderboard_metrics

    bt = json.loads(JUDGE.read_text())["system"]["bradley_terry"]
    out = {"judge": JUDGE.parent.name, "boards": {}}
    for key, run in BOARDS.items():
        met = leaderboard_metrics(run)
        arms = [a for a in bt if a in met and met[a].get("blockscore") is not None]
        row = {"run": run.name, "arms": arms,
               "dropped": sorted(set(bt) - set(arms))}
        strength = [bt[a]["strength"] for a in arms]
        for m in ("blockscore", "mv_kid", "v_kid"):
            # BT strength is higher-better, the metrics lower-better.
            rho = spearmanr(strength, [-met[a][m] for a in arms]).statistic
            row[f"spearman_{m}"] = float(rho)
        order = sorted(arms, key=lambda a: -bt[a]["strength"])
        row["real_rank_by_judge"] = order.index("real_test") + 1
        out["boards"][key] = row
        print(key, json.dumps(row))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
