"""Item 3 report: leaderboard with and without the near-twin gate, the gate's rate on real
held-out draws and on the 40 rank-flip generators, and its threshold sensitivity.

Reads the twin-gate rescore board (scorecard.json with voxel_twin_* metrics) and the
twin_gate_audit run; writes results.json / results.md / samples.png into the audit run.

    .venv/bin/python scripts/twin_gate_report.py outputs/run_<...>_twin_gate_audit
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import blockscore_pillar_stability as bps  # noqa: E402
from blockgen.eval.bench import composite as C  # noqa: E402

RESCORE = Path("outputs/run_20261104_214146_bench_twin_gate_rescore")
VANILLA = {"vanilla_village_houses": 202.95, "vanilla_village_all": 194.98}


def board(gated_twin: bool):
    s, arms = bps.load(RESCORE)
    gates = C.GATES
    if not gated_twin:
        C.GATES = tuple(g for g in gates if g.key != "voxel_twin_rate")
    try:
        cal = C.Calibration.from_arm(arms["real_test"], "real_test",
                                     noise_sd=s["context"].get("noise_floor_sd"))
        rows = [C.score_arm(n, a, cal) for n, a in arms.items()]
    finally:
        C.GATES = gates
    ok = sorted([r for r in rows if r.status == "ok"], key=lambda r: r.score)
    return ([{"arm": r.arm, "score": r.score, "worst": r.worst, "rank": i + 1}
             for i, r in enumerate(ok)],
            {r.arm: r.disqualified for r in rows if r.status != "ok"})


def main():
    run = Path(sys.argv[1])
    shutil.copy(__file__, run / Path(__file__).name)
    a = json.loads((run / "results.json").read_text())
    with_g, dq_with = board(True)
    without_g, dq_without = board(False)
    bar = a["bar_n256"]
    real = a["real_draws"]
    fp = {n: [r["rates"]["0.8"] > bar for r in real if r["n"] == n] for n in (128, 256)}
    gens = a["generators"]
    gated = {k: v["rates"]["0.8"] for k, v in gens.items() if v["rates"]["0.8"] > bar}
    fam = {}
    for k, v in gens.items():
        fam.setdefault(v["family"], []).append(v["rates"]["0.8"])
    learned = {k: v for k, v in gens.items() if v["family"] in ("voxel_ar", "facade", "external")}
    out = {
        "bar": bar,
        "real_draws_max": {n: max(r["rates"]["0.8"] for r in real if r["n"] == n) for n in (128, 256)},
        "real_draws_mean": {n: float(np.mean([r["rates"]["0.8"] for r in real if r["n"] == n]))
                            for n in (128, 256)},
        "real_false_positive": {n: f"{sum(v)}/{len(v)}" for n, v in fp.items()},
        "generators_gated": gated,
        "n_generators": len(gens),
        "learned_max": max(v["rates"]["0.8"] for v in learned.values()),
        "learned_nonzero": sum(v["rates"]["0.8"] > 0 for v in learned.values()),
        "n_learned": len(learned),
        "learned_gated": {k: v["rates"]["0.8"] for k, v in learned.items()
                          if v["rates"]["0.8"] > bar},
        "family_rates": {f: [float(np.min(v)), float(np.median(v)), float(np.max(v))]
                         for f, v in fam.items()},
        "extra": a["extra"],
        "board_with_gate": with_g, "dq_with_gate": dq_with,
        "board_without_gate": without_g, "dq_without_gate": dq_without,
        "vanilla_templates_ungated_blockscore": VANILLA,
    }
    a["report"] = out
    (run / "results.json").write_text(json.dumps(a, indent=2) + "\n")

    L = ["# Near-twin gate audit", "",
         f"Bar (leaderboard of record, n=256): {bar:.4f} = real 0.0078 + 3 x 0.02. "
         "Twin = nearest other build in the same set at occupancy IoU >= t (D4 x translation).", "",
         "## Leaderboard with and without the gate (twin-gate rescore board, n=256)", "",
         "| rank with | arm | BlockScore | worst | rank without |", "|--:|---|--:|---|--:|"]
    rk = {r["arm"]: r["rank"] for r in without_g}
    for r in with_g:
        L.append(f"| {r['rank']} | {r['arm']} | {r['score']:.2f} | {r['worst']} | {rk[r['arm']]} |")
    for r in without_g:
        if r["arm"] not in {x["arm"] for x in with_g}:
            L.append(f"| DQ | {r['arm']} | {r['score']:.2f} | {r['worst']} | {r['rank']} |")
    L += ["", "Vanilla village templates (separate run, n=174/183), ungated BlockScore "
          "202.95 / 194.98: rank 3 without the gate (behind real and Dream-Cubed, ahead of the "
          "contaminated Voxel-AR at 229.29).", "",
          "## Real held-out draws (test, val, test+val; 12 per n)", "",
          "| n | mean rate | max | over bar |", "|--:|--:|--:|--:|"]
    for n in (128, 256):
        L.append(f"| {n} | {out['real_draws_mean'][n]:.4f} | {out['real_draws_max'][n]:.4f} | "
                 f"{out['real_false_positive'][n]} |")
    L += ["", "## The 40 rank-flip generators", "",
          "| arm | family | n | t=0.8 | t=0.9 | t=0.95 | gated |", "|---|---|--:|--:|--:|--:|---|"]
    for k, v in sorted(gens.items(), key=lambda kv: -kv[1]["rates"]["0.8"]):
        r = v["rates"]
        L.append(f"| {k} | {v['family']} | {v['n']} | {r['0.8']:.3f} | {r['0.9']:.3f} | "
                 f"{r['0.95']:.3f} | {'DQ' if r['0.8'] > bar else ''} |")
    L += ["", "## Extra arms", "", "| arm | n | t=0.8 | t=0.9 | t=0.95 |", "|---|--:|--:|--:|--:|"]
    for k, v in a["extra"].items():
        r = v["rates"]
        L.append(f"| {k} | {v.get('n', '')} | {r['0.8']:.3f} | {r.get('0.9', float('nan')):.3f} | "
                 f"{r.get('0.95', float('nan')):.3f} |")
    L += ["", f"Learned generators (voxel AR, facade, BrickNet): {out['n_learned']}, "
          f"{out['learned_nonzero']} with a nonzero rate, max {out['learned_max']:.4f}; "
          f"gated: {out['learned_gated']}.",
          f"All generators gated: {gated}."]
    (run / "results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    cols = {"voxel_ar": "#2b6cb0", "facade": "#38a169", "llm_program": "#c05621",
            "external": "#805ad5", "leaderboard": "#718096"}
    pts = [(k, v) for k, v in gens.items()] + [(k, v) for k, v in a["extra"].items()]
    pts.sort(key=lambda kv: kv[1]["rates"]["0.8"])
    for i, (k, v) in enumerate(pts):
        ax.bar(i, v["rates"]["0.8"], color=cols.get(v.get("family", "leaderboard"), "#718096"))
    rr = [r["rates"]["0.8"] for r in real]
    ax.axhline(bar, color="k", ls="--", lw=0.8, label=f"gate bar {bar:.3f}")
    ax.axhline(max(rr), color="#e53e3e", ls=":", lw=0.8, label=f"max of 24 real draws {max(rr):.3f}")
    ax.set_xticks(range(len(pts)))
    ax.set_xticklabels([k[:28] for k, _ in pts], rotation=90, fontsize=5)
    ax.set_ylabel("within-set twin rate (IoU>=0.8)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(run / "samples.png", dpi=140)


if __name__ == "__main__":
    main()
