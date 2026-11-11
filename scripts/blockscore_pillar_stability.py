"""BlockScore worst-pillar identity at n=128 (protocol run) vs n=256 (leaderboard), and
BlockScore on the n-free effect-size unit.

z unit (the board's): every pillar is (v - v_real) / sd, BlockScore = max pillar, computed
by `composite.score_arm` from each card's own scorecard.json with its own calibration.
Gates are switched off so disqualified arms still get their pillars (reported separately).

Effect-size unit (n-free; stats_fixes, Supp. D.5): per pillar the arm's distance as a
fraction of Random-Voxels' distance under the same statistic and card:
  appearance, geometry (unbiased MMD^2):  sqrt(max(KID_arm, 0)) / sqrt(KID_RV)
  palette (JSD):                           sqrt(max(JSD_arm - JSD_real, 0)) / sqrt(JSD_RV - JSD_real)
  coherence, geometry_scalars (RMS W1):    max(rms_arm - rms_real, 0) / (rms_RV - rms_real)
BlockScore_e = max over pillars.

    .venv/bin/python scripts/blockscore_pillar_stability.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blockgen.eval.bench import composite as C  # noqa: E402
from blockgen.eval.bench import scorecard as sc  # noqa: E402
from blockgen.utils.runs import new_run_dir  # noqa: E402

CARDS = {128: "outputs/run_20260925_002631_bench_canonical-split_fixed_transparency_cull",
         256: "outputs/run_20261103_090315_bench_n_256_leaderboard_of_record"}
STRUCT = ("coherence", "geometry_scalars")
ANCHOR = "uniform_random"


def to_metric(d):
    if isinstance(d, dict) and "value" in d and "direction" in d:
        v = d["value"]
        if v is None:
            return sc.Metric(value=None, gate_failed=d.get("gate_failed") or "skipped",
                             direction=d["direction"])
        ci = d.get("ci") or [float("nan")] * 2
        ci = [float("nan") if x is None else x for x in ci]
        return sc.Metric(value=float(v), ci=tuple(ci), direction=d["direction"])
    return d


def load(card):
    s = json.loads((Path(card) / "scorecard.json").read_text())
    arms = {}
    for name, a in s["arms"].items():
        arms[name] = {b: ({k: to_metric(v) for k, v in blk.items()} if isinstance(blk, dict)
                          else blk) for b, blk in a.items()}
    return s, arms


def zscores(s, arms, gated):
    cal = C.Calibration.from_arm(arms["real_test"], "real_test",
                                 noise_sd=s["context"].get("noise_floor_sd"))
    gates = C.GATES
    if not gated:
        C.GATES = ()
    try:
        rows = {n: C.score_arm(n, a, cal) for n, a in arms.items()}
    finally:
        C.GATES = gates
    return rows


def rms(block):
    return C._rms([e.get("w1_norm") for e in block.values() if isinstance(e, dict)])


def effects(raw):
    """raw = scorecard.json arms (plain dicts)."""
    def v(a, b, k):
        return raw[a][b][k]["value"]
    out = {}
    real, rv = "real_test", ANCHOR
    for a in raw:
        p = {}
        for pil, (b, k) in (("appearance", ("realism", "mv_dino_kid")),
                            ("geometry", ("realism", "geom_kid"))):
            p[pil] = float(np.sqrt(max(v(a, b, k), 0)) / np.sqrt(v(rv, b, k)))
        j, jr, jv = (v(a, "dataset_stats", "palette_jsd_exact"),
                     v(real, "dataset_stats", "palette_jsd_exact"),
                     v(rv, "dataset_stats", "palette_jsd_exact"))
        p["palette"] = float(np.sqrt(max(j - jr, 0)) / np.sqrt(jv - jr))
        for b in STRUCT:
            r, rr, rvv = rms(raw[a][b]), rms(raw[real][b]), rms(raw[rv][b])
            p[b] = float(max(r - rr, 0) / (rvv - rr))
        w = max(p, key=p.get)
        w2 = max(("appearance", "geometry"), key=p.get)
        out[a] = {"pillars": p, "worst": w, "score": p[w], "worst2": w2, "score2": p[w2]}
    return out


def main():
    res = {"cards": CARDS, "n": {}}
    for n, card in CARDS.items():
        s, arms = load(card)
        zg = zscores(s, arms, gated=True)
        zu = zscores(s, arms, gated=False)
        e = effects(s["arms"])
        res["n"][n] = {a: {"gated_status": zg[a].status, "dq": zg[a].disqualified,
                           "z_pillars": zu[a].pillars, "z_worst": zu[a].worst,
                           "z_score": zu[a].score, "e_pillars": e[a]["pillars"],
                           "e_worst": e[a]["worst"], "e_score": e[a]["score"],
                           "e2_worst": e[a]["worst2"], "e2_score": e[a]["score2"]}
                       for a in arms}
    A128, A256 = res["n"][128], res["n"][256]
    common = [a for a in A256 if a in A128 and a != "real_test"]
    flips = [a for a in common if A128[a]["z_worst"] != A256[a]["z_worst"]]
    res["common_arms"] = common
    res["worst_pillar_flips_z"] = flips
    res["worst_pillar_flips_e"] = [a for a in common if A128[a]["e_worst"] != A256[a]["e_worst"]]
    res["z_vs_e_worst_disagree"] = {n: [a for a in res["n"][n] if a != "real_test" and
                                        res["n"][n][a]["z_worst"] != res["n"][n][a]["e_worst"]]
                                    for n in CARDS}
    corr = {}
    for n in CARDS:
        R = res["n"][n]
        for subset, keep in (("all_arms", lambda a: a != "real_test"),
                             ("ranked_by_card_gates", lambda a: a != "real_test"
                              and R[a]["gated_status"] == "ok")):
            aa = [a for a in R if keep(a)]
            zs = [R[a]["z_score"] for a in aa]
            for unit in ("e_score", "e2_score"):
                es = [R[a][unit] for a in aa]
                corr[f"z_vs_{unit[:-6]}_n{n}_{subset}"] = {
                    "arms": aa, "spearman": float(spearmanr(zs, es).correlation),
                    "kendall": float(kendalltau(zs, es).correlation)}
    res["worst_pillar_flips_e2"] = [a for a in common
                                    if A128[a]["e2_worst"] != A256[a]["e2_worst"]]
    res["z_vs_e2_worst_disagree"] = {n: [a for a in res["n"][n] if a != "real_test" and
                                         res["n"][n][a]["z_worst"] != res["n"][n][a]["e2_worst"]]
                                     for n in CARDS}
    for unit in ("z_score", "e_score", "e2_score"):
        zs = [A128[a][unit] for a in common]
        zz = [A256[a][unit] for a in common]
        corr[f"{unit}_n128_vs_n256"] = {"spearman": float(spearmanr(zs, zz).correlation),
                                        "arms": common}
    res["rank_correlation"] = corr
    run = new_run_dir("blockscore_pillar_stability")
    shutil.copy(__file__, run / Path(__file__).name)
    (run / "results.json").write_text(json.dumps(res, indent=2) + "\n")

    L = ["# BlockScore worst pillar at n=128 vs n=256, and BlockScore on the effect-size unit", "",
         f"Cards: n=128 `{CARDS[128]}`, n=256 `{CARDS[256]}`. Pillars recomputed by "
         "`composite.score_arm` from each card's scorecard.json with gates off (DQ arms keep "
         "their pillars; the gated status is the last column). e = effect size as a fraction of "
         "Random-Voxels (see script docstring). Script: blockscore_pillar_stability.py.", "",
         "| arm | z worst n=128 | BS z 128 | z worst n=256 | BS z 256 | e worst 128 | BS e 128 | "
         "e worst 256 | BS e 256 | e2 worst 128 / 256 | BS e2 128 / 256 | card gates 128 / 256 |",
         "|---|---|--:|---|--:|---|--:|---|--:|---|---|---|"]
    allarms = list(A256) + [a for a in A128 if a not in A256]
    for a in allarms:
        r1, r2 = A128.get(a), A256.get(a)
        c = lambda r, k, f="{:.2f}": (f.format(r[k]) if r and r[k] is not None else "--")
        L.append(f"| {a} | {c(r1,'z_worst','{}')} | {c(r1,'z_score')} | {c(r2,'z_worst','{}')} | "
                 f"{c(r2,'z_score')} | {c(r1,'e_worst','{}')} | {c(r1,'e_score','{:.3f}')} | "
                 f"{c(r2,'e_worst','{}')} | {c(r2,'e_score','{:.3f}')} | "
                 f"{c(r1,'e2_worst','{}')} / {c(r2,'e2_worst','{}')} | "
                 f"{c(r1,'e2_score','{:.3f}')} / {c(r2,'e2_score','{:.3f}')} | "
                 f"{r1['gated_status'] if r1 else '--'} / {r2['gated_status'] if r2 else '--'} |")
    L += ["", f"Worst-pillar (z) changes between n=128 and n=256: {len(flips)} of {len(common)} "
          f"common non-anchor arms {flips}.",
          f"Worst-pillar (e) changes: {len(res['worst_pillar_flips_e'])} {res['worst_pillar_flips_e']}.",
          f"Worst-pillar (e2 = max of appearance and geometry) changes: "
          f"{res['worst_pillar_flips_e2']}; z-worst vs e2-worst disagree: "
          f"n=128 {res['z_vs_e2_worst_disagree'][128]}, n=256 {res['z_vs_e2_worst_disagree'][256]}.",
          "Card gates = the scorecard's own novelty gates; the within-set near-twin gate is "
          "applied on the twin-gate rescore board, not here.",
          f"z-worst vs e-worst disagree: n=128 {res['z_vs_e_worst_disagree'][128]}; "
          f"n=256 {res['z_vs_e_worst_disagree'][256]}.", ""]
    for k, v in corr.items():
        L.append(f"- Spearman {k}: {v['spearman']:.3f}" + (f", Kendall {v['kendall']:.3f}"
                                                            if "kendall" in v else "")
                 + f" ({len(v['arms'])} arms)")
    (run / "results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("->", run)


if __name__ == "__main__":
    main()
