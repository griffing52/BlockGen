"""Position-debiased reanalysis of the GPT-4o pairwise judge (no new API calls).

Reads records.jsonl of the judge run of record and reports, per condition:
  * raw P(prefers real), P(first slot), order agreement;
  * P(prefers real | order-consistent pair): the pairs whose two presentation orders
    pick the same build (the content-decided pairs under a mixture reading);
  * a Bradley-Terry fit with a position term, logit P(slot-1 picked) =
    delta + (b_slot1 - b_slot2), fit by maximum likelihood over trials; for a two-arm
    condition the debiased P(real) is sigmoid(b_real - b_probe);
  * 95% cluster-bootstrap intervals over source houses (both orders and every pair of a
    house resampled together).
Also refits the eleven-arm system Bradley-Terry with a position term and reports real's
rank and the Spearman with the leaderboard metrics.

    .venv/bin/python scripts/vlm_judge_debias.py [--run outputs/run_20261102_214026_vlm_judge]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blockgen.utils.runs import new_run_dir  # noqa: E402

TWO_ARM = ["interior_ext", "interior_cut", "interior_cutq", "hollow_ext", "hollow_cut",
           "catch", "gabled_patchwork"]
VIEW = {"interior_ext": "exterior", "interior_cut": "roof off", "interior_cutq": "quadrant cut",
        "hollow_ext": "exterior", "hollow_cut": "roof off", "catch": "exterior",
        "gabled_patchwork": "exterior"}


def house(build_key: str) -> str:
    # "real3dc@solidify#170@ext4" -> "170"; system builds keep their arm in the key
    return build_key.split("#")[1].split("@")[0]


def fit_bt_pos(trials, arms, l2=1e-3):
    """trials: list of (i_slot1, i_slot2, picked_slot1: bool). Returns (b, delta)."""
    I = np.array([t[0] for t in trials])
    J = np.array([t[1] for t in trials])
    y = np.array([t[2] for t in trials], float)
    K = len(arms)

    def nll(th):
        b = np.concatenate([[0.0], th[:K - 1]])
        d = th[K - 1]
        z = d + b[I] - b[J]
        ll = y * z - np.logaddexp(0, z)
        return -ll.sum() + l2 * (th ** 2).sum()

    th = minimize(nll, np.zeros(K), method="L-BFGS-B").x
    b = np.concatenate([[0.0], th[:K - 1]])
    return b - b.mean(), th[K - 1]


def two_arm_stats(recs, real_arm):
    by_pair = defaultdict(list)
    for r in recs:
        by_pair[r["pair"]].append(r)
    pairs = [v for v in by_pair.values() if len(v) == 2]
    trials = []
    for r in recs:
        real_first = (r["a"].split("#")[0] == real_arm)
        picked_first = r["choice"] == "A"
        trials.append((0 if real_first else 1, 1 if real_first else 0, picked_first))
    p_real = np.mean([r["picked_arm"] == real_arm for r in recs])
    p_first = np.mean([r["choice"] == "A" for r in recs])
    cons = [v for v in pairs if v[0]["picked_arm"] == v[1]["picked_arm"]]
    p_cons = np.mean([v[0]["picked_arm"] == real_arm for v in cons]) if cons else np.nan
    # per-order rates
    rf = [r["picked_arm"] == real_arm for r in recs if r["a"].split("#")[0] == real_arm]
    rs = [r["picked_arm"] == real_arm for r in recs if r["a"].split("#")[0] != real_arm]
    b, d = fit_bt_pos(trials, ["real", "other"])
    p_bt = 1 / (1 + np.exp(-(b[0] - b[1])))
    return {"n_trials": len(recs), "n_pairs": len(pairs), "p_real": float(p_real),
            "p_first": float(p_first),
            "order_agreement": len(cons) / len(pairs) if pairs else np.nan,
            "n_consistent": len(cons), "p_real_consistent": float(p_cons),
            "p_real_when_first": float(np.mean(rf)), "p_real_when_second": float(np.mean(rs)),
            "bt_delta_position": float(d), "bt_p_real": float(p_bt)}


def boot_two_arm(recs, real_arm, rng, B=2000):
    # cluster = the source house of the real-arm build (both orders and every pair of a
    # house together); conditions without a shared house fall back to the build pair
    by_house = defaultdict(list)
    for r in recs:
        k = r["a"] if r["a"].split("#")[0] == real_arm else r["b"]
        by_house[house(k) if real_arm.startswith("real") else r["pair"]].append(r)
    hs = list(by_house)
    out = defaultdict(list)
    for _ in range(B):
        pick = rng.integers(len(hs), size=len(hs))
        rr = []
        for k, h in enumerate(pick):
            for r in by_house[hs[h]]:
                rr.append({**r, "pair": f"{r['pair']}#{k}"})
        s = two_arm_stats(rr, real_arm)
        for key in ("p_real", "p_real_consistent", "bt_p_real"):
            out[key].append(s[key])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for k, v in out.items()}


def system(recs, rng, B=1000):
    arms = sorted({r["arm_a"] for r in recs} | {r["arm_b"] for r in recs})
    ix = {a: i for i, a in enumerate(arms)}

    def trials_of(rs):
        t = []
        for r in rs:
            s1 = r["a"].split("#")[0]
            s2 = r["b"].split("#")[0]
            t.append((ix[s1], ix[s2], r["choice"] == "A"))
        return t

    b, d = fit_bt_pos(trials_of(recs), arms)
    by_pair = defaultdict(list)
    for r in recs:
        by_pair[(r["condition"], r["pair"])].append(r)
    by_cond = defaultdict(list)
    for k, v in by_pair.items():
        by_cond[k[0]].append(v)
    boots = []
    for _ in range(B):
        rr = []
        for c, groups in by_cond.items():
            for g in rng.integers(len(groups), size=len(groups)):
                rr += groups[g]
        boots.append(fit_bt_pos(trials_of(rr), arms)[0])
    boots = np.array(boots)
    ranks = np.argsort(np.argsort(-boots, 1), 1) + 1
    return arms, b, d, boots, ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/run_20261102_214026_vlm_judge")
    ap.add_argument("--board", default="outputs/run_20261102_214026_vlm_judge/analysis.json")
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(Path(args.run) / "records.jsonl")]
    recs = [r for r in recs if r.get("valid")]
    rng = np.random.default_rng(0)
    out = {"source": args.run, "n_valid": len(recs), "conditions": {}}
    for c in TWO_ARM:
        rc = [r for r in recs if r["condition"] == c]
        real_arm = rc[0]["arm_a"]
        s = two_arm_stats(rc, real_arm)
        s["ci"] = boot_two_arm(rc, real_arm, rng)
        s["view"] = VIEW[c]
        s["arm_a"], s["arm_b"] = rc[0]["arm_a"], rc[0]["arm_b"]
        out["conditions"][c] = s
        print(c, json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                             for k, v in s.items() if k != "ci"}), s["ci"])

    # pooled rows: both interior probes per view (pair keys made unique per condition)
    pools = {"pooled_exterior": ("interior_ext", "hollow_ext"),
             "pooled_roof_off": ("interior_cut", "hollow_cut")}
    pooled = {}
    for name, cs in pools.items():
        rc = [{**r, "pair": r["condition"] + "/" + r["pair"]} for r in recs
              if r["condition"] in cs]
        st = two_arm_stats(rc, "real3dc")
        st["ci"] = boot_two_arm(rc, "real3dc", rng)
        from scipy.stats import binomtest
        k = round(st["p_real_consistent"] * st["n_consistent"])
        st["binom_p_consistent_vs_half"] = float(binomtest(k, st["n_consistent"]).pvalue)
        pooled[name] = st
        print(name, st)
    out["pooled"] = pooled

    sysrecs = [r for r in recs if r["condition"].startswith("system:")]
    arms, b, d, boots, ranks = system(sysrecs, rng)
    an = json.loads(Path(args.board).read_text())
    old = an.get("system", {})
    out["system"] = {"arms": arms, "bt": dict(zip(arms, map(float, b))),
                     "delta_position": float(d),
                     "bt_ci": {a: [float(np.percentile(boots[:, i], 2.5)),
                                   float(np.percentile(boots[:, i], 97.5))]
                               for i, a in enumerate(arms)},
                     "rank": {a: int(r) for a, r in zip(arms, np.argsort(np.argsort(-b)) + 1)},
                     "rank_ci": {a: [int(np.percentile(ranks[:, i], 2.5)),
                                     int(np.percentile(ranks[:, i], 97.5))]
                                 for i, a in enumerate(arms)}}
    # metrics of the leaderboard as used in the original analysis
    met = {a: v for a, v in old.get("bradley_terry", {}).items()}
    corr = {}
    if met:
        for m in ("blockscore", "mv_kid", "v_kid"):
            aa = [a for a in arms if a in met and met[a].get(m) is not None]
            if len(aa) >= 3:
                rho = spearmanr([out["system"]["bt"][a] for a in aa],
                                [-met[a][m] for a in aa]).correlation
                corr[m] = {"spearman": float(rho), "arms": len(aa)}
    out["system"]["spearman_vs_board"] = corr
    out["system"]["old_bt"] = {a: v["strength"] for a, v in met.items()}
    out["system"]["old_rank"] = {a: i + 1 for i, a in enumerate(
        sorted(met, key=lambda a: -met[a]["strength"]))}
    print(json.dumps(out["system"]["rank"]), d, corr)
    run = new_run_dir("vlm_judge_debias")
    shutil.copy(__file__, run / Path(__file__).name)
    (run / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    print("->", run)


if __name__ == "__main__":
    main()
