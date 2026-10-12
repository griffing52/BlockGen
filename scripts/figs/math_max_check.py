"""Statistical audit of BlockScore (max over pillars) and of the noise-floor unit.

Owned by the `math_max` agent (see deliverables/paper/iclr/FIGBRIEF.md). Pure
computation: it prints a report and writes a JSON of every number quoted in
`deliverables/paper/iclr/math/math_max.md`. It does not touch the paper, the
run of record, or any module in the tree.

Four measurements, one per item of the brief:

A. `theory`  -- E[max] and sd[max] of k equicorrelated unit Gaussians, exactly
   (via the one-factor representation) and by Monte Carlo, against the
   sqrt(2 ln k) asymptote.
B. `null`    -- the *measured* null of BlockScore. Two disjoint draws of n=128
   real held-out builds play "control" and "arm"; every pillar of the run of
   record is recomputed on both, z-scored exactly as `composite.Calibration`
   does it, and the max is taken. Also reports each pillar's declared unit
   against its measured fresh-draw spread, and the inter-pillar correlation.
C. `scale`   -- sd of the two kernel statistics as a function of n at fixed and
   at halved reference size, to test the 1/n claim and the finite-reference
   cross term that breaks it.
D. `cld`     -- the compact letter display of the run of record: greedy (what
   the card and the paper's table show) against maximal-clique (what the code
   does now), from the stored recompute of 2026-09-19.

Usage:  .venv/bin/python scripts/figs/math_max_check.py [a] [b] [c] [p] [d]
        (`p` is C at 1000 draws per n, with a bootstrap interval on the exponent)
        (no argument runs everything; B and C take a few minutes on CPU)
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/run_20260925_002631_bench_canonical-split_fixed_transparency_cull"
RECOMPUTE = ROOT / "outputs/run_20260919_022424_significance_recompute"
OUT = ROOT / "outputs/figstage/math_max_check.json"
CACHE = Path("/home/griffing52/tmp/claude-1000/-home-griffing52-programming-gen-BlockGen"
             "/0494f924-6c46-4bb2-9a34-d9f3543e957d/scratchpad/math_max_cache.npz")

VIEW_KEY = "v4_az45-135-225-315_el30-30-30-30_px224_ortho"
CORPUS, SEED, NREF, NARM = "houses_32", 0, 399, 128


# ---------------------------------------------------------------- A. theory --
def max_moments_mc(k: int, rho: float, reps: int = 400_000,
                   seed: int = 0) -> Tuple[float, float]:
    """(E, sd) of max of k equicorrelated standard Gaussians, by Monte Carlo."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((reps, 1))
    e = rng.standard_normal((reps, k))
    x = math.sqrt(rho) * z + math.sqrt(1.0 - rho) * e
    m = x.max(axis=1)
    return float(m.mean()), float(m.std(ddof=1))


def a_k_exact(k: int) -> float:
    """E[max of k iid N(0,1)] by numerical integration of 1 - F^k."""
    from scipy import integrate, stats as sps

    f = lambda x: x * k * sps.norm.pdf(x) * sps.norm.cdf(x) ** (k - 1)
    val, _ = integrate.quad(f, -12, 12, limit=400)
    return float(val)


def theory(pillar_sd: float = math.sqrt(2.0)) -> dict:
    rows = []
    for k in (1, 2, 3, 4, 5, 8, 10):
        a = a_k_exact(k)
        mc0 = max_moments_mc(k, 0.0)
        rows.append({
            "k": k, "a_k": a, "sqrt_2lnk": math.sqrt(2 * math.log(k)) if k > 1 else 0.0,
            "mc_mean_rho0": mc0[0], "mc_sd_rho0": mc0[1],
            "E_blockscore_rho0": pillar_sd * a,
        })
    grid = []
    for rho in (0.0, 0.2, 0.4, 0.6, 0.8, 0.95):
        m, s = max_moments_mc(5, rho)
        grid.append({"k": 5, "rho": rho, "E_max": m, "sd_max": s,
                     "sqrt_1mrho_a5": math.sqrt(1 - rho) * a_k_exact(5),
                     "E_blockscore": pillar_sd * m})
    out = {"per_k": rows, "k5_vs_rho": grid, "pillar_sd_assumed": pillar_sd}
    print("\n== A. null of a max of k unit-scale pillars ==")
    print(f"{'k':>3} {'E[max] iid':>11} {'sqrt(2 ln k)':>13} {'sd[max]':>9} "
          f"{'E[BlockScore] at pillar sd %.3f' % pillar_sd:>34}")
    for r in rows:
        print(f"{r['k']:>3} {r['a_k']:>11.4f} {r['sqrt_2lnk']:>13.4f} "
              f"{r['mc_sd_rho0']:>9.4f} {r['E_blockscore_rho0']:>34.3f}")
    print(f"\nk=5, equicorrelated (E[max] = sqrt(1-rho) * a_5 exactly):")
    for g in grid:
        print(f"  rho={g['rho']:.2f}  E[max]={g['E_max']:.4f} "
              f"(closed form {g['sqrt_1mrho_a5']:.4f})  sd={g['sd_max']:.4f}  "
              f"E[BlockScore]={g['E_blockscore']:.3f}")
    return out


# ------------------------------------------------------------------ B. null --
PILLAR_PATHS = {
    "appearance": "realism.mv_dino_kid",
    "geometry": "realism.geom_kid",
    "palette": "dataset_stats.palette_jsd_exact",
    "coherence": "coherence.__rms__",
    "geometry_scalars": "geometry_scalars.__rms__",
}


def card() -> dict:
    return json.loads((RUN / "scorecard.json").read_text())


def card_calibration() -> dict:
    """Floor and declared spread per pillar, exactly as `composite` derives them."""
    c = card()
    arm = c["arms"]["real_test"]
    noise = c["context"].get("noise_floor_sd", {})
    out: Dict[str, dict] = {}
    for pillar, path in PILLAR_PATHS.items():
        block, key = path.split(".", 1)
        if key == "__rms__":
            vals = [e.get("w1_norm") for e in arm[block].values()
                    if isinstance(e, dict)]
            v = np.asarray([x for x in vals if x is not None and np.isfinite(x)])
            out[pillar] = {"floor": float(np.sqrt((v ** 2).mean())),
                           "spread": 1.0, "spread_from": "unit by construction"}
            continue
        m = arm[block][key]
        lo, hi = m["ci"]
        boot = (hi - lo) / (2 * 1.96)
        if path in noise:
            out[pillar] = {"floor": float(m["value"]), "spread": float(noise[path]),
                           "spread_from": "measured 48 fresh real draws",
                           "bootstrap_spread": float(boot)}
        else:
            out[pillar] = {"floor": float(m["value"]), "spread": float(boot),
                           "spread_from": "bootstrap CI of the control arm"}
    return out


def _load_pillar_inputs():
    """Per-structure tables for the 399 test builds plus the val reference."""
    from blockgen.eval.bench import (features as ft, geometry as geom,
                                     palette as pal, splits, topology)

    split = splits.load_split(CORPUS, SEED)
    val = splits.split_structures(split, "val")
    test = splits.split_structures(split, "test")
    tc = [s.crop_to_non_air() for s in test]

    std = geom.Standardizer.fit(geom.descriptor(val))
    geom_ref = geom.geom_features(val, std)
    geom_test = geom.geom_features(tc, std)

    coh_ref = topology.coherence_table(val)
    coh_test = topology.coherence_table(tc)
    gsc_ref = geom.geometry_table(val)
    gsc_test = geom.geometry_table(tc)

    keys = pal.union_vocabulary(tc, val, "exact")
    counts_test, _, _ = pal.palette_matrix(tc, "exact", keys)
    counts_val, _, _ = pal.palette_matrix(val, "exact", keys)
    ref_macro = pal.macro_distribution(counts_val)

    fk = lambda s: ft.FeatureKey(CORPUS, s, VIEW_KEY, "dinov2b")
    feats_val = ft.pooled(np.load(fk("val").path())["feats"], "mean")
    feats_test = ft.pooled(np.load(fk("test").path())["feats"], "mean")

    return dict(geom_ref=geom_ref, geom_test=geom_test, coh_ref=coh_ref,
                coh_test=coh_test, gsc_ref=gsc_ref, gsc_test=gsc_test,
                counts_test=counts_test, ref_macro=ref_macro,
                feats_val=feats_val, feats_test=feats_test)


def _pillars(idx: np.ndarray, D_, geom_, pal_, stats_, inp) -> Dict[str, float]:
    def rms_w1(test_tbl, ref_tbl, keys):
        vals = []
        for k in keys:
            g, r = test_tbl[k][idx], ref_tbl[k]
            sp = stats_.robust_spread(r)
            if np.isfinite(sp):
                vals.append(stats_.w1(g, r) / sp)
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        return float(np.sqrt((v ** 2).mean()))

    return {
        "appearance": float(D_.kid(inp["feats_test"][idx], inp["feats_val"])),
        "geometry": float(geom_.geom_kid(inp["geom_test"][idx], inp["geom_ref"])),
        "palette": float(pal_.jsd_from_counts(inp["counts_test"][idx],
                                              inp["ref_macro"])),
        "coherence": rms_w1(inp["coh_test"], inp["coh_ref"], list(inp["coh_ref"])),
        "geometry_scalars": rms_w1(inp["gsc_test"], inp["gsc_ref"],
                                   list(inp["gsc_ref"])),
    }


def null(reps: int = 300) -> dict:
    from blockgen.eval.bench import (distances as D, geometry as geom,
                                     palette as pal, stats as st)

    t0 = time.time()
    inp = _load_pillar_inputs()
    print(f"[null] inputs ready in {time.time() - t0:.1f}s")
    cal = card_calibration()
    names = list(PILLAR_PATHS)
    N = len(inp["geom_test"])

    rng = np.random.default_rng(12345)
    raw: List[Dict[str, float]] = []      # single-draw pillar values
    pairs: List[Tuple[Dict[str, float], Dict[str, float]]] = []
    for r in range(reps):
        perm = rng.permutation(N)
        a, c = perm[:NARM], perm[NARM:2 * NARM]      # disjoint: arm and control
        va = _pillars(a, D, geom, pal, st, inp)
        vc = _pillars(c, D, geom, pal, st, inp)
        raw.append(va)
        pairs.append((va, vc))
        if r % 50 == 0:
            print(f"[null] {r}/{reps} {time.time() - t0:.0f}s", flush=True)

    V = np.array([[d[p] for p in names] for d in raw])           # [reps, k]
    fpc = math.sqrt(max(1.0 - NARM / N, 1e-6))
    measured_sd = V.std(axis=0, ddof=1) / fpc                    # population sd
    corr = np.corrcoef(V.T)

    declared = np.array([cal[p]["spread"] for p in names])
    floor = np.array([cal[p]["floor"] for p in names])

    # (i) faithful protocol: arm vs the *card's* control values, declared units
    z_card = (V - floor) / declared
    # (ii) arm vs a fresh independent control draw, declared units
    Z = np.array([[(a[p] - c[p]) for p in names] for a, c in pairs]) / declared
    # (iii) same, in honest (measured fresh-draw) units
    Zh = np.array([[(a[p] - c[p]) for p in names] for a, c in pairs]) / measured_sd

    def summary(M: np.ndarray) -> dict:
        mx = M.max(axis=1)
        arg = M.argmax(axis=1)
        return {"mean": float(mx.mean()), "sd": float(mx.std(ddof=1)),
                "median": float(np.median(mx)), "p05": float(np.quantile(mx, .05)),
                "p95": float(np.quantile(mx, .95)), "max": float(mx.max()),
                "frac_above_3": float((mx > 3).mean()),
                "argmax_share": {names[i]: float((arg == i).mean())
                                 for i in range(len(names))},
                "mean_if_mean_aggregated": float(M.mean(axis=1).mean())}

    # How the null offset depends on WHICH pillars a run happens to have: a
    # FAST-tier card has no `appearance` pillar, and a missing pillar is simply
    # dropped, so k -- and with it the offset -- changes between cards.
    subsets = {}
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(range(len(names)), r):
            key = "+".join(names[i] for i in combo)
            mx = Z[:, list(combo)].max(axis=1)
            subsets[key] = {"k": r, "mean": float(mx.mean()),
                            "median": float(np.median(mx)),
                            "p95": float(np.quantile(mx, .95))}
    by_k = {}
    for r in range(1, len(names) + 1):
        vals = [v["mean"] for v in subsets.values() if v["k"] == r]
        by_k[r] = {"mean_of_means": float(np.mean(vals)),
                   "min": float(np.min(vals)), "max": float(np.max(vals))}

    out = {
        "reps": reps, "n_arm": NARM, "pool": N, "fpc_sd": fpc,
        "null_by_pillar_subset": subsets, "null_by_k": by_k,
        "null_full_tier": subsets["+".join(names)]["mean"],
        "null_fast_tier": subsets["+".join(n for n in names
                                           if n != "appearance")]["mean"],
        "pillars": names,
        "calibration": cal,
        "measured_population_sd": dict(zip(names, measured_sd.tolist())),
        "declared_spread": dict(zip(names, declared.tolist())),
        "unit_error_declared_over_measured":
            dict(zip(names, (declared / measured_sd).tolist())),
        "pillar_correlation": corr.tolist(),
        "mean_offdiag_corr": float(corr[~np.eye(len(names), dtype=bool)].mean()),
        "z_sd_declared_units": dict(zip(names, Z.std(axis=0, ddof=1).tolist())),
        "z_sd_honest_units": dict(zip(names, Zh.std(axis=0, ddof=1).tolist())),
        "blockscore_null_vs_card_control": summary(z_card),
        "blockscore_null_fresh_control_declared": summary(Z),
        "blockscore_null_fresh_control_honest": summary(Zh),
        "card_real_test_blockscore": 0.0,
    }

    print("\n== B. measured null of BlockScore (real arm vs real control) ==")
    print(f"{'pillar':>18} {'declared unit':>14} {'measured sd':>12} "
          f"{'ratio':>7} {'sd(z) declared':>15} {'sd(z) honest':>13}")
    for i, p in enumerate(names):
        print(f"{p:>18} {declared[i]:>14.5f} {measured_sd[i]:>12.5f} "
              f"{declared[i]/measured_sd[i]:>7.2f} "
              f"{Z[:, i].std(ddof=1):>15.3f} {Zh[:, i].std(ddof=1):>13.3f}")
    for label, key in (("vs card control (declared units)",
                        "blockscore_null_vs_card_control"),
                       ("vs fresh control (declared units)",
                        "blockscore_null_fresh_control_declared"),
                       ("vs fresh control (honest units)",
                        "blockscore_null_fresh_control_honest")):
        s = out[key]
        print(f"\n  BlockScore null, {label}:")
        print(f"    mean {s['mean']:.3f}  sd {s['sd']:.3f}  median {s['median']:.3f} "
              f" 90% [{s['p05']:.3f}, {s['p95']:.3f}]  P(>3)={s['frac_above_3']:.3f}")
        print(f"    mean of a MEAN-aggregated score: {s['mean_if_mean_aggregated']:.3f}")
        print(f"    which pillar wins the max: {s['argmax_share']}")
    print("\n  derivation vs measurement (Gaussian model with the measured "
          "covariance of z):")
    for tag, M in (("declared", Z), ("honest", Zh)):
        cov = np.cov(M.T)
        g = np.random.default_rng(5).multivariate_normal(
            np.zeros(len(names)), cov, size=200_000)
        pred = float(g.max(axis=1).mean())
        obs = float(M.max(axis=1).mean())
        sd = np.sqrt(np.diag(cov))
        rho = (cov / np.outer(sd, sd))
        off = rho[~np.eye(len(names), dtype=bool)].mean()
        # equicorrelated closed form with a common scale, for orientation
        closed = float(np.mean(sd) * math.sqrt(max(1 - off, 0)) * a_k_exact(len(names)))
        out[f"theory_{tag}"] = {"gaussian_mc_E_max": pred, "measured_E_max": obs,
                               "mean_offdiag_rho_of_z": float(off),
                               "equicorrelated_closed_form": closed,
                               "skew_of_max": float(((M.max(axis=1) - obs) ** 3).mean()
                                                    / M.max(axis=1).std() ** 3)}
        print(f"    {tag}: Gaussian-model E[max]={pred:.3f} vs measured "
              f"{obs:.3f}; equicorrelated closed form {closed:.3f} "
              f"(mean rho {off:+.2f})")
    print("\n  null offset by pillar count k (mean over all subsets of that size):")
    for r, v in out["null_by_k"].items():
        print(f"    k={r}: mean {v['mean_of_means']:.3f}  (range over subsets "
              f"{v['min']:.3f}..{v['max']:.3f})")
    print(f"    full tier (5 pillars) {out['null_full_tier']:.3f}; "
          f"FAST tier (no appearance) {out['null_fast_tier']:.3f}")
    print("\n  pillar correlation (single-draw values):")
    print("    " + "  ".join(f"{p[:6]:>6}" for p in names))
    for i, p in enumerate(names):
        print(f"{p[:16]:>16} " + "  ".join(f"{corr[i, j]:>6.2f}"
                                           for j in range(len(names))))
    return out


# ----------------------------------------------------------------- C. scale --
def scale(reps: int = 240) -> dict:
    """sd(n) for both kernel statistics, in the null and away from it.

    Three reference conditions, all with the held-out test pool as the probe:

    * `R=399` -- the run of record's protocol (full val reference);
    * `R=200/100 random` -- a random subset of the reference, which shrinks R
      while keeping the null (E[statistic] = 0) intact, so it isolates the
      finite-reference cross term;
    * `R=200/100 first-rows` -- the first rows of the split, which turn out to
      be a biased subsample (the statistic is far from 0), included as the
      P != Q control: there the sd must fall as 1/sqrt(n), and it does.

    A fourth condition uses the *train* pool against the val reference, which is
    also not a null on this corpus, as a second P != Q reading at large n.
    """
    from blockgen.eval.bench import distances as D, geometry as geom, splits

    inp = _load_pillar_inputs()
    rng = np.random.default_rng(7)
    ns = [16, 24, 32, 48, 64, 96, 128, 160, 192]
    nv = len(inp["geom_ref"])
    conds = [("R=399", np.arange(nv))]
    for R in (200, 100):
        conds.append((f"R={R} random", rng.choice(nv, size=R, replace=False)))
        conds.append((f"R={R} first-rows", np.arange(R)))

    rows = []
    for label, sub in conds:
        gref, fref = inp["geom_ref"][sub], inp["feats_val"][sub]
        N = len(inp["geom_test"])
        for n in ns:
            gv, fv = [], []
            for _ in range(reps):
                idx = rng.choice(N, size=n, replace=False)
                gv.append(geom.geom_kid(inp["geom_test"][idx], gref))
                fv.append(D.kid(inp["feats_test"][idx], fref))
            fpc = math.sqrt(max(1 - n / N, 1e-6))
            rows.append({"ref": label, "R": len(sub), "n": n,
                         "geom_kid_mean": float(np.mean(gv)),
                         "geom_kid_sd": float(np.std(gv, ddof=1) / fpc),
                         "mv_dino_kid_mean": float(np.mean(fv)),
                         "mv_dino_kid_sd": float(np.std(fv, ddof=1) / fpc)})
            print(f"[scale] {label:>18} n={n:>4} geom {rows[-1]['geom_kid_mean']:+.4f}"
                  f" sd {rows[-1]['geom_kid_sd']:.5f} | dino "
                  f"{rows[-1]['mv_dino_kid_mean']:+.5f} sd "
                  f"{rows[-1]['mv_dino_kid_sd']:.6f}", flush=True)

    def fit(metric: str, ref: str) -> dict:
        sub = [r for r in rows if r["ref"] == ref]
        n = np.array([r["n"] for r in sub], float)
        sd = np.array([r[metric + "_sd"] for r in sub], float)
        slope, _ = np.polyfit(np.log(n), np.log(sd), 1)
        A = np.vstack([1 / n ** 2, 1 / n]).T          # degenerate + non-degenerate
        coef, *_ = np.linalg.lstsq(A, sd ** 2, rcond=None)
        return {"loglog_slope": float(slope),
                "mean_at_n128": float([r[metric + "_mean"] for r in sub
                                       if r["n"] == 128][0]),
                "two_term_a": float(coef[0]), "two_term_b": float(coef[1]),
                "crossover_n": float(coef[0] / coef[1]) if coef[1] > 0
                else float("inf")}

    labels = [c[0] for c in conds]
    fits = {f"{m}@{l}": fit(m, l) for m in ("geom_kid", "mv_dino_kid")
            for l in labels}
    print("\n== C. how the noise floor scales with n ==")
    print(f"{'metric@reference':>30} {'slope':>7} {'mean@128':>10} "
          f"{'crossover n*':>13}")
    for k, v in fits.items():
        print(f"{k:>30} {v['loglog_slope']:>7.3f} {v['mean_at_n128']:>10.4f} "
              f"{v['crossover_n']:>13.1f}")
    return {"rows": rows, "fits": fits, "reps": reps, "ns": ns}


def scale_precise(reps: int = 1000, n_boot: int = 400) -> dict:
    """The headline n-scaling number, with a bootstrap interval on the exponent.

    One condition only -- the run of record's own protocol (held-out test pool
    against the full 399-build val reference, which is a null: the mean of both
    statistics is within noise of 0) -- at 1000 draws per n, so the exponent can
    be quoted with an interval and 1/n can be separated from 1/sqrt(n).
    """
    from blockgen.eval.bench import distances as D, geometry as geom

    inp = _load_pillar_inputs()
    rng = np.random.default_rng(101)
    N = len(inp["geom_test"])
    ns = [16, 32, 64, 128, 192]
    vals = {n: {"geom_kid": [], "mv_dino_kid": []} for n in ns}
    for n in ns:
        for _ in range(reps):
            idx = rng.choice(N, size=n, replace=False)
            vals[n]["geom_kid"].append(
                geom.geom_kid(inp["geom_test"][idx], inp["geom_ref"]))
            vals[n]["mv_dino_kid"].append(
                D.kid(inp["feats_test"][idx], inp["feats_val"]))
        print(f"[precise] n={n} done", flush=True)

    out: dict = {"reps": reps, "ns": ns, "R": len(inp["geom_ref"]), "pool": N}
    x = np.log(np.asarray(ns, float))
    fpc = {n: math.sqrt(1 - n / N) for n in ns}
    print("\n== C2. the n-scaling exponent, with an interval ==")
    for metric in ("geom_kid", "mv_dino_kid"):
        sd = np.array([np.std(vals[n][metric], ddof=1) / fpc[n] for n in ns])
        slope = float(np.polyfit(x, np.log(sd), 1)[0])
        boots = []
        for b in range(n_boot):
            r = np.random.default_rng(b)
            sdb = np.array([np.std(r.choice(vals[n][metric], size=reps,
                                            replace=True), ddof=1) / fpc[n]
                            for n in ns])
            boots.append(np.polyfit(x, np.log(sdb), 1)[0])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[metric] = {"sd": dict(zip(map(str, ns), sd.tolist())),
                       "mean_at_n128": float(np.mean(vals[128][metric])),
                       "slope": slope, "slope_ci": [float(lo), float(hi)],
                       "sd_ratio_64_to_128": float(sd[3] / sd[2])}
        print(f"  {metric:>12}: sd {np.round(sd, 6).tolist()}")
        print(f"                slope {slope:+.3f} 95% CI "
              f"[{lo:+.3f}, {hi:+.3f}]; sd(128)/sd(64) = {sd[3]/sd[2]:.3f} "
              f"(1/n predicts 0.50, 1/sqrt(n) 0.71); mean at n=128 "
              f"{np.mean(vals[128][metric]):+.5f}")
    return out


# ------------------------------------------------------------------- D. cld --
def _greedy_letters(order: Sequence[str], sep_pairs: set) -> Dict[str, str]:
    """The pre-2026-09-19 display: one letter per arm, first fit."""
    letters: Dict[str, str] = {}
    groups: List[List[str]] = []
    for arm in order:
        for gi, g in enumerate(groups):
            if all(frozenset((arm, o)) not in sep_pairs for o in g):
                g.append(arm)
                letters[arm] = chr(ord("a") + gi)
                break
        else:
            groups.append([arm])
            letters[arm] = chr(ord("a") + len(groups) - 1)
    return letters


def cld_from_recompute() -> dict:
    """Greedy vs maximal-clique letters, from the stored pairwise verdicts.

    `recompute_r1.json` already carries all four displays; this recomputes them
    from the 153 comparisons so the comparison is not taken on trust, and adds
    the transitivity audit the paper's wording depends on.
    """
    from blockgen.eval.bench.compare import _letter_groups

    d = json.loads((RECOMPUTE / "recompute_r1.json").read_text())
    out: dict = {"n_perm": d["n_perm"]}
    for metric, tbl in d["metrics"].items():
        order = tbl["order"]
        sep: Dict[str, set] = {"legacy": set(), "permutation": set()}
        for c in tbl["comparisons"]:
            pair = frozenset((c["a"], c["b"]))
            # 0.0 is a legitimate adjusted p; `or` would silently discard it.
            pl = c.get("p_adj_legacy")
            pp = c.get("p_adj_perm", c.get("p_adj"))
            if pl is not None and pl < 0.05:
                sep["legacy"].add(pair)
            if pp is not None and pp < 0.05:
                sep["permutation"].add(pair)
        entry: dict = {"order": order, "scores": tbl["scores"]}
        for test, pairs in sep.items():
            greedy = _greedy_letters(order, pairs)
            valid = _letter_groups(order, lambda a, b, s=pairs:
                                   "x" if frozenset((a, b)) in s else None)
            # transitivity audit: triples with a~b, b~c but a|c separated
            viol = 0
            for a, b, c in itertools.combinations(order, 3):
                ab = frozenset((a, b)) in pairs
                bc = frozenset((b, c)) in pairs
                ac = frozenset((a, c)) in pairs
                if (not ab) and (not bc) and ac:
                    viol += 1
            entry[test] = {
                "n_separated": len(pairs), "n_pairs": len(tbl["comparisons"]),
                "greedy": greedy, "clique": valid,
                "n_cliques": len({l for ls in valid.values() for l in ls}),
                "multi_letter_arms": {a: l for a, l in valid.items() if len(l) > 1},
                "intransitive_triples": viol,
                "stored_greedy_matches": greedy == tbl.get(
                    "groups_greedy_" + test, greedy),
                "stored_clique_matches": valid == tbl.get(
                    "groups_cld_" + test, valid),
                "greedy_equals_clique": greedy == valid,
            }
        entry["card_groups"] = tbl.get("card_groups")
        out[metric] = entry
    return out


# ------------------------------------------------------------------- main ----
def main(argv: List[str]) -> int:
    which = [a.lower() for a in argv[1:]] or ["a", "b", "c", "p", "d"]
    res: dict = {}
    if "a" in which:
        res["theory"] = theory()
    if "b" in which:
        res["null"] = null()
    if "c" in which:
        res["scale"] = scale()
    if "p" in which:
        res["scale_precise"] = scale_precise()
    if "d" in which:
        res["cld"] = cld_from_recompute()
        print("\n== D. compact letter display ==")
        for metric in ("geom_kid", "mv_dino_kid"):
            e = res["cld"][metric]
            for test in ("legacy", "permutation"):
                t = e[test]
                print(f"\n {metric} / {test}: {t['n_separated']}/{t['n_pairs']} "
                      f"separated, {t['n_cliques']} maximal cliques, "
                      f"{t['intransitive_triples']} intransitive triples; "
                      f"greedy==clique: {t['greedy_equals_clique']}; "
                      f"reproduces stored: greedy={t['stored_greedy_matches']} "
                      f"clique={t['stored_clique_matches']}")
                print(f"   multi-letter arms: {t['multi_letter_arms']}")
                for arm in ("real_test", "real@solidify", "native_oriented",
                            "canon_oriented", "llm_control", "llm_bias"):
                    if arm in t["greedy"]:
                        print(f"     {arm:>24}  greedy {t['greedy'][arm]:>4}  "
                              f"clique {t['clique'][arm]:>4}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    prev.update(res)
    OUT.write_text(json.dumps(prev, indent=1))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
