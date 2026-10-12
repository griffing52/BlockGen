"""Checks for the mixture identity, the sqrt readout and its finite-sample bias.

Owned by the `math_sqrt` audit agent (see deliverables/paper/iclr/math/math_sqrt.md).
Computes, and prints, four things:

  A  the exact expectation of the UNBIASED estimator on the mixture the dose
     ladder actually builds (fixed counts, not an iid draw from the mixture),
     and the exponent a one-parameter power fit reads off that curve;
  B  a Monte-Carlo of E[sqrt(max(KID_hat, 0))] against sqrt(E[KID_hat]) on the
     real DINOv2 feature cache at the paper's n, with a second real feature
     distribution standing in for the bad pool, plus a mean-shift surrogate
     calibrated to the run-of-record's fully-bad value;
  C  the same on the real geometric descriptor with the real procedural bad
     pool (`uniform_random`), i.e. V-KID exactly as the paper computes it;
  D  the quadratic decomposition theta(beta) = theta0 + L beta + Q beta^2 of the
     ladder's own mixture response, which is what makes the measured exponent
     differ from 2.

Run: .venv/bin/python scripts/figs/math_sqrt_check.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockgen.eval.bench import distances as D
from blockgen.eval.bench import features as ft

SCALE = 1000.0
CACHE = Path("/home/griffing52/tmp/claude-1000/-home-griffing52-programming-gen-BlockGen/"
             "0494f924-6c46-4bb2-9a34-d9f3543e957d/scratchpad/math_sqrt_feats.npz")


# --- kernel bookkeeping -----------------------------------------------------
def gram(a: np.ndarray, b: np.ndarray, gamma: float, degree: int = 3,
         coef0: float = 1.0) -> np.ndarray:
    return SCALE * (gamma * (a @ b.T) + coef0) ** degree


def offdiag_mean(k: np.ndarray) -> float:
    n = len(k)
    return float((k.sum() - np.trace(k)) / (n * (n - 1)))


def coefficients(P: np.ndarray, Q: np.ndarray, V: np.ndarray) -> dict:
    """Exact E[KID_hat] as a quadratic in the bad fraction beta.

    Arm members are iid draws from the empirical pools (with replacement), so
    the pools ARE the populations and every expectation below is exact, not an
    approximation. V (the reference) is held fixed, exactly as the suite does.
    """
    g = 1.0 / P.shape[1]
    b_pp, b_qq, b_pq = (float(gram(P, P, g).mean()), float(gram(Q, Q, g).mean()),
                        float(gram(P, Q, g).mean()))
    c_p, c_q = float(gram(P, V, g).mean()), float(gram(Q, V, g).mean())
    t_v = offdiag_mean(gram(V, V, g))
    return {
        "b_pp": b_pp, "b_qq": b_qq, "b_pq": b_pq, "c_p": c_p, "c_q": c_q, "t_v": t_v,
        # theta(beta) = theta0 + L beta + Q2 beta^2 for an iid mixture draw
        "theta0": b_pp + t_v - 2 * c_p,
        "L": 2 * ((b_pq - b_pp) - (c_q - c_p)),
        "Q2": b_pp + b_qq - 2 * b_pq,          # = MMD^2(Q, P), population
        "gamma": g,
    }


def theta_iid(co: dict, beta: float) -> float:
    return co["theta0"] + co["L"] * beta + co["Q2"] * beta ** 2


def theta_strat(co: dict, beta: float, n: int) -> float:
    """Exact E[KID_hat] when the arm holds EXACTLY k=beta*n bad builds."""
    k = int(round(beta * n))
    m = n - k                                    # good builds
    pairs = (m * (m - 1) * co["b_pp"] + k * (k - 1) * co["b_qq"]
             + 2 * k * m * co["b_pq"]) / (n * (n - 1))
    cross = (m * co["c_p"] + k * co["c_q"]) / n
    return pairs + co["t_v"] - 2 * cross


# --- A: the stratified-mixture correction, in the ideal case ----------------
def part_a() -> None:
    print("=" * 78)
    print("A. exact expectation of the unbiased estimator on a FIXED-COUNT mixture")
    print("   ideal case mu_P = mu_V (arm's clean part = reference distribution):")
    print("   E[KID_hat] = MMD^2(Q,P) * (beta^2 - beta(1-beta)/(n-1))")
    doses = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    for n in (64, 128, 256):
        fac = doses ** 2 - doses * (1 - doses) / (n - 1)
        ks = np.arange(0.5, 4.001, 0.001)
        # same one-parameter anchored family the paper fits (bench_doseresponse)
        v0, worst = fac[0], fac[-1]
        errs = np.abs(fac[None, :] - (v0 + (worst - v0) * doses[None, :] ** ks[:, None])
                      ).sum(1)
        k_hat = float(ks[int(np.argmin(errs))])
        print(f"   n={n:4d}  factor/beta^2 = "
              + " ".join(f"{f/ (d**2):.4f}" for d, f in zip(doses[1:], fac[1:]))
              + f"   best-fit exponent = {k_hat:.3f}")
    z = np.random.default_rng(0).standard_normal(4_000_000)
    print(f"   E[sqrt(Z+)] for Z~N(0,1) = {np.sqrt(np.maximum(z, 0)).mean():.5f} "
          f"(closed form 2^(1/4)Gamma(3/4)/(2 sqrt(pi)) = "
          f"{2 ** 0.25 * 1.2254167024651776 / (2 * np.pi ** 0.5):.5f})")


# --- B/C: Monte Carlo of the sqrt readout ----------------------------------
def simulate(name: str, P: np.ndarray, Q: np.ndarray, V: np.ndarray, n: int,
             reps: int, doses=(0.0, 0.25, 0.5, 0.75, 1.0), seed: int = 0) -> dict:
    co = coefficients(P, Q, V)
    rng = np.random.default_rng(seed)
    print("=" * 78)
    print(f"{name}: n={n}, ref={len(V)}, pools P={len(P)} Q={len(Q)}, reps={reps}")
    print(f"   theta0={co['theta0']:.4f}  L={co['L']:.4f}  Q2=MMD^2(Q,P)={co['Q2']:.4f}"
          f"   |L|/Q2={abs(co['L'])/co['Q2']:.4f}")
    rows = []
    for beta in doses:
        k = int(round(beta * n))
        vals = np.empty(reps)
        for r in range(reps):
            good = P[rng.integers(0, len(P), n - k)] if n - k else P[:0]
            bad = Q[rng.integers(0, len(Q), k)] if k else Q[:0]
            arm = np.concatenate([good, bad], 0)
            vals[r] = D.kid(arm, V)
        th = theta_strat(co, beta, n)
        sd = float(vals.std(ddof=1))
        root = np.sqrt(np.maximum(vals, 0.0))
        snr = th / sd if sd > 0 else float("nan")
        delta = -sd ** 2 / (8 * th ** 1.5) if th > 0 else float("nan")
        rows.append({
            "beta": float(beta), "theta_exact": th, "mc_mean": float(vals.mean()),
            "sd": sd, "p_negative": float((vals < 0).mean()),
            "sqrt_theta": float(np.sqrt(max(th, 0.0))),
            "mc_mean_sqrt": float(root.mean()), "mc_sd_sqrt": float(root.std(ddof=1)),
            "snr": float(snr), "delta_method_bias": float(delta),
            "measured_bias": float(root.mean() - np.sqrt(max(th, 0.0))),
        })
        r = rows[-1]
        print(f"   beta={beta:4.2f}  E[X]={th:10.4f} (MC {r['mc_mean']:10.4f})  "
              f"sd={sd:8.4f}  P(X<0)={r['p_negative']:.3f}  "
              f"sqrt(E X)={r['sqrt_theta']:8.4f}  E[sqrt X+]={r['mc_mean_sqrt']:8.4f}  "
              f"bias={r['measured_bias']:+8.4f} (delta {delta:+8.4f})  SNR={snr:8.2f}")
    # what the readout does to the recovered-fraction MAE
    worst = rows[-1]["mc_mean_sqrt"]
    rec = np.array([r["mc_mean_sqrt"] / worst for r in rows])
    mae = float(np.abs(rec - np.array(doses)).mean())
    print(f"   E[rec_sqrt] = {np.round(rec, 4).tolist()}  -> MAE {mae:.4f} "
          f"(interior {np.abs(rec - np.array(doses))[1:-1].mean():.4f})")
    return {"name": name, "n": n, "coefficients": co, "rows": rows,
            "mae_rec_sqrt_at_expectation": mae}


def part_f(n: int = 128, trials: int = 5) -> None:
    """Numerically verify the ideal-case specialisation used in part A.

    With the arm's clean component distributed exactly as the reference
    (c_p = b_pp = t_v and c_q = b_pq), the general fixed-count expectation must
    collapse to MMD^2(Q,P) * (beta^2 - beta(1-beta)/(n-1)).
    """
    print("=" * 78)
    print("F. ideal-case collapse of the exact fixed-count expectation")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(trials):
        b_pp, b_pq = rng.uniform(1, 5), rng.uniform(1, 5)
        b_qq = b_pp + 2 * b_pq + rng.uniform(0.1, 3)     # keeps MMD^2 >= 0
        co = {"b_pp": b_pp, "b_qq": b_qq, "b_pq": b_pq,
              "c_p": b_pp, "c_q": b_pq, "t_v": b_pp,
              "Q2": b_pp + b_qq - 2 * b_pq, "theta0": 0.0, "L": 0.0}
        for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
            got = theta_strat(co, beta, n)
            want = co["Q2"] * (beta ** 2 - beta * (1 - beta) / (n - 1))
            worst = max(worst, abs(got - want))
    print(f"   max |exact - closed form| over {trials} random kernels x 5 doses "
          f"at n={n}: {worst:.3e}")


def part_g(P: np.ndarray, V: np.ndarray, theta_worst: float, label: str,
           ns=(32, 64, 128, 256), reps: int = 600, seed: int = 3) -> dict:
    """The sqrt readout's irreducible floor, and how it scales with n.

    Under the null the unbiased MMD^2 estimator is a degenerate U-statistic, so
    sd ~ 1/n; sqrt of it is ~ n^{-1/2}, and the *apparent failure rate* floor
    `E[sqrt(X+)] / sqrt(theta_worst)` therefore falls only as n^{-1/2}.
    """
    print("=" * 78)
    print(f"G. null behaviour of the sqrt readout vs n -- {label}")
    rng = np.random.default_rng(seed)
    co = coefficients(P, P, V)          # Q = P: pure null arm
    rows = []
    for n in ns:
        vals = np.array([D.kid(P[rng.integers(0, len(P), n)], V) for _ in range(reps)])
        root = np.sqrt(np.maximum(vals, 0.0))
        rows.append({"n": n, "theta": co["theta0"], "sd": float(vals.std(ddof=1)),
                     "p_negative": float((vals < 0).mean()),
                     "mean_sqrt": float(root.mean()),
                     "pred_pure_null": 0.41109 * float(vals.std(ddof=1)) ** 0.5,
                     "apparent_bad_fraction": float(root.mean()
                                                    / np.sqrt(theta_worst))})
        r = rows[-1]
        print(f"   n={n:4d}  sd={r['sd']:.5f}  n*sd={n*r['sd']:.4f}  "
              f"P(X<0)={r['p_negative']:.3f}  E[sqrt X+]={r['mean_sqrt']:.4f}  "
              f"(0.411 sqrt(sd) = {r['pred_pure_null']:.4f})  "
              f"apparent bad fraction = {r['apparent_bad_fraction']:.4f}")
    return {"name": f"G {label}", "theta_worst": theta_worst, "rows": rows}


def fit_exponent(dose, value) -> float:
    """Best exponent in the anchored one-parameter family `bench_doseresponse` fits."""
    d, v = np.asarray(dose, float), np.asarray(value, float)
    v0, worst = v[0], v[-1]
    ks = np.arange(0.2, 6.001, 0.001)
    errs = np.abs(v[None, :] - (v0 + (worst - v0) * d[None, :] ** ks[:, None])).sum(1)
    return float(ks[int(np.argmin(errs))])


def part_e(name: str, P: np.ndarray, Q: np.ndarray, doses=(0.0, 0.25, 0.5, 0.75, 1.0)
           ) -> dict:
    """POPULATION Frechet distance of the mixture, from exact mixture moments.

    m_S = (1-b) m_P + b m_Q, Sigma_S = (1-b) Sigma_P + b Sigma_Q + b(1-b) dd^T.
    No sampling: this isolates the mixture response of the functional itself.
    """
    from scipy import linalg
    print("=" * 78)
    print(f"E. population Frechet distance vs bad fraction -- {name}")
    mP, mQ = P.mean(0), Q.mean(0)
    SP, SQ = np.cov(P, rowvar=False), np.cov(Q, rowvar=False)
    dd = np.outer(mQ - mP, mQ - mP)
    vals, mean_part, cov_part = [], [], []
    for b in doses:
        mS = (1 - b) * mP + b * mQ
        SS = (1 - b) * SP + b * SQ + b * (1 - b) * dd
        cm = linalg.sqrtm(SS @ SP)
        cm = cm.real if np.iscomplexobj(cm) else cm
        diff = mS - mP
        mean_part.append(float(diff @ diff))
        cov_part.append(float(np.trace(SS) + np.trace(SP) - 2 * np.trace(cm)))
        vals.append(mean_part[-1] + cov_part[-1])
    k = fit_exponent(doses, vals)
    print(f"   rank(Sigma_P) = {np.linalg.matrix_rank(SP)} of d={P.shape[1]}")
    for b, v, mp, cp in zip(doses, vals, mean_part, cov_part):
        r1 = cp / b if b else float("nan")
        r2 = cp / b ** 2 if b else float("nan")
        print(f"   beta={b:5.3f}  FD={v:12.5f}  mean-part={mp:12.5f} (beta^2 exact) "
              f" cov-part={cp:12.5f}  cov/beta={r1:9.5f}  cov/beta^2={r2:10.5f}  "
              f"FD/FD(1)={v/vals[-1]:7.4f}")
    print(f"   best-fit exponent for FD = {k:.3f}  (KID identity would give 2)")
    return {"name": name, "doses": list(doses), "fd": vals,
            "mean_part": mean_part, "cov_part": cov_part, "exponent": k}


def dino_pools():
    view = ft.ViewConfig()
    def load(corpus, split):
        k = ft.FeatureKey(corpus, split, view.key(), "dinov2b")
        with np.load(k.path()) as z:                    # cache only; never renders
            return ft.pooled(z["feats"].astype(np.float64), "mean")
    return load("houses_32", "test"), load("houses_32", "val"), load("lego_void_20", "test")


def geom_pools():
    if CACHE.exists():
        z = np.load(CACHE)
        return z["P"], z["V"], z["Q"]
    from blockgen.eval.bench import baselines as bl, geometry as geom, splits
    sp = splits.load_split("houses_32", 0)
    tr = splits.split_structures(sp, "train")
    va = [s.crop_to_non_air() for s in splits.split_structures(sp, "val")]
    te = [s.crop_to_non_air() for s in splits.split_structures(sp, "test")]
    std = geom.Standardizer.fit(geom.descriptor(va))
    bad = [s.crop_to_non_air() for s in bl.build("uniform_random", len(te), tr, seed=0)]
    P, V, Q = (geom.geom_features(te, std), geom.geom_features(va, std),
               geom.geom_features(bad, std))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, P=P, V=V, Q=Q)
    return P, V, Q


def main() -> int:
    part_a()
    part_f()
    out = []
    P, V, Qlego = dino_pools()
    print(f"\n[dino] P(test)={P.shape} V(val)={V.shape} Q(lego test)={Qlego.shape}")
    out.append(simulate("B1 mv_dino_kid, bad pool = lego builds (real features)",
                        P, Qlego, V, n=128, reps=400))
    # mean-shift surrogate calibrated so MMD^2(Q,P) matches the run of record's
    # fully-bad mv_dino_kid value (1.7178, uniform_random at n=128).
    g = 1.0 / P.shape[1]
    u = np.zeros(P.shape[1]); u[0] = 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        Qs = P + mid * u
        c = coefficients(P, Qs, V)
        if c["Q2"] < 1.7178:
            lo = mid
        else:
            hi = mid
    Qs = P + ((lo + hi) / 2) * u
    print(f"[dino] mean-shift surrogate delta={(lo+hi)/2:.5f} -> "
          f"MMD^2(Q,P)={coefficients(P, Qs, V)['Q2']:.4f}")
    out.append(simulate("B2 mv_dino_kid, mean-shift surrogate at the paper's signal",
                        P, Qs, V, n=128, reps=400))
    Pg, Vg, Qg = geom_pools()
    print(f"\n[geom] P={Pg.shape} V={Vg.shape} Q={Qg.shape}")
    out.append(simulate("C geom_kid (V-KID), real uniform_random bad pool",
                        Pg, Qg, Vg, n=128, reps=400))
    # exponent a power fit reads off the EXACT expectation curve, i.e. what the
    # ladder's measured exponent should be once the cross term is accounted for
    print("=" * 78)
    print("D. exponent implied by the exact expectation curve (no sampling noise)")
    for s in out:
        co, rows = s["coefficients"], s["rows"]
        d = [r["beta"] for r in rows]
        print(f"   {s['name'][:52]:54s} k_exact={fit_exponent(d, [r['theta_exact'] for r in rows]):.3f}"
              f"  theta0={co['theta0']:.4f} L={co['L']:+.4f} Q2={co['Q2']:.3f}")
        print(f"      sd(sqrt X) MC vs delta-method sigma/(2 sqrt theta): "
              + " ".join(f"{r['mc_sd_sqrt']:.4f}/"
                         f"{r['sd']/(2*max(r['sqrt_theta'],1e-9)):.4f}" for r in rows[1:]))
    fine = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
    out.append(part_e("mv_dino_kid features (768-d), Q = lego builds", P, Qlego, fine))
    out.append(part_e("mv_dino_kid features, Q = mean-shift surrogate", P, Qs, fine))
    out.append(part_e("geom descriptor (77-d), Q = uniform_random", Pg, Qg, fine))
    # same DINOv2 pair after projecting to 64 dims, where the reference covariance
    # is full rank: this isolates the rank deficiency as the cause of the linear term
    pooled = np.concatenate([P, Qlego], 0)
    mu = pooled.mean(0)
    _, _, vt = np.linalg.svd(pooled - mu, full_matrices=False)
    B = vt[:64].T
    out.append(part_e("mv_dino_kid features projected to 64-d (full-rank Sigma)",
                      (P - mu) @ B, (Qlego - mu) @ B, fine))
    out.append(part_g(P, V, 1.7178, "mv_dino_kid, theta_worst = 1.7178"))
    out.append(part_g(Pg, Vg, 37.7295, "geom_kid, theta_worst = 37.7295"))
    dst = CACHE.parent / "math_sqrt_results.json"
    dst.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\n-> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
