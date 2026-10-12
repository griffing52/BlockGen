"""Does the benchmark reward *more* realistic output, or just detect artifacts?

The validation ladder answers a binary question -- can this metric tell damaged
builds from real ones. A benchmark that ranks submissions has to answer a harder
one: **does the number move smoothly and in the right direction as quality
changes?** A metric can pass every ladder gate by firing on an artifact while
being flat across the range that actually separates two submitted models.

The dose axes, each with a known ground-truth ordering:

    solidify        fraction of interior volume filled, from the walls inward
    occ_noise       fraction of blocks displaced to adjacent air
    canon           decimation to a smaller grid, then back
    mixture_<pool>  fraction of the arm drawn from a bad generator

`mixture` is the one that most resembles a real generator: a model does not
degrade uniformly, it produces some good builds and some failures, and the
question a leaderboard answers is whether an arm with more good builds scores
better. It is also the only axis on which a per-sample detector and a
distribution metric are expected to behave differently, and the only one the
quadratic analysis below applies to.

**The bad pool is a procedural baseline, not decimated real data.** The first
version of this axis mixed real builds with their canon-8 decimations, and the
coefficients it produced went into the paper. That pool was then disavowed as a
quality reference for the reason `fast.control_arms` records: requantising real
builds to a coarser grid produces a *smaller* building, not a worse one, so
"25% of this arm is canon-8" is 25% of a resolution change and not 25% of a
failure. `--bad-pool` now names arms from `bench.baselines` -- `uniform_random`
(the floor) and `shell_box` (a hollow box; the simplest thing anyone would call a
building) -- which are bad along a quality axis at fixed resolution. `canon8`
stays available to reproduce the retracted numbers.

Reported per metric: Spearman rho against the dose, whether the sequence is
strictly monotone, and the dynamic range in units of the metric's own noise
floor. A metric that is monotone but spans one noise floor cannot rank anything.
For mixture axes, also the quadratic-response fit and the `sqrt(KID)` recovery
of the true bad fraction (see `mixture_fit`).

    python scripts/bench_doseresponse.py --n 128 --bad-pool uniform_random shell_box
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from blockgen.eval.bench import baselines as bl
from blockgen.eval.bench import distances as D
from blockgen.eval.bench import fast as fast_tier
from blockgen.eval.bench import features as ft
from blockgen.eval.bench import geometry as geom
from blockgen.eval.bench import ladder as ld
from blockgen.eval.bench import probes as pb
from blockgen.eval.bench import scorecard as sc
from blockgen.eval.bench import splits
from blockgen.utils.data import Structure
from blockgen.utils.runs import new_run_dir

Rung = Tuple[float, List[Structure]]

#: Degradation pools the `mixture` axis can be built from. Everything but
#: `canon8` is a procedural baseline fitted on the train split only; `canon8`
#: is the decimation pool the paper's original coefficients were measured
#: against, kept so the retraction can be reproduced rather than asserted.
BAD_POOLS = ("uniform_random", "shell_box", "gabled_house", "patchwork", "canon8")


def bad_pool(name: str, pool: Sequence[Structure], train: Sequence[Structure],
             seed: int) -> List[Structure]:
    """`n` bad builds to contaminate the arm with, one per real build.

    Baselines are built from the *train* split, never from the test pool they
    are mixed into, so the bad half of a mixture shares no build with the good
    half -- otherwise the axis would interpolate between real data and a
    corruption of the same data, which is a harder thing to interpret than a
    mixture of two generators.
    """
    if name == "canon8":
        return pb.canon(pool, 8)
    return bl.build(name, len(pool), train, seed=seed)


def ladders(pool: Sequence[Structure], rng: np.random.Generator,
            bad: Dict[str, Sequence[Structure]],
            mix_rng: np.random.Generator | None = None) -> Dict[str, List[Rung]]:
    """`{axis: [(dose, structures), ...]}`, dose 0 = untouched real.

    `mix_rng` reseeds *only* which builds land in which mixture rung, leaving the
    corruption axes drawing from `rng` exactly as before. That separation is what
    makes a second mixture draw a robustness check rather than a different
    experiment: `--mix-seed 1` moves the mixture curve and provably nothing else.
    Defaulting it to `rng` keeps the original draw stream, so the coefficients
    this script produced before the bad pool changed are still reproducible.
    """
    pool = list(pool)
    n = len(pool)
    mix_rng = mix_rng if mix_rng is not None else rng

    def mix(damaged: Sequence[Structure], alpha: float) -> List[Structure]:
        """`alpha` of the arm is real, the rest comes from the bad pool."""
        k = int(round(alpha * n))
        order = mix_rng.permutation(n)
        return ([pool[i] for i in order[:k]]
                + [damaged[i] for i in order[k:]])

    axes: Dict[str, List[Rung]] = {
        "solidify": [(f, [pb.partial_solidify(s, f, rng) for s in pool])
                     for f in (0.0, 0.25, 0.5, 0.75, 1.0)],
        "occ_noise": [(p, [pb.occupancy_noise(s, p, rng) for s in pool] if p else pool)
                      for p in (0.0, 0.02, 0.05, 0.10, 0.20)],
        "canon": [(1.0 - d / 32.0, pb.canon(pool, d) if d < 32 else pool)
                  for d in (32, 24, 20, 16, 12, 8)],
    }
    # dose = fraction NOT real, so every axis runs 0 (perfect) -> 1 (worst).
    # Mixture axes come last and in a fixed order so that adding a second bad
    # pool cannot change the draws the earlier axes made from `rng` -- the
    # solidify and occ_noise numbers stay comparable to the previous run, which
    # is how "these axes are unaffected by the bad pool" gets *checked* rather
    # than asserted.
    for label in sorted(bad):
        axes[f"mixture_{label}"] = [(1.0 - a, mix(bad[label], a))
                                    for a in (1.0, 0.75, 0.5, 0.25, 0.0)]
    return axes


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    from scipy.stats import spearmanr
    if len(set(y)) < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def analyse(dose: Sequence[float], value: Sequence[float], sd: float) -> dict:
    v = np.asarray(value, dtype=float)
    d = np.asarray(dose, dtype=float)
    ok = np.isfinite(v)
    return {
        "dose": [float(x) for x in d],
        "value": [float(x) for x in v],
        "spearman": spearman(d[ok], v[ok]),
        "monotone": bool(np.all(np.diff(v[ok]) > 0)),
        "range_sd": float((v[ok].max() - v[ok].min()) / sd) if sd > 0 else float("nan"),
        "noise_sd": float(sd),
    }


def mixture_fit(dose: Sequence[float], value: Sequence[float]) -> dict:
    """Quadratic-response fit and `sqrt` recovery for one mixture axis.

    `MMD^2(alpha P + (1-alpha) Q, P) = (1-alpha)^2 MMD^2(Q, P)` -- Binkowski's
    Appendix B.2 -- so a squared-MMD metric answers the *fraction* of an arm
    that is bad quadratically, and an arm failing on a quarter of its prompts
    reads as a small fraction of the way to fully failed.

    Both curves are anchored at the measured endpoints rather than least-squares
    fitted, because the identity makes a prediction with no free parameters once
    the two ends are known: `v(dose) = v(0) + (v(1) - v(0)) * dose^k` for k=2
    (the identity) and k=1 (what a reader assuming proportionality would
    expect). Anchoring also keeps the comparison honest -- a two-parameter fit
    would beat a one-parameter one on a monotone curve of any shape.

    `recovered_*` is the failure rate a reader would infer: the metric
    normalised by its fully-bad value, with and without the square root. The
    square root is the fix, and the mean absolute error against the true dose is
    how much it buys.
    """
    d = np.asarray(dose, dtype=float)
    v = np.asarray(value, dtype=float)
    order = np.argsort(d)
    d, v = d[order], v[order]
    if not (np.isclose(d[0], 0.0) and np.isclose(d[-1], 1.0)):
        raise ValueError("mixture_fit needs the dose-0 and dose-1 endpoints")
    v0, worst = float(v[0]), float(v[-1])
    quad = v0 + (worst - v0) * d ** 2
    lin = v0 + (worst - v0) * d
    # Clamp before the root: an unbiased KID estimator is legitimately negative
    # on a real-vs-real draw, and `scorecard.with_mmd` clamps the same way.
    root = np.sqrt(np.maximum(v, 0.0))
    rec_sqrt = root / max(float(np.sqrt(max(worst, 0.0))), 1e-12)
    rec_raw = v / worst if worst != 0 else np.full_like(v, np.nan)
    err_q = float(np.abs(v - quad).sum())
    err_l = float(np.abs(v - lin).sum())
    # The exponent the data actually prefers, on the same one-parameter family
    # the two named curves come from. Reported because "quadratic fits 14x
    # better than linear" is consistent with any exponent near 2, and the honest
    # form of the claim is where the best fit lands: the identity says 2.
    ks = np.arange(0.5, 4.001, 0.01)
    errs = np.abs(v[None, :] - (v0 + (worst - v0) * d[None, :] ** ks[:, None])).sum(1)
    k_hat = float(ks[int(np.argmin(errs))])
    return {
        "exponent_best_fit": k_hat,
        "abs_err_best_fit": float(errs.min()),
        "dose": [float(x) for x in d],
        "value": [float(x) for x in v],
        "quadratic": [float(x) for x in quad],
        "linear": [float(x) for x in lin],
        "measured_over_linear": [float(a / b) if b else float("nan")
                                 for a, b in zip(v, lin)],
        "abs_err_quadratic": err_q,
        "abs_err_linear": err_l,
        "quadratic_better_by": float(err_l / err_q) if err_q > 0 else float("inf"),
        "recovered_sqrt": [float(x) for x in rec_sqrt],
        "recovered_raw": [float(x) for x in rec_raw],
        "mae_sqrt": float(np.mean(np.abs(rec_sqrt - d))),
        "mae_raw": float(np.mean(np.abs(rec_raw - d))),
        # The same two numbers with the endpoints dropped. Both are anchors, not
        # measurements: dose 1 is exactly 1 by construction for either transform,
        # and at dose 0 the clamp hands the square root a free zero while the raw
        # ratio is charged for a negative unbiased estimate. Averaging five rungs
        # of which two are fixed flatters both readouts, the root more than the
        # raw value, so the interior-only MAE is the comparison that is actually
        # about the transform.
        "mae_sqrt_interior": float(np.mean(np.abs(rec_sqrt - d)[1:-1])),
        "mae_raw_interior": float(np.mean(np.abs(rec_raw - d)[1:-1])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default="houses_32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=64, help="builds per rung")
    ap.add_argument("--n-ref", type=int, default=399)
    ap.add_argument("--bad-pool", nargs="+", default=["uniform_random", "shell_box"],
                    choices=BAD_POOLS,
                    help="degradation pool(s) the mixture axis contaminates with")
    ap.add_argument("--mix-seed", type=int, default=0,
                    help="reseed only the mixture draws (0 keeps the original "
                         "draw stream); a second value is a robustness check")
    ap.add_argument("--axes", nargs="*", default=None,
                    help="restrict to these axis names (default: all)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-render", action="store_true",
                    help="geometry only; skips the GPU tier")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    split = splits.load_split(args.corpus, args.seed)
    train = splits.split_structures(split, "train")
    val = splits.split_structures(split, "val")
    test = splits.split_structures(split, "test")
    if args.n_ref < len(val):
        val = [val[i] for i in np.random.default_rng(args.seed).permutation(len(val))[:args.n_ref]]

    pool = [s.crop_to_non_air() for s in test]
    pool = [pool[i] for i in rng.permutation(len(pool))[:args.n]]
    print(f"[dose] reference {len(val)}, pool {len(pool)}", flush=True)

    ctx = sc.BenchContext(corpus=args.corpus, seed=args.seed)
    ref = fast_tier.FastReference(train, val, ctx)
    bad = {name: bad_pool(name, pool, train, args.seed) for name in args.bad_pool}
    for name, structs in bad.items():
        blocks = np.mean([int(s.occupied_mask.sum()) for s in structs])
        print(f"[dose] bad pool {name}: {len(structs)} builds, "
              f"{blocks:.0f} blocks mean", flush=True)

    mix_rng = (np.random.default_rng(args.seed + 1000 * args.mix_seed)
               if args.mix_seed else None)
    axes = ladders(pool, rng, bad, mix_rng)
    if args.axes:
        missing = [a for a in args.axes if a not in axes]
        if missing:
            raise SystemExit(f"unknown axes {missing}; have {sorted(axes)}")
        axes = {a: axes[a] for a in args.axes}

    # --- metric definitions, each with its own measured noise floor ---------
    metrics: Dict[str, Callable[[Sequence[Structure]], float]] = {
        "geom_kid": lambda xs: geom.geom_kid(
            geom.geom_features(xs, ref.geom_std), ref.geom_ref),
    }
    held = [s.crop_to_non_air() for s in test]
    _, geom_sd, _ = ld.noise_floor(
        lambda r, p: geom.geom_kid(p, r), ref.geom_ref,
        geom.geom_features(held, ref.geom_std), min(args.n, len(held) - 1),
        reps=48, rng=np.random.default_rng(args.seed + 5))
    noise = {"geom_kid": geom_sd}

    if not args.no_render:
        view = ft.ViewConfig()
        ref_feats = ft.pooled(ft.load_or_build(
            ft.FeatureKey(args.corpus, "val", view.key(), "dinov2b"), val, view,
            device=args.device, verbose=False), "mean")
        test_feats = ft.pooled(ft.load_or_build(
            ft.FeatureKey(args.corpus, "test", view.key(), "dinov2b"), test, view,
            device=args.device, verbose=False), "mean")
        metrics["mv_dino_kid"] = lambda xs: D.kid(
            ft.pooled(ft.embed_views(xs, view, "dinov2b", args.device, verbose=False),
                      "mean"), ref_feats)
        _, dino_sd, _ = ld.noise_floor(lambda r, p: D.kid(p, r), ref_feats, test_feats,
                                       min(args.n, len(test_feats) - 1), reps=48,
                                       rng=np.random.default_rng(args.seed + 6))
        noise["mv_dino_kid"] = dino_sd

    results: Dict[str, Dict[str, dict]] = {}
    fits: Dict[str, Dict[str, dict]] = {}
    for axis, rungs in axes.items():
        results[axis] = {}
        doses = [d for d, _ in rungs]
        for mname, fn in metrics.items():
            vals = []
            for d, structs in rungs:
                vals.append(fn([s.crop_to_non_air() for s in structs]))
                print(f"  [{axis}/{mname}] dose {d:.2f} -> {vals[-1]:.4f}", flush=True)
            results[axis][mname] = analyse(doses, vals, noise[mname])
            if axis.startswith("mixture_"):
                fits.setdefault(axis, {})[mname] = mixture_fit(doses, vals)

    # The run name says which bad pool it used and which mixture draw, because
    # two dose runs that differ only in their degradation pool are two different
    # experiments and a directory called `doseresponse` says neither.
    name = "doseresponse_" + "-".join(args.bad_pool)
    if args.mix_seed:
        name += f"_mix{args.mix_seed}"
    run_dir = Path(args.out) if args.out else new_run_dir(name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "doseresponse.json").write_text(json.dumps(
        {"context": {"corpus": args.corpus, "seed": args.seed, "n": len(pool),
                     "n_ref": len(val), "noise_sd": noise,
                     "bad_pools": list(args.bad_pool),
                     "mix_seed": args.mix_seed,
                     "split_key": split.key(),
                     "git_sha": sc.git_sha()},
         "axes": results, "mixture_fits": fits}, indent=2) + "\n")

    md = render(results, noise, fits)
    (run_dir / "doseresponse.md").write_text(md)
    print()
    print(md)
    print(f"-> {run_dir}")
    return 0


def render(results: Dict[str, Dict[str, dict]], noise: Dict[str, float],
           fits: Dict[str, Dict[str, dict]] | None = None) -> str:
    lines = ["# Dose-response — does a better build score better?", ""]
    lines.append("Noise floors: "
                 + ", ".join(f"`{k}` {v:.4f}" for k, v in noise.items()))
    lines.append("")
    for axis, per_metric in results.items():
        lines.append(f"## {axis}")
        lines.append("")
        doses = next(iter(per_metric.values()))["dose"]
        lines.append("| metric | " + " | ".join(f"dose {d:g}" for d in doses)
                     + " | Spearman | monotone | range (sd) |")
        lines.append("|" + "---|" * (len(doses) + 4))
        for m, r in per_metric.items():
            cells = " | ".join(f"{v:.3f}" for v in r["value"])
            lines.append(f"| `{m}` | {cells} | {r['spearman']:+.3f} | "
                         f"{'yes' if r['monotone'] else '**no**'} | "
                         f"{r['range_sd']:.0f} |")
        lines.append("")
    lines.append("`range (sd)` is the span from best to worst rung in units of the "
                 "metric's own noise floor. Monotone but low-range means the metric "
                 "detects the artifact without being able to rank it.")
    lines.append("")
    for axis, per_metric in (fits or {}).items():
        lines.append(f"## {axis} — response to the failure rate")
        lines.append("")
        for m, f in per_metric.items():
            lines.append(f"### `{m}`")
            lines.append("")
            lines.append("| bad fraction | measured | quadratic | linear | "
                         "measured / linear | sqrt recovery | raw recovery |")
            lines.append("|---|---|---|---|---|---|---|")
            for i, d in enumerate(f["dose"]):
                lines.append(
                    f"| {d:.2f} | {f['value'][i]:.4f} | {f['quadratic'][i]:.4f} | "
                    f"{f['linear'][i]:.4f} | {f['measured_over_linear'][i]:.2f} | "
                    f"{f['recovered_sqrt'][i]:.3f} | {f['recovered_raw'][i]:.3f} |")
            lines.append("")
            lines.append(
                f"Total absolute error against a quadratic **{f['abs_err_quadratic']:.3f}**, "
                f"against a linear **{f['abs_err_linear']:.3f}** — "
                f"{f['quadratic_better_by']:.1f}x better; "
                f"best-fit exponent **{f['exponent_best_fit']:.2f}** "
                f"(error {f['abs_err_best_fit']:.3f}). "
                f"MAE recovering the true bad fraction: **{f['mae_sqrt']:.3f}** with "
                f"the square root, **{f['mae_raw']:.3f}** raw "
                f"(interior rungs only, both endpoints dropped: "
                f"{f['mae_sqrt_interior']:.3f} vs {f['mae_raw_interior']:.3f}).")
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
