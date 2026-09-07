"""Does the benchmark reward *more* realistic output, or just detect artifacts?

The validation ladder answers a binary question -- can this metric tell damaged
builds from real ones. A benchmark that ranks submissions has to answer a harder
one: **does the number move smoothly and in the right direction as quality
changes?** A metric can pass every ladder gate by firing on an artifact while
being flat across the range that actually separates two submitted models.

Four dose axes, each with a known ground-truth ordering:

    solidify    fraction of interior volume filled, from the walls inward
    occ_noise   fraction of blocks displaced to adjacent air
    canon       decimation to a smaller grid, then back
    mixture     fraction of the arm that is real, the rest canon-8

`mixture` is the one that most resembles a real generator: a model does not
degrade uniformly, it produces some good builds and some failures, and the
question a leaderboard answers is whether an arm with more good builds scores
better. It is also the only axis on which a per-sample detector and a
distribution metric are expected to behave differently.

Reported per metric: Spearman rho against the dose, whether the sequence is
strictly monotone, and the dynamic range in units of the metric's own noise
floor. A metric that is monotone but spans one noise floor cannot rank anything.

    python scripts/bench_doseresponse.py --n 64 --out outputs/run_.../
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

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


def ladders(pool: Sequence[Structure], rng: np.random.Generator,
            damaged: Sequence[Structure]) -> Dict[str, List[Rung]]:
    """`{axis: [(dose, structures), ...]}`, dose 0 = untouched real."""
    pool = list(pool)
    n = len(pool)

    def mix(alpha: float) -> List[Structure]:
        """`alpha` of the arm is real, the rest is canon-8."""
        k = int(round(alpha * n))
        order = rng.permutation(n)
        return ([pool[i] for i in order[:k]]
                + [damaged[i] for i in order[k:]])

    return {
        "solidify": [(f, [pb.partial_solidify(s, f, rng) for s in pool])
                     for f in (0.0, 0.25, 0.5, 0.75, 1.0)],
        "occ_noise": [(p, [pb.occupancy_noise(s, p, rng) for s in pool] if p else pool)
                      for p in (0.0, 0.02, 0.05, 0.10, 0.20)],
        "canon": [(1.0 - d / 32.0, pb.canon(pool, d) if d < 32 else pool)
                  for d in (32, 24, 20, 16, 12, 8)],
        # dose = fraction NOT real, so every axis runs 0 (perfect) -> 1 (worst)
        "mixture": [(1.0 - a, mix(a)) for a in (1.0, 0.75, 0.5, 0.25, 0.0)],
    }


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default="houses_32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=64, help="builds per rung")
    ap.add_argument("--n-ref", type=int, default=399)
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
    damaged = pb.canon(pool, 8)

    axes = ladders(pool, rng, damaged)

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
    for axis, rungs in axes.items():
        results[axis] = {}
        doses = [d for d, _ in rungs]
        for mname, fn in metrics.items():
            vals = []
            for d, structs in rungs:
                vals.append(fn([s.crop_to_non_air() for s in structs]))
                print(f"  [{axis}/{mname}] dose {d:.2f} -> {vals[-1]:.4f}", flush=True)
            results[axis][mname] = analyse(doses, vals, noise[mname])

    run_dir = Path(args.out) if args.out else new_run_dir("doseresponse")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "doseresponse.json").write_text(json.dumps(
        {"context": {"corpus": args.corpus, "seed": args.seed, "n": len(pool),
                     "n_ref": len(val), "noise_sd": noise,
                     "git_sha": sc.git_sha()},
         "axes": results}, indent=2) + "\n")

    md = render(results, noise)
    (run_dir / "doseresponse.md").write_text(md)
    print()
    print(md)
    print(f"-> {run_dir}")
    return 0


def render(results: Dict[str, Dict[str, dict]], noise: Dict[str, float]) -> str:
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
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
