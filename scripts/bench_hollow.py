"""The converse probe: is the render tier blind to hollowing as well as filling?

T25/T26a measured one direction. `solidify` fills every block a camera cannot
see and MV-DINO-KID does not move, which is the paper's evidence that a
render-and-embed metric scores the facade. Read on its own that finding invites
one objection: maybe *adding* hidden mass is a special case -- a filled interior
is still a closed shell, and closed shells are what the corpus is made of.

`probes.hollow` is the same claim run backwards. It deletes every block a camera
cannot see, leaving the exterior air region -- and therefore every rendered face
-- exactly invariant (see the probe's docstring for why that is exact and not
approximate). The result is the failure mode the wider 3D-generation field
actually ships: a shell with nothing inside it. If the render metric is blind to
interior mass, it must be blind in both directions; if it turns out to *see*
hollowing, that bounds the paper's claim, which is worth more than a second
confirmation.

Both probes are scored here on the same pool, against the same reference, so the
+31% / -13% asymmetry in block count is the only thing that differs between the
two readings.

    python scripts/bench_hollow.py --n 128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

from blockgen.eval.bench import distances as D
from blockgen.eval.bench import features as ft
from blockgen.eval.bench import geometry as geom
from blockgen.eval.bench import ladder as ld
from blockgen.eval.bench import probes as pb
from blockgen.eval.bench import scorecard as sc
from blockgen.eval.bench import splits, stats
from blockgen.utils.data import Structure
from blockgen.utils.runs import new_run_dir

#: The two structural probes, as `name -> builder`. `real_test` is the floor
#: every separation is measured from: a fresh real draw, not the reference
#: scored against itself (which is an exact zero and calibrates nothing).
ARMS: Dict[str, Callable[[Structure], Structure]] = {
    "real_test": lambda s: s,
    "real@hollow": pb.hollow,
    "real@solidify": pb.solidify,
}


def block_stats(pool: Sequence[Structure],
                arms: Dict[str, List[Structure]]) -> Dict[str, dict]:
    """Per-build block-count change, the units the paper states damage in.

    Reported as a *distribution* rather than a total because a handful of large
    builds would otherwise carry the number: "median 31% more blocks, 91% of
    builds affected" is the claim `solidify` is documented with, and the hollow
    rung has to be stated the same way to be read beside it.
    """
    base = np.array([int(s.occupied_mask.sum()) for s in pool], dtype=float)
    out: Dict[str, dict] = {}
    for name, structs in arms.items():
        got = np.array([int(s.occupied_mask.sum()) for s in structs], dtype=float)
        rel = (got - base) / np.maximum(base, 1.0)
        out[name] = {
            "blocks_mean": float(got.mean()),
            "delta_median": float(np.median(rel)),
            "delta_mean": float(rel.mean()),
            "delta_p90": float(np.percentile(np.abs(rel), 90)),
            "delta_max_abs": float(np.abs(rel).max()),
            "affected_frac": float((np.abs(rel) > 0).mean()),
        }
    return out


def geometry_scalars(arms: Dict[str, List[Structure]]) -> Dict[str, dict]:
    """A few geometry-tier scalars per arm, so the damage is visible in numbers
    that are not distribution distances -- interior ratio above all, since that
    is the quantity both probes are built to destroy."""
    keys = ("interior_ratio_open", "interior_ratio_sealed", "thickness_mean",
            "wall_frac", "surface_to_volume")
    out: Dict[str, dict] = {}
    for name, structs in arms.items():
        table = geom.geometry_table(structs)
        out[name] = {k: float(np.mean(table[k])) for k in keys}
    return out


def write_examples(arms: Dict[str, List[Structure]], run_dir: Path, k: int) -> None:
    """Render the probe before writing numbers about it -- T21's standing lesson.

    One row per arm, the same `k` builds in every row, so the rows line up. The
    row that matters is the one a reader cannot tell from the real row: that is
    the finding, drawn. A cutaway would show the difference, and deliberately is
    not what is drawn here -- the claim is about what the *metric's own view*
    contains.
    """
    from blockgen.renderer.grid import save_grid

    tiles = [s for structs in arms.values() for s in structs[:k]]
    save_grid(tiles, str(run_dir / "samples.png"), cols=k, tile_px=192,
              verbose=False)
    (run_dir / "samples.txt").write_text(
        "rows, top to bottom: " + ", ".join(arms) + "\n")


def pixel_deltas(pool: Sequence[Structure], arms: Dict[str, List[Structure]],
                 view: "ft.ViewConfig", k: int = 32) -> Dict[str, dict]:
    """How much of the metric's *input* changes -- blindness, or no information?

    A metric cannot be blind to what its input does not contain, and the two
    readings call for different corrections: an encoder that sees the pixels and
    ignores them could be swapped, while a view protocol that never renders the
    difference cannot be fixed by any encoder. `hollow` leaves the exterior face
    set exactly invariant, so the only pixels that can move are the ones seen
    *through* a door or window; `solidify` is the same argument in reverse.

    Reported in the units F34.2 used for the cutaway pilot -- mean absolute
    pixel difference over all views -- plus the share of views that come back
    bit-identical, which is the unambiguous half of the statement.
    """
    src = list(pool)[:k]
    base = ft.render_views(src, view, verbose=False)
    out: Dict[str, dict] = {}
    for name, structs in arms.items():
        imgs = ft.render_views([s.crop_to_non_air() for s in structs][:k], view,
                               verbose=False)
        diffs = np.array([np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()
                          for a, b in zip(base, imgs)], dtype=float)
        out[name] = {
            "n_views": int(len(diffs)),
            "mean_abs_pixel_delta": float(diffs.mean()),
            "max_abs_pixel_delta": float(diffs.max()),
            "identical_view_frac": float((diffs == 0).mean()),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default="houses_32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=128,
                    help="builds per arm; the protocol's min_n is 128")
    ap.add_argument("--n-ref", type=int, default=399)
    ap.add_argument("--n-rep", type=int, default=200,
                    help="subsampling reps for the interval")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-render", action="store_true",
                    help="geometry only; skips the GPU tier")
    ap.add_argument("--examples", type=int, default=6,
                    help="builds per arm rendered into samples.png (0 to skip)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    split = splits.load_split(args.corpus, args.seed)
    val = splits.split_structures(split, "val")
    test = splits.split_structures(split, "test")
    if args.n_ref < len(val):
        val = [val[i] for i in
               np.random.default_rng(args.seed).permutation(len(val))[:args.n_ref]]

    # Same pool construction as `fast.control_arms`, so `real@solidify` here is
    # the arm the leaderboard scores and not a differently-drawn cousin.
    pool = [s.crop_to_non_air() for s in test]
    pool = [pool[i] for i in rng.permutation(len(pool))[:args.n]]
    print(f"[hollow] reference {len(val)}, pool {len(pool)}", flush=True)

    arms = {name: [fn(s) for s in pool] for name, fn in ARMS.items()}
    ctx = sc.BenchContext(corpus=args.corpus, seed=args.seed, n_ref=len(val))

    metrics: Dict[str, Callable[[Sequence[Structure]], np.ndarray]] = {}
    scorers: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {}
    refs: Dict[str, np.ndarray] = {}
    noise: Dict[str, dict] = {}

    held = [s.crop_to_non_air() for s in test]
    std = geom.Standardizer.fit(geom.descriptor(val))
    refs["geom_kid"] = geom.geom_features(val, std)
    metrics["geom_kid"] = lambda xs: geom.geom_features(xs, std)
    scorers["geom_kid"] = lambda a, b: geom.geom_kid(a, b)
    g_mean, g_sd, g_sem = ld.noise_floor(
        lambda r, p: geom.geom_kid(p, r), refs["geom_kid"],
        geom.geom_features(held, std), min(args.n, len(held) - 1), reps=48,
        rng=np.random.default_rng(args.seed + 5))
    noise["geom_kid"] = {"mean": g_mean, "sd": g_sd, "sem": g_sem}

    if not args.no_render:
        view = ft.ViewConfig()
        refs["mv_dino_kid"] = ft.pooled(ft.load_or_build(
            ft.FeatureKey(args.corpus, "val", view.key(), "dinov2b"), val, view,
            device=args.device, verbose=False), "mean")
        test_feats = ft.pooled(ft.load_or_build(
            ft.FeatureKey(args.corpus, "test", view.key(), "dinov2b"), test, view,
            device=args.device, verbose=False), "mean")
        metrics["mv_dino_kid"] = lambda xs: ft.pooled(ft.embed_views(
            xs, view, "dinov2b", args.device, verbose=False), "mean")
        scorers["mv_dino_kid"] = lambda a, b: D.kid(a, b)
        d_mean, d_sd, d_sem = ld.noise_floor(
            lambda r, p: D.kid(p, r), refs["mv_dino_kid"], test_feats,
            min(args.n, len(test_feats) - 1), reps=48,
            rng=np.random.default_rng(args.seed + 6))
        noise["mv_dino_kid"] = {"mean": d_mean, "sd": d_sd, "sem": d_sem}
        ctx.image_backbone = "dinov2b"
        ctx.view = {"key": view.key()}

    pixels = (pixel_deltas(pool, arms, view) if not args.no_render else {})

    results: Dict[str, Dict[str, dict]] = {}
    for mname, featurize in metrics.items():
        results[mname] = {}
        sd = noise[mname]["sd"]
        for arm, structs in arms.items():
            feats = featurize([s.crop_to_non_air() for s in structs])
            point, lo, hi = stats.subsample_ci(
                scorers[mname], feats, refs[mname], frac=0.8, n_rep=args.n_rep,
                rng=ctx.rng(7))
            results[mname][arm] = {
                "value": point, "ci": [lo, hi],
                # Two denominators, deliberately. `sep_sd` is the pillar unit the
                # composite uses -- distance from the *null mean* of many real
                # draws. `sep_from_real_sd` is the distance from the single
                # real_test draw scored in this run, which is the row a reader
                # compares against on the leaderboard.
                "sep_sd": (point - noise[mname]["mean"]) / sd if sd > 0 else float("nan"),
                "mmd": float(np.sqrt(max(point, 0.0))),
            }
            print(f"  [{mname}] {arm:<14} {point:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]  {results[mname][arm]['sep_sd']:+.2f} sd",
                  flush=True)
        real = results[mname]["real_test"]["value"]
        for arm, row in results[mname].items():
            row["sep_from_real_sd"] = (row["value"] - real) / sd if sd > 0 else float("nan")

    payload = {
        "context": {"corpus": args.corpus, "seed": args.seed, "n": len(pool),
                    "n_ref": len(val), "n_rep": args.n_rep,
                    "split_key": split.key(),
                    "noise": noise, "git_sha": sc.git_sha()},
        "realism": results,
        "blocks": block_stats(pool, arms),
        "geometry_scalars": geometry_scalars(arms),
        "pixels": pixels,
    }
    run_dir = Path(args.out) if args.out else new_run_dir("bench_hollow")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "hollow.json").write_text(json.dumps(payload, indent=2) + "\n")
    if args.examples:
        write_examples(arms, run_dir, args.examples)
    md = render(payload)
    (run_dir / "hollow.md").write_text(md)
    print()
    print(md)
    print(f"-> {run_dir}")
    return 0


def render(payload: dict) -> str:
    ctx = payload["context"]
    lines = ["# The converse probe — hollowing, measured beside solidify", ""]
    lines.append(f"n={ctx['n']} per arm, {ctx['n_ref']}-build real reference, "
                 f"corpus `{ctx['corpus']}`, seed {ctx['seed']}.")
    lines.append("")
    lines.append("Noise floors (sd of a fresh real draw of the same size): "
                 + ", ".join(f"`{k}` {v['sd']:.4f} (null mean {v['mean']:+.4f})"
                             for k, v in ctx["noise"].items()))
    lines.append("")
    for mname, rows in payload["realism"].items():
        lines.append(f"## {mname}")
        lines.append("")
        lines.append("| arm | value | 95% interval | sd from null | sd from real_test |")
        lines.append("|---|---|---|---|---|")
        for arm, r in rows.items():
            lines.append(f"| `{arm}` | {r['value']:+.4f} | "
                         f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | "
                         f"{r['sep_sd']:+.1f} | {r['sep_from_real_sd']:+.1f} |")
        lines.append("")
    lines.append("## Damage, in blocks")
    lines.append("")
    lines.append("| arm | mean blocks | median change | mean change | affected |")
    lines.append("|---|---|---|---|---|")
    for arm, b in payload["blocks"].items():
        lines.append(f"| `{arm}` | {b['blocks_mean']:.0f} | "
                     f"{b['delta_median']:+.1%} | {b['delta_mean']:+.1%} | "
                     f"{b['affected_frac']:.1%} |")
    lines.append("")
    if payload.get("pixels"):
        lines.append("## What the camera is given")
        lines.append("")
        lines.append("The `real_test` row is the control: re-rendering the *same* "
                     "build is not bit-deterministic, so `identical views` is only "
                     "readable against it.")
        lines.append("")
        lines.append("| arm | views | mean abs pixel delta | max | identical views |")
        lines.append("|---|---|---|---|---|")
        for arm, p in payload["pixels"].items():
            lines.append(f"| `{arm}` | {p['n_views']} | "
                         f"{p['mean_abs_pixel_delta']:.4f} | "
                         f"{p['max_abs_pixel_delta']:.4f} | "
                         f"{p['identical_view_frac']:.1%} |")
        lines.append("")
    lines.append("## Geometry scalars (arm mean)")
    lines.append("")
    keys = sorted(next(iter(payload["geometry_scalars"].values())))
    lines.append("| arm | " + " | ".join(f"`{k}`" for k in keys) + " |")
    lines.append("|" + "---|" * (len(keys) + 1))
    for arm, g in payload["geometry_scalars"].items():
        lines.append(f"| `{arm}` | " + " | ".join(f"{g[k]:.4f}" for k in keys) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
