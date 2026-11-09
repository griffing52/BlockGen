"""Re-score a bench card with the within-set near-twin gate (`bench.twins`).

The gate's input is a pure function of an arm's builds, so a card scored before
the gate existed can be brought up to date without re-rendering anything: the
arms are rebuilt exactly as the runner built them (controls, baselines, npz
submissions; each checked against the card's `structures_sha`), the twin
metrics are added to every arm's `novelty` block, and the leaderboard is
recomputed with the card's own measured noise floors. Every other number on the
card is carried over unchanged.

Also measures the gate's null: the twin rate of fresh real draws of the same n
from each split, which is what a bar "3 spreads above the real control" has to
clear honestly.

    python scripts/bench_twin_rescore.py \
        --run outputs/run_20261103_090315_bench_n_256_leaderboard_of_record
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np

from blockgen.eval.bench import baselines as bl
from blockgen.eval.bench import composite as comp
from blockgen.eval.bench import fast as fast_tier
from blockgen.eval.bench import features as ftr
from blockgen.eval.bench import scorecard as sc
from blockgen.eval.bench import splits
from blockgen.eval.bench import twins as tw
from blockgen.utils.runs import new_run_dir


def to_metric(d):
    if isinstance(d, dict) and "direction" in d and "value" in d:
        extra = {k: v for k, v in d.items()
                 if k not in ("value", "ci", "direction", "gate_failed")}
        ci = d.get("ci") or [None, None]
        ci = tuple(float("nan") if c is None else float(c) for c in ci)
        if d["value"] is None:
            return sc.Metric(value=None, ci=ci, direction=d["direction"],
                             gate_failed=d.get("gate_failed") or "null", extra=extra)
        return sc.Metric(value=float(d["value"]), ci=ci, direction=d["direction"],
                         gate_failed=d.get("gate_failed"), extra=extra)
    if isinstance(d, dict):
        return {k: to_metric(v) for k, v in d.items()}
    return d


def rebuild_arms(card: dict):
    ctx = card["context"]
    corpus, seed = ctx["corpus"], int(ctx["seed"])
    n = int(card["arms"]["real_test"]["meta"]["n"])
    split = splits.load_split(corpus, seed)
    train = splits.split_structures(split, "train")
    test = splits.split_structures(split, "test")
    bctx = sc.BenchContext(corpus=corpus, seed=seed)
    arms = {a.name: a for a in fast_tier.control_arms(
        test, train=train, n=n, rng=bctx.rng(1), corpus=corpus, seed=seed)}
    for name, structs in bl.arms(train, n, seed=seed).items():
        arms[name] = sc.ArmSpec(name, track="baseline", structures=structs)
    for name, arm in card["arms"].items():
        meta = arm.get("meta") or {}
        npz = meta.get("npz") or (meta.get("provenance") or {}).get("npz")
        if name not in arms and npz:
            arms[name] = sc.ArmSpec(name, npz=npz)
    out = {}
    for name in card["arms"]:
        structs = [s.crop_to_non_air() for s in arms[name].load()]
        structs = [s for s in structs if int(s.occupied_mask.sum()) > 0]
        want = card["arms"][name]["meta"].get("structures_sha")
        got = ftr.structures_sha(structs)
        if want and got != want:
            raise SystemExit(f"arm {name}: rebuilt builds do not match the card "
                             f"({got} != {want})")
        out[name] = structs
    return out, split, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--null-draws", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = Path(args.run)
    raw = json.loads((src / "scorecard.json").read_text())
    structs, split, n = rebuild_arms(raw)
    print(f"[twins] rebuilt {len(structs)} arms, all match the card's structures_sha",
          flush=True)

    card = sc.Scorecard(context=raw["context"], arms=to_metric(raw["arms"]),
                        warnings=list(raw.get("warnings", [])),
                        run=dict(raw["run"]), ladder=raw.get("ladder"))
    bctx = sc.BenchContext(corpus=raw["context"]["corpus"],
                           seed=int(raw["context"]["seed"]))
    per_arm = {}
    cache = {}
    for name, ss in structs.items():
        # Material probes share their source's occupancy, and the statistic
        # reads occupancy only, so they are computed once.
        okey = tuple(np.packbits(s.block_ids != 0).tobytes() + bytes(str(s.block_ids.shape), "ascii")
                     for s in ss)
        if okey not in cache:
            r0 = tw.within_set_twins(ss)
            cache[okey] = (r0, fast_tier.twin_metrics(ss, bctx, rep=r0))
        rep, metrics = cache[okey]
        card.arms[name]["novelty"].update(metrics)
        per_arm[name] = {"n": len(ss), "twin_rate": rep.rate,
                         "nn_iou_median": float(np.median(rep.nn_iou)),
                         "frac_nn_ge_0.9": float((rep.nn_iou >= 0.9).mean())}
        print(f"  {name:28s} twin_rate {rep.rate:.4f}  "
              f"nn median {per_arm[name]['nn_iou_median']:.3f}", flush=True)

    # The null: fresh real draws of the same n, from each split.
    null = {}
    for part in ("test", "val", "train"):
        pool = [s.crop_to_non_air() for s in splits.split_structures(split, part)]
        rates = []
        for d in range(args.null_draws):
            r = np.random.default_rng(1000 + d)
            take = [pool[i] for i in r.permutation(len(pool))[:n]]
            rates.append(tw.within_set_twins(take).rate)
        null[part] = {"n_pool": len(pool), "draws": args.null_draws,
                      "mean": float(np.mean(rates)), "sd": float(np.std(rates, ddof=1)),
                      "max": float(np.max(rates)), "rates": rates}
        print(f"  null {part}: mean {null[part]['mean']:.4f} "
              f"sd {null[part]['sd']:.4f} max {null[part]['max']:.4f}", flush=True)

    noise_sd = raw["context"].get("noise_floor_sd") or {}
    rows = comp.leaderboard(card, min_n=int(raw["context"].get("min_n", 16)),
                            noise_sd=noise_sd)
    checks = comp.validate(rows)
    cal = comp.Calibration.from_arm(card.arms["real_test"], "real_test", noise_sd=noise_sd)
    gate = next(g for g in comp.GATES if g.key == "voxel_twin_rate")
    floor = cal.floor["novelty.voxel_twin_rate"]
    spread = max(cal.spread.get("novelty.voxel_twin_rate", 0.0), gate.min_spread)
    bar = floor + gate.z * spread

    run_dir = Path(args.out) if args.out else new_run_dir("bench_twin_gate_rescore")
    run_dir.mkdir(parents=True, exist_ok=True)
    card.run.update({
        "dir": str(run_dir), "name": "twin-gate re-score of " + src.name,
        "note": "Arms rebuilt and verified against the source card's structures_sha; "
                "only novelty.voxel_twin_* are new, every other metric is carried over.",
        "rescored_from": str(src),
        "cmd": shlex.join([sys.executable, *sys.argv]),
        "blockscore": [r.to_json() for r in rows],
        "blockscore_validation": checks,
    })
    card.write(run_dir / "scorecard.json")
    board = comp.render_leaderboard(rows)
    (run_dir / "scorecard.md").write_text(
        sc.render_markdown(card) + "\n## leaderboard — BlockScore\n\n" + board)
    summary = {"source": str(src), "n": n, "threshold": tw.TWIN_THRESHOLD,
               "gate": {"z": gate.z, "min_spread": gate.min_spread,
                        "real_floor": floor, "spread_used": spread, "bar": bar},
               "per_arm": per_arm, "null": null,
               "status": {r.arm: {"status": r.status, "score": r.score,
                                  "disqualified": r.disqualified} for r in rows},
               "validation": checks}
    (run_dir / "twins.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[twins] bar = {floor:.4f} + {gate.z} x {spread:.4f} = {bar:.4f}")
    print(board)
    print(f"-> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
