"""Benchmark entry point.

    python -m blockgen.eval.bench --arms ar:path/to/samples.npz --tier fast

**The runner consumes `.npz` structure caches only.** It never imports a model,
loads a checkpoint, or calls a provider API. That format
(`curation.houses.save_house_cache` / `load_structures_from_cache`) is the one
artifact every track already writes -- the agentic track by design, the neural
tracks via `scripts/dump_samples.py` -- so the benchmark is decoupled from every
generator and keeps working when they change.

Real-data controls are appended as ordinary arms unless `--no-controls`, because
without a floor and a known-damage rung the other rows cannot be read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

from blockgen.eval.bench import fast as fast_tier
from blockgen.eval.bench import scorecard as sc
from blockgen.eval.bench import splits
from blockgen.eval.bench.ladder import ladder_path as ld_path
from blockgen.utils.runs import new_run_dir


def parse_arm(spec: str) -> sc.ArmSpec:
    """`name:path.npz`, `track/name:path.npz`, or bare `path.npz`."""
    if ":" in spec:
        head, path = spec.split(":", 1)
        track, _, name = head.rpartition("/")
        return sc.ArmSpec(name=name or head, track=track or "ar", npz=path)
    return sc.ArmSpec(name=Path(spec).stem, track="ar", npz=spec)


def attach_cost(arm: sc.ArmSpec) -> None:
    """Pull token/dollar accounting from a sibling agentic `metrics.json`.

    Cost is reported on its own scale and never combined with GPU-hours -- there
    is no defensible common denominator between "$0.004 of tokens per build" and
    "amortized pretraining". The renderer prints the missing side as an em dash,
    not a zero.
    """
    if arm.track != "agentic" or not arm.npz:
        return
    path = Path(arm.npz).parent / "metrics.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text())
    except Exception:
        return
    arms = data.get("arms", data)
    if not isinstance(arms, dict):
        return
    # The agentic writer names files `<arm>_<canvas>.npz`, so the key inside
    # metrics.json is the filename stem minus the canvas size -- not whatever
    # label the caller gave this arm on the command line.
    stem = Path(arm.npz).stem
    candidates = [arm.name, stem, stem.rsplit("_", 1)[0]]
    row = next((arms[c] for c in candidates if isinstance(arms.get(c), dict)), None)
    if isinstance(row, dict):
        arm.cost = {k: row[k] for k in
                    ("cost_usd", "completion_tokens", "prompt_tokens", "elapsed_s",
                     "coherence_rate", "n_builds") if k in row}


def cost_block(arm: sc.ArmSpec, coherence_rate: float | None) -> Dict[str, sc.Metric]:
    """Cost normalized per *coherent* build, on the track's own scale."""
    if not arm.cost:
        return {}
    out: Dict[str, sc.Metric] = {}
    n = float(arm.cost.get("n_builds") or 0)
    rate = arm.cost.get("coherence_rate", coherence_rate)
    coherent = (n * rate) if (n and rate) else 0.0
    for key in ("cost_usd", "completion_tokens", "elapsed_s"):
        if key in arm.cost:
            out[key] = sc.metric(float(arm.cost[key]), direction="lower_better")
    if coherent > 0:
        if "cost_usd" in arm.cost:
            out["usd_per_coherent_build"] = sc.metric(
                float(arm.cost["cost_usd"]) / coherent, direction="lower_better")
        if "completion_tokens" in arm.cost:
            out["tokens_per_coherent_build"] = sc.metric(
                float(arm.cost["completion_tokens"]) / coherent,
                direction="lower_better")
    else:
        out["usd_per_coherent_build"] = sc.skipped("no coherent builds recorded")
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m blockgen.eval.bench",
        description="Score generated structures against a fixed real reference.")
    ap.add_argument("--arms", nargs="*", default=[],
                    help="each as [track/]name:path.npz")
    ap.add_argument("--tier", choices=("fast", "full", "both"), default="fast")
    ap.add_argument("--corpus", default="houses_32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--n", type=int, default=None,
                    help="cap samples per arm (and per control)")
    ap.add_argument("--n-ref", type=int, default=512)
    ap.add_argument("--min-n", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--no-controls", action="store_true",
                    help="omit real-data controls (not recommended: no floor)")
    ap.add_argument("--backbone", default="dinov2b")
    ap.add_argument("--text-backbone", default="clipL")
    ap.add_argument("--px", type=int, default=224)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-ladder-gate", action="store_true",
                    help="report unvalidated metrics (exploration only)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    t0 = time.time()
    ctx = sc.BenchContext(corpus=args.corpus, seed=args.seed, grid=args.grid,
                          min_n=args.min_n, n_boot=args.n_boot, n_ref=args.n_ref)
    split = splits.load_split(args.corpus, args.seed)
    ctx.split_key, ctx.split_sha, ctx.sizes = split.key(), split.source_sha, split.sizes

    print(f"[bench] split {split.key()} {split.sizes}", flush=True)
    train = splits.split_structures(split, "train")
    val = splits.split_structures(split, "val")
    test = splits.split_structures(split, "test")

    rng = ctx.rng()
    if args.n_ref and args.n_ref < len(val):
        val = [val[i] for i in rng.permutation(len(val))[:args.n_ref]]

    print(f"[bench] reference: {len(val)} val builds; train pool {len(train)}",
          flush=True)
    ref = fast_tier.FastReference(train, val, ctx)

    arms = [parse_arm(a) for a in args.arms]
    for arm in arms:
        attach_cost(arm)
    if not args.no_controls:
        arms = fast_tier.control_arms(test, train=train, n=args.n,
                                      rng=ctx.rng(1)) + arms
    if not arms:
        ap.error("no arms to score (pass --arms, or drop --no-controls)")

    run_dir = Path(args.out) if args.out else new_run_dir("bench")
    run_dir.mkdir(parents=True, exist_ok=True)

    card = sc.Scorecard(
        context={"corpus": args.corpus, "seed": args.seed, "grid": args.grid,
                 "split_key": split.key(), "split_sha": split.source_sha,
                 "n_train": len(train), "n_val": len(val), "n_test": len(test),
                 "n_ref_used": len(val), "min_n": args.min_n,
                 "vocab_size": ref.vocab.vocab_size,
                 "palette_keys_exact": len(ref.exact_keys),
                 "palette_keys_family": len(ref.family_keys),
                 "bootstrap": {"n_boot": args.n_boot, "alpha": ctx.alpha,
                               "unit": "structure"}},
        run={"dir": str(run_dir), "git_sha": sc.git_sha(),
             "cmd": " ".join(sys.argv), "tier": args.tier})

    full_ref = None
    if args.tier in ("full", "both"):
        from blockgen.eval.bench import features as ftr
        from blockgen.eval.bench import full as full_tier
        view = ftr.ViewConfig(px=args.px)
        ctx.text_backbone = args.text_backbone
        full_ref = full_tier.build_reference(
            train, val, test, ctx, view=view, backbone=args.backbone,
            corpus=args.corpus, device=args.device,
            require_ladder=not args.no_ladder_gate)
        card.context["view"] = {"key": view.key(), "px": view.px,
                                "views": [list(v) for v in view.views]}
        card.context["backbone"] = {"image": args.backbone,
                                    "text": args.text_backbone, "pool": "mean"}
        card.context["mmd_sigma"] = full_ref.sigma
        if full_ref.ladder is not None:
            card.ladder = {"path": str(ld_path(args.backbone, view.key())),
                           "passing": full_ref.ladder.passing,
                           "failing": {m: full_ref.ladder.failed_gate(m)
                                       for m in full_ref.ladder.gates
                                       if not full_ref.ladder.passed(m)}}

    for arm in arms:
        print(f"[bench] scoring {arm.name} ({arm.track})", flush=True)
        try:
            blocks = fast_tier.score_fast(arm, ref, card)
            if full_ref is not None:
                from blockgen.eval.bench import full as full_tier
                extra = full_tier.score_full(arm, full_ref, corpus=args.corpus,
                                             device=args.device, card=card)
                for section, block in extra.items():
                    blocks.setdefault(section, {}).update(block)
        except Exception as exc:                     # one bad arm must not sink the run
            card.warn(f"arm {arm.name!r} failed: {type(exc).__name__}: {exc}")
            continue
        rate = None
        coh = blocks.get("coherence", {}).get("lcc_ratio")
        if coh:
            rate = coh["gen"]["mean"]
        cost = cost_block(arm, rate)
        if cost:
            blocks["cost"] = cost
        card.add_arm(arm.name, blocks)

    card.run["elapsed_s"] = round(time.time() - t0, 2)
    json_path = card.write(run_dir / "scorecard.json")
    md_path = run_dir / "scorecard.md"
    md_path.write_text(sc.render_markdown(card))

    print()
    print(sc.render_markdown(card))
    print(f"-> {json_path}")
    print(f"-> {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
