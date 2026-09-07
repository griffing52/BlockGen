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

**A run says who it is.** Eleven cards accumulated in `outputs/` before this,
every one of them in a directory called `run_<stamp>_bench`, distinguishable
only by opening the JSON and reading the arm names -- which is why `--name` and
`--note` exist, why `--name` also names the directory (keeping the `bench_`
prefix, so every `run_*_bench*` glob in the notes keeps working), and why an
unnamed run now says so on the way out. The *identity* of a run is still its
directory name and never `run.dir`: two of those eleven cards carry an absolute
scratchpad path there, from outside `outputs/` entirely.

**`cmd` is not re-runnable and never was.** `" ".join(sys.argv)` starts with the
absolute path of `__main__.py`, so the string that looks like the command is one
nobody can paste. `bench/2` keeps it verbatim for the eyeball habit and adds
`argv` (a list, so a path with a space survives) plus `rerun`
(`shlex.join`-quoted, pasteable). That is the whole reason `build_parser()` is
factored out of `main()`: a test can re-parse `rerun` through the real parser
and prove the round trip, which a parser built inside `main` cannot support.

**The context block is the object the arms were scored against.** Until
`bench/2` this module hand-built a parallel `context` dict beside `BenchContext`
and the two disagreed -- `palette_level`, `dup_threshold`, `n_ref` and `sizes`
went into every score and onto no card. There is now one source of that block
(`BenchContext.to_json`), the run-computed extras arrive through `ctx.derived`,
and the late-measured values (the full tier's view/backbone, the noise floor)
are set on `ctx` and re-serialized once.
"""

from __future__ import annotations

import argparse
import json
import shlex
import socket
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from blockgen.eval.bench import examples as ex
from blockgen.eval.bench import protocol as proto
from blockgen.eval.bench import fast as fast_tier
from blockgen.eval.bench import scorecard as sc
from blockgen.eval.bench import splits
from blockgen.eval.bench.ladder import ladder_path as ld_path
from blockgen.utils.data import Structure
from blockgen.utils.runs import new_run_dir

#: Where `python -m tools.lab` serves the leaderboard (tools/lab/__init__.py:109).
#: Copied rather than imported, because `blockgen/` may not depend on `tools/`;
#: a stale port costs one wrong URL in a print, and nothing else.
LAB_PORT = 8765


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


def measured_noise(card: sc.Scorecard, ref, full_ref, test, args) -> Dict[str, float]:
    """Real-draw noise floor for each ladder-gated distribution metric.

    One draw of held-out real builds against the same reference every arm is
    scored against, repeated -- i.e. exactly the protocol the arms themselves go
    through, with a real generator substituted in. Anything cheaper measures a
    different quantity and mislabels the axis (see `composite.Calibration`).
    """
    from blockgen.eval.bench import geometry as geom
    from blockgen.eval.bench import ladder as ld

    n = int((card.arms.get("real_test", {}).get("meta") or {}).get("n") or 0)
    out: Dict[str, float] = {}
    if n < 4 or len(test) < n + 1:
        return out
    rng = card_rng(args)
    try:
        pool = geom.geom_features([s.crop_to_non_air() for s in test], ref.geom_std)
        _, sd, _ = ld.noise_floor(lambda r, p: geom.geom_kid(p, r),
                                  ref.geom_ref, pool, n, reps=48, rng=rng)
        out["realism.geom_kid"] = float(sd)
    except Exception as exc:
        card.warn(f"geom_kid noise floor unavailable: {type(exc).__name__}: {exc}")
    if full_ref is not None:
        from blockgen.eval.bench import distances as D
        try:
            _, sd, _ = ld.noise_floor(lambda r, p: D.kid(p, r), full_ref.ref_feats,
                                      full_ref.calib_feats, n, reps=48, rng=rng)
            out["realism.mv_dino_kid"] = float(sd)
        except Exception as exc:
            card.warn(f"mv_dino_kid noise floor unavailable: "
                      f"{type(exc).__name__}: {exc}")
    return out


def card_rng(args) -> "np.random.Generator":
    import numpy as np
    return np.random.default_rng(args.seed + 991)


def run_slug(name: str, limit: int = 40) -> str:
    """`--name` as a directory-safe suffix, truncated on a WORD boundary.

    A naive `name[:limit]` turns "ontology arms vs agentic ontology probe" into
    `..._bench_ontology_arms_vs_agentic_ontol`, which is worse than no label at
    all: the eye reads that as a typo rather than as an ellipsis. Cutting back to
    the last underscore costs one word and keeps every word that survives
    readable. A single word longer than the limit has no boundary to cut back to
    and is hard-cut, which is still better than a 90-character directory name.

    Case is preserved deliberately: `utils.runs.new_run_dir` does not fold case
    either, so a slug that lowercased would produce the only directory in
    `outputs/` that disagrees with the name it was given.
    """
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    while "__" in safe:
        safe = safe.replace("__", "_")
    safe = safe.strip("_")
    if len(safe) <= limit:
        return safe
    cut = safe[:limit]
    head, sep, _tail = cut.rpartition("_")
    return (head if sep and head else cut).strip("_")


def _lab_discoverable(run_dir: Path) -> bool:
    """Will `tools/lab` ever find an npz written into this directory?

    The lab discovers arm datasets by rglobbing `outputs/run_*/**/*.npz` and
    `outputs/bench_arms/**/*.npz` (`catalog.ARM_ROOTS`), so a run written with
    `--out /tmp/whatever` produces example builds that are real files, are
    recorded on the card, and are reachable by nothing: `example_build_ids`
    returns `{}` and the leaderboard shows an empty strip. The examples are
    still written -- the file is the artifact, the browsability is a bonus --
    but the runner says out loud which of the two happened, because a silently
    dead build id is the harder bug to find later.

    This duplicates a rule that lives in `tools/lab/catalog.py`, on purpose:
    `blockgen/` must not import the lab. It is a warning, never a decision, so
    the copies drifting costs a warning and nothing else.
    """
    try:
        rel = run_dir.resolve().relative_to(Path("outputs").resolve())
    except (ValueError, OSError):
        return False
    return bool(rel.parts) and rel.parts[0].startswith("run_")


def build_parser() -> argparse.ArgumentParser:
    """The CLI, factored out of `main` so `run.rerun` can be tested.

    `run.rerun` claims to be a copy-pasteable command; the only honest way to
    check that claim is to re-parse the string through the parser that produced
    the run (`tests/eval/test_runner_naming.py`), and a parser constructed
    inside `main()` cannot be reached to do it.
    """
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
    ap.add_argument("--baselines", action="store_true",
                    help="add the procedural baselines (blockgen.eval.bench.baselines): "
                         "the floor a leaderboard needs, fitted on train only")
    ap.add_argument("--backbone", default="dinov2b")
    ap.add_argument("--text-backbone", default="clipL")
    ap.add_argument("--px", type=int, default=224)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-ladder-gate", action="store_true",
                    help="report unvalidated metrics (exploration only)")
    ap.add_argument("--no-head-to-head", action="store_true",
                    help="skip the paired significance tests between arms")
    ap.add_argument("--n-rep", type=int, default=200,
                    help="resamples per head-to-head comparison")
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default="",
                    help="what this run is asking, in words: 'ontology arms v2'. "
                         "Free text; also names the run directory "
                         "(run_<stamp>_bench_<slug>)")
    ap.add_argument("--note", default="",
                    help="one line of context for the leaderboard (display only)")
    ap.add_argument("--examples", type=int, default=fast_tier.EXAMPLES_K,
                    help="example builds to keep per arm, written as one "
                         "examples_<max_dim>.npz in the run dir; 0 disables")
    return ap


def main(argv: List[str] | None = None) -> int:
    ap = build_parser()
    # Kept as DATA, not as a joined string: `argv` goes on the card verbatim so
    # an argument containing a space survives the round trip, and `rerun` is
    # built from it by `shlex.join` rather than by hopeful quoting.
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    args = ap.parse_args(argv)
    rerun = shlex.join(["python", "-m", "blockgen.eval.bench", *argv])

    t0 = time.time()
    ctx = sc.BenchContext(corpus=args.corpus, seed=args.seed, grid=args.grid,
                          min_n=args.min_n, n_boot=args.n_boot, n_ref=args.n_ref,
                          text_backbone=args.text_backbone)
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
    if args.baselines:
        from blockgen.eval.bench import baselines as bl
        # Fitted on train only: a baseline that had seen val or test would be
        # competing with an advantage no submitted model has.
        n_base = args.n or 64
        arms = [sc.ArmSpec(
                    name, track="baseline", structures=structs,
                    # Built here rather than in `baselines.arms`, which returns
                    # structures and knows nothing about the run that asked for
                    # them. Without this an in-memory baseline renders as
                    # "nothing on disk to show" -- the same sentence the page
                    # shows for a dataset join that failed.
                    provenance_override={
                        "writer": "in_memory",
                        "builder": "blockgen.eval.bench.baselines.arms",
                        # `.get`, not `[...]`: a baseline added without a recipe
                        # should cost one empty cell on the page, not a run.
                        "recipe": bl.RECIPES.get(name, ""),
                        "drawn_from": f"split:{args.corpus}:train",
                        "n_requested": n_base,
                        # Mirrors `baselines.arms`, which seeds the i-th recipe
                        # in ORDER with `seed + i`.
                        "seed": args.seed + i})
                for i, (name, structs)
                in enumerate(bl.arms(train, n_base, seed=args.seed).items())
                ] + arms
    if not args.no_controls:
        arms = fast_tier.control_arms(test, train=train, n=args.n,
                                      rng=ctx.rng(1), corpus=args.corpus,
                                      seed=args.seed) + arms
    if not arms:
        ap.error("no arms to score (pass --arms, or drop --no-controls)")

    # The `bench_` prefix is kept in front of the slug so that every
    # `outputs/run_*_bench*` glob written in anyone's notes keeps matching.
    slug = run_slug(args.name)
    run_dir = Path(args.out) if args.out else new_run_dir(
        f"bench_{slug}" if slug else "bench")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Measurable only now that the reference exists, and part of the context an
    # arm was scored against, so it goes onto `ctx` rather than into a second
    # dict beside it. See `BenchContext.to_json`.
    ctx.derived = {"n_train": len(train), "n_val": len(val), "n_test": len(test),
                   "n_ref_used": len(val),
                   "vocab_size": ref.vocab.vocab_size,
                   "palette_keys_exact": len(ref.exact_keys),
                   "palette_keys_family": len(ref.family_keys)}

    git = sc.git_info()
    # Built FROM `sc.RUN_KEYS` rather than beside it: the contract and the writer
    # are then one edit, and a key added to the tuple without a value here fails
    # loudly on this line instead of silently producing a card that claims to
    # satisfy a contract it does not.
    run_fields: Dict[str, Any] = {
        # NOT identity -- readers use the directory name (`path.parent.name`).
        # Kept because things already read it, and because it is the only record
        # of an `--out` that pointed somewhere else entirely.
        "dir": str(run_dir),
        "name": args.name,               # verbatim, spaces and all
        "note": args.note,
        "git_sha": git["sha"],
        "git_branch": git["branch"],
        # None means "we could not ask git", which is not the same as clean, and
        # the page renders the two differently.
        "git_dirty": git["dirty"],
        # Written here, beside t0, so a run that crashes late still carries the
        # time it started.
        "started_at": sc.iso_utc(t0),
        "finished_at": "",               # filled beside elapsed_s
        "cmd": " ".join(sys.argv),       # legacy; argv[0] is absolute
        "rerun": rerun,
        "argv": list(argv),
        "tier": args.tier,
        "host": socket.gethostname(),
        # As REQUESTED, not as resolved: "cuda" here with no GPU present is a
        # fact about the invocation, and the fallback is reported by the tier.
        "device": args.device,
        "elapsed_s": 0.0,
    }
    undeclared = sorted(set(run_fields) - set(sc.RUN_KEYS))
    if undeclared:
        raise AssertionError(f"run fields not declared in scorecard.RUN_KEYS: "
                            f"{undeclared}")
    card = sc.Scorecard(context=ctx.to_json(),
                        run={k: run_fields[k] for k in sc.RUN_KEYS})

    full_ref = None
    if args.tier in ("full", "both"):
        from blockgen.eval.bench import features as ftr
        from blockgen.eval.bench import full as full_tier
        view = ftr.ViewConfig(px=args.px)
        ctx.px, ctx.view = args.px, {"key": view.key(), "px": view.px,
                                     "views": [list(v) for v in view.views]}
        full_ref = full_tier.build_reference(
            train, val, test, ctx, view=view, backbone=args.backbone,
            corpus=args.corpus, device=args.device,
            require_ladder=not args.no_ladder_gate)
        # Set on `ctx`, which is what emits `backbone` -- and only ever on the
        # full tier, so a fast-tier card cannot claim a text backbone it never
        # loaded.
        ctx.image_backbone = args.backbone
        ctx.mmd_sigma = full_ref.sigma
        if full_ref.ladder is not None:
            card.ladder = {"path": str(ld_path(args.backbone, view.key())),
                           "passing": full_ref.ladder.passing,
                           "failing": {m: full_ref.ladder.failed_gate(m)
                                       for m in full_ref.ladder.gates
                                       if not full_ref.ladder.passed(m)}}

    # Which pinned protocol, if any, this run satisfies -- recorded on the card
    # as what the code believed at scoring time. The lab RE-DERIVES it rather
    # than trusting this, so tightening a protocol later takes effect on the next
    # page load instead of invalidating every card on disk.
    verdict = proto.verdict(card.context, card.run, arms=card.arms)
    card.run["protocol"] = verdict
    _pro = proto.get(verdict.get("id"))
    slate: Tuple[str, ...] = _pro.slate.prompts if _pro else ()

    geom_feats: Dict[str, "np.ndarray"] = {}
    dino_feats: Dict[str, "np.ndarray"] = {}
    # arm -> (the k chosen builds, their rows in that arm's own scored list), in
    # card order, which is the order the examples cache is written in.
    picked: "OrderedDict[str, Tuple[List[Structure], List[int]]]" = OrderedDict()
    for arm in arms:
        print(f"[bench] scoring {arm.name} ({arm.track})", flush=True)
        ex_sink: List[Tuple[List[Structure], List[int]]] = []
        try:
            blocks = fast_tier.score_fast(
                arm, ref, card, sink=geom_feats,
                examples=ex_sink if args.examples > 0 else None,
                # Without this, `--examples 12` would silently cap at the
                # picker's own default.
                examples_k=args.examples,
                # A protocol may pin the prompts every arm should render, so the
                # leaderboard's columns line up. Empty for an unconditional
                # protocol, and ignored by any arm without prompts.
                examples_slate=slate)
            # Truncated HERE, between the two tiers, not after the loop:
            # `ex_sink[0]` holds this arm's entire non-empty list alive, and the
            # next call is where a run's worth of DINO features gets allocated.
            # Keeping only the k chosen builds puts peak retention at
            # k x n_arms (~128 builds) instead of every arm's full set.
            try:
                if ex_sink:
                    structs, rows = ex_sink[0]
                    picked[arm.name] = ([structs[i] for i in rows], list(rows))
            except Exception as exc:
                # D21: its own guard. One arm's example strip is never worth the
                # two hours of scoring standing behind it.
                card.warn(f"arm {arm.name!r}: example builds unavailable: "
                          f"{type(exc).__name__}: {exc}")
            finally:
                ex_sink.clear()
            if full_ref is not None:
                from blockgen.eval.bench import full as full_tier
                extra = full_tier.score_full(arm, full_ref, corpus=args.corpus,
                                             device=args.device, card=card,
                                             sink=dino_feats)
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

    # --- the leaderboard ----------------------------------------------------
    # Assembled last because it is calibrated against the `real_test` control
    # arm scored in this same run: the floor is a property of the corpus, the
    # split, n and the protocol, and importing one from another run would rank
    # arms against conditions they were not measured under.
    board_md = ""
    if not args.no_controls and "real_test" in card.arms:
        from blockgen.eval.bench import composite as comp
        # The unit BlockScore is quoted in: how far a *fresh* real sample of the
        # same size lands from the reference, under this run's exact protocol.
        # Measured here rather than imported, because it depends on n, on the
        # reference, and on the split.
        noise_sd = measured_noise(card, ref, full_ref, test, args)
        ctx.noise_floor_sd = noise_sd
        rows = comp.leaderboard(card, min_n=args.min_n, noise_sd=noise_sd)
        checks = comp.validate(rows)
        card.run["blockscore"] = [r.to_json() for r in rows]
        card.run["blockscore_validation"] = checks
        failed = [k for k, ok in checks.items() if not ok]
        if failed:
            # The aggregate is held to the same standard as the metrics: if it
            # cannot reject the strategies it was built to reject, it does not
            # get to rank anything quietly.
            card.warn("BlockScore self-validation FAILED: " + ", ".join(failed)
                      + " — the aggregate did not reject a known-degenerate "
                        "control; do not quote the ranking")
        board_md = ("## leaderboard — BlockScore\n\n"
                    + comp.render_leaderboard(rows)
                    + "\nSelf-validation: "
                    + ("all checks pass" if not failed else "FAILED " + ", ".join(failed))
                    + f" ({len(checks)} checks)\n")
    elif args.no_controls:
        card.warn("--no-controls: no real floor, so no BlockScore was computed")

    # Re-serialized ONCE, here, now that every late-measured value has landed on
    # `ctx` -- the full tier's view and backbones, and the noise floor above.
    # One writer for the block, one place it is built.
    card.context = ctx.to_json()

    # --- head-to-head: which differences are actually resolved --------------
    # Marginal intervals are not a comparison. Non-overlap demands a difference
    # 41% larger than significance does, so a leaderboard read off the scorecard
    # silently declines to separate arms the design can separate (`compare`).
    head_md = ""
    if not args.no_head_to_head and len(card.arms) > 1:
        from blockgen.eval.bench import compare as cmpr
        from blockgen.eval.bench import distances as Dz
        from blockgen.eval.bench import geometry as geomz
        tables = []
        if len(geom_feats) > 1:
            tables.append(cmpr.rank_table(
                "geom_kid", lambda a, b: geomz.geom_kid(a, b), geom_feats,
                ref.geom_ref, n_rep=args.n_rep, rng=ctx.rng(3),
                arm_term=Dz.kid_arm_term))
        if len(dino_feats) > 1 and full_ref is not None:
            tables.append(cmpr.rank_table(
                "mv_dino_kid", lambda a, b: Dz.kid(a, b), dino_feats,
                full_ref.ref_feats, n_rep=args.n_rep, rng=ctx.rng(4),
                arm_term=Dz.kid_arm_term))
        if tables:
            card.run["head_to_head"] = [t.to_json() for t in tables]
            head_md = ("## head-to-head — paired, Holm-corrected\n\n"
                       + "\n".join(cmpr.render_rank_table(t) for t in tables))

    # --- example builds -----------------------------------------------------
    # Written after everything else is assembled and merged before `card.write`,
    # so the card and the npz beside it agree or neither exists. The row indices
    # are only meaningful against the file this call produces.
    examples_path: Optional[Path] = None
    if picked:
        blob: Dict[str, Any] = {}
        try:
            blob = ex.write_run_examples(run_dir, picked, corpus=args.corpus,
                                         k=args.examples, seed=args.seed,
                                         run_name=args.name)
        except Exception as exc:
            card.warn(f"example builds not written: {type(exc).__name__}: {exc}")
        if blob:
            card.run["examples"] = blob["run"]
            for name, entry in blob.get("arms", {}).items():
                blocks = card.arms.get(name)
                if isinstance(blocks, dict) and isinstance(blocks.get("meta"), dict):
                    blocks["meta"]["examples"] = entry
            examples_path = run_dir / blob["run"]["npz"]
            if not _lab_discoverable(run_dir):
                card.warn(
                    f"example builds written to {examples_path}, but that path is "
                    f"not under outputs/run_*/ — the lab only discovers datasets "
                    f"there, so the leaderboard's example strip will be empty for "
                    f"this run (drop --out, or pass one under outputs/)")

    finished = time.time()
    card.run["elapsed_s"] = round(finished - t0, 2)
    card.run["finished_at"] = sc.iso_utc(finished)
    json_path = card.write(run_dir / "scorecard.json")
    md_path = run_dir / "scorecard.md"
    md_path.write_text(sc.render_markdown(card) + "\n" + board_md + "\n" + head_md)

    print()
    print(sc.render_markdown(card))
    print(board_md)
    print(head_md)
    print(f"-> {json_path}")
    print(f"-> {md_path}")
    if examples_path is not None:
        print(f"-> {examples_path}")
    # The one link that turns a directory of JSON into something readable.
    print(f"view: http://127.0.0.1:{LAB_PORT}/leaderboard?run={run_dir.name}")
    if not args.name:
        # Cheap habit, no breaking change: the eleven cards that predate this
        # flag are all called `run_<stamp>_bench` and can only be told apart by
        # opening them.
        print("[bench] no --name: this run will be listed by its arm names. "
              "Next time, --name \"what you were asking\" (it names the "
              "directory too) and --note for one line of context.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
