#!/usr/bin/env python
"""Produce the committed golden scorecards in `tests/fixtures/scorecards/`.

**Why these files exist at all.** `outputs/` is gitignored (`.gitignore:27`;
`git ls-files outputs` is empty), so not one scorecard is in the repository, and
the only assertion that ever checked `load_scorecard`'s key set --
`tests/lab/test_lab_data.py:142` -- is `@pytest.mark.slow` and *skips* on any
machine without a bench run. A required key could therefore be added to the
reader, or dropped from the writer, and CI would stay green. These fixtures are
the cheap half of a reference card (D3/D33): four trimmed real cards, one per
shape that exists on disk, plus one synthesized `bench/2` run directory with a
real examples cache beside it.

**Why a generator and not a documented snippet.** A contributor with no corpus,
no GPU and no checkpoint cannot re-run the bench, so a fixture they cannot
regenerate is a fixture they cannot update. `--synthesize` needs none of those
things; `--trim` needs only a run directory the maintainer already has.

Two modes, plus a checker:

* `--trim <run_dir> --out <fixture.json>` — a real card, cut to three arms.
* `--synthesize <out_dir>` — a whole `bench/2` run directory, built by calling
  the real writers (`sc.Scorecard`, `sc.ArmSpec`, `save_house_cache`,
  `examples.write_run_examples`, `composite.leaderboard`, `compare.rank_table`)
  over hand-built metrics and 4x4x4 huts. Nothing here loads a corpus, touches a
  GPU, or imports a renderer.
* `--verify [dir]` — every committed fixture parses, and the synthesized one
  still satisfies `scorecard.RUN_KEYS` and survives `tools.lab.cards.migrate`.

**Not collected by pytest.** The filename is deliberately not `test_*.py` and
there is no top-level `test_` function in it, because this is a producer that
writes into the source tree; running it as part of the suite would make the
fixtures a moving target for the tests that pin them.

**The synthesized card is deterministic on purpose.** `git_sha`, `host`, the
timestamps and the manifest mtimes are pinned to constants rather than measured,
so regenerating it produces a byte-identical file and a review diff shows the
schema change instead of this week's commit hash. `--trim` is deterministic for
the same reason -- it only ever copies.

The one thing this file may not do is invent schema. Every key in the
synthesized card comes from a real writer or from `scorecard.RUN_KEYS`, and the
run block is asserted against that tuple before it is written, so the fixture
cannot quietly drift from what `runner.main` emits.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:          # runnable as `python tests/fixtures/...`
    sys.path.insert(0, str(REPO))

from blockgen.curation.houses import save_house_cache          # noqa: E402
from blockgen.eval.bench import compare as cmpr                # noqa: E402
from blockgen.eval.bench import composite                      # noqa: E402
from blockgen.eval.bench import examples as ex                 # noqa: E402
from blockgen.eval.bench import fast as fast_tier              # noqa: E402
from blockgen.eval.bench import scorecard as sc                # noqa: E402
from blockgen.utils.data import Structure                      # noqa: E402

#: Where the committed fixtures live. Both modes default here.
FIXTURES = Path(__file__).resolve().parent / "scorecards"

#: The synthesized run's directory name. The stamp is a fiction (2026-01-02,
#: before every real card) so the fixture sorts to the *bottom* of a picker that
#: is ordered by directory stamp, and so `cards.dir_stamp` has something to
#: parse that is not today.
FIXTURE_RUN = "run_20260102_000000_bench_fixture"

#: Every timestamp in the synthesized card, and the mtime forced onto its npz
#: manifest so `provenance.written_at` is stable across regenerations.
FIXTURE_EPOCH = 1767312000.0           # 2026-01-02T00:00:00Z


# --------------------------------------------------------------------------
# mode 1: trim a real card
# --------------------------------------------------------------------------
#: Kept whole. `run` and `context` are the point of the legacy fixtures --
#: they are what the reader has to survive -- and `ladder` is 350 bytes.
#: `arms` is the only block big enough to be worth cutting.
KEEP_TOP = ("schema_version", "run", "context", "ladder", "warnings")


def _kind(meta: Dict[str, Any]) -> str:
    """`submission` | `control` | `baseline`, by the lab's own rule.

    Imported rather than re-derived: `tools.lab.cards.arm_kind` is the single
    definition (D11), and a fixture picked by a second copy of the rule would
    stop representing what the reader sees the moment the two disagreed.
    """
    from tools.lab import cards
    return cards.arm_kind(meta)


def choose_arms(arms: Dict[str, Any]) -> List[str]:
    """The three arms a trimmed fixture keeps, in card order.

    `real_test` first because it is the calibration floor every reader special-
    cases; then one *other* reference row, preferring a baseline over a control
    so a card that has both keeps the harder shape (a baseline must stay visible
    when controls are hidden -- D36); then one submission when the card has one.

    Three is not arbitrary: it is the smallest set that can carry one of each
    `kind`, which is what `n_submissions`/`n_controls`/`n_baselines` are counted
    from, and it keeps a 377 KB card under 40 KB.
    """
    kept: List[str] = []
    if "real_test" in arms:
        kept.append("real_test")

    by_kind: Dict[str, List[str]] = {}
    for name, arm in arms.items():
        if name in kept:
            continue
        meta = (arm or {}).get("meta") or {}
        by_kind.setdefault(_kind(meta), []).append(name)

    for kind in ("baseline", "control"):          # a baseline is the better keep
        if by_kind.get(kind):
            kept.append(by_kind[kind][0])
            break
    if by_kind.get("submission"):
        kept.append(by_kind["submission"][0])

    # Card order, not selection order, so the fixture reads like its source and
    # `sections()`'s first-seen order is the real one.
    order = list(arms)
    return sorted(set(kept), key=order.index)


def _trim_head_to_head(tables: Any, keep: List[str]) -> Optional[List[dict]]:
    """The first rank table only, restricted to the kept arms.

    One table, because the second is a copy of the same shape on another metric
    and the fixture pins shape. Restricted rather than copied whole because a
    table that ranks thirteen arms the card no longer contains is an internally
    inconsistent card -- and readers that join `head_to_head` against `arms`
    would be exercised against a state a real run cannot produce.
    """
    if not isinstance(tables, list) or not tables:
        return None
    t = dict(tables[0])
    ks = set(keep)
    if isinstance(t.get("order"), list):
        t["order"] = [a for a in t["order"] if a in ks]
    for field in ("scores", "groups"):
        if isinstance(t.get(field), dict):
            t[field] = {k: v for k, v in t[field].items() if k in ks}
    if isinstance(t.get("comparisons"), list):
        t["comparisons"] = [c for c in t["comparisons"]
                            if c.get("a") in ks and c.get("b") in ks]
    return [t]


def trim(run_dir: Path, out: Path) -> Path:
    """Copy one real card, keeping three arms. Never edits the source."""
    src = run_dir / "scorecard.json" if run_dir.is_dir() else run_dir
    blob = json.loads(src.read_text())
    arms = blob.get("arms") or {}
    keep = choose_arms(arms)

    out_blob: Dict[str, Any] = {}
    for key in blob:                       # source order, so the diff reads right
        if key == "arms":
            out_blob["arms"] = {k: arms[k] for k in keep}
        elif key in KEEP_TOP:
            out_blob[key] = blob[key]

    run = dict(out_blob.get("run") or {})
    if isinstance(run.get("blockscore"), list):
        run["blockscore"] = [r for r in run["blockscore"] if r.get("arm") in set(keep)]
    if "head_to_head" in run:
        trimmed = _trim_head_to_head(run["head_to_head"], keep)
        if trimmed is None:
            run.pop("head_to_head")
        else:
            run["head_to_head"] = trimmed
    out_blob["run"] = run

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_blob, indent=1) + "\n")
    kinds = ", ".join(f"{n} ({_kind((arms[n].get('meta') or {}))})" for n in keep)
    print(f"[trim] {src}\n"
          f"       -> {out}  {out.stat().st_size / 1024:.1f} KB "
          f"({out.stat().st_size} bytes)\n"
          f"       arms: {kinds}")
    return out


# --------------------------------------------------------------------------
# mode 2: synthesize a bench/2 run
# --------------------------------------------------------------------------
def _hut(tag: int) -> Structure:
    """A 4x4x4 hut: a floor, a wall, and one contrasting block.

    The same shape `tests/lab/test_lab_data.py:160` uses, with `tag` varying the
    contrasting block so the builds are distinguishable -- `structures_sha` is a
    hash over `block_ids`, and an arm of identical builds would give every arm
    the same identity and make the examples strip a wall of one thumbnail.
    """
    ids = np.zeros((4, 4, 4), np.int32)
    ids[:, 0, :] = 4                                   # floor
    ids[0, :, :] = 5                                   # one wall
    ids[1 + tag % 3, 1 + tag % 2, 2] = 98 + tag        # the contrasting block
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


#: One row per arm of the synthesized card: name, track, damage `d`, and how
#: many builds it has. `d` is the arm's distance from real *in real-sample
#: spreads*, and it is the whole trick that makes this card's numbers legible
#: without a corpus: every metric below is written as `base + d` with a 95%
#: interval of exactly +-1.96, so the control arm's spread is 1.0 by
#: construction and `composite.leaderboard` -- the real one, run over these
#: metrics -- returns a BlockScore of exactly `d` for every arm.
#:
#: Two of the names are ones `composite.validate` knows, so the card carries a
#: real `blockscore_validation` rather than an empty dict: `real_test` gives C1
#: (the floor is near zero) and `real@canon8` gives C3 (known damage must score
#: worse than real). The other two exist so all three `kind`s and both `origin`s
#: appear on one card -- which is what the leaderboard's Kind column, its
#: controls-hidden filter (a baseline must stay visible) and the
#: `n_submissions`/`n_controls`/`n_baselines` counts are read against.
#:
#: FOUR arms and `k = 2`, which is exactly `SMOKE_N` (catalog.py:301) example
#: rows. Deliberate: `_flag_smoke` labels any arm dataset of <= 8 builds a
#: "smoke run", and the examples cache is exempted by its note (D23). At nine
#: rows or more the fixture could not tell a working exemption from a threshold
#: it never reached. A fifth arm is not worth losing that.
ARMS: Tuple[Tuple[str, str, float, int], ...] = (
    ("real_test", "control", 0.0, 3),
    ("real@canon8", "control", 3.0, 3),
    ("gabled_house", "baseline", 6.0, 3),
    ("fixture_ar", "ar", 2.0, 2),
)

#: The npz-backed arm's manifest `report`, in `scripts/sample_to_npz.py`'s shape
#: -- the one `PROMOTED` was verified against. The checkpoint path is what makes
#: `meta.source_run_id` non-null, which is the edge from a leaderboard row back
#: to the training run.
AR_REPORT = {
    "model": "fixture_ar",
    "checkpoint": "outputs/run_20260101_000000_fixture_train/model.pt",
    "n": 2, "n_empty": 0, "seed": 7, "temperature": 0.9, "top_k": 40,
    # Deliberately outside PROMOTED: the fixture pins that the allowlist is
    # closed, so this key must appear in `provenance.report` and nowhere else.
    "note": "synthetic fixture arm; no model was run",
}


def _ci(v: float) -> Tuple[float, float]:
    """A 95% interval whose implied sd is exactly 1.0. See `ARMS`.

    `Calibration` recovers the sd as half the interval width over 1.96, so the
    half-width has to be 1.96 -- not 0.98. Getting that backwards silently
    doubles every BlockScore, which is exactly the class of arithmetic slip a
    fixture with a stated invariant is supposed to catch.
    """
    return (v - 1.96, v + 1.96)


def _blocks(d: float, *, cost: bool) -> Dict[str, Dict[str, Any]]:
    """One arm's metric families, all built through the real constructors."""
    realism = {
        "geom_kid": sc.with_mmd(sc.metric(
            1.0 + d, _ci(1.0 + d), "lower_better", x=1000,
            note="synthetic fixture value; see make_scorecard_fixtures.ARMS")),
        "mv_dino_kid": sc.metric(0.2 + d, _ci(0.2 + d), "lower_better", x=1000),
    }
    dataset_stats = {
        "palette_jsd_exact": sc.metric(0.05 + d, _ci(0.05 + d), "lower_better",
                                       level="exact", n_symbols=3),
        "palette_jsd_family": sc.metric(0.02 + d, _ci(0.02 + d), "lower_better",
                                        level="family", n_symbols=2),
    }
    # Held flat across arms on purpose: these three feed `composite.GATES`, and
    # a fixture whose arms are disqualified would pin a DQ instead of a score.
    novelty = {
        "voxel_nn_iou_mean": sc.metric(0.35, _ci(0.35), "lower_better", grid=24,
                                       note="IoU to nearest TRAIN build; high = memorized"),
        "voxel_dup_rate": sc.metric(0.0, (0.0, 0.0), "lower_better", threshold=0.95),
        "voxel_diversity": sc.metric(0.9, _ci(0.9), "higher_better"),
        "dino_nn_percentile": sc.metric(0.5, _ci(0.5), "higher_better"),
    }
    # Not Metrics: `coherence` and `geometry_scalars` are gen-vs-real dicts, and
    # `composite` reads their `w1_norm` and RMSes it into one pillar. Same shape
    # as `topology.coherence_report` writes.
    def _pair(base: float) -> Dict[str, Any]:
        return {"gen": {"mean": base + d, "ci": [base + d - 0.1, base + d + 0.1]},
                "real": {"mean": base, "ci": [base - 0.1, base + 0.1]},
                "w1": 0.2 + d, "w1_norm": 0.2 + d, "real_spread": 1.0,
                "direction": "distance_to_real"}
    blocks: Dict[str, Dict[str, Any]] = {
        "realism": realism,
        "coherence": {"lcc_ratio": _pair(0.98), "n_components": _pair(2.7)},
        "geometry_scalars": {"thickness_mean": _pair(1.05)},
        "dataset_stats": dataset_stats,
        "novelty": novelty,
        # Every leaf null with a reason, exactly as the real full-tier cards
        # carry it: the ladder failed these gates, so the page must print the
        # reason and never a blank that reads as zero.
        "fidelity": {k: sc.skipped(f"{k} failed G8_n_stability", "higher_better")
                     for k in ("density", "coverage", "precision", "recall")},
    }
    if cost:
        blocks["cost"] = {
            "cost_usd": sc.metric(0.0049, (float("nan"),) * 2, "lower_better"),
            "elapsed_s": sc.metric(18.16, (float("nan"),) * 2, "lower_better"),
        }
    return blocks


def _context() -> sc.BenchContext:
    """A FULL-tier context, so `view`/`backbone`/`mmd_sigma` are exercised.

    Full tier and not fast, because `fidelity` only exists on a full-tier card
    and `cards.missing()` names it: a fast-tier fixture could never have an
    empty `missing` list, and the empty list is what proves the reader's
    presence checks all pass on a complete card.
    """
    ctx = sc.BenchContext(corpus="houses_32", seed=0, grid=24, min_n=2,
                          n_boot=1000, n_ref=512, alpha=0.05,
                          split_key="houses_32.s0.70-15-15.v1",
                          split_sha="dc0cca4bb8671d39",
                          sizes={"train": 6, "val": 3, "test": 3})
    ctx.image_backbone = "dinov2b"
    ctx.px = 224
    ctx.view = {"key": "o4x224", "px": 224,
                "views": [[45, 30], [135, 30], [225, 30], [315, 30]]}
    ctx.mmd_sigma = 12.44
    # Measured spread wins over the bootstrap interval (Calibration.from_arm);
    # 1.0 keeps the `d`-is-the-score identity true for the geometry pillar too.
    ctx.noise_floor_sd = {"realism.geom_kid": 1.0}
    ctx.derived = {"n_train": 6, "n_val": 3, "n_test": 3, "n_ref_used": 3,
                   "vocab_size": 8, "palette_keys_exact": 3,
                   "palette_keys_family": 2}
    return ctx


def _npz_arm(run_dir: Path, structures: List[Structure]) -> str:
    """Write the two-build cache the `origin="npz"` arm points at.

    Returns the path as recorded on the card: repo-relative when the fixture is
    under the repo, so the committed JSON names a file that exists in every
    checkout instead of a path on this machine. The manifest's mtime is forced
    to `FIXTURE_EPOCH` because `provenance.written_at` reads it, and a fixture
    that changed every time it was regenerated would be unreviewable.
    """
    path = Path(save_house_cache(structures, max_dim=4, cache_dir=str(run_dir),
                                 report=AR_REPORT, name="fixture_ar"))
    manifest = Path(str(path).replace(".npz", "_manifest.json"))
    for p in (path, manifest):
        os.utime(p, (FIXTURE_EPOCH, FIXTURE_EPOCH))
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _head_to_head(rng: np.random.Generator, names: List[str]) -> List[dict]:
    """A real `compare.rank_table` over synthetic features.

    The table is the writer's own output, not a hand-typed dict: `RankTable`
    derives `order`, `beats` and the letter groups from the comparisons, and a
    fixture that hard-coded them would pin a shape the code cannot produce.
    Small `n_rep` because this is four arms of eight fake rows, not a benchmark.
    """
    ref = rng.normal(size=(8, 3))
    feats = {name: ref + i * 0.5 + rng.normal(scale=0.05, size=(8, 3))
             for i, name in enumerate(names)}

    def dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(np.asarray(a).mean(0) - np.asarray(b).mean(0)).sum())

    table = cmpr.rank_table("geom_kid", dist, feats, ref, direction="lower_better",
                            n_rep=20, rng=rng)
    return [table.to_json()]


def synthesize(out_dir: Path) -> Path:
    """Write a whole `bench/2` run directory. No corpus, no GPU, no renderer."""
    run_dir = out_dir / FIXTURE_RUN
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in run_dir.iterdir():           # regeneration must not leave orphans
        if stale.is_file():
            stale.unlink()

    from tools.lab import cards        # the merged CONTROL_RECIPES/RECIPES view

    ctx = _context()
    name = "fixture run"

    built: "OrderedDict[str, sc.ArmSpec]" = OrderedDict()
    for i, (arm_name, track, _d, n) in enumerate(ARMS):
        structs = [_hut(10 * i + j) for j in range(n)]
        if track == "ar":
            built[arm_name] = sc.ArmSpec(arm_name, track=track,
                                         npz=_npz_arm(run_dir, structs))
        else:
            # The in-memory half: an arm with nothing on disk is otherwise
            # indistinguishable from a failed dataset join, so it carries the
            # recipe its builder would have written (fast.CONTROL_RECIPES /
            # baselines.RECIPES, via the lab's merged view of both).
            built[arm_name] = sc.ArmSpec(
                arm_name, track=track, structures=structs,
                provenance_override={
                    "writer": "in_memory",
                    "builder": ("blockgen.eval.bench.fast.control_arms"
                                if track == "control" else
                                "blockgen.eval.bench.baselines.build_arms"),
                    "recipe": cards.RECIPES().get(arm_name, ""),
                    "drawn_from": f"split:{ctx.corpus}:test",
                    "n_requested": n, "seed": ctx.seed,
                })

    card = sc.Scorecard(context=ctx.to_json(),
                        warnings=["synthesized fixture: no arm was generated by "
                                  "a model, and no metric here was measured"],
                        ladder={"path": "outputs/analysis/ladder/fixture.json",
                                "passing": ["kid", "mmd_rbf"],
                                "failing": {"fd": "G8_n_stability"}})

    picked: "OrderedDict[str, Tuple[List[Structure], List[int]]]" = OrderedDict()
    k = 2
    for (arm_name, _track, d, _n), arm in zip(ARMS, built.values()):
        structs = [s.crop_to_non_air() for s in arm.load()]
        provenance = arm.provenance()
        # The same keys, in the same order, `fast.score_fast` writes. Built here
        # rather than by calling `score_fast` because that needs a FastReference,
        # which needs the corpus this generator exists to do without.
        meta: Dict[str, Any] = {
            "track": arm.track, "n": len(structs), "n_empty": 0,
            "source": arm.npz or "in-memory",
            "has_prompts": bool(arm.prompts),
            "palette_oov_frac_exact": 0.0,
            "palette_oov_frac_family": 0.0,
            "family_unclassified_frac": 0.0,
            "kind": arm.kind,
            "origin": arm.origin,
            "source_run_id": fast_tier._source_run_id(arm.npz, provenance),
            "structures_sha": fast_tier._structures_sha(structs),
            "provenance": provenance,
        }
        card.add_arm(arm_name, {"meta": meta,
                                **_blocks(d, cost=arm.kind == "submission")})
        rows = ex.pick(len(structs), k, ctx.rng(9))
        picked[arm_name] = ([structs[i] for i in rows], rows)

    # --- the run block: built FROM RUN_KEYS, exactly as the runner does ------
    argv = ["--tier", "both", "--n", "3", "--min-n", "2", "--name", name]
    run_fields: Dict[str, Any] = {
        "dir": f"outputs/{FIXTURE_RUN}",
        "name": name,
        "note": "synthesized by tests/fixtures/make_scorecard_fixtures.py",
        # Pinned, not measured: see the module docstring on determinism.
        "git_sha": "0000000", "git_branch": "main", "git_dirty": False,
        "started_at": sc.iso_utc(FIXTURE_EPOCH),
        "finished_at": sc.iso_utc(FIXTURE_EPOCH + 12),
        # `cmd` is deliberately the unusable one -- argv[0] absolute, quoting
        # lost -- because that is what the eleven legacy cards carry and what
        # `rerun` exists to replace. `rerun` is computed by the runner's own
        # expression, so the fixture cannot claim a command the parser rejects.
        "cmd": "/abs/path/blockgen/eval/bench/__main__.py " + " ".join(argv),
        "rerun": shlex.join(["python", "-m", "blockgen.eval.bench", *argv]),
        "argv": argv,
        "tier": "both", "host": "fixture", "device": "cpu", "elapsed_s": 12.0,
    }
    # The contract, checked in the producer as well as in the test: a key added
    # to RUN_KEYS without a value here must fail loudly rather than emit a card
    # that claims to satisfy a contract it does not.
    if set(run_fields) != set(sc.RUN_KEYS):
        raise AssertionError(
            f"run block disagrees with scorecard.RUN_KEYS: "
            f"missing={sorted(set(sc.RUN_KEYS) - set(run_fields))}, "
            f"undeclared={sorted(set(run_fields) - set(sc.RUN_KEYS))}")
    card.run = {key: run_fields[key] for key in sc.RUN_KEYS}

    rows = composite.leaderboard(card, min_n=ctx.min_n,
                                 noise_sd=ctx.noise_floor_sd)
    card.run["blockscore"] = [r.to_json() for r in rows]
    card.run["blockscore_validation"] = composite.validate(rows)
    card.run["head_to_head"] = _head_to_head(ctx.rng(3), list(card.arms))

    blob = ex.write_run_examples(run_dir, picked, corpus=ctx.corpus, k=k,
                                 seed=ctx.seed, run_name=name)
    card.run["examples"] = blob["run"]
    for arm_name, entry in blob["arms"].items():
        card.arms[arm_name]["meta"]["examples"] = entry
    for pair in ("examples_4.npz", "examples_4_manifest.json"):
        os.utime(run_dir / pair, (FIXTURE_EPOCH, FIXTURE_EPOCH))

    path = card.write(run_dir / "scorecard.json")
    total = sum(p.stat().st_size for p in run_dir.iterdir())
    print(f"[synthesize] {run_dir}  {total} bytes total")
    for p in sorted(run_dir.iterdir()):
        print(f"             {p.name:34s} {p.stat().st_size:>7d} bytes")
    _check(path)
    return path


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------
def _check(path: Path) -> None:
    """One fixture: it parses, and the lab's read chokepoint survives it.

    `migrate` is where every lab reader enters, so a fixture it cannot normalise
    is a fixture no test can use. The `bench/2` card is additionally held to
    `RUN_KEYS`, which is the point of `--synthesize` existing at all.
    """
    from tools.lab import cards
    blob = json.loads(path.read_text())
    run_id = path.parent.name
    out = cards.migrate(blob, run_id, path.stat().st_mtime)
    if out.get("arms") is not blob.get("arms"):
        raise AssertionError(f"{path}: migrate copied `arms` (D27)")
    if cards.migrate(out, run_id, 0.0) != out:
        raise AssertionError(f"{path}: migrate is not idempotent on this card")

    note = ""
    if blob.get("schema_version") == sc.SCHEMA_VERSION:
        keys = set(blob["run"])
        expected = set(sc.RUN_KEYS) | (keys & set(sc.RUN_KEYS_OPTIONAL))
        if keys != expected:
            raise AssertionError(
                f"{path}: run keys != RUN_KEYS | present optionals; "
                f"missing={sorted(expected - keys)}, extra={sorted(keys - expected)}")
        if out["compat"]["missing"]:
            raise AssertionError(f"{path}: a complete bench/2 card still reports "
                                 f"missing {out['compat']['missing']}")
        note = "  RUN_KEYS ok, compat.missing empty"
    print(f"[check] {path.name:20s} {path.stat().st_size:>7d} bytes  "
          f"{len(blob.get('arms') or {})} arms  "
          f"schema={blob.get('schema_version')}{note}")


def verify(root: Path) -> int:
    """Every committed fixture under `root`. Returns a process exit code."""
    paths = sorted(root.glob("*.json")) + sorted(root.glob("run_*/scorecard.json"))
    if not paths:
        print(f"[check] no fixtures under {root}", file=sys.stderr)
        return 1
    for p in paths:
        _check(p)
    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    print(f"[check] {len(paths)} cards, {total} bytes total under {root}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trim", metavar="RUN_DIR",
                    help="a real run directory (or its scorecard.json) to cut down")
    ap.add_argument("--out", metavar="FIXTURE.JSON",
                    help="where --trim writes (default: tests/fixtures/scorecards/)")
    ap.add_argument("--synthesize", nargs="?", const=str(FIXTURES), metavar="DIR",
                    help="write a whole bench/2 run dir (default: "
                         "tests/fixtures/scorecards/)")
    ap.add_argument("--verify", nargs="?", const=str(FIXTURES), metavar="DIR",
                    help="check the committed fixtures parse and hold their contracts")
    args = ap.parse_args(argv)

    if not (args.trim or args.synthesize or args.verify):
        ap.error("nothing to do: pass --trim, --synthesize or --verify")
    if args.trim:
        src = Path(args.trim)
        out = Path(args.out) if args.out else FIXTURES / f"{src.name}.json"
        _check(trim(src, out))
    if args.synthesize:
        synthesize(Path(args.synthesize))
    if args.verify:
        return verify(Path(args.verify))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
