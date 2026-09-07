"""The scorecard: what a benchmark run emits, and the invariants it enforces.

Two rules are enforced by the *type*, not by convention, because both have
already produced wrong readings in this project:

1. **No bare floats.** Every leaf metric is a `Metric` carrying a value, a
   confidence interval, and a direction. There is no way to emit `lcc_ratio:
   0.42` with nothing beside it, so the "higher is better" misreading -- wrong
   for coherence, where only ~66% of *real* houses are single-component -- is
   unrepresentable rather than merely discouraged.

2. **No silent nulls.** A metric that failed the validation ladder, or was
   skipped for too few samples, carries `gate_failed` explaining why. The
   markdown renderer prints the reason; it never prints a blank that reads as
   zero.

The renderer additionally refuses to place two sample-size-dependent values
(Frechet distance, and the legacy `perceptual.cmmd`) side by side when their `n`
differs, because those estimators are only comparable at fixed n.

**The schema version is advisory. Nothing branches on it.** `SCHEMA_VERSION` is
stamped into every card and read back for exactly one purpose: the lab copies it
into `compat.version` and shows an amber "written by a newer BlockGen" banner
when its major is greater than the reader understands, while still rendering
whatever parses. Every other decision a reader makes is on the *presence* of a
section or a key.

That is not fastidiousness. `bench/1` is already three incompatible shapes on
disk -- the two oldest cards carry `fidelity` and no `geometry_scalars`, the
2026-08-31 batch carries `geometry_scalars` and `blockscore`, the newest carry
`cost` and no `fidelity` -- so a migration keyed on the version *string* would be
wrong about eight of the eleven cards before it ran. `bench/2` is therefore
`bench/1` plus keys: nothing is removed, renamed, or re-typed, and no writer here
may break that rule. `scripts/bench_report.py` reads `card["arms"]` and
`entry["gen"]["mean"]` bare and is the one consumer that hard-crashes rather than
degrading, so "additive only" is enforced by what would break, not by taste.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from blockgen.utils.data import Structure

SCHEMA_VERSION = "bench/2"
DIRECTIONS = ("lower_better", "higher_better", "distance_to_real")

#: The `run` block a bench run writes, declared HERE rather than in the runner so
#: one constant pins the writer and the golden fixtures together. The synthesized
#: fixture in `tests/fixtures/scorecards/` is not a real run's output, so without
#: a shared constant it would drift from what the runner actually emits and the
#: schema test would pass while being wrong.
#:
#: Every key listed is ALWAYS written by a `bench/2` run; a missing one is a bug,
#: not an old card. The eleven `bench/1` cards on disk carry only
#: {dir, git_sha, cmd, tier, elapsed_s}, and readers FILL the rest rather than
#: demanding it -- see the schema-is-advisory paragraph at the top of this module.
RUN_KEYS = ("dir", "name", "note", "git_sha", "git_branch", "git_dirty",
            "started_at", "finished_at", "cmd", "rerun", "argv",
            "tier", "host", "device", "elapsed_s")

#: Conditionally present, by construction rather than by version: `blockscore`
#: and `blockscore_validation` only when controls ran and `real_test` scored,
#: `head_to_head` only when more than one arm was scored and the flag was not
#: passed, `examples` only when at least one arm's example builds were written.
#:
#: `protocol` is the odd one: the runner now always writes it, so it is optional
#: only in the backward-looking sense that the first `bench/2` cards predate it.
#: It stays here rather than moving to `RUN_KEYS` because the lab re-derives the
#: verdict from `context` and `run` anyway -- the recorded block is a record of
#: what the code believed at scoring time, never the authority.
RUN_KEYS_OPTIONAL = ("blockscore", "blockscore_validation", "head_to_head",
                     "examples", "protocol")

#: Keys lifted out of a manifest's free-form `report` and promoted to the top of
#: an arm's provenance block. The allowlist is closed on purpose and was verified
#: against all four report shapes that exist on disk: `scripts/sample_to_npz.py`
#: writes model/checkpoint/n/n_empty/seed/temperature/top_k,
#: `scripts/dump_samples.py` writes source/seed/cropped, `blockgen/agentic/report.py`
#: writes arm/n_requested/n_nonempty, and the pick-and-place manifest writes
#: model/epochs. Those three writers share no key but `report` itself, which is
#: exactly why `report` is ALSO kept verbatim: the promotion is a convenience for
#: whoever renders the page, never the record.
PROMOTED = ("model", "checkpoint", "seed", "temperature", "top_k", "arm",
            "epochs", "source")


def _clean(x: Any) -> Any:
    """JSON-safe: numpy scalars out, non-finite floats to None."""
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    return x


def iso_utc(ts: float) -> str:
    """A POSIX timestamp as ISO-8601 UTC, seconds, trailing `Z`.

    One timestamp format on a card, so `started_at`, `finished_at` and a
    manifest's `written_at` sort as plain strings and read the same everywhere.
    """
    return (datetime.fromtimestamp(ts, timezone.utc)
            .isoformat(timespec="seconds").replace("+00:00", "Z"))


@dataclass
class Metric:
    """One number, its interval, and which way is good."""
    value: Optional[float]
    ci: Tuple[float, float] = (float("nan"), float("nan"))
    direction: str = "lower_better"
    gate_failed: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, "
                             f"got {self.direction!r}")
        if self.value is None and not self.gate_failed:
            raise ValueError("a null metric must explain itself via gate_failed")

    @property
    def comparable_only_at_n(self) -> Optional[int]:
        return self.extra.get("comparable_only_at_n")

    def to_json(self) -> dict:
        out: Dict[str, Any] = {"value": _clean(self.value),
                               "ci": _clean(list(self.ci)),
                               "direction": self.direction}
        if self.gate_failed:
            out["gate_failed"] = self.gate_failed
        out.update(_clean(self.extra))
        return out


def metric(value: Optional[float], ci: Sequence[float] = (float("nan"),) * 2,
           direction: str = "lower_better", **extra: Any) -> Metric:
    """Shorthand constructor. `gate_failed` may be passed through `extra`."""
    return Metric(value=value, ci=(float(ci[0]), float(ci[1])), direction=direction,
                  gate_failed=extra.pop("gate_failed", None), extra=extra)


def skipped(reason: str, direction: str = "lower_better") -> Metric:
    """A metric deliberately not computed. Never rendered as a number."""
    return Metric(value=None, direction=direction, gate_failed=reason)


def from_ci(triple: Tuple[float, float, float], direction: str = "lower_better",
            **extra: Any) -> Metric:
    """Build from a `(point, lo, hi)` triple as returned by `stats`."""
    point, lo, hi = triple
    return metric(float(point), (lo, hi), direction, **extra)


def with_mmd(m: Metric) -> Metric:
    """Attach `mmd = sqrt(max(value, 0))` to a squared-MMD metric.

    KID and its geometric twin are squared MMDs, and a squared MMD is
    **quadratic in the fraction of an arm's output that is bad**: for a mixture
    of a fraction `f` of failures with otherwise-real samples,
    `MMD^2 = f^2 * MMD^2(bad, real)`. Measured on this corpus, an arm whose
    output is 25% canon-8 scores only 31% of the way from real to fully-canon-8
    on `mv_dino_kid` -- so the raw number substantially under-penalises partial
    failure, which is exactly how real generators fail.

    Taking the square root undoes it. On the same mixture ladder,
    `sqrt(KID)/sqrt(KID_worst)` tracks the true contaminated fraction to a mean
    absolute error of 0.022, against 0.119 for the raw value -- it reads the
    failure rate almost directly.

    Reported as a companion readout, not as a ranking metric: it is a monotone
    transform, so it inherits the ordering gates but not the noise-floor units
    the ladder states its other criteria in. Read it when the question is "what
    fraction of this arm's builds are bad"; read the metric itself when the
    question is "is this arm's distribution the right one".
    """
    if m.value is None:
        return m
    m.extra["mmd"] = float(np.sqrt(max(m.value, 0.0)))
    return m


# --- run inputs ------------------------------------------------------------
@dataclass
class ArmSpec:
    """One thing to score. Real-data controls are ArmSpecs too, deliberately.

    Routing controls through the identical code path is what makes the numbers
    readable -- `real_val` is the floor, `real@canon16` is a known-damage rung --
    and removes any possibility of reference and arms being scored differently.
    """
    name: str
    track: str = "ar"                      # "ar" | "agentic" | "control" | "baseline"
    npz: Optional[str] = None
    structures: Optional[List[Structure]] = None
    prompts: Optional[List[str]] = None
    cost: Optional[Dict[str, float]] = None
    # --- appended at the END, all defaulted, and it must stay that way --------
    # `scripts/bench_doseresponse.py:44` and `scripts/human_study_export.py:40`
    # construct ArmSpecs POSITIONALLY, so a field inserted above this line
    # silently shifts their arguments instead of failing.
    #: The manifest `load()` read, kept rather than dropped. See `load`.
    manifest: Optional[Dict[str, Any]] = field(default=None, repr=False)
    #: For arms with nothing on disk: the provenance block to report verbatim.
    #: `fast.control_arms` and the baseline arms fill this with their recipe,
    #: because an in-memory arm is otherwise indistinguishable from a failed
    #: lookup on the page -- both render as "nothing on disk to show".
    provenance_override: Optional[Dict[str, Any]] = None

    @property
    def kind(self) -> str:
        """What this arm is FOR: `submission` | `control` | `baseline`.

        The single definition of the rule. It is written into `meta.kind` at
        score time so that a card read on its own -- by `bench_report`, by
        `human_study_analyze`, by the leaderboard's row filter -- groups arms by
        reading one field, instead of re-deriving `track not in {"control",
        "baseline"}` in three places across two languages, which is how the
        three places came to disagree about baselines.
        """
        return self.track if self.track in ("control", "baseline") else "submission"

    @property
    def origin(self) -> str:
        """Whether anything was written to disk: `npz` | `in_memory`.

        Not cosmetic. The lab renders "generated in-process -- nothing on disk
        to show" both for an arm that genuinely has no file and for an arm whose
        dataset join failed, and those are different bugs with different fixes.
        This field is what lets the page tell them apart.
        """
        return "npz" if self.npz else "in_memory"

    def load(self) -> List[Structure]:
        if self.structures is not None:
            return self.structures
        if not self.npz:
            raise ValueError(f"arm {self.name!r} has neither structures nor npz")
        from blockgen.curation.houses import load_structures_from_cache
        # The manifest is KEPT, not dropped into `_` as it was until bench/2.
        # It is the only record of which checkpoint, seed and temperature
        # produced this arm; it was already in memory, in scope, on this line;
        # and every provenance block on every card was lost right here.
        #
        # `structures` is deliberately NOT memoized on the way past. Both tiers
        # call `load()` independently, and holding 16 arms x 128 cropped builds
        # is ~260 MB resident to save ~0.3 s on a machine that is at the same
        # time holding a run's worth of DINO features.
        structs, manifest = load_structures_from_cache(self.npz)
        self.manifest = manifest
        return [s.crop_to_non_air() for s in structs]

    def provenance(self) -> Dict[str, Any]:
        """Where this arm came from. Never raises; never returns None.

        Self-sufficient on purpose: it reads the `_manifest.json` sibling itself
        when `load()` has not run. The first version of this method depended on
        `score_fast` calling `load()` before it built the meta block -- an
        invariant nothing in the type states, and one that `score_full` and the
        examples writer each break by calling `load()` on their own.

        A missing or unparseable manifest is a fact about the arm, not a reason
        to lose a two-hour scoring run, so failure degrades to a block that still
        names the two paths it looked at and carries an empty `report`.
        """
        if self.provenance_override is not None:
            # Copied, so a caller stashing this in a scorecard cannot reach back
            # into the ArmSpec that produced it.
            return dict(self.provenance_override)
        if not self.npz:
            # An in-memory arm whose builder recorded nothing. Say exactly that
            # rather than inventing a recipe for it.
            return {"writer": "in_memory", "builder": "?"}

        man_path = str(self.npz).replace(".npz", "_manifest.json")
        out: Dict[str, Any] = {"writer": "npz", "npz": str(self.npz),
                               "manifest": man_path, "written_at": None,
                               "count": None, "max_dim": None, "report": {}}
        try:
            if self.manifest is None:
                self.manifest = json.loads(Path(man_path).read_text())
            man = self.manifest or {}
            out["count"] = man.get("count")
            out["max_dim"] = man.get("max_dim")
            report = man.get("report") or {}
            out["report"] = report
            for key in PROMOTED:
                if key in report:
                    out[key] = report[key]
            try:
                out["written_at"] = iso_utc(Path(man_path).stat().st_mtime)
            except OSError:
                pass                          # the npz may outlive its manifest
        except Exception:
            pass
        return out


@dataclass
class BenchContext:
    """Everything an arm is scored *against*. Identical across arms, by contract."""
    corpus: str = "houses_32"
    seed: int = 0
    grid: int = 24
    min_n: int = 16
    n_boot: int = 1000
    n_ref: int = 512
    alpha: float = 0.05
    palette_level: str = "exact"
    dup_threshold: float = 0.95
    # Populated by the runner.
    split_key: str = ""
    split_sha: str = ""
    sizes: Dict[str, int] = field(default_factory=dict)
    # --- the FULL tier's half, declared instead of monkey-patched -------------
    # `text_backbone` used to be set on this object from outside and read back
    # through a `getattr` default in `full.py`, which is a field with none of a
    # field's guarantees: nothing listed it, nothing defaulted it, and nothing
    # could serialize it.
    text_backbone: str = "clipL"
    image_backbone: str = ""
    px: Optional[int] = None
    view: Optional[Dict[str, Any]] = None
    mmd_sigma: Optional[float] = None
    noise_floor_sd: Dict[str, float] = field(default_factory=dict)
    #: Values the runner can only measure once the reference is built -- split
    #: sizes, vocab size, palette key counts. They belong in `context` and they
    #: are not knowable at construction, so they arrive here rather than
    #: justifying a second parallel dict.
    derived: Dict[str, Any] = field(default_factory=dict)

    def rng(self, salt: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.seed + salt)

    def to_json(self) -> Dict[str, Any]:
        """The `context` block, from the object the arms were actually scored against.

        Until `bench/2` the runner hand-built a parallel dict beside this object,
        and the two disagreed: `palette_level`, `dup_threshold`, `n_ref` and
        `sizes` were used to score every arm on every card and reported nowhere.
        A scorecard whose context is not the context is worse than one with no
        context at all, so there is now one source of the block and this is it.

        The exact legacy key set, of which this is a strict SUPERSET:
        `corpus, seed, grid, min_n, split_key, split_sha, n_train, n_val, n_test,
        n_ref_used, vocab_size, palette_keys_exact, palette_keys_family,
        bootstrap` -- the first six from declared fields, the rest from
        `derived`. Dropping any of them breaks `scripts/bench_report.py`, which
        reads `context.corpus`, `context.split_key` and `context.n_ref_used`.

        `view`, `backbone` and `mmd_sigma` are emitted ONLY once set, because
        they are FULL-tier facts. A fast-tier card that reported
        `backbone.text = "clipL"` would be claiming a backbone it never loaded,
        which is why `text_backbone` is never emitted on its own: it appears
        inside `backbone`, beside the image backbone that was actually used, or
        it does not appear at all.
        """
        out: Dict[str, Any] = {
            "corpus": self.corpus, "seed": self.seed, "grid": self.grid,
            "min_n": self.min_n, "n_ref": self.n_ref,
            "palette_level": self.palette_level,
            "dup_threshold": self.dup_threshold,
            "split_key": self.split_key, "split_sha": self.split_sha,
            "sizes": dict(self.sizes),
            "bootstrap": {"n_boot": self.n_boot, "alpha": self.alpha,
                          "unit": "structure"},
        }
        out.update(self.derived)
        if self.view is not None:
            out["view"] = self.view
        if self.image_backbone:
            out["backbone"] = {"image": self.image_backbone,
                               "text": self.text_backbone, "pool": "mean"}
        if self.mmd_sigma is not None:
            out["mmd_sigma"] = float(self.mmd_sigma)
        if self.noise_floor_sd:
            # Measured whenever the controls ran, fast tier included, so it is
            # keyed off content rather than off the tier.
            out["noise_floor_sd"] = dict(self.noise_floor_sd)
        return _clean(out)


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def git_info() -> Dict[str, Any]:
    """`{sha, branch, dirty}`, each measured independently and best-effort.

    Three subprocesses rather than one, because the failure modes are different:
    a detached HEAD has no branch but still has a sha, and a machine with no
    `git` on PATH must yield a dict of the same shape rather than an exception
    in the middle of writing a card.

    `dirty` is the one that earns its keep. This working tree has carried 40+
    modified files for most of a week, so a `git_sha` recorded without it names
    a commit that does not describe the code that ran -- and the leaderboard now
    says so on the chip instead of quietly implying reproducibility.

    (`dirty` is a bool, hence the `Any`; the other two are `str`/`None`.)
    """
    info: Dict[str, Any] = {"sha": git_sha(), "branch": None, "dirty": None}
    try:
        info["branch"] = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True,
            text=True, timeout=5, check=True).stdout.strip() or None
    except Exception:
        pass
    try:
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            timeout=5, check=True).stdout
        info["dirty"] = bool(porcelain.strip())
    except Exception:
        pass
    return info


# --- assembly --------------------------------------------------------------
@dataclass
class Scorecard:
    """One run's output: the context, the arms, and the run block.

    `run` is the run's identity and is contracted by `RUN_KEYS` -- the runner
    builds it from that tuple and a test pins the two together, so the key set
    cannot drift between the writer and the fixtures that claim to represent it.
    `RUN_KEYS_OPTIONAL` names the keys whose absence is meaningful rather than a
    defect (no controls ran; one arm; no examples requested; a card written
    before protocols existed).
    """
    context: Dict[str, Any]
    arms: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    run: Dict[str, Any] = field(default_factory=dict)
    ladder: Optional[Dict[str, Any]] = None

    def add_arm(self, name: str, blocks: Dict[str, Any]) -> None:
        self.arms[name] = blocks

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[bench] WARNING: {message}", flush=True)

    def to_json(self) -> dict:
        def render(block: Any) -> Any:
            if isinstance(block, Metric):
                return block.to_json()
            if isinstance(block, dict):
                return {k: render(v) for k, v in block.items()}
            return _clean(block)

        out = {"schema_version": SCHEMA_VERSION, "run": _clean(self.run),
               "context": _clean(self.context)}
        if self.ladder is not None:
            out["ladder"] = _clean(self.ladder)
        out["arms"] = {k: render(v) for k, v in self.arms.items()}
        out["warnings"] = list(self.warnings)
        return out

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")
        return path


# --- rendering -------------------------------------------------------------
def _fmt(m: Metric, digits: int = 3) -> str:
    if m.value is None:
        return f"— ({m.gate_failed})"
    lo, hi = m.ci
    if np.isfinite(lo) and np.isfinite(hi):
        return f"{m.value:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
    return f"{m.value:.{digits}f}"


def _fmt_vs_real(entry: Dict[str, Any], digits: int = 3) -> str:
    """Coherence cells always show the real value beside the generated one."""
    g, r = entry["gen"]["mean"], entry["real"]["mean"]
    wn = entry.get("w1_norm")
    tail = f", {wn:.2f} spreads" if wn is not None and np.isfinite(wn) else ""
    return f"{g:.{digits}f} (real {r:.{digits}f}{tail})"


def _n_of(arm: Dict[str, Any]) -> Optional[int]:
    return (arm.get("meta") or {}).get("n")


def render_markdown(card: Scorecard, digits: int = 3) -> str:
    """Human-readable table. Refuses to line up n-dependent metrics across arms."""
    lines: List[str] = []
    ctx = card.context
    lines.append(f"# Benchmark — {ctx.get('corpus')} / {ctx.get('split_key')}")
    lines.append("")
    lines.append(f"- seed `{ctx.get('seed')}` · reference `{ctx.get('n_ref_used', '?')}` "
                 f"· bootstrap unit **structure**")
    if card.run.get("git_sha"):
        lines.append(f"- git `{card.run['git_sha']}`")
    lines.append("")

    arms = list(card.arms)
    if not arms:
        return "\n".join(lines) + "\n(no arms scored)\n"

    sections = [k for k in ("realism", "fidelity", "novelty", "dataset_stats",
                            "faithfulness", "cost")
                if any(k in card.arms[a] for a in arms)]

    for section in sections:
        keys: List[str] = []
        for a in arms:
            for k in card.arms[a].get(section, {}):
                if k not in keys:
                    keys.append(k)
        if not keys:
            continue
        lines.append(f"## {section.replace('_', ' ')}")
        lines.append("")
        lines.append("| arm | n | " + " | ".join(keys) + " |")
        lines.append("|" + "---|" * (len(keys) + 2))
        for a in arms:
            block = card.arms[a].get(section, {})
            cells = []
            for k in keys:
                m = block.get(k)
                if not isinstance(m, Metric):
                    cells.append("—")
                    continue
                # n-dependent estimators must not be read across arms of
                # different size; show the n they are valid at instead.
                pin = m.comparable_only_at_n
                if pin is not None and len({_n_of(card.arms[x]) for x in arms}) > 1:
                    cells.append(f"{_fmt(m, digits)} @n={pin}")
                else:
                    cells.append(_fmt(m, digits))
            lines.append(f"| {a} | {_n_of(card.arms[a]) or '?'} | " + " | ".join(cells) + " |")
        lines.append("")

    if any("coherence" in card.arms[a] for a in arms):
        keys = []
        for a in arms:
            for k in card.arms[a].get("coherence", {}):
                if k not in keys:
                    keys.append(k)
        lines.append("## coherence — distance to real (not 'higher is better')")
        lines.append("")
        lines.append("| arm | " + " | ".join(keys) + " |")
        lines.append("|" + "---|" * (len(keys) + 1))
        for a in arms:
            block = card.arms[a].get("coherence", {})
            cells = [_fmt_vs_real(block[k], digits) if k in block else "—" for k in keys]
            lines.append(f"| {a} | " + " | ".join(cells) + " |")
        lines.append("")

    if card.warnings:
        lines.append("## warnings")
        lines.append("")
        lines.extend(f"- {w}" for w in card.warnings)
        lines.append("")
    return "\n".join(lines)
