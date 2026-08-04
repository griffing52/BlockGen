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
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from blockgen.utils.data import Structure

SCHEMA_VERSION = "bench/1"
DIRECTIONS = ("lower_better", "higher_better", "distance_to_real")


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


# --- run inputs ------------------------------------------------------------
@dataclass
class ArmSpec:
    """One thing to score. Real-data controls are ArmSpecs too, deliberately.

    Routing controls through the identical code path is what makes the numbers
    readable -- `real_val` is the floor, `real@canon16` is a known-damage rung --
    and removes any possibility of reference and arms being scored differently.
    """
    name: str
    track: str = "ar"                      # "ar" | "agentic" | "control"
    npz: Optional[str] = None
    structures: Optional[List[Structure]] = None
    prompts: Optional[List[str]] = None
    cost: Optional[Dict[str, float]] = None

    def load(self) -> List[Structure]:
        if self.structures is not None:
            return self.structures
        if not self.npz:
            raise ValueError(f"arm {self.name!r} has neither structures nor npz")
        from blockgen.curation.houses import load_structures_from_cache
        structs, _ = load_structures_from_cache(self.npz)
        return [s.crop_to_non_air() for s in structs]


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

    def rng(self, salt: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.seed + salt)


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


# --- assembly --------------------------------------------------------------
@dataclass
class Scorecard:
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
