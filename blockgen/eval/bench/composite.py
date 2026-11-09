"""BlockScore: one ranked number, built so that cheating loses.

A competition needs a single column to sort on. That is a genuine hazard here,
because this suite has already measured what happens when realism is allowed to
stand alone: in T23c a model that does nothing but recite its training set
scored **second best in the table** on MV-DINO-KID, closer to real than any
corruption. Realism alone has a trivial winning strategy, and any leaderboard
that ranks on it is a leaderboard for that strategy.

So the aggregate is built around two rules.

**Novelty and diversity are gates, not terms.** A term can be bought: an arm may
accept a poor novelty score and pay for it with an excellent realism score, and
the arithmetic will let it. A gate cannot. An arm that copies training data or
emits one build repeatedly is *disqualified*, not merely penalised, and the
scorecard says which gate it hit. Note the gate is a *statistical* test against
the real control, not a fixed cutoff: real builds are themselves somewhat close
to their training neighbours, and a threshold that ignored that would either
never fire or disqualify reality.

**Pillars combine by worst case, not by average.** BlockScore is the *maximum*
distance-from-real across pillars. A mean would let a generator trade a pillar
away -- and the trade the render tier invites is exactly the dangerous one,
since `probes.solidify` shows a build can be structurally hollowed out and stay
within about one noise unit of real in appearance. Under a max, every pillar is
a veto, and "looks right from outside" cannot buy "is built wrong inside".

Units are **noise-floor multiples**: every pillar is `(value - real) / sd(real)`,
calibrated against the `real_test` control arm scored in the same run under the
identical protocol. So a BlockScore of 0 means "indistinguishable from held-out
real on every pillar", 3 means "three real-sample spreads away on its worst
pillar", and the number is comparable across corpora, resolutions and tracks
without any hand-set weights. There are no weights to tune, which is deliberate:
a weighted sum is where a benchmark's authors put their thumb.

The aggregate is validated the same way the metrics are -- `validate()` scores
the known-degenerate control arms and asserts that real wins and every cheat
loses. A composite that has not been shown to reject `train_verbatim`,
`real@monochrome` and `real@solidify` is an untested claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from blockgen.eval.bench import scorecard as sc

#: pillar -> (scorecard block, metric key). Each is a distribution distance
#: whose real-data floor the control arm supplies.
PILLARS: Dict[str, Tuple[str, str]] = {
    "appearance": ("realism", "mv_dino_kid"),
    "geometry": ("realism", "geom_kid"),
    "palette": ("dataset_stats", "palette_jsd_exact"),
}

#: Pillars assembled from a family of already-normalized `w1_norm` entries
#: rather than from one metric. Combined by RMS *within* the pillar: these are
#: several views of one property, so a single odd component should raise the
#: pillar without dominating it the way a max would.
#: `traversal` (2026-10-23) is the player's reading of a build: whether a
#: two-block body can get in, how much of the standing room it reaches, how many
#: storeys, and whether there is headroom. It is its own pillar because it is
#: the measure the render tier cannot see and the geometric descriptor sees only
#: through its blind variant; under the max, a house nobody can enter loses.
STRUCTURE_BLOCKS = ("coherence", "geometry_scalars", "traversal")
#: The pillars of the pinned v1 protocol (`geometry.DESCRIPTOR_VERSIONS`): the
#: five-pillar BlockScore of every card before 2026-10-23.
STRUCTURE_BLOCKS_V1 = ("coherence", "geometry_scalars")


def structure_blocks() -> Tuple[str, ...]:
    """The structure pillars of the active descriptor version."""
    from blockgen.eval.bench import geometry as geom
    return STRUCTURE_BLOCKS if geom.has_traversal() else STRUCTURE_BLOCKS_V1

#: Gates. Each names the metric, the direction that indicates cheating, and the
#: number of real-control spreads beyond which the arm is disqualified.
@dataclass(frozen=True)
class Gate:
    block: str
    key: str
    fails_when: str          # "below" or "above" the real control
    z: float
    reason: str
    min_spread: float = 0.0  # smallest difference worth calling a difference
    #: False = diagnostic probes of real builds (`is_probe_arm`) are not judged
    #: by this gate; see the twin gate for why.
    applies_to_probes: bool = True


#: Real-derived controls that are the gates' own positive controls, so every
#: gate must keep judging them even though they are built from real builds.
GATE_POSITIVE_CONTROLS = frozenset({"real@single_mode"})


def is_probe_arm(name: str) -> bool:
    """A diagnostic perturbation of held-out real builds, not a submission."""
    if name in GATE_POSITIVE_CONTROLS:
        return False
    return name.startswith("real@") or name == "real_shuffled_materials"


GATES: Tuple[Gate, ...] = (
    Gate("novelty", "dino_nn_percentile", "below", 3.0,
         "memorization: closer to training data than real held-out builds are",
         min_spread=0.05),
    Gate("novelty", "voxel_dup_rate", "above", 3.0,
         "memorization: reproduces training builds at IoU >= threshold",
         min_spread=0.02),
    # Deliberately a far higher bar than the memorization gates. Real
    # corruptions move diversity for honest reasons -- `real@solidify` fills
    # interiors, which makes builds genuinely more alike, and at a 3-spread bar
    # it was disqualified as "mode collapse" and its (enormous) geometry pillar
    # was hidden behind a wrong label. Actual collapse is not subtle: the
    # `real@single_mode` control repeats one build and lands far past this.
    Gate("novelty", "voxel_diversity", "below", 10.0,
         "mode collapse: generated builds resemble each other more than real ones do",
         min_spread=0.02),
    # Added 2026-11-04 after the Dream-Cubed audit: a set made of ~160 clusters
    # of near-copies passed both gates above, because one compares to the TRAIN
    # split and the other averages over all pairs. Same rule as the train-split
    # duplicate gate (a rate, 3 spreads, 0.02 floor); real held-out sets sit near
    # 0.01. Cards scored before the metric existed lack the key and are skipped.
    # Probes are exempt (user decision 2026-11-04): filling interiors makes real
    # houses genuinely more alike in shape (`real@solidify` reads 0.066, 2.93
    # spreads), and a diagnostic that rewrites real builds is not a generator
    # that could be copying itself. `real@single_mode` stays judged.
    Gate("novelty", "voxel_twin_rate", "above", 3.0,
         "self-duplication: builds have near-copies (IoU >= 0.8) within the set "
         "far more often than real ones do",
         min_spread=0.02, applies_to_probes=False),
)


def _value(block: Dict[str, object], key: str) -> Optional[float]:
    m = block.get(key) if block else None
    if isinstance(m, sc.Metric):
        return m.value
    return None


def _spread(block: Dict[str, object], key: str) -> float:
    """Real-control spread for one metric, from its bootstrap interval.

    A 95% interval is about +-1.96 sd, so half its width over 1.96 recovers the
    sd on the control arm's own scale -- which is exactly the "one real-sample
    spread" unit every pillar is quoted in.
    """
    m = block.get(key) if block else None
    if not isinstance(m, sc.Metric) or m.value is None:
        return float("nan")
    lo, hi = m.ci
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return float("nan")
    return float(hi - lo) / (2.0 * 1.96)


@dataclass
class Calibration:
    """Real-data floor and spread for every metric, from the control arm.

    Taken from an arm scored in the *same run*, never from a stored constant:
    the floor is a property of the corpus, the split, the sample size and the
    protocol, and a floor imported from another run silently compares an arm to
    conditions it was not measured under.
    """
    floor: Dict[str, float] = field(default_factory=dict)
    spread: Dict[str, float] = field(default_factory=dict)
    source: str = ""
    #: paths whose spread is a measured real-draw noise floor, not a bootstrap CI
    measured: set = field(default_factory=set)

    @classmethod
    def from_arm(cls, arm: Dict[str, Dict[str, object]], name: str,
                 noise_sd: Optional[Dict[str, float]] = None) -> "Calibration":
        """`noise_sd` overrides the spread for metrics that have a measured one.

        A control arm's own bootstrap interval describes how much *that arm's
        score* moves when its 128 builds are resampled. For a set-level kernel
        statistic that is not the same quantity as how much the score moves for
        a *different* real sample of the same size, and it is the smaller of the
        two -- measured on `geom_kid`, by a factor of about 2.6. Quoting the
        smaller one while calling the unit a "real-sample spread" inflates every
        arm's distance from real. Where the run can measure the real thing (a
        ladder-style noise floor over held-out real draws), it is passed in here
        and wins; the bootstrap interval is the fallback for the per-structure
        statistics, where resampling builds *is* the right operation.
        """
        cal = cls(source=name)
        for block_name, block in arm.items():
            if not isinstance(block, dict):
                continue
            for key in block:
                v, s = _value(block, key), _spread(block, key)
                if v is not None:
                    cal.floor[f"{block_name}.{key}"] = float(v)
                    cal.spread[f"{block_name}.{key}"] = s
        # `w1_norm` entries are already quoted in real-spread units, so their
        # floor is whatever the control arm scores against its own reference
        # and their unit is 1 by construction.
        for block_name in structure_blocks():
            block = arm.get(block_name) or {}
            vals = [e.get("w1_norm") for e in block.values() if isinstance(e, dict)]
            rms = _rms(vals)
            if np.isfinite(rms):
                cal.floor[f"{block_name}.__rms__"] = rms
                cal.spread[f"{block_name}.__rms__"] = 1.0
        for path, sd in (noise_sd or {}).items():
            if np.isfinite(sd) and sd > 0:
                cal.spread[path] = float(sd)
                cal.measured.add(path)
        return cal

    def z(self, path: str, value: Optional[float], higher_is_worse: bool = True,
          min_spread: float = 0.0) -> float:
        """Distance from the real floor in real-sample spreads.

        `min_spread` guards the case that silently disabled the memorization
        gate on its first run: a control metric pinned at a boundary -- voxel
        duplicate rate is exactly 0.000 on held-out real -- has a bootstrap
        interval of zero width, which gives no scale, which made `z` NaN, which
        made the gate a no-op. `train_verbatim` sailed through with a duplicate
        rate of 1.000. A gate whose scale cannot be estimated must fall back to
        the smallest difference that would matter, never to "no opinion".
        """
        if value is None or path not in self.floor:
            return float("nan")
        sd = self.spread.get(path, float("nan"))
        if not np.isfinite(sd):
            sd = min_spread
        sd = max(sd, min_spread)
        if sd <= 1e-12:
            return float("nan")
        d = (float(value) - self.floor[path]) / sd
        return d if higher_is_worse else -d


def _rms(values: Sequence[Optional[float]]) -> float:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)],
                   dtype=float)
    return float(np.sqrt((v ** 2).mean())) if v.size else float("nan")


@dataclass
class BlockScore:
    """One arm's aggregate, with everything needed to argue about it."""
    arm: str
    score: Optional[float]
    pillars: Dict[str, float] = field(default_factory=dict)
    worst: Optional[str] = None
    disqualified: Optional[str] = None
    unranked: Optional[str] = None
    missing: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.disqualified:
            return "DQ"
        if self.unranked:
            return "unranked"
        return "ok"

    def to_json(self) -> dict:
        return {"arm": self.arm, "score": self.score, "status": self.status,
                "pillars": self.pillars, "worst_pillar": self.worst,
                "disqualified": self.disqualified, "unranked": self.unranked,
                "missing_pillars": self.missing}


def score_arm(name: str, arm: Dict[str, Dict[str, object]], cal: Calibration,
              min_n: int = 16) -> BlockScore:
    """BlockScore for one scored arm. See module docstring for the two rules."""
    out = BlockScore(arm=name, score=None)

    n = (arm.get("meta") or {}).get("n")
    if isinstance(n, int) and n < min_n:
        out.unranked = f"n={n} < min_n={min_n}"

    # --- gates first: a disqualified arm is never given a number to quote ---
    for g in GATES:
        if not g.applies_to_probes and is_probe_arm(name):
            continue
        block = arm.get(g.block) or {}
        v = _value(block, g.key)
        if v is None:
            continue
        z = cal.z(f"{g.block}.{g.key}", v, higher_is_worse=(g.fails_when == "above"),
                  min_spread=g.min_spread)
        if np.isfinite(z) and z > g.z:
            out.disqualified = f"{g.key}: {g.reason} ({z:+.1f} spreads)"
            return out

    for pillar, (block_name, key) in PILLARS.items():
        block = arm.get(block_name) or {}
        z = cal.z(f"{block_name}.{key}", _value(block, key))
        if np.isfinite(z):
            out.pillars[pillar] = float(z)
        else:
            out.missing.append(pillar)

    for block_name in structure_blocks():
        block = arm.get(block_name) or {}
        rms = _rms([e.get("w1_norm") for e in block.values() if isinstance(e, dict)])
        z = cal.z(f"{block_name}.__rms__", rms)
        if np.isfinite(z):
            out.pillars[block_name] = float(z)
        elif block:
            out.missing.append(block_name)

    if not out.pillars:
        out.unranked = out.unranked or "no pillar could be computed"
        return out

    # Worst case, not average: every pillar is a veto (see module docstring).
    out.worst = max(out.pillars, key=lambda k: out.pillars[k])
    out.score = float(out.pillars[out.worst])
    return out


def leaderboard(card: sc.Scorecard, control: str = "real_test",
                min_n: int = 16,
                noise_sd: Optional[Dict[str, float]] = None) -> List[BlockScore]:
    """Score every arm against the run's own real control, best first.

    Raises if the control arm is absent: without it there is no floor, and a
    BlockScore computed against an imagined floor of zero would rank arms by how
    large their metrics happen to be rather than by how far from real they are.
    """
    if control not in card.arms:
        raise ValueError(
            f"no control arm {control!r} in this run; BlockScore is calibrated "
            "against real held-out data scored under the identical protocol. "
            "Re-run without --no-controls.")
    cal = Calibration.from_arm(card.arms[control], control, noise_sd=noise_sd)
    rows = [score_arm(name, arm, cal, min_n=min_n) for name, arm in card.arms.items()]
    rows.sort(key=lambda r: (r.status != "ok",
                             r.score if r.score is not None else float("inf")))
    return rows


def render_leaderboard(rows: Sequence[BlockScore], digits: int = 2) -> str:
    """Markdown leaderboard: the score, its worst pillar, and the full breakdown."""
    pillars: List[str] = []
    for r in rows:
        for p in r.pillars:
            if p not in pillars:
                pillars.append(p)
    lines = ["| # | arm | BlockScore ↓ | worst pillar | "
             + " | ".join(pillars) + " |",
             "|" + "---|" * (len(pillars) + 4)]
    rank = 0
    for r in rows:
        cells = [f"{r.pillars[p]:+.{digits}f}" if p in r.pillars else "—"
                 for p in pillars]
        if r.status == "ok":
            rank += 1
            head, worst = f"{rank}", r.worst or "—"
            val = f"**{r.score:.{digits}f}**"
        else:
            head = "—"
            val = f"**{r.status.upper()}**"
            worst = r.disqualified or r.unranked or ""
        lines.append(f"| {head} | {r.arm} | {val} | {worst} | " + " | ".join(cells) + " |")
    lines += ["", "Units are **real-sample spreads**: 0 is indistinguishable from "
              "held-out real, higher is worse. BlockScore is the **worst** pillar, "
              "never the mean, so no pillar can be traded away. Novelty and "
              "diversity are disqualification gates rather than terms — see T23c "
              "for why a realism-only ranking is won by copying the training set.", ""]
    return "\n".join(lines)


def validate(rows: Sequence[BlockScore]) -> Dict[str, bool]:
    """Assert the aggregate does what it claims, on the known-degenerate arms.

    The same discipline the metric ladder applies to metrics, applied to the
    aggregate: a composite nobody has tried to cheat is an untested claim. Each
    check names a strategy that must not win.
    """
    by = {r.arm: r for r in rows}

    def score(name: str) -> float:
        r = by.get(name)
        return r.score if (r and r.score is not None) else float("nan")

    checks: Dict[str, bool] = {}
    real = score("real_test")
    pillars = set(by["real_test"].pillars) if "real_test" in by else set()

    #: A corruption is only required to be caught by a tier that can see it.
    #: Material shuffle permutes placement without touching the multiset, so the
    #: palette statistics are blind to it by construction and the geometric
    #: descriptor is blind to it by design. `appearance` resolves it, and so
    #: does `traversal`, which reads doors and stairs by their ids: shuffled
    #: materials put doors in the roof. Asserting otherwise of the blind tiers
    #: would demand that a metric detect information it was built not to read.
    NEEDS = {"real_shuffled_materials": ("appearance", "traversal")}

    if "real_test" in by:
        checks["C1_real_is_near_zero"] = bool(np.isfinite(real) and abs(real) < 3.0)
    if "train_verbatim" in by:
        checks["C2_memorization_disqualified"] = by["train_verbatim"].status == "DQ"
    if "real@single_mode" in by:
        checks["C5_mode_collapse_disqualified"] = by["real@single_mode"].status == "DQ"
    #: Decimation rungs are not in this list because they are no longer
    #: leaderboard controls -- see `fast.control_arms`. A coarser grid is a
    #: change of resolution, not a loss of quality, so "worse than real" is not
    #: a property the suite should assert about it.
    for arm in ("real@monochrome", "real_shuffled_materials", "real@solidify"):
        if arm not in by:
            continue
        need = NEEDS.get(arm)
        if need and not any(p in pillars for p in need):
            continue                    # no pillar here can see it; not a failure
        checks[f"C3_{arm}_worse_than_real"] = bool(
            by[arm].status == "DQ"
            or (np.isfinite(score(arm)) and np.isfinite(real)
                and score(arm) > real))
    return checks
