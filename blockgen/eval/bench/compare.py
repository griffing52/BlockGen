"""Head-to-head tests between arms. What a leaderboard needs and CIs cannot give.

The scorecard reports every arm with a marginal interval, and the temptation is
to read two of them side by side and call overlapping intervals a tie. That is
wrong in both directions, and the error is not small.

**Overlapping intervals are not a null result**, and the gap is a fixed factor,
not a judgement call. Two arms are distinguishable when their difference exceeds
about `1.96 * sqrt(sd_a^2 + sd_b^2)`, i.e. `sqrt(h_a^2 + h_b^2)` in half-widths.
Non-overlap of the two intervals demands `h_a + h_b`. For equal-width intervals
that is `2h` against `1.41h`: **the overlap test needs the difference to be 41%
larger before it will call anything.** Read that way, a benchmark silently
declines to separate arms it has the power to separate.

So `paired_delta` tests the difference directly, and `rank_table` is the
leaderboard built from it.

**On the pairing, honestly.** Both arms are scored against the same reference
subsample in each replicate. That is the correct null -- the reference really is
shared, and treating it as independently drawn per arm would model a design that
was not run -- but it should not be advertised as a power gain. Measured here on
DINO-scale features across four reference and arm sizes, pairing moved the
interval width by less than 10% and not consistently in one direction: the
reference's contribution to an unbiased MMD is a term the difference already
cancels analytically, so there is little left for the pairing to remove. The
power in this module comes from testing a difference at all, not from pairing it.

**What the interval is a claim about.** Replicates subsample the arm's own
builds, so the interval says how much the score would move had a different 80%
of *these* builds been scored. Generalizing from that to "this generator beats
that generator" is an extra step the resampling does not take, and it is the
usual position for sample-based generative metrics (no FID interval is a
population claim either). It is why `min_n` exists and why headline numbers want
n >= 256.

Multiplicity is handled, because a leaderboard runs every pair. With k arms
there are k(k-1)/2 tests and at k=7 the chance of at least one spurious
"significant" difference at alpha=0.05 is about two in three. `rank_table`
applies Holm-Bonferroni across the family and reports both raw and adjusted
p-values, so the correction is visible rather than assumed.

That correction dictates the p-value. A sign-balance bootstrap p cannot go below
`1/n_rep`, and Holm over the 55 pairs of an 11-arm run multiplies that floor to
0.275 -- so with 200 replicates *nothing* is separable, however large the
difference. The first run of this module duly put held-out real and a monochrome
corpus in the same "not separated" group. The reported `p` is therefore
studentized (`|delta| / se`, normal reference), which the replicates estimate
well and which is not floored; the sign-balance value is kept alongside as
`p_boot`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Comparison:
    """One head-to-head result. `delta = score(a) - score(b)`."""
    a: str
    b: str
    delta: float
    ci: Tuple[float, float]
    p: float                       # studentized; see `_normal_p`
    p_adj: Optional[float] = None
    p_boot: Optional[float] = None  # sign-balance; floored at 1/n_rep
    n_rep: int = 0
    direction: str = "lower_better"
    paired_on: str = "reference"

    @property
    def significant(self) -> bool:
        """At the adjusted level when one exists, the raw level otherwise."""
        return (self.p_adj if self.p_adj is not None else self.p) < 0.05

    @property
    def winner(self) -> Optional[str]:
        """The better arm, or None when the difference is not resolved."""
        if not self.significant or self.delta == 0:
            return None
        a_better = (self.delta < 0) if self.direction == "lower_better" else (self.delta > 0)
        return self.a if a_better else self.b

    def to_json(self) -> dict:
        return {"a": self.a, "b": self.b, "delta": self.delta,
                "ci": list(self.ci), "p": self.p, "p_adj": self.p_adj,
                "p_boot": self.p_boot,
                "n_rep": self.n_rep, "direction": self.direction,
                "winner": self.winner, "paired_on": self.paired_on}


def _empirical_p(draws: np.ndarray) -> float:
    """Sign-balance p: `2 * min(P(d <= 0), P(d >= 0))`.

    Reported for transparency, **not** used for significance. It is floored at
    `1/n_rep` -- a bootstrap with no crossing replicate has not demonstrated
    `p = 0`, only `p < 1/n_rep` -- and that floor is fatal under a multiplicity
    correction. With 200 replicates the smallest attainable value is 0.005;
    Holm over the 55 pairs of an 11-arm run multiplies it to 0.275, so *no*
    comparison can reach 0.05 however large the difference. The first run of
    this module put every arm, from held-out real to a monochrome corpus, in a
    single "not separated" group for exactly that reason.
    """
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    lo = float(np.mean(d <= 0)), float(np.mean(d >= 0))
    # The ceiling matters too: when every replicate is exactly zero both tail
    # masses are 1 and `2 * min` is 2, a p-value above 1.
    return float(min(max(2.0 * min(lo), 1.0 / d.size), 1.0))


def _normal_p(point: float, draws: np.ndarray) -> float:
    """Studentized p for `delta = 0`: `2 * (1 - Phi(|point| / se))`.

    The primary p-value, because it is not floored by the replicate count. The
    replicates are used only to estimate the standard error of the difference,
    which a few hundred of them do well; the significance claim then comes from
    a normal reference rather than from counting crossings, so a large effect
    can produce a small enough p to survive Holm over dozens of pairs.

    The cost is an assumption -- that the resampling distribution of the
    difference is approximately normal -- which is reasonable for a difference of
    two kernel statistics and is the same assumption `stats.resample_ci` already
    makes for divergences. The interval reported beside it stays a percentile
    interval and makes no such assumption, so a disagreement between the two is
    visible rather than hidden.
    """
    from scipy.stats import norm

    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 2 or not np.isfinite(point):
        return float("nan")
    se = float(d.std(ddof=1))
    if se <= 1e-15:
        return 0.0 if abs(point) > 1e-15 else 1.0
    return float(min(2.0 * norm.sf(abs(point) / se), 1.0))


def paired_delta(
    fn: Callable[[np.ndarray, np.ndarray], float],
    a: np.ndarray,
    b: np.ndarray,
    ref: np.ndarray,
    frac: float = 0.8,
    n_rep: int = 400,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    direction: str = "lower_better",
    name_a: str = "a",
    name_b: str = "b",
    arm_term: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> Comparison:
    """Test `fn(a, ref) - fn(b, ref)` with the reference subsample **shared**.

    `arm_term` is an optional cheaper stand-in obeying
    `fn(x, ref) = arm_term(x, ref) + c(ref)`. Because the reference draw is
    shared, `c(ref)` cancels in every replicate, so the difference is unchanged
    and the reference self-kernel -- the dominant cost, the reference being much
    larger than any arm -- is never formed. See `distances.kid_arm_term`.

    Subsampling is without replacement on all three sides, for the reason
    `stats.subsample_ci` documents: the unbiased kernel estimators treat a
    duplicated row as an independent pair and are biased upward by it.

    The point estimate is computed on the full sets; only the interval and the
    p-value come from the replicates.
    """
    a, b, ref = np.asarray(a), np.asarray(b), np.asarray(ref)
    point = float(fn(a, ref)) - float(fn(b, ref))
    ma, mb, mr = (int(round(frac * len(x))) for x in (a, b, ref))
    # A subsample that is not smaller than its source is not a subsample: every
    # replicate is the identical set, the spread is exactly zero, and the test
    # reports p = 1/n_rep from no evidence at all. With two builds and
    # frac=0.8 that is precisely what happens, so it is refused here rather
    # than reported as a confident verdict.
    if min(ma, mb, mr) < 2 or ma >= len(a) or mb >= len(b) or mr >= len(ref):
        return Comparison(name_a, name_b, point, (float("nan"),) * 2, float("nan"),
                          direction=direction)

    rng = rng or np.random.default_rng(0)
    g = arm_term or fn
    draws = np.empty(n_rep, dtype=float)
    for r in range(n_rep):
        ir = rng.choice(len(ref), size=mr, replace=False)     # shared by both arms
        ia = rng.choice(len(a), size=ma, replace=False)
        ib = rng.choice(len(b), size=mb, replace=False)
        draws[r] = g(a[ia], ref[ir]) - g(b[ib], ref[ir])

    lo = float(np.percentile(draws, 100 * alpha / 2))
    hi = float(np.percentile(draws, 100 * (1 - alpha / 2)))
    return Comparison(name_a, name_b, point, (lo, hi), _normal_p(point, draws),
                      n_rep=n_rep, direction=direction,
                      p_boot=_empirical_p(draws))


def paired_delta_per_sample(
    x: Sequence[float],
    y: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    direction: str = "lower_better",
    name_a: str = "a",
    name_b: str = "b",
) -> Comparison:
    """The same test for metrics that are means over independent per-build values.

    Coherence, novelty and faithfulness are of this kind, and there is nothing to
    pair on -- two arms are different builds -- so this is an ordinary two-sample
    bootstrap on the difference of means. It is kept here rather than in `stats`
    so that every head-to-head number in a report comes out of one module with
    one convention for `direction` and one p-value definition.
    """
    xv = np.asarray(list(x), dtype=float)
    yv = np.asarray(list(y), dtype=float)
    xv, yv = xv[np.isfinite(xv)], yv[np.isfinite(yv)]
    point = float(xv.mean() - yv.mean()) if xv.size and yv.size else float("nan")
    if xv.size < 2 or yv.size < 2:
        return Comparison(name_a, name_b, point, (float("nan"),) * 2, float("nan"),
                          direction=direction, paired_on="none")
    rng = rng or np.random.default_rng(0)
    draws = (xv[rng.integers(0, xv.size, (n_boot, xv.size))].mean(1)
             - yv[rng.integers(0, yv.size, (n_boot, yv.size))].mean(1))
    lo = float(np.percentile(draws, 100 * alpha / 2))
    hi = float(np.percentile(draws, 100 * (1 - alpha / 2)))
    return Comparison(name_a, name_b, point, (lo, hi), _normal_p(point, draws),
                      n_rep=n_boot, direction=direction, paired_on="none",
                      p_boot=_empirical_p(draws))


def holm(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values, order preserved.

    Holm rather than plain Bonferroni: it controls the same family-wise error
    rate and is uniformly more powerful, which matters at the sample sizes here
    where a leaderboard is already short of power. NaNs pass through so a
    comparison that could not be computed is not silently counted as a test.
    """
    p = np.asarray(list(p_values), dtype=float)
    ok = np.where(np.isfinite(p))[0]
    out = np.full(p.shape, np.nan)
    if ok.size == 0:
        return out.tolist()
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        out[idx] = min(running, 1.0)
    return out.tolist()


@dataclass
class RankTable:
    """A leaderboard: arms ordered, with every pairwise verdict behind it."""
    metric: str
    direction: str
    scores: Dict[str, float]
    comparisons: List[Comparison] = field(default_factory=list)
    groups: Dict[str, str] = field(default_factory=dict)

    @property
    def order(self) -> List[str]:
        rev = self.direction == "higher_better"
        return sorted(self.scores, key=lambda k: (np.isnan(self.scores[k]),
                                                  -self.scores[k] if rev
                                                  else self.scores[k]))

    def beats(self, a: str, b: str) -> Optional[str]:
        for c in self.comparisons:
            if {c.a, c.b} == {a, b}:
                return c.winner
        return None

    def to_json(self) -> dict:
        return {"metric": self.metric, "direction": self.direction,
                "order": self.order, "scores": self.scores,
                "groups": self.groups,
                "comparisons": [c.to_json() for c in self.comparisons]}


def _letter_groups(order: Sequence[str], beats: Callable[[str, str], Optional[str]]
                   ) -> Dict[str, str]:
    """Compact letter display: arms sharing a letter are not separated.

    The standard way to report an all-pairs comparison without making the reader
    scan a triangular matrix. An arm joins a group only if it is
    indistinguishable from *every* current member, so a shared letter always
    means "no pair inside this group was resolved" -- never merely "adjacent
    ones were not".
    """
    groups: List[List[str]] = []
    for arm in order:
        for g in groups:
            if all(beats(arm, other) is None for other in g):
                g.append(arm)
                break
        else:
            groups.append([arm])
    letters: Dict[str, List[str]] = {a: [] for a in order}
    for i, g in enumerate(groups):
        for arm in g:
            letters[arm].append(chr(ord("a") + i))
    return {a: "".join(v) for a, v in letters.items()}


def rank_table(
    metric: str,
    fn: Callable[[np.ndarray, np.ndarray], float],
    arm_feats: Dict[str, np.ndarray],
    ref: np.ndarray,
    direction: str = "lower_better",
    frac: float = 0.8,
    n_rep: int = 400,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    arm_term: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> RankTable:
    """Rank arms on one distribution metric, with all pairs tested and corrected."""
    rng = rng or np.random.default_rng(0)
    names = list(arm_feats)
    scores = {k: float(fn(np.asarray(v), ref)) for k, v in arm_feats.items()}

    comps: List[Comparison] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            comps.append(paired_delta(fn, arm_feats[a], arm_feats[b], ref,
                                      frac=frac, n_rep=n_rep, alpha=alpha, rng=rng,
                                      direction=direction, name_a=a, name_b=b,
                                      arm_term=arm_term))
    for c, p in zip(comps, holm([c.p for c in comps])):
        c.p_adj = None if np.isnan(p) else float(p)

    table = RankTable(metric=metric, direction=direction, scores=scores,
                      comparisons=comps)
    table.groups = _letter_groups(table.order, table.beats)
    return table


def render_rank_table(table: RankTable, digits: int = 3) -> str:
    """Markdown leaderboard. Arms sharing a letter were not separated."""
    lines = [f"### {table.metric} ({table.direction.replace('_', ' ')})", "",
             "| # | arm | score | group |", "|---|---|---|---|"]
    for i, arm in enumerate(table.order, 1):
        lines.append(f"| {i} | {arm} | {table.scores[arm]:.{digits}f} "
                     f"| {table.groups.get(arm, '')} |")
    lines += ["", "Arms sharing a group letter are **not** separated at "
              "alpha=0.05 after Holm correction.", ""]
    sig = [c for c in table.comparisons if c.significant]
    if sig:
        lines += ["| comparison | delta | 95% CI | p | p (Holm) |",
                  "|---|---|---|---|---|"]
        for c in sorted(sig, key=lambda c: c.p):
            lines.append(
                f"| {c.a} vs {c.b} | {c.delta:+.{digits}f} "
                f"| [{c.ci[0]:+.{digits}f}, {c.ci[1]:+.{digits}f}] "
                f"| {c.p:.4f} | {c.p_adj:.4f} |")
        lines.append("")
    return "\n".join(lines)
