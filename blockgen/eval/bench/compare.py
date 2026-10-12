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
corpus in the same "not separated" group. The sign-balance value is therefore
kept only as `p_boot`.

**The p-value is a permutation p-value (`method="permutation"`), and the
studentized one it replaced was miscalibrated.** The history matters, because
every separation claim made before this change was made at the wrong level.

The old test took the subsample replicates below -- 80% of each arm, drawn
*without* replacement -- and used `draws.std(ddof=1)` as the standard error of
the full-sample difference. It is not one. Those draws measure how far an
80%-subsample statistic sits from the full-sample value; the SE asks how far the
full-sample statistic sits from the truth, and m-out-of-n subsampling converts
between the two only after a rescale, which the code never applied
(`ladder._noise_floor` applies exactly such a correction three modules away,
which is what makes this an inconsistency rather than a convention). Measured on
exact nulls -- two arms drawn from ONE distribution, so every rejection is a
false one -- at d=32, n=96 arms against a 240-row reference:

    null regime            P(p<0.05)   P(p<0.01)
    arms == reference        0.170       0.078      (3.4x / 7.8x)
    arms != reference        0.360       0.225      (7.2x / 22.5x)

The second row is the one a leaderboard lives on: two generators, neither of them
distributed like the reference. (The 2.1x / 6.7x first recorded for this bug was
the first row, measured at the run's own scale.) And it is worst where this module
actually operates, because Holm over the 153 pairs of an 18-arm run tests at an
effective level near 3e-4.

Three ways out were considered, and the rescale is the one that cannot be made
to work:

* **Rescale the subsampling.** The factor depends on how *degenerate* the
  statistic is, and this statistic changes regime between comparisons. Write the
  arm term conditional on a fixed reference; its Hajek projection has per-build
  influence `u(x) - v(x)` with `u(x) = E_P k(x, X')` and
  `v(x) = mean_j k(x, ref_j)`. When an arm's distribution differs from the
  reference's that influence is non-zero, the difference is a non-degenerate
  U-statistic with `Var = O(1/n)`, and the correct factor is `sqrt(m/(n-m))` = 2
  at `frac=0.8`. When an arm matches the reference -- held-out real, `solidify`,
  any near-real control, and every pair in a calibration experiment drawn from
  one distribution -- the influence vanishes, MMD^2 becomes a *degenerate*
  U-statistic with `Var = O(1/n^2)`, and the factor is its square, `m/(n-m)` = 4.
  Measured against a Monte-Carlo truth (3000 fresh arm pairs against one fixed
  reference), the non-degenerate side lands on its derived factor exactly --
  true SE / used SE = 2.06 against 2.01 predicted -- while the degenerate side
  lands at 1.31 rather than 4, because a finite reference leaves the statistic
  only partly degenerate and the subsample draws resample the reference too.
  Either way the number needed is not a constant: across the 153 pairs of one real
  table the ratio of the permutation SE to this one runs from 0.30 to 11.9. A
  single factor is wrong at one end of a leaderboard or the other, and the rungs it
  would be wrong about are exactly the ones the headline is stated on.
* **Estimate the variance analytically.** Doable, but it means writing and
  defending an influence-function estimator whose leading term *is* the thing
  that vanishes in the degenerate regime -- where the null is a weighted
  chi-square mixture rather than a normal, so a variance alone does not give the
  tail. The measured null is visibly non-normal: the table above inflates by 3.4x
  at alpha=0.05 and 7.8x at alpha=0.01 in the SAME experiment, which no single
  wrong scale factor can produce.
* **Permute the arm labels.** Pool the two arms, re-split at the observed sizes,
  recompute the difference against the same (full, fixed) reference. Under the
  null that the two arms are draws from one distribution the pooled rows are
  exchangeable, so this is *exact* -- at any n, in either regime, with no
  normality and no variance estimate. It is also what the MMD literature does,
  which is worth something in a rebuttal.

Permutation it is, with the Gram matrix formed once and every replicate read off
it by index (`distances.kid_gram`), which is what makes an exact test affordable:
10k replicates of a 128-vs-128 pair cost about 0.1 s, and 100k about 1 s.

Two consequences, stated rather than buried.

**The floor moves but does not leave.** An exact permutation p cannot go below
`1/(n_perm + 1)`. Holm over 153 pairs needs a raw p below 3.3e-4, so `n_perm`
must be at least ~3k for such a table to resolve anything at all; the default is
10k and the runner's `--n-perm` exists to raise it. `p_normal` is reported beside
it -- the same `|delta| / se` form as the old test, but with `se` taken from the
permutation null instead of from mis-scaled subsamples -- so a difference sitting
far beyond the floor can still be quoted as such. It is a companion, never the
verdict: it assumes the normality that the degenerate regime denies.

**The null is sharper than the claim.** Permuting labels tests "these two arms
are one distribution", while the quantity reported is a difference of distances
to a reference. Two genuinely different arms can sit equidistant from the
reference, and against such a pair this test may reject while `delta` is ~0 -- so
a rejection licenses "these arms differ, and the point estimate says which way",
not "the distances differ" on its own. The direction is read off `delta`, which
is why `winner` requires both significance and a non-zero delta. The reference is
held FIXED (never resubsampled) inside the permutation, which is the honest
conditioning: one reference was drawn, every arm was scored against it, and the
claim is about the arms.

The legacy behaviour stays reachable as `method="subsample"`, so a card written
before this change reproduces exactly -- including its CI, which comes from the
same subsample draws and is untouched by any of this. Those draws still carry the
rescale problem, so that interval is a spread-of-subsamples interval, as its own
paragraph above says, and not a calibrated interval for the full-sample
difference.
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
    p: float                       # permutation, unless `method` says otherwise
    p_adj: Optional[float] = None
    p_boot: Optional[float] = None  # sign-balance; floored at 1/n_rep
    n_rep: int = 0
    direction: str = "lower_better"
    paired_on: str = "reference"
    method: str = "permutation"     # "permutation" | "subsample" | "bootstrap"
    n_perm: int = 0                 # permutation replicates behind `p`
    se_null: Optional[float] = None   # sd of the null distribution of `delta`
    p_normal: Optional[float] = None  # |delta|/se_null, normal reference

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
                "winner": self.winner, "paired_on": self.paired_on,
                "method": self.method, "n_perm": self.n_perm,
                "se_null": self.se_null, "p_normal": self.p_normal}


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
    """Studentized p for `delta = 0`: `2 * (1 - Phi(|point| / se))`, `se` from `draws`.

    **Whether this is a p-value at all depends entirely on what `draws` are.**

    * Given permutation replicates it is a sound companion to the exact test:
      `se` is then the null sd, and the only assumption left is normality of the
      null. That is what `p_normal` reports, and it exists because an exact p is
      floored at `1/(n_perm + 1)` while this one is not -- so a difference far
      beyond the floor can still be quoted.
    * Given the `frac`-subsample replicates it is the pre-2026-09 primary
      p-value, and it is **wrong**: the spread of an m-out-of-n subsample
      statistic is not the standard error of the full-sample one, so the ratio is
      inflated by a factor that varies per comparison (measured 1.3x to 2.1x, and
      to 11.9x on a real table). It is kept for `method="subsample"`, which
      exists to reproduce cards written before the fix and for nothing else.

    Normality is the assumption `stats.resample_ci` already makes for
    divergences, and it fails hardest exactly where MMD^2 is degenerate -- the
    near-real rungs -- where the null is a weighted chi-square mixture. The
    interval reported beside it stays a percentile interval and makes no such
    assumption, so a disagreement between the two is visible rather than hidden.
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


def _permutation_p(obs: float, draws: np.ndarray) -> float:
    """Two-sided exact permutation p: `(1 + #{|d| >= |obs|}) / (n + 1)`.

    The observed split is one of the permutations, which is why the numerator and
    the denominator each carry a `+1`: without it the p-value can reach 0, which
    no finite permutation set can demonstrate, and the test loses its guarantee
    at small `n_perm`. The floor is `1/(n_perm + 1)` and it is real -- see the
    module docstring on what `p_normal` is for.

    `|.|` on both sides makes it two-sided without assuming the null is
    symmetric, which it is not when the two arms have different sizes: the
    statistic's permutation distribution is then skewed, and comparing absolute
    values against absolute values is still exact because it is a function of the
    permuted labels alone.

    The tolerance absorbs the floating-point difference between the observed
    statistic and the identical split appearing as a replicate; without it a
    replicate that *is* the observed configuration can land on the wrong side of
    the comparison and the test reads as more significant than it is.
    """
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0 or not np.isfinite(obs):
        return float("nan")
    tol = 1e-12 * max(1.0, abs(float(obs)))
    hits = int(np.count_nonzero(np.abs(d) >= abs(float(obs)) - tol))
    return float((1.0 + hits) / (d.size + 1.0))


def _perm_draws_kernel(kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
                       a: np.ndarray, b: np.ndarray, ref: np.ndarray,
                       n_perm: int, rng: np.random.Generator,
                       chunk: int = 512) -> Tuple[float, np.ndarray]:
    """Label-permutation replicates of the arm-term difference, from one Gram.

    The statistic is
    `sum_{i != j in S} K_ij / (m(m-1)) - 2 * mean_j K(S, ref)` differenced
    between `S` and its complement in the pooled set -- i.e. exactly
    `kid(a, ref) - kid(b, ref)`, whose reference self-term cancels
    (`distances.kid_arm_term`). Every quantity in it is a sum over index sets of
    a matrix that does not depend on the labels, so the kernel is formed ONCE and
    a replicate costs two mask multiplications. Writing it with indicator
    matrices lets a whole chunk of replicates go through one BLAS call, which is
    the difference between 0.1 s and 30 s per pair at 10k replicates.

    Returns `(observed statistic, draws)`. The observed value is recomputed here
    from the same expression rather than taken from the caller, so that it and
    the replicates share every rounding decision and `_permutation_p`'s tie
    handling means what it says.
    """
    a, b, ref = np.asarray(a), np.asarray(b), np.asarray(ref)
    pooled = np.concatenate([a, b], axis=0)
    na, nb = len(a), len(b)
    n = na + nb
    kmat = np.asarray(kernel(pooled, pooled), dtype=float)
    kref = np.asarray(kernel(pooled, ref), dtype=float).mean(axis=1)
    diag = np.diag(kmat).copy()

    def term(mask: np.ndarray, m: int) -> np.ndarray:
        # mask: [k, n] of 0/1. `((M @ K) * M).sum(1)` is the full block sum
        # including the diagonal, which `M @ diag` removes -- the same
        # off-diagonal convention that makes the estimator unbiased.
        block = ((mask @ kmat) * mask).sum(axis=1) - mask @ diag
        return block / (m * (m - 1)) - 2.0 * (mask @ kref) / m

    def delta(mask: np.ndarray) -> np.ndarray:
        return term(mask, na) - term(1.0 - mask, nb)

    m0 = np.zeros((1, n))
    m0[0, :na] = 1.0
    obs = float(delta(m0)[0])

    out = np.empty(int(n_perm), dtype=float)
    for start in range(0, int(n_perm), chunk):
        k = min(chunk, int(n_perm) - start)
        # A uniformly random size-`na` subset per row: rank `k*n` uniforms and
        # keep the `na` smallest. Cheaper than `k` calls to `rng.choice` and
        # consumes the stream in one draw, which keeps a run reproducible.
        take = np.argpartition(rng.random((k, n)), na - 1, axis=1)[:, :na]
        mask = np.zeros((k, n))
        np.put_along_axis(mask, take, 1.0, axis=1)
        out[start:start + k] = delta(mask)
    return obs, out


def _perm_draws_generic(g: Callable[[np.ndarray, np.ndarray], float],
                        a: np.ndarray, b: np.ndarray, ref: np.ndarray,
                        n_perm: int, rng: np.random.Generator
                        ) -> Tuple[float, np.ndarray]:
    """The same permutations for a statistic with no kernel on offer.

    Correct and general; a few hundred times slower, because the statistic is
    recomputed from features on every replicate. `n_perm` defaults are set for
    the kernel path, so a caller landing here should expect to lower them and to
    lose the ability to clear a Holm-corrected level in a wide table.
    """
    a, b, ref = np.asarray(a), np.asarray(b), np.asarray(ref)
    pooled = np.concatenate([a, b], axis=0)
    na, n = len(a), len(a) + len(b)
    obs = float(g(a, ref)) - float(g(b, ref))
    out = np.empty(int(n_perm), dtype=float)
    for r in range(int(n_perm)):
        idx = rng.permutation(n)
        out[r] = float(g(pooled[idx[:na]], ref)) - float(g(pooled[idx[na:]], ref))
    return obs, out


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
    method: str = "permutation",
    n_perm: int = 10_000,
    kernel: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
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

    `method` selects the test, and the default changed (module docstring):

    * `"permutation"` -- exchange arm labels between `a` and `b`, `n_perm` times,
      against the FULL reference; exact under the null that the arms are one
      distribution. Pass `kernel` (e.g. `distances.kid_gram`) whenever `fn` is a
      kernel statistic: it makes the exact test ~300x cheaper and is the only way
      an `n_perm` large enough to clear a Holm-corrected level is affordable.
      Without it the permutations recompute `fn` one at a time.
    * `"subsample"` -- the pre-2026-09 studentized p, kept only so that cards
      written before the fix reproduce. It is anticonservative (2.1x at
      alpha=0.05, 6.7x at alpha=0.01) and must not be used for a new claim.

    The subsample replicates still run under either method, because the reported
    interval and `p_boot` come from them and are unchanged by the choice -- and
    they are drawn FIRST, from the same `rng`, so a re-run with
    `method="subsample"` reproduces an old card's interval and p exactly.
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
                          direction=direction, method=method)
    if method not in ("permutation", "subsample"):
        raise ValueError(f"unknown method {method!r}: "
                         f"expected 'permutation' or 'subsample'")

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

    if method == "subsample":
        # Legacy. The spread of the subsample draws, used directly as an SE;
        # `se_null` records the number that was wrong rather than hiding it.
        finite = draws[np.isfinite(draws)]
        se = float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
        return Comparison(name_a, name_b, point, (lo, hi), _normal_p(point, draws),
                          n_rep=n_rep, direction=direction,
                          p_boot=_empirical_p(draws), method=method,
                          se_null=se, p_normal=_normal_p(point, draws))

    if kernel is not None:
        obs, perm = _perm_draws_kernel(kernel, a, b, ref, n_perm, rng)
        # The kernel has to be the one behind `fn`, and the cheapest proof is
        # that it reproduces the point estimate. A mismatch means the caller
        # paired a statistic with someone else's kernel, which would silently
        # test a different hypothesis -- so it fails loudly instead.
        if np.isfinite(point) and abs(obs - point) > 1e-6 * max(1.0, abs(point)):
            raise ValueError(
                f"kernel does not reproduce fn: permutation statistic {obs!r} "
                f"vs point estimate {point!r} for {name_a} vs {name_b}")
    else:
        obs, perm = _perm_draws_generic(g, a, b, ref, n_perm, rng)
    finite = perm[np.isfinite(perm)]
    se = float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
    return Comparison(name_a, name_b, point, (lo, hi),
                      _permutation_p(obs, perm),
                      n_rep=n_rep, direction=direction,
                      p_boot=_empirical_p(draws), method=method,
                      n_perm=int(n_perm), se_null=se,
                      p_normal=_normal_p(point, perm))


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

    **This one is not touched by the subsampling bug** that moved `paired_delta`
    to a permutation test. These replicates are an ordinary bootstrap -- with
    replacement, at full size -- of a mean, so the spread of the draws *is* an
    estimate of the standard error and needs no rescale; the arms are different
    builds, so there is nothing to permute that is not already resampled.
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
    se = float(draws.std(ddof=1))
    return Comparison(name_a, name_b, point, (lo, hi), _normal_p(point, draws),
                      n_rep=n_boot, direction=direction, paired_on="none",
                      p_boot=_empirical_p(draws), method="bootstrap",
                      se_null=se, p_normal=_normal_p(point, draws))


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


def _maximal_cliques(n: int, adj: List[set]) -> List[List[int]]:
    """Every maximal clique of an undirected graph (Bron-Kerbosch with a pivot).

    Kept local and index-based: the graphs here have as many vertices as a
    leaderboard has arms (18 in the largest run to date), so the exponential
    worst case is not reachable in practice and a dependency would not pay for
    itself.
    """
    out: List[List[int]] = []

    def expand(r: set, cand: set, excl: set) -> None:
        if not cand and not excl:
            out.append(sorted(r))
            return
        pivot = max(cand | excl, key=lambda v: len(adj[v] & cand))
        for v in sorted(cand - adj[pivot]):
            expand(r | {v}, cand & adj[v], excl & adj[v])
            cand = cand - {v}
            excl = excl | {v}

    expand(set(), set(range(n)), set())
    return out


def _letter_groups(order: Sequence[str], beats: Callable[[str, str], Optional[str]]
                   ) -> Dict[str, str]:
    """A **valid** compact letter display: arms share a letter iff no test
    separated them.

    The standard way to report an all-pairs comparison without making the reader
    scan a triangular matrix -- and the standard way to get it wrong is to assign
    each arm one letter greedily, which is what this function did until
    2026-09-19. "Not separated" is not transitive: with `a~b`, `b~c` and `a`
    beating `c`, no single-letter partition can be honest, and the greedy version
    resolved the conflict by giving `c` a fresh letter. That made `letters[arm]` a
    list whose `break` guaranteed one entry, and it meant the *reverse* reading of
    the display -- "these two do not share a letter, so they were separated" --
    was unsupported. Every "occupies a group of its own" sentence written off such
    a table was an unproven claim; several of them were true, but the table was
    not the evidence.

    A CLD is a covering by *maximal* sets of mutually indistinguishable arms, not
    a partition, so an arm can carry several letters. Each letter is one maximal
    clique of the "not separated" graph, which gives exactly the biconditional a
    reader assumes:

    * two arms share a letter  <=>  their comparison did not reject;
    * two arms share no letter <=>  it did.

    An unavailable comparison (`beats` returning None because the test could not
    be computed) counts as "not separated", which is conservative in the same
    direction the rest of the module is: it can only merge groups, never split
    them.

    Letters run `a`, `b`, ... in the order a clique's best-ranked member appears
    in `order`. Past 26 cliques the labels become comma-separated numbers, since
    `aa` next to `ab` in a concatenated string is not readable as two groups.
    """
    arms = list(order)
    idx = {a: i for i, a in enumerate(arms)}
    adj: List[set] = [set() for _ in arms]
    for i, a in enumerate(arms):
        for j in range(i + 1, len(arms)):
            if beats(a, arms[j]) is None:          # not separated
                adj[i].add(j)
                adj[j].add(i)
    cliques = _maximal_cliques(len(arms), adj)
    # Deterministic, and reading down the table the letters advance: order the
    # groups by their members' ranks.
    cliques.sort(key=lambda c: [idx[arms[v]] for v in c])
    letters = [chr(ord("a") + i) for i in range(len(cliques))] if len(cliques) <= 26 \
        else [str(i + 1) for i in range(len(cliques))]
    sep = "" if len(cliques) <= 26 else ","
    held: Dict[str, List[str]] = {a: [] for a in arms}
    for label, clique in zip(letters, cliques):
        for v in clique:
            held[arms[v]].append(label)
    return {a: sep.join(v) for a, v in held.items()}


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
    method: str = "permutation",
    n_perm: int = 10_000,
    kernel: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> RankTable:
    """Rank arms on one distribution metric, with all pairs tested and corrected.

    `method`, `n_perm` and `kernel` go straight to `paired_delta`. Holm over
    k(k-1)/2 pairs is what makes `n_perm` a real constraint rather than a knob: at
    k=18 the smallest adjusted p a raw permutation p can produce is
    `153 / (n_perm + 1)`, so 10k replicates bottom out at an adjusted 0.015 and
    anything below ~3k cannot separate a single pair however large the effect.
    """
    rng = rng or np.random.default_rng(0)
    names = list(arm_feats)
    scores = {k: float(fn(np.asarray(v), ref)) for k, v in arm_feats.items()}

    comps: List[Comparison] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            comps.append(paired_delta(fn, arm_feats[a], arm_feats[b], ref,
                                      frac=frac, n_rep=n_rep, alpha=alpha, rng=rng,
                                      direction=direction, name_a=a, name_b=b,
                                      arm_term=arm_term, method=method,
                                      n_perm=n_perm, kernel=kernel))
    for c, p in zip(comps, holm([c.p for c in comps])):
        c.p_adj = None if np.isnan(p) else float(p)

    table = RankTable(metric=metric, direction=direction, scores=scores,
                      comparisons=comps)
    table.groups = _letter_groups(table.order, table.beats)
    return table


def render_rank_table(table: RankTable, digits: int = 3) -> str:
    """Markdown leaderboard. Arms share a letter exactly when no test separated
    them -- both directions, since `_letter_groups` is a valid CLD."""
    lines = [f"### {table.metric} ({table.direction.replace('_', ' ')})", "",
             "| # | arm | score | group |", "|---|---|---|---|"]
    for i, arm in enumerate(table.order, 1):
        lines.append(f"| {i} | {arm} | {table.scores[arm]:.{digits}f} "
                     f"| {table.groups.get(arm, '')} |")
    methods = sorted({c.method for c in table.comparisons})
    perms = sorted({c.n_perm for c in table.comparisons if c.n_perm})
    how = "/".join(methods) + (f", n_perm={min(perms)}" if perms else "")
    lines += ["", "Arms sharing a group letter are **not** separated at "
              "alpha=0.05 after Holm correction; arms sharing **no** letter "
              f"are. Test: {how}.", ""]
    sig = [c for c in table.comparisons if c.significant]
    if sig:
        lines += ["| comparison | delta | 95% CI | p | p (Holm) | p (normal) |",
                  "|---|---|---|---|---|---|"]
        for c in sorted(sig, key=lambda c: (c.p, -abs(c.delta))):
            pn = "" if c.p_normal is None else f"{c.p_normal:.2e}"
            lines.append(
                f"| {c.a} vs {c.b} | {c.delta:+.{digits}f} "
                f"| [{c.ci[0]:+.{digits}f}, {c.ci[1]:+.{digits}f}] "
                f"| {c.p:.4f} | {c.p_adj:.4f} | {pn} |")
        lines.append("")
    return "\n".join(lines)
