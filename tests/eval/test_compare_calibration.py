"""Calibration of the significance test, pinned so it cannot be forgotten.

`paired_delta` used to estimate the standard error of a difference by
subsampling `frac`=0.8 of each sample **without replacement** and taking the
spread of the resulting draws. That spread is the variability of a *subsample*
statistic around the full-sample value, which is not the variability of the
full-sample statistic around the truth: it needs a subsampling rescale, and the
code applied none (`ladder._noise_floor` applies exactly such a correction,
which is the inconsistency that made this a bug rather than a choice).

The consequence was a p-value that rejects too often, worst where the suite uses
it: Holm over the 153 pairs of an 18-arm run puts the operative per-test level
near 3e-4.

**Fixed 2026-09-19** by making the default an exact label-permutation test
(`compare.paired_delta(method="permutation")`), for the reason the module
docstring derives: no single rescale can serve both regimes this statistic moves
between. The arm-vs-arm difference of KIDs is a *degenerate* U-statistic when the
arms match the reference (held-out real, `solidify`, and any pair in a
same-distribution calibration experiment) and a *non-degenerate* one when they do
not (every generator pair on a real leaderboard), and the deficit measured on the
old test differs accordingly -- a factor 1.31 in the first regime, 2.06 in the
second.

The old test's damage, measured, is therefore regime-dependent and worse than the
first measurement of it suggested:

    exact null, d=32, n=96 arms vs 240 reference, kid, n_rep=80
    regime                       P(p<0.05)   P(p<0.01)
    arms == reference             0.170       0.078      (3.4x / 7.8x)
    arms != reference             0.360       0.225      (7.2x / 22.5x)

Direction of harm, which is what a reader of the paper needs:

* A claim that two arms are **not** separated was *conservative* under the old
  test -- if a too-eager test still failed to reject, a correct one certainly
  would. The interior-blindness headline is of this form and was never at risk.
* Every claim that two arms **are** separated was at risk, and the recomputation
  (`outputs/run_20260919_022424_significance_recompute/FINDINGS.md`) found real
  casualties, including the power control this paper leans on.

These tests do not assert that any particular implementation is in place. They
measure the type-I error and fail if it drifts from what was recorded, so that a
fix -- or a regression -- is visible. Update the recorded intervals deliberately,
with the measurement that justifies them.
"""

from __future__ import annotations

import numpy as np
import pytest

from blockgen.eval.bench.compare import paired_delta
from blockgen.eval.bench.distances import kid, kid_gram

#: Measured 2026-09-19 for the permutation default, on exact nulls: three
#: independent standard-normal samples (d=32, n=96 arms against a 240-row
#: reference, kid), `n_perm=4000`, 6000 trials per seed:
#:
#:     arms == reference   seeds 0/1/2   P(p<.05) .052 .047 .051
#:                                       P(p<.01) .0127 .0105 .0097
#:     arms != reference   (shift 0.6)   P(p<.05) .0485   P(p<.01) .0112
#:
#: and at the run-of-record's scale (d=768, n=128 vs 399, 800 trials):
#: P(p<.05) = .043, P(p<.01) = .0075. Pooled over 18000 trials the degenerate
#: regime lands at .0499 and .0110 -- nominal at both levels, which is the whole
#: point of the change.
#:
#: The bounds below are for the CHEAPER configuration these tests actually run
#: (n_perm=1500, n_rep=8, 800 trials). Observed there over eight runs: .036-.063
#: at alpha=0.05 and .004-.015 at alpha=0.01. The bounds absorb that Monte-Carlo
#: spread and still sit far below the old test's .155-.360.
PERMUTATION_ALPHA_05 = (0.025, 0.085)
PERMUTATION_ALPHA_01 = (0.001, 0.025)

#: The legacy test, same cheap configuration, 400 trials. Recorded so that a
#: "fix" that quietly restores the old default is caught by the test that
#: measures the old default rather than by nobody.
LEGACY_ALPHA_05 = (0.10, 0.26)

TRIALS = 800
CONFIG = dict(d=32, n=96, n_ref=240)


def _type_i(method: str, trials: int, shift: float = 0.0, seed: int = 0,
            n_rep: int = 8, n_perm: int = 1500) -> tuple:
    """`(P(p<0.05), P(p<0.01))` on an exact null: two arms, one distribution."""
    rng = np.random.default_rng(seed)
    d, n, n_ref = CONFIG["d"], CONFIG["n"], CONFIG["n_ref"]
    ps = []
    for t in range(trials):
        a = rng.normal(size=(n, d)) + shift
        b = rng.normal(size=(n, d)) + shift
        ref = rng.normal(size=(n_ref, d))
        c = paired_delta(kid, a, b, ref, n_rep=n_rep, n_perm=n_perm, method=method,
                         kernel=kid_gram if method == "permutation" else None,
                         rng=np.random.default_rng(1000 + t))
        if np.isfinite(c.p):
            ps.append(c.p)
    p = np.asarray(ps)
    return float(np.mean(p < 0.05)), float(np.mean(p < 0.01))


def _check(rate5: float, rate1: float, what: str) -> None:
    lo, hi = PERMUTATION_ALPHA_05
    assert lo <= rate5 <= hi, (
        f"{what}: type-I error at alpha=0.05 is {rate5:.4f}, outside the "
        f"recorded [{lo}, {hi}]. If it ROSE, the test has become liberal again "
        f"and every separation claim on the leaderboard is suspect. If it FELL "
        f"far below, the test has gone conservative and the power controls need "
        f"rechecking. Either way: re-measure, then update these bounds and "
        f"re-derive the separation claims.")
    lo, hi = PERMUTATION_ALPHA_01
    assert lo <= rate1 <= hi, (
        f"{what}: type-I error at alpha=0.01 is {rate1:.4f}, outside the "
        f"recorded [{lo}, {hi}]. The small-alpha tail is the one Holm operates "
        f"in -- a 153-pair table tests near 3e-4 -- so a drift here matters more "
        f"than the same drift at alpha=0.05.")


@pytest.mark.slow
def test_permutation_p_is_calibrated_when_the_arms_match_the_reference():
    """The degenerate regime: MMD^2 has null variance O(1/n^2), not O(1/n).

    Every near-real rung sits here -- held-out real, the solidify probe, a
    verbatim training copy -- and it is where a subsampling rescale derived for a
    linear statistic would be wrong by its own square.
    """
    _check(*_type_i("permutation", TRIALS), what="arms == reference")


@pytest.mark.slow
def test_permutation_p_is_calibrated_when_the_arms_differ_from_the_reference():
    """The non-degenerate regime, and the one a leaderboard actually lives in.

    Both arms are still one distribution, so the null holds; it is simply not the
    reference's distribution, which leaves a linear term in the difference. The
    old test was 7.2x too liberal here -- worse than in the regime it was first
    measured in, and the reason several generator-vs-generator separations did
    not survive the recomputation.
    """
    _check(*_type_i("permutation", TRIALS, shift=0.6), what="arms != reference")


@pytest.mark.slow
def test_the_legacy_subsample_p_is_still_anticonservative():
    """What the old default did, kept measurable so it cannot come back quietly.

    `method="subsample"` remains available to reproduce cards written before the
    fix. This pins that it is NOT a second opinion: it rejects roughly three
    times too often at alpha=0.05 even in the gentler of the two regimes.
    """
    rate5, rate1 = _type_i("subsample", 400, n_rep=80)
    lo, hi = LEGACY_ALPHA_05
    assert lo <= rate5 <= hi, (
        f"legacy type-I error at alpha=0.05 is {rate5:.4f}, outside the recorded "
        f"[{lo}, {hi}] -- the legacy path is meant to be frozen, so a change here "
        f"means `method='subsample'` no longer reproduces the cards it exists for.")
    assert rate5 > PERMUTATION_ALPHA_05[1], "legacy must remain the liberal one"


def test_the_default_is_the_calibrated_test():
    """A default is a claim about every number the suite prints unattended."""
    rng = np.random.default_rng(0)
    c = paired_delta(kid, rng.normal(size=(40, 8)), rng.normal(size=(40, 8)),
                     rng.normal(size=(90, 8)), n_rep=20, n_perm=500,
                     kernel=kid_gram, rng=np.random.default_rng(1))
    assert c.method == "permutation" and c.n_perm == 500
    assert c.p >= 1.0 / 501                      # exact tests have a floor
    assert c.se_null is not None and c.p_normal is not None


def test_a_failure_to_separate_stays_a_failure_to_separate():
    """The headline claim's direction, pinned across the change of test.

    Two identical distributions must not be separated. This passed under the
    liberal test, so it must pass under a calibrated one -- and the recomputation
    confirms it does on the real arms as well: `real@solidify` against
    `real_test` on mv_dino_kid moved from p=0.61 to p=0.77.
    """
    rng = np.random.default_rng(7)
    x = rng.normal(size=(96, 32))
    y = rng.normal(size=(96, 32))
    ref = rng.normal(size=(240, 32))
    c = paired_delta(kid, x, y, ref, n_rep=120, n_perm=4000, kernel=kid_gram,
                     rng=np.random.default_rng(8))
    assert c.p > 0.05 or not np.isfinite(c.p)
