"""Uncertainty primitives: the intervals have to be honest at small n.

The suite's whole premise is that a point estimate at n=16 is uninterpretable, so
these tests pin the cases where a CI could silently lie -- degenerate input, and
the with-replacement-vs-without distinction for kernel statistics.
"""

from __future__ import annotations

import numpy as np
import pytest

from blockgen.eval.bench import stats


def test_bootstrap_covers_the_mean():
    rng = np.random.default_rng(0)
    v = rng.normal(5.0, 1.0, size=200)
    point, lo, hi = stats.bootstrap_ci(v, n_boot=500, rng=rng)
    assert lo < point < hi
    assert lo < 5.0 < hi


def test_bootstrap_narrows_with_n():
    rng = np.random.default_rng(0)
    widths = []
    for n in (16, 256):
        v = rng.normal(0.0, 1.0, size=n)
        _, lo, hi = stats.bootstrap_ci(v, n_boot=500, rng=rng)
        widths.append(hi - lo)
    assert widths[1] < widths[0] / 2


def test_bootstrap_degenerate_input_is_nan_not_zero():
    """A single sample must not produce a zero-width interval."""
    point, lo, hi = stats.bootstrap_ci([3.0])
    assert point == 3.0
    assert np.isnan(lo) and np.isnan(hi)
    assert all(np.isnan(x) for x in stats.bootstrap_ci([]))


def test_bootstrap_ignores_non_finite():
    point, _, _ = stats.bootstrap_ci([1.0, 2.0, np.nan, 3.0], n_boot=100)
    assert point == pytest.approx(2.0)


def test_subsample_never_repeats_a_row():
    """The reason kernel statistics use subsampling: no row may pair with itself."""
    seen = []

    def fn(a, b):
        seen.append((len(set(a.ravel().tolist())), len(a)))
        return 0.0

    a = np.arange(50).reshape(50, 1)
    stats.subsample_ci(fn, a, a, frac=0.8, n_rep=20, rng=np.random.default_rng(0))
    assert all(uniq == n for uniq, n in seen), "subsample drew a duplicate row"


def test_subsample_point_estimate_uses_full_sets():
    a = np.zeros((40, 1))
    b = np.ones((40, 1))
    point, lo, hi = stats.subsample_ci(lambda x, y: float(len(x) + len(y)), a, b,
                                       frac=0.5, n_rep=10)
    assert point == 80.0          # full sets
    assert lo == hi == 40.0       # subsamples, 20 + 20


def test_subsample_degenerate_is_nan():
    a = np.zeros((2, 1))
    _, lo, hi = stats.subsample_ci(lambda x, y: 1.0, a, a, frac=0.2)
    assert np.isnan(lo) and np.isnan(hi)


def test_resample_ci_contains_its_point_estimate():
    """Regression: the percentile interval used to sit entirely above the point.

    Divergences are biased upward at small n, and resampling with replacement
    biases them upward again, so a percentile interval on a JSD-like statistic
    excluded its own estimate (measured: 0.237 with [0.246, 0.323]).
    """
    rng = np.random.default_rng(0)
    ref = rng.random(24)

    def skewed(items):
        # Stands in for a divergence: strictly decreasing in sample diversity,
        # so any duplication in a bootstrap draw inflates it.
        return float(1.0 / len(set(items)))

    items = list(range(24))
    point, lo, hi = stats.resample_ci(skewed, items, n_boot=200, rng=rng)
    assert lo <= point <= hi


@pytest.mark.parametrize("n", [4, 16, 64])
def test_every_ci_helper_brackets_its_point(n):
    rng = np.random.default_rng(1)
    v = rng.normal(size=n)
    for triple in (
        stats.bootstrap_ci(v, n_boot=200, rng=rng),
        stats.resample_ci(lambda xs: float(np.mean(xs)), list(v), n_boot=200, rng=rng),
        stats.paired_diff_ci(v, v + 1, n_boot=200, rng=rng),
    ):
        point, lo, hi = triple
        assert lo <= point <= hi


def test_jsd_bounds_and_symmetry():
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 1.0, 0.0])
    assert stats.jsd(p, p) == pytest.approx(0.0, abs=1e-12)
    assert stats.jsd(p, q) == pytest.approx(1.0, abs=1e-9)   # base 2 => max 1
    assert stats.jsd(p, q) == pytest.approx(stats.jsd(q, p))


def test_jsd_is_scale_invariant():
    p = np.array([2.0, 6.0, 2.0])
    assert stats.jsd(p, p * 100) == pytest.approx(0.0, abs=1e-12)


def test_jsd_handles_zero_vector():
    """An arm that emitted nothing must not crash the palette metric."""
    assert np.isfinite(stats.jsd(np.zeros(4), np.array([1.0, 1, 1, 1])))


def test_w1_and_iqr():
    assert stats.w1([0, 0, 0], [1, 1, 1]) == pytest.approx(1.0)
    assert stats.w1([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(0.0)
    assert stats.iqr([1, 2, 3, 4, 5]) == pytest.approx(2.0)


def test_paired_diff_ci_excludes_zero_when_separated():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 0.1, 100)
    y = rng.normal(0.0, 0.1, 100)
    point, lo, hi = stats.paired_diff_ci(x, y, n_boot=500, rng=rng)
    assert point > 0 and lo > 0


def test_chance_floor():
    assert stats.chance_floor(31) == pytest.approx(1.0 / 32)
