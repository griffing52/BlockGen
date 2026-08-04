"""Feature-space distances: unbiasedness, and the properties each metric claims.

A note on what "unbiased" does and does not mean here, because it is easy to
write the wrong assertion. The unbiased MMD^2 estimator drops the diagonal from
the two within-set sums but keeps every cross term, so evaluating it on one
array against *itself* yields a small **negative** number (measured -0.011 on
real DINOv2 features), not zero. The property worth testing is that it is ~0
for two *independent* samples drawn from the same distribution, and that it does
not drift with n. An assertion like `mmd(x, x) == 0` would be testing the biased
estimator's behaviour by mistake.
"""

from __future__ import annotations

import numpy as np
import pytest

from blockgen.eval.bench import distances as D


def _unit(n, d=32, loc=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=loc, size=(n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# --- unbiasedness ----------------------------------------------------------
def _null_kid(n: int, reps: int = 40):
    """Distribution of KID between two disjoint halves of one population."""
    vals = []
    for s in range(reps):
        rng = np.random.default_rng(1000 + s)
        pool = _unit(2 * n, seed=s)
        pi = rng.permutation(len(pool))
        vals.append(D.kid(pool[pi[:n]], pool[pi[n:]]))
    v = np.asarray(vals, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(reps))


@pytest.mark.parametrize("n", [16, 64, 256])
def test_kid_null_is_zero_in_expectation(n):
    """Unbiasedness, tested against its own standard error rather than a
    hard threshold -- the statistic's scale depends on how spread the features
    are, and these synthetic unit vectors are far more spread than real DINOv2
    features (whose pairwise cosine sits around 0.998)."""
    mean, sem = _null_kid(n)
    assert abs(mean) < 3 * sem, f"null mean {mean:.4f} is {mean / sem:.1f} SE from 0"


def test_kid_null_does_not_drift_with_n():
    """The property that makes arms of different sample size comparable at all.

    Contrast `test_legacy_cmmd_is_biased_and_ours_is_not`, where the legacy
    estimator's null falls by more than half between these same two sizes.
    """
    m16, s16 = _null_kid(16)
    m256, s256 = _null_kid(256)
    assert abs(m16 - m256) < 3 * (s16 + s256)


def test_kid_is_symmetric():
    a, b = _unit(40, seed=1), _unit(40, loc=0.4, seed=2)
    assert D.kid(a, b) == pytest.approx(D.kid(b, a), rel=1e-9)


def test_kid_detects_a_shifted_distribution():
    a, b = _unit(80, seed=1), _unit(80, loc=1.5, seed=2)
    assert D.kid(a, b) > 10 * abs(D.kid(a, _unit(80, seed=3)))


def test_kid_degenerate_input_is_nan():
    assert np.isnan(D.kid(_unit(1), _unit(40)))


def test_legacy_cmmd_is_biased_and_ours_is_not():
    """Documents in code why `perceptual.cmmd` numbers only hold at fixed n.

    The legacy estimator includes k(x,x)=1 diagonal terms in its within-set
    means, so two disjoint samples from one distribution score well above zero
    and the score shrinks as ~1/n.
    """
    import torch
    from blockgen.eval.perceptual import cmmd

    legacy = []
    for n in (16, 128):
        vals = []
        for s in range(20):
            rng = np.random.default_rng(1000 + s)
            pool = _unit(2 * n, seed=s)
            pi = rng.permutation(len(pool))
            vals.append(cmmd(torch.tensor(pool[pi[:n]]), torch.tensor(pool[pi[n:]])))
        legacy.append(float(np.mean(vals)))

    assert legacy[0] > 0, "legacy null should be positive (it is biased)"
    assert legacy[0] > 2 * legacy[1], "legacy null should shrink as ~1/n"

    # And ours does not: the null is flat in n to within its own noise.
    m16, s16 = _null_kid(16)
    m128, s128 = _null_kid(128)
    assert abs(m16 - m128) < 3 * (s16 + s128)


def test_cmmd_on_identical_input_is_exactly_zero():
    """Guards against writing `cmmd(x, x) > 0` as the bias test -- it is 0.0."""
    import torch
    from blockgen.eval.perceptual import cmmd
    x = torch.tensor(_unit(32))
    assert cmmd(x, x) == pytest.approx(0.0, abs=1e-12)


# --- RBF -------------------------------------------------------------------
def test_median_bandwidth_is_in_the_informative_range():
    """sigma=10 on unit-norm features puts every kernel value in [0.98, 1]."""
    sigma = D.median_bandwidth(_unit(200))
    assert 0.1 < sigma < 3.0


def test_mmd_rbf_null_is_near_zero():
    pool = _unit(200, seed=5)
    sigma = D.median_bandwidth(pool)
    assert abs(D.mmd_rbf(pool[:100], pool[100:], sigma=sigma)) < 5.0


def test_mmd_rbf_separates_distributions():
    a, b = _unit(100, seed=1), _unit(100, loc=1.5, seed=2)
    sigma = D.median_bandwidth(np.concatenate([a, b]))
    assert D.mmd_rbf(a, b, sigma=sigma) > abs(
        D.mmd_rbf(a, _unit(100, seed=3), sigma=sigma))


# --- Frechet ---------------------------------------------------------------
def test_fd_is_zero_ish_against_itself_and_positive_otherwise():
    a = _unit(80, d=16, seed=1)
    assert D.fd(a, a) == pytest.approx(0.0, abs=1e-6)
    assert D.fd(a, _unit(80, d=16, loc=1.0, seed=2)) > 0.1


def test_fd_drifts_with_n_which_is_why_it_is_gated():
    """The measured justification for `comparable_only_at_n`."""
    pool = _unit(600, d=32, seed=7)
    small = D.fd(pool[:32], pool[300:332])
    large = D.fd(pool[:256], pool[300:556])
    assert small > 2 * large


# --- PRDC ------------------------------------------------------------------
def test_prdc_is_near_one_for_real_against_real():
    pool = _unit(300, seed=11)
    r = D.prdc(pool[:150], pool[150:], k=5)
    assert r["coverage"] > 0.85
    assert r["density"] > 0.7
    assert r["precision"] > 0.7 and r["recall"] > 0.7


def test_coverage_detects_mode_collapse():
    """Duplicating one good sample must not look like covering the data."""
    pool = _unit(200, seed=13)
    real, one = pool[:100], pool[100:101]
    collapsed = np.repeat(one, 100, axis=0)
    r = D.prdc(real, collapsed, k=5)
    assert r["coverage"] < 0.1
    assert r["recall"] < 0.1
    # Precision stays high on purpose: every sample IS a real point. That is
    # exactly why precision alone cannot detect collapse.
    assert r["precision"] > 0.5


def test_prdc_degenerate_input_is_nan():
    r = D.prdc(_unit(1), _unit(40))
    assert all(np.isnan(v) for v in r.values())


# --- memorization ----------------------------------------------------------
def test_nn_distance_finds_the_exact_match():
    ref = _unit(50, seed=3)
    d, idx = D.nn_distance(ref[[7, 21]], ref)
    assert idx.tolist() == [7, 21]
    assert np.allclose(d, 0.0, atol=1e-9)


def test_memorization_detector_fires_on_verbatim_copies():
    train = _unit(120, seed=2)
    heldout = _unit(60, seed=9)
    rep = D.memorization_report(train[:30], train, heldout)
    assert rep["dup_rate"] == pytest.approx(1.0)
    assert float(np.mean(rep["nn_percentile"])) < 0.05


def test_memorization_detector_is_quiet_on_genuinely_new_samples():
    train = _unit(120, seed=2)
    heldout = _unit(60, seed=9)
    fresh = _unit(40, seed=17)
    rep = D.memorization_report(fresh, train, heldout)
    assert rep["dup_rate"] < 0.2
    assert float(np.mean(rep["nn_percentile"])) > 0.1


def test_nn_percentile_is_scale_free():
    """A raw cosine is uninterpretable; the percentile is the reportable number."""
    calib = np.linspace(0.0, 1.0, 101)
    pct = D.nn_percentile(np.array([0.0, 0.5, 1.0]), calib)
    assert pct[0] == pytest.approx(0.0)
    assert pct[1] == pytest.approx(0.5, abs=0.02)
    assert pct[2] == pytest.approx(1.0, abs=0.02)


def test_nn_percentile_empty_calibration_is_nan():
    assert np.isnan(D.nn_percentile(np.array([0.3]), np.array([]))).all()
