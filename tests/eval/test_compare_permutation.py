"""The permutation test itself: what it computes, and that the fast path is it.

Calibration lives in `test_compare_calibration.py`; this file is about mechanics.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from blockgen.eval.bench import compare as cmp
from blockgen.eval.bench import distances as D


def _cloud(rng, n=40, d=12, shift=0.0):
    x = rng.normal(size=(n, d)) + shift
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _offdiag_mean(k):
    m = len(k)
    return (k.sum() - np.trace(k)) / (m * (m - 1))


def test_kid_gram_is_the_kernel_kid_is_built_from():
    """The contract the fast path stands on: slice the Gram, recover the KID."""
    rng = np.random.default_rng(0)
    x, y = _cloud(rng, 30, 16), _cloud(rng, 45, 16)
    via_gram = (_offdiag_mean(D.kid_gram(x, x)) + _offdiag_mean(D.kid_gram(y, y))
                - 2.0 * D.kid_gram(x, y).mean())
    assert via_gram == pytest.approx(D.kid(x, y), rel=1e-12)


def test_the_gram_fast_path_computes_the_same_statistic_as_the_slow_one():
    """Same index split, same number -- the speedup must be only a speedup."""
    rng = np.random.default_rng(1)
    a, b, ref = _cloud(rng, 24), _cloud(rng, 24), _cloud(rng, 60)
    obs_k, draws_k = cmp._perm_draws_kernel(D.kid_gram, a, b, ref, 64,
                                            np.random.default_rng(5))
    obs_g, draws_g = cmp._perm_draws_generic(D.kid_arm_term, a, b, ref, 64,
                                             np.random.default_rng(5))
    assert obs_k == pytest.approx(obs_g, rel=1e-10)
    # The observed statistic IS the reported delta, which is what lets the two
    # be compared with a tolerance rather than a hope.
    assert obs_k == pytest.approx(D.kid(a, ref) - D.kid(b, ref), rel=1e-9)
    # The two paths draw their permutations from the same generator differently,
    # so the replicates are not row-for-row equal; their distributions are.
    assert np.mean(draws_k) == pytest.approx(np.mean(draws_g), abs=3 * np.std(draws_g))
    assert np.std(draws_k) == pytest.approx(np.std(draws_g), rel=0.4)


def test_the_replicates_are_exactly_the_statistic_on_a_relabelled_split():
    """Spot-check one replicate against a hand-built split, with no Gram."""
    rng = np.random.default_rng(2)
    a, b, ref = _cloud(rng, 16), _cloud(rng, 20), _cloud(rng, 50)
    pooled = np.concatenate([a, b], 0)
    idx = np.random.default_rng(3).permutation(len(pooled))
    want = (D.kid(pooled[idx[:16]], ref) - D.kid(pooled[idx[16:]], ref))
    kmat = D.kid_gram(pooled, pooled)
    kref = D.kid_gram(pooled, ref).mean(1)
    mask = np.zeros((1, len(pooled)))
    mask[0, idx[:16]] = 1.0

    def term(m, size):
        return (((m @ kmat) * m).sum(1) - m @ np.diag(kmat)) / (size * (size - 1)) \
            - 2.0 * (m @ kref) / size

    got = float(term(mask, 16)[0] - term(1.0 - mask, 20)[0])
    assert got == pytest.approx(want, rel=1e-9)


def test_a_mismatched_kernel_fails_loudly_rather_than_testing_something_else():
    rng = np.random.default_rng(4)
    a, b, ref = _cloud(rng), _cloud(rng), _cloud(rng, 80)
    with pytest.raises(ValueError, match="does not reproduce"):
        cmp.paired_delta(lambda x, y: D.kid(x, y), a, b, ref, n_rep=20, n_perm=32,
                         kernel=lambda x, y: D.kid_gram(x, y, degree=2),
                         rng=np.random.default_rng(0))


def test_the_p_value_is_floored_by_the_permutation_count_and_says_so():
    rng = np.random.default_rng(5)
    a, b, ref = _cloud(rng, 64), _cloud(rng, 64, shift=2.0), _cloud(rng, 128)
    c = cmp.paired_delta(D.kid, a, b, ref, n_rep=50, n_perm=200,
                         kernel=D.kid_gram, rng=np.random.default_rng(6))
    assert c.p == pytest.approx(1 / 201)          # no replicate beat the observed
    assert c.n_perm == 200 and c.method == "permutation"
    # ... and the unfloored companion still says how far past the floor it is.
    assert c.p_normal < 1e-6


def test_the_monte_carlo_p_tracks_the_exhaustive_one():
    """With 7 vs 7 builds there are 3432 splits; enumerate them all and compare."""
    rng = np.random.default_rng(7)
    a, b, ref = _cloud(rng, 7, 8), _cloud(rng, 7, 8, shift=0.8), _cloud(rng, 30, 8)
    pooled = np.concatenate([a, b], 0)
    obs = D.kid(a, ref) - D.kid(b, ref)
    exact = []
    for combo in itertools.combinations(range(14), 7):
        take = list(combo)
        rest = [i for i in range(14) if i not in combo]
        exact.append(D.kid(pooled[take], ref) - D.kid(pooled[rest], ref))
    p_exact = float(np.mean(np.abs(exact) >= abs(obs) - 1e-12))
    _, draws = cmp._perm_draws_kernel(D.kid_gram, a, b, ref, 20_000,
                                      np.random.default_rng(8))
    p_mc = cmp._permutation_p(obs, draws)
    assert p_mc == pytest.approx(p_exact, abs=0.02)


def test_the_legacy_method_reproduces_the_old_number_exactly():
    """`method="subsample"` must be the pre-fix code path, not an approximation.

    It is what an old card is re-derived with, so it reproduces the studentized
    p from the spread of the subsample draws -- and the interval, which both
    methods share, must not depend on the choice at all.
    """
    rng_args = dict(n_rep=64, rng=np.random.default_rng(9))
    rng = np.random.default_rng(10)
    a, b, ref = _cloud(rng, 48), _cloud(rng, 48, shift=0.4), _cloud(rng, 96)
    legacy = cmp.paired_delta(D.kid, a, b, ref, method="subsample", **rng_args)
    perm = cmp.paired_delta(D.kid, a, b, ref, method="permutation", n_perm=500,
                            kernel=D.kid_gram, n_rep=64,
                            rng=np.random.default_rng(9))
    assert legacy.ci == pytest.approx(perm.ci)        # same draws, drawn first
    assert legacy.p_boot == perm.p_boot
    assert legacy.p == legacy.p_normal                # studentized, by definition
    assert legacy.method == "subsample" and legacy.n_perm == 0
    # The whole point: the legacy p is the smaller one, because its se is too
    # small by the factor the module docstring derives.
    assert legacy.se_null < perm.se_null and legacy.p < perm.p


def test_an_unknown_method_is_refused():
    rng = np.random.default_rng(11)
    with pytest.raises(ValueError, match="unknown method"):
        cmp.paired_delta(D.kid, _cloud(rng), _cloud(rng), _cloud(rng, 80),
                         n_rep=10, method="bootstrap")
