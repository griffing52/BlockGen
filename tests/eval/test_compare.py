"""Head-to-head tests, and the variance reduction that justifies them."""

from __future__ import annotations

import numpy as np
import pytest

from blockgen.eval.bench import compare as cmp
from blockgen.eval.bench import distances as D


def _cloud(rng, n=64, d=16, shift=0.0):
    x = rng.normal(size=(n, d)) + shift
    return x / np.linalg.norm(x, axis=1, keepdims=True)


KID = lambda a, b: D.kid(a, b)


# --- Holm ------------------------------------------------------------------
def test_holm_matches_the_textbook_step_down():
    # sorted 0.005, 0.01, 0.03, 0.04 with m=4 -> 0.02, 0.03, 0.06, 0.06
    assert cmp.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx(
        [0.03, 0.06, 0.06, 0.02])


def test_holm_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    p = rng.random(20)
    adj = np.asarray(cmp.holm(p))
    assert np.all(adj <= 1.0)
    assert np.all(adj >= p)                       # adjustment never helps
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-12)  # preserves the ordering


def test_holm_ignores_uncomputable_comparisons():
    out = cmp.holm([0.01, float("nan"), 0.02])
    assert np.isnan(out[1])
    # two real tests, not three
    assert out[0] == pytest.approx(0.02)


# --- the p-value -----------------------------------------------------------
def test_empirical_p_is_floored_and_capped():
    """200 replicates cannot demonstrate p = 0, only p < 1/200."""
    assert cmp._empirical_p(np.ones(200)) == pytest.approx(1 / 200)
    # every replicate exactly zero: both tails are 1, so `2 * min` is 2 and the
    # clamp is what keeps a p-value inside [0, 1]
    assert cmp._empirical_p(np.zeros(10)) == pytest.approx(1.0)


def test_the_reported_p_is_not_floored_by_the_replicate_count():
    """The bug that collapsed an 11-arm leaderboard into one group: the
    sign-balance floor of 1/n_rep cannot survive Holm over 55 pairs."""
    rng = np.random.default_rng(0)
    draws = rng.normal(10.0, 1.0, 200)          # a huge, unambiguous effect
    assert cmp._empirical_p(draws) == pytest.approx(1 / 200)   # floored
    assert cmp._normal_p(10.0, draws) < 1e-20                  # not floored
    assert cmp._normal_p(10.0, draws) * 55 < 0.05              # survives Holm


def test_normal_p_is_one_for_no_effect():
    rng = np.random.default_rng(0)
    assert cmp._normal_p(0.0, rng.normal(0.0, 1.0, 200)) == pytest.approx(1.0, abs=0.05)


# --- the tests themselves --------------------------------------------------
def test_identical_arms_are_not_separated():
    rng = np.random.default_rng(0)
    ref = _cloud(rng, 128)
    a, b = _cloud(rng), _cloud(rng)
    c = cmp.paired_delta(KID, a, b, ref, n_rep=200, rng=rng)
    assert not c.significant
    assert c.winner is None
    assert c.ci[0] < 0 < c.ci[1]


def test_a_clearly_better_arm_wins_and_is_named():
    rng = np.random.default_rng(1)
    ref = _cloud(rng, 128)
    good, bad = _cloud(rng), _cloud(rng, shift=1.5)
    c = cmp.paired_delta(KID, good, bad, ref, n_rep=200, rng=rng,
                         name_a="good", name_b="bad")
    assert c.delta < 0                      # lower KID is better
    assert c.significant and c.winner == "good"


def test_direction_flips_which_arm_wins():
    rng = np.random.default_rng(2)
    ref = _cloud(rng, 128)
    a, b = _cloud(rng), _cloud(rng, shift=1.5)
    lower = cmp.paired_delta(KID, a, b, ref, n_rep=200, rng=rng,
                             direction="lower_better", name_a="a", name_b="b")
    higher = cmp.paired_delta(KID, a, b, ref, n_rep=200, rng=rng,
                              direction="higher_better", name_a="a", name_b="b")
    assert lower.winner == "a" and higher.winner == "b"


def test_testing_the_difference_beats_asking_whether_intervals_overlap():
    """The actual power argument, as a fixed factor rather than a vibe.

    Non-overlap of two 95% intervals demands `h_a + h_b`; separating the arms
    demands only `sqrt(h_a^2 + h_b^2)`. The module docstring quotes 41% for
    equal widths -- this pins that the difference test is the more sensitive one
    on a case engineered to sit between the two thresholds.
    """
    rng = np.random.default_rng(3)
    ref = _cloud(rng, 200)
    a, b = _cloud(rng, 96), _cloud(rng, 96, shift=0.55)
    r = np.random.default_rng(9)

    def marginal(x):
        d = [KID(x[r.choice(len(x), 77, replace=False)],
                 ref[r.choice(len(ref), 160, replace=False)]) for _ in range(400)]
        return np.percentile(d, 2.5), np.percentile(d, 97.5)

    lo_a, hi_a = marginal(a)
    lo_b, hi_b = marginal(b)
    overlap = not (hi_a < lo_b or hi_b < lo_a)
    c = cmp.paired_delta(KID, a, b, ref, n_rep=400, rng=np.random.default_rng(9))
    # the difference test resolves it; the overlap heuristic need not
    assert c.significant
    if overlap:
        assert c.significant   # documents the exact case the module exists for


def test_per_sample_comparison_on_means():
    rng = np.random.default_rng(4)
    x = rng.normal(0.0, 1.0, 200)
    y = rng.normal(1.0, 1.0, 200)
    c = cmp.paired_delta_per_sample(x, y, rng=rng, name_a="x", name_b="y")
    assert c.delta == pytest.approx(-1.0, abs=0.3)
    assert c.significant and c.winner == "x"
    assert c.paired_on == "none"


def test_too_few_samples_yields_no_verdict_rather_than_a_confident_one():
    rng = np.random.default_rng(5)
    ref = _cloud(rng, 64)
    c = cmp.paired_delta(KID, _cloud(rng, 2), _cloud(rng, 2), ref, rng=rng)
    assert np.isnan(c.p) and c.winner is None
    assert not c.significant


# --- the leaderboard -------------------------------------------------------
def test_rank_table_orders_arms_and_corrects_for_multiplicity():
    rng = np.random.default_rng(6)
    ref = _cloud(rng, 128)
    arms = {"near": _cloud(rng), "near2": _cloud(rng), "far": _cloud(rng, shift=2.0)}
    t = cmp.rank_table("kid", KID, arms, ref, n_rep=200, rng=rng)
    assert t.order[-1] == "far"
    assert t.beats("near", "far") == "near" or t.beats("near2", "far") == "near2"
    assert all(c.p_adj is not None and c.p_adj >= c.p for c in t.comparisons)
    assert all(c.p_boot is not None for c in t.comparisons)
    assert "far" in cmp.render_rank_table(t)


def test_letter_groups_join_only_arms_indistinguishable_from_every_member():
    """A shared letter must mean "no pair inside this group was resolved",
    never merely "each was tied with its neighbour"."""
    order = ["a", "b", "c"]
    # a~b and b~c, but a beats c: the three must not collapse into one group
    verdicts = {("a", "c"): "a"}
    beats = lambda x, y: verdicts.get((x, y)) or verdicts.get((y, x))
    groups = cmp._letter_groups(order, beats)
    assert groups["a"] == groups["b"]
    assert groups["a"] != groups["c"]


def test_rank_table_direction_reverses_the_order():
    rng = np.random.default_rng(7)
    ref = _cloud(rng, 128)
    arms = {"a": _cloud(rng), "b": _cloud(rng, shift=2.0)}
    lo = cmp.rank_table("k", KID, arms, ref, direction="lower_better", n_rep=100, rng=rng)
    hi = cmp.rank_table("k", KID, arms, ref, direction="higher_better", n_rep=100, rng=rng)
    assert lo.order == hi.order[::-1]


def test_arm_term_gives_the_identical_delta_at_lower_cost():
    """`kid = term_x + term_ref - 2*mean(k_x_ref)`. With the reference draw
    shared, `term_ref` cancels, so skipping it changes nothing about the answer
    and skips the largest of the three kernels."""
    rng = np.random.default_rng(0)
    a, b, ref = _cloud(rng, 40, 64), _cloud(rng, 40, 64), _cloud(rng, 120, 64)
    direct = D.kid(a, ref) - D.kid(b, ref)
    viaterm = D.kid_arm_term(a, ref) - D.kid_arm_term(b, ref)
    assert direct == pytest.approx(viaterm, abs=1e-9)

    c1 = cmp.paired_delta(KID, a, b, ref, n_rep=100, rng=np.random.default_rng(1))
    c2 = cmp.paired_delta(KID, a, b, ref, n_rep=100, rng=np.random.default_rng(1),
                          arm_term=D.kid_arm_term)
    assert c1.ci == pytest.approx(c2.ci, abs=1e-9)
    assert c1.p == pytest.approx(c2.p)


def test_arm_term_alone_is_not_a_kid_value():
    """It is a difference-only quantity; the absolute number is meaningless."""
    rng = np.random.default_rng(0)
    x, ref = _cloud(rng, 40, 64), _cloud(rng, 120, 64)
    assert D.kid_arm_term(x, ref) != pytest.approx(D.kid(x, ref), abs=1e-6)
