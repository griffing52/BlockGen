"""BlockScore: the aggregate, and the strategies it has to reject.

Every check here corresponds to a way of winning a leaderboard without building
anything good. They are the reason the aggregate is a max over pillars with
disqualification gates rather than a weighted sum.
"""

from __future__ import annotations

import numpy as np
import pytest

from blockgen.eval.bench import composite as comp
from blockgen.eval.bench import scorecard as sc


def _arm(n=128, geom=0.0, dino=0.0, palette=0.0, dup=0.0, div=0.6, pct=0.5,
         ci=0.01, coh=0.0):
    """A synthetic scored arm in the shape the runner produces."""
    def m(v, d="lower_better", w=ci):
        return sc.metric(v, (v - w, v + w), d)
    return {
        "meta": {"n": n},
        "realism": {"geom_kid": m(geom), "mv_dino_kid": m(dino)},
        "dataset_stats": {"palette_jsd_exact": m(palette)},
        "novelty": {"voxel_dup_rate": m(dup),
                    "voxel_diversity": m(div, "higher_better"),
                    "dino_nn_percentile": m(pct, "higher_better")},
        "coherence": {"lcc_ratio": {"w1_norm": coh}},
    }


def _card(arms):
    card = sc.Scorecard(context={})
    for k, v in arms.items():
        card.add_arm(k, v)
    return card


# --- calibration -----------------------------------------------------------
def test_units_are_real_sample_spreads():
    cal = comp.Calibration.from_arm(_arm(), "real_test")
    # spread = half-width / 1.96; a value one half-width above the floor is
    # therefore 1.96 spreads away
    assert cal.z("realism.geom_kid", 0.01) == pytest.approx(1.96, rel=1e-3)


def test_measured_noise_floor_overrides_the_bootstrap_interval():
    """The control arm's own interval is not the spread of a *fresh* real
    sample, and the axis is labelled with the latter."""
    cal = comp.Calibration.from_arm(_arm(), "real_test",
                                    noise_sd={"realism.geom_kid": 0.1})
    assert cal.z("realism.geom_kid", 0.1) == pytest.approx(1.0)
    assert "realism.geom_kid" in cal.measured


def test_a_degenerate_control_interval_falls_back_to_a_minimum_spread():
    """The bug that silently disabled the memorization gate: duplicate rate is
    exactly 0 on real data, its interval has zero width, and `z` was NaN."""
    arm = _arm()
    arm["novelty"]["voxel_dup_rate"] = sc.metric(0.0, (0.0, 0.0), "lower_better")
    cal = comp.Calibration.from_arm(arm, "real_test")
    assert np.isnan(cal.z("novelty.voxel_dup_rate", 1.0))          # no scale
    assert cal.z("novelty.voxel_dup_rate", 1.0, min_spread=0.02) == 50.0


# --- the two rules ---------------------------------------------------------
def test_score_is_the_worst_pillar_never_the_mean():
    cal = comp.Calibration.from_arm(_arm(), "real_test")
    row = comp.score_arm("x", _arm(geom=0.10, dino=0.0, palette=0.0), cal)
    assert row.worst == "geometry"
    assert row.score == pytest.approx(row.pillars["geometry"])
    assert row.score > np.mean(list(row.pillars.values()))


def test_an_excellent_pillar_cannot_buy_a_terrible_one():
    """A weighted sum would let this arm outrank the honest one."""
    cal = comp.Calibration.from_arm(_arm(), "real_test")
    honest = comp.score_arm("honest", _arm(geom=0.02, dino=0.02), cal)
    lopsided = comp.score_arm("lopsided", _arm(geom=0.10, dino=-0.05), cal)
    assert lopsided.score > honest.score


def test_memorization_is_disqualified_not_merely_penalized():
    cal = comp.Calibration.from_arm(_arm(), "real_test")
    row = comp.score_arm("copier", _arm(geom=0.0, dino=0.0, dup=1.0), cal)
    assert row.status == "DQ" and row.score is None
    assert "memorization" in row.disqualified


def test_mode_collapse_is_disqualified():
    cal = comp.Calibration.from_arm(_arm(), "real_test")
    row = comp.score_arm("collapsed", _arm(div=0.0), cal)
    assert row.status == "DQ" and "mode collapse" in row.disqualified


def test_an_honest_corruption_is_ranked_not_disqualified():
    """`real@solidify` legitimately lowers diversity -- builds really do become
    more alike -- and at a 3-spread bar it was disqualified as mode collapse,
    hiding the geometry pillar that was the actual finding."""
    cal = comp.Calibration.from_arm(_arm(), "real_test")
    row = comp.score_arm("solidified", _arm(geom=0.5, div=0.6 - 4 * 0.01 / 1.96), cal)
    assert row.status == "ok"
    assert row.worst == "geometry"


def test_too_few_samples_is_unranked_rather_than_scored():
    cal = comp.Calibration.from_arm(_arm(), "real_test")
    row = comp.score_arm("tiny", _arm(n=8, geom=0.0), cal, min_n=16)
    assert row.status == "unranked" and "min_n" in row.unranked
    assert row.pillars            # the breakdown is still reported


def test_missing_pillars_are_named_not_silently_dropped():
    cal = comp.Calibration.from_arm(_arm(), "real_test")
    arm = _arm()
    arm["realism"].pop("mv_dino_kid")
    row = comp.score_arm("fast_only", arm, cal)
    assert "appearance" in row.missing


# --- the leaderboard and its self-validation -------------------------------
def test_leaderboard_requires_a_real_control():
    with pytest.raises(ValueError, match="calibrated"):
        comp.leaderboard(_card({"someone": _arm()}))


def test_leaderboard_orders_by_score_and_pushes_dq_to_the_bottom():
    card = _card({"real_test": _arm(),
                  "good": _arm(geom=0.01),
                  "bad": _arm(geom=0.20),
                  "cheat": _arm(geom=0.0, dup=1.0)})
    rows = comp.leaderboard(card)
    assert [r.arm for r in rows[:3]] == ["real_test", "good", "bad"]
    assert rows[-1].arm == "cheat" and rows[-1].status == "DQ"
    assert "DQ" in comp.render_leaderboard(rows)


def test_self_validation_passes_on_a_well_behaved_run():
    card = _card({"real_test": _arm(),
                  "real@canon16": _arm(geom=0.05),
                  "real@canon8": _arm(geom=0.20),
                  "real@solidify": _arm(geom=0.30),
                  "real@monochrome": _arm(palette=0.5),
                  "train_verbatim": _arm(dup=1.0),
                  "real@single_mode": _arm(div=0.0)})
    checks = comp.validate(comp.leaderboard(card))
    assert checks and all(checks.values()), checks


def test_self_validation_fails_when_a_cheat_is_not_caught():
    """The aggregate is held to the same standard as the metrics: it must be
    shown to reject what it claims to reject."""
    card = _card({"real_test": _arm(), "train_verbatim": _arm(dup=0.0)})
    checks = comp.validate(comp.leaderboard(card))
    assert checks["C2_memorization_disqualified"] is False


def test_material_shuffle_is_not_required_of_a_tier_that_cannot_see_it():
    """FAST-only runs have no appearance pillar, and both the palette and the
    geometry descriptor are blind to a material permutation by construction."""
    fast_only = _arm()
    fast_only["realism"].pop("mv_dino_kid")
    shuffled = _arm()
    shuffled["realism"].pop("mv_dino_kid")
    card = _card({"real_test": fast_only, "real_shuffled_materials": shuffled})
    checks = comp.validate(comp.leaderboard(card))
    assert "C3_real_shuffled_materials_worse_than_real" not in checks
