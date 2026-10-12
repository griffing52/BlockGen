"""The view-protocol ablation: does it decide the question it claims to decide?

`scripts/ablate_view_protocol.py` exists to adjudicate between "DINOv2 at 4 views
is blind to interiors" and "render-and-embed metrics are blind to interiors".
The machinery that must be right for its answer to mean anything is small and
testable without a GPU:

* the view budgets must **nest**, or "16 views sees more" could be 16 views
  seeing somewhere else;
* the separation statistic must read ~0 when two samples come from one
  distribution and large when they do not;
* `verdict` must actually flip. A verdict function that can only print the
  thesis is not a result, and this is where that would hide.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ablate_view_protocol",
    Path(__file__).resolve().parents[2] / "scripts" / "ablate_view_protocol.py")
abl = importlib.util.module_from_spec(_SPEC)
# Registered before execution because `@dataclass` resolves annotations through
# `sys.modules[cls.__module__]`, which does not exist yet for a path import.
sys.modules[_SPEC.name] = abl
_SPEC.loader.exec_module(abl)


# --- view budgets ----------------------------------------------------------
def test_view_budgets_have_the_advertised_sizes():
    assert [len(abl.view_budget(n)) for n in (4, 8, 16)] == [4, 8, 16]


def test_view_budgets_nest():
    small, mid, big = (set(abl.view_budget(n)) for n in (4, 8, 16))
    assert small < mid < big


def test_view_budgets_avoid_the_degenerate_pole():
    """`renderer.textured._look_at` divides by zero when the view direction is
    parallel to the up vector, which is exactly elevation +/-90."""
    for n in (4, 8, 16):
        assert all(abs(elev) < 89.0 for _, elev in abl.view_budget(n))


# --- conditions ------------------------------------------------------------
def test_every_condition_is_constructible_and_well_formed():
    conds = abl.conditions()
    names = [c.name for c in conds]
    assert len(names) == len(set(names))
    for c in conds:
        assert not c.view.cuts or len(c.view.cuts) == len(c.view.views)


def test_the_ablation_covers_all_four_axes():
    fams = {c.family for c in abl.conditions()}
    assert {"baseline", "A encoder", "B views", "C projection", "D cutaway"} <= fams


def test_every_quadrant_cut_faces_its_own_camera():
    """A cutaway hidden behind a wall is not a cutaway. The quadrant removed for
    a camera at azimuth a must be the quadrant that camera looks at."""
    from blockgen.eval.bench import features as ft

    for azim, _ in ft.DEFAULT_VIEWS:
        spec = abl._QUAD_FOR_AZIM[azim]
        _, xside, zside = spec.split(":")
        rad = np.radians(azim)
        assert (xside == "hi") == (np.sin(rad) > 0)
        assert (zside == "hi") == (np.cos(rad) > 0)


# --- the separation statistic ----------------------------------------------
def _cluster(n, d=32, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    x[:, 0] += shift
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def test_separation_is_near_zero_when_the_probe_is_just_more_real_data():
    ref = _cluster(200, seed=1)
    pool = _cluster(160, seed=2)
    sol = _cluster(64, seed=3)            # same distribution, different draw
    sc = abl.score(ref, pool, sol, n=64, reps=16, ci_reps=32)
    assert abs(sc.sep) < 3.0
    assert sc.sep_ci[0] < 0 < sc.sep_ci[1]
    assert sc.self_consistent


def test_separation_is_large_and_signed_when_the_probe_is_shifted():
    ref = _cluster(200, seed=1)
    pool = _cluster(160, seed=2)
    sol = _cluster(64, shift=3.0, seed=3)
    sc = abl.score(ref, pool, sol, n=64, reps=16, ci_reps=32)
    assert sc.solidify > sc.floor and sc.sd > 0
    assert sc.sep > 10.0 and sc.sep_ci[0] > 0


# --- the verdict must be able to go either way -----------------------------
def _row(name, family, sep):
    ci = (sep - 0.5, sep + 0.5)
    return abl.Row(name, family, "dinov2b", "v", 768, 0.0, (0.0, 0.0),
                   sep * 0.01, (0.0, 0.0), 0.01, sep, ci)


def test_an_atypical_floor_draw_is_flagged():
    """A floor that is not a typical real draw invalidates everything measured
    against it, so it has to surface rather than silently shift a separation."""
    ok = abl.Row("a", "A encoder", "d", "v", 768, 0.010, (0.0, 0.0), 0.011,
                 (0.0, 0.0), 0.002, 0.5, (0.0, 1.0), floor_mean=0.0101)
    bad = abl.Row("a", "A encoder", "d", "v", 768, 0.030, (0.0, 0.0), 0.031,
                  (0.0, 0.0), 0.002, 0.5, (0.0, 1.0), floor_mean=0.0101)
    assert ok.self_consistent and not bad.self_consistent
    assert ok.to_json()["floor_is_a_typical_real_draw"] is True


def test_row_sees_needs_three_floors_and_an_interval_clear_of_zero():
    assert _row("a", "A encoder", 6.0).sees
    assert not _row("a", "A encoder", 2.0).sees
    wide = abl.Row("a", "A encoder", "d", "v", 768, 0.0, (0.0, 0.0), 0.0,
                   (0.0, 0.0), 0.01, 5.0, (-1.0, 11.0))
    assert not wide.sees


def test_verdict_says_view_protocol_when_only_the_cutaway_sees_it():
    rows = [_row("paper", "baseline", 0.3), _row("clip", "A encoder", 1.1),
            _row("16 views", "B views", 0.8), _row("cut", "D cutaway", 40.0)]
    v = abl.verdict(rows)
    assert "view protocol" in v and "weaken" not in v


def test_verdict_weakens_the_claim_when_an_exterior_condition_sees_it():
    rows = [_row("paper", "baseline", 0.3), _row("16 views", "B views", 9.0),
            _row("cut", "D cutaway", 40.0)]
    v = abl.verdict(rows)
    assert "must weaken" in v and "under-viewed" in v
    assert "16 views" in v


def test_verdict_refuses_to_conclude_without_a_working_positive_control():
    rows = [_row("paper", "baseline", 0.3), _row("cut", "D cutaway", 0.6)]
    v = abl.verdict(rows)
    assert "Inconclusive" in v


def test_the_reference_metric_row_is_not_counted_as_a_render_condition():
    """`geom_kid` separates `solidify` enormously. If the verdict logic treated
    it as an exterior render condition it would report the opposite conclusion."""
    rows = [_row("paper", "baseline", 0.3), _row("cut", "D cutaway", 40.0),
            _row("geom_kid", "reference metric", 200.0)]
    assert "view protocol" in abl.verdict(rows)


def test_table_renders_one_row_per_condition():
    rows = [_row("paper", "baseline", 0.3), _row("cut", "D cutaway", 40.0)]
    body = abl.table(rows).splitlines()
    assert len(body) == 4                      # header, separator, two rows
    assert "**YES**" in body[-1] and "| no |" in body[-2]


# --- the unpooled arm ------------------------------------------------------
def _views(n, v=4, d=32, shift_view=None, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, v, d))
    if shift_view is not None:
        x[:, shift_view, 0] += shift
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def test_best_single_view_finds_a_difference_that_mean_pooling_could_dilute():
    """The point of the arm: a signal present in one view only must not be able
    to hide behind the average."""
    from blockgen.eval.bench import features as ft

    ref = _views(200, seed=1)
    pool = _views(160, seed=2)
    sol = _views(64, shift_view=2, shift=2.0, seed=3)
    row = abl.best_single_view_row(ref, pool, sol, n=64, reps=16, ci_reps=32)
    assert row.sees and row.family == "E pooling"
    pooled = abl.score(ft.pooled(ref, "mean"), ft.pooled(pool, "mean"),
                       ft.pooled(sol, "mean"), n=64, reps=16, ci_reps=32)
    assert row.sep > pooled.sep


def test_best_single_view_stays_at_the_floor_when_no_view_carries_anything():
    ref, pool, sol = _views(200, seed=1), _views(160, seed=2), _views(64, seed=3)
    assert not abl.best_single_view_row(ref, pool, sol, 64, 16, 32).sees


def test_the_unpooled_arm_counts_as_an_exterior_condition():
    """If `E pooling` were excluded from the exterior set, a positive there would
    not weaken the verdict -- which is exactly the failure this row exists to
    catch."""
    rows = [_row("paper", "baseline", 0.3), _row("single view", "E pooling", 12.0),
            _row("cut", "D cutaway", 40.0)]
    assert "must weaken" in abl.verdict(rows)
