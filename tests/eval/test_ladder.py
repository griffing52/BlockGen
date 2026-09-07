"""Gate logic for the metric-validation ladder.

The ladder is what stands between a plausible-looking metric and a results
table, so its own logic needs to be exercised directly rather than only through
a 3-minute GPU run. These tests drive `run_ladder` with a stubbed embedder whose
geometry is known, so the expected verdict for each synthetic metric is known
too.
"""

from __future__ import annotations

import numpy as np
import pytest

from blockgen.eval.bench import ladder
from blockgen.eval.bench import features as ft
from blockgen.utils.data import Structure


def _struct(nx=8, ny=6, nz=8, seed=0) -> Structure:
    rng = np.random.default_rng(seed)
    ids = np.zeros((nx, ny, nz), dtype=np.int32)
    ids[:, 0, :] = 4
    ids[0, :, :] = ids[-1, :, :] = 5
    ids[:, :, 0] = ids[:, :, -1] = 98
    ids[:, -1, :] = 5
    ids[rng.integers(1, nx - 1), rng.integers(1, ny - 1), rng.integers(1, nz - 1)] = 35
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


@pytest.fixture
def fake_embed(monkeypatch):
    """Embed each structure as a point whose offset encodes how damaged it is.

    Occupancy fraction drives one coordinate and palette size another, so a
    metric computed on these features behaves monotonically in real damage --
    enough to exercise every gate.
    """
    def _embed(structures, view=ft.ViewConfig(), backbone="dinov2b", device="cuda",
               batch=64, verbose=True):
        rows = []
        for i, s in enumerate(structures):
            c = s.crop_to_non_air()
            occ = c.occupied_mask
            fill = float(occ.sum()) / max(occ.size, 1)
            npal = len(np.unique(c.block_ids[occ])) if occ.any() else 0
            jitter = np.random.default_rng(i).normal(0, 0.01, size=6)
            v = np.array([fill, npal / 10.0, float(max(c.shape)) / 32.0,
                          0.0, 0.0, 0.0]) + jitter
            rows.append(v / max(np.linalg.norm(v), 1e-9))
        f = np.asarray(rows)[:, None, :]
        return np.repeat(f, view.n_views, axis=1)

    monkeypatch.setattr(ft, "embed_views", _embed)
    return _embed


@pytest.fixture
def result(fake_embed):
    ref = [_struct(seed=i) for i in range(40)]
    probe = [_struct(seed=100 + i) for i in range(60)]
    return ladder.run_ladder(ref, probe, n=20, reps=6, include_legacy=False,
                             verbose=False)


# --- structure of the result ----------------------------------------------
def test_every_metric_gets_every_blocking_gate(result):
    expected = {"G1_damage_ordering", "G2_noise_ordering", "G6_invariance",
                "G7_self_consistency", "G8_n_stability"}
    for metric, gates in result.gates.items():
        assert set(gates) == expected, metric


def test_sensitivity_is_recorded_separately_from_validity(result):
    """Power is not validity: a resolution limit must not suppress a value."""
    expected = set(ladder.RENDER_RUNGS.sensitivity)
    assert "S6_resolves_solidify" in expected     # the tier-separating rung
    for metric, flags in result.sensitivity.items():
        assert set(flags) == expected, metric
        assert not (set(flags) & set(result.gates[metric])), metric


def test_a_metric_with_no_sensitivity_can_still_pass():
    r = ladder.LadderResult("b", "v", 8)
    r.gates["m"] = {"G1": True, "G2": True}
    r.sensitivity["m"] = {"S3_resolves_canon16": False}
    assert r.passed("m")
    assert r.resolves("m") == []


def test_every_rung_is_scored(result):
    for metric, scores in result.scores.items():
        for rung in ("real_heldout", "canon16", "canon8", "noise_10",
                     "material_shuffle", "rot90_1", "mirror_x", "chunk_delete_20"):
            assert rung in scores, f"{metric}/{rung}"
            assert np.isfinite(scores[rung]), f"{metric}/{rung}"


def test_noise_floor_is_recorded(result):
    for metric, floor in result.noise.items():
        assert "mean" in floor and "sd" in floor, metric


# --- gate semantics --------------------------------------------------------
def test_failed_gate_names_the_first_failure():
    r = ladder.LadderResult("b", "v", 8)
    r.gates["m"] = {"G1": True, "G2": False, "G3": False}
    assert r.failed_gate("m") == "G2"
    assert not r.passed("m")
    assert r.passing == []


def test_passing_metric_reports_no_failure():
    r = ladder.LadderResult("b", "v", 8)
    r.gates["m"] = {"G1": True, "G2": True}
    assert r.passed("m")
    assert r.failed_gate("m") is None
    assert r.passing == ["m"]


def test_invariance_gate_rejects_a_pose_sensitive_metric(fake_embed, monkeypatch):
    """A metric that penalizes a rotated building measures pose, not quality."""
    ref = [_struct(seed=i) for i in range(30)]
    probe = [_struct(seed=100 + i) for i in range(60)]

    calls = {"n": 0}

    def pose_sensitive(r, p):
        # Returns a large value on every third call, standing in for a metric
        # whose value depends on which rung (and hence which pose) it sees.
        calls["n"] += 1
        return 10.0 if calls["n"] % 3 == 0 else 0.0

    monkeypatch.setattr(ladder, "METRICS", {"pose": pose_sensitive})
    res = ladder.run_ladder(ref, probe, n=20, reps=6, include_legacy=False,
                            verbose=False)
    assert not res.passed("pose")


def test_nan_noise_floor_fails_gates_rather_than_crashing(fake_embed, monkeypatch):
    monkeypatch.setattr(ladder, "METRICS", {"broken": lambda r, p: float("nan")})
    ref = [_struct(seed=i) for i in range(30)]
    probe = [_struct(seed=100 + i) for i in range(60)]
    res = ladder.run_ladder(ref, probe, n=20, reps=6, include_legacy=False,
                            verbose=False)
    assert not res.passed("broken")


# --- bandwidth freezing ----------------------------------------------------
def test_build_metrics_freezes_the_rbf_bandwidth():
    """Per-call bandwidth silently changes the kernel between rungs."""
    a = np.eye(6)[:, :6] + 0.0
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = a[::-1]
    table = ladder.build_metrics(sigma=0.7)
    first = table["mmd_rbf"](a, b)
    second = table["mmd_rbf"](a, b)
    assert first == pytest.approx(second)
    other = ladder.build_metrics(sigma=2.5)["mmd_rbf"](a, b)
    assert first != pytest.approx(other), "sigma must actually be applied"


# --- persistence -----------------------------------------------------------
def test_save_and_load_round_trip(result, tmp_path):
    path = ladder.save(result, root=tmp_path)
    assert path.exists()
    back = ladder.load(result.backbone, result.view, root=tmp_path)
    assert back is not None
    assert back.gates == result.gates
    assert back.passing == result.passing


def test_load_missing_returns_none(tmp_path):
    assert ladder.load("nope", "nope", root=tmp_path) is None


def test_render_marks_pass_and_fail(result):
    text = ladder.render(result)
    assert "PASS" in text or "FAIL" in text
    assert "real_heldout" in text


# --- the two noise-floor corrections ---------------------------------------
def test_noise_floor_corrects_for_overlapping_draws():
    """Repeated n-subsets of an N-build pool share members, so their spread is
    the spread of a subsample, not of an independent draw. Dividing out the
    finite-population factor recovers the quantity every gate is stated in."""
    rng = np.random.default_rng(0)
    pool = rng.normal(size=(100, 4))
    ref = rng.normal(size=(40, 4))
    fn = lambda r, p: float(p.mean() - r.mean())

    _, sd_small_n, _ = ladder.noise_floor(fn, ref, pool, 10, reps=400,
                                          rng=np.random.default_rng(1))
    _, sd_big_n, _ = ladder.noise_floor(fn, ref, pool, 90, reps=400,
                                        rng=np.random.default_rng(1))
    # For a sample mean the corrected spread is sigma/sqrt(n), so drawing 9x
    # more builds must cut it by about 3 -- uncorrected, the n=90 draws would
    # nearly all be the same 90 builds and the spread would collapse further.
    assert sd_small_n / sd_big_n == pytest.approx(3.0, rel=0.25)


def test_noise_floor_reports_a_separate_standard_error_for_the_mean():
    rng = np.random.default_rng(0)
    pool = rng.normal(size=(80, 4))
    ref = rng.normal(size=(40, 4))
    fn = lambda r, p: float(p.mean())
    _, sd, sem = ladder.noise_floor(fn, ref, pool, 20, reps=100, rng=rng)
    # the mean of 100 draws is pinned far more precisely than one draw is spread
    assert 0 < sem < sd


def test_g8_rejects_an_estimator_whose_null_mean_decays_with_n(fake_embed,
                                                              monkeypatch):
    """The `legacy_cmmd` failure mode, in miniature: a statistic whose expected
    value under the null halves every time n doubles. The earlier form of G8
    compared this drift against the spread of a *single* draw -- larger by
    sqrt(reps) -- and passed it."""
    ref = [_struct(seed=i) for i in range(24)]
    probe = [_struct(seed=100 + i) for i in range(60)]
    biased = {"one_over_n": lambda r, p: 1.0 / len(p)}
    monkeypatch.setattr(ladder, "METRICS", biased)
    monkeypatch.setattr(ladder, "build_metrics", lambda sigma: dict(biased))
    res = ladder.run_ladder(ref, probe, n=20, reps=24, include_legacy=False,
                            verbose=False)
    assert res.gates["one_over_n"]["G8_n_stability"] is False


def test_g8_accepts_an_n_stable_estimator(fake_embed, monkeypatch):
    ref = [_struct(seed=i) for i in range(24)]
    probe = [_struct(seed=100 + i) for i in range(60)]
    stable = {"const": lambda r, p: float(p.mean() - r.mean())}
    monkeypatch.setattr(ladder, "METRICS", stable)
    monkeypatch.setattr(ladder, "build_metrics", lambda sigma: dict(stable))
    res = ladder.run_ladder(ref, probe, n=20, reps=24, include_legacy=False,
                            verbose=False)
    assert res.gates["const"]["G8_n_stability"] is True


# --- rung specifications ---------------------------------------------------
def test_geometry_rungs_swap_the_noise_family_and_widen_invariance():
    """An occupancy metric is exactly blind to retyping blocks, so gating it on
    `noise_*` would fail it for doing what it was built to do. Material
    corruptions become invariances it must *not* move under."""
    assert ladder.GEOMETRY_RUNGS.ordered_noise == ("occ_noise_1", "occ_noise_5",
                                                   "occ_noise_10")
    assert "material_shuffle" in ladder.GEOMETRY_RUNGS.invariant
    assert "material_shuffle" not in ladder.RENDER_RUNGS.invariant
    assert "noise_1" not in ladder.GEOMETRY_RUNGS.ordered_noise


def test_solidify_is_a_sensitivity_rung_for_both_families():
    """It is the rung that separates the two tiers, so both must report on it."""
    assert "S6_resolves_solidify" in ladder.RENDER_RUNGS.sensitivity
    assert "S6_resolves_solidify" in ladder.GEOMETRY_RUNGS.sensitivity


def test_geometry_ladder_runs_without_a_gpu():
    from blockgen.eval.bench import ladder as ld
    ref = [_struct(seed=i) for i in range(20)]
    probe = [_struct(seed=100 + i) for i in range(40)]
    res = ld.run_geometry_ladder(ref, probe, n=16, reps=6, verbose=False)
    assert set(res.scores) == {"geom_kid", "geom_mmd_rbf"}
    assert res.backbone == ld.GEOM_BACKBONE
    for m in res.scores:
        # blind to material by construction: these must be bit-identical
        assert res.scores[m]["material_shuffle"] == res.scores[m]["real_heldout"]
        assert res.scores[m]["monochrome"] == res.scores[m]["real_heldout"]
        assert res.scores[m]["rot90_1"] == res.scores[m]["real_heldout"]
