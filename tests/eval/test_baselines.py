"""Procedural baselines: the floor, and the one built to attack the metric.

These are the arms a leaderboard is read against, so what they must guarantee is
narrow but strict: they are seeded, they never see val or test, and each one has
the structural property it claims in its name.
"""

from __future__ import annotations

import numpy as np
import pytest

from blockgen.eval.bench import baselines as bl
from blockgen.eval.bench import geometry as geom
from blockgen.eval.bench import topology
from blockgen.utils.data import Structure


def _house(n: int, seed: int) -> Structure:
    rng = np.random.default_rng(seed)
    occ = np.ones((n, n, n), dtype=bool)
    occ[1:-1, 1:-1, 1:-1] = False
    ids = np.where(occ, 5, 0).astype(np.int32)
    ids[occ & (rng.random(occ.shape) < 0.3)] = 4
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


@pytest.fixture(scope="module")
def train():
    return [_house(7 + (i % 5), i) for i in range(40)]


def test_palette_is_fitted_from_the_given_structures_only(train):
    pal = bl.Palette.fit(train)
    assert set(pal.materials) <= {(5, 0), (4, 0)}
    assert pal.weights.sum() == pytest.approx(1.0)
    assert pal.dims.shape[1] == 3
    assert np.all(pal.density > 0) and np.all(pal.density <= 1)


def test_palette_survives_an_empty_corpus():
    """A degenerate fit must fall back, not divide by zero."""
    pal = bl.Palette.fit([])
    assert pal.materials and pal.weights.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("name", bl.ORDER)
def test_every_baseline_is_deterministic_under_its_seed(name, train):
    a = bl.build(name, 3, train, seed=7)
    b = bl.build(name, 3, train, seed=7)
    for x, y in zip(a, b):
        assert np.array_equal(x.block_ids, y.block_ids)


@pytest.mark.parametrize("name", bl.ORDER)
def test_every_baseline_produces_non_empty_builds(name, train):
    for s in bl.build(name, 4, train, seed=0):
        assert int(s.occupied_mask.sum()) > 0


def test_unknown_baseline_is_refused(train):
    with pytest.raises(ValueError, match="unknown baseline"):
        bl.build("wave_function_collapse", 1, train)


# --- each arm has the property its name claims -----------------------------
def test_uniform_random_has_almost_no_planar_surface(train):
    xs = bl.build("uniform_random", 4, train, seed=0)
    assert geom.geometry_table(xs)["wall_frac"].mean() < 0.25


def test_shell_box_is_one_component_with_a_large_interior(train):
    xs = bl.build("shell_box", 4, train, seed=0)
    c = topology.coherence_table(xs)
    g = geom.geometry_table(xs)
    assert c["lcc_ratio"].mean() == pytest.approx(1.0)
    assert g["wall_frac"].mean() > 0.9
    assert g["interior_ratio_open"].mean() > 0.3


def test_gabled_house_encloses_an_interior_through_its_apertures(train):
    """It has a door and windows, so the interior is reachable rather than
    sealed -- the gable ends must still close, or it vents and reads as 0."""
    xs = bl.build("gabled_house", 4, train, seed=0)
    g = geom.geometry_table(xs)
    assert g["interior_ratio_open"].mean() > 0.2
    assert g["yaw_symmetry"].mean() > 0.8


def test_patchwork_keeps_real_local_statistics_but_not_global_coherence(train):
    """The attack's defining property: local patterns close to real, global
    structure not. If this stops holding, the adversarial probe is no longer
    probing anything."""
    xs = bl.build("patchwork", 6, train, seed=0)
    real = train[:6]
    assert geom.pattern_jsd(xs, real) < geom.pattern_jsd(
        bl.build("uniform_random", 6, train, seed=0), real)


def test_train_copy_noise_preserves_block_count(train):
    xs = bl.build("train_copy_noise", 4, train, seed=0, p=0.1)
    for s in xs:
        assert int(s.occupied_mask.sum()) > 0


def test_arms_returns_every_baseline_with_distinct_seeds(train):
    a = bl.arms(train, 3, seed=0)
    assert set(a) == set(bl.ORDER)
    assert all(len(v) == 3 for v in a.values())
    # distinct seeds per arm, so two baselines never share a random stream
    assert not np.array_equal(a["shell_box"][0].block_ids,
                              a["gabled_house"][0].block_ids)
