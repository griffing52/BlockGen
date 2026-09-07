"""Geometric descriptors, and the invariances they claim by construction.

The claims this module makes are strong ones -- *exact* invariance to the D4 yaw
group and *exact* blindness to material -- so they are tested for bit equality
rather than for closeness. An approximate check would pass a descriptor that had
quietly started reading orientation.
"""

from __future__ import annotations

import numpy as np
import pytest

from blockgen.eval.bench import geometry as geom
from blockgen.eval.bench import probes as pb
from blockgen.utils.data import Structure


def _struct(occ: np.ndarray, block_id: int = 1) -> Structure:
    ids = np.where(occ.astype(bool), block_id, 0).astype(np.int32)
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


def _hollow_box(n: int = 7) -> np.ndarray:
    occ = np.ones((n, n, n), dtype=bool)
    occ[1:-1, 1:-1, 1:-1] = False
    return occ


def _house(rng: np.random.Generator) -> Structure:
    """A hollow box with a doorway and a few windows -- a caricature of a house."""
    occ = _hollow_box(9)
    occ[0, 1:3, 4] = False          # doorway through one wall
    occ[4, 5, 0] = False            # a window
    ids = np.where(occ, 5, 0).astype(np.int32)
    ids[occ & (np.arange(9)[:, None, None] % 3 == 0)] = 4
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


# --- the orbit table -------------------------------------------------------
def test_orbit_table_is_a_partition_of_the_pattern_space():
    assert geom.ORBIT_OF.shape == (256,)
    assert set(np.unique(geom.ORBIT_OF)) == set(range(geom.N_ORBITS))


def test_orbit_table_is_closed_under_the_group():
    """Every element of D4 must map a pattern to a pattern in the same orbit."""
    rng = np.random.default_rng(0)
    for code in rng.choice(256, size=64, replace=False):
        grid = ((int(code) >> np.arange(8)) & 1).astype(bool).reshape(2, 2, 2)
        for mirror in (False, True):
            base = grid[::-1] if mirror else grid
            for k in range(4):
                g = np.rot90(base, k, axes=(0, 2))
                other = int((g.reshape(-1).astype(np.int64) << np.arange(8)).sum())
                assert geom.ORBIT_OF[other] == geom.ORBIT_OF[int(code)]


# --- the invariance claims, tested exactly ---------------------------------
@pytest.mark.parametrize("transform", ["rot90_1", "rot90_2", "rot90_3", "mirror_x"])
def test_descriptor_is_exactly_d4_invariant(transform):
    s = _house(np.random.default_rng(1))
    moved = (pb.mirror_x(s) if transform == "mirror_x"
             else pb.rotate_y(s, int(transform[-1])))
    assert np.array_equal(geom.descriptor([s]), geom.descriptor([moved]))


def test_descriptor_is_exactly_blind_to_material():
    """Occupancy-only by construction. If this drifts, the ladder's G6 for the
    geometry family is silently testing something else."""
    s = _house(np.random.default_rng(2))
    rng = np.random.default_rng(0)
    assert np.array_equal(geom.descriptor([s]),
                          geom.descriptor([pb.shuffle_materials(s, rng)]))
    assert np.array_equal(geom.descriptor([s]), geom.descriptor([pb.monochrome(s)]))


# --- the individual statistics ---------------------------------------------
def test_pattern_histogram_normalizes_and_drops_the_empty_orbit():
    h = geom.pattern_histogram(_hollow_box())
    assert h[geom.EMPTY_ORBIT] == 0.0
    assert h.sum() == pytest.approx(1.0)
    assert np.all(h >= 0)


def test_pattern_histogram_of_empty_volume_is_zero():
    assert geom.pattern_histogram(np.zeros((4, 4, 4), bool)).sum() == 0.0


def test_thickness_is_one_for_a_single_wall_and_grows_with_solidity():
    wall = np.zeros((7, 7, 7), dtype=bool)
    wall[3] = True
    assert geom.thickness(wall).max() == 1

    solid = np.ones((9, 9, 9), dtype=bool)
    # taxicab depth of the centre of a 9-cube is 5 (distance to the padded air)
    assert geom.thickness(solid).max() == 5
    assert geom.thickness(solid).mean() > geom.thickness(_hollow_box(9)).mean()


def test_surface_to_volume_separates_dust_from_a_solid():
    dust = np.zeros((9, 9, 9), dtype=bool)
    dust[::3, ::3, ::3] = True                 # isolated blocks: all six faces
    assert geom.surface_to_volume(dust) == pytest.approx(6.0)
    assert geom.surface_to_volume(np.ones((9, 9, 9), bool)) < 1.0


def test_wall_fraction_is_one_for_a_plane_and_zero_for_scattered_blocks():
    plane = np.zeros((9, 3, 9), dtype=bool)
    plane[:, 1, :] = True
    assert geom.wall_fraction(plane) == pytest.approx(1.0)

    dust = np.zeros((9, 9, 9), dtype=bool)
    dust[::3, ::3, ::3] = True                 # every face is its own patch
    assert geom.wall_fraction(dust) == 0.0


def test_wall_fraction_falls_when_columns_are_jittered():
    """The rung `wall_fraction` exists to catch."""
    s = _struct(_hollow_box(11))
    jittered = pb.jitter_columns(s, 1, np.random.default_rng(0))
    assert (geom.wall_fraction(jittered.crop_to_non_air().occupied_mask)
            < geom.wall_fraction(s.occupied_mask))


def test_yaw_symmetry_is_one_for_a_symmetric_build():
    assert geom.yaw_symmetry(_hollow_box()) == pytest.approx(1.0)
    asym = _hollow_box(9)
    asym[6:, :, 6:] = True
    assert geom.yaw_symmetry(asym) < 1.0


def test_height_profile_is_scale_free_and_sums_to_one():
    occ = _hollow_box(9)
    hp = geom.height_profile(occ)
    assert hp.sum() == pytest.approx(1.0)
    assert len(hp) == geom.HEIGHT_BINS
    # doubling the build's height must not change where the mass sits. Uses a
    # height that is a multiple of HEIGHT_BINS so the binning is exact; at a
    # height of 9 into 8 bins the quantization itself moves mass around.
    box = _hollow_box(9)
    tall = np.repeat(box, 2, axis=1)
    assert (geom.height_profile(np.repeat(box, 8, axis=1))
            == pytest.approx(geom.height_profile(np.repeat(tall, 4, axis=1)), abs=0.02))


# --- interiors, sealed and reachable ---------------------------------------
def test_sealed_interior_matches_the_hollow_volume():
    sealed, _ = geom.interior_volumes(_hollow_box(7))
    assert sealed == 5 ** 3


def test_a_doorway_moves_interior_from_sealed_to_aperture_reachable():
    """The reason `interior_ratio_open` exists: `enclosed_air` counts a room
    with an open door as no room at all."""
    occ = _hollow_box(9)
    occ[0, 1:3, 4] = False                     # a 1x2 doorway
    sealed, through = geom.interior_volumes(occ)
    assert sealed == 0                         # the old measure sees nothing
    assert through > 100                       # the room is still there


def test_solidify_removes_interior_and_geometry_sees_it():
    s = _struct(_hollow_box(9))
    filled = pb.solidify(s)
    g0 = geom.geometry_stats(s)
    g1 = geom.geometry_stats(filled)
    assert g0.interior_ratio_open > 0.1
    assert g1.interior_ratio_open == pytest.approx(0.0)
    assert g1.thickness_mean > g0.thickness_mean


# --- the descriptor and its distance ---------------------------------------
def test_descriptor_layout_matches_its_width():
    s = _house(np.random.default_rng(3))
    assert geom.descriptor([s]).shape == (1, sum(w for _, w in geom.descriptor_layout()))


def test_standardizer_yields_unit_norm_rows_and_tolerates_constant_columns():
    x = np.zeros((8, 5))
    x[:, 0] = np.arange(8)                     # column 1..4 are constant
    std = geom.Standardizer.fit(x)
    z = std(x)
    assert np.allclose(np.linalg.norm(z, axis=1), 1.0)
    assert np.isfinite(z).all()


def test_geom_kid_is_near_zero_between_two_samples_of_one_distribution():
    """The unbiased estimator's defining property, and the reason it is the one
    reused here. Note `geom_kid(x, x)` on an *identical* array is NOT zero: the
    within-set terms exclude the diagonal and the cross term does not."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 12))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = rng.normal(size=(100, 12)) + 2.0
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    same = geom.geom_kid(x[:100], x[100:])
    shifted = geom.geom_kid(y, x[:100])
    assert abs(same) < 0.05 * shifted


def test_geom_kid_ranks_solidify_far_above_a_real_split():
    """The claim the geometry tier is built on, in miniature."""
    rng = np.random.default_rng(5)
    pool = []
    for _ in range(24):
        # Sealed boxes of varying size: `solidify` fills holes, so a build with
        # an opening to the outside has nothing for it to fill.
        n = int(rng.integers(7, 12))
        occ = _hollow_box(n)
        occ[1, 1, 1] = True
        pool.append(_struct(occ))
    a, b = pool[:12], pool[12:]
    std = geom.Standardizer.fit(geom.descriptor(a))
    fa, fb = geom.geom_features(a, std), geom.geom_features(b, std)
    fs = geom.geom_features([pb.solidify(s) for s in b], std)
    assert geom.geom_kid(fs, fa) > geom.geom_kid(fb, fa)


def test_pattern_jsd_is_zero_against_itself_and_positive_against_damage():
    pool = [_struct(_hollow_box(9)), _struct(_hollow_box(11))]
    assert geom.pattern_jsd(pool, pool) == pytest.approx(0.0, abs=1e-12)
    assert geom.pattern_jsd([pb.solidify(s) for s in pool], pool) > 0.05
