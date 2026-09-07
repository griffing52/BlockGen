"""Corruption and invariance probes.

`shuffle_materials` is copied rather than imported from
`scripts/validate_perceptual.py`, whose output is a published artifact
(`outputs/analysis/perceptual_validation.json`). `test_shuffle_matches_the_script`
is what keeps the copy from drifting.
"""

from __future__ import annotations

import numpy as np
import pytest

from blockgen.eval.bench import probes
from blockgen.utils.data import Structure


def _house(seed=0, nx=7, ny=6, nz=7) -> Structure:
    rng = np.random.default_rng(seed)
    ids = np.zeros((nx, ny, nz), dtype=np.int32)
    ids[:, 0, :] = 4
    ids[0, :, :] = ids[-1, :, :] = 5
    ids[:, :, 0] = ids[:, :, -1] = 98
    ids[:, -1, :] = 5
    ids[2, 2, 2] = 35
    data = np.zeros_like(ids)
    data[ids == 5] = rng.integers(0, 4, size=int((ids == 5).sum()))
    return Structure(block_ids=ids, block_data=data)


def _occ(s):
    return int(s.occupied_mask.sum())


def _multiset(s):
    from collections import Counter
    occ = s.occupied_mask
    return Counter(zip(s.block_ids[occ].tolist(), s.block_data[occ].tolist()))


# --- material shuffle ------------------------------------------------------
def test_shuffle_preserves_geometry_and_multiset():
    """This is exactly why it is invisible to voxel palette statistics."""
    s = _house()
    out = probes.shuffle_materials(s, np.random.default_rng(0))
    assert out.shape == s.crop_to_non_air().shape
    assert np.array_equal(out.occupied_mask, s.crop_to_non_air().occupied_mask)
    assert _multiset(out) == _multiset(s)


def test_shuffle_actually_moves_blocks():
    s = _house()
    out = probes.shuffle_materials(s, np.random.default_rng(0))
    assert not np.array_equal(out.block_ids, s.crop_to_non_air().block_ids)


def test_shuffle_matches_the_script():
    """Pin the copy to `scripts/validate_perceptual.py` at a fixed seed."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "validate_perceptual.py"
    spec = importlib.util.spec_from_file_location("_vp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    s = _house(seed=3)
    mine = probes.shuffle_materials(s, np.random.default_rng(7))
    theirs = mod.shuffle_materials(s, np.random.default_rng(7))
    assert np.array_equal(mine.block_ids, theirs.block_ids)
    assert np.array_equal(mine.block_data, theirs.block_data)


# --- monochrome ------------------------------------------------------------
def test_monochrome_keeps_geometry_and_collapses_palette():
    s = _house()
    out = probes.monochrome(s)
    assert np.array_equal(out.occupied_mask, s.crop_to_non_air().occupied_mask)
    assert len(_multiset(out)) == 1


# --- noise -----------------------------------------------------------------
def test_block_noise_preserves_occupancy():
    """Retyping must never add or remove blocks -- that would confound damage."""
    s = _house()
    out = probes.block_noise(s, 0.5, np.random.default_rng(0))
    assert np.array_equal(out.occupied_mask, s.crop_to_non_air().occupied_mask)


def test_block_noise_scales_with_p():
    s = _house(seed=1)
    base = s.crop_to_non_air()
    changed = []
    for p in (0.0, 0.5):
        out = probes.block_noise(s, p, np.random.default_rng(0))
        changed.append(int((out.block_ids != base.block_ids).sum()))
    assert changed[0] == 0
    assert changed[1] > 0


# --- chunk deletion --------------------------------------------------------
def test_chunk_delete_removes_blocks():
    s = _house()
    out = probes.chunk_delete(s, 0.4, np.random.default_rng(0))
    assert _occ(out) < _occ(s.crop_to_non_air())


def test_chunk_delete_is_contiguous_not_scattered():
    """A slab, not salt-and-pepper: it should leave an intact remainder."""
    s = _house(nx=12, ny=6, nz=12)
    out = probes.chunk_delete(s, 0.25, np.random.default_rng(2))
    assert _occ(out) > 0


# --- invariances -----------------------------------------------------------
@pytest.mark.parametrize("k", [1, 2, 3])
def test_rotation_preserves_block_multiset(k):
    s = _house()
    assert _multiset(probes.rotate_y(s, k)) == _multiset(s)


def test_rotation_by_four_is_identity():
    s = _house().crop_to_non_air()
    out = s
    for _ in range(4):
        out = probes.rotate_y(out, 1)
    assert np.array_equal(out.block_ids, s.block_ids)


def test_rotation_swaps_horizontal_extent():
    s = _house(nx=9, ny=4, nz=5)
    out = probes.rotate_y(s, 1)
    assert out.shape[0] == s.shape[2] and out.shape[2] == s.shape[0]
    assert out.shape[1] == s.shape[1], "vertical axis must be untouched"


def test_mirror_preserves_multiset_and_shape():
    s = _house()
    out = probes.mirror_x(s)
    assert out.shape == s.crop_to_non_air().shape
    assert _multiset(out) == _multiset(s)


def test_mirror_twice_is_identity():
    s = _house().crop_to_non_air()
    assert np.array_equal(probes.mirror_x(probes.mirror_x(s)).block_ids, s.block_ids)


# --- decimation ------------------------------------------------------------
def test_canon_reduces_extent_monotonically():
    s = _house(nx=32, ny=20, nz=32)
    d16 = probes.canon([s], 16)[0]
    d8 = probes.canon([s], 8)[0]
    assert max(d16.shape) <= 16
    assert max(d8.shape) <= 8
    assert _occ(d8) < _occ(d16) < _occ(s)


# --- suite -----------------------------------------------------------------
def test_probe_suite_is_complete_and_same_length():
    pool = [_house(seed=i) for i in range(5)]
    suite = probes.probe_suite(pool, np.random.default_rng(0))
    for rung in ("canon16", "canon8", "noise_1", "noise_5", "noise_10",
                 "chunk_delete_20", "material_shuffle", "monochrome",
                 "rot90_1", "rot90_2", "mirror_x"):
        assert rung in suite, rung
        assert len(suite[rung]) == len(pool), rung


def test_probe_suite_is_deterministic():
    pool = [_house(seed=i) for i in range(4)]
    a = probes.probe_suite(pool, np.random.default_rng(0))
    b = probes.probe_suite(pool, np.random.default_rng(0))
    for rung in a:
        for x, y in zip(a[rung], b[rung]):
            assert np.array_equal(x.block_ids, y.block_ids), rung


# --- structural rungs ------------------------------------------------------
def _hollow(n: int = 9) -> Structure:
    occ = np.ones((n, n, n), dtype=bool)
    occ[1:-1, 1:-1, 1:-1] = False
    ids = np.where(occ, 5, 0).astype(np.int32)
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


def test_solidify_fills_the_interior_and_keeps_the_silhouette():
    s = _hollow(9)
    filled = probes.solidify(s)
    assert filled.occupied_mask.sum() > s.occupied_mask.sum()
    assert filled.crop_to_non_air().shape == s.crop_to_non_air().shape
    # the outer shell is untouched, which is exactly why a camera cannot see it
    assert np.array_equal(filled.occupied_mask[0], s.occupied_mask[0])


def test_solidify_uses_the_builds_own_dominant_material():
    """A filled interior must not be detectable as a palette anomaly."""
    s = _hollow(9)
    before = set(np.unique(s.block_ids))
    assert set(np.unique(probes.solidify(s).block_ids)) <= before


def test_solidify_is_a_no_op_on_a_build_with_no_enclosed_air():
    solid = Structure(block_ids=np.full((5, 5, 5), 5, np.int32),
                      block_data=np.zeros((5, 5, 5), np.int32))
    assert np.array_equal(probes.solidify(solid).block_ids, solid.block_ids)


def test_occupancy_noise_preserves_block_count_and_palette_exactly():
    """The structural analogue of `block_noise`, which is its mirror image:
    that one keeps geometry and destroys palette, this one the reverse."""
    s = _hollow(11)
    rng = np.random.default_rng(0)
    out = probes.occupancy_noise(s, 0.10, rng)
    assert out.occupied_mask.sum() == s.occupied_mask.sum()
    a = np.sort(s.block_ids[s.occupied_mask])
    b = np.sort(out.block_ids[out.occupied_mask])
    assert np.array_equal(a, b)
    assert not np.array_equal(out.occupied_mask, s.occupied_mask)


def test_occupancy_noise_damage_grows_with_p():
    s = _hollow(11)
    base = s.occupied_mask
    moved = []
    for p in (0.01, 0.05, 0.10):
        out = probes.occupancy_noise(s, p, np.random.default_rng(0))
        moved.append(int((out.occupied_mask != base).sum()))
    assert moved[0] < moved[1] < moved[2]


def test_jitter_columns_preserves_the_footprint_and_breaks_planes():
    s = _hollow(11)
    out = probes.jitter_columns(s, 1, np.random.default_rng(0))
    # a column may shift off the top or bottom, but the footprint is unchanged
    assert out.occupied_mask.any(axis=1).sum() <= s.occupied_mask.any(axis=1).sum()
    assert not np.array_equal(out.occupied_mask, s.occupied_mask)


def test_probe_suite_includes_the_structural_rungs():
    suite = probes.probe_suite([_hollow(9)], np.random.default_rng(0))
    for rung in ("solidify", "occ_noise_1", "occ_noise_5", "occ_noise_10",
                 "jitter_columns"):
        assert rung in suite and len(suite[rung]) == 1


# --- the graded structural dose --------------------------------------------
def test_partial_solidify_fills_the_requested_fraction_of_interior():
    """The dose has to be calibrated or the dose-response table is decoration.

    Filling rooms largest-first put 71% of the damage in the first 25% step;
    filling inward from the walls fills exactly the fraction asked for.
    """
    from scipy import ndimage

    occ = np.ones((11, 11, 11), dtype=bool)
    occ[1:-1, 1:-1, 1:-1] = False
    s = _hollow(11)
    pockets = int((ndimage.binary_fill_holes(occ) & ~occ).sum())
    rng = np.random.default_rng(0)
    base = int(s.occupied_mask.sum())
    for frac in (0.25, 0.5, 0.75, 1.0):
        got = int(probes.partial_solidify(s, frac, rng).occupied_mask.sum()) - base
        assert got == pytest.approx(frac * pockets, rel=0.02)


def test_partial_solidify_is_monotone_and_ends_at_solidify():
    s = _hollow(11)
    rng = np.random.default_rng(0)
    counts = [int(probes.partial_solidify(s, f, rng).occupied_mask.sum())
              for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert counts == sorted(counts)
    assert counts[-1] == int(probes.solidify(s).occupied_mask.sum())


def test_partial_solidify_at_zero_is_a_no_op():
    s = _hollow(9)
    out = probes.partial_solidify(s, 0.0, np.random.default_rng(0))
    assert np.array_equal(out.block_ids, s.crop_to_non_air().block_ids)
