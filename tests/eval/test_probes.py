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
