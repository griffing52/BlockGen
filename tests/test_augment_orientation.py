"""D4 augmentation must remap orientation block_data so facings stay correct.

Without this, an augmented copy shows wrong-facing stairs/logs (notes §17), which turns
the orientation-aware vocab's facing tokens into noise. Verified against np.rot90's
geometry (empirically: a +x block lands at +z under k=1, so east->south).

Run:  python -m pytest tests/test_augment_orientation.py -q
"""

from __future__ import annotations

import numpy as np

from blockgen.utils.augment import (_d4_arrays, _remap_log, _remap_stair,
                                     augment_structure)
from blockgen.utils.data import Structure


def test_stair_facing_rotates_with_geometry():
    assert _remap_stair(0, False, 1) == 2   # east -> south (matches +x -> +z)
    assert _remap_stair(2, False, 1) == 1   # south -> west
    assert _remap_stair(1, False, 1) == 3   # west -> north
    assert _remap_stair(3, False, 1) == 0   # north -> east
    assert _remap_stair(4, False, 1) == 6   # upside-down bit preserved (east+ud -> south+ud)


def test_stair_mirror_flips_x_only():
    assert _remap_stair(0, True, 0) == 1    # east <-> west
    assert _remap_stair(2, True, 0) == 2    # south unchanged under x-mirror
    assert _remap_stair(3, True, 0) == 3    # north unchanged


def test_four_rotations_are_identity():
    for d in range(8):
        f = d
        for _ in range(4):
            f = _remap_stair(f, False, 1)
        assert f == d, f"stair {d} not restored after 4 rotations"
    for d in (0, 1, 2, 4, 8, 12, 5, 9):     # species|axis combos
        a = d
        for _ in range(4):
            a = _remap_log(a, False, 1)
        assert a == d


def test_log_axis_swaps_xz():
    # species=1, x-axis (axis bits = 1<<2=4) -> data 5;  after 90° -> z-axis (2<<2=8) -> 9
    assert _remap_log(0b0101, False, 1) == 0b1001
    assert _remap_log(0b0000, False, 1) == 0b0000   # y-axis unchanged
    assert _remap_log(0b1101, False, 1) == 0b1101   # bark (axis 3) unchanged


def test_geometry_and_facing_stay_consistent():
    """An east-facing stair at the east edge must end facing south after one 90° rot."""
    bi = np.zeros((3, 1, 3), np.int32)
    bd = np.zeros((3, 1, 3), np.int32)
    bi[2, 0, 1] = 53                         # oak stairs at east edge
    bd[2, 0, 1] = 0                          # facing east
    orbit = list(_d4_arrays(bi, bd))
    bi1, bd1 = orbit[1]                      # (mirror=False, k=1)
    pos = tuple(np.argwhere(bi1 == 53)[0])
    assert bd1[pos] == 2, "stair facing did not rotate with the geometry"


def test_augment_structure_still_produces_orbit():
    s = Structure(block_ids=np.array([[[53, 1]]], np.int32),
                  block_data=np.array([[[0, 0]]], np.int32))
    orbit = augment_structure(s, dedupe=False)
    assert len(orbit) == 8
    # identity first, unchanged
    assert np.array_equal(orbit[0].block_data, s.block_data)
