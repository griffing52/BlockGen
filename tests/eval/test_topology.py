"""Structural coherence, pinned against the pre-existing connectivity code.

`blockgen/eval/validity.py` is deliberately left byte-identical so published
numbers stay reproducible, which means this module now holds a *second*
implementation of 6-connectivity. `test_matches_validity_module` is what keeps
the two from drifting apart.
"""

from __future__ import annotations

import numpy as np
import pytest

from blockgen.eval import validity
from blockgen.eval.bench import topology
from blockgen.utils.data import Structure


def _struct(occ: np.ndarray, block_id: int = 1) -> Structure:
    ids = np.where(occ.astype(bool), block_id, 0).astype(np.int32)
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


def _box(nx=4, ny=4, nz=4) -> np.ndarray:
    return np.ones((nx, ny, nz), dtype=bool)


def _hollow_box(n=5) -> np.ndarray:
    occ = np.ones((n, n, n), dtype=bool)
    occ[1:-1, 1:-1, 1:-1] = False
    return occ


# --- connectivity ----------------------------------------------------------
def test_label6_counts_components():
    occ = np.zeros((7, 3, 3), dtype=bool)
    occ[0:2] = True
    occ[5:7] = True                      # separated along x
    assert topology.label6(occ)[1] == 2


def test_label6_is_6_connected_not_26():
    """Diagonal touch must NOT join components."""
    occ = np.zeros((2, 2, 2), dtype=bool)
    occ[0, 0, 0] = True
    occ[1, 1, 1] = True
    assert topology.label6(occ)[1] == 2


def test_component_sizes_descending():
    occ = np.zeros((8, 2, 2), dtype=bool)
    occ[0:1] = True                      # 4 voxels
    occ[4:7] = True                      # 12 voxels
    sizes = topology.component_sizes(occ)
    assert sizes.tolist() == [12, 4]


def test_component_sizes_empty():
    assert topology.component_sizes(np.zeros((3, 3, 3), dtype=bool)).size == 0


@pytest.mark.parametrize("seed", range(8))
def test_matches_validity_module(seed):
    """The scipy path and the pure-python DFS must agree, always."""
    rng = np.random.default_rng(seed)
    occ = rng.random((8, 8, 8)) < 0.25
    if not occ.any():
        occ[0, 0, 0] = True
    s = _struct(occ)
    assert topology.n_components_fast(s) == validity.n_components(s)


# --- coherence fields ------------------------------------------------------
def test_solid_box_is_one_grounded_component():
    c = topology.coherence(_struct(_box()))
    assert c.n_components == 1
    assert c.lcc_ratio == 1.0
    assert c.disconnected_block_ratio == 0.0
    assert c.floating_component_frac == 0.0
    assert c.floating_block_frac == 0.0
    assert c.bbox_fill == 1.0


def test_empty_structure_does_not_crash():
    c = topology.coherence(_struct(np.zeros((3, 3, 3), dtype=bool)))
    assert c.n_blocks == 0
    assert c.lcc_ratio == 0.0
    assert c.disconnected_block_ratio == 1.0


def test_hollow_box_has_enclosed_air():
    c = topology.coherence(_struct(_hollow_box(5)))
    assert c.enclosed_air == 27          # the 3x3x3 interior
    assert c.enclosed_air_ratio == pytest.approx(27 / 125)


def test_solid_box_has_no_enclosed_air():
    assert topology.coherence(_struct(_box())).enclosed_air == 0


def test_floating_component_detected():
    """A blob above the base plane is floating; the slab below is grounded."""
    occ = np.zeros((4, 9, 4), dtype=bool)
    occ[:, 0:2, :] = True                # grounded slab, 32 voxels
    occ[0:2, 6:8, 0:2] = True            # floating blob, 8 voxels
    c = topology.coherence(_struct(occ))
    assert c.n_components == 2
    assert c.floating_component_frac == pytest.approx(0.5)
    assert c.floating_block_frac == pytest.approx(8 / 40)


def test_floating_is_not_trivially_zero_after_crop():
    """The whole point of the redefinition: cropping must not hide floating mass.

    A naive "does it touch y=0 of the array" test would call this grounded,
    because crop_to_non_air() slides the blob down to y=0.
    """
    occ = np.zeros((4, 9, 4), dtype=bool)
    occ[:, 0:2, :] = True
    occ[0:2, 6:8, 0:2] = True
    padded = np.zeros((4, 20, 4), dtype=bool)
    padded[:, 5:14, :] = occ             # lift everything off the array floor
    assert topology.coherence(_struct(padded)).floating_block_frac == pytest.approx(0.2)


def test_single_floating_component_is_grounded_after_crop():
    """One component alone always touches its own cropped base -- by definition."""
    occ = np.zeros((4, 20, 4), dtype=bool)
    occ[0:2, 10:12, 0:2] = True
    assert topology.coherence(_struct(occ)).floating_block_frac == 0.0


def test_lcc_ratio_matches_curation_definition():
    occ = np.zeros((10, 2, 2), dtype=bool)
    occ[0:3] = True                      # 12
    occ[8:10] = True                     # 8
    c = topology.coherence(_struct(occ))
    assert c.lcc_ratio == pytest.approx(12 / 20)
    assert c.disconnected_block_ratio == pytest.approx(8 / 20)


# --- reporting contract ----------------------------------------------------
def test_report_carries_real_reference_and_direction():
    """No coherence number may be emitted without the real value beside it."""
    gen = [_struct(_box()) for _ in range(6)]
    real = [_struct(_hollow_box(5)) for _ in range(6)]
    rep = topology.coherence_report(gen, real, n_boot=50)
    for key, entry in rep.items():
        assert entry["direction"] == "distance_to_real", key
        assert "gen" in entry and "real" in entry, key
        assert "w1" in entry and "w1_norm" in entry, key
        assert len(entry["gen"]["ci"]) == 2


def test_report_w1_is_zero_against_itself():
    real = [_struct(_hollow_box(5)), _struct(_box()), _struct(_box(3, 3, 3))]
    rep = topology.coherence_report(real, real, n_boot=50)
    assert rep["lcc_ratio"]["w1"] == pytest.approx(0.0)
    assert rep["enclosed_air_ratio"]["w1"] == pytest.approx(0.0)


def test_coherence_table_columns():
    t = topology.coherence_table([_struct(_box()), _struct(_hollow_box(5))])
    for key in topology.COHERENCE_METRICS:
        assert t[key].shape == (2,)
    assert t["max_dim"].tolist() == [4.0, 5.0]
