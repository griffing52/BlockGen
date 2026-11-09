import numpy as np
import pytest

from blockgen.eval.bench import pointcloud as pc
from blockgen.eval.bench import probes as pb
from blockgen.utils.data import Structure

torch = pytest.importorskip("torch")


def _box(n=10, h=8, hollow=True):
    ids = np.zeros((n, h, n), np.int32)
    ids[1:-1, 1:-1, 1:-1] = 1
    if hollow:
        ids[2:-2, 2:-2, 2:-2] = 0
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


def test_surface_mask_is_the_shell_of_a_solid_box():
    occ = _box(hollow=False).occupied_mask
    sm = pc.surface_mask(occ)
    inner = occ.copy()
    inner[2:-2, 2:-2, 2:-2] = False
    assert np.array_equal(sm, inner)


def test_solidify_moves_volume_chamfer_and_not_surface_chamfer():
    rng = np.random.default_rng(0)
    hollow = _box()
    solid = pb.solidify(hollow)
    assert solid.occupied_mask.sum() > hollow.occupied_mask.sum()
    for mode, expect_zero in (("surface", True), ("volume", False)):
        a = pc.points(hollow, mode, 2048, np.random.default_rng(1))
        b = pc.points(solid, mode, 2048, np.random.default_rng(1))
        dm = pc.chamfer_matrix(np.stack([a, b]), device="cpu", block=2)
        assert dm.shape == (2, 2) and dm[0, 0] == 0 and abs(dm[0, 1] - dm[1, 0]) < 1e-6
        if expect_zero:
            assert dm[0, 1] < 1e-6, mode
        else:
            assert dm[0, 1] > 1e-4, mode


def test_set_statistics_on_a_toy_matrix():
    # three reference shapes, three probes identical to them: perfect scores
    dm = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            dm[i, 3 + j] = dm[3 + j, i] = 0.0 if i == j else 1.0
            dm[i, j] = dm[3 + i, 3 + j] = 0.0 if i == j else 1.0
    ref, probe = np.arange(3), np.arange(3, 6)
    assert pc.mmd(dm, ref, probe) == 0.0
    assert pc.one_minus_cov(dm, ref, probe) == 0.0
    # probes all collapse onto reference 0: coverage drops to one third
    dm2 = np.ones((6, 6)); np.fill_diagonal(dm2, 0)
    dm2[0, 3:] = dm2[3:, 0] = 0.1
    assert abs(pc.one_minus_cov(dm2, ref, probe) - 2 / 3) < 1e-9
    # 1-NNA: two clusters perfectly apart -> 1.0
    dm3 = np.ones((6, 6)) * 5; dm3[:3, :3] = 1; dm3[3:, 3:] = 1; np.fill_diagonal(dm3, 0)
    assert pc.one_nna(dm3, ref, probe) == 1.0


def test_iou_matrix_and_grids():
    s = _box()
    g = pc.grids([s, s, pb.solidify(s)])
    dm = pc.iou_matrix(g, device="cpu")
    assert dm[0, 1] == 0 and 0 < dm[0, 2] < 1
