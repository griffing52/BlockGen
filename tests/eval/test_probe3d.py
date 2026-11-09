"""The 3D-native rows: samplers, pure-torch grouping, exact EMD (notes §65)."""
import numpy as np
import pytest

from blockgen.eval.bench import mesh_probe as mp
from blockgen.eval.bench import pc_encoders as pe
from blockgen.utils.data import Structure


def _hollow_cube(side=8):
    bi = np.zeros((side, side, side), np.int32)
    bi[...] = 1
    bi[1:-1, 1:-1, 1:-1] = 0          # sealed room
    return Structure(block_ids=bi, block_data=np.zeros_like(bi))


def test_structure_mesh_visibility_excludes_sealed_room():
    pytest.importorskip("embreex")
    m = mp.structure_mesh(_hollow_cube())
    masks = mp.sampler_masks(m)
    a = m.face_areas
    # 6 outer faces of 8x8 and 6 inner faces of 6x6, in unit-block areas
    outer, inner = 6 * 64, 6 * 36
    assert a[masks["vis"]].sum() / a.sum() == pytest.approx(outer / (outer + inner), abs=0.01)
    # the cutaway cameras look into the room
    assert a[masks["cut"]].sum() / a.sum() > 0.1


def test_surface_samples_colour_and_mask():
    m = mp.structure_mesh(_hollow_cube())
    x = mp.surface_samples(m, 500, np.random.default_rng(0))
    assert x.shape == (500, 6) and (x[:, 3:] >= 0).all() and (x[:, 3:] <= 1).all()
    e = mp.surface_samples(m, 10, np.random.default_rng(0), mask=np.zeros(len(m.faces), bool))
    assert np.allclose(e[:, :3], 0)


def test_fps_matches_bruteforce():
    torch = pytest.importorskip("torch")
    x = torch.rand(2, 200, 3)
    idx = pe.fps(x, 10)
    for b in range(2):
        pts = x[b].numpy()
        sel = [0]
        d = np.full(200, np.inf)
        for _ in range(9):
            d = np.minimum(d, ((pts - pts[sel[-1]]) ** 2).sum(1))
            sel.append(int(d.argmax()))
        assert idx[b].tolist() == sel


def test_emd_exact_small():
    rng = np.random.default_rng(0)
    a = rng.random((16, 3))
    clouds = np.stack([a, a[rng.permutation(16)], a + [1.0, 0, 0]])
    dm = pe.emd_matrix(clouds, workers=1)
    assert dm[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert dm[0, 2] == pytest.approx(1.0, abs=1e-6)


def test_needed_pairs_cover_statistics():
    from blockgen.eval.bench import pointcloud as pc
    N = 12
    ref, s = np.arange(4), np.arange(4, 8)
    need = pe.needed_pairs(N, ref, {"a": s}, 2)
    dm = np.where(need | need.T, np.random.default_rng(0).random((N, N)), np.nan)
    dm = np.minimum(dm, dm.T)
    for fn in (pc.mmd, pc.one_minus_cov, pc.one_nna):
        assert np.isfinite(fn(dm, ref, s[:2]))
