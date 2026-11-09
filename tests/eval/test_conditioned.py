"""blockgen.eval.conditioned on synthetic classes: a faithful sampler scores high, a shuffled one does not."""
import numpy as np

from blockgen.eval import conditioned as cd
from blockgen.utils.data import Structure


def _tower(rng):
    h, w = rng.integers(14, 22), rng.integers(3, 5)
    ids = np.zeros((w, h, w), np.int32); ids[:, :, 0] = ids[:, :, -1] = ids[0] = ids[-1] = 4
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


def _slab(rng):
    a, b = rng.integers(10, 16, 2)
    ids = np.zeros((a, 2, b), np.int32); ids[:, 0] = 35; ids[rng.integers(a), 1, rng.integers(b)] = 1
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


def _house(rng):
    a = rng.integers(7, 11)
    ids = np.zeros((a, 6, a), np.int32)
    ids[:, 0] = 5; ids[0, :5] = ids[-1, :5] = ids[:, :5, 0] = ids[:, :5, -1] = 17; ids[:, 5] = 53
    ids[a // 2, 1:3, 0] = 0
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


MAKERS = {"tower": _tower, "pixelart": _slab, "house": _house}


def _set(n, seed):
    rng = np.random.default_rng(seed)
    return {c: cd.features([f(rng) for _ in range(n)]) for c, f in MAKERS.items()}


def test_faithful_beats_shuffled():
    train, ref, smp = _set(40, 0), _set(24, 1), _set(24, 2)
    geo = np.concatenate([g for g, _ in train.values()]); pal = np.concatenate([p for _, p in train.values()])
    y = [c for c, (g, _) in train.items() for _ in range(len(g))]
    probe = cd.Probe().fit(geo, pal, y)
    good = cd.score(smp, ref, probe, n_boot=20)
    bad = cd.score(cd.shuffled(smp, seed=3), ref, probe)
    assert good["mean_probe_acc"] > 0.9 and bad["mean_probe_acc"] < 0.7
    assert good["mean_margin"] > 0 and good["mean_margin"] > bad["mean_margin"] + 1.0
    for c, r in good["per_class"].items():
        assert r["n"] == 24 and r["n_ref"] == 24 and r["geom_kid"] < r["kid_to_others"]
        lo, hi = r["geom_kid_ci90"]
        assert lo <= hi
    assert "| mean |" in cd.table(good)


def test_n_caps_samples_and_palette_ignores_air():
    s = Structure(block_ids=np.array([[[0, 4], [4, 5]]], np.int32), block_data=np.zeros((1, 2, 2), np.int32))
    p = cd.palette([s])[0]
    assert p[0] == 0 and abs(p[4] - 2 / 3) < 1e-9 and abs(p.sum() - 1) < 1e-9
    train, ref = _set(20, 0), _set(10, 1)
    geo = np.concatenate([g for g, _ in train.values()]); pal = np.concatenate([p for _, p in train.values()])
    probe = cd.Probe().fit(geo, pal, [c for c, (g, _) in train.items() for _ in range(len(g))])
    res = cd.score(_set(30, 2), ref, probe, n=12)
    assert all(r["n"] == 12 for r in res["per_class"].values())
