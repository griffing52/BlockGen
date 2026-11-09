"""Rewards on synthetic builds, the pillar refusal, and face_stats parity with the old script."""
import numpy as np
import pytest
from scipy import ndimage as ndi

from blockgen.corpus import facade as fc
from blockgen.eval.facade_stats import face_stats
from blockgen.posttrain import rewards as rw
from blockgen.utils.data import Structure


def box(n=8, h=6, speckle=0.0, seed=0) -> Structure:
    """A hollow box; ``speckle`` pushes that share of the shell blocks one voxel out."""
    ids = np.zeros((n + 2, h + 2, n + 2), np.int32)
    ids[1:-1, 1:-1, 1:-1] = 5
    ids[2:-2, 2:-2, 2:-2] = 0
    if speckle:
        rng = np.random.default_rng(seed)
        for x, y, z in np.argwhere(ids == 5):
            if rng.random() < speckle:
                for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1), (0, 1, 0)):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if ids[nx, ny, nz] == 0 and not ids[nx + dx:, :, :].any() if dx > 0 else ids[nx, ny, nz] == 0:
                        ids[nx, ny, nz] = 5
                        break
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


def speckled_box(n=8, h=6, seed=0) -> Structure:
    """The same box with relief jumps: a third of the shell cells pushed outward by two."""
    base = box(n, h)
    ids = np.zeros((n + 6, h + 6, n + 6), np.int32)
    ids[2:-2, 2:-2, 2:-2] = base.block_ids
    rng = np.random.default_rng(seed)
    shell = np.argwhere(ids == 5)
    for x, y, z in shell[rng.random(len(shell)) < 0.33]:
        for d in ((2, 0, 0), (-2, 0, 0), (0, 0, 2), (0, 0, -2), (0, 2, 0)):
            nx, ny, nz = x + d[0], y + d[1], z + d[2]
            if ids[nx, ny, nz] == 0:
                ids[nx, ny, nz] = 7
                break
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


def test_face_solidity_prefers_the_solid_box():
    r = rw.get("face_solidity")
    solid, speckled = box(), speckled_box()
    s = r.score([solid, speckled])
    assert s.shape == (2,) and np.isfinite(s).all()
    assert s[0] == pytest.approx(1.0)            # lcc 1, no jumps, holes, lone cells or dust
    assert s[1] < s[0]


def test_face_solidity_handles_empty_builds():
    empty = Structure(block_ids=np.zeros((3, 3, 3), np.int32), block_data=np.zeros((3, 3, 3), np.int32))
    s = rw.get("face_solidity").score([empty, box()])
    assert np.isfinite(s).all() and s[0] < s[1]


def test_lcc26_is_the_largest_26_connected_share():
    ids = np.zeros((6, 3, 3), np.int32)
    ids[0, 0, 0] = ids[1, 1, 1] = 1              # diagonal pair: one piece at 26, two at 6
    ids[4, 0, 0] = 1                             # a loose block
    s = rw.get("lcc26").score([Structure(block_ids=ids, block_data=np.zeros_like(ids))])
    assert s[0] == pytest.approx(2 / 3)
    assert rw.get("lcc26").score([box()])[0] == pytest.approx(1.0)
    empty = Structure(block_ids=np.zeros((2, 2, 2), np.int32), block_data=np.zeros((2, 2, 2), np.int32))
    assert rw.get("lcc26").score([empty])[0] == 0.0


def test_novelty_penalty_punishes_copies_of_train():
    train = [box(6, 4), box(8, 5), box(10, 6)]
    r = rw.get("novelty_penalty", train=train, grid=16)
    copy, fresh = box(8, 5), speckled_box(7, 5)
    s = r.score([copy, fresh])
    assert s[0] == pytest.approx(-2.0)           # IoU 1 and a duplicate
    assert s[1] > s[0] and np.isfinite(s).all()


def test_pillar_rewards_are_refused_without_allow_pillar():
    for name in rw.PILLAR_SIGNALS:
        with pytest.raises(rw.PillarReward):
            rw.get(name)
        r = rw.get(name, allow_pillar=True)
        assert r.pillar and r.name == name
        with pytest.raises(NotImplementedError):
            r.score([box()])
    with pytest.raises(KeyError):
        rw.get("not_a_reward")


def test_composite_weights_and_pillar_inheritance():
    structs = [box(), speckled_box()]
    a = rw.get("face_solidity").score(structs)
    b = rw.get("lcc26").score(structs)
    c = rw.get("composite", weights={"face_solidity": 2.0, "lcc26": 0.5})
    assert not c.pillar
    assert np.allclose(c.score(structs), 2.0 * a + 0.5 * b)
    assert np.allclose(rw.get("composite", weights={"lcc26": 1.0}).score(structs), b)
    with pytest.raises(rw.PillarReward):
        rw.get("composite", weights={"face_solidity": 1.0, "coherence": 1.0})
    assert rw.get("composite", weights={"coherence": 1.0}, allow_pillar=True).pillar
    with pytest.raises(ValueError):
        rw.get("composite", weights={})
    with pytest.raises(ValueError):
        rw.get("composite", weights={"composite": 1.0})


def test_signal_tokens_round_trip():
    assert rw.signal_token("lcc26") == "lcc26" and rw.signal_token("lcc26", 1.0) == "lcc26"
    assert rw.signal_token("lcc26", 0.5) == "lcc26@0.5" and rw.parse_signal("lcc26@0.5") == ("lcc26", 0.5)
    assert rw.parse_signal("lcc26") == ("lcc26", 1.0)
    assert rw.parse_weights("face_solidity:2, lcc26") == {"face_solidity": 2.0, "lcc26": 1.0}
    with pytest.raises(ValueError):
        rw.parse_weights(",")
    with pytest.raises(ValueError):
        rw.signal_token("bad:name")
    assert rw.from_signals(["lcc26"]).name == "lcc26"
    c = rw.from_signals(["lcc26@0.5", "face_solidity"])
    assert c.name == "composite-face_solidity-lcc26@0.5" and rw.signals_of(c) == ["face_solidity", "lcc26@0.5"]
    assert rw.signals_of(rw.get("lcc26")) == ["lcc26"]
    assert set(rw.CLI_REWARDS) <= set(rw.REWARDS) and not any(n in rw.CLI_REWARDS for n in rw.PILLAR_SIGNALS + ("composite",))


def _old_face_stats(s):
    """The function as it stood in scripts/facade_sheets.py before the move, verbatim."""
    s = s.crop_to_non_air()
    faces = fc.project_all(s, fc.ALL_FACES)
    comps, holes, jumps, lone = [], [], [], []
    for f in faces.values():
        m = f.mask
        if not m.any():
            continue
        _, n = ndi.label(m, structure=np.ones((3, 3)))
        comps.append(n)
        filled = ndi.binary_fill_holes(m)
        holes.append(int((filled & ~m).sum()) / max(m.sum(), 1))
        d = f.depth.astype(int)
        pairs = []
        for ax in (0, 1):
            a, b = np.moveaxis(m, ax, 0)[:-1], np.moveaxis(m, ax, 0)[1:]
            da, db = np.moveaxis(d, ax, 0)[:-1], np.moveaxis(d, ax, 0)[1:]
            both = a & b
            pairs.append((np.abs(da - db)[both] >= 2))
        pairs = np.concatenate(pairs)
        jumps.append(pairs.mean() if len(pairs) else 0.0)
        nb = ndi.convolve(m.astype(int), np.array([[0,1,0],[1,0,1],[0,1,0]]), mode='constant')
        lone.append(int(((nb == 0) & m).sum()) / max(m.sum(), 1))
    occ = s.occupied_mask
    lab, n3 = ndi.label(occ, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel())[1:]
    sizes.sort()
    big = sizes[-1] if len(sizes) else 0
    dust = int((sizes < 5).sum())
    return dict(face_comps=np.mean(comps), face_holes=np.mean(holes), face_jumps=np.mean(jumps),
                face_lone=np.mean(lone), comps3d=n3, lcc=big / max(occ.sum(), 1), dust=dust,
                dust_blocks=int(sizes[sizes < 5].sum()) / max(occ.sum(), 1))


def test_face_stats_matches_the_old_script():
    rng = np.random.default_rng(1)
    fixtures = [box(), speckled_box()]
    for _ in range(3):                           # random rubble, so dust and holes are exercised
        ids = (rng.random((9, 7, 9)) < 0.35).astype(np.int32) * 3
        fixtures.append(Structure(block_ids=ids, block_data=np.zeros_like(ids)))
    for s in fixtures:
        new, old = face_stats(s), _old_face_stats(s)
        assert set(new) == set(old) == {"face_comps", "face_holes", "face_jumps", "face_lone", "comps3d", "lcc", "dust", "dust_blocks"}
        for k in old:
            assert new[k] == pytest.approx(old[k]), k


def test_facade_sheets_script_imports_face_stats():
    src = open("scripts/facade_sheets.py").read()
    assert "from blockgen.eval.facade_stats import face_stats" in src
    assert "def face_stats" not in src
