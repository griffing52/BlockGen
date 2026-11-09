import numpy as np
import pytest

from blockgen.corpus import canon
from blockgen.corpus.dedup import _UF, _band_keys, _groups


def _build(seed=0):
    rng = np.random.default_rng(seed)
    ids = np.zeros((20, 10, 15), np.uint16)
    ids[2:18, 0:8, 3:12] = rng.integers(1, 50, (16, 8, 9))
    return ids


def test_canon_hash_invariant_to_translation_rotation_mirror():
    ids = _build()
    h = canon.canon_hash(ids)
    shifted = np.zeros((40, 20, 30), np.uint16)
    shifted[7:27, 5:15, 9:24] = ids
    assert canon.canon_hash(shifted) == h
    for k in range(4):
        r = np.rot90(ids, k, axes=(0, 2))
        assert canon.canon_hash(r) == h
        assert canon.canon_hash(r[::-1]) == h
    # a vertical flip is NOT a symmetry we fold
    assert canon.canon_hash(ids[:, ::-1, :]) != h


def test_canon_hash_respects_mask():
    ids = _build()
    mask = ids > 0
    mask[2:6] = False
    assert canon.canon_hash(ids, mask) != canon.canon_hash(ids)
    assert canon.canon_hash(ids * mask) == canon.canon_hash(ids, mask)


def test_minhash_estimates_jaccard():
    ids = _build()
    _, s1, _ = canon.canon_and_sig(ids)
    edited = ids.copy()
    rng = np.random.default_rng(1)
    edited.flat[rng.choice(edited.size, 80)] = 0
    _, s2, _ = canon.canon_and_sig(edited)
    inter = np.sum((ids > 0) & (edited > 0)); union = np.sum((ids > 0) | (edited > 0))
    true = inter / union
    assert abs(canon.jaccard_from_sigs(s1, s2) - true) < 0.08
    _, s3, _ = canon.canon_and_sig(_build(seed=5))
    assert canon.jaccard_from_sigs(s1, s3) < 0.2


def test_lsh_bands_find_near_duplicates_and_union_find_clusters():
    a = _build()
    b = a.copy(); b.flat[np.random.default_rng(2).choice(b.size, 10)] = 0
    c = _build(seed=9)
    sigs = np.stack([canon.canon_and_sig(x)[1] for x in (a, b, c)])
    keys = _band_keys(sigs, bands=16)
    shared = [(i, j) for i in range(3) for j in range(i + 1, 3) if np.any(keys[i] == keys[j])]
    assert (0, 1) in shared and (0, 2) not in shared and (1, 2) not in shared
    uf = _UF(3)
    uf.union(0, 1)
    assert uf.find(1) == uf.find(0) != uf.find(2)
    groups = list(_groups(np.array([5, 1, 5, 2, 1])))
    assert sorted(sorted(g.tolist()) for g in groups) == [[0, 2], [1, 4]]


def test_kinds_rule():
    from blockgen.corpus.kinds import classify
    share = np.array([0.95, 0.95, 0.7, 0.3, 0.0, 0.95])
    tell = np.array([True, False, True, True, False, True])
    worlds = np.array([1, 1, 1, 1, 7, 9])
    assert classify(share, tell, worlds).tolist() == ["mineshaft", "build", "mixed", "build", "village", "mineshaft"]


def test_template_convert_maps_states_and_jigsaw():
    import gzip, io
    import nbtlib
    from nbtlib import tag as T
    from blockgen.corpus.templates import convert
    pal = T.List[T.Compound]([
        T.Compound({"Name": T.String("minecraft:oak_planks")}),
        T.Compound({"Name": T.String("minecraft:jigsaw"), "Properties": T.Compound({"orientation": T.String("up_north")})}),
        T.Compound({"Name": T.String("minecraft:structure_void")}),
        T.Compound({"Name": T.String("minecraft:white_wool")}),
    ])
    blocks = T.List[T.Compound]([
        T.Compound({"pos": T.List[T.Int]([0, 0, 0]), "state": T.Int(0)}),
        T.Compound({"pos": T.List[T.Int]([1, 0, 0]), "state": T.Int(1),
                    "nbt": T.Compound({"final_state": T.String("minecraft:cobblestone")})}),
        T.Compound({"pos": T.List[T.Int]([2, 0, 0]), "state": T.Int(2)}),
        T.Compound({"pos": T.List[T.Int]([0, 1, 0]), "state": T.Int(3)}),
    ])
    root = nbtlib.File({"size": T.List[T.Int]([3, 2, 1]), "palette": pal, "blocks": blocks,
                        "entities": T.List[T.Compound]([]), "DataVersion": T.Int(3955)})
    buf = io.BytesIO(); root.write(buf); raw = gzip.compress(buf.getvalue())
    ids, data, unknown = convert(raw)
    assert ids.shape == (3, 2, 1)
    assert ids[0, 0, 0] == 5 and ids[1, 0, 0] == 4 and ids[2, 0, 0] == 0 and ids[0, 1, 0] == 35
    assert not unknown


def test_tokenizer_roundtrip_and_hash_stability():
    from blockgen.corpus.tokenizer import CorpusTokenizer
    t = CorpusTokenizer.default()
    ids = _build().astype(np.uint16)
    ids[ids > 0] = ids[ids > 0] % 5 + 1          # ids the standard vocab knows (36 etc. decode as <unk>)
    data = (ids % 4).astype(np.uint8)
    tags = {"src": "archive", "q": "3", "size": "32", "win": "whole"}
    s = t.encode(ids, data, tags)
    assert s.dtype == np.uint16 and s[0] == 1 and s[-1] == 2 and len(s) == 2 + 4 + 4 * np.count_nonzero(ids)
    ids2, data2, tags2 = t.decode(s)
    assert tags2 == tags
    # encode takes the window as given; decode's box ends at the last occupied voxel
    assert ids2.shape == (18, 8, 12) and (ids2 == ids[:18, :8, :12]).all()
    # data survives only where the standard vocab keys on it (class-level fidelity)
    assert ((data2 == data[:18, :8, :12]) | (ids2 == 0)).mean() > 0.5
    assert t.hash == CorpusTokenizer.default().hash
    assert CorpusTokenizer.load(t.save("/tmp/_tok_test.json")).hash == t.hash


def test_shard_windows_are_reproducible_and_bounded():
    from blockgen.corpus.shards import _windows
    ids = np.ones((60, 40, 60), np.uint16)
    data = np.zeros_like(ids, np.uint8)
    # fully dense: even a 16^3 window holds 4096 > 2048 blocks, so nothing is yielded
    assert list(_windows(ids, data, max_blocks=2048, max_windows=4, rng=np.random.default_rng(7))) == []
    # checkerboard (half dense): 48 and 32 and 24 windows are too big, 16^3 / 2 = 2048 fits
    x, y, z = np.indices(ids.shape)
    ids = (((x + y + z) % 2) == 0).astype(np.uint16)
    w1 = list(_windows(ids, data, max_blocks=2048, max_windows=4, rng=np.random.default_rng(7)))
    w2 = list(_windows(ids, data, max_blocks=2048, max_windows=4, rng=np.random.default_rng(7)))
    assert len(w1) == 4 and all(e == 16 and np.count_nonzero(wi) <= 2048 for _, _, _, e, wi, _ in w1)
    assert [w[:4] for w in w1] == [w[:4] for w in w2]
