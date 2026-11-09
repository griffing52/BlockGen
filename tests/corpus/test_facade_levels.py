"""The level stack of a face: encode/decode, in-box grids, the vocabulary, parents."""
import numpy as np
import pytest

from blockgen.corpus import facade as fc
from blockgen.corpus import facade_levels as fl


def _face(U=13, V=10, extent=9, seed=0):
    rng = np.random.default_rng(seed)
    depth = rng.integers(-1, extent, (U, V)).astype(np.int16)
    depth[0, 0] = 3                                   # at least one filled cell
    ids = np.where(depth >= 0, rng.choice([1, 4, 5, 17], (U, V)), 0).astype(np.int16)
    data = np.where(depth >= 0, rng.integers(0, 3, (U, V)), 0).astype(np.int16)
    return fc.Facade(face="-z", depth=depth, ids=ids, data=data, extent=extent)


def test_vocab_index_and_blocks_roundtrip():
    f = _face()
    vocab = fl.BlockVocab.fit([f], min_count=1)
    cls_ = vocab.index(f.ids, f.data)
    assert cls_[f.mask].min() >= 1                    # every train key is known
    ids, data = vocab.blocks(cls_)
    assert np.array_equal(ids[f.mask], f.ids[f.mask]) and np.array_equal(data[f.mask], f.data[f.mask])
    unseen = vocab.index(np.array([[99]]), np.array([[0]]))
    assert unseen[0, 0] == fl.UNK
    again = fl.BlockVocab.from_dict(vocab.to_dict())
    assert np.array_equal(again.keys, vocab.keys)


def test_finest_level_decodes_to_the_face():
    f = _face()
    vocab = fl.BlockVocab.fit([f], min_count=1)
    st = fl.encode(f, vocab)
    assert set(st.levels) == set(fl.FACTORS) and st.box == (13, 10, 9)
    back = fl.decode(st.levels[1], st.shape, st.extent, f.face, vocab)
    assert np.array_equal(back.depth, f.depth)
    assert np.array_equal(back.ids, f.ids) and np.array_equal(back.data, f.data)


def test_coarse_level_matches_pool_then_upsample():
    f = _face()
    vocab = fl.BlockVocab.fit([f], min_count=1)
    st = fl.encode(f, vocab, rule="any")
    for fac in (2, 4, 8, 16):
        want = fc.upsample(fc.pool(f, fac, "any"), fac, f.depth.shape)
        got = fl.decode(st.levels[fac], st.shape, st.extent, f.face, vocab)
        assert np.array_equal(got.depth, want.depth), fac
        assert np.array_equal(got.ids, want.ids), fac


def test_inbox_grid_covers_partial_blocks_only():
    m = fl.inbox_grid((13, 10), 4)
    assert m.shape == (8, 8)
    assert m[:4, :3].all() and not m[4:].any() and not m[:, 3:].any()
    st = fl.encode(_face(), fl.BlockVocab.fit([_face()], min_count=1))
    for fac, lv in st.levels.items():
        assert not lv.depth[~lv.inbox].any() and not lv.block[~lv.inbox].any()


def test_depth_classes_offset_by_one():
    f = _face()
    st = fl.encode(f, fl.BlockVocab.fit([f], min_count=1))
    lv = st.levels[1]
    assert np.array_equal(lv.depth[:13, :10], np.where(f.depth >= 0, f.depth + 1, 0))


def test_parent_repeats_coarse_cells():
    f = _face()
    st = fl.encode(f, fl.BlockVocab.fit([f], min_count=1))
    pd, pb = fl.parent(st.levels[4], 2)
    assert pd.shape == (16, 16)
    assert np.array_equal(pd[::2, ::2], st.levels[4].depth)


def test_stack_arrays_and_flip():
    f = _face()
    vocab = fl.BlockVocab.fit([f], min_count=1)
    arr = fl.stack_arrays([f, fl.flip_u(f)], vocab)
    assert arr["depth"][1].shape == (2, 32, 32) and arr["box"].shape == (2, 3)
    assert np.array_equal(arr["depth"][1][1][:13, :10], arr["depth"][1][0][:13, :10][::-1])


def test_oversized_face_is_refused():
    big = fc.Facade(face="-z", depth=np.zeros((33, 4), np.int16), ids=np.zeros((33, 4), np.int16),
                    data=np.zeros((33, 4), np.int16), extent=4)
    with pytest.raises(ValueError):
        fl.encode(big, fl.BlockVocab(keys=np.array([1], np.int32), counts=np.array([1])))
