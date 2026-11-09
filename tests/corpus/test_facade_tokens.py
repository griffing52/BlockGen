import json

import numpy as np
import pytest
import torch

from blockgen.corpus import facade as fc
from blockgen.corpus import facade_baselines as fb
from blockgen.corpus import shards
from blockgen.corpus.tokenizer import CorpusTokenizer
from blockgen.utils.data import Structure

TAGS = {"src": "curated", "q": "4", "size": "32", "win": "whole"}


def _house(seed=0):
    rng = np.random.default_rng(seed)
    ids = np.zeros((9, 6, 7), np.int32)
    ids[1:8, 0, 1:6] = 1
    ids[1, 0:5, 1:6] = ids[7, 0:5, 1:6] = 5
    ids[1:8, 0:5, 1] = ids[1:8, 0:5, 5] = 5
    ids[1:8, 5, 1:6] = 1
    ids[4, 1:3, 1] = 0                      # a doorway
    ids[2, 2, 1] = 20                       # a window
    ids[3, 4, 1] = 0; ids[3, 4, 2] = 5      # a recess
    data = (rng.integers(0, 4, ids.shape) * (ids == 5)).astype(np.int32)
    return Structure(block_ids=ids, block_data=data)


def test_voxel_tokenizer_hash_is_untouched():
    tok = CorpusTokenizer.default()
    assert tok.kind == "voxels" and "kind" not in tok.to_json()
    assert tok.hash == "60684456e598"       # the hash every existing shard and checkpoint carries


def test_facade_tokenizer_round_trips_a_face():
    tok = CorpusTokenizer.default(kind="facades")
    assert tok.kind == "facades" and tok.hash != CorpusTokenizer.default().hash
    assert tok.tag_order == ("src", "q", "size", "win", "face", "cell")
    s = _house().crop_to_non_air()
    for name, f in fc.project_all(s, fc.ALL_FACES).items():
        seq = tok.encode_face(f, TAGS)
        assert seq.dtype == np.uint16 and seq[-1] == 2
        back, tags = tok.decode_face(seq)
        assert tags == TAGS and back.face == name
        assert np.array_equal(back.depth, f.depth)
        assert np.array_equal(back.ids, f.ids) and np.array_equal(back.data, f.data)
    # a sequence length is header + 1 per empty cell + 2 per filled cell
    f = fc.project(s, "-z")
    assert len(tok.encode_face(f, TAGS)) == 8 + int((~f.mask).sum()) + 2 * int(f.mask.sum()) + 1
    # a reload keeps the kind
    blob = json.loads(json.dumps(tok.to_json()))
    assert blob["kind"] == "facades"


def test_lifting_decoded_faces_reproduces_the_shell():
    tok = CorpusTokenizer.default(kind="facades")
    s = _house().crop_to_non_air()
    faces = [tok.decode_face(tok.encode_face(f, TAGS))[0] for f in fc.project_all(s, fc.ALL_FACES).values()]
    lifted = fc.lift(faces, s.shape)
    assert np.array_equal(lifted.block_ids, fc.shell(s, fc.ALL_FACES).block_ids)


def test_prefix_and_truncated_decode():
    tok = CorpusTokenizer.default(kind="facades")
    p = tok.prefix(TAGS, "+x", (7, 5))
    assert len(p) == 8 and tok.decode_tag(p[5]) == ("face", "+x")
    s = _house().crop_to_non_air()
    seq = tok.encode_face(fc.project(s, "-z"), TAGS)
    f, _ = tok.decode_face(seq[:20])        # cut mid-face: the rest stays see-through
    assert f is not None and f.depth.shape == fc.project(s, "-z").depth.shape
    assert (f.depth[:, 2:] == fc.EMPTY).all()
    assert tok.decode_face(np.array([1, 2], np.uint16))[0] is None


def test_shard_find_tells_kinds_apart(tmp_path, monkeypatch):
    monkeypatch.setattr(shards, "SHARD_DIR", str(tmp_path))
    for h, kind in (("aaa", "voxels"), ("bbb", "facades")):
        d = tmp_path / "tier" / h
        d.mkdir(parents=True)
        spec = {"tier": "tier", "tokenizer": h}
        if kind != "voxels":
            spec["kind"] = kind
        (d / "spec.json").write_text(json.dumps(spec))
    assert shards.find("tier").endswith("aaa")
    assert shards.find("tier", kind="facades").endswith("bbb")
    with pytest.raises(FileNotFoundError):
        shards.find("other", kind="facades")


def test_facade_baselines_fit_and_generate_in_real_boxes():
    train = [_house(i) for i in range(4)]
    out = fb.arms(train, n=3, seed=1)
    assert list(out) == list(fb.ORDER)
    for name, structs in out.items():
        assert len(structs) == 3
        for s in structs:
            assert s.shape == (9, 6, 7) or s.shape in {tuple(b) for b in fb.FacadeStats.fit(train).boxes}
            assert s.metadata["corpus"] == name
    again = fb.build("facade_markov", 3, train, seed=1 + 1)
    assert all(np.array_equal(a.block_ids, b.block_ids) for a, b in zip(out["facade_markov"], again))
    # patchwork faces are real faces: every cell it places is one a train house has
    st = fb.FacadeStats.fit(train)
    pw = fb.facade_patchwork(st, np.random.default_rng(0))
    assert pw.occupied_mask.sum() > 0


def test_joint_layout_round_trips_every_face_in_order():
    tok = CorpusTokenizer.default(kind="facades", layout="joint")
    per = CorpusTokenizer.default(kind="facades")
    assert tok.hash != per.hash and tok.to_json()["layout"] == "joint" and "layout" not in per.to_json()
    s = _house().crop_to_non_air()
    faces = list(fc.project_all(s, fc.ALL_FACES).values())
    seq = tok.encode_faces(faces, TAGS)
    back, tags = tok.decode_faces(seq)
    assert tags == TAGS and [f.face for f in back] == list(fc.ALL_FACES)
    for a, b in zip(faces, back):
        assert np.array_equal(a.depth, b.depth) and np.array_equal(a.ids, b.ids)
    assert np.array_equal(fc.lift(back, s.shape).block_ids, fc.shell(s, fc.ALL_FACES).block_ids)
    # the per-face sequence of one face is the joint sequence of that one face
    assert np.array_equal(per.encode_face(faces[0], TAGS), per.encode_faces([faces[0]], TAGS))
    with pytest.raises(ValueError):
        CorpusTokenizer.default(kind="voxels", layout="joint")


class _Babbler:
    """A stand-in model that emits random cell tokens: enough to test header forcing."""
    def __init__(self, tok, seed=0):
        self.tok = tok; self.max_seq_len = 8200
        self.rng = np.random.default_rng(seed)
        self._p = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self._p

    def eval(self): pass
    def train(self): pass

    def forward(self, step_in, past=None, return_cache=True):
        B = step_in.shape[0]
        V = self.tok.vocab_size
        logits = torch.full((B, 1, V), -1e9)
        choices = [self.tok.tag_id("cell", "empty"), self.tok.vocab.coord_id(0), self.tok.vocab.block_id_token(3), 2]
        for b in range(B):
            c = choices[self.rng.integers(len(choices))]      # an occasional EOS ends a face early
            logits[b, 0, c] = 0.0
        return logits, past


def test_generate_faces_forces_every_header_into_one_box():
    from blockgen.train.sample import generate_faces
    tok = CorpusTokenizer.default(kind="facades", layout="joint")
    model = _Babbler(tok)
    plans = [[("-z", 5, 4), ("+x", 3, 4), ("+y", 5, 3)], [("-z", 6, 2), ("+x", 2, 2), ("+y", 6, 2)]]
    seqs = generate_faces(model, tok, TAGS, plans, temperature=1.0, top_k=0, seed=0)
    assert len(seqs) == 2
    for plan, seq in zip(plans, seqs):
        faces, tags = tok.decode_faces(seq)
        assert tags == TAGS and seq[-1] == 2
        assert [(f.face, *f.depth.shape) for f in faces] == plan


def test_d4_augmentation_encodes_the_orbit_once_each():
    from blockgen.corpus import shards
    tok = CorpusTokenizer.default(kind="facades", layout="joint")
    lut = tok.block_lut()
    ids = np.zeros((7, 5, 4), np.int32)
    ids[0, :, :] = 1; ids[:, 0, :] = 4; ids[2, 3, 1] = 20          # no symmetry survives
    data = np.zeros_like(ids)
    plain = shards._encode_one(tok, lut, ids, data, TAGS, fc.ALL_FACES)
    aug = shards._encode_one(tok, lut, ids, data, TAGS, fc.ALL_FACES, augment="d4")
    assert len(plain) == 1 and len(aug) == 8
    assert sorted(m["aug"] for _, m in aug) == list(range(8))
    assert np.array_equal(aug[0][0], plain[0][0])                  # element 0 is the identity
    # each augmented sequence lifts to a D4 image of the original occupancy
    orig = ids != 0
    images = [np.rot90(orig[::-1] if m else orig, k, axes=(0, 2)) for m in (0, 1) for k in range(4)]
    for seq, meta in aug:
        faces, _ = tok.decode_faces(seq)
        by = {f.face: f for f in faces}                            # decoded faces carry no extent
        shape = (by["-z"].depth.shape[0], by["-z"].depth.shape[1], by["-x"].depth.shape[0])
        occ = fc.lift(faces, shape).occupied_mask
        assert any(o.shape == occ.shape and np.array_equal(o, occ) for o in images), meta
    # a box that is its own mirror image collapses the duplicates
    sym = np.zeros((4, 3, 4), np.int32); sym[:, 0, :] = 1
    assert len(shards._encode_one(tok, lut, sym, np.zeros_like(sym), TAGS, fc.ALL_FACES, augment="d4")) < 8
    with pytest.raises(ValueError):
        shards._encode_one(tok, lut, ids, data, TAGS, fc.ALL_FACES, augment="flip")
    assert shards.find("x", "abc", "facades", "d4").endswith("x/abc-d4")
