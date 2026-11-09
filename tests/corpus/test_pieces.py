"""The pieces (3D-BPE) tokenizer kind: artifact, round trip, shards and the sampler's decode."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from blockgen.corpus import shards, tiers
from blockgen.corpus import tokenizer as tkn
from blockgen.corpus.tokenizer import CorpusTokenizer
from blockgen.tokenizers.cluster_bpe import _apply_merge, _atomic_labeling, learn_clusters, structure_to_cluster_tokens
from blockgen.utils.data import Structure
from blockgen.utils.serialize import BOS_TOKEN, EOS_TOKEN

TAGS = {"src": "curated", "q": "4", "size": "32", "win": "whole"}


def _house(seed=0):
    """Walls of planks, a stone floor and roof, a door and a bed, so forced and learned merges both fire."""
    rng = np.random.default_rng(seed)
    ids = np.zeros((9, 7, 8), np.int32); data = np.zeros_like(ids)
    ids[:, 0, :] = 1
    ids[0, 1:6, :] = ids[8, 1:6, :] = 5
    ids[:, 1:6, 0] = ids[:, 1:6, 7] = 5
    ids[:, 6, :] = 4
    ids[4, 1, 0], data[4, 1, 0] = 64, 0         # a door: two halves, one piece
    ids[4, 2, 0], data[4, 2, 0] = 64, 8
    ids[2, 1, 2], ids[2, 1, 3] = 26, 26         # a bed
    data[2, 1, 2], data[2, 1, 3] = 0, 8
    ids[6, 3, 0] = 20
    ids[1 + rng.integers(7), 1, 1 + rng.integers(6)] = 17
    return ids, data


@pytest.fixture(scope="module")
def tok():
    """A pieces tokenizer learned from three toy houses (a handful of merges)."""
    t = CorpusTokenizer.default(kind="pieces")
    lut = t.block_lut()
    structs = [t.canonical(*_house(i), lut) for i in range(3)]
    cv = learn_clusters(structs, max_dim=t.max_dim, n_merges=6, max_corpus=10, min_count=1, base=t.vocab, verbose=False)
    t.pieces = {"merges": [[pa, pb, list(d)] for pa, pb, d in cv.merges], "fit_tier": "toy"}
    t._cv = None
    return t


def test_voxel_and_facade_hashes_unchanged():
    assert CorpusTokenizer.default().hash == "60684456e598"
    assert CorpusTokenizer.default(oriented=True).hash == "0e6ace66bcdf"
    assert CorpusTokenizer.default(kind="facades").hash == "a9546c1f2f94"
    assert CorpusTokenizer.default(kind="facades", layout="joint").hash == "a4ca45ac1fe5"
    assert "pieces" not in CorpusTokenizer.default().to_json()


def test_pieces_artifact_layout_and_hash(tok, tmp_path):
    assert tok.n_learned == len(tok.pieces["merges"]) > 2
    vox = CorpusTokenizer.default()
    # blocks, coordinates and tags keep the voxel ids; learned pieces follow the tags
    assert tok.tag_offset == vox.tag_offset and tok.piece_offset == vox.vocab_size
    assert tok.vocab_size == vox.vocab_size + tok.n_learned
    loaded = CorpusTokenizer.load(tok.save(str(tmp_path / "tok.json")))
    assert loaded.hash == tok.hash and loaded.kind == "pieces" and loaded.pieces == json.loads(json.dumps(tok.pieces))
    assert loaded.cluster_vocab().patterns == tok.cluster_vocab().patterns
    other = CorpusTokenizer.load(str(tmp_path / "tok.json"))
    other.pieces = {**other.pieces, "merges": other.pieces["merges"][:-1]}
    assert other.hash != tok.hash                          # the merges are in the hash
    assert CorpusTokenizer.default(kind="pieces").hash != vox.hash


def test_round_trip_equals_the_voxel_decode(tok):
    vox = CorpusTokenizer.default()
    lut = tok.block_lut()
    for seed in range(4):                                  # seed 3 was not in the fit set
        ids, data = _house(seed)
        seq = tok.encode(ids, data, TAGS, lut)
        assert seq[0] == BOS_TOKEN and seq[-1] == EOS_TOKEN and int(seq.max()) < tok.vocab_size
        body = seq[1 + len(TAGS):-1]
        assert len(body) % 4 == 0 and (body >= tok.piece_offset).any()
        assert len(seq) < len(vox.encode(ids, data, TAGS, lut))
        a_ids, a_data, a_tags = tok.decode(seq)
        b_ids, b_data, _ = vox.decode(vox.encode(ids, data, TAGS, lut))
        assert a_tags == TAGS and a_ids.shape == b_ids.shape
        assert (a_ids == b_ids).all() and (a_data == b_data).all()


def test_anchor_order_is_the_legacy_one_and_skipping_no_op_merges_changes_nothing(tok):
    cv = tok.cluster_vocab()
    s = tok.canonical(*_house(1))
    fast = structure_to_cluster_tokens(s, cv)
    inst, owner, nid = _atomic_labeling(s, cv_base(cv))  # every merge replayed, none skipped
    for pid, (pa, pb, d) in enumerate(cv.merges, start=cv.num_blocks):
        nid = _apply_merge(inst, owner, nid, pa, pb, d, pid)
    order = sorted(inst.values(), key=lambda d: (d["anchor"][1], d["anchor"][2], d["anchor"][0]))
    slow = [BOS_TOKEN] + [t for d in order for t in (*[cv.coord_id(c) for c in d["anchor"]], cv.piece_id(d["piece"]))]
    assert fast == slow + [EOS_TOKEN]


def cv_base(cv):
    from blockgen.utils.data import _token_for
    from blockgen.utils.serialize import BlockVocab
    t2i = {}
    for i, p in enumerate(cv.block_index_to_pair):
        t2i.setdefault(_token_for(*p), i)
    return BlockVocab(max_dim=cv.max_dim, block_token_to_id=t2i, id_to_block_token=list(t2i),
                      block_index_to_pair=cv.block_index_to_pair)


def test_decode_survives_malformed_output(tok):
    vox = tok.vocab
    learned = tok.piece_offset                             # the first learned piece
    seq = tok.prefix(TAGS) + [vox.coord_id(2), vox.coord_id(0), vox.coord_id(1), learned,
                              vox.coord_id(5), tok.tag_id("q", "2"), vox.coord_id(1), vox.block_id_token(3),
                              vox.coord_id(0), vox.coord_id(0), vox.coord_id(0), vox.block_id_token(3), EOS_TOKEN, 7]
    ids, data, tags = tok.decode(np.array(seq))
    pat = tok.cluster_vocab().patterns[tok.cluster_vocab().num_blocks]
    assert tags == TAGS and int((ids != 0).sum()) == len(pat) + 1
    empty, _, _ = tok.decode(np.array(tok.prefix(TAGS) + [EOS_TOKEN]))
    assert empty.shape == (1, 1, 1) and not empty.any()


def test_to_structures_and_sampling_on_pieces(tok):
    import torch
    from blockgen.train import archs
    from blockgen.train.config import TrainConfig
    from blockgen.train.sample import sample_structures, to_structures
    ids, data = _house(0)
    s, = to_structures([tok.encode(ids, data, TAGS)], tok, corpus="toy")
    assert s.metadata["tags"] == TAGS and int(s.occupied_mask.sum()) == int((ids != 0).sum())
    cfg = TrainConfig(d_model=32, nhead=4, layers=1, ff=64, max_seq_len=96, sample_max_blocks=8, top_k=0, kind="pieces")
    m = archs.get("ar2").build(cfg, tok)
    torch.manual_seed(0)
    structs, extra = sample_structures(m, tok, cfg, 3, seed=1)
    assert len(structs) + extra["n_empty"] == 3


@pytest.fixture
def corpus_dirs(tmp_path, monkeypatch):
    shard_dir, tier_dir, cfg_dir = (str(tmp_path / d) for d in ("shards", "tiers", "configs"))
    os.makedirs(cfg_dir)
    monkeypatch.setattr(shards, "SHARD_DIR", shard_dir)
    monkeypatch.setattr(shards, "TIER_DIR", tier_dir)
    monkeypatch.setattr(tiers, "TIER_DIR", tier_dir)
    monkeypatch.setattr(tiers, "CONFIG_DIR", cfg_dir)
    return tmp_path


def test_shard_build_with_a_learned_tokenizer(tok, corpus_dirs, monkeypatch):
    tmp = corpus_dirs
    rows = []
    for i in range(3):
        ids, data = _house(i)
        p = str(tmp / f"b{i}.npz")
        np.savez(p, ids=ids.astype(np.uint16), data=data.astype(np.uint8))
        rows.append({"id": f"b{i}", "source": "curated", "path": p, "n_blocks": int((ids != 0).sum()),
                     "p_built": None, "kind": "build"})
    monkeypatch.setattr(shards, "_tier_rows", lambda tier: ({"sha256": "x" * 64}, pd.DataFrame(rows)))
    tok_path = tok.save(str(tmp / "pieces.json"))
    with pytest.raises(ValueError, match="needs --tokenizer"):
        shards.build("toy", kind="pieces", workers=1)
    out = shards.build("toy", workers=1, limit=2, tokenizer=tok_path)
    assert os.path.basename(out) == tok.hash
    sh = shards.Shard(out)
    assert sh.spec["kind"] == "pieces" and len(sh) == 2 and sh.tokenizer.hash == tok.hash
    assert shards.find("toy", kind="pieces") == out
    m = sh.meta()
    for i, bid in enumerate(m.id):
        ids, data = _house(int(bid[1:]))
        assert (sh[i] == tok.encode(ids, data, TAGS)).all()
    # a hash resolves through corpus/tokenizers/<hash>.json
    monkeypatch.setattr(shards, "TOKENIZER_DIR", str(tmp))
    os.rename(tok_path, str(tmp / f"{tok.hash}.json"))
    assert shards.resolve_tokenizer(tok.hash).hash == tok.hash


def test_learn_pieces_refuses_a_holdout_tier():
    with pytest.raises(ValueError, match="holdout"):
        tkn.learn_pieces("eval-holdout")
