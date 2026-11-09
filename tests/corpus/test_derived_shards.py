"""Derived tiers (shards.from_structures), the staleness guard and the empty-tier error."""
import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from blockgen.corpus import facade as fc
from blockgen.corpus import shards, tiers
from blockgen.corpus.tokenizer import CorpusTokenizer
from blockgen.train.data import open_shards
from blockgen.utils.data import Structure


def _house(seed=0, pad=2):
    """A small house with a doorway, a window and a recess, embedded in air so the crop matters."""
    rng = np.random.default_rng(seed)
    ids = np.zeros((9 + pad, 6 + pad, 7 + pad), np.int32)
    ids[1:8, 0, 1:6] = 1
    ids[1, 0:5, 1:6] = ids[7, 0:5, 1:6] = 5
    ids[1:8, 0:5, 1] = ids[1:8, 0:5, 5] = 5
    ids[1:8, 5, 1:6] = 1
    ids[4, 1:3, 1] = 0
    ids[2, 2, 1] = 20
    ids[3, 4, 1] = 0; ids[3, 4, 2] = 5
    extra = rng.integers(1, 7, 3)              # a few blocks that differ per seed
    ids[2 + extra[0], 5, 2 + extra[1] % 3] = 1
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


@pytest.fixture
def corpus_dirs(tmp_path, monkeypatch):
    """Shards, tiers and tier configs under tmp so nothing touches data/."""
    shard_dir, tier_dir, cfg_dir = (str(tmp_path / d) for d in ("shards", "tiers", "configs"))
    os.makedirs(cfg_dir)
    monkeypatch.setattr(shards, "SHARD_DIR", shard_dir)
    monkeypatch.setattr(shards, "TIER_DIR", tier_dir)
    monkeypatch.setattr(tiers, "TIER_DIR", tier_dir)
    monkeypatch.setattr(tiers, "CONFIG_DIR", cfg_dir)
    return shard_dir, tier_dir, cfg_dir


def _tight(s):
    return shards._tight(np.asarray(s.block_ids), np.asarray(s.block_data))


def _json(path):
    with open(path) as fh:
        return json.load(fh)


def test_from_structures_voxels_round_trip(corpus_dirs):
    shard_dir, tier_dir, _ = corpus_dirs
    tok = CorpusTokenizer.default()
    structs = [_house(i) for i in range(3)]
    out = shards.from_structures(structs, "rft-test", tok, source="unit")
    assert out == os.path.join(shard_dir, "rft-test", tok.hash)
    assert sorted(os.listdir(out)) == ["meta.parquet", "offsets.npy", "spec.json", "tokenizer.json", "tokens.u16"]

    spec = _json(os.path.join(out, "spec.json"))
    for k in ("kind", "layout", "faces", "augment", "tier", "source", "source_sha256", "n_structures"):
        assert k in spec, k
    assert spec["kind"] == "voxels" and spec["tier"] == "rft-test" and spec["source"] == "unit"
    assert spec["n_structures"] == 3 and spec["n_seqs"] == 3 and spec["skipped"] == {k: 0 for k in shards.SKIP_REASONS}

    rec = _json(os.path.join(tier_dir, "rft-test.json"))
    assert rec["name"] == "rft-test" and rec["kind"] == "derived" and rec["source"] == "unit" and rec["n"] == 3
    assert rec["sha256"] == spec["tier_sha256"] == spec["source_sha256"]
    with open(os.path.join(tier_dir, "rft-test.ids.txt")) as fh:
        kept = fh.read().split()
    assert len(kept) == 3 and rec["sha256"] == tiers._ids_hash(kept)

    # find resolves it both by hash and as the newest voxel shard, and open_shards loads it unchanged
    assert shards.find("rft-test", tok.hash) == out and shards.find("rft-test") == out
    opened, w, h = open_shards([("rft-test", 1.0)], tok.hash)
    assert h == tok.hash and len(opened) == 1 and len(opened[0]) == 3
    meta = opened[0].meta()
    assert meta.src.tolist() == ["curated"] * 3 and meta.q.tolist() == ["4"] * 3
    assert meta.win.tolist() == ["whole"] * 3 and meta["size"].tolist() == ["32"] * 3
    assert meta.id.tolist() == kept and meta.source.tolist() == ["unit"] * 3
    for i, s in enumerate(structs):
        ids, data = _tight(s)
        back_ids, back_data, tags = tok.decode(opened[0][i])
        assert tags == {"src": "curated", "q": "4", "size": "32", "win": "whole"}
        assert back_ids.shape == ids.shape and np.array_equal(back_ids, ids)
        assert np.array_equal(back_data, data)
        assert meta.n_blocks[i] == np.count_nonzero(ids)
        # the sequence is the one a whole manifest build of the same size would get
        assert np.array_equal(opened[0][i], tok.encode(ids, data, tags))


def test_from_structures_facades_round_trip(corpus_dirs):
    structs = [_house(i) for i in range(3)]
    for layout in ("per_face", "joint"):
        tok = CorpusTokenizer.default(kind="facades", layout=layout)
        tier = f"rft-faces-{layout}"
        out = shards.from_structures(structs, tier, tok, faces="all")
        spec = _json(os.path.join(out, "spec.json"))
        assert spec["kind"] == "facades" and spec["layout"] == layout and spec["faces"] == list(fc.ALL_FACES)
        opened, _, h = open_shards([(tier, 1.0)], tok.hash, kind="facades")
        assert h == tok.hash
        sh = opened[0]
        meta = sh.meta()
        if layout == "joint":
            assert len(sh) == 3 and meta.face.tolist() == ["joint"] * 3
            for i, s in enumerate(structs):
                ids, _ = _tight(s)
                faces, tags = tok.decode_faces(sh[i])
                assert tags["src"] == "curated" and tags["win"] == "whole"
                proj = fc.project_all(Structure(block_ids=ids, block_data=np.zeros_like(ids)), fc.ALL_FACES)
                assert [f.face for f in faces] == list(proj)
                for f in faces:
                    assert np.array_equal(f.depth, proj[f.face].depth) and np.array_equal(f.ids, proj[f.face].ids)
                lifted = fc.lift(faces, ids.shape)
                occ = lifted.block_ids != 0
                assert occ.any() and np.all(ids[occ] != 0)   # the lift is a non-empty subset of the build
        else:
            assert len(sh) == 3 * len(fc.ALL_FACES)
            for j, (_, row) in enumerate(meta.iterrows()):
                ids, _ = _tight(structs[j // len(fc.ALL_FACES)])
                f, tags = tok.decode_face(sh[j])
                ref = fc.project(Structure(block_ids=ids, block_data=np.zeros_like(ids)), row.face)
                assert f.face == row.face and tags["size"] == "32"
                assert np.array_equal(f.depth, ref.depth) and np.array_equal(f.ids, ref.ids)
                assert row.n_cells == int(ref.mask.sum())
        # the voxel kind does not pick up the facade shard by being newest
        with pytest.raises(FileNotFoundError):
            shards.find(tier)


def test_from_structures_tags_skips_and_name_guard(corpus_dirs):
    _, tier_dir, cfg_dir = corpus_dirs
    tok = CorpusTokenizer.default()
    structs = [_house(0), Structure(block_ids=np.zeros((4, 4, 4), np.int32), block_data=np.zeros((4, 4, 4), np.int32))]
    tiny = np.zeros((3, 3, 3), np.int32); tiny[0, 0, 0] = 1
    structs.append(Structure(block_ids=tiny, block_data=np.zeros_like(tiny)))
    big = np.zeros((60, 4, 4), np.int32); big[:, 0, :] = 1
    structs.append(Structure(block_ids=big, block_data=np.zeros_like(big)))
    cube = np.ones((10, 10, 10), np.int32)              # 1000 blocks: dense under the max_blocks below
    dense = Structure(block_ids=cube, block_data=np.zeros_like(cube), metadata={"id": "sample-0007"})
    structs.append(dense)
    named = _house(0); named.metadata = {"id": "sample-0001"}
    assert np.count_nonzero(named.block_ids) < 500
    structs[0] = named
    out = shards.from_structures(structs, "rft-tags", tok, tags={"src": "archive", "q": 2}, max_blocks=500)
    with open(os.path.join(out, "spec.json")) as fh:
        spec = json.load(fh)
    assert spec["n_structures"] == 5 and spec["n_seqs"] == 1 and spec["max_blocks"] == 500
    assert spec["skipped"] == {"empty": 1, "small": 1, "large": 1, "dense": 1}
    meta = shards.Shard(out).meta()
    assert meta.src.tolist() == ["archive"] and meta.q.tolist() == ["2"]
    # metadata["id"] is the id in meta and ids.txt; the dense house's id is not kept
    assert meta.id.tolist() == ["sample-0001"]
    with open(os.path.join(tier_dir, "rft-tags.ids.txt")) as fh:
        assert fh.read().split() == ["sample-0001"]
    with open(os.path.join(tier_dir, "rft-tags.json")) as fh:
        assert json.load(fh)["n"] == 1
    # without max_blocks the cube is kept, and a structure with no metadata id gets its content hash
    out2 = shards.from_structures([Structure(block_ids=cube, block_data=np.zeros_like(cube))], "rft-tags2", tok)
    m2 = shards.Shard(out2).meta()
    assert m2.id.tolist() == [shards._content_hash(cube, np.zeros_like(cube))] and m2.n_blocks.tolist() == [1000]
    with pytest.raises(ValueError):
        shards.from_structures([_house(0)], "rft-bad", tok, tags={"q": "9"})
    with pytest.raises(ValueError):
        shards.from_structures([_house(0)], "rft-bad", tok, tags={"colour": "red"})
    # a derived tier may not take a configured tier's name, nor overwrite a manifest tier
    with open(os.path.join(cfg_dir, "sft-houses32.yaml"), "w") as fh:
        fh.write("description: x\n")
    with pytest.raises(ValueError):
        shards.from_structures([_house(0)], "sft-houses32", tok)
    with open(os.path.join(tier_dir, "anneal.json"), "w") as fh:
        json.dump({"name": "anneal", "sha256": "0" * 64}, fh)
    with pytest.raises(ValueError):
        shards.from_structures([_house(0)], "anneal", tok)


def test_from_npz_cli(corpus_dirs, tmp_path):
    shard_dir, tier_dir, _ = corpus_dirs
    structs = [_house(i) for i in range(3)]
    npz = str(tmp_path / "samples.npz")
    np.savez(npz, block_ids=np.array([s.block_ids for s in structs], dtype=object),
             block_data=np.array([s.block_data for s in structs], dtype=object))
    tok = CorpusTokenizer.default()
    shards.main(["from-npz", npz, "--tier", "rft-cli", "--tokenizer", tok.hash, "--tags", "q=3", "--source", "cli"])
    out = os.path.join(shard_dir, "rft-cli", tok.hash)
    sh = shards.Shard(out)
    assert len(sh) == 3 and sh.meta().q.tolist() == ["3"] * 3 and sh.spec["source"] == "cli"
    assert _json(os.path.join(tier_dir, "rft-cli.json"))["kind"] == "derived"
    with pytest.raises(ValueError):
        shards.resolve_tokenizer("deadbeef0000")


def test_stale_shard_warns_and_fresh_does_not(corpus_dirs):
    _, tier_dir, _ = corpus_dirs
    tok = CorpusTokenizer.default()
    out = shards.from_structures([_house(i) for i in range(3)], "rft-stale", tok)
    with warnings.catch_warnings():
        warnings.simplefilter("error", shards.StaleShardWarning)
        shards.Shard(out)                     # fresh: no warning
    p = os.path.join(tier_dir, "rft-stale.json")
    with open(p) as fh:
        rec = json.load(fh)
    old = rec["sha256"]; rec["sha256"] = "f" * 64
    with open(p, "w") as fh:
        json.dump(rec, fh)
    with pytest.warns(shards.StaleShardWarning) as w:
        sh = shards.Shard(out)
    assert len(sh) == 3
    msg = str(w[0].message)
    assert old[:12] in msg and "f" * 12 in msg and "rft-stale" in msg
    # a tier json that vanished is not staleness (derived tiers may be pruned)
    os.remove(p)
    with warnings.catch_warnings():
        warnings.simplefilter("error", shards.StaleShardWarning)
        shards.Shard(out)


def _frame(ids):
    n = len(ids)
    return pd.DataFrame({"id": ids, "source": ["grabcraft"] * n, "world_id": [f"w{i}" for i in range(n)],
                         "n_blocks": [100] * n, "is_rep": [True] * n, "cluster": [-1] * n,
                         "cluster_worlds": [1] * n, "kind": ["build"] * n, "mine_share": [0.0] * n,
                         "max_dim": [20] * n})


def test_materialise_raises_on_empty_split_tier(corpus_dirs, tmp_path, monkeypatch):
    _, tier_dir, _ = corpus_dirs
    monkeypatch.setattr(tiers, "ROOT", str(tmp_path))
    split = {"corpus": "houses_32", "train": [0, 1], "val": [2], "test": [3]}
    with open(tmp_path / "split.json", "w") as fh:
        json.dump(split, fh)
    cfg = {"holdout_split": "split.json", "holdout_parts": ["val", "test"], "dedup": False}
    # the manifest has no houses_32 rows: the holdout tier would be empty
    with pytest.raises(tiers.EmptyTierError) as e:
        tiers.materialise("eval-holdout", cfg, _frame(["cache:all_32:0", "cache:all_32:1"]), {})
    assert "houses_32" in str(e.value) and "0 rows" in str(e.value) and "manifest curated" in str(e.value)
    assert not os.path.exists(os.path.join(tier_dir, "eval-holdout.json"))
    # the cache is present but the tier's own filter removed every row: the advice names the filter, not the cache
    with pytest.raises(tiers.EmptyTierError) as e:
        tiers.materialise("eval-holdout", {**cfg, "filter": "n_blocks > 1000"},
                          _frame([f"cache:houses_32:{i}" for i in range(5)]), {})
    assert "5 rows" in str(e.value) and "filter" in str(e.value) and "manifest curated" not in str(e.value)
    # the sft parts too
    with pytest.raises(tiers.EmptyTierError):
        tiers.materialise("sft-houses32", {**cfg, "holdout_parts": ["train"]}, _frame(["cache:all_32:0"]), {})
    # with the cache present the tier materialises as before
    rec = tiers.materialise("eval-holdout", cfg, _frame([f"cache:houses_32:{i}" for i in range(5)]), {})
    assert rec["n"] == 2 and rec["sha256"] == tiers._ids_hash(["cache:houses_32:2", "cache:houses_32:3"])
    # a filter tier with no rows still writes an empty tier (it has no pinned source to be missing)
    rec = tiers.materialise("vanilla", {"sources": ["worldcrop"], "filter": "kind == 'village'"}, _frame(["a"]), {})
    assert rec["n"] == 0


def test_from_structures_all_skipped_leaves_nothing(corpus_dirs):
    shard_dir, tier_dir, _ = corpus_dirs
    tok = CorpusTokenizer.default()
    empty = Structure(block_ids=np.zeros((3, 3, 3), np.int32), block_data=np.zeros((3, 3, 3), np.int32))
    tiny = np.zeros((3, 3, 3), np.int32); tiny[0, 0, 0] = 1
    with pytest.raises(ValueError, match="rft-empty: none of 2 structures kept"):
        shards.from_structures([empty, Structure(block_ids=tiny, block_data=np.zeros_like(tiny))], "rft-empty", tok)
    assert not os.path.exists(os.path.join(shard_dir, "rft-empty"))
    assert not os.path.exists(os.path.join(tier_dir, "rft-empty.json"))
    assert not os.path.exists(os.path.join(tier_dir, "rft-empty.ids.txt"))
    with pytest.raises(FileNotFoundError):
        shards.find("rft-empty")
    # a sibling shard of the same tier survives when a later call keeps nothing
    out = shards.from_structures([_house(0)], "rft-keep", tok)
    with pytest.raises(ValueError):
        shards.from_structures([empty], "rft-keep", CorpusTokenizer.default(oriented=True))
    assert os.path.isdir(out) and os.path.exists(os.path.join(tier_dir, "rft-keep.json"))


def test_from_structures_d4_augment(corpus_dirs):
    shard_dir, _, _ = corpus_dirs
    tok = CorpusTokenizer.default()
    structs = [_house(i) for i in range(2)]
    out = shards.from_structures(structs, "rft-d4", tok, augment="d4")
    assert out == os.path.join(shard_dir, "rft-d4", tok.hash + "-d4")
    sh = shards.Shard(out)
    assert sh.spec["augment"] == "d4" and len(sh) == 16            # the houses are asymmetric: eight each
    meta = sh.meta()
    assert sorted(meta.aug.tolist()) == sorted(list(range(8)) * 2)
    assert meta.id.nunique() == 2 and meta.win.tolist() == ["whole"] * 16
    # the identity element decodes to the build; find resolves by augment and the plain lookup does not see it
    ids, data = _tight(structs[0])
    row = meta[(meta.id == meta.id[0]) & (meta.aug == 0)].index[0]
    back_ids, back_data, _ = tok.decode(sh[int(row)])
    assert np.array_equal(back_ids, ids) and np.array_equal(back_data, data)
    assert shards.find("rft-d4", augment="d4") == out
    with pytest.raises(FileNotFoundError):
        shards.find("rft-d4")
    with pytest.raises(ValueError):
        shards.from_structures(structs, "rft-d4", tok, augment="d8")
