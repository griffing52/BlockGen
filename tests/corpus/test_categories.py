"""Category conditioning: the cat tokenizer variant, the label rules, the category tiers and shards."""
import json
import os

import numpy as np
import pytest

from blockgen.corpus import labels as lb
from blockgen.corpus import shards, tiers
from blockgen.corpus.tokenizer import CAT_CLASSES, TOKENIZER_DIR, CorpusTokenizer
from blockgen.utils.data import Structure
from blockgen.utils.serialize import BOS_TOKEN, EOS_TOKEN

CAT_HASH = "116e68927d45"
TAGS = {"src": "curated", "q": "4", "size": "32", "win": "whole"}


def _house(seed=0):
    rng = np.random.default_rng(seed)
    ids = np.zeros((9, 6, 7), np.int32)
    ids[1:8, 0, 1:6] = 1
    ids[1, 0:5, 1:6] = ids[7, 0:5, 1:6] = 5
    ids[1:8, 0:5, 1] = ids[1:8, 0:5, 5] = 5
    ids[1:8, 5, 1:6] = 17
    ids[2 + rng.integers(0, 5), 5, 2] = 1
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


# --------------------------------------------------------------------------- tokenizer

def test_existing_hashes_are_unchanged():
    assert CorpusTokenizer.default().hash == "60684456e598"
    assert CorpusTokenizer.default(oriented=True).hash == "0e6ace66bcdf"
    assert CorpusTokenizer.default(kind="facades").hash == "a9546c1f2f94"
    assert CorpusTokenizer.default(kind="facades", layout="joint").hash == "a4ca45ac1fe5"
    learned = os.path.join(TOKENIZER_DIR, "cdce56584ebf.json")
    if os.path.exists(learned):
        assert CorpusTokenizer.load(learned).hash == "cdce56584ebf"


def test_cat_variant_hash_vocab_and_ids(tmp_path):
    base, cat = CorpusTokenizer.default(), CorpusTokenizer.default(cats=True)
    assert cat.hash == CAT_HASH and cat.hash != base.hash
    assert cat.cats == list(CAT_CLASSES) and cat.cats[-1] == "none"
    assert cat.cat_offset == base.vocab_size == 507 and cat.vocab_size == 507 + len(CAT_CLASSES)
    for g, vals in base.tags.items():                       # every base id means the same thing
        for v in vals:
            assert cat.tag_id(g, v) == base.tag_id(g, v)
    assert cat.decode_tag(cat.tag_id("cat", "ship")) == ("cat", "ship")
    assert cat.extends(base) and not base.extends(cat) and not cat.extends(CorpusTokenizer.default(oriented=True))
    assert cat.base().hash == base.hash
    again = CorpusTokenizer.load(cat.save(str(tmp_path / "tokenizer.json")))
    assert again.hash == CAT_HASH and again.cats == cat.cats


def test_class_token_replaces_bos_and_keeps_every_position():
    base, cat = CorpusTokenizer.default(), CorpusTokenizer.default(cats=True)
    s = _house()
    a = base.encode(s.block_ids, s.block_data, TAGS)
    b = cat.encode(s.block_ids, s.block_data, {**TAGS, "cat": "castle"})
    assert len(a) == len(b) and a[0] == BOS_TOKEN and b[0] == cat.tag_id("cat", "castle")
    assert np.array_equal(a[1:], b[1:]) and b[-1] == EOS_TOKEN
    ids, data, tags = cat.decode(b)
    assert tags == {**TAGS, "cat": "castle"}
    assert np.array_equal(ids, base.decode(a)[0]) and np.array_equal(data, base.decode(a)[1])
    # no cat tag: the unlabeled class; a base tokenizer ignores a cat tag entirely
    assert cat.encode(s.block_ids, s.block_data, TAGS)[0] == cat.tag_id("cat", "none")
    assert base.prefix({**TAGS, "cat": "ship"}) == base.prefix(TAGS)
    assert cat.prefix({**TAGS, "cat": "ship"}) == [cat.tag_id("cat", "ship")] + base.prefix(TAGS)[1:]


def test_cat_group_is_kind_agnostic():
    from blockgen.corpus import facade as fc
    for kind, layout in (("facades", "per_face"), ("facades", "joint")):
        base = CorpusTokenizer.default(kind=kind, layout=layout)
        cat = base.with_cats()
        assert cat.extends(base) and cat.cat_offset == base.vocab_size
        faces = list(fc.project_all(_house(), fc.ALL_FACES).values())
        a, b = base.encode_faces(faces, TAGS), cat.encode_faces(faces, {**TAGS, "cat": "tower"})
        assert np.array_equal(a[1:], b[1:]) and cat.decode_faces(b)[1]["cat"] == "tower"
        assert len(cat.decode_faces(b)[0]) == len(faces)
    learned = os.path.join(TOKENIZER_DIR, "cdce56584ebf.json")
    if os.path.exists(learned):
        p = CorpusTokenizer.load(learned)
        pc = p.with_cats()
        assert pc.extends(p) and pc.cat_offset == p.vocab_size and pc.piece_offset == p.piece_offset
        s = _house()
        a, b = p.encode(s.block_ids, s.block_data, TAGS), pc.encode(s.block_ids, s.block_data, {**TAGS, "cat": "tree"})
        assert np.array_equal(a[1:], b[1:])
        assert np.array_equal(pc.decode(b)[0] != 0, p.decode(a)[0] != 0) and pc.decode(b)[2]["cat"] == "tree"


def test_resolve_tokenizer_knows_the_cat_default():
    assert shards.resolve_tokenizer(CAT_HASH).hash == CAT_HASH


# --------------------------------------------------------------------------- label rules

@pytest.mark.parametrize("title,want", [
    ("Medieval Tree House 2", "house"), ("Ancient Lighthouse", "tower"), ("Dark Kingdom Small Airship", "aircraft"),
    ("Pirate Ship With Canons", "ship"), ("Fire Truck 1", "vehicle"), ("Creeper Statue", "statue"),
    ("Savanna Baobab Tree 2", "tree"), ("Medieval Church", "church"), ("Medieval Kingdom Stone Keep", "castle"),
    ("Small Wizard Tower 2", "tower"), ("Large Medieval Windmill", "farm"), ("Medieval Village Tavern", "shop"),
    ("Nordic Small House 2", "house"), ("Adidas Logo", "pixelart"), ("Planetarium", None),
])
def test_title_rules(title, want):
    hit = lb.classify_text(title)
    assert (hit[0] if hit else None) == want


def test_classify_order_and_fallbacks():
    assert lb.classify("Douglas C-47 Skytrain", "planes")[:2] == ("aircraft", "subcat")
    assert lb.classify("", "house")[:2] == ("house", "subcat")                          # 3D-Craft route
    assert lb.classify("Thing", "other-190", ["natural looking tree"])[:2] == ("tree", "tags")
    assert lb.classify("", "", None, "medieval inn")[:2] == ("shop", "vlm")
    assert lb.classify("", "", None, "", house_cache=True)[:2] == ("house", "cache")
    assert lb.classify("Batman", "superheroes-176", min_edge=1)[:2] == ("pixelart", "flat")
    assert lb.classify("Batman", "superheroes-176", min_edge=6)[0] == "statue"
    assert lb.classify("Gong", "other-190")[0] == "none"
    assert set(lb.CLASSES) == set(CAT_CLASSES) and lb.CLASSES == CAT_CLASSES


def test_build_key_and_quota():
    a = lb.build_key("3dcraft", "/x/BlockGen/data/3d_craft/houses/workdir.2018-06-14-07:08:27.ip-1")
    b = lb.build_key("3dcraft", "/x/BlockGen/data/minecraft/3d_craft/houses/workdir.2018-06-14-07:08:27.ip-1")
    assert a == b
    assert lb.build_key("grabcraft", "https://www.grabcraft.com/minecraft/x/boats/") == \
        "https://www.grabcraft.com/minecraft/x/boats"
    assert lb.eval_quota(100) == 32 and lb.eval_quota(400) == 60 and lb.eval_quota(5000) == 128


# --------------------------------------------------------------------------- shards

def test_from_structures_with_a_class(tmp_path, monkeypatch):
    shard_dir, tier_dir, cfg_dir = (str(tmp_path / d) for d in ("shards", "tiers", "configs"))
    os.makedirs(cfg_dir)
    for mod in (shards, tiers):
        monkeypatch.setattr(mod, "TIER_DIR", tier_dir)
    monkeypatch.setattr(shards, "SHARD_DIR", shard_dir)
    monkeypatch.setattr(tiers, "CONFIG_DIR", cfg_dir)
    cat = CorpusTokenizer.default(cats=True)
    out = shards.from_structures([_house(i) for i in range(3)], "rft-cat", cat, tags={"cat": "tower"}, source="unit")
    sh = shards.Shard(out)
    assert sh.spec["cats"] is True and sh.spec["by_cat"] == {"tower": 3}
    assert set(sh.meta().cat) == {"tower"} and all(sh[i][0] == cat.tag_id("cat", "tower") for i in range(len(sh)))
    with pytest.raises(ValueError, match="cat"):
        shards.from_structures([_house()], "rft-bad", cat, tags={"cat": "spaceship"})
    # a plain shard of the same tier wins find() without a hash; the cat one by its hash
    plain = shards.from_structures([_house(i) for i in range(2)], "rft-cat", CorpusTokenizer.default(), source="unit")
    os.utime(os.path.join(out, "spec.json"))                 # the cat shard is the newest
    assert shards.find("rft-cat") == plain and shards.find("rft-cat", CAT_HASH) == out


# --------------------------------------------------------------------------- the built tiers (data on disk)

def _tier_ids(name):
    p = os.path.join(tiers.TIER_DIR, f"{name}.ids.txt")
    if not os.path.exists(p):
        pytest.skip(f"tier {name} not built")
    return set(open(p).read().split())


def test_category_tiers_do_not_leak():
    if not os.path.exists(lb.LABELS):
        pytest.skip("labels sidecar not built")
    sft, ev, hold, h32 = (_tier_ids(t) for t in ("sft-cat", "eval-cat", "eval-holdout", "sft-houses32"))
    assert not (sft & ev) and not (sft & hold) and not (ev & h32)
    import pandas as pd
    lab = pd.read_parquet(lb.LABELS)
    s, e = lab[lab.id.isin(sft)], lab[lab.id.isin(ev)]
    assert not set(s.build_key) & set(e.build_key)
    assert set(e.cat) <= set(lb.CLASSES) - {"none"} and (s.cat_split == "train").all() and (e.cat_split == "eval").all()
    df = tiers._frame(aux=lb.AUX)
    rows = df.set_index("id")
    assert not set(rows.loc[list(ev), "canon_hash"]) & set(rows.loc[list(sft), "canon_hash"])
    spec = json.load(open(os.path.join(tiers.TIER_DIR, "eval-cat.json")))
    assert all(32 <= n <= 128 for n in spec["by_cat"].values())
    # the house class of the eval set comes from eval-holdout, which no tier trains on
    assert {i for i in ev if lab.set_index("id").loc[i, "cat"] == "house"} <= hold
