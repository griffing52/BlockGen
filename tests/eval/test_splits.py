"""The canonical split: deterministic, exhaustive, and family-tight.

The load-bearing test here is `test_no_group_leakage`. Under a naive random row
split, 35.8% of val items share a build family with train ("American Middle Class
House 10" vs "... 22"), which inflates the held-out-real floor and deflates every
memorization metric at once.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from blockgen.eval.bench import splits


_WORDS = ("Cottage", "Manor", "Tower", "Barn", "Chapel", "Forge", "Mill", "Lodge",
          "Villa", "Hut", "Keep", "Hall", "Shed", "Inn", "Store", "Stable")


def _unique_titles(n: int):
    """Titles that are genuinely distinct builds, not a numbered sibling series.

    `group_key` strips trailing digits on purpose, so "House 1".."House 300"
    is *one* family (see `test_numbered_series_is_one_family`). Fixtures that
    want n independent groups must vary the words, not just the counter.
    """
    return [f"{_WORDS[i % len(_WORDS)]} {'x' * (i // len(_WORDS) + 1)}" for i in range(n)]


def _manifest(titles, categories=None):
    cats = categories or ["cat"] * len(titles)
    return {"items": [{"title": t, "category": c, "source_path": f"p{i}"}
                      for i, (t, c) in enumerate(zip(titles, cats))]}


# --- grouping --------------------------------------------------------------
def test_siblings_share_a_group():
    a = {"title": "American Middle Class House 10", "category": "brick-houses"}
    b = {"title": "American Middle Class House 22", "category": "brick-houses"}
    assert splits.group_key(a) == splits.group_key(b)


def test_different_builds_do_not_collide():
    a = {"title": "Seashore Brick House 1", "category": "brick-houses"}
    b = {"title": "Medieval Blacksmith 1", "category": "medieval-houses"}
    assert splits.group_key(a) != splits.group_key(b)


def test_same_title_different_category_does_not_collide():
    a = {"title": "Small House 1", "category": "brick-houses"}
    b = {"title": "Small House 1", "category": "medieval-houses"}
    assert splits.group_key(a) != splits.group_key(b)


def test_numbered_series_is_one_family():
    """Trailing counters are stripped by design -- that IS the sibling rule.

    On the real corpus this yields 2054 groups over 2661 builds (1846 singletons,
    largest group 23 = the "Medieval Middle Class House 1..23" series), so the
    rule collapses genuine series without over-merging distinct builds.
    """
    m = _manifest([f"House {i}" for i in range(300)])
    assert splits.make_split("toy", seed=0, manifest=m).n_groups == 1


def test_untitled_rows_are_their_own_group():
    """An empty title must not create one giant bucket that swallows the corpus."""
    a = {"title": "", "category": "", "source_path": "a"}
    b = {"title": "", "category": "", "source_path": "b"}
    assert splits.group_key(a) != splits.group_key(b)


# --- partition properties --------------------------------------------------
def test_split_is_a_partition():
    m = _manifest(_unique_titles(200))
    s = splits.make_split("toy", seed=0, manifest=m)
    allidx = np.concatenate([s.train, s.val, s.test])
    assert sorted(allidx.tolist()) == list(range(200))


def test_no_group_leakage():
    titles = [f"Family {i // 4} House {i % 4}" for i in range(400)]
    m = _manifest(titles)
    s = splits.make_split("toy", seed=0, manifest=m)
    leak = splits.group_leakage(s, m)
    assert leak["n_leaked"] == 0
    assert leak["n_groups"] == 100


def test_split_is_deterministic():
    m = _manifest(_unique_titles(300))
    a = splits.make_split("toy", seed=0, manifest=m)
    b = splits.make_split("toy", seed=0, manifest=m)
    assert a.to_json() == b.to_json()


def test_seed_changes_the_split():
    m = _manifest(_unique_titles(300))
    a = splits.make_split("toy", seed=0, manifest=m)
    b = splits.make_split("toy", seed=1, manifest=m)
    assert a.train.tolist() != b.train.tolist()


def test_realized_fracs_are_close_to_target():
    m = _manifest(_unique_titles(1000))
    s = splits.make_split("toy", seed=0, fracs=(0.7, 0.15, 0.15), manifest=m)
    n = 1000
    assert abs(len(s.train) / n - 0.70) < 0.02
    assert abs(len(s.val) / n - 0.15) < 0.02
    assert abs(len(s.test) / n - 0.15) < 0.02


def test_large_families_do_not_break_the_partition():
    """One family bigger than a whole split must still land somewhere intact."""
    titles = ["Mega Family House"] * 60 + _unique_titles(40)
    m = _manifest(titles)
    s = splits.make_split("toy", seed=0, manifest=m)
    sides = [set(s.train.tolist()), set(s.val.tolist()), set(s.test.tolist())]
    mega = set(range(60))
    assert sum(bool(mega & side) for side in sides) == 1


def test_fracs_must_sum_to_one():
    with pytest.raises(ValueError):
        splits.make_split("toy", fracs=(0.5, 0.2, 0.2), manifest=_manifest(["a"]))


def test_empty_manifest_raises():
    with pytest.raises(ValueError):
        splits.make_split("toy", manifest={"items": []})


# --- persistence -----------------------------------------------------------
def test_source_sha_changes_with_corpus():
    a = _manifest(["A", "B"])
    b = _manifest(["A", "C"])
    assert splits.source_sha(a) != splits.source_sha(b)


def test_json_roundtrip():
    m = _manifest(_unique_titles(50))
    s = splits.make_split("toy", seed=3, manifest=m)
    back = splits.Split.from_json(json.loads(json.dumps(s.to_json())))
    assert back.key() == s.key()
    assert back.train.tolist() == s.train.tolist()
    assert back.source_sha == s.source_sha


def test_key_encodes_corpus_seed_and_fracs():
    s = splits.make_split("toy", seed=7, fracs=(0.6, 0.2, 0.2),
                          manifest=_manifest(["a", "b", "c"]))
    assert s.key() == "toy.s7.60-20-20.v1"


# --- against the real corpus ----------------------------------------------
@pytest.mark.slow
def test_real_corpus_split_has_no_leakage():
    s = splits.load_split("houses_32", seed=0)
    _, m = splits.load_corpus("houses_32")
    assert sum(s.sizes.values()) == len(m["items"])
    assert splits.group_leakage(s, m)["n_leaked"] == 0
