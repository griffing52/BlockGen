"""Derived datasets and the provenance tree.

Several of these pin failures this feature already produced. The recursion test
is the sharpest: resolving a subset needs its parent's size, the obvious way to
get that is `catalog.get_dataset`, and the catalog resolves subsets as part of
its own listing -- so the first working version recursed until the stack died
the moment a subset existed on disk. It resolved fine in isolation, which is
exactly why an isolated test would not have caught it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.lab import catalog, subsets, tree
from blockgen.utils.data import Structure


@pytest.fixture()
def lab(tmp_path, monkeypatch):
    """A lab rooted in tmp_path with one fake corpus of 40 builds.

    Builds alternate corpus/category and grow in block count so every filter
    dimension has something to bite on.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subsets, "SUBSET_DIR", tmp_path / "outputs/lab/subsets")
    monkeypatch.setattr(store_mod := __import__("tools.lab.store", fromlist=["x"]),
                        "DB_PATH", tmp_path / "outputs/lab/lab.db")
    catalog.forget()

    n = 40
    ids = np.empty(n, dtype=object)
    data = np.empty(n, dtype=object)
    meta = []
    for i in range(n):
        grid = np.zeros((4, 4, 4), dtype=np.int16)
        grid[:, : 1 + i % 4, :] = 1                     # 16, 32, 48 or 64 blocks
        ids[i], data[i] = grid, np.zeros_like(grid)
        meta.append({"source_path": f"toy/{i}", "corpus": "grabcraft" if i % 2 == 0
                     else "3dcraft", "category": "houses" if i % 4 else "towers",
                     "title": f"Build {i}", "url": "",
                     "dims": [4, 4, 4], "n_blocks": int((grid > 0).sum())})

    cache = tmp_path / "data/minecraft/cache"
    cache.mkdir(parents=True)
    # The real `save_house_cache` layout: `sources` is what tells the loader
    # this is a house cache rather than a build cache.
    np.savez_compressed(cache / "toy_16.npz", block_ids=ids, block_data=data,
                        sources=np.array([m["source_path"] for m in meta]),
                        corpus=np.array([m["corpus"] for m in meta]))
    (cache / "toy_16_manifest.json").write_text(json.dumps(
        {"max_dim": 16, "count": n, "items": meta,
         "report": {"quality_drops": {"too_flat": 3}}}))
    yield tmp_path
    catalog.forget()


def _mk(**kw):
    kw.setdefault("name", kw["id"])
    return subsets.Subset(**kw)


# --- rules -----------------------------------------------------------------
def test_field_rule_reads_the_manifest_not_the_npz(lab, monkeypatch):
    """A field filter must not open structures when a manifest can answer it.

    Loading is the difference between instant and minutes on a real corpus, so
    the fast path is a behaviour worth pinning, not an implementation detail.
    """
    monkeypatch.setattr(catalog, "load_builds",
                        lambda *a, **k: pytest.fail("field rule loaded structures"))
    sub = _mk(id="gc", parent="corpus:toy_16", rule={"corpus": ["grabcraft"]})
    assert len(subsets.resolve_indices(sub)) == 20


def test_rules_compose_and_narrow(lab):
    parent = "corpus:toy_16"
    only_gc = subsets.resolve_indices(_mk(id="a", parent=parent, rule={"corpus": ["grabcraft"]}))
    big = subsets.resolve_indices(_mk(id="b", parent=parent, rule={"min_blocks": 48}))
    both = subsets.resolve_indices(_mk(id="c", parent=parent,
                                       rule={"corpus": ["grabcraft"], "min_blocks": 48}))
    assert set(both) == set(only_gc) & set(big)
    assert 0 < len(both) < len(only_gc)


def test_random_sample_is_seeded_and_sorted(lab):
    sub = _mk(id="s", parent="corpus:toy_16", rule={"limit": 10, "seed": 3})
    first = subsets.resolve_indices(sub)
    assert first == sorted(first) and len(first) == 10
    assert first == subsets.resolve_indices(sub)                     # same process
    other = subsets.resolve_indices(_mk(id="s", parent="corpus:toy_16",
                                        rule={"limit": 10, "seed": 4}))
    assert first != other                                            # seed matters


def test_limit_larger_than_the_parent_keeps_everything(lab):
    sub = _mk(id="s", parent="corpus:toy_16", rule={"limit": 999})
    assert len(subsets.resolve_indices(sub)) == 40


def test_label_rule_reads_the_store(lab):
    from tools.lab.store import Store
    with Store() as s:
        for i in (1, 5, 9):
            s.set_label(f"corpus:toy_16:{i}", "good")
        s.set_label("corpus:toy_16:2", "bad")
    keep = subsets.resolve_indices(_mk(id="g", parent="corpus:toy_16",
                                       rule={"labels": ["good"]}))
    assert keep == [1, 5, 9]
    drop = subsets.resolve_indices(_mk(id="d", parent="corpus:toy_16",
                                       rule={"exclude_labels": ["bad"]}))
    assert 2 not in drop and len(drop) == 39


def test_a_field_rule_on_a_huge_unmanifested_parent_is_refused(lab, monkeypatch):
    """Better a clear error than minutes of silent per-file reads."""
    monkeypatch.setattr(subsets, "_SCAN_LIMIT", 5)
    monkeypatch.setattr(subsets, "_manifest_rows", lambda *a, **k: None)
    with pytest.raises(subsets.SubsetError, match="no manifest"):
        subsets.resolve_indices(_mk(id="x", parent="corpus:toy_16", rule={"min_blocks": 1}))


# --- validation ------------------------------------------------------------
@pytest.mark.parametrize("kw, match", [
    ({"id": "Bad Id", "parent": "corpus:toy_16"}, "lowercase"),
    ({"id": "ok", "parent": "corpus:nope"}, "unknown parent"),
    ({"id": "ok", "parent": "corpus:toy_16", "rule": {"corpsu": ["g"]}}, "unknown filter key"),
    ({"id": "ok", "parent": "corpus:toy_16", "rule": {"labels": ["great"]}}, "not one of"),
    ({"id": "ok", "parent": "corpus:toy_16", "rule": {"limit": 0}}, "positive"),
    ({"id": "ok", "parent": "corpus:toy_16", "mode": "indices", "rule": {}}, "non-empty"),
])
def test_save_rejects_bad_definitions(lab, kw, match):
    with pytest.raises(subsets.SubsetError, match=match):
        subsets.save(_mk(**kw), known=[d.id for d in catalog.list_datasets()])


def test_a_typo_in_a_filter_key_is_an_error_not_a_wider_selection(lab):
    """The failure worth an error message: `corpsu` would keep all 40 builds and
    look like a plausible answer forever."""
    known = [d.id for d in catalog.list_datasets()]
    with pytest.raises(subsets.SubsetError):
        subsets.save(_mk(id="t", parent="corpus:toy_16", rule={"corpsu": ["grabcraft"]}),
                     known=known)


def test_save_refuses_to_clobber_without_overwrite(lab):
    known = [d.id for d in catalog.list_datasets()]
    subsets.save(_mk(id="a", parent="corpus:toy_16", rule={}), known=known)
    with pytest.raises(subsets.SubsetError, match="already exists"):
        subsets.save(_mk(id="a", parent="corpus:toy_16", rule={}), known=known)
    subsets.save(_mk(id="a", parent="corpus:toy_16", rule={"limit": 5}),
                 known=known, overwrite=True)
    assert subsets.get("a").rule == {"limit": 5}


def test_a_corrupt_file_is_skipped_not_fatal(lab):
    known = [d.id for d in catalog.list_datasets()]
    subsets.save(_mk(id="good", parent="corpus:toy_16", rule={}), known=known)
    (subsets.SUBSET_DIR / "broken.json").write_text("{not json")
    assert [s.id for s in subsets.load_all()] == ["good"]


# --- nesting ---------------------------------------------------------------
def test_a_branch_of_a_branch_composes(lab):
    known = [d.id for d in catalog.list_datasets()]
    subsets.save(_mk(id="gc", parent="corpus:toy_16", rule={"corpus": ["grabcraft"]}),
                 known=known)
    catalog.forget()
    subsets.save(_mk(id="gc5", parent="subset:gc", rule={"limit": 5, "seed": 0}),
                 known=[d.id for d in catalog.list_datasets()])
    catalog.forget()
    builds = catalog.load_builds("subset:gc5")
    assert len(builds) == 5
    assert all((b.metadata or {}).get("corpus") == "grabcraft" for b in builds)


def test_listing_datasets_with_a_subset_on_disk_does_not_recurse(lab):
    """The catalog resolves subsets while building its own listing; asking the
    catalog for a parent from in there recursed until the stack died."""
    subsets.save(_mk(id="a", parent="corpus:toy_16", rule={"limit": 4}),
                 known=[d.id for d in catalog.list_datasets()])
    catalog.forget()
    ids = {d.id: d for d in catalog.list_datasets()}          # would RecursionError
    assert ids["subset:a"].n == 4


def test_a_subset_cannot_parent_itself(lab):
    with pytest.raises(subsets.SubsetError, match="own parent"):
        subsets.save(_mk(id="loop", parent="subset:loop", rule={}), known=())


def test_deleting_a_parent_leaves_the_child_visible_with_a_reason(lab):
    known = [d.id for d in catalog.list_datasets()]
    subsets.save(_mk(id="p", parent="corpus:toy_16", rule={"limit": 6}), known=known)
    catalog.forget()
    subsets.save(_mk(id="c", parent="subset:p", rule={}),
                 known=[d.id for d in catalog.list_datasets()])
    subsets.delete("p")
    catalog.forget()
    child = {d.id: d for d in catalog.list_datasets()}["subset:c"]
    assert child.n == 0 and "not found" in child.note


# --- freezing --------------------------------------------------------------
def test_a_frozen_branch_ignores_later_labels_and_a_live_one_does_not(lab):
    """The reason both modes exist: an experiment cites a set that cannot move."""
    from tools.lab.store import Store
    known = [d.id for d in catalog.list_datasets()]
    with Store() as s:
        for i in (0, 1, 2):
            s.set_label(f"corpus:toy_16:{i}", "good")

    live = _mk(id="live", parent="corpus:toy_16", rule={"labels": ["good"]})
    frozen = _mk(id="frozen", parent="corpus:toy_16", mode="indices",
                 rule={"indices": subsets.resolve_indices(live)})
    subsets.save(live, known=known)
    subsets.save(frozen, known=known)

    with Store() as s:
        s.set_label("corpus:toy_16:7", "good")
    assert len(subsets.resolve_indices(live)) == 4
    assert len(subsets.resolve_indices(frozen)) == 3


def test_frozen_indices_past_the_end_are_dropped(lab):
    """A parent can shrink under a frozen list; an out-of-range index would be
    an IndexError at browse time instead of a smaller, honest set."""
    sub = _mk(id="f", parent="corpus:toy_16", mode="indices",
              rule={"indices": [0, 5, 999]})
    assert subsets.resolve_indices(sub) == [0, 5]


# --- tree ------------------------------------------------------------------
def test_tree_counts_each_build_once(lab):
    """The flat list summed `raw:all` beside its own five children and every
    split beside its corpus, reporting ~178k builds where ~96k exist."""
    known = [d.id for d in catalog.list_datasets()]
    subsets.save(_mk(id="half", parent="corpus:toy_16", rule={"limit": 20}), known=known)
    catalog.forget()
    roots = tree.build()
    assert tree.totals(roots)["builds"] == 40              # not 60


def test_tree_hangs_a_subset_under_its_parent(lab):
    subsets.save(_mk(id="kid", parent="corpus:toy_16", rule={"limit": 3}),
                 known=[d.id for d in catalog.list_datasets()])
    catalog.forget()
    corpus = next(r for r in tree.build() if r.id == "corpus:toy_16")
    assert [c.id for c in corpus.children] == ["subset:kid"]
    assert corpus.children[0].rule                          # the branch says what it keeps


def test_corpus_provenance_is_annotation_not_an_edge(lab):
    """`toy_16` draws from two corpora at once; making that a tree edge would
    force a lie about one of them."""
    corpus = next(r for r in tree.build() if r.id == "corpus:toy_16")
    assert corpus.sources == {"grabcraft": 20, "3dcraft": 20}
    assert corpus.drops == {"too_flat": 3}
    assert all(c.kind == "subset" for c in corpus.children)
