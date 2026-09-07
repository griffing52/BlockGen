"""The raw source corpora: lazy indexing, per-item reading, and dedup.

These skip when a corpus is not on disk. A fresh clone has none of them -- they
are 3.5 GB of scraped data that `data_sources.md` explains how to fetch -- and a
test suite that fails on their absence would be reporting on the machine rather
than on the code.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.lab import rawcorpora as rc
from blockgen.utils.data import Structure

PRESENT = [s.id for s in rc.SPECS if s.available()]
requires_raw = pytest.mark.skipif(not PRESENT, reason="no raw corpora on this machine")


@pytest.fixture(params=PRESENT or ["none"])
def spec_id(request):
    if not PRESENT:
        pytest.skip("no raw corpora on this machine")
    return request.param


# --- indexing ---------------------------------------------------------------
@requires_raw
def test_index_is_stable_across_rebuilds(spec_id):
    """An unstable index silently reattaches every stored label to a different
    build, which is worse than losing them -- the labels would still look valid.

    Stability is the invariant, not string-sortedness: the walk sorts `Path`
    objects component-wise, so `cartoon-characters/ash.json` precedes
    `cartoon-characters-183/baymax.json` where a string sort reverses them.
    What has to hold is that a rebuild from disk reproduces it exactly.
    """
    cached = rc.index(spec_id)
    assert cached == rc.index(spec_id)
    spec = rc.BY_ID[spec_id]
    assert rc.build_index(spec, force=True) == cached, "rebuild changed the order"


@requires_raw
def test_index_is_cached_to_disk(spec_id):
    rc.index(spec_id)
    assert rc.hash_index_path(spec_id).parent.is_dir()
    assert rc._index_path(spec_id).is_file()


@requires_raw
def test_union_is_the_concatenation_of_its_parts():
    """Kept virtual rather than indexed separately, so it cannot drift."""
    total = sum(len(rc.index(s.id)) for s in rc.SPECS if s.available())
    assert len(rc.index(rc.UNION_ID)) == total
    assert rc.counts()[rc.UNION_ID] == total


def test_missing_corpus_indexes_empty_rather_than_raising(tmp_path):
    spec = rc.RawSpec("ghost", "Ghost", tmp_path / "nope", "**/*.json", "", "grabcraft")
    assert not spec.available()
    assert rc.build_index(spec) == []


# --- reading ----------------------------------------------------------------
@requires_raw
def test_reads_a_real_build_with_legacy_block_ids(spec_id):
    """Every reader must land in the same legacy `(id, data)` space, or the
    renderer paints builds in whatever blocks the integers collide with."""
    got = None
    for i in range(0, min(len(rc.index(spec_id)), 200), 7):
        got = rc.load_one(spec_id, i)
        if got is not None:
            break
    assert got is not None, f"{spec_id}: no readable build in the first 200"
    assert got.block_ids.ndim == 3
    assert got.occupied_mask.any()
    # Legacy ids are small; a token-space grid would run to the thousands.
    assert int(got.block_ids.max()) < 512, "looks like un-remapped token ids"


@requires_raw
def test_out_of_range_returns_none_not_an_exception(spec_id):
    assert rc.load_one(spec_id, 10 ** 9) is None
    assert rc.load_one(spec_id, -1) is None


def test_unknown_corpus_returns_none():
    assert rc.load_one("not_a_corpus", 0) is None


# --- the lazy sequence ------------------------------------------------------
@requires_raw
def test_lazy_builds_presents_a_list_surface(spec_id):
    lb = rc.LazyBuilds(spec_id)
    assert len(lb) == len(rc.index(spec_id))
    first = lb[0]
    assert isinstance(first, Structure)
    assert lb[0] is first, "repeat access must hit the cache"
    assert isinstance(lb[0:3], list) and len(lb[0:3]) == 3


@requires_raw
def test_lazy_builds_yields_an_empty_structure_for_a_broken_entry():
    """These corpora are unfiltered scrapes -- text2mc's own index records
    2,700+ entries with zero blocks. Returning None would crash a grid on
    `None.shape`; an empty structure renders as an empty tile, which is true."""
    lb = rc.LazyBuilds("text2mc_h5") if "text2mc_h5" in PRESENT else None
    if lb is None:
        pytest.skip("text2mc not present")
    empties = [i for i in range(60) if not lb[i].occupied_mask.any()]
    assert empties, "expected some unreadable/empty entries in a raw scrape"
    assert lb[empties[0]].metadata.get("unreadable") == "1"


@requires_raw
def test_lazy_builds_bounds_its_cache(spec_id):
    lb = rc.LazyBuilds(spec_id, max_cached=4)
    for i in range(min(12, len(lb))):
        lb[i]
    assert len(lb._cache) <= 4


@requires_raw
def test_lazy_builds_rejects_an_index_past_the_end(spec_id):
    lb = rc.LazyBuilds(spec_id)
    with pytest.raises(IndexError):
        lb[len(lb)]


# --- dedup ------------------------------------------------------------------
def test_content_hash_ignores_metadata_and_tracks_content():
    ids = np.zeros((4, 4, 4), np.int32)
    ids[:, 0, :] = 5
    a = Structure(block_ids=ids, block_data=np.zeros_like(ids),
                  metadata={"title": "one"})
    b = Structure(block_ids=ids.copy(), block_data=np.zeros_like(ids),
                  metadata={"title": "two"})
    assert rc.content_hash(a) == rc.content_hash(b)
    c = ids.copy()
    c[0, 1, 0] = 4
    assert rc.content_hash(a) != rc.content_hash(
        Structure(block_ids=c, block_data=np.zeros_like(c)))


@requires_raw
def test_raw_corpora_are_registered_as_datasets():
    from tools.lab import catalog
    raw = {d.id: d for d in catalog.list_datasets() if d.kind == "raw"}
    assert raw, "raw corpora present on disk but not registered"
    assert f"raw:{rc.UNION_ID}" in raw
    for d in raw.values():
        assert d.n > 0
