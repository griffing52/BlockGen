"""The lab's data layer: ids, discovery, the render cache, and the store.

Two of these tests exist because of specific failures the repo has already paid
for. `test_render_is_not_blank` pins T25c -- a broken renderer returning uniform
frames scored better than a working one, so "it produced bytes" is not a passing
render. `test_parse_build_id_splits_from_the_right` pins the fact that dataset
ids contain colons (`split:houses_32:test`), which a left-split silently turns
into dataset "split" at index 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.lab import catalog, renders, store


# --- build ids -------------------------------------------------------------
def test_make_and_parse_round_trip():
    for dataset in ("arm:bench_arms__native_oriented_32", "corpus:houses_32",
                    "split:houses_32:test"):
        bid = catalog.make_build_id(dataset, 7)
        assert catalog.parse_build_id(bid) == (dataset, 7)


def test_parse_build_id_splits_from_the_right():
    """Split ids carry two colons; only the last one separates the index."""
    assert catalog.parse_build_id("split:houses_32:test:12") == ("split:houses_32:test", 12)


@pytest.mark.parametrize("bad", ["", "nocolon", "arm:x:", "arm:x:notanint",
                                 ":5", "arm:x:-1"])
def test_parse_build_id_rejects_junk(bad):
    with pytest.raises(ValueError):
        catalog.parse_build_id(bad)


# --- discovery -------------------------------------------------------------
def test_list_datasets_never_raises(tmp_path, monkeypatch):
    """A fresh clone has no outputs/ and no data/. That is an empty list, not a
    traceback -- the hub page has to render something."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(catalog, "OUTPUTS_ROOT", Path("outputs"))
    monkeypatch.setattr(catalog, "CORPUS_DIRS", (Path("data/cache"),))
    assert catalog.list_datasets() == []
    assert catalog.list_scorecards() == []
    assert catalog.get_dataset("corpus:houses_32") is None


def test_npz_without_manifest_is_not_a_dataset(tmp_path, monkeypatch):
    """`outputs/run_*/*/samples.npz` from the T-series attachment runs are bare
    arrays; `load_structures_from_cache` raises on them, so they must not appear
    as rows that break when clicked."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(catalog, "OUTPUTS_ROOT", Path("outputs"))
    monkeypatch.setattr(catalog, "CORPUS_DIRS", ())
    d = tmp_path / "outputs" / "run_20260101_000000_x" / "sub"
    d.mkdir(parents=True)
    np.savez(d / "samples.npz", block_ids=np.zeros((2, 2, 2, 2), np.int32))
    assert catalog.list_datasets() == []


def test_discovery_reads_n_from_the_npz_header(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(catalog, "OUTPUTS_ROOT", Path("outputs"))
    monkeypatch.setattr(catalog, "CORPUS_DIRS", ())
    d = tmp_path / "outputs" / "bench_arms"
    d.mkdir(parents=True)
    ids = np.empty(3, dtype=object)
    for i in range(3):
        ids[i] = np.ones((2, 2, 2), np.int32)
    np.savez(d / "toy_16.npz", block_ids=ids, block_data=ids,
             sources=np.array(["", "", ""]), corpus=np.array(["", "", ""]))
    (d / "toy_16_manifest.json").write_text(json.dumps(
        {"count": 3, "items": [{"title": f"t{i}"} for i in range(3)]}))

    found = catalog.list_datasets()
    assert [(x.kind, x.n) for x in found] == [("arm", 3)]
    assert found[0].id == "arm:bench_arms__toy_16"
    assert catalog.get_dataset(found[0].id) == found[0]


def test_dataset_ids_are_url_and_path_safe():
    for d in catalog.list_datasets():
        assert "/" not in d.id and " " not in d.id
        assert catalog.parse_build_id(catalog.make_build_id(d.id, 0))[0] == d.id


# --- real artifacts on this machine ---------------------------------------
@pytest.mark.slow
def test_finds_the_bench_arms_and_the_corpus():
    ids = {d.id for d in catalog.list_datasets()}
    assert "arm:bench_arms__native_oriented_32" in ids
    assert "corpus:houses_32" in ids


@pytest.mark.slow
def test_build_row_matches_the_loaded_structure():
    ds = "arm:bench_arms__native_oriented_32"
    row = catalog.build_row(ds, 0)
    s = catalog.load_builds(ds)[0]
    assert row["build_id"] == f"{ds}:0"
    assert row["dims"] == [int(v) for v in s.shape]
    assert row["n_blocks"] == int(s.occupied_mask.sum()) > 0
    # Cropped, like `ArmSpec.load` and `split_structures` -- so the dims shown in
    # the lab are the dims the benchmark scored.
    assert s.crop_to_non_air().shape == s.shape


@pytest.mark.slow
def test_build_row_out_of_range_raises_index_error():
    ds = "arm:bench_arms__native_oriented_32"
    n = catalog.get_dataset(ds).n
    with pytest.raises(IndexError):
        catalog.build_row(ds, n)


@pytest.mark.slow
def test_load_builds_is_memoized():
    ds = "arm:bench_arms__native_oriented_32"
    catalog.forget(ds)
    assert catalog.load_builds(ds) is catalog.load_builds(ds)


@pytest.mark.slow
def test_split_datasets_are_the_canonical_split():
    from blockgen.eval.bench import splits

    if not splits.split_path("houses_32", 0, (0.70, 0.15, 0.15)).is_file():
        pytest.skip("no split on this machine")
    by_id = {d.id: d for d in catalog.list_datasets()}
    test = by_id["split:houses_32:test"]
    assert test.kind == "split"
    assert test.n == len(catalog.load_builds("split:houses_32:test"))


@pytest.mark.slow
def test_scorecards_parse_and_expose_a_leaderboard():
    # Skips on any machine with no bench output, so `tests/lab/test_scorecard_read.py`
    # is the version of this contract that actually runs in CI (committed fixtures).
    cards = catalog.list_scorecards()
    if not cards:
        pytest.skip("no benchmark run on this machine")
    assert all(c["run"] and c["when"] for c in cards)
    card = catalog.load_scorecard(cards[0]["run"])
    # The raw `arms` block survives; the names are additive (see load_scorecard).
    assert isinstance(card["arms"], dict)
    assert card["arm_names"] == list(card["arms"])
    assert isinstance(card["leaderboard"], list)
    assert all("." in m for m in card["metrics"])


def test_load_scorecard_missing_run_raises():
    with pytest.raises(FileNotFoundError):
        catalog.load_scorecard("run_does_not_exist")


@pytest.mark.slow
def test_real_bench2_card_matches_run_keys():
    """What a real run writes, checked against the constant the runner builds from.

    The other half of D34. The synthesized fixture in `tests/fixtures/scorecards/`
    pins `RUN_KEYS` too, but it is not a run's output -- it is a card the fixture
    generator types -- so it can agree with `RUN_KEYS` while the runner has quietly
    stopped writing one of them. Only a card produced by an actual `python -m
    blockgen.eval.bench` closes that gap, which is why this one test is `slow` and
    skips rather than being folded into the fixture suite.

    Read from disk with `json.loads`, deliberately NOT through `catalog`:
    `cards.migrate` fills `name`, `note` and `started_at` defaults, so a migrated
    blob would satisfy `RUN_KEYS` no matter what the writer emitted.
    """
    from blockgen.eval.bench.scorecard import RUN_KEYS, RUN_KEYS_OPTIONAL

    rows = [r for r in catalog.list_scorecards() if Path(r["path"]).is_file()]
    cards = [b for b in (json.loads(Path(r["path"]).read_text()) for r in rows)
             if b.get("schema_version") == "bench/2"]
    if not cards:
        pytest.skip("no bench/2 run on this machine")
    for blob in cards:
        run = blob["run"]
        extra = set(run) - set(RUN_KEYS) - set(RUN_KEYS_OPTIONAL)
        # An unlisted key is a schema change that RUN_KEYS never heard about.
        assert not extra, extra
        assert set(run) == set(RUN_KEYS) | (set(run) & set(RUN_KEYS_OPTIONAL))


# --- renders ---------------------------------------------------------------
def _tiny():
    """A 4x4x4 hut: a floor, a wall, and one contrasting block, so the frame has
    real variance to test against."""
    from blockgen.utils.data import Structure
    ids = np.zeros((4, 4, 4), np.int32)
    ids[:, 0, :] = 4
    ids[0, :, :] = 5
    ids[2, 2, 2] = 98
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


def test_cache_key_includes_px_and_view():
    keys = {renders.cache_key("corpus:nope", 0, px, v)
            for px in (128, 256) for v in range(4)}
    assert len(keys) == 8


def test_cache_key_is_stable_for_a_missing_dataset():
    a = renders.cache_key("corpus:nope", 3, 256, 0)
    assert a == renders.cache_key("corpus:nope", 3, 256, 0)
    assert a != renders.cache_key("corpus:nope", 4, 256, 0)


def test_cache_path_is_sharded():
    key = renders.cache_key("corpus:nope", 0, 256, 0)
    p = renders.cache_path(key)
    assert p.name == f"{key}.jpg" and p.parent.name == key[:2]


def test_placeholder_is_a_jpeg_and_is_not_uniform():
    """A flat tile is indistinguishable from a broken renderer's output, which is
    exactly the confusion T25c cost a day to untangle."""
    from PIL import Image
    import io
    data = renders.placeholder(64)
    img = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    assert data[:2] == b"\xff\xd8" and img.shape == (64, 64, 3)
    assert not renders.looks_blank(img)


def test_looks_blank_detects_a_uniform_frame():
    assert renders.looks_blank(np.full((8, 8, 3), 255, np.uint8))
    assert not renders.looks_blank(np.arange(192, dtype=np.uint8).reshape(8, 8, 3))


@pytest.mark.parametrize("build", [
    pytest.param(lambda a: a, id="all_white"),
    pytest.param(lambda a: (a.__setitem__((0, 0), 0), a)[1], id="one_stray_pixel"),
    pytest.param(lambda a: (a.__setitem__((slice(90, 96), slice(90, 96)), 40), a)[1],
                 id="tiny_sliver"),
])
def test_looks_blank_catches_every_near_blank_frame(build):
    """An exact `var == 0` test -- the first version -- passes all but the
    first of these. This is the canary for T25c, the most expensive silent
    failure in the project, so it is calibrated with margin on both sides."""
    assert renders.looks_blank(build(np.full((192, 192, 3), 255, np.uint8)))


def test_looks_blank_refuses_encoded_bytes():
    """JPEG bytes have high variance whatever they depict, so accepting them
    would make the check silently always-pass -- which is how it was first
    called by mistake."""
    with pytest.raises(TypeError, match="image array"):
        renders.looks_blank(b"\xff\xd8\xff\xe0not pixels")


def test_failed_render_returns_a_placeholder_and_does_not_cache_it(tmp_path, monkeypatch):
    monkeypatch.setattr(renders, "THUMB_CACHE", tmp_path / "thumbs")
    monkeypatch.setattr(renders, "_render_one",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no EGL")))
    data = renders.thumb("arm:does_not_exist", 0, px=64)
    assert data[:2] == b"\xff\xd8"
    assert not list((tmp_path / "thumbs").rglob("*.jpg"))


@pytest.mark.slow
def test_render_is_not_blank():
    """The T25c canary: bytes are not enough, the pixels have to vary.

    A PyOpenGL/Python mismatch once made every render raise, every frame come
    back uniform white, and every arm score a near-perfect KID. Uniform output is
    the signature of a dead renderer.
    """
    from PIL import Image
    import io

    img = renders.render_array(_tiny(), px=96, view=0)
    assert img.shape == (96, 96, 3)
    assert not renders.looks_blank(img), "renderer produced a uniform frame"

    data = renders.encode_jpeg(img)
    assert data[:2] == b"\xff\xd8"
    decoded = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    assert float(decoded.var()) > 1.0


@pytest.mark.slow
def test_thumb_caches_to_disk_and_reuses(tmp_path, monkeypatch):
    monkeypatch.setattr(renders, "THUMB_CACHE", tmp_path / "thumbs")
    ds = "arm:bench_arms__native_oriented_32"
    if catalog.get_dataset(ds) is None:
        pytest.skip("no bench arms on this machine")

    first = renders.thumb(ds, 0, px=96)
    files = list((tmp_path / "thumbs").rglob("*.jpg"))
    assert len(files) == 1 and files[0].read_bytes() == first

    # Second call must come off disk, not off the GPU.
    monkeypatch.setattr(renders, "_render_one",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("cache miss on a warm thumbnail")))
    assert renders.thumb(ds, 0, px=96) == first


@pytest.mark.slow
def test_views_returns_four_distinct_orbit_angles(tmp_path, monkeypatch):
    monkeypatch.setattr(renders, "THUMB_CACHE", tmp_path / "thumbs")
    ds = "arm:bench_arms__native_oriented_32"
    if catalog.get_dataset(ds) is None:
        pytest.skip("no bench arms on this machine")
    out = renders.views(ds, 0, px=96)
    assert len(out) == 4 == len(set(out)), "four views, four different images"


# --- store -----------------------------------------------------------------
@pytest.fixture()
def db(tmp_path):
    s = store.Store(tmp_path / "lab.db")
    yield s
    s.close()


def test_label_set_get_clear(db):
    bid = "arm:a:0"
    assert db.get_label(bid) is None
    db.set_label(bid, "good")
    assert db.get_label(bid) == "good"
    db.set_label(bid, "bad")
    assert db.get_label(bid) == "bad"
    db.set_label(bid, None)
    assert db.get_label(bid) is None
    assert db.labels() == {}


def test_relabelling_keeps_created_at(db):
    """A re-labelled build still records when it was first judged."""
    db.set_label("arm:a:0", "good")
    first = db.export("labels")[0]
    db.set_label("arm:a:0", "bad")
    again = db.export("labels")[0]
    assert again["created_at"] == first["created_at"]
    assert again["label"] == "bad"


def test_label_rejects_unknown_values(db):
    with pytest.raises(ValueError):
        db.set_label("arm:a:0", "meh")


def test_label_rejects_malformed_build_id(db):
    with pytest.raises(ValueError):
        db.set_label("nocolon", "good")


def test_counts_are_per_dataset_and_zero_filled(db):
    db.set_label("arm:a:0", "good")
    db.set_label("arm:a:1", "good")
    db.set_label("arm:a:2", "bad")
    db.set_label("arm:b:0", "unsure")
    assert db.counts() == {"good": 2, "bad": 1, "unsure": 1}
    assert db.counts("arm:a") == {"good": 2, "bad": 1, "unsure": 0}
    assert db.counts("arm:nothing") == {"good": 0, "bad": 0, "unsure": 0}


def test_labels_filtered_by_dataset(db):
    db.set_label("split:houses_32:test:4", "good")
    db.set_label("arm:a:0", "bad")
    assert db.labels("split:houses_32:test") == {"split:houses_32:test:4": "good"}


def test_notes_upsert_and_blank_deletes(db):
    db.set_note("arm:a:0", "  roof floats  ")
    assert db.get_note("arm:a:0") == "roof floats"
    db.set_note("arm:a:0", "")
    assert db.get_note("arm:a:0") is None
    db.set_note("arm:a:0", "back")
    db.set_note("arm:a:0", "   ")
    assert db.notes() == {}


def test_compares_are_append_only_and_ordered(db):
    db.add_compare("arm:a:0", "arm:b:0", "arm:a:0", 900, tag="realism")
    db.add_compare("arm:a:1", "arm:b:1", None, 4200)
    rows = db.compares()
    assert [r["winner"] for r in rows] == ["arm:a:0", None]
    assert rows[0]["tag"] == "realism" and rows[1]["tag"] == ""
    assert rows[0]["ms"] == 900
    assert all(r["created_at"] for r in rows)


def test_compare_rejects_a_winner_that_is_neither_side(db):
    with pytest.raises(ValueError):
        db.add_compare("arm:a:0", "arm:b:0", "arm:c:0", 100)


def test_export_shapes(db):
    db.set_label("arm:a:0", "good")
    db.set_note("arm:a:0", "nice gable")
    db.add_compare("arm:a:0", "arm:b:0", None, 10)
    assert set(db.export("labels")[0]) == {"build_id", "dataset", "label",
                                           "created_at", "updated_at"}
    assert set(db.export("notes")[0]) == {"build_id", "dataset", "text",
                                          "created_at", "updated_at"}
    assert set(db.export("compares")[0]) == {"id", "a", "b", "winner", "ms",
                                             "tag", "created_at"}
    with pytest.raises(ValueError):
        db.export("everything")


def test_store_survives_reopen(tmp_path):
    path = tmp_path / "lab.db"
    with store.Store(path) as s:
        s.set_label("arm:a:0", "good")
        s.set_note("arm:a:0", "keep")
    with store.Store(path) as s:
        assert s.get_label("arm:a:0") == "good"
        assert s.get_note("arm:a:0") == "keep"


def test_store_is_usable_from_several_threads(db):
    """The server is threaded; a label lost to `database is locked` would look
    like a mis-click rather than an error."""
    import threading

    errors: list = []

    def work(k: int) -> None:
        try:
            for i in range(25):
                db.set_label(f"arm:t{k}:{i}", "good")
                db.set_note(f"arm:t{k}:{i}", f"n{i}")
                db.labels(f"arm:t{k}")
        except Exception as exc:                      # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert sum(db.counts().values()) == 100


# --- cache schemas and ordering --------------------------------------------
def test_both_cache_schemas_load():
    """`data/minecraft/cache` holds two incompatible layouts and neither
    existing reader covers both. Loading a build-cache npz with the house
    reader raises `KeyError: 'sources'`, which the render guard swallowed into
    a placeholder tile -- five of nine corpora looked like a broken renderer.
    """
    seen = {}
    for d in catalog.list_datasets():
        if d.kind != "corpus":
            continue
        builds = catalog.load_builds(d.id)
        assert len(builds) == d.n, d.id
        assert builds[0].occupied_mask.any(), d.id
        seen[d.id] = len(builds)
    assert len(seen) >= 2, "expected several corpora on this machine"


def test_build_cache_titles_come_from_the_meta_sidecar():
    """The build-cache schema keeps no titles in the npz; without the sidecar
    the older corpora are thousands of anonymous tiles."""
    ds = next((d for d in catalog.list_datasets()
               if d.id.startswith("corpus:gc_small")), None)
    if ds is None:
        pytest.skip("no gc_small_* cache on this machine")
    titles = [catalog.build_row(ds.id, i)["title"] for i in range(12)]
    assert any(t for t in titles), "every title blank -- sidecar not joined"


def test_corpora_sort_above_smoke_runs():
    """Discovery order put nine real corpora below seventeen 1-8 build smoke
    runs, which made the picker look like it had found the wrong thing."""
    ds = catalog.list_datasets()
    kinds = [d.kind for d in ds]
    if "corpus" in kinds and "arm" in kinds:
        assert kinds.index("corpus") < kinds.index("arm")
    for a, b in zip(ds, ds[1:]):
        if a.kind == b.kind:
            assert a.n >= b.n, f"{a.id} ({a.n}) before {b.id} ({b.n})"


def test_every_tiny_arm_is_flagged_even_when_its_name_says_smoke():
    for d in catalog.list_datasets():
        if d.kind == "arm" and d.n <= catalog.SMOKE_N:
            assert catalog.SMOKE_NOTE in d.note, d.id


def test_concurrent_renders_do_not_lose_the_egl_context():
    """The bug a mutex did not fix.

    An EGL context is thread-affine: `eglMakeCurrent` fails with
    `EGL_BAD_ACCESS` while the context is current on another thread. A lock
    serialises calls but leaves each on a different thread, which is exactly the
    failing condition, and `ThreadingHTTPServer` gives every request its own
    thread. Symptom was half a grid of placeholder tiles, intermittently.

    Hammering from many threads is the only way to catch it -- serial rendering
    passes happily.
    """
    import concurrent.futures as cf

    ds = next((d for d in catalog.list_datasets() if d.kind == "corpus"), None)
    if ds is None:
        pytest.skip("no corpus on this machine")
    n = min(24, ds.n)
    place = renders.placeholder(96)

    def one(i: int) -> bytes:
        return renders.thumb(ds.id, i, px=96)

    with cf.ThreadPoolExecutor(12) as ex:
        out = list(ex.map(one, range(n)))

    assert all(b[:2] == b"\xff\xd8" for b in out), "not every response was a JPEG"
    fell_back = [i for i, b in enumerate(out) if b == place]
    assert not fell_back, f"{len(fell_back)}/{n} renders fell back to the placeholder"
