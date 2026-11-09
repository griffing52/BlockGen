"""The phone queue: captions in the store, and what the planner hands out.

The properties pinned here are the ones a phone session silently depends on:
progress through a dataset is the same walk on every visit, nothing already
labelled or still in the page's undo window comes back, an identical pair is
never asked twice under one tag, and a dry pool in Mix mode does not stop the
page -- the other kinds keep coming.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools.lab import phone
from tools.lab.store import Store


class FakePools:
    """Stands in for `phone._Pools`: a fixed catalog, same seeded order."""

    def __init__(self, sizes):
        self._ds = [SimpleNamespace(id=k, n=v, kind=k.split(":")[0], name=k)
                    for k, v in sizes.items()]
        self._real = phone._Pools()

    def datasets(self):
        return list(self._ds)

    def resolve(self, patterns):
        import fnmatch
        seen, out = set(), []
        for pat in patterns:
            for d in self._ds:
                if d.id not in seen and fnmatch.fnmatchcase(d.id, pat):
                    seen.add(d.id)
                    out.append(d)
        return out

    def order(self, dataset_id, n):
        return self._real.order(dataset_id, n)


@pytest.fixture()
def store(tmp_path):
    with Store(tmp_path / "lab.db") as s:
        yield s


@pytest.fixture(autouse=True)
def _no_catalog_rows(monkeypatch):
    """`_row` would open a real corpus; the planner's choices do not depend on it."""
    monkeypatch.setattr(phone, "_row", lambda ds, i: {
        "build_id": f"{ds}:{i}", "dataset": ds, "index": i, "dims": [8, 8, 8],
        "n_blocks": 100, "title": "", "category": ""})


CONFIG = {
    "mix": {"compare": 0.5, "curate": 0.5},
    "compare": [{"tag": "t", "question": "which?", "a": ["real:*"], "b": ["arm:*"],
                 "pairing": "across"}],
    "curate": [{"question": "keep?", "datasets": ["corpus:one", "corpus:two"]}],
    "caption": [{"prompt": "say", "datasets": ["corpus:one"],
                 "prefer_label": "good", "skip_label": "bad"}],
}
SIZES = {"real:test": 5, "arm:x": 4, "arm:y": 4, "corpus:one": 6, "corpus:two": 3}


def planner(store, seed=0, config=CONFIG, sizes=SIZES):
    return phone.Planner(store, config=json.loads(json.dumps(config)),
                         pools=FakePools(sizes), seed=seed)


# --- store -------------------------------------------------------------------
def test_caption_upsert_blank_deletes_and_exports(store):
    store.set_caption("corpus:one:1", "  a small   stone cottage ")
    assert store.get_caption("corpus:one:1") == "a small stone cottage"
    store.set_caption("corpus:one:1", "a tower")
    assert store.captions("corpus:one") == {"corpus:one:1": "a tower"}
    rows = store.export("captions")
    assert rows[0]["text"] == "a tower" and rows[0]["dataset"] == "corpus:one"
    store.set_caption("corpus:one:1", "   ")
    assert store.get_caption("corpus:one:1") is None
    assert store.export("captions") == []


def test_activity_counts_since(store):
    store.set_label("corpus:one:0", "good")
    store.add_compare("arm:x:0", "arm:y:1", "arm:x:0", 500, "t")
    store.set_caption("corpus:one:0", "hut")
    total = store.activity()
    assert (total["labels"], total["compares"], total["captions"]) == (1, 1, 1)
    assert store.activity("2999-01-01T00:00:00Z") == {
        "labels": 0, "notes": 0, "captions": 0, "compares": 0}


def test_compared_pairs_and_keys(store):
    store.add_compare("arm:x:0", "arm:y:1", None, 10, "t")
    store.add_compare("arm:y:1", "arm:x:2", "arm:x:2", 10, "t")
    store.add_compare("arm:x:0", "arm:y:1", "arm:x:0", 10, "other")
    assert store.compared_pairs("t") == {"arm:x:0": 1, "arm:y:1": 2, "arm:x:2": 1}
    assert store.compared_pair_keys("t") == {("arm:x:0", "arm:y:1"), ("arm:x:2", "arm:y:1")}
    assert len(store.compared_pair_keys()) == 2      # same pair under two tags


# --- curate ------------------------------------------------------------------
def test_curate_walk_is_stable_and_skips_labelled(store):
    first = [t["build_id"] for t in planner(store).next_tasks("curate", 3)]
    again = [t["build_id"] for t in planner(store, seed=99).next_tasks("curate", 3)]
    assert first == again, "the walk must not depend on the session"
    assert all(b.startswith("corpus:one:") for b in first)
    store.set_label(first[0], "bad")
    after = [t["build_id"] for t in planner(store).next_tasks("curate", 3)]
    assert after == first[1:] + [b for b in after if b not in first]
    assert first[0] not in after


def test_curate_finishes_one_dataset_before_the_next(store):
    for i in range(6):
        store.set_label(f"corpus:one:{i}", "good")
    got = [t["build_id"] for t in planner(store).next_tasks("curate", 10)]
    assert len(got) == 3 and all(b.startswith("corpus:two:") for b in got)
    assert got[0] == got[0] and len(set(got)) == 3


def test_curate_respects_exclude_and_returns_remaining(store):
    p = planner(store)
    head = p.next_tasks("curate", 2)
    ids = [t["build_id"] for t in head]
    nxt = planner(store).next_tasks("curate", 2, exclude=ids)
    assert not {t["build_id"] for t in nxt} & set(ids)
    assert head[0]["remaining"] == 6


# --- compare -----------------------------------------------------------------
def test_compare_pairs_cross_pools_and_randomize_sides(store):
    tasks = []
    for seed in range(4):
        tasks += planner(store, seed=seed).next_tasks("compare", 8)
    assert len(tasks) == 32, "a batch is not capped by the smaller pool"
    for t in tasks:
        assert t["a"].startswith("real:test:") and t["b"].startswith("arm:")
        assert {t["left"], t["right"]} == {t["a"], t["b"]}
        assert t["tag"] == "t" and t["kind"] == "compare"
    flips = sum(t["left"] == t["b"] for t in tasks)
    assert 4 <= flips <= 28, f"sides should be randomized, got {flips}/32 flipped"


def test_compare_batch_spreads_over_builds(store):
    tasks = planner(store).next_tasks("compare", 8)
    reals = [t["a"] for t in tasks]
    assert len(set(reals)) >= 4, "the balance draw should spread trials across the pool"


def test_compare_never_repeats_a_judged_pair(store):
    # 5 real x 8 arm builds = 40 possible pairs; judge most of them.
    judged = set()
    for i in range(5):
        for arm in ("x", "y"):
            for j in range(4):
                if len(judged) >= 36:
                    break
                a, b = f"real:test:{i}", f"arm:{arm}:{j}"
                store.add_compare(a, b, a, 100, "t")
                judged.add((a, b))
    tasks = planner(store).next_tasks("compare", 24)
    for t in tasks:
        assert (t["a"], t["b"]) not in judged
    assert tasks, "four unjudged pairs remain and should be found"


# --- caption -----------------------------------------------------------------
def test_caption_prefers_good_then_skips_bad_and_done(store):
    store.set_label("corpus:one:3", "good")
    store.set_label("corpus:one:0", "bad")
    store.set_caption("corpus:one:3", "already")
    got = [t["build_id"] for t in planner(store).next_tasks("caption", 10)]
    assert "corpus:one:3" not in got and "corpus:one:0" not in got
    assert len(got) == 4
    store.set_caption("corpus:one:3", "")
    first = planner(store).next_tasks("caption", 1)[0]
    assert first["build_id"] == "corpus:one:3" and first["label"] == "good"


# --- mix ---------------------------------------------------------------------
def test_mix_keeps_serving_when_one_kind_runs_dry(store):
    for ds, n in (("corpus:one", 6), ("corpus:two", 3)):
        for i in range(n):
            store.set_label(f"{ds}:{i}", "good")
    tasks = planner(store).next_tasks("mix", 6)
    assert len(tasks) == 6 and all(t["kind"] == "compare" for t in tasks)


def test_single_mode_stops_when_dry(store):
    for i in range(6):
        store.set_caption(f"corpus:one:{i}", "x")
    assert planner(store).next_tasks("caption", 4) == []


def test_bad_mode_rejected(store):
    with pytest.raises(ValueError):
        planner(store).next_tasks("nope", 1)


def test_summary_reports_progress(store):
    store.set_label("corpus:one:1", "good")
    store.set_caption("corpus:one:1", "x")
    s = planner(store).summary()
    assert s["jobs"]["curate"][0] == {"question": "keep?", "datasets": ["corpus:one", "corpus:two"],
                                      "total": 9, "done": 1}
    assert s["jobs"]["caption"][0]["done"] == 1
    assert s["jobs"]["compare"][0]["b"] == ["arm:x", "arm:y"]


# --- config ------------------------------------------------------------------
def test_load_config_falls_back_on_broken_json(tmp_path):
    p = tmp_path / "phone.json"
    p.write_text("{not json")
    assert phone.load_config(p)["mix"] == phone.DEFAULT_CONFIG["mix"]
    p.write_text(json.dumps({"_comment": "x", "mix": {"compare": 1.0}}))
    cfg = phone.load_config(p)
    assert cfg["mix"] == {"compare": 1.0} and "_comment" not in cfg
    assert cfg["curate"] == phone.DEFAULT_CONFIG["curate"]   # untouched keys keep defaults
    assert phone.load_config(tmp_path / "missing.json")["compare"]


def test_committed_config_resolves_to_real_pools():
    """The config in the repo must name datasets the catalog actually has, or
    the phone opens on an empty page. Skipped when the data is not on disk."""
    from tools.lab import catalog
    if not catalog.list_datasets():
        pytest.skip("no datasets on this machine")
    cfg = phone.load_config()
    pools = phone._Pools()
    for job in cfg["compare"]:
        assert pools.resolve(job["a"]) and pools.resolve(job["b"]), job["tag"]
    for job in cfg["curate"] + cfg["caption"]:
        assert pools.resolve(job["datasets"]), job["datasets"]
