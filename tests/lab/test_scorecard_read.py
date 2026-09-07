"""The lab's scorecard read path, over the five committed golden cards.

`tools/lab/test_lab_data.py:142` already asserts `load_scorecard`'s key set --
but it is `@pytest.mark.slow` and *skips* on a machine with no bench output, and
`outputs/` is gitignored, so in CI a key could be dropped from the reader or
added to the writer and nothing would go red. This file is the version that
actually runs: the fixtures are staged into properly named run directories under
a monkeypatched `OUTPUTS_ROOT`, so every shape on disk is exercised with no
corpus, no GPU and no renderer.

The load-bearing test is `test_read_time_enrichment_never_reaches_the_memo`.
`_read_scorecard` memoizes the parsed blob process-wide and `load_scorecard`
returns a *shallow* copy of it, so `blob["arms"]` is one object shared by every
reader for the life of the server. Enrichment written inside `arms` at read time
is permanent, cumulative and invisible -- the second request is served a card the
file does not contain, and the only symptom is a page that disagrees with the
JSON on disk. `cards.migrate` shares `arms` by identity precisely so this is
checkable, and this is where it is checked end to end, through the API payload
builder that does the enriching.

`test_ordering_is_by_dir_stamp_and_survives_utime` is the other one worth
naming: the picker used to sort by `st_mtime`, so a `git checkout`, an rsync or
a stray `touch` silently reordered it and "the newest run" was whichever card
the filesystem had been poked at last.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import time
from pathlib import Path

import pytest

from tools.lab import api, cards, catalog

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "scorecards"

#: `{key: (fixture, the run directory it is staged into)}`. The directory names
#: are the runs the fixtures were cut from (see the fixtures' README): the run id
#: is the DIRECTORY name, never `run.dir`, and `card_legacy_a`'s `run.dir` is an
#: absolute scratchpad path that proves the difference.
STAGE = {
    "legacy_a": ("card_legacy_a.json", "run_20260803_033522_bench_compare"),
    "legacy_allcontrol": ("card_legacy_allcontrol.json", "run_20260831_195140_bench"),
    "legacy_c": ("card_legacy_c.json", "run_20260831_225029_bench"),
    "legacy_d": ("card_legacy_d.json", "run_20260906_190029_bench"),
    "bench2": ("run_20260102_000000_bench_fixture", "run_20260102_000000_bench_fixture"),
}

#: Newest run first, by the stamp in the directory name.
EXPECTED_ORDER = ["run_20260906_190029_bench", "run_20260831_225029_bench",
                  "run_20260831_195140_bench", "run_20260803_033522_bench_compare",
                  "run_20260102_000000_bench_fixture"]

#: The keys `load_scorecard` adds on top of whatever the file itself carries.
#: Asserted as an exact difference so that dropping one fails here rather than in
#: the browser: `index.html` and `leaderboard.html` read every one of them.
ADDED_KEYS = {"run_id", "path", "when", "leaderboard", "head_to_head", "arm_names",
              "metrics", "label", "note", "started", "n_submissions", "n_controls",
              "n_baselines", "compat"}

#: The original `list_scorecards` contract, frozen. Everything after it is
#: additive; these five may not move.
LIST_KEYS_ORIGINAL = {"run", "path", "when", "n_arms", "corpus"}
LIST_KEYS_ADDED = {"label", "note", "started", "tier", "n_submissions",
                   "n_controls", "n_baselines", "git_dirty", "has_examples"}


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    """Every fixture staged under a private `outputs/`, memos cleared.

    `chdir` as well as repointing `OUTPUTS_ROOT`, because `_arm_datasets` writes
    a dataset `source` of `str(Path("outputs") / rel)` -- a literal prefix, not
    the configured root -- and `_load_uncached` opens that string relative to the
    working directory. Repointing alone would discover the caches and then fail
    to open them, which is not a shape this test wants to be pinning.

    Both memos are cleared and the dataset-source index is dropped: the scorecard
    memo is keyed on `(path, mtime)` and would serve one test's card to the next
    inside a single filesystem tick, and `_SOURCE_INDEX` is keyed on the *string*
    `"outputs"`, which every test here shares while pointing at different tmp
    directories.
    """
    root = tmp_path / "outputs"
    root.mkdir()
    for name, run_id in STAGE.values():
        src = FIXTURES / name
        if src.is_dir():
            shutil.copytree(src, root / run_id)
        else:
            (root / run_id).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, root / run_id / "scorecard.json")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(catalog, "OUTPUTS_ROOT", Path("outputs"))
    monkeypatch.setattr(catalog, "CORPUS_DIRS", ())
    monkeypatch.setattr(catalog, "_SOURCE_INDEX", None)
    catalog.forget_scorecards()
    catalog.forget()
    yield Path("outputs")
    catalog.forget_scorecards()
    catalog.forget()


def run_dir(key: str) -> Path:
    return Path("outputs") / STAGE[key][1]


def card_path(key: str) -> Path:
    return run_dir(key) / "scorecard.json"


def raw(key: str) -> dict:
    """The fixture exactly as committed -- the file the staged copy came from."""
    src = FIXTURES / STAGE[key][0]
    return json.loads((src if src.is_file() else src / "scorecard.json").read_text())


@pytest.fixture(params=sorted(STAGE))
def any_key(request):
    return request.param


# --- loading ---------------------------------------------------------------
def test_all_shapes_load(outputs, any_key):
    card = catalog.load_scorecard(STAGE[any_key][1])
    assert card["run_id"] == STAGE[any_key][1]
    assert card["arms"] and card["arm_names"] == list(card["arms"])
    assert all("." in m for m in card["metrics"])


def test_load_scorecard_key_set(outputs, any_key):
    """The assertion that today lives only behind `@pytest.mark.slow`.

    An exact difference, not a subset: a key added here without a page reading it
    is dead payload, and a key removed is a blank table nobody notices until a
    leaderboard is open in front of someone.
    """
    card = catalog.load_scorecard(STAGE[any_key][1])
    on_disk = set(raw(any_key))
    assert set(card) - on_disk == ADDED_KEYS
    assert on_disk <= set(card)                       # nothing from the file lost
    assert isinstance(card["compat"], dict)
    assert {"version", "read_as", "sections", "missing", "unsupported"} \
        <= set(card["compat"])


def test_all_three_resolution_forms(outputs):
    """A bare directory name, a directory path and a full file path all resolve
    to the same run -- `?run=` bookmarks, the picker's `<option value>` and a
    hand-typed path are the three callers."""
    run_id = STAGE["legacy_c"][1]
    cards = [catalog.load_scorecard(run_id),
             catalog.load_scorecard(str(run_dir("legacy_c"))),
             catalog.load_scorecard(str(card_path("legacy_c")))]
    assert {c["run_id"] for c in cards} == {run_id}
    assert cards[0] == cards[1] == cards[2]


def test_missing_run_still_raises(outputs):
    with pytest.raises(FileNotFoundError):
        catalog.load_scorecard("run_does_not_exist")


def test_a_directory_without_a_scorecard_raises(outputs):
    (Path("outputs") / "run_20260101_000000_empty").mkdir()
    with pytest.raises(FileNotFoundError):
        catalog.load_scorecard("run_20260101_000000_empty")


# --- ordering --------------------------------------------------------------
def test_ordering_is_by_dir_stamp_and_survives_utime(outputs):
    """Sorting by `st_mtime` meant a `git checkout` reordered the run picker.

    The oldest card is touched to *now* -- the exact thing a fresh checkout does
    to every file -- and the newest run must still be the newest run.
    """
    assert [r["run"] for r in catalog.list_scorecards()] == EXPECTED_ORDER
    oldest = card_path("bench2")
    now = time.time()
    os.utime(oldest, (now, now))
    catalog.forget_scorecards()
    assert [r["run"] for r in catalog.list_scorecards()] == EXPECTED_ORDER


def test_a_directory_with_no_stamp_falls_back_to_mtime(outputs):
    """`_scorecard_paths` still has to place a run directory that does not follow
    the naming convention -- `--out` may point anywhere -- and mtime is genuinely
    the best guess left for one."""
    odd = Path("outputs") / "bench_compare"
    odd.mkdir()
    shutil.copyfile(card_path("legacy_c"), odd / "scorecard.json")
    runs = [r["run"] for r in catalog.list_scorecards()]
    assert set(runs) == set(EXPECTED_ORDER) | {"bench_compare"}
    assert runs[0] == "bench_compare"                 # mtime = just now


# --- list_scorecards -------------------------------------------------------
def test_list_scorecards_keeps_the_existing_keys(outputs):
    rows = catalog.list_scorecards()
    assert len(rows) == len(STAGE)
    for row in rows:
        assert LIST_KEYS_ORIGINAL <= set(row)
        assert set(row) == LIST_KEYS_ORIGINAL | LIST_KEYS_ADDED
        assert row["path"].endswith("scorecard.json")
        assert row["n_arms"] == len(json.loads(Path(row["path"]).read_text())["arms"])


def test_list_scorecards_row_values(outputs):
    rows = {r["run"]: r for r in catalog.list_scorecards()}
    bench2 = rows["run_20260102_000000_bench_fixture"]
    assert bench2["label"] == "fixture run"
    assert bench2["started"] == "2026-01-02T00:00:00Z"
    assert bench2["tier"] == "both" and bench2["has_examples"] is True
    assert bench2["git_dirty"] is False
    assert bench2["corpus"] == "houses_32"

    legacy = rows["run_20260906_190029_bench"]
    assert legacy["corpus"] == "houses_48"
    assert legacy["started"] == "2026-09-06T19:00:29Z"     # from the directory
    assert legacy["has_examples"] is False
    # Not False: a card written before the flag existed does not know whether the
    # tree was clean, and "unknown" must not render as "clean" beside a sha.
    assert legacy["git_dirty"] is None


def test_the_all_control_card_labels_itself_with_its_run_id(outputs):
    rows = {r["run"]: r for r in catalog.list_scorecards()}
    row = rows["run_20260831_195140_bench"]
    assert row["label"] == "run_20260831_195140_bench"
    assert row["n_submissions"] == 0


def test_submission_counts_exclude_controls_and_baselines(outputs):
    """`n_arms` is a number no human recognises: the card everyone points at has
    sixteen arms and three results. (These are the post-trim counts; the
    untrimmed ones are asserted in the `slow` test on real output.)"""
    card = catalog.load_scorecard(STAGE["legacy_c"][1])
    assert (card["n_submissions"], card["n_controls"], card["n_baselines"]) == (1, 1, 1)
    assert len(card["arm_names"]) == 3
    row = {r["run"]: r for r in catalog.list_scorecards()}[STAGE["legacy_c"][1]]
    assert (row["n_submissions"], row["n_controls"], row["n_baselines"]) == (1, 1, 1)
    assert row["n_arms"] == 3


# --- the memo --------------------------------------------------------------
@pytest.mark.parametrize("key", sorted(STAGE))
def test_read_time_enrichment_never_reaches_the_memo(outputs, key):
    """The regression that matters most (D27/D29).

    A full read cycle -- `load_scorecard`, then the API payload builder that adds
    `arms_index` -- must leave the memoized blob's `arms` byte-identical to the
    file, and two consecutive reads must be equal, because enrichment that leaked
    into the shared object would be cumulative.

    Parametrized over EVERY fixture, and the legacy ones are the point. On the
    `bench/2` card the derived values (`kind`, `origin`, `provenance`) already
    equal what is stored, so a write-back into the shared `arms` would be a no-op
    and would escape. The eleven real cards on disk are all `bench/1` and carry
    none of those keys -- they are the shape where a leak actually changes the
    memo, so they are the shape this test has to run against.
    """
    path = card_path(key)
    on_disk = json.loads(path.read_text())

    catalog.load_scorecard(str(path))
    payload = api._scorecard_payload(catalog.load_scorecard(str(path)))
    assert payload["arms_index"]                      # enrichment really happened

    memo = catalog._read_scorecard(path)
    assert memo["arms"] == on_disk["arms"]
    assert set(memo) - set(on_disk) == {"compat"}
    assert payload["arms"] is memo["arms"]            # shared, and left alone

    a = catalog.load_scorecard(str(path))
    b = catalog.load_scorecard(str(path))
    assert a == b
    assert set(a) - set(on_disk) == ADDED_KEYS        # not cumulative


def test_the_memo_is_shared_and_not_deep_copied(outputs):
    """A `deepcopy` would satisfy every behavioural test above and cost a 390 KB
    copy on every request; identity is the property actually wanted."""
    path = card_path("legacy_c")
    assert catalog.load_scorecard(str(path))["arms"] \
        is catalog._read_scorecard(path)["arms"]


def test_forget_scorecards_rereads_the_file(outputs):
    """Two different cards written to one path inside a single mtime tick is the
    case this exists for."""
    path = card_path("legacy_c")
    assert "cost" not in catalog.load_scorecard(str(path))["compat"]["sections"]
    shutil.copyfile(FIXTURES / "card_legacy_d.json", path)
    os.utime(path, (0, 0))
    catalog.forget_scorecards()
    assert catalog.load_scorecard(str(path))["compat"]["sections"].count("cost") == 1


# --- example builds --------------------------------------------------------
def test_example_build_ids_are_well_formed(outputs):
    """Ids are minted by `_arm_id` from the path the bench recorded, so the lab
    has exactly one slugging rule and the two sides cannot drift."""
    blob = catalog.load_scorecard(STAGE["bench2"][1])
    ids = catalog.example_build_ids(run_dir("bench2"), blob)
    dataset = catalog._arm_id(run_dir("bench2") / "examples_4.npz")
    assert dataset == "arm:run_20260102_000000_bench_fixture__examples_4"
    assert set(ids) == set(blob["arms"])
    for arm, rows in ids.items():
        assert rows == [f"{dataset}:{r}"
                        for r in blob["arms"][arm]["meta"]["examples"]["rows"]]
        for bid in rows:
            assert catalog.parse_build_id(bid)[0] == dataset
    flat = [b for rows in ids.values() for b in rows]
    assert len(flat) == len(set(flat)) == blob["run"]["examples"]["count"]


def test_example_build_ids_when_the_npz_is_missing(outputs):
    """The card still claims examples; the file is gone. `{}` beats ids that 404
    on every thumbnail in the strip."""
    (run_dir("bench2") / "examples_4.npz").unlink()
    blob = catalog.load_scorecard(STAGE["bench2"][1])
    assert catalog.example_build_ids(run_dir("bench2"), blob) == {}
    payload = api._scorecard_payload(blob)
    assert all(row["examples"] == [] for row in payload["arms_index"])


def test_example_build_ids_without_the_manifest_sibling(outputs):
    """`_openable` is discovery's own admission test, and this is the shape it
    exists for: an npz with no manifest is not a dataset and never will be."""
    (run_dir("bench2") / "examples_4_manifest.json").unlink()
    blob = catalog.load_scorecard(STAGE["bench2"][1])
    assert catalog.example_build_ids(run_dir("bench2"), blob) == {}


def test_example_build_ids_refuses_a_run_dir_outside_outputs(tmp_path, outputs):
    """`--out` may point anywhere and two real cards record an absolute
    scratchpad path. Discovery walks `outputs/{bench_arms,run_*}` only, so ids
    minted elsewhere would resolve to nothing (D25)."""
    outside = tmp_path / "scratch" / "run_20260102_000000_bench_fixture"
    shutil.copytree(FIXTURES / "run_20260102_000000_bench_fixture", outside)
    blob = json.loads((outside / "scorecard.json").read_text())
    assert catalog.example_build_ids(outside, blob) == {}


def test_example_build_ids_refuses_a_dir_that_discovery_does_not_glob(outputs):
    """Inside `outputs/`, but not under `bench_arms/` or a `run_*` directory."""
    odd = Path("outputs") / "bench_compare"
    shutil.copytree(FIXTURES / "run_20260102_000000_bench_fixture", odd)
    blob = json.loads((odd / "scorecard.json").read_text())
    assert catalog.example_build_ids(odd, blob) == {}


def test_a_legacy_card_has_no_examples(outputs):
    blob = catalog.load_scorecard(STAGE["legacy_c"][1])
    assert catalog.example_build_ids(run_dir("legacy_c"), blob) == {}
    assert "run.examples" in blob["compat"]["missing"]


def test_examples_dataset_is_not_flagged_as_a_smoke_run(outputs):
    """`SMOKE_N` is 8 and the bench keeps `k` builds per arm, so by size alone a
    single-arm run's examples cache is indistinguishable from a scratch run --
    and labelling a benchmark artifact "smoke run (8 builds)" is a lie (D23)."""
    by_id = {d.id: d for d in catalog.list_datasets()}
    examples = by_id["arm:run_20260102_000000_bench_fixture__examples_4"]
    assert examples.n == catalog.SMOKE_N
    assert examples.note.startswith(catalog.EXAMPLES_NOTE)
    assert "run_20260102_000000_bench_fixture" in examples.note
    assert catalog.SMOKE_NOTE not in examples.note

    # The arm cache beside it is two builds and IS flagged -- the exemption is
    # about what the file is, not about how small it is.
    arm = by_id["arm:run_20260102_000000_bench_fixture__fixture_ar_4"]
    assert catalog.SMOKE_NOTE in arm.note


def test_examples_dataset_is_openable(outputs):
    """The producer's file passes the consumer's own admission test and opens
    with the captions the writer set on copies."""
    dataset = "arm:run_20260102_000000_bench_fixture__examples_4"
    builds = catalog.load_builds(dataset)
    assert len(builds) == 8
    manifest = json.loads((run_dir("bench2") / "examples_4_manifest.json").read_text())
    assert [s.metadata["category"] for s in builds] == \
        [item["category"] for item in manifest["items"]]
    assert all(s.metadata["title"] for s in builds)
    row = catalog.build_row(dataset, 0)
    assert row["build_id"] == f"{dataset}:0" and row["n_blocks"] > 0


# --- the API payload -------------------------------------------------------
def test_arms_index_joins_every_arm(outputs, any_key):
    payload = api._scorecard_payload(catalog.load_scorecard(STAGE[any_key][1]))
    rows = payload["arms_index"]
    assert [r["name"] for r in rows] == sorted(payload["arms"])
    assert "metric_index" not in payload
    for row in rows:
        assert row["kind"] in ("submission", "control", "baseline")
        assert row["origin"] in ("npz", "in_memory")
        assert isinstance(row["provenance"], dict)
        assert isinstance(row["examples"], list)


def test_arms_index_recipe_falls_back_for_a_legacy_card(outputs):
    """13 of 16 rows on the newest real card are generated in-process; without
    the static fallback every one of them says only "nothing on disk to show"."""
    payload = api._scorecard_payload(catalog.load_scorecard(STAGE["legacy_c"][1]))
    rows = {r["name"]: r for r in payload["arms_index"]}
    assert rows["real_test"]["recipe"].startswith("genuinely held-out real builds")
    assert rows["uniform_random"]["recipe"]
    assert rows["real_test"]["provenance"] == {}      # the card predates it


def test_arms_index_examples_are_populated_for_a_bench2_card(outputs):
    payload = api._scorecard_payload(catalog.load_scorecard(STAGE["bench2"][1]))
    rows = {r["name"]: r for r in payload["arms_index"]}
    assert all(len(r["examples"]) == 2 for r in rows.values())
    assert rows["fixture_ar"]["source_run_id"] == "run_20260101_000000_fixture_train"
    assert rows["fixture_ar"]["structures_sha"]


def test_a_cards_own_recipe_beats_the_static_table(outputs):
    """`arm_recipe` prefers what the run recorded over what this repo believes now.

    Asserted with a sentinel, because the fixture's recorded recipe was generated
    FROM `cards.RECIPES()` -- comparing the two would compare a string to itself
    and would still pass with the preference deleted. The preference is the whole
    point of the field: a control's prose can be reworded in a later release, and
    a card must keep describing the rung that actually ran in it.
    """
    card = catalog.load_scorecard(STAGE["bench2"][1])
    sentinel = "SENTINEL: the rung as it was worded when this run scored it"
    assert sentinel not in cards.RECIPES().values()
    meta = card["arms"]["real_test"]["meta"]
    assert meta["provenance"]["recipe"] != sentinel

    patched = copy.deepcopy(card)
    patched["arms"]["real_test"]["meta"]["provenance"]["recipe"] = sentinel
    rows = {r["name"]: r for r in api._scorecard_payload(patched)["arms_index"]}
    assert rows["real_test"]["recipe"] == sentinel

    # ...and the static table is still the fallback when the card recorded nothing.
    patched["arms"]["real_test"]["meta"]["provenance"].pop("recipe")
    rows = {r["name"]: r for r in api._scorecard_payload(patched)["arms_index"]}
    assert rows["real_test"]["recipe"] == cards.RECIPES()["real_test"]
