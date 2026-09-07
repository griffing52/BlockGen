"""`tools.lab.cards`: normalising a scorecard whose shape predates its reader.

Read against the five committed fixtures in `tests/fixtures/scorecards/`, which
are the four incompatible `bench/1` shapes on disk plus one synthesized `bench/2`
run directory. `outputs/` is gitignored, so without them the only test of this
read path is `@pytest.mark.slow` and *skips* on a machine with no bench output.

Three assertions here are structural rather than behavioural, and they are the
reason this file exists:

* `test_migrate_shares_arms_by_identity` -- `migrate(b, ...)["arms"] is b["arms"]`.
  `catalog._read_scorecard` memoizes the blob process-wide and `load_scorecard`
  hands out a *shallow* copy, so the input to `migrate` is the one object every
  later reader in the process will be served. A single write inside `arms` is
  permanent, cumulative and invisible. `is` is the only check that survives a
  careless future edit; a behavioural one would pass for a `deepcopy` too, and a
  deepcopy of a 390 KB card on every read is the other failure mode.
* `test_sections_is_a_union_not_the_first_arm` -- an arm that produced nothing
  returns a truncated block set, so reading the first arm's keys deletes whole
  metric families from the page depending on which arm failed.
* `test_recipes_fallback_covers_every_control_and_baseline_name` -- a new
  calibration rung with no prose renders as "nothing on disk to show" and is
  indistinguishable from a broken join.

Nothing here branches on `schema_version`, for the same reason `cards` does not:
that string says `bench/1` on all four legacy shapes and would be wrong about
eight of the eleven real cards before it ran.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from tools.lab import cards

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "scorecards"

#: `{key: (card path, the run id the lab would pass in)}`. The run id is the
#: name of the DIRECTORY the card was read from and never `run.dir` (D7) -- the
#: four flat fixtures are excerpts of runs whose directories are not in the
#: repository, so their ids are the ones `README.md` names as their sources.
CARDS = {
    "legacy_a": (FIXTURES / "card_legacy_a.json",
                 "run_20260803_033522_bench_compare"),
    "legacy_allcontrol": (FIXTURES / "card_legacy_allcontrol.json",
                          "run_20260831_195140_bench"),
    "legacy_c": (FIXTURES / "card_legacy_c.json",
                 "run_20260831_225029_bench"),
    "legacy_d": (FIXTURES / "card_legacy_d.json",
                 "run_20260906_190029_bench"),
    "bench2": (FIXTURES / "run_20260102_000000_bench_fixture" / "scorecard.json",
               "run_20260102_000000_bench_fixture"),
}

#: A fixed stand-in for `st_mtime` -- 2026-01-01T00:00:00Z. Passed everywhere so
#: the `started_at` fallback chain is testable: a card that carries neither a
#: recorded instant nor a parseable directory name lands on exactly this.
MTIME = 1767225600.0
MTIME_ISO = "2026-01-01T00:00:00Z"


def read(key: str) -> tuple[dict, str]:
    path, run_id = CARDS[key]
    return json.loads(path.read_text()), run_id


@pytest.fixture(params=sorted(CARDS))
def any_card(request):
    """Every committed shape, one at a time. Parametrized rather than looped so
    a failure names the shape that broke."""
    return read(request.param)


# --- migrate: total, idempotent, and non-writing ---------------------------
def test_migrate_is_total_and_idempotent(any_card):
    blob, run_id = any_card
    out = cards.migrate(blob, run_id, MTIME)
    assert isinstance(out, dict) and out["arms"]
    assert cards.migrate(out, run_id, MTIME) == out


def test_migrate_shares_arms_by_identity(any_card):
    """The shallow-copy memo trap, closed structurally (D27).

    `migrate` may return a new top level and nothing else: `arms` -- and every
    metric block under it -- is the object in `catalog._SCORECARDS`, shared by
    every reader for the life of the process.
    """
    blob, run_id = any_card
    out = cards.migrate(blob, run_id, MTIME)
    assert out is not blob
    assert out["arms"] is blob["arms"]
    assert out["run"] is not blob.get("run")     # the one block migrate owns


def test_migrate_does_not_mutate_its_input(any_card):
    blob, run_id = any_card
    before = deepcopy(blob)
    cards.migrate(blob, run_id, MTIME)
    assert blob == before


def test_migrate_never_raises_on_junk():
    """The lab's read side degrades to an empty result, never to a stack trace."""
    assert cards.migrate(None, "run_x", MTIME) == {}          # type: ignore[arg-type]
    assert cards.migrate([], "run_x", MTIME) == {}            # type: ignore[arg-type]
    empty = cards.migrate({}, "run_x", MTIME)
    assert empty["compat"]["missing"] == list(cards.MISSING_CHECKS)
    assert empty["run"] == {"name": "", "note": "", "started_at": MTIME_ISO}
    weird = {"run": "not a dict", "arms": {}, "schema_version": 7}
    out = cards.migrate(weird, "run_x", MTIME)
    assert out["compat"]["version"] == "" and out["compat"]["sections"] == []
    assert out["run"]["name"] == ""          # a non-dict `run` is replaced, not merged


def test_migrate_of_a_non_dict_arms_block_still_returns_a_readable_dict():
    """The documented degradation path, pinned rather than papered over.

    `sections()` guards against a non-dict `arms`; `missing()` does not (it calls
    `arms.values()` bare), so this shape raises *inside* `migrate` and takes the
    except branch. The contract that survives is the one the module promises:
    never raises, returns a new top-level dict, and `arms` is still shared by
    identity. What is lost is the `compat` block -- which every consumer already
    treats as optional -- and the warning is printed. No card on disk has ever
    had a non-dict `arms`; this exists so the fallback stays a fallback.
    """
    blob = {"arms": "not a dict", "schema_version": "bench/1"}
    out = cards.migrate(blob, "run_x", MTIME)
    assert isinstance(out, dict) and out is not blob
    assert out["arms"] is blob["arms"]


# --- identity --------------------------------------------------------------
def test_run_id_comes_from_the_directory_not_run_dir():
    """`card_legacy_a`'s `run.dir` is an absolute scratchpad path (D7).

    Its basename does not parse as a run stamp, so if identity were taken from
    `run.dir` the started-at fallback would drop straight through to the file
    mtime. The caller passes the directory name in, and that is what is used.
    """
    blob, run_id = read("legacy_a")
    assert Path(blob["run"]["dir"]).is_absolute()
    assert cards.dir_stamp(Path(blob["run"]["dir"]).name) is None
    out = cards.migrate(blob, run_id, MTIME)
    assert out["run"]["started_at"] == "2026-08-03T03:35:22Z"
    assert out["run"]["dir"] == blob["run"]["dir"]      # kept verbatim, never fixed


def test_started_at_falls_through_to_mtime_for_an_unparseable_dir():
    blob, _ = read("legacy_a")
    out = cards.migrate(blob, "bench_compare", MTIME)
    assert out["run"]["started_at"] == MTIME_ISO


def test_recorded_started_at_wins_over_the_directory_stamp():
    blob, run_id = read("bench2")
    assert cards.migrate(blob, "run_19990101_000000_x", MTIME)["run"]["started_at"] \
        == blob["run"]["started_at"]


@pytest.mark.parametrize("key", sorted(CARDS))
def test_dir_stamp_parses_every_fixture_run_id(key):
    assert cards.dir_stamp(CARDS[key][1]) is not None


@pytest.mark.parametrize("bad", ["", "bench_compare", "run_2026083_225029_x",
                                 "run_99999999_999999_x", "run_20260231_000000_x",
                                 "outputs/run_20260803_033522_bench"])
def test_dir_stamp_rejects_what_it_cannot_date(bad):
    """`strptime` is the check, not the regex: the regex alone accepts
    `run_99999999_999999_x`, and a fabricated date would sort the picker."""
    assert cards.dir_stamp(bad) is None


def test_dir_stamp_is_sortable_iso():
    assert cards.dir_stamp("run_20260831_225029_bench") == "2026-08-31T22:50:29Z"
    assert (cards.dir_stamp("run_20260102_000000_bench_fixture")
            < cards.dir_stamp("run_20260906_190029_bench"))


# --- sections and missing --------------------------------------------------
def test_sections_is_a_union_not_the_first_arm():
    """An arm that produced no non-empty structures returns the truncated block
    set at `fast.py:243-253`; `real_test` is only first when `--no-controls` was
    not passed. Reading the first arm's keys alone would drop `cost` -- and on a
    full-tier card `fidelity` -- off the page because one arm failed (D28)."""
    card = {"arms": {
        "dead_arm": {"meta": {}, "dataset_stats": {}, "novelty": {},
                     "realism": {}, "coherence": {}, "geometry_scalars": {}},
        "live_arm": {"meta": {}, "realism": {}, "fidelity": {}, "cost": {}},
    }}
    got = cards.sections(card["arms"])
    assert got == ["meta", "dataset_stats", "novelty", "realism", "coherence",
                   "geometry_scalars", "fidelity", "cost"]


def test_sections_union_on_a_real_card():
    """`card_legacy_d` carries `cost` on its submission arm only -- the shape
    that made the union non-negotiable."""
    blob, run_id = read("legacy_d")
    first = next(iter(blob["arms"].values()))
    assert "cost" not in first
    assert "cost" in cards.migrate(blob, run_id, MTIME)["compat"]["sections"]


def test_sections_of_junk_is_empty():
    assert cards.sections({}) == []
    assert cards.sections("nope") == []          # type: ignore[arg-type]
    assert cards.sections({"a": None}) == []


def test_missing_names_what_each_shape_predates():
    """Shape A predates BlockScore; every legacy card predates examples and
    provenance; the `bench/2` fixture is missing nothing."""
    def missing(key):
        blob, run_id = read(key)
        return cards.migrate(blob, run_id, MTIME)["compat"]["missing"]

    assert "run.blockscore" in missing("legacy_a")
    assert "geometry_scalars" in missing("legacy_a")
    assert "ladder" in missing("legacy_allcontrol")
    assert "fidelity" in missing("legacy_d")
    for key in ("legacy_a", "legacy_allcontrol", "legacy_c", "legacy_d"):
        assert {"run.examples", "meta.provenance"} <= set(missing(key)), key
    assert missing("bench2") == []


def test_missing_is_ordered_by_the_declared_check_list():
    blob, run_id = read("legacy_a")
    got = cards.migrate(blob, run_id, MTIME)["compat"]["missing"]
    assert got == [n for n in cards.MISSING_CHECKS if n in got]


# --- version ---------------------------------------------------------------
def test_compat_records_the_version_without_branching_on_it():
    blob, run_id = read("legacy_c")
    compat = cards.migrate(blob, run_id, MTIME)["compat"]
    assert compat["version"] == "bench/1"
    assert compat["read_as"] == f"bench/{cards.SCHEMA_MAJOR_SUPPORTED}"
    assert compat["unsupported"] is None


def test_unsupported_major_version():
    """A card from a newer BlockGen renders behind an amber banner; it does not
    fail to open, and its arms still come back."""
    blob, run_id = read("bench2")
    future = dict(blob, schema_version="bench/9")
    out = cards.migrate(future, run_id, MTIME)
    assert isinstance(out["compat"]["unsupported"], str)
    assert "bench/9" in out["compat"]["unsupported"]
    assert out["arms"] is future["arms"] and out["arms"]


def test_an_unparseable_version_is_not_unsupported():
    out = cards.migrate({"schema_version": "wat"}, "run_x", MTIME)
    assert out["compat"]["unsupported"] is None


# --- labels ----------------------------------------------------------------
def test_run_label_three_tiers():
    """`run.name` first, then the submission arms, then the run id.

    `card_legacy_c` is trimmed to one submission, so it exercises the second
    tier in its single-name form; the `+N` join is asserted below against a
    hand-built card, per the fixtures' README.
    """
    blob, run_id = read("bench2")
    assert cards.run_label(blob, run_id) == "fixture run"

    blob, run_id = read("legacy_c")
    assert cards.run_label(blob, run_id) == "native_oriented"

    blob, run_id = read("legacy_allcontrol")
    assert cards.run_label(blob, run_id) == run_id


def test_run_label_plus_n_form():
    card = {"arms": {name: {"meta": {"track": "ar"}}
                     for name in ("a", "b", "c", "d")}}
    assert cards.run_label(card, "run_x") == "a vs b vs c +1"


def test_run_label_ignores_controls_and_baselines():
    """A run with 3 submissions and 13 calibration rungs is not '16 arms'."""
    card = {"arms": {"real_test": {"meta": {"track": "control"}},
                     "uniform_random": {"meta": {"track": "baseline"}},
                     "native_oriented": {"meta": {"track": "ar"}}}}
    assert cards.run_label(card, "run_x") == "native_oriented"


def test_run_label_is_free_text_and_may_collide():
    """Two different runs may derive the identical label. That is fine and must
    stay fine: nothing may key on this string (D8)."""
    a = {"arms": {"native_oriented": {"meta": {"track": "ar"}}}}
    b = {"arms": {"native_oriented": {"meta": {"track": "ar"}}}}
    assert cards.run_label(a, "run_1") == cards.run_label(b, "run_2")


def test_run_label_of_a_whitespace_name_falls_through():
    card = {"run": {"name": "   "}, "arms": {"x": {"meta": {"track": "ar"}}}}
    assert cards.run_label(card, "run_x") == "x"


# --- kind and origin -------------------------------------------------------
def test_arm_kind_prefers_the_written_field():
    """`bench/2` records `kind` at scoring time, where the arm's construction is
    actually known; the derivation from `track` is the legacy fallback."""
    assert cards.arm_kind({"kind": "baseline", "track": "ar"}) == "baseline"


@pytest.mark.parametrize("track,expected", [
    ("control", "control"), ("baseline", "baseline"),
    ("ar", "submission"), ("agentic", "submission"), (None, "submission")])
def test_arm_kind_derives_from_track_when_unwritten(track, expected):
    assert cards.arm_kind({"track": track}) == expected


def test_arm_kind_of_an_unrecognised_value_falls_back():
    """A mystery pill on the page is worse than the derivation."""
    assert cards.arm_kind({"kind": "winner", "track": "control"}) == "control"
    assert cards.arm_kind({}) == "submission"
    assert cards.arm_kind(None) == "submission"          # type: ignore[arg-type]


def test_arm_origin_falls_back_to_the_source_string():
    """`meta.source` is the literal `"in-memory"` (hyphen) on a legacy card --
    a display string five consumers key off, so `bench/2` added a field beside
    it rather than changing it."""
    assert cards.arm_origin({"source": "in-memory"}) == "in_memory"
    assert cards.arm_origin({"source": "outputs/bench_arms/x.npz"}) == "npz"
    assert cards.arm_origin({"origin": "in_memory", "source": "x.npz"}) == "in_memory"
    assert cards.arm_origin({"origin": "guessed", "source": "in-memory"}) == "in_memory"


def test_kind_and_origin_agree_with_the_bench2_card():
    """The written fields on the synthesized card are what the derivation would
    have produced anyway -- so a legacy card and a current one group alike."""
    blob, _ = read("bench2")
    for name, arm in blob["arms"].items():
        meta = arm["meta"]
        assert cards.arm_kind(meta) == meta["kind"], name
        assert cards.arm_origin(meta) == meta["origin"], name


# --- recipes ---------------------------------------------------------------
def test_recipes_fallback_covers_every_control_and_baseline_name():
    """Every in-memory arm the bench can build has a sentence (D14/D15).

    The names come from `control_arms` actually running -- over three 4x4x4 huts,
    no corpus and no GPU -- rather than from a list, so a rung added to the
    builder and not to `CONTROL_RECIPES` fails here instead of shipping a row
    that says only "generated in-process, nothing on disk to show".
    """
    import numpy as np

    from blockgen.eval.bench import baselines, fast
    from blockgen.utils.data import Structure

    def hut(seed: int) -> Structure:
        ids = np.zeros((4, 4, 4), np.int32)
        ids[:, 0, :] = 4
        ids[0, :, :] = 5
        ids[2, 2, 2] = 90 + seed
        return Structure(block_ids=ids, block_data=np.zeros_like(ids))

    pool = [hut(i) for i in range(3)]
    built = [arm.name for arm in fast.control_arms(pool, train=[hut(9)], n=3)]
    assert "train_verbatim" in built          # the train-split rung really ran

    recipes = cards.RECIPES()
    for name in list(built) + list(baselines.ORDER):
        assert recipes.get(name, "").strip(), name


def test_recipes_is_memoized():
    assert cards.RECIPES() is cards.RECIPES()


def test_recipes_matches_the_eval_packages_own_tables():
    """Imported, never copied: the prose lives in one place per arm family."""
    from blockgen.eval.bench import baselines, fast

    recipes = cards.RECIPES()
    for source in (fast.CONTROL_RECIPES, baselines.RECIPES):
        for name, sentence in source.items():
            assert recipes[name] == sentence
