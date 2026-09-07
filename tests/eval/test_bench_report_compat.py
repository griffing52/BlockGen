"""`scripts/bench_report.py` still reads every scorecard shape on disk.

This is the additive-only rule (D6) enforced by what would actually break rather
than by taste. Every other consumer of a scorecard degrades -- the leaderboard
page renders a banner, `catalog` fills defaults, `cards.migrate` is total -- but
`bench_report` reads `card["arms"]` bare (bench_report.py:87) and
`entry["gen"]["mean"]` / `entry["real"]["mean"]` bare (:69-70), so a removed,
renamed or re-typed key surfaces there as a `KeyError` on a run that has already
finished, with the numbers on disk and no way to print them.

The five committed fixtures in `tests/fixtures/scorecards/` are the three
`bench/1` shapes plus the all-control card plus a synthesized `bench/2` run, so
running the real script over all of them is the cheapest statement of "nothing
was removed" that exists. It is also the only test that reads the *script*: it
is not importable (`scripts/` is not a package and has no `__init__.py`), which
is why it is loaded by path through `importlib.util.spec_from_file_location`
instead of imported -- if it ever becomes importable, this indirection is the
only thing to delete.

Fast and unmarked: fixtures only, no corpus, no GPU, no renderer.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "bench_report.py"
FIXTURES = REPO / "tests" / "fixtures" / "scorecards"

#: Every committed card, legacy and `bench/2`. The synthesized one lives in a
#: run directory (it owns an examples cache); the four trimmed ones are flat.
CARD_PATHS = sorted(FIXTURES.glob("card_*.json")) + [
    FIXTURES / "run_20260102_000000_bench_fixture" / "scorecard.json"]

#: The five `run` keys a `bench/1` card carries -- everything else in
#: `scorecard.RUN_KEYS` is a `bench/2` addition. Written out rather than derived
#: from that constant on purpose: this test asserts what the OLD shape was, and
#: deriving it would make the assertion move whenever the new one does.
BENCH1_RUN_KEYS = ("dir", "git_sha", "cmd", "tier", "elapsed_s")


def _report():
    """Load `scripts/bench_report.py` as a module without importing it."""
    spec = importlib.util.spec_from_file_location("bench_report_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None, SCRIPT
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report():
    return _report()


@pytest.fixture(scope="module")
def cards() -> Dict[str, dict]:
    """Every fixture card, keyed by something a failure message can be read
    from: the flat file's name, or the synthesized card's run directory."""
    return {(p.parent.name if p.name == "scorecard.json" else p.name):
            json.loads(p.read_text()) for p in CARD_PATHS}


def test_every_fixture_is_present():
    """A deleted fixture must fail loudly here, not silently shrink the sweep."""
    assert len(CARD_PATHS) == 5
    assert all(p.is_file() for p in CARD_PATHS), [str(p) for p in CARD_PATHS]


@pytest.mark.parametrize("path", CARD_PATHS, ids=lambda p: p.parent.name if
                         p.name == "scorecard.json" else p.stem)
def test_render_survives_every_shape(report, path: Path):
    """No exception, and a table with a row per arm. The whole point."""
    card = json.loads(path.read_text())
    text = report.render(card)
    assert isinstance(text, str) and text.strip()
    for name in card["arms"]:
        assert f"`{name}`" in text, name


@pytest.mark.parametrize("path", CARD_PATHS, ids=lambda p: p.parent.name if
                         p.name == "scorecard.json" else p.stem)
def test_real_test_sorts_first(report, path: Path):
    """The real-data floor is the top row, so every other number is read against
    it. `order_arms` derives that from `meta.track` + the literal name
    `real_test`, both of which `bench/2` leaves byte-identical."""
    arms = json.loads(path.read_text())["arms"]
    order = report.order_arms(arms)
    assert order[0] == "real_test"
    # Nothing dropped and nothing duplicated by the sort.
    assert sorted(order) == sorted(arms)
    # Below the pinned floor row, controls come last: a control never sorts
    # above a submission or a baseline. (`real_test` is itself a control, which
    # is exactly why the script special-cases its name instead of its track.)
    kinds = [(arms[n].get("meta") or {}).get("track") for n in order[1:]]
    assert kinds == sorted(kinds, key=lambda t: t == "control")


@pytest.mark.parametrize("path", CARD_PATHS, ids=lambda p: p.parent.name if
                         p.name == "scorecard.json" else p.stem)
def test_render_without_ci_also_survives(report, path: Path):
    """`--no-ci` takes a second path through `cell`, which reads `metric["value"]`
    on cards where some leaves are `gate_failed` nulls."""
    assert report.render(json.loads(path.read_text()), show_ci=False).strip()


def test_load_runs_accepts_a_run_directory(report):
    """`load_runs` appends `scorecard.json` to a directory. The synthesized
    fixture is a real run directory, so this covers the path users actually
    type (`--run outputs/run_<stamp>_bench`)."""
    got = report.load_runs([str(FIXTURES / "run_20260102_000000_bench_fixture")])
    assert len(got) == 1 and got[0]["schema_version"] == "bench/2"


def test_load_runs_of_nothing_exits(report, tmp_path: Path):
    with pytest.raises(SystemExit):
        report.load_runs([str(tmp_path / "no_such_run")])


def test_bench2_additions_are_invisible_to_this_consumer(report):
    """The strongest form of "additive": strip every `bench/2` addition out of
    the synthesized card and the rendered table is byte-identical.

    If a future key ever changes what `bench_report` prints, it is no longer an
    addition -- it is a re-typing of something this script already read, and the
    eleven `bench/1` cards would print differently through the same code.
    """
    path = FIXTURES / "run_20260102_000000_bench_fixture" / "scorecard.json"
    full = json.loads(path.read_text())
    stripped = copy.deepcopy(full)
    stripped["schema_version"] = "bench/1"
    run = stripped["run"]
    stripped["run"] = {k: run[k] for k in BENCH1_RUN_KEYS if k in run}
    for blocks in stripped["arms"].values():
        meta = blocks.get("meta") or {}
        for key in ("kind", "origin", "source_run_id", "structures_sha",
                    "provenance", "examples"):
            meta.pop(key, None)
    assert report.render(full) == report.render(stripped)


@pytest.mark.parametrize("path", CARD_PATHS, ids=lambda p: p.parent.name if
                         p.name == "scorecard.json" else p.stem)
def test_render_survives_the_lab_migration(report, path: Path):
    """`cards.migrate` is what every lab reader sees. It adds `compat` and fills
    `run` defaults, and it shares `arms` by identity (D27) -- so a card that came
    back out of the lab's memo must still render through the script that never
    heard of the lab."""
    cards_mod = pytest.importorskip("tools.lab.cards")
    blob = cards_mod.migrate(json.loads(path.read_text()), path.parent.name, 0.0)
    assert report.render(blob).strip()


def test_coherence_entries_have_the_shape_the_script_reads_bare(cards: Dict[str, Any]):
    """`coherence_cell` reads `entry["gen"]["mean"]` and `entry["real"]["mean"]`
    with no `.get`. Asserted directly as well as through `render`, because this
    is the exact pair of subscripts named in the module docstring."""
    seen = 0
    for label, card in cards.items():
        for name, blocks in card["arms"].items():
            for key, entry in (blocks.get("coherence") or {}).items():
                if not isinstance(entry, dict):
                    continue
                assert "mean" in entry["gen"], (label, name, key)
                assert "mean" in entry["real"], (label, name, key)
                seen += 1
    assert seen > 0


def test_columns_still_name_real_sections(report, cards: Dict[str, Any]):
    """`COLUMNS` is a hard-coded list of `(section, key)` pairs. A renamed section
    would not crash -- it would quietly print a column of em dashes -- so every
    configured pair must resolve to a real metric leaf on at least one fixture."""
    live = {f"{section}.{key}"
            for card in cards.values()
            for blocks in card["arms"].values()
            for section, key, _, _, _ in report.COLUMNS
            if isinstance((blocks.get(section) or {}).get(key), dict)}
    missing = {f"{s}.{k}" for s, k, _, _, _ in report.COLUMNS} - live
    assert not missing, missing
