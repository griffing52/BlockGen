"""Inspection checks, the ``inspect`` turn / step / method, and the ``lessons`` context."""

from __future__ import annotations

import pytest

from blockgen.agentic import inspection, lessons as lessons_mod
from blockgen.agentic.episode import run_episode
from blockgen.agentic.recipes.inspect import inspect as inspect_method

from .fakes import HUT_PROGRAM, FakeActions, RecordingTrace, ScriptedLLM, make_attempt, make_fake_kit

FLOOR = "fill stone 0 0 0 9 0 9\n"
SHELL = FLOOR + "walls oak_planks 0 1 0 9 4 9 thickness=1\n"
BOX = SHELL + "fill oak_planks 0 5 0 9 5 9\n"                       # roofed, sealed
HOUSE = BOX + "fill air 4 1 0 4 2 0\nfill glass 2 2 0 2 3 0\nfill glass 7 2 0 7 3 0\n"
SOLID = "fill stone 0 0 0 9 6 9\n"
NO_WINDOWS = BOX + "fill air 4 1 0 4 2 0\n"                       # door, no windows
FLOATING = HOUSE + "set glowstone 20 10 20\nset glowstone 20 11 20\n"
LIFTED = "fill stone 0 3 0 9 3 9\nwalls oak_planks 0 4 0 9 7 9\nfill oak_planks 0 8 0 9 8 9\n"


def checks(program):
    return {f.check: f for f in inspection.inspect_attempt(make_attempt(program)).findings}


def test_house_passes_everything_but_notes():
    insp = inspection.inspect_attempt(make_attempt(HOUSE), canvas=(32, 32, 32))
    assert [f.check for f in insp.failed] == []
    assert insp.score == 1.0 and insp.clean("info") and insp.worst() is None
    assert insp.stats["rooms"] == 1 and insp.stats["doorways"] == 1
    assert "[ok]" in insp.text() and "FAIL" not in insp.text()


def test_sealed_box_is_not_enterable():
    f = checks(BOX)
    assert f["interior"].passed and f["roofed"].passed and f["headroom"].passed
    assert not f["enterable"].passed and f["enterable"].severity == "error"
    assert "door" in f["enterable"].hint
    assert not f["openings"].passed and f["openings"].severity == "info"


def test_roofless_shell_fails_interior_and_roof_not_solid():
    f = checks(SHELL)
    assert not f["interior"].passed and "solid mass" not in f["interior"].text
    assert not f["roofed"].passed
    assert not f["enterable"].applicable               # nothing to stand in under cover


def test_solid_mass_is_named():
    f = checks(SOLID)
    assert not f["interior"].passed and "solid mass" in f["interior"].text
    assert f["roofed"].passed


def test_floating_piece_and_lifted_build():
    f = checks(FLOATING)
    assert not f["connected"].passed and "2 blocks" in f["connected"].text
    assert "(20, 10, 20)" in f["connected"].text
    f = checks(LIFTED)
    assert not f["grounded"].passed and "y=3" in f["grounded"].text
    assert not f["supported"].passed


def test_empty_build():
    insp = inspection.inspect_attempt(make_attempt(""))
    assert not insp.findings[0].passed and insp.findings[0].check == "built"
    assert insp.score == 0.0 and len(insp.applicable) == 1
    assert "nothing was built" in insp.text()


def test_text_orders_failures_worst_first_and_to_dict():
    insp = inspection.inspect_attempt(make_attempt(LIFTED))
    lines = [l for l in insp.text().splitlines() if l.startswith("[")]
    tags = [l[1:5] for l in lines]
    assert tags.index("FAIL") < tags.index("WARN") < tags.index("ok] ")
    d = insp.to_dict()
    assert d["n_applicable"] == len(insp.applicable) and "grounded" in d["failed"]


def test_inspection_scorer_registered():
    from blockgen.agentic import core
    core.load_builtins()
    assert "inspection" in core.names("scorer")
    assert core.meta("scorer", "inspection")["needs_renders"] is False
    att = make_attempt(BOX)
    term = inspection.inspection(None, att)
    assert 0 < term.value < 1 and "enterable" in term.note
    assert att.extra["inspection"]["failed"] == ["enterable", "openings"]


# --- the method -------------------------------------------------------------
def run(params, replies, *, trace=None, **kit_kw):
    llm = ScriptedLLM(replies)
    kit = make_fake_kit(llm=llm, method=inspect_method,
                        spec_kw={"params": params, "keep_history": True}, **kit_kw)
    res = run_episode(kit, "a small hut", trace=trace)
    return res, llm


def test_inspect_rounds_feed_findings_and_stop_when_clean():
    trace = RecordingTrace()
    res, llm = run({"repair_rounds": 0, "inspect_rounds": 3}, [BOX, HOUSE, HOUSE], trace=trace)
    turn = llm.calls[1][-1].text
    assert "INSPECTION of the build" in turn and "[FAIL] enterable" in turn
    assert "Rewrite the COMPLETE program" in turn
    assert turn.rstrip().endswith(FakeActions.OUTPUT_RULE)
    assert [a.stage for a in res.attempts] == ["generate", "inspect1"]   # HOUSE is clean: stop
    assert llm.n_calls == 2
    events = [e for e in trace.events if e[0] == "inspection"]
    assert [e[1]["failed"] for e in events] == [["enterable", "openings"], []]
    assert res.attempts[-1].extra["inspection"]["failed"] == []
    m = res.metrics()
    assert m["inspect_pass"] == 1.0 and m["inspect_enterable"] == 1.0


def test_clean_at_info_keeps_going_and_patch_mode():
    res, llm = run({"repair_rounds": 0, "inspect_rounds": 2}, [BOX, NO_WINDOWS, HOUSE])
    assert llm.n_calls == 2                       # 'openings' is info: clean at warning
    res, llm = run({"repair_rounds": 0, "inspect_rounds": 2, "clean_at": "info"},
                   [BOX, NO_WINDOWS, HOUSE])
    assert llm.n_calls == 3 and "[note] openings" in llm.calls[2][-1].text
    res, llm = run({"repair_rounds": 0, "inspect_rounds": 1, "inspect_mode": "patch"},
                   [BOX, "fill air 4 1 0 4 2 0"])
    assert res.attempts[-1].patch and "additional commands" in llm.calls[1][-1].text
    assert res.attempts[-1].extra["inspection"]["failed"] == ["openings"]


def test_then_critique_and_bad_params():
    res, llm = run({"repair_rounds": 0, "inspect_rounds": 1, "critique_rounds": 1},
                   [BOX, HOUSE, HUT_PROGRAM])
    assert [a.stage for a in res.attempts] == ["generate", "inspect1", "critique1"]
    assert sum(p.kind == "image" for p in llm.calls[2][-1].parts) == 2
    # run_episode turns a method error into a truncated result, never a raise
    res, _ = run({"inspect_mode": "nope"}, [BOX])
    assert res.truncated == "error" and "inspect_mode" in res.error
    res, _ = run({"clean_at": "never"}, [BOX])
    assert res.truncated == "error" and "clean_at" in res.error


def test_empty_rejected_reply_stops_the_stage():
    res, llm = run({"repair_rounds": 0, "inspect_rounds": 3}, [BOX, "", ""])
    assert llm.n_calls == 2 and res.blocks > 0


# --- lessons ----------------------------------------------------------------
def test_lessons_file_loads_and_retrieves_by_tags():
    pool = lessons_mod.load_lessons()
    assert len(pool) >= 10 and all(l.text for l in pool)
    chosen = lessons_mod.select_lessons("a two-storey cottage with a chimney", pool, k=3)
    ids = [l.id for l in chosen]
    assert all(l.always for l in chosen[:sum(l.always for l in pool)])
    assert "chimney" in ids and "storeys" in ids
    assert len(ids) == sum(l.always for l in pool) + 3
    none = lessons_mod.select_lessons("qwertyuiop", pool, k=3, always=False)
    assert none == []


def test_lessons_context_and_shuffled_control():
    from blockgen.agentic import core
    core.load_builtins()
    assert "lessons" in core.names("context")

    class Ep:
        prompt = "a stone tower with a balcony"
        seed = 0
        notes = {}
    ep = Ep()
    parts = lessons_mod.lessons(ep, k=4)
    assert parts.system_text.startswith(lessons_mod.HEADER)
    assert "tower" in ep.notes["lessons"]["ids"] and "supports" in ep.notes["lessons"]["ids"]
    n = parts.system_text.count("\n- ")
    ep2 = Ep()
    ctrl = lessons_mod.lessons(ep2, k=4, shuffled=True)
    assert ctrl.system_text.count("\n- ") == n
    assert ctrl.system_text != parts.system_text and ep2.notes["lessons"]["shuffled"]
    pool = lessons_mod.load_lessons()
    retrieved = [l for l in lessons_mod.select_lessons(ep.prompt, pool, k=4) if not l.always]
    assert retrieved and not any(l.text in ctrl.system_text for l in retrieved)
    assert all(l.text in ctrl.system_text for l in pool if l.always)   # always lessons stay
    with pytest.raises(FileNotFoundError):
        lessons_mod.load_lessons("/nonexistent/lessons.yaml")


def test_recipes_load_and_hash():
    from blockgen.agentic.recipe import load_recipe
    specs = {n: load_recipe(n) for n in ("inspect", "inspect-critique", "inspect-lessons",
                                         "inspect-lessons-shuffled", "classic-lessons")}
    assert specs["inspect"].method == "inspect"
    assert specs["classic-lessons"].method == "classic"
    assert len({s.hash() for s in specs.values()}) == len(specs)
    assert specs["inspect-critique"].params["critique_rounds"] == 1
