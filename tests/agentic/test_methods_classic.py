"""``classic`` on a non-DSL action space: the generic ``turns`` path."""

from __future__ import annotations

import pytest

from blockgen.agentic.episode import run_episode
from blockgen.agentic.recipes.classic import classic

from .fakes import (HUT_PROGRAM, FakeActions, RecordingTrace, ScriptedLLM, fake_context,
                    make_fake_kit)

BROKEN = "fill stone 0 0 0 9 0 9\nfill notablock 0 1 0 1 1 1\n"
FIXED = "fill stone 0 0 0 9 0 9\nwalls oak_planks 0 1 0 9 3 9\n"


def run(params, replies, *, keep_history=True, trace=None, **kit_kw):
    llm = ScriptedLLM(replies)
    kit = make_fake_kit(llm=llm, method=classic,
                        spec_kw={"params": params, "keep_history": keep_history}, **kit_kw)
    res = run_episode(kit, "a small hut", trace=trace)
    return res, llm


def test_generic_build_uses_action_grammar_contexts_and_output_rule():
    res, llm = run({"repair_rounds": 0}, [FIXED])
    system, turn = llm.calls[0][0], llm.calls[0][-1]
    assert "FAKE GRAMMAR" in system.text and "FAKE CONTEXT" in system.text
    assert turn.text.startswith("Build: a small hut")
    assert turn.text.rstrip().endswith(FakeActions.OUTPUT_RULE)
    assert [a.stage for a in res.attempts] == ["generate"] and res.blocks > 0


def test_examples_come_from_the_action_space_unless_an_examples_context_exists():
    res, llm = run({"n_examples": 1, "repair_rounds": 0}, [FIXED])
    msgs = llm.calls[0]
    assert [m.role for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[1].text == "Build: a small hut" and msgs[2].text == HUT_PROGRAM.strip()
    res, llm = run({"n_examples": 1, "repair_rounds": 0}, [FIXED],
                   contexts=[("examples", fake_context)])
    assert [m.role for m in llm.calls[0]] == ["system", "user"]


def test_plan_repair_and_history():
    res, llm = run({"plan": True, "repair_rounds": 2}, ["PLAN TEXT", BROKEN, FIXED])
    assert res.plan_text == "PLAN TEXT"
    assert [m.role for m in llm.calls[0]] == ["system", "user"]          # plan: no history
    assert "Your plan:\nPLAN TEXT" in llm.calls[1][-1].text
    repair_turn = llm.calls[2][-1].text
    assert "notablock" in repair_turn and repair_turn.rstrip().endswith(FakeActions.OUTPUT_RULE)
    assert [m.role for m in llm.calls[2]] == ["system", "user", "assistant", "user"]
    assert [a.stage for a in res.attempts] == ["generate", "repair1"]   # clean -> stop


def test_repair_without_history_uses_the_legacy_shape():
    res, llm = run({"repair_rounds": 1}, [BROKEN, FIXED], keep_history=False)
    msgs = llm.calls[1]
    assert [m.role for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[1].text == "Build: a small hut" and msgs[2].text == BROKEN


def test_critique_rewrite_and_patch():
    res, llm = run({"repair_rounds": 0, "critique_rounds": 1}, [FIXED, HUT_PROGRAM])
    turn = llm.calls[1][-1]
    assert sum(p.kind == "image" for p in turn.parts) == 2
    assert "45° azimuth, 225° azimuth" in turn.text and "COMPLETE program" in turn.text
    assert res.program_text == HUT_PROGRAM
    res, llm = run({"repair_rounds": 0, "critique_rounds": 1, "critique_mode": "patch"},
                   [FIXED, "set glowstone 1 4 1"])
    assert res.attempts[-1].patch and res.program_text.endswith("# --- patch ---\nset glowstone 1 4 1")


def test_critique_without_renderer_is_skipped():
    trace = RecordingTrace()
    res, llm = run({"repair_rounds": 0, "critique_rounds": 2}, [FIXED], renderer=None,
                   trace=trace)
    assert llm.n_calls == 1 and "critique_skipped" in trace.kinds()


def test_empty_critique_is_rolled_back_and_stops():
    res, llm = run({"repair_rounds": 0, "critique_rounds": 3}, [FIXED, "clear 0 0 0 31 31 31"])
    assert [a.stage for a in res.attempts] == ["generate", "critique1"]
    assert not res.attempts[1].accepted and res.best_idx == 0 and res.blocks > 0


def test_bad_critique_mode_is_an_error():
    res, _ = run({"critique_mode": "nope"}, [FIXED])
    assert res.truncated == "error" and "critique_mode" in res.error


def test_reference_images_reach_the_build_turn():
    llm = ScriptedLLM([FIXED])
    kit = make_fake_kit(llm=llm, method=classic, spec_kw={"params": {"repair_rounds": 0}})
    run_episode(kit, "a hut", images=[b"\x89PNGref"])
    turn = llm.calls[0][-1]
    assert turn.parts[-1].image_png == b"\x89PNGref" and "Reference image" in turn.text


@pytest.mark.parametrize("seed,rider", [(None, False), (4, True)])
def test_variation_rider_only_for_an_explicit_seed(seed, rider):
    llm = ScriptedLLM([FIXED])
    kit = make_fake_kit(llm=llm, method=classic, spec_kw={"params": {"repair_rounds": 0}})
    run_episode(kit, "a hut", seed=seed)
    assert ("Design variation #4" in llm.calls[0][-1].text) is rider
