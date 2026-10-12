"""The ``dream`` method end to end, offline: fakes + scripted replies.

Covers the WP2 acceptance list: dream before build and shown to the builder,
refine (K regenerations, early stop), REDREAM (reply and stall) with re-anchored
scores, stop rules, accept/rollback, select=best from a rejected attempt, budget
truncation, max_images, patch replay, feedback modes and multi-dream specs.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from blockgen.agentic import turns
from blockgen.agentic.core import ScoreTerm
from blockgen.agentic.dsl import run_program
from blockgen.agentic.episode import run_episode
from blockgen.agentic.kit import resolve_dreams
from blockgen.agentic.recipes.dream import dream
from blockgen.agentic.recipes.steps import ANCHOR_NOTE

from .fakes import FakeImageGen, RecordingTrace, ScriptedLLM, make_fake_kit


def prog(n: int) -> str:
    """A floor of n x 10 blocks."""
    return f"fill stone 0 0 0 {n - 1} 0 9\n"


def seq(*items):
    """Reply callable: items in order, the last one repeating."""
    left = list(items)

    def reply(_msgs):
        return left.pop(0) if len(left) > 1 else left[0]
    return reply


def verdict(score=8, **kw):
    return json.dumps({"score": score, "buildable": True, "on_prompt": True, **kw})


def match_scorer(ep, att):
    return ScoreTerm(min(1.0, att.blocks / 500.0))


match_scorer.needs_dream = True


def improve_accept(prev, cand):
    if cand.blocks == 0:
        return False, "empty"
    if prev is None or prev.score is None:
        return True, "first"
    ok = cand.score.value > prev.score.value
    return ok, "improved" if ok else "not better"


REVIEW, REACH, JUDGE, COMPARE = ("IMAGE TO REVIEW", "BEST BUILD SO FAR", '"differences"',
                                 "Compare your build with the target")


def run(params, *, rules=(), build=None, dreams=("iso",), select="best", budget=None,
        accept=improve_accept, scorers=None, image=None, **kit_kw):
    llm = ScriptedLLM([build or prog(10)], rules=list(rules))
    kit = make_fake_kit(llm=llm, method=dream, dreams=list(dreams), accept=accept,
                        scorers=scorers or {"match": (match_scorer, 1.0)},
                        image=image or FakeImageGen(),
                        spec_kw={"params": params, "select": select, "budget": budget or {}},
                        **kit_kw)
    trace = RecordingTrace()
    res = run_episode(kit, "a small hut", trace=trace)
    return res, llm, kit, trace


def stages(res):
    return [a.stage for a in res.attempts]


def roles(trace):
    return [r["role"] for r in trace.requests]


# ------------------------------------------------------------------ dream first
def test_dream_is_generated_before_the_first_build_and_shown_to_the_builder():
    gen = FakeImageGen()
    seen = {}

    def build(msgs):
        seen["images_at_first_build"] = gen.calls
        seen["msg"] = msgs[-1]
        return prog(10)

    res, llm, kit, trace = run({"max_iters": 0}, build=build, image=gen)
    assert seen["images_at_first_build"] == 1
    parts = seen["msg"].parts
    assert parts[0].text.startswith("TARGET iso") and parts[1].image_png == res.dreams["iso"][0].png
    assert "TARGET image(s) above" in seen["msg"].text
    assert stages(res) == ["generate"] and res.blocks == 100
    assert res.truncated == "" and res.metrics()["n_dream_versions"] == 1


# ----------------------------------------------------------------------- refine
def test_refine_regenerates_up_to_k():
    bad = verdict(3, buildable=False, revised_prompt="a plain box hut", keep_composition=True)
    gen = FakeImageGen()
    res, llm, kit, trace = run({"refine": 2, "max_iters": 0}, rules=[(REVIEW, bad)], image=gen)
    vs = res.dreams["iso"]
    assert len(vs) == 3 and roles(trace).count("dream_critic") == 2
    assert vs[0].critique["score"] == 3 and vs[1].critique["buildable"] is False
    assert vs[2].critique == {}                                  # the K-th is not re-reviewed
    assert "a plain box hut" in vs[1].prompt and "a small hut" not in vs[1].prompt
    assert gen.requests[1].refs == (vs[0].png,)                  # keep_composition
    assert len({v.seed for v in vs}) == 3 and len({v.digest for v in vs}) == 3
    assert trace.kinds().count("dream_regenerate") == 2


def test_refine_stops_early_on_a_good_verdict():
    res, llm, kit, trace = run({"refine": 3, "max_iters": 0},
                               rules=[(REVIEW, seq(verdict(2), verdict(9)))])
    assert len(res.dreams["iso"]) == 2 and roles(trace).count("dream_critic") == 2
    assert res.dreams["iso"][1].critique["score"] == 9


def test_unparseable_review_keeps_the_dream():
    res, llm, kit, trace = run({"refine": 2, "max_iters": 0}, rules=[(REVIEW, "looks fine!")])
    assert len(res.dreams["iso"]) == 1 and "dream_review_unparsed" in trace.kinds()


def test_max_images_caps_regenerations_but_still_builds():
    res, llm, kit, trace = run({"refine": 5, "max_iters": 0}, rules=[(REVIEW, verdict(1))],
                               budget={"max_images": 3})
    assert len(res.dreams["iso"]) == 3 and "refine_budget_stop" in trace.kinds()
    assert res.blocks == 100 and res.truncated == ""


def test_multiview_dependants_use_the_parents_final_version():
    specs = resolve_dreams(["iso", "back", "interior", "plan", "mood"])
    gen = FakeImageGen()
    res, llm, kit, trace = run(
        {"refine": 1, "max_iters": 0}, image=gen, dreams=specs,
        rules=[("IMAGE TO REVIEW (iso)", seq(verdict(1), verdict(9))), (REVIEW, verdict(9))])
    order = [r.view for r in gen.requests]
    assert order == ["iso", "iso", "iso_back", "cutaway", "plan", ""]
    final_iso = res.dreams["iso"][-1].png
    for r in gen.requests[2:5]:
        assert r.refs == (final_iso,)
    first = llm.calls[[i for i, r in enumerate(trace.requests) if r["role"] == "builder"][0]][-1]
    labels = [p.text for p in first.parts if p.kind == "text" and p.text.startswith("TARGET")]
    assert [lab.split()[1] for lab in labels] == ["iso", "back", "interior", "plan", "mood"]


# ---------------------------------------------------------------------- redream
def test_redream_reply_regenerates_dependants_and_reanchors_scores():
    gen = FakeImageGen()
    res, llm, kit, trace = run(
        {"max_iters": 3, "redream": 1, "stop_at": 2.0, "patience": 5},
        dreams=("iso", "back"), image=gen,
        rules=[(COMPARE, seq("REDREAM: a tiny shed", prog(20), prog(30)))])
    assert len(res.dreams["iso"]) == 2 and len(res.dreams["back"]) == 2
    assert "a tiny shed" in res.dreams["iso"][1].prompt
    assert gen.requests[-1].refs == (res.dreams["iso"][1].png,)   # back v1 on iso v1
    assert stages(res) == ["generate", "match1", "match2"]
    final = res.attempts[-1].score.dream_id
    assert final and all(a.score.dream_id == final for a in res.attempts)
    assert "redream" in trace.kinds() and "rescore_all" in trace.kinds()
    compare_turns = [r["messages"][-1].text for r in trace.requests if r["stage"].startswith("match")]
    assert turns.REDREAM_TOKEN in compare_turns[0]               # escape hatch offered...
    assert turns.REDREAM_TOKEN not in compare_turns[1]           # ...not once it is used up
    assert res.blocks == 300


def test_redream_reply_without_budget_is_not_executed():
    res, llm, kit, trace = run({"max_iters": 2, "redream": 0, "patience": 5},
                               rules=[(COMPARE, "REDREAM: nope")])
    assert stages(res) == ["generate"] and len(res.dreams["iso"]) == 1
    assert trace.kinds().count("redream_denied") == 2


def test_stall_triggers_reachability_and_anchored_redream():
    gen = FakeImageGen()
    reach = json.dumps({"unreachable": True, "reason": "too ornate",
                        "revised_prompt": "a cube hut"})
    res, llm, kit, trace = run(
        {"max_iters": 6, "redream": 1, "redream_patience": 2, "patience": 3},
        image=gen, rules=[(REACH, reach), (COMPARE, prog(5))])
    assert roles(trace).count("dream_critic") == 1
    assert len(res.dreams["iso"]) == 2
    v1 = res.dreams["iso"][1]
    assert "a cube hut" in v1.prompt and ANCHOR_NOTE in v1.prompt
    anchor_render = res.attempts[0].renders["iso"]
    assert gen.requests[1].refs[0] == anchor_render
    # 2 stalls -> redream; 3 more stalls -> patience stop
    assert stages(res) == ["generate"] + [f"match{i}" for i in range(5)]
    final = hashlib.sha256(f"iso:{v1.digest}".encode()).hexdigest()[:12]
    assert all(a.score.dream_id == final for a in res.attempts)
    assert not any(a.accepted for a in res.attempts[1:])


def test_reachable_stall_does_not_redream():
    res, llm, kit, trace = run(
        {"max_iters": 6, "redream": 1, "redream_patience": 2, "patience": 2},
        rules=[(REACH, '{"unreachable": false}'), (COMPARE, prog(5))])
    assert len(res.dreams["iso"]) == 1 and stages(res) == ["generate", "match0", "match1"]


# ------------------------------------------------------------------- stop rules
def test_stop_at_threshold_skips_the_match_loop():
    res, llm, kit, trace = run({"max_iters": 3, "stop_at": 0.15})
    assert stages(res) == ["generate"] and "critic" not in roles(trace)
    assert ("stop", {"rule": "stop_at", "iter": 0, "value": 0.2}) in trace.events


def test_patience_stops_after_non_improving_iterations():
    res, llm, kit, trace = run({"max_iters": 5, "patience": 2}, rules=[(COMPARE, prog(5))],
                               select="last")
    assert stages(res) == ["generate", "match0", "match1"]
    # improve rejected the worse candidates: the world rolled back to 'generate'
    assert res.best_idx == 0 and res.blocks == 100
    assert [a.accepted for a in res.attempts] == [True, False, False]


def test_max_iters_bounds_an_improving_loop():
    res, llm, kit, trace = run({"max_iters": 3, "patience": 9},
                               rules=[(COMPARE, seq(prog(12), prog(14), prog(16), prog(18)))])
    assert stages(res) == ["generate", "match0", "match1", "match2"]
    assert res.blocks == 160 and res.best_idx == 3


def test_select_best_can_pick_a_rejected_attempt():
    small = lambda ep, a: ScoreTerm(1.0 / max(1, a.blocks))  # noqa: E731
    res, llm, kit, trace = run({"max_iters": 3, "patience": 9},
                               rules=[(COMPARE, seq(prog(5), prog(3), prog(8)))],
                               final_scorers={"small": (small, 1.0)})
    assert not res.attempts[2].accepted and res.best_idx == 2 and res.blocks == 30


# ------------------------------------------------------------------------ budget
def test_budget_exceeded_mid_loop_keeps_best_so_far():
    res, llm, kit, trace = run({"max_iters": 5, "patience": 9},
                               rules=[(COMPARE, seq(prog(20), prog(30), prog(40)))],
                               budget={"max_llm_calls": 3})
    assert res.truncated == "budget" and stages(res) == ["generate", "match0", "match1"]
    assert res.blocks == 300 and res.best_idx == 2
    assert "budget_stop" in trace.kinds()


def test_usd_budget_before_build_gives_empty_truncated_result():
    res, llm, kit, trace = run({"max_iters": 1}, budget={"usd_per_build": 0.005})
    assert res.truncated == "budget" and res.blocks == 0 and res.attempts == []


# --------------------------------------------------------------- patch / modes
def test_patch_mode_program_text_is_cumulative_and_replays():
    patches = seq("fill glass 0 1 0 3 1 0", "fill oak_planks 0 2 0 9 2 9")
    res, llm, kit, trace = run({"max_iters": 2, "mode": "patch", "patience": 9},
                               rules=[(COMPARE, patches)])
    assert [a.patch for a in res.attempts] == [False, True, True]
    assert res.program_text.count("# --- patch ---") == 2
    canvas, _, _ = run_program(res.program_text, size=(32, 32, 32))
    a, b = canvas.to_structure(), res.structure
    assert (a.block_ids == b.block_ids).all() and (a.block_data == b.block_data).all()
    compare = [r["messages"][-1].text for r in trace.requests if r["stage"] == "match0"][0]
    assert "ONLY the additional commands" in compare


def test_feedback_raw_shows_target_and_render_pairs_for_scored_dreams_only():
    res, llm, kit, trace = run({"max_iters": 1}, dreams=("iso", "mood"),
                               rules=[(COMPARE, prog(20))])
    msg = [r["messages"][-1] for r in trace.requests if r["stage"] == "match0"][0]
    labels = [p.text for p in msg.parts if p.kind == "text"]
    assert any(t.startswith("TARGET iso") for t in labels)
    assert any(t.startswith("YOURS iso") for t in labels)
    assert not any("mood" in t for t in labels)
    assert sum(p.kind == "image" for p in msg.parts) == 2
    assert msg.text.rstrip().endswith(kit.actions.output_rule())


def test_feedback_score_is_text_only():
    res, llm, kit, trace = run({"max_iters": 1, "feedback": "score"}, rules=[(COMPARE, prog(20))])
    msg = [r["messages"][-1] for r in trace.requests if r["stage"] == "match0"][0]
    assert all(p.kind == "text" for p in msg.parts) and "Score: 0.200" in msg.text


@pytest.mark.parametrize("feedback,n_images", [("vdm", 0), ("both", 2)])
def test_feedback_vdm_asks_the_judge(feedback, n_images):
    judge = json.dumps({"differences": ["roof is missing", "too narrow"]})
    res, llm, kit, trace = run({"max_iters": 1, "feedback": feedback},
                               rules=[(JUDGE, judge), (COMPARE, prog(20))])
    assert roles(trace).count("judge") == 1
    msg = [r["messages"][-1] for r in trace.requests if r["stage"] == "match0"][0]
    assert "- roof is missing" in msg.text and sum(p.kind == "image" for p in msg.parts) == n_images


def test_feedback_vdm_reuses_a_vlm_scorer_note():
    vlm = lambda ep, a: ScoreTerm(0.5, note="walls too short")  # noqa: E731
    res, llm, kit, trace = run({"max_iters": 1, "feedback": "vdm"}, rules=[(COMPARE, prog(20))],
                               scorers={"match": (match_scorer, 1.0), "vlm": (vlm, 1.0)})
    assert "judge" not in roles(trace)
    msg = [r["messages"][-1] for r in trace.requests if r["stage"] == "match0"][0]
    assert "walls too short" in msg.text


def test_plan_and_repair_run_before_the_match_loop():
    broken = "fill stone 0 0 0 9 0 9\nfill notablock 0 1 0 1 1 1\n"
    res, llm, kit, trace = run({"max_iters": 0, "plan": True, "repair_rounds": 2},
                               rules=[("plan the build", "PLAN: a box")],
                               build=seq(broken, prog(12)))
    assert stages(res) == ["generate", "repair1"] and res.plan_text == "PLAN: a box"
    assert roles(trace)[:1] == ["planner"]


def test_invalid_mode_and_feedback_raise_as_errors():
    res, *_ = run({"mode": "nope"})
    assert res.truncated == "error" and "mode" in res.error
    res, *_ = run({"feedback": "nope"})
    assert res.truncated == "error"
