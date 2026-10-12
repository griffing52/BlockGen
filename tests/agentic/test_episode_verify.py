"""Regression tests from the WP2 verification pass.

Each test pins one defect found in review: stale empty attempts forcing a second
full rescore, non-atomic rescoring, final-rescore failures mislabelled, REDREAM
replies the loop missed, a planner blind to the dream, duplicated reference
images, re-dreams that ended the build on an image cap, silent no-op knobs
(redream_patience > patience, patch on a non-patchable action space),
best_of_n re-paying the dream phase per branch, and legacy ignoring
config.provider.
"""

from __future__ import annotations

import dataclasses
import functools
import json

import pytest

from blockgen.agentic import turns
from blockgen.agentic.agent import AgentConfig
from blockgen.agentic.core import BudgetExceeded, ScoreTerm
from blockgen.agentic.episode import Episode, run_episode
from blockgen.agentic.recipe import from_agent_config
from blockgen.agentic.recipes.classic import classic
from blockgen.agentic.recipes.dream import dream
from blockgen.agentic.recipes.legacy import legacy
from blockgen.agentic.recipes.variants import best_of_n
from blockgen.experiments_agentic import ARMS

from .fakes import FakeActions, FakeImageGen, RecordingTrace, ScriptedLLM, make_fake_kit


def prog(n: int) -> str:
    return f"fill stone 0 0 0 {n - 1} 0 9\n"


def seq(*items):
    left = list(items)

    def reply(_msgs):
        return left.pop(0) if len(left) > 1 else left[0]
    return reply


class CountingScorer:
    """Dream-dependent scorer that counts its calls (a stand-in for a paid judge)."""

    needs_dream = True

    def __init__(self) -> None:
        self.calls = 0
        self.fail_after: int = -1

    def __call__(self, ep, att):
        self.calls += 1
        if self.fail_after >= 0 and self.calls > self.fail_after:
            raise BudgetExceeded("judge budget gone")
        dv = ep.latest_dream("iso")
        return ScoreTerm(min(1.0, att.blocks / 500.0) / (1 + (dv.version if dv else 0)))


def improve_accept(prev, cand):
    if cand.blocks == 0:
        return False, "empty"
    if prev is None or prev.score is None:
        return True, "first"
    ok = cand.score.value > prev.score.value
    return ok, "improved" if ok else "not better"


COMPARE, REACH = "Compare your build with the target", "BEST BUILD SO FAR"


def dream_run(params, *, rules=(), scorer=None, dreams=("iso",), budget=None, select="best",
              image=None, **kw):
    scorer = scorer or CountingScorer()
    llm = ScriptedLLM([prog(10)], rules=list(rules))
    kit = make_fake_kit(llm=llm, method=dream, dreams=list(dreams), accept=improve_accept,
                        scorers={"match": (scorer, 1.0)}, image=image or FakeImageGen(),
                        spec_kw={"params": params, "select": select, "budget": budget or {}},
                        **kw)
    trace = RecordingTrace()
    return run_episode(kit, "a small hut", trace=trace), llm, kit, trace, scorer


# ------------------------------------------------------------------- rescoring
def test_redream_restamps_empty_attempts_so_there_is_no_second_full_rescore():
    res, llm, kit, trace, scorer = dream_run(
        {"max_iters": 3, "redream": 1, "stop_at": 2.0, "patience": 5},
        rules=[(COMPARE, seq("", "REDREAM: a shed", prog(20)))])
    assert [a.blocks for a in res.attempts] == [100, 0, 200]
    final = res.attempts[-1].score.dream_id
    assert all(a.score.dream_id == final for a in res.attempts)       # the empty one too
    assert trace.kinds().count("rescore_all") == 1                     # no end-of-run repeat
    # generate + its rescore + match2: the scorer never re-ran on a current score
    assert scorer.calls == 3


def test_rescore_all_is_atomic_and_skips_current_scores():
    scorer = CountingScorer()
    kit = make_fake_kit(dreams=["iso"], scorers={"m": (scorer, 1.0)})
    ep = Episode("a hut", kit)
    ep.dream("iso")
    a, b = ep.try_("generate", prog(10)), ep.try_("retry", prog(20))
    old = (a.score.dream_id, b.score.dream_id)
    ep.rescore_all()                         # nothing changed: no scorer call
    assert scorer.calls == 2
    ep.dream("iso", prompt="v1")
    scorer.fail_after = 3                    # the second recomputation fails
    with pytest.raises(BudgetExceeded):
        ep.rescore_all()
    assert (a.score.dream_id, b.score.dream_id) == old   # no half-rescored mix


def test_best_compares_only_current_scores_when_a_rescore_failed():
    scorer = CountingScorer()
    kit = make_fake_kit(dreams=["iso"], scorers={"m": (scorer, 1.0)},
                        spec_kw={"select": "best"})
    ep = Episode("a hut", kit)
    ep.dream("iso")
    big = ep.try_("generate", prog(30))      # 0.6 against v0
    ep.dream("iso", prompt="v1")
    small = ep.try_("retry", prog(20))       # 0.2 against v1 (v1 halves the score)
    assert big.score.value > small.score.value and ep.stale()
    assert ep.best() is small                # v0 and v1 scores are not comparable


def test_final_rescore_budget_failure_is_labelled_budget():
    scorer = CountingScorer()

    def method(ep):
        ep.dream("iso")
        ep.try_("generate", prog(10))
        ep.dream("iso", prompt="v1")         # stale, no explicit rescore
        scorer.fail_after = scorer.calls

    kit = make_fake_kit(dreams=["iso"], scorers={"m": (scorer, 1.0)}, method=method)
    trace = RecordingTrace()
    res = run_episode(kit, "a hut", trace=trace)
    assert res.truncated == "budget" and "rescore_error" in trace.kinds()
    assert res.blocks == 100


# --------------------------------------------------------------------- REDREAM
@pytest.mark.parametrize("reply,want", [
    ("REDREAM: a hut", "a hut"),
    ("`REDREAM: a hut`", "a hut"),
    ("```\nREDREAM: a hut\n```", "a hut"),
    ("```text\nredream: a hut\n```", "a hut"),
    ("  \n Redream:   ", ""),
    ("fill stone 0 0 0 1 1 1\nREDREAM: x", None),
    ("", None),
])
def test_parse_redream(reply, want):
    assert turns.parse_redream(reply) == want


def test_backticked_redream_reply_is_a_redream_not_a_program():
    res, llm, kit, trace, _ = dream_run(
        {"max_iters": 2, "redream": 1, "stop_at": 2.0, "patience": 5},
        rules=[(COMPARE, seq("`REDREAM: a tiny shed`", prog(20)))])
    assert len(res.dreams["iso"]) == 2 and "a tiny shed" in res.dreams["iso"][1].prompt
    assert [a.stage for a in res.attempts] == ["generate", "match1"]


def test_redream_that_does_not_fit_the_image_cap_is_skipped_and_the_loop_goes_on():
    gen = FakeImageGen()
    res, llm, kit, trace, _ = dream_run(
        {"max_iters": 3, "redream": 1, "stop_at": 2.0, "patience": 5},
        dreams=("iso", "back"), image=gen, budget={"max_images": 3},
        rules=[(COMPARE, seq("REDREAM: a shed", prog(20), prog(30)))])
    assert gen.calls == 2                                  # no half re-dream (iso without back)
    assert "redream_budget_stop" in trace.kinds() and res.truncated == ""
    assert [a.stage for a in res.attempts] == ["generate", "match1", "match2"]
    compare = [r["messages"][-1].text for r in trace.requests if r["stage"] == "match1"][0]
    assert turns.REDREAM_TOKEN not in compare              # the hatch is closed afterwards


def test_stall_redream_skips_the_reachability_call_without_image_budget():
    res, llm, kit, trace, _ = dream_run(
        {"max_iters": 3, "redream": 1, "redream_patience": 1, "patience": 3},
        budget={"max_images": 1},
        rules=[(REACH, json.dumps({"unreachable": True})), (COMPARE, prog(5))])
    assert "dream_critic" not in [r["role"] for r in trace.requests]
    assert "redream_budget_stop" in trace.kinds() and len(res.dreams["iso"]) == 1


def test_redream_patience_above_patience_is_an_error():
    res, *_ = dream_run({"redream": 1, "redream_patience": 3, "patience": 2})
    assert res.truncated == "error" and "redream_patience" in res.error


class NoPatch(FakeActions):
    patchable = False


def test_patch_mode_needs_a_patchable_action_space():
    res, *_ = dream_run({"mode": "patch"}, actions=NoPatch())
    assert res.truncated == "error" and "patchable" in res.error
    kit = make_fake_kit(method=classic, actions=NoPatch(),
                        spec_kw={"params": {"critique_rounds": 1, "critique_mode": "patch"}})
    res = run_episode(kit, "a hut")
    assert res.truncated == "error" and "patchable" in res.error


# ---------------------------------------------------------------------- turns
def test_planner_sees_the_dreams():
    res, llm, kit, trace, _ = dream_run({"max_iters": 0, "plan": True},
                                        dreams=("iso", "mood"),
                                        rules=[("plan the build", "PLAN: a box")])
    plan = [r["messages"][-1] for r in trace.requests if r["role"] == "planner"][0]
    labels = [p.text for p in plan.parts if p.kind == "text" and p.text.startswith("TARGET")]
    assert [lab.split()[1] for lab in labels] == ["iso", "mood"]
    assert sum(p.kind == "image" for p in plan.parts) == 2
    assert res.plan_text == "PLAN: a box"


def test_plan_without_dreams_is_text_only():
    kit = make_fake_kit()
    ep = Episode("a hut", kit)
    msg = turns.plan(ep)
    assert all(p.kind == "text" for p in msg.parts) and "TARGET" not in msg.text


def test_reference_context_is_not_duplicated_by_the_build_turn():
    ref = b"\x89PNG-ref"
    kit = make_fake_kit()
    ep = Episode("a hut", kit, images=[ref])
    assert sum(p.kind == "image" for p in turns.build(ep).parts) == 1
    kit.contexts = kit.contexts + [("reference", lambda ep: None)]
    assert sum(p.kind == "image" for p in turns.build(ep).parts) == 0


# ------------------------------------------------------------------- variants
REVIEW = "IMAGE TO REVIEW"


def test_best_of_n_shares_the_dreams_across_branches():
    gen = FakeImageGen()
    bad = json.dumps({"score": 2, "buildable": False})
    llm = ScriptedLLM([prog(10), prog(30), prog(20)], rules=[(REVIEW, bad)])
    scorer = CountingScorer()
    method = functools.partial(best_of_n, n=3, inner="dream",
                               inner_params={"refine": 1, "max_iters": 0, "repair_rounds": 0})
    kit = make_fake_kit(llm=llm, method=method, dreams=["iso"], image=gen,
                        scorers={"m": (scorer, 1.0)})
    trace = RecordingTrace()
    res = run_episode(kit, "a hut", trace=trace)
    assert gen.calls == 2 and len(res.dreams["iso"]) == 2         # refined once, not per branch
    assert [r["role"] for r in trace.requests].count("dream_critic") == 1
    assert [a.stage for a in res.attempts] == ["generate"] * 3
    assert res.best_idx == 1 and res.blocks == 300
    assert len({a.score.dream_id for a in res.attempts}) == 1


def test_best_of_n_without_scores_is_an_error():
    kit = make_fake_kit(method=best_of_n, scorers={})
    res = run_episode(kit, "a hut")
    assert res.truncated == "error" and "score" in res.error


# --------------------------------------------------------------------- legacy
def test_legacy_rejects_a_config_provider_it_would_not_call():
    cfg = AgentConfig(provider="mock", verbose=False, **ARMS["zeroshot"])
    spec = from_agent_config(cfg, name="legacy-test")
    spec = dataclasses.replace(spec, params={"config": {**spec.params["config"],
                                                        "provider": "openai:gpt-5"}})
    llm = ScriptedLLM([prog(10)])
    kit = make_fake_kit(llm=llm, spec=spec, method=legacy)
    res = run_episode(kit, "a hut")
    assert res.truncated == "error" and "provider" in res.error and llm.n_calls == 0
