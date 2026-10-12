"""Episode verbs (ask / try_ / render / dream / score / best) and run_episode."""

from __future__ import annotations

import pytest

from blockgen.agentic.core import (Budget, BudgetExceeded, CacheMiss, DreamSpec, Ledger,
                                   ScoreTerm, SpendNotAllowed)
from blockgen.agentic.dsl import run_program
from blockgen.agentic.episode import (REVIEWER_SYSTEM, Episode, extract_json,
                                      run_episode)
from blockgen.agentic.providers import Part

from .fakes import (FAKE_DREAMS, HUT_PROGRAM, FakeImageGen, FakeRenderer, RecordingTrace,
                    ScriptedLLM, fake_context, fake_scorer, make_fake_kit)


def pixel_scorer(ep, att):
    """fake_scorer, declared as reading renders (the Episode renders score views)."""
    return fake_scorer(ep, att)


pixel_scorer.needs_renders = True


def prog(n: int) -> str:
    """A floor of n x 10 blocks (n*10 blocks)."""
    return f"fill stone 0 0 0 {n - 1} 0 9\n"


def episode(**kw) -> Episode:
    trace = kw.pop("trace", None)
    return Episode("a small hut", make_fake_kit(**kw), trace=trace)


def improve_accept(prev, cand):
    if cand.blocks == 0:
        return False, "empty"
    if prev is None or prev.score is None:
        return True, "first"
    ok = cand.score.value > prev.score.value
    return ok, "improved" if ok else "not better"


# ---------------------------------------------------------------- extract_json
@pytest.mark.parametrize("text,want", [
    ('{"a": 1}', {"a": 1}),
    ('Here you go:\n```json\n{"a": [1, 2]}\n```', {"a": [1, 2]}),
    ('prose {"a": "}"} trailing', {"a": "}"}),
    ('{broken} then {"b": true}', {"b": True}),
    ("[1, 2]", {}),
    ("no json at all", {}),
    ("", {}),
])
def test_extract_json(text, want):
    assert extract_json(text) == want


# ------------------------------------------------------------------------- ask
def test_ask_builds_system_from_actions_and_contexts_and_meters():
    llm = ScriptedLLM(["hello"], model="gpt-5-mini", prompt_tokens=1000, completion_tokens=100)
    trace = RecordingTrace()
    ep = episode(llm=llm, trace=trace)
    assert ep.ask("Build it", stage="generate") == "hello"
    msgs = llm.calls[0]
    assert msgs[0].role == "system" and "FAKE GRAMMAR" in msgs[0].text
    assert "FAKE CONTEXT" in msgs[0].text
    assert [m.role for m in msgs] == ["system", "user"]
    # gen params are passed per call exactly like BuildAgent (cache-key parity)
    assert llm.overrides[0] == {"max_tokens": 16000, "temperature": None,
                                "reasoning_effort": "medium"}
    assert len(ep.ledger.charges) == 1 and ep.ledger.charges[0].role == "builder"
    assert ep.ledger.spent_usd > 0 and ep.ledger.charges[0].priced
    assert trace.requests[0]["stage"] == "generate"
    assert ep.transcript[0]["response"] == "hello"


def test_reviewer_roles_get_short_system_and_no_history():
    llm = ScriptedLLM(['{"score": 9}'])
    ep = episode(llm=llm)
    ep.history = [Part("text", text="x")]  # would break if it were sent
    v = ep.ask("judge this", role="dream_critic", history=False, json_reply=True)
    assert v == {"score": 9}
    assert llm.calls[0][0].text == REVIEWER_SYSTEM and len(llm.calls[0]) == 2


def test_json_reply_failure_returns_empty_and_traces():
    trace = RecordingTrace()
    ep = episode(replies=["not json"], trace=trace)
    assert ep.ask("verdict?", role="judge", json_reply=True, history=False) == {}
    assert "json_parse_fail" in trace.kinds()


def test_explicit_history_list_is_used_verbatim():
    llm = ScriptedLLM(["ok"])
    ep = episode(llm=llm)
    from blockgen.agentic.providers import text_message
    h = [text_message("user", "u0"), text_message("assistant", "a0")]
    ep.ask("turn", history=h)
    assert [m.text for m in llm.calls[0][1:]] == ["u0", "a0", "turn"]


def test_first_builder_turn_gets_context_images_and_target_dreams():
    from blockgen.agentic.core import ContextParts

    img_ctx = lambda ep: ContextParts(system_text="SHEET", user_parts=[  # noqa: E731
        Part("text", text="TEXTURE SHEET"), Part("image", image_png=b"sheet")])
    llm = ScriptedLLM([HUT_PROGRAM])
    ep = episode(llm=llm, dreams=["iso", "mood"], contexts=[("sheet", img_ctx)])
    ep.dream("iso")
    ep.dream("mood")
    ep.ask("Build it")
    ep.ask("Again")
    first, second = llm.calls[0][-1], llm.calls[1][-1]
    kinds = [p.kind for p in first.parts]
    texts = [p.text for p in first.parts if p.kind == "text"]
    assert kinds == ["text", "image", "text", "image", "text", "image", "text"]
    assert texts[0] == "TEXTURE SHEET" and texts[1].startswith("TARGET iso")
    assert "mood reference only" in texts[2] and texts[-1] == "Build it"
    assert first.parts[3].image_png == ep.latest_dream("iso").png
    assert [p.kind for p in second.parts] == ["text"]          # only the first turn
    assert "SHEET" in llm.calls[0][0].text


def test_inject_first_turn_can_be_disabled():
    llm = ScriptedLLM(["x"])
    ep = episode(llm=llm, dreams=["iso"])
    ep.dream("iso")
    ep.inject_first_turn = False
    ep.ask("Build it")
    assert [p.kind for p in llm.calls[0][-1].parts] == ["text"]


# ------------------------------------------------------------------------ try_
def test_try_accepts_and_extends_history():
    ep = episode(replies=[HUT_PROGRAM], scorers={"s": (pixel_scorer, 1.0)})
    reply = ep.ask("Build it")
    att = ep.try_("generate", reply)
    assert att.accepted and att.blocks > 0 and att.idx == 0
    assert ep.world is att.world and ep.current() is att
    assert [m.role for m in ep.history] == ["user", "assistant"]
    assert att.score.value == pytest.approx(att.blocks / 500)
    assert set(att.renders) == {"iso", "iso_back"}               # default score views
    assert att.response.text == HUT_PROGRAM


def test_empty_candidate_is_rejected_and_world_rolls_back():
    ep = episode(replies=[HUT_PROGRAM, "clear 0 0 0 31 31 31"])
    good = ep.try_("generate", ep.ask("Build"))
    bad = ep.try_("repair1", ep.ask("Fix"))
    assert not bad.accepted and bad.blocks == 0 and bad.score.gated == "empty"
    assert ep.world is good.world and ep.current() is good
    assert len(ep.history) == 2                                   # rejected pair dropped
    assert bad.renders == {}


def test_improve_policy_rejects_worse_and_best_picks_rejected_argmax():
    ep = episode(accept=improve_accept, spec_kw={"select": "best"})
    a = ep.try_("generate", prog(10))     # 100 blocks
    b = ep.try_("match0", prog(5))        # worse -> rejected
    c = ep.try_("match1", prog(20))       # better -> accepted
    assert a.accepted and not b.accepted and c.accepted
    assert ep.world is c.world
    # keep-best is separate from acceptance: make the rejected one the argmax
    b.score.value = 0.99
    assert ep.best() is b
    ep.select = "last"
    assert ep.best() is c


def test_best_ties_go_to_the_later_attempt_and_gated_rank_last():
    ep = episode(spec_kw={"select": "best"})
    a = ep.try_("generate", prog(10))
    b = ep.try_("match0", prog(10))
    assert ep.best() is b
    b.score.gated = "validity"
    assert ep.best() is a


def test_best_tie_prefers_the_accepted_attempt_over_a_later_rejected_one():
    ep = episode(spec_kw={"select": "best"})
    a = ep.try_("generate", prog(10))
    b = ep.try_("match0", prog(10))
    b.accepted = False                    # e.g. improve-only rejected an equal score
    assert ep.best() is a


def test_patch_program_text_is_cumulative_and_replays_to_same_blocks():
    ep = episode()
    base = ep.try_("generate", HUT_PROGRAM)
    p = ep.try_("critique1", "set glowstone 2 3 2\nfill glass 0 1 0 0 2 0", patch=True)
    assert p.patch and p.accepted
    assert p.program_text == base.program_text + "\n# --- patch ---\n" + p.delta_text
    canvas, _, _ = run_program(p.program_text, size=(32, 32, 32))
    s1, s2 = canvas.to_structure(), p.world.to_structure()
    assert (s1.block_ids == s2.block_ids).all() and (s1.block_data == s2.block_data).all()


def test_patch_paste_after_clone_keeps_clipboard():
    ep = episode()
    ep.try_("generate", "fill stone 0 0 0 1 0 1\ncopy 0 0 0 1 0 1\n")
    p = ep.try_("critique1", "paste 5 0 5", patch=True)
    assert p.blocks == 8


def test_bad_model_text_never_raises():
    ep = episode()
    att = ep.try_("generate", "this is not a program at all ###")
    assert att.blocks == 0 and att.score.value == 0


# ---------------------------------------------------------------------- render
def test_render_caches_on_attempt_and_handles_empty_and_no_renderer():
    renderer = FakeRenderer()
    ep = episode(renderer=renderer, scorers={"s": (pixel_scorer, 1.0)})
    att = ep.try_("generate", HUT_PROGRAM)
    n = len(renderer.calls)
    again = ep.render(att, ["iso", "plan"])
    assert set(again) == {"iso", "plan"} and len(renderer.calls) == n + 1
    assert renderer.calls[-1][1] == ("plan",)                     # only the missing view
    assert ep.render(ep.world, "iso")["iso"][:4] == b"\x89PNG"
    empty = ep.try_("x", "")
    assert ep.render(empty, ["iso"]) == {}
    ep2 = episode(renderer=None)
    assert ep2.render(ep2.try_("generate", HUT_PROGRAM), "score") == {}


def test_score_views_are_dream_views_plus_view_search():
    iso = DreamSpec("iso", "{desc}", view="iso", view_search=("iso_left",))
    ep = episode(dreams=[iso, FAKE_DREAMS["back"], FAKE_DREAMS["mood"]])
    assert ep._view_names("score") == ["iso", "iso_left", "iso_back"]


# ----------------------------------------------------------------------- score
def test_score_weights_and_gates():
    s1 = lambda ep, a: ScoreTerm(0.8, note="n1")  # noqa: E731
    s2 = lambda ep, a: ScoreTerm(0.2, per_view={"iso": 0.2})  # noqa: E731
    ep = episode(scorers={"a": (s1, 3.0), "b": (s2, 1.0)})
    att = ep.try_("generate", HUT_PROGRAM)
    assert att.score.value == pytest.approx((0.8 * 3 + 0.2) / 4)
    assert att.score.parts == {"a": 0.8, "b": 0.2} and att.score.notes == {"a": "n1"}
    assert att.score.per_view == {"b": {"iso": 0.2}} and att.score.dream_id == ""
    ep2 = episode(scorers={"a": (s1, 3.0), "b": (s2, 1.0)}, spec_kw={"gates": {"b": 0.5}})
    att2 = ep2.try_("generate", HUT_PROGRAM)
    assert att2.score.value == 0 and att2.score.gated == "b"


def test_final_score_drives_best_selection():
    small_wins = lambda ep, a: ScoreTerm(1.0 / max(1, a.blocks))  # noqa: E731
    ep = episode(spec_kw={"select": "best"}, final_scorers={"inv": (small_wins, 1.0)})
    a = ep.try_("generate", prog(3))
    ep.try_("match0", prog(20))
    assert ep.best() is a


# ----------------------------------------------------------------------- dream
def needs_dream_scorer(ep, att):
    dv = ep.latest_dream("iso")
    return ScoreTerm(min(1.0, att.blocks / 500) * (1.0 if dv is None else 1.0 / (1 + dv.version)))


needs_dream_scorer.needs_dream = True


def test_dream_versions_refs_and_ledger():
    gen = FakeImageGen()
    trace = RecordingTrace()
    ep = episode(image=gen, dreams=["iso", "back"], trace=trace)
    iso = ep.dream("iso")
    back = ep.dream("back")
    assert iso.version == 0 and back.version == 0
    assert gen.requests[1].refs == (iso.png,) and gen.requests[1].view == "iso_back"
    assert "a small hut" in iso.prompt and gen.requests[0].seed == ep.seed
    iso2 = ep.dream("iso", prompt="other", seed=5)
    assert iso2.version == 1 and ep.latest_dream("iso") is iso2
    assert ep.ledger.n_images == 3 and ep.ledger.spent_usd == pytest.approx(0.03)
    assert [d.key for d in trace.dreams] == ["iso", "back", "iso"]
    tmp = ep.dream("iso", seed=9, store=False)
    assert len(ep.dreams["iso"]) == 2 and tmp.version == 2


def test_refs_dropped_when_generator_cannot_take_them():
    trace = RecordingTrace()
    ep = episode(image=FakeImageGen(refs=False), dreams=["iso", "back"], trace=trace)
    ep.dream("iso")
    ep.dream("back")
    assert "refs_dropped" in trace.kinds()


def test_dream_without_image_generator_raises():
    ep = episode(image=None, dreams=["iso"])
    with pytest.raises(RuntimeError, match="no image generator"):
        ep.dream("iso")
    with pytest.raises(KeyError, match="unknown dream"):
        ep.dream("nope")


def test_max_images_caps_dreams():
    kit = make_fake_kit(dreams=["iso"])
    ep = Episode("x", kit, ledger=kit.ledger.child(Budget(max_images=1)))
    ep.dream("iso")
    with pytest.raises(BudgetExceeded):
        ep.dream("iso", seed=1)


def test_dream_id_and_rescore_all_reanchor_scores():
    ep = episode(dreams=["iso", "mood"], scorers={"m": (needs_dream_scorer, 1.0)})
    ep.dream("iso")
    did0 = ep.dream_id()
    att = ep.try_("generate", prog(10))
    assert att.score.dream_id == did0 and att.score.value == pytest.approx(0.2)
    ep.dream("mood")                         # context-only dreams do not change dream_id
    assert ep.dream_id() == did0 and not ep.stale()
    ep.dream("iso", prompt="new")
    assert ep.dream_id() != did0 and ep.stale()
    ep.rescore_all()
    assert att.score.dream_id == ep.dream_id() and att.score.value == pytest.approx(0.1)
    assert not ep.stale()


# ------------------------------------------------------------------ run_episode
def test_run_episode_result_fields():
    trace = RecordingTrace()
    kit = make_fake_kit(replies=[HUT_PROGRAM], spec_kw={"params": {"n": 2}})
    res = run_episode(kit, "a hut", trace=trace)
    assert res.blocks > 0 and res.error == "" and res.truncated == ""
    assert len(res.attempts) == 2 and res.best_idx == 1
    assert res.program_text == HUT_PROGRAM and res.spec_name == "fake"
    assert res.recipe_hash == kit.spec.hash() and res.provenance["method"] == "fake_method"
    assert len(res.charges) == 2 and trace.result is res
    assert res.metrics()["n_attempts"] == 2
    assert kit.ledger.charges == res.charges                   # propagated to the arm


def test_run_episode_seed_sets_variation():
    kit = make_fake_kit()
    seen = {}

    def method(ep):
        seen.update(seed=ep.seed, variation=ep.variation)

    kit.method = method
    run_episode(kit, "x")
    assert seen == {"seed": 0, "variation": None}
    run_episode(kit, "x", seed=7)
    assert seen == {"seed": 7, "variation": 7}


def test_budget_truncation_keeps_best_so_far():
    kit = make_fake_kit(replies=[HUT_PROGRAM], spec_kw={"params": {"n": 5},
                                                       "budget": {"max_llm_calls": 2}})
    trace = RecordingTrace()
    res = run_episode(kit, "a hut", trace=trace)
    assert res.truncated == "budget" and len(res.attempts) == 2 and res.blocks > 0
    assert "budget_stop" in trace.kinds() and "truncated: budget" in res.error


@pytest.mark.parametrize("exc,kind", [(CacheMiss("miss"), "cache_miss"),
                                      (ValueError("boom"), "error")])
def test_errors_truncate(exc, kind):
    kit = make_fake_kit()

    def method(ep):
        ep.try_("generate", HUT_PROGRAM)
        raise exc

    kit.method = method
    trace = RecordingTrace()
    res = run_episode(kit, "x", trace=trace)
    assert res.truncated == kind and res.blocks > 0
    if kind == "error":
        assert "boom" in res.error and any("traceback" in d for k, d in trace.events)


def test_spend_not_allowed_propagates():
    kit = make_fake_kit()

    def method(ep):
        raise SpendNotAllowed("no")

    kit.method = method
    with pytest.raises(SpendNotAllowed):
        run_episode(kit, "x")


def test_empty_episode_result():
    kit = make_fake_kit()
    kit.method = lambda ep: None
    res = run_episode(kit, "x")
    assert res.blocks == 0 and res.best_idx is None and "empty build" in res.error
    assert res.structure.block_ids.shape == (32, 32, 32)


def test_run_episode_rescores_stale_attempts_at_the_end():
    kit = make_fake_kit(dreams=["iso"], scorers={"m": (needs_dream_scorer, 1.0)})

    def method(ep):
        ep.dream("iso")
        ep.try_("generate", prog(10))
        ep.dream("iso", prompt="v1")          # no explicit rescore

    kit.method = method
    res = run_episode(kit, "x")
    did = res.attempts[0].score.dream_id
    assert did and res.attempts[0].score.value == pytest.approx(0.1)


def test_new_branch_isolates_current_but_keeps_attempts():
    ep = episode(spec_kw={"select": "best"})
    a = ep.try_("generate", prog(20))
    ep.new_branch()
    assert ep.current() is None and ep.world.block_count() == 0 and ep.history == []
    b = ep.try_("generate", prog(5))
    assert ep.current() is b and ep.best() is a and b.idx == 1


def test_contexts_called_once():
    calls = []

    def ctx(ep):
        calls.append(1)
        return fake_context(ep)

    ep = episode(contexts=[("c", ctx)])
    ep.ask("a")
    ep.ask("b")
    assert len(calls) == 1


def test_ledger_child_defaults_to_kit_ledger():
    kit = make_fake_kit()
    ep = Episode("p", kit)
    assert ep.ledger.parent is kit.ledger and isinstance(ep.ledger, Ledger)


def test_example_turns_from_contexts_become_prior_turns_when_enabled():
    from blockgen.agentic.core import ContextParts
    from blockgen.agentic.providers import text_message

    class Ex(ContextParts):
        pass

    turns = [text_message("user", "Build: demo"), text_message("assistant", "DEMO")]

    def ex_ctx(ep):
        parts = Ex(user_parts=[Part("text", text="FALLBACK TEXT")])
        parts.turns = turns
        return parts

    for keep in (True, False):
        llm = ScriptedLLM([HUT_PROGRAM])
        ep = episode(llm=llm, contexts=[("examples", ex_ctx)], spec_kw={"keep_history": keep})
        ep.inserts_example_turns = True
        ep.ask("Build it")
        msgs = llm.calls[0]
        assert [m.text for m in msgs[1:3]] == ["Build: demo", "DEMO"]
        assert (ep.history[:2] == turns) is keep
    llm = ScriptedLLM([HUT_PROGRAM])
    ep = episode(llm=llm, contexts=[("examples", ex_ctx)])
    ep.ask("Build it")                                  # default: text fallback only
    assert len(llm.calls[0]) == 2 and "FALLBACK TEXT" in llm.calls[0][-1].text


# ------------------------------------------------- scorer crash keeps the build
def test_scorer_crash_keeps_the_build_and_ranks_it_last():
    """A CUDA OOM in a scorer must not discard a build the model already paid for."""
    calls = {"n": 0}

    def flaky(ep, att):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory")
        return fake_scorer(ep, att)

    ep = episode(spec_kw={"select": "best"}, scorers={"flaky": (flaky, 1.0)})
    with pytest.raises(RuntimeError, match="out of memory"):
        ep.try_("generate", prog(10))
    (a,) = ep.attempts
    assert a.blocks == 100 and not a.accepted and a.score.gated == "score_error"
    assert ep.best() is a                       # the only build: still returned
    b = ep.try_("match0", prog(3))
    assert ep.best() is b                       # any scored build outranks it


def test_resolve_device_falls_back_to_cpu_when_the_gpu_is_full(monkeypatch):
    import torch
    from blockgen.agentic import scoring
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (300 * 2 ** 20, 16 * 2 ** 30))
    with pytest.warns(RuntimeWarning, match="GiB free on the GPU"):
        assert scoring.resolve_device() == "cpu"
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (8 * 2 ** 30, 16 * 2 ** 30))
    assert scoring.resolve_device() == "cuda"
    assert scoring.resolve_device("cuda") == "cuda"


def test_best_accepted_never_picks_a_rejected_attempt():
    ep = episode(spec_kw={"select": "best_accepted"})
    a = ep.try_("generate", prog(10))
    b = ep.try_("match0", prog(5))
    b.accepted = False
    b.score.value = 0.99                  # the selector likes it, the accept policy did not
    assert ep.best() is a
    a.accepted = False                    # nothing accepted: still return a build
    assert ep.best() is b
