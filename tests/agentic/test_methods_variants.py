"""``best_of_n`` and ``dream_vote``."""

from __future__ import annotations

import json

from blockgen.agentic.core import ScoreTerm
from blockgen.agentic.episode import run_episode
from blockgen.agentic.recipes.variants import best_of_n, dream_vote

from .fakes import FakeImageGen, RecordingTrace, ScriptedLLM, make_fake_kit


def prog(n: int) -> str:
    return f"fill stone 0 0 0 {n - 1} 0 9\n"


def test_best_of_n_returns_the_best_scoring_candidate():
    llm = ScriptedLLM([prog(10), prog(30), prog(20)])
    kit = make_fake_kit(llm=llm, method=best_of_n,
                        spec_kw={"params": {"n": 3, "inner_params": {"repair_rounds": 0}},
                                 "select": "last"})
    trace = RecordingTrace()
    res = run_episode(kit, "a hut", trace=trace)
    assert [a.stage for a in res.attempts] == ["generate"] * 3
    assert res.best_idx == 1 and res.blocks == 300            # select forced to best
    turns = [c[-1].text for c in llm.calls]
    assert "Design variation" not in turns[0]
    assert "Design variation #1" in turns[1] and "Design variation #2" in turns[2]
    # every branch starts from scratch: no earlier dialogue in its request
    assert all(len(c) == 2 for c in llm.calls)
    assert trace.kinds().count("branch") == 3


def test_best_of_n_validates_inner_params():
    kit = make_fake_kit(method=best_of_n, spec_kw={"params": {"inner_params": {"nope": 1}}})
    res = run_episode(kit, "a hut")
    assert res.truncated == "error" and "nope" in res.error


def match(ep, att):
    return ScoreTerm(min(1.0, att.blocks / 500.0))


match.needs_dream = True


def vote_kit(pick_reply, *, params=None, dreams=("iso",)):
    gen = FakeImageGen()
    llm = ScriptedLLM([prog(10)], rules=[("CANDIDATE 0", pick_reply)])
    kit = make_fake_kit(llm=llm, method=dream_vote, image=gen, dreams=list(dreams),
                        scorers={"match": (match, 1.0)},
                        spec_kw={"params": {"n_dreams": 3, "max_iters": 0, **(params or {})},
                                 "select": "best"})
    return kit, gen, llm


def test_dream_vote_picks_the_indexed_candidate():
    kit, gen, llm = vote_kit(json.dumps({"index": 2, "reason": "simplest"}), dreams=("iso", "back"))
    trace = RecordingTrace()
    res = run_episode(kit, "a hut", trace=trace)
    assert [r.seed for r in gen.requests[:3]] == [0, 1, 2]
    iso = res.dreams["iso"]
    assert len(iso) == 1 and iso[0].seed == 2 and iso[0].version == 0
    assert iso[0].critique["index"] == 2
    assert gen.requests[3].view == "iso_back" and gen.requests[3].refs == (iso[0].png,)
    assert gen.calls == 4                                       # iso not regenerated
    pick_msg = llm.calls[0][-1]
    assert sum(p.kind == "image" for p in pick_msg.parts) == 3
    assert res.blocks == 100 and "dream_vote" in trace.kinds()
    first_build = llm.calls[1][-1]
    assert first_build.parts[1].image_png == iso[0].png


def test_dream_vote_wraps_and_tolerates_bad_indices():
    kit, gen, llm = vote_kit('{"index": 4}')
    assert run_episode(kit, "a hut").dreams["iso"][0].seed == 1
    kit, gen, llm = vote_kit("the second one")
    assert run_episode(kit, "a hut").dreams["iso"][0].seed == 0


def test_dream_vote_validates_loop_params():
    kit, gen, llm = vote_kit('{"index": 0}', params={"max_iterz": 2})
    res = run_episode(kit, "a hut")
    assert res.truncated == "error" and "max_iterz" in res.error and gen.calls == 0


# ------------------------------------------------------- registry / build_kit
def test_methods_are_registered_and_resolve_through_build_kit():
    import pytest

    from blockgen.agentic import core
    from blockgen.agentic import kit as kitmod

    from .fakes import fake_spec

    assert {"legacy", "classic", "dream", "best_of_n", "dream_vote"} <= set(core.names("method"))
    kitmod.clear_cache()
    try:
        spec = fake_spec(method="dream", llm="mock", image="fake_image", dreams=["iso", "plan"],
                         params={"max_iters": 1, "refine": 1}, select="best")
        kit = kitmod.build_kit(spec)
        assert [d.key for d in kit.dreams] == ["iso", "plan"]
        res = run_episode(kit, "a hut")
        assert res.truncated == "" and res.blocks > 0
        assert set(res.dreams) == {"iso", "plan"}
        with pytest.raises(TypeError, match="max_iterz"):
            kitmod.build_kit(fake_spec(method="dream", llm="mock", params={"max_iterz": 1}))
        with pytest.raises(TypeError, match="inner_param"):
            kitmod.build_kit(fake_spec(method="best_of_n", llm="mock", params={"inner_param": {}}))
    finally:
        kitmod.clear_cache()
