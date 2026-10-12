"""``legacy``: BuildAgent wrapped in the suite harness, bit-identical output."""

from __future__ import annotations

import dataclasses

import pytest

from blockgen.agentic.agent import AgentConfig, BuildAgent
from blockgen.agentic.episode import run_episode
from blockgen.agentic.providers import CachingProvider, ScriptedProvider
from blockgen.agentic.recipe import from_agent_config
from blockgen.agentic.recipes.legacy import MeteredProvider, legacy, legacy_config
from blockgen.experiments_agentic import ARMS

from .fakes import RecordingTrace, ScriptedLLM, fake_scorer, make_fake_kit

BROKEN = "fill stone 0 0 0 9 0 9\nfill notablock 0 1 0 1 1 1\n"
FIXED = "fill stone 0 0 0 9 0 9\nwalls oak_planks 0 1 0 9 3 9\n"
EMPTY = "clear 0 0 0 31 31 31\n"


def _no_render(self, structure):
    return []


def legacy_kit(cfg: AgentConfig, provider, **kw):
    spec = from_agent_config(cfg, name="legacy-test")
    spec = dataclasses.replace(spec, renderer="fake_renderer", **kw.pop("spec_kw", {}))
    return make_fake_kit(llm=provider, spec=spec, method=legacy, **kw)


def test_legacy_is_bit_identical_to_buildagent_and_meters_every_call(monkeypatch):
    monkeypatch.setattr(BuildAgent, "_render", _no_render)
    cfg = AgentConfig(provider="mock", canvas_size=(32, 32, 32), verbose=False, **ARMS["full"])
    script = ["PLAN", BROKEN, FIXED]
    want = BuildAgent(ScriptedProvider(replies=script), cfg).build("a hut")

    llm = ScriptedLLM(script, model="gpt-5-mini", prompt_tokens=100, completion_tokens=10)
    kit = legacy_kit(cfg, llm, scorers={})
    trace = RecordingTrace()
    got = run_episode(kit, "a hut", trace=trace)
    assert got.truncated == "", got.error
    assert got.program_text == want.program_text
    assert (got.structure.block_ids == want.structure.block_ids).all()
    assert (got.structure.block_data == want.structure.block_data).all()
    assert [a.stage for a in got.attempts] == [r.stage for r in want.rounds]
    assert got.plan_text == "PLAN" and got.transcript[0]["stage"] == "plan"
    # the plan call is metered too (BuildResult.rounds would miss it)
    assert len(got.charges) == 3 == llm.n_calls and got.cost_known
    assert got.cost_usd == pytest.approx(3 * (100 * 0.25 + 10 * 2.0) / 1e6)
    assert len(trace.requests) == 3 and isinstance(got.config, AgentConfig)
    assert got.config.plan and got.config.n_examples == 1


def test_rollback_flags_and_scores(monkeypatch):
    monkeypatch.setattr(BuildAgent, "_render", _no_render)
    cfg = AgentConfig(provider="mock", canvas_size=(32, 32, 32), verbose=False,
                      **ARMS["repair"])
    def pixel_scorer(ep, att):        # reads renders: the Episode renders score views
        return fake_scorer(ep, att)
    pixel_scorer.needs_renders = True
    kit = legacy_kit(cfg, ScriptedProvider(replies=[BROKEN, EMPTY]),
                     scorers={"s": (pixel_scorer, 1.0)})
    got = run_episode(kit, "a hut")
    assert [a.accepted for a in got.attempts] == [True, False]
    assert got.best_idx == 0 and got.blocks == 100
    assert got.attempts[0].score.value == pytest.approx(100 / 500)
    assert set(got.attempts[0].renders) == {"iso", "iso_back"}
    assert got.attempts[1].renders == {} and got.attempts[1].score.gated == "empty"


def test_arm_param_and_config_override_precedence():
    spec = from_agent_config(AgentConfig(provider="mock", canvas_size=(16, 16, 16)))

    class Ep:
        pass

    ep = Ep()
    ep.spec = dataclasses.replace(spec, target_blocks=500)
    cfg = legacy_config(ep, "oneshot", {"repair_rounds": 3})
    assert cfg.n_examples == 1 and cfg.repair_rounds == 3 and cfg.target_blocks == 500
    assert cfg.canvas_size == (16, 16, 16) and cfg.provider == "mock" and not cfg.verbose
    with pytest.raises(KeyError, match="unknown legacy arm"):
        legacy_config(ep, "nope", None)


def test_legacy_hits_the_same_cache_entries_as_buildagent(tmp_path, monkeypatch):
    """Requests are byte-identical, so a run through the suite replays BuildAgent's
    cached responses (the inner provider here would answer differently)."""
    monkeypatch.setattr(BuildAgent, "_render", _no_render)
    cfg = AgentConfig(provider="mock", canvas_size=(32, 32, 32), verbose=False, **ARMS["full"])
    gen = cfg.provider_params()
    first = CachingProvider(ScriptedProvider(replies=["PLAN", BROKEN, FIXED], **gen), tmp_path)
    want = BuildAgent(first, cfg).build("a hut")
    second = CachingProvider(ScriptedProvider(replies=["WRONG"], **gen), tmp_path)
    got = run_episode(legacy_kit(cfg, second, scorers={}), "a hut")
    assert got.program_text == want.program_text and got.cached
    assert got.cost_usd == 0.0


def test_budget_is_checked_before_the_build(monkeypatch):
    cfg = AgentConfig(provider="mock", verbose=False, **ARMS["zeroshot"])
    llm = ScriptedLLM([FIXED])
    kit = legacy_kit(cfg, llm, spec_kw={"budget": {"max_llm_calls": 0}})
    got = run_episode(kit, "a hut")
    assert got.truncated == "budget" and llm.n_calls == 0 and got.blocks == 0


def test_metered_provider_is_transparent():
    inner = ScriptedLLM(["x"], model="gpt-5-mini")

    class Ep:
        pass

    ep = Ep()
    from blockgen.agentic.core import Ledger, NullTrace
    ep.ledger, ep.trace = Ledger(), NullTrace()
    m = MeteredProvider(inner, ep)
    assert m.describe() == inner.describe() and m.billing == "mock"
    assert m.complete([]).text == "x" and len(ep.ledger.charges) == 1
    assert ep.ledger.charges[0].billing == "mock"
