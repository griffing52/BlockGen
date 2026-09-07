"""Agent-loop tests, driven entirely by the scripted provider.

No network, no API key: :class:`ScriptedProvider` replays canned completions
through the *real* loop, so the repair/critique control flow, the transcript, the
metrics and the run artifacts are all covered offline.
"""

from __future__ import annotations

import json

from blockgen.agentic.agent import AgentConfig, BuildAgent
from blockgen.agentic.providers import (CachingProvider, ScriptedProvider,
                                        get_provider, text_message)
from blockgen.agentic.report import aggregate, write_run
from blockgen.agentic.tasks import load_prompts

BROKEN = "fill notablock 0 0 0 3 3 3\nfill stone 0 0 0 3 0 3"
FIXED = "fill cobblestone 0 0 0 3 0 3\nwalls oak_planks 0 1 0 3 3 3"
EMPTY = "# nothing here"


def _agent(replies, **cfg_kw):
    cfg = AgentConfig(canvas_size=(16, 16, 16), verbose=False, **cfg_kw)
    return BuildAgent(ScriptedProvider(replies=replies), cfg)


def test_single_pass_build():
    agent = _agent([FIXED], repair_rounds=0)
    result = agent.build("a small house")
    assert result.blocks > 0
    assert result.report.n_failed == 0
    assert [r.stage for r in result.rounds] == ["generate"]


def test_repair_round_fixes_a_broken_program():
    agent = _agent([BROKEN, FIXED], repair_rounds=1)
    result = agent.build("a small house")
    assert [r.stage for r in result.rounds] == ["generate", "repair1"]
    assert result.rounds[0].report.n_failed == 1
    assert result.report.n_failed == 0            # the repaired program is kept
    assert result.program_text == FIXED


def test_repair_is_skipped_when_the_program_is_clean():
    agent = _agent([FIXED, BROKEN], repair_rounds=2)
    result = agent.build("a small house")
    assert len(result.rounds) == 1                # never asked for a repair


def test_an_empty_repair_does_not_destroy_a_good_build():
    agent = _agent([BROKEN, EMPTY], repair_rounds=1)
    result = agent.build("a small house")
    assert result.blocks > 0
    assert result.program_text == BROKEN          # rolled back to what worked


def test_plan_stage_runs_first_and_is_recorded():
    agent = _agent(["PLAN: a 4x4 hut", FIXED], plan=True, repair_rounds=0)
    result = agent.build("a hut")
    assert result.plan_text.startswith("PLAN")
    assert [t["stage"] for t in result.transcript] == ["plan", "generate"]


def test_examples_are_injected_as_prior_turns():
    provider = ScriptedProvider(replies=[FIXED])
    agent = BuildAgent(provider, AgentConfig(canvas_size=(16, 16, 16), n_examples=1,
                                             repair_rounds=0, verbose=False))
    agent.build("a small oak cottage")
    roles = [m.role for m in provider.calls[0]]
    assert roles == ["system", "user", "assistant", "user"]   # one demo pair


def test_no_examples_by_default():
    provider = ScriptedProvider(replies=[FIXED])
    BuildAgent(provider, AgentConfig(canvas_size=(16, 16, 16), repair_rounds=0,
                                     verbose=False)).build("a house")
    assert [m.role for m in provider.calls[0]] == ["system", "user"]


def test_system_prompt_carries_the_command_reference_and_canvas():
    provider = ScriptedProvider(replies=[FIXED])
    BuildAgent(provider, AgentConfig(canvas_size=(16, 20, 16), repair_rounds=0,
                                     verbose=False)).build("a house")
    system = provider.calls[0][0].text
    assert "gable" in system and "walls" in system
    assert "16 wide (x) x 20 tall (y) x 16 deep (z)" in system


def test_repair_prompt_quotes_the_failing_line():
    provider = ScriptedProvider(replies=[BROKEN, FIXED])
    BuildAgent(provider, AgentConfig(canvas_size=(16, 16, 16), repair_rounds=1,
                                     verbose=False)).build("a house")
    repair_turn = provider.calls[1][-1].text
    assert "notablock" in repair_turn and "line 1" in repair_turn


def test_metrics_shape():
    result = _agent([FIXED], repair_rounds=0).build("a small house")
    m = result.metrics()
    assert m["blocks"] > 0 and m["n_components"] == 1
    assert m["command_success_rate"] == 1.0
    assert set(("blocks", "n_commands", "cost_usd", "elapsed_s")) <= set(m)


def test_disconnected_build_is_visible_in_the_metrics():
    prog = "fill stone 0 0 0 1 1 1\nfill stone 10 10 10 11 11 11"
    result = _agent([prog], repair_rounds=0).build("two islands")
    m = result.metrics()
    assert m["n_components"] == 2 and m["largest_component_frac"] == 0.5


def test_empty_build_is_reported_not_raised():
    result = _agent([EMPTY], repair_rounds=0).build("nothing")
    assert result.blocks == 0 and result.error


# --- providers -------------------------------------------------------------
def test_caching_provider_returns_the_cached_text(tmp_path):
    inner = ScriptedProvider(replies=["first", "second"])
    cached = CachingProvider(inner, cache_dir=tmp_path, enabled=True)
    msgs = [text_message("user", "hello")]
    assert cached.complete(msgs).text == "first"
    again = cached.complete(msgs)
    assert again.text == "first" and again.cached          # not "second"
    assert len(inner.calls) == 1                            # inner never re-called


def test_cache_key_separates_different_requests(tmp_path):
    inner = ScriptedProvider(replies=["first", "second"])
    cached = CachingProvider(inner, cache_dir=tmp_path, enabled=True)
    assert cached.complete([text_message("user", "a")]).text == "first"
    assert cached.complete([text_message("user", "b")]).text == "second"


def test_get_provider_rejects_an_unknown_vendor():
    try:
        get_provider("notavendor:model")
    except ValueError as exc:
        assert "unknown provider" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_mock_provider_needs_no_key():
    provider = get_provider("mock")
    assert provider.complete([text_message("user", "x")]).text


# --- prompt sets / reporting ----------------------------------------------
def test_prompt_sets_load():
    assert len(load_prompts("short", n=3)) == 3
    assert len(load_prompts("detailed", n=2)) == 2
    assert load_prompts("large", n=1)[0]


def test_write_run_emits_the_standard_artifacts(tmp_path):
    results = [_agent([FIXED], repair_rounds=0).build(p)
               for p in ("a house", "a hut")]
    metrics = write_run(tmp_path, arms={"demo": results}, config={"k": 1},
                        title="test run", max_dim=16, render=False)
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "summary.md").is_file()
    assert (tmp_path / "builds" / "demo" / "000_program.txt").read_text() == FIXED
    assert (tmp_path / "demo_16.npz").is_file()          # standard structure cache
    assert metrics["arms"]["demo"]["n_builds"] == 2
    assert json.loads((tmp_path / "metrics.json").read_text())["arms"]["demo"]


def test_saved_structures_reload_through_the_standard_loader(tmp_path):
    from blockgen.curation.houses import load_structures_from_cache
    results = [_agent([FIXED], repair_rounds=0).build("a house")]
    write_run(tmp_path, arms={"demo": results}, config={}, max_dim=16, render=False)
    structures, manifest = load_structures_from_cache(str(tmp_path / "demo_16.npz"))
    assert len(structures) == 1
    assert manifest["items"][0]["title"] == "a house"   # the prompt travels with it


def test_aggregate_reports_coherence():
    good = _agent([FIXED], repair_rounds=0).build("a house")
    split = _agent(["fill stone 0 0 0 1 1 1\nfill stone 9 9 9 10 10 10"],
                   repair_rounds=0).build("islands")
    agg = aggregate([good, split])
    assert agg["n_builds"] == 2 and agg["coherence_rate"] == 0.5
