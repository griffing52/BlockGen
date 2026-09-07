"""The agentic backend and the model-group registry, offline.

Everything here runs against the scripted provider (``agentic:mock``), so the whole
serving path — group resolution, vendor inference, program execution, per-command
steps, block-state mapping and the cost stats — is covered with no API key and no
network. The two things worth guarding hardest:

* **Group naming.** ``/model agentic gemini-3.5-flash`` has to reach
  ``gemini:gemini-3.5-flash``; a regression here is invisible until someone tries
  the command in game.
* **The step stream is the build.** The agentic backend replays a program, so the
  blocks a step reports must be exactly the voxels that command changed — including
  the ones it set back to *air*, since a cut doorway is part of the build.

Run:  python -m pytest deploy/inference/tests/ -q
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from blockgen_server.backends import (AgenticBackend, GenerateRequest, ModelSpec,
                                      as_batch, build_backend)
from blockgen_server.registry import Registry

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry(HERE / "models.json", REPO)


def _mock_backend(**extra) -> AgenticBackend:
    spec = ModelSpec(name="agentic:mock", kind="agentic")
    spec.extra = {"provider": "mock", "canvas": 32, "examples": 0,
                  "repair_rounds": 0, "critique_rounds": 0, **extra}
    backend = build_backend(spec, REPO)
    backend.load()
    return backend


# --- vendor inference ------------------------------------------------------
@pytest.mark.parametrize("model,expected", [
    ("gpt-5-mini", "openai:gpt-5-mini"),
    ("gpt-4.1-mini", "openai:gpt-4.1-mini"),
    ("gemini-3.5-flash", "gemini:gemini-3.5-flash"),
    ("gemini-2.5-pro", "gemini:gemini-2.5-pro"),
    ("claude-sonnet-4-5", "anthropic:claude-sonnet-4-5"),
    ("mock", "mock"),
    ("openai:some-new-model", "openai:some-new-model"),   # explicit wins
])
def test_vendor_is_inferred_from_the_model_name(model, expected):
    assert AgenticBackend.qualify(model) == expected


def test_an_unrecognizable_model_name_asks_for_a_vendor():
    with pytest.raises(ValueError, match="cannot tell which vendor"):
        AgenticBackend.qualify("llama-4-800b")


# --- registry groups -------------------------------------------------------
def test_agentic_is_a_group_with_members(registry):
    groups = registry.groups()
    assert "agentic" in groups and len(groups["agentic"]) > 4


def test_group_members_resolve_to_qualified_providers(registry):
    assert registry.resolve("agentic:gemini-3.5-flash").provider_spec == \
        "gemini:gemini-3.5-flash"
    assert registry.resolve("agentic:gpt-5").provider_spec == "openai:gpt-5"


def test_group_member_backends_are_cached(registry):
    a = registry.resolve("agentic:gpt-5-nano")
    b = registry.resolve("agentic:gpt-5-nano")
    assert a is b


def test_bare_group_name_uses_its_default_model(registry):
    assert registry.resolve("agentic").provider_spec == "openai:gpt-5-mini"


def test_member_inherits_the_group_loop_config(registry):
    """agentic_plus pins the full loop; a member must not silently drop it."""
    cfg = registry.resolve("agentic_plus:gpt-5-mini").agent_config()
    assert cfg.plan and cfg.critique_rounds == 1 and cfg.repair_rounds == 2


def test_top_level_listing_shows_a_group_as_one_row(registry):
    rows = {r["name"]: r for r in registry.describe()}
    assert "agentic" in rows
    assert rows["agentic"]["group_size"] == len(registry.groups()["agentic"])
    assert not any(name.startswith("agentic:") for name in rows)


def test_group_listing_expands_the_members(registry):
    rows = registry.describe("agentic")
    names = [r["name"] for r in rows]
    assert "agentic:gpt-5-mini" in names and "agentic:gemini-3.5-flash" in names
    assert all(r["supports_text"] for r in rows)


def test_describing_a_non_group_is_an_error(registry):
    with pytest.raises(KeyError):
        registry.describe("native_bpe")


def test_unknown_model_names_the_group_syntax(registry):
    with pytest.raises(KeyError, match="<group>"):
        registry.get("no_such_model")


def test_missing_api_key_is_reported_as_the_reason(monkeypatch, registry):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    row = next(r for r in registry.describe("agentic")
               if r["name"].endswith("claude-sonnet-4-5"))
    assert not row["available"]
    assert "ANTHROPIC_API_KEY" in row["unavailable_reason"]


# --- generation ------------------------------------------------------------
def test_stream_yields_one_labelled_step_per_command():
    backend = _mock_backend()
    batches = [as_batch(b) for b in backend.stream(GenerateRequest(prompt="a cottage"))]
    steps = [b for b in batches if b.step]
    assert len(steps) >= 5
    assert steps[0].step["index"] == 1
    assert steps[0].step["total"] == len(steps)
    assert steps[0].step["command"].startswith("fill cobblestone")
    # Steps arrive in program order — that is what makes the build legible in world.
    assert [b.step["index"] for b in steps] == list(range(1, len(steps) + 1))


def test_a_step_reports_exactly_the_blocks_it_changed():
    backend = _mock_backend()
    for batch in (as_batch(b) for b in backend.stream(GenerateRequest(prompt="x"))):
        if batch.step and batch.step["command"].startswith("fill cobblestone 0 0 0 9 0 7"):
            assert len(batch.blocks) == 10 * 1 * 8
            return
    raise AssertionError("the foundation step was never streamed")


def test_cleared_voxels_are_streamed_as_air():
    """A doorway cut after the wall must reach the client, or the wall stays solid."""
    backend = _mock_backend()
    air = [b for batch in (as_batch(x) for x in backend.stream(GenerateRequest(prompt="x")))
           for b in batch.blocks if b.state == "minecraft:air"]
    assert air, "no air placements streamed for the `clear` command"


def test_block_states_are_modern_and_oriented():
    backend = _mock_backend()
    states = {b.state for batch in (as_batch(x) for x in
                                    backend.stream(GenerateRequest(prompt="x")))
              for b in batch.blocks}
    assert "minecraft:cobblestone" in states
    # The gable orients its two slopes oppositely; both facings must survive the
    # legacy round trip into modern block states.
    stairs = {s for s in states if "stairs" in s}
    assert any("facing=north" in s for s in stairs)
    assert any("facing=south" in s for s in stairs)


def test_final_batch_carries_cost_stats():
    backend = _mock_backend()
    batches = [as_batch(b) for b in backend.stream(GenerateRequest(prompt="x"))]
    stats = [b.stats for b in batches if b.stats][0]
    assert stats["provider"] == "mock"
    assert stats["commands"] > 0 and stats["failed_commands"] == 0
    assert "cost_usd" in stats and "cost_known" in stats
    assert stats["rounds"] == ["generate"]


def test_a_bare_gen_still_builds_something():
    """No prompt is a demo, not an error: the backend picks one and reports it."""
    backend = _mock_backend()
    batches = [as_batch(b) for b in backend.stream(GenerateRequest(seed=3))]
    stats = [b.stats for b in batches if b.stats][0]
    assert stats["prompt"]
    assert sum(len(b.blocks) for b in batches) > 0


def test_the_agentic_entry_declares_no_checkpoint(registry):
    """It has no weights; the registry's missing-file check must not trip on it."""
    b = registry.backends["agentic"]
    assert b.spec.checkpoint == ""
    assert registry.missing_files(b) == []


def test_models_json_stays_loadable():
    blob = json.loads((HERE / "models.json").read_text())
    assert blob["default"] in blob["models"]
    for name, entry in blob["models"].items():
        assert "kind" in entry and "description" in entry
