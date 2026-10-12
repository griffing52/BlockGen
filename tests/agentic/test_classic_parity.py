"""Native ``classic`` vs BuildAgent: identical provider requests, programs and blocks.

This is the guard that lets ``classic`` stand in for BuildAgent when a recipe
wants an axis legacy cannot change. For every experiments_agentic arm, both
critique modes and both keep_history settings, BuildAgent and
``run_episode(classic)`` get identical scripted providers; we assert

    (1) the fingerprint sequence of every provider call (= CachingProvider keys)
    (2) the final program_text
    (3) the final structure's block ids and data
    (4) the stage names

The renderer is the fake for both: BuildAgent's ``_render`` is monkeypatched to
the same images classic's renderer produces, so critique runs headless.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import pytest

from blockgen.agentic.agent import AgentConfig, BuildAgent
from blockgen.agentic import core
from blockgen.agentic.core import VIEWS, parse_spec
from blockgen.agentic.episode import run_episode
from blockgen.agentic.providers import ScriptedProvider, encode_png
from blockgen.agentic.recipe import from_agent_config
from blockgen.agentic.recipes.classic import classic, legacy_prompt_mode
from blockgen.experiments_agentic import ARMS

from .fakes import FakeActions, FakeRenderer, fake_accept, fake_context, make_fake_kit, solid_png

_REPO = Path(__file__).resolve().parents[2]
HAS_ONTOLOGY = (_REPO / "data" / "ontology" / "minecraft_houses_32.json").is_file()

BROKEN = ("fill cobblestone 0 0 0 9 0 7\nwalls oak_planks 0 1 0 9 4 7\n"
          "fill notablock 0 5 0 9 5 7\n")
FIXED = ("fill cobblestone 0 0 0 9 0 7\nwalls oak_planks 0 1 0 9 4 7\n"
         "fill oak_planks 0 5 0 9 5 7\nclear 4 1 0 5 3 0\n")
CRIT = "fill glass 2 2 0 2 3 0\nfill glass 7 2 0 7 3 0\nset torch 3 3 1\n"
EMPTY = "clear 0 0 0 47 47 47\n"

SCRIPTS = {
    "fix": [BROKEN, FIXED, CRIT, CRIT],
    "empty_repair": [BROKEN, EMPTY, CRIT, CRIT],      # repair rolled back, critique still runs
    "empty_critique": [BROKEN, FIXED, EMPTY, CRIT],   # critique rolled back
    "clean": [FIXED, CRIT, CRIT],                     # nothing to repair
}
PROMPTS = ["a small oak cottage with a gabled roof",
           "a round stone watchtower",
           "a modern glass house"]


class DslActions(FakeActions):
    """Identity lowering under the name 'dsl': exactly BuildAgent's executor input.
    Stand-in used only while the real 'dsl' action space is not registered."""

    name = "dsl"


def dsl_actions():
    try:
        return core.make("actions", "dsl")
    except KeyError:
        return DslActions()


def fingerprints(calls):
    return [[[m.role] + [p.fingerprint() for p in m.parts] for m in call] for call in calls]


def _fake_render(self, structure):
    if int(structure.occupied_mask.sum()) == 0:
        return []
    imgs = FakeRenderer().render(structure, [VIEWS["legacy_a"], VIEWS["legacy_b"]])
    return [encode_png(i) for i in imgs]


def run_both(cfg: AgentConfig, script, prompt, monkeypatch, *, seed=None, images=()):
    monkeypatch.setattr(BuildAgent, "_render", _fake_render)
    legacy_llm = ScriptedProvider(replies=list(script))
    want = BuildAgent(legacy_llm, cfg).build(prompt, images=images, seed=seed)

    spec = from_agent_config(cfg, name="parity", native=True)
    spec = dataclasses.replace(spec, renderer="fake_renderer", score={})
    native_llm = ScriptedProvider(replies=list(script))
    kit = make_fake_kit(llm=native_llm, spec=spec, method=classic, actions=dsl_actions(),
                        contexts=[(parse_spec(c)[0], fake_context) for c in spec.contexts],
                        scorers={}, accept=fake_accept, renderer=FakeRenderer())
    got = run_episode(kit, prompt, seed=seed, images=images)
    return want, legacy_llm, got, native_llm


def assert_parity(want, legacy_llm, got, native_llm):
    assert got.truncated == "", got.error
    assert fingerprints(native_llm.calls) == fingerprints(legacy_llm.calls)
    assert got.program_text == want.program_text
    assert (got.structure.block_ids == want.structure.block_ids).all()
    assert (got.structure.block_data == want.structure.block_data).all()
    assert got.structure.block_ids.shape == want.structure.block_ids.shape
    assert [a.stage for a in got.attempts] == [r.stage for r in want.rounds]
    assert got.plan_text == want.plan_text
    assert [a.blocks for a in got.attempts] == [r.blocks for r in want.rounds]


ARM_NAMES = [a for a in ARMS if HAS_ONTOLOGY or not a.startswith("ont_")]


@pytest.mark.parametrize("keep_history", [True, False])
@pytest.mark.parametrize("mode", ["rewrite", "patch"])
@pytest.mark.parametrize("arm", ARM_NAMES)
def test_classic_matches_buildagent_on_every_arm(arm, mode, keep_history, monkeypatch):
    for prompt in PROMPTS:
        for name, script in SCRIPTS.items():
            if ARMS[arm].get("plan"):
                script = ["PLAN: footprint 10x8, cobblestone base, oak walls."] + script
            cfg = AgentConfig(provider="mock", canvas_size=(32, 32, 32), critique_mode=mode,
                              keep_history=keep_history, verbose=False, **ARMS[arm])
            want, l_llm, got, n_llm = run_both(cfg, script, prompt, monkeypatch)
            try:
                assert_parity(want, l_llm, got, n_llm)
            except AssertionError as exc:  # pragma: no cover - name the failing combo
                raise AssertionError(f"arm={arm} mode={mode} keep_history={keep_history} "
                                     f"script={name} prompt={prompt!r}: {exc}") from exc


@pytest.mark.parametrize("arm", ["full", "critique"])
def test_parity_with_seed_images_and_scale_hint(arm, monkeypatch):
    cfg = AgentConfig(provider="mock", canvas_size=(24, 20, 24), target_blocks=800,
                      verbose=False, **ARMS[arm])
    script = (["PLAN"] if ARMS[arm].get("plan") else []) + SCRIPTS["fix"]
    ref = solid_png((10, 200, 30))
    assert_parity(*run_both(cfg, script, PROMPTS[0], monkeypatch, seed=3, images=(ref,)))


def test_classic_is_in_legacy_prompt_mode_only_for_dsl_and_legacy_contexts():
    spec = from_agent_config(AgentConfig(n_examples=1), native=True)
    kit = make_fake_kit(spec=spec, actions=DslActions(),
                        contexts=[("palette", fake_context), ("examples", fake_context)])

    class Ep:
        pass

    ep = Ep()
    ep.kit, ep.spec = kit, spec
    assert legacy_prompt_mode(ep)
    kit.contexts = [("palette", fake_context), ("texture_sheet", fake_context)]
    assert not legacy_prompt_mode(ep)
    # No palette context: the legacy system prompt would add one anyway.
    kit.contexts = [("examples", fake_context)]
    assert not legacy_prompt_mode(ep)
    # A verb subset: the legacy reference would list every command.
    kit.contexts = [("palette", fake_context)]
    subset = DslActions()
    subset.allowed = ("fill", "set")
    kit.actions = subset
    assert not legacy_prompt_mode(ep)
    kit.actions = DslActions()
    assert legacy_prompt_mode(ep)
    # A corpus example pool is not the legacy builtin selection.
    ep.spec = dataclasses.replace(spec, contexts=["palette", {"examples": {"k": 1, "pool": "corpus"}}])
    kit.contexts = [("palette", fake_context), ("examples", fake_context)]
    assert not legacy_prompt_mode(ep)
    ep.spec = spec
    kit.contexts = [("palette", fake_context)]
    kit.actions = FakeActions()
    assert not legacy_prompt_mode(ep)


def test_clipboard_difference_is_the_documented_one(monkeypatch):
    """Patch-mode `paste` of a region copied in an earlier round: BuildAgent's
    Canvas.clone drops the clipboard (paste fails), MinecraftWorld keeps it."""
    cfg = AgentConfig(provider="mock", canvas_size=(32, 32, 32), critique_mode="patch",
                      verbose=False, **ARMS["critique"])
    script = ["fill stone 0 0 0 3 0 3\ncopy 0 0 0 3 0 3\n", "paste 10 0 10\n"]
    want, l_llm, got, n_llm = run_both(cfg, script, PROMPTS[0], monkeypatch)
    assert fingerprints(n_llm.calls) == fingerprints(l_llm.calls)
    assert want.blocks == 16 and got.blocks == 32


def _registered(kind, name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return name in core.names(kind)


@pytest.mark.skipif(not all(_registered(k, n) for k, n in [
    ("actions", "dsl"), ("renderer", "null"), ("accept", "nonempty_legacy"),
    ("context", "palette"), ("context", "examples")]), reason="WP3/WP5/WP6 not landed")
@pytest.mark.parametrize("arm,mode", [("full", "rewrite"), ("full", "patch"),
                                      ("critique", "patch"), ("oneshot", "rewrite")])
def test_parity_through_build_kit_with_registered_components(arm, mode, monkeypatch):
    """The real path: from_agent_config(native=True) -> build_kit -> run_episode,
    with the registered dsl / palette / examples / nonempty_legacy / null renderer.
    Only the provider is swapped for the script (and BuildAgent renders with the
    same null renderer)."""
    from blockgen.agentic import kit as kitmod

    cfg = AgentConfig(provider="mock", canvas_size=(32, 32, 32), critique_mode=mode,
                      verbose=False, **ARMS[arm])
    script = (["PLAN"] if ARMS[arm].get("plan") else []) + SCRIPTS["fix"]
    spec = dataclasses.replace(from_agent_config(cfg, name="parity", native=True),
                               renderer="null", cache="off")
    kitmod.clear_cache()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kit = kitmod.build_kit(spec)
        native_llm = ScriptedProvider(replies=list(script))
        kit.llms = {r: native_llm for r in core.ROLES}
        renderer = kit.renderer

        def null_render(self, structure):
            if int(structure.occupied_mask.sum()) == 0:
                return []
            return [encode_png(i) for i in renderer.render(
                structure, [VIEWS["legacy_a"], VIEWS["legacy_b"]])]

        monkeypatch.setattr(BuildAgent, "_render", null_render)
        legacy_llm = ScriptedProvider(replies=list(script))
        want = BuildAgent(legacy_llm, cfg).build(PROMPTS[1])
        got = run_episode(kit, PROMPTS[1])
        assert_parity(want, legacy_llm, got, native_llm)
        if ARMS[arm]["critique_rounds"]:
            assert any(p.kind == "image" for c in native_llm.calls for m in c for p in m.parts)
    finally:
        kitmod.clear_cache()
