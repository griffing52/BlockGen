"""Legacy parity on the REAL path (spec section 7.4), promoted from WP2's draft.

test_classic_parity.py checks ``classic`` against BuildAgent with fake contexts.
This file runs the path a recipe actually takes: ``from_agent_config`` ->
``build_kit`` (the registered ``dsl`` action space, ``palette`` / ``examples`` /
``ontology`` contexts, ``nonempty_legacy`` accept, ``null`` renderer) ->
``run_episode``. Only the provider is swapped for the shared reply script, and
BuildAgent renders through the same null renderer so critique runs headless.

Asserted equal for every experiments_agentic arm x critique mode x keep_history x
3 prompts x 4 scripts: (1) the fingerprint sequence of every provider call (=
CachingProvider keys), (2) final program_text, (3) block ids/data, (4) stage names.

The ``legacy`` method (BuildAgent wrapped) and the virtual ``legacy-<arm>``
recipes are checked too: same outputs, every call metered, traces written.
"""

from __future__ import annotations

import dataclasses
import warnings

import pytest

from blockgen.agentic import core
from blockgen.agentic.agent import AgentConfig, BuildAgent
from blockgen.agentic.core import VIEWS
from blockgen.agentic.episode import run_episode
from blockgen.agentic.providers import ScriptedProvider, encode_png
from blockgen.agentic.recipe import from_agent_config, load_recipe
from blockgen.experiments_agentic import ARMS

from .test_classic_parity import (HAS_ONTOLOGY, PROMPTS, SCRIPTS, assert_parity,
                                  fingerprints)

ARM_NAMES = [a for a in ARMS if HAS_ONTOLOGY or not a.startswith("ont_")]


@pytest.fixture
def kitmod():
    from blockgen.agentic import kit
    kit.clear_cache()
    yield kit
    kit.clear_cache()


def _kit(kitmod, spec, replies):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kit = kitmod.build_kit(dataclasses.replace(spec, renderer="null", cache="off"))
    llm = ScriptedProvider(replies=list(replies))
    kit.llms = {r: llm for r in core.ROLES}
    return kit, llm


def _null_render(kit):
    renderer = kit.renderer

    def render(self, structure):
        if int(structure.occupied_mask.sum()) == 0:
            return []
        return [encode_png(i) for i in renderer.render(
            structure, [VIEWS["legacy_a"], VIEWS["legacy_b"]])]
    return render


def _script(arm, script):
    return (["PLAN: footprint 10x8, cobblestone base, oak walls."] if ARMS[arm].get("plan")
            else []) + list(script)


@pytest.mark.parametrize("keep_history", [True, False])
@pytest.mark.parametrize("mode", ["rewrite", "patch"])
@pytest.mark.parametrize("arm", ARM_NAMES)
def test_native_classic_matches_buildagent_through_build_kit(arm, mode, keep_history,
                                                             kitmod, monkeypatch):
    cfg = AgentConfig(provider="mock", canvas_size=(32, 32, 32), critique_mode=mode,
                      keep_history=keep_history, verbose=False, **ARMS[arm])
    spec = from_agent_config(cfg, name="parity", native=True)
    for prompt in PROMPTS:
        for name, script in SCRIPTS.items():
            script = _script(arm, script)
            kit, native_llm = _kit(kitmod, spec, script)
            monkeypatch.setattr(BuildAgent, "_render", _null_render(kit))
            legacy_llm = ScriptedProvider(replies=list(script))
            want = BuildAgent(legacy_llm, cfg).build(prompt)
            got = run_episode(kit, prompt)
            try:
                assert_parity(want, legacy_llm, got, native_llm)
            except AssertionError as exc:  # pragma: no cover - name the failing combo
                raise AssertionError(f"arm={arm} mode={mode} keep_history={keep_history} "
                                     f"script={name} prompt={prompt!r}: {exc}") from exc


@pytest.mark.parametrize("arm", ARM_NAMES)
def test_legacy_method_is_buildagent(arm, kitmod, monkeypatch, tmp_path):
    """method `legacy` (the dsl-classic / legacy-<arm> path) gives BuildAgent's exact
    outputs, meters every call, and leaves a trace."""
    from blockgen.agentic.trace import Trace
    cfg = AgentConfig(provider="mock", canvas_size=(32, 32, 32), verbose=False, **ARMS[arm])
    spec = from_agent_config(cfg, name=f"legacy-{arm}")
    script = _script(arm, SCRIPTS["fix"])
    kit, native_llm = _kit(kitmod, spec, script)
    monkeypatch.setattr(BuildAgent, "_render", _null_render(kit))
    legacy_llm = ScriptedProvider(replies=list(script))
    want = BuildAgent(legacy_llm, cfg).build(PROMPTS[0])
    got = run_episode(kit, PROMPTS[0], trace=Trace(tmp_path / "t"))
    assert_parity(want, legacy_llm, got, native_llm)
    assert len([c for c in got.charges if c.kind == "llm"]) == len(legacy_llm.calls)
    assert (tmp_path / "t" / "result.json").exists()


def test_virtual_legacy_recipes_match_arms():
    """legacy-<arm> recipes rebuild exactly the ARMS AgentConfig (incl. ontology knobs)."""
    for arm in ARMS:
        spec = load_recipe(f"legacy-{arm}")
        assert spec.method == "legacy" and spec.params["arm"] == arm
        for k, v in ARMS[arm].items():
            assert spec.params["config"][k] == v, (arm, k)


def test_dsl_classic_recipe_is_the_full_arm():
    spec = load_recipe("dsl-classic")
    assert spec.method == "legacy" and spec.params == {"arm": "full"}
    # gen and llm match experiments_agentic's defaults, so cache keys line up.
    assert spec.gen == {"max_tokens": 16000, "temperature": None, "reasoning_effort": "medium"}
    assert spec.llm == "openai:gpt-5-mini"


def test_clipboard_quirk_stays_in_legacy_only(kitmod, monkeypatch):
    """Patch-mode paste of an earlier copy: BuildAgent (legacy Canvas.clone) loses the
    clipboard, native classic on MinecraftWorld keeps it. Requests stay identical."""
    cfg = AgentConfig(provider="mock", canvas_size=(32, 32, 32), critique_mode="patch",
                      verbose=False, **ARMS["critique"])
    script = ["fill stone 0 0 0 3 0 3\ncopy 0 0 0 3 0 3\n", "paste 10 0 10\n"]
    kit, native_llm = _kit(kitmod, from_agent_config(cfg, native=True), script)
    monkeypatch.setattr(BuildAgent, "_render", _null_render(kit))
    legacy_llm = ScriptedProvider(replies=list(script))
    want = BuildAgent(legacy_llm, cfg).build(PROMPTS[0])
    got = run_episode(kit, PROMPTS[0])
    assert fingerprints(native_llm.calls) == fingerprints(legacy_llm.calls)
    assert want.blocks == 16 and got.blocks == 32
