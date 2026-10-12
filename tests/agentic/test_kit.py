"""build_kit: registry resolution, LLM roles, scorers, dreams, mock substitution."""

from __future__ import annotations

import warnings

import pytest

from blockgen.agentic import core, kit as kitmod, providers, recipe
from blockgen.agentic.core import DreamSpec, Kit, Ledger, MethodSpec
from blockgen.agentic.kit import build_kit

from .fakes import (FAKE_DREAMS, FakeActions, FakeImageGen, FakeRenderer, ScriptedLLM,
                    fake_spec, make_attempt, make_fake_kit)


@pytest.fixture(autouse=True)
def _fresh_cache():
    kitmod.clear_cache()
    yield
    kitmod.clear_cache()


def spec(**kw) -> MethodSpec:
    return fake_spec(**{"llm": "mock", **kw})


def test_build_kit_resolves_every_axis():
    k = build_kit(spec(image="fake_image?usd=0.02", params={"n": 2}))
    assert isinstance(k, Kit)
    assert isinstance(k.actions, FakeActions) and isinstance(k.renderer, FakeRenderer)
    assert isinstance(getattr(k.image, "inner", k.image), FakeImageGen) and k.image.usd == 0.02
    assert [n for n, _ in k.contexts] == ["fake_context"]
    fn, w = k.scorers["fake_scorer"]
    assert w == 1.0 and fn(None, make_attempt()).value > 0
    assert k.accept(None, make_attempt()) == (True, "ok")
    assert k.method.keywords == {"n": 2} and k.method.spec_name == "fake_method"
    assert k.ledger.name == "fake" and k.dreams == [] and k.mock is False
    assert k.world_factory((8, 8, 8)).size == (8, 8, 8)


def test_roles_share_one_provider_per_spec_string():
    k = build_kit(spec(llm={"builder": "mock", "judge": "mock:"}))
    assert set(k.llms) == set(core.ROLES)
    assert k.llms["critic"] is k.llms["builder"] is k.llms["planner"]
    assert k.llms["judge"] is not k.llms["builder"]
    assert k.llm("dream_critic") is k.llms["builder"]


def test_llm_gets_gen_params_exactly_like_legacy(monkeypatch):
    """Same construction as experiments_agentic.run_arm => same cache keys:
    cache_mode / spend_gate are NAMED arguments, never part of ``**params``."""
    seen = []

    def fake_get_provider(s, *, cache=True, cache_dir=None, cache_mode=None, spend_gate=False,
                          **params):
        seen.append((s, cache, cache_mode, spend_gate, params))
        return ScriptedLLM()

    monkeypatch.setattr(providers, "get_provider", fake_get_provider)
    build_kit(spec(llm="openai:gpt-5-mini", gen={"max_tokens": 123}), cache_mode="read_write")
    assert seen == [("openai:gpt-5-mini", True, "read_write", True,
                     {"max_tokens": 123, "temperature": None, "reasoning_effort": "medium"})]
    build_kit(spec(llm="openai:gpt-5-mini"), cache_mode="read_only", spend_gate=True)
    assert seen[-1][2:4] == ("read_only", True)
    build_kit(spec(llm="openai:gpt-5-mini"), cache_mode="off")
    assert seen[-1][1] is False


def test_unknown_method_param_fails_at_build_time():
    with pytest.raises(TypeError, match=r"method 'fake_method'.*\['nn'\].*did you mean 'n'"):
        build_kit(spec(params={"nn": 3}))


def test_unknown_component_names_suggest():
    with pytest.raises(KeyError, match="did you mean 'fake_dsl'"):
        build_kit(spec(actions="fake_dls"))


def test_scorer_weights_params_and_gates():
    k = build_kit(spec(score={"fake_scorer?scale=10": 2.0,
                              "big": {"type": "fake_scorer", "weight": 0.5, "scale": 1000}},
                       gates={"big": 0.1}, final_score={"fake_scorer": 1.0}))
    fn, w = k.scorers["fake_scorer"]
    assert w == 2.0 and fn.keywords == {"scale": 10}
    big, wb = k.scorers["big"]
    assert wb == 0.5 and big.keywords == {"scale": 1000}
    assert set(k.final_scorers) == {"fake_scorer"}
    with pytest.raises(ValueError, match="gate"):
        build_kit(spec(gates={"nope": 0.5}))
    with pytest.raises(ValueError, match="listed twice"):
        build_kit(spec(score={"fake_scorer": 1.0, "fake_scorer?scale=3": 1.0}))


def test_dreams_presets_inline_and_validation(monkeypatch):
    monkeypatch.setattr(kitmod.dreams_mod, "DREAMS", dict(FAKE_DREAMS))
    ds = kitmod.resolve_dreams(["iso", {"key": "back", "weight": 0.25},
                                {"key": "side", "template": "{desc}", "view": "side",
                                 "condition_on": ["iso"], "view_search": "iso_left"}])
    assert [d.key for d in ds] == ["iso", "back", "side"]
    assert ds[1].weight == 0.25 and ds[1].condition_on == ("iso",)    # preset overridden
    assert ds[2].condition_on == ("iso",) and ds[2].view_search == ("iso_left",)
    assert all(isinstance(d, DreamSpec) for d in ds)
    with pytest.raises(KeyError, match="unknown dream preset"):
        kitmod.resolve_dreams(["nope"])
    with pytest.raises(ValueError, match="duplicate"):
        kitmod.resolve_dreams(["iso", "iso"])
    with pytest.raises(ValueError, match="unknown key"):
        kitmod.resolve_dreams(["back"])                              # conditions on missing iso
    with pytest.raises(KeyError, match="unknown view"):
        kitmod.resolve_dreams([{"key": "x", "template": "t", "view": "upside"}])
    with pytest.raises(TypeError, match="unknown field"):
        kitmod.resolve_dreams([{"key": "x", "template": "t", "wieght": 1}])
    k = build_kit(spec(image="fake_image", dreams=["iso", "mood"]))
    assert [d.key for d in k.dreams] == ["iso", "mood"]


def test_ledger_parenting_and_arm_cap():
    battery = Ledger(name="battery")
    k = build_kit(spec(budget={"usd_per_arm": 1.5, "usd_per_build": 0.1}), ledger=battery)
    assert k.ledger.parent is battery and k.ledger.budget.usd == 1.5
    b = kitmod.episode_budget(k.spec)
    assert b.usd == 0.1 and b.max_images is None


def test_heavy_components_cached_across_kits():
    a = build_kit(spec(image="fake_image"))
    b = build_kit(spec(image="fake_image"))
    assert a.renderer is b.renderer and a.image is b.image
    assert a.actions is not b.actions                     # cheap ones are per kit
    kitmod.clear_cache()
    assert build_kit(spec()).renderer is not a.renderer


def test_renderer_none():
    assert build_kit(spec(renderer="none")).renderer is None


def test_bad_cache_mode():
    with pytest.raises(ValueError, match="cache mode"):
        build_kit(spec(), cache_mode="sometimes")


# --- mock substitution ----------------------------------------------------------
def test_mock_spec_table(monkeypatch):
    monkeypatch.setattr(kitmod, "textured_available", lambda: False)
    s = MethodSpec("d", llm={"builder": "openai:gpt-5", "judge": "claude-code:sonnet"},
                   image="openai:gpt-image-1", renderer="textured",
                   score={"embed?backbone=siglipB": 1.0, "silhouette": 0.5,
                          "clip_text": {"weight": 0.2, "backbone": "clipB16"}},
                   final_score={"clip_text": 1.0}, gates={"embed": 0.1})
    m = kitmod.mock_spec(s)
    assert m.llm == "mock:auto" and m.image == "mock:render" and m.renderer == "flat"
    assert m.score == {"embed": {"type": "pixel", "weight": 1.0}, "silhouette": 0.5,
                       "clip_text": {"type": "pixel", "weight": 0.2}}
    assert m.final_score == {"clip_text": {"type": "pixel", "weight": 1.0}}
    assert m.tags["mock"] is True and m.hash() != s.hash()
    assert kitmod.mock_spec(MethodSpec("x")).image is None


def test_mock_keeps_the_recipe_renderer_when_it_can_draw(monkeypatch):
    monkeypatch.setattr(kitmod, "textured_available", lambda: True)
    assert kitmod.mock_spec(MethodSpec("d", renderer="textured")).renderer == "textured"
    assert kitmod.mock_spec(MethodSpec("d", renderer="flat?px=64")).renderer == "flat?px=64"
    monkeypatch.setattr(kitmod, "textured_available", lambda: False)
    with pytest.warns(UserWarning, match="falls back to the flat renderer"):
        assert kitmod.mock_spec(MethodSpec("d", renderer="textured?px=256")).renderer == "flat"


def test_build_kit_mock_mechanics(monkeypatch):
    """With the real twins absent (pre WP4/5), point the table at fakes and check
    the swap, the provenance flag and the mock:auto fallback."""
    monkeypatch.setitem(kitmod.MOCK_SUBS, "image", "fake_image")
    monkeypatch.setitem(kitmod.MOCK_SUBS, "renderer", "fake_renderer")
    monkeypatch.setattr(kitmod, "textured_available", lambda: False)   # force the fallback
    monkeypatch.setitem(kitmod.MOCK_SUBS, "scorer", {"real_scorer": "fake_scorer"})
    s = spec(llm="openai:gpt-5", image="openai:gpt-image-1", renderer="textured",
             score={"real_scorer?backbone=x": 1.0})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = build_kit(s, mock=True)
    assert k.mock is True and k.spec.tags["mock"] is True
    assert isinstance(getattr(k.image, "inner", k.image), FakeImageGen) and isinstance(k.renderer, FakeRenderer)
    assert "real_scorer" in k.scorers
    reply = k.llm("builder").complete([providers.text_message("user", "hi")]).text
    assert reply == FakeActions().mock_reply()
    assert k.spec.provenance()["mock"] is True


def test_make_fake_kit_overrides():
    k = make_fake_kit(replies=["a", "b"], dreams=["iso"], image=None,
                      spec_kw={"params": {"n": 2}})
    assert k.image is None and k.dreams[0].key == "iso"
    assert k.method.keywords == {"n": 2}
    assert k.llm("judge").complete([]).text == "a"
    with pytest.raises(TypeError, match="unknown Kit field"):
        make_fake_kit(imagegen=None)


# --- shipped recipes (WP7) -------------------------------------------------------
_SHIPPED = recipe.list_recipes(virtual=False)


@pytest.mark.parametrize("name", _SHIPPED or ["<none>"])
def test_every_shipped_recipe_builds_in_mock_mode(name):
    if not _SHIPPED:
        pytest.skip("configs/agentic/recipes is empty (WP7 not landed)")
    core.load_builtins()
    if core._BUILTINS_SKIPPED:
        pytest.skip(f"built-in modules missing: {sorted(core._BUILTINS_SKIPPED)}")
    k = build_kit(recipe.load_recipe(name), mock=True, cache_mode="off")
    assert k.mock and k.spec.llm == "mock:auto"
