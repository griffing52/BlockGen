"""Recipe/battery loading: extends, overrides, validation, legacy virtual recipes, sweeps."""

from __future__ import annotations

import textwrap

import pytest

from blockgen.agentic import recipe
from blockgen.agentic.agent import AgentConfig
from blockgen.agentic.core import MethodSpec
from blockgen.experiments_agentic import ARMS


def write(d, name, text):
    (d / f"{name}.yaml").write_text(textwrap.dedent(text))


@pytest.fixture
def rdir(tmp_path, monkeypatch):
    r = tmp_path / "recipes"
    b = tmp_path / "batteries"
    r.mkdir()
    b.mkdir()
    monkeypatch.setattr(recipe, "RECIPE_DIRS", [r])
    monkeypatch.setattr(recipe, "BATTERY_DIRS", [b])
    write(r, "_base", """
        llm: openai:gpt-5-mini
        actions: dsl
        contexts: [palette]
        canvas: [32, 32, 32]
        gen: {max_tokens: 16000, temperature: null, reasoning_effort: medium}
        budget: {usd_per_build: 0.25}
    """)
    write(r, "dream", """
        extends: _base
        description: one dream
        method: dream
        llm: {builder: openai:gpt-5-mini, judge: openai:gpt-5}
        image: openai:gpt-image-1-mini?quality=low
        dreams: [iso]
        contexts: [palette, {examples: {k: 1}}]
        score: {embed: 1.0, validity: 0.3}
        gates: {validity: 0.5}
        select: best
        params: {max_iters: 3, stop_at: 0.85}
        budget: {usd_per_build: 0.30, max_images: 4}
    """)
    write(r, "dream-refine", """
        extends: dream
        params: {refine: 2}
        budget: {max_images: 6}
    """)
    return r, b


def test_extends_deep_merges(rdir):
    s = recipe.load_recipe("dream-refine")
    assert isinstance(s, MethodSpec) and s.name == "dream-refine"
    assert s.params == {"max_iters": 3, "stop_at": 0.85, "refine": 2}     # mappings merge
    assert s.budget == {"usd_per_build": 0.30, "max_images": 6}
    assert s.contexts == ["palette", {"examples": {"k": 1}}]              # lists replace
    assert s.canvas == (32, 32, 32) and isinstance(s.canvas, tuple)
    assert s.description == "one dream" and s.method == "dream"


def test_load_by_path_and_name_is_stem(rdir):
    r, _ = rdir
    s = recipe.load_recipe(r / "dream.yaml")
    assert s.name == "dream" and s.select == "best"
    assert recipe.load_recipe(str(r / "dream.yaml")).hash() == s.hash()


def test_dotted_overrides(rdir):
    s = recipe.load_recipe("dream", {"params.max_iters": 5, "llm.critic": "claude-code:sonnet",
                                     "canvas": [48, 48, 48]})
    assert s.params["max_iters"] == 5 and s.params["stop_at"] == 0.85
    assert s.roles()["critic"] == "claude-code:sonnet" and s.roles()["judge"] == "openai:gpt-5"
    assert s.canvas == (48, 48, 48)
    assert s.hash() != recipe.load_recipe("dream").hash()


def test_llm_role_override_on_string_llm_keeps_other_roles(rdir):
    s = recipe.load_recipe("_base", {"llm.judge": "claude-code:sonnet"})
    roles = s.roles()
    assert roles["judge"] == "claude-code:sonnet" and roles["builder"] == roles["critic"] == "openai:gpt-5-mini"


def test_unknown_fields_raise(rdir):
    r, _ = rdir
    write(r, "typo", "extends: _base\nparms: {max_iters: 2}\n")
    with pytest.raises(TypeError, match=r"unknown field\(s\) \['parms'\]"):
        recipe.load_recipe("typo")
    with pytest.raises(TypeError, match="unknown recipe field 'parms'"):
        recipe.load_recipe("dream", {"parms.x": 1})
    with pytest.raises(TypeError, match="not a mapping"):
        recipe.load_recipe("dream", {"params.max_iters.deeper": 1})
    write(r, "badrole", "llm: {buidler: openai:x}\n")
    with pytest.raises(ValueError, match="unknown llm role"):
        recipe.load_recipe("badrole")
    write(r, "badbudget", "budget: {usd: 1}\n")
    with pytest.raises(ValueError, match="unknown budget key"):
        recipe.load_recipe("badbudget")
    write(r, "badsel", "select: max\n")
    with pytest.raises(ValueError, match="select"):
        recipe.load_recipe("badsel")
    write(r, "badcanvas", "canvas: [32, 32]\n")
    with pytest.raises(ValueError, match="canvas"):
        recipe.load_recipe("badcanvas")
    with pytest.raises(FileNotFoundError, match="known"):
        recipe.load_recipe("no-such-recipe")


def test_extends_cycle_detected(rdir):
    r, _ = rdir
    write(r, "a", "extends: b\n")
    write(r, "b", "extends: [_base, a]\n")
    with pytest.raises(ValueError, match="cycle: a -> b -> a"):
        recipe.load_recipe("a")


def test_name_never_inherited(rdir):
    r, _ = rdir
    write(r, "parent", "name: parent-name\nmethod: classic\n")
    write(r, "child", "extends: parent\n")
    assert recipe.load_recipe("child").name == "child"
    assert recipe.load_recipe("child", {"name": "renamed"}).name == "renamed"


def test_canvas_int_and_gen_defaults(rdir):
    r, _ = rdir
    write(r, "cube", "canvas: 16\ngen: {max_tokens: 100}\n")
    s = recipe.load_recipe("cube")
    assert s.canvas == (16, 16, 16)
    assert s.gen == {"max_tokens": 100, "temperature": None, "reasoning_effort": "medium"}


def test_list_recipes_excludes_private_and_adds_legacy(rdir):
    names = recipe.list_recipes()
    assert "dream" in names and "dream-refine" in names and "_base" not in names
    assert {f"legacy-{a}" for a in ARMS} <= set(names)
    assert recipe.list_recipes(virtual=False) == ["dream", "dream-refine"]


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_legacy_virtual_recipes(rdir, arm):
    s = recipe.load_recipe(f"legacy-{arm}")
    cfg = AgentConfig(**{**recipe.LEGACY_CLI_DEFAULTS, **ARMS[arm]})
    assert s.name == f"legacy-{arm}" and s.method == "legacy"
    assert s.canvas == (48, 48, 48)          # experiments_agentic's --canvas default
    assert s.params["arm"] == arm
    rebuilt = AgentConfig(**{**s.params["config"], "provider": s.llm,
                             "canvas_size": s.canvas, **s.gen})
    assert rebuilt.to_dict() == cfg.to_dict()
    assert s.accept == "nonempty_legacy" and s.select == "last"


def test_legacy_virtual_overrides_and_file_wins(rdir):
    r, _ = rdir
    s = recipe.load_recipe("legacy-full", {"canvas": [32, 32, 32], "llm": "mock"})
    assert s.canvas == (32, 32, 32) and s.llm == "mock" and s.params["config"]["plan"] is True
    write(r, "legacy-full", "method: classic\n")
    assert recipe.load_recipe("legacy-full").method == "classic"


def test_from_agent_config_legacy_and_native():
    cfg = AgentConfig(provider="anthropic:claude-haiku-4-5", canvas_size=(24, 24, 24), plan=True,
                      n_examples=1, repair_rounds=2, critique_rounds=1, critique_mode="patch",
                      ontology="mined", ontology_seed=3, keep_history=False, max_tokens=1000,
                      cache=False)
    s = recipe.from_agent_config(cfg, name="x")
    assert s.method == "legacy" and s.llm == "anthropic:claude-haiku-4-5"
    assert s.canvas == (24, 24, 24) and s.gen["max_tokens"] == 1000 and s.cache == "off"
    assert not set(s.params["config"]) & {"provider", "canvas_size", "max_tokens", "cache"}
    assert s.params["config"]["critique_mode"] == "patch" and s.keep_history is False
    n = recipe.from_agent_config(cfg, name="x", native=True)
    assert n.method == "classic"
    assert n.params == {"plan": True, "n_examples": 1, "repair_rounds": 2,
                        "critique_rounds": 1, "critique_mode": "patch"}
    assert n.contexts == ["palette", {"ontology": {"variant": "mined", "path": None, "seed": 3}},
                          {"examples": {"k": 1}}]
    assert recipe.from_agent_config(AgentConfig()).contexts == ["palette"]


def test_set_dotted_and_parse_sets():
    d = {}
    recipe.set_dotted(d, "a.b.c", 1)
    recipe.set_dotted(d, "a.d", 2)
    assert d == {"a": {"b": {"c": 1}, "d": 2}}
    with pytest.raises(ValueError):
        recipe.set_dotted(d, "a..b", 1)
    assert recipe.parse_sets(["params.max_iters=5", "dreams=[iso, plan]", "image=null",
                              "llm=openai:gpt-5"]) == {
        "params.max_iters": 5, "dreams": ["iso", "plan"], "image": None, "llm": "openai:gpt-5"}
    with pytest.raises(ValueError):
        recipe.parse_sets(["novalue"])


def test_deep_merge_does_not_alias():
    base = {"a": {"b": [1]}}
    out = recipe.deep_merge(base, {"a": {"c": 2}})
    out["a"]["b"].append(9)
    assert base == {"a": {"b": [1]}}


# --- batteries -----------------------------------------------------------------
def test_battery_inline_overrides_and_sweep(rdir):
    _, b = rdir
    write(b, "bat", """
        name: dream_v1
        prompts: {source: detailed, n: 12, seed: 0}
        budget: {usd: 8.0}
        cache: read_only
        recipes:
          - dream
          - {dream: {as: dream_vdm, params.feedback: vdm}}
          - legacy-zeroshot
        sweep:
          base: dream
          grid: {params.max_iters: [1, 3], dreams: [[iso], [iso, interior, plan]]}
          max_arms: 4
    """)
    bat = recipe.load_battery("bat", {"params.stop_at": 0.9})
    assert bat.name == "dream_v1" and bat.prompts["n"] == 12 and bat.budget.usd == 8.0
    assert bat.arm_names() == ["dream", "dream_vdm", "legacy-zeroshot",
                               "dream@max_iters=1,dreams=iso",
                               "dream@max_iters=1,dreams=iso+interior+plan",
                               "dream@max_iters=3,dreams=iso",
                               "dream@max_iters=3,dreams=iso+interior+plan"]
    vdm = bat.recipes[1]
    assert vdm.params["feedback"] == "vdm" and vdm.params["stop_at"] == 0.9
    assert all(r.cache == "read_only" for r in bat.recipes)
    sw = bat.recipes[-1]
    assert sw.params["max_iters"] == 3 and sw.dreams == ["iso", "interior", "plan"]
    assert len({r.hash() for r in bat.recipes[3:]}) == 4


def test_sweep_max_arms_and_duplicates(rdir):
    _, b = rdir
    write(b, "big", """
        recipes: []
        sweep: {base: dream, grid: {params.max_iters: [1, 2, 3], params.patience: [1, 2, 3]}}
    """)
    with pytest.raises(ValueError, match="9 arms > max_arms=8"):
        recipe.load_battery("big")
    write(b, "dup", "recipes: [dream, dream]\n")
    with pytest.raises(ValueError, match="duplicate arm"):
        recipe.load_battery("dup")
    write(b, "badkey", "recipes: [dream]\nbudgt: {usd: 1}\n")
    with pytest.raises(TypeError, match="budgt"):
        recipe.load_battery("badkey")


def test_sweep_arm_name_slugs():
    assert recipe.sweep_arm_name("dream", {"params.max_iters": 1, "dreams": ["iso"]}) == \
        "dream@max_iters=1,dreams=iso"
    assert recipe.sweep_arm_name("d", {"image": "diffusers:sdxl?steps=4"}) == "d@image=diffusers-sdxl-steps-4"
    assert recipe.sweep_arm_name("d", {"dreams": [{"key": "back", "template": "t"}]}) == "d@dreams=back"


def test_real_recipe_dir_loads_if_present():
    """Every shipped recipe (WP7) must load. Skipped until the directory exists."""
    names = recipe.list_recipes(virtual=False)
    if not names:
        pytest.skip("configs/agentic/recipes is empty (WP7 not landed)")
    for n in names:
        assert recipe.load_recipe(n).name == n
