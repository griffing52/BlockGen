"""Regression tests for the review fixes in core / kit / recipe / episode / dream
loop / suite / CLI (one test per fix; all offline, no spend)."""

from __future__ import annotations

import json
import socket
import sys
import typing

import pytest

from blockgen.agentic import core, suite
from blockgen.agentic import kit as kitmod
from blockgen.agentic.core import (Budget, BudgetExceeded, ConfigError, EpisodeLike, Ledger,
                                   ScoreTerm)
from blockgen.agentic.episode import Episode, run_episode
from blockgen.agentic.recipe import load_recipe
from blockgen.agentic.recipes import steps
from blockgen.agentic.recipes.dream import dream

from .fakes import (FakeImageGen, FakeRenderer, RecordingTrace, ScriptedLLM, fake_scorer,
                    make_fake_kit)

COMPARE, REACH, REVIEW = ("Compare your build with the target", "BEST BUILD SO FAR",
                          "IMAGE TO REVIEW")
OFFLINE = {"renderer": "flat"}


def prog(n: int) -> str:
    return f"fill stone 0 0 0 {n - 1} 0 9\n"


def seq(*items):
    left = list(items)

    def reply(_msgs):
        return left.pop(0) if len(left) > 1 else left[0]
    return reply


def match_scorer(ep, att):
    return ScoreTerm(min(1.0, att.blocks / 500.0))


match_scorer.needs_dream = True


def improve_accept(prev, cand):
    if cand.blocks == 0:
        return False, "empty"
    if prev is None or prev.score is None:
        return True, "first"
    ok = cand.score.value > prev.score.value
    return ok, "improved" if ok else "not better"


def run_dream(params, *, rules=(), dreams=("iso",), image=None):
    llm = ScriptedLLM([prog(10)], rules=list(rules))
    kit = make_fake_kit(llm=llm, method=dream, dreams=list(dreams), accept=improve_accept,
                        scorers={"match": (match_scorer, 1.0)}, image=image or FakeImageGen(),
                        spec_kw={"params": params, "select": "best"})
    trace = RecordingTrace()
    return run_episode(kit, "a small hut", trace=trace), llm, trace


@pytest.fixture(autouse=True)
def _clean_kit_cache():
    kitmod.clear_cache()
    yield
    kitmod.clear_cache()


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)


# ------------------------------------------------------------- dream loop fixes
def test_redream_dependants_use_the_new_description():
    res, _, _ = run_dream({"max_iters": 3, "redream": 1, "stop_at": 2.0, "patience": 5},
                          dreams=("iso", "back"),
                          rules=[(COMPARE, seq("REDREAM: a tiny shed", prog(20), prog(30)))])
    iso1, back1 = res.dreams["iso"][1], res.dreams["back"][1]
    assert "a tiny shed" in iso1.prompt and "a tiny shed" in back1.prompt
    assert "a small hut" not in back1.prompt


def test_empty_redream_desc_keeps_the_refined_description():
    bad = json.dumps({"score": 2, "buildable": False, "on_prompt": True,
                      "revised_prompt": "a plain box hut"})
    reach = json.dumps({"unreachable": True, "reason": "x", "revised_prompt": ""})
    res, _, _ = run_dream({"refine": 1, "max_iters": 6, "redream": 1, "redream_patience": 1,
                           "patience": 3},
                          rules=[(REVIEW, bad), (REACH, reach), (COMPARE, prog(5))])
    vs = res.dreams["iso"]
    assert len(vs) == 3
    assert "a plain box hut" in vs[1].prompt
    assert "a plain box hut" in vs[2].prompt and "a small hut" not in vs[2].prompt


def test_after_redream_the_next_match_turn_shows_the_new_target():
    res, llm, trace = run_dream({"max_iters": 3, "redream": 1, "stop_at": 2.0, "patience": 5,
                                 "feedback": "score"},
                                rules=[(COMPARE, seq("REDREAM: a tiny shed", prog(20),
                                                     prog(30)))])
    new_png = res.dreams["iso"][1].png
    idx = [i for i, r in enumerate(trace.requests) if r["stage"].startswith("match")]
    after = llm.calls[idx[1]]                                # first match turn after REDREAM
    pngs = [p.image_png for m in after for p in m.parts if p.kind == "image"]
    assert new_png in pngs
    assert "TARGET has changed" in after[-1].text
    # Once that turn was committed to the history it is not repeated.
    last = llm.calls[idx[-1]][-1]
    assert "TARGET has changed" not in last.text


def test_reachable_verdict_is_not_re_asked_every_stalled_iteration():
    res, _, trace = run_dream({"max_iters": 6, "redream": 1, "redream_patience": 2,
                               "patience": 6},
                              rules=[(REACH, '{"unreachable": false}'), (COMPARE, prog(5))])
    assert len(res.attempts) == 7                    # generate + 6 stalled match turns
    n = [r["stage"] for r in trace.requests].count("reachability:iso")
    assert n == 3          # at stall 2, 4, 6 (was 5: at every stall >= 2)


# ------------------------------------------------------------------ episode fixes
def test_llm_usd_cap_is_checked_against_the_next_calls_estimate():
    llm = ScriptedLLM(["x"], model="gpt-5", prompt_tokens=10, completion_tokens=10)
    kit = make_fake_kit(llm=llm, spec_kw={"gen": {"max_tokens": 16000}})
    ep = Episode("a hut", kit, ledger=Ledger(Budget(usd=0.01), name="b"))
    with pytest.raises(BudgetExceeded):          # 4800 tok x $10/M = $0.048 > $0.01
        ep.ask("Build it")
    assert llm.calls == []


def test_needs_renders_false_scorer_skips_score_renders():
    renderer = FakeRenderer()
    kit = make_fake_kit(replies=[prog(10)], renderer=renderer)   # fake_scorer: no renders
    ep = Episode("a hut", kit)
    att = ep.try_("generate", prog(10))
    assert att.renders == {} and renderer.calls == []

    def pixel(ep, att):
        return fake_scorer(ep, att)
    pixel.needs_renders = True
    ep2 = Episode("a hut", make_fake_kit(renderer=renderer, scorers={"p": (pixel, 1.0)}))
    assert set(ep2.try_("generate", prog(10)).renders) == {"iso", "iso_back"}


def test_episode_implements_every_episodelike_member():
    ep = Episode("a hut", make_fake_kit())
    members = set(typing.get_type_hints(EpisodeLike)) | {
        n for n, v in vars(EpisodeLike).items() if callable(v) and not n.startswith("_")}
    missing = sorted(m for m in members if not hasattr(ep, m))
    assert missing == []
    assert "p" not in members and not hasattr(ep, "p")


def test_method_result_cost_usd_metric_is_full_price():
    llm = ScriptedLLM([prog(10)], model="gpt-5-mini", prompt_tokens=100, completion_tokens=10,
                      cached=True)
    res = run_episode(make_fake_kit(llm=llm), "a hut")
    m = res.metrics()
    assert res.cost_usd == 0.0 and m["paid_usd"] == 0.0
    assert m["cost_usd"] == pytest.approx(res.would_cost_usd) and m["cost_usd"] > 0


# ------------------------------------------------------------------- suite fixes
class _CountingImage(FakeImageGen):
    seeds: list = []

    def generate(self, req):
        type(self).seeds.append(req.seed)
        return super().generate(req)


@pytest.mark.parametrize("variation", [None, 5])
def test_dreams_first_pays_each_dream_once_and_counts_would_cost_once(
        tmp_path, registry_snapshot, variation):
    core.register("image", "counting_probe")(_CountingImage)
    _CountingImage.seeds = []
    spec = load_recipe("dream", {"params.max_iters": 0})
    run = suite.run_recipes([spec], ["a hut"], name="df", mock=True,
                            post_mock={**OFFLINE, "image": "counting_probe"},
                            dreams_first=True, seed=variation, render=False, base=tmp_path)
    assert _CountingImage.seeds == [0 if variation is None else variation]
    led = json.loads((run / "metrics.json").read_text())["ledger"]
    assert led["usd"] == pytest.approx(0.01)
    assert led["would_cost_usd"] == pytest.approx(0.01)


def test_estimate_counts_every_repair_round(no_network):
    rows = suite.estimate([load_recipe("classic-full")], ["a hut"])
    # plan + generate + 2 repairs + 1 critique (repairs used to be skipped: mock
    # replies never fail, so the 'upper bound' was 3).
    assert rows[0]["llm_calls"] == 5


def test_needs_repair_worst_case_kit():
    kit = make_fake_kit()
    ep = Episode("a hut", kit)
    att = ep.try_("generate", prog(10))
    assert not steps.needs_repair(att, ep)
    kit.worst_case = True
    assert steps.needs_repair(att, ep)


def test_config_typo_in_any_arm_fails_before_the_first_call(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(suite, "_dreams_first", lambda *a, **k: called.append(1))
    import blockgen.agentic.episode as epmod
    monkeypatch.setattr(epmod, "run_episode", lambda *a, **k: called.append(1))
    good = load_recipe("classic-matched")
    bad = load_recipe("dream", {"params.max_iter": 3, "name": "dream_typo"})
    with pytest.raises(ConfigError, match=r"dream_typo.*max_iters"):
        suite.run_recipes([good, bad], ["a hut"], name="typo", mock=True, post_mock=OFFLINE,
                          render=False, base=tmp_path)
    assert called == [] and not list(tmp_path.iterdir())


def test_cli_config_errors_exit_2_with_one_line(capsys, tmp_path):
    from blockgen.agentic.__main__ import main
    assert main(["run", "--recipe", "dream", "--set", "params.max_iter=3", "--mock",
                 "--prompts", "short:1", "--base", str(tmp_path), "--no-render"]) == 2
    err = capsys.readouterr().err
    assert "config error" in err and "max_iters" in err and "Traceback" not in err
    assert main(["show", "dream", "--set", "score.embedd=1"]) == 2
    assert main(["show", "dreem"]) == 2
    assert "did you mean 'dream'" in capsys.readouterr().err


def test_run_exits_nonzero_when_an_arm_cannot_be_built(tmp_path, monkeypatch):
    from blockgen.agentic import __main__ as cli
    real = kitmod.build_kit

    def broken(spec, **kw):
        if spec.name == "classic-matched":
            raise ImportError("optional dependency missing")
        return real(spec, **kw)
    monkeypatch.setattr(kitmod, "build_kit", broken)
    rc = cli.main(["run", "--recipe", "classic-matched", "--mock", "--prompts", "short:1",
                   "--base", str(tmp_path), "--no-render"])
    assert rc == 1


# ------------------------------------------------------------------ recipe fixes
def test_legacy_recipe_uses_experiments_agentic_cli_defaults():
    s = load_recipe("legacy-ont_mined")
    assert s.canvas == (48, 48, 48) and s.llm == "openai:gpt-5-mini"
    assert s.gen == {"max_tokens": 16000, "temperature": None, "reasoning_effort": "medium"}
    assert "ont_mined" in s.description and "ontology=mined" in s.description


def test_recipe_lookup_suggests_and_lists_only_loadable_names():
    with pytest.raises(FileNotFoundError) as exc:
        load_recipe("dreem")
    msg = str(exc.value)
    assert "did you mean 'dream'" in msg and "'_base'" not in msg and "legacy-full" in msg


def test_show_output_loads_back_as_the_same_recipe(capsys, tmp_path):
    from blockgen.agentic.__main__ import main
    assert main(["show", "dream", "--set", "params.max_iters=5"]) == 0
    out = capsys.readouterr().out
    assert "# recipe_hash:" in out
    path = tmp_path / "shown.yaml"
    path.write_text(out)
    again = load_recipe(path)
    assert again.hash() == load_recipe("dream", {"params.max_iters": 5}).hash()


# --------------------------------------------------------------------- CLI list
def test_list_accepts_singular_and_plural_and_shows_lazy_params(capsys):
    from blockgen.agentic.__main__ import list_kind, main
    assert main(["list", "scorers"]) == 0 and main(["list", "recipe"]) == 0
    capsys.readouterr()
    lines = "\n".join(list_kind("image"))
    assert "diffusers(model=" in lines
    method = "\n".join(list_kind("method"))
    assert "and keep the best-scoring build" in method       # full first paragraph


def test_load_builtins_raises_when_a_present_module_has_a_broken_import(
        tmp_path, monkeypatch):
    (tmp_path / "broken_builtin_probe.py").write_text(
        "import blockgen.agentic.does_not_exist_probe\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(core, "BUILTIN_MODULES", ("broken_builtin_probe",))
    monkeypatch.setattr(core, "_BUILTINS_LOADED", False)
    with pytest.raises(ModuleNotFoundError, match="does_not_exist_probe"):
        core.load_builtins()
    sys.modules.pop("broken_builtin_probe", None)
    monkeypatch.setattr(core, "BUILTIN_MODULES", ("absent_builtin_probe_xyz",))
    monkeypatch.setattr(core, "_BUILTINS_SKIPPED", {})
    with pytest.warns(UserWarning, match="skipped"):
        core.load_builtins()                     # a wholly absent module is skipped
