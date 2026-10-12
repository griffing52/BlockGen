"""``python -m blockgen.agentic`` and the ``experiments_agentic`` wiring, offline."""

from __future__ import annotations

import argparse
import socket
import sys

import pytest

from blockgen.agentic import core
from blockgen.agentic.__main__ import LIST_KINDS, list_kind, main


@pytest.fixture(autouse=True)
def _clean_kit_cache():
    from blockgen.agentic import kit
    kit.clear_cache()
    yield
    kit.clear_cache()


def test_list_shows_every_kind(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    for kind in LIST_KINDS:
        assert f"{kind}:" in out
    # Every registered component of every registry kind is listed with its params.
    for kind in core.KINDS:
        for name in core.names(kind):
            if not name.startswith("fake_"):
                assert f"  {name}" in out, (kind, name)
    assert "dream(refine=0" in out and "turnaround expands to" in out


def test_list_recipes_has_every_shipped_recipe():
    from blockgen.agentic.recipe import list_recipes
    text = "\n".join(list_kind("recipes"))
    assert "ERROR" not in text
    for name in list_recipes():
        assert name in text
    for name in ("dream", "dream-refine", "dream-multiview", "dream-vote", "dream-local",
                 "dream-oracle", "classic-matched", "blocks-oneshot", "python-oneshot",
                 "mixed-dream", "texsheet-oneshot", "texsheet-shuffled", "best-of-4",
                 "dream-python-vlm", "dsl-classic", "legacy-full"):
        assert name in list_recipes()


def test_show_applies_overrides(capsys):
    assert main(["show", "dream", "--set", "params.max_iters=5"]) == 0
    out = capsys.readouterr().out
    assert "max_iters: 5" in out and "recipe_hash:" in out and "provenance:" in out


def test_run_mock(tmp_path, capsys):
    rc = main(["run", "--recipe", "dream,classic-matched", "--prompts", "short:2", "--mock",
               "--imagegen", "mock:solid", "--no-render", "--base", str(tmp_path),
               "--name", "cli"])
    assert rc == 0
    runs = list(tmp_path.glob("run_*_cli_mock"))
    assert len(runs) == 1
    assert (runs[0] / "dream_32.npz").is_file() and (runs[0] / "classic-matched_32.npz").is_file()
    assert "paid $0.0000" in capsys.readouterr().out


def test_run_without_allow_spend_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(core.SPEND_ENV, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    rc = main(["run", "--recipe", "classic-full", "--prompts", "short:1", "--no-render",
               "--cache", "off", "--base", str(tmp_path)])
    assert rc == 2
    assert "--allow-spend" in capsys.readouterr().err


def test_estimate_prints_table_without_network(monkeypatch, capsys):
    monkeypatch.setattr(socket.socket, "connect",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    assert main(["estimate", "--recipe", "dream-refine", "--prompts", "short:1"]) == 0
    out = capsys.readouterr().out
    assert "est_full_usd" in out and "dream-refine" in out and "TOTAL" in out


def test_estimate_battery(monkeypatch, capsys):
    monkeypatch.setattr(socket.socket, "connect",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    assert main(["estimate", "--battery", "repr-v1", "--prompts", "short:1"]) == 0
    out = capsys.readouterr().out
    for arm in ("dsl_oneshot", "blocks-oneshot", "texsheet-shuffled"):
        assert arm in out


def test_calibrate_delegates(capsys):
    main(["calibrate", "--scorer", "pixel", "--n", "4", "--distractors", "3",
          "--synthetic", "--renderer", "flat"])
    assert "top1" in capsys.readouterr().out


def test_mock_flags_are_post_mock():
    from blockgen.agentic.__main__ import _overrides, _post_mock, build_parser
    a = build_parser().parse_args(["run", "--recipe", "dream", "--mock", "--renderer",
                                   "textured", "--set", "params.max_iters=2"])
    assert _overrides(a) == {"params.max_iters": 2}
    assert _post_mock(a) == {"renderer": "textured"}
    a = build_parser().parse_args(["run", "--recipe", "dream", "--llm-role",
                                   "judge=claude-code:sonnet", "--renderer", "flat"])
    assert _overrides(a) == {"llm.judge": "claude-code:sonnet", "renderer": "flat"}
    assert _post_mock(a) is None


# ---------------------------------------------------------------------------
# experiments_agentic
# ---------------------------------------------------------------------------
def _legacy_args(**kw):
    base = dict(provider="mock", canvas=16, canvas_height=None, max_tokens=1000,
                reasoning_effort=None, temperature=None, target_blocks=None,
                critique_mode="rewrite", ontology_path=None, ontology_seed=0,
                no_cache=True, quiet=True)
    base.update(kw)
    return argparse.Namespace(**base)


def test_arm_may_override_critique_mode(monkeypatch):
    """Used to raise 'got multiple values for keyword argument critique_mode'."""
    from blockgen.agentic.agent import BuildAgent
    from blockgen import experiments_agentic as ea
    monkeypatch.setattr(BuildAgent, "_render", lambda self, s: [])
    res = ea.run_arm("x", dict(ea.ARMS["critique"], critique_mode="patch",
                               target_blocks=50), ["a hut"], _legacy_args())
    assert res[0].config.critique_mode == "patch" and res[0].config.target_blocks == 50


def test_experiments_agentic_recipes_route_to_suite(tmp_path, monkeypatch):
    from blockgen import experiments_agentic as ea
    from blockgen.agentic import suite
    seen = {}

    def fake_run(specs, prompts, **kw):
        seen.update(names=[s.name for s in specs], n=len(prompts), **kw)
        return tmp_path
    monkeypatch.setattr(suite, "run_recipes", fake_run)
    monkeypatch.setattr(sys, "argv", ["x", "--recipes", "dream,classic-matched", "--mock",
                                      "--prompts", "short", "--n", "2", "--no-render"])
    ea.main()
    assert seen["names"] == ["dream", "classic-matched"] and seen["n"] == 2
    assert seen["mock"] is True and seen["render"] is False


def test_experiments_agentic_battery_uses_battery_prompts(tmp_path, monkeypatch):
    from blockgen import experiments_agentic as ea
    from blockgen.agentic import suite
    seen = {}
    monkeypatch.setattr(suite, "run_battery",
                        lambda b, **kw: seen.update(b=b, **kw) or tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--battery", "repr-v1", "--mock"])
    ea.main()
    assert seen["b"] == "repr-v1" and seen["prompts"] is None and seen["mock"] is True
