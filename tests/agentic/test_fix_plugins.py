"""Regression tests for the staged-variant usability review.

* ``--plugin`` / ``BLOCKGEN_AGENTIC_PLUGINS``: a method in a scratch file is
  reachable from a recipe without editing ``BUILTIN_MODULES``.
* docs/agentic-suite.md "Writing a staged variant": the snippet runs as written.
* ``estimate`` is an upper bound for unit loops: ``max_room_iters`` above 1 must
  raise the call count (the unit patience stop used to cap it at 1 round).
"""

from __future__ import annotations

import re
import socket
from pathlib import Path

import pytest

from blockgen.agentic import core
from blockgen.agentic.__main__ import main

ROOT = Path(__file__).resolve().parents[2]
SUITE_DOC = ROOT / "docs" / "agentic-suite.md"

RECIPE = """\
extends: _base
description: rooms first, then the exterior dream
method: rooms_first
llm: {builder: openai:gpt-5-mini, planner: openai:gpt-5-mini, critic: openai:gpt-5-mini}
image: openai:gpt-image-1-mini?quality=low&size=1024x1024
dream_style: render_match
dreams: [iso]
score: {silhouette: 0.5, validity: 0.3}
final_score: {validity: 0.5}
accept: improve_valid
select: best
params: {max_room_iters: 1, max_iters: 1}
budget: {usd_per_build: 0.80, max_images: 8}
"""


def _staged_snippet() -> str:
    text = SUITE_DOC.read_text()
    sec = text[text.index("#### Writing a staged variant"):text.index("#### Ideas to try next")]
    return re.search(r"```python\n(.*?)```", sec, re.S).group(1)


@pytest.fixture
def plugin_env(monkeypatch, registry_snapshot):
    """Plugins loaded by a test leave neither registry entries nor env behind."""
    # setenv first so monkeypatch records the var's absence and removes what main()
    # sets (delenv on an absent var records nothing).
    monkeypatch.setenv(core.PLUGINS_ENV, "")
    monkeypatch.delenv(core.PLUGINS_ENV)
    monkeypatch.setattr(core, "_PLUGINS_LOADED", {})
    yield


def _no_network(monkeypatch) -> None:
    monkeypatch.setattr(socket.socket, "connect",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))


def test_unknown_method_hints_at_plugin():
    with pytest.raises(KeyError, match="--plugin"):
        core.lookup("method", "rooms_first_not_registered")


def test_load_plugins_file_registers_method_and_is_idempotent(tmp_path, plugin_env):
    f = tmp_path / "rooms_first.py"
    f.write_text(_staged_snippet())
    mods = core.load_plugins([str(f)])
    assert core.lookup("method", "rooms_first").meta.get("prefetch_dreams") is False
    assert core.load_plugins([str(f)]) == mods       # second load: no re-import


def test_load_plugins_missing_raises_config_error(tmp_path, plugin_env):
    with pytest.raises(core.ConfigError, match="not found"):
        core.load_plugins([str(tmp_path / "nope.py")])
    with pytest.raises(core.ConfigError, match="not found"):
        core.load_plugins(["no_such_pkg_xyz.plugin"])


def test_cli_plugin_bad_path_exits_2(tmp_path, plugin_env, capsys):
    assert main(["list", "method", "--plugin", str(tmp_path / "nope.py")]) == 2
    assert "plugin file not found" in capsys.readouterr().err
    assert not core.os.environ.get(core.PLUGINS_ENV)    # a failed plugin is not exported


def test_docs_staged_variant_runs_with_plugin(tmp_path, plugin_env, monkeypatch, capsys):
    """The documented rooms-first variant, from a scratch dir, offline."""
    _no_network(monkeypatch)
    f = tmp_path / "rooms_first.py"
    f.write_text(_staged_snippet())
    rec = tmp_path / "rooms-first.yaml"
    rec.write_text(RECIPE)
    rc = main(["run", "--plugin", str(f), "--recipe", str(rec), "--prompts", "short:1",
               "--mock", "--imagegen", "mock:solid", "--renderer", "flat", "--no-render",
               "--base", str(tmp_path), "--name", "plug"])
    assert rc == 0, capsys.readouterr().err
    run = next(tmp_path.glob("run_*_plug_mock"))
    iters = sorted(p.name for p in (run / "builds" / "rooms-first" / "000" / "iters").iterdir())
    stages = [n.split("_", 1)[1] for n in iters]
    assert stages[0] == "shell" and any(s.startswith("u_") for s in stages)
    # rooms precede the roof, and the exterior dream exists (made after the roof)
    assert stages.index("roof") > max(i for i, s in enumerate(stages) if s.startswith("u_"))
    assert (run / "builds" / "rooms-first" / "000" / "dreams" / "iso_v0.png").is_file()
    assert str(f) in core.os.environ[core.PLUGINS_ENV]


def _estimate_calls(capsys, iters: int) -> int:
    assert main(["estimate", "--recipe", "blueprint", "--prompts", "short:1",
                 "--set", f"params.max_room_iters={iters}"]) == 0
    row = next(line for line in capsys.readouterr().out.splitlines()
               if line.startswith("blueprint "))
    return int(row.split()[2])


def test_estimate_counts_every_unit_round(monkeypatch, capsys):
    """Each extra room round must add calls to the upper bound (it used to plateau)."""
    _no_network(monkeypatch)
    one, three = _estimate_calls(capsys, 1), _estimate_calls(capsys, 3)
    assert three > one
