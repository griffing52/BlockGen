"""ClaudeCodeProvider against a fake ``claude`` executable on PATH.

The fake records what it was given (argv, cwd, env, stdin, which image files
existed during the call) into a JSON log and replies like ``claude -p
--output-format json``. Its behaviour is chosen by ``FAKE_CLAUDE_MODE``. This pins
the exact CLI contract without spending quota; the one real call lives in
test_providers_cli_live.py (marked ``spend``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from blockgen.agentic import core, providers_cli
from blockgen.agentic.core import SpendNotAllowed
from blockgen.agentic.providers import (CachingProvider, Message, Part, encode_png,
                                        get_provider, text_message)
from blockgen.agentic.providers_cli import (CLOSING_INSTRUCTION, FALLBACK_SYSTEM,
                                            ClaudeCodeProvider, child_env)

FAKE = textwrap.dedent('''\
    #!{python}
    import json, os, sys, time
    argv = sys.argv[1:]
    if argv == ["--version"]:
        print(os.environ.get("FAKE_CLAUDE_VERSION", "9.9.9") + " (Claude Code)")
        sys.exit(0)
    body = sys.stdin.read()
    imgs = {{}}
    for tok in body.split("[image: ")[1:]:
        p = tok.split("]", 1)[0]
        imgs[p] = os.path.isfile(p) and open(p, "rb").read().hex()
    log = os.environ["FAKE_CLAUDE_LOG"]
    calls = json.load(open(log)) if os.path.exists(log) else []
    calls.append({{"argv": argv, "cwd": os.getcwd(), "stdin": body, "images": imgs,
                   "env": {{k: v for k, v in os.environ.items()
                            if k.startswith(("ANTHROPIC", "CLAUDE"))}}}})
    json.dump(calls, open(log, "w"))
    mode = os.environ.get("FAKE_CLAUDE_MODE", "ok")
    if mode == "sleep":
        time.sleep(5)
    if mode == "exit1":
        sys.stderr.write("API Error: 529 overloaded\\n")
        sys.exit(1)
    if mode == "badjson":
        print("not json at all")
        sys.exit(0)
    out = {{"type": "result", "subtype": "success", "is_error": False,
            "result": "fill stone 0 0 0 1 1 1", "num_turns": 2, "duration_ms": 1234,
            "total_cost_usd": 0.0123, "session_id": "abc",
            "usage": {{"input_tokens": 10, "cache_read_input_tokens": 100,
                       "cache_creation_input_tokens": 1000, "output_tokens": 7}}}}
    if mode == "is_error":
        out.update(subtype="error_during_execution", is_error=True, result="boom happened")
    print(json.dumps(out))
''')


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "claude"
    exe.write_text(FAKE.format(python=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "calls.json"
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")
    monkeypatch.setenv("BLOCKGEN_SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("CLAUDE_CODE_SSE_PORT", "1234")
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "keep-me")
    monkeypatch.setattr(providers_cli, "_VERSIONS", {})
    monkeypatch.setattr(ClaudeCodeProvider, "_SEMS", {})
    monkeypatch.setattr(ClaudeCodeProvider, "retry_delay", 0.0)

    class Log:
        path = log

        @property
        def calls(self):
            return json.loads(log.read_text()) if log.exists() else []
    return Log()


def _png(rgb):
    return encode_png(np.full((8, 8, 3), rgb, dtype=np.uint8))


RED, BLUE = _png((255, 0, 0)), _png((0, 0, 255))


def _msgs():
    return [text_message("system", "You build houses."),
            Message("user", [Part("text", text="Build this:"), Part("image", image_png=RED),
                             Part("text", text="and this"), Part("image", image_png=BLUE),
                             Part("image", image_png=RED)]),
            text_message("assistant", "ok"),
            text_message("user", "Now the final program.")]


def test_argv_cwd_env_and_stdin(fake_claude, tmp_path):
    p = ClaudeCodeProvider("haiku", max_turns=4)
    assert p.params["cli_version"] == "9.9.9"
    r = p.complete(_msgs())
    (call,) = fake_claude.calls
    argv = call["argv"]
    td = argv[argv.index("--add-dir") + 1]
    assert argv == ["-p", "--output-format", "json", "--model", "haiku",
                    "--system-prompt", "You build houses.", "--tools", "Read",
                    "--allowedTools", "Read", "--add-dir", td, "--max-turns", "4",
                    "--strict-mcp-config", "--setting-sources", "",
                    "--no-session-persistence", "--safe-mode"]
    assert "--bare" not in argv                    # --bare cannot use OAuth subscription auth
    assert os.path.realpath(call["cwd"]) == os.path.realpath(td)
    assert Path(td).parent == tmp_path / "scratch"
    assert not Path(td).exists()                   # workspace removed after the call
    # env: API key and parent-session variables stripped, OAuth token kept
    env = call["env"]
    assert "ANTHROPIC_API_KEY" not in env and "CLAUDE_CODE_SSE_PORT" not in env
    assert "CLAUDECODE" not in env and env.get("CLAUDE_CODE_OAUTH_TOKEN") == "keep-me"
    # stdin: tagged turns, image markers in part order, closing instruction
    body = call["stdin"]
    assert body.rstrip().endswith(CLOSING_INSTRUCTION)
    assert '<turn role="user">' in body and '<turn role="assistant">\nok\n</turn>' in body
    assert "You build houses." not in body         # system text goes to --system-prompt
    markers = re.findall(r"\[image: ([^\]]+)\]", body)
    assert len(markers) == 3 and markers[0] == markers[2] != markers[1]
    assert body.index("Build this:") < body.index(markers[0]) < body.index("and this")
    # content-addressed names, files present during the call with the right bytes
    for m, png in zip(markers, (RED, BLUE, RED)):
        assert Path(m).name == f"img_{hashlib.sha256(png).hexdigest()[:32]}.png"
        assert os.path.isabs(m) and call["images"][m] == png.hex()
    # response
    assert r.text == "fill stone 0 0 0 1 1 1" and r.model == "claude-code/haiku"
    assert r.prompt_tokens == 1110 and r.completion_tokens == 7
    assert r.finish_reason == "success" and r.priced is False and r.cost_usd == 0.0
    assert r.meta["total_cost_usd"] == 0.0123 and r.meta["n_images"] == 2


def test_max_turns_grows_with_images(fake_claude):
    p = ClaudeCodeProvider("haiku", max_turns=2)
    imgs = [Part("image", image_png=_png((i, 0, 0))) for i in range(5)]
    p.complete([Message("user", [Part("text", text="look")] + imgs)])
    argv = fake_claude.calls[0]["argv"]
    assert argv[argv.index("--max-turns") + 1] == "7"


def test_no_system_message_uses_fallback_and_options(fake_claude):
    p = ClaudeCodeProvider("opus", safe_mode=False, bare=True, effort="low")
    p.complete([text_message("user", "hi")])
    argv = fake_claude.calls[0]["argv"]
    assert argv[argv.index("--system-prompt") + 1] == FALLBACK_SYSTEM
    assert "--safe-mode" not in argv and "--bare" in argv
    assert argv[argv.index("--effort") + 1] == "low"


def test_keep_workspace(fake_claude):
    p = ClaudeCodeProvider("haiku", keep_workspace=True)
    p.complete(_msgs())
    argv = fake_claude.calls[0]["argv"]
    td = Path(argv[argv.index("--add-dir") + 1])
    assert sorted(f.name for f in td.iterdir()) == sorted(
        {f"img_{hashlib.sha256(x).hexdigest()[:32]}.png" for x in (RED, BLUE)})


def test_cli_version_is_in_the_cache_key(fake_claude, tmp_path, monkeypatch):
    a = CachingProvider(ClaudeCodeProvider("haiku"), cache_dir=tmp_path)
    monkeypatch.setattr(providers_cli, "_VERSIONS", {})
    monkeypatch.setenv("FAKE_CLAUDE_VERSION", "10.0.0")
    b = CachingProvider(ClaudeCodeProvider("haiku"), cache_dir=tmp_path)
    assert a.params["cli_version"] == "9.9.9" and b.params["cli_version"] == "10.0.0"
    assert a._key(_msgs(), {}) != b._key(_msgs(), {})
    # operational knobs do NOT move the key
    c = CachingProvider(ClaudeCodeProvider("haiku", timeout_s=5, concurrency=7), cache_dir=tmp_path)
    assert c._key(_msgs(), {}) == b._key(_msgs(), {})


def test_spend_gate_then_cache(fake_claude, tmp_path, monkeypatch):
    p = get_provider("claude-code:haiku", cache_dir=tmp_path / "cache", spend_gate=True)
    with pytest.raises(SpendNotAllowed):
        p.complete(_msgs())
    assert fake_claude.calls == []                 # binary never invoked
    monkeypatch.setenv(core.SPEND_ENV, "1")
    first = p.complete(_msgs())
    second = p.complete(_msgs())
    assert not first.cached and second.cached and second.text == first.text
    assert second.meta["total_cost_usd"] == 0.0123
    assert len(fake_claude.calls) == 1


def test_is_error_raises_with_message_no_retry(fake_claude, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "is_error")
    with pytest.raises(RuntimeError, match="boom happened"):
        ClaudeCodeProvider("haiku").complete(_msgs())
    assert len(fake_claude.calls) == 1             # not transient: no retry


def test_transient_exit_is_retried_once_with_stderr_tail(fake_claude, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "exit1")
    with pytest.raises(RuntimeError, match="exited 1: API Error: 529 overloaded"):
        ClaudeCodeProvider("haiku").complete(_msgs())
    assert len(fake_claude.calls) == 2


def test_bad_json_raises(fake_claude, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "badjson")
    with pytest.raises(RuntimeError, match="not JSON"):
        ClaudeCodeProvider("haiku").complete(_msgs())


def test_timeout(fake_claude, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "sleep")
    t0 = time.time()
    with pytest.raises(RuntimeError, match="timed out after 0.5s"):
        ClaudeCodeProvider("haiku", timeout_s=0.5).complete(_msgs())
    assert time.time() - t0 < 4.5                  # two 0.5 s attempts, not 5 s sleeps
    assert len(fake_claude.calls) == 2


def test_missing_binary(monkeypatch, tmp_path):
    monkeypatch.setattr(providers_cli, "_VERSIONS", {})
    p = ClaudeCodeProvider("haiku", binary=str(tmp_path / "no-such-claude"))
    assert p.params["cli_version"] == "missing"
    with pytest.raises(RuntimeError, match="not found"):
        p.complete(_msgs())


def test_child_env_rules():
    env = child_env({"ANTHROPIC_API_KEY": "x", "CLAUDE_CODE_ENTRYPOINT": "cli",
                     "CLAUDE_CODE_OAUTH_TOKEN": "t", "CLAUDE_CONFIG_DIR": "/c",
                     "PATH": "/bin", "CLAUDE_PID": "1"})
    assert env == {"CLAUDE_CODE_OAUTH_TOKEN": "t", "CLAUDE_CONFIG_DIR": "/c", "PATH": "/bin"}


# ===========================================================================
# Verifier regressions
# ===========================================================================
def test_relative_scratch_gives_absolute_add_dir(fake_claude, tmp_path, monkeypatch):
    """cwd=td for the child, so a relative --add-dir would point at td/td."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BLOCKGEN_SCRATCH", "rel_scratch")
    ClaudeCodeProvider("haiku").complete(_msgs())
    (call,) = fake_claude.calls
    td = call["argv"][call["argv"].index("--add-dir") + 1]
    assert os.path.isabs(td) and Path(td).parent == (tmp_path / "rel_scratch").resolve()
    assert all(v for v in call["images"].values())          # every marker was readable


def test_long_system_prompt_goes_through_a_file(fake_claude, monkeypatch):
    """One argv string is capped at 128 KiB on Linux (E2BIG)."""
    big = "palette row\n" * 20000                              # ~240 KB
    ClaudeCodeProvider("haiku").complete([text_message("system", big),
                                         text_message("user", "hi")])
    argv = fake_claude.calls[0]["argv"]
    assert "--system-prompt" not in argv and big not in argv
    path = Path(argv[argv.index("--system-prompt-file") + 1])
    assert path.name == "system_prompt.txt" and str(path).startswith(
        argv[argv.index("--add-dir") + 1])
    assert not path.exists()                                   # removed with the workspace
