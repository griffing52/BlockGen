"""providers.py suite additions (cache modes, spend gate, atomic writes, new vendors)
and the offline AutoMockProvider (``mock:auto``).

Everything here is offline: paying vendors are constructed but their clients are
monkeypatched to fail loudly if touched.
"""

from __future__ import annotations

import json

import pytest

from blockgen.agentic import core, providers, providers_cli
from blockgen.agentic.core import CacheMiss, SpendNotAllowed, billing_of
from blockgen.agentic.providers import (CachingProvider, LLMResponse, LocalProvider, Message,
                                        OllamaProvider, OpenAIProvider, Part, ScriptedProvider,
                                        get_provider, text_message)
from blockgen.agentic.providers_cli import AUTO_JSON, AutoMockProvider

from .fakes import FakeActions, ScriptedLLM


def _msgs():
    return [text_message("system", "SYS"),
            Message("user", [Part("text", text="hi"), Part("image", image_png=b"\x89PNGfake")])]


@pytest.fixture
def no_network(monkeypatch):
    """Any attempt to build an OpenAI client fails the test."""
    def boom(self):
        raise AssertionError("network client constructed")
    monkeypatch.setattr(OpenAIProvider, "_client_lazy", boom)


# ===========================================================================
# Key stability (existing cache files must still hit)
# ===========================================================================
def test_cache_key_is_byte_stable_golden():
    """Golden keys computed with the PRE-WP4 providers.py: the suite additions must
    not move a single legacy cache entry."""
    p = CachingProvider(ScriptedProvider(model="gpt-5-mini", max_tokens=16000, temperature=None,
                                         reasoning_effort="medium"), cache_dir="/nonexistent")
    assert p._key(_msgs(), {}) == \
        "26363b98ed8cd910362ea83bd17b457b0fafab352471bb658a7b38df73ac75b8"
    assert p._key(_msgs(), {"temperature": 0.5}) == \
        "5d384dfd9250b2431dce696cd7c0fef3e59185229ef8dbc290df5970df74e781"


def test_legacy_cache_record_has_no_new_keys(tmp_path):
    inner = ScriptedLLM(["prog"])
    p = CachingProvider(inner, cache_dir=tmp_path)
    p.complete(_msgs())
    (f,) = tmp_path.glob("*.json")
    assert set(json.loads(f.read_text())) == {"text", "model", "prompt_tokens",
                                              "completion_tokens", "finish_reason"}


def test_legacy_get_provider_shapes(no_network):
    assert isinstance(get_provider("mock", cache=False), ScriptedProvider)
    assert isinstance(get_provider("mock"), ScriptedProvider)            # mock never cached
    bare = get_provider("openai:gpt-5-mini", cache=False, max_tokens=5)
    assert type(bare) is OpenAIProvider and bare.params["max_tokens"] == 5
    cached = get_provider("openai")
    assert isinstance(cached, CachingProvider) and cached.mode == "read_write"
    assert cached.spend_gate is False and cached.model == "gpt-5-mini"
    with pytest.raises(ValueError, match="unknown provider"):
        get_provider("notavendor:model")


def test_meta_round_trips_through_cache(tmp_path):
    class WithMeta(ScriptedLLM):
        def complete(self, messages, **o):
            r = super().complete(messages, **o)
            r.meta = {"total_cost_usd": 0.01}
            return r
    p = CachingProvider(WithMeta(["x"]), cache_dir=tmp_path)
    assert p.complete(_msgs()).meta == {"total_cost_usd": 0.01}
    hit = p.complete(_msgs())
    assert hit.cached and hit.meta == {"total_cost_usd": 0.01}
    assert "meta" not in LLMResponse("t", "m").to_dict()      # transcripts unchanged


# ===========================================================================
# Cache modes, spend gate, peek, atomic writes
# ===========================================================================
def test_read_only_miss_raises_and_hit_serves(tmp_path):
    inner = ScriptedLLM(["one"])
    ro = CachingProvider(inner, cache_dir=tmp_path, mode="read_only")
    with pytest.raises(CacheMiss):
        ro.complete(_msgs())
    assert inner.n_calls == 0
    CachingProvider(inner, cache_dir=tmp_path).complete(_msgs())        # populate
    assert ro.complete(_msgs()).cached and inner.n_calls == 1


def test_bad_cache_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match="cache mode"):
        CachingProvider(ScriptedLLM(), cache_dir=tmp_path, mode="sometimes")
    with pytest.raises(ValueError, match="cache mode"):
        get_provider("openai", cache_mode="sometimes")


def test_spend_gate_blocks_paying_miss_without_network(tmp_path, no_network):
    p = get_provider("openai:gpt-5-mini", cache_dir=tmp_path, spend_gate=True)
    assert p.spends and billing_of(p) == "api"
    with pytest.raises(SpendNotAllowed, match="BLOCKGEN_ALLOW_SPEND"):
        p.complete(_msgs())
    assert not list(tmp_path.glob("*.json"))


def test_spend_gate_also_gates_cache_off(tmp_path, no_network):
    p = get_provider("openai:gpt-5-mini", cache_mode="off", spend_gate=True)
    assert isinstance(p, CachingProvider) and not p.enabled and p.mode == "off"
    with pytest.raises(SpendNotAllowed):
        p.complete(_msgs())
    # without the gate, 'off' is the legacy bare provider
    assert type(get_provider("openai:gpt-5-mini", cache_mode="off")) is OpenAIProvider


def test_spend_gate_passes_with_env_and_hits_are_free(tmp_path, monkeypatch):
    inner = ScriptedLLM(["paid"])
    inner.spends = True
    p = CachingProvider(inner, cache_dir=tmp_path, spend_gate=True)
    monkeypatch.setenv(core.SPEND_ENV, "1")
    assert p.complete(_msgs()).text == "paid"
    monkeypatch.delenv(core.SPEND_ENV)
    assert p.complete(_msgs()).cached               # hit needs no permission
    assert inner.n_calls == 1


def test_spend_gate_ignores_free_backends(tmp_path):
    p = CachingProvider(ScriptedLLM(["free"]), cache_dir=tmp_path, spend_gate=True)
    assert not p.spends and p.billing == "mock"
    assert p.complete(_msgs()).text == "free"


def test_peek(tmp_path):
    p = CachingProvider(ScriptedLLM(["x"]), cache_dir=tmp_path)
    assert not p.peek(_msgs())
    p.complete(_msgs())
    assert p.peek(_msgs()) and not p.peek(_msgs(), temperature=1.0)
    assert not CachingProvider(ScriptedLLM(), cache_dir=tmp_path, enabled=False).peek(_msgs())


def test_atomic_write_leaves_no_partial_file(tmp_path, monkeypatch):
    p = CachingProvider(ScriptedLLM(["x"]), cache_dir=tmp_path)

    def failing_replace(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(providers.os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        p.complete(_msgs())
    assert list(tmp_path.iterdir()) == []            # neither entry nor temp file
    monkeypatch.undo()
    p.complete(_msgs())
    assert len(list(tmp_path.glob("*.json"))) == 1 and not list(tmp_path.glob("*.tmp"))


def test_billing_and_spends_attributes():
    assert providers.LLMProvider.spends is True and providers.LLMProvider.billing == "api"
    assert ScriptedProvider.spends is False and ScriptedProvider.billing == "mock"
    assert LocalProvider.billing == "local" and not LocalProvider.spends
    assert OllamaProvider.billing == "local" and not OllamaProvider.spends
    assert providers_cli.ClaudeCodeProvider.billing == "subscription"
    assert providers_cli.ClaudeCodeProvider.spends is True


# ===========================================================================
# New vendors
# ===========================================================================
def test_default_models_and_vendor_table():
    dm = providers.DEFAULT_MODELS
    assert dm["claude-code"] == "sonnet" and dm["local"] == "default"
    assert dm["ollama"] == "qwen2.5vl:7b"
    for v in ("claude-code", "cc", "local", "ollama"):
        assert v in providers._PROVIDERS


def test_local_and_ollama_resolve_base_url(monkeypatch, tmp_path):
    monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_KEY", raising=False)
    monkeypatch.setattr(providers, "_ENV_LOADED", True)      # ignore the repo .env
    loc = get_provider("local", cache_dir=tmp_path)
    assert isinstance(loc, CachingProvider) and loc.model == "default"
    assert loc.inner._base_url == "http://localhost:8000/v1"
    assert loc.params["base_url"] == "http://localhost:8000/v1"      # part of the cache key
    assert get_provider("ollama:llava", cache=False)._base_url == "http://localhost:11434/v1"
    monkeypatch.setenv("LOCAL_LLM_URL", "http://gpu-box:9000/v1")
    assert LocalProvider("m")._base_url == "http://gpu-box:9000/v1"
    assert LocalProvider("m", base_url="http://x/v1")._base_url == "http://x/v1"

    seen = {}

    class FakeOpenAI:
        def __init__(self, **kw):
            seen.update(kw)
    import openai
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    LocalProvider("m")._client_lazy()
    assert seen == {"api_key": "local", "base_url": "http://gpu-box:9000/v1"}
    monkeypatch.setenv("LOCAL_LLM_KEY", "sekret")
    LocalProvider("m")._client_lazy()
    assert seen["api_key"] == "sekret"


def test_local_keys_differ_by_server(tmp_path):
    a = CachingProvider(LocalProvider("m", base_url="http://a/v1"), cache_dir=tmp_path)
    b = CachingProvider(LocalProvider("m", base_url="http://b/v1"), cache_dir=tmp_path)
    assert a._key(_msgs(), {}) != b._key(_msgs(), {})


def test_local_is_never_gated(tmp_path, monkeypatch):
    p = get_provider("ollama:llava", cache_dir=tmp_path, spend_gate=True)
    calls = []
    monkeypatch.setattr(p.inner, "complete", lambda m, **o: calls.append(1) or
                        LLMResponse("ok", "llava"))
    assert p.complete(_msgs()).text == "ok" and calls == [1]


def test_claude_code_vendor_via_get_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(providers_cli, "cli_version", lambda binary="claude": "1.2.3")
    p = get_provider("cc:haiku", cache_dir=tmp_path, spend_gate=True, max_tokens=99)
    assert isinstance(p, CachingProvider) and isinstance(p.inner, providers_cli.ClaudeCodeProvider)
    assert p.model == "haiku" and p.params["cli_version"] == "1.2.3"
    assert p.params["max_tokens"] == 99 and billing_of(p) == "subscription"
    assert get_provider("claude-code", cache=False).model == "sonnet"


# ===========================================================================
# mock:auto
# ===========================================================================
class OtherActions(FakeActions):
    name = "other_dsl"
    OUTPUT_RULE = "Reply with only the OTHER program, in one fenced block."

    def __init__(self) -> None:
        super().__init__(reply="fill stone 0 0 0 1 1 1\n")


@pytest.fixture
def two_spaces(registry_snapshot):
    registry_snapshot.register("actions", "other_dsl")(OtherActions)
    providers_cli.reset_action_space_cache()
    yield
    providers_cli.reset_action_space_cache()


def test_get_provider_mock_auto_is_uncached():
    p = get_provider("mock:auto", max_tokens=16000)
    assert isinstance(p, AutoMockProvider) and not p.spends and p.billing == "mock"
    assert p.describe() == "mock:auto" and billing_of(p) == "mock"
    r = p.complete([text_message("user", "hello")])
    assert r.text == providers._DEMO_PROGRAM and not r.priced and r.cost_usd == 0.0


def test_mock_auto_replies_per_action_space_rule(two_spaces):
    p = AutoMockProvider()
    sys_msg = text_message("system", "grammar...\n" + FakeActions.OUTPUT_RULE)
    # rule in the user turn wins over the system message
    r = p.complete([sys_msg, text_message("user", "Build a hut.\n" + OtherActions.OUTPUT_RULE)])
    assert r.text == "fill stone 0 0 0 1 1 1\n" and p.decisions[-1] == "actions:other_dsl"
    # no rule in the turn -> the system message decides
    r = p.complete([sys_msg, text_message("user", "Build a hut.")])
    assert r.text == FakeActions().mock_reply() and p.decisions[-1] == "actions:fake_dsl"
    # both rules in one text: the LAST occurrence wins
    both = FakeActions.OUTPUT_RULE + " ... " + OtherActions.OUTPUT_RULE
    p.complete([text_message("user", both)])
    assert p.decisions[-1] == "actions:other_dsl"


def test_mock_auto_json_rule(monkeypatch):
    rule = "Reply with only a JSON object."
    monkeypatch.setattr(providers_cli, "_json_rule", lambda: rule)
    p = AutoMockProvider(json_overrides={"score": 3})
    sys_msg = text_message("system", FakeActions.OUTPUT_RULE)       # JSON beats the action rule
    r = p.complete([sys_msg, text_message("user", "Review the dream.\n" + rule + "  ")])
    blob = json.loads(r.text)
    assert blob == {**AUTO_JSON, "score": 3} and p.decisions[-1] == "json"
    # JSON rule not at the END of the turn -> not a JSON turn
    p.complete([text_message("user", rule + "\nnow build")])
    assert p.decisions[-1] != "json"


def test_auto_json_satisfies_rubrics():
    for k, v in {"score": 7, "buildable": True, "on_prompt": True, "issues": [],
                 "revised_prompt": "", "index": 0, "unreachable": False,
                 "differences": []}.items():
        assert AUTO_JSON[k] == v


def test_mock_auto_json_fallback_without_turns(monkeypatch):
    monkeypatch.setattr(providers_cli, "_json_rule", lambda: None)
    p = AutoMockProvider()
    p.complete([text_message("user", "Rate it. Answer as JSON.")])
    assert p.decisions[-1] == "json"


def test_mock_auto_script_overrides(monkeypatch):
    monkeypatch.setattr(providers_cli, "_json_rule", lambda: "JSON please.")
    p = AutoMockProvider(script={"Compare": "REDREAM: a smaller house",
                                 "reachable": {"unreachable": True, "revised_prompt": "tiny"}})
    assert p.complete([text_message("user", "Compare TARGET and YOURS.")]).text \
        == "REDREAM: a smaller house"
    r = p.complete([text_message("user", "Is it reachable? JSON please.")])
    assert json.loads(r.text) == {"unreachable": True, "revised_prompt": "tiny"}


def test_mock_auto_token_estimates():
    p = AutoMockProvider()
    r = p.complete([Message("user", [Part("text", text="x" * 400),
                                     Part("image", image_png=b"png")])])
    assert r.prompt_tokens == 100 + 800 and r.completion_tokens == len(r.text) // 4
    assert r.meta["decision"] == "demo"


# ------------------------------------------- local: thinking + vision switches
def test_local_thinking_switch_and_text_only_vision(monkeypatch):
    from blockgen.agentic.providers import LocalProvider, Message, Part
    p = LocalProvider("m", thinking="false", vision="false")
    assert p._extra_kwargs(p.params) == {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    assert LocalProvider("m")._extra_kwargs(LocalProvider("m").params) == {}
    sent = {}

    class FakeChat:
        def create(self, **kw):
            sent.update(kw)
            from types import SimpleNamespace as NS
            return NS(choices=[NS(message=NS(content="ok"), finish_reason="stop")],
                      usage=NS(prompt_tokens=1, completion_tokens=1))

    from types import SimpleNamespace as NS
    p._client = NS(chat=NS(completions=FakeChat()))
    msg = Message("user", [Part("text", text="TARGET:"), Part("image", image_png=b"png")])
    assert p.complete([msg]).text == "ok"
    wire = sent["messages"][0]["content"]
    assert "image_url" not in str(wire) and LocalProvider.IMAGE_PLACEHOLDER in str(wire)
    assert sent["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_local_max_images_keeps_the_most_recent():
    from blockgen.agentic.providers import LocalProvider, Message, Part
    p = LocalProvider("m", max_images=2)
    msgs = [Message("user", [Part("image", image_png=bytes([i])), Part("text", text=f"t{i}")])
            for i in range(4)]
    out = p._cap_images(msgs, 2)
    kinds = [pt.kind for m in out for pt in m.parts if pt.kind == "image" or "omitted" in pt.text]
    assert kinds == ["text", "text", "image", "image"]
    assert [pt.image_png for m in out for pt in m.parts if pt.kind == "image"] == [b"\x02", b"\x03"]
