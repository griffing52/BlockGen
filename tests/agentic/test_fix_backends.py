"""Regression tests for the backend review fixes (imagegen + providers). Offline."""

from __future__ import annotations

import base64
import inspect
from types import SimpleNamespace

import pytest

from blockgen.agentic import imagegen
from blockgen.agentic.core import ImageRequest
from blockgen.agentic.imagegen import OpenAIImageGen
from blockgen.agentic.providers import (DEFAULT_MODELS, LocalProvider, get_provider,
                                        text_message)

from .fakes import solid_png


def _req(**kw):
    return ImageRequest(**{"prompt": "a small oak cottage", **kw})


def _resp(n, usage=None):
    png = base64.b64encode(solid_png((1, 2, 3))).decode()
    return SimpleNamespace(data=[SimpleNamespace(b64_json=png, revised_prompt=None)
                                 for _ in range(n)], usage=usage)


class StrictImages:
    """Mirrors the openai 2.x signatures: edit() has no `moderation`, generate()
    has no `input_fidelity`; neither takes **kwargs."""

    def __init__(self, usage=None):
        self.calls, self.usage = [], usage

    def generate(self, *, prompt, model=None, n=None, size=None, quality=None,
                 background=None, moderation=None, style=None, extra_body=None):
        self.calls.append(("generate", {k: v for k, v in locals().items()
                                        if k not in ("self",) and v is not None}))
        return _resp(n, self.usage)

    def edit(self, *, image, prompt, model=None, n=None, size=None, quality=None,
             background=None, input_fidelity=None, mask=None, extra_body=None):
        self.calls.append(("edit", {k: v for k, v in locals().items()
                                    if k not in ("self", "image") and v is not None}))
        return _resp(n, self.usage)


# --- generate-only params must not crash edits ------------------------------
def test_moderation_is_dropped_on_edit_and_kept_on_generate():
    fake = StrictImages()
    gen = OpenAIImageGen("gpt-image-1-mini", moderation="low", input_fidelity="high")
    gen._client = SimpleNamespace(images=fake)
    gen.generate(_req())
    kind, kw = fake.calls[-1]
    assert kind == "generate" and kw["moderation"] == "low" and "input_fidelity" not in kw
    gen.generate(_req(refs=(b"ref",)))        # used to raise TypeError
    kind, kw = fake.calls[-1]
    assert kind == "edit" and "moderation" not in kw and kw["input_fidelity"] == "high"


def test_unknown_params_route_through_extra_body():
    fake = StrictImages()
    gen = OpenAIImageGen("gpt-image-1-mini", future_knob=3)
    gen._client = SimpleNamespace(images=fake)
    gen.generate(_req(refs=(b"ref",)))
    _, kw = fake.calls[-1]
    assert kw["extra_body"] == {"future_knob": 3} and "future_knob" not in kw


def test_real_sdk_edit_signature_has_no_moderation():
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(api_key="x", base_url="http://127.0.0.1:9")
    out = imagegen._endpoint_kwargs(client.images.edit, {"prompt": "p", "moderation": "low"},
                                    imagegen.GENERATE_ONLY_PARAMS)
    assert "moderation" not in out
    inspect.signature(client.images.edit).bind(image=b"", **out)   # would raise TypeError


# --- cost ledger counts input tokens ----------------------------------------
def test_estimate_prices_prompt_text_tokens():
    short = OpenAIImageGen("gpt-image-1-mini").estimate_usd(_req(prompt="x"))
    long = OpenAIImageGen("gpt-image-1-mini").estimate_usd(_req(prompt="x" * 1500))
    assert long - short == pytest.approx(
        imagegen.INPUT_TOKEN_USD["gpt-image-1-mini"][0] * (500 - 1))


def test_ref_surcharge_covers_a_1024px_input_image():
    for model, (_, img_rate) in imagegen.INPUT_TOKEN_USD.items():
        assert imagegen.REF_SURCHARGE_USD[model] >= 765 * img_rate


def test_measured_cost_uses_response_usage():
    usage = SimpleNamespace(input_tokens=1100, output_tokens=272,
                            input_tokens_details=SimpleNamespace(text_tokens=100,
                                                                 image_tokens=1000))
    fake = StrictImages(usage=usage)
    gen = OpenAIImageGen("gpt-image-1")
    gen._client = SimpleNamespace(images=fake)
    r = gen.generate(_req(refs=(b"ref",)))
    text_rate, img_rate = imagegen.INPUT_TOKEN_USD["gpt-image-1"]
    assert r.priced and r.cost_usd == pytest.approx(0.011 + 100 * text_rate + 1000 * img_rate)


# --- FLUX cannot load whole on a 16 GB GPU ----------------------------------
def test_flux_vram_check_and_sequential_offload():
    pytest.importorskip("diffusers")
    flux = imagegen.DiffusersImageGen("flux-schnell")
    with pytest.raises(RuntimeError, match="offload=true"):
        flux.check_vram(16303 * 2**20)
    imagegen.DiffusersImageGen("flux-schnell", offload=True).check_vram(16303 * 2**20)
    imagegen.DiffusersImageGen("sdxl-turbo").check_vram(16303 * 2**20)
    assert flux._family(flux.SEQUENTIAL_OFFLOAD) == "flux"


# --- providers ----------------------------------------------------------------
def test_bare_cc_alias_resolves_default_model():
    assert DEFAULT_MODELS["cc"] == DEFAULT_MODELS["claude-code"]
    p = get_provider("cc", cache=False)
    assert "sonnet" in str(getattr(p, "model", ""))


def test_local_provider_defaults_to_small_max_tokens():
    assert LocalProvider("m", base_url="http://x/v1").params["max_tokens"] == 4096
    assert LocalProvider("m", base_url="http://x/v1", max_tokens=900).params["max_tokens"] == 900


def test_local_provider_retries_without_cap_on_context_error():
    calls = []

    def create(**kw):
        calls.append(kw)
        if "max_tokens" in kw:
            raise RuntimeError("Error code: 400 - This model's maximum context length is "
                               "8192 tokens. However, you requested 17000 tokens")
        msg = SimpleNamespace(content="ok")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=5, completion_tokens=1))

    p = LocalProvider("m", base_url="http://x/v1")
    p._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    r = p.complete([text_message("user", "hi")], max_tokens=16000)
    assert r.text == "ok" and calls[0]["max_tokens"] == 16000 and "max_tokens" not in calls[-1] and len(calls) == 2


def test_local_provider_other_errors_propagate():
    def create(**kw):
        raise RuntimeError("Error code: 404 - model not found")

    p = LocalProvider("m", base_url="http://x/v1")
    p._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(RuntimeError, match="not found"):
        p.complete([text_message("user", "hi")])


class _StatusError(Exception):
    def __init__(self, msg, status_code=None):
        super().__init__(msg)
        self.status_code = status_code


@pytest.mark.parametrize("exc, transient", [
    # a vLLM 400 whose text merely contains '500': never retried
    (RuntimeError("Error code: 400 - you requested 16500 tokens"), False),
    (_StatusError("requested 16500 tokens (503 in prompt)", status_code=400), False),
    # genuine transients: by status code, or by text when there is no status
    (_StatusError("Rate limit reached", status_code=429), True),
    (_StatusError("server error", status_code=503), True),
    (_StatusError("Overloaded", status_code=529), True),
    (RuntimeError("Error code: 502 - Bad Gateway"), True),
    (RuntimeError("Request timed out."), True),
    (ConnectionError("Connection refused"), True),
])
def test_retry_transient_detection(monkeypatch, exc, transient):
    from blockgen.agentic import providers

    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    calls = []

    def fn():
        calls.append(1)
        if len(calls) == 1:
            raise exc
        return "ok"

    if transient:
        assert providers._retry(fn) == "ok" and len(calls) == 2
    else:
        with pytest.raises(type(exc)):
            providers._retry(fn)
        assert len(calls) == 1
