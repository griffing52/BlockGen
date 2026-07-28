"""Provider-agnostic LLM access for the agentic track.

The agent (``blockgen/agentic/agent.py``) never imports a vendor SDK. It talks to
:class:`LLMProvider`, which is a two-method interface: send a list of
:class:`Message`, get an :class:`LLMResponse`. That is what lets the same
experiment run against OpenAI, Gemini, Anthropic or a scripted stub without
touching the pipeline — the provider is a string in the config
(``--provider openai:gpt-5-mini``).

Three things live here because every provider needs them and none should
reimplement them:

* **Multimodal messages.** A message's content is a list of parts — text or a PNG
  image — so image conditioning (build from this reference picture) and visual
  self-critique (here is a render of what you built) are provider-independent.
  Providers adapt the parts to their own wire format.
* **A response cache on disk.** Keyed by the full request (model, messages,
  sampling params), so re-running an experiment for a new metric, a re-render, or
  a crashed sweep costs nothing and returns byte-identical text. This matters more
  than usual here: LLM sampling is the expensive, non-reproducible step, and a
  cached run is a *deterministic* run.
* **Usage and cost.** Every call records prompt/completion tokens; :data:`PRICES`
  turns them into dollars so a battery reports what it cost to run.

Keys are read from the process environment, falling back to the repo ``.env``
(``OPENAI_API_KEY``, ``GEMINI_API_KEY``/``GOOGLE_API_KEY``, ``ANTHROPIC_API_KEY``).
Nothing here prints or logs a key.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = _REPO / "outputs" / "agentic_cache"

# Approximate list prices in USD per 1M tokens, (input, output). Used only for the
# cost line in run reports -- update freely, nothing depends on the numbers.
PRICES: Dict[str, Tuple[float, float]] = {
    "gpt-5": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-5-nano": (0.05, 0.4),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.3, 2.5),
    "gemini-2.0-flash": (0.1, 0.4),
    "claude-opus-4-5": (5.0, 25.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


# --- messages --------------------------------------------------------------
@dataclass
class Part:
    """One piece of a message: text, or a PNG image."""

    kind: str                      # "text" | "image"
    text: str = ""
    image_png: Optional[bytes] = None

    def fingerprint(self) -> str:
        if self.kind == "text":
            return f"t:{self.text}"
        digest = hashlib.sha256(self.image_png or b"").hexdigest()[:32]
        return f"i:{digest}"


@dataclass
class Message:
    role: str                      # "system" | "user" | "assistant"
    parts: List[Part] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.parts if p.kind == "text")


def text_message(role: str, text: str) -> Message:
    return Message(role=role, parts=[Part("text", text=text)])


def image_message(role: str, text: str, images: Sequence[bytes]) -> Message:
    parts = [Part("text", text=text)] + [Part("image", image_png=img) for img in images]
    return Message(role=role, parts=parts)


def encode_png(image) -> bytes:
    """``np.ndarray`` (H,W,3|4) or PIL image -> PNG bytes, for image parts."""
    from PIL import Image
    if hasattr(image, "save"):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
    import numpy as np
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = (255 * arr.clip(0, 1)).astype(np.uint8)
    if arr.shape[-1] == 4:  # composite onto white; models see the build, not alpha
        alpha = arr[..., 3:4].astype(np.float32) / 255.0
        arr = (arr[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    latency_s: float = 0.0
    finish_reason: str = ""

    @property
    def cost_usd(self) -> float:
        price = PRICES.get(self.model)
        if price is None:  # unknown model -> unpriced, not guessed
            return 0.0
        return (self.prompt_tokens * price[0] + self.completion_tokens * price[1]) / 1e6

    def to_dict(self) -> Dict[str, Any]:
        return {"model": self.model, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "cached": self.cached,
                "latency_s": round(self.latency_s, 2), "cost_usd": round(self.cost_usd, 6),
                "finish_reason": self.finish_reason}


# --- env / keys ------------------------------------------------------------
_ENV_LOADED = False


def load_env() -> None:
    """Load the repo ``.env`` once (process env always wins)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    env_path = _REPO / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:  # tiny fallback so python-dotenv stays optional
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def _require_key(*names: str) -> str:
    load_env()
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError(
        f"no API key found: set {' or '.join(names)} in the environment or in "
        f"{_REPO / '.env'}")


# --- provider interface ----------------------------------------------------
class LLMProvider:
    """Minimal interface every backend implements."""

    name = "base"

    def __init__(self, model: str, **params: Any):
        self.model = model
        self.params = params

    def complete(self, messages: Sequence[Message], **overrides: Any) -> LLMResponse:
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name}:{self.model}"


class CachingProvider(LLMProvider):
    """Wraps a provider with an on-disk response cache (see the module docstring)."""

    def __init__(self, inner: LLMProvider, cache_dir: Path | str = DEFAULT_CACHE_DIR,
                 enabled: bool = True):
        super().__init__(inner.model, **inner.params)
        self.inner = inner
        self.name = inner.name
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

    def _key(self, messages: Sequence[Message], overrides: Dict[str, Any]) -> str:
        payload = {
            "provider": self.inner.name, "model": self.model,
            "params": {k: v for k, v in sorted({**self.params, **overrides}.items())},
            "messages": [[m.role] + [p.fingerprint() for p in m.parts] for m in messages],
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()

    def complete(self, messages: Sequence[Message], **overrides: Any) -> LLMResponse:
        if not self.enabled:
            return self.inner.complete(messages, **overrides)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{self._key(messages, overrides)}.json"
        if path.is_file():
            try:
                blob = json.loads(path.read_text())
                return LLMResponse(text=blob["text"], model=blob["model"],
                                   prompt_tokens=blob.get("prompt_tokens", 0),
                                   completion_tokens=blob.get("completion_tokens", 0),
                                   cached=True, latency_s=0.0,
                                   finish_reason=blob.get("finish_reason", ""))
            except (json.JSONDecodeError, KeyError):
                pass  # corrupt cache entry: fall through and re-request
        resp = self.inner.complete(messages, **overrides)
        path.write_text(json.dumps({
            "text": resp.text, "model": resp.model,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "finish_reason": resp.finish_reason}, indent=1))
        return resp

    def describe(self) -> str:
        return self.inner.describe()


def _retry(fn, attempts: int = 4, base_delay: float = 2.0):
    """Retry transient API failures (rate limits, 5xx) with exponential backoff."""
    last: Optional[Exception] = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - vendor SDKs raise their own types
            message = f"{type(exc).__name__}: {exc}".lower()
            transient = any(s in message for s in (
                "rate limit", "ratelimit", "429", "timeout", "timed out",
                "overloaded", "503", "502", "500", "connection"))
            if not transient or i == attempts - 1:
                raise
            last = exc
            time.sleep(base_delay * (2 ** i))
    raise last  # pragma: no cover


# --- OpenAI (and OpenAI-compatible endpoints) ------------------------------
class OpenAIProvider(LLMProvider):
    """OpenAI Chat Completions. Also the transport for OpenAI-compatible hosts."""

    name = "openai"

    def __init__(self, model: str = "gpt-5-mini", *, api_key_names=("OPENAI_API_KEY",),
                 base_url: Optional[str] = None, max_tokens: int = 16000,
                 temperature: Optional[float] = None,
                 reasoning_effort: Optional[str] = "medium", **params: Any):
        super().__init__(model, max_tokens=max_tokens, temperature=temperature,
                         reasoning_effort=reasoning_effort, **params)
        self._api_key_names = api_key_names
        self._base_url = base_url
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            from openai import OpenAI
            kwargs: Dict[str, Any] = {"api_key": _require_key(*self._api_key_names)}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    @property
    def _is_reasoning(self) -> bool:
        m = self.model
        return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") \
            or m.startswith("o4")

    def _wire(self, messages: Sequence[Message]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in messages:
            if all(p.kind == "text" for p in m.parts):
                out.append({"role": m.role, "content": m.text})
                continue
            content: List[Dict[str, Any]] = []
            for p in m.parts:
                if p.kind == "text":
                    content.append({"type": "text", "text": p.text})
                else:
                    b64 = base64.b64encode(p.image_png or b"").decode()
                    content.append({"type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{b64}"}})
            out.append({"role": m.role, "content": content})
        return out

    def complete(self, messages: Sequence[Message], **overrides: Any) -> LLMResponse:
        params = {**self.params, **overrides}
        kwargs: Dict[str, Any] = {"model": self.model, "messages": self._wire(messages)}
        max_tokens = params.get("max_tokens")
        if max_tokens:
            # Reasoning models bill thinking against the completion budget, so a
            # small cap silently returns empty content (see scripts/zeroshot_brickgpt).
            kwargs["max_completion_tokens" if self._is_reasoning else "max_tokens"] = \
                int(max_tokens)
        if self._is_reasoning:
            if params.get("reasoning_effort"):
                kwargs["reasoning_effort"] = params["reasoning_effort"]
        elif params.get("temperature") is not None:
            kwargs["temperature"] = float(params["temperature"])

        client = self._client_lazy()
        t0 = time.time()

        def call():
            return client.chat.completions.create(**kwargs)

        try:
            resp = _retry(call)
        except Exception as exc:  # unsupported-parameter fallback, then give up
            message = str(exc).lower()
            dropped = [k for k in ("reasoning_effort", "temperature") if k in kwargs
                       and k in message]
            if not dropped:
                raise
            for k in dropped:
                kwargs.pop(k)
            resp = _retry(call)

        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=choice.message.content or "", model=self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=time.time() - t0,
            finish_reason=getattr(choice, "finish_reason", "") or "")


class GeminiProvider(OpenAIProvider):
    """Google Gemini.

    Uses the native ``google-genai`` SDK when it is installed, and otherwise
    Google's OpenAI-compatible endpoint — which keeps Gemini usable with no extra
    dependency, since the repo already has the ``openai`` client.
    """

    name = "gemini"
    OPENAI_COMPAT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(self, model: str = "gemini-2.5-flash", **params: Any):
        params.setdefault("reasoning_effort", None)
        super().__init__(model, api_key_names=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
                         base_url=self.OPENAI_COMPAT_BASE, **params)

    @property
    def _is_reasoning(self) -> bool:
        return False  # the compat endpoint takes max_tokens/temperature directly

    def _native_client(self):
        try:
            from google import genai  # type: ignore
        except ImportError:
            return None
        return genai.Client(api_key=_require_key("GEMINI_API_KEY", "GOOGLE_API_KEY"))

    def complete(self, messages: Sequence[Message], **overrides: Any) -> LLMResponse:
        client = self._native_client()
        if client is None:
            return super().complete(messages, **overrides)
        from google.genai import types  # type: ignore
        params = {**self.params, **overrides}
        system = "\n\n".join(m.text for m in messages if m.role == "system")
        contents = []
        for m in messages:
            if m.role == "system":
                continue
            parts = []
            for p in m.parts:
                if p.kind == "text":
                    parts.append(types.Part.from_text(text=p.text))
                else:
                    parts.append(types.Part.from_bytes(data=p.image_png or b"",
                                                       mime_type="image/png"))
            contents.append(types.Content(
                role="model" if m.role == "assistant" else "user", parts=parts))
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            max_output_tokens=int(params.get("max_tokens") or 16000),
            temperature=params.get("temperature"))
        t0 = time.time()
        resp = _retry(lambda: client.models.generate_content(
            model=self.model, contents=contents, config=config))
        usage = getattr(resp, "usage_metadata", None)
        return LLMResponse(
            text=resp.text or "", model=self.model,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            latency_s=time.time() - t0)


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API."""

    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-5", *, max_tokens: int = 16000,
                 temperature: Optional[float] = None, **params: Any):
        super().__init__(model, max_tokens=max_tokens, temperature=temperature, **params)
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=_require_key("ANTHROPIC_API_KEY"))
        return self._client

    def complete(self, messages: Sequence[Message], **overrides: Any) -> LLMResponse:
        params = {**self.params, **overrides}
        system = "\n\n".join(m.text for m in messages if m.role == "system")
        wire = []
        for m in messages:
            if m.role == "system":
                continue
            content = []
            for p in m.parts:
                if p.kind == "text":
                    content.append({"type": "text", "text": p.text})
                else:
                    content.append({"type": "image", "source": {
                        "type": "base64", "media_type": "image/png",
                        "data": base64.b64encode(p.image_png or b"").decode()}})
            wire.append({"role": m.role, "content": content})
        kwargs: Dict[str, Any] = {"model": self.model, "messages": wire,
                                  "max_tokens": int(params.get("max_tokens") or 16000)}
        if system:
            kwargs["system"] = system
        if params.get("temperature") is not None:
            kwargs["temperature"] = float(params["temperature"])
        client = self._client_lazy()
        t0 = time.time()
        resp = _retry(lambda: client.messages.create(**kwargs))
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return LLMResponse(
            text=text, model=self.model,
            prompt_tokens=getattr(resp.usage, "input_tokens", 0) or 0,
            completion_tokens=getattr(resp.usage, "output_tokens", 0) or 0,
            latency_s=time.time() - t0,
            finish_reason=getattr(resp, "stop_reason", "") or "")


class ScriptedProvider(LLMProvider):
    """Returns canned responses in order — the offline stand-in for tests and demos.

    ``mock:path/to/replies.json`` loads a JSON list of strings; ``mock:`` with no
    path replays a small built-in house program. Every test in
    ``tests/test_agentic_*.py`` runs the *real* agent loop through this, so the
    pipeline is covered without a network call or an API key.
    """

    name = "mock"

    def __init__(self, model: str = "scripted", replies: Optional[Sequence[str]] = None,
                 **params: Any):
        super().__init__(model, **params)
        self.replies = list(replies) if replies else [_DEMO_PROGRAM]
        self.calls: List[List[Message]] = []
        self._i = 0

    def complete(self, messages: Sequence[Message], **overrides: Any) -> LLMResponse:
        self.calls.append(list(messages))
        reply = self.replies[min(self._i, len(self.replies) - 1)]
        self._i += 1
        return LLMResponse(text=reply, model=self.model, prompt_tokens=0,
                           completion_tokens=0, latency_s=0.0)


_DEMO_PROGRAM = """\
# A small cottage — the built-in scripted reply.
fill cobblestone 0 0 0 9 0 7
walls oak_planks 0 1 0 9 4 7
fill oak_planks 0 5 0 9 5 7
clear 4 1 0 5 3 0
fill glass 2 2 0 2 3 0
fill glass 7 2 0 7 3 0
gable oak_stairs 0 5 0 9 5 7 axis=x overhang=1
"""


# --- factory ---------------------------------------------------------------
_PROVIDERS = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "google": GeminiProvider,
    "anthropic": AnthropicProvider,
    "claude": AnthropicProvider,
    "mock": ScriptedProvider,
}

DEFAULT_MODELS = {
    "openai": "gpt-5-mini",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-4-5",
    "mock": "scripted",
}


def get_provider(spec: str, *, cache: bool = True,
                 cache_dir: Path | str = DEFAULT_CACHE_DIR, **params: Any) -> LLMProvider:
    """Build a provider from a ``"<vendor>:<model>"`` string.

    ``"openai:gpt-5-mini"``, ``"gemini:gemini-2.5-pro"``, ``"anthropic:claude-opus-4-5"``,
    ``"mock"`` (built-in program) or ``"mock:replies.json"``. A bare vendor name uses
    that vendor's default model. Wrapped in the disk cache unless ``cache=False``.
    """
    vendor, _, model = spec.partition(":")
    vendor = vendor.strip().lower()
    if vendor not in _PROVIDERS:
        raise ValueError(f"unknown provider '{vendor}'; choose from "
                         f"{sorted(set(_PROVIDERS))}")
    cls = _PROVIDERS[vendor]
    model = model.strip()
    if vendor == "mock":
        replies = None
        if model:
            blob = json.loads(Path(model).read_text())
            replies = blob if isinstance(blob, list) else [blob]
        inner: LLMProvider = ScriptedProvider(replies=replies, **params)
    else:
        inner = cls(model or DEFAULT_MODELS[vendor], **params)  # type: ignore[arg-type]
    if not cache or vendor == "mock":
        return inner
    return CachingProvider(inner, cache_dir=cache_dir, enabled=True)


__all__ = ["AnthropicProvider", "CachingProvider", "GeminiProvider", "LLMProvider",
           "LLMResponse", "Message", "OpenAIProvider", "PRICES", "Part",
           "ScriptedProvider", "encode_png", "get_provider", "image_message",
           "load_env", "text_message"]
