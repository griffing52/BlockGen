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

Methods-suite additions (``blockgen.agentic.core``; all additive, every legacy
cache key is unchanged):

* ``spends`` / ``billing`` on every provider, so the suite's spend gate
  (``BLOCKGEN_ALLOW_SPEND=1``) refuses a cache MISS on a paying or quota backend.
* cache modes ``off | read_write | read_only`` (``read_only`` raises
  :class:`core.CacheMiss` instead of calling out), atomic cache writes, ``peek``.
* vendors ``claude-code`` (the local Claude Code CLI, subscription quota),
  ``local`` / ``ollama`` (OpenAI-compatible servers) and ``mock:auto`` (an offline
  LLM that answers any suite turn plausibly); the CLI and mock classes live in
  :mod:`blockgen.agentic.providers_cli`.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = _REPO / "outputs" / "agentic_cache"

# Approximate list prices in USD per 1M tokens, (input, output). Used only for the
# cost line in run reports -- update freely, nothing depends on the numbers.
#
# A model that is NOT in this table is reported as *unpriced*, not as free: printing
# "$0.00" for a paid API call is worse than saying you do not know. Vendors ship
# models faster than this table is maintained, so unpriced is the normal case for
# anything new -- add a row when you care about the number.
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
    # Backend extras that are NOT part of the legacy record (e.g. the Claude Code
    # CLI's notional ``total_cost_usd``, which is quota, not money, so it must never
    # be summed into a cost). Kept out of to_dict() so legacy transcripts are unchanged.
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def priced(self) -> bool:
        """Whether :data:`PRICES` knows this model — see the note on that table."""
        return self.model in PRICES

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
                "priced": self.priced, "finish_reason": self.finish_reason}


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
    #: True if a call costs money or quota. The suite's caching wrapper refuses a
    #: cache MISS on such a backend unless BLOCKGEN_ALLOW_SPEND=1 (core.require_spend).
    spends = True
    #: 'api' | 'subscription' | 'local' | 'mock' (reported per charge in ledgers).
    billing = "api"

    def __init__(self, model: str, **params: Any):
        self.model = model
        self.params = params

    def complete(self, messages: Sequence[Message], **overrides: Any) -> LLMResponse:
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name}:{self.model}"


CACHE_MODES = ("off", "read_write", "read_only")   # mirrors core.CACHE_MODES


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a per-process temp file + ``os.replace``.

    Why: a crash (or a parallel arm) mid-write must never leave a truncated JSON
    that a later run reads as a cache entry. ``os.replace`` is atomic on POSIX, so
    readers see either nothing or the whole file.
    """
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


class CachingProvider(LLMProvider):
    """Wraps a provider with an on-disk response cache (see the module docstring).

    ``mode`` (suite addition): ``read_write`` (default, legacy behaviour),
    ``read_only`` (a miss raises ``core.CacheMiss``, so a re-run provably costs $0)
    or ``off`` (same as ``enabled=False``: pass-through). ``spend_gate``: a miss on
    an inner provider with ``spends=True`` calls ``core.require_spend`` first;
    this also gates the pass-through modes. Both default to legacy behaviour.
    """

    def __init__(self, inner: LLMProvider, cache_dir: Path | str = DEFAULT_CACHE_DIR,
                 enabled: bool = True, mode: str = "read_write", spend_gate: bool = False):
        super().__init__(inner.model, **inner.params)
        if mode not in CACHE_MODES:
            raise ValueError(f"cache mode must be one of {CACHE_MODES}, got {mode!r}")
        self.inner = inner
        self.name = inner.name
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled and mode != "off"
        self.mode = mode if self.enabled else "off"
        self.spend_gate = spend_gate

    # Delegated so a wrapped provider reports its backend's billing, not 'api'.
    @property
    def spends(self) -> bool:  # type: ignore[override]
        return bool(getattr(self.inner, "spends", True))

    @property
    def billing(self) -> str:  # type: ignore[override]
        return str(getattr(self.inner, "billing", "api"))

    def _path(self, messages: Sequence[Message], overrides: Dict[str, Any]) -> Path:
        return self.cache_dir / f"{self._key(messages, overrides)}.json"

    def peek(self, messages: Sequence[Message], **overrides: Any) -> bool:
        """Whether this request is already cached (``estimate`` prices hits as $0)."""
        return self.enabled and self._path(messages, overrides).is_file()

    def _miss(self, messages: Sequence[Message], overrides: Dict[str, Any]) -> LLMResponse:
        """Call the inner provider after the read-only and spend checks."""
        if self.mode == "read_only":
            from blockgen.agentic.core import CacheMiss
            raise CacheMiss(f"{self.describe()}: request not in cache {self.cache_dir} "
                            f"(cache mode read_only)")
        if self.spend_gate and self.spends:
            from blockgen.agentic.core import require_spend
            require_spend(self.describe())
        return self.inner.complete(messages, **overrides)

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
            return self._miss(messages, overrides)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(messages, overrides)
        if path.is_file():
            try:
                blob = json.loads(path.read_text())
                return LLMResponse(text=blob["text"], model=blob["model"],
                                   prompt_tokens=blob.get("prompt_tokens", 0),
                                   completion_tokens=blob.get("completion_tokens", 0),
                                   cached=True, latency_s=0.0,
                                   finish_reason=blob.get("finish_reason", ""),
                                   meta=dict(blob.get("meta") or {}))
            except (json.JSONDecodeError, KeyError):
                pass  # corrupt cache entry: fall through and re-request
        resp = self._miss(messages, overrides)
        record = {
            "text": resp.text, "model": resp.model,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "finish_reason": resp.finish_reason}
        if resp.meta:  # only new backends set meta; legacy entries stay byte-identical
            record["meta"] = resp.meta
        _atomic_write_text(path, json.dumps(record, indent=1, default=str))
        return resp

    def describe(self) -> str:
        return self.inner.describe()


#: HTTP statuses worth retrying: timeout, rate limit, 5xx, Anthropic "overloaded".
_TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504, 529})
#: Fallback for errors without a status (connection errors, timeouts, CLI text).
#: Codes are word-bounded so a 400 such as "requested 16500 tokens" is not retried.
_TRANSIENT_TEXT = re.compile(
    r"rate ?limit|timeout|timed out|overloaded|connection|\b(?:429|500|502|503|504)\b")


def _status_of(exc: BaseException) -> Optional[int]:
    """The HTTP status an SDK error carries (openai/anthropic ``status_code``,
    google ``code``, or ``response.status_code``), else None."""
    for obj, attr in ((exc, "status_code"), (exc, "code"),
                      (getattr(exc, "response", None), "status_code")):
        val = getattr(obj, attr, None)
        if isinstance(val, int) and not isinstance(val, bool) and 100 <= val < 600:
            return val
    return None


def _is_transient(exc: BaseException) -> bool:
    status = _status_of(exc)
    if status is not None:
        return status in _TRANSIENT_STATUS
    return bool(_TRANSIENT_TEXT.search(f"{type(exc).__name__}: {exc}".lower()))


def _retry(fn, attempts: int = 4, base_delay: float = 2.0):
    """Retry transient API failures (rate limits, 5xx) with exponential backoff."""
    last: Optional[Exception] = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - vendor SDKs raise their own types
            if not _is_transient(exc) or i == attempts - 1:
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

    def _extra_kwargs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Vendor-specific request fields; none for OpenAI itself (cache bytes unchanged)."""
        return {}

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
        kwargs.update(self._extra_kwargs(params))

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


class LocalProvider(OpenAIProvider):
    """A local OpenAI-compatible server (vLLM, llama.cpp, LM Studio, ...).

    ``base_url`` comes from the argument, then ``$LOCAL_LLM_URL``, then
    ``http://localhost:8000/v1``. The key is ``$LOCAL_LLM_KEY`` or the dummy
    ``'local'`` (most local servers ignore it, but the OpenAI client requires one).
    ``base_url`` is part of ``params`` and therefore of the cache key: two servers
    serving a model under the same name must not share cached replies.

    ``max_tokens`` defaults to ``DEFAULT_MAX_TOKENS`` (4096), not the hosted 16000:
    vLLM rejects (HTTP 400) any request whose prompt + max_tokens exceeds the
    served ``--max-model-len``, often 8-16k for a VL model on one 16 GB GPU. A
    recipe's ``gen.max_tokens`` still overrides it per call; if the server rejects
    the budget as over its context, the call is retried once with no cap (the
    server then fills the remaining context). Ollama's OpenAI endpoint cannot set
    ``num_ctx`` and silently truncates long prompts to the server default; raise
    it with ``OLLAMA_CONTEXT_LENGTH`` on the server.
    """

    name = "local"
    spends = False
    billing = "local"
    DEFAULT_URL = "http://localhost:8000/v1"
    URL_ENV = "LOCAL_LLM_URL"
    DEFAULT_MAX_TOKENS = 4096
    #: Substrings of a server error meaning prompt + max_tokens is over the context.
    _CONTEXT_ERRORS = ("maximum context length", "max_model_len", "context length",
                       "context window", "exceeds the model")

    def __init__(self, model: str = "default", *, base_url: Optional[str] = None,
                 **params: Any):
        load_env()
        url = base_url or os.environ.get(self.URL_ENV) or self.DEFAULT_URL
        # Local chat models are not OpenAI reasoning models: no reasoning_effort.
        params.setdefault("reasoning_effort", None)
        params.setdefault("max_tokens", self.DEFAULT_MAX_TOKENS)
        super().__init__(model, api_key_names=("LOCAL_LLM_KEY",), base_url=url, **params)
        self.params["base_url"] = url

    def _client_lazy(self):
        if self._client is None:
            from openai import OpenAI
            load_env()
            key = os.environ.get("LOCAL_LLM_KEY") or "local"
            self._client = OpenAI(api_key=key, base_url=self._base_url)
        return self._client

    @property
    def _is_reasoning(self) -> bool:
        return False   # local servers take max_tokens/temperature directly

    def _extra_kwargs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """``thinking=true|false`` -> the chat template's ``enable_thinking`` switch.

        Why: hybrid local models (Gemma 4, Qwen 3) think by default, and on a
        program-writing turn the thinking alone can exhaust the token budget and
        return no answer at all (measured on gemma-4-12b: 4000 tokens of reasoning,
        empty content). ``None`` (the default) sends nothing, leaving the server's
        template default; vLLM and llama.cpp accept ``chat_template_kwargs``.
        """
        thinking = params.get("thinking")
        if thinking is None:
            return {}
        if isinstance(thinking, str):
            thinking = thinking.strip().lower() in ("1", "true", "yes", "on")
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}}

    #: What replaces an image when ``vision=false``.
    IMAGE_PLACEHOLDER = "[image omitted: this model cannot see images]"

    def _strip_images(self, messages: Sequence[Message]) -> List[Message]:
        """Every image part -> one text placeholder (``vision=false``).

        Why: a text-only server (vLLM started with images disabled, or a text
        model) rejects any request with an image (HTTP 400), and dream, critique
        and room turns all attach renders. Stripping here, not in each method,
        lets every method run blind on such a server; the placeholder keeps the
        surrounding "TARGET:" / "YOURS:" labels from reading as dangling.
        """
        out: List[Message] = []
        for m in messages:
            parts = [p if p.kind != "image" else Part("text", text=self.IMAGE_PLACEHOLDER)
                     for p in m.parts]
            out.append(Message(role=m.role, parts=parts))
        return out

    #: What replaces an older image dropped by ``max_images``.
    OLD_IMAGE_PLACEHOLDER = "[earlier image omitted: only the most recent images are shown]"

    def _cap_images(self, messages: Sequence[Message], keep: int) -> List[Message]:
        """Keep the LAST ``keep`` images, replacing earlier ones with a placeholder.

        Why: vLLM enforces ``--limit-mm-per-prompt`` per request (HTTP 400 above
        it), and a match loop with history re-sends every earlier TARGET/YOURS
        pair, so turn 2 or 3 crosses a small limit. The newest images are the ones
        the current turn is about.
        """
        total = sum(p.kind == "image" for m in messages for p in m.parts)
        drop = max(0, total - keep)
        out: List[Message] = []
        for m in messages:
            parts = []
            for p in m.parts:
                if p.kind == "image" and drop > 0:
                    parts.append(Part("text", text=self.OLD_IMAGE_PLACEHOLDER))
                    drop -= 1
                else:
                    parts.append(p)
            out.append(Message(role=m.role, parts=parts))
        return out

    def complete(self, messages: Sequence[Message], **overrides: Any) -> LLMResponse:
        cap = {**self.params, **overrides}.get("max_images")
        if cap is not None:
            messages = self._cap_images(messages, int(cap))
        vision = {**self.params, **overrides}.get("vision")
        if isinstance(vision, str):
            vision = vision.strip().lower() not in ("0", "false", "no", "off")
        if vision is False:
            messages = self._strip_images(messages)
        try:
            return super().complete(messages, **overrides)
        except Exception as exc:  # noqa: BLE001 - SDK error types vary by server
            message = str(exc).lower()
            if not any(s in message for s in self._CONTEXT_ERRORS):
                raise
            # The budget does not fit the served context: let the server cap it.
            return super().complete(messages, **{**overrides, "max_tokens": None})


class OllamaProvider(LocalProvider):
    """Ollama's OpenAI-compatible endpoint (``$OLLAMA_URL`` or localhost:11434)."""

    name = "ollama"
    DEFAULT_URL = "http://localhost:11434/v1"
    URL_ENV = "OLLAMA_URL"


class ScriptedProvider(LLMProvider):
    """Returns canned responses in order — the offline stand-in for tests and demos.

    ``mock:path/to/replies.json`` loads a JSON list of strings; ``mock:`` with no
    path replays a small built-in house program. Every test in
    ``tests/test_agentic_*.py`` runs the *real* agent loop through this, so the
    pipeline is covered without a network call or an API key.
    """

    name = "mock"
    spends = False
    billing = "mock"

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
    # Suite vendors. String targets are imported on first use: providers_cli
    # imports this module, so it cannot be imported at the top of it.
    "claude-code": "blockgen.agentic.providers_cli:ClaudeCodeProvider",
    "cc": "blockgen.agentic.providers_cli:ClaudeCodeProvider",
    "local": LocalProvider,
    "ollama": OllamaProvider,
}

DEFAULT_MODELS = {
    "openai": "gpt-5-mini",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-4-5",
    "mock": "scripted",
    "claude-code": "sonnet",
    "cc": "sonnet",            # alias of claude-code: a bare `cc` must resolve too
    "local": "default",
    "ollama": "qwen2.5vl:7b",
}
#: The spec ``mock:auto``: an offline LLM that answers any suite turn (see
#: providers_cli.AutoMockProvider). Not a file path, unlike ``mock:<replies.json>``.
AUTO_MOCK_SPEC = "mock:auto"


def _provider_class(vendor: str) -> Any:
    cls = _PROVIDERS[vendor]
    if isinstance(cls, str):
        import importlib
        mod, _, attr = cls.partition(":")
        cls = getattr(importlib.import_module(mod), attr)
    return cls


def get_provider(spec: str, *, cache: bool = True,
                 cache_dir: Path | str = DEFAULT_CACHE_DIR,
                 cache_mode: Optional[str] = None, spend_gate: bool = False,
                 **params: Any) -> LLMProvider:
    """Build a provider from a ``"<vendor>:<model>"`` string.

    ``"openai:gpt-5-mini"``, ``"gemini:gemini-2.5-pro"``, ``"anthropic:claude-opus-4-5"``,
    ``"mock"`` (built-in program) or ``"mock:replies.json"``. A bare vendor name uses
    that vendor's default model. Wrapped in the disk cache unless ``cache=False``.

    Suite additions: ``"claude-code:haiku"`` (alias ``cc``), ``"local:<model>"``,
    ``"ollama:<model>"`` and ``"mock:auto"``. ``cache_mode`` (``off | read_write |
    read_only``) overrides ``cache`` when given; ``spend_gate=True`` makes a cache
    miss on a paying/quota backend require BLOCKGEN_ALLOW_SPEND=1, including with
    the cache off. The defaults reproduce the legacy behaviour exactly.
    """
    vendor, _, model = spec.partition(":")
    vendor = vendor.strip().lower()
    if vendor not in _PROVIDERS:
        raise ValueError(f"unknown provider '{vendor}'; choose from "
                         f"{sorted(set(_PROVIDERS))}")
    model = model.strip()
    mode = cache_mode if cache_mode is not None else ("read_write" if cache else "off")
    if mode not in CACHE_MODES:
        raise ValueError(f"cache mode must be one of {CACHE_MODES}, got {mode!r}")
    if vendor == "mock":
        if model == "auto":
            from blockgen.agentic.providers_cli import AutoMockProvider
            return AutoMockProvider(**params)
        replies = None
        if model:
            blob = json.loads(Path(model).read_text())
            replies = blob if isinstance(blob, list) else [blob]
        return ScriptedProvider(replies=replies, **params)
    cls = _provider_class(vendor)
    inner: LLMProvider = cls(model or DEFAULT_MODELS[vendor], **params)
    if mode == "off":
        # Legacy: the bare provider. With the gate on, a pass-through wrapper so
        # every call (all of them are misses) is still checked.
        if spend_gate and getattr(inner, "spends", True):
            return CachingProvider(inner, cache_dir=cache_dir, enabled=False,
                                   spend_gate=True)
        return inner
    return CachingProvider(inner, cache_dir=cache_dir, enabled=True, mode=mode,
                           spend_gate=spend_gate)


__all__ = ["AUTO_MOCK_SPEC", "AnthropicProvider", "CACHE_MODES", "CachingProvider",
           "GeminiProvider", "LLMProvider", "LLMResponse", "LocalProvider", "Message",
           "OllamaProvider", "OpenAIProvider", "PRICES", "Part", "ScriptedProvider",
           "encode_png", "get_provider", "image_message", "load_env", "text_message"]
