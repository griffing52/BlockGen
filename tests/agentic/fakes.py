"""Offline fakes for every suite axis, shared by all tests under tests/agentic.

Each fake satisfies a ``core`` protocol with no network, no GPU and no model
weights, and is deterministic, so a test can assert exact values. They exist so
the work packages can test against each other before the real modules land:
FakeActions stands in for actions.py, FakeRenderer for render.py, FakeImageGen for
imagegen.py, fake_scorer for scoring.py, and so on.

Import them RELATIVELY from a test module in this directory::

    from .fakes import ScriptedLLM, make_fake_kit

(pytest imports this directory as the package ``agentic``; importing it a second
way, as ``tests.agentic.fakes``, creates a second copy of every class.
:func:`register_fakes` tolerates that, but ``isinstance`` across copies does not.)

Registry names (after :func:`register_fakes`, which conftest.py calls)::

    actions 'fake_dsl'      renderer 'fake_renderer'   image 'fake_image'
    scorer  'fake_scorer'   context  'fake_context'    accept 'fake_accept'
    method  'fake_method'
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from blockgen.agentic import core
from blockgen.agentic.core import (Attempt, ContextParts, DreamSpec, DreamVersion,
                                   EmptyBuild, ImageRequest, ImageResult, Kit, Ledger,
                                   Lowered, MethodResult, MethodSpec, MinecraftWorld,
                                   PromptCtx, ScoreTerm, View)
from blockgen.agentic.providers import (_DEMO_PROGRAM, LLMProvider, LLMResponse,
                                        Message, encode_png)
from blockgen.utils.data import Structure

#: A tiny valid DSL program (a 5x3x5 hut), for tests that want few blocks.
HUT_PROGRAM = "fill cobblestone 0 0 0 4 0 4\nwalls oak_planks 0 1 0 4 2 4\n"
DEMO_PROGRAM = _DEMO_PROGRAM


# ===========================================================================
# LLM
# ===========================================================================
Reply = Union[str, Callable[[Sequence[Message]], str]]


class ScriptedLLM(LLMProvider):
    """Replies from a script: ``rules`` first, then the ``replies`` queue.

    ``rules`` are ``(substring, reply)`` pairs matched against the LAST message's
    text, which lets one provider serve several roles (e.g. answer any turn that
    ends with the JSON rule with a verdict, and everything else with programs).
    The queue repeats its last entry once exhausted. A reply may be a callable
    ``messages -> str``. Every call is recorded in ``calls``.

    ``model`` defaults to the unpriced 'scripted'; pass a PRICES model (e.g.
    'gpt-5-mini') with ``prompt_tokens``/``completion_tokens`` to test ledgers.
    """

    name = "scripted"
    spends = False
    billing = "mock"

    def __init__(self, replies: Optional[Sequence[Reply]] = None, *, model: str = "scripted",
                 rules: Sequence[Tuple[str, Reply]] = (), prompt_tokens: int = 0,
                 completion_tokens: int = 0, cached: bool = False, **params: Any) -> None:
        super().__init__(model, **params)
        self.replies: List[Reply] = list(replies) if replies else [DEMO_PROGRAM]
        self.rules = list(rules)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cached = cached
        self.calls: List[List[Message]] = []
        self.overrides: List[Dict[str, Any]] = []
        self._i = 0

    def _pick(self, messages: Sequence[Message]) -> Reply:
        last = messages[-1].text if messages else ""
        for needle, reply in self.rules:
            if needle in last:
                return reply
        reply = self.replies[min(self._i, len(self.replies) - 1)]
        self._i += 1
        return reply

    def complete(self, messages: Sequence[Message], **overrides: Any) -> LLMResponse:
        self.calls.append(list(messages))
        self.overrides.append(dict(overrides))
        reply = self._pick(messages)
        text = reply(messages) if callable(reply) else reply
        return LLMResponse(text=text, model=self.model, prompt_tokens=self.prompt_tokens,
                           completion_tokens=self.completion_tokens, cached=self.cached)

    @property
    def n_calls(self) -> int:
        return len(self.calls)


# ===========================================================================
# Actions
# ===========================================================================
class FakeActions:
    """Identity lowering: the model text IS the DSL. Execution is still real
    (MinecraftWorld.run -> dsl.run_program), so blocks and reports are genuine."""

    name = "fake_dsl"
    patchable = True
    OUTPUT_RULE = "Reply with only the fake program."

    def __init__(self, reply: str = DEMO_PROGRAM) -> None:
        self.reply = reply

    def instructions(self, ctx: PromptCtx) -> str:
        w, h, d = ctx.size
        return f"FAKE GRAMMAR. Canvas {w}x{h}x{d}. Palette: {', '.join(list(ctx.palette)[:5])}."

    def output_rule(self) -> str:
        return self.OUTPUT_RULE

    def examples(self, query: str, k: int) -> List[Tuple[str, str]]:
        return [("a small hut", HUT_PROGRAM)][:max(0, k)]

    def lower(self, text: str) -> Lowered:
        return Lowered(source=text, program_text=text, issues=[],
                       n_source_lines=len(text.splitlines()))

    def mock_reply(self) -> str:
        return self.reply


# ===========================================================================
# Renderer
# ===========================================================================
def _colour(*parts: Any) -> Tuple[int, int, int]:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return (h[0] // 2 + 64, h[1] // 2 + 64, h[2] // 2 + 64)   # never white


class FakeRenderer:
    """Deterministic image per (view, block count): a coloured square on white
    whose side grows with the block count, so pixel/silhouette-style scorers can
    tell builds apart. Applies View.apply (cutaways change the count) and raises
    EmptyBuild like the real renderers. ``px`` overrides the views' px (fast)."""

    name = "fake_renderer"

    def __init__(self, px: Optional[int] = 64) -> None:
        self.px = px
        self.calls: List[Tuple[int, Tuple[str, ...]]] = []

    def render(self, structure: Structure, views: Sequence[View]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        names = []
        for v in views:
            v = core.get_view(v)
            s = v.apply(structure)
            n = int(s.occupied_mask.sum())
            if n == 0:
                raise EmptyBuild(f"empty structure for view {v.name}")
            px = int(self.px or v.px)
            img = np.full((px, px, 3), 255, dtype=np.uint8)
            side = max(1, int(round(px * min(1.0, 0.1 + n / 600.0))))
            lo = (px - side) // 2
            img[lo:lo + side, lo:lo + side] = _colour(v.name, n)
            out.append(img)
            names.append(v.name)
        self.calls.append((int(structure.occupied_mask.sum()), tuple(names)))
        return out


# ===========================================================================
# Image generation
# ===========================================================================
def solid_png(rgb: Tuple[int, int, int], px: int = 32) -> bytes:
    return encode_png(np.full((px, px, 3), rgb, dtype=np.uint8))


class FakeImageGen:
    """Solid-colour PNG keyed by (prompt, seed, refs). Priced at ``usd`` per image
    (so ledger caps can be exercised) but ``spends=False`` (never gated)."""

    name = "fake_image"
    spends = False
    billing = "mock"

    def __init__(self, model: str = "fake-solid", *, usd: float = 0.01, px: int = 32,
                 refs: bool = True) -> None:
        self.model = model
        self.usd = usd
        self.px = px
        self._refs = refs
        self.calls = 0
        self.requests: List[ImageRequest] = []

    def generate(self, req: ImageRequest) -> ImageResult:
        self.calls += 1
        self.requests.append(req)
        refs = [hashlib.sha256(r).hexdigest()[:8] for r in req.refs]
        imgs = [solid_png(_colour(req.prompt, req.seed, refs, i), self.px) for i in range(req.n)]
        return ImageResult(images=imgs, model=self.model, cost_usd=self.usd * req.n, priced=True)

    def estimate_usd(self, req: ImageRequest) -> Optional[float]:
        return self.usd * req.n

    def supports_refs(self) -> bool:
        return self._refs

    def describe(self) -> str:
        return f"{self.name}:{self.model}"


# ===========================================================================
# Function kinds
# ===========================================================================
def fake_scorer(ep: Any, att: Attempt, *, scale: float = 500.0) -> ScoreTerm:
    """Bigger is better: min(1, blocks / scale). Needs neither dreams nor renders."""
    return ScoreTerm(value=min(1.0, att.blocks / float(scale)), note=f"{att.blocks} blocks")


def fake_context(ep: Any, *, text: str = "FAKE CONTEXT") -> ContextParts:
    """A fixed system-prompt line (assert it shows up in the system message)."""
    return ContextParts(system_text=text)


def fake_accept(prev: Optional[Attempt], cand: Attempt) -> Tuple[bool, str]:
    """Legacy semantics: reject only an empty build that would replace a non-empty one."""
    if cand.blocks == 0 and prev is not None and prev.blocks > 0:
        return False, "empty after non-empty"
    return True, "ok"


def fake_method(ep: Any, *, n: int = 1) -> None:
    """``n`` rounds of ask-then-try, using only EpisodeLike verbs."""
    for i in range(n):
        stage = "generate" if i == 0 else f"retry{i}"
        text = ep.ask(f"Build: {ep.prompt}", stage=stage)
        ep.try_(stage, text)


# ===========================================================================
# Trace
# ===========================================================================
class RecordingTrace:
    """In-memory TraceLike: everything a method traced, for assertions."""

    root = None

    def __init__(self) -> None:
        self.assets: Dict[str, bytes] = {}
        self.events: List[Tuple[str, Dict[str, Any]]] = []
        self.requests: List[Dict[str, Any]] = []
        self.dreams: List[DreamVersion] = []
        self.attempts: List[Attempt] = []
        self.result: Optional[MethodResult] = None

    def asset(self, png: bytes) -> str:
        key = f"assets/{hashlib.sha256(png).hexdigest()[:16]}.png"
        self.assets[key] = png
        return key

    def event(self, kind: str, **data: Any) -> None:
        self.events.append((kind, data))

    def request(self, role: str, messages: Sequence[Message], response: LLMResponse,
                stage: str) -> None:
        self.requests.append({"role": role, "messages": list(messages),
                              "response": response, "stage": stage})

    def dream(self, d: DreamVersion) -> None:
        self.dreams.append(d)

    def attempt(self, a: Attempt) -> None:
        self.attempts.append(a)

    def close(self, result: MethodResult) -> None:
        self.result = result

    def kinds(self) -> List[str]:
        return [k for k, _ in self.events]


# ===========================================================================
# Registration and kit
# ===========================================================================
FAKE_NAMES: Dict[str, Tuple[str, Any]] = {
    "fake_dsl": ("actions", FakeActions),
    "fake_renderer": ("renderer", FakeRenderer),
    "fake_image": ("image", FakeImageGen),
    "fake_scorer": ("scorer", fake_scorer),
    "fake_context": ("context", fake_context),
    "fake_accept": ("accept", fake_accept),
    "fake_method": ("method", fake_method),
}

#: Dream specs for tests (dreams.DREAMS may not exist yet).
FAKE_DREAMS: Dict[str, DreamSpec] = {
    "iso": DreamSpec("iso", "{style} {desc}", view="iso"),
    "back": DreamSpec("back", "{style} {desc}, from behind", view="iso_back", weight=0.5,
                      condition_on=("iso",)),
    "mood": DreamSpec("mood", "{style} {desc}, evening", view=None),
}


def register_fakes() -> None:
    """Register every fake under its ``fake_*`` name. Idempotent, also across a
    second import of this module under another package name (see module doc)."""
    for name, (kind, obj) in FAKE_NAMES.items():
        if name in core._REG[kind]:
            continue
        core.register(kind, name, meta_fake=True,
                      **({"needs_dream": False, "needs_renders": False}
                         if kind == "scorer" else {}))(obj)


def fake_spec(**fields: Any) -> MethodSpec:
    """A MethodSpec naming the fakes (so provenance/hash work), fields overridable."""
    base: Dict[str, Any] = dict(
        name="fake", method="fake_method", llm="scripted", actions="fake_dsl",
        contexts=["fake_context"], renderer="fake_renderer",
        score={"fake_scorer": 1.0}, accept="fake_accept", canvas=(32, 32, 32),
        cache="off")
    base.update(fields)
    return MethodSpec(**base)


_KIT_FIELDS = {f.name for f in dataclasses.fields(Kit)}


def make_fake_kit(*, replies: Optional[Sequence[Reply]] = None,
                  llm: Optional[LLMProvider] = None, spec_kw: Optional[Dict[str, Any]] = None,
                  **overrides: Any) -> Kit:
    """A complete offline Kit built from the fakes (no registry lookups needed).

    ``replies``/``llm`` set the provider for EVERY role. ``spec_kw`` overrides
    MethodSpec fields. Any other keyword replaces the Kit field of that name
    (``image=None``, ``dreams=[FAKE_DREAMS['iso']]``, ``method=my_fn``, ...); a
    ``dreams`` list may name FAKE_DREAMS keys. ``method`` may be a plain function
    (it is bound with the spec's params).
    """
    register_fakes()
    bad = sorted(set(overrides) - _KIT_FIELDS)
    if bad:
        raise TypeError(f"make_fake_kit: unknown Kit field(s) {bad}; valid: {sorted(_KIT_FIELDS)}")
    spec = overrides.pop("spec", None) or fake_spec(**(spec_kw or {}))
    provider = llm or ScriptedLLM(replies)
    dreams = [FAKE_DREAMS[d] if isinstance(d, str) else d
              for d in overrides.pop("dreams", [])]
    method = overrides.pop("method", fake_method)
    if not isinstance(method, functools.partial):
        method = functools.partial(method, **spec.params)
    kit = Kit(
        spec=spec,
        llms={r: provider for r in core.ROLES},
        image=FakeImageGen(),
        actions=FakeActions(),
        renderer=FakeRenderer(),
        contexts=[("fake_context", fake_context)],
        scorers={"fake_scorer": (fake_scorer, 1.0)},
        final_scorers={},
        accept=fake_accept,
        method=method,
        dreams=dreams,
        ledger=Ledger(name=spec.name),
        world_factory=MinecraftWorld.empty,
    )
    for k, v in overrides.items():
        setattr(kit, k, v)
    return kit


def make_attempt(program: str = HUT_PROGRAM, *, idx: int = 0, stage: str = "generate",
                 size: Tuple[int, int, int] = (32, 32, 32), **fields: Any) -> Attempt:
    """An executed Attempt without an Episode (scorer / accept / trace tests)."""
    world = MinecraftWorld.empty(size)
    report = world.run(Lowered(source=program, program_text=program))
    resp = LLMResponse(text=program, model="scripted")
    return Attempt(stage, program, report, resp, world.block_count(), idx=idx,
                   source_text=program, delta_text=program, world=world, **fields)


__all__ = ["DEMO_PROGRAM", "FAKE_DREAMS", "FAKE_NAMES", "FakeActions", "FakeImageGen",
           "FakeRenderer", "HUT_PROGRAM", "RecordingTrace", "ScriptedLLM", "fake_accept",
           "fake_context", "fake_method", "fake_scorer", "fake_spec", "make_attempt",
           "make_fake_kit", "register_fakes", "solid_png"]
