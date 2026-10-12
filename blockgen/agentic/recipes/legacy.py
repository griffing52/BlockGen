"""Method ``legacy``: run BuildAgent itself inside the suite harness.

Why wrap instead of port: the live /gen server, the published scaffolding and
ont_* numbers, and the on-disk response cache all depend on BuildAgent's exact
request bytes. Running the real object guarantees them by construction, and the
suite still adds what BuildAgent lacks: a ledger with every call (plan included),
traces, scorers, contact sheets, and battery integration.

The arm config is ``AgentConfig(**{**spec_defaults, **ARMS[arm], **config})``.
``spec_defaults`` carry what the recipe owns (provider, canvas, gen, target
blocks, keep_history, cache), so a ``legacy-<arm>`` virtual recipe and
``experiments_agentic.run_arm`` build the same AgentConfig.

Limits, by design: BuildAgent cannot be interrupted mid-build, so the budget is
checked once before it starts, and its own textured ``_render`` does the
critique renders (the kit's renderer is only used for the suite's trace/scores).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from blockgen.agentic.core import (Attempt, Charge, EpisodeLike, Lowered, MinecraftWorld,
                                   billing_of, register)
from blockgen.agentic.providers import LLMProvider, LLMResponse, Message


class MeteredProvider(LLMProvider):
    """Pass-through provider that charges the episode ledger and traces each call.

    Wraps the (already cache-wrapped) kit provider, so the cache key, and the
    request bytes the inner provider sees, are untouched. Charging happens
    AFTER the call and never raises: BuildAgent cannot resume from an exception.
    """

    def __init__(self, inner: LLMProvider, ep: Any, role: str = "builder") -> None:
        super().__init__(inner.model, **getattr(inner, "params", {}))
        self.inner = inner
        self.name = getattr(inner, "name", "base")
        self.ep = ep
        self.role = role
        self.n = 0

    @property
    def billing(self) -> str:            # read by core.billing_of
        return billing_of(self.inner)

    def complete(self, messages: Sequence[Message], **overrides: Any) -> LLMResponse:
        resp = self.inner.complete(messages, **overrides)
        self.ep.ledger.charge(Charge.from_llm(self.role, resp, billing_of(self.inner)))
        self.ep.trace.request(self.role, list(messages), resp, f"legacy{self.n}")
        self.n += 1
        return resp

    def describe(self) -> str:
        return self.inner.describe()


def legacy_config(ep: Any, arm: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
    """The AgentConfig this episode runs (recipe defaults < ARMS[arm] < config)."""
    from blockgen.agentic.agent import AgentConfig

    spec = ep.spec
    gen = {"max_tokens": 16000, "temperature": None, "reasoning_effort": "medium",
           **(spec.gen or {})}
    defaults: Dict[str, Any] = dict(
        provider=spec.roles()["builder"], canvas_size=tuple(int(v) for v in spec.canvas),
        max_tokens=gen["max_tokens"], temperature=gen["temperature"],
        reasoning_effort=gen["reasoning_effort"], target_blocks=spec.target_blocks,
        keep_history=bool(spec.keep_history), cache=spec.cache != "off", verbose=False)
    arm_over: Dict[str, Any] = {}
    if arm is not None:
        from blockgen.experiments_agentic import ARMS
        if arm not in ARMS:
            raise KeyError(f"unknown legacy arm {arm!r}; arms: {sorted(ARMS)}")
        arm_over = dict(ARMS[arm])
    cfg_over = dict(config or {})
    if "canvas_size" in cfg_over:
        cfg_over["canvas_size"] = tuple(cfg_over["canvas_size"])
    return AgentConfig(**{**defaults, **arm_over, **cfg_over})


def adopt_result(ep: Any, res: Any, cfg: Any) -> None:
    """Turn a BuildResult into Episode attempts.

    Each Round becomes an Attempt; its world is rebuilt by replaying its program
    on a fresh MinecraftWorld, except the final accepted one, which IS
    BuildAgent's canvas (bit-identical blocks by construction). A round is
    rejected exactly when BuildAgent rolled it back: empty after a non-empty build.
    """
    size = tuple(int(v) for v in cfg.canvas_size)
    rounds = list(res.rounds)
    prev_blocks = 0
    flags: List[bool] = []
    for r in rounds:
        rejected = r.blocks == 0 and prev_blocks > 0
        flags.append(not rejected)
        if not rejected:
            prev_blocks = r.blocks
    last_acc = max((i for i, ok in enumerate(flags) if ok), default=None)
    for i, (r, ok) in enumerate(zip(rounds, flags)):
        if i == last_acc:
            world = MinecraftWorld(res.canvas)
        else:
            world = MinecraftWorld.empty(size)
            world.run(Lowered(source=r.program_text, program_text=r.program_text))
        patch = r.stage.startswith("critique") and cfg.critique_mode == "patch"
        att = Attempt(r.stage, r.program_text, r.report, r.response, r.blocks,
                      source_text=r.response.text, delta_text=r.response.text,
                      patch=patch, world=world)
        ep.adopt(att, accepted=ok, reason="ok" if ok else "empty after non-empty (legacy rollback)")


@register("method", "legacy")
def legacy(ep: EpisodeLike, *, arm: Optional[str] = None,
           config: Optional[Dict[str, Any]] = None) -> None:
    """BuildAgent unchanged (byte-identical requests), with suite traces and ledger."""
    from blockgen.agentic.agent import BuildAgent

    cfg = legacy_config(ep, arm, config)
    builder = ep.spec.roles()["builder"]
    if cfg.provider != builder and not getattr(ep.kit, "mock", False):
        # The kit's builder provider is what actually runs; a different
        # config.provider would be recorded in config.json but never called.
        raise ValueError(f"legacy: config.provider {cfg.provider!r} differs from the recipe's "
                         f"builder llm {builder!r}; set the recipe's `llm` instead")
    ep.ledger.check("llm")                     # once: BuildAgent cannot stop mid-build
    provider = MeteredProvider(ep.kit.llm("builder"), ep)
    res = BuildAgent(provider, cfg).build(ep.prompt, images=ep.reference_images,
                                          seed=getattr(ep, "variation", None))
    ep.config = cfg
    ep.plan = res.plan_text
    ep.transcript = list(res.transcript)
    adopt_result(ep, res, cfg)


__all__ = ["MeteredProvider", "adopt_result", "legacy", "legacy_config"]
