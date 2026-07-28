"""The build agent: text (or image) -> program -> structure, with feedback loops.

This is the control loop for the agentic track. It is deliberately thin — the
language lives in ``dsl.py``, the wording in ``prompts.py``, the vendor in
``providers.py`` — because the interesting variable is *which loop you run*, and
every stage here is independently switchable so the experiment battery can ablate
them:

    plan?  ->  generate  ->  execute  ->  repair(N)?  ->  critique(M)?

``plan``      a natural-language design pass before any code (does thinking first help?)
``examples``  k in-context example programs (does one demonstration fix grounding?)
``repair``    re-prompt with the executor's error/no-op report (symbolic feedback)
``critique``  re-prompt with a *render* of the build (visual feedback, multimodal)

Every stage's request and response is recorded in the transcript, and the
structure after every round is kept, so a run can be replayed, diffed, or turned
into a growth video without re-querying the model.

The loop is failure-tolerant by construction: a program that half-parses still
produces a build, and the parts that failed become the next round's prompt. The
only unrecoverable outcome is an empty canvas, which is reported rather than
raised — a sweep over 50 prompts must not die on one of them.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from blockgen.agentic.canvas import Canvas, DEFAULT_SIZE
from blockgen.agentic.dsl import ExecutionReport, Program, execute, parse_program
from blockgen.agentic.examples import EXAMPLES, Example, select_examples
from blockgen.agentic.prompts import (build_prompt, critique_prompt, plan_prompt,
                                      repair_prompt, scale_hint, system_prompt)
from blockgen.agentic.providers import (LLMProvider, LLMResponse, Message,
                                        encode_png, text_message)
from blockgen.utils.data import Structure

# Rendering the build for visual critique must never take the run down: a headless
# box without EGL, a degenerate empty structure, a driver hiccup -- all of these
# degrade to "no critique this round" rather than an exception.
CRITIQUE_VIEWS = ((45.0, 25.0), (225.0, 25.0))


@dataclass
class AgentConfig:
    """Every knob of the loop. Serialized verbatim into each run's ``config.json``."""

    provider: str = "openai:gpt-5-mini"
    canvas_size: Tuple[int, int, int] = DEFAULT_SIZE
    plan: bool = False
    n_examples: int = 0
    repair_rounds: int = 1
    critique_rounds: int = 0
    critique_mode: str = "rewrite"        # rewrite | patch
    keep_history: bool = True             # show the model its own previous programs
    target_blocks: Optional[int] = None   # optional size nudge in the system prompt
    max_tokens: int = 16000
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = "medium"
    critique_px: int = 384
    cache: bool = True
    verbose: bool = True

    def provider_params(self) -> Dict[str, Any]:
        return {"max_tokens": self.max_tokens, "temperature": self.temperature,
                "reasoning_effort": self.reasoning_effort}

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["canvas_size"] = list(self.canvas_size)
        return d


@dataclass
class Round:
    """One generate/repair/critique iteration and what it produced."""

    stage: str
    program_text: str
    report: ExecutionReport
    response: LLMResponse
    blocks: int


@dataclass
class BuildResult:
    """Everything one build produced — the unit the battery aggregates over."""

    description: str
    structure: Structure
    canvas: Canvas
    program_text: str
    report: ExecutionReport
    rounds: List[Round] = field(default_factory=list)
    plan_text: str = ""
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    config: Optional[AgentConfig] = None
    elapsed_s: float = 0.0
    error: str = ""

    # --- aggregates -------------------------------------------------------
    @property
    def blocks(self) -> int:
        return int(self.structure.occupied_mask.sum())

    @property
    def cost_usd(self) -> float:
        return sum(r.response.cost_usd for r in self.rounds)

    @property
    def tokens(self) -> Tuple[int, int]:
        return (sum(r.response.prompt_tokens for r in self.rounds),
                sum(r.response.completion_tokens for r in self.rounds))

    @property
    def cached(self) -> bool:
        return bool(self.rounds) and all(r.response.cached for r in self.rounds)

    def metrics(self, connectivity: bool = True) -> Dict[str, Any]:
        """Per-build metrics, shared with the rest of the repo's eval vocabulary.

        ``n_components``/``largest_component_frac`` are the same connectivity
        notion ``blockgen/eval/validity.py`` uses for the neural tracks, so an
        agentic build and an AR sample are judged the same way. The component
        scan is skipped above 40k blocks (pure-python flood fill) — that is a
        speed guard, not a semantic one, and it is reported as ``null``.
        """
        prompt_tokens, completion_tokens = self.tokens
        shape = self.structure.shape if self.blocks else (0, 0, 0)
        out: Dict[str, Any] = {
            "description": self.description,
            "blocks": self.blocks,
            "shape": list(shape),
            "volume_fill": round(self.blocks / max(1, int(shape[0]) * int(shape[1])
                                                   * int(shape[2])), 4) if self.blocks else 0.0,
            "n_block_types": len(self.canvas.palette_counts()),
            "n_rounds": len(self.rounds),
            "n_commands": self.report.n_commands,
            "n_failed_commands": self.report.n_failed,
            "n_noop_commands": self.report.n_noop,
            "command_success_rate": round(
                1.0 - self.report.n_failed / max(1, self.report.n_commands), 4),
            "blocks_per_command": round(self.blocks / max(1, self.report.n_commands), 2),
            "clipped_writes": self.report.clipped_writes,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "elapsed_s": round(self.elapsed_s, 2),
            "cached": self.cached,
            "error": self.error,
        }
        out["n_components"] = None
        out["largest_component_frac"] = None
        if connectivity and 0 < self.blocks <= 40000:
            from blockgen.eval.validity import _components
            comps = _components(self.structure.crop_to_non_air().occupied_mask)
            out["n_components"] = len(comps)
            out["largest_component_frac"] = round(len(comps[0]) / self.blocks, 4) if comps else 0.0
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "plan": self.plan_text,
            "program": self.program_text,
            "metrics": self.metrics(),
            "execution": self.report.to_dict(),
            "rounds": [{"stage": r.stage, "blocks": r.blocks,
                        "report": r.report.to_dict(), "usage": r.response.to_dict()}
                       for r in self.rounds],
            "config": self.config.to_dict() if self.config else {},
        }


class BuildAgent:
    """Runs the loop for one provider + config. Reusable across many prompts."""

    def __init__(self, provider: LLMProvider, config: Optional[AgentConfig] = None,
                 examples: Sequence[Example] = tuple(EXAMPLES)):
        self.provider = provider
        self.config = config or AgentConfig()
        self.example_pool = list(examples)

    # --- helpers ----------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(f"  [agent] {msg}", flush=True)

    def _ask(self, messages: Sequence[Message]) -> LLMResponse:
        resp = self.provider.complete(messages, **self.config.provider_params())
        tag = " (cached)" if resp.cached else ""
        self._log(f"{self.provider.describe()} -> {len(resp.text)} chars, "
                  f"{resp.completion_tokens} completion tokens{tag}")
        return resp

    def _system(self) -> Message:
        return system_prompt(tuple(self.config.canvas_size),
                             extra=scale_hint(self.config.target_blocks))

    def _render(self, structure: Structure) -> List[bytes]:
        """Textured renders for visual critique; empty list if rendering is
        unavailable (headless without EGL) or the build is empty."""
        if int(structure.occupied_mask.sum()) == 0:
            return []
        try:
            from blockgen.renderer.textured import render_structure
            from blockgen.renderer.textures import load_face_textures
            tex = load_face_textures()
            return [encode_png(render_structure(structure, px=self.config.critique_px,
                                                azim_deg=az, elev_deg=el, bg=(1, 1, 1),
                                                face_textures=tex))
                    for az, el in CRITIQUE_VIEWS]
        except Exception as exc:  # noqa: BLE001 - critique is optional by design
            self._log(f"render for critique unavailable ({type(exc).__name__}: {exc})")
            return []

    @staticmethod
    def _run_program(text: str, canvas: Optional[Canvas], size) -> Tuple[
            Canvas, ExecutionReport, Program]:
        program = parse_program(text)
        canvas, report = execute(program, canvas=canvas, size=size)
        return canvas, report, program

    # --- the loop ---------------------------------------------------------
    def build(self, description: str, images: Sequence[bytes] = (),
              image_note: str = "") -> BuildResult:
        """Generate one structure for ``description`` (optionally image-conditioned)."""
        cfg = self.config
        t0 = time.time()
        transcript: List[Dict[str, Any]] = []
        rounds: List[Round] = []
        messages: List[Message] = [self._system()]

        # --- stage 1: plan ------------------------------------------------
        plan_text = ""
        if cfg.plan:
            self._log("planning")
            ask = plan_prompt(description, tuple(cfg.canvas_size))
            request = [messages[0], ask]
            resp = self._ask(request)
            plan_text = resp.text.strip()
            transcript.append({"stage": "plan", "request": ask.text,
                               "response": plan_text, "usage": resp.to_dict()})

        # --- stage 2: generate --------------------------------------------
        examples = select_examples(description, cfg.n_examples, self.example_pool)
        gen_msgs = build_prompt(description, plan=plan_text or None, examples=examples,
                                images=images, image_note=image_note)
        messages.extend(gen_msgs)
        self._log(f"generating (plan={bool(plan_text)}, examples={len(examples)}, "
                  f"images={len(images)})")
        resp = self._ask(messages)
        transcript.append({"stage": "generate",
                           "request": gen_msgs[-1].text, "response": resp.text,
                           "usage": resp.to_dict()})
        program_text = resp.text
        canvas, report, _ = self._run_program(program_text, None, cfg.canvas_size)
        rounds.append(Round("generate", program_text, report, resp, canvas.block_count()))
        self._log(f"executed: {canvas.block_count()} blocks, {report.n_failed} failed, "
                  f"{report.n_noop} no-op commands")
        if cfg.keep_history:
            messages.append(text_message("assistant", program_text))

        # --- stage 3: symbolic repair -------------------------------------
        for i in range(cfg.repair_rounds):
            needs_repair = report.n_failed or report.n_noop or canvas.block_count() == 0
            if not needs_repair:
                self._log("nothing to repair")
                break
            self._log(f"repair round {i + 1}/{cfg.repair_rounds}")
            ask = repair_prompt(report, description=description)
            request = (messages + [ask]) if cfg.keep_history else \
                [self._system(), text_message("user", f"Build: {description}"),
                 text_message("assistant", program_text), ask]
            resp = self._ask(request)
            transcript.append({"stage": f"repair{i + 1}", "request": ask.text,
                               "response": resp.text, "usage": resp.to_dict()})
            new_text = resp.text
            new_canvas, new_report, _ = self._run_program(new_text, None, cfg.canvas_size)
            rounds.append(Round(f"repair{i + 1}", new_text, new_report, resp,
                                new_canvas.block_count()))
            # Keep the repair only if it did not make things worse: an empty or
            # emptier build after a "fix" is a regression, and silently accepting
            # it is how a loop turns a good build into nothing.
            if new_canvas.block_count() == 0 and canvas.block_count() > 0:
                self._log("repair produced an empty build — keeping the previous one")
                break
            canvas, report, program_text = new_canvas, new_report, new_text
            if cfg.keep_history:
                messages += [ask, text_message("assistant", program_text)]

        # --- stage 4: visual critique -------------------------------------
        for i in range(cfg.critique_rounds):
            structure = canvas.to_structure()
            views = self._render(structure)
            if not views:
                self._log("skipping critique (no render available)")
                break
            self._log(f"critique round {i + 1}/{cfg.critique_rounds} "
                      f"({cfg.critique_mode})")
            ask = critique_prompt(description, report, views,
                                  views=[f"{int(a)}° azimuth" for a, _ in CRITIQUE_VIEWS],
                                  mode=cfg.critique_mode)
            request = (messages + [ask]) if cfg.keep_history else [self._system(), ask]
            resp = self._ask(request)
            transcript.append({"stage": f"critique{i + 1}", "request": ask.text,
                               "response": resp.text, "usage": resp.to_dict()})
            if cfg.critique_mode == "patch":
                patched = canvas.clone()
                patched, patch_report, _ = self._run_program(resp.text, patched,
                                                             cfg.canvas_size)
                new_canvas, new_report, new_text = (
                    patched, patch_report, program_text + "\n# --- patch ---\n" + resp.text)
            else:
                new_canvas, new_report, _ = self._run_program(resp.text, None,
                                                              cfg.canvas_size)
                new_text = resp.text
            rounds.append(Round(f"critique{i + 1}", new_text, new_report, resp,
                                new_canvas.block_count()))
            if new_canvas.block_count() == 0 and canvas.block_count() > 0:
                self._log("critique produced an empty build — keeping the previous one")
                break
            canvas, report, program_text = new_canvas, new_report, new_text
            if cfg.keep_history:
                messages += [ask, text_message("assistant", resp.text)]

        structure = canvas.to_structure()
        error = "" if canvas.block_count() else "empty build (no blocks placed)"
        result = BuildResult(description=description, structure=structure, canvas=canvas,
                             program_text=program_text, report=report, rounds=rounds,
                             plan_text=plan_text, transcript=transcript, config=cfg,
                             elapsed_s=time.time() - t0, error=error)
        self._log(f"done: {result.blocks} blocks in {result.elapsed_s:.1f}s "
                  f"(${result.cost_usd:.4f})")
        return result


def build_one(description: str, *, provider: Optional[LLMProvider] = None,
              config: Optional[AgentConfig] = None,
              images: Sequence[bytes] = ()) -> BuildResult:
    """One-liner entry point: build ``description`` with a config's provider."""
    from blockgen.agentic.providers import get_provider
    cfg = config or AgentConfig()
    if provider is None:
        provider = get_provider(cfg.provider, cache=cfg.cache, **cfg.provider_params())
    return BuildAgent(provider, cfg).build(description, images=images)


__all__ = ["AgentConfig", "BuildAgent", "BuildResult", "Round", "build_one"]
