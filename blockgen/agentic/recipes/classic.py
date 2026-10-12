"""Method ``classic``: BuildAgent's loop re-implemented on Episode verbs.

    plan?  ->  generate  ->  repair(N)?  ->  critique(M)?

Why a re-implementation next to the ``legacy`` wrapper: BuildAgent hard-wires the
DSL, its acceptance rule, the textured renderer and its prompts. ``classic`` runs
the same loop with every one of those as a recipe axis (action space, accept
policy, renderer, scorers, select), which is what the compute-matched controls of
the dream experiments need.

Parity. With ``actions: dsl`` and contexts within {palette, ontology, examples},
classic runs in *legacy-prompt mode*: it sends ``prompts.py``'s exact messages,
in BuildAgent's exact request shapes (agent.py's keep_history=False repair and
critique requests included), so the provider sees identical fingerprints and the
response cache hits the same entries. tests/agentic/test_classic_parity.py
guards this. Any other action space or context uses the generic ``turns``.

Legacy quirks kept on purpose: an empty, rejected repair or critique ends that
stage (``break``); the patch-mode concatenation format; critique on the two fixed
views ``legacy_a``/``legacy_b`` (agent.CRITIQUE_VIEWS). One deliberate difference:
``MinecraftWorld.clone`` keeps the clipboard, so a patch that ``paste``s a region
copied in an earlier round works here and silently does nothing in BuildAgent.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from blockgen.agentic.core import EpisodeLike, get_view, parse_spec, register
from blockgen.agentic.providers import text_message
from blockgen.agentic.recipes import steps

#: Context modules whose effect legacy prompts reproduce exactly.
LEGACY_CONTEXTS = frozenset({"palette", "ontology", "examples"})
#: Context modules legacy-prompt mode APPENDS to BuildAgent's system message (they
#: add text legacy never had, and change nothing else). A recipe without them
#: sends BuildAgent's exact bytes; one with them sends those bytes plus their
#: text, so ``classic-matched`` can carry the same scale rules as the dream arms
#: it controls for without leaving the legacy prompts.
LEGACY_ADDITIVE_CONTEXTS = frozenset({"scale"})
#: Action-space names that mean "BuildAgent's DSL" (legacy prompts apply).
LEGACY_ACTIONS = frozenset({"dsl"})


def legacy_prompt_mode(ep: Any) -> bool:
    """True when classic can (and so must) send BuildAgent's exact bytes.

    Only when the legacy prompt says exactly what the recipe asks for: the full
    DSL (a ``commands`` subset would be listed in full by the legacy reference),
    a palette context (the legacy system prompt always carries the palette, so a
    recipe without one is an ablation the generic path honours), and contexts
    whose params the legacy prompt reproduces (builtin example pool, palette
    hidden under an active ontology).
    """
    actions = ep.kit.actions
    if getattr(actions, "name", "") not in LEGACY_ACTIONS or getattr(actions, "allowed", None):
        return False
    ctx = {n for n, _ in ep.kit.contexts} - LEGACY_ADDITIVE_CONTEXTS
    if not (ctx <= LEGACY_CONTEXTS and "palette" in ctx):
        return False
    if (_context_params(ep, "palette") or {}).get("skip_if_ontology", True) is False:
        return False
    ex = _context_params(ep, "examples")
    if ex is not None and ex.get("pool", "builtin") != "builtin":
        return False
    return True


def _context_params(ep: Any, name: str) -> Optional[dict]:
    for c in ep.spec.contexts or []:
        n, _, params = parse_spec(c)
        if n == name:
            return params
    return None


def legacy_system(ep: Any):
    """BuildAgent's system message for this recipe (same AgentConfig fields).

    Reuses ``BuildAgent._system`` rather than re-assembling it: the bytes must be
    BuildAgent's, including the ontology table and scale hint, forever.
    """
    from blockgen.agentic.agent import AgentConfig, BuildAgent

    onto = _context_params(ep, "ontology")
    o = onto or {}
    cfg = AgentConfig(canvas_size=tuple(ep.size), target_blocks=ep.spec.target_blocks,
                      ontology="none" if onto is None else str(o.get("variant", "mined")),
                      ontology_path=o.get("path"), ontology_seed=int(o.get("seed") or 0),
                      verbose=False)
    msg = BuildAgent(None, cfg)._system()  # type: ignore[arg-type]
    extra = [fn(ep) for n, fn in ep.kit.contexts if n in LEGACY_ADDITIVE_CONTEXTS]
    extra_text = [str(getattr(p, "system_text", p) or "").strip("\n") for p in extra]
    extra_text = [t for t in extra_text if t]
    if not extra_text:
        return msg
    return text_message("system", "\n\n".join([msg.text] + extra_text))


def _legacy_generate(ep: Any, n_examples: int) -> Any:
    from blockgen.agentic.examples import EXAMPLES, select_examples
    from blockgen.agentic.prompts import build_prompt

    k = n_examples
    if not k:
        ex = _context_params(ep, "examples")
        k = int(ex.get("k", 1)) if ex is not None else 0
    examples = select_examples(ep.prompt, k, tuple(EXAMPLES))
    msgs = build_prompt(ep.prompt, plan=ep.plan or None, examples=examples,
                        images=ep.reference_images, variation=ep.variation)
    pairs, ask = msgs[:-1], msgs[-1]
    if ep.spec.keep_history:
        ep.history.extend(pairs)
        reply = ep.ask(ask, stage="generate")
    else:
        reply = ep.ask(ask, history=pairs, stage="generate")
    return ep.try_("generate", reply)


def _legacy_repair(ep: Any, rounds: int) -> None:
    from blockgen.agentic.prompts import repair_prompt

    for i in range(rounds):
        cur = ep.current() or (ep.attempts[-1] if ep.attempts else None)
        if cur is None or not steps.needs_repair(cur, ep):
            break
        stage = f"repair{i + 1}"
        ask = repair_prompt(cur.report, description=ep.prompt)
        if ep.spec.keep_history:
            reply = ep.ask(ask, stage=stage)
        else:   # agent.py: [system, user 'Build: ...', assistant <program>, ask]
            reply = ep.ask(ask, history=[text_message("user", f"Build: {ep.prompt}"),
                                         text_message("assistant", cur.program_text)],
                           stage=stage)
        att = ep.try_(stage, reply)
        if not att.accepted and att.blocks == 0:
            break


def _legacy_critique(ep: Any, rounds: int, mode: str, views: Sequence[str]) -> None:
    from blockgen.agentic.prompts import critique_prompt

    for i in range(rounds):
        cur = ep.current()
        if cur is None or cur.blocks == 0:
            break
        renders = ep.render(cur, list(views))
        imgs = [renders[v] for v in views if v in renders]
        if not imgs:
            ep.trace.event("critique_skipped", reason="no render")
            break
        stage = f"critique{i + 1}"
        ask = critique_prompt(ep.prompt, cur.report, imgs,
                              views=[f"{int(get_view(v).azim)}° azimuth" for v in views],
                              mode=mode)
        # agent.py: keep_history=False critique request is [system, ask]
        reply = ep.ask(ask, role="critic", history=bool(ep.spec.keep_history), stage=stage)
        att = ep.try_(stage, reply, patch=(mode == "patch"))
        if not att.accepted and att.blocks == 0:
            break


@register("method", "classic")
def classic(ep: EpisodeLike, *, plan: bool = False, n_examples: int = 0, repair_rounds: int = 1,
            critique_rounds: int = 0, critique_mode: str = "rewrite",
            critique_views: Sequence[str] = ("legacy_a", "legacy_b")) -> None:
    """BuildAgent's plan / generate / repair / critique loop on any action space."""
    if critique_mode not in ("rewrite", "patch"):
        raise ValueError(f"critique_mode must be rewrite|patch, got {critique_mode!r}")
    if critique_mode == "patch" and critique_rounds and not getattr(ep.kit.actions, "patchable", False):
        raise ValueError(f"critique_mode 'patch' needs a patchable action space; "
                         f"'{getattr(ep.kit.actions, 'name', '?')}' is not")
    views = [get_view(v).name for v in critique_views]
    if legacy_prompt_mode(ep):
        from blockgen.agentic.prompts import plan_prompt

        ep.set_system(legacy_system(ep))
        ep.inject_first_turn = False
        if plan:
            # agent.py: request = [system, plan_prompt]; not part of the dialogue
            ep.plan = str(ep.ask(plan_prompt(ep.prompt, tuple(ep.size)), role="planner",
                                 history=False, stage="plan")).strip()
            steps.record_text(ep, "plan.txt", ep.plan)
        _legacy_generate(ep, n_examples)
        _legacy_repair(ep, repair_rounds)
        _legacy_critique(ep, critique_rounds, critique_mode, views)
        return

    if plan:
        steps.plan(ep)
    has_examples_ctx = any(n == "examples" for n, _ in ep.kit.contexts)
    att = steps.generate(ep, n_examples=0 if has_examples_ctx else n_examples)
    steps.repair(ep, att, rounds=repair_rounds)
    for i in range(critique_rounds):
        att = steps.critique_turn(ep, i, views=views, mode=critique_mode)
        if att is None or (not att.accepted and att.blocks == 0):
            break


__all__ = ["LEGACY_ACTIONS", "LEGACY_ADDITIVE_CONTEXTS", "LEGACY_CONTEXTS", "classic",
           "legacy_prompt_mode", "legacy_system"]
