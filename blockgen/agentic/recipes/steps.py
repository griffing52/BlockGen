"""Reusable steps for method functions.

A method is a short Python function; these are the paragraphs it is written in.
Each step uses only Episode verbs (``ask``, ``try_``, ``render``, ``dream``,
``rescore_all``) so a new method can mix them freely, and each one encodes a
decision we want made the same way everywhere:

* :func:`repair` only runs on a failed / no-op / empty build (symbolic feedback is
  worthless on a clean one) and stops on an empty reply, like BuildAgent.
* :func:`review_dream` treats an unparseable verdict as "keep" (an image costs
  money; a JSON hiccup should not buy another).
* :func:`redream` regenerates the dreams conditioned on the one it replaces, so a
  back view or plan never describes a building that no longer exists, and it
  re-anchors every score (``rescore_all``) so best-of selection stays like with like.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from blockgen.agentic import describe as describe_mod
from blockgen.agentic import dreams as dreams_mod
from blockgen.agentic import turns
from blockgen.agentic.core import Attempt, BudgetExceeded, DreamVersion, ImageRequest
from blockgen.agentic.providers import text_message

#: Appended to a re-dream prompt anchored on a render of the current build.
ANCHOR_NOTE = ("Keep it buildable: stay close to the massing of the block build shown in "
               "the reference render, simplify anything finer than one block.")


# ===========================================================================
# Build steps
# ===========================================================================
def record_text(ep: Any, name: str, text: str, *, kind: str = "plan_text") -> None:
    """Keep a plain-text design artefact (a plan, a design brief) in the trace:
    ``<trace>/<name>`` when the trace writes files, and an event either way, so a
    reader of the build sees the reasoning the program was written from."""
    root = getattr(getattr(ep, "trace", None), "root", None)
    if root is not None:
        try:
            from pathlib import Path

            Path(root).mkdir(parents=True, exist_ok=True)
            (Path(root) / name).write_text(text or "", encoding="utf-8")
        except OSError as exc:      # a trace write must not cost the build
            ep.trace.event("record_text_error", name=name, error=repr(exc))
    ep.trace.event(kind, file=name, chars=len(text or ""), text=(text or "")[:4000])


def plan(ep: Any) -> str:
    """Design pass (role planner, no history). Stored on ``ep.plan`` and in the
    trace (``plan.txt``). With dreams the planner sees them (turns.plan)."""
    ep.plan = str(ep.ask(turns.plan(ep), role="planner", history=False, stage="plan")).strip()
    record_text(ep, "plan.txt", ep.plan)
    return ep.plan


def example_pairs(ep: Any, k: int) -> List[Any]:
    """``k`` (request, answer) examples in the ACTION SPACE's own format, as
    user/assistant message pairs."""
    out = []
    for req, ans in ep.kit.actions.examples(ep.prompt, k) if k > 0 else []:
        out += [text_message("user", f"Build: {req}"), text_message("assistant", ans.strip())]
    return out


def generate(ep: Any, *, n_examples: int = 0, stage: str = "generate") -> Attempt:
    """The first build turn and its execution.

    Example pairs become permanent dialogue when ``keep_history`` (the model keeps
    seeing them), otherwise they are sent with this turn only.
    """
    pairs = example_pairs(ep, n_examples)
    if ep.spec.keep_history:
        ep.history.extend(pairs)
        reply = ep.ask(turns.build(ep), stage=stage)
    else:
        reply = ep.ask(turns.build(ep), history=pairs, stage=stage)
    return ep.try_(stage, reply)


def needs_repair(att: Optional[Attempt], ep: Any = None) -> bool:
    """Failed / no-op lines or an empty build. In a worst-case cost estimate
    (``ep.kit.worst_case``) always True, since mock replies never fail."""
    if ep is not None and getattr(ep.kit, "worst_case", False):
        return True
    return att is None or bool(att.report.n_failed or att.report.n_noop or att.blocks == 0)


def repair(ep: Any, att: Optional[Attempt] = None, *, rounds: int = 1) -> Optional[Attempt]:
    """Symbolic repair rounds: only while the current build has failed / no-op
    lines or is empty. An empty rejected repair stops the stage (BuildAgent)."""
    last = att
    for i in range(max(0, int(rounds))):
        cur = ep.current() or last
        if cur is None or not needs_repair(cur, ep):
            break
        stage = f"repair{i + 1}"
        if ep.spec.keep_history:
            reply = ep.ask(turns.repair(ep, cur), stage=stage)
        else:
            reply = ep.ask(turns.repair(ep, cur), history=[
                text_message("user", f"Build: {ep.prompt}"),
                text_message("assistant", cur.program_text)], stage=stage)
        last = ep.try_(stage, reply)
        if not last.accepted and last.blocks == 0:
            break
    return ep.current() or last


def critique_turn(ep: Any, i: int, *, views: Sequence[str], mode: str = "rewrite") -> Optional[Attempt]:
    """One generic visual self-critique round on the current build. None = no
    render available (no renderer, empty build): the caller stops critiquing."""
    cur = ep.current()
    if cur is None or cur.blocks == 0:
        return None
    renders = ep.render(cur, list(views))
    imgs = [renders[v] for v in views if v in renders]
    if not imgs:
        ep.trace.event("critique_skipped", reason="no render")
        return None
    stage = f"critique{i + 1}"
    turn = turns.critique_generic(ep, cur, [v for v in views if v in renders], imgs, mode=mode)
    reply = ep.ask(turn, role="critic", history=bool(ep.spec.keep_history), stage=stage)
    return ep.try_(stage, reply, patch=(mode == "patch"))


# ===========================================================================
# Dream steps
# ===========================================================================
def condition_refs(ep: Any, key: str) -> List[bytes]:
    ds = ep.dream_spec(key)
    return [dv.png for dv in (ep.latest_dream(k) for k in ds.condition_on) if dv is not None]


def next_seed(ep: Any, key: str) -> int:
    """Seed of the next version of ``key``: a new seed is a new image-cache key,
    i.e. a genuinely new sample (API backends ignore the seed itself)."""
    return int(ep.seed) + 1000 * len(ep.dreams.get(key, []))


def review_dream(ep: Any, key: str) -> Dict[str, Any]:
    """Dream critic verdict on the latest version of ``key`` (stored on it).

    Returns ``{}`` when the reply could not be parsed; callers keep the dream.
    """
    raw = ep.ask(turns.dream_review(ep, key), role="dream_critic", history=False,
                 json_reply=True, stage=f"dream_review:{key}")
    if not raw:
        ep.trace.event("dream_review_unparsed", key=key)
        return {}
    v = dreams_mod.normalize_verdict(raw)
    dv = ep.latest_dream(key)
    if dv is not None:
        dv.critique = v
    return v


def dream_ok(v: Dict[str, Any], min_score: float) -> bool:
    return bool(v) and v["score"] >= min_score and v["buildable"] and v["on_prompt"]


def refine_dream(ep: Any, key: str, *, rounds: int, min_score: float = 7) -> DreamVersion:
    """Review the latest version of ``key`` and regenerate it up to ``rounds``
    times while the critic rejects it (option (a), before any build).

    A regeneration uses the critic's revised description (style and camera stay
    pinned by the template), the rejected image as a reference only when the
    critic says the composition is worth keeping, plus the condition refs.
    """
    for _ in range(max(0, int(rounds))):
        v = review_dream(ep, key)
        if not v or dream_ok(v, min_score):
            break
        latest = ep.latest_dream(key)
        ds = ep.dream_spec(key)
        if v["revised_prompt"]:
            ep.dream_desc[key] = v["revised_prompt"]   # later re-dreams keep it
        prompt = dreams_mod.render_prompt(ds, ep)
        refs = ([latest.png] if (v["keep_composition"] and latest is not None) else [])
        refs += condition_refs(ep, key)
        ep.trace.event("dream_regenerate", key=key, score=v["score"], issues=v["issues"])
        try:
            ep.dream(key, prompt=prompt, refs=refs, seed=next_seed(ep, key))
        except BudgetExceeded as exc:
            # The image cap (max_images) bounds regeneration, it should not cancel
            # the build: keep the latest dream and go on. A usd cap that is truly
            # exhausted stops the episode at the next LLM call anyway.
            ep.trace.event("refine_budget_stop", key=key, error=str(exc))
            break
    return ep.latest_dream(key)


def ensure_image_budget(ep: Any, n: int) -> None:
    """Raise BudgetExceeded unless ``n`` more images fit every ledger level.

    Why up front: a re-dream regenerates a dream AND its dependants. Running out
    half-way would leave a back view or plan of a building that no longer exists,
    so either all ``n`` images fit or none is generated. Counts are pessimistic
    (a cache hit is free but counted); the usd side uses the generator's estimate
    for a plain request (refs may add a surcharge, so it is an estimate).
    """
    image = ep.kit.image
    est = 0.0
    if image is not None:
        req = ImageRequest("estimate", size=str(getattr(image, "size", "1024x1024")),
                           quality=str(getattr(image, "quality", "low")))
        est = float(image.estimate_usd(req) or 0.0)
    led = ep.ledger
    while led is not None:
        b = led.budget
        if b.max_images is not None and led.n_images + n > b.max_images:
            raise BudgetExceeded(f"ledger '{led.name}': {n} more image(s) would pass the "
                                 f"image cap {b.max_images} ({led.n_images} used)")
        if b.usd is not None and est and led.spent_usd + est * n > b.usd + 1e-12:
            raise BudgetExceeded(f"ledger '{led.name}': {n} image(s) at ~${est:.4f} would pass "
                                 f"the cap ${b.usd:.4f} (${led.spent_usd:.4f} spent)")
        led = led.parent


def redream_dependants(ep: Any, key: str) -> List[Any]:
    """The existing dreams a re-dream of ``key`` must regenerate (topo order)."""
    return [d for d in dreams_mod.dependants(ep.kit.dreams, key)
            if ep.latest_dream(d.key) is not None]


def redream(ep: Any, key: str, desc: Optional[str] = None, *, anchor: Optional[Attempt] = None,
            dependants: bool = True, reason: str = "") -> DreamVersion:
    """Re-imagine dream ``key`` mid-loop, then re-anchor every score.

    ``desc`` replaces the template's ``{desc}``; ``anchor`` adds a render of that
    attempt as a reference ("make the target look like something buildable").
    Dependent dreams (condition_on, transitively) are regenerated against the new
    version so a multi-view target stays one building: a non-empty ``desc``
    becomes the current description of ``key`` and every dependant
    (``ep.dream_desc``); an empty one keeps the current descriptions (an earlier
    critic rewrite, not the original prompt). The images for all of them are
    checked against the budget first (:func:`ensure_image_budget`). The dialogue
    is then told the target changed (:meth:`Episode.note_target_changed`).
    """
    ds = ep.dream_spec(key)
    deps = redream_dependants(ep, key) if dependants else []
    ensure_image_budget(ep, 1 + len(deps))
    refs: List[bytes] = []
    if anchor is not None and ds.view:
        png = ep.render(anchor, [ds.view]).get(ds.view)
        if png is not None:
            refs.append(png)
    refs += condition_refs(ep, key)
    desc = (desc or "").strip()
    if desc:
        for k in [key] + [d.key for d in deps]:
            ep.dream_desc[k] = desc
    prompt = dreams_mod.render_prompt(ds, ep)
    if anchor is not None:
        prompt += " " + ANCHOR_NOTE
    ep.trace.event("redream", key=key, desc=desc, anchored=anchor is not None,
                   reason=reason)
    dv = ep.dream(key, prompt=prompt, refs=refs, seed=next_seed(ep, key))
    for d in deps:
        ep.dream(d.key, prompt=dreams_mod.render_prompt(d, ep), seed=next_seed(ep, d.key))
    ep.rescore_all()
    note = getattr(ep, "note_target_changed", None)
    if note is not None:
        note(desc or ep.dream_desc.get(key, ""))
    return dv


def reachability(ep: Any, att: Attempt, key: Optional[str] = None) -> Dict[str, Any]:
    """Dream critic: is dream ``key`` (default: the primary dream) unreachable from
    this build? ``{}`` if the verdict could not be parsed (treated as reachable)."""
    if key is None:
        key = dreams_mod.primary(ep.kit.dreams).key
    v = ep.ask(turns.reachability(ep, att, key), role="dream_critic", history=False,
               json_reply=True, stage=f"reachability:{key}")
    if not v:
        return {}
    unreachable = v.get("unreachable")
    if isinstance(unreachable, str):
        unreachable = unreachable.strip().lower() in ("true", "yes", "1")
    return {"unreachable": bool(unreachable), "reason": str(v.get("reason") or ""),
            "revised_prompt": str(v.get("revised_prompt") or "").strip()}


def vdm_text(ep: Any, att: Attempt, stage: str = "judge") -> str:
    """Verbal difference text for ``feedback: vdm``: a 'vlm' scorer's note if the
    recipe has one (already paid for), else one judge call."""
    if att.score is not None and att.score.notes.get("vlm"):
        return att.score.notes["vlm"]
    v = ep.ask(turns.judge(ep, att), role="judge", history=False, json_reply=True, stage=stage)
    diffs = v.get("differences") if isinstance(v, dict) else None
    if isinstance(diffs, list):
        return "\n".join(f"- {d}" for d in diffs if str(d).strip())
    return str(diffs or "")


def match_step(ep: Any, i: int, cur: Attempt, *, feedback: str, mode: str,
               allow_redream: bool) -> Any:
    """One compare turn. Returns the critic's reply text (may be a REDREAM).
    A describe stage first captions any dream that changed (a re-dream)."""
    describe_mod.refresh(ep)
    vdm = vdm_text(ep, cur, stage=f"judge{i}") if feedback in ("vdm", "both") else ""
    turn = turns.compare(ep, cur, feedback=feedback, mode=mode, allow_redream=allow_redream,
                         vdm=vdm)
    return ep.ask(turn, role="critic", stage=f"match{i}")


__all__ = ["ANCHOR_NOTE", "condition_refs", "critique_turn", "dream_ok", "ensure_image_budget",
           "example_pairs", "generate", "match_step", "needs_repair", "next_seed", "plan", "reachability",
           "record_text",
           "redream", "redream_dependants", "refine_dream", "repair", "review_dream", "vdm_text"]
