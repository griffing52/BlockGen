"""Representation-agnostic prompt templates for suite methods.

``prompts.py`` holds the legacy wording, which is frozen (its bytes are the
response-cache keys of every published agentic number) and DSL-specific. These
templates are the generic counterparts: they never mention a command, and every
turn that asks for a build ends with ``ep.kit.actions.output_rule()``, so the same
method runs unchanged with the DSL, per-block lines, the Python API or a mix.
Turns that ask for a verdict end with :data:`JSON_RULE` instead.

Each function returns a :class:`~blockgen.agentic.providers.Message` (images
interleaved with their labels where order matters) or plain text; the Episode
adds the system prompt, history, context parts and TARGET dreams.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from blockgen.agentic.core import Attempt, DreamSpec, get_view
from blockgen.agentic.providers import Message, Part, image_message, text_message

#: The sentence that ends every verdict turn. Stable on purpose: mock providers
#: (and test fakes) recognise a JSON turn by it.
JSON_RULE = "Reply with a single JSON object and nothing else: no prose, no code fences."

#: The escape hatch of the match loop. A reply starting with this token asks for
#: a re-imagined target instead of a program.
REDREAM_TOKEN = "REDREAM:"

MODE_INSTRUCTIONS = {
    "rewrite": "Rewrite the COMPLETE program, improved. Keep what works, fix what does not.",
    "patch": ("Write ONLY the additional commands that improve this build. They run on "
              "the build as it already is, so do not repeat what is already correct; "
              "remove anything wrong explicitly."),
}


def _rule(ep: Any) -> str:
    return ep.kit.actions.output_rule()


def _mode(mode: str) -> str:
    if mode not in MODE_INSTRUCTIONS:
        raise ValueError(f"mode must be one of {sorted(MODE_INSTRUCTIONS)}, got {mode!r}")
    return MODE_INSTRUCTIONS[mode]


def _variation(ep: Any) -> str:
    # Same rider as prompts.build_prompt: changes the request (so the response cache
    # does not return one build N times) and asks for a different design.
    v = getattr(ep, "variation", None)
    if v is None:
        return ""
    return (f"\n\n(Design variation #{v}: make deliberately different choices — "
            f"massing, proportions, materials — from any other variation of this request.)")


def parse_redream(reply: str) -> Optional[str]:
    """The new description if ``reply`` is a REDREAM request, else None.

    Tolerant on purpose: the compare turn quotes the token in backticks, and models
    echo them, wrap the line in a code fence or change its case. A REDREAM that is
    missed is executed as a program (an empty build that wastes an iteration).
    """
    for line in (reply or "").splitlines():
        line = line.strip().strip("`").strip()
        if not line:
            continue
        if line.upper().startswith(REDREAM_TOKEN):
            return line[len(REDREAM_TOKEN):].strip().strip("`").strip()
        if line.lower() in ("text", "txt", "plain", "plaintext"):   # a fence's info string
            continue
        return None
    return None


def builder_dream_parts(ep: Any) -> List[Part]:
    """Labelled builder-visible dream images (what the first build turn shows)."""
    parts: List[Part] = []
    for ds in ep.kit.dreams:
        dv = ep.latest_dream(ds.key)
        if dv is None or not ds.show_builder:
            continue
        label = (f"TARGET {ds.key} (view: {ds.view}):" if ds.view
                 else f"TARGET {ds.key} (mood reference only, not compared):")
        parts += [Part("text", text=label), Part("image", image_png=dv.png)]
    return parts


def _describe_builder(ep: Any) -> str:
    """The describe stage's spec / caption text for a build or plan turn ('' when
    the recipe has no describe stage, so the turn is byte-identical)."""
    from blockgen.agentic.describe import builder_block
    return builder_block(ep)


def _describe_match(ep: Any) -> str:
    from blockgen.agentic.describe import match_block
    return match_block(ep)


def scored_dreams(ep: Any) -> List[DreamSpec]:
    """Dreams that are compared with the build and have at least one version."""
    return [d for d in ep.kit.dreams if d.view and ep.latest_dream(d.key) is not None]


# ===========================================================================
# Build-side turns (end with the action space's output rule)
# ===========================================================================
#: Opens the text plan turn (generic and dream-reading forms). Stable on purpose:
#: the mock:auto LLM answers a turn containing it with a plain-text plan.
PLAN_HEAD = "Before writing anything executable"


def plan(ep: Any) -> Message:
    """Design pass before any program. Plain text, no format-specific words.

    Without dreams (and without a ``scale`` context) the text is exactly the
    classic plan turn. When the recipe has dreams, the planner sees them
    (labelled TARGET images first) and is asked to READ them into blocks: what
    the image shows, then storeys, footprint and parts with coordinate ranges.
    A plan made blind to the target would plan some other building, and one
    that only says "match the image" leaves the pixels-to-blocks step to the
    program. With a ``scale`` context it also gets the player-scale line (its
    dream form keeps the TARGET's proportions and sizes them by its doors and
    storeys, since the build is scored against that image).
    """
    from blockgen.agentic.scale import plan_line, scale_active

    w, h, d = ep.size
    dream_parts = builder_dream_parts(ep)
    scale = [plan_line(dream=bool(dream_parts))] if scale_active(ep) else []
    if not dream_parts:
        lines = [f"Build request: {ep.prompt}", "",
                 f"{PLAN_HEAD}, plan the build. Answer in at most 12 short lines:",
                 f"- Overall footprint and height in blocks (must fit {w}x{h}x{d}).",
                 "- The material palette: which block for foundation, walls, trim, roof, "
                 "glazing.",
                 "- The parts list, in build order, each with its coordinate range.",
                 "- Two details that will make it look deliberate rather than generic.",
                 *scale,
                 "Plain text only."]
        return text_message("user", "\n".join(lines))
    described = _describe_builder(ep)
    lines = [f"Build request: {ep.prompt}", "", *([described, ""] if described else []),
             f"The TARGET image(s) above show what to build. {PLAN_HEAD}, plan the build: "
             "read the TARGET and work the design out IN BLOCKS. Answer in at most 20 "
             "short lines:",
             "- What the TARGET shows: massing, number of storeys, roof shape, openings, "
             "materials.",
             f"- Footprint (x by z) and height in blocks (must fit {w}x{h}x{d}); the floor "
             "y of each storey and where the roof starts.",
             "- The material palette: which block for foundation, walls, trim, roof, "
             "glazing.",
             "- The parts list, in build order, each with its coordinate range "
             "(x a..b, y c..d, z e..f).",
             "- The front door and the windows, with coordinates.",
             "- Two details from the TARGET that will make it look deliberate.",
             *scale,
             "Plain text only."]
    return Message("user", dream_parts + [Part("text", text="\n".join(lines))])


def build(ep: Any, images: Sequence[bytes] = ()) -> Message:
    """The generation turn. TARGET dreams / context images are prepended by
    Episode.ask on the first builder turn; this text refers to them."""
    text = f"Build: {ep.prompt}" + _variation(ep)
    if any(d.show_builder for d in ep.kit.dreams if ep.latest_dream(d.key) is not None):
        text += ("\n\nThe TARGET image(s) above show what to build. Match the massing, "
                 "proportions, roof shape, openings and colours as closely as blocks "
                 "at this canvas size allow. Scored targets are compared with renders "
                 "of your build from the same camera.")
    described = _describe_builder(ep)
    if described:
        text += "\n\n" + described
    if ep.plan:
        text += f"\n\nYour plan:\n{ep.plan.strip()}\n\nNow write the program that realizes it."
    # A 'reference' context already shows the user's images in this turn; sending
    # them twice would double their tokens and their weight.
    has_ref_ctx = any(n == "reference" for n, _ in getattr(ep.kit, "contexts", []) or [])
    imgs = list(images) if images else ([] if has_ref_ctx else list(ep.reference_images))
    if imgs:
        text = ("Reference image(s) of what to build. Match the massing, proportions and "
                "materials as closely as the block palette allows.\n\n" + text)
    text += "\n\n" + _rule(ep)
    return image_message("user", text, imgs) if imgs else text_message("user", text)


def repair(ep: Any, att: Attempt, *, max_issues: int = 20) -> Message:
    """Execution feedback: what ran, what failed, what placed nothing."""
    report = att.report
    errors = [i for i in report.issues if i.severity == "error"][:max_issues]
    warns = [i for i in report.issues if i.severity == "warning"][:max_issues]
    lines = [f"Your program ran. Result for: {ep.prompt}", "", report.summary(), ""]
    if errors:
        lines.append("Fix these failed lines:")
        lines += [f"  - {i.render()}" for i in errors]
    if warns:
        lines.append("These lines placed nothing (usually a coordinate mistake):")
        lines += [f"  - {i.render()}" for i in warns]
    if att.blocks == 0:
        lines.append("Nothing was built at all. Re-read the coordinate rules and start "
                     "from the ground (y=0).")
    lines += ["", "Rewrite the COMPLETE program, corrected.", _rule(ep)]
    return text_message("user", "\n".join(lines))


def critique_generic(ep: Any, att: Attempt, views: Sequence[str], images: Sequence[bytes],
                     *, mode: str = "rewrite") -> Message:
    """Self-critique on renders of the current build (classic, non-DSL actions)."""
    labels = ", ".join(f"{int(get_view(v).azim)}° azimuth" for v in views)
    return image_message("user", f"""\
Here is a render of the build your program produced for: {ep.prompt} The views are {labels}.

{att.report.summary()}

Critique it against the request in your head (do not write the critique), then act on
the three biggest problems. Look specifically for: floating or disconnected parts,
walls that do not close, a roof that does not sit on the walls, missing door/windows,
flat untextured surfaces that need trim, and whether the size and style match the request.

{_mode(mode)}
{_rule(ep)}""", images)


def matched_view(att: Attempt, ds: DreamSpec) -> str:
    """The view of ``ds``'s search set that scored best for this attempt.

    Scorers report per-view values (``ScoreTerm.per_view``) keyed either by view
    name or by ``'<dream>@<view>'``; the view with the highest summed value is
    the camera the dream was matched at. Falls back to the dream's own view.
    """
    cands = (ds.view,) + tuple(ds.view_search)
    if att.score is None or not att.score.per_view:
        return ds.view or "iso"
    totals = {v: 0.0 for v in cands}
    hit = False
    for pv in att.score.per_view.values():
        for v in cands:
            for k in (f"{ds.key}@{v}", v):
                if k in pv:
                    totals[v] += float(pv[k])
                    hit = True
                    break
    if not hit:
        return ds.view or "iso"
    return max(cands, key=lambda v: (totals[v], v == ds.view))


def score_table(att: Attempt) -> str:
    """Composite, parts and per-view values as a compact text table."""
    s = att.score
    if s is None:
        return "Score: (not scored)"
    lines = [f"Score: {s.value:.3f} (0 = nothing like the target, 1 = identical)"]
    if s.gated:
        lines.append(f"  GATED by '{s.gated}' (below its minimum): the score is 0 until it is fixed")
    for k, v in s.parts.items():
        pv = s.per_view.get(k)
        extra = ("  [" + ", ".join(f"{vk} {vv:.2f}" for vk, vv in pv.items()) + "]") if pv else ""
        lines.append(f"  {k}: {v:.3f}{extra}")
    return "\n".join(lines)


def compare(ep: Any, att: Attempt, *, feedback: str = "raw", mode: str = "rewrite",
            allow_redream: bool = False, vdm: str = "") -> Message:
    """The match-loop turn: TARGET vs YOURS, in the chosen feedback channel(s).

    ``raw`` image pairs per scored dream; ``score`` a text table; ``vdm`` judge
    text (passed in as ``vdm``; steps.vdm_text computes it); ``both`` = raw + vdm.
    Always: execution summary, the mode instruction, the output rule, and (if
    allowed) the REDREAM escape hatch.
    """
    if feedback not in ("raw", "score", "vdm", "both"):
        raise ValueError(f"feedback must be raw|score|vdm|both, got {feedback!r}")
    parts: List[Part] = [Part("text", text=f"Compare your build with the target for: {ep.prompt}")]
    if feedback in ("raw", "both"):
        for ds in scored_dreams(ep):
            view = matched_view(att, ds)
            yours = ep.render(att, [view]).get(view)
            parts += [Part("text", text=f"TARGET {ds.key}:"),
                      Part("image", image_png=ep.latest_dream(ds.key).png)]
            if yours is not None:
                parts += [Part("text", text=f"YOURS {ds.key} (render, view {view}):"),
                          Part("image", image_png=yours)]
    text: List[str] = []
    described = _describe_match(ep)
    if described:
        text.append(described)
    if feedback == "score":
        text.append(score_table(att))
    if feedback in ("vdm", "both") and vdm:
        text.append("Differences a reviewer saw between the target and your build:\n" + vdm)
    text.append("Execution of your current program:\n" + att.report.summary())
    text.append("Fix the biggest differences first: overall massing and footprint, then "
                "roof shape, then openings, then materials and colours, then detail.")
    text.append(_mode(mode))
    if allow_redream:
        text.append(f"If the target is impossible to build at this canvas size, reply with "
                    f"only `{REDREAM_TOKEN} <a new, buildable description of the building>` "
                    f"instead of a program.")
    text.append(_rule(ep))
    parts.append(Part("text", text="\n\n".join(text)))
    return Message("user", parts)


# ===========================================================================
# Verdict turns (end with JSON_RULE)
# ===========================================================================
def dream_review(ep: Any, key: str) -> Message:
    """Dream critic: is the latest version of ``key`` a usable build target?"""
    from blockgen.agentic.dreams import DREAM_RUBRIC

    ds = ep.dream_spec(key)
    dv = ep.latest_dream(key)
    w, h, d = ep.size
    view = ds.view or "any (context only)"
    parts: List[Part] = [Part("text", text=f"Build request: {ep.prompt}\n\n"
                              + DREAM_RUBRIC.format(W=w, H=h, D=d, view=view))]
    for c in ds.condition_on:
        ref = ep.latest_dream(c)
        if ref is not None:
            parts += [Part("text", text=f"REFERENCE {c}:"), Part("image", image_png=ref.png)]
    if dv is not None:
        parts += [Part("text", text=f"IMAGE TO REVIEW ({key}):"), Part("image", image_png=dv.png)]
    parts.append(Part("text", text=JSON_RULE))
    return Message("user", parts)


def reachability(ep: Any, att: Attempt, key: Optional[str] = None) -> Message:
    """Dream critic after a stall: is the dream ``key`` (default: the primary
    dream) reachable from this build?"""
    if key is None:
        from blockgen.agentic.dreams import primary
        key = primary(ep.kit.dreams).key
    ds = ep.dream_spec(key)
    dv = ep.latest_dream(key)
    view = ds.view or "iso"
    yours = ep.render(att, [view]).get(view)
    w, h, d = ep.size
    parts: List[Part] = [Part("text", text=(
        f"Build request: {ep.prompt}\nCanvas: {w} x {h} x {d} blocks.\n\n"
        "A builder has stopped improving at matching the TARGET image. Decide whether the "
        "target is unreachable with full cube blocks at this canvas size (too detailed, "
        "too large, organic shapes, several buildings), or whether the builder should just "
        "keep going."))]
    if dv is not None:
        parts += [Part("text", text="TARGET:"), Part("image", image_png=dv.png)]
    if yours is not None:
        parts += [Part("text", text=f"BEST BUILD SO FAR (render, view {view}):"),
                  Part("image", image_png=yours)]
    parts.append(Part("text", text=(
        'Reply as JSON: {"unreachable": true/false, "reason": "...", "revised_prompt": '
        '"a simpler, buildable description of the building (no camera or style words), '
        'or empty"}\n' + JSON_RULE)))
    return Message("user", parts)


def judge(ep: Any, att: Attempt) -> Message:
    """Judge: list the visible differences between target(s) and build (VDM text)."""
    parts: List[Part] = [Part("text", text=(
        f"Build request: {ep.prompt}\n\nList the most important visible differences "
        "between each TARGET image and the render of the build (YOURS): massing, "
        "footprint, roof, openings, materials, colours, missing parts. Be concrete "
        "(which side, roughly how many blocks)."))]
    for ds in scored_dreams(ep):
        view = matched_view(att, ds)
        yours = ep.render(att, [view]).get(view)
        parts += [Part("text", text=f"TARGET {ds.key}:"),
                  Part("image", image_png=ep.latest_dream(ds.key).png)]
        if yours is not None:
            parts += [Part("text", text=f"YOURS {ds.key} (view {view}):"),
                      Part("image", image_png=yours)]
    parts.append(Part("text", text='Reply as JSON: {"differences": ["...", "..."]} '
                                   "(at most 8, most important first).\n" + JSON_RULE))
    return Message("user", parts)


def pick_buildable(ep: Any, pngs: Sequence[bytes]) -> Message:
    """Dream critic: choose the most buildable of several candidate dreams."""
    w, h, d = ep.size
    parts: List[Part] = [Part("text", text=(
        f"Build request: {ep.prompt}\nCanvas: {w} x {h} x {d} blocks.\n\n"
        f"Here are {len(pngs)} candidate TARGET images. Pick the one that is the best "
        "target for a Minecraft build: on-prompt, a single structure, buildable from full "
        "cube blocks at this size, clearly visible."))]
    for i, png in enumerate(pngs):
        parts += [Part("text", text=f"CANDIDATE {i}:"), Part("image", image_png=png)]
    parts.append(Part("text", text='Reply as JSON: {"index": <0-based candidate number>, '
                                   '"reason": "..."}\n' + JSON_RULE))
    return Message("user", parts)


__all__ = ["JSON_RULE", "MODE_INSTRUCTIONS", "PLAN_HEAD", "REDREAM_TOKEN", "build", "builder_dream_parts",
           "compare", "parse_redream",
           "critique_generic", "dream_review", "judge", "matched_view", "pick_buildable",
           "plan", "reachability", "repair", "score_table", "scored_dreams"]
