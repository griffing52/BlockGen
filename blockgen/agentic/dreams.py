"""Dream presets: what the image model is asked to imagine, and how it is checked.

A *dream* is an image of the build the method wants, generated before (and
sometimes during) the build loop. Each :class:`~blockgen.agentic.core.DreamSpec`
here is data: a prompt template, the camera it is compared against, a weight in
the score, and the other dreams it is conditioned on (passed as reference
images, which is what keeps a back view or a floor plan consistent with the iso
exterior).

Why the style text is pinned (:data:`DEFAULT_STYLE`) and the camera lives in the
template: the in-loop scorer compares a dream with a *render* of the build. Every
degree of freedom the image model takes (lighting, backdrop, terrain, a second
building, a different camera) is distance the build can never close, so the
template removes as many as it can. The free part is ``{desc}``, which is what a
dream critic or a REDREAM reply rewrites; style and camera never drift.

A recipe names presets (``dreams: [iso, interior, plan]``) or gives inline dicts.
``turnaround`` is an expansion (several presets at once), see :func:`expand`.
"""

from __future__ import annotations

import string
from typing import Any, Dict, List, Mapping, Optional, Sequence

from blockgen.agentic.core import DreamSpec

#: Style rider shared by every preset (``{style}``). Camera-neutral on purpose:
#: each template states its own camera, so a plan and an iso view share one look.
DEFAULT_STYLE = (
    "A single Minecraft build made of full cube blocks, rendered in the game's "
    "style with flat, even daylight and no shadows, on a plain white background. "
    "Only the structure: no terrain, no sky, no people, no text, no frame."
)

#: A style that imitates our own renders rather than the game: crisp flat textures,
#: orthographic, no lighting effects. Why it exists: most of the distance between
#: a DEFAULT_STYLE dream and a render is picture style (soft shading, backdrop),
#: which no build edit can close, so the loop's score moves by noise. Not the
#: default: switching it would change every existing dream recipe's requests, so a
#: recipe opts in with ``dream_style: render_match`` (staged-v1 A/B-tests it).
RENDER_MATCH_STYLE = (
    "A single Minecraft build made of full cube blocks with crisp, flat, unshaded "
    "pixel textures, drawn like a clean orthographic game render: no perspective, "
    "flat even light, no cast shadows, no drop shadow, no ambient occlusion, no "
    "ground plane, on a pure white (#FFFFFF) background. Only the structure: no "
    "terrain, no sky, no people, no text, no frame.")

#: For CLIP-encoded local models (SDXL / SD Turbo). Their text encoder reads only
#: the first 77 tokens, and the template puts ~80 words of style and camera before
#: the description, so the building itself was cut off. ``compact`` puts the
#: description FIRST and keeps the rest to a short, explicit list (SDXL-Turbo runs at
#: guidance 0, so it has no negative prompt: the exclusions must be in the prompt).
COMPACT_STYLE = ("Minecraft voxel build made of cube blocks, isolated on a plain white "
                 "background, no terrain, no grass, no water, no trees")

#: Short camera phrases for ``compact``, by the view a dream targets.
COMPACT_VIEWS: Dict[str, str] = {
    "iso": "isometric three-quarter view from above",
    "iso_back": "isometric view from behind", "iso_left": "isometric side view",
    "iso_right": "isometric side view", "cutaway": "cutaway view showing the interior",
    "plan": "top-down floor plan view", "room": "cutaway view of one room",
    "room_top": "top-down view of one room",
}

#: Named styles a recipe's ``dream_style`` may give instead of literal text.
STYLES: Dict[str, str] = {"default": DEFAULT_STYLE, "render_match": RENDER_MATCH_STYLE,
                          "compact": COMPACT_STYLE}

_ISO_CAMERA = ("Isometric three-quarter view from above (camera about 45 degrees "
               "around and 30 degrees up), the whole building in frame and centred.")
_FIT = "It fits in {W} x {D} blocks of footprint and {H} blocks of height."
_SAME = "The SAME building as the reference image (same massing, materials and colours)"

#: Built-in dream specs. Add one entry to make a preset available to every recipe.
DREAMS: Dict[str, DreamSpec] = {
    # The primary exterior. view_search absorbs the image model's approximate
    # camera: its term is the max over these views.
    "iso": DreamSpec(
        "iso", "{style} " + _ISO_CAMERA + " {desc}. " + _FIT,
        view="iso", weight=1.0, view_search=("iso_back", "iso_left", "iso_right")),
    "back": DreamSpec(
        "back", "{style} " + _SAME + ", seen from behind (the opposite side), same "
        "isometric camera height. {desc}. " + _FIT,
        view="iso_back", weight=0.5, condition_on=("iso",)),
    "left": DreamSpec(
        "left", "{style} " + _SAME + ", seen from its left side, same isometric "
        "camera height. {desc}. " + _FIT,
        view="iso_left", weight=0.5, condition_on=("iso",)),
    "right": DreamSpec(
        "right", "{style} " + _SAME + ", seen from its right side, same isometric "
        "camera height. {desc}. " + _FIT,
        view="iso_right", weight=0.5, condition_on=("iso",)),
    # Interior cutaway: compared against View 'cutaway' (roof and upper walls removed).
    "interior": DreamSpec(
        "interior", "{style} Cutaway of " + _SAME + " with the roof and the upper part "
        "of the walls removed, seen from above at a steep angle so the rooms, floors "
        "and furniture inside are visible. {desc}. " + _FIT,
        view="cutaway", weight=0.3, condition_on=("iso",)),
    # Ground-floor plan: compared against View 'plan' (top-down, bottom 30%).
    "plan": DreamSpec(
        "plan", "{style} Top-down orthographic floor plan of the ground floor of "
        + _SAME + ": the walls, doors and room layout seen straight from above, "
        "nothing above the first storey. {desc}. " + _FIT,
        view="plan", weight=0.5, condition_on=("iso",)),
    # Context only: shown to the builder for mood, never scored.
    "mood": DreamSpec(
        "mood", "{style} {desc}, cosy evening lighting, eye-level view.",
        view=None, weight=0.0, show_builder=True),
}

#: Names that expand to several presets (a recipe may list them like a preset).
EXPANSIONS: Dict[str, Sequence[str]] = {
    "turnaround": ("iso", "back", "left", "right"),
}


def expand(entries: Sequence[Any]) -> List[Any]:
    """Replace expansion names (``turnaround``) by their presets, keeping order
    and dropping duplicates of an already-listed key.

    Why a separate step: an expansion is a list of dreams, while every other entry
    is one, so resolving it inside the per-entry loop would muddle the error for
    an unknown name.
    """
    out: List[Any] = []
    seen = set()
    for e in entries or []:
        names = EXPANSIONS.get(e, (e,)) if isinstance(e, str) else (e,)
        for n in names:
            key = n if isinstance(n, str) else (n.key if isinstance(n, DreamSpec)
                                                else dict(n).get("key"))
            if key in seen:
                continue
            seen.add(key)
            out.append(n)
    return out


class _Blank(dict):
    """format_map dict that leaves unknown ``{fields}`` visible instead of raising:
    a template typo should show up in the trace, not kill the build."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def scale_cue(ep: Any) -> str:
    """The player-scale cue for this episode's dreams ('' unless the recipe lists
    the ``scale`` context with ``dream_cue`` on, its default).

    It rides on the style, so every template with ``{style}`` (iso, back, room and
    storey dreams) carries it and the camera wording stays where it was. It sizes
    the openings and never adds a person: DEFAULT_STYLE says "no people", and a
    figure in the dream is something the build can never match.
    """
    from blockgen.agentic.scale import dream_cue
    return dream_cue(ep)


def render_prompt(ds: DreamSpec, ep: Any, *, desc: Optional[str] = None,
                  style: Optional[str] = None) -> str:
    """Fill ``{style} {desc} {W} {H} {D}`` for one dream.

    With a ``scale`` context (``dream_cue`` on) the style gains
    :data:`scale.DREAM_CUE`; without one the request is byte-identical to before.

    ``desc`` defaults to the episode's current description of this dream
    (``ep.dream_desc[key]``, set when a critic or REDREAM rewrote it), else the
    build prompt. A rewrite replaces only this field, so style and camera stay
    pinned.
    """
    w, h, d = tuple(ep.spec.canvas)
    st = style if style is not None else (ep.spec.dream_style or DEFAULT_STYLE)
    st = STYLES.get(st, st)          # a style name, or literal style text
    cue = scale_cue(ep)
    if cue:
        st = f"{st.rstrip()} {cue}"
    if not desc:
        desc = (getattr(ep, "dream_desc", None) or {}).get(ds.key) or ep.prompt
    text = desc.strip().rstrip(".")
    if (style if style is not None else ep.spec.dream_style) == "compact":
        # Description first, template ignored: see COMPACT_STYLE. Cue last, so it
        # is the part a 77-token encoder drops first.
        view = COMPACT_VIEWS.get(ds.view or "", "isometric view")
        tail = f" {cue}" if cue else ""
        return " ".join(f"{text}. {COMPACT_STYLE}, {view}.{tail}".split())
    try:
        out = string.Formatter().vformat(ds.template, (), _Blank(
            style=st, desc=text, W=w, H=h, D=d))
    except (ValueError, IndexError):  # a stray brace: fall back to plain concatenation
        out = f"{st} {ds.template} {text}"
    return " ".join(out.split())


def topo(dreams: Sequence[DreamSpec]) -> List[DreamSpec]:
    """Order dreams so every dream comes after the ones it is conditioned on.

    Stable (list order among independent dreams); a ``condition_on`` key that is
    not in the list is ignored (it cannot be satisfied, and kit validation already
    reports it). Cycles raise, since neither dream could ever go first.
    """
    by_key = {d.key: d for d in dreams}
    out: List[DreamSpec] = []
    state: Dict[str, int] = {}          # 1 = visiting, 2 = done

    def visit(d: DreamSpec, path: Sequence[str]) -> None:
        s = state.get(d.key)
        if s == 2:
            return
        if s == 1:
            raise ValueError(f"dream condition_on cycle: {' -> '.join(list(path) + [d.key])}")
        state[d.key] = 1
        for c in d.condition_on:
            if c in by_key:
                visit(by_key[c], list(path) + [d.key])
        state[d.key] = 2
        out.append(d)

    for d in dreams:
        visit(d, [])
    return out


def dependants(dreams: Sequence[DreamSpec], key: str) -> List[DreamSpec]:
    """Dreams conditioned on ``key``, directly or transitively, in topo order.
    A re-dream of ``key`` makes all of them stale."""
    hit = {key}
    out = []
    for d in topo(dreams):
        if d.key != key and any(c in hit for c in d.condition_on):
            hit.add(d.key)
            out.append(d)
    return out


def primary(dreams: Sequence[DreamSpec]) -> Optional[DreamSpec]:
    """The dream REDREAM / dream_vote act on: the first scored root dream (usually
    iso), else the first scored one, else the first one."""
    ordered = topo(dreams)
    for d in ordered:
        if d.view and not d.condition_on:
            return d
    for d in ordered:
        if d.view:
            return d
    return ordered[0] if ordered else None


#: What the dream critic checks (turns.dream_review). Each field maps to a failure
#: we have seen make a dream useless as a build target.
DREAM_RUBRIC = """\
You review a generated image that will be used as the TARGET for a Minecraft build
on a {W} x {H} x {D} block canvas (x by y-up by z). Judge it on:
- buildable: can it be built from full cube blocks at this size? Fine curves,
  organic shapes, tiny ornament or a structure far larger than the canvas are not.
- on_prompt: does it show what was asked for?
- single_structure: exactly one building, not a village, a landscape or a collage.
- view_ok: the camera matches the requested view ({view}).
- consistent_with_refs: if reference images were given, it is the SAME building.
Give an overall score from 0 (useless) to 10 (ideal target).

Reply as JSON with exactly these keys:
{{"score": 0-10, "buildable": true/false, "on_prompt": true/false,
  "single_structure": true/false, "view_ok": true/false,
  "consistent_with_refs": true/false, "issues": ["..."],
  "revised_prompt": "a rewritten description of the building only (no camera, style or background words) that fixes the issues, or empty if none",
  "keep_composition": true/false (true if the image is close and should be used as a reference for the next try)}}"""


def normalize_verdict(v: Mapping[str, Any]) -> Dict[str, Any]:
    """Coerce a parsed dream-critic reply into typed fields with safe defaults.

    Models return "7/10", "yes", or omit keys. Missing booleans default to True
    (absence of a complaint), so only an explicit problem triggers regeneration.
    """
    def as_bool(x: Any, default: bool = True) -> bool:
        if isinstance(x, bool):
            return x
        if isinstance(x, str):
            return x.strip().lower() in ("true", "yes", "y", "1")
        return default if x is None else bool(x)

    raw = v.get("score", 0)
    try:
        score = float(str(raw).split("/")[0].strip())
    except ValueError:
        score = 0.0
    issues = v.get("issues") or []
    return {
        "score": score,
        "buildable": as_bool(v.get("buildable")),
        "on_prompt": as_bool(v.get("on_prompt")),
        "single_structure": as_bool(v.get("single_structure")),
        "view_ok": as_bool(v.get("view_ok")),
        "consistent_with_refs": as_bool(v.get("consistent_with_refs")),
        "issues": [str(i) for i in issues] if isinstance(issues, list) else [str(issues)],
        "revised_prompt": str(v.get("revised_prompt") or "").strip(),
        "keep_composition": as_bool(v.get("keep_composition"), default=False),
    }


__all__ = ["DEFAULT_STYLE", "DREAMS", "DREAM_RUBRIC", "EXPANSIONS", "RENDER_MATCH_STYLE",
           "STYLES", "dependants", "expand", "normalize_verdict", "primary", "render_prompt",
           "scale_cue", "topo"]
