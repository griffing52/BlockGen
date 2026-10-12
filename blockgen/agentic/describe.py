"""Describe stage for dream-style methods: the building in words, next to its image.

Two optional LLM turns around the dream, chosen by the method param ``describe``:

``before`` (EXPAND)
    Role ``planner`` (falls back to the builder), one text turn, no images: the
    user's request is written out as a detailed design spec in a fixed schema
    (:data:`SPEC_KEYS`: summary, footprint, storeys with clear heights, roof,
    materials as palette block names, openings, distinctive features, what to
    leave out). With ``expand_into_image`` the spec, rendered as one compact
    description (:func:`image_desc`), is appended to ``{desc}`` of every recipe
    dream, so the image is drawn from it; with ``show_spec`` the builder sees it.
    A staged method whose blueprint already fixes storeys and materials appends
    only the parts the blueprint leaves open (:func:`augment_desc`).
    Stored as ``dreams/spec.json`` and a ``describe_spec`` trace event.
``after`` (CAPTION)
    Role ``judge`` (vision), per scored dream (``caption_keys``: ``primary`` or
    ``all``): a structured caption of what the IMAGE shows, in a schema of its own
    (the spec's visible fields: no floor material and no clear heights, which an
    exterior view does not show) plus estimated block dimensions, per-face notes
    and an ``uncertain`` list. The captioner is not told the request: it
    describes the image alone and cannot bend it towards the request. Captions are
    cached on the image digest, so a regenerated or re-dreamed image gets a new
    caption and an unchanged one never costs a second call. Stored next to the
    dream (``dreams/<key>_v<n>.caption.json``). The builder sees it in the build
    turn and, with ``caption_in_match``, the critic sees it in every match turn as
    ``TARGET DESCRIPTION``.
``both``
    Expand, dream, then caption. When the caption disagrees with the spec, the
    builder is told to follow the image and its caption (the image is what is
    scored); the disagreement (visible materials, storeys, roof shape) is logged.
    Disagreements need ``caption_format: json``; under prose a
    ``describe_disagreement_skipped`` event says they were not computed.

``caption_format`` (``json`` | ``prose``) is the reply format of both turns.

``caption_control: shuffled`` is the control for ``after``/``both``: the same
calls are made, but the builder and critic are shown the caption of a DIFFERENT
prompt's dream from the same arm (the previous prompt that the arm captioned;
the arm's first prompt gets the :data:`LIBRARY` caption that shares the fewest
words with its prompt, ties broken by a hash). It holds the extra text constant and breaks its link to the target, so
``dream-caption`` minus ``dream-caption-shuffled`` is what the description adds
beyond more tokens. The pairing follows run order; a re-run of one prompt alone
pairs it with the library.

Everything is off by default (``describe: none``): no call is made and every
request is byte-identical to before. A method opts in by calling :func:`setup`
before its dreams and :func:`refresh` after any dream may have changed; turns
read the text through :func:`builder_block` and :func:`match_block`.
"""

from __future__ import annotations

import hashlib
import json
import re
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

MODES = ("none", "before", "after", "both")
FORMATS = ("json", "prose")
CONTROLS = ("none", "shuffled")
CAPTION_KEYS = ("primary", "all")

#: Literal tags that open the two turns. Stable on purpose: the mock:auto LLM
#: recognises the turns by them.
SPEC_TAG = "WRITE THE DESIGN SPEC"
CAPTION_TAG = "DESCRIBE THE TARGET IMAGE"

MATERIAL_SLOTS = ("walls", "trim", "roof", "floor", "windows", "door", "foundation")
#: The slots a caption names: the floor is inside, so an exterior image cannot show it.
CAPTION_MATERIAL_SLOTS = tuple(s for s in MATERIAL_SLOTS if s != "floor")
SPEC_KEYS = ("summary", "footprint_blocks", "storeys", "roof", "materials", "openings",
             "distinctive_features", "surroundings_to_omit")
CAPTION_EXTRA_KEYS = ("dimensions_blocks", "faces", "uncertain")

#: Method params this stage adds (with defaults). Every method that supports it
#: takes exactly these.
PARAMS: Dict[str, Any] = {
    "describe": "none", "caption_format": "json", "expand_into_image": True,
    "show_spec": True, "caption_keys": "primary", "caption_in_match": True,
    "caption_control": "none"}

_SPEC_JSON = (
    '{"summary": "one sentence", "footprint_blocks": [w, d], '
    '"storeys": {"count": n, "clear_heights": [h, ...]}, '
    '"roof": {"shape": "gable|hip|flat|dome|spire|...", "material": "block"}, '
    '"materials": {"walls": "block", "trim": "block", "roof": "block", "floor": "block", '
    '"windows": "block", "door": "block", "foundation": "block"}, '
    '"openings": ["front door ...", "windows ..."], '
    '"distinctive_features": ["..."], "surroundings_to_omit": ["..."]')
#: The caption's own schema: only what an image of the outside can show.
_CAPTION_JSON = (
    '{"summary": "one sentence", "dimensions_blocks": [w, h, d], "footprint_blocks": [w, d], '
    '"storeys": {"count": n}, '
    '"roof": {"shape": "gable|hip|flat|dome|spire|...", "material": "block"}, '
    '"materials": {"walls": "block", "trim": "block", "roof": "block", '
    '"windows": "block", "door": "block", "foundation": "block"}, '
    '"openings": ["front door ...", "windows ..."], "distinctive_features": ["..."], '
    '"faces": {"front": "...", "side": "..."}, "surroundings_to_omit": ["..."], '
    '"uncertain": ["..."]}')
#: Values that mean "not shown"; normalised to empty, never compared.
_UNKNOWN = {"unknown", "unclear", "not visible", "none visible", "n/a", "na", "?", "-"}

#: Shuffled-control fallback: captions of other buildings, in the caption schema
#: and with palette block names only, so one reads like a real caption. Varied in
#: material, height and roof, and none is an archetype of the built-in prompt
#: sets; :func:`library_caption` also picks the one furthest from the request.
LIBRARY: Tuple[Dict[str, Any], ...] = (
    {"summary": "A low mossy-cobblestone crypt half sunk into the ground, with a flat roof.",
     "footprint_blocks": [9, 7], "storeys": {"count": 1},
     "roof": {"shape": "flat", "material": "stone_slab"},
     "materials": {"walls": "mossy_cobblestone", "trim": "cobblestone", "roof": "stone_slab",
                   "windows": "", "door": "", "foundation": "andesite"},
     "openings": ["an open stairway leading down at the front"],
     "distinctive_features": ["a lantern on each side of the stairway", "moss on every face"],
     "surroundings_to_omit": [], "dimensions_blocks": [9, 4, 7],
     "faces": {"front": "stairway cut into the wall, two lanterns", "side": "blank wall"},
     "uncertain": ["what is below ground"]},
    {"summary": "A low sandstone desert temple with a stepped pyramid roof.",
     "footprint_blocks": [15, 15], "storeys": {"count": 1},
     "roof": {"shape": "stepped pyramid", "material": "sandstone"},
     "materials": {"walls": "sandstone", "trim": "smooth_sandstone", "roof": "sandstone",
                   "windows": "", "door": "", "foundation": "smooth_sandstone"},
     "openings": ["wide open archway at the front"],
     "distinctive_features": ["orange_terracotta bands", "two corner towers"],
     "surroundings_to_omit": [], "dimensions_blocks": [15, 9, 15],
     "faces": {"front": "archway flanked by towers", "side": "plain wall, one band"},
     "uncertain": ["what is inside"]},
    {"summary": "A red-brick factory hall with a tall chimney.",
     "footprint_blocks": [18, 10], "storeys": {"count": 2},
     "roof": {"shape": "sawtooth", "material": "stone_slab"},
     "materials": {"walls": "bricks", "trim": "stone", "roof": "stone_slab",
                   "windows": "glass", "door": "oak_door", "foundation": "stone"},
     "openings": ["large double door on the long side", "a row of tall windows"],
     "distinctive_features": ["brick chimney 6 blocks above the roof"],
     "surroundings_to_omit": [], "dimensions_blocks": [18, 14, 10],
     "faces": {"front": "double door, six windows", "side": "chimney at the back"},
     "uncertain": ["roof pitch"]},
    {"summary": "A black-concrete office block with a grid of blue glass windows.",
     "footprint_blocks": [14, 10], "storeys": {"count": 3},
     "roof": {"shape": "flat", "material": "black_concrete"},
     "materials": {"walls": "black_concrete", "trim": "gray_concrete",
                   "roof": "black_concrete", "windows": "light_blue_stained_glass",
                   "door": "glass", "foundation": "gray_concrete"},
     "openings": ["glass entrance in the middle of the front", "a window every 2 blocks"],
     "distinctive_features": ["a sea_lantern strip over the entrance", "rooftop vents"],
     "surroundings_to_omit": [], "dimensions_blocks": [14, 13, 10],
     "faces": {"front": "entrance, three rows of windows", "side": "same window grid"},
     "uncertain": ["the roof edge"]},
    {"summary": "A dark netherrack gatehouse with an arched passage and obsidian trim.",
     "footprint_blocks": [13, 6], "storeys": {"count": 2},
     "roof": {"shape": "flat with spikes", "material": "netherrack"},
     "materials": {"walls": "netherrack", "trim": "obsidian", "roof": "netherrack",
                   "windows": "", "door": "", "foundation": "obsidian"},
     "openings": ["3-wide arched passage through the middle"],
     "distinctive_features": ["two square towers", "glowstone niches"],
     "surroundings_to_omit": [], "dimensions_blocks": [13, 12, 6],
     "faces": {"front": "arch between two towers", "side": "solid wall"},
     "uncertain": ["back face"]},
)


# ===========================================================================
# Config and state
# ===========================================================================
@dataclass(frozen=True)
class DescribeConfig:
    mode: str = "none"
    caption_format: str = "json"
    expand_into_image: bool = True
    show_spec: bool = True
    caption_keys: str = "primary"
    caption_in_match: bool = True
    caption_control: str = "none"

    @property
    def expands(self) -> bool:
        return self.mode in ("before", "both")

    @property
    def captions(self) -> bool:
        return self.mode in ("after", "both")


def config(describe: str = "none", caption_format: str = "json",
           expand_into_image: bool = True, show_spec: bool = True,
           caption_keys: str = "primary", caption_in_match: bool = True,
           caption_control: str = "none") -> Optional[DescribeConfig]:
    """Validated config, or None for ``describe: none``. ValueError before any call."""
    if describe not in MODES:
        raise ValueError(f"describe must be one of {MODES}, got {describe!r}")
    if caption_format not in FORMATS:
        raise ValueError(f"caption_format must be one of {FORMATS}, got {caption_format!r}")
    if caption_keys not in CAPTION_KEYS:
        raise ValueError(f"caption_keys must be one of {CAPTION_KEYS}, got {caption_keys!r}")
    if caption_control not in CONTROLS:
        raise ValueError(f"caption_control must be one of {CONTROLS}, got {caption_control!r}")
    if caption_control != "none" and describe not in ("after", "both"):
        raise ValueError("caption_control 'shuffled' needs describe: after or both "
                         "(it replaces the caption; without one there is nothing to shuffle)")
    if describe == "none":
        return None
    return DescribeConfig(describe, caption_format, bool(expand_into_image), bool(show_spec),
                          caption_keys, bool(caption_in_match), caption_control)


def config_from_params(params: Mapping[str, Any]) -> Optional[DescribeConfig]:
    """:func:`config` from a method's params mapping (other keys are ignored)."""
    return config(**{k: params[k] for k in PARAMS if k in params})


@dataclass
class DescribeState:
    cfg: DescribeConfig
    spec: Optional[Dict[str, Any]] = None       # normalized JSON spec
    spec_text: str = ""                         # what the builder is shown
    image_desc: str = ""                        # what the image prompt gets
    expanded_desc: str = ""                     # the {desc} expand() set on the dreams
    captions: Dict[str, Dict[str, Any]] = field(default_factory=dict)   # digest -> record
    latest: Dict[str, str] = field(default_factory=dict)                # key -> digest
    shown: Dict[str, Dict[str, Any]] = field(default_factory=dict)      # key -> shuffled record


def state(ep: Any) -> Optional[DescribeState]:
    st = (getattr(ep, "notes", None) or {}).get("describe")
    return st if isinstance(st, DescribeState) else None


# ===========================================================================
# Normalisation and rendering
# ===========================================================================
def _ints(x: Any, n: int) -> List[int]:
    vals = x if isinstance(x, (list, tuple)) else re.findall(r"\d+", str(x or ""))
    out = []
    for v in list(vals)[:n]:
        try:
            out.append(int(float(v)))
        except (TypeError, ValueError):
            continue
    return out


def _str(x: Any) -> str:
    """A stripped string; a "not shown" value (``unknown``, ``n/a``) becomes ''."""
    t = str(x or "").strip()
    return "" if t.lower().rstrip(".") in _UNKNOWN else t


def _strs(x: Any) -> List[str]:
    vals = x if isinstance(x, (list, tuple)) else [x]
    return [t for t in (_str(v) for v in vals) if t]


def normalize(raw: Mapping[str, Any], *, caption: bool = False) -> Dict[str, Any]:
    """Coerce a parsed spec / caption into the schema with safe defaults.

    Models write ``"2"`` for a count, a string for a list, or leave keys out;
    a missing or ``"unknown"`` field becomes empty, never a guess. A caption keeps
    only what an image of the outside shows: no floor material, no clear heights.
    """
    raw = dict(raw or {})
    st = raw.get("storeys")
    if not isinstance(st, Mapping):
        st = {"count": st}
    count = _ints([st.get("count")], 1)
    roof = raw.get("roof")
    if not isinstance(roof, Mapping):
        roof = {"shape": roof}
    mats = raw.get("materials") if isinstance(raw.get("materials"), Mapping) else {}
    storeys: Dict[str, Any] = {"count": count[0] if count else None}
    if not caption:
        storeys["clear_heights"] = _ints(st.get("clear_heights"), 8)
    slots = CAPTION_MATERIAL_SLOTS if caption else MATERIAL_SLOTS
    out: Dict[str, Any] = {
        "summary": _str(raw.get("summary")),
        "footprint_blocks": _ints(raw.get("footprint_blocks"), 2),
        "storeys": storeys,
        "roof": {"shape": _str(roof.get("shape")), "material": _str(roof.get("material"))},
        "materials": {s: _str(mats.get(s)) for s in slots},
        "openings": _strs(raw.get("openings")),
        "distinctive_features": _strs(raw.get("distinctive_features")),
        "surroundings_to_omit": _strs(raw.get("surroundings_to_omit")),
    }
    if caption:
        faces = raw.get("faces") if isinstance(raw.get("faces"), Mapping) else {}
        out["dimensions_blocks"] = _ints(raw.get("dimensions_blocks"), 3)
        out["faces"] = {str(k): _str(v) for k, v in faces.items() if _str(v)}
        out["uncertain"] = _strs(raw.get("uncertain"))
    return out


def render_fields(d: Mapping[str, Any]) -> str:
    """A normalized spec / caption as short labelled lines (what a model is shown)."""
    lines: List[str] = []
    if d.get("summary"):
        lines.append(f"summary: {d['summary']}")
    fp = d.get("footprint_blocks") or []
    if len(fp) == 2:
        lines.append(f"footprint: {fp[0]} x {fp[1]} blocks")
    dims = d.get("dimensions_blocks") or []
    if len(dims) == 3:
        lines.append(f"size: about {dims[0]} wide x {dims[1]} tall x {dims[2]} deep blocks")
    st = d.get("storeys") or {}
    if st.get("count"):
        hs = st.get("clear_heights") or []
        lines.append(f"storeys: {st['count']}" + (f" (clear heights {', '.join(map(str, hs))})"
                                                   if hs else ""))
    roof = d.get("roof") or {}
    if roof.get("shape") or roof.get("material"):
        lines.append("roof: " + ", ".join(v for v in (roof.get("shape"), roof.get("material"))
                                          if v))
    mats = [f"{s} {v}" for s, v in (d.get("materials") or {}).items() if v]
    if mats:
        lines.append("materials: " + "; ".join(mats))
    for k, label in (("openings", "openings"), ("distinctive_features", "features")):
        if d.get(k):
            lines.append(f"{label}: " + "; ".join(d[k]))
    for face, note in (d.get("faces") or {}).items():
        lines.append(f"{face} face: {note}")
    if d.get("uncertain"):
        lines.append("uncertain: " + "; ".join(d["uncertain"]))
    if d.get("surroundings_to_omit"):
        lines.append("leave out: " + "; ".join(d["surroundings_to_omit"]))
    return "\n".join(lines)


#: The spec fields a blueprint does not fix (see :func:`augment_desc`).
OPEN_FIELDS = ("openings", "distinctive_features", "surroundings_to_omit")


def image_desc(spec: Mapping[str, Any], *, max_chars: int = 700,
               fields: Optional[Sequence[str]] = None) -> str:
    """One compact description of a JSON spec for the image prompt's ``{desc}``.

    Block names lose their underscores (an image model reads "oak planks", not
    "oak_planks"); camera and style never appear (the template pins those).
    ``fields`` keeps only those spec keys (default: all of them).
    """
    def b(x: str) -> str:
        return x.replace("_", " ")

    if fields is not None:
        spec = {k: v for k, v in spec.items() if k in fields}
    bits: List[str] = []
    if spec.get("summary"):
        bits.append(str(spec["summary"]).rstrip("."))
    st = spec.get("storeys") or {}
    if st.get("count"):
        bits.append(f"{st['count']} storey{'s' if st['count'] != 1 else ''}")
    fp = spec.get("footprint_blocks") or []
    if len(fp) == 2:
        bits.append(f"footprint {fp[0]} by {fp[1]} blocks")
    roof = spec.get("roof") or {}
    if roof.get("shape"):
        bits.append(f"{roof['shape']} roof" + (f" of {b(roof['material'])}"
                                               if roof.get("material") else ""))
    mats = spec.get("materials") or {}
    mat = [f"{slot} of {b(mats[slot])}" for slot in ("walls", "trim", "foundation", "windows",
                                                    "door") if mats.get(slot)]
    if mat:
        bits.append(", ".join(mat))
    if spec.get("openings"):
        bits.append(b("; ".join(spec["openings"][:3])))
    if spec.get("distinctive_features"):
        bits.append(b("; ".join(spec["distinctive_features"][:3])))
    if spec.get("surroundings_to_omit"):
        bits.append("no " + ", no ".join(spec["surroundings_to_omit"][:3]))
    return _cut(". ".join(x.strip().rstrip(".") for x in bits if x.strip()), max_chars)


def _cut(text: str, max_chars: int) -> str:
    if len(text) > max_chars:           # cut at a word, never inside a block name
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;.")
    return text


_LABEL = re.compile(r"^\s*[-*]?\s*[A-Za-z][A-Za-z /()_-]{0,30}:\s+")


def prose_image_desc(text: str, *, max_chars: int = 700) -> str:
    """A prose spec for the image prompt: one line per point, leading field labels
    ("materials:", "Canvas:") dropped and block names without underscores, as
    :func:`image_desc` does for JSON."""
    bits = [_LABEL.sub("", ln).replace("_", " ").strip().rstrip(".")
            for ln in str(text or "").splitlines()]
    return _cut(". ".join(" ".join(b.split()) for b in bits if b.strip()), max_chars)


def _norm_block(x: Any) -> str:
    return re.sub(r"[\s-]+", "_", str(x or "").strip().lower()).removeprefix("minecraft:")


def disagreements(spec: Mapping[str, Any], cap: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Field-wise diff of a JSON spec and a JSON caption: the visible material slots
    (never the floor, which a caption does not name), storey count and roof shape,
    where BOTH sides name a value. Deliberately simple: a signal for the log and
    the builder, not a judge."""
    out: List[Dict[str, Any]] = []
    sm, cm = spec.get("materials") or {}, cap.get("materials") or {}
    for slot in CAPTION_MATERIAL_SLOTS:
        a, b = _norm_block(sm.get(slot)), _norm_block(cm.get(slot))
        if a and b and a != b:
            out.append({"field": f"materials.{slot}", "spec": sm.get(slot), "image": cm.get(slot)})
    a, b = (spec.get("storeys") or {}).get("count"), (cap.get("storeys") or {}).get("count")
    if a and b and a != b:
        out.append({"field": "storeys", "spec": a, "image": b})
    a, b = (spec.get("roof") or {}).get("shape"), (cap.get("roof") or {}).get("shape")
    if a and b and _norm_block(a) != _norm_block(b):
        out.append({"field": "roof.shape", "spec": a, "image": b})
    return out


# ===========================================================================
# Turns
# ===========================================================================
def _palette_names() -> str:
    from blockgen.agentic.blockstate import PALETTE
    return ", ".join(PALETTE)


def spec_turn(ep: Any, fmt: str) -> Any:
    """EXPAND: the request -> a detailed design spec (planner, text only)."""
    from blockgen.agentic.providers import text_message
    from blockgen.agentic.scale import scale_active
    from blockgen.agentic.turns import JSON_RULE

    w, h, d = ep.size
    lines = [f"{SPEC_TAG} for this Minecraft build request: {ep.prompt}",
             f"Canvas: {w} x {h} x {d} blocks (x by y-up by z); the building must fit inside it.",
             "",
             "Write the building out in detail before anyone draws or builds it. Decide every "
             "choice the request leaves open, and keep it ONE building that can be built from "
             "full cube blocks at this size.",
             "Name every material with a Minecraft block name from the palette in your "
             "instructions."]
    if scale_active(ep):
        lines.append("Size it for a Minecraft player (2 blocks tall): doors 1 x 2, at least 3 "
                     "blocks of clear ceiling per storey (4-5 floor-to-floor), rooms at least "
                     "3 x 3 inside.")
    if fmt == "json":
        lines += ["", "Reply as JSON with exactly these keys:", _SPEC_JSON + "}", JSON_RULE]
    else:
        lines += ["", "Answer in at most 12 short lines covering: summary; footprint in blocks; "
                  "storeys and their clear heights; roof shape and material; materials for "
                  "walls, trim, roof, floor, windows, door and foundation; openings; "
                  "distinctive features; surroundings to leave out.", "Plain text only."]
    return text_message("user", "\n".join(lines))


def caption_turn(ep: Any, key: str, png: bytes, fmt: str) -> Any:
    """CAPTION: what one dream image shows (judge, vision)."""
    from blockgen.agentic.providers import Message, Part
    from blockgen.agentic.turns import JSON_RULE

    w, h, d = ep.size
    ds = ep.dream_spec(key)
    # The request is left out on purpose: the caption describes the image alone, so
    # it cannot lean towards the request, and a caption shown under the shuffled
    # control cannot be matched against the request it came from.
    head = (f"{CAPTION_TAG} below. It is the target image of a Minecraft build. "
            f"Canvas: {w} x {h} x {d} blocks.\n\n"
            "A builder will copy the image using your description, so describe ONLY what "
            "the image shows. Do not invent hidden parts: a field the image does not show "
            "(the back, the inside) is \"unknown\" and goes under 'uncertain'. "
            "Estimate sizes in blocks (a door is 1 x 2 blocks, a storey 4-5 blocks "
            "floor-to-floor). Name materials with the closest Minecraft block names from: "
            f"{_palette_names()}.")
    if fmt == "json":
        tail = ("Reply as JSON with exactly these keys:\n" + _CAPTION_JSON
                + '\n("surroundings_to_omit": anything around the building in the image '
                "that is not part of it.)\n" + JSON_RULE)
    else:
        tail = ("Answer in at most 14 short lines covering: summary; size in blocks (wide, "
                "tall, deep); number of storeys; roof shape and material; materials for "
                "walls, trim, roof, windows, door and foundation; openings; distinctive "
                "features; the front face; a side face; what is uncertain. Write \"unknown\" "
                "for anything the image does not show.\nPlain text only.")
    return Message("user", [Part("text", text=head),
                            Part("text", text=f"IMAGE ({key}, view {ds.view or 'any'}):"),
                            Part("image", image_png=png), Part("text", text=tail)])


# ===========================================================================
# Steps
# ===========================================================================
def _root(ep: Any) -> Optional[Path]:
    root = getattr(getattr(ep, "trace", None), "root", None)
    return Path(root) if root is not None else None


def _write(ep: Any, rel: str, text: str) -> None:
    root = _root(ep)
    if root is None:
        return
    try:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:          # a trace write must not cost the build
        ep.trace.event("describe_write_error", file=rel, error=repr(exc))


def setup(ep: Any, cfg: Optional[DescribeConfig], *,
          extra_keys: Sequence[str] = ()) -> Optional[DescribeState]:
    """Attach the stage to ``ep`` and, for before/both, write the spec now (before
    any dream). No-op (None) for ``describe: none``. Idempotent per episode."""
    if cfg is None:
        return None
    st = state(ep)
    if st is not None:
        return st
    st = DescribeState(cfg)
    ep.notes["describe"] = st
    if cfg.expands:
        expand(ep, st, extra_keys=extra_keys)
    return st


def expand(ep: Any, st: DescribeState, *, extra_keys: Sequence[str] = ()) -> None:
    """EXPAND: one planner turn; then (expand_into_image) every recipe dream's
    ``{desc}`` becomes the request plus the spec's description."""
    fmt = st.cfg.caption_format
    raw = ep.ask(spec_turn(ep, fmt), role="planner", history=False,
                 json_reply=(fmt == "json"), stage="describe_spec")
    if fmt == "json":
        if not raw:
            ep.trace.event("describe_spec_unparsed")
            return
        st.spec = normalize(raw)
        st.spec_text = render_fields(st.spec)
        st.image_desc = image_desc(st.spec)
    else:
        st.spec_text = str(raw or "").strip()[:2000]
        st.image_desc = prose_image_desc(st.spec_text)
    if not st.spec_text:
        ep.trace.event("describe_spec_unparsed")
        return
    if st.cfg.expand_into_image and st.image_desc:
        desc = f"{ep.prompt.strip().rstrip('.')}. {st.image_desc}"
        st.expanded_desc = desc
        for k in [d.key for d in ep.kit.dreams] + list(extra_keys):
            ep.dream_desc[k] = desc
    blob = {"format": fmt, "spec": st.spec, "text": st.spec_text, "image_desc": st.image_desc,
            "expand_into_image": st.cfg.expand_into_image}
    _write(ep, "dreams/spec.json", json.dumps(blob, indent=1))
    ep.trace.event("describe_spec", format=fmt, chars=len(st.spec_text),
                   text=st.spec_text[:2000], image_desc=st.image_desc)
    write_md(ep)


def augment_desc(ep: Any, desc: str, *, blueprint: bool = False) -> str:
    """``desc`` plus the spec's image description (for a method that writes its
    own dream description, e.g. the staged exterior); unchanged when not expanding.

    ``blueprint``: ``desc`` already fixes storeys, roof and materials (a staged
    method's blueprint, which never saw the spec), so only the spec's
    :data:`OPEN_FIELDS` are appended; a summary or storey count from the spec
    would contradict it. A prose spec cannot be split, so nothing is appended.
    """
    st = state(ep)
    if st is None or not st.cfg.expand_into_image or not st.image_desc:
        return desc
    extra = st.image_desc
    if blueprint:
        extra = image_desc(st.spec, fields=OPEN_FIELDS) if st.spec else ""
        ep.trace.event("describe_spec_partial", fields=list(OPEN_FIELDS) if extra else [],
                       reason="blueprint fixes storeys, roof and materials")
        if not extra:
            return desc
    return f"{desc.strip().rstrip('.')}. {extra}"


def expanded_desc(ep: Any) -> str:
    """The ``{desc}`` the expand step set on the recipe dreams ('' if none)."""
    st = state(ep)
    return st.expanded_desc if st is not None else ""


def caption_keys(ep: Any, st: DescribeState) -> List[str]:
    from blockgen.agentic import dreams as dreams_mod
    scored = [d for d in ep.kit.dreams if d.view and ep.latest_dream(d.key) is not None]
    if st.cfg.caption_keys == "all":
        return [d.key for d in scored]
    prim = dreams_mod.primary(ep.kit.dreams)
    return [prim.key] if prim is not None and ep.latest_dream(prim.key) is not None else []


def refresh(ep: Any) -> None:
    """Caption every caption key whose latest image has no caption yet. Cheap when
    nothing changed (a digest lookup); a no-op without the stage."""
    st = state(ep)
    if st is None or not st.cfg.captions:
        return
    for key in caption_keys(ep, st):
        dv = ep.latest_dream(key)
        if dv is None:
            continue
        rec = st.captions.get(dv.digest)
        if rec is None:
            rec = caption(ep, st, key, dv)
        elif rec["key"] != key or rec["version"] != dv.version:
            # The same image under another key or version: reuse, but keep a file
            # next to this version too.
            rec = {**rec, "key": key, "version": dv.version}
            _write(ep, f"dreams/{_slug(key)}_v{dv.version}.caption.json", json.dumps(rec, indent=1))
        st.latest[key] = dv.digest
        if st.cfg.caption_control == "shuffled" and key not in st.shown:
            st.shown[key] = _shuffled(ep, key, rec)
    write_md(ep)


def _slug(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key)


def caption(ep: Any, st: DescribeState, key: str, dv: Any) -> Dict[str, Any]:
    """CAPTION one dream version (judge). Stored and cached on the image digest."""
    fmt = st.cfg.caption_format
    raw = ep.ask(caption_turn(ep, key, dv.png, fmt), role="judge", history=False,
                 json_reply=(fmt == "json"), stage=f"describe_caption:{key}")
    cap: Optional[Dict[str, Any]] = None
    if fmt == "json":
        cap = normalize(raw, caption=True) if raw else None
        text = render_fields(cap) if cap else ""
    else:
        text = str(raw or "").strip()[:2000]
    if not text:
        ep.trace.event("describe_caption_unparsed", key=key, version=dv.version)
    diffs = disagreements(st.spec, cap) if (st.spec and cap) else []
    if st.cfg.mode == "both" and fmt != "json":
        ep.trace.event("describe_disagreement_skipped", key=key, version=dv.version,
                       reason="caption_format prose: disagreements need json")
    rec = {"key": key, "version": dv.version, "digest": dv.digest, "format": fmt,
           "caption": cap, "text": text, "disagreements": diffs}
    st.captions[dv.digest] = rec
    _write(ep, f"dreams/{_slug(key)}_v{dv.version}.caption.json", json.dumps(rec, indent=1))
    ep.trace.event("describe_caption", key=key, version=dv.version, digest=dv.digest,
                   chars=len(text), text=text[:2000])
    if diffs:
        ep.trace.event("describe_disagreement", key=key, version=dv.version, fields=diffs)
    _pool_add(ep, key, rec)
    return rec


# ---------------------------------------------------------------- shuffled control
#: Per arm (keyed on the arm-level ledger, which every episode of an arm shares):
#: key -> [(prompt, caption record)] in the order the arm captioned them.
_POOLS: "weakref.WeakKeyDictionary[Any, Dict[str, List[Tuple[str, Dict[str, Any]]]]]" = \
    weakref.WeakKeyDictionary()


def _pool(ep: Any) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    led = getattr(getattr(ep, "kit", None), "ledger", None)
    try:
        return _POOLS.setdefault(led, {})
    except TypeError:               # no weak-referenceable arm handle: per episode
        return ep.notes.setdefault("_describe_pool", {})


def _pool_add(ep: Any, key: str, rec: Dict[str, Any]) -> None:
    st = state(ep)
    if st is None or st.cfg.caption_control != "shuffled" or not rec.get("text"):
        return
    entries = _pool(ep).setdefault(key, [])
    if not any(p == ep.prompt for p, _ in entries):      # one entry per prompt (its first)
        entries.append((ep.prompt, rec))


def _words(text: str) -> set:
    """Content words (4+ letters, a plural 's' dropped) for the library pick."""
    return {w[:-1] if w.endswith("s") and len(w) > 4 else w
            for w in re.findall(r"[a-z]{4,}", text.lower().replace("_", " "))}


def library_caption(prompt: str) -> Dict[str, Any]:
    """The fixed-library caption for ``prompt``: the entry sharing the fewest words
    with it (so the 'wrong' caption is never close to right), ties broken by a
    hash of the prompt. Deterministic."""
    want = _words(prompt)
    start = int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % len(LIBRARY)
    order = [(start + j) % len(LIBRARY) for j in range(len(LIBRARY))]
    i = min(order, key=lambda k: len(want & _words(render_fields(
        normalize(LIBRARY[k], caption=True)))))
    cap = normalize(LIBRARY[i], caption=True)
    return {"caption": cap, "text": render_fields(cap), "source": f"library:{i}"}


def _shuffled(ep: Any, key: str, own: Dict[str, Any]) -> Dict[str, Any]:
    """The caption shown in place of ``own``: the latest one from another prompt
    of this arm, else a library caption. Picked once per episode and key."""
    others = [(p, r) for p, r in _pool(ep).get(key, []) if p != ep.prompt and r.get("text")]
    if others:
        p, r = others[-1]
        out = {"caption": r.get("caption"), "text": r["text"], "source": f"prompt:{p}"}
    else:
        out = library_caption(ep.prompt)
    ep.trace.event("describe_shuffled", key=key, source=out["source"],
                   own_digest=own.get("digest"))
    return out


# ===========================================================================
# What the turns show
# ===========================================================================
def shown_captions(ep: Any) -> List[Tuple[str, str]]:
    """(key, caption text) as the builder and critic see them (shuffled under the
    control), for the caption keys in order; empty captions are skipped."""
    st = state(ep)
    if st is None or not st.cfg.captions:
        return []
    out = []
    for key, digest in st.latest.items():
        rec = (st.shown.get(key) if st.cfg.caption_control == "shuffled"
               else st.captions.get(digest))
        if rec and rec.get("text"):
            out.append((key, rec["text"]))
    return out


def _caption_heading(key: str, many: bool) -> str:
    return (f"TARGET DESCRIPTION{' ' + key if many else ''} (what the TARGET image shows, "
            "read off the image):")


def builder_block(ep: Any) -> str:
    """Text for the build (and plan) turn: the spec, the caption(s), and which one
    wins. '' without the stage, so the turn stays byte-identical."""
    st = state(ep)
    if st is None:
        return ""
    out: List[str] = []
    spec = bool(st.cfg.show_spec and st.spec_text)
    if spec:
        out.append("DESIGN SPEC (written from the request before the TARGET image was "
                   "made):\n" + st.spec_text)
    caps = shown_captions(ep)
    for key, text in caps:
        out.append(_caption_heading(key, len(caps) > 1) + "\n" + text)
    if spec and caps:
        note = ("Where the DESIGN SPEC and the TARGET DESCRIPTION disagree, follow the TARGET "
                "image and its description: your build is compared with the image, not "
                "with the spec.")
        if st.cfg.caption_control == "none":
            diffs = [f"{d['field']} (spec {d['spec']}, image {d['image']})"
                     for k in st.latest for d in (st.captions[st.latest[k]].get("disagreements")
                                                  or [])]
            if diffs:
                note += " They disagree on: " + "; ".join(diffs) + "."
        out.append(note)
    elif spec and any(ep.latest_dream(d.key) is not None for d in ep.kit.dreams):
        if st.cfg.expand_into_image and st.image_desc:
            out.append("The TARGET image was drawn from this spec; where they differ, match "
                       "the image.")
        else:
            out.append("This spec was written separately from the TARGET image; where they "
                       "differ, match the image.")
    return "\n\n".join(out)


def match_block(ep: Any) -> str:
    """Text for a compare turn: the caption(s), when ``caption_in_match``."""
    st = state(ep)
    if st is None or not st.cfg.caption_in_match:
        return ""
    caps = shown_captions(ep)
    return "\n\n".join(_caption_heading(k, len(caps) > 1) + "\n" + t for k, t in caps)


def write_md(ep: Any) -> None:
    """``describe.md`` in the build's trace: the spec, every caption, what was shown."""
    st = state(ep)
    if st is None or _root(ep) is None:
        return
    c = st.cfg
    lines = ["# Describe stage", "", f"Request: {ep.prompt}", "",
             f"Mode `{c.mode}`, format `{c.caption_format}`, caption keys `{c.caption_keys}`, "
             f"control `{c.caption_control}`, expand_into_image `{c.expand_into_image}`, "
             f"show_spec `{c.show_spec}`, caption_in_match `{c.caption_in_match}`.", ""]
    if c.expands:
        lines += ["## Design spec (before the image)", "", "```text",
                  st.spec_text or "(none: unparsed)", "```", ""]
        if c.expand_into_image and st.image_desc:
            lines += [f"Image prompt description: {st.image_desc}", ""]
    if c.captions:
        lines += ["## Captions (after the image)", ""]
        for rec in st.captions.values():
            lines += [f"### {rec['key']} v{rec['version']} (image {rec['digest']})", "",
                      "```text", rec["text"] or "(none: unparsed)", "```", ""]
            for dd in rec.get("disagreements") or []:
                lines.append(f"- disagrees with the spec on {dd['field']}: spec {dd['spec']}, "
                             f"image {dd['image']}")
            if rec.get("disagreements"):
                lines.append("")
        for key, rec in st.shown.items():
            lines += [f"### Shown for {key} under the shuffled control ({rec['source']})", "",
                      "```text", rec["text"], "```", ""]
    _write(ep, "describe.md", "\n".join(lines))


# ===========================================================================
# Mock replies (providers_cli.AutoMockProvider)
# ===========================================================================
_CANVAS = re.compile(r"Canvas: (\d+) x (\d+) x (\d+) blocks")
_REQUEST = re.compile(r"build request: (.+?)(?:\n|$)")


def mock_reply(turn: str) -> Optional[Tuple[str, str]]:
    """(decision, reply) for a spec or caption turn, else None. JSON when the turn
    ends with the JSON rule, else the same content as labelled prose."""
    from blockgen.agentic.turns import JSON_RULE

    if SPEC_TAG in turn:
        kind = "spec"
    elif CAPTION_TAG in turn:
        kind = "caption"
    else:
        return None
    m = _CANVAS.search(turn)
    w, h, d = (int(v) for v in m.groups()) if m else (32, 32, 32)
    fw, fd = max(5, min(12, w - 2)), max(5, min(10, d - 2))
    # The spec echoes the request; the caption turn does not carry one.
    m = _REQUEST.search(turn)
    req = m.group(1).strip().rstrip(".") if m else ""
    summary = "a one-storey oak cottage with a gable roof and a stone foundation."
    out: Dict[str, Any] = {
        "summary": f"{req}: {summary}" if req else summary[0].upper() + summary[1:],
        "footprint_blocks": [fw, fd], "storeys": {"count": 1, "clear_heights": [3]},
        "roof": {"shape": "gable", "material": "spruce_stairs"},
        "materials": {"walls": "oak_planks", "trim": "oak_log", "roof": "spruce_stairs",
                      "floor": "spruce_planks", "windows": "glass_pane", "door": "oak_door",
                      "foundation": "cobblestone"},
        "openings": ["front door centred on the long side", "two windows per long side"],
        "distinctive_features": ["log corner posts", "a chimney at the back"],
        "surroundings_to_omit": ["garden", "path"]}
    if kind == "caption":
        out["materials"]["walls"] = "stone_bricks"        # so 'both' logs a disagreement
        del out["materials"]["floor"]                     # the caption schema has neither
        out["storeys"] = {"count": 1}
        out.update({"dimensions_blocks": [fw, 8, fd],
                    "faces": {"front": "door centred, a window each side",
                              "side": "one window under the gable"},
                    "uncertain": ["the back face"], "surroundings_to_omit": []})
    if turn.rstrip().endswith(JSON_RULE):
        return f"describe:{kind}", json.dumps(out)
    return f"describe:{kind}", render_fields(normalize(out, caption=(kind == "caption")))


__all__ = ["CAPTION_KEYS", "CAPTION_MATERIAL_SLOTS", "CAPTION_TAG", "CONTROLS",
           "DescribeConfig", "DescribeState", "FORMATS", "LIBRARY", "MATERIAL_SLOTS", "MODES",
           "OPEN_FIELDS", "PARAMS", "SPEC_KEYS", "SPEC_TAG", "augment_desc", "builder_block",
           "caption", "caption_turn", "config", "config_from_params", "disagreements",
           "expand", "expanded_desc", "image_desc", "library_caption", "match_block",
           "mock_reply", "normalize", "prose_image_desc", "refresh", "render_fields", "setup",
           "shown_captions", "spec_turn", "state", "write_md"]
