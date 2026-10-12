"""Reusable steps for the staged methods (``blueprint``, ``floors``).

A staged method builds a house in units (a room, or a storey) instead of all at
once. These steps are the vocabulary it is written in; like :mod:`steps`, each one
uses only Episode verbs, and each encodes a decision we want made the same way in
every staged method:

* **A typed plan first.** :func:`make_blueprint` turns model text (or a dreamed
  plan image) into a :class:`~blockgen.agentic.blueprint.Blueprint` and always
  repairs it, so nothing downstream ever builds from a plan that overlaps itself,
  leaves the canvas or has a room nobody can walk into. Bad model text falls back
  to :func:`~blockgen.agentic.blueprint.default_blueprint`; it never raises.
* **Render-anchored dreams.** An image EDIT keeps the camera and composition of
  the image it starts from (pilot finding P2). So the exterior dream starts from an
  iso render of our own massing (:func:`massing_render`), each room dream from a
  cutaway render of the empty room (:func:`unit_dream`, ref ``cutaway`` first),
  and the loop's score then measures content, not camera or framing (P1).
* **Unit-level scores and accepts.** An interior change barely moves an exterior
  score, so a room is scored on a cutaway of that room only (:func:`unit_score`)
  and accepted by its own rule (:func:`unit_accept`), passed to ``try_`` per call.
* **Region-scoped writes.** A unit's turns run with ``try_(region=...)``: writes
  outside the unit are reverted and reported line by line, for any action space.
* **Stage budgets.** :func:`stage` gives each stage a slice of the build's usd cap,
  so one greedy room cannot starve the roof; a stage that runs out ends quietly and
  the build goes on.
"""

from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence,
                    Tuple, Union)

import numpy as np

from blockgen.agentic import describe as describe_mod
from blockgen.agentic import dreams as dreams_mod
from blockgen.agentic import turns, turns_staged
from blockgen.agentic.blueprint import (Blueprint, BlueprintError, default_blueprint,
                                        order_rooms, render_grid, render_schematic,
                                        render_schematics)
from blockgen.agentic.core import (AcceptFn, Attempt, Budget, BudgetExceeded, DreamSpec,
                                   DreamVersion, Ledger, Lowered, ScoreTerm)
from blockgen.agentic.providers import Message, Part, text_message
from blockgen.agentic.recipes import steps
from blockgen.agentic.region import Box, Region, region_structure

#: The unit score's terms when a recipe gives none. Silhouette and colour survive
#: the dream/render domain gap; embed carries content. Under ``--mock`` embed runs
#: as ``pixel`` (no model weights), see :func:`unit_terms`.
DEFAULT_UNIT_TERMS: Dict[str, float] = {"silhouette": 1.0, "color_lab": 0.5, "embed": 1.0}

#: Allowed reference names for a unit dream (:func:`unit_dream`).
UNIT_REFS: Tuple[str, ...] = ("cutaway", "exterior", "below", "schematic")
EXTERIOR_REFS: Tuple[str, ...] = ("massing", "schematic", "none")
PLAN_SOURCES: Tuple[str, ...] = ("llm", "image", "both", "text")
#: Plan sources that make an image (the others are text only).
IMAGE_PLAN_SOURCES: Tuple[str, ...] = ("image", "both")
FEEDBACKS: Tuple[str, ...] = ("raw", "score", "vdm", "both")

#: Appended to the exterior dream prompt when it starts from our massing render.
MASSING_NOTE = ("The reference image is a plain block massing of this exact building: "
                "keep its footprint, wall height, roof line and camera exactly, and add "
                "the doors, windows, trim and materials.")

#: Views the no-dream final critique looks at (same pair as the score fallback).
CRITIQUE_VIEWS: Tuple[str, ...] = ("iso", "iso_back")


@dataclass
class Unit:
    """One stage target: a room (blueprint) or a storey (floors)."""

    name: str                       # slug; render/dream key = f"u_{name}"
    kind: str                       # 'room' | 'storey'
    storey: int
    region: Region                  # where writes are allowed (deny = keep_clear)
    view_box: Box                   # what its cutaway render shows
    cut_y: Optional[int]
    purpose: str
    desc: str

    @property
    def key(self) -> str:
        return f"u_{self.name}"


# ===========================================================================
# Stages and dialogue
# ===========================================================================
def _root_ledger(ep: Any) -> Ledger:
    """The build's own ledger (the one run_episode gave the Episode), remembered on
    first use so nested stages divide the BUILD's cap, not their parent stage's."""
    root = ep.notes.get("_stage_root_ledger")
    if root is None:
        root = ep.ledger
        ep.notes["_stage_root_ledger"] = root
    return root


@contextmanager
def stage(ep: Any, name: str, frac: Optional[float], *,
          usd: Optional[float] = None) -> Iterator[Ledger]:
    """Run a block of a method on a child ledger capped at ``frac`` of the build's
    usd cap (or at ``usd`` if given; no cap if neither applies).

    Why: a stage that overspends (a room whose turns keep failing) must not starve
    the stages after it. A BudgetExceeded raised by THIS stage's ledger ends the
    stage quietly (event ``stage_budget_stop``) and the build goes on; one raised by
    a parent (the build itself is out of money) propagates to run_episode.
    Yields the stage ledger, so a caller can split what is left among sub-stages.
    """
    parent = ep.ledger
    cap: Optional[float] = None
    if usd is not None:
        cap = max(0.0, float(usd))
    elif frac is not None:
        root_cap = _root_ledger(ep).budget.usd
        if root_cap is not None:
            cap = max(0.0, float(frac) * float(root_cap))
    else:
        _root_ledger(ep)
    child = parent.child(Budget(usd=cap), name=f"stage:{name}")
    ep.ledger = child
    ep.trace.event("stage_start", stage=name, cap_usd=cap)
    try:
        yield child
    except BudgetExceeded as exc:
        if f"ledger '{child.name}'" not in str(exc):
            raise
        ep.trace.event("stage_budget_stop", stage=name, error=str(exc))
    finally:
        ep.ledger = parent
        ep.trace.event("stage_end", stage=name, spent_usd=round(child.spent_usd, 6),
                       would_cost_usd=round(child.would_cost_usd, 6),
                       llm_calls=child.n_llm_calls, images=child.n_images)


def is_image_cap(exc: BaseException) -> bool:
    """True for a BudgetExceeded raised by an image-count cap (``max_images``).

    Why the distinction: running out of IMAGES only means the rest of the build
    goes on from text (a unit without its dream), so the dream steps catch it.
    Running out of MONEY must still end the stage (:func:`stage`) or the build
    (run_episode), so those errors are never swallowed here.
    """
    return isinstance(exc, BudgetExceeded) and "image cap" in str(exc)


def fresh_dialogue(ep: Any) -> None:
    """Start a stage with an empty dialogue, so a room's turns never carry earlier
    rooms' images (they cost tokens and show the wrong target)."""
    ep.history.clear()


def _always(prev: Optional[Attempt], cand: Attempt) -> Tuple[bool, str]:
    return True, "deterministic stage"


def _nonempty(prev: Optional[Attempt], cand: Attempt) -> Tuple[bool, str]:
    if cand.blocks == 0:
        return False, "empty build"
    return True, "non-empty"


def _style_text(name: str) -> str:
    """A named dream style (``dreams.STYLES``) or the text itself."""
    return str(getattr(dreams_mod, "STYLES", {}).get(name, name))


def _primary_scored(ep: Any) -> Optional[DreamSpec]:
    p = dreams_mod.primary(ep.kit.dreams)
    return p if p is not None and p.view else None


def _primary_png(ep: Any) -> Optional[bytes]:
    p = _primary_scored(ep)
    dv = ep.latest_dream(p.key) if p is not None else None
    return dv.png if dv is not None else None


# ===========================================================================
# Plan
# ===========================================================================
def record_blueprint(ep: Any, bp: Blueprint, *, tag: str) -> None:
    """Trace the plan (JSON with its repair notes) and keep its schematic as the
    unscored runtime dream ``blueprint``, so the contact sheet shows the plan the
    build followed. ``ep.notes['blueprint']`` is what later stages read."""
    ep.trace.event("blueprint", tag=tag, blueprint=bp.to_dict(notes=True))
    ep.register_dream(DreamSpec("blueprint", "", view=None, weight=0.0, show_builder=False))
    try:
        png = render_schematics(bp)
    except Exception as exc:  # noqa: BLE001 - a drawing bug must not cost the build
        ep.trace.event("schematic_error", error=repr(exc))
        png = None
    if png:
        ep.put_dream("blueprint", png, prompt=f"schematic ({tag})", generator="schematic")
    ep.notes["blueprint"] = bp


def _parse(raw: Any) -> Optional[Blueprint]:
    if not isinstance(raw, Mapping) or not raw:
        return None
    try:
        return Blueprint.from_dict(raw)
    except (BlueprintError, ValueError, TypeError, KeyError):
        return None


def _parse_and_fix(ep: Any, raw: Any, *, retries: int, max_rooms: int, max_storeys: int,
                   stage_name: str, brief: str = "") -> Blueprint:
    """Parse ``raw``; while it is unparseable or :meth:`Blueprint.validate` lists
    problems, show the planner its JSON and the problems (up to ``retries`` times);
    then repair. Unparseable to the end -> :func:`default_blueprint`."""
    size = tuple(ep.size)
    bp = _parse(raw)
    for i in range(max(0, int(retries))):
        if bp is None:
            problems = ["the reply held no usable blueprint JSON object (no footprint and "
                        "no rooms)"]
        else:
            problems = bp.validate(size)
            if not problems:
                break
        ep.trace.event("blueprint_problems", stage=stage_name, problems=list(problems)[:20])
        fixed = ep.ask(turns_staged.blueprint_fix(ep, raw if isinstance(raw, Mapping) else {},
                                                  problems, brief=brief),
                       role="planner", history=False, json_reply=True,
                       stage=f"{stage_name}_fix{i + 1}")
        cand = _parse(fixed)
        if cand is not None:
            bp, raw = cand, fixed
    if bp is None:
        ep.trace.event("blueprint_fallback", stage=stage_name)
        bp = default_blueprint(size, ep.prompt)
    return bp.repaired(size, max_rooms=max_rooms, max_storeys=max_storeys)


def blueprint_from_llm(ep: Any, *, max_rooms: int, max_storeys: int, retries: int) -> Blueprint:
    """Planner writes the blueprint as JSON (one fix turn on problems), repaired."""
    raw = ep.ask(turns_staged.blueprint_json(ep, max_rooms=max_rooms, max_storeys=max_storeys),
                 role="planner", history=False, json_reply=True, stage="plan")
    return _parse_and_fix(ep, raw, retries=retries, max_rooms=max_rooms,
                          max_storeys=max_storeys, stage_name="plan")


def blueprint_from_text(ep: Any, *, max_rooms: int, max_storeys: int,
                        retries: int) -> Blueprint:
    """plan_source 'text': a prose design brief first, then the JSON turn
    transcribes it (same parse, fix and repair as 'llm').

    The brief is kept as ``ep.plan`` (so ``result.plan_text`` and the run's
    ``*_plan.txt`` hold it), in ``ep.notes['design_brief']`` and in the trace as
    ``plan_brief.txt``. An empty brief falls back to the plain JSON turn.
    """
    brief = str(ep.ask(turns_staged.design_brief(ep, max_rooms=max_rooms,
                                                 max_storeys=max_storeys),
                       role="planner", history=False, stage="plan_brief") or "").strip()
    ep.notes["design_brief"] = brief
    ep.plan = brief
    steps.record_text(ep, "plan_brief.txt", brief, kind="design_brief")
    raw = ep.ask(turns_staged.blueprint_json(ep, max_rooms=max_rooms, max_storeys=max_storeys,
                                             brief=brief),
                 role="planner", history=False, json_reply=True, stage="plan")
    return _parse_and_fix(ep, raw, retries=retries, max_rooms=max_rooms,
                          max_storeys=max_storeys, stage_name="plan", brief=brief)


def blueprint_from_image(ep: Any, *, max_rooms: int, max_storeys: int,
                         retries: int = 1) -> Blueprint:
    """Dream a plan ON our blank grid paper, then a VLM reads it into JSON.

    Why the grid is the reference: an edit keeps its reference's composition, so a
    plan drawn on the grid inherits the axis labels the reader needs (pilot P2: a
    plan edited from the iso exterior was just another exterior). Only the ground
    floor comes from the image; upper storeys are the reader's proposal.
    """
    size = tuple(ep.size)
    grid = render_grid(size)
    ds = DreamSpec("plan_sketch", turns_staged.PLAN_DREAM, view=None, weight=0.0,
                   show_builder=False)
    ep.register_dream(ds)
    prompt = dreams_mod.render_prompt(ds, ep, desc=ep.prompt)
    dv = ep.dream("plan_sketch", prompt=prompt, refs=[grid],
                  seed=steps.next_seed(ep, "plan_sketch"), view_hint="top")
    raw = ep.ask(turns_staged.plan_extract(ep, grid, dv.png, max_rooms=max_rooms,
                                           max_storeys=max_storeys),
                 role="planner", history=False, json_reply=True, stage="plan_extract")
    return _parse_and_fix(ep, raw, retries=retries, max_rooms=max_rooms,
                          max_storeys=max_storeys, stage_name="plan_extract")


def verify_with_plan_dream(ep: Any, bp: Blueprint, *, max_rooms: Optional[int] = None,
                           max_storeys: Optional[int] = None) -> Blueprint:
    """Picture the plan (its schematic stylized as a top-down cutaway) and let the
    dream critic correct it.

    A correction is taken only if it repairs with no more notes than the original:
    a critic that returns a broken plan must not replace a clean one.
    """
    size = tuple(ep.size)
    ds = DreamSpec("plan_dream", turns_staged.PLAN_STYLIZE, view=None, weight=0.0,
                   show_builder=False)
    ep.register_dream(ds)
    sch = render_schematic(bp, 0)
    prompt = dreams_mod.render_prompt(ds, ep, style=_style_text("render_match"))
    dv = ep.dream("plan_dream", prompt=prompt, refs=[sch],
                  seed=steps.next_seed(ep, "plan_dream"), view_hint="top")
    v = ep.ask(turns_staged.plan_verify(ep, bp, sch, dv.png), role="dream_critic",
               history=False, json_reply=True, stage="plan_verify")
    corrected = v.get("blueprint") if isinstance(v, Mapping) else None
    if not isinstance(corrected, Mapping) or not corrected:
        ep.trace.event("plan_verify", ok=bool(v.get("ok", True)) if v else None,
                       issues=list(v.get("issues") or []) if v else [], taken=False)
        return bp
    cand = _parse(corrected)
    taken = False
    if cand is not None:
        cand = cand.repaired(size, max_rooms=max_rooms, max_storeys=max_storeys)
        taken = len(cand.notes) <= len(bp.notes)
    ep.trace.event("plan_verify", ok=bool(v.get("ok", False)),
                   issues=list(v.get("issues") or []), taken=taken)
    return cand if taken and cand is not None else bp


def make_blueprint(ep: Any, *, source: str, max_rooms: int, max_storeys: int,
                   retries: int = 1) -> Blueprint:
    """``llm`` | ``image`` | ``both`` (llm, then verified against a plan dream) |
    ``text`` (a prose design brief, then JSON transcribed from it). Never raises on bad model text (:func:`default_blueprint` instead); budget
    errors propagate, as everywhere."""
    if source not in PLAN_SOURCES:
        raise ValueError(f"plan source must be one of {PLAN_SOURCES}, got {source!r}")
    if source == "image":
        return blueprint_from_image(ep, max_rooms=max_rooms, max_storeys=max_storeys,
                                    retries=retries)
    if source == "text":
        return blueprint_from_text(ep, max_rooms=max_rooms, max_storeys=max_storeys,
                                   retries=retries)
    bp = blueprint_from_llm(ep, max_rooms=max_rooms, max_storeys=max_storeys, retries=retries)
    if source == "both":
        bp = verify_with_plan_dream(ep, bp, max_rooms=max_rooms, max_storeys=max_storeys)
    return bp


# ===========================================================================
# Shell, roof, exterior dream
# ===========================================================================
def _ask_repair(ep: Any, turn: Message, reply: str, fix: Message, stage_name: str) -> str:
    """One repair turn whose history is exactly (turn, reply): the stage's own
    exchange, whether or not it was accepted."""
    return ep.ask(fix, history=[turn, text_message("assistant", reply or "")], stage=stage_name)


def build_shell(ep: Any, bp: Blueprint, *, how: str) -> Optional[Attempt]:
    """``det``: the blueprint's deterministic shell (IR, no LLM). ``llm``: the
    builder writes it from the plan (+ one repair). ``none``: nothing."""
    if how == "none":
        return None
    if how == "det":
        return ep.try_("shell", bp.shell_dsl(), ir=True, accept=_always)
    if how != "llm":
        raise ValueError(f"shell must be det|llm|none, got {how!r}")
    schematics = [render_schematic(bp, s.index) for s in bp.storeys]
    turn = turns_staged.shell_build(ep, bp, schematics)
    reply = ep.ask(with_context(ep, turn), stage="shell")
    att = ep.try_("shell", reply, accept=_nonempty)
    if steps.needs_repair(att, ep):
        fix = turns.repair(ep, att)
        reply2 = _ask_repair(ep, turn, reply, fix, "shell_repair")
        att = ep.try_("shell_repair", reply2, accept=_nonempty)
    return ep.current() or att


def interior_boxes(bp: Blueprint) -> List[Box]:
    """Every room's interior (floor included); with no rooms, each storey's
    footprint interior. What the roof and the final exterior pass may not touch."""
    if bp.rooms:
        return [bp.room_box(r.name) for r in bp.rooms]
    out: List[Box] = []
    for s in bp.storeys:
        fp = bp.footprint_of(s.index).interior()
        if fp is not None:
            out.append(Box.of(fp.x0, s.floor_y + 1, fp.z0, fp.x1, s.floor_y + s.height, fp.z1))
    return out


def build_roof(ep: Any, bp: Blueprint, *, how: str,
               protect: Optional[Sequence[Box]] = None) -> Optional[Attempt]:
    """``det``: the blueprint's roof (IR patch). ``llm``: the builder writes it,
    patched onto the build, region = canvas minus ``protect`` (default: every
    interior). ``none``: nothing."""
    if how == "none":
        return None
    if how == "det":
        text = bp.roof_dsl(tuple(ep.size))
        if not text.strip():
            ep.trace.event("roof_skipped", kind=bp.roof.kind)
            return None
        return ep.try_("roof", text, ir=True, patch=True, accept=_always)
    if how != "llm":
        raise ValueError(f"roof must be det|llm|none, got {how!r}")
    cur = ep.current()
    current = ep.render(cur, ["iso"]).get("iso") if cur is not None and cur.blocks else None
    region = Region.whole(ep.size, deny=list(interior_boxes(bp) if protect is None else protect),
                          name="exterior")
    turn = turns_staged.roof_build(ep, bp, exterior=_primary_png(ep), current=current)
    reply = ep.ask(with_context(ep, turn), stage="roof")
    return ep.try_("roof", reply, patch=True, region=region, accept=_nonempty)


def massing_render(ep: Any, bp: Blueprint) -> Optional[bytes]:
    """Iso render of the current world plus the deterministic roof, on a SCRATCH
    clone (not an attempt): the reference the exterior dream is edited from."""
    scratch = ep.world.clone()
    text = bp.roof_dsl(tuple(ep.size))
    if text.strip():
        scratch.run(Lowered(text, text, [], len(text.splitlines())))
    if scratch.block_count() == 0:
        return None
    return ep.render(scratch, ["iso"]).get("iso")


def _exterior_desc(ep: Any, bp: Optional[Blueprint]) -> str:
    desc = ep.prompt.strip().rstrip(".")
    if bp is None:
        return describe_mod.augment_desc(ep, desc)
    bits = []
    if bp.style:
        bits.append(bp.style.strip().rstrip("."))
    m = bp.materials
    roof = bp.roof.kind if bp.roof.kind != "none" else "no"
    bits.append(f"{len(bp.storeys)} storey(s), {roof} roof of {bp.roof.block or m.get('roof', '')}, "
                f"walls of {m.get('wall', '')}, trim of {m.get('trim', '')}")
    # The blueprint never saw a describe spec, so the spec adds only what the
    # blueprint leaves open (openings, features, what to omit): its storeys,
    # materials and summary would contradict the blueprint in the same prompt.
    return describe_mod.augment_desc(ep, desc + ". " + "; ".join(b for b in bits if b),
                                     blueprint=True)


def exterior_dream(ep: Any, bp: Optional[Blueprint], *, ref: str, refine: int,
                   refine_min_score: float = 7) -> Optional[DreamVersion]:
    """Generate the recipe's primary (scored) dream, anchored on our own picture.

    ``ref``: ``massing`` = iso render of shell + roof (same camera, footprint and
    roof line as every render the loop scores), ``schematic`` = the plan drawing,
    ``none`` = text only. Then ``steps.refine_dream`` if ``refine > 0``. None if the
    recipe has no scored dream or no image generator.
    """
    if ref not in EXTERIOR_REFS:
        raise ValueError(f"exterior_ref must be one of {EXTERIOR_REFS}, got {ref!r}")
    prim = _primary_scored(ep)
    if prim is None or ep.kit.image is None:
        ep.trace.event("exterior_dream_skipped",
                       reason="no scored dream" if prim is None else "no image generator")
        return None
    ep.dream_desc[prim.key] = _exterior_desc(ep, bp)
    refs: List[bytes] = []
    if ref == "massing" and bp is not None:
        png = massing_render(ep, bp)
        if png is not None:
            refs.append(png)
    elif ref == "schematic" and bp is not None:
        refs.append(render_schematic(bp, 0))
    prompt = dreams_mod.render_prompt(prim, ep)
    if ref == "massing" and refs:
        prompt += " " + MASSING_NOTE
    ep.trace.event("exterior_dream", ref=ref, n_refs=len(refs))
    ep.dream(prim.key, prompt=prompt, refs=refs, seed=steps.next_seed(ep, prim.key))
    if refine > 0:
        steps.refine_dream(ep, prim.key, rounds=refine, min_score=refine_min_score)
    return ep.latest_dream(prim.key)


# ===========================================================================
# Units
# ===========================================================================
def stairwell_boxes(bp: Blueprint, room_name: str) -> List[Box]:
    """Cells of room ``room_name``'s FLOOR SLAB that a stair from the storey below
    occupies: the top step (it sits in the slab) and the two cleared cells of the
    stairwell (a ladder: its top cell). Clipped to the room's interior.

    Why: a room's region includes its floor layer (so a model may change the
    floor's material), but :meth:`Blueprint.keep_clear` guards the arrival only
    from ``floor_y + 1`` up. Without these boxes one floor or carpet fill over the
    room closes the stairwell and replaces the top step, and the upper storey is
    unreachable.
    """
    r = bp.room(room_name)
    it = r.interior
    if it is None or r.storey < 1:
        return []
    fy = bp.storey(r.storey).floor_y
    out: List[Box] = []
    for st in bp.stairs:
        if st.from_storey != r.storey - 1:
            continue
        try:
            below = bp.storey(st.from_storey)
        except KeyError:
            continue
        cells = st.cells(below.height)
        cells = cells[-1:] if st.kind == "ladder" else cells[-3:]
        xs = [c[0] for c in cells if it.contains(c[0], c[2])]
        zs = [c[2] for c in cells if it.contains(c[0], c[2])]
        if xs:
            out.append(Box.of(min(xs), fy, min(zs), max(xs), fy, max(zs)))
    return out


def room_units(bp: Blueprint, *, order: str, max_units: int) -> List[Unit]:
    """One Unit per room (in ``order``), at most ``max_units``.

    Region: the room's interior, floor included, minus its keep-clear cells
    (doorways, stair run) and the stairwell in its floor (:func:`stairwell_boxes`).
    View box: the room WITH its walls, cut at half height (the architectural
    cutaway convention, so a steep camera sees in).
    """
    out: List[Unit] = []
    for r in order_rooms(bp.rooms, order)[:max(0, int(max_units))]:
        st = bp.storey(r.storey)
        cut = st.floor_y + int(math.ceil(st.height / 2.0))
        region = Region(r.name, (bp.room_box(r.name, walls=False, floor=True),),
                        tuple(bp.keep_clear(r.name)) + tuple(stairwell_boxes(bp, r.name)))
        view = Box.of(r.rect.x0, st.floor_y, r.rect.z0, r.rect.x1, cut, r.rect.z1)
        out.append(Unit(r.name, "room", r.storey, region, view, cut, r.purpose or r.name,
                        r.desc))
    return out


def _clip_xz(box: Box, rect: Any) -> Optional[Box]:
    """``box`` cut to the (x, z) of ``rect`` (a blueprint Rect); None if nothing is left."""
    if rect is None:
        return None
    x0, z0 = max(box.x0, rect.x0), max(box.z0, rect.z0)
    x1, z1 = min(box.x1, rect.x1), min(box.z1, rect.z1)
    if x0 > x1 or z0 > z1:
        return None
    return Box(x0, box.y0, z0, x1, box.y1, z1)


def storey_stair_boxes(bp: Blueprint, k: int) -> Tuple[List[Box], List[Box]]:
    """``(allow, deny)`` extras of storey ``k``'s region for the plan's stairs.

    Why: a stair's top step and its stairwell sit IN the slab of the storey above
    (``floor_y(k + 1)``), one layer above storey ``k``'s box, so a storey region of
    its box alone can never hold a working stair: the builder of storey ``k`` has
    every write up there reverted, and storey ``k + 1`` (its builder, or the
    deterministic slab of ``shell: det``) fills the slab over the run. So storey
    ``k`` may also write the slab cells above its own stairs' top (allow), and
    storey ``k + 1`` may not write them, nor the two cells of headroom above the
    arrival and its landing cell (deny). The landing's slab cell is NOT denied: it
    is ordinary floor, and nobody else would build it.
    """
    allow: List[Box] = []
    deny: List[Box] = []
    idx = {s.index for s in bp.storeys}
    for st in bp.stairs:
        if st.from_storey == k and k + 1 in idx:
            lower, upper = bp.storey(k), bp.storey(k + 1)
            cells = st.cells(lower.height)
            top = cells[-1:] if st.kind == "ladder" else cells[-3:]
            y = upper.floor_y
            box = Box.of(min(c[0] for c in top), y, min(c[2] for c in top),
                         max(c[0] for c in top), y, max(c[2] for c in top))
            clipped = _clip_xz(box, bp.footprint_of(k + 1).interior())
            if clipped is not None:
                allow.append(clipped)
        elif st.from_storey == k - 1 and k - 1 in idx and k in idx:
            lower, here = bp.storey(k - 1), bp.storey(k)
            cells = st.cells(lower.height)
            top = cells[-1:] if st.kind == "ladder" else cells[-3:]
            xs, zs = [c[0] for c in top], [c[2] for c in top]
            fy = here.floor_y
            interior = bp.footprint_of(k).interior()
            well = _clip_xz(Box.of(min(xs), fy, min(zs), max(xs), fy, max(zs)), interior)
            if st.kind != "ladder":
                dx, dz = _stair_step(st.dir)
                xs.append(top[-1][0] + dx)
                zs.append(top[-1][2] + dz)
            head = _clip_xz(Box.of(min(xs), fy + 1, min(zs), max(xs), fy + 2, max(zs)),
                            interior)
            deny += [b for b in (well, head) if b is not None]
    return allow, deny


def _stair_step(direction: str) -> Tuple[int, int]:
    """(dx, dz) of one step of a stair rising along ``direction`` (as Stair.cells)."""
    from blockgen.agentic.blueprint import Stair
    cells = Stair(0, 0, 0, direction).cells(1)
    return cells[1][0] - cells[0][0], cells[1][2] - cells[0][2]


def storey_units(bp: Blueprint) -> List[Unit]:
    """One Unit per storey, bottom up: region and view = the storey box (its slab,
    walls and everything up to floor_y + height; nothing above), plus the slab
    cells above its own stairs and minus the arrival of the stair from below
    (:func:`storey_stair_boxes`)."""
    out: List[Unit] = []
    for s in bp.storeys:
        box = bp.storey_box(s.index, floor=True)
        rooms = bp.rooms_on(s.index)
        purpose = ("the " + ", ".join(r.purpose for r in rooms) if rooms
                   else ("the ground floor" if s.index == 0 else f"upper floor {s.index}"))
        allow, deny = storey_stair_boxes(bp, s.index)
        out.append(Unit(f"storey{s.index}", "storey", s.index,
                        Region(f"storey{s.index}", (box, *allow), tuple(deny)), box, None,
                        purpose, ""))
    return out


def unit_render(ep: Any, target: Union[Attempt, Any], unit: Unit) -> Optional[bytes]:
    """Cutaway render of ``unit`` in an attempt (cached as ``renders[unit.key]``)
    or a world. None if the unit is empty or nothing can render."""
    att = target if isinstance(target, Attempt) else None
    if att is not None and unit.key in att.renders:
        return att.renders[unit.key]
    world = att.world if att is not None else target
    if world is None or ep.kit.renderer is None:
        return None
    s = region_structure(world, unit.view_box, cut_y=unit.cut_y)
    if int(s.occupied_mask.sum()) == 0:
        return None
    png = ep.render(s, ["room"]).get("room")
    if att is not None and png is not None:
        att.renders[unit.key] = png
    return png


def _interior_wd(unit: Unit) -> Tuple[int, int]:
    """Floor size the dream prompt names: a room's region is already its interior;
    a storey's region includes its outer walls, so they are taken off."""
    b = unit.region.allow[0]
    sx, _, sz = b.shape
    if unit.kind == "storey":
        return max(1, sx - 2), max(1, sz - 2)
    return sx, sz


def unit_dream(ep: Any, bp: Optional[Blueprint], unit: Unit, *, refs: Sequence[str],
               desc: Optional[str] = None) -> Optional[DreamVersion]:
    """Dream one unit, anchored on a render of the unit as it is now.

    Registers the runtime dream ``u_<name>`` (never scored globally, paired with
    the ``u_<name>`` render row on the contact sheet). Refs in this order, each
    only if available: ``cutaway`` (the CURRENT world's unit render, i.e. the empty
    room: the composition the edit keeps), ``exterior`` (latest primary dream,
    materials only), ``below`` (the storey underneath), ``schematic`` (the plan
    with this room highlighted). None without an image generator, or when the
    image budget is spent (the unit is then built from text).
    """
    if ep.kit.image is None:
        return None
    bad = [r for r in refs if r not in UNIT_REFS]
    if bad:
        raise ValueError(f"unknown unit dream ref(s) {bad}; known: {UNIT_REFS}")
    template = turns_staged.ROOM_DREAM if unit.kind == "room" else turns_staged.STOREY_DREAM
    ds = DreamSpec(unit.key, template, view="room", weight=0.0, show_builder=False)
    ep.register_dream(ds, trace_view=unit.key)
    pngs: List[bytes] = []
    for name in refs:
        png: Optional[bytes] = None
        if name == "cutaway":
            png = unit_render(ep, ep.world, unit)
        elif name == "exterior":
            png = _primary_png(ep)
        elif name == "below" and bp is not None and unit.storey > 0:
            below = [u for u in storey_units(bp) if u.storey == unit.storey - 1]
            png = unit_render(ep, ep.world, below[0]) if below else None
        elif name == "schematic" and bp is not None:
            png = render_schematic(bp, unit.storey,
                                   highlight=unit.name if unit.kind == "room" else None)
        if png is not None:
            pngs.append(png)
    text = (desc or unit.desc or f"furnished as a typical {unit.purpose} for: {ep.prompt}")
    iw, idp = _interior_wd(unit)
    prompt = dreams_mod.render_prompt(ds, ep, desc=text)
    prompt = (prompt.replace("{purpose}", unit.purpose).replace("{iw}", str(iw))
              .replace("{id}", str(idp)))
    try:
        return ep.dream(unit.key, prompt=prompt, refs=pngs, seed=steps.next_seed(ep, unit.key),
                        view_hint="room")
    except BudgetExceeded as exc:
        if not is_image_cap(exc):
            raise           # out of money: the stage (or the build) ends, as anywhere
        ep.trace.event("unit_dream_budget_stop", unit=unit.name, error=str(exc))
        return None


def unit_terms(ep: Any, terms: Optional[Mapping[str, float]]) -> Dict[str, float]:
    """:data:`DEFAULT_UNIT_TERMS` overridden by ``terms``; under ``--mock`` the
    ``embed`` term runs as ``pixel`` (as ``kit.mock_spec`` does for recipe scorers)."""
    out = dict(DEFAULT_UNIT_TERMS)
    out.update({str(k): float(v) for k, v in (terms or {}).items()})
    if getattr(ep.kit, "mock", False) and "embed" in out:
        out["pixel"] = out.get("pixel", 0.0) + out.pop("embed")
    bad = sorted(set(out) - {"silhouette", "color_lab", "pixel", "embed"})
    if bad:
        raise ValueError(f"unknown unit score term(s) {bad}; known: silhouette, color_lab, "
                         f"pixel, embed")
    return out


def _embed_sim(a: bytes, b: bytes) -> float:
    from blockgen.agentic.scoring import embed_baseline, image_features
    backbone = "siglipB"
    base = embed_baseline(backbone)
    fa, fb = image_features([a, b], backbone)
    return float(np.clip((float(fa @ fb) - base) / max(1e-6, 1.0 - base), 0.0, 1.0))


def _term_fns() -> Dict[str, Callable[[bytes, bytes], float]]:
    from blockgen.agentic.scoring import color_lab_sim, pixel_sim, silhouette_iou
    return {"silhouette": silhouette_iou, "color_lab": color_lab_sim, "pixel": pixel_sim,
            "embed": _embed_sim}


def unit_score(ep: Any, att: Attempt, unit: Unit, target_png: bytes,
               terms: Mapping[str, float], *, annotate: bool = True) -> ScoreTerm:
    """Weighted mean of image similarities between the unit's target dream and a
    cutaway render of the unit in ``att``. Cached per (unit, attempt, target).

    ``annotate`` writes ``extra['unit']`` / ``extra['unit_score']`` on ``att``;
    off when scoring the state a candidate is compared against (the shell would
    otherwise be labelled with whichever room was scored against it last).
    """
    digest = hashlib.sha256(target_png).hexdigest()[:16]
    cache = ep.notes.setdefault("unit_scores", {})
    key = (unit.name, int(att.idx), digest)
    if key in cache:
        if annotate:
            att.extra["unit"] = unit.name
            att.extra["unit_score"] = float(cache[key].value)
        return cache[key]
    png = unit_render(ep, att, unit) if att.blocks > 0 else None
    parts: Dict[str, float] = {}
    if png is None:
        value = 0.0
    else:
        fns = _term_fns()
        num = den = 0.0
        for name, w in terms.items():
            if float(w) == 0.0:
                continue
            v = float(fns[name](target_png, png))
            parts[name] = v
            num += float(w) * v
            den += float(w)
        value = num / den if den > 0 else 0.0
    term = ScoreTerm(float(value), per_view={unit.key: float(value)},
                     note="unit " + ", ".join(f"{k} {v:.2f}" for k, v in parts.items()))
    cache[key] = term
    if annotate:
        att.extra["unit"] = unit.name
        att.extra["unit_score"] = float(value)
    ep.trace.event("unit_score", unit=unit.name, idx=int(att.idx), value=round(value, 4),
                   parts={k: round(v, 4) for k, v in parts.items()}, target=digest)
    return term


def _validity(att: Optional[Attempt]) -> float:
    from blockgen.agentic.scoring import largest_component_frac, structure_of
    if att is None or att.blocks <= 0:
        return 0.0
    if att.score is not None and "validity" in att.score.parts:
        return float(att.score.parts["validity"])
    return largest_component_frac(structure_of(att))


def region_blocks(att: Optional[Attempt], region: Region) -> int:
    """Non-air voxels of ``att``'s world inside ``region``."""
    if att is None or att.world is None:
        return 0
    s = att.world.to_structure(crop=False)
    return int((s.occupied_mask & region.mask(s.occupied_mask.shape)).sum())


def unit_accept(ep: Any, unit: Unit, target_png: Optional[bytes], *,
                terms: Mapping[str, float], eps: float = 0.0,
                max_validity_drop: float = 0.1) -> AcceptFn:
    """The per-call accept rule of a unit's attempts.

    With a target: accept iff the candidate is non-empty, its unit score beats the
    current state's by more than ``eps``, and the build's connectivity did not fall
    by more than ``max_validity_drop`` (floating furniture games an image score as
    easily as floating decoration does). Without a target (the no-dream control):
    accept iff the region gained blocks and connectivity held.
    """
    def accept(prev: Optional[Attempt], cand: Attempt) -> Tuple[bool, str]:
        if cand.blocks == 0:
            return False, "empty build"
        # Score the candidate before any rejection: the unit render and score then
        # exist for every attempt, so the contact sheet's u_<name> row and the
        # attempt header show what was rejected, not a blank.
        if target_png is not None:
            b = unit_score(ep, cand, unit, target_png, terms).value
        else:
            # No target (the image-free control): render anyway, so its contact
            # sheet has the same u_<name> rows as the image arm's.
            b = 0.0
            for att in (prev, cand):
                if att is not None and att.blocks > 0:
                    unit_render(ep, att, unit)
        if prev is not None and prev.blocks > 0:
            va, vb = _validity(prev), _validity(cand)
            if vb < va - max_validity_drop:
                return False, f"validity dropped {va:.2f} -> {vb:.2f}"
        if target_png is not None:
            a = (unit_score(ep, prev, unit, target_png, terms, annotate=False).value
                 if prev is not None else 0.0)
            if b > a + eps:
                return True, f"unit {unit.name} improved {a:.3f} -> {b:.3f}"
            return False, f"unit {unit.name} no improvement {a:.3f} -> {b:.3f} (eps {eps:g})"
        na, nb = region_blocks(prev, unit.region), region_blocks(cand, unit.region)
        if nb > na:
            return True, f"unit {unit.name} gained blocks {na} -> {nb}"
        return False, f"unit {unit.name} gained no blocks ({na} -> {nb})"

    return accept


def _schematic_for(ep: Any, bp: Optional[Blueprint], unit: Unit) -> Optional[bytes]:
    if bp is None or (unit.kind == "storey" and not bp.rooms):
        return None
    try:
        return render_schematic(bp, unit.storey,
                                highlight=unit.name if unit.kind == "room" else None)
    except Exception as exc:  # noqa: BLE001 - a drawing bug must not cost the build
        ep.trace.event("schematic_error", unit=unit.name, error=repr(exc))
        return None


def build_unit(ep: Any, bp: Optional[Blueprint], unit: Unit, *, target: Optional[DreamVersion],
               iters: int, mode: str, local_coords: bool, terms: Mapping[str, float],
               eps: float, feedback: str) -> Optional[Attempt]:
    """Build one unit inside its region: first turn, one repair if it failed, then
    up to ``iters`` compare (with a target) or critique (without) rounds; stops
    after 2 consecutive rejections."""
    if feedback not in FEEDBACKS:
        raise ValueError(f"feedback must be one of {FEEDBACKS}, got {feedback!r}")
    fresh_dialogue(ep)
    terms = unit_terms(ep, terms)
    tpng = target.png if target is not None else None
    acc = unit_accept(ep, unit, tpng, terms=terms, eps=eps)
    offset = unit.region.origin if local_coords else None
    patch = mode == "patch"

    def run(stage_name: str, reply: str) -> Attempt:
        return ep.try_(stage_name, reply, patch=patch, region=unit.region, offset=offset,
                       accept=acc)

    current = unit_render(ep, ep.world, unit)
    if unit.kind == "storey" and bp is not None:
        below = None
        if unit.storey > 0:
            lower = [u for u in storey_units(bp) if u.storey == unit.storey - 1]
            below = unit_render(ep, ep.world, lower[0]) if lower else None
        turn = turns_staged.storey_build(ep, bp, unit, target=tpng, below=below,
                                         exterior=_primary_png(ep),
                                         schematic=_schematic_for(ep, bp, unit), mode=mode,
                                         local=local_coords)
    else:
        turn = turns_staged.unit_build(ep, bp, unit, target=tpng, current=current,
                                       schematic=_schematic_for(ep, bp, unit), mode=mode,
                                       local=local_coords)
    reply = ep.ask(with_context(ep, turn), stage=unit.key)
    att = run(unit.key, reply)
    if needs_unit_repair(att, ep):
        fix = turns_staged.unit_repair(ep, unit, att, local=local_coords)
        reply2 = _ask_repair(ep, turn, reply, fix, f"{unit.key}_fix")
        att = run(f"{unit.key}_fix", reply2)
    # The opening turn counts toward patience: a rejected first build plus one
    # rejected round is the "2 consecutive rejections" that ends the unit.
    stall = 0 if att.accepted else 1
    last = att
    # The cost estimate (kit.worst_case) runs every round: its mock replies are
    # always rejected, so the patience stop would cap any max_*_iters at 1 and the
    # "upper bound" would undercount a real run whose rounds get accepted.
    worst = bool(getattr(ep.kit, "worst_case", False))
    for i in range(max(0, int(iters))):
        if stall >= 2 and not worst:
            ep.trace.event("stop", rule="unit_patience", unit=unit.name, iter=i)
            break
        # Images and score show the STATE the next program patches (the current
        # accepted world); the execution report is this unit's own last program,
        # never an earlier unit's accepted attempt.
        cur = ep.current() or last
        yours = unit_render(ep, cur, unit)
        if tpng is not None:
            sc = (unit_score(ep, cur, unit, tpng, terms, annotate=False).value
                  if cur.blocks else None)
            t = turns_staged.unit_compare(ep, unit, last, target=tpng, yours=yours, score=sc,
                                          mode=mode, local=local_coords, feedback=feedback)
        else:
            t = turns_staged.unit_critique(ep, unit, last, yours=yours, mode=mode,
                                           local=local_coords)
        stage_name = f"{unit.key}_m{i}"
        last = run(stage_name, ep.ask(t, role="critic", stage=stage_name))
        stall = 0 if last.accepted else stall + 1
    return ep.current() or att


def needs_unit_repair(att: Optional[Attempt], ep: Any = None) -> bool:
    """:func:`steps.needs_repair`, or some of the program's writes fell outside the
    unit's region and were reverted. Why the second case: in a unit it is the
    usual failure (the model furnished the wrong room, or used whole-house
    coordinates), and it leaves no failed line and no empty build behind."""
    return steps.needs_repair(att, ep) or bool(att is not None
                                               and int(att.extra.get("reverted", 0) or 0) > 0)


def context_parts(ep: Any) -> List[Part]:
    """The recipe's context modules' user-visible parts (e.g. a texture sheet).

    Why the staged methods add them to each unit's FIRST turn themselves: every
    unit starts a fresh dialogue, so the Episode's automatic first-builder-turn
    prefix would reach only whichever unit happens to build first, together with
    the global exterior TARGET, a confusing second target in a room turn. The
    staged methods turn that prefix off (``ep.inject_first_turn = False``) and
    each unit sees the same context instead.
    """
    fn = getattr(ep, "contexts", None)
    if not callable(fn):
        return []
    return [p for _, c in fn() for p in (getattr(c, "user_parts", None) or [])]


def with_context(ep: Any, turn: Message) -> Message:
    """``turn`` with :func:`context_parts` in front (unchanged if there are none)."""
    pre = context_parts(ep)
    return Message(turn.role, pre + list(turn.parts)) if pre else turn


# ===========================================================================
# Final exterior pass
# ===========================================================================
def _protect_note(region: Region) -> Part:
    text = ("The finished rooms are protected: anything you place inside "
            + "; ".join(b.describe() for b in region.deny)
            + " is discarded.") if region.deny else "The whole canvas is open to you."
    return Part("text", text=text)


def _final_turn(ep: Any, i: int, cur: Attempt, *, feedback: str, mode: str,
                region: Region, has_dream: bool) -> Optional[Message]:
    """steps.match_step's compare turn (no REDREAM) or, without dreams, the generic
    critique, each told which boxes are protected."""
    if has_dream:
        describe_mod.refresh(ep)
        vdm = steps.vdm_text(ep, cur, stage=f"final_judge{i}") if feedback in ("vdm", "both") else ""
        turn = turns.compare(ep, cur, feedback=feedback, mode=mode, allow_redream=False, vdm=vdm)
    else:
        if cur.blocks == 0:
            return None
        renders = ep.render(cur, list(CRITIQUE_VIEWS))
        views = [v for v in CRITIQUE_VIEWS if v in renders]
        if not views:
            ep.trace.event("critique_skipped", reason="no render")
            return None
        turn = turns.critique_generic(ep, cur, views, [renders[v] for v in views], mode=mode)
    return Message("user", [_protect_note(region)] + list(turn.parts))


def render_units(ep: Any, units: Sequence[Unit], start: int = 0) -> None:
    """Unit cutaways of every attempt from ``start`` on (the complete buildings).

    Why: unit renders are otherwise made only while that unit is built, so the
    contact sheet's ``u_<name>`` rows were blank under the roof and final columns,
    including the selected (boxed) build. Rendering them at the end shows what each
    room looks like in the build that is actually returned. Renders only; no score.
    """
    for att in list(ep.attempts)[max(0, int(start)):]:
        if att.blocks <= 0 or att.world is None:
            continue
        for u in units:
            unit_render(ep, att, u)


def final_match(ep: Any, *, max_iters: int, feedback: str, mode: str,
                protect: Sequence[Box], patience: int = 2, stop_at: float = 0.85) -> None:
    """Exterior pass after the roof: compare with the exterior dream (or critique
    without one), patch or rewrite outside ``protect``; recipe accept policy.
    Stops on ``stop_at`` / ``patience`` like the dream loop."""
    if feedback not in FEEDBACKS:
        raise ValueError(f"feedback must be one of {FEEDBACKS}, got {feedback!r}")
    region = Region.whole(ep.size, deny=list(protect), name="exterior")
    prim = _primary_scored(ep)
    has_dream = prim is not None and ep.latest_dream(prim.key) is not None
    patience = max(1, int(patience))
    stall = 0
    for i in range(max(0, int(max_iters))):
        cur = ep.current()
        if cur is None or cur.blocks == 0:
            break
        if cur.score is not None and cur.score.value >= stop_at:
            ep.trace.event("stop", rule="stop_at", iter=i, value=cur.score.value)
            break
        turn = _final_turn(ep, i, cur, feedback=feedback, mode=mode, region=region,
                           has_dream=has_dream)
        if turn is None:
            break
        reply = ep.ask(turn, role="critic", stage=f"final{i}")
        cand = ep.try_(f"final{i}", reply, patch=(mode == "patch"), region=region)
        improved = (cand.accepted and cand.score is not None and cur.score is not None
                    and cand.score.value > cur.score.value)
        stall = 0 if improved else stall + 1
        if stall >= patience:
            ep.trace.event("stop", rule="patience", iter=i, stall=stall)
            break


__all__ = ["CRITIQUE_VIEWS", "DEFAULT_UNIT_TERMS", "EXTERIOR_REFS", "FEEDBACKS",
           "IMAGE_PLAN_SOURCES", "MASSING_NOTE", "PLAN_SOURCES", "UNIT_REFS", "Unit",
           "blueprint_from_image", "blueprint_from_llm", "blueprint_from_text", "build_roof",
           "build_shell", "build_unit", "context_parts",
           "exterior_dream", "final_match", "fresh_dialogue", "interior_boxes",
           "is_image_cap", "make_blueprint", "massing_render", "needs_unit_repair",
           "record_blueprint", "region_blocks", "render_units", "room_units", "stage", "stairwell_boxes",
           "storey_stair_boxes", "storey_units", "unit_accept", "unit_dream", "unit_render", "unit_score",
           "unit_terms", "verify_with_plan_dream", "with_context"]
