"""Prompt templates for the staged methods (``blueprint``, ``floors``).

Same rules as :mod:`blockgen.agentic.turns`: no turn names a DSL command, every
turn that asks for a build ends with ``ep.kit.actions.output_rule()`` (so the same
method runs with the DSL, per-block lines, the Python API or a mix), and every
verdict turn ends with :data:`turns.JSON_RULE`.

Why stable literal tags (:data:`BLUEPRINT_TAG`, :data:`VERIFY_TAG`,
:data:`UNIT_TAG`, :data:`REGION_LINE`, :data:`CANVAS_LINE`): the offline
``mock:auto`` LLM recognises a staged turn by them and answers with a valid
blueprint, or with a small furnishing INSIDE the stated region, so a ``--mock``
run exercises the real parse, repair, region and accept paths instead of
answering every unit turn with a house that the region would clip away.

Why the region is stated in the turn and not only enforced: writes outside it are
reverted after execution (``region.restore_outside``) and reported line by line on
the next turn, but a model that knows the box up front wastes no turn on it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Sequence, Tuple

from blockgen.agentic.core import Attempt
from blockgen.agentic.providers import Message, Part
from blockgen.agentic.turns import JSON_RULE, MODE_INSTRUCTIONS

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids an import cycle)
    from blockgen.agentic.blueprint import Blueprint
    from blockgen.agentic.recipes.steps_staged import Unit

#: In blueprint_json, blueprint_fix and plan_extract (the mock answers with a plan).
BLUEPRINT_TAG = "BLUEPRINT JSON"
#: In plan_verify (the mock answers ``{"ok": true, ...}``).
VERIFY_TAG = "BLUEPRINT CHECK"
#: In design_brief (plan_source 'text'; the mock answers with a prose brief). Never
#: in a JSON turn: the mock checks BLUEPRINT_TAG first, and the JSON turn that
#: shows the brief calls it "your design brief" (lower case).
BRIEF_TAG = "DESIGN BRIEF"
#: In unit_build, unit_compare, unit_critique, unit_repair and storey_build.
UNIT_TAG = "UNIT BUILD"
#: Exact format; the mock parses it to build inside the box.
REGION_LINE = "ALLOWED REGION: x {x0}..{x1}, y {y0}..{y1}, z {z0}..{z1}"
#: Exact format; the mock parses it to size its blueprint.
CANVAS_LINE = "Canvas: {W} x {H} x {D} blocks"

#: Said in every unit turn (the region contract, in the model's words).
DISCARD_NOTE = "Blocks you place outside this region are discarded and reported."

# ---------------------------------------------------------------------------
# Image prompt templates (DreamSpec templates). {style} {desc} {W} {H} {D} are
# filled by dreams.render_prompt; {purpose} {iw} {id} by steps_staged with
# str.replace (render_prompt leaves unknown fields visible on purpose).
# ---------------------------------------------------------------------------
#: A flat drawing ON the reference grid paper. No {style}: the house style asks
#: for a 3D render, which is exactly what a plan must not be (pilot finding P2).
PLAN_DREAM = (
    "A flat architectural floor plan drawn ON the reference image, which is grid "
    "paper: one grid square is one block, and the axis numbers along the top (x) and "
    "the left (z) edges must stay readable. Draw the ground floor of: {desc}. Thick "
    "dark walls exactly on grid lines, gaps in the walls for doors, the name of each "
    "room written inside it, the front door on the bottom edge. Inside the dashed "
    "outline only. Straight top-down, no perspective, no 3D, no shading, no "
    "furniture, white background. The whole canvas is {W} x {D} blocks.")

#: The schematic redrawn as a top-down Minecraft cutaway (plan_source 'both').
PLAN_STYLIZE = (
    "{style} The reference floor plan redrawn as a top-down Minecraft cutaway seen "
    "straight from above, with every wall cut at half height: the same walls, doors "
    "and rooms in exactly the same places, wall and floor materials as blocks, and "
    "simple furniture from blocks in each room. Building: {desc}. Footprint within "
    "{W} x {D} blocks.")

ROOM_DREAM = (
    "{style} Cutaway of ONE room of a Minecraft house: the {purpose}. Walls cut at "
    "half height, everything above removed, seen from above at a steep isometric "
    "angle (about 55 degrees down), the room centred and filling the frame. Furnish "
    "it with block furniture (slabs, stairs, fences, trapdoors, carpets, lanterns): "
    "{desc}. The FIRST reference image is the empty room: keep its floor, walls and "
    "doorways exactly where they are and add only furniture. Any other reference only "
    "shows the house's materials. Floor {iw} x {id} blocks.")

STOREY_DREAM = (
    "{style} One storey of a Minecraft house, {purpose}, with everything above it "
    "removed: the floor, the outer walls at full height, the inner walls, doorways, "
    "windows, stairs up and block furniture (slabs, stairs, fences, trapdoors, "
    "carpets, lanterns), seen from above at a steep isometric angle (about 55 degrees "
    "down), the storey centred and filling the frame. {desc}. If a reference shows a "
    "storey below, this one sits on top of it with the same footprint and materials; "
    "any other reference only shows the house's materials. Floor {iw} x {id} blocks.")


# ===========================================================================
# Helpers
# ===========================================================================
def _rule(ep: Any) -> str:
    return ep.kit.actions.output_rule()


def _mode(mode: str) -> str:
    if mode not in MODE_INSTRUCTIONS:
        raise ValueError(f"mode must be one of {sorted(MODE_INSTRUCTIONS)}, got {mode!r}")
    return MODE_INSTRUCTIONS[mode]


def canvas_line(ep: Any) -> str:
    w, h, d = ep.size
    return CANVAS_LINE.format(W=w, H=h, D=d)


def region_line(unit: "Unit", *, local: bool) -> str:
    """The unit's first allow box as :data:`REGION_LINE`; in room-local
    coordinates (origin = the box's low corner) when ``local``."""
    b = unit.region.allow[0]
    if local:
        sx, sy, sz = b.shape
        return REGION_LINE.format(x0=0, x1=sx - 1, y0=0, y1=sy - 1, z0=0, z1=sz - 1)
    return REGION_LINE.format(x0=b.x0, x1=b.x1, y0=b.y0, y1=b.y1, z0=b.z0, z1=b.z1)


def _box_text(b: Any, origin: Optional[Sequence[int]] = None) -> str:
    """``x a..b, y c..d, z e..f``, shifted into local coordinates if ``origin``."""
    ox, oy, oz = origin if origin is not None else (0, 0, 0)
    return (f"x {b.x0 - ox}..{b.x1 - ox}, y {b.y0 - oy}..{b.y1 - oy}, "
            f"z {b.z0 - oz}..{b.z1 - oz}")


def region_block(unit: "Unit", *, local: bool, mode: str = "patch") -> str:
    """Every unit turn's region paragraph: the box, the cells to keep clear, the
    discard rule, and (local) what the origin is."""
    origin = unit.region.origin if local else None
    lines = [UNIT_TAG + f": {unit.kind} '{unit.name}' ({unit.purpose})",
             region_line(unit, local=local)]
    for extra in unit.region.allow[1:]:
        lines.append(f"Also allowed: {_box_text(extra, origin)}")
    if unit.region.deny:
        lines.append("Keep these cells clear (doorways and stairs; they must stay air): "
                     + "; ".join(_box_text(b, origin) for b in unit.region.deny))
    lines.append(floor_line(unit, local=local, mode=mode))
    lines.append(DISCARD_NOTE)
    if local:
        lines.append("Coordinates in this turn are LOCAL to the region: (0, 0, 0) is its "
                     "lowest corner (the interior floor corner); x, y-up and z keep their "
                     "directions.")
    return "\n".join(lines)


def floor_line(unit: "Unit", *, local: bool, mode: str = "patch") -> str:
    """Where the unit's floor is. Why: the region's lowest layer IS the floor slab
    (so a model may change its material); without this line a model reads
    "y 5..9" as free air and sinks its furniture into the floor. In ``rewrite``
    mode the program replaces everything inside the region, floor included, so
    the line says the floor must be part of it."""
    b = unit.region.allow[0]
    y0 = 0 if local else b.y0
    if unit.kind == "room":
        if mode == "rewrite":
            return (f"The floor is the layer y = {y0}. Your program REPLACES everything in "
                    f"the region, so include the floor; furniture stands on it, from "
                    f"y = {y0 + 1} up.")
        return (f"The floor is the layer y = {y0} (already built; inside the region, so you "
                f"may change its material); furniture stands on it, from y = {y0 + 1} up.")
    return (f"y = {y0} is this storey's floor slab; its walls and furniture go from "
            f"y = {y0 + 1} up.")


def _bp_json(bp: Any) -> str:
    # No fallback on purpose: a plan-less "{}" would reach the model silently.
    return bp.to_json(indent=None)


def _last_note(att: Attempt) -> List[str]:
    """Tell the model when its last program was not kept: the images then show
    the room WITHOUT it, which otherwise reads as "my program did nothing"."""
    if att.accepted:
        return []
    why = f" ({att.reason})" if att.reason else ""
    return [f"Your last program was NOT kept{why}; the images show the room without it."]


def _schema_text() -> str:
    from blockgen.agentic.blueprint import SCHEMA_TEXT
    return SCHEMA_TEXT


def _limits(ep: Any, max_rooms: int, max_storeys: int) -> str:
    w, h, d = ep.size
    rooms = ("No rooms: leave \"rooms\" empty, one open space per storey."
             if int(max_rooms) <= 0 else f"At most {int(max_rooms)} rooms in total.")
    return (f"Limits: at most {int(max_storeys)} storey(s); {rooms} Everything, roof "
            f"included, must fit the {w} x {h} x {d} canvas. Neighbouring rooms SHARE "
            "their wall line (one room's x1 is the next room's x0). Every room must be "
            "reachable through doors from the front door.")


def _scale(ep: Any) -> List[str]:
    """The player-scale paragraph of a planner turn, only while the recipe lists
    the ``scale`` context (so a ``*-noscale`` control differs in that one thing;
    the repair's hard minimum, 5 x 5 rects, is in SCHEMA_TEXT for every arm)."""
    from blockgen.agentic.scale import blueprint_line, scale_active
    return [blueprint_line()] if scale_active(ep) else []


def _purpose_lines(bp: Optional["Blueprint"], storey: int,
                   origin: Tuple[int, int] = (0, 0)) -> List[str]:
    """One line per room of ``storey``; rects shifted by ``origin`` (x, z)."""
    if bp is None:
        return []
    ox, oz = origin
    out = []
    for r in bp.rooms_on(storey):
        rect = [r.rect.x0 - ox, r.rect.z0 - oz, r.rect.x1 - ox, r.rect.z1 - oz]
        out.append(f"  - {r.name} ({r.purpose}): rect {rect}"
                   + (f"; {r.desc}" if r.desc else ""))
    return out


# ===========================================================================
# Blueprint turns (JSON; role planner / dream_critic)
# ===========================================================================
def design_brief(ep: Any, *, max_rooms: int, max_storeys: int) -> Message:
    """Step 1 of plan_source 'text': the planner REASONS about the house in prose.

    Why prose first: the JSON turn asks for coordinates, room list and openings in
    one go, so the model commits to numbers before it has thought about how the
    house is used. The brief is where it decides the rooms, their sizes in blocks,
    which room opens onto which, and how a player walks from the front door to
    every room; the JSON turn then only transcribes it (:func:`blueprint_json`
    with ``brief``).
    """
    w, h, d = ep.size
    text = "\n\n".join([
        f"{BRIEF_TAG}: think this building through in words before it is drawn.",
        f"Build request: {ep.prompt}",
        canvas_line(ep) + " (x by y-up by z). The front of the building faces +z (the z1 "
                          "side), where the front door goes.",
        _limits(ep, max_rooms, max_storeys),
        *_scale(ep),
        "Write a short design brief in plain text (at most 25 lines), with these headings:\n"
        "- Footprint: outer size in blocks (x by z) and where it sits on the canvas.\n"
        "- Storeys: how many, and the clear height of each (air between floor and "
        "ceiling).\n"
        "- Rooms: per storey, each room's purpose and its size in blocks INCLUDING its "
        "walls, and which rooms share a wall.\n"
        "- Circulation: how a player walks from the front door to every room (which "
        "doors, which hall or corridor, where the stairs go up).\n"
        "- Openings: the front door, inner doors and windows, by wall.\n"
        "- Roof and materials: roof shape and ridge direction; wall, floor, trim and "
        "roof blocks.\n"
        f"Check the arithmetic: the rooms of a storey must tile its footprint, and the "
        f"storeys plus the roof must fit {h} blocks of height.",
        "Plain text only: no JSON and no code; the plan is converted to JSON next."])
    return Message("user", [Part("text", text=text)])


def blueprint_json(ep: Any, *, max_rooms: int, max_storeys: int,
                   brief: str = "") -> Message:
    """The planner writes the typed floor plan the whole build follows.

    With ``brief`` (plan_source 'text') the turn shows the planner's own design
    brief and asks for a faithful transcription of it. Without one (and without
    a ``scale`` context) the text is the same as before the brief existed.
    """
    if brief.strip():
        head = [f"{BLUEPRINT_TAG}: turn your design brief into the floor plan.",
                f"Build request: {ep.prompt}",
                "Your design brief:\n" + brief.strip()]
        ask = ("Transcribe the brief faithfully: the same rooms, sizes, adjacencies, "
               "doors and storey heights, in coordinates. Where the brief's numbers do not "
               "add up, fix them the least you can and keep its intent.")
    else:
        head = [f"{BLUEPRINT_TAG}: plan this building before anything is built.",
                f"Build request: {ep.prompt}"]
        ask = ""
    text = "\n\n".join([
        *head,
        canvas_line(ep) + " (x by y-up by z). The front of the building faces +z (the z1 "
                          "side): put the front door there.",
        _limits(ep, max_rooms, max_storeys),
        *_scale(ep),
        "Write the floor plan in this JSON shape:\n" + _schema_text(),
        *([ask] if ask else []),
        "Choose materials that fit the request, and a one-line \"style\" describing the "
        "exterior look. Give each room a short \"desc\": the furniture it should hold.",
        JSON_RULE])
    return Message("user", [Part("text", text=text)])


def blueprint_fix(ep: Any, raw: Mapping[str, Any], problems: Sequence[str], *,
                  brief: str = "") -> Message:
    """One retry: the planner sees its own JSON and what is wrong with it (and,
    for plan_source 'text', the brief it was transcribing, so the fix keeps it)."""
    try:
        shown = json.dumps(dict(raw or {}), separators=(",", ":"))[:6000]
    except (TypeError, ValueError):
        shown = str(raw)[:6000]
    probs = "\n".join(f"  - {p}" for p in list(problems)[:20]) or "  - (none listed)"
    text = "\n\n".join([
        f"{BLUEPRINT_TAG}: your floor plan has problems. Fix them and send the whole "
        "corrected plan.",
        f"Build request: {ep.prompt}",
        canvas_line(ep) + " (x by y-up by z); the front faces +z (the z1 side).",
        *(["Your design brief (keep to it):\n" + brief.strip()] if brief.strip() else []),
        "Your plan:\n" + shown,
        "Problems:\n" + probs,
        *_scale(ep),
        "The JSON shape again:\n" + _schema_text(),
        JSON_RULE])
    return Message("user", [Part("text", text=text)])


def plan_extract(ep: Any, grid_png: bytes, plan_png: bytes, *, max_rooms: int,
                 max_storeys: int) -> Message:
    """A VLM reads a dreamed plan drawn on our grid paper into blueprint JSON.

    Both images are shown: the blank grid gives the axis labels at full contrast,
    the plan may have smudged them.
    """
    head = "\n\n".join([
        f"{BLUEPRINT_TAG}: read a drawn floor plan into a typed blueprint.",
        f"Build request: {ep.prompt}",
        canvas_line(ep) + " (x by y-up by z). One grid square is one block. The numbers "
                          "along the top edge are x, along the left edge z.",
    ])
    tail = "\n\n".join([
        "Read the walls, doors and room names of the PLAN as exactly as you can, using the "
        "axis numbers for coordinates. The plan shows the ground floor only (storey 0); "
        "if the request needs more storeys, propose the upper ones yourself.",
        _limits(ep, max_rooms, max_storeys),
        *_scale(ep),
        "Reply in this JSON shape:\n" + _schema_text(),
        JSON_RULE])
    return Message("user", [Part("text", text=head),
                            Part("text", text="GRID (blank, with the axis numbers):"),
                            Part("image", image_png=grid_png),
                            Part("text", text="PLAN (drawn on that grid):"),
                            Part("image", image_png=plan_png),
                            Part("text", text=tail)])


def plan_verify(ep: Any, bp: "Blueprint", schematic_png: bytes, dream_png: bytes) -> Message:
    """Dream critic: does the plan make sense once it is pictured as a house?

    JSON ``{"ok": bool, "issues": [...], "blueprint": <corrected or null>}``.
    """
    head = "\n\n".join([
        f"{VERIFY_TAG}: a floor plan for a Minecraft build, and a picture of it as a "
        "top-down cutaway.",
        f"Build request: {ep.prompt}",
        canvas_line(ep) + " (x by y-up by z); the front faces +z (the z1 side).",
        "Plan summary:\n" + bp.summary(),
    ])
    tail = "\n\n".join([
        "Check: are the rooms the right ones for the request, sized sensibly, connected "
        "by doors, and does the cutaway look like a coherent house? If something is wrong, "
        "return a corrected plan in the same JSON shape as the plan below; else null.",
        "The plan as JSON:\n" + _bp_json(bp),
        'Reply as JSON: {"ok": true/false, "issues": ["..."], "blueprint": <the corrected '
        'plan object, or null>}',
        JSON_RULE])
    return Message("user", [Part("text", text=head),
                            Part("text", text="SCHEMATIC:"), Part("image", image_png=schematic_png),
                            Part("text", text="PICTURED AS A CUTAWAY:"),
                            Part("image", image_png=dream_png),
                            Part("text", text=tail)])


# ===========================================================================
# Build turns (end with the action space's output rule)
# ===========================================================================
def shell_build(ep: Any, bp: "Blueprint", schematics: Sequence[bytes]) -> Message:
    """The LLM writes the shell from the plan (control arm ``shell: llm``)."""
    parts: List[Part] = [Part("text", text=f"Build the SHELL of this building: {ep.prompt}")]
    for i, png in enumerate(schematics):
        parts += [Part("text", text=f"PLAN storey {i}:"), Part("image", image_png=png)]
    text = "\n\n".join([
        canvas_line(ep) + " (x by y-up by z); the front faces +z (the z1 side).",
        "The floor plan as JSON (all coordinates inclusive, in blocks; room rects include "
        "their walls, neighbouring rooms share a wall line):\n" + _bp_json(bp),
        "Build only: the foundation, the floors, the outer walls, the inner walls between "
        "rooms, the door and window openings (with doors and glass), and the stairs. No "
        "furniture, and no roof: the roof and the rooms come later. Use the plan's "
        "materials.",
        "Write the COMPLETE program.",
        _rule(ep)])
    parts.append(Part("text", text=text))
    return Message("user", parts)


def unit_build(ep: Any, bp: Optional["Blueprint"], unit: "Unit", *, target: Optional[bytes],
               current: Optional[bytes], schematic: Optional[bytes], mode: str,
               local: bool) -> Message:
    """First turn of one room: target dream, the room as it is now, the plan."""
    parts: List[Part] = [Part("text", text=f"Furnish one room of this building: {ep.prompt}")]
    if target is not None:
        parts += [Part("text", text=f"TARGET {unit.key} (what this room should look like; "
                                    "compared with a cutaway render of your room):"),
                  Part("image", image_png=target)]
    if current is not None:
        parts += [Part("text", text=f"CURRENT {unit.key} (cutaway render of the room as it "
                                    "is now, walls cut at half height):"),
                  Part("image", image_png=current)]
    if schematic is not None:
        parts += [Part("text", text=f"PLAN storey {unit.storey} (this room outlined in red):"),
                  Part("image", image_png=schematic)]
    brief = [f"Room: {unit.name}, a {unit.purpose}, on storey {unit.storey}."]
    if unit.desc:
        brief.append(f"Brief: {unit.desc}")
    if target is not None:
        brief.append("Match the TARGET's furniture layout, materials and colours as "
                     "closely as blocks allow.")
    else:
        brief.append("Furnish it so it clearly reads as a " + unit.purpose + ": furniture "
                     "from blocks (slabs, stairs, fences, trapdoors, carpets, lanterns), "
                     "against the walls, leaving a path from every doorway.")
    text = "\n\n".join([region_block(unit, local=local, mode=mode), "\n".join(brief),
                        canvas_line(ep) + " (x by y-up by z).",
                        _mode(mode), _rule(ep)])
    parts.append(Part("text", text=text))
    return Message("user", parts)


def unit_compare(ep: Any, unit: "Unit", att: Attempt, *, target: bytes,
                 yours: Optional[bytes], score: Optional[float], mode: str, local: bool,
                 feedback: str) -> Message:
    """A room's match turn: TARGET room dream vs a cutaway render of the room."""
    parts: List[Part] = [Part("text", text=f"Compare your room with its target, for: "
                                           f"{ep.prompt}")]
    if feedback in ("raw", "both", "score"):
        parts += [Part("text", text=f"TARGET {unit.key}:"), Part("image", image_png=target)]
        if yours is not None:
            parts += [Part("text", text=f"YOURS {unit.key} (cutaway render, same camera):"),
                      Part("image", image_png=yours)]
    text: List[str] = [region_block(unit, local=local, mode=mode)]
    if score is not None and feedback in ("score", "both"):
        text.append(f"Room match score: {score:.3f} (0 = nothing like the target, 1 = "
                    "identical).")
    text.append("Execution of your last program:\n" + att.report.summary())
    text += _last_note(att)
    text.append("Fix the biggest differences first: missing furniture, then its placement, "
                "then materials and colours. Keep every doorway clear.")
    text.append(_mode(mode))
    text.append(_rule(ep))
    parts.append(Part("text", text="\n\n".join(text)))
    return Message("user", parts)


def unit_critique(ep: Any, unit: "Unit", att: Attempt, *, yours: Optional[bytes], mode: str,
                  local: bool) -> Message:
    """A room's critique turn without a target image (the no-dream control)."""
    parts: List[Part] = [Part("text", text=f"Review the room you furnished, for: {ep.prompt}")]
    if yours is not None:
        parts += [Part("text", text=f"YOURS {unit.key} (cutaway render, walls cut at half "
                                    "height):"),
                  Part("image", image_png=yours)]
    text = "\n\n".join([
        region_block(unit, local=local, mode=mode),
        f"Room: {unit.name}, a {unit.purpose}." + (f" Brief: {unit.desc}" if unit.desc else ""),
        "Execution of your last program:\n" + att.report.summary(),
        *_last_note(att),
        "Critique it in your head (do not write the critique), then act on the three "
        "biggest problems: does it read as a " + unit.purpose + ", is the furniture against "
        "the walls and in proportion, is every doorway clear, is anything floating?",
        _mode(mode), _rule(ep)])
    parts.append(Part("text", text=text))
    return Message("user", parts)


def unit_repair(ep: Any, unit: "Unit", att: Attempt, *, local: bool,
                max_issues: int = 20) -> Message:
    """Execution feedback for one unit (the unit-scoped twin of ``turns.repair``).

    Restates the region: the reverted writes ARE the usual reason for this turn.
    """
    report = att.report
    errors = [i for i in report.issues if i.severity == "error"][:max_issues]
    warns = [i for i in report.issues if i.severity != "error"][:max_issues]
    lines = [region_block(unit, local=local), "",
             f"Your program for {unit.key} ran. Result:", report.summary(), ""]
    if errors:
        lines.append("Fix these failed lines:")
        lines += [f"  - {i.render()}" for i in errors]
    if warns:
        lines.append("These lines placed nothing, or placed blocks outside the region:")
        lines += [f"  - {i.render()}" for i in warns]
    if att.blocks == 0:
        lines.append("Nothing is left at all.")
    lines += ["", "Rewrite your program for this unit, corrected.", _rule(ep)]
    return Message("user", [Part("text", text="\n".join(lines))])


def storey_build(ep: Any, bp: "Blueprint", unit: "Unit", *, target: Optional[bytes],
                 below: Optional[bytes], exterior: Optional[bytes],
                 schematic: Optional[bytes], mode: str, local: bool) -> Message:
    """First turn of one storey (method ``floors``), bottom up."""
    parts: List[Part] = [Part("text", text=f"Build one storey of this building: {ep.prompt}")]
    if exterior is not None:
        parts += [Part("text", text="EXTERIOR (the whole building, for its look and "
                                    "materials):"), Part("image", image_png=exterior)]
    if target is not None:
        parts += [Part("text", text=f"TARGET {unit.key} (this storey, everything above "
                                    "removed; compared with a render of your storey):"),
                  Part("image", image_png=target)]
    if below is not None:
        parts += [Part("text", text="BELOW (the storey underneath, already built):"),
                  Part("image", image_png=below)]
    if schematic is not None:
        parts += [Part("text", text=f"PLAN storey {unit.storey}:"),
                  Part("image", image_png=schematic)]
    st = bp.storey(unit.storey)
    fp = bp.footprint_of(unit.storey)
    # In local mode every number in the turn is in the region's frame, like the
    # REGION line (a mixed frame would put the walls one footprint off).
    ox, oy, oz = unit.region.origin if local else (0, 0, 0)
    brief = [f"Storey {unit.storey}: floor at y={st.floor_y - oy}, walls y "
             f"{st.floor_y + 1 - oy}..{st.floor_y + st.height - oy}, footprint x "
             f"{fp.x0 - ox}..{fp.x1 - ox}, z {fp.z0 - oz}..{fp.z1 - oz} (walls on its border).",
             "Build the floor slab, outer walls, inner walls, doors, windows, stairs up "
             "and the furniture of THIS storey only."]
    rooms = _purpose_lines(bp, unit.storey, origin=(ox, oz))
    if rooms:
        brief.append("Rooms (rects include their walls):\n" + "\n".join(rooms))
    if unit.storey == 0:
        brief.append("The front door is on the +z side (z1).")
    # Where the plan's stair up goes: its top step and stairwell lie in the slab
    # above, which this storey may write only there ("Also allowed" in the region).
    above = {s.index for s in bp.storeys}
    for stair in bp.stairs:
        if stair.from_storey != unit.storey or unit.storey + 1 not in above:
            continue
        cells = stair.cells(st.height)
        (x0, dy0, z0), (x1, dy1, z1) = cells[0], cells[-1]
        brief.append(f"Stairs up ({stair.kind}): first step at x {x0 - ox}, y "
                     f"{st.floor_y + dy0 - oy}, z {z0 - oz}; top step at x {x1 - ox}, y "
                     f"{st.floor_y + dy1 - oy}, z {z1 - oz}, in the floor slab above. Leave "
                     "the slab cells over the last steps open (air) as the stairwell.")
    text = "\n\n".join([region_block(unit, local=local, mode=mode), "\n".join(brief),
                        canvas_line(ep) + " (x by y-up by z).", _mode(mode), _rule(ep)])
    parts.append(Part("text", text=text))
    return Message("user", parts)


def roof_build(ep: Any, bp: "Blueprint", *, exterior: Optional[bytes],
               current: Optional[bytes]) -> Message:
    """The LLM writes the roof on the finished storeys (``roof: llm``)."""
    parts: List[Part] = [Part("text", text=f"Put the roof on this building: {ep.prompt}")]
    if exterior is not None:
        parts += [Part("text", text="TARGET exterior:"), Part("image", image_png=exterior)]
    if current is not None:
        parts += [Part("text", text="CURRENT build (render, roof missing):"),
                  Part("image", image_png=current)]
    fp = bp.footprint
    r = bp.roof
    text = "\n\n".join([
        canvas_line(ep) + " (x by y-up by z).",
        f"The walls end at y={bp.top_y() - 1}; the roof starts at y={bp.top_y()} over the "
        f"footprint x {fp.x0}..{fp.x1}, z {fp.z0}..{fp.z1}. Planned roof: {r.kind}"
        + (f" with its ridge along {r.axis}" if r.kind == "gable" else "")
        + f", overhang {r.overhang}, material {r.block or bp.materials.get('roof', '')}.",
        "Build only the roof (and its trim). The rooms inside are finished: anything you "
        "place inside them is discarded.",
        MODE_INSTRUCTIONS["patch"], _rule(ep)])
    parts.append(Part("text", text=text))
    return Message("user", parts)


__all__ = ["BLUEPRINT_TAG", "BRIEF_TAG", "CANVAS_LINE", "DISCARD_NOTE", "PLAN_DREAM", "PLAN_STYLIZE",
           "REGION_LINE", "ROOM_DREAM", "STOREY_DREAM", "UNIT_TAG", "VERIFY_TAG",
           "blueprint_fix", "blueprint_json", "canvas_line", "design_brief", "floor_line",
           "plan_extract",
           "plan_verify",
           "region_block", "region_line", "roof_build", "shell_build", "storey_build",
           "unit_build", "unit_compare", "unit_critique", "unit_repair"]
