"""Prompt construction for the agentic track.

Every prompt is assembled here, and the command reference inside the system
prompt is *generated from the DSL registry* (:func:`blockgen.agentic.dsl.
command_reference`). Adding a command therefore documents it to the model
automatically; a hand-maintained copy would drift on the first change and the
failure mode (the model using a command that no longer parses) is silent.

The four prompt stages map to the four agent stages:

``system``   the language, the canvas, the palette, the house-building method
``plan``     natural-language design pass — dimensions, palette, part list
``build``    emit the program (optionally after a plan, optionally with examples)
``repair``   here is what your program did and what broke; fix it
``critique`` here is a *render* of what you built; revise it

Each returns a list of :class:`blockgen.agentic.providers.Message`, so the agent
stays a control loop and the wording lives in one file you can iterate on.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from blockgen.agentic.blockstate import PALETTE
from blockgen.agentic.dsl import ExecutionReport, command_reference
from blockgen.agentic.examples import Example
from blockgen.agentic.providers import Message, image_message, text_message

# The build method we want the model to follow. This is the one piece of prompt
# text that is *domain* knowledge rather than mechanics: our failure analyses
# (notes.md §T10/T11) all trace bad builds to missing structure -- no floor, walls
# that do not close, a roof that floats -- so the order is stated explicitly.
METHOD = """\
Method — follow this order, it is what makes a build read as a real structure:
1. Decide the footprint and lay a floor/foundation first, so nothing floats.
2. Raise the shell (`walls` / `box`), keeping walls CLOSED — a gap that is not a
   deliberate window or door reads as a broken build.
3. Cut openings with `clear`, then dress them (`oak_door`, `glass_pane`).
4. Roof it: `gable` for pitched roofs, a flat `fill` slab for modern ones. A roof
   must sit ON the walls (its base y = the wall top y), never one block above.
5. Add depth last: corner posts, trim, stairs, torches, a chimney, planting.
Keep every part connected to its neighbour — the whole build should be one solid
piece, not islands."""

RULES = """\
Rules:
- Output ONLY command lines. No prose, no explanation, no code fences.
- One command per line. `#` starts a comment; comments are encouraged and free.
- Coordinates are integers, INCLUSIVE at both ends, y is up, origin (0,0,0) is the
  corner of the canvas. Stay inside the canvas — writes outside it are dropped.
- Build ON the ground: the lowest solid part of the build should sit at y=0.
- Use the block names from the palette below (other modern Minecraft names usually
  work too, but unknown names fail and waste the line).
- Prefer the region commands (`fill`, `walls`, `box`, `gable`, `copy`/`paste`,
  `stack`) over placing blocks one at a time — that is the whole point of the
  language, and it is what lets you build something large."""


def _palette_block(palette: Sequence[str] = PALETTE) -> str:
    return "Palette (known-good block names):\n  " + "\n  ".join(
        ", ".join(palette[i:i + 8]) for i in range(0, len(palette), 8))


def system_prompt(canvas_size: Tuple[int, int, int],
                  palette: Sequence[str] = PALETTE,
                  extra: str = "") -> Message:
    """The shared system message: language + canvas + palette + method."""
    sx, sy, sz = canvas_size
    body = f"""\
You are a master Minecraft builder. You do not place blocks by hand — you write a
short program in the BlockGen build language, and the program is executed to
produce the structure.

Canvas: {sx} wide (x) x {sy} tall (y) x {sz} deep (z). Valid coordinates are
x in 0..{sx - 1}, y in 0..{sy - 1}, z in 0..{sz - 1}.

Commands:
{command_reference()}

{RULES}

{METHOD}

{_palette_block(palette)}"""
    if extra:
        body += f"\n\n{extra}"
    return text_message("system", body)


def plan_prompt(description: str, canvas_size: Tuple[int, int, int]) -> Message:
    """Ask for a design pass before any code — the 'plan first' ablation arm."""
    sx, sy, sz = canvas_size
    return text_message("user", f"""\
Build request: {description}

Before writing any commands, plan the build. Answer in at most 12 short lines:
- Overall footprint and height in blocks (must fit {sx}x{sy}x{sz}).
- The material palette: which block for foundation, walls, trim, roof, glazing.
- The parts list, in build order, each with its coordinate range — e.g.
  "foundation: fill 0 0 0 to 15 0 11", "walls: 1..14 x, y 1..5, z 1..10",
  "roof: gable over the footprint from y=6".
- Two details that will make it look deliberate rather than generic.
Plain text only. No commands yet.""")


def build_prompt(description: str, *, plan: Optional[str] = None,
                 examples: Sequence[Example] = (),
                 images: Sequence[bytes] = (),
                 image_note: str = "") -> List[Message]:
    """The generation turn: optional examples, optional plan, optional reference
    images, then the request."""
    messages: List[Message] = []
    for ex in examples:
        messages.append(text_message("user", f"Build: {ex.caption}"))
        messages.append(text_message("assistant", ex.program.strip()))
    ask = f"Build: {description}"
    if plan:
        ask += (f"\n\nYour plan:\n{plan.strip()}\n\nNow write the program that "
                f"realizes this plan. Output only command lines.")
    else:
        ask += "\n\nWrite the program. Output only command lines."
    if images:
        note = image_note or ("Reference image(s) of what to build. Match the massing, "
                              "proportions and materials as closely as the block "
                              "palette allows.")
        messages.append(image_message("user", f"{note}\n\n{ask}", images))
    else:
        messages.append(text_message("user", ask))
    return messages


def repair_prompt(report: ExecutionReport, *, description: str,
                  max_issues: int = 20) -> Message:
    """Feed execution results back. Errors first — they are actionable and cheap."""
    errors = [i for i in report.issues if i.severity == "error"][:max_issues]
    warnings = [i for i in report.issues if i.severity == "warning"][:max_issues]
    lines = [f"Your program ran. Result for: {description}", "", report.summary(), ""]
    if errors:
        lines.append("Fix these failed lines:")
        lines += [f"  - {i.render()}" for i in errors]
    if warnings:
        lines.append("These lines placed nothing — usually a coordinate mistake:")
        lines += [f"  - {i.render()}" for i in warnings]
    if report.blocks == 0:
        lines.append("Nothing was built at all. Re-read the coordinate rules and "
                     "start from a `fill` at y=0.")
    lines.append("")
    lines.append("Rewrite the COMPLETE program, corrected. Output only command lines.")
    return text_message("user", "\n".join(lines))


def critique_prompt(description: str, report: ExecutionReport,
                    images: Sequence[bytes], *, views: Sequence[str] = (),
                    mode: str = "rewrite") -> Message:
    """Show the model a render of its own build and ask for a revision.

    ``mode='rewrite'`` asks for the whole program again (simplest, and the model
    can restructure); ``mode='patch'`` asks for additional commands appended to the
    existing canvas (cheaper, and it cannot destroy what already works).
    """
    view_note = f" The views are {', '.join(views)}." if views else ""
    stats = report.summary()
    if mode == "patch":
        instruction = (
            "Write ONLY the additional commands that improve this build. They run "
            "on the canvas as it already is, so do not repeat what is already "
            "correct; use `clear` to remove anything wrong. Output only command lines.")
    else:
        instruction = (
            "Rewrite the COMPLETE program, improved. Keep what works, fix what does "
            "not. Output only command lines.")
    return image_message("user", f"""\
Here is a render of the build your program produced for: {description}{view_note}

{stats}

Critique it against the request in your head (do not write the critique), then act
on the three biggest problems. Look specifically for: floating or disconnected
parts, walls that do not close, a roof that does not sit on the walls, missing
door/windows, flat untextured surfaces that need trim, and whether the size and
style actually match the request.

{instruction}""", images)


def scale_hint(target_blocks: Optional[int]) -> str:
    """Optional system-prompt rider nudging build size (used by the scale arms)."""
    if not target_blocks:
        return ""
    return (f"Aim for roughly {target_blocks} placed blocks — a build of that mass "
            f"needs multiple parts (wings, storeys, outbuildings, landscaping), not "
            f"one bigger box.")


__all__ = ["METHOD", "RULES", "build_prompt", "critique_prompt", "plan_prompt",
           "repair_prompt", "scale_hint", "system_prompt"]
