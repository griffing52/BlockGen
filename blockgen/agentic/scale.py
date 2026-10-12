"""Player scale: the wording that tells a model how big a Minecraft player is, and
a cheap heuristic that checks a finished build against it.

Why this exists: without a stated unit a model sizes a house by eye, and the
usual failure is a build that looks right from outside but is a doll's house
inside: 1-block doorways, 2-block ceilings, 2 x 2 rooms. One sentence of scale
("a player is 2 blocks tall") fixes most of it, and the numbers below are the
ones a Minecraft builder uses without thinking.

Three consumers share the same numbers so they never disagree:

``scale_rules``      the system text of the ``scale`` context (context.py), which
                     every role that sees the builder system prompt receives
                     (builder, planner, critic), in any action space or method
``plan_line`` /      one-line reminders for the text plan (turns.plan) and the
``blueprint_line``   blueprint planner turns (turns_staged), only while the recipe
                     lists the ``scale`` context, so ``*-noscale`` controls stay clean
``DREAM_CUE``        a phrase added after the dream style (dreams.render_prompt)
                     with ``scale: {dream_cue: true}``. It describes the openings
                     and never asks for a person in the picture: the style text
                     says "no people", and a figure would be a second structure
                     for the silhouette and embedding scores to match.

The diagnostic (:func:`scale_stats`) is a heuristic and is documented as one: it
reads the final voxel grid, finds the cells a player could stand in under a roof,
and measures their headroom; and it counts 1-wide wall openings by their height.
It is reported in the method metrics, never used for selection by default.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

#: Player height and width in blocks (1.8 x 0.6 in the game; a player needs a
#: 2-block-tall, 1-block-wide gap to pass).
PLAYER_HEIGHT, PLAYER_WIDTH = 2, 1
#: Minimum clear air between a floor and the ceiling above it.
MIN_CLEAR_HEIGHT = 3
#: Minimum open floor inside a room's walls (blocks per side).
MIN_ROOM_INTERIOR = 3

RULES = """\
Player scale (applies to buildings and anything a player walks into): one block is
one metre, and a Minecraft player is 2 blocks tall and 1 block wide (1.8 x 0.6
exactly). Size everything for that player:
- Doors: 1 wide x 2 tall (double doors 2 wide); a door's top is 2 blocks above the
  floor it stands on. A doorway is never 1 block tall.
- Ceilings: at least 3 blocks of clear air between a floor and the ceiling above it,
  so one storey is 4-5 blocks floor-to-floor including its floor/ceiling slab.
- Rooms, measured as open floor inside the walls: at least 3 x 3; a bedroom about
  4 x 5 to 6 x 6, a hall or living room 6 x 8 or more.
- Corridors: at least 1 block wide inside, 2 is better.
- Windows: start 1 block above the floor, 1-2 blocks tall.
- Stairs: rise 1 block per 1 block of run, with at least 2 blocks of air above every
  step (cut the stairwell through the floor above).
- Furniture is 1 block (a bed is 1 x 2, a table or chair 1 block high).
- Outside size follows the rooms: a 2-3 room cottage is about 10-16 blocks per side,
  walls included, and 1-2 storeys."""

STRICT_NOTE = ("These are hard rules: a build that breaks them is wrong even if it "
               "looks right from outside.")

#: After the dream style (``{style}``); the image model gets openings to size the
#: building by, not a person to draw.
DREAM_CUE = ("Sized for a Minecraft player: doors are two blocks tall and one block "
             "wide, ceilings three blocks high, windows one block above the floor.")


def storeys_that_fit(height: int, roof: int = 4) -> int:
    """Storeys of 5 blocks floor-to-floor under a ``roof``-block roof (>= 0)."""
    return max(0, (int(height) - 1 - int(roof)) // 5)


def canvas_note(size: Tuple[int, int, int], max_storeys: Optional[int] = None) -> str:
    """What the canvas holds at player scale, and what to do when it is too small.

    ``max_storeys`` (a method's own cap, e.g. blueprint's 2) replaces the canvas's
    storey count when it is lower, so the system text never offers more storeys
    than the planner turn allows."""
    w, h, d = (int(v) for v in size)
    n = storeys_that_fit(h)
    if n <= 0:
        fits = (f"This {w} x {h} x {d} canvas is low at player scale: one storey with a "
                "flat or very shallow roof at most.")
    elif min(w, d) < 7:
        fits = (f"This {w} x {h} x {d} canvas is narrow at player scale: one small room "
                "per storey at most.")
    elif max_storeys is not None and 0 < int(max_storeys) < n:
        m = int(max_storeys)
        fits = (f"This {w} x {h} x {d} canvas holds the {m} storey"
                f"{'s' if m != 1 else ''} this method allows under a roof, on a "
                f"footprint of up to {w} x {d}; use only what the request needs.")
    else:
        fits = (f"This {w} x {h} x {d} canvas has room for at most about {n} storey"
                f"{'s' if n != 1 else ''} under a roof, on a footprint of up to "
                f"{w} x {d}; use only what the request needs.")
    return (fits + " If the request does not fit at player scale, simplify it (fewer "
            "rooms, fewer storeys, a smaller footprint) rather than shrinking doors, "
            "ceilings or rooms.")


def scale_rules(size: Optional[Tuple[int, int, int]] = None, *, strict: bool = False,
                canvas: bool = True, max_storeys: Optional[int] = None) -> str:
    """The ``scale`` context's system text."""
    out = [RULES]
    if canvas and size is not None:
        out.append(canvas_note(size, max_storeys))
    if strict:
        out.append(STRICT_NOTE)
    return "\n".join(out)


def plan_line(dream: bool = False) -> str:
    """The scale reminder of a text plan turn (turns.plan).

    ``dream`` is the dream-reading form (``dream-plan``): there the TARGET image
    is what the build is scored against, so the line keeps its proportions and
    uses its own openings as the ruler instead of telling the planner to ignore
    how big it looks."""
    if dream:
        return ("- Size it for a Minecraft player (2 blocks tall): keep the TARGET's "
                "proportions and count its doors and storeys to set the scale: a door "
                "is 1 x 2 blocks, a storey 4-5 blocks floor-to-floor (at least 3 of "
                "clear ceiling), rooms at least 3 x 3 inside.")
    return ("- Size it for a Minecraft player (2 blocks tall): doors 1 x 2, at least 3 "
            "blocks of clear ceiling height per storey (4-5 floor-to-floor), rooms at "
            "least 3 x 3 inside. Get the footprint from the rooms it needs at this "
            "scale, not from how big it looks.")


def blueprint_line() -> str:
    """The scale paragraph of the blueprint planner turns.

    Only the blueprint-unit facts: the system prompt already carries
    :data:`RULES` (sent only with the ``scale`` context, as is this line), whose
    sizes count open floor. Corridors follow the repair's minimum
    (``blueprint.MIN_ROOM_SIDE``): the blueprint has no corridor primitive, so a
    narrower corridor would be dropped."""
    return ("Player scale in blueprint units: room rects include their walls, so a "
            "rect is 2 more per side than its open floor. A room needs a rect of at "
            "least 5 x 5 (3 x 3 of floor); a bedroom is about 6 x 7 to 8 x 8, a hall "
            "or living room 8 x 10 or more. A hall or corridor is a room like any "
            "other, at least 5 wide including walls (3 of floor); there are no "
            "narrower corridors. Keep every storey's \"height\" (clear air between "
            "slabs) at 3 or more. If the rooms do not fit the canvas at this size, plan "
            "fewer rooms rather than smaller ones.")


# ===========================================================================
# Recipe lookup (which recipes have the scale context, with which params)
# ===========================================================================
def scale_params(spec: Any) -> Optional[Dict[str, Any]]:
    """The params of the recipe's ``scale`` context, or None when it has none.

    Read from the recipe (``spec.contexts``), not from the evaluated contexts:
    dreams are generated BEFORE the system prompt is built, and the dream cue
    must not wait for it.
    """
    from blockgen.agentic.core import parse_spec

    for c in getattr(spec, "contexts", None) or []:
        try:
            name, _, params = parse_spec(c)
        except ValueError:
            continue
        if name == "scale":
            return dict(params or {})
    return None


def scale_active(ep: Any) -> bool:
    """Whether the episode's recipe lists the ``scale`` context."""
    return scale_params(getattr(ep, "spec", None)) is not None


def method_max_storeys(spec: Any) -> Optional[int]:
    """The recipe's storey cap: ``params.max_storeys``, else the default of its
    method function's ``max_storeys`` keyword (blueprint, floors), else None
    (methods without a cap, e.g. classic and dream)."""
    params = dict(getattr(spec, "params", None) or {})
    if params.get("max_storeys") is not None:
        return int(params["max_storeys"])
    import inspect

    from blockgen.agentic.core import lookup
    try:
        fn = lookup("method", str(getattr(spec, "method", ""))).load()
        p = inspect.signature(fn).parameters.get("max_storeys")
    except Exception:  # unknown method, or not introspectable: no cap
        return None
    if p is None or p.default is inspect.Parameter.empty or p.default is None:
        return None
    return int(p.default)


def dream_cue(ep: Any) -> str:
    """:data:`DREAM_CUE` when the recipe's scale context has ``dream_cue`` on
    (the default), else ''."""
    p = scale_params(getattr(ep, "spec", None))
    if p is None or not bool(p.get("dream_cue", True)):
        return ""
    return DREAM_CUE


# ===========================================================================
# Diagnostic (heuristic): headroom of covered standing cells, door-like openings
# ===========================================================================
#: Legacy block ids a player walks through (air, doors, carpets, torches, plants,
#: signs, ladders, rails, buttons, plates, snow layers). Everything else counts as
#: solid. Approximate on purpose: a slab or a fence is "solid" here.
PASSABLE_IDS = frozenset({0, 6, 27, 28, 31, 32, 37, 38, 39, 40, 50, 55, 59, 63, 64, 65, 66,
                          68, 69, 70, 71, 72, 75, 76, 77, 78, 83, 93, 94, 106, 111, 131,
                          132, 141, 142, 143, 147, 148, 149, 150, 157, 171, 175, 193, 194,
                          195, 196, 197})
#: Door blocks (wooden, iron, spruce .. dark oak).
DOOR_IDS = frozenset({64, 71, 193, 194, 195, 196, 197})
#: Furniture-height solids (note block, bed, bookshelf, chests, crafting table,
#: furnaces, jukebox, fences, cake, enchanting table, brewing stand, cauldron,
#: anvil, hopper, skull, flower pot). Solid for headroom, but never the floor of
#: a doorway: a bed or table under an open window is not a threshold.
FURNITURE_IDS = frozenset({25, 26, 47, 54, 58, 61, 62, 84, 85, 92, 113, 116, 117, 118,
                           130, 140, 144, 145, 146, 154, 188, 189, 190, 191, 192})


def _passable(block_ids: np.ndarray) -> np.ndarray:
    return np.isin(block_ids, np.fromiter(PASSABLE_IDS, dtype=np.int64))


def _shift(a: np.ndarray, axis: int, k: int, fill: bool) -> np.ndarray:
    """``out[i] = a[i + k]`` along ``axis`` (``fill`` past the edge)."""
    out = np.full_like(a, fill)
    n = a.shape[axis]
    if abs(k) >= n:
        return out
    src = [slice(None)] * a.ndim
    dst = [slice(None)] * a.ndim
    src[axis] = slice(k, None) if k > 0 else slice(0, n + k)
    dst[axis] = slice(0, n - k) if k > 0 else slice(-k, None)
    out[tuple(dst)] = a[tuple(src)]
    return out


def scale_stats(structure: Any) -> Dict[str, Any]:
    """Heuristic player-scale numbers for a finished build (XYZ grid, y up).

    * ``scale_cells``: covered standing cells, i.e. a passable cell on top of a
      solid block with a solid block somewhere above it in the same column
      (inside a building or under a roof), minus the sloped low ones: a cell
      whose ceiling is not level with its passable neighbours' at the same height
      AND that has under :data:`MIN_CLEAR_HEIGHT` of air. Those are the eaves of
      a pitched roof or attic, the edge of an overhang and the doorway itself,
      none of them a room; a level ceiling (a flat-roofed room, a ceiling slab)
      always counts, so a doll's house with 2-block ceilings still reads low.
    * ``scale_headroom2`` / ``scale_headroom3``: fraction of those cells with at
      least 2 / 3 passable cells straight up (a player fits / a comfortable
      ceiling). Furniture tops count as floor, so a furnished room scores a bit
      lower than an empty one.
    * ``scale_doorways``: wall openings at least 2 tall, one per opening: a
      1-wide gap at floor level under a lintel (solid somewhere above), walls on
      both sides along one axis, open on both sides along the other with a real
      floor (not furniture, :data:`FURNITURE_IDS`) on at least one open side,
      through a wall at most 2 thick. Longer runs are corridors, not doorways,
      except at a door block, which always counts; open-topped gaps (crenels)
      have no lintel.
    * ``scale_low_openings``: the same openings only 1 tall (a player cannot pass).
    * ``scale_door_blocks``: door columns (a door block with a door block above).
    * ``scale_door_ok``: doorways / (doorways + low openings), None if neither.

    None for every value when there are no blocks.
    """
    ids = np.asarray(getattr(structure, "block_ids", structure))
    out: Dict[str, Any] = {"scale_cells": 0, "scale_headroom2": None,
                           "scale_headroom3": None, "scale_doorways": 0,
                           "scale_low_openings": 0, "scale_door_blocks": 0,
                           "scale_door_ok": None}
    if ids.ndim != 3 or ids.size == 0 or not (ids != 0).any():
        return out
    pas = _passable(ids)
    solid = ~pas
    X, Y, Z = ids.shape
    # covered[x, y, z]: some solid block strictly above (x, y, z).
    above = np.zeros_like(solid)
    above[:, :-1, :] = np.flip(np.logical_or.accumulate(np.flip(solid[:, 1:, :], 1), 1), 1)
    on_solid = np.zeros_like(solid)
    on_solid[:, 1:, :] = solid[:, :-1, :]
    stand = pas & on_solid & above
    # Headroom: consecutive passable cells from y upward (including y).
    run = np.zeros(ids.shape, dtype=np.int32)
    run[:, Y - 1, :] = pas[:, Y - 1, :]
    for y in range(Y - 2, -1, -1):
        run[:, y, :] = np.where(pas[:, y, :], run[:, y + 1, :] + 1, 0)
    # Ceiling height (y of the first solid above; Y for open sky) of every passable
    # cell, compared with the passable horizontal neighbours at the same y (outside
    # the grid is open sky).
    ceil = np.where(pas, np.arange(Y)[None, :, None] + run, -1)
    level = np.ones_like(solid)
    for axis in (0, 2):
        for k in (-1, 1):
            npas = _shift(pas, axis, k, True)
            nceil = _shift(ceil, axis, k, Y)
            level &= ~npas | (nceil == ceil)
    counted = stand & (level | (run >= MIN_CLEAR_HEIGHT))
    n = int(counted.sum())
    out["scale_cells"] = n
    if n:
        hr = run[counted]
        out["scale_headroom2"] = round(float((hr >= 2).mean()), 4)
        out["scale_headroom3"] = round(float((hr >= 3).mean()), 4)
    # Wall openings (see the docstring). The floor rule keeps an unglazed window
    # out (its open sides drop to the room floor and to the ground, or rest on
    # furniture); a front door passes on its inside floor.
    floor = np.zeros_like(solid)
    furn = np.isin(ids, np.fromiter(FURNITURE_IDS, dtype=np.int64))
    floor[:, 1:, :] = (solid & ~furn)[:, :-1, :]
    base = pas & on_solid & above
    door = np.isin(ids, np.fromiter(DOOR_IDS, dtype=np.int64))
    doorways = low = 0
    for wall_axis, open_axis in ((0, 2), (2, 0)):
        walls = _shift(solid, wall_axis, -1, False) & _shift(solid, wall_axis, 1, False)
        opn = (_shift(pas, open_axis, -1, True) & _shift(pas, open_axis, 1, True)
               & (_shift(floor, open_axis, -1, False) | _shift(floor, open_axis, 1, False)))
        gap = base & walls & opn
        # One count per opening: the first cell of a run along the open axis, and
        # only for runs of at most 2 (a wall up to 2 thick); longer is a corridor.
        # A door block is an opening wherever it stands (a door at the end of a
        # corridor), and a short run holding one is counted by its door alone.
        start = gap & ~_shift(gap, open_axis, -1, False)
        short = ~(_shift(gap, open_axis, 1, False) & _shift(gap, open_axis, 2, False))
        gdoor = gap & door
        has_door = gdoor | (_shift(gap, open_axis, 1, False) & _shift(gdoor, open_axis, 1, False))
        first = (start & short & ~has_door) | gdoor
        doorways += int((first & (run >= 2)).sum())
        low += int((first & (run == 1)).sum())
    out["scale_doorways"] = doorways
    out["scale_low_openings"] = low
    out["scale_door_blocks"] = int((door[:, :-1, :] & door[:, 1:, :]).sum())
    tot = out["scale_doorways"] + out["scale_low_openings"]
    out["scale_door_ok"] = round(out["scale_doorways"] / tot, 4) if tot else None
    return out


__all__ = ["DOOR_IDS", "DREAM_CUE", "FURNITURE_IDS", "MIN_CLEAR_HEIGHT", "MIN_ROOM_INTERIOR",
           "PASSABLE_IDS", "PLAYER_HEIGHT", "PLAYER_WIDTH", "RULES", "STRICT_NOTE",
           "blueprint_line", "canvas_note", "dream_cue", "method_max_storeys", "plan_line",
           "scale_active", "scale_params", "scale_rules", "scale_stats", "storeys_that_fit"]
