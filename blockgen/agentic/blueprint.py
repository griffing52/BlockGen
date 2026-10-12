"""Blueprints: a typed floor plan the staged methods build from.

Why a typed plan: a free-text plan cannot be validated, rendered, turned into a
region, or built deterministically. A Blueprint is all four. It is produced by an
LLM (JSON), by a VLM reading a dreamed plan image, or by both, and is always passed
through :meth:`Blueprint.repaired` before use, so a method never builds from a plan
that overlaps itself, leaves the canvas, or has a room nobody can walk into.

Conventions (the staged-methods spec, section A1; every method codes against them):

* canvas ``(W, H, D)`` = ``(x, y, z)``, ``y`` up, inclusive integer coordinates;
* a :class:`Rect` is a footprint rectangle in ``(x, z)`` INCLUDING its walls;
  neighbouring rooms SHARE the wall line, and a room's interior is the rect shrunk by 1;
* storey ``i`` has its floor slab at ``floor_y``, walls and air at
  ``floor_y+1 .. floor_y+height`` and the next slab (or the roof base) at
  ``ceiling_y = floor_y + height + 1``; storey 0 has ``floor_y = 0``;
* the front is the ``z1`` side (faces +z), the side the iso camera sees.

Import-light: numpy only at module level; PIL and the block resolver are imported
inside the functions that need them.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from blockgen.agentic.region import Box

Coord2 = Tuple[int, int]
Size3 = Tuple[int, int, int]

ROOF_KINDS: Tuple[str, ...] = ("gable", "hip", "pyramid", "flat", "none")
OPENING_KINDS: Tuple[str, ...] = ("door", "window", "arch")
STAIR_DIRS: Tuple[str, ...] = ("+x", "-x", "+z", "-z")
ROOM_ORDERS: Tuple[str, ...] = ("blueprint", "size", "public_first")
SHELL_PARTS: Tuple[str, ...] = ("foundation", "floors", "walls", "corners", "partitions",
                                "openings", "stairs", "ceiling")
#: Storey ``height`` is the clear air between slabs, so the minimum 3 is already
#: the player-scale ceiling (a 2-tall player plus a block of headroom); one storey
#: is ``height + 1`` blocks floor-to-floor (4..9).
MIN_STOREY_HEIGHT, MAX_STOREY_HEIGHT = 3, 8
#: Rect side INCLUDING walls: 5 = a 3 x 3 interior, the smallest room a player can
#: use (was 4 = 2 x 2 until 2026-10-03; scale.MIN_ROOM_INTERIOR + 2).
MIN_ROOM_SIDE = 5
OUTSIDE = "outside"

#: role -> block. Unknown or unresolvable blocks are replaced by these in repair.
DEFAULT_MATERIALS: Dict[str, str] = {
    "foundation": "cobblestone", "floor": "oak_planks", "wall": "stone_bricks",
    "trim": "oak_log", "interior_wall": "oak_planks", "ceiling": "oak_planks",
    "roof": "spruce_stairs", "roof_flat": "spruce_slab", "window": "glass_pane",
    "door": "oak_door", "stairs": "oak_stairs", "light": "lantern",
}
#: Purposes visited first by room_order=public_first (substring match, lowercase).
PUBLIC_PURPOSES: Tuple[str, ...] = ("entrance", "hall", "foyer", "living", "great hall",
                                    "kitchen", "dining", "shop", "tavern", "workshop")
#: Pastel fill per purpose keyword in schematics (fallback: hash colour).
PURPOSE_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "hall": (255, 236, 179), "entrance": (255, 236, 179), "foyer": (255, 236, 179),
    "living": (255, 214, 165), "kitchen": (202, 255, 191), "dining": (253, 255, 182),
    "bed": (189, 224, 254), "sleep": (189, 224, 254), "bath": (160, 196, 255),
    "study": (255, 198, 255), "library": (255, 198, 255), "office": (255, 198, 255),
    "storage": (222, 222, 222), "store": (222, 222, 222), "pantry": (222, 222, 222),
    "workshop": (255, 173, 173), "forge": (255, 173, 173), "shop": (255, 205, 210),
    "tavern": (255, 224, 178), "loft": (204, 229, 255), "attic": (204, 229, 255),
    "chapel": (230, 210, 255), "garden": (190, 240, 200),
}

#: Ascent direction -> (dx, dz, block-state facing, schematic arrow).
_DIRS: Dict[str, Tuple[int, int, str]] = {
    "+x": (1, 0, "east"), "-x": (-1, 0, "west"), "+z": (0, 1, "south"), "-z": (0, -1, "north"),
}

#: The JSON shape shown to LLM planners and VLM extractors (turns_staged embeds it).
SCHEMA_TEXT = """\
Blueprint JSON (all units are blocks; x right, z towards the front, y up):
{
  "footprint": [x0, z0, x1, z1],          // outer walls of the building, inclusive
  "storeys": [{"height": 4}, ...],         // bottom up; height = air blocks between slabs (3..8)
  "rooms": [{"name": "hall", "purpose": "entrance hall", "storey": 0,
             "rect": [x0, z0, x1, z1], "desc": "what furnishes it"}],
  "openings": [{"kind": "door"|"window"|"arch", "storey": 0, "at": [x, z],
                "width": 1, "between": ["hall", "outside"]}],
  "stairs": [{"from_storey": 0, "at": [x, z], "dir": "+x"|"-x"|"+z"|"-z"}],
  "roof": {"kind": "gable"|"hip"|"pyramid"|"flat"|"none", "axis": "x"|"z", "overhang": 1},
  "materials": {"wall": "stone_bricks", "floor": "oak_planks", "roof": "spruce_stairs", ...},
  "style": "one line describing the exterior look"
}
Rules: a room rect INCLUDES its walls and neighbouring rooms SHARE a wall line
(hall [8,12,16,23] and kitchen [16,12,24,23] share x=16); every rect lies inside the
footprint; a room's interior is its rect shrunk by 1 on every side, and a rect is at
least 5 x 5 (smaller rooms are dropped). The front is the
z1 side of the footprint (put the entrance door there). A door or window "at" is a wall
cell: on a shared wall for a door between rooms, on the outer wall for "outside".
Storey 0's floor is at y=0; each next storey starts on the previous one's ceiling.
Stairs start inside a room at "at" and climb (height+1) blocks in "dir".
Example on a 32 x 32 x 32 canvas:
{"footprint": [8, 12, 24, 23], "storeys": [{"height": 4}],
 "rooms": [{"name": "hall", "purpose": "entrance hall", "storey": 0, "rect": [8, 12, 16, 23],
            "desc": "bench, coat hooks, lantern"},
           {"name": "kitchen", "purpose": "kitchen", "storey": 0, "rect": [16, 12, 24, 23],
            "desc": "furnace, crafting table, barrel, table and chairs"}],
 "openings": [{"kind": "door", "storey": 0, "at": [12, 23], "between": ["hall", "outside"]},
              {"kind": "door", "storey": 0, "at": [16, 17], "between": ["hall", "kitchen"]},
              {"kind": "window", "storey": 0, "at": [20, 23], "width": 2,
               "between": ["kitchen", "outside"]}],
 "stairs": [], "roof": {"kind": "gable", "axis": "x", "overhang": 1},
 "materials": {"wall": "spruce_planks", "trim": "oak_log", "roof": "dark_oak_stairs"},
 "style": "a cosy timber cottage with a dark roof"}"""


class BlueprintError(ValueError):
    """The text held no usable blueprint (no JSON object, or no footprint and no rooms)."""


# ===========================================================================
# Geometry
# ===========================================================================
def _num(v: Any) -> int:
    return int(round(float(v)))


@dataclass(frozen=True)
class Rect:
    """Inclusive footprint rectangle in ``(x, z)``, including its walls."""

    x0: int
    z0: int
    x1: int
    z1: int

    @classmethod
    def of(cls, x0: int, z0: int, x1: int, z1: int) -> "Rect":
        """Sorts the corners, so any two opposite corners give the same rect."""
        xs, zs = sorted((int(x0), int(x1))), sorted((int(z0), int(z1)))
        return cls(xs[0], zs[0], xs[1], zs[1])

    @classmethod
    def parse(cls, v: Any) -> "Rect":
        """``[x0,z0,x1,z1]`` | ``{"x0","z0","x1","z1"}`` | ``{"x","z","w","d"}``;
        numeric strings are accepted. ValueError otherwise."""
        if isinstance(v, Rect):
            return v
        try:
            if isinstance(v, Mapping):
                if all(k in v for k in ("x0", "z0", "x1", "z1")):
                    return cls.of(_num(v["x0"]), _num(v["z0"]), _num(v["x1"]), _num(v["z1"]))
                if all(k in v for k in ("x", "z", "w", "d")):
                    x, z, w, d = (_num(v[k]) for k in ("x", "z", "w", "d"))
                    if w < 1 or d < 1:
                        raise ValueError(f"rect w/d must be >= 1, got {dict(v)!r}")
                    return cls.of(x, z, x + w - 1, z + d - 1)
                raise ValueError(f"rect dict needs x0 z0 x1 z1 or x z w d, got {dict(v)!r}")
            if isinstance(v, str):
                v = [t for t in re.split(r"[\s,;\[\]()]+", v) if t]
            if isinstance(v, (list, tuple)) and len(v) == 4:
                return cls.of(*(_num(t) for t in v))
        except (TypeError, OverflowError) as exc:
            raise ValueError(f"cannot read a rect from {v!r}") from exc
        raise ValueError(f"cannot read a rect from {v!r}")

    @property
    def w(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def d(self) -> int:
        return self.z1 - self.z0 + 1

    @property
    def area(self) -> int:
        return self.w * self.d

    def interior(self) -> Optional["Rect"]:
        if self.x1 - self.x0 < 2 or self.z1 - self.z0 < 2:
            return None
        return Rect(self.x0 + 1, self.z0 + 1, self.x1 - 1, self.z1 - 1)

    def contains(self, x: int, z: int) -> bool:
        return self.x0 <= x <= self.x1 and self.z0 <= z <= self.z1

    def intersect(self, o: "Rect") -> Optional["Rect"]:
        x0, z0, x1, z1 = max(self.x0, o.x0), max(self.z0, o.z0), min(self.x1, o.x1), min(self.z1, o.z1)
        if x0 > x1 or z0 > z1:
            return None
        return Rect(x0, z0, x1, z1)

    def on_border(self, x: int, z: int) -> bool:
        return self.contains(x, z) and (x in (self.x0, self.x1) or z in (self.z0, self.z1))

    def is_corner(self, x: int, z: int) -> bool:
        return x in (self.x0, self.x1) and z in (self.z0, self.z1)

    def border_cells(self) -> List[Coord2]:
        """Every border cell once, sorted."""
        cells = {(x, z) for x in range(self.x0, self.x1 + 1) for z in (self.z0, self.z1)}
        cells |= {(x, z) for z in range(self.z0, self.z1 + 1) for x in (self.x0, self.x1)}
        return sorted(cells)

    def shared_wall(self, o: "Rect") -> List[Coord2]:
        """Cells on BOTH borders minus the segment's two end cells (corners are
        structural, so no door goes there)."""
        both = sorted(set(self.border_cells()) & set(o.border_cells()))
        if not both:
            return []
        xs = {c[0] for c in both}
        zs = {c[1] for c in both}
        if len(xs) == 1:          # a wall along z
            line = sorted(both, key=lambda c: c[1])
        elif len(zs) == 1:        # a wall along x
            line = sorted(both, key=lambda c: c[0])
        else:                     # an overlap, not a shared wall line
            return []
        return line[1:-1]

    def to_list(self) -> List[int]:
        return [self.x0, self.z0, self.x1, self.z1]


@dataclass
class Storey:
    index: int
    floor_y: int
    height: int = 4
    footprint: Optional[Rect] = None                    # None = blueprint footprint

    @property
    def ceiling_y(self) -> int:
        return self.floor_y + self.height + 1


@dataclass
class Room:
    name: str
    purpose: str
    storey: int
    rect: Rect
    desc: str = ""                                      # furnishing brief (room dream + turn)

    @property
    def interior(self) -> Optional[Rect]:
        return self.rect.interior()


@dataclass
class Opening:
    kind: str                                           # door | window | arch
    storey: int
    x: int
    z: int                                              # a wall cell
    width: int = 1                                      # along the wall
    height: int = 2                                     # door 2; window 1..2
    sill: int = 0                                       # y offset above floor_y+1 (window default 1)
    between: Tuple[str, str] = ("", OUTSIDE)            # room names or OUTSIDE


@dataclass
class Stair:
    from_storey: int
    x: int
    z: int                                              # first (lowest) step, inside a room
    dir: str = "+x"                                     # direction of ascent
    #: 'stairs' or 'ladder' (repair's fallback when no straight run fits).
    kind: str = "stairs"

    def cells(self, height: int) -> List[Tuple[int, int, int]]:
        """``(x, y_offset, z)`` per step: ``height+1`` steps with y offsets
        ``1..height+1`` above floor_y (the last step sits in the upper slab). A
        ladder is one column."""
        dx, dz, _ = _DIRS.get(self.dir, _DIRS["+x"])
        if self.kind == "ladder":
            dx = dz = 0
        return [(self.x + i * dx, i + 1, self.z + i * dz) for i in range(height + 1)]


@dataclass
class Roof:
    kind: str = "gable"
    axis: str = "x"                                     # ridge axis for gable
    overhang: int = 1
    block: str = ""                                     # "" = materials["roof"]

    def spans(self, footprint: Rect) -> Tuple[int, int]:
        ov = max(0, int(self.overhang))
        return footprint.w + 2 * ov, footprint.d + 2 * ov

    def height(self, footprint: Rect) -> int:
        """Layers ABOVE the roof base ``top_y`` (so the highest roof block is at
        ``top_y + height``), as the DSL in :meth:`Blueprint.roof_dsl` builds it."""
        if self.kind == "none":
            return 0
        if self.kind == "flat":
            return 1
        sx, sz = self.spans(footprint)
        if self.kind == "gable":
            return ((sz if self.axis == "x" else sx) - 1) // 2
        return (min(sx, sz) - 1) // 2                   # hip / pyramid: shrinking rings


# ===========================================================================
# Blueprint
# ===========================================================================
def _slug(s: Any) -> str:
    out = re.sub(r"[^a-z0-9_]+", "_", str(s or "").strip().lower()).strip("_")
    return out[:32] or "room"


def _pair(v: Any) -> Tuple[int, int]:
    if isinstance(v, Mapping):
        return _num(v["x"]), _num(v["z"])
    if isinstance(v, str):
        v = [t for t in re.split(r"[\s,;\[\]()]+", v) if t]
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        if len(v) == 3:                                  # [x, y, z] from a careless model
            return _num(v[0]), _num(v[2])
        return _num(v[0]), _num(v[1])
    raise ValueError(f"cannot read an (x, z) position from {v!r}")


@dataclass
class Blueprint:
    footprint: Rect
    storeys: List[Storey]
    rooms: List[Room] = field(default_factory=list)
    openings: List[Opening] = field(default_factory=list)
    stairs: List[Stair] = field(default_factory=list)
    roof: Roof = field(default_factory=Roof)
    materials: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MATERIALS))
    style: str = ""                                     # one line: exterior look (dream desc)
    notes: List[str] = field(default_factory=list)      # parse + repair log

    # --- (de)serialization ---------------------------------------------------
    @classmethod
    def from_dict(cls, d: Mapping[str, Any], *, strict: bool = False) -> "Blueprint":
        """Lenient by default: a malformed entry is dropped with a note, a missing
        footprint becomes the bbox of the rooms, missing storeys one storey of
        height 4. ``strict=True`` raises BlueprintError on the first malformed entry.
        No footprint and no rooms always raises (there is nothing to build)."""
        if not isinstance(d, Mapping):
            raise BlueprintError(f"a blueprint must be a JSON object, got {type(d).__name__}")
        if isinstance(d.get("blueprint"), Mapping) and "footprint" not in d and "rooms" not in d:
            d = d["blueprint"]
        notes: List[str] = []

        def bad(what: str, exc: Exception) -> None:
            if strict:
                raise BlueprintError(f"{what}: {exc}") from exc
            notes.append(f"dropped {what}: {exc}")

        def ints(e: Mapping[str, Any], key: str, default: int) -> int:
            v = e.get(key, default)
            return default if v is None else _num(v)

        rooms: List[Room] = []
        for i, e in enumerate(_as_list(d.get("rooms"))):
            try:
                if not isinstance(e, Mapping):
                    raise ValueError("not an object")
                rect_v = e.get("rect", e.get("bounds", e.get("box")))
                if rect_v is None and all(k in e for k in ("x0", "z0", "x1", "z1")):
                    rect_v = {k: e[k] for k in ("x0", "z0", "x1", "z1")}
                if rect_v is None:
                    raise ValueError("no rect")
                name = str(e.get("name") or e.get("purpose") or f"room{i}")
                rooms.append(Room(name=name, purpose=str(e.get("purpose") or ""),
                                  storey=ints(e, "storey", 0), rect=Rect.parse(rect_v),
                                  desc=str(e.get("desc") or e.get("description") or "")))
            except (ValueError, TypeError, KeyError) as exc:
                bad(f"room #{i}", exc)

        fp_v = d.get("footprint")
        footprint: Optional[Rect] = None
        if fp_v is not None:
            try:
                footprint = Rect.parse(fp_v)
            except ValueError as exc:
                bad("footprint", exc)
        if footprint is None:
            if not rooms:
                raise BlueprintError("blueprint has no footprint and no rooms")
            footprint = Rect(min(r.rect.x0 for r in rooms), min(r.rect.z0 for r in rooms),
                             max(r.rect.x1 for r in rooms), max(r.rect.z1 for r in rooms))
            notes.append("footprint missing: used the bounding box of the rooms")

        storeys: List[Storey] = []
        raw_storeys = d.get("storeys")
        if isinstance(raw_storeys, (int, float)) and not isinstance(raw_storeys, bool):
            raw_storeys = [{} for _ in range(max(1, int(raw_storeys)))]
        for i, e in enumerate(_as_list(raw_storeys)):
            try:
                if isinstance(e, (int, float)) and not isinstance(e, bool):
                    e = {"height": e}
                if not isinstance(e, Mapping):
                    raise ValueError("not an object")
                fp = Rect.parse(e["footprint"]) if e.get("footprint") is not None else None
                storeys.append(Storey(index=ints(e, "index", i),
                                      floor_y=ints(e, "floor_y", -1),
                                      height=ints(e, "height", 4), footprint=fp))
            except (ValueError, TypeError, KeyError) as exc:
                bad(f"storey #{i}", exc)
        if not storeys:
            storeys = [Storey(0, 0, 4)]
            if raw_storeys is None:
                notes.append("storeys missing: one storey of height 4")
        # Unknown floor_y (-1): stack contiguously in the given order.
        y = 0
        for s in storeys:
            if s.floor_y < 0:
                s.floor_y = y
            y = s.ceiling_y

        openings: List[Opening] = []
        for i, e in enumerate(_as_list(d.get("openings", d.get("doors")))):
            try:
                if not isinstance(e, Mapping):
                    raise ValueError("not an object")
                kind = str(e.get("kind") or e.get("type") or "door").strip().lower()
                if "at" in e or "pos" in e:
                    x, z = _pair(e.get("at", e.get("pos")))
                else:
                    x, z = _num(e["x"]), _num(e["z"])
                btw = e.get("between")
                if isinstance(btw, str):
                    btw = [t for t in re.split(r"[\s,;/|]+|<->|-", btw) if t]
                pair = tuple(str(t) for t in (btw or [])[:2])
                pair = (pair + ("", OUTSIDE))[:2] if len(pair) < 2 else pair
                default_h = 1 if kind == "window" else 2
                default_sill = 1 if kind == "window" else 0
                openings.append(Opening(kind=kind, storey=ints(e, "storey", 0), x=x, z=z,
                                        width=ints(e, "width", 1),
                                        height=ints(e, "height", default_h),
                                        sill=ints(e, "sill", default_sill),
                                        between=(pair[0], pair[1])))  # type: ignore[arg-type]
            except (ValueError, TypeError, KeyError) as exc:
                bad(f"opening #{i}", exc)

        stairs: List[Stair] = []
        for i, e in enumerate(_as_list(d.get("stairs"))):
            try:
                if not isinstance(e, Mapping):
                    raise ValueError("not an object")
                if "at" in e or "pos" in e:
                    x, z = _pair(e.get("at", e.get("pos")))
                else:
                    x, z = _num(e["x"]), _num(e["z"])
                stairs.append(Stair(from_storey=ints(e, "from_storey", ints(e, "from", 0)),
                                    x=x, z=z, dir=str(e.get("dir") or "+x").strip().lower(),
                                    kind=str(e.get("kind") or "stairs").strip().lower()))
            except (ValueError, TypeError, KeyError) as exc:
                bad(f"stair #{i}", exc)

        roof = Roof()
        rv = d.get("roof")
        if isinstance(rv, str):
            roof = Roof(kind=rv.strip().lower())
        elif isinstance(rv, Mapping):
            try:
                roof = Roof(kind=str(rv.get("kind") or rv.get("type") or "gable").strip().lower(),
                            axis=str(rv.get("axis") or "x").strip().lower(),
                            overhang=ints(rv, "overhang", 1), block=str(rv.get("block") or ""))
            except (ValueError, TypeError) as exc:
                bad("roof", exc)

        materials = dict(DEFAULT_MATERIALS)
        mv = d.get("materials")
        if isinstance(mv, Mapping):
            materials.update({str(k): str(v) for k, v in mv.items() if v})
        elif mv is not None:
            bad("materials", ValueError("not an object"))

        return cls(footprint=footprint, storeys=storeys, rooms=rooms, openings=openings,
                   stairs=stairs, roof=roof, materials=materials,
                   style=str(d.get("style") or ""),
                   notes=[str(n) for n in _as_list(d.get("notes"))] + notes)

    @classmethod
    def from_json(cls, text: str, *, strict: bool = False) -> "Blueprint":
        """Tolerant extraction (code fences, prose around the object), then from_dict."""
        from blockgen.agentic.episode import extract_json

        obj = extract_json(text or "")
        if not obj:
            raise BlueprintError("no JSON object in the reply")
        return cls.from_dict(obj, strict=strict)

    def to_dict(self, *, notes: bool = False) -> Dict[str, Any]:
        storeys = []
        for s in self.storeys:
            e: Dict[str, Any] = {"index": s.index, "floor_y": s.floor_y, "height": s.height}
            if s.footprint is not None:
                e["footprint"] = s.footprint.to_list()
            storeys.append(e)
        out: Dict[str, Any] = {
            "footprint": self.footprint.to_list(),
            "storeys": storeys,
            "rooms": [{"name": r.name, "purpose": r.purpose, "storey": r.storey,
                       "rect": r.rect.to_list(), "desc": r.desc} for r in self.rooms],
            "openings": [{"kind": o.kind, "storey": o.storey, "at": [o.x, o.z],
                          "width": o.width, "height": o.height, "sill": o.sill,
                          "between": list(o.between)} for o in self.openings],
            "stairs": [{"from_storey": s.from_storey, "at": [s.x, s.z], "dir": s.dir,
                        **({"kind": s.kind} if s.kind != "stairs" else {})}
                       for s in self.stairs],
            "roof": {"kind": self.roof.kind, "axis": self.roof.axis,
                     "overhang": self.roof.overhang,
                     **({"block": self.roof.block} if self.roof.block else {})},
            "materials": dict(self.materials),
            "style": self.style,
        }
        if notes:
            out["notes"] = list(self.notes)
        return out

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def copy(self) -> "Blueprint":
        return copy.deepcopy(self)

    # --- queries -------------------------------------------------------------
    def storey(self, i: int) -> Storey:
        for s in self.storeys:
            if s.index == i:
                return s
        raise KeyError(f"no storey {i}; storeys: {[s.index for s in self.storeys]}")

    def rooms_on(self, storey: int) -> List[Room]:
        return [r for r in self.rooms if r.storey == storey]

    def room(self, name: str) -> Room:
        for r in self.rooms:
            if r.name == name:
                return r
        raise KeyError(f"no room '{name}'; rooms: {[r.name for r in self.rooms]}")

    def footprint_of(self, storey: int) -> Rect:
        try:
            s = self.storey(storey)
        except KeyError:
            return self.footprint
        return s.footprint or self.footprint

    def top_y(self) -> int:
        """Ceiling of the top storey: the roof base."""
        return max((s.ceiling_y for s in self.storeys), default=0)

    def total_height(self) -> int:
        """y of the highest roof block (top_y + roof height)."""
        top = max(self.storeys, key=lambda s: s.index).index if self.storeys else 0
        return self.top_y() + self.roof.height(self.footprint_of(top))

    def adjacency(self) -> Dict[str, Set[str]]:
        """Room -> rooms it shares a wall with on its storey (+ OUTSIDE when a wall
        segment of it lies on the footprint border)."""
        out: Dict[str, Set[str]] = {r.name: set() for r in self.rooms}
        for i, a in enumerate(self.rooms):
            if _exterior_candidates(self, a, a.storey):
                out[a.name].add(OUTSIDE)
            for b in self.rooms[i + 1:]:
                if b.storey == a.storey and a.rect.shared_wall(b.rect):
                    out[a.name].add(b.name)
                    out[b.name].add(a.name)
        return out

    def _room_at(self, storey: int, x: int, z: int) -> Optional[Room]:
        """The room whose interior holds (x, z) on that storey (else whose rect does)."""
        rooms = self.rooms_on(storey)
        for r in rooms:
            it = r.interior
            if it is not None and it.contains(x, z):
                return r
        for r in rooms:
            if r.rect.contains(x, z):
                return r
        return None

    def _links(self) -> List[Tuple[str, str]]:
        """Graph edges: doors and arches between their rooms, stairs between the
        rooms (or, on a storey without rooms, the open floor ``_open<k>``) at their
        two ends."""
        names = {r.name for r in self.rooms}
        links: List[Tuple[str, str]] = []
        for o in self.openings:
            if o.kind in ("door", "arch"):
                a, b = (t if t or self.rooms_on(o.storey) else f"_open{o.storey}"
                        for t in o.between)
                ok = {OUTSIDE, f"_open{o.storey}"} | names
                if a in ok and b in ok and a != b:
                    links.append((a, b))
        for s in self.stairs:
            h = self._height_of(s.from_storey)
            cells = s.cells(h)
            ends = []
            for k, c in ((s.from_storey, cells[0]), (s.from_storey + 1, cells[-1])):
                r = self._room_at(k, c[0], c[2])
                ends.append(r.name if r is not None else
                            (f"_open{k}" if not self.rooms_on(k) else None))
            if ends[0] is not None and ends[1] is not None:
                links.append((ends[0], ends[1]))
        return links

    def _height_of(self, storey: int) -> int:
        try:
            return self.storey(storey).height
        except KeyError:
            return 4

    def reachable(self) -> Set[str]:
        """Rooms reachable from OUTSIDE over doors, arches and stairs (OUTSIDE excluded)."""
        graph: Dict[str, Set[str]] = {}
        for a, b in self._links():
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)
        seen = {OUTSIDE}
        todo = deque([OUTSIDE])
        while todo:
            for nb in graph.get(todo.popleft(), ()):
                if nb not in seen:
                    seen.add(nb)
                    todo.append(nb)
        return {n for n in seen if n != OUTSIDE and not n.startswith("_open")}

    def wall_axis(self, x: int, z: int, storey: int) -> Optional[str]:
        """'x' if (x, z) lies on a wall running along x, 'z' along z, None if it is
        not a wall cell or is a corner/junction of both."""
        rects = [self.footprint_of(storey)] + [r.rect for r in self.rooms_on(storey)]
        along_x = any(r.contains(x, z) and z in (r.z0, r.z1) and r.x0 < x < r.x1 for r in rects)
        along_z = any(r.contains(x, z) and x in (r.x0, r.x1) and r.z0 < z < r.z1 for r in rects)
        corner = any(r.is_corner(x, z) for r in rects)
        if along_x and not along_z and not corner:
            return "x"
        if along_z and not along_x and not corner:
            return "z"
        return None

    def summary(self) -> str:
        """At most 14 plain lines, for prompts and traces."""
        fp = self.footprint
        lines = [f"footprint x {fp.x0}..{fp.x1}, z {fp.z0}..{fp.z1} ({fp.w} x {fp.d}); "
                 f"{len(self.storeys)} storey(s); roof {self.roof.kind}"
                 + (f" (ridge along {self.roof.axis})" if self.roof.kind == "gable" else "")]
        for s in self.storeys:
            lines.append(f"storey {s.index}: floor y={s.floor_y}, height {s.height}, "
                         f"ceiling y={s.ceiling_y}")
        for r in self.rooms:
            if len(lines) >= 12:
                lines.append(f"... {len(self.rooms)} rooms in all")
                break
            lines.append(f"room {r.name} ({r.purpose or r.name}), storey {r.storey}: "
                         f"x {r.rect.x0}..{r.rect.x1}, z {r.rect.z0}..{r.rect.z1}")
        n_doors = sum(o.kind == "door" for o in self.openings)
        n_win = sum(o.kind == "window" for o in self.openings)
        lines.append(f"{n_doors} door(s), {n_win} window(s), {len(self.stairs)} stair(s); "
                     f"walls {self.materials.get('wall')}, roof {self.roof.block or self.materials.get('roof')}")
        return "\n".join(lines[:14])

    # --- validation / repair -------------------------------------------------
    def validate(self, size: Size3) -> List[str]:
        """The problems :meth:`repaired` would fix (``[]`` = valid). No mutation."""
        return self.repaired(size).notes[len(self.notes):]

    def repaired(self, size: Size3, *, max_rooms: Optional[int] = None,
                 max_storeys: Optional[int] = None) -> "Blueprint":
        """A NEW, buildable blueprint. Never raises; idempotent
        (``bp.repaired(s).repaired(s) == bp.repaired(s)``); every change is
        appended to ``.notes`` in plain words."""
        return _repair(self.copy(), tuple(int(v) for v in size), max_rooms, max_storeys)  # type: ignore[arg-type]

    # --- geometry for methods -------------------------------------------------
    def room_box(self, name: str, *, walls: bool = False, floor: bool = True) -> Box:
        """walls=False: the interior x/z; y from floor_y (floor=True) or floor_y+1
        up to floor_y+height. walls=True: the rect including its walls."""
        r = self.room(name)
        s = self.storey(r.storey)
        rect = r.rect if walls else (r.interior or r.rect)
        return Box(rect.x0, s.floor_y if floor else s.floor_y + 1, rect.z0,
                   rect.x1, s.floor_y + s.height, rect.z1)

    def storey_box(self, i: int, *, floor: bool = True) -> Box:
        s = self.storey(i)
        fp = self.footprint_of(i)
        return Box(fp.x0, s.floor_y if floor else s.floor_y + 1, fp.z0,
                   fp.x1, s.floor_y + s.height, fp.z1)

    def _inward(self, o: Opening, room: Room, x: int, z: int) -> Optional[Coord2]:
        it = room.interior
        if it is None:
            return None
        for dx, dz in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            if it.contains(x + dx, z + dz):
                return (x + dx, z + dz)
        return None

    def keep_clear(self, name: str) -> List[Box]:
        """Boxes inside room ``name`` that must stay as the shell left them: the cell
        in front of each of its doors and arches (2 high), its stair runs plus
        headroom and landing, and for a stair arriving from below its top steps, the
        stairwell cut into this room's floor slab and the headroom above them. Used
        as Region.deny for rooms."""
        r = self.room(name)
        s = self.storey(r.storey)
        out: List[Box] = []
        for o in self.openings:
            if o.storey != r.storey or o.kind not in ("door", "arch") or name not in o.between:
                continue
            for (x, z) in _opening_cells(self, o):
                cell = self._inward(o, r, x, z)
                if cell is not None:
                    out.append(Box(cell[0], s.floor_y + 1, cell[1],
                                   cell[0], s.floor_y + 2, cell[1]))
        it = r.interior
        for st in self.stairs:
            if st.from_storey == r.storey:
                cells = st.cells(s.height)
                if self._room_at(r.storey, cells[0][0], cells[0][2]) is not r:
                    continue
                xs = [c[0] for c in cells]
                zs = [c[2] for c in cells]
                out.append(Box(min(xs), s.floor_y + 1, min(zs), max(xs), s.floor_y + s.height,
                               max(zs)))
                dx, dz, _ = _DIRS.get(st.dir, _DIRS["+x"])
                land = (cells[0][0] - dx, cells[0][2] - dz)
                if st.kind != "ladder" and it is not None and it.contains(*land):
                    out.append(Box(land[0], s.floor_y + 1, land[1], land[0], s.floor_y + 2,
                                   land[1]))
            elif st.from_storey == r.storey - 1:
                try:
                    below = self.storey(st.from_storey)
                except KeyError:
                    continue
                cells = st.cells(below.height)
                top = cells[-1]
                if self._room_at(r.storey, top[0], top[2]) is not r:
                    continue
                last = cells[-3:]
                xs = [c[0] for c in last]
                zs = [c[2] for c in last]
                dx, dz, _ = _DIRS.get(st.dir, _DIRS["+x"])
                xs.append(top[0] + dx)
                zs.append(top[2] + dz)
                # From floor_y: the top step and the two cleared stairwell cells sit
                # IN this room's floor slab, and a floor or carpet fill over the room
                # would otherwise close the stairwell.
                box = Box.of(min(xs), s.floor_y, min(zs), max(xs), s.floor_y + 2, max(zs))
                if it is not None:
                    box = Box.of(max(box.x0, it.x0), box.y0, max(box.z0, it.z0),
                                 min(box.x1, it.x1), box.y1, min(box.z1, it.z1))
                out.append(box)
        return out

    def door_cells(self, storey: Optional[int] = None) -> List[Tuple[int, int, int]]:
        """(x, y, z) of every door block (both halves), optionally on one storey."""
        out: List[Tuple[int, int, int]] = []
        for o in self.openings:
            if o.kind != "door" or (storey is not None and o.storey != storey):
                continue
            try:
                fy = self.storey(o.storey).floor_y
            except KeyError:
                continue
            for (x, z) in _opening_cells(self, o):
                for dy in range(o.height):
                    out.append((x, fy + 1 + dy, z))
        return out

    # --- deterministic DSL ---------------------------------------------------
    def shell_dsl(self, *, parts: Sequence[str] = SHELL_PARTS,
                  storeys: Optional[Sequence[int]] = None) -> str:
        """Canonical DSL for the shell (no furniture, no roof). Every block of lines
        starts with ``# shell: <part> storey <i>``. Runs via ``try_(..., ir=True)``
        whatever the recipe's action space."""
        return _shell_dsl(self, parts, storeys)

    def roof_dsl(self, size: Optional[Size3] = None) -> str:
        """Canonical DSL for the roof (empty for kind 'none'). Coordinates are clipped
        to the canvas in Python (``size``; default: the smallest canvas holding the
        footprint plus overhang)."""
        return _roof_dsl(self, size)


def _as_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


# ===========================================================================
# Openings
# ===========================================================================
def _wall_dir(bp: Blueprint, o: Opening) -> Tuple[int, int]:
    """Unit step along the wall the opening sits in."""
    axis = bp.wall_axis(o.x, o.z, o.storey)
    if axis == "z":
        return (0, 1)
    return (1, 0)


def _opening_cells(bp: Blueprint, o: Opening) -> List[Coord2]:
    dx, dz = _wall_dir(bp, o)
    return [(o.x + i * dx, o.z + i * dz) for i in range(max(1, o.width))]


def _exterior_candidates(bp: Blueprint, room: Optional[Room], storey: int) -> List[Coord2]:
    """Wall cells of ``room`` on the storey's footprint border, corners excluded."""
    fp = bp.footprint_of(storey)
    rect = room.rect if room is not None else fp
    out = []
    for (x, z) in rect.border_cells():
        if rect.is_corner(x, z) or fp.is_corner(x, z) or not fp.on_border(x, z):
            continue
        out.append((x, z))
    return out


def _candidates(bp: Blueprint, o: Opening, rooms: Dict[str, Room]) -> List[Coord2]:
    a, b = o.between
    if b == OUTSIDE or a == OUTSIDE:
        name = a if b == OUTSIDE else b
        return _exterior_candidates(bp, rooms.get(name), o.storey)
    if a in rooms and b in rooms:
        return rooms[a].rect.shared_wall(rooms[b].rect)
    return []


# ===========================================================================
# Repair
# ===========================================================================
def _clip_rect(r: Rect, lo_x: int, lo_z: int, hi_x: int, hi_z: int) -> Optional[Rect]:
    x0, z0, x1, z1 = max(r.x0, lo_x), max(r.z0, lo_z), min(r.x1, hi_x), min(r.z1, hi_z)
    if x0 > x1 or z0 > z1:
        return None
    return Rect(x0, z0, x1, z1)


def _rects_overlap(a: Rect, b: Rect) -> bool:
    """True when two room rects overlap by more than a shared wall line (their
    intersection is at least 2 x 2). Implies, but is stricter than, intersecting
    interiors."""
    i = a.intersect(b)
    return i is not None and i.w >= 2 and i.d >= 2


def _merge_double_walls(bp: Blueprint, b: Room, rooms: Sequence[Room], note: Any) -> bool:
    """Grow ``b`` by one onto the wall of another room it stands right next to
    (``b.x0 == a.x1 + 1`` with at least 3 cells of the walls side by side), and
    onto the storey's outer wall when it stops one short of it.

    Why: a plan written with exclusive bounds ('hall 6..15, kitchen 16..25') gives
    two walls side by side. No cell is then on both rooms' borders, so no door can
    join them and repair would give every such room its own outside door. The
    shared-wall convention is one wall, so the room grown first takes the line.
    Returns whether ``b`` changed.
    """
    fp = bp.footprint_of(b.storey)
    r = b.rect
    others = [a.rect for a in rooms if a.storey == b.storey and a is not b]
    x0, z0, x1, z1 = r.x0, r.z0, r.x1, r.z1

    def grow(cand: Tuple[int, int, int, int]) -> None:
        # One edge at a time, and only when it overlaps nothing: a step the
        # overlap cut would undo again would make repair non-idempotent.
        nonlocal x0, z0, x1, z1
        c = Rect(*cand)
        if not any(_rects_overlap(o, c) for o in others):
            x0, z0, x1, z1 = cand

    if x0 == fp.x0 + 1:
        grow((fp.x0, z0, x1, z1))
    if x1 == fp.x1 - 1:
        grow((x0, z0, fp.x1, z1))
    if z0 == fp.z0 + 1:
        grow((x0, fp.z0, x1, z1))
    if z1 == fp.z1 - 1:
        grow((x0, z0, x1, fp.z1))
    for ar in others:
        z_run = min(ar.z1, z1) - max(ar.z0, z0) + 1
        x_run = min(ar.x1, x1) - max(ar.x0, x0) + 1
        if z_run >= 3 and x0 == ar.x1 + 1:
            grow((ar.x1, z0, x1, z1))
        elif z_run >= 3 and x1 == ar.x0 - 1:
            grow((x0, z0, ar.x0, z1))
        elif x_run >= 3 and z0 == ar.z1 + 1:
            grow((x0, ar.z1, x1, z1))
        elif x_run >= 3 and z1 == ar.z0 - 1:
            grow((x0, z0, x1, ar.z0))
    new = Rect(x0, z0, x1, z1)
    if new != r:
        note(f"room '{b.name}' widened to x {x0}..{x1}, z {z0}..{z1}: it stood one block "
             "off a neighbouring wall (a double wall), so it now shares that wall")
        b.rect = new
        return True
    return False


def _grow_to(r: Rect, lo: int, hi: int, axis: str, n: int) -> Rect:
    """Widen ``r`` along ``axis`` to at least ``n`` cells, staying in [lo, hi]."""
    a0, a1 = (r.x0, r.x1) if axis == "x" else (r.z0, r.z1)
    while a1 - a0 + 1 < n and (a0 > lo or a1 < hi):
        if a1 < hi:
            a1 += 1
        if a1 - a0 + 1 < n and a0 > lo:
            a0 -= 1
    return Rect(a0, r.z0, a1, r.z1) if axis == "x" else Rect(r.x0, a0, r.x1, a1)


def _repair(bp: Blueprint, size: Size3, max_rooms: Optional[int],
            max_storeys: Optional[int]) -> Blueprint:
    from blockgen.agentic.blockstate import try_resolve_block

    W, H, D = size
    note = bp.notes.append

    # 1. materials ------------------------------------------------------------
    mats: Dict[str, str] = {}
    for role, block in bp.materials.items():
        block = str(block).strip()
        if try_resolve_block(block) is None:
            if role in DEFAULT_MATERIALS:
                note(f"material {role} '{block}' is not a known block: used "
                     f"'{DEFAULT_MATERIALS[role]}'")
                mats[role] = DEFAULT_MATERIALS[role]
            else:
                note(f"material {role} '{block}' is not a known block: dropped")
            continue
        mats[role] = block
    for role, block in DEFAULT_MATERIALS.items():
        if role not in mats:
            mats[role] = block
            note(f"material {role} missing: used '{block}'")
    bp.materials = mats

    # 2. roof -----------------------------------------------------------------
    roof = bp.roof
    if roof.kind not in ROOF_KINDS:
        note(f"roof kind '{roof.kind}' unknown: used gable")
        roof.kind = "gable"
    if roof.axis not in ("x", "z"):
        axis = "x" if bp.footprint.w >= bp.footprint.d else "z"
        note(f"roof axis '{roof.axis}' unknown: used {axis}")
        roof.axis = axis
    ov = min(2, max(0, int(roof.overhang)))
    if ov != roof.overhang:
        note(f"roof overhang {roof.overhang} clamped to {ov}")
        roof.overhang = ov
    if roof.block and try_resolve_block(roof.block) is None:
        note(f"roof block '{roof.block}' is not a known block: used materials['roof']")
        roof.block = ""

    # 3. footprint ------------------------------------------------------------
    lo_x, hi_x, lo_z, hi_z = ov, W - 1 - ov, ov, D - 1 - ov
    if hi_x - lo_x + 1 < MIN_ROOM_SIDE or hi_z - lo_z + 1 < MIN_ROOM_SIDE:
        ov = roof.overhang = 0
        lo_x, hi_x, lo_z, hi_z = 0, W - 1, 0, D - 1
        note("canvas too small for a roof overhang: overhang 0")
    fp = _clip_rect(bp.footprint, lo_x, lo_z, hi_x, hi_z)
    if fp is None:
        w, d = max(MIN_ROOM_SIDE, int(W * 0.6)), max(MIN_ROOM_SIDE, int(D * 0.6))
        x0, z0 = (W - w) // 2, (D - d) // 2
        fp = _clip_rect(Rect(x0, z0, x0 + w - 1, z0 + d - 1), lo_x, lo_z, hi_x, hi_z)
        assert fp is not None
        note(f"footprint outside the canvas: used a centred {fp.w} x {fp.d} footprint")
    elif fp != bp.footprint:
        note(f"footprint clipped to x {fp.x0}..{fp.x1}, z {fp.z0}..{fp.z1} (canvas and roof overhang)")
    if fp.w < MIN_ROOM_SIDE or fp.d < MIN_ROOM_SIDE:
        fp2 = _grow_to(_grow_to(fp, lo_x, hi_x, "x", MIN_ROOM_SIDE), lo_z, hi_z, "z", MIN_ROOM_SIDE)
        note(f"footprint widened to {fp2.w} x {fp2.d} (minimum side {MIN_ROOM_SIDE})")
        fp = fp2
    bp.footprint = fp

    # 4. storeys --------------------------------------------------------------
    storeys = sorted(bp.storeys, key=lambda s: (s.floor_y, s.index))
    old_index = [s.index for s in storeys]
    y = 0
    for i, s in enumerate(storeys):
        h = min(MAX_STOREY_HEIGHT, max(MIN_STOREY_HEIGHT, int(s.height)))
        if h != s.height:
            note(f"storey {i} height {s.height} clamped to {h}")
        if s.index != i:
            note(f"storey {s.index} renumbered {i}")
        if s.floor_y != y:
            note(f"storey {i} floor moved to y={y} (storeys stack contiguously)")
        s.index, s.height, s.floor_y = i, h, y
        if s.footprint is not None:
            sfp = s.footprint.intersect(fp)
            if sfp is None or sfp.w < MIN_ROOM_SIDE or sfp.d < MIN_ROOM_SIDE:
                note(f"storey {i} footprint outside the building: uses the main footprint")
                sfp = None
            elif sfp != s.footprint:
                note(f"storey {i} footprint clipped to the main footprint")
            s.footprint = None if sfp == fp else sfp
        y = s.ceiling_y
    remap = {o: n for n, o in enumerate(old_index)}
    bp.storeys = storeys
    if max_storeys is not None and len(bp.storeys) > max(1, int(max_storeys)):
        note(f"kept the lowest {max(1, int(max_storeys))} of {len(bp.storeys)} storeys")
        bp.storeys = bp.storeys[:max(1, int(max_storeys))]
    while len(bp.storeys) > 1 and bp.total_height() > H - 1:
        note(f"storey {bp.storeys[-1].index} dropped: the building would be taller than the canvas")
        bp.storeys.pop()
    while bp.total_height() > H - 1 and bp.storeys[0].height > MIN_STOREY_HEIGHT:
        bp.storeys[0].height -= 1
        note(f"storey 0 lowered to height {bp.storeys[0].height} to fit the canvas")
    if bp.total_height() > H - 1 and roof.kind not in ("flat", "none"):
        note(f"roof {roof.kind} does not fit under the canvas top: flat roof")
        roof.kind = "flat"
    n_st = len(bp.storeys)

    def map_storey(i: int) -> Optional[int]:
        j = remap.get(i, None)
        return j if j is not None and j < n_st else None

    # 5. rooms ----------------------------------------------------------------
    rooms: List[Room] = []
    used: Set[str] = set()
    for r in bp.rooms:
        st = map_storey(r.storey)
        if st is None:
            note(f"room '{r.name}' dropped: storey {r.storey} does not exist")
            continue
        if st != r.storey:
            r.storey = st
        sfp = bp.footprint_of(st)
        rect = r.rect.intersect(sfp)
        if rect is None or rect.w < MIN_ROOM_SIDE or rect.d < MIN_ROOM_SIDE:
            note(f"room '{r.name}' dropped: smaller than {MIN_ROOM_SIDE} x {MIN_ROOM_SIDE} "
                 "inside the footprint")
            continue
        if rect != r.rect:
            note(f"room '{r.name}' clipped to x {rect.x0}..{rect.x1}, z {rect.z0}..{rect.z1} "
                 "(inside the footprint)")
            r.rect = rect
        slug = _slug(r.name)
        name, k = slug, 2
        while name in used:
            name = f"{slug}_{k}"
            k += 1
        if name != r.name:
            note(f"room '{r.name}' renamed '{name}'")
            if name == slug:
                # A plain re-spelling ('Great Hall' -> 'great_hall'): openings that
                # named the old spelling follow. A de-duplicated name ('kitchen_2')
                # keeps them on the first room of that name.
                for o in bp.openings:
                    o.between = tuple(name if t == r.name else t
                                      for t in o.between)  # type: ignore[assignment]
            r.name = name
        used.add(name)
        if not r.purpose.strip():
            r.purpose = name.replace("_", " ")
            note(f"room '{name}' had no purpose: used '{r.purpose}'")
        rooms.append(r)
    if max_rooms is not None and len(rooms) > max(0, int(max_rooms)):
        note(f"kept the first {max(0, int(max_rooms))} of {len(rooms)} rooms")
        rooms = rooms[:max(0, int(max_rooms))]

    # 6. overlaps -------------------------------------------------------------
    # Two rooms may share at most a wall LINE. Testing interiors alone is not enough:
    # hall [6..16] and kitchen [15..25] have disjoint interiors (7..15 vs 16..24) but
    # each one's wall runs through the other's interior (a two-thick double wall),
    # so no cell is on both borders and no door can join them.
    kept: List[Room] = []
    for b in rooms:
        alive = True
        for _ in range(16):
            hit = next((a for a in kept if a.storey == b.storey
                        and _rects_overlap(a.rect, b.rect)), None)
            if hit is None:
                break
            a = hit
            cuts = [Rect(a.rect.x1, b.rect.z0, b.rect.x1, b.rect.z1),
                    Rect(b.rect.x0, b.rect.z0, a.rect.x0, b.rect.z1),
                    Rect(b.rect.x0, a.rect.z1, b.rect.x1, b.rect.z1),
                    Rect(b.rect.x0, b.rect.z0, b.rect.x1, a.rect.z0)]
            valid = [c for c in cuts if c.x0 <= c.x1 and c.z0 <= c.z1 and c.w >= MIN_ROOM_SIDE
                     and c.d >= MIN_ROOM_SIDE and c.interior() is not None
                     and not _rects_overlap(a.rect, c)]
            if not valid:
                alive = False
                break
            best = max(valid, key=lambda c: c.area)   # first of equals: list order
            note(f"room '{b.name}' cut to x {best.x0}..{best.x1}, z {best.z0}..{best.z1} "
                 f"to share a wall with '{a.name}'")
            b.rect = best
        else:
            alive = False
        if not alive:
            note(f"room '{b.name}' dropped: it overlaps another room and no cut leaves a room")
            continue
        kept.append(b)
    bp.rooms = kept
    # Then close double walls. Growth only ever adds a wall line that overlaps no
    # room, so it cannot undo a cut, and running it to a fixed point here is what
    # keeps repair idempotent (a second pass finds nothing left to grow).
    for _ in range(4 * len(kept) + 4):
        if not any([_merge_double_walls(bp, b, kept, note) for b in kept]):
            break
    names = {r.name: r for r in bp.rooms}

    # 7. openings -------------------------------------------------------------
    openings: List[Opening] = []
    for o in bp.openings:
        if o.kind not in OPENING_KINDS:
            note(f"opening kind '{o.kind}' unknown: dropped")
            continue
        st = map_storey(o.storey)
        if st is None:
            note(f"{o.kind} at ({o.x}, {o.z}) dropped: storey {o.storey} does not exist")
            continue
        o.storey = st
        a, b = o.between
        if b != OUTSIDE and a == OUTSIDE:
            a, b = b, a
        if a not in names and a != "":
            a = ""
        if b not in names and b != OUTSIDE:
            b = OUTSIDE
        if a == "":
            # Infer the room from the position: the nearest room of this storey whose
            # border the opening touches, else the room containing it.
            here = [r for r in bp.rooms_on(st) if r.rect.on_border(o.x, o.z)
                    and r.name != b]
            near = here or [r for r in bp.rooms_on(st) if r.rect.contains(o.x, o.z)]
            if near:
                a = near[0].name
            elif bp.rooms_on(st):
                a = min(bp.rooms_on(st), key=lambda r: abs(r.rect.x0 + r.rect.x1 - 2 * o.x)
                        + abs(r.rect.z0 + r.rect.z1 - 2 * o.z)).name
        if o.kind == "window" and b != OUTSIDE:
            note(f"window at ({o.x}, {o.z}) moved to an exterior wall of '{a}'")
            b = OUTSIDE
        if o.kind != "window" and b == OUTSIDE and st > 0:
            # An exterior door above the ground floor opens onto a drop (typically
            # a door whose far room was dropped by max_rooms and so became
            # 'outside'). Keep the hole as a window instead of losing it.
            note(f"{o.kind} at ({o.x}, {o.z}) on storey {st} opens onto a drop: "
                 "became a window")
            o.kind, o.height, o.sill = "window", 1, 1
        if (a, b) != tuple(o.between):
            o.between = (a, b)
        cands = _candidates(bp, o, names)
        if not cands:
            note(f"{o.kind} at ({o.x}, {o.z}) between {a or '?'} and {b} dropped: "
                 "no wall cell between them")
            continue
        cell = min(cands, key=lambda c: (abs(c[0] - o.x) + abs(c[1] - o.z), c))
        if cell != (o.x, o.z):
            note(f"{o.kind} at ({o.x}, {o.z}) snapped to wall cell ({cell[0]}, {cell[1]})")
            o.x, o.z = cell
        cset = set(cands)
        width = max(1, int(o.width))
        while width > 1:
            dx, dz = _wall_dir(bp, o)
            if all((o.x + i * dx, o.z + i * dz) in cset for i in range(width)):
                break
            width -= 1
        if width != o.width:
            note(f"{o.kind} at ({o.x}, {o.z}) narrowed to width {width} (stays off corners)")
            o.width = width
        sh = bp.storey(st).height
        if o.kind == "door":
            if (o.height, o.sill) != (2, 0):
                note(f"door at ({o.x}, {o.z}) set to height 2 on the floor")
            o.height, o.sill = 2, 0
        elif o.kind == "arch":
            h = min(max(2, int(o.height)), sh)
            if (h, o.sill) != (o.height, 0):
                note(f"arch at ({o.x}, {o.z}) set to height {h} on the floor")
            o.height, o.sill = h, 0
        else:
            h = min(2, max(1, int(o.height)))
            sill = min(max(0, int(o.sill)), sh - h)
            if (h, sill) != (o.height, o.sill):
                note(f"window at ({o.x}, {o.z}) set to height {h}, sill {sill}")
            o.height, o.sill = h, sill
        openings.append(o)
    # Windows never on a door cell; duplicates dropped.
    bp.openings = []
    taken: Set[Tuple[int, int, int]] = set()
    for o in sorted(openings, key=lambda o: o.kind == "window"):
        cells = {(o.storey, x, z) for (x, z) in _opening_cells(bp, o)}
        if cells & taken:
            note(f"{o.kind} at ({o.x}, {o.z}) dropped: overlaps another opening")
            continue
        taken |= cells
        bp.openings.append(o)
    bp.openings.sort(key=lambda o: openings.index(o))

    # 8. stairs ---------------------------------------------------------------
    stairs: List[Stair] = []
    for s in bp.stairs:
        st = map_storey(s.from_storey)
        if st is None or st + 1 >= n_st:
            note(f"stair at ({s.x}, {s.z}) dropped: no storey above storey {s.from_storey}")
            continue
        s.from_storey = st
        if s.dir not in STAIR_DIRS:
            note(f"stair at ({s.x}, {s.z}) direction '{s.dir}' unknown: used +x")
            s.dir = "+x"
        if s.kind not in ("stairs", "ladder"):
            s.kind = "stairs"
        if any(t.from_storey == st for t in stairs):
            note(f"second stair from storey {st} dropped")
            continue
        if not (_ladder_ok(bp, s) if s.kind == "ladder" else _stair_ok(bp, s)):
            note(f"stair at ({s.x}, {s.z}) dropped: its run leaves its room or the floor above")
            continue
        stairs.append(s)
    bp.stairs = stairs
    for k in range(n_st - 1):
        if any(s.from_storey == k for s in bp.stairs):
            continue
        s = _place_stair(bp, k)
        if s.kind == "ladder":
            note(f"no straight stair fits on storey {k}: ladder at ({s.x}, {s.z})")
        else:
            note(f"stair added from storey {k} at ({s.x}, {s.z}) climbing {s.dir}")
        bp.stairs.append(s)
    bp.stairs.sort(key=lambda s: s.from_storey)

    # 9. connectivity ---------------------------------------------------------
    if not any(o.kind in ("door", "arch") and OUTSIDE in o.between and o.storey == 0
               for o in bp.openings):
        door = _front_door(bp)
        if door is not None:
            bp.openings.append(door)
            note(f"front door added at ({door.x}, {door.z})"
                 + (f" into '{door.between[0]}'" if door.between[0] else ""))
    for _ in range(len(bp.rooms) + 1):
        reach = bp.reachable()
        todo = [r for r in bp.rooms if r.name not in reach]
        if not todo:
            break
        added = False
        for r in todo:
            best: Optional[Tuple[int, Room, List[Coord2]]] = None
            for o in bp.rooms_on(r.storey):
                if o.name not in reach:
                    continue
                wall = r.rect.shared_wall(o.rect)
                free = [c for c in wall if _door_cell_ok(bp, r.storey, c)]
                if free and (best is None or len(wall) > best[0]):
                    best = (len(wall), o, free)
            if best is None:
                continue
            _, o, free = best
            cell = free[len(free) // 2]
            bp.openings.append(Opening("door", r.storey, cell[0], cell[1], 1, 2, 0,
                                       (r.name, o.name)))
            note(f"door added at ({cell[0]}, {cell[1]}) between '{r.name}' and '{o.name}' "
                 "(room was unreachable)")
            added = True
            break
        if not added:
            # A ground-floor room cut off from the others still gets its own
            # entrance when it has an outer wall (a side door).
            for r in todo:
                if r.storey != 0:
                    continue
                cells = [c for c in _exterior_candidates(bp, r, 0) if _door_cell_ok(bp, 0, c)]
                if cells:
                    c = cells[len(cells) // 2]
                    bp.openings.append(Opening("door", 0, c[0], c[1], 1, 2, 0, (r.name, OUTSIDE)))
                    note(f"outside door added at ({c[0]}, {c[1]}) into '{r.name}' "
                         "(room was unreachable)")
                    added = True
                    break
        if not added:
            for r in todo:
                msg = f"room '{r.name}' is unreachable (no wall shared with a reachable room)"
                if msg not in bp.notes:          # a standing condition, noted once
                    note(msg)
            break
    return bp


def _cell_taken(bp: Blueprint, storey: int, cell: Coord2) -> bool:
    return any(o.storey == storey and cell in _opening_cells(bp, o) for o in bp.openings)


def _stair_cells_on(bp: Blueprint, storey: int) -> Set[Coord2]:
    """(x, z) of stair runs on ``storey`` (and the top of runs arriving from below)."""
    out: Set[Coord2] = set()
    for st in bp.stairs:
        if st.from_storey == storey:
            out |= {(x, z) for (x, _dy, z) in st.cells(bp._height_of(storey))}
        elif st.from_storey == storey - 1:
            out |= {(x, z) for (x, _dy, z) in st.cells(bp._height_of(storey - 1))[-3:]}
    return out


def _door_cell_ok(bp: Blueprint, storey: int, cell: Coord2) -> bool:
    """A new door may go at ``cell``: not on another opening, and not opening onto a
    stair run (a later repair pass would otherwise move the stair: not idempotent)."""
    if _cell_taken(bp, storey, cell):
        return False
    run = _stair_cells_on(bp, storey)
    x, z = cell
    return not any((x + dx, z + dz) in run for dx, dz in ((0, 1), (0, -1), (1, 0), (-1, 0)))


def _stair_ok(bp: Blueprint, s: Stair) -> bool:
    """The run stays in one room's interior (or the storey's interior when it has no
    rooms), under the floor above, clear of door fronts; the top lands in a room
    above (or the upper storey's interior)."""
    k = s.from_storey
    try:
        h = bp.storey(k).height
    except KeyError:
        return False
    cells = s.cells(h)
    rooms = bp.rooms_on(k)
    room = bp._room_at(k, cells[0][0], cells[0][2]) if rooms else None
    area = (room.interior if room is not None else
            (bp.footprint_of(k).interior() if not rooms else None))
    if area is None:
        return False
    upper = bp.footprint_of(k + 1).interior()
    if upper is None:
        return False
    for (x, _dy, z) in cells:
        if not area.contains(x, z) or not upper.contains(x, z):
            return False
    up_rooms = bp.rooms_on(k + 1)
    if up_rooms:
        # The top steps come up through the slab: they must land in a room above,
        # not under one of its walls.
        if not all(any(r.interior is not None and r.interior.contains(c[0], c[2])
                       for r in up_rooms) for c in cells[-3:]):
            return False
    fronts = set()
    for o in bp.openings:
        if o.kind in ("door", "arch") and o.storey in (k, k + 1):
            for (x, z) in _opening_cells(bp, o):
                for dx, dz in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    fronts.add((o.storey, x + dx, z + dz))
    if any((k, x, z) in fronts for (x, _dy, z) in cells):
        return False
    if any((k + 1, x, z) in fronts for (x, _dy, z) in cells[-3:]):
        return False
    return True


def _place_stair(bp: Blueprint, k: int) -> Stair:
    """A straight stair along the longest interior wall of the largest room of
    storey k that fits under storey k+1; else a ladder column."""
    rooms = sorted(bp.rooms_on(k), key=lambda r: -(r.interior.area if r.interior else 0))
    areas: List[Rect] = [r.interior for r in rooms if r.interior is not None]
    if not rooms:
        it = bp.footprint_of(k).interior()
        areas = [it] if it is not None else []
    for it in areas:
        # Rows/columns along each interior wall, longest wall first.
        runs: List[Tuple[int, str, List[Coord2]]] = []
        for z in (it.z0, it.z1):
            runs.append((it.w, "x", [(x, z) for x in range(it.x0, it.x1 + 1)]))
        for x in (it.x0, it.x1):
            runs.append((it.d, "z", [(x, z) for z in range(it.z0, it.z1 + 1)]))
        runs.sort(key=lambda r: -r[0])
        for _, axis, cells in runs:
            for direction in ("+", "-"):
                seq = cells if direction == "+" else cells[::-1]
                for start in seq:
                    s = Stair(k, start[0], start[1], f"{direction}{axis}")
                    if _stair_ok(bp, s):
                        return s
    # Ladder fallback: the first interior cell that is not a door front, landing
    # in a room above if possible, else anywhere a ladder is legal.
    for check in (_stair_ok, _ladder_ok):
        for it in areas + [bp.footprint_of(k).interior()]:
            if it is None:
                continue
            for x in range(it.x0, it.x1 + 1):
                for z in range(it.z0, it.z1 + 1):
                    s = Stair(k, x, z, "+x", kind="ladder")
                    if check(bp, s):
                        return s
    it = bp.footprint_of(k).interior() or bp.footprint_of(k)
    return Stair(k, it.x0, it.z0, "+x", kind="ladder")


def _ladder_ok(bp: Blueprint, s: Stair) -> bool:
    """The relaxed rule a ladder (the last resort) must meet: its column stands inside
    both storeys' footprint interiors and not in front of a door."""
    k = s.from_storey
    lo, hi = bp.footprint_of(k).interior(), bp.footprint_of(k + 1).interior()
    if lo is None or hi is None or not lo.contains(s.x, s.z) or not hi.contains(s.x, s.z):
        return False
    for o in bp.openings:
        if o.kind in ("door", "arch") and o.storey in (k, k + 1):
            for (x, z) in _opening_cells(bp, o):
                if abs(x - s.x) + abs(z - s.z) <= 1:
                    return False
    return True


def _front_door(bp: Blueprint) -> Optional[Opening]:
    """A door in the middle of the front (z1) wall segment of the storey-0 room with
    the longest front wall; else any exterior wall."""
    fp = bp.footprint_of(0)
    rooms = bp.rooms_on(0)
    best: Optional[Tuple[int, str, List[Coord2]]] = None
    for r in rooms or [None]:  # type: ignore[list-item]
        rect = r.rect if r is not None else fp
        front = [(x, fp.z1) for x in range(rect.x0 + 1, rect.x1)
                 if rect.z1 == fp.z1 and not fp.is_corner(x, fp.z1)]
        front = [c for c in front if _door_cell_ok(bp, 0, c)]
        if front and (best is None or len(front) > best[0]):
            best = (len(front), r.name if r is not None else "", front)
    if best is None:
        for r in rooms or [None]:  # type: ignore[list-item]
            cells = [c for c in _exterior_candidates(bp, r, 0) if _door_cell_ok(bp, 0, c)]
            if cells:
                best = (len(cells), r.name if r is not None else "", cells)
                break
    if best is None:
        return None
    _, name, cells = best
    c = cells[len(cells) // 2]
    return Opening("door", 0, c[0], c[1], 1, 2, 0, (name, OUTSIDE))


# ===========================================================================
# DSL
# ===========================================================================
def _runs(cells: Set[Coord2]) -> List[Tuple[Coord2, Coord2]]:
    """Cover a cell set with axis-aligned segments, each cell once: maximal runs
    along x first, then the leftover single cells merged along z."""
    left = set(cells)
    out: List[Tuple[Coord2, Coord2]] = []
    for z in sorted({c[1] for c in left}):
        xs = sorted(x for (x, zz) in left if zz == z)
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
                j += 1
            if j > i:
                out.append(((xs[i], z), (xs[j], z)))
                for x in xs[i:j + 1]:
                    left.discard((x, z))
            i = j + 1
    for x in sorted({c[0] for c in left}):
        zs = sorted(z for (xx, z) in left if xx == x)
        i = 0
        while i < len(zs):
            j = i
            while j + 1 < len(zs) and zs[j + 1] == zs[j] + 1:
                j += 1
            out.append(((x, zs[i]), (x, zs[j])))
            i = j + 1
    return out


def _shell_dsl(bp: Blueprint, parts: Sequence[str], storeys: Optional[Sequence[int]]) -> str:
    m = bp.materials
    want = [s for s in bp.storeys if storeys is None or s.index in set(storeys)]
    out: List[str] = []
    top = max((s.index for s in bp.storeys), default=0)
    for part in SHELL_PARTS:
        if part not in parts:
            continue
        for s in want:
            fp = bp.footprint_of(s.index)
            lines: List[str] = []
            y0, y1 = s.floor_y + 1, s.floor_y + s.height
            if part == "foundation" and s.index == 0:
                lines.append(f"fill {m['foundation']} {fp.x0} 0 {fp.z0} {fp.x1} 0 {fp.z1}")
                it = fp.interior()
                if it is not None and m["floor"] != m["foundation"]:
                    lines.append(f"fill {m['floor']} {it.x0} 0 {it.z0} {it.x1} 0 {it.z1}")
            elif part == "floors" and s.index >= 1:
                below = bp.footprint_of(s.index - 1)
                lines.append(f"fill {m['floor']} {fp.x0} {s.floor_y} {fp.z0} "
                             f"{fp.x1} {s.floor_y} {fp.z1}")
                if below.intersect(fp) != below:
                    lines.append(f"fill {m['floor']} {below.x0} {s.floor_y} {below.z0} "
                                 f"{below.x1} {s.floor_y} {below.z1}")
            elif part == "walls":
                lines.append(f"walls {m['wall']} {fp.x0} {y0} {fp.z0} {fp.x1} {y1} {fp.z1}")
            elif part == "corners" and m["trim"] != m["wall"]:
                for (x, z) in ((fp.x0, fp.z0), (fp.x1, fp.z0), (fp.x0, fp.z1), (fp.x1, fp.z1)):
                    lines.append(f"line {m['trim']} {x} {y0} {z} {x} {y1} {z}")
            elif part == "partitions":
                cells: Set[Coord2] = set()
                for r in bp.rooms_on(s.index):
                    for c in r.rect.border_cells():
                        if not fp.on_border(*c):
                            cells.add(c)
                for (a, b) in _runs(cells):
                    lines.append(f"fill {m['interior_wall']} {a[0]} {y0} {a[1]} "
                                 f"{b[0]} {y1} {b[1]}")
            elif part == "openings":
                for o in bp.openings:
                    if o.storey != s.index:
                        continue
                    for (x, z) in _opening_cells(bp, o):
                        if o.kind == "window":
                            ya = s.floor_y + 1 + o.sill
                            lines.append(f"fill {m['window']} {x} {ya} {z} {x} "
                                         f"{ya + o.height - 1} {z}")
                        else:
                            lines.append(f"clear {x} {y0} {z} {x} {y0 + o.height - 1} {z}")
                            if o.kind == "door":
                                lines.append(f"set {m['door']} {x} {y0} {z}")
                                lines.append(f"set {m['door']} {x} {y0 + 1} {z}")
            elif part == "stairs":
                for st in bp.stairs:
                    if st.from_storey != s.index:
                        continue
                    cells3 = st.cells(s.height)
                    if st.kind == "ladder":
                        x, _, z = cells3[0]
                        lines.append(f"line ladder {x} {s.floor_y + 1} {z} "
                                     f"{x} {s.floor_y + s.height + 1} {z}")
                        continue
                    facing = _DIRS[st.dir][2]
                    for (x, dy, z) in cells3:
                        lines.append(f"set {m['stairs']}[facing={facing}] {x} {s.floor_y + dy} {z}")
                    for (x, _dy, z) in cells3[-3:-1]:
                        lines.append(f"clear {x} {s.ceiling_y} {z} {x} {s.ceiling_y} {z}")
            elif part == "ceiling" and s.index == top and bp.roof.kind != "flat":
                ty = bp.top_y()
                lines.append(f"fill {m['ceiling']} {fp.x0} {ty} {fp.z0} {fp.x1} {ty} {fp.z1}")
            if lines:
                out.append(f"# shell: {part} storey {s.index}")
                out.extend(lines)
    return "\n".join(out) + ("\n" if out else "")


def _roof_dsl(bp: Blueprint, size: Optional[Size3]) -> str:
    roof = bp.roof
    if roof.kind == "none":
        return ""
    top = max((s.index for s in bp.storeys), default=0)
    fp = bp.footprint_of(top)
    ov = max(0, int(roof.overhang))
    ty = bp.top_y()
    if size is None:
        size = (fp.x1 + ov + 1, ty + roof.height(fp) + 1, fp.z1 + ov + 1)
    W, H, D = (int(v) for v in size)
    block = roof.block or bp.materials.get("roof", DEFAULT_MATERIALS["roof"])
    x0, z0 = max(0, fp.x0 - ov), max(0, fp.z0 - ov)
    x1, z1 = min(W - 1, fp.x1 + ov), min(D - 1, fp.z1 + ov)
    if ty > H - 1:
        return ""
    lines = [f"# roof: {roof.kind}"]
    if roof.kind == "flat":
        flat = roof.block or bp.materials.get("roof_flat", DEFAULT_MATERIALS["roof_flat"])
        lines.append(f"fill {flat} {fp.x0} {ty} {fp.z0} {fp.x1} {ty} {fp.z1}")
        if ty + 1 <= H - 1:
            lines.append(f"walls {bp.materials.get('wall', DEFAULT_MATERIALS['wall'])} "
                         f"{fp.x0} {ty + 1} {fp.z0} {fp.x1} {ty + 1} {fp.z1}")
        return "\n".join(lines) + "\n"
    if roof.kind == "gable":
        lines.append(f"gable {block} {x0} {ty} {z0} {x1} {ty} {z1} axis={roof.axis}")
        return "\n".join(lines) + "\n"
    # hip / pyramid: a square odd footprint is the pyramid primitive; anything else
    # shrinking one-high rings, the cap filled (a hip roof over a rectangle).
    sx, sz = x1 - x0 + 1, z1 - z0 + 1
    if sx == sz and sx % 2 == 1:
        half = (sx - 1) // 2
        lines.append(f"pyramid {block} {x0 + half} {ty} {z0 + half} {half} hollow=true")
        return "\n".join(lines) + "\n"
    level = 0
    while True:
        a0, b0, a1, b1 = x0 + level, z0 + level, x1 - level, z1 - level
        y = ty + level
        if a0 > a1 or b0 > b1 or y > H - 1:
            break
        last = (a1 - a0 < 2) or (b1 - b0 < 2)
        cmd = "fill" if last else "walls"
        lines.append(f"{cmd} {block} {a0} {y} {b0} {a1} {y} {b1}")
        if last:
            break
        level += 1
    return "\n".join(lines) + "\n"


# ===========================================================================
# Factories and ordering
# ===========================================================================
def default_blueprint(size: Size3, prompt: str = "") -> Blueprint:
    """The fallback when no plan could be parsed: one storey, one room 'main', a
    front door, a gable roof. Always valid for any canvas of at least 8^3."""
    W, H, D = (int(v) for v in size)
    w, d = max(MIN_ROOM_SIDE, int(W * 0.5)), max(MIN_ROOM_SIDE, int(D * 0.4))
    x0, z0 = (W - w) // 2, (D - d) // 2
    fp = Rect(x0, z0, x0 + w - 1, z0 + d - 1)
    mid = (fp.x0 + fp.x1) // 2
    bp = Blueprint(footprint=fp, storeys=[Storey(0, 0, 4)],
                   rooms=[Room("main", "main room", 0, fp, desc=prompt[:200])],
                   openings=[Opening("door", 0, mid, fp.z1, between=("main", OUTSIDE))],
                   roof=Roof("gable", "x" if fp.w >= fp.d else "z", 1),
                   style=prompt[:200])
    return bp.repaired(size)


def mock_blueprint(size: Size3, *, storeys: int = 1) -> Blueprint:
    """A deterministic multi-room plan for mock:auto and tests.

    storeys=1: hall + kitchen in front, bedroom behind, sharing walls; front door
    into the hall, inner doors, windows, gable along x. storeys=2 (needs H >= 24):
    adds a study and a loft upstairs and one stair from the hall. A repair fixed
    point, and its shell runs with zero failed / no-op commands.
    """
    W, H, D = (int(v) for v in size)
    two = storeys >= 2 and H >= 24
    # Laid out for 32^3 and shifted so it is centred on other canvases.
    ox, oz = (W - 32) // 2, (D - 32) // 2

    def R(x0: int, z0: int, x1: int, z1: int) -> Rect:
        return Rect(x0 + ox, z0 + oz, x1 + ox, z1 + oz)

    def O(kind: str, st: int, x: int, z: int, a: str, b: str, **kw: Any) -> Opening:
        h = kw.pop("height", 1 if kind == "window" else 2)
        sill = kw.pop("sill", 1 if kind == "window" else 0)
        return Opening(kind, st, x + ox, z + oz, kw.pop("width", 1), h, sill, (a, b))

    rooms = [Room("hall", "entrance hall", 0, R(6, 14, 15, 23),
                  "a bench, a coat rack, a rug and a lantern"),
             Room("kitchen", "kitchen", 0, R(15, 14, 25, 23),
                  "a furnace, a crafting table, barrels and a small table with chairs"),
             Room("bedroom", "bedroom", 0, R(6, 8, 25, 14),
                  "two beds, a chest, a bookshelf and a carpet")]
    openings = [O("door", 0, 10, 23, "hall", OUTSIDE),
                O("door", 0, 15, 18, "hall", "kitchen"),
                O("door", 0, 10, 14, "hall", "bedroom"),
                O("window", 0, 20, 23, "kitchen", OUTSIDE, width=2),
                O("window", 0, 25, 18, "kitchen", OUTSIDE),
                O("window", 0, 12, 8, "bedroom", OUTSIDE, width=2),
                O("window", 0, 20, 8, "bedroom", OUTSIDE, width=2)]
    stairs: List[Stair] = []
    st_list = [Storey(0, 0, 4)]
    if two:
        st_list.append(Storey(1, 5, 4))
        rooms += [Room("study", "study", 1, R(6, 8, 15, 23),
                       "a desk, bookshelves and a reading chair"),
                  Room("loft", "loft bedroom", 1, R(15, 8, 25, 23),
                       "a bed, a wardrobe and a carpet")]
        openings += [O("door", 1, 15, 12, "study", "loft"),
                     O("window", 1, 10, 23, "study", OUTSIDE, width=2),
                     O("window", 1, 20, 23, "loft", OUTSIDE, width=2),
                     O("window", 1, 25, 15, "loft", OUTSIDE)]
        stairs.append(Stair(0, 8 + ox, 17 + oz, "+x"))
    return Blueprint(footprint=R(6, 8, 25, 23), storeys=st_list, rooms=rooms,
                     openings=openings, stairs=stairs, roof=Roof("gable", "x", 1),
                     materials=dict(DEFAULT_MATERIALS),
                     style="a stone-brick cottage with oak trim and a spruce gable roof")


def order_rooms(rooms: Sequence[Room], order: str) -> List[Room]:
    """``blueprint`` = as listed; ``size`` = largest interior first; ``public_first`` =
    PUBLIC_PURPOSES first (in that ranking), then by storey, then size. Ties keep
    list order (sorts are stable)."""
    rooms = list(rooms)
    if order not in ROOM_ORDERS:
        raise ValueError(f"room order '{order}' unknown; one of {ROOM_ORDERS}")

    def size(r: Room) -> int:
        it = r.interior
        return it.area if it is not None else 0

    if order == "blueprint":
        return rooms
    if order == "size":
        return sorted(rooms, key=lambda r: -size(r))

    def rank(r: Room) -> int:
        text = f"{r.purpose} {r.name}".lower()
        for i, p in enumerate(PUBLIC_PURPOSES):
            if p in text:
                return i
        return len(PUBLIC_PURPOSES)

    return sorted(rooms, key=lambda r: (rank(r), r.storey, -size(r)))


def front_side() -> str:
    """'z1': the front of a blueprint is its z1 side (faces +z, seen by the iso camera)."""
    return "z1"


# ===========================================================================
# Schematics
# ===========================================================================
def _purpose_colour(purpose: str, name: str) -> Tuple[int, int, int]:
    text = f"{purpose} {name}".lower()
    for key, rgb in PURPOSE_COLOURS.items():
        if key in text:
            return rgb
    h = hashlib.sha256(text.encode()).digest()
    return (180 + h[0] % 70, 180 + h[1] % 70, 180 + h[2] % 70)


def _frame(px: int, region: Rect) -> Tuple[float, int, int]:
    """(cell size, x offset, y offset) fitting ``region`` cells into a px square with
    a margin for the axis labels and the title strip."""
    margin, title = max(28, px // 20), max(22, px // 28)
    avail = px - 2 * margin
    cell = min(avail / region.w, (avail - title) / region.d)
    ox = margin + int((avail - cell * region.w) / 2)
    oy = margin + title + int((avail - title - cell * region.d) / 2)
    return cell, ox, oy


def _draw_axes(draw: Any, region: Rect, cell: float, ox: int, oy: int, font: Any,
               grid: bool = True) -> None:
    for i in range(region.w + 1):
        x = ox + i * cell
        if grid:
            draw.line([(x, oy), (x, oy + region.d * cell)],
                      fill=(170, 170, 170) if (region.x0 + i) % 4 == 0 else (225, 225, 225))
    for j in range(region.d + 1):
        y = oy + j * cell
        if grid:
            draw.line([(ox, y), (ox + region.w * cell, y)],
                      fill=(170, 170, 170) if (region.z0 + j) % 4 == 0 else (225, 225, 225))
    for i in range(region.w):
        gx = region.x0 + i
        if gx % 4 == 0:
            draw.text((ox + i * cell + 1, oy - 14), str(gx), fill=(40, 40, 40), font=font)
    for j in range(region.d):
        gz = region.z0 + j
        if gz % 4 == 0:
            draw.text((max(0, ox - 22), oy + j * cell), str(gz), fill=(40, 40, 40), font=font)


def _font(size: int) -> Any:
    try:
        from blockgen.agentic.texsheet import load_font

        return load_font(size)
    except Exception:                                     # noqa: BLE001
        from PIL import ImageFont

        return ImageFont.load_default()


def _view_rect(bp: Blueprint, size: Optional[Size3] = None) -> Rect:
    fp = bp.footprint
    pad = 2
    return Rect(max(0, fp.x0 - pad), max(0, fp.z0 - pad), fp.x1 + pad, fp.z1 + pad)


def render_schematic(bp: Blueprint, storey: int = 0, *, px: int = 768,
                     highlight: Optional[str] = None, labels: bool = True,
                     axes: bool = True) -> bytes:
    """Top-down labelled plan of one storey as PNG bytes (one cell per block).

    Grid lines every cell (darker every 4), coordinate labels every 4 cells on the
    top (x) and left (z) edges so a VLM can read positions, rooms filled by purpose,
    walls dark, doors as gaps with an orange marker, arches as gaps, windows blue,
    stairs hatched with an arrow, the ``highlight`` room outlined in red with the
    others dimmed, and a title strip 'storey i (y=floor_y)'.
    """
    from PIL import Image, ImageDraw

    view = _view_rect(bp)
    cell, ox, oy = _frame(px, view)
    img = Image.new("RGB", (px, px), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _font(max(10, px // 64))
    big = _font(max(12, px // 40))

    def box(x: int, z: int) -> List[float]:
        X, Z = x - view.x0, z - view.z0
        return [ox + X * cell, oy + Z * cell, ox + (X + 1) * cell - 1, oy + (Z + 1) * cell - 1]

    def blend(rgb: Tuple[int, int, int], f: float) -> Tuple[int, int, int]:
        return tuple(int(c + (255 - c) * f) for c in rgb)  # type: ignore[return-value]

    try:
        s = bp.storey(storey)
        title = f"storey {storey} (y={s.floor_y})"
    except KeyError:
        title = f"storey {storey} (missing)"
    fp = bp.footprint_of(storey)
    rooms = bp.rooms_on(storey)
    # floor
    for x in range(fp.x0, fp.x1 + 1):
        for z in range(fp.z0, fp.z1 + 1):
            draw.rectangle(box(x, z), fill=(245, 240, 230))
    for r in rooms:
        rgb = _purpose_colour(r.purpose, r.name)
        if highlight and r.name != highlight:
            rgb = blend(rgb, 0.6)
        it = r.interior
        if it is not None:
            for x in range(it.x0, it.x1 + 1):
                for z in range(it.z0, it.z1 + 1):
                    draw.rectangle(box(x, z), fill=rgb)
    if axes:
        _draw_axes(draw, view, cell, ox, oy, font)
    # walls
    walls: Set[Coord2] = set(fp.border_cells())
    for r in rooms:
        walls |= set(r.rect.border_cells())
    gaps: Dict[Coord2, str] = {}
    for o in bp.openings:
        if o.storey == storey:
            for c in _opening_cells(bp, o):
                gaps[c] = o.kind
    for c in sorted(walls):
        kind = gaps.get(c)
        if kind == "window":
            draw.rectangle(box(*c), fill=(80, 150, 230))
        elif kind in ("door", "arch"):
            b = box(*c)
            draw.rectangle(b, fill=(255, 255, 255))
            if kind == "door":
                m = cell * 0.3
                draw.rectangle([b[0] + m, b[1] + m, b[2] - m, b[3] - m], fill=(240, 140, 30))
        else:
            draw.rectangle(box(*c), fill=(55, 55, 60))
    # stairs (this storey's runs; arrivals from below as hatched cells too)
    for st in bp.stairs:
        if st.from_storey not in (storey, storey - 1):
            continue
        try:
            h = bp.storey(st.from_storey).height
        except KeyError:
            continue
        cells = st.cells(h)
        show = cells if st.from_storey == storey else cells[-3:]
        for (x, _dy, z) in show:
            b = box(x, z)
            draw.rectangle(b, fill=(200, 185, 160))
            cw = b[2] - b[0]
            for t in range(max(3, int(cell / 3)), int(cw * 2), max(3, int(cell / 3))):
                draw.line([(b[0] + min(t, cw), b[1] + max(0, t - cw)),
                           (b[0] + max(0, t - cw), b[1] + min(t, cw))], fill=(120, 100, 80))
        if st.from_storey == storey and st.kind != "ladder":
            a, z_ = box(cells[0][0], cells[0][2]), box(cells[-1][0], cells[-1][2])
            p0 = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
            p1 = ((z_[0] + z_[2]) / 2, (z_[1] + z_[3]) / 2)
            draw.line([p0, p1], fill=(150, 30, 30), width=max(2, int(cell / 6)))
            dx, dz, _ = _DIRS[st.dir]
            hs = cell * 0.4
            tip = (p1[0] + dx * hs, p1[1] + dz * hs)
            draw.polygon([tip, (p1[0] - dz * hs, p1[1] + dx * hs),
                          (p1[0] + dz * hs, p1[1] - dx * hs)], fill=(150, 30, 30))
    if highlight:
        for r in rooms:
            if r.name == highlight:
                a, b = box(r.rect.x0, r.rect.z0), box(r.rect.x1, r.rect.z1)
                draw.rectangle([a[0] - 2, a[1] - 2, b[2] + 2, b[3] + 2], outline=(230, 0, 0),
                               width=max(3, int(cell / 4)))
    if labels:
        for r in rooms:
            a, b = box(r.rect.x0, r.rect.z0), box(r.rect.x1, r.rect.z1)
            cx, cy = (a[0] + b[2]) / 2, (a[1] + b[3]) / 2
            for k, text in enumerate([r.name] + ([r.purpose] if r.purpose and r.purpose != r.name else [])):
                tw = draw.textlength(text, font=font) if hasattr(draw, "textlength") else 6 * len(text)
                y = cy - 8 + k * 14
                draw.rectangle([cx - tw / 2 - 2, y - 1, cx + tw / 2 + 2, y + 12], fill=(255, 255, 255))
                draw.text((cx - tw / 2, y), text, fill=(0, 0, 0), font=font)
    draw.rectangle([0, 0, px, max(22, px // 28) - 2], fill=(235, 235, 235))
    draw.text((6, 3), title, fill=(0, 0, 0), font=big)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_schematics(bp: Blueprint, *, px: int = 768) -> bytes:
    """Every storey's schematic side by side (one px x px panel each)."""
    from PIL import Image

    panels = [Image.open(io.BytesIO(render_schematic(bp, s.index, px=px)))
              for s in sorted(bp.storeys, key=lambda s: s.index)] or \
             [Image.open(io.BytesIO(render_schematic(bp, 0, px=px)))]
    img = Image.new("RGB", (px * len(panels), px), (255, 255, 255))
    for i, p in enumerate(panels):
        img.paste(p.convert("RGB"), (i * px, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_grid(size: Size3, *, footprint: Optional[Rect] = None, px: int = 768) -> bytes:
    """The schematic frame with only the grid, axes and the footprint outline
    (dashed): the reference image a dreamed plan is drawn on."""
    from PIL import Image, ImageDraw

    W, _H, D = (int(v) for v in size)
    if footprint is not None:
        view = Rect(max(0, footprint.x0 - 2), max(0, footprint.z0 - 2),
                    footprint.x1 + 2, footprint.z1 + 2)
    else:
        view = Rect(0, 0, W - 1, D - 1)
    cell, ox, oy = _frame(px, view)
    img = Image.new("RGB", (px, px), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    _draw_axes(draw, view, cell, ox, oy, _font(max(10, px // 64)))
    if footprint is not None:
        x0 = ox + (footprint.x0 - view.x0) * cell
        z0 = oy + (footprint.z0 - view.z0) * cell
        x1 = ox + (footprint.x1 - view.x0 + 1) * cell
        z1 = oy + (footprint.z1 - view.z0 + 1) * cell
        dash = max(4, int(cell / 2))
        for (a, b) in (((x0, z0), (x1, z0)), ((x1, z0), (x1, z1)),
                       ((x1, z1), (x0, z1)), ((x0, z1), (x0, z0))):
            n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) // dash) or 1
            for i in range(0, n, 2):
                t0, t1 = i / n, min(1.0, (i + 1) / n)
                draw.line([(a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0),
                           (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)],
                          fill=(60, 60, 60), width=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


__all__ = ["Blueprint", "BlueprintError", "Coord2", "DEFAULT_MATERIALS", "MAX_STOREY_HEIGHT",
           "MIN_ROOM_SIDE", "MIN_STOREY_HEIGHT", "OPENING_KINDS", "OUTSIDE", "Opening",
           "PUBLIC_PURPOSES", "PURPOSE_COLOURS", "ROOF_KINDS", "ROOM_ORDERS", "Rect", "Roof",
           "Room", "SCHEMA_TEXT", "SHELL_PARTS", "STAIR_DIRS", "Size3", "Stair", "Storey",
           "default_blueprint", "front_side", "mock_blueprint", "order_rooms",
           "render_grid", "render_schematic", "render_schematics"]
