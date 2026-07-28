"""The build DSL: a WorldEdit-flavored command language an LLM can write.

Why a command language at all? Every other track in this repo makes the model emit
*one token per voxel* (``blockgen/tokenizers``, ``scripts/train_llm_brickgpt.py``),
which caps builds at whatever fits the context window and spends all the model's
capacity on bookkeeping. A program says ``walls oak_planks 0 1 0 15 5 11`` in nine
tokens instead of 320 block lines, so the same context buys a build one to two
orders of magnitude larger, and the *structure* of the build (a box, a roof, a
repeated window) is expressed directly rather than implied by a voxel cloud.

The language is deliberately small, flat and positional:

* one command per line, ``name arg arg …``, optional ``key=value`` for the tail
  arguments, ``#`` starts a comment;
* a leading ``/`` or ``//`` is tolerated (frontier models reflexively write
  WorldEdit syntax, and rejecting it would waste a round-trip);
* coordinates are integers, inclusive on both ends, ``y`` is up, origin at the
  canvas corner — the same convention as ``Structure`` indexing;
* nothing is stateful except the clipboard, so a program can be read, diffed and
  re-executed deterministically.

Adding a command is one :func:`register` decorator: the parser, the ``--help``
text, the system prompt's command reference and the repair prompt all read the
same registry, so a new primitive is documented to the model the moment it exists.
That is the whole extensibility story — see ``docs/agentic.md``.

Parsing and execution are separate on purpose. Parse errors (bad name, missing
argument) are reported with line numbers *without* running anything, and execution
errors (unknown block, empty region) are collected per line while the rest of the
program still runs, so a program with three bad lines out of eighty still produces
a build plus a precise list of what to fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from blockgen.agentic.blockstate import AIR, UnknownBlockError, resolve_block
from blockgen.agentic.canvas import Canvas, Coord, DEFAULT_SIZE

_REQUIRED = object()

Handler = Callable[[Canvas, Dict[str, Any]], int]


# --- command specification -------------------------------------------------
@dataclass(frozen=True)
class Param:
    """One argument of a command. ``kind`` drives both coercion and the docs."""

    name: str
    kind: str  # block | int | bool | str | choice
    default: Any = _REQUIRED
    doc: str = ""
    choices: Tuple[str, ...] = ()

    @property
    def required(self) -> bool:
        return self.default is _REQUIRED

    def render(self) -> str:
        if self.required:
            return f"<{self.name}>"
        shown = "false" if self.default is False else (
            "true" if self.default is True else self.default)
        return f"[{self.name}={shown}]"

    def coerce(self, raw: str) -> Any:
        if self.kind == "int":
            return int(round(float(raw)))  # models sometimes emit "4.0"
        if self.kind == "bool":
            low = str(raw).strip().lower()
            if low in ("true", "1", "yes", "y", "on", "hollow"):
                return True
            if low in ("false", "0", "no", "n", "off", "solid"):
                return False
            raise ValueError(f"{self.name} must be true/false, got '{raw}'")
        if self.kind == "choice":
            low = str(raw).strip().lower()
            if self.choices and low not in self.choices:
                raise ValueError(
                    f"{self.name} must be one of {'|'.join(self.choices)}, got '{raw}'")
            return low
        return str(raw)  # block specs stay strings until execution resolves them


@dataclass(frozen=True)
class CommandSpec:
    name: str
    params: Tuple[Param, ...]
    summary: str
    example: str
    handler: Handler
    aliases: Tuple[str, ...] = ()

    def signature(self) -> str:
        return " ".join([self.name] + [p.render() for p in self.params])

    def help_block(self) -> str:
        return f"  {self.signature()}\n      {self.summary}\n      e.g. {self.example}"


COMMANDS: Dict[str, CommandSpec] = {}   # canonical name -> spec
_ALIASES: Dict[str, str] = {}           # alias or canonical name -> canonical name


def register(name: str, params: Sequence[Param], summary: str, example: str,
             aliases: Sequence[str] = ()) -> Callable[[Handler], Handler]:
    """Register a DSL command. The decorated function returns voxels changed."""

    def deco(fn: Handler) -> Handler:
        spec = CommandSpec(name=name, params=tuple(params), summary=summary,
                           example=example, handler=fn, aliases=tuple(aliases))
        if name in COMMANDS:
            raise ValueError(f"duplicate DSL command '{name}'")
        COMMANDS[name] = spec
        _ALIASES[name] = name
        for a in spec.aliases:
            if a in _ALIASES:
                raise ValueError(f"duplicate DSL alias '{a}'")
            _ALIASES[a] = name
        return fn

    return deco


def lookup(name: str) -> Optional[CommandSpec]:
    """Resolve a (possibly aliased, possibly ``//``-prefixed) command name."""
    key = name.strip().lower().lstrip("/")
    canonical = _ALIASES.get(key)
    return COMMANDS.get(canonical) if canonical else None


def command_reference() -> str:
    """The whole language as prompt-ready text (also used by ``--list-commands``)."""
    lines = []
    for spec in COMMANDS.values():
        lines.append(spec.help_block())
        if spec.aliases:
            lines.append(f"      aliases: {', '.join(spec.aliases)}")
    return "\n".join(lines)


# --- shared parameter groups ----------------------------------------------
def _region_params(doc_lo: str = "one corner", doc_hi: str = "opposite corner"
                   ) -> Tuple[Param, ...]:
    return (
        Param("x1", "int", doc=doc_lo), Param("y1", "int", doc=doc_lo),
        Param("z1", "int", doc=doc_lo), Param("x2", "int", doc=doc_hi),
        Param("y2", "int", doc=doc_hi), Param("z2", "int", doc=doc_hi),
    )


def _region(a: Dict[str, Any]) -> Tuple[Coord, Coord]:
    lo = (min(a["x1"], a["x2"]), min(a["y1"], a["y2"]), min(a["z1"], a["z2"]))
    hi = (max(a["x1"], a["x2"]), max(a["y1"], a["y2"]), max(a["z1"], a["z2"]))
    return lo, hi


# --- commands: placement ---------------------------------------------------
@register(
    "set", (Param("block", "block"), Param("x", "int"), Param("y", "int"),
            Param("z", "int")),
    "Place a single block.", "set torch 4 3 4", aliases=("block", "place"),
)
def _cmd_set(canvas: Canvas, a: Dict[str, Any]) -> int:
    return canvas.set_voxel(a["x"], a["y"], a["z"], resolve_block(a["block"]))


@register(
    "fill", (Param("block", "block"),) + _region_params(),
    "Fill a solid cuboid (inclusive corners).",
    "fill stone_bricks 0 0 0 15 0 11", aliases=("cuboid",),
)
def _cmd_fill(canvas: Canvas, a: Dict[str, Any]) -> int:
    lo, hi = _region(a)
    return canvas.fill_box(lo, hi, resolve_block(a["block"]))


@register(
    "clear", _region_params(),
    "Erase a cuboid back to air — how you cut doors, windows and interiors.",
    "clear 3 1 0 5 3 0", aliases=("erase", "delete"),
)
def _cmd_clear(canvas: Canvas, a: Dict[str, Any]) -> int:
    lo, hi = _region(a)
    return canvas.fill_box(lo, hi, AIR)


@register(
    "box", (Param("block", "block"),) + _region_params() + (
        Param("thickness", "int", 1, "wall thickness in blocks"),),
    "Hollow box: all six faces, empty inside.",
    "box stone_bricks 0 0 0 9 5 9", aliases=("hollow", "faces"),
)
def _cmd_box(canvas: Canvas, a: Dict[str, Any]) -> int:
    lo, hi = _region(a)
    block = resolve_block(a["block"])
    t = max(1, int(a["thickness"]))
    n = canvas.fill_box(lo, hi, block)
    inner_lo = tuple(v + t for v in lo)
    inner_hi = tuple(v - t for v in hi)
    if all(inner_lo[i] <= inner_hi[i] for i in range(3)):
        n -= canvas.fill_box(inner_lo, inner_hi, AIR)  # type: ignore[arg-type]
    return n


@register(
    "walls", (Param("block", "block"),) + _region_params() + (
        Param("thickness", "int", 1, "wall thickness in blocks"),),
    "The four vertical walls of a cuboid — no floor, no ceiling.",
    "walls oak_planks 0 1 0 11 5 9",
)
def _cmd_walls(canvas: Canvas, a: Dict[str, Any]) -> int:
    lo, hi = _region(a)
    block = resolve_block(a["block"])
    t = max(1, int(a["thickness"]))
    n = canvas.fill_box(lo, hi, block)
    inner_lo = (lo[0] + t, lo[1], lo[2] + t)
    inner_hi = (hi[0] - t, hi[1], hi[2] - t)
    if inner_lo[0] <= inner_hi[0] and inner_lo[2] <= inner_hi[2]:
        n -= canvas.fill_box(inner_lo, inner_hi, AIR)
    return n


@register(
    "replace", (Param("from_block", "block"), Param("to_block", "block")
                ) + _region_params(),
    "Swap one block for another inside a region (use `any` to match everything "
    "non-air).",
    "replace oak_planks glass 2 3 0 4 4 0",
)
def _cmd_replace(canvas: Canvas, a: Dict[str, Any]) -> int:
    lo, hi = _region(a)
    to_block = resolve_block(a["to_block"])
    src = a["from_block"].strip().lower()
    if src in ("any", "*", "all"):
        occ = canvas.occupied_mask
        coords = [tuple(c) for c in np.argwhere(occ).tolist()
                  if all(lo[i] <= c[i] <= hi[i] for i in range(3))]
        return canvas.set_coords(coords, to_block)
    return canvas.fill_box(lo, hi, to_block, only_replace=resolve_block(src),
                           match_id_only=True)


@register(
    "line", (Param("block", "block"),) + _region_params("start", "end") + (
        Param("thickness", "int", 1, "radius in blocks (1 = single voxel wide)"),),
    "Straight line between two points (beams, rafters, fence runs).",
    "line oak_log 0 5 0 11 5 0",
)
def _cmd_line(canvas: Canvas, a: Dict[str, Any]) -> int:
    p0 = np.array([a["x1"], a["y1"], a["z1"]], dtype=float)
    p1 = np.array([a["x2"], a["y2"], a["z2"]], dtype=float)
    steps = int(max(abs(p1 - p0).max(), 1)) + 1
    pts = {tuple(int(round(v)) for v in p0 + (p1 - p0) * (i / (steps - 1) if steps > 1 else 0))
           for i in range(steps)}
    t = max(1, int(a["thickness"])) - 1
    if t:
        thick = set()
        for x, y, z in pts:
            for dx in range(-t, t + 1):
                for dy in range(-t, t + 1):
                    for dz in range(-t, t + 1):
                        thick.add((x + dx, y + dy, z + dz))
        pts = thick
    return canvas.set_coords(sorted(pts), resolve_block(a["block"]))


# --- commands: shapes ------------------------------------------------------
@register(
    "sphere", (Param("block", "block"), Param("cx", "int"), Param("cy", "int"),
               Param("cz", "int"), Param("radius", "int"),
               Param("hollow", "bool", False, "shell only")),
    "Sphere centered on a point (domes, tree crowns).",
    "sphere oak_leaves 8 12 8 4",
)
def _cmd_sphere(canvas: Canvas, a: Dict[str, Any]) -> int:
    c = np.array([a["cx"], a["cy"], a["cz"]], dtype=float)
    r = max(0, int(a["radius"]))
    rng = np.arange(-r, r + 1)
    gx, gy, gz = np.meshgrid(rng, rng, rng, indexing="ij")
    d = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    mask = d <= r + 0.5
    if a["hollow"]:
        mask &= d >= r - 0.5
    offs = np.argwhere(mask) - r
    return canvas.set_coords((offs + c).astype(int).tolist(), resolve_block(a["block"]))


@register(
    "cylinder", (Param("block", "block"), Param("cx", "int"), Param("cy", "int"),
                 Param("cz", "int"), Param("radius", "int"), Param("height", "int"),
                 Param("hollow", "bool", False, "shell only"),
                 Param("axis", "choice", "y", "axis the cylinder runs along",
                       ("x", "y", "z"))),
    "Cylinder from (cx,cy,cz) extending `height` along `axis` (towers, pillars).",
    "cylinder cobblestone 8 0 8 3 12",
)
def _cmd_cylinder(canvas: Canvas, a: Dict[str, Any]) -> int:
    r = max(0, int(a["radius"]))
    h = max(1, int(a["height"]))
    axis = {"x": 0, "y": 1, "z": 2}[a["axis"]]
    plane = [i for i in range(3) if i != axis]
    rng = np.arange(-r, r + 1)
    gu, gv = np.meshgrid(rng, rng, indexing="ij")
    d = np.sqrt(gu ** 2 + gv ** 2)
    mask = d <= r + 0.5
    if a["hollow"]:
        mask &= d >= r - 0.5
    ring = np.argwhere(mask) - r
    base = np.array([a["cx"], a["cy"], a["cz"]], dtype=int)
    pts = []
    for step in range(h):
        for u, v in ring.tolist():
            p = base.copy()
            p[plane[0]] += u
            p[plane[1]] += v
            p[axis] += step
            pts.append(tuple(int(q) for q in p))
    return canvas.set_coords(pts, resolve_block(a["block"]))


@register(
    "pyramid", (Param("block", "block"), Param("cx", "int"), Param("cy", "int"),
                Param("cz", "int"), Param("size", "int", doc="half-width at the base"),
                Param("hollow", "bool", False, "shell only")),
    "Square pyramid rising from a base centered at (cx,cy,cz).",
    "pyramid sandstone 8 6 8 5",
)
def _cmd_pyramid(canvas: Canvas, a: Dict[str, Any]) -> int:
    size = max(0, int(a["size"]))
    block = resolve_block(a["block"])
    n = 0
    for level in range(size + 1):
        half = size - level
        lo = (a["cx"] - half, a["cy"] + level, a["cz"] - half)
        hi = (a["cx"] + half, a["cy"] + level, a["cz"] + half)
        if a["hollow"] and half >= 1:
            n += canvas.fill_box(lo, hi, block)
            n -= canvas.fill_box((lo[0] + 1, lo[1], lo[2] + 1),
                                 (hi[0] - 1, hi[1], hi[2] - 1), AIR)
        else:
            n += canvas.fill_box(lo, hi, block)
    return n


@register(
    "gable", (Param("block", "block"),) + _region_params("footprint corner",
                                                         "opposite footprint corner")
    + (Param("axis", "choice", "x", "the ridge runs along this axis", ("x", "z")),
       Param("overhang", "int", 0, "blocks the eaves stick out past the walls"),
       Param("solid", "bool", False, "fill under the roof surface"),
       Param("riser", "bool", True, "close the vertical face of each step")),
    "Pitched (gable) roof over a footprint: the single highest-value house "
    "primitive. y1 is the eaves height; the ridge rises from there.",
    "gable oak_stairs 0 6 0 11 6 9 axis=x overhang=1",
)
def _cmd_gable(canvas: Canvas, a: Dict[str, Any]) -> int:
    lo, hi = _region(a)
    block = resolve_block(a["block"])
    over = max(0, int(a["overhang"]))
    x0, x1 = lo[0] - over, hi[0] + over
    z0, z1 = lo[2] - over, hi[2] + over
    y0 = lo[1]
    solid, riser = bool(a["solid"]), bool(a["riser"])
    n = 0
    # Each step rises one block and moves one block inward, so consecutive steps
    # touch only DIAGONALLY -- a roof built that way reads as a stack of floating
    # rows under the 6-connectivity validity metric the rest of the repo uses
    # (blockgen/eval/validity.py). The riser is the vertical face that closes the
    # step, which is also what a real Minecraft stair roof has behind its stairs.
    if a["axis"] == "x":  # ridge along x, slope across z
        for level in range(((z1 - z0) // 2) + 1):
            y = y0 + level
            za, zb = z0 + level, z1 - level
            if za > zb:
                break
            if solid:
                n += canvas.fill_box((x0, y, za), (x1, y, zb), block)
                continue
            if riser and level:
                n += canvas.fill_box((x0, y, za - 1), (x1, y, za - 1), block)
                if zb != za:
                    n += canvas.fill_box((x0, y, zb + 1), (x1, y, zb + 1), block)
            n += canvas.fill_box((x0, y, za), (x1, y, za), block)
            if zb != za:
                n += canvas.fill_box((x0, y, zb), (x1, y, zb), block)
    else:  # ridge along z, slope across x
        for level in range(((x1 - x0) // 2) + 1):
            y = y0 + level
            xa, xb = x0 + level, x1 - level
            if xa > xb:
                break
            if solid:
                n += canvas.fill_box((xa, y, z0), (xb, y, z1), block)
                continue
            if riser and level:
                n += canvas.fill_box((xa - 1, y, z0), (xa - 1, y, z1), block)
                if xb != xa:
                    n += canvas.fill_box((xb + 1, y, z0), (xb + 1, y, z1), block)
            n += canvas.fill_box((xa, y, z0), (xa, y, z1), block)
            if xb != xa:
                n += canvas.fill_box((xb, y, z0), (xb, y, z1), block)
    return n


# --- commands: composition -------------------------------------------------
@register(
    "copy", _region_params(),
    "Copy a region into the clipboard (one clipboard, overwritten each time).",
    "copy 0 1 0 3 4 0",
)
def _cmd_copy(canvas: Canvas, a: Dict[str, Any]) -> int:
    lo, hi = _region(a)
    clip = canvas.copy_region(lo, hi)
    return int(np.count_nonzero(clip.block_ids != canvas.air_block_id))


@register(
    "paste", (Param("x", "int"), Param("y", "int"), Param("z", "int"),
              Param("rotate", "int", 0, "degrees about the vertical axis (0/90/180/270)"),
              Param("mirror", "choice", "none", "flip the copy", ("none", "x", "z"))),
    "Paste the clipboard with its minimum corner at (x,y,z). Air is not pasted.",
    "paste 8 1 0 mirror=x",
)
def _cmd_paste(canvas: Canvas, a: Dict[str, Any]) -> int:
    return canvas.paste((a["x"], a["y"], a["z"]), rotate=a["rotate"],
                        mirror=a["mirror"])


@register(
    "stack", _region_params() + (
        Param("dir", "choice", doc="direction to repeat in",
              choices=("+x", "-x", "+y", "-y", "+z", "-z")),
        Param("count", "int", doc="number of extra copies")),
    "Repeat a region `count` times in a direction (window rows, floors, fences).",
    "stack 2 2 0 3 4 0 +x 3",
)
def _cmd_stack(canvas: Canvas, a: Dict[str, Any]) -> int:
    lo, hi = _region(a)
    clip = canvas.copy_region(lo, hi)
    sx, sy, sz = clip.shape
    step = {"+x": (sx, 0, 0), "-x": (-sx, 0, 0), "+y": (0, sy, 0),
            "-y": (0, -sy, 0), "+z": (0, 0, sz), "-z": (0, 0, -sz)}[a["dir"]]
    n = 0
    for k in range(1, max(0, int(a["count"])) + 1):
        at = tuple(clip.origin[i] + step[i] * k for i in range(3))
        n += canvas.paste(at)  # type: ignore[arg-type]
    return n


# --- parsing ---------------------------------------------------------------
@dataclass
class Call:
    """One parsed command line, ready to execute."""

    spec: CommandSpec
    args: Dict[str, Any]
    line_no: int
    text: str


@dataclass
class Issue:
    """A parse or execution problem, addressed to the model by line number."""

    line_no: int
    text: str
    message: str
    severity: str = "error"

    def render(self) -> str:
        return f"line {self.line_no}: {self.text.strip()!r} -> {self.message}"


@dataclass
class Program:
    calls: List[Call] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)
    n_lines: int = 0
    n_skipped: int = 0  # prose / fences the model wrapped around the program

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def source(self) -> str:
        return "\n".join(c.text for c in self.calls)


_FENCE_PREFIXES = ("```", "~~~")


def _is_keyword(token: str, spec: CommandSpec) -> Optional[Tuple[str, str]]:
    """``key=value`` only when ``key`` names a parameter — so a block spec like
    ``oak_stairs[facing=north]`` is never mistaken for a keyword argument."""
    if "=" not in token or token.startswith("["):
        return None
    key, _, value = token.partition("=")
    key = key.strip().lower()
    if "[" in key or not key:
        return None
    if any(p.name == key for p in spec.params):
        return key, value
    return None


def parse_line(raw: str, line_no: int) -> Tuple[Optional[Call], Optional[Issue]]:
    """Parse a single line into a :class:`Call` (or an :class:`Issue`)."""
    text = raw.split("#", 1)[0].strip()
    if not text:
        return None, None
    tokens = text.split()
    spec = lookup(tokens[0])
    if spec is None:
        return None, Issue(line_no, raw, f"unknown command '{tokens[0]}'")

    positional: List[str] = []
    keywords: Dict[str, str] = {}
    for token in tokens[1:]:
        kv = _is_keyword(token, spec)
        if kv is None:
            positional.append(token)
        else:
            keywords[kv[0]] = kv[1]

    args: Dict[str, Any] = {}
    remaining = list(positional)
    for param in spec.params:
        if param.name in keywords:
            raw_value: Any = keywords.pop(param.name)
        elif remaining:
            raw_value = remaining.pop(0)
        elif param.required:
            return None, Issue(
                line_no, raw,
                f"missing argument '{param.name}'; usage: {spec.signature()}")
        else:
            args[param.name] = param.default
            continue
        try:
            args[param.name] = param.coerce(raw_value)
        except ValueError as exc:
            return None, Issue(line_no, raw, f"{exc}; usage: {spec.signature()}")
    if remaining:
        return None, Issue(
            line_no, raw,
            f"{len(remaining)} extra argument(s) {remaining}; usage: {spec.signature()}")
    return Call(spec=spec, args=args, line_no=line_no, text=text), None


def parse_program(text: str, *, strict: bool = False) -> Program:
    """Parse a whole program, tolerating the prose and code fences LLMs add.

    ``strict=False`` (the default) records non-command lines as skipped instead of
    erroring: a model that writes "Here is the build:" before the program should
    still get a build, and the skipped count tells us how often that happens.
    """
    program = Program()
    in_fence = False
    for i, raw in enumerate(text.splitlines(), start=1):
        program.n_lines += 1
        stripped = raw.strip()
        if stripped.startswith(_FENCE_PREFIXES):
            in_fence = not in_fence
            continue
        if not stripped or stripped.startswith("#"):
            continue
        call, issue = parse_line(raw, i)
        if call is not None:
            program.calls.append(call)
        elif issue is not None:
            # An unknown *command* on a prose line is noise, not a program error,
            # unless the caller asked for strictness.
            unknown_cmd = issue.message.startswith("unknown command")
            if unknown_cmd and not strict:
                program.n_skipped += 1
            else:
                program.issues.append(issue)
    return program


# --- execution -------------------------------------------------------------
@dataclass
class ExecutionReport:
    """What happened when a program ran — the agent's feedback channel."""

    n_commands: int = 0
    n_failed: int = 0
    n_noop: int = 0            # commands that changed zero voxels
    n_skipped_lines: int = 0
    voxels_written: int = 0
    clipped_writes: int = 0
    blocks: int = 0
    bbox: Optional[Tuple[Coord, Coord]] = None
    issues: List[Issue] = field(default_factory=list)
    per_command: List[Tuple[int, str, int]] = field(default_factory=list)
    palette: List[Tuple[str, int]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.n_failed == 0 and not any(i.severity == "error" for i in self.issues)

    def summary(self) -> str:
        """Compact human/model-readable status line block."""
        size = "empty"
        if self.bbox is not None:
            lo, hi = self.bbox
            size = "x".join(str(hi[i] - lo[i] + 1) for i in range(3))
        parts = [
            f"commands: {self.n_commands} ({self.n_failed} failed, {self.n_noop} no-op)",
            f"blocks placed: {self.blocks}",
            f"bounding box: {size}" + (f" at {self.bbox[0]}" if self.bbox else ""),
        ]
        if self.clipped_writes:
            parts.append(f"voxels dropped outside the canvas: {self.clipped_writes}")
        if self.n_skipped_lines:
            parts.append(f"non-command lines ignored: {self.n_skipped_lines}")
        if self.issues:
            parts.append("problems:")
            parts += [f"  - {i.render()}" for i in self.issues[:20]]
            if len(self.issues) > 20:
                parts.append(f"  … and {len(self.issues) - 20} more")
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_commands": self.n_commands, "n_failed": self.n_failed,
            "n_noop": self.n_noop, "n_skipped_lines": self.n_skipped_lines,
            "voxels_written": self.voxels_written,
            "clipped_writes": self.clipped_writes, "blocks": self.blocks,
            "bbox": [list(self.bbox[0]), list(self.bbox[1])] if self.bbox else None,
            "issues": [{"line": i.line_no, "text": i.text.strip(),
                        "message": i.message, "severity": i.severity}
                       for i in self.issues],
            "palette": self.palette[:24],
        }


def execute(program: Program, canvas: Optional[Canvas] = None, *,
            size: Sequence[int] = DEFAULT_SIZE) -> Tuple[Canvas, ExecutionReport]:
    """Run a parsed program. A failing line is recorded and skipped, never fatal."""
    canvas = canvas if canvas is not None else Canvas(size)
    report = ExecutionReport(n_skipped_lines=program.n_skipped)
    report.issues.extend(program.issues)
    for call in program.calls:
        report.n_commands += 1
        try:
            changed = int(call.spec.handler(canvas, call.args))
        except (UnknownBlockError, ValueError, KeyError, IndexError) as exc:
            report.n_failed += 1
            report.issues.append(Issue(call.line_no, call.text, str(exc)))
            continue
        report.voxels_written += max(changed, 0)
        report.per_command.append((call.line_no, call.spec.name, changed))
        if changed == 0:
            report.n_noop += 1
            report.issues.append(Issue(
                call.line_no, call.text,
                "changed 0 blocks (region empty, out of bounds, or already that block)",
                severity="warning"))
    report.clipped_writes = canvas.clipped_writes
    report.blocks = canvas.block_count()
    report.bbox = canvas.bbox()
    report.palette = canvas.palette_counts()
    return canvas, report


def run_program(text: str, *, size: Sequence[int] = DEFAULT_SIZE,
                canvas: Optional[Canvas] = None
                ) -> Tuple[Canvas, ExecutionReport, Program]:
    """Parse + execute in one call — the convenience entry point."""
    program = parse_program(text)
    canvas, report = execute(program, canvas=canvas, size=size)
    return canvas, report, program


__all__ = ["COMMANDS", "Call", "CommandSpec", "ExecutionReport", "Issue", "Param",
           "Program", "command_reference", "execute", "lookup", "parse_line",
           "parse_program", "register", "run_program"]
