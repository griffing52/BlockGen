"""Region-scoped execution: let a model build inside one room without touching the rest.

Why post-execution clipping rather than a restricted grammar: it works for every action
space (dsl, blocks, python, mixed all lower to the same IR and run on the same world),
and the model gets precise feedback (which of ITS lines wrote outside) instead of a
silent no-op.

The pieces, in the order :meth:`Episode.try_` uses them:

* :func:`translate_lowered` moves a reply written in room-local coordinates into the
  canvas frame (``offset`` = the room's interior floor corner);
* the program runs on the whole world as usual;
* :func:`restore_outside` copies every voxel outside the :class:`Region` back from the
  current accepted world, so a stray ``fill`` cannot damage another room;
* :func:`clipped_issues` re-runs the program with per-command voxel tracking and turns
  the discarded writes into ``warning`` Issues addressed to the model's own lines.

:func:`region_structure` cuts one room out of a world for a cutaway render.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from blockgen.agentic.core import Lowered
from blockgen.agentic.dsl import COMMANDS, Issue, ProgramRunner, lookup, parse_program
from blockgen.utils.data import Structure

Coord3 = Tuple[int, int, int]


# ===========================================================================
# Boxes and regions
# ===========================================================================
@dataclass(frozen=True)
class Box:
    """Inclusive axis-aligned voxel box in canvas coordinates ``(x, y, z)``, y up."""

    x0: int
    y0: int
    z0: int
    x1: int
    y1: int
    z1: int

    @classmethod
    def of(cls, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int) -> "Box":
        """A box from any two opposite corners (sorted per axis)."""
        xs, ys, zs = sorted((int(x0), int(x1))), sorted((int(y0), int(y1))), sorted((int(z0), int(z1)))
        return cls(xs[0], ys[0], zs[0], xs[1], ys[1], zs[1])

    @classmethod
    def parse(cls, v: Any) -> "Box":
        """``[x0, y0, z0, x1, y1, z1]`` or a dict with those keys (numeric strings ok)."""
        if isinstance(v, Box):
            return v
        if isinstance(v, Mapping):
            try:
                vals = [v[k] for k in ("x0", "y0", "z0", "x1", "y1", "z1")]
            except KeyError as exc:
                raise ValueError(f"box dict needs x0 y0 z0 x1 y1 z1, got {dict(v)!r}") from exc
        elif isinstance(v, (list, tuple)) and len(v) == 6:
            vals = list(v)
        else:
            raise ValueError(f"cannot read a box from {v!r}")
        try:
            return cls.of(*(int(round(float(t))) for t in vals))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"box coordinates must be numbers, got {v!r}") from exc

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.x1 - self.x0 + 1, self.y1 - self.y0 + 1, self.z1 - self.z0 + 1)

    @property
    def volume(self) -> int:
        w, h, d = self.shape
        return w * h * d

    @property
    def origin(self) -> Coord3:
        return (self.x0, self.y0, self.z0)

    def contains(self, x: int, y: int, z: int) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1 and self.z0 <= z <= self.z1

    def clip(self, size: Sequence[int]) -> Optional["Box"]:
        """This box intersected with the canvas ``[0, size)``; None if nothing is left."""
        w, h, d = (int(v) for v in size)
        x0, y0, z0 = max(self.x0, 0), max(self.y0, 0), max(self.z0, 0)
        x1, y1, z1 = min(self.x1, w - 1), min(self.y1, h - 1), min(self.z1, d - 1)
        if x0 > x1 or y0 > y1 or z0 > z1:
            return None
        return Box(x0, y0, z0, x1, y1, z1)

    def expand(self, dx: int = 0, dy: int = 0, dz: int = 0) -> "Box":
        """Grown by ``d`` on both sides of each axis (negative shrinks; re-sorted)."""
        return Box.of(self.x0 - dx, self.y0 - dy, self.z0 - dz,
                      self.x1 + dx, self.y1 + dy, self.z1 + dz)

    def mask(self, size: Sequence[int]) -> np.ndarray:
        """Bool ``(W, H, D)`` array, True inside the box (clipped to the canvas)."""
        out = np.zeros(tuple(int(v) for v in size), dtype=bool)
        c = self.clip(size)
        if c is not None:
            out[c.x0:c.x1 + 1, c.y0:c.y1 + 1, c.z0:c.z1 + 1] = True
        return out

    def describe(self) -> str:
        return f"x {self.x0}..{self.x1}, y {self.y0}..{self.y1}, z {self.z0}..{self.z1}"

    def to_list(self) -> List[int]:
        return [self.x0, self.y0, self.z0, self.x1, self.y1, self.z1]


@dataclass(frozen=True)
class Region:
    """Where a stage may write: the union of ``allow`` boxes minus the ``deny`` boxes.

    ``deny`` is how a room keeps its doorways and stair run free: the model may
    furnish the room, but a block in front of a door would be reverted.
    """

    name: str
    allow: Tuple[Box, ...]
    deny: Tuple[Box, ...] = ()

    def mask(self, size: Sequence[int]) -> np.ndarray:
        out = np.zeros(tuple(int(v) for v in size), dtype=bool)
        for b in self.allow:
            out |= b.mask(size)
        for b in self.deny:
            out &= ~b.mask(size)
        return out

    def describe(self) -> str:
        text = "; ".join(b.describe() for b in self.allow) or "(nothing)"
        if self.deny:
            text += " except " + "; ".join(b.describe() for b in self.deny)
        return text

    @property
    def origin(self) -> Coord3:
        """Corner of the first allow box: the frame of room-local coordinates."""
        if not self.allow:
            return (0, 0, 0)
        return self.allow[0].origin

    @classmethod
    def whole(cls, size: Sequence[int], *, deny: Sequence[Box] = (),
              name: str = "canvas") -> "Region":
        w, h, d = (int(v) for v in size)
        return cls(name, (Box(0, 0, 0, w - 1, h - 1, d - 1),), tuple(deny))


# ===========================================================================
# Coordinates of DSL commands
# ===========================================================================
_AXIS_OF = {"x": 0, "y": 1, "z": 2, "x1": 0, "y1": 1, "z1": 2, "x2": 0, "y2": 1, "z2": 2,
            "cx": 0, "cy": 1, "cz": 2}


def _coord_params() -> Dict[str, Tuple[Tuple[str, int], ...]]:
    """Read from the DSL registry, so a command added later is translated too."""
    out: Dict[str, Tuple[Tuple[str, int], ...]] = {}
    for name, spec in COMMANDS.items():
        out[name] = tuple((p.name, _AXIS_OF[p.name]) for p in spec.params
                          if p.name in _AXIS_OF and p.kind == "int")
    return out


#: command -> ((param, axis), ...) for every coordinate parameter of every DSL command
#: (set/fill/clear/box/walls/replace/line/sphere/cylinder/pyramid/gable/copy/paste/stack).
COORD_PARAMS: Dict[str, Tuple[Tuple[str, int], ...]] = _coord_params()

_LEAD_TRAIL = re.compile(r"^(\s*)(.*?)(\s*)$", re.S)


def _translate_line(raw: str, offset: Coord3) -> str:
    """One DSL line with every coordinate argument shifted; unparseable -> verbatim."""
    cut = raw.find("#")
    code, comment = (raw, "") if cut < 0 else (raw[:cut], raw[cut:])
    m = _LEAD_TRAIL.match(code)
    if m is None:
        return raw
    lead, body, trail = m.groups()
    tokens = body.split()
    if not tokens:
        return raw
    spec = lookup(tokens[0])
    if spec is None:
        return raw
    coords = dict(COORD_PARAMS.get(spec.name, ()))
    if not coords:
        return raw
    names = {p.name for p in spec.params}
    positional: List[int] = []
    keyword: Dict[str, int] = {}
    for i, tok in enumerate(tokens[1:], start=1):
        key = tok.partition("=")[0].strip().lower() if "=" in tok and not tok.startswith("[") else ""
        if key and "[" not in key and key in names:
            keyword[key] = i
        else:
            positional.append(i)
    new = list(tokens)
    for param in spec.params:
        if param.name in keyword:
            i = keyword[param.name]
            is_kw = True
        elif positional:
            i = positional.pop(0)
            is_kw = False
        else:
            break
        if param.name not in coords:
            continue
        axis = coords[param.name]
        raw_v = new[i].partition("=")[2] if is_kw else new[i]
        try:
            v = int(round(float(raw_v))) + int(offset[axis])
        except (TypeError, ValueError, OverflowError):
            return raw                   # leave the error to the parser, at its own line
        new[i] = f"{new[i].partition('=')[0]}={v}" if is_kw else str(v)
    return lead + " ".join(new) + trail + comment


def translate_dsl(text: str, offset: Tuple[int, int, int]) -> str:
    """Add ``offset`` to every coordinate argument (positional or key=value) of every
    command line. Comments, ``# L<n>`` tags, blank and unparseable lines are kept
    verbatim and the result is line-aligned with the input, so execution issues
    still address the right lines."""
    off = tuple(int(v) for v in offset)
    if off == (0, 0, 0):
        return text
    return "\n".join(_translate_line(line, off) for line in (text or "").split("\n"))  # type: ignore[arg-type]


def translate_lowered(low: Lowered, offset: Tuple[int, int, int]) -> Lowered:
    """``low`` with its program moved by ``offset`` (source and issues unchanged)."""
    return Lowered(source=low.source, program_text=translate_dsl(low.program_text, offset),
                   issues=list(low.issues), n_source_lines=low.n_source_lines)


# ===========================================================================
# Worlds
# ===========================================================================
def world_arrays(world: Any) -> Tuple[np.ndarray, np.ndarray]:
    """``(block_ids, block_data)`` views of a MinecraftWorld-like world (``.canvas``)."""
    canvas = getattr(world, "canvas", None)
    if canvas is None or not hasattr(canvas, "block_ids"):
        raise TypeError("region-scoped execution needs a world with a .canvas")
    return canvas.block_ids, canvas.block_data


def changed_mask(after: Any, before: Optional[Any]) -> np.ndarray:
    """Bool mask of voxels where ``after`` differs from ``before`` (None = empty)."""
    ids, data = world_arrays(after)
    if before is None:
        air = int(getattr(after.canvas, "air_block_id", 0))
        return (ids != air) | (data != 0)
    b_ids, b_data = world_arrays(before)
    return (ids != b_ids) | (data != b_data)


def restore_outside(after: Any, before: Optional[Any], region: Region) -> np.ndarray:
    """Copy every voxel OUTSIDE ``region`` from ``before`` (None = an empty world) into
    ``after``, in place. Returns the bool mask of voxels that differed there (the
    reverted writes, when ``after`` started as a copy of ``before``)."""
    ids, data = world_arrays(after)
    outside = ~region.mask(ids.shape)
    diff = changed_mask(after, before) & outside
    if before is None:
        ids[outside] = int(getattr(after.canvas, "air_block_id", 0))
        data[outside] = 0
    else:
        b_ids, b_data = world_arrays(before)
        ids[outside] = b_ids[outside]
        data[outside] = b_data[outside]
    return diff


_L_TAG = re.compile(r"# L(\d+)\s*$")


def clipped_issues(start: Any, lowered: Lowered, region: Region, *,
                   max_lines: int = 8) -> List[Issue]:
    """Per source line, how many voxels it wrote outside ``region``, as warnings.

    Re-runs ``lowered`` on a clone of ``start`` (the world the attempt started from)
    with voxel tracking, so each discarded block is charged to the command that
    wrote it, then to the model's own line (``# L<n>`` tags of python / mixed
    lowerings). At most ``max_lines`` issues, plus one summary issue for the rest.
    """
    canvas = start.clone().canvas
    mask = region.mask(canvas.size)
    program = parse_program(lowered.program_text or "")
    runner = ProgramRunner(program, canvas, size=canvas.size, track_voxels=True)
    low_lines = (lowered.program_text or "").split("\n")
    src_lines = (lowered.source or "").split("\n")
    tagged = lowered.program_text != lowered.source
    aligned = len(low_lines) == len(src_lines)
    counts: Dict[int, int] = {}
    texts: Dict[int, str] = {}
    for step in runner.steps():
        n = sum(1 for (x, y, z, _b, _d) in step.voxels if not mask[x, y, z])
        if not n:
            continue
        ln, text = step.call.line_no, step.call.text
        m = _L_TAG.search(low_lines[ln - 1]) if tagged and 0 < ln <= len(low_lines) else None
        if m is not None:
            ln = int(m.group(1))
            if 0 < ln <= len(src_lines):
                text = src_lines[ln - 1].strip()
        elif tagged and aligned and 0 < ln <= len(src_lines):
            # A line-aligned lowering (dsl moved by an offset, blocks): quote the
            # line the model wrote, not its translation into canvas coordinates.
            text = src_lines[ln - 1].split("#", 1)[0].strip() or text
        counts[ln] = counts.get(ln, 0) + n
        texts.setdefault(ln, text)
    where = region.describe()
    lines = sorted(counts)
    out = [Issue(ln, texts[ln], f"{counts[ln]} block(s) outside {where} were discarded",
                 severity="warning") for ln in lines[:max_lines]]
    rest = lines[max_lines:]
    if rest:
        out.append(Issue(rest[0], texts[rest[0]],
                         f"... and {len(rest)} more line(s) wrote "
                         f"{sum(counts[ln] for ln in rest)} block(s) outside {where}; "
                         "all were discarded", severity="warning"))
    return out


def region_structure(src: Any, box: Box, *, cut_y: Optional[int] = None) -> Structure:
    """``src`` (a World, or a Structure in uncropped canvas coordinates) reduced to the
    voxels inside ``box`` with ``y <= cut_y`` (if given), cropped to non-air.

    Renderers frame the occupied bbox, so the cutaway camera is framed on the room
    for free. The result may be empty (an unbuilt room).
    """
    s = src if isinstance(src, Structure) else src.to_structure(crop=False)
    ids, data = s.block_ids.copy(), s.block_data.copy()
    keep = box.mask(ids.shape)
    if cut_y is not None:
        keep[:, max(0, int(cut_y) + 1):, :] = False     # a negative cut keeps nothing
    ids[~keep] = s.air_block_id
    data[~keep] = 0
    out = Structure(ids, data, air_block_id=s.air_block_id, metadata=dict(s.metadata))
    return out.crop_to_non_air()


__all__ = ["Box", "COORD_PARAMS", "Region", "changed_mask", "clipped_issues",
           "region_structure", "restore_outside", "translate_dsl", "translate_lowered",
           "world_arrays"]
