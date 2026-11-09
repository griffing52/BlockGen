"""Inspect a build the way a site engineer would, and say so in words the builder
can act on.

The repair turn (``turns.repair``) feeds back what the EXECUTOR saw: failed lines
and commands that placed nothing. It is silent about what was BUILT: a sealed box
with no door, a roof floating two blocks above its walls, a hut whose interior is
open to the sky all execute cleanly. ldraw-nova (an LLM agent that writes LDraw
LEGO models, 2026) closes that gap with an ``inspect`` stage: collision, gap and
contact checks whose report the agent reads before it repairs. This module is the
Minecraft twin. It reuses the probes the benchmark already trusts (the player
walk of ``eval.traversal``, the room statistics of ``recipes.staged_scoring``,
the player-scale heuristics of ``agentic.scale``) and turns them into a short
list of pass / fail findings with a fix hint each.

Every check is deterministic and GPU-free, so a battery can afford one per
attempt. The checks are deliberately *house-shaped* (interior, door, roof): a
bridge or a statue fails ``interior`` and ``enterable`` legitimately, and the
turn says so ("skip a finding that does not apply"). The ``inspection`` scorer
is the share of applicable checks passed; like ``scale`` it is a diagnostic
that selection may use, never a benchmark pillar.

::

    insp = inspect_structure(structure, canvas=(32, 32, 32))
    print(insp.text())          # the block the builder reads
    insp.score                  # passed / applicable, in [0, 1]
    insp.clean("warning")       # no error or warning left
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from blockgen.agentic.core import Attempt, ScoreTerm, register
from blockgen.utils.data import Structure

SEVERITIES = ("error", "warning", "info")
#: Severity rank, for ``Inspection.clean(level)``.
_RANK = {s: i for i, s in enumerate(SEVERITIES)}

#: Share of blocks in pieces not attached to the main one: above STRAY_ERROR_FRAC
#: ``connected`` is an error, above STRAY_WARN_FRAC a warning, else a note. On 400
#: real houses_32 builds 21% have some stray block, 6% more than 2%, 3% more than 5%.
STRAY_ERROR_FRAC = 0.05
STRAY_WARN_FRAC = 0.02
#: Share of blocks above the ground row with nothing below them in their column and
#: no supported horizontal neighbour (see ``_supported_frac``) before ``supported``
#: warns. 4% of real houses exceed it.
OVERHANG_WARN_FRAC = 0.25
#: Share of the interior floor area (footprint eroded by one) with no block more
#: than one high over it before ``roofed`` warns.
OPEN_SKY_WARN_FRAC = 0.25
#: Enclosed air below this is "no interior" (a doll's house of a few cells).
MIN_INTERIOR_CELLS = 8
#: Blocks / bounding-box volume above this is a solid mass, not a building.
SOLID_FILL_FRAC = 0.5
#: Share of interior standing cells that need 3 clear blocks above them.
HEADROOM_OK = 0.5
#: Footprint / canvas footprint below this earns a size note (8 x 8 on 32 x 32 passes).
SMALL_FOOTPRINT_FRAC = 0.06
#: Legacy ids of glazing: glass, glass pane, stained glass, stained glass pane.
GLASS_IDS = frozenset({20, 102, 95, 160})


@dataclass(frozen=True)
class Finding:
    """One check's verdict. ``passed`` findings carry no hint."""

    check: str
    passed: bool
    severity: str = "warning"          # of the failure; ignored when passed
    text: str = ""                     # what is wrong, with numbers
    hint: str = ""                     # how to fix it, in the build language's terms
    applicable: bool = True            # False: the check does not apply (empty build ...)

    def render(self) -> str:
        if not self.applicable:
            return f"[skip] {self.check}: {self.text}"
        if self.passed:
            return f"[ok]   {self.check}" + (f": {self.text}" if self.text else "")
        tag = {"error": "FAIL", "warning": "WARN", "info": "note"}[self.severity]
        line = f"[{tag}] {self.check}: {self.text}"
        return line + (f"\n       fix: {self.hint}" if self.hint else "")


@dataclass
class Inspection:
    findings: List[Finding] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def applicable(self) -> List[Finding]:
        return [f for f in self.findings if f.applicable]

    @property
    def failed(self) -> List[Finding]:
        return [f for f in self.applicable if not f.passed]

    @property
    def n_passed(self) -> int:
        return sum(1 for f in self.applicable if f.passed)

    @property
    def score(self) -> float:
        """Passed / applicable checks, ``info`` failures counting half."""
        app = self.applicable
        if not app:
            return 0.0
        v = sum(1.0 if f.passed else (0.5 if f.severity == "info" else 0.0) for f in app)
        return float(v / len(app))

    def clean(self, level: str = "warning") -> bool:
        """True when no failure at ``level`` or worse remains."""
        r = _RANK[level]
        return not any(_RANK[f.severity] <= r for f in self.failed)

    def worst(self) -> Optional[str]:
        fails = self.failed
        return min((f.severity for f in fails), key=_RANK.__getitem__) if fails else None

    def text(self, *, show_passed: bool = True) -> str:
        """The block the builder reads: failures first (worst first), then passes."""
        app = self.applicable
        head = (f"INSPECTION of the build (automatic geometry checks: "
                f"{self.n_passed} of {len(app)} passed)")
        fails = sorted(self.failed, key=lambda f: _RANK[f.severity])
        lines = [head] + [f.render() for f in fails]
        if show_passed:
            ok = [f.check for f in app if f.passed]
            if ok:
                lines.append("[ok]   " + ", ".join(ok))
        if self.stats:
            lines.append("Numbers: " + ", ".join(f"{k} {v}" for k, v in self.stats.items()))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"score": round(self.score, 4), "n_passed": self.n_passed,
                "n_applicable": len(self.applicable), "worst": self.worst(),
                "failed": [f.check for f in self.failed],
                "findings": [{"check": f.check, "passed": f.passed, "severity": f.severity,
                              "applicable": f.applicable, "text": f.text}
                             for f in self.findings],
                "stats": dict(self.stats)}


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------
def _components(occ: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    from scipy import ndimage
    lab, k = ndimage.label(occ, structure=np.ones((3, 3, 3), bool))
    sizes = np.bincount(lab.ravel())[1:] if k else np.zeros(0, int)
    return lab, int(k), sizes


def _bbox_str(mask: np.ndarray, offset: Sequence[int] = (0, 0, 0)) -> str:
    pts = np.argwhere(mask)
    lo, hi = pts.min(0), pts.max(0)
    size = "x".join(str(int(v)) for v in (hi - lo + 1))
    at = ", ".join(str(int(v + o)) for v, o in zip(lo, offset))
    return f"{size} at ({at})"


def _supported_frac(occ: np.ndarray, y0: int) -> Tuple[int, int]:
    """(unsupported blocks, blocks above the ground row).

    A block is supported when some block sits below it in its column, when it
    stands on the ground (``y0 == 0``), or when a horizontally adjacent block at
    the same height is supported (an eave, a sill, a one-block cantilever).
    Calibrated on 400 real houses_32 builds: the raw column rule flags 44% of
    them above 15% (corpus crops often lack their floor, so every roof block over
    a room 'hangs'); the eave rule's 90th percentile is 18% and 4% of real
    houses exceed :data:`OVERHANG_WARN_FRAC`."""
    from scipy import ndimage
    below = np.zeros_like(occ)
    below[:, 1:, :] = np.logical_or.accumulate(occ, axis=1)[:, :-1, :]
    if y0 == 0:
        below[:, 0, :] = True            # the ground row rests on the ground
    supported = occ & below
    cross = np.zeros((3, 3, 3), bool)
    cross[1, 1, 1] = True
    cross[0, 1, 1] = cross[2, 1, 1] = cross[1, 1, 0] = cross[1, 1, 2] = True   # x and z neighbours
    near = ndimage.binary_dilation(supported, structure=cross)
    above_ground = occ.copy()
    if y0 == 0:
        above_ground[:, 0, :] = False
    unsupported = occ & ~below & ~near
    return int(unsupported.sum()), int(above_ground.sum())


def _open_sky_frac(occ: np.ndarray) -> Tuple[float, int]:
    """Share of the interior floor area (footprint eroded by one in x and z) whose
    column holds nothing higher than one block: floor or ground only, no roof."""
    from scipy import ndimage
    footprint = occ.any(axis=1)
    inner = ndimage.binary_erosion(footprint, structure=np.ones((3, 3), bool))
    n = int(inner.sum())
    if n == 0:
        return 0.0, 0
    ys = np.arange(occ.shape[1])[None, :, None]
    top = np.where(occ, ys, -1).max(axis=1)        # topmost block y per column
    open_cols = inner & (top <= 1)
    return float(open_cols.sum() / n), n


def _inner_air(occ: np.ndarray) -> int:
    """Air cells above the floor row inside the footprint eroded by one: zero for a
    solid mass, positive for a hollow (even roofless) shell."""
    from scipy import ndimage
    footprint = occ.any(axis=1)
    inner = ndimage.binary_erosion(footprint, structure=np.ones((3, 3), bool))
    if not inner.any() or occ.shape[1] < 2:
        return 0
    cols = ~occ[:, 1:, :] & inner[:, None, :]
    return int(cols.sum())


def inspect_structure(structure: Structure, *, canvas: Optional[Sequence[int]] = None,
                      offset: Sequence[int] = (0, 0, 0)) -> Inspection:
    """Run every check on ``structure``.

    ``structure`` should be the UNCROPPED world (``world.to_structure(crop=False)``)
    so the ground check sees where the build stands; a cropped one is accepted and
    then ``offset`` is where its origin sits in the world. ``canvas`` adds the
    canvas-use note.
    """
    occ_full = np.asarray(structure.occupied_mask, bool)
    insp = Inspection()
    add = insp.findings.append
    n_blocks = int(occ_full.sum())
    if n_blocks == 0:
        add(Finding("built", False, "error", "nothing was built: zero blocks placed",
                    "start from the ground (y=0) and check the coordinate rules"))
        for c in ("grounded", "connected", "supported", "interior", "roofed", "enterable",
                  "headroom", "openings"):
            add(Finding(c, False, "warning", "no build to check", applicable=False))
        insp.stats = {"blocks": 0}
        return insp
    add(Finding("built", True, text=f"{n_blocks} blocks"))

    pts = np.argwhere(occ_full)
    lo, hi = pts.min(0), pts.max(0)
    y0 = int(lo[1] + offset[1])
    occ = occ_full[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
    bbox = tuple(int(v) for v in (hi - lo + 1))
    origin = tuple(int(a + b) for a, b in zip(lo, offset))
    stats: Dict[str, Any] = {"blocks": n_blocks, "bbox": "x".join(map(str, bbox))}

    # grounded: the lowest block is on the ground
    if y0 > 0:
        add(Finding("grounded", False, "warning",
                    f"the lowest block is at y={y0}; nothing touches the ground (y=0)",
                    "move the build down so its floor or plinth sits at y=0"))
    else:
        add(Finding("grounded", True))

    # connected: one piece
    lab, k, sizes = _components(occ)
    stats["pieces"] = k
    if k > 1:
        main = int(sizes.argmax()) + 1
        stray = int(sizes.sum() - sizes.max())
        frac = stray / n_blocks
        biggest = int(np.argsort(sizes)[-2]) + 1
        where = _bbox_str(lab == biggest, origin)
        sev = ("error" if frac > STRAY_ERROR_FRAC else
               "warning" if frac > STRAY_WARN_FRAC else "info")
        add(Finding("connected", False, sev,
                    f"{stray} block{'s' if stray != 1 else ''} ({frac:.0%}) in {k - 1} "
                    f"piece{'s' if k > 2 else ''} {'are' if stray != 1 else 'is'} "
                    f"not attached to the main structure (largest stray piece {where})",
                    "attach every part to the main structure or remove it; a roof must "
                    "rest on its walls, a chimney on the roof"))
        _ = main
    else:
        add(Finding("connected", True))

    # supported: blocks with nothing below them in their column
    unsup, above = _supported_frac(occ, y0)
    frac_unsup = unsup / max(1, above)
    stats["unsupported"] = unsup
    if frac_unsup > OVERHANG_WARN_FRAC:
        add(Finding("supported", False, "warning",
                    f"{unsup} blocks ({frac_unsup:.0%} of those above the ground) hang with "
                    "nothing beneath them and no supported neighbour beside them",
                    "eaves may overhang by one block; walls, floors and roofs must stand on "
                    "something, so extend the walls up to the roof, lower the roof, or add "
                    "posts under what sticks out"))
    else:
        add(Finding("supported", True))

    # interior: enclosed rooms, and not a solid mass
    from blockgen.agentic.recipes.staged_scoring import interior_stats
    ist = interior_stats(occ)
    fill = n_blocks / float(np.prod(bbox))
    inner_air = _inner_air(occ)
    stats["enclosed_air"] = ist["enclosed_air"]
    stats["rooms"] = ist["rooms"]
    if fill > SOLID_FILL_FRAC and inner_air == 0 and ist["enclosed_air"] < MIN_INTERIOR_CELLS:
        add(Finding("interior", False, "error",
                    f"the build is a solid mass ({fill:.0%} of its {stats['bbox']} box is "
                    "filled) with no room inside",
                    "build walls one block thick around an empty interior (hollow the "
                    "footprint, keep a floor and a roof)"))
    elif ist["enclosed_air"] < MIN_INTERIOR_CELLS:
        add(Finding("interior", False, "error",
                    f"no enclosed interior: only {ist['enclosed_air']} air cells are closed "
                    "in by walls and a roof",
                    "close the walls on all four sides and cover them with a roof; a room "
                    "needs a floor, four walls and a ceiling"))
    else:
        add(Finding("interior", True,
                    text=f"{ist['rooms']} room{'s' if ist['rooms'] != 1 else ''}, "
                         f"{ist['enclosed_air']} enclosed cells"))

    # roofed: interior floor area under open sky
    open_frac, inner_cols = _open_sky_frac(occ)
    stats["open_sky"] = f"{open_frac:.0%}"
    if inner_cols >= 4 and open_frac > OPEN_SKY_WARN_FRAC:
        add(Finding("roofed", False, "warning",
                    f"{open_frac:.0%} of the floor area inside the walls has no roof over it",
                    "put a roof over the whole footprint (fill a slab one above the wall "
                    "tops, or a gable spanning the short axis)"))
    else:
        add(Finding("roofed", True))

    # enterable / headroom / openings: the player walk and the scale heuristics
    from blockgen.agentic.scale import scale_stats
    from blockgen.eval.traversal import aperture_mask, traversal_stats
    sub = Structure(block_ids=structure.block_ids[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1,
                                                  lo[2]:hi[2] + 1],
                    block_data=structure.block_data[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1,
                                                    lo[2]:hi[2] + 1])
    ts = traversal_stats(sub)
    sc = scale_stats(sub)
    stats["doorways"] = sc.get("scale_doorways", 0)
    # Windows: glazed blocks anywhere, plus open apertures above a sill (an opening
    # whose cell below is wall, so a doorway's upper cell does not count).
    ap = aperture_mask(occ, ground=True)
    sill = np.zeros_like(ap)
    sill[:, 1:, :] = ~ap[:, :-1, :]
    glazed = np.isin(sub.block_ids, np.fromiter(GLASS_IDS, dtype=np.int64))
    windows = int((ap & sill)[:, 2:, :].sum() + glazed.sum())
    stats["window_cells"] = windows
    has_interior = ts.interior_standing > 0
    if not has_interior:
        add(Finding("enterable", False, "warning", "no interior a player could stand in",
                    applicable=False))
        add(Finding("headroom", False, "warning", "no interior", applicable=False))
    else:
        if not ts.enterable:
            add(Finding("enterable", False, "error",
                        "a player cannot get inside: no doorway reachable from the ground "
                        f"({sc.get('scale_low_openings', 0)} openings are only 1 block tall)",
                        "cut a door 1 wide and 2 tall at floor level through an outer wall "
                        "(air at y=floor and y=floor+1), facing open ground"))
        else:
            txt = f"{ts.entrances} entrance{'s' if ts.entrances != 1 else ''}"
            if ts.floors > ts.floors_reached:
                add(Finding("enterable", False, "warning",
                            f"{ts.floors - ts.floors_reached} of {ts.floors} storeys cannot "
                            "be reached on foot",
                            "add stairs or a ladder with a 1 x 2 hole in the floor above"))
            else:
                add(Finding("enterable", True, text=txt))
        hr = ts.headroom3
        if hr == hr and hr < HEADROOM_OK:       # not NaN
            add(Finding("headroom", False, "warning",
                        f"only {hr:.0%} of the interior has 3 clear blocks of ceiling",
                        "raise the ceiling: a storey is 4 to 5 blocks floor to floor"))
        else:
            add(Finding("headroom", True))
    if has_interior and windows == 0:
        add(Finding("openings", False, "info", "no window openings in the walls",
                    "cut 1 x 1 or 2 x 1 openings at y=floor+1..+2 and glaze them"))
    elif has_interior:
        add(Finding("openings", True, text=f"{windows} window cells"))
    else:
        add(Finding("openings", False, "info", "no interior", applicable=False))

    # canvas use: a note only
    if canvas is not None:
        cw, _, cd = (int(v) for v in canvas)
        if bbox[0] * bbox[2] < SMALL_FOOTPRINT_FRAC * cw * cd:
            add(Finding("size", False, "info",
                        f"the footprint is {bbox[0]} x {bbox[2]} on a {cw} x {cd} canvas",
                        "a house for a 2-block player needs rooms at least 5 x 5 outside; "
                        "use the space the request needs"))
        else:
            add(Finding("size", True))
    insp.stats = stats
    return insp


# ---------------------------------------------------------------------------
# attempts and the scorer
# ---------------------------------------------------------------------------
def inspect_attempt(att: Attempt, *, canvas: Optional[Sequence[int]] = None) -> Inspection:
    """Inspect an attempt's world (uncropped, so the ground check is meaningful) and
    leave the dict form in ``att.extra['inspection']``."""
    w = getattr(att, "world", None)
    if w is None or att.blocks == 0:
        insp = inspect_structure(Structure(block_ids=np.zeros((1, 1, 1), np.int32),
                                           block_data=np.zeros((1, 1, 1), np.int32)),
                                 canvas=canvas)
    else:
        size = canvas if canvas is not None else getattr(w, "size", None)
        insp = inspect_structure(w.to_structure(crop=False), canvas=size)
    try:
        att.extra["inspection"] = insp.to_dict()
    except Exception:  # noqa: BLE001 - a frozen test double
        pass
    return insp


@register("scorer", "inspection", needs_dream=False, needs_renders=False)
def inspection(ep: Any, att: Attempt) -> ScoreTerm:
    """Share of applicable inspection checks passed (``inspect_structure``):
    grounded, connected, supported, interior, roofed, enterable, headroom,
    openings. A diagnostic the loop may select on; not a benchmark pillar."""
    insp = inspect_attempt(att, canvas=getattr(ep, "size", None))
    failed = ", ".join(f.check for f in insp.failed) or "none"
    return ScoreTerm(insp.score, note=f"inspection: {insp.n_passed}/{len(insp.applicable)} "
                                      f"passed; failed: {failed}")


__all__ = ["Finding", "Inspection", "inspect_attempt", "inspect_structure", "inspection"]
