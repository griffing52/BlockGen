"""LDraw model -> stud-grid voxel occupancy, with real LDraw colours.

Why this exists
---------------
The paper's second-domain replication of the interior-blindness result is
currently measured on `data/lego/stabletext2brick` (BrickGPT's brick-ified
ShapeNet objects). Two things are wrong with that as the *published* second
domain: the corpus is a ShapeNetCore derivative whose redistribution terms are
not the MIT label its dataset card carries, and it has **no colours at all**, so
`eval/bench/lego.brick_wool` invents one per brick *footprint* -- a figure made
from it shows colours that encode brick dimensions, not anything real.

`data/lego/omr` has 1,819 official LEGO models in LDraw `.mpd` form, every part
carrying a real LEGO colour code, under Creative Commons Attribution (see
`outputs/figstage/lego_ldraw.md` for the audited per-file licence breakdown).
What it does not have is an occupancy grid, which is what every metric in
`blockgen/eval` consumes.

Why bounding boxes and not meshes
---------------------------------
LDraw parts are *shells*: `parts/3001.dat` is four quads plus a subpart, with no
closed bottom (`s/3001s01.dat` has the front/back faces commented out on
purpose). Watertight-mesh voxelizers therefore leak, and a surface voxelizer
returns a hollow crust -- fatal here, because the whole experiment is about
*enclosed* voids, and a crust would report interior air where the real model has
solid plastic.

So this module ignores the mesh topology and uses the one thing LDraw states
exactly: each placed part's pose. It walks the line-type-1 records (colour, 3x3
matrix, translation, filename), resolves submodels recursively, computes each
leaf part's axis-aligned bounding box from its (transformed) vertices, and
rasterizes those boxes onto a 20 x 8 x 20 LDU stud/plate lattice -- one cell =
one 1x1 plate, three cells = one brick's height.

Three facts the rasterizer has to respect:

* **y is down.** Up is `-y`. `cell_index` flips it, so the grid's y axis points
  up like every `Structure` in `blockgen`.
* **A brick's bounding box is 28 LDU tall, not 24**: the origin sits on the top
  face of the body (`3001.dat` spans y 0..24, i.e. downward) and the studs stick
  4 LDU out the other side. A box is therefore counted into a cell only when it
  covers **more than** `thresh` (default 0.5) of that cell's volume: a brick's
  stud layer covers exactly 4/8 = 0.5 of the cell above it and is dropped, while
  a plate's 8 LDU body covers 1.0 and is kept. `test_single_parts` checks that
  a 2x4 brick comes out as exactly 4x3x2 cells.

* **0.5 is not only the stud's score, so the threshold cannot be the whole
  rule.** A classic thin baseplate (`3857.dat`, `3811.dat`, `4186.dat`) has a
  4 LDU body under its 4 LDU studs. On the model's own lattice, whose up phase
  is fixed by the bricks standing on the plate, that body straddles a cell wall
  and scores 4/8 = 0.5 in each of the two cells it touches -- and 0.5 at the
  best alignment, at every alignment. A strict `>` therefore deleted every
  baseplate in the corpus, silently, and with it the floor of the ground-floor
  room: the air below the walls ran out through the bottom of the bounding box,
  the flood fill walked in from underneath, and a whole sealed storey was
  reported as outdoors (measured on `5891-1_Small-House.mpd`: a 1,276-cell air
  component touching only the grid's bottom face). Any part thinner than half a
  cell on some axis went the same way -- bars, pins, antennas, minifig limbs,
  29,810 placements in a 120-model sample.

  So the threshold keeps its meaning and gains a floor (`_hit_mask`): **a part
  that occupies no cell at all is rescued into its best-covered cells, with any
  axis whose best coverage is below 1.0 collapsed to the lower of the (at most
  two) cells that tie.** A baseplate lands on exactly one layer, the body's.
  Nothing that already occupied a cell is touched, so a brick's stud layer is
  still dropped: the brick's body cells score 1.0, the rescue never fires, and
  the 0.5 stud cell stays empty. **A baseplate is not a stud**: a stud is the
  4 LDU that sticks *out of* a part whose body is already in the grid, while a
  baseplate's 4 LDU *is* the part. The threshold cannot tell them apart from
  coverage alone, which is why the rule is "did this part get a cell", not "is
  this 0.5 a stud".

The cost, stated plainly: a bounding box is an over-estimate for any part that
is not a cuboid (slopes, wheels, cones, minifigs), and so is a rescued cell for a
part much thinner than a plate. The over-estimate is not one-directional for the
interior probe: extra plastic can bury air that was enclosed, but it can also
*seal* air that was leaking, and the baseplate above is exactly that case (the
1,276 cells it shuts in are the difference between "no ground-floor interior" and
one). Either way the grids are "the model's part-occupancy at plate resolution",
not "the model", and a figure made from them must be captioned as such.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LDRAW_LIB = REPO / "data" / "lego" / "ldraw" / "ldraw"
OMR_DIR = REPO / "data" / "lego" / "omr" / "files"

#: One cell = one 1x1 plate: 20 LDU stud pitch in x/z, 8 LDU plate height in y.
CELL = (20.0, 8.0, 20.0)

#: Fallback lattice *phase* in LDU, in (x, up, z), used when `phase="auto"` has
#: nothing to go on. Phase matters more than it looks: LDraw part origins sit at
#: the footprint centre, so a 1x1 brick spans x = -10..10 and an even-footprint
#: part spans x = -20..20. In a *model* both are placed against one global stud
#: lattice, so every part boundary lands on the same residue (20k + 10 in x/z
#: when studs are on multiples of 20, multiples of 8 in the up direction). Snap
#: to the other phase and every part straddles two cells at 50% coverage, the
#: threshold rejects both, and the grid comes out empty -- which is how this was
#: found (the first version scored 1/12 on `test_single_parts`).
PHASE = (10.0, 0.0, 10.0)

#: Decimals kept when deduplicating a part's vertices; see `PartGeometry`.
VERT_DP = 4

#: LDraw "inherit the parent's colour" and "use the edge colour" codes.
COLOUR_INHERIT = 16
COLOUR_EDGE = 24


# --------------------------------------------------------------------------- #
# library / file resolution
# --------------------------------------------------------------------------- #
class LDrawLibrary:
    """Case-insensitive name -> path index over `parts/`, `p/` and `models/`.

    LDraw references are written in DOS style (`s\\3001s01.dat`, mixed case) and
    the library on disk is lower case, so every lookup normalises separators and
    case. The index is built once by walking the library (≈37k files, <1 s).
    """

    def __init__(self, root: Path | str = LDRAW_LIB):
        self.root = Path(root)
        self.index: Dict[str, Path] = {}
        for sub in ("parts", "p", "models"):
            base = self.root / sub
            if not base.is_dir():
                continue
            for path in base.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in (".dat", ".ldr"):
                    continue
                rel = path.relative_to(base).as_posix().lower()
                self.index.setdefault(rel, path)
                self.index.setdefault(path.name.lower(), path)

    def resolve(self, name: str) -> Optional[Path]:
        key = name.replace("\\", "/").strip().lower()
        hit = self.index.get(key)
        if hit is not None:
            return hit
        return self.index.get(key.rsplit("/", 1)[-1])


def parse_colours(path: Path | str = LDRAW_LIB / "LDConfig.ldr"
                  ) -> Dict[int, Tuple[int, int, int, int]]:
    """LDConfig.ldr -> `{code: (r, g, b, alpha)}`, the official colour table."""
    out: Dict[int, Tuple[int, int, int, int]] = {}
    pat = re.compile(r"^0\s+!COLOUR\s+(\S+)\s+CODE\s+(\d+)\s+VALUE\s+#([0-9A-Fa-f]{6})")
    for raw in Path(path).read_text(errors="ignore").splitlines():
        m = pat.match(raw.strip())
        if m is None:
            continue
        code = int(m.group(2))
        rgb = m.group(3)
        alpha = 255
        am = re.search(r"ALPHA\s+(\d+)", raw)
        if am:
            alpha = int(am.group(1))
        out[code] = (int(rgb[0:2], 16), int(rgb[2:4], 16), int(rgb[4:6], 16), alpha)
    return out


def colour_rgba(code: int, table: Dict[int, Tuple[int, int, int, int]]
                ) -> Tuple[int, int, int, int]:
    """Colour code -> RGBA, honouring LDraw direct colours (`0x2RRGGBB`)."""
    if code in table:
        return table[code]
    if 0x2000000 <= code <= 0x2FFFFFF:          # direct / "extended" colour
        return ((code >> 16) & 0xFF, (code >> 8) & 0xFF, code & 0xFF, 255)
    return (128, 128, 128, 255)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
@dataclass
class Placement:
    """One leaf part, posed in model space."""
    part: str
    colour: int
    matrix: np.ndarray          # (3, 3)
    offset: np.ndarray          # (3,)


@dataclass
class ParseReport:
    parts: int = 0
    missing: Dict[str, int] = field(default_factory=dict)
    empty: Dict[str, int] = field(default_factory=dict)
    subfiles: int = 0
    max_depth: int = 0


def _split_mpd(text: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """`.mpd` text -> (`{subfile name (lower): lines}`, ordered names).

    An `.ldr` with no `0 FILE` header comes back as a single anonymous subfile
    under the empty key, which `flatten_model` uses as the main model.
    """
    files: Dict[str, List[str]] = {}
    order: List[str] = []
    cur = ""
    files[cur] = []
    order.append(cur)
    for raw in text.splitlines():
        line = raw.strip()
        low = line.lower()
        if low.startswith("0 file "):
            cur = line[7:].strip().replace("\\", "/").lower()
            if cur not in files:
                files[cur] = []
                order.append(cur)
            continue
        if low.startswith("0 nofile"):
            continue
        files[cur].append(line)
    if not files[""]:
        files.pop("", None)
        order = [n for n in order if n != ""]
    return files, order


_NUM = r"([-+0-9.eE]+)"
TYPE1_RE = re.compile(r"^1\s+(\S+)" + r"\s+" + r"\s+".join([_NUM] * 12) + r"\s+(.+?)\s*$")


def _parse_type1(line: str) -> Optional[Tuple[int, np.ndarray, np.ndarray, str]]:
    m = TYPE1_RE.match(line)
    if m is None:
        return None
    try:
        colour = int(m.group(1), 0) if not m.group(1).isdigit() else int(m.group(1))
    except ValueError:
        return None
    try:
        v = [float(m.group(i)) for i in range(2, 14)]
    except ValueError:
        return None
    offset = np.array(v[0:3], dtype=np.float64)
    matrix = np.array(v[3:12], dtype=np.float64).reshape(3, 3)
    name = m.group(14).replace("\\", "/").strip()
    return colour, matrix, offset, name


class PartGeometry:
    """Local-frame vertex clouds for library parts, cached and deduplicated.

    Only the *positions* matter -- everything downstream is a bounding box -- so
    line/triangle/quad vertices are pooled, rounded to `VERT_DP` decimals and
    uniquified, which shrinks a stud-covered brick from thousands of vertices to
    dozens. `VERT_DP = 4`, not 0: LDraw's curve primitives live on the *unit*
    circle and are scaled by the referencing matrix, so rounding to whole LDU
    first snaps `(0.7071, 0, 0.7071)` to `(1, 0, 1)` and inflates every round
    part into its circumscribing square (it made a 2x2 round plate 3x3 cells).
    Line type 5's last two values are control points, not geometry, and are
    dropped; including them would inflate boxes around curved parts.
    """

    def __init__(self, lib: LDrawLibrary):
        self.lib = lib
        self._cache: Dict[str, np.ndarray] = {}

    def vertices(self, name: str, _depth: int = 0) -> np.ndarray:
        key = name.replace("\\", "/").lower()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if _depth > 24:
            return np.zeros((0, 3))
        path = self.lib.resolve(name)
        if path is None:
            self._cache[key] = np.zeros((0, 3))
            return self._cache[key]
        pools: List[np.ndarray] = []
        for raw in path.read_text(errors="ignore").splitlines():
            line = raw.strip()
            if not line:
                continue
            kind = line[0]
            if kind == "1":
                rec = _parse_type1(line)
                if rec is None:
                    continue
                _c, mat, off, sub = rec
                sv = self.vertices(sub, _depth + 1)
                if len(sv):
                    pools.append(sv @ mat.T + off)
            elif kind in "2345":
                toks = line.split()
                nvert = {"2": 2, "3": 3, "4": 4, "5": 2}[kind]
                need = 2 + 3 * nvert
                if len(toks) < need:
                    continue
                try:
                    vals = [float(t) for t in toks[2:need]]
                except ValueError:
                    continue
                pools.append(np.asarray(vals, dtype=np.float64).reshape(nvert, 3))
        verts = (np.concatenate(pools, axis=0) if pools else np.zeros((0, 3)))
        if len(verts):
            verts = np.unique(np.round(verts, VERT_DP), axis=0)
        self._cache[key] = verts
        return verts


def flatten_model(path: Path | str, lib: LDrawLibrary,
                  geom: PartGeometry) -> Tuple[List[Placement], ParseReport]:
    """`.mpd`/`.ldr` -> flat list of posed leaf parts.

    A reference is a *submodel* (recurse) when the name is one of the file's own
    `0 FILE` subfiles, and a *part* (stop) otherwise. That rule is what makes the
    walk terminate on a corpus where the same set can nest six deep; it also
    means an external `.ldr` reference the `.mpd` failed to inline is counted as
    a leaf and, if the library cannot resolve it, shows up in
    `report.missing` rather than silently vanishing.
    """
    path = Path(path)
    files, order = _split_mpd(path.read_text(errors="ignore"))
    report = ParseReport(subfiles=len(files))
    out: List[Placement] = []

    def walk(name: str, colour: int, mat: np.ndarray, off: np.ndarray, depth: int,
             stack: Tuple[str, ...]) -> None:
        report.max_depth = max(report.max_depth, depth)
        if depth > 32 or name in stack:
            return
        for line in files[name]:
            if not line.startswith("1 "):
                continue
            rec = _parse_type1(line)
            if rec is None:
                continue
            c, m, o, sub = rec
            eff = colour if c in (COLOUR_INHERIT, COLOUR_EDGE) else c
            world_m = mat @ m
            world_o = mat @ o + off
            key = sub.lower()
            if key in files:
                walk(key, eff, world_m, world_o, depth + 1, stack + (name,))
                continue
            if lib.resolve(sub) is None:
                report.missing[sub] = report.missing.get(sub, 0) + 1
                continue
            if len(geom.vertices(sub)) == 0:
                report.empty[sub] = report.empty.get(sub, 0) + 1
                continue
            out.append(Placement(part=sub, colour=eff, matrix=world_m, offset=world_o))
            report.parts += 1

    main = order[0]
    walk(main, COLOUR_INHERIT, np.eye(3), np.zeros(3), 0, ())
    return out, report


# --------------------------------------------------------------------------- #
# voxelization
# --------------------------------------------------------------------------- #
@dataclass
class VoxelModel:
    """Occupancy + per-cell colour code on the stud/plate lattice (y up)."""
    occ: np.ndarray             # (nx, ny, nz) bool
    colour: np.ndarray          # (nx, ny, nz) int64, LDraw colour codes, -1 = air
    cell: Tuple[float, float, float]
    origin: np.ndarray          # LDU coords (x, up, z) of cell (0,0,0)'s corner
    n_parts: int
    report: ParseReport
    source: str = ""
    #: Parts that occupied no cell under the volume test and were rescued.
    rescued: int = 0
    #: Parts whose box has zero thickness on some axis; they occupy nothing.
    degenerate: int = 0

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(int(v) for v in self.occ.shape)  # type: ignore[return-value]

    def fill_fraction(self) -> float:
        return float(self.occ.mean())


def lattice_phase(lo: np.ndarray, hi: np.ndarray,
                  cell: Tuple[float, float, float] = CELL) -> np.ndarray:
    """Per-axis lattice phase read off the parts themselves.

    The right cell walls are the planes the model's own part faces lie on, so
    take, per axis, the modal residue of every part-box face modulo the cell
    size (1 LDU bins). On a model built on one stud lattice this is unambiguous;
    ties break towards 0 because `np.argmax` takes the first maximum, which is
    the convention an isolated even-footprint part wants.
    """
    c = np.asarray(cell, dtype=float)
    out = np.zeros(3)
    for ax in range(3):
        faces = np.concatenate([lo[:, ax], hi[:, ax]])
        res = np.mod(np.round(faces).astype(np.int64), int(round(c[ax])))
        out[ax] = float(np.bincount(res, minlength=int(round(c[ax]))).argmax())
    return out


def _world_boxes(placements: Sequence[Placement], geom: PartGeometry
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Posed parts -> (lo, hi) AABBs in (x, up, z) LDU, up = -y."""
    lo = np.empty((len(placements), 3))
    hi = np.empty((len(placements), 3))
    for i, p in enumerate(placements):
        v = geom.vertices(p.part) @ p.matrix.T + p.offset
        v = np.stack([v[:, 0], -v[:, 1], v[:, 2]], axis=1)   # y down -> up
        lo[i] = v.min(axis=0)
        hi[i] = v.max(axis=0)
    return lo, hi


def _axis_keep_max(f: np.ndarray, single: bool) -> np.ndarray:
    """Cells achieving this axis's best 1-D coverage.

    `single` collapses a tie to its lowest index, which is what a box straddling
    a cell wall needs: two cells at 0.5 are one layer of plastic, not two. A tie
    at full coverage is not a straddle (every inner cell of a long part is 1.0),
    so the caller leaves those alone.
    """
    fmax = float(f.max())
    keep = f >= fmax - 1e-9
    if single:
        j = int(np.argmax(keep))
        keep = np.zeros(f.shape, dtype=bool)
        keep[j] = True
    return keep


def _hit_mask(frac: Sequence[np.ndarray], vol: np.ndarray, thresh: float
              ) -> np.ndarray:
    """Which cells of a part's box footprint the part occupies.

    The volume test first: every cell the box covers by more than `thresh`. If
    that leaves the part with nothing -- a thin baseplate, a bar, a pin -- it is
    rescued into its best-covered cells instead, with any axis whose best
    coverage is below 1.0 collapsed to the lower of the cells that tie, so a
    straddling body gets one layer and not two.

    The rescue is deliberately the narrower of the two rules considered. The
    alternative, taking the product of the per-axis coverage tests (each axis
    keeping the cells above `thresh`, falling back to its single best), fixes the
    baseplate identically but also promotes parts the volume test had *already*
    placed: measured over 120 OMR models it added 14,479 cells that the volume
    test rejects (up to 5.2% of a model's mass), because 0.6 of a cell on three
    axes passes an axis test and fails a 0.216 volume test. Rescuing only what
    would otherwise vanish cannot move a part that already occupies a cell.
    """
    hit = vol > thresh
    if hit.any():
        return hit
    keeps = [_axis_keep_max(f, single=float(f.max()) < 1.0 - 1e-9) for f in frac]
    return keeps[0][:, None, None] & keeps[1][None, :, None] & keeps[2][None, None, :]


def voxelize(placements: Sequence[Placement], geom: PartGeometry,
             cell: Tuple[float, float, float] = CELL,
             thresh: float = 0.5,
             snap: bool = True,
             crop: bool = True,
             phase: Sequence[float] | str = "auto",
             max_cells: int = 40_000_000,
             source: str = "",
             report: Optional[ParseReport] = None) -> VoxelModel:
    """Rasterize posed parts' bounding boxes onto the stud/plate lattice.

    A cell is occupied when a part's box covers **more than** `thresh` of its
    volume, except that a part covering no cell by that test is rescued into its
    best-covered cells (`_hit_mask`); see the module docstring on why 0.5 exactly
    is the stud-rejecting choice and why it needs that floor. `vm.rescued` counts
    the parts the rescue placed and `vm.degenerate` the ones with no thickness to
    place. `snap` aligns the lattice to the tallest, most common structural
    plane rather than to the model's minimum corner: models are built on
    baseplates whose bottom face sits on a multiple of 8 LDU already, but a
    hanging minifig hand or an antenna below it would otherwise shift every
    layer by a fraction of a cell and smear the layering away.
    """
    if not placements:
        raise ValueError("no placed parts")
    lo, hi = _world_boxes(placements, geom)
    c = np.asarray(cell, dtype=float)
    ph = (lattice_phase(lo, hi, cell) if phase == "auto"
          else np.asarray(phase, dtype=float))
    gmin = lo.min(axis=0)
    if snap:
        # Keep the lattice on LDraw's own 20/8/20 lattice *and* its phase, taking
        # the origin down to the next lattice plane at or below the minimum corner.
        gmin = np.floor((gmin - ph) / c) * c + ph
    gspan = hi.max(axis=0) - gmin
    dims = np.maximum(np.ceil(gspan / c).astype(int), 1)
    if int(np.prod(dims)) > max_cells:
        raise ValueError(f"grid {tuple(dims)} = {int(np.prod(dims)):,} cells "
                         f"exceeds max_cells={max_cells:,}")
    occ = np.zeros(tuple(dims), dtype=bool)
    best = np.zeros(tuple(dims), dtype=np.float32)   # overlap of the winning part
    # int64, not int32: LDraw colour codes are nominally small, but the corpus
    # has models written by editors that emit 0xFFFFFFFF ("no colour set") as a
    # literal code -- `10220-1.mpd` puts it on six `97122.dat` -- and an int32
    # array raises OverflowError on the assignment rather than rendering them.
    # `colour_rgba` already has a documented grey fallback for a code it cannot
    # resolve, so the value is carried through rather than reinterpreted here.
    col = np.full(tuple(dims), -1, dtype=np.int64)
    rescued = 0
    degenerate = 0

    glo = (lo - gmin) / c
    ghi = (hi - gmin) / c
    for i, p in enumerate(placements):
        a, b = glo[i], ghi[i]
        i0 = np.maximum(np.floor(a).astype(int), 0)
        i1 = np.minimum(np.ceil(b).astype(int), dims)
        if np.any(i1 <= i0):
            continue
        frac = []
        for ax in range(3):
            edges = np.arange(i0[ax], i1[ax] + 1, dtype=float)
            cov = np.minimum(edges[1:], b[ax]) - np.maximum(edges[:-1], a[ax])
            frac.append(np.clip(cov, 0.0, 1.0))
        vol = frac[0][:, None, None] * frac[1][None, :, None] * frac[2][None, None, :]
        sl = np.s_[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]]
        if vol.max() <= 0.0:
            # A box with no thickness on some axis (a decal, a pattern-only
            # subpart) has no cell to claim; counted, not rasterized.
            degenerate += 1
            continue
        hitmask = _hit_mask(frac, vol, thresh)
        if not hitmask.any():
            continue
        if not (vol > thresh).any():
            rescued += 1
        occ[sl] |= hitmask
        # Per-cell colour goes to whichever part covers most of the cell, so a
        # thin decorative part never repaints a cell a brick fills.
        win = hitmask & (vol > best[sl])
        sub_best = best[sl]
        sub_col = col[sl]
        sub_best[win] = vol[win].astype(np.float32)
        sub_col[win] = p.colour
        best[sl] = sub_best
        col[sl] = sub_col
    col[~occ] = -1
    if crop and occ.any():
        # Trim the empty margin the snap-and-ceil bounds leave behind (a brick's
        # rejected stud layer is one such margin), so `shape` is the model's own
        # extent in studs/plates and is comparable with a catalogue dimension.
        idx = np.argwhere(occ)
        p0, p1 = idx.min(axis=0), idx.max(axis=0) + 1
        sl = np.s_[p0[0]:p1[0], p0[1]:p1[1], p0[2]:p1[2]]
        occ, col = occ[sl].copy(), col[sl].copy()
        gmin = gmin + p0 * c
    return VoxelModel(occ=occ, colour=col, cell=tuple(cell), origin=gmin,
                      n_parts=len(placements), report=report or ParseReport(),
                      source=source, rescued=rescued, degenerate=degenerate)


def voxelize_file(path: Path | str, lib: Optional[LDrawLibrary] = None,
                  geom: Optional[PartGeometry] = None, **kw) -> VoxelModel:
    """Convenience: `.mpd` path -> `VoxelModel` (builds a library if needed)."""
    lib = lib or LDrawLibrary()
    geom = geom or PartGeometry(lib)
    placements, report = flatten_model(path, lib, geom)
    return voxelize(placements, geom, source=str(path), report=report, **kw)


# --------------------------------------------------------------------------- #
# occupancy diagnostics (the numbers the probe cares about)
# --------------------------------------------------------------------------- #
def enclosed_air(occ: np.ndarray) -> np.ndarray:
    """Air cells not 6-connected to the outside: what `solidify` would fill.

    Same definition as `blockgen.curation.houses.enclosed_air_count`, restated
    on a bare array so this module stays importable without the blockgen deps.
    """
    from collections import deque
    air = ~occ
    out = np.zeros_like(air)
    nx, ny, nz = air.shape
    q: deque = deque()
    for idx in np.argwhere(air):
        x, y, z = idx
        if x in (0, nx - 1) or y in (0, ny - 1) or z in (0, nz - 1):
            if not out[x, y, z]:
                out[x, y, z] = True
                q.append((x, y, z))
    while q:
        x, y, z = q.popleft()
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                           (0, 0, 1), (0, 0, -1)):
            a, b, c = x + dx, y + dy, z + dz
            if 0 <= a < nx and 0 <= b < ny and 0 <= c < nz and air[a, b, c] \
                    and not out[a, b, c]:
                out[a, b, c] = True
                q.append((a, b, c))
    return air & ~out


def stats(vm: VoxelModel) -> Dict[str, float]:
    occ = vm.occ
    void = enclosed_air(occ)
    n_occ = int(occ.sum())
    return {
        "nx": vm.shape[0], "ny": vm.shape[1], "nz": vm.shape[2],
        "parts": vm.n_parts,
        "cells": n_occ,
        "cells_per_part": n_occ / max(vm.n_parts, 1),
        "fill_frac": float(occ.mean()),
        "enclosed_void": int(void.sum()),
        "void_frac_of_mass": float(void.sum() / max(n_occ, 1)),
        "missing_refs": int(sum(vm.report.missing.values())),
        "empty_refs": int(sum(vm.report.empty.values())),
    }


# --------------------------------------------------------------------------- #
# round-trip back to LDraw (the honest check)
# --------------------------------------------------------------------------- #
#: 1x1 plate: footprint 20x20 LDU, body 8 LDU tall -- exactly one lattice cell.
PLATE_1X1 = "3024.dat"


def to_ldraw(vm: VoxelModel, part: str = PLATE_1X1, surface_only: bool = False,
             cut: Optional[Sequence[Tuple[int, int]]] = None) -> str:
    """Emit the occupancy grid as an LDraw model of 1x1 plates, real colours.

    This is the verification path: the resulting `.ldr` goes through the *same*
    `blender_render.py` camera as the original `.mpd`, so a side-by-side is a
    like-for-like comparison of the parse against the model rather than against
    a second renderer's idea of framing. `cut` is a list of `(axis, index)`
    pairs and removes the cells that are at or past `index` on **every** listed
    axis, so one pair is a half cut and two are the corner quadrant the figure
    uses (cutting the quadrant nearest the camera opens the interior without
    throwing away half the silhouette).
    """
    occ = vm.occ
    keep = occ.copy()
    if cut:
        drop = np.ones_like(occ)
        for axis, idx in cut:
            sl = [slice(None)] * 3
            sl[axis] = slice(idx, None)
            m = np.zeros_like(occ)
            m[tuple(sl)] = True
            drop &= m
        keep &= ~drop
    if surface_only:
        pad = np.pad(occ, 1, constant_values=False)
        exposed = np.zeros_like(occ)
        for ax in range(3):
            for d in (1, -1):
                exposed |= ~np.roll(pad, d, axis=ax)[1:-1, 1:-1, 1:-1]
        keep &= exposed
    cx, cy, cz = vm.cell
    lines = ["0 Voxelized " + Path(vm.source).name,
             "0 Name: voxel.ldr",
             "0 Author: blockgen legogen/renderer/ldraw_voxel.py",
             "0 !LDRAW_ORG Unofficial_Model", ""]
    for x, y, z in np.argwhere(keep):
        # Cell centre in LDraw coords; a 1x1 plate's origin is its top face, and
        # up is -y, so the top of cell y sits at -( (y+1) * cy ).
        lx = (x + 0.5) * cx
        ly = -((y + 1) * cy)
        lz = (z + 0.5) * cz
        code = int(vm.colour[x, y, z])
        lines.append(f"1 {code} {lx:.1f} {ly:.1f} {lz:.1f} 1 0 0 0 1 0 0 0 1 {part}")
    lines.append("0")
    return "\n".join(lines)


def placements_to_ldraw(placements: Sequence[Placement], title: str = "cutaway",
                        keep: Optional[Sequence[bool]] = None) -> str:
    """Posed parts -> a flat `.ldr` of the same parts, optionally filtered.

    This is how the cutaway panels are made: drop the placements in the quadrant
    nearest the camera and re-emit the rest at their absolute poses. Nothing is
    re-modelled -- every surviving line carries the original part, colour and
    matrix -- so the panel is the real model with bricks removed, and a parsing
    error would show up immediately as scrambled geometry.
    """
    lines = [f"0 {title}", "0 Name: cutaway.ldr",
             "0 Author: blockgen legogen/renderer/ldraw_voxel.py",
             "0 !LDRAW_ORG Unofficial_Model", ""]
    for i, p in enumerate(placements):
        if keep is not None and not keep[i]:
            continue
        m = p.matrix.reshape(-1)
        o = p.offset
        lines.append("1 %d %g %g %g %s %s" % (
            p.colour, o[0], o[1], o[2],
            " ".join(f"{v:g}" for v in m), p.part))
    lines.append("0")
    return "\n".join(lines)


def placement_centres(placements: Sequence[Placement], geom: PartGeometry
                      ) -> np.ndarray:
    """(N, 3) bounding-box centres in (x, up, z) LDU -- the cutaway's cut test."""
    lo, hi = _world_boxes(placements, geom)
    return (lo + hi) * 0.5


def model_header(path: Path | str) -> Dict[str, str]:
    """Title / author / licence of an LDraw model's main file.

    Every OMR file carries these in its header, and CC BY attribution needs all
    three, so the figure script reads them from the file rather than from a note
    someone wrote down once.
    """
    out = {"title": "", "author": "", "license": ""}
    for raw in Path(path).read_text(errors="ignore").splitlines():
        line = raw.strip()
        low = line.lower()
        if low.startswith("0 file "):
            continue
        if not out["title"] and low.startswith("0 ") and not low.startswith("0 !") \
                and not low.startswith("0 name:") and not low.startswith("0 author:"):
            out["title"] = line[2:].strip()
        elif low.startswith("0 author:"):
            out["author"] = out["author"] or line.split(":", 1)[1].strip()
        elif low.startswith("0 !license"):
            out["license"] = out["license"] or line.split(None, 2)[-1].strip()
        if line.startswith("1 ") or line.startswith("4 "):
            break
    return out


# --------------------------------------------------------------------------- #
# self-test: single parts with known dimensions
# --------------------------------------------------------------------------- #
#: part -> expected (x, y, z) cell extent. Studs must NOT add a layer.
KNOWN_PARTS: Dict[str, Tuple[int, int, int]] = {
    "3005.dat": (1, 3, 1),      # brick 1x1
    "3004.dat": (2, 3, 1),      # brick 1x2
    "3001.dat": (4, 3, 2),      # brick 2x4
    "3024.dat": (1, 1, 1),      # plate 1x1
    "3020.dat": (4, 1, 2),      # plate 2x4
    "3068b.dat": (2, 1, 2),     # tile 2x2
    "3811.dat": (32, 1, 32),    # baseplate 32x32
    "3009.dat": (6, 3, 1),      # brick 1x6
    "3958.dat": (6, 1, 6),      # plate 6x6
    "3245c.dat": (2, 6, 1),     # brick 1x2x2 (the x2 is 2 brick-heights = 6 plates)
    "4032a.dat": (2, 1, 2),     # plate 2x2 round with axlehole (curve primitives)
    "3062b.dat": (1, 3, 1),     # brick 1x1 round
}


def test_single_parts(lib: Optional[LDrawLibrary] = None,
                      geom: Optional[PartGeometry] = None,
                      verbose: bool = True) -> Tuple[int, int]:
    """Voxelize one part at the origin and compare to its catalogue size."""
    lib = lib or LDrawLibrary()
    geom = geom or PartGeometry(lib)
    ok = 0
    for part, want in KNOWN_PARTS.items():
        pl = [Placement(part=part, colour=4, matrix=np.eye(3), offset=np.zeros(3))]
        vm = voxelize(pl, geom, source=part)
        got = vm.shape
        good = got == want
        ok += good
        if verbose:
            flag = "ok  " if good else "FAIL"
            print(f"  [{flag}] {part:>10}  got {got}  want {want}  "
                  f"filled {int(vm.occ.sum())}/{vm.occ.size}")
    return ok, len(KNOWN_PARTS)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m legogen.renderer.ldraw_voxel",
        description="LDraw -> stud-grid occupancy; --self-test or voxelize models.")
    ap.add_argument("models", nargs="*", help=".mpd/.ldr paths (or OMR file names)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--ldraw-out", default="", help="write the round-trip .ldr here")
    args = ap.parse_args(argv)

    lib = LDrawLibrary()
    geom = PartGeometry(lib)
    if args.self_test:
        ok, n = test_single_parts(lib, geom)
        print(f"[ldraw_voxel] single-part self-test {ok}/{n}")
        if not args.models:
            return 0 if ok == n else 1
    for name in args.models:
        path = Path(name)
        if not path.exists():
            path = OMR_DIR / name
        vm = voxelize_file(path, lib, geom, thresh=args.thresh)
        st = stats(vm)
        print(f"[{path.name}] {st['nx']}x{st['ny']}x{st['nz']} cells, "
              f"{st['parts']} parts -> {st['cells']} filled "
              f"({st['fill_frac']:.1%}), enclosed void {st['enclosed_void']} "
              f"({st['void_frac_of_mass']:.1%} of mass), "
              f"missing refs {st['missing_refs']}")
        if args.ldraw_out:
            Path(args.ldraw_out).write_text(to_ldraw(vm))
            print(f"  -> {args.ldraw_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
