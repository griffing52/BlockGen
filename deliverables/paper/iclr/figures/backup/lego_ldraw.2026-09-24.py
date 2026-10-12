"""Figure: the LEGO second domain, on CC BY models with their real colors.

    .venv/bin/python scripts/figs/lego_ldraw.py            # the figure
    .venv/bin/python scripts/figs/lego_ldraw.py --audit 200  # corpus survey
    .venv/bin/python scripts/figs/lego_ldraw.py --iou 8       # render check
    .venv/bin/python scripts/figs/lego_ldraw.py --dump        # numbers + sources

Writes `outputs/figstage/lego_ldraw.{png,pdf}`; the findings, the licence audit
and the draft caption live in `outputs/figstage/lego_ldraw.md`.

Why this figure is not made from the corpus the result currently uses
--------------------------------------------------------------------
The paper's second domain is `data/lego/stabletext2brick` (BrickGPT's
brick-ified ShapeNet objects). It carries **no color data at all**:
`blockgen/eval/bench/lego.brick_wool` assigns a wool color per brick
*footprint*, so a published picture of it would show eight colors that encode
brick dimensions and nothing else -- a reader would reasonably read them as the
model's colors. Its provenance is also ShapeNetCore, whose own terms are
narrower than the MIT label on the derived dataset card (see the .md).

`data/lego/omr` is 1,819 official LEGO models in LDraw form under Creative
Commons Attribution, every part carrying a real LEGO color code resolved
against `data/lego/ldraw/ldraw/LDConfig.ldr`. This figure is rendered from those
files, and every color in it is the color the model's author specified.

What each row is, exactly
-------------------------
1. **Exterior** -- the LDraw model rendered as-is (Blender + ImportLDraw,
   `legogen/renderer/blender_render.py`), the view an exterior render protocol
   would embed.
2. **Cut open** -- the *same parts*, with every part whose center lies past one
   plane deleted and the rest re-emitted at their original poses
   (`ldraw_voxel.placements_to_ldraw`). Nothing is re-modelled; this is the real
   model with bricks taken away, so the hollow it reveals is the model's, not a
   voxelizer's.
3. **Occupancy section** -- a slice of the derived 20x8x20 LDU plate-grid
   occupancy at *that same plane*, in the parts' real colors, with the cells a
   6-connected flood fill calls enclosed drawn in the same peach the Minecraft
   probe gallery uses. This row is the one that is a *derived* quantity, and the
   caption says so.

Honesty notes that belong with the picture
------------------------------------------
* Row 3 is bounding-box occupancy (see `legogen/renderer/ldraw_voxel`), so
  non-cuboid parts are drawn as their boxes. That over-fills rather than
  under-fills, which can only shrink the interior it reports.
* The three models are not a random sample: they were chosen among models that
  *have* an interior, because a probe that fills sealed voids has nothing to do
  on a model with none. The `--audit` survey in this file reports how common
  that is across OMR, and the .md quotes it rather than leaving the selection
  looking typical.
"""

from __future__ import annotations

import argparse
import os
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from armnames import display          # noqa: E402

from legogen.renderer.ldraw_voxel import (  # noqa: E402
    LDRAW_LIB, OMR_DIR, LDrawLibrary, PartGeometry, _world_boxes, colour_rgba,
    enclosed_air, flatten_model, model_header, parse_colours,
    placement_centres, placements_to_ldraw, stats, voxelize)

STAGE = ROOT / "outputs" / "figstage"
WORK = STAGE / "lego_ldraw_work"
BLENDER = Path(os.environ.get("BLENDER", "blender"))  # set BLENDER=/path/to/blender
RENDER_SCRIPT = ROOT / "legogen" / "renderer" / "blender_render.py"
MANIFEST = ROOT / "data" / "lego" / "omr" / "manifest.json"

# --- palette (matches scripts/paper_figures.py; copied, not imported) -------
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e4e0"
SURFACE = "#fcfcfb"
#: The interior highlight of the Minecraft probe gallery (`fig7_probe_gallery`),
#: reused so the two domains' sections read as the same picture.
INTERIOR = (0.98, 0.71, 0.55)
#: Section background: a light neutral, so white and light-gray LEGO parts stay
#: visible (a white ground would swallow them).
SECTION_BG = (0.898, 0.906, 0.918)
#: Plate height / stud pitch = 8/20: the aspect a section must be drawn at for
#: one cell to be the shape of the plate it stands for.
SECTION_ASPECT = 8.0 / 20.0

#: The three models drawn, by OMR file name. Chosen from the models that have a
#: sealed interior at all (`--audit`). An earlier version used a car as the
#: third model, to head off "your result is only about houses". That objection
#: is already answered by the corpus we actually measure, which is tables,
#: sofas, jars and vessels rather than buildings, and a car makes a poor
#: illustration of enclosed interior volume. All three are now buildings with
#: substantial sealed interiors (59%, 36% and 50% of mass).
MODELS: Tuple[str, ...] = (
    "31009-1_Skaters-House.mpd",      # set 31009, Small Cottage
    "5891-1_Small-House.mpd",         # set 5891, Apple Tree House
    "31048-1_Small-Cabin.mpd",        # set 31048, Small Cabin
)

#: Blender tile settings. 512 px is more than the figure needs at 5.5 in / 3
#: columns; the slack is there so the PDF survives being printed.
RENDER_RES = 512
RENDER_SAMPLES = 48


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8,
        "text.color": INK,
        "figure.dpi": 200,
        # TrueType, not Type 3: text stays selectable in the PDF.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# --------------------------------------------------------------------------- #
# model bookkeeping: title, author, licence, OMR page -- CC BY needs all of it
# --------------------------------------------------------------------------- #
def omr_index() -> Dict[str, Tuple[str, str]]:
    """OMR file name -> (set page id, set title), from the fetch manifest."""
    man = json.loads(MANIFEST.read_text())
    return {f: (sid, rec["title"]) for sid, rec in man.items()
            for f in rec["files"]}


def credit(name: str, index: Dict[str, Tuple[str, str]]) -> Dict[str, str]:
    """Everything the licence asks for, read out of the files themselves."""
    path = OMR_DIR / name
    head = model_header(path)
    sid, title = index.get(name, ("", name))
    short = title.replace("LDraw.org Official Model Repository - ", "")
    return {"file": name, "title": short, "author": head["author"],
            "license": head["license"],
            "url": f"https://library.ldraw.org/omr/sets/{sid}" if sid else ""}


# --------------------------------------------------------------------------- #
# the three panels of a column
# --------------------------------------------------------------------------- #
def section_axis_index(occ: np.ndarray, inter: np.ndarray,
                       axis: int = 0) -> int:
    """The slice to show: most enclosed air, restricted to the middle half.

    The restriction is not cosmetic. The cut plane of row 2 is this same plane,
    and a plane at 10% of the model's width would either remove nothing or
    remove the model, so the cutaway would stop being a cutaway. Reported by
    `--dump` so the choice is visible rather than tuned in silence.
    """
    n = occ.shape[axis]
    per = inter.sum(axis=tuple(a for a in range(3) if a != axis))
    lo, hi = max(n // 4, 0), max(n - n // 4, 1)
    window = per[lo:hi]
    if window.sum() == 0:
        return n // 2
    return int(lo + np.argmax(window))


def section_image(occ: np.ndarray, col: np.ndarray, inter: np.ndarray,
                  axis: int, idx: int,
                  table: Dict[int, Tuple[int, int, int, int]]) -> np.ndarray:
    """One slice as an RGB image: real part colors, enclosed air in peach."""
    sl = [slice(None)] * 3
    sl[axis] = idx
    o = occ[tuple(sl)]
    c = col[tuple(sl)]
    it = inter[tuple(sl)]
    img = np.empty(o.shape + (3,), dtype=float)
    img[:] = SECTION_BG
    for code in np.unique(c[o]):
        r, g, b, _a = colour_rgba(int(code), table)
        img[(c == code) & o] = (r / 255.0, g / 255.0, b / 255.0)
    img[it] = INTERIOR
    # Slicing axis 0 leaves (y, z) with y up, so rows are already height; flip
    # them so the model stands up (the equivalent of imshow's origin="lower",
    # done to the array because the panel is padded before it is drawn).
    return img[::-1]


def build_column(name: str, lib: LDrawLibrary, geom: PartGeometry,
                 table: Dict[int, Tuple[int, int, int, int]]) -> Dict[str, object]:
    """Voxelize one model and write the .ldr files its panels need."""
    src = OMR_DIR / name
    placements, report = flatten_model(src, lib, geom)
    vm = voxelize(placements, geom, source=str(src), report=report)
    inter = enclosed_air(vm.occ)
    axis = 0
    idx = section_axis_index(vm.occ, inter, axis)
    plane = vm.origin[axis] + (idx + 0.5) * vm.cell[axis]

    # Row 2's cut plane IS row 3's section plane. Along x the camera looks from
    # the +x side (verified by rendering both half-cuts), so dropping the parts
    # past the plane opens the model towards the reader.
    centers = placement_centres(placements, geom)
    keep = centers[:, axis] <= plane

    # Keep the ground plate whatever side of the cut it sits on. A baseplate is
    # one wide part whose center is usually past the plane, so the center test
    # deletes the whole thing and the cutaway loses the floor the model stands
    # on, which reads as a building sliced off its foundation rather than opened
    # up. Anything lying within one plate of the model's floor is kept.
    _lo, _hi = _world_boxes(placements, geom)
    floor = _lo[:, 1].min()
    on_ground = _hi[:, 1] <= floor + 12.0        # one plate is 8 LDU
    keep = keep | on_ground
    cut_path = WORK / f"{Path(name).stem}_cut.ldr"
    cut_path.write_text(placements_to_ldraw(placements, f"cutaway of {name}", keep))

    # Blender frames each scene to its own bounding sphere, so a tile's scale is
    # res / (2 * radius * margin). Recording both radii here is what lets the
    # figure put the exterior and the cutaway back on ONE scale (`rescale`);
    # without it the cutaway, being smaller, would be drawn larger.
    lo, hi = _world_boxes(placements, geom)
    r_full = float(np.linalg.norm(hi.max(0) - lo.min(0)) * 0.5)
    r_cut = float(np.linalg.norm(hi[keep].max(0) - lo[keep].min(0)) * 0.5)

    st = stats(vm)
    return {"name": name, "src": src, "vm": vm, "inter": inter, "axis": axis,
            "idx": idx, "plane": float(plane), "cut_path": cut_path,
            "kept_parts": int(keep.sum()), "stats": st,
            "r_full": r_full, "r_cut": r_cut,
            "section": section_image(vm.occ, vm.colour, inter, axis, idx, table)}


# --------------------------------------------------------------------------- #
# Blender
# --------------------------------------------------------------------------- #
def render_tiles(tiles: List[Dict[str, str]], res: int = RENDER_RES,
                 samples: int = RENDER_SAMPLES) -> None:
    """One batched Blender session for every tile (the kernel compiles once)."""
    WORK.mkdir(parents=True, exist_ok=True)
    job = {"ldraw_lib": str(LDRAW_LIB), "res": res, "samples": samples,
           "engine": "CYCLES", "azim": 40.0, "elev": 28.0, "margin": 1.12,
           "colour_scheme": "lgeo", "look": "normal", "tiles": tiles}
    job_path = WORK / "job.json"
    job_path.write_text(json.dumps(job, indent=1))
    log = WORK / "blender.log"
    print(f"[lego_ldraw] Blender: {len(tiles)} tiles -> {log}", flush=True)
    with open(log, "w") as fh:
        proc = subprocess.run([str(BLENDER), "--background", "--python",
                               str(RENDER_SCRIPT), "--", str(job_path)],
                              stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"Blender exit {proc.returncode}; see {log}")


def load_tile(path: Path) -> np.ndarray:
    """PNG with alpha -> RGB on the figure surface (no gray halo)."""
    img = plt.imread(path)
    if img.shape[-1] == 4:
        a = img[..., 3:4]
        bg = np.array(matplotlib.colors.to_rgb(SURFACE))
        img = img[..., :3] * a + bg * (1 - a)
    return np.clip(img[..., :3], 0, 1)


def rescale(img: np.ndarray, factor: float) -> np.ndarray:
    """Resample an RGB float image (PIL; only ever downscales here)."""
    if abs(factor - 1.0) < 1e-3:
        return img
    from PIL import Image
    h, w = img.shape[:2]
    out = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize(
        (max(int(round(w * factor)), 1), max(int(round(h * factor)), 1)),
        Image.LANCZOS)
    return np.asarray(out, dtype=float) / 255.0


def pad_to(img: np.ndarray, h: int, w: int,
           bg: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
    """Centre an image on a `h x w` canvas so every panel aligns."""
    ground = np.array(matplotlib.colors.to_rgb(SURFACE) if bg is None else bg)
    canvas = np.ones((max(h, img.shape[0]), max(w, img.shape[1]), 3)) * ground
    y = (canvas.shape[0] - img.shape[0]) // 2
    x = (canvas.shape[1] - img.shape[1]) // 2
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
    return canvas


def trim(img: np.ndarray, pad: int = 6) -> np.ndarray:
    """Crop the empty margin Blender's bbox framing leaves around a tile."""
    bg = np.array(matplotlib.colors.to_rgb(SURFACE))
    mask = np.abs(img - bg).max(axis=-1) > 0.01
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1]


# --------------------------------------------------------------------------- #
# the figure
# --------------------------------------------------------------------------- #
def make_figure(skip_render: bool = False, dump: bool = False) -> None:
    _style()
    WORK.mkdir(parents=True, exist_ok=True)
    lib = LDrawLibrary()
    geom = PartGeometry(lib)
    table = parse_colours()
    index = omr_index()

    cols = [build_column(n, lib, geom, table) for n in MODELS]
    tiles = []
    for c in cols:
        stem = Path(str(c["name"])).stem
        c["ext_png"] = WORK / f"{stem}_ext.png"
        c["cut_png"] = WORK / f"{stem}_cutaway.png"
        tiles += [{"src": str(c["src"]), "out": str(c["ext_png"])},
                  {"src": str(c["cut_path"]), "out": str(c["cut_png"])}]
    if not skip_render or not all(Path(t["out"]).exists() for t in tiles):
        render_tiles(tiles)

    creds = [credit(str(c["name"]), index) for c in cols]
    if dump:
        print("\n[lego_ldraw] every number on the figure, with its source")
        print(f"  parse + voxelization: legogen/renderer/ldraw_voxel.py "
              f"(cell 20x8x20 LDU, coverage threshold 0.5)")
        for c, cr in zip(cols, creds):
            st = c["stats"]
            print(f"  {cr['title']}  [{cr['file']}]")
            print(f"    author {cr['author']!r}; licence {cr['license']!r}")
            print(f"    OMR page {cr['url']}")
            print(f"    parts {st['parts']}, missing refs {st['missing_refs']}, "
                  f"grid {st['nx']}x{st['ny']}x{st['nz']} cells, "
                  f"{st['cells']} filled ({st['fill_frac']:.1%} of the box)")
            print(f"    enclosed air {st['enclosed_void']} cells = "
                  f"{st['void_frac_of_mass']:.1%} of filled mass")
            print(f"    section/cut plane: axis {c['axis']} index {c['idx']} "
                  f"(x = {c['plane']:.0f} LDU); cutaway keeps "
                  f"{c['kept_parts']}/{st['parts']} parts")

    # --- one physical scale for every rendered panel -----------------------
    # Each tile was framed to its own scene, so pixels-per-LDU differs between
    # tiles. Put them all on the smallest of those scales (downsample only) and
    # pad to a common canvas, so a brick is the same size in all six renders and
    # the cutaway cannot read as a different, larger model.
    ppl = {}
    for c in cols:
        ppl[(id(c), "ext")] = RENDER_RES / (2 * float(c["r_full"]) * 1.12)
        ppl[(id(c), "cut")] = RENDER_RES / (2 * float(c["r_cut"]) * 1.12)
    target = min(ppl.values())
    shots: Dict[Tuple[int, str], np.ndarray] = {}
    for c in cols:
        for key, png in (("ext", c["ext_png"]), ("cut", c["cut_png"])):
            img = trim(load_tile(Path(str(png))))
            shots[(id(c), key)] = rescale(img, target / ppl[(id(c), key)])
    # Pad per row, not globally: one canvas per row keeps the three columns
    # aligned without giving every row the tallest model's empty sky.
    canvas = {}
    for key in ("ext", "cut"):
        h = max(shots[(id(c), key)].shape[0] for c in cols)
        w = max(shots[(id(c), key)].shape[1] for c in cols)
        canvas[key] = (h, w)
        for c in cols:
            shots[(id(c), key)] = pad_to(shots[(id(c), key)], h, w)

    # Sections: pad to a common cell canvas for the same reason. One cell is
    # 20 LDU wide and 8 tall, so the section is drawn on the grid's own
    # anisotropic lattice -- a plate looks like a plate.
    secs = [np.asarray(c["section"]) for c in cols]
    sh = max(im.shape[0] for im in secs)
    sw = max(im.shape[1] for im in secs)
    secs = [pad_to(im, sh, sw, SECTION_BG) for im in secs]

    # A cell is 20 LDU wide and 8 tall, so the section must be drawn with that
    # aspect (`SECTION_ASPECT`) or the model comes out 2.5x too tall and the
    # picture lies about the proportions of a build it also shows rendered.
    # Each row's height ratio is its image's aspect, so no row letterboxes.
    ratios = [canvas["ext"][0] / canvas["ext"][1],
              canvas["cut"][0] / canvas["cut"][1],
              sh * SECTION_ASPECT / sw]
    # Margins in inches, not in fractions of the page: the title, the interior
    # key and the attribution line are all set in points, so the room they need
    # does not scale with the figure. `block` is the height the three rows of
    # panels had when this layout was tuned, kept to the thousandth of an inch
    # so the panels themselves are untouched; everything saved here comes out of
    # the margins around them.
    block = 0.76 * (1.62 * sum(ratios) + 1.15)
    top_in, bot_in = 0.28, 0.58
    fig_h = block + top_in + bot_in
    fig = plt.figure(figsize=(5.5, fig_h))
    gs = fig.add_gridspec(3, 3, height_ratios=ratios,
                          hspace=0.13, wspace=0.04,
                          left=0.075, right=0.995,
                          top=1.0 - top_in / fig_h, bottom=bot_in / fig_h)

    def Y(inches: float) -> float:
        """Inches from the bottom of the page -> figure fraction."""
        return inches / fig_h

    # One word or two per row; what each row *is* belongs in the caption, not
    # inside the picture. Shorter labels also free the left margin the three
    # stacked lines used to need.
    row_labels = ("Exterior", "Cut open", "Occupancy section")
    for r in range(3):
        for c_i, col in enumerate(cols):
            ax = fig.add_subplot(gs[r, c_i])
            if r < 2:
                ax.imshow(shots[(id(col), "ext" if r == 0 else "cut")])
                for sp in ax.spines.values():
                    sp.set_visible(False)
            else:
                ax.imshow(secs[c_i], interpolation="nearest",
                          aspect=SECTION_ASPECT)
                for sp in ax.spines.values():
                    sp.set_color(GRID)
                    sp.set_linewidth(0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                cr = creds[c_i]
                st = col["stats"]
                ax.set_title(f"{cr['title']}\n{st['parts']} parts, "
                             f"{st['nx']}×{st['ny']}×{st['nz']} cells",
                             fontsize=7.0, color=INK, pad=2.0, linespacing=1.3)
            if c_i == 0:
                ax.set_ylabel(row_labels[r], fontsize=6.4, color=INK_2,
                              labelpad=2.5, linespacing=1.25)
            if r == 2:
                st = col["stats"]
                ax.set_xlabel(f"sealed interior:\n{st['enclosed_void']} cells, "
                              f"{st['void_frac_of_mass']:.0%} of mass",
                              fontsize=6.4, color=INK_2, labelpad=2.0,
                              linespacing=1.25)

    # Key for the one color in the figure that is not a LEGO color, set just
    # under the row of section labels rather than a third of an inch below it.
    fig.patches.append(plt.Rectangle((0.078, Y(0.176)), 0.014, Y(0.128),
                                     transform=fig.transFigure,
                                     facecolor=INTERIOR, edgecolor=MUTED,
                                     linewidth=0.4))
    fig.text(0.099, Y(0.24),
             f"sealed interior, the mass {display('real@solidify')} fills",
             fontsize=6.4, color=INK_2, va="center")
    # CC BY: the licence asks for the model, its author and the repository, so
    # this line stays on the image itself and not only in the caption.
    fig.text(0.5, Y(0.06),
             "Models: " + " · ".join(f"{c['title']} ({c['author'].split(' [')[0]})"
                                     for c in creds)
             + " — LDraw.org OMR, CC BY 2.0.",
             ha="center", fontsize=5.6, color=MUTED)

    STAGE.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(STAGE / f"lego_ldraw.{ext}", dpi=300, bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)
    print(f"[lego_ldraw] wrote {STAGE.relative_to(ROOT)}/lego_ldraw.png (+ .pdf)")


# --------------------------------------------------------------------------- #
# audits: is the voxel path sane, and is the corpus usable?
# --------------------------------------------------------------------------- #
def audit(n: int, seed: int = 0) -> None:
    """Voxelize a random sample of OMR and report what the probe would find."""
    lib = LDrawLibrary()
    geom = PartGeometry(lib)
    files = sorted(OMR_DIR.glob("*.mpd"))
    rng = random.Random(seed)
    sample = files if n >= len(files) else rng.sample(files, n)
    rows, failed = [], []
    import time
    t0 = time.time()
    for p in sample:
        try:
            placements, rep = flatten_model(p, lib, geom)
            vm = voxelize(placements, geom, source=str(p), report=rep)
            st = stats(vm)
            st["name"] = p.name
            rows.append(st)
        except Exception as exc:  # noqa: BLE001
            failed.append((p.name, f"{type(exc).__name__}: {exc}"))
    dt = time.time() - t0
    print(f"[audit] {len(rows)} voxelized, {len(failed)} failed, "
          f"{dt:.1f}s ({dt / max(len(sample), 1):.2f}s/model), seed {seed}")
    for name, err in failed[:10]:
        print(f"  FAIL {name}: {err}")

    def pct(key, f=float):
        v = np.array([f(r[key]) for r in rows])
        return (f"median {np.median(v):.4g}  p10 {np.percentile(v, 10):.4g}  "
                f"p90 {np.percentile(v, 90):.4g}  max {v.max():.4g}")
    for key in ("parts", "cells", "cells_per_part", "fill_frac", "nx", "ny",
                "nz", "enclosed_void", "void_frac_of_mass"):
        print(f"  {key:18s} {pct(key)}")
    refs = sum(r["parts"] for r in rows)
    miss = sum(r["missing_refs"] for r in rows)
    empty = sum(r["empty_refs"] for r in rows)
    print(f"  unresolved part refs: {miss} of {refs + miss} "
          f"({miss / max(refs + miss, 1):.3%}); refs with no geometry: {empty}")
    for thr in (1, 8, 64):
        k = sum(1 for r in rows if r["enclosed_void"] >= thr)
        print(f"  models with >= {thr:3d} sealed interior cells: "
              f"{k}/{len(rows)} ({k / max(len(rows), 1):.0%})")
    for cap in (32, 48, 64):
        fit = sum(1 for r in rows if max(r["nx"], r["ny"], r["nz"]) <= cap)
        print(f"  models that fit a {cap}^3 grid uncropped: {fit}/{len(rows)} "
              f"({fit / max(len(rows), 1):.0%})")
    # The corpus the probe could actually be run on: damage comparable to the
    # domains already in the paper (houses +31%, StableText2Brick +10.3% median
    # mass), on a grid the bench's descriptors are sized for.
    n_total = len(sorted(OMR_DIR.glob("*.mpd")))
    for dmg in (0.05, 0.10, 0.30):
        usable = [r for r in rows if r["void_frac_of_mass"] >= dmg
                  and r["enclosed_void"] >= 8]
        fits = [r for r in usable if max(r["nx"], r["ny"], r["nz"]) <= 48]
        print(f"  solidify damage >= {dmg:.0%} of mass: {len(usable)}/{len(rows)}"
              f" ({len(usable) / max(len(rows), 1):.0%}) -> ~"
              f"{round(n_total * len(usable) / max(len(rows), 1))} of {n_total} "
              f"OMR files; of those {len(fits)} fit 48^3")
    hollow = [r for r in rows if r["enclosed_void"] >= 8]
    if hollow:
        v = np.array([r["void_frac_of_mass"] for r in hollow])
        print(f"  among the >=8-cell models, solidify would add a median "
              f"{np.median(v):.1%} mass (p90 {np.percentile(v, 90):.1%})")


def _silhouette(img: np.ndarray, factor: float) -> np.ndarray:
    """Alpha channel -> boolean silhouette, resampled to a common scale."""
    a = img[..., 3]
    if abs(factor - 1.0) > 1e-3:
        from PIL import Image
        h, w = a.shape
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
            (max(int(round(w * factor)), 1), max(int(round(h * factor)), 1)),
            Image.BILINEAR), dtype=float) / 255.0
    return a > 0.5


def _centre_on(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Put two masks on one canvas, centroids aligned (integer shift)."""
    h = max(a.shape[0], b.shape[0]) * 2
    w = max(a.shape[1], b.shape[1]) * 2
    out = []
    for m in (a, b):
        ys, xs = np.where(m)
        cy, cx = (ys.mean(), xs.mean()) if len(ys) else (m.shape[0] / 2,
                                                         m.shape[1] / 2)
        canvas = np.zeros((h, w), dtype=bool)
        y0 = int(round(h / 2 - cy))
        x0 = int(round(w / 2 - cx))
        canvas[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
        out.append(canvas)
    return out[0], out[1]


def iou_check(n: int, seed: int = 0, res: int = 320) -> None:
    """Render each model and its voxel round-trip; compare silhouettes.

    The visual check the eye does ("the grid looks like the model") made
    numeric. Both images come from the same camera code, but each is framed to
    its own scene, so the two are first put on one pixels-per-LDU scale (the
    scale law `res / (2 * radius * margin)`, verified to 0.4% against a
    controlled reframing) and aligned on their silhouette centroids. What is
    left is shape disagreement, which is the thing being measured: 1.0 would
    mean the plate-grid occupancy projects to exactly the model's silhouette.

    It cannot reach 1.0 and should not: a part's bounding box is bigger than the
    part, so wheels, slopes and rounded shells project fuller than the model
    does. The number to watch is the *floor* -- a model in the 0.5s is one whose
    parts are mostly not cuboids, and the .md names them rather than averaging
    them away.
    """
    from legogen.renderer.ldraw_voxel import to_ldraw
    lib = LDrawLibrary()
    geom = PartGeometry(lib)
    files = sorted(OMR_DIR.glob("*.mpd"))
    rng = random.Random(seed)
    WORK.mkdir(parents=True, exist_ok=True)
    picked, tiles = [], []
    while len(picked) < n and files:
        p = files.pop(rng.randrange(len(files)))
        try:
            placements, rep = flatten_model(p, lib, geom)
            vm = voxelize(placements, geom, source=str(p), report=rep)
        except Exception:  # noqa: BLE001
            continue
        if not (50 <= vm.n_parts <= 900) or int(vm.occ.sum()) > 12000:
            continue
        vpath = WORK / f"iou_{p.stem}_vox.ldr"
        vpath.write_text(to_ldraw(vm))
        a = WORK / f"iou_{p.stem}_a.png"
        b = WORK / f"iou_{p.stem}_b.png"
        lo, hi = _world_boxes(placements, geom)
        r_model = float(np.linalg.norm(hi.max(0) - lo.min(0)) * 0.5)
        tiles += [{"src": str(p), "out": str(a)}, {"src": str(vpath), "out": str(b)}]
        picked.append((p.name, a, b, vm, r_model))
    render_tiles(tiles, res=res, samples=16)
    print(f"[iou] silhouette IoU, model vs its voxel round-trip (n={len(picked)})")
    ious = []
    for name, a, b, vm, r_model in picked:
        r_vox = float(np.linalg.norm(np.asarray(vm.shape) * np.asarray(vm.cell))
                      * 0.5)
        ppl_a, ppl_b = res / (2 * r_model * 1.12), res / (2 * r_vox * 1.12)
        target = min(ppl_a, ppl_b)
        ma = _silhouette(plt.imread(a), target / ppl_a)
        mb = _silhouette(plt.imread(b), target / ppl_b)
        ma, mb = _centre_on(ma, mb)
        iou = float((ma & mb).sum() / max((ma | mb).sum(), 1))
        ious.append(iou)
        print(f"  {iou:.3f}  {name}  ({vm.n_parts} parts, "
              f"{int(vm.occ.sum())} cells)")
    if ious:
        v = np.array(ious)
        print(f"  median {np.median(v):.3f}  min {v.min():.3f}  max {v.max():.3f}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-render", action="store_true",
                    help="reuse the Blender tiles already in the work dir")
    ap.add_argument("--dump", action="store_true",
                    help="print every number the figure draws, with its source")
    ap.add_argument("--audit", type=int, default=0, metavar="N",
                    help="voxelize N random OMR models and report the survey")
    ap.add_argument("--iou", type=int, default=0, metavar="N",
                    help="render N models and their voxel round-trips, "
                         "compare silhouettes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.audit:
        audit(args.audit, args.seed)
        return 0
    if args.iou:
        iou_check(args.iou, args.seed)
        return 0
    make_figure(skip_render=args.skip_render, dump=args.dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
