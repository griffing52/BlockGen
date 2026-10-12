"""Figure: where render-and-embed loses the interior.

A schematic, not a chart. Three paths leave the *same* pair of voxel builds --
a held-out real build and its `solidify` copy -- and end at the *same* unbiased
KID estimator. They differ only in phi, the map from structure to feature:

    row 1   phi = encoder o render, four exterior orthographic views   blind
    row 2   phi = encoder o render, the same renderer cut open          sees it
    row 3   phi = D4-invariant occupancy descriptor, no renderer        sees it

Row 2 exists because the first reading of row 1 against row 3 is "you replaced
a perceptual metric with a geometric one, of course the geometric one wins".
Row 2 holds the encoder and the statistic fixed and changes only which views
are rendered, which is the experiment that actually locates the loss.

Every number is read out of a recorded run at draw time; nothing is typed in.
Separations are quoted in each run's own noise floor and are therefore only
comparable within a run -- rows 1 and 3 come from one bench run at n=128, row 2
from the view-protocol ablation at n=256.

Each endpoint used to print that provenance beside its number: the KID value it
came from, the real floor it was measured against, the run and its n. Six lines
of 5.7 pt type down the right-hand edge, three times over, against three numbers
that are the actual finding. The provenance now leaves the image and goes into
the caption; `build()` prints it verbatim under CAPTION PROVENANCE so the caption
can be kept in step with the runs without re-typing anything. What stays beside
each number is the number, its unit, and the verdict.

    .venv/bin/python scripts/figs/pipeline.py

Writes outputs/figstage/pipeline.{png,pdf}.

Drawn at 4.95 x 2.40in and included in the method section at
`width=0.95\\linewidth` of the 5.5in ICLR column (376.2 pt), so the printed page
scales it by 1.056: every point size below is a floor, and the figure sets
182.4 pt of height. `_check` measures every string after layout, reports any that
leaves its column, and reports the tightest pair of strings on the page in
printed points -- "these two look close" is a measurement here, not a taste.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from armnames import display          # noqa: E402
import figstyle                        # noqa: E402
from figstyle import (BLUE, ORANGE, ORANGE_DEEP, INK, INK_2,  # noqa: E402
                      PAPER, RULE, VOID, SOLID)
OUT = ROOT / "outputs" / "figstage"

#: Rows 1 and 3: the bench run of record. One run, one n, one reference draw.
BENCH = (ROOT / "outputs"
         / "run_20260925_002631_bench_canonical-split_fixed_transparency_cull")
#: Row 2: the view-protocol ablation. A different n, so its separations are
#: never compared against BENCH's -- only against the exterior baseline measured
#: inside that same file, which the figure prints beside them.
ABLATION = ROOT / "outputs" / "run_20260925_003516_view_protocol_ablation"
#: How much of the corruption survives into the metric's *input*. Not a
#: separation, so it carries no noise-floor unit and no cross-run hazard; it is
#: here because "the metric never had the information" is the wrong reading and
#: this run measured that it did.
HOLLOW = ROOT / "outputs" / "run_20260918_203947_bench_hollow"

#: The cutaway condition drawn in row 2: the strongest of the ablation's four.
#: The figure reports the range over all four beside it.
CUT_CONDITION = "cut: floorplan slabs"
EXTERIOR_CONDITION = "dinov2b / 4 ortho (paper)"

# Colour and font come from scripts/figs/figstyle.py. Colour follows the entity:
# ORANGE the render tier as published, ORANGE_DEEP the same render tier cut open
# (a darkened ORANGE, not the old VIOLET, because row 2 *is* the render tier),
# BLUE the geometry tier. The near-white PANEL that used to sit behind rows 1 and
# 2 is gone; the family is now grouped by a hairline outline and its header, so
# nothing on the page is painted a colour that is nearly the paper.

#: Every published render uses FLAT PER-BLOCK COLOURS, never Minecraft texture
#: assets: the Usage Guidelines forbid redistributing game files, and an
#: offscreen render built from extracted PNGs is not the "screenshot of you
#: playing" their permission covers. An empty pack directory makes
#: `FaceTextureLibrary.__missing__` fall back to a per-block solid colour that
#: keeps the baked face shading. The metric itself renders textured; see the
#: caveat in outputs/figstage/pipeline.md.
_FLAT_DIR = tempfile.mkdtemp(prefix="blockgen-flat-")


def flat_textures():
    """A texture library with no texture files, so every block renders solid."""
    from blockgen.renderer.textures import load_face_textures
    return load_face_textures(pack_dir=_FLAT_DIR)


def _corpus_mask(n: int, corpus: str = "3dcraft") -> List[bool]:
    """Which test-split rows may be published as pictures, by upstream source.

    Only 3D-Craft builds are publishable as images: GrabCraft's terms name
    pictures of blueprints as a prohibited form of distribution, and text2mc's
    builds come from PlanetMinecraft, whose terms prohibit redistributing site
    data. 3D-Craft is an MIT-licensed research release.
    """
    from blockgen.eval.bench import splits as _sp
    man = json.loads((ROOT / "data" / "minecraft" / "cache"
                      / "houses_32_manifest.json").read_text())
    rows = list(_sp.load_split("houses_32").indices("test"))
    return [man["items"][rows[i]].get("corpus") == corpus for i in range(n)]


# --- numbers ----------------------------------------------------------------

def bench_numbers() -> Dict[str, float]:
    """Rows 1 and 3, from one scorecard at one n."""
    card = json.loads((BENCH / "scorecard.json").read_text())
    arms, sd = card["arms"], card["context"]["noise_floor_sd"]

    def val(arm: str, metric: str) -> float:
        return arms[arm]["realism"][metric]["value"]

    def sep(metric: str) -> float:
        return ((val("real@solidify", metric) - val("real_test", metric))
                / sd[f"realism.{metric}"])

    return {
        "n": arms["real@solidify"]["meta"]["n"],
        "n_ref": card["context"]["n_ref_used"],
        "dino_floor": val("real_test", "mv_dino_kid"),
        "dino_solid": val("real@solidify", "mv_dino_kid"),
        "dino_sep": sep("mv_dino_kid"),
        "geom_floor": val("real_test", "geom_kid"),
        "geom_solid": val("real@solidify", "geom_kid"),
        "geom_sep": sep("geom_kid"),
    }


def ablation_numbers() -> Dict[str, float]:
    """Row 2, from the ablation at its own n, with its own exterior baseline."""
    d = json.loads((ABLATION / "ablation.json").read_text())
    rows = {r["condition"]: r for r in d["rows"]}
    cuts = [r for r in d["rows"] if r["family"] == "D cutaway"]
    return {
        "n": d["meta"]["n_probe"],
        "n_ref": d["meta"]["n_ref"],
        "cut_sep": rows[CUT_CONDITION]["separation_sd"],
        "ext_sep": rows[EXTERIOR_CONDITION]["separation_sd"],
        "cut_min": min(r["separation_sd"] for r in cuts),
        "cut_max": max(r["separation_sd"] for r in cuts),
        "n_cuts": len(cuts),
        "view_key": rows[CUT_CONDITION]["view"],
    }


def pixel_residue() -> Dict[str, float]:
    """What survives the exterior render, measured on the metric's own frames."""
    p = json.loads((HOLLOW / "hollow.json").read_text())["pixels"]["real@solidify"]
    return {"mean_abs": p["mean_abs_pixel_delta"],
            "altered": 1.0 - p["identical_view_frac"],
            "n_views": p["n_views"]}


# --- the build and its renders ----------------------------------------------

def pick_build():
    """The most-hollow publishable held-out build: the probe has most to do.

    The same rule `fig1_teaser` uses, so the paper's first two figures show the
    same house and a reader can carry one picture between them.
    """
    from scipy import ndimage
    from blockgen.eval.bench import splits, probes

    split = splits.load_split("houses_32")
    test = splits.split_structures(split, "test")
    allowed = _corpus_mask(len(test))

    def interior(s) -> int:
        occ = s.crop_to_non_air().occupied_mask
        filled = ndimage.binary_fill_holes(
            occ, structure=ndimage.generate_binary_structure(3, 1))
        return int((filled & ~occ).sum())

    cells, idx = max((interior(s), i) for i, s in enumerate(test) if allowed[i])
    real = test[idx].crop_to_non_air()
    solid = probes.solidify(real)
    n_real = int(real.occupied_mask.sum())
    n_solid = int(solid.occupied_mask.sum())
    pct = 100.0 * (n_solid - n_real) / max(n_real, 1)
    print(f"  build: houses_32 test #{idx} (3dcraft), {cells} sealed air cells, "
          f"{n_real} -> {n_solid} blocks (+{pct:.0f}%)")
    return idx, real, solid, pct


def _flatten(img: np.ndarray) -> np.ndarray:
    if img.shape[-1] == 4:
        a = img[..., 3:4].astype(np.float32) / 255.0
        img = (img[..., :3] * a + 255.0 * (1 - a)).astype(np.uint8)
    return img[..., :3]


def _crop_group(imgs: Sequence[np.ndarray], pad: int = 3) -> List[np.ndarray]:
    """Crop a set of frames to ONE box, so they stay at one scale.

    Cropping each to its own content would silently resize the solidified frame
    relative to the real one and undercut the very comparison the row makes.
    """
    ink = np.zeros(imgs[0].shape[:2], dtype=bool)
    for im in imgs:
        ink |= (im < 250).any(axis=2)
    ys, xs = np.where(ink)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, ink.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, ink.shape[1])
    return [im[y0:y1, x0:x1] for im in imgs]


def render_rows(real, solid):
    """The protocols' own cameras and cuts, drawn flat at their own 224 px."""
    from blockgen.eval.bench import features as ft
    from blockgen.renderer.textured import render_structure

    tex = flat_textures()
    #: Restated from `ablate_view_protocol.conditions()` rather than imported so
    #: this script stays standalone; the resulting view key is asserted against
    #: ablation.json below, so a drift in either file fails loudly here.
    ext = [(a, e, "") for a, e in ft.DEFAULT_VIEWS]
    cut = [(45.0, 75.0, f"slab:y:{p}:1") for p in (20, 40, 60, 80)]
    key = ft.ViewConfig(views=tuple((a, e) for a, e, _ in cut),
                        cuts=tuple(c for _, _, c in cut)).key()

    def frames(spec):
        out = []
        for s in (real, solid):
            out.append([_flatten(render_structure(
                ft.apply_cut(s, c) if c else s, px=224, azim_deg=a, elev_deg=e,
                ortho=True, bg=(1.0, 1.0, 1.0), face_textures=tex))
                for a, e, c in spec])
        return out

    (ext_real, ext_solid), (cut_real, cut_solid) = frames(ext), frames(cut)
    # Per-view common crop: within a column the two arms share a scale.
    ext_pairs = [_crop_group([r, s]) for r, s in zip(ext_real, ext_solid)]
    cut_pairs = [_crop_group([r, s]) for r, s in zip(cut_real, cut_solid)]

    # A self-check on what the reader is being shown, printed rather than drawn:
    # these flat frames are NOT the textured frames the metric consumes, and the
    # figure must not be read as a measurement of the residue. The residue it
    # prints comes from `pixel_residue()`; see pipeline.md.
    def d(pairs):
        return float(np.mean([np.abs(r.astype(float) - s.astype(float)).mean()
                              for r, s in pairs]))
    print(f"  drawn frames, mean |real - solidify| (flat colours): "
          f"exterior {d(ext_pairs):.2f}/255, cutaway {d(cut_pairs):.2f}/255")
    return ext_pairs, cut_pairs, key


def cross_sections(real, solid) -> Tuple[np.ndarray, np.ndarray, int]:
    """Two slices of the actual voxel grid: the input, drawn as voxels.

    Deliberately not a render. The left column of the figure is the *structure*,
    and depicting it with an image would blur the one distinction the figure is
    built to make -- that rendering is a step, and it has a cost.
    """
    from scipy import ndimage
    occ = real.occupied_mask
    filled = ndimage.binary_fill_holes(
        occ, structure=ndimage.generate_binary_structure(3, 1))
    void = filled & ~occ
    y = int(np.argmax(void.sum(axis=(0, 2))))          # the most-hollow storey

    def panel(o: np.ndarray, v: np.ndarray) -> np.ndarray:
        img = np.empty(o.shape + (3,), dtype=float)
        img[:] = matplotlib.colors.to_rgb(PAPER)
        img[v] = matplotlib.colors.to_rgb(VOID)
        img[o] = matplotlib.colors.to_rgb(SOLID)
        return np.transpose(img, (1, 0, 2))[::-1]      # (x, z) -> image rows

    return (panel(occ[:, y, :], void[:, y, :]),
            panel(solid.occupied_mask[:, y, :], np.zeros_like(void[:, y, :])),
            y)


# ---------------------------------------------------------------------------
# layout. Everything is in INCHES on a 4.95 x 2.92in canvas and converted at the
# last moment, so a column can be checked against the text it has to hold
# without arithmetic in the head. `_check` then measures the drawn strings.
# ---------------------------------------------------------------------------

W, H = 4.95, 2.40
#: Coordinates below are inches on a notional 2.92in-tall canvas kept from the
#: first draft so the row positions did not all have to move; DY is where the
#: canvas actually starts. It was 0.25, which left 0.25in of nothing under the
#: descriptor row. Raising it to 0.50 and dropping the six provenance lines took
#: the figure from 202.9 to 182.4 printed points at no cost to any type size:
#: the print scale follows the WIDTH, so height trimmed here is height saved on
#: a page that is already at its limit.
DY = 0.50

#: Printed points per drawing inch, for the measurements `_check` reports.
PT = 72.0 * (376.2 / (72.0 * W))


def _x(a: float, b: float) -> Tuple[float, float]:
    return (a / W, b / W)


def _y(a: float, b: float) -> Tuple[float, float]:
    return ((a - DY) / H, (b - DY) / H)


#: The horizontal budget. The pictures are what a reader looks at and they were
#: the smallest things on the page: the voxel sections were 0.36in across and the
#: render thumbnails 0.22in, while the two caption boxes beside them took 0.98in
#: of prose apiece and the endpoint column took 1.14in to print three numbers and
#: nine lines of provenance. The text columns gave width back (X_OP 0.98 -> 0.86,
#: X_EN 0.72 -> 0.64, X_RES 1.14 -> 0.70) and the pictures took it: sections
#: 0.36 -> 0.48in, thumbnails 0.22 -> 0.277in. No string lost a point of size.
#: The gaps between columns are 0.13in, not the old 0.04-0.10. That is the one
#: number holding "arrows look like arrows": the head is 2.4 x 1.7 = 4.08 pt
#: long, so a connector needs 0.113in end to end before any shaft shows at all,
#: and every connector here now runs 0.124in for a shaft as long as its head.
#: The one gap that is not 0.13in is X_OP -> X_TH, at 0.07in: no connector runs
#: there, and the thumbnails belong to the protocol box beside them. Column
#: widths are not guesses -- `_check` prints what fraction of each column its
#: widest string actually uses, and the text columns sit at 91-97%.
X_IN = _x(0.03, 0.57)        # the structure, as voxels
X_OP = _x(0.75, 1.61)        # phi's first step: view protocol, or descriptor
X_TH = _x(1.68, 2.78)        # what that step produced
X_EN = _x(2.91, 3.65)        # the encoder, shared by rows 1 and 2
X_KID = _x(3.78, 4.12)       # the statistic, shared by all three
X_RES = _x(4.25, 4.93)       # the reading
BUS_X = 0.62 / W

R1 = _y(1.96, 2.66)          # exterior renders
R2 = _y(1.16, 1.86)          # cutaway renders
R3 = _y(0.56, 1.04)          # voxel descriptor
PANEL_Y = _y(1.10, 2.72)     # the two render rows are one family
HEAD_Y = _y(2.80, 0)[0]      # the header band, clear of both box tops

_CHECKS: List[Tuple[object, Tuple[float, float], str]] = []


def _c(span):
    return 0.5 * (span[0] + span[1])


def _box(fig, x, y, fc="none", ec=RULE, lw=0.7, z=1):
    fig.patches.append(FancyBboxPatch(
        (x[0], y[0]), x[1] - x[0], y[1] - y[0],
        boxstyle="round,pad=0,rounding_size=0.006",
        transform=fig.transFigure, facecolor=fc, edgecolor=ec,
        linewidth=lw, zorder=z))


#: Head geometry, in printed points: 2.4 x 1.6 long by 1.5 x 1.6 wide. Every
#: connector in this figure runs 0.124 drawing inches = 8.9 pt end to end, so
#: 3.8 pt of head leaves 5.1 pt of shaft. The figure used to run its connectors
#: at 0.04-0.10in against the same head, which is why every one of them printed
#: as a bare triangle with no line behind it.
ARROW_SCALE = 1.6


def _arrow(fig, p0, p1, color=INK_2, lw=0.7):
    fig.patches.append(FancyArrowPatch(
        p0, p1, transform=fig.transFigure, shrinkA=0, shrinkB=0,
        arrowstyle="-|>,head_length=2.4,head_width=1.5",
        mutation_scale=ARROW_SCALE,
        color=color, linewidth=lw, zorder=4))


def _line(fig, p0, p1, color=INK_2, lw=0.7):
    fig.add_artist(plt.Line2D([p0[0], p1[0]], [p0[1], p1[1]],
                              transform=fig.transFigure, color=color,
                              linewidth=lw, solid_capstyle="butt", zorder=3))


#: Nothing is set below 5.7 pt. The figure is included at 0.95\linewidth, which
#: scales it up by 1.056, so 5.7 pt is 6.02 pt on the printed page -- the floor
#: the paper's figures hold to, and one this figure has to keep now that it is
#: set in the method section rather than the appendix. The 5.2 and 5.5 pt strings
#: that used to sit here printed at 5.5 and 5.8 pt. The one exception is the
#: exponent inside the mathtext kernel, which matplotlib sets at 0.7 x its base
#: and which no size here controls.
def _text(fig, x, y, s, size=6.0, color=INK, ha="center", va="center",
          weight="normal", style="normal", rotation=0, within=None,
          linespacing=1.3):
    t = fig.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                 fontweight=weight, fontstyle=style, rotation=rotation,
                 zorder=6, linespacing=linespacing)
    _CHECKS.append((t, within, s.splitlines()[0][:32]))
    return t


def _thumb(fig, img, x, y, ec=RULE):
    ax = fig.add_axes([x[0], y[0], x[1] - x[0], y[1] - y[0]], zorder=5)
    ax.imshow(img, interpolation="antialiased")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_anchor("C")
    for s in ax.spines.values():
        s.set_linewidth(0.4)
        s.set_color(ec)


def _strip(fig, pairs, y_real, y_solid):
    """Four frames per arm, real above solidified, one column per view."""
    n, gap = len(pairs), 0.004
    w = (X_TH[1] - X_TH[0] - gap * (n - 1)) / n
    for i, (r, s) in enumerate(pairs):
        x = (X_TH[0] + i * (w + gap), X_TH[0] + i * (w + gap) + w)
        _thumb(fig, r, x, y_real)
        _thumb(fig, s, x, y_solid)


#: Two strings closer than this, measured at the size the figure prints at, read
#: as one block of text rather than two. Everything on this page clears it.
MIN_GAP_PT = 2.0


def _check(fig, show: int = 4) -> int:
    """Every string, measured after layout. An overlap is a bug, not a taste.

    Two passes. The first is the old one: does a string leave the column it was
    declared to live in. The second measures the gap between every *pair* of
    strings and lists the tightest, because the complaint this figure came back
    with was not that anything overflowed -- nothing did -- but that neighbours
    were touching. A number beats an opinion about whether they still are.
    """
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    bad, boxes = 0, []
    for artist, within, label in _CHECKS:
        bb = artist.get_window_extent(fig.canvas.get_renderer())
        (x0, y0), (x1, y1) = inv.transform(bb.get_points())
        lo, hi = within if within else (0.002, 0.998)
        if x0 < lo - 0.004 or x1 > hi + 0.004:
            bad += 1
            print(f"  ! overflows [{lo:.3f},{hi:.3f}] -> [{x0:.3f},{x1:.3f}]"
                  f"  {label!r}")
        if y0 < 0.0 or y1 > 1.0:
            bad += 1
            print(f"  ! leaves the canvas vertically: {label!r}")
        boxes.append((x0 * W, y0 * H, x1 * W, y1 * H, label))

    # Separation between two axis-aligned boxes, in printed points: 0 when they
    # overlap, the horizontal or vertical clearance when they share a band, the
    # corner distance when they share neither.
    pairs = []
    for i, (ax0, ay0, ax1, ay1, la) in enumerate(boxes):
        for bx0, by0, bx1, by1, lb in boxes[i + 1:]:
            dx = max(bx0 - ax1, ax0 - bx1, 0.0)
            dy = max(by0 - ay1, ay0 - by1, 0.0)
            pairs.append(((dx * dx + dy * dy) ** 0.5 * PT, la, lb))
    pairs.sort()
    tight = sum(1 for g, _, _ in pairs if g < MIN_GAP_PT)
    print(f"  layout check: {len(_CHECKS)} strings, {bad} overflowing, "
          f"{tight} pairs under {MIN_GAP_PT:g} pt")
    for gap, la, lb in pairs[:show]:
        flag = "!" if gap < MIN_GAP_PT else " "
        print(f"  {flag} {gap:5.2f} pt  {la!r} / {lb!r}")

    # How much of each declared column its widest string actually uses. This is
    # what says whether a text column can give more width to the pictures beside
    # it: a column running at 70% is carrying 30% of air it does not need.
    used: Dict[Tuple[float, float], Tuple[float, str]] = {}
    for (x0, _, x1, _, label), (_, within, _) in zip(boxes, _CHECKS):
        if within is None:
            continue
        w = x1 - x0
        if w > used.get(within, (0.0, ""))[0]:
            used[within] = (w, label)
    for within, (w, label) in sorted(used.items()):
        span = (within[1] - within[0]) * W
        print(f"    column {span * PT:5.1f} pt, widest string {w * PT:5.1f} pt "
              f"({100 * w / span:3.0f}%)  {label!r}")
    return bad + tight


def build() -> None:
    figstyle.apply(size=6.0)

    b, ab, px = bench_numbers(), ablation_numbers(), pixel_residue()
    idx, real, solid, mass = pick_build()   # printed, not drawn
    ext_pairs, cut_pairs, key = render_rows(real, solid)
    assert key == ab["view_key"], (key, ab["view_key"])
    sec_real, sec_solid, y_slice = cross_sections(real, solid)

    fig = plt.figure(figsize=(W, H))
    panel_x = (X_OP[0] - 0.012, X_EN[1] + 0.012)

    # --- the header band ---------------------------------------------------
    # One line across the top, clear of every box below it, so the two labels
    # that name the whole panel are not crowding its top rule.
    _box(fig, panel_x, PANEL_Y, fc="none", ec=RULE, lw=0.6, z=0)
    _text(fig, _c(X_IN), HEAD_Y, "held-out build",
          size=6.4, weight="bold", within=(0.0, X_OP[0]))
    _text(fig, panel_x[0] + 0.008, HEAD_Y,
          "$\\varphi$ = encoder $\\circ$ render",
          size=5.9, color=INK_2, ha="left", style="italic",
          within=(panel_x[0], _x(0, 2.38)[1]))
    _text(fig, panel_x[1] - 0.008, HEAD_Y,
          f"{display('real_test')} above {display('real@solidify')}",
          size=5.7, color=INK_2, ha="right",
          within=(_x(0, 2.36)[1], panel_x[1]))

    # --- input: the structure, as voxels ----------------------------------
    # The two sections are the only place the reader sees the actual input, and
    # they were the smallest pictures on the page. 0.48 x 0.43in now.
    _box(fig, X_IN, _y(1.50, 2.72))
    _thumb(fig, sec_real, _x(0.06, 0.54), _y(2.25, 2.68))
    _text(fig, _c(X_IN), _y(2.17, 0)[0], display("real_test"), size=5.7,
          color=INK_2, within=X_IN)
    _thumb(fig, sec_solid, _x(0.06, 0.54), _y(1.65, 2.08))
    _text(fig, _c(X_IN), _y(1.57, 0)[0], display("real@solidify"),
          size=5.7, color=INK_2, within=X_IN)
    _text(fig, _c(X_IN), _y(1.42, 0)[0], "peach =\nsealed air", size=5.7,
          color=INK_2, va="top", within=(0.0, X_OP[0]))

    # --- the bus: one input, three phis -----------------------------------
    _line(fig, (X_IN[1], _c((R1[0], R2[1]))), (BUS_X, _c((R1[0], R2[1]))))
    _line(fig, (BUS_X, _c(R3)), (BUS_X, _c(R1)))
    for row in (R1, R2, R3):
        _arrow(fig, (BUS_X, _c(row)), (X_OP[0] - 0.002, _c(row)))

    # --- row 1: the published protocol ------------------------------------
    # The three blocks inside this box used to sit 1.1 pt apart and read as one
    # six-line paragraph. They are the same three blocks, spaced.
    _box(fig, X_OP, R1, ec=ORANGE, lw=0.9)
    _text(fig, _c(X_OP), _y(2.555, 0)[0], "4 exterior views", size=7.0,
          weight="bold", within=X_OP)
    _text(fig, _c(X_OP), _y(2.475, 0)[0],
          "az 45/135/225/315\nel 30, ortho, 224 px", size=5.8, color=INK_2,
          va="top", within=X_OP)
    _text(fig, _c(X_OP), _y(2.21, 0)[0],
          "fill survives\n"
          f"{px['mean_abs']:.1f}/255 per frame", size=5.7, color=INK_2,
          va="top", within=X_OP)
    _strip(fig, ext_pairs, _y(2.31, 2.66), _y(1.96, 2.31))

    # --- row 2: the same everything, cut open ------------------------------
    _box(fig, X_OP, R2, ec=ORANGE_DEEP, lw=0.9)
    _text(fig, _c(X_OP), _y(1.600, 0)[0], "4 cutaway views", size=7.0,
          weight="bold", within=X_OP)
    _text(fig, _c(X_OP), _y(1.520, 0)[0],
          "four 1-voxel floor slabs", size=5.8, color=INK_2,
          va="top", within=X_OP)
    _strip(fig, cut_pairs, _y(1.51, 1.86), _y(1.16, 1.51))

    # --- row 3: no renderer at all ----------------------------------------
    _box(fig, (X_OP[0], X_EN[1]), R3, ec=BLUE, lw=0.9)
    _text(fig, X_OP[0] + 0.010, _y(0.930, 0)[0],
          "$\\varphi$ = D4-invariant occupancy descriptor", size=7.0,
          weight="bold",
          ha="left", within=(X_OP[0], X_EN[1]))
    _text(fig, X_OP[0] + 0.010, _y(0.845, 0)[0],
          "55 local-pattern orbits + 5 thickness bins\n"
          "+ 8 height bins + 9 scalars = 77-d",
          size=5.8, color=INK_2, ha="left", va="top",
          within=(X_OP[0], X_EN[1]))

    # --- the shared encoder ------------------------------------------------
    _box(fig, X_EN, (R2[0], R1[1]))
    _text(fig, _c(X_EN), _y(2.03, 0)[0], "DINOv2-B/14", size=7.0,
          weight="bold", within=X_EN)
    _text(fig, _c(X_EN), _y(1.95, 0)[0],
          "mean-pool 4 views\nL2-norm $\\to$ 768-d", size=5.8,
          color=INK_2, va="top", within=X_EN)
    for y in (_c(R1), _c(R2)):
        _arrow(fig, (X_TH[1] + 0.004, y), (X_EN[0] - 0.002, y))

    # --- the shared statistic ---------------------------------------------
    _box(fig, X_KID, (R3[0], R1[1]), ec=INK_2, lw=0.9)
    mid = _c((R3[0], R1[1]))
    _text(fig, _c(X_KID) - 0.016, mid, "unbiased KID",
          size=6.8, weight="bold", rotation=90, within=X_KID)
    _text(fig, _c(X_KID) + 0.015, mid,
          "$k\\,{=}\\,(a\\!\\cdot\\!b/D+1)^3$",
          size=5.7, color=INK_2, rotation=90, within=X_KID)
    for row in (R1, R2, R3):
        _arrow(fig, (X_EN[1] + 0.004, _c(row)), (X_KID[0] - 0.002, _c(row)))

    # --- the endpoints -----------------------------------------------------
    # Three lines each, centred on the row they belong to: the number, the unit
    # it is quoted in, the verdict. `moved` collects what used to be a fourth,
    # fifth and sixth line and is printed for the caption instead of drawn.
    moved: List[Tuple[str, str]] = []

    def endpoint(mid_in, colour, sep, verdict, detail, source):
        x = X_RES[0] + 0.004
        y = _y(mid_in, 0)[0]
        _arrow(fig, (X_KID[1] + 0.002, y), (X_RES[0] - 0.004, y))
        # Two significant figures, so the figure reads the same as the text:
        # the paper quotes 0.38 and 202, not 0.4 and 200.
        digits = 2 if abs(sep) < 1 else (1 if abs(sep) < 10 else 0)
        num = f"{sep:+.{digits}f}"
        _text(fig, x, _y(mid_in + 0.190, 0)[0], num,
              size=12.0, color=colour, ha="left", va="top", weight="bold",
              within=X_RES)
        _text(fig, x, _y(mid_in - 0.010, 0)[0], "noise floors",
              size=5.7, color=INK_2, ha="left", va="top", within=X_RES)
        _text(fig, x, _y(mid_in - 0.120, 0)[0], verdict, size=6.8, color=colour,
              ha="left", va="top", weight="bold", within=X_RES)
        for line in (detail + "\n" + source).split("\n"):
            moved.append((f"{num} {verdict}", line))

    endpoint(2.31, ORANGE, b["dino_sep"], "not separated",
             f"KID {b['dino_solid']:+.3f}\n"
             f"real floor {b['dino_floor']:+.3f}",
             f"bench run, n = {b['n']}")
    endpoint(1.51, ORANGE_DEEP, ab["cut_sep"], "separated",
             f"{ab['n_cuts']} cutaway protocols,\n"
             f"{ab['cut_min']:+.0f} to {ab['cut_max']:+.0f}",
             f"view ablation, n = {ab['n']}\n"
             f"row 1: {ab['ext_sep']:+.1f}")
    endpoint(0.80, BLUE, b["geom_sep"], "separated",
             f"KID {b['geom_solid']:+.2f}\n"
             f"real floor {b['geom_floor']:+.2f}",
             f"bench run, n = {b['n']}")

    print("  CAPTION PROVENANCE (moved out of the image, quote verbatim):")
    for owner, line in moved:
        print(f"    {owner:<20} {line}")

    bad = _check(fig)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"pipeline.{ext}", dpi=300)
    plt.close(fig)
    print(f"  wrote {OUT.relative_to(ROOT)}/pipeline.png (+ .pdf)"
          + ("" if not bad else f"  [{bad} layout warnings]"))


if __name__ == "__main__":
    build()
