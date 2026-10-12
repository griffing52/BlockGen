"""What the V-KID descriptor actually is, drawn from the implementation.

Three panels, all computed at run time from `blockgen/eval/bench/geometry.py` and
the pinned `houses_32` split -- nothing is transcribed:

 1. a 2x2x2 occupancy window pulled out of a real 3D-Craft build;
 2. its D4 orbit -- the eight images under the group the code quotients over --
    collapsing onto the single canonical code that indexes the histogram bin;
 3. the 77-dimensional descriptor, block by block, with the mean shift the
    `solidify` probe produces in each component.

Run:

    .venv/bin/python scripts/figs/descriptor.py

Writes `outputs/figstage/descriptor.{png,pdf}`.

Isometric drawing is hand-rolled rather than `mplot3d`: the same helper then
draws a 2481-block build and a single 2x2x2 window in one projection, the
highlight in panel 1 is guaranteed to sit on the cells it names (no projection
maths to get wrong), and the output stays vector and crisp at print size.
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
from matplotlib.collections import PolyCollection

#: Head- and footroom around the bars, in reference SD. The headroom carries
#: the block ruler and the callouts; the footroom is now only breathing space,
#: which is why it is small. The figure's inches-per-SD is unchanged: see
#: `_bar_unit`.
PAD_HI, PAD_LO = 3.9, 1.0
#: The axes height the original 5.05in canvas gave the bars, in inches.
_AX_D_IN = 0.195 * 5.05

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle                      # noqa: E402
from figstyle import BLUE, INK, INK_2, PAPER, RULE  # noqa: E402
from armnames import display          # noqa: E402
OUTDIR = ROOT / "outputs" / "figstage"
BENCH_RUN = (ROOT / "outputs"
             / "run_20260925_002631_bench_canonical-split_fixed_transparency_cull")

# Colour and font come from scripts/figs/figstyle.py; nothing is declared here.
# This figure draws one metric tier only -- the geometry tier -- so BLUE is its
# single data colour and the render tier's ORANGE does not appear. The local
# constants it used to carry mapped on: the near-white page fill is gone, the two
# light greys became INK_2 for anything read as text and figstyle.RULE for
# anything drawn as structure, and the gridline colour was never used.

#: Flat per-block colours, never Minecraft texture files: an empty pack dir makes
#: `FaceTextureLibrary.__missing__` fall back to a solid colour per block.
_FLAT_DIR = tempfile.mkdtemp(prefix="blockgen-flat-")


def flat_textures():
    from blockgen.renderer.textures import load_face_textures
    return load_face_textures(pack_dir=_FLAT_DIR)


def _style() -> None:
    """Sizes are this figure's; colour, font and the PDF settings are shared."""
    figstyle.apply(size=8, **{
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })


# ---------------------------------------------------------------------------
# isometric cube drawing
# ---------------------------------------------------------------------------
_SX = float(np.cos(np.pi / 6))      # 0.866


def _iso(x: float, y: float, z: float) -> Tuple[float, float]:
    """Lattice point -> screen point. Viewer sits at (-x, +y, -z)."""
    return ((x - z) * _SX, (x + z) * 0.5 + y)


def _faces(x: int, y: int, z: int):
    """The three faces of cell (x,y,z) that face the viewer, as screen polygons.

    `top` = the y+1 face, `left` = the x face (toward -x), `right` = the z face
    (toward -z). Returned with the neighbour cell that would hide each, so a
    caller can cull faces buried inside the solid.
    """
    top = [_iso(x, y + 1, z), _iso(x + 1, y + 1, z),
           _iso(x + 1, y + 1, z + 1), _iso(x, y + 1, z + 1)]
    left = [_iso(x, y, z), _iso(x, y, z + 1),
            _iso(x, y + 1, z + 1), _iso(x, y + 1, z)]
    right = [_iso(x, y, z), _iso(x + 1, y, z),
             _iso(x + 1, y + 1, z), _iso(x, y + 1, z)]
    return (("top", top, (x, y + 1, z), 1.00),
            ("left", left, (x - 1, y, z), 0.78),
            ("right", right, (x, y, z - 1), 0.60))


def _shade(rgb: Sequence[float], f: float) -> Tuple[float, float, float]:
    return tuple(float(np.clip(c * f, 0, 1)) for c in rgb[:3])


def draw_voxels(ax, occ: np.ndarray, colors: Dict[Tuple[int, int, int], tuple],
                *, offset: Tuple[float, float] = (0.0, 0.0),
                edge: str | None = None, lw: float = 0.12,
                ghost: bool = False) -> np.ndarray:
    """Painter's-algorithm isometric voxel drawing; returns the drawn bbox.

    Depth along the view axis is `x + z - y` (the projection is constant along
    (1,-1,1)), so far cells are drawn first. Interior faces are culled against
    the occupied neighbour that would hide them. `PolyCollection` does not grow
    the axes data limits reliably, so the bbox comes back for the caller to set
    limits by hand.
    """
    occ = np.asarray(occ, dtype=bool)
    polys: List[Tuple[float, List[Tuple[float, float]], tuple]] = []
    ghosts: List[List[Tuple[float, float]]] = []
    pts: List[Tuple[float, float]] = []
    for x, y, z in np.argwhere(occ if not ghost else np.ones_like(occ)):
        depth = x + z - y
        filled = bool(occ[x, y, z])
        base = colors.get((x, y, z), (0.6, 0.6, 0.6))
        for _, poly, nb, f in _faces(x, y, z):
            inb = all(0 <= n < d for n, d in zip(nb, occ.shape))
            if filled and inb and occ[tuple(nb)]:
                continue                     # buried face
            p = [(px + offset[0], py + offset[1]) for px, py in poly]
            if filled:
                polys.append((depth, p, _shade(base, f)))
                pts.extend(p)
            elif ghost:
                ghosts.append(p)
                pts.extend(p)
    if ghosts:
        ax.add_collection(PolyCollection(ghosts, facecolors="none",
                                         edgecolors=RULE, linewidths=0.35,
                                         linestyle=(0, (1.4, 1.2)), zorder=1))
    polys.sort(key=lambda t: -t[0])
    ax.add_collection(PolyCollection(
        [p for _, p, _ in polys], facecolors=[c for _, _, c in polys],
        edgecolors=(edge or "none"), linewidths=lw, zorder=2))
    a = np.asarray(pts, dtype=float)
    return np.array([a[:, 0].min(), a[:, 0].max(), a[:, 1].min(), a[:, 1].max()])


def _fit(ax, bbox, padx: float = 0.4, pady: float = 0.4) -> None:
    ax.set_xlim(bbox[0] - padx, bbox[1] + padx)
    ax.set_ylim(bbox[2] - pady, bbox[3] + pady)


def drawn_box(fig, ax) -> Tuple[float, float, float, float]:
    """Figure-coordinate bbox of the data region of an equal-aspect axes.

    With `set_aspect("equal")` matplotlib letterboxes the data inside the axes
    box, so the axes rectangle is not where the drawing is. Downstream labels
    are positioned against this instead of against a guess.
    """
    box = ax.get_position()
    w_in, h_in = box.width * fig.get_figwidth(), box.height * fig.get_figheight()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    sc = min(w_in / (x1 - x0), h_in / (y1 - y0))
    dw, dh = (x1 - x0) * sc / fig.get_figwidth(), (y1 - y0) * sc / fig.get_figheight()
    return (box.x0 + (box.width - dw) / 2, box.y0 + (box.height - dh) / 2,
            box.x0 + (box.width + dw) / 2, box.y0 + (box.height + dh) / 2)


def _hex(c: str) -> Tuple[float, float, float]:
    return matplotlib.colors.to_rgb(c)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _encode(w: np.ndarray) -> int:
    """`geometry.pattern_histogram`'s bit order: bit index 4*dx + 2*dy + dz."""
    return int(sum(int(w[dx, dy, dz]) << (4 * dx + 2 * dy + dz)
                   for dx in range(2) for dy in range(2) for dz in range(2)))


def _group_images(w: np.ndarray) -> List[Tuple[str, np.ndarray, int]]:
    """The eight D4 images of `w`, in the order `_d4_orbit_table` enumerates.

    Exactly that code's transforms: `np.rot90(., k, axes=(x, z))` about the
    vertical axis, optionally after the x-mirror `w[::-1]`.
    """
    out = []
    for mirror in (False, True):
        base = w[::-1] if mirror else w
        for k in range(4):
            g = np.rot90(base, k, axes=(0, 2))
            name = ("m" if mirror else "") + (f"r{k}" if k else "")
            out.append((name or "e", g, _encode(g)))
    return out


def gather() -> dict:
    """Everything the figure draws, computed from the code and the split."""
    from blockgen.eval.bench import splits, probes, geometry as G

    split = splits.load_split("houses_32")
    val = splits.split_structures(split, "val")
    test = splits.split_structures(split, "test")

    # Licensing gate: only 3D-Craft builds may be shown as a picture.
    man = json.loads((ROOT / "data" / "minecraft" / "cache"
                      / "houses_32_manifest.json").read_text())
    rows = list(split.indices("test"))
    allowed = [man["items"][rows[i]].get("corpus") == "3dcraft"
               for i in range(len(test))]

    # Selection rule, stated and unchanged: the publishable test build with the
    # most sealed interior volume -- the same rule fig1 uses, so the reader sees
    # the same house in both figures.
    def interior_cells(s) -> int:
        from scipy import ndimage
        occ = s.crop_to_non_air().occupied_mask
        filled = ndimage.binary_fill_holes(
            occ, structure=ndimage.generate_binary_structure(3, 1))
        return int((filled & ~occ).sum())

    scored = sorted(((interior_cells(s), i) for i, s in enumerate(test)
                     if allowed[i]), reverse=True)
    bidx = scored[0][1]
    build = test[bidx].crop_to_non_air()
    occ = build.occupied_mask

    # The example window: the most frequent full-orbit (|orbit| = 8) pattern in
    # the publishable pool, so the orbit panel shows eight *distinct* pictures
    # rather than repeats. Chosen by frequency, not by appearance.
    sizes = np.bincount(G.ORBIT_OF)
    pooled = np.zeros(G.N_ORBITS)
    for i, s in enumerate(test):
        if allowed[i]:
            pooled += G.pattern_histogram(s.crop_to_non_air().occupied_mask)
    pooled /= pooled.sum()
    orbit = int(max((o for o in range(G.N_ORBITS) if sizes[o] == 8),
                    key=lambda o: pooled[o]))

    # An occurrence of that orbit inside this build, fully in bounds, whose
    # cells are actually visible in the isometric view (otherwise the highlight
    # is buried and panel 1 shows nothing), and whose raw code differs from the
    # canonical one so the orbit panel shows a real canonicalization step.
    canon = int(min(c for c in range(256) if G.ORBIT_OF[c] == orbit))

    def visible(x: int, y: int, z: int) -> bool:
        # The projection is constant along (1,-1,1), so the camera lies along
        # (-1,+1,-1). A cube exactly on that ray covers this one completely;
        # cubes one step off it cover part of the hexagon, and several can add
        # up to a full occlusion. Requiring the whole 2x2 cone toward the camera
        # to be empty is therefore sufficient, not merely necessary -- a cell
        # that passes is certainly drawn.
        for t in range(1, max(occ.shape)):
            xs = [x - t, x - t + 1]
            zs = [z - t, z - t + 1]
            if y + t >= occ.shape[1]:
                return True
            gone = True
            for xx in xs:
                for zz in zs:
                    if not (0 <= xx < occ.shape[0] and 0 <= zz < occ.shape[2]):
                        continue
                    gone = False
                    if occ[xx, y + t, zz]:
                        return False
            if gone:
                return True
        return True

    cands = []
    for x in range(occ.shape[0] - 1):
        for y in range(occ.shape[1] - 1):
            for z in range(occ.shape[2] - 1):
                w = occ[x:x + 2, y:y + 2, z:z + 2]
                c = _encode(w)
                if G.ORBIT_OF[c] != orbit or c == canon:
                    continue
                seen = sum(visible(x + dx, y + dy, z + dz)
                           for dx, dy, dz in np.argwhere(w))
                cands.append((seen, x, y, z, c))
    best = max(s_ for s_, *_ in cands)
    cands = [t[1:] for t in cands if t[0] == best]
    mid = np.array([occ.shape[0] / 2, occ.shape[1] / 2, occ.shape[2] / 2])
    x0, y0, z0, code = min(cands, key=lambda t: np.hypot(*(np.array(t[:3]) - mid)[[0, 2]]))
    window = occ[x0:x0 + 2, y0:y0 + 2, z0:z0 + 2].copy()

    images = _group_images(window)
    assert min(c for _, _, c in images) == canon
    assert int(G.ORBIT_OF[code]) == orbit

    # --- the descriptor, and what `solidify` moves in it -------------------
    std = G.Standardizer.fit(G.descriptor(val))         # fitted on the reference
    real = [s.crop_to_non_air() for s in test]
    sol = [probes.solidify(s) for s in real]
    d_real, d_sol = G.descriptor(real), G.descriptor(sol)

    def whiten(x):                                     # the pipeline's stage 1
        return np.clip((x - std.mean) / std.scale, -std.clip, std.clip)

    shift = whiten(d_sol).mean(0) - whiten(d_real).mean(0)
    solid_ratio = float(d_sol[:, 54].mean() / d_real[:, 54].mean())
    raw_sd = ((d_sol.mean(0) - d_real.mean(0))
              / np.where(np.isfinite(std.scale), std.scale, np.inf))

    # Block colours for the build, from the flat (texture-free) library.
    lib = flat_textures()
    colors: Dict[Tuple[int, int, int], tuple] = {}
    cache: Dict[Tuple[int, int], tuple] = {}
    ids, dat = build.block_ids, build.block_data
    for x, y, z in np.argwhere(occ):
        key = (int(ids[x, y, z]), int(dat[x, y, z]))
        if key not in cache:
            tile = np.asarray(lib[key]["side"], dtype=float)
            cache[key] = tuple(tile.reshape(-1, tile.shape[-1]).mean(0)[:3] / 255.0)
        colors[(x, y, z)] = cache[key]

    card = json.loads((BENCH_RUN / "scorecard.json").read_text())

    return dict(build=build, occ=occ, colors=colors, build_index=bidx,
                n_blocks=int(occ.sum()), window=window, wpos=(x0, y0, z0),
                code=code, canon=canon, orbit=orbit, images=images,
                orbit_freq=float(pooled[orbit]), n_orbits=int(G.N_ORBITS),
                layout=G.descriptor_layout(), shift=shift, raw_sd=raw_sd,
                solid_ratio=solid_ratio,
                metrics=list(G.GEOMETRY_METRICS),
                thickness_bins=list(G.THICKNESS_BINS), card=card,
                n_val=len(val), n_test=len(test))


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------
def _panel_title(fig, x, y, n, text):
    fig.text(x, y, n, fontsize=9, weight="bold", color=INK, va="baseline")
    fig.text(x + 0.020, y, text, fontsize=8.5, color=INK, va="baseline")


def panel_build(ax, D) -> None:
    occ, colors = D["occ"], dict(D["colors"])
    x0, y0, z0 = D["wpos"]
    for dx in range(2):
        for dy in range(2):
            for dz in range(2):
                c = (x0 + dx, y0 + dy, z0 + dz)
                if occ[c]:
                    colors[c] = _hex(BLUE)
    bb = draw_voxels(ax, occ, colors, lw=0.04)

    # The pin is drawn in lattice coordinates, so the marker cannot drift off
    # the cells it names.
    px, py = _iso(x0 + 1, y0 + 1, z0)
    tx, ty = _iso(x0 + 1, occ.shape[1] + 8.0, z0)
    ax.plot([px, tx], [py, ty], color=BLUE, lw=0.7, zorder=5,
            solid_capstyle="butt")
    ax.plot([px], [py], marker="o", ms=2.2, color=BLUE, zorder=6)
    ax.annotate("one 2×2×2 window", (tx, ty), (0, 2),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=7.2, color=BLUE, zorder=6)
    bb[2] = min(bb[2], py)
    bb[3] = max(bb[3], ty + 2.5)
    _fit(ax, bb, padx=1.0, pady=0.6)
    ax.annotate(f"a real 3D-Craft house\n{D['n_blocks']} blocks",
                (0.5, 0.0), xycoords="axes fraction", xytext=(0, -2),
                textcoords="offset points", ha="center", va="top",
                fontsize=7.0, color=INK_2, linespacing=1.2)


def panel_window(ax, D) -> None:
    w = D["window"]
    cols = {tuple(c): _hex(BLUE) for c in np.argwhere(w)}
    bb = draw_voxels(ax, w, cols, edge=PAPER, lw=0.55, ghost=True)
    _fit(ax, bb, padx=0.25, pady=0.25)
    ax.annotate(f"8 cells, occupancy only\n(materials discarded)\ncode {D['code']}",
                (0.5, 0.0), xycoords="axes fraction", xytext=(0, -2),
                textcoords="offset points", ha="center", va="top",
                fontsize=7.0, color=INK_2, linespacing=1.2)


def panel_orbit(ax, D) -> None:
    """The eight group images and the min-code representative they collapse to."""
    step = 4.35
    bbs = []
    for i, (name, g, code) in enumerate(D["images"]):
        off = (i * step, 0.0)
        cols = {tuple(c): _hex(BLUE) for c in np.argwhere(g)}
        bbs.append(draw_voxels(ax, g, cols, offset=off, edge=PAPER, lw=0.45,
                               ghost=True))
        canon = code == D["canon"]
        ax.annotate(name, (off[0], -0.35), ha="center", va="top",
                    fontsize=7.2, style="italic", color=INK_2)
        ax.annotate(str(code), (off[0], -1.55), ha="center", va="top",
                    fontsize=7.0, weight="bold" if canon else "normal",
                    color=BLUE if canon else INK_2)

    bx = 8 * step + 2.1
    ax.annotate("", xy=(bx - 2.5, 2.0), xytext=(7 * step + 2.15, 2.0),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK_2,
                                shrinkA=0, shrinkB=0, mutation_scale=7))
    ax.annotate("min", ((bx - 2.5 + 7 * step + 2.15) / 2, 2.25), ha="center",
                va="bottom", fontsize=6.8, color=INK_2)
    canon_g = next(g for _, g, c in D["images"] if c == D["canon"])
    cols = {tuple(c): _hex(BLUE) for c in np.argwhere(canon_g)}
    bbs.append(draw_voxels(ax, canon_g, cols, offset=(bx, 0.0), edge=PAPER,
                           lw=0.45, ghost=True))
    ax.annotate(f"code {D['canon']}\n= bin {D['orbit']} of {D['n_orbits']}",
                (bx, -0.35), ha="center", va="top", fontsize=7.0, color=INK,
                linespacing=1.3)
    bb = np.array([min(b[0] for b in bbs), max(b[1] for b in bbs),
                   min(b[2] for b in bbs) - 3.6, max(b[3] for b in bbs)])
    _fit(ax, bb, padx=0.4, pady=0.15)


def panel_descriptor(ax, D) -> None:
    shift, layout = D["shift"], D["layout"]
    n = len(shift)
    ax.axhline(0, color=INK_2, lw=0.6)
    ax.bar(np.arange(n), shift, width=0.8, color=BLUE, linewidth=0)

    edges, o = [], 0
    for name, w in layout:
        edges.append((name, o, o + w))
        o += w
    lo, hi = float(shift.min()), float(shift.max())
    ax.set_ylim(lo - PAD_LO, hi + PAD_HI)
    top = ax.get_ylim()[1]

    # Block spans as a bracket ruler above the bars, keyed by letter: the
    # narrow blocks have no room for a name over them. The ruler is drawn in
    # figstyle.RULE because the letters carry it; the callout leaders below are
    # INK_2, because a leader the eye cannot follow back to its bar is useless.
    keys = "ABCD"
    for k, (name, a, b) in zip(keys, edges):
        y = top - 0.55
        ax.plot([a - 0.4, b - 0.6], [y, y], color=RULE, lw=0.6,
                clip_on=False, solid_capstyle="butt")
        for xx in (a - 0.4, b - 0.6):
            ax.plot([xx, xx], [y, y - 0.28], color=RULE, lw=0.6, clip_on=False)
        ax.annotate(k, ((a + b) / 2 - 0.5, y + 0.12), ha="center", va="bottom",
                    fontsize=7.2, weight="bold", color=INK_2)

    # callouts for the components that move
    sc = edges[0][2] + len(D["thickness_bins"]) + 8
    ix = {m: sc + D["metrics"].index(m) for m in D["metrics"]}
    calls = [
        (D["orbit"], (11.0, 2.9),
         f"bin {D['orbit']}: the window in panel 2", BLUE),
        (54, (38.0, 3.4),
         f"bin 54, the all-solid window:\n{D['solid_ratio']:.1f}× as common",
         INK),
        (ix["thickness_mean"], (66.0, 5.9), "thickness mean, p90", INK),
    ]
    for i, (tx, ty), text, col in calls:
        ax.annotate(text, (i, shift[i]), xytext=(tx, ty), textcoords="data",
                    fontsize=7.0, color=col, ha="center", va="center",
                    linespacing=1.25,
                    arrowprops=dict(arrowstyle="-", lw=0.5,
                                    color=BLUE if col is BLUE else INK_2,
                                    shrinkA=2, shrinkB=1.5))
    ax.set_xlim(-1.2, n + 0.2)
    # Three lines, not two: the probe's name is wider than this axis is
    # tall, and a rotated label longer than its axis runs into panel 3's
    # heading.
    ax.set_ylabel(f"{display('real@solidify')}\n− {display('real_test')}"
                  "\n(reference SD)", labelpad=0)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.set_xticks([0, 54, 59, 67, 76])
    ax.set_xticklabels(["1", "55", "60", "68", "77"])
    ax.set_yticks([-4, 0, 4])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _bar_unit(shift: np.ndarray) -> float:
    """Inches per reference SD, pinned to the scale the 5.05in canvas drew.

    The axes box then follows the padding rather than the other way round, so
    dropping footroom shortens the figure without rescaling a single bar.
    """
    lo, hi = float(shift.min()), float(shift.max())
    return _AX_D_IN / (hi - lo + 3.6 + 3.9)


def build_figure() -> None:
    _style()
    D = gather()

    # Layout is in inches measured DOWN from the top edge, because that is the
    # order the reader meets the panels in; `fy` converts at the last moment.
    lo, hi = float(D["shift"].min()), float(D["shift"].max())
    h_d = (hi - lo + PAD_LO + PAD_HI) * _bar_unit(D["shift"])
    # The canvas is as tall as the panels need and no taller. It used to be a
    # fixed 4.60in, whose bottom 0.57in held a four-line key decoding the A/B/C/D
    # brackets; that text is the caption's job, so the height it was using is
    # given back rather than left as white space. The figure now ends just under
    # panel 3, which starts at TOP_D: FOOT is what its x tick labels (7 pt on a
    # 1.5 pt pad) need below the axes, plus a little breathing space.
    TOP_D, FOOT = 2.855, 0.22
    H = TOP_D + h_d + FOOT

    def fy(down: float) -> float:
        return 1.0 - down / H

    def rect(x, w, down, h):
        return [x, fy(down + h), w, h / H]

    fig = plt.figure(figsize=(5.5, H))

    # Panel 1 spans the full column, like panels 2 and 3: the house on the
    # left, the window it feeds on the right, the arrow carrying the frequency.
    ax_b = fig.add_axes(rect(0.100, 0.245, 0.35, 1.00))
    ax_w = fig.add_axes(rect(0.730, 0.135, 0.37, 0.96))
    ax_o = fig.add_axes(rect(0.020, 0.960, 1.87, 0.78))
    ax_d = fig.add_axes(rect(0.115, 0.865, TOP_D, h_d))
    for ax in (ax_b, ax_w, ax_o):
        ax.set_axis_off()
        ax.set_aspect("equal")

    panel_build(ax_b, D)
    panel_window(ax_w, D)
    panel_orbit(ax_o, D)
    panel_descriptor(ax_d, D)

    _panel_title(fig, 0.02, fy(0.20), "1",
                 "a local window, taken from a real build")
    _panel_title(fig, 0.02, fy(1.72), "2", "its D4 orbit: eight images, one bin")
    _panel_title(fig, 0.02, fy(2.79), "3", "the 77-d descriptor")

    # Arrow and label placed against the *drawn* edge of the house, not the
    # axes box: an equal-aspect axes letterboxes its content.
    right = drawn_box(fig, ax_b)[2]
    left = drawn_box(fig, ax_w)[0]
    a0, a1 = right + 0.018, left - 0.018
    fig.patches.append(matplotlib.patches.FancyArrow(
        a0, fy(0.85), a1 - a0, 0.0, width=0.003, head_width=0.012,
        head_length=0.010, transform=fig.transFigure, figure=fig,
        color=INK_2, length_includes_head=True, linewidth=0))
    fig.text((a0 + a1) / 2, fy(0.784),
             f"{D['orbit_freq'] * 100:.1f}% of\nall windows",
             ha="center", va="bottom", fontsize=7.0, color=INK_2,
             linespacing=1.2)

    # What the construction buys is reported in the text, not on the drawing;
    # the separations stay here because the run log is the paper's audit trail.
    arms = D["card"]["arms"]
    sd = D["card"]["context"]["noise_floor_sd"]

    def sep(metric: str) -> float:
        v = arms["real@solidify"]["realism"][metric]["value"]
        r = arms["real_test"]["realism"][metric]["value"]
        return (v - r) / sd[f"realism.{metric}"]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"descriptor.{ext}", dpi=300)
    plt.close(fig)

    print(f"build index {D['build_index']} (3dcraft), {D['n_blocks']} blocks, "
          f"window at {D['wpos']} code {D['code']} -> orbit {D['orbit']} "
          f"(canonical code {D['canon']}, {D['orbit_freq']*100:.2f}% of windows)")
    o = 0
    for name, w in D["layout"]:
        s = D["shift"][o:o + w]
        r = D["raw_sd"][o:o + w]
        print(f"  {name:10s} dims {o+1:2d}-{o+w:2d}  max |shift| {np.abs(s).max():6.3f}"
              f"   (unclipped {np.abs(r).max():8.2f})")
        o += w
    print(f"  geom_kid sep {sep('geom_kid'):.1f} floors; "
          f"mv_dino_kid sep {sep('mv_dino_kid'):.2f} floors")
    print(f"canvas 5.5 x {H:.3f} in; bars {h_d:.3f} in")
    print(f"wrote {OUTDIR}/descriptor.png (+ .pdf)")


if __name__ == "__main__":
    build_figure()
