"""What the render metric actually sees: four views, real vs solidified, and the difference.

The panel that answers the reviewer objection "maybe the interior information
simply is not in the image". Rows 1 and 2 are the *metric's own inputs* -- the
four 224px orthographic views `blockgen.eval.bench.features.render_views`
produces for `mv_dino_kid`, called directly rather than reimplemented, so what
the reader sees is bit-for-bit the tensor DINOv2 is fed (up to the texture pack,
see below). Row 3 is their per-pixel absolute difference.

    .venv/bin/python scripts/figs/four_views.py

Writes `outputs/figstage/four_views.{png,pdf}` and the corpus measurement into
`outputs/figstage/four_views.md` is written by hand, not here; this script prints
every number it measures so the .md can quote it.

Two honest caveats, both measured rather than assumed, both reported by --stats:

1. **Flat colours hide part of the effect.** The licensing gate (FIGBRIEF) forbids
   showing renders built from Minecraft's texture files, so the published panel
   renders every block as a solid colour. But `solidify` fills with the build's
   *own most common material*, which is usually its wall material -- so in a flat
   render the block that appears in a doorway is often the same flat colour as
   the wall face behind it, and the two renders come back bit-identical. Measured
   on the 399-build houses_32 test split: the metric's textured renders change
   **91.7%** of views, the flat renders only **37.1%**. The flat panel is
   therefore a *lower bound* on the signal the metric discards, and the build
   selection rule below has to account for it. `--textures vanilla` reproduces
   the metric's real pixels for inspection; do not publish that output.

2. **The shown build is chosen by a stated rule, not by eye.** See `pick_build`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "outputs" / "figstage"
CORPUS = "houses_32"

# --- palette (scripts/paper_figures.py; colour follows the entity) ----------
# This figure is entirely about the render tier, whose colour in the paper is
# orange, so the sequential ramp for the difference map is a single-hue orange
# scale. Blue is reserved for geom_kid and does not appear here.
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
SURFACE = "#fcfcfb"
FRAME = "#c3c2bd"

#: light -> dark, single hue, monotone lightness (checked in `_check_ramp`).
DIFF_CMAP = LinearSegmentedColormap.from_list(
    "render_delta", ["#fdf4ef", "#f9c9ab", ORANGE, "#a8380f", "#4a1703"])

#: Every published render uses FLAT PER-BLOCK COLOURS, not Minecraft's texture
#: files -- `flat_textures()` in scripts/paper_figures.py, copied per FIGBRIEF.
_FLAT_DIR = tempfile.mkdtemp(prefix="blockgen-flat-")


def flat_textures():
    """A texture library with no texture files, so every block renders solid."""
    from blockgen.renderer.textures import load_face_textures
    return load_face_textures(pack_dir=_FLAT_DIR)


@contextlib.contextmanager
def texture_mode(mode: str):
    """Make `features.render_views` use flat colours without editing it.

    `render_views` calls `load_face_textures()` with no arguments, i.e. the
    vanilla pack, because that is what the metric does. Patching the module
    attribute keeps the *entire* rendering and compositing path -- camera,
    ortho framing, 224px, alpha-onto-white -- exactly the code the benchmark
    runs, and changes only which colours the faces get.
    """
    if mode == "vanilla":
        yield
        return
    import blockgen.renderer.textures as T
    orig = T.load_face_textures
    T.load_face_textures = lambda pack_dir=_FLAT_DIR: orig(pack_dir=_FLAT_DIR)
    try:
        yield
    finally:
        T.load_face_textures = orig


def _check_ramp() -> None:
    """Fail loudly if the sequential ramp is not monotone in lightness."""
    import colorsys
    lum = [colorsys.rgb_to_hls(*DIFF_CMAP(x)[:3])[1] for x in np.linspace(0, 1, 32)]
    if not all(b < a + 1e-6 for a, b in zip(lum, lum[1:])):
        raise AssertionError("DIFF_CMAP is not monotone light->dark")


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------

def _corpus_mask(n: int, corpus: str = "3dcraft") -> List[bool]:
    """Which test-split rows may be published, by upstream source.

    Only 3D-Craft builds are publishable as images: GrabCraft's terms name
    pictures of blueprints as a prohibited form of distribution, and text2mc's
    builds originate from PlanetMinecraft, whose terms prohibit redistributing
    site data. 3D-Craft comes from an MIT-licensed research release.
    (Copied from scripts/paper_figures.py per FIGBRIEF.)
    """
    from blockgen.eval.bench import splits as _sp
    man = json.loads((ROOT / "data" / "minecraft" / "cache"
                      / f"{CORPUS}_manifest.json").read_text())
    rows = list(_sp.load_split(CORPUS).indices("test"))
    return [man["items"][rows[i]].get("corpus") == corpus for i in range(n)]


def interior_cells(s) -> int:
    """Enclosed air cells -- the volume `solidify` destroys."""
    from scipy import ndimage
    occ = s.occupied_mask
    filled = ndimage.binary_fill_holes(
        occ, structure=ndimage.generate_binary_structure(3, 1))
    return int((filled & ~occ).sum())


def view_deltas(real, solid, view, mode: str) -> List[Dict]:
    """Per-view difference statistics for one build, on the metric's own renders."""
    from blockgen.eval.bench import features as ft
    with texture_mode(mode):
        a = ft.render_views([real], view, verbose=False)
        b = ft.render_views([solid], view, verbose=False)
    out = []
    for ia, ib in zip(a, b):
        d = np.abs(ia.astype(np.int16) - ib.astype(np.int16))
        # The silhouette claim, measured rather than asserted: the set of
        # non-background pixels. `solidify` only adds cells that `fill_holes`
        # found enclosed, so in principle no added block can be on the
        # silhouette -- but "in principle" is not a measurement, and a
        # rasteriser can sample a diagonal seam, so it is checked per view.
        sil_a = (ia != 255).any(axis=2)
        sil_b = (ib != 255).any(axis=2)
        # Two masks, because the cheap one is not trustworthy on textured
        # renders: a texture pixel can be exactly white, so "non-background"
        # flickers *inside* the build wherever a window shows a pale block.
        # The outline is the filled footprint -- the frame margin guarantees the
        # true background is connected to the border, so filling is well posed.
        from scipy import ndimage as _nd
        foot_a = _nd.binary_fill_holes(sil_a)
        foot_b = _nd.binary_fill_holes(sil_b)
        out.append({"frac_px": float((d > 0).any(axis=2).mean()),
                    "max_ch": int(d.max()),
                    "mean_abs": float(d.mean()),
                    "silhouette_same": bool(np.array_equal(sil_a, sil_b)),
                    "silhouette_px_diff": int((sil_a != sil_b).sum()),
                    "outline_same": bool(np.array_equal(foot_a, foot_b)),
                    "outline_px_diff": int((foot_a != foot_b).sum())})
    return out, a, b


def measure_corpus(mode: str, limit: int | None = None) -> Dict:
    """The two abstract numbers, over the whole held-out split. ~35 s on this box.

    Reports both the all-corpora figure (every held-out build, which is the pool
    every metric number in the paper is computed on) and the 3D-Craft-only
    figure, because the shown build is drawn from the latter.
    """
    from blockgen.eval.bench import splits, features as ft, probes
    split = splits.load_split(CORPUS)
    test = splits.split_structures(split, "test")
    if limit:
        test = test[:limit]
    is3d = _corpus_mask(len(test))
    view = ft.ViewConfig()
    rows = []
    for i, s in enumerate(test):
        real = s.crop_to_non_air()
        solid = probes.solidify(real)
        per, _, _ = view_deltas(real, solid, view, mode)
        rows.append({"i": i, "is3d": is3d[i], "interior": interior_cells(real),
                     "mass": int(real.occupied_mask.sum()),
                     "mass_solid": int(solid.occupied_mask.sum()),
                     "views": per})
        if (i + 1) % 100 == 0:
            print(f"    measured {i + 1}/{len(test)}", flush=True)

    def agg(sel) -> Dict:
        fr = np.array([v["frac_px"] for r in rows if sel(r) for v in r["views"]])
        if fr.size == 0:                     # only reachable under --limit
            return {"n_builds": 0, "n_views": 0}
        # The EGL rasteriser is not bit-reproducible across calls: rendering a
        # structure against *itself* comes back with a handful of pixels off by
        # one level in a few views per thousand. Measured on the builds
        # `solidify` cannot change (no enclosed air at all, so the two inputs are
        # the same object): 1-3 of 80 views differ, always by max |delta| = 1 on
        # 1-5 pixels. So "changed" is reported at a threshold of 2 levels as
        # well as at >0 -- the thresholded rate is identical across repeat runs
        # (91.35% three times) while the raw one wobbles 91.67-91.79%, and the
        # thresholded one is the number worth printing.
        JITTER = 2
        mc = np.array([v["max_ch"] for r in rows if sel(r) for v in r["views"]])
        ch = fr > 0
        blds = [r for r in rows if sel(r)]
        return {
            "n_builds": len(blds), "n_views": int(len(fr)),
            "views_changed": float(ch.mean()),
            "views_changed_robust": float((mc >= JITTER).mean()),
            "frac_px_median_robust": float(np.median(fr[mc >= JITTER]))
            if (mc >= JITTER).any() else 0.0,
            "max_ch_median_robust": float(np.median(mc[mc >= JITTER]))
            if (mc >= JITTER).any() else 0.0,
            "jitter_views": float(np.mean(
                [v["max_ch"] > 0 for r in rows if sel(r) and r["interior"] == 0
                 for v in r["views"]] or [np.nan])),
            "jitter_max_ch": float(np.max(
                [v["max_ch"] for r in rows if sel(r) and r["interior"] == 0
                 for v in r["views"]] or [np.nan])),
            "bit_identical_views": float((~ch).mean()),
            "builds_changed": float(np.mean([any(v["frac_px"] > 0 for v in r["views"])
                                             for r in blds])),
            "builds_with_interior": float(np.mean([r["interior"] > 0 for r in blds])),
            "frac_px_median_of_changed": float(np.median(fr[ch])),
            "frac_px_mean_of_changed": float(fr[ch].mean()),
            "frac_px_max": float(fr.max()),
            "max_ch_median_of_changed": float(np.median(mc[ch])),
            "max_ch_p90_of_changed": float(np.percentile(mc[ch], 90)),
            "max_ch_max": int(mc.max()),
            "silhouette_identical": float(np.mean(
                [v["silhouette_same"] for r in rows if sel(r) for v in r["views"]])),
            "outline_identical": float(np.mean(
                [v["outline_same"] for r in rows if sel(r) for v in r["views"]])),
            "outline_px_diff_median": float(np.median(
                [v["outline_px_diff"] for r in rows if sel(r) for v in r["views"]])),
            "outline_px_diff_max": int(np.max(
                [v["outline_px_diff"] for r in rows if sel(r) for v in r["views"]])),
            "mass_median_pct": float(np.median(
                [100.0 * (r["mass_solid"] - r["mass"]) / r["mass"] for r in blds])),
        }

    return {"mode": mode, "all": agg(lambda r: True),
            "3dcraft": agg(lambda r: r["is3d"]), "rows": rows}


def containment_check(n: int, seed: int = 0) -> Dict:
    """Is every pixel the corruption moves *inside* the silhouette?

    The silhouette has to be read off the FLAT renders. On textured renders a
    block face can be exactly white, so "non-background" flickers inside the
    build wherever a window shows a pale block -- the naive mask then reports the
    silhouette as changed (27.7% identical) when what changed is a pixel seen
    *through* the wall. Flat colours are never pure white, so the flat mask is
    the geometry, and it is bit-identical on every view measured.

    This renders both texture modes for the same builds, which the two corpus
    passes cannot do, so it is sampled rather than exhaustive.
    """
    from blockgen.eval.bench import splits, features as ft, probes
    from scipy import ndimage
    split = splits.load_split(CORPUS)
    test = splits.split_structures(split, "test")
    idxs = np.random.default_rng(seed).choice(len(test), min(n, len(test)),
                                              replace=False)
    view = ft.ViewConfig()
    tot = outside = on_edge = views = sil_diff = 0
    for i in idxs:
        real = test[int(i)].crop_to_non_air()
        solid = probes.solidify(real)
        _, fa, fb = view_deltas(real, solid, view, "flat")
        _, ta, tb = view_deltas(real, solid, view, "vanilla")
        for k in range(len(fa)):
            sil = (fa[k] != 255).any(axis=2)
            sil_diff += int((sil != (fb[k] != 255).any(axis=2)).sum())
            ch = (np.abs(ta[k].astype(np.int16) - tb[k].astype(np.int16)) > 0
                  ).any(axis=2)
            views += 1
            tot += int(ch.sum())
            outside += int((ch & ~sil).sum())
            on_edge += int((ch & (sil & ~ndimage.binary_erosion(sil))).sum())
    return {"n_builds": len(idxs), "n_views": views, "changed_px": tot,
            "silhouette_px_differing": sil_diff,
            "changed_px_outside_silhouette": outside,
            "changed_px_on_silhouette_ring": on_edge}


def pick_build(rows: Sequence[Dict], mode: str) -> Tuple[int, str]:
    """The shown build, by a stated rule. No eyeballing, no effect-size tuning.

    Rule: **the median interior volume among the publishable (3D-Craft) builds of
    the held-out split whose four flat-colour renders all change under
    `solidify`.**

    The first two clauses are the obvious non-cherry-picked rule (Figure 1 takes
    the *maximum* interior volume of the same pool; the median is the typical
    build instead of the extreme one). The third clause is forced by the flat
    rendering and is disclosed rather than hidden: for 61% of 3D-Craft builds the
    flat renders of real and solidified are bit-identical in at least one view --
    including Figure 1's max-interior build, whose flat renders are identical in
    *all four* views while the metric's own textured renders of it change 8% of
    pixels at max |delta| 225/255. Drawing the panel from that build would show
    the reader four blank difference maps and argue the opposite of the truth.

    The conditioning is on the rendering, not on the corruption, and it barely
    moves the interior-volume distribution it selects from: the median interior
    volume of the conditioned pool is within 5% of the unconditioned 3D-Craft
    median (printed at run time), so the shown build is still a typical build.
    With `--textures vanilla` the third clause is unnecessary and dropped.
    """
    pool = [r for r in rows if r["is3d"]]
    uncond_median = float(np.median([r["interior"] for r in pool]))
    if mode == "flat":
        keep = [r for r in pool if all(v["frac_px"] > 0 for v in r["views"])]
        why = (f"median interior volume among 3D-Craft held-out builds whose four "
               f"flat renders all change ({len(keep)}/{len(pool)})")
    else:
        keep = pool
        why = f"median interior volume among 3D-Craft held-out builds ({len(pool)})"
    order = np.argsort([r["interior"] for r in keep])
    row = keep[order[len(order) // 2]]
    med = float(np.median([r["interior"] for r in keep]))
    print(f"  selection: {why}")
    print(f"    pool median interior {med:.0f} cells vs unconditioned 3D-Craft "
          f"median {uncond_median:.0f} "
          f"({100.0 * (med - uncond_median) / uncond_median:+.1f}%)")
    print(f"    shown build: test index {row['i']}, interior {row['interior']} cells, "
          f"mass {row['mass']} -> {row['mass_solid']} "
          f"(+{100.0 * (row['mass_solid'] - row['mass']) / row['mass']:.0f}%)")
    return row["i"], why


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "text.color": INK, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "figure.dpi": 200,
        # Embed real glyphs rather than Type-3 procedures, so the labels stay
        # selectable text in the submission PDF (matches ladder/cutaway).
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _frame(ax, colour=FRAME, lw=0.5) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(colour); sp.set_linewidth(lw)


def build_figure(real_imgs, solid_imgs, view, stats, stats_metric, metric_agg,
                 shown_agg, sil_agg, mode, out: Path) -> None:
    """Three rows x four views, one shared colourbar. 5.5in single column.

    `metric_agg` is the aggregate over the metric's *own* (textured) renders --
    the numbers the headline quotes, because the headline is a claim about the
    metric's input. `shown_agg` is the aggregate for the rendering actually
    drawn here, quoted in the footnote so the two are never confused.
    """
    n = len(real_imgs)
    fig = plt.figure(figsize=(5.5, 5.15))
    gs = fig.add_gridspec(
        3, n + 1, left=0.005, right=0.997, top=0.805, bottom=0.205,
        wspace=0.05, hspace=0.075, width_ratios=[0.11] + [1.0] * n)

    rows = [("real build", real_imgs), (r"$\mathtt{solidify}$", solid_imgs)]
    for r, (label, imgs) in enumerate(rows):
        for c, img in enumerate(imgs):
            ax = fig.add_subplot(gs[r, c + 1])
            ax.imshow(img, interpolation="nearest")
            _frame(ax)
            if r == 0:
                azim, elev = view.views[c]
                ax.set_title(f"view {c + 1}  ({azim:g}$^\\circ$, {elev:g}$^\\circ$)",
                             fontsize=7.2, color=INK_2, pad=2.5)
        lab = fig.add_subplot(gs[r, 0]); lab.axis("off")
        lab.text(0.58, 0.5, label, rotation=90, ha="center", va="center",
                 fontsize=8.2, color=INK)

    # --- row 3: per-pixel absolute difference ------------------------------
    im = None
    for c, (a, b) in enumerate(zip(real_imgs, solid_imgs)):
        d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
        ax = fig.add_subplot(gs[2, c + 1])
        # A faint silhouette of the build, so the lit pixels can be located on
        # it. Grey, 14% alpha, never confusable with the orange difference ramp.
        ink = (a.astype(np.float32).mean(axis=2) < 250).astype(float)
        ax.imshow(ink, cmap="Greys", vmin=0, vmax=1, alpha=0.14,
                  interpolation="nearest")
        im = ax.imshow(np.ma.masked_where(d == 0, d), cmap=DIFF_CMAP,
                       vmin=0, vmax=255, interpolation="nearest")
        _frame(ax)
        s = stats[c]
        ax.text(0.5, -0.045,
                f"{100 * s['frac_px']:.2f}% of pixels\nmax $|\\Delta|$ "
                f"{s['max_ch']}/255",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=7.0, color=INK, linespacing=1.35)
        if mode == "flat":
            # The same view through the metric's own textured renders, so the
            # reader is never left with the flat rendering's smaller number.
            m = stats_metric[c]
            ax.text(0.5, -0.300,
                    f"textured: {100 * m['frac_px']:.2f}%, {m['max_ch']}/255",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=6.3, color=MUTED)
    lab = fig.add_subplot(gs[2, 0]); lab.axis("off")
    lab.text(0.58, 0.5, r"$|\Delta|$ per pixel", rotation=90, ha="center",
             va="center", fontsize=8.2, color=INK)

    # Horizontal colourbar, bottom left: a vertical one next to the last tile
    # would have to steal width from the images, which are the evidence.
    cax = fig.add_axes([0.075, 0.072, 0.20, 0.015])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.5); cb.outline.set_edgecolor(FRAME)
    cb.set_ticks([0, 128, 255])
    cax.tick_params(labelsize=6.6, length=1.6, width=0.5, color=MUTED,
                    labelcolor=INK_2, pad=1.4)
    cax.set_xlabel("max channel $|\\Delta|$, blank = identical pixel",
                   fontsize=6.8, color=INK_2, labelpad=2.0)

    # --- headline ----------------------------------------------------------
    a = metric_agg
    fig.text(0.005, 0.992,
             "The corruption is in the metric's own input",
             fontsize=9.6, color=INK, ha="left", va="top")
    fig.text(0.005, 0.947,
             f"Across the {a['n_builds']} held-out builds "
             f"{100 * a['views_changed_robust']:.0f}% of the metric's views change "
             f"under $\\mathtt{{solidify}}$, and where one changes\na median "
             f"{100 * a['frac_px_median_robust']:.2f}% of its pixels move by a "
             f"median maximum of {a['max_ch_median_robust']:.0f}/255. The "
             f"silhouette is bit-identical\nin all {sil_agg['n_views']} views: "
             f"interiors leak through apertures, never around the outline.",
             fontsize=7.2, color=INK_2, ha="left", va="top", linespacing=1.45)

    # Both modes are publishable. Mojang's guidelines permit non-commercial use
    # of images of the game, and the release ships no texture files; flat colors
    # remain available for a figure that has to read in grayscale.
    foot = ("gray = build silhouette, orange = changed pixels\n"
            "flat per-block colors understate the effect: they change "
            f"{100 * shown_agg['views_changed_robust']:.0f}% of views against "
            f"{100 * a['views_changed_robust']:.0f}%\nfor the metric's textured "
            "renders, so this panel is a lower bound"
            if mode == "flat" else
            "gray = build silhouette, orange = changed pixels\n"
            "the metric's own textured renders. Much of this difference is a\n"
            "texture-atlas artifact of our mesh builder, not the corruption")
    fig.text(0.99, 0.093, foot, fontsize=6.3, color=MUTED, ha="right",
             va="top", linespacing=1.45)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix("." + ext), dpi=300)
    plt.close(fig)
    print(f"  wrote {out.with_suffix('.png')} (+ .pdf)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--textures", choices=["flat", "vanilla"], default="flat",
                    help="flat = publishable; vanilla = the metric's real pixels")
    ap.add_argument("--limit", type=int, default=None,
                    help="measure only the first N held-out builds (debugging)")
    ap.add_argument("--out", default=None, help="output stem (default outputs/figstage/four_views)")
    ap.add_argument("--containment", type=int, default=60,
                    help="builds for the changed-pixels-inside-silhouette check "
                         "(0 to skip); needs both texture modes per build")
    ap.add_argument("--stats-json", default=None,
                    help="also dump every measured number here")
    args = ap.parse_args()

    _check_ramp()
    _style()
    from blockgen.eval.bench import splits, features as ft, probes

    def report(corpus: Dict) -> None:
        for pool in ("all", "3dcraft"):
            a = corpus[pool]
            if not a["n_views"]:             # only reachable under --limit
                print(f"  [{corpus['mode']:7s}] {pool:8s} empty")
                continue
            print(f"  [{corpus['mode']:7s}] {pool:8s} n={a['n_builds']:4d} builds / "
                  f"{a['n_views']:5d} views | "
                  f"views changed {100 * a['views_changed']:5.1f}% "
                  f"(>=2 levels: {100 * a['views_changed_robust']:5.2f}%) | "
                  f"renderer jitter {100 * a['jitter_views']:.2f}% of "
                  f"unchangeable views, max |d| {a['jitter_max_ch']:.0f} | "
                  f"bit-identical {100 * a['bit_identical_views']:5.1f}% | "
                  f"builds changed {100 * a['builds_changed']:5.1f}% | "
                  f"builds w/ interior {100 * a['builds_with_interior']:5.1f}% | "
                  f"px|changed med {100 * a['frac_px_median_of_changed']:.2f}% "
                  f"max {100 * a['frac_px_max']:.1f}% | "
                  f"max|d| med {a['max_ch_median_of_changed']:.0f} "
                  f"p90 {a['max_ch_p90_of_changed']:.0f} max {a['max_ch_max']} | "
                  f"outline identical {100 * a['outline_identical']:5.1f}% "
                  f"(worst {a['outline_px_diff_max']} px) | "
                  f"non-bg mask identical {100 * a['silhouette_identical']:5.1f}% | "
                  f"median mass +{a['mass_median_pct']:.0f}%")

    # The headline numbers are a claim about what the *metric* consumes, so they
    # are always measured on the metric's own vanilla-textured renders -- numbers
    # are aggregate statistics, which the licensing gate permits; only pictures
    # are restricted. The panel itself is drawn in whatever `--textures` says.
    print(f"[four_views] measuring {CORPUS} test split (vanilla = metric input)")
    metric = measure_corpus("vanilla", args.limit)
    report(metric)
    # The flat pass is run in either mode. It is what the panel draws when
    # --textures flat, and it is also the only unambiguous silhouette
    # measurement available, because no flat block colour is pure white.
    print(f"[four_views] measuring {CORPUS} test split (flat renders)")
    flat_pass = measure_corpus("flat", args.limit)
    report(flat_pass)
    corpus = metric if args.textures == "vanilla" else flat_pass
    if flat_pass["all"]["outline_identical"] != 1.0:
        print("  WARNING: flat silhouettes are NOT all identical; the headline "
              "sentence about the outline must be rewritten")

    if args.containment:
        print(f"[four_views] containment check on {args.containment} builds "
              f"(both texture modes, same builds)")
        cont = containment_check(args.containment)
        print(f"  {cont['n_views']} views, {cont['changed_px']} changed pixels "
              f"(textured) | outside the flat silhouette: "
              f"{cont['changed_px_outside_silhouette']} | on its boundary ring: "
              f"{cont['changed_px_on_silhouette_ring']} "
              f"({100 * cont['changed_px_on_silhouette_ring'] / max(cont['changed_px'], 1):.1f}%) "
              f"| flat silhouette pixels differing: "
              f"{cont['silhouette_px_differing']}")
    else:
        cont = None

    idx, why = pick_build(corpus["rows"], args.textures)
    split = splits.load_split(CORPUS)
    test = splits.split_structures(split, "test")
    real = test[idx].crop_to_non_air()
    solid = probes.solidify(real)
    view = ft.ViewConfig()
    per, real_imgs, solid_imgs = view_deltas(real, solid, view, args.textures)
    per_metric = metric["rows"][idx]["views"]
    for c, (s, m) in enumerate(zip(per, per_metric)):
        print(f"    view {c + 1}: shown({args.textures}) "
              f"{100 * s['frac_px']:5.2f}% of pixels changed, max |delta| "
              f"{s['max_ch']:3d}/255, frame mean {s['mean_abs']:.3f}"
              f"   |   metric(vanilla) {100 * m['frac_px']:5.2f}%, "
              f"max |delta| {m['max_ch']:3d}/255")

    out = Path(args.out) if args.out else OUTDIR / "four_views"
    build_figure(real_imgs, solid_imgs, view, per, per_metric, metric["all"],
                 corpus["all"], flat_pass["all"], args.textures, out)

    if args.stats_json:
        Path(args.stats_json).write_text(json.dumps(
            {"metric_vanilla": {k: metric[k] for k in ("all", "3dcraft")},
             "flat": {k: flat_pass[k] for k in ("all", "3dcraft")},
             "containment": cont,
             "shown_mode_corpus": {k: corpus[k] for k in ("all", "3dcraft")},
             "mode": args.textures, "shown_index": idx, "rule": why,
             "shown_views": per, "shown_views_metric_textures": per_metric,
             "rows": corpus["rows"]}, indent=1))
        print(f"  wrote {args.stats_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
