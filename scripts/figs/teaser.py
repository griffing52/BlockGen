"""Fig. 1 of the CVPR paper, and an asset pack for redrawing it by hand.

The teaser's claim: a real held-out house and its Fill-Interior copy look the
same from every exterior camera, differ completely once cut open, and the
metrics the field uses agree with the camera rather than the cut. The metric
strip reads its numbers from the recorded runs at draw time:

* MV-KID and V-KID: the appearance and geometry pillars of `real@solidify`
  in the n=256 leaderboard of record (noise floors);
* Uni3D KID on visible-surface points: `uni3d_vis_kid` on `solidify` in the
  Minecraft 3D-native run (seed 0, n=128; seed 1 in the caption);
* GPT-4o: P(prefers real) on the exterior Fill-Interior condition, with the
  pair-clustered bootstrap interval.

    BLOCKGEN_FIG_RENDER=cpu .venv/bin/python scripts/figs/teaser.py            # both
    BLOCKGEN_FIG_RENDER=cpu .venv/bin/python scripts/figs/teaser.py --figure   # teaser only
    BLOCKGEN_FIG_RENDER=cpu .venv/bin/python scripts/figs/teaser.py --assets   # pack only

Output: `deliverables/paper/cvpr27/figures/fig1_teaser.{pdf,svg,png}` and
`deliverables/paper/cvpr27/figures/fig1_assets/` (transparent PNGs, vector SVG
sections and the metric strip; see its README). Only the 3D-Craft build
`paper_figures.TEASER_BUILD` is drawn (licensing rule in `paper_figures`).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.ticker
import matplotlib.transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "figs"))
import cpu_render                      # noqa: E402
cpu_render.install()
import figstyle                        # noqa: E402
from figstyle import (BLUE, ORANGE, INK, INK_2, RULE, SHADE, SOLID, VOID,
                      CVPR_FULL)       # noqa: E402

PAPER_DIR = ROOT / "deliverables" / "paper" / "cvpr27" / "figures"
ASSETS = PAPER_DIR / "fig1_assets"

LEADERBOARD = ROOT / "outputs" / "run_20261103_090315_bench_n_256_leaderboard_of_record"
PROBE3D = {0: ROOT / "outputs" / "run_20261103_004358_mc_probe3d_s0",
           1: ROOT / "outputs" / "run_20261103_004358_mc_probe3d_s1"}
VLM = ROOT / "outputs" / "run_20261102_214026_vlm_judge"

#: Neutral mark for observers that are neither tier (a 3D encoder, a VLM).
GREY = "#7a7a7a"


# --- numbers ----------------------------------------------------------------

def numbers() -> Dict[str, float]:
    card = json.loads((LEADERBOARD / "scorecard.json").read_text())
    rows = {r["arm"]: r for r in card["run"]["blockscore"]}
    sol = rows["real@solidify"]["pillars"]

    def uni3d(seed: int) -> float:
        for line in (PROBE3D[seed] / "table.md").read_text().splitlines():
            if line.startswith("| uni3d_vis_kid |"):
                return float(re.match(r"\|\s*uni3d_vis_kid\s*\|\s*([+-][\d.]+)",
                                      line).group(1))
        raise KeyError("uni3d_vis_kid")

    vlm = json.loads((VLM / "analysis.json").read_text())["conditions"]["interior_ext"]
    return {"mv": sol["appearance"], "vkid": sol["geometry"],
            "uni3d": uni3d(0), "uni3d_s1": uni3d(1),
            "vlm": vlm["order_averaged_rate_a"],
            "vlm_lo": vlm["cluster_boot_ci"][0], "vlm_hi": vlm["cluster_boot_ci"][1],
            "vlm_pairs": vlm["n_pairs"]}


# --- the build --------------------------------------------------------------

def builds():
    """(real, filled, hollow) for the teaser build, all cropped."""
    import paper_figures as pf
    from blockgen.eval.bench import splits, probes
    test = splits.split_structures(splits.load_split("houses_32"), "test")
    i = pf.TEASER_BUILD
    assert pf._corpus_mask(len(test))[i], "teaser build must be 3D-Craft"
    real = test[i].crop_to_non_air()
    return i, real, probes.solidify(real), probes.hollow(real)


def textures():
    import paper_figures as pf
    return pf.flat_textures()


def render(s, *, px: int, az: float, el: float, cut: str = "", tex=None) -> np.ndarray:
    """RGBA uint8, transparent background."""
    from blockgen.eval.bench import features as ft
    from blockgen.renderer.textured import render_structure
    shown = ft.apply_cut(s, cut) if cut else s
    return render_structure(shown, px=px, azim_deg=az, elev_deg=el, ortho=True,
                            face_textures=tex)


def crop_alpha(imgs, pad: int = 8):
    """Crop RGBA images to one shared box (keeps them at one scale)."""
    ink = np.zeros(imgs[0].shape[:2], bool)
    for im in imgs:
        ink |= im[..., 3] > 0
    ys, xs = np.where(ink)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, ink.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, ink.shape[1])
    return [im[y0:y1, x0:x1] for im in imgs]


def interior(occ: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    filled = ndimage.binary_fill_holes(occ, structure=ndimage.generate_binary_structure(3, 1))
    return filled & ~occ


def best_planes(real) -> Tuple[int, int]:
    """(z for the vertical section, y for the floor plan): most rooms, then area."""
    from scipy import ndimage
    inter = interior(real.occupied_mask)
    best_z = max(((int(ndimage.label(inter[:, :, z])[1]), int(inter[:, :, z].sum())), z)
                 for z in range(inter.shape[2]))[1]
    best_y = max(((int(ndimage.label(inter[:, y, :])[1]), int(inter[:, y, :].sum())), y)
                 for y in range(inter.shape[1]))[1]
    return best_z, best_y


# --- vector sections ----------------------------------------------------------

def section_svg(occ2: np.ndarray, int2: np.ndarray, path: Path, cell: int = 12,
                flip_y: bool = True) -> None:
    """One <rect> per cell, grouped by layer, so each layer recolours in one click."""
    h, w = occ2.shape          # rows = vertical axis when flip_y
    def rects(mask, color):
        out = []
        for r, c in zip(*np.nonzero(mask)):
            y = (h - 1 - r) if flip_y else r
            out.append(f'<rect x="{c*cell}" y="{y*cell}" width="{cell}" height="{cell}"/>')
        return f'<g fill="{color}">' + "".join(out) + "</g>"
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w*cell}" height="{h*cell}" '
           f'viewBox="0 0 {w*cell} {h*cell}">'
           f'<g id="interior_air">{rects(int2, VOID)}</g>'
           f'<g id="blocks">{rects(occ2, SOLID)}</g></svg>')
    path.write_text(svg)


# --- metric strip -------------------------------------------------------------

def draw_strip(fig, rect, nums) -> None:
    """Three kernel distances on one symlog axis, the VLM on a probability axis.

    `rect` = (left, bottom, width, height) in figure fraction for the whole strip;
    the left 40% holds the observer labels, the rest the two axes.
    """
    L, B, W, H = rect
    lab_w = 0.36 * W
    px0, pw = L + lab_w, W - lab_w
    top_h, gap, bot_h = 0.64 * H, 0.22 * H, 0.08 * H
    ax = fig.add_axes((px0, B + bot_h + gap, pw, top_h))
    ax2 = fig.add_axes((px0, B, pw, bot_h))
    rows = [("MV-KID", "renders", nums["mv"], ORANGE),
            ("Uni3D KID", "visible points", nums["uni3d"], GREY),
            ("V-KID", "voxels", nums["vkid"], BLUE)]
    n = len(rows)
    ax.axvspan(-2, 2, color=SHADE, lw=0, zorder=0)
    ax.text(0, n - 0.45, "\u00b12", ha="center", va="bottom", fontsize=7, color=INK_2)
    for k, (name, obs, v, col) in enumerate(rows):
        y = n - 1 - k
        ax.plot([0, v], [y, y], color=col, lw=1.2, solid_capstyle="butt", zorder=2)
        ax.plot(v, y, "o", ms=5, color=col, mec="white", mew=0.8, zorder=3)
        lab = f"{v:+.2f}" if abs(v) < 10 else f"{v:+.0f}"
        if v > 10:
            ax.text(v, y + 0.32, lab, ha="center", va="bottom", fontsize=7, color=INK)
        else:
            ax.text(3.2, y, lab, ha="left", va="center", fontsize=7, color=INK)
        tr = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(-0.04, y, name, ha="right", va="bottom", fontsize=7.5, color=INK,
                fontweight="bold", transform=tr)
        ax.text(-0.04, y - 0.02, obs, ha="right", va="top", fontsize=7, color=INK_2,
                transform=tr)
    ax.set_xscale("symlog", linthresh=2, linscale=0.6)
    ax.set_xlim(-3, 2000)
    ax.set_ylim(-0.75, n - 0.1)
    ax.set_xticks([0, 10, 100, 1000])
    ax.set_xticklabels(["0", "10", "100", "1000"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=7, length=2.5, pad=1.5)
    ax.set_xlabel("noise floors from real", fontsize=7, labelpad=1.5)

    ax2.axvline(0.5, color=INK_2, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax2.plot([nums["vlm_lo"], nums["vlm_hi"]], [0, 0], color=GREY, lw=1.6, zorder=2)
    ax2.plot(nums["vlm"], 0, "o", ms=5, color=GREY, mec="white", mew=0.8, zorder=3)
    ax2.text(nums["vlm_hi"] + 0.025, 0, f"{nums['vlm']:.2f}", fontsize=7, va="center",
             color=INK)
    ax2.text(0.48, 0.0, "chance", ha="right", va="center", fontsize=7, color=INK_2)
    tr = matplotlib.transforms.blended_transform_factory(ax2.transAxes, ax2.transData)
    ax2.text(-0.04, 0, "GPT-4o judge", ha="right", va="bottom", fontsize=7.5, color=INK,
             fontweight="bold", transform=tr)
    ax2.text(-0.04, 0, "exterior pairs", ha="right", va="top", fontsize=7,
             color=INK_2, transform=tr)
    ax2.set_xlim(0.2, 1.0)
    ax2.set_ylim(-0.7, 0.9)
    ax2.set_xticks([0.5, 0.75, 1.0])
    ax2.set_xticklabels(["0.5", "0.75", "1"])
    ax2.set_yticks([])
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="x", labelsize=7, length=2.5, pad=1.5)
    ax2.set_xlabel("P(prefers real)", fontsize=7, labelpad=1.5)


# --- the teaser ---------------------------------------------------------------

def figure() -> None:
    figstyle.apply(size=8)
    nums = numbers()
    i, real, filled, _ = builds()
    tex = textures()
    ext = crop_alpha([render(real, px=900, az=45, el=30, tex=tex),
                      render(filled, px=900, az=45, el=30, tex=tex)])
    cut = crop_alpha([render(real, px=900, az=45, el=60, cut="half:y:lo", tex=tex),
                      render(filled, px=900, az=45, el=60, cut="half:y:lo", tex=tex)])
    diff = int(np.abs(ext[0].astype(int) - ext[1].astype(int)).max())

    def white(im):
        a = im[..., 3:4] / 255.0
        return (im[..., :3] * a + 255 * (1 - a)).astype(np.uint8)

    Wd, Ht = CVPR_FULL, 1.95
    fig = plt.figure(figsize=(Wd, Ht))
    img_h = 1.45 / Ht                     # image row height (fraction)
    img_b = 0.20 / Ht
    tile_w = 1.02 / Wd
    lefts = [0.00, 1.06, 2.30, 3.36]
    tiles = [ext[0], ext[1], cut[0], cut[1]]
    labs = ["Real", "Fill-Interior", "Real", "Fill-Interior"]
    for k, (x, im, lab) in enumerate(zip(lefts, tiles, labs)):
        ax = fig.add_axes((x / Wd, img_b, tile_w, img_h))
        ax.imshow(white(im))
        ax.set_aspect("equal", anchor="C")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        h_in = min(1.02 * im.shape[0] / im.shape[1], img_h * Ht)
        y_lab = img_b * Ht + (img_h * Ht - h_in) / 2 - 0.03
        fig.text((x + 0.51) / Wd, y_lab / Ht, lab, ha="center", va="top",
                 fontsize=7.5, color=INK, fontweight="bold" if k % 2 else "normal")
    fig.text(0.0, 1.0, "a  Exterior views", ha="left", va="top", fontsize=8, color=INK,
             fontweight="bold")
    fig.text(2.30 / Wd, 1.0, "b  Roof removed", ha="left", va="top", fontsize=8,
             color=INK, fontweight="bold")
    fig.text(4.50 / Wd, 1.0, "c  Fill-Interior vs. real, per observer", ha="left",
             va="top", fontsize=8, color=INK, fontweight="bold")
    fig.add_artist(matplotlib.lines.Line2D([2.20 / Wd] * 2, [0.03, 0.92], color=RULE,
                                           lw=0.6))
    fig.add_artist(matplotlib.lines.Line2D([4.43 / Wd] * 2, [0.03, 0.92], color=RULE,
                                           lw=0.6))
    draw_strip(fig, (4.50 / Wd, 0.34 / Ht, 2.25 / Wd, 1.40 / Ht), nums)
    print(f"  build {i}: exterior max channel diff {diff}; numbers {nums}")
    figstyle.save_all(fig, PAPER_DIR, "fig1_teaser", dpi=300, tight=False)
    plt.close(fig)


# --- the asset pack -------------------------------------------------------------

def save_png(im: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(im, "RGBA").save(path, optimize=True)


def assets() -> Dict[str, str]:
    i, real, filled, hollow = builds()
    tex = textures()
    arms = {"real": real, "filled": filled, "hollow": hollow}
    made: Dict[str, str] = {}

    def group(name, desc, specs, px):
        """Render every arm under every spec, cropped to one shared box per spec."""
        for tag, kw in specs:
            ims = [render(s, px=px, tex=tex, **kw) for s in arms.values()]
            ims = crop_alpha(ims)
            for (arm, _), im in zip(arms.items(), ims):
                p = ASSETS / name / f"{arm}_{tag}.png"
                save_png(im, p)
            made[f"{name}/<arm>_{tag}.png"] = desc.format(**kw, tag=tag)

    group("exterior", "paper's metric view, azimuth {az:g}, elevation {el:g}, orthographic",
          [(f"az{a:03d}_el30", dict(az=a, el=30)) for a in (45, 135, 225, 315)], 1200)
    group("hero", "isometric hero shot, azimuth {az:g}, elevation {el:g}",
          [("az030_el25", dict(az=30, el=25)), ("az060_el25", dict(az=60, el=25)),
           ("az045_el35", dict(az=45, el=35))], 2000)
    group("cutaway_roofoff", "lower half kept (roof off), azimuth {az:g}, elevation {el:g}",
          [(f"az{a:03d}_el60", dict(az=a, el=60, cut="half:y:lo")) for a in (45, 225)], 1600)
    group("cutaway_quadrant", "facing quadrant removed, azimuth {az:g}, elevation {el:g}",
          [("az045_el30", dict(az=45, el=30, cut="quad:hi:hi")),
           ("az225_el30", dict(az=225, el=30, cut="quad:lo:lo"))], 1600)
    group("floor_slabs", "one-voxel floor slab at {tag}, azimuth 45, elevation 75",
          [(f"slab{p:02d}", dict(az=45, el=75, cut=f"slab:y:{p}:1")) for p in (20, 40, 60, 80)],
          1400)

    # Vector sections: same planes for every arm, picked on the real build.
    z, y = best_planes(real)
    sec = ASSETS / "sections"
    sec.mkdir(parents=True, exist_ok=True)
    for arm, s in arms.items():
        occ = s.occupied_mask
        inter = interior(occ) if arm != "filled" else np.zeros_like(occ)
        # vertical section: fix z, show (y up, x across)
        section_svg(occ[:, :, z].T, inter[:, :, z].T, sec / f"{arm}_vertical_z{z}.svg")
        # floor plan: fix y, show (z down, x across) from above
        section_svg(occ[:, y, :].T, inter[:, y, :].T, sec / f"{arm}_plan_y{y}.svg", flip_y=False)
    made[f"sections/<arm>_vertical_z{z}.svg"] = "vertical section (vector): layer 'blocks' and layer 'interior_air'"
    made[f"sections/<arm>_plan_y{y}.svg"] = "horizontal floor plan from above (vector), same layers"

    # Top-down occupancy silhouettes as transparent PNG masks.
    for arm, s in arms.items():
        occ = s.occupied_mask
        foot = occ.any(axis=1).T
        inter = interior(occ)
        im = np.zeros(foot.shape + (4,), np.uint8)
        im[foot] = (*[int(255 * c) for c in matplotlib.colors.to_rgb(SOLID)], 255)
        sealed = inter[:, y, :].T
        im[sealed] = (*[int(255 * c) for c in matplotlib.colors.to_rgb(VOID)], 255)
        im = np.kron(im, np.ones((24, 24, 1), np.uint8))
        save_png(im, ASSETS / "plan_masks" / f"{arm}_footprint_plan_y{y}.png")
    made[f"plan_masks/<arm>_footprint_plan_y{y}.png"] = (
        "top-down footprint (grey) with the sealed air of floor y as peach, 24 px per block")

    # Metric strip on its own, as editable SVG/PDF.
    figstyle.apply(size=8)
    fig = plt.figure(figsize=(2.4, 1.75))
    draw_strip(fig, (0.0, 0.20, 0.95, 0.76), numbers())
    figstyle.save_all(fig, ASSETS, "metric_strip", dpi=300, pad=0.04)
    plt.close(fig)
    made["metric_strip.{svg,pdf,png}"] = "the metric strip alone; SVG text is live <text>"

    (ASSETS / "manifest.json").write_text(json.dumps(
        {"build_index": i, "section_z": z, "plan_y": y, "files": made,
         "numbers": numbers()}, indent=2))
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--assets", action="store_true")
    a = ap.parse_args()
    both = not (a.figure or a.assets)
    if a.figure or both:
        figure()
    if a.assets or both:
        for k, v in assets().items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
