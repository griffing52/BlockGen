"""Figure: the interior blindness is in the VIEW PROTOCOL, not the encoder.

One held-out build and its `solidify()` twin, rendered twice: under the
published exterior protocol (left) and under the cutaway protocols of the view
ablation (right). Encoder (DINOv2-B), statistic (unbiased polynomial KID),
builds and reference set are identical across the panel; only what the camera
is allowed to see changes. The left pair is indistinguishable and the metric
says so (+1.1 noise floors, inside the noise); the right pairs are obviously
different and the same metric says that too (+9.1 to +56.5 noise floors).

Every number is read at run time out of

    outputs/run_20260925_003516_view_protocol_ablation/ablation.json

and nothing is transcribed by hand. The camera/cut configurations used for the
pictures are rebuilt from `scripts/ablate_view_protocol.conditions()`'s recipe
and their `ViewConfig.key()` is asserted equal to the `view` key recorded in
that JSON, so the figure cannot drift from the condition it labels.

    .venv/bin/python scripts/figs/cutaway.py

Writes outputs/figstage/cutaway.png and .pdf.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from armnames import display          # noqa: E402
import figstyle                        # noqa: E402
from figstyle import ORANGE, ORANGE_DEEP, INK, INK_2, PAPER, RULE  # noqa: E402
RUN = ROOT / "outputs" / "run_20260925_003516_view_protocol_ablation"
OUTDIR = ROOT / "outputs" / "figstage"


def _pin_hash_seed() -> None:
    """Re-exec under PYTHONHASHSEED=0 so the flat block colours are stable.

    `blockgen.utils.data._hashed_color` seeds a block's fallback colour with
    Python's builtin `hash()` of its resource name, and CPython salts string
    hashing per process. Every flat-textured render therefore comes out in a
    different palette on every run -- which also means the measured pixel
    difference printed on this figure moves between runs. Pinning the seed for
    this script alone fixes both without touching the renderer, which is not
    this agent's file.
    """
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execve(sys.executable, [sys.executable, *sys.argv],
                  {**os.environ, "PYTHONHASHSEED": "0"})

# Colour and font come from scripts/figs/figstyle.py.
#
# This panel is entirely the RENDER tier: both halves are DINOv2 on rendered
# views, and they differ only in what the camera may see. So both halves are
# coloured from the render hue -- ORANGE for the published exterior protocol,
# ORANGE_DEEP for the same renderer cut open -- which is exactly how `pipeline`
# colours its rows 1 and 2, the same two conditions. The BLUE this figure used
# for the cutaway half is the paper's GEOMETRY-tier colour, and nothing in this
# figure is the geometry tier; it made a render-protocol result look like a
# descriptor result. `DEEMPH` (a light grey, used for the exterior +1.1) is gone:
# the exterior number is now drawn in its own tier colour like every other
# number in the paper, and the reading "inside the noise" is carried by the
# confidence interval beneath it and by the caption.

#: Flat per-block colours, never Minecraft's texture files -- an empty pack
#: directory makes `FaceTextureLibrary.__missing__` fall back to a per-block
#: solid colour while keeping the baked face shading.
_FLAT_DIR = tempfile.mkdtemp(prefix="blockgen-flat-")


def flat_textures():
    from blockgen.renderer.textures import load_face_textures
    return load_face_textures()


def _style() -> None:
    figstyle.apply(size=8)


def _corpus_mask(n: int, corpus: str = "3dcraft") -> List[bool]:
    """Which test-split rows may be published, by upstream source.

    Only 3D-Craft builds are publishable as images: GrabCraft's terms name
    pictures of blueprints as a prohibited form of distribution, and text2mc's
    builds originate from PlanetMinecraft, whose terms prohibit redistributing
    site data. 3D-Craft comes from an MIT-licensed research release.
    """
    man = json.loads((ROOT / "data" / "minecraft" / "cache"
                      / "houses_32_manifest.json").read_text())
    from blockgen.eval.bench import splits as _sp
    rows = list(_sp.load_split("houses_32").indices("test"))
    return [man["items"][rows[i]].get("corpus") == corpus for i in range(n)]


# --- the conditions the pictures illustrate ---------------------------------
# Rebuilt from the ablation's own recipe. `_check` asserts each ViewConfig key
# matches the key recorded beside the number in ablation.json, so a picture can
# never be labelled with a separation measured under a different camera.

def _conditions():
    from blockgen.eval.bench import features as ft
    quad_for_azim = {45.0: "quad:hi:hi", 135.0: "quad:hi:lo",
                     225.0: "quad:lo:lo", 315.0: "quad:lo:hi"}
    base = ft.ViewConfig()
    return {
        "dinov2b / 4 ortho (paper)": base,
        "cut: facing quadrant": ft.ViewConfig(
            cuts=tuple(quad_for_azim[a] for a, _ in ft.DEFAULT_VIEWS)),
        "cut: roof off (top half)": ft.ViewConfig(
            views=tuple((a, 60.0) for a, _ in ft.DEFAULT_VIEWS),
            cuts=("half:y:lo",) * 4),
        "cut: floorplan slabs": ft.ViewConfig(
            views=tuple((45.0, 75.0) for _ in range(4)),
            cuts=tuple(f"slab:y:{p}:1" for p in (20, 40, 60, 80))),
    }


def _rows() -> Tuple[Dict[str, dict], dict]:
    card = json.loads((RUN / "ablation.json").read_text())
    return {r["condition"]: r for r in card["rows"]}, card["meta"]


def _check(cfgs, rows) -> None:
    for name, cfg in cfgs.items():
        if cfg.key() != rows[name]["view"]:
            raise SystemExit(
                f"view config for {name!r} does not match the recorded run: "
                f"rebuilt {cfg.key()!r} vs recorded {rows[name]['view']!r}")


# --- rendering ---------------------------------------------------------------

def _flatten(img: np.ndarray) -> np.ndarray:
    if img.shape[-1] == 4:
        a = img[..., 3:4].astype(np.float32) / 255.0
        img = (img[..., :3] * a + 255.0 * (1 - a)).astype(np.uint8)
    return img[..., :3]


def _crop_pair(a: np.ndarray, b: np.ndarray, pad: int = 6):
    """Crop both renders to the SAME box, so the pair stays at one scale.

    Cropping each to its own content would silently resize one building
    relative to the other and undercut the "identical outside" claim.
    """
    ink = ((a < 250).any(axis=2)) | ((b < 250).any(axis=2))
    if not ink.any():
        return a, b
    ys, xs = np.where(ink)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, a.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, a.shape[1])
    return a[y0:y1, x0:x1], b[y0:y1, x0:x1]


def _pad_to_aspect(img: np.ndarray, aspect: float) -> np.ndarray:
    """Letterbox onto the figure surface at a fixed height/width ratio.

    Every tile then has the same shape, so the grid aligns and no build is
    scaled relative to another by the layout.
    """
    h, w = img.shape[:2]
    if h / w > aspect:
        H, W = h, int(round(h / aspect))
    else:
        H, W = int(round(w * aspect)), w
    bg = np.array(                             # the page, so the tile has no edge
        [round(255 * c) for c in matplotlib.colors.to_rgb(PAPER)], np.uint8)
    out = np.tile(bg, (H, W, 1))
    y0, x0 = (H - h) // 2, (W - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img
    return out


def _render(struct, cfg, j: int, px: int, tex):
    """View `j` of condition `cfg`, with its cut applied. Flat colours."""
    from blockgen.eval.bench import features as ft
    from blockgen.renderer.textured import render_structure
    shown = ft.apply_cut(struct, cfg.cuts[j]) if cfg.cuts else struct
    azim, elev = cfg.views[j]
    return _flatten(render_structure(shown, px=px, azim_deg=azim,
                                     elev_deg=elev, ortho=cfg.ortho,
                                     face_textures=tex))


def main() -> None:
    from blockgen.eval.bench import splits, probes
    from scipy import ndimage

    rows, meta = _rows()
    cfgs = _conditions()
    _check(cfgs, rows)

    split = splits.load_split("houses_32", meta["seed"])
    test = [s.crop_to_non_air() for s in splits.split_structures(split, "test")]

    def interior_cells(s) -> int:
        occ = s.occupied_mask
        filled = ndimage.binary_fill_holes(
            occ, structure=ndimage.generate_binary_structure(3, 1))
        return int((filled & ~occ).sum())

    # Selection rule: the most hollow build in the test split -- a build with no
    # enclosed air is not a test of anything -- restricted to the corpus whose
    # licence permits publishing pictures. The rule is unchanged, only the pool
    # it draws from is narrowed.
    allowed = _corpus_mask(len(test))
    scored = sorted(((interior_cells(s), i) for i, s in enumerate(test)
                     if allowed[i]), reverse=True)
    n_int, idx = scored[0]
    real = test[idx]
    sol = probes.solidify(real)
    print(f"  build {idx} (3dcraft): {n_int} interior cells, "
          f"{int(real.occupied_mask.sum())} -> {int(sol.occupied_mask.sum())} blocks "
          f"(+{100 * (sol.occupied_mask.sum() - real.occupied_mask.sum()) / real.occupied_mask.sum():.0f}%)")

    tex = flat_textures()
    PX = 480

    #: (condition, view index, column title). Two exterior views of the paper's
    #: four-view protocol, then one view from each of three cutaway conditions.
    cols = [("dinov2b / 4 ortho (paper)", 0, "view 1 of 4"),
            ("dinov2b / 4 ortho (paper)", 1, "view 2 of 4"),
            ("cut: facing quadrant", 0, "facing quadrant"),
            ("cut: roof off (top half)", 0, "roof off"),
            ("cut: floorplan slabs", 2, "floor plan\n(1-voxel slab)")]

    ASPECT = 0.80                    # tile height / width, every tile alike
    tiles, pixdiff = [], []
    for name, j, _ in cols:
        a = _render(real, cfgs[name], j, PX, tex)
        b = _render(sol, cfgs[name], j, PX, tex)
        # Measured, not asserted: how different the two renders are in PIXELS,
        # on the FULL uncropped frame so the number cannot move when the layout
        # does. It is the wrong question -- the metric is KID on DINOv2
        # features, and an earlier analysis rejected the cutaway on exactly this
        # number -- but the exterior pair should also *look* the same, and this
        # says by how much for this build.
        pixdiff.append(float(np.abs(a.astype(np.float32)
                                    - b.astype(np.float32)).mean()))
        a, b = _crop_pair(a, b)
        tiles.append((_pad_to_aspect(a, ASPECT), _pad_to_aspect(b, ASPECT)))
    for (name, j, _), d in zip(cols, pixdiff):
        print(f"  mean |pixel diff| real vs solidify, {name} view {j}: {d:.2f}/255")

    # --- layout --------------------------------------------------------------
    # Laid out top down in figure fractions so nothing can collide: every band
    # is placed relative to the one above it.
    _style()
    FW, FH = 5.5, 2.99
    fig = plt.figure(figsize=(FW, FH))

    tw = 0.1655                      # tile width, figure fraction
    th = tw * FW / FH * ASPECT       # matching tile height
    gap, group_gap, x0 = 0.0125, 0.050, 0.068
    xs, x = [], x0
    for k in range(5):
        if k == 2:
            x += group_gap - gap
        xs.append(x)
        x += tw + gap

    def dy(inches: float) -> float:
        """A vertical distance in inches, as a fraction of this figure.

        Every band below the tiles is placed this way so that shortening the
        figure moves the bands up together instead of squeezing the gaps
        between them.
        """
        return inches / FH

    y_title, y_sub, y_rule = 0.970, 0.919, 0.890
    y_top = 0.790 - th               # first image row, titles grow up from it
    y_bot = y_top - dy(0.062) - th   # second image row
    y_pix = y_bot - dy(0.103)        # measured pixel difference, per group
    y_rule2 = y_bot - dy(0.213)
    y_num, y_ci, y_note = (y_rule2 - dy(0.25), y_rule2 - dy(0.44),
                           y_rule2 - dy(0.56))

    for k, ((a, b), (name, j, title)) in enumerate(zip(tiles, cols)):
        for y, img in ((y_top, a), (y_bot, b)):
            ax = fig.add_axes([xs[k], y, tw, th])
            ax.imshow(img, interpolation="antialiased")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(RULE); sp.set_linewidth(0.5)
        # anchored to the tile top and growing upward, so a two-line title
        # never pushes into the images
        fig.text(xs[k] + tw / 2, y_top + th + 0.013, title, ha="center",
                 va="bottom", fontsize=7.2, linespacing=1.35,
                 color=INK_2 if k < 2 else INK)

    # row labels
    for y, lab, weight in ((y_top, display("real_test"), "normal"),
                           (y_bot, display("real@solidify"), "bold")):
        fig.text(x0 - 0.014, y + th / 2, lab, ha="right", va="center",
                 fontsize=8, rotation=90, color=INK, fontweight=weight)

    # --- group headers -------------------------------------------------------
    gl0, gl1 = xs[0], xs[1] + tw
    gr0, gr1 = xs[2], xs[4] + tw
    fig.text((gl0 + gl1) / 2, y_title, "exterior views", ha="center",
             va="center", fontsize=9.5, color=ORANGE, fontweight="bold")
    fig.text((gl0 + gl1) / 2, y_sub, "the published protocol", ha="center",
             va="center", fontsize=7.4, color=INK_2, style="italic")
    fig.text((gr0 + gr1) / 2, y_title, "cutaway views", ha="center",
             va="center", fontsize=9.5, color=ORANGE_DEEP, fontweight="bold")
    for x_a, x_b, c, w in ((gl0, gl1, ORANGE, 1.1), (gr0, gr1, ORANGE_DEEP, 1.3)):
        fig.add_artist(plt.Line2D([x_a, x_b], [y_rule, y_rule], color=c, lw=w))

    # measured pixel difference of each pair, under its group
    fig.text((gl0 + gl1) / 2, y_pix,
             f"pixel difference {pixdiff[0]:.2f}/255", ha="center",
             va="center", fontsize=6.9, color=INK_2)
    fig.text((gr0 + gr1) / 2, y_pix,
             "pixel difference "
             + " / ".join(f"{d:.1f}" for d in pixdiff[2:]) + " /255",
             ha="center", va="center", fontsize=6.9, color=INK_2)

    # --- the numbers ---------------------------------------------------------
    def sep(name: str) -> Tuple[float, float, float]:
        r = rows[name]
        return (r["separation_sd"], *r["separation_ci"])

    v, lo, hi = sep("dinov2b / 4 ortho (paper)")
    fig.text((gl0 + gl1) / 2, y_num, f"+{v:.1f}", ha="center", va="center",
             fontsize=16, color=ORANGE, fontweight="bold")
    fig.text((gl0 + gl1) / 2, y_ci, f"[{lo:.1f}, {hi:.1f}]",
             ha="center", va="center", fontsize=7.2, color=INK_2)
    fig.text((gl0 + gl1) / 2, y_note, "noise floors",
             ha="center", va="center", fontsize=7.6, color=INK_2,
             style="italic")

    for k in (2, 3, 4):
        v, lo, hi = sep(cols[k][0])
        fig.text(xs[k] + tw / 2, y_num, f"+{v:.1f}", ha="center", va="center",
                 fontsize=16, color=ORANGE_DEEP, fontweight="bold")
        fig.text(xs[k] + tw / 2, y_ci, f"[{lo:.1f}, {hi:.1f}]", ha="center",
                 va="center", fontsize=7.2, color=INK_2)
    fig.text((gr0 + gr1) / 2, y_note, "noise floors", ha="center",
             va="center", fontsize=7.6, color=INK_2, style="italic")

    # a hairline linking the numbers to their group
    for x_a, x_b, c in ((gl0, gl1, ORANGE), (gr0, gr1, ORANGE_DEEP)):
        fig.add_artist(plt.Line2D([x_a, x_b], [y_rule2, y_rule2], color=c,
                                  lw=0.8, alpha=0.6))

    # The setup line (encoder, statistic, n) and the reading of the
    # pixel-difference numbers both live in the caption, so neither is repeated
    # here; the figure keeps only the measurements themselves.

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"cutaway.{ext}", dpi=300, bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {OUTDIR.relative_to(ROOT)}/cutaway.pdf (+ .png)")


if __name__ == "__main__":
    _pin_hash_seed()
    main()
