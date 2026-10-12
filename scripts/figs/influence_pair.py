"""The two metrics disagree about individual real builds, not just about probes.

Every headline number in the paper comes from a corruption we introduced. This
figure makes the same point on real, uncorrupted builds and without any probe:
decompose each metric's KID into its exact per-build contributions and rank the
same corpus by each. The two orderings are nearly independent, and only one of
them carries information about interiors.

Rendered with the metric's own texture pack and restricted to 3D-Craft, per the
publication rules in `paper_figures.flat_textures`. Influence is computed on the
FULL arm, because it is a property of the arm; only the displayed builds are
filtered.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.build_influence import influence          # noqa: E402
import figstyle                                        # noqa: E402
from figstyle import BLUE, ORANGE, INK_2, PAPER        # noqa: E402

# Colour and font come from scripts/figs/figstyle.py; nothing is declared here.
# The two rows are the two metric tiers, so they take the two tier colours:
# ORANGE for the render metric, BLUE for the geometry metric, as everywhere
# else. The near-orange and near-blue this figure used to mix locally were the
# same two tiers in slightly different paint. Its third constant was a light
# grey for the per-build cell counts; those are text, so they are INK_2.

#: Staged next to the other figures. The paper's copy is installed from here by
#: the main loop, which is why this script no longer writes into
#: `deliverables/paper/iclr/figures/` itself.
OUTDIR = ROOT / "outputs" / "figstage"
K = 6


def main() -> int:
    figstyle.apply(size=8)
    from scipy import stats, ndimage
    from blockgen.eval.bench import splits, features as ft, geometry as geom
    from blockgen.renderer.textured import render_structure
    from blockgen.renderer.textures import load_face_textures

    sp = splits.load_split("houses_32")
    test = [s.crop_to_non_air() for s in splits.split_structures(sp, "test")]
    ref = [s.crop_to_non_air() for s in splits.split_structures(sp, "val")]
    idx = np.random.default_rng(1).permutation(len(test))[:128]
    arm = [test[i] for i in idx]

    man = json.loads((ROOT / "data" / "minecraft" / "cache"
                      / "houses_32_manifest.json").read_text())
    rows = list(sp.indices("test"))
    pub = np.array([man["items"][rows[int(b)]].get("corpus") == "3dcraft"
                    for b in idx])

    print("features ...", flush=True)
    fa = ft.pooled(ft.embed_views(arm, ft.ViewConfig(), verbose=False))
    fr = ft.pooled(ft.embed_views(ref, ft.ViewConfig(), verbose=False))
    c_r = influence(fa, fr)
    std = geom.Standardizer.fit(geom.descriptor(ref))
    c_g = influence(geom.geom_features(arm, std), geom.geom_features(ref, std))

    sealed = np.array([
        int((ndimage.binary_fill_holes(
            s.occupied_mask,
            structure=ndimage.generate_binary_structure(3, 1))
            & ~s.occupied_mask).sum()) for s in arm])

    rho_pair = stats.spearmanr(c_r, c_g)[0]
    rho_r, p_r = stats.spearmanr(c_r, sealed)
    rho_g, p_g = stats.spearmanr(c_g, sealed)
    print(f"rho(render,geom)={rho_pair:+.3f}  "
          f"rho(render,sealed)={rho_r:+.3f} p={p_r:.3g}  "
          f"rho(geom,sealed)={rho_g:+.3f} p={p_g:.3g}")

    tex = load_face_textures()   # the metric's own pack; see paper_figures.flat_textures

    def shot(st):
        img = render_structure(st, px=420, azim_deg=45, elev_deg=28,
                               ortho=True, face_textures=tex)
        if img.shape[-1] == 4:
            a = img[..., 3:4].astype(np.float32) / 255.0
            img = (img[..., :3] * a + 255.0 * (1 - a)).astype(np.uint8)
        img = img[..., :3]
        ink = (img < 248).any(axis=2)
        if ink.any():
            ys, xs = np.where(ink)
            img = img[max(ys.min() - 2, 0):ys.max() + 3,
                      max(xs.min() - 2, 0):xs.max() + 3]
        return img

    fig = plt.figure(figsize=(5.5, 2.80))
    gs = fig.add_gridspec(2, K, hspace=0.60, wspace=0.06,
                          left=0.055, right=0.995, top=0.845, bottom=0.035)

    for r, (c, colour, name, rho, pv) in enumerate((
            (c_r, ORANGE, "render metric", rho_r, p_r),
            (c_g, BLUE, "geometric metric", rho_g, p_g))):
        cand = [j for j in np.argsort(c)[::-1] if pub[j]][:K]
        for n_i, j in enumerate(cand):
            ax = fig.add_subplot(gs[r, n_i])
            ax.imshow(shot(arm[j]))
            ax.set_xticks([]); ax.set_yticks([])
            for spn in ax.spines.values():
                spn.set_visible(False)
            ax.set_title(f"{c[j]-c.mean():+.2f}", fontsize=7,
                         color=colour, fontweight="bold", pad=1.5)
            ax.set_xlabel(f"{sealed[j]} cells", fontsize=6.4, color=INK_2,
                          labelpad=1.5)
        fig.text(0.055, 0.900 - r * 0.468,
                 f"worst {K} by the {name}", fontsize=8.2,
                 color=colour, fontweight="bold", ha="left")
        fig.text(0.995, 0.900 - r * 0.468,
                 r"$\rho$ with sealed interior volume = "
                 f"{rho:+.3f}" + ("  (n.s.)" if pv > 0.05 else "  (p < 0.001)"),
                 fontsize=7.4, color=colour, ha="right")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"influence_pair.{ext}", dpi=300,
                    bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print(f"wrote {OUTDIR}/influence_pair.{{pdf,png}}")
    (OUTDIR / "influence_pair.json").write_text(json.dumps(
        {"rho_render_geom": float(rho_pair),
         "rho_render_sealed": float(rho_r), "p_render_sealed": float(p_r),
         "rho_geom_sealed": float(rho_g), "p_geom_sealed": float(p_g),
         "n": int(len(arm))}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
