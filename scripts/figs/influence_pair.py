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

import argparse
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
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None,
                    help="output directory; writes fig11_influence.{pdf,svg,png,json} "
                         "there (default: outputs/figstage/influence_pair.*)")
    ap.add_argument("--batch", type=int, default=16,
                    help="DINOv2 embedding batch (small: the GPU is shared)")
    ap.add_argument("--feat-cache", default=None,
                    help="npz to load/save the pooled DINOv2 features (debug)")
    args = ap.parse_args()
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
    cache = Path(args.feat_cache) if args.feat_cache else None
    if cache is not None and cache.exists():
        z = np.load(cache); fa, fr = z["fa"], z["fr"]
        print(f"  render features from cache {cache}")
    else:
        fa = ft.pooled(ft.embed_views(arm, ft.ViewConfig(), batch=args.batch,
                                      verbose=False))
        fr = ft.pooled(ft.embed_views(ref, ft.ViewConfig(), batch=args.batch,
                                      verbose=False))
        if cache is not None:
            np.savez(cache, fa=fa, fr=fr)
    c_r = influence(fa, fr)
    # The published figure used the 77-d descriptor of the run of record.
    # geometry.py has since appended three `walk_*` scalars (2026-10-23); drop
    # their raw columns before whitening so the standardiser and the unit-norm
    # match the 77-d features exactly (as scripts/figs/descriptor.py does).
    # BLOCKGEN_DESCRIPTOR_FULL=1 keeps the new 80-d descriptor.
    import os
    d_arm, d_ref = geom.descriptor(arm), geom.descriptor(ref)
    if os.environ.get("BLOCKGEN_DESCRIPTOR_FULL") != "1":
        layout = geom.descriptor_layout()
        sc0 = sum(w for name, w in layout if name != "scalars")
        drop = {sc0 + i for i, m in enumerate(geom.GEOMETRY_METRICS)
                if m.startswith("walk_")}
        keep = np.array([i for i in range(d_arm.shape[1]) if i not in drop])
        d_arm, d_ref = d_arm[:, keep], d_ref[:, keep]
    print(f"  geometry descriptor: {d_arm.shape[1]}-d", flush=True)
    std = geom.Standardizer.fit(d_ref)
    c_g = influence(std(d_arm), std(d_ref))

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

    # Authored at the CVPR full text width, so every size prints at its value.
    FW = figstyle.CVPR_FULL if args.out else 5.5
    fig = plt.figure(figsize=(FW, 3.05))
    gs = fig.add_gridspec(2, K, hspace=0.55, wspace=0.06,
                          left=0.01, right=0.99, top=0.865, bottom=0.04)

    rank_r = {int(j): k for k, j in enumerate(np.argsort(c_r)[::-1])}
    rank_g = {int(j): k for k, j in enumerate(np.argsort(c_g)[::-1])}
    shown = {}
    for r, (c, colour, name, rho, pv) in enumerate((
            (c_r, ORANGE, "MV-KID", rho_r, p_r),
            (c_g, BLUE, "V-KID", rho_g, p_g))):
        cand = [int(j) for j in np.argsort(c)[::-1] if pub[j]][:K]
        shown[name] = [{"j": j, "test_row": int(idx[j]),
                        "split_row": int(rows[int(idx[j])]),
                        "infl_render": float(c_r[j] - c_r.mean()),
                        "infl_geom": float(c_g[j] - c_g.mean()),
                        "rank_render": rank_r[j], "rank_geom": rank_g[j],
                        "sealed": int(sealed[j])} for j in cand]
        for n_i, j in enumerate(cand):
            ax = fig.add_subplot(gs[r, n_i])
            ax.imshow(shot(arm[j]))
            ax.set_xticks([]); ax.set_yticks([])
            for spn in ax.spines.values():
                spn.set_visible(False)
            ax.set_title(f"{c[j]-c.mean():+.2f}", fontsize=7.5,
                         color=colour, fontweight="bold", pad=1.5)
            ax.set_xlabel(f"{sealed[j]} cells", fontsize=7, color=INK_2,
                          labelpad=1.5)
        y_head = 0.935 - r * 0.475
        fig.text(0.01, y_head, f"{name}: worst {K}", fontsize=8.5,
                 color=colour, fontweight="bold", ha="left")
        fig.text(0.99, y_head,
                 "ρ vs sealed volume = "
                 f"{rho:+.3f}" + (" (n.s.)" if pv > 0.05 else " (p < 0.001)"),
                 fontsize=7.5, color=colour, ha="right")

    for name, items in shown.items():
        print(f"  {name} row:")
        for d in items:
            print("   ", d)
    both = sorted({d["j"] for d in shown["MV-KID"]}
                  & {d["j"] for d in shown["V-KID"]})
    print(f"  builds in both rows (arm index j): {both}")
    # identical-occupancy check among all shown builds (duplicate builds in split)
    allj = sorted({d["j"] for v in shown.values() for d in v})
    dups = [(a, b) for ai, a in enumerate(allj) for b in allj[ai + 1:]
            if arm[a].occupied_mask.shape == arm[b].occupied_mask.shape
            and (arm[a].occupied_mask == arm[b].occupied_mask).all()]
    print(f"  distinct-j pairs with identical occupancy: {dups}")

    stats_json = {"rho_render_geom": float(rho_pair),
                  "rho_render_sealed": float(rho_r), "p_render_sealed": float(p_r),
                  "rho_geom_sealed": float(rho_g), "p_geom_sealed": float(p_g),
                  "n": int(len(arm))}
    if args.out:
        out = Path(args.out)
        figstyle.save_all(fig, out, "fig11_influence", tight=True)
        plt.close(fig)
        (out / "fig11_influence.json").write_text(json.dumps(
            {**stats_json, "n_publishable": int(pub.sum()), "shown": shown,
             "in_both_rows": both, "identical_occupancy_pairs": dups}, indent=1))
        return 0

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"influence_pair.{ext}", dpi=300,
                    bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print(f"wrote {OUTDIR}/influence_pair.{{pdf,png}}")
    (OUTDIR / "influence_pair.json").write_text(json.dumps(stats_json, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
