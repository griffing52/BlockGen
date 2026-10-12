"""Which individual builds push an arm away from real, and which look most real.

KID is a *distribution* distance: it scores a set, not a build, so "this build
scores 0.31" is not a thing the metric says. But the unbiased estimator is a sum
over builds, and that sum decomposes exactly. Dropping the reference self-term,
which is constant across arms scored against one reference,

    kid_arm_term(X, R) / scale  =  (1/m) * sum_i c_i

    c_i = (1/(m-1)) * sum_{i' != i} k(x_i, x_i')  -  2 * mean_j k(x_i, r_j)

so `c_i` is build i's exact contribution to its arm's score. A build with a
large `c_i` is one the metric is unhappy about: either it sits far from the
reference cloud (the second term is small) or it is crowded together with other
arm members (the first term is large), which is what mode collapse looks like
from inside the statistic.

This is the Hajek influence that `compare.py` derives when explaining why a
subsampling rescale cannot work; here it is used for attribution rather than for
variance.

Two things it is NOT:

* **Not a per-build quality score.** `c_i` is defined relative to one arm and
  one reference draw. The same build in a different arm gets a different number,
  and that is correct: the statistic is about a distribution.
* **Not comparable across arms of different size**, for the same reason the
  noise floor is not comparable across runs.

Usage:
    .venv/bin/python scripts/build_influence.py --arm real_test --top 8
    .venv/bin/python scripts/build_influence.py --arm real@solidify --render
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCALE = 1000.0
DEGREE = 3
COEF0 = 1.0


def influence(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-row contribution `c_i`, scaled like `kid`. Mean == kid_arm_term."""
    x = np.asarray(x, np.float64)
    ref = np.asarray(ref, np.float64)
    m = len(x)
    if m < 2:
        raise ValueError("need at least two builds in the arm")
    g = 1.0 / x.shape[1]
    kxx = (g * (x @ x.T) + COEF0) ** DEGREE
    kxr = (g * (x @ ref.T) + COEF0) ** DEGREE
    within = (kxx.sum(axis=1) - np.diag(kxx)) / (m - 1)
    return SCALE * (within - 2.0 * kxr.mean(axis=1))



def _contact_sheet(arm, idx, dev, order, man, rows, args) -> None:
    """Render the two extremes for human review.

    **Internal review only.** This sheet deliberately shows every corpus,
    GrabCraft included, because the question it answers is whether the quality
    gate let something through, and that question cannot be asked of a filtered
    view. Nothing derived from it may be published without re-applying the
    3D-Craft gate that `scripts/paper_figures.py` uses for every figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from blockgen.renderer.textured import render_structure
    from blockgen.renderer.textures import load_face_textures

    # The metric's own renderer and texture pack: these are the images the
    # influence scores were computed from. Internal review only, as above.
    tex = load_face_textures()
    k = args.top
    per_row = args.per_row

    def shot(st):
        """Render, flatten onto white, and crop to the build's own ink.

        The renderer centres a build in a square frame, so most of every tile is
        empty. Cropping to content is what lets the sheet be packed: it roughly
        doubles the size each build is drawn at for the same page area.
        """
        img = render_structure(st, px=460, azim_deg=45, elev_deg=28,
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

    def section(st):
        """The vertical plane through the build showing the most rooms."""
        from scipy import ndimage as _n
        occ = st.occupied_mask
        filled = _n.binary_fill_holes(
            occ, structure=_n.generate_binary_structure(3, 1))
        inter = filled & ~occ
        best = (-1, -1, occ.shape[1] // 2)
        for zi in range(occ.shape[1]):
            pl = inter[:, zi, :]
            if pl.sum() < 8:
                continue
            r = int(_n.label(pl)[1])
            if (r, int(pl.sum())) > (best[0], best[1]):
                best = (r, int(pl.sum()), zi)
        z = best[2]
        o, it = occ[:, z, :], inter[:, z, :]
        canvas = np.ones((*o.shape, 3))
        canvas[o] = (0.30, 0.30, 0.29)
        canvas[it] = (0.98, 0.71, 0.55)
        return canvas, max(best[0], 0)

    groups = [("MOST real-like", list(order[:k]), "#1e6f4c"),
              ("LEAST real-like", list(order[::-1][:k]), "#b03a2e")]
    rows_per_group = int(np.ceil(k / per_row))
    # One empty row between the groups, so the second group's heading has
    # somewhere to sit. Without it the heading lands on top of the last row of
    # the first group.
    n_rows = 2 * rows_per_group + 1
    spacer = rows_per_group

    fig, axes = plt.subplots(n_rows, per_row,
                             figsize=(1.55 * per_row, 1.72 * n_rows),
                             squeeze=False,
                             gridspec_kw={"height_ratios":
                                          [1] * rows_per_group + [0.28]
                                          + [1] * rows_per_group})
    fig.subplots_adjust(left=0.004, right=0.996, top=0.955, bottom=0.008,
                        wspace=0.02, hspace=0.30)
    for ax in axes.ravel():
        ax.axis("off")

    for g, (title, js, colour) in enumerate(groups):
        base = g * (rows_per_group + 1)
        for n_i, j in enumerate(js):
            ax = axes[base + n_i // per_row][n_i % per_row]
            b = int(idx[j])
            if args.sections:
                from PIL import Image as _I
                sec, n_rooms = section(arm[j])
                img = shot(arm[j])
                h = img.shape[0]
                w = max(int(h * sec.shape[1] / sec.shape[0]), 1)
                sec_img = np.asarray(
                    _I.fromarray((sec * 255).astype(np.uint8))
                      .resize((w, h), _I.NEAREST))
                gap = np.full((h, max(h // 40, 2), 3), 255, np.uint8)
                ax.imshow(np.hstack([img, gap, sec_img]))
                plural = "s" if n_rooms != 1 else ""
                ax.set_xlabel(f"[{b}] - {n_rooms} room{plural}",
                              fontsize=6, color="#777", labelpad=1)
            else:
                ax.imshow(shot(arm[j]))
                ax.set_xlabel(f"[{b}]", fontsize=6, color="#777", labelpad=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{dev[j]:+.2f}", fontsize=8, fontweight="bold",
                         color=colour, pad=1.5)
            ax.xaxis.set_label_position("bottom")
            ax.axis("on")
            for sp in ax.spines.values():
                sp.set_visible(False)
        axes[base][0].text(0.0, 1.22, title, transform=axes[base][0].transAxes,
                           fontsize=11, fontweight="bold", color=colour,
                           ha="left", va="bottom", zorder=10)

    fig.suptitle(f"arm {args.arm}, {args.metric} metric: contribution to KID "
                 f"(deviation from the arm mean); textured, internal review only",
                 fontsize=8.5, y=0.998)
    out = (ROOT / "outputs" / "figstage"
           / f"influence_{args.metric}_{args.arm.replace('@','_')}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white",
                pad_inches=0.06)
    plt.close(fig)
    print(f"wrote {out}  (internal review only: shows all corpora)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", default="real_test")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--metric", choices=("render", "geometry"),
                    default="render", help="which metric to rank by")
    ap.add_argument("--sections", action="store_true",
                    help="draw each build's cross-section beside it")
    ap.add_argument("--per-row", type=int, default=10,
                    help="builds per row in the contact sheet")
    ap.add_argument("--render", action="store_true",
                    help="render the extreme builds to outputs/figstage/")
    args = ap.parse_args()

    from blockgen.eval.bench import splits, probes, features as ft, distances as D

    split = splits.load_split("houses_32")
    test = [s.crop_to_non_air() for s in splits.split_structures(split, "test")]
    ref_s = [s.crop_to_non_air() for s in splits.split_structures(split, "val")]

    idx = np.random.default_rng(1).permutation(len(test))[:args.n]
    arm = [test[i] for i in idx]
    if args.arm == "real@solidify":
        arm = [probes.solidify(s) for s in arm]
    elif args.arm not in ("real_test",):
        raise SystemExit(f"unknown arm {args.arm!r}; this tool covers the real arms")

    if args.metric == "render":
        print(f"embedding {len(arm)} arm and {len(ref_s)} reference builds ...",
              flush=True)
        view = ft.ViewConfig()
        fa = ft.pooled(ft.embed_views(arm, view, verbose=False))
        fr = ft.pooled(ft.embed_views(ref_s, view, verbose=False))
    else:
        # geom_kid IS distances.kid on the 77-d descriptor, so the identical
        # decomposition applies. Ranking by it asks the complementary question:
        # which builds are structurally unusual rather than unusual-looking.
        from blockgen.eval.bench import geometry as geom
        print("computing geometric descriptors ...", flush=True)
        std = geom.Standardizer.fit(geom.descriptor(ref_s))
        fa = geom.geom_features(arm, std)
        fr = geom.geom_features(ref_s, std)

    c = influence(fa, fr)

    # the decomposition must be exact, or the attribution means nothing
    got, want = float(c.mean()), D.kid_arm_term(fa, fr)
    rel = abs(got - want) / max(abs(want), 1e-12)
    print(f"\ncheck: mean(c_i) = {got:.9f}  kid_arm_term = {want:.9f}  "
          f"rel.err = {rel:.2e}")
    if rel > 1e-9:
        print("*** decomposition does not reconcile; attribution is not valid ***")
        return 1

    order = np.argsort(c)
    man = json.loads((ROOT / "data" / "minecraft" / "cache"
                      / "houses_32_manifest.json").read_text())
    rows = list(split.indices("test"))

    # Every c_i carries the same large offset (the within-arm kernel term is
    # near-constant for unit-norm features), so the absolute value is unreadable
    # and meaningless on its own. What attributes is the deviation from the arm
    # mean: how much more, or less, this build contributes than a typical member.
    dev = c - c.mean()

    def describe(rank: int, j: int) -> str:
        b = int(idx[j])
        item = man["items"][rows[b]]
        return (f"  {rank:>3}. {dev[j]:+8.4f}  test[{b:>3}]  "
                f"{item.get('corpus','?'):<9} blocks={int(arm[j].occupied_mask.sum()):>5}  "
                f"{(item.get('title') or '')[:42]}")

    print(f"\nDeviation from the arm mean; the arm total is {want:.4f}.")
    print(f"\nMOST real-like in arm {args.arm!r} (pulls the score down):")
    for r, j in enumerate(order[:args.top], 1):
        print(describe(r, j))
    print(f"\nLEAST real-like (pushes the score up):")
    for r, j in enumerate(order[::-1][:args.top], 1):
        print(describe(r, j))

    if args.render:
        _contact_sheet(arm, idx, dev, order, man, rows, args)

    out = ROOT / "outputs" / "figstage" / f"influence_{args.metric}_{args.arm.replace('@','_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "arm": args.arm, "n": len(arm), "n_ref": len(fr),
        "kid_arm_term": want, "mean_influence": got,
        "builds": [{"test_index": int(idx[j]), "influence": float(c[j]), "deviation": float(dev[j]),
                    "corpus": man["items"][rows[int(idx[j])]].get("corpus"),
                    "blocks": int(arm[j].occupied_mask.sum())}
                   for j in order],
    }, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
