"""The nearest-neighbour double dissociation, shown on three houses.

    BLOCKGEN_FIG_RENDER=cpu .venv/bin/python scripts/figs/nn_dissociation.py \
        --out deliverables/paper/cvpr27/figures_new

One row per example: a Fill-Interior (`real@solidify`) house | its nearest
held-out real house in MV-KID's DINO feature space (always its own source) |
its nearest held-out real house in V-KID's descriptor space. A cross-section
under each render shows sealed interior air, so the fill is visible.

Where everything comes from
---------------------------
* Features: `features.npz` of the n=256 leaderboard of record, written by the
  bench (`--save-features`) exactly as the two KIDs see them (whitened /
  unit-norm). Distances are Euclidean on those vectors, the same as
  `outputs/run_20261103_185719_nn_geom_vs_dino/nn_analysis.py` (notes §71).
* Gallery: the run's 256 `real_test` builds. The probe arm is index-aligned with
  `real_test` (probe[i] = solidify(real_test[i])), so "its own source" is row i.
* Builds: rebuilt with `fast.control_arms` under the run's seed, and checked
  against the scorecard's `structures_sha` for both arms before anything is
  drawn, so the pictures are the scored builds.

Selection rule (no build chosen by eye)
---------------------------------------
Licensing: only 3D-Craft builds may be published as images, so a row is
eligible only when the query's source AND its V-KID neighbour are 3D-Craft
(`vlm_judge.split_sources`; real_test row i is test-split row perm[i] of
control_arms' draw). Among eligible rows whose DINO neighbour is the source and
whose V-KID neighbour is not (the dissociation itself), the three with the
largest fraction of blocks added by the probe are shown, since that is where
the fill has most to show. Counts are printed, so the share of eligible rows
is on record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
import cpu_render                     # noqa: E402
cpu_render.install()
import figstyle                       # noqa: E402
from figstyle import INK, INK_2, SOLID, VOID  # noqa: E402

RUN = ROOT / "outputs" / "run_20261103_090315_bench_n_256_leaderboard_of_record"
PROBE = "real@solidify"
K = 3


def _dist(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(2.0 - 2.0 * x @ g.T, 0.0))


def rebuild(card: dict):
    """real_test and the probe arm as the run scored them, sha-checked."""
    from blockgen.eval.bench import fast, splits
    from blockgen.eval.bench import features as ft
    ctx = card["context"]
    corpus, seed = ctx.get("corpus", "houses_32"), int(ctx.get("seed", 0))
    n = int(card["arms"]["real_test"]["meta"]["n"])
    split = splits.load_split(corpus, seed)
    train = splits.split_structures(split, "train")
    test = splits.split_structures(split, "test")
    arms = {a.name: [s.crop_to_non_air() for s in a.structures]
            for a in fast.control_arms(test, train=train, n=n,
                                       rng=np.random.default_rng(seed + 1),
                                       corpus=corpus, seed=seed)
            if a.name in ("real_test", PROBE)}
    for name, structs in arms.items():
        want = card["arms"][name]["meta"]["structures_sha"]
        got = ft.structures_sha(structs)
        if got != want:
            raise SystemExit(f"rebuilt {name} sha {got} != scored {want}")
    perm = np.random.default_rng(seed + 1).permutation(len(test))[:n]
    # control_arms' draw, checked build by build rather than assumed
    for i in range(n):
        if not np.array_equal(arms["real_test"][i].block_ids,
                              test[perm[i]].crop_to_non_air().block_ids):
            raise SystemExit(f"real_test[{i}] is not test row perm[{i}]")
    return arms, perm, corpus, seed


def select(card: dict):
    from blockgen.eval.bench.vlm_judge import split_sources, PUBLISHABLE
    z = np.load(RUN / "features.npz")
    gal = {sp: z[f"{sp}::real_test"].astype(np.float64) for sp in ("geom", "dino")}
    q = {sp: z[f"{sp}::{PROBE}"].astype(np.float64) for sp in ("geom", "dino")}
    d = {sp: _dist(q[sp], gal[sp]) for sp in ("geom", "dino")}
    nn = {sp: d[sp].argmin(1) for sp in d}
    own = {sp: d[sp][np.arange(len(q[sp])), np.arange(len(q[sp]))] for sp in d}
    rank = {sp: (d[sp] < own[sp][:, None]).sum(1) for sp in d}

    arms, perm, corpus, seed = rebuild(card)
    src = split_sources(corpus, seed)["test"]
    pub = np.array([src[p] == PUBLISHABLE for p in perm])
    n = len(pub)
    idx = np.arange(n)
    dissoc = (nn["dino"] == idx) & (nn["geom"] != idx)
    elig = dissoc & pub & pub[nn["geom"]]
    added = np.array([(arms[PROBE][i].block_ids != 0).sum()
                      / max((arms["real_test"][i].block_ids != 0).sum(), 1) - 1.0
                      for i in range(n)])
    picks = sorted(np.flatnonzero(elig), key=lambda i: -added[i])[:K]
    hub_src = {int(h): src[perm[h]] for h in np.argsort(-np.bincount(nn["geom"], minlength=n))[:4]}
    stats = {
        "run": RUN.name, "n": int(n),
        "dino_src_top1": float((nn["dino"] == idx).mean()),
        "geom_src_top1": float((nn["geom"] == idx).mean()),
        "geom_src_rank_median": float(np.median(rank["geom"])),
        "dissociated_rows": int(dissoc.sum()),
        "dissociated_and_publishable": int(elig.sum()),
        # V-KID sends filled houses to a few dense "hub" neighbours (notes §71:
        # the probe collapses into dense regions of the reference), which is
        # why the right-hand column can repeat one house.
        "geom_nn_distinct": int(len(np.unique(nn["geom"]))),
        "geom_nn_top4_hub_share": float(np.sort(np.bincount(nn["geom"], minlength=n))[::-1][:4].sum() / n),
        "geom_nn_hubs_top4": [[int(h), int(c)] for h, c in
                              sorted(enumerate(np.bincount(nn["geom"], minlength=n)),
                                     key=lambda hc: -hc[1])[:4]],
        "geom_nn_hub_sources": hub_src,
        "picks": [{"real_test_row": int(i), "test_split_row": int(perm[i]),
                   "geom_nn_real_test_row": int(nn["geom"][i]),
                   "geom_nn_test_split_row": int(perm[nn["geom"][i]]),
                   "source_rank_in_geom": int(rank["geom"][i]),
                   "source_rank_in_dino": int(rank["dino"][i]),
                   "blocks_added_frac": float(added[i])} for i in picks],
    }
    return arms, picks, nn, stats


def shot(s, tex, px=300):
    from blockgen.renderer.textured import render_structure
    img = render_structure(s, px=px, azim_deg=45, elev_deg=25, ortho=True,
                           face_textures=tex)
    a = img[..., 3:4].astype(np.float32) / 255.0
    return (img[..., :3] * a + 255.0 * (1 - a)).astype(np.uint8)


def slab(s, y=None):
    """Horizontal section: blocks dark, sealed interior air peach."""
    from scipy import ndimage
    occ = s.crop_to_non_air().occupied_mask
    f = ndimage.binary_fill_holes(occ, structure=ndimage.generate_binary_structure(3, 1))
    inter = f & ~occ
    if y is None:
        per = inter.sum(axis=(1, 2))
        y = int(np.argmax(per)) if per.max() else occ.shape[0] // 2
    y = min(y, occ.shape[0] - 1)
    canvas = np.ones(occ.shape[1:] + (3,))
    canvas[occ[y]] = to_rgb(SOLID)
    canvas[inter[y]] = to_rgb(VOID)
    return canvas, y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "outputs" / "figstage"))
    args = ap.parse_args()
    card = json.loads((RUN / "scorecard.json").read_text())
    arms, picks, nn, stats = select(card)
    print(json.dumps(stats, indent=1))
    if len(picks) < K:
        raise SystemExit("not enough eligible rows")

    from blockgen.renderer.textures import load_face_textures
    tex = load_face_textures()          # the metric's own pack, as fig7
    figstyle.apply(size=8)
    cols = ["Fill-Interior", "MV-KID nearest\n(its own source)", "V-KID nearest"]
    fig, axs = plt.subplots(2 * K, 3, figsize=(figstyle.CVPR_COL, 3.9),
                            gridspec_kw={"height_ratios": [1, 0.55] * K,
                                         "hspace": 0.05, "wspace": 0.04})
    for r, i in enumerate(picks):
        q = arms[PROBE][i]
        src = arms["real_test"][i]
        gnn = arms["real_test"][int(nn["geom"][i])]
        _, y = slab(src)                 # the source's most-interior plane
        for c, (st, yy) in enumerate([(q, y), (src, y), (gnn, None)]):
            ax = axs[2 * r, c]
            ax.imshow(shot(st, tex))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=7.5, pad=3, linespacing=1.2,
                             color=INK if c else INK_2,
                             fontweight="normal" if c else "bold")
            ax2 = axs[2 * r + 1, c]
            ax2.imshow(slab(st, yy)[0], interpolation="nearest", origin="lower")
            ax2.set_xticks([]); ax2.set_yticks([])
            for sp in ax2.spines.values():
                sp.set_visible(False)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    figstyle.save_all(fig, args.out, "fig_nn_dissociation", dpi=300, pad=0.02)
    plt.close(fig)
    (ROOT / "outputs" / "figstage").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "figstage" / "nn_dissociation.json").write_text(
        json.dumps(stats, indent=1) + "\n")
    print(f"wrote {args.out}/fig_nn_dissociation.{{pdf,svg,png}}")


if __name__ == "__main__":
    main()
