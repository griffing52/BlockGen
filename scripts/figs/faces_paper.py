"""Draft figures for the cross-face decoding paper (deliverables/paper/cvpr27_faces).

    BLOCKGEN_FIG_RENDER=cpu .venv/bin/python scripts/figs/faces_paper.py [teaser|method|quality|cost|all]

teaser   Fig. 1: Grammar (free) faces with their disagreeing cells marked, their lift,
         Cross-Face faces of the same box and seed, their lift. Boxes are picked by a
         fixed rule (the first three test boxes at least 14 wide and deep), not by eye.
method   Fig. 2: a top-down slice of the shell state: what the front face committed,
         and which depths stay legal for one cell of the right face.
quality  Fig. 3: consistency, parts per building and MV-KID per decoder (T117, T122).
cost     (not in the paper: face cells against box volume; run explicitly with `cost`)

Run outputs and caches are read from BLOCKGEN_DATA_ROOT (default: this checkout), so
the script also works from a git worktree. Writes PDF + PNG to
deliverables/paper/cvpr27_faces/figures/ of this checkout.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("BLOCKGEN_DATA_ROOT", ROOT))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle  # noqa: E402
from figstyle import BLUE, ORANGE, INK  # noqa: E402

OUT = ROOT / "deliverables/paper/cvpr27_faces/figures"
RUN = DATA / "outputs/run_20261028_125240_houses32_facades_joint_d4_20m"
ARM_GRAMMAR = RUN / "constrained_none_t0.7_n399/faces.npz"
ARM_CROSS = RUN / "constrained_constrain_t0.7_n399/faces.npz"
GREY = "#8a8a86"
MUTED = "#c9c9c4"
RED = np.array([220, 30, 30], np.float32)
FACES = ("-z", "+x", "+z", "-x", "+y", "-y")


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def _bare(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------- face sets
def load_face_sets(path: Path):
    """faces.npz (keys '<i>_<face>_<depth|ids|data>') -> list of (shape, {face: Facade}) or None."""
    from blockgen.corpus import facade as fc

    z = np.load(path)
    n = 1 + max(int(k.split("_")[0]) for k in z.files)
    out = []
    for i in range(n):
        if f"{i}_-z_depth" not in z.files:
            out.append(None)
            continue
        X, Y = z[f"{i}_-z_depth"].shape
        Z = z[f"{i}_+x_depth"].shape[0]
        shape = (X, Y, Z)
        faces = {f: fc.Facade(f, z[f"{i}_{f}_depth"], z[f"{i}_{f}_ids"], z[f"{i}_{f}_data"],
                              shape["xyz".index(f[1])]) for f in FACES}
        out.append((shape, faces))
    return out


def disagreement(faces, shape):
    """The lift, and per face the (u, v) mask of cells that change under project(lift(.))."""
    from blockgen.corpus import facade as fc

    s = fc.lift(list(faces.values()), shape)
    bad = {}
    for f, F in faces.items():
        B = fc.project(s, f)
        same = (F.mask == B.mask) & ((~F.mask) | ((F.depth == B.depth) & (F.ids == B.ids)))
        bad[f] = ~same
    return s, bad


def face_sheet(faces, bad=None, scale=6):
    """The fc.sheet layout (sides on one row, roof and underside below), red tint on `bad` cells."""
    from blockgen.corpus import facade as fc

    pics = {}
    for f, F in faces.items():
        p = fc.face_picture(F).astype(np.float32)
        if bad is not None:
            m = bad[f].T
            if F.axis != 1:
                m = m[::-1]
            if fc._FLIP_U[F.face]:
                m = m[:, ::-1]
            p[m] = 0.35 * p[m] + 0.65 * RED
        pics[f] = np.repeat(np.repeat(p.astype(np.uint8), scale, 0), scale, 1)
    rows = [[f for f in faces if faces[f].axis != 1], [f for f in faces if faces[f].axis == 1]]
    gap = 2
    width = max(sum(pics[f].shape[1] for f in r) + gap * (len(r) + 1) for r in rows)
    height = sum(max(pics[f].shape[0] for f in r) for r in rows) + gap * (len(rows) + 1)
    out = np.full((height, width, 3), 255, np.uint8)
    y = gap
    for r in rows:
        x, h = gap, max(pics[f].shape[0] for f in r)
        for f in r:
            p = pics[f]
            out[y + h - p.shape[0]:y + h, x:x + p.shape[1]] = p
            x += p.shape[1] + gap
        y += h + gap
    return out


def render(struct, px=360):
    import cpu_render

    cpu_render.install()
    from blockgen.renderer import textured

    img = textured.render_structure(struct, px=px, azim_deg=45.0, elev_deg=30.0)
    a = img[..., 3:4].astype(np.float32) / 255.0
    rgb = (img[..., :3].astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
    ys, xs = np.where(img[..., 3] > 0)
    if len(ys):
        rgb = rgb[max(ys.min() - 6, 0):ys.max() + 7, max(xs.min() - 6, 0):xs.max() + 7]
    return rgb


def n_parts(struct) -> int:
    from scipy import ndimage as ndi

    return int(ndi.label(struct.block_ids != 0)[1])          # 6-connected, as the bench counts parts


REAL_SHELLS = DATA / "outputs/bench_arms/real_facades6_32.npz"


def fig_teaser() -> None:
    """One row, picked by a stated rule: the first sampled box at least 14 wide and deep whose
    Grammar lift has more parts than the median Grammar lift. The reference column is the
    held-out real exterior (Real-Shell) whose box is closest to it (different house)."""
    from blockgen.curation.houses import load_structures_from_cache

    g, c = load_face_sets(ARM_GRAMMAR), load_face_sets(ARM_CROSS)
    idx = [i for i in range(min(len(g), len(c))) if g[i] and c[i]]
    gparts = {i: n_parts(disagreement(g[i][1], g[i][0])[0]) for i in idx}
    med = float(np.median(list(gparts.values())))
    i = next(i for i in idx if g[i][0][0] >= 14 and g[i][0][2] >= 14 and gparts[i] > med)
    (shape, gf), (_, cf) = g[i], c[i]
    real, _ = load_structures_from_cache(str(REAL_SHELLS))
    real = [s for s in real if s.occupied_mask.any()]
    j = int(np.argmin([np.abs(np.array(s.shape) - np.array(shape)).sum() for s in real]))
    rs = real[j]
    gs, gbad = disagreement(gf, shape)
    cs, _ = disagreement(cf, shape)
    share = np.mean(np.concatenate([b.ravel() for b in gbad.values()]))
    tiles = [face_sheet(gf, gbad), render(gs), face_sheet(cf), render(cs), render(rs)]
    notes = [f"{100 * share:.0f}% of cells disagree", f"{n_parts(gs)} parts",
             "0% disagree", f"{n_parts(cs)} parts", f"{n_parts(rs)} parts"]
    cols = ["(a) Grammar faces\n(red: cells that disagree)", "(b) their lift", "(c) Cross-Face faces",
            "(d) their lift", "(e) a real exterior\n(another house)"]
    print("teaser box", i, shape, "median grammar parts", med, "real shell", j, rs.shape)
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 1.75),
                             gridspec_kw=dict(width_ratios=[1.7, 1, 1.7, 1, 1], wspace=0.06))
    for r in range(1):
        for j, (ax, t) in enumerate(zip(axes, tiles)):
            ax.imshow(t)
            _bare(ax)
            ax.text(0.5, -0.03, notes[j], transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color=INK)
            if r == 0:
                ax.text(0.5, 1.0, cols[j], transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=7, color=INK)
    _save(fig, "fig1_teaser")


# ---------------------------------------------------------------- method
def fig_method() -> None:
    """Top-down slice (x right, z up). The front face looks up the page; the right face looks left."""
    N = 8
    hits = {1: 3, 2: 3, 3: 4, 4: 4, 5: 3, 6: 3}            # front-face first hit depth per column x
    known = {(x, z) for x, z in hits.items()}
    empty = {(x, z) for x, zz in hits.items() for z in range(zz)}
    row_z = 4                                              # the right-face cell's ray runs along z=row_z
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1))
    for ax, stage in zip(axes, ("front", "right")):
        for x in range(N):
            for z in range(N):
                face = "#9c6b3c" if (x, z) in known else ("#ecebe6" if (x, z) in empty else "white")
                ax.add_patch(Rectangle((x, z), 1, 1, facecolor=face, edgecolor=MUTED, lw=0.6))
        ax.set_xlim(-0.6, N + 1.6); ax.set_ylim(-2.4, N + 0.3)
        ax.set_aspect("equal"); ax.axis("off")
        if stage == "front":
            for x in hits:
                ax.annotate("", xy=(x + 0.5, -0.05), xytext=(x + 0.5, -1.0),
                            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))
            ax.text(N / 2, -1.25, "front-face rays commit first hits (brown)\nand mark the voxels in front empty (grey)",
                    ha="center", va="top", fontsize=7, color=INK)
            ax.set_title("(a) shell state after the front face", fontsize=8, color=INK)
        else:
            ax.annotate("", xy=(N + 0.05, row_z + 0.5), xytext=(N + 1.3, row_z + 0.5),
                        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
            first_known = next(d for d in range(N) if (N - 1 - d, row_z) in known)
            for d in range(N):
                x = N - 1 - d
                if d < first_known and (x, row_z) not in empty:
                    ax.add_patch(Rectangle((x + 0.14, row_z + 0.14), 0.72, 0.72, fill=False, edgecolor=ORANGE, lw=1.5))
                elif d == first_known:
                    ax.add_patch(Rectangle((x + 0.14, row_z + 0.14), 0.72, 0.72, fill=False, edgecolor=ORANGE,
                                           lw=1.5, ls=(0, (2, 1.5))))
                else:
                    ax.plot([x + 0.25, x + 0.75], [row_z + 0.25, row_z + 0.75], color=INK, lw=0.8)
                    ax.plot([x + 0.25, x + 0.75], [row_z + 0.75, row_z + 0.25], color=INK, lw=0.8)
            ax.text(N / 2, -0.25, "one right-face cell: legal first hits (orange);\n"
                    "dashed: a known voxel, its block is forced;\ncrossed: behind a known voxel, illegal",
                    ha="center", va="top", fontsize=7, color=INK)
            ax.set_title("(b) legal tokens for a later face", fontsize=8, color=INK)
    _save(fig, "fig2_method")


# ---------------------------------------------------------------- quality
def fig_quality() -> None:
    # T117 / T122 cards, n=393-399, 20M Joint-Face D4. Reprojection error from the samplers'
    # summary.json (Cross-Face arms are 0 by Proposition 1); free best-of-4 is left out because its
    # face sets were not saved.
    rows = [  # label, reprojection error, parts, mv_kid, colour
        ("Grammar, T=1.0", 0.386, 62.7, 0.496, BLUE),
        ("Cross-Face, T=1.0", 0.0, 29.5, 0.331, ORANGE),
        ("Grammar, T=0.7", 0.281, 33.7, 0.299, BLUE),
        ("Cross-Face, T=0.7", 0.0, 12.9, 0.166, ORANGE),
        ("Cross-Face Best-of-4", 0.0, 7.7, 0.098, ORANGE),
    ]
    panels = [("reprojection error\n(share of cells that change)", 1, None),
              ("parts per building\n(dotted: real exteriors, 13.5)", 2, 13.5),
              ("MV-KID\n(lower is better)", 3, None)]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0), sharey=True,
                             gridspec_kw=dict(wspace=0.12))
    y = np.arange(len(rows))[::-1]
    for ax, (title, col, ref) in zip(axes, panels):
        for yi, r in zip(y, rows):
            v = r[col]
            if v is None:
                ax.text(0.01, yi, "not measured", va="center", fontsize=6, color=GREY)
                continue
            ax.barh(yi, v, height=0.64, color=r[4], edgecolor="white", linewidth=1)
            ax.text(v, yi, f" {v:g}", va="center", ha="left", fontsize=6.5, color=INK)
        if ref is not None:
            ax.axvline(ref, color=GREY, lw=0.8, ls=":")
        ax.set_title(title, fontsize=6.8, color=INK, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6.5)
        ax.margins(x=0.28)
        if col == 1:
            ax.set_xlim(0, 0.5); ax.set_xticks([0, 0.2, 0.4])
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[0] for r in rows], fontsize=7)
    axes[0].legend(handles=[Rectangle((0, 0), 1, 1, color=BLUE), Rectangle((0, 0), 1, 1, color=ORANGE)],
                   labels=["no cross-face rule", "Cross-Face"], fontsize=6, frameon=False,
                   loc="lower left", bbox_to_anchor=(-0.9, 1.3), ncol=2)
    _save(fig, "fig3_quality")


# ---------------------------------------------------------------- cost
def fig_cost() -> None:
    z = np.load(DATA / "data/minecraft/cache/houses_48.npz", allow_pickle=True)
    sh = np.array([np.asarray(b).shape for b in z["block_ids"]])
    X, Y, Zd = sh.T
    faces = 2 * (X * Y + Y * Zd + X * Zd)
    vox = X * Y * Zd
    N = sh.max(1)
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    rng = np.random.default_rng(0)
    bins = np.arange(4, 49, 4)
    for vals, c, lab in ((vox, BLUE, "voxels"), (faces, ORANGE, "face cells")):
        ax.scatter(N + rng.uniform(-0.35, 0.35, len(N)), vals, s=3, color=c, alpha=0.18, lw=0)
        sel = [(N > b - 4) & (N <= b) for b in bins]
        med = np.array([np.median(vals[s]) if s.any() else np.nan for s in sel])
        ax.plot(bins - 2, med, color=c, lw=2)
        last = np.where(~np.isnan(med))[0][-1]
        ax.text(bins[last] - 1.5, med[last], f"  {lab} (median)", color=INK, fontsize=7, va="center")
    ax.set_yscale("log")
    ax.set_xlabel("largest box side (blocks)", fontsize=7)
    ax.set_ylabel("cells to generate per house", fontsize=7)
    ax.axvline(32, color=GREY, lw=0.8, ls=":")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.5)
    ax.set_xlim(0, 66)
    _save(fig, "fig4_cost")


def main() -> None:
    figstyle.apply()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("method", fig_method), ("quality", fig_quality), ("teaser", fig_teaser)):
        if which in (name, "all"):
            fn()
    if which == "cost":
        fig_cost()


if __name__ == "__main__":
    main()
