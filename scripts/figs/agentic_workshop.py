"""Figures for the agentic workshop paper (deliverables/paper/workshop_agentic).

    cd /home/griffing52/programming/gen/BlockGen        # outputs/ and data/ live here
    BLOCKGEN_FIG_RENDER=cpu PYTHONPATH=<worktree> .venv/bin/python \
        <worktree>/scripts/figs/agentic_workshop.py [teaser method results twins cost] \
        [--card outputs/run_.../scorecard.json --suffix _n256]

Figures
  teaser   prompt | dream | Classic-Matched | Dream | Dream-Caption | real house, three
           prompts: the first test-split prompt of each of the three largest categories.
  method   the build-render-compare-repair loop and the arm x ingredient matrix.
  results  (a) MV-KID vs Geom-KID with 95% intervals, (b) enterable, (c) twin rate.
  twins    12 random Classic-Matched builds, each beside its nearest other build.
  cost     USD per build vs MV-KID.

`results`, `twins` and `cost` read a bench scorecard (`--card`, default the n=133 card)
and write `<name><suffix>.{pdf,png}`. Everything renders on the CPU (OSMesa).
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import cpu_render  # noqa: E402
cpu_render.install()
import figstyle  # noqa: E402
from figstyle import INK, INK_2, RULE, SHADE, PAPER, CVPR_FULL, CVPR_COL  # noqa: E402

MAIN = Path("/home/griffing52/programming/gen/BlockGen")
OUT = WT / "deliverables" / "paper" / "workshop_agentic" / "figures"
CARD133 = MAIN / "outputs/run_20261109_121625_bench_agentic_arms_beside_the_20M_legacy/scorecard.json"

#: the n=133 test-split runs behind the card (scorecard meta.source)
RUNS133 = {
    "classic-matched": MAIN / "outputs/run_20261007_092654_bench_houses32_classic",
    "dream": MAIN / "outputs/run_20261007_092654_bench_houses32_dream",
    "dream-caption": MAIN / "outputs/run_20261008_004051_bench_houses32_dream_caption133",
    "legacy-oneshot": MAIN / "outputs/run_20261109_130512_bench_houses32_oneshot_test",
}
#: the n=123 validation-split runs that make up n=256
RUNS_VAL = {
    "classic-matched": MAIN / "outputs/run_20261109_165950_bench_houses32_agentic_val",
    "dream": MAIN / "outputs/run_20261109_165950_bench_houses32_agentic_val",
    "dream-caption": MAIN / "outputs/run_20261109_165950_bench_houses32_agentic_val",
    "legacy-oneshot": MAIN / "outputs/run_20261109_145342_bench_houses32_oneshot_val",
}

AGENTIC = ["legacy-oneshot", "classic-matched", "dream", "dream-caption"]
NAME = {"legacy-oneshot": "One-Shot", "classic-matched": "Classic-Matched",
        "dream": "Dream", "dream-caption": "Dream-Caption",
        "real_test": "Real (held out)", "canon_oriented": "Voxel-AR 5M",
        "recanon20m_s0": "Voxel-AR 27M"}
#: categorical arm palette, fixed order (validated with dataviz validate_palette.js,
#: light mode: lightness, chroma, adjacent CVD and normal-vision checks PASS; all-pairs
#: CVD aqua/magenta 6.1 is legal only with secondary encoding, so every mark carries a
#: direct label and its own marker shape). ORANGE and BLUE stay reserved for the
#: render/geometry tiers.
COLOR = {"legacy-oneshot": "#eda100", "classic-matched": "#e87ba4",
         "dream": "#6250d6", "dream-caption": "#1baf7a"}
MARK = {"legacy-oneshot": "v", "classic-matched": "s", "dream": "D", "dream-caption": "o"}
REF_COLOR = "#454545"
REF_MARK = {"real_test": "*", "canon_oriented": "^", "recanon20m_s0": "P"}
TWIN_GATE_SPREAD = 0.02  # composite.GATES voxel_twin_rate: 3 spreads, 0.02 floor
TWIN_RED = "#d03b3b"


# --- data -------------------------------------------------------------------

def load_npz(path):
    from blockgen.curation.houses import load_structures_from_cache
    return load_structures_from_cache(str(path))[0]


def arm_structs(arm, run=None):
    run = run or RUNS133[arm]
    return load_npz(next(Path(run).glob(f"{arm}_32.npz")))


def build_dirs(arm, run):
    d = Path(run) / "builds" / arm
    return sorted(p for p in d.glob("[0-9][0-9][0-9]") if p.is_dir())


def per_build_cost(arm, runs):
    """Full-price USD per finished build: the sum of would_cost_usd over its charges.

    Reads builds/<arm>/NNN/charges.jsonl (written as each build finishes; the
    NNN_build.json files appear only when the whole run ends), falling back to
    NNN_build.json. A build counts once its result.json exists.
    """
    out = []
    for run in runs:
        for d in build_dirs(arm, run):
            if not (d / "result.json").exists():
                continue
            cj = d / "charges.jsonl"
            if cj.exists():
                ch = [json.loads(l) for l in cj.read_text().splitlines() if l.strip()]
            else:
                bj = d.parent / f"{d.name}_build.json"
                ch = json.loads(bj.read_text()).get("charges", []) if bj.exists() else []
            out.append(sum(c.get("would_cost_usd") or 0.0 for c in ch))
    return np.array(out)


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def card_arms(card):
    """{canonical arm: arm dict} for the arms the figures plot."""
    arms = card["arms"]
    out = {}
    alias = {"llm-oneshot": "legacy-oneshot", "oneshot": "legacy-oneshot"}
    for key, a in arms.items():
        base = key.split("/")[-1]
        base = alias.get(base, base)
        for want in AGENTIC + list(REF_MARK):
            if base == want or base.startswith(want + "_n") or base == want + "256":
                out.setdefault(want, a)
    # the trained arms in the 256 card may carry another suffix; fall back by prefix
    for want in ("canon_oriented", "recanon20m_s0"):
        if want not in out:
            for key, a in arms.items():
                if key.startswith(want):
                    out[want] = a
                    break
    return out


def metric(a, block, key):
    m = a[block][key]
    if "gen" in m:
        return m["gen"]["mean"], m["gen"]["ci"][0], m["gen"]["ci"][1]
    return m["value"], m["ci"][0], m["ci"][1]


# --- rendering --------------------------------------------------------------

_TEX = None


def render(s, px=360, az=45.0, el=30.0):
    global _TEX
    from blockgen.renderer.textured import render_structure
    from blockgen.renderer.textures import load_face_textures
    if _TEX is None:
        _TEX = load_face_textures()
    im = render_structure(s.crop_to_non_air(), px=px, azim_deg=az, elev_deg=el,
                          ortho=True, face_textures=_TEX)
    return on_white(crop_alpha(im))


def crop_alpha(im, pad=6):
    a = im[..., 3] > 0
    if not a.any():
        return im
    ys, xs = np.where(a)
    return im[max(ys.min() - pad, 0):ys.max() + pad + 1,
              max(xs.min() - pad, 0):xs.max() + pad + 1]


def on_white(im):
    a = im[..., 3:4].astype(np.float32) / 255.0
    return (im[..., :3] * a + 255.0 * (1 - a)).astype(np.uint8)


def crop_white(im, thr=245, pad=16):
    im = np.asarray(im.convert("RGB"))
    ink = (im < thr).any(-1)
    ys, xs = np.where(ink)
    return im[max(ys.min() - pad, 0):ys.max() + pad + 1,
              max(xs.min() - pad, 0):xs.max() + pad + 1]


def show(ax, im, frame=None, lw=1.2):
    """Draw an image centred in a square axes, letterboxed on white."""
    h, w = im.shape[:2]
    s = max(h, w)
    canvas = np.full((s, s, 3), 255, np.uint8)
    y0, x0 = (s - h) // 2, (s - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = im[..., :3]
    ax.imshow(canvas, interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(frame is not None)
        if frame is not None:
            sp.set_edgecolor(frame); sp.set_linewidth(lw)


def blank(ax):
    ax.set_axis_off()


# --- figure 1: teaser -------------------------------------------------------

def teaser_prompts(k=3):
    """First test-split prompt of each of the k most common categories."""
    lines = (WT / "configs/agentic/prompts/houses32_test_grabcraft.txt").read_text().splitlines()
    lines = [l for l in lines if l.strip()]
    cats = {}
    for l in lines:
        c = l.rsplit("(a Minecraft ", 1)[1].rstrip(")")
        cats.setdefault(c, []).append(l)
    top = sorted(cats, key=lambda c: -len(cats[c]))[:k]
    return [cats[c][0] for c in top]


def real_by_title():
    from blockgen.eval.bench import splits
    cache = MAIN / "data" / "minecraft" / "cache"
    sp = splits.load_split("houses_32", cache_dir=cache)
    structs, man = splits.load_corpus("houses_32", cache)
    out = {}
    for i in sp.indices("test"):
        it = man["items"][i]
        cat = it.get("category", "").replace("-houses", "")
        for ttl in (str(it.get("title")), splits._TRAILING_NUM.sub("", str(it.get("title"))).strip()):
            out.setdefault(f"{ttl} (a Minecraft {cat} house)", structs[i])
    return out


def fig_teaser():
    from blockgen.eval.traversal import traversal_stats
    prompts = teaser_prompts()
    print("teaser prompts:", prompts)
    arms = ["classic-matched", "dream", "dream-caption"]
    data = {a: arm_structs(a) for a in arms}
    titles = {a: [s.metadata["title"] for s in data[a]] for a in arms}
    reals = real_by_title()
    cap_dirs = build_dirs("dream-caption", RUNS133["dream-caption"])

    cols = ["Prompt", "Dream image", "Classic-Matched", "Dream", "Dream-Caption", "Real house"]
    W = CVPR_FULL
    fig = plt.figure(figsize=(W, 0.25 + 1.12 * len(prompts)))
    widths = [1.2, 1, 1, 1, 1, 1]
    gs = fig.add_gridspec(len(prompts), 6, width_ratios=widths, wspace=0.04, hspace=0.18,
                          left=0.0, right=1.0, top=1 - 0.2 / (0.25 + 1.12 * len(prompts)),
                          bottom=0.0)
    for r, p in enumerate(prompts):
        ax = fig.add_subplot(gs[r, 0]); blank(ax)
        title, cat = p.rsplit(" (a Minecraft ", 1)
        ax.text(0.02, 0.53, "\n".join(textwrap.wrap(title, 20)), fontsize=7.5,
                fontweight="bold", va="bottom", ha="left", transform=ax.transAxes)
        ax.text(0.02, 0.47, f"(a Minecraft {cat}", fontsize=6.5, color=INK_2,
                va="top", ha="left", transform=ax.transAxes, style="italic")
        # the dream (identical for Dream and Dream-Caption: shared image cache)
        j = titles["dream-caption"].index(p)
        dream = Image.open(cap_dirs[j] / "dreams" / "iso_v0.png")
        ax = fig.add_subplot(gs[r, 1]); show(ax, crop_white(dream), frame=RULE, lw=0.6)
        for c, a in enumerate(arms):
            s = data[a][titles[a].index(p)]
            ent = traversal_stats(s).enterable
            ax = fig.add_subplot(gs[r, 2 + c]); show(ax, render(s))
            ax.text(0.98, 0.03, "enters" if ent else "no entry", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=6, color=INK_2 if ent else TWIN_RED,
                    fontweight="normal" if ent else "bold")
        real = reals[p]
        ent = traversal_stats(real).enterable
        ax = fig.add_subplot(gs[r, 5]); show(ax, render(real))
        ax.text(0.98, 0.03, "enters" if ent else "no entry", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6, color=INK_2 if ent else TWIN_RED,
                fontweight="normal" if ent else "bold")
    for c, name in enumerate(cols):
        x = sum(widths[:c]) + widths[c] / 2
        fig.text(x / sum(widths), 1.0, name, ha="center", va="top", fontsize=7.5,
                 fontweight="bold", color=INK)
    figstyle.save_all(fig, OUT, "teaser", exts=("pdf", "png"))
    plt.close(fig)


# --- figure 2: method -------------------------------------------------------

def fig_method():
    d = build_dirs("dream-caption", RUNS133["dream-caption"])[0]
    dream = crop_white(Image.open(d / "dreams" / "iso_v0.png"))
    rend = crop_white(Image.open(d / "iters" / "00_generate" / "render_iso.png"))
    W, H = CVPR_FULL, 2.0
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 0.63, 1]); ax.set_xlim(0, 63); ax.set_ylim(-4.2, 20)
    ax.set_axis_off()
    LW = 0.8

    def box(x, y, w, h, title, sub=None, mono=False, left=False, ec=INK_2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.7",
                                    fc=PAPER, ec=ec, lw=LW))
        tx, ha = (x + 0.8, "left") if left else (x + w / 2, "center")
        ax.text(tx, y + h - 0.9, title, ha=ha, va="top", fontsize=7, fontweight="bold",
                color=INK)
        if sub:
            ax.text(tx, y + h - 3.0, sub, ha=ha, va="top", fontsize=6, color=INK_2,
                    family="monospace" if mono else None, linespacing=1.2)

    def path(pts, color=INK_2):
        xs, ys = zip(*pts)
        ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, lw=LW,
                solid_capstyle="butt", solid_joinstyle="miter")
        ax.annotate("", pts[-1], pts[-2], arrowprops=dict(arrowstyle="-|>", color=color,
                    lw=LW, shrinkA=0, shrinkB=0, mutation_scale=7))

    def thumb(im, x, y, h):
        w = h * im.shape[1] / im.shape[0]
        ax.imshow(im, extent=(x, x + w, y, y + h), interpolation="lanczos", zorder=3,
                  aspect="auto")
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=RULE, lw=0.5, zorder=4))

    # top row: prompt -> dream -> caption
    box(0.3, 13, 11, 6.5, "Prompt", "GrabCraft title\n+ category")
    box(14.5, 13, 15.5, 6.5, "Dream", "text-to-image\ntarget", left=True,
        ec=COLOR["dream"])
    thumb(dream, 23.6, 13.5, 5.5)
    box(33, 13, 15.5, 6.5, "Caption", "JSON: footprint,\nstoreys, roof,\nmaterials, faces",
        ec=COLOR["dream-caption"])
    path([(11.3, 16.2), (14.5, 16.2)])
    path([(30.0, 16.2), (33.0, 16.2)])
    # bottom row: builder -> execute -> render -> compare
    box(0.3, 2.5, 16.5, 7.5, "Builder", "walls oak_planks ..\ngable spruce_stairs ..\nset oak_door ..",
        mono=True)
    box(19.5, 2.5, 10.5, 7.5, "Execute", "32\u00b3 canvas,\nerrors\nper line")
    box(32.7, 2.5, 14, 7.5, "Render", "textured,\ndream\ncamera", left=True)
    thumb(rend, 40.2, 3.0, 5.6)
    box(49.5, 2.5, 13.2, 7.5, "Compare", "to the dream:\nSigLIP, silhouette,\nconnectivity")
    path([(4.0, 13.0), (4.0, 10.0)])
    path([(19.5, 13.0), (19.5, 11.5), (10.0, 11.5), (10.0, 10.0)], COLOR["dream"])
    path([(40.7, 13.0), (40.7, 11.2), (13.5, 11.2), (13.5, 10.0)], COLOR["dream-caption"])
    path([(16.8, 6.2), (19.5, 6.2)])
    path([(30.0, 6.2), (32.7, 6.2)])
    path([(46.7, 6.2), (49.5, 6.2)])
    # loops back to the builder
    path([(24.75, 2.5), (24.75, 1.0), (11.0, 1.0), (11.0, 2.5)])
    ax.text(17.9, 0.7, "repair execution errors (1 turn)", fontsize=6, color=INK_2,
            ha="center", va="top")
    path([(56.1, 2.5), (56.1, -2.6), (6.0, -2.6), (6.0, 2.5)])
    ax.text(31.0, -2.9, "rewrite from target beside own render (Classic-Matched: own renders only), \u22643 turns, keep the best",
            fontsize=6, color=INK_2, ha="center", va="top")

    # arm x ingredient matrix
    ingred = ["exec.\nrepair", "self-\ncritique", "dream\nimage", "match\nturns",
              "caption"]
    have = {"legacy-oneshot": [0, 0, 0, 0, 0], "classic-matched": [1, 1, 0, 0, 0],
            "dream": [1, 0, 1, 1, 0], "dream-caption": [1, 0, 1, 1, 1]}
    mx = fig.add_axes([0.645, 0.03, 0.355, 0.84])
    mx.set_xlim(-2.9, len(ingred) - 0.5); mx.set_ylim(len(AGENTIC) - 0.5, -0.5)
    mx.set_axis_off()
    for j, ing in enumerate(ingred):
        mx.text(j, -0.62, ing, ha="center", va="bottom", fontsize=6, color=INK_2,
                linespacing=1.05)
    for i, a in enumerate(AGENTIC):
        mx.text(-0.85, i, NAME[a], ha="right", va="center", fontsize=6.5, color=INK)
        mx.plot([-0.6], [i], marker=MARK[a], ms=4.5, color=COLOR[a], mec=PAPER, mew=0.5,
                clip_on=False, ls="none")
        for j, h in enumerate(have[a]):
            if h:
                mx.plot(j, i, "o", ms=6, color=INK, mec=INK)
            else:
                mx.plot(j, i, "o", ms=6, mfc=PAPER, mec=RULE, mew=0.8)
        if i:
            mx.plot([-2.9, len(ingred) - 0.5], [i - 0.5] * 2, color=RULE, lw=0.4)
    # the matrix needs no title: the caption names it
    figstyle.save_all(fig, OUT, "method", exts=("pdf", "png"))
    plt.close(fig)


# --- figure 3: main results -------------------------------------------------

def fig_results(card, suffix=""):
    A = card_arms(card)
    n_of = {k: v["meta"]["n"] for k, v in A.items()}
    fig = plt.figure(figsize=(CVPR_FULL, 2.15))
    ax = fig.add_axes([0.065, 0.19, 0.39, 0.72])
    bx = fig.add_axes([0.615, 0.19, 0.165, 0.72])
    cx = fig.add_axes([0.815, 0.19, 0.165, 0.72])

    # (a) scatter
    def pt(k, color, marker, ms, label_xy, label, ha="left", short=None):
        gx, gl, gh = metric(A[k], "realism", "geom_kid")
        mv, ml, mh = metric(A[k], "realism", "mv_dino_kid")
        ax.errorbar(gx, mv, xerr=[[gx - gl], [gh - gx]], yerr=[[mv - ml], [mh - mv]],
                    fmt="none", ecolor=color, elinewidth=0.9, capsize=0, zorder=2)
        ax.plot(gx, mv, marker=marker, ms=ms, color=color, mec=PAPER, mew=0.8, zorder=3,
                ls="none", label=short or label, clip_on=False)
        ax.annotate(label, (gx, mv), xytext=label_xy, textcoords="offset points",
                    fontsize=6.5, ha=ha, va="center", color=INK)
    offs = {"classic-matched": (-7, 6), "dream": (-7, -3), "dream-caption": (6, -8),
            "legacy-oneshot": (6, 6)}
    has = {"classic-matched": "right", "dream": "right", "dream-caption": "left",
           "legacy-oneshot": "left"}
    for k in AGENTIC:
        if k in A:
            pt(k, COLOR[k], MARK[k], 6, offs[k], NAME[k], has[k])
    refoff = {"real_test": (6, 5), "canon_oriented": (-7, 5), "recanon20m_s0": (-7, 5)}
    refha = {"real_test": "left", "canon_oriented": "right", "recanon20m_s0": "right"}
    for k in REF_MARK:
        if k in A:
            pt(k, REF_COLOR, REF_MARK[k], 7 if k == "real_test" else 6, refoff[k],
               NAME[k] + ("" if n_of[k] == n_of.get("dream", n_of[k]) else f" (n={n_of[k]})"),
               refha[k], short=NAME[k])
    ax.set_xlabel("Geom-KID (geometry) ↓")
    ax.set_ylabel("MV-KID (appearance) ↓")
    ax.set_xlim(-1.0, 21.0); ax.set_ylim(-0.02, 0.48)
    ax.grid(True, lw=0.4); ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.5)
    ax.legend(loc="lower right", fontsize=6, handletextpad=0.2, borderaxespad=0.3,
              labelspacing=0.25, numpoints=1, ncol=2, columnspacing=0.8)

    # (b) enterable and (c) twin rate: one row per arm
    rows = [k for k in AGENTIC if k in A] + [k for k in REF_MARK if k in A]
    ys = np.arange(len(rows))[::-1]

    def dots(axx, block, key, xlabel, gate=None, xlim=(0, 1)):
        for y, k in zip(ys, rows):
            v, lo, hi = metric(A[k], block, key)
            c = COLOR.get(k, REF_COLOR); m = MARK.get(k, REF_MARK.get(k))
            axx.plot([lo, hi], [y, y], color=c, lw=1.4, solid_capstyle="butt")
            axx.plot(v, y, marker=m, ms=5.5 if m != "*" else 7, color=c, mec=PAPER, mew=0.7,
                      clip_on=False, zorder=4)
            axx.annotate(f"{v:.2f}", (max(hi, v), y), xytext=(5, 0),
                         textcoords="offset points", fontsize=6, va="center", color=INK_2)
        axx.set_yticks(ys); axx.set_ylim(-0.6, len(rows) - 0.4)
        axx.set_xlim(*xlim); axx.set_xlabel(xlabel)
        axx.grid(True, axis="x", lw=0.4); axx.set_axisbelow(True)
        axx.tick_params(labelsize=6.5)
        axx.spines["left"].set_visible(False); axx.tick_params(axis="y", length=0)
        if gate is not None:
            axx.axvspan(gate, xlim[1], color=SHADE, zorder=0, lw=0)
            axx.text(gate + 0.01, len(rows) - 0.55, "DQ by gate", fontsize=6, color=INK_2,
                     va="top", ha="left")
        if len(rows) > 3:
            axx.axhline(ys[len([k for k in rows if k in AGENTIC])] + 0.5, color=RULE, lw=0.5)

    dots(bx, "traversal", "enterable", "Enterable (share) ↑", xlim=(0, 1.08))
    bx.set_yticklabels([NAME[k] for k in rows], fontsize=6.5)
    real_twin = metric(A["real_test"], "novelty", "voxel_twin_rate")[0]
    dots(cx, "novelty", "voxel_twin_rate", "Near-twin rate ↓",
         gate=real_twin + 3 * TWIN_GATE_SPREAD, xlim=(0, 0.5))
    cx.set_yticklabels([])
    n = n_of.get("dream", 0)
    for axx, lab in ((ax, "a"), (bx, "b"), (cx, "c")):
        axx.text(-0.02, 1.03, lab, transform=axx.transAxes, fontsize=8, fontweight="bold",
                 ha="right", va="bottom")
    # n and interval kind are stated in the LaTeX caption
    figstyle.save_all(fig, OUT, f"results{suffix}", exts=("pdf", "png"))
    plt.close(fig)


# --- figure 4: template collapse -------------------------------------------

def nearest_aligned(occs, i):
    """(j, iou, variant k, shift) of the nearest other build to i: D4 + translation."""
    from scipy import fft as sfft
    from blockgen.eval.bench.twins import d4
    a = occs[i]
    dxz = max(max(o.shape[0], o.shape[2]) for o in occs)
    dy = max(o.shape[1] for o in occs)
    shape = (sfft.next_fast_len(2 * dxz), sfft.next_fast_len(2 * dy),
             sfft.next_fast_len(2 * dxz))

    def spec(x):
        g = np.zeros(shape, np.float32); g[:x.shape[0], :x.shape[1], :x.shape[2]] = x
        return sfft.rfftn(g)
    vs = np.stack([spec(v) for v in d4(a)])
    best = (-1, 0.0, 0)
    sa = int(a.sum())
    for j, b in enumerate(occs):
        if j == i or not b.any():
            continue
        corr = sfft.irfftn(vs * np.conj(spec(b))[None], s=shape, axes=(1, 2, 3))
        inter = corr.reshape(8, -1).max(1)
        k = int(inter.argmax()); it = float(np.round(inter[k]))
        iou = it / (sa + int(b.sum()) - it)
        if iou > best[1]:
            best = (j, iou, k)
    return best


def fig_twins(card=None, suffix="", arm="classic-matched", n_show=12, seed=0):
    from blockgen.eval.bench.twins import _occ, TWIN_THRESHOLD
    from blockgen.eval.bench import twins as tw
    S = twins_structs(card, arm)
    occs = [_occ(s) for s in S]
    rep = tw.within_set_twins(S)
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(S), n_show, replace=False).tolist())
    print(f"twins: {arm} n={len(S)} rate={rep.rate:.3f} picked {pick}")
    ncol_pairs, nrow = 4, int(np.ceil(n_show / 4))
    tile = CVPR_FULL / (ncol_pairs * 2 + (ncol_pairs - 1) * 0.25)
    H = nrow * (tile + 0.46) + 0.05
    fig = plt.figure(figsize=(CVPR_FULL, H))
    n_tw = 0
    for q, i in enumerate(pick):
        j, iou, k = nearest_aligned(occs, i)
        is_twin = iou >= TWIN_THRESHOLD
        n_tw += is_twin
        r, c = divmod(q, ncol_pairs)
        x0 = c * (2 + 0.25) * tile / CVPR_FULL
        y0 = 1 - (r + 1) * (tile + 0.46) / H
        frame = TWIN_RED if is_twin else None
        # show the neighbour in the D4 pose that matches i, so the copy is visible
        from blockgen.eval.bench.twins import d4
        si = S[i]
        sj = aligned_struct(S[j], k)
        for t, s in enumerate((si, sj)):
            axx = fig.add_axes([x0 + t * tile / CVPR_FULL, y0 + 0.27 / H,
                                tile / CVPR_FULL * 0.97, tile / H * 0.97])
            show(axx, render(s, px=300), frame=frame, lw=1.1)
            import html
            ttl = html.unescape(S[i if t == 0 else j].metadata["title"].split(" (a Minecraft")[0])
            lines = textwrap.wrap(ttl, 17)
            if len(lines) > 2:
                lines = [lines[0], textwrap.shorten(" ".join(lines[1:]), 17, placeholder="…")]
            axx.text(0.5, -0.02, "\n".join(lines), linespacing=1.0,
                     transform=axx.transAxes, ha="center", va="top", fontsize=6,
                     color=INK_2)
        fig.text(x0 + tile / CVPR_FULL, y0 + 0.27 / H + tile / H * 0.97 + 0.01 / H,
                 f"IoU {iou:.2f}" + ("  twin" if is_twin else ""), ha="center",
                 va="bottom", fontsize=6.5, color=TWIN_RED if is_twin else INK,
                 fontweight="bold" if is_twin else "normal")
    print(f"twins: {n_tw}/{n_show} sampled builds have a twin")
    figstyle.save_all(fig, OUT, f"twins{suffix}", exts=("pdf", "png"), tight=True)
    plt.close(fig)
    return pick, n_tw


def aligned_struct(s, k):
    """The Structure s in the pose of the query whose D4 variant k it matched.

    twins.d4 lists (mirror m, rotation r): variant = rot90(mirror^m(q), r). The
    neighbour b matches that variant, so b in the query's frame is mirror^m(rot90(b, -r)).
    Block states (stair facing) are carried along unrotated: a display-only liberty.
    """
    from blockgen.utils.data import Structure
    m, r = divmod(k, 4)

    def f(x):
        x = np.rot90(x, -r, axes=(0, 2))
        return np.ascontiguousarray(x[::-1] if m else x)
    return Structure(f(s.block_ids), f(s.block_data), metadata=dict(s.metadata))


def twins_structs(card, arm):
    if card is None:
        return arm_structs(arm)
    A = card_arms(card)
    src = A[arm]["meta"]["source"]
    p = Path(src) if Path(src).is_absolute() else MAIN / src
    return load_npz(p)


# --- figure 5: cost ---------------------------------------------------------

def fig_cost(card, suffix="", runs=None):
    A = card_arms(card)
    runs = runs or {k: [RUNS133[k]] for k in AGENTIC}
    fig, ax = plt.subplots(figsize=(CVPR_COL, 2.0))
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.2, top=0.97)
    xmax = 0.06
    for k in AGENTIC:
        c = per_build_cost(k, runs[k])
        if not len(c):
            continue
        m, lo, hi = boot_ci(c)
        if k in A:
            v, vl, vh = metric(A[k], "realism", "mv_dino_kid")
            ax.errorbar(m, v, xerr=[[m - lo], [hi - m]], yerr=[[v - vl], [vh - v]],
                        fmt="none", ecolor=COLOR[k], elinewidth=0.9)
            ax.plot(m, v, marker=MARK[k], ms=6, color=COLOR[k], mec=PAPER, mew=0.8,
                    ls="none", label=f"{NAME[k]} (n={len(c)})")
            ax.annotate(NAME[k], (m, v), xytext=(-6, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=6.5)
        else:
            ax.axvline(m, color=COLOR[k], lw=1.0, ls=(0, (3, 2)),
                       label=f"{NAME[k]} cost (n={len(c)}, not scored)")
            ax.text(m + 0.0008, 0.47, f"{NAME[k]}\n${m:.3f}, MV-KID\nnot yet scored",
                    fontsize=6, va="top", ha="left", color=INK_2)
        xmax = max(xmax, hi * 1.1)
    refs = []
    for k, ls in (("recanon20m_s0", (0, (1, 1.5))), ("canon_oriented", (0, (4, 2)))):
        if k in A:
            v = metric(A[k], "realism", "mv_dino_kid")[0]
            ax.axhline(v, color=REF_COLOR, lw=0.8, ls=ls)
            refs.append((v, f"{NAME[k]}, no API cost"))
    ax.set_xlim(0, xmax); ax.set_ylim(0, 0.5)
    for v, s in refs:
        ax.text(xmax * 0.995, v + 0.007, s, fontsize=6, color=INK_2, va="bottom", ha="right")
    ax.set_xlabel("USD per build (full price)")
    ax.set_ylabel("MV-KID ↓")
    ax.grid(True, lw=0.4); ax.set_axisbelow(True); ax.tick_params(labelsize=6.5)
    ax.legend(loc="lower right", fontsize=6, handletextpad=0.3, borderaxespad=0.3,
              labelspacing=0.3, numpoints=1)
    figstyle.save_all(fig, OUT, f"cost{suffix}", exts=("pdf", "png"))
    plt.close(fig)


def find_card256():
    """Newest bench card written by the AG256 driver (scratchpad/recanon/agentic256.sh)."""
    cands = sorted(MAIN.glob("outputs/run_*/scorecard.json"))
    for p in reversed(cands):
        if "agentic_and_LLM_one-shot" in p.parent.name or "agentic_and_llm" in p.parent.name.lower():
            return p
    for p in reversed(cands):
        try:
            if json.loads(p.read_text())["run"]["name"].startswith("agentic and LLM one-shot"):
                return p
        except Exception:
            pass
    raise FileNotFoundError("no n=256 agentic card yet")


# --- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("figs", nargs="*", default=["teaser", "method", "results", "twins", "cost"])
    ap.add_argument("--card", default=str(CARD133),
                    help="scorecard.json, or 'n256' for the newest card named "
                         "'agentic and LLM one-shot arms at n 256'")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--with-val", action="store_true",
                    help="cost: add the n=123 validation runs (the n=256 card)")
    a = ap.parse_args()
    figstyle.apply(7.0)
    if a.card == "n256":
        a.card = str(find_card256())
        print("card:", a.card)
    card = json.loads(Path(a.card).read_text())
    runs = None
    if a.with_val:
        runs = {k: [RUNS133[k], RUNS_VAL[k]] for k in AGENTIC}
    for f in a.figs:
        print("==", f)
        if f == "teaser":
            fig_teaser()
        elif f == "method":
            fig_method()
        elif f == "results":
            fig_results(card, a.suffix)
        elif f == "twins":
            fig_twins(None if a.card == str(CARD133) else card, a.suffix)
        elif f == "cost":
            fig_cost(card, a.suffix, runs)


if __name__ == "__main__":
    main()
