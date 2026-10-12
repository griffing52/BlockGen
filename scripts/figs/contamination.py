"""Case-study figure: a provenance gate, the leak it found, and what it cost.

    .venv/bin/python scripts/figs/contamination.py

Writes `outputs/figstage/contamination.{png,pdf}`.

Every number is read out of a recorded artefact or recomputed here from the
pinned split; nothing is transcribed by hand. `--dump` prints each drawn number
with the file it came from, which is what `outputs/figstage/contamination.md`
quotes.

Panel A is a waffle per arm over the 399 canonical test builds: how many of them
that arm's training set contains. The tabular version of this ("87%", "13%",
"0%") reads as three similar-looking percentages; the point is that they are not
similar at all, and 346 filled cells beside 53 filled cells says so without a
sentence. The three arms sit in one row, each waffle beside its own stack of
type, which is what lets the figure print at 0.9 of a 5.5in column with no type
under 6 pt.

`family_overlap` still runs and still reaches `--dump`. A random *row* split
leaks build *families* rather than identical builds, which is why the canonical
split is group-aware, and the mean over 20 seeds is the number to quote because
seed 0 -- the 35.8% this project long quoted -- is the worst of the twenty. It is
a sentence in section 5 rather than a fourth waffle, because a fourth waffle
would put a second unit on a shared axis and cost the panel a third of its
width.

Panel B is the consequence on one axis. The contaminated arm is the best number
on the chart and is struck out, because the reason it wins is drawn directly
above it. The three arms that remain are inside one Holm group, so the honest
reading of the cleaned retrain is a tie -- and the LLM comparators carry a 13%
leak of their own, which is why the tie is stated as a tie and not as a win.

There is deliberately no near-duplicate inset: none exists. The script measures
the best overlap between a contaminated sample and any leaked test build
(`--dump` prints it) and it is 0.56 IoU, far under the 0.95 duplicate threshold
-- this leak is invisible to a duplicate counter, which is the point of the
footnote rather than a picture.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from armnames import display          # noqa: E402
OUT = ROOT / "outputs" / "figstage"
NAME = "contamination"

#: The run of record for every quantitative figure in the paper.
BENCH_RUN = (ROOT / "outputs"
             / "run_20260925_002631_bench_canonical-split_fixed_transparency_cull")
#: The two prompted-LLM arms record how many canonical-test captions their
#: checkpoint trained on, in the manifest written when they were sampled.
LLM_MANIFEST = ROOT / "outputs" / "bench_arms" / "llm_control_32_manifest.json"
CORPUS_MANIFEST = ROOT / "data" / "minecraft" / "cache" / "houses_32_manifest.json"
SPLIT_JSON = (ROOT / "data" / "minecraft" / "splits"
              / "houses_32.s0.70-15-15.v1.json")
#: The contaminated arm's samples, for the "is anything actually copied" check.
DIRTY_NPZ = ROOT / "outputs" / "bench_arms" / "native_oriented_32.npz"

# --- palette (the paper's, unchanged: CVD-validated, no red/green pair) -----
BLUE = "#2a78d6"      # the clean retrain
ORANGE = "#eb6834"    # contamination -- the leak, and the arm that carries it
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e4e0"
SURFACE = "#fcfcfb"
DEEMPH = "#c3c2bd"
CELL_BG = "#e8e7e3"   # a held-out test build


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "serif",
        # Keep the $p_{holm}$ / Delta glyphs in the body face rather than
        # matplotlib's sans-serif mathtext default.
        "mathtext.fontset": "dejavuserif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK_2,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
    })


def _save(fig: plt.Figure) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{NAME}.{ext}", bbox_inches="tight", pad_inches=0.02,
                    dpi=300)
    plt.close(fig)
    print(f"  wrote {(OUT / NAME).relative_to(ROOT)}.pdf (+ .png)")


# ---------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------
def scorecard() -> Dict:
    return json.loads((BENCH_RUN / "scorecard.json").read_text())


def h2h(card: Dict, metric: str = "geom_kid") -> Dict:
    return [b for b in card["run"]["head_to_head"] if b["metric"] == metric][0]


def pair(card: Dict, a: str, b: str, metric: str = "geom_kid") -> Dict:
    """The recorded paired, Holm-corrected comparison between two arms."""
    for c in h2h(card, metric)["comparisons"]:
        if {c["a"], c["b"]} == {a, b}:
            return c
    raise KeyError(f"no recorded {metric} comparison for {a} vs {b}")


def legacy_overlap() -> Tuple[int, int, int]:
    """Recompute `experiments_transfer._split_houses(seed=0, val_frac=0.15)`.

    That function permutes `range(len(houses))` under `default_rng(0)` and takes
    the first 15% as val, the rest as train -- over the *same* cached corpus the
    canonical split indexes, so the overlap is exact index arithmetic and needs
    no model or corpus load. Returns (n_test_in_legacy_train, n_val_in_legacy
    _train, n_test).
    """
    split = json.loads(SPLIT_JSON.read_text())
    n = sum(split["sizes"].values())
    perm = np.random.default_rng(0).permutation(n)
    n_val = max(8, int(round(0.15 * n)))
    legacy_train = set(perm[n_val:].tolist())
    test, val = set(split["test"]), set(split["val"])
    return (len(legacy_train & test), len(legacy_train & val), len(test))


N_SEEDS = 20


def family_overlap() -> Dict[str, float]:
    """Family leakage a naive random *row* split would produce on this corpus.

    Uses the shipped `splits.group_key` (normalized title minus trailing digits,
    plus category) so "American Middle Class House 10" and "... 22" collide.

    The number to quote is the **mean over `N_SEEDS` seeds**, not one seed: the
    leak a random split produces is itself random, and seed 0 -- the number this
    project has been quoting -- turns out to be the worst of the twenty. Section
    5 quotes the mean, the s.d. and the range from `--dump`; the figure no longer
    draws any of it, because a waffle of leaked build *families* beside three
    waffles of leaked *builds* puts two units on one axis.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from blockgen.eval.bench.splits import group_key

    items = json.loads(CORPUS_MANIFEST.read_text())["items"]
    keys = [group_key(it) for it in items]
    n = len(items)
    n_tr, n_va = int(round(0.70 * n)), int(round(0.15 * n))

    def leak(s: int) -> int:
        perm = np.random.default_rng(s).permutation(n)
        tr, va = perm[:n_tr], perm[n_tr:n_tr + n_va]
        g = {keys[i] for i in tr}
        return sum(1 for i in va if keys[i] in g)

    counts = np.array([leak(s) for s in range(N_SEEDS)], dtype=float)
    frac = counts / n_va
    return {
        "n_val": n_va,
        "mean_count": float(counts.mean()),
        "draw_count": int(np.floor(counts.mean() + 0.5)),
        "mean_pct": float(100 * frac.mean()),
        # Sample s.d. (ddof=1): these seeds are a sample of the splits one could
        # draw, not the population of them.
        "sd_pct": float(100 * frac.std(ddof=1)),
        "sd_pct_pop": float(100 * frac.std()),
        "min_pct": float(100 * frac.min()),
        "max_pct": float(100 * frac.max()),
        "seed0_count": int(counts[0]),
        "seed0_pct": float(100 * frac[0]),
        "seed0_is_max": bool(counts[0] == counts.max()),
        "seed0_z": float((frac[0] - frac.mean()) / frac.std(ddof=1)),
        "n_seeds": N_SEEDS,
    }


def trackd_overlap() -> Tuple[int, int]:
    """How many canonical-test builds the Track D checkpoints trained on.

    Recorded by `scripts/sample_geobias.py` as `prompts_dropped_as_trained`: the
    canonical-test captions it had to withhold because the checkpoint had
    trained on those builds. Both arms share the split and the seed, so the
    count is one number.
    """
    rep = json.loads(LLM_MANIFEST.read_text())["report"]
    n_test = sum(json.loads(SPLIT_JSON.read_text())["sizes"][k] for k in ("test",))
    return int(rep["prompts_dropped_as_trained"]), n_test


def best_copy_iou() -> Tuple[float, float, int]:
    """Best occupancy IoU between a contaminated sample and a leaked test build.

    The question a reader asks next: if 87% of the test set was in training, is
    the model handing back copies? Measured with the bench's own voxelization
    and IoU (`blockgen.eval.novelty`, grid 24), against exactly the 346 test
    builds that leaked. Returns (max_iou, mean_nn_iou, n_at_or_above_0.95).
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from blockgen.curation.houses import load_structures_from_cache
    from blockgen.eval.bench.splits import load_split, split_structures
    from blockgen.eval.novelty import _iou_matrix, voxelize_occupancy
    from blockgen.utils.serialize import build_block_vocab

    gen, _ = load_structures_from_cache(str(DIRTY_NPZ))
    split = load_split("houses_32")
    test = split_structures(split, "test")
    train = split_structures(split, "train")

    n = len(split.train) + len(split.val) + len(split.test)
    perm = np.random.default_rng(0).permutation(n)
    legacy_train = set(perm[max(8, int(round(0.15 * n))):].tolist())
    leaked = [j for j, i in enumerate(split.indices("test").tolist())
              if i in legacy_train]

    grid = 24
    vocab = build_block_vocab(train, max_dim=grid)
    g_occ, _ = voxelize_occupancy(gen, grid, vocab)
    t_occ, _ = voxelize_occupancy(test, grid, vocab)
    iou = _iou_matrix(g_occ, t_occ)[:, leaked]
    nn = iou.max(1)
    return float(iou.max()), float(nn.mean()), int((nn >= 0.95).sum())


# ---------------------------------------------------------------------------
# panel A -- the overlap itself
# ---------------------------------------------------------------------------
COLS, ROWS = 21, 19          # 21 x 19 = 399, the canonical test split exactly
PITCH, CELL = 1.0, 0.80      # a surface gap between cells, per mark spec

#: One quadrant, in points at figure scale: waffle, gutter, type column. Three
#: arms in a row leave 125 pt per quadrant, so every offset in the quadrant is
#: measured in points and converted into the waffle's data units. Type size then
#: stops depending on how many cells the waffle happens to have.
FIG_W = 5.30
QUAD_PT = 0.98 * FIG_W * 72.0 / 3.0
GAP_PT, TEXT_PT = 6.0, 72.0
WAFFLE_PT = QUAD_PT - GAP_PT - TEXT_PT
W_UNITS = (COLS - 1) * PITCH + CELL     # waffle width, data units
H_UNITS = (ROWS - 1) * PITCH + CELL     # waffle height, data units
UNIT = WAFFLE_PT / W_UNITS              # points per data unit
X0 = -CELL / 2                          # the waffle's top-left corner
TEXT_X = X0 + (WAFFLE_PT + GAP_PT) / UNIT
PANEL_W = X0 + QUAD_PT / UNIT

#: The type stack beside a waffle: arm, split, share, count. The arm name gets
#: `TITLE_LINES` lines whether it needs them or not, so the three big
#: percentages land on one baseline and the eye reads across the row instead of
#: down it.
F_ARM, F_SPLIT, F_PCT, F_COUNT, LS = 7.4, 7.0, 14.0, 6.9, 1.3
TITLE_LINES, LS_TITLE = 2, 1.16
Y_ARM = 0.0
Y_SPLIT = F_ARM * LS_TITLE * TITLE_LINES + 1.0
Y_PCT = Y_SPLIT + F_SPLIT * LS + 2.0 + 0.72 * F_PCT     # a baseline
Y_COUNT = Y_PCT + F_COUNT + 3.5                         # a baseline
STACK_PT = Y_COUNT + 0.25 * F_COUNT
CONTENT_PT = max(STACK_PT, H_UNITS * UNIT)
#: Vertical slack between the shorter of the two columns and the taller.
WAFFLE_DY = (CONTENT_PT - H_UNITS * UNIT) / 2.0 / UNIT
TEXT_DY = (CONTENT_PT - STACK_PT) / 2.0 / UNIT


def _waffle(ax, n_filled: int, n_total: int, color: str) -> None:
    """`n_total` cells, filled from the top-left, one cell per test build.

    Cell *positions* carry nothing -- a waffle encodes a count. Filling in a
    fixed order keeps the three waffles comparable at a glance; it does not
    claim that the 53 leaked builds of one arm are among the 346 of another.

    Everything in the quadrant is drawn in these data units, with the axes on
    `aspect="equal"`, so the type stays locked to the waffle at whatever size
    the figure is printed.
    """
    ax.set_xlim(X0, PANEL_W)
    ax.set_ylim(X0 + CONTENT_PT / UNIT, X0)
    ax.set_aspect("equal")
    # Left-align the quadrant: `aspect="equal"` otherwise centres the drawing in
    # its cell and the three waffles drift apart from the panel below.
    ax.set_anchor("W")
    ax.axis("off")
    for k in range(n_total):
        r, c = divmod(k, COLS)
        on = k < n_filled
        ax.add_patch(Rectangle(
            (c * PITCH - CELL / 2, r * PITCH - CELL / 2 + WAFFLE_DY),
            CELL, CELL,
            facecolor=color if on else CELL_BG, edgecolor="none", zorder=2))


def panel_a(axes: List[plt.Axes], rows: List[dict]) -> None:
    def y(offset_pt: float) -> float:
        return X0 + (TEXT_DY * UNIT + offset_pt) / UNIT

    for ax, row in zip(axes, rows):
        _waffle(ax, row["n"], row["total"], row["color"])
        ax.text(TEXT_X, y(Y_ARM), row["title"], ha="left", va="top",
                fontsize=F_ARM, color=INK, linespacing=LS_TITLE)
        ax.text(TEXT_X, y(Y_SPLIT), row["split"], ha="left", va="top",
                fontsize=F_SPLIT, color=INK_2, style="italic")
        pct = 100.0 * row["n"] / row["total"]
        ax.text(TEXT_X, y(Y_PCT), f"{pct:.0f}%" if row["n"] else "none",
                ha="left", va="baseline",
                fontsize=F_PCT if row["n"] else F_PCT - 3.5,
                color=INK if row["n"] else MUTED)
        ax.text(TEXT_X, y(Y_COUNT), f"{row['n']} of {row['total']} {row['unit']}",
                ha="left", va="baseline", fontsize=F_COUNT, color=INK_2)


# ---------------------------------------------------------------------------
# panel B -- the consequence
# ---------------------------------------------------------------------------
#: Where the Holm-group column and the tie bracket sit, in geom-kid units. The
#: axis runs past its last tick so that the column, the bracket and its label
#: all land inside the axes: anything drawn outside them widens the saved
#: bounding box, which would shrink every type size on the printed page.
GCOL, BRACKET, XMAX = 25.2, 26.5, 33.5


def panel_b(ax: plt.Axes, card: Dict) -> dict:
    block = h2h(card, "geom_kid")
    scores, groups = block["scores"], block["groups"]
    tie = sorted(a for a, g in groups.items()
                 if g == groups["canon_oriented"])

    # Labels are `armnames.display`, so a tick here is the same string the
    # prose and Table 2 use. `real_test` keeps a descriptive label: it is the
    # reference floor rather than a submission, and "Real" alone would not say
    # so. The keys are the scorecard's and never change.
    arms = [
        ("real_test", "real held-out builds (the floor)", "context"),
        ("native_oriented", display("native_oriented"), "dirty"),
        ("canon_oriented", display("canon_oriented") + " (clean)", "clean"),
        ("llm_control", display("llm_control"), "context"),
        ("llm_bias", display("llm_bias"), "context"),
        ("patchwork", display("patchwork"), "context"),
    ]
    rows = []
    for key, label, role in arms:
        blk = card["arms"][key]["realism"]["geom_kid"]
        lo, hi = blk["ci"]
        rows.append((key, label, role, float(blk["value"]), float(lo), float(hi)))
    rows.sort(key=lambda r: r[3])

    y_of = {}
    for i, (key, label, role, v, lo, hi) in enumerate(rows):
        y = len(rows) - 1 - i
        y_of[key] = y
        color = {"dirty": ORANGE, "clean": BLUE}.get(role, DEEMPH)
        ax.plot([lo, hi], [y, y], color=color, zorder=2,
                linewidth=2.2 if role != "context" else 1.5,
                linestyle=(0, (2.2, 1.6)) if role == "dirty" else "-",
                solid_capstyle="round")
        ax.plot([v], [y], "o", color=SURFACE if role == "dirty" else color,
                markersize=6.5 if role != "context" else 5.2,
                markeredgecolor=color if role == "dirty" else SURFACE,
                markeredgewidth=1.4 if role == "dirty" else 1.0, zorder=3)
        lab = f"{v:.3f}" if role in ("dirty", "clean") else f"{v:.2f}"
        ax.text(hi + 0.5, y, lab, va="center", ha="left",
                fontsize=7.5 if role in ("dirty", "clean") else 7,
                color=INK if role in ("dirty", "clean") else INK_2)
        ax.text(GCOL, y, groups[key], va="center", ha="center", fontsize=7,
                color=INK_2, family="monospace")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[1] for r in reversed(rows)])
    for tick, r in zip(ax.get_yticklabels(), reversed(rows)):
        tick.set_color(INK if r[2] in ("dirty", "clean") else INK_2)
        tick.set_fontsize(7.6 if r[2] in ("dirty", "clean") else 7.2)
    ax.tick_params(axis="y", length=0, pad=2)

    # The tie bracket: one Holm group, drawn as a brace beside the three rows.
    ys = sorted(y_of[a] for a in tie)
    ax.plot([BRACKET, BRACKET + 0.45, BRACKET + 0.45, BRACKET],
            [ys[0] - 0.28, ys[0] - 0.28, ys[-1] + 0.28, ys[-1] + 0.28],
            color=MUTED, linewidth=0.7, clip_on=False, zorder=3)
    ax.text(BRACKET + 0.8, sum(ys) / len(ys), "not\nseparated",
            ha="left", va="center", fontsize=6.9, color=INK_2, linespacing=1.25)

    # The provenance verdict: a badge on the row, its reason in the empty
    # lower-left, and a dashed arrow for where cleaning the split moves the arm.
    yd, yc = y_of["native_oriented"], y_of["canon_oriented"]
    d = pair(card, "canon_oriented", "native_oriented")
    ax.annotate("", xy=(scores["canon_oriented"] - 0.55, yc),
                xytext=(scores["native_oriented"] + 0.45, yd),
                arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.9,
                                linestyle=(0, (3, 2)),
                                connectionstyle="arc3,rad=-0.5"))
    # "UNRANKED", not "DQ": in this suite a DQ is a fired metric gate
    # (memorization, mode collapse), while a training-provenance failure makes an
    # arm unrankable under the protocol -- `protocol.match_arm`. The arm's
    # memorization gates did not fire, and saying DQ here would imply they did.
    ax.text(17.9, yd, "UNRANKED", ha="left", va="center", fontsize=7.2,
            color=ORANGE)
    # Both paired contrasts stay computed -- they are what `--dump` reports and
    # what the caption and the prose quote -- but neither is drawn in the axes.
    t = pair(card, "canon_oriented", "llm_control")

    ax.set_xlabel("raw V-KID, lower is better (95% CI, n=128, one bench run)",
                  labelpad=2)
    ax.set_xlim(-1.8, XMAX)
    ax.set_ylim(-0.75, len(rows) - 0.35)
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", zorder=0)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_axisbelow(True)
    ax.text(GCOL, len(rows) - 0.55, "Holm\ngroup", ha="center", va="bottom",
            fontsize=6.8, color=MUTED, family="monospace", linespacing=1.15)
    return {"tie": tie, "delta": d, "tie_pair": t,
            "scores": {k: scores[k] for k, _, _, _, _, _ in rows},
            "groups": {k: groups[k] for k, _, _, _, _, _ in rows}}


def _strike(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    """Rule through one y tick label: the number is right, the ranking is void.

    Measured from the laid-out text rather than guessed from a character count,
    so it cannot drift when the label or the font does.
    """
    fig.canvas.draw()
    for tick in ax.get_yticklabels():
        if tick.get_text() != label:
            continue
        bb = tick.get_window_extent(renderer=fig.canvas.get_renderer())
        inv = fig.transFigure.inverted()
        (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        fig.add_artist(plt.Line2D([x0 - 0.004, x1 + 0.004],
                                  [(y0 + y1) / 2] * 2,
                                  transform=fig.transFigure, color=ORANGE,
                                  linewidth=0.8, zorder=5))


# ---------------------------------------------------------------------------
# layout, in inches from the top of the figure: the waffle row is exactly as
# tall as its type stack, and everything else is measured against it. `FIG_W` is
# set with the quadrant geometry above. At 0.90 of a 5.5in column the figure
# prints at 0.949 scale, so the 6.8 pt Holm header, the smallest type here,
# lands at 6.46 pt. The three bands below pay for the arm names' second line:
# the names are the paper's, they no longer fit on one line of a 72 pt column,
# and the height a wrap costs comes out of white space rather than type size.
TOP_BAND = 0.275           # panel-A title and legend
QUAD_H = CONTENT_PT / 72.0 + 0.02
B_TITLE = 0.225            # panel-B title
B_PLOT = 1.085             # panel-B axes
B_FOOT = 0.38              # panel-B tick labels and axis label
FIG_H = TOP_BAND + QUAD_H + B_TITLE + B_PLOT + B_FOOT


def _f(inches_from_top: float) -> float:
    """Figure fraction for a distance measured down from the top edge."""
    return 1.0 - inches_from_top / FIG_H


def _fx(points_from_left: float) -> float:
    """Figure fraction for a distance measured in points from the left edge."""
    return points_from_left / (FIG_W * 72.0)


def build(dump: bool = False) -> None:
    card = scorecard()
    dirty_test, dirty_val, n_test = legacy_overlap()
    trackd, _ = trackd_overlap()
    fam = family_overlap()
    iou_max, iou_mean, n_dup = best_copy_iou()

    # The first and third arms are the same model on two splits, so they carry
    # the same base name and the italic line under each is the whole difference
    # between 87% and none. Titles wrap at the parenthesis or the comma: at
    # 7.4 pt "Voxel-AR (contaminated)" measures 92.6 pt against a 72 pt column,
    # and shrinking the type instead would print it under 6 pt on the page.
    rows = [
        dict(title=display("native_oriented").replace(" (", "\n("),
             split="legacy 85/15 split",
             n=dirty_test, total=n_test, unit="builds", color=ORANGE),
        dict(title=f"{display('llm_control')},\n{display('llm_bias')}",
             split="90/10 shuffle",
             n=trackd, total=n_test, unit="builds", color=ORANGE),
        dict(title=display("canon_oriented"), split="canonical split",
             n=0, total=n_test, unit="builds", color=ORANGE),
    ]

    _style()
    # Panel A and panel B get their own gridspecs rather than one grid with a
    # spacer row. Panel A is one row of three, as wide as the figure and as tall
    # as its type stack; panel B keeps the full width and is inset from the left
    # so its arm names start at the figure's edge instead of hanging off it.
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gsa = fig.add_gridspec(1, 3, wspace=0.0, left=0.01, right=0.99,
                           top=_f(TOP_BAND), bottom=_f(TOP_BAND + QUAD_H))
    axes = [fig.add_subplot(gsa[0, i]) for i in range(3)]
    panel_a(axes, rows)

    fig.text(0.01, _f(0.03), "A   Training-set overlap",
             ha="left", va="top", fontsize=8.5, color=INK)
    fig.text(0.99, _f(0.03), f"one cell = one of the {n_test} test builds",
             ha="right", va="top", fontsize=7, color=MUTED)
    # The legend, stated once, in ink with coloured marks beside it. Its x
    # positions are points from the left edge, so they hold if the figure is
    # ever re-proportioned.
    x0, ly = _fx(0.01 * FIG_W * 72.0), _f(0.205)
    sw, sh = _fx(6.0), 0.055 / FIG_H
    for x, label, fill in ((x0, "in that model's training data", ORANGE),
                           (x0 + _fx(128.0), "genuinely held out", CELL_BG)):
        fig.patches.append(Rectangle((x, ly - sh / 2), sw, sh,
                                     transform=fig.transFigure,
                                     facecolor=fill, edgecolor="none"))
        fig.text(x + sw + _fx(3.5), ly, label, ha="left", va="center",
                 fontsize=7, color=INK_2)

    gsb = fig.add_gridspec(1, 1, left=0.325, right=0.99,
                           top=_f(TOP_BAND + QUAD_H + B_TITLE),
                           bottom=_f(FIG_H - B_FOOT))
    axb = fig.add_subplot(gsb[0, 0])
    info = panel_b(axb, card)
    # In figure coordinates, so B sits in the same left-hand column as A rather
    # than above the arm names.
    fig.text(0.01, _f(TOP_BAND + QUAD_H + 0.06),
             "B   Score after cleaning the split",
             ha="left", va="top", fontsize=8.5, color=INK)

    _strike(fig, axb, display("native_oriented"))
    _save(fig)

    if dump:
        out = {
            "bench_run": str(BENCH_RUN.relative_to(ROOT)),
            "legacy_train_x_canon_test": [dirty_test, n_test,
                                          round(100 * dirty_test / n_test, 1)],
            "legacy_train_x_canon_val": [dirty_val, n_test],
            "trackd_x_canon_test": [trackd, n_test,
                                    round(100 * trackd / n_test, 1)],
            "canonical_x_canon_test": [0, n_test],
            "random_row_split_family_leak": {
                k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in fam.items()},
            "geom_kid": {k: round(v, 4) for k, v in info["scores"].items()},
            "holm_groups": info["groups"],
            "clean_vs_dirty": {k: info["delta"][k]
                               for k in ("delta", "ci", "p", "p_adj")},
            "clean_vs_llm_control": {k: info["tie_pair"][k]
                                     for k in ("delta", "ci", "p", "p_adj", "winner")},
            "blockscore": {r["arm"]: r["score"] for r in card["run"]["blockscore"]
                           if r["arm"] in ("native_oriented", "canon_oriented",
                                           "llm_control", "llm_bias")},
            "noise_floor_sd_geom_kid":
                card["context"]["noise_floor_sd"]["realism.geom_kid"],
            "best_gen_vs_leaked_test_iou": round(iou_max, 4),
            "mean_gen_vs_leaked_test_iou": round(iou_mean, 4),
            "n_gen_at_iou_0.95": n_dup,
        }
        print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dump", action="store_true",
                    help="print every drawn number as JSON")
    build(dump=ap.parse_args().dump)


if __name__ == "__main__":
    main()
