"""The validation ladder, as one picture: what each metric can and cannot see.

The paper's methodological claim is "validate a metric before you rank with
it": score every candidate on rungs of *known* damage, quote every separation
in units of that metric's own noise floor, and gate whatever fails ordering,
invariance or n-stability. That argument currently lives as prose plus an
appendix table, which is exactly the form in which a reader skips it.

This figure is the one-glance version. One row per rung, two markers per row
(the render-space statistic and the geometry statistic), x = distance from the
real floor in noise floors. The eye is meant to catch one thing: short pairs at
the top, where the two metrics agree about how bad a corruption is, and three
long ones at the bottom, where they disagree by two orders of magnitude *and in
opposite directions*.

Honesty notes, because both are easy to get wrong here:

*The exact zeros.* `geom_kid` under material shuffle and under monochrome is not
approximately the real value, it is bit-identical to it -- the descriptor never
reads block type, so the two arms produce the same feature matrix and the same
float. A log axis cannot render 0, and a symlog axis that places it at the left
edge invites the reader to read "very small" where "exactly, provably zero" is
meant. So exact zeros go in a separate narrow gutter column, labelled as such,
and are never plotted on the continuous axis. The reference row (`real_test`) is
zero by definition for both metrics and sits in the same gutter, which is the
honest place for it: the invariance rows are bit-identical *to that row*.

*One run, one n.* Noise floors fall like 1/n, so a separation is only meaningful
against the floor measured in the same run at the same n. Everything here comes
from one scorecard at n=128 and every denominator is that scorecard's own
`context.noise_floor_sd`, the idiom `scripts/audit_claims.py` `Card.sep()`
enforces. Rungs the run of record does not contain (`hollow`, `chunk_delete`,
`occ_noise`) are omitted rather than imported from a run at a different n, and
the two decimation arms it does contain are omitted because decimation is not a
rung of quality damage at all -- see `NOT_A_QUALITY_RUNG` below.

Run as:  .venv/bin/python scripts/figs/ladder.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle                                            # noqa: E402
from figstyle import BLUE, ORANGE, INK_2, PAPER, RULE, SHADE  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "outputs" / "figstage"

#: The run of record. One run, one n: see the module docstring.
BENCH = (ROOT / "outputs"
         / "run_20260925_002631_bench_canonical-split_fixed_transparency_cull")

# Colours and font come from scripts/figs/figstyle.py; nothing is declared here.
# ORANGE is the render tier and BLUE the geometry tier, as in every other figure.
# GUTTER (the "= 0 exactly" column) and BAND (the sub-noise-floor span) were two
# separate near-white greys; they are one structural fill, figstyle.SHADE, since
# they do the same job -- marking an area that is not the continuous axis.

GEOM, REND = "geom_kid", "mv_dino_kid"


def _style() -> None:
    """Recessive axes, no chartjunk. Sizes are this figure's; colour is shared."""
    figstyle.apply(size=8, **{
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2,
    })


class Card:
    """A scorecard with the one accessor this figure needs.

    `sep` reads numerator and denominator from the same file on purpose: a
    difference measured at n=128 divided by a floor measured at n=64 is a
    number with no meaning. Same logic as `scripts/audit_claims.py` `Card.sep`.
    """

    def __init__(self, path: Path):
        self.path = path
        self.d = json.loads(path.read_text())
        self.arms = self.d["arms"]
        self.sd = self.d["context"]["noise_floor_sd"]
        self.n = int(self.arms["real_test"]["meta"]["n"])

    def value(self, arm: str, metric: str) -> float:
        return float(self.arms[arm]["realism"][metric]["value"])

    def raw_diff(self, arm: str, metric: str, ref: str = "real_test") -> float:
        return self.value(arm, metric) - self.value(ref, metric)

    def sep(self, arm: str, metric: str, ref: str = "real_test") -> float:
        return self.raw_diff(arm, metric, ref) / self.sd[f"realism.{metric}"]


#: (arm, label) in plotting order, top to bottom, with `None` marking a group
#: header row. The grouping is the argument: the metrics agree on the top five
#: rungs and split on the bottom three.
ROWS: List[Tuple[Optional[str], str]] = [
    (None, "the metrics agree"),
    ("real_test", "real held-out (the floor)"),
    ("train_verbatim", "verbatim training copies"),
    ("real@single_mode", "one build repeated $n$ times"),
    (None, "the metrics disagree"),
    ("real@solidify", "every sealed room filled"),
    ("real_shuffled_materials", "block types shuffled in place"),
    ("real@monochrome", "repainted in one material"),
]

#: Present in the scorecard and deliberately NOT drawn. `real@canon16` and
#: `real@canon8` decimate a build to a coarser grid, which changes its
#: resolution -- its SIZE -- and not its quality. The project's standing
#: decision (see `blockgen/eval/bench/fast.py: control_arms`, and §5) is that
#: this is not a dose of damage and not a rung a leaderboard may rank against;
#: the leaderboard table drops both arms. Drawing them here would show the paper
#: ranking two arms it elsewhere refuses to rank. Recorded here so their absence
#: is a decision on the page rather than a silent gap against the scorecard.
NOT_A_QUALITY_RUNG = ("real@canon16", "real@canon8")

#: Rungs in the project's probe suite that this run does not carry as arms. Named
#: on the figure so a reader who knows the appendix table does not read their
#: absence as a result. Borrowing them from the n=64 ladder analysis would mean
#: dividing by a different run's noise floor.
ABSENT = ("hollow", "chunk_delete", "occ_noise")


def fmt(v: float) -> str:
    return f"{v:.1f}" if abs(v) < 10 else f"{v:.0f}"


def main() -> None:
    _style()
    card = Card(BENCH / "scorecard.json")

    # The claim the gutter column makes, checked rather than asserted in prose:
    # these are bit-identical, not small. If a change to the geometry descriptor
    # ever breaks that, this figure must not quietly keep saying "exactly 0".
    for arm in ("real_shuffled_materials", "real@monochrome"):
        d = card.raw_diff(arm, GEOM)
        if d != 0.0:
            raise SystemExit(
                f"{arm}: geom_kid is no longer bit-identical to real "
                f"(difference {d!r}); the 'exactly 0' gutter would be a lie")

    rows = []
    for arm, label in ROWS:
        if arm is None:
            rows.append((None, label, None, None))
            continue
        rows.append((arm, label, card.sep(arm, REND), card.sep(arm, GEOM)))

    fig = plt.figure(figsize=(5.2, 2.42))
    gs = GridSpec(1, 2, width_ratios=[0.075, 1.0], wspace=0.045,
                  left=0.30, right=0.985, top=0.902, bottom=0.167)
    ax0 = fig.add_subplot(gs[0])    # the "exactly 0" gutter
    ax1 = fig.add_subplot(gs[1])    # the continuous axis, in noise floors

    ny = len(rows)
    ax0.set_ylim(ny - 0.5, -0.5)
    ax1.set_ylim(ny - 0.5, -0.5)

    # --- the gutter: a separate space, so "0" is never read as "small" -------
    ax0.set_facecolor(SHADE)
    ax0.set_xlim(-1, 1)
    ax0.set_xticks([])
    ax0.set_yticks([i for i, r in enumerate(rows) if r[0] is not None])
    ax0.set_yticklabels([r[1] for r in rows if r[0] is not None], fontsize=7.4)
    ax0.tick_params(axis="y", length=0, pad=3)
    for side in ("top", "right"):
        ax0.spines[side].set_visible(False)
    ax0.spines["bottom"].set_visible(False)
    ax0.spines["left"].set_visible(False)
    ax0.text(0.5, -0.032, "= 0\nexactly", transform=ax0.transAxes,
             ha="center", va="top", fontsize=6.8, color=INK_2,
             linespacing=1.25)

    # --- the continuous axis ------------------------------------------------
    # It deliberately starts just above zero and carries no "0" tick: zero lives
    # in the gutter and nowhere else, so the reader is never offered two
    # different zeros and left to work out whether they mean the same thing.
    lo, hi = 0.05, 1100.0
    ax1.set_xscale("symlog", linthresh=1.0, linscale=0.8)
    ax1.set_xlim(lo, hi)
    ax1.axvspan(lo, 1, color=SHADE, zorder=0, linewidth=0)
    ax1.set_xticks([1, 10, 100, 1000])
    ax1.set_xticklabels(["1", "10", "100", "1000"])
    ax1.set_yticks([])
    ax1.spines["left"].set_visible(False)
    ax1.grid(axis="x", zorder=0)
    ax1.set_axisbelow(True)
    for x in (1, 10, 100, 1000):
        ax1.axvline(x, color=RULE, linewidth=0.6, zorder=0)

    # break marks, so the gap between the two panels reads as a cut axis
    for ax, xa in ((ax0, 1.0), (ax1, 0.0)):
        ax.plot([xa - 0.012, xa + 0.012], [-0.012, 0.012], transform=ax.transAxes,
                color=INK_2, linewidth=0.7, clip_on=False, zorder=6)
        ax.plot([xa - 0.012, xa + 0.012], [0.988, 1.012], transform=ax.transAxes,
                color=INK_2, linewidth=0.7, clip_on=False, zorder=6)

    dy = 0.175
    EPS = 1e-9

    def marker(ax, x, y, metric):
        c = ORANGE if metric == REND else BLUE
        ax.plot([x], [y], marker="o" if metric == REND else "s", color=c,
                markersize=5.6 if metric == REND else 4.9,
                markeredgecolor=PAPER, markeredgewidth=0.9,
                linestyle="", zorder=5, clip_on=False)

    for i, (arm, label, sr, sg) in enumerate(rows):
        if arm is None:
            continue
        yr, yg = i - dy, i + dy
        zero_r, zero_g = abs(sr) < EPS, abs(sg) < EPS
        # the connector: its length IS the disagreement, so it is drawn first
        # and in grey, and it crosses the axis cut when one end is an exact 0.
        if zero_r and zero_g:
            ax0.plot([0, 0], [yr, yg], color=RULE, linewidth=2.4,
                     solid_capstyle="round", zorder=3)
        elif zero_g or zero_r:
            y_far, y_zero = (yr, yg) if zero_g else (yg, yr)
            x_far = sr if zero_g else sg
            ax0.plot([0, 1], [y_zero, y_zero], color=RULE, linewidth=2.4,
                     solid_capstyle="butt", zorder=3)
            ax1.plot([lo, x_far], [y_zero, y_far], color=RULE, linewidth=2.4,
                     solid_capstyle="round", zorder=3)
        else:
            ax1.plot([sr, sg], [yr, yg], color=RULE, linewidth=2.4,
                     solid_capstyle="round", zorder=3)
        marker(ax0 if zero_r else ax1, 0 if zero_r else sr, yr, REND)
        marker(ax0 if zero_g else ax1, 0 if zero_g else sg, yg, GEOM)

    # --- group headers ------------------------------------------------------
    for i, (arm, label, _, _) in enumerate(rows):
        if arm is not None:
            continue
        ax1.text(1.35, i + 0.06, label, fontsize=6.9, color=INK_2,
                 va="center", ha="left", style="italic")
        if i:
            for ax in (ax0, ax1):
                ax.axhline(i - 0.62, color=RULE, linewidth=0.7, zorder=1)

    # --- selective direct labels: only the rows the figure is about ---------
    # Row positions are looked up by arm name, never written as literals: rows
    # have been removed from this figure once already (the decimation rungs) and
    # hard-coded indices would have silently moved every annotation one row.
    at = {arm: i for i, (arm, _, _, _) in enumerate(rows) if arm}

    def tag(x, y, v, colour, dx=1.35):
        ax1.text(x * dx, y, fmt(v), fontsize=6.8, color=colour, va="center",
                 ha="left", zorder=6)

    lut = {r[0]: r for r in rows if r[0]}
    i_sol = at["real@solidify"]
    _, _, sr, sg = lut["real@solidify"]
    tag(sr, i_sol - dy, sr, ORANGE, dx=1.9)
    tag(sg, i_sol + dy, sg, BLUE)
    for arm in ("real_shuffled_materials", "real@monochrome"):
        tag(lut[arm][2], at[arm] - dy, lut[arm][2], ORANGE)

    # Three asides used to sit here: that the render metric cannot see
    # `solidify`, that the two material rows are bit-identical to real, and that
    # the shaded band is one noise floor. All three are analysis, so they belong
    # to the caption and the running text, not to the image. What remains on the
    # page still carries them: the band and its x tick at 1, the "= 0 exactly"
    # gutter, and the row labels. The figure now draws data and nothing else.

    ax1.set_xlabel("distance from real, in each metric's own noise floors",
                   labelpad=2.5, fontsize=7.2)

    fig.legend(handles=[
        Line2D([], [], marker="o", color=ORANGE, linestyle="", markersize=5.6,
               markeredgecolor=PAPER, markeredgewidth=0.9,
               label="MV-KID (render tier)"),
        Line2D([], [], marker="s", color=BLUE, linestyle="", markersize=4.9,
               markeredgecolor=PAPER, markeredgewidth=0.9,
               label="V-KID (geometry tier)")],
        loc="upper right", bbox_to_anchor=(0.995, 1.005), ncol=2,
        handletextpad=0.35, columnspacing=1.4, borderpad=0.2,
        framealpha=0.0)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"ladder.{ext}", bbox_inches="tight",
                    pad_inches=0.02, dpi=300)
    plt.close(fig)

    print(f"scorecard: {card.path.relative_to(ROOT)}  n={card.n}")
    print(f"noise floors: {card.sd}")
    for arm, label, sr, sg in rows:
        if arm is None:
            continue
        print(f"  {arm:<24} {label:<30} render={sr:10.4f}  geom={sg:10.4f}")
    for arm in NOT_A_QUALITY_RUNG:          # in the scorecard, not on the figure
        print(f"  [not drawn: not a quality rung] {arm:<16} "
              f"render={card.sep(arm, REND):8.2f}  geom={card.sep(arm, GEOM):8.2f}")
    print(f"absent from this run, omitted: {', '.join(ABSENT)}")
    print(f"wrote {(OUTDIR / 'ladder.png').relative_to(ROOT)} (+ .pdf)")


if __name__ == "__main__":
    main()
