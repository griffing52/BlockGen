"""One palette and one font for every figure in the paper.

Nine figure scripts each re-declared the same dozen constants, which is how they
drifted: four different near-white page fills, three greys all called "muted",
and a violet that appeared in exactly one figure and read as a third metric tier
that does not exist. This module is the single copy. A script imports it, deletes
its local constants, and renders the same figure in the agreed style.

Three standing decisions live here, because they are the ones that kept slipping:

*No page fill.* A figure sits on the paper. The old ``SURFACE = "#fcfcfb"`` was
painted behind the whole canvas and over the LaTeX page, which is why every
figure carried a faintly grey rectangle that nothing in the design asked for.
``PAPER`` is plain white, so the fill is invisible on the page and the PNG still
reads on a dark viewer background. Nothing else is filled by default: panels and
axes get no background, only the two structural fills below.

*Two data hues, one meaning each.* ORANGE is the render tier and BLUE is the
geometry tier, everywhere, in every figure. Neither is up for re-picking.

*No light text.* Two weights, both readable at the 6 pt sizes these figures use.

Run this file directly to print the resolved font and the palette.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# --- font -------------------------------------------------------------------

#: Arial first so the figures come out identical on a machine that has it.
#: Liberation Sans is metrically identical to Arial and is what actually renders
#: here: verified with `matplotlib.font_manager` on 2026-09-25, which reported
#:
#:     Arial            NOT INSTALLED
#:     Helvetica        NOT INSTALLED
#:     Liberation Sans  /usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf
#:     DejaVu Sans      <matplotlib>/mpl-data/fonts/ttf/DejaVuSans.ttf
#:
#: so the chain resolved to Liberation Sans. Do not trust that comment on
#: another machine; call `resolved_font()`, which asks font_manager.
SANS = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]

#: Mathtext face. The paper's figures carry a handful of maths strings ($\varphi$,
#: the KID kernel); `dejavusans` is the only bundled set that does not set them in
#: a serif beside sans prose, which reads as a typo.
MATHTEXT = "dejavusans"


def resolved_font() -> str:
    """The font file `SANS` actually resolves to, asked of font_manager."""
    from matplotlib.font_manager import FontProperties, findfont
    return findfont(FontProperties(family=SANS))


# --- data colours -----------------------------------------------------------

ORANGE = "#eb6834"      #: the render tier: phi = encoder o render
BLUE = "#2a78d6"        #: the geometry tier: phi = occupancy descriptor

#: A third data colour, and the only one. `pipeline` and `cutaway` show the
#: render tier TWICE -- once as published (exterior views) and once with the same
#: renderer cut open -- so those two figures need to tell the two apart. It is
#: ORANGE darkened rather than a new hue, precisely because that condition *is*
#: the render tier: same phi family, different view protocol. The VIOLET it
#: replaces was a fourth hue, which read as a third tier of metric.
#: Any figure that is not contrasting two view protocols does not use this.
ORANGE_DEEP = "#a34012"


# --- text -------------------------------------------------------------------

INK = "#111111"         #: primary text
#: The one secondary. 9.6:1 on white, so it is still readable set at 5.2 pt,
#: which is the smallest size any of these figures uses. It replaces both
#: `MUTED` ("#8a8985", 3.2:1) and `DEEMPH` ("#c3c2bd", 1.8:1); there is no
#: third, lighter text colour to fall back to, on purpose.
INK_2 = "#454545"


# --- structure --------------------------------------------------------------

PAPER = "#ffffff"       #: the page. Also marker halos and letterbox padding.

#: The only neutral fill, for areas that are structurally not the plot: the
#: shaded sub-noise-floor band, the "= 0 exactly" gutter, a section's
#: outside-the-model area. Light, but visibly a fill rather than off-white paper.
SHADE = "#e9e9e9"

#: Hairlines: spines, gridlines, image-tile borders, the connectors that join a
#: pair of markers. Everything drawn as structure rather than read as a value.
RULE = "#c8c8c8"


# --- build renders ----------------------------------------------------------
# The two colours the voxel sections and cutaway diagrams use for the build
# itself, shared with the teaser so one house looks the same across figures.

VOID = "#f7b48b"        #: sealed interior air
SOLID = "#3f3f3f"       #: blocks


def apply(size: float = 8.0, **overrides) -> None:
    """Set the rcParams every figure shares. Extra keys override them.

    Sizes other than the base `size` stay with the figure that chose them: this
    sets the style, not the layout.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": SANS,
        "mathtext.fontset": MATHTEXT,
        "font.size": size,
        # no page fill, no panel fill
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "savefig.edgecolor": PAPER,
        # text, none of it light
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        # recessive structure
        "axes.edgecolor": INK_2,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.color": RULE,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        # TrueType, not the Type 3 default: text stays text and no camera-ready
        # checker objects.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # SVG keeps text as <text> elements (Illustrator-editable), not paths.
        "svg.fonttype": "none",
        **overrides,
    })


# --- CVPR 2027 page geometry -------------------------------------------------

#: CVPR two-column page: one column and the full text width, in inches. A figure
#: authored at exactly one of these widths and included at `\linewidth` (or
#: `\textwidth` in a `figure*`) prints every point size at its own value.
CVPR_COL = 3.25
CVPR_FULL = 6.875
#: Smallest type allowed at printed size.
MIN_PT = 7.0


def save_all(fig, out_dir, name: str, *, dpi: int = 300, tight: bool = True,
             pad: float = 0.02, exts=("pdf", "svg", "png")) -> None:
    """Write `name`.{pdf,svg,png} into `out_dir`.

    PDF: TrueType-embedded live text (fonttype 42). SVG: live <text> elements
    (svg.fonttype none) and raster tiles embedded as data-URI <image> elements.
    PNG: preview only.
    """
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = {"dpi": dpi, "facecolor": PAPER}
    if tight:
        kw.update(bbox_inches="tight", pad_inches=pad)
    for ext in exts:
        fig.savefig(out_dir / f"{name}.{ext}", **kw)
    print(f"  wrote {out_dir}/{name}.{{{','.join(exts)}}}")


if __name__ == "__main__":
    print(f"font chain  : {SANS}")
    print(f"resolves to : {resolved_font()}")
    for name in ("ORANGE", "BLUE", "ORANGE_DEEP", "INK", "INK_2",
                 "PAPER", "SHADE", "RULE", "VOID", "SOLID"):
        print(f"  {name:<12} {globals()[name]}")
