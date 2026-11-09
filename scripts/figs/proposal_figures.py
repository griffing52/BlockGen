"""Proposal figures: hand-laid SVG (live text, Illustrator-editable) + PDF + PNG.

    .venv/bin/python scripts/figs/proposal_figures.py            # all figures
    .venv/bin/python scripts/figs/proposal_figures.py overview   # one figure
    .venv/bin/python scripts/figs/proposal_figures.py --render   # re-render the image assets first

Every figure is plain SVG with <text> elements (no outlined glyphs, no SVG
patterns or markers, which Illustrator imports badly). Arrowheads, check marks
and crosses are explicit paths. Raster imagery (voxel renders from
blockgen.renderer.textured, one cached LDraw OMR render) is embedded base64 and
also kept loose under deliverables/figures/assets/ for relinking in Illustrator.
Inkscape converts SVG -> PDF (text kept as text) and SVG -> PNG preview.

Output: deliverables/figures/proposal_<name>.{svg,pdf,png}

Style: palette from scripts/figs/figstyle.py, plus two port hues (inputs BLUE,
outputs ORANGE) and two status hues (VALID green, FAIL red). Status is never
colour-alone: valid = solid line + check, masked/fail = dashed grey + cross.
"""
from __future__ import annotations

import base64
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figs"))
import figstyle as fs  # noqa: E402

OUT = ROOT / "deliverables" / "figures"
ASSETS = OUT / "assets"
LEGO_CABIN = ROOT / "outputs" / "figstage" / "lego_ldraw_work" / "31048-1_Small-Cabin_ext.png"

FONT = "Helvetica, Arial, 'Liberation Sans', sans-serif"
INK, INK2, RULE, SHADE = fs.INK, fs.INK_2, fs.RULE, fs.SHADE
BLUE, ORANGE = fs.BLUE, fs.ORANGE          # port types: in / out
VALID = "#2f8f4e"                          # one accent for "valid"
FAIL = "#d0342c"                           # one accent for "fail"
GREY = "#9a9a9a"                           # masked candidates
BLOCK = "#d9d9d9"


# ------------------------------------------------------------------- assets
def build_house(wrong: bool):
    """A 9x8x9 cabin. `wrong` fills the top of the doorway with one red block."""
    import numpy as np
    ids = np.zeros((9, 8, 9), np.int32)
    data = np.zeros_like(ids)
    X, Y, Z = ids.shape
    ids[:, 0, :] = 4                                   # cobblestone floor
    for x in range(X):
        for z in range(Z):
            if x in (0, X - 1) or z in (0, Z - 1):
                ids[x, 1:5, z] = 5                     # plank walls
                if x in (0, X - 1) and z in (0, Z - 1):
                    ids[x, 1:5, z] = 17                # log corners
    for z in (2, 6):
        ids[0, 2:4, z] = 20
        ids[X - 1, 2:4, z] = 20                        # glass
    for x in (2, 6):
        ids[x, 2:4, 0] = 20
        ids[x, 2:4, Z - 1] = 20
    ids[4, 1:3, Z - 1] = 0                             # doorway
    for k in range(3):                                 # stepped roof
        y = 5 + k
        ids[k:X - k, y, k:Z - k] = 5
        data[k:X - k, y, k:Z - k] = 1
    if wrong:
        ids[4, 2, Z - 1] = 35
        data[4, 2, Z - 1] = 14                         # red wool seals the door
    return ids, data


def render_assets() -> None:
    """Voxel renders via the project's textured renderer; trimmed to content."""
    import numpy as np
    from PIL import Image
    from blockgen.utils.data import Structure
    from blockgen.renderer.textured import render_structure

    ASSETS.mkdir(parents=True, exist_ok=True)

    def save(img, name):
        im = Image.fromarray(img)
        im = im.crop(im.getchannel("A").getbbox())
        im.save(ASSETS / f"{name}.png")

    for wrong in (False, True):
        ids, data = build_house(wrong)
        save(render_structure(Structure(ids, data), px=900), "house_wrong" if wrong else "house_ok")
    blocks = {"planks": (5, 0), "log": (17, 0), "glass": (20, 0), "lamp": (123, 0),
              "redstone_block": (152, 0), "cobble": (4, 0)}
    for name, (bid, d) in blocks.items():
        b = np.full((1, 1, 1), bid, np.int32)
        dd = np.full((1, 1, 1), d, np.int32)
        save(render_structure(Structure(b, dd), px=240), f"block_{name}")
    if LEGO_CABIN.exists():
        im = Image.open(LEGO_CABIN).convert("RGBA")
        im = im.crop(im.getchannel("A").getbbox())
        im.save(ASSETS / "lego_cabin.png")


def asset(name: str) -> Path:
    p = ASSETS / f"{name}.png"
    if not p.exists():
        render_assets()
    return p


def png_size(p: Path):
    from PIL import Image
    with Image.open(p) as im:
        return im.size


# ---------------------------------------------------------------------- svg
class SVG:
    def __init__(self, w: float, h: float):
        self.w, self.h, self.parts = w, h, []

    def add(self, s: str) -> None:
        self.parts.append(s)

    def g(self, gid: str) -> None:
        self.add(f'<g id="{gid}">')

    def end(self) -> None:
        self.add("</g>")

    def text(self, x, y, s, size=9, weight="normal", anchor="start", fill=INK, italic=False):
        style = f"font-family:{FONT};font-size:{size}px;font-weight:{weight};"
        if italic:
            style += "font-style:italic;"
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" fill="{fill}" '
                 f'style="{style}">{s}</text>')

    def text_sub(self, x, y, s, sub, size=9, anchor="start", fill=INK, italic=True):
        st = f"font-family:{FONT};font-size:{size}px;" + ("font-style:italic;" if italic else "")
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" fill="{fill}" style="{st}">{s}'
                 f'<tspan dy="2.5" style="font-size:{size * 0.75:.1f}px">{sub}</tspan></text>')

    def rect(self, x, y, w, h, fill="none", stroke=INK, sw=0.75, rx=0, dash=None, opacity=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' opacity="{opacity}"' if opacity is not None else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}{o}/>')

    def line(self, x1, y1, x2, y2, stroke=INK, sw=0.75, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                 f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"{d}/>')

    def path(self, d, fill="none", stroke=INK, sw=0.75, dash=None):
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
                 f'stroke-linecap="round" stroke-linejoin="round"{dd}/>')

    def circle(self, cx, cy, r, fill="none", stroke=INK, sw=0.75):
        self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"/>')

    def ellipse(self, cx, cy, rx, ry, fill="none", stroke=INK, sw=0.6):
        self.add(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"/>')

    def poly(self, pts, fill=INK, stroke="none", sw=0.75):
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.add(f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def image(self, path: Path, x, y, w=None, h=None, opacity=None):
        """Embed a PNG, keeping its aspect ratio inside (w, h). Returns the drawn box."""
        pw, ph = png_size(path)
        if w is None:
            w = h * pw / ph
        if h is None:
            h = w * ph / pw
        sc = min(w / pw, h / ph)
        dw, dh = pw * sc, ph * sc
        dx, dy = x + (w - dw) / 2, y + (h - dh) / 2
        b64 = base64.b64encode(path.read_bytes()).decode()
        o = f' opacity="{opacity}"' if opacity is not None else ""
        self.add(f'<image x="{dx:.1f}" y="{dy:.1f}" width="{dw:.1f}" height="{dh:.1f}"{o} '
                 f'xlink:href="data:image/png;base64,{b64}"/>')
        return dx, dy, dw, dh

    # --- compound marks -----------------------------------------------------
    def arrowhead(self, x, y, angle, color=INK, size=4.0):
        a = math.radians(angle)
        l = (x - size * math.cos(a) + size * 0.5 * math.sin(a),
             y - size * math.sin(a) - size * 0.5 * math.cos(a))
        r = (x - size * math.cos(a) - size * 0.5 * math.sin(a),
             y - size * math.sin(a) + size * 0.5 * math.cos(a))
        self.poly([(x, y), l, r], fill=color)

    def arrow(self, x1, y1, x2, y2, color=INK, sw=0.9, dash=None, head=4.0):
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        dx, dy = x2 - x1, y2 - y1
        n = math.hypot(dx, dy) or 1
        self.line(x1, y1, x2 - dx / n * head * 0.8, y2 - dy / n * head * 0.8,
                  stroke=color, sw=sw, dash=dash)
        self.arrowhead(x2, y2, ang, color=color, size=head)

    def curve_arrow(self, x1, y1, cx, cy, x2, y2, color=INK, sw=0.9, dash=None, head=4.0):
        ang = math.degrees(math.atan2(y2 - cy, x2 - cx))
        self.path(f"M{x1:.1f},{y1:.1f} Q{cx:.1f},{cy:.1f} {x2:.1f},{y2:.1f}",
                  stroke=color, sw=sw, dash=dash)
        self.arrowhead(x2, y2, ang, color=color, size=head)

    def check(self, x, y, color=VALID, s=3.2, sw=1.4):
        self.path(f"M{x - s:.1f},{y:.1f} L{x - s * 0.3:.1f},{y + s * 0.8:.1f} "
                  f"L{x + s:.1f},{y - s * 0.9:.1f}", stroke=color, sw=sw)

    def cross(self, x, y, color=FAIL, s=2.8, sw=1.4):
        self.line(x - s, y - s, x + s, y + s, stroke=color, sw=sw)
        self.line(x - s, y + s, x + s, y - s, stroke=color, sw=sw)

    def badge_check(self, x, y, r=6.5):
        self.circle(x, y, r, fill=VALID, stroke=fs.PAPER, sw=1)
        self.check(x, y + 0.5, color=fs.PAPER, s=3.2, sw=1.6)

    def badge_cross(self, x, y, r=6.5):
        self.circle(x, y, r, fill=FAIL, stroke=fs.PAPER, sw=1)
        self.cross(x, y, color=fs.PAPER, s=2.8, sw=1.6)

    def port_in(self, x, y, r=2.8):      # inputs: blue filled circle
        self.circle(x, y, r, fill=BLUE, stroke=fs.PAPER, sw=0.8)

    def port_out(self, x, y, r=2.8):     # outputs: orange filled square
        self.rect(x - r, y - r, 2 * r, 2 * r, fill=ORANGE, stroke=fs.PAPER, sw=0.8)

    def port_attach(self, x, y, r=2.4):  # undirected attach (block faces): open circle
        self.circle(x, y, r, fill=fs.PAPER, stroke=INK2, sw=0.9)

    def nand(self, x, y, w=16, h=12, stroke=INK, sw=0.9):
        r = h / 2
        self.path(f"M{x:.1f},{y:.1f} L{x + w - r:.1f},{y:.1f} "
                  f"A{r:.1f},{r:.1f} 0 0 1 {x + w - r:.1f},{y + h:.1f} "
                  f"L{x:.1f},{y + h:.1f} Z", fill=fs.PAPER, stroke=stroke, sw=sw)
        self.circle(x + w + 1.8, y + r, 1.8, fill=fs.PAPER, stroke=stroke, sw=sw)

    def lever(self, x, y, s=18):
        """A lever on a cobblestone-ish base, vector."""
        self.rect(x, y + s * 0.62, s, s * 0.3, fill=BLOCK, stroke=INK2, sw=0.6, rx=1)
        self.line(x + s * 0.5, y + s * 0.7, x + s * 0.82, y + s * 0.12, stroke=INK, sw=1.6)
        self.circle(x + s * 0.84, y + s * 0.1, 1.6, fill=INK, stroke="none")

    def brick(self, x, y, w=28, cols=4, rows=2, fill="#d94848"):
        """Isometric LEGO brick (cols x rows studs) with top-left of the front face at (x, y)."""
        from colorsys import rgb_to_hls, hls_to_rgb
        from matplotlib.colors import to_rgb, to_hex

        def shade(c, k):
            h, l, s = rgb_to_hls(*to_rgb(c))
            return to_hex(hls_to_rgb(h, max(0, min(1, l * k)), s))
        u = w / cols            # one stud pitch
        d = u * 0.55            # depth per row (isometric foreshortening)
        h = u * 0.95            # brick height
        depth = d * rows
        top = [(x, y), (x + depth, y - depth * 0.6), (x + w + depth, y - depth * 0.6), (x + w, y)]
        self.poly([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], fill=fill, stroke=INK2, sw=0.5)
        self.poly([(x + w, y), (x + w + depth, y - depth * 0.6), (x + w + depth, y - depth * 0.6 + h),
                   (x + w, y + h)], fill=shade(fill, 0.75), stroke=INK2, sw=0.5)
        self.poly(top, fill=shade(fill, 1.25), stroke=INK2, sw=0.5)
        for r in range(rows):
            for c in range(cols):
                cx = x + (c + 0.5) * u + (r + 0.5) * d
                cy = y - (r + 0.5) * d * 0.6
                sr = u * 0.3
                self.ellipse(cx, cy - sr * 0.55, sr, sr * 0.6, fill=shade(fill, 1.1), stroke=INK2)
                self.rect(cx - sr, cy - sr * 0.55 - sr * 0.6, 2 * sr, sr * 0.6, fill=shade(fill, 1.1), stroke="none")
                self.line(cx - sr, cy - sr * 1.15, cx - sr, cy - sr * 0.55, stroke=INK2, sw=0.5)
                self.line(cx + sr, cy - sr * 1.15, cx + sr, cy - sr * 0.55, stroke=INK2, sw=0.5)
                self.ellipse(cx, cy - sr * 1.15, sr, sr * 0.6, fill=shade(fill, 1.35), stroke=INK2)

    def save(self, stem: str) -> Path:
        OUT.mkdir(parents=True, exist_ok=True)
        svg = OUT / f"{stem}.svg"
        body = "\n".join(self.parts)
        svg.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{self.w}pt" height="{self.h}pt" viewBox="0 0 {self.w} {self.h}">\n'
            f'<rect width="{self.w}" height="{self.h}" fill="{fs.PAPER}"/>\n{body}\n</svg>\n')
        pdf, png = svg.with_suffix(".pdf"), svg.with_suffix(".png")
        subprocess.run(["inkscape", str(svg), "--export-type=pdf", f"--export-filename={pdf}"],
                       check=True, capture_output=True)
        subprocess.run(["inkscape", str(svg), "--export-type=png", "--export-dpi=220",
                        "--export-background=white", f"--export-filename={png}"],
                       check=True, capture_output=True)
        return svg


def card(S: SVG, x, y, w, h, title, ins=(), outs=(), attach=False, dashed=False,
         image=None, icon=None, note=None, lines=()):
    """A datasheet-style property card: picture left, title + optional lines right, ports on edges."""
    stroke = INK2 if dashed else INK
    S.rect(x, y, w, h, fill=fs.PAPER, stroke=stroke, sw=0.8, rx=3, dash="3,2" if dashed else None)
    tx = x + 8
    if image is not None:
        S.image(image, x + 7, y + 6, w=h - 12, h=h - 12)
        tx = x + h + 2
    elif icon == "nand":
        S.nand(x + 9, y + h / 2 - 7, w=16, h=13)
        tx = x + 34
    elif icon == "brick":
        S.brick(x + 8, y + h / 2 + 2, w=22, cols=4, rows=2)
        tx = x + 46
    elif icon == "unknown":
        S.rect(x + 8, y + 7, h - 14, h - 14, fill="none", stroke=INK2, sw=0.8, rx=2, dash="2,2")
        S.text(x + 8 + (h - 14) / 2, y + h / 2 + 4, "?", size=12, anchor="middle", fill=INK2)
        tx = x + h + 2
    S.text(tx, y + 13, title, size=9, weight="bold", fill=stroke)
    for i, ln in enumerate(lines):
        S.text(tx, y + 24 + i * 11, ln, size=9, fill=INK2)
    n = len(ins)
    for i, lab in enumerate(ins):
        py = y + h * (i + 1) / (n + 1)
        S.port_in(x, py)
        if lab:
            S.text(x - 5.5, py + 3, lab, size=9, anchor="end", fill=INK2)
    n = len(outs)
    for i, lab in enumerate(outs):
        py = y + h * (i + 1) / (n + 1)
        S.port_out(x + w, py)
        if lab:
            S.text(x + w + 5.5, py + 3, lab, size=9, fill=INK2)
    if attach:
        for px in (x + w * 0.3, x + w * 0.7):
            S.port_attach(px, y)
            S.port_attach(px, y + h)
        S.port_attach(x, y + h / 2)
        S.port_attach(x + w, y + h / 2)
    if note:
        S.text(tx, y + h - 6, note, size=9, fill=INK2, italic=True)


# ------------------------------------------------------------------ figure 1
def fig_overview() -> Path:
    W, H = 504, 250
    S = SVG(W, H)
    ax0, ax1 = 0, 124
    bx0, bx1 = 134, 282
    cx0, cx1 = 292, 504
    S.line(ax1 + 5, 6, ax1 + 5, 206, stroke=RULE)
    S.line(bx1 + 5, 6, bx1 + 5, 206, stroke=RULE)

    # ---------------------------------------------------------------- (a)
    S.g("panel-a")
    S.text(ax0, 14, "(a) Predict all cells", size=10, weight="bold")
    S.text(ax0, 26, "one shot, no checker", size=9, fill=INK2)
    hx, hy, hw, hh = S.image(asset("house_wrong"), ax0 + 2, 32, w=118, h=118)
    # the wrong cell sits above the doorway, at about (0.27, 0.61) of the trimmed render
    wx, wy = hx + hw * 0.27, hy + hh * 0.61
    S.circle(wx, wy, 9, fill="none", stroke=FAIL, sw=1.2)
    S.line(wx + 9, wy, ax0 + 112, 160, stroke=FAIL, sw=0.8)
    S.badge_cross(ax0 + 112, 168)
    S.text(ax0 + 102, 171, "sealed door", size=9, anchor="end", fill=FAIL)
    S.text(ax0, 196, "Looks right,", size=10, weight="bold", italic=True)
    S.text(ax0, 207, "doesn’t work.", size=10, weight="bold", italic=True)
    S.end()

    # ---------------------------------------------------------------- (b)
    S.g("panel-b")
    S.text(bx0, 14, "(b) Property cards", size=10, weight="bold")
    S.text(bx0, 26, "what a part is and how it connects", size=9, fill=INK2)
    cw, ch = 118, 46
    cx = bx0 + 14
    card(S, cx, 36, cw, ch, "wood block", attach=True, image=asset("block_planks"),
         lines=["solid · flammable"])
    card(S, cx, 94, cw, ch, "NAND", ins=("A", "B"), outs=("out",), icon="nand",
         lines=["out = ¬(A ∧ B)"])
    card(S, cx, 152, cw, ch, "comparator", ins=("A", "s"), outs=("out",), dashed=True,
         icon="unknown", note="unseen in training")
    ly = 212
    S.port_in(cx + 2, ly - 3)
    S.text(cx + 7, ly, "in", size=9, fill=INK2)
    S.port_out(cx + 22, ly - 3)
    S.text(cx + 27, ly, "out", size=9, fill=INK2)
    S.port_attach(cx + 48, ly - 3)
    S.text(cx + 53, ly, "attach", size=9, fill=INK2)
    S.badge_cross(cx + 92, ly - 3, r=4.5)
    S.text(cx + 99, ly, "masked", size=9, fill=INK2)
    S.end()

    # ---------------------------------------------------------------- (c)
    S.g("panel-c")
    S.text(cx0, 14, "(c) Constructive policy", size=10, weight="bold")
    S.text(cx0, 26, "grow the graph one typed edge at a time", size=9, fill=INK2)

    gx0, gy0, gw, gh = cx0 + 2, 34, 128, 118
    S.rect(gx0, gy0, gw, gh, fill="none", stroke=RULE, sw=0.6, rx=3)
    S.text_sub(gx0 + 5, gy0 + 12, "G", "t", size=10)

    # nodes: lever (vector), NAND (symbol), lamp (render)
    lx, ly_ = gx0 + 10, gy0 + 20
    S.rect(lx, ly_, 24, 24, fill=fs.PAPER, stroke=INK, sw=0.8, rx=2)
    S.lever(lx + 3, ly_ + 3, s=18)
    lev_out = (lx + 24, ly_ + 12)
    S.port_out(*lev_out)

    nx, ny = gx0 + 50, gy0 + 52
    S.rect(nx, ny, 38, 26, fill=fs.PAPER, stroke=INK, sw=0.8, rx=2)
    S.nand(nx + 8, ny + 6, w=18, h=14)
    nand_in = [(nx, ny + 26 / 3), (nx, ny + 52 / 3)]
    nand_out = (nx + 38, ny + 13)
    for p in nand_in:
        S.port_in(*p)
    S.port_out(*nand_out)

    px_, py_ = gx0 + 96, gy0 + 86
    S.rect(px_, py_, 24, 24, fill=fs.PAPER, stroke=INK, sw=0.8, rx=2)
    S.image(asset("block_lamp"), px_ + 3, py_ + 3, w=18, h=18)
    lamp_in = (px_, py_ + 12)
    S.port_in(*lamp_in)

    # existing edge lever.out -> NAND.in A
    S.curve_arrow(lev_out[0] + 3, lev_out[1], nx - 2, ly_ + 16, nand_in[0][0] - 3.5, nand_in[0][1],
                  color=INK, sw=0.9)
    # candidate 1: valid, NAND.out -> lamp.in
    S.curve_arrow(nand_out[0] + 3, nand_out[1], lamp_in[0] - 8, nand_out[1] + 6,
                  lamp_in[0] - 3.5, lamp_in[1], color=VALID, sw=1.2)
    S.badge_check(gx0 + 86, gy0 + 76, r=5.5)
    # candidate 2: masked out -> out, NAND.out -> lever.out, over the top
    S.curve_arrow(nand_out[0] + 3, nand_out[1] - 3, gx0 + 118, gy0 + 4, lev_out[0] + 6, lev_out[1] - 3,
                  color=GREY, sw=0.9, dash="2.5,2", head=3.5)
    S.badge_cross(gx0 + 86, gy0 + 22, r=5.5)
    S.text(gx0 + 94, gy0 + 25, "out→out", size=9, fill=INK2)
    # candidate 3: masked overlap, ghost block over the NAND corner
    S.rect(gx0 + 26, gy0 + 68, 24, 24, fill=fs.PAPER, stroke=GREY, sw=0.8, rx=2, dash="2.5,2")
    S.image(asset("block_planks"), gx0 + 29, gy0 + 71, w=18, h=18, opacity=0.45)
    S.badge_cross(gx0 + 18, gy0 + 100, r=5.5)
    S.text(gx0 + 26, gy0 + 103, "overlap", size=9, fill=INK2)

    # policy box + loop
    bx, by, bw, bh = gx0, gy0 + gh + 12, gw, 24
    S.rect(bx, by, bw, bh, fill=SHADE, stroke="none", rx=3)
    S.text(bx + bw / 2 - 20, by + 15.5, "π(u, v, e | ", size=10, anchor="middle", italic=True)
    S.text_sub(bx + bw / 2 + 7, by + 15.5, "G", "t", size=10)
    S.text(bx + bw / 2 + 17, by + 15.5, ")", size=10, italic=True)
    S.arrow(gx0 + gw / 2, gy0 + gh, gx0 + gw / 2, by - 1, color=INK, sw=0.9)
    S.path(f"M{bx:.1f},{by + bh / 2:.1f} L{bx - 7:.1f},{by + bh / 2:.1f} L{bx - 7:.1f},{gy0 + gh / 2:.1f}",
           stroke=INK, sw=0.9)
    S.arrowhead(bx - 1, gy0 + gh / 2, 0, color=INK, size=4)
    ry = (by + gy0 + gh / 2) / 2
    S.add(f'<text x="{bx - 9.5}" y="{ry + 10}" text-anchor="middle" fill="{INK2}" '
          f'transform="rotate(-90 {bx - 9.5} {ry + 10})" style="font-family:{FONT};font-size:9px">repeat</text>')

    # execute box
    ex, ey, ew, eh = cx0 + 140, 34, 72, 154
    S.rect(ex, ey, ew, eh, fill=fs.PAPER, stroke=INK, sw=0.9, rx=3)
    S.text(ex + ew / 2, ey + 13, "Execute", size=10, weight="bold", anchor="middle")
    S.arrow(gx0 + gw + 1, gy0 + gh / 2, ex - 1, gy0 + gh / 2, color=INK, sw=0.9)
    tx, ty = ex + 10, ey + 32
    S.text(tx, ty, "A", size=9, weight="bold")
    S.text(tx + 14, ty, "B", size=9, weight="bold")
    S.text(tx + 30, ty, "out", size=9, weight="bold")
    S.line(tx - 2, ty + 3, tx + 50, ty + 3, stroke=INK, sw=0.6)
    for i, (a, b, o) in enumerate([("0", "0", "1"), ("0", "1", "1"), ("1", "0", "1"), ("1", "1", "0")]):
        yy = ty + 13 + i * 10.5
        S.text(tx, yy, a, size=9)
        S.text(tx + 14, yy, b, size=9)
        S.text(tx + 30, yy, o, size=9)
    S.badge_check(ex + ew - 12, ty + 32)
    S.line(ex + 8, ey + 92, ex + ew - 8, ey + 92, stroke=RULE)
    S.image(asset("lego_cabin"), ex + 6, ey + 96, w=ew - 12, h=42)
    S.badge_check(ex + ew - 12, ey + 145)
    S.text(ex + 8, ey + 148, "stands", size=9, fill=INK2)
    S.end()

    # ------------------------------------------------------------ aims strip
    S.g("aims")
    S.line(0, 222, W, 222, stroke=RULE)

    def aim(xc, n, what):
        S.text(xc, 238, f"Aim {n}", size=9, weight="bold", anchor="middle")
        S.text(xc, 248, what, size=9, anchor="middle", fill=INK2)

    aim((bx0 + bx1) / 2, 1, "cards for blocks, gates, bricks")
    aim(gx0 + gw / 2, 2, "policy under hard masks")
    aim(ex + ew / 2 - 8, 3, "score by execution")
    S.end()
    return S.save("proposal_overview")


# ------------------------------------------------------------------ figure 2
def fig_card_anatomy() -> Path:
    """One card dissected, then the same schema in three mediums (pictures, few words)."""
    W, H = 504, 204
    S = SVG(W, H)
    S.text(0, 12, "A property card", size=10, weight="bold")

    x, y, w, h = 20, 22, 150, 100
    S.rect(x, y, w, h, fill=fs.PAPER, stroke=INK, sw=0.9, rx=4)
    S.nand(x + 10, y + 10, w=20, h=15)
    S.text(x + 40, y + 20, "NAND", size=11, weight="bold")
    S.line(x + 8, y + 34, x + w - 8, y + 34, stroke=RULE)
    rows = [("footprint", "3×1×2"), ("ports", "A:in  B:in  out:out"),
            ("behaviour", "out = ¬(A ∧ B)"), ("test", "truth table")]
    for i, (k, v) in enumerate(rows):
        yy = y + 48 + i * 13
        S.text(x + 10, yy, k, size=9, fill=INK2)
        S.text(x + 62, yy, v, size=9)
    for i, lab in enumerate(("A", "B")):
        py = y + h * (i + 1) / 3
        S.port_in(x, py)
        S.text(x - 6, py + 3, lab, size=9, anchor="end", fill=INK2)
    S.port_out(x + w, y + h / 2)

    cx = x + w + 46
    notes = [(y + 48, "geometry mask"), (y + 61, "type mask"),
             (y + 74, "a function, not a part ID"), (y + 87, "executable check")]
    for yy, s in notes:
        S.line(x + w + 3, yy - 3, cx - 4, yy - 3, stroke=RULE, sw=0.6, dash="1.5,2")
        S.text(cx, yy, s, size=9, fill=INK2)

    # three mediums, pictures first
    yy = 142
    S.line(0, yy - 10, W, yy - 10, stroke=RULE)
    cw, chh = 128, 42
    card(S, 20, yy, cw, chh, "wood block", attach=True, image=asset("block_planks"))
    card(S, 188, yy, cw, chh, "lamp", ins=("",), image=asset("block_lamp"))
    card(S, 356, yy, cw, chh, "brick 2×4", ins=("",), outs=("",), icon="brick")
    S.text(20 + cw / 2, yy + chh + 12, "Minecraft block", size=9, anchor="middle", fill=INK2)
    S.text(188 + cw / 2, yy + chh + 12, "Redstone part", size=9, anchor="middle", fill=INK2)
    S.text(356 + cw / 2, yy + chh + 12, "LEGO part", size=9, anchor="middle", fill=INK2)
    return S.save("proposal_card_anatomy")


# ------------------------------------------------------------------ figure 3
def fig_action_mask() -> Path:
    """The action (u, v, e) and three masks, drawn as pictograms."""
    W, H = 504, 132
    S = SVG(W, H)
    S.text(0, 12, "An action is one typed edge (u, v, e). Three masks decide whether it can be proposed.",
           size=10, weight="bold")

    # the action, as pictures: planks block -> lamp
    S.g("action")
    S.rect(0, 30, 44, 44, fill=fs.PAPER, stroke=INK, rx=3)
    S.image(asset("block_planks"), 5, 35, w=34, h=34)
    S.text(22, 86, "u", size=10, anchor="middle", italic=True)
    S.port_out(44, 52)
    S.rect(86, 30, 44, 44, fill=fs.PAPER, stroke=INK, rx=3)
    S.image(asset("block_lamp"), 91, 35, w=34, h=34)
    S.text(108, 86, "v", size=10, anchor="middle", italic=True)
    S.port_in(86, 52)
    S.arrow(49, 52, 82, 52, color=INK)
    S.text(65, 47, "e", size=9, anchor="middle", italic=True)
    S.end()

    # three mask columns, pictogram + one line each
    x0, cw = 170, 112

    def col(i, title, draw_ok, draw_bad):
        x = x0 + i * cw
        S.text(x, 30, title, size=9, weight="bold")
        S.line(x, 34, x + cw - 14, 34, stroke=INK, sw=0.6)
        draw_ok(x, 42)
        S.badge_check(x + cw - 11, 52, r=5.5)
        draw_bad(x, 68)
        S.badge_cross(x + cw - 11, 78, r=5.5)
        if i < 2:
            S.text(x + cw - 1, 64, "∧", size=11, anchor="middle")

    def ports_ok(x, y):
        S.port_out(x + 8, y + 10)
        S.arrow(x + 13, y + 10, x + 40, y + 10, color=INK)
        S.port_in(x + 44, y + 10)
        S.text(x + 50, y + 13, "out→in", size=9, fill=INK2)

    def ports_bad(x, y):
        S.port_out(x + 8, y + 10)
        S.arrow(x + 13, y + 10, x + 40, y + 10, color=GREY, dash="2,2")
        S.port_out(x + 44, y + 10)
        S.text(x + 50, y + 13, "out→out", size=9, fill=INK2)

    def geom_ok(x, y):
        S.rect(x + 4, y + 2, 16, 16, fill=BLOCK, stroke=INK2, sw=0.6)
        S.rect(x + 22, y + 2, 16, 16, fill=BLOCK, stroke=INK2, sw=0.6)
        S.text(x + 50, y + 13, "disjoint", size=9, fill=INK2)

    def geom_bad(x, y):
        S.rect(x + 4, y + 2, 16, 16, fill=BLOCK, stroke=INK2, sw=0.6)
        S.rect(x + 14, y + 2, 16, 16, fill=FAIL, stroke=INK2, sw=0.6, opacity=0.55)
        S.text(x + 50, y + 13, "overlap", size=9, fill=INK2)

    def budget_ok(x, y):
        for i in range(5):
            S.rect(x + 4 + i * 7, y + 6, 5, 8, fill=INK2 if i < 3 else "none", stroke=INK2, sw=0.6)
        S.text(x + 50, y + 13, "3 of 5", size=9, fill=INK2)

    def budget_bad(x, y):
        for i in range(5):
            S.rect(x + 4 + i * 7, y + 6, 5, 8, fill=INK2, stroke=INK2, sw=0.6)
        S.rect(x + 40, y + 6, 5, 8, fill=FAIL, stroke=FAIL, sw=0.6)
        S.text(x + 50, y + 13, "6 of 5", size=9, fill=INK2)

    col(0, "port types", ports_ok, ports_bad)
    col(1, "geometry", geom_ok, geom_bad)
    col(2, "budget", budget_ok, budget_bad)

    S.line(0, 100, W, 100, stroke=RULE)
    S.text(0, 116, "Masked actions have probability zero: invalid graphs are unreachable, so only semantics is left to learn.",
           size=9, fill=INK2)
    return S.save("proposal_action_mask")


# ------------------------------------------------------------------ figure 4
def fig_compounding() -> Path:
    """Why joint prediction fails: P(all N cells right) = p^N. Illustrative arithmetic."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs.apply(size=9)
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    N = np.arange(1, 2001)
    shades = ["#9dbde8", "#5a93dc", BLUE]
    for p, c, yl in zip((0.99, 0.995, 0.999), shades, (0.8, 0.5, 0.3)):
        ax.plot(N, p ** N, color=c, lw=1.8, solid_capstyle="round")
        xl = math.log(yl) / math.log(p)
        ax.text(xl + 40, yl + 0.01, f"p = {p}", color=INK, fontsize=8, ha="left", va="bottom")
    ax.axvline(500, color=RULE, lw=0.8, ls=(0, (2, 2)))
    ax.text(515, 0.97, "a 500-block house", fontsize=8, color=INK2, va="top")
    ax.set_xlabel("cells in the build, N")
    ax.set_ylabel("P(every cell correct) = pᴺ")
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Per-cell accuracy does not survive the build", fontsize=9, loc="left")
    fig.text(0.01, 0.01, "Illustrative: independent per-cell errors.", fontsize=7, color=INK2)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"proposal_compounding.{ext}", dpi=220)
    plt.close(fig)
    return OUT / "proposal_compounding.svg"


# ------------------------------------------------------------------ figure 5
def fig_timeline() -> Path:
    """Three-year plan: aims as swimlanes, milestones as diamonds. Dates are placeholders."""
    W, H = 504, 146
    S = SVG(W, H)
    S.text(0, 12, "Timeline", size=10, weight="bold")
    S.text(44, 12, "(placeholder dates)", size=9, fill=INK2)
    x0, x1, y0 = 150, 490, 24
    years = 3
    yw = (x1 - x0) / years
    for i in range(years):
        x = x0 + i * yw
        S.line(x, y0, x, y0 + 100, stroke=RULE)
        S.text(x + yw / 2, y0 + 8, f"Year {i + 1}", size=9, anchor="middle", fill=INK2)
    S.line(x1, y0, x1, y0 + 100, stroke=RULE)
    lanes = [
        ("Aim 1  card schema + deck", [(0.0, 1.2), (2.2, 2.6)], [(1.2, "deck v1"), (2.6, "LEGO cards")]),
        ("Aim 2  constructive policy", [(0.6, 2.4)], [(1.5, "Redstone"), (2.4, "transfer")]),
        ("Aim 3  execution benchmark", [(0.3, 0.9), (1.8, 3.0)], [(0.9, "harness"), (3.0, "release")]),
    ]
    for j, (name, bars, marks) in enumerate(lanes):
        y = y0 + 24 + j * 24
        S.text(x0 - 8, y + 6, name, size=9, anchor="end")
        for a, b in bars:
            S.rect(x0 + a * yw, y - 1, (b - a) * yw, 12, fill=SHADE, stroke="none", rx=2)
        for t, lab in marks:
            mx, my = x0 + t * yw, y + 5
            S.poly([(mx, my - 5), (mx + 5, my), (mx, my + 5), (mx - 5, my)], fill=INK)
            S.text(mx, y - 4, lab, size=9, anchor="middle", fill=INK2)
    S.line(x0, y0 + 100, x1, y0 + 100, stroke=INK, sw=0.6)
    S.text(0, 140, "Bars: active work. Diamonds: deliverables.", size=9, fill=INK2)
    return S.save("proposal_timeline")


# ------------------------------------------------------------------ figure 6
def fig_gallery() -> Path:
    """Loose imagery strip: the same cabin right and wrong, the block deck, the LEGO cabin."""
    W, H = 504, 150
    S = SVG(W, H)
    S.image(asset("house_ok"), 0, 10, w=120, h=120)
    S.badge_check(108, 124)
    S.text(60, 144, "executes", size=9, anchor="middle", fill=INK2)
    S.image(asset("house_wrong"), 130, 10, w=120, h=120)
    S.badge_cross(238, 124)
    S.text(190, 144, "one wrong cell", size=9, anchor="middle", fill=INK2)
    names = ["planks", "log", "cobble", "glass", "lamp", "redstone_block"]
    for i, n in enumerate(names):
        r, c = divmod(i, 3)
        S.image(asset(f"block_{n}"), 266 + c * 36, 24 + r * 44, w=32, h=32)
    S.text(320, 144, "block cards", size=9, anchor="middle", fill=INK2)
    S.image(asset("lego_cabin"), 384, 10, w=120, h=120)
    S.text(444, 144, "LEGO (OMR, CC BY)", size=9, anchor="middle", fill=INK2)
    return S.save("proposal_gallery")


FIGS = {
    "overview": fig_overview,
    "card_anatomy": fig_card_anatomy,
    "action_mask": fig_action_mask,
    "compounding": fig_compounding,
    "timeline": fig_timeline,
    "gallery": fig_gallery,
}

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--render" in args:
        args.remove("--render")
        render_assets()
    for n in args or list(FIGS):
        print(FIGS[n]())
