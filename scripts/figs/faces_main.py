"""Main-paper figures for the faces paper (deliverables/paper/cvpr27_faces/figures).

    BLOCKGEN_DATA_ROOT=<main checkout> BLOCKGEN_FIG_RENDER=cpu .venv/bin/python scripts/figs/faces_main.py [teaser|method|gallery|all]

teaser   Fig. 1, full width: (a) a real house's six faces and shell, (b) Grammar faces with the cells
         that change under lift-and-reproject tinted red, and their lift, (c) Cross-Face faces of the
         same box and seed, and their lift. One pixels-per-block scale for every render.
         Box rule: among paired draws whose box is at least 14 wide and deep and 10 tall, the one with
         the median Grammar-minus-Cross-Face part reduction. Real rule: among held-out real shells
         whose box is within 2 blocks of that box on every axis (widening to 4, then 8), the one with
         the most distinct block types (the reference shows what a detailed exterior looks like).
method   Fig. 2, full width: (a, b) a horizontal slice of the shell state, (c) one real decode: faces,
         what the rule decided before each cell, lift (the median build by forced share).
gallery  Fig. 4, full width: 6 seeded random boxes; rows Grammar, Grammar + Carve, Cross-Face (same
         boxes and seeds), Cross-Face best-of-4 (own draws), Real-Shell (test split, own draws).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import faces_qual as fq  # noqa: E402  (re-execs with PYTHONHASHSEED=0; puts the checkout on sys.path)
import faces_paper as fp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from figstyle import ORANGE, INK, INK_2, CVPR_FULL  # noqa: E402

OUT = fp.ROOT / "deliverables/paper/cvpr27_faces/figures"
LETTER = dict(zip(fq.FACES, ("F", "R", "B", "L", "T", "Bo")))
TINT = np.array([215, 40, 40], np.float32)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def labelled_sheet(faces, bad=None, scale=7):
    """Face sheet in the fc.sheet layout; `bad` cells blended 50% with red. Returns image, boxes."""
    from blockgen.corpus import facade as fc

    pics = {}
    for f in fq.FACES:
        F = faces[f]
        p = fc.face_picture(F).astype(np.float32)
        if bad is not None:
            m = fq._to_pic(F, bad[f])
            p[m] = 0.5 * p[m] + 0.5 * TINT
        pics[f] = np.repeat(np.repeat(p.astype(np.uint8), scale, 0), scale, 1)
    return fq.sheet_layout(pics, faces, gap=4)


def put_sheet(fig, img, boxes, x0, y0, w_in, fig_w, fig_h):
    """Draw a sheet at (x0, y0) inches (bottom-left), `w_in` wide, with face letters inside each face."""
    h_in = img.shape[0] * w_in / img.shape[1]
    ax = fig.add_axes([x0 / fig_w, y0 / fig_h, w_in / fig_w, h_in / fig_h])
    ax.imshow(img, interpolation="nearest")
    fq.bare(ax)
    for f, (x, y, w, h) in boxes.items():
        ax.text(x + 3, y + 3, LETTER[f], ha="left", va="top", fontsize=6, color="white",
                bbox=dict(boxstyle="square,pad=0.08", fc=(0, 0, 0, 0.55), ec="none"))
    return h_in


def put_img(fig, img, x0, y0, w_in, h_in, fig_w, fig_h):
    ax = fig.add_axes([x0 / fig_w, y0 / fig_h, w_in / fig_w, h_in / fig_h])
    ax.imshow(img, interpolation="lanczos")
    fq.bare(ax)


def _real_pick(box, real):
    from blockgen.corpus import facade as fc  # noqa: F401

    dims = np.array([s.crop_to_non_air().shape for s in real])
    kinds = np.array([len(np.unique(s.block_ids[s.block_ids != 0])) for s in real])
    for tol in (2, 4, 8):
        cand = np.where((np.abs(dims - np.array(box)) <= tol).all(1))[0]
        if len(cand):
            break
    return int(cand[np.argmax(kinds[cand])]), tol


def fig_teaser():
    from blockgen.corpus import facade as fc

    g, c = fq.load_face_sets_cached("grammar"), fq.load_face_sets_cached("cross")
    gl, cl = fq.lift_all(g), fq.lift_all(c)
    pairs = [i for i in range(min(len(g), len(c))) if gl[i] is not None and cl[i] is not None
             and g[i][0][0] >= 14 and g[i][0][2] >= 14 and g[i][0][1] >= 10]
    red = {i: fp.n_parts(gl[i]) - fp.n_parts(cl[i]) for i in pairs}
    order = sorted(pairs, key=lambda i: red[i])
    i = order[len(order) // 2]
    shape = g[i][0]
    real = fq.load_structs(fq.REAL_SHELL)
    j, tol = _real_pick(shape, real)
    rs = real[j].crop_to_non_air()
    rfaces = fc.project_all(rs, fc.ALL_FACES)
    print(f"teaser box {i} {shape}: grammar {fp.n_parts(gl[i])} parts, cross {fp.n_parts(cl[i])}; "
          f"real #{j} {rs.shape} (tol {tol}), {fp.n_parts(rs)} parts")
    _, gbad = fp.disagreement(g[i][1], shape)
    share = np.mean(np.concatenate([b.ravel() for b in gbad.values()]))
    sheets = [labelled_sheet(rfaces), labelled_sheet(g[i][1], gbad), labelled_sheet(c[i][1])]
    ppb = 7.0
    renders = [fq.iso(rs, ppb), fq.iso(gl[i], ppb), fq.iso(cl[i], ppb)]
    titles = ["(a) a real house: faces and shell", "(b) Grammar faces and their lift",
              "(c) Cross-Face, same box and seed"]
    notes = [("consistent", f"{fp.n_parts(rs)} parts"),
             (f"{100 * share:.0f}% of cells disagree (red)", f"{fp.n_parts(gl[i])} parts"),
             ("0% disagree", f"{fp.n_parts(cl[i])} parts")]
    W = CVPR_FULL
    col_w = W / 3
    sheet_w = 0.56 * col_w
    ipp = 0.36 * col_w / max(r.shape[1] for r in renders)          # one scale for every render
    rh = max(r.shape[0] for r in renders) * ipp
    sh = max(s[0].shape[0] * sheet_w / s[0].shape[1] for s in sheets)
    body = max(rh, sh)
    H = body + 0.36
    fig = plt.figure(figsize=(W, H))
    for k in range(3):
        x0 = k * col_w
        img, boxes = sheets[k]
        hs = img.shape[0] * sheet_w / img.shape[1]
        put_sheet(fig, img, boxes, x0 + 0.02, 0.16 + (body - hs) / 2, sheet_w, W, H)
        r = renders[k]
        w_r, h_r = r.shape[1] * ipp, r.shape[0] * ipp
        put_img(fig, r, x0 + sheet_w + 0.06 + (0.36 * col_w - w_r) / 2, 0.16, w_r, h_r, W, H)
        fig.text((x0 + col_w / 2) / W, (H - 0.02) / H, titles[k], ha="center", va="top", fontsize=7.5, color=INK)
        fig.text((x0 + 0.02 + sheet_w / 2) / W, 0.02 / H, notes[k][0], ha="center", va="bottom",
                 fontsize=6.5, color=INK_2)
        fig.text((x0 + sheet_w + 0.06 + 0.18 * col_w) / W, 0.02 / H, notes[k][1], ha="center", va="bottom",
                 fontsize=6.5, color=INK_2)
    save(fig, "fig1_teaser")


def _slice(ax, stage, N=8):
    from figstyle import BLUE  # noqa: F401  (kept out of the schematic: arrows are ink)

    hits = {1: 3, 2: 3, 3: 4, 4: 4, 5: 3, 6: 3}
    known = {(x, z) for x, z in hits.items()}
    empty = {(x, z) for x, zz in hits.items() for z in range(zz)}
    row_z = 4
    for x in range(N):
        for z in range(N):
            face = "#9c6b3c" if (x, z) in known else ("#e6e5df" if (x, z) in empty else "white")
            ax.add_patch(Rectangle((x, z), 1, 1, facecolor=face, edgecolor="#c9c9c4", lw=0.5))
    ax.set_xlim(-0.4, N + 1.5); ax.set_ylim(-1.3, N + 0.2)
    ax.set_aspect("equal"); ax.axis("off")
    if stage == "front":
        for x in hits:
            ax.annotate("", xy=(x + 0.5, -0.05), xytext=(x + 0.5, -1.0),
                        arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    else:
        ax.annotate("", xy=(N + 0.05, row_z + 0.5), xytext=(N + 1.3, row_z + 0.5),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
        first = next(d for d in range(N) if (N - 1 - d, row_z) in known)
        for d in range(N):
            x = N - 1 - d
            if d < first and (x, row_z) not in empty:
                ax.add_patch(Rectangle((x + 0.14, row_z + 0.14), 0.72, 0.72, fill=False, edgecolor=ORANGE, lw=1.4))
            elif d == first:
                ax.add_patch(Rectangle((x + 0.14, row_z + 0.14), 0.72, 0.72, fill=False, edgecolor=ORANGE,
                                       lw=1.4, ls=(0, (2, 1.5))))
            else:
                ax.plot([x + 0.25, x + 0.75], [row_z + 0.25, row_z + 0.75], color=INK, lw=0.7)
                ax.plot([x + 0.25, x + 0.75], [row_z + 0.75, row_z + 0.25], color=INK, lw=0.7)


def fig_method():
    c = fq.load_face_sets_cached("cross")
    cl = fq.lift_all(c)
    ok = [i for i, e in enumerate(c) if e is not None and cl[i] is not None]
    share, masks = {}, {}
    for i in ok:
        m = fq.replay(c[i][1], c[i][0])
        masks[i] = m
        share[i] = sum(int(v[0].sum()) for v in m.values()) / sum(v[0].size for v in m.values())
    order = sorted(ok, key=lambda i: share[i])
    i = order[len(order) // 2]
    print(f"method decode: build {i}, forced share {share[i]:.3f}")
    a, b, boxes = fq.forced_sheets(c[i][1], masks[i], scale=7)
    lift = fq.iso(cl[i], ppb=7)
    W, H = CVPR_FULL, 2.15
    fig = plt.figure(figsize=(W, H))
    sw = 1.25
    for k, (stage, title) in enumerate((("front", "(a) after the front face"),
                                        ("right", "(b) one cell of a later face"))):
        ax = fig.add_axes([(0.02 + k * (sw + 0.05)) / W, 0.42 / H, sw / W, (H - 0.62) / H])
        _slice(ax, stage)
        fig.text((0.02 + k * (sw + 0.05) + sw / 2) / W, (H - 0.03) / H, title, ha="center", va="top",
                 fontsize=7.5, color=INK)
    for col, lines in enumerate((("brown: committed first hit", "grey: seen through, so empty"),
                                 ("orange: legal first hit", "dashed: block forced", "x: illegal"))):
        for k, line in enumerate(lines):
            fig.text((0.02 + col * (sw + 0.05)) / W, (0.34 - k * 0.11) / H, line, fontsize=6.3,
                     color=INK_2, va="top")
    x0 = 2 * sw + 0.22
    rem = W - x0
    w_sheet = 0.36 * rem
    for k, (img, t) in enumerate(((a, "(c) decoded faces"), (b, "what the rule decided"))):
        xs = x0 + k * (w_sheet + 0.06)
        hs = put_sheet(fig, img, boxes, xs, 0.42, w_sheet, W, H)
        fig.text((xs + w_sheet / 2) / W, (H - 0.03) / H, t, ha="center", va="top", fontsize=7.5, color=INK)
    wl = rem - 2 * (w_sheet + 0.06) - 0.02
    hl = lift.shape[0] * wl / lift.shape[1]
    if hl > H - 0.65:
        hl = H - 0.65
        wl = lift.shape[1] * hl / lift.shape[0]
    xl = x0 + 2 * (w_sheet + 0.06)
    put_img(fig, lift, xl, 0.42, wl, hl, W, H)
    fig.text((xl + wl / 2) / W, (H - 0.03) / H, "lift", ha="center", va="top", fontsize=7.5, color=INK)
    y = 0.12
    xs = x0 + 0.15
    o = fq._hex(ORANGE)
    for colr, lab in ((tuple(o / 255), "forced"), (tuple((0.30 * o + 0.70 * 255) / 255), "narrowed"),
                      ("#ececec", "free")):
        fig.patches.append(Rectangle((xs / W, y / H), 0.09 / W, 0.09 / H, transform=fig.transFigure,
                                     facecolor=colr, edgecolor="#c9c9c4", lw=0.4))
        fig.text((xs + 0.12) / W, (y + 0.045) / H, lab, va="center", fontsize=6.5, color=INK)
        xs += 0.75
    fig.text(xs / W, (y + 0.045) / H, f"{100 * share[i]:.0f}% of this build's cells forced", va="center",
             fontsize=6.5, color=INK_2)
    save(fig, "fig2_method")


def fig_gallery(k=6):
    from blockgen.corpus.facade_carve import carve

    g, c, b = (fq.load_face_sets_cached(a) for a in ("grammar", "cross", "cross_bon"))
    gl, cl, bl = fq.lift_all(g), fq.lift_all(c), fq.lift_all(b)
    real = fq.load_structs(fq.REAL_SHELL)
    pair = [i for i in range(min(len(gl), len(cl))) if gl[i] is not None and cl[i] is not None]
    pick = fq.draw(0, pair, 8)[:k]                                 # the first k of the seeded gallery draw
    pick_b = fq.draw(1, [i for i, s in enumerate(bl) if s is not None], 8)[:k]
    pick_r = fq.draw(2, range(len(real)), 8)[:k]
    cv = [carve(g[i][1], g[i][0]) for i in pick]
    ppb = 6.0
    blank = np.full((40, 40, 3), 255, np.uint8)
    rows = [[fq.iso(gl[i], ppb) for i in pick],
            [fq.iso(s, ppb) if s.occupied_mask.any() else blank for s in cv],
            [fq.iso(cl[i], ppb) for i in pick], [fq.iso(bl[i], ppb) for i in pick_b],
            [fq.iso(real[i], ppb) for i in pick_r]]
    notes = [[fq.parts(gl[i]) for i in pick],
             [fq.parts(s) if s.occupied_mask.any() else "carved to nothing" for s in cv],
             [fq.parts(cl[i]) for i in pick], [fq.parts(bl[i]) for i in pick_b],
             [fq.parts(real[i]) for i in pick_r]]
    labels = ["Grammar", "Grammar\n+ Carve", "Cross-Face", "Cross-Face\nbest-of-4", "Real-Shell"]
    subs = ["", "same faces", "same boxes\nand seeds", "own draws", "test split"]
    print("gallery", pick, pick_b, pick_r)
    fig = fq.tile_grid(rows, labels, notes, sublabels=subs, label_w=0.72)
    save(fig, "fig4_gallery")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("teaser", fig_teaser), ("method", fig_method), ("gallery", fig_gallery)):
        if which in (name, "all"):
            fn()


if __name__ == "__main__":
    main()
