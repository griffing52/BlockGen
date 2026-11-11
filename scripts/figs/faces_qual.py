"""Qualitative figure candidates for the cross-face decoding paper (cvpr27_faces).

    BLOCKGEN_FIG_RENDER=cpu BLOCKGEN_DATA_ROOT=/path/to/main/checkout \\
        .venv/bin/python scripts/figs/faces_qual.py [gallery|forced|exteriors|failures|explode|all]

gallery    Uncurated draws: Grammar lift, Cross-Face lift, Cross-Face best-of-4 lift, Real-Shell.
forced     Face sheets with the cells the cross-face rule forces (one legal token) and narrows,
           recomputed by replaying ShellState over the saved faces in decode order.
exteriors  Exteriors against exteriors: Real-Shell, the voxel transformer reduced to its six-face
           shell (recanon20m_s0 shell6), Cross-Face best-of-4.
failures   (a) thin Cross-Face shells (lowest block-count quartile), (b) the RFT + constrained
           best-of-4 arm whose reward saturates (near-identical solid boxes).
explode    One lift coloured by the face that wrote each voxel: whole, cut open, exploded.

Every build shown is chosen by a stated rule or a seeded draw (seeds below), never by eye;
candidates/README.md lists the rules. All renders in one figure share one camera (orthographic,
azimuth 45, elevation 30) and one block size in pixels, so sizes compare across tiles.

Writes PDF + PNG to deliverables/paper/cvpr27_faces/figures/candidates/qual_*.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    # Fallback (untextured) block colours are seeded by Python's salted str hash; pin it.
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import faces_paper as fp  # noqa: E402  (also puts the checkout root on sys.path)
import figstyle  # noqa: E402
from figstyle import BLUE, ORANGE, INK, INK_2, RULE, SHADE, CVPR_FULL, CVPR_COL  # noqa: E402

ROOT, DATA = fp.ROOT, fp.DATA
OUT = ROOT / "deliverables/paper/cvpr27_faces/figures/candidates"
PACK = str(DATA / "data/textures/vanilla")
RUN = fp.RUN
RFT = DATA / "outputs/run_20261101_154120_rft_facades_joint_d4_20m_face_solidity_rsel_t07"
ARM = {
    "grammar": RUN / "constrained_none_t0.7_n399/faces.npz",
    "cross": RUN / "constrained_constrain_t0.7_n399/faces.npz",
    "cross_bon": RUN / "constrained_bon_face_solidity_k4_t0.7_n399/faces.npz",
    "rft_bon": RFT / "constrained_bon_face_solidity_k4_t0.7_n399/faces.npz",
}
REAL_SHELL = DATA / "outputs/bench_arms/real_facades6_32.npz"
VOXEL_SHELL = DATA / "outputs/bench_arms/n256/recanon20m_s0_shell6_32.npz"
FACES = fp.FACES                                     # decode order: front right back left top bottom
FACE_NAME = dict(zip(FACES, ("front", "right", "back", "left", "top", "bottom")))
AZIM, ELEV = 45.0, 30.0
FS = 6.5                                             # body text size in the figures (pt)


# ---------------------------------------------------------------- io
def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def load_structs(path: Path):
    from blockgen.utils.data import Structure

    z = np.load(path, allow_pickle=True)
    return [Structure(block_ids=np.asarray(i, np.int32), block_data=np.asarray(d, np.int32))
            for i, d in zip(z["block_ids"], z["block_data"])]


def lift_all(face_sets):
    """Lifted Structure per face set (None where the set is missing or lifts to nothing)."""
    out = []
    for e in face_sets:
        if e is None:
            out.append(None)
            continue
        s = fp.disagreement(e[1], e[0])[0]
        out.append(s if s.occupied_mask.any() else None)
    return out


def draw(rng_seed: int, pool, k: int):
    """`k` indices drawn without replacement from `pool` with a fixed seed, kept in draw order."""
    pool = np.asarray(sorted(pool))
    return [int(i) for i in np.random.default_rng(rng_seed).choice(pool, size=k, replace=False)]


# ---------------------------------------------------------------- rendering
_LIB = None


def _mag(dims) -> float:
    """Half-extent of the renderer's orthographic frame for a box (textured.render_structure)."""
    import math

    a, e = math.radians(AZIM), math.radians(ELEV)
    d = np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])
    fwd = -d
    right = np.cross(fwd, [0.0, 1.0, 0.0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    c = np.array([(x, y, z) for x in (0, dims[0]) for y in (0, dims[1]) for z in (0, dims[2])], float)
    c -= np.asarray(dims, float) / 2
    return 1.05 * max(np.abs(c @ right).max(), np.abs(c @ up).max())


def iso(struct, ppb: float = 9.0) -> np.ndarray:
    """RGB render on white at `ppb` pixels per block (constant across builds), cropped to the build.
    Set BLOCKGEN_QUAL_CACHE=<dir> to reuse renders across runs (keyed by content and ppb)."""
    import hashlib

    cache = os.environ.get("BLOCKGEN_QUAL_CACHE")
    if cache:
        h = hashlib.sha1(struct.block_ids.tobytes() + struct.block_data.tobytes()
                         + str((struct.shape, ppb, AZIM, ELEV)).encode()).hexdigest()[:20]
        fn = Path(cache) / f"{h}.npy"
        if fn.exists():
            return np.load(fn)
        img = _iso(struct, ppb)
        fn.parent.mkdir(parents=True, exist_ok=True)
        np.save(fn, img)
        return img
    return _iso(struct, ppb)


def _iso(struct, ppb: float) -> np.ndarray:
    import cpu_render
    from PIL import Image

    cpu_render.install()
    global _LIB
    from blockgen.renderer import textured
    from blockgen.renderer.textures import load_face_textures

    if _LIB is None:
        _LIB = load_face_textures(PACK)
    c = struct.crop_to_non_air()
    want = 2 * _mag(c.shape) * ppb
    px = int(np.ceil(want / 32) * 32)
    img = textured.render_structure(c, px=px, azim_deg=AZIM, elev_deg=ELEV, pack_dir=PACK, face_textures=_LIB)
    if px != round(want):
        img = np.asarray(Image.fromarray(img).resize((round(want), round(want)), Image.LANCZOS))
    a = img[..., 3:4].astype(np.float32) / 255.0
    rgb = (img[..., :3].astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
    ys, xs = np.where(img[..., 3] > 8)
    if len(ys):
        rgb = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return rgb


def pad_to(img: np.ndarray, H: int, W: int, align: str = "center") -> np.ndarray:
    """Bottom-aligned on a white H x W canvas (same block size everywhere, ground on one line)."""
    out = np.full((H, W, 3), 255, np.uint8)
    h, w = img.shape[:2]
    x0 = (W - w) // 2 if align == "center" else 0
    out[H - h:, x0:x0 + w] = img
    return out


def bare(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def parts(s) -> str:
    n = fp.n_parts(s)
    return f"{n} part" if n == 1 else f"{n} parts"


def tile_grid(rows, labels, notes=None, width=CVPR_FULL, label_w=0.62, note_fs=6.0, row_gap=0.07,
              col_gap=0.03, labels_above=False, sublabels=None):
    """A grid of renders with one pixel-per-block scale for every tile of the figure.

    Every tile shares the widest render's width; each row is as tall as its tallest render, so
    sizes compare across the whole figure and short rows do not waste height. rows: lists of RGB
    images; labels: one per row, in a left margin (or above the row with `labels_above`);
    sublabels: optional smaller second line per row; notes: a short string under each tile.
    """
    W = max(im.shape[1] for r in rows for im in r)
    ncol = max(len(r) for r in rows)
    if labels_above:
        label_w = 0.0
    tile_w = (width - label_w) / ncol
    inch_per_px = (tile_w - col_gap) / W
    row_h = [max(im.shape[0] for im in r) * inch_per_px for r in rows]
    note_h = 0.10 if notes else 0.0
    head_h = 0.13 if labels_above else 0.0
    fig_h = sum(row_h) + len(rows) * (note_h + head_h) + (len(rows) - 1) * row_gap
    fig = plt.figure(figsize=(width, fig_h))
    top = fig_h
    for r, row in enumerate(rows):
        if labels_above:
            fig.text(0.0, (top - 0.01) / fig_h, labels[r], ha="left", va="top", fontsize=7, color=INK)
            top -= head_h
        Hr = max(im.shape[0] for im in row)
        for c, im in enumerate(row):
            ax = fig.add_axes([(label_w + c * tile_w + col_gap / 2) / width, (top - row_h[r]) / fig_h,
                               (tile_w - col_gap) / width, row_h[r] / fig_h])
            ax.imshow(pad_to(im, Hr, W), interpolation="lanczos")
            bare(ax)
            if notes and notes[r][c]:
                ax.text(0.5, -0.02, notes[r][c], transform=ax.transAxes, ha="center", va="top",
                        fontsize=note_fs, color=INK_2)
        if not labels_above:
            y = (top - row_h[r] / 2) / fig_h
            fig.text(0.0, y, labels[r], ha="left", va="center" if not (sublabels and sublabels[r]) else "bottom",
                     fontsize=7, color=INK)
            if sublabels and sublabels[r]:
                fig.text(0.0, y - 0.02 / fig_h, sublabels[r], ha="left", va="top", fontsize=6, color=INK_2)
        top -= row_h[r] + note_h + row_gap
    return fig


# ---------------------------------------------------------------- 1. gallery
def fig_gallery(k: int = 8) -> None:
    g, c, b = (load_face_sets_cached(a) for a in ("grammar", "cross", "cross_bon"))
    gl, cl, bl = lift_all(g), lift_all(c), lift_all(b)
    real = load_structs(REAL_SHELL)
    pair = [i for i in range(min(len(gl), len(cl))) if gl[i] is not None and cl[i] is not None]
    pick = draw(0, pair, k)
    pick_b = draw(1, [i for i, s in enumerate(bl) if s is not None], k)
    pick_r = draw(2, range(len(real)), k)
    print("gallery grammar/cross", pick, "best-of-4", pick_b, "real", pick_r)
    rows = [[iso(gl[i]) for i in pick], [iso(cl[i]) for i in pick], [iso(bl[i]) for i in pick_b],
            [iso(real[i]) for i in pick_r]]
    notes = [[parts(gl[i]) for i in pick], [parts(cl[i]) for i in pick],
             [parts(bl[i]) for i in pick_b], [parts(real[i]) for i in pick_r]]
    labels = ["Grammar", "Cross-Face", "Cross-Face\nbest-of-4", "Real-Shell"]
    subs = ["", "same boxes and\nseeds as Grammar", "", "test split"]
    fig = tile_grid(rows, labels, notes, sublabels=subs, label_w=0.75)
    save(fig, "qual_gallery")


_FACE_CACHE = {}


def load_face_sets_cached(arm: str):
    if arm not in _FACE_CACHE:
        _FACE_CACHE[arm] = fp.load_face_sets(ARM[arm])
    return _FACE_CACHE[arm]


# ---------------------------------------------------------------- 2. forced cells
def replay(faces, shape):
    """Per face, (forced, narrowed) (u, v) masks seen by each cell just before it is committed, when
    the faces are committed in decode order. forced: exactly one legal token. narrowed: the legal set
    is smaller than the unconstrained one (some depth ruled out, or see-through ruled out)."""
    from blockgen.corpus.facade_constraints import ShellState, FREE

    st = ShellState(shape)
    out = {}
    for name in FACES:
        f = faces[name]
        U, V = f.depth.shape
        forced = np.zeros((U, V), bool)
        narrowed = np.zeros((U, V), bool)
        for v in range(V):
            for u in range(U):
                allowed, frc, allow_empty = st.allowed(name, u, v)
                one = (not allow_empty) and int(allowed.sum()) == 1 and frc[np.argmax(allowed)] != FREE
                forced[u, v] = one
                narrowed[u, v] = (not one) and ((not allow_empty) or (not allowed.all()) or (frc != FREE).any())
                d = int(f.depth[u, v])
                st.commit_cell(name, u, v, d, int(f.ids[u, v]) * 16 + int(f.data[u, v]))
        out[name] = (forced, narrowed)
    return out


def _to_pic(f, m):
    """A (u, v) cell mask in face_picture orientation."""
    from blockgen.corpus import facade as fc

    m = m.T
    if f.axis != 1:
        m = m[::-1]
    if fc._FLIP_U[f.face]:
        m = m[:, ::-1]
    return m


def sheet_layout(pics, faces, gap=3):
    """fc.sheet layout for prepared per-face pictures: sides in a row, top and bottom below.
    Returns the image and each face's (x, y, w, h) box for labelling."""
    rows = [[f for f in FACES if faces[f].axis != 1], [f for f in FACES if faces[f].axis == 1]]
    width = max(sum(pics[f].shape[1] for f in r) + gap * (len(r) + 1) for r in rows)
    height = sum(max(pics[f].shape[0] for f in r) for r in rows) + gap * (len(rows) + 1)
    out = np.full((height, width, 3), 255, np.uint8)
    boxes = {}
    y = gap
    for r in rows:
        x, h = gap, max(pics[f].shape[0] for f in r)
        for f in r:
            p = pics[f]
            out[y + h - p.shape[0]:y + h, x:x + p.shape[1]] = p
            boxes[f] = (x, y + h - p.shape[0], p.shape[1], p.shape[0])
            x += p.shape[1] + gap
        y += h + gap
    return out, boxes


def _hex(c):
    return np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)], np.float32)


def forced_sheets(faces, masks, scale=8):
    """(face colours sheet, constraint-map sheet). The map: forced cells ORANGE, narrowed cells a
    light tint of it, free cells light grey; a see-through cell is drawn white with a grey dot."""
    from blockgen.corpus import facade as fc

    col, cmap = {}, {}
    o = _hex(ORANGE)
    light = 0.30 * o + 0.70 * 255
    free = _hex("#ececec")
    for f in FACES:
        F = faces[f]
        col[f] = np.repeat(np.repeat(fc.face_picture(F), scale, 0), scale, 1)
        forced, narrowed = (_to_pic(F, m) for m in masks[f])
        m = np.empty(forced.shape + (3,), np.float32)
        m[:] = free
        m[narrowed] = light
        m[forced] = o
        big = np.repeat(np.repeat(m, scale, 0), scale, 1).astype(np.uint8)
        # thin white grid so single cells read
        big[::scale] = 255; big[:, ::scale] = 255
        cmap[f] = big
    a, boxes = sheet_layout(col, faces)
    b, _ = sheet_layout(cmap, faces)
    return a, b, boxes


def fig_forced() -> None:
    c = load_face_sets_cached("cross")
    cl = lift_all(c)
    ok = [i for i, e in enumerate(c) if e is not None and cl[i] is not None]
    share = {}
    masks = {}
    tot_forced = tot_cells = 0
    for i in ok:
        shape, faces = c[i]
        m = replay(faces, shape)
        masks[i] = m
        nf = sum(int(v[0].sum()) for v in m.values())
        nc = sum(v[0].size for v in m.values())
        share[i] = nf / nc
        tot_forced += nf; tot_cells += nc
    print(f"replayed forced share over {len(ok)} builds: {tot_forced / tot_cells:.4f} (summary.json 0.2145)")
    order = sorted(ok, key=lambda i: share[i])
    q = [order[int(round(p * (len(order) - 1)))] for p in (0.25, 0.5, 0.75)]
    print("forced picks (25/50/75th pct of forced share)", q, [round(share[i], 3) for i in q])
    sheets = [forced_sheets(c[i][1], masks[i]) for i in q]
    renders = [iso(cl[i], ppb=8) for i in q]
    Ws = max(s[0].shape[1] for s in sheets)
    Wr = max(r.shape[1] for r in renders)
    width = CVPR_FULL
    wr = 1.15                                     # render column width (in)
    ws = (width - wr - 0.16) / 2                  # each sheet column
    ips, ipr = ws / Ws, wr / Wr                   # inches per pixel: one scale per column
    rows_h = [max(sh[0].shape[0] * ips, im.shape[0] * ipr) for sh, im in zip(sheets, renders)]
    head, gap, note, leg = 0.17, 0.10, 0.13, 0.20
    fig_h = head + sum(rows_h) + len(q) * note + (len(q) - 1) * gap + leg
    fig = plt.figure(figsize=(width, fig_h))
    titles = ["six faces as decoded (front, right, back, left; top, bottom)",
              "what the rule decided before each cell was drawn", "lift"]
    top = fig_h - head
    for r, i in enumerate(q):
        rh = rows_h[r]
        a, b, boxes = sheets[r]
        for j, (img, x0) in enumerate(((a, 0.0), (b, ws + 0.08))):
            h, w = img.shape[0] * ips, img.shape[1] * ips
            ax = fig.add_axes([x0 / width, (top - rh) / fig_h, w / width, h / fig_h])
            ax.imshow(img, interpolation="nearest")
            bare(ax)
            if r == 0:
                fig.text((x0 + ws / 2) / width, (fig_h - 0.02) / fig_h, titles[j], ha="center", va="top",
                         fontsize=7, color=INK)
        im = renders[r]
        h, w = im.shape[0] * ipr, im.shape[1] * ipr
        x0 = 2 * ws + 0.16 + (wr - w) / 2
        ax = fig.add_axes([x0 / width, (top - rh) / fig_h, w / width, h / fig_h])
        ax.imshow(im)
        bare(ax)
        if r == 0:
            fig.text((2 * ws + 0.16 + wr / 2) / width, (fig_h - 0.02) / fig_h, titles[2], ha="center",
                     va="top", fontsize=7, color=INK)
        shape = c[i][0]
        fs = [100 * masks[i][f][0].sum() / masks[i][f][0].size for f in FACES]
        txt = (f"box {shape[0]}x{shape[1]}x{shape[2]}: {100 * share[i]:.0f}% of cells forced "
               f"(front {fs[0]:.0f}%, right {fs[1]:.0f}%, back {fs[2]:.0f}%, left {fs[3]:.0f}%, "
               f"top {fs[4]:.0f}%, bottom {fs[5]:.0f}%)")
        fig.text(0.0, (top - rh - 0.025) / fig_h, txt, ha="left", va="top", fontsize=FS, color=INK_2)
        top -= rh + note + gap
    # legend
    y = 0.06 / fig_h
    xs = 0.0
    for colr, lab in ((ORANGE, "forced: one legal token, written without the model"),
                      (tuple((0.30 * _hex(ORANGE) + 0.70 * 255) / 255), "narrowed: some depths or see-through ruled out"),
                      ("#ececec", "free: the model's choice")):
        fig.patches.append(Rectangle((xs, y), 0.10 / width, 0.08 / fig_h, transform=fig.transFigure,
                                     facecolor=colr, edgecolor=RULE, lw=0.4))
        fig.text(xs + 0.14 / width, y + 0.04 / fig_h, lab, va="center", fontsize=FS, color=INK)
        xs += (0.25 + 0.052 * len(lab)) / width
    save(fig, "qual_forced")


# ---------------------------------------------------------------- 3. exteriors vs exteriors
def fig_exteriors(k: int = 6) -> None:
    real = load_structs(REAL_SHELL)
    vox = [s for s in load_structs(VOXEL_SHELL)]
    bl = lift_all(load_face_sets_cached("cross_bon"))
    ok_v = [i for i, s in enumerate(vox) if s.occupied_mask.any()]
    pr, pv = draw(10, range(len(real)), k), draw(11, ok_v, k)
    pb = draw(12, [i for i, s in enumerate(bl) if s is not None], k)
    print("exteriors real", pr, "voxel", pv, "best-of-4", pb)
    rows = [[iso(real[i]) for i in pr], [iso(vox[i]) for i in pv], [iso(bl[i]) for i in pb]]
    notes = [[parts(real[i]) for i in pr], [parts(vox[i]) for i in pv], [parts(bl[i]) for i in pb]]
    labels = ["Real-Shell", "Voxel-AR\nshell", "Cross-Face\nbest-of-4"]
    fig = tile_grid(rows, labels, notes, label_w=0.66)
    save(fig, "qual_exteriors")


# ---------------------------------------------------------------- 4. failures
def fig_failures(k: int = 8) -> None:
    cl = lift_all(load_face_sets_cached("cross"))
    ok = [i for i, s in enumerate(cl) if s is not None]
    nb = np.array([int(cl[i].occupied_mask.sum()) for i in ok])
    q1 = np.quantile(nb, 0.25)
    thin = [i for i, n in zip(ok, nb) if n <= q1]
    pa = draw(20, thin, k)
    rl = lift_all(load_face_sets_cached("rft_bon"))
    pb = draw(21, [i for i, s in enumerate(rl) if s is not None], k)
    print(f"failures thin (<= {q1:.0f} blocks, {len(thin)} builds)", pa, "rft", pb)
    rows = [[iso(cl[i]) for i in pa], [iso(rl[i]) for i in pb]]
    notes = [[f"{int(cl[i].occupied_mask.sum())} blocks" for i in pa],
             [f"{int(rl[i].occupied_mask.sum())} blocks" for i in pb]]
    labels = [f"(a) Cross-Face, T=0.7: random builds from the thinnest quarter (at most {q1:.0f} blocks)",
              "(b) RFT model + Cross-Face best-of-4 on face solidity (the reward-saturated arm): random builds"]
    fig = tile_grid(rows, labels, notes, labels_above=True)
    save(fig, "qual_failures")


# ---------------------------------------------------------------- 5. explode
#: concrete colour (block 251, data) per face for the source-face colouring
FACE_CONCRETE = {"-z": 1, "+x": 11, "+z": 4, "-x": 13, "+y": 14, "-y": 10}
# Validated categorical set (dataviz validator, light mode: all checks pass) that avoids the
# figures' Grammar blue and Cross-Face orange, so face colour never reads as a decoder.
FACE_RGB = {"-z": "#1a9e8f", "+x": "#8e3b8a", "+z": "#a8860a", "-x": "#5a4fc4", "+y": "#c2577a",
            "-y": "#5b8a1e"}


def face_voxels(F, shape):
    """Boolean voxel mask of one face's first hits."""
    from blockgen.corpus import facade as fc

    idx = fc._voxel_index(F.depth, F.axis, F.sign, shape)
    m = F.mask & (F.depth < shape[F.axis])
    out = np.zeros(shape, bool)
    out[tuple(i[m] for i in idx)] = True
    return out


def fig_explode() -> None:
    from blockgen.utils.data import Structure

    c = load_face_sets_cached("cross")
    cl = lift_all(c)
    ok = [i for i, e in enumerate(c) if e is not None and cl[i] is not None
          and e[0][0] >= 14 and e[0][2] >= 14]
    nb = {i: int(cl[i].occupied_mask.sum()) for i in ok}
    med = np.median(list(nb.values()))
    i = min(ok, key=lambda j: (abs(nb[j] - med), j))
    shape, faces = c[i]
    print("explode pick", i, shape, nb[i], "median", med)
    lift = cl[i]
    vox = {f: face_voxels(faces[f], shape) for f in FACES}
    # (b) first writer in decode order
    first = np.full(shape, -1, np.int64)
    for k, f in enumerate(FACES):
        first[(first < 0) & vox[f]] = k
    seen_by = sum(v.astype(int) for v in vox.values())
    shared = (seen_by >= 2).sum() / max((seen_by >= 1).sum(), 1)
    ids = np.where(first >= 0, 251, 0).astype(np.int32)
    data = np.zeros(shape, np.int32)
    for k, f in enumerate(FACES):
        data[first == k] = FACE_CONCRETE[f]
    by_face = Structure(block_ids=ids, block_data=data)
    # (c) cut open: keep the back half along the camera so the hollow inside shows
    X, Y, Z = shape
    keep = np.zeros(shape, bool)
    keep[:, :, : Z // 2] = True             # drop the +z half (nearest the camera at azimuth 45)
    keep[: X // 2, :, :] = True              # ... and the +x half: a corner cut, both camera-facing faces open
    cut = Structure(block_ids=np.where(keep, ids, 0).astype(np.int32), block_data=np.where(keep, data, 0).astype(np.int32))
    # (d) exploded: each face's voxels moved G blocks out along its normal
    G = 9
    big = (X + 2 * G, Y + 2 * G, Z + 2 * G)
    eids = np.zeros(big, np.int32)
    edat = np.zeros(big, np.int32)
    for f in reversed(FACES):
        off = [G, G, G]
        ax = "xyz".index(f[1])
        off[ax] += -G if f[0] == "-" else G
        sl = tuple(slice(o, o + n) for o, n in zip(off, shape))
        m = vox[f]
        sub_i, sub_d = eids[sl], edat[sl]
        sub_i[m] = 251
        sub_d[m] = FACE_CONCRETE[f]
    exploded = Structure(block_ids=eids, block_data=edat)
    ppb = 7.0
    imgs = [iso(lift, ppb), iso(by_face, ppb), iso(cut, ppb), iso(exploded, ppb)]
    titles = ["(a) Cross-Face lift", "(b) coloured by the face\nthat wrote each voxel",
              "(c) cut open: the lift\nis a shell, no interior", "(d) exploded: each face's\nfirst hits"]
    widths = np.array([im.shape[1] for im in imgs], float)
    H = max(im.shape[0] for im in imgs)
    width = CVPR_FULL
    gap = 0.08
    scale = (width - gap * 3) / widths.sum()      # inches per pixel, same for all panels
    h_img = H * scale
    head, leg = 0.30, 0.22
    fig_h = h_img + head + leg
    fig = plt.figure(figsize=(width, fig_h))
    x = 0.0
    for im, t in zip(imgs, titles):
        w = im.shape[1] * scale
        h = im.shape[0] * scale
        ax = fig.add_axes([x / width, leg / fig_h, w / width, h / fig_h])
        ax.imshow(im)
        bare(ax)
        ax.set_title(t, fontsize=7, color=INK, pad=2)
        x += w + gap
    xs = 0.0
    for f in FACES:
        fig.patches.append(Rectangle((xs, 0.05 / fig_h), 0.10 / width, 0.09 / fig_h, transform=fig.transFigure,
                                     facecolor=FACE_RGB[f], edgecolor="none"))
        fig.text(xs + 0.14 / width, 0.095 / fig_h, f"{FACE_NAME[f]} ({f})", va="center", fontsize=FS)
        xs += 0.78 / width
    fig.text(1.0, 0.095 / fig_h, f"{100 * shared:.0f}% of voxels are seen by two or more faces",
             ha="right", va="center", fontsize=FS, color=INK_2)
    save(fig, "qual_explode")


def main() -> None:
    figstyle.apply()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("gallery", fig_gallery), ("forced", fig_forced), ("exteriors", fig_exteriors),
                     ("failures", fig_failures), ("explode", fig_explode)):
        if which in (name, "all"):
            fn()


if __name__ == "__main__":
    main()
