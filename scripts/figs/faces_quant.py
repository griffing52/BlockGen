"""Quantitative / diagnostic figure candidates for the faces paper (cvpr27_faces).

    PYTHONPATH=. .venv/bin/python scripts/figs/faces_quant.py [order|dist|temp|cardA|cardC|all]

order   quant_face_order: per face position in decode order (front, right, back, left,
        top, bottom), the share of a face's cells that contradict what the earlier faces
        committed (Grammar, T=0.7 and 1.0), and the share of cells the earlier faces fully
        determine (Cross-Face, both T; Real-Shell faces of the real test houses as the
        reference). Replays `ShellState` over the saved faces.npz in decode order.
dist    quant_dist_ecdf / quant_dist_box: per-build parts (6-connected components of the
        lifted build, `topology.coherence`) and block counts for Grammar, Cross-Face,
        Cross-Face Best-of-4 (T=0.7), Real-Shell and real houses.
temp    quant_temperature: T=1.0 -> 0.7 for Grammar and Cross-Face, parts and V-KID,
        read from the T117 bench scorecard (bootstrap 95% CIs).
cardA   quant_cardA: the overnight paired card at n=256 (face arms, voxel arms and their
        six-face shells, Real-Shell), V-KID with CIs and Holm groups.
cardC   quant_cardC: the overnight seeds card, seed spread per decoder.

Run data is read from BLOCKGEN_DATA_ROOT (default: the main checkout, which owns
outputs/ and data/), so the script also works from a git worktree. Writes PDF + PNG to
deliverables/paper/cvpr27_faces/figures/candidates/ of this checkout. The replays are
cached in $TMPDIR (BLOCKGEN_QUANT_CACHE to override); delete the cache to recompute.
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
_MAIN = ROOT.parents[2] if ROOT.parent.name == "worktrees" else ROOT
DATA = Path(os.environ.get("BLOCKGEN_DATA_ROOT", _MAIN))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle  # noqa: E402
from figstyle import BLUE, ORANGE, ORANGE_DEEP, INK, INK_2, RULE, SHADE  # noqa: E402

OUT = ROOT / "deliverables/paper/cvpr27_faces/figures/candidates"
RUN = DATA / "outputs/run_20261028_125240_houses32_facades_joint_d4_20m"
CARD_T117 = DATA / "outputs/run_20261109_050232_bench_facade_cross-face_constrained_decoding"
LOG = DATA / "outputs/logs/faces_overnight.log"
CACHE = Path(os.environ.get("BLOCKGEN_QUANT_CACHE",
                            Path(os.environ.get("TMPDIR", "/tmp")) / "faces_quant_cache.pkl"))

FACES = ("-z", "+x", "+z", "-x", "+y", "-y")
FACE_LABEL = ("front", "right", "back", "left", "top", "bottom")
COL1, COL2 = figstyle.CVPR_COL, figstyle.CVPR_FULL
FS = 7.0                       # base size; nothing printed below 6 pt
REAL_HOUSES = "#8a8a86"        # real houses: a neutral reference, dashed, never a data hue

# Decoder identity, one colour each (validated: BLUE / ORANGE / ORANGE_DEEP pass all six checks).
ARMS = {
    "grammar": dict(label="Grammar", color=BLUE),
    "cross": dict(label="Cross-Face", color=ORANGE),
    "cross_bo4": dict(label="Cross-Face + Best-of-4", color=ORANGE_DEEP),
    "shell": dict(label="Real-Shell", color=INK),
    "real": dict(label="Real houses", color=REAL_HOUSES),
}


def _style():
    figstyle.apply(FS, **{"axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS - 0.5,
                          "ytick.labelsize": FS - 0.5, "legend.fontsize": FS - 0.5,
                          "axes.titlelocation": "left", "axes.titleweight": "bold"})


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def _grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=RULE, lw=0.4, zorder=0)
    ax.set_axisbelow(True)


def _boot_ci(x, n_boot=2000, seed=0, stat=np.mean):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = [stat(x[rng.integers(len(x), size=len(x))]) for _ in range(n_boot)]
    return float(stat(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# --------------------------------------------------------------------------- data


def load_faces(npz_path: Path):
    """faces.npz -> list of ({face: Facade}, shape) in decode order."""
    from blockgen.corpus import facade as fc
    z = np.load(npz_path)
    idx = sorted({int(k.split("_", 1)[0]) for k in z.files})
    out = []
    for i in idx:
        if f"{i}_-z_depth" not in z.files:
            continue
        fd = {}
        for f in FACES:
            fd[f] = (z[f"{i}_{f}_depth"], z[f"{i}_{f}_ids"], z[f"{i}_{f}_data"])
        X, Y = fd["-z"][0].shape
        Z = fd["+x"][0].shape[0]
        shape = (X, Y, Z)
        ext = {"x": X, "y": Y, "z": Z}
        faces = {f: fc.Facade(face=f, depth=d, ids=a, data=b, extent=ext[f[1]]) for f, (d, a, b) in fd.items()}
        out.append((faces, shape))
    return out


def replay(faces, shape):
    """Walk the faces in decode order through a ShellState.

    Per face: (cells, conflict, determined). `conflict` counts cells whose (depth, block)
    is illegal given the earlier faces' commitments (those cells are not committed, so the
    earlier face wins, as a first-come lift would); `determined` counts cells that had one
    legal answer before the face was drawn (what the Cross-Face sampler forces).
    """
    from blockgen.corpus.facade_constraints import FREE, ShellState
    st = ShellState(shape)
    rows = []
    for name in FACES:
        f = faces[name]
        U, V = f.depth.shape
        n_conf = n_det = 0
        commits = []
        for v in range(V):
            for u in range(U):
                allowed, forced, allow_empty = st.allowed(name, u, v)
                d = int(f.depth[u, v])
                key = int(f.ids[u, v]) * 16 + int(f.data[u, v])
                if (not allow_empty) and allowed.sum() == 1 and forced[int(np.argmax(allowed))] != FREE:
                    n_det += 1
                if d < 0:
                    bad = not allow_empty
                else:
                    bad = d >= len(allowed) or not allowed[d] or (forced[d] != FREE and forced[d] != key)
                if bad:
                    n_conf += 1
                else:
                    commits.append((u, v, d, key))
        for u, v, d, key in commits:        # rays of one face never share a voxel
            st.commit_cell(name, u, v, d, key)
        rows.append((U * V, n_conf, n_det))
    return np.array(rows)                    # (6, 3)


def real_test_houses():
    from blockgen.eval.bench import splits
    cache = DATA / "data/minecraft/cache"
    split = splits.load_split("houses_32", 0, cache_dir=cache, root=DATA / "data/minecraft/splits")
    return [s.crop_to_non_air() for s in splits.split_structures(split, "test", cache_dir=cache)]


def _per_build(structs):
    from blockgen.eval.bench import topology
    cs = [topology.coherence(s) for s in structs if s.occupied_mask.any()]
    return dict(parts=np.array([c.n_components for c in cs]), blocks=np.array([c.n_blocks for c in cs]))


def compute():
    if CACHE.exists():
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)
    from blockgen.corpus import facade as fc
    from blockgen.curation.houses import load_structures_from_cache
    cwd = os.getcwd()
    os.chdir(DATA)                                # split loaders resolve data/ relative to the checkout
    try:
        res = {"order": {}, "dist": {}}
        for key, d in (("grammar_t0.7", "constrained_none_t0.7_n399"), ("grammar_t1.0", "constrained_none_t1.0_n399"),
                       ("cross_t0.7", "constrained_constrain_t0.7_n399"), ("cross_t1.0", "constrained_constrain_t1.0_n399")):
            builds = load_faces(RUN / d / "faces.npz")
            res["order"][key] = np.stack([replay(f, s) for f, s in builds])
            print(key, "replayed", len(builds), flush=True)
        houses = real_test_houses()
        real_faces = [({f: fc.project(h, f) for f in FACES}, h.shape) for h in houses]
        res["order"]["real"] = np.stack([replay(f, s) for f, s in real_faces])
        for key, p in (("grammar", RUN / "constrained_none_t0.7_n399/samples_step399_48.npz"),
                       ("cross", RUN / "constrained_constrain_t0.7_n399/samples_step399_48.npz"),
                       ("cross_bo4", RUN / "constrained_bon_face_solidity_k4_t0.7_n399/bon_48.npz"),
                       ("shell", DATA / "outputs/bench_arms/real_facades6_32.npz")):
            s, _ = load_structures_from_cache(str(p))
            res["dist"][key] = _per_build(s)
        res["dist"]["real"] = _per_build(houses)
    finally:
        os.chdir(cwd)
    with open(CACHE, "wb") as fh:
        pickle.dump(res, fh)
    return res


# --------------------------------------------------------------------------- 1. face order


def fig_order(res):
    _style()
    o = res["order"]
    x = np.arange(6)
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL1, 2.75), sharex=True, gridspec_kw=dict(hspace=0.32))

    def share(arr, col):                      # per build share, then mean + bootstrap CI over builds
        r = arr[:, :, col] / np.maximum(arr[:, :, 0], 1)
        stats = np.array([_boot_ci(r[:, j] * 100) for j in range(6)])
        return stats

    series_a = [("grammar_t0.7", "Grammar, T=0.7", BLUE, "-", "o"),
                ("grammar_t1.0", "Grammar, T=1.0", BLUE, "--", "o"),
                ("cross_t0.7", "Cross-Face (both T)", ORANGE, "-", "o")]
    for key, lab, c, ls, mk in series_a:
        s = share(o[key], 1)
        hollow = ls == "--"
        a.fill_between(x, s[:, 1], s[:, 2], color=c, alpha=0.15, lw=0, zorder=1)
        a.plot(x, s[:, 0], ls=ls, color=c, lw=1.0, marker=mk, ms=3.2, zorder=3,
               mfc="white" if hollow else c, mec=c, mew=0.8, label=lab)
    a.set_ylabel("cells contradicting\nearlier faces (%)")
    a.set_title("a  Grammar: later faces disagree more", fontsize=FS)
    a.set_ylim(-2, 98)
    _grid(a)
    a.legend(loc="upper left", frameon=False, handlelength=2.2, borderaxespad=0.1)

    series_b = [("cross_t0.7", "Cross-Face, T=0.7", ORANGE, "-", "o"),
                ("cross_t1.0", "Cross-Face, T=1.0", ORANGE, "--", "o"),
                ("real", "Real-Shell (real faces)", INK, ":", "s")]
    for key, lab, c, ls, mk in series_b:
        s = share(o[key], 2)
        hollow = ls == "--"
        b.fill_between(x, s[:, 1], s[:, 2], color=c, alpha=0.15, lw=0, zorder=1)
        b.plot(x, s[:, 0], ls=ls, color=c, lw=1.0, marker=mk, ms=3.2 if mk == "o" else 2.8, zorder=3,
               mfc="white" if hollow else c, mec=c, mew=0.8, label=lab)
    b.set_ylabel("cells fixed by\nearlier faces (%)")
    b.set_title("b  Cross-Face: share of cells forced", fontsize=FS)
    b.set_ylim(-2, 88)
    _grid(b)
    b.legend(loc="upper left", frameon=False, handlelength=2.2, borderaxespad=0.1)
    b.set_xticks(x, [f"{i + 1}\n{l}" for i, l in enumerate(FACE_LABEL)])
    b.set_xlabel("face position in decode order")
    b.set_xlim(-0.3, 5.3)
    _save(fig, "quant_face_order")

    # printed so the README / text can quote the numbers
    for key in o:
        c = (o[key][:, :, 1] / np.maximum(o[key][:, :, 0], 1)).mean(0) * 100
        d = (o[key][:, :, 2] / np.maximum(o[key][:, :, 0], 1)).mean(0) * 100
        tot_d = o[key][:, :, 2].sum() / o[key][:, :, 0].sum()
        print(f"  {key:13s} conflict% {np.round(c, 1)}  determined% {np.round(d, 1)}  pooled determined {tot_d:.3f}")


# --------------------------------------------------------------------------- 2. distributions

DIST_ORDER = ("grammar", "cross", "cross_bo4", "shell", "real")


def _ecdf(ax, v, **kw):
    v = np.sort(v)
    y = np.arange(1, len(v) + 1) / len(v)
    ax.step(np.r_[v[0], v], np.r_[0, y], where="post", **kw)


def fig_dist(res):
    _style()
    d = res["dist"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2 * 0.6, 1.85), gridspec_kw=dict(wspace=0.12), sharey=True)
    for key in DIST_ORDER:
        st = ARMS[key]
        ref = key in ("shell", "real")
        kw = dict(color=st["color"], lw=1.1 if not ref else 0.9, ls="--" if key == "real" else "-",
                  label=st["label"], zorder=3 if not ref else 2)
        _ecdf(a, d[key]["parts"], **kw)
        _ecdf(b, d[key]["blocks"], **kw)
    a.set_xscale("log")
    a.set_xlim(0.9, 300)
    a.set_xticks([1, 3, 10, 30, 100, 300], ["1", "3", "10", "30", "100", "300"])
    a.set_xlabel("parts per build (6-connected components)")
    a.set_ylabel("share of builds")
    a.set_title("a  Parts", fontsize=FS)
    b.set_xlabel("blocks per build")
    b.set_xlim(0, 2600)
    b.set_title("b  Blocks", fontsize=FS)
    for ax in (a, b):
        _grid(ax, "both")
        ax.set_ylim(0, 1.0)
    b.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, handlelength=1.8, borderaxespad=0.1)
    _save(fig, "quant_dist_ecdf")

    # box + strip variant: one column, parts only on log scale, blocks beside
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL1, 1.9), sharey=True, gridspec_kw=dict(wspace=0.08))
    rng = np.random.default_rng(0)
    ys = np.arange(len(DIST_ORDER))[::-1]
    for y, key in zip(ys, DIST_ORDER):
        st = ARMS[key]
        for ax, m in ((a, "parts"), (b, "blocks")):
            v = d[key][m].astype(float)
            jit = rng.uniform(-0.22, 0.22, len(v))
            ax.scatter(v, y + jit, s=1.2, color=st["color"], alpha=0.35, lw=0, zorder=2, rasterized=True)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([q1, q3], [y, y], color=INK, lw=2.4, solid_capstyle="butt", zorder=3, alpha=0.85)
            ax.plot([med], [y], marker="|", ms=7, mew=1.4, color="white", zorder=4)
    a.set_xscale("log")
    a.set_xlim(0.8, 400)
    a.set_xticks([1, 10, 100], ["1", "10", "100"])
    a.set_yticks(ys, [ARMS[k]["label"].replace(" + ", "\n+ ") for k in DIST_ORDER])
    a.set_xlabel("parts per build")
    b.set_xlabel("blocks per build")
    b.set_xlim(0, 2600)
    b.set_xticks([0, 1000, 2000])
    for ax in (a, b):
        _grid(ax, "x")
        ax.tick_params(axis="y", length=0)
    for ax in (a, b):
        ax.axhline(1.5, color=RULE, lw=0.5)       # generated above, references below
    _save(fig, "quant_dist_box")

    for key in DIST_ORDER:
        p, bl = d[key]["parts"], d[key]["blocks"]
        print(f"  {key:10s} n={len(p)} parts mean {p.mean():.1f} median {np.median(p):.0f} share1 {np.mean(p == 1):.2f}"
              f"  blocks median {np.median(bl):.0f}")


# --------------------------------------------------------------------------- scorecards


def load_card(run_dir: Path):
    with open(run_dir / "scorecard.json") as fh:
        j = json.load(fh)
    md = (run_dir / "scorecard.md").read_text()
    return j, md


def holm_groups(md: str, metric: str) -> dict:
    """arm -> group letters, from the scorecard.md head-to-head table of `metric`."""
    m = re.search(rf"### {re.escape(metric)} \(.*?\)\n\n\| # \| arm \| score \| group \|\n\|[-|]+\|\n((?:\|.*\|\n)+)", md)
    if not m:
        return {}
    out = {}
    for line in m.group(1).strip().splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 4:
            out[cells[1]] = cells[3]
    return out


def metric(j, arm, section, name):
    """(value, lo, hi) of a scorecard metric; coherence metrics give the generated mean."""
    v = j["arms"][arm][section][name]
    if section in ("coherence", "geometry_scalars", "traversal"):
        g = v["gen"]
        return g["mean"], g["ci"][0], g["ci"][1]
    return v["value"], v["ci"][0], v["ci"][1]


def real_coh(j, arm, name):
    return j["arms"][arm]["coherence"][name]["real"]["mean"]


# --------------------------------------------------------------------------- 3. temperature


def fig_temp():
    _style()
    j, md = load_card(CARD_T117)
    pairs = [("Grammar", BLUE, "joint_d4_20m_grammar", "joint_d4_20m_grammar_t0.7"),
             ("Cross-Face", ORANGE, "joint_d4_20m_constrained", "joint_d4_20m_constrained_t0.7")]
    panels = [("coherence", "n_components", "parts per build (mean)"),
              ("realism", "geom_kid", r"V-KID ($\times10^{3}$)")]
    shell = "real@facades6"
    fig, axes = plt.subplots(1, 2, figsize=(COL1, 1.5), sharey=True, gridspec_kw=dict(wspace=0.12))
    ys = [1, 0]
    for ax, (sec, name, xl) in zip(axes, panels):
        for y, (lab, c, a10, a07) in zip(ys, pairs):
            v1, l1, h1 = metric(j, a10, sec, name)
            v7, l7, h7 = metric(j, a07, sec, name)
            ax.annotate("", xy=(v7, y), xytext=(v1, y),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=0.9, shrinkA=3.2, shrinkB=3.2,
                                        mutation_scale=6), zorder=2)
            for v, lo, hi, filled in ((v1, l1, h1, False), (v7, l7, h7, True)):
                ax.plot([lo, hi], [y, y], color=c, lw=2.6, alpha=0.3, solid_capstyle="butt", zorder=2)
                ax.plot([v], [y], "o", ms=4.2, mfc=c if filled else "white", mec=c, mew=0.9, zorder=3)
            if y == ys[0]:                     # direct T labels on the top row only
                ax.text(v1, y + 0.26, "T=1.0", ha="center", va="bottom", fontsize=FS - 1, color=INK_2)
                ax.text(v7, y + 0.26, "T=0.7", ha="center", va="bottom", fontsize=FS - 1, color=INK_2)
        sv = metric(j, shell, sec, name)[0]
        ax.axvline(sv, color=INK, lw=0.7, ls=":", zorder=1)
        ax.text(sv + 0.6, 1.98 if name == "n_components" else -0.62, "Real-Shell", ha="left", va="top",
                fontsize=FS - 1, color=INK)
        if name == "n_components":
            rv = real_coh(j, shell, name)
            ax.axvline(rv, color=REAL_HOUSES, lw=0.7, ls="--", zorder=1)
            ax.text(rv + 0.6, -0.62, "Real", ha="left", va="top", fontsize=FS - 1, color=INK_2)
        ax.set_xlabel(xl)
        _grid(ax, "x")
        ax.set_ylim(-0.95, 2.0)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ys, [p[0] for p in pairs])
    axes[0].set_xlim(0, 70)
    axes[1].set_xlim(10, 33)
    fig.text(0.0, 1.0, "T = 1.0 (hollow)  →  T = 0.7 (filled)", fontsize=FS - 1, color=INK_2,
             ha="left", va="bottom", transform=axes[0].transAxes)
    _save(fig, "quant_temperature")
    for lab, c, a10, a07 in pairs:
        print(" ", lab, [round(metric(j, a, s, n)[0], 2) for a in (a10, a07) for s, n, _ in panels])


# --------------------------------------------------------------------------- 4. overnight cards


def _card_dir(prefix: str):
    hits = sorted(glob.glob(str(DATA / f"outputs/run_*_bench_faces_paper_{prefix}*")))
    hits = [h for h in hits if os.path.exists(os.path.join(h, "scorecard.json"))]
    return Path(hits[-1]) if hits else None


def card_done(tag: str) -> bool:
    return LOG.exists() and f"FACES CARD {tag} DONE" in LOG.read_text()


def _cardA_family(arm: str):
    """(display label, colour, filled, family) for an arm of card A."""
    if arm == "real@facades6":
        return "Real-Shell", INK, True, "ref"
    if arm == "real_test":
        return "Real houses", REAL_HOUSES, True, "ref"
    vox = {"canon_oriented": "Voxel-AR 5M", "recanon20m_s0": "Voxel-AR 27M seed 0", "recanon20m_s1": "Voxel-AR 27M seed 1"}
    for k, lab in vox.items():
        if arm == f"{k}@shell6":
            return f"{lab}, shell", INK_2, True, "voxshell"
        if arm == k:
            return f"{lab}, whole", INK_2, False, "voxwhole"
    if "constrained_bon" in arm:
        return "Cross-Face + Best-of-4", ORANGE_DEEP, True, "face"
    if "constrained" in arm:
        return "Cross-Face", ORANGE, True, "face"
    if "bon" in arm:
        return "Best-of-4 (no rule)", BLUE, False, "face"
    if "grammar" in arm:
        return "Grammar", BLUE, True, "face"
    return None


def plot_card_A(run_dir: Path | None = None, name: str = "quant_cardA"):
    """Exterior-only comparison at n=256: face arms vs voxel arms whole and as six-face
    shells vs Real-Shell. V-KID with 95% CIs, Holm groups at the right; parts beside."""
    run_dir = run_dir or _card_dir("paired")
    if run_dir is None:
        print("card A not found")
        return False
    _style()
    j, md = load_card(run_dir)
    groups = holm_groups(md, "geom_kid")
    rows = []
    for arm in j["arms"]:
        fam = _cardA_family(arm)
        if fam is None:
            continue
        try:
            g = metric(j, arm, "realism", "geom_kid")
            p = metric(j, arm, "coherence", "n_components")
        except (KeyError, TypeError):
            continue
        rows.append((arm, fam, g, p))
    order = {"face": 0, "voxshell": 1, "voxwhole": 2, "ref": 3}
    rows.sort(key=lambda r: (order[r[1][3]], r[2][0]))
    n = len(rows)
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2 * 0.72, 0.2 * n + 0.65), sharey=True,
                               gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.14))
    ys = np.arange(n)[::-1]
    for y, (arm, (lab, c, filled, fam), g, p) in zip(ys, rows):
        for ax, (v, lo, hi) in ((a, g), (b, p)):
            ax.plot([lo, hi], [y, y], color=c, lw=1.0, solid_capstyle="butt", zorder=2)
            ax.plot([v], [y], "o", ms=4.0, mfc=c if filled else "white", mec=c, mew=0.9, zorder=3)
        if arm in groups:
            a.annotate(groups[arm], xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(3, 0),
                       textcoords="offset points", ha="left", va="center", fontsize=FS - 1, color=INK_2)
    # separators between families
    fams = [r[1][3] for r in rows]
    for i in range(1, n):
        if fams[i] != fams[i - 1]:
            for ax in (a, b):
                ax.axhline(ys[i] + 0.5, color=RULE, lw=0.5)
    shell = [r for r in rows if r[0] == "real@facades6"]
    if shell:
        for ax, v in ((a, shell[0][2][0]), (b, shell[0][3][0])):
            ax.axvline(v, color=INK, lw=0.6, ls=":", zorder=1)
    a.set_yticks(ys, [r[1][0] for r in rows])
    a.set_xlabel(r"V-KID ($\times10^{3}$, lower is closer to real)")
    b.set_xlabel("parts per build (mean)")
    b.set_xscale("log")
    b.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    b.minorticks_off()
    b.set_xlim(min(r[3][1] for r in rows) * 0.7, max(r[3][2] for r in rows) * 1.4)
    a.set_xlim(left=min(-1, min(r[2][1] for r in rows) - 0.5))
    for ax in (a, b):
        _grid(ax, "x")
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, n - 0.4)
    a.annotate("Holm", xy=(1.0, n - 0.4), xycoords=("axes fraction", "data"), xytext=(3, 1),
               textcoords="offset points", ha="left", va="bottom", fontsize=FS - 1.5, color=INK_2)
    ns = sorted({j["arms"][r[0]]["meta"]["n"] for r in rows})
    ntxt = f"n = {ns[0]}" if len(ns) == 1 else f"n = {ns[0]} to {ns[-1]}"
    a.set_title(f"{ntxt} per arm, 95% CI; hollow = whole voxel build, filled grey = its six-face shell; "
                "dotted = Real-Shell", fontsize=FS - 1, fontweight="normal", color=INK_2, loc="left", pad=9)
    _save(fig, name)
    return True


def _cardC_decoder(arm: str):
    m = re.match(r"(grammar|constrained|constrained_bon_k4)_t0\.7_s(\d+)$", arm)
    if not m:
        return None
    return {"grammar": "Grammar", "constrained": "Cross-Face", "constrained_bon_k4": "Cross-Face + Best-of-4"}[m.group(1)], int(m.group(2))


def plot_card_C(run_dir: Path | None = None, name: str = "quant_cardC"):
    """Seed spread per decoder: three seeds per decoder as dots with CIs, the seed mean as a
    tick, on V-KID, MV-KID and parts; Patchwork and Real-Shell as reference lines."""
    run_dir = run_dir or _card_dir("seeds")
    if run_dir is None:
        print("card C not found")
        return False
    _style()
    j, md = load_card(run_dir)
    dec_col = {"Grammar": BLUE, "Cross-Face": ORANGE, "Cross-Face + Best-of-4": ORANGE_DEEP}
    decs = list(dec_col)
    panels = [("realism", "geom_kid", r"V-KID ($\times10^{3}$)"),
              ("coherence", "n_components", "parts per build"),
              ("realism", "mv_dino_kid", "MV-KID")]
    refs = [("real@facades6", "Real-Shell", ":"), ("facade_patchwork", "Patchwork", "--")]
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 1.45), sharey=True, gridspec_kw=dict(wspace=0.1))
    ys = {d: i for i, d in enumerate(decs[::-1])}
    for ax, (sec, mname, xl) in zip(axes, panels):
        for arm in j["arms"]:
            dc = _cardC_decoder(arm)
            if dc is None:
                continue
            dname, seed = dc
            try:
                v, lo, hi = metric(j, arm, sec, mname)
            except (KeyError, TypeError):
                continue
            y = ys[dname] + (seed - 1) * 0.22
            c = dec_col[dname]
            ax.plot([lo, hi], [y, y], color=c, lw=0.9, zorder=2)
            ax.plot([v], [y], "o", ms=3.4, color=c, zorder=3)
        for dname in decs:
            vals = [metric(j, a, sec, mname)[0] for a in j["arms"]
                    if (_cardC_decoder(a) or ("",))[0] == dname]
            if vals:
                ax.plot([np.mean(vals)] * 2, [ys[dname] - 0.4, ys[dname] + 0.4], color=INK, lw=0.8, zorder=1.5)
        for arm, lab, ls in refs:
            if arm in j["arms"]:
                try:
                    v = metric(j, arm, sec, mname)[0]
                except (KeyError, TypeError):
                    continue
                ax.axvline(v, color=INK if lab == "Real-Shell" else INK_2, lw=0.6, ls=ls, zorder=1)
                lo_, hi_ = ax.get_xlim()
                right = (v - lo_) / (hi_ - lo_) > 0.8
                ax.text(v, -0.9, f"{lab} " if right else f" {lab}", ha="right" if right else "left",
                        va="bottom", fontsize=FS - 1.5, color=INK_2)
        ax.set_xlabel(xl)
        _grid(ax, "x")
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.95, len(decs) - 0.45)
    axes[0].set_yticks([ys[d] for d in decs], decs)
    axes[0].text(0.0, 1.12, "three sampling seeds per decoder (seed 0 lowest in each row), 95% CI each; black tick = seed mean",
                 transform=axes[0].transAxes, ha="left", va="bottom", fontsize=FS - 1, color=INK_2)
    _save(fig, name)
    return True


# --------------------------------------------------------------------------- main


def main(argv=None):
    what = (argv or sys.argv[1:] or ["all"])[0]
    if what in ("order", "dist", "all"):
        res = compute()
        if what in ("order", "all"):
            fig_order(res)
        if what in ("dist", "all"):
            fig_dist(res)
    if what in ("temp", "all"):
        fig_temp()
    if what in ("cardA", "all"):
        if card_done("A") or _card_dir("paired"):
            plot_card_A()
        else:
            print("card A not done yet (no FACES CARD A DONE in the log)")
    if what in ("cardC", "all"):
        if card_done("C") or _card_dir("seeds"):
            plot_card_C()
        else:
            print("card C not done yet (no FACES CARD C DONE in the log)")
    if what == "preview":                 # dry run of the card plotters on the T117 card
        plot_card_A(CARD_T117, "preview_cardA")


if __name__ == "__main__":
    main()
