"""Is V-KID's result a property of the 77 features, or of observing the volume?

Three questions a reviewer asks about a hand-designed descriptor, answered with
the bench's own scoring code (`ladder.score_rungs` for the blocking gates, the
leaderboard scorer `ablate_view_facade_section.score_space` for separations):

1. **Feature-group robustness.** The 77 dims are grouped into seven families
   (pattern orbits, thickness, height, surface/planarity, symmetry, interior,
   size). Leave-one-family-out, keep-one-family, and 20 uniform random subsets
   at 50% and 25% of the dims, each whitened on the reference and scored under
   the same kernel.
2. **Hand-design-free descriptors.** Generic maps of the occupancy grid made
   exactly D4-invariant by orbit averaging, f(x) = mean_g phi(g x), the one
   generic step every row shares: a random-weight 3-layer 3D CNN (three weight
   draws), a feature-hashed histogram of raw 3^3 occupancy patterns, and
   multi-scale 3^3 neighbour-count histograms.
3. **Global coherence.** Coarse global terms appended to V-KID: largest
   26-connected part ratio and component count, and 2^3 orbit histograms of the
   max-pooled grid at 2x and 4x. Scored on Patch-Stitch and on every gate.

Protocols, both reproduced from `bench_headtohead.py`:
  ladder: houses_32 s0, ladder subset, val reference (256), n=128, 48 reps, geometry rungs;
  board:  n=256 arms of the leaderboard of record (build hashes checked), 399 val
          reference, 399 test calibration, noise floor = sd of 48 real draws.

    PYTHONPATH=. .venv/bin/python scripts/vkid_robustness.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr

from blockgen.eval.bench import distances as D
from blockgen.eval.bench import geometry as geom
from blockgen.eval.bench import ladder as L
from blockgen.eval.bench import probes as pb
from blockgen.eval.bench import splits
from blockgen.utils.runs import new_run_dir

# --- descriptor layout (v1) --------------------------------------------------------
SCAL0 = 55 + 5 + 8
SCALARS = list(geom.BASE_METRICS) + ["log1p_n_blocks"]
FAMILIES: Dict[str, List[int]] = {
    "pattern": list(range(0, 55)),
    "thickness": list(range(55, 60)) + [SCAL0 + 0, SCAL0 + 1],
    "height": list(range(60, 68)) + [SCAL0 + 7],
    "surface": [SCAL0 + 2, SCAL0 + 3],
    "symmetry": [SCAL0 + 4],
    "interior": [SCAL0 + 5, SCAL0 + 6],
    "size": [SCAL0 + 8],
}
assert sorted(sum(FAMILIES.values(), [])) == list(range(77))

STRUCT_ARMS = ("real@solidify", "real@hollow", "shell_box", "patchwork", "gabled_house",
               "canon_oriented", "canon_unoriented", "native_oriented", "dreamcubed")
LADDER_PROBES = ("solidify", "hollow", "material_shuffle", "monochrome", "chunk_delete_40",
                 "occ_noise_5", "canon16", "jitter_columns", "rot90_1", "mirror_x")


# --- per-structure features -----------------------------------------------------------
def _d4(occ: np.ndarray) -> List[np.ndarray]:
    out = []
    for m in (occ, occ[::-1]):
        for k in range(4):
            out.append(np.ascontiguousarray(np.rot90(m, k, axes=(0, 2))))
    return out


_CNN = None


def _cnn():
    global _CNN
    if _CNN is None:
        import torch
        torch.set_num_threads(1)
        nets = []
        for seed in (0, 1, 2):
            g = torch.Generator().manual_seed(1000 + seed)
            chans = [1, 16, 32, 64]
            layers = []
            for a, b in zip(chans, chans[1:]):
                w = torch.randn(b, a, 3, 3, 3, generator=g) * np.sqrt(2.0 / (a * 27))
                bias = torch.randn(b, generator=g) * 0.1
                layers.append((w, bias))
            nets.append(layers)
        _CNN = nets
    return _CNN


def rand_cnn(occ: np.ndarray) -> List[np.ndarray]:
    """Random-weight 3D CNN, global mean+max pool per layer, orbit-averaged (224-d each)."""
    import torch
    import torch.nn.functional as F
    vs = _d4(occ)
    # rot90 swaps the x and z extents, so the orbit splits into two same-shape batches
    batches = [torch.as_tensor(np.stack([np.pad(v, 2) for v in vs[j::2]])[:, None]
                               .astype(np.float32)) for j in (0, 1)]
    feats = []
    with torch.no_grad():
        for layers in _cnn():
            acc = 0.0
            for xs in batches:
                h = xs
                f = []
                for li, (w, b) in enumerate(layers):
                    h = F.relu(F.conv3d(h, w, b, padding=1))
                    f.append(h.mean((2, 3, 4)))
                    f.append(h.amax((2, 3, 4)))
                    if li < len(layers) - 1:
                        h = F.avg_pool3d(h, 2, ceil_mode=True)
                acc = acc + torch.cat(f, 1).sum(0)
            feats.append((acc / 8.0).numpy().astype(np.float64))
    return feats


_OFFS = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)])


def _codes(occ: np.ndarray) -> np.ndarray:
    """27-bit raw 3^3 pattern code at every cell of the 1-padded grid."""
    p = np.pad(occ, 2).astype(np.int64)
    X, Y, Z = occ.shape[0] + 2, occ.shape[1] + 2, occ.shape[2] + 2
    code = np.zeros((X, Y, Z), np.int64)
    for k, (a, b, c) in enumerate(_OFFS):
        code |= p[1 + a:1 + a + X, 1 + b:1 + b + Y, 1 + c:1 + c + Z] << k
    return code


def hash_hist(occ: np.ndarray, bins: int = 256) -> np.ndarray:
    """Feature-hashed histogram of raw 3^3 patterns (all-air windows dropped), orbit-averaged."""
    h = np.zeros(bins)
    for v in _d4(occ):
        c = _codes(v).ravel()
        c = c[c != 0]
        b = ((c * 2654435761) % (1 << 32)) % bins
        hh = np.bincount(b, minlength=bins).astype(float)
        h += hh / max(hh.sum(), 1.0)
    return h / 8.0


def nbr_hist(occ: np.ndarray) -> np.ndarray:
    """3^3 neighbour-count histograms for solid and air cells, at 1x/2x/4x max-pooling (168-d)."""
    out = []
    for s in (1, 2, 4):
        acc = np.zeros(56)
        for v in _d4(occ):
            if s > 1:
                pad = [(0, (-n) % s) for n in v.shape]
                vv = np.pad(v, pad)
                vv = vv.reshape(vv.shape[0] // s, s, vv.shape[1] // s, s,
                                vv.shape[2] // s, s).any((1, 3, 5))
            else:
                vv = v
            p = np.pad(vv, 1).astype(np.int32)
            cnt = ndimage.convolve(p, np.ones((3, 3, 3), np.int32), mode="constant")
            solid = p.astype(bool)
            hs = np.bincount(cnt[solid], minlength=28)[:28].astype(float)
            ha = np.bincount(cnt[~solid], minlength=28)[:28].astype(float)
            acc += np.concatenate([hs / max(hs.sum(), 1), ha / max(ha.sum(), 1)])
        out.append(acc / 8.0)
    return np.concatenate(out)


S26 = ndimage.generate_binary_structure(3, 3)


def global_terms(occ: np.ndarray) -> np.ndarray:
    """[lcc_ratio, log1p(n_components)] + 2^3 orbit histograms of 2x and 4x max-pooled occupancy."""
    lab, n = ndimage.label(occ, structure=S26)
    if n:
        sizes = np.bincount(lab.ravel())[1:]
        lcc = sizes.max() / sizes.sum()
    else:
        lcc = 0.0
    ms = []
    for s in (2, 4):
        acc = np.zeros(geom.N_ORBITS)
        for v in _d4(occ):
            pad = [(0, (-k) % s) for k in v.shape]
            vv = np.pad(v, pad)
            vv = vv.reshape(vv.shape[0] // s, s, vv.shape[1] // s, s,
                            vv.shape[2] // s, s).any((1, 3, 5))
            acc += geom.pattern_histogram(vv)
        ms.append(acc / 8.0)
    return np.concatenate([[lcc, np.log1p(n)], *ms])


def item(job):
    i, s = job
    geom.set_descriptor_version("v1")
    c = s.crop_to_non_air()
    occ = c.occupied_mask
    cnn = rand_cnn(occ)
    return i, {"vkid": geom.descriptor([c])[0], "cnn0": cnn[0], "cnn1": cnn[1], "cnn2": cnn[2],
               "hash3": hash_hist(occ), "nbr_ms": nbr_hist(occ), "glob": global_terms(occ)}


def featurize(structs, workers, log, tag) -> Dict[str, np.ndarray]:
    t0 = time.time()
    res: Dict[int, dict] = {}
    with ProcessPoolExecutor(workers) as ex:
        for k, (i, f) in enumerate(ex.map(item, list(enumerate(structs)), chunksize=8)):
            res[i] = f
            if (k + 1) % 1000 == 0:
                log(f"  [{tag}] {k + 1}/{len(structs)} {time.time() - t0:.0f}s")
    keys = res[0].keys()
    return {k: np.stack([res[i][k] for i in range(len(structs))]) for k in keys}


# --- rows -------------------------------------------------------------------------------
def build_rows(F: Dict[str, np.ndarray], seed: int) -> Dict[str, dict]:
    """`{row: {"X": raw features, "group": ...}}`; whitening happens per protocol."""
    V = F["vkid"]
    rows: Dict[str, dict] = {"full V-KID (77)": {"X": V, "group": "full"}}
    for fam, ix in FAMILIES.items():
        keep = [j for j in range(77) if j not in ix]
        rows[f"drop {fam} ({len(keep)})"] = {"X": V[:, keep], "group": "lofo"}
    for fam, ix in FAMILIES.items():
        rows[f"only {fam} ({len(ix)})"] = {"X": V[:, ix], "group": "only"}
    keep = [j for j in range(77) if j not in FAMILIES["interior"] + FAMILIES["size"]]
    rows[f"drop interior+size ({len(keep)})"] = {"X": V[:, keep], "group": "lofo"}
    rng = np.random.default_rng(seed + 77)
    for frac in (0.5, 0.25):
        k = int(round(frac * 77))
        for d in range(20):
            ix = np.sort(rng.choice(77, size=k, replace=False))
            rows[f"rand{int(frac * 100)}_{d:02d}"] = {"X": V[:, ix], "group": f"rand{int(frac * 100)}",
                                                    "dims": ix.tolist()}
    for c in ("cnn0", "cnn1", "cnn2"):
        rows[f"generic: random 3D CNN w{c[-1]} (224)"] = {"X": F[c], "group": "generic"}
    rows["generic: hashed 3^3 patterns (256)"] = {"X": F["hash3"], "group": "generic"}
    rows["generic: multi-scale neighbour counts (168)"] = {"X": F["nbr_ms"], "group": "generic"}
    rows["generic: CNN w0 + neighbour counts (392)"] = {
        "X": np.concatenate([F["cnn0"], F["nbr_ms"]], 1), "group": "generic"}
    G = F["glob"]
    rows["V-KID + LCC (79)"] = {"X": np.concatenate([V, G[:, :2]], 1), "group": "global"}
    rows["V-KID + 2x pooled orbits (132)"] = {"X": np.concatenate([V, G[:, 2:57]], 1),
                                             "group": "global"}
    rows["V-KID + 2x,4x pooled orbits (187)"] = {"X": np.concatenate([V, G[:, 2:]], 1),
                                                "group": "global"}
    rows["V-KID + LCC + 2x,4x pooled (189)"] = {"X": np.concatenate([V, G], 1), "group": "global"}
    rows["global only: LCC (2)"] = {"X": G[:, :2], "group": "global"}
    rows["global only: 2x,4x pooled orbits (110)"] = {"X": G[:, 2:], "group": "global"}
    return rows


def whiten(X, ref_ix):
    std = geom.Standardizer.fit(X[ref_ix])
    return std(X)


def kidfn(r, p):
    return D.kid(p, r)


# --- protocols ----------------------------------------------------------------------------
def ladder_structs(seed: int, n_ladder: int):
    split = splits.load_split("houses_32", seed)

    def usable(s):
        return max(s.shape) > 16 and 300 <= int(s.occupied_mask.sum()) <= 4000

    ref_all = splits.split_structures(split, "val", predicate=usable)
    rng = np.random.default_rng(seed)
    ref = [ref_all[i].crop_to_non_air() for i in rng.permutation(len(ref_all))[:256]]
    probe = [s.crop_to_non_air() for s in splits.split_structures(split, "test", predicate=usable)]
    order = np.random.default_rng(seed).permutation(len(probe))
    probe = [probe[i] for i in order]
    n = min(n_ladder, len(probe) - 1)
    src = probe[:n]
    suite = pb.probe_suite(src, np.random.default_rng(seed))
    suite["real_heldout"] = src
    structs = list(ref)
    sets = {}
    for name, ss in suite.items():
        sets[name] = np.arange(len(structs), len(structs) + len(ss))
        structs.extend(ss)
    sets["null"] = np.arange(len(structs), len(structs) + len(probe))
    structs.extend(probe)
    return structs, np.arange(len(ref)), sets, n


def score_ladder(X, ref_ix, sets, reps, seed):
    r = L.score_rungs({"m": kidfn}, X[ref_ix], {k: X[v] for k, v in sets.items() if k != "null"},
                      X[sets["null"]], L.GEOMETRY_RUNGS, backbone="robust", view="robust",
                      reps=reps, rng=np.random.default_rng(seed + 3))
    sd = r.noise["m"]["sd"]
    sc = r.scores["m"]
    base = sc["real_heldout"]
    z = {k: float((sc[k] - base) / sd) for k in LADDER_PROBES if k in sc}
    return {"gates": r.gates["m"], "passed": bool(all(r.gates["m"].values())),
            "failed": [g for g, v in r.gates["m"].items() if not v],
            "sensitivity": r.sensitivity["m"], "sd": float(sd), "z": z}


def score_board(X, ref_ix, sets, arms, n, seed, ci_reps, vm):
    res = vm.score_space(X[ref_ix], X[sets["calib"]], {a: X[sets[a]] for a in arms}, n, seed,
                         kidfn=lambda a, r: kidfn(r, a), ci_reps=ci_reps)
    return {"sep": {a: res[a]["sep"] for a in arms if a != "real_test"},
            "ci": {a: res[a]["ci"] for a in arms if a != "real_test"}, "sd": res["_sd"]}


# --- samples.png -------------------------------------------------------------------------
def samples_png(run: Path, seed: int) -> None:
    import os
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from blockgen.eval.bench import baselines as bl
    from blockgen.renderer.textured import render_structure
    root = Path(__file__).resolve().parents[1]
    man = json.loads((root / "data/minecraft/cache/houses_32_manifest.json").read_text())
    split = splits.load_split("houses_32", seed)
    test_rows, train_rows = list(split.indices("test")), list(split.indices("train"))
    test = splits.split_structures(split, "test")
    train = splits.split_structures(split, "train")
    ok_t = [i for i in range(len(test)) if man["items"][test_rows[i]].get("corpus") == "3dcraft"]
    ok_tr = [train[i] for i in range(len(train))
             if man["items"][train_rows[i]].get("corpus") == "3dcraft"]
    s = max((test[i].crop_to_non_air() for i in ok_t[:60]),
            key=lambda t: int(t.occupied_mask.sum()) if max(t.shape) <= 32 else 0)
    ps = bl.patchwork(ok_tr, np.random.default_rng(seed)).crop_to_non_air()
    arms = {"real (3D-Craft)": s, "Fill-Interior": pb.solidify(s), "Hollow-Shell": pb.hollow(s),
            "Patch-Stitch (3D-Craft patches)": ps}
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.6))
    for c, (name, t) in enumerate(arms.items()):
        img = render_structure(t, px=320, azim_deg=45, elev_deg=28, bg=(1, 1, 1))
        ax[0, c].imshow(img)
        ax[0, c].set_title(name, fontsize=9)
        occ = t.occupied_mask
        y = occ.shape[1] // 3
        lab, ncomp = ndimage.label(occ, structure=S26)
        sizes = np.bincount(lab.ravel())[1:] if ncomp else np.array([0])
        ax[1, c].imshow(occ[:, y, :].T, cmap="Greys", origin="lower")
        ax[1, c].set_title(f"slice y={y}; {ncomp} parts, LCC {sizes.max() / max(sizes.sum(), 1):.2f}",
                           fontsize=8)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(run / "samples.png", dpi=110)
    (run / "samples.txt").write_text(
        "top: textured renders; bottom: occupancy slice with 26-connected part count and "
        "largest-part ratio. All builds 3D-Craft: one test house, its two probes, and a "
        "Patch-Stitch built from 3D-Craft training builds only (the scored Patch-Stitch arm "
        "draws from the whole training split).\n")


# --- main ----------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-ladder", type=int, default=128)
    ap.add_argument("--n-board", type=int, default=256)
    ap.add_argument("--reps", type=int, default=48)
    ap.add_argument("--ci-reps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    geom.set_descriptor_version("v1")
    run = Path(args.out) if args.out else new_run_dir("vkid_robustness")
    run.mkdir(parents=True, exist_ok=True)
    log_f = open(run / "log.txt", "a")

    def log(s):
        print(s, flush=True)
        log_f.write(s + "\n"); log_f.flush()

    log(f"[robust] -> {run}")
    try:
        samples_png(run, args.seed)
    except Exception as e:                                       # noqa: BLE001
        log(f"  samples.png failed: {e!r}")
    out = {"args": vars(args), "families": {k: len(v) for k, v in FAMILIES.items()},
           "struct_arms": list(STRUCT_ARMS)}

    # ladder
    structs, ref_ix, sets, n = ladder_structs(args.seed, args.n_ladder)
    log(f"[ladder] {len(structs)} structures, n={n}")
    F = featurize(structs, args.workers, log, "ladder")
    np.savez_compressed(run / "features_ladder.npz", **F)
    rows = build_rows(F, args.seed)
    lad = {}
    for name, r in rows.items():
        lad[name] = score_ladder(whiten(r["X"], ref_ix), ref_ix, sets, args.reps, args.seed)
        z = lad[name]["z"]
        log(f"  {name:44s} solid {z['solidify']:+7.1f} hollow {z['hollow']:+7.1f} "
            f"{'PASS' if lad[name]['passed'] else lad[name]['failed']}")
    out["ladder"] = {"n": n, "n_ref": int(len(ref_ix)), "rows": lad}
    (run / "results.json").write_text(json.dumps(out, indent=1, default=float))

    # board
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ablate_view_facade_section", Path(__file__).resolve().parent / "ablate_view_facade_section.py")
    vm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vm)
    ref, calib, arms, check = vm.build_arms(args.seed, args.n_board)
    log(f"[board] ref {len(ref)} calib {len(calib)} arms {len(arms)} same builds "
        f"{all(check.values()) if check else 'unchecked'}")
    bstructs = list(ref)
    bref = np.arange(len(ref))
    bsets = {}
    for name, ss in {"calib": calib, **arms}.items():
        bsets[name] = np.arange(len(bstructs), len(bstructs) + len(ss))
        bstructs.extend(ss)
    BF = featurize(bstructs, args.workers, log, "board")
    np.savez_compressed(run / "features_board.npz", **BF)
    brows = build_rows(BF, args.seed)
    board = {}
    full = None
    for name, r in brows.items():
        ci = args.ci_reps if r["group"] not in ("rand50", "rand25") else 20
        b = score_board(whiten(r["X"], bref), bref, bsets, arms, args.n_board, args.seed, ci, vm)
        if full is None:
            full = b
        b["rho_full"] = float(spearmanr([b["sep"][a] for a in STRUCT_ARMS],
                                        [full["sep"][a] for a in STRUCT_ARMS]).correlation)
        board[name] = b
        s = b["sep"]
        log(f"  {name:44s} solid {s['real@solidify']:+7.1f} hollow {s['real@hollow']:+7.1f} "
            f"patch {s['patchwork']:+7.1f} rho {b['rho_full']:+.2f}")
    out["board"] = {"n": args.n_board, "same_builds": check, "rows": board}
    out["row_meta"] = {k: {"group": v["group"], "dims": int(v["X"].shape[1]),
                           **({"ix": v["dims"]} if "dims" in v else {})} for k, v in rows.items()}
    (run / "results.json").write_text(json.dumps(out, indent=1, default=float))
    log(f"-> {run}")


if __name__ == "__main__":
    main()
