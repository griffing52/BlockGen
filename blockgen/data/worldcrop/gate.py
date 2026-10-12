"""Learned gate: is a world crop a built thing or a piece of terrain?

The detector (``detect.py``) finds components of player-placed block types; it has
no notion of *shape*, so edited terrain (WorldPainter plates, superflat layers, snow
block fields) comes through whenever it is made of a "built" id. This module scores
each crop with a classifier trained on

* positives: builds people labelled as builds (GrabCraft, 3D-Craft, the curated
  house caches) -- masked to their blocks, like a crop is masked to its component;
* negatives: natural terrain patches cut from real worlds by ``gate_data.py``
  (chunks with no built block at all), plus optionally a hand-labelled set.

Features are cheap hand-made descriptors (block-id histogram, family fractions,
heightmap statistics, vertical profile, surface/enclosure/symmetry/planarity), so
scoring 400k crops is minutes, not hours, and the model is a gradient-boosted
tree that can be inspected.

    python -m blockgen.data.worldcrop.gate train                 # -> gate/model.pkl
    python -m blockgen.data.worldcrop.gate score [--workers 8]   # -> <month>/gate_scores.jsonl
    python -m blockgen.data.worldcrop.gate sheets [--thr 0.5]    # top / bottom / borderline sheets
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from . import blocks

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
WORLDCROP = os.path.normpath(os.path.join(ROOT, "data", "minecraft", "worldcrop"))
GATE_DIR = os.path.join(WORLDCROP, "gate")
MODEL_PATH = os.path.join(GATE_DIR, "model.pkl")

N_HIST = 256
PROFILE_BINS = 8
_LUTS = None


def _luts():
    global _LUTS
    if _LUTS is None:
        strong, weak = blocks.class_tables()
        terrain = np.zeros(blocks.MAX_ID, bool); terrain[list(blocks.TERRAIN)] = True
        ore = np.zeros(blocks.MAX_ID, bool); ore[list(blocks.ORES)] = True
        plant = np.zeros(blocks.MAX_ID, bool); plant[[6, 18, 161, 31, 32, 37, 38, 39, 40, 81, 83, 86, 103, 106, 111, 175, 59, 141, 142]] = True
        liquid = np.zeros(blocks.MAX_ID, bool); liquid[[8, 9, 10, 11]] = True
        _LUTS = (strong, weak, terrain, ore, plant, liquid)
    return _LUTS


def feature_names() -> List[str]:
    names = [f"id{i}" for i in range(N_HIST)]
    names += ["f_strong", "f_weak", "f_terrain", "f_ore", "f_plant", "f_liquid", "n_distinct", "entropy",
              "log_n", "log_d0", "log_d1", "log_d2", "fill", "h_over_max", "h_over_min",
              "col_frac", "hm_std", "hm_range", "hm_rough", "solid_col", "thick_mean", "thick_std",
              "exposed", "enclosed", "sym_x", "sym_z", "plane_y", "plane_x", "plane_z", "top_layer_frac",
              "bottom_layer_frac"]
    names += [f"prof{i}" for i in range(PROFILE_BINS)]
    return names


def features(ids: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Descriptor of one build. ``ids`` is XYZ legacy ids; ``mask`` the member voxels
    (None = every non-air voxel)."""
    occ = (mask.astype(bool) if mask is not None else (ids != 0))
    out = np.zeros(N_HIST + 31 + PROFILE_BINS, np.float32)
    n = int(occ.sum())
    if n == 0:
        return out
    # tight crop to the member bbox
    xs, ys, zs = np.nonzero(occ)
    sl = np.s_[xs.min():xs.max() + 1, ys.min():ys.max() + 1, zs.min():zs.max() + 1]
    occ = occ[sl]; ids = np.asarray(ids)[sl]
    X, Y, Z = occ.shape
    strong, weak, terrain, ore, plant, liquid = _luts()
    v = ids[occ].astype(np.int64)
    vc = np.clip(v, 0, blocks.MAX_ID - 1)
    hist = np.bincount(np.clip(v, 0, N_HIST - 1), minlength=N_HIST).astype(np.float32) / n
    out[:N_HIST] = hist
    k = N_HIST
    out[k:k + 6] = [strong[vc].mean(), weak[vc].mean(), terrain[vc].mean(), ore[vc].mean(), plant[vc].mean(), liquid[vc].mean()]
    p = hist[hist > 0]
    out[k + 6] = len(p)
    out[k + 7] = float(-(p * np.log(p)).sum())
    dims = sorted((X, Y, Z))
    out[k + 8] = np.log(n)
    out[k + 9:k + 12] = np.log(dims)
    out[k + 12] = n / (X * Y * Z)
    out[k + 13] = Y / max(X, Z)
    out[k + 14] = Y / min(X, Z)
    # column statistics (heightmap over x,z)
    col = occ.sum(1)                                  # (X, Z) member count per column
    has = col > 0
    top = np.where(has, Y - 1 - np.argmax(occ[:, ::-1, :], axis=1), -1)
    bot = np.where(has, np.argmax(occ, axis=1), -1)
    out[k + 15] = has.mean()
    t = top[has].astype(np.float32)
    out[k + 16] = t.std() / Y
    out[k + 17] = (t.max() - t.min()) / Y
    dx = np.abs(np.diff(top, axis=0))[has[1:] & has[:-1]]
    dz = np.abs(np.diff(top, axis=1))[has[:, 1:] & has[:, :-1]]
    d = np.concatenate([dx, dz]).astype(np.float32)
    out[k + 18] = d.mean() / Y if d.size else 0.0
    span = top - bot + 1
    out[k + 19] = float(((col == span) & has).sum()) / max(has.sum(), 1)   # gap-free columns
    th = col[has].astype(np.float32) / Y
    out[k + 20] = th.mean(); out[k + 21] = th.std()
    # exposed faces per member voxel
    pad = np.pad(occ, 1)
    nb = (pad[2:, 1:-1, 1:-1].astype(np.int8) + pad[:-2, 1:-1, 1:-1] + pad[1:-1, 2:, 1:-1]
          + pad[1:-1, :-2, 1:-1] + pad[1:-1, 1:-1, 2:] + pad[1:-1, 1:-1, :-2])
    out[k + 22] = float((6 - nb[occ]).sum()) / (6.0 * n)
    # enclosed air (rooms) relative to bbox volume
    filled = ndimage.binary_fill_holes(occ)
    out[k + 23] = float((filled & ~occ).sum()) / (X * Y * Z)
    # mirror symmetry
    out[k + 24] = float((occ & occ[::-1]).sum()) / n
    out[k + 25] = float((occ & occ[:, :, ::-1]).sum()) / n
    # planarity: biggest single layer / plane share
    out[k + 26] = float(occ.sum((0, 2)).max()) / n
    out[k + 27] = float(occ.sum((1, 2)).max()) / n
    out[k + 28] = float(occ.sum((0, 1)).max()) / n
    out[k + 29] = float(occ[:, -1, :].sum()) / n
    out[k + 30] = float(occ[:, 0, :].sum()) / n
    # vertical occupancy profile
    prof = occ.sum((0, 2)).astype(np.float32)
    idx = (np.arange(Y) * PROFILE_BINS) // Y
    out[k + 31:k + 31 + PROFILE_BINS] = np.bincount(idx, weights=prof, minlength=PROFILE_BINS) / n
    return out


# --------------------------------------------------------------------------- #
# training data
# --------------------------------------------------------------------------- #
def _positives(max_per_source: int = 20000) -> Iterator[Tuple[np.ndarray, np.ndarray, str]]:
    """Builds people labelled as builds, masked to their non-air voxels."""
    sys.path.insert(0, ROOT)
    from tools.lab import rawcorpora as rc
    for sid in ("grabcraft", "3dcraft"):
        idx = rc.index(sid)
        for i in range(min(len(idx), max_per_source)):
            s = rc.load_one(sid, i)
            if s is None:
                continue
            yield s.block_ids, s.block_ids != 0, sid
    from blockgen.curation.houses import load_structures_from_cache
    for name in ("houses_48",):
        p = os.path.join(ROOT, "data", "minecraft", "cache", f"{name}.npz")
        if os.path.exists(p):
            for s in load_structures_from_cache(p)[0]:
                if "text2mc" in str(s.metadata.get("corpus", "")):
                    yield s.block_ids, s.block_ids != 0, f"{name}:text2mc"


def _negatives() -> Iterator[Tuple[np.ndarray, np.ndarray, str]]:
    files = sorted(glob.glob(os.path.join(GATE_DIR, "terrain_*.npz")) + glob.glob(os.path.join(GATE_DIR, "synth_*.npz"))
                   + glob.glob(os.path.join(GATE_DIR, "mined_*.npz")))
    for p in files:
        z = np.load(p, allow_pickle=True)
        base = os.path.basename(p)
        tag = base.rsplit("_", 1)[0] if base.startswith(("synth", "mined")) else "terrain"
        for ids, mask in zip(z["ids"], z["mask"]):
            yield ids, mask, tag


def _labelled() -> Iterator[Tuple[np.ndarray, np.ndarray, str, int]]:
    """Optional hand labels: gate/labels.jsonl rows {month, file, label: built|terrain|fragment}."""
    p = os.path.join(GATE_DIR, "labels.jsonl")
    if not os.path.exists(p):
        return
    for line in open(p):
        r = json.loads(line)
        if r["label"] not in ("built", "terrain"):
            continue
        z = np.load(os.path.join(WORLDCROP, r["month"], "crops", r["file"]))
        yield z["ids"], z["mask"], "hand:" + r["label"], int(r["label"] == "built")


def _featurize(items, cache: Optional[str] = None):
    """[(ids, mask, source)] -> (X, src); positives are cached since they never change."""
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["X"], list(z["src"])
    X, src = [], []
    for ids, m, s in items:
        X.append(features(ids, m)); src.append(s)
    X = np.stack(X) if X else np.zeros((0, len(feature_names())), np.float32)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, X=X, src=np.asarray(src))
    return X, src


def build_dataset(limit_pos: int = 20000) -> dict:
    Xp, sp = _featurize(_positives(limit_pos), os.path.join(GATE_DIR, "feat_positives.npz"))
    Xn, sn = _featurize(_negatives())
    Xl, sl, yl = [], [], []
    for ids, m, s, lab in _labelled():
        Xl.append(features(ids, m)); sl.append(s); yl.append(lab)
    Xl = np.stack(Xl) if Xl else np.zeros((0, Xp.shape[1]), np.float32)
    X = np.concatenate([Xp, Xn, Xl]); y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn)), np.asarray(yl)]).astype(int)
    return {"X": X, "y": y, "src": np.asarray(list(sp) + list(sn) + sl)}


def train(out_path: str = MODEL_PATH, seed: int = 0) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score, average_precision_score
    os.makedirs(GATE_DIR, exist_ok=True)
    ds = build_dataset()
    X, y, src = ds["X"], ds["y"], ds["src"]
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    rep = {"n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
           "cv_auc": float(roc_auc_score(y, p)), "cv_ap": float(average_precision_score(y, p)),
           "by_source": {}}
    for s in sorted(set(src)):
        sel = src == s
        yy = y[sel]
        rep["by_source"][s] = {"n": int(sel.sum()),
                               "acc@0.5": float(((p[sel] > 0.5) == yy).mean()),
                               "mean_p": float(p[sel].mean())}
    clf.fit(X, y)
    with open(out_path, "wb") as f:
        pickle.dump({"model": clf, "names": feature_names()}, f)
    with open(os.path.join(GATE_DIR, "train_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    return rep


# --------------------------------------------------------------------------- #
# scoring the corpus
# --------------------------------------------------------------------------- #
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        with open(MODEL_PATH, "rb") as f:
            _MODEL = pickle.load(f)["model"]
    return _MODEL


def _score_month(month_dir: str) -> Tuple[str, int]:
    out_p = os.path.join(month_dir, "gate_scores.jsonl")
    done = set()
    if os.path.exists(out_p):
        for line in open(out_p):
            try:
                done.add(json.loads(line)["file"])
            except Exception:
                pass
    rows, files = [], []
    for line in open(os.path.join(month_dir, "crops.jsonl")):
        c = json.loads(line)
        if c.get("file") and c["file"] not in done:
            files.append(c["file"])
    if not files:
        return month_dir, 0
    clf = _model()
    feats, keep = [], []
    for fn in files:
        try:
            z = np.load(os.path.join(month_dir, "crops", fn))
            feats.append(features(z["ids"], z["mask"])); keep.append(fn)
        except Exception:
            continue
        if len(feats) >= 2000:
            p = clf.predict_proba(np.stack(feats))[:, 1]
            rows += [{"file": f, "p_built": round(float(v), 4)} for f, v in zip(keep, p)]
            feats, keep = [], []
    if feats:
        p = clf.predict_proba(np.stack(feats))[:, 1]
        rows += [{"file": f, "p_built": round(float(v), 4)} for f, v in zip(keep, p)]
    with open(out_p, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return month_dir, len(rows)


def score_all(workers: int = 8) -> Dict[str, int]:
    months = sorted(d for d in glob.glob(os.path.join(WORLDCROP, "2*")) if os.path.isdir(d))
    out = {}
    with ProcessPoolExecutor(workers) as ex:
        for d, n in ex.map(_score_month, months):
            out[os.path.basename(d)] = n
            print(f"{os.path.basename(d)}: scored {n}", flush=True)
    return out


def load_scores() -> List[dict]:
    rows = []
    for d in sorted(glob.glob(os.path.join(WORLDCROP, "2*"))):
        p = os.path.join(d, "gate_scores.jsonl")
        if os.path.exists(p):
            m = os.path.basename(d)
            for line in open(p):
                r = json.loads(line); r["month"] = m; rows.append(r)
    return rows


def sheets(thr: float = 0.5, n: int = 96, cols: int = 12, max_dim: int = 128, seed: int = 0):
    from blockgen.renderer.grid import save_grid
    from blockgen.utils.data import Structure
    rows = load_scores()
    rng = random.Random(seed)
    def pick(lo, hi):
        pool = [r for r in rows if lo <= r["p_built"] < hi]
        rng.shuffle(pool)
        out = []
        for r in pool:
            z = np.load(os.path.join(WORLDCROP, r["month"], "crops", r["file"]))
            if max(z["ids"].shape) > max_dim:
                continue
            m = z["mask"]
            out.append(Structure(block_ids=z["ids"].astype(np.int32) * m, block_data=z["data"].astype(np.int32) * m,
                                 metadata={"corpus": "worldcrop"}))
            if len(out) >= n:
                break
        return out, len(pool)
    stats = {}
    for name, lo, hi in (("pass", thr, 1.01), ("reject", -0.01, min(thr, 0.2)), ("border", max(thr - 0.15, 0), min(thr + 0.15, 1.01))):
        sel, total = pick(lo, hi)
        if sel:
            save_grid(sel, os.path.join(GATE_DIR, f"sheet_{name}.png"), cols=cols)
        stats[name] = total
    ps = np.array([r["p_built"] for r in rows])
    stats["n"] = int(len(ps))
    stats["hist"] = {f"{a/10:.1f}": int(((ps >= a / 10) & (ps < (a + 1) / 10)).sum()) for a in range(10)}
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "score", "sheets"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args(argv)
    if args.cmd == "train":
        print(json.dumps(train(), indent=1))
    elif args.cmd == "score":
        print(json.dumps(score_all(args.workers)))
    else:
        print(json.dumps(sheets(args.thr), indent=1))


if __name__ == "__main__":
    main()
