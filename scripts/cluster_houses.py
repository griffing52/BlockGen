"""Cluster the houses_32 builds by style, to pick a narrower subset to train on.

    python scripts/cluster_houses.py [--ks 4,6,8,12] [--views geom,dino,both] [--per-sheet 24]

Every build of the pinned split (train, val and test) is placed in one of three
feature spaces, the same ones the benchmark measures with:

- ``geom``: the v2 geometric descriptor behind ``geom_kid``, whitened on train
  (``geometry.Standardizer``), unit rows.
- ``dino``: the four-view DINOv2 features behind ``mv_dino_kid`` (cached), view
  mean, PCA to 32 dims fitted on train, unit rows.
- ``both``: the two concatenated, each block scaled to unit mean norm so neither
  dominates.

K-means is fitted on train only and assigns val and test, so a cluster's split
membership stays the split's. For every (view, k) the run writes one sheet per
cluster (the builds nearest its centre), ``clusters.json`` (labels per split
index, sizes per split, silhouette, and the spread ratio: mean within-cluster
distance over the mean distance of the whole set, below 1 = tighter than all
houses) and ``summary.md``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from blockgen.eval.bench import geometry as geom
from blockgen.eval.bench import splits
from blockgen.utils.runs import new_run_dir

FEATURES = Path("data/minecraft/features/houses_32")
DINO_VIEW = "v4_az45-135-225-315_el30-30-30-30_px224_ortho"
SPLITS = ("train", "val", "test")


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def load_features(structs: dict) -> dict:
    geom.set_descriptor_version("v2")
    raw = {s: geom.descriptor(structs[s]) for s in SPLITS}
    std = geom.Standardizer.fit(raw["train"])
    g = {s: std(raw[s]) for s in SPLITS}
    d = {s: np.load(FEATURES / s / DINO_VIEW / "dinov2b.npz")["feats"].mean(axis=1) for s in SPLITS}
    pca = PCA(32, random_state=0).fit(d["train"])
    d = {s: unit(pca.transform(d[s])) for s in SPLITS}
    both = {s: np.concatenate([g[s], d[s]], axis=1) / np.sqrt(2) for s in SPLITS}
    return {"geom": g, "dino": d, "both": both}


def spread_ratio(x: np.ndarray, labels: np.ndarray) -> list:
    def mean_dist(a):
        if len(a) < 2:
            return float("nan")
        sq = (a * a).sum(1)
        dd = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * a @ a.T, 0))
        return float(dd.sum() / (len(a) * (len(a) - 1)))
    whole = mean_dist(x)
    return [round(mean_dist(x[labels == c]) / whole, 3) for c in range(labels.max() + 1)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ks", default="4,6,8,12")
    ap.add_argument("--views", default="geom,dino,both")
    ap.add_argument("--per-sheet", type=int, default=24)
    a = ap.parse_args(argv)
    from blockgen.renderer.grid import save_grid

    split = splits.load_split("houses_32", 0)
    structs = {s: splits.split_structures(split, s) for s in SPLITS}
    feats = load_features(structs)
    out = new_run_dir("cluster_houses")
    report, lines = {}, ["# houses_32 style clusters", ""]
    for view in a.views.split(","):
        f = feats[view]
        for k in map(int, a.ks.split(",")):
            km = KMeans(k, n_init=10, random_state=0).fit(f["train"])
            labels = {s: km.predict(f[s]) for s in SPLITS}
            sizes = {s: np.bincount(labels[s], minlength=k).tolist() for s in SPLITS}
            sil = float(silhouette_score(f["train"], labels["train"]))
            spread = spread_ratio(f["train"], labels["train"])
            key = f"{view}_k{k}"
            report[key] = {"silhouette": round(sil, 3), "sizes": sizes, "spread_ratio": spread,
                           "labels": {s: labels[s].tolist() for s in SPLITS}}
            lines += [f"## {key}  silhouette {sil:.3f}", "",
                      "| cluster | train | val | test | spread ratio |", "|--:|--:|--:|--:|--:|"]
            for c in range(k):
                lines.append(f"| {c} | {sizes['train'][c]} | {sizes['val'][c]} | {sizes['test'][c]} | {spread[c]} |")
                idx = np.where(labels["train"] == c)[0]
                near = idx[np.argsort(np.linalg.norm(f["train"][idx] - km.cluster_centers_[c], axis=1))]
                save_grid([structs["train"][i] for i in near[:a.per_sheet]],
                          str(out / "sheets" / key / f"cluster{c}.png"), cols=8)
            lines.append("")
            print(f"[cluster] {key} silhouette {sil:.3f} train sizes {sizes['train']}", flush=True)
    (out / "clusters.json").write_text(json.dumps(report) + "\n")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"[cluster] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
