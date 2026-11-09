"""Grow the houses_32 style clusters with builds from a pretraining tier.

    python scripts/cluster_assign.py features [--tier anneal] [--max-dim 32] [--chunk 2048] [--skip anneal]
    python scripts/cluster_assign.py assign --clusters <cluster_houses run> [--tier anneal] [--views dino_k6,geom_k6,both_k6]

The styles are defined on the curated houses only (``scripts/cluster_houses.py``):
clustering a pool that is mostly world crops would spend its centres on terrain
and fragments. ``features`` embeds every tier build that fits ``max-dim`` in the
two spaces the clusters use (v2 geometric descriptor, raw; four-view DINOv2, the
benchmark's view) and writes them in resumable chunks to
``data/minecraft/features/pool_<tier><max-dim>/``. ``assign`` refits the houses
clusters (same seed, so the same labels as the cluster run, checked), places
every pool build at its nearest centre and accepts it when it is as close as
the cluster's own train houses (distance at most the members' 90th percentile).
A pool build whose voxels equal a houses_32 build is flagged and not counted as
new. The run writes ``assign.json`` (per view, per cluster: accepted ids by
source and distance), ``summary.md`` and sheets of accepted pool builds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from blockgen.utils.data import Structure

ROOT = Path(os.environ.get("BLOCKGEN_ROOT", "."))
FEAT_ROOT = ROOT / "data/minecraft/features"
SPLITS = ("train", "val", "test")


def pool_dir(tier: str, max_dim: int) -> Path:
    return FEAT_ROOT / f"pool_{tier}{max_dim}"


def pool_rows(tier: str, max_dim: int):
    from blockgen.corpus.manifest import load
    from blockgen.corpus.shards import _tier_rows
    _, df = _tier_rows(tier)
    dims = load(columns=["id", "dx", "dy", "dz"])
    df = df.merge(dims, on="id")
    df = df[df[["dx", "dy", "dz"]].max(axis=1) <= max_dim]
    return df.sort_values("path").reset_index(drop=True)


def load_structs(paths, cache):
    from blockgen.corpus.shards import _load_build, _tight
    out = []
    for p in paths:
        ids, data = _tight(*_load_build(p, cache))
        out.append(Structure(block_ids=ids, block_data=data))
    return out


def voxel_sha(s: Structure) -> str:
    c = s.crop_to_non_air()
    return hashlib.sha1(np.ascontiguousarray(c.block_ids).tobytes() + str(c.shape).encode()).hexdigest()[:16]


def features(tier: str, max_dim: int, chunk: int, skip: tuple = ()) -> None:
    from blockgen.eval.bench import features as F
    from blockgen.eval.bench import geometry as geom
    geom.set_descriptor_version("v2")
    import pandas as pd
    df = pool_rows(tier, max_dim)
    for t in skip:                                   # builds another pool already embedded
        done = pd.read_parquet(pool_dir(t, max_dim) / "rows.parquet", columns=["id"]).id
        df = df[~df.id.isin(set(done))].reset_index(drop=True)
    out = pool_dir(tier, max_dim)
    out.mkdir(parents=True, exist_ok=True)
    df[["id", "source", "path", "n_blocks", "p_built"]].to_parquet(out / "rows.parquet")
    n_chunks = (len(df) + chunk - 1) // chunk
    print(f"[features] {tier} max_dim {max_dim}: {len(df)} builds, {n_chunks} chunks -> {out}", flush=True)
    cache: dict = {}
    for c in range(n_chunks):
        f = out / f"chunk{c:04d}.npz"
        if f.exists():
            continue
        rows = df.iloc[c * chunk:(c + 1) * chunk]
        structs = load_structs(rows.path.tolist(), cache)
        g = geom.descriptor(structs)
        d = F.embed_views(structs, verbose=False).astype(np.float16)
        sha = np.array([voxel_sha(s) for s in structs])
        np.savez(f, ids=rows.id.to_numpy(), geom=g, dino=d, sha=sha)
        print(f"[features] chunk {c + 1}/{n_chunks}", flush=True)
    print("[features] done", flush=True)


def load_pool(tier: str, max_dim: int):
    """Features of one pool, or of several (``anneal,pretrain-all``) concatenated."""
    parts = [np.load(f, allow_pickle=True)                       # ids are object arrays
             for t in tier.split(",") for f in sorted(pool_dir(t, max_dim).glob("chunk*.npz"))]
    return {k: np.concatenate([p[k] for p in parts]) for k in ("ids", "geom", "dino", "sha")}


def pool_table(tier: str, max_dim: int):
    """The rows (id-indexed) of one pool or a comma list of pools."""
    import pandas as pd
    return pd.concat([pd.read_parquet(pool_dir(t, max_dim) / "rows.parquet") for t in tier.split(",")]).set_index("id")


def house_space():
    """The houses_32 structures and the embedding the clusters live in, rebuilt exactly as
    cluster_houses does: ``(structs, houses, embed)`` where ``embed(geom_raw, dino_view_mean)``
    maps any builds into the same three views."""
    from sklearn.decomposition import PCA

    from blockgen.eval.bench import geometry as geom
    from blockgen.eval.bench import splits
    import cluster_houses as ch

    geom.set_descriptor_version("v2")
    split = splits.load_split("houses_32", 0)
    structs = {s: splits.split_structures(split, s) for s in SPLITS}
    raw = {s: geom.descriptor(structs[s]) for s in SPLITS}
    std = geom.Standardizer.fit(raw["train"])
    dino = {s: np.load(ch.FEATURES / s / ch.DINO_VIEW / "dinov2b.npz")["feats"].mean(axis=1) for s in SPLITS}
    pca = PCA(32, random_state=0).fit(dino["train"])

    def embed(g_raw, d_mean):
        g = std(g_raw)
        d = ch.unit(pca.transform(d_mean))
        return {"geom": g, "dino": d, "both": np.concatenate([g, d], 1) / np.sqrt(2)}

    return structs, {s: embed(raw[s], dino[s]) for s in SPLITS}, embed


def fit_view(clusters_run: str, key: str, houses: dict):
    """The k-means of one ``<view>_k<k>`` refitted on the train houses (checked against the
    cluster run's labels) and each cluster's acceptance radius (members' 90th percentile)."""
    from sklearn.cluster import KMeans
    ref = json.loads((Path(clusters_run) / "clusters.json").read_text())
    view, k = key.split("_k")
    k = int(k)
    km = KMeans(k, n_init=10, random_state=0).fit(houses["train"][view])
    if not (km.labels_ == np.array(ref[key]["labels"]["train"])).all():
        raise RuntimeError(f"{key}: refit labels differ from {clusters_run}")
    member_d = np.linalg.norm(houses["train"][view] - km.cluster_centers_[km.labels_], axis=1)
    thr = np.array([np.percentile(member_d[km.labels_ == c], 90) for c in range(k)])
    return view, km, thr


def assign(clusters_run: str, tier: str, max_dim: int, views: list, per_sheet: int) -> None:
    import pandas as pd

    from blockgen.renderer.grid import save_grid
    from blockgen.utils.runs import new_run_dir

    structs, houses, embed = house_space()
    house_sha = {s: {voxel_sha(x) for x in structs[s]} for s in SPLITS}
    pool = load_pool(tier, max_dim)
    rows = pool_table(tier, max_dim)
    src = rows.loc[pool["ids"], "source"].to_numpy()
    pf = embed(pool["geom"], pool["dino"].astype(np.float32).mean(axis=1))
    dup = np.array([h in house_sha["train"] for h in pool["sha"]])
    leak = np.array([h in house_sha["val"] or h in house_sha["test"] for h in pool["sha"]])

    out = new_run_dir(f"cluster_assign_{tier.replace(',', '+')}{max_dim}")
    report = {"clusters_run": clusters_run, "tier": tier, "max_dim": max_dim, "n_pool": int(len(src)),
              "dup_train": int(dup.sum()), "dup_holdout": int(leak.sum()), "views": {}}
    lines = [f"# {tier} (max dim {max_dim}) assigned to the houses_32 clusters of `{clusters_run}`", "",
             f"Pool {len(src)} builds; {int(dup.sum())} equal a houses_32 train build and {int(leak.sum())} "
             f"a val/test build (both excluded from the counts).", ""]
    cache: dict = {}
    for key in views:
        view, km, thr = fit_view(clusters_run, key, houses)
        k = len(thr)
        dist = np.linalg.norm(pf[view][:, None] - km.cluster_centers_[None], axis=2)
        lab = dist.argmin(1)
        dmin = dist[np.arange(len(lab)), lab]
        ok = (dmin <= thr[lab]) & ~dup & ~leak
        vr = {}
        lines += [f"## {key}", "", "| cluster | houses train | pool accepted | worldcrop | grabcraft | 3dcraft | templates | text2mc |",
                  "|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for c in range(k):
            sel = np.where(ok & (lab == c))[0]
            by = pd.Series(src[sel]).value_counts().to_dict()
            vr[c] = {"accepted": int(len(sel)), "by_source": by, "threshold": float(thr[c]),
                     "ids": pool["ids"][sel].tolist(), "dist": np.round(dmin[sel], 4).tolist()}
            t2m = by.get("text2mc", 0) + by.get("text2mc_schem", 0)
            lines.append(f"| {c} | {int((km.labels_ == c).sum())} | {len(sel)} | {by.get('worldcrop', 0)} | "
                         f"{by.get('grabcraft', 0)} | {by.get('3dcraft', 0)} | {by.get('templates', 0)} | {t2m} |")
            if len(sel):
                pick = np.random.default_rng(c).choice(sel, min(per_sheet, len(sel)), replace=False)
                paths = rows.loc[pool["ids"][np.sort(pick)], "path"].tolist()
                save_grid(load_structs(paths, cache), str(out / "sheets" / key / f"cluster{c}_pool.png"), cols=8)
        lines.append("")
        report["views"][key] = vr
        print(f"[assign] {key}: accepted {[vr[c]['accepted'] for c in range(k)]}", flush=True)
    (out / "assign.json").write_text(json.dumps(report) + "\n")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"[assign] -> {out}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    a = sub.add_parser("assign")
    for p in (f, a):
        p.add_argument("--tier", default="anneal")
        p.add_argument("--max-dim", type=int, default=32)
    f.add_argument("--chunk", type=int, default=2048)
    f.add_argument("--skip", default="", help="comma list of pools whose builds are already embedded")
    a.add_argument("--clusters", required=True)
    a.add_argument("--views", default="dino_k6,geom_k6,both_k6")
    a.add_argument("--per-sheet", type=int, default=32)
    x = ap.parse_args(argv)
    if x.cmd == "features":
        features(x.tier, x.max_dim, x.chunk, tuple(t for t in x.skip.split(",") if t))
    else:
        assign(x.clusters, x.tier, x.max_dim, x.views.split(","), x.per_sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
