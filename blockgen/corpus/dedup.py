"""Duplicate clusters over the manifest.

Two passes (docs/pipeline.md §3.2):

1. exact: identical ``canon_hash`` (same build up to translation and the eight
   horizontal symmetries);
2. near: MinHash LSH banding over the 128-value signatures. Rows that share a
   band are candidates; a candidate pair is accepted when the signature
   agreement (an estimate of voxel-set Jaccard) is at least ``--jaccard``.

Union-find joins both into clusters. Output ``data/minecraft/corpus/dedup.parquet``
with columns ``id, cluster, is_rep, cluster_size, cluster_worlds``
(``cluster_worlds`` = distinct worlds the cluster occurs in; >= 5 means generated
content such as vanilla villages, not a human build); the representative of a
cluster is its highest ``p_built`` row, ties broken by block count.

CLI::

    python -m blockgen.corpus.dedup [--bands 16] [--jaccard 0.8] [--min-blocks 20]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from . import canon
from .manifest import CORPUS_DIR, load

OUT = os.path.join(CORPUS_DIR, "dedup.parquet")


class _UF:
    def __init__(self, n: int):
        self.p = np.arange(n)

    def find(self, i: int) -> int:
        p = self.p
        while p[i] != i:
            p[i] = p[p[i]]
            i = p[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.p[rb] = ra
            else:
                self.p[ra] = rb


def _groups(keys: np.ndarray):
    """Yield index arrays of rows sharing a key (groups of size >= 2)."""
    order = np.argsort(keys, kind="stable")
    ks = keys[order]
    bounds = np.flatnonzero(ks[1:] != ks[:-1]) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(ks)]])
    for s, e in zip(starts, ends):
        if e - s >= 2:
            yield order[s:e]


def _band_keys(sigs: np.ndarray, bands: int) -> np.ndarray:
    """[n, bands] uint64 hash of each band of the signature matrix."""
    n, L = sigs.shape
    r = L // bands
    out = np.empty((n, bands), np.uint64)
    mult = np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        for b in range(bands):
            h = np.zeros(n, np.uint64)
            for j in range(b * r, (b + 1) * r):
                h = (h ^ sigs[:, j].astype(np.uint64)) * mult
                h ^= h >> np.uint64(31)
            out[:, b] = h
    return out


def run(bands: int = 16, jaccard: float = 0.8, min_blocks: int = 20, max_group: int = 500) -> None:
    t0 = time.time()
    df = load(columns=["id", "world_id", "n_blocks", "p_built", "canon_hash", "sig"])
    n = len(df)
    print(f"{n} rows loaded in {time.time() - t0:.0f}s", flush=True)
    uf = _UF(n)

    # 1. exact
    hkeys = df.canon_hash.to_numpy()
    n_exact = 0
    for g in _groups(hkeys):
        for i in g[1:]:
            uf.union(int(g[0]), int(i))
        n_exact += len(g) - 1
    print(f"exact: {n_exact} rows folded", flush=True)

    # 2. near (skip tiny builds: a few voxels agree by chance)
    sigs = np.frombuffer(b"".join(df.sig.to_numpy()), dtype=np.uint32).reshape(n, canon.N_HASH)
    eligible = df.n_blocks.to_numpy() >= min_blocks
    bk = _band_keys(sigs, bands)
    n_pairs = n_acc = 0
    for b in range(bands):
        keys = np.where(eligible, bk[:, b], np.uint64(0))
        for g in _groups(keys):
            if keys[g[0]] == 0:
                continue
            if len(g) > max_group:
                g = g[:max_group]
            anchor = g[0]
            agree = (sigs[g[1:]] == sigs[anchor]).mean(axis=1)
            for i, a in zip(g[1:], agree):
                n_pairs += 1
                if a >= jaccard:
                    uf.union(int(anchor), int(i))
                    n_acc += 1
    print(f"near: {n_pairs} candidate pairs, {n_acc} accepted at jaccard>={jaccard}", flush=True)

    roots = np.array([uf.find(i) for i in range(n)])
    import pandas as pd
    out = pd.DataFrame({"id": df.id, "cluster": roots, "p_built": df.p_built.fillna(1.0), "n_blocks": df.n_blocks})
    out["cluster_size"] = out.groupby("cluster")["id"].transform("size")
    # how many distinct worlds a cluster appears in: vanilla village pieces and
    # other generated structures recur across hundreds of unrelated worlds,
    # while a human build is copied into a handful at most
    out["cluster_worlds"] = df.world_id.groupby(roots).transform("nunique").to_numpy()
    best = out.sort_values(["cluster", "p_built", "n_blocks"], ascending=[True, False, False]).drop_duplicates("cluster")
    out["is_rep"] = out.index.isin(best.index)
    out[["id", "cluster", "is_rep", "cluster_size", "cluster_worlds"]].to_parquet(OUT, index=False)
    n_clusters = out.cluster.nunique()
    print(f"{n} rows -> {n_clusters} clusters ({n - n_clusters} duplicates, "
          f"{(n - n_clusters) / max(1, n):.1%}); largest cluster {out.cluster_size.max()}; "
          f"{time.time() - t0:.0f}s -> {OUT}", flush=True)
    print(f"clusters in >=5 worlds: {int((out.drop_duplicates('cluster').cluster_worlds >= 5).sum())} "
          f"({int((out.cluster_worlds >= 5).sum())} rows) -- generated content, see docs/pipeline.md §3.2", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bands", type=int, default=16)
    ap.add_argument("--jaccard", type=float, default=0.8)
    ap.add_argument("--min-blocks", type=int, default=20)
    a = ap.parse_args(argv)
    run(a.bands, a.jaccard, a.min_blocks)


if __name__ == "__main__":
    main()
