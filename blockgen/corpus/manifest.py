"""Build the corpus manifest: one row per build, every source.

Output, under ``data/minecraft/corpus/manifest/``::

    worldcrop_<month>.parquet      one row per saved crop of that month
    worldcrop_<month>_feats.npy    float16 [n, 295] gate features, row-aligned
    curated_<cache>.parquet        one row per build of a curated cache
    curated_<cache>_feats.npy

Auxiliary caches (``curated --aux``) go to ``data/minecraft/corpus/manifest_aux/``
instead, with the same schema. ``load`` reads them only when asked by name
(``load(aux=[...])``, a tier's ``aux:`` key), so a cache added for one family of
tiers (the GrabCraft 48 cache behind the category tiers, ``blockgen.corpus.labels``)
never changes what an existing tier resolves to when it is rebuilt.

Row schema (see docs/pipeline.md §3.1): ``id, source, path, month, world_id, title,
author, url, dx, dy, dz, n_blocks, fill, n_ids, p_built, cut, oversize, route,
canon_hash, sig`` where ``sig`` is the 128 x uint32 MinHash signature as bytes.

Resumable per month: a month whose parquet already holds one row per saved crop
is skipped, and the month the ingest is currently writing is skipped too.

CLI::

    python -m blockgen.corpus.manifest build   [--workers 6] [--months 201407 ...] [--limit N]
    python -m blockgen.corpus.manifest curated [--caches all_32 houses_32 houses_48]
    python -m blockgen.corpus.manifest curated --aux --caches gc_small_48
    python -m blockgen.corpus.manifest summary
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import canon

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORLDCROP = os.path.join(ROOT, "data", "minecraft", "worldcrop")
CACHE_DIR = os.path.join(ROOT, "data", "minecraft", "cache")
CORPUS_DIR = os.path.join(ROOT, "data", "minecraft", "corpus")
MANIFEST_DIR = os.path.join(CORPUS_DIR, "manifest")
MANIFEST_AUX_DIR = os.path.join(CORPUS_DIR, "manifest_aux")

COLUMNS = ["id", "source", "path", "month", "world_id", "title", "author", "url",
           "dx", "dy", "dz", "n_blocks", "fill", "n_ids", "p_built", "cut", "oversize",
           "route", "canon_hash", "sig"]

_CLF = None


def _clf():
    global _CLF
    if _CLF is None:
        from blockgen.data.worldcrop.gate import _model
        _CLF = _model()
    return _CLF


def describe(ids: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[dict, np.ndarray]:
    """Geometry columns + canonical hash + MinHash + gate features for one build."""
    from blockgen.data.worldcrop.gate import features
    occ = mask.astype(bool) if mask is not None else (ids != 0)
    tight = canon.tighten(ids, occ)
    n_blocks = int(np.count_nonzero(tight))
    h, sig, c = canon.canon_and_sig(tight)
    feats = features(ids, occ).astype(np.float32)
    row = {
        "dx": int(tight.shape[0]), "dy": int(tight.shape[1]), "dz": int(tight.shape[2]),
        "n_blocks": n_blocks,
        "fill": float(n_blocks / max(1, tight.size)),
        "n_ids": int(len(np.unique(tight[tight != 0]))),
        "canon_hash": h, "sig": sig.tobytes(),
    }
    return row, feats


# --------------------------------------------------------------------------- worldcrop

def _world_meta(month_dir: str) -> Dict[str, Tuple[str, str, str]]:
    out = {}
    p = os.path.join(month_dir, "worlds.jsonl")
    if not os.path.exists(p):
        return out
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        e = r.get("entry") or {}
        meta = e.get("meta") or {}
        out[r["world_id"]] = (e.get("title") or "", meta.get("author") or "", e.get("url") or "")
    return out


def _gate_scores(month_dir: str) -> Dict[str, float]:
    p = os.path.join(month_dir, "gate_scores.jsonl")
    out = {}
    if os.path.exists(p):
        for line in open(p):
            try:
                r = json.loads(line)
                out[r["file"]] = float(r["p_built"])
            except Exception:
                pass
    return out


def _crop_rows(month_dir: str) -> List[dict]:
    rows = []
    p = os.path.join(month_dir, "crops.jsonl")
    if not os.path.exists(p):
        return rows
    for line in open(p):
        try:
            c = json.loads(line)
        except Exception:
            continue
        if c.get("file"):
            rows.append(c)
    return rows


def _chunk_worker(args) -> Tuple[List[dict], np.ndarray, List[float]]:
    """Describe a chunk of crops of one month. Returns rows, feats, p_built (NaN if unknown)."""
    month_dir, crops, scores = args
    rows, feats, probs, need = [], [], [], []
    for c in crops:
        fp = os.path.join(month_dir, "crops", c["file"])
        try:
            z = np.load(fp)
            row, f = describe(z["ids"], z["mask"])
        except Exception:
            continue
        rows.append({**c, **row})
        feats.append(f)
        p = scores.get(c["file"])
        probs.append(np.nan if p is None else p)
        need.append(p is None)
    if not rows:
        return rows, np.zeros((0, 295), np.float16), probs
    F = np.stack(feats)
    if any(need):
        idx = np.flatnonzero(need)
        pr = _clf().predict_proba(F[idx])[:, 1]
        for i, v in zip(idx, pr):
            probs[i] = float(v)
    return rows, F.astype(np.float16), probs


def _write(name: str, rows: List[dict], feats: np.ndarray, out_dir: str = MANIFEST_DIR) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(out_dir, exist_ok=True)
    cols = {k: [r.get(k) for r in rows] for k in COLUMNS}
    table = pa.table(cols)
    pq.write_table(table, os.path.join(out_dir, f"{name}.parquet"), compression="zstd")
    np.save(os.path.join(out_dir, f"{name}_feats.npy"), feats)


def _month_done(month: str, n_expected: int) -> bool:
    p = os.path.join(MANIFEST_DIR, f"worldcrop_{month}.parquet")
    if not os.path.exists(p):
        return False
    import pyarrow.parquet as pq
    return pq.read_metadata(p).num_rows >= n_expected


def build_month(month: str, workers: int, limit: int = 0, chunk: int = 400) -> int:
    month_dir = os.path.join(WORLDCROP, month)
    crops = _crop_rows(month_dir)
    if limit:
        crops = crops[:limit]
    saved = [c for c in crops if os.path.exists(os.path.join(month_dir, "crops", c["file"]))]
    if not saved:
        return 0
    if not limit and _month_done(month, len(saved)):
        return -1
    meta = _world_meta(month_dir)
    scores = _gate_scores(month_dir)
    jobs = [(month_dir, saved[i:i + chunk], {c["file"]: scores[c["file"]] for c in saved[i:i + chunk] if c["file"] in scores})
            for i in range(0, len(saved), chunk)]
    rows, feats = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r, F, probs in ex.map(_chunk_worker, jobs):
            for row, p in zip(r, probs):
                title, author, url = meta.get(row["world_id"], ("", "", ""))
                rows.append({
                    "id": "wc:" + row["file"][:-4], "source": "worldcrop",
                    "path": os.path.relpath(os.path.join(month_dir, "crops", row["file"]), ROOT),
                    "month": month, "world_id": row["world_id"], "title": title, "author": author, "url": url,
                    "dx": row["dx"], "dy": row["dy"], "dz": row["dz"], "n_blocks": row["n_blocks"],
                    "fill": row["fill"], "n_ids": row["n_ids"], "p_built": p,
                    "cut": bool(row.get("cut", False)), "oversize": bool(row.get("oversize", False)),
                    "route": row.get("route", ""), "canon_hash": row["canon_hash"], "sig": row["sig"],
                })
            feats.append(F)
    F = np.concatenate(feats) if feats else np.zeros((0, 295), np.float16)
    _write(f"worldcrop_{month}" + ("_sample" if limit else ""), rows, F)
    return len(rows)


def _in_progress() -> Optional[str]:
    try:
        from blockgen.data.worldcrop.provenance import _month_in_progress
        return _month_in_progress(WORLDCROP)
    except Exception:
        return None


def build(workers: int, months: Optional[List[str]] = None, limit: int = 0) -> None:
    skip = _in_progress()
    if months is None:
        months = sorted(os.path.basename(d) for d in glob.glob(os.path.join(WORLDCROP, "2*"))
                        if os.path.exists(os.path.join(d, "crops.jsonl")))
    for m in months:
        if m == skip and not limit:
            print(f"{m}: ingest in progress, skipped", flush=True)
            continue
        t0 = time.time()
        n = build_month(m, workers, limit)
        if n == -1:
            print(f"{m}: up to date", flush=True)
        else:
            print(f"{m}: {n} rows in {time.time() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- curated caches

def _cache_worker(args):
    name, idx, block_ids = args
    rows, feats = [], []
    for i, b in zip(idx, block_ids):
        b = np.asarray(b)
        row, f = describe(b.astype(np.uint16), None)
        row["_i"] = int(i)
        rows.append(row)
        feats.append(f)
    return rows, np.stack(feats).astype(np.float16) if feats else np.zeros((0, 295), np.float16)


def _cache_items(name: str, blob) -> List[dict]:
    """Per-build metadata of a cache: ``<name>_manifest.json`` ``items``, or, for a GrabCraft
    cache (``gc_small_<dim>``: no items, a ``<name>_meta.json`` keyed by url and a ``urls``
    array in the npz), the same fields rebuilt from those, with the url as ``source_path``
    exactly as the pooled caches (all_32, houses_32) record a GrabCraft build."""
    man = json.load(open(os.path.join(CACHE_DIR, f"{name}_manifest.json")))
    if "items" in man:
        return man["items"]
    meta_p = os.path.join(CACHE_DIR, f"{name}_meta.json")
    if not (os.path.exists(meta_p) and "urls" in blob.files):
        raise KeyError(f"{name}_manifest.json has no items and {name} is not a GrabCraft cache (meta + urls)")
    meta = json.load(open(meta_p))
    items = []
    for u in blob["urls"]:
        m = meta.get(str(u)) or {}
        items.append({"source_path": str(u), "corpus": "grabcraft", "category": m.get("subtitle", ""),
                      "title": m.get("title", ""), "url": ""})
    return items


def build_cache(name: str, workers: int, aux: bool = False) -> int:
    npz = os.path.join(CACHE_DIR, f"{name}.npz")
    blob = np.load(npz, allow_pickle=True)
    man = _cache_items(name, blob)
    block_ids = blob["block_ids"]
    n = len(block_ids)
    chunk = 200
    jobs = [(name, list(range(i, min(n, i + chunk))), [block_ids[j] for j in range(i, min(n, i + chunk))])
            for i in range(0, n, chunk)]
    rows, feats = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r, F in ex.map(_cache_worker, jobs):
            for row in r:
                it = man[row["_i"]]
                rows.append({
                    "id": f"cache:{name}:{row['_i']}", "source": it.get("corpus", name),
                    "path": f"data/minecraft/cache/{name}.npz#{row['_i']}", "month": "",
                    "world_id": it.get("source_path", "") or f"{name}:{row['_i']}",
                    "title": it.get("title", ""), "author": "", "url": it.get("url", "") or it.get("source_path", ""),
                    "dx": row["dx"], "dy": row["dy"], "dz": row["dz"], "n_blocks": row["n_blocks"],
                    "fill": row["fill"], "n_ids": row["n_ids"], "p_built": np.nan,
                    "cut": False, "oversize": False, "route": it.get("category", ""),
                    "canon_hash": row["canon_hash"], "sig": row["sig"],
                })
            feats.append(F)
    F = np.concatenate(feats)
    pr = _clf().predict_proba(F.astype(np.float32))[:, 1]
    for row, p in zip(rows, pr):
        row["p_built"] = float(p)
    _write(f"curated_{name}", rows, F, MANIFEST_AUX_DIR if aux else MANIFEST_DIR)
    return len(rows)


# --------------------------------------------------------------------------- read / summary

def load(sources: Optional[List[str]] = None, columns: Optional[List[str]] = None,
         aux: Optional[List[str]] = None):
    """Concatenate every manifest table into one pandas DataFrame, plus the auxiliary
    caches named in ``aux`` (``manifest_aux/curated_<name>.parquet``)."""
    import pandas as pd
    import pyarrow.parquet as pq
    frames = []
    for p in sorted(glob.glob(os.path.join(MANIFEST_DIR, "*.parquet"))):
        if p.endswith("_sample.parquet"):
            continue
        base = os.path.basename(p)
        if sources and not any(base.startswith(s) for s in sources):
            continue
        frames.append(pq.read_table(p, columns=columns).to_pandas())
    for name in aux or []:
        p = os.path.join(MANIFEST_AUX_DIR, f"curated_{name}.parquet")
        if not os.path.exists(p):
            raise FileNotFoundError(f"no auxiliary manifest {p}; run python -m blockgen.corpus.manifest "
                                    f"curated --aux --caches {name}")
        frames.append(pq.read_table(p, columns=columns).to_pandas())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns or COLUMNS)


def summary() -> None:
    df = load(columns=["source", "month", "n_blocks", "p_built", "cut", "dx", "dy", "dz"])
    print(f"rows {len(df)}")
    print(df.groupby("source").size().to_string())
    if len(df):
        print("p_built>=0.5:", int((df.p_built >= 0.5).sum()), " cut:", int(df.cut.sum()),
              " fits32:", int((df[["dx", "dy", "dz"]].max(axis=1) <= 32).sum()))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--workers", type=int, default=6)
    b.add_argument("--months", nargs="*"); b.add_argument("--limit", type=int, default=0)
    c = sub.add_parser("curated"); c.add_argument("--workers", type=int, default=6)
    # houses_32 is the cache sft-houses32 and eval-holdout index; leaving it out empties both tiers
    c.add_argument("--caches", nargs="*", default=["all_32", "houses_32", "houses_48"])
    c.add_argument("--aux", action="store_true",
                   help="write to manifest_aux/: read only by tiers that name the cache in aux:")
    sub.add_parser("summary")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.workers, a.months, a.limit)
    elif a.cmd == "curated":
        for n in a.caches:
            t0 = time.time()
            print(f"{n}: {build_cache(n, a.workers, aux=a.aux)} rows in {time.time() - t0:.0f}s", flush=True)
    else:
        summary()


if __name__ == "__main__":
    main()
