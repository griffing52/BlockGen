"""Build a whole-corpus (non-house-filtered) structure cache.

``build_house_dataset`` pools the four corpora but keeps only house-category /
house-keyword builds. For the full-corpus conditioned run (ideas.md / data-scaling
lever) we want *everything coherent*, so this mirrors that pipeline — pool → quality
filter → dedupe → cache — but drops the house filter:

* grabcraft: all cached builds (every subtitle, not just ``GRABCRAFT_HOUSE_CATS``)
* 3d-craft: all 2,586 CraftAssist builds (already all houses)
* text2mc ``.h5``: every indexed build (not just house-tagged), remapped + ground-stripped
* text2mc ``.schem``: every shard (the ~28k raw builds), decoded + ground-stripped

The VLM ``is_build`` coherence filter (labeling stage) removes the junk this lets in;
here we only apply the cheap geometric ``quality_filter`` + IoU dedupe. Saved as
``all_<max_dim>.npz`` + ``_manifest.json`` in the same layout as the house caches, so
every downstream stage (render_views ``--cache``, make_index, embed_conditions,
training) consumes it unchanged. Row ``i`` == manifest item ``i`` == id ``h{i:05d}``.

    python -m blockgen.curation.corpus --max-dim 32 --out data/minecraft/all_32.npz
    python -m blockgen.curation.corpus --max-dim 32 --limit-schem 200   # smoke
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np

from blockgen.curation.curate import Curator
from blockgen.curation.houses import (DEFAULT_CACHE_DIR, REPO, Structure, _strip_ground,
                                       load_3dcraft_houses, quality_filter,
                                       save_house_cache)
from blockgen.utils.block_remap import build_token_lut, remap_token_array


def _grabcraft_all(max_dim: int, cache_dir: str) -> List[Structure]:
    from blockgen.data.grabcraft_dataset import load_grabcraft_structures
    structs, meta = load_grabcraft_structures(max_dim=max_dim, cache_dir=cache_dir)
    out = []
    for s in structs:
        m = meta.get(s.source_path or "", {})
        s.metadata.update({"corpus": "grabcraft", "category": m.get("subtitle", ""),
                           "title": m.get("title", "")})
        out.append(s.crop_to_non_air())
    print(f"[grabcraft] {len(out)} (all categories)", flush=True)
    return out


def _text2mc_rows() -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """(h5_meta, schem_meta): file/shard name -> {url, tags} for ALL builds."""
    import csv
    csv.field_size_limit(10_000_000)
    rows = list(csv.DictReader(open(
        REPO / "data" / "minecraft" / "text2mc" / "processed_build_dataframe.csv")))
    index = json.loads((REPO / "data" / "minecraft" / "text2mc_index.json").read_text())
    h5_meta, schem_meta = {}, {}
    for fname in index:
        m = re.match(r"batch_\d+_(\d+)\.h5", fname)
        if m and int(m.group(1)) < len(rows):
            row = rows[int(m.group(1))]
            h5_meta[fname] = {"url": row["PAGE_URL"], "tags": row["TAGS"]}
    for row in rows:
        for name in re.findall(r"build_batch_[\w]+\.schem", row.get("PROCESSED_PATHS") or ""):
            schem_meta[name] = {"url": row["PAGE_URL"], "tags": row["TAGS"]}
    return h5_meta, schem_meta


def _category(tags: str, url: str) -> str:
    from blockgen.labeling.categorize import category_phrase
    return category_phrase(tags, url)


def _text2mc_h5_all(max_dim, min_blocks, h5_meta) -> List[Structure]:
    index = json.loads((REPO / "data" / "minecraft" / "text2mc_index.json").read_text())
    files = {p.name: p for p in
             (REPO / "data" / "minecraft" / "text2mc" / "processed_builds").rglob("*.h5")}
    lut = build_token_lut()
    out = []
    for fname, info in index.items():
        if info["occ"] < min_blocks or max(info["shape"]) > max_dim + 24:
            continue
        path = files.get(fname)
        if path is None:
            continue
        with h5py.File(path, "r") as h:
            arr = np.asarray(h[list(h.keys())[0]])
        ids, data = remap_token_array(arr, lut)
        meta = h5_meta.get(fname, {})
        s = Structure(block_ids=ids, block_data=data, source_path=str(path),
                      metadata={"corpus": "text2mc", "url": meta.get("url", ""),
                                "tags": meta.get("tags", ""),
                                "category": _category(meta.get("tags", ""), meta.get("url", ""))})
        s, _ = _strip_ground(s)
        if max(s.shape) <= max_dim and int(s.occupied_mask.sum()) >= min_blocks:
            out.append(s)
    print(f"[text2mc/h5] {len(out)}", flush=True)
    return out


def _text2mc_schem_all(max_dim, min_blocks, schem_meta, limit=None) -> List[Structure]:
    from blockgen.utils.schem import schem_to_legacy
    files = {p.name: p for p in
             (REPO / "data" / "minecraft" / "text2mc" / "processed_builds").rglob("*.schem")}
    items = list(schem_meta.items())
    if limit:
        items = items[:limit]
    out, seen, t0 = [], 0, time.time()
    for k, (fname, meta) in enumerate(items):
        path = files.get(fname)
        if path is None:
            continue
        res = schem_to_legacy(path, max_bytes=1_500_000)   # skips huge world exports fast
        if res is None:
            continue
        ids, data = res
        if int((ids != 0).sum()) < min_blocks or max(ids.shape) > max_dim + 24:
            continue
        seen += 1
        s = Structure(block_ids=ids, block_data=data, source_path=str(path),
                      metadata={"corpus": "text2mc_schem", "url": meta["url"],
                                "tags": meta["tags"], "category": _category(meta["tags"], meta["url"])})
        s, _ = _strip_ground(s)
        if max(s.shape) <= max_dim and int(s.occupied_mask.sum()) >= min_blocks:
            out.append(s)
        if k % 2000 == 0 and k:
            print(f"[text2mc/schem] {k}/{len(items)} scanned, {len(out)} kept "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"[text2mc/schem] {len(out)} kept of {seen} decodable", flush=True)
    return out


def build_full_corpus(max_dim: int = 32, min_blocks: int = 80,
                      cache_dir: str = DEFAULT_CACHE_DIR,
                      include: Sequence[str] = ("grabcraft", "3dcraft", "text2mc",
                                                "text2mc_schem"),
                      dedupe_iou: float = 0.95, min_enclosed_air: int = 0,
                      limit_schem: int = None) -> Tuple[List[Structure], dict]:
    """Pool ALL corpora (no house filter) -> quality filter -> dedupe."""
    pool: List[Structure] = []
    h5_meta, schem_meta = _text2mc_rows()
    if "grabcraft" in include:
        pool += _grabcraft_all(max_dim, cache_dir)
    if "3dcraft" in include:
        dc = load_3dcraft_houses(max_dim=max_dim, min_blocks=min_blocks)  # all houses
        pool += dc
    if "text2mc" in include:
        pool += _text2mc_h5_all(max_dim, min_blocks, h5_meta)
    if "text2mc_schem" in include:
        pool += _text2mc_schem_all(max_dim, min_blocks, schem_meta, limit=limit_schem)

    by_corpus = Counter(s.metadata.get("corpus", "?") for s in pool)
    print(f"pooled {len(pool)}: {dict(by_corpus)}", flush=True)

    cur = Curator.from_structures(pool, max_dim=max_dim)
    cur.decisions.clear()
    # min_enclosed_air=0 by default: the corpus has non-enclosed builds (statues,
    # bridges) that the VLM is_build filter, not geometry, should judge.
    passing, drops = quality_filter(cur, min_blocks=min_blocks,
                                    min_enclosed_air=min_enclosed_air)
    print(f"quality filter: kept {len(passing)}/{len(pool)}; drops {dict(drops.most_common())}",
          flush=True)
    n_dups = passing.dedupe_keep_variants(iou_threshold=dedupe_iou, reason="exact-duplicate")
    kept = cur.apply()
    report = {"pooled": len(pool), "by_corpus": dict(by_corpus),
              "quality_drops": dict(drops), "exact_duplicates": n_dups, "final": len(kept),
              "final_by_corpus": dict(Counter(s.metadata.get("corpus", "?") for s in kept))}
    print(f"deduped {n_dups} -> final {len(kept)}: {report['final_by_corpus']}", flush=True)
    return kept, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-dim", type=int, default=32)
    ap.add_argument("--min-blocks", type=int, default=80)
    ap.add_argument("--name", default="all")
    ap.add_argument("--limit-schem", type=int, default=None, help="cap schem scan (smoke)")
    ap.add_argument("--include", nargs="+",
                    default=["grabcraft", "3dcraft", "text2mc", "text2mc_schem"])
    args = ap.parse_args()

    kept, report = build_full_corpus(max_dim=args.max_dim, min_blocks=args.min_blocks,
                                     include=args.include, limit_schem=args.limit_schem)
    path = save_house_cache(kept, args.max_dim, report=report, name=args.name)
    print(f"saved {len(kept)} -> {path}", flush=True)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
