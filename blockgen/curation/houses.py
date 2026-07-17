"""Build one clean, deduped, cross-corpus HOUSE dataset.

Pools house examples from all three corpora, remaps everything onto the shared
legacy ``(block_id, block_data)`` vocabulary, applies feature-based quality
filters, drops duplicates, and saves a cache + manifest:

* **GrabCraft** (labeled scrape): every ``*-houses`` subcategory.
* **3D-Craft** (CraftAssist): all entries are crowd-built houses by
  construction; the quality filter weeds out abandoned/partial builds.
* **text2mc** (PlanetMinecraft dump): builds whose page URL/tags look
  house-like. These are world *cuts* that include terrain, so a
  ground-stripping pass removes bottom layers that are mostly
  dirt/stone/bedrock before size filtering. Tokens are remapped to legacy ids
  via :mod:`blockgen.utils.block_remap`.

CLI::

    python -m blockgen.curation.houses --max-dim 32        # -> data/cache/houses_32.npz

Load back with :func:`load_house_structures`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from blockgen.curation.curate import Curator, compute_features
from blockgen.data.build_cache import DEFAULT_CACHE_DIR
from blockgen.utils import corpora
from blockgen.utils.block_remap import build_token_lut, remap_token_array
from blockgen.utils.data import Structure

REPO = Path(__file__).resolve().parents[2]

# GrabCraft subcategories that are houses (see data/grabcraft/raw/).
GRABCRAFT_HOUSE_CATS = {
    "houses", "medieval-houses", "modern-houses", "wooden-houses",
    "stone-houses", "brick-houses", "quartz-houses", "starter-houses",
    "survival-houses", "tree-houses",
}

# text2mc PAGE_URL / TAGS keywords that indicate a house-like build.
HOUSE_KEYWORDS = ("house", "cottage", "cabin", "villa", "manor", "mansion",
                  "farmhouse", "hut", "home", "residence")

# Legacy ids counted as "terrain" for ground stripping (world-cut corpora).
_TERRAIN_IDS = frozenset({1, 2, 3, 7, 8, 9, 12, 13, 14, 15, 16, 21, 24, 31, 32,
                          37, 38, 56, 73, 78, 82, 110, 121, 129})


# --- text2mc ---------------------------------------------------------------
def _strip_ground(s: Structure, terrain_frac: float = 0.60,
                  fill_frac: float = 0.45) -> Tuple[Structure, int]:
    """Drop consecutive bottom layers that look like terrain, re-crop.

    A layer is "ground" when it is mostly filled across the footprint AND its
    occupied voxels are mostly terrain blocks (dirt/stone/bedrock/water/...).
    Returns (stripped structure, number of layers removed).
    """
    c = s.crop_to_non_air()
    ids = c.block_ids
    sx, sy, sz = c.shape
    cut = 0
    for y in range(sy - 1):  # never strip the whole build
        layer = ids[:, y, :]
        occ = layer != 0
        n_occ = int(occ.sum())
        if n_occ == 0:
            cut = y + 1
            continue
        terrain = float(np.isin(layer[occ], list(_TERRAIN_IDS)).mean())
        fill = n_occ / float(sx * sz)
        if terrain >= terrain_frac and fill >= fill_frac:
            cut = y + 1
        else:
            break
    if cut == 0:
        return c, 0
    out = Structure(block_ids=ids[:, cut:, :], block_data=c.block_data[:, cut:, :],
                    source_path=c.source_path, metadata=dict(c.metadata))
    return out.crop_to_non_air(), cut


def _text2mc_house_rows() -> Dict[str, dict]:
    """h5 filename -> {url, tags} for builds whose page looks house-like."""
    csv.field_size_limit(10_000_000)
    csv_path = REPO / "data" / "minecraft" / "text2mc" / "processed_build_dataframe.csv"
    index_path = REPO / "data" / "minecraft" / "text2mc_index.json"
    rows = list(csv.DictReader(open(csv_path)))
    index = json.loads(index_path.read_text())
    out: Dict[str, dict] = {}
    for fname in index:
        m = re.match(r"batch_\d+_(\d+)\.h5", fname)
        if not m:
            continue
        row = rows[int(m.group(1))]
        text = (row["PAGE_URL"] + " " + row["TAGS"]).lower()
        if any(k in text for k in HOUSE_KEYWORDS):
            out[fname] = {"url": row["PAGE_URL"], "tags": row["TAGS"]}
    return out


def load_text2mc_houses(max_dim: int = 32, min_blocks: int = 80,
                        strip_ground: bool = True, verbose: bool = True) -> List[Structure]:
    """House-tagged text2mc builds, remapped to legacy ids and ground-stripped."""
    import h5py
    house_rows = _text2mc_house_rows()
    index = json.loads((REPO / "data" / "minecraft" / "text2mc_index.json").read_text())
    files = {p.name: p for p in
             (REPO / "data" / "minecraft" / "text2mc" / "processed_builds").rglob("*.h5")}
    lut = build_token_lut()
    out: List[Structure] = []
    n_seen = n_stripped = 0
    for fname, meta in house_rows.items():
        info = index.get(fname)
        # Pre-crop prefilter: allow some slack, ground stripping shrinks height.
        if info is None or info["occ"] < min_blocks or max(info["shape"]) > max_dim + 24:
            continue
        path = files.get(fname)
        if path is None:
            continue
        n_seen += 1
        with h5py.File(path, "r") as h:
            arr = np.asarray(h[list(h.keys())[0]])
        ids, data = remap_token_array(arr, lut)
        s = Structure(block_ids=ids, block_data=data, source_path=str(path),
                      metadata={"corpus": "text2mc", "category": "house",
                                "url": meta["url"], "tags": meta["tags"]})
        if strip_ground:
            s, cut = _strip_ground(s)
            n_stripped += bool(cut)
        else:
            s = s.crop_to_non_air()
        if max(s.shape) > max_dim or int(s.occupied_mask.sum()) < min_blocks:
            continue
        out.append(s)
    if verbose:
        print(f"[text2mc] {len(out)} houses kept (of {n_seen} size-plausible, "
              f"{len(house_rows)} house-tagged; ground-stripped {n_stripped})", flush=True)
    return out


# --- text2mc raw .schem -----------------------------------------------------
def _text2mc_schem_house_rows() -> Dict[str, dict]:
    """schem filename -> {url, tags} for house-like builds (from PROCESSED_PATHS).

    The 28k raw ``.schem`` builds text2mc never converted to ``.h5`` are listed
    per source page in the ``PROCESSED_PATHS`` column (one page splits into
    several ``build_batch_*.schem`` shards). We keep shards whose page URL/TAGS
    match :data:`HOUSE_KEYWORDS`, same rule as the ``.h5`` path.
    """
    csv.field_size_limit(10_000_000)
    csv_path = REPO / "data" / "minecraft" / "text2mc" / "processed_build_dataframe.csv"
    out: Dict[str, dict] = {}
    for row in csv.DictReader(open(csv_path)):
        text = (row["PAGE_URL"] + " " + row["TAGS"]).lower()
        if not any(k in text for k in HOUSE_KEYWORDS):
            continue
        for name in re.findall(r"build_batch_[\w]+\.schem", row.get("PROCESSED_PATHS") or ""):
            out[name] = {"url": row["PAGE_URL"], "tags": row["TAGS"]}
    return out


def load_text2mc_schem_houses(max_dim: int = 32, min_blocks: int = 80,
                              strip_ground: bool = True, verbose: bool = True) -> List[Structure]:
    """House-tagged text2mc raw ``.schem`` builds, remapped to legacy ids and ground-stripped.

    Complements :func:`load_text2mc_houses` (the ``.h5`` set). The IoU dedupe in
    :func:`build_house_dataset` drops any shard that duplicates an ``.h5`` build
    from the same source page, so pooling both is safe.
    """
    from blockgen.utils.schem import schem_to_legacy
    from blockgen.labeling.categorize import category_phrase
    house_rows = _text2mc_schem_house_rows()
    files = {p.name: p for p in
             (REPO / "data" / "minecraft" / "text2mc" / "processed_builds").rglob("*.schem")}
    out: List[Structure] = []
    n_seen = n_stripped = 0
    for fname, meta in house_rows.items():
        path = files.get(fname)
        if path is None:
            continue
        # A house-scale build (<= max_dim + slack) gzips well under 1.5MB; the
        # cap skips huge multi-shard world exports before the slow nbtlib parse.
        res = schem_to_legacy(path, max_bytes=1_500_000)
        if res is None:
            continue
        ids, data = res
        # Pre-crop prefilter: allow slack since ground stripping shrinks height.
        if int((ids != 0).sum()) < min_blocks or max(ids.shape) > max_dim + 24:
            continue
        n_seen += 1
        s = Structure(block_ids=ids, block_data=data, source_path=str(path),
                      metadata={"corpus": "text2mc_schem",
                                "category": category_phrase(meta["tags"], meta["url"]),
                                "url": meta["url"], "tags": meta["tags"]})
        if strip_ground:
            s, cut = _strip_ground(s)
            n_stripped += bool(cut)
        else:
            s = s.crop_to_non_air()
        if max(s.shape) > max_dim or int(s.occupied_mask.sum()) < min_blocks:
            continue
        out.append(s)
    if verbose:
        print(f"[text2mc_schem] {len(out)} houses kept (of {n_seen} size-plausible, "
              f"{len(house_rows)} house-tagged shards; ground-stripped {n_stripped})", flush=True)
    return out


# --- grabcraft ------------------------------------------------------------
def load_grabcraft_houses(max_dim: int = 32, cache_dir: str = DEFAULT_CACHE_DIR,
                          verbose: bool = True) -> List[Structure]:
    from blockgen.data.grabcraft_dataset import load_grabcraft_structures
    structures, meta = load_grabcraft_structures(max_dim=max_dim, cache_dir=cache_dir)
    out: List[Structure] = []
    for s in structures:
        m = meta.get(s.source_path or "", {})
        cat = m.get("subtitle", "")
        if cat in GRABCRAFT_HOUSE_CATS:
            s.metadata.update({"corpus": "grabcraft", "category": cat,
                               "title": m.get("title", "")})
            out.append(s.crop_to_non_air())
    if verbose:
        print(f"[grabcraft] {len(out)} houses (of {len(structures)} cached)", flush=True)
    return out


# --- 3d-craft ---------------------------------------------------------------
def load_3dcraft_houses(max_dim: int = 32, min_blocks: int = 80,
                        verbose: bool = True) -> List[Structure]:
    structs = corpora.load_3dcraft(min_blocks=min_blocks, max_dim=max_dim, verbose=verbose)
    return [s.crop_to_non_air() for s in structs]


# --- pooled build -----------------------------------------------------------
def enclosed_air_count(block_ids: np.ndarray) -> int:
    """Number of air voxels NOT reachable from the bbox boundary (interior rooms).

    Real houses enclose air; trees, vehicles, roofs-only fragments, and solid
    decorations don't. Vectorized 6-connected flood fill from all six faces.
    """
    air = block_ids == 0
    reach = np.zeros_like(air)
    for sl in (np.s_[0, :, :], np.s_[-1, :, :], np.s_[:, 0, :],
               np.s_[:, -1, :], np.s_[:, :, 0], np.s_[:, :, -1]):
        reach[sl] = air[sl]
    prev = -1
    while True:
        n = int(reach.sum())
        if n == prev:
            break
        prev = n
        grown = reach.copy()
        grown[1:, :, :] |= reach[:-1, :, :]
        grown[:-1, :, :] |= reach[1:, :, :]
        grown[:, 1:, :] |= reach[:, :-1, :]
        grown[:, :-1, :] |= reach[:, 1:, :]
        grown[:, :, 1:] |= reach[:, :, :-1]
        grown[:, :, :-1] |= reach[:, :, 1:]
        reach = grown & air
    return int(air.sum() - reach.sum())


def quality_filter(cur: Curator, *, min_blocks: int = 80, min_height: int = 4,
                   min_footprint: int = 16, min_density: float = 0.03,
                   max_density: float = 0.85, min_block_types: int = 3,
                   max_dominant_frac: float = 0.92,
                   min_largest_component_frac: float = 0.55,
                   min_enclosed_air: int = 0,
                   enclosed_air_corpora: Sequence[str] = ("3dcraft", "text2mc",
                                                          "text2mc_schem"),
                   ) -> Tuple[Curator, Counter]:
    """Flag structures failing house-quality rules; returns (passing view, drop reasons).

    The enclosed-air (interior rooms) gate applies only to *enclosed_air_corpora*:
    the crowdsourced pools where non-house junk (trees, vehicles, fragments)
    hides. GrabCraft is category-labeled ground truth, and ~17% of its real
    houses legitimately have open interiors, so it is exempt by default.
    """
    rules = (
        ("too_few_blocks", lambda r: r["n_blocks"] < min_blocks),
        ("too_flat", lambda r: r["height"] < min_height),
        ("footprint_too_small", lambda r: r["footprint"] < min_footprint),
        ("too_sparse", lambda r: r["density"] < min_density),
        ("solid_blob", lambda r: r["density"] > max_density),
        ("too_few_materials", lambda r: r["n_block_types"] < min_block_types),
        ("single_material_blob", lambda r: r["dominant_frac"] > max_dominant_frac),
        ("fragmented", lambda r: r["largest_component_frac"] < min_largest_component_frac),
    )
    reasons: Counter = Counter()
    keep: List[int] = []
    for i in cur.indices:
        r = cur.features[i]
        for name, bad in rules:
            if bad(r):
                reasons[name] += 1
                cur.mark_remove([i], reason=name)
                break
        else:
            # Enclosed-air check last: costliest, so only run on survivors.
            if (min_enclosed_air > 0
                    and cur.structures[i].metadata.get("corpus") in enclosed_air_corpora
                    and enclosed_air_count(
                        cur.structures[i].block_ids) < min_enclosed_air):
                reasons["no_interior"] += 1
                cur.mark_remove([i], reason="no_interior")
                continue
            keep.append(i)
    return cur._view(keep), reasons


def build_house_dataset(max_dim: int = 32, min_blocks: int = 80,
                        cache_dir: str = DEFAULT_CACHE_DIR,
                        include: Sequence[str] = ("grabcraft", "3dcraft", "text2mc",
                                                  "text2mc_schem"),
                        dedupe_iou: float = 0.95,
                        min_enclosed_air: int = 8) -> Tuple[List[Structure], dict]:
    """Pool -> quality filter -> dedupe. Returns (curated structures, report)."""
    pool: List[Structure] = []
    if "grabcraft" in include:
        pool += load_grabcraft_houses(max_dim=max_dim, cache_dir=cache_dir)
    if "3dcraft" in include:
        pool += load_3dcraft_houses(max_dim=max_dim, min_blocks=min_blocks)
    if "text2mc" in include:
        pool += load_text2mc_houses(max_dim=max_dim, min_blocks=min_blocks)
    if "text2mc_schem" in include:
        pool += load_text2mc_schem_houses(max_dim=max_dim, min_blocks=min_blocks)
    by_corpus = Counter(s.metadata.get("corpus", "?") for s in pool)
    print(f"pooled {len(pool)} candidates: {dict(by_corpus)}", flush=True)

    cur = Curator.from_structures(pool, max_dim=max_dim)
    cur.decisions.clear()  # fresh, reproducible pass — no stale global decisions
    passing, drop_reasons = quality_filter(cur, min_blocks=min_blocks,
                                           min_enclosed_air=min_enclosed_air)
    print(f"quality filter: kept {len(passing)}/{len(pool)}; drops: "
          f"{dict(drop_reasons.most_common())}", flush=True)

    n_dups = passing.dedupe_keep_variants(iou_threshold=dedupe_iou,
                                          reason="exact-duplicate")
    kept = cur.apply()
    report = {
        "pooled": len(pool), "by_corpus": dict(by_corpus),
        "quality_drops": dict(drop_reasons), "exact_duplicates": n_dups,
        "final": len(kept),
        "final_by_corpus": dict(Counter(s.metadata.get("corpus", "?") for s in kept)),
    }
    print(f"deduped {n_dups} exact copies -> final {len(kept)}", flush=True)
    return kept, report


# --- cache io ---------------------------------------------------------------
def house_cache_path(max_dim: int, cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    return os.path.join(cache_dir, f"houses_{max_dim}.npz")


def save_house_cache(structures: Sequence[Structure], max_dim: int,
                     cache_dir: str = DEFAULT_CACHE_DIR,
                     report: Optional[dict] = None) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    n = len(structures)
    ids = np.empty(n, dtype=object)
    data = np.empty(n, dtype=object)
    for i, s in enumerate(structures):
        ids[i] = s.block_ids.astype(np.int16)
        data[i] = s.block_data.astype(np.int16)
    path = house_cache_path(max_dim, cache_dir)
    np.savez_compressed(
        path, block_ids=ids, block_data=data,
        sources=np.array([s.source_path or "" for s in structures]),
        corpus=np.array([s.metadata.get("corpus", "") for s in structures]),
    )
    manifest = {
        "max_dim": max_dim, "count": n, "report": report or {},
        "items": [
            {"source_path": s.source_path or "", **{k: s.metadata.get(k, "")
             for k in ("corpus", "category", "title", "url")},
             "dims": list(s.shape), "n_blocks": int(s.occupied_mask.sum())}
            for s in structures
        ],
    }
    with open(path.replace(".npz", "_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"saved {n} houses -> {path}", flush=True)
    return path


def load_structures_from_cache(npz_path: str) -> Tuple[List[Structure], dict]:
    """Load any structure cache written by :func:`save_house_cache`.

    Corpus-agnostic: works for the house caches *and* any future whole-corpus
    cache saved in the same ``.npz`` + ``_manifest.json`` layout. Row ``i`` aligns
    to manifest item ``i`` and render/label id ``h{i:05d}``.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Structure cache not found at {npz_path}.")
    blob = np.load(npz_path, allow_pickle=True)
    manifest = json.loads(Path(npz_path.replace(".npz", "_manifest.json")).read_text())
    structures = []
    for i in range(len(blob["block_ids"])):
        item = manifest["items"][i]
        structures.append(Structure(
            block_ids=blob["block_ids"][i], block_data=blob["block_data"][i],
            source_path=str(blob["sources"][i]),
            metadata={k: item.get(k, "") for k in ("corpus", "category", "title", "url")}))
    return structures, manifest


def load_house_structures(max_dim: int = 32, cache_dir: str = DEFAULT_CACHE_DIR
                          ) -> Tuple[List[Structure], dict]:
    """Load the curated house cache: (structures, manifest)."""
    path = house_cache_path(max_dim, cache_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"House cache not found at {path}. Build it with "
            f"`python -m blockgen.curation.houses --max-dim {max_dim}`.")
    return load_structures_from_cache(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the curated cross-corpus house dataset.")
    parser.add_argument("--max-dim", type=int, default=32)
    parser.add_argument("--min-blocks", type=int, default=80)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--include", nargs="+",
                        default=["grabcraft", "3dcraft", "text2mc", "text2mc_schem"],
                        choices=["grabcraft", "3dcraft", "text2mc", "text2mc_schem"])
    parser.add_argument("--dedupe-iou", type=float, default=0.95)
    parser.add_argument("--min-enclosed-air", type=int, default=8,
                        help="require >= this many interior air voxels (house-ness "
                             "gate; applies to 3dcraft/text2mc only)")
    args = parser.parse_args()
    kept, report = build_house_dataset(
        max_dim=args.max_dim, min_blocks=args.min_blocks, cache_dir=args.cache_dir,
        include=args.include, dedupe_iou=args.dedupe_iou,
        min_enclosed_air=args.min_enclosed_air)
    save_house_cache(kept, args.max_dim, args.cache_dir, report=report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
