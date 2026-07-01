"""Turn scraped GrabCraft raw JSON into a labeled Structure cache.

:mod:`blockgen.data.grabcraft_scraper` writes one JSON per build under
``data/grabcraft/raw/<category>/<slug>.json`` holding page metadata plus the raw
``renderObject`` block dict. This module is the offline half: decode those blocks
into :class:`Structure` volumes, keep the small ones, and write a single ``.npz`` +
per-url metadata JSON — byte-for-byte the same layout as
:mod:`blockgen.data.tfrecord_dataset` (``block_ids``/``block_data``/``shapes``/``urls``
object arrays), so the existing :class:`~blockgen.curation.Curator` loads it unchanged.

The key decode step is exact, not fuzzy: every GrabCraft block carries a ``texture``
field ``"<legacy_id>_<data>.png"`` — the classic numeric id + data value — which is
precisely the representation :class:`Structure` and ``STANDARD_VOCAB`` already use.
A name-based reverse lookup against ``STANDARD_VOCAB`` is only a fallback for the rare
block whose texture is blank/malformed.

Run as a script::

    python -m blockgen.data.grabcraft_dataset --max-dim 24 --min-blocks 8
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from blockgen.data.tfrecord_dataset import DEFAULT_CACHE_DIR
from blockgen.tokenizers.standard_vocab import STANDARD_VOCAB
from blockgen.utils.data import Structure

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = str(_REPO_ROOT / "data" / "grabcraft" / "raw")

_TEXTURE = re.compile(r"^(\d+)_(\d+)\.png$")

# Metadata fields carried onto each url record (matches the tf cache's schema so
# blockgen.curation.attach_metadata consumes it directly; absent fields default there).
_META_FIELDS = ("title", "subtitle", "tags", "views")


# --- block-name fallback ---------------------------------------------------
def _build_name_lookup() -> Dict[str, Tuple[int, int]]:
    """Reverse STANDARD_VOCAB: normalized display name -> (block_id, block_data).

    Used only when a block's ``texture`` is missing. GrabCraft names line up closely
    with the vocab's display names (e.g. "Oak Wood Plank(minecraft:planks)"), modulo
    trailing state parens and pluralization, which :func:`_normalize_name` strips.
    """
    lookup: Dict[str, Tuple[int, int]] = {}
    for token, value in STANDARD_VOCAB.items():
        display = value.split("(")[0]
        norm = _normalize_name(display)
        if not norm:
            continue
        if ":" in token:
            bid, bdata = token.split(":")
        else:
            bid, bdata = token, "0"
        lookup.setdefault(norm, (int(bid), int(bdata)))
    return lookup


def _normalize_name(name: str) -> str:
    """Lowercase, drop trailing state parens and pluralized 'planks'/'stairs' noise."""
    name = name.split("(")[0].strip().lower()
    name = name.replace("planks", "plank").replace("slabs", "slab")
    return re.sub(r"\s+", " ", name)


_NAME_LOOKUP = _build_name_lookup()


def _block_id_data(record: dict) -> Optional[Tuple[int, int]]:
    """Recover legacy (id, data) from a render-object block record."""
    tex = record.get("texture") or ""
    m = _TEXTURE.match(tex)
    if m:
        return int(m.group(1)), int(m.group(2))
    hit = _NAME_LOOKUP.get(_normalize_name(record.get("name", "")))
    return hit  # may be None -> caller skips (leaves air)


# --- decode a raw artifact into a Structure --------------------------------
def structure_from_artifact(art: dict) -> Optional[Structure]:
    """Build a cropped :class:`Structure` from one raw build artifact.

    Coordinates in the render object are 1-indexed (x=width, y=height, z=depth); we
    place blocks into a dense (W, H, D) grid at ``[x-1, y-1, z-1]`` to match
    :class:`Structure`'s XYZ convention, then crop to the occupied bounding box.
    """
    blocks = art.get("blocks") or {}
    coords: List[Tuple[int, int, int, int, int]] = []  # (x, y, z, id, data)
    mx = my = mz = 0
    for _xk, yz in blocks.items():
        for _yk, zdict in yz.items():
            for _n, rec in zdict.items():
                idd = _block_id_data(rec)
                if idd is None or idd[0] == 0:  # unknown or air
                    continue
                x, y, z = int(rec["x"]), int(rec["y"]), int(rec["z"])
                coords.append((x, y, z, idd[0], idd[1]))
                mx, my, mz = max(mx, x), max(my, y), max(mz, z)
    if not coords:
        return None

    block_ids = np.zeros((mx, my, mz), dtype=np.int32)
    block_data = np.zeros((mx, my, mz), dtype=np.int32)
    for x, y, z, bid, bdata in coords:
        block_ids[x - 1, y - 1, z - 1] = bid
        block_data[x - 1, y - 1, z - 1] = bdata
    return Structure(block_ids=block_ids, block_data=block_data,
                     source_path=art.get("url", "")).crop_to_non_air()


def _artifact_metadata(art: dict) -> dict:
    return {
        "title": art.get("title", ""),
        "subtitle": art.get("category", ""),   # curation maps subtitle -> category
        "tags": art.get("tags", []),
        "views": art.get("views", 0),
    }


def iter_artifacts(raw_dir: str = DEFAULT_RAW_DIR):
    """Yield parsed raw-artifact dicts across all category subfolders."""
    for path in sorted(glob.glob(os.path.join(raw_dir, "**", "*.json"), recursive=True)):
        try:
            with open(path) as f:
                yield json.load(f)
        except (json.JSONDecodeError, OSError):
            continue


# --- cache build / load ----------------------------------------------------
@dataclass
class GCCacheConfig:
    raw_dir: str = DEFAULT_RAW_DIR
    cache_dir: str = DEFAULT_CACHE_DIR
    max_dim: int = 24
    min_blocks: int = 8
    max_blocks: int = 4096
    limit: Optional[int] = None


def cache_path(config: GCCacheConfig) -> str:
    return os.path.join(config.cache_dir, f"gc_small_{config.max_dim}.npz")


def meta_path(config: GCCacheConfig) -> str:
    return os.path.join(config.cache_dir, f"gc_small_{config.max_dim}_meta.json")


def build_cache(config: GCCacheConfig, log_every: int = 500) -> dict:
    """Decode raw artifacts, keep small structures, write cache + metadata."""
    kept_ids: List[np.ndarray] = []
    kept_data: List[np.ndarray] = []
    kept_shapes: List[tuple] = []
    kept_urls: List[str] = []
    out_meta: Dict[str, dict] = {}

    n_scanned = n_bad = n_too_big = n_too_small = 0
    for art in iter_artifacts(config.raw_dir):
        n_scanned += 1
        if config.limit is not None and n_scanned > config.limit:
            n_scanned -= 1
            break
        if log_every and n_scanned % log_every == 0:
            print(f"  scanned={n_scanned} kept={len(kept_ids)} bad={n_bad} "
                  f"big={n_too_big} small={n_too_small}", flush=True)
        try:
            structure = structure_from_artifact(art)
        except Exception:
            structure = None
        if structure is None:
            n_bad += 1
            continue
        if max(structure.shape) > config.max_dim:
            n_too_big += 1
            continue
        n_occ = int(structure.occupied_mask.sum())
        if n_occ < config.min_blocks or n_occ > config.max_blocks:
            n_too_small += 1
            continue
        url = art.get("url", "")
        kept_ids.append(structure.block_ids.astype(np.int32))
        kept_data.append(structure.block_data.astype(np.int32))
        kept_shapes.append(tuple(structure.shape))
        kept_urls.append(url)
        out_meta[url] = _artifact_metadata(art)

    os.makedirs(config.cache_dir, exist_ok=True)
    np.savez(
        cache_path(config),
        block_ids=np.array(kept_ids, dtype=object),
        block_data=np.array(kept_data, dtype=object),
        shapes=np.array(kept_shapes, dtype=np.int64) if kept_shapes else np.zeros((0, 3), np.int64),
        urls=np.array(kept_urls, dtype=object),
    )
    with open(meta_path(config), "w") as f:
        json.dump(out_meta, f)

    summary = {
        "cache_path": cache_path(config),
        "meta_path": meta_path(config),
        "scanned": n_scanned,
        "kept": len(kept_ids),
        "with_metadata": len(out_meta),
        "skipped_bad": n_bad,
        "skipped_too_big": n_too_big,
        "skipped_too_small": n_too_small,
        "max_dim": config.max_dim,
    }
    with open(os.path.join(config.cache_dir, f"gc_small_{config.max_dim}_manifest.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_grabcraft_structures(
    max_dim: int = 24, cache_dir: str = DEFAULT_CACHE_DIR
) -> Tuple[List[Structure], Dict[str, dict]]:
    """Load the GrabCraft cache: (structures with url in source_path, url->metadata)."""
    path = os.path.join(cache_dir, f"gc_small_{max_dim}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"GrabCraft cache not found at {path}. Scrape with "
            f"`python -m blockgen.data.grabcraft_scraper --category houses`, then build "
            f"with `python -m blockgen.data.grabcraft_dataset --max-dim {max_dim}`."
        )
    blob = np.load(path, allow_pickle=True)
    ids, data, urls = blob["block_ids"], blob["block_data"], blob["urls"]
    structures = [
        Structure(block_ids=ids[i], block_data=data[i], source_path=str(urls[i]))
        for i in range(len(ids))
    ]
    meta_file = os.path.join(cache_dir, f"gc_small_{max_dim}_meta.json")
    meta = json.load(open(meta_file)) if os.path.exists(meta_file) else {}
    return structures, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build labeled cache from scraped GrabCraft JSON.")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-dim", type=int, default=24)
    parser.add_argument("--min-blocks", type=int, default=8)
    parser.add_argument("--max-blocks", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    config = GCCacheConfig(
        raw_dir=args.raw_dir, cache_dir=args.cache_dir, max_dim=args.max_dim,
        min_blocks=args.min_blocks, max_blocks=args.max_blocks, limit=args.limit,
    )
    print(json.dumps(build_cache(config), indent=2))


if __name__ == "__main__":
    main()
