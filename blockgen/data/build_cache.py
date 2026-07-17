"""Build a cached, filtered subset of small Minecraft structures.

Parsing 10.9k gzipped-NBT schematics on every epoch is slow, and ~184 of them
are corrupt or wrong-format (see ``safe_load_schematic``). This module scans
``data/raw/*.schematic`` once, keeps only small, non-trivial structures (cropped
to their occupied bounding box), and writes a single ``.npz`` cache plus a JSON
manifest. All downstream training/eval loads from the cache, so it never touches
NBT or the bad files again.

Run as a script::

    python -m blockgen.data.build_cache --max-dim 24 --min-blocks 8

The cache stores ragged per-structure arrays (structures have different shapes)
using ``np.savez`` with object arrays.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from blockgen.utils.data import Structure
from blockgen.utils.data_loader import safe_load_schematic


# Anchor data paths to the repo root (parent of the ``blockgen`` package) so they
# resolve no matter the current working directory — notably, Jupyter runs with the
# notebook's own directory as CWD, which otherwise breaks plain ``data/raw`` paths.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = str(_REPO_ROOT / "data" / "minecraft" / "raw")
DEFAULT_CACHE_DIR = str(_REPO_ROOT / "data" / "minecraft" / "cache")


@dataclass
class CacheConfig:
    raw_dir: str = DEFAULT_RAW_DIR
    cache_dir: str = DEFAULT_CACHE_DIR
    max_dim: int = 24
    min_blocks: int = 8
    max_blocks: int = 4096
    limit: Optional[int] = None  # cap number of scanned files (for quick tests)


def cache_path(config: CacheConfig) -> str:
    return os.path.join(config.cache_dir, f"small_{config.max_dim}.npz")


def _normalize_block_ids(block_ids: np.ndarray) -> np.ndarray:
    # Legacy block ids are unsigned bytes (0-255) but are read as int8, so ids
    # >=128 come back negative. Map them back to the unsigned 0-255 range.
    return (block_ids.astype(np.int32) & 0xFF).astype(np.int32)


def build_cache(config: CacheConfig) -> dict:
    """Scan the raw directory and write the filtered cache. Returns a summary."""
    paths = sorted(glob.glob(os.path.join(config.raw_dir, "*.schematic")))
    if config.limit is not None:
        paths = paths[: config.limit]

    kept_ids: List[np.ndarray] = []
    kept_data: List[np.ndarray] = []
    kept_shapes: List[tuple] = []
    kept_paths: List[str] = []

    n_bad = 0
    n_too_big = 0
    n_too_small = 0

    for path in paths:
        schematic = safe_load_schematic(path)
        if schematic is None:
            n_bad += 1
            continue
        try:
            structure = Structure.from_schematic(schematic, source_path=path)
            structure = structure.crop_to_non_air()
        except Exception:
            n_bad += 1
            continue

        if max(structure.shape) > config.max_dim:
            n_too_big += 1
            continue

        n_occ = int(structure.occupied_mask.sum())
        if n_occ < config.min_blocks or n_occ > config.max_blocks:
            n_too_small += 1
            continue

        kept_ids.append(_normalize_block_ids(structure.block_ids))
        kept_data.append(structure.block_data.astype(np.int32))
        kept_shapes.append(tuple(structure.shape))
        kept_paths.append(path)

    os.makedirs(config.cache_dir, exist_ok=True)
    out_path = cache_path(config)
    np.savez(
        out_path,
        block_ids=np.array(kept_ids, dtype=object),
        block_data=np.array(kept_data, dtype=object),
        shapes=np.array(kept_shapes, dtype=np.int64) if kept_shapes else np.zeros((0, 3), np.int64),
        paths=np.array(kept_paths, dtype=object),
    )

    summary = {
        "cache_path": out_path,
        "scanned": len(paths),
        "kept": len(kept_ids),
        "skipped_bad": n_bad,
        "skipped_too_big": n_too_big,
        "skipped_too_small": n_too_small,
        "max_dim": config.max_dim,
        "min_blocks": config.min_blocks,
    }
    with open(os.path.join(config.cache_dir, f"small_{config.max_dim}_manifest.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_cached_structures(
    max_dim: int = 24, cache_dir: str = DEFAULT_CACHE_DIR
) -> List[Structure]:
    """Load the cached subset back into a list of ``Structure`` objects."""
    path = os.path.join(cache_dir, f"small_{max_dim}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cache not found at {path}. Run `python -m blockgen.data.build_cache "
            f"--max-dim {max_dim}` first."
        )
    blob = np.load(path, allow_pickle=True)
    ids = blob["block_ids"]
    data = blob["block_data"]
    paths = blob["paths"]
    structures: List[Structure] = []
    for i in range(len(ids)):
        structures.append(
            Structure(
                block_ids=ids[i],
                block_data=data[i],
                source_path=str(paths[i]),
            )
        )
    return structures


def main() -> None:
    parser = argparse.ArgumentParser(description="Build small-structure cache.")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-dim", type=int, default=24)
    parser.add_argument("--min-blocks", type=int, default=8)
    parser.add_argument("--max-blocks", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = CacheConfig(
        raw_dir=args.raw_dir,
        cache_dir=args.cache_dir,
        max_dim=args.max_dim,
        min_blocks=args.min_blocks,
        max_blocks=args.max_blocks,
        limit=args.limit,
    )
    summary = build_cache(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
