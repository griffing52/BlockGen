"""Read the minecraft-schematics-dataset tfrecords and build a *labeled* cache.

Unlike ``data/raw/*.schematic`` (a separate, drifted download whose numeric
filenames don't map to any metadata), the dataset's ``schematics/*.tfrecords``
store each schematic's bytes **next to its source url**, in the same order as
``schematics.json`` / ``fullSchematics.json``. That url is the join key to rich
metadata: title, category, tags, description, and popularity (views / downloads /
diamonds / favorites).

This module parses the tfrecords with **no TensorFlow dependency** (the TFRecord
framing and the ``tf.train.Example`` protobuf are both simple enough to read by
hand), decodes each schematic into a :class:`Structure`, keeps the small ones,
and writes a single ``.npz`` plus a per-url metadata JSON. Downstream curation
loads structures that already carry their url in ``source_path`` and a metadata
table keyed by that url.

Run as a script::

    python -m blockgen.data.tfrecord_dataset --max-dim 24 --min-blocks 8
"""

from __future__ import annotations

import argparse
import glob
import gzip
import io
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from blockgen.utils.data import Structure


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = str(_REPO_ROOT / "data" / "more" / "minecraft-schematics-dataset")
DEFAULT_CACHE_DIR = str(_REPO_ROOT / "data" / "cache")

# Metadata fields we keep (the full records carry much more we don't need).
_META_FIELDS = ("title", "subtitle", "tags", "description", "user", "date",
                "views", "downloads", "diamondCount", "favorites", "comments")


# --- TFRecord framing + minimal protobuf parser ----------------------------
def _read_tfrecords(path: str) -> Iterator[bytes]:
    """Yield each record payload (a serialized tf.train.Example) from a tfrecord."""
    with open(path, "rb") as f:
        data = f.read()
    i, n = 0, len(data)
    while i < n:
        length = struct.unpack_from("<Q", data, i)[0]
        i += 8 + 4  # uint64 length + uint32 length-CRC
        payload = data[i:i + length]
        i += length + 4  # payload + uint32 data-CRC
        yield payload


def _varint(b: bytes, i: int) -> Tuple[int, int]:
    shift = val = 0
    while True:
        byte = b[i]; i += 1
        val |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return val, i
        shift += 7


def _fields(b: bytes) -> Iterator[Tuple[int, int, object]]:
    """Walk a protobuf message, yielding (field_number, wire_type, value)."""
    i, n = 0, len(b)
    while i < n:
        key, i = _varint(b, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i); yield fn, wt, v
        elif wt == 2:
            ln, i = _varint(b, i); yield fn, wt, b[i:i + ln]; i += ln
        elif wt == 5:
            yield fn, wt, b[i:i + 4]; i += 4
        elif wt == 1:
            yield fn, wt, b[i:i + 8]; i += 8
        else:
            raise ValueError(f"bad wire type {wt}")


def _parse_example(payload: bytes) -> Dict[str, bytes]:
    """tf.train.Example -> {feature_name: first bytes value}."""
    out: Dict[str, bytes] = {}
    for fn, wt, val in _fields(payload):
        if fn != 1 or wt != 2:  # Features features = 1
            continue
        for ffn, fwt, fval in _fields(val):
            if ffn != 1 or fwt != 2:  # map<string,Feature> entry
                continue
            key = feat = None
            for efn, ewt, eval_ in _fields(fval):
                if efn == 1 and ewt == 2:
                    key = eval_.decode("utf-8", "replace")
                elif efn == 2 and ewt == 2:
                    feat = eval_  # Feature message
            if key is None or feat is None:
                continue
            for bfn, bwt, bval in _fields(feat):  # Feature -> BytesList bytes_list = 1
                if bfn == 1 and bwt == 2:
                    vals = [v for vfn, vwt, v in _fields(bval) if vfn == 1 and vwt == 2]
                    if vals:
                        out[key] = vals[0]
    return out


def iter_dataset_records(dataset_dir: str = DEFAULT_DATASET_DIR) -> Iterator[Tuple[str, bytes]]:
    """Yield (url, schematic_bytes) across all shards in record order."""
    shards = sorted(
        glob.glob(os.path.join(dataset_dir, "schematics", "*.tfrecords")),
        key=lambda p: int(p.split("_")[-1].split(".")[0]),
    )
    for shard in shards:
        for payload in _read_tfrecords(shard):
            ex = _parse_example(payload)
            sd = ex.get("schematicData")
            if sd is None:
                continue
            yield ex.get("url", b"").decode("utf-8", "replace"), sd


def _structure_from_bytes(schematic_bytes: bytes) -> Structure:
    """Decode raw .schematic bytes (gzipped NBT) into a Structure."""
    from nbtschematic import SchematicFile
    sf = SchematicFile.from_fileobj(io.BytesIO(gzip.decompress(schematic_bytes)))
    return Structure.from_schematic(sf)


# --- metadata --------------------------------------------------------------
def load_url_metadata(dataset_dir: str = DEFAULT_DATASET_DIR) -> Dict[str, dict]:
    """url -> trimmed metadata record, from the richest available JSON."""
    for name in ("fullSchematics.json", "schematicsWithFinalUrl.json", "schematics.json"):
        path = os.path.join(dataset_dir, name)
        if os.path.exists(path):
            records = json.load(open(path))
            recs = records.values() if isinstance(records, dict) else records
            out: Dict[str, dict] = {}
            for r in recs:
                url = r.get("url")
                if url:
                    out[url] = {k: r[k] for k in _META_FIELDS if k in r}
            return out
    return {}


# --- cache build / load ----------------------------------------------------
@dataclass
class TFCacheConfig:
    dataset_dir: str = DEFAULT_DATASET_DIR
    cache_dir: str = DEFAULT_CACHE_DIR
    max_dim: int = 24
    min_blocks: int = 8
    max_blocks: int = 4096
    limit: Optional[int] = None  # cap records scanned (quick tests)


def cache_path(config: TFCacheConfig) -> str:
    return os.path.join(config.cache_dir, f"tf_small_{config.max_dim}.npz")


def meta_path(config: TFCacheConfig) -> str:
    return os.path.join(config.cache_dir, f"tf_small_{config.max_dim}_meta.json")


def _normalize_block_ids(block_ids: np.ndarray) -> np.ndarray:
    # Legacy ids are unsigned bytes read as int8; map >=128 back to 0-255.
    return (block_ids.astype(np.int32) & 0xFF).astype(np.int32)


def build_labeled_cache(config: TFCacheConfig, log_every: int = 2000) -> dict:
    """Scan the tfrecords, keep small structures, write cache + metadata. Returns a summary."""
    url_meta = load_url_metadata(config.dataset_dir)

    kept_ids: List[np.ndarray] = []
    kept_data: List[np.ndarray] = []
    kept_shapes: List[tuple] = []
    kept_urls: List[str] = []
    out_meta: Dict[str, dict] = {}

    n_scanned = n_bad = n_too_big = n_too_small = 0
    for url, sd in iter_dataset_records(config.dataset_dir):
        n_scanned += 1
        if config.limit is not None and n_scanned > config.limit:
            n_scanned -= 1
            break
        if log_every and n_scanned % log_every == 0:
            print(f"  scanned={n_scanned} kept={len(kept_ids)} bad={n_bad} "
                  f"big={n_too_big} small={n_too_small}", flush=True)
        try:
            structure = _structure_from_bytes(sd).crop_to_non_air()
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
        kept_urls.append(url)
        if url in url_meta:
            out_meta[url] = url_meta[url]

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
    with open(os.path.join(config.cache_dir, f"tf_small_{config.max_dim}_manifest.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_labeled_structures(
    max_dim: int = 24, cache_dir: str = DEFAULT_CACHE_DIR
) -> Tuple[List[Structure], Dict[str, dict]]:
    """Load the labeled cache: (structures with url in source_path, url->metadata)."""
    path = os.path.join(cache_dir, f"tf_small_{max_dim}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Labeled cache not found at {path}. Run "
            f"`python -m blockgen.data.tfrecord_dataset --max-dim {max_dim}` first."
        )
    blob = np.load(path, allow_pickle=True)
    ids, data, urls = blob["block_ids"], blob["block_data"], blob["urls"]
    structures = [
        Structure(block_ids=ids[i], block_data=data[i], source_path=str(urls[i]))
        for i in range(len(ids))
    ]
    meta_file = os.path.join(cache_dir, f"tf_small_{max_dim}_meta.json")
    meta = json.load(open(meta_file)) if os.path.exists(meta_file) else {}
    return structures, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build labeled small-structure cache from tfrecords.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-dim", type=int, default=24)
    parser.add_argument("--min-blocks", type=int, default=8)
    parser.add_argument("--max-blocks", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    config = TFCacheConfig(
        dataset_dir=args.dataset_dir, cache_dir=args.cache_dir, max_dim=args.max_dim,
        min_blocks=args.min_blocks, max_blocks=args.max_blocks, limit=args.limit,
    )
    print(json.dumps(build_labeled_cache(config), indent=2))


if __name__ == "__main__":
    main()
