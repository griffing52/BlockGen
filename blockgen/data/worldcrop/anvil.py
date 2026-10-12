"""Minimal, dependency-light readers for Minecraft Java world saves.

Covers the three on-disk chunk encodings a 2009-2018 corpus contains:

* **McRegion** (``region/*.mcr``, Beta 1.3 - 1.1): ``Level.Blocks`` is a
  32768-byte array in XZY order (``x*2048 + z*128 + y``), 128 blocks tall.
* **Anvil, numeric ids** (``region/*.mca``, 1.2 - 1.12): ``Level.Sections[]``
  with ``Blocks`` (4096 bytes, YZX), optional ``Add`` nibbles (high id bits) and
  ``Data`` nibbles.
* **Anvil, palette** (1.13+): per-section ``Palette`` + packed ``BlockStates``
  (1.13-1.17, under ``Level``) or ``block_states.palette`` / ``.data`` (1.18+, at
  the chunk root). Block-state names are mapped onto the legacy ``(id, data)``
  vocabulary through :mod:`blockgen.utils.block_remap`.

Every chunk comes back as a :class:`Chunk` holding only its non-empty 16^3
sections as ``(ids uint16, data uint8)`` arrays in **XYZ** order, the same
layout as :class:`blockgen.utils.data.Structure`.

Only reading is implemented. No Amulet (non-free), no anvil-parser.
"""
from __future__ import annotations

import gzip
import io
import os
import zlib
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

try:  # nbtlib is an existing project dependency (used by utils/schem.py)
    import nbtlib
except ImportError as e:  # pragma: no cover
    raise ImportError("blockgen.data.worldcrop needs nbtlib") from e

SECTION = 16
CHUNK_H_LEGACY = 256


# --------------------------------------------------------------------------- #
# NBT helpers
# --------------------------------------------------------------------------- #
def _root(tag):
    """nbtlib wraps the unnamed root compound under key ''."""
    if isinstance(tag, nbtlib.File):
        return tag[""] if "" in tag else tag
    return tag


def parse_nbt(data: bytes):
    return _root(nbtlib.File.parse(io.BytesIO(data)))


def load_nbt_file(path: str):
    """Load a (possibly gzipped) NBT file such as level.dat or data/*.dat."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return parse_nbt(raw)


def _nibbles(arr: np.ndarray) -> np.ndarray:
    """Expand an n-byte nibble array into 2n uint8 values (low nibble first)."""
    a = np.asarray(arr, dtype=np.uint8)
    out = np.empty(a.size * 2, dtype=np.uint8)
    out[0::2] = a & 0x0F
    out[1::2] = a >> 4
    return out


def _yzx_to_xyz(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a.reshape(SECTION, SECTION, SECTION).transpose(2, 0, 1))


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    cx: int
    cz: int
    sections: Dict[int, Tuple[np.ndarray, np.ndarray]]  # sy -> (ids[x,y,z] u16, data u8)
    inhabited_time: int = -1
    n_tile_entities: int = 0
    n_entities: int = 0
    data_version: int = 0
    fmt: str = "anvil"  # mcregion | anvil | palette
    unknown_states: int = 0  # palette worlds: voxels whose state had no legacy mapping

    @property
    def n_blocks(self) -> int:
        return int(sum((ids != 0).sum() for ids, _ in self.sections.values()))

    def min_sy(self) -> int:
        return min(self.sections) if self.sections else 0

    def max_sy(self) -> int:
        return max(self.sections) if self.sections else -1


@dataclass
class LevelInfo:
    name: str = ""
    generator: str = ""
    generator_options: str = ""
    seed: Optional[int] = None
    version_id: int = 0  # Data.version (19132 mcregion, 19133 anvil)
    data_version: int = 0  # Data.DataVersion (1.9+)
    version_name: str = ""  # Data.Version.Name (1.9+)
    map_features: Optional[bool] = None
    last_played: int = 0
    game_type: Optional[int] = None
    extra: Dict[str, str] = field(default_factory=dict)

    @property
    def is_flat(self) -> bool:
        g = self.generator.lower()
        return g in ("flat", "minecraft:flat")


def read_level_dat(path: str) -> LevelInfo:
    tag = load_nbt_file(path)
    if not isinstance(tag, nbtlib.Compound):
        # Bedrock/Pocket Edition level.dat is little-endian with a header; nbtlib reads garbage
        raise ValueError("level.dat is not a Java-edition NBT compound (Bedrock/PE world?)")
    d = tag["Data"] if "Data" in tag else tag
    if not isinstance(d, nbtlib.Compound):
        raise ValueError("level.dat has no Data compound")
    info = LevelInfo()
    info.name = str(d.get("LevelName", ""))
    info.generator = str(d.get("generatorName", ""))
    info.generator_options = str(d.get("generatorOptions", ""))
    if "RandomSeed" in d:
        info.seed = int(d["RandomSeed"])
    elif "WorldGenSettings" in d and "seed" in d["WorldGenSettings"]:
        info.seed = int(d["WorldGenSettings"]["seed"])
        # 1.16+: generator lives under dimensions.minecraft:overworld.generator.type
        try:
            gen = d["WorldGenSettings"]["dimensions"]["minecraft:overworld"]["generator"]
            info.generator = str(gen.get("type", ""))
        except Exception:
            pass
    info.version_id = int(d.get("version", 0))
    info.data_version = int(d.get("DataVersion", 0))
    if "Version" in d:
        try:
            info.version_name = str(d["Version"].get("Name", ""))
        except Exception:
            pass
    if "MapFeatures" in d:
        info.map_features = bool(int(d["MapFeatures"]))
    info.last_played = int(d.get("LastPlayed", 0))
    if "GameType" in d:
        info.game_type = int(d["GameType"])
    return info


# --------------------------------------------------------------------------- #
# Region files
# --------------------------------------------------------------------------- #
def iter_region_blobs(path: str) -> Iterator[Tuple[int, int, bytes]]:
    """Yield (local_cx, local_cz, decompressed NBT bytes) for every stored chunk."""
    with open(path, "rb") as f:
        buf = f.read()
    if len(buf) < 8192:
        return
    for i in range(1024):
        off = int.from_bytes(buf[i * 4 : i * 4 + 3], "big")
        if off == 0:
            continue
        start = off * 4096
        if start + 5 > len(buf):
            continue
        ln = int.from_bytes(buf[start : start + 4], "big")
        comp = buf[start + 4]
        raw = buf[start + 5 : start + 4 + ln]
        try:
            if comp == 2:
                data = zlib.decompress(raw)
            elif comp == 1:
                data = gzip.decompress(raw)
            elif comp == 3:
                data = raw
            else:
                continue
        except Exception:
            continue
        yield i & 31, i >> 5, data


def region_coords(path: str) -> Tuple[int, int]:
    base = os.path.basename(path)
    _, rx, rz, _ = base.split(".", 3)
    return int(rx), int(rz)


def _decode_mcregion(lvl) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    blocks = np.asarray(lvl["Blocks"], dtype=np.uint8)  # XZY, 16*16*128
    if blocks.size != 32768:
        return {}
    data = _nibbles(np.asarray(lvl["Data"], dtype=np.uint8))[: blocks.size]
    b = blocks.reshape(16, 16, 128).transpose(0, 2, 1)  # x,z,y -> x,y,z
    d = data.reshape(16, 16, 128).transpose(0, 2, 1)
    out = {}
    for sy in range(8):
        ids = b[:, sy * 16 : (sy + 1) * 16, :]
        if not ids.any():
            continue
        out[sy] = (np.ascontiguousarray(ids, dtype=np.uint16),
                   np.ascontiguousarray(d[:, sy * 16 : (sy + 1) * 16, :], dtype=np.uint8))
    return out


def _decode_numeric_sections(sections) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    out = {}
    for s in sections:
        if "Blocks" not in s:
            continue
        sy = int(s["Y"])
        blocks = np.asarray(s["Blocks"], dtype=np.uint8).astype(np.uint16)
        if blocks.size != 4096:
            continue
        if "Add" in s:
            add = _nibbles(np.asarray(s["Add"], dtype=np.uint8))[:4096].astype(np.uint16)
            blocks = blocks | (add << 8)
        if not blocks.any():
            continue
        data = _nibbles(np.asarray(s["Data"], dtype=np.uint8))[:4096] if "Data" in s else np.zeros(4096, np.uint8)
        out[sy] = (_yzx_to_xyz(blocks), _yzx_to_xyz(data))
    return out


# ---- palette format (1.13+) ----------------------------------------------- #
_remap_cache: Dict[str, Tuple[int, int]] = {}


def _state_to_legacy(name: str, props: Dict[str, str]) -> Tuple[int, int]:
    """Map a 1.13+ block state onto legacy (id, data); (0, 0) for air/unknown."""
    key = name + ("[" + ",".join(f"{k}={v}" for k, v in sorted(props.items())) + "]" if props else "")
    hit = _remap_cache.get(key)
    if hit is not None:
        return hit
    try:
        from blockgen.utils.block_remap import remap_name
        r = remap_name(key)
        val = (int(r[0]), int(r[1])) if r is not None else (-1, 0)
    except Exception:
        val = (-1, 0)
    _remap_cache[key] = val
    return val


def _unpack_states(longs: np.ndarray, bits: int, n: int, padded: bool) -> np.ndarray:
    """Unpack n `bits`-wide indices from an int64 array.

    padded=False: 1.13-1.15 (values straddle long boundaries).
    padded=True : 1.16+ (each long holds floor(64/bits) whole values).
    """
    u = np.asarray(longs, dtype=np.int64).astype(np.uint64)
    if padded:
        per = 64 // bits
        idx = np.arange(n)
        li = idx // per
        sh = ((idx % per) * bits).astype(np.uint64)
        li = np.minimum(li, u.size - 1)
        return ((u[li] >> sh) & np.uint64((1 << bits) - 1)).astype(np.int64)
    # bitstream
    bitsarr = np.unpackbits(u.view(np.uint8).reshape(-1, 8)[:, ::-1], axis=1, bitorder="big").reshape(-1)
    # above gives per-long big-endian bits; we need little-endian bit order within the stream
    bitsarr = np.unpackbits(u.view(np.uint8), bitorder="little")
    bitsarr = bitsarr[: n * bits].reshape(n, bits)
    weights = (np.uint64(1) << np.arange(bits, dtype=np.uint64))
    return (bitsarr.astype(np.uint64) * weights).sum(axis=1).astype(np.int64)


def _decode_palette_sections(sections, data_version: int):
    out = {}
    unknown = 0
    for s in sections:
        if "block_states" in s:  # 1.18+
            bs = s["block_states"]
            palette = bs.get("palette", [])
            longs = bs.get("data", None)
        elif "Palette" in s:  # 1.13-1.17
            palette = s["Palette"]
            longs = s.get("BlockStates", None)
        else:
            continue
        sy = int(s["Y"])
        if len(palette) == 0:
            continue
        pal_ids = np.zeros(len(palette), np.int64)
        pal_data = np.zeros(len(palette), np.uint8)
        for i, p in enumerate(palette):
            name = str(p["Name"])
            props = {str(k): str(v) for k, v in p["Properties"].items()} if "Properties" in p else {}
            bid, bdat = _state_to_legacy(name, props)
            pal_ids[i] = bid
            pal_data[i] = bdat
        if longs is None or len(palette) == 1:
            idx = np.zeros(4096, np.int64)
        else:
            bits = max(4, (len(palette) - 1).bit_length())
            padded = data_version >= 2529  # 20w17a
            idx = _unpack_states(np.asarray(longs), bits, 4096, padded)
            idx = np.clip(idx, 0, len(palette) - 1)
        ids = pal_ids[idx]
        unknown += int((ids < 0).sum())
        ids = np.where(ids < 0, 0, ids).astype(np.uint16)
        if not ids.any():
            continue
        out[sy] = (_yzx_to_xyz(ids), _yzx_to_xyz(pal_data[idx]))
    return out, unknown


def decode_chunk(data: bytes) -> Optional[Chunk]:
    tag = parse_nbt(data)
    data_version = int(tag.get("DataVersion", 0))
    if "Level" in tag:
        lvl = tag["Level"]
        cx, cz = int(lvl["xPos"]), int(lvl["zPos"])
        if "Blocks" in lvl:  # McRegion
            secs = _decode_mcregion(lvl)
            fmt = "mcregion"
            unknown = 0
        else:
            sections = lvl.get("Sections", [])
            if sections and "Palette" in sections[0] or any("Palette" in s for s in sections):
                secs, unknown = _decode_palette_sections(sections, data_version)
                fmt = "palette"
            else:
                secs = _decode_numeric_sections(sections)
                fmt = "anvil"
                unknown = 0
        ch = Chunk(cx, cz, secs, fmt=fmt, data_version=data_version, unknown_states=unknown)
        ch.inhabited_time = int(lvl.get("InhabitedTime", -1))
        ch.n_tile_entities = len(lvl.get("TileEntities", []))
        ch.n_entities = len(lvl.get("Entities", []))
        return ch
    if "sections" in tag:  # 1.18+
        cx, cz = int(tag["xPos"]), int(tag["zPos"])
        secs, unknown = _decode_palette_sections(tag["sections"], data_version)
        ch = Chunk(cx, cz, secs, fmt="palette", data_version=data_version, unknown_states=unknown)
        ch.inhabited_time = int(tag.get("InhabitedTime", -1))
        ch.n_tile_entities = len(tag.get("block_entities", []))
        return ch
    return None


def iter_region_chunks(path: str) -> Iterator[Chunk]:
    for _, _, blob in iter_region_blobs(path):
        try:
            ch = decode_chunk(blob)
        except Exception:
            ch = None
        if ch is not None and ch.sections:
            yield ch


def find_region_files(world_dir: str, dimension: str = "overworld") -> List[str]:
    """Region files for a dimension; overworld = <world>/region, nether = DIM-1, end = DIM1."""
    sub = {"overworld": "region", "nether": os.path.join("DIM-1", "region"),
           "end": os.path.join("DIM1", "region")}[dimension]
    d = os.path.join(world_dir, sub)
    if not os.path.isdir(d):
        return []
    files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".mca")]
    if not files:  # fall back to McRegion
        files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".mcr")]
    return sorted(files)


def find_world_dirs(root: str) -> List[str]:
    """Every directory under root that holds a level.dat (an extracted archive may hold several)."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "level.dat" in filenames:
            out.append(dirpath)
            dirnames[:] = []  # don't descend into a world
    return sorted(out)


# --------------------------------------------------------------------------- #
# Vanilla structure bounding boxes (pre-1.13 live in data/<Name>.dat)
# --------------------------------------------------------------------------- #
STRUCTURE_DAT_NAMES = ("Village", "Mineshaft", "Stronghold", "Temple", "Fortress",
                       "Monument", "EndCity", "Mansion")


def read_structure_boxes(world_dir: str) -> List[Tuple[str, Tuple[int, int, int, int, int, int]]]:
    """Return [(structure_name, (minX,minY,minZ,maxX,maxY,maxZ)), ...] from data/*.dat.

    Uses the per-start BB *and* each child's BB (the start BB can be loose)."""
    boxes = []
    ddir = os.path.join(world_dir, "data")
    if not os.path.isdir(ddir):
        return boxes
    for fn in os.listdir(ddir):
        stem = fn[:-4]
        if not fn.endswith(".dat") or stem not in STRUCTURE_DAT_NAMES:
            continue
        try:
            tag = load_nbt_file(os.path.join(ddir, fn))
            feats = tag["data"]["Features"]
        except Exception:
            continue
        for start in feats.values():
            try:
                bb = [int(v) for v in start["BB"]]
                if len(bb) == 6:
                    boxes.append((stem, tuple(bb)))
                for ch in start.get("Children", []):
                    bb = [int(v) for v in ch["BB"]]
                    if len(bb) == 6:
                        boxes.append((stem, tuple(bb)))
            except Exception:
                continue
    return boxes
