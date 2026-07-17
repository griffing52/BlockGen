"""Read Sponge-format WorldEdit ``.schem`` files (the text2mc raw dump).

text2mc shipped ~11k builds as processed ``.h5`` token arrays and ~28k more as
raw **Sponge schematics** it never converted (see the dataset ``README.txt``:
"if you can fix this process, then you will have about 40,000 builds"). Those
``.schem`` files are the modern WorldEdit/Sponge format — an NBT ``Palette``
(block-state string -> small int) plus a varint (LEB128) ``BlockData`` byte
array laid out in **YZX** order — which the legacy ``nbtschematic`` loader (classic
MCEdit ``(id, data)`` only) cannot read.

This module decodes them and, via
:func:`blockgen.utils.block_remap.remap_name`, lands each build in our shared
legacy ``(block_id, block_data)`` vocabulary — the same space every other corpus
and all our tokenizers/renderers use. Coverage is ~97-98% of palette entries by
name (far higher by voxel); the tail falls back to stone, family-level, matching
the fidelity contract in ``block_remap``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Sponge air block-states that mean "empty" (map to legacy id 0).
AIR_NAMES = {"minecraft:air", "minecraft:cave_air", "minecraft:void_air"}


def _decode_varints(raw: bytes) -> np.ndarray:
    """Decode a Sponge ``BlockData`` LEB128 varint byte stream to palette indices.

    Vectorized: the overwhelmingly common case (every index < 128) is a single
    ``.astype`` on the raw bytes. Multi-byte values are handled by grouping bytes
    on the continuation bit and folding each group's 7-bit payloads with the right
    shift — no Python-level per-byte loop, so 34MB builds decode in milliseconds.
    """
    b = np.frombuffer(raw, dtype=np.uint8)
    if b.size == 0:
        return np.zeros(0, dtype=np.int64)
    cont = (b & 0x80) != 0
    if not cont.any():  # fast path: every palette index is one byte
        return b.astype(np.int64)
    payload = (b & 0x7F).astype(np.int64)
    ends = ~cont  # a value's final byte has the high bit clear
    # grp[i] = index of the value byte i belongs to (# completed values before it).
    grp = np.empty(b.size, dtype=np.int64)
    grp[0] = 0
    np.cumsum(ends[:-1], out=grp[1:])
    nvals = int(grp[-1]) + 1
    first = np.searchsorted(grp, np.arange(nvals))  # first byte index of each value
    pos = np.arange(b.size) - first[grp]  # 0-based byte position within its value
    out = np.zeros(nvals, dtype=np.int64)
    np.add.at(out, grp, payload << (7 * pos))
    return out


def read_schem(path: Path | str
               ) -> Optional[Tuple[Dict[str, int], np.ndarray, Tuple[int, int, int]]]:
    """Parse a Sponge ``.schem`` -> ``(palette, index_grid[y, z, x], (W, H, L))``.

    Handles both the v2 layout (fields at the NBT root) and the v3 container
    (nested under a ``Schematic`` compound). Returns ``None`` for anything
    unreadable or malformed (truncated file, wrong format, block-count mismatch)
    so directory sweeps can skip instead of crashing.
    """
    import nbtlib

    try:
        nb = nbtlib.load(str(path))
    except Exception:  # noqa: BLE001 - truncated/corrupt/non-NBT downloads
        return None
    sch = nb["Schematic"] if "Schematic" in nb else nb
    try:
        width, height, length = int(sch["Width"]), int(sch["Height"]), int(sch["Length"])
        palette = {str(k): int(v) for k, v in sch["Palette"].items()}
        raw = np.asarray(sch["BlockData"], dtype=np.uint8).tobytes()
    except (KeyError, TypeError, ValueError):
        return None
    idx = _decode_varints(raw)
    if idx.size != width * height * length:
        return None  # block count disagrees with dims -> malformed
    return palette, idx.reshape(height, length, width), (width, height, length)


def _palette_lut(palette: Dict[str, int]) -> np.ndarray:
    """Palette index -> legacy ``(block_id, block_data)`` via :func:`remap_name`."""
    from blockgen.utils.block_remap import remap_name

    n = (max(palette.values()) + 1) if palette else 1
    lut = np.zeros((n, 2), dtype=np.int32)  # default air (0, 0)
    for name, i in palette.items():
        base = name if name.startswith("minecraft:") else "minecraft:" + name
        if base.split("[")[0] in AIR_NAMES:
            continue
        mapped = remap_name(name)
        lut[i] = mapped if mapped is not None else (1, 0)  # unknown -> stone
    return lut


def schem_to_legacy(path: Path | str, *, max_voxels: int = 8_000_000,
                    max_bytes: Optional[int] = None
                    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Read a Sponge ``.schem`` -> ``(block_ids, block_data)`` in **XYZ** legacy vocab.

    ``max_bytes`` skips oversized files *before* the (pure-Python, whole-file)
    ``nbtlib`` parse — the dump has multi-shard world exports up to 34MB that we'd
    only discard anyway, and reading them dominates runtime. A dense 48^3 build
    gzips to a few hundred KB, so a ~1-2MB cap loses nothing at house scale.
    ``max_voxels`` is the post-parse guard for anything that slips through. Returns
    ``None`` if unreadable, over the byte cap, or over the voxel cap.
    """
    if max_bytes is not None:
        try:
            if os.path.getsize(path) > max_bytes:
                return None
        except OSError:
            return None
    parsed = read_schem(path)
    if parsed is None:
        return None
    palette, grid_yzx, (width, height, length) = parsed
    if max_voxels and width * height * length > max_voxels:
        return None
    lut = _palette_lut(palette)
    mapped = lut[grid_yzx]  # (H, L, W, 2)
    # YZX -> XYZ, matching Structure.from_schematic's transpose(2, 0, 1).
    ids = np.ascontiguousarray(np.transpose(mapped[..., 0], (2, 0, 1))).astype(np.int32)
    data = np.ascontiguousarray(np.transpose(mapped[..., 1], (2, 0, 1))).astype(np.int32)
    return ids, data
