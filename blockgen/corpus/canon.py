"""Canonical hashes and MinHash signatures for voxel builds.

A build is a dense XYZ ``ids`` array (0 = air) with an optional ``mask``. Two
builds are *exact duplicates* when they agree up to translation and one of the
eight horizontal symmetries (rotations about y and mirrors). They are *near
duplicates* when the Jaccard similarity of their voxel sets, in canonical pose,
is high; MinHash signatures estimate that cheaply and LSH banding finds the
candidate pairs without an all-pairs scan.
"""
from __future__ import annotations

import hashlib
from typing import Tuple

import numpy as np

N_HASH = 128                 # MinHash signature length
_RNG = np.random.default_rng(20261012)
_A = _RNG.integers(1, 2**63, size=N_HASH, dtype=np.uint64) | np.uint64(1)  # odd multipliers
_B = _RNG.integers(0, 2**63, size=N_HASH, dtype=np.uint64)


def tighten(ids: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Apply the mask and crop to the tight bounding box of the non-air voxels."""
    ids = np.asarray(ids)
    if mask is not None:
        ids = ids * np.asarray(mask).astype(ids.dtype)
    nz = np.nonzero(ids)
    if len(nz[0]) == 0:
        return np.zeros((1, 1, 1), dtype=ids.dtype)
    sl = tuple(slice(int(a.min()), int(a.max()) + 1) for a in nz)
    return np.ascontiguousarray(ids[sl])


def d4_variants(a: np.ndarray):
    """The eight horizontal symmetries of an XYZ array (axes 0 and 2)."""
    for k in range(4):
        r = np.rot90(a, k, axes=(0, 2))
        yield r
        yield r[::-1, :, :]


def canonical(ids: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Tight crop in the lexicographically smallest of its eight poses.

    The order key is (shape, bytes), so two builds that differ only by a
    rotation or mirror map to the same array.
    """
    t = tighten(ids, mask)
    best = None
    for v in d4_variants(t):
        v = np.ascontiguousarray(v)
        key = (v.shape, v.tobytes())
        if best is None or key < best[0]:
            best = (key, v)
    return best[1]


def canon_hash(ids: np.ndarray, mask: np.ndarray | None = None) -> str:
    c = canonical(ids, mask)
    h = hashlib.blake2b(digest_size=16)
    h.update(np.asarray(c.shape, dtype=np.int32).tobytes())
    h.update(c.astype(np.uint16).tobytes())
    return h.hexdigest()


def voxel_shingles(canon: np.ndarray) -> np.ndarray:
    """One 64-bit token per non-air voxel of a canonical array: (x, y, z, id)."""
    xs, ys, zs = np.nonzero(canon)
    v = canon[xs, ys, zs].astype(np.uint64)
    return ((xs.astype(np.uint64) << np.uint64(48)) | (ys.astype(np.uint64) << np.uint64(32))
            | (zs.astype(np.uint64) << np.uint64(16)) | v)


def minhash(shingles: np.ndarray) -> np.ndarray:
    """MinHash signature (uint32[N_HASH]) of a set of uint64 tokens.

    Uses the multiply-shift family h(x) = ((a*x + b) mod 2^64) >> 32 with odd a,
    vectorised in chunks so a 100k-voxel build stays under 100 MB.
    """
    if shingles.size == 0:
        return np.full(N_HASH, np.iinfo(np.uint32).max, dtype=np.uint32)
    x = (shingles.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15))  # pre-mix
    x ^= x >> np.uint64(29)
    out = np.full(N_HASH, np.iinfo(np.uint64).max, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for i in range(0, x.size, 8192):
            chunk = x[i:i + 8192, None]
            hv = (chunk * _A[None, :] + _B[None, :]) >> np.uint64(32)
            np.minimum(out, hv.min(axis=0), out=out)
    return out.astype(np.uint32)


def canon_and_sig(ids: np.ndarray, mask: np.ndarray | None = None) -> Tuple[str, np.ndarray, np.ndarray]:
    """Return (canon_hash, minhash signature, canonical array)."""
    c = canonical(ids, mask)
    h = hashlib.blake2b(digest_size=16)
    h.update(np.asarray(c.shape, dtype=np.int32).tobytes())
    h.update(c.astype(np.uint16).tobytes())
    return h.hexdigest(), minhash(voxel_shingles(c)), c


def jaccard_from_sigs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a == b))
