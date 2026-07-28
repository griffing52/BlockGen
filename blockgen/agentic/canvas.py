"""The mutable voxel buffer the agentic DSL writes into.

A :class:`Canvas` is a bounded ``(size_x, size_y, size_z)`` volume of legacy
``(block_id, block_data)`` pairs with ``y`` up, matching
:class:`blockgen.utils.data.Structure` indexing exactly — ``to_structure()`` is a
copy, not a conversion. Commands never touch the arrays directly; they call the
handful of primitives here (``set_voxel``, ``fill_box``, ``set_coords``, the
clipboard ops), which is what keeps bounds handling, change counting and the
"replace-only" filter in one place.

Two behaviors matter for the agent loop:

* **Writes outside the canvas are clipped, not fatal.** An LLM that plans a 40-tall
  tower in a 32-tall canvas still produces a build; the clipped-voxel count comes
  back in the execution report so the next round can fix the plan. Silently failing
  would be worse than either alternative.
* **Every write returns the number of voxels it changed.** Per-command change
  counts are the cheapest possible signal that a program did nothing useful (a
  ``fill`` that changes 0 voxels is almost always a coordinate bug), and they are
  what the repair prompt quotes back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from blockgen.agentic.blockstate import AIR, IdData
from blockgen.utils.data import Structure

Coord = Tuple[int, int, int]

DEFAULT_SIZE = (64, 64, 64)


@dataclass
class Clipboard:
    """A copied region: legacy id/data arrays plus the origin they were cut from."""

    block_ids: np.ndarray
    block_data: np.ndarray
    origin: Coord

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.block_ids.shape)  # type: ignore[return-value]


class Canvas:
    """Bounded voxel buffer in ``(x, y, z)`` order with ``y`` up."""

    def __init__(self, size: Sequence[int] = DEFAULT_SIZE, air_block_id: int = 0):
        size = tuple(int(v) for v in size)
        if len(size) != 3 or any(v <= 0 for v in size):
            raise ValueError(f"canvas size must be three positive ints, got {size!r}")
        self.size: Tuple[int, int, int] = size  # type: ignore[assignment]
        self.block_ids = np.zeros(size, dtype=np.int32)
        self.block_data = np.zeros(size, dtype=np.int32)
        self.air_block_id = int(air_block_id)
        self.clipboard: Optional[Clipboard] = None
        self.clipped_writes = 0  # voxels dropped for falling outside the canvas

    # --- bounds -----------------------------------------------------------
    def in_bounds(self, x: int, y: int, z: int) -> bool:
        sx, sy, sz = self.size
        return 0 <= x < sx and 0 <= y < sy and 0 <= z < sz

    def _clip_span(self, lo: int, hi: int, axis: int) -> Tuple[int, int, int]:
        """Clip an inclusive ``[lo, hi]`` span to the canvas; returns
        ``(lo, hi_exclusive, n_dropped)``."""
        if hi < lo:
            lo, hi = hi, lo
        limit = self.size[axis]
        dropped = max(0, -lo) + max(0, hi - (limit - 1))
        return max(lo, 0), min(hi, limit - 1) + 1, dropped

    # --- primitive writes -------------------------------------------------
    def set_voxel(self, x: int, y: int, z: int, block: IdData) -> int:
        """Write one voxel; returns 1 if it landed in bounds (0 otherwise)."""
        if not self.in_bounds(x, y, z):
            self.clipped_writes += 1
            return 0
        self.block_ids[x, y, z] = block[0]
        self.block_data[x, y, z] = block[1]
        return 1

    def fill_box(self, lo: Coord, hi: Coord, block: IdData, *,
                 only_replace: Optional[IdData] = None,
                 match_id_only: bool = False) -> int:
        """Fill the inclusive cuboid ``lo..hi``; returns voxels written.

        ``only_replace`` restricts the write to voxels currently holding that
        pair (``match_id_only`` ignores the data value, so ``replace stone …``
        catches every stone variant). This is the primitive behind ``fill``,
        ``clear`` and ``replace``.
        """
        x0, x1, _ = self._clip_span(lo[0], hi[0], 0)
        y0, y1, _ = self._clip_span(lo[1], hi[1], 1)
        z0, z1, _ = self._clip_span(lo[2], hi[2], 2)
        requested = 1
        for axis in range(3):
            requested *= abs(int(hi[axis]) - int(lo[axis])) + 1
        self.clipped_writes += requested - max(0, x1 - x0) * max(0, y1 - y0) * max(0, z1 - z0)
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            return 0
        ids = self.block_ids[x0:x1, y0:y1, z0:z1]
        data = self.block_data[x0:x1, y0:y1, z0:z1]
        if only_replace is None:
            n = ids.size
            ids[...] = block[0]
            data[...] = block[1]
            return int(n)
        mask = ids == only_replace[0]
        if not match_id_only:
            mask &= data == only_replace[1]
        n = int(mask.sum())
        ids[mask] = block[0]
        data[mask] = block[1]
        return n

    def set_coords(self, coords: Iterable[Coord], block: IdData) -> int:
        """Write an arbitrary point set (shape primitives use this)."""
        arr = np.asarray(list(coords), dtype=np.int64)
        if arr.size == 0:
            return 0
        arr = arr.reshape(-1, 3)
        sx, sy, sz = self.size
        keep = ((arr[:, 0] >= 0) & (arr[:, 0] < sx) & (arr[:, 1] >= 0)
                & (arr[:, 1] < sy) & (arr[:, 2] >= 0) & (arr[:, 2] < sz))
        self.clipped_writes += int((~keep).sum())
        arr = arr[keep]
        if arr.size == 0:
            return 0
        self.block_ids[arr[:, 0], arr[:, 1], arr[:, 2]] = block[0]
        self.block_data[arr[:, 0], arr[:, 1], arr[:, 2]] = block[1]
        return int(arr.shape[0])

    # --- clipboard --------------------------------------------------------
    def copy_region(self, lo: Coord, hi: Coord) -> Clipboard:
        x0, x1, _ = self._clip_span(lo[0], hi[0], 0)
        y0, y1, _ = self._clip_span(lo[1], hi[1], 1)
        z0, z1, _ = self._clip_span(lo[2], hi[2], 2)
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            raise ValueError("copy region is empty after clipping to the canvas")
        self.clipboard = Clipboard(
            block_ids=self.block_ids[x0:x1, y0:y1, z0:z1].copy(),
            block_data=self.block_data[x0:x1, y0:y1, z0:z1].copy(),
            origin=(x0, y0, z0),
        )
        return self.clipboard

    def paste(self, at: Coord, *, rotate: int = 0, mirror: str = "none",
              include_air: bool = False) -> int:
        """Paste the clipboard with its min corner at ``at``.

        ``rotate`` is degrees about the vertical axis (0/90/180/270) and
        ``mirror`` flips the copied block over ``x`` or ``z`` — together they
        cover the symmetric-wing / repeated-window patterns that make a program
        much shorter than the equivalent block list.
        """
        if self.clipboard is None:
            raise ValueError("nothing in the clipboard; run `copy` first")
        ids = self.clipboard.block_ids
        data = self.clipboard.block_data
        if mirror in ("x", "X"):
            ids, data = ids[::-1], data[::-1]
        elif mirror in ("z", "Z"):
            ids, data = ids[:, :, ::-1], data[:, :, ::-1]
        elif mirror not in ("none", "", None):
            raise ValueError(f"mirror must be none|x|z, got '{mirror}'")
        turns = (int(rotate) // 90) % 4
        if int(rotate) % 90:
            raise ValueError(f"rotate must be a multiple of 90, got {rotate}")
        for _ in range(turns):  # rotate in the x-z plane, y untouched
            ids = np.rot90(ids, k=1, axes=(0, 2))
            data = np.rot90(data, k=1, axes=(0, 2))
        written = 0
        occ = np.ones(ids.shape, dtype=bool) if include_air else (ids != self.air_block_id)
        coords = np.argwhere(occ)
        if coords.size == 0:
            return 0
        target = coords + np.asarray(at, dtype=np.int64)
        sx, sy, sz = self.size
        keep = ((target[:, 0] >= 0) & (target[:, 0] < sx) & (target[:, 1] >= 0)
                & (target[:, 1] < sy) & (target[:, 2] >= 0) & (target[:, 2] < sz))
        self.clipped_writes += int((~keep).sum())
        src, dst = coords[keep], target[keep]
        if src.size == 0:
            return 0
        self.block_ids[dst[:, 0], dst[:, 1], dst[:, 2]] = ids[src[:, 0], src[:, 1], src[:, 2]]
        self.block_data[dst[:, 0], dst[:, 1], dst[:, 2]] = data[src[:, 0], src[:, 1], src[:, 2]]
        written += int(src.shape[0])
        return written

    # --- readback ---------------------------------------------------------
    @property
    def occupied_mask(self) -> np.ndarray:
        return self.block_ids != self.air_block_id

    def block_count(self) -> int:
        return int(self.occupied_mask.sum())

    def bbox(self) -> Optional[Tuple[Coord, Coord]]:
        occ = np.argwhere(self.occupied_mask)
        if occ.size == 0:
            return None
        lo = tuple(int(v) for v in occ.min(axis=0))
        hi = tuple(int(v) for v in occ.max(axis=0))
        return lo, hi  # type: ignore[return-value]

    def palette_counts(self) -> List[Tuple[str, int]]:
        """``[(\"id:data\", count), …]`` most-used first — the built palette."""
        occ = self.occupied_mask
        if not occ.any():
            return []
        keys = (self.block_ids[occ].astype(np.int64) << 8) | (
            self.block_data[occ].astype(np.int64) & 0xFF)
        uniq, counts = np.unique(keys, return_counts=True)
        order = np.argsort(-counts)
        return [(f"{int(k) >> 8}:{int(k) & 0xFF}", int(c))
                for k, c in zip(uniq[order], counts[order])]

    def to_structure(self, crop: bool = True) -> Structure:
        """Copy the canvas into a :class:`Structure` (cropped to non-air by default)."""
        s = Structure(block_ids=self.block_ids.copy(),
                      block_data=self.block_data.copy(),
                      air_block_id=self.air_block_id,
                      metadata={"source": "agentic"})
        return s.crop_to_non_air() if crop else s

    def clone(self) -> "Canvas":
        other = Canvas(self.size, self.air_block_id)
        other.block_ids = self.block_ids.copy()
        other.block_data = self.block_data.copy()
        other.clipped_writes = self.clipped_writes
        return other

    def clear(self) -> None:
        self.block_ids[...] = self.air_block_id
        self.block_data[...] = 0
        self.clipped_writes = 0


def structure_to_canvas(structure: Structure, size: Optional[Sequence[int]] = None,
                        at: Coord = (0, 0, 0)) -> Canvas:
    """Load an existing build into a canvas — the entry point for *editing*
    (in-fill, extension, style transfer) rather than generating from scratch."""
    s = structure.crop_to_non_air()
    if size is None:
        size = tuple(int(a + b) for a, b in zip(s.shape, at))
    canvas = Canvas(size, air_block_id=s.air_block_id)
    occ = np.argwhere(s.occupied_mask)
    for x, y, z in occ.tolist():
        canvas.set_voxel(x + at[0], y + at[1], z + at[2],
                         (int(s.block_ids[x, y, z]), int(s.block_data[x, y, z])))
    return canvas


__all__ = ["Canvas", "Clipboard", "structure_to_canvas", "DEFAULT_SIZE", "AIR"]
