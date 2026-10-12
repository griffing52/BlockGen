"""Small structures with a known ground-truth graph.

A :class:`Scene` is a list of parts (one text line each) plus the adjacency matrix the
attention is scored against. Three sources:

* ``house_chunks``: small sub-volumes cropped from the curated house cache
  (``data/minecraft/cache/houses_32.npz``), graph = 6-neighbour face adjacency.
* ``synthetic_scenes``: a wall, a pillar, an L-shape, a tiny hut (same graph).
* ``lego_scenes``: random brick stacks, graph = stud connectivity (a brick is joined
  to every brick in the layer above or below whose footprint overlaps its own).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .adjacency import face_adjacency, stud_adjacency

REPO = Path(__file__).resolve().parents[3]
HOUSE_CACHE = REPO / "data" / "minecraft" / "cache" / "houses_32.npz"


@dataclass
class Scene:
    coords: np.ndarray            # [N, 3] int, (x, y, z)
    names: List[str]              # one label per part, e.g. "planks" or "brick_2x4"
    adjacency: np.ndarray         # [N, N] bool, symmetric, zero diagonal
    source: str = ""
    extra: List[str] = field(default_factory=list)   # optional per-part suffix (e.g. "rot90")

    def __len__(self) -> int:
        return len(self.names)

    def subset(self, order: Sequence[int]) -> "Scene":
        """The same scene with parts re-ordered (adjacency permuted to match)."""
        idx = np.asarray(order, dtype=int)
        return Scene(coords=self.coords[idx], names=[self.names[i] for i in idx],
                     adjacency=self.adjacency[np.ix_(idx, idx)], source=self.source,
                     extra=[self.extra[i] for i in idx] if self.extra else [])


def voxel_scene(coords: np.ndarray, names: Sequence[str], source: str = "") -> Scene:
    coords = np.asarray(coords, dtype=int)
    return Scene(coords=coords, names=list(names), adjacency=face_adjacency(coords),
                 source=source)


def _short_name(resource_location: str) -> str:
    return resource_location.split(":", 1)[-1]


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest 6-connected component of a boolean volume."""
    from scipy.ndimage import label
    lab, n = label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())[1:]
    return lab == (1 + int(np.argmax(sizes)))


def house_chunks(n: int, size: int = 5, min_blocks: int = 20, max_blocks: int = 60,
                 seed: int = 0, cache: Path = HOUSE_CACHE, max_tries: int = 200
                 ) -> List[Scene]:
    """Crop ``n`` random ``size``^3 windows; keep the largest face-connected component
    if it has ``min_blocks..max_blocks`` blocks (the relative format needs connectivity).

    Each chunk comes from a different house, so a train/test split over chunks is also
    a split over builds.
    """
    from blockgen.curation.houses import load_structures_from_cache
    from blockgen.utils.data import _resource_location_for

    structures, _ = load_structures_from_cache(str(cache))
    rng = np.random.default_rng(seed)
    scenes: List[Scene] = []
    for h in rng.permutation(len(structures)):
        if len(scenes) >= n:
            break
        s = structures[h]
        occ = s.occupied_mask
        if min(occ.shape) < size:
            continue
        for _ in range(max_tries):
            x, y, z = (int(rng.integers(0, d - size + 1)) for d in occ.shape)
            win = largest_component(occ[x:x + size, y:y + size, z:z + size])
            if min_blocks <= win.sum() <= max_blocks:
                local = np.argwhere(win)
                names = [_short_name(_resource_location_for(
                    int(s.block_ids[x + a, y + b, z + c]), int(s.block_data[x + a, y + b, z + c])))
                    for a, b, c in local]
                scenes.append(voxel_scene(local, names,
                                          source=f"{s.metadata.get('title', h)}@{x},{y},{z}"))
                break
    return scenes


def synthetic_scenes() -> List[Scene]:
    """Hand-made toy structures with obvious graphs."""
    out = []
    wall = [(x, y, 0) for x in range(5) for y in range(3)]
    out.append(voxel_scene(wall, ["cobblestone"] * len(wall), "wall_5x3"))
    pillar = [(0, y, 0) for y in range(6)]
    out.append(voxel_scene(pillar, ["log"] * 6, "pillar_6"))
    ell = [(x, 0, 0) for x in range(5)] + [(0, 0, z) for z in range(1, 5)]
    out.append(voxel_scene(ell, ["stone"] * len(ell), "L_shape"))
    hut, names = [], []
    for x in range(4):
        for z in range(4):
            for y in range(3):
                edge = x in (0, 3) or z in (0, 3)
                if y == 0:
                    hut.append((x, y, z)); names.append("planks")
                elif edge and y == 1 and not (x == 1 and z == 0):   # door gap
                    hut.append((x, y, z)); names.append("cobblestone")
                elif y == 2 and edge:
                    hut.append((x, y, z)); names.append("stone_slab")
    out.append(voxel_scene(hut, names, "tiny_hut"))
    return out


BRICKS = [(2, 4), (2, 2), (1, 4), (1, 2), (2, 3)]


def lego_scenes(n: int, n_bricks: int = 16, extent: int = 8, seed: int = 0) -> List[Scene]:
    """Random LEGO-like stacks; edges = stud connections between touching layers.

    Every brick after the first must overlap a brick in the layer above or below, so
    each stack is connected. Coordinates are the brick's min corner (x, layer, z).
    """
    rng = np.random.default_rng(seed)
    scenes = []
    for k in range(n):
        boxes: List[tuple] = []              # (x0, y, z0, w, d)
        occupied = set()
        tries = 0
        while len(boxes) < n_bricks and tries < 2000:
            tries += 1
            w, d = BRICKS[rng.integers(len(BRICKS))]
            if rng.random() < 0.5:
                w, d = d, w
            y = 0 if not boxes else int(rng.integers(0, max(b[1] for b in boxes) + 2))
            x0, z0 = int(rng.integers(0, extent - w + 1)), int(rng.integers(0, extent - d + 1))
            cells = {(x0 + i, y, z0 + j) for i in range(w) for j in range(d)}
            if cells & occupied:
                continue
            touches = any((cx, y + dy, cz) in occupied for cx, _, cz in cells for dy in (-1, 1))
            if boxes and not touches:
                continue
            boxes.append((x0, y, z0, w, d))
            occupied |= cells
        coords = np.array([(b[0], b[1], b[2]) for b in boxes], dtype=int)
        names = [f"brick_{min(b[3], b[4])}x{max(b[3], b[4])}" for b in boxes]
        extra = ["rot0" if b[3] <= b[4] else "rot90" for b in boxes]
        scenes.append(Scene(coords=coords, names=names, adjacency=stud_adjacency(boxes),
                            source=f"lego_{k}", extra=extra))
    return scenes
