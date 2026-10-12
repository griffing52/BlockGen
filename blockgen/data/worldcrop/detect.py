"""Find player-built structures in a world and crop them.

Pipeline per world (see ``run.py`` for orchestration):

1. **Chunk scan** - decode every chunk once; per chunk record the number of STRONG
   (seed) voxels and their y-range. Vanilla-structure bounding boxes from
   ``data/*.dat`` are dilated and zero out seeds inside them.
2. **Chunk clustering** - seed-bearing chunks within Chebyshev distance
   ``chunk_gap+1`` of each other form a cluster (union-find). Clusters are tiled to at
   most ``tile_chunks`` on a side.
3. **Voxel labelling** - for each tile, materialise the dense ``(ids, data)`` volume
   over the seeds' y-range (padded), build ``mask = strong | weak``, dilate by one
   voxel (so slabs/stairs separated by a one-block gap connect), 26-connected
   components, keep components with enough seed voxels and a sane bounding box.
4. **Crop** - cut the component's bbox (+1 xz margin, one ground layer below) from the
   *full* volume and emit ``ids``, ``data`` and the component ``mask`` so downstream
   can choose "structure only" or "structure in context".

Flat worlds use the preset layer stack instead of the STRONG/WEAK tiers: anything
that is not the layer block at that height (and not a decoration/fluid) is a seed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from . import anvil, blocks


@dataclass
class DetectConfig:
    dimension: str = "overworld"
    chunk_gap: int = 1            # clusters bridge this many empty chunks
    tile_chunks: int = 24         # max tile side, in chunks (24*16 = 384 blocks)
    struct_dilate: int = 3        # voxels added around vanilla structure boxes
    min_seed: int = 12            # seed voxels a component needs
    min_total: int = 24           # strong+weak voxels a component needs
    min_dims: Tuple[int, int, int] = (2, 2, 2)   # sorted bbox dims must be >= these
    min_max_dim: int = 4
    max_dim: int = 320            # larger components are recorded but not saved
    y_pad: int = 2
    seed_fraction_min: float = 0.10  # seeds / (seeds+weak) of a component must reach this
    flat_ore_frac_max: float = 0.05  # a 'flat' level.dat with ores in more chunks than this is not flat
    min_distinct_ids: int = 2
    orelike_min_family: int = 8
    orelike_max_seed: int = 150
    max_candidates: int = 4000
    retile_depth: int = 3        # re-tile edge-touching components this many times before giving up


@dataclass
class ChunkStat:
    cx: int
    cz: int
    n_seed: int
    n_weak: int
    n_blocks: int
    y_min: int
    y_max: int
    inhabited_time: int
    n_tile_entities: int
    n_ore: int = 0


@dataclass
class Crop:
    ids: np.ndarray        # (X, Y, Z) uint16, legacy ids
    data: np.ndarray       # (X, Y, Z) uint8
    mask: np.ndarray       # (X, Y, Z) bool, component membership
    bbox: Tuple[int, int, int, int, int, int]   # world coords minx,miny,minz,maxx,maxy,maxz (inclusive)
    n_seed: int
    n_weak: int
    n_component: int
    n_nonair: int
    distinct_ids: int
    inhabited_time_mean: float
    route: str             # flat | heuristic
    oversize: bool = False
    cut: bool = False            # touched a tile edge and could not be re-tiled whole
    top_ids: Dict[int, int] = field(default_factory=dict)        # all component voxels
    top_seed_ids: Dict[int, int] = field(default_factory=dict)   # seed voxels only

    @property
    def shape(self):
        return tuple(self.ids.shape)

    def record(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k not in ("ids", "data", "mask")}
        d["shape"] = list(self.shape)
        d["top_ids"] = {str(k): v for k, v in self.top_ids.items()}
        d["top_seed_ids"] = {str(k): v for k, v in self.top_seed_ids.items()}
        return d


# --------------------------------------------------------------------------- #
# Seed rules
# --------------------------------------------------------------------------- #
class SeedRule:
    def __init__(self, level: anvil.LevelInfo, force_heuristic: bool = False):
        self.strong_lut, self.weak_lut = blocks.class_tables()
        self.route = "heuristic"
        self.flat_col = None
        if level.is_flat and not force_heuristic:
            self.route = "flat"
            self.flat_col = blocks.flat_column(blocks.flat_layers(level.generator_options))
            self.flat_ignore = np.zeros(blocks.MAX_ID, bool)
            self.flat_ignore[list(blocks.FLAT_IGNORE)] = True

    def masks(self, ids: np.ndarray, data: np.ndarray, y0: int) -> Tuple[np.ndarray, np.ndarray]:
        """(seed, weak) masks for an (X, Y, Z) block volume whose y index 0 is world y0."""
        if self.route == "flat":
            ys = np.arange(ids.shape[1]) + y0
            ys = np.clip(ys, 0, self.flat_col.size - 1)
            expected = self.flat_col[ys][None, :, None]
            seed = (ids != expected) & ~self.flat_ignore[ids]
            weak = np.zeros_like(seed)
            # logs/leaves are ignored as seeds but logs may still grow a component
            weak |= (ids == 17) | (ids == 162)
            return seed, weak
        return blocks.classify(ids, data, self.strong_lut, self.weak_lut)


# --------------------------------------------------------------------------- #
# Pass 1: chunk scan
# --------------------------------------------------------------------------- #
def _struct_mask_for_chunk(boxes: Sequence[Tuple[int, int, int, int, int, int]], cx: int, cz: int,
                           dil: int) -> Optional[List[Tuple[int, int, int, int, int, int]]]:
    """Boxes (dilated) intersecting this chunk, in chunk-local xz / world y coordinates."""
    x0, z0 = cx * 16, cz * 16
    hits = []
    for (ax, ay, az, bx, by, bz) in boxes:
        ax, ay, az, bx, by, bz = ax - dil, ay - dil, az - dil, bx + dil, by + dil, bz + dil
        if bx < x0 or ax > x0 + 15 or bz < z0 or az > z0 + 15:
            continue
        hits.append((max(ax - x0, 0), ay, max(az - z0, 0), min(bx - x0, 15), by, min(bz - z0, 15)))
    return hits or None


def _apply_struct_boxes(seed: np.ndarray, hits, y0: int) -> None:
    for (ax, ay, az, bx, by, bz) in hits:
        ya, yb = max(ay - y0, 0), min(by - y0, seed.shape[1] - 1)
        if yb < 0 or ya >= seed.shape[1]:
            continue
        seed[ax : bx + 1, ya : yb + 1, az : bz + 1] = False


def chunk_volume(ch: anvil.Chunk, y_lo_sec: int, y_hi_sec: int) -> Tuple[np.ndarray, np.ndarray]:
    """Dense (16, H, 16) volume over sections [y_lo_sec, y_hi_sec]."""
    h = (y_hi_sec - y_lo_sec + 1) * 16
    ids = np.zeros((16, h, 16), np.uint16)
    data = np.zeros((16, h, 16), np.uint8)
    for sy, (a, d) in ch.sections.items():
        if y_lo_sec <= sy <= y_hi_sec:
            o = (sy - y_lo_sec) * 16
            ids[:, o : o + 16, :] = a
            data[:, o : o + 16, :] = d
    return ids, data


def scan_chunks(world_dir: str, level: anvil.LevelInfo, rule: SeedRule, cfg: DetectConfig,
                boxes=None) -> Tuple[Dict[Tuple[int, int], ChunkStat], dict]:
    boxes = [b for _, b in (boxes if boxes is not None else anvil.read_structure_boxes(world_dir))]
    stats: Dict[Tuple[int, int], ChunkStat] = {}
    ore_lut = np.zeros(blocks.MAX_ID, bool)
    ore_lut[list(blocks.ORES)] = True
    summary = {"chunks": 0, "sections": 0, "fmt": {}, "unknown_states": 0, "struct_boxes": len(boxes)}
    for rf in anvil.find_region_files(world_dir, cfg.dimension):
        for ch in anvil.iter_region_chunks(rf):
            summary["chunks"] += 1
            summary["sections"] += len(ch.sections)
            summary["fmt"][ch.fmt] = summary["fmt"].get(ch.fmt, 0) + 1
            summary["unknown_states"] += ch.unknown_states
            lo, hi = ch.min_sy(), ch.max_sy()
            ids, data = chunk_volume(ch, lo, hi)
            y0 = lo * 16
            seed, weak = rule.masks(ids, data, y0)
            hits = _struct_mask_for_chunk(boxes, ch.cx, ch.cz, cfg.struct_dilate) if boxes else None
            if hits:
                _apply_struct_boxes(seed, hits, y0)
            n_ore = int(ore_lut[ids].sum())
            if ids.max() > 255:
                summary["ids_over_255"] = summary.get("ids_over_255", 0) + int((ids > 255).sum())
            n_seed = int(seed.sum())
            if n_seed:
                ys = np.nonzero(seed.any(axis=(0, 2)))[0]
                y_min, y_max = int(ys.min()) + y0, int(ys.max()) + y0
            else:
                y_min = y_max = -1
            stats[(ch.cx, ch.cz)] = ChunkStat(ch.cx, ch.cz, n_seed, int(weak.sum()), ch.n_blocks, y_min, y_max,
                                              ch.inhabited_time, ch.n_tile_entities, n_ore)
    return stats, summary


# --------------------------------------------------------------------------- #
# Pass 2: clustering
# --------------------------------------------------------------------------- #
def cluster_chunks(stats: Dict[Tuple[int, int], ChunkStat], gap: int) -> List[List[Tuple[int, int]]]:
    seeds = {k for k, s in stats.items() if s.n_seed > 0}
    parent = {k: k for k in seeds}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    r = gap + 1
    for (cx, cz) in seeds:
        for dx in range(-r, r + 1):
            for dz in range(-r, r + 1):
                nb = (cx + dx, cz + dz)
                if nb in seeds and nb != (cx, cz):
                    ra, rb = find((cx, cz)), find(nb)
                    if ra != rb:
                        parent[ra] = rb
    groups: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for k in seeds:
        groups.setdefault(find(k), []).append(k)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def tiles_for_cluster(cluster: List[Tuple[int, int]], tile: int, halo: int = 1) -> Iterator[Tuple[int, int, int, int]]:
    """Yield (cx0, cz0, cx1, cz1) inclusive chunk ranges (with halo) covering the cluster."""
    xs = [c[0] for c in cluster]
    zs = [c[1] for c in cluster]
    x0, x1, z0, z1 = min(xs), max(xs), min(zs), max(zs)
    inner = tile - 2 * halo
    for tx in range(x0, x1 + 1, inner):
        for tz in range(z0, z1 + 1, inner):
            yield tx - halo, tz - halo, min(tx + inner - 1, x1) + halo, min(tz + inner - 1, z1) + halo


# --------------------------------------------------------------------------- #
# Pass 3: label + crop
# --------------------------------------------------------------------------- #
_STRUCT26 = np.ones((3, 3, 3), bool)


def _load_tile(world_dir: str, cfg: DetectConfig, cx0: int, cz0: int, cx1: int, cz1: int,
               sy_lo: int, sy_hi: int, stats: Dict[Tuple[int, int], ChunkStat]):
    """Dense volume for a chunk range over sections [sy_lo, sy_hi]."""
    W, D = (cx1 - cx0 + 1) * 16, (cz1 - cz0 + 1) * 16
    H = (sy_hi - sy_lo + 1) * 16
    ids = np.zeros((W, H, D), np.uint16)
    data = np.zeros((W, H, D), np.uint8)
    itime = np.full(((cx1 - cx0 + 1), (cz1 - cz0 + 1)), -1, np.int64)
    # which region files intersect
    rx0, rx1, rz0, rz1 = cx0 >> 5, cx1 >> 5, cz0 >> 5, cz1 >> 5
    base = {"overworld": "region", "nether": "DIM-1/region", "end": "DIM1/region"}[cfg.dimension]
    for rx in range(rx0, rx1 + 1):
        for rz in range(rz0, rz1 + 1):
            for ext in (".mca", ".mcr"):
                rf = os.path.join(world_dir, base, f"r.{rx}.{rz}{ext}")
                if os.path.exists(rf):
                    break
            else:
                continue
            for ch in anvil.iter_region_chunks(rf):
                if not (cx0 <= ch.cx <= cx1 and cz0 <= ch.cz <= cz1):
                    continue
                ox, oz = (ch.cx - cx0) * 16, (ch.cz - cz0) * 16
                for sy, (a, d) in ch.sections.items():
                    if sy_lo <= sy <= sy_hi:
                        oy = (sy - sy_lo) * 16
                        ids[ox : ox + 16, oy : oy + 16, oz : oz + 16] = a
                        data[ox : ox + 16, oy : oy + 16, oz : oz + 16] = d
                itime[ch.cx - cx0, ch.cz - cz0] = ch.inhabited_time
    return ids, data, itime


def label_tile(ids, data, seed, weak, cfg: DetectConfig):
    mask = seed | weak
    if not mask.any():
        return None, 0
    grown = ndimage.binary_dilation(mask, structure=_STRUCT26)
    labels, n = ndimage.label(grown, structure=_STRUCT26)
    # restrict labels to actual member voxels (not the dilation shell)
    labels = labels * mask
    return labels, n


def _chunk_range_of(bbox, halo: int = 1):
    bx0, _, bz0, bx1, _, bz1 = bbox
    return (bx0 >> 4) - halo, (bz0 >> 4) - halo, (bx1 >> 4) + halo, (bz1 >> 4) + halo


def _tile_crops(world_dir, rule, cfg, stats, boxes, seen, cx0, cz0, cx1, cz1, sy_lo, sy_hi,
                allow_cut: bool, tile_cb=None):
    """Label one tile. Yields ("crop", Crop) for whole components and ("cut", bbox) for
    components touching the tile's x/z edge (unless allow_cut, when they are yielded as
    crops flagged cut=True). Edge components are cut by the tile, not by the world, so
    the caller re-tiles around them instead of saving half a boat."""
    ids, data, itime = _load_tile(world_dir, cfg, cx0, cz0, cx1, cz1, sy_lo, sy_hi, stats)
    W, D = ids.shape[0], ids.shape[2]
    y0 = sy_lo * 16
    seed, weak = rule.masks(ids, data, y0)
    if boxes:
        X0, Z0 = cx0 * 16, cz0 * 16
        for (ax, ay, az, bx, by, bz) in boxes:
            d = cfg.struct_dilate
            sl = (slice(max(ax - d - X0, 0), max(bx + d - X0 + 1, 0)),
                  slice(max(ay - d - y0, 0), max(by + d - y0 + 1, 0)),
                  slice(max(az - d - Z0, 0), max(bz + d - Z0 + 1, 0)))
            seed[sl] = False
    labels, n = label_tile(ids, data, seed, weak, cfg)
    if tile_cb:
        tile_cb(cx0, cz0, cx1, cz1, n)
    if not n:
        return
    objs = ndimage.find_objects(labels)
    n_seed_all = ndimage.sum_labels(seed, labels, index=np.arange(1, n + 1))
    n_mem_all = ndimage.sum_labels(labels > 0, labels, index=np.arange(1, n + 1))
    for li, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        ns, nm = int(n_seed_all[li - 1]), int(n_mem_all[li - 1])
        if ns < cfg.min_seed or nm < cfg.min_total or ns < cfg.seed_fraction_min * nm:
            continue
        dims = sorted(s.stop - s.start for s in sl)
        if dims[2] < cfg.min_max_dim or any(dm < m for dm, m in zip(dims, sorted(cfg.min_dims))):
            continue
        bx0, by0, bz0 = sl[0].start + cx0 * 16, sl[1].start + y0, sl[2].start + cz0 * 16
        bx1, by1, bz1 = sl[0].stop - 1 + cx0 * 16, sl[1].stop - 1 + y0, sl[2].stop - 1 + cz0 * 16
        bbox = (bx0, by0, bz0, bx1, by1, bz1)
        touches = sl[0].start == 0 or sl[0].stop == W or sl[2].start == 0 or sl[2].stop == D
        if touches and not allow_cut:
            yield ("cut", bbox)
            continue
        if any(_contains(sb, bbox) or _iou(sb, bbox) > 0.6 for sb in seen):
            continue
        seen.append(bbox)
        xs = slice(max(sl[0].start - 1, 0), min(sl[0].stop + 1, ids.shape[0]))
        ys = slice(max(sl[1].start - 1, 0), sl[1].stop)
        zs = slice(max(sl[2].start - 1, 0), min(sl[2].stop + 1, ids.shape[2]))
        comp = labels[xs, ys, zs] == li
        cids = ids[xs, ys, zs]
        cdat = data[xs, ys, zs]
        vals, counts = np.unique(cids[comp], return_counts=True)
        order = np.argsort(-counts)
        top = {int(vals[i]): int(counts[i]) for i in order[:12]}
        sv, sc = np.unique(cids[comp & seed[xs, ys, zs]], return_counts=True)
        sorder = np.argsort(-sc)
        top_seed = {int(sv[i]): int(sc[i]) for i in sorder[:8]}
        if len(vals) < cfg.min_distinct_ids:
            continue
        cxs = slice((xs.start + cx0 * 16) >> 4, ((xs.stop - 1 + cx0 * 16) >> 4) + 1)
        czs = slice((zs.start + cz0 * 16) >> 4, ((zs.stop - 1 + cz0 * 16) >> 4) + 1)
        it = itime[cxs.start - cx0 : cxs.stop - cx0, czs.start - cz0 : czs.stop - cz0]
        it = it[it >= 0]
        crop = Crop(ids=cids.copy(), data=cdat.copy(), mask=comp, bbox=bbox, n_seed=ns,
                    n_weak=nm - ns, n_component=nm, n_nonair=int((cids != 0).sum()),
                    distinct_ids=int(len(vals)),
                    inhabited_time_mean=float(it.mean()) if it.size else -1.0,
                    route=rule.route, oversize=(max(dims) > cfg.max_dim) or touches, cut=touches,
                    top_ids=top, top_seed_ids=top_seed)
        yield ("crop", crop)


def extract_crops(world_dir: str, level: anvil.LevelInfo, rule: SeedRule, cfg: DetectConfig,
                  stats: Dict[Tuple[int, int], ChunkStat], boxes=None,
                  tile_cb=None) -> Iterator[Crop]:
    boxes = [b for _, b in (boxes if boxes is not None else anvil.read_structure_boxes(world_dir))]
    clusters = cluster_chunks(stats, cfg.chunk_gap)
    seen: List[Tuple[int, int, int, int, int, int]] = []
    for cluster in clusters:
        ymins = [stats[c].y_min for c in cluster]
        ymaxs = [stats[c].y_max for c in cluster]
        y_lo = max(min(ymins) - cfg.y_pad - 1, 0)
        y_hi = max(ymaxs) + cfg.y_pad
        sy_lo, sy_hi = y_lo >> 4, y_hi >> 4
        # (tile range, depth): depth 0 = regular tiling, >0 = re-tiles around cut components
        queue = [(t, 0) for t in tiles_for_cluster(cluster, cfg.tile_chunks)]
        tried = set()
        while queue:
            (cx0, cz0, cx1, cz1), depth = queue.pop(0)
            if (cx0, cz0, cx1, cz1) in tried:
                continue
            tried.add((cx0, cz0, cx1, cz1))
            for kind, obj in _tile_crops(world_dir, rule, cfg, stats, boxes, seen, cx0, cz0, cx1, cz1,
                                         sy_lo, sy_hi, allow_cut=False, tile_cb=tile_cb):
                if kind == "crop":
                    yield obj
                    continue
                # a component cut by this tile: re-tile around its bbox (plus a halo chunk)
                rx0, rz0, rx1, rz1 = _chunk_range_of(obj)
                span = max(rx1 - rx0 + 1, rz1 - rz0 + 1)
                if depth < cfg.retile_depth and span <= cfg.tile_chunks:
                    queue.append(((rx0, rz0, rx1, rz1), depth + 1))
                else:
                    # too big to hold in one tile (or still growing): record it as cut + oversize
                    for kind2, obj2 in _tile_crops(world_dir, rule, cfg, stats, boxes, seen, rx0, rz0,
                                                   min(rx1, rx0 + cfg.tile_chunks - 1), min(rz1, rz0 + cfg.tile_chunks - 1),
                                                   sy_lo, sy_hi, allow_cut=True):
                        if kind2 == "crop" and _iou(obj2.bbox, obj) > 0.3:
                            yield obj2


def _iou(a, b) -> float:
    ix = max(0, min(a[3], b[3]) - max(a[0], b[0]) + 1)
    iy = max(0, min(a[4], b[4]) - max(a[1], b[1]) + 1)
    iz = max(0, min(a[5], b[5]) - max(a[2], b[2]) + 1)
    inter = ix * iy * iz
    if not inter:
        return 0.0
    va = (a[3] - a[0] + 1) * (a[4] - a[1] + 1) * (a[5] - a[2] + 1)
    vb = (b[3] - b[0] + 1) * (b[4] - b[1] + 1) * (b[5] - b[2] + 1)
    return inter / (va + vb - inter)


def _contains(outer, inner) -> bool:
    return (outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] <= inner[2]
            and outer[3] >= inner[3] and outer[4] >= inner[4] and outer[5] >= inner[5])


# --------------------------------------------------------------------------- #
# Whole-world driver
# --------------------------------------------------------------------------- #
def detect_world(world_dir: str, cfg: Optional[DetectConfig] = None):
    """Yield (level_info, summary, crops_iterator) for one extracted world directory."""
    cfg = cfg or DetectConfig()
    level = anvil.read_level_dat(os.path.join(world_dir, "level.dat"))
    rule = SeedRule(level)
    boxes = anvil.read_structure_boxes(world_dir)
    stats, summary = scan_chunks(world_dir, level, rule, cfg, boxes=boxes)
    ore_chunks = sum(1 for s in stats.values() if s.n_ore > 0)
    summary["ore_chunk_frac"] = (ore_chunks / len(stats)) if stats else 0.0
    if rule.route == "flat" and stats and ore_chunks > cfg.flat_ore_frac_max * len(stats):
        # level.dat lies (converted / re-typed world): natural ores never come from a flat preset
        summary["flat_rejected"] = True
        rule = SeedRule(level, force_heuristic=True)
        stats, summary2 = scan_chunks(world_dir, level, rule, cfg, boxes=boxes)
        summary2["flat_rejected"] = True
        summary2["ore_chunk_frac"] = summary["ore_chunk_frac"]
        summary = summary2
    summary["route"] = rule.route
    summary["seed_chunks"] = sum(1 for s in stats.values() if s.n_seed > 0)
    summary["seed_voxels"] = int(sum(s.n_seed for s in stats.values()))
    summary["inhabited0_frac"] = (sum(1 for s in stats.values() if s.inhabited_time == 0) / len(stats)) if stats else None
    crops = list(extract_crops(world_dir, level, rule, cfg, stats, boxes=boxes))
    crops, n_orelike = drop_orelike(crops, cfg)
    summary["orelike_dropped"] = n_orelike
    summary["modded_suspect"] = bool(summary.get("ids_over_255", 0) > 0)
    return level, summary, iter(crops)


def drop_orelike(crops: List[Crop], cfg: DetectConfig) -> Tuple[List[Crop], int]:
    """Drop families of many small look-alike components (modded ores read as vanilla ids,
    silverfish stone, replaced-ore server maps). A *family* is the sorted set of the
    component's top-2 seed ids; if a family has >= ``orelike_min_family`` members and its
    median seed count is < ``orelike_max_seed`` they are all removed."""
    fam: Dict[Tuple[int, ...], List[int]] = {}
    for i, c in enumerate(crops):
        key = tuple(sorted(list(c.top_seed_ids)[:2]))
        fam.setdefault(key, []).append(i)
    drop = set()
    for key, idx in fam.items():
        if len(idx) >= cfg.orelike_min_family:
            med = float(np.median([crops[i].n_seed for i in idx]))
            if med < cfg.orelike_max_seed:
                drop.update(idx)
    return [c for i, c in enumerate(crops) if i not in drop], len(drop)
