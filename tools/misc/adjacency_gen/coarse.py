"""Window-aligned chunks and their coarse-shape condition (the face formulation's data).

A chunk is the largest face-connected component of an ``S^3`` window of a houses_32
build (default S = 8), with 40..120 blocks. Unlike :func:`data.voxel_chunks_raw`, coordinates
stay in the *window frame* (0..S-1), because the coarse condition is pinned to that
frame. Window origins are multiples of 2, so with ``f = 2`` the coarse cells line up
with the repo's canon16 decimation of the 32^3 grid.

Coarse condition at factor ``f`` (grid ``G = S / f``): per coarse cell, the number of
chunk blocks inside it (0..f^3) and the dominant block type (``EMPTY`` when the cell is
empty). The representative raw ``(id, data)`` of the dominant type is kept for the figure.

The inventory (the multiset of block types) still comes from the real chunk.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import Graph, NodeVocab, _largest_cc, load_houses, voxel_token, voxel_vocab


# --------------------------------------------------------------------------- #
# chunks in the window frame
# --------------------------------------------------------------------------- #
def window_chunks_raw(build_ids: Sequence[int], per_build: int, size: int = 8, min_n: int = 40,
                      max_n: int = 120, stride: int = 2, seed: int = 0) -> list:
    """Raw chunks ``(build, coords [N,3] window frame, ids [N], data [N], origin)``.

    Origins are drawn without replacement from the multiples of ``stride`` that keep the
    window inside the build."""
    structures, _ = load_houses()
    out = []
    for b in build_ids:
        rng = np.random.default_rng(seed * 1_000_003 + int(b))
        s = structures[b]
        occ = s.occupied_mask
        if min(occ.shape) < size:
            continue
        axes = [np.arange(0, d - size + 1, stride) for d in occ.shape]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        got = 0
        for o in grid[rng.permutation(len(grid))]:
            if got >= per_build:
                break
            x, y, z = (int(v) for v in o)
            win = _largest_cc(occ[x:x + size, y:y + size, z:z + size])
            n = int(win.sum())
            if not (min_n <= n <= max_n):
                continue
            loc = np.argwhere(win)
            g = loc + o
            ids = s.block_ids[g[:, 0], g[:, 1], g[:, 2]].astype(np.int64)
            dat = s.block_data[g[:, 0], g[:, 1], g[:, 2]].astype(np.int64)
            out.append((int(b), loc.astype(np.int64), ids, dat, (x, y, z)))
            got += 1
    return out


def window_graphs(raw, vocab: NodeVocab) -> List[Graph]:
    gs = []
    for b, loc, ids, dat, o in raw:
        types = np.array([vocab.token_to_type.get(voxel_token(i, d), 0) for i, d in zip(ids, dat)],
                         dtype=np.int64)
        blocks = [[(0, 0, 0, int(i), int(d))] for i, d in zip(ids, dat)]
        gs.append(Graph(types=types, coords=loc, blocks=blocks, build=int(b), source=f"h{b:05d}@{o}"))
    return gs


def build_window_dataset(n_train_builds: Optional[int] = None, per_build_train: int = 24,
                         per_build_eval: int = 2, n_eval_builds: int = 399, size: int = 8,
                         min_n: int = 40, max_n: int = 120, seed: int = 0) -> dict:
    """{"vocab", "train", "val", "test", "stats"}; split by build (houses_32 v1 split)."""
    _, split = load_houses()
    tr = split["train"][:n_train_builds] if n_train_builds else split["train"]
    kw = dict(size=size, min_n=min_n, max_n=max_n)
    raw_tr = window_chunks_raw(tr, per_build_train, seed=seed, **kw)
    vocab = voxel_vocab([[voxel_token(i, d) for i, d in zip(r[2], r[3])] for r in raw_tr])
    out = {"train": window_graphs(raw_tr, vocab),
           "val": window_graphs(window_chunks_raw(split["val"][:n_eval_builds], per_build_eval,
                                                  seed=seed + 1, **kw), vocab),
           "test": window_graphs(window_chunks_raw(split["test"][:n_eval_builds], per_build_eval,
                                                   seed=seed + 2, **kw), vocab),
           "vocab": vocab}
    stats = {k: len(out[k]) for k in ("train", "val", "test")}
    stats["nodes_median"] = {k: float(np.median([len(g) for g in out[k]])) for k in ("train", "val", "test")}
    stats["window"] = size
    stats["block_range"] = [min_n, max_n]
    out["stats"] = stats
    return out


def house_chunks_raw(build_ids: Sequence[int], min_n: int = 64, max_n: int = 512,
                     yaw_aug: bool = False) -> list:
    """Whole houses_32 builds (largest face-connected component) with min_n..max_n blocks,
    in the build's 32^3 frame. ``yaw_aug`` adds the 3 other 90-degree yaw rotations
    (positions only: block states such as stair facing are not rotated)."""
    structures, _ = load_houses()
    out = []
    for b in build_ids:
        s = structures[b]
        occ = _largest_cc(s.occupied_mask)
        n = int(occ.sum())
        if not (min_n <= n <= max_n):
            continue
        loc = np.argwhere(occ).astype(np.int64)
        ids = s.block_ids[loc[:, 0], loc[:, 1], loc[:, 2]].astype(np.int64)
        dat = s.block_data[loc[:, 0], loc[:, 1], loc[:, 2]].astype(np.int64)
        c = loc.copy()
        for r in range(4 if yaw_aug else 1):
            if r:
                c = np.stack([c[:, 2], c[:, 1], -c[:, 0]], 1)
                c = c - c.min(0)
            out.append((int(b), c.copy(), ids, dat, f"yaw{r}"))
    return out


def build_house_dataset(min_n: int = 64, max_n: int = 512) -> dict:
    """Full-house variant: train builds with yaw augmentation, val/test unrotated."""
    _, split = load_houses()
    raw_tr = house_chunks_raw(split["train"], min_n, max_n, yaw_aug=True)
    vocab = voxel_vocab([[voxel_token(i, d) for i, d in zip(r[2], r[3])] for r in raw_tr])
    out = {"train": window_graphs(raw_tr, vocab),
           "val": window_graphs(house_chunks_raw(split["val"], min_n, max_n), vocab),
           "test": window_graphs(house_chunks_raw(split["test"], min_n, max_n), vocab), "vocab": vocab}
    stats = {k: len(out[k]) for k in ("train", "val", "test")}
    stats["nodes_median"] = {k: float(np.median([len(g) for g in out[k]])) for k in ("train", "val", "test")}
    stats["window"] = 32
    stats["block_range"] = [min_n, max_n]
    out["stats"] = stats
    return out


# --------------------------------------------------------------------------- #
# coarse condition
# --------------------------------------------------------------------------- #
@dataclass
class Coarse:
    f: int
    G: int
    count: np.ndarray            # [G,G,G] int, blocks per coarse cell
    dom: np.ndarray              # [G,G,G] int, dominant type, EMPTY (= n_types) if count 0
    dom_raw: Dict[Tuple[int, int, int], Tuple[int, int]]   # cell -> raw (id, data) for figures

    @property
    def occ(self) -> np.ndarray:
        return self.count > 0

    def flat(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.count.reshape(-1), self.dom.reshape(-1)


def coarse_cell(coords: np.ndarray, f: int, G: int) -> np.ndarray:
    """Flat coarse-cell index of each fine coordinate (row-major x, y, z)."""
    c = np.asarray(coords) // f
    return c[:, 0] * G * G + c[:, 1] * G + c[:, 2]


def coarsen(coords: np.ndarray, types: np.ndarray, f: int, size: int = 8, empty: int = 0,
            raw: Optional[Sequence[Tuple[int, int]]] = None) -> Coarse:
    """Downsample a window-frame block set by ``f``. Cells outside the window are ignored.
    Ties in the dominant type go to the smallest type id (deterministic)."""
    G = size // f
    count = np.zeros((G, G, G), np.int64)
    dom = np.full((G, G, G), empty, np.int64)
    coords = np.asarray(coords)
    inside = np.all((coords >= 0) & (coords < size), axis=1)
    per: Dict[Tuple[int, int, int], Counter] = {}
    rawc: Dict[Tuple[int, int, int], Counter] = {}
    for k in np.nonzero(inside)[0]:
        c = tuple(int(v) for v in coords[k] // f)
        count[c] += 1
        per.setdefault(c, Counter())[int(types[k])] += 1
        if raw is not None:
            rawc.setdefault(c, Counter())[tuple(raw[k])] += 1
    for c, cnt in per.items():
        top = max(cnt.values())
        dom[c] = min(t for t, v in cnt.items() if v == top)
    dom_raw = {c: cnt.most_common(1)[0][0] for c, cnt in rawc.items()}
    return Coarse(f=f, G=G, count=count, dom=dom, dom_raw=dom_raw)


def graph_coarse(g: Graph, f: int, vocab: NodeVocab, size: int = 8) -> Coarse:
    raw = [(b[0][3], b[0][4]) for b in g.blocks]
    return coarsen(g.coords, g.types, f, size=size, empty=vocab.n_types, raw=raw)


def coarse_structure(c: Coarse):
    """The condition as a Structure for figures: each occupied cell is an ``f^3`` cube of
    its dominant block, with the cube's fill fraction shown by filling only the bottom
    ``ceil(count / f^2)`` layers."""
    from blockgen.utils.data import Structure
    S = c.G * c.f
    bi = np.zeros((S, S, S), np.int32)
    bd = np.zeros((S, S, S), np.int32)
    for cell, (b, d) in c.dom_raw.items():
        layers = int(np.ceil(c.count[cell] / (c.f * c.f)))
        x, y, z = (v * c.f for v in cell)
        bi[x:x + c.f, y:y + layers, z:z + c.f] = b
        bd[x:x + c.f, y:y + layers, z:z + c.f] = d
    return Structure(block_ids=bi, block_data=bd)


# --------------------------------------------------------------------------- #
# the key baseline: naive coarse upsampling
# --------------------------------------------------------------------------- #
def upsample_baseline(types: np.ndarray, c: Coarse, seed: int = 0) -> dict:
    """Fill each coarse cell with ``count`` of the inventory's blocks at random free
    positions inside the cell. Blocks are assigned to cells by type: each cell first takes
    blocks of its dominant type (as many as it needs and the inventory has left), the
    remaining slots are filled with random leftover blocks. Returns ``pos`` [N,3] and
    ``placed`` [N] bool (False only if the counts and the inventory disagree)."""
    rng = np.random.default_rng(seed)
    N = len(types)
    pos = np.zeros((N, 3), np.int64)
    placed = np.zeros(N, bool)
    left = {t: list(rng.permutation(np.nonzero(types == t)[0])) for t in np.unique(types)}
    cells = [tuple(int(v) for v in idx) for idx in np.argwhere(c.count > 0)]
    order = rng.permutation(len(cells))
    need = {cells[k]: int(c.count[cells[k]]) for k in order}
    assign: Dict[Tuple[int, int, int], List[int]] = {cell: [] for cell in need}
    # pass 1: dominant type, round-robin one block per cell so no cell starves the rest
    progress = True
    while progress:
        progress = False
        for k in order:
            cell = cells[k]
            t = int(c.dom[cell])
            if len(assign[cell]) < need[cell] and left.get(t):
                assign[cell].append(int(left[t].pop()))
                progress = True
    rest = [i for v in left.values() for i in v]
    rng.shuffle(rest)
    for k in order:
        cell = cells[k]
        while len(assign[cell]) < need[cell] and rest:
            assign[cell].append(int(rest.pop()))
    f = c.f
    for cell, members in assign.items():
        local = np.stack(np.meshgrid(*[np.arange(f)] * 3, indexing="ij"), -1).reshape(-1, 3)
        spots = local[rng.permutation(len(local))[:len(members)]] + np.array(cell) * f
        for i, p in zip(members, spots):
            pos[i] = p
            placed[i] = True
    return {"pos": pos, "placed": placed}
