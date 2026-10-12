"""Full houses in cell frames: data, cell assignments, and assignment sampling.

A *house* is the largest face-connected component of a houses_32 build, shifted so its
bounding box starts at the origin of a 32^3 grid. The coarse condition at factor ``f``
(grid ``G = 32 / f``) is :func:`coarse.coarsen` of the house (count and dominant type per
coarse cell). A *cell assignment* is the integer table ``A[c, t]``: how many blocks of
type t sit in coarse cell c. Its row sums are the condition counts and its column sums
are the inventory (the block multiset). Stage 1 samples A given (condition, inventory);
stage 2 builds the house inside the frame A provides.

Type slots: stage 2 never handles individual inventory items. "An oak plank in cell c" is
a slot, and all blocks of one (cell, type) slot are interchangeable, so the decoders
consume ``A`` (remaining counts per slot) instead of a block list.

Splits: ``data/minecraft/splits/houses_32.s0.70-15-15.v1.json`` (grouped by title).
Training houses get the 3 other 90-degree yaw rotations (positions only; block states
such as stair facing are not rotated, which only matters for renders).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .coarse import Coarse, coarsen
from .data import NodeVocab, _largest_cc, load_houses, voxel_token, voxel_vocab

S = 32                                   # house frame side
DIRS = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], np.int64)


@dataclass
class House:
    coords: np.ndarray                   # [N,3] int, origin-normalized, < 32
    types: np.ndarray                    # [N] int (vocab ids)
    raw: np.ndarray                      # [N,2] int (block id, data) for renders
    build: int = -1
    yaw: int = 0
    extra: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.types)

    @property
    def blocks(self):
        """graph_to_structure-compatible per-node block lists."""
        return [[(0, 0, 0, int(b), int(d))] for b, d in self.raw]


def _rot_yaw(c: np.ndarray) -> np.ndarray:
    c = np.stack([c[:, 2], c[:, 1], -c[:, 0]], 1)
    return c - c.min(0)


def raw_houses(build_ids: Sequence[int], yaw_aug: bool = False, max_n: Optional[int] = None) -> list:
    structures, _ = load_houses()
    out = []
    for b in build_ids:
        s = structures[b]
        occ = _largest_cc(s.occupied_mask)
        loc = np.argwhere(occ).astype(np.int64)
        if len(loc) < 2 or (max_n and len(loc) > max_n):
            continue
        ids = s.block_ids[loc[:, 0], loc[:, 1], loc[:, 2]].astype(np.int64)
        dat = s.block_data[loc[:, 0], loc[:, 1], loc[:, 2]].astype(np.int64)
        c = loc - loc.min(0)
        for r in range(4 if yaw_aug else 1):
            if r:
                c = _rot_yaw(c)
            out.append((int(b), c.copy(), ids, dat, r))
    return out


def to_houses(raw, vocab: NodeVocab) -> List[House]:
    hs = []
    for b, c, ids, dat, r in raw:
        t = np.array([vocab.token_to_type.get(voxel_token(i, d), 0) for i, d in zip(ids, dat)], np.int64)
        hs.append(House(coords=c.astype(np.int16), types=t.astype(np.int16),
                        raw=np.stack([ids, dat], 1).astype(np.int16), build=b, yaw=r))
    return hs


def build_house_cells_dataset(max_n: Optional[int] = None) -> dict:
    """{"vocab", "train", "val", "test", "stats"}: every house of the v1 split (largest
    component, no size cap unless ``max_n``), train with yaw augmentation."""
    _, split = load_houses()
    raw_tr = raw_houses(split["train"], yaw_aug=True, max_n=max_n)
    vocab = voxel_vocab([[voxel_token(i, d) for i, d in zip(r[2], r[3])] for r in raw_tr if r[4] == 0])
    out = {"vocab": vocab, "train": to_houses(raw_tr, vocab),
           "val": to_houses(raw_houses(split["val"], max_n=max_n), vocab),
           "test": to_houses(raw_houses(split["test"], max_n=max_n), vocab)}
    out["stats"] = {k: len(out[k]) for k in ("train", "val", "test")}
    out["stats"]["blocks_median"] = {k: float(np.median([len(h) for h in out[k]])) for k in ("train", "val", "test")}
    out["stats"]["blocks_max"] = {k: int(max(len(h) for h in out[k])) for k in ("train", "val", "test")}
    out["stats"]["n_types"] = vocab.n_types
    return out


def mirror_x(h: House) -> House:
    """Mirror a house along x (positions only), for augmentation."""
    c = np.asarray(h.coords).copy()
    c[:, 0] = c[:, 0].max() - c[:, 0]
    return House(coords=c, types=h.types, raw=h.raw, build=h.build, yaw=h.yaw, extra={"mirror": True})


def save_cells_dataset(ds: dict, path: Path):
    with open(path, "wb") as fh:
        pickle.dump({k: v for k, v in ds.items() if k != "vocab"} | {"vocab": ds["vocab"].to_json()}, fh)


def load_cells_dataset(path: Path) -> dict:
    with open(path, "rb") as fh:
        ds = pickle.load(fh)
    ds["vocab"] = NodeVocab.from_json(ds["vocab"])
    return ds


# --------------------------------------------------------------------------- #
# condition and assignment
# --------------------------------------------------------------------------- #
def house_coarse(h: House, f: int, n_types: int) -> Coarse:
    return coarsen(np.asarray(h.coords, np.int64), np.asarray(h.types, np.int64), f, size=S, empty=n_types, raw=[tuple(r) for r in h.raw])


def cell_of(coords: np.ndarray, f: int) -> np.ndarray:
    G = S // f
    c = np.asarray(coords, np.int64) // f
    return c[:, 0] * G * G + c[:, 1] * G + c[:, 2]


def assignment(h: House, f: int, n_types: int) -> np.ndarray:
    """[G^3, T] int32 true cell assignment (blocks of type t in cell c)."""
    G = S // f
    A = np.zeros((G ** 3, n_types), np.int32)
    np.add.at(A, (cell_of(h.coords, f), np.asarray(h.types, np.int64)), 1)
    return A


def inventory(types: np.ndarray, n_types: int) -> np.ndarray:
    return np.bincount(np.asarray(types, np.int64), minlength=n_types).astype(np.int64)


def ipf(P: np.ndarray, rows: np.ndarray, cols: np.ndarray, iters: int = 60) -> np.ndarray:
    """Iterative proportional fitting: scale a non-negative [C, T] matrix to row sums
    ``rows`` and column sums ``cols`` (both totals equal). Zero rows/cols stay zero."""
    Q = np.asarray(P, np.float64) + 0.0
    Q[:, cols == 0] = 0
    Q[rows == 0] = 0
    # every needed row must have some support among the available columns
    dead = (rows > 0) & (Q.sum(1) <= 0)
    if dead.any():
        Q[np.ix_(dead, cols > 0)] = 1.0
    for _ in range(iters):
        r = Q.sum(1)
        Q *= np.where(r > 0, rows / np.maximum(r, 1e-12), 0)[:, None]
        c = Q.sum(0)
        Q *= np.where(c > 0, cols / np.maximum(c, 1e-12), 0)[None, :]
    return Q


def sample_assignment(P: np.ndarray, rows: np.ndarray, cols: np.ndarray, rng: np.random.Generator,
                      greedy: bool = False) -> np.ndarray:
    """Draw an integer table with exact row sums ``rows`` (condition counts) and column
    sums ``cols`` (inventory) from a per-cell type distribution ``P`` [C, T].

    IPF first turns ``P`` into an expected table Q with the right margins. Then every slot
    (one per block, ``rows[c]`` slots in cell c) is visited in a random interleaved order
    and takes a type with probability proportional to the still-unused part of Q in its
    row, restricted to types with inventory left. Visiting slots of all cells interleaved
    spreads the leftovers instead of dumping them on the last cell. ``greedy`` takes the
    largest remaining Q entry instead of sampling."""
    Q = ipf(P, rows, cols)
    left = cols.astype(np.int64).copy()
    out = np.zeros(Q.shape, np.int32)
    slots = np.repeat(np.arange(len(rows)), rows.astype(np.int64))
    slots = slots[rng.permutation(len(slots))]
    Qr = Q.copy()
    for c in slots:
        w = np.where(left > 0, np.maximum(Qr[c], 0) + 1e-9 * (left > 0), 0.0)
        if greedy:
            t = int(np.argmax(w))
        else:
            t = int(rng.choice(len(w), p=w / w.sum()))
        out[c, t] += 1
        Qr[c, t] -= 1
        left[t] -= 1
    return out


def prior_cell_dist(cond: Coarse, inv: np.ndarray, alpha: float = 4.0) -> np.ndarray:
    """Heuristic stage-1 baseline: each cell's dominant type gets weight ``alpha``, the
    rest follow the global inventory frequencies (no learning)."""
    cnt, dom = cond.flat()
    T = len(inv)
    P = np.tile(inv / max(inv.sum(), 1), (len(cnt), 1))
    for c in np.nonzero(cnt > 0)[0]:
        if dom[c] < T:
            P[c, dom[c]] += alpha
    return P


def assignment_l1(A: np.ndarray, B: np.ndarray) -> float:
    """Share of blocks in a different (cell, type) slot: sum |A - B| / (2 N)."""
    return float(np.abs(A.astype(np.int64) - B).sum() / max(2 * A.sum(), 1))
