"""Faces as a fixed coarse-to-fine stack of categorical grids, the data side of the cascade.

Step 2 of the 2D next-scale plan (research.md H.7, notes §63): the levels are
``facade.pool`` at factors 16, 8, 4, 2, 1 of a face placed on a 32 x 32 canvas, so
level ``f`` is a ``(32/f) x (32/f)`` grid whose cell ``(i, j)`` covers canvas cells
``[i f, (i+1) f) x [j f, (j+1) f)``. Each cell holds two categorical values:

``depth class``  0 is see-through, ``1 + d`` is a filled cell whose first block
                 sits ``d`` voxels in (``d < extent <= 32``), so 33 classes.
``block class``  index into a train-split vocabulary of ``(id, data)`` keys, 0 the
                 unknown block, ``1..K`` the kept keys. Zero where see-through.

A cell is *in the box* when any of the canvas cells it covers lies inside the
face's ``U x V`` rectangle; cells outside are never predicted. ``encode`` turns a
face into its stack, ``decode`` turns one level back into a ``Facade`` at the face's
own resolution (repeating cells, the naive decode of the probe), and ``parent``
gives a level the values its coarser neighbour implies, cell by cell.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from blockgen.corpus import facade as fc

CANVAS = 32
FACTORS: Tuple[int, ...] = (16, 8, 4, 2, 1)       # coarse to fine
N_DEPTH = CANVAS + 1                               # 0 = see-through, 1 + d otherwise
UNK = 0


def grid_size(factor: int) -> int:
    return CANVAS // factor


@dataclass
class BlockVocab:
    """``(id, data)`` keys of the train faces, most common first; index 0 is unknown."""
    keys: np.ndarray            # [K] int32, key = id * 16 + data
    counts: np.ndarray          # [K] int64 train cell counts

    @classmethod
    def fit(cls, faces: Iterable[fc.Facade], max_size: int = 512, min_count: int = 2) -> "BlockVocab":
        c: Counter = Counter()
        for f in faces:
            k = (f.ids.astype(np.int32) * 16 + f.data.astype(np.int32))[f.mask]
            c.update(k.tolist())
        items = [(k, n) for k, n in c.most_common() if n >= min_count][:max_size]
        keys = np.array([k for k, _ in items], np.int32)
        counts = np.array([n for _, n in items], np.int64)
        return cls(keys=keys, counts=counts)

    @property
    def size(self) -> int:
        """Number of block classes including unknown."""
        return len(self.keys) + 1

    def index(self, ids: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Block classes of ``(ids, data)`` arrays; keys not in the vocabulary are ``UNK``."""
        key = ids.astype(np.int32) * 16 + data.astype(np.int32)
        order = np.argsort(self.keys)
        sorted_keys = self.keys[order]
        pos = np.searchsorted(sorted_keys, key)
        pos = np.clip(pos, 0, len(sorted_keys) - 1)
        hit = sorted_keys[pos] == key
        return np.where(hit, order[pos] + 1, UNK).astype(np.int32)

    def blocks(self, cls_: np.ndarray, fallback: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Block classes to ``(ids, data)``; unknown becomes ``fallback`` (default: the
        most common key)."""
        fb = int(self.keys[0]) if fallback is None else fallback
        key = np.where(cls_ > 0, self.keys[np.clip(cls_ - 1, 0, len(self.keys) - 1)], fb).astype(np.int32)
        return (key // 16).astype(np.int16), (key % 16).astype(np.int16)

    def to_dict(self) -> dict:
        return {"keys": self.keys.tolist(), "counts": self.counts.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "BlockVocab":
        return cls(keys=np.array(d["keys"], np.int32), counts=np.array(d["counts"], np.int64))


@dataclass
class Level:
    """One level of one face on the canvas grid. Arrays are ``(G, G)``."""
    factor: int
    depth: np.ndarray           # int64 depth class
    block: np.ndarray           # int64 block class
    inbox: np.ndarray           # bool


@dataclass
class FaceStack:
    face: str
    shape: Tuple[int, int]      # (U, V) of the real face
    extent: int
    levels: Dict[int, Level]    # factor -> Level

    @property
    def box(self) -> Tuple[int, int, int]:
        return (self.shape[0], self.shape[1], self.extent)


def inbox_grid(shape: Tuple[int, int], factor: int) -> np.ndarray:
    G = grid_size(factor)
    nu, nv = -(-shape[0] // factor), -(-shape[1] // factor)
    m = np.zeros((G, G), bool)
    m[:nu, :nv] = True
    return m


def encode(f: fc.Facade, vocab: BlockVocab, rule: str = "any",
           factors: Sequence[int] = FACTORS) -> FaceStack:
    """The level stack of a face (``U, V <= 32``) under a pooling rule."""
    U, V = f.depth.shape
    if U > CANVAS or V > CANVAS or f.extent > CANVAS:
        raise ValueError(f"face {f.depth.shape} x {f.extent} exceeds the {CANVAS} canvas")
    levels = {}
    for fac in factors:
        p = fc.pool(f, fac, rule)
        G = grid_size(fac)
        depth = np.zeros((G, G), np.int64)
        block = np.zeros((G, G), np.int64)
        nu, nv = p.depth.shape
        d = p.depth.astype(np.int64)
        depth[:nu, :nv] = np.where(d >= 0, 1 + np.minimum(d, CANVAS - 1), 0)
        block[:nu, :nv] = np.where(d >= 0, vocab.index(p.ids, p.data), 0)
        levels[fac] = Level(fac, depth, block, inbox_grid((U, V), fac))
    return FaceStack(face=f.face, shape=(U, V), extent=f.extent, levels=levels)


def decode(level: Level, stack_shape: Tuple[int, int], extent: int, face: str,
           vocab: BlockVocab) -> fc.Facade:
    """One level back to a face of ``stack_shape`` cells by repeating its cells."""
    fac = level.factor
    nu, nv = -(-stack_shape[0] // fac), -(-stack_shape[1] // fac)
    d = level.depth[:nu, :nv]
    depth = np.where(d > 0, np.minimum(d - 1, extent - 1), fc.EMPTY).astype(np.int16)
    ids, data = vocab.blocks(level.block[:nu, :nv])
    ids = np.where(d > 0, ids, 0).astype(np.int16)
    data = np.where(d > 0, data, 0).astype(np.int16)
    coarse = fc.Facade(face=face, depth=depth, ids=ids, data=data, extent=extent)
    return coarse if fac == 1 else fc.upsample(coarse, fac, stack_shape)


def parent(level_coarse: Level, factor_fine: int) -> Tuple[np.ndarray, np.ndarray]:
    """The coarser level's ``(depth, block)`` repeated onto the finer grid: what
    every fine cell is told about its parent."""
    r = level_coarse.factor // factor_fine
    rep = lambda a: np.repeat(np.repeat(a, r, 0), r, 1)
    return rep(level_coarse.depth), rep(level_coarse.block)


def face_from_arrays(depth_cls: np.ndarray, block_cls: np.ndarray, factor: int,
                     shape: Tuple[int, int], extent: int, face: str, vocab: BlockVocab) -> fc.Facade:
    """Convenience: a level's arrays (as sampled) to a face."""
    return decode(Level(factor, depth_cls, block_cls, inbox_grid(shape, factor)), shape, extent, face, vocab)


def side_faces(structures, faces: Sequence[str] = fc.SIDES) -> List[fc.Facade]:
    """The requested faces of every build that has blocks, dropping empty faces and
    faces larger than the canvas."""
    out = []
    for s in structures:
        if not s.occupied_mask.any():
            continue
        for face in faces:
            f = fc.project(s, face)
            if f.mask.any() and max(f.depth.shape) <= CANVAS and f.extent <= CANVAS:
                out.append(f)
    return out


def flip_u(f: fc.Facade) -> fc.Facade:
    """Mirror a face along ``u``: the one D4 element that keeps a side face a side
    face seen from outside (a reflection of the building)."""
    return fc.Facade(face=f.face, depth=np.ascontiguousarray(f.depth[::-1]),
                     ids=np.ascontiguousarray(f.ids[::-1]), data=np.ascontiguousarray(f.data[::-1]),
                     extent=f.extent)


def stack_arrays(faces: Sequence[fc.Facade], vocab: BlockVocab, rule: str = "any",
                 factors: Sequence[int] = FACTORS) -> Dict[str, object]:
    """Every face's stack as batched arrays: ``depth[f]``, ``block[f]``, ``inbox[f]`` of
    shape ``[N, G, G]`` (int16 / int16 / bool) per factor, ``box`` ``[N, 3]`` int16 and
    the face names. The training set in memory, a few tens of MB."""
    N = len(faces)
    depth = {f: np.zeros((N, grid_size(f), grid_size(f)), np.int16) for f in factors}
    block = {f: np.zeros((N, grid_size(f), grid_size(f)), np.int16) for f in factors}
    inbox = {f: np.zeros((N, grid_size(f), grid_size(f)), bool) for f in factors}
    box = np.zeros((N, 3), np.int16)
    names = []
    for i, face in enumerate(faces):
        st = encode(face, vocab, rule, factors)
        for f in factors:
            depth[f][i], block[f][i], inbox[f][i] = st.levels[f].depth, st.levels[f].block, st.levels[f].inbox
        box[i] = st.box
        names.append(st.face)
    return {"depth": depth, "block": block, "inbox": inbox, "box": box, "face": names}
