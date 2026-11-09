"""Non-learned facade generators: what a 2D-first model has to beat.

Each one draws a real bounding box from the train split, makes every face of it
in the facade representation (``blockgen.corpus.facade``) and lifts the faces into
the box, so its output is a build the benchmark scores like any other arm. They
are fitted on the train split only, as ``blockgen.eval.bench.baselines`` are, and
sit beside them rather than inside them so that ``--baselines`` keeps meaning
what every scorecard so far has meant.

    facade_unigram     every cell drawn from the train cell distribution of that
                       face type, no neighbours. The floor.
    facade_markov      every cell drawn given the cells to its left and below,
                       with backoff to the unigram. A 2D bigram, the smallest
                       model that knows a window has a sill.
    facade_patchwork   each face is a real train face of that type from a
                       *different* house, resampled to the box. Every face real,
                       the house nonsense: the facade twin of ``patchwork``.

A cell is ``None`` (see-through) or ``(depth, id, data)``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from blockgen.corpus import facade as fc
from blockgen.utils.data import Structure

Cell = Optional[Tuple[int, int, int]]
MAX_DEPTH = 47


def _cells(f: fc.Facade) -> np.ndarray:
    """Object array of cells, shape (u, v)."""
    out = np.empty(f.depth.shape, object)
    for u, v in np.ndindex(f.depth.shape):
        d = int(f.depth[u, v])
        out[u, v] = None if d < 0 else (min(d, MAX_DEPTH), int(f.ids[u, v]), int(f.data[u, v]))
    return out


def _facade(face: str, cells: np.ndarray, extent: int) -> fc.Facade:
    U, V = cells.shape
    depth = np.full((U, V), fc.EMPTY, np.int16)
    ids = np.zeros((U, V), np.int16); data = np.zeros((U, V), np.int16)
    for u, v in np.ndindex(U, V):
        c = cells[u, v]
        if c is not None:
            depth[u, v], ids[u, v], data[u, v] = min(c[0], extent - 1), c[1], c[2]
    return fc.Facade(face=face, depth=depth, ids=ids, data=data, extent=extent)


def _size(face: str, box: Tuple[int, int, int]) -> Tuple[int, int]:
    X, Y, Z = box
    axis = face[1]
    return (X, Y) if axis == "z" else (Z, Y) if axis == "x" else (X, Z)


def _extent(face: str, box: Tuple[int, int, int]) -> int:
    return box[{"x": 0, "y": 1, "z": 2}[face[1]]]


@dataclass
class FacadeStats:
    """Cell statistics of the train faces, per face type, plus the real boxes."""
    faces: Tuple[str, ...]
    boxes: np.ndarray                                   # [N, 3]
    unigram: Dict[str, Tuple[List[Cell], np.ndarray]]   # face -> (cells, probabilities)
    bigram: Dict[str, Dict[Hashable, Tuple[List[Cell], np.ndarray]]]
    real: Dict[str, List[np.ndarray]]                   # face -> cell arrays, one per train house

    @classmethod
    def fit(cls, train: Sequence[Structure], faces: Sequence[str] = fc.ALL_FACES) -> "FacadeStats":
        uni: Dict[str, Counter] = defaultdict(Counter)
        bi: Dict[str, Dict[Hashable, Counter]] = defaultdict(lambda: defaultdict(Counter))
        real: Dict[str, List[np.ndarray]] = defaultdict(list)
        boxes = []
        for s in train:
            c = s.crop_to_non_air()
            if not c.occupied_mask.any():
                continue
            boxes.append(c.shape)
            for name, f in fc.project_all(c, faces).items():
                cells = _cells(f)
                real[name].append(cells)
                U, V = cells.shape
                for u, v in np.ndindex(U, V):
                    cell = cells[u, v]
                    uni[name][cell] += 1
                    left = cells[u - 1, v] if u else "edge"
                    below = cells[u, v - 1] if v else "edge"
                    bi[name][(left, below)][cell] += 1

        def table(cnt: Counter):
            keys = list(cnt)
            p = np.array([cnt[k] for k in keys], float)
            return keys, p / p.sum()
        return cls(faces=tuple(faces), boxes=np.array(boxes, int),
                   unigram={k: table(v) for k, v in uni.items()},
                   bigram={k: {ctx: table(v) for ctx, v in d.items()} for k, d in bi.items()},
                   real=dict(real))

    def box(self, rng: np.random.Generator) -> Tuple[int, int, int]:
        return tuple(int(x) for x in self.boxes[rng.integers(len(self.boxes))])


def _draw(table: Tuple[List[Cell], np.ndarray], rng: np.random.Generator) -> Cell:
    keys, p = table
    return keys[rng.choice(len(keys), p=p)]


# --- the arms ---------------------------------------------------------------
def facade_unigram(st: FacadeStats, rng: np.random.Generator) -> Structure:
    box = st.box(rng)
    faces = []
    for face in st.faces:
        U, V = _size(face, box)
        cells = np.empty((U, V), object)
        for u, v in np.ndindex(U, V):
            cells[u, v] = _draw(st.unigram[face], rng)
        faces.append(_facade(face, cells, _extent(face, box)))
    return fc.lift(faces, box)


def facade_markov(st: FacadeStats, rng: np.random.Generator) -> Structure:
    box = st.box(rng)
    faces = []
    for face in st.faces:
        U, V = _size(face, box)
        cells = np.empty((U, V), object)
        for v in range(V):
            for u in range(U):
                ctx = (cells[u - 1, v] if u else "edge", cells[u, v - 1] if v else "edge")
                table = st.bigram[face].get(ctx) or st.unigram[face]
                cells[u, v] = _draw(table, rng)
        faces.append(_facade(face, cells, _extent(face, box)))
    return fc.lift(faces, box)


def _resample(cells: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    U, V = size
    iu = np.minimum((np.arange(U) * cells.shape[0] / U).astype(int), cells.shape[0] - 1)
    iv = np.minimum((np.arange(V) * cells.shape[1] / V).astype(int), cells.shape[1] - 1)
    return cells[np.ix_(iu, iv)]


def facade_patchwork(st: FacadeStats, rng: np.random.Generator) -> Structure:
    box = st.box(rng)
    n = min(len(v) for v in st.real.values())
    pick = rng.choice(n, size=len(st.faces), replace=n < len(st.faces))   # a different house per face
    faces = []
    for face, i in zip(st.faces, pick):
        cells = _resample(st.real[face][i], _size(face, box))
        faces.append(_facade(face, cells, _extent(face, box)))
    return fc.lift(faces, box)


ORDER = ("facade_unigram", "facade_markov", "facade_patchwork")

RECIPES: Dict[str, str] = {
    "facade_unigram":
        "every cell of every face drawn independently from the train split's cell "
        "distribution for that face type, lifted into a real bounding box; the floor "
        "of the facade track",
    "facade_markov":
        "every cell drawn given the cells to its left and below (a 2D bigram fitted "
        "on the train split, backing off to the unigram), lifted into a real box; the "
        "smallest model that knows a window has a sill",
    "facade_patchwork":
        "each face a real train face of that type from a different house, resampled "
        "to a real box: every face real and the house nonsense, the facade twin of "
        "patchwork",
}


def build(name: str, n: int, train: Sequence[Structure], seed: int = 0,
          faces: Sequence[str] = fc.ALL_FACES, stats: Optional[FacadeStats] = None) -> List[Structure]:
    rng = np.random.default_rng(seed)
    st = stats or FacadeStats.fit(train, faces)
    makers: Dict[str, Callable[[], Structure]] = {
        "facade_unigram": lambda: facade_unigram(st, rng),
        "facade_markov": lambda: facade_markov(st, rng),
        "facade_patchwork": lambda: facade_patchwork(st, rng),
    }
    if name not in makers:
        raise ValueError(f"unknown facade baseline {name!r}; have {sorted(makers)}")
    out = []
    for _ in range(n):
        s = makers[name]()
        s.metadata.update({"corpus": name, "faces": ",".join(faces)})
        out.append(s)
    return out


def arms(train: Sequence[Structure], n: int, seed: int = 0,
         faces: Sequence[str] = fc.ALL_FACES) -> Dict[str, List[Structure]]:
    st = FacadeStats.fit(train, faces)
    return {name: build(name, n, train, seed=seed + i, faces=faces, stats=st) for i, name in enumerate(ORDER)}
