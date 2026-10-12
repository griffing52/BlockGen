"""Coordinate-free node sets and their directed adjacency, from houses_32.

A :class:`Graph` is what the model sees and what it must reproduce:

* ``types``  [N] int   node type (block class, or 3D-BPE piece id). The only input.
* ``shapes`` list of [M,3] int offsets: the voxels a node occupies relative to its
  anchor. A voxel node is ``[(0,0,0)]``; a piece node is its BPE pattern. Shape is a
  property of the *type*, so it is not a coordinate leak.
* ``coords`` [N,3] int  anchor positions. Ground truth only; never fed to the model.

Edge label for an ordered pair ``(i, j)``: 0 if the two nodes do not touch, else the
class of the anchor offset ``coords[j] - coords[i]`` in a cube ``[-B, B]^3 \\ {0}``.
For voxel nodes this reduces to the 6 face directions (the shape-validity mask leaves
only unit offsets). The label of ``(j, i)`` is always the class of the negated offset,
so a labelled graph pins down relative geometry up to translation (per component).

Two node vocabularies:

* ``voxel``: one node per block, type = unoriented block token (``_token_for``),
  chunks are ``size^3`` windows (default 6) with 20..80 blocks (largest component).
* ``piece``: the window is re-tokenized with the houses_32 3D-BPE merges (exactly as
  ``scripts/probe_llm_attention.piece_instances`` does), type = piece id, chunks keep
  the largest touching component with 12..64 pieces.

Splits follow ``data/minecraft/splits/houses_32.s0.70-15-15.v1.json`` (grouped by
title), so no build contributes chunks to two splits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[3]
HOUSE_CACHE = REPO / "data" / "minecraft" / "cache" / "houses_32.npz"
SPLIT_FILE = REPO / "data" / "minecraft" / "splits" / "houses_32.s0.70-15-15.v1.json"
PIECE_VOCAB = REPO / "data" / "minecraftace" / "houses_32_bpe" / "houses_32_bpe_piece_vocab.json"
NBR = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])


# --------------------------------------------------------------------------- #
# label space: anchor offsets in a cube, with an antisymmetry permutation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OffsetSpace:
    """Classes 1..K are the offsets of ``[-B,B]^3 \\ {0}``; class 0 is "no contact"."""

    B: int

    @property
    def offsets(self) -> np.ndarray:            # [K, 3]
        r = np.arange(-self.B, self.B + 1)
        g = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
        return g[np.any(g != 0, axis=1)]

    @property
    def K(self) -> int:
        return (2 * self.B + 1) ** 3 - 1

    def cls(self, d: Sequence[int]) -> int:
        """Class index (1..K) of an offset, or -1 if outside the cube."""
        B = self.B
        if any(abs(int(v)) > B for v in d):
            return -1
        s = 2 * B + 1
        flat = (int(d[0]) + B) * s * s + (int(d[1]) + B) * s + (int(d[2]) + B)
        centre = B * s * s + B * s + B
        return flat + 1 if flat < centre else flat     # skip the zero offset

    @property
    def opp(self) -> np.ndarray:
        """[K+1] permutation: class of ``-d`` for every class (0 -> 0)."""
        out = np.zeros(self.K + 1, dtype=np.int64)
        for k, d in enumerate(self.offsets, start=1):
            out[k] = self.cls(-d)
        return out


def contact_offsets(shape_a: np.ndarray, shape_b: np.ndarray) -> set:
    """Anchor offsets ``d`` at which shape_b (at d) touches shape_a (at 0) without overlap."""
    a = {tuple(v) for v in np.asarray(shape_a).tolist()}
    cand = set()
    for va in a:
        for n in NBR.tolist():
            for vb in np.asarray(shape_b).tolist():
                cand.add((va[0] + n[0] - vb[0], va[1] + n[1] - vb[1], va[2] + n[2] - vb[2]))
    bad = {(va[0] - vb[0], va[1] - vb[1], va[2] - vb[2])
           for va in a for vb in np.asarray(shape_b).tolist()}
    return cand - bad


class ValidTable:
    """Lazy cache: ``mask(ta, tb)`` -> [K+1] bool of labels possible for a type pair.

    Class 0 (no contact) is always valid. This is what the type's shape implies about
    which relative placements are even physically possible, and is used both in the
    loss and at decode time.
    """

    def __init__(self, shapes_by_type: Dict[int, np.ndarray], space: OffsetSpace):
        self.shapes = shapes_by_type
        self.space = space
        self._cache: Dict[Tuple[int, int], np.ndarray] = {}

    def mask(self, ta: int, tb: int) -> np.ndarray:
        key = (int(ta), int(tb))
        m = self._cache.get(key)
        if m is None:
            m = np.zeros(self.space.K + 1, dtype=bool)
            m[0] = True
            for d in contact_offsets(self.shapes[key[0]], self.shapes[key[1]]):
                c = self.space.cls(d)
                if c > 0:
                    m[c] = True
            self._cache[key] = m
        return m


# --------------------------------------------------------------------------- #
# graphs
# --------------------------------------------------------------------------- #
@dataclass
class Graph:
    types: np.ndarray                         # [N] int
    coords: np.ndarray                        # [N, 3] int anchor (ground truth only)
    blocks: List[List[Tuple[int, int, int, int, int]]]   # per node (dx,dy,dz,id,data)
    build: int = -1
    source: str = ""
    extra: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.types)


def touching(coords: np.ndarray, shapes: Sequence[np.ndarray]) -> np.ndarray:
    """[N,N] bool: node i has a voxel face-adjacent to a voxel of node j."""
    owner: Dict[Tuple[int, int, int], int] = {}
    for i, (c, sh) in enumerate(zip(coords, shapes)):
        for v in (np.asarray(sh) + c).tolist():
            owner[tuple(v)] = i
    N = len(coords)
    adj = np.zeros((N, N), dtype=bool)
    for i, (c, sh) in enumerate(zip(coords, shapes)):
        for v in (np.asarray(sh) + c).tolist():
            for n in NBR.tolist():
                j = owner.get((v[0] + n[0], v[1] + n[1], v[2] + n[2]))
                if j is not None and j != i:
                    adj[i, j] = adj[j, i] = True
    return adj


def edge_labels(coords: np.ndarray, shapes: Sequence[np.ndarray], space: OffsetSpace
                ) -> Tuple[np.ndarray, int]:
    """[N,N] int labels (0 none, 1..K offset class); also the count of contacts whose
    offset falls outside the cube (labelled 0, reported, should be ~0)."""
    adj = touching(coords, shapes)
    N = len(coords)
    lab = np.zeros((N, N), dtype=np.int64)
    dropped = 0
    for i, j in zip(*np.nonzero(adj)):
        c = space.cls(coords[j] - coords[i])
        if c < 0:
            dropped += 1
        else:
            lab[i, j] = c
    return lab, dropped // 2


def components(adj: np.ndarray) -> List[List[int]]:
    N = len(adj)
    seen = np.zeros(N, bool)
    out = []
    for s in range(N):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.nonzero(adj[u])[0]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
        out.append(comp)
    return out


# --------------------------------------------------------------------------- #
# vocabularies
# --------------------------------------------------------------------------- #
@dataclass
class NodeVocab:
    kind: str                                 # "voxel" | "piece"
    n_types: int
    shapes: Dict[int, np.ndarray]             # type -> [M,3] offsets
    names: List[str]
    space: OffsetSpace
    token_to_type: Dict[str, int] = field(default_factory=dict)   # voxel only

    def to_json(self) -> dict:
        return {"kind": self.kind, "n_types": self.n_types, "names": self.names,
                "B": self.space.B, "token_to_type": self.token_to_type,
                "shapes": {str(k): v.tolist() for k, v in self.shapes.items()}}

    @classmethod
    def from_json(cls, d: dict) -> "NodeVocab":
        return cls(kind=d["kind"], n_types=d["n_types"], names=d["names"],
                   space=OffsetSpace(d["B"]), token_to_type=d.get("token_to_type", {}),
                   shapes={int(k): np.asarray(v, dtype=np.int64) for k, v in d["shapes"].items()})


@lru_cache(maxsize=1)
def load_houses():
    from blockgen.curation.houses import load_structures_from_cache
    structures, _ = load_structures_from_cache(str(HOUSE_CACHE))
    split = json.loads(SPLIT_FILE.read_text())
    return structures, {k: list(split[k]) for k in ("train", "val", "test")}


@lru_cache(maxsize=1)
def load_cluster_vocab():
    from blockgen.export.minecraftace import load_piece_vocab
    return load_piece_vocab(str(PIECE_VOCAB))


def _base_vocab(cv):
    from blockgen.utils.data import _token_for
    from blockgen.utils.serialize import BlockVocab
    return BlockVocab(
        max_dim=cv.max_dim,
        block_token_to_id={_token_for(*p, oriented=cv.oriented): i
                           for i, p in enumerate(cv.block_index_to_pair)},
        id_to_block_token=[_token_for(*p, oriented=cv.oriented) for p in cv.block_index_to_pair],
        block_index_to_pair=cv.block_index_to_pair, oriented=cv.oriented)


def piece_vocab(B: int = 4) -> NodeVocab:
    cv = load_cluster_vocab()
    shapes = {pid: np.array([c[:3] for c in pat], dtype=np.int64)
              for pid, pat in enumerate(cv.patterns)}
    names = []
    for pid, pat in enumerate(cv.patterns):
        bidx = [c[3] for c in pat]
        maj = max(set(bidx), key=bidx.count)
        names.append(f"{cv.block_index_to_pair[maj][0]}:{cv.block_index_to_pair[maj][1]}x{len(pat)}")
    return NodeVocab(kind="piece", n_types=cv.n_pieces, shapes=shapes, names=names,
                     space=OffsetSpace(B))


def voxel_token(bid: int, bdata: int) -> str:
    from blockgen.utils.data import _token_for
    return _token_for(int(bid), int(bdata))


def voxel_vocab(train_graph_tokens: Sequence[Sequence[str]], max_types: int = 255) -> NodeVocab:
    """Type 0 = <unk>; the most frequent ``max_types`` tokens over the train chunks."""
    from collections import Counter
    cnt = Counter(t for g in train_graph_tokens for t in g)
    keep = [t for t, _ in cnt.most_common(max_types)]
    tok2 = {t: i + 1 for i, t in enumerate(keep)}
    n = len(keep) + 1
    return NodeVocab(kind="voxel", n_types=n, names=["<unk>"] + keep,
                     shapes={i: np.zeros((1, 3), np.int64) for i in range(n)},
                     space=OffsetSpace(1), token_to_type=tok2)


# --------------------------------------------------------------------------- #
# chunk extraction
# --------------------------------------------------------------------------- #
def _largest_cc(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import label
    lab, n = label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())[1:]
    return lab == (1 + int(np.argmax(sizes)))


def voxel_chunks_raw(build_ids: Sequence[int], per_build: int, size: int = 6,
                     min_n: int = 20, max_n: int = 80, seed: int = 0, tries: int = 60):
    """Yield raw voxel chunks: (build, coords [N,3], ids [N], data [N], origin)."""
    structures, _ = load_houses()
    rng = np.random.default_rng(seed)
    out = []
    for b in build_ids:
        s = structures[b]
        occ = s.occupied_mask
        if min(occ.shape) < size:
            continue
        got, seen = 0, set()
        for _ in range(tries):
            if got >= per_build:
                break
            o = tuple(int(rng.integers(0, d - size + 1)) for d in occ.shape)
            if o in seen:
                continue
            seen.add(o)
            x, y, z = o
            win = _largest_cc(occ[x:x + size, y:y + size, z:z + size])
            n = int(win.sum())
            if not (min_n <= n <= max_n):
                continue
            loc = np.argwhere(win)
            g = loc + np.array(o)
            ids = s.block_ids[g[:, 0], g[:, 1], g[:, 2]].astype(np.int64)
            dat = s.block_data[g[:, 0], g[:, 1], g[:, 2]].astype(np.int64)
            out.append((b, loc - loc.min(0), ids, dat, o))
            got += 1
    return out


def voxel_graphs(raw, vocab: NodeVocab) -> List[Graph]:
    gs = []
    for b, loc, ids, dat, o in raw:
        types = np.array([vocab.token_to_type.get(voxel_token(i, d), 0) for i, d in zip(ids, dat)])
        blocks = [[(0, 0, 0, int(i), int(d))] for i, d in zip(ids, dat)]
        gs.append(Graph(types=types, coords=loc.astype(np.int64), blocks=blocks, build=int(b),
                        source=f"h{b:05d}@{o}"))
    return gs


def tokenize_window(s, cv) -> Optional[List[dict]]:
    """3D-BPE piece instances of a Structure (replays the learned merges)."""
    from blockgen.tokenizers.cluster_bpe import _apply_merge, _atomic_labeling
    inst, owner, nid = _atomic_labeling(s, _base_vocab(cv))
    if not inst:
        return None
    for pid_new, (pa, pb, delta) in enumerate(cv.merges, start=cv.num_blocks):
        nid = _apply_merge(inst, owner, nid, pa, pb, delta, pid_new)
    return list(inst.values())


def piece_graphs(build_ids: Sequence[int], per_build: int, vocab: NodeVocab, size: int = 8,
                 min_n: int = 12, max_n: int = 64, seed: int = 0, tries: int = 40,
                 stats: Optional[dict] = None) -> List[Graph]:
    """Windows of ``size^3`` re-tokenized into pieces; keep the largest touching
    component when it has ``min_n..max_n`` pieces."""
    from blockgen.utils.data import Structure
    structures, _ = load_houses()
    cv = load_cluster_vocab()
    out: List[Graph] = []
    stats = stats if stats is not None else {}
    for b in build_ids:
        rng = np.random.default_rng(seed * 100003 + int(b))
        s = structures[b]
        occ = s.occupied_mask
        if min(occ.shape) < 3:
            continue
        got, seen = 0, set()
        for _ in range(tries):
            if got >= per_build:
                break
            o = tuple(int(rng.integers(0, max(1, d - size + 1))) for d in occ.shape)
            if o in seen:
                continue
            seen.add(o)
            x, y, z = o
            sub = Structure(block_ids=s.block_ids[x:x + size, y:y + size, z:z + size].copy(),
                            block_data=s.block_data[x:x + size, y:y + size, z:z + size].copy())
            if sub.occupied_mask.sum() < min_n:
                continue
            inst = tokenize_window(sub, cv)
            if inst is None or len(inst) < min_n:
                continue
            coords = np.array([d["anchor"] for d in inst], dtype=np.int64)
            shapes = [vocab.shapes[d["piece"]] for d in inst]
            comp = max(components(touching(coords, shapes)), key=len)
            if not (min_n <= len(comp) <= max_n):
                continue
            comp = sorted(comp)
            inst = [inst[i] for i in comp]
            coords = coords[comp]
            coords = coords - coords.min(0)
            types = np.array([d["piece"] for d in inst], dtype=np.int64)
            blocks = []
            for d in inst:
                pat = cv.patterns[d["piece"]]
                blocks.append([(dx, dy, dz, *cv.block_index_to_pair[bi]) for dx, dy, dz, bi in pat])
            g = Graph(types=types, coords=coords, blocks=blocks, build=int(b),
                      source=f"h{b:05d}@{o}",
                      extra={"n_voxels": int(sum(len(d["vox"]) for d in inst))})
            _, dropped = edge_labels(coords, [vocab.shapes[t] for t in types], vocab.space)
            stats["dropped_edges"] = stats.get("dropped_edges", 0) + dropped
            stats["edges"] = stats.get("edges", 0) + int(touching(coords, [vocab.shapes[t] for t in types]).sum() // 2)
            out.append(g)
            got += 1
    return out


def _piece_worker(args):
    ids, per_build, B, size, seed = args
    st: dict = {}
    gs = piece_graphs(ids, per_build, piece_vocab(B), size=size, seed=seed, stats=st)
    return gs, st


def piece_graphs_parallel(build_ids: Sequence[int], per_build: int, vocab: NodeVocab,
                          size: int = 8, seed: int = 0, workers: int = 8,
                          stats: Optional[dict] = None) -> List[Graph]:
    """Same as :func:`piece_graphs` (per-build seeding, so identical output), forked."""
    import multiprocessing as mp
    load_houses(); load_cluster_vocab()
    ids = list(build_ids)
    parts = [ids[k::workers * 4] for k in range(workers * 4)]
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_piece_worker, [(p, per_build, vocab.space.B, size, seed) for p in parts])
    out = [g for gs, _ in res for g in gs]
    out.sort(key=lambda g: (g.build, g.source))
    if stats is not None:
        for _, st in res:
            for k, v in st.items():
                stats[k] = stats.get(k, 0) + v
    return out


def graph_to_structure(coords: np.ndarray, blocks, keep: Optional[Sequence[int]] = None):
    """Realize placed nodes as a Structure (later nodes lose a cell collision)."""
    from blockgen.utils.data import Structure
    idx = range(len(blocks)) if keep is None else keep
    cells = {}
    for i in idx:
        c = coords[i]
        for dx, dy, dz, bid, bd in blocks[i]:
            cells.setdefault((int(c[0] + dx), int(c[1] + dy), int(c[2] + dz)), (bid, bd))
    if not cells:
        return Structure(block_ids=np.zeros((1, 1, 1), np.int32), block_data=np.zeros((1, 1, 1), np.int32))
    arr = np.array(list(cells.keys()))
    arr = arr - arr.min(0)
    shp = tuple(int(v) + 1 for v in arr.max(0))
    bi = np.zeros(shp, np.int32)
    bd = np.zeros(shp, np.int32)
    for (x, y, z), (b, d) in zip(arr.tolist(), cells.values()):
        bi[x, y, z], bd[x, y, z] = b, d
    return Structure(block_ids=bi, block_data=bd)


def build_dataset(kind: str, n_train_builds: Optional[int] = None, per_build_train: int = 24,
                  per_build_eval: int = 2, n_eval_builds: int = 200, seed: int = 0,
                  B: int = 4, voxel_size: int = 6, piece_size: int = 8, workers: int = 8) -> dict:
    """{"vocab", "train", "val", "test", "stats"}; val/test = few chunks per held-out build."""
    _, split = load_houses()
    tr = split["train"][:n_train_builds] if n_train_builds else split["train"]
    va = split["val"][:n_eval_builds]
    te = split["test"][:n_eval_builds]
    stats: dict = {}
    if kind == "voxel":
        raw_tr = voxel_chunks_raw(tr, per_build_train, size=voxel_size, seed=seed)
        vocab = voxel_vocab([[voxel_token(i, d) for i, d in zip(r[2], r[3])] for r in raw_tr])
        out = {"train": voxel_graphs(raw_tr, vocab),
               "val": voxel_graphs(voxel_chunks_raw(va, per_build_eval, size=voxel_size, seed=seed + 1), vocab),
               "test": voxel_graphs(voxel_chunks_raw(te, per_build_eval, size=voxel_size, seed=seed + 2), vocab)}
    elif kind == "piece":
        vocab = piece_vocab(B)
        out = {"train": piece_graphs_parallel(tr, per_build_train, vocab, size=piece_size, seed=seed,
                                              stats=stats, workers=workers),
               "val": piece_graphs_parallel(va, per_build_eval, vocab, size=piece_size, seed=seed + 1,
                                            workers=workers),
               "test": piece_graphs_parallel(te, per_build_eval, vocab, size=piece_size, seed=seed + 2,
                                             workers=workers)}
    else:
        raise ValueError(kind)
    out["vocab"] = vocab
    stats.update({k: len(out[k]) for k in ("train", "val", "test")})
    stats["nodes_median"] = {k: float(np.median([len(g) for g in out[k]])) for k in ("train", "val", "test")}
    out["stats"] = stats
    return out
