"""The corpus tokenizer for the sparse raster arm (arm A), as a versioned artifact.

One fixed vocabulary for every tier, so a checkpoint, a shard and a sample all
speak the same ids and the tokenizer hash in a checkpoint says exactly which:

    0..2                       PAD, BOS, EOS                (``blockgen.utils.serialize``)
    3 .. 3+max_dim-1           coordinate tokens, shared across x/y/z (max_dim 48)
    block_offset ..            block classes: every ``id`` / ``id:data`` class the
                               standard vocabulary knows (441 + ``<unk>``), fixed,
                               not derived from a dataset
    tag_offset ..              prefix tags (docs/pipeline.md §3.4):
                               src:{archive,curated,vanilla,templates}
                               q:{0..4}  bucket of p_built (curated builds are 4)
                               size:{16,24,32,48}  window edge
                               win:{whole,window}  whole build, or a window of a bigger one

A sequence is ``[BOS, src, q, size, win, (X, Y, Z, BLOCK)*, EOS]`` in (y, z, x)
raster order, the same body as ``structure_to_tokens``. The block and coordinate
ranges are a ``BlockVocab``, so every decoder in the repo still works on the body.

The ``facades`` kind (``CorpusTokenizer.default(kind="facades")``) reads a build
as its face images (``blockgen.corpus.facade``) and is a 2D problem: one sequence
per face, ``[BOS, src, q, size, win, face:<f>, U, V, cell*, EOS]`` with ``U`` and
``V`` as coordinate tokens and exactly ``U*V`` cells in scan order (``v`` outer,
``u`` inner); a cell is ``cell:empty`` where the column is see-through and
``(DEPTH, BLOCK)`` otherwise, with depth as a coordinate token. Two more tag
groups (``face`` and ``cell``) are the only difference in vocabulary, and only a
non-default kind is written into the artifact, so the voxel hash is unchanged.

Its ``joint`` layout (``default(kind="facades", layout="joint")``) puts every face
of a build in one sequence, ``[BOS, tags, (face, U, V, cell*)+, EOS]``, in the
shard's face order (the sides in walking order, then roof plan and underside), so
each face is generated in sight of the ones before it; ``per_face`` is one face per
sequence, each face independent. The layout is written into the artifact when it
is ``joint``, so the two have different hashes and different shards.

The ``pooled4`` layout (``default(kind="facades", layout="pooled4")``) is the
cascade-and-AR hybrid of notes §74: one face per sequence, preceded by its own
level f=4 (``facade.pool(f, 4, "any")``, a ceil(U/4) x ceil(V/4) face in the same
cell grammar), ``[BOS, tags, face, U/4, V/4, cell*, face, U, V, cell*, EOS]``. The
model draws the gestalt first and the fine face in sight of it; a sampler that
forces the real coarse level is the f=4 oracle. ``decode_face`` returns the fine face.

The ``pieces`` kind is 3D-BPE (``blockgen.tokenizers.cluster_bpe``): a build is
re-tokenised by replaying merges learned once from ``sft-houses32``
(``learn_pieces``), and the body is ``(X, Y, Z, PIECE)*`` with ``X, Y, Z`` the
piece's anchor (min corner) in (y, z, x) order of the anchors. A single-block
piece is the voxel kind's block token; the learned pieces sit after the tags
(``piece_offset``), so tags, blocks and coordinates keep the voxel ids. The
merges are written into the artifact, so the hash names the piece vocabulary.

Category conditioning (``default(..., cats=True)``, the ``voxels-cat`` variant when
the kind is voxels) adds one prompt class per sequence, ``cat:<c>`` for ``c`` in
``CAT_CLASSES`` (``blockgen.corpus.labels.CLASSES``, ``cat:none`` for an unlabeled
build). The class token takes the place of BOS::

    [CAT, src, q, size, win, (X, Y, Z, BLOCK)*, EOS]

so every sequence keeps its length and every body token its absolute position:
the ``phase4`` positional encoding (pos mod 4, pos div 4) sees the body exactly
where an unconditioned model saw it, a conditioned house has the same NLL
denominator as an unconditioned one, and a class token acts as a class-specific
start token (the class-conditional GPT recipe). The class ids are appended after
everything the base tokenizer has (``cat_offset = vocab_size`` of the base, after
the learned pieces for that kind), so every existing id is unchanged and a
pretrained checkpoint extends to the new vocabulary by appending embedding rows
(``blockgen.train.trainer.extend_vocab``). The class list is written into the
artifact only when present, so every existing hash is unchanged, and the
mechanism is the same for every kind (the face tag of a facade sequence and the
pieces of a pieces sequence follow the shared tags as before).

``hash`` is the first 12 hex digits of the sha256 of the saved JSON, and names the
shard directory and the checkpoint field.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from blockgen.corpus import facade as fc
from blockgen.utils.data import Structure, _token_for
from blockgen.utils.serialize import BOS_TOKEN, EOS_TOKEN, BlockVocab
from blockgen.tokenizers.standard_vocab import STANDARD_VOCAB

MAX_DIM = 48
UNK = "<unk>"
TAG_GROUPS: Dict[str, List[str]] = {
    "src": ["archive", "curated", "vanilla", "templates"],
    "q": ["0", "1", "2", "3", "4"],
    "size": ["16", "24", "32", "48"],
    "win": ["whole", "window"],
}
TAG_ORDER = ["src", "q", "size", "win"]
N_TAGS = len(TAG_ORDER)
#: Extra tag groups of the ``facades`` kind, after the four shared ones.
FACADE_TAG_GROUPS: Dict[str, List[str]] = {"face": list(fc.ALL_FACES), "cell": ["empty"]}
KINDS = ("voxels", "facades", "pieces")
#: Where ``learn-pieces`` writes a learned tokenizer, as ``<hash>.json``.
TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             "data", "minecraft", "corpus", "tokenizers")
LAYOUTS = ("per_face", "joint", "pooled4")
#: The category tag group (``cats=True``): fixed, not derived from a dataset, so the hash
#: does not move when the labels do. Mirrors ``blockgen.corpus.labels.CLASSES``.
CAT_GROUP = "cat"
#: The coarse level the ``pooled4`` layout puts before a face.
POOL_FACTOR = 4
CAT_CLASSES: Tuple[str, ...] = ("house", "castle", "tower", "church", "shop", "farm", "ship", "vehicle",
                                "aircraft", "statue", "tree", "pixelart", "none")


def fixed_block_tokens(oriented: bool = False) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Every block class the standard vocabulary can produce, plus ``<unk>``."""
    seen: Dict[str, Tuple[int, int]] = {}
    known_ids = {int(k.split(":")[0]) for k in STANDARD_VOCAB}
    for bid in sorted(known_ids):
        if bid == 0:
            continue
        for data in range(16):
            tok = _token_for(bid, data, oriented=oriented)
            seen.setdefault(tok, (bid, data))
    toks = sorted(seen)
    toks.append(UNK)
    pairs = [seen[t] for t in toks[:-1]] + [(1, 0)]
    return toks, pairs


@dataclass
class CorpusTokenizer:
    vocab: BlockVocab
    tags: Dict[str, List[str]] = field(default_factory=lambda: {k: list(v) for k, v in TAG_GROUPS.items()})
    version: int = 1
    kind: str = "voxels"
    layout: str = "per_face"     # facades only: one face per sequence, or every face of a build in one
    pieces: Optional[dict] = None  # pieces only: learned merges and how they were learned (``learn_pieces``)
    cats: Optional[List[str]] = None  # category conditioning: the cat:<c> classes, ids after everything else

    @property
    def tag_order(self) -> Tuple[str, ...]:
        """Groups in prefix order: the four shared ones, then the kind's own."""
        return tuple(self.tags)

    # --- layout ---------------------------------------------------------
    @property
    def tag_offset(self) -> int:
        return self.vocab.block_offset + self.vocab.num_blocks

    @property
    def n_tags(self) -> int:
        return sum(len(v) for v in self.tags.values())

    @property
    def piece_offset(self) -> int:
        """First learned piece token (pieces kind); single-block pieces are block tokens."""
        return self.tag_offset + self.n_tags

    @property
    def n_learned(self) -> int:
        return len(self.pieces["merges"]) if self.kind == "pieces" and self.pieces else 0

    @property
    def cat_offset(self) -> int:
        """First ``cat:<c>`` token: the base vocabulary's size, so no existing id moves."""
        return self.piece_offset + self.n_learned

    @property
    def n_cats(self) -> int:
        return len(self.cats) if self.cats else 0

    @property
    def vocab_size(self) -> int:
        return self.cat_offset + self.n_cats

    def base(self) -> "CorpusTokenizer":
        """The same tokenizer without the category group: the one a pretrained checkpoint used."""
        return CorpusTokenizer(vocab=self.vocab, tags={k: list(v) for k, v in self.tags.items()},
                               version=self.version, kind=self.kind, layout=self.layout, pieces=self.pieces)

    def extends(self, other: "CorpusTokenizer") -> bool:
        """True when ``self`` is ``other`` plus appended category ids (every id of ``other``
        means the same here), so ``other``'s weights extend to this vocabulary."""
        return other.cats is None and self.base().hash == other.hash

    @property
    def max_dim(self) -> int:
        return self.vocab.max_dim

    def tag_id(self, group: str, value) -> int:
        if group == CAT_GROUP and self.cats:
            return self.cat_offset + self.cats.index(str(value))
        off = self.tag_offset
        for g in self.tag_order:
            vals = self.tags[g]
            if g == group:
                return off + vals.index(str(value))
            off += len(vals)
        raise KeyError(group)

    def decode_tag(self, tok: int) -> Optional[Tuple[str, str]]:
        if self.cats and self.cat_offset <= tok < self.cat_offset + self.n_cats:
            return CAT_GROUP, self.cats[tok - self.cat_offset]
        off = self.tag_offset
        for g in self.tag_order:
            vals = self.tags[g]
            if off <= tok < off + len(vals):
                return g, vals[tok - off]
            off += len(vals)
        return None

    # --- persistence ----------------------------------------------------
    def to_json(self) -> dict:
        d = {"version": self.version, "max_dim": self.vocab.max_dim, "oriented": self.vocab.oriented,
             "block_tokens": self.vocab.id_to_block_token,
             "block_pairs": [list(p) for p in self.vocab.block_index_to_pair],
             "tags": {g: self.tags[g] for g in self.tag_order}}
        if self.kind != "voxels":      # absent for voxels, so the voxel hash is the one every shard has
            d["kind"] = self.kind
        if self.layout != "per_face":
            d["layout"] = self.layout
        if self.kind == "pieces":
            d["pieces"] = self.pieces or {"merges": []}
        if self.cats:
            d["cats"] = list(self.cats)
        return d

    @property
    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True).encode()).hexdigest()[:12]

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        blob = self.to_json(); blob["hash"] = self.hash
        json.dump(blob, open(path, "w"))
        return path

    @classmethod
    def load(cls, path: str) -> "CorpusTokenizer":
        b = json.load(open(path))
        toks = b["block_tokens"]
        vocab = BlockVocab(max_dim=b["max_dim"], block_token_to_id={t: i for i, t in enumerate(toks)},
                           id_to_block_token=list(toks), block_index_to_pair=[tuple(p) for p in b["block_pairs"]],
                           oriented=b.get("oriented", False))
        return cls(vocab=vocab, tags=b["tags"], version=b.get("version", 1), kind=b.get("kind", "voxels"),
                   layout=b.get("layout", "per_face"), pieces=b.get("pieces"), cats=b.get("cats"))

    @classmethod
    def default(cls, oriented: bool = False, max_dim: int = MAX_DIM, kind: str = "voxels",
                layout: str = "per_face", cats=None) -> "CorpusTokenizer":
        """The fixed tokenizer of ``kind``; ``cats=True`` (or a class list) adds the category group."""
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        if layout not in LAYOUTS or (layout != "per_face" and kind != "facades"):
            raise ValueError(f"layout {layout!r} is not one of {LAYOUTS} for kind {kind!r}")
        toks, pairs = fixed_block_tokens(oriented)
        vocab = BlockVocab(max_dim=max_dim, block_token_to_id={t: i for i, t in enumerate(toks)},
                           id_to_block_token=toks, block_index_to_pair=pairs, oriented=oriented)
        tags = {k: list(v) for k, v in TAG_GROUPS.items()}
        if kind == "facades":
            tags.update({k: list(v) for k, v in FACADE_TAG_GROUPS.items()})
        # pieces: no merges yet (every piece a single block); ``learn_pieces`` fills them in
        return cls(vocab=vocab, tags=tags, kind=kind, layout=layout,
                   pieces={"merges": []} if kind == "pieces" else None,
                   cats=(list(CAT_CLASSES) if cats is True else list(cats)) if cats else None)

    def with_cats(self, cats=True) -> "CorpusTokenizer":
        """This tokenizer (any kind, learned pieces included) plus the category group."""
        return CorpusTokenizer(vocab=self.vocab, tags={k: list(v) for k, v in self.tags.items()},
                               version=self.version, kind=self.kind, layout=self.layout, pieces=self.pieces,
                               cats=list(CAT_CLASSES) if cats is True else list(cats))

    def head(self, tags: Dict[str, str]) -> List[int]:
        """``[BOS, src, q, size, win]``, or ``[CAT, src, q, size, win]`` with the category group
        (``tags["cat"]``, default ``none``): the class token is the start token."""
        first = self.tag_id(CAT_GROUP, tags.get(CAT_GROUP, "none")) if self.cats else BOS_TOKEN
        return [first] + [self.tag_id(g, tags[g]) for g in TAG_ORDER]

    # --- encode / decode -------------------------------------------------
    def block_lut(self) -> np.ndarray:
        """[256, 16] -> block token id (absolute), for vectorised encoding."""
        lut = np.full((256, 16), self.vocab.block_id_token(self.vocab.block_token_to_id[UNK]), np.int32)
        for bid in range(1, 256):
            for data in range(16):
                tok = _token_for(bid, data, oriented=self.vocab.oriented)
                i = self.vocab.block_token_to_id.get(tok)
                if i is not None:
                    lut[bid, data] = self.vocab.block_id_token(i)
        return lut

    def encode(self, ids: np.ndarray, data: np.ndarray, tags: Dict[str, str], lut: Optional[np.ndarray] = None) -> np.ndarray:
        """``ids``/``data`` XYZ arrays already cropped to the window (max edge <= max_dim)."""
        assert max(ids.shape) <= self.vocab.max_dim, ids.shape
        lut = self.block_lut() if lut is None else lut
        if self.kind == "pieces":
            return self._encode_pieces(ids, data, tags, lut)
        coords = np.argwhere(ids != 0)
        order = np.lexsort((coords[:, 0], coords[:, 2], coords[:, 1]))
        coords = coords[order]
        bid = np.clip(ids[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.int64), 0, 255)
        bdat = np.clip(data[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.int64), 0, 15)
        body = np.empty((len(coords), 4), np.int32)
        body[:, :3] = coords + self.vocab.coord_offset
        body[:, 3] = lut[bid, bdat]
        head = self.head(tags)
        return np.concatenate([np.array(head, np.int32), body.reshape(-1), np.array([EOS_TOKEN], np.int32)]).astype(np.uint16)

    def decode(self, tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
        """Inverse of ``encode``: (ids, data, tags); unknown blocks come back as stone."""
        toks = [int(t) for t in np.asarray(tokens).reshape(-1)]
        tags: Dict[str, str] = {}
        i = 0
        if toks and toks[0] == BOS_TOKEN:
            i = 1
        while i < len(toks) and self.decode_tag(toks[i]) is not None:
            g, v = self.decode_tag(toks[i]); tags[g] = v; i += 1
        if self.kind == "pieces":
            return (*self._decode_pieces(toks[i:]), tags)
        body = []
        while i + 3 < len(toks) and toks[i] != EOS_TOKEN:
            x, y, z, b = toks[i:i + 4]
            if self.vocab.is_coord(x) and self.vocab.is_coord(y) and self.vocab.is_coord(z) and self.vocab.is_block(b):
                body.append((self.vocab.decode_coord(x), self.vocab.decode_coord(y), self.vocab.decode_coord(z),
                             *self.vocab.decode_block(b)))
            i += 4
        if not body:
            return np.zeros((1, 1, 1), np.uint16), np.zeros((1, 1, 1), np.uint8), tags
        arr = np.array(body, np.int64)
        shape = tuple(arr[:, :3].max(0) + 1)
        ids = np.zeros(shape, np.uint16); data = np.zeros(shape, np.uint8)
        ids[arr[:, 0], arr[:, 1], arr[:, 2]] = arr[:, 3]
        data[arr[:, 0], arr[:, 1], arr[:, 2]] = arr[:, 4]
        return ids, data, tags

    # --- the pieces kind -------------------------------------------------
    def cluster_vocab(self):
        """The learned merges as a ``ClusterVocab`` (patterns rebuilt from the merges)."""
        cv = getattr(self, "_cv", None)
        if cv is None:
            from blockgen.tokenizers.cluster_bpe import ClusterVocab, _reanchor
            nb = self.vocab.num_blocks
            patterns = [((0, 0, 0, b),) for b in range(nb)]
            merges = []
            for pa, pb, d in (self.pieces or {}).get("merges", []):
                d = tuple(d)
                patterns.append(_reanchor(list(patterns[pa]) + [(x + d[0], y + d[1], z + d[2], bb)
                                                                for x, y, z, bb in patterns[pb]]))
                merges.append((int(pa), int(pb), d))
            cv = ClusterVocab(max_dim=self.vocab.max_dim, patterns=patterns,
                              block_index_to_pair=list(self.vocab.block_index_to_pair), merges=merges,
                              num_blocks=nb, oriented=self.vocab.oriented)
            self._cv = cv
        return cv

    def canonical(self, ids: np.ndarray, data: np.ndarray, lut: Optional[np.ndarray] = None) -> Structure:
        """The build as the voxel kind would decode it: every block mapped to its class's
        (id, data) pair, unknown blocks to stone, so a piece covers what a block token would."""
        lut = self.block_lut() if lut is None else lut
        pairs = np.asarray(self.vocab.block_index_to_pair, np.int32)
        occ = ids != 0
        idx = lut[np.clip(ids, 0, 255).astype(np.int64), np.clip(data, 0, 15).astype(np.int64)] - self.vocab.block_offset
        bi = np.where(occ, pairs[idx, 0], 0).astype(np.int32)
        bd = np.where(occ, pairs[idx, 1], 0).astype(np.int32)
        return Structure(block_ids=bi, block_data=bd)

    def _encode_pieces(self, ids, data, tags, lut) -> np.ndarray:
        from blockgen.tokenizers.cluster_bpe import structure_to_cluster_tokens
        cv = self.cluster_vocab()
        body = structure_to_cluster_tokens(self.canonical(ids, data, lut), cv)[1:-1]
        out = self.head(tags)
        nb, boff = cv.num_blocks, self.vocab.block_offset
        for t in body:
            if cv.is_piece(t):
                p = cv.decode_piece(t)
                t = boff + p if p < nb else self.piece_offset + p - nb
            out.append(t)
        out.append(EOS_TOKEN)
        return np.asarray(out, np.uint16)

    def _decode_pieces(self, body: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """The body after the tags back to (ids, data), cropped to its occupied box."""
        from blockgen.tokenizers.cluster_bpe import cluster_tokens_to_structure
        cv = self.cluster_vocab()
        nb, boff, poff = cv.num_blocks, self.vocab.block_offset, self.piece_offset
        mapped = []
        for t in body:
            if self.vocab.is_block(t):
                t = cv.piece_id(t - boff)
            elif poff <= t < poff + self.n_learned:
                t = cv.piece_id(nb + t - poff)
            elif not (self.vocab.is_coord(t) or t in (EOS_TOKEN, BOS_TOKEN)):
                t = -1                                              # a stray tag: breaks the current triple
            mapped.append(t)
        s = cluster_tokens_to_structure(mapped, cv)
        return s.block_ids.astype(np.uint16), s.block_data.astype(np.uint8)

    # --- the facades kind ------------------------------------------------
    def prefix(self, tags: Dict[str, str], face: Optional[str] = None,
               size: Optional[Tuple[int, int]] = None) -> List[int]:
        """The prompt a sampler feeds the model: ``[BOS, tags...]`` for voxels (``[CAT, tags...]``
        with the category group), plus ``face, U, V`` for facades when they are given."""
        out = self.head(tags)
        if face is not None:
            out.append(self.tag_id("face", face))
        if size is not None:
            out += [self.vocab.coord_id(size[0]), self.vocab.coord_id(size[1])]
        return out

    def _face_body(self, f: fc.Facade, lut: np.ndarray) -> List[int]:
        U, V = f.depth.shape
        assert max(U, V) <= self.vocab.max_dim, f.depth.shape
        empty = self.tag_id("cell", "empty")
        depth = np.clip(f.depth, 0, self.vocab.max_dim - 1).astype(np.int64) + self.vocab.coord_offset
        block = lut[np.clip(f.ids, 0, 255).astype(np.int64), np.clip(f.data, 0, 15).astype(np.int64)]
        body: List[int] = [self.tag_id("face", f.face), self.vocab.coord_id(U), self.vocab.coord_id(V)]
        for v in range(V):
            for u in range(U):
                if f.depth[u, v] < 0:
                    body.append(empty)
                else:
                    body += [int(depth[u, v]), int(block[u, v])]
        return body

    def encode_faces(self, faces: Sequence[fc.Facade], tags: Dict[str, str],
                     lut: Optional[np.ndarray] = None) -> np.ndarray:
        """Faces in the order given as one sequence (the joint layout; one face is the per-face layout)."""
        assert self.kind == "facades", self.kind
        lut = self.block_lut() if lut is None else lut
        body = [tok for f in faces for tok in self._face_body(f, lut)]
        return np.array(self.prefix(tags) + body + [EOS_TOKEN], np.uint16)

    def encode_face(self, f: fc.Facade, tags: Dict[str, str], lut: Optional[np.ndarray] = None) -> np.ndarray:
        """One face image as a sequence; see the module docstring for the layout. Under
        ``pooled4`` the face's own level f=4 goes first."""
        if self.layout == "pooled4":
            return self.encode_faces([fc.pool(f, POOL_FACTOR, "any"), f], tags, lut)
        return self.encode_faces([f], tags, lut)

    def decode_faces(self, tokens: np.ndarray) -> Tuple[List[fc.Facade], Dict[str, str]]:
        """Inverse of ``encode_faces``: every face the sequence holds, in order. A face
        that stops early leaves its remaining cells see-through; a face tag without
        two sizes after it ends the parse."""
        toks = [int(x) for x in np.asarray(tokens).reshape(-1)]
        tags: Dict[str, str] = {}
        i = 1 if toks and toks[0] == BOS_TOKEN else 0
        while i < len(toks):
            dt = self.decode_tag(toks[i])
            if dt is None or dt[0] == "face":
                break
            tags[dt[0]] = dt[1]; i += 1
        empty = self.tag_id("cell", "empty")
        faces: List[fc.Facade] = []
        while i < len(toks) and toks[i] != EOS_TOKEN:
            dt = self.decode_tag(toks[i])
            if dt is None or dt[0] != "face":
                i += 1; continue                                   # stray token between faces
            face = dt[1]; i += 1
            if i + 1 >= len(toks) or not (self.vocab.is_coord(toks[i]) and self.vocab.is_coord(toks[i + 1])):
                break
            U, V = self.vocab.decode_coord(toks[i]), self.vocab.decode_coord(toks[i + 1]); i += 2
            if U == 0 or V == 0:
                continue
            depth = np.full((U, V), fc.EMPTY, np.int16)
            ids = np.zeros((U, V), np.int16); data = np.zeros((U, V), np.int16)
            cells = 0
            while cells < U * V and i < len(toks) and toks[i] != EOS_TOKEN:
                dt = self.decode_tag(toks[i])
                if dt is not None and dt[0] == "face":
                    break                                           # the next face starts early
                u, v = cells % U, cells // U
                if toks[i] == empty:
                    i += 1; cells += 1
                elif i + 1 < len(toks) and self.vocab.is_coord(toks[i]) and self.vocab.is_block(toks[i + 1]):
                    depth[u, v] = self.vocab.decode_coord(toks[i])
                    ids[u, v], data[u, v] = self.vocab.decode_block(toks[i + 1])
                    i += 2; cells += 1
                else:
                    i += 1                                          # a malformed cell is skipped, not fatal
            extent = int(depth.max()) + 1 if (depth >= 0).any() else 1
            faces.append(fc.Facade(face=face, depth=depth, ids=ids, data=data, extent=extent))
        return faces, tags

    def decode_face(self, tokens: np.ndarray) -> Tuple[Optional[fc.Facade], Dict[str, str]]:
        """The first face of a sequence, or ``None`` when it holds none; under ``pooled4``
        the fine face (the second), ``None`` when the sequence stopped inside the coarse one."""
        faces, tags = self.decode_faces(tokens)
        if self.layout == "pooled4":
            return (faces[1] if len(faces) > 1 else None), tags
        return (faces[0] if faces else None), tags


# --------------------------------------------------------------------------- learning pieces

def learn_pieces(tier: str = "sft-houses32", n_merges: int = 256, max_corpus: int = 400, seed: int = 0,
                 min_count: int = 3, force_families: bool = True, max_dim: int = MAX_DIM,
                 out_dir: str = TOKENIZER_DIR, verbose: bool = True, oriented: bool = False) -> Tuple["CorpusTokenizer", str]:
    """Learn a ``pieces`` tokenizer from a training tier and save it as ``<out_dir>/<hash>.json``.

    The legacy recipe (``experiments_native.arm_bpe``): ``learn_clusters`` over a
    seeded sample of ``max_corpus`` builds, ``n_merges`` frequency merges after the
    forced bed/door/double-plant merges, here over the fixed block classes of the
    voxel kind (unoriented by default; ``oriented`` keeps stair/log/door facings as
    distinct classes, as the legacy ``--oriented`` arm did). A holdout tier is refused: the pieces are part of the
    model, so they may only be fit on what it trains on.
    """
    from blockgen.tokenizers.cluster_bpe import learn_clusters
    from .shards import _load_build, _tier_rows, _tight
    if tier.startswith("eval"):
        raise ValueError(f"{tier} is a holdout tier; learn pieces from a training tier (sft-houses32)")
    tok = CorpusTokenizer.default(oriented=oriented, max_dim=max_dim, kind="pieces")
    lut = tok.block_lut()
    spec, df = _tier_rows(tier)
    cache: Dict[str, object] = {}
    structs = []
    for r in df.to_dict("records"):
        ids, data = _tight(*_load_build(r["path"], cache))
        if ids.size and max(ids.shape) <= max_dim:
            structs.append(tok.canonical(ids, data, lut))
    if verbose:
        print(f"[pieces] {tier}: {len(structs)} of {len(df)} builds fit {max_dim}^3; learning {n_merges} merges "
              f"from {min(max_corpus, len(structs))}", flush=True)
    cv = learn_clusters(structs, max_dim=max_dim, n_merges=n_merges, max_corpus=max_corpus, min_count=min_count,
                        seed=seed, force_families=force_families, oriented=oriented, verbose=verbose, base=tok.vocab)
    tok.pieces = {"merges": [[int(pa), int(pb), [int(x) for x in d]] for pa, pb, d in cv.merges],
                  "fit_tier": tier, "fit_tier_sha256": spec["sha256"], "n_fit": min(max_corpus, len(structs)),
                  "n_merges": n_merges, "max_corpus": max_corpus, "seed": seed, "min_count": min_count,
                  "force_families": force_families}
    tok._cv = None
    path = tok.save(os.path.join(out_dir, f"{tok.hash}.json"))
    return tok, path


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="python -m blockgen.corpus.tokenizer learn-pieces [--tier sft-houses32] "
                                             "[--n-merges 256] [--max-corpus 400] [--seed 0]  |  show <hash|path>")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("learn-pieces", help="learn a 3D-BPE (pieces) tokenizer from a training tier")
    lp.add_argument("--tier", default="sft-houses32"); lp.add_argument("--n-merges", type=int, default=256)
    lp.add_argument("--max-corpus", type=int, default=400); lp.add_argument("--seed", type=int, default=0)
    lp.add_argument("--min-count", type=int, default=3); lp.add_argument("--out-dir", default=TOKENIZER_DIR)
    lp.add_argument("--oriented", action="store_true", help="orientation-aware block classes (stair/log/door facings)")
    sh = sub.add_parser("show", help="kind, vocabulary size and hash of a tokenizer")
    sh.add_argument("tokenizer")
    a = ap.parse_args(argv)
    if a.cmd == "learn-pieces":
        tok, path = learn_pieces(a.tier, a.n_merges, a.max_corpus, a.seed, a.min_count, out_dir=a.out_dir,
                                 oriented=a.oriented)
        print(f"[pieces] {tok.n_learned} learned pieces, vocab {tok.vocab_size} -> {path}")
        print(tok.hash)
    else:
        from .shards import resolve_tokenizer
        tok = resolve_tokenizer(a.tokenizer)
        print(json.dumps({"hash": tok.hash, "kind": tok.kind, "vocab_size": tok.vocab_size,
                          "n_learned": tok.n_learned, **{k: v for k, v in (tok.pieces or {}).items() if k != "merges"}},
                         indent=1))


if __name__ == "__main__":
    main()
