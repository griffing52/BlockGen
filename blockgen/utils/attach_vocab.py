"""Op-stream <-> token-id vocabulary for the attachment/growth model.

The insight that makes Phase 1 cheap: an attachment-op stream is *already* a token
sequence, so the growth model does not need a graph encoder at all for the MVP. A
plain causal transformer over op tokens is the autoregressive attachment model --
``VoxelTransformerAR2`` and ``train_ar_ext.train_from_sequences`` are reused
unchanged, and all the geometry lives in the decoder
(``attach_order.attach_ops_to_structure``), which replays the frontier.

That decoder is what buys the headline property: whatever the model emits, pose is
derived from the connection, so **connectivity holds by construction** -- there is no
coordinate to get wrong and no adjacency gate to run. The model's only job is to
choose, per open face, between CLOSE and (piece, direction).

Vocabulary layout (small and dense -- this is the whole point vs raster's
coord-token blowup):

    0                BOS
    1                EOS
    2 + d            CLOSE for direction d           (6 ids)
    8 + p            SEED with piece p               (P ids)
    8 + P + p*6 + d  ATTACH piece p, direction d     (P*6 ids)

Directions index ``graph_data.PORT_DIRECTIONS``. ``P`` is the number of distinct
block tokens observed in the corpus, so the vocab is a function of the data, not a
fixed 4096.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

from blockgen.utils.attach_order import (OP_ATTACH, OP_CLOSE, OP_SEED, Op,
                                         structure_to_attach_ops)

BOS_ID = 0
EOS_ID = 1
_CLOSE_BASE = 2
_N_DIR = 6


@dataclass
class AttachVocab:
    """Maps ops to ids. Save this next to every checkpoint (notes.md §18b)."""

    pieces: List[int] = field(default_factory=list)
    oriented: bool = False

    def __post_init__(self) -> None:
        self.piece_to_idx: Dict[int, int] = {p: i for i, p in enumerate(self.pieces)}

    @property
    def seed_base(self) -> int:
        return _CLOSE_BASE + _N_DIR

    @property
    def attach_base(self) -> int:
        return self.seed_base + len(self.pieces)

    @property
    def size(self) -> int:
        return self.attach_base + len(self.pieces) * _N_DIR

    # -- encode ---------------------------------------------------------
    def op_to_id(self, op: Op) -> int:
        if op.kind == OP_CLOSE:
            return _CLOSE_BASE + op.direction
        idx = self.piece_to_idx.get(op.piece)
        if idx is None:
            raise KeyError(f"piece {op.piece} not in vocab")
        if op.kind == OP_SEED:
            return self.seed_base + idx
        return self.attach_base + idx * _N_DIR + op.direction

    def ops_to_ids(self, ops: Sequence[Op]) -> List[int]:
        return [BOS_ID] + [self.op_to_id(o) for o in ops] + [EOS_ID]

    # -- decode ---------------------------------------------------------
    def id_to_op(self, tid: int) -> Op | None:
        """None for BOS/EOS/out-of-range."""
        if tid in (BOS_ID, EOS_ID):
            return None
        if tid < self.seed_base:
            return Op(OP_CLOSE, direction=tid - _CLOSE_BASE)
        if tid < self.attach_base:
            return Op(OP_SEED, piece=self.pieces[tid - self.seed_base])
        rel = tid - self.attach_base
        idx, d = divmod(rel, _N_DIR)
        if idx >= len(self.pieces):
            return None
        return Op(OP_ATTACH, piece=self.pieces[idx], direction=d)

    def ids_to_ops(self, ids: Sequence[int]) -> List[Op]:
        out: List[Op] = []
        for t in ids:
            if t == EOS_ID:
                break
            op = self.id_to_op(int(t))
            if op is not None:
                out.append(op)
        return out

    # -- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            {"pieces": [int(p) for p in self.pieces], "oriented": self.oriented}))

    @classmethod
    def load(cls, path: str | Path) -> "AttachVocab":
        blob = json.loads(Path(path).read_text())
        return cls(pieces=[int(p) for p in blob["pieces"]],
                   oriented=bool(blob.get("oriented", False)))


def build_vocab(structures, *, ordering: str = "bfs_bottom_center",
                oriented: bool = False, limit: int | None = None) -> AttachVocab:
    """Collect the distinct piece tokens the corpus actually uses."""
    seen = set()
    for s in structures[:limit] if limit else structures:
        try:
            ops, _ = structure_to_attach_ops(s, ordering=ordering, oriented=oriented)
        except Exception:
            continue
        for o in ops:
            if o.kind in (OP_SEED, OP_ATTACH):
                seen.add(o.piece)
    return AttachVocab(pieces=sorted(seen), oriented=oriented)


def build_sequences(structures, vocab: AttachVocab, *,
                    ordering: str = "bfs_bottom_center",
                    max_seq_len: int = 4096,
                    min_seq_len: int = 8) -> List[List[int]]:
    """Tokenize a corpus. Over-long builds are dropped, and the caller is expected
    to report how many -- silent truncation would misrepresent the "no box" claim.
    """
    seqs: List[List[int]] = []
    for s in structures:
        try:
            ops, _info = structure_to_attach_ops(s, ordering=ordering,
                                                 oriented=vocab.oriented)
            ids = vocab.ops_to_ids(ops)
        except Exception:
            continue
        if min_seq_len <= len(ids) <= max_seq_len:
            seqs.append(ids)
    return seqs
