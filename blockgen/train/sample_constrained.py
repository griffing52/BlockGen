"""Joint-layout facade sampling under cross-face constraints.

`generate_faces_constrained` draws the faces of a build in the shard's order, like
`sample.generate_faces`, with two additions. The grammar is enforced at every
position (a cell is `empty` or a depth inside the box followed by a block; headers
and the final EOS are forced), so a face is never ended early. And with
`constrain=True` the logits of every cell are masked by a `ShellState` holding what
the earlier faces committed: a depth behind a voxel another face filled is
impossible, a depth at such a voxel forces its block, a column another face saw
through cannot hold a first hit, and a cell whose first open voxel is already known
has exactly one answer and is written without sampling. Faces passed in `given` are
forced verbatim (the oracle: the real front face in context) and committed to the
state like drawn ones.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from blockgen.corpus import facade as fc
from blockgen.corpus.facade_constraints import FREE, ShellState, face_keys
from blockgen.corpus.tokenizer import CorpusTokenizer
from blockgen.utils.serialize import EOS_TOKEN, PAD_TOKEN

from .sample import Decoder, _cut_at_eos, _pick

Plan = List[Tuple[str, int, int]]


def face_size(face: str, shape: Sequence[int]) -> Tuple[int, int]:
    """`(U, V)` of a face of a box, the rule `facade_builds` uses."""
    X, Y, Z = (int(x) for x in shape)
    axis = face[1]
    return (X, Y) if axis == "z" else (Z, Y) if axis == "x" else (X, Z)


class _Row:
    """Cursor of one sequence: which face, which cell, depth or block next.

    `pending` holds tokens to emit before anything is sampled, each flagged with
    whether it moves the cursor: headers and a dictated face's body do not (the
    cursor already sits past them), a determined cell's depth and block do.
    `given` maps a face name or a plan index to the facade dictated there (the
    index form serves the `pooled4` layout, where a face's coarse level and the
    face share a name).
    """

    def __init__(self, plan: Plan, shape: Sequence[int], given: Dict[str, fc.Facade], tokenizer: CorpusTokenizer,
                 lut: np.ndarray, constrain: bool):
        self.plan, self.shape, self.given, self.tok, self.lut = plan, tuple(int(s) for s in shape), given, tokenizer, lut
        self.state = ShellState(shape) if constrain else None
        self.constrain = constrain
        self.empty_tok = tokenizer.tag_id("cell", "empty")
        self.fi = -1
        self.cell = 0
        self.phase = "depth"
        self.depth = -1
        self.pending: List[Tuple[int, bool]] = []
        self.done = False
        self.n_forced_cells = 0
        self.n_cells = 0
        self._next_face()

    # --- bookkeeping ---------------------------------------------------------
    @property
    def face(self) -> str:
        return self.plan[self.fi][0]

    @property
    def extent(self) -> int:
        return self.shape[fc._parse(self.face)[0]]

    @property
    def uv(self) -> Tuple[int, int]:
        U = self.plan[self.fi][1]
        return self.cell % U, self.cell // U

    def _next_face(self) -> None:
        self.fi += 1
        self.cell = 0
        self.phase = "depth"
        if self.fi >= len(self.plan):
            self.pending.append((EOS_TOKEN, True))
            return
        face, U, V = self.plan[self.fi]
        vocab = self.tok.vocab
        self.pending += [(t, False) for t in (self.tok.tag_id("face", face), vocab.coord_id(U), vocab.coord_id(V))]
        g = self.given.get(self.fi, self.given.get(face))   # by plan index, or by face name
        if g is not None:                                  # the whole face is dictated
            assert g.depth.shape == (U, V), (face, g.depth.shape, (U, V))
            self.pending += [(t, False) for t in self.tok._face_body(g, self.lut)[3:]]
            if self.state is not None:
                self.state.commit_face(g, face_keys(g, self.lut))
            self._next_face()

    def _advance(self) -> None:
        self.cell += 1
        self.n_cells += 1
        self.phase = "depth"
        face, U, V = self.plan[self.fi]
        if self.cell >= U * V:
            self._next_face()

    # --- the mask for the next token -------------------------------------------
    def mask(self, V: int) -> Optional[np.ndarray]:
        """Allowed vocabulary for the next sampled token, or None when a pending token is forced."""
        if self.pending or self.done:
            return None
        vocab = self.tok.vocab
        m = np.zeros(V, bool)
        if self.phase == "block":
            m[vocab.block_offset:vocab.block_offset + vocab.num_blocks] = True
            return m
        extent = self.extent
        if self.constrain:
            u, v = self.uv
            allowed, forced, allow_empty = self.state.allowed(self.face, u, v)
            if not allow_empty and allowed.sum() == 1 and forced[int(np.argmax(allowed))] != FREE:
                d = int(np.argmax(allowed))                 # determined: write it, do not sample it
                self.pending += [(vocab.coord_id(d), True), (int(forced[d]), True)]
                self.n_forced_cells += 1
                return None
            m[vocab.coord_offset:vocab.coord_offset + extent] = allowed
            if allow_empty:
                m[self.empty_tok] = True
        else:
            m[vocab.coord_offset:vocab.coord_offset + extent] = True
            m[self.empty_tok] = True
        return m

    # --- consume one token --------------------------------------------------------
    def push(self, tok: int, counts: bool) -> None:
        """Account for the token just appended and move the cursor when it is a cell token."""
        if tok == EOS_TOKEN:
            self.done = True
            return
        if not counts:
            return
        u, v = self.uv
        if self.phase == "depth":
            if tok == self.empty_tok:
                if self.state is not None:
                    self.state.commit_cell(self.face, u, v, -1)
                self._advance()
            else:
                self.depth = self.tok.vocab.decode_coord(tok)
                self.phase = "block"
                if self.constrain and not self.pending:     # a sampled depth onto a claimed voxel forces its block
                    _, forced, _ = self.state.allowed(self.face, u, v)
                    if 0 <= self.depth < len(forced) and forced[self.depth] != FREE:
                        self.pending.append((int(forced[self.depth]), True))
        else:
            if self.state is not None:
                self.state.commit_cell(self.face, u, v, self.depth, tok)
            self._advance()


@torch.no_grad()
def generate_faces_constrained(model, tokenizer: CorpusTokenizer, tags: Dict[str, str], plans: List[Plan],
                               shapes: List[Sequence[int]], temperature: float, top_k: int, seed: int = 0,
                               batch: int = 16, constrain: bool = True,
                               given: Optional[List[Dict[str, fc.Facade]]] = None) -> Tuple[List[np.ndarray], dict]:
    """One joint-layout sequence per plan, grammar enforced, faces consistent when `constrain`.

    Returns the sequences (decode them with `tokenizer.decode_faces`) and a summary:
    `forced_cells` (cells written without sampling because earlier faces determined
    them) over `cells` (cells drawn, dictated faces excluded).
    """
    device = next(model.parameters()).device
    V = tokenizer.vocab_size
    lut = tokenizer.block_lut()
    given = given or [{} for _ in plans]
    torch.manual_seed(seed)
    model.eval()
    out: List[np.ndarray] = []
    forced_cells = cells = 0
    neg = torch.finfo(torch.float32).min
    for b0 in range(0, len(plans), batch):
        rows = [_Row(p, s, g, tokenizer, lut, constrain)
                for p, s, g in zip(plans[b0:b0 + batch], shapes[b0:b0 + batch], given[b0:b0 + batch])]
        B = len(rows)
        prompt = tokenizer.prefix(tags)
        tokens = torch.tensor([prompt] * B, dtype=torch.long, device=device)
        budget = min(max(sum(3 + 2 * U * Vv for _, U, Vv in r.plan) for r in rows) + 1,
                     model.max_seq_len - len(prompt) - 1)
        dec = Decoder(model)
        for _ in range(budget):
            logits = dec.next_logits(tokens)
            masks = [r.mask(V) for r in rows]
            need = [i for i, m in enumerate(masks) if m is not None]
            picked = {}
            if need:
                mt = torch.as_tensor(np.stack([masks[i] for i in need]), device=device)
                nl = logits[need].float().masked_fill(~mt, neg)
                pk = _pick(nl, temperature, top_k)[:, 0].tolist()
                picked = dict(zip(need, pk))
            nxt = []
            for i, r in enumerate(rows):
                if r.done:
                    nxt.append(PAD_TOKEN)
                    continue
                if i in picked:
                    tok, counts = picked[i], True
                else:
                    tok, counts = r.pending.pop(0)
                r.push(tok, counts)
                nxt.append(tok)
            tokens = torch.cat([tokens, torch.tensor(nxt, dtype=torch.long, device=device)[:, None]], dim=1)
            if all(r.done for r in rows):
                break
        for r in rows:
            forced_cells += r.n_forced_cells
            cells += r.n_cells
        out.extend(_cut_at_eos(tokens))
    model.train()
    return out, {"forced_cells": forced_cells, "cells": cells}


def decode_builds(seqs: List[np.ndarray], shapes: List[Sequence[int]], tokenizer: CorpusTokenizer,
                  corpus: str = "generated"):
    """Sequences to `(structures, per_build faces)`, faces lifted into their boxes as `facade_builds` does."""
    from blockgen.utils.data import Structure
    builds, per_build = [], []
    for seq, shape in zip(seqs, shapes):
        faces: Dict[str, fc.Facade] = {}
        for f in tokenizer.decode_faces(seq)[0]:
            faces.setdefault(f.face, f)
        shape = tuple(int(x) for x in shape)
        s = fc.lift(faces, shape) if faces else Structure(block_ids=np.zeros(shape, np.int32),
                                                          block_data=np.zeros(shape, np.int32))
        s.metadata.update({"corpus": corpus, "faces": ",".join(sorted(faces))})
        builds.append(s)
        per_build.append(faces)
    return builds, per_build
