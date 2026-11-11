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
                 lut: np.ndarray, constrain: bool, state: Optional[ShellState] = None):
        self.plan, self.shape, self.given, self.tok, self.lut = plan, tuple(int(s) for s in shape), given, tokenizer, lut
        self.state = (state if state is not None else ShellState(shape)) if constrain else None
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
        # log Z of every constrained step, per face: log of the model's probability mass (at the
        # sampling temperature, before top-k) left on the legal set, relative to the grammar's set.
        # Their sum is the log importance weight of the face set for target p(F | consistent).
        self.logz: Dict[str, float] = {}
        self.removed: Dict[str, float] = {}
        self.steps: Dict[str, int] = {}
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

    def grammar_mask(self, V: int) -> np.ndarray:
        """The per-face-syntax set for the next counted token (what the Grammar control allows)."""
        vocab = self.tok.vocab
        m = np.zeros(V, bool)
        if self.phase == "block":
            m[vocab.block_offset:vocab.block_offset + vocab.num_blocks] = True
        else:
            m[vocab.coord_offset:vocab.coord_offset + self.extent] = True
            m[self.empty_tok] = True
        return m

    def note_logz(self, lz: float) -> None:
        f = self.face
        self.logz[f] = self.logz.get(f, 0.0) + lz
        self.removed[f] = self.removed.get(f, 0.0) + (1.0 - float(np.exp(min(lz, 0.0))))
        self.steps[f] = self.steps.get(f, 0) + 1

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
                               given: Optional[List[Dict[str, fc.Facade]]] = None,
                               track: bool = False) -> Tuple[List[np.ndarray], dict]:
    """One joint-layout sequence per plan, grammar enforced, faces consistent when `constrain`.

    Returns the sequences (decode them with `tokenizer.decode_faces`) and a summary:
    `forced_cells` (cells written without sampling because earlier faces determined
    them) over `cells` (cells drawn, dictated faces excluded).
    """
    lut = tokenizer.block_lut()
    given = given or [{} for _ in plans]
    torch.manual_seed(seed)
    model.eval()
    out: List[np.ndarray] = []
    forced_cells = cells = 0
    tracked: List[_Row] = []
    for b0 in range(0, len(plans), batch):
        rows = [_Row(p, s, g, tokenizer, lut, constrain)
                for p, s, g in zip(plans[b0:b0 + batch], shapes[b0:b0 + batch], given[b0:b0 + batch])]
        tokens = _decode_rows(model, tokenizer, rows, tokenizer.prefix(tags), temperature, top_k, track=track)
        for r in rows:
            forced_cells += r.n_forced_cells
            cells += r.n_cells
        if track:
            tracked.extend(rows)
        out.extend(_cut_at_eos(tokens))
    model.train()
    info = {"forced_cells": forced_cells, "cells": cells}
    if track:
        info.update(_track_summary(tracked))
    return out, info


def _track_summary(rows: List["_Row"]) -> dict:
    """Per-sequence log weights and, per face, the mean probability mass the rule removed per step."""
    faces = sorted({f for r in rows for f in r.steps})
    per_face = {}
    for f in faces:
        n = sum(r.steps.get(f, 0) for r in rows)
        per_face[f] = {"steps": n,
                       "removed_mean": sum(r.removed.get(f, 0.0) for r in rows) / max(n, 1),
                       "logz_per_build": sum(r.logz.get(f, 0.0) for r in rows) / max(len(rows), 1)}
    return {"logw": [float(sum(r.logz.values())) for r in rows], "mask_mass": per_face}


def _decode_rows(model, tokenizer: CorpusTokenizer, rows: List[_Row], prompt: List[int], temperature: float,
                 top_k: int, track: bool = False) -> torch.Tensor:
    """Drive one batch of rows from a shared prompt to their EOS; returns the token rectangle."""
    device = next(model.parameters()).device
    V = tokenizer.vocab_size
    neg = torch.finfo(torch.float32).min
    B = len(rows)
    tokens = torch.tensor([prompt] * B, dtype=torch.long, device=device)
    budget = min(max(sum(3 + 2 * U * Vv for _, U, Vv in r.plan) for r in rows) + 1,
                 model.max_seq_len - len(prompt) - 1)
    dec = Decoder(model)
    for _ in range(budget):
        logits = dec.next_logits(tokens)
        masks = [r.mask(V) for r in rows]
        need = [i for i, m in enumerate(masks) if m is not None]
        if track:
            # normaliser of this step for every constrained row about to emit a counted token: a
            # sampled token's legal set, or a forced cell token (the rule left one answer)
            lt = logits.float() / max(temperature, 1e-5)
            tr_rows, legal, gram = [], [], []
            for i, r in enumerate(rows):
                if r.done or not r.constrain:
                    continue
                if masks[i] is not None:
                    tr_rows.append(i); legal.append(masks[i]); gram.append(r.grammar_mask(V))
                elif r.pending and r.pending[0][1] and r.pending[0][0] != EOS_TOKEN:
                    one = np.zeros(V, bool); one[r.pending[0][0]] = True
                    tr_rows.append(i); legal.append(one); gram.append(r.grammar_mask(V))
            if tr_rows:
                lg = lt[tr_rows]
                lm = torch.as_tensor(np.stack(legal), device=device)
                gm = torch.as_tensor(np.stack(gram), device=device)
                lz = (torch.logsumexp(lg.masked_fill(~lm, neg), -1)
                      - torch.logsumexp(lg.masked_fill(~gm, neg), -1)).tolist()
                for i, z in zip(tr_rows, lz):
                    rows[i].note_logz(float(z))
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
    return tokens


@torch.no_grad()
def generate_faces_constrained_per_face(model, tokenizer: CorpusTokenizer, tags: Dict[str, str], faces: Sequence[str],
                                        shapes: List[Sequence[int]], temperature: float, top_k: int, seed: int = 0,
                                        batch: int = 16, constrain: bool = True,
                                        orders: Optional[List[Sequence[str]]] = None
                                        ) -> Tuple[List[Dict[str, fc.Facade]], dict]:
    """Per-face layout: the faces of every build drawn in `faces` order, one sequence each.

    Face `k` of all builds is decoded (in batches across builds) before face `k + 1`,
    so with `constrain` each build's `ShellState` already holds faces `0..k-1` when
    face `k` is masked. Every sequence is `[tags, face, U, V, cell*, EOS]`, the
    per-face training format, grammar enforced. Returns the faces of each build (in
    draw order) and the same summary as `generate_faces_constrained`.
    """
    assert tokenizer.layout == "per_face", tokenizer.layout
    lut = tokenizer.block_lut()
    shapes = [tuple(int(x) for x in s) for s in shapes]
    states = [ShellState(s) if constrain else None for s in shapes]
    per_build: List[Dict[str, fc.Facade]] = [{} for _ in shapes]
    torch.manual_seed(seed)
    model.eval()
    forced_cells = cells = 0
    prompt = tokenizer.prefix(tags)
    # `orders[i]` is build i's face order (default: `faces` for every build). Position k of every
    # build is decoded before position k + 1, grouped by the face each build draws there.
    orders = [tuple(o) for o in orders] if orders is not None else [tuple(faces)] * len(shapes)
    work = []
    for k in range(len(faces)):
        for face in faces:
            group = [i for i in range(len(shapes)) if orders[i][k] == face]
            work += [(face, group[b0:b0 + batch]) for b0 in range(0, len(group), batch)]
    for face, idx in work:
        if True:
            rows = [_Row([(face, *face_size(face, shapes[i]))], shapes[i], {}, tokenizer, lut, constrain,
                         state=states[i]) for i in idx]
            tokens = _decode_rows(model, tokenizer, rows, prompt, temperature, top_k)
            for i, r, seq in zip(idx, rows, _cut_at_eos(tokens)):
                forced_cells += r.n_forced_cells
                cells += r.n_cells
                f, _ = tokenizer.decode_face(seq)
                if f is not None:
                    assert f.face == face and f.depth.shape == face_size(face, shapes[i]), (f.face, f.depth.shape)
                    per_build[i][face] = f
    model.train()
    return per_build, {"forced_cells": forced_cells, "cells": cells}


def decode_builds(seqs: List[np.ndarray], shapes: List[Sequence[int]], tokenizer: CorpusTokenizer,
                  corpus: str = "generated"):
    """Sequences to `(structures, per_build faces)`, faces lifted into their boxes as `facade_builds` does."""
    per_build = []
    for seq in seqs:
        faces: Dict[str, fc.Facade] = {}
        for f in tokenizer.decode_faces(seq)[0]:
            faces.setdefault(f.face, f)
        per_build.append(faces)
    return lift_builds(per_build, shapes, corpus), per_build


def lift_builds(per_build: List[Dict[str, fc.Facade]], shapes: List[Sequence[int]], corpus: str = "generated"):
    """Each build's faces lifted into its box, as `facade_builds` does (an empty box when it has none)."""
    from blockgen.utils.data import Structure
    builds = []
    for faces, shape in zip(per_build, shapes):
        shape = tuple(int(x) for x in shape)
        s = fc.lift(faces, shape) if faces else Structure(block_ids=np.zeros(shape, np.int32),
                                                          block_data=np.zeros(shape, np.int32))
        s.metadata.update({"corpus": corpus, "faces": ",".join(sorted(faces))})
        builds.append(s)
    return builds
