"""Adjacency-constrained decoding for the flat AR track.

At each (X, Y, Z, BLOCK) emission the coordinate logits are masked so the voxel
lands 6-adjacent to the already-generated structure (the first voxel is free).
Because the coordinate is emitted one axis at a time, the constraint factorizes
exactly: allowed X = {x of any allowed voxel}, then allowed Y given X, then
allowed Z given (X, Y). EOS stays available at every quadruple boundary once at
least ``min_blocks`` voxels exist.

This upgrades the post-hoc LCC repair gate to *in-loop* enforcement: connectivity
holds by construction, and later tokens condition on a valid partial structure
(the LegoGPT validity-during-sampling stance; research.md §B.2). Pair it with
BFS-ordered training sequences (``utils/ordering.py``) so the constraint matches
the training distribution.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

import torch

from blockgen.utils.data import Structure
from blockgen.utils.serialize import BOS_TOKEN, EOS_TOKEN, BlockVocab, tokens_to_structure

_NEIGHBORS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def _allowed_next_voxels(occupied: Set[Tuple[int, int, int]], max_dim: int
                         ) -> Set[Tuple[int, int, int]]:
    if not occupied:
        return set()  # signals "first voxel: unconstrained"
    out: Set[Tuple[int, int, int]] = set()
    for x, y, z in occupied:
        for dx, dy, dz in _NEIGHBORS:
            n = (x + dx, y + dy, z + dz)
            if n not in occupied and all(0 <= c < max_dim for c in n):
                out.add(n)
    return out


def _sample_masked(logits: torch.Tensor, allowed_ids: List[int], temperature: float,
                   top_k: Optional[int]) -> int:
    masked = torch.full_like(logits, -float("inf"))
    masked[allowed_ids] = logits[allowed_ids]
    masked = masked / max(temperature, 1e-5)
    if top_k is not None and 0 < top_k < len(allowed_ids):
        vals, idxs = torch.topk(masked, k=top_k)
        pick = torch.multinomial(torch.softmax(vals, dim=-1), 1)
        return int(idxs[pick].item())
    return int(torch.multinomial(torch.softmax(masked, dim=-1), 1).item())


@torch.no_grad()
def sample_constrained_structures(model, vocab: BlockVocab, num_samples: int = 16,
                                  temperature: float = 1.0, top_k: Optional[int] = 40,
                                  min_blocks: int = 12) -> List[Structure]:
    """Sample flat-AR structures with in-loop 6-adjacency enforcement."""
    model.eval()
    device = next(model.parameters()).device
    out: List[Structure] = []

    for _ in range(num_samples):
        seq: List[int] = [BOS_TOKEN]
        occupied: Set[Tuple[int, int, int]] = set()
        while len(seq) + 4 < model.max_seq_len:
            allowed = _allowed_next_voxels(occupied, vocab.max_dim)
            toks = torch.tensor([seq], dtype=torch.long, device=device)

            # X slot (EOS also legal here, at the quadruple boundary)
            logits = model(toks)[0, -1]
            if allowed:
                xs = sorted({v[0] for v in allowed})
            else:
                xs = list(range(vocab.max_dim))
            ids = [vocab.coord_id(c) for c in xs]
            if len(occupied) >= min_blocks:
                ids.append(EOS_TOKEN)
            tok = _sample_masked(logits, ids, temperature, top_k)
            if tok == EOS_TOKEN:
                break
            x = vocab.decode_coord(tok)
            seq.append(tok)

            # Y slot given X
            logits = model(torch.tensor([seq], dtype=torch.long, device=device))[0, -1]
            ys = sorted({v[1] for v in allowed if v[0] == x}) if allowed else list(range(vocab.max_dim))
            tok = _sample_masked(logits, [vocab.coord_id(c) for c in ys], temperature, top_k)
            y = vocab.decode_coord(tok)
            seq.append(tok)

            # Z slot given (X, Y)
            logits = model(torch.tensor([seq], dtype=torch.long, device=device))[0, -1]
            zs = sorted({v[2] for v in allowed if v[0] == x and v[1] == y}) if allowed \
                else list(range(vocab.max_dim))
            tok = _sample_masked(logits, [vocab.coord_id(c) for c in zs], temperature, top_k)
            z = vocab.decode_coord(tok)
            seq.append(tok)

            # BLOCK slot — any block class
            logits = model(torch.tensor([seq], dtype=torch.long, device=device))[0, -1]
            ids = [vocab.block_id_token(i) for i in range(vocab.num_blocks)]
            tok = _sample_masked(logits, ids, temperature, top_k)
            seq.append(tok)
            occupied.add((x, y, z))

        seq.append(EOS_TOKEN)
        out.append(tokens_to_structure(seq, vocab))
    return out
