"""Token-stream orderings for the flat AR track.

The default serialization (``structure_to_tokens``) emits voxels in (y, z, x)
raster order. ``bfs_structure_to_tokens`` instead emits them in **BFS-from-ground
order**: start at the lowest occupied voxel, breadth-first over 6-connectivity, so
every voxel after the first is adjacent to an already-emitted one (within its
connected component). Components are emitted lowest-first; each new component
restarts the BFS at its own lowest voxel.

Why: this makes "the next block attaches to the structure" true *in the training
data*, which is the precondition for adjacency-constrained decoding
(``training/constrained_decode.py``) — the LegoGPT/VoxelCNN recipe (research.md §B.2:
humans build in spatially local, connected order and modeling that order helps).
"""

from __future__ import annotations

from collections import deque
from typing import List

import numpy as np

from blockgen.utils.data import Structure, _token_for
from blockgen.utils.serialize import BOS_TOKEN, EOS_TOKEN, BlockVocab

_NEIGHBORS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def bfs_order(occ: np.ndarray) -> List[tuple]:
    """BFS-from-ground visiting order over occupied voxels (all components)."""
    remaining = {tuple(c) for c in np.argwhere(occ).tolist()}
    order: List[tuple] = []
    while remaining:
        # seed: lowest y, then z, then x — "start on the ground"
        seed = min(remaining, key=lambda c: (c[1], c[2], c[0]))
        q = deque([seed])
        remaining.discard(seed)
        while q:
            x, y, z = q.popleft()
            order.append((x, y, z))
            for dx, dy, dz in _NEIGHBORS:
                n = (x + dx, y + dy, z + dz)
                if n in remaining:
                    remaining.discard(n)
                    q.append(n)
    return order


def bfs_structure_to_tokens(structure: Structure, vocab: BlockVocab) -> List[int]:
    """``structure_to_tokens`` with BFS-from-ground voxel order (same format)."""
    s = structure.crop_to_non_air()
    if max(s.shape) > vocab.max_dim:
        raise ValueError(
            f"structure max dim {max(s.shape)} exceeds vocab.max_dim {vocab.max_dim}")
    tokens: List[int] = [BOS_TOKEN]
    for x, y, z in bfs_order(s.occupied_mask):
        tok = _token_for(int(s.block_ids[x, y, z]), int(s.block_data[x, y, z]))
        block_index = vocab.block_token_to_id.get(tok)
        if block_index is None:
            continue
        tokens.extend([vocab.coord_id(x), vocab.coord_id(y), vocab.coord_id(z),
                       vocab.block_id_token(block_index)])
    tokens.append(EOS_TOKEN)
    return tokens


def build_bfs_sequences(structures, vocab: BlockVocab, max_seq_len: int) -> List[List[int]]:
    seqs: List[List[int]] = []
    for s in structures:
        try:
            toks = bfs_structure_to_tokens(s, vocab)
        except ValueError:
            continue
        if 2 < len(toks) <= max_seq_len:
            seqs.append(toks)
    return seqs
