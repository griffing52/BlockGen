"""Factored token embedding for 3D-BPE piece vocabularies.

Replaces the flat ``nn.Embedding(vocab_size, d_model)`` over piece tokens with

    E[piece] = E_shape[shape ↑D4] + E_rot[rotation] + E_family[block family]
               + E_variant[block data]

so that rotations and material variants of one motif share parameters and train in
parallel. See ``blockgen/tokenizers/piece_factors.py`` for the measurement that
motivates this (41% of merge slots are same-shape/same-family duplicates) and for
why we factor the embedding rather than canonicalize the tokens.

``ClusterVocab`` lays tokens out as ``[PAD,BOS,EOS] + coords(max_dim) + pieces``, so
piece ids are contiguous at the tail. Specials and coordinate tokens keep an ordinary
flat table; only the tail is factored. The full weight matrix is rebuilt per forward
(vocab is ~700 rows, so this is free) and stays differentiable end to end.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from blockgen.tokenizers.piece_factors import PieceFactors


class FactoredPieceEmbedding(nn.Module):
    """Embedding whose piece-token rows are a sum of factor embeddings."""

    def __init__(self, *, vocab_size: int, d_model: int, piece_offset: int,
                 factors: PieceFactors) -> None:
        super().__init__()
        n_pieces = len(factors.shape_idx)
        if piece_offset + n_pieces != vocab_size:
            raise ValueError(
                f"piece tokens must be contiguous at the tail of the vocab: "
                f"piece_offset {piece_offset} + n_pieces {n_pieces} != vocab {vocab_size}")
        self.vocab_size = vocab_size
        self.piece_offset = piece_offset

        # Specials + coordinate tokens: ordinary flat table.
        self.head = nn.Embedding(piece_offset, d_model)

        self.shape_emb = nn.Embedding(factors.n_shapes, d_model)
        self.rot_emb = nn.Embedding(factors.n_rots, d_model)
        self.family_emb = nn.Embedding(factors.n_families, d_model)
        self.variant_emb = nn.Embedding(factors.n_variants, d_model)

        for name, arr in (("shape_idx", factors.shape_idx), ("rot_idx", factors.rot_idx),
                          ("family_idx", factors.family_idx),
                          ("variant_idx", factors.variant_idx)):
            self.register_buffer(name, torch.as_tensor(arr, dtype=torch.long),
                                 persistent=False)

        # Sum of 4 factors has ~4x the variance of one table; scale down so the
        # initial embedding distribution matches a plain nn.Embedding's N(0,1).
        for emb in (self.shape_emb, self.rot_emb, self.family_emb, self.variant_emb):
            nn.init.normal_(emb.weight, std=0.5)

    def piece_weight(self) -> torch.Tensor:
        return (self.shape_emb(self.shape_idx)
                + self.rot_emb(self.rot_idx)
                + self.family_emb(self.family_idx)
                + self.variant_emb(self.variant_idx))

    def weight_matrix(self) -> torch.Tensor:
        return torch.cat([self.head.weight, self.piece_weight()], dim=0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.embedding(tokens, self.weight_matrix())
