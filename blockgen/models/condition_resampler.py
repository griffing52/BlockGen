"""Idea #6: conditioning-channel upgrade for ``CondVoxelAR2`` (see ``ideas.md``).

``CondVoxelAR2`` conditions on a single *pooled* CLIP token projected to one prefix
vector (``n_prefix=1, cond_dim=512``). That is the textbook weak conditioning link —
Point-E/Shap-E's single-token CLIP prefix "struggles with intricate prompts," and it
is exactly our T15 symptom (palette transfers, geometry does not; notes.md §16).

This replaces the pooled-vector projection with a **Perceiver/Q-Former-lite
resampler**: ``n_prefix`` learned query tokens cross-attend to the *full* condition
token sequence (e.g. CLIP text ``last_hidden_state`` ``[B, S, cond_dim]``, or DINOv2
patch tokens), so the prefix summarizes the whole prompt rather than a lossy mean.
This is the BLIP-2 Q-Former mechanism (arXiv:2301.12597) at small scale, and it is the
cheap-but-high-payoff move ranked #6 in ideas.md.

The resampler emits ``[B, n_prefix, d_model]`` — the same shape the existing prefix
machinery consumes — so it drops into ``ResampledCondVoxelAR2`` by overriding
``_make_prefix`` only; the rest of the decoder, CFG, and KV-cache paths are unchanged.

**Data note:** training this needs *sequence-level* condition embeddings, not the
pooled ``houses_32_cond_embeds.npz`` we have. Re-embed with the encoder's
``last_hidden_state`` (CLIP text) / patch tokens (DINOv2) first — see
``ideas.md`` status and ``labeling/embed_conditions.py``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from blockgen.models.voxel_transformer_cond import CondVoxelAR2


class ConditionResampler(nn.Module):
    """Cross-attention resampler: (condition sequence) -> fixed ``n_queries`` vectors.

    A stack of [self-attention over queries -> cross-attention queries→condition ->
    FFN] blocks, ending at ``d_model``. ``cond_mask`` marks padded condition positions
    (True = pad) so variable-length prompts attend correctly.
    """

    def __init__(self, *, cond_dim: int, d_model: int, n_queries: int,
                 n_layers: int = 2, nhead: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_queries = n_queries
        self.d_model = d_model
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.cond_proj = nn.Linear(cond_dim, d_model)
        self.cond_norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                   batch_first=True),
                "self_norm": nn.LayerNorm(d_model),
                "cross_attn": nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                    batch_first=True),
                "cross_norm": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(4 * d_model, d_model)),
                "ffn_norm": nn.LayerNorm(d_model),
            }) for _ in range(n_layers)])

    def forward(self, cond_tokens: torch.Tensor,
                cond_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """cond_tokens ``[B, S, cond_dim]`` -> ``[B, n_queries, d_model]``.

        ``cond_mask`` ``[B, S]`` is True at padded positions (dropped from attention).
        """
        B = cond_tokens.shape[0]
        kv = self.cond_norm(self.cond_proj(cond_tokens))
        q = self.queries[None].expand(B, -1, -1)
        for blk in self.layers:
            h = blk["self_norm"](q)
            q = q + blk["self_attn"](h, h, h, need_weights=False)[0]
            h = blk["cross_norm"](q)
            q = q + blk["cross_attn"](h, kv, kv, key_padding_mask=cond_mask,
                                      need_weights=False)[0]
            q = q + blk["ffn"](blk["ffn_norm"](q))
        return q


class ResampledCondVoxelAR2(CondVoxelAR2):
    """``CondVoxelAR2`` whose prefix is a resampled summary of the full condition seq.

    Set ``cond`` to a sequence ``[B, S, cond_dim]`` and (optionally) ``cond_mask``
    ``[B, S]``. The unconditional branch (``cond=None``) uses a learned null prefix of
    ``n_prefix`` vectors — the CFG / cond-dropout branch, same role as ``null_cond``.
    """

    def __init__(self, *, cond_dim: int, n_prefix: int, resampler_layers: int = 2,
                 resampler_heads: int = 8, **kwargs) -> None:
        super().__init__(cond_dim=cond_dim, n_prefix=n_prefix, **kwargs)
        d_model = self.token_embedding.embedding_dim
        self.resampler = ConditionResampler(
            cond_dim=cond_dim, d_model=d_model, n_queries=n_prefix,
            n_layers=resampler_layers, nhead=resampler_heads,
            dropout=kwargs.get("dropout", 0.1))
        # Learned null prefix in d_model space (the resampler path never uses null_cond,
        # which lives in cond_dim space). Small init, like prefix_pos.
        self.null_prefix = nn.Parameter(torch.randn(n_prefix, d_model) * 0.02)

    def _make_prefix(self, cond: Optional[torch.Tensor], B: int,
                     cond_mask: Optional[torch.Tensor] = None,
                     cond_drop: Optional[torch.Tensor] = None) -> torch.Tensor:
        null = self.null_prefix[None].expand(B, -1, -1)
        if cond is None:
            prefix = null
        else:
            prefix = self.resampler(cond, cond_mask)
            if cond_drop is not None:  # per-sample cond-dropout -> null prefix (CFG)
                prefix = torch.where(cond_drop[:, None, None], null, prefix)
        return prefix + self.prefix_pos[None]
