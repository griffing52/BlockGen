"""Prefix-conditioned variant of ``VoxelTransformerAR2`` (LegoACE strategy).

LegoACE conditions a decoder-only LM by projecting frozen-encoder embeddings
(DINOv2 multi-view / CLIP text) through a linear layer and prepending them to
the token embeddings as a prefix, with classifier-free guidance at sampling.
This ports that recipe onto our PE-pluggable AR backbone:

* ``cond`` [B, P, cond_dim] -> Linear -> P prefix vectors (+ learned prefix
  position embedding to distinguish views/captions).
* A learned **null condition** replaces ``cond`` on a random fraction of
  training examples (cond-dropout), which trains the unconditional branch used
  for CFG: ``logits = uncond + scale * (cond - uncond)``.
* Token-side PE (learned/phase4/...) is computed on token positions exactly as
  in the parent, so the (X,Y,Z,PIECE) grammar alignment is unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from blockgen.models.voxel_transformer_ar2 import VoxelTransformerAR2


class CondVoxelAR2(VoxelTransformerAR2):
    def __init__(self, *, cond_dim: int, n_prefix: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cond_dim = cond_dim
        self.n_prefix = n_prefix
        self.cond_proj = nn.Linear(cond_dim, self.token_embedding.embedding_dim)
        self.prefix_pos = nn.Parameter(
            torch.randn(n_prefix, self.token_embedding.embedding_dim) * 0.02)
        self.null_cond = nn.Parameter(torch.zeros(n_prefix, cond_dim))

    def _token_h(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token embeddings + token-side PE, exactly as the parent computes them."""
        B, L = input_ids.shape
        h = self.token_embedding(input_ids)
        pos = torch.arange(L, device=input_ids.device)
        if self.pe == "learned":
            h = h + self.position_embedding(pos)[None]
        elif self.pe == "sin":
            h = h + self.sin_table[:L].to(h.dtype)[None]
        elif self.pe == "phase4":
            grammar_pos = (pos - 1).clamp(min=0)
            h = h + self.phase_embedding(grammar_pos % 4)[None] \
                  + self.block_index_embedding(grammar_pos // 4)[None]
        return h

    def _make_prefix(self, cond: Optional[torch.Tensor], B: int,
                     cond_mask: Optional[torch.Tensor] = None,
                     cond_drop: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Map a condition to ``[B, n_prefix, d_model]`` prefix vectors.

        Default (LegoACE recipe): ``cond`` is ``[B, n_prefix, cond_dim]`` (e.g. one
        pooled CLIP token), linearly projected. ``None`` selects the learned null
        condition (CFG / cond-dropout). ``ResampledCondVoxelAR2`` overrides this to
        cross-attend a full condition *sequence* into the prefix — the idea #6
        conditioning-channel upgrade — and is the only path that uses ``cond_mask`` /
        ``cond_drop`` (per-sample cond-dropout to the null prefix).
        """
        if cond is None:
            cond = self.null_cond[None].expand(B, -1, -1)
        return self.cond_proj(cond) + self.prefix_pos[None]

    def forward(self, input_ids: torch.Tensor,
                cond: Optional[torch.Tensor] = None,
                pad_mask: Optional[torch.Tensor] = None,
                cond_mask: Optional[torch.Tensor] = None,
                cond_drop: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns logits aligned with ``input_ids`` positions (prefix stripped)."""
        B, L = input_ids.shape
        if L + self.n_prefix > self.max_seq_len + self.n_prefix:
            raise ValueError(f"seq_len {L} exceeds max_seq_len {self.max_seq_len}")
        prefix = self._make_prefix(cond, B, cond_mask, cond_drop)

        h = torch.cat([prefix.to(self.token_embedding.weight.dtype),
                       self._token_h(input_ids)], dim=1)
        h = self.emb_dropout(h)

        total = L + self.n_prefix
        full_pad = None
        if pad_mask is not None:
            full_pad = torch.cat([
                torch.zeros(B, self.n_prefix, dtype=pad_mask.dtype,
                            device=pad_mask.device), pad_mask], dim=1)
        bias = self._bias(total, full_pad, input_ids.device, h.dtype)
        for blk in self.blocks:
            # _CausalBlock returns (hidden, kv_cache); the cache is unused here.
            h, _ = blk(h, bias)
        return self.lm_head(self.norm(h))[:, self.n_prefix:]

    @torch.no_grad()
    def generate_cond(self, *, cond: torch.Tensor, bos_token_id: int,
                      eos_token_id: int, max_new_tokens: int,
                      temperature: float = 1.0, top_k: Optional[int] = 32,
                      cfg_scale: float = 1.0,
                      cond_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Batched conditional sampling with classifier-free guidance.

        cond: [B, P, cond_dim] (pooled prefix) or [B, S, cond_dim] (sequence, for the
        resampler). ``cond_mask`` [B, S] is a key-padding mask (True = pad), used only
        by the resampler path. Returns [B, L] token ids (right-padded with EOS).
        """
        device = next(self.parameters()).device
        B = cond.shape[0]
        tokens = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            logits_c = self.forward(tokens, cond=cond, cond_mask=cond_mask)[:, -1]
            if cfg_scale != 1.0:
                logits_u = self.forward(tokens, cond=None)[:, -1]
                logits = logits_u + cfg_scale * (logits_c - logits_u)
            else:
                logits = logits_c
            logits = logits / max(temperature, 1e-5)
            if top_k is not None and 0 < top_k < logits.shape[-1]:
                vals, idxs = torch.topk(logits, k=top_k, dim=-1)
                pick = torch.multinomial(torch.softmax(vals, dim=-1), 1)
                nxt = idxs.gather(-1, pick)
            else:
                nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            nxt[finished] = eos_token_id
            tokens = torch.cat([tokens, nxt], dim=1)
            finished |= nxt.squeeze(1) == eos_token_id
            if bool(finished.all()) or tokens.shape[1] >= self.max_seq_len:
                break
        return tokens
