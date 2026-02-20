from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class VoxelTransformerAR(nn.Module):
    """Causal transformer for autoregressive voxel-token generation."""

    def __init__(
        self,
        *,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, input_ids: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len})")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)

        causal_mask = self._causal_mask(seq_len, input_ids.device)
        hidden = self.decoder(hidden, mask=causal_mask, src_key_padding_mask=pad_mask)
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return logits

    @torch.no_grad()
    def generate(
        self,
        *,
        bos_token_id: int,
        eos_token_id: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = 32,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device

        tokens = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            logits = self.forward(tokens)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None and top_k > 0 and top_k < next_logits.shape[-1]:
                vals, idxs = torch.topk(next_logits, k=top_k, dim=-1)
                probs = torch.softmax(vals, dim=-1)
                pick = torch.multinomial(probs, num_samples=1)
                next_token = idxs.gather(-1, pick)
            else:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)
            if int(next_token.item()) == eos_token_id:
                break

            if tokens.shape[1] >= self.max_seq_len:
                break

        return tokens
