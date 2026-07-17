"""PE-pluggable causal transformer for AR voxel-token generation.

Same interface as ``VoxelTransformerAR`` (forward(input_ids, pad_mask) -> logits,
max_seq_len / vocab_size attrs) but with a selectable positional encoding:

  * ``learned`` — absolute learned embedding (the original model's scheme; baseline)
  * ``sin``     — fixed 1D sinusoidal absolute embedding
  * ``rope``    — rotary position embedding applied to q/k (relative; Su et al. 2021)
  * ``alibi``   — no position embedding; per-head linear attention bias
                  (relative; Press et al. 2022 "train short, test long")
  * ``phase4``  — structured PE for our (X,Y,Z,BLOCK) grammar: learned phase
                  (pos mod 4) + learned block-index (pos div 4) embeddings, so the
                  period-4 slot structure is given rather than learned
  * ``none``    — no positional signal (causal mask alone; NoPE baseline)

Motivation (research.md §B): learned absolute PE trains tail positions on very few
examples (only the largest builds reach them) and must rediscover the period-4
token grammar; Scaffold Diffusion's ablation shows PE choice can be decisive at
our data scale. RoPE/ALiBi are the literature-standard relative schemes.

Uses a hand-rolled attention block (F.scaled_dot_product_attention) because stock
``nn.TransformerEncoderLayer`` exposes neither q/k rotation nor per-head biases.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

PE_CHOICES = ("learned", "sin", "rope", "alibi", "phase4", "none")


def _sinusoidal_table(max_len: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(max_len).float().unsqueeze(1)
    div = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2).float() / d_model)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: (d_model + 1) // 2])
    return pe


def _alibi_slopes(nhead: int) -> torch.Tensor:
    """Geometric per-head slopes as in the ALiBi paper (power-of-2 heads exact)."""
    def pow2_slopes(n: int):
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]

    if math.log2(nhead).is_integer():
        s = pow2_slopes(nhead)
    else:
        closest = 2 ** math.floor(math.log2(nhead))
        s = pow2_slopes(closest)
        s += pow2_slopes(2 * closest)[0::2][: nhead - closest]
    return torch.tensor(s)


class _RotaryCache(nn.Module):
    def __init__(self, head_dim: int, max_len: int, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv)                       # (L, hd/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def rotate(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # x: (B, H, L, hd). `offset` = absolute position of x[..., 0, :]; nonzero only
        # under KV-cached decode, where x is the single NEW token at position `offset`.
        # Getting this wrong applies the position-0 rotation to a token at position t
        # and degrades silently.
        L = x.shape[2]
        cos = self.cos[offset:offset + L].to(x.dtype)     # (L, hd/2)
        sin = self.sin[offset:offset + L].to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out


class _CausalBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float,
                 rope: Optional[_RotaryCache]):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model))
        self.dropout = nn.Dropout(dropout)
        self.rope = rope

    def forward(self, x: torch.Tensor, bias: Optional[torch.Tensor],
                past_kv=None, pos_offset: int = 0):
        # x: (B, L, D); bias: (1|B, H, L, L) additive float mask (-inf = blocked),
        # or None to use the fused causal kernel.
        # past_kv: (k, v) from previous steps -> returns (out, (k, v)) for reuse.
        B, L, D = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            # rotate the NEW q/k at their absolute positions, before concatenating
            # the (already-rotated) cache.
            q = self.rope.rotate(q, offset=pos_offset)
            k = self.rope.rotate(k, offset=pos_offset)
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_kv = (k, v)
        q_len, k_len = q.shape[2], k.shape[2]
        if bias is None and q_len != k_len:
            # Cached decode: the query is the single new token and every cached key
            # precedes it, so attending to all of them IS causal. is_causal=True here
            # would align the mask to the top-left of a (1, t) score matrix and let the
            # token see only position 0 -- silently, with no error. Refuse anything we
            # cannot prove correct rather than guess.
            if q_len != 1:
                raise ValueError(
                    f"cached attention supports only single-token decode "
                    f"(q_len={q_len}, k_len={k_len}); a multi-token query against a "
                    f"cache needs an explicit mask")
            att = F.scaled_dot_product_attention(q, k, v)
            att = att.transpose(1, 2).reshape(B, L, D)
            x = x + self.dropout(self.proj(att))
            x = x + self.dropout(self.mlp(self.norm2(x)))
            return x, new_kv
        if bias is None:
            # An explicit attn_mask disqualifies the flash / mem-efficient SDPA
            # kernels and forces the math fallback, which materializes the full
            # [B, H, L, L] score matrix -- ~9.6 GB at B=8/H=8/L=5480, and the
            # reason long-sequence runs were near-OOM. is_causal uses the fused
            # kernel at O(L) memory. Equivalence: sequences are RIGHT-padded and
            # the loss uses ignore_index=PAD, so a real token at i attends only to
            # j<=i (all real) either way; only pad rows differ, and they are
            # dropped from the loss. Verified exact (0.0 diff in float64) on real
            # positions. See notes.md §17.
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        att = att.transpose(1, 2).reshape(B, L, D)
        x = x + self.dropout(self.proj(att))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x, new_kv


class VoxelTransformerAR2(nn.Module):
    """Drop-in AR transformer with selectable positional encoding (see module doc)."""

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
        pe: str = "learned",
        piece_factors=None,
        piece_offset: Optional[int] = None,
    ) -> None:
        super().__init__()
        if pe not in PE_CHOICES:
            raise ValueError(f"pe must be one of {PE_CHOICES}, got {pe!r}")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pe = pe

        # Opt-in factored piece embedding: rotations/variants of one motif share
        # parameters instead of each piece id learning from scratch. Pass the
        # PieceFactors from build_piece_factors(cv) plus cv.piece_offset.
        if piece_factors is not None:
            if piece_offset is None:
                raise ValueError("piece_offset is required when piece_factors is given")
            from blockgen.models.factored_embedding import FactoredPieceEmbedding
            self.token_embedding = FactoredPieceEmbedding(
                vocab_size=vocab_size, d_model=d_model,
                piece_offset=piece_offset, factors=piece_factors)
        else:
            self.token_embedding = nn.Embedding(vocab_size, d_model)
        if pe == "learned":
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
        elif pe == "sin":
            self.register_buffer("sin_table", _sinusoidal_table(max_seq_len, d_model),
                                 persistent=False)
        elif pe == "phase4":
            self.phase_embedding = nn.Embedding(4, d_model)
            self.block_index_embedding = nn.Embedding(max_seq_len // 4 + 1, d_model)
        if pe == "alibi":
            self.register_buffer("alibi_slopes", _alibi_slopes(nhead), persistent=False)

        rope = _RotaryCache(d_model // nhead, max_seq_len) if pe == "rope" else None
        self.blocks = nn.ModuleList([
            _CausalBlock(d_model, nhead, dim_feedforward, dropout, rope)
            for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.emb_dropout = nn.Dropout(dropout)
        self.nhead = nhead

    def needs_bias(self) -> bool:
        """Only ALiBi genuinely needs an additive score bias.

        Every other PE only wants causal masking, which the fused SDPA kernel does
        natively at O(L) memory -- see _CausalBlock.forward for why passing an
        explicit mask instead costs O(B*H*L^2).
        """
        return self.pe == "alibi"

    def _bias(self, seq_len: int, pad_mask: Optional[torch.Tensor],
              device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        neg = torch.finfo(dtype).min
        causal = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), 1)
        bias = torch.zeros(1, 1, seq_len, seq_len, dtype=dtype, device=device)
        bias = bias.masked_fill(causal, neg)
        if self.pe == "alibi":
            dist = torch.arange(seq_len, device=device).view(1, -1) - \
                   torch.arange(seq_len, device=device).view(-1, 1)   # j - i (<=0 kept)
            alibi = self.alibi_slopes.to(dtype).view(1, -1, 1, 1) * dist.to(dtype)
            bias = bias + alibi                                        # (1, H, L, L)
        if pad_mask is not None:  # True where padded
            pm = pad_mask[:, None, None, :].to(torch.bool)
            bias = bias.expand(pad_mask.size(0), self.nhead if self.pe == "alibi" else 1,
                               seq_len, seq_len).clone()
            bias = bias.masked_fill(pm, neg)
        return bias

    def supports_cache(self) -> bool:
        """ALiBi needs a full [B,H,L,L] bias, which cached decode cannot build."""
        return not self.needs_bias()

    def forward(self, input_ids: torch.Tensor, pad_mask: Optional[torch.Tensor] = None,
                past=None, return_cache: bool = False):
        """Standard forward; pass `past` (list of per-block (k,v)) for cached decode.

        With `past`, `input_ids` is the single NEW token and absolute positions resume
        at len(cache) -- critical for phase4/rope/learned/sin, which all key off
        absolute position. Feeding arange(1) would tell phase4 every generated token
        sits at position 0 and silently break the grammar prior.
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        B, L = input_ids.shape
        past_len = 0 if past is None else past[0][0].shape[2]
        if past is not None and not self.supports_cache():
            raise ValueError(f"pe={self.pe!r} cannot use a KV cache (needs a full bias)")
        if past_len + L > self.max_seq_len:
            raise ValueError(
                f"seq_len ({past_len + L}) exceeds max_seq_len ({self.max_seq_len})")
        if L > self.max_seq_len:
            raise ValueError(f"seq_len ({L}) exceeds max_seq_len ({self.max_seq_len})")
        if input_ids.numel() > 0:
            lo, hi = int(input_ids.min()), int(input_ids.max())
            if lo < 0 or hi >= self.vocab_size:
                raise ValueError(f"token id out of range [0,{self.vocab_size}): min={lo}, max={hi}")

        h = self.token_embedding(input_ids)
        # absolute positions resume after the cache (past_len == 0 when uncached)
        pos = torch.arange(past_len, past_len + L, device=input_ids.device)
        if self.pe == "learned":
            h = h + self.position_embedding(pos)[None]
        elif self.pe == "sin":
            h = h + self.sin_table[past_len:past_len + L].to(h.dtype)[None]
        elif self.pe == "phase4":
            # token 0 is BOS; the (X,Y,Z,BLOCK) grammar starts at position 1
            grammar_pos = (pos - 1).clamp(min=0)
            h = h + self.phase_embedding(grammar_pos % 4)[None] \
                  + self.block_index_embedding(grammar_pos // 4)[None]
        # rope: applied inside attention; alibi/none: no additive embedding
        h = self.emb_dropout(h)

        # None => fused causal kernel (O(L) memory). Only ALiBi needs the explicit
        # [B,H,L,L] additive bias, which forces SDPA's math fallback.
        bias = self._bias(L, pad_mask, input_ids.device, h.dtype) \
            if self.needs_bias() else None
        new_cache = []
        for i, blk in enumerate(self.blocks):
            h, kv = blk(h, bias, past_kv=None if past is None else past[i],
                        pos_offset=past_len)
            new_cache.append(kv)
        logits = self.lm_head(self.norm(h))
        return (logits, new_cache) if return_cache else logits

    @torch.no_grad()
    def generate(self, *, bos_token_id: int, eos_token_id: int, max_new_tokens: int,
                 temperature: float = 1.0, top_k: Optional[int] = 32,
                 device: Optional[torch.device] = None,
                 use_cache: bool = True) -> torch.Tensor:
        """Autoregressive sampling.

        With ``use_cache`` (default) each step feeds only the new token and reuses the
        cached keys/values, so producing L tokens costs O(L) token-forwards instead of
        the O(L^2/2) of re-running the whole prefix every step (~800x less attention
        work at L=1600). Sampling is otherwise identical -- verified to emit the same
        token ids as the uncached path under a fixed seed. ALiBi falls back to the
        uncached path (it needs a full [B,H,L,L] bias).
        """
        if device is None:
            device = next(self.parameters()).device
        use_cache = use_cache and self.supports_cache()
        tokens = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
        past = None
        for _ in range(max_new_tokens):
            if use_cache:
                # first step primes the cache with the prefix; later steps feed 1 token
                step_in = tokens if past is None else tokens[:, -1:]
                logits, past = self.forward(step_in, past=past, return_cache=True)
            else:
                logits = self.forward(tokens)
            nl = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None and 0 < top_k < nl.shape[-1]:
                vals, idxs = torch.topk(nl, k=top_k, dim=-1)
                pick = torch.multinomial(torch.softmax(vals, dim=-1), 1)
                nxt = idxs.gather(-1, pick)
            else:
                nxt = torch.multinomial(torch.softmax(nl, dim=-1), 1)
            tokens = torch.cat([tokens, nxt], dim=1)
            if int(nxt.item()) == eos_token_id or tokens.shape[1] >= self.max_seq_len:
                break
        return tokens
