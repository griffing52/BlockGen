"""Incremental token sampling for the AR models, yielding one token at a time.

The training-side samplers (``VoxelTransformerAR2.generate``,
``CondVoxelAR2.generate_cond``) return a finished tensor, which is the wrong shape
for a live demo: the whole point here is to hand tokens to Minecraft *as they are
produced*. These are the same sampling rules (temperature, top-k, CFG) restructured
as generators, plus an explicit ``torch.Generator`` so a seed reproduces a build.

KV caching matters a lot at this sequence length. ``native_bpe`` runs to 5,480
tokens; re-running the full prefix each step is O(L^2) token-forwards (~15M vs 5.5k)
and would make the demo unusable. ``VoxelTransformerAR2.forward`` already supports
``past``/``return_cache``; ``CondVoxelAR2`` does not, so ``stream_cond_tokens``
reimplements its forward with a cache. That reimplementation is verified against the
stock uncached path in ``tests/test_sampling_parity.py`` — if you touch it, re-run
that test, because a wrong position offset degrades output silently rather than
raising.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch


def _pick(logits: torch.Tensor, temperature: float, top_k: Optional[int],
          generator: Optional[torch.Generator]) -> torch.Tensor:
    """Temperature + top-k sample from a [1, vocab] logits row."""
    logits = logits / max(temperature, 1e-5)
    if top_k is not None and 0 < top_k < logits.shape[-1]:
        vals, idxs = torch.topk(logits, k=top_k, dim=-1)
        pick = torch.multinomial(torch.softmax(vals, dim=-1), 1, generator=generator)
        return idxs.gather(-1, pick)
    return torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator)


@torch.no_grad()
def stream_tokens(model, *, bos_token_id: int, eos_token_id: int,
                  max_new_tokens: int, temperature: float = 1.0,
                  top_k: Optional[int] = 40,
                  generator: Optional[torch.Generator] = None) -> Iterator[int]:
    """Yield sampled token ids from an unconditional ``VoxelTransformerAR2``.

    Stops after EOS (which is not yielded), ``max_new_tokens``, or ``max_seq_len``.
    """
    device = next(model.parameters()).device
    model.eval()
    use_cache = model.supports_cache()
    tokens = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
    past = None
    for _ in range(max_new_tokens):
        if use_cache:
            step_in = tokens if past is None else tokens[:, -1:]
            logits, past = model.forward(step_in, past=past, return_cache=True)
        else:
            logits = model.forward(tokens)
        nxt = _pick(logits[:, -1, :], temperature, top_k, generator)
        tok = int(nxt.item())
        if tok == eos_token_id:
            return
        yield tok
        tokens = torch.cat([tokens, nxt], dim=1)
        if tokens.shape[1] >= model.max_seq_len:
            return


def _cond_token_h(model, input_ids: torch.Tensor, pos_offset: int) -> torch.Tensor:
    """``CondVoxelAR2._token_h`` but at an absolute position offset.

    The stock version hardcodes ``pos = arange(L)``, which is only right when the
    whole sequence is fed at once. Under cached decode the single new token sits at
    ``pos_offset``, and phase4/learned/sin all key off absolute position -- feeding
    position 0 every step would quietly destroy the grammar prior.
    """
    L = input_ids.shape[1]
    h = model.token_embedding(input_ids)
    pos = torch.arange(pos_offset, pos_offset + L, device=input_ids.device)
    if model.pe == "learned":
        h = h + model.position_embedding(pos)[None]
    elif model.pe == "sin":
        h = h + model.sin_table[pos_offset:pos_offset + L].to(h.dtype)[None]
    elif model.pe == "phase4":
        grammar_pos = (pos - 1).clamp(min=0)
        h = h + model.phase_embedding(grammar_pos % 4)[None] \
              + model.block_index_embedding(grammar_pos // 4)[None]
    return h


@torch.no_grad()
def _cond_logits(model, input_ids: torch.Tensor, cond: Optional[torch.Tensor],
                 past, pos_offset: int):
    """One cached forward step of ``CondVoxelAR2``; returns (last_logits, new_past).

    Mirrors ``CondVoxelAR2.forward``: the conditioning prefix is prepended to the
    token embeddings and stripped from the output. On the priming step the prefix is
    part of the sequence, so token positions are offset by ``n_prefix`` inside the
    cache -- that is why ``pos_offset`` (a *token* position) is tracked separately
    from the cache length.
    """
    B = input_ids.shape[0]
    if past is None:
        if cond is None:
            cond = model.null_cond[None].expand(B, -1, -1)
        prefix = model.cond_proj(cond) + model.prefix_pos[None]
        h = torch.cat([prefix.to(model.token_embedding.weight.dtype),
                       _cond_token_h(model, input_ids, 0)], dim=1)
    else:
        h = _cond_token_h(model, input_ids, pos_offset)

    new_past = []
    for i, blk in enumerate(model.blocks):
        # bias=None -> fused causal kernel. Training used an explicit causal mask
        # over [prefix, tokens]; for real (non-pad) positions the two are
        # equivalent, same argument as VoxelTransformerAR2._CausalBlock.
        h, kv = blk(h, None, past_kv=None if past is None else past[i],
                    pos_offset=(0 if past is None else pos_offset + model.n_prefix))
        new_past.append(kv)
    logits = model.lm_head(model.norm(h))
    return logits[:, -1, :], new_past


@torch.no_grad()
def stream_cond_tokens(model, *, cond: Optional[torch.Tensor], bos_token_id: int,
                       eos_token_id: int, max_new_tokens: int,
                       temperature: float = 1.0, top_k: Optional[int] = 40,
                       cfg_scale: float = 1.0,
                       generator: Optional[torch.Generator] = None) -> Iterator[int]:
    """Yield sampled token ids from ``CondVoxelAR2``, with classifier-free guidance.

    ``cond`` is [1, n_prefix, cond_dim], or None for the unconditional branch (the
    learned ``null_cond`` that cond-dropout trained). With ``cfg_scale != 1`` two
    caches run in parallel -- conditional and unconditional -- and the logits are
    combined as ``uncond + scale * (cond - uncond)``.
    """
    device = next(model.parameters()).device
    model.eval()
    tokens = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
    past_c = past_u = None
    guided = cond is not None and cfg_scale != 1.0
    for step in range(max_new_tokens):
        step_in = tokens if past_c is None else tokens[:, -1:]
        pos_offset = tokens.shape[1] - step_in.shape[1]
        logits, past_c = _cond_logits(model, step_in, cond, past_c, pos_offset)
        if guided:
            logits_u, past_u = _cond_logits(model, step_in, None, past_u, pos_offset)
            logits = logits_u + cfg_scale * (logits - logits_u)
        nxt = _pick(logits, temperature, top_k, generator)
        tok = int(nxt.item())
        if tok == eos_token_id:
            return
        yield tok
        tokens = torch.cat([tokens, nxt], dim=1)
        if tokens.shape[1] + model.n_prefix >= model.max_seq_len:
            return
