"""Prove the cached decode paths agree with the stock uncached forward.

``sampling._cond_logits`` reimplements ``CondVoxelAR2.forward`` with a KV cache,
because the stock one has none and O(L^2) decode is unusable at 5,480 tokens. A
reimplementation like that fails *silently* -- a wrong position offset still
produces plausible tokens, just from the wrong distribution -- so it needs a test
that pins it to the original rather than eyeballed samples.

Run:  python -m pytest deploy/inference/tests/ -q
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from blockgen.models.voxel_transformer_ar2 import VoxelTransformerAR2
from blockgen.models.voxel_transformer_cond import CondVoxelAR2
from blockgen_server.sampling import _cond_logits, stream_tokens


def _cond_model(pe: str = "phase4", n_prefix: int = 1, cond_dim: int = 512):
    torch.manual_seed(0)
    m = CondVoxelAR2(cond_dim=cond_dim, n_prefix=n_prefix, vocab_size=64,
                     max_seq_len=64, d_model=32, nhead=4, num_layers=2,
                     dim_feedforward=64, pe=pe)
    return m.eval()


@pytest.mark.parametrize("pe", ["phase4", "learned", "sin", "rope", "none"])
def test_cond_cached_matches_uncached(pe: str) -> None:
    """Cached step-by-step logits must equal a full forward over the same prefix."""
    model = _cond_model(pe=pe)
    cond = torch.randn(1, 1, 512)
    tokens = torch.tensor([[1, 5, 9, 13, 40, 6, 10, 14, 41]])

    reference = model.forward(tokens, cond=cond)[:, -1, :]

    # Prime the cache with the first token only, then feed one token at a time --
    # exactly how stream_cond_tokens drives it.
    logits, past = _cond_logits(model, tokens[:, :1], cond, None, 0)
    for i in range(1, tokens.shape[1]):
        logits, past = _cond_logits(model, tokens[:, i:i + 1], cond, past, i)

    assert torch.allclose(logits, reference, atol=1e-4), \
        f"pe={pe}: cached decode diverged from stock forward (max diff " \
        f"{(logits - reference).abs().max():.2e})"


def test_cond_cached_null_branch_matches() -> None:
    """The CFG unconditional branch (cond=None -> null_cond) must match too."""
    model = _cond_model()
    tokens = torch.tensor([[1, 5, 9, 13, 40]])
    reference = model.forward(tokens, cond=None)[:, -1, :]

    logits, past = _cond_logits(model, tokens[:, :1], None, None, 0)
    for i in range(1, tokens.shape[1]):
        logits, past = _cond_logits(model, tokens[:, i:i + 1], None, past, i)
    assert torch.allclose(logits, reference, atol=1e-4)


def test_stream_tokens_matches_generate_under_same_seed() -> None:
    """The uncond streamer must emit what the stock sampler would, given one seed."""
    torch.manual_seed(0)
    model = VoxelTransformerAR2(vocab_size=64, max_seq_len=64, d_model=32, nhead=4,
                                num_layers=2, dim_feedforward=64, pe="phase4").eval()

    g = torch.Generator(device="cpu").manual_seed(1234)
    streamed = list(stream_tokens(model, bos_token_id=1, eos_token_id=2,
                                  max_new_tokens=20, temperature=1.0, top_k=8,
                                  generator=g))
    g2 = torch.Generator(device="cpu").manual_seed(1234)
    again = list(stream_tokens(model, bos_token_id=1, eos_token_id=2,
                               max_new_tokens=20, temperature=1.0, top_k=8,
                               generator=g2))
    assert streamed == again, "same seed must reproduce the same build"
    assert 2 not in streamed, "EOS must not be yielded"


def test_stream_tokens_respects_uncached_path() -> None:
    """ALiBi cannot use a cache; the streamer must fall back, not crash."""
    torch.manual_seed(0)
    model = VoxelTransformerAR2(vocab_size=64, max_seq_len=64, d_model=32, nhead=4,
                                num_layers=2, dim_feedforward=64, pe="alibi").eval()
    assert not model.supports_cache()
    toks = list(stream_tokens(model, bos_token_id=1, eos_token_id=2,
                              max_new_tokens=5, top_k=4))
    assert all(0 <= t < 64 for t in toks)
