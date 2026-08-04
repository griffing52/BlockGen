"""The `model=` path in `train_from_sequences` — bring-your-own-architecture.

`docs/custom-model.md` tells people the contract is exactly three things
(`forward(input_ids, pad_mask)`, `.vocab_size`, `.max_seq_len`). These tests are
what keeps that promise true, and they pin the two guards that turn a silent
wrong-output into a loud failure: a vocab that does not match the sequences, and
passing both a model and instructions for building one.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from blockgen.training.train_ar import ARTrainConfig
from blockgen.training.train_ar_ext import train_from_sequences


class MinimalModel(nn.Module):
    """Everything a custom architecture is required to provide, and nothing else."""

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, pad_mask=None):
        return self.lm_head(self.embed(input_ids))


def _seqs(n=12, length=20, vocab=32):
    rng = torch.Generator().manual_seed(0)
    return [torch.randint(3, vocab, (length,), generator=rng).tolist() for _ in range(n)]


def _cfg(**kw):
    base = dict(max_seq_len=64, epochs=1, batch_size=4, device="cpu",
                amp=False, log_every=99)
    base.update(kw)
    return ARTrainConfig(**base)


def test_custom_model_trains_and_is_returned():
    model = MinimalModel(vocab_size=32, max_seq_len=64)
    out, hist = train_from_sequences(_seqs(), 32, _cfg(), model=model)
    assert out is model, "the trained instance should be handed back"
    assert len(hist["loss"]) == 1
    assert all(x == x for x in hist["loss"]), "loss must not be NaN"


def test_custom_model_actually_updates_weights():
    model = MinimalModel(vocab_size=32, max_seq_len=64)
    before = model.lm_head.weight.detach().clone()
    train_from_sequences(_seqs(), 32, _cfg(epochs=2), model=model)
    assert not torch.allclose(before, model.lm_head.weight)


def test_vocab_mismatch_is_rejected():
    """A same-shape mismatch would otherwise decode silently to wrong blocks."""
    model = MinimalModel(vocab_size=16, max_seq_len=64)
    with pytest.raises(ValueError, match="vocab"):
        train_from_sequences(_seqs(vocab=32), 32, _cfg(), model=model)


def test_model_and_pe_together_is_rejected():
    model = MinimalModel(vocab_size=32, max_seq_len=64)
    with pytest.raises(ValueError, match="not both"):
        train_from_sequences(_seqs(), 32, _cfg(), pe="phase4", model=model)


def test_stock_path_is_unchanged():
    """The default behaviour must be identical to before `model=` existed."""
    out, hist = train_from_sequences(_seqs(), 32, _cfg())
    from blockgen.models.voxel_transformer_ar import VoxelTransformerAR
    assert isinstance(out, VoxelTransformerAR)
    assert len(hist["loss"]) == 1


def test_pe_path_still_builds_ar2():
    out, _ = train_from_sequences(_seqs(), 32, _cfg(), pe="phase4")
    from blockgen.models.voxel_transformer_ar2 import VoxelTransformerAR2
    assert isinstance(out, VoxelTransformerAR2)


def test_custom_model_can_be_sampled_from():
    """A trained custom model must work with the shared sampler."""
    from blockgen.training.train_ar_ext import generate_from_prefix

    model = MinimalModel(vocab_size=32, max_seq_len=64)
    train_from_sequences(_seqs(), 32, _cfg(), model=model)
    toks = generate_from_prefix(model.eval(), [1], eos_token_id=2,
                                max_new_tokens=16, temperature=1.0, top_k=8)
    assert len(toks) >= 1
    assert all(isinstance(t, int) for t in toks)
