"""Idea #7 semantic node-embedding prior: shape, contract, and integration tests.

These use an injected fake encoder so they run offline; the real CLIP build is
exercised by scripts/run_semantic_prior.py.

Run:  python -m pytest tests/test_semantic_embedding.py -q
"""

from __future__ import annotations

import numpy as np
import torch

from blockgen.models.semantic_embedding import (SemanticTokenEmbedding, block_text,
                                                build_semantic_embedding,
                                                build_semantic_matrix)
from blockgen.models.voxel_transformer_ar2 import VoxelTransformerAR2
from blockgen.utils.serialize import build_block_vocab
from blockgen.utils.data import Structure


def _toy_vocab():
    # Two structures over a handful of distinct blocks -> a small BlockVocab.
    a = Structure(block_ids=np.array([[[5, 35]]], np.int32),
                  block_data=np.array([[[0, 14]]], np.int32))
    b = Structure(block_ids=np.array([[[17, 4]]], np.int32),
                  block_data=np.array([[[0, 0]]], np.int32))
    return build_block_vocab([a, b], max_dim=8)


def _fake_encoder(dim=32):
    # Deterministic per-text vector so tests are reproducible without a download.
    def enc(texts):
        out = np.zeros((len(texts), dim), np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            out[i] = rng.standard_normal(dim)
        return out
    return enc


def test_block_text_reads_display_names():
    assert block_text(35, 14) == "red wool"
    assert block_text(5, 0) == "oak wood plank"
    # A block with no display name must still yield a usable phrase, not crash.
    assert "block" in block_text(9999, 0)


def test_semantic_matrix_is_normalized_and_shaped():
    vocab = _toy_vocab()
    mat = build_semantic_matrix(vocab, encode_fn=_fake_encoder(16))
    assert mat.shape == (vocab.num_blocks, 16)
    norms = mat.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_embedding_contract_and_block_rows_come_from_projection():
    vocab = _toy_vocab()
    emb = build_semantic_embedding(vocab, d_model=24, encode_fn=_fake_encoder(16))
    assert emb.embedding_dim == 24
    assert emb.vocab_size == vocab.vocab_size
    assert emb.weight.shape == (vocab.vocab_size, 24)

    ids = torch.arange(vocab.vocab_size)
    got = emb(ids)
    # Block rows must equal proj(semantic); special/coord rows must equal head table.
    block_rows = got[vocab.block_offset:]
    assert torch.allclose(block_rows, emb.proj(emb.semantic), atol=1e-6)
    assert torch.allclose(got[:vocab.block_offset], emb.head.weight, atol=1e-6)


def test_semantic_matrix_is_frozen_but_projection_trains():
    vocab = _toy_vocab()
    emb = build_semantic_embedding(vocab, d_model=24, encode_fn=_fake_encoder(16))
    assert "semantic" in dict(emb.named_buffers())          # frozen: a buffer
    assert not any(p is emb.semantic for p in emb.parameters())
    trainable = {n for n, p in emb.named_parameters() if p.requires_grad}
    assert any(n.startswith("proj") for n in trainable)
    assert any(n.startswith("head") for n in trainable)


def test_integrates_into_ar2_and_trains_one_step():
    vocab = _toy_vocab()
    emb = build_semantic_embedding(vocab, d_model=32, encode_fn=_fake_encoder(16))
    model = VoxelTransformerAR2(vocab_size=vocab.vocab_size, max_seq_len=32,
                                d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
                                pe="phase4", semantic_embedding=emb)
    tokens = torch.randint(0, vocab.vocab_size, (2, 10))
    logits = model(tokens)
    assert logits.shape == (2, 10, vocab.vocab_size)

    # One optimizer step must change proj but leave the frozen semantic matrix fixed.
    before = emb.semantic.clone()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = logits.reshape(-1, vocab.vocab_size).float().logsumexp(-1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    assert torch.equal(emb.semantic, before), "frozen semantic matrix changed"
    assert emb.proj.weight.grad is not None


def test_rejects_double_embedding_and_size_mismatch():
    vocab = _toy_vocab()
    emb = build_semantic_embedding(vocab, d_model=32, encode_fn=_fake_encoder(16))
    import pytest
    with pytest.raises(ValueError):
        VoxelTransformerAR2(vocab_size=vocab.vocab_size, max_seq_len=32, d_model=32,
                            nhead=4, num_layers=2, semantic_embedding=emb,
                            piece_factors=object())
    with pytest.raises(ValueError):  # d_model mismatch
        VoxelTransformerAR2(vocab_size=vocab.vocab_size, max_seq_len=32, d_model=64,
                            nhead=4, num_layers=2, semantic_embedding=emb)
