"""Idea #6 conditioning-channel resampler: shapes, masking, CFG, gradients.

Run:  python -m pytest tests/test_condition_resampler.py -q
"""

from __future__ import annotations

import torch

from blockgen.models.condition_resampler import (ConditionResampler,
                                                 ResampledCondVoxelAR2)


def test_resampler_shape_and_fixed_output_length():
    r = ConditionResampler(cond_dim=512, d_model=64, n_queries=8, n_layers=2, nhead=4)
    # Variable-length condition sequences both collapse to n_queries prefix vectors.
    for S in (1, 5, 77):
        out = r(torch.randn(3, S, 512))
        assert out.shape == (3, 8, 64)


def test_resampler_mask_ignores_padded_positions():
    torch.manual_seed(0)
    r = ConditionResampler(cond_dim=32, d_model=48, n_queries=4, n_layers=2, nhead=4)
    r.eval()
    real = torch.randn(1, 3, 32)
    padded = torch.cat([real, torch.randn(1, 5, 32)], dim=1)   # 3 real + 5 junk
    mask = torch.tensor([[False, False, False, True, True, True, True, True]])
    with torch.no_grad():
        a = r(real)
        b = r(padded, cond_mask=mask)
    # Output must depend only on the unmasked positions.
    assert torch.allclose(a, b, atol=1e-5)


def _model(**kw):
    return ResampledCondVoxelAR2(
        cond_dim=32, n_prefix=4, vocab_size=48, max_seq_len=48, d_model=32, nhead=4,
        num_layers=2, dim_feedforward=64, pe="phase4", resampler_layers=2,
        resampler_heads=4, **kw).eval()


def test_forward_consumes_sequence_condition():
    model = _model()
    tokens = torch.randint(0, 48, (2, 9))
    cond = torch.randn(2, 6, 32)                # a 6-token condition sequence
    logits = model(tokens, cond=cond)
    assert logits.shape == (2, 9, 48)


def test_null_branch_and_cfg_differ_from_conditional():
    torch.manual_seed(0)
    model = _model()
    tokens = torch.randint(0, 48, (1, 7))
    cond = torch.randn(1, 6, 32)
    lc = model(tokens, cond=cond)[:, -1]
    lu = model(tokens, cond=None)[:, -1]        # learned null prefix (CFG branch)
    assert lc.shape == lu.shape == (1, 48)
    assert not torch.allclose(lc, lu), "conditional and null branches must differ"


def test_gradients_reach_resampler_and_queries():
    model = _model()
    tokens = torch.randint(0, 48, (2, 8))
    cond = torch.randn(2, 5, 32)
    loss = model(tokens, cond=cond).float().logsumexp(-1).mean()
    loss.backward()
    assert model.resampler.queries.grad is not None
    assert model.resampler.cond_proj.weight.grad is not None
    assert model.null_prefix.grad is None  # null branch not exercised this step


def test_generate_cond_runs_with_pooled_cond_on_base_class():
    # The base CondVoxelAR2 path (pooled cond) must still work after the refactor.
    from blockgen.models.voxel_transformer_cond import CondVoxelAR2
    m = CondVoxelAR2(cond_dim=32, n_prefix=1, vocab_size=48, max_seq_len=48, d_model=32,
                     nhead=4, num_layers=2, dim_feedforward=64, pe="phase4").eval()
    toks = m.generate_cond(cond=torch.randn(2, 1, 32), bos_token_id=1, eos_token_id=2,
                           max_new_tokens=5, cfg_scale=2.0)
    assert toks.shape[0] == 2
