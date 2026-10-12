"""The per-head geometric attention bias: grammar, causality, and flex == dense.

The property that matters most is the one a training run cannot tell you about:
the bias at a query position must depend only on coordinates the model has already
emitted. Getting that wrong trains a model that cannot sample.
"""

from __future__ import annotations

import pytest
import torch

from blockgen.models.geo_attention_bias import (GeoAttentionBias, anchors_from_tokens,
                                                piece_and_slot)

COORD_OFFSET, MAX_DIM = 3, 32


def _tokens(anchors, pieces=None):
    """`[BOS, (X,Y,Z,PIECE)*, EOS]` for the given anchors."""
    pieces = pieces or [100 + i for i in range(len(anchors))]
    t = [1]
    for (x, y, z), p in zip(anchors, pieces):
        t += [COORD_OFFSET + x, COORD_OFFSET + y, COORD_OFFSET + z, p]
    t.append(2)
    return torch.tensor([t])


ANCHORS = [(1, 0, 0), (1, 0, 2), (3, 1, 2), (5, 4, 4)]


def test_piece_and_slot_matches_the_grammar():
    pos = torch.arange(10)
    piece, slot = piece_and_slot(pos)
    assert piece.tolist() == [-1, 0, 0, 0, 0, 1, 1, 1, 1, 2]
    assert slot.tolist() == [-1, 0, 1, 2, 3, 0, 1, 2, 3, 0]


def test_anchors_are_read_back_out_of_the_tokens():
    tok = _tokens(ANCHORS)
    got = anchors_from_tokens(tok, COORD_OFFSET, MAX_DIM, len(ANCHORS))
    assert got[0].tolist() == [list(a) for a in ANCHORS]


def test_reference_is_this_piece_once_its_z_is_emitted():
    """Slots Z and PIECE reason from their own anchor; X and Y from the previous."""
    geo = GeoAttentionBias(nhead=2, radius=4)
    tok = _tokens(ANCHORS)
    ref_axes, _, _, ref_ok = geo.buckets(tok, COORD_OFFSET, MAX_DIM)

    def ref_at(pos):
        return [int(ax[0, pos]) for ax in ref_axes]

    for k, a in enumerate(ANCHORS):
        base = 1 + 4 * k
        assert ref_at(base + 2) == list(a), "Z slot should see its own anchor"
        assert ref_at(base + 3) == list(a), "PIECE slot should see it too"
        if k > 0:
            assert ref_at(base) == list(ANCHORS[k - 1])
            assert ref_at(base + 1) == list(ANCHORS[k - 1])
    assert not bool(ref_ok[0, 1]) and not bool(ref_ok[0, 2]), "piece 0 X/Y have no ref"


def test_no_coordinate_leaks_backward():
    """Changing a piece's Z must not change the bias at that piece's X or Y query.

    Those positions precede the Z token, so any dependence on it would be
    information the sampler cannot supply -- the model would train on a signal that
    does not exist at generation time.
    """
    geo = GeoAttentionBias(nhead=2, radius=4)
    with torch.no_grad():
        geo.table.normal_(std=1.0)
    a = list(ANCHORS)
    b = list(ANCHORS)
    b[2] = (b[2][0], b[2][1], b[2][2] + 3)               # move piece 2 along z
    d_a = geo.dense(_tokens(a), COORD_OFFSET, MAX_DIM)
    d_b = geo.dense(_tokens(b), COORD_OFFSET, MAX_DIM)
    base = 1 + 4 * 2
    for slot in (0, 1):                                   # X and Y of piece 2
        assert torch.equal(d_a[:, :, base + slot, :base + slot + 1],
                           d_b[:, :, base + slot, :base + slot + 1])
    assert not torch.equal(d_a[:, :, base + 2, :], d_b[:, :, base + 2, :]), \
        "the Z slot SHOULD react to its own anchor"


def test_same_piece_keys_get_the_shared_bucket():
    geo = GeoAttentionBias(nhead=2, radius=4)
    with torch.no_grad():
        geo.table.normal_(std=1.0)
    d = geo.dense(_tokens(ANCHORS), COORD_OFFSET, MAX_DIM)
    base = 1 + 4 * 1
    block = d[0, :, base:base + 4, base:base + 4]
    want = geo.table[:, geo.n_geo].view(-1, 1, 1).expand_as(block)
    assert torch.allclose(block, want)


def test_bos_and_eos_get_the_special_bucket():
    geo = GeoAttentionBias(nhead=2, radius=4)
    with torch.no_grad():
        geo.table.normal_(std=1.0)
    d = geo.dense(_tokens(ANCHORS), COORD_OFFSET, MAX_DIM)
    col = d[0, :, :, 0]                                   # everything attending to BOS
    want = geo.table[:, geo.n_geo + 1].view(-1, 1).expand_as(col)
    assert torch.allclose(col, want)


def test_zero_init_is_exactly_no_bias():
    geo = GeoAttentionBias(nhead=4, radius=4)
    d = geo.dense(_tokens(ANCHORS), COORD_OFFSET, MAX_DIM)
    assert torch.equal(d, torch.zeros_like(d))


def test_heads_are_independent():
    geo = GeoAttentionBias(nhead=3, radius=2)
    with torch.no_grad():
        geo.table.zero_()
        geo.table[1].fill_(0.5)
    d = geo.dense(_tokens(ANCHORS), COORD_OFFSET, MAX_DIM)
    assert torch.equal(d[:, 0], torch.zeros_like(d[:, 0]))
    assert torch.allclose(d[:, 1], torch.full_like(d[:, 1], 0.5))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex_attention needs CUDA")
def test_flex_score_mod_matches_the_dense_oracle():
    """The two paths must agree: the dense one is testable, the flex one is trainable."""
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    torch.manual_seed(0)
    dev = "cuda"
    geo = GeoAttentionBias(nhead=4, radius=4).to(dev)
    with torch.no_grad():
        geo.table.normal_(std=0.5)
    anchors = [(i % 5, (i // 5) % 4, (2 * i) % 6) for i in range(30)]
    tok = _tokens(anchors).to(dev)
    L = tok.shape[1]
    q, k, v = (torch.randn(1, 4, L, 16, device=dev) for _ in range(3))

    def causal(b, h, qi, ki):
        return qi >= ki

    bm = create_block_mask(causal, B=None, H=None, Q_LEN=L, KV_LEN=L, device=dev)
    got = flex_attention(q, k, v, score_mod=geo.make_score_mod(tok, COORD_OFFSET, MAX_DIM),
                         block_mask=bm)

    bias = geo.dense(tok, COORD_OFFSET, MAX_DIM)
    mask = torch.ones(L, L, dtype=torch.bool, device=dev).triu(1)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=bias.masked_fill(mask, float("-inf")))
    assert torch.allclose(got, ref, atol=2e-3), (got - ref).abs().max()
