"""The geometric attention bias: causality, reachability, and the two-implementation trap.

`GeoBias.matrix` builds the full `[T, T]` bias used in training; `GeoBias.row` builds
one query's row for KV-cached sampling. Two implementations of one function is how a
model that trains well starts sampling badly, so their equality is pinned here.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

from train_llm_geobias import GeoBias, causal_mask  # noqa: E402


def _fixture():
    """3 prompt tokens, then 4 lines of varying token counts, with known anchors."""
    tok_line = torch.tensor([-1, -1, -1] + [0] * 4 + [1] * 3 + [2] * 5 + [3] * 4)
    anchors = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 0, 2], [3, 1, 2]])
    geo = GeoBias(radius=4)
    torch.manual_seed(0)
    with torch.no_grad():
        geo.table.normal_(std=1.0)
        geo.prompt.fill_(-0.3)
        geo.same_line.fill_(0.2)
        geo.no_ref.fill_(0.7)
    return geo, tok_line, anchors


def test_row_matches_matrix_last_row():
    """`row()` is the incremental form of `matrix()`, for every line."""
    geo, tok_line, anchors = _fixture()
    full = geo.matrix(tok_line, anchors)
    for q in range(tok_line.numel()):
        q_line = int(tok_line[q])
        if q_line < 0:
            continue
        row = geo.row(tok_line[: q + 1], anchors, q_line)
        assert torch.allclose(row, full[q, : q + 1], atol=1e-6), f"drift at token {q}"


def test_no_anchor_leaks_backward():
    """A query on line L must not see line L's own anchor.

    The anchor is only known once the line's coordinates have been emitted, so any
    dependence on it inside the line would be information the sampler cannot supply.
    Every same-line entry must therefore be anchor-free: the shared `same_line`
    scalar, or `no_ref` on line 0 where the row is a constant anyway.
    """
    geo, tok_line, anchors = _fixture()
    full = geo.matrix(tok_line, anchors)
    for L in range(int(tok_line.max()) + 1):
        q = (tok_line == L).nonzero().flatten()
        block = full[q][:, q]
        want = geo.no_ref if L == 0 else geo.same_line
        assert torch.allclose(block, want.expand_as(block))


def test_special_precedence():
    """`no_ref` > `prompt` > `same_line` > table -- the order the `where`s apply.

    Load-bearing: line 0 has no reference anchor, so its entire row is `no_ref`
    including the prompt columns. Asserting the precedence here is what keeps
    `row()` and `matrix()` from being "fixed" apart later.
    """
    geo, tok_line, anchors = _fixture()
    full = geo.matrix(tok_line, anchors)
    prompt = (tok_line < 0).nonzero().flatten()
    later = (tok_line > 0).nonzero().flatten()
    assert torch.allclose(full[later][:, prompt],
                          geo.prompt.expand_as(full[later][:, prompt]))
    first = (tok_line == 0).nonzero().flatten()
    assert torch.allclose(full[first], geo.no_ref.expand_as(full[first]))


def test_zero_init_is_exactly_the_control():
    """A zero table must leave the mask bit-identical to a plain causal mask."""
    _, tok_line, anchors = _fixture()
    T = tok_line.numel()
    zero = GeoBias(radius=4)
    with_bias = causal_mask(T, torch.float32, "cpu", zero.matrix(tok_line, anchors))
    plain = causal_mask(T, torch.float32, "cpu", None)
    assert torch.equal(with_bias["full_attention"], plain["full_attention"])


@pytest.mark.parametrize("radius", [2, 4])
def test_table_shape(radius):
    geo = GeoBias(radius=radius)
    assert geo.table.numel() == (2 * radius + 1) ** 3
