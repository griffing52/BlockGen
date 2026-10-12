"""Nodes-in, adjacency-out prototype: label space, tying, equivariance, realization, decode.

CPU only, random tiny models, hand-built graphs; nothing is read from disk.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.misc.adjacency_gen.data import (NodeVocab, OffsetSpace, ValidTable, contact_offsets,
                                           edge_labels, graph_to_structure, touching)
from tools.misc.adjacency_gen.decode import Placement, TypePrior, decode, model_scorer, random_tree, realize
from tools.misc.adjacency_gen.model import AdjacencyModel, ModelConfig, obs_codes
from tools.misc.adjacency_gen.train import ShapeIndex, collate, forward_batch, masked_loss, prepare


def voxel_vocab(n=4):
    return NodeVocab(kind="voxel", n_types=n, names=[f"t{i}" for i in range(n)],
                     shapes={i: np.zeros((1, 3), np.int64) for i in range(n)}, space=OffsetSpace(1))


def piece_vocab_small():
    shapes = {0: np.zeros((1, 3), np.int64), 1: np.array([[0, 0, 0], [1, 0, 0]]),
              2: np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0]])}
    return NodeVocab(kind="piece", n_types=3, names=["a", "b", "c"], shapes=shapes, space=OffsetSpace(2))


L_SHAPE = np.array([(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (2, 2, 0), (0, 0, 1)])


def test_offset_space_roundtrip_and_opp():
    for B in (1, 2, 4):
        sp = OffsetSpace(B)
        offs = sp.offsets
        assert len(offs) == sp.K
        for k, d in enumerate(offs, start=1):
            assert sp.cls(d) == k
        opp = sp.opp
        assert opp[0] == 0
        assert np.array_equal(opp[opp], np.arange(sp.K + 1))
        assert np.array_equal(offs[opp[1:] - 1], -offs)
    assert OffsetSpace(1).cls((2, 0, 0)) == -1


def test_contact_offsets_voxel_and_piece():
    one = np.zeros((1, 3), np.int64)
    assert contact_offsets(one, one) == {tuple(v) for v in
                                         [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]}
    bar = np.array([[0, 0, 0], [1, 0, 0]])
    c = contact_offsets(bar, one)
    assert (2, 0, 0) in c and (-1, 0, 0) in c and (1, 1, 0) in c
    assert (0, 0, 0) not in c and (1, 0, 0) not in c          # overlaps
    vt = ValidTable({0: one}, OffsetSpace(1))
    assert vt.mask(0, 0).sum() == 7                           # none + 6 faces


def test_edge_labels_antisymmetric():
    sp = OffsetSpace(1)
    lab, dropped = edge_labels(L_SHAPE, [np.zeros((1, 3), int)] * len(L_SHAPE), sp)
    assert dropped == 0
    assert np.array_equal(lab > 0, (lab > 0).T)
    assert np.array_equal(sp.opp[lab], lab.T)
    assert (lab > 0).sum() // 2 == 5                          # a tree


def tiny_model(vocab, rnf=4):
    torch.manual_seed(0)
    return AdjacencyModel(ModelConfig(n_types=vocab.n_types, B=vocab.space.B, d=32, layers=2, heads=4,
                                      d_rel=8, d_pair=16, rnf=rnf)).eval()


@pytest.mark.parametrize("vocab_fn", [voxel_vocab, piece_vocab_small])
def test_logits_tied_antisymmetric(vocab_fn):
    vocab = vocab_fn()
    m = tiny_model(vocab)
    N = 7
    types = torch.randint(0, vocab.n_types, (2, N))
    pad = torch.zeros(2, N, dtype=torch.bool)
    labels = torch.zeros(2, N, N, dtype=torch.long)
    labels[:, 0, 1] = 1
    labels[:, 1, 0] = int(vocab.space.opp[1])
    obs = obs_codes(labels, torch.rand(2, N, N) < 0.5)
    z = m(types, pad, obs)
    opp = torch.as_tensor(vocab.space.opp)
    assert torch.allclose(z, z.transpose(1, 2)[..., opp], atol=1e-5)


def test_permutation_equivariance():
    vocab = voxel_vocab()
    m = tiny_model(vocab)
    m.set_rnf(False)                                   # RNF is equivariant only in distribution
    N = 6
    types = torch.tensor([[0, 1, 1, 2, 3, 1]])
    pad = torch.zeros(1, N, dtype=torch.bool)
    lab, _ = edge_labels(L_SHAPE, [np.zeros((1, 3), int)] * N, vocab.space)
    lab = torch.as_tensor(lab)[None]
    observed = torch.zeros(1, N, N, dtype=torch.bool)
    observed[0, :3, :3] = True
    obs = obs_codes(lab, observed)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    z = m(types, pad, obs)
    zp = m(types[:, perm], pad, obs[:, perm][:, :, perm])
    assert torch.allclose(zp, z[:, perm][:, :, perm], atol=1e-5)


def test_realize_recovers_hand_built_graph():
    sp = OffsetSpace(1)
    shapes = [np.zeros((1, 3), int)] * len(L_SHAPE)
    lab, _ = edge_labels(L_SHAPE, shapes, sp)
    r = realize(lab, np.ones_like(lab, dtype=float), shapes, sp)
    assert not r["conflict"].any() and r["edge_ok"] == 1.0
    assert len(set(r["comp"].tolist())) == 1
    assert np.array_equal(r["pos"] - r["pos"][0], L_SHAPE - L_SHAPE[0])


def test_realize_detects_collision_and_contradiction():
    sp = OffsetSpace(1)
    shapes = [np.zeros((1, 3), int)] * 3
    lab = np.zeros((3, 3), int)
    px, mx = sp.cls((1, 0, 0)), sp.cls((-1, 0, 0))
    lab[0, 1], lab[1, 0] = px, mx          # 1 = 0 + x
    lab[0, 2], lab[2, 0] = px, mx          # 2 = 0 + x  -> same cell as 1
    r = realize(lab, np.ones((3, 3)), shapes, sp)
    assert r["conflict"].sum() == 1
    # a cycle that does not close: 0->1 +x, 1->2 +y, 2->0 +x (should be -x -y)
    lab = np.zeros((3, 3), int)
    for (i, j, d) in [(0, 1, (1, 0, 0)), (1, 2, (0, 1, 0)), (2, 0, (1, 0, 0))]:
        lab[i, j], lab[j, i] = sp.cls(d), sp.cls(tuple(-np.array(d)))
    r = realize(lab, np.ones((3, 3)), shapes, sp)
    assert r["edge_ok"] < 1.0


def test_placement_rejects_overlap():
    pl = Placement([np.zeros((1, 3), int)] * 3)
    assert pl.try_merge(0, 1, (1, 0, 0))
    assert not pl.try_merge(0, 2, (1, 0, 0))          # cell taken by node 1
    assert pl.try_merge(1, 2, (1, 0, 0))
    assert np.array_equal(pl.pos[2] - pl.pos[0], [2, 0, 0])


@pytest.mark.parametrize("vocab_fn", [voxel_vocab, piece_vocab_small])
def test_decode_tiny_model(vocab_fn):
    vocab = vocab_fn()
    m = tiny_model(vocab)
    sidx = ShapeIndex(vocab)
    sc = model_scorer(m, sidx.type_valid(), torch.device("cpu"))
    types = np.array([0, 1, 1, 2, 0, 1, 2, 2])
    shapes = [vocab.shapes[t] for t in types]
    res = decode(types, vocab, sc, steps=4, check=True)
    assert len(set(res["comp"].tolist())) == 1                         # connected
    lab = res["labels"]
    assert np.array_equal(vocab.space.opp[lab], lab.T)
    # committed labels agree with the placed geometry, no voxel collisions
    true_lab, _ = edge_labels(res["pos"], shapes, vocab.space)
    assert np.array_equal(true_lab, lab)
    s = graph_to_structure(res["pos"], [[(*v, 1, 0) for v in sh.tolist()] for sh in shapes])
    assert s.occupied_mask.sum() == sum(len(sh) for sh in shapes)
    res2 = decode(types, vocab, sc, steps=4, check=False, temperature=1.0)
    assert np.array_equal(vocab.space.opp[res2["labels"]], res2["labels"].T)
    r = realize(res2["labels"], res2["conf"], shapes, vocab.space)
    assert r["pos"].shape == (len(types), 3)


def test_training_step_and_baselines():
    from tools.misc.adjacency_gen.data import Graph
    vocab = voxel_vocab()
    gs = [Graph(types=np.array([0, 1, 1, 2, 3, 1]), coords=L_SHAPE.copy(),
                blocks=[[(0, 0, 0, 1, 0)]] * 6) for _ in range(3)]
    sidx = ShapeIndex(vocab)
    items = prepare(gs, vocab, sidx)
    m = tiny_model(vocab).train()
    batch = collate(items, torch.device("cpu"), [1.0, 0.5, 0.25], np.random.default_rng(0))
    z, labels, masked = forward_batch(m, sidx, batch, torch.device("cpu"))
    loss = masked_loss(z, labels, masked)
    loss.backward()
    assert torch.isfinite(loss)
    prior = TypePrior(gs, vocab, sidx.type_valid())
    res = decode(gs[0].types, vocab, prior.scorer(), steps=4, check=True, temperature=1.0)
    assert len(set(res["comp"].tolist())) == 1
    rt = random_tree(gs[0].types, vocab, sidx.type_valid(), check=True)
    assert not rt["conflict"].any()
    assert touching(rt["pos"], [np.zeros((1, 3), int)] * 6).sum() >= 10   # spanning tree


def test_prefix_mask_and_grow_decode():
    from tools.misc.adjacency_gen.train import prefix_mask
    sp = OffsetSpace(1)
    lab, _ = edge_labels(L_SHAPE, [np.zeros((1, 3), int)] * len(L_SHAPE), sp)
    pm = prefix_mask(lab, np.random.default_rng(1))
    obs = ~pm
    placed = obs.any(1)
    assert np.array_equal(pm, pm.T)
    assert 1 <= placed.sum() < len(L_SHAPE) or obs.sum() == 0
    vocab = voxel_vocab()
    m = tiny_model(vocab)
    sc = model_scorer(m, ShapeIndex(vocab).type_valid(), torch.device("cpu"))
    res = decode(np.array([0, 1, 1, 2, 0, 1]), vocab, sc, steps=3, check=True, grow=True)
    assert len(set(res["comp"].tolist())) == 1
