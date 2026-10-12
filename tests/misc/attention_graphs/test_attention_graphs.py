"""Attention-as-adjacency toolkit: spans, pooling, metrics, serialisations, both models.

Everything runs on CPU against randomly initialised tiny models (a two-layer Qwen2 built
from a config, a two-layer VoxelTransformerAR2) and a character-level stand-in
tokenizer, so nothing is downloaded and no checkpoint is read.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tools.misc.attention_graphs.adjacency import (
    auroc, average_precision, baseline_scores, causal_pairs, face_adjacency, hard_mask,
    parent_top1, score_matrix, stud_adjacency, tree_hops)
from tools.misc.attention_graphs.ar2 import block_attention, extract_ar2, tokens_for_order
from tools.misc.attention_graphs.experiment import Arm, run_probes, summarize
from tools.misc.attention_graphs.extract import (
    extract, head_scores, pool_to_nodes, renorm_rows, rollout, symmetrize)
from tools.misc.attention_graphs.probe import evaluate, probe_scores, train_probe
from tools.misc.attention_graphs.scenes import (
    largest_component, lego_scenes, synthetic_scenes, voxel_scene)
from tools.misc.attention_graphs.serialize import DIRS, order_scene, serialize


class CharTokenizer:
    """One token per character; ``add_special_tokens`` prepends a BOS id."""

    def __call__(self, text, add_special_tokens=True):
        ids = [ord(c) % 250 + 3 for c in text]
        return {"input_ids": ([1] if add_special_tokens else []) + ids}


@pytest.fixture(scope="module")
def tiny_llm():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048)
    cfg._attn_implementation = "eager"
    return Qwen2ForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def tiny_ar2():
    from blockgen.models.voxel_transformer_ar2 import VoxelTransformerAR2
    torch.manual_seed(0)
    return VoxelTransformerAR2(vocab_size=40, max_seq_len=128, d_model=32, nhead=4,
                               num_layers=2, dim_feedforward=64, dropout=0.0, pe="phase4").eval()


# ---------------------------------------------------------------- graphs + metrics

def test_face_adjacency_on_an_L():
    adj = face_adjacency(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 1, 1]]))
    assert adj[0, 1] and adj[0, 2] and not adj[1, 2] and not adj[3].any()
    assert (adj == adj.T).all() and not adj.diagonal().any()


def test_stud_adjacency_needs_overlap_and_consecutive_layers():
    boxes = [(0, 0, 0, 2, 4), (1, 1, 3, 2, 2), (4, 1, 0, 2, 2), (0, 2, 0, 2, 2)]
    adj = stud_adjacency(boxes)
    assert adj[0, 1] and not adj[0, 2] and not adj[0, 3]


def _connected(adj):
    seen, stack = {0}, [0]
    while stack:
        for j in np.flatnonzero(adj[stack.pop()]):
            if j not in seen:
                seen.add(int(j)); stack.append(int(j))
    return len(seen) == len(adj)


def test_lego_stacks_and_chunks_are_connected():
    assert all(_connected(s.adjacency) for s in lego_scenes(5, n_bricks=10, seed=1))
    m = np.zeros((4, 4, 4), bool)
    m[0, 0, :3] = True
    m[3, 3, 3] = True
    assert largest_component(m).sum() == 3


def test_metrics_match_hand_values():
    labels = np.array([1, 0, 1, 0], dtype=bool)
    assert auroc(np.array([0.9, 0.1, 0.8, 0.2]), labels) == 1.0
    assert auroc(np.zeros(4), labels) == 0.5                       # ties count half
    assert average_precision(np.array([0.9, 0.8, 0.7, 0.1]), labels) == pytest.approx((1 + 2 / 3) / 2)


def test_oracle_perfect_recency_tracks_raster_and_hard_pairs_defeat_matching():
    hut = synthetic_scenes()[3]
    raster, _ = order_scene(hut, "raster")
    b = baseline_scores(raster.coords, raster.names)
    m = {k: score_matrix(v[None], raster.adjacency, raster.coords) for k, v in b.items()}
    assert m["coord_L1"]["auc"][0] == 1.0 and m["coord_L1"]["hard"][0] == 1.0
    assert m["recency"]["auc"][0] > 0.7
    assert m["coord_match"]["hard"][0] == 0.5          # constant within hard pairs
    ii, jj = causal_pairs(len(raster))
    assert ((raster.coords[ii] == raster.coords[jj]).sum(-1)[hard_mask(raster.coords)] == 2).all()


def test_shuffle_permutes_adjacency_consistently():
    hut = synthetic_scenes()[3]
    sh, _ = order_scene(hut, "shuffle", seed=3)
    assert (sh.adjacency == face_adjacency(sh.coords)).all()
    assert sh.adjacency.sum() == hut.adjacency.sum()


def test_tree_hops_and_parent_top1():
    parents = [-1, 0, 0, 1]
    h = tree_hops(parents)
    assert h[3, 0] == 2 and h[3, 2] == 3 and h[1, 2] == 2
    S = np.zeros((4, 4))
    S[2, 0] = S[3, 1] = 1.0
    assert parent_top1(S, parents) == 1.0


# ---------------------------------------------------------------- serialisation

def test_absolute_spans_cover_exactly_each_line():
    tok = CharTokenizer()
    scene = voxel_scene([[0, 0, 0], [1, 0, 0], [12, 3, 4]], ["stone", "planks", "glass"])
    ser = serialize(scene, tok, order="raster")
    assert len(ser.spans) == 3 and ser.spans[-1][1] == len(ser.input_ids)
    for (s, e), line in zip(ser.spans, ser.lines):
        assert e - s == len(line) + 1
    assert ser.lines[0] == "0 0 0 stone"


@pytest.mark.parametrize("order", ["bfs", "random"])
def test_relative_format_is_coordinate_free_and_decodes_back(order):
    hut = synthetic_scenes()[3]
    ser = serialize(hut, CharTokenizer(), order=order, seed=5, fmt="relative")
    inv = {v: k for k, v in DIRS.items()}
    pos = {}
    for i, line in enumerate(ser.lines):
        parts = line.split()
        assert int(parts[0]) == i
        if parts[2] == "root":
            pos[i] = np.zeros(3, int)
        else:
            p = int(parts[3])
            assert p < i and ser.parents[i] == p
            pos[i] = pos[p] + np.array(inv[parts[4]])
    rebuilt = np.stack([pos[i] for i in range(len(pos))])
    assert (rebuilt - rebuilt.min(0) == ser.scene.coords - ser.scene.coords.min(0)).all()


# ---------------------------------------------------------------- pooling

def _mass_outside(att, spans):
    inside = torch.zeros(att.shape[-1])
    for s, e in spans:
        inside[s:e] = 1
    out = (att * (1 - inside)).sum(-1)
    return torch.stack([out[..., s:e].mean(-1) for s, e in spans], -1)


def test_pooling_rows_renorm_and_symmetry():
    att = torch.softmax(torch.randn(2, 3, 10, 10), -1)
    spans = [(1, 3), (3, 7), (7, 10)]
    node = pool_to_nodes(att, spans, query="mean", key="sum")
    assert torch.allclose(node.sum(-1), 1 - _mass_outside(att, spans), atol=1e-5)
    r = renorm_rows(node.numpy())
    assert np.allclose(r[..., 1:, :].sum(-1), 1) and np.allclose(np.triu(r), 0)
    s = symmetrize(node.numpy())
    assert np.allclose(s, np.swapaxes(s, -1, -2))
    one = pool_to_nodes(att, spans, query_spans=[(2, 3), (4, 5), (9, 10)])
    # a single-token query: node 1's row is token 4's attention summed per key span
    assert torch.allclose(one[..., 1, 1], att[..., 4, 3:7].sum(-1))
    assert torch.allclose(one[..., 1, 2], att[..., 4, 7:10].sum(-1))


def test_rollout_rows_stay_stochastic():
    r = rollout(torch.softmax(torch.randn(3, 2, 6, 6), -1))
    assert r.shape == (3, 6, 6) and torch.allclose(r.sum(-1), torch.ones(3, 6))


# ---------------------------------------------------------------- models

def test_llm_extract_end_to_end(tiny_llm):
    ser = serialize(synthetic_scenes()[2], CharTokenizer(), order="shuffle", seed=0)
    ex = extract(tiny_llm, ser.input_ids, ser.spans, hidden_layers=(0, 2), keep_tokens=True)
    n = len(ser.spans)
    assert ex.node_att.shape == (2, 4, n, n) and ex.rollout.shape == (2, n, n)
    assert ex.hidden[2].shape == (n, 32)
    assert np.allclose(np.triu(ex.node_att, 1), 0, atol=1e-6)        # causal


def test_ar2_recomputed_attention_matches_the_fused_kernel(tiny_ar2):
    blk = tiny_ar2.blocks[0]
    x = torch.randn(1, 9, 32)
    w = block_attention(blk, x)
    q, k, v = blk.qkv(blk.norm1(x)).chunk(3, dim=-1)
    v = v.view(1, 9, 4, 8).transpose(1, 2)
    q = q.view(1, 9, 4, 8).transpose(1, 2)
    k = k.view(1, 9, 4, 8).transpose(1, 2)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(w @ v, ref, atol=1e-5)


def test_ar2_extract_slots_and_order(tiny_ar2):
    toks = [1] + [5, 6, 7, 30, 6, 6, 7, 31, 5, 7, 7, 30] + [2]
    t2, perm = tokens_for_order(toks, "shuffle", seed=1)
    assert sorted(perm.tolist()) == [0, 1, 2] and len(t2) == len(toks)
    exs = extract_ar2(tiny_ar2, toks, hidden_layers=(0, 2))
    assert set(exs) == {"X", "Y", "Z", "B", "mean"}
    assert exs["Y"].node_att.shape == (2, 4, 3, 3) and exs["mean"].hidden[2].shape == (3, 32)


def test_arm_summary_and_probe_on_tiny_llm(tiny_llm):
    tok = CharTokenizer()
    arm = Arm("rel")
    for k, scene in enumerate(synthetic_scenes() * 2):
        ser = serialize(scene, tok, order="random", seed=k, fmt="relative")
        ex = extract(tiny_llm, ser.input_ids, ser.spans, hidden_layers=(1,))
        heads, roll = head_scores(ex)
        arm.add(ser.scene, heads, roll, ser.parents, ex.hidden)
    s = summarize(arm)
    assert {"auc", "strat", "ap", "hard", "parent_top1", "unstated_auc"} <= set(s["heads"])
    assert s["baselines"]["parent"]["parent_top1"] == 1.0
    probes = run_probes(arm, steps=5, kinds=("distance",))
    assert 0.0 <= probes["distance"][1]["auc"] <= 1.0


def test_probe_learns_adjacency_from_coordinates():
    rng = np.random.default_rng(0)
    feats, adjs = [], []
    for _ in range(12):
        c = np.unique(rng.integers(0, 4, size=(15, 3)), axis=0)
        feats.append(np.concatenate([c, rng.normal(size=(len(c), 5))], 1).astype(np.float32))
        adjs.append(face_adjacency(c))
    probe, mu, sd = train_probe(feats[:8], adjs[:8], kind="distance", rank=4, steps=200)
    assert evaluate(probe_scores(probe, mu, sd, feats[8:]), adjs[8:])["auc"] > 0.85
