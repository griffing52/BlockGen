"""Two-stage full-house generator: cell assignments, grow-in-cells, merge-in-cells, metrics.

CPU only, tiny random models, synthetic houses; nothing is read from disk.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.misc.adjacency_gen.cells import (S, House, assignment, assignment_l1, cell_of, house_coarse, inventory,
                                            ipf, prior_cell_dist, sample_assignment)
from tools.misc.adjacency_gen.eval_cells import build_metrics, glue_counts, upsample_assignment, _grid
from tools.misc.adjacency_gen.faces import multiset_f1, typed_glue
from tools.misc.adjacency_gen.grow_cells import (GrowConfig, GrowNet, frontier, grow_decode, grow_loss,
                                                 growth_order, make_state)
from tools.misc.adjacency_gen.merge_cells import (MState, MergeConfig, MergeNet, blocks_from_assignment,
                                                  legal_mask, make_example, merge_cells_decode, merge_loss,
                                                  real_contacts, state_features, state_from_partition,
                                                  static_candidates, targets)
from tools.misc.adjacency_gen.stage1 import CellAssignModel, Stage1Config, stage1_items, stage1_loss

T = 6


def house(seed=0, n=120):
    """A random connected blob (bottom row on the ground) with 5 block types."""
    rng = np.random.default_rng(seed)
    cells = {(10, 0, 10)}
    D = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    while len(cells) < n:
        c = list(cells)[rng.integers(len(cells))]
        d = D[rng.integers(6)]
        nc = tuple(int(v) for v in np.clip(np.array(c) + d, 0, 20))
        cells.add(nc)
    c = np.array(sorted(cells), np.int64)
    c = c - c.min(0)
    t = rng.integers(1, T, len(c))
    return House(coords=c, types=t, raw=np.stack([t + 1, np.zeros_like(t)], 1), build=seed)


# --------------------------------------------------------------------------- #
# stage 1
# --------------------------------------------------------------------------- #
def test_assignment_margins_and_sampling():
    h = house(1)
    A = assignment(h, 4, T)
    cond = house_coarse(h, 4, T)
    inv = inventory(h.types, T)
    assert (A.sum(1) == cond.count.ravel()).all() and (A.sum(0) == inv).all()
    P = prior_cell_dist(cond, inv)
    Q = ipf(P, cond.count.ravel(), inv)
    assert np.allclose(Q.sum(1), cond.count.ravel(), atol=1e-3) and np.allclose(Q.sum(0), inv, atol=1e-3)
    rng = np.random.default_rng(0)
    draws = [sample_assignment(P, cond.count.ravel(), inv, rng) for _ in range(4)]
    for B in draws:                                   # exact margins, every sample
        assert (B.sum(1) == cond.count.ravel()).all() and (B.sum(0) == inv).all()
    assert len({B.tobytes() for B in draws}) > 1      # a distribution, not a function
    assert assignment_l1(A, A) == 0.0
    assert 0 < assignment_l1(A, draws[0]) <= 1


def test_stage1_model_starts_at_prior_and_trains():
    torch.manual_seed(0)
    items = stage1_items([house(s) for s in range(3)], 4, T)
    m = CellAssignModel(Stage1Config(n_types=T, f=4, d=32, layers=1, heads=2))
    from tools.misc.adjacency_gen.stage1 import stage1_probs
    P = stage1_probs(m, items, torch.device("cpu"))[0]
    it = items[0]
    occ = it["count"] > 0
    Pp = prior_cell_dist(it["coarse"], it["inv"])
    Pp = Pp / Pp.sum(1, keepdims=True)
    assert np.allclose(P[occ], Pp[occ], atol=1e-5)    # zero-initialized residual over the prior
    assert (P[:, it["inv"] == 0] == 0).all()
    loss = stage1_loss(m, items, torch.device("cpu"))
    loss.backward()
    assert torch.isfinite(loss)


# --------------------------------------------------------------------------- #
# grow in cells
# --------------------------------------------------------------------------- #
def test_growth_order_is_an_attachment_order():
    h = house(2)
    o = growth_order(h.coords, np.random.default_rng(0))
    assert sorted(o.tolist()) == list(range(len(h)))
    seen = {tuple(h.coords[o[0]])}
    for i in o[1:]:
        c = h.coords[i]
        assert any(tuple(c + d) in seen for d in np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                                                            (0, 0, 1), (0, 0, -1)]))
        seen.add(tuple(c))


def _batch(states):
    return {k: torch.as_tensor(np.stack([s[k] for s in states]).astype(np.int64 if k != "rem" else np.float32))
            for k in states[0]}


def test_frontier_and_state_consistency():
    h = house(3)
    A = assignment(h, 4, T)
    st = make_state(h, A, 4, T, np.random.default_rng(0), k=40)
    assert st["rem"].min() >= 0 and st["rem"].sum() == len(h) - 40
    assert int((st["grid"] > 0).sum()) == 40
    b = _batch([st])
    front, seed = frontier(b["grid"], b["rem"], 4)
    assert not bool(seed[0])
    occ = b["grid"][0] > 0
    fr = front[0]
    assert not (fr & occ).any()
    # every legal cell touches the build and lies in a coarse cell with slots left
    from tools.misc.adjacency_gen.grow_cells import face_dilate, fine_cell_index
    assert (fr <= face_dilate(occ[None])[0]).all()
    fc = fine_cell_index(4, torch.device("cpu"))
    assert (b["rem"][0].sum(-1)[fc][fr] > 0).all()
    # every unplaced real block next to the build is legal (the target is reachable)
    real = b["real"][0] > 0
    assert (fr[real & ~occ & face_dilate(occ[None])[0]]).all()


def test_grow_loss_and_decode_respect_assignment():
    torch.manual_seed(0)
    hs = [house(4), house(5, n=80)]
    m = GrowNet(GrowConfig(n_types=T, f=4, e=8, c1=8, c2=8, d=16, layers=1, heads=2, dk=8))
    rng = np.random.default_rng(0)
    b = _batch([make_state(h, assignment(h, 4, T), 4, T, rng) for h in hs])
    r = grow_loss(m, b, torch.device("cpu"), 4)
    r["loss"].backward()
    assert torch.isfinite(r["loss"])
    A = [assignment(h, 4, T) for h in hs]
    C = [house_coarse(h, 4, T) for h in hs]
    for pol, temp in (("model", 0.0), ("model", 1.0), ("random", 1.0)):
        out = grow_decode(m.eval(), hs, A, C, torch.device("cpu"), 4, temperature=temp, steps=16, policy=pol)
        for o, h, Ah in zip(out, hs, A):
            assert len(o["coords"]) == len(h)                      # every slot is used
            assert len({tuple(c) for c in o["coords"].tolist()}) == len(h)
            got = np.zeros_like(Ah)
            np.add.at(got, (cell_of(o["coords"], 4), o["types"]), 1)
            assert (got == Ah).all()                               # exactly the assignment
            assert o["n_seeds"] >= 1


# --------------------------------------------------------------------------- #
# merge in cells
# --------------------------------------------------------------------------- #
def test_static_candidates_cell_rule():
    h = house(6)
    cells = cell_of(h.coords, 4)
    cand = static_candidates(cells, 4)
    G = S // 4
    cx = np.stack([cells // (G * G), (cells // G) % G, cells % G], 1)
    dlt = cx[cand[:, 1]] - cx[cand[:, 0]]
    ok = np.all(dlt == 0, 1) | np.all(dlt == np.eye(3, dtype=np.int64)[cand[:, 2]], 1)
    assert ok.all() and (cand[:, 0] != cand[:, 1]).all()
    con = real_contacts(h.coords)
    key = set(map(tuple, cand.tolist()))
    assert all((u, v, a) in key for u, v, a in con.tolist())     # every real contact is a candidate


def test_try_merge_box_and_overlap():
    st = MState(np.array([1, 1, 2]), np.array([0, 0, 1]), 4)     # cells 0,0 and (0,0,1)
    assert st.try_merge(0, 1, 0)                                  # 1 right of 0 along x, same cell
    assert (st.hi[st.root[0]] - st.lo[st.root[0]] == [2, 3, 3]).all()   # box shrank along x
    assert not st.try_merge(0, 1, 2)                              # same cluster
    # block 2 lives in cell (0,0,1) (z in 4..7); gluing its +y face to block 0 would put
    # it at block 0's z (0..3), so the box intersection is empty
    assert not st.try_merge(2, 0, 1)
    assert st.try_merge(0, 2, 2)                                  # +z of block 0, box -> z = 3
    assert st.n_clusters() == 1


def test_merge_state_incremental_equals_rebuild():
    """Guard against train/decode feature skew (T24b): replay a random decode, rebuild
    the state from its clusters and positions, compare the features the model sees."""
    h = house(7, n=60)
    A = assignment(h, 4, T)
    out = merge_cells_decode(None, [h], [A], torch.device("cpu"), 4, policy="random", steps=8,
                             snapshots=[[3]])
    st, _ = out[0]["snaps"][0]
    comp = np.unique(st.root, return_inverse=True)[1]
    re = state_from_partition(st.t, st.cell, st.off, comp, 4)
    fa, fb = state_features(st), state_features(re)
    for k in ("est", "ext", "lsize", "multi", "fstate", "lslot", "types", "cell"):
        assert np.allclose(fa[k], fb[k]), k
    assert np.array_equal(st.reps(), re.reps())


def test_merge_targets_teacher_forced():
    h = house(8, n=70)
    c, t = np.asarray(h.coords), np.asarray(h.types)
    cells = cell_of(c, 4)
    con = real_contacts(c)
    cand = static_candidates(cells, 4)
    # all singletons: positives are real contacts between slot representatives
    st = state_from_partition(t, cells, c, np.arange(len(t)), 4)
    ex = make_example(st, h, con, cand, house_coarse(h, 4, T))
    assert ex["pos"].any()
    rep = st.reps()
    for i, j, a in ex["cand"][ex["pos"]].tolist():
        assert rep[i] and rep[j]
        assert any(st.slot[u] == st.slot[i] and st.slot[v] == st.slot[j] and aa == a for u, v, aa in con.tolist())
    # one big real cluster minus one block: the missing block's contacts are the targets
    comp = np.zeros(len(t), np.int64)
    comp[5] = 1
    st = state_from_partition(t, cells, c, comp, 4)
    ex = make_example(st, h, con, cand, house_coarse(h, 4, T))
    want = {(int(u), int(v), int(a)) for u, v, a in con.tolist() if (u == 5) != (v == 5)}
    got = {tuple(x) for x in ex["cand"][ex["pos"]].tolist()}
    assert got == want


def test_merge_model_loss_and_decode():
    torch.manual_seed(0)
    hs = [house(9, n=60), house(10, n=50)]
    m = MergeNet(MergeConfig(n_types=T, f=4, d=16, rounds=1, heads=2, dk=8))
    rng = np.random.default_rng(0)
    from tools.misc.adjacency_gen.merge_cells import tf_example
    exs = [tf_example(h, 4, T, rng) for h in hs]
    r = merge_loss(m, exs, torch.device("cpu"))
    r["loss"].backward()
    assert torch.isfinite(r["loss"])
    A = [assignment(h, 4, T) for h in hs]
    out = merge_cells_decode(m.eval(), hs, A, torch.device("cpu"), 4, steps=8)
    for o, h, Ah in zip(out, hs, A):
        assert len(o["coords"]) + o["dropped"] == len(h)
        assert len({tuple(x) for x in o["coords"].tolist()}) == len(o["coords"])   # no overlaps
        got = np.zeros_like(Ah)
        np.add.at(got, (cell_of(o["coords"], 4), o["types"]), 1)
        assert (got <= Ah).all()                                  # every block stays in its cell


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_glue_counts_match_typed_glue():
    h = house(11, n=50)
    g = _grid(h.coords, h.types)
    gc = glue_counts(g, T)
    tg = typed_glue(h.coords, h.types)
    T2 = T * T
    for (tp, tm, a), n in tg.items():
        assert gc[a * T2 + tp * T + tm] == n
    assert gc.sum() == sum(tg.values())


def test_build_metrics_real_is_perfect():
    h = house(12)
    m = build_metrics(h.coords, h.types, h, house_coarse(h, 4, T), T, assignment(h, 4, T))
    for k in ("voxel_iou", "typed_f1", "glue_f1", "coarse_iou", "connectivity", "single", "complete",
              "shift_iou", "shift_typed_f1"):
        assert m[k] == pytest.approx(1.0), k
    sh = build_metrics(h.coords + np.array([1, 0, 2]), h.types, h, house_coarse(h, 4, T), T, assignment(h, 4, T))
    assert sh["voxel_iou"] < 1 and sh["shift_iou"] == pytest.approx(1.0) and sh["shift_typed_f1"] == pytest.approx(1.0)
    assert m["slot_l1"] == 0 and m["coarse_l1"] == 0
    rng = np.random.default_rng(0)
    c, t = upsample_assignment(assignment(h, 4, T), 4, rng)
    m2 = build_metrics(c, t, h, house_coarse(h, 4, T), T, assignment(h, 4, T))
    assert m2["coarse_iou"] == 1.0 and m2["slot_l1"] == 0 and m2["voxel_iou"] < 1
