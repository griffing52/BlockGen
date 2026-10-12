"""Face formulation + coarse conditioning: masks, targets, coarse grid, decoders, baseline.

CPU only, tiny random models, hand-built chunks; nothing is read from disk.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.misc.adjacency_gen.coarse import Coarse, coarse_cell, coarsen, upsample_baseline
from tools.misc.adjacency_gen.decode_faces import (align_to_condition, match_axis, oneshot_realize,
                                                   oneshot_scores, sequential_decode)
from tools.misc.adjacency_gen.faces import (DIRS, OPP, action_scores, attach_targets, buried_state,
                                            correct_actions, exposed_faces, face_glue_full,
                                            glue_matrices, multiset_f1, random_attachment_order,
                                            typed_glue)
from tools.misc.adjacency_gen.model_faces import FaceConfig, FaceModel
from tools.misc.adjacency_gen.train_faces import losses, sample_states


def chunk(seed=0, n=14, size=8):
    """A random connected blob in an 8^3 window, with 3 block types."""
    rng = np.random.default_rng(seed)
    cells = {(3, 0, 3)}
    while len(cells) < n:
        c = list(cells)[rng.integers(len(cells))]
        d = DIRS[rng.integers(6)]
        nc = tuple(int(v) for v in np.clip(np.array(c) + d, 0, size - 1))
        cells.add(nc)
    coords = np.array(sorted(cells), np.int64)
    types = rng.integers(1, 4, len(coords))
    return coords, types


def item(coords, types, f=2, size=8, n_types=5):
    from tools.misc.adjacency_gen.coarse import coarsen as cz
    c = cz(coords, types, f, size=size, empty=n_types)
    ct = -np.ones(size ** 3, np.int64)
    ct[coords[:, 0] * size * size + coords[:, 1] * size + coords[:, 2]] = types
    cnt, dom = c.flat()
    return {"types": types, "coords": coords, "glue": glue_matrices(coords), "exposed": exposed_faces(coords),
            "cell_type": ct, "cond_count": cnt, "cond_dom": dom, "coarse": c,
            "ocell": coarse_cell(coords, 2, size // 2)}


def tiny(f=2, oracle=False):
    torch.manual_seed(0)
    return FaceModel(FaceConfig(n_types=5, f=f, oracle=oracle, d=32, layers=1, heads=2, dk=16))


# --------------------------------------------------------------------------- #
def test_face_matrix_symmetric_complementary_one_partner():
    coords, _ = chunk(1, n=20)
    G = glue_matrices(coords)
    F = face_glue_full(G)
    N = len(coords)
    assert np.array_equal(F, F.T)                                   # symmetric
    for u, v in zip(*np.nonzero(F)):
        bu, du = divmod(int(u), 6)
        bv, dv = divmod(int(v), 6)
        assert bu != bv                                             # never within one block
        assert dv == OPP[du]                                        # +x only with -x, ...
        assert np.array_equal(coords[bv] - coords[bu], DIRS[du])
    assert F.sum(1).max() <= 1                                      # at most one partner per face
    for a in range(3):
        assert G[a].sum(0).max() <= 1 and G[a].sum(1).max() <= 1
        assert not np.diag(G[a]).any()
    # exposed = no partner
    ex = exposed_faces(coords)
    assert np.array_equal(~ex.reshape(-1), F.any(1))
    # every touching pair appears exactly once over the three axes
    d = np.abs(coords[:, None] - coords[None]).sum(-1)
    assert int(G.sum()) == int((d == 1).sum() // 2)


def test_action_scores_mapping():
    X = np.random.default_rng(0).normal(size=(3, 5, 5))
    A = action_scores(X)
    for a in range(3):
        assert np.allclose(A[2 * a], X[a])
        assert np.allclose(A[2 * a + 1], X[a].T)                   # p's -a face = X[a][j, p]
    At = action_scores(torch.as_tensor(X)[None])
    assert np.allclose(At[0].numpy(), A)


def test_attachment_order_and_targets():
    coords, types = chunk(2, n=18)
    rng = np.random.default_rng(0)
    order = random_attachment_order(coords, rng)
    assert sorted(order.tolist()) == list(range(len(coords)))
    for k in range(1, len(order)):                                  # each block touches an earlier one
        assert (np.abs(coords[order[:k]] - coords[order[k]]).sum(1) == 1).any()
    for k in range(1, len(order)):
        placed = np.zeros(len(coords), bool)
        placed[order[:k]] = True
        tgt = attach_targets(coords, types, placed)
        C = correct_actions(tgt, types, placed)
        assert C.any()                                              # a correct move always exists
        assert C[:, ~placed].sum() == 0 and C[:, :, placed].sum() == 0
        d, p, j = (int(v[0]) for v in np.nonzero(C))
        cell = coords[p] + DIRS[d]
        real = [i for i in range(len(coords)) if np.array_equal(coords[i], cell)]
        assert len(real) == 1 and not placed[real[0]] and types[real[0]] == types[j]


def test_coarse_downsampling():
    coords, types = chunk(3, n=30)
    c = coarsen(coords, types, 2, size=8, empty=9)
    assert c.G == 4 and c.count.sum() == len(coords)
    for cell in np.argwhere(c.count > 0):
        inside = np.all(coords // 2 == cell, 1)
        assert c.count[tuple(cell)] == inside.sum()
        vals, cnt = np.unique(types[inside], return_counts=True)
        assert c.dom[tuple(cell)] == vals[cnt == cnt.max()].min()
    assert (c.dom[c.count == 0] == 9).all()
    idx = coarse_cell(coords, 2, 4)
    assert np.array_equal(np.bincount(idx, minlength=64), c.count.ravel())
    c4 = coarsen(coords, types, 4, size=8)
    assert c4.G == 2 and c4.count.sum() == len(coords)


def test_upsample_baseline_matches_condition():
    coords, types = chunk(4, n=40)
    c = coarsen(coords, types, 2, size=8, empty=9)
    u = upsample_baseline(types, c, seed=0)
    assert u["placed"].all()
    assert len({tuple(v) for v in u["pos"].tolist()}) == len(types)    # no collisions
    c2 = coarsen(u["pos"], types, 2, size=8, empty=9)
    assert np.array_equal(c2.count, c.count)                          # coarse-consistent by construction


def test_losses_finite_and_backward():
    m = tiny()
    items = [item(*chunk(s, n=12 + s)) for s in range(3)]
    pl, tg = sample_states(items, np.random.default_rng(0), p0=0.34)
    cfg = m.cfg
    r = losses(m, items, pl, tg, torch.device("cpu"), cfg)
    assert torch.isfinite(r["loss"])
    r["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    # unconditioned and oracle variants
    for mm in (tiny(f=0), tiny(f=2, oracle=True)):
        r = losses(mm, items, pl, tg, torch.device("cpu"), mm.cfg)
        assert torch.isfinite(r["loss"])


def test_sequential_decode_tiny_model():
    m = tiny().eval()
    items = [item(*chunk(s, n=10)) for s in range(2)]
    outs = sequential_decode(m, items, torch.device("cpu"), check=True, temperature=0.0)
    for o, it in zip(outs, items):
        assert o["placed"].all() and not o["conflict"].any()
        P = o["pos"]
        assert len({tuple(v) for v in P.tolist()}) == len(P)            # no collisions
        assert ((P >= 0) & (P < 8)).all()                               # inside the window
        d = np.abs(P[:, None] - P[None]).sum(-1)
        from tools.misc.adjacency_gen.data import components
        assert len(components(d == 1)) == 1                             # connected
    outs = sequential_decode(m, items, torch.device("cpu"), check=False, temperature=1.0, seed=3)
    for o in outs:
        assert o["placed"].all()                                        # every block gets attached


def test_oneshot_matching_one_partner_and_realize():
    rng = np.random.default_rng(0)
    N = 9
    S = rng.normal(size=(N, N))
    m = match_axis(S, rng.normal(size=N), rng.normal(size=N))
    rows = [i for i, _, _ in m]
    cols = [j for _, j, _ in m]
    assert len(set(rows)) == len(rows) and len(set(cols)) == len(cols)
    assert all(i != j for i, j, _ in m)
    model = tiny().eval()
    items = [item(*chunk(5, n=11))]
    sc = oneshot_scores(model, items, torch.device("cpu"))
    for check in (True, False):
        r = oneshot_realize(sc[0]["X"], sc[0]["ex"], check=check)
        keep = r["keep"]
        if check:
            assert len({tuple(v) for v in r["pos"][keep].tolist()}) == keep.sum()
        p = align_to_condition(r["pos"], keep, items[0]["coarse"])
        assert p.shape == r["pos"].shape


def test_typed_glue_f1():
    coords, types = chunk(6, n=15)
    g = typed_glue(coords, types)
    assert multiset_f1(g, g)[2] == pytest.approx(1.0)
    assert sum(g.values()) == int(glue_matrices(coords).sum())


def test_buried_state():
    coords = np.array([[0, 0, 0], [1, 0, 0], [5, 5, 5]])
    st = buried_state(coords, np.array([True, True, False]))
    assert st[0, 0] == 2 and st[1, 1] == 2 and st[0, 1] == 1 and (st[2] == 0).all()


# --------------------------------------------------------------------------- #
# merge formulation
# --------------------------------------------------------------------------- #
def tiny_merge(f=2):
    torch.manual_seed(0)
    return FaceModel(FaceConfig(n_types=5, f=f, merge=True, d=32, layers=1, heads=2, dk=16))


def test_merge_state_and_targets():
    from tools.misc.adjacency_gen.merge import merge_targets, random_merge_state
    coords, types = chunk(7, n=16)
    G = glue_matrices(coords)
    rng = np.random.default_rng(0)
    for k in (0, 5, 14):
        comp = random_merge_state(coords, G, rng, k=k)
        assert len(np.unique(comp)) == len(coords) - k
        for c in np.unique(comp):                       # every component is connected in the real build
            m = np.nonzero(comp == c)[0]
            d = np.abs(coords[m][:, None] - coords[m][None]).sum(-1)
            from tools.misc.adjacency_gen.data import components
            assert len(components(d == 1)) == 1
        T = merge_targets(G, types, comp)
        assert T.any()
        for a, i, j in np.argwhere(T):
            assert comp[i] != comp[j] and i != j
        # every real crossing edge is a target
        for a, i, j in np.argwhere(G):
            if comp[i] != comp[j]:
                assert T[a, i, j]


def test_merge_decode_tiny_model():
    from tools.misc.adjacency_gen.merge import merge_decode
    m = tiny_merge().eval()
    items = [item(*chunk(s, n=10)) for s in range(2)]
    for check in (True, False):
        outs = merge_decode(m, items, torch.device("cpu"), check=check, temperature=1.0, seed=1)
        for o, it in zip(outs, items):
            n = len(it["types"])
            assert o["merges"] == n - 1 and len(np.unique(o["comp"])) == 1   # ends in one component
            if check:
                assert len({tuple(v) for v in o["pos"].tolist()}) == n        # no overlaps
    outs = merge_decode(None, items, torch.device("cpu"), policy="random")
    assert all(o["merges"] == len(it["types"]) - 1 for o, it in zip(outs, items))


def test_merge_losses_finite():
    from tools.misc.adjacency_gen.merge import merge_losses
    m = tiny_merge()
    items = [item(*chunk(s, n=12 + s)) for s in range(3)]
    r = merge_losses(m, items, np.random.default_rng(0), torch.device("cpu"), m.cfg)
    assert torch.isfinite(r["loss"])
    r["loss"].backward()


def _scratch_state(n, edges):
    """Components and positions rebuilt from scratch (BFS over the glued pairs)."""
    from collections import defaultdict
    nb = defaultdict(list)
    for a, i, j in edges:
        nb[i].append((j, DIRS[2 * a]))
        nb[j].append((i, -DIRS[2 * a]))
    comp = -np.ones(n, np.int64)
    pos = np.zeros((n, 3), np.int64)
    for r in range(n):
        if comp[r] >= 0:
            continue
        comp[r] = r
        q = [r]
        while q:
            u = q.pop()
            for v, d in nb[u]:
                if comp[v] < 0:
                    comp[v] = r
                    pos[v] = pos[u] + d
                    q.append(v)
    return comp, pos


def test_merge_features_incremental_equal_scratch():
    """T24b-style guard: the state features the decoder feeds after k merges equal the
    features computed from scratch from the same set of connections."""
    from tools.misc.adjacency_gen.merge import _State, _apply_merge, comp_fstate, comp_positions, merge_decode
    m = tiny_merge().eval()
    it = item(*chunk(9, n=14))
    n = len(it["types"])
    for temp, check in ((1.0, True), (0.0, True), (1.0, False)):
        o = merge_decode(m, [it], torch.device("cpu"), check=check, temperature=temp, seed=4)[0]
        st = _State(n)
        for k, (a, i, j) in enumerate(o["history"]):
            assert _apply_merge(st, i, j, a, check=False)
            comp_s, pos_s = _scratch_state(n, o["history"][:k + 1])
            # same partition
            assert np.array_equal(comp_s[:, None] == comp_s[None], st.comp[:, None] == st.comp[None])
            # same anchor-relative offsets and face states
            assert np.array_equal(comp_positions(pos_s, comp_s), comp_positions(st.pos, st.comp))
            assert np.array_equal(comp_fstate(pos_s, comp_s), comp_fstate(st.pos, st.comp))
        if check:
            assert np.array_equal(comp_positions(st.pos, st.comp), comp_positions(o["pos"], o["comp"]))


def test_fits_condition():
    from tools.misc.adjacency_gen.merge import fits_condition
    coords, types = chunk(10, n=20)
    c = coarsen(coords, types, 2, size=8, empty=9)
    assert fits_condition(coords, c, 8)                       # the real chunk fits its own condition
    cube = np.argwhere(np.ones((8, 8, 8), bool))
    assert not fits_condition(cube, c, 8)                     # 512 blocks exceed the counts
    assert not fits_condition(np.array([[0, 0, 0], [9, 0, 0]]), c, 8)   # wider than the window
