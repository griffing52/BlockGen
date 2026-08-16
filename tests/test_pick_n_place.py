"""Pick-and-place: representation fidelity, mask correctness, no future leakage.

Three classes of test, in order of how badly a failure would mislead:

1. **Round-trip.** If a growth sequence does not replay to the structure it came
   from, every number downstream is meaningless. This is T21's Phase 0 gate and
   it is checked on synthetic shapes and on real builds.
2. **Legality.** The placer's distribution is defined by a mask. If the mask
   admits a cell that is already full, the model can "learn" to overwrite; if it
   forbids a cell the ground truth uses, the loss is unlearnable. Both are silent.
3. **Causality.** Node features must not encode anything placed later. A leak
   here inflates training accuracy and evaporates at sampling time -- the exact
   shape of failure that is hardest to notice.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from blockgen.models import pick_n_place as pnp
from blockgen.training.train_pick_n_place import (GrowthDataset, PieceCodec,
                                                  TrainConfig, collate, evaluate, train)
from blockgen.utils.data import Structure
from blockgen.utils import growth_order as go


# --- fixtures ---------------------------------------------------------------
def _solid(nx=3, ny=3, nz=3, block=5) -> Structure:
    ids = np.full((nx, ny, nz), block, dtype=np.int32)
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


def _house(seed=0) -> Structure:
    rng = np.random.default_rng(seed)
    ids = np.zeros((6, 5, 6), dtype=np.int32)
    ids[:, 0, :] = 4
    ids[0, :, :] = ids[-1, :, :] = 5
    ids[:, :, 0] = ids[:, :, -1] = 98
    ids[:, -1, :] = 5
    ids[2, 2, 2] = int(rng.integers(1, 40))
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


def _two_components() -> Structure:
    ids = np.zeros((9, 3, 3), dtype=np.int32)
    ids[0:3] = 5           # big
    ids[7:9] = 4           # small, disconnected
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


# --- 1. round-trip ----------------------------------------------------------
@pytest.mark.parametrize("ordering", list(go.ORDERINGS))
def test_roundtrip_is_exact(ordering):
    for s in (_solid(), _solid(4, 2, 5), _house(1), _house(2)):
        assert go.roundtrip_iou(s, ordering=ordering) == pytest.approx(1.0)


def test_replay_ignores_recorded_coordinates():
    """Replay must rebuild geometry from parent/direction alone."""
    seq = go.structure_to_growth(_house(3))
    scrambled = go.GrowthSequence(
        coords=np.zeros_like(seq.coords), pieces=seq.pieces, parent=seq.parent,
        direction=seq.direction, neighbor_node=seq.neighbor_node)
    a = go.growth_to_structure(seq).occupied_mask
    b = go.growth_to_structure(scrambled).occupied_mask
    assert np.array_equal(a, b)


def test_growth_is_connected_by_construction():
    seq = go.structure_to_growth(_house(4))
    assert seq.parent[0] == -1 and seq.direction[0] == -1
    assert np.all(seq.parent[1:] < np.arange(1, seq.n_nodes)), "attaches to a later node"
    assert np.all(seq.parent[1:] >= 0)


def test_only_the_largest_component_is_kept():
    seq = go.structure_to_growth(_two_components())
    assert seq.n_nodes == 27, "should keep the 3x3x3 slab, drop the 2x3x3"


def test_empty_structure_returns_none():
    z = np.zeros((3, 3, 3), dtype=np.int32)
    assert go.structure_to_growth(Structure(block_ids=z, block_data=z.copy())) is None


def test_max_nodes_truncates_to_a_valid_prefix():
    seq = go.structure_to_growth(_solid(5, 5, 5), max_nodes=20)
    assert seq.n_nodes == 20
    assert go.growth_to_structure(seq).occupied_mask.sum() == 20


@pytest.mark.slow
def test_roundtrip_on_real_builds():
    from blockgen.eval.bench import splits
    st = splits.split_structures(splits.load_split("houses_32"), "val")[:60]
    stats = go.growth_stats(st)
    assert stats["min_roundtrip_iou"] == pytest.approx(1.0)
    assert stats["block_retention"] > 0.9


# --- 2. legality ------------------------------------------------------------
def test_seed_row_has_no_legal_face():
    m = go.structure_to_growth(_house(5)).placement_masks()
    assert not m[0].any()


def test_every_ground_truth_placement_is_legal():
    seq = go.structure_to_growth(_house(6))
    m, tp = seq.placement_masks(), seq.target_ports()
    for t in range(1, seq.n_nodes):
        assert m[t].reshape(-1)[tp[t]], f"true placement illegal at step {t}"


def test_mask_forbids_occupied_cells_and_unplaced_parents():
    seq = go.structure_to_growth(_solid(3, 3, 3))
    m = seq.placement_masks()
    for t in range(1, seq.n_nodes):
        assert not m[t][t:].any(), "a not-yet-placed node cannot be a parent"
        for i in range(t):
            for d in range(go.N_DIR):
                j = seq.neighbor_node[i, d]
                if 0 <= j < t:
                    assert not m[t][i, d], "cell already occupied"


def test_live_legality_matches_precomputed():
    """Sampling-time legality must agree with the training-time mask."""
    seq = go.structure_to_growth(_house(7))
    m = seq.placement_masks()
    for t in (1, 5, 12):
        live = pnp.live_legality(seq.coords[:t])
        assert np.array_equal(live, m[t][:t]), f"disagreement at t={t}"


def test_live_legality_respects_max_extent():
    coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    assert pnp.live_legality(coords, max_extent=2)[1, 0] == False  # noqa: E712
    assert pnp.live_legality(coords, max_extent=8)[1, 0] == True   # noqa: E712


# --- 3. causality / leakage -------------------------------------------------
def _tiny_model(n_pieces=8, n=12, **kw):
    cfg = pnp.PickAndPlaceConfig(n_pieces=n_pieces, d_model=32, nhead=4,
                                 num_layers=2, max_nodes=n, **kw)
    return pnp.PickAndPlace(cfg).eval()


def _batch(n=12, n_pieces=8, seed=0):
    rng = np.random.default_rng(seed)
    pieces = torch.from_numpy(
        rng.integers(pnp.PIECE_OFFSET, pnp.PIECE_OFFSET + n_pieces, (1, n)))
    direction = torch.from_numpy(rng.integers(0, pnp.N_DIR, (1, n)))
    coords = torch.from_numpy(rng.integers(0, 5, (1, n, 3)))
    legal = torch.ones(1, n, n, pnp.N_DIR, dtype=torch.bool)
    return pieces, direction, pieces.clone(), coords, legal


def test_encoder_is_causal():
    """Changing node t must not move any state at or before t."""
    model = _tiny_model()
    pieces, direction, parent, coords, _ = _batch()
    a = model.encoder(pieces, direction, parent, coords)
    pieces2 = pieces.clone()
    pieces2[0, 7] = (pieces2[0, 7] + 3)
    coords2 = coords.clone()
    coords2[0, 7] += 2
    b = model.encoder(pieces2, direction, parent, coords2)
    # state[t] sees nodes 0..t-1, so states 0..7 must be untouched.
    assert torch.allclose(a[:, :8], b[:, :8], atol=1e-5)
    assert not torch.allclose(a[:, 8:], b[:, 8:], atol=1e-5)


def test_prefix_features_match_full_sequence():
    """Train/inference skew guard, and the reason it exists.

    Generation feeds the model a growing prefix; training feeds a padded batch.
    Any node feature computed from the *current* length therefore means something
    different in the two regimes. The step index was originally normalized by
    `n`, so a node three steps into generation read as "end of build" while the
    same node in training read as 3/192 -- the model learned STOP-at-1.0 and
    collapsed to a median of 30 blocks from sequences that were all 192 long.
    Teacher-forced metrics looked excellent throughout.
    """
    model = _tiny_model(n=32)
    pieces, direction, parent, coords, _ = _batch(n=12)
    full = model.encoder.node_features(pieces, direction, parent, coords)
    for k in (1, 5, 11):
        part = model.encoder.node_features(pieces[:, :k], direction[:, :k],
                                           parent[:, :k], coords[:, :k])
        assert torch.allclose(full[:, :k], part, atol=1e-6), f"skew at prefix {k}"


def test_prefix_states_match_full_sequence():
    """The same invariant end-to-end: causality plus no length-dependent features
    means a prefix must encode identically inside a longer sequence."""
    model = _tiny_model(n=32)
    pieces, direction, parent, coords, _ = _batch(n=12)
    full = model.encoder(pieces, direction, parent, coords)
    part = model.encoder(pieces[:, :6], direction[:, :6], parent[:, :6], coords[:, :6])
    assert torch.allclose(full[:, :7], part, atol=1e-5)


def test_state_zero_is_the_empty_graph():
    model = _tiny_model()
    a = model.encoder(*_batch(seed=1)[:4])
    b = model.encoder(*_batch(seed=2)[:4])
    assert torch.allclose(a[:, 0], b[:, 0]), "state 0 must not depend on the build"


def test_forward_shapes_and_illegal_faces_are_minus_inf():
    n, n_pieces = 12, 8
    model = _tiny_model(n_pieces, n)
    pieces, direction, parent, coords, legal = _batch(n, n_pieces)
    legal[0, :, 5:, :] = False
    pick, place = model(pieces, direction, parent, coords, legal)
    assert pick.shape == (1, n + 1, n_pieces + pnp.PIECE_OFFSET)
    assert place.shape == (1, n, n * pnp.N_DIR)
    flat = legal.reshape(1, n, -1)
    assert torch.isinf(place[~flat]).all() and (place[~flat] < 0).all()
    assert torch.isfinite(place[flat]).all()


def test_picker_never_proposes_pad():
    model = _tiny_model()
    pick, _ = model(*_batch())
    assert torch.isinf(pick[..., pnp.PAD]).all()


def test_relative_bias_is_translation_invariant():
    """Shifting the whole build must not change the geometry bias."""
    model = _tiny_model()
    _, _, _, coords, _ = _batch()
    a = model.encoder.rel_bias(coords)
    b = model.encoder.rel_bias(coords + 17)
    assert torch.allclose(a, b)


# --- end to end -------------------------------------------------------------
def test_train_and_generate_end_to_end():
    structures = [_house(i) for i in range(12)]
    seqs = [go.structure_to_growth(s) for s in structures]
    codec = PieceCodec.from_sequences([s for s in seqs if s])
    ds = GrowthDataset(structures, codec, max_nodes=64, min_nodes=4)
    assert len(ds) > 0

    cfg = pnp.PickAndPlaceConfig(n_pieces=codec.n_pieces, d_model=32, nhead=4,
                                 num_layers=2, max_nodes=64)
    model = pnp.PickAndPlace(cfg)
    model, hist = train(model, ds, ds, TrainConfig(epochs=2, batch_size=2,
                                                   device="cpu", log_every=99))
    assert len(hist["pick_loss"]) == 2
    assert all(np.isfinite(v) for v in hist["pick_loss"] + hist["place_loss"])

    r = pnp.generate(model, max_nodes=24, device="cpu", top_k=4)
    assert len(r.pieces) <= 24
    if len(r.coords) > 1:
        # Every generated node must attach to an earlier one, and no two nodes
        # may share a cell -- the growth invariant, now under a learned placer.
        assert np.all(r.parent[1:] < np.arange(1, len(r.parent)))
        assert len({tuple(c) for c in r.coords}) == len(r.coords)


def test_generated_structure_is_connected():
    from blockgen.eval.bench.topology import label6
    structures = [_house(i) for i in range(8)]
    codec = PieceCodec.from_sequences(
        [s for s in (go.structure_to_growth(x) for x in structures) if s])
    cfg = pnp.PickAndPlaceConfig(n_pieces=codec.n_pieces, d_model=32, nhead=4,
                                 num_layers=2, max_nodes=32)
    model = pnp.PickAndPlace(cfg)
    r = pnp.generate(model, max_nodes=20, device="cpu", top_k=4)
    s = pnp.rollout_to_structure(r, codec.decode)
    if s.occupied_mask.sum() > 1:
        assert label6(s.occupied_mask)[1] == 1, "growth cannot produce a gap"


def test_evaluate_reports_the_chance_baseline():
    """place_acc is meaningless without it -- the mask already does work."""
    from torch.utils.data import DataLoader
    structures = [_house(i) for i in range(6)]
    codec = PieceCodec.from_sequences(
        [s for s in (go.structure_to_growth(x) for x in structures) if s])
    ds = GrowthDataset(structures, codec, max_nodes=48, min_nodes=4)
    cfg = pnp.PickAndPlaceConfig(n_pieces=codec.n_pieces, d_model=32, nhead=4,
                                 num_layers=2, max_nodes=48)
    m = pnp.PickAndPlace(cfg)
    out = evaluate(m, DataLoader(ds, batch_size=2, collate_fn=collate),
                   TrainConfig(device="cpu"))
    assert 0.0 < out["place_chance"] < 1.0
    assert "place_lift" in out


def test_codec_roundtrips_and_reserves_ids():
    codec = PieceCodec.from_sequences(
        [go.structure_to_growth(_house(1)), go.structure_to_growth(_house(2))])
    for t in codec.tokens:
        assert codec.decode(codec.encode(t)) == t
        assert codec.encode(t) >= pnp.PIECE_OFFSET      # never STOP or PAD
    assert PieceCodec.from_json(codec.to_json()).tokens == codec.tokens


def test_truncated_builds_get_no_stop_target():
    """STOP must mean "finished", not "hit max_nodes".

    With every training build truncated to the cap, STOP always fired at the same
    position and carried no completeness information -- the model emitted a fixed
    ~350-node budget regardless of what it was shown (prefix-test length_corr
    -0.05). A cut-off sequence is not an ending, so it gets no STOP target.
    """
    structures = [_solid(6, 6, 6)]          # 216 nodes, well over the cap below
    codec = PieceCodec.from_sequences([go.structure_to_growth(structures[0])])
    ds = GrowthDataset(structures, codec, max_nodes=32, min_nodes=4)
    assert ds.stats()["complete_frac"] == 0.0
    batch = collate([ds[0]])
    assert (batch["pick_target"][0] == pnp.STOP).sum() == 0, "truncated: no STOP"


def test_complete_builds_do_get_a_stop_target():
    structures = [_house(0)]
    codec = PieceCodec.from_sequences([go.structure_to_growth(structures[0])])
    ds = GrowthDataset(structures, codec, max_nodes=4096, min_nodes=4)
    assert ds.stats()["complete_frac"] == 1.0
    batch = collate([ds[0]])
    assert (batch["pick_target"][0] == pnp.STOP).sum() == 1, "complete: one STOP"
