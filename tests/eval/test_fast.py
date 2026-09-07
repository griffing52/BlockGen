"""FAST tier: the scorecard contract, and equivalence with the published code.

`novelty_cached` reimplements `blockgen.eval.novelty.evaluate_novelty` with the
reference voxelization hoisted out of the per-arm loop. `novelty.py` is left
byte-identical so results.md T1-T22 stay reproducible, which means the two
implementations can drift; `test_novelty_cached_matches_published` is what stops
that.
"""

from __future__ import annotations

import numpy as np
import pytest

from blockgen.eval.bench import fast, probes
from blockgen.eval.bench import scorecard as sc
from blockgen.eval.novelty import evaluate_novelty
from blockgen.utils.data import Structure


def _house(rng, nx=6, ny=5, nz=6) -> Structure:
    """A hollow box with a couple of materials -- enough to exercise every path."""
    ids = np.zeros((nx, ny, nz), dtype=np.int32)
    ids[:, 0, :] = 4                      # cobblestone floor
    ids[0, :, :] = ids[-1, :, :] = 5      # plank walls
    ids[:, :, 0] = ids[:, :, -1] = 5
    ids[:, -1, :] = 5                     # roof
    if rng.random() < 0.5:
        ids[1, 1, 0] = 0                  # a doorway, sometimes
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


@pytest.fixture(scope="module")
def ref():
    rng = np.random.default_rng(0)
    train = [_house(rng) for _ in range(24)]
    val = [_house(rng) for _ in range(16)]
    ctx = sc.BenchContext(grid=12, min_n=4, n_boot=60)
    return fast.FastReference(train, val, ctx)


def test_novelty_cached_matches_published(ref):
    """Bit-for-bit agreement with the implementation that produced T1-T22."""
    rng = np.random.default_rng(3)
    gen = [_house(rng) for _ in range(10)]
    mine = fast.novelty_cached(gen, ref, k=3, dup_threshold=0.95)
    theirs = evaluate_novelty(gen, ref.train, ref.vocab, grid=ref.ctx.grid, k=3,
                              dup_threshold=0.95)
    assert np.allclose(mine.nn_iou, theirs.nn_iou)
    assert np.array_equal(mine.nn_index, theirs.nn_index)
    assert np.allclose(mine.nn_iou_topk, theirs.nn_iou_topk)
    assert np.allclose(mine.block_agreement, theirs.block_agreement)
    assert mine.duplicate_rate == pytest.approx(theirs.duplicate_rate)
    assert mine.mean_nn_iou == pytest.approx(theirs.mean_nn_iou)
    assert mine.diversity == pytest.approx(theirs.diversity)


def test_memorization_detector_fires_on_verbatim_copies(ref):
    """A model reciting its training set must be caught, not rewarded."""
    arm = sc.ArmSpec("copy", track="control", structures=list(ref.train[:8]))
    blocks = fast.score_fast(arm, ref)
    assert blocks["novelty"]["voxel_dup_rate"].value == pytest.approx(1.0)
    assert blocks["novelty"]["voxel_nn_iou_mean"].value == pytest.approx(1.0)


def test_palette_metrics_are_blind_to_material_shuffle(ref):
    """Documents the FAST tier's blind spot rather than leaving it implicit.

    Shuffling permutes placement but not the multiset, so voxel palette
    statistics cannot see it. This probe only bites once rendering enters.
    """
    rng = np.random.default_rng(7)
    base = list(ref.val[:8])
    shuffled = [probes.shuffle_materials(s, rng) for s in base]
    a = fast.score_fast(sc.ArmSpec("base", structures=base), ref)
    b = fast.score_fast(sc.ArmSpec("shuf", structures=shuffled), ref)
    for key in ("palette_jsd_exact", "palette_jsd_family", "palette_cooccur_jsd"):
        assert a["dataset_stats"][key].value == pytest.approx(
            b["dataset_stats"][key].value), key


def test_monochrome_moves_palette_but_not_geometry(ref):
    base = list(ref.val[:8])
    mono = [probes.monochrome(s) for s in base]
    a = fast.score_fast(sc.ArmSpec("base", structures=base), ref)
    b = fast.score_fast(sc.ArmSpec("mono", structures=mono), ref)
    assert (b["dataset_stats"]["palette_jsd_family"].value
            > a["dataset_stats"]["palette_jsd_family"].value + 0.3)
    assert a["dataset_stats"]["n_blocks_w1"].value == pytest.approx(
        b["dataset_stats"]["n_blocks_w1"].value)


def test_every_metric_has_an_interval_and_a_direction(ref):
    """The schema invariant: no bare floats anywhere in a scorecard."""
    blocks = fast.score_fast(sc.ArmSpec("x", structures=list(ref.val[:8])), ref)
    for section in ("dataset_stats", "novelty"):
        for name, m in blocks[section].items():
            assert isinstance(m, sc.Metric), f"{section}.{name}"
            assert m.direction in sc.DIRECTIONS, name
            assert m.value is None or len(m.ci) == 2, name


def test_coherence_never_reports_a_bare_rate(ref):
    blocks = fast.score_fast(sc.ArmSpec("x", structures=list(ref.val[:8])), ref)
    for name, entry in blocks["coherence"].items():
        assert entry["direction"] == "distance_to_real", name
        assert "real" in entry and "gen" in entry, name


def test_empty_arm_is_skipped_not_crashed(ref):
    empty = Structure(block_ids=np.zeros((2, 2, 2), dtype=np.int32),
                      block_data=np.zeros((2, 2, 2), dtype=np.int32))
    card = sc.Scorecard(context={})
    blocks = fast.score_fast(sc.ArmSpec("empty", structures=[empty] * 4), ref, card)
    assert blocks["novelty"]["voxel_nn_iou_mean"].value is None
    assert blocks["novelty"]["voxel_nn_iou_mean"].gate_failed
    assert any("empty" in w for w in card.warnings)


def test_small_n_raises_a_warning(ref):
    card = sc.Scorecard(context={})
    fast.score_fast(sc.ArmSpec("tiny", structures=list(ref.val[:2])), ref, card)
    assert any("min_n" in w for w in card.warnings)


def test_scorecard_rejects_a_null_without_a_reason():
    with pytest.raises(ValueError):
        sc.Metric(value=None)


def test_scorecard_rejects_an_unknown_direction():
    with pytest.raises(ValueError):
        sc.Metric(value=1.0, direction="bigger_is_nicer")


def test_scorecard_json_is_serializable(ref):
    import json
    card = sc.Scorecard(context={"corpus": "toy"})
    card.add_arm("x", fast.score_fast(sc.ArmSpec("x", structures=list(ref.val[:6])), ref))
    text = json.dumps(card.to_json())
    assert "\"direction\"" in text
    assert "NaN" not in text          # non-finite floats must become null


def test_markdown_shows_real_beside_generated(ref):
    card = sc.Scorecard(context={"corpus": "toy", "split_key": "toy.v1"})
    card.add_arm("x", fast.score_fast(sc.ArmSpec("x", structures=list(ref.val[:6])), ref))
    md = sc.render_markdown(card)
    assert "not 'higher is better'" in md
    assert "real " in md


# --- the bench/2 write side ------------------------------------------------
# What `score_fast` puts in `meta` is the whole provenance feature as the lab
# actually receives it. The pieces are unit-tested elsewhere (`ArmSpec.provenance`
# in test_scorecard.py, `structures_sha` in test_features.py) and the *shape* is
# asserted against a static fixture in test_bench_report_compat.py -- but a
# fixture is a copy of what the writer once did, not a check that it still does
# it. Delete the `meta["provenance"] = arm.provenance()` line and every one of
# those stays green; these do not.
def test_meta_carries_arm_identity_for_an_in_memory_arm(ref):
    arm = sc.ArmSpec("real@toy", track="control", structures=list(ref.val[:6]),
                     provenance_override={"writer": "in_memory",
                                          "builder": "test", "recipe": "a toy rung"})
    meta = fast.score_fast(arm, ref)["meta"]
    assert meta["kind"] == "control" and meta["origin"] == "in_memory"
    assert meta["source"] == "in-memory"          # byte-identical, five consumers
    assert meta["provenance"]["recipe"] == "a toy rung"
    assert meta["source_run_id"] is None
    assert isinstance(meta["structures_sha"], str) and meta["structures_sha"]


def test_meta_carries_provenance_and_source_run_id_for_an_npz_arm(ref, tmp_path):
    from blockgen.curation.houses import save_house_cache
    save_house_cache(list(ref.val[:6]), max_dim=12, cache_dir=str(tmp_path),
                     name="toy", report={"model": "toy_model", "seed": 7,
                                         "temperature": 0.8, "top_k": 40,
                                         "checkpoint": "outputs/run_20260101_000000_toy/m.pt"})
    arm = sc.ArmSpec("toy_model", track="ar", npz=str(tmp_path / "toy_12.npz"))
    meta = fast.score_fast(arm, ref)["meta"]
    assert meta["kind"] == "submission" and meta["origin"] == "npz"
    # The severing point, end to end: sampling recorded these, the bench kept them.
    assert meta["provenance"]["model"] == "toy_model"
    assert meta["provenance"]["seed"] == 7 and meta["provenance"]["top_k"] == 40
    # The edge back to the training run, derived from the checkpoint path.
    assert meta["source_run_id"] == "run_20260101_000000_toy"


def test_structures_sha_is_stable_across_arms_with_the_same_builds(ref):
    builds = list(ref.val[:6])
    a = fast.score_fast(sc.ArmSpec("a", structures=list(builds)), ref)["meta"]
    b = fast.score_fast(sc.ArmSpec("b", structures=list(builds)), ref)["meta"]
    c = fast.score_fast(sc.ArmSpec("c", structures=list(ref.val[2:8])), ref)["meta"]
    assert a["structures_sha"] == b["structures_sha"]   # identity is the builds,
    assert a["structures_sha"] != c["structures_sha"]   # not the arm's name


def test_examples_sink_is_total_and_indexes_the_scored_builds(ref):
    """One tuple per scored arm, always -- the runner indexes `sink[0]` blind."""
    sink: list = []
    arm = sc.ArmSpec("x", structures=list(ref.val[:6]))
    fast.score_fast(arm, ref, examples=sink, examples_k=3)
    assert len(sink) == 1
    # `(the whole non-empty list, the chosen rows into it)` -- the runner does the
    # truncating, so that peak retention is k builds per arm and not every arm's
    # full list held alive at once.
    scored, rows = sink[0]
    assert len(scored) == 6 and len(rows) == 3
    assert rows == sorted(set(rows)) and all(0 <= r < 6 for r in rows)
    assert [scored[r] for r in rows]                  # the rows really index it

    empty_sink: list = []
    dead = Structure(block_ids=np.zeros((4, 4, 4), dtype=np.int32),
                     block_data=np.zeros((4, 4, 4), dtype=np.int32))
    fast.score_fast(sc.ArmSpec("empty", structures=[dead] * 4), ref,
                    sc.Scorecard(context={}), examples=empty_sink)
    assert len(empty_sink) == 1 and empty_sink[0] == ([], [])
