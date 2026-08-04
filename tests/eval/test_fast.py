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
