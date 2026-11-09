"""Within-set near-twins and the self-duplication gate built on them."""

from __future__ import annotations

import numpy as np
import pytest

from blockgen.eval.bench import composite as comp
from blockgen.eval.bench import scorecard as sc
from blockgen.eval.bench import twins as tw


def _house(rng, shape=(7, 6, 9)):
    ids = np.zeros(shape, np.int32)
    ids[0, :, :] = ids[-1, :, :] = ids[:, :, 0] = ids[:, :, -1] = 4
    ids[:, -1, :] = 5
    ids[:, 0, :] = 1
    ids[rng.integers(1, shape[0] - 1), rng.integers(1, shape[1] - 1), 0] = 0
    return ids


def test_a_rotated_mirrored_shifted_copy_is_a_twin():
    rng = np.random.default_rng(0)
    a = _house(rng)
    b = np.zeros((12, 9, 14), np.int32)
    v = np.rot90(a[::-1], 1, axes=(0, 2))
    b[2:2 + v.shape[0], 1:1 + v.shape[1], 3:3 + v.shape[2]] = v
    rep = tw.within_set_twins([a, b])
    assert rep.nn_iou == pytest.approx([1.0, 1.0])
    assert rep.rate == 1.0


def test_different_builds_are_not_twins():
    a = np.ones((4, 4, 4), np.int32)
    b = np.ones((8, 2, 8), np.int32)          # same family, different shape
    c = np.ones((3, 10, 3), np.int32)
    rep = tw.within_set_twins([a, b, c])
    assert rep.rate == 0.0
    assert (rep.nn_iou < tw.TWIN_THRESHOLD).all()


def test_iou_matches_brute_force_on_a_small_pair():
    rng = np.random.default_rng(3)
    a = (rng.random((4, 3, 5)) < 0.6).astype(np.int32)
    b = (rng.random((5, 3, 4)) < 0.6).astype(np.int32)
    a, b = tw._occ(a), tw._occ(b)
    best = 0.0
    for v in tw.d4(b):
        for dx in range(-v.shape[0], a.shape[0] + 1):
            for dy in range(-v.shape[1], a.shape[1] + 1):
                for dz in range(-v.shape[2], a.shape[2] + 1):
                    g = np.zeros((20, 20, 20), bool)
                    h = np.zeros((20, 20, 20), bool)
                    o = 8
                    g[o:o + a.shape[0], o:o + a.shape[1], o:o + a.shape[2]] = a
                    h[o + dx:o + dx + v.shape[0], o + dy:o + dy + v.shape[1],
                      o + dz:o + dz + v.shape[2]] = v
                    best = max(best, (g & h).sum() / (g | h).sum())
    rep = tw.within_set_twins([a.astype(np.int32), b.astype(np.int32)], threshold=0.0)
    assert rep.nn_iou[0] == pytest.approx(best)


def test_a_repeated_build_is_all_twins():
    rng = np.random.default_rng(1)
    a = _house(rng)
    assert tw.within_set_twins([a] * 5).rate == 1.0


def test_gate_disqualifies_self_duplication_and_spares_real():
    def arm(rate, ci=0.005):
        return {"meta": {"n": 256},
                "realism": {"geom_kid": sc.metric(0.0, (-0.01, 0.01))},
                "novelty": {"voxel_twin_rate": sc.metric(rate, (rate - ci, rate + ci))}}
    cal = comp.Calibration.from_arm(arm(0.008), "real_test")
    assert comp.score_arm("real", arm(0.008), cal).status == "ok"
    # 3 x max(spread, 0.02) above 0.008 is the bar: 0.06 passes, 0.6 does not
    assert comp.score_arm("ok", arm(0.06), cal).status == "ok"
    row = comp.score_arm("copies", arm(0.6), cal)
    assert row.status == "DQ" and "voxel_twin_rate" in row.disqualified


def test_cards_without_the_metric_are_not_gated():
    arm = {"meta": {"n": 256}, "realism": {"geom_kid": sc.metric(0.0, (-0.01, 0.01))},
           "novelty": {}}
    cal = comp.Calibration.from_arm(arm, "real_test")
    assert comp.score_arm("old", arm, cal).status == "ok"


def test_probe_arms_are_exempt_but_gate_positive_controls_are_not():
    def arm(rate, ci=0.005):
        return {"meta": {"n": 256},
                "realism": {"geom_kid": sc.metric(0.0, (-0.01, 0.01))},
                "novelty": {"voxel_twin_rate": sc.metric(rate, (rate - ci, rate + ci))}}
    cal = comp.Calibration.from_arm(arm(0.008), "real_test")
    assert comp.is_probe_arm("real@solidify") and comp.is_probe_arm("real_shuffled_materials")
    assert not comp.is_probe_arm("real@single_mode") and not comp.is_probe_arm("dreamcubed")
    # a probe that rewrites real builds is a diagnostic, not a self-copying generator
    assert comp.score_arm("real@solidify", arm(0.6), cal).status == "ok"
    assert comp.score_arm("real@single_mode", arm(0.6), cal).status == "DQ"
    assert comp.score_arm("dreamcubed", arm(0.6), cal).status == "DQ"


# --- validation on real data ------------------------------------------------
@pytest.mark.slow
def test_gate_spares_fresh_real_draws_and_catches_template_copies():
    """The gate's null on houses_32: every held-out real draw passes the bar the
    val split calibrates, and a set built from a handful of real builds, each
    shifted and rotated, is disqualified. This is the real-data check the other
    gates have; the bar is the same 3-spread rule, not tuned on Dream-Cubed."""
    from blockgen.eval.bench import splits
    split = splits.load_split("houses_32", 0)
    val = [s.crop_to_non_air().block_ids for s in splits.split_structures(split, "val")]
    test = [s.crop_to_non_air().block_ids for s in splits.split_structures(split, "test")]
    rng = np.random.default_rng(0)
    n = 128

    def as_arm(structs):
        rep = tw.within_set_twins(structs)
        ci = float(np.std(rep.twin.astype(float)) / np.sqrt(max(len(structs), 1)))
        return {"meta": {"n": len(structs)},
                "realism": {"geom_kid": sc.metric(0.0, (-0.01, 0.01))},
                "novelty": {"voxel_twin_rate": sc.metric(
                    rep.rate, (rep.rate - 1.96 * ci, rep.rate + 1.96 * ci))}}, rep.rate

    ref, ref_rate = as_arm([val[i] for i in rng.permutation(len(val))[:n]])
    assert ref_rate < 0.05
    cal = comp.Calibration.from_arm(ref, "real_test")
    for seed in range(3):
        idx = np.random.default_rng(100 + seed).permutation(len(test))[:n]
        arm, rate = as_arm([test[i] for i in idx])
        assert rate < 0.05, (seed, rate)
        assert comp.score_arm("real_draw", arm, cal).status == "ok", (seed, rate)
    # 16 templates, 8 placements each: what a template-reproducing generator emits
    base = [test[i] for i in rng.permutation(len(test))[:16]]
    copies = [np.rot90(b, k, axes=(0, 2)) for b in base for k in range(4)] * 2
    arm, rate = as_arm(copies)
    assert rate == 1.0
    assert comp.score_arm("template_copier", arm, cal).status == "DQ"
