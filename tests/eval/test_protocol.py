"""What makes two runs comparable — the rule the leaderboard rests on.

`match_run` is the only thing standing between "a leaderboard" and "a table of
numbers from different rulers laid out in one column". The tests that matter
most here are the ones asserting a run is REJECTED: a protocol that is too
permissive produces a board that looks authoritative and is not, and that failure
is silent, which is why it is pinned rather than left to review.

The calibration case is a measured one, not a hypothetical. `native_oriented`
scored 87.14 against a 128-build `real_test` and 19.05 against a 32-build one --
same model, same corpus, same split, same tier. BlockScore's unit is that arm's
spread, so its size is part of the protocol.
"""

from __future__ import annotations

import numpy as np
import pytest

from blockgen.eval.bench import protocol as proto


def ctx(**over):
    base = {"corpus": "houses_32", "split_key": "houses_32.s0.70-15-15.v1",
            "seed": 0, "n_ref_used": 399, "geom_descriptor": "patch2_orb55_t5_h8_s12"}
    base.update(over)
    return base


def arms(control_n=128):
    return {"real_test": {"meta": {"n": control_n, "track": "control"}}}


def run(**over):
    base = {"tier": "both"}
    base.update(over)
    return base


P = proto.get("houses32-v1")


def test_the_default_protocol_exists_and_is_the_one_results_were_measured_under():
    assert P is not None and P.id == proto.DEFAULT_ID
    assert P.corpus == "houses_32"
    assert P.split_key == "houses_32.s0.70-15-15.v1"
    assert P.needs_full_tier


def test_a_conforming_run_matches():
    assert proto.match_run(ctx(), run(), P, arms()) == []


@pytest.mark.parametrize("over,expect", [
    ({"corpus": "houses_48"}, "corpus"),
    ({"split_key": "houses_32.s1.70-15-15.v1"}, "split"),
    ({"seed": 1}, "seed"),
    ({"n_ref_used": 64}, "reference"),
])
def test_context_mismatches_are_rejected_with_a_reason(over, expect):
    why = proto.match_run(ctx(**over), run(), P, arms())
    assert why and any(expect in w for w in why), why


def test_a_fast_tier_run_is_rejected_because_it_measures_no_appearance():
    why = proto.match_run(ctx(), run(tier="fast"), P, arms())
    assert why and any("appearance" in w for w in why)


def test_a_small_calibration_arm_bends_the_ruler_and_is_rejected():
    """The measured 4.5x swing. A 32-build control is a different unit."""
    why = proto.match_run(ctx(), run(), P, arms(control_n=32))
    assert why and any("calibrated on 32" in w for w in why), why


def test_a_missing_calibration_arm_is_rejected():
    why = proto.match_run(ctx(), run(), P, {"native_oriented": {"meta": {"n": 999}}})
    assert why and any("nothing calibrates" in w for w in why), why


def test_arms_are_optional_so_a_card_without_them_still_gets_a_verdict():
    assert proto.match_run(ctx(), run(), P) == []


def test_min_n_is_checked_per_arm_not_per_run():
    """`--n` caps controls but not file-backed arms, so one run holds both."""
    assert proto.match_arm({"n": 128}, P) == []
    why = proto.match_arm({"n": 64}, P)
    assert why and "at least 128" in why[0]
    assert proto.match_arm({}, P)                      # no n recorded is not a pass


def test_verdict_is_json_shaped_and_never_raises():
    v = proto.verdict(ctx(), run(), arms=arms())
    assert v == {"id": "houses32-v1", "official": True, "reasons": []}
    bad = proto.verdict({}, {}, arms={})
    assert bad["official"] is False and bad["reasons"]


def test_an_unknown_protocol_id_is_none_not_an_exception():
    assert proto.get("nope") is None
    assert proto.get(None) is None


def test_slate_reports_whether_it_can_actually_align():
    assert proto.Slate(k=8).aligned is False
    assert proto.Slate(k=2, prompts=("a hut",)).aligned is True
    assert P.slate.to_json()["aligned"] is False        # houses32-v1 is unconditional


# --- training provenance ----------------------------------------------------
# The gap these close: `match_run` validates the *evaluation* and a pair of LLM
# arms trained on Track D's pre-split 90/10 shuffle -- 53 canonical-test builds in
# their training set -- scored on the canonical split at n=128 and were ranked.

def test_an_arm_trained_off_protocol_is_not_rankable():
    why = proto.match_arm({"n": 128, "report": {"train_split": "trackd"}}, P)
    assert why and "trackd" in why[0] and "test builds" in why[0]


def test_an_arm_trained_on_the_protocol_split_is_rankable():
    assert proto.match_arm({"n": 128, "report": {"train_split": "canonical"}}, P) == []


def test_silence_about_the_training_split_is_not_consent():
    """An arm that records a report but no split is unranked, not assumed clean."""
    why = proto.match_arm({"n": 128, "report": {"model": "x"}}, P)
    assert why and "not recorded" in why[0]


def test_controls_carry_no_report_and_are_unaffected():
    """`real_test` and the procedural baselines have no training provenance to
    check -- requiring one would disqualify the ruler itself."""
    assert proto.match_arm({"n": 128}, P) == []
    assert proto.match_arm({"n": 128, "report": {}}, P) == []


def test_both_reasons_are_reported_together():
    why = proto.match_arm({"n": 64, "report": {"train_split": "trackd"}}, P)
    assert len(why) == 2


# --- the descriptor is part of the ruler -----------------------------------
def test_a_card_scored_with_the_legacy_descriptor_is_unranked_with_a_reason():
    """Every card written before 2026-10-23: no stamp, no room scalars."""
    legacy = ctx()
    legacy.pop("geom_descriptor")
    why = proto.match_run(legacy, run(), P, arms())
    assert len(why) == 1 and "descriptor" in why[0] and "re-score" in why[0]


def test_room_scalars_in_an_arm_identify_the_current_descriptor_without_a_stamp():
    legacy = ctx()
    legacy.pop("geom_descriptor")
    a = arms()
    a["real_test"]["geometry_scalars"] = {"walk_standing": {"w1_norm": 0.1}}
    assert proto.match_run(legacy, run(), P, a) == []
    assert proto.run_descriptor(legacy, a) == P.descriptor
    assert proto.run_descriptor(legacy, arms()) == proto.LEGACY_DESCRIPTOR


def test_the_stamp_wins_over_inference():
    assert proto.run_descriptor({"geom_descriptor": "something_else"}, arms()) == "something_else"
    assert "descriptor" in proto.match_run(ctx(geom_descriptor="something_else"), run(), P, arms())[0]
