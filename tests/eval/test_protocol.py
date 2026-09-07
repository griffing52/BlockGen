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
            "seed": 0, "n_ref_used": 399}
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
