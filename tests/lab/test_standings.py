"""The cross-run board.

The assertions that matter are the exclusions and the separation. A board that
quietly includes an incomparable run, or quietly ranks an under-powered arm
beside a qualified one, is worse than no board: both render as a clean ordered
table, and neither announces that the ordering is meaningless.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from blockgen.eval.bench import protocol as proto
from tools.lab import catalog, standings

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "scorecards"

PRO = proto.get("houses32-v1")


def write(root: Path, run_id: str, card: dict) -> None:
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "scorecard.json").write_text(json.dumps(card))


def card(*, corpus="houses_32", tier="both", control_n=128,
         subs=(("m1", 3.0, 128),), n_ref=399, seed=0) -> dict:
    arms = {"real_test": {"meta": {"n": control_n, "track": "control",
                                   "kind": "control", "source": "in-memory"}}}
    board = [{"arm": "real_test", "score": 0.0, "status": "ok",
              "pillars": {}, "worst_pillar": "geometry"}]
    for name, score, n in subs:
        arms[name] = {"meta": {"n": n, "track": "ar", "kind": "submission",
                               "source": f"outputs/bench_arms/{name}.npz",
                               "provenance": {"writer": "npz", "model": name}}}
        board.append({"arm": name, "score": score, "status": "ok",
                      "pillars": {"geometry": score}, "worst_pillar": "geometry"})
    return {
        "schema_version": "bench/2",
        "context": {"corpus": corpus, "seed": seed, "n_ref_used": n_ref,
                    "split_key": f"{corpus}.s{seed}.70-15-15.v1"},
        "run": {"tier": tier, "name": "", "blockscore": board},
        "arms": arms, "warnings": [],
    }


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    root = tmp_path / "outputs"
    root.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(catalog, "OUTPUTS_ROOT", Path("outputs"))
    monkeypatch.setattr(catalog, "CORPUS_DIRS", ())
    monkeypatch.setattr(catalog, "_SOURCE_INDEX", None)
    catalog.forget_scorecards(); catalog.forget()
    yield root
    catalog.forget_scorecards(); catalog.forget()


def test_an_empty_outputs_gives_an_empty_board_not_an_error(outputs):
    b = standings.board()
    assert b["ranked"] == [] and b["provisional"] == [] and b["n_runs"] == 0
    assert b["protocol"]["id"] == "houses32-v1"


def test_a_qualifying_model_is_ranked(outputs):
    write(outputs, "run_20260101_000000_bench", card())
    b = standings.board()
    assert [r["model"] for r in b["ranked"]] == ["m1"]
    assert b["provisional"] == []
    assert b["n_matching"] == 1


def test_an_under_powered_arm_is_provisional_with_its_reason_not_ranked(outputs):
    write(outputs, "run_20260101_000000_bench", card(subs=(("m1", 3.0, 64),)))
    b = standings.board()
    assert b["ranked"] == []
    assert [r["model"] for r in b["provisional"]] == ["m1"]
    assert "at least 128" in b["provisional"][0]["unranked_reason"]


def test_ranking_is_by_score_ascending(outputs):
    write(outputs, "run_20260101_000000_bench",
          card(subs=(("slow", 9.0, 128), ("fast", 1.0, 128), ("mid", 4.0, 128))))
    assert [r["model"] for r in standings.board()["ranked"]] == ["fast", "mid", "slow"]


def test_a_model_in_several_runs_is_one_row_at_its_best_score(outputs):
    write(outputs, "run_20260101_000000_bench", card(subs=(("m1", 9.0, 128),)))
    write(outputs, "run_20260102_000000_bench", card(subs=(("m1", 2.0, 128),)))
    ranked = standings.board()["ranked"]
    assert len(ranked) == 1
    assert ranked[0]["score"] == 2.0 and ranked[0]["appearances"] == 2
    assert ranked[0]["run"] == "run_20260102_000000_bench"
    assert len(ranked[0]["history"]) == 2


def test_a_wrong_corpus_run_is_excluded_with_a_reason_never_silently(outputs):
    write(outputs, "run_20260101_000000_bench", card(corpus="houses_48"))
    b = standings.board()
    assert b["ranked"] == [] and b["provisional"] == []
    assert b["n_matching"] == 0 and len(b["skipped"]) == 1
    assert any("corpus" in r for r in b["skipped"][0]["reasons"])


def test_a_run_with_a_small_calibration_arm_is_excluded(outputs):
    """The measured 4.5x swing: the ruler, not the arm, is what failed."""
    write(outputs, "run_20260101_000000_bench", card(control_n=32))
    b = standings.board()
    assert b["n_matching"] == 0
    assert any("calibrated on 32" in r for r in b["skipped"][0]["reasons"])


def test_a_fast_tier_run_is_excluded(outputs):
    write(outputs, "run_20260101_000000_bench", card(tier="fast"))
    assert standings.board()["n_matching"] == 0


def test_controls_and_baselines_never_appear_on_the_board(outputs):
    """The board is models. `real_test` scoring 0.0 would otherwise win it."""
    write(outputs, "run_20260101_000000_bench", card())
    names = [r["model"] for r in standings.board()["ranked"]]
    assert "real_test" not in names


def test_a_disqualified_model_is_never_given_a_position(outputs):
    c = card(subs=(("m1", 1.0, 128),))
    c["run"]["blockscore"][1].update(status="DQ", score=None,
                                     disqualified="memorization")
    b = standings.board()  # nothing written yet
    write(outputs, "run_20260101_000000_bench", c)
    catalog.forget_scorecards()
    row = standings.board()["ranked"][0]
    assert row["status"] == "DQ"


def test_a_scored_run_beats_a_disqualified_one_for_the_same_model(outputs):
    dq = card(subs=(("m1", 1.0, 128),))
    dq["run"]["blockscore"][1].update(status="DQ", score=None, disqualified="memorization")
    write(outputs, "run_20260101_000000_bench", dq)
    write(outputs, "run_20260102_000000_bench", card(subs=(("m1", 7.0, 128),)))
    row = standings.board()["ranked"][0]
    assert row["status"] != "DQ" and row["score"] == 7.0


def test_model_identity_prefers_provenance_over_the_arm_label(outputs):
    """The same checkpoint under two `--arms` labels is one model."""
    a = card(subs=(("label_a", 5.0, 128),))
    a["arms"]["label_a"]["meta"]["provenance"]["model"] = "shared_ckpt"
    b = card(subs=(("label_b", 2.0, 128),))
    b["arms"]["label_b"]["meta"]["provenance"]["model"] = "shared_ckpt"
    write(outputs, "run_20260101_000000_bench", a)
    write(outputs, "run_20260102_000000_bench", b)
    ranked = standings.board()["ranked"]
    assert [r["model"] for r in ranked] == ["shared_ckpt"]
    assert ranked[0]["appearances"] == 2 and ranked[0]["score"] == 2.0


def test_an_unknown_protocol_is_reported_not_raised(outputs):
    b = standings.board("no-such-protocol")
    assert b["protocol"] is None and b["error"]


def test_a_float_dust_tie_goes_to_the_newer_run_not_to_arithmetic_noise(outputs):
    """Re-scoring one model under one protocol reproduced 87.14427315448478 and
    87.14427315440719 -- agreement to ten significant figures. Comparing those
    with `<` let 8e-11 decide which run represented the model, and it picked the
    older card, which predates example builds. The row lost its renders to dust.
    """
    write(outputs, "run_20260101_000000_bench",
          card(subs=(("m1", 87.14427315440719, 128),)))
    write(outputs, "run_20260202_000000_bench",
          card(subs=(("m1", 87.14427315448478, 128),)))
    row = standings.board()["ranked"][0]
    assert row["run"] == "run_20260202_000000_bench"


def test_a_genuinely_better_score_still_wins_regardless_of_age(outputs):
    """The tie-break must not become "newest wins"."""
    write(outputs, "run_20260101_000000_bench", card(subs=(("m1", 2.0, 128),)))
    write(outputs, "run_20260202_000000_bench", card(subs=(("m1", 9.0, 128),)))
    row = standings.board()["ranked"][0]
    assert row["run"] == "run_20260101_000000_bench" and row["score"] == 2.0
