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
         subs=(("m1", 3.0, 128),), n_ref=399, seed=0, reports=None) -> dict:
    """`reports` is `{arm: sample report}`: the block the trainer and the
    post-training stages write at `provenance.report`, where `train_split`,
    `trained_on_signals`, `model` and `checkpoint` live."""
    arms = {"real_test": {"meta": {"n": control_n, "track": "control",
                                   "kind": "control", "source": "in-memory"}}}
    board = [{"arm": "real_test", "score": 0.0, "status": "ok",
              "pillars": {}, "worst_pillar": "geometry"}]
    for name, score, n in subs:
        prov = {"writer": "npz", "model": name}
        if reports and name in reports:
            prov["report"] = dict(reports[name])
        arms[name] = {"meta": {"n": n, "track": "ar", "kind": "submission",
                               "source": f"outputs/bench_arms/{name}.npz",
                               "provenance": prov}}
        board.append({"arm": name, "score": score, "status": "ok",
                      "pillars": {"geometry": score}, "worst_pillar": "geometry"})
    return {
        "schema_version": "bench/2",
        "context": {"corpus": corpus, "seed": seed, "n_ref_used": n_ref,
                    "split_key": f"{corpus}.s{seed}.70-15-15.v1",
                    # a current card: scored with the pinned descriptor
                    "geom_descriptor": "patch2_orb55_t5_h8_s12"},
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


def test_an_appearance_at_the_board_n_beats_a_lower_score_at_another_n(outputs):
    """Scores at different n are not ordered, so the row is the n=256 appearance, then the larger n."""
    write(outputs, "run_20260101_000000_bench", card(subs=(("m1", 2.0, 128),)))
    write(outputs, "run_20260202_000000_bench", card(subs=(("m1", 9.0, 256),)))
    write(outputs, "run_20260303_000000_bench", card(subs=(("m1", 1.0, 399),)))
    row = standings.board()["ranked"][0]
    assert row["run"] == "run_20260202_000000_bench" and row["score"] == 9.0


# --- equal n ---------------------------------------------------------------
# BlockScore is comparable only at equal n. The board does not enforce that by
# unranking: it says what n every row stands on, counts the arms at each, picks
# the group a page should open on, and filters to one when asked. The tests pin
# that nothing goes missing on the way.

def test_rows_carry_n_and_the_board_groups_by_it(outputs):
    write(outputs, "run_20260101_000000_bench",
          card(subs=(("a", 3.0, 128), ("b", 4.0, 128), ("big", 1.0, 399))))
    b = standings.board()
    assert {r["model"]: r["n"] for r in b["ranked"]} == {"a": 128, "b": 128, "big": 399}
    assert b["n_groups"] == [
        {"n": 399, "arms": 1, "ranked": 1, "provisional": 0},
        {"n": 128, "arms": 2, "ranked": 2, "provisional": 0}]
    # the largest n with at least two arms: one arm is not a ranking
    assert b["default_n"] == 128
    assert b["mixed_n"] is True and b["n_filter"] is None and b["n_hidden"] == 0


def test_filtering_by_n_hides_rows_and_says_how_many_never_unranks(outputs):
    write(outputs, "run_20260101_000000_bench",
          card(subs=(("a", 3.0, 128), ("b", 4.0, 128), ("big", 1.0, 399), ("small", 2.0, 64))))
    b = standings.board(n=128)
    assert [r["model"] for r in b["ranked"]] == ["a", "b"]
    assert b["provisional"] == []                 # `small` is at another n, so out of view
    assert b["n_hidden"] == 2 and b["n_filter"] == 128 and b["mixed_n"] is False
    # the groups still count every arm, so the filter cannot pass for the whole
    assert [(g["n"], g["arms"]) for g in b["n_groups"]] == [(399, 1), (128, 2), (64, 1)]
    assert b["n_groups"][2] == {"n": 64, "arms": 1, "ranked": 0, "provisional": 1}
    # and the under-powered arm is still provisional, with its reason, in the full view
    full = standings.board()
    assert [r["model"] for r in full["provisional"]] == ["small"]
    assert "at least 128" in full["provisional"][0]["unranked_reason"]


def test_default_n_falls_back_to_the_largest_n_when_every_group_is_one_arm(outputs):
    write(outputs, "run_20260101_000000_bench",
          card(subs=(("a", 3.0, 128), ("big", 1.0, 399))))
    assert standings.board()["default_n"] == 399


def test_an_unrecorded_n_is_its_own_group_not_a_crash(outputs):
    c = card(subs=(("a", 3.0, 128), ("odd", 2.0, 128)))
    del c["arms"]["odd"]["meta"]["n"]
    write(outputs, "run_20260101_000000_bench", c)
    b = standings.board()
    assert b["n_groups"][-1]["n"] is None and b["n_groups"][-1]["arms"] == 1
    assert b["default_n"] == 128                  # None never wins the default
    assert "n=?" in next(r for r in b["provisional"] if r["model"] == "odd")["unranked_reason"]


def test_an_empty_board_has_empty_n_fields(outputs):
    b = standings.board()
    assert b["n_groups"] == [] and b["default_n"] is None and b["mixed_n"] is False
    b = standings.board("no-such-protocol")
    assert b["n_groups"] == [] and b["default_n"] is None


# --- training provenance -----------------------------------------------------
REPORT = {"run": "outputs/run_20260101_000000_m1", "step": 8000, "n": 128, "seed": 0,
          "train_split": "canonical",
          "model": {"arch": "ar2", "preset": "5m", "d_model": 256, "layers": 6},
          "checkpoint": "outputs/run_20260101_000000_m1/ckpt_0008000.pt"}


def test_model_checkpoint_and_split_from_the_report_are_surfaced(outputs):
    write(outputs, "run_20260101_000000_bench",
          card(subs=(("m1", 3.0, 128),), reports={"m1": REPORT}))
    row = standings.board()["ranked"][0]
    assert row["model"] == "m1"                   # identity is still provenance.model
    assert row["model_label"] == "m1"             # the name the sampler wrote wins
    assert row["model_detail"] == REPORT["model"] # the architecture dict rides beside it
    assert row["checkpoint"] == REPORT["checkpoint"]
    assert row["train_split"] == "canonical"
    assert row["training_run"] == REPORT["run"]
    assert row["trained_on_signals"] == []
    assert row["unranked_reason"] == ""           # canonical split, n met: ranked


def test_a_report_model_string_is_the_identity_when_provenance_has_none(outputs):
    a = card(subs=(("label_a", 5.0, 128),), reports={"label_a": {**REPORT, "model": "shared"}})
    del a["arms"]["label_a"]["meta"]["provenance"]["model"]
    b = card(subs=(("label_b", 2.0, 128),), reports={"label_b": {**REPORT, "model": "shared"}})
    del b["arms"]["label_b"]["meta"]["provenance"]["model"]
    write(outputs, "run_20260101_000000_bench", a)
    write(outputs, "run_20260102_000000_bench", b)
    ranked = standings.board()["ranked"]
    assert [r["model"] for r in ranked] == ["shared"] and ranked[0]["appearances"] == 2


def test_a_report_model_dict_never_becomes_an_identity(outputs):
    """An architecture dict describes a model and does not name one: two
    different checkpoints of one preset must not collapse to one row."""
    a = card(subs=(("ck1", 5.0, 128),), reports={"ck1": REPORT})
    del a["arms"]["ck1"]["meta"]["provenance"]["model"]
    b = card(subs=(("ck2", 2.0, 128),), reports={"ck2": REPORT})
    del b["arms"]["ck2"]["meta"]["provenance"]["model"]
    write(outputs, "run_20260101_000000_bench", a)
    write(outputs, "run_20260102_000000_bench", b)
    ranked = standings.board()["ranked"]
    assert [r["model"] for r in ranked] == ["ck2", "ck1"]
    assert ranked[0]["model_label"] == "ar2/5m 256x6"   # the dict, rendered for display


def test_a_dict_checkpoint_from_the_trainer_report_is_unwrapped_to_its_file(outputs):
    """`blockgen/train/sample.py` writes `checkpoint: {"file", "step"}`; the row
    carries the file (so a page finds the `run_` segment) and the step beside it."""
    rep = {**REPORT, "checkpoint": {"file": "outputs/run_20261101_000000_x/ckpt_0008000.pt",
                                    "step": 8000}}
    write(outputs, "run_20260101_000000_bench", card(subs=(("m1", 3.0, 128),), reports={"m1": rep}))
    row = standings.board()["ranked"][0]
    assert row["checkpoint"] == "outputs/run_20261101_000000_x/ckpt_0008000.pt"
    assert row["checkpoint_step"] == 8000
    assert "run_20261101_000000_x" in row["checkpoint"].split("/")
    # the string shape is untouched, and carries no step
    assert standings.training({"provenance": {"report": REPORT}})["checkpoint"] == REPORT["checkpoint"]
    assert standings.training({"provenance": {"report": REPORT}})["checkpoint_step"] is None
    # a dict with no file is "unrecorded", not the string "None"
    assert standings.training({"provenance": {"report": {"checkpoint": {"step": 3}}}})["checkpoint"] is None
    assert standings.checkpoint_file({"file": "a/ckpt.pt", "step": True}) == ("a/ckpt.pt", None)


def test_trained_on_signals_is_lifted_and_flagged_but_not_unranked(outputs):
    """Flagged the way an oracle_train arm is flagged (a visible marker, the
    names in the tooltip), and nothing more: the protocol decides ranking, and
    it says nothing about reward signals."""
    rft = {**REPORT, "trained_on_signals": ["geom_kid_reward", "traversal"]}
    write(outputs, "run_20260101_000000_bench",
          card(subs=(("rft", 2.0, 128), ("sft", 3.0, 128)), reports={"rft": rft, "sft": REPORT}))
    b = standings.board()
    rows = {r["model"]: r for r in b["ranked"]}
    assert rows["rft"]["trained_on_signals"] == ["geom_kid_reward", "traversal"]
    assert rows["sft"]["trained_on_signals"] == []
    assert rows["rft"]["unranked_reason"] == ""
    assert [r["model"] for r in b["ranked"]] == ["rft", "sft"]
    # a bare string is one signal, and a top-level provenance field is read too
    c = card(subs=(("bon", 1.0, 128),))
    c["arms"]["bon"]["meta"]["provenance"]["trained_on_signals"] = "reward_x"
    write(outputs, "run_20260102_000000_bench", c)
    rows = {r["model"]: r for r in standings.board()["ranked"]}
    assert rows["bon"]["trained_on_signals"] == ["reward_x"]


def test_an_oracle_split_arm_still_goes_provisional_with_its_reason(outputs):
    """The existing flag, unchanged: this is the behaviour the signals flag copies."""
    write(outputs, "run_20260101_000000_bench",
          card(subs=(("orc", 1.0, 128),), reports={"orc": {**REPORT, "train_split": "oracle_train"}}))
    b = standings.board()
    assert b["ranked"] == [] and b["provisional"][0]["train_split"] == "oracle_train"
    assert "oracle_train" in b["provisional"][0]["unranked_reason"]


# --- cards under outputs/<run>/bench/ ----------------------------------------
def test_a_card_the_trainer_wrote_under_bench_reaches_the_board(outputs):
    """The trainer's bench step writes `outputs/<run>/bench/scorecard.json`; 27
    of them were invisible to a board that globbed one level deep."""
    run = "run_20260105_120000_houses32_5m"
    d = outputs / run
    (d / "bench").mkdir(parents=True)
    (d / "bench" / "scorecard.json").write_text(json.dumps(
        card(subs=(("houses32_5m", 2.5, 128),), reports={"houses32_5m": REPORT})))
    (d / "run.json").write_text(json.dumps({"config": {
        "name": "houses32_5m", "stage": "sft", "arch": "ar2", "preset": "5m",
        "train": ["sft-houses32"], "val": ["eval-holdout"], "steps": 8000,
        "trained_on_signals": []}, "meta": {}}))
    (outputs / "registry.jsonl").write_text(json.dumps(
        {"run": f"outputs/{run}", "name": "houses32_5m", "params": 5525243,
         "steps": 8000, "git": "abc1234"}) + "\n")
    catalog.forget_train_runs()
    b = standings.board()
    assert b["n_matching"] == 1
    row = b["ranked"][0]
    assert row["model"] == "houses32_5m"
    assert row["run"] == f"{run}/bench" and row["run_dir"] == run
    assert row["train_run_name"] == "houses32_5m"
    assert row["history"][0]["run"] == f"{run}/bench"


def test_the_api_route_passes_n_through_and_rejects_junk(outputs):
    from tools.lab import api
    write(outputs, "run_20260101_000000_bench",
          card(subs=(("a", 3.0, 128), ("big", 1.0, 399), ("c", 2.0, 128))))
    out = api.api_standings({"n": ["128"]})
    assert [r["model"] for r in out["ranked"]] == ["c", "a"] and out["n_hidden"] == 1
    assert api.api_standings({})["n_filter"] is None
    assert api.api_standings({"n": [""]})["n_filter"] is None
    for junk in ("many", "128.9", "1e2", "nan"):
        with pytest.raises(api.LabError):
            api.api_standings({"n": [junk]})
    assert api.api_standings({"n": [" 128 "]})["n_filter"] == 128
