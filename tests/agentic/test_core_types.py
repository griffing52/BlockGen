"""Core data types: View transforms, MinecraftWorld, Attempt, MethodSpec, MethodResult."""

from __future__ import annotations

import json

import numpy as np
import pytest

from blockgen.agentic.agent import Round
from blockgen.agentic.canvas import Canvas
from blockgen.agentic.core import (VIEWS, Charge, DreamVersion, ImageRequest,
                                   MethodResult, MethodSpec, MinecraftWorld, NullTrace,
                                   Score, TraceLike, View, get_view)
from blockgen.agentic.dsl import ExecutionReport
from blockgen.utils.data import Structure

from .fakes import (FakeActions, FakeImageGen, FakeRenderer, HUT_PROGRAM,
                    RecordingTrace, make_attempt)


def solid(x=4, y=10, z=4) -> Structure:
    return Structure(np.ones((x, y, z), np.int32), np.zeros((x, y, z), np.int32))


# --- views -------------------------------------------------------------------
def test_view_keep_below_cuts_height():
    s = solid()
    out = VIEWS["cutaway"].apply(s)
    occ = np.argwhere(out.occupied_mask)
    assert occ[:, 1].max() == 5                    # ceil(10 * 0.6) = 6 layers kept
    assert out.shape == s.shape                    # same box, roof removed
    plan = VIEWS["plan"].apply(s)
    assert np.argwhere(plan.occupied_mask)[:, 1].max() == 2


def test_view_section_cuts_axis():
    out = VIEWS["section_x"].apply(solid())
    occ = np.argwhere(out.occupied_mask)
    assert occ[:, 0].max() == 1 and occ[:, 1].max() == 9
    assert View("z", section=("z", 0.25)).apply(solid()).occupied_mask[:, :, 1:].sum() == 0


def test_view_crops_before_cutting_and_plain_view_is_identity():
    s = Structure(np.zeros((8, 20, 8), np.int32), np.zeros((8, 20, 8), np.int32))
    s.block_ids[2:4, 10:20, 2:4] = 1               # floating tower: crop first
    out = View("c", keep_below=0.5).apply(s)
    assert out.occupied_mask.sum() == 2 * 5 * 2
    assert VIEWS["iso"].apply(s) is s


def test_get_view():
    assert get_view("iso") is VIEWS["iso"]
    v = View("custom", 10, 20)
    assert get_view(v) is v
    with pytest.raises(KeyError, match="unknown view"):
        get_view("nope")
    assert VIEWS["legacy_a"].azim == 45 and VIEWS["legacy_b"].azim == 225


# --- world -------------------------------------------------------------------
COPY_PROG = "fill stone 0 0 0 1 1 1\ncopy 0 0 0 1 1 1\n"


def test_minecraft_world_runs_lowered_dsl_in_place():
    w = MinecraftWorld.empty((16, 16, 16))
    lowered = FakeActions().lower(HUT_PROGRAM)
    rep = w.run(lowered)
    assert rep.n_failed == 0 and w.block_count() == 25 + 2 * 16   # floor + 2 wall rings
    assert w.size == (16, 16, 16)
    assert w.to_structure().shape == (5, 3, 5)


def test_world_run_prepends_lowering_issues():
    from blockgen.agentic.core import Lowered
    from blockgen.agentic.dsl import Issue
    w = MinecraftWorld.empty((8, 8, 8))
    issue = Issue(line_no=1, text="bad", message="lowering failed")
    rep = w.run(Lowered(source="x", program_text="set stone 0 0 0", issues=[issue]))
    assert rep.issues[0] is issue and w.block_count() == 1


def test_minecraft_world_clone_copies_clipboard():
    w = MinecraftWorld.empty((16, 16, 16))
    w.run(FakeActions().lower(COPY_PROG))
    c = w.clone()
    rep = c.run(FakeActions().lower("paste 5 0 0"))
    assert rep.n_failed == 0 and c.block_count() == 16
    assert w.block_count() == 8                    # original untouched
    c.canvas.clipboard.block_ids[...] = 0          # deep copy, not shared
    assert w.canvas.clipboard.block_ids.any()


def test_legacy_canvas_clone_drops_clipboard_documented_quirk():
    """Legacy Canvas.clone does NOT copy the clipboard; kept for byte-parity of
    published legacy patch-mode runs. Only MinecraftWorld.clone fixes it."""
    from blockgen.agentic.dsl import run_program
    canvas, _, _ = run_program(COPY_PROG, size=(16, 16, 16))
    assert canvas.clipboard is not None
    clone = canvas.clone()
    assert clone.clipboard is None
    _, rep, _ = run_program("paste 5 0 0", canvas=clone, size=(16, 16, 16))
    assert rep.n_failed == 1


def test_world_from_structure_round_trip():
    w = MinecraftWorld.empty((16, 16, 16))
    w.run(FakeActions().lower(HUT_PROGRAM))
    w2 = MinecraftWorld.from_structure(w.to_structure(), size=(16, 16, 16))
    assert w2.block_count() == w.block_count()


# --- attempt / dreams / image request ------------------------------------------
def test_attempt_is_a_round_with_defaults():
    att = make_attempt()
    assert isinstance(att, Round) and att.blocks > 0 and att.accepted is False
    summary = att.summary()
    assert summary["n_commands"] == 2 and summary["score"] is None
    att.score = Score(0.5, parts={"x": 0.5})
    assert json.dumps(att.summary())


def test_dream_version_digest_and_image_request_key():
    dv = DreamVersion("iso", 0, b"png-bytes", "p", "fake")
    assert len(dv.digest) == 16
    r = ImageRequest("a house", refs=(b"r",), seed=1, view="iso")
    k = r.key("openai", "gpt-image-1-mini")
    assert k == ImageRequest("a house", refs=(b"r",), seed=1, view="iso").key("openai", "gpt-image-1-mini")
    for other in (ImageRequest("a house", refs=(b"x",), seed=1, view="iso"),
                  ImageRequest("a house", refs=(b"r",), seed=2, view="iso"),
                  ImageRequest("a house", refs=(b"r",), seed=1, view="plan")):
        assert other.key("openai", "gpt-image-1-mini") != k
    assert r.key("openai", "gpt-image-1") != k


def test_fakes_satisfy_protocols():
    from blockgen.agentic.core import ActionSpace, ImageGen, Renderer
    assert isinstance(FakeActions(), ActionSpace)
    assert isinstance(FakeImageGen(), ImageGen)
    assert isinstance(FakeRenderer(), Renderer)
    assert isinstance(NullTrace(), TraceLike) and isinstance(RecordingTrace(), TraceLike)
    assert isinstance(MinecraftWorld.empty((4, 4, 4)), __import__(
        "blockgen.agentic.core", fromlist=["World"]).World)


# --- MethodSpec ----------------------------------------------------------------
def test_method_spec_hash_stable_and_sensitive():
    a = MethodSpec("a", params={"x": 1, "y": [1, 2]}, score={"embed": 1.0, "validity": 0.3})
    b = MethodSpec("b", description="other", tags={"t": 1},
                   score={"validity": 0.3, "embed": 1.0}, params={"y": [1, 2], "x": 1})
    assert a.hash() == b.hash() and len(a.hash()) == 12          # name/desc/tags cosmetic
    assert MethodSpec("a", params={"x": 2, "y": [1, 2]}).hash() != a.hash()
    assert MethodSpec("a", gen={}).hash() == MethodSpec("a").hash()   # defaults filled
    assert MethodSpec("a", canvas=(32, 32, 32)).hash() == MethodSpec("a", canvas=[32, 32, 32]).hash()


def test_method_spec_roles():
    s = MethodSpec("a", llm={"builder": "openai:a", "judge": "claude-code:sonnet"})
    roles = s.roles()
    assert roles["critic"] == "openai:a" and roles["judge"] == "claude-code:sonnet"
    assert MethodSpec("a", llm={"critic": "x"}).roles()["builder"] == "x"
    with pytest.raises(ValueError, match="empty"):
        MethodSpec("a", llm={}).roles()


def test_method_spec_provenance():
    s = MethodSpec("dream", method="dream", image="openai:gpt-image-1-mini?quality=low",
                   contexts=["palette", {"examples": {"k": 1, "pool": "houses_32"}}],
                   dreams=["iso", {"key": "back", "template": "t"}],
                   accept={"improve": {"eps": 0.01}}, params={"max_iters": 3},
                   tags={"train_split": "oracle_train"})
    p = s.provenance()
    assert p["recipe_hash"] == s.hash() and p["dreams"] == ["iso", "back"]
    assert p["contexts"] == ["palette", "examples?k=1&pool=houses_32"]
    assert p["loop"]["accept"] == "improve?eps=0.01" and p["loop"]["max_iters"] == 3
    assert p["train_split"] == "oracle_train"                 # tags override
    assert MethodSpec("x", contexts=[{"examples": {"k": 1, "pool": "houses_32"}}]
                      ).provenance()["train_split"] == "train_only"
    assert MethodSpec("x").provenance()["train_split"] == "none"
    assert json.dumps(p)


# --- MethodResult ----------------------------------------------------------------
def _result(**kw) -> MethodResult:
    a0 = make_attempt(idx=0, score=Score(0.4), accepted=True)
    a1 = make_attempt(idx=1, stage="match0", score=Score(0.7), accepted=False)
    w = a1.world
    charges = [Charge("builder", "llm", "gpt-5-mini", usd=0.01, would_cost_usd=0.01,
                      prompt_tokens=100, completion_tokens=50),
               Charge("critic", "llm", "gpt-5-mini", usd=0.0, would_cost_usd=0.02, cached=True),
               Charge("image", "image", "gpt-image-1-mini", usd=0.005, would_cost_usd=0.005, images=1)]
    base = dict(description="a hut", structure=w.to_structure(), canvas=w.canvas,
                program_text=a1.program_text, report=a1.report, rounds=[a0, a1],
                charges=charges, best_idx=1,
                dreams={"iso": [DreamVersion("iso", 0, b"x", "p", "fake")]})
    base.update(kw)
    return MethodResult(**base)


def test_method_result_cost_from_charges():
    r = _result()
    assert r.cost_usd == pytest.approx(0.015) and r.would_cost_usd == pytest.approx(0.035)
    assert r.image_cost_usd == pytest.approx(0.005) and r.tokens == (100, 50)
    assert r.cost_known is True and r.cached is False
    assert _result(charges=[]).cost_known is False
    unpriced = Charge("builder", "llm", "claude-code/sonnet", priced=False)
    assert _result(charges=[unpriced]).cost_known is False


def test_method_result_metrics_and_to_dict():
    r = _result()
    m = r.method_metrics()
    assert m["n_attempts"] == 2 and m["accept_rate"] == 0.5 and m["best_idx"] == 1
    assert m["dream_score_first"] == 0.4 and m["dream_score_best"] == 0.7
    assert m["dream_gain"] == pytest.approx(0.3) and m["n_dream_versions"] == 1
    assert m["n_llm_calls"] == 2 and m["n_images"] == 1
    full = r.metrics()
    # cost_usd keeps the legacy (full price) meaning; what was paid is paid_usd.
    assert full["cost_usd"] == pytest.approx(0.035) and full["paid_usd"] == pytest.approx(0.015)
    assert full["blocks"] > 0 and "dream_gain" in full
    d = r.to_dict()
    assert len(d["attempts"]) == 2 and d["dreams"]["iso"][0]["digest"] and len(d["charges"]) == 3
    json.dumps(d)
    assert _result(best_idx=7).method_metrics()["dream_score_best"] is None


def test_method_result_is_a_build_result_for_report():
    from blockgen.agentic.agent import BuildResult
    r = _result()
    assert isinstance(r, BuildResult) and r.attempts == r.rounds
    empty = MethodResult(description="x", structure=Canvas((4, 4, 4)).to_structure(crop=False),
                         canvas=Canvas((4, 4, 4)), program_text="", report=ExecutionReport())
    assert empty.method_metrics()["dream_gain"] is None and empty.metrics()["blocks"] == 0
