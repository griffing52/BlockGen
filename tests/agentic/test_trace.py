"""Trace writer and contact sheets (WP6): exact layout, resolvable assets, sheet sizes."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import List

import numpy as np
import pytest
from PIL import Image

from blockgen.agentic import core, trace as tr
from blockgen.agentic.core import (Charge, DreamVersion, MethodResult, MinecraftWorld,
                                   Score, TraceLike)
from blockgen.agentic.providers import (LLMResponse, encode_png, image_message,
                                        text_message)

from .fakes import (FAKE_DREAMS, HUT_PROGRAM, DEMO_PROGRAM, FakeRenderer, make_attempt,
                    make_fake_kit, solid_png)


def _size(path: Path):
    return Image.open(path).size


def _renders(att, views=("iso", "iso_back")):
    r = FakeRenderer(px=32)
    imgs = r.render(att.world.to_structure(), [core.get_view(v) for v in views])
    return {v: encode_png(a) for v, a in zip(views, imgs)}


def _result(attempts, dreams, best_idx, charges=()) -> MethodResult:
    best = attempts[best_idx]
    return MethodResult(description="a small hut", structure=best.world.to_structure(),
                        canvas=best.world.canvas, program_text=best.program_text,
                        report=best.report, rounds=list(attempts), dreams=dreams,
                        charges=list(charges), best_idx=best_idx, spec_name="fake",
                        recipe_hash="abc123def456")


def _scenario(root: Path, *, contact: bool = True):
    """A fake dream episode driven through the TraceLike verbs, in Episode order."""
    t = tr.Trace(root, contact=contact)
    d0 = DreamVersion("iso", 0, solid_png((200, 30, 30)), "p0", "fake_image:fake-solid", seed=0)
    d1 = DreamVersion("iso", 1, solid_png((30, 200, 30)), "p1", "fake_image:fake-solid", seed=1000)
    db = DreamVersion("back", 0, solid_png((30, 30, 200)), "pb", "fake_image:fake-solid")
    for d in (d0, d1, db):
        t.dream(d)
    sys_msg = text_message("system", "SYSTEM")
    turn = image_message("user", "TARGET iso", [d1.png])
    t.request("builder", [sys_msg, turn], LLMResponse(text=HUT_PROGRAM, model="scripted"),
              "generate")
    a0 = make_attempt(HUT_PROGRAM, idx=0, stage="generate")
    a0.renders = _renders(a0)
    a0.score = Score(0.4, {"fake_scorer": 0.4}, dream_id="d1")
    a0.accepted, a0.reason = True, "ok"
    t.attempt(a0)
    t.request("critic", [sys_msg, image_message("user", "compare", [a0.renders["iso"], d1.png])],
              LLMResponse(text=DEMO_PROGRAM, model="scripted"), "match0")
    a1 = make_attempt(DEMO_PROGRAM, idx=1, stage="match0")
    a1.renders = _renders(a1)
    a1.score = Score(0.7, {"fake_scorer": 0.7}, dream_id="d1")
    a1.accepted, a1.reason = True, "improved"
    t.attempt(a1)
    a2 = make_attempt("", idx=2, stage="match1")
    a2.score = Score(0.0, gated="empty")
    a2.accepted, a2.reason = False, "empty after non-empty"
    t.attempt(a2)
    t.event("redream", key="iso")
    d1.critique = {"score": 8, "buildable": True}       # verdict lands after dream()
    a0.score = Score(0.45, {"fake_scorer": 0.45}, dream_id="d2")   # rescore_all
    charges = [Charge("builder", "llm", "scripted", priced=False),
               Charge("image", "image", "fake-solid", usd=0.01, would_cost_usd=0.01, images=1)]
    res = _result([a0, a1, a2], {"iso": [d0, d1], "back": [db]}, 1, charges)
    t.close(res)
    return t, res


def test_trace_is_tracelike(tmp_path):
    assert isinstance(tr.Trace(tmp_path / "t"), TraceLike)


def test_exact_file_layout(tmp_path):
    root = tmp_path / "000"
    _scenario(root)
    files = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    assets = [f for f in files if f.startswith("assets/")]
    rest = [f for f in files if not f.startswith("assets/")]
    it = ["source.txt", "program.dsl", "delta.dsl", "report.json", "score.json",
          "structure.npz"]
    expected = (["charges.jsonl", "contact.png", "events.jsonl", "messages.jsonl",
                 "result.json"]
                + [f"dreams/{k}.{e}" for k in ("back_v0", "iso_v0", "iso_v1")
                   for e in ("json", "png")]
                + [f"iters/00_generate/{f}" for f in it + ["render_iso.png",
                                                            "render_iso_back.png"]]
                + [f"iters/01_match0/{f}" for f in it + ["render_iso.png",
                                                          "render_iso_back.png"]]
                + [f"iters/02_match1/{f}" for f in it])
    assert rest == sorted(expected)
    # Content-addressed: d1.png referenced twice, stored once; plus the a0 render.
    assert len(assets) == 2
    assert all(len(Path(a).stem) == 16 for a in assets)


def test_messages_image_refs_resolve(tmp_path):
    root = tmp_path / "b"
    _scenario(root)
    lines = (root / "messages.jsonl").read_text().splitlines()
    assert len(lines) == 2
    recs = [json.loads(line) for line in lines]
    assert [r["role"] for r in recs] == ["builder", "critic"]
    assert [r["stage"] for r in recs] == ["generate", "match0"]
    assert recs[0]["response"] == HUT_PROGRAM
    assert recs[0]["usage"]["model"] == "scripted"
    n_images = 0
    for r in recs:
        for m in r["messages"]:
            for p in m["parts"]:
                assert set(p) <= {"text", "image"} and len(p) == 1
                if "image" in p:
                    n_images += 1
                    assert (root / p["image"]).is_file()
    assert n_images == 3
    dream_ref = recs[0]["messages"][1]["parts"][1]["image"]
    assert (root / dream_ref).read_bytes() == (root / "dreams/iso_v1.png").read_bytes()


def test_iteration_files_content(tmp_path):
    root = tmp_path / "b"
    _, res = _scenario(root)
    d = root / "iters" / "01_match0"
    assert (d / "source.txt").read_text() == DEMO_PROGRAM
    assert (d / "program.dsl").read_text() == res.attempts[1].program_text
    assert json.loads((d / "report.json").read_text())["blocks"] == res.attempts[1].blocks
    s = tr.load_structure(d / "structure.npz")
    w = res.attempts[1].world.to_structure(crop=False)
    np.testing.assert_array_equal(s.block_ids, w.block_ids)
    np.testing.assert_array_equal(s.block_data, w.block_data)
    assert MinecraftWorld.from_structure(s).block_count() == res.attempts[1].blocks
    sc = json.loads((d / "score.json").read_text())
    assert sc["accepted"] and sc["score"]["value"] == 0.7 and sc["stage"] == "match0"


def test_close_refreshes_rescored_and_critique(tmp_path):
    root = tmp_path / "b"
    _scenario(root)
    sc = json.loads((root / "iters/00_generate/score.json").read_text())
    assert sc["score"]["value"] == 0.45 and sc["score"]["dream_id"] == "d2"
    meta = json.loads((root / "dreams/iso_v1.json").read_text())
    assert meta["critique"] == {"score": 8, "buildable": True}
    assert meta["view"] == "iso" and meta["seed"] == 1000
    assert json.loads((root / "dreams/back_v0.json").read_text())["view"] == "iso_back"


def test_events_charges_result(tmp_path):
    root = tmp_path / "b"
    _, res = _scenario(root)
    ev = [json.loads(x) for x in (root / "events.jsonl").read_text().splitlines()]
    assert ev[0]["kind"] == "redream" and ev[0]["key"] == "iso"
    ch = [json.loads(x) for x in (root / "charges.jsonl").read_text().splitlines()]
    assert [c["role"] for c in ch] == ["builder", "image"] and ch[1]["usd"] == 0.01
    blob = json.loads((root / "result.json").read_text())
    assert blob["recipe_hash"] == "abc123def456" and blob["best_idx"] == 1
    assert blob["best_dir"] == "01_match0"
    assert len(blob["attempts"]) == 3 and blob["trace_dir"] == str(root)
    assert res.trace_dir == str(root)


def test_contact_png_dimensions(tmp_path):
    root = tmp_path / "b"
    _scenario(root)
    lay = tr.contact_layout(root)
    assert lay["views"] == ["iso", "iso_back"]
    assert lay["rows"] == 2 + 1                       # two views + dream strip
    assert lay["cols"] == 1 + 3                       # dream column + three attempts
    assert _size(root / "contact.png") == (lay["cols"] * tr.TILE,
                                           tr.HEADER + lay["rows"] * tr.TILE)
    img = np.asarray(Image.open(root / "contact.png").convert("RGB"))
    # Dream column, iso row, holds the LATEST iso dream (green), back row the back dream.
    y_iso, y_back = tr.HEADER + tr.TILE // 2, tr.HEADER + tr.TILE + tr.TILE // 2
    assert tuple(img[y_iso, tr.TILE // 2]) == (30, 200, 30)
    assert tuple(img[y_back, tr.TILE // 2]) == (30, 30, 200)
    # Strip row: first cell = superseded iso v0, dimmed towards white.
    y_strip = tr.HEADER + 2 * tr.TILE + tr.TILE // 2
    r, g, b = img[y_strip, tr.TILE // 2]
    assert r > 200 and g > 100 and b > 100           # 200,30,30 blended with white
    assert tuple(img[y_strip, tr.TILE + tr.TILE // 2]) == (30, 200, 30)
    # Best column (attempt 1 -> column 2) is boxed in red.
    assert tuple(img[tr.HEADER + 5, 2 * tr.TILE + 1]) == (220, 0, 0)


def test_contact_without_dreams_or_renders(tmp_path):
    t = tr.Trace(tmp_path / "x")
    a = make_attempt(HUT_PROGRAM)
    t.attempt(a)
    t.close(_result([a], {}, 0))
    lay = tr.contact_layout(tmp_path / "x")
    assert lay["rows"] == 1 and lay["cols"] == 2 and lay["views"] == []
    assert _size(tmp_path / "x" / "contact.png") == (2 * tr.TILE, tr.HEADER + tr.TILE)


def test_duplicate_idx_attempts_kept(tmp_path):
    t = tr.Trace(tmp_path / "x")
    a, b = make_attempt(HUT_PROGRAM, idx=0), make_attempt(DEMO_PROGRAM, idx=0)
    t.attempt(a)
    t.attempt(b)
    t.attempt(a)                                      # same object: same dir
    names = sorted(p.name for p in (tmp_path / "x" / "iters").iterdir())
    assert names == ["00_generate", "00_generate~1"]


def test_asset_is_content_addressed(tmp_path):
    t = tr.Trace(tmp_path / "x")
    png = solid_png((1, 2, 3))
    rel = t.asset(png)
    assert rel == t.asset(png) and rel.startswith("assets/") and rel.endswith(".png")
    assert (tmp_path / "x" / rel).read_bytes() == png


def test_dream_view_mapping():
    assert tr.dream_view("iso") == "iso"
    assert tr.dream_view("back") == "iso_back"
    assert tr.dream_view("interior") == "cutaway"      # dreams.DREAMS preset (or None)
    assert tr.dream_view("mood") is None
    assert tr.dream_view("custom", {"custom": "top"}) == "top"


def test_arm_contact_sheet_stacks(tmp_path):
    roots = []
    for i in range(3):
        root = tmp_path / "builds" / "arm" / f"{i:03d}"
        _scenario(root)
        roots.append(root)
    out = tr.arm_contact_sheet(roots, tmp_path / "contact_arm.png", k=2)
    w, h = _size(out)
    one_w, one_h = _size(roots[0] / "contact.png")
    assert (w, h) == (one_w, 2 * (tr.TITLE + one_h))
    empty = tr.arm_contact_sheet([], tmp_path / "none.png")
    assert empty.is_file()


def test_arm_contact_sheet_draws_missing_contact(tmp_path):
    root = tmp_path / "000"
    _scenario(root, contact=False)
    assert not (root / "contact.png").exists()
    tr.arm_contact_sheet([root], tmp_path / "c.png")
    assert (root / "contact.png").is_file()


def test_variants_sheet_two_arms_three_prompts(tmp_path):
    run = tmp_path / "run"
    for arm, with_dreams in (("dream", True), ("classic", False)):
        for i in range(3):
            root = run / "builds" / arm / f"{i:03d}"
            t = tr.Trace(root)
            a = make_attempt(HUT_PROGRAM)
            a.renders = _renders(a)
            dreams = {}
            if with_dreams:
                d = DreamVersion("iso", 0, solid_png((200, 30, 30)), "p", "fake")
                t.dream(d)
                dreams = {"iso": [d]}
            t.attempt(a)
            res = _result([a], dreams, 0)
            res.description = f"prompt number {i}"
            t.close(res)
    out = tr.variants_sheet(run, ["dream", "classic"], k=12)
    assert out == run / "variants.png"
    assert _size(out) == (tr.LABEL_W + 2 * 2 * tr.TILE, tr.TITLE + 3 * tr.TILE)
    img = np.asarray(Image.open(out).convert("RGB"))
    y = tr.TITLE + tr.TILE // 2
    assert tuple(img[y, tr.LABEL_W + tr.TILE // 2]) == (200, 30, 30)       # dream arm: dream
    assert tuple(img[y, tr.LABEL_W + 2 * tr.TILE + tr.TILE // 2]) == (255, 255, 255)  # none
    final = img[y, tr.LABEL_W + 2 * tr.TILE + tr.TILE + tr.TILE // 2]
    assert tuple(final) != (255, 255, 255)                                # classic: render
    small = tr.variants_sheet(run, ["dream"], k=2, out=tmp_path / "v2.png")
    assert _size(small) == (tr.LABEL_W + 2 * tr.TILE, tr.TITLE + 2 * tr.TILE)


def test_variants_sheet_missing_arm_dir(tmp_path):
    out = tr.variants_sheet(tmp_path, ["nothing"], k=4)
    assert _size(out) == (tr.LABEL_W + 2 * tr.TILE, tr.TITLE + tr.TILE)


def test_close_survives_contact_failure(tmp_path, monkeypatch):
    def boom(root, out=None):
        raise RuntimeError("draw failed")
    monkeypatch.setattr(tr, "contact_sheet", boom)
    with pytest.warns(UserWarning, match="contact sheet failed"):
        _scenario(tmp_path / "b")
    kinds = [json.loads(x)["kind"] for x in (tmp_path / "b" / "events.jsonl").read_text()
             .splitlines()]
    assert "contact_error" in kinds
    assert (tmp_path / "b" / "result.json").is_file()


def test_run_episode_with_real_trace(tmp_path):
    """End to end through the landed Episode (WP2), on the fakes."""
    episode = pytest.importorskip("blockgen.agentic.episode")

    def method(ep, n: int = 2):
        ep.dream("iso")
        for i in range(n):
            stage = "generate" if i == 0 else f"match{i - 1}"
            ep.try_(stage, ep.ask(f"Build: {ep.prompt}", stage=stage))

    kit = make_fake_kit(replies=[HUT_PROGRAM, DEMO_PROGRAM], dreams=["iso"], method=method)
    root = tmp_path / "builds" / "fake" / "000"
    res = episode.run_episode(kit, "a small hut", trace=tr.Trace(root))
    assert res.trace_dir == str(root) and not res.truncated, res.error
    assert (root / "dreams" / "iso_v0.png").is_file()
    assert sorted(p.name for p in (root / "iters").iterdir()) == ["00_generate", "01_match0"]
    recs = [json.loads(x) for x in (root / "messages.jsonl").read_text().splitlines()]
    assert len(recs) == 2
    for r in recs:
        for m in r["messages"]:
            for p in m["parts"]:
                if "image" in p:
                    assert (root / p["image"]).is_file()
    lay = tr.contact_layout(root)
    assert _size(root / "contact.png") == (lay["width"], lay["height"])
    assert json.loads((root / "result.json").read_text())["description"] == "a small hut"


def test_attempt_ids_not_reused_after_gc(tmp_path):
    """A discarded attempt's id() reused by a new object must not alias its dir."""
    import gc

    t = tr.Trace(tmp_path / "x")
    for i in range(20):
        a = make_attempt(HUT_PROGRAM, idx=i, stage="generate")
        t.attempt(a)
        del a
        gc.collect()
    names = sorted(p.name for p in (tmp_path / "x" / "iters").iterdir())
    assert names == [f"{i:02d}_generate" for i in range(20)]


def test_best_dir_found_by_idx_not_position(tmp_path):
    """best_idx is Attempt.idx; attempts whose idx differs from their list
    position (adopted rounds) must still box the right column."""
    t = tr.Trace(tmp_path / "x", contact=False)
    a = make_attempt(HUT_PROGRAM, idx=5, stage="generate")
    b = make_attempt(DEMO_PROGRAM, idx=7, stage="match0")
    for x in (a, b):
        t.attempt(x)
    res = _result([a, b], {}, 1)
    res.best_idx = 7
    t.close(res)
    blob = json.loads((tmp_path / "x" / "result.json").read_text())
    assert blob["best_dir"] == "07_match0"


def test_for_kit_maps_inline_dreams_to_their_views(tmp_path):
    """An inline dream key (not a DREAMS preset) takes its view from the recipe."""
    from types import SimpleNamespace

    kit = SimpleNamespace(dreams=[core.DreamSpec("rear", "t", view="iso_left"),
                                  core.DreamSpec("mood", "t", view=None)])
    t = tr.Trace.for_kit(tmp_path / "x", kit)
    assert t._view_of("rear") == "iso_left" and t._view_of("mood") is None
    t2 = tr.Trace.for_kit(tmp_path / "y", kit, dream_views={"rear": "iso"})
    assert t2._view_of("rear") == "iso"
