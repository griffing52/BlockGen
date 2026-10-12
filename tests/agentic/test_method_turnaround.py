"""The ``turnaround`` method end to end, offline.

WP-C acceptance: the four recipe dreams are filled from ONE ``sheet`` generation
(one image charge); a bad sheet falls back to per-view edits; ``per_view`` mode
makes exactly today's multi-view requests; the cyclic assignment recovers a
rotated sheet once, after the first build; the recipe runs under --mock.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import pytest

import blockgen.agentic.recipes.turnaround  # noqa: F401  (registers the method)
from blockgen.agentic import dreams as dreams_mod
from blockgen.agentic import scoring, sheets
from blockgen.agentic.core import ImageResult, get_view
from blockgen.agentic.episode import run_episode
from blockgen.agentic.imagegen import GRID_HINT, MockImageGen
from blockgen.agentic.providers import encode_png
from blockgen.agentic.recipes.dream import dream
from blockgen.agentic.recipes.turnaround import SHEET_KEY, turnaround
from blockgen.agentic.render import FlatRenderer

from .fakes import RecordingTrace, ScriptedLLM, make_fake_kit
from .test_sheets import ASYM, ORDER, asym_world

KEY_OF = {"iso": "iso", "iso_left": "left", "iso_back": "back", "iso_right": "right"}


def turnaround_dreams():
    """The turnaround recipe's dreams: iso without view_search, back, left, right."""
    iso = dataclasses.replace(dreams_mod.DREAMS["iso"], view_search=())
    return [iso] + [dreams_mod.DREAMS[k] for k in ("back", "left", "right")]


class SheetGen:
    """Draws the ASYM build with the flat renderer (mock:render on a one-house pool),
    priced so the ledger counts images. ``rotate`` draws a sheet whose panels are
    rotated by k views (the image model's camera is off); ``bad`` returns a single
    full-colour picture for sheet requests (an unsplittable sheet)."""

    name = "sheet_fake"
    spends = False
    billing = "mock"

    def __init__(self, *, rotate: int = 0, bad: bool = False, usd: float = 0.01) -> None:
        self.inner = MockImageGen("render", renderer="flat", px=128,
                                  structures=[asym_world().to_structure()])
        self.rotate, self.bad, self.usd = rotate, bad, usd
        self.requests = []

    def generate(self, req):
        self.requests.append(req)
        if req.view.startswith(GRID_HINT):
            if self.bad:
                png = encode_png(np.full((128, 128, 3), (90, 60, 30), np.uint8))
                return ImageResult([png], model="sheet-bad", cost_usd=self.usd, priced=True)
            views = req.view[len(GRID_HINT):].split(",")
            k = self.rotate
            req = dataclasses.replace(req, view=GRID_HINT + ",".join(views[k:] + views[:k]))
        res = self.inner.generate(req)
        return ImageResult(res.images, model="sheet-fake", cost_usd=self.usd, priced=True)

    def estimate_usd(self, req):
        return self.usd

    def supports_refs(self):
        return True

    def describe(self):
        return "sheet_fake"


def improve_accept(prev, cand):
    if cand.blocks == 0:
        return False, "empty"
    if prev is None or prev.score is None:
        return True, "first"
    ok = cand.score.value > prev.score.value
    return ok, "improved" if ok else "not better"


def run(params=None, *, image=None, rules=(), method=turnaround, build=ASYM, dreams=None,
        budget=None):
    llm = ScriptedLLM([build], rules=list(rules))
    image = image or SheetGen()
    kit = make_fake_kit(llm=llm, method=method, dreams=dreams or turnaround_dreams(),
                        image=image, renderer=FlatRenderer(px=128), accept=improve_accept,
                        scorers={"silhouette": (scoring.silhouette, 1.0)},
                        spec_kw={"params": dict(params or {}), "select": "best",
                                 "budget": budget or {}})
    trace = RecordingTrace()
    res = run_episode(kit, "a long hall with a corner tower", trace=trace)
    return res, kit, trace, image


def events(trace, kind):
    return [d for k, d in trace.events if k == kind]


def digest(png: bytes) -> str:
    return hashlib.sha256(png).hexdigest()


# ===========================================================================
# sheet mode
# ===========================================================================
def test_one_sheet_image_fills_the_four_recipe_dreams():
    res, kit, trace, image = run({"assign": "fixed", "max_iters": 1})
    assert not res.truncated, res.error
    assert len(image.requests) == 1
    req = image.requests[0]
    assert req.view == GRID_HINT + ",".join(ORDER)
    assert "turnaround sheet" in req.prompt.lower()
    assert sum(1 for c in res.charges if c.role == "image") == 1
    sheet = res.dreams[SHEET_KEY][-1]
    for key in KEY_OF.values():
        dv = res.dreams[key][-1]
        assert dv.generator == f"split:sheet_v{sheet.version}"
        assert len(res.dreams[key]) == 1
    # each tile is the matching panel: the iso target looks like the iso render
    r = FlatRenderer(px=128).render(asym_world().to_structure(), [get_view("iso")])[0]
    assert scoring.silhouette_iou(res.dreams["iso"][-1].png, encode_png(r)) > 0.9
    assert not events(trace, "turnaround_fallback")
    assert [a.stage for a in res.attempts][0] == "generate"


def test_the_sheet_is_not_a_scored_dream():
    res, kit, trace, _ = run({"assign": "fixed", "max_iters": 0})
    assert SHEET_KEY not in [d.key for d in kit.dreams]
    att = res.attempts[0]
    per = att.score.per_view["silhouette"]          # "<key>@<view>"
    assert {k.split("@")[0] for k in per} == set(KEY_OF.values())


def test_the_builder_sees_the_four_tiles_not_the_sheet():
    res, kit, trace, _ = run({"assign": "fixed", "max_iters": 0})
    first = trace.requests[0]["messages"]
    imgs = [digest(p.image_png) for m in first for p in m.parts if p.kind == "image"]
    tiles = {digest(res.dreams[k][-1].png) for k in KEY_OF.values()}
    assert tiles <= set(imgs)
    assert digest(res.dreams[SHEET_KEY][-1].png) not in imgs


def test_bad_sheet_falls_back_to_per_view_dreams():
    res, kit, trace, image = run({"assign": "cyclic", "max_iters": 0},
                                 image=SheetGen(bad=True))
    ev = events(trace, "turnaround_fallback")
    assert ev and ev[0]["fallback"] is True
    assert [r.view for r in image.requests] == [GRID_HINT + ",".join(ORDER), "iso",
                                                 "iso_back", "iso_left", "iso_right"]
    for key in KEY_OF.values():
        assert not res.dreams[key][-1].generator.startswith("split:")
    assert not events(trace, "turnaround_assign")       # no tiles, nothing to assign


def test_fallback_keeps_the_sheet_critics_rewritten_description():
    verdicts = iter([{"consistent": False, "score": 2, "revised_prompt": "one stone hall"},
                     {"consistent": False, "score": 2, "revised_prompt": "one stone hall"}])
    rules = [(sheets.SHEET_REVIEW_TAG, lambda _m: json.dumps(next(verdicts)))]
    res, kit, trace, image = run({"refine": 1, "assign": "fixed", "max_iters": 0},
                                 image=SheetGen(bad=True), rules=rules,
                                 budget={"max_images": 6})
    assert events(trace, "turnaround_fallback")
    per_view = [r for r in image.requests if not r.view.startswith(GRID_HINT)]
    assert len(per_view) == 4
    assert all("one stone hall" in r.prompt for r in per_view)


def test_bad_sheet_without_fallback_uses_the_tiles_anyway():
    res, kit, trace, image = run({"fallback_per_view": False, "assign": "fixed",
                                  "max_iters": 0}, image=SheetGen(bad=True))
    assert len(image.requests) == 1
    assert events(trace, "turnaround_fallback")[0]["fallback"] is False
    assert res.dreams["iso"][-1].generator.startswith("split:")


def test_fallback_that_does_not_fit_the_image_cap_keeps_the_tiles():
    res, kit, trace, image = run({"max_iters": 0, "assign": "fixed"},
                                 image=SheetGen(bad=True), budget={"max_images": 3})
    assert not res.truncated, res.error
    assert len(image.requests) == 1
    assert events(trace, "turnaround_fallback_budget")
    assert res.dreams["iso"][-1].generator.startswith("split:")


# ===========================================================================
# assignment
# ===========================================================================
def test_cyclic_assignment_recovers_a_rotated_sheet_once():
    res, kit, trace, image = run({"assign": "cyclic", "max_iters": 2},
                                 image=SheetGen(rotate=1))
    ev = events(trace, "turnaround_assign")
    assert len(ev) == 1
    drawn = list(ORDER[1:] + ORDER[:1])
    assert ev[0]["changed"] is True and ev[0]["chosen"] == drawn
    assert ev[0]["value"] > ev[0]["identity"]
    # every key got a new version, and it is now the right panel
    renders = FlatRenderer(px=128)
    s = asym_world().to_structure()
    for view, key in KEY_OF.items():
        assert len(res.dreams[key]) == 2
        assert res.dreams[key][-1].generator.endswith(":assigned")
        ref = encode_png(renders.render(s, [get_view(view)])[0])
        assert scoring.silhouette_iou(res.dreams[key][-1].png, ref) > 0.9
    # scores were re-anchored to the new targets
    did = res.attempts[-1].score.dream_id
    assert all(a.score.dream_id == did for a in res.attempts if a.score and a.blocks)


def test_cyclic_assignment_keeps_a_correct_sheet():
    res, kit, trace, _ = run({"assign": "cyclic", "max_iters": 0})
    ev = events(trace, "turnaround_assign")
    assert len(ev) == 1 and ev[0]["changed"] is False
    assert all(len(res.dreams[k]) == 1 for k in KEY_OF.values())


def test_fixed_assignment_trusts_the_prompt_order():
    res, kit, trace, _ = run({"assign": "fixed", "max_iters": 0}, image=SheetGen(rotate=1))
    assert not events(trace, "turnaround_assign")


def test_free_assignment_searches_all_views_without_touching_the_shared_kit():
    res, kit, trace, _ = run({"assign": "free", "max_iters": 0})
    assert events(trace, "turnaround_free")
    assert all(not d.view_search for d in kit.dreams)     # the shared kit is untouched
    per = res.attempts[0].score.per_view["silhouette"]   # "<key>@<view>"
    assert {k for k in per if k.startswith("back@")} == {f"back@{v}" for v in ORDER}


def test_free_assignment_keeps_dream_spec_in_step_with_the_episode_kit():
    """dream_spec() (turns, refine) must see the same search views the scorers use."""
    from blockgen.agentic.episode import Episode
    from blockgen.agentic.recipes.turnaround import free_views
    kit = make_fake_kit(llm=ScriptedLLM([ASYM]), method=turnaround,
                        dreams=turnaround_dreams(), image=SheetGen())
    ep = Episode("x", kit, trace=RecordingTrace())
    free_views(ep, ORDER)
    for ds in ep.kit.dreams:
        assert ep.dream_spec(ds.key) == ds
        assert set((ds.view,) + ds.view_search) == set(ORDER)
    assert all(not d.view_search for d in kit.dreams)


# ===========================================================================
# refine, sheet_ref, per_view
# ===========================================================================
def test_refine_regenerates_a_rejected_sheet_with_a_new_seed():
    verdicts = iter([{"consistent": False, "score": 3, "issues": ["two houses"],
                      "revised_prompt": "one hall"},
                     {"consistent": True, "score": 9}])
    rules = [(sheets.SHEET_REVIEW_TAG, lambda _m: json.dumps(next(verdicts)))]
    res, kit, trace, image = run({"refine": 2, "assign": "fixed", "max_iters": 0},
                                 rules=rules)
    assert len(image.requests) == 2
    assert image.requests[0].seed != image.requests[1].seed
    assert "one hall" in image.requests[1].prompt
    assert len(res.dreams[SHEET_KEY]) == 2
    assert res.dreams["iso"][-1].generator == "split:sheet_v1"


def test_sheet_ref_iso_dreams_the_iso_view_first():
    res, kit, trace, image = run({"sheet_ref": "iso", "assign": "fixed", "max_iters": 0})
    assert [r.view for r in image.requests] == ["iso", GRID_HINT + ",".join(ORDER)]
    assert image.requests[1].refs == (res.dreams["iso"][0].png,)
    assert res.dreams["iso"][-1].generator.startswith("split:")


def test_per_view_mode_makes_todays_multiview_requests():
    def key(r):
        return (r.prompt, tuple(digest(x) for x in r.refs), r.seed, r.view, r.size)
    _, _, _, a = run({"mode": "per_view", "max_iters": 0})
    _, _, _, b = run({"max_iters": 0}, method=dream)
    assert [key(r) for r in a.requests] == [key(r) for r in b.requests]
    assert all(not r.view.startswith(GRID_HINT) for r in a.requests)


# ===========================================================================
# validation and the shipped recipe
# ===========================================================================
@pytest.mark.parametrize("params,needle", [
    ({"mode": "grid"}, "mode"),
    ({"assign": "best"}, "assign"),
    ({"sheet_ref": "back"}, "sheet_ref"),
    ({"order": ["iso", "iso", "iso_back", "iso_right"]}, "distinct"),
    ({"order": ["iso", "nope", "iso_back", "iso_right"]}, "unknown view"),
    ({"order": ["iso", "top", "iso_back", "iso_right"]}, "no recipe dream"),
    ({"build_mode": "replace"}, "build_mode"),
    ({"feedback": "loud"}, "feedback"),
])
def test_invalid_params_raise_before_any_call(params, needle):
    res, kit, trace, image = run(params)
    assert res.truncated == "error" and needle in res.error
    assert not image.requests and not trace.requests


def test_needs_an_image_generator():
    llm = ScriptedLLM([ASYM])
    kit = make_fake_kit(llm=llm, method=turnaround, dreams=turnaround_dreams(), image=None)
    res = run_episode(kit, "x", trace=RecordingTrace())
    assert res.truncated == "error" and "image generator" in res.error


def test_registered_without_dream_prefetch():
    from blockgen.agentic import core
    assert core.lookup("method", "turnaround").load() is turnaround
    assert core.meta("method", "turnaround").get("prefetch_dreams") is False


def test_shipped_recipe_runs_offline_under_mock(tmp_path):
    from blockgen.agentic.kit import build_kit
    from blockgen.agentic.recipe import load_recipe
    spec = load_recipe("turnaround")
    assert spec.method == "turnaround" and spec.dream_style == "render_match"
    assert [d if isinstance(d, str) else d.get("key") for d in spec.dreams] == \
        ["iso", "back", "left", "right"]
    kit = build_kit(spec, mock=True, cache_mode="off")
    trace = RecordingTrace()
    res = run_episode(kit, "a small stone tower", trace=trace)
    assert not res.truncated or res.truncated == "budget", res.error
    assert len(res.dreams[SHEET_KEY]) == 1
    assert all(res.dreams[k][-1].generator.startswith("split:") for k in KEY_OF.values())
    assert res.attempts and res.attempts[0].blocks > 0
