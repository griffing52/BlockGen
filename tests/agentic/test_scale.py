"""Player scale and text planning (2026-10-03).

* the ``scale`` context (system text, canvas note, strict), its dream cue and the
  plan / blueprint lines, all switched by the recipe's contexts only;
* byte-identity where nothing was asked for: no-scale dreams and the no-dream
  plan turn, and classic's legacy prompts (scale is appended, nothing else moves);
* the dream-reading plan turn (``dream-plan``) and plan_source ``text``
  (``blueprint-text-plan``): turns, mock replies, traces;
* the scale heuristic (``scale.scale_stats``), its scorer and the method metrics;
* the blueprint room minimum (5 x 5 rects) and the shipped recipes / battery.
"""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

import blockgen.agentic.recipes.blueprint  # noqa: F401  (registers the method)
from blockgen.agentic import dreams as dreams_mod
from blockgen.agentic import scale as scale_mod
from blockgen.agentic import turns, turns_staged
from blockgen.agentic.blueprint import (MIN_ROOM_SIDE, Blueprint, Rect, Room,
                                        mock_blueprint)
from blockgen.agentic.core import MethodSpec, lookup
from blockgen.agentic.episode import run_episode
from blockgen.agentic.kit import build_kit
from blockgen.agentic.providers import Message, Part, text_message
from blockgen.agentic.providers_cli import AutoMockProvider, mock_brief, mock_plan
from blockgen.agentic.recipe import load_battery, load_recipe
from blockgen.agentic.recipes.blueprint import blueprint
from blockgen.agentic.recipes.classic import legacy_prompt_mode, legacy_system
from blockgen.agentic.trace import Trace

from .fakes import RecordingTrace, make_fake_kit

SIZE = (32, 32, 32)


def fake_ep(contexts=("palette",), dreams=(), size=SIZE, prompt="a hut"):
    spec = MethodSpec("x", contexts=list(contexts), canvas=size)
    return NS(size=size, prompt=prompt, spec=spec, notes={},
              kit=NS(dreams=list(dreams)), latest_dream=lambda k: None, dream_desc={})


# ===========================================================================
# the context and its switches
# ===========================================================================
def test_scale_context_is_system_text_with_canvas_note_and_strict():
    fn = lookup("context", "scale").load()
    ep = fake_ep(contexts=("palette", "scale"))
    parts = fn(ep)
    assert not parts.user_parts
    t = parts.system_text
    assert "2 blocks tall" in t and "1 wide x 2 tall" in t and "at least 3 blocks" in t
    assert "32 x 32 x 32 canvas" in t and "rather than shrinking doors" in t
    assert scale_mod.STRICT_NOTE not in t
    assert ep.notes["scale"]["dream_cue"] is True
    strict = fn(ep, strict=True, canvas=False).system_text
    assert scale_mod.STRICT_NOTE in strict and "canvas" not in strict.split("\n")[-2]
    assert "32 x 32 x 32" not in strict


def test_canvas_note_tracks_canvas_size():
    assert "5 storeys" in scale_mod.canvas_note((32, 32, 32))
    assert "low at player scale" in scale_mod.canvas_note((16, 8, 16))
    assert "narrow at player scale" in scale_mod.canvas_note((6, 24, 12))


def test_canvas_note_never_offers_more_storeys_than_the_method_allows():
    assert "the 2 storeys this method allows" in scale_mod.canvas_note(SIZE, 2)
    assert "5 storeys" not in scale_mod.canvas_note(SIZE, 2)
    assert scale_mod.canvas_note(SIZE, 9) == scale_mod.canvas_note(SIZE)
    assert scale_mod.method_max_storeys(load_recipe("blueprint")) == 2
    assert scale_mod.method_max_storeys(load_recipe("floors")) == 2
    assert scale_mod.method_max_storeys(load_recipe("dream")) is None
    assert scale_mod.method_max_storeys(MethodSpec("a", method="blueprint",
                                                   params={"max_storeys": 1})) == 1
    fn = lookup("context", "scale").load()
    ep = fake_ep(contexts=("scale",))
    ep.spec = load_recipe("blueprint")
    assert "the 2 storeys this method allows" in fn(ep).system_text
    ep.spec = load_recipe("dream")
    assert "at most about 5 storeys" in fn(ep).system_text


def test_rules_give_one_size_convention_and_no_conflicting_anchors():
    rules = scale_mod.RULES
    assert "7-12" not in rules and "10-16 blocks per side" in rules
    assert "open floor inside the walls" in rules
    assert "2 blocks of air above every" in rules and "3 blocks of headroom" not in rules


def test_blueprint_line_matches_the_repair_minimum():
    line = scale_mod.blueprint_line()
    assert f"at least {MIN_ROOM_SIDE} wide including walls" in line
    assert "3 wide (1 of floor)" not in line and "no narrower corridors" in line
    assert "a player is 2 blocks tall" not in line           # already in the system RULES
    # what the line warns about: a 3-wide corridor rect is dropped by the repair
    bp = mock_blueprint(SIZE)
    b = Blueprint.from_dict(bp.to_dict())
    b.rooms = [r for r in b.rooms if r.name != "bedroom"] + [
        Room("corridor", "corridor", 0, Rect(6, 8, 8, 18), "")]
    assert "corridor" not in [r.name for r in b.repaired(SIZE).rooms]


def test_blueprint_json_turn_states_the_unit_once():
    t = turns_staged.blueprint_json(staged_ep(("scale",)), max_rooms=4, max_storeys=2).text
    assert t.count("all units are blocks") == 1


def test_scale_params_read_from_recipe_contexts():
    assert scale_mod.scale_params(MethodSpec("a")) is None
    assert scale_mod.scale_params(MethodSpec("a", contexts=["palette", "scale"])) == {}
    on = MethodSpec("a", contexts=[{"scale": {"strict": True}}])
    assert scale_mod.scale_params(on) == {"strict": True}
    assert scale_mod.dream_cue(NS(spec=on)) == scale_mod.DREAM_CUE
    off = MethodSpec("a", contexts=[{"scale": {"dream_cue": False}}])
    assert scale_mod.scale_active(NS(spec=off)) and scale_mod.dream_cue(NS(spec=off)) == ""


def test_dream_cue_rides_on_the_style_and_is_absent_without_scale():
    ds = dreams_mod.DREAMS["iso"]
    plain = dreams_mod.render_prompt(ds, fake_ep())
    want = " ".join((f"{dreams_mod.DEFAULT_STYLE} " + ds.template.split("{style} ", 1)[1])
                    .format(desc="a hut", W=32, H=32, D=32).split())
    assert plain == want                                    # byte-identical to before
    cued = dreams_mod.render_prompt(ds, fake_ep(contexts=("palette", "scale")))
    assert cued == plain.replace("no frame.", "no frame. " + scale_mod.DREAM_CUE, 1)
    assert "people" in cued and "player figure" not in cued
    off = dreams_mod.render_prompt(ds, fake_ep(contexts=({"scale": {"dream_cue": False}},)))
    assert off == plain
    room = dreams_mod.render_prompt(dreams_mod.DREAMS["interior"],
                                    fake_ep(contexts=("scale",)))
    assert scale_mod.DREAM_CUE in room


# ===========================================================================
# plan turns
# ===========================================================================
CLASSIC_PLAN = """\
Build request: a hut

Before writing anything executable, plan the build. Answer in at most 12 short lines:
- Overall footprint and height in blocks (must fit 32x32x32).
- The material palette: which block for foundation, walls, trim, roof, glazing.
- The parts list, in build order, each with its coordinate range.
- Two details that will make it look deliberate rather than generic.
Plain text only."""


def test_plan_turn_unchanged_without_dreams_or_scale():
    assert turns.plan(fake_ep()).text == CLASSIC_PLAN


def test_plan_turn_scale_line_only_with_scale():
    t = turns.plan(fake_ep(contexts=("palette", "scale"))).text
    assert t.replace(scale_mod.plan_line() + "\n", "") == CLASSIC_PLAN


def test_plan_turn_reads_the_dream_in_blocks():
    png = b"\x89PNG-fake"
    ds = dreams_mod.DREAMS["iso"]
    ep = fake_ep(contexts=("scale",), dreams=[ds])
    ep.latest_dream = lambda k: NS(png=png) if k == "iso" else None
    msg = turns.plan(ep)
    kinds = [p.kind for p in msg.parts]
    assert kinds[:2] == ["text", "image"] and msg.parts[1].image_png == png
    t = msg.text
    assert "TARGET" in t and "IN BLOCKS" in t and "x a..b, y c..d, z e..f" in t
    assert turns.PLAN_HEAD in t and "plan the build" in t
    assert scale_mod.plan_line(dream=True) in t and t.rstrip().endswith("Plain text only.")
    assert "not from how big it looks" not in t          # the TARGET is the scale
    assert "TARGET's proportions" in t and "count its doors and storeys" in t


# ===========================================================================
# classic legacy prompts + scale
# ===========================================================================
def test_classic_matched_keeps_legacy_prompts_with_scale_appended():
    kit = build_kit(load_recipe("classic-matched"), mock=True, cache_mode="off")
    ep = NS(kit=kit, spec=kit.spec, size=SIZE, notes={})
    assert legacy_prompt_mode(ep)
    sys_scaled = legacy_system(ep).text
    kit_plain = build_kit(load_recipe("classic-matched", {"contexts": ["palette"]}),
                          mock=True, cache_mode="off")
    plain = legacy_system(NS(kit=kit_plain, spec=kit_plain.spec, size=SIZE, notes={})).text
    assert sys_scaled.startswith(plain + "\n\n")
    assert sys_scaled[len(plain) + 2:] == scale_mod.scale_rules(SIZE)


def test_classic_full_stays_scale_free_for_parity():
    spec = load_recipe("classic-full")
    assert scale_mod.scale_params(spec) is None
    assert scale_mod.scale_params(load_recipe("dsl-classic")) is None


# ===========================================================================
# staged planner turns
# ===========================================================================
def staged_ep(contexts=("palette",)):
    ep = fake_ep(contexts=contexts)
    return ep


def test_blueprint_turns_scale_paragraph_only_with_scale():
    plain = turns_staged.blueprint_json(staged_ep(), max_rooms=4, max_storeys=2).text
    scaled = turns_staged.blueprint_json(staged_ep(("palette", "scale")), max_rooms=4,
                                         max_storeys=2).text
    assert "Player scale" not in plain and scale_mod.blueprint_line() in scaled
    assert scaled.replace(scale_mod.blueprint_line() + "\n\n", "") == plain
    assert "at\nleast 5 x 5" in plain or "least 5 x 5" in plain    # the repair's hard minimum
    fix = turns_staged.blueprint_fix(staged_ep(("scale",)), {"footprint": [0, 0, 3, 3]},
                                     ["too small"], brief="Rooms: one hall.")
    assert scale_mod.blueprint_line() in fix.text and "Rooms: one hall." in fix.text


def test_design_brief_turn_is_prose_and_json_turn_transcribes_it():
    ep = staged_ep(("scale",))
    brief = turns_staged.design_brief(ep, max_rooms=4, max_storeys=2).text
    assert brief.startswith(turns_staged.BRIEF_TAG)
    assert turns_staged.BLUEPRINT_TAG not in brief and turns.JSON_RULE not in brief
    for head in ("Footprint:", "Storeys:", "Rooms:", "Circulation:", "Openings:"):
        assert head in brief
    assert "how a player walks from the front door" in brief
    assert scale_mod.blueprint_line() in brief
    js = turns_staged.blueprint_json(ep, max_rooms=4, max_storeys=2,
                                     brief="Footprint: 12 x 10.").text
    assert "Your design brief:\nFootprint: 12 x 10." in js and "Transcribe" in js
    assert turns_staged.BRIEF_TAG not in js and js.endswith(turns.JSON_RULE)


# ===========================================================================
# mock LLM
# ===========================================================================
def _decide(text, system="You are a master Minecraft builder.\nOutput only command lines."):
    return AutoMockProvider()._decide([text_message("system", system),
                                       text_message("user", text)])


def test_mock_answers_plan_and_brief_turns_with_prose():
    kind, text = _decide(CLASSIC_PLAN)
    assert kind == "plan" and "Footprint" in text and "fill " not in text
    ep = staged_ep(("scale",))
    kind, text = _decide(turns_staged.design_brief(ep, max_rooms=4, max_storeys=2).text)
    assert kind == "staged:brief" and "Circulation:" in text and "{" not in text
    js = turns_staged.blueprint_json(ep, max_rooms=4, max_storeys=2, brief=mock_brief(""))
    kind, text = _decide(js.text)
    assert kind == "staged:blueprint" and json.loads(text)["rooms"]
    assert mock_plan("TARGET iso\nmust fit 16x16x16").startswith("- TARGET")
    assert "x 2..13" in mock_plan("must fit 16x16x16")          # sized to the canvas


# ===========================================================================
# methods end to end (offline)
# ===========================================================================
def test_plan_source_text_runs_offline_and_keeps_the_brief(tmp_path):
    kit = make_fake_kit(llm=AutoMockProvider(), method=blueprint, dreams=["iso"], mock=True,
                        spec_kw={"params": {"plan_source": "text", "stop_at": 2.0},
                                 "select": "best", "contexts": ["scale"]})
    trace = Trace(tmp_path)
    res = run_episode(kit, "a cosy cottage with a kitchen", trace=trace)
    assert not res.truncated, res.error
    stages = [json.loads(l)["stage"] for l in (tmp_path / "messages.jsonl").open()]
    assert stages[:2] == ["plan_brief", "plan"]
    brief = (tmp_path / "plan_brief.txt").read_text()
    assert "Circulation:" in brief and res.plan_text == brief
    first_json = [json.loads(l) for l in (tmp_path / "messages.jsonl").open()][1]
    user = first_json["messages"][-1]["parts"][0]["text"]
    assert "Your design brief:\n" + brief in user
    assert "roof" in [a.stage for a in res.attempts]


def test_plan_source_text_needs_no_image_generator():
    kit = make_fake_kit(llm=AutoMockProvider(), method=blueprint, image=None, mock=True,
                        spec_kw={"params": {"plan_source": "text", "exterior_dream": False,
                                            "interior_dreams": False, "stop_at": 2.0}})
    res = run_episode(kit, "a cottage", trace=RecordingTrace())
    assert not res.truncated, res.error


@pytest.mark.slow
def test_recipe_dream_plan_mock_plans_from_the_dream(tmp_path):
    kit = build_kit(load_recipe("dream-plan"), mock=True, cache_mode="off")
    res = run_episode(kit, "a brick house", trace=Trace(tmp_path))
    assert not res.truncated, res.error
    reqs = [json.loads(l) for l in (tmp_path / "messages.jsonl").open()]
    assert reqs[0]["stage"] == "plan" and reqs[0]["role"] == "planner"
    user = reqs[0]["messages"][-1]
    assert any("image" in p for p in user["parts"])           # the dream is shown
    assert "IN BLOCKS" in user["parts"][-1]["text"]
    assert "Footprint" in str(reqs[0]["response"]) and "fill " not in str(reqs[0]["response"])
    assert (tmp_path / "plan.txt").read_text() == res.plan_text
    assert "Your plan:\n" in reqs[1]["messages"][-1]["parts"][-1]["text"]
    assert scale_mod.DREAM_CUE in res.dreams["iso"][0].prompt


@pytest.mark.slow
def test_recipe_blueprint_text_plan_mock():
    kit = build_kit(load_recipe("blueprint-text-plan"), mock=True, cache_mode="off")
    trace = RecordingTrace()
    res = run_episode(kit, "a cosy cottage", trace=trace)
    assert not res.truncated, res.error
    assert [r["stage"] for r in trace.requests][:2] == ["plan_brief", "plan"]
    assert res.plan_text and "roof" in [a.stage for a in res.attempts]


# ===========================================================================
# the heuristic, scorer and metrics
# ===========================================================================
def hut(ceiling_y=4, door_h=2):
    ids = np.zeros((9, 6, 9), int)
    ids[:, 0, :] = 4
    ids[0, 1:ceiling_y, :] = ids[8, 1:ceiling_y, :] = 1
    ids[:, 1:ceiling_y, 0] = ids[:, 1:ceiling_y, 8] = 1
    ids[:, ceiling_y, :] = 5
    ids[4, 1:1 + door_h, 8] = 64 if door_h == 2 else 0
    if door_h == 1:
        ids[4, 1, 8] = 0
    return ids


def test_scale_stats_reads_headroom_and_doorways():
    good = scale_mod.scale_stats(hut())
    assert good["scale_headroom2"] == 1.0 and good["scale_headroom3"] > 0.95
    assert good["scale_doorways"] == 1 and good["scale_low_openings"] == 0
    assert good["scale_door_blocks"] == 1 and good["scale_door_ok"] == 1.0
    low = scale_mod.scale_stats(hut(ceiling_y=3, door_h=1))
    assert low["scale_headroom3"] == 0.0
    assert low["scale_low_openings"] == 1 and low["scale_door_ok"] == 0.0
    empty = scale_mod.scale_stats(np.zeros((4, 4, 4), int))
    assert empty["scale_cells"] == 0 and empty["scale_headroom2"] is None


def test_unglazed_window_is_not_a_low_doorway():
    ids = hut()
    ids[2, 2, 8] = 0                       # a 1-tall hole one block above the floor
    st = scale_mod.scale_stats(ids)
    assert st["scale_low_openings"] == 0 and st["scale_doorways"] == 1


def gabled(ids):
    """``ids`` with a stepped gable (ridge along z, solid gable ends) on its roof."""
    X, H, Z = ids.shape
    top = max(y for y in range(H) if (ids[:, y, :] != 0).any())
    out = np.zeros((X, top + 2 + X // 2, Z), int)
    out[:, :H, :] = ids
    for k in range(X // 2 + 1):
        y = top + 1 + k
        out[k, y, :] = out[X - 1 - k, y, :] = 53
        out[k + 1:X - 1 - k, y, 0] = out[k + 1:X - 1 - k, y, Z - 1] = 1
    return out


def test_pitched_roof_attic_does_not_lower_headroom():
    flat = scale_mod.scale_stats(hut())
    roofed = scale_mod.scale_stats(gabled(hut()))
    assert roofed["scale_headroom3"] == flat["scale_headroom3"] == 1.0
    # a doll's house stays low under a gable: its level 2-high ceilings still count
    low = scale_mod.scale_stats(gabled(hut(ceiling_y=3)))
    assert low["scale_headroom3"] < 0.5


def corridor_house(back_door=True):
    ids = np.zeros((11, 6, 11), int)
    ids[:, 0, :] = 4
    ids[:, 4, :] = 5
    ids[0, 1:4, :] = ids[10, 1:4, :] = 1
    ids[:, 1:4, 0] = ids[:, 1:4, 10] = 1
    ids[4, 1:4, 1:7] = ids[6, 1:4, 1:7] = 1          # a 1-wide corridor x=5, z 1..6
    ids[5, 1:3, 10] = 64                              # front door
    if back_door:
        ids[5, 1:3, 0] = 64                           # back door at the corridor's end
    return ids


def test_corridor_is_not_a_row_of_doorways():
    st = scale_mod.scale_stats(corridor_house(back_door=False))
    assert st["scale_doorways"] == 1 and st["scale_low_openings"] == 0
    st = scale_mod.scale_stats(corridor_house())
    assert st["scale_doorways"] == 2 and st["scale_door_blocks"] == 2


def test_two_thick_wall_door_counts_once_and_crenels_not_at_all():
    ids = np.zeros((9, 6, 10), int)
    ids[:, 0, :] = 4
    ids[0, 1:4, :] = ids[8, 1:4, :] = 1
    ids[:, 1:4, 0] = 1
    ids[:, 1:4, 8:10] = 1                             # a front wall 2 thick
    ids[:, 4, :] = 5
    ids[4, 1:3, 8:10] = 0                             # one opening through it
    assert scale_mod.scale_stats(ids)["scale_doorways"] == 1
    cren = np.zeros((9, 7, 9), int)
    cren[:, :6, :] = hut()
    cren[0, 5, ::2] = cren[8, 5, ::2] = 1             # merlons on the roof edge
    assert scale_mod.scale_stats(cren)["scale_doorways"] == 1


def test_open_window_over_furniture_is_not_a_low_opening():
    ids = hut()
    ids[2, 2, 8] = 0                                  # an unglazed window
    ids[2, 1, 7] = 26                                 # a bed under it, inside
    st = scale_mod.scale_stats(ids)
    assert st["scale_low_openings"] == 0 and st["scale_door_ok"] == 1.0


def test_scale_scorer_sealed_box_scores_below_a_house_with_a_door():
    fn = lookup("scorer", "scale").load()
    sealed = hut()
    sealed[4, 1:3, 8] = 1
    assert scale_mod.scale_stats(sealed)["scale_door_ok"] is None    # metric: unmeasured
    as_att = lambda ids: NS(blocks=1, world=NS(to_structure=lambda crop=True: NS(block_ids=ids)))
    assert fn(NS(), as_att(sealed)).value == pytest.approx(0.5)
    assert fn(NS(), as_att(hut())).value > fn(NS(), as_att(sealed)).value


def test_scale_scorer_registered_and_metrics_report_it():
    fn = lookup("scorer", "scale").load()
    att = NS(blocks=1, world=NS(to_structure=lambda crop=True: NS(block_ids=hut())))
    term = fn(NS(), att)
    assert term.value > 0.95 and "door_ok 1.0" in term.note
    att_low = NS(blocks=1, world=NS(to_structure=lambda crop=True: NS(
        block_ids=hut(ceiling_y=3, door_h=1))))
    assert fn(NS(), att_low).value == 0.0
    assert fn(NS(), NS(blocks=0, world=None)).value == 0.0
    res = run_episode(make_fake_kit(replies=["fill stone 0 0 0 3 3 3\n"]), "x",
                      trace=RecordingTrace())
    m = res.metrics()
    for key in ("scale_headroom2", "scale_headroom3", "scale_door_ok", "scale_cells"):
        assert key in m


# ===========================================================================
# blueprint room minimum
# ===========================================================================
def test_room_minimum_is_a_3x3_interior_and_repair_stays_idempotent():
    assert MIN_ROOM_SIDE == 5
    bp = mock_blueprint(SIZE)
    tiny = Room("closet", "closet", 0, Rect(6, 8, 9, 11), "")          # 4 x 4 incl. walls
    ok = Room("pantry", "pantry", 0, Rect(6, 8, 10, 12), "")           # 5 x 5 incl. walls
    for extra, kept in ((tiny, False), (ok, True)):
        b = Blueprint.from_dict(bp.to_dict())
        b.rooms = [r for r in b.rooms if r.name != "bedroom"] + [extra]
        r1 = b.repaired(SIZE)
        assert (extra.name in [r.name for r in r1.rooms]) is kept
        r2 = r1.repaired(SIZE)
        assert r2.to_dict() == r1.to_dict() and len(r2.notes) == len(r1.notes)
    assert all(r.rect.w >= 5 and r.rect.d >= 5 for r in mock_blueprint(SIZE).repaired(SIZE).rooms)


# ===========================================================================
# shipped recipes and battery
# ===========================================================================
@pytest.mark.parametrize("name", ["dream", "dream-plan", "dream-refine", "dream-texpalette",
                                  "blueprint", "blueprint-text-plan", "blueprint-image-plan",
                                  "blueprint-nodream", "floors", "turnaround",
                                  "classic-matched", "best-of-4"])
def test_native_recipes_carry_scale(name):
    assert scale_mod.scale_params(load_recipe(name)) is not None


@pytest.mark.parametrize("name", ["dream-noscale", "blueprint-noscale", "classic-full",
                                  "dsl-classic", "legacy-full", "blocks-oneshot"])
def test_controls_and_legacy_arms_have_no_scale(name):
    assert scale_mod.scale_params(load_recipe(name)) is None


def test_noscale_controls_are_the_pre_scale_recipes():
    assert load_recipe("dream-noscale").hash() == load_recipe(
        "dream", {"contexts": ["palette", {"examples": {"k": 1}}]}).hash()
    assert load_recipe("blueprint-noscale").hash() == load_recipe(
        "blueprint", {"contexts": ["palette"]}).hash()
    assert load_recipe("dream-plan").params["plan"] is True
    assert load_recipe("blueprint-text-plan").params["plan_source"] == "text"


def test_scale_plan_battery_loads():
    bat = load_battery("scale-plan-v1")
    names = [r.name for r in bat.recipes]
    assert names == ["classic-matched", "dream", "dream-noscale", "dream_scale_textonly",
                     "dream-plan", "blueprint", "blueprint-noscale", "blueprint-text-plan",
                     "blueprint-image-plan"]
    textonly = bat.recipes[3]
    assert scale_mod.scale_active(NS(spec=textonly))
    assert scale_mod.dream_cue(NS(spec=textonly)) == ""
    assert len({r.hash() for r in bat.recipes}) == len(names)


def test_compact_dream_style_puts_the_description_first_and_stays_short():
    from types import SimpleNamespace as NS
    from blockgen.agentic import dreams
    from blockgen.agentic.recipe import load_recipe
    spec = load_recipe("dream-gemma-local")
    ep = NS(spec=spec, prompt="a Dutch windmill with four sails", dream_desc={})
    p = dreams.render_prompt(dreams.DREAMS["iso"], ep)
    assert p.startswith("a Dutch windmill with four sails. Minecraft voxel build")
    assert "isometric three-quarter view" in p and len(p.split()) < 60
    # other styles keep the template order (description last)
    p2 = dreams.render_prompt(dreams.DREAMS["iso"], ep, style="default")
    assert not p2.startswith("a Dutch windmill")
