"""Staged turns (turns_staged) and the mock:auto staged dispatcher, offline.

Every build turn must end with the action space's output rule and every verdict
turn with JSON_RULE, for every action space; unit turns carry the exact
REGION_LINE the mock parses, and the mock's unit reply stays inside it in each
space's own format.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from blockgen.agentic import core, turns_staged as ts
from blockgen.agentic.blueprint import SCHEMA_TEXT, Blueprint, mock_blueprint
from blockgen.agentic.core import MinecraftWorld
from blockgen.agentic.providers import Message, text_message
from blockgen.agentic.providers_cli import AutoMockProvider, mock_furnishing
from blockgen.agentic.recipes.steps_staged import room_units, storey_units
from blockgen.agentic.turns import JSON_RULE

from .fakes import make_attempt, solid_png

SPACES = ("dsl", "blocks", "python", "mixed")
SIZE = (32, 32, 32)
PNG = solid_png((120, 80, 40))
REGION_RE = re.compile(r"ALLOWED REGION: x (-?\d+)\.\.(-?\d+), y (-?\d+)\.\.(-?\d+), "
                       r"z (-?\d+)\.\.(-?\d+)")


def fake_ep(space: str, size=SIZE):
    actions = core.make("actions", space, size=tuple(size))
    return SimpleNamespace(kit=SimpleNamespace(actions=actions), size=tuple(size),
                           prompt="a cosy two-room cottage")


@pytest.fixture(scope="module")
def bp() -> Blueprint:
    return mock_blueprint(SIZE, storeys=2)


def last_text(msg: Message) -> str:
    return msg.parts[-1].text.rstrip()


def build_turns(ep, bp):
    room = room_units(bp, order="blueprint", max_units=1)[0]
    storey = storey_units(bp)[0]
    att = make_attempt()
    return {
        "shell_build": ts.shell_build(ep, bp, [PNG]),
        "unit_build": ts.unit_build(ep, bp, room, target=PNG, current=PNG, schematic=PNG,
                                    mode="patch", local=False),
        "unit_build_notarget": ts.unit_build(ep, bp, room, target=None, current=None,
                                             schematic=None, mode="rewrite", local=True),
        "unit_compare": ts.unit_compare(ep, room, att, target=PNG, yours=PNG, score=0.4,
                                        mode="patch", local=False, feedback="both"),
        "unit_critique": ts.unit_critique(ep, room, att, yours=PNG, mode="patch", local=False),
        "unit_repair": ts.unit_repair(ep, room, att, local=False),
        "storey_build": ts.storey_build(ep, bp, storey, target=PNG, below=None, exterior=PNG,
                                        schematic=PNG, mode="patch", local=False),
        "roof_build": ts.roof_build(ep, bp, exterior=PNG, current=PNG),
    }


def verdict_turns(ep, bp):
    return {
        "blueprint_json": ts.blueprint_json(ep, max_rooms=4, max_storeys=2),
        "blueprint_fix": ts.blueprint_fix(ep, {"footprint": [0, 0, 3, 3]}, ["too small"]),
        "plan_extract": ts.plan_extract(ep, PNG, PNG, max_rooms=4, max_storeys=2),
        "plan_verify": ts.plan_verify(ep, bp, PNG, PNG),
    }


@pytest.mark.parametrize("space", SPACES)
def test_build_turns_end_with_output_rule(space, bp):
    ep = fake_ep(space)
    rule = ep.kit.actions.output_rule()
    for name, msg in build_turns(ep, bp).items():
        assert last_text(msg).endswith(rule), name


@pytest.mark.parametrize("space", SPACES)
def test_verdict_turns_end_with_json_rule(space, bp):
    ep = fake_ep(space)
    for name, msg in verdict_turns(ep, bp).items():
        assert last_text(msg).endswith(JSON_RULE), name


def test_tags_and_lines(bp):
    ep = fake_ep("dsl")
    for name, msg in verdict_turns(ep, bp).items():
        text = msg.text
        if name == "plan_verify":
            assert ts.VERIFY_TAG in text and ts.BLUEPRINT_TAG not in text
            continue
        assert ts.BLUEPRINT_TAG in text, name
        assert "Canvas: 32 x 32 x 32 blocks" in text, name
        assert SCHEMA_TEXT in text, name
    bj = verdict_turns(ep, bp)["blueprint_json"].text
    assert "at most 4 rooms" in bj.lower() and "2 storey" in bj
    for name, msg in build_turns(ep, bp).items():
        if name.startswith("unit") or name == "storey_build":
            text = msg.text
            assert ts.UNIT_TAG in text, name
            assert REGION_RE.search(text), name
            assert ts.DISCARD_NOTE in text, name


def test_region_line_absolute_and_local(bp):
    room = room_units(bp, order="blueprint", max_units=1)[0]
    b = room.region.allow[0]
    absolute = ts.region_line(room, local=False)
    assert absolute == (f"ALLOWED REGION: x {b.x0}..{b.x1}, y {b.y0}..{b.y1}, "
                        f"z {b.z0}..{b.z1}")
    local = ts.region_line(room, local=True)
    sx, sy, sz = b.shape
    assert local == f"ALLOWED REGION: x 0..{sx - 1}, y 0..{sy - 1}, z 0..{sz - 1}"
    block = ts.region_block(room, local=True)
    assert "LOCAL" in block
    if room.region.deny:
        assert "Keep these cells clear" in block


def test_unit_build_images_and_mode(bp):
    ep = fake_ep("dsl")
    msg = build_turns(ep, bp)["unit_build"]
    labels = [p.text for p in msg.parts if p.kind == "text"]
    assert any(l.startswith("TARGET u_") for l in labels)
    assert any(l.startswith("CURRENT u_") for l in labels)
    assert sum(p.kind == "image" for p in msg.parts) == 3
    assert "additional commands" in msg.text        # patch mode instruction


def test_dream_templates_have_fields():
    for t in (ts.ROOM_DREAM, ts.STOREY_DREAM):
        for f in ("{style}", "{purpose}", "{desc}", "{iw}", "{id}"):
            assert f in t
    assert "{style}" not in ts.PLAN_DREAM        # a flat drawing, not the house style
    assert "{desc}" in ts.PLAN_DREAM and "{style}" in ts.PLAN_STYLIZE


# ===========================================================================
# mock:auto staged dispatcher
# ===========================================================================
def ask_mock(msg: Message, system: str = "SYSTEM") -> tuple:
    m = AutoMockProvider()
    resp = m.complete([text_message("system", system), msg])
    return m.decisions[-1], resp.text


def test_mock_blueprint_reply_parses(bp):
    ep = fake_ep("dsl")
    for name in ("blueprint_json", "blueprint_fix", "plan_extract"):
        decision, text = ask_mock(verdict_turns(ep, bp)[name])
        assert decision == "staged:blueprint", name
        got = Blueprint.from_json(text)
        assert got.validate(SIZE) == []
        assert len(got.storeys) == 2                  # H = 32 >= 24
    small = fake_ep("dsl", size=(24, 16, 24))
    decision, text = ask_mock(ts.blueprint_json(small, max_rooms=3, max_storeys=2))
    assert decision == "staged:blueprint"
    assert len(Blueprint.from_json(text).storeys) == 1


def test_mock_verify_reply(bp):
    decision, text = ask_mock(verdict_turns(fake_ep("dsl"), bp)["plan_verify"])
    assert decision == "staged:verify"
    assert json.loads(text) == {"ok": True, "issues": [], "blueprint": None}


def test_mock_sheet_review_literal():
    decision, text = ask_mock(text_message("user", "TURNAROUND SHEET REVIEW: look.\n"
                                           + JSON_RULE))
    assert decision == "staged:sheet_review"
    assert json.loads(text)["consistent"] is True


@pytest.mark.parametrize("space", SPACES)
def test_mock_unit_reply_inside_region(space, bp):
    ep = fake_ep(space)
    room = room_units(bp, order="blueprint", max_units=1)[0]
    msg = ts.unit_build(ep, bp, room, target=None, current=None, schematic=None,
                        mode="patch", local=False)
    decision, text = ask_mock(msg, system=ep.kit.actions.instructions(
        core.PromptCtx(size=SIZE, palette=["stone"])))
    assert decision == f"staged:unit:{space}"
    low = ep.kit.actions.lower(text)
    world = MinecraftWorld.empty(SIZE)
    report = world.run(low)
    assert report.n_failed == 0 and world.block_count() > 0
    s = world.to_structure(crop=False)
    allow = room.region.allow[0].mask(SIZE)
    assert not (s.occupied_mask & ~allow).any()


def test_mock_unit_without_region_line_falls_through():
    decision, _ = ask_mock(text_message("user", f"{ts.UNIT_TAG}: no region here.\n"
                                        "Output only command lines."))
    assert not decision.startswith("staged:")


def test_mock_furnishing_clamps_to_flat_box():
    text = mock_furnishing(5, 5, 3, 3, 2, 2)
    world = MinecraftWorld.empty(SIZE)
    from blockgen.agentic.core import Lowered
    world.run(Lowered(text, text, [], 4))
    s = world.to_structure(crop=False)
    occ = s.occupied_mask
    assert occ.sum() == 1 and occ[5, 3, 2]


def test_generic_mock_paths_unchanged():
    m = AutoMockProvider()
    m.complete([text_message("user", "anything at all.\n" + JSON_RULE)])
    m.complete([text_message("user", "Build a hut.\nOutput only command lines.")])
    assert m.decisions == ["json", "actions:dsl"]


# ===========================================================================
# verifier additions
# ===========================================================================
def test_unit_turns_say_where_the_floor_is(bp):
    ep = fake_ep("dsl")
    room = room_units(bp, order="blueprint", max_units=1)[0]
    storey = storey_units(bp)[1]
    y0 = room.region.allow[0].y0
    t = ts.unit_build(ep, bp, room, target=None, current=None, schematic=None,
                      mode="patch", local=False).text
    assert f"The floor is the layer y = {y0}" in t and f"from y = {y0 + 1} up" in t
    t = ts.unit_build(ep, bp, room, target=None, current=None, schematic=None,
                      mode="patch", local=True).text
    assert "The floor is the layer y = 0" in t
    t = ts.storey_build(ep, bp, storey, target=None, below=None, exterior=None,
                        schematic=None, mode="patch", local=False).text
    assert f"y = {storey.region.allow[0].y0} is this storey's floor slab" in t


def test_storey_build_local_frame_is_consistent(bp):
    """In local mode the brief's numbers are in the REGION line's frame."""
    ep = fake_ep("dsl")
    u = storey_units(bp)[1]
    fp = bp.footprint_of(1)
    st = bp.storey(1)
    t = ts.storey_build(ep, bp, u, target=None, below=None, exterior=None, schematic=None,
                        mode="patch", local=True).text
    assert f"footprint x 0..{fp.x1 - fp.x0}, z 0..{fp.z1 - fp.z0}" in t
    assert f"floor at y=0, walls y 1..{st.height}" in t
    r = bp.rooms_on(1)[0]
    assert f"rect {[r.rect.x0 - fp.x0, r.rect.z0 - fp.z0, r.rect.x1 - fp.x0, r.rect.z1 - fp.z0]}" in t
    t = ts.storey_build(ep, bp, u, target=None, below=None, exterior=None, schematic=None,
                        mode="patch", local=False).text
    assert f"footprint x {fp.x0}..{fp.x1}" in t and f"floor at y={st.floor_y}," in t


def test_compare_turn_says_when_the_last_program_was_not_kept(bp):
    ep = fake_ep("dsl")
    room = room_units(bp, order="blueprint", max_units=1)[0]
    att = make_attempt()
    att.accepted, att.reason = False, "unit hall no improvement"
    t = ts.unit_compare(ep, room, att, target=PNG, yours=PNG, score=0.4, mode="patch",
                        local=False, feedback="both").text
    assert "NOT kept (unit hall no improvement)" in t
    att.accepted = True
    t = ts.unit_critique(ep, room, att, yours=PNG, mode="patch", local=False).text
    assert "NOT kept" not in t


@pytest.mark.parametrize("space", SPACES)
def test_mock_room_furnishing_spares_floor_and_keep_clear(space, bp):
    """The mock furnishes ON a room's floor and never inside a keep-clear box, so a
    mock room turn is not reverted (and repaired) on every room with a door."""
    ep = fake_ep(space)
    for room in room_units(bp, order="blueprint", max_units=9):
        assert room.region.deny
        msg = ts.unit_build(ep, bp, room, target=None, current=None, schematic=None,
                            mode="patch", local=False)
        decision, text = ask_mock(msg, system=ep.kit.actions.instructions(
            core.PromptCtx(size=SIZE, palette=["stone"])))
        world = MinecraftWorld.empty(SIZE)
        report = world.run(ep.kit.actions.lower(text))
        assert report.n_failed == 0 and world.block_count() > 0
        occ = world.to_structure(crop=False).occupied_mask
        assert not (occ & ~room.region.mask(SIZE)).any(), room.name
        assert not occ[:, room.region.allow[0].y0, :].any(), room.name


def test_rewrite_mode_room_turn_asks_for_the_floor(bp):
    ep = fake_ep("dsl")
    room = room_units(bp, order="blueprint", max_units=1)[0]
    msg = ts.unit_build(ep, bp, room, target=None, current=None, schematic=None,
                        mode="rewrite", local=False)
    assert "include the floor" in msg.text
    _, text = ask_mock(msg)
    world = MinecraftWorld.empty(SIZE)
    world.run(ep.kit.actions.lower(text))
    occ = world.to_structure(crop=False).occupied_mask
    assert occ[:, room.region.allow[0].y0, :].any()          # the mock lays the floor
