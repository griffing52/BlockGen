"""steps_staged, offline: stages and their budgets, plan parsing and fallback,
units and their regions, unit renders, scores and accept rules."""

from __future__ import annotations

import json

import numpy as np
import pytest

from blockgen.agentic.blueprint import default_blueprint, mock_blueprint
from blockgen.agentic.core import Budget, BudgetExceeded, Ledger, MinecraftWorld
from blockgen.agentic.episode import Episode
from blockgen.agentic.providers_cli import AutoMockProvider
from blockgen.agentic.recipes import steps_staged as ss
from blockgen.agentic.region import Box, Region

from .fakes import (FAKE_DREAMS, RecordingTrace, ScriptedLLM, make_attempt, make_fake_kit,
                    solid_png)

SIZE = (32, 32, 32)


def episode(llm=None, *, budget_usd=None, dreams=("iso",), **kit_kw):
    kit = make_fake_kit(llm=llm or AutoMockProvider(), dreams=list(dreams), mock=True,
                        **kit_kw)
    trace = RecordingTrace()
    led = kit.ledger.child(Budget(usd=budget_usd), name="build")
    return Episode("a cosy cottage", kit, ledger=led, trace=trace), trace


# ===========================================================================
# stage()
# ===========================================================================
def test_stage_caps_and_restores_ledger():
    ep, trace = episode(budget_usd=1.0)
    root = ep.ledger
    with ss.stage(ep, "plan", 0.1) as led:
        assert ep.ledger is led and led.parent is root
        assert led.budget.usd == pytest.approx(0.1)
        with ss.stage(ep, "inner", 0.5) as inner:
            # fractions are of the BUILD's cap, also when nested
            assert inner.budget.usd == pytest.approx(0.5)
    assert ep.ledger is root
    kinds = trace.kinds()
    assert kinds.count("stage_start") == 2 and kinds.count("stage_end") == 2


def test_stage_own_budget_stop_is_quiet_parent_propagates():
    ep, trace = episode(budget_usd=1.0)
    with ss.stage(ep, "rooms", None, usd=0.0) as led:
        led.check("llm", 0.01)               # raises from THIS stage's ledger
        pytest.fail("unreachable")
    assert "stage_budget_stop" in trace.kinds()
    with pytest.raises(BudgetExceeded):
        with ss.stage(ep, "final", 0.5):
            ep.notes["_stage_root_ledger"].check("llm", 5.0)   # the build is out of money
    assert ep.ledger is ep.notes["_stage_root_ledger"]


def test_stage_without_cap():
    ep, _ = episode(budget_usd=None)
    with ss.stage(ep, "x", 0.5) as led:
        assert led.budget.usd is None


# ===========================================================================
# plan
# ===========================================================================
def test_blueprint_from_llm_mock_is_clean():
    ep, trace = episode()
    bp = ss.make_blueprint(ep, source="llm", max_rooms=4, max_storeys=2)
    assert bp.validate(SIZE) == []
    assert len(bp.rooms) >= 3 and len(bp.storeys) == 2
    assert "blueprint_problems" not in trace.kinds()


def test_blueprint_garbage_falls_back_to_default():
    llm = ScriptedLLM(["no json here, sorry"])
    ep, trace = episode(llm)
    bp = ss.make_blueprint(ep, source="llm", max_rooms=4, max_storeys=2, retries=1)
    assert "blueprint_fallback" in trace.kinds()
    assert len(llm.calls) == 2                       # the plan turn + one fix turn
    want = default_blueprint(SIZE, ep.prompt).repaired(SIZE, max_rooms=4, max_storeys=2)
    assert bp.to_dict() == want.to_dict()


def test_blueprint_problems_get_one_fix_turn():
    bad = {"footprint": [0, 0, 40, 40], "storeys": [{"height": 4}],
           "rooms": [{"name": "hall", "purpose": "hall", "storey": 0, "rect": [0, 0, 40, 40]}]}
    good = json.loads(mock_blueprint(SIZE).to_json())
    llm = ScriptedLLM([json.dumps(bad), json.dumps(good)])
    ep, trace = episode(llm)
    bp = ss.make_blueprint(ep, source="llm", max_rooms=4, max_storeys=2)
    assert "blueprint_problems" in trace.kinds()
    assert "problems" in llm.calls[1][-1].text.lower()
    assert bp.to_dict() == mock_blueprint(SIZE).repaired(SIZE, max_rooms=4,
                                                         max_storeys=2).to_dict()


def test_blueprint_from_image_uses_grid_ref():
    ep, _ = episode()
    bp = ss.make_blueprint(ep, source="image", max_rooms=4, max_storeys=2)
    req = ep.kit.image.requests[-1]
    assert len(req.refs) == 1 and req.view == "top"
    assert "plan_sketch" in ep.dreams and bp.validate(SIZE) == []


def test_record_blueprint_does_not_touch_dream_id():
    ep, trace = episode()
    ep.dream("iso")
    before = ep.dream_id()
    bp = mock_blueprint(SIZE)
    ss.record_blueprint(ep, bp, tag="llm")
    ss.record_blueprint(ep, bp, tag="again")      # re-register = no-op
    assert ep.dream_id() == before
    assert len(ep.dreams["blueprint"]) == 2 and ep.notes["blueprint"] is bp
    assert "blueprint" in trace.kinds()
    assert ep.kit.image.calls == 1                 # the schematic was not generated


# ===========================================================================
# units
# ===========================================================================
def test_room_units_geometry():
    bp = mock_blueprint(SIZE, storeys=2)
    units = ss.room_units(bp, order="public_first", max_units=10)
    assert len(units) == len(bp.rooms)
    assert units[0].purpose and units[0].key == f"u_{units[0].name}"
    for u in units:
        r = bp.room(u.name)
        st = bp.storey(r.storey)
        b = u.region.allow[0]
        assert (b.x0, b.z0, b.x1, b.z1) == (r.rect.x0 + 1, r.rect.z0 + 1,
                                            r.rect.x1 - 1, r.rect.z1 - 1)
        assert u.view_box.y0 == st.floor_y and u.cut_y == u.view_box.y1
        assert u.cut_y == st.floor_y + -(-st.height // 2)
    assert len(ss.room_units(bp, order="size", max_units=2)) == 2


def test_storey_units_bottom_up():
    bp = mock_blueprint(SIZE, storeys=2)
    units = ss.storey_units(bp)
    assert [u.storey for u in units] == [0, 1]
    assert units[1].region.allow[0].y0 == bp.storey(1).floor_y
    assert units[0].region.allow[0].y1 < units[1].region.allow[0].y0


def test_unit_render_caches_on_attempt():
    ep, _ = episode()
    bp = mock_blueprint(SIZE)
    att = ep.try_("shell", bp.shell_dsl(), ir=True)
    u = ss.room_units(bp, order="blueprint", max_units=1)[0]
    png = ss.unit_render(ep, att, u)
    assert png and att.renders[u.key] == png
    assert ss.unit_render(ep, MinecraftWorld.empty(SIZE), u) is None


def test_unit_terms_mock_and_unknown():
    ep, _ = episode()
    assert ss.unit_terms(ep, None) == {"silhouette": 1.0, "color_lab": 0.5, "pixel": 1.0}
    assert ss.unit_terms(ep, {"embed": 0.0, "pixel": 2})["pixel"] == 2.0
    with pytest.raises(ValueError):
        ss.unit_terms(ep, {"nope": 1})


def test_unit_accept_without_target_needs_region_gain():
    ep, _ = episode()
    region = Region("r", (Box.of(2, 1, 2, 6, 3, 6),))
    u = ss.Unit("r", "room", 0, region, Box.of(1, 0, 1, 7, 3, 7), 3, "room", "")
    acc = ss.unit_accept(ep, u, None, terms={"pixel": 1.0})
    base = make_attempt("fill stone 0 0 0 8 0 8\n")
    same = make_attempt("fill stone 0 0 0 8 0 8\nset stone 0 1 0\n")
    more = make_attempt("fill stone 0 0 0 8 0 8\nset stone 3 1 3\n")
    assert acc(base, more)[0] is True
    assert acc(base, same)[0] is False
    assert acc(base, make_attempt(""))[0] is False


def test_unit_accept_with_target_and_score_cache():
    ep, trace = episode()
    bp = mock_blueprint(SIZE)
    shell = ep.try_("shell", bp.shell_dsl(), ir=True)
    u = ss.room_units(bp, order="blueprint", max_units=1)[0]
    target = ss.unit_render(ep, shell, u)
    terms = {"silhouette": 1.0}
    acc = ss.unit_accept(ep, u, target, terms=terms)
    # the same room again scores the same: no improvement, rejected
    twin = ep.try_("twin", bp.shell_dsl(), ir=True, accept=lambda p, c: (False, "x"))
    ok, why = acc(shell, twin)
    assert ok is False and "no improvement" in why
    assert twin.extra["unit"] == u.name and "unit_score" in twin.extra
    n = len([k for k in trace.kinds() if k == "unit_score"])
    ss.unit_score(ep, twin, u, target, terms)            # cached: no new event
    assert len([k for k in trace.kinds() if k == "unit_score"]) == n


def test_interior_boxes_rooms_or_storeys():
    bp = mock_blueprint(SIZE)
    assert len(ss.interior_boxes(bp)) == len(bp.rooms)
    bare = bp.copy()
    bare.rooms = []
    assert len(ss.interior_boxes(bare)) == len(bp.storeys)


def test_massing_render_is_scratch():
    ep, _ = episode()
    bp = mock_blueprint(SIZE)
    ep.try_("shell", bp.shell_dsl(), ir=True)
    n_before = ep.world.block_count()
    png = ss.massing_render(ep, bp)
    assert png and ep.world.block_count() == n_before


def test_exterior_dream_massing_ref():
    ep, trace = episode()
    bp = mock_blueprint(SIZE)
    ep.try_("shell", bp.shell_dsl(), ir=True)
    dv = ss.exterior_dream(ep, bp, ref="massing", refine=0)
    assert dv is not None and dv.key == "iso"
    req = ep.kit.image.requests[-1]
    assert len(req.refs) == 1 and ss.MASSING_NOTE in req.prompt
    with pytest.raises(ValueError):
        ss.exterior_dream(ep, bp, ref="bogus", refine=0)


def test_exterior_dream_none_without_scored_dream():
    ep, trace = episode(dreams=())
    assert ss.exterior_dream(ep, None, ref="none", refine=0) is None
    assert "exterior_dream_skipped" in trace.kinds()


def test_unit_dream_registers_runtime_dream_with_cutaway_first():
    ep, _ = episode()
    bp = mock_blueprint(SIZE)
    shell = ep.try_("shell", bp.shell_dsl(), ir=True)
    ep.dream("iso")
    did = ep.dream_id()
    u = ss.room_units(bp, order="blueprint", max_units=1)[0]
    dv = ss.unit_dream(ep, bp, u, refs=("cutaway", "exterior"))
    assert dv is not None and dv.key == u.key
    req = ep.kit.image.requests[-1]
    assert req.view == "room" and len(req.refs) == 2
    assert req.refs[0] == ss.unit_render(ep, ep.world, u)
    assert req.refs[1] == ep.latest_dream("iso").png
    assert u.purpose in req.prompt and "{purpose}" not in req.prompt
    assert ep.dream_id() == did
    with pytest.raises(ValueError):
        ss.unit_dream(ep, bp, u, refs=("nope",))
    del shell


def test_build_unit_patch_stays_in_region():
    ep, trace = episode()
    bp = mock_blueprint(SIZE)
    ep.try_("shell", bp.shell_dsl(), ir=True)
    u = ss.room_units(bp, order="blueprint", max_units=1)[0]
    before = ep.world.to_structure(crop=False)
    att = ss.build_unit(ep, bp, u, target=None, iters=1, mode="patch", local_coords=False,
                        terms={}, eps=0.0, feedback="both")
    assert att is not None and att.accepted
    after = ep.world.to_structure(crop=False)
    changed = (before.block_ids != after.block_ids) | (before.block_data != after.block_data)
    assert changed.any()
    assert not (changed & ~u.region.allow[0].mask(SIZE)).any()
    assert [a.stage for a in ep.attempts][1:3] == [u.key, f"{u.key}_m0"]


def test_build_unit_local_coords_lands_in_region():
    ep, _ = episode()
    bp = mock_blueprint(SIZE)
    ep.try_("shell", bp.shell_dsl(), ir=True)
    u = ss.room_units(bp, order="blueprint", max_units=1)[0]
    before = ep.world.to_structure(crop=False)
    ss.build_unit(ep, bp, u, target=None, iters=0, mode="patch", local_coords=True,
                  terms={}, eps=0.0, feedback="both")
    after = ep.world.to_structure(crop=False)
    changed = before.block_ids != after.block_ids
    assert changed.any() and not (changed & ~u.region.allow[0].mask(SIZE)).any()
    sent = ep.transcript[-1]["request"]
    sx, sy, sz = u.region.allow[0].shape
    assert f"ALLOWED REGION: x 0..{sx - 1}, y 0..{sy - 1}, z 0..{sz - 1}" in sent


def test_final_match_protects_boxes():
    ep, _ = episode(dreams=())
    bp = mock_blueprint(SIZE)
    ep.try_("shell", bp.shell_dsl(), ir=True)
    protect = ss.interior_boxes(bp)
    before = ep.world.to_structure(crop=False)
    ss.final_match(ep, max_iters=1, feedback="raw", mode="patch", protect=protect,
                   stop_at=2.0)
    last = ep.attempts[-1]
    assert last.stage == "final0"
    after = last.world.to_structure(crop=False)
    changed = before.block_ids != after.block_ids
    mask = np.zeros(SIZE, bool)
    for b in protect:
        mask |= b.mask(SIZE)
    assert not (changed & mask).any()
    assert "protected" in ep.transcript[-1]["request"]


def test_unit_score_annotates_only_the_candidate():
    ep, _ = episode()
    bp = mock_blueprint(SIZE)
    shell = ep.try_("shell", bp.shell_dsl(), ir=True)
    u = ss.room_units(bp, order="blueprint", max_units=1)[0]
    target = ss.unit_render(ep, shell, u)
    acc = ss.unit_accept(ep, u, target, terms={"silhouette": 1.0})
    cand = ep.try_("cand", bp.shell_dsl() + "set lantern 1 1 1\n", ir=True,
                   accept=lambda p, c: (False, "x"))
    acc(shell, cand)
    assert "unit" not in shell.extra and cand.extra["unit"] == u.name


# ===========================================================================
# verifier additions
# ===========================================================================
def test_room_regions_exclude_the_stairwell_in_the_floor():
    """The top step and the two cleared stairwell cells sit in the upper floor
    slab, which an upper room's region otherwise includes."""
    bp = mock_blueprint(SIZE, storeys=2)
    units = ss.room_units(bp, order="blueprint", max_units=9)
    masks = {u.name: u.region.mask(SIZE) for u in units}
    assert bp.stairs
    for st in bp.stairs:
        s = bp.storey(st.from_storey)
        cells = st.cells(s.height)
        slab = [(x, s.ceiling_y, z) for (x, _dy, z) in cells[-3:]]
        for name, m in masks.items():
            for (x, y, z) in slab:
                assert not m[x, y, z], (name, (x, y, z))
    # the rest of the upper rooms' floors stays editable
    up = [u for u in units if u.storey == 1]
    assert up and all(masks[u.name][:, bp.storey(1).floor_y, :].sum() > 4 for u in up)
    assert all(ss.stairwell_boxes(bp, u.name) == [] for u in units if u.storey == 0)


def test_needs_unit_repair_on_reverted_writes():
    att = make_attempt()
    assert not ss.needs_unit_repair(att)
    att.extra["reverted"] = 3
    assert ss.needs_unit_repair(att)


def test_build_unit_repairs_writes_outside_the_region():
    """Whole-house coordinates in a room turn: reverted, then one repair turn."""
    bp = mock_blueprint(SIZE)
    u = ss.room_units(bp, order="blueprint", max_units=1)[0]
    b = u.region.allow[0]
    inside = f"set lantern {b.x0} {b.y0 + 1} {b.z0}\n"
    outside = "fill stone 0 0 0 3 0 3\n" + inside
    llm = ScriptedLLM([outside, inside])
    ep, trace = episode(llm)
    ep.try_("shell", bp.shell_dsl(), ir=True)
    ss.build_unit(ep, bp, u, target=None, iters=0, mode="patch", local_coords=False,
                  terms={}, eps=0.0, feedback="both")
    st = [a.stage for a in ep.attempts]
    assert st[1:] == [u.key, f"{u.key}_fix"]
    assert ep.attempts[1].extra.get("reverted", 0) > 0
    assert "outside" in ep.transcript[-1]["request"]


def test_compare_turn_reports_the_units_own_rejected_program():
    bp = mock_blueprint(SIZE)
    u = ss.room_units(bp, order="blueprint", max_units=1)[0]
    b = u.region.allow[0]
    llm = ScriptedLLM([f"set lantern {b.x0} {b.y0 + 1} {b.z0}\n"])
    ep, _ = episode(llm)
    shell = ep.try_("shell", bp.shell_dsl(), ir=True)
    target = ss.unit_render(ep, shell, u)
    from blockgen.agentic.core import DreamVersion
    tv = DreamVersion(u.key, 0, target, "", "fake")
    ss.build_unit(ep, bp, u, target=tv, iters=1, mode="patch", local_coords=False,
                  terms={"silhouette": 1.0}, eps=10.0, feedback="both")
    compare = [t for t in ep.transcript if t["stage"] == f"{u.key}_m0"][0]["request"]
    assert "NOT kept" in compare


def test_unit_dream_swallows_only_image_caps():
    bp = mock_blueprint(SIZE)
    u = ss.room_units(bp, order="blueprint", max_units=1)[0]
    # image cap -> None (the room is built from text)
    ep, trace = episode()
    ep.ledger = ep.ledger.child(Budget(max_images=0), name="imgcap")
    assert ss.unit_dream(ep, bp, u, refs=()) is None
    assert "unit_dream_budget_stop" in trace.kinds()
    # money cap -> propagates (stage() or run_episode decides)
    ep2, _ = episode()

    def broke(kind, est=0.0):
        raise BudgetExceeded("ledger 'build': $1.0000 spent + $0.1000 next > cap $1.0000")

    ep2.ledger.check = broke
    with pytest.raises(BudgetExceeded):
        ss.unit_dream(ep2, bp, u, refs=())
    assert ss.is_image_cap(BudgetExceeded("ledger 'x': image cap 3 reached"))
    assert not ss.is_image_cap(BudgetExceeded("ledger 'x': $1 spent"))


def test_storey_dream_floor_size_excludes_walls():
    bp = mock_blueprint(SIZE)
    u = ss.storey_units(bp)[0]
    fp = bp.footprint_of(0)
    assert ss._interior_wd(u) == (fp.w - 2, fp.d - 2)
