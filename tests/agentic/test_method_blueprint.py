"""The ``blueprint`` method end to end, offline: on a fake kit with the mock:auto
LLM, and on the shipped recipes built with ``mock=True``.

Covers the WP-B acceptance list: a ``blueprint`` schematic dream, a shell
attempt, ``u_<room>`` dreams and renders, room attempts carrying
``extra.unit_score``, a roof attempt, final attempts, ``best()`` within
``select_from``; image / both plan sources; the no-dream recipe makes zero image
calls; a stage budget ends a stage without ending the build; the shuffle control
rotates the room targets.
"""

from __future__ import annotations

import pytest

import blockgen.agentic.recipes.blueprint  # noqa: F401  (registers the method)
from blockgen.agentic.blueprint import Blueprint, order_rooms
from blockgen.agentic.episode import Episode, run_episode
from blockgen.agentic.kit import build_kit
from blockgen.agentic.providers_cli import AutoMockProvider
from blockgen.agentic.recipe import load_recipe
from blockgen.agentic.recipes.blueprint import blueprint

from .fakes import FakeActions, RecordingTrace, ScriptedLLM, make_fake_kit

SIZE = (32, 32, 32)


class PricedAuto(ScriptedLLM):
    """mock:auto's answers at gpt-5-mini prices, so stage budgets have teeth."""

    def __init__(self) -> None:
        auto = AutoMockProvider()
        super().__init__([lambda msgs: auto._decide(msgs)[1]], model="gpt-5-mini",
                         prompt_tokens=2000, completion_tokens=500)


def run(params=None, *, llm=None, dreams=("iso",), budget=None, image=True, **kit_kw):
    kw = dict(kit_kw)
    if not image:
        kw["image"] = None
    kit = make_fake_kit(llm=llm or AutoMockProvider(), method=blueprint, dreams=list(dreams),
                        mock=True, spec_kw={"params": {"stop_at": 2.0, **(params or {})}, "select": "best",
                                            "budget": budget or {}}, **kw)
    trace = RecordingTrace()
    res = run_episode(kit, "a cosy cottage with a kitchen", trace=trace)
    return res, kit, trace


def stages(res):
    return [a.stage for a in res.attempts]


def unit_keys(res):
    return sorted(k for k in res.dreams if k.startswith("u_"))


def test_blueprint_on_fake_kit_full_pipeline():
    res, kit, trace = run()
    assert not res.truncated, res.error
    st = stages(res)
    assert st[0] == "shell"
    assert "blueprint" in res.dreams and "iso" in res.dreams
    rooms = unit_keys(res)
    assert len(rooms) >= 3
    for key in rooms:
        assert key in st                               # first build turn of each room
        room_atts = [a for a in res.attempts if a.stage.startswith(key)]
        assert any(key in a.renders for a in room_atts)
        assert all("unit_score" in a.extra for a in room_atts)
        assert all(a.extra.get("region") == key[2:] for a in room_atts)
    assert "roof" in st
    assert any(s.startswith("final") for s in st)
    roof_idx = st.index("roof")
    assert res.best_idx is not None and res.best_idx >= roof_idx
    kinds = trace.kinds()
    assert "blueprint" in kinds and "rescore_all" in kinds
    assert kinds.count("stage_start") == kinds.count("stage_end")
    # the global score never depends on a room dream
    iso_only = {a.score.dream_id for a in res.attempts if a.score is not None and a.blocks}
    assert len(iso_only) == 1


def test_exterior_dream_is_anchored_on_massing():
    res, kit, _ = run()
    iso_req = [r for r in kit.image.requests if r.view == "iso"]
    assert iso_req and len(iso_req[0].refs) == 1
    room_reqs = [r for r in kit.image.requests if r.view == "room"]
    assert room_reqs and all(len(r.refs) == 2 for r in room_reqs)   # cutaway + exterior


def test_rooms_follow_public_first_order():
    res, _, trace = run()
    plan = [d for k, d in trace.events if k == "blueprint"][0]["blueprint"]
    want = [r.name for r in order_rooms(Blueprint.from_dict(plan).rooms, "public_first")]
    got = [a.stage[2:] for a in res.attempts
           if a.stage.startswith("u_") and a.stage[2:] in want]
    assert list(dict.fromkeys(got)) == want


@pytest.mark.parametrize("source", ["image", "both"])
def test_plan_sources_run_offline(source):
    res, kit, trace = run({"plan_source": source})
    assert not res.truncated, res.error
    key = "plan_sketch" if source == "image" else "plan_dream"
    assert key in res.dreams
    assert "roof" in stages(res)
    if source == "both":
        assert "plan_verify" in trace.kinds()


def test_nodream_params_make_no_images():
    res, kit, trace = run({"exterior_dream": False, "interior_dreams": False}, dreams=(),
                          image=False)
    assert not res.truncated, res.error
    assert set(res.dreams) == {"blueprint"}
    assert sum(c.images for c in res.charges if c.kind == "image") == 0
    room_atts = [a for a in res.attempts if a.stage.startswith("u_")]
    assert room_atts and all("unit_score" not in a.extra for a in room_atts)
    assert any("gained" in a.reason for a in room_atts)


def test_image_params_need_an_image_generator():
    kit = make_fake_kit(llm=AutoMockProvider(), method=blueprint, image=None, mock=True)
    ep = Episode("x", kit)
    with pytest.raises(ValueError, match="image generator"):
        blueprint(ep)
    with pytest.raises(ValueError, match="room_order"):
        blueprint(ep, room_order="random", exterior_dream=False, interior_dreams=False)
    with pytest.raises(ValueError, match="room_refs"):
        blueprint(ep, room_refs=("below",), exterior_dream=False, interior_dreams=False)


def test_patch_needs_patchable_actions():
    class Rigid(FakeActions):
        patchable = False

    kit = make_fake_kit(llm=AutoMockProvider(), method=blueprint, actions=Rigid(), mock=True)
    with pytest.raises(ValueError, match="patchable"):
        blueprint(Episode("x", kit))


def test_stage_budget_stops_a_stage_not_the_build():
    res, kit, trace = run({"stage_budget": {"rooms": 0.001}}, llm=PricedAuto(),
                          budget={"usd_per_build": 1.0})
    assert not res.truncated, res.error
    stops = [d for k, d in trace.events if k == "stage_budget_stop"]
    assert stops and all(d["stage"].startswith("room") for d in stops)
    st = stages(res)
    assert not any(s.startswith("u_") for s in st)      # every room was cut short
    assert "roof" in st and any(s.startswith("final") for s in st)


def test_build_budget_still_truncates():
    uncapped = {k: None for k in ("plan", "shell", "rooms", "roof", "final")}
    res, _, trace = run({"stage_budget": uncapped}, llm=PricedAuto(),
                        budget={"usd_per_build": 0.03})
    assert res.truncated == "budget"
    assert res.attempts                                  # the shell (free) is kept


def test_room_dream_shuffle_rotates_targets():
    res, kit, trace = run({"room_dream_shuffle": True})
    assert not res.truncated, res.error
    pairs = [d for k, d in trace.events if k == "room_dream_shuffle"][0]["pairs"]
    assert len(pairs) >= 2
    digest = {k[2:]: v[-1].digest for k, v in res.dreams.items() if k.startswith("u_")}
    for room, other in pairs.items():
        assert other != room
        scored = {d["target"] for k, d in trace.events
                  if k == "unit_score" and d["unit"] == room}
        assert scored == {digest[other]}
    # all room dreams were generated before the first room was built
    room_reqs = [i for i, r in enumerate(kit.image.requests) if r.view == "room"]
    assert len(room_reqs) == len(pairs)


def test_select_from_falls_inside_complete_buildings():
    res, _, _ = run({"max_iters": 0})
    st = stages(res)
    assert st[res.best_idx] == "roof"


# ===========================================================================
# shipped recipes under --mock
# ===========================================================================
def run_recipe(name, prompt="a cosy cottage with a kitchen"):
    spec = load_recipe(name)
    kit = build_kit(spec, mock=True, cache_mode="off")
    trace = RecordingTrace()
    return run_episode(kit, prompt, trace=trace), kit, trace


@pytest.mark.slow
def test_recipe_blueprint_mock():
    res, kit, trace = run_recipe("blueprint")
    assert not res.truncated, res.error
    st = stages(res)
    assert st[0] == "shell" and "roof" in st
    assert unit_keys(res) and any("unit_score" in a.extra for a in res.attempts)
    assert res.best_idx >= st.index("roof")


@pytest.mark.slow
def test_recipe_blueprint_nodream_mock_zero_images():
    res, kit, trace = run_recipe("blueprint-nodream")
    assert not res.truncated, res.error
    assert kit.image is None
    assert sum(c.images for c in res.charges if c.kind == "image") == 0
    assert set(res.dreams) == {"blueprint"}
    assert "roof" in stages(res)


@pytest.mark.slow
def test_recipe_blueprint_image_plan_mock():
    res, kit, trace = run_recipe("blueprint-image-plan")
    assert not res.truncated, res.error
    assert "plan_sketch" in res.dreams and "roof" in stages(res)


# ===========================================================================
# verifier additions: composition with the shell, dialogue context
# ===========================================================================
def test_two_storey_rooms_keep_stairs_doors_and_stairwell():
    """Rooms are furnished (no-dream arm: accepted when they gain blocks), and the
    shell's circulation survives: door blocks, every stair step (the top one sits
    in the upper floor slab, inside the upper room's region) and the stairwell."""
    from blockgen.agentic.blockstate import resolve_block
    from blockgen.agentic.region import world_arrays

    import re

    auto = AutoMockProvider()
    region_re = re.compile(r"ALLOWED REGION: x (\d+)\.\.(\d+), y (\d+)\.\.(\d+), "
                           r"z (\d+)\.\.(\d+)")

    def reply(msgs):
        # A model that re-floors each room (the region's lowest layer IS the floor)
        # and adds a lantern; everything else as mock:auto.
        m = region_re.search(msgs[-1].text)
        if m is None or "UNIT BUILD: room" not in msgs[-1].text:
            return auto._decide(msgs)[1]
        x0, x1, y0, y1, z0, z1 = (int(v) for v in m.groups())
        return (f"fill birch_planks {x0} {y0} {z0} {x1} {y0} {z1}\n"
                f"set lantern {x0} {y0 + 1} {z0}\n")

    res, kit, trace = run({"exterior_dream": False, "interior_dreams": False, "max_iters": 0},
                          dreams=(), image=False, llm=ScriptedLLM([reply]))
    assert not res.truncated, res.error
    bp = Blueprint.from_dict([d for k, d in trace.events if k == "blueprint"][0]["blueprint"])
    assert len(bp.storeys) == 2 and bp.stairs
    rooms_up = {r.name for r in bp.rooms_on(1)}
    accepted = {a.stage[2:] for a in res.attempts if a.stage.startswith("u_") and a.accepted}
    assert accepted & rooms_up                          # an upper room was furnished
    ids, _ = world_arrays(res.attempts[res.best_idx].world)
    air = resolve_block("air")[0]
    stair_id = resolve_block(bp.materials["stairs"])[0]
    door_id = resolve_block(bp.materials["door"])[0]
    for (x, y, z) in bp.door_cells():
        assert ids[x, y, z] == door_id, (x, y, z)
    for st in bp.stairs:
        s = bp.storey(st.from_storey)
        cells = st.cells(s.height)
        for (x, dy, z) in cells:
            assert ids[x, s.floor_y + dy, z] == stair_id, (x, dy, z)
        for (x, _dy, z) in cells[-3:-1]:
            assert ids[x, s.ceiling_y, z] == air, (x, z)


def test_first_room_turn_carries_no_exterior_target():
    """Each unit turn is self-contained: the Episode's first-builder-turn prefix
    (context + global TARGET dreams) must not land on whichever room is first."""
    llm = AutoMockProvider()
    res, kit, trace = run(llm=llm)
    assert not res.truncated, res.error
    room_turns = [r for r in trace.requests if r["stage"].startswith("u_")
                  and not r["stage"].endswith("_fix") and "_m" not in r["stage"]]
    assert len(room_turns) >= 3
    texts = [[p.text for p in r["messages"][-1].parts if p.kind == "text"]
             for r in room_turns]
    for t in texts:
        assert not any(x.startswith("TARGET iso") for x in t)
    # every room's first turn has the same context prefix
    firsts = [t[0] for t in texts]
    assert len(set(firsts)) == 1


def test_shuffle_needs_interior_dreams():
    kit = make_fake_kit(llm=AutoMockProvider(), method=blueprint, mock=True)
    with pytest.raises(ValueError, match="room_dream_shuffle"):
        blueprint(Episode("x", kit), room_dream_shuffle=True, interior_dreams=False)


def test_room_patience_counts_the_first_turn():
    """A rejected first build + one rejected round = 2 consecutive rejections."""
    res, _, trace = run({"max_room_iters": 5, "room_eps": 10.0})
    for key in unit_keys(res):
        n = sum(1 for a in res.attempts if a.stage.startswith(key))
        assert n == 2, (key, n)
