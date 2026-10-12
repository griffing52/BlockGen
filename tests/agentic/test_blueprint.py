"""Blueprints: parsing, repair (clipping, overlaps, connectivity, stairs, idempotence),
the deterministic shell/roof DSL and the schematic renderers."""

from __future__ import annotations

import io

import numpy as np
import pytest

from blockgen.agentic.blockstate import STAIRS_IDS, resolve_block
from blockgen.agentic.blueprint import (DEFAULT_MATERIALS, MAX_STOREY_HEIGHT,
                                        MIN_STOREY_HEIGHT, OUTSIDE, ROOF_KINDS,
                                        SCHEMA_TEXT, SHELL_PARTS, Blueprint,
                                        BlueprintError, Rect, Stair,
                                        default_blueprint, front_side,
                                        mock_blueprint, order_rooms, render_grid,
                                        render_schematic, render_schematics)
from blockgen.agentic.dsl import run_program
from blockgen.agentic.episode import extract_json
from blockgen.agentic.region import Box

SIZE = (32, 32, 32)


def build(bp: Blueprint, size=SIZE):
    text = bp.shell_dsl() + bp.roof_dsl(size)
    canvas, report, _ = run_program(text, size=size)
    return canvas, report, text


def interiors_disjoint(bp: Blueprint) -> bool:
    for i, a in enumerate(bp.rooms):
        for b in bp.rooms[i + 1:]:
            if a.storey == b.storey and a.interior.intersect(b.interior) is not None:
                return False
    return True


MESSY = {
    "footprint": ["-3", "2", 40, "29"],
    "storeys": [{"height": "4"}, {"height": 12}, "junk"],
    "rooms": [
        {"name": "Great Hall", "purpose": "", "storey": 0, "rect": [2, 2, 18, 29]},
        {"name": "kitchen", "purpose": "kitchen", "storey": 0,
         "rect": {"x0": 12, "z0": 10, "x1": 30, "z1": 29}},          # overlaps the hall
        {"name": "kitchen", "purpose": "pantry", "storey": 0,
         "rect": {"x": 2, "z": 2, "w": 2, "d": 2}},                  # too small
        {"name": "attic", "purpose": "storage", "storey": 1, "rect": [2, 2, 29, 20]},
        {"name": "ghost", "storey": 7, "rect": [0, 0, 5, 5]},        # no such storey
        "not a room",
        {"name": "no rect"},
    ],
    "openings": [{"kind": "portal", "at": [5, 5]},
                 {"kind": "window", "storey": 0, "at": "9, 15", "between": "hall-kitchen"}],
    "stairs": [{"from_storey": 0, "at": [0, 0], "dir": "up"}],
    "roof": {"kind": "dome", "axis": "y", "overhang": 9},
    "materials": {"wall": "unobtanium", "roof": "dark_oak_stairs"},
}


# ---------------------------------------------------------------- Rect
def test_rect_of_and_parse_normalize():
    assert Rect.of(10, 8, 2, 3) == Rect(2, 3, 10, 8)
    assert Rect.parse([10, 8, 2, 3]) == Rect(2, 3, 10, 8)
    assert Rect.parse(["1", "2.0", "5", "6"]) == Rect(1, 2, 5, 6)
    assert Rect.parse({"x0": 5, "z0": 6, "x1": 1, "z1": 2}) == Rect(1, 2, 5, 6)
    assert Rect.parse({"x": 1, "z": 2, "w": 4, "d": 3}) == Rect(1, 2, 4, 4)
    assert Rect.parse("[1, 2, 5, 6]") == Rect(1, 2, 5, 6)
    for bad in ([1, 2, 3], {"x": 1}, "abc", None, {"x": 0, "z": 0, "w": 0, "d": 2}):
        with pytest.raises(ValueError):
            Rect.parse(bad)


def test_rect_geometry():
    r = Rect(0, 0, 10, 12)
    assert (r.w, r.d, r.area) == (11, 13, 143)
    assert r.interior() == Rect(1, 1, 9, 11)
    assert Rect(0, 0, 1, 5).interior() is None
    assert r.on_border(0, 5) and r.on_border(4, 12) and not r.on_border(4, 5)
    assert len(r.border_cells()) == len(set(r.border_cells())) == 2 * 11 + 2 * 11
    # shared wall excludes the two end cells (corners are structural)
    k = Rect(10, 0, 20, 12)
    assert r.shared_wall(k) == [(10, z) for z in range(1, 12)]
    assert r.shared_wall(Rect(15, 0, 20, 12)) == []
    assert r.intersect(Rect(5, 5, 30, 30)) == Rect(5, 5, 10, 12)
    assert r.intersect(Rect(11, 0, 12, 1)) is None


def test_stair_cells():
    st = Stair(0, 3, 4, "+z")
    assert st.cells(4) == [(3, 1, 4), (3, 2, 5), (3, 3, 6), (3, 4, 7), (3, 5, 8)]
    assert Stair(0, 3, 4, kind="ladder").cells(2) == [(3, 1, 4), (3, 2, 4), (3, 3, 4)]


# ---------------------------------------------------------------- parsing
def test_from_dict_lenient_drops_junk_with_notes():
    bp = Blueprint.from_dict(MESSY)
    assert [r.name for r in bp.rooms] == ["Great Hall", "kitchen", "kitchen", "attic", "ghost"]
    assert len(bp.storeys) == 2 and bp.storeys[1].floor_y == bp.storeys[0].ceiling_y
    assert any("room #5" in n for n in bp.notes) and any("room #6" in n for n in bp.notes)
    assert any("storey #2" in n for n in bp.notes)
    assert bp.openings[1].between == ("hall", "kitchen") and (bp.openings[1].x, bp.openings[1].z) == (9, 15)
    assert bp.openings[1].sill == 1 and bp.openings[1].height == 1     # window defaults
    assert bp.materials["roof"] == "dark_oak_stairs" and bp.materials["floor"] == "oak_planks"


def test_from_dict_strict_and_errors():
    with pytest.raises(BlueprintError):
        Blueprint.from_dict(MESSY, strict=True)
    with pytest.raises(BlueprintError):
        Blueprint.from_dict({"style": "nothing to build"})
    with pytest.raises(BlueprintError):
        Blueprint.from_json("no json here")
    # missing footprint = bbox of rooms; missing storeys = one of height 4
    bp = Blueprint.from_dict({"rooms": [{"name": "a", "rect": [2, 3, 9, 9]},
                                        {"name": "b", "rect": [9, 3, 14, 9]}]})
    assert bp.footprint == Rect(2, 3, 14, 9)
    assert [(s.index, s.floor_y, s.height) for s in bp.storeys] == [(0, 0, 4)]
    # a model that wraps its JSON in prose and fences
    text = "Here is the plan:\n```json\n" + mock_blueprint(SIZE).to_json() + "\n```\nDone."
    assert Blueprint.from_json(text) == mock_blueprint(SIZE)


def test_round_trip():
    for n in (1, 2):
        bp = mock_blueprint(SIZE, storeys=n)
        assert Blueprint.from_dict(bp.to_dict()) == bp
        assert Blueprint.from_json(bp.to_json(indent=2)) == bp
        assert "notes" not in bp.to_dict() and bp.to_dict(notes=True)["notes"] == []
    rep = Blueprint.from_dict(MESSY).repaired(SIZE)
    again = Blueprint.from_dict(rep.to_dict(notes=True))
    assert again == rep


def test_schema_example_is_a_valid_blueprint():
    start = SCHEMA_TEXT.index("Example on")
    obj = extract_json(SCHEMA_TEXT[start:])
    bp = Blueprint.from_dict(obj)
    assert bp.validate(SIZE) == []
    assert {r.name for r in bp.rooms} == bp.reachable() == {"hall", "kitchen"}


# ---------------------------------------------------------------- repair
def test_repair_clips_to_canvas_and_roof_overhang():
    bp = Blueprint.from_dict(MESSY).repaired(SIZE)
    ov = bp.roof.overhang
    fp = bp.footprint
    assert 0 <= ov <= 2 and bp.roof.kind in ROOF_KINDS and bp.roof.axis in ("x", "z")
    assert fp.x0 >= ov and fp.z0 >= ov and fp.x1 <= 31 - ov and fp.z1 <= 31 - ov
    assert bp.total_height() <= 31
    assert all(MIN_STOREY_HEIGHT <= s.height <= MAX_STOREY_HEIGHT for s in bp.storeys)
    assert all(fp.intersect(r.rect) == r.rect for r in bp.rooms)
    assert bp.materials["wall"] == DEFAULT_MATERIALS["wall"]          # unresolvable block
    assert all(resolve_block(b) for b in bp.materials.values())
    assert any("unobtanium" in n for n in bp.notes)
    names = [r.name for r in bp.rooms]
    assert "great_hall" in names and "ghost" not in names and len(set(names)) == len(names)


def test_repair_overlaps_become_shared_walls():
    d = {"footprint": [4, 4, 27, 27], "storeys": [{"height": 4}],
         "rooms": [{"name": "a", "rect": [4, 4, 16, 27]},
                   {"name": "b", "rect": [10, 4, 27, 27]},       # overlaps a: cut to x 16..27
                   {"name": "c", "rect": [5, 5, 15, 15]}]}       # inside a: no valid cut
    bp = Blueprint.from_dict(d).repaired(SIZE)
    assert interiors_disjoint(bp)
    assert bp.room("b").rect == Rect(16, 4, 27, 27)
    assert "c" not in [r.name for r in bp.rooms]
    assert any("cut to" in n for n in bp.notes) and any("'c' dropped" in n for n in bp.notes)


def test_repair_connectivity_and_front_door():
    d = {"footprint": [6, 8, 25, 23], "storeys": [{"height": 4}],
         "rooms": [{"name": "hall", "rect": [6, 14, 15, 23]},
                   {"name": "kitchen", "rect": [15, 14, 25, 23]},
                   {"name": "bedroom", "rect": [6, 8, 25, 14]}]}
    bp = Blueprint.from_dict(d).repaired(SIZE)
    assert bp.reachable() == {"hall", "kitchen", "bedroom"}
    ext = [o for o in bp.openings if o.kind == "door" and OUTSIDE in o.between]
    assert len(ext) == 1 and ext[0].z == bp.footprint.z1          # front = z1 side
    assert front_side() == "z1"
    assert not bp.footprint.is_corner(ext[0].x, ext[0].z)


def test_repair_snaps_openings_and_keeps_windows_outside():
    bp = mock_blueprint(SIZE)
    d = bp.to_dict()
    d["openings"] = [{"kind": "door", "at": [11, 21], "between": ["hall", "outside"]},
                     {"kind": "window", "at": [15, 17], "between": ["hall", "kitchen"]},
                     {"kind": "door", "at": [14, 19], "between": ["hall", "kitchen"]}]
    rep = Blueprint.from_dict(d).repaired(SIZE)
    door = rep.openings[0]
    assert (door.x, door.z) == (11, 23)
    hk = next(o for o in rep.openings if o.between == ("hall", "kitchen"))
    assert hk.x == 15 and 15 <= hk.z <= 22
    win = next(o for o in rep.openings if o.kind == "window")
    assert win.between[1] == OUTSIDE and rep.footprint.on_border(win.x, win.z)


def test_repair_adds_a_stair_per_transition():
    bp = mock_blueprint(SIZE, storeys=2)
    d = bp.to_dict()
    d["stairs"] = []
    rep = Blueprint.from_dict(d).repaired(SIZE)
    assert len(rep.stairs) == 1 and rep.stairs[0].from_storey == 0
    assert rep.stairs[0].kind == "stairs"
    assert rep.reachable() == {r.name for r in rep.rooms}
    st = rep.stairs[0]
    hall_like = rep._room_at(0, st.x, st.z)
    for (x, _dy, z) in st.cells(rep.storey(0).height):
        assert hall_like.interior.contains(x, z)
    # three storeys -> one stair per transition
    d3 = dict(d, storeys=[{"height": 4}] * 3, rooms=[], openings=[])
    rep3 = Blueprint.from_dict(d3).repaired(SIZE)
    assert sorted(s.from_storey for s in rep3.stairs) == list(range(len(rep3.storeys) - 1))


def test_repair_limits_and_storey_height():
    bp = mock_blueprint(SIZE, storeys=2)
    rep = bp.repaired(SIZE, max_rooms=2, max_storeys=1)
    assert len(rep.storeys) == 1 and len(rep.rooms) <= 2 and not rep.stairs
    tall = bp.to_dict()
    tall["storeys"] = [{"height": 8}] * 6
    rep = Blueprint.from_dict(tall).repaired(SIZE)
    assert rep.total_height() <= 31 and len(rep.storeys) < 6
    assert any("taller than the canvas" in n for n in rep.notes)


@pytest.mark.parametrize("d", [MESSY,
                               {"rooms": [{"name": "x", "rect": [0, 0, 31, 31]}]},
                               {"footprint": [0, 0, 3, 3]},
                               {"footprint": [5, 5, 26, 26], "storeys": 3,
                                "roof": "hip"}])
def test_repair_is_idempotent(d):
    once = Blueprint.from_dict(d).repaired(SIZE)
    twice = once.repaired(SIZE)
    assert twice == once
    assert once.validate(SIZE) == []
    assert interiors_disjoint(once)
    canvas, report, _ = build(once)
    assert report.n_failed == 0 and canvas.clipped_writes == 0


def test_validate_does_not_mutate():
    bp = Blueprint.from_dict(MESSY)
    before = bp.to_dict(notes=True)
    problems = bp.validate(SIZE)
    assert problems and bp.to_dict(notes=True) == before


@pytest.mark.parametrize("n", [1, 2])
def test_mock_blueprint_is_a_repair_fixed_point(n):
    bp = mock_blueprint(SIZE, storeys=n)
    assert len(bp.storeys) == n
    assert bp.repaired(SIZE) == bp and bp.validate(SIZE) == []
    assert bp.reachable() == {r.name for r in bp.rooms}
    assert mock_blueprint((32, 16, 32), storeys=2).storeys.__len__() == 1   # H < 24


def test_default_blueprint_is_valid():
    bp = default_blueprint(SIZE, "a tiny shed")
    assert bp.validate(SIZE) == [] and bp.reachable() == {"main"}
    _, report, _ = build(bp)
    assert report.n_failed == 0 and report.blocks > 0


# ---------------------------------------------------------------- DSL
@pytest.mark.parametrize("n", [1, 2])
def test_shell_and_roof_dsl_run_clean(n):
    bp = mock_blueprint(SIZE, storeys=n)
    canvas, report, text = build(bp)
    assert report.n_failed == 0 and report.n_noop == 0 and canvas.clipped_writes == 0
    blocks = text.strip().splitlines()
    assert blocks[0].startswith("# shell: foundation storey 0")
    assert all(p in text for p in ("# shell: walls", "# shell: partitions",
                                   "# shell: openings", "# shell: ceiling", "# roof: gable"))
    ids = canvas.block_ids
    door_id = resolve_block(bp.materials["door"])[0]
    for (x, y, z) in bp.door_cells():
        assert ids[x, y, z] == door_id
    part_id = resolve_block(bp.materials["interior_wall"])[0]
    s0 = bp.storey(0)
    for z in range(15, 23):                                   # hall | kitchen wall at x=15
        if (15, z) not in [(o.x, o.z) for o in bp.openings]:
            assert ids[15, s0.floor_y + 2, z] == part_id
    for st in bp.stairs:
        base = bp.storey(st.from_storey)
        for (x, dy, z) in st.cells(base.height):
            assert ids[x, base.floor_y + dy, z] in STAIRS_IDS
        # stairwell: the upper slab above the two steps below the top is open
        for (x, _dy, z) in st.cells(base.height)[-3:-1]:
            assert ids[x, base.ceiling_y, z] == 0


def test_shell_parts_and_storey_filter():
    bp = mock_blueprint(SIZE, storeys=2)
    one = bp.shell_dsl(parts=("floors", "walls", "corners"), storeys=[1])
    assert "storey 0" not in one and "# shell: floors storey 1" in one
    assert set(SHELL_PARTS) >= {"foundation", "ceiling"}
    assert bp.shell_dsl(parts=()) == ""


@pytest.mark.parametrize("kind", ["gable", "hip", "pyramid", "flat", "none"])
def test_every_roof_kind_builds(kind):
    for fp in ([6, 8, 25, 23], [8, 8, 22, 22]):            # rectangular and square
        d = mock_blueprint(SIZE).to_dict()
        d["roof"] = {"kind": kind, "axis": "z", "overhang": 1}
        d["footprint"] = fp
        d["rooms"], d["openings"] = [], []
        bp = Blueprint.from_dict(d).repaired(SIZE)
        canvas, report, text = build(bp)
        assert report.n_failed == 0 and canvas.clipped_writes == 0
        assert (bp.roof_dsl(SIZE) == "") == (kind == "none")
        top = int(np.argwhere(canvas.block_ids != 0)[:, 1].max())
        assert top == bp.total_height()


# ---------------------------------------------------------------- geometry
def test_boxes_and_keep_clear():
    bp = mock_blueprint(SIZE, storeys=2)
    assert bp.room_box("hall") == Box(7, 0, 15, 14, 4, 22)
    assert bp.room_box("hall", walls=True, floor=False) == Box(6, 1, 14, 15, 4, 23)
    assert bp.storey_box(1) == Box(6, 5, 8, 25, 9, 23)
    clear = bp.keep_clear("hall")
    cells = {(x, y, z) for b in clear for x in range(b.x0, b.x1 + 1)
             for y in range(b.y0, b.y1 + 1) for z in range(b.z0, b.z1 + 1)}
    assert (10, 1, 22) in cells and (10, 2, 22) in cells        # inside the front door
    assert (14, 1, 18) in cells                                 # inside the kitchen door
    assert (8, 1, 17) in cells and (12, 4, 17) in cells         # the stair run
    up = bp.keep_clear("study")
    assert any(b.contains(12, 6, 17) for b in up)               # stair arrival upstairs
    assert len(bp.door_cells()) == 2 * 4 and len(bp.door_cells(1)) == 2


def test_order_rooms():
    bp = mock_blueprint(SIZE, storeys=2)
    assert [r.name for r in order_rooms(bp.rooms, "blueprint")] == [r.name for r in bp.rooms]
    assert [r.name for r in order_rooms(bp.rooms, "public_first")][:2] == ["hall", "kitchen"]
    sizes = [r.interior.area for r in order_rooms(bp.rooms, "size")]
    assert sizes == sorted(sizes, reverse=True)
    with pytest.raises(ValueError):
        order_rooms(bp.rooms, "random")


def test_adjacency_and_summary():
    bp = mock_blueprint(SIZE, storeys=2)
    adj = bp.adjacency()
    assert adj["hall"] == {"kitchen", "bedroom", OUTSIDE}
    assert bp.wall_axis(10, 23, 0) == "x" and bp.wall_axis(25, 18, 0) == "z"
    assert bp.wall_axis(6, 8, 0) is None and bp.wall_axis(10, 18, 0) is None
    assert len(bp.summary().splitlines()) <= 14 and "hall" in bp.summary()


# ---------------------------------------------------------------- images
def _png(data: bytes):
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def test_schematics_decode_and_highlight_changes_pixels():
    bp = mock_blueprint(SIZE, storeys=2)
    plain = _png(render_schematic(bp, 0, px=256))
    hi = _png(render_schematic(bp, 0, px=256, highlight="kitchen"))
    assert plain.shape == (256, 256, 3) and hi.shape == plain.shape
    assert (plain != hi).any()
    assert (hi == (230, 0, 0)).all(axis=-1).any()                 # the red outline
    both = _png(render_schematics(bp, px=128))
    assert both.shape == (128, 256, 3)
    grid = _png(render_grid(SIZE, footprint=bp.footprint, px=200))
    assert grid.shape == (200, 200, 3) and (grid < 128).any()


# ---------------------------------------------------------------- verifier additions
def rects_overlap_beyond_a_wall(bp: Blueprint) -> bool:
    """Stricter than interiors_disjoint: two rects may share at most a wall LINE."""
    for i, a in enumerate(bp.rooms):
        for b in bp.rooms[i + 1:]:
            hit = a.rect.intersect(b.rect)
            if a.storey == b.storey and hit is not None and hit.w >= 2 and hit.d >= 2:
                return True
    return False


def test_repair_cuts_a_two_thick_overlap_to_one_shared_wall():
    # hall 6..16 and kitchen 15..25: interiors are disjoint (7..15 vs 16..24) but each
    # wall runs through the other's interior, so no door could join them.
    d = {"footprint": [6, 8, 25, 23],
         "rooms": [{"name": "hall", "rect": [6, 8, 16, 23]},
                   {"name": "kitchen", "rect": [15, 8, 25, 23]}],
         "openings": [{"kind": "door", "at": [10, 23], "between": ["hall", "outside"]},
                      {"kind": "door", "at": [15, 15], "between": ["hall", "kitchen"]}]}
    bp = Blueprint.from_dict(d).repaired(SIZE)
    assert not rects_overlap_beyond_a_wall(bp)
    assert bp.room("hall").rect.shared_wall(bp.room("kitchen").rect)
    inner = [o for o in bp.openings if set(o.between) == {"hall", "kitchen"}]
    assert len(inner) == 1                                    # the planned door survives
    assert not [o for o in bp.openings if o.between == ("kitchen", OUTSIDE)]
    assert bp.repaired(SIZE) == bp


def test_repair_merges_double_walls_and_walls_one_off_the_outside():
    # Exclusive-bounds thinking: kitchen starts one block past the hall's wall, and
    # the bedroom stops one block short of the outer back wall.
    d = {"footprint": [6, 8, 25, 23],
         "rooms": [{"name": "hall", "rect": [6, 14, 15, 23]},
                   {"name": "kitchen", "rect": [16, 14, 25, 23]},
                   {"name": "bedroom", "rect": [6, 9, 25, 14]}]}
    bp = Blueprint.from_dict(d).repaired(SIZE)
    hall, kitchen = bp.room("hall").rect, bp.room("kitchen").rect
    assert hall.x1 == kitchen.x0 and hall.shared_wall(kitchen)   # one wall, not two
    assert bp.room("bedroom").rect.z0 == 8                   # on the outer wall now
    assert any("double wall" in n for n in bp.notes)
    assert bp.reachable() == {"hall", "kitchen", "bedroom"}
    outside = [o for o in bp.openings if OUTSIDE in o.between and o.kind == "door"]
    assert len(outside) == 1                                  # one front door, inner doors
    assert bp.repaired(SIZE) == bp and bp.validate(SIZE) == []


def test_repair_fuzz_rects_share_at_most_a_wall_and_stay_idempotent():
    import random

    for seed in range(300):
        rng = random.Random(seed)
        fx0, fz0 = rng.randint(0, 10), rng.randint(0, 10)
        fx1, fz1 = fx0 + rng.randint(3, 22), fz0 + rng.randint(3, 22)
        rooms = []
        for _ in range(rng.randint(0, 6)):
            x0, z0 = rng.randint(fx0 - 2, fx1), rng.randint(fz0 - 2, fz1)
            rooms.append({"name": rng.choice(["hall", "kitchen", "bed"]),
                          "storey": rng.randint(0, 1),
                          "rect": [x0, z0, x0 + rng.randint(2, 14), z0 + rng.randint(2, 14)]})
        d = {"footprint": [fx0, fz0, fx1, fz1], "rooms": rooms,
             "storeys": [{"height": 4}] * rng.randint(1, 2)}
        once = Blueprint.from_dict(d).repaired(SIZE)
        assert once.repaired(SIZE) == once, seed
        assert not rects_overlap_beyond_a_wall(once), seed


def test_keep_clear_guards_the_stairwell_in_the_upper_floor_slab():
    bp = mock_blueprint(SIZE, storeys=2)
    st = bp.stairs[0]
    up = bp.storey(1)
    cells = st.cells(bp.storey(0).height)
    hole = [(x, up.floor_y, z) for (x, _dy, z) in cells[-3:-1]]     # cleared slab cells
    top = (cells[-1][0], up.floor_y, cells[-1][2])                   # top step in the slab
    upper = next(r.name for r in bp.rooms_on(1)
                 if r.interior is not None and r.interior.contains(top[0], top[2]))
    boxes = bp.keep_clear(upper)
    for c in hole + [top]:
        assert any(b.contains(*c) for b in boxes), c
