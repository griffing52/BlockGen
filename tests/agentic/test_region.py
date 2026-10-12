"""Region-scoped execution helpers: boxes, regions, DSL translation, restoring the
outside of a region and charging discarded writes to the model's own lines."""

from __future__ import annotations

import numpy as np
import pytest

from blockgen.agentic.core import Lowered, MinecraftWorld
from blockgen.agentic.dsl import run_program
from blockgen.agentic.region import (COORD_PARAMS, Box, Region, clipped_issues,
                                     region_structure, restore_outside, translate_dsl,
                                     translate_lowered, world_arrays)

SIZE = (32, 32, 32)


def world(program: str = "") -> MinecraftWorld:
    w = MinecraftWorld.empty(SIZE)
    if program:
        w.run(Lowered(program, program))
    return w


# ---------------------------------------------------------------- Box / Region
def test_box_parse_and_normalize():
    assert Box.of(5, 6, 7, 1, 2, 3) == Box(1, 2, 3, 5, 6, 7)
    assert Box.parse([5, 6, 7, 1, 2, 3]) == Box(1, 2, 3, 5, 6, 7)
    assert Box.parse(["1", "2", "3.0", 4, 5, 6]) == Box(1, 2, 3, 4, 5, 6)
    assert Box.parse({"x0": 4, "y0": 5, "z0": 6, "x1": 1, "y1": 2, "z1": 3}) == Box(1, 2, 3, 4, 5, 6)
    for bad in ([1, 2, 3], {"x0": 1}, "box", [1, 2, 3, 4, 5, "x"]):
        with pytest.raises(ValueError):
            Box.parse(bad)


def test_box_geometry():
    b = Box(2, 1, 3, 4, 2, 6)
    assert b.shape == (3, 2, 4) and b.volume == 24 and b.origin == (2, 1, 3)
    assert b.contains(2, 2, 6) and not b.contains(5, 1, 3)
    assert b.describe() == "x 2..4, y 1..2, z 3..6" and b.to_list() == [2, 1, 3, 4, 2, 6]
    assert Box(-3, 0, 0, 2, 40, 2).clip(SIZE) == Box(0, 0, 0, 2, 31, 2)
    assert Box(40, 0, 0, 50, 1, 1).clip(SIZE) is None
    assert b.expand(1, 0, 2) == Box(1, 1, 1, 5, 2, 8)
    m = b.mask(SIZE)
    assert m.shape == SIZE and m.sum() == 24 and m[2, 1, 3] and not m[5, 1, 3]


def test_region_mask_deny_and_whole():
    r = Region("room", (Box(0, 0, 0, 3, 3, 3),), (Box(1, 1, 1, 1, 2, 1),))
    m = r.mask(SIZE)
    assert m.sum() == 64 - 2 and not m[1, 1, 1] and m[0, 0, 0]
    assert r.origin == (0, 0, 0)
    assert "except x 1..1, y 1..2, z 1..1" in r.describe()
    w = Region.whole(SIZE, deny=[Box(0, 0, 0, 0, 0, 0)])
    assert w.mask(SIZE).sum() == 32 ** 3 - 1 and w.name == "canvas"


# ---------------------------------------------------------------- translation
def test_coord_params_cover_every_coordinate_command():
    want = {"set": 3, "fill": 6, "clear": 6, "box": 6, "walls": 6, "replace": 6, "line": 6,
            "sphere": 3, "cylinder": 3, "pyramid": 3, "gable": 6, "copy": 6, "paste": 3,
            "stack": 6}
    for cmd, n in want.items():
        assert len(COORD_PARAMS[cmd]) == n, cmd
        assert {ax for _, ax in COORD_PARAMS[cmd]} == {0, 1, 2}


@pytest.mark.parametrize("line,want", [
    ("set oak_planks 1 2 3", "set oak_planks 11 22 33"),
    ("fill stone 0 0 0 1 1 1", "fill stone 10 20 30 11 21 31"),
    ("clear 0 0 0 1 1 1", "clear 10 20 30 11 21 31"),
    ("box stone 0 0 0 2 2 2 thickness=1", "box stone 10 20 30 12 22 32 thickness=1"),
    ("walls stone 0 1 0 4 3 4", "walls stone 10 21 30 14 23 34"),
    ("replace stone glass 0 0 0 1 1 1", "replace stone glass 10 20 30 11 21 31"),
    ("line oak_log 0 0 0 0 3 0 thickness=2", "line oak_log 10 20 30 10 23 30 thickness=2"),
    ("sphere glass 4 4 4 2", "sphere glass 14 24 34 2"),
    ("cylinder stone 4 0 4 2 5 axis=y", "cylinder stone 14 20 34 2 5 axis=y"),
    ("pyramid sandstone 5 6 5 3", "pyramid sandstone 15 26 35 3"),
    ("gable oak_stairs 0 4 0 6 4 6 axis=x overhang=1",
     "gable oak_stairs 10 24 30 16 24 36 axis=x overhang=1"),
    ("copy 0 0 0 1 1 1", "copy 10 20 30 11 21 31"),
    ("paste 3 0 3 rotate=90", "paste 13 20 33 rotate=90"),
    ("stack 0 0 0 1 1 1 +x 3", "stack 10 20 30 11 21 31 +x 3"),
    # key=value coordinates, a block state containing '=', and a leading slash
    ("fill stone x1=0 y1=0 z1=0 x2=1 y2=1 z2=1", "fill stone x1=10 y1=20 z1=30 x2=11 y2=21 z2=31"),
    ("set oak_stairs[facing=north] z=3 x=1 y=2", "set oak_stairs[facing=north] z=33 x=11 y=22"),
    ("//set stone 1 1 1", "//set stone 11 21 31"),
    ("  set stone 1 1 1  # L7", "  set stone 11 21 31  # L7"),
])
def test_translate_dsl_every_command(line, want):
    assert translate_dsl(line, (10, 20, 30)) == want


def test_translate_dsl_keeps_comments_and_unparseable_lines_aligned():
    text = "# a comment\n\nHere is the build:\nset stone 1 1 1  # L4\nset stone one two three\n"
    out = translate_dsl(text, (1, 0, 0))
    assert out.split("\n") == ["# a comment", "", "Here is the build:",
                               "set stone 2 1 1  # L4", "set stone one two three", ""]
    assert translate_dsl(text, (0, 0, 0)) == text


def test_translated_program_builds_the_shifted_build():
    prog = ("fill cobblestone 0 0 0 4 0 4\nwalls oak_planks 0 1 0 4 2 4\n"
            "set glass_pane 2 2 0\ngable oak_stairs 0 3 0 4 3 4 axis=x\n"
            "copy 0 0 0 4 3 4\npaste 6 0 0\n")
    a, _, _ = run_program(prog, size=SIZE)
    b, _, _ = run_program(translate_dsl(prog, (3, 2, 5)), size=SIZE)
    assert np.array_equal(b.block_ids[3:, 2:, 5:], a.block_ids[:-3, :-2, :-5])
    assert b.block_count() == a.block_count()


def test_translate_lowered():
    low = Lowered("src", "set stone 0 0 0  # L1", [], 1)
    out = translate_lowered(low, (1, 1, 1))
    assert out.program_text == "set stone 1 1 1  # L1" and out.source == "src"
    assert out.n_source_lines == 1


# ---------------------------------------------------------------- restore / issues
def test_world_arrays_needs_a_canvas():
    ids, data = world_arrays(world())
    assert ids.shape == SIZE
    with pytest.raises(TypeError, match="needs a world with a .canvas"):
        world_arrays(object())


def test_restore_outside_reverts_only_outside():
    before = world("fill stone 10 0 10 12 0 12\n")
    after = before.clone()
    after.run(Lowered("x", "fill oak_planks 0 0 0 5 0 5\nfill glass 20 0 20 21 0 21\n"))
    region = Region("r", (Box(0, 0, 0, 5, 3, 5),))
    rev = restore_outside(after, before, region)
    assert rev.sum() == 4                                     # the glass, reverted
    ids, _ = world_arrays(after)
    assert ids[20, 0, 20] == 0 and ids[11, 0, 11] != 0 and ids[0, 0, 0] != 0
    # before=None: everything outside the region becomes air
    empty = world("fill stone 0 0 0 31 0 31\n")
    restore_outside(empty, None, region)
    assert empty.block_count() == 36


def test_clipped_issues_attribute_writes_to_lines():
    start = world()
    prog = ("# furnish\n"
            "fill red_wool 1 1 1 3 1 3\n"           # inside
            "fill oak_planks 0 1 0 7 1 0\n"         # 4 of 8 outside (x 4..7)
            "set lantern 9 1 9\n")                  # outside
    region = Region("u_room", (Box(0, 0, 0, 3, 3, 3),))
    issues = clipped_issues(start, Lowered(prog, prog), region)
    assert [(i.line_no, i.severity) for i in issues] == [(3, "warning"), (4, "warning")]
    assert issues[0].message.startswith("4 block(s) outside x 0..3, y 0..3, z 0..3")
    assert issues[1].message.startswith("1 block(s)") and "set lantern" in issues[1].text


def test_clipped_issues_map_tagged_lines_and_cap():
    # A python-style lowering: every op tagged with its SOURCE line.
    src = "for x in range(10):\n    place('stone', x, 1, 9)\nplace('glass', 1, 1, 1)\n"
    low_text = "\n".join(f"set stone {x} 1 9  # L2" for x in range(10)) + "\nset glass 1 1 1  # L3"
    low = Lowered(src, low_text, [], 3)
    region = Region("r", (Box(0, 0, 0, 3, 3, 3),))
    issues = clipped_issues(world(), low, region)
    assert len(issues) == 1 and issues[0].line_no == 2
    assert issues[0].message.startswith("10 block(s)") and "place('stone'" in issues[0].text
    many = "\n".join(f"set stone {x} 5 5" for x in range(12))
    capped = clipped_issues(world(), Lowered(many, many), region, max_lines=3)
    assert len(capped) == 4 and "9 more line(s)" in capped[-1].message


def test_region_structure_crops_and_cuts():
    w = world("fill stone 0 0 0 9 5 9\nset glass 20 0 20\n")
    s = region_structure(w, Box(2, 0, 2, 4, 5, 4), cut_y=2)
    assert s.shape == (3, 3, 3) and int(s.occupied_mask.sum()) == 27
    full = w.to_structure(crop=False)
    assert region_structure(full, Box(2, 0, 2, 4, 5, 4)).shape == (3, 6, 3)
    assert int(region_structure(w, Box(25, 0, 25, 30, 3, 30)).occupied_mask.sum()) == 0


# ---------------------------------------------------------------- verifier additions
def test_clipped_issues_quote_the_models_line_when_translated():
    src = "set lantern 0 0 0\nfill stone 0 0 0 9 0 0\n"
    low = translate_lowered(Lowered(src, src, [], 2), (1, 1, 1))
    issues = clipped_issues(world(), low, Region("r", (Box(1, 1, 1, 4, 3, 4),)))
    assert [(i.line_no, i.text) for i in issues] == [(2, "fill stone 0 0 0 9 0 0")]
    assert issues[0].message.startswith("6 block(s)")


def test_region_structure_negative_cut_keeps_nothing():
    w = world("fill stone 0 0 0 3 3 3\n")
    assert int(region_structure(w, Box(0, 0, 0, 3, 3, 3), cut_y=-3).occupied_mask.sum()) == 0
    assert int(region_structure(w, Box(0, 0, 0, 3, 3, 3), cut_y=0).occupied_mask.sum()) == 16
