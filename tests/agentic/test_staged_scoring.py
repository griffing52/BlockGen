"""Scorer ``interior`` (recipes/staged_scoring.py): rooms and furnishing from voxels."""

from __future__ import annotations

import pytest

from blockgen.agentic import core
from blockgen.agentic.core import Lowered, MinecraftWorld
from blockgen.agentic.recipes import staged_scoring
from blockgen.agentic.recipes.staged_scoring import interior, interior_stats, interior_value

from .fakes import make_attempt

#: Two storeys on a 15 x 11 footprint, two ground-floor rooms split by a partition
#: with a door, one upper room, a ceiling slab. No furniture.
SHELL = """fill cobblestone 0 0 0 14 0 10
walls stone_bricks 0 1 0 14 4 10
fill oak_planks 7 1 1 7 4 9
set oak_door 7 1 5
set oak_door 7 2 5
fill oak_planks 0 5 0 14 5 10
walls stone_bricks 0 6 0 14 9 10
fill oak_planks 0 10 0 14 10 10
"""
FURNITURE = """set crafting_table 2 1 2
set lantern 3 1 2
fill red_carpet 9 1 2 12 1 4
set oak_stairs 2 6 2
set bookshelf 3 6 2
set bookshelf 4 6 2
set chest 10 6 8
"""


def stats(prog: str):
    w = MinecraftWorld.empty((32, 32, 32))
    rep = w.run(Lowered(prog, prog))
    assert rep.n_failed == 0
    st = interior_stats(w.to_structure().occupied_mask)
    return st, interior_value(st)


def test_solid_block_and_open_platform_score_zero():
    for prog in ("fill stone 0 0 0 9 9 9", "fill stone 0 0 0 15 0 15",
                 "fill stone 0 0 0 9 0 9\nfill stone 0 6 0 9 6 9\nline oak_log 0 1 0 0 5 0"):
        st, v = stats(prog)
        assert v == 0.0, (prog, st)
        assert st["rooms"] == 0


def test_bare_shell_has_rooms_but_no_furnishing():
    st, v = stats(SHELL)
    assert st["rooms"] == 3
    assert st["furnish"] == 0          # walls, partition, door and slabs never count
    assert v == pytest.approx(0.5)


def test_two_storey_shell_with_furniture_scores_above_half():
    st, v = stats(SHELL + FURNITURE)
    assert st["rooms"] == 3 and st["furnish"] > 0
    assert v > 0.5


def test_open_doorways_still_count_the_rooms():
    """Doorways left as air connect the rooms to the outside; the covered-air test
    keeps them (only the cells lined up with an opening drop out)."""
    prog = (SHELL + FURNITURE).replace("set oak_door 7 1 5\nset oak_door 7 2 5\n",
                                       "clear 7 1 5 7 2 5\n") + "clear 3 1 10 3 2 10\n"
    st, v = stats(prog)
    assert st["rooms"] >= 2
    assert v > 0.5


def test_floorless_hut_with_an_open_door_counts_its_furniture():
    st, v = stats("walls oak_planks 0 0 0 6 3 6\nclear 3 0 6 3 1 6\n"
                  "fill oak_planks 0 4 0 6 4 6\nset crafting_table 1 0 1\nset chest 5 0 1")
    assert st["rooms"] == 1 and st["furnish"] == 2
    assert 0.25 < v <= 1.0


def test_value_formula():
    assert interior_value({"rooms": 0, "enclosed_air": 0, "furnish": 0}) == 0.0
    assert interior_value({"rooms": 4, "enclosed_air": 100, "furnish": 3}) == pytest.approx(0.75)
    assert interior_value({"rooms": 1, "enclosed_air": 100, "furnish": 50}) == pytest.approx(0.75)


def test_registered_scorer_measures_an_attempt():
    entry = core.lookup("scorer", "interior")
    assert entry.load() is interior
    assert core.meta("scorer", "interior").get("needs_dream") is False
    assert core.meta("scorer", "interior").get("needs_renders") is False
    term = interior(None, make_attempt(SHELL + FURNITURE))
    assert term.value > 0.5 and "3 room(s)" in term.note
    assert interior(None, make_attempt("")).value == 0.0


def test_mock_blueprint_shell_plus_furniture():
    """The deterministic 2-storey mock blueprint (WP-A) plus a little furniture per
    room scores above 0.5."""
    blueprint = pytest.importorskip("blockgen.agentic.blueprint")
    bp = blueprint.mock_blueprint((32, 32, 32), storeys=2)
    prog = bp.shell_dsl() + "\n" + bp.roof_dsl()
    for room in bp.rooms:
        box = bp.room_box(room.name)
        prog += f"\nset crafting_table {box.x0} {box.y0 + 1} {box.z0}"
    w = MinecraftWorld.empty((32, 32, 32))
    w.run(Lowered(prog, prog))
    st = interior_stats(w.to_structure().occupied_mask)
    assert st["rooms"] >= 2
    assert interior_value(st) > 0.5


def test_module_exports():
    assert set(staged_scoring.__all__) == {"interior", "interior_stats", "interior_value"}
