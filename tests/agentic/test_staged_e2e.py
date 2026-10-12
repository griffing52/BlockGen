"""Integration of the staged methods (blueprint, floors, turnaround) into the suite.

What the per-package tests cannot see on their own: registration through the
``recipes`` package (no explicit module import), the suite's dream prefetch
skipping methods that make their own dreams, the staged-v1 battery resolving,
and the shipped recipes producing complete traces offline (contact sheet with a
``u_<unit>`` row under the selected build). Plus regressions for the problems
the integration pass found by looking at the mock contact sheets.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from blockgen.agentic.blueprint import OUTSIDE, Blueprint, mock_blueprint
from blockgen.agentic.core import lookup
from blockgen.agentic.episode import run_episode
from blockgen.agentic.kit import build_kit
from blockgen.agentic.providers_cli import (AutoMockProvider, _fit_mock_rooms, mock_furnishing,
                                            mock_roof)
from blockgen.agentic.recipe import load_battery, load_recipe
from blockgen.agentic.suite import _prefetches_dreams
from blockgen.agentic.trace import Trace

SIZE = (32, 32, 32)
ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- registry
def test_staged_methods_register_through_the_package_alone():
    """A fresh interpreter that imports nothing but the registry finds every staged
    component (before integration, ``turnaround`` was unknown unless some other
    module happened to import it first)."""
    code = ("from blockgen.agentic.core import lookup\n"
            "for k, n in [('method','blueprint'),('method','floors'),"
            "('method','turnaround'),('scorer','interior')]:\n"
            "    lookup(k, n)\n"
            "print('ok')\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("ok")


@pytest.mark.parametrize("method", ["blueprint", "floors", "turnaround"])
def test_staged_methods_opt_out_of_dream_prefetch(method):
    assert lookup("method", method).meta.get("prefetch_dreams") is False


def test_dreams_first_skips_methods_that_make_their_own_dreams():
    class _Kit:  # only what _prefetches_dreams reads
        def __init__(self, method):
            self.spec = type("S", (), {"method": method})()
    assert _prefetches_dreams(_Kit("dream")) is True
    assert _prefetches_dreams(_Kit("turnaround")) is False
    assert _prefetches_dreams(_Kit("blueprint")) is False
    assert _prefetches_dreams(_Kit("no_such_method")) is True   # fails later, clearly


# --------------------------------------------------------------------------- battery
def test_staged_v1_battery_resolves_with_its_controls():
    b = load_battery("staged-v1")
    names = [r.name for r in b.recipes]
    for arm in ("classic-matched", "classic_long", "dream", "dream_rm", "dream-multiview",
                "blueprint", "blueprint-nodream", "blueprint_extonly", "blueprint_shuffled",
                "blueprint_llmshell", "blueprint-image-plan", "floors", "turnaround",
                "turnaround_perview"):
        assert arm in names, arm
    by = {r.name: r for r in b.recipes}
    assert by["classic_long"].params["critique_rounds"] == 12
    assert by["dream_rm"].dream_style == "render_match"
    assert by["blueprint_extonly"].params["interior_dreams"] is False
    assert by["blueprint_shuffled"].params["room_dream_shuffle"] is True
    assert by["blueprint_llmshell"].params["shell"] == "llm"
    assert by["turnaround_perview"].params["mode"] == "per_view"
    # The overrides merge into the base params instead of replacing them.
    assert by["blueprint_shuffled"].params["max_rooms"] == by["blueprint"].params["max_rooms"]


def test_smoke_mock_battery_lists_every_staged_recipe():
    names = {r.name for r in load_battery("smoke-mock").recipes}
    assert {"blueprint", "blueprint-nodream", "blueprint-image-plan", "floors",
            "turnaround"} <= names


# --------------------------------------------------------------------------- repair
def test_upper_storey_exterior_door_becomes_a_window():
    """The mock 2-storey plan has 5 rooms; under max_rooms=4 the loft is dropped and
    the study->loft door used to become an exterior door on storey 1 (a door into
    thin air, visible on the schematic)."""
    bp = mock_blueprint(SIZE, storeys=2).repaired(SIZE, max_rooms=4, max_storeys=2)
    for o in bp.openings:
        if o.storey > 0 and OUTSIDE in o.between:
            assert o.kind == "window", o
    assert any("opens onto a drop" in n for n in bp.notes)
    again = bp.repaired(SIZE, max_rooms=4, max_storeys=2)
    assert again.to_json() == bp.to_json()                      # still a fixed point


def test_ground_floor_exterior_door_is_kept():
    bp = mock_blueprint(SIZE, storeys=2).repaired(SIZE, max_rooms=5, max_storeys=2)
    doors = [o for o in bp.openings if o.kind == "door" and OUTSIDE in o.between]
    assert doors and all(o.storey == 0 for o in doors)


# --------------------------------------------------------------------------- mock
@pytest.mark.parametrize("n", [0, 3, 4, 5])
def test_mock_plan_obeys_the_room_limit_and_survives_repair_unchanged(n):
    bp = mock_blueprint(SIZE, storeys=2)
    _fit_mock_rooms(bp, n)
    assert len(bp.rooms) <= n
    rep = bp.repaired(SIZE, max_rooms=n, max_storeys=2)
    assert len(rep.rooms) == len(bp.rooms)
    assert not any("kept the first" in note or "onto a drop" in note for note in rep.notes)
    assert rep.reachable() == {r.name for r in rep.rooms}


def _ask(text: str) -> str:
    from blockgen.agentic.providers import text_message
    return AutoMockProvider()._decide([text_message("system", "Output format: DSL"),
                                       text_message("user", text)])[1]


def test_mock_answers_a_blueprint_turn_within_its_limits():
    from blockgen.agentic import turns_staged as ts
    turn = (f"{ts.BLUEPRINT_TAG}: plan this building.\nCanvas: 32 x 32 x 32 blocks\n"
            "Limits: at most 2 storey(s); At most 3 rooms in total. ...")
    bp = Blueprint.from_json(_ask(turn))
    assert len(bp.rooms) == 3


def test_mock_answers_an_llm_shell_turn_with_the_plans_shell():
    bp = mock_blueprint(SIZE, storeys=1)
    turn = ("Build the SHELL of this building: a house\n\nCanvas: 32 x 32 x 32 blocks\n\n"
            "The floor plan as JSON (all coordinates inclusive, in blocks; room rects "
            "include their walls, neighbouring rooms share a wall line):\n" + bp.to_json())
    reply = _ask(turn)
    assert "# shell:" in reply or "walls" in reply
    assert reply.strip() != ""


def test_mock_roof_sits_on_the_footprint():
    text = mock_roof(5, 6, 25, 8, 23, 32)
    first = text.splitlines()[0].split()
    assert first[:2] == ["fill", "spruce_planks"]
    assert [int(v) for v in first[2:]] == [6, 5, 8, 25, 5, 23]
    assert all(int(line.split()[3]) < 32 for line in text.splitlines())


def test_mock_storey_walls_ring_with_a_front_door_and_keep_clear():
    keep = [(10, 10, 1, 3, 10, 10)]
    text = mock_furnishing(6, 25, 0, 4, 8, 23, walls=True, front_door=True, keep_clear=keep)
    sets = {tuple(int(v) for v in ln.split()[2:5]) for ln in text.splitlines()
            if ln.startswith("set cobblestone")}
    assert (6, 1, 8) in sets and (25, 4, 23) in sets           # ring corners
    assert (15, 1, 23) not in sets and (15, 2, 23) not in sets   # doorway on +z
    assert (15, 3, 23) in sets                                  # lintel above it
    assert not any(x == 10 and z == 10 for (x, _y, z) in sets)


# --------------------------------------------------------------------------- end to end
def _run(recipe: str, tmp_path: Path, **params):
    spec = load_recipe(recipe)
    if params:
        import dataclasses
        spec = dataclasses.replace(spec, params={**spec.params, **params})
    kit = build_kit(spec, mock=True)
    trace = Trace.for_kit(tmp_path / recipe, kit)
    res = run_episode(kit, "a brick house", trace=trace)
    assert not res.truncated, res.error
    return res, tmp_path / recipe


@pytest.mark.slow
@pytest.mark.parametrize("recipe,unit_prefix", [("blueprint", "u_"), ("floors", "u_storey"),
                                                ("blueprint-nodream", "u_")])
def test_selected_build_carries_every_unit_render(recipe, unit_prefix, tmp_path):
    """The contact sheet's u_<unit> rows used to be blank under the roof and final
    columns, including the boxed (returned) build."""
    res, root = _run(recipe, tmp_path)
    best = res.attempts[res.best_idx]
    units = [k for k in best.renders if k.startswith(unit_prefix)]
    assert units, sorted(best.renders)
    assert (root / "contact.png").exists()
    score = json.loads((root / "result.json").read_text())
    assert score


@pytest.mark.slow
def test_floors_mock_builds_walls_on_every_storey(tmp_path):
    res, _ = _run("floors", tmp_path)
    best = res.attempts[res.best_idx]
    s = best.world.to_structure(crop=False)
    occ = s.occupied_mask
    # Wall ring present at mid-height of both storeys (y 2 and y 7 in the mock plan).
    assert occ[6, 2, 12] and occ[6, 7, 12]


@pytest.mark.slow
def test_llm_shell_arm_builds_on_the_plan_under_the_mock(tmp_path):
    res, _ = _run("blueprint", tmp_path, shell="llm")
    shell = [a for a in res.attempts if a.stage == "shell"][0]
    assert shell.blocks > 1000          # the plan's shell, not the 206-block demo hut


@pytest.mark.slow
def test_turnaround_makes_one_sheet_image(tmp_path):
    res, _ = _run("turnaround", tmp_path)
    assert "sheet" in res.dreams
    assert {"iso", "back", "left", "right"} <= set(res.dreams)
