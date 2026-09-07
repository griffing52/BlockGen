"""DSL tests: block resolution, parsing, execution, and the failure paths.

The executor's contract is what the agent loop depends on — a broken line must be
*reported*, not raised, and everything else must still run — so the error paths
get as much coverage here as the happy ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from blockgen.agentic.blockstate import (PALETTE, UnknownBlockError, parse_block,
                                         resolve_block)
from blockgen.agentic.canvas import Canvas, structure_to_canvas
from blockgen.agentic.dsl import COMMANDS, parse_program, run_program


# --- block resolution ------------------------------------------------------
def test_every_palette_name_resolves():
    """The palette is advertised to the model as known-good; prove it is."""
    unresolved = [n for n in PALETTE if resolve_block(n) is None]
    assert unresolved == []


def test_palette_is_not_all_stone():
    """A palette that silently collapses to stone would render as a grey blob."""
    pairs = {resolve_block(n) for n in PALETTE}
    assert len(pairs) > 30


def test_parse_block_states():
    ref = parse_block("minecraft:oak_stairs[facing=north,half=top]")
    assert ref.name == "oak_stairs"
    assert ref.props == {"facing": "north", "half": "top"}


def test_stair_facing_becomes_legacy_bits():
    east = resolve_block("oak_stairs[facing=east]")
    north = resolve_block("oak_stairs[facing=north]")
    top = resolve_block("oak_stairs[facing=north,half=top]")
    assert east[1] & 3 == 0 and north[1] & 3 == 3
    assert top[1] & 4 == 4


def test_log_axis_becomes_legacy_bits():
    assert (resolve_block("oak_log[axis=x]")[1] >> 2) & 3 == 1
    assert (resolve_block("oak_log[axis=z]")[1] >> 2) & 3 == 2


def test_unknown_block_raises():
    with pytest.raises(UnknownBlockError):
        resolve_block("definitely_not_a_block")


def test_air_names():
    assert resolve_block("air") == (0, 0)


# --- parsing ---------------------------------------------------------------
def test_parse_tolerates_fences_and_prose():
    prog = parse_program("Here is the build:\n```\nfill stone 0 0 0 1 1 1\n```\nDone!")
    assert len(prog.calls) == 1
    assert prog.n_skipped == 2  # the prose lines, not the fences


def test_worldedit_style_slashes_are_accepted():
    prog = parse_program("//set stone 1 1 1")
    assert len(prog.calls) == 1 and prog.calls[0].spec.name == "set"


def test_keyword_arguments_and_defaults():
    prog = parse_program("box stone 0 0 0 5 5 5 thickness=2")
    assert prog.calls[0].args["thickness"] == 2
    prog = parse_program("box stone 0 0 0 5 5 5")
    assert prog.calls[0].args["thickness"] == 1


def test_block_state_is_not_mistaken_for_a_keyword():
    prog = parse_program("set oak_stairs[facing=north] 1 1 1")
    assert prog.calls[0].args["block"] == "oak_stairs[facing=north]"
    assert prog.calls[0].args["x"] == 1


def test_missing_argument_is_an_issue_not_an_exception():
    prog = parse_program("fill stone 0 0 0")
    assert not prog.ok
    assert "missing argument" in prog.issues[0].message


def test_extra_argument_is_reported():
    prog = parse_program("set stone 1 1 1 9")
    assert not prog.ok and "extra argument" in prog.issues[0].message


def test_comments_and_blank_lines_are_free():
    prog = parse_program("# a comment\n\nfill stone 0 0 0 1 1 1  # trailing\n")
    assert len(prog.calls) == 1 and not prog.issues


# --- execution -------------------------------------------------------------
def test_fill_counts_and_places():
    canvas, report, _ = run_program("fill stone 0 0 0 3 1 2", size=(8, 8, 8))
    assert report.blocks == 4 * 2 * 3
    assert canvas.block_ids[0, 0, 0] == 1


def test_walls_are_hollow_and_open_topped():
    canvas, _, _ = run_program("walls stone 0 0 0 5 3 5", size=(8, 8, 8))
    assert canvas.block_ids[2, 2, 2] == 0     # interior empty
    assert canvas.block_ids[0, 2, 2] != 0     # side wall present
    assert canvas.block_ids[2, 3, 2] == 0     # no ceiling


def test_box_is_closed_on_all_six_faces():
    canvas, _, _ = run_program("box stone 0 0 0 5 5 5", size=(8, 8, 8))
    assert canvas.block_ids[2, 0, 2] != 0 and canvas.block_ids[2, 5, 2] != 0
    assert canvas.block_ids[2, 2, 2] == 0


def test_clear_removes_blocks():
    canvas, _, _ = run_program(
        "fill stone 0 0 0 3 3 3\nclear 1 1 1 2 2 2", size=(8, 8, 8))
    assert canvas.block_ids[1, 1, 1] == 0 and canvas.block_ids[0, 0, 0] != 0


def test_replace_matches_the_whole_id_family():
    canvas, report, _ = run_program(
        "fill oak_planks 0 0 0 3 3 3\nreplace oak_planks glass 0 0 0 1 1 1",
        size=(8, 8, 8))
    assert canvas.block_ids[0, 0, 0] == 20      # glass
    assert canvas.block_ids[3, 3, 3] != 20
    assert report.n_failed == 0


def test_replace_any():
    canvas, _, _ = run_program(
        "fill oak_planks 0 0 0 2 2 2\nfill stone 0 0 0 0 2 2\n"
        "replace any glass 0 0 0 2 2 2", size=(8, 8, 8))
    assert set(np.unique(canvas.block_ids[canvas.occupied_mask]).tolist()) == {20}


def test_copy_paste_duplicates_geometry():
    canvas, report, _ = run_program(
        "fill stone 0 0 0 1 1 1\ncopy 0 0 0 1 1 1\npaste 4 0 0", size=(8, 8, 8))
    assert canvas.block_ids[4, 0, 0] != 0 and report.n_failed == 0


def test_paste_without_copy_is_reported_not_raised():
    _, report, _ = run_program("paste 1 1 1", size=(8, 8, 8))
    assert report.n_failed == 1 and "clipboard" in report.issues[0].message


def test_stack_repeats_a_region():
    canvas, _, _ = run_program(
        "fill stone 0 0 0 1 1 1\nstack 0 0 0 1 1 1 +x 2", size=(16, 8, 8))
    assert canvas.block_ids[2, 0, 0] != 0 and canvas.block_ids[4, 0, 0] != 0


def test_gable_roof_is_a_pitched_shell():
    canvas, _, _ = run_program("gable oak_planks 0 0 0 7 0 7 axis=x", size=(16, 16, 16))
    occ = canvas.occupied_mask
    heights = [int(np.argwhere(occ[:, :, z])[:, 1].max()) if occ[:, :, z].any() else -1
               for z in range(8)]
    # z=0 is the eave (roof block at y=0 plus the riser closing the first step);
    # the ridge is the tallest row, in the middle of the span.
    assert heights[0] == 1 and max(heights) == 3
    assert heights[3] == max(heights)


def test_sphere_and_cylinder_place_blocks():
    canvas, report, _ = run_program(
        "sphere stone 8 8 8 3\ncylinder stone 2 0 2 1 5", size=(16, 16, 16))
    assert report.n_failed == 0 and report.blocks > 50


def test_out_of_bounds_is_clipped_and_counted():
    canvas, report, _ = run_program("fill stone 0 0 0 20 0 0", size=(8, 8, 8))
    assert report.blocks == 8
    assert report.clipped_writes == 13


def test_unknown_block_is_a_per_line_error_and_the_rest_still_runs():
    canvas, report, _ = run_program(
        "fill notablock 0 0 0 1 1 1\nfill stone 0 0 0 1 1 1", size=(8, 8, 8))
    assert report.n_failed == 1 and report.blocks == 8
    assert report.issues[0].line_no == 1


def test_noop_command_is_flagged_as_a_warning():
    _, report, _ = run_program("fill stone 100 100 100 101 101 101", size=(8, 8, 8))
    assert report.n_noop == 1
    assert report.issues[0].severity == "warning"


def test_report_summary_mentions_failures():
    _, report, _ = run_program("fill notablock 0 0 0 1 1 1", size=(8, 8, 8))
    assert "1 failed" in report.summary()


# --- canvas ----------------------------------------------------------------
def test_to_structure_round_trips_through_structure_to_canvas():
    canvas, _, _ = run_program("fill stone 1 1 1 3 3 3", size=(8, 8, 8))
    structure = canvas.to_structure()
    again = structure_to_canvas(structure)
    assert again.block_count() == canvas.block_count()


def test_canvas_rejects_a_bad_size():
    with pytest.raises(ValueError):
        Canvas((0, 4, 4))


def test_palette_counts_are_sorted_by_frequency():
    canvas, _, _ = run_program(
        "fill stone 0 0 0 3 3 3\nfill glass 0 0 0 0 0 1", size=(8, 8, 8))
    counts = canvas.palette_counts()
    assert counts[0][1] > counts[1][1]


# --- registry --------------------------------------------------------------
def test_every_command_example_parses_and_runs():
    """The examples are shown to the model verbatim — they must be valid."""
    for name, spec in COMMANDS.items():
        canvas = Canvas((32, 32, 32))
        # `paste` needs a clipboard, so prime one for every example.
        run_program("fill stone 0 0 0 2 2 2\ncopy 0 0 0 2 2 2", canvas=canvas)
        prog = parse_program(spec.example, strict=True)
        assert prog.ok, f"{name}: {[i.render() for i in prog.issues]}"
        _, report, _ = run_program(spec.example, canvas=canvas)
        assert report.n_failed == 0, f"{name}: {[i.render() for i in report.issues]}"


# --- in-context examples ---------------------------------------------------
def test_in_context_examples_execute_cleanly_and_are_connected():
    """The example programs are what the model imitates, so they must be
    exemplary: no failed lines, no no-ops, and ONE connected component — the
    property every generated build is then scored on."""
    from blockgen.agentic.examples import EXAMPLES
    from blockgen.eval.validity import _components
    for ex in EXAMPLES:
        canvas, report, _ = run_program(ex.program, size=(48, 48, 48))
        assert report.n_failed == 0, f"{ex.caption}: {[i.render() for i in report.issues]}"
        assert report.n_noop == 0, f"{ex.caption}: {[i.render() for i in report.issues]}"
        comps = _components(canvas.to_structure().occupied_mask)
        assert len(comps) == 1, f"{ex.caption}: {len(comps)} components"


def test_gable_roof_connects_to_the_walls_it_sits_on():
    """Stepped roofs touch only diagonally without risers; that would read as a
    fragmented build under the 6-connectivity validity metric."""
    from blockgen.eval.validity import _components
    canvas, _, _ = run_program(
        "walls oak_planks 0 0 0 9 4 7\nfill oak_planks 0 5 0 9 5 7\n"
        "gable oak_stairs 0 5 0 9 5 7 axis=x", size=(24, 24, 24))
    assert len(_components(canvas.to_structure().occupied_mask)) == 1


def test_gable_riser_can_be_turned_off():
    with_riser, _, _ = run_program("gable stone 0 0 0 9 0 7", size=(24, 24, 24))
    without, _, _ = run_program("gable stone 0 0 0 9 0 7 riser=false", size=(24, 24, 24))
    assert with_riser.block_count() > without.block_count()


def test_select_examples_is_deterministic_and_topical():
    from blockgen.agentic.examples import select_examples
    assert select_examples("a stone watchtower on a hill", 1)[0].caption.startswith(
        "Round stone watchtower")
    assert select_examples("something unrecognizable", 1)[0].caption.startswith("Small oak")
    assert select_examples("anything", 0) == []
