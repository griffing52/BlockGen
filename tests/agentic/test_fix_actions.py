"""Regression tests for the sandbox/action-space review fixes (2026-10-03).

Each test pins one finding: parent-side argument bounds and the work budget
(the child's limits never cover the executor), OverflowError on huge integers,
the child output cap, deduplicated loop issues, and warnings for ignored states.
"""

from __future__ import annotations

import time

from blockgen.agentic.actions import (BlockActions, MixedActions, dedupe_issues,
                                      remap_issues, state_warning)
from blockgen.agentic.core import Lowered, MinecraftWorld
from blockgen.agentic.dsl import Issue, run_program
from blockgen.agentic.pyapi import MAX_STR, PythonActions

SIZE = (32, 32, 32)


def _ops(low):
    return [l for l in low.program_text.splitlines() if l.strip()]


# -- 1. parent-side bounds: emitted ops cannot stall the parent's executor --------
def test_huge_line_is_rejected_at_lowering_and_parent_run_is_fast():
    space = PythonActions(timeout_s=2, size=SIZE)
    for code in ('line("oak_log", 0, 0, 0, 10**6, 0, 0)',
                 'stack(0, 0, 0, 0, 0, 0, "+x", 10**6)',
                 'sphere("stone", 16, 16, 16, 10**5)'):
        low = space.lower(code)
        assert _ops(low) == [], code
        assert any(i.severity == "error" and "skipped" in i.message for i in low.issues), code
    t0 = time.time()
    run_program(space.lower('line("oak_log", 0, 0, 0, 10**6, 0, 0)').program_text, size=SIZE)
    assert time.time() - t0 < 1.0


def test_in_range_coordinates_still_lower():
    space = PythonActions(size=SIZE)
    low = space.lower('line("oak_log", -5, 0, 0, 40, 0, 0)\nsphere("stone", 16, 16, 16, 20)')
    assert len(_ops(low)) == 2 and not [i for i in low.issues if i.severity == "error"]


def test_work_budget_drops_ops_past_it_with_one_issue():
    space = PythonActions(size=SIZE, max_work=100_000)
    low = space.lower('for i in range(5000):\n    sphere("stone", 16, 16, 16, 20)\n')
    # 41^3 = 68921 per sphere: the first fits, the second crosses the budget
    assert len(_ops(low)) == 1
    budget = [i for i in low.issues if "voxel writes" in i.message]
    assert len(budget) == 1 and budget[0].line_no == 2


# -- 2. huge integers are Issues, not OverflowError -------------------------------
def test_huge_int_in_python_is_an_issue():
    low = PythonActions().lower('place("stone", 10**400, 0, 0)')
    assert _ops(low) == []
    assert low.issues and low.issues[0].line_no == 1
    assert "far outside" in low.issues[0].message


def test_huge_int_in_blocks_is_an_issue():
    low = BlockActions().lower("stone " + "9" * 400 + " 0 0\nstone 1 0 0")
    assert [i.line_no for i in low.issues] == [1]
    assert low.program_text.splitlines()[1] == "set stone 1 0 0"


def test_huge_int_in_mixed_blocks_section_is_an_issue():
    low = MixedActions().lower("```blocks\nstone " + "9" * 400 + " 0 0\n```\n")
    assert [i.line_no for i in low.issues] == [2]


# -- 3. child output is capped -----------------------------------------------------
def test_long_string_argument_is_rejected_in_child():
    low = PythonActions().lower(f's = "stone[" + "a" * {MAX_STR} + "]"\nplace(s, 0, 0, 0)\n')
    assert _ops(low) == []
    assert any(i.line_no == 2 and "characters long" in i.message for i in low.issues)


def test_output_byte_cap_truncates_like_op_cap():
    space = PythonActions(max_output_mb=0.01)       # ~10 KB
    low = space.lower('for i in range(100000):\n    place("stone", 0, 0, 0)\n')
    assert 0 < len(_ops(low)) < 1000
    assert any("truncated" in i.message for i in low.issues)


# -- 4. one bad call in a loop is one issue ---------------------------------------
def test_loop_errors_are_deduplicated_with_count():
    code = ("for x in range(32):\n    for y in range(4):\n"
            "        place('oak_plank', x, y, 0)\nplace('unobtainium', 0, 0, 0)\n")
    low = PythonActions(size=SIZE).lower(code)
    assert [i.line_no for i in low.issues] == [3, 4]
    assert low.issues[0].message.endswith("(x128 calls)")


def test_remapped_execution_issues_are_deduplicated():
    low = Lowered(source="x", program_text="set stone 90 0 0  # L3\nset stone 91 0 0  # L3\n",
                  issues=[], n_source_lines=3)
    iss = [Issue(1, "a", "out of bounds", "warning"), Issue(2, "b", "out of bounds", "warning")]
    out = remap_issues(low, iss)
    assert len(out) == 1 and out[0].line_no == 3 and out[0].message.endswith("(x2 calls)")


def test_dedupe_keeps_distinct_and_singletons_unchanged():
    a, b = Issue(1, "", "m"), Issue(2, "", "m")
    assert dedupe_issues([a, b]) == [a, b]


# -- 5. ignored block states warn ---------------------------------------------------
def test_blocks_warns_on_ignored_states():
    low = BlockActions().lower("oak_stairs[facing=up] 1 0 0\noak_stairs[facng=north] 2 0 0\n"
                               "oak_log[axis=q] 3 0 0\nstone[waterlogged=true] 4 0 0\n"
                               "oak_stairs[facing=north,half=top] 5 0 0\noak_slab[type=top] 6 0 0\n")
    warn = [i for i in low.issues if i.severity == "warning"]
    assert [i.line_no for i in warn] == [1, 2, 3, 4]
    assert "'facing=up'" in warn[0].message and "'facng'" in warn[1].message
    assert len(_ops(low)) == 6        # still placed (default state), only warned


def test_python_warns_on_ignored_states_once_per_line():
    low = PythonActions().lower("for x in range(3):\n    place('oak_log[axis=q]', x, 0, 0)\n")
    assert len(_ops(low)) == 3
    assert [(i.line_no, i.severity) for i in low.issues] == [(2, "warning")]
    assert "axis=q" in low.issues[0].message


def test_state_warning_is_quiet_for_understood_states():
    for spec in ("oak_stairs[facing=east,half=top]", "oak_log[axis=x]", "oak_slab[type=double]",
                 "stone", "unobtainium[foo=bar]"):
        assert state_warning(spec) is None, spec


def test_mock_and_examples_raise_no_state_warnings():
    for space in (BlockActions(), PythonActions()):
        low = space.lower(space.mock_reply())
        assert low.issues == [], type(space).__name__
        assert MinecraftWorld.empty((16, 16, 16)).run(low) is not None
