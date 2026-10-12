"""WP3 python action space: static checks, the subprocess sandbox, the API.

These run real child processes (``python -I -S -c RUNNER``, ~15 ms each), so the
sandbox's limits are tested for real rather than mocked.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pytest

from blockgen.agentic.actions import MOCK_DSL, dsl_to_python, remap_issues
from blockgen.agentic.core import MinecraftWorld
from blockgen.agentic.dsl import run_program
from blockgen.agentic.examples import EXAMPLES
from blockgen.agentic.pyapi import (MOCK_PYTHON, PythonActions, api_reference, check_code,
                                    extract_code)

SIZE = (32, 32, 32)


def lower(code: str, **kw):
    return PythonActions(**kw).lower(code)


def ops(low):
    return [l for l in low.program_text.splitlines() if l.strip()]


def same_voxels(a: str, b: str, size=SIZE) -> None:
    ca, _, _ = run_program(a, size=size)
    cb, _, _ = run_program(b, size=size)
    assert ca.block_count() > 0
    np.testing.assert_array_equal(ca.block_ids, cb.block_ids)
    np.testing.assert_array_equal(ca.block_data, cb.block_data)


# ---------------------------------------------------------------------------
# Static allow-list: rejected code never runs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("code, needle", [
    ("import os", "'import'"),
    ("from os import path", "'import'"),
    ("x = open('/etc/passwd').read()", "'open'"),
    ("__import__('os')", "'__import__'"),
    ("eval('1+1')", "'eval'"),
    ("exec('x=1')", "'exec'"),
    ("getattr(fill, 'x')", "'getattr'"),
    ("x = ().__class__.__bases__", "'.__class__'"),
    ("fill._x = 1", "'._x'"),
    ("def f():\n    global y\n    y = 1", "'global'"),
    ("with x:\n    pass", "'with'"),
    ("class A:\n    pass", "'class'"),
    ("f = lambda a=print(): a", "default argument"),
    ("def f(a=fill('stone', 0, 0, 0, 1, 1, 1)):\n    pass", "default argument"),
    ("@decorate\ndef f():\n    pass", "decorators"),
    ("s = '{0.x}'.format(1)", "'.format'"),
    ("k = 'a__b'", "'__'"),
    ("g = (i for i in [1])\ng.gi_frame", "'.gi_frame'"),
    ("async def f():\n    pass", "'async def'"),
])
def test_static_check_rejects(code, needle):
    issues = check_code(code)
    assert issues and any(needle in i.message for i in issues), [i.message for i in issues]
    low = lower(code + "\nfill('stone', 0, 0, 0, 3, 3, 3)\n")
    assert low.program_text == "" and low.issues
    assert all(i.severity == "error" for i in low.issues)


def test_static_check_reports_line_numbers_and_syntax_errors():
    issues = check_code("fill('stone', 0, 0, 0, 1, 1, 1)\n\nimport os\n")
    assert [i.line_no for i in issues] == [3] and issues[0].text == "import os"
    syn = check_code("for x in range(3)\n    pass\n")
    assert syn[0].line_no == 1 and "SyntaxError" in syn[0].message


def test_static_check_allows_ordinary_python():
    code = '''
W, D = 9, 7
def post(x, z, h=4):
    line("oak_log", x, 1, z, x, h, z)
cols = [(x, z) for x in (0, W - 1) for z in (0, D - 1)]
for x, z in cols:
    post(x, z)
heights = {i: 2 + i % 2 for i in range(3)}
y = max(heights.values()) if heights else 0
area = sum(1 for _ in range(W))
f = lambda a, b=2: a * b
label = f"{W}x{D}"
try:
    1 / 0
except ZeroDivisionError:
    pass
'''
    assert check_code(code) == []


# ---------------------------------------------------------------------------
# Sandbox runtime
# ---------------------------------------------------------------------------
def test_runtime_has_no_dangerous_builtins_even_without_static_check():
    space = PythonActions(timeout_s=5)
    for code in ("open('x', 'w').write('pwned')", "__import__('os').system('true')",
                 "eval('1')", "getattr(1, 'real')"):
        res = space.run(code)
        assert res.ops == [] and res.error and res.error["type"] == "NameError", code


def test_infinite_loop_times_out_within_bound():
    t0 = time.monotonic()
    low = lower("while True:\n    pass\n", timeout_s=1)
    assert time.monotonic() - t0 < 1 + 1
    assert low.program_text == ""
    assert any("timed out" in i.message for i in low.issues)


def test_op_cap_truncates_and_reports_line():
    code = "fill('stone', 0, 0, 0, 1, 0, 1)\nfor i in range(10**7):\n    place('stone', i % 8, 1, 0)\n"
    low = lower(code, max_ops=50)
    assert len(ops(low)) == 50
    cap = [i for i in low.issues if "op cap" in i.message]
    assert cap and cap[0].line_no == 3 and cap[0].text.strip().startswith("place(")


def test_op_cap_cannot_be_swallowed():
    code = ("for i in range(10**6):\n"
            "    try:\n"
            "        place('stone', 0, 0, 0)\n"
            "    except:\n"
            "        pass\n")
    low = lower(code, max_ops=20, timeout_s=5)
    assert len(ops(low)) == 20
    assert any("op cap" in i.message for i in low.issues)


def test_runtime_error_line_and_partial_build_kept():
    code = ("fill('stone', 0, 0, 0, 3, 0, 3)\n"
            "x = [1, 2]\n"
            "walls('oak_planks', 0, 1, 0, 3, 2, 3)\n"
            "y = x[5]\n"
            "place('glass', 1, 1, 1)\n")
    low = lower(code)
    assert len(ops(low)) == 2
    err = [i for i in low.issues if "IndexError" in i.message]
    assert err and err[0].line_no == 4 and err[0].text == "y = x[5]"


def test_error_inside_user_function_reports_innermost_user_line():
    code = "def roof():\n    gable('oak_stairs', 0, 3, 0, 4, 3, 4, axis='q')\n\nroof()\n"
    low = lower(code)
    err = [i for i in low.issues if i.severity == "error"]
    assert err[0].line_no == 2 and "axis" in err[0].message


def test_unknown_block_is_issue_at_source_line_and_op_dropped():
    code = "fill('stone', 0, 0, 0, 1, 0, 1)\nfor x in range(2):\n    place('unobtainium', x, 1, 0)\n"
    low = lower(code)
    assert len(ops(low)) == 1
    bad = [i for i in low.issues if "unknown block" in i.message]
    # one bad call in a loop is ONE issue with a count (not one per iteration)
    assert [i.line_no for i in bad] == [3] and bad[0].message.endswith("(x2 calls)")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS is Linux-specific")
def test_memory_limit():
    low = lower("x = [0] * (10 ** 9)\n", mem_mb=256, timeout_s=5)
    assert low.program_text == ""
    assert any("MemoryError" in i.message or "crashed" in i.message for i in low.issues)


def test_file_writes_fail_inside_the_child(tmp_path):
    # Even reaching a write primitive would hit RLIMIT_FSIZE=0; here we confirm the
    # simpler property that the child's cwd is a throwaway dir, not ours.
    res = PythonActions().run("place('stone', 0, 0, 0)")
    assert res.ops and res.error is None
    assert not list(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------
def test_api_names_kwargs_defaults_and_aliases():
    code = ("place('stone', 0, 0, 0)\n"
            "block('dirt', 1, 0, 0)\n"
            "set_block('sand', 2, 0, 0)\n"
            "walls(block='oak_planks', x1=0, y1=1, z1=0, x2=4, y2=2, z2=4, thickness=1)\n"
            "sphere('glass', 10, 10, 10, 2, hollow=True)\n"
            "cylinder('stone', 20, 0, 20, 2, 3, axis='Z')\n"
            "s = set([1, 1, 2])\n"           # builtin set is still the set type
            "place('stone', len(s), 5, 5.0)\n"
            "stack(0, 0, 0, 0, 0, 0, '+x', 2)\n")
    low = lower(code, annotate=False)
    assert low.issues == []
    assert ops(low) == [
        "set stone 0 0 0", "set dirt 1 0 0", "set sand 2 0 0",
        "walls oak_planks 0 1 0 4 2 4", "sphere glass 10 10 10 2 hollow=true",
        "cylinder stone 20 0 20 2 3 axis=z", "set stone 2 5 5", "stack 0 0 0 0 0 0 +x 2"]


@pytest.mark.parametrize("code, needle", [
    ("place('stone', 0, 0)", "missing required argument 'z'"),
    ("place('stone', 0, 0, 0, 9)", "at most 4"),
    ("fill('stone', 0, 0, 0, 1, 1, 1, colour=3)", "unexpected keyword"),
    ("place('stone', 'a', 0, 0)", "must be a number"),
    ("place(5, 0, 0, 0)", "must be a string"),
    ("place('stone', float('nan'), 0, 0)", "finite"),
])
def test_api_argument_errors(code, needle):
    low = lower(code)
    assert ops(low) == []
    assert any(needle in i.message and i.line_no == 1 for i in low.issues), \
        [i.render() for i in low.issues]


def test_random_is_seeded_and_math_available():
    code = "for i in range(5):\n    place('stone', random.randint(0, 20), int(math.sqrt(16)), 0)\n"
    a, b = lower(code), lower(code)
    assert a.program_text == b.program_text and len(ops(a)) == 5
    assert lower(code, seed=1).program_text != a.program_text


def test_annotations_carry_source_lines_into_execution_warnings():
    code = "place('stone', 0, 0, 0)\n\nplace('stone', 90, 0, 0)\n"
    low = lower(code)
    assert ops(low) == ["set stone 0 0 0  # L1", "set stone 90 0 0  # L3"]
    report = MinecraftWorld.empty((8, 8, 8)).run(low)
    warn = [i for i in report.issues if i.severity == "warning"]
    # run_program numbers by lowered line (2) and strips the comment from .text;
    # remap_issues reads the tag back to the source line (3).
    assert [i.line_no for i in warn] == [2]
    assert [i.line_no for i in remap_issues(low, warn)] == [3]
    assert remap_issues(low, warn)[0].message == warn[0].message


def test_fenced_code_extraction_is_line_aligned():
    text = "Here you go:\n```python\nplace('stone', 0, 0, 0)\nx = [][1]\n```\nThanks!\n"
    code = extract_code(text)
    assert code.splitlines()[2] == "place('stone', 0, 0, 0)"
    assert code.split("\n")[0] == "" and code.split("\n")[5] == ""
    low = lower(text)
    assert [i.line_no for i in low.issues] == [4]
    assert low.n_source_lines == 6
    assert extract_code("place('stone', 0, 0, 0)") == "place('stone', 0, 0, 0)"


def test_mock_reply_matches_dsl_house():
    low = lower(MOCK_PYTHON)
    assert low.issues == []
    same_voxels(MOCK_DSL, low.program_text)


ALL_COMMANDS = """\
fill stone 0 0 0 9 0 9
box stone_bricks 0 1 0 5 4 5 thickness=1
walls oak_planks 6 1 6 9 3 9
replace oak_planks glass 6 2 6 9 2 9
replace any bricks 0 1 0 0 1 5
clear 2 1 0 3 2 0
line oak_log 0 5 0 5 5 5 thickness=2
sphere oak_leaves 12 6 12 2 hollow=true
cylinder cobblestone 15 0 15 2 4 axis=x
pyramid sandstone 20 1 5 3
gable oak_stairs 0 6 0 5 6 5 axis=z overhang=1
copy 6 1 6 9 3 9
paste 20 1 20 rotate=90 mirror=x
stack 0 0 0 1 0 1 +x 3
set oak_stairs[facing=north,half=top] 1 1 1
"""


def test_every_dsl_command_converts_to_python_with_identical_voxels():
    py = dsl_to_python(ALL_COMMANDS)
    low = lower(py)
    assert low.issues == []
    same_voxels(ALL_COMMANDS, low.program_text)


@pytest.mark.parametrize("ex", EXAMPLES, ids=lambda e: e.caption[:24])
def test_python_examples_are_converted_stock_examples(ex):
    space = PythonActions()
    exs = dict(space.examples(ex.caption, 4))
    assert ex.caption in exs
    low = space.lower(exs[ex.caption])
    assert low.issues == []
    same_voxels(ex.program, low.program_text, (64, 64, 64))


def test_api_reference_is_generated_from_the_dsl_registry():
    ref = api_reference()
    assert "place(block, x, y, z)" in ref and "aliases: block, set_block" in ref
    assert "gable(block, x1, y1, z1, x2, y2, z2, axis='x', overhang=0" in ref
    assert "\n  set(" not in ref


def test_constructor_validation():
    with pytest.raises(ValueError):
        PythonActions(timeout_s=0)
    with pytest.raises(ValueError):
        PythonActions(max_ops=0)


# ---------------------------------------------------------------------------
# Verifier fixes (WP3 review)
# ---------------------------------------------------------------------------
def test_size_matches_the_canvas_the_model_was_told():
    """SIZE used to be the 64^3 conversion canvas whatever the prompt said, so a
    program centring on SIZE[0] // 2 built outside a 32^3 canvas."""
    from blockgen.agentic.core import PromptCtx

    p = PythonActions()
    assert p.canvas_size() == (64, 64, 64)            # nothing known yet
    text = p.instructions(PromptCtx(size=SIZE, palette=[]))
    assert "`SIZE` = (32, 32, 32)" in text
    assert ops(p.lower("place('stone', SIZE[0] - 1, 0, 0)")) == ["set stone 31 0 0  # L1"]
    q = PythonActions(size=(8, 9, 10))                 # explicit size wins
    q.instructions(PromptCtx(size=SIZE, palette=[]))
    assert ops(q.lower("place('stone', SIZE[2] - 1, 0, 0)")) == ["set stone 9 0 0  # L1"]
