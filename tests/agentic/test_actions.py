"""WP3 action spaces: dsl / blocks / mixed (python's sandbox has test_pyapi.py).

The load-bearing property is the single IR: whatever the model writes in, the
lowered DSL must build the same voxels, so every space is checked against the
plain DSL run of the same house.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from blockgen.agentic import core
from blockgen.agentic.actions import (MAX_BLOCK_EXAMPLE, MOCK_DETAIL, MOCK_DSL, MOCK_MASSING,
                                      BlockActions, DslActions, MixedActions, dsl_to_blocks,
                                      dsl_to_python, remap_issues)
from blockgen.agentic.blockstate import resolve_block
from blockgen.agentic.core import ActionSpace, Lowered, MinecraftWorld, PromptCtx
from blockgen.agentic.dsl import COMMANDS, command_reference, run_program
from blockgen.agentic.examples import EXAMPLES, select_examples
from blockgen.agentic.prompts import METHOD, RULES, scale_hint

SIZE = (32, 32, 32)
CTX = PromptCtx(size=SIZE, palette=["stone", "oak_planks"], prompt="a cottage")

ALL_SPECS = ["dsl", "blocks", "python", "mixed",
             {"mixed": {"parts": ["python", "blocks"]}},
             {"mixed": {"parts": ["blocks", "dsl"]}},
             {"mixed": {"parts": ["dsl"]}}]


def voxels(program: str, size=SIZE):
    canvas, report, _ = run_program(program, size=size)
    return canvas.block_ids.copy(), canvas.block_data.copy(), report


def assert_same_voxels(a: str, b: str, size=SIZE) -> None:
    ia, da, _ = voxels(a, size)
    ib, db, _ = voxels(b, size)
    assert ia.sum() > 0
    np.testing.assert_array_equal(ia, ib)
    np.testing.assert_array_equal(da, db)


def errors(low: Lowered):
    return [i for i in low.issues if i.severity == "error"]


# ---------------------------------------------------------------------------
# Registry + protocol
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spec", ALL_SPECS, ids=str)
def test_registered_and_satisfies_protocol(spec):
    space = core.make("actions", spec)
    assert isinstance(space, ActionSpace)
    assert space.patchable is True
    assert isinstance(space.name, str) and space.name


def test_aliases_and_param_validation():
    assert isinstance(core.make("actions", "block_lines"), BlockActions)
    assert isinstance(core.make("actions", "dsl?commands=fill,walls"), DslActions)
    with pytest.raises(TypeError, match="allow_states"):
        core.make("actions", {"blocks": {"allow_state": False}})


# ---------------------------------------------------------------------------
# The single-IR acceptance test: one house, four spaces, identical voxels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spec", ALL_SPECS, ids=str)
def test_mock_reply_lowers_cleanly_to_the_same_house(spec):
    space = core.make("actions", spec)
    low = space.lower(space.mock_reply())
    assert low.issues == [], [i.render() for i in low.issues]
    assert low.source == space.mock_reply()
    assert_same_voxels(MOCK_DSL, low.program_text)
    # ...and through the World contract the episode actually uses.
    world = MinecraftWorld.empty(SIZE)
    report = world.run(low)
    assert report.ok and world.block_count() == voxels(MOCK_DSL)[2].blocks


def test_house_expressed_by_hand_in_each_space_is_identical():
    """Independent of the mock replies: hand-written blocks and mixed text."""
    blocks_text = "Here is the build:\n" + dsl_to_blocks(MOCK_DSL) + "Done.\n"
    mixed_text = ("I will do the massing in code.\n```python\n"
                  + dsl_to_python(MOCK_MASSING) + "```\nthen details:\n```blocks\n"
                  + dsl_to_blocks(MOCK_DETAIL) + "```\n")
    for space, text in ((BlockActions(), blocks_text), (MixedActions(), mixed_text)):
        low = space.lower(text)
        assert errors(low) == []
        assert_same_voxels(MOCK_DSL, low.program_text)


# ---------------------------------------------------------------------------
# dsl
# ---------------------------------------------------------------------------
LEGACY_REPLY = """Sure! Here is the program:
```
fill stone 0 0 0 4 0 4
walls oak_planks 0 1 0 4 3 4   # shell
/set torch 2 2 2
bogus_command 1 2 3
fill stone 0 0
```
"""


def test_dsl_unrestricted_lowering_is_identity_so_reports_match_legacy():
    low = DslActions().lower(LEGACY_REPLY)
    assert low.program_text == LEGACY_REPLY and low.issues == []
    assert low.n_source_lines == len(LEGACY_REPLY.splitlines())
    _, legacy_report, _ = run_program(LEGACY_REPLY, size=SIZE)
    world = MinecraftWorld.empty(SIZE)
    report = world.run(low)
    assert report.to_dict() == legacy_report.to_dict()
    assert report.summary() == legacy_report.summary()


def test_dsl_instructions_reuse_legacy_pieces_and_end_with_rule():
    text = DslActions().instructions(CTX)
    assert command_reference() in text and RULES in text and METHOD in text
    assert "x in 0..31" in text
    assert text.rstrip().endswith(DslActions().output_rule())
    assert DslActions().output_rule() == "Output only command lines."
    hinted = DslActions().instructions(PromptCtx(size=SIZE, palette=[], target_blocks=900))
    assert scale_hint(900) in hinted and scale_hint(900) not in text


def test_dsl_examples_are_the_legacy_selection():
    got = DslActions().examples("a stone watchtower", 2)
    want = [(e.caption, e.program.strip()) for e in select_examples("a stone watchtower", 2)]
    assert got == want
    assert DslActions().examples("x", 0) == []


def test_dsl_commands_subset_is_per_instance_and_never_global():
    before = dict(COMMANDS)
    narrow = DslActions(commands=["fill", "walls", "place"])     # 'place' aliases set
    wide = DslActions()
    assert narrow.allowed == ("set", "fill", "walls")
    text = "fill stone 0 0 0 4 0 4\nsphere glass 2 5 2 2\nwalls oak_planks 0 1 0 4 3 4\n"
    low = narrow.lower(text)
    assert [(i.line_no, i.severity) for i in low.issues] == [(2, "error")]
    assert "sphere" in low.issues[0].message and "fill" in low.issues[0].message
    assert low.program_text.splitlines()[1].startswith("#")       # dropped, lines aligned
    assert len(low.program_text.splitlines()) == 3
    # Executing: the sphere never runs.
    _, report, prog = run_program(low.program_text, size=SIZE)
    assert [c.spec.name for c in prog.calls] == ["fill", "walls"]
    # The other instance and the global registry are untouched.
    assert wide.lower(text).issues == [] and wide.lower(text).program_text == text
    assert COMMANDS == before
    assert "sphere" in wide.instructions(CTX)
    ref = narrow.reference()
    assert "sphere" not in ref and "gable" not in ref and "walls" in ref
    assert "sphere" not in narrow.instructions(CTX).split("Rules:")[0]


def test_dsl_commands_subset_examples_and_mock_respect_subset():
    narrow = DslActions(commands="fill,walls,set,clear,line,gable")
    for _, prog in narrow.examples("a house", 4):
        _, _, p = run_program(prog)
        assert {c.spec.name for c in p.calls} <= set(narrow.allowed)
    assert narrow.lower(narrow.mock_reply()).issues == []
    only_set = DslActions(commands=["set"])
    low = only_set.lower(only_set.mock_reply())
    assert low.issues == [] and voxels(low.program_text)[2].blocks > 0
    only_fill = DslActions(commands=["fill"])
    assert voxels(only_fill.lower(only_fill.mock_reply()).program_text)[2].blocks > 0


def test_dsl_commands_unknown_name_raises_with_suggestion():
    with pytest.raises(ValueError, match="did you mean 'walls'"):
        DslActions(commands=["wals"])
    with pytest.raises(ValueError):
        DslActions(commands=[])


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spec", [
    "oak_stairs[facing=north,half=top]", "oak_stairs[facing=east]", "spruce_stairs[half=top]",
    "oak_log[axis=x]", "oak_log[axis=z]", "oak_slab[type=top]", "stone_slab[type=double]",
    "minecraft:stone_bricks", "glass_pane", "oak_door",
])
def test_block_lines_with_states_resolve_like_dsl_set(spec):
    low = BlockActions().lower(f"{spec} 3 4 5")
    assert low.issues == []
    ids, data, _ = voxels(low.program_text)
    dsl_ids, dsl_data, _ = voxels(f"set {spec} 3 4 5")
    assert (int(ids[3, 4, 5]), int(data[3, 4, 5])) == resolve_block(spec)
    np.testing.assert_array_equal(ids, dsl_ids)
    np.testing.assert_array_equal(data, dsl_data)


def test_blocks_tolerated_decoration():
    text = ("- stone 1 2 3\n"
            "oak_stairs[facing=north, half=top] (4, 5, 6)\n"
            "glass, 7, 0, 7  # window\n"
            "dirt 2.0 0 2\n")
    low = BlockActions().lower(text)
    assert low.issues == []
    lines = low.program_text.splitlines()
    assert lines == ["set stone 1 2 3", "set oak_stairs[facing=north,half=top] 4 5 6",
                     "set glass 7 0 7", "set dirt 2 0 2"]


def test_blocks_bad_lines_are_issues_with_source_line_numbers():
    text = ("Here is the build:\n"            # 1 prose: ignored
            "stone 0 0 0\n"                   # 2 ok
            "unobtainium 1 0 0\n"             # 3 unknown block
            "oak_planks 1 2\n"                # 4 missing coordinate
            "fill stone 0 0 0 4 0 4\n"        # 5 a DSL line: malformed here
            "\n"                              # 6
            "# a comment\n"                   # 7
            "```\n"                           # 8 fence
            "air 0 0 0\n"                     # 9 ok: removal
            "glass 1 x 0\n")                  # 10 non-integer
    low = BlockActions().lower(text)
    assert [(i.line_no, i.severity) for i in low.issues] == [
        (3, "error"), (4, "error"), (5, "error"), (10, "error")]
    assert "unknown block" in low.issues[0].message
    assert all(i.text == text.splitlines()[i.line_no - 1] for i in low.issues)
    out = low.program_text.split("\n")
    assert len(out) == len(text.splitlines())                # line-aligned
    assert out[1] == "set stone 0 0 0" and out[8] == "set air 0 0 0"
    assert out[6] == "# a comment" and out[0] == "" and out[2] == ""
    assert voxels(low.program_text)[2].blocks == 0           # air removed the stone


def test_blocks_execution_warnings_point_at_source_lines():
    # line-aligned lowering => an out-of-canvas voxel's no-op warning names line 2.
    world = MinecraftWorld.empty((8, 8, 8))
    report = world.run(BlockActions().lower("stone 0 0 0\nstone 50 0 0\n"))
    warn = [i for i in report.issues if i.severity == "warning"]
    assert [i.line_no for i in warn] == [2]


def test_remap_issues_is_identity_for_aligned_spaces():
    low = BlockActions().lower("stone 0 0 0\nstone 50 0 0\n")
    report = MinecraftWorld.empty((8, 8, 8)).run(low)
    assert remap_issues(low, report.issues) == report.issues


def test_blocks_allow_states_false_strips_with_warning():
    low = BlockActions(allow_states=False).lower("oak_stairs[facing=north] 1 1 1")
    assert [i.severity for i in low.issues] == ["warning"]
    assert low.program_text == "set oak_stairs 1 1 1"
    assert "[facing" not in BlockActions(allow_states=False).grammar(CTX).split("not supported")[0]
    assert "facing=north" in BlockActions().grammar(CTX)


def test_blocks_examples_respect_the_size_cap_and_round_trip():
    space = BlockActions()
    exs = space.examples("a small oak hut", 3)
    assert exs, "the mock house keeps the per-block space from having no example"
    for caption, body in exs:
        assert len(body.splitlines()) <= MAX_BLOCK_EXAMPLE
        assert space.lower(body).issues == []
    assert space.examples("anything", 0) == []


def test_blocks_instructions():
    text = BlockActions().instructions(CTX)
    assert "x in 0..31" in text and "<block> x y z" in text
    assert text.rstrip().endswith(BlockActions().output_rule())


# ---------------------------------------------------------------------------
# converters
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ex", EXAMPLES, ids=lambda e: e.caption[:24])
def test_dsl_to_blocks_round_trips_every_stock_example(ex):
    text = dsl_to_blocks(ex.program, (64, 64, 64))
    low = BlockActions().lower(text)
    assert low.issues == []
    assert_same_voxels(ex.program, low.program_text, (64, 64, 64))
    ys = [int(l.split()[2]) for l in text.splitlines()]
    assert ys == sorted(ys)                                     # bottom-up


def test_dsl_to_blocks_cap_and_empty():
    with pytest.raises(ValueError, match="max_blocks"):
        dsl_to_blocks(EXAMPLES[0].program, max_blocks=10)
    assert dsl_to_blocks("# nothing\n") == ""


def test_dsl_to_python_is_line_aligned():
    src = "# roof\nfill stone 0 0 0 1 1 1  # base\nSure, here you go\n```\ngable oak_stairs 0 2 0 4 2 4 axis=z overhang=1\n"
    out = dsl_to_python(src).splitlines()
    assert out[0] == "# roof"
    assert out[1] == "fill('stone', 0, 0, 0, 1, 1, 1)  # base"
    assert out[2] == "# Sure, here you go"
    assert out[3] == ""
    assert out[4] == "gable('oak_stairs', 0, 2, 0, 4, 2, 4, axis='z', overhang=1)"


# ---------------------------------------------------------------------------
# mixed
# ---------------------------------------------------------------------------
def test_mixed_sections_and_issue_line_mapping():
    text = ("Plan: massing in python, trim in blocks.\n"   # 1 prose
            "```python\n"                                   # 2
            "fill('stone', 0, 0, 0, 4, 0, 4)\n"             # 3
            "place('unobtainium', 1, 1, 1)\n"               # 4
            "```\n"                                         # 5
            "```blocks\n"                                   # 6
            "glass 2 1 2\n"                                 # 7
            "dirt 9\n"                                      # 8
            "```\n"                                         # 9
            "```lua\n"                                      # 10
            "set torch 0 1 0\n"                             # 11
            "```\n")                                        # 12
    low = MixedActions().lower(text)
    got = [(i.line_no, i.severity) for i in low.issues]
    assert (4, "error") in got and (8, "error") in got and (10, "warning") in got
    assert all("[" in i.message for i in low.issues if i.line_no in (4, 8))
    ids, _, _ = voxels(low.program_text)
    assert ids[2, 1, 2] != 0 and ids[0, 1, 0] != 0 and ids[0, 0, 0] != 0
    # python op annotations are whole-reply line numbers
    assert re.search(r"fill stone 0 0 0 4 0 4  # L3$", low.program_text, re.M)


def test_mixed_disabled_parts_and_unfenced_commands():
    m = MixedActions(parts=["python", "blocks"])
    text = "intro prose\nfill stone 0 0 0 1 0 1\n```dsl\nset stone 0 0 0\n```\n```blocks\nstone 3 0 3\n```\n"
    low = m.lower(text)
    msgs = {(i.line_no, i.message.split(";")[0]) for i in low.issues}
    assert (2, "commands outside a fenced section are ignored") in msgs
    assert (3, "```dsl sections are not enabled here") in msgs
    ids, _, _ = voxels(low.program_text)
    assert int((ids != 0).sum()) == 1 and ids[3, 0, 3] != 0


def test_mixed_unfenced_is_dsl_and_unclosed_fence_still_lowers():
    low = MixedActions().lower("fill stone 0 0 0 2 0 2\n```blocks\nglass 1 1 1\n")
    assert low.issues == []
    ids, _, _ = voxels(low.program_text)
    assert ids[1, 1, 1] != 0 and int((ids != 0).sum()) == 10


def test_mixed_params_validation_and_sub_params():
    with pytest.raises(ValueError, match="unknown part"):
        MixedActions(parts=["pyhton"])
    with pytest.raises(ValueError, match="not in parts"):
        MixedActions(parts=["dsl"], sub={"blocks": {}})
    m = MixedActions(parts="python+blocks", sub={"python": {"timeout_s": 3}})
    assert m.parts == ("python", "blocks") and m.spaces["python"].timeout_s == 3


def test_mixed_examples_are_fenced_in_an_enabled_part():
    for parts in (("python", "dsl"), ("blocks",), ("dsl", "blocks")):
        m = MixedActions(parts=parts)
        exs = m.examples("a cottage", 1)
        assert len(exs) == 1
        assert exs[0][1].startswith(f"```{parts[0]}") and exs[0][1].endswith("```")
        assert errors(m.lower(exs[0][1])) == []


# ---------------------------------------------------------------------------
# output rules (mock:auto depends on these)
# ---------------------------------------------------------------------------
def test_output_rules_distinct_and_unambiguous():
    spaces = {str(s): core.make("actions", s) for s in ALL_SPECS}
    rules = {k: s.output_rule() for k, s in spaces.items()}
    by_name = {s.name: s.output_rule() for s in spaces.values()}
    assert len(set(by_name.values())) == len(by_name) == 4
    for k, space in spaces.items():
        text = space.instructions(CTX).lower()
        assert rules[k].lower() in text
        for other_name, other_rule in by_name.items():
            if other_name != space.name:
                assert other_rule.lower() not in text, (k, other_name)


def test_instructions_do_not_repeat_the_palette():
    # The palette comes from the `palette` context module, not the action space.
    ctx = PromptCtx(size=SIZE, palette=["zz_unique_block_name"])
    for s in ALL_SPECS:
        assert "zz_unique_block_name" not in core.make("actions", s).instructions(ctx)


# ---------------------------------------------------------------------------
# Verifier fixes (WP3 review)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("line", ["1. oak_planks 1 2 3", "2) oak_planks 1 2 3",
                                  "- oak_planks: 1, 2, 3", "• oak_planks 1 2 3"])
def test_blocks_tolerates_numbering_and_name_colon(line):
    """Numbered lists used to be dropped SILENTLY (no voxel, no issue)."""
    low = BlockActions().lower(line)
    assert low.program_text == "set oak_planks 1 2 3" and not low.issues


@pytest.mark.parametrize("line", ["Here are 3 walls:", "Layer 1 (y=0):", "Step 2 - the roof"])
def test_blocks_prose_with_digits_is_not_an_error(line):
    low = BlockActions().lower(line)
    assert low.program_text == "" and low.issues == []


def test_blocks_reports_dsl_commands_and_bad_known_blocks():
    low = BlockActions().lower("fill stone 0 0 0 3 3 3\nstone 1 2\nzzz 1 2 3")
    assert [i.line_no for i in low.issues] == [1, 2, 3]
    assert "region command" in low.issues[0].message
    assert all(i.severity == "error" for i in low.issues)


def test_mixed_tagged_fence_while_open_starts_a_new_section():
    """A model that forgets the closing fence before the next ```blocks must not
    lose the blocks section (it used to be read as unfenced text and dropped)."""
    m = MixedActions(parts=["python", "blocks"])
    reply = "```python\nfill('stone', 0, 0, 0, 2, 0, 2)\n```blocks\noak_planks 1 1 1\n```\n"
    low = m.lower(reply)
    assert not low.issues
    assert [s.part for s in m.sections(reply)] == ["python", "blocks"]
    assert_same_voxels(low.program_text, "fill stone 0 0 0 2 0 2\nset oak_planks 1 1 1\n")


def test_mixed_execution_issues_remap_to_reply_lines():
    """dsl/blocks sections are not aligned with the reply once concatenated; their
    commands carry '# L<n>' tags so remap_issues finds the source line."""
    m = MixedActions(parts=["python", "blocks", "dsl"])
    reply = ("Massing:\n```python\nfill('stone', 0, 0, 0, 2, 0, 2)\n```\nprose\n"
             "```blocks\noak_planks 1 1 1\noak_planks 99 1 1\n```\n"
             "```dsl\nset stone 0 5 0\nset stone 0 99 0\n```\n")
    src = reply.splitlines()
    low = m.lower(reply)
    assert not low.issues
    report = MinecraftWorld.empty(SIZE).run(low)
    warn = remap_issues(low, [i for i in report.issues if i.severity == "warning"])
    assert sorted(i.line_no for i in warn) == [8, 12]
    assert src[7] == "oak_planks 99 1 1" and src[11] == "set stone 0 99 0"


def test_mixed_forwards_size_to_python_and_mentions_section_scope():
    m = MixedActions(parts=["python", "blocks"], size=(16, 16, 16))
    assert m.spaces["python"].canvas_size() == (16, 16, 16)
    assert "variables and functions do not carry" in m.instructions(CTX)
    assert "variables and functions do not carry" not in MixedActions(
        parts=["dsl", "blocks"]).instructions(CTX)


def test_blocks_without_states_shows_no_states_and_mock_is_clean():
    """allow_states=False used to show stateful examples and a mock reply that
    lowered with ~80 warnings, contradicting its own instructions."""
    b = BlockActions(allow_states=False)
    assert "[" not in b.mock_reply() and not b.lower(b.mock_reply()).issues
    assert all("[" not in ans for _q, ans in b.examples("a small hut", 2))
    m = MixedActions(parts=["blocks"], sub={"blocks": {"allow_states": False}})
    assert not m.lower(m.mock_reply()).issues
    assert all("[" not in ans for _q, ans in m.examples("a small hut", 1))
