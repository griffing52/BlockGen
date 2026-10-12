"""Episode plumbing for the staged methods: region-scoped try_, room-local offsets,
deterministic (ir) steps, per-call accept, runtime dreams, select_from, salt, and the
render_match dream style."""

from __future__ import annotations

import pytest

from blockgen.agentic import core
from blockgen.agentic.core import VIEWS, DreamSpec, MethodSpec, ScoreTerm
from blockgen.agentic.dreams import (DEFAULT_STYLE, DREAMS, RENDER_MATCH_STYLE, STYLES,
                                     render_prompt)
from blockgen.agentic.episode import Episode, run_episode
from blockgen.agentic.recipe import load_recipe
from blockgen.agentic.region import Box, Region

from .fakes import (FAKE_DREAMS, HUT_PROGRAM, RecordingTrace, ScriptedLLM, make_fake_kit)

SHELL = "fill cobblestone 0 0 0 15 0 15\nwalls stone_bricks 0 1 0 15 3 15\n"
ROOM = Region("u_room", (Box(1, 1, 1, 6, 3, 6),))


def episode(**kw) -> Episode:
    trace = kw.pop("trace", None)
    return Episode("a small house", make_fake_kit(**kw), trace=trace)


def ids(ep: Episode):
    return ep.world.canvas.block_ids


def python_kit(**kw):
    return make_fake_kit(actions=core.make("actions", "python"), **kw)


# ---------------------------------------------------------------- views / styles
def test_unit_views_exist():
    assert VIEWS["room"].elev == 55 and VIEWS["room"].keep_below is None
    assert VIEWS["room_top"].elev == 89


def test_render_match_style_is_named_and_default_unchanged():
    ep = episode()
    assert STYLES["render_match"] == RENDER_MATCH_STYLE and STYLES["default"] == DEFAULT_STYLE
    assert render_prompt(DREAMS["iso"], ep).startswith(DEFAULT_STYLE[:40])
    ep.spec.dream_style = "render_match"
    assert render_prompt(DREAMS["iso"], ep).startswith(RENDER_MATCH_STYLE[:40])
    ep.spec.dream_style = "Watercolour, please."
    assert render_prompt(DREAMS["iso"], ep).startswith("Watercolour, please.")


# ---------------------------------------------------------------- ir / region / offset
def test_ir_step_runs_canonical_dsl_and_never_commits_a_pair():
    ep = Episode("a house", python_kit())
    ep.ask("Build it", stage="generate")             # a pending builder pair
    att = ep.try_("shell", SHELL, ir=True)
    assert att.accepted and att.blocks > 0 and att.report.n_failed == 0
    assert ep.history == []                           # the LLM pair was not committed
    assert att.response.text == SHELL                 # not attributed to the model reply
    # the same text through the python action space does not even parse
    bad = ep.try_("generate", SHELL)
    assert bad.report.n_failed + len(bad.report.issues) > 0


def test_region_patch_adds_inside_and_reverts_outside_dsl():
    ep = episode()
    ep.try_("shell", SHELL, ir=True)
    before = ids(ep).copy()
    prog = ("fill red_wool 1 1 1 6 1 6\n"            # inside
            "fill glass 0 1 0 15 3 0\n"              # outside: the front wall
            "set lantern 20 1 20\n")                 # outside the house
    att = ep.try_("u_room", prog, patch=True, region=ROOM)
    assert att.extra["region"] == "u_room" and att.extra["reverted"] == 16 * 3 + 1
    assert att.extra["replay_exact"] is False
    after = ids(ep)
    assert (after[1:7, 1, 1:7] != 0).all()                      # furnished
    mask = ROOM.mask(ep.size)
    assert (after[~mask] == before[~mask]).all()                # nothing else changed
    warn = [i for i in att.report.issues if "outside" in i.message]
    assert [(i.line_no, i.severity) for i in warn] == [(2, "warning"), (3, "warning")]
    assert "48 block(s) outside x 1..6, y 1..3, z 1..6" in warn[0].message
    assert "# --- region u_room ---" in att.program_text


def test_region_rewrite_replaces_the_room_and_keeps_the_rest():
    ep = episode()
    ep.try_("shell", SHELL, ir=True)
    ep.try_("u_room", "fill red_wool 1 1 1 6 1 6\n", patch=True, region=ROOM)
    outside = ~ROOM.mask(ep.size)
    before = ids(ep).copy()
    att = ep.try_("u_room_m0", "set lantern 3 1 3\n", region=ROOM)     # rewrite
    after = ids(ep)
    assert att.accepted and (after[outside] == before[outside]).all()
    assert after[1, 1, 1] == 0 and after[3, 1, 3] != 0           # the old carpet is gone
    assert "reverted" not in att.extra                           # nothing it wrote was outside
    assert att.extra["replay_exact"] is False                    # old room content dropped


def test_region_deny_keeps_door_fronts_clear():
    ep = episode()
    ep.try_("shell", SHELL, ir=True)
    r = Region("u_room", ROOM.allow, (Box(3, 1, 1, 3, 2, 1),))
    att = ep.try_("u_room", "fill oak_planks 1 1 1 6 2 1\n", patch=True, region=r)
    assert ids(ep)[3, 1, 1] == 0 and ids(ep)[2, 1, 1] != 0 and att.extra["reverted"] == 2


def test_region_with_python_actions_attributes_source_lines():
    ep = Episode("a house", python_kit())
    ep.try_("shell", SHELL, ir=True)
    reply = ('fill("red_wool", 1, 1, 1, 3, 1, 3)\n'
             "for x in range(0, 9):\n"
             '    place("stone", x, 2, 1)\n')
    att = ep.try_("u_room", reply, patch=True, region=ROOM)
    warn = [i for i in att.report.issues if "outside" in i.message]
    assert len(warn) == 1 and warn[0].line_no == 3 and 'place("stone"' in warn[0].text
    assert warn[0].message.startswith("3 block(s)") and att.extra["reverted"] == 3   # x 0, 7, 8
    assert ids(ep)[2, 2, 1] != 0 and ids(ep)[0, 2, 1] != 0       # (0,2,1) is the shell wall


def test_offset_moves_room_local_coordinates():
    ep = episode()
    ep.try_("shell", SHELL, ir=True)
    att = ep.try_("u_room", "set lantern 0 0 0\nset lantern 5 2 5\nset lantern 6 0 0\n",
                  patch=True, region=ROOM, offset=ROOM.origin)
    assert ids(ep)[1, 1, 1] != 0 and ids(ep)[6, 3, 6] != 0
    assert att.extra["reverted"] == 1                             # local (6,0,0) -> (7,1,1)
    assert "set lantern 1 1 1" in att.delta_text


def test_offset_with_python_actions():
    ep = Episode("a house", python_kit())
    att = ep.try_("u_room", 'place("lantern", 0, 0, 0)\n', region=ROOM, offset=(1, 1, 1))
    assert ids(ep)[1, 1, 1] != 0 and att.blocks == 1


def test_accept_override_replaces_the_recipe_policy():
    seen = []

    def never(prev, cand):
        seen.append((prev, cand))
        return False, "unit not better"

    ep = episode()
    first = ep.try_("shell", SHELL, ir=True)
    att = ep.try_("u_room", "fill red_wool 1 1 1 6 1 6\n", patch=True, region=ROOM,
                  accept=never)
    assert not att.accepted and att.reason == "unit not better"
    assert seen[0][0] is first and ep.current() is first
    assert ids(ep)[2, 1, 2] == 0                                  # rolled back
    assert ep.try_("next", HUT_PROGRAM).accepted                   # recipe policy again


def test_try_defaults_are_unchanged():
    ep = episode()
    att = ep.try_("generate", HUT_PROGRAM)
    assert att.extra == {} and att.selection is None and "region" not in att.program_text


# ---------------------------------------------------------------- runtime dreams
def test_register_and_put_dream_do_not_touch_dream_id():
    trace = RecordingTrace()
    trace.dream_views = {}
    ep = episode(dreams=["iso"], trace=trace)
    ep.dream("iso")
    did = ep.dream_id()
    ds = DreamSpec("u_kitchen", "{style} a kitchen. {desc}", view="room", weight=0,
                   show_builder=False)
    assert ep.register_dream(ds, trace_view="u_kitchen") is ds
    assert ep.register_dream(ds) is ds                                  # equal: no-op
    assert trace.dream_views["u_kitchen"] == "u_kitchen"
    n_img = ep.ledger.n_images
    dv = ep.dream("u_kitchen", view_hint="room", refs=[b"fake-ref"])
    assert ep.kit.image.requests[-1].view == "room" and ep.ledger.n_images == n_img + 1
    assert ep.latest_dream("u_kitchen") is dv and ep.dream_id() == did
    put = ep.put_dream("u_kitchen", b"PNG-tile", generator="split:sheet_v0")
    assert put.version == 1 and put.generator == "split:sheet_v0"
    assert ep.ledger.n_images == n_img + 1 and trace.dreams[-1] is put
    assert ep.dream_id() == did
    # never in the builder's TARGET prefix, never a score view
    from blockgen.agentic.turns import builder_dream_parts
    assert all(dv.png != p.image_png for p in builder_dream_parts(ep))
    assert "room" not in ep._view_names("score")


def test_register_dream_rejects_a_changed_recipe_spec_and_put_needs_a_key():
    ep = episode(dreams=["iso"])
    assert ep.register_dream(FAKE_DREAMS["iso"]) == FAKE_DREAMS["iso"]
    with pytest.raises(ValueError):
        ep.register_dream(DreamSpec("iso", "other {desc}", view="iso"))
    with pytest.raises(KeyError):
        ep.put_dream("never_registered", b"x")
    # a runtime spec may be replaced
    ep.register_dream(DreamSpec("u_a", "a {desc}", view=None))
    ep.register_dream(DreamSpec("u_a", "b {desc}", view=None))
    assert ep.dream_spec("u_a").template == "b {desc}"


def test_put_dream_on_a_scored_dream_changes_dream_id():
    ep = episode(dreams=["iso"])
    ep.dream("iso")
    did = ep.dream_id()
    ep.put_dream("iso", b"another image")
    assert ep.dream_id() != did


# ---------------------------------------------------------------- selection
def test_select_from_with_fallback():
    ep = episode(spec_kw={"select": "best"})
    big = ep.try_("generate", "fill stone 0 0 0 9 0 9\n")
    small = ep.try_("roof", "fill stone 0 0 0 1 0 1\n")
    assert ep.best() is big
    ep.select_from = 1
    assert ep.best() is small
    ep.select_from = 2                                   # nothing there yet: fall back
    assert ep.best() is big
    ep.select_from = 1
    ep.try_("final0", "clear 0 0 0 1 1 1\n")             # empty: never selected
    assert ep.best() is small


def test_result_records_selection_values():
    small_wins = lambda ep, a: ScoreTerm(1.0 / max(1, a.blocks))  # noqa: E731
    ep = episode(spec_kw={"select": "best"}, final_scorers={"inv": (small_wins, 1.0)})
    a = ep.try_("generate", "fill stone 0 0 0 1 0 1\n")
    b = ep.try_("match0", "fill stone 0 0 0 3 0 3\n")
    res = ep.result()
    assert res.best_idx == a.idx
    assert a.selection == pytest.approx(0.25) and b.selection == pytest.approx(1 / 16)
    assert res.to_dict()["attempts"][0]["selection"] == pytest.approx(0.25)
    plain = episode(spec_kw={"select": "best"})
    plain.try_("generate", HUT_PROGRAM)
    plain.result()
    assert plain.attempts[0].selection is None


# ---------------------------------------------------------------- salt
# 2026-10-03: dream, dream-multiview and classic-matched gained the `scale` context
# (deliberate hash changes). The pre-scale hashes live on in the *-noscale controls,
# which are those exact recipes (name and description are cosmetic).
# 2026-10-04: blueprint/floors switched to select: best_accepted after the staged
# pilot picked a rejected, damaged final pass (deliberate change for blueprint-noscale).
PINNED = {"dream": "6974c014229e", "dream-multiview": "69253ce81bcf",
          "classic-matched": "5557761f11dd", "dsl-classic": "6f9ecfefc43a",
          "dream-noscale": "fc8963b57243", "blueprint-noscale": "38ab4e580719"}


@pytest.mark.parametrize("name", sorted(PINNED))
def test_salt_field_leaves_shipped_recipe_hashes_unchanged(name):
    spec = load_recipe(name)
    assert spec.salt == "" and "salt" not in spec.resolved()
    assert spec.hash() == PINNED[name]
    assert "salt" not in spec.provenance()


def test_salt_changes_hash_seed_and_system_text():
    plain = MethodSpec(name="a")
    salted = MethodSpec(name="a", salt="s1")
    assert plain.hash() != salted.hash() and salted.provenance()["salt"] == "s1"
    llm = ScriptedLLM(["ok"])
    ep0 = episode(llm=llm)
    ep1 = episode(llm=llm, spec_kw={"salt": "s1"})
    assert ep0.salt == "" and ep1.salt == "s1" and ep1.seed != ep0.seed
    ep0.ask("hi")
    ep1.ask("hi")
    ep1.ask("judge this", role="judge")
    sys0, sys1, sysj = llm.calls[0][0], llm.calls[1][0], llm.calls[2][0]
    assert "[sample" not in sys0.text
    assert sys1.parts[-1].text == "[sample s1]" and sys1.text.startswith(sys0.text)
    assert sysj.parts[-1].text == "[sample s1]"
    # dream seeds follow the salted episode seed
    ep2 = episode(dreams=["iso"], spec_kw={"salt": "s1"})
    ep2.dream("iso")
    assert ep2.kit.image.requests[-1].seed == ep2.seed != 0


def test_run_episode_with_region_method_end_to_end():
    def method(ep):
        ep.try_("shell", SHELL, ir=True)
        ep.select_from = len(ep.attempts)
        ep.try_("u_room", ep.ask("furnish"), patch=True, region=ROOM)

    kit = make_fake_kit(replies=["fill red_wool 1 1 1 6 1 6\nset lantern 30 1 30\n"],
                        method=method, spec_kw={"select": "best"})
    res = run_episode(kit, "a house")
    assert not res.truncated, res.error
    assert res.best_idx == 1 and res.attempts[1].extra["reverted"] == 1


# ---------------------------------------------------------------- verifier additions
def test_offset_issues_quote_the_models_own_coordinates():
    ep = episode()
    ep.try_("shell", SHELL, ir=True)
    att = ep.try_("u_room", "set lantern 0 0 0\nset no_such_block 1 0 1\n",
                  patch=True, region=ROOM, offset=ROOM.origin)
    bad = [i for i in att.report.issues if i.severity == "error"]
    assert bad and bad[0].line_no == 2 and bad[0].text == "set no_such_block 1 0 1"


def test_room_region_with_keep_clear_cannot_close_the_stairwell():
    from blockgen.agentic.blueprint import mock_blueprint

    bp = mock_blueprint((32, 32, 32), storeys=2)
    ep = episode(spec_kw={"canvas": (32, 32, 32)})
    ep.try_("shell", bp.shell_dsl(), ir=True)
    st = bp.stairs[0]
    top = st.cells(bp.storey(0).height)[-1]
    room = next(r for r in bp.rooms_on(1)
                if r.interior is not None and r.interior.contains(top[0], top[2]))
    region = Region(room.name, (bp.room_box(room.name),), tuple(bp.keep_clear(room.name)))
    before = ids(ep).copy()
    fy = bp.storey(1).floor_y
    it = room.interior
    ep.try_("u_" + room.name, f"fill red_carpet {it.x0} {fy} {it.z0} {it.x1} {fy} {it.z1}\n",
            patch=True, region=region)
    for (x, _dy, z) in st.cells(bp.storey(0).height)[-3:]:
        assert ids(ep)[x, fy, z] == before[x, fy, z]          # hole and top step intact
