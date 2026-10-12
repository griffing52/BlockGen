"""The describe stage (describe.py): a spec before the dream, a caption after it.

Covers: config validation, byte-identical requests when off, EXPAND (planner turn
first, spec in the image prompt and the build turn), CAPTION (judge turn on the
image, in the build and match turns, cached on the image digest, recaptioned after
a re-dream), BOTH (disagreements logged, the image wins), prose format, the
shuffled control (another prompt's caption, else the library), the mock replies,
trace files, recipes and batteries, and the staged methods' exterior hook.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from blockgen.agentic import describe as dm
from blockgen.agentic.core import ScoreTerm
from blockgen.agentic.episode import run_episode
from blockgen.agentic.kit import build_kit, validate_spec
from blockgen.agentic.providers_cli import AutoMockProvider
from blockgen.agentic.providers import text_message
from blockgen.agentic.recipe import load_battery, load_recipe
from blockgen.agentic.recipes.dream import dream
from blockgen.agentic.trace import Trace, _dream_versions
from blockgen.agentic.turns import JSON_RULE

from .fakes import RecordingTrace, ScriptedLLM, make_fake_kit

SPEC = {"summary": "A two-storey brick townhouse.", "footprint_blocks": [10, 8],
        "storeys": {"count": 2, "clear_heights": [3, 3]},
        "roof": {"shape": "gable", "material": "dark_oak_stairs"},
        "materials": {"walls": "bricks", "trim": "stone_bricks", "roof": "dark_oak_stairs",
                      "floor": "oak_planks", "windows": "glass_pane", "door": "oak_door",
                      "foundation": "stone"},
        "openings": ["front door centred"], "distinctive_features": ["bay window"],
        "surroundings_to_omit": ["street"]}
CAPTION = {**SPEC, "materials": {**SPEC["materials"], "walls": "terracotta"},
           "storeys": {"count": 3, "clear_heights": [3, 3, 3]},
           "dimensions_blocks": [10, 14, 8], "faces": {"front": "door, two windows"},
           "uncertain": ["back face"]}
PROG = "fill stone 0 0 0 9 0 9\n"


def match_scorer(ep, att):
    return ScoreTerm(min(1.0, att.blocks / 500.0))


match_scorer.needs_dream = True


def _llm(caption=CAPTION, spec=SPEC, extra=()):
    return ScriptedLLM([PROG], rules=[(dm.SPEC_TAG, json.dumps(spec)),
                                      (dm.CAPTION_TAG, json.dumps(caption)
                                       if isinstance(caption, dict) else caption),
                                      *extra])


def _kit(params, llm=None, **kw):
    return make_fake_kit(llm=llm or _llm(), method=dream, dreams=["iso"],
                         scorers={"match": (match_scorer, 1.0)},
                         spec_kw={"params": {"max_iters": 1, **params}, "select": "best"}, **kw)


def _run(params, prompt="a brick house", llm=None, kit=None, trace=None):
    kit = kit or _kit(params, llm=llm)
    trace = trace or RecordingTrace()
    res = run_episode(kit, prompt, trace=trace)
    assert not res.truncated, res.error
    return res, kit, trace


def _req(trace, stage):
    return [r for r in trace.requests if r["stage"] == stage]


def _turn_text(r):
    return r["messages"][-1].text


# --------------------------------------------------------------------------- config
def test_config_validation_and_none():
    assert dm.config() is None and dm.config_from_params({"max_iters": 3}) is None
    for bad in ({"describe": "later"}, {"describe": "after", "caption_format": "yaml"},
                {"describe": "after", "caption_keys": "some"},
                {"describe": "after", "caption_control": "random"},
                {"describe": "before", "caption_control": "shuffled"},
                {"caption_control": "shuffled"}):
        with pytest.raises(ValueError):
            dm.config(**bad)
    c = dm.config_from_params({"describe": "both", "caption_format": "prose", "other": 1})
    assert c.expands and c.captions and c.caption_format == "prose"
    assert set(dm.PARAMS) <= set(dream.__code__.co_varnames)


def test_bad_param_fails_before_any_call():
    llm = _llm()
    res = run_episode(_kit({"describe": "sideways"}, llm=llm), "a hut")
    assert res.truncated == "error" and "describe must be" in res.error and not llm.calls


# --------------------------------------------------------------------------- off
def test_describe_none_sends_byte_identical_requests():
    _, _, t0 = _run({})
    _, _, t1 = _run({"describe": "none", "caption_format": "prose"})
    assert [r["stage"] for r in t0.requests] == [r["stage"] for r in t1.requests]
    for a, b in zip(t0.requests, t1.requests):
        assert [m.text for m in a["messages"]] == [m.text for m in b["messages"]]
    assert not any("TARGET DESCRIPTION" in _turn_text(r) or "DESIGN SPEC" in _turn_text(r)
                   for r in t0.requests)


# --------------------------------------------------------------------------- before
def test_before_writes_spec_first_into_image_and_build_turn():
    res, kit, trace = _run({"describe": "before"})
    first = trace.requests[0]
    assert first["role"] == "planner" and first["stage"] == "describe_spec"
    assert not any(p.kind == "image" for m in first["messages"] for p in m.parts)
    assert "Canvas: 32 x 32 x 32 blocks" in _turn_text(first) or "Canvas:" in _turn_text(first)
    assert _turn_text(first).rstrip().endswith(JSON_RULE)
    prompt = kit.image.requests[0].prompt
    assert "a brick house. A two-storey brick townhouse" in prompt
    assert "walls of bricks" in prompt and "no street" in prompt
    build = _turn_text(_req(trace, "generate")[0])
    assert "DESIGN SPEC" in build and "materials: walls bricks" in build
    assert "The TARGET image was drawn from this spec" in build
    assert "TARGET DESCRIPTION" not in build
    assert not _req(trace, "describe_caption:iso")
    assert any(k == "describe_spec" for k, _ in trace.events)


def test_before_without_expand_into_image_keeps_the_dream_request():
    _, k0, _ = _run({})
    _, k1, t1 = _run({"describe": "before", "expand_into_image": False, "show_spec": False})
    assert k0.image.requests[0].prompt == k1.image.requests[0].prompt
    assert "DESIGN SPEC" not in _turn_text(_req(t1, "generate")[0])


def test_spec_turn_carries_scale_line_only_with_scale_context():
    ep_like = type("E", (), {"size": (16, 20, 16), "prompt": "a hut",
                             "kit": type("K", (), {"contexts": []})(),
                             "spec": type("S", (), {"contexts": []})()})()
    text = dm.spec_turn(ep_like, "json").text
    assert "Canvas: 16 x 20 x 16 blocks" in text and "Minecraft player" not in text


# --------------------------------------------------------------------------- after
def test_after_captions_the_dream_and_shows_it_to_builder_and_critic():
    res, kit, trace = _run({"describe": "after"})
    cap = _req(trace, "describe_caption:iso")
    assert len(cap) == 1 and cap[0]["role"] == "judge"
    imgs = [p.image_png for p in cap[0]["messages"][-1].parts if p.kind == "image"]
    assert imgs == [res.dreams["iso"][-1].png]
    assert "describe ONLY what the image shows" in _turn_text(cap[0])
    build = _turn_text(_req(trace, "generate")[0])
    assert "TARGET DESCRIPTION (what the TARGET image shows" in build
    assert "materials: walls terracotta" in build and "DESIGN SPEC" not in build
    match = _turn_text(_req(trace, "match0")[0])
    assert "TARGET DESCRIPTION" in match and "front face: door, two windows" in match
    assert not _req(trace, "describe_spec")


def test_caption_in_match_false_keeps_compare_turn_unchanged():
    _, _, t0 = _run({})
    _, _, t1 = _run({"describe": "after", "caption_in_match": False})
    assert _turn_text(_req(t0, "match0")[0]) == _turn_text(_req(t1, "match0")[0])


def test_caption_is_cached_on_digest_and_redone_after_a_redream():
    # The critic asks for a REDREAM once: the new image must get a new caption.
    replies = iter(["REDREAM: a taller brick house", PROG, PROG, PROG])
    llm = _llm(extra=[("Compare your build with the target", lambda _m: next(replies))])
    res, kit, trace = _run({"describe": "after", "redream": 1, "max_iters": 2}, llm=llm)
    caps = [r for r in trace.requests if r["stage"] == "describe_caption:iso"]
    assert len(res.dreams["iso"]) == 2 and len(caps) == 2
    digests = [d["digest"] for k, d in trace.events if k == "describe_caption"]
    assert digests == [v.digest for v in res.dreams["iso"]]
    n = len(llm.calls)
    from blockgen.agentic.episode import Episode
    ep = Episode("a brick house", kit)
    dm.setup(ep, dm.config(describe="after"))
    ep.dream("iso")
    dm.refresh(ep)
    dm.refresh(ep)
    assert len(llm.calls) == n + 1       # one caption, the second refresh is free


def test_caption_keys_all_captions_every_scored_dream():
    kit = make_fake_kit(llm=_llm(), method=dream, dreams=["iso", "back"],
                        scorers={"match": (match_scorer, 1.0)},
                        spec_kw={"params": {"max_iters": 0, "describe": "after",
                                            "caption_keys": "all"}})
    _, _, trace = _run({}, kit=kit)
    stages = [r["stage"] for r in trace.requests if r["stage"].startswith("describe_caption")]
    assert stages == ["describe_caption:iso", "describe_caption:back"]
    build = _turn_text(_req(trace, "generate")[0])
    assert "TARGET DESCRIPTION iso" in build and "TARGET DESCRIPTION back" in build


# --------------------------------------------------------------------------- both
def test_both_logs_disagreements_and_tells_builder_to_follow_the_image():
    _, _, trace = _run({"describe": "both"})
    order = [r["stage"] for r in trace.requests]
    assert order[:3] == ["describe_spec", "describe_caption:iso", "generate"]
    diff = [d for k, d in trace.events if k == "describe_disagreement"]
    fields = {f["field"] for f in diff[0]["fields"]}
    assert fields == {"materials.walls", "storeys"}
    build = _turn_text(_req(trace, "generate")[0])
    assert build.index("DESIGN SPEC") < build.index("TARGET DESCRIPTION")
    assert "follow the TARGET image and its description" in build
    assert "materials.walls (spec bricks, image terracotta)" in build


def test_disagreements_compare_normalized_names_only_where_both_say_something():
    a = dm.normalize(SPEC)
    b = dm.normalize({**SPEC, "materials": {"walls": "Bricks", "roof": ""},
                      "roof": {"shape": "Gable"}})
    assert dm.disagreements(a, b) == []


# --------------------------------------------------------------------------- prose
def test_prose_format_has_no_json_rule_and_passes_text_through():
    llm = _llm(caption="A small red house with a flat roof.", spec="Two storeys, bricks.")
    llm.rules[0] = (dm.SPEC_TAG, "Two storeys, bricks.")
    _, kit, trace = _run({"describe": "both", "caption_format": "prose"}, llm=llm)
    spec_turn = _turn_text(_req(trace, "describe_spec")[0])
    assert "Plain text only." in spec_turn and JSON_RULE not in spec_turn
    assert "Two storeys, bricks" in kit.image.requests[0].prompt
    build = _turn_text(_req(trace, "generate")[0])
    assert "A small red house with a flat roof." in build and "Two storeys, bricks." in build


# --------------------------------------------------------------------------- control
def test_shuffled_control_shows_another_prompts_caption():
    n = iter(range(1, 100))

    def cap_for(msgs):                  # the caption turn has no request: number them
        return json.dumps({**CAPTION, "summary": f"CAP {next(n)}"})

    llm = ScriptedLLM([PROG], rules=[(dm.CAPTION_TAG, cap_for)])
    kit = _kit({"describe": "after", "caption_control": "shuffled"}, llm=llm)
    _, _, t1 = _run({}, prompt="a red barn", kit=kit)
    _, _, t2 = _run({}, prompt="a stone tower", kit=kit)
    b1, b2 = (_turn_text(_req(t, "generate")[0]) for t in (t1, t2))
    # First prompt of the arm: a library caption; the second: the first one's caption.
    assert "CAP " not in b1 and dm.library_caption("a red barn")["text"] in b1
    assert "summary: CAP 1" in b2 and "CAP 2" not in b2
    m2 = _turn_text(_req(t2, "match0")[0])
    assert "CAP 1" in m2
    src = [d["source"] for k, d in t2.events if k == "describe_shuffled"]
    assert src == ["prompt:a red barn"]
    # Same calls as the real arm: one caption per build.
    assert len(_req(t1, "describe_caption:iso")) == len(_req(t2, "describe_caption:iso")) == 1


def test_shuffled_pools_are_per_arm():
    a = _kit({"describe": "after", "caption_control": "shuffled"})
    b = _kit({"describe": "after", "caption_control": "shuffled"})
    _run({}, prompt="a red barn", kit=a)
    _, _, t = _run({}, prompt="a stone tower", kit=b)
    assert [d["source"] for k, d in t.events if k == "describe_shuffled"][0].startswith("library")


def test_library_caption_is_deterministic():
    assert dm.library_caption("x") == dm.library_caption("x")
    assert all(dm.render_fields(dm.normalize(c, caption=True)) for c in dm.LIBRARY)


def test_library_captions_read_like_real_captions():
    """Palette block names only (what the captioner must use), the caption schema
    (no floor, no clear heights), and every entry a different building."""
    from blockgen.agentic.blockstate import PALETTE

    for c in dm.LIBRARY:
        names = [*c["materials"].values(), c["roof"]["material"]]
        assert all(v in PALETTE for v in names if v), names
        assert "floor" not in c["materials"] and "clear_heights" not in c["storeys"]
        assert dm.normalize(c, caption=True)["materials"] == c["materials"]


def test_library_caption_is_the_one_furthest_from_the_request():
    from blockgen.agentic.tasks import DETAILED_PROMPTS

    texts = [dm.render_fields(dm.normalize(c, caption=True)) for c in dm.LIBRARY]
    for p in DETAILED_PROMPTS:
        got = dm.library_caption(p)
        i = int(got["source"].split(":")[1])
        overlap = [len(dm._words(p) & dm._words(t)) for t in texts]
        assert overlap[i] == min(overlap), (p, overlap)
    # The built-in villa prompt does not get the office block (concrete, glass).
    villa = next(p for p in DETAILED_PROMPTS if "villa" in p)
    assert dm.library_caption(villa)["source"] != "library:3"


# --------------------------------------------------------------------------- mock
@pytest.mark.parametrize("fmt", ["json", "prose"])
def test_automock_answers_spec_and_caption_turns(fmt):
    ep_like = type("E", (), {"size": (20, 24, 18), "prompt": "a mill",
                             "kit": type("K", (), {"contexts": []})(),
                             "dream_spec": lambda self, k: type("D", (), {"view": "iso"})()})()
    mock = AutoMockProvider()
    for turn, kind in ((dm.spec_turn(ep_like, fmt), "spec"),
                       (dm.caption_turn(ep_like, "iso", b"png", fmt), "caption")):
        resp = mock.complete([text_message("system", "s"), turn])
        assert mock.decisions[-1] == f"describe:{kind}"
        # The spec echoes the request; the caption turn never sees it.
        if fmt == "json":
            d = json.loads(resp.text)
            assert d["footprint_blocks"][0] <= 20
            assert d["summary"].startswith("a mill") == (kind == "spec")
            assert ("uncertain" in d) == (kind == "caption")
            assert ("floor" in d["materials"]) == (kind == "spec")
        else:
            assert resp.text.startswith("summary: a mill") == (kind == "spec")
            assert "{" not in resp.text


# --------------------------------------------------------------------------- trace files
def test_trace_files_spec_caption_and_describe_md(tmp_path):
    kit = _kit({"describe": "both"})
    trace = Trace.for_kit(tmp_path / "b", kit)
    run_episode(kit, "a brick house", trace=trace)
    root = tmp_path / "b"
    spec = json.loads((root / "dreams" / "spec.json").read_text())
    assert spec["spec"]["materials"]["walls"] == "bricks" and spec["image_desc"]
    cap = json.loads((root / "dreams" / "iso_v0.caption.json").read_text())
    assert cap["caption"]["materials"]["walls"] == "terracotta" and cap["disagreements"]
    md = (root / "describe.md").read_text()
    assert "## Design spec" in md and "### iso v0" in md and "disagrees with the spec" in md
    # Caption and spec JSON are not mistaken for dream versions.
    assert [v["key"] for v in _dream_versions(root)] == ["iso"]


# --------------------------------------------------------------------------- recipes
NEW = ("dream-describe-before", "dream-caption", "dream-describe-both", "dream-caption-shuffled")


@pytest.mark.parametrize("name", NEW)
def test_new_recipes_validate_and_extend_dream(name):
    spec = load_recipe(name)
    validate_spec(spec)
    base = load_recipe("dream")
    assert spec.method == "dream" and spec.dreams == base.dreams and spec.image == base.image
    assert {k: v for k, v in spec.params.items() if k not in dm.PARAMS} == base.params
    assert spec.hash() != base.hash()


def test_caption_arms_share_dream_requests_with_their_twins():
    """dream-caption(-shuffled) ask for exactly dream's images; both for before's."""
    a, b = load_recipe("dream"), load_recipe("dream-caption")
    for k in ("dreams", "image", "seed", "dream_style", "contexts", "canvas"):
        assert getattr(a, k) == getattr(b, k), k


def test_batteries_list_the_describe_arms():
    names = [r.name for r in load_battery("describe-v1").recipes]
    assert names == ["classic-matched", "dream", *NEW]
    smoke = {r.name for r in load_battery("smoke-mock").recipes}
    assert set(NEW) <= smoke


@pytest.mark.slow
@pytest.mark.parametrize("name", NEW)
def test_new_recipes_run_offline(name, tmp_path):
    spec = load_recipe(name)
    kit = build_kit(dataclasses.replace(spec, cache="off"), mock=True)
    trace = Trace.for_kit(tmp_path / name, kit)
    res = run_episode(kit, "a brick house", trace=trace)
    assert not res.truncated, res.error
    assert (tmp_path / name / "describe.md").exists()


# --------------------------------------------------------------------------- staged
@pytest.mark.slow
def test_blueprint_exterior_gets_spec_and_final_pass_gets_caption(tmp_path):
    spec = load_recipe("blueprint")
    spec = dataclasses.replace(spec, cache="off", params={
        **spec.params, "describe": "both", "interior_dreams": False, "max_iters": 1})
    kit = build_kit(spec, mock=True)
    trace = Trace.for_kit(tmp_path / "bp", kit)
    res = run_episode(kit, "a brick house", trace=trace)
    assert not res.truncated, res.error
    iso = res.dreams["iso"][0]
    # The blueprint fixes storeys and materials; the spec adds only what it leaves
    # open (the mock spec's openings and features), never its own summary.
    assert "front door centred on the long side" in iso.prompt
    assert "log corner posts" in iso.prompt
    assert "one-storey oak cottage" not in iso.prompt and "walls of oak planks" not in iso.prompt
    rows = [json.loads(l) for l in (tmp_path / "bp" / "messages.jsonl").read_text().splitlines()]
    stages = [r["stage"] for r in rows]
    assert stages[0] == "describe_spec" and "describe_caption:iso" in stages
    finals = [r for r in rows if r["stage"].startswith("final")]
    assert finals and any("TARGET DESCRIPTION" in json.dumps(r["messages"]) for r in finals)


def test_dreams_first_prefetch_draws_the_expanded_image():
    """The suite's dream prefetch must ask for the SAME images the build phase will
    (with the spec in the prompt), or a paid run buys every dream twice."""
    from blockgen.agentic.suite import _dreams_first

    kit = _kit({"describe": "before"})
    _dreams_first(kit, ["a brick house"], [])
    pre = kit.image.requests[-1].prompt
    assert "A two-storey brick townhouse" in pre
    run_episode(kit, "a brick house", trace=RecordingTrace())
    assert kit.image.requests[-1].prompt == pre


# --------------------------------------------------------------------------- review fixes
def test_caption_has_its_own_schema_without_hidden_fields():
    """An exterior image shows neither the floor material nor the clear heights: the
    caption turn does not ask for them, normalize drops them, 'unknown' is empty,
    and disagreements never compare the floor."""
    ep_like = type("E", (), {"size": (20, 24, 18), "prompt": "a mill",
                             "dream_spec": lambda self, k: type("D", (), {"view": "iso"})()})()
    text = "\n".join(p.text for p in dm.caption_turn(ep_like, "iso", b"png", "json").parts
                     if p.kind == "text")
    assert '"floor"' not in text and "clear_heights" not in text and '"unknown"' in text
    assert text.count("{") == text.count("}")
    cap = dm.normalize({**CAPTION, "materials": {**CAPTION["materials"], "door": "unknown",
                                                 "floor": "dirt"},
                        "roof": {"shape": "Unknown", "material": "n/a"}}, caption=True)
    assert "floor" not in cap["materials"] and "clear_heights" not in cap["storeys"]
    assert cap["materials"]["door"] == "" and cap["roof"] == {"shape": "", "material": ""}
    spec = dm.normalize({**SPEC, "materials": {**SPEC["materials"], "floor": "stone"}})
    assert "materials.floor" not in {d["field"] for d in dm.disagreements(spec, cap)}
    assert "clear heights" not in dm.render_fields(cap)


def test_caption_turn_does_not_carry_the_request():
    _, _, trace = _run({"describe": "after"}, prompt="a very particular brick house")
    turn = json.dumps(_req(trace, "describe_caption:iso")[0]["messages"][-1].text)
    assert "very particular" not in turn and "(request:" not in turn


def test_spec_turn_shows_a_closed_json_template():
    _, _, trace = _run({"describe": "before"})
    turn = _turn_text(_req(trace, "describe_spec")[0])
    line = next(ln for ln in turn.splitlines() if ln.startswith('{"summary"'))
    assert line.endswith('"surroundings_to_omit": ["..."]}')
    assert line.count("{") == line.count("}")


def test_prose_spec_reaches_the_image_without_labels_or_underscores():
    prose = ("summary: a two-storey townhouse\nmaterials: walls bricks; trim stone_bricks; "
             "roof dark_oak_stairs\nCanvas: fits 20 x 20")
    llm = _llm()
    llm.rules[0] = (dm.SPEC_TAG, prose)
    _, kit, _ = _run({"describe": "before", "caption_format": "prose"}, llm=llm)
    img = kit.image.requests[0].prompt
    assert "stone bricks" in img and "dark oak stairs" in img
    assert "stone_bricks" not in img and "materials:" not in img and "Canvas" not in img
    assert dm.prose_image_desc(prose) == ("a two-storey townhouse. walls bricks; trim stone "
                                          "bricks; roof dark oak stairs. fits 20 x 20")


def test_prose_both_marks_disagreements_as_not_computed():
    llm = _llm(caption="A small red house.", spec="Two storeys, bricks.")
    llm.rules[0] = (dm.SPEC_TAG, "Two storeys, bricks.")
    _, _, trace = _run({"describe": "both", "caption_format": "prose"}, llm=llm)
    assert [k for k, _ in trace.events if k == "describe_disagreement_skipped"]
    assert not [k for k, _ in trace.events if k == "describe_disagreement"]


def test_spec_without_expand_is_not_said_to_have_drawn_the_image():
    _, _, t0 = _run({"describe": "before", "expand_into_image": False})
    _, _, t1 = _run({"describe": "before"})
    b0, b1 = (_turn_text(_req(t, "generate")[0]) for t in (t0, t1))
    assert "drawn from this spec" not in b0 and "written separately" in b0
    assert "drawn from this spec" in b1 and "written separately" not in b1


def test_augment_desc_for_a_blueprint_adds_only_the_open_fields():
    kit = _kit({"describe": "before"})
    trace = RecordingTrace()

    class Ep:
        notes, prompt, dream_desc = {}, "a brick house", {}

    ep = Ep()
    ep.trace, ep.kit = trace, kit
    st = dm.DescribeState(dm.config("before"), spec=dm.normalize(SPEC))
    st.image_desc = dm.image_desc(st.spec)
    ep.notes = {"describe": st}
    full = dm.augment_desc(ep, "a brick house. 1 storey(s)")
    part = dm.augment_desc(ep, "a brick house. 1 storey(s)", blueprint=True)
    assert "A two-storey brick townhouse" in full and "2 storeys" in full
    assert part == "a brick house. 1 storey(s). front door centred. bay window. no street"
    assert [k for k, _ in trace.events if k == "describe_spec_partial"]
    st.spec = None                                  # a prose spec: nothing appended
    assert dm.augment_desc(ep, "x", blueprint=True) == "x"


def test_turnaround_fallback_replaces_the_spec_description_the_critic_rejected():
    """With describe expanding, every per-view key starts on the spec description;
    a sheet critic's rewrite must still reach the per-view fallback dreams."""
    from blockgen.agentic import sheets
    from blockgen.agentic.imagegen import GRID_HINT

    from .test_method_turnaround import SheetGen, events, run

    verdicts = iter([{"consistent": False, "score": 2, "revised_prompt": "one stone hall"},
                     {"consistent": False, "score": 2, "revised_prompt": "one stone hall"}])
    rules = [(sheets.SHEET_REVIEW_TAG, lambda _m: json.dumps(next(verdicts))),
             (dm.SPEC_TAG, json.dumps(SPEC))]
    _, _, trace, image = run({"refine": 1, "assign": "fixed", "max_iters": 0,
                              "describe": "before"},
                             image=SheetGen(bad=True), rules=rules, budget={"max_images": 6})
    assert events(trace, "turnaround_fallback")
    sheet = [r for r in image.requests if r.view.startswith(GRID_HINT)]
    assert "A two-storey brick townhouse" in sheet[0].prompt
    per_view = [r for r in image.requests if not r.view.startswith(GRID_HINT)]
    assert len(per_view) == 4
    assert all("one stone hall" in r.prompt for r in per_view)
    assert not any("A two-storey brick townhouse" in r.prompt for r in per_view)
