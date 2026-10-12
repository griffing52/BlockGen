"""dreams.py: presets, prompt rendering, ordering, verdict normalization."""

from __future__ import annotations

import pytest

from blockgen.agentic import dreams
from blockgen.agentic.core import VIEWS, DreamSpec, MethodSpec
from blockgen.agentic.kit import resolve_dreams


class _Ep:
    def __init__(self, prompt="a red barn", style=None, canvas=(32, 24, 16)):
        self.prompt = prompt
        self.spec = MethodSpec(name="t", canvas=canvas, dream_style=style)


def test_presets_are_valid_and_resolve_through_the_kit():
    for key, ds in dreams.DREAMS.items():
        assert ds.key == key
        for v in ((ds.view,) if ds.view else ()) + ds.view_search:
            assert v in VIEWS
    assert dreams.DREAMS["iso"].view_search == ("iso_back", "iso_left", "iso_right")
    assert dreams.DREAMS["interior"].view == "cutaway"
    assert dreams.DREAMS["plan"].view == "plan" and dreams.DREAMS["plan"].condition_on == ("iso",)
    assert dreams.DREAMS["mood"].view is None and dreams.DREAMS["mood"].show_builder
    got = resolve_dreams(["iso", "back", "interior", "plan", "mood"])
    assert [d.key for d in got] == ["iso", "back", "interior", "plan", "mood"]
    over = resolve_dreams([{"key": "iso", "weight": 0.25}])[0]
    assert over.weight == 0.25 and over.view_search == dreams.DREAMS["iso"].view_search


def test_expand_turnaround():
    assert dreams.expand(["turnaround", "plan"]) == ["iso", "back", "left", "right", "plan"]
    assert dreams.expand(["iso", "turnaround"]) == ["iso", "back", "left", "right"]
    inline = {"key": "x", "template": "{desc}"}
    assert dreams.expand([inline, "mood"]) == [inline, "mood"]


def test_render_prompt_fills_fields_and_pins_style():
    ep = _Ep()
    p = dreams.render_prompt(dreams.DREAMS["iso"], ep)
    assert p.startswith(dreams.DEFAULT_STYLE) and "a red barn" in p
    assert "32 x 16 blocks of footprint and 24 blocks of height" in p
    p2 = dreams.render_prompt(dreams.DREAMS["iso"], ep, desc="a simpler barn.")
    assert "a simpler barn" in p2 and "a red barn" not in p2 and p2.startswith(dreams.DEFAULT_STYLE)
    ep_style = _Ep(style="PIXEL STYLE")
    assert dreams.render_prompt(dreams.DREAMS["iso"], ep_style).startswith("PIXEL STYLE")
    odd = DreamSpec("x", "{style} {desc} {unknown} {W}")
    assert "{unknown}" in dreams.render_prompt(odd, ep) and dreams.render_prompt(odd, ep).endswith("32")


def test_topo_orders_parents_first_and_detects_cycles():
    a = DreamSpec("plan", "", view="plan", condition_on=("iso",))
    b = DreamSpec("iso", "", view="iso")
    c = DreamSpec("back", "", view="iso_back", condition_on=("plan",))
    assert [d.key for d in dreams.topo([a, c, b])] == ["iso", "plan", "back"]
    assert [d.key for d in dreams.dependants([a, c, b], "iso")] == ["plan", "back"]
    with pytest.raises(ValueError, match="cycle"):
        dreams.topo([DreamSpec("x", "", condition_on=("y",)), DreamSpec("y", "", condition_on=("x",))])
    lone = DreamSpec("z", "", condition_on=("missing",))
    assert dreams.topo([lone]) == [lone]


def test_primary_prefers_scored_root():
    mood = dreams.DREAMS["mood"]
    assert dreams.primary([mood, dreams.DREAMS["plan"], dreams.DREAMS["iso"]]).key == "iso"
    assert dreams.primary([mood]).key == "mood"
    assert dreams.primary([]) is None


def test_normalize_verdict():
    v = dreams.normalize_verdict({"score": "6/10", "buildable": "no", "issues": "too big"})
    assert v["score"] == 6.0 and v["buildable"] is False and v["on_prompt"] is True
    assert v["issues"] == ["too big"] and v["keep_composition"] is False
    assert dreams.normalize_verdict({"score": "n/a"})["score"] == 0.0
    assert "{W} x {H} x {D}" not in dreams.DREAM_RUBRIC.format(W=1, H=2, D=3, view="iso")
