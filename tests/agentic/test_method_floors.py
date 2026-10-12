"""The ``floors`` method end to end, offline: storeys built bottom up, each inside
its own region, with storey dreams conditioned on the storey below."""

from __future__ import annotations

import numpy as np
import pytest

import blockgen.agentic.recipes.floors  # noqa: F401  (registers the method)
from blockgen.agentic.blueprint import Blueprint
from blockgen.agentic.episode import Episode, run_episode
from blockgen.agentic.kit import build_kit
from blockgen.agentic.providers_cli import AutoMockProvider
from blockgen.agentic.recipe import load_recipe
from blockgen.agentic.recipes.floors import floors

from .fakes import RecordingTrace, make_fake_kit

SIZE = (32, 32, 32)


def run(params=None, *, dreams=("iso",), image=True):
    kw = {} if image else {"image": None}
    kit = make_fake_kit(llm=AutoMockProvider(), method=floors, dreams=list(dreams), mock=True,
                        spec_kw={"params": {"stop_at": 2.0, **(params or {})}, "select": "best"}, **kw)
    trace = RecordingTrace()
    res = run_episode(kit, "a two storey townhouse", trace=trace)
    return res, kit, trace


def plan_of(trace) -> Blueprint:
    return Blueprint.from_dict([d for k, d in trace.events if k == "blueprint"][0]["blueprint"])


def test_floors_bottom_up_region_scoped():
    res, kit, trace = run()
    assert not res.truncated, res.error
    bp = plan_of(trace)
    assert len(bp.storeys) == 2
    st = [a.stage for a in res.attempts]
    first = [s for s in st if s in ("u_storey0", "u_storey1")]
    assert first == ["u_storey0", "u_storey1"]
    assert st.index("u_storey0") < st.index("u_storey1") < st.index("roof")
    for a in res.attempts:
        if not a.stage.startswith("u_storey"):
            continue
        k = int(a.stage[len("u_storey")])
        assert a.extra.get("region") == f"storey{k}"
        base = [b for b in res.attempts[:a.idx] if b.accepted]
        if not base:
            continue
        prev = base[-1].world.to_structure(crop=False)
        cur = a.world.to_structure(crop=False)
        changed = prev.block_ids != cur.block_ids
        box = bp.storey_box(k, floor=True).mask(SIZE)
        assert not (changed & ~box).any(), a.stage
    assert {"u_storey0", "u_storey1"} <= set(res.dreams)
    assert any(s.startswith("final") for s in st)
    assert res.best_idx >= st.index("roof")


def test_storey_dream_refs_below_then_exterior():
    res, kit, _ = run()
    reqs = [r for r in kit.image.requests if r.view == "room"]
    assert len(reqs) == 2
    # storey 0: nothing below, only the exterior; storey 1: the storey below + exterior
    assert len(reqs[0].refs) == 1 and len(reqs[1].refs) == 2
    ext = res.dreams["iso"][-1].png
    assert reqs[0].refs[0] == ext and reqs[1].refs[1] == ext
    iso = [r for r in kit.image.requests if r.view == "iso"]
    assert iso and iso[0] is kit.image.requests[0]          # the exterior is dreamt first


def test_floors_det_shell_per_storey():
    res, _, _ = run({"shell": "det", "roof": "det"})
    assert not res.truncated, res.error
    st = [a.stage for a in res.attempts]
    assert st.index("shell0") < st.index("u_storey0") < st.index("shell1") < st.index("u_storey1")


def test_floors_without_images_or_rooms():
    res, kit, trace = run({"exterior_dream": False, "storey_dreams": False,
                           "plan_rooms": False}, dreams=(), image=False)
    assert not res.truncated, res.error
    assert set(res.dreams) == {"blueprint"}
    assert "roof" in [a.stage for a in res.attempts]


def test_floors_validation():
    kit = make_fake_kit(llm=AutoMockProvider(), method=floors, mock=True)
    ep = Episode("x", kit)
    with pytest.raises(ValueError, match="exterior_ref"):
        floors(ep, exterior_ref="massing")
    with pytest.raises(ValueError, match="storey_refs"):
        floors(ep, storey_refs=("upstairs",))
    with pytest.raises(ValueError, match="shell"):
        floors(ep, shell="llm")


@pytest.mark.slow
def test_recipe_floors_mock():
    kit = build_kit(load_recipe("floors"), mock=True, cache_mode="off")
    trace = RecordingTrace()
    res = run_episode(kit, "a two storey townhouse", trace=trace)
    assert not res.truncated, res.error
    st = [a.stage for a in res.attempts]
    assert "u_storey0" in st and "roof" in st
    assert np.isfinite(res.best_idx)
