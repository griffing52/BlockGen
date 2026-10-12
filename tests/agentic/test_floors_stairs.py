"""``floors``: a storey's region must be able to hold a working stair up.

A stair's top step and its stairwell sit in the slab of the storey ABOVE, one layer
over the lower storey's box. Before ``ss.storey_stair_boxes`` the lower storey had
every write there reverted and the upper storey (or its ``shell: det`` slab fill)
closed the slab over the run, so the upper floor of a ``floors`` build was
unreachable by construction.
"""

from __future__ import annotations

import re

import blockgen.agentic.recipes.floors  # noqa: F401  (registers the method)
from blockgen.agentic.blockstate import resolve_block
from blockgen.agentic.blueprint import Blueprint, mock_blueprint
from blockgen.agentic.episode import run_episode
from blockgen.agentic.providers_cli import AutoMockProvider
from blockgen.agentic.recipes import steps_staged as ss
from blockgen.agentic.recipes.floors import floors
from blockgen.agentic.region import world_arrays

from .fakes import RecordingTrace, ScriptedLLM, make_fake_kit

SIZE = (32, 32, 32)


def test_storey_regions_hold_the_stair_top_and_stay_disjoint():
    bp = mock_blueprint(SIZE, storeys=2)
    assert bp.stairs
    lo, hi = ss.storey_units(bp)
    m0, m1 = lo.region.mask(SIZE), hi.region.mask(SIZE)
    assert not (m0 & m1).any()
    st = bp.stairs[0]
    base = bp.storey(st.from_storey)
    cells = st.cells(base.height)
    for (x, dy, z) in cells:                       # every step, the top one included
        assert m0[x, base.floor_y + dy, z], (x, dy, z)
    up = bp.storey(1).floor_y
    for (x, _dy, z) in cells[-3:]:                 # the stairwell is storey 0's to build
        assert m0[x, up, z] and not m1[x, up, z]
        assert not m1[x, up + 1, z] and not m1[x, up + 2, z]   # arrival headroom
    # the rest of the upper slab is still storey 1's floor
    it = bp.footprint_of(1).interior()
    free = [(x, z) for x in range(it.x0, it.x1 + 1) for z in range(it.z0, it.z1 + 1)
            if (x, z) not in {(c[0], c[2]) for c in cells[-3:]}]
    assert all(m1[x, up, z] and not m0[x, up, z] for x, z in free)
    # a single storey has nothing extra
    one = ss.storey_units(mock_blueprint(SIZE, storeys=1))[0]
    assert len(one.region.allow) == 1 and one.region.deny == ()


_STAIRS = re.compile(r"first step at x (\d+), y (\d+), z (\d+); top step at x (\d+), "
                     r"y (\d+), z (\d+)")
_REGION = re.compile(r"ALLOWED REGION: x (\d+)\.\.(\d+), y (\d+)\.\.(\d+), z (\d+)\.\.(\d+)")


def test_floors_stair_survives_the_upper_slab_and_storey():
    """Storey 0 builds the planned stair (told where by its turn); storey 1's det
    slab and a builder that re-floors its whole region must not close it."""
    auto = AutoMockProvider()

    def reply(msgs):
        text = msgs[-1].text
        if "UNIT BUILD: storey" not in text:
            return auto._decide(msgs)[1]
        m = _STAIRS.search(text)
        if m is not None:
            x0, y0, z0, x1, y1, z1 = (int(v) for v in m.groups())
            n = y1 - y0
            dx, dz = (x1 - x0) // max(1, n), (z1 - z0) // max(1, n)
            return "\n".join(f"set oak_stairs {x0 + i * dx} {y0 + i} {z0 + i * dz}"
                             for i in range(n + 1)) + "\n"
        r = _REGION.search(text)
        x0, x1, y0, _y1, z0, z1 = (int(v) for v in r.groups())
        return (f"fill birch_planks {x0} {y0} {z0} {x1} {y0} {z1}\n"
                f"set lantern {x0 + 1} {y0 + 1} {z0 + 1}\n")

    kit = make_fake_kit(llm=ScriptedLLM([reply]), method=floors, dreams=[], image=None,
                        mock=True, spec_kw={"params": {
                            "stop_at": 2.0, "shell": "det", "roof": "det", "max_iters": 0,
                            "exterior_dream": False, "storey_dreams": False},
                            "select": "best"})
    trace = RecordingTrace()
    res = run_episode(kit, "a two storey townhouse", trace=trace)
    assert not res.truncated, res.error
    bp = Blueprint.from_dict([d for k, d in trace.events if k == "blueprint"][0]["blueprint"])
    assert len(bp.storeys) == 2 and bp.stairs
    ids, _ = world_arrays(res.attempts[res.best_idx].world)
    stair_id = resolve_block("oak_stairs")[0]
    air = resolve_block("air")[0]
    st = bp.stairs[0]
    base = bp.storey(st.from_storey)
    cells = st.cells(base.height)
    for (x, dy, z) in cells:
        assert ids[x, base.floor_y + dy, z] == stair_id, (x, dy, z)
    for (x, _dy, z) in cells[-3:-1]:
        assert ids[x, base.ceiling_y, z] == air, (x, z)
