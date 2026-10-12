"""Method ``floors``: plan first, then build the house storey by storey, bottom up.

    1. PLAN      LLM blueprint JSON (repaired); only the storeys, footprint, roof,
                 materials and (``plan_rooms``) the room list are used
    2. EXTERIOR  the primary dream FIRST (nothing is built yet, so it can only be
                 anchored on the plan drawing, ``exterior_ref: schematic``, or on
                 nothing)
    3. STOREYS   per storey k: optional deterministic slab + walls (``shell: det``),
                 a storey dream (refs: the storey below as built, the exterior),
                 then build / repair / compare inside the storey's region (its
                 slab, walls and everything up to floor_y + height, plus the
                 slab cells above its planned stair, minus the arrival of the
                 stair from below: ``ss.storey_stair_boxes``)
    4. ROOF      (``select_from`` set first), 5. FINAL, 6. SELECT as ``blueprint``

Why it exists next to ``blueprint``: it changes one thing, the unit (a storey
instead of a room), so ``floors`` vs ``blueprint`` isolates unit granularity. A
storey is a bigger, cheaper unit than a room (fewer turns) but each turn changes
a smaller fraction of what its score sees.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from blockgen.agentic import describe as describe_mod
from blockgen.agentic.core import EpisodeLike, register
from blockgen.agentic.recipes import steps_staged as ss
from blockgen.agentic.recipes.blueprint import check_common, exterior_stage, plan_stage
from blockgen.agentic.region import Region

#: Share of the build's usd cap per stage (storeys are split among the storeys).
DEFAULT_STAGE_BUDGET: Dict[str, Optional[float]] = {
    "plan": 0.10, "storeys": 0.60, "roof": 0.10, "final": 0.20}
STOREY_REFS = ("below", "exterior", "schematic", "cutaway")


@register("method", "floors", prefetch_dreams=False)
def floors(ep: EpisodeLike, *, max_storeys: int = 2, plan_rooms: bool = True,
           plan_retries: int = 1,
           shell: str = "none",
           exterior_dream: bool = True, exterior_ref: str = "none",
           refine: int = 0,
           storey_dreams: bool = True,
           storey_refs: Sequence[str] = ("below", "exterior"),
           max_storey_iters: int = 2, storey_mode: str = "patch",
           local_coords: bool = False, unit_score_terms: Optional[Dict[str, float]] = None,
           unit_eps: float = 0.0, roof: str = "llm",
           max_iters: int = 2, stop_at: float = 0.85, patience: int = 2,
           final_feedback: str = "both", final_mode: str = "patch",
           stage_budget: Optional[Dict[str, float]] = None,
           describe: str = "none", caption_format: str = "json",
           expand_into_image: bool = True, show_spec: bool = True,
           caption_keys: str = "primary", caption_in_match: bool = True,
           caption_control: str = "none") -> None:
    """Plan, exterior dream, storey-by-storey builds (bottom up), roof, exterior pass."""
    # -- validation (before any call) ---------------------------------------
    if shell not in ("none", "det"):
        raise ValueError(f"shell must be none|det, got {shell!r}")
    if exterior_ref not in ("none", "schematic"):
        raise ValueError(f"exterior_ref must be none|schematic (nothing is built before the "
                         f"exterior dream), got {exterior_ref!r}")
    if roof not in ("det", "llm", "none"):
        raise ValueError(f"roof must be det|llm|none, got {roof!r}")
    bad = [r for r in storey_refs if r not in STOREY_REFS]
    if bad:
        raise ValueError(f"storey_refs {bad} not in {STOREY_REFS}")
    check_common(ep, modes={"storey_mode": storey_mode, "final_mode": final_mode},
                 feedbacks={"final_feedback": final_feedback},
                 needs_image={"exterior_dream": exterior_dream,
                              "storey_dreams": storey_dreams})
    ss.unit_terms(ep, unit_score_terms)
    shares = {**DEFAULT_STAGE_BUDGET, **(stage_budget or {})}
    # Every storey turn carries its own context and targets (ss.context_parts).
    ep.inject_first_turn = False

    # 0. DESCRIBE before: the spec reaches the exterior dream's description; a
    # caption of the exterior dream (after/both) reaches the final exterior pass.
    describe_mod.setup(ep, describe_mod.config(
        describe, caption_format, expand_into_image, show_spec, caption_keys,
        caption_in_match, caption_control))

    # 1. PLAN
    max_rooms = 6 if plan_rooms else 0
    bp = plan_stage(ep, source="llm", max_rooms=max_rooms, max_storeys=max_storeys,
                    retries=plan_retries, frac=shares.get("plan"))
    if not plan_rooms and bp.rooms:
        ep.trace.event("floors_rooms_ignored", n=len(bp.rooms))

    # 2. EXTERIOR DREAM (first: nothing is built yet)
    if exterior_dream:
        exterior_stage(ep, bp, ref=exterior_ref, refine=refine, refine_min_score=7,
                       frac=shares.get("exterior"))

    # 3. STOREYS, bottom up
    units = ss.storey_units(bp)
    with ss.stage(ep, "storeys", shares.get("storeys")) as led:
        for i, u in enumerate(units):
            cap = led.budget.usd
            share = None if cap is None else max(0.0, cap - led.spent_usd) / (len(units) - i)
            with ss.stage(ep, f"storey:{u.storey}", None, usd=share):
                if shell == "det":
                    parts = ("foundation" if u.storey == 0 else "floors", "walls", "corners")
                    # The slab fill must not close the stairwell the storey below
                    # built into it (ss.storey_stair_boxes: the unit's deny boxes).
                    keep = (Region.whole(ep.size, deny=u.region.deny, name=f"shell{u.storey}")
                            if u.region.deny else None)
                    ep.try_(f"shell{u.storey}", bp.shell_dsl(parts=parts, storeys=[u.storey]),
                            ir=True, patch=True, region=keep, accept=ss._always)
                target = (ss.unit_dream(ep, bp, u, refs=storey_refs) if storey_dreams else None)
                ss.build_unit(ep, bp if plan_rooms else _without_rooms(bp), u, target=target,
                              iters=max_storey_iters, mode=storey_mode,
                              local_coords=local_coords, terms=unit_score_terms or {},
                              eps=unit_eps, feedback=final_feedback)

    # 4. ROOF (selection sees complete buildings only from here on)
    ep.select_from = len(ep.attempts)
    protect = ss.interior_boxes(bp if plan_rooms else _without_rooms(bp))
    with ss.stage(ep, "roof", shares.get("roof")):
        ss.build_roof(ep, bp, how=roof, protect=protect)

    # 5. FINAL
    ss.fresh_dialogue(ep)
    with ss.stage(ep, "final", shares.get("final")):
        ss.final_match(ep, max_iters=max_iters, feedback=final_feedback, mode=final_mode,
                       protect=protect, patience=patience, stop_at=stop_at)
    ss.render_units(ep, units, start=ep.select_from or 0)


def _without_rooms(bp):
    """The plan minus its rooms (``plan_rooms: false``): the storey turns then show
    no room list and no schematic, and the protected boxes are whole storeys."""
    out = bp.copy()
    out.rooms = []
    return out


__all__ = ["DEFAULT_STAGE_BUDGET", "STOREY_REFS", "floors"]
