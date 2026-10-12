"""Method ``blueprint``: plan first, then build the house room by room.

    1. PLAN      a typed Blueprint: LLM JSON (``llm``), a plan dreamed on grid
                 paper and read by a VLM (``image``), LLM JSON checked against
                 a pictured plan (``both``), or a prose design brief transcribed
                 into JSON (``text``); always repaired
    2. SHELL     foundation, floors, walls, partitions, openings, stairs: the
                 blueprint's deterministic DSL (``det``) or the builder (``llm``)
    3. EXTERIOR  the recipe's primary dream, an img2img of OUR massing render
                 (shell + roof), so it shares the camera, footprint and roof line
                 of every render the loop scores; then every score is re-anchored
    4. ROOMS     per room: a room dream edited from a cutaway of the EMPTY room,
                 then build / repair / compare inside the room's region, scored
                 and accepted on a cutaway of that room only
    5. ROOF      deterministic or LLM; selection considers complete buildings
                 only from here on (``ep.select_from``)
    6. FINAL     exterior compare (or critique) rounds that cannot touch the
                 finished room interiors
    7. SELECT    (run_episode) ``select: best`` over attempts[select_from:] by
                 ``final_score`` (dream-independent, so like with like across stages)

Why stages: the pilot's single-image loop barely moved its score, because a whole
house is too much to change per turn and the dream/render gap is mostly style. A
room is a unit where one turn's change is a large fraction of the image, the
target shares the render's camera, and a write outside the room is reverted and
reported instead of silently breaking the rest of the house.

Each stage runs on its own slice of the build's usd cap
(``steps_staged.stage``): default plan 0.10, shell 0.10, rooms 0.55 (divided
adaptively: what is left / rooms left), roof 0.05, final 0.20. A stage that runs
out ends quietly; the build goes on with the next one.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from blockgen.agentic import describe as describe_mod
from blockgen.agentic import turns
from blockgen.agentic.blueprint import ROOM_ORDERS, default_blueprint
from blockgen.agentic.core import BudgetExceeded, EpisodeLike, register
from blockgen.agentic.recipes import steps_staged as ss

#: Share of the build's usd cap per stage (rooms are split among the rooms).
DEFAULT_STAGE_BUDGET: Dict[str, Optional[float]] = {
    "plan": 0.10, "shell": 0.10, "rooms": 0.55, "roof": 0.05, "final": 0.20}
ROOM_REFS = ("cutaway", "exterior", "schematic")


def check_common(ep: Any, *, modes: Dict[str, str], feedbacks: Dict[str, str],
                 needs_image: Dict[str, bool]) -> None:
    """Entry validation shared by the staged methods (ValueError before any call)."""
    for name, mode in modes.items():
        if mode not in turns.MODE_INSTRUCTIONS:
            raise ValueError(f"{name} must be one of {sorted(turns.MODE_INSTRUCTIONS)}, "
                             f"got {mode!r}")
        if mode == "patch" and not getattr(ep.kit.actions, "patchable", False):
            raise ValueError(f"{name} 'patch' needs a patchable action space; "
                             f"'{getattr(ep.kit.actions, 'name', '?')}' is not")
    for name, fb in feedbacks.items():
        if fb not in ss.FEEDBACKS:
            raise ValueError(f"{name} must be one of {ss.FEEDBACKS}, got {fb!r}")
    for name, on in needs_image.items():
        if on and ep.kit.image is None:
            raise ValueError(f"{name} needs an image generator; recipe "
                             f"'{ep.spec.name}' has none (set it false, or set image:)")


def stage_shares(stage_budget: Optional[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    out = dict(DEFAULT_STAGE_BUDGET)
    out.update(stage_budget or {})
    return out


def plan_stage(ep: Any, *, source: str, max_rooms: int, max_storeys: int, retries: int,
               frac: Optional[float]):
    """Stage 1, shared with ``floors``: the plan, or the default plan if the stage
    ran out of budget before one existed. Recorded either way."""
    bp = None
    with ss.stage(ep, "plan", frac):
        bp = ss.make_blueprint(ep, source=source, max_rooms=max_rooms,
                               max_storeys=max_storeys, retries=retries)
    if bp is None:
        ep.trace.event("blueprint_fallback", stage="plan", reason="stage budget")
        bp = default_blueprint(tuple(ep.size), ep.prompt).repaired(
            tuple(ep.size), max_rooms=max_rooms, max_storeys=max_storeys)
    ss.record_blueprint(ep, bp, tag=source)
    return bp


def exterior_stage(ep: Any, bp: Any, *, ref: str, refine: int, refine_min_score: float,
                   frac: Optional[float] = None) -> None:
    """Stage 3, shared with ``floors``: the anchored exterior dream, then a
    rescore (attempts so far were scored before the dream existed). An image cap
    ends the stage, not the build: the rooms can still be built from text."""
    with ss.stage(ep, "exterior", frac):
        try:
            ss.exterior_dream(ep, bp, ref=ref, refine=refine, refine_min_score=refine_min_score)
        except BudgetExceeded as exc:
            if not ss.is_image_cap(exc):
                raise       # out of money: stage() ends the stage, or the build ends
            ep.trace.event("exterior_dream_budget_stop", error=str(exc))
    ep.rescore_all()
    describe_mod.refresh(ep)        # a describe stage captions the exterior dream


@register("method", "blueprint", prefetch_dreams=False)
def blueprint(ep: EpisodeLike, *,
              plan_source: str = "llm",
              max_rooms: int = 4, max_storeys: int = 2, plan_retries: int = 1,
              shell: str = "det",
              roof: str = "det",
              exterior_dream: bool = True,
              exterior_ref: str = "massing",
              refine: int = 0, refine_min_score: float = 7,
              interior_dreams: bool = True,
              room_refs: Sequence[str] = ("cutaway", "exterior"),
              room_dream_shuffle: bool = False,
              room_order: str = "public_first",
              max_room_iters: int = 2, room_mode: str = "patch",
              local_coords: bool = False,
              room_score: Optional[Dict[str, float]] = None, room_eps: float = 0.0,
              max_iters: int = 2, stop_at: float = 0.85, patience: int = 2,
              final_feedback: str = "both", final_mode: str = "patch",
              protect_interiors: bool = True,
              stage_budget: Optional[Dict[str, float]] = None,
              describe: str = "none", caption_format: str = "json",
              expand_into_image: bool = True, show_spec: bool = True,
              caption_keys: str = "primary", caption_in_match: bool = True,
              caption_control: str = "none") -> None:
    """Plan, shell, anchored exterior dream, room-by-room interiors, roof, exterior pass."""
    # -- validation (before any call) ---------------------------------------
    if plan_source not in ss.PLAN_SOURCES:
        raise ValueError(f"plan_source must be one of {ss.PLAN_SOURCES}, got {plan_source!r}")
    if shell not in ("det", "llm"):
        raise ValueError(f"shell must be det|llm, got {shell!r}")
    if roof not in ("det", "llm"):
        raise ValueError(f"roof must be det|llm, got {roof!r}")
    if exterior_ref not in ss.EXTERIOR_REFS:
        raise ValueError(f"exterior_ref must be one of {ss.EXTERIOR_REFS}, got {exterior_ref!r}")
    if room_order not in ROOM_ORDERS:
        raise ValueError(f"room_order must be one of {ROOM_ORDERS}, got {room_order!r}")
    bad = [r for r in room_refs if r not in ROOM_REFS]
    if bad:
        raise ValueError(f"room_refs {bad} not in {ROOM_REFS}")
    check_common(ep, modes={"room_mode": room_mode, "final_mode": final_mode},
                 feedbacks={"final_feedback": final_feedback},
                 needs_image={"exterior_dream": exterior_dream,
                              "interior_dreams": interior_dreams,
                              f"plan_source '{plan_source}'":
                                  plan_source in ss.IMAGE_PLAN_SOURCES})
    if room_dream_shuffle and not interior_dreams:
        raise ValueError("room_dream_shuffle needs interior_dreams: true (it shuffles the "
                         "room dreams; without them the control would silently equal the "
                         "exterior-only arm)")
    ss.unit_terms(ep, room_score)              # unknown term names fail here
    shares = stage_shares(stage_budget)
    # Every unit turn carries its own context and targets (ss.context_parts).
    ep.inject_first_turn = False

    # 0. DESCRIBE before: the spec reaches the exterior dream's description; a
    # caption of the exterior dream (after/both) reaches the final exterior pass.
    describe_mod.setup(ep, describe_mod.config(
        describe, caption_format, expand_into_image, show_spec, caption_keys,
        caption_in_match, caption_control))

    # 1. PLAN
    bp = plan_stage(ep, source=plan_source, max_rooms=max_rooms, max_storeys=max_storeys,
                    retries=plan_retries, frac=shares.get("plan"))

    # 2. SHELL
    with ss.stage(ep, "shell", shares.get("shell")):
        ss.build_shell(ep, bp, how=shell)

    # 3. EXTERIOR DREAM
    if exterior_dream:
        exterior_stage(ep, bp, ref=exterior_ref, refine=refine,
                       refine_min_score=refine_min_score, frac=shares.get("exterior"))

    # 4. ROOMS
    units = ss.room_units(bp, order=room_order, max_units=max_rooms)
    with ss.stage(ep, "rooms", shares.get("rooms")) as rooms_ledger:
        targets = None
        if room_dream_shuffle and interior_dreams:
            # Control: every room is matched against ANOTHER room's dream (same
            # images and turns as the real arm, wrong target).
            if len(units) < 2:
                # A rotation of one is the identity: this arm is NOT a control here.
                ep.trace.event("room_dream_shuffle_degenerate", n_units=len(units))
            dreamt = [ss.unit_dream(ep, bp, u, refs=room_refs) for u in units]
            targets = dreamt[1:] + dreamt[:1]
            ep.trace.event("room_dream_shuffle",
                           pairs={u.name: (units[(i + 1) % len(units)].name)
                                  for i, u in enumerate(units)})
        for i, u in enumerate(units):
            cap = rooms_ledger.budget.usd
            share = (None if cap is None
                     else max(0.0, cap - rooms_ledger.spent_usd) / (len(units) - i))
            with ss.stage(ep, f"room:{u.name}", None, usd=share):
                if targets is not None:
                    target = targets[i]
                else:
                    target = (ss.unit_dream(ep, bp, u, refs=room_refs)
                              if interior_dreams else None)
                ss.build_unit(ep, bp, u, target=target, iters=max_room_iters,
                              mode=room_mode, local_coords=local_coords, terms=room_score or {},
                              eps=room_eps, feedback=final_feedback)

    # 5. ROOF (selection sees complete buildings only from here on)
    ep.select_from = len(ep.attempts)
    protect = ss.interior_boxes(bp) if protect_interiors else []
    with ss.stage(ep, "roof", shares.get("roof")):
        ss.build_roof(ep, bp, how=roof, protect=protect)

    # 6. FINAL
    ss.fresh_dialogue(ep)
    with ss.stage(ep, "final", shares.get("final")):
        ss.final_match(ep, max_iters=max_iters, feedback=final_feedback, mode=final_mode,
                       protect=protect, patience=patience, stop_at=stop_at)
    ss.render_units(ep, units, start=ep.select_from or 0)


__all__ = ["DEFAULT_STAGE_BUDGET", "ROOM_REFS", "blueprint", "check_common",
           "exterior_stage", "plan_stage", "stage_shares"]
