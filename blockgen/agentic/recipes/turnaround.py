"""Method ``turnaround``: four target views from ONE generated image.

    1. SHEET      one image: a 2x2 turnaround sheet of the building (runtime dream
                  ``sheet``, never scored); optionally conditioned on an iso dream
    2. REFINE     dream critic checks the sheet; regenerate while rejected
    3. SPLIT      cut the sheet into four tiles (sheets.split_grid); a bad split
                  falls back to today's per-view edits (event turnaround_fallback)
    4. FILL       each tile becomes the next version of its recipe dream (put_dream,
                  no image charge)
    5. BUILD      plan? -> generate -> symbolic repair
    6. ASSIGN     ``cyclic``: which tile is which camera, by silhouette IoU against
                  the first build's renders over the 8 cyclic orders; done ONCE
    7. MATCH      the dream method's match loop (no re-dream: the sheet is the target)

Why: per-view dreams are image edits of the iso dream, and an edit keeps its
reference's camera (pilot finding P2), so the back and side targets drift. One
sheet is drawn in one pass, so its panels show one building by construction.

Knobs:

    mode                'sheet' | 'per_view' (the within-method control: today's
                        per-view edits, nothing else changes)
    order               panel views in reading order (top-left first)
    assign              'cyclic' (reassign once after the first build) | 'fixed'
                        (trust the prompt's order) | 'free' (every key scored at
                        the best of all four views, an ablation)
    sheet_ref           'none' | 'iso' (dream the iso view alone first and pass it
                        as the sheet's reference; one more image)
    refine, refine_min_score, fallback_per_view
    max_iters, stop_at, patience, build_mode, feedback   the match loop
    repair_rounds, plan, n_examples                     the build phase

The recipe's dreams must include one per ``order`` view (the turnaround preset:
iso, back, left, right). ``assign`` applies only when tiles were used: per-view
dreams already know their camera.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence

from blockgen.agentic import describe as describe_mod
from blockgen.agentic import dreams as dreams_mod
from blockgen.agentic import sheets, turns
from blockgen.agentic.core import (VIEWS, BudgetExceeded, DreamSpec, EmptyBuild, EpisodeLike,
                                   register)
from blockgen.agentic.recipes import steps
from blockgen.agentic.recipes.dream import make_dreams, match_loop

SHEET_KEY = "sheet"
MODES = ("sheet", "per_view")
ASSIGNS = ("fixed", "cyclic", "free")
SHEET_REFS = ("none", "iso")


def _keys_for(ep: Any, order: Sequence[str]) -> Dict[str, str]:
    """view -> recipe dream key, for every view in ``order`` (ValueError otherwise)."""
    by_view = {ds.view: ds.key for ds in ep.kit.dreams if ds.view}
    missing = [v for v in order if v not in by_view]
    if missing:
        raise ValueError(f"turnaround: no recipe dream is scored at view(s) {missing}; "
                         f"dreams: {[(d.key, d.view) for d in ep.kit.dreams]} (use the "
                         f"turnaround preset: iso, back, left, right)")
    return {v: by_view[v] for v in order}


def _validate(ep: Any, *, mode: str, order: Sequence[str], assign: str, sheet_ref: str,
              build_mode: str, feedback: str) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if assign not in ASSIGNS:
        raise ValueError(f"assign must be one of {ASSIGNS}, got {assign!r}")
    if sheet_ref not in SHEET_REFS:
        raise ValueError(f"sheet_ref must be one of {SHEET_REFS}, got {sheet_ref!r}")
    if len(order) != 4 or len(set(order)) != 4:
        raise ValueError(f"order must name 4 distinct views (a 2x2 sheet), got {list(order)}")
    unknown = [v for v in order if v not in VIEWS]
    if unknown:
        raise ValueError(f"order: unknown view(s) {unknown}; known: {sorted(VIEWS)}")
    if build_mode not in turns.MODE_INSTRUCTIONS:
        raise ValueError(f"build_mode must be one of {sorted(turns.MODE_INSTRUCTIONS)}, "
                         f"got {build_mode!r}")
    if feedback not in ("raw", "score", "vdm", "both"):
        raise ValueError(f"feedback must be raw|score|vdm|both, got {feedback!r}")
    if build_mode == "patch" and not getattr(ep.kit.actions, "patchable", False):
        raise ValueError(f"build_mode 'patch' needs a patchable action space; "
                         f"'{getattr(ep.kit.actions, 'name', '?')}' is not")
    if ep.kit.image is None:
        raise ValueError("turnaround needs an image generator (recipe 'image:')")
    _keys_for(ep, order)


def review_sheet(ep: Any, png: bytes, order: Sequence[str]) -> Dict[str, Any]:
    """Dream critic verdict on a sheet; {} when unparseable (keep the sheet)."""
    raw = ep.ask(sheets.sheet_review(ep, png, order), role="dream_critic", history=False,
                 json_reply=True, stage="sheet_review")
    if not raw:
        ep.trace.event("dream_review_unparsed", key=SHEET_KEY)
        return {}
    return sheets.normalize_sheet_verdict(raw)


def make_sheet(ep: Any, order: Sequence[str], *, sheet_ref: str, refine: int,
               refine_min_score: float) -> Any:
    """Steps 1-2: register and generate the sheet, then refine it while rejected.
    Returns the final sheet DreamVersion."""
    ep.register_dream(DreamSpec(SHEET_KEY, sheets.SHEET_TEMPLATE, view=None, weight=0.0,
                                show_builder=False))
    refs: List[bytes] = []
    if sheet_ref == "iso":
        iso_key = _keys_for(ep, order)[order[0]]
        refs = [ep.dream(iso_key).png]
    hint = sheets_hint(order)
    ds = ep.dream_spec(SHEET_KEY)
    dv = ep.dream(SHEET_KEY, prompt=dreams_mod.render_prompt(ds, ep), refs=refs,
                  view_hint=hint)
    for _ in range(max(0, int(refine))):
        v = review_sheet(ep, dv.png, order)
        dv.critique = v
        if not v or (v["consistent"] and v["score"] >= refine_min_score):
            break
        if v["revised_prompt"]:
            ep.dream_desc[SHEET_KEY] = v["revised_prompt"]
        ep.trace.event("dream_regenerate", key=SHEET_KEY, score=v["score"], issues=v["issues"])
        try:
            dv = ep.dream(SHEET_KEY, prompt=dreams_mod.render_prompt(ds, ep), refs=refs,
                          seed=steps.next_seed(ep, SHEET_KEY), view_hint=hint)
        except BudgetExceeded as exc:
            ep.trace.event("refine_budget_stop", key=SHEET_KEY, error=str(exc))
            break
    return ep.latest_dream(SHEET_KEY)


def sheets_hint(order: Sequence[str]) -> str:
    """The ``view_hint`` of a sheet request (mock backends draw it, APIs ignore it)."""
    return "grid2x2:" + ",".join(order)


def fill_from_tiles(ep: Any, tiles: Sequence[bytes], assignment: Sequence[str], *,
                    generator: str, prompt: str = "") -> None:
    """Step 4: tile i becomes the next version of the dream at ``assignment[i]``."""
    keys = _keys_for(ep, assignment)
    for tile, view in zip(tiles, assignment):
        ep.put_dream(keys[view], tile, prompt=prompt, generator=generator)


def _missing_dreams(ep: Any) -> int:
    return sum(1 for ds in ep.kit.dreams if ep.latest_dream(ds.key) is None)


def per_view_fallback(ep: Any, *, refine: int, refine_min_score: float) -> bool:
    """Today's per-view edits for the dreams that have no version yet. False (and
    nothing generated) when their images do not fit the budget: a half-made set
    of targets would describe different buildings."""
    try:
        steps.ensure_image_budget(ep, _missing_dreams(ep))
    except BudgetExceeded as exc:
        ep.trace.event("turnaround_fallback_budget", error=str(exc))
        return False
    have = tuple(ds.key for ds in ep.kit.dreams if ep.latest_dream(ds.key) is not None)
    # A sheet critic's rewrite describes the building only (no layout words), so it
    # applies to the per-view dreams too: a rejected description should not return.
    # A describe stage set every key to the spec-expanded description before the
    # sheet; that one is replaced too (it is what the critic rejected). Any other
    # description a key already has is kept.
    desc = ep.dream_desc.get(SHEET_KEY)
    if desc:
        expanded = describe_mod.expanded_desc(ep)
        for ds in ep.kit.dreams:
            cur = ep.dream_desc.get(ds.key)
            if cur is None or (expanded and cur == expanded):
                ep.dream_desc[ds.key] = desc
    make_dreams(ep, refine=refine, refine_min_score=refine_min_score, skip=have)
    return True


def free_views(ep: Any, order: Sequence[str]) -> None:
    """assign='free': every turnaround key is scored at the best of all four views.
    The episode gets its own Kit copy, so the shared kit (other prompts) is untouched."""
    keys = set(_keys_for(ep, order).values())
    new = [dataclasses.replace(ds, view_search=tuple(v for v in order if v != ds.view))
           if ds.key in keys else ds for ds in ep.kit.dreams]
    ep.kit = dataclasses.replace(ep.kit, dreams=new)
    # Episode.dream_spec reads a key -> spec table built from the kit at __init__;
    # keep it in step, or dream_spec('iso') (turns, refine, re-dreams) would still
    # describe the spec without the search views.
    specs = getattr(ep, "_dream_specs", None)
    if isinstance(specs, dict):
        specs.update({ds.key: ds for ds in new if ds.key in keys})
    ep.trace.event("turnaround_free", views=list(order))


def cyclic_assign(ep: Any, att: Any, tiles: Sequence[bytes], order: Sequence[str], *,
                  generator: str, prompt: str = "") -> Optional[Sequence[str]]:
    """Step 6: reassign the tiles to cameras once, by silhouette against ``att``.
    Returns the chosen assignment (None when skipped)."""
    if att is None or att.blocks <= 0:
        ep.trace.event("turnaround_assign", skipped="empty build")
        return None
    try:
        renders = ep.render(att, list(order))
    except EmptyBuild:
        renders = {}
    if not renders:
        ep.trace.event("turnaround_assign", skipped="no renders")
        return None
    chosen, value = sheets.best_assignment(tiles, renders, order)
    identity = sheets.assignment_score(tiles, renders, order)
    changed = tuple(chosen) != tuple(order)
    ep.trace.event("turnaround_assign", order=list(order), chosen=list(chosen),
                   value=value, identity=identity, changed=changed)
    if changed:
        fill_from_tiles(ep, tiles, chosen, generator=generator + ":assigned", prompt=prompt)
        ep.rescore_all()
        note = getattr(ep, "note_target_changed", None)
        if note is not None:
            note("the same building, with the views reassigned to the right cameras")
    return chosen


@register("method", "turnaround", prefetch_dreams=False)
def turnaround(ep: EpisodeLike, *, mode: str = "sheet",
               order: Sequence[str] = ("iso", "iso_left", "iso_back", "iso_right"),
               assign: str = "cyclic", sheet_ref: str = "none", refine: int = 0,
               refine_min_score: float = 7, fallback_per_view: bool = True,
               max_iters: int = 3, stop_at: float = 0.85, patience: int = 2,
               build_mode: str = "rewrite", feedback: str = "raw", repair_rounds: int = 1,
               plan: bool = False, n_examples: int = 0,
               describe: str = "none", caption_format: str = "json",
               expand_into_image: bool = True, show_spec: bool = True,
               caption_keys: str = "primary", caption_in_match: bool = True,
               caption_control: str = "none") -> None:
    """One turnaround sheet -> four view targets -> build -> match loop."""
    order = tuple(str(v) for v in order)
    _validate(ep, mode=mode, order=order, assign=assign, sheet_ref=sheet_ref,
              build_mode=build_mode, feedback=feedback)
    dcfg = describe_mod.config(describe, caption_format, expand_into_image, show_spec,
                               caption_keys, caption_in_match, caption_control)
    # The spec (describe: before/both) reaches the sheet and the per-view dreams.
    describe_mod.setup(ep, dcfg, extra_keys=(SHEET_KEY,))

    tiles: Optional[List[bytes]] = None
    generator = prompt = ""
    if mode == "per_view":
        make_dreams(ep, refine=refine, refine_min_score=refine_min_score)
    else:
        # 1-2. SHEET (+ critique)
        sheet = make_sheet(ep, order, sheet_ref=sheet_ref, refine=refine,
                           refine_min_score=refine_min_score)
        # 3. SPLIT
        cut = sheets.split_grid(sheet.png, rows=2, cols=2)
        checks = [sheets.tile_check(t) for t in cut]
        ok = all(c[0] for c in checks)
        generator, prompt = f"split:sheet_v{sheet.version}", sheet.prompt
        if not ok:
            ep.trace.event("turnaround_fallback", reasons=[c[1] for c in checks],
                           fallback=bool(fallback_per_view))
        if ok or not fallback_per_view or not per_view_fallback(
                ep, refine=refine, refine_min_score=refine_min_score):
            # 4. FILL (a rejected split is used anyway when no fallback runs)
            tiles = cut
            fill_from_tiles(ep, tiles, order, generator=generator, prompt=prompt)
            if assign == "free":
                free_views(ep, order)

    # 5. BUILD (a describe stage captions the iso target first)
    describe_mod.refresh(ep)
    if plan:
        steps.plan(ep)
    att = steps.generate(ep, n_examples=n_examples)
    att = steps.repair(ep, att, rounds=repair_rounds) or att

    # 6. ASSIGN (once: a later reassignment would move the target under the loop)
    if tiles is not None and assign == "cyclic":
        cyclic_assign(ep, ep.current() or att, tiles, order, generator=generator,
                      prompt=prompt)

    # 7. MATCH
    match_loop(ep, att, primary=dreams_mod.primary(ep.kit.dreams), max_iters=max_iters,
               stop_at=stop_at, patience=patience, mode=build_mode, feedback=feedback,
               redream=0, redream_patience=1, redream_dependants=False)


__all__ = ["ASSIGNS", "MODES", "SHEET_KEY", "SHEET_REFS", "cyclic_assign", "fill_from_tiles",
           "free_views", "make_sheet", "per_view_fallback", "review_sheet", "sheets_hint",
           "turnaround"]
