"""Method ``dream``: imagine the build, build it, compare renders with the image.

    1. DREAM      every recipe dream, parents before the dreams conditioned on them
    2. REFINE     dream critic vets each one; regenerate up to ``refine`` times
    3. BUILD      plan? -> generate (TARGET images shown) -> symbolic repair
    4. MATCH      compare TARGET vs YOURS, revise, accept-or-roll-back, until
                  ``stop_at`` / ``patience`` / ``max_iters``; REDREAM on request or
                  on a stall the dream critic calls unreachable
    5. FINALIZE   (run_episode) re-anchor scores, pick best per select/final_score

Every option is a parameter, so recipes and sweeps reach all of them:

    refine, refine_min_score    pre-build dream critique (K regenerations)
    max_iters, stop_at,         match-loop bounds and stop rules (score threshold,
    patience                    non-improving iterations)
    mode                        'rewrite' (full program) | 'patch' (extra commands)
    feedback                    'raw' image pairs | 'score' table | 'vdm' judge text | 'both'
    redream, redream_patience   mid-loop re-imagination budget and its stall trigger
    redream_dependants          also regenerate dreams conditioned on the re-dreamed one
    repair_rounds, plan,        build-phase knobs (as classic)
    n_examples
    skip_dreams                 dream keys already provided (dream_vote)
    describe, caption_format,   the describe stage (describe.py): a detailed spec
    expand_into_image,          written before the dream (``before``), a caption of
    show_spec, caption_keys,    what the dream shows (``after``), or both; and the
    caption_in_match,           shuffled-caption control. ``none`` (the default)
    caption_control             makes no call and changes no request

Budget: every ask/dream checks the ledger first; ``BudgetExceeded`` unwinds to
run_episode, which keeps the best attempt so far (``truncated='budget'``). Two
exceptions, both about dream images: refine stops regenerating when the next
image does not fit, and a re-dream whose images (the dream plus its dependants)
do not all fit is skipped up front. The build goes on with the current dreams.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from blockgen.agentic import describe as describe_mod
from blockgen.agentic import dreams as dreams_mod
from blockgen.agentic import turns
from blockgen.agentic.core import Attempt, BudgetExceeded, DreamSpec, EpisodeLike, register
from blockgen.agentic.recipes import steps


def make_dreams(ep: Any, *, refine: int = 0, refine_min_score: float = 7,
                skip: Sequence[str] = ()) -> None:
    """Steps 1-2: generate (and optionally refine) every dream in topo order, so a
    dependant is generated against its parent's FINAL version."""
    for ds in dreams_mod.topo(ep.kit.dreams):
        if ds.key in skip:
            continue
        ep.dream(ds.key)
        if refine > 0:
            steps.refine_dream(ep, ds.key, rounds=refine, min_score=refine_min_score)


def _redream_fits(ep: Any, key: str, dependants: bool, i: int) -> bool:
    """Whether a re-dream's images fit the budget. If not, the loop goes on with
    the current dreams (event ``redream_budget_stop``): ``max_images`` bounds dream
    regeneration, it should not end the build, exactly as in refine."""
    n = 1 + (len(steps.redream_dependants(ep, key)) if dependants else 0)
    try:
        steps.ensure_image_budget(ep, n)
    except BudgetExceeded as exc:
        ep.trace.event("redream_budget_stop", iter=i, key=key, error=str(exc))
        return False
    return True


def match_loop(ep: Any, att: Attempt, *, primary: Optional[DreamSpec], max_iters: int,
               stop_at: float, patience: int, mode: str, feedback: str, redream: int,
               redream_patience: int, redream_dependants: bool) -> None:
    """Step 4 of :func:`dream`: compare TARGET vs YOURS, revise, accept or roll back.

    Stops at ``stop_at`` / ``patience`` / ``max_iters``. ``primary`` is the dream a
    REDREAM acts on (None: no re-dreams). Pulled out of :func:`dream` so other
    methods that make their dreams another way (``turnaround`` splits one sheet)
    run exactly the same loop; :func:`dream` validates the knobs, this only
    clamps them, so a direct caller cannot pass a zero patience.
    """
    patience = max(1, int(patience))
    redream = max(0, int(redream))
    redream_patience = max(1, int(redream_patience))
    stall = redreams = 0
    checked = 0          # stall length at the last 'reachable' verdict (ask once per window)
    for i in range(max(0, int(max_iters))):
        cur = ep.current() or att
        if cur.score is not None and cur.score.value >= stop_at:
            ep.trace.event("stop", rule="stop_at", iter=i, value=cur.score.value)
            break
        can_redream = primary is not None and redreams < redream
        reply = str(steps.match_step(ep, i, cur, feedback=feedback, mode=mode,
                                     allow_redream=can_redream))
        desc = turns.parse_redream(reply)
        if desc is not None:
            if can_redream and _redream_fits(ep, primary.key, redream_dependants, i):
                steps.redream(ep, primary.key, desc or None, dependants=redream_dependants,
                              reason="builder")
                redreams += 1
                stall = checked = 0
                continue
            if not can_redream:
                ep.trace.event("redream_denied", iter=i)
            else:
                redreams = redream          # the image budget is gone: no more re-dreams
            # The reply is not a program, so do not execute it; count it as a
            # non-improving iteration.
            stall += 1
        else:
            cand = ep.try_(f"match{i}", reply, patch=(mode == "patch"))
            improved = (cand.accepted and cand.score is not None and cur.score is not None
                        and cand.score.value > cur.score.value)
            stall = 0 if improved else stall + 1
            if improved:
                checked = 0
        # Mid-loop REDREAM, option (a): a stalled loop asks the dream critic whether
        # the target is reachable at all, and re-imagines it (anchored on the best
        # build) if not.
        # The budget is checked first: a verdict nobody can act on is not worth a call.
        # After a 'reachable' verdict the critic is asked again only after another
        # ``redream_patience`` stalled iterations, not on every one.
        if stall - checked >= redream_patience and primary is not None and redreams < redream:
            if not _redream_fits(ep, primary.key, redream_dependants, i):
                redreams = redream
                v = {}
            else:
                anchor = ep.best() or ep.current() or cur
                v = steps.reachability(ep, anchor, primary.key)
            if v.get("unreachable"):
                steps.redream(ep, primary.key, v.get("revised_prompt") or None, anchor=anchor,
                              dependants=redream_dependants, reason=v.get("reason", "stall"))
                redreams += 1
                stall = checked = 0
                continue
            checked = stall
        if stall >= patience:
            ep.trace.event("stop", rule="patience", iter=i, stall=stall)
            break


@register("method", "dream")
def dream(ep: EpisodeLike, *, refine: int = 0, refine_min_score: float = 7, max_iters: int = 3,
          stop_at: float = 0.85, patience: int = 2, mode: str = "rewrite",
          feedback: str = "raw", redream: int = 0, redream_patience: int = 2,
          redream_dependants: bool = True, repair_rounds: int = 1, plan: bool = False,
          n_examples: int = 0, skip_dreams: Sequence[str] = (),
          describe: str = "none", caption_format: str = "json", expand_into_image: bool = True,
          show_spec: bool = True, caption_keys: str = "primary", caption_in_match: bool = True,
          caption_control: str = "none") -> None:
    """Imagine the target image(s), build, and iterate on render-vs-dream feedback."""
    dcfg = describe_mod.config(describe, caption_format, expand_into_image, show_spec,
                               caption_keys, caption_in_match, caption_control)
    if mode not in turns.MODE_INSTRUCTIONS:
        raise ValueError(f"mode must be one of {sorted(turns.MODE_INSTRUCTIONS)}, got {mode!r}")
    if feedback not in ("raw", "score", "vdm", "both"):
        raise ValueError(f"feedback must be raw|score|vdm|both, got {feedback!r}")
    if mode == "patch" and not getattr(ep.kit.actions, "patchable", False):
        # try_ would silently run each reply as a full program while the turn asks
        # for additions only: the model and the executor would disagree.
        raise ValueError(f"mode 'patch' needs a patchable action space; "
                         f"'{getattr(ep.kit.actions, 'name', '?')}' is not")
    patience = max(1, int(patience))
    redream = max(0, int(redream))
    redream_patience = max(1, int(redream_patience))
    if redream and redream_patience > patience:
        # The patience stop would always fire first: the stall re-dream could never run.
        raise ValueError(f"redream_patience ({redream_patience}) must be <= patience "
                         f"({patience}), else a stalled loop stops before it can re-dream")

    # 0. DESCRIBE before (a spec the dreams are drawn from), if asked
    describe_mod.setup(ep, dcfg)

    # 1-2. DREAM (+ critique / regeneration before building), then DESCRIBE after
    make_dreams(ep, refine=refine, refine_min_score=refine_min_score, skip=tuple(skip_dreams))
    primary = dreams_mod.primary(ep.kit.dreams)
    describe_mod.refresh(ep)

    # 3. BUILD
    if plan:
        steps.plan(ep)
    att = steps.generate(ep, n_examples=n_examples)
    att = steps.repair(ep, att, rounds=repair_rounds) or att

    # 4. MATCH
    match_loop(ep, att, primary=primary, max_iters=max_iters, stop_at=stop_at,
               patience=patience, mode=mode, feedback=feedback, redream=redream,
               redream_patience=redream_patience, redream_dependants=redream_dependants)


__all__ = ["dream", "make_dreams", "match_loop"]
