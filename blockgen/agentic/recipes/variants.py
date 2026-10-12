"""Methods built from other methods: ``best_of_n`` and ``dream_vote``.

Both are compute controls as much as methods. ``best_of_n`` is the "just sample
more" baseline any loop must beat at matched calls; ``dream_vote`` tests whether
picking among dream samples beats refining one (image models vary a lot per seed,
and picking costs one critic call instead of K regenerations).
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

from blockgen.agentic import dreams as dreams_mod
from blockgen.agentic import turns
from blockgen.agentic.core import EpisodeLike, lookup, register, validate_params
from blockgen.agentic.recipes.dream import dream


@register("method", "best_of_n")
def best_of_n(ep: EpisodeLike, *, n: int = 4, inner: str = "classic",
              inner_params: Optional[Dict[str, Any]] = None) -> None:
    """Run ``inner`` n times from scratch (distinct design variations) and keep the
    best-scoring build across all of them.

    Dreams are shared: branches after the first pass the dreams branch 0 made as
    ``skip_dreams`` (when the inner method takes it, e.g. ``dream``). Re-running
    the dream phase per branch would re-pay every refine review and regeneration,
    and would replace the target under the earlier branches, so their scores
    would no longer compare with the later ones. What best_of_n samples is the
    build loop, with the target held fixed.
    """
    fn = lookup("method", inner).load()
    params = dict(inner_params or {})
    validate_params(fn, params, where=f"best_of_n inner method '{inner}'")
    if not (ep.kit.scorers or ep.kit.final_scorers):
        # Without a score every candidate is 0 and "best" is just the last branch.
        raise ValueError(f"recipe '{ep.spec.name}': best_of_n needs 'score' or 'final_score' "
                         f"terms to pick a best candidate")
    shares_dreams = ("skip_dreams" in inspect.signature(fn).parameters
                     and "skip_dreams" not in params)
    base = getattr(ep, "variation", None)
    ep.select = "best"            # selection spans branches, whatever the recipe says
    for i in range(max(1, int(n))):
        kw = dict(params)
        if i > 0:
            ep.new_branch()
            # A distinct rider per candidate: a different request (else the response
            # cache returns one build n times) that asks for a different design.
            ep.variation = (base or 0) + i
            if shares_dreams:
                kw["skip_dreams"] = tuple(k for k, v in ep.dreams.items() if v)
        ep.trace.event("branch", index=i, variation=ep.variation)
        fn(ep, **kw)
    ep.variation = base


@register("method", "dream_vote")
def dream_vote(ep: EpisodeLike, *, n_dreams: int = 3, **loop: Any) -> None:
    """Sample several primary dreams, let the dream critic pick the most buildable,
    then run the normal dream loop on it (``**loop`` = dream's params)."""
    validate_params(dream, loop, where="dream_vote (dream params)")
    if "skip_dreams" in loop:
        raise TypeError("dream_vote: skip_dreams is set by dream_vote itself")
    primary = dreams_mod.primary(ep.kit.dreams)
    if primary is None:
        raise ValueError(f"recipe '{ep.spec.name}': dream_vote needs at least one dream")
    key = primary.key
    n = max(1, int(n_dreams))
    cands = [ep.dream(key, seed=ep.seed + i, store=False) for i in range(n)]
    pick = ep.ask(turns.pick_buildable(ep, [c.png for c in cands]), role="dream_critic",
                  history=False, json_reply=True, stage="dream_vote")
    try:
        idx = int(pick.get("index", 0)) % n
    except (TypeError, ValueError):
        idx = 0
    chosen = cands[idx]
    chosen.version = 0
    chosen.critique = {"vote": dict(pick), "index": idx, "n": n}
    ep.dreams[key] = [chosen]
    ep.trace.dream(chosen)
    ep.trace.event("dream_vote", key=key, index=idx, n=n)
    dream(ep, skip_dreams=(key,), **loop)


__all__ = ["best_of_n", "dream_vote"]
