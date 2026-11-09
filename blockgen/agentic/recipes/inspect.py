"""``inspect``: the classic loop with inspect-and-repair rounds in place of (or
before) render critiques.

ldraw-nova's agent cycle is brief, retrieve, build, inspect, repair, deliver; the
inspect step is a set of geometry checks whose report the agent reads before it
repairs. ``classic`` already has brief (plan), build, repair (execution errors)
and a visual critique; this method adds the symbolic inspection
(``inspection.inspect_structure``) as its own repair signal::

    plan? -> generate -> repair (executor errors) -> inspect/repair x N -> critique x M

Controls: ``classic-matched`` (the same number of turns, all visual) isolates
symbolic against visual feedback; ``inspect_rounds: 0`` is ``classic``.
"""

from __future__ import annotations

from typing import Sequence

from blockgen.agentic.core import EpisodeLike, get_view, register
from blockgen.agentic.recipes import steps


@register("method", "inspect")
def inspect(ep: EpisodeLike, *, plan: bool = False, n_examples: int = 0, repair_rounds: int = 1,
            inspect_rounds: int = 3, inspect_mode: str = "rewrite", clean_at: str = "warning",
            show_passed: bool = True, critique_rounds: int = 0, critique_mode: str = "rewrite",
            critique_views: Sequence[str] = ("legacy_a", "legacy_b")) -> None:
    """Plan / generate / repair, then inspect-and-repair rounds, then render critiques."""
    for name, mode in (("inspect_mode", inspect_mode), ("critique_mode", critique_mode)):
        if mode not in ("rewrite", "patch"):
            raise ValueError(f"{name} must be rewrite|patch, got {mode!r}")
        if mode == "patch" and not getattr(ep.kit.actions, "patchable", False):
            raise ValueError(f"{name} 'patch' needs a patchable action space; "
                             f"'{getattr(ep.kit.actions, 'name', '?')}' is not")
    if clean_at not in ("error", "warning", "info"):
        raise ValueError(f"clean_at must be error|warning|info, got {clean_at!r}")
    views = [get_view(v).name for v in critique_views]
    if plan:
        steps.plan(ep)
    has_examples_ctx = any(n == "examples" for n, _ in ep.kit.contexts)
    att = steps.generate(ep, n_examples=0 if has_examples_ctx else n_examples)
    att = steps.repair(ep, att, rounds=repair_rounds) or att
    steps.inspect_repair(ep, att, rounds=inspect_rounds, mode=inspect_mode, clean_at=clean_at,
                         show_passed=show_passed)
    for i in range(critique_rounds):
        att = steps.critique_turn(ep, i, views=views, mode=critique_mode)
        if att is None or (not att.accepted and att.blocks == 0):
            break


__all__ = ["inspect"]
