"""WP5 accept policies: truth tables (spec §3.2). Policies see only (prev, cand)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

from blockgen.agentic import core, scoring
from blockgen.agentic.calibrate import attempt_for
from blockgen.agentic.core import Attempt, Score
from blockgen.utils.data import Structure


def att(blocks: int, score: Optional[float] = None, idx: int = 1, validity: Optional[float] = None,
        structure: Optional[Structure] = None) -> Attempt:
    if structure is None:
        ids = np.zeros((10, 10, 10), np.int32)
        ids.reshape(-1)[:blocks] = 1
        structure = Structure(ids, np.zeros_like(ids))
    a = attempt_for(structure, idx=idx)
    a.blocks = blocks
    if score is not None:
        parts = {} if validity is None else {"validity": validity}
        a.score = Score(score, parts=parts)
    return a


def accept(spec, prev, cand):
    return core.make("accept", spec)(prev, cand)


# --- always / nonempty_legacy ------------------------------------------------------------
@pytest.mark.parametrize("prev,cand", [(None, 0), (None, 5), (0, 0), (5, 0), (5, 3)])
def test_always(prev, cand):
    p = None if prev is None else att(prev)
    assert accept("always", p, att(cand))[0] is True


@pytest.mark.parametrize("prev,cand,ok", [
    (None, 0, True),        # first round empty: legacy keeps it (canvas stays empty)
    (None, 5, True),
    (0, 0, True),
    (0, 5, True),
    (5, 0, False),          # the only rollback BuildAgent does
    (50, 1, True),          # "emptier" is NOT rejected: the legacy code checks == 0
    (5, 9, True),
])
def test_nonempty_legacy(prev, cand, ok):
    p = None if prev is None else att(prev)
    got, why = accept("nonempty_legacy", p, att(cand))
    assert got is ok and why


@pytest.mark.parametrize("prev,cand,min_frac,ok", [
    (None, 0, 0.5, False), (None, 3, 0.5, True), (0, 3, 0.5, True),
    (10, 5, 0.5, True), (10, 4, 0.5, False), (10, 4, 0.4, True), (10, 0, 0.0, False),
])
def test_nonempty(prev, cand, min_frac, ok):
    p = None if prev is None else att(prev)
    assert accept({"type": "nonempty", "min_frac": min_frac}, p, att(cand))[0] is ok


# --- improve ---------------------------------------------------------------------------
@pytest.mark.parametrize("prev,cand,eps,ok", [
    (None, (5, 0.1), 0.0, True),            # first non-empty
    ((0, 0.0), (5, 0.1), 0.0, True),         # previous was empty
    (None, (0, 0.0), 0.0, False),            # empty is never accepted
    ((5, 0.5), (0, 0.0), 0.0, False),
    ((5, 0.5), (6, 0.6), 0.0, True),
    ((5, 0.5), (6, 0.4), 0.0, False),        # worse
    ((5, 0.5), (6, 0.5), 0.0, False),        # tie is not an improvement
    ((5, 0.5), (6, 0.55), 0.1, False),       # inside eps
    ((5, 0.5), (6, 0.65), 0.1, True),
    ((5, None), (6, None), 0.0, True),       # unscored: non-empty rule
])
def test_improve(prev, cand, eps, ok):
    p = None if prev is None else att(*prev)
    got, why = accept(f"improve?eps={eps}", p, att(*cand))
    assert got is ok, why


# --- improve_valid -------------------------------------------------------------------
@pytest.mark.parametrize("pv,cv,drop,ok", [
    (1.0, 1.0, 0.1, True), (1.0, 0.95, 0.1, True), (1.0, 0.85, 0.1, False),
    (0.6, 0.55, 0.1, True), (1.0, 0.5, 0.6, True),
])
def test_improve_valid_validity_drop(pv, cv, drop, ok):
    got, why = accept({"type": "improve_valid", "max_validity_drop": drop},
                      att(5, 0.5, validity=pv), att(6, 0.7, validity=cv))
    assert got is ok, why
    if not ok:
        assert "validity" in why


def test_improve_valid_still_requires_improvement():
    assert accept("improve_valid", att(5, 0.7, validity=1.0), att(6, 0.6, validity=1.0))[0] is False
    assert accept("improve_valid", None, att(6, 0.1, validity=0.2))[0] is True


def test_improve_valid_computes_validity_when_not_scored():
    ids = np.zeros((8, 8, 8), np.int32)
    ids[0:3, 0:3, 0:3] = 1
    solid = Structure(ids, np.zeros_like(ids))
    ids2 = ids.copy()
    ids2[0:3, 0:3, 0:3] = 0
    ids2[0, 0, 0] = ids2[7, 7, 7] = ids2[4, 0, 0] = 1             # 3 disconnected singletons
    debris = Structure(ids2, np.zeros_like(ids2))
    prev = att(27, 0.5, structure=solid)
    cand = att(3, 0.9, structure=debris)
    got, why = accept("improve_valid", prev, cand)
    assert got is False and "validity dropped 1.00 -> 0.33" in why


# --- anneal -----------------------------------------------------------------------------
def test_anneal_improvement_always_accepted_and_empty_rejected():
    assert accept("anneal", att(5, 0.5), att(6, 0.5))[0] is True
    assert accept("anneal", att(5, 0.5), att(6, 0.9))[0] is True
    assert accept("anneal", att(5, 0.5), att(0, 0.0))[0] is False
    assert accept("anneal", None, att(3, 0.0))[0] is True


def test_anneal_large_drop_rejected_small_drop_hot_accepted():
    assert accept("anneal?t0=0.01", att(5, 0.9), att(6, 0.1, idx=3))[0] is False
    assert accept("anneal?t0=1000", att(5, 0.9), att(6, 0.89, idx=0))[0] is True


def test_anneal_deterministic_per_idx_and_seed():
    def run(seed):
        return [accept(f"anneal?t0=0.05&seed={seed}", att(5, 0.5), att(6, 0.47, idx=i))[0]
                for i in range(30)]
    a, b = run(0), run(0)
    assert a == b
    assert any(a) and not all(a)                  # it is actually stochastic
    assert run(1) != a


def test_anneal_cools_with_idx():
    """At a fixed drop, the acceptance probability printed in the reason falls with idx."""
    def p(idx):
        why = accept("anneal?t0=0.05", att(5, 0.5), att(6, 0.45, idx=idx))[1]
        return float(why.split("p=")[1].rstrip(")"))
    assert p(0) > p(3) > p(8)


def test_unknown_accept_param_raises():
    with pytest.raises(TypeError):
        core.make("accept", "improve?epsilon=0.1")


def test_policies_never_mutate_attempts():
    prev, cand = att(5, 0.5), att(6, 0.6)
    before = (cand.accepted, cand.reason, cand.score.value)
    for name in ("always", "nonempty_legacy", "nonempty", "improve", "improve_valid", "anneal"):
        core.make("accept", name)(prev, cand)
    assert (cand.accepted, cand.reason, cand.score.value) == before
    assert scoring.improve is core.lookup("accept", "improve").load()
