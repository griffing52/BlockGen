"""Slate selection: the claim that column *i* means the same thing on every row.

`pick_slate` is small, and every one of these tests exists because the honest
failure and the dishonest one look identical on screen. A slate that silently
substitutes a build puts one model's cottage under another model's prompt label;
a slate that matches on the wrong index does the same thing more quietly. Both
render as a tidy grid.
"""

from __future__ import annotations

import numpy as np

from blockgen.eval.bench import examples as ex


def rng():
    return np.random.default_rng(0)


def test_without_a_slate_it_is_the_seeded_permutation():
    assert ex.pick_slate(10, 3, rng()) == ex.pick(10, 3, rng())


def test_without_prompts_a_slate_cannot_align_and_falls_back():
    """An unconditional arm under a conditional protocol: slots, not matches."""
    assert ex.pick_slate(10, 3, rng(), slate=("a hut",)) == ex.pick(10, 3, rng())


def test_rows_come_back_in_slate_order_not_row_order():
    """The columns line up across models only if slate order wins."""
    prompts = ["a tower", "a hut", "a barn"]
    assert ex.pick_slate(3, 8, rng(), prompts=prompts,
                         slate=("a barn", "a hut")) == [2, 1]


def test_matching_ignores_case_and_whitespace():
    prompts = ["A  Stone   Cottage"]
    assert ex.pick_slate(1, 8, rng(), prompts=prompts,
                         slate=("a stone cottage",)) == [0]


def test_a_slate_entry_with_no_build_leaves_a_hole_rather_than_substituting():
    prompts = ["a hut"]
    assert ex.pick_slate(1, 8, rng(), prompts=prompts,
                         slate=("a windmill", "a hut")) == [0]


def test_a_duplicate_prompt_takes_the_first_build_and_is_not_repeated():
    prompts = ["a hut", "a hut", "a barn"]
    assert ex.pick_slate(3, 8, rng(), prompts=prompts,
                         slate=("a hut", "a hut", "a barn")) == [0, 2]


def test_k_caps_the_slate():
    prompts = ["a", "b", "c", "d"]
    assert ex.pick_slate(4, 2, rng(), prompts=prompts, slate=("a", "b", "c")) == [0, 1]


def test_a_slate_that_matches_nothing_falls_back_rather_than_returning_empty():
    """An empty strip would read as "this model produced nothing", which is a
    different and false claim from "this protocol's prompts are not this arm's"."""
    prompts = ["a hut"]
    got = ex.pick_slate(1, 1, rng(), prompts=prompts, slate=("a windmill",))
    assert got == ex.pick(1, 1, rng())


def test_prompts_beyond_the_build_count_are_ignored():
    """`prompts` is parallel to the UNFILTERED list; empties are dropped before
    this is called, so a longer prompt list must not index past the builds."""
    assert ex.pick_slate(2, 8, rng(), prompts=["a", "b", "c", "d"],
                         slate=("c",)) == ex.pick(2, 8, rng())
