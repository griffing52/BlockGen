"""Palette statistics, with the family map as the thing under real suspicion.

The family rule table is hand-written priority-ordered regex, which is exactly
the shape of code that ships a silently wrong number. The tests below pin the
cases where an earlier rule could swallow a later one -- fences that are not
wood, walls that are not wood, bars that are not wood, foliage that is not
lumber -- plus the macro-averaging behaviour that keeps one cathedral from
defining the corpus palette.
"""

from __future__ import annotations

import numpy as np
import pytest

from blockgen.eval.bench import palette
from blockgen.utils.data import Structure

# Legacy (id, data) pairs, named for readability.
OAK_PLANK = (5, 0)
SPRUCE_PLANK = (5, 1)
DARK_OAK_PLANK = (5, 5)
OAK_FENCE = (85, 0)
NETHER_BRICK_FENCE = (113, 0)
IRON_BARS = (101, 0)
COBBLE = (4, 0)
COBBLE_WALL = (139, 0)
STONE_BRICK = (98, 0)
OAK_LEAVES = (18, 0)
WHITE_WOOL = (35, 0)
RED_WOOL = (35, 14)
GLASS_PANE = (102, 0)
OAK_STAIRS_N = (53, 0)
OAK_STAIRS_S = (53, 1)


def _make(pairs_with_counts) -> Structure:
    """A 1-D bar of blocks; only the multiset of blocks matters here."""
    flat = []
    for (bid, bd), n in pairs_with_counts:
        flat.extend([(bid, bd)] * n)
    ids = np.array([[[p[0]] for p in flat]], dtype=np.int32).reshape(len(flat), 1, 1)
    data = np.array([[[p[1]] for p in flat]], dtype=np.int32).reshape(len(flat), 1, 1)
    return Structure(block_ids=ids, block_data=data)


# --- family map ------------------------------------------------------------
@pytest.mark.parametrize("pair,expected", [
    (OAK_PLANK, "wood_oak"),
    (SPRUCE_PLANK, "wood_spruce"),
    (DARK_OAK_PLANK, "wood_dark_oak"),
    (OAK_STAIRS_N, "wood_oak"),
    (OAK_FENCE, "wood_oak"),
    (COBBLE, "cobblestone"),
    (COBBLE_WALL, "cobblestone"),
    (STONE_BRICK, "stone_brick"),
    (WHITE_WOOL, "wool"),
    (GLASS_PANE, "glass"),
    (OAK_LEAVES, "plant"),
])
def test_family_assignments(pair, expected):
    assert palette.block_family(*pair) == expected


def test_non_wooden_fences_are_not_wood():
    """The `fence`->wood rule must not capture nether brick or iron."""
    assert palette.block_family(*NETHER_BRICK_FENCE) == "nether_brick"
    assert palette.block_family(*IRON_BARS) == "metal_gem"


def test_dark_oak_beats_oak():
    """Species are matched longest-first, or 'dark oak' degrades to 'oak'."""
    assert palette.block_family(*DARK_OAK_PLANK) == "wood_dark_oak"


def test_leaves_are_not_lumber():
    assert palette.block_family(*OAK_LEAVES) != "wood_oak"


def test_family_collapses_wool_colour_but_exact_does_not():
    assert palette.block_family(*WHITE_WOOL) == palette.block_family(*RED_WOOL)
    assert (palette.block_key(*WHITE_WOOL, level="exact")
            != palette.block_key(*RED_WOOL, level="exact"))


def test_exact_collapses_orientation_but_pair_does_not():
    """Stair facing is orientation noise at the `exact` level, signal at `pair`."""
    assert (palette.block_key(*OAK_STAIRS_N, level="exact")
            == palette.block_key(*OAK_STAIRS_S, level="exact"))
    assert (palette.block_key(*OAK_STAIRS_N, level="pair")
            != palette.block_key(*OAK_STAIRS_S, level="pair"))


def test_unknown_level_raises():
    with pytest.raises(ValueError):
        palette.block_key(5, 0, level="nonsense")


# --- histograms ------------------------------------------------------------
def test_hist_excludes_air():
    s = _make([(OAK_PLANK, 4)])
    ids = np.zeros((3, 3, 3), dtype=np.int32)
    ids[0, 0, 0] = 5
    s2 = Structure(block_ids=ids, block_data=np.zeros_like(ids))
    assert sum(palette.palette_hist(s2).values()) == 1
    assert sum(palette.palette_hist(s).values()) == 4


def test_palette_size():
    s = _make([(OAK_PLANK, 3), (COBBLE, 2), (GLASS_PANE, 1)])
    assert palette.palette_size(s, "family") == 3
    assert palette.palette_size(_make([(OAK_PLANK, 3), (OAK_FENCE, 2)]), "family") == 1


def test_matrix_reports_oov_separately():
    """Blocks outside the reference vocabulary must be counted, not dropped."""
    structs = [_make([(OAK_PLANK, 3), (COBBLE, 1)])]
    counts, keys, oov = palette.palette_matrix(structs, "family", keys=["wood_oak"])
    assert counts.sum() == 3
    assert oov.tolist() == [1.0]


def test_oov_frac():
    structs = [_make([(OAK_PLANK, 3), (COBBLE, 1)])]
    assert palette.palette_oov_frac(structs, ["wood_oak"], "family") == pytest.approx(0.25)
    assert palette.palette_oov_frac(structs, ["wood_oak", "cobblestone"],
                                    "family") == pytest.approx(0.0)


# --- macro averaging -------------------------------------------------------
def test_macro_distribution_is_per_structure_normalized():
    """One huge build must not outvote many small ones (the pooling trap)."""
    counts = np.array([[1000.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    macro = palette.macro_distribution(counts)
    assert macro[0] == pytest.approx(0.25)
    assert macro[1] == pytest.approx(0.75)


def test_macro_handles_empty_structure_row():
    macro = palette.macro_distribution(np.array([[0.0, 0.0], [2.0, 2.0]]))
    assert np.isfinite(macro).all()


# --- divergences -----------------------------------------------------------
def test_jsd_zero_against_itself():
    ref = [_make([(OAK_PLANK, 5), (COBBLE, 3)]), _make([(GLASS_PANE, 2), (COBBLE, 6)])]
    assert palette.palette_jsd(ref, ref, "family") == pytest.approx(0.0, abs=1e-12)


def test_jsd_detects_a_wrong_palette():
    ref = [_make([(OAK_PLANK, 8)]) for _ in range(4)]
    gen = [_make([(COBBLE, 8)]) for _ in range(4)]
    assert palette.palette_jsd(gen, ref, "family") > 0.9


def test_fully_out_of_vocabulary_generation_scores_maximally_wrong():
    """Regression: scoring on the reference alphabet alone reported JSD 0.0 here.

    Every generated block was absent from the reference vocabulary, so the
    histogram was all zeros, renormalized to uniform, and compared to a
    reference that was also effectively uniform on its single symbol -- a
    perfect palette score for a build sharing no material with the corpus.
    """
    ref = [_make([(OAK_PLANK, 8)]) for _ in range(4)]
    gen = [_make([(GLASS_PANE, 8)]) for _ in range(4)]
    assert palette.palette_jsd(gen, ref, "family") == pytest.approx(1.0, abs=1e-6)
    assert palette.palette_oov_frac(gen, palette.vocabulary(ref, "family"),
                                    "family") == pytest.approx(1.0)


def test_family_jsd_is_gentler_than_exact_on_species_swap():
    """Family granularity should tolerate a variant swap that `exact` punishes."""
    ref = [_make([(OAK_STAIRS_N, 8)]) for _ in range(4)]
    gen = [_make([(OAK_STAIRS_S, 8)]) for _ in range(4)]
    assert palette.palette_jsd(gen, ref, "family") == pytest.approx(0.0, abs=1e-12)
    assert palette.palette_jsd(gen, ref, "pair") > 0.5


def test_cooccurrence_jsd_zero_against_itself():
    ref = [_make([(OAK_PLANK, 4), (COBBLE, 4)]),
           _make([(GLASS_PANE, 2), (COBBLE, 4)])]
    assert palette.cooccurrence_jsd(ref, ref, "family") == pytest.approx(0.0, abs=1e-12)


def test_cooccurrence_detects_broken_pairing():
    """Same marginals, different partnerships -- only co-occurrence sees this."""
    ref = [_make([(OAK_PLANK, 4), (COBBLE, 4)]) for _ in range(6)]
    gen = [_make([(OAK_PLANK, 8)]) for _ in range(3)] + \
          [_make([(COBBLE, 8)]) for _ in range(3)]
    assert palette.palette_jsd(gen, ref, "family") == pytest.approx(0.0, abs=1e-9)
    assert palette.cooccurrence_jsd(gen, ref, "family") > 0.1


# --- the correctness gate --------------------------------------------------
@pytest.mark.slow
def test_family_map_covers_the_real_corpus():
    """The gate from the plan: unclassified blocks must stay under 5%."""
    from blockgen.eval.bench import splits
    for which in ("train", "val", "test"):
        st = splits.split_structures(splits.load_split("houses_32"), which)
        assert palette.family_other_frac(st) < 0.05, which
