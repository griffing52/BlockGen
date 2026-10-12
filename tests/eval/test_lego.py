"""The StableText2Brick loader: axis convention, brick fidelity, and the split
metadata the benchmark reads.

The tests that matter here are the ones that would catch a *silent* encoding
error, because a transposed axis or a dropped brick does not raise -- it
produces a corpus that renders plausibly and scores meaninglessly. Two invariants
do most of that work: a non-overlapping brick list must voxelize to exactly
`sum(h*w)` blocks, and a 1-unit-tall brick must occupy exactly one horizontal
layer of the `Structure` (y is vertical there, z is vertical in the corpus).

The parquet-backed tests skip when the corpus is absent, so the suite still runs
on a checkout without `data/lego/`.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from blockgen.eval.bench import lego, probes
from blockgen.utils.data import Structure


def _has_corpus() -> bool:
    return any(lego.DEFAULT_ROOT.glob("test-*.parquet"))


needs_corpus = pytest.mark.skipif(not _has_corpus(),
                                  reason="data/lego/stabletext2brick not present")


# --- parsing ---------------------------------------------------------------
def test_parse_bricks_reads_dims_and_position():
    bricks = lego.parse_bricks("1x1 (0,0,0)\n2x6 (3,4,5)\n")
    assert bricks == [(1, 1, 0, 0, 0), (2, 6, 3, 4, 5)]


def test_parse_bricks_ignores_blank_lines():
    assert len(lego.parse_bricks("\n1x2 (0,0,0)\n\n  \n2x2 (0,2,0)\n")) == 2


@pytest.mark.parametrize("bad", ["1x1 (0,0)", "1x (0,0,0)", "red 1x1 (0,0,0)",
                                 "1x1 (0,0,0) # comment", "1x1 0,0,0"])
def test_parse_bricks_refuses_malformed_lines(bad):
    """A format change must fail loudly, not silently shrink the structure."""
    with pytest.raises(ValueError):
        lego.parse_bricks(bad)


# --- voxelization ----------------------------------------------------------
def test_brick_occupies_one_layer_along_the_vertical_axis():
    """Bricks are 1 unit tall in the corpus's z; `Structure`'s vertical axis is y."""
    s = lego.bricks_to_structure([(2, 6, 1, 2, 3)])
    occ = s.occupied_mask
    assert occ.sum() == 12
    ys = np.unique(np.argwhere(occ)[:, 1])
    assert ys.tolist() == [3]                       # lego z -> structure y
    xs = np.unique(np.argwhere(occ)[:, 0])
    zs = np.unique(np.argwhere(occ)[:, 2])
    assert xs.tolist() == [1, 2]                    # h spans x
    assert zs.tolist() == [2, 3, 4, 5, 6, 7]        # w spans the other horizontal


def test_block_count_equals_total_brick_footprint():
    bricks = [(1, 1, 0, 0, 0), (2, 4, 2, 0, 0), (1, 8, 5, 5, 3)]
    s = lego.bricks_to_structure(bricks)
    assert int(s.occupied_mask.sum()) == sum(h * w for h, w, *_ in bricks)


def test_overlapping_bricks_raise():
    """The cheapest available proof that h/w/z land on the right axes."""
    with pytest.raises(ValueError, match="overlaps"):
        lego.bricks_to_structure([(2, 2, 0, 0, 0), (2, 2, 1, 1, 0)])


def test_brick_leaving_the_grid_raises():
    with pytest.raises(ValueError, match="leaves"):
        lego.bricks_to_structure([(2, 6, 19, 0, 0)])


def test_footprint_colours_are_orientation_free_and_distinct():
    """1x2 and 2x1 are one brick type; the eight types must not collide."""
    assert lego.brick_wool(1, 2) == lego.brick_wool(2, 1)
    assert len(set(lego.BRICK_WOOL.values())) == len(lego.BRICK_WOOL) == 8
    assert all(0 <= d < 16 for d in lego.BRICK_WOOL.values())


def test_blocks_are_wool_so_the_textured_renderer_has_a_texture():
    s = lego.bricks_to_structure([(1, 1, 0, 0, 0), (2, 6, 2, 2, 1)])
    occ = s.occupied_mask
    assert set(np.unique(s.block_ids[occ]).tolist()) == {lego.WOOL_ID}
    assert set(np.unique(s.block_data[occ]).tolist()) == {
        lego.brick_wool(1, 1), lego.brick_wool(2, 6)}


# --- interiors, which is what the corpus is selected on --------------------
def test_enclosed_void_counts_a_sealed_pocket():
    """A hollow shell: 5^3 solid minus its centre cell."""
    ids = np.full((5, 5, 5), lego.WOOL_ID, dtype=np.int32)
    ids[2, 2, 2] = 0
    s = Structure(block_ids=ids, block_data=np.zeros_like(ids))
    assert lego.enclosed_void_blocks(s) == 1


def test_solidify_runs_unchanged_on_a_lego_structure():
    """`probes.solidify` is the whole point of this corpus; it must not need
    a LEGO-specific branch."""
    ids = np.full((5, 5, 5), lego.WOOL_ID, dtype=np.int32)
    ids[1:4, 1:4, 1:4] = 0
    s = Structure(block_ids=ids, block_data=np.full_like(ids, 14))
    out = probes.solidify(s)
    assert int(out.occupied_mask.sum()) == 125
    assert out.shape == s.shape
    # Filled cells take the build's own dominant material, so the probe is not
    # detectable as a palette anomaly.
    assert set(np.unique(out.block_ids[out.occupied_mask]).tolist()) == {lego.WOOL_ID}


# --- the real corpus -------------------------------------------------------
@needs_corpus
def test_real_rows_load_without_overlaps_or_parse_errors():
    for row in itertools.islice(lego.iter_parquet(split="test"), 200):
        s = lego.row_structure(row, "test")          # raises on either fault
        assert int(s.occupied_mask.sum()) > 0
        assert max(s.crop_to_non_air().shape) <= lego.GRID


@needs_corpus
def test_row_metadata_gives_the_split_a_usable_family_key():
    """Siblings of one ShapeNet object must group; `splits` strips trailing
    digits, so a bare hex object id would collide with its neighbours."""
    from blockgen.eval.bench import splits

    row = next(lego.iter_parquet(split="test"))
    s = lego.row_structure(row, "test")
    assert s.metadata["corpus"] == "stabletext2brick"
    assert s.metadata["category"] in set(lego.CATEGORY_NAMES.values())
    key = splits.group_key({"title": s.metadata["title"],
                            "category": s.metadata["category"],
                            "source_path": s.source_path})
    assert str(row["object_id"]).lower() in key


@needs_corpus
def test_every_category_id_in_the_corpus_has_a_name():
    ids = {str(r["category_id"]) for r in itertools.islice(
        lego.iter_parquet(split="test"), 2000)}
    assert ids and ids <= set(lego.CATEGORY_NAMES)
