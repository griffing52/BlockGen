"""Multi-block objects (beds/doors) must tokenize as ONE piece (user request).

A bed is two horizontally-adjacent id-26 voxels; a door is two vertically-stacked
id-64 voxels. With force_families (default), each becomes a single piece token instead
of two atomic tokens.

Run:  python -m pytest tests/test_forced_family_merge.py -q
"""

from __future__ import annotations

import numpy as np

from blockgen.tokenizers.cluster_bpe import (learn_clusters, structure_to_cluster_tokens,
                                             cluster_tokens_to_structure)
from blockgen.utils.data import Structure


def _with_door_and_bed():
    """A stone box with a 2-high oak door and a 2-wide bed, plus filler."""
    ids = np.zeros((6, 4, 6), np.int32)
    data = np.zeros((6, 4, 6), np.int32)
    # stone walls so there are non-family blocks too
    ids[:, 0, :] = 1                      # stone floor
    ids[0, 1:3, 0] = 1                    # a bit of wall
    # oak door: two vertical voxels, id 64, halves differ by data (lower/upper)
    ids[2, 1, 0] = 64; data[2, 1, 0] = 0
    ids[2, 2, 0] = 64; data[2, 2, 0] = 8
    # bed: two horizontal voxels, id 26, foot/head differ by data
    ids[4, 1, 4] = 26; data[4, 1, 4] = 9
    ids[5, 1, 4] = 26; data[5, 1, 4] = 10
    return Structure(block_ids=ids, block_data=data)


def _n_piece_tokens(tokens, cv):
    return sum(1 for t in tokens if cv.is_piece(t))


def _pieces_covering(cv, tokens, legacy_id):
    """Pattern sizes of emitted pieces whose cells include the given legacy id."""
    id_of = [p[0] for p in cv.block_index_to_pair]
    out = []
    for t in tokens:
        if cv.is_piece(t):
            pat = cv.patterns[cv.decode_piece(t)]
            if any(id_of[c[3]] == legacy_id for c in pat):
                out.append(len(pat))
    return out


def test_door_and_bed_are_single_pieces():
    s = _with_door_and_bed()
    cv = learn_clusters([s], max_dim=8, n_merges=0, min_count=1, force_families=True)
    tokens = structure_to_cluster_tokens(s, cv)

    # The door's two id-64 voxels must be covered by ONE 2-voxel piece, not two.
    door_pieces = _pieces_covering(cv, tokens, 64)
    assert door_pieces == [2], f"door not a single 2-voxel piece: {door_pieces}"
    bed_pieces = _pieces_covering(cv, tokens, 26)
    assert bed_pieces == [2], f"bed not a single 2-voxel piece: {bed_pieces}"


def test_force_off_leaves_them_split():
    s = _with_door_and_bed()
    cv = learn_clusters([s], max_dim=8, n_merges=0, min_count=1, force_families=False)
    tokens = structure_to_cluster_tokens(s, cv)
    # Without forcing and no frequency merges, halves stay as separate 1-voxel pieces.
    assert _pieces_covering(cv, tokens, 64) == [1, 1]
    assert _pieces_covering(cv, tokens, 26) == [1, 1]


def test_forced_merge_roundtrips_to_same_structure():
    s = _with_door_and_bed()
    cv = learn_clusters([s], max_dim=8, n_merges=0, min_count=1, force_families=True)
    rebuilt = cluster_tokens_to_structure(structure_to_cluster_tokens(s, cv), cv)
    # Decoding the forced-merged tokens must reproduce the occupied voxels.
    a = s.crop_to_non_air(); b = rebuilt.crop_to_non_air()
    assert a.shape == b.shape
    assert np.array_equal(a.block_ids != 0, b.block_ids != 0)
