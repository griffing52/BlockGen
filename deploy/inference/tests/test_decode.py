"""The streaming decoders must agree with the batch decoders they replace.

``PieceDecoder``/``VoxelDecoder`` restate ``cluster_tokens_to_structure`` and
``tokens_to_structure`` incrementally. Two independent implementations of the same
grammar drift, and the failure is quiet: a resync rule that differs only on
malformed input still looks fine on clean input, right up until a model emits a bad
span mid-demo. So these tests feed both paths the same tokens -- including
deliberately malformed ones -- and compare the placed voxels.

Run:  python -m pytest deploy/inference/tests/ -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from blockgen.tokenizers.cluster_bpe import ClusterVocab, cluster_tokens_to_structure
from blockgen.utils.serialize import BlockVocab, tokens_to_structure
from blockgen_server.blockmap import modern_state
from blockgen_server.decode import PieceDecoder, VoxelDecoder, decode_all


def _cv() -> ClusterVocab:
    """Tiny piece vocab: two atomic blocks + one 2-voxel merged piece."""
    return ClusterVocab(
        max_dim=8,
        patterns=[((0, 0, 0, 0),), ((0, 0, 0, 1),), ((0, 0, 0, 0), (1, 0, 0, 1))],
        block_index_to_pair=[(5, 0), (35, 14)],   # oak planks, red wool
        merges=[(0, 1, (1, 0, 0))],
        num_blocks=2,
    )


def _bv() -> BlockVocab:
    return BlockVocab(max_dim=8,
                      block_token_to_id={"5": 0, "35:14": 1},
                      id_to_block_token=["5", "35:14"],
                      block_index_to_pair=[(5, 0), (35, 14)])


def _voxels_from_structure(s):
    """{(x,y,z): (id, data)} for every occupied voxel."""
    occ = np.argwhere(s.occupied_mask)
    return {(int(x), int(y), int(z)): (int(s.block_ids[x, y, z]), int(s.block_data[x, y, z]))
            for x, y, z in occ}


def _voxels_from_blocks(blocks):
    return {(b.x, b.y, b.z): b.state for b in blocks}


# Token layout for _cv(): PAD/BOS/EOS = 0/1/2, coords 3..10 (max_dim 8, 0-based),
# pieces 11..13 (piece_offset = 3 + max_dim).
@pytest.mark.parametrize("tokens", [
    # clean: two records, the second uses the merged 2-voxel piece (id 2 -> token 13)
    [1, 3, 3, 3, 11, 4, 3, 3, 13, 2],
    # malformed: a piece token where a coordinate belongs -> must resync, not crash
    [1, 11, 3, 3, 3, 11, 2],
    # malformed: a coordinate where a piece belongs
    [1, 3, 3, 3, 4, 3, 3, 3, 11, 2],
    # truncated trailing record (no EOS)
    [1, 3, 3, 3, 11, 4, 4],
])
def test_piece_decoder_matches_batch(tokens) -> None:
    cv = _cv()
    streamed = _voxels_from_blocks(decode_all(PieceDecoder(cv), tokens))

    batch = cluster_tokens_to_structure(tokens, cv)
    expected = {}
    for (x, y, z), (bid, data) in _voxels_from_structure(batch).items():
        expected[(x, y, z)] = modern_state(bid, data)

    # cluster_tokens_to_structure crops to the bounding box; the stream cannot (it
    # has no finished bbox), so compare shapes relative to each one's own origin.
    assert _normalize(streamed) == _normalize(expected), \
        f"stream and batch disagree for {tokens}"


def test_voxel_decoder_matches_batch() -> None:
    bv = _bv()
    tokens = [1, 3, 3, 3, 11, 4, 3, 3, 12, 2]
    streamed = _voxels_from_blocks(decode_all(VoxelDecoder(bv), tokens))
    batch = tokens_to_structure(tokens, bv)
    expected = {p: modern_state(*v) for p, v in _voxels_from_structure(batch).items()}
    assert _normalize(streamed) == _normalize(expected)


def _normalize(voxels: dict) -> dict:
    """Shift a voxel dict so its minimum corner sits at the origin."""
    if not voxels:
        return {}
    ox = min(p[0] for p in voxels)
    oy = min(p[1] for p in voxels)
    oz = min(p[2] for p in voxels)
    return {(x - ox, y - oy, z - oz): v for (x, y, z), v in voxels.items()}


def test_eos_stops_decoding() -> None:
    cv = _cv()
    blocks = decode_all(PieceDecoder(cv), [1, 3, 3, 3, 11, 2, 4, 4, 4, 12])
    assert len(blocks) == 1, "tokens after EOS must be ignored"


def test_air_blocks_are_not_emitted() -> None:
    """Air is absence, not a placement -- emitting it would punch holes in the world."""
    cv = ClusterVocab(max_dim=8, patterns=[((0, 0, 0, 0),)],
                      block_index_to_pair=[(0, 0)], merges=[], num_blocks=1)
    assert decode_all(PieceDecoder(cv), [1, 3, 3, 3, 11, 2]) == []
