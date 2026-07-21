"""Shared (de)serialization between ``Structure`` and the model representations.

Two representations, one block vocabulary, used by all three generative tracks:

1. **Token sequence** (Track A autoregressive transformer, Track C graph decoder):
   ``[BOS, X, Y, Z, BLOCK, X, Y, Z, BLOCK, ..., EOS]``. Occupied voxels are
   emitted in ``(y, z, x)`` raster order. Variable structure size is handled by
   the EOS terminator — no fixed grid, no size prior required.

2. **Fixed canonical grid** (Track B diffusion, and all nearest-neighbor eval):
   crop to the occupied bounding box, then center-pad into a ``grid^3`` array of
   block-class indices (0 == air). Translation-tolerant, so it doubles as the
   common representation for novelty search.

The vocabulary is built from a corpus (the small cached subset), keeping it tiny:

    [PAD, BOS, EOS] + coordinate tokens (max_dim per axis) + block-class tokens

Every emitted id is ``< vocab.vocab_size`` by construction (asserted), which is
what removes the CUDA device-side assert seen in ``VoxelTransformerAR.generate``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from blockgen.utils.data import Structure, _token_for


PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
NUM_SPECIAL = 3


@dataclass
class BlockVocab:
    """Unified token vocabulary shared across tracks.

    Layout (contiguous id ranges):
        0..2                              special (PAD, BOS, EOS)
        coord_offset .. +max_dim-1        coordinate tokens (shared across x/y/z)
        block_offset .. +num_blocks-1     block-class tokens
    """

    max_dim: int
    block_token_to_id: Dict[str, int]          # "id" / "id:data" string -> block index
    id_to_block_token: List[str]
    # The (block_id, block_data) pair each block index decodes back to.
    block_index_to_pair: List[Tuple[int, int]]
    # If True, orientation-bearing blocks (stairs/logs/doors/...) keep their facing as
    # a distinct token; must match how the vocab was built (see _token_for oriented).
    oriented: bool = False

    @property
    def coord_offset(self) -> int:
        return NUM_SPECIAL

    @property
    def block_offset(self) -> int:
        return NUM_SPECIAL + self.max_dim

    @property
    def num_blocks(self) -> int:
        return len(self.id_to_block_token)

    @property
    def vocab_size(self) -> int:
        return NUM_SPECIAL + self.max_dim + self.num_blocks

    # --- token id helpers -------------------------------------------------
    def coord_id(self, c: int) -> int:
        if not (0 <= c < self.max_dim):
            raise ValueError(f"coordinate {c} out of range [0,{self.max_dim})")
        return self.coord_offset + int(c)

    def block_id_token(self, block_index: int) -> int:
        return self.block_offset + int(block_index)

    def is_coord(self, tok: int) -> bool:
        return self.coord_offset <= tok < self.coord_offset + self.max_dim

    def is_block(self, tok: int) -> bool:
        return self.block_offset <= tok < self.block_offset + self.num_blocks

    def decode_coord(self, tok: int) -> int:
        return int(tok) - self.coord_offset

    def decode_block(self, tok: int) -> Tuple[int, int]:
        return self.block_index_to_pair[int(tok) - self.block_offset]


def build_block_vocab(structures: Sequence[Structure], max_dim: int,
                      oriented: bool = False) -> BlockVocab:
    """Build a vocabulary from a corpus of structures.

    Block classes are derived via ``_token_for`` (the existing id/id:data rule in
    ``data.py``), so the vocab stays consistent with the rest of the codebase and
    stays small when restricted to the cached subset. With ``oriented``,
    orientation-bearing blocks (stairs/logs/doors/...) keep their facing/axis as
    distinct tokens so the model can learn orientation (notes.md §17).
    """
    # Collect distinct block tokens with a representative (id, data) pair.
    token_to_pair: Dict[str, Tuple[int, int]] = {}
    for s in structures:
        occ = s.occupied_mask
        ids = s.block_ids[occ]
        datas = s.block_data[occ]
        for bid, bdata in zip(ids.tolist(), datas.tolist()):
            tok = _token_for(int(bid), int(bdata), oriented=oriented)
            if tok not in token_to_pair:
                token_to_pair[tok] = (int(bid), int(bdata))

    sorted_tokens = sorted(token_to_pair.keys())
    block_token_to_id = {tok: i for i, tok in enumerate(sorted_tokens)}
    block_index_to_pair = [token_to_pair[tok] for tok in sorted_tokens]

    return BlockVocab(
        max_dim=max_dim,
        block_token_to_id=block_token_to_id,
        id_to_block_token=sorted_tokens,
        block_index_to_pair=block_index_to_pair,
        oriented=oriented,
    )


# --- vocabulary persistence -----------------------------------------------
# A checkpoint does not record the vocabulary it was trained against, and token ids
# are meaningless without one, so a run that does not save its vocab is not loadable
# later -- see notes.md §18. Any script that trains a model should call
# ``save_block_vocab`` (or ``export.minecraftace.save_piece_vocab`` for a
# ClusterVocab) next to its ``torch.save``.
def save_block_vocab(vocab: BlockVocab, path: str) -> None:
    """Write a BlockVocab as JSON. ``id_to_block_token`` is redundant (recoverable
    from ``block_token_to_id``) and is left out, matching train_conditioned.py."""
    import json
    import os

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"max_dim": vocab.max_dim,
                   "block_token_to_id": vocab.block_token_to_id,
                   "block_index_to_pair": [list(p) for p in vocab.block_index_to_pair],
                   "oriented": vocab.oriented}, f)


def load_block_vocab(path: str) -> BlockVocab:
    """Inverse of ``save_block_vocab``."""
    import json

    with open(path) as f:
        blob = json.load(f)
    return BlockVocab(
        max_dim=blob["max_dim"],
        block_token_to_id=blob["block_token_to_id"],
        id_to_block_token=[t for t, _ in sorted(blob["block_token_to_id"].items(),
                                                key=lambda kv: kv[1])],
        block_index_to_pair=[tuple(p) for p in blob["block_index_to_pair"]],
        oriented=blob.get("oriented", False))


# --- token sequence <-> Structure -----------------------------------------
def structure_to_tokens(structure: Structure, vocab: BlockVocab) -> List[int]:
    """Serialize a structure to ``[BOS, (X,Y,Z,BLOCK)*, EOS]``.

    The structure is cropped to its occupied bounding box first, so coordinates
    are local (0..size-1) and fit within ``vocab.max_dim``.
    """
    s = structure.crop_to_non_air()
    if max(s.shape) > vocab.max_dim:
        raise ValueError(
            f"structure max dim {max(s.shape)} exceeds vocab.max_dim {vocab.max_dim}"
        )

    occ = s.occupied_mask
    # (y, z, x) raster order: iterate so the sequence is deterministic.
    coords = np.argwhere(occ)  # rows of (x, y, z)
    order = np.lexsort((coords[:, 0], coords[:, 2], coords[:, 1]))  # sort by y, then z, then x
    coords = coords[order]

    tokens: List[int] = [BOS_TOKEN]
    for x, y, z in coords.tolist():
        bid = int(s.block_ids[x, y, z])
        bdata = int(s.block_data[x, y, z])
        tok = _token_for(bid, bdata, oriented=vocab.oriented)
        block_index = vocab.block_token_to_id.get(tok)
        if block_index is None:
            # Unknown block (not in corpus) -> skip the voxel rather than crash.
            continue
        tokens.extend([vocab.coord_id(x), vocab.coord_id(y), vocab.coord_id(z),
                       vocab.block_id_token(block_index)])
    tokens.append(EOS_TOKEN)

    assert max(tokens) < vocab.vocab_size, "token id out of vocab range"
    return tokens


def tokens_to_structure(tokens: Sequence[int], vocab: BlockVocab) -> Structure:
    """Inverse of ``structure_to_tokens``. Robust to malformed model output.

    Parses the stream greedily as ``(coord, coord, coord, block)`` quadruples,
    skipping any token that doesn't fit the expected slot. Stops at EOS.
    """
    placed: List[Tuple[int, int, int, int, int]] = []  # (x, y, z, block_id, block_data)
    buf: List[int] = []
    for tok in tokens:
        tok = int(tok)
        if tok == EOS_TOKEN:
            break
        if tok in (BOS_TOKEN, PAD_TOKEN):
            continue
        # Expect 3 coords then a block token.
        if len(buf) < 3:
            if vocab.is_coord(tok):
                buf.append(vocab.decode_coord(tok))
            else:
                buf.clear()  # desync: drop and resync on next coord
        else:
            if vocab.is_block(tok):
                x, y, z = buf
                bid, bdata = vocab.decode_block(tok)
                placed.append((x, y, z, bid, bdata))
            buf = []

    if not placed:
        # Empty / degenerate generation -> 1x1x1 air.
        return Structure(block_ids=np.zeros((1, 1, 1), np.int32),
                         block_data=np.zeros((1, 1, 1), np.int32))

    arr = np.array(placed, dtype=np.int32)
    sx = int(arr[:, 0].max()) + 1
    sy = int(arr[:, 1].max()) + 1
    sz = int(arr[:, 2].max()) + 1
    block_ids = np.zeros((sx, sy, sz), np.int32)
    block_data = np.zeros((sx, sy, sz), np.int32)
    for x, y, z, bid, bdata in arr.tolist():
        block_ids[x, y, z] = bid
        block_data[x, y, z] = bdata
    return Structure(block_ids=block_ids, block_data=block_data)


# --- fixed canonical grid <-> Structure ------------------------------------
def structure_to_grid(structure: Structure, grid: int, vocab: BlockVocab) -> np.ndarray:
    """Crop + center-pad into a ``grid^3`` array of block-class indices (0=air).

    Block indices are ``vocab`` block index + 1 (so 0 is reserved for air).
    Structures larger than ``grid`` along any axis are downsampled to fit.
    """
    s = structure.crop_to_non_air()
    if max(s.shape) > grid:
        s = s.downsample(max_dim=grid)
        s = s.crop_to_non_air()

    out = np.zeros((grid, grid, grid), dtype=np.int64)
    sx, sy, sz = s.shape
    ox, oy, oz = (grid - sx) // 2, (grid - sy) // 2, (grid - sz) // 2

    occ = s.occupied_mask
    coords = np.argwhere(occ)
    for x, y, z in coords.tolist():
        tok = _token_for(int(s.block_ids[x, y, z]), int(s.block_data[x, y, z]),
                         oriented=vocab.oriented)
        block_index = vocab.block_token_to_id.get(tok)
        if block_index is None:
            class_id = 1  # unknown -> generic non-air class
        else:
            class_id = block_index + 1
        out[ox + x, oy + y, oz + z] = class_id
    return out


def grid_to_structure(grid_arr: np.ndarray, vocab: BlockVocab) -> Structure:
    """Inverse of ``structure_to_grid`` (class index -> (id, data))."""
    block_ids = np.zeros(grid_arr.shape, np.int32)
    block_data = np.zeros(grid_arr.shape, np.int32)
    occ = np.argwhere(grid_arr > 0)
    for x, y, z in occ.tolist():
        class_id = int(grid_arr[x, y, z]) - 1
        if 0 <= class_id < vocab.num_blocks:
            bid, bdata = vocab.block_index_to_pair[class_id]
        else:
            bid, bdata = 1, 0  # fallback to stone
        block_ids[x, y, z] = bid
        block_data[x, y, z] = bdata
    s = Structure(block_ids=block_ids, block_data=block_data)
    return s.crop_to_non_air()


def num_grid_classes(vocab: BlockVocab) -> int:
    """Number of voxel classes for the grid representation (air + blocks)."""
    return vocab.num_blocks + 1
