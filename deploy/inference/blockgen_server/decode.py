"""Turn a live token stream into Minecraft block placements, one token at a time.

``cluster_tokens_to_structure`` / ``tokens_to_structure`` decode a *finished*
sequence into a ``Structure``. That is the wrong shape for this: we need blocks the
moment the token that determines them arrives, so the build appears in the world as
the model thinks. These decoders are incremental restatements of those functions and
deliberately keep their exact grammar and resync behaviour:

    [BOS, (X, Y, Z, PIECE)*, EOS]

A token that does not fit its slot drops the partial record and resyncs on the next
coordinate, rather than raising -- models emit malformed spans and a live demo must
survive them.

Coordinates are the model's own local frame (0-based, y-up, origin at the build's
corner). The mod offsets them to a world anchor. Note the trailing
``crop_to_non_air()`` that the batch decoders apply is *not* reproducible here (it
needs the finished bounding box), so a stream may sit a voxel or two off where the
batch path would put it -- irrelevant for placement, but do not expect
byte-identical coordinates between the two.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence, Tuple

from blockgen_server.blockmap import AIR, modern_state

BOS_TOKEN = 1
EOS_TOKEN = 2


@dataclass(frozen=True)
class Block:
    """One placement in the model's local coordinate frame."""
    x: int
    y: int
    z: int
    state: str


class Decoder(Protocol):
    def feed(self, token: int) -> List[Block]:
        """Consume one token; return any blocks it completed (usually none)."""


class _QuadDecoder:
    """Shared (X, Y, Z, THING) state machine; subclasses expand THING to blocks."""

    def __init__(self) -> None:
        self._buf: List[int] = []

    def _is_coord(self, tok: int) -> bool:
        raise NotImplementedError

    def _decode_coord(self, tok: int) -> int:
        raise NotImplementedError

    def _expand(self, tok: int, anchor: Tuple[int, int, int]) -> List[Block]:
        raise NotImplementedError

    def feed(self, token: int) -> List[Block]:
        token = int(token)
        if token in (BOS_TOKEN, 0):
            return []
        if len(self._buf) < 3:
            if self._is_coord(token):
                self._buf.append(self._decode_coord(token))
            else:
                self._buf.clear()  # desync: resync on the next coordinate
            return []
        anchor = (self._buf[0], self._buf[1], self._buf[2])
        self._buf = []
        return self._expand(token, anchor)


class PieceDecoder(_QuadDecoder):
    """3D-BPE piece tokens -> blocks (one piece expands to many voxels).

    Mirrors ``cluster_bpe.cluster_tokens_to_structure``.
    """

    def __init__(self, cv) -> None:
        super().__init__()
        self.cv = cv
        self._states = [modern_state(bid, data) for bid, data in cv.block_index_to_pair]

    def _is_coord(self, tok: int) -> bool:
        return self.cv.is_coord(tok)

    def _decode_coord(self, tok: int) -> int:
        return self.cv.decode_coord(tok)

    def _expand(self, tok: int, anchor) -> List[Block]:
        if not self.cv.is_piece(tok):
            return []
        ax, ay, az = anchor
        out = []
        for dx, dy, dz, bidx in self.cv.patterns[self.cv.decode_piece(tok)]:
            if 0 <= bidx < len(self._states):
                state = self._states[bidx]
                if state != AIR:
                    out.append(Block(ax + dx, ay + dy, az + dz, state))
        return out


class VoxelDecoder(_QuadDecoder):
    """Flat per-voxel block tokens -> one block each.

    Mirrors ``serialize.tokens_to_structure``.
    """

    def __init__(self, vocab) -> None:
        super().__init__()
        self.vocab = vocab
        self._states = [modern_state(bid, data) for bid, data in vocab.block_index_to_pair]

    def _is_coord(self, tok: int) -> bool:
        return self.vocab.is_coord(tok)

    def _decode_coord(self, tok: int) -> int:
        return self.vocab.decode_coord(tok)

    def _expand(self, tok: int, anchor) -> List[Block]:
        if not self.vocab.is_block(tok):
            return []
        idx = int(tok) - self.vocab.block_offset
        if not (0 <= idx < len(self._states)):
            return []
        state = self._states[idx]
        if state == AIR:
            return []
        return [Block(anchor[0], anchor[1], anchor[2], state)]


def decode_all(decoder: Decoder, tokens: Sequence[int]) -> List[Block]:
    """Run a whole token sequence through a decoder (used by tests/CLI)."""
    out: List[Block] = []
    for t in tokens:
        if int(t) == EOS_TOKEN:
            break
        out.extend(decoder.feed(t))
    return out
