"""Pick-and-place: two heads over a shared graph encoder.

    Picker :  G      -> P(V)     which piece to place next (or STOP)
    Placer :  G x V  -> P(E)     which open face to attach it to

**Why this shape.** T21 built the same growth process with placement *implicit* —
a hand-written frontier heap decided where each piece went, and the model only
chose the piece. It failed, and the diagnosis was specific: the model "learned
the op-frequency distribution but cannot read its own op history as geometry",
closing the frontier down *harder* the more real structure it was shown
(close-rate 0.693 -> 0.838 -> 0.896 against a flat ground-truth 0.606). That is
blindness, not drift, and `implementation_plan.md` §3 concluded a state encoder
over the placed structure is a **prerequisite**, not a capacity upgrade. This
module is that encoder, plus a placement head that makes "where" a learned
decision instead of an inherited one.

**Connection encoding instead of positional encoding.** A node's input features
say what it is and *how it joined*: its own piece, the face direction it arrived
through, and its parent's piece. Nothing about the future. Geometry then enters
attention directly as a learned bias on the **relative 3D offset** between every
pair of placed nodes — so two 6-adjacent nodes get a specific learned
interaction, and the general lattice relationship is available at every layer.
That is the "connection embedding" doing the job absolute position would do in a
text transformer, and it is the mechanism intended to fix blindness: the model
cannot help but see the geometry, because the geometry is in the attention.

Offsets are clamped to +-`rel_clamp` on each axis. Locality is the point; two
nodes 30 voxels apart do not need their exact separation, and clamping keeps the
bias table at `(2r+1)^3` entries.

**The placer is a pointer network, not an n^2 map.** Scoring every ordered pair
would be wasteful and mostly illegal. The candidate set is the *open faces*: at
most `6N`, usually far fewer. Query comes from the current state and the piece
the picker just chose; keys come from each placed node paired with each of the
six directions. Illegal faces are masked with `-inf` **before** the softmax —
not renormalized after, which would be numerically worse and would make the
cross-entropy target inconsistent with the distribution actually sampled from.

**Training is one encoder pass per build.** Nodes are laid out in growth order
under a causal mask, so the state before every step is available simultaneously
and all steps are supervised in parallel (`implementation_plan.md` §3's
teacher-forcing shortcut). Incremental encoding is a sampling-time concern; this
module re-encodes per step when generating, which is honest O(N^2) and fine at
MVP sizes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from blockgen.utils.graph_data import PORT_DIRECTIONS

N_DIR = len(PORT_DIRECTIONS)
_DIRS_NP = np.array(PORT_DIRECTIONS, dtype=np.int64)

#: Reserved piece ids. Real piece tokens are remapped to >= PIECE_OFFSET.
STOP = 0
PAD = 1
PIECE_OFFSET = 2


@dataclass
class PickAndPlaceConfig:
    n_pieces: int                      # palette size, EXCLUDING STOP/PAD
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    rel_clamp: int = 4                 # relative-offset bias radius per axis
    max_nodes: int = 256

    @property
    def vocab_size(self) -> int:
        return self.n_pieces + PIECE_OFFSET

    @property
    def n_rel_buckets(self) -> int:
        return (2 * self.rel_clamp + 1) ** 3


# --------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------
class RelativeGeometryBias(nn.Module):
    """Per-head attention bias from the relative 3D offset between nodes.

    This is where the graph enters the model. An edge is just the special case
    `offset == one of the six unit directions`, and the same table covers
    "two apart", "diagonal", "far" without enumerating edges.
    """

    def __init__(self, cfg: PickAndPlaceConfig):
        super().__init__()
        self.r = cfg.rel_clamp
        self.side = 2 * cfg.rel_clamp + 1
        self.table = nn.Embedding(cfg.n_rel_buckets, cfg.nhead)
        nn.init.zeros_(self.table.weight)

    def bucket(self, coords: torch.Tensor) -> torch.Tensor:
        """coords [B, N, 3] -> bucket ids [B, N, N]."""
        rel = coords[:, :, None, :] - coords[:, None, :, :]      # [B,N,N,3]
        rel = rel.clamp(-self.r, self.r) + self.r
        return (rel[..., 0] * self.side + rel[..., 1]) * self.side + rel[..., 2]

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """-> [B, nhead, N, N] additive bias."""
        return self.table(self.bucket(coords)).permute(0, 3, 1, 2)


class _Block(nn.Module):
    def __init__(self, cfg: PickAndPlaceConfig):
        super().__init__()
        self.nhead = cfg.nhead
        self.head_dim = cfg.d_model // cfg.nhead
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.dim_feedforward), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.dim_feedforward, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        shape = (B, N, self.nhead, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.drop(self.proj(out.transpose(1, 2).reshape(B, N, D)))
        return x + self.drop(self.mlp(self.norm2(x)))


class GraphEncoder(nn.Module):
    """Causal transformer over placed nodes, in growth order.

    Node features carry only what was knowable when the node was placed -- its
    piece, the direction it arrived from, and its parent's piece. Encoding the
    node's *final* 6-neighbourhood here would leak nodes placed later and inflate
    every number in training while doing nothing at sampling time.
    """

    def __init__(self, cfg: PickAndPlaceConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.piece_embed = nn.Embedding(cfg.vocab_size, d, padding_idx=PAD)
        self.dir_embed = nn.Embedding(N_DIR + 1, d)        # +1 = "seed, no parent"
        self.parent_embed = nn.Embedding(cfg.vocab_size, d, padding_idx=PAD)
        self.depth_proj = nn.Linear(2, d)
        self.bos = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.bos, std=0.02)
        self.rel_bias = RelativeGeometryBias(cfg)
        self.blocks = nn.ModuleList(_Block(cfg) for _ in range(cfg.num_layers))
        self.norm = nn.LayerNorm(d)

    def node_features(self, pieces: torch.Tensor, direction: torch.Tensor,
                      parent_piece: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        n = pieces.shape[1]
        # Normalize the step index by a FIXED constant, never by the current
        # sequence length. Dividing by `n` makes the feature mean different
        # things at train and sample time -- in training n is the padded batch
        # max (192), while three nodes into generation n is 3, so every early
        # node reads as "we are at the end of the build". Measured: the model
        # learned STOP-at-1.0 and generated a median of 30 blocks against
        # sequences that were all exactly 192 long. Pinned by
        # `test_prefix_features_match_full_sequence`.
        step = torch.arange(n, device=pieces.device, dtype=torch.float32)
        step = (step / float(self.cfg.max_nodes)).view(1, n, 1).expand(
            pieces.shape[0], n, 1)
        height = coords[..., 1:2].float() / 32.0
        x = (self.piece_embed(pieces)
             + self.dir_embed(direction)
             + self.parent_embed(parent_piece)
             + self.depth_proj(torch.cat([step, height], dim=-1)))
        return x

    def forward(self, pieces: torch.Tensor, direction: torch.Tensor,
                parent_piece: torch.Tensor, coords: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """-> ``[B, N+1, D]`` states. Index ``t`` is the state with nodes ``0..t-1``
        placed, so state 0 is the empty graph and state ``t`` is what the model
        sees when deciding node ``t``."""
        B, N = pieces.shape
        x = self.node_features(pieces, direction, parent_piece, coords)

        bias = self.rel_bias(coords)                                   # [B,H,N,N]
        causal = torch.full((N, N), float("-inf"), device=x.device)
        bias = bias + torch.triu(causal, diagonal=1)
        if pad_mask is not None:
            bias = bias.masked_fill(pad_mask[:, None, None, :], float("-inf"))
            # A fully-masked row yields NaN from softmax; let padded queries see
            # themselves. Their outputs are discarded by the loss either way.
            eye = torch.eye(N, device=x.device, dtype=torch.bool)
            bias = torch.where(pad_mask[:, None, :, None] & eye[None, None],
                               torch.zeros_like(bias), bias)

        for blk in self.blocks:
            x = blk(x, bias)
        x = self.norm(x)
        return torch.cat([self.bos.expand(B, 1, -1), x], dim=1)


# --------------------------------------------------------------------------
# Heads
# --------------------------------------------------------------------------
class Picker(nn.Module):
    """G -> P(V). A plain LM head over the palette plus STOP."""

    def __init__(self, cfg: PickAndPlaceConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, cfg.vocab_size))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        logits = self.net(states)
        logits[..., PAD] = float("-inf")      # PAD is never a legal choice
        return logits


class Placer(nn.Module):
    """G x V -> P(E), as a pointer over open faces.

    Query = current graph state + the piece just picked. Keys = every
    (placed node, direction) pair. The illegal-face mask arrives as an argument
    because legality is a property of the *world*, not of the network -- keeping
    it outside means the same head works for a bounded canvas, a collision rule,
    or a piece-specific docking rule without retraining.
    """

    def __init__(self, cfg: PickAndPlaceConfig):
        super().__init__()
        d = cfg.d_model
        self.piece_embed = nn.Embedding(cfg.vocab_size, d, padding_idx=PAD)
        self.dir_key = nn.Embedding(N_DIR, d)
        self.q = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.k = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.scale = 1.0 / math.sqrt(d)

    def forward(self, states: torch.Tensor, node_states: torch.Tensor,
                piece: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        """
        states      [B, T, D]      graph state when placing each of T nodes
        node_states [B, N, D]      contextualized state of every placed node
        piece       [B, T]         the piece the picker chose at each step
        legal       [B, T, N, 6]   bool, True where attachment is allowed
        -> logits   [B, T, N*6]    with -inf on illegal faces
        """
        B, T, D = states.shape
        N = node_states.shape[1]
        q = self.q(torch.cat([states, self.piece_embed(piece)], dim=-1))   # [B,T,D]

        dir_e = self.dir_key.weight.view(1, 1, N_DIR, D).expand(B, N, N_DIR, D)
        k = self.k(torch.cat([node_states[:, :, None, :].expand(B, N, N_DIR, D),
                              dir_e], dim=-1))                             # [B,N,6,D]
        k = k.reshape(B, N * N_DIR, D)

        logits = torch.einsum("btd,bkd->btk", q, k) * self.scale           # [B,T,N*6]
        return logits.masked_fill(~legal.reshape(B, T, N * N_DIR), float("-inf"))


class PickAndPlace(nn.Module):
    """The two heads over one encoder."""

    def __init__(self, cfg: PickAndPlaceConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = GraphEncoder(cfg)
        self.picker = Picker(cfg)
        self.placer = Placer(cfg)

    @property
    def max_nodes(self) -> int:
        return self.cfg.max_nodes

    def forward(self, pieces: torch.Tensor, direction: torch.Tensor,
                parent_piece: torch.Tensor, coords: torch.Tensor,
                legal: torch.Tensor, pad_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced pass. Returns ``(pick_logits, place_logits)``.

        `pick_logits` is ``[B, N+1, V]`` -- one prediction per node plus the
        final STOP. `place_logits` is ``[B, N, N*6]``; row 0 (the seed) has no
        target and is ignored by the loss.
        """
        states = self.encoder(pieces, direction, parent_piece, coords, pad_mask)
        node_states = states[:, 1:, :]
        pick_logits = self.picker(states)
        place_logits = self.placer(states[:, :-1, :], node_states, pieces, legal)
        return pick_logits, place_logits


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
def live_legality(coords: np.ndarray, max_extent: Optional[int] = None
                  ) -> np.ndarray:
    """``[N, 6]`` bool over the faces of an actually-placed set of coordinates.

    The sampling-time counterpart of `GrowthSequence.placement_masks`, which can
    precompute from a known final structure. Here the world is only what has been
    built so far.
    """
    n = len(coords)
    if n == 0:
        return np.zeros((0, N_DIR), dtype=bool)
    occupied = {tuple(int(v) for v in c) for c in coords}
    out = np.ones((n, N_DIR), dtype=bool)
    for i, c in enumerate(coords):
        for d in range(N_DIR):
            nb = (int(c[0] + _DIRS_NP[d, 0]), int(c[1] + _DIRS_NP[d, 1]),
                  int(c[2] + _DIRS_NP[d, 2]))
            if nb in occupied:
                out[i, d] = False
    if max_extent is not None and n:
        arr = np.asarray(coords)
        for i, c in enumerate(coords):
            for d in range(N_DIR):
                nb = arr[i] + _DIRS_NP[d]
                lo = np.minimum(arr.min(axis=0), nb)
                hi = np.maximum(arr.max(axis=0), nb)
                if int((hi - lo).max()) + 1 > max_extent:
                    out[i, d] = False
    return out


def _sample(logits: torch.Tensor, temperature: float, top_k: Optional[int]) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    if top_k:
        k = min(top_k, int(torch.isfinite(logits).sum().item()))
        if k > 0:
            keep = torch.topk(logits, k).indices
            masked = torch.full_like(logits, float("-inf"))
            masked[keep] = logits[keep]
            logits = masked
    if not torch.isfinite(logits).any():
        return -1
    return int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())


@dataclass
class Rollout:
    pieces: np.ndarray
    coords: np.ndarray
    parent: np.ndarray
    direction: np.ndarray
    stopped: bool           # True = emitted STOP, False = hit a cap or got stuck


@torch.no_grad()
def generate(model: PickAndPlace, *, max_nodes: Optional[int] = None,
             temperature: float = 1.0, top_k: Optional[int] = 40,
             max_extent: Optional[int] = None, device: str = "cuda",
             prefix: Optional["Rollout"] = None,
             generator: Optional[torch.Generator] = None) -> Rollout:
    """Grow one build. Re-encodes each step -- O(N^2) per build, MVP-acceptable.

    `implementation_plan.md` §3 lists cached incremental message passing as the
    fix; it is a sampling-time optimization and deliberately not done here.

    `prefix` teacher-forces an opening -- the first K nodes of a *real* build --
    and lets the model continue from there. That is T21's diagnostic: a model
    that reads its own history as geometry should continue a real prefix roughly
    as far as the real build goes, while a blind one shuts the frontier down
    harder the more real structure it is handed.
    """
    model.eval()
    cfg = model.cfg
    cap = max_nodes or cfg.max_nodes

    pieces: List[int] = []
    coords: List[np.ndarray] = []
    parents: List[int] = []
    dirs: List[int] = []
    if prefix is not None and len(prefix.pieces):
        pieces = [int(x) for x in prefix.pieces]
        coords = [np.asarray(c, dtype=np.int64) for c in prefix.coords]
        parents = [int(x) for x in prefix.parent]
        dirs = [int(x) for x in prefix.direction]

    def encode():
        if not pieces:
            p = torch.full((1, 1), PAD, dtype=torch.long, device=device)
            d = torch.full((1, 1), N_DIR, dtype=torch.long, device=device)
            c = torch.zeros((1, 1, 3), dtype=torch.long, device=device)
            pad = torch.ones((1, 1), dtype=torch.bool, device=device)
            return model.encoder(p, d, p, c, pad)[:, :1, :], None
        p = torch.tensor(pieces, device=device).view(1, -1)
        d = torch.tensor([x if x >= 0 else N_DIR for x in dirs],
                         device=device).view(1, -1)
        pp = torch.tensor([pieces[x] if x >= 0 else PAD for x in parents],
                          device=device).view(1, -1)
        c = torch.tensor(np.stack(coords), device=device).view(1, -1, 3)
        st = model.encoder(p, d, pp, c, None)
        return st[:, -1:, :], st[:, 1:, :]

    for _ in range(cap + 1):
        state, node_states = encode()
        pick = model.picker(state)[0, 0]
        piece = _sample(pick, temperature, top_k)
        if piece == STOP or piece < 0:
            return Rollout(np.array(pieces), np.array(coords) if coords
                           else np.zeros((0, 3), int), np.array(parents),
                           np.array(dirs), True)
        if len(pieces) >= cap:
            break

        if not pieces:                                   # the seed: no choice
            pieces.append(piece); coords.append(np.zeros(3, dtype=np.int64))
            parents.append(-1); dirs.append(-1)
            continue

        legal_np = live_legality(np.stack(coords), max_extent)
        if not legal_np.any():
            break
        legal = torch.from_numpy(legal_np).to(device).view(1, 1, -1, N_DIR)
        pl = model.placer(state, node_states,
                          torch.tensor([[piece]], device=device), legal)[0, 0]
        port = _sample(pl, temperature, top_k)
        if port < 0:
            break
        parent_i, d = divmod(port, N_DIR)
        pieces.append(piece)
        coords.append(coords[parent_i] + _DIRS_NP[d])
        parents.append(parent_i); dirs.append(d)

    return Rollout(np.array(pieces), np.array(coords) if coords
                   else np.zeros((0, 3), int), np.array(parents),
                   np.array(dirs), False)


def rollout_to_structure(r: Rollout, piece_decode, oriented: bool = False):
    """Rollout -> Structure. ``piece_decode`` maps a model piece id to a token."""
    from blockgen.utils.data import Structure
    from blockgen.utils.growth_order import unpack_piece

    if len(r.coords) == 0:
        z = np.zeros((1, 1, 1), dtype=np.int32)
        return Structure(block_ids=z, block_data=z.copy())
    pos = r.coords - r.coords.min(axis=0)
    dims = pos.max(axis=0) + 1
    ids = np.zeros(tuple(int(x) for x in dims), dtype=np.int32)
    data = np.zeros_like(ids)
    for i in range(len(pos)):
        bid, bdata = unpack_piece(piece_decode(int(r.pieces[i])), oriented)
        ids[tuple(pos[i])] = bid
        data[tuple(pos[i])] = bdata
    return Structure(block_ids=ids, block_data=data)
