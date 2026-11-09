"""The fixed-pooling face cascade: one bidirectional transformer that draws level
``f`` of a face from level ``2f``, all cells of a level in parallel.

The NCM/VAR recipe with the learned tokenizer taken out (research.md H.7, notes
§63): the levels are the pooled faces of ``blockgen.corpus.facade_levels``. The model
sees, for every cell of the level it is drawing, what its parent cell holds
(depth class, block class; "none" at the coarsest level), what the cell itself
already holds if revealed (MaskGIT-style partial masking, so a level can be drawn
in one pass or refined over several), whether the cell is inside the face's box,
its canvas position and the level index, and three global tokens for the box
(``U``, ``V``, ``extent``). Two heads: the depth class (0 see-through, ``1 + d``) and
the block class. Cells of one level are conditionally independent given the
inputs, which is the property this arm is testing.

``sample_level`` draws one level in ``steps`` passes (``steps=1`` is the plain
cascade: five forward passes per face), ``cascade`` draws the whole stack from the
box, optionally starting from a given coarse level (oracle-coarse completion).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from blockgen.corpus.facade_levels import CANVAS, FACTORS, N_DEPTH, UNK, grid_size


class FaceCascade(nn.Module):
    def __init__(self, n_blocks: int, dim: int = 256, layers: int = 6, heads: int = 8,
                 factors: Sequence[int] = FACTORS, dropout: float = 0.0):
        super().__init__()
        self.n_blocks = n_blocks
        self.factors = tuple(factors)
        self.dim = dim
        self.depth_mask_id = N_DEPTH              # own-state "masked"
        self.block_mask_id = n_blocks
        self.depth_none_id = N_DEPTH              # parent "none" (coarsest level)
        self.block_none_id = n_blocks
        self.parent_depth = nn.Embedding(N_DEPTH + 1, dim)
        self.parent_block = nn.Embedding(n_blocks + 1, dim)
        self.own_depth = nn.Embedding(N_DEPTH + 1, dim)
        self.own_block = nn.Embedding(n_blocks + 1, dim)
        self.inbox = nn.Embedding(2, dim)
        self.level = nn.Embedding(len(self.factors), dim)
        self.row = nn.Embedding(CANVAS, dim)
        self.col = nn.Embedding(CANVAS, dim)
        self.box_u = nn.Embedding(CANVAS + 1, dim)
        self.box_v = nn.Embedding(CANVAS + 1, dim)
        self.box_e = nn.Embedding(CANVAS + 1, dim)
        self.global_pos = nn.Parameter(torch.zeros(3, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head_depth = nn.Linear(dim, N_DEPTH)
        self.head_block = nn.Linear(dim, n_blocks)
        nn.init.normal_(self.global_pos, std=0.02)
        for emb in (self.parent_depth, self.parent_block, self.own_depth, self.own_block, self.inbox,
                    self.level, self.row, self.col, self.box_u, self.box_v, self.box_e):
            nn.init.normal_(emb.weight, std=0.02)

    def level_index(self, factor: int) -> int:
        return self.factors.index(factor)

    def forward(self, factor: int, parent_depth: torch.Tensor, parent_block: torch.Tensor,
                own_depth: torch.Tensor, own_block: torch.Tensor, inbox: torch.Tensor,
                box: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """All tensors ``[B, G, G]`` long (``inbox`` bool), ``box`` ``[B, 3]`` long
        ``(U, V, extent)``. Returns depth logits ``[B, G, G, N_DEPTH]`` and block
        logits ``[B, G, G, n_blocks]``."""
        B, G, _ = parent_depth.shape
        k = self.level_index(factor)
        dev = parent_depth.device
        origin = torch.arange(G, device=dev) * factor
        pos = self.row(origin)[:, None, :] + self.col(origin)[None, :, :]          # [G, G, D]
        x = (self.parent_depth(parent_depth) + self.parent_block(parent_block)
             + self.own_depth(own_depth) + self.own_block(own_block)
             + self.inbox(inbox.long()) + pos[None] + self.level.weight[k])
        x = x.reshape(B, G * G, self.dim)
        g = torch.stack([self.box_u(box[:, 0]), self.box_v(box[:, 1]), self.box_e(box[:, 2])], 1)
        g = g + self.global_pos[None] + self.level.weight[k]
        h = self.blocks(torch.cat([g, x], 1))
        h = self.norm(h[:, 3:]).reshape(B, G, G, self.dim)
        return self.head_depth(h), self.head_block(h)

    def none_parent(self, B: int, G: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.full((B, G, G), self.depth_none_id, dtype=torch.long, device=device),
                torch.full((B, G, G), self.block_none_id, dtype=torch.long, device=device))

    def masked_own(self, B: int, G: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.full((B, G, G), self.depth_mask_id, dtype=torch.long, device=device),
                torch.full((B, G, G), self.block_mask_id, dtype=torch.long, device=device))


def repeat2(a: torch.Tensor, r: int) -> torch.Tensor:
    """``[B, G, G] -> [B, rG, rG]`` by repetition: a coarse level onto the finer grid."""
    return a.repeat_interleave(r, 1).repeat_interleave(r, 2)


def level_loss(model: FaceCascade, factor: int, parent_depth, parent_block, own_depth, own_block,
               inbox, box, target_depth, target_block, loss_mask) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Mean cross-entropy per cell in ``loss_mask`` (depth) and per filled such cell
    (block), summed; also the two sums and their counts for token-weighted means."""
    ld, lb = model(factor, parent_depth, parent_block, own_depth, own_block, inbox, box)
    ce_d = F.cross_entropy(ld.reshape(-1, ld.shape[-1]), target_depth.reshape(-1), reduction="none")
    ce_b = F.cross_entropy(lb.reshape(-1, lb.shape[-1]), target_block.reshape(-1), reduction="none")
    m = loss_mask.reshape(-1)
    mb = m & (target_depth.reshape(-1) > 0)
    n_d = m.sum().clamp(min=1)
    n_b = mb.sum().clamp(min=1)
    sum_d = (ce_d * m).sum()
    sum_b = (ce_b * mb).sum()
    loss = sum_d / n_d + sum_b / n_b
    return loss, {"sum_depth": sum_d.detach(), "n_depth": m.sum().detach(),
                  "sum_block": sum_b.detach(), "n_block": mb.sum().detach()}


def _mask_logits(ld: torch.Tensor, lb: torch.Tensor, box: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Depth classes at or beyond the extent and the unknown block are never drawn."""
    cls_ = torch.arange(ld.shape[-1], device=ld.device)
    bad = cls_[None, :] > box[:, 2:3]                      # 1 + d > extent  <=>  d >= extent
    ld = ld.masked_fill(bad[:, None, None, :], float("-inf"))
    lb = lb.clone()
    lb[..., UNK] = float("-inf")
    return ld, lb


@torch.no_grad()
def sample_level(model: FaceCascade, factor: int, parent_depth: torch.Tensor, parent_block: torch.Tensor,
                 inbox: torch.Tensor, box: torch.Tensor, steps: int = 1, temperature: float = 1.0,
                 generator: Optional[torch.Generator] = None,
                 known_depth: Optional[torch.Tensor] = None, known_block: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Draw one level. ``temperature=0`` takes the mode of every cell (the model's
    deterministic completion). ``steps=1`` samples every in-box cell at once from one forward
    pass; ``steps>1`` is MaskGIT refinement: after each pass the most confident
    fraction of the still-masked cells (cosine schedule) is fixed and the rest are
    re-drawn with those revealed. Cells outside the box come back see-through."""
    B, G, _ = parent_depth.shape
    dev = parent_depth.device
    own_d, own_b = model.masked_own(B, G, dev)
    known = torch.zeros((B, G, G), dtype=torch.bool, device=dev)
    if known_depth is not None:
        known = known_depth != model.depth_mask_id
        own_d = torch.where(known, known_depth, own_d)
        own_b = torch.where(known, known_block, own_b)
    out_d = torch.zeros((B, G, G), dtype=torch.long, device=dev)
    out_b = torch.zeros((B, G, G), dtype=torch.long, device=dev)
    todo = inbox & ~known
    n_total = todo.reshape(B, -1).sum(1)
    for s in range(steps):
        ld, lb = model(factor, parent_depth, parent_block, own_d, own_b, inbox, box)
        ld, lb = _mask_logits(ld.float(), lb.float(), box)
        if temperature <= 0:                                   # the mode: deterministic completion
            pd, pb = F.softmax(ld, -1), F.softmax(lb, -1)
            d, b = ld.argmax(-1), lb.argmax(-1)
        else:
            pd = F.softmax(ld / temperature, -1)
            pb = F.softmax(lb / temperature, -1)
            d = torch.multinomial(pd.reshape(-1, pd.shape[-1]), 1, generator=generator).reshape(B, G, G)
            b = torch.multinomial(pb.reshape(-1, pb.shape[-1]), 1, generator=generator).reshape(B, G, G)
        b = torch.where(d > 0, b, torch.zeros_like(b))
        if s == steps - 1:
            reveal = todo
        else:
            conf = torch.log(pd.gather(-1, d[..., None])[..., 0].clamp(min=1e-9))
            conf = conf + torch.where(d > 0, torch.log(pb.gather(-1, b[..., None])[..., 0].clamp(min=1e-9)), 0.0)
            u = torch.rand(conf.shape, device=dev, generator=generator).clamp(min=1e-9)
            conf = conf - torch.log(-torch.log(u)) * (1.0 - (s + 1) / steps)     # annealed Gumbel noise
            conf = conf.masked_fill(~todo, float("-inf"))
            ratio = math.cos(math.pi / 2 * (s + 1) / steps)
            n_reveal = (n_total.float() * (1 - ratio)).ceil().long() - (n_total - todo.reshape(B, -1).sum(1))
            order = conf.reshape(B, -1).argsort(1, descending=True)
            rank = torch.empty_like(order)
            rank.scatter_(1, order, torch.arange(G * G, device=dev)[None].expand(B, -1))
            reveal = (rank < n_reveal.clamp(min=0)[:, None]).reshape(B, G, G) & todo
        out_d = torch.where(reveal, d, out_d)
        out_b = torch.where(reveal, b, out_b)
        own_d = torch.where(reveal, d, own_d)
        own_b = torch.where(reveal, b, own_b)
        todo = todo & ~reveal
    out_d = torch.where(known, own_d, out_d)
    out_b = torch.where(known, own_b, out_b)
    out_d = torch.where(inbox, out_d, torch.zeros_like(out_d))
    out_b = torch.where(inbox, out_b, torch.zeros_like(out_b))
    return out_d, out_b


@torch.no_grad()
def cascade(model: FaceCascade, box: torch.Tensor, inbox: Dict[int, torch.Tensor], steps: int = 1,
            temperature: float = 1.0, generator: Optional[torch.Generator] = None,
            start: Optional[Tuple[int, torch.Tensor, torch.Tensor]] = None,
            per_level: Optional[Dict[int, FaceCascade]] = None
            ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Draw every level coarse to fine. ``inbox[f]`` is the in-box grid of level ``f``.
    ``start=(f0, depth, block)`` fixes level ``f0`` to the given arrays and draws
    only the finer levels (oracle-coarse completion). ``per_level`` substitutes
    another copy of the model for a level (the checkpoint selected on that level's
    validation NLL). Returns ``{f: (depth, block)}``."""
    out: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    prev: Optional[Tuple[int, torch.Tensor, torch.Tensor]] = None
    for f in model.factors:
        m = per_level.get(f, model) if per_level else model
        if start is not None and f > start[0]:
            continue
        if start is not None and f == start[0]:
            out[f] = (start[1], start[2])
            prev = (f, start[1], start[2])
            continue
        G = grid_size(f)
        if prev is None:
            pdp, pbp = m.none_parent(box.shape[0], G, box.device)
        else:
            r = prev[0] // f
            pdp, pbp = repeat2(prev[1], r), repeat2(prev[2], r)
        d, b = sample_level(m, f, pdp, pbp, inbox[f], box, steps, temperature, generator)
        out[f] = (d, b)
        prev = (f, d, b)
    return out
