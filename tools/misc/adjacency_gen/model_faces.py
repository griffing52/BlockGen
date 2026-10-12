"""Face-node transformer with coarse-shape conditioning.

Tokens: 6 face tokens per block plus (optionally) one token per coarse cell.

* Face token of (block i, direction d): type embedding of i, direction embedding of d, a
  random per-block identity vector (RNF, fresh every forward, shared by the block's 6
  faces, so faces know their block and identical blocks are distinguishable), two
  order-free set statistics (log same-type count, log set size), a placed flag, the
  block's position if placed (Fourier features of the window-frame coordinate; unplaced
  blocks get none), the face state (unplaced / placed and open / placed and facing a
  placed block) and, for the oracle ablation, the block's coarse-cell position.
* Coarse token: occupancy-count embedding + dominant-type embedding (+ with
  ``residual``, an embedding of how many blocks the current partial build already has
  in the cell, so the builder can see which cells are still short) + the cell centre's
  Fourier position (the same encoder as placed blocks, so fine and coarse positions live
  in one space). Face tokens and coarse tokens share one transformer (joint attention).

Heads:

* glue: three axis matrices ``X_a[i, j] = <Wq_a h(i,+a), Wk_a h(j,-a)> / sqrt(dk) + b_a``.
  One entry per unordered face pair, so the 6N x 6N face matrix is symmetric and
  complementary by construction; the diagonal (same block) is masked by the users.
* exposed: one logit per face (exposed vs buried).
* seed: block-by-fine-cell score for the first placement of the sequential builder
  (block embedding vs cell embedding = Fourier position + the containing coarse cell's
  hidden state when conditioned).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class FaceConfig:
    n_types: int
    size: int = 8                 # window side
    f: int = 2                    # coarse factor; 0 = no conditioning
    oracle: bool = False          # give each block its true coarse cell (f must be > 0)
    residual: bool = False        # coarse tokens also see how many blocks are placed in the cell
    merge: bool = False           # merge formulation: blocks get a component feature (see merge.py)
    d: int = 192
    layers: int = 6
    heads: int = 6
    dk: int = 64
    rid: int = 8
    n_freq: int = 4

    def to_dict(self):
        return asdict(self)

    @property
    def G(self) -> int:
        return self.size // self.f if self.f else 0


class Fourier(nn.Module):
    """Continuous 3D position (window units) -> d."""

    def __init__(self, size: int, n_freq: int, d: int):
        super().__init__()
        self.size = size
        self.register_buffer("freq", (2.0 ** torch.arange(n_freq)) * math.pi, persistent=False)
        self.lin = nn.Linear(3 + 6 * n_freq, d)

    def forward(self, x):                                  # [..., 3] float
        u = x / self.size
        a = u[..., None] * self.freq                       # [...,3,F]
        f = torch.cat([u, a.sin().flatten(-2), a.cos().flatten(-2)], -1)
        return self.lin(f)


class FaceModel(nn.Module):
    def __init__(self, cfg: FaceConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.type_emb = nn.Embedding(cfg.n_types, d)
        self.dir_emb = nn.Embedding(6, d)
        self.state_emb = nn.Embedding(3, d)
        self.placed_emb = nn.Embedding(2, d)
        self.feat = nn.Linear(2 + cfg.rid, d)
        if cfg.merge:
            self.comp_feat = nn.Linear(cfg.rid, d)
            self.comp_size = nn.Linear(1, d)
        self.pos = Fourier(cfg.size, cfg.n_freq, d)
        if cfg.f:
            G = cfg.G
            self.count_emb = nn.Embedding(cfg.f ** 3 + 1, d)
            self.dom_emb = nn.Embedding(cfg.n_types + 1, d)          # + EMPTY
            self.tok_emb = nn.Embedding(2, d)                        # face / coarse
            if cfg.residual:
                self.placed_c_emb = nn.Embedding(cfg.f ** 3 + 1, d)
            c = torch.stack(torch.meshgrid(*[torch.arange(G)] * 3, indexing="ij"), -1).reshape(-1, 3)
            self.register_buffer("cell_centre", (c.float() + 0.5) * cfg.f - 0.5, persistent=False)
            if cfg.oracle:
                self.ocell_lin = nn.Linear(d, d)
        S = cfg.size
        g = torch.stack(torch.meshgrid(*[torch.arange(S)] * 3, indexing="ij"), -1).reshape(-1, 3)
        self.register_buffer("fine_cells", g.float(), persistent=False)
        if cfg.f:
            self.register_buffer("fine_to_coarse",
                                 ((g // cfg.f) * torch.tensor([cfg.G ** 2, cfg.G, 1])).sum(-1),
                                 persistent=False)
        layer = nn.TransformerEncoderLayer(d, cfg.heads, 4 * d, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.q = nn.Linear(d, cfg.dk * 3)
        self.k = nn.Linear(d, cfg.dk * 3)
        self.glue_bias = nn.Parameter(torch.full((3,), -3.0))
        self.ex_head = nn.Linear(d, 1)
        self.seed_u = nn.Linear(d, cfg.dk)
        self.seed_v = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg.dk))
        if cfg.f:
            self.seed_c = nn.Linear(d, d)

    def forward(self, types, pad, placed, pos, fstate, cond_count=None, cond_dom=None, ocell=None,
                rid=None, cond_placed=None, comp=None):
        """types/pad/placed [B,N]; pos [B,N,3] long; fstate [B,N,6]; cond_* [B,C];
        ocell [B,N] long (oracle coarse cell). Returns dict X [B,3,N,N], ex [B,N,6],
        seed [B,N,S^3]."""
        cfg = self.cfg
        Bsz, N = types.shape
        valid = (~pad).float()
        same = (types[:, :, None] == types[:, None, :]).float() * valid[:, None, :]
        cnt = same.sum(-1).clamp(min=1)
        n = valid.sum(-1, keepdim=True).expand_as(cnt).clamp(min=1)
        if rid is None:
            rid = torch.randn(Bsz, N, cfg.rid, device=types.device)
        stats = torch.stack([cnt.log(), n.log() / 2], -1)
        blk = (self.type_emb(types) + self.feat(torch.cat([stats, rid], -1))
               + self.placed_emb(placed.long())
               + self.pos(pos.float()) * placed[..., None].float())
        if cfg.merge:
            # component feature: a random vector per component (fresh every forward, shared by
            # its blocks, like the per-block identity), plus log component size. ``pos`` then
            # holds offsets to the component anchor and ``placed`` marks multi-block components.
            cz = torch.randn(Bsz, N, cfg.rid, device=types.device)
            cr = torch.gather(cz, 1, comp[..., None].expand(-1, -1, cfg.rid))
            same_c = (comp[:, :, None] == comp[:, None, :]).float() * valid[:, None, :]
            blk = blk + self.comp_feat(cr) + self.comp_size(same_c.sum(-1, keepdim=True).clamp(min=1).log())
        if cfg.oracle:
            centre = self.cell_centre[ocell.clamp(min=0)]
            blk = blk + self.ocell_lin(self.pos(centre))
        x = blk[:, :, None, :] + self.dir_emb.weight[None, None] + self.state_emb(fstate)   # [B,N,6,d]
        x = x.reshape(Bsz, N * 6, -1)
        mask = pad[:, :, None].expand(Bsz, N, 6).reshape(Bsz, N * 6)
        if cfg.f:
            x = x + self.tok_emb.weight[0]
            c = (self.count_emb(cond_count) + self.dom_emb(cond_dom)
                 + self.pos(self.cell_centre)[None] + self.tok_emb.weight[1])
            if cfg.residual:
                c = c + self.placed_c_emb(cond_placed.clamp(max=cfg.f ** 3))
            x = torch.cat([x, c], 1)
            mask = torch.cat([mask, torch.zeros(Bsz, c.shape[1], dtype=torch.bool, device=x.device)], 1)
        # no padding in the batch (e.g. batch size 1 on full houses): let SDPA use a fused kernel
        h = self.ln(self.enc(x, src_key_padding_mask=mask if bool(mask.any()) else None))
        hf = h[:, :N * 6].reshape(Bsz, N, 6, -1)
        dk = cfg.dk
        q = self.q(hf[:, :, 0::2]).view(Bsz, N, 3, 3, dk)      # [B,N,axis(+face),head axis,dk]
        k = self.k(hf[:, :, 1::2]).view(Bsz, N, 3, 3, dk)
        q = torch.stack([q[:, :, a, a] for a in range(3)], 1)   # [B,3,N,dk]
        k = torch.stack([k[:, :, a, a] for a in range(3)], 1)
        X = q @ k.transpose(-1, -2) / math.sqrt(dk) + self.glue_bias[None, :, None, None]
        ex = self.ex_head(hf).squeeze(-1)
        if cfg.merge:                                            # the merge decoder has no seed step
            return {"X": X, "ex": ex, "seed": None}
        v = self.pos(self.fine_cells)[None]                      # [1,S^3,d]
        if cfg.f:
            hc = h[:, N * 6:]
            v = v + self.seed_c(hc[:, self.fine_to_coarse])
        seed = self.seed_u(hf.mean(2)) @ self.seed_v(v).transpose(-1, -2) / math.sqrt(dk)
        return {"X": X, "ex": ex, "seed": seed}
