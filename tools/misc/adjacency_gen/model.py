"""Permutation-equivariant masked-edge model: node set in, directed adjacency out.

Input per node: type embedding + two order-free set statistics (log count of same-type
nodes in the set, log set size) + an optional random node feature (RNF). No positions.

RNF (Sato et al. 2021; Abboud et al. 2021): a fresh Gaussian vector per node per
forward pass. Without it, two same-type nodes with identical observed edges get
identical embeddings, so every pair between them gets the identical prediction and
an all-masked set cannot break the symmetry. RNF keeps the model equivariant *in
distribution* (permuting nodes permutes the outputs, the noise is i.i.d.) and is the
standard fix; the equivariance test turns it off.

Observed edges condition the encoder in two ways (Graphormer-style attention bias plus
Shaw-style relation values): each ordered pair carries a state code
``0 = masked, 1 = observed none, 1 + k = observed offset class k, K + 2 = self`` whose
embedding is factorized over (state, dx, dy, dz) so offsets share parameters.

Edge head: an MLP over ``(A h_i + B h_j + rel_ij)`` gives raw scores ``f(i, j, k)`` for
the ``K + 1`` labels; the logits are tied, ``z(i, j, k) = f(i, j, k) + f(j, i, opp(k))``,
so ``z(j, i, opp(k)) == z(i, j, k)`` exactly: label(i,j) = +x iff label(j,i) = -x.
Labels impossible for the pair's two shapes (pieces that would overlap or not touch)
are masked to ``-inf``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import OffsetSpace


@dataclass
class ModelConfig:
    n_types: int
    B: int = 1                      # offset cube half-width (1 for voxels)
    d: int = 192
    layers: int = 6
    heads: int = 8
    d_rel: int = 32
    d_pair: int = 128
    rnf: int = 8
    dropout: float = 0.0

    def to_dict(self):
        return asdict(self)


class RelLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, h = cfg.d, cfg.heads
        self.h, self.dh = h, d // h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.bias = nn.Linear(cfg.d_rel, h)
        self.rel_v = nn.Linear(cfg.d_rel, d)          # applied after the attention sum
        self.out = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, rel, key_pad):
        Bsz, N, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(Bsz, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)            # [B,h,N,N]
        att = att + self.bias(rel).permute(0, 3, 1, 2)
        att = att.masked_fill(key_pad[:, None, None, :], float("-inf"))
        a = att.softmax(-1)
        o = a @ v                                                        # [B,h,N,dh]
        # relation values: sum_j a_hij * rel_ij, then project per head
        r = torch.einsum("bhij,bijr->bhir", a, rel)                      # [B,h,N,d_rel]
        rv = self.rel_v.weight.view(self.h, self.dh, -1)                 # [h,dh,d_rel]
        o = o + torch.einsum("bhir,hdr->bhid", r, rv)
        o = o.transpose(1, 2).reshape(Bsz, N, d)
        x = x + self.drop(self.out(o))
        return x + self.drop(self.ff(self.ln2(x)))


class AdjacencyModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        space = OffsetSpace(cfg.B)
        self.K = space.K
        self.register_buffer("opp", torch.as_tensor(space.opp), persistent=False)
        offs = torch.as_tensor(space.offsets) + cfg.B                    # [K,3] in 0..2B
        self.register_buffer("offs", offs, persistent=False)
        d = cfg.d
        self.type_emb = nn.Embedding(cfg.n_types, d)
        self.feat = nn.Linear(2 + cfg.rnf, d)
        # relation code embedding, factorized: state (masked/none/edge/self) + dx + dy + dz
        s = 2 * cfg.B + 1
        self.state_emb = nn.Embedding(4, cfg.d_rel)
        self.ax_emb = nn.ModuleList([nn.Embedding(s, cfg.d_rel) for _ in range(3)])
        self.layers = nn.ModuleList([RelLayer(cfg) for _ in range(cfg.layers)])
        self.ln = nn.LayerNorm(d)
        self.pa = nn.Linear(d, cfg.d_pair)
        self.pb = nn.Linear(d, cfg.d_pair)
        self.pr = nn.Linear(cfg.d_rel, cfg.d_pair)
        self.pair_mlp = nn.Sequential(nn.GELU(), nn.Linear(cfg.d_pair, cfg.d_pair), nn.GELU(),
                                      nn.Linear(cfg.d_pair, self.K + 1))

    # -- relation table: code -> embedding ------------------------------------
    def rel_table(self) -> torch.Tensor:
        K = self.K
        st = torch.zeros(K + 3, dtype=torch.long, device=self.offs.device)
        st[1] = 1
        st[2:K + 2] = 2
        st[K + 2] = 3
        tab = self.state_emb(st)
        ax = sum(self.ax_emb[a](self.offs[:, a]) for a in range(3))      # [K,d_rel]
        tab = torch.cat([tab[:2], tab[2:K + 2] + ax, tab[K + 2:]], 0)
        return tab

    def node_features(self, types, pad):
        valid = (~pad).float()
        same = (types[:, :, None] == types[:, None, :]).float() * valid[:, None, :]
        cnt = (same.sum(-1)).clamp(min=1)
        n = valid.sum(-1, keepdim=True).expand_as(cnt)
        f = [torch.log(cnt), torch.log(n.clamp(min=1)) / 2]
        if self.cfg.rnf:
            rnf = torch.randn(*types.shape, self.cfg.rnf, device=types.device) if self.training or self._rnf_on \
                else torch.zeros(*types.shape, self.cfg.rnf, device=types.device)
            return torch.cat([torch.stack(f, -1), rnf], -1)
        return torch.stack(f, -1)

    _rnf_on = True

    def set_rnf(self, on: bool):
        self._rnf_on = on

    def forward(self, types, pad, obs, valid=None):
        """types [B,N] long, pad [B,N] bool (True = padding), obs [B,N,N] long codes,
        valid [B,N,N,K+1] bool or None. Returns tied logits [B,N,N,K+1]."""
        Bsz, N = types.shape
        x = self.type_emb(types) + self.feat(self.node_features(types, pad))
        obs = obs.clone()
        eye = torch.eye(N, dtype=torch.bool, device=types.device)
        obs[:, eye] = self.K + 2
        rel = self.rel_table()[obs]                                       # [B,N,N,d_rel]
        for layer in self.layers:
            x = layer(x, rel, pad)
        x = self.ln(x)
        p = self.pa(x)[:, :, None, :] + self.pb(x)[:, None, :, :] + self.pr(rel)
        f = self.pair_mlp(p)                                              # [B,N,N,K+1]
        z = f + f.transpose(1, 2)[..., self.opp]
        if valid is not None:
            z = z.masked_fill(~valid, float("-inf"))
        return z


def obs_codes(labels: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """Pair state codes: 0 masked, 1 + label where observed (label 0 -> code 1)."""
    return torch.where(observed, labels + 1, torch.zeros_like(labels))
