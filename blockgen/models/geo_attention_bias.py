"""Per-head geometric attention bias for the native (X,Y,Z,PIECE) piece grammar.

The idea T29 tested on a text LLM and could not answer there: instead of hoping a
transformer infers 3D neighbourhood, add it to the attention scores as a learned
function of the **relative offset between the pieces two positions refer to**. That
run was inconclusive for two reasons this module fixes.

**It was one shared scalar across every layer and head.** Here the table is
per-head, `(H, n^3 + 3)`, which is the capacity axis notes.md §30 named as untested.
`VoxelTransformerAR2`'s attention block is hand-rolled precisely because stock
PyTorch "exposes neither q/k rotation nor per-head biases", and ALiBi already
passes a per-head bias, so this is the seam the model was built for.

**Its reference anchor was the *previous* line.** The native grammar is
`[BOS, (X, Y, Z, PIECE)*, EOS]`, so by the time a piece's type is predicted its own
X, Y and Z have already been emitted. The query that predicts piece *k* sits on
*k*'s Z token, and the reference can therefore be *k*'s own anchor: the geometry
answers "what belongs **here**", not "what belongs near where I just was". Nothing
here depends on a coordinate the model has not already produced -- the rule is
simply *the last fully-emitted anchor*:

    slot X, Y   -> the previous piece's anchor   (this piece's is incomplete)
    slot Z, PIECE -> this piece's own anchor

**Not fixed here, and worth stating:** the offset table is still only half
reachable. Under `(y,z,x)` raster emission order every earlier piece has `dy >= 0`,
which admits 364 of 729 clamped buckets (plus one for the self-reference case) --
a property of the *order*, not of the reference anchor, as notes.md §30's correction
records. Pairing this with a BFS/growth order is the follow-up that unlocks the rest.

**Why FlexAttention.** An explicit `attn_mask` disqualifies the fused SDPA kernels
and materializes `[B, H, L, L]` scores -- ~9.6 GB at B=8/H=8/L=5480, which the model
docstring already names as the reason long-sequence runs were near-OOM. A
`score_mod` computes the same bias per score without ever building that tensor, and
every input it needs is O(L).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn

def ensure_triton_compiler() -> None:
    """Point Triton's runtime C shim at a compiler that can see Python's headers.

    Triton compiles a small shim at import of the fused kernel and honours `$CC`. If
    that is a conda toolchain it does not search `/usr/include/x86_64-linux-gnu`, so
    the build dies on a missing `pyconfig.h`. The consequence is *not* a clean error:
    FlexAttention falls back to an unfused path that materializes the full score
    matrix, and training OOMs at ~12 GB with nothing pointing at the cause.

    Only redirects when the configured compiler cannot actually find `Python.h`, so
    a working toolchain -- including a deliberately chosen one -- is left alone.
    """
    import subprocess
    import sysconfig
    cc = os.environ.get("CC")
    if not cc:
        return
    inc = sysconfig.get_paths()["include"]
    probe = f"#include <Python.h>\nint main(void){{return 0;}}\n"
    try:
        ok = subprocess.run([cc, "-x", "c", "-I", inc, "-fsyntax-only", "-"],
                            input=probe, text=True, capture_output=True,
                            timeout=60).returncode == 0
    except Exception:  # noqa: BLE001 - a compiler we cannot even run is not usable
        ok = False
    if not ok and Path("/usr/bin/gcc").exists():
        print(f"[geo] $CC={cc} cannot compile against {inc}; using /usr/bin/gcc for "
              f"Triton (FlexAttention would otherwise fall back to an unfused kernel "
              f"and OOM)", flush=True)
        os.environ["CC"] = "/usr/bin/gcc"

#: Slot layout of the native grammar, after the leading BOS.
SLOT_X, SLOT_Y, SLOT_Z, SLOT_PIECE = 0, 1, 2, 3


def piece_and_slot(positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """`(piece_index, slot)` for token positions in `[BOS, (X,Y,Z,PIECE)*, EOS]`.

    BOS (position 0) and any trailing EOS/pad get piece -1, which every caller
    treats as "no geometry here".
    """
    rel = positions - 1
    piece = torch.where(rel >= 0, rel // 4, torch.full_like(rel, -1))
    slot = torch.where(rel >= 0, rel % 4, torch.full_like(rel, -1))
    return piece, slot


def anchors_from_tokens(tokens: torch.Tensor, coord_offset: int, max_dim: int,
                        n_pieces_max: int) -> torch.Tensor:
    """`[B, P, 3]` piece anchors read back out of the token ids.

    Coordinate tokens are `coord_offset + value`, so the anchors the bias needs are
    already in the input -- no extra collate plumbing, and no risk of the anchors
    disagreeing with the sequence the model is actually reading.

    Positions past the end of a (right-padded) sequence read whatever the pad token
    is; they are masked out downstream by `piece < 0`, and clamped here so a stray
    pad id cannot index out of range.
    """
    B, L = tokens.shape
    out = tokens.new_zeros((B, n_pieces_max, 3))
    for axis in range(3):
        pos = 1 + axis + 4 * torch.arange(n_pieces_max, device=tokens.device)
        valid = pos < L
        idx = pos.clamp(max=max(L - 1, 0))
        vals = tokens[:, idx] - coord_offset
        vals = vals.clamp(0, max_dim - 1)
        out[:, :, axis] = torch.where(valid.unsqueeze(0), vals, torch.zeros_like(vals))
    return out


class GeoAttentionBias(nn.Module):
    """Learned per-head bias on the clamped relative offset between two pieces.

    Zero-initialized, so at step 0 the arm is numerically identical to its control
    and anything it gains it learned. The last three table columns are the
    non-geometric cases, kept inside the same tensor so `score_mod` is one gather:

        n^3      same piece as the query   (its anchor may be incomplete -- no leak)
        n^3 + 1  BOS / EOS / pad
        n^3 + 2  no reference yet          (piece 0, before its Z is emitted)
    """

    def __init__(self, nhead: int, radius: int = 4):
        super().__init__()
        self.nhead = nhead
        self.r = radius
        self.n = 2 * radius + 1
        self.n_geo = self.n ** 3
        self.table = nn.Parameter(torch.zeros(nhead, self.n_geo + 3))

    # --- bucket construction (shared by the flex and dense paths) -----------
    def buckets(self, tokens: torch.Tensor, coord_offset: int, max_dim: int):
        """The O(L) tensors `score_mod` indexes, as **2-D tensors per axis**.

        For every query position, the anchor it reasons *from*; for every key
        position, the anchor it refers *to*. Keeping them off the L x L grid is the
        whole point, and keeping each one 2-D (`[B, L]`) is a FlexAttention
        requirement, not a style choice: a `score_mod` that indexes a 3-D capture as
        `t[b, q_idx, 0]` mixes advanced and constant indexing, and Dynamo cannot
        specialize the result ("could not extract specialized integer from
        data-dependent expression"). Three separate `[B, L]` gathers trace cleanly.

        Returns `(ref_axes, key_axes, piece, ref_valid)` where the first two are
        3-tuples of `[B, L]` int tensors, one per axis.
        """
        B, L = tokens.shape
        device = tokens.device
        pos = torch.arange(L, device=device)
        piece, slot = piece_and_slot(pos)                       # [L]
        n_pieces_max = int(max((L - 1) // 4 + 1, 1))
        anch = anchors_from_tokens(tokens, coord_offset, max_dim, n_pieces_max)

        # The last fully-emitted anchor: this piece once its Z is in, else the one
        # before it. Never an anchor the model has not produced.
        ref_piece = torch.where(slot >= SLOT_Z, piece, piece - 1)
        ref_piece = torch.where(piece >= 0, ref_piece, torch.full_like(ref_piece, -1))
        ref_valid = ref_piece >= 0
        ref_idx = ref_piece.clamp(min=0).clamp(max=n_pieces_max - 1)
        key_idx = piece.clamp(min=0).clamp(max=n_pieces_max - 1)

        ref_axes = tuple(anch[:, ref_idx, a].contiguous() for a in range(3))   # 3x [B, L]
        key_axes = tuple(anch[:, key_idx, a].contiguous() for a in range(3))
        return (ref_axes, key_axes, piece.expand(B, L).contiguous(),
                ref_valid.expand(B, L).contiguous())

    def make_score_mod(self, tokens: torch.Tensor, coord_offset: int, max_dim: int):
        """A `score_mod` closure for `torch.nn.attention.flex_attention`."""
        (rx, ry, rz), (kx, ky, kz), piece, ref_ok = self.buckets(
            tokens, coord_offset, max_dim)
        table, r, n, n_geo = self.table, self.r, self.n, self.n_geo

        def score_mod(score, b, h, q_idx, kv_idx):
            d0 = (rx[b, q_idx] - kx[b, kv_idx]).clamp(-r, r) + r
            d1 = (ry[b, q_idx] - ky[b, kv_idx]).clamp(-r, r) + r
            d2 = (rz[b, q_idx] - kz[b, kv_idx]).clamp(-r, r) + r
            geo = (d0 * n + d1) * n + d2
            qp, kp = piece[b, q_idx], piece[b, kv_idx]
            bucket = torch.where(kp < 0, n_geo + 1,
                                 torch.where(qp == kp, n_geo,
                                             torch.where(ref_ok[b, q_idx], geo, n_geo + 2)))
            return score + table[h, bucket]

        return score_mod

    def dense(self, tokens: torch.Tensor, coord_offset: int, max_dim: int
              ) -> torch.Tensor:
        """`[B, H, L, L]` reference implementation -- the oracle the flex path is
        tested against, and a usable fallback at short sequence lengths."""
        ref_axes, key_axes, piece, ref_ok = self.buckets(tokens, coord_offset, max_dim)
        B, L = piece.shape
        geo = torch.zeros((B, L, L), dtype=torch.long, device=tokens.device)
        for ra, ka in zip(ref_axes, key_axes):
            d = (ra.unsqueeze(2) - ka.unsqueeze(1)).clamp(-self.r, self.r) + self.r
            geo = geo * self.n + d
        qp = piece.unsqueeze(2).expand(B, L, L)
        kp = piece.unsqueeze(1).expand(B, L, L)
        bucket = torch.where(kp < 0, torch.full_like(geo, self.n_geo + 1),
                             torch.where(qp == kp, torch.full_like(geo, self.n_geo),
                                         torch.where(ref_ok.unsqueeze(2).expand(B, L, L),
                                                     geo,
                                                     torch.full_like(geo, self.n_geo + 2))))
        return self.table[:, bucket.reshape(-1)].reshape(self.nhead, B, L, L).permute(1, 0, 2, 3)
