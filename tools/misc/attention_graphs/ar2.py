"""Attention from the repo's own voxel AR transformer (``VoxelTransformerAR2``).

The model reads ``[BOS, (X, Y, Z, BLOCK)*, EOS]``: one dedicated token per coordinate
value and one per block class, emitted in (y, z, x) raster order, trained from scratch
on houses (no pretrained numeracy, no text). Its blocks call the fused SDPA kernel, which
never materialises attention weights, so we capture each block's input with a hook and
recompute ``softmax(q k^T / sqrt(d))`` from the block's own ``norm1`` and ``qkv``.

Default checkpoint: ``canon16_flat`` from ``run_20260917_000431_canon_flat_native`` (the
canonical-split arm of T§9.0), evaluated on canonical *val* builds it never trained on.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .extract import Extraction, pool_to_nodes, rollout
from .scenes import REPO, Scene, voxel_scene

DEFAULT_RUN = REPO / "outputs" / "run_20260917_000431_canon_flat_native" / "canon16_flat"
SLOTS = ("X", "Y", "Z", "B")


def load_ar2(run_dir: Path = DEFAULT_RUN, device: str = "cpu", max_seq_len: int = 1600):
    """``(model, vocab)`` from a flat-token run dir (``model.pt`` + ``block_vocab.json``)."""
    from blockgen.models.voxel_transformer_ar2 import VoxelTransformerAR2
    from blockgen.training.train_ar import ARTrainConfig
    from blockgen.utils.serialize import load_block_vocab

    vocab = load_block_vocab(str(Path(run_dir) / "block_vocab.json"))
    sd = torch.load(Path(run_dir) / "model.pt", map_location="cpu")
    cfg = ARTrainConfig()
    model = VoxelTransformerAR2(vocab_size=sd["token_embedding.weight"].shape[0],
                                max_seq_len=max_seq_len, d_model=cfg.d_model, nhead=cfg.nhead,
                                num_layers=cfg.num_layers, dim_feedforward=cfg.dim_feedforward,
                                dropout=0.0, pe="phase4")
    model.load_state_dict(sd)
    return model.to(device).eval(), vocab


def val_builds(vocab, n: int, dim: int = 16, max_tokens: int = 1600, min_blocks: int = 20,
               seed: int = 0) -> List[Tuple[Scene, List[int]]]:
    """Canonical-split *val* houses, canonicalised like training: ``(scene, tokens)``."""
    from blockgen.experiments_gen import canonicalize
    from blockgen.experiments_transfer import _split_houses_canonical
    from blockgen.utils.serialize import structure_to_tokens

    _, h_val = _split_houses_canonical(seed=seed)
    out = []
    for s in canonicalize(h_val, dim):
        if len(out) >= n:
            break
        try:
            toks = structure_to_tokens(s, vocab)
        except ValueError:
            continue
        if not (4 * min_blocks + 2 <= len(toks) <= max_tokens):
            continue
        body = np.array(toks[1:-1]).reshape(-1, 4)
        coords = np.stack([[vocab.decode_coord(int(t)) for t in row[:3]] for row in body])
        names = [f"b{int(t)}" for t in body[:, 3]]
        scene = voxel_scene(coords, names, source=str(s.source_path))
        out.append((scene, toks))
    return out


def tokens_for_order(tokens: Sequence[int], order: str, seed: int = 0
                     ) -> Tuple[List[int], np.ndarray]:
    """Re-order the ``(X,Y,Z,B)`` groups; returns ``(tokens, permutation of blocks)``."""
    body = np.array(tokens[1:-1]).reshape(-1, 4)
    perm = (np.arange(len(body)) if order == "raster"
            else np.random.default_rng(seed).permutation(len(body)))
    return [tokens[0]] + body[perm].ravel().tolist() + [tokens[-1]], perm


def block_attention(blk, x: torch.Tensor) -> torch.Tensor:
    """Recompute one ``_CausalBlock``'s attention weights ``[B, H, T, T]`` from its input."""
    B, T, D = x.shape
    q, k, _ = blk.qkv(blk.norm1(x)).chunk(3, dim=-1)
    q = q.view(B, T, blk.nhead, blk.head_dim).transpose(1, 2)
    k = k.view(B, T, blk.nhead, blk.head_dim).transpose(1, 2)
    if blk.rope is not None:
        q, k = blk.rope.rotate(q), blk.rope.rotate(k)
    s = q @ k.transpose(-1, -2) / math.sqrt(blk.head_dim)
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
    return torch.softmax(s.masked_fill(causal, float("-inf")), dim=-1)


@torch.no_grad()
def ar2_attentions(model, tokens: Sequence[int]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """``(att [L, H, T, T], residual stream per layer [L+1] x [T, D])`` for one sequence."""
    if model.needs_bias() or model.geo is not None:
        raise NotImplementedError("ALiBi / geometric-bias checkpoints add a score bias; "
                                  "recompute it before using this")
    device = next(model.parameters()).device
    ids = torch.tensor([list(tokens)], device=device)
    inputs: List[torch.Tensor] = []
    hooks = [blk.register_forward_pre_hook(lambda m, args: inputs.append(args[0]))
             for blk in model.blocks]
    last: List[torch.Tensor] = []
    hooks.append(model.blocks[-1].register_forward_hook(lambda m, a, out: last.append(out[0])))
    try:
        model(ids)
    finally:
        for h in hooks:
            h.remove()
    att = torch.stack([block_attention(blk, x)[0] for blk, x in zip(model.blocks, inputs)])
    return att.float(), [x[0].float() for x in inputs + last]


def block_spans(n_blocks: int) -> List[Tuple[int, int]]:
    return [(1 + 4 * i, 5 + 4 * i) for i in range(n_blocks)]


@torch.no_grad()
def extract_ar2(model, tokens: Sequence[int], hidden_layers: Sequence[int] = (),
                keep_tokens: bool = False) -> Dict[str, Extraction]:
    """Per query slot (``X``, ``Y``, ``Z``, ``B`` and ``mean``), node-level attention.

    Keys always sum over all four tokens of an earlier block. Slot ``Y`` is the query
    that predicts the block's ``Z``; slot ``B`` sees the full block. Hidden states are
    mean-pooled over each block's four tokens.
    """
    att, resid = ar2_attentions(model, tokens)
    spans = block_spans((len(tokens) - 2) // 4)
    roll = rollout(att)
    hidden = {}
    if hidden_layers:
        Q = torch.zeros(len(spans), att.shape[-1], device=att.device)
        for a, (s, e) in enumerate(spans):
            Q[a, s:e] = 0.25
        hidden = {l: (Q @ resid[l]).cpu().numpy() for l in hidden_layers}
    out = {}
    for name in SLOTS + ("mean",):
        if name == "mean":
            qs = None
        else:
            k = SLOTS.index(name)
            qs = [(s + k, s + k + 1) for s, _ in spans]
        out[name] = Extraction(
            node_att=pool_to_nodes(att, spans, "mean", "sum", qs).cpu().numpy(),
            rollout=pool_to_nodes(roll, spans, "mean", "sum", qs).cpu().numpy(),
            hidden=hidden, token_att=att.cpu().numpy() if keep_tokens else None)
    return out

