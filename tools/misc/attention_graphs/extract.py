"""Self-attention extraction from any HF causal LM, pooled from tokens to nodes.

A *node* is a span of tokens (one block line). Token attention ``A[l, h, q, k]`` is
pooled to node attention ``N[l, h, a, b]`` by averaging over the query tokens of node
``a`` (or taking its last token) and summing over the key tokens of node ``b``. The
model must be loaded with ``attn_implementation="eager"``: the fused SDPA / flash
kernels never materialise the attention matrix, so they return none.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch


def pick_device(min_free_gb: float = 4.0) -> str:
    """CUDA if at least ``min_free_gb`` is free right now (the GPU is shared), else CPU."""
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free / 2**30 >= min_free_gb:
            return "cuda"
    return "cpu"


def load_model(name: str, device: Optional[str] = None, dtype: Optional[torch.dtype] = None):
    """Load ``(model, tokenizer)`` with eager attention, offline from the local HF cache."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    device = device or pick_device()
    dtype = dtype or (torch.bfloat16 if device == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager",
                                                 dtype=dtype)
    return model.to(device).eval(), tok


# ---------------------------------------------------------------- pooling

def pooling_matrices(spans: Sequence[Tuple[int, int]], n_tokens: int, query: str = "mean",
                     key: str = "sum", query_spans: Optional[Sequence[Tuple[int, int]]] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """``Q [N, T]`` and ``K [N, T]`` such that node attention is ``Q @ A @ K^T``.

    ``query_spans`` overrides which tokens act as node ``a``'s query (e.g. one slot of the
    voxel AR's ``X Y Z BLOCK`` group); keys always cover the whole ``spans`` entry.
    """
    Q = torch.zeros(len(spans), n_tokens)
    K = torch.zeros(len(spans), n_tokens)
    for a, (s, e) in enumerate(spans):
        qs, qe = query_spans[a] if query_spans is not None else (s, e)
        if query == "mean":
            Q[a, qs:qe] = 1.0 / (qe - qs)
        elif query == "last":
            Q[a, qe - 1] = 1.0
        else:
            raise ValueError(f"query pooling must be 'mean' or 'last', got {query!r}")
        if key == "sum":
            K[a, s:e] = 1.0
        elif key == "mean":
            K[a, s:e] = 1.0 / (e - s)
        else:
            raise ValueError(f"key pooling must be 'sum' or 'mean', got {key!r}")
    return Q, K


def pool_to_nodes(att: torch.Tensor, spans: Sequence[Tuple[int, int]], query: str = "mean",
                  key: str = "sum", query_spans=None) -> torch.Tensor:
    """``[..., T, T] -> [..., N, N]`` token-to-node pooling."""
    Q, K = pooling_matrices(spans, att.shape[-1], query, key, query_spans)
    Q, K = Q.to(att), K.to(att)
    return Q @ att @ K.T


def renorm_rows(node_att: np.ndarray) -> np.ndarray:
    """Renormalise each row over the *earlier* nodes only (T28's convention).

    Drops the prompt (and its attention sink, often half the mass) and the node's own
    line, so a row is a distribution over candidate parents ``j < i``.
    """
    low = np.tril(node_att, -1)
    return low / np.maximum(low.sum(-1, keepdims=True), 1e-9)


def symmetrize(node_att: np.ndarray) -> np.ndarray:
    """``A + A^T`` with a zero diagonal (a causal row ``a`` only sees ``b < a``)."""
    s = node_att + np.swapaxes(node_att, -1, -2)
    idx = np.arange(s.shape[-1])
    s[..., idx, idx] = 0.0
    return s


def rollout(att: torch.Tensor, residual: float = 0.5) -> torch.Tensor:
    """Attention rollout (Abnar and Zuidema, 2020), cumulative per layer.

    ``att [L, H, T, T] -> [L, T, T]``: ``R_l = norm(residual*I + (1-residual)*mean_h A_l) @ R_{l-1}``.
    """
    eye = torch.eye(att.shape[-1], dtype=att.dtype, device=att.device)
    out, R = [], eye
    for layer in att:
        a = residual * eye + (1 - residual) * layer.mean(0)
        a = a / a.sum(-1, keepdim=True)
        R = a @ R
        out.append(R)
    return torch.stack(out)


# ---------------------------------------------------------------- forward pass

@dataclass
class Extraction:
    node_att: np.ndarray                  # [L, H, N, N] pooled, unsymmetrised
    rollout: np.ndarray                   # [L, N, N] node-pooled rollout
    hidden: Dict[int, np.ndarray]         # layer -> [N, D] mean-pooled hidden states
    token_att: Optional[np.ndarray] = None   # [L, H, T, T] if keep_tokens


@torch.no_grad()
def extract(model, input_ids: Sequence[int], spans: Sequence[Tuple[int, int]],
            query: str = "mean", key: str = "sum", hidden_layers: Sequence[int] = (),
            keep_tokens: bool = False) -> Extraction:
    """One forward pass -> node-level attention, rollout and pooled hidden states.

    ``hidden_layers`` index ``hidden_states`` (0 = token embeddings, ``L`` = last layer).
    """
    device = next(model.parameters()).device
    ids = torch.tensor([list(input_ids)], device=device)
    out = model(input_ids=ids, output_attentions=True,
                output_hidden_states=bool(hidden_layers), use_cache=False)
    if out.attentions is None or out.attentions[0] is None:
        raise RuntimeError("model returned no attentions; load it with attn_implementation='eager'")
    att = torch.stack([a[0] for a in out.attentions]).float()        # [L, H, T, T]
    node = pool_to_nodes(att, spans, query, key)
    roll = pool_to_nodes(rollout(att), spans, query, key)
    hidden = {}
    if hidden_layers:
        Qm, _ = pooling_matrices(spans, att.shape[-1], "mean", "mean")
        Qm = Qm.to(device)
        for layer in hidden_layers:
            hidden[layer] = (Qm @ out.hidden_states[layer][0].float()).cpu().numpy()
    return Extraction(node_att=node.cpu().numpy(), rollout=roll.cpu().numpy(), hidden=hidden,
                      token_att=att.cpu().numpy() if keep_tokens else None)


def head_scores(ex: Extraction, normalize: str = "renorm") -> Tuple[np.ndarray, np.ndarray]:
    """Row-renormalised scores: ``([L, H, N, N] per head, [L, N, N] rollout)``.

    Only the lower triangle (``j < i``) is meaningful and scored.
    """
    if normalize == "renorm":
        return renorm_rows(ex.node_att), renorm_rows(ex.rollout)
    return np.tril(ex.node_att, -1), np.tril(ex.rollout, -1)

