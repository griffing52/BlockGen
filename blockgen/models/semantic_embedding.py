"""Idea #7: semantic node-embedding prior (see ``ideas.md``).

The default ``nn.Embedding(vocab_size, d_model)`` learns every block token's vector
from scratch, from co-occurrence alone — so nothing tells the model that ``oak_planks``
and ``spruce_planks`` behave alike, or that a furnace is not a wall material. That is
visible in our runs as block-agreement 0.03–0.13 and palette collapse (notes.md §16,
research.md B).

This module replaces the *block* rows of the input embedding with a projection of a
**frozen semantic vector** — the CLIP/SigLIP text embedding of the block's name
("red wool", "oak wood planks"). Semantically similar blocks start near each other, so
the model inherits a material/placement prior instead of rediscovering it. This is the
text-attributed-graph idea (OFA/ZeroG/UniGraph) applied to a voxel token table, and it
is the cheapest lever in ideas.md.

**Scope: input side only.** The ``lm_head`` stays an ordinary indexed softmax, so
generation is unchanged and this is *not* a zero-shot-new-block mechanism — it is a
placement prior (which is the well-supported use; see ideas.md "zero-shot" section).
Coordinate and special tokens keep an ordinary learned table; only the contiguous
block tail is semantic.

Layout assumption (matches ``serialize.BlockVocab``): ``[PAD,BOS,EOS] + coords(max_dim)
+ blocks``, i.e. block tokens occupy ``[block_offset, vocab_size)`` contiguously.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from blockgen.tokenizers.standard_vocab import STANDARD_VOCAB
from blockgen.utils.data import _token_for
from blockgen.utils.serialize import BlockVocab


def block_text(block_id: int, block_data: int) -> str:
    """A natural-language phrase for a block, e.g. ``(35, 14) -> 'red wool'``.

    Uses the classic display name from ``STANDARD_VOCAB`` (``"Red Wool(minecraft:
    wool)"`` -> ``"red wool"``), lower-cased so it lands in CLIP's caption
    distribution. Falls back to the namespaced id for blocks with no display name.
    """
    raw = STANDARD_VOCAB.get(_token_for(block_id, block_data)) or \
        STANDARD_VOCAB.get(str(block_id))
    if raw:
        m = re.match(r"^(.*)\(.*\)$", raw)
        name = (m.group(1) if m else raw).strip()
        if name and name.lower() != "none":
            return name.lower()
    return f"minecraft block {block_id}"


def block_texts(vocab: BlockVocab) -> List[str]:
    """One phrase per block index, in block-index order."""
    return [block_text(bid, data) for bid, data in vocab.block_index_to_pair]


@torch.no_grad()
def _clip_text_embeddings(texts: Sequence[str], device: str,
                          model_name: str = "openai/clip-vit-base-patch32") -> np.ndarray:
    """Pooled CLIP text features per phrase — the same encoder embed_conditions uses."""
    from transformers import CLIPModel, CLIPProcessor
    proc = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    out = []
    for start in range(0, len(texts), 256):
        batch = list(texts[start:start + 256])
        inputs = proc(text=batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=77).to(device)
        out.append(model.get_text_features(**inputs).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def build_semantic_matrix(
    vocab: BlockVocab,
    *,
    encoder: str = "clip",
    device: str = "cpu",
    cache_path: Optional[str] = None,
    encode_fn: Optional[Callable[[Sequence[str]], np.ndarray]] = None,
) -> torch.Tensor:
    """Frozen ``[num_blocks, sem_dim]`` semantic matrix, L2-normalized, cached.

    ``encode_fn`` overrides the built-in encoder (used by tests to avoid a CLIP
    download); otherwise ``encoder`` selects the text tower. The cache is keyed only
    by path — pass a name that encodes the vocab + encoder if you reuse it.
    """
    if cache_path and Path(cache_path).exists():
        return torch.from_numpy(np.load(cache_path)).float()

    texts = block_texts(vocab)
    if encode_fn is not None:
        mat = np.asarray(encode_fn(texts), dtype=np.float32)
    elif encoder == "clip":
        mat = _clip_text_embeddings(texts, device)
    else:
        raise ValueError(f"unknown encoder {encoder!r} (have: clip, or pass encode_fn)")

    t = torch.from_numpy(mat).float()
    t = F.normalize(t, dim=-1)                     # unit vectors: stable projection init
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, t.numpy())
    return t


class SemanticTokenEmbedding(nn.Module):
    """Input embedding whose block rows are a projection of frozen semantic vectors.

    ``E[t] = learned[t]``                          for specials + coordinate tokens
    ``E[t] = proj(semantic[t - block_offset])``    for block tokens

    The block rows are rebuilt each forward (vocab is small, so this is free) and stay
    differentiable through ``proj`` while ``semantic`` is frozen. Same call/attribute
    contract as ``nn.Embedding``/``FactoredPieceEmbedding`` (``embedding_dim``,
    ``weight``, ``forward``), so it drops into ``VoxelTransformerAR2`` unchanged.
    """

    def __init__(self, *, vocab_size: int, d_model: int, block_offset: int,
                 semantic_matrix: torch.Tensor, freeze_semantic: bool = True,
                 hidden: int = 0) -> None:
        super().__init__()
        num_blocks = semantic_matrix.shape[0]
        if block_offset + num_blocks != vocab_size:
            raise ValueError(
                f"block tokens must be contiguous at the tail: block_offset "
                f"{block_offset} + num_blocks {num_blocks} != vocab_size {vocab_size}")
        self.vocab_size = vocab_size
        self.embedding_dim = d_model
        self.block_offset = block_offset

        # Specials + coordinate tokens: ordinary learned table.
        self.head = nn.Embedding(block_offset, d_model)

        # Frozen semantic vectors. A parameter (not buffer) when unfrozen so it trains.
        if freeze_semantic:
            self.register_buffer("semantic", semantic_matrix.clone(), persistent=True)
        else:
            self.semantic = nn.Parameter(semantic_matrix.clone())

        sem_dim = semantic_matrix.shape[1]
        if hidden and hidden > 0:
            self.proj: nn.Module = nn.Sequential(
                nn.Linear(sem_dim, hidden), nn.GELU(), nn.Linear(hidden, d_model))
        else:
            self.proj = nn.Linear(sem_dim, d_model)

    def block_weight(self) -> torch.Tensor:
        return self.proj(self.semantic)

    def weight_matrix(self) -> torch.Tensor:
        return torch.cat([self.head.weight, self.block_weight()], dim=0)

    @property
    def weight(self) -> torch.Tensor:
        return self.weight_matrix()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.embedding(tokens, self.weight_matrix())


def build_semantic_embedding(
    vocab: BlockVocab, d_model: int, *, encoder: str = "clip", device: str = "cpu",
    cache_path: Optional[str] = None, freeze_semantic: bool = True, hidden: int = 0,
    encode_fn: Optional[Callable[[Sequence[str]], np.ndarray]] = None,
) -> SemanticTokenEmbedding:
    """Convenience: build the semantic matrix and the embedding module together."""
    mat = build_semantic_matrix(vocab, encoder=encoder, device=device,
                                cache_path=cache_path, encode_fn=encode_fn)
    return SemanticTokenEmbedding(
        vocab_size=vocab.vocab_size, d_model=d_model, block_offset=vocab.block_offset,
        semantic_matrix=mat, freeze_semantic=freeze_semantic, hidden=hidden)
