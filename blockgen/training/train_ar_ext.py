"""Extended AR training/sampling: generic token streams, cluster (BPE), and
category-conditioned generation. Reuses ``VoxelTransformerAR`` and the collate/
dataset from ``train_ar`` so the model itself is unchanged.

Three entry points beyond flat per-voxel AR:
  * ``train_from_sequences`` — train on any list of token id sequences.
  * cluster: tokenize with a ``ClusterVocab`` and decode pieces back to structures.
  * conditioned: prepend a category token so one model generates any class.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from blockgen.models.voxel_transformer_ar import VoxelTransformerAR
from blockgen.training.train_ar import ARTrainConfig, _collate, _TokenDataset
from blockgen.utils.data import Structure
from blockgen.utils.serialize import (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, BlockVocab,
                                       structure_to_tokens, tokens_to_structure)


def train_from_sequences(sequences: List[List[int]], vocab_size: int,
                         config: ARTrainConfig,
                         pe: Optional[str] = None,
                         semantic_embedding=None) -> Tuple[VoxelTransformerAR, dict]:
    """Train the AR transformer on pre-built token sequences. Returns (model, history).

    ``pe=None`` uses the original ``VoxelTransformerAR`` (learned absolute PE);
    any other value selects that positional-encoding scheme in the PE-pluggable
    ``VoxelTransformerAR2`` (see its module doc: sin / rope / alibi / phase4 / none).

    ``semantic_embedding`` (idea #7) is an optional ``SemanticTokenEmbedding`` that
    replaces the learned block-token table with a projection of frozen CLIP/SigLIP
    text embeddings; only valid with a PE (the AR2 backbone).
    """
    if semantic_embedding is not None and pe is None:
        raise ValueError("semantic_embedding requires an AR2 backbone; pass pe != None")
    device = config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
    if not sequences:
        raise ValueError("No training sequences.")
    loader = DataLoader(_TokenDataset(sequences), batch_size=config.batch_size,
                        shuffle=True, collate_fn=_collate,
                        num_workers=config.num_workers,
                        pin_memory=config.pin_memory and device != "cpu",
                        persistent_workers=config.num_workers > 0)
    kwargs = dict(
        vocab_size=vocab_size, max_seq_len=config.max_seq_len, d_model=config.d_model,
        nhead=config.nhead, num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward, dropout=config.dropout)
    if pe is None:
        model = VoxelTransformerAR(**kwargs).to(device)
    else:
        from blockgen.models.voxel_transformer_ar2 import VoxelTransformerAR2
        if semantic_embedding is not None:
            kwargs["semantic_embedding"] = semantic_embedding
        model = VoxelTransformerAR2(pe=pe, **kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    history = {"loss": []}
    use_amp = config.amp and device != "cpu" and torch.cuda.is_bf16_supported()
    model.train()
    for epoch in range(config.epochs):
        # On-device accumulation: a per-step .item() syncs the GPU every iteration.
        total = torch.zeros((), device=device, dtype=torch.float32)
        n_tok = torch.zeros((), device=device, dtype=torch.float32)
        for input_ids, targets, pad_mask in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            pad_mask = pad_mask.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(input_ids, pad_mask=pad_mask)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       targets.reshape(-1), ignore_index=PAD_TOKEN)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            n = (targets != PAD_TOKEN).sum()
            total += loss.detach().float() * n; n_tok += n
        avg = float(total / n_tok.clamp(min=1))
        history["loss"].append(avg)
        if epoch % config.log_every == 0 or epoch == config.epochs - 1:
            print(f"[AR-ext] epoch {epoch:3d}  loss {avg:.4f}  (n_seq={len(sequences)})", flush=True)
    return model, history


@torch.no_grad()
def generate_from_prefix(model: VoxelTransformerAR, prefix: Sequence[int], eos_token_id: int,
                         max_new_tokens: int, temperature: float = 1.0,
                         top_k: Optional[int] = 40) -> List[int]:
    """Autoregressive generation seeded with an arbitrary prefix (e.g. [BOS, CAT])."""
    model.eval()
    device = next(model.parameters()).device
    tokens = torch.tensor([list(prefix)], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model.forward(tokens)
        nl = logits[:, -1, :] / max(temperature, 1e-5)
        if top_k and 0 < top_k < nl.shape[-1]:
            vals, idxs = torch.topk(nl, k=top_k, dim=-1)
            pick = torch.multinomial(torch.softmax(vals, dim=-1), 1)
            nxt = idxs.gather(-1, pick)
        else:
            nxt = torch.multinomial(torch.softmax(nl, dim=-1), 1)
        tokens = torch.cat([tokens, nxt], dim=1)
        if int(nxt.item()) == eos_token_id or tokens.shape[1] >= model.max_seq_len:
            break
    return tokens[0].tolist()


# --- cluster (BPE) sampling ------------------------------------------------- #
@torch.no_grad()
def sample_cluster_structures(model, cv, num_samples: int = 16, temperature: float = 1.0,
                              top_k: Optional[int] = 40) -> List[Structure]:
    from blockgen.tokenizers.cluster_bpe import cluster_tokens_to_structure
    out = []
    for _ in range(num_samples):
        toks = generate_from_prefix(model, [BOS_TOKEN], EOS_TOKEN,
                                    max_new_tokens=model.max_seq_len - 1,
                                    temperature=temperature, top_k=top_k)
        out.append(cluster_tokens_to_structure(toks, cv))
    return out


# --- category-conditioned AR ------------------------------------------------ #
def cat_token_id(vocab: BlockVocab, cat_index: int) -> int:
    """Category token ids live just above the block vocab."""
    return vocab.vocab_size + cat_index


def build_conditioned_sequences(structs: Sequence[Structure], labels: Sequence[str],
                                vocab: BlockVocab, categories: Sequence[str],
                                max_seq_len: int) -> Tuple[List[List[int]], List[str]]:
    """Build [BOS, CAT, (X,Y,Z,BLOCK)*, EOS] sequences. Returns (sequences, kept_labels)."""
    cat_idx = {c: i for i, c in enumerate(categories)}
    seqs: List[List[int]] = []
    kept: List[str] = []
    for s, lab in zip(structs, labels):
        if lab not in cat_idx:
            continue
        try:
            body = structure_to_tokens(s, vocab)  # [BOS, ..., EOS]
        except ValueError:
            continue
        seq = [BOS_TOKEN, cat_token_id(vocab, cat_idx[lab])] + body[1:]
        if 3 < len(seq) <= max_seq_len:
            seqs.append(seq)
            kept.append(lab)
    return seqs, kept


@torch.no_grad()
def sample_conditioned_structures(model, vocab: BlockVocab, cat_index: int,
                                  num_samples: int = 16, temperature: float = 1.0,
                                  top_k: Optional[int] = 40) -> List[Structure]:
    out = []
    prefix = [BOS_TOKEN, cat_token_id(vocab, cat_index)]
    for _ in range(num_samples):
        toks = generate_from_prefix(model, prefix, EOS_TOKEN,
                                    max_new_tokens=model.max_seq_len - 1,
                                    temperature=temperature, top_k=top_k)
        body = [BOS_TOKEN] + toks[2:]  # strip [BOS, CAT], restore a BOS for the decoder
        out.append(tokens_to_structure(body, vocab))
    return out
