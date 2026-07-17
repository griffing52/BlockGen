"""Track A: autoregressive token transformer (LegoGPT-style).

Serializes each structure to a variable-length token sequence and trains
``VoxelTransformerAR`` with next-token prediction. Variable structure size is
handled entirely by the EOS terminator — there is no fixed grid and no explicit
size head. Sampling draws a fresh token stream and decodes it back to a
``Structure``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from blockgen.models.voxel_transformer_ar import VoxelTransformerAR
from blockgen.utils.data import Structure
from blockgen.utils.serialize import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    BlockVocab,
    structure_to_tokens,
    tokens_to_structure,
)


@dataclass
class ARTrainConfig:
    max_seq_len: int = 1024
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    epochs: int = 30
    batch_size: int = 16
    lr: float = 3e-4
    device: str = "cuda"
    log_every: int = 5
    # Throughput knobs. `amp` is bf16 autocast (no GradScaler needed); it changes
    # numerics slightly, so flip it off to reproduce pre-2026-07-15 runs exactly.
    # `num_workers` is usually a no-op here: sequences are pre-tokenized in memory
    # and _collate is trivial, so worker IPC can cost more than it saves.
    amp: bool = True
    num_workers: int = 0
    pin_memory: bool = True


class _TokenDataset(Dataset):
    def __init__(self, sequences: List[List[int]]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.sequences[idx], dtype=torch.long)


def _collate(batch: List[torch.Tensor]):
    """Pad a batch to its longest sequence. Returns (input_ids, targets, pad_mask)."""
    lengths = [len(b) for b in batch]
    max_len = max(lengths)
    bsz = len(batch)
    padded = torch.full((bsz, max_len), PAD_TOKEN, dtype=torch.long)
    for i, b in enumerate(batch):
        padded[i, : len(b)] = b
    # Next-token objective.
    input_ids = padded[:, :-1]
    targets = padded[:, 1:]
    pad_mask = input_ids == PAD_TOKEN  # True where padded (ignored by attention)
    return input_ids, targets, pad_mask


def build_sequences(
    structures: List[Structure], vocab: BlockVocab, max_seq_len: int
) -> List[List[int]]:
    seqs: List[List[int]] = []
    for s in structures:
        try:
            toks = structure_to_tokens(s, vocab)
        except ValueError:
            continue
        if 2 < len(toks) <= max_seq_len:
            seqs.append(toks)
    return seqs


def train_ar(
    structures: List[Structure],
    vocab: BlockVocab,
    config: Optional[ARTrainConfig] = None,
):
    """Train the AR transformer. Returns (model, history)."""
    config = config or ARTrainConfig()
    device = config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"

    sequences = build_sequences(structures, vocab, config.max_seq_len)
    if not sequences:
        raise ValueError("No training sequences after serialization/filtering.")

    dataset = _TokenDataset(sequences)
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, collate_fn=_collate,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device != "cpu",
        persistent_workers=config.num_workers > 0,
    )

    model = VoxelTransformerAR(
        vocab_size=vocab.vocab_size,
        max_seq_len=config.max_seq_len,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    history = {"loss": []}

    use_amp = config.amp and device != "cpu" and torch.cuda.is_bf16_supported()

    model.train()
    for epoch in range(config.epochs):
        # Accumulate on-device: a per-step .item() would sync the GPU every
        # iteration and serialize the pipeline. One sync per epoch instead.
        total = torch.zeros((), device=device, dtype=torch.float32)
        n_tok = torch.zeros((), device=device, dtype=torch.float32)
        for input_ids, targets, pad_mask in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            pad_mask = pad_mask.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(input_ids, pad_mask=pad_mask)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=PAD_TOKEN,
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            n = (targets != PAD_TOKEN).sum()
            total += loss.detach().float() * n
            n_tok += n

        avg = float(total / n_tok.clamp(min=1))
        history["loss"].append(avg)
        if epoch % config.log_every == 0 or epoch == config.epochs - 1:
            print(f"[AR] epoch {epoch:3d}  loss {avg:.4f}  (n_seq={len(sequences)})")

    return model, history


@torch.no_grad()
def sample_structures(
    model: VoxelTransformerAR,
    vocab: BlockVocab,
    num_samples: int = 16,
    temperature: float = 1.0,
    top_k: Optional[int] = 40,
) -> List[Structure]:
    """Sample structures by autoregressive generation + token decoding."""
    model.eval()
    out: List[Structure] = []
    for _ in range(num_samples):
        tokens = model.generate(
            bos_token_id=BOS_TOKEN,
            eos_token_id=EOS_TOKEN,
            max_new_tokens=model.max_seq_len - 1,
            temperature=temperature,
            top_k=top_k,
        )
        out.append(tokens_to_structure(tokens[0].tolist(), vocab))
    return out
