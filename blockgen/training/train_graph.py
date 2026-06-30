"""Track C: graph latent VAE.

Reuses ``LargePyGGraphGenerator``: a TransformerConv stack encodes the block+port
PyG graph into a graph-level Gaussian latent, and a GRU decoder reconstructs the
structure's serialized token sequence (the *same* unified vocabulary used by the
AR transformer, so coordinates + block types are both recovered). Trained as a
VAE (reconstruction cross-entropy + KL). Sampling draws ``z ~ N(0, I)`` and
autoregressively decodes a token stream, which is decoded back to a ``Structure``.

This is the highest-novelty / most transferable track: the block+port graph maps
directly onto electronics netlists (components + pins) and lego stud connectivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from blockgen.models.large_pyg_graph_generator import (
    LargePyGGraphGenerator,
    LargePyGGraphGeneratorConfig,
)
from blockgen.utils.data import Structure
from blockgen.utils.graph_data import structure_to_pyg_data
from blockgen.utils.serialize import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    BlockVocab,
    structure_to_tokens,
    tokens_to_structure,
)


@dataclass
class GraphTrainConfig:
    max_seq_len: int = 1024
    hidden_dim: int = 192
    latent_dim: int = 128
    encoder_layers: int = 4
    decoder_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    beta: float = 1e-3          # KL weight
    epochs: int = 40
    batch_size: int = 16
    lr: float = 3e-4
    device: str = "cuda"
    log_every: int = 5


def _build_pairs(
    structures: List[Structure], vocab: BlockVocab, max_seq_len: int
) -> List[Tuple[Data, torch.Tensor]]:
    """Build (graph, token-sequence) pairs for the encoder/decoder respectively."""
    pairs: List[Tuple[Data, torch.Tensor]] = []
    for s in structures:
        try:
            toks = structure_to_tokens(s, vocab)
        except ValueError:
            continue
        if not (2 < len(toks) <= max_seq_len):
            continue
        g = structure_to_pyg_data(s)
        if g.num_nodes == 0:
            continue
        pairs.append((g, torch.tensor(toks, dtype=torch.long)))
    return pairs


def _collate(batch: List[Tuple[Data, torch.Tensor]]):
    """Batch graphs via PyG and pad token sequences manually."""
    graphs = [g for g, _ in batch]
    token_list = [t for _, t in batch]
    graph_batch = Batch.from_data_list(graphs)
    max_len = max(t.numel() for t in token_list)
    tokens = torch.full((len(token_list), max_len), PAD_TOKEN, dtype=torch.long)
    for i, t in enumerate(token_list):
        tokens[i, : t.numel()] = t
    return graph_batch, tokens


def train_graph(
    structures: List[Structure],
    vocab: BlockVocab,
    config: Optional[GraphTrainConfig] = None,
):
    """Train the graph VAE. Returns (model, history)."""
    config = config or GraphTrainConfig()
    device = config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"

    pairs = _build_pairs(structures, vocab, config.max_seq_len)
    if not pairs:
        raise ValueError("No graph/token pairs after serialization/filtering.")

    loader = DataLoader(
        pairs, batch_size=config.batch_size, shuffle=True, collate_fn=_collate
    )

    model_cfg = LargePyGGraphGeneratorConfig(
        num_block_tokens=vocab.vocab_size,
        max_block_nodes=config.max_seq_len,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
    )
    model = LargePyGGraphGenerator(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    history = {"loss": [], "recon": [], "kl": []}

    model.train()
    for epoch in range(config.epochs):
        tot, tot_r, tot_k, n = 0.0, 0.0, 0.0, 0
        for batch, tokens in loader:
            batch = batch.to(device)
            tokens = tokens.to(device)
            dec_input = tokens[:, :-1]
            target = tokens[:, 1:]

            out = model(batch, dec_input)
            logits = out["token_logits"]
            recon = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=PAD_TOKEN,
            )
            mu, logvar = out["mu"], out["logvar"]
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + config.beta * kl

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bsz = tokens.size(0)
            tot += loss.item() * bsz
            tot_r += recon.item() * bsz
            tot_k += kl.item() * bsz
            n += bsz

        history["loss"].append(tot / max(n, 1))
        history["recon"].append(tot_r / max(n, 1))
        history["kl"].append(tot_k / max(n, 1))
        if epoch % config.log_every == 0 or epoch == config.epochs - 1:
            print(
                f"[GRAPH] epoch {epoch:3d}  loss {history['loss'][-1]:.4f}  "
                f"recon {history['recon'][-1]:.4f}  kl {history['kl'][-1]:.4f}  (n={len(pairs)})"
            )

    return model, history


@torch.no_grad()
def sample_structures(
    model: LargePyGGraphGenerator,
    vocab: BlockVocab,
    num_samples: int = 16,
    max_len: int = 1024,
    temperature: float = 1.0,
    top_k: Optional[int] = 40,
    device: str = "cuda",
) -> List[Structure]:
    """Sample by drawing z ~ N(0, I) and autoregressively decoding tokens."""
    device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
    model.eval()
    out: List[Structure] = []
    latent_dim = model.config.latent_dim

    for _ in range(num_samples):
        z = torch.randn(1, latent_dim, device=device)
        seq = [BOS_TOKEN]
        for _ in range(max_len - 1):
            inp = torch.tensor([seq], dtype=torch.long, device=device)
            logits = model.decode_tokens(z, inp)[:, -1, :] / max(temperature, 1e-5)
            # Disallow PAD/BOS as emissions.
            logits[:, PAD_TOKEN] = -1e9
            logits[:, BOS_TOKEN] = -1e9
            if top_k is not None and 0 < top_k < logits.size(-1):
                vals, idxs = torch.topk(logits, k=top_k, dim=-1)
                probs = torch.softmax(vals, dim=-1)
                nxt = idxs.gather(-1, torch.multinomial(probs, 1))
            else:
                nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            tok = int(nxt.item())
            if tok == EOS_TOKEN:
                break
            seq.append(tok)
        out.append(tokens_to_structure(seq, vocab))
    return out
