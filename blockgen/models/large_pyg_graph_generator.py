from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn
from torch_geometric.nn import TransformerConv, global_max_pool, global_mean_pool


@dataclass
class LargePyGGraphGeneratorConfig:
    num_block_tokens: int
    max_block_nodes: int = 2048
    node_feature_dim: int = 12
    hidden_dim: int = 512
    latent_dim: int = 256
    encoder_layers: int = 8
    decoder_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    edge_type_vocab: int = 2
    node_type_vocab: int = 2
    block_data_vocab: int = 64
    port_direction_vocab: int = 8


class LargePyGGraphGenerator(nn.Module):
    """Scalable PyG model for graph-conditioned autoregressive generation.

    Architecture:
    1) Graph encoder (TransformerConv stack) over PyG graph structure.
    2) VAE latent bottleneck at graph level.
    3) Autoregressive token decoder (GRU) for block-token sequences.
    4) Graph-size head for sampling graph sizes from latent space.
    """

    def __init__(self, config: LargePyGGraphGeneratorConfig) -> None:
        super().__init__()
        self.config = config

        embed_dim = max(32, config.hidden_dim // 4)

        self.node_type_embedding = nn.Embedding(config.node_type_vocab, embed_dim)
        self.block_data_embedding = nn.Embedding(config.block_data_vocab, embed_dim)
        self.port_direction_embedding = nn.Embedding(config.port_direction_vocab, embed_dim)
        self.edge_type_embedding = nn.Embedding(config.edge_type_vocab, embed_dim)

        input_dim = config.node_feature_dim + embed_dim * 3
        self.input_projection = nn.Linear(input_dim, config.hidden_dim)

        encoder = []
        norms = []
        for _ in range(config.encoder_layers):
            encoder.append(
                TransformerConv(
                    in_channels=config.hidden_dim,
                    out_channels=config.hidden_dim // config.num_heads,
                    heads=config.num_heads,
                    dropout=config.dropout,
                    edge_dim=embed_dim,
                    beta=True,
                )
            )
            norms.append(nn.LayerNorm(config.hidden_dim))

        self.encoder_layers = nn.ModuleList(encoder)
        self.encoder_norms = nn.ModuleList(norms)

        pooled_dim = config.hidden_dim * 2
        self.mu_head = nn.Linear(pooled_dim, config.latent_dim)
        self.logvar_head = nn.Linear(pooled_dim, config.latent_dim)

        self.size_head = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.max_block_nodes + 1),
        )

        self.token_embedding = nn.Embedding(config.num_block_tokens, config.hidden_dim)
        self.decoder_gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.decoder_layers,
            batch_first=True,
            dropout=config.dropout if config.decoder_layers > 1 else 0.0,
        )
        self.latent_to_decoder = nn.Linear(config.latent_dim, config.hidden_dim * config.decoder_layers)
        self.decoder_norm = nn.LayerNorm(config.hidden_dim)
        self.token_head = nn.Linear(config.hidden_dim, config.num_block_tokens)

    def _safe_idx(self, values: torch.Tensor, limit: int, shift: int = 0) -> torch.Tensor:
        values = values.long() + shift
        values = torch.clamp(values, min=0)
        if limit > 0:
            values = torch.remainder(values, limit)
        return values

    def _node_features(self, data) -> torch.Tensor:
        node_type = self._safe_idx(data.node_type, self.config.node_type_vocab)
        block_data = self._safe_idx(data.block_data, self.config.block_data_vocab)
        port_direction = self._safe_idx(data.port_direction, self.config.port_direction_vocab, shift=1)

        feats = [
            data.x.float(),
            self.node_type_embedding(node_type),
            self.block_data_embedding(block_data),
            self.port_direction_embedding(port_direction),
        ]
        return self.input_projection(torch.cat(feats, dim=-1))

    def encode_graph(self, data) -> Dict[str, torch.Tensor]:
        h = self._node_features(data)

        edge_type = self._safe_idx(data.edge_type, self.config.edge_type_vocab)
        edge_attr = self.edge_type_embedding(edge_type)

        for conv, norm in zip(self.encoder_layers, self.encoder_norms):
            residual = h
            h = conv(h, data.edge_index, edge_attr=edge_attr)
            h = norm(h + residual)
            h = torch.gelu(h)

        pooled = torch.cat(
            [
                global_mean_pool(h, data.batch),
                global_max_pool(h, data.batch),
            ],
            dim=-1,
        )

        mu = self.mu_head(pooled)
        logvar = self.logvar_head(pooled)
        return {"mu": mu, "logvar": logvar}

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_tokens(self, z: torch.Tensor, decoder_input_tokens: torch.Tensor) -> torch.Tensor:
        token_emb = self.token_embedding(decoder_input_tokens)

        init_state = self.latent_to_decoder(z)
        init_state = init_state.view(
            z.shape[0], self.config.decoder_layers, self.config.hidden_dim
        ).permute(1, 0, 2).contiguous()

        out, _ = self.decoder_gru(token_emb, init_state)
        out = self.decoder_norm(out)
        return self.token_head(out)

    def forward(self, data, decoder_input_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encode_graph(data)
        mu = encoded["mu"]
        logvar = encoded["logvar"]
        z = self.reparameterize(mu, logvar)

        token_logits = self.decode_tokens(z, decoder_input_tokens)
        size_logits = self.size_head(z)

        return {
            "token_logits": token_logits,
            "size_logits": size_logits,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    @torch.no_grad()
    def sample_num_blocks(
        self,
        z: torch.Tensor,
        *,
        temperature: float = 1.0,
        min_blocks: int = 1,
    ) -> torch.Tensor:
        logits = self.size_head(z) / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return torch.clamp(sampled, min=min_blocks)

    @torch.no_grad()
    def sample_block_tokens(
        self,
        z: torch.Tensor,
        *,
        bos_token_id: int,
        num_tokens: int,
        valid_token_ids: Sequence[int],
        temperature: float = 1.0,
        top_k: Optional[int] = 64,
    ) -> List[int]:
        if z.ndim == 1:
            z = z.unsqueeze(0)

        seq: List[int] = [int(bos_token_id)]
        valid_ids = torch.tensor(list(valid_token_ids), device=z.device, dtype=torch.long)

        for _ in range(num_tokens):
            inp = torch.tensor([seq], dtype=torch.long, device=z.device)
            logits = self.decode_tokens(z, inp)[:, -1, :] / max(temperature, 1e-6)

            restricted = torch.full_like(logits, -1e9)
            restricted[:, valid_ids] = logits[:, valid_ids]

            if top_k is not None and top_k > 0 and top_k < valid_ids.shape[0]:
                vals, idxs = torch.topk(restricted, k=top_k, dim=-1)
                probs = torch.softmax(vals, dim=-1)
                pick = torch.multinomial(probs, num_samples=1)
                next_tok = idxs.gather(-1, pick)
            else:
                probs = torch.softmax(restricted, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)

            seq.append(int(next_tok.item()))

        return seq[1:]
