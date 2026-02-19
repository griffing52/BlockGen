from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class VoxelPortGNN(nn.Module):
    """Starter graph model for voxel-structure generation experiments.

    Expects the graph schema from `structure_to_pyg_data` in
    `blockgen.utils.graph_data`:
    - block nodes at indices 0, 7, 14, ...
    - 6 port children per block node
    - `node_type`, `block_id`, `block_data`, `port_direction`, and `edge_type`
    """

    def __init__(
        self,
        *,
        num_block_id_embeddings: int = 4096,
        num_block_data_embeddings: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_output_block_ids: int = 4096,
    ) -> None:
        super().__init__()

        self.block_id_embedding = nn.Embedding(num_block_id_embeddings, hidden_dim)
        self.block_data_embedding = nn.Embedding(num_block_data_embeddings, hidden_dim)
        self.node_type_embedding = nn.Embedding(2, hidden_dim)
        self.port_direction_embedding = nn.Embedding(7, hidden_dim)  # 0..6 (0 used for non-port)

        self.input_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 12, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        conv_layers = []
        for _ in range(num_layers):
            conv_layers.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs = nn.ModuleList(conv_layers)

        self.block_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_output_block_ids),
        )

        self.port_pair_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _safe_embedding_indices(self, values: torch.Tensor, limit: int) -> torch.Tensor:
        values = values.long()
        values = torch.clamp(values, min=0)
        if limit > 0:
            values = torch.remainder(values, limit)
        return values

    def encode(self, data: Data) -> torch.Tensor:
        block_id = self._safe_embedding_indices(data.block_id, self.block_id_embedding.num_embeddings)
        block_data = self._safe_embedding_indices(data.block_data, self.block_data_embedding.num_embeddings)
        node_type = self._safe_embedding_indices(data.node_type, self.node_type_embedding.num_embeddings)

        # port_direction in data is -1 for block nodes. Shift by +1 so block nodes map to 0.
        port_direction = self._safe_embedding_indices(
            data.port_direction + 1,
            self.port_direction_embedding.num_embeddings,
        )

        features = [
            data.x.float(),
            self.block_id_embedding(block_id),
            self.block_data_embedding(block_data),
            self.node_type_embedding(node_type),
            self.port_direction_embedding(port_direction),
        ]
        h = self.input_mlp(torch.cat(features, dim=-1))

        for conv in self.convs:
            h = conv(h, data.edge_index)
            h = torch.relu(h)

        return h

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """Returns predictions useful for early generative experimentation.

        Outputs:
        - block_logits: logits over block IDs for block nodes only
        - port_edge_logits: logits for existing port-to-port edges in `edge_index`
        - latent: node embeddings after message passing
        """

        latent = self.encode(data)

        block_mask = data.node_type == 0
        block_logits = self.block_decoder(latent[block_mask])

        port_edge_mask = data.edge_type == 1
        port_edges = data.edge_index[:, port_edge_mask]
        if port_edges.numel() == 0:
            port_edge_logits = latent.new_zeros((0,))
        else:
            src = latent[port_edges[0]]
            dst = latent[port_edges[1]]
            port_edge_logits = self.port_pair_scorer(torch.cat([src, dst], dim=-1)).squeeze(-1)

        return {
            "block_logits": block_logits,
            "port_edge_logits": port_edge_logits,
            "latent": latent,
        }
