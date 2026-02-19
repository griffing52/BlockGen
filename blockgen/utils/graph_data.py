from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from blockgen.utils.data import Structure
from blockgen.utils.data_loader import load_schematic


# Order is important and used in node features.
PORT_DIRECTIONS: Tuple[Tuple[int, int, int], ...] = (
    (1, 0, 0),   # +X
    (-1, 0, 0),  # -X
    (0, 1, 0),   # +Y
    (0, -1, 0),  # -Y
    (0, 0, 1),   # +Z
    (0, 0, -1),  # -Z
)

OPPOSITE_PORT: Dict[int, int] = {
    0: 1,
    1: 0,
    2: 3,
    3: 2,
    4: 5,
    5: 4,
}

EDGE_BLOCK_TO_PORT = 0
EDGE_PORT_TO_PORT = 1

NODE_BLOCK = 0
NODE_PORT = 1


@dataclass(frozen=True)
class GraphBuildConfig:
    include_air: bool = False
    crop_non_air: bool = True
    max_dim: Optional[int] = None


def _preprocess_structure(structure: Structure, config: GraphBuildConfig) -> Structure:
    processed = structure
    if config.crop_non_air:
        processed = processed.crop_to_non_air()
    if config.max_dim is not None:
        processed = processed.downsample(max_dim=config.max_dim)
    return processed


def structure_to_pyg_data(
    structure: Structure,
    *,
    include_air: bool = False,
) -> Data:
    """Convert a Structure into a torch_geometric graph.

    Graph schema:
    - Each voxel becomes one block node.
    - Each block node owns six port child nodes (one per axis direction).
    - Block nodes are only connected to their own ports.
    - Inter-voxel connectivity is represented by port-to-port edges only.
    """

    occupancy = np.ones(structure.shape, dtype=bool) if include_air else structure.occupied_mask
    coords = np.argwhere(occupancy)

    if coords.shape[0] == 0:
        empty = torch.empty((0, 12), dtype=torch.float32)
        return Data(
            x=empty,
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_type=torch.empty((0,), dtype=torch.long),
            node_type=torch.empty((0,), dtype=torch.long),
            parent_block=torch.empty((0,), dtype=torch.long),
            port_direction=torch.empty((0,), dtype=torch.long),
            block_id=torch.empty((0,), dtype=torch.long),
            block_data=torch.empty((0,), dtype=torch.long),
            pos=torch.empty((0, 3), dtype=torch.long),
            block_node_index=torch.empty((0,), dtype=torch.long),
            num_block_nodes=0,
            source_path=structure.source_path or "",
            shape=torch.tensor(structure.shape, dtype=torch.long),
        )

    num_blocks = coords.shape[0]
    num_nodes = num_blocks * 7

    # Block index by coordinate for O(1) neighbor lookups.
    block_index_by_coord: Dict[Tuple[int, int, int], int] = {}
    for block_idx, (x, y, z) in enumerate(coords.tolist()):
        block_index_by_coord[(x, y, z)] = block_idx

    node_type = np.empty((num_nodes,), dtype=np.int64)
    parent_block = np.full((num_nodes,), -1, dtype=np.int64)
    port_direction = np.full((num_nodes,), -1, dtype=np.int64)
    block_id = np.empty((num_nodes,), dtype=np.int64)
    block_data = np.empty((num_nodes,), dtype=np.int64)
    pos = np.empty((num_nodes, 3), dtype=np.int64)
    x_features = np.zeros((num_nodes, 12), dtype=np.float32)

    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_type: List[int] = []

    max_x, max_y, max_z = structure.shape
    norm_scale = np.array(
        [max(1, max_x - 1), max(1, max_y - 1), max(1, max_z - 1)],
        dtype=np.float32,
    )

    for block_idx, (x, y, z) in enumerate(coords.tolist()):
        main_idx = block_idx * 7
        bid = int(structure.block_ids[x, y, z])
        bdata = int(structure.block_data[x, y, z])

        node_type[main_idx] = NODE_BLOCK
        block_id[main_idx] = bid
        block_data[main_idx] = bdata
        pos[main_idx] = (x, y, z)

        x_features[main_idx, 0] = 1.0  # is_block
        x_features[main_idx, 1] = float(bid)
        x_features[main_idx, 2] = float(bdata)
        x_features[main_idx, 3:6] = np.array((x, y, z), dtype=np.float32) / norm_scale

        for direction_idx, (dx, dy, dz) in enumerate(PORT_DIRECTIONS):
            port_idx = main_idx + 1 + direction_idx

            node_type[port_idx] = NODE_PORT
            parent_block[port_idx] = main_idx
            port_direction[port_idx] = direction_idx
            block_id[port_idx] = bid
            block_data[port_idx] = bdata
            pos[port_idx] = (x, y, z)

            x_features[port_idx, 1] = float(bid)
            x_features[port_idx, 2] = float(bdata)
            x_features[port_idx, 3:6] = np.array((x, y, z), dtype=np.float32) / norm_scale
            x_features[port_idx, 6 + direction_idx] = 1.0  # one-hot port direction

            # parent<->child edges
            edge_src.extend((main_idx, port_idx))
            edge_dst.extend((port_idx, main_idx))
            edge_type.extend((EDGE_BLOCK_TO_PORT, EDGE_BLOCK_TO_PORT))

            # Connect this port to the opposite port of the neighboring voxel.
            neighbor = (x + dx, y + dy, z + dz)
            neighbor_block_idx = block_index_by_coord.get(neighbor)
            if neighbor_block_idx is None:
                continue

            neighbor_main = neighbor_block_idx * 7
            neighbor_port = neighbor_main + 1 + OPPOSITE_PORT[direction_idx]
            edge_src.append(port_idx)
            edge_dst.append(neighbor_port)
            edge_type.append(EDGE_PORT_TO_PORT)

    data = Data(
        x=torch.from_numpy(x_features),
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
        edge_type=torch.tensor(edge_type, dtype=torch.long),
        node_type=torch.from_numpy(node_type),
        parent_block=torch.from_numpy(parent_block),
        port_direction=torch.from_numpy(port_direction),
        block_id=torch.from_numpy(block_id),
        block_data=torch.from_numpy(block_data),
        pos=torch.from_numpy(pos),
        block_node_index=torch.arange(0, num_nodes, 7, dtype=torch.long),
        num_block_nodes=int(num_blocks),
        source_path=structure.source_path or "",
        shape=torch.tensor(structure.shape, dtype=torch.long),
    )
    return data


class SchematicGraphDataset(Dataset):
    """Simple on-demand dataset for schematic -> PyG graph conversion."""

    def __init__(
        self,
        schematic_paths: Sequence[str],
        *,
        include_air: bool = False,
        crop_non_air: bool = True,
        max_dim: Optional[int] = None,
    ) -> None:
        self.schematic_paths = [str(Path(path)) for path in schematic_paths]
        self.config = GraphBuildConfig(
            include_air=include_air,
            crop_non_air=crop_non_air,
            max_dim=max_dim,
        )

    def __len__(self) -> int:
        return len(self.schematic_paths)

    def __getitem__(self, index: int) -> Data:
        path = self.schematic_paths[index]
        schematic = load_schematic(path)
        structure = Structure.from_schematic(schematic, source_path=path)
        structure = _preprocess_structure(structure, self.config)
        return structure_to_pyg_data(structure, include_air=self.config.include_air)


def list_schematic_files(root: str | Path, *, suffix: str = ".schematic") -> List[str]:
    root_path = Path(root)
    return [str(path) for path in sorted(root_path.glob(f"*{suffix}"))]


def dataset_from_directory(
    root: str | Path,
    *,
    include_air: bool = False,
    crop_non_air: bool = True,
    max_dim: Optional[int] = None,
) -> SchematicGraphDataset:
    paths = list_schematic_files(root)
    return SchematicGraphDataset(
        paths,
        include_air=include_air,
        crop_non_air=crop_non_air,
        max_dim=max_dim,
    )


def dataset_from_list_file(
    list_path: str | Path,
    *,
    include_air: bool = False,
    crop_non_air: bool = True,
    max_dim: Optional[int] = None,
) -> SchematicGraphDataset:
    path = Path(list_path)
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return SchematicGraphDataset(
        lines,
        include_air=include_air,
        crop_non_air=crop_non_air,
        max_dim=max_dim,
    )
