from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from blockgen.tokenizers.standard_vocab import STANDARD_VOCAB
from blockgen.utils.data_loader import load_schematic


RGBA = Tuple[float, float, float, float]


BASE_TEXTURE_COLORS: Dict[str, RGBA] = {
    "minecraft:air": (0.0, 0.0, 0.0, 0.0),
    "minecraft:stone": (0.49, 0.49, 0.49, 1.0),
    "minecraft:grass": (0.38, 0.62, 0.25, 1.0),
    "minecraft:dirt": (0.53, 0.35, 0.20, 1.0),
    "minecraft:cobblestone": (0.45, 0.45, 0.45, 1.0),
    "minecraft:planks": (0.63, 0.51, 0.31, 1.0),
    "minecraft:bedrock": (0.25, 0.25, 0.25, 1.0),
    "minecraft:sand": (0.87, 0.83, 0.60, 1.0),
    "minecraft:gravel": (0.53, 0.53, 0.53, 1.0),
    "minecraft:log": (0.42, 0.33, 0.20, 1.0),
    "minecraft:log2": (0.46, 0.35, 0.19, 1.0),
    "minecraft:leaves": (0.25, 0.50, 0.18, 0.95),
    "minecraft:leaves2": (0.27, 0.50, 0.20, 0.95),
    "minecraft:glass": (0.75, 0.88, 0.88, 0.35),
    "minecraft:water": (0.22, 0.44, 0.86, 0.50),
    "minecraft:flowing_water": (0.24, 0.45, 0.85, 0.50),
    "minecraft:lava": (0.95, 0.37, 0.10, 0.75),
    "minecraft:flowing_lava": (0.92, 0.34, 0.08, 0.75),
    "minecraft:sandstone": (0.85, 0.80, 0.63, 1.0),
    "minecraft:brick_block": (0.61, 0.27, 0.25, 1.0),
    "minecraft:stonebrick": (0.52, 0.52, 0.52, 1.0),
    "minecraft:obsidian": (0.12, 0.08, 0.20, 1.0),
    "minecraft:netherrack": (0.44, 0.15, 0.16, 1.0),
    "minecraft:soul_sand": (0.34, 0.26, 0.20, 1.0),
    "minecraft:glowstone": (0.72, 0.56, 0.23, 1.0),
    "minecraft:quartz_block": (0.92, 0.90, 0.87, 1.0),
    "minecraft:snow": (0.96, 0.97, 0.98, 1.0),
    "minecraft:wool": (0.80, 0.80, 0.80, 1.0),
    "minecraft:stained_glass": (0.70, 0.75, 0.80, 0.35),
    "minecraft:stained_hardened_clay": (0.68, 0.61, 0.54, 1.0),
}


WOOL_DATA_COLORS: Dict[int, RGBA] = {
    0: (0.93, 0.93, 0.93, 1.0),
    1: (0.86, 0.50, 0.20, 1.0),
    2: (0.70, 0.34, 0.85, 1.0),
    3: (0.40, 0.60, 0.85, 1.0),
    4: (0.89, 0.78, 0.23, 1.0),
    5: (0.50, 0.78, 0.20, 1.0),
    6: (0.88, 0.50, 0.73, 1.0),
    7: (0.30, 0.31, 0.31, 1.0),
    8: (0.59, 0.60, 0.60, 1.0),
    9: (0.17, 0.57, 0.78, 1.0),
    10: (0.47, 0.33, 0.74, 1.0),
    11: (0.20, 0.30, 0.70, 1.0),
    12: (0.46, 0.29, 0.18, 1.0),
    13: (0.36, 0.48, 0.20, 1.0),
    14: (0.63, 0.22, 0.22, 1.0),
    15: (0.12, 0.12, 0.12, 1.0),
}


def _token_for(block_id: int, block_data: int) -> str:
    keyed = f"{block_id}:{block_data}"
    if keyed in STANDARD_VOCAB:
        return keyed
    return str(block_id)


def _extract_resource_location(vocab_name: str) -> Optional[str]:
    if "(" not in vocab_name or ")" not in vocab_name:
        return None
    start = vocab_name.rfind("(") + 1
    end = vocab_name.rfind(")")
    if start <= 0 or end <= start:
        return None
    return vocab_name[start:end]


def _resource_location_for(block_id: int, block_data: int) -> str:
    token = _token_for(block_id, block_data)
    value = STANDARD_VOCAB.get(token, STANDARD_VOCAB.get(str(block_id), "Unknown"))
    resource_location = _extract_resource_location(value)
    if resource_location:
        return resource_location
    return f"minecraft:block_{block_id}"


def _hashed_color(resource_location: str) -> RGBA:
    color_seed = abs(hash(resource_location))
    red = ((color_seed >> 0) & 0xFF) / 255.0
    green = ((color_seed >> 8) & 0xFF) / 255.0
    blue = ((color_seed >> 16) & 0xFF) / 255.0
    return (0.35 + 0.45 * red, 0.35 + 0.45 * green, 0.35 + 0.45 * blue, 1.0)


def _color_for(resource_location: str, block_data: int) -> RGBA:
    if resource_location == "minecraft:wool":
        return WOOL_DATA_COLORS.get(block_data, WOOL_DATA_COLORS[0])
    if resource_location in BASE_TEXTURE_COLORS:
        return BASE_TEXTURE_COLORS[resource_location]
    return _hashed_color(resource_location)


@dataclass
class Structure:
    """Standardized block-volume representation using XYZ indexing.

    `block_ids[x, y, z]` and `block_data[x, y, z]` align to classic Minecraft
    numeric id + metadata formats (pre-flattening block-state IDs).
    """

    block_ids: np.ndarray
    block_data: np.ndarray
    source_path: Optional[str] = None
    air_block_id: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.block_ids = np.asarray(self.block_ids, dtype=np.int32)
        self.block_data = np.asarray(self.block_data, dtype=np.int32)
        if self.block_ids.shape != self.block_data.shape:
            raise ValueError("block_ids and block_data must have the same shape")
        if self.block_ids.ndim != 3:
            raise ValueError("block_ids and block_data must be 3D arrays in XYZ order")

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.block_ids.shape)

    @property
    def occupied_mask(self) -> np.ndarray:
        return self.block_ids != self.air_block_id

    @classmethod
    def from_schematic(
        cls,
        schematic,
        source_path: Optional[str] = None,
    ) -> "Structure":
        blocks_yzx = np.asarray(schematic.blocks, dtype=np.int32)
        data_yzx = np.asarray(schematic.data, dtype=np.int32)

        # Standardize to XYZ order for rendering/manipulation.
        block_ids_xyz = np.transpose(blocks_yzx, (2, 0, 1))
        block_data_xyz = np.transpose(data_yzx, (2, 0, 1))

        return cls(
            block_ids=block_ids_xyz,
            block_data=block_data_xyz,
            source_path=source_path,
        )

    @classmethod
    def from_schematic_path(cls, path: str) -> "Structure":
        schematic = load_schematic(path)
        return cls.from_schematic(schematic=schematic, source_path=path)

    def token_grid(self) -> np.ndarray:
        token_grid = np.empty(self.shape, dtype=object)
        it = np.nditer(
            [self.block_ids, self.block_data, token_grid],
            flags=["multi_index", "refs_ok"],
            op_flags=[["readonly"], ["readonly"], ["writeonly"]],
        )
        for block_id, block_data, target in it:
            target[...] = _token_for(int(block_id), int(block_data))
        return token_grid

    def resource_location_grid(self) -> np.ndarray:
        resource_grid = np.empty(self.shape, dtype=object)
        it = np.nditer(
            [self.block_ids, self.block_data, resource_grid],
            flags=["multi_index", "refs_ok"],
            op_flags=[["readonly"], ["readonly"], ["writeonly"]],
        )
        for block_id, block_data, target in it:
            target[...] = _resource_location_for(int(block_id), int(block_data))
        return resource_grid

    def facecolors(self) -> np.ndarray:
        colors = np.zeros(self.shape + (4,), dtype=float)
        it = np.nditer(
            [self.block_ids, self.block_data],
            flags=["multi_index"],
            op_flags=[["readonly"], ["readonly"]],
        )
        for block_id, block_data in it:
            voxel_index = it.multi_index
            block_int = int(block_id)
            if block_int == self.air_block_id:
                colors[voxel_index] = BASE_TEXTURE_COLORS["minecraft:air"]
                continue
            resource_location = _resource_location_for(block_int, int(block_data))
            colors[voxel_index] = _color_for(resource_location, int(block_data))
        return colors

    def crop_to_non_air(self) -> "Structure":
        occupied = self.occupied_mask
        if not np.any(occupied):
            return self

        nonzero = np.argwhere(occupied)
        mins = nonzero.min(axis=0)
        maxs = nonzero.max(axis=0) + 1

        x0, y0, z0 = mins.tolist()
        x1, y1, z1 = maxs.tolist()

        return Structure(
            block_ids=self.block_ids[x0:x1, y0:y1, z0:z1],
            block_data=self.block_data[x0:x1, y0:y1, z0:z1],
            source_path=self.source_path,
            air_block_id=self.air_block_id,
            metadata=dict(self.metadata),
        )

    def downsample(self, max_dim: int) -> "Structure":
        if max_dim <= 0:
            raise ValueError("max_dim must be > 0")

        current_max = max(self.shape)
        if current_max <= max_dim:
            return self

        step = int(np.ceil(current_max / max_dim))
        return Structure(
            block_ids=self.block_ids[::step, ::step, ::step],
            block_data=self.block_data[::step, ::step, ::step],
            source_path=self.source_path,
            air_block_id=self.air_block_id,
            metadata=dict(self.metadata),
        )