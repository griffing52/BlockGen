"""Block names for the agentic DSL -> legacy ``(block_id, block_data)`` pairs.

The agentic track lets an LLM write build *programs* in modern Minecraft
vocabulary (``oak_planks``, ``stone_bricks``, ``oak_stairs[facing=north]``),
because that is the vocabulary frontier models actually know. Everything
downstream of :class:`blockgen.utils.data.Structure` — tokenizers, renderer,
eval, the deploy blockmap — speaks the classic pre-flattening ``(id, data)``
pair, so this module is the one-way bridge in.

Two steps:

1. **Base name -> legacy family** via :func:`blockgen.utils.block_remap.remap_name`,
   the same table the text2mc corpus import uses. Family-level fidelity by design
   (``polished_deepslate_stairs`` becomes plain stone stairs).
2. **State properties -> legacy metadata bits**, for the three families where the
   corpora carry true metadata and the renderer/deploy path can read it back:
   stairs (``facing``/``half``), logs (``axis``), slabs (``type``). The bit
   conventions mirror ``deploy/inference/blockgen_server/blockmap.py`` exactly, so
   a program's ``[facing=north]`` survives the round trip into a live server.

Anything else (doors, trapdoors) keeps its default state — legacy door metadata
entangles facing with open/hinge bits and inventing one risks an illegal state.

:data:`PALETTE` is the curated name list that goes into the system prompt. Keeping
it short and *verified to resolve* (see ``tests/test_agentic_dsl.py``) is what keeps
generations inside a vocabulary the renderer can actually draw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from blockgen.utils.block_remap import remap_name

IdData = Tuple[int, int]

AIR: IdData = (0, 0)
AIR_NAMES = frozenset({"air", "minecraft:air", "empty", "none"})

# Families whose legacy data value we can synthesize from a modern property.
STAIRS_IDS = frozenset({53, 67, 108, 109, 114, 128, 134, 135, 136, 156, 163,
                        164, 180, 203})
_STAIRS_IDS = STAIRS_IDS   # internal alias, kept so _apply_props reads naturally
_LOG_IDS = frozenset({17, 162})
_SLAB_IDS = frozenset({44, 126, 182, 205})

# Legacy stairs bits: data & 3 -> facing, data & 4 -> upside-down.
_STAIR_FACING = {"east": 0, "west": 1, "south": 2, "north": 3}
# Legacy log bits: (data >> 2) & 3 -> axis (3 = all-bark "wood").
_LOG_AXIS = {"y": 0, "x": 1, "z": 2, "none": 3, "all": 3}


class UnknownBlockError(ValueError):
    """Raised when a program names a block no legacy family matches."""


@dataclass(frozen=True)
class BlockRef:
    """A parsed block spec: base name plus modern state properties."""

    name: str
    props: Dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:  # round-trips back into a program line
        if not self.props:
            return self.name
        body = ",".join(f"{k}={v}" for k, v in sorted(self.props.items()))
        return f"{self.name}[{body}]"


def parse_block(spec: str) -> BlockRef:
    """``"oak_stairs[facing=north,half=top]"`` -> :class:`BlockRef`.

    The namespace prefix is optional and stripped; unknown property *keys* are
    kept (harmless) so a model can be verbose without failing the parse.
    """
    text = spec.strip()
    props: Dict[str, str] = {}
    if "[" in text:
        head, _, tail = text.partition("[")
        text = head
        body = tail.rstrip("]")
        for part in body.split(","):
            if not part.strip():
                continue
            key, _, value = part.partition("=")
            props[key.strip().lower()] = value.strip().lower()
    name = text.strip().lower().split(":", 1)[-1]
    return BlockRef(name=name, props=props)


def _apply_props(pair: IdData, props: Dict[str, str]) -> IdData:
    """Fold modern state properties into the legacy metadata nibble."""
    block_id, data = pair
    if block_id in _STAIRS_IDS:
        facing = _STAIR_FACING.get(props.get("facing", ""))
        if facing is not None:
            data = (data & ~3) | facing
        if props.get("half") == "top" or props.get("upside_down") == "true":
            data |= 4
    elif block_id in _LOG_IDS:
        axis = _LOG_AXIS.get(props.get("axis", ""))
        if axis is not None:
            data = (data & 3) | (axis << 2)
    elif block_id in _SLAB_IDS:
        if props.get("type") == "top" or props.get("half") == "top":
            data |= 8
        if props.get("type") == "double":
            # Legacy doubles are a separate id (44 -> 43, 126 -> 125).
            block_id = {44: 43, 126: 125}.get(block_id, block_id)
    return (int(block_id), int(data))


def resolve_block(spec: str) -> IdData:
    """Resolve a DSL block spec to a legacy ``(block_id, block_data)`` pair.

    Raises :class:`UnknownBlockError` if no family matches — the executor turns
    that into a per-line error the agent can see and repair, rather than silently
    substituting stone (which would hide palette drift from the metrics).
    """
    ref = parse_block(spec)
    if ref.name in AIR_NAMES:
        return AIR
    pair = remap_name(ref.name)
    if pair is None:
        raise UnknownBlockError(
            f"unknown block '{spec}'; use a modern Minecraft block name such as "
            f"{', '.join(PALETTE[:6])}"
        )
    return _apply_props(pair, ref.props)


def try_resolve_block(spec: str) -> Optional[IdData]:
    """:func:`resolve_block` but ``None`` instead of raising (for validation)."""
    try:
        return resolve_block(spec)
    except UnknownBlockError:
        return None


# Curated palette advertised in the system prompt. Chosen for coverage of the
# house-building materials our corpora actually contain (and therefore what the
# renderer has textures for), not for breadth.
PALETTE = (
    # structure
    "stone", "cobblestone", "stone_bricks", "mossy_cobblestone", "andesite",
    "bricks", "sandstone", "smooth_sandstone", "quartz_block", "clay",
    "white_concrete", "gray_concrete", "black_concrete", "red_concrete",
    # wood
    "oak_planks", "spruce_planks", "birch_planks", "jungle_planks",
    "acacia_planks", "dark_oak_planks", "oak_log", "spruce_log", "birch_log",
    "dark_oak_log", "stripped_oak_log",
    # detail / roofing
    "oak_stairs", "spruce_stairs", "brick_stairs", "stone_brick_stairs",
    "cobblestone_stairs", "sandstone_stairs", "oak_slab", "stone_slab",
    "cobblestone_wall", "oak_fence", "oak_fence_gate", "oak_door",
    "oak_trapdoor", "ladder",
    # glass / light / trim
    "glass", "glass_pane", "white_stained_glass", "light_blue_stained_glass",
    "glowstone", "sea_lantern", "torch", "lantern",
    # nature / ground
    "grass_block", "dirt", "coarse_dirt", "gravel", "sand", "oak_leaves",
    "spruce_leaves", "water", "podzol",
    # color accents
    "white_wool", "red_wool", "blue_wool", "green_wool", "black_wool",
    "white_terracotta", "orange_terracotta", "bookshelf", "crafting_table",
    "furnace", "chest", "hay_block", "netherrack", "obsidian",
)
