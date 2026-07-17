"""Pin the legacy->modern block mappings that are wrong-but-plausible.

``scripts/export_blockmap.py`` proves every mapping names a *real* block. It cannot
prove it names the *right* one -- `minecraft:oak_wood` and `minecraft:grass` are real
blocks, they are just not what those legacy ids mean. Each case below is one that a
name-similarity mapping gets confidently wrong, so they are pinned here rather than
left to a human noticing a weird-looking house.

Run:  python -m pytest deploy/inference/tests/ -q
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from blockgen_server.blockmap import is_fallback, modern_state


@pytest.mark.parametrize("pair,expected,why", [
    # "Grass" is the grass BLOCK for id 2; modern `grass` was a plant and was renamed
    # `short_grass` in 1.20.3, so the obvious mapping is both wrong AND now nonexistent.
    ((2, 0), "minecraft:grass_block", "legacy 2 is the block, not the plant"),
    ((31, 1), "minecraft:short_grass", "legacy 31:1 is the one-block plant"),
    ((175, 2), "minecraft:tall_grass", "'Double Tallgrass' is the two-block plant"),
    # Legacy had no six-sided bark blocks, so "Oak Wood" is a log. `oak_wood` exists
    # and is wrong. 17:4 / 162:9 are the data values the corpora actually emit.
    ((17, 0), "minecraft:oak_log", "no bark blocks in legacy"),
    ((17, 4), "minecraft:oak_log", "stray data value still means oak log"),
    ((162, 9), "minecraft:acacia_log", "stray data value still means acacia log"),
    # Renames.
    ((208, 0), "minecraft:dirt_path", "grass_path renamed in 1.17"),
    ((172, 0), "minecraft:terracotta", "hardened clay renamed"),
    ((159, 14), "minecraft:red_terracotta", "stained hardened clay renamed"),
    ((116, 0), "minecraft:enchanting_table", "'Enchantment Table' misnames it"),
    # The flattening sent double/stone slabs somewhere non-obvious.
    ((44, 0), "minecraft:smooth_stone_slab", "44:0 flattened to smooth_stone_slab"),
    ((43, 0), "minecraft:smooth_stone_slab[type=double]", "double slab -> type=double"),
])
def test_trap_mappings(pair, expected, why) -> None:
    assert modern_state(*pair) == expected, why


@pytest.mark.parametrize("pair", [(26, 9), (26, 10), (62, 2), (62, 3), (96, 4), (96, 5),
                                  (143, 0), (143, 3), (106, 1), (106, 8), (170, 0), (170, 8)])
def test_data_value_variants_map_the_same(pair) -> None:
    """Any data value of a block must map, not just the ones one corpus happened to use.

    The corpora put texture-variant indices in block_data, so the same block appears
    under different data values in different vocabularies -- (26, 9) and (26, 10) are
    both just "Bed". Overrides are keyed by display name for exactly this reason; an
    (id, data)-keyed table fixed one and left the other resolving to a nonexistent
    block, which is how `minecraft:bed` reached Minecraft.
    """
    state = modern_state(*pair)
    assert state != "minecraft:stone", f"{pair} fell back to stone"
    assert not state.startswith("minecraft:bed"), f"{pair} -> {state}, not a real block"


def test_bed_variants_resolve_to_a_real_bed() -> None:
    assert modern_state(26, 9) == "minecraft:red_bed"
    assert modern_state(26, 10) == "minecraft:red_bed"


def test_air_is_air_not_stone() -> None:
    assert modern_state(0, 0) == "minecraft:air"
    # id 36 (piston moving-block) has no vocabulary entry at all; it must not become a
    # solid block, or every one of them litters the build with stray stone.
    assert modern_state(36, 0) == "minecraft:air"


def test_unknown_ids_degrade_to_stone_rather_than_raise() -> None:
    """A single odd voxel must never abort a live generation."""
    assert modern_state(9999, 0) == "minecraft:stone"
    assert is_fallback(9999, 0)
    assert not is_fallback(5, 0)
