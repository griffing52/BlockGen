"""Remap modern (post-flattening) block-state names to legacy (block_id, data).

text2mc stores builds over a 3,717-entry modern block-state vocabulary
(``tok2block.json``, e.g. ``minecraft:oak_stairs[facing=east,half=bottom]``),
while every other corpus — and all our tokenizers, colors, and models — uses
classic numeric ``(block_id, block_data)`` pairs (1.12-era). This module maps
modern base names onto the nearest legacy family so the corpora share one
vocabulary and one render path.

Mapping strategy (first hit wins):
1. exact base-name table (``_EXACT``)
2. pattern parses: ``{color}_{thing}`` (wool/concrete/terracotta/…) and
   ``{wood}_{thing}`` (planks/log/stairs/…), including ``stripped_`` logs
3. material-matched stairs/slabs/walls
4. coarse substring families (deepslate → stone, blackstone → cobble, …)
5. fallback ``(1, 0)`` stone, counted so coverage is measurable

Fidelity is *family-level* by design: a ``polished_deepslate_stairs`` becomes
plain stone stairs. Good enough for shape/material learning and rendering;
not a lossless format conversion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

IdData = Tuple[int, int]

# Dye order shared by wool / concrete / terracotta / stained glass / carpet.
_COLORS = {
    "white": 0, "orange": 1, "magenta": 2, "light_blue": 3, "yellow": 4,
    "lime": 5, "pink": 6, "gray": 7, "light_gray": 8, "silver": 8, "cyan": 9,
    "purple": 10, "blue": 11, "brown": 12, "green": 13, "red": 14, "black": 15,
}
_WOODS = {"oak": 0, "spruce": 1, "birch": 2, "jungle": 3, "acacia": 4, "dark_oak": 5}
# Modern-only woods folded onto the closest legacy species by color.
_WOOD_ALIASES = {
    "cherry": "birch", "mangrove": "jungle", "bamboo": "jungle",
    "crimson": "dark_oak", "warped": "spruce", "pale_oak": "birch",
}

_COLORED_THINGS: Dict[str, int] = {           # {color}_{thing} -> block id
    "wool": 35, "carpet": 171, "stained_glass": 95, "stained_glass_pane": 160,
    "concrete": 251, "concrete_powder": 252, "terracotta": 159,
    "glazed_terracotta": 159, "shulker_box": 251, "banner": 35, "bed": 35,
    "candle": 35, "stained_hardened_clay": 159,
}

_WOOD_THINGS: Dict[str, Optional[int]] = {    # {wood}_{thing}; None = per-species table
    "planks": 5, "slab": 126, "sapling": 6,
    "log": None, "wood": None, "leaves": None, "stairs": None, "fence": None,
    "fence_gate": None, "door": None, "trapdoor": 96, "button": 143,
    "pressure_plate": 72, "sign": 63, "wall_sign": 68, "hanging_sign": 63,
}
_WOOD_LOG = {0: (17, 0), 1: (17, 1), 2: (17, 2), 3: (17, 3), 4: (162, 0), 5: (162, 1)}
_WOOD_LEAVES = {0: (18, 0), 1: (18, 1), 2: (18, 2), 3: (18, 3), 4: (161, 0), 5: (161, 1)}
_WOOD_STAIRS = {0: (53, 0), 1: (134, 0), 2: (135, 0), 3: (136, 0), 4: (163, 0), 5: (164, 0)}
_WOOD_FENCE = {0: (85, 0), 1: (188, 0), 2: (189, 0), 3: (190, 0), 4: (192, 0), 5: (191, 0)}
_WOOD_GATE = {0: (107, 0), 1: (183, 0), 2: (184, 0), 3: (185, 0), 4: (187, 0), 5: (186, 0)}
_WOOD_DOOR = {0: (64, 0), 1: (193, 0), 2: (194, 0), 3: (195, 0), 4: (196, 0), 5: (197, 0)}

_STONE_STAIRS: Dict[str, IdData] = {
    "cobblestone": (67, 0), "stone": (67, 0), "smooth_stone": (67, 0),
    "stone_brick": (109, 0), "mossy_stone_brick": (109, 0), "mossy_cobblestone": (67, 0),
    "brick": (108, 0), "sandstone": (128, 0), "smooth_sandstone": (128, 0),
    "red_sandstone": (180, 0), "quartz": (156, 0), "smooth_quartz": (156, 0),
    "nether_brick": (114, 0), "purpur": (203, 0), "andesite": (67, 0),
    "polished_andesite": (67, 0), "granite": (67, 0), "polished_granite": (67, 0),
    "diorite": (67, 0), "polished_diorite": (67, 0), "end_stone_brick": (128, 0),
    "prismarine": (67, 0), "prismarine_brick": (67, 0), "dark_prismarine": (67, 0),
}
_STONE_SLAB: Dict[str, int] = {               # stone_slab (44) data values
    "stone": 0, "smooth_stone": 0, "sandstone": 1, "cut_sandstone": 1,
    "smooth_sandstone": 1, "cobblestone": 3, "mossy_cobblestone": 3, "brick": 4,
    "stone_brick": 5, "mossy_stone_brick": 5, "nether_brick": 6, "quartz": 7,
    "smooth_quartz": 7,
}

_EXACT: Dict[str, IdData] = {
    "air": (0, 0), "cave_air": (0, 0), "void_air": (0, 0),
    "barrier": (0, 0), "light": (0, 0), "structure_void": (0, 0),
    "moving_piston": (0, 0), "end_portal": (0, 0), "end_gateway": (0, 0),
    "nether_portal": (90, 0), "bubble_column": (9, 0),
    "carpet": (171, 0), "concrete_powder": (252, 0), "button": (77, 0),
    "banner": (35, 0), "wall_banner": (35, 0), "bone_block": (155, 0),
    "bamboo_block": (17, 0), "stripped_bamboo_block": (17, 0),
    "cake": (35, 0), "snow_cauldron": (118, 0), "lava_cauldron": (118, 0),
    "cave_vines": (106, 0), "cave_vines_plant": (106, 0),
    "big_dripleaf": (111, 0), "small_dripleaf": (111, 0),
    "azalea_leaves_flowers": (18, 0), "bee_hive": (5, 0),
    "command_block": (137, 0), "chain_command_block": (137, 0),
    "repeating_command_block": (137, 0),
    "stone": (1, 0), "granite": (1, 1), "polished_granite": (1, 2),
    "diorite": (1, 3), "polished_diorite": (1, 4), "andesite": (1, 5),
    "polished_andesite": (1, 6), "smooth_stone": (1, 0),
    "grass_block": (2, 0), "dirt": (3, 0), "coarse_dirt": (3, 1),
    "rooted_dirt": (3, 0), "dirt_path": (2, 0), "podzol": (3, 2), "mud": (3, 0),
    "cobblestone": (4, 0), "mossy_cobblestone": (48, 0), "bedrock": (7, 0),
    "water": (9, 0), "flowing_water": (9, 0), "lava": (11, 0), "flowing_lava": (11, 0),
    "grass_path": (2, 0), "bed": (35, 0), "dead_shrub": (32, 0), "honeycomb_block": (45, 0),
    "sand": (12, 0), "red_sand": (12, 1),
    "gravel": (13, 0), "gold_ore": (14, 0), "iron_ore": (15, 0), "coal_ore": (16, 0),
    "copper_ore": (15, 0), "diamond_ore": (56, 0), "emerald_ore": (129, 0),
    "redstone_ore": (73, 0), "lapis_ore": (21, 0), "nether_quartz_ore": (153, 0),
    "sponge": (19, 0), "wet_sponge": (19, 1), "glass": (20, 0), "tinted_glass": (20, 0),
    "glass_pane": (102, 0), "lapis_block": (22, 0), "sandstone": (24, 0),
    "chiseled_sandstone": (24, 1), "cut_sandstone": (24, 2), "smooth_sandstone": (24, 2),
    "red_sandstone": (179, 0), "chiseled_red_sandstone": (179, 1),
    "cut_red_sandstone": (179, 2), "smooth_red_sandstone": (179, 2),
    "note_block": (25, 0), "cobweb": (30, 0), "grass": (31, 1), "short_grass": (31, 1),
    "tall_grass": (31, 1), "fern": (31, 2), "large_fern": (31, 2), "dead_bush": (32, 0),
    "wool": (35, 0), "dandelion": (37, 0), "poppy": (38, 0),
    "brown_mushroom": (39, 0), "red_mushroom": (40, 0),
    "gold_block": (41, 0), "iron_block": (42, 0), "brick_block": (45, 0),
    "bricks": (45, 0), "tnt": (46, 0), "bookshelf": (47, 0), "chiseled_bookshelf": (47, 0),
    "obsidian": (49, 0), "crying_obsidian": (49, 0), "torch": (50, 0),
    "wall_torch": (50, 0), "lantern": (89, 0), "soul_lantern": (89, 0),
    "fire": (51, 0), "spawner": (52, 0), "chest": (54, 0), "trapped_chest": (146, 0),
    "ender_chest": (130, 0), "barrel": (54, 0), "redstone_wire": (55, 0),
    "diamond_block": (57, 0), "crafting_table": (58, 0), "wheat": (59, 0),
    "farmland": (60, 0), "furnace": (61, 0), "blast_furnace": (61, 0),
    "smoker": (61, 0), "ladder": (65, 0), "rail": (66, 0), "powered_rail": (27, 0),
    "detector_rail": (28, 0), "activator_rail": (157, 0),
    "lever": (69, 0), "stone_pressure_plate": (70, 0), "iron_door": (71, 0),
    "redstone_torch": (76, 0), "stone_button": (77, 0), "snow": (78, 0),
    "snow_layer": (78, 0), "snow_block": (80, 0), "ice": (79, 0),
    "packed_ice": (174, 0), "blue_ice": (174, 0), "frosted_ice": (79, 0),
    "cactus": (81, 0), "clay": (82, 0), "sugar_cane": (83, 0), "reeds": (83, 0),
    "jukebox": (84, 0), "pumpkin": (86, 0), "carved_pumpkin": (86, 0),
    "jack_o_lantern": (91, 0), "netherrack": (87, 0), "soul_sand": (88, 0),
    "soul_soil": (88, 0), "glowstone": (89, 0), "shroomlight": (89, 0),
    "sea_lantern": (169, 0), "iron_trapdoor": (167, 0),
    "infested_stone": (1, 0), "infested_stone_bricks": (98, 0),
    "infested_cobblestone": (4, 0), "stone_bricks": (98, 0),
    "mossy_stone_bricks": (98, 1), "cracked_stone_bricks": (98, 2),
    "chiseled_stone_bricks": (98, 3), "iron_bars": (101, 0), "chain": (101, 0),
    "melon": (103, 0), "vine": (106, 0), "glow_lichen": (106, 0),
    "nether_bricks": (112, 0), "cracked_nether_bricks": (112, 0),
    "chiseled_nether_bricks": (112, 0), "red_nether_bricks": (112, 0),
    "nether_brick_fence": (113, 0), "enchanting_table": (116, 0),
    "end_stone": (121, 0), "end_stone_bricks": (206, 0),
    "emerald_block": (133, 0), "cobblestone_wall": (139, 0),
    "mossy_cobblestone_wall": (139, 1), "flower_pot": (140, 0),
    "carrots": (141, 0), "potatoes": (142, 0), "anvil": (145, 0),
    "chipped_anvil": (145, 0), "damaged_anvil": (145, 0),
    "redstone_block": (152, 0), "hopper": (154, 0),
    "quartz_block": (155, 0), "chiseled_quartz_block": (155, 1),
    "quartz_pillar": (155, 2), "smooth_quartz": (155, 0), "quartz_bricks": (155, 0),
    "terracotta": (172, 0), "hardened_clay": (172, 0), "coal_block": (173, 0),
    "prismarine": (168, 0), "prismarine_bricks": (168, 1), "dark_prismarine": (168, 2),
    "hay_block": (170, 0), "slime_block": (165, 0), "honey_block": (165, 0),
    "purpur_block": (201, 0), "purpur_pillar": (202, 0), "purpur_slab": (205, 0),
    "purpur_stairs": (203, 0), "mycelium": (110, 0), "lily_pad": (111, 0),
    "moss_block": (2, 0), "moss_carpet": (171, 13), "azalea": (18, 0),
    "flowering_azalea": (18, 0), "mangrove_roots": (17, 3), "muddy_mangrove_roots": (3, 0),
    "scaffolding": (85, 0), "composter": (54, 0), "lectern": (58, 0),
    "cartography_table": (58, 0), "fletching_table": (58, 0), "smithing_table": (58, 0),
    "loom": (58, 0), "stonecutter": (1, 0), "grindstone": (1, 0), "bell": (41, 0),
    "campfire": (50, 0), "soul_campfire": (50, 0), "beehive": (5, 0), "bee_nest": (5, 0),
    "dried_kelp_block": (170, 0), "target": (170, 0), "lodestone": (98, 0),
    "respawn_anchor": (49, 0), "cauldron": (118, 0), "water_cauldron": (118, 0),
    "brewing_stand": (117, 0), "end_portal_frame": (120, 0), "dragon_egg": (122, 0),
    "beacon": (138, 0), "conduit": (138, 0), "daylight_detector": (151, 0),
    "observer": (1, 0), "dispenser": (23, 0), "dropper": (158, 0), "piston": (33, 0),
    "sticky_piston": (29, 0), "piston_head": (34, 0), "tripwire_hook": (131, 0),
    "tripwire": (132, 0), "repeater": (93, 0), "comparator": (149, 0),
    "sculk": (49, 0), "sculk_catalyst": (49, 0), "amethyst_block": (35, 10),
    "budding_amethyst": (35, 10), "calcite": (155, 0), "smooth_basalt": (49, 0),
    "kelp": (106, 0), "kelp_plant": (106, 0), "seagrass": (31, 1),
    "tall_seagrass": (31, 1), "bamboo": (83, 0), "cocoa": (127, 0),
    "melon_stem": (105, 0), "pumpkin_stem": (104, 0), "sweet_berry_bush": (31, 1),
    "sunflower": (175, 0), "lilac": (175, 1), "rose_bush": (175, 4),
    "peony": (175, 5), "cornflower": (38, 0), "lily_of_the_valley": (38, 0),
    "wither_rose": (38, 0), "allium": (38, 2), "azure_bluet": (38, 3),
    "red_tulip": (38, 4), "orange_tulip": (38, 5), "white_tulip": (38, 6),
    "pink_tulip": (38, 7), "oxeye_daisy": (38, 8),
}

# Coarse substring families, checked in order (modern-only materials).
_FAMILIES: Tuple[Tuple[str, IdData], ...] = (
    ("cobbled_deepslate", (4, 0)), ("deepslate_brick", (98, 0)),
    ("deepslate_tile", (98, 0)), ("polished_deepslate", (1, 0)),
    ("chiseled_deepslate", (98, 3)), ("reinforced_deepslate", (49, 0)),
    ("deepslate", (1, 0)), ("polished_blackstone_brick", (112, 0)),
    ("blackstone", (49, 0)), ("basalt", (1, 5)), ("tuff", (1, 5)),
    ("dripstone", (172, 0)), ("copper_grate", (101, 0)),
    ("oxidized", (168, 0)), ("weathered", (168, 0)),
    ("exposed", (172, 0)), ("copper", (172, 0)),
    ("raw_iron", (42, 0)), ("raw_gold", (41, 0)), ("raw_copper", (172, 0)),
    ("netherite", (49, 0)), ("ancient_debris", (49, 0)),
    ("end_rod", (50, 0)), ("chorus", (201, 0)),
    ("nylium", (110, 0)), ("wart_block", (112, 0)), ("nether_wart", (115, 0)),
    ("froglight", (89, 0)), ("ochre", (89, 0)), ("verdant", (89, 0)),
    ("pearlescent", (89, 0)), ("magma", (87, 0)),
    ("mushroom_stem", (17, 0)), ("mushroom_block", (39, 0)),
    ("coral", (168, 1)), ("sculk", (49, 0)), ("amethyst", (35, 10)),
    ("skull", (1, 0)), ("head", (1, 0)), ("candle", (50, 0)),
    ("pointed_dripstone", (172, 0)), ("mud_brick", (172, 0)),
    ("packed_mud", (172, 0)), ("resin", (172, 0)),
)

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_TOK2BLOCK = _REPO / "data" / "minecraft" / "text2mc" / "tok2block.json"


def _base_name(state: str) -> str:
    """'minecraft:oak_stairs[facing=east]' -> 'oak_stairs'."""
    name = state.split("[", 1)[0]
    return name.split(":", 1)[-1]


def _split_prefix(name: str, table: Dict[str, int]) -> Optional[Tuple[int, str]]:
    """Match '{key}_{rest}' for the longest key in *table*; returns (value, rest)."""
    for key in sorted(table, key=len, reverse=True):
        if name.startswith(key + "_"):
            return table[key], name[len(key) + 1:]
    return None


def remap_name(state: str) -> Optional[IdData]:
    """Map one modern block-state string to legacy (id, data); None if unknown."""
    name = _base_name(state)
    if name in _EXACT:
        return _EXACT[name]

    # {color}_{thing}: wool, concrete, terracotta, stained glass, carpet, ...
    hit = _split_prefix(name, _COLORS)
    if hit is not None:
        color, rest = hit
        if rest in _COLORED_THINGS:
            return (_COLORED_THINGS[rest], color)

    # {wood}_{thing}: planks, logs, stairs, fences, doors, ...
    wood_name = name
    stripped = wood_name.startswith("stripped_")
    if stripped:
        wood_name = wood_name[len("stripped_"):]
    for alias, target in _WOOD_ALIASES.items():
        if wood_name.startswith(alias + "_"):
            wood_name = target + wood_name[len(alias):]
            break
    hit = _split_prefix(wood_name, _WOODS)
    if hit is not None:
        species, rest = hit
        if rest in _WOOD_THINGS:
            flat = _WOOD_THINGS[rest]
            if flat is not None:
                return (flat, species if rest in ("planks", "slab", "sapling") else 0)
            per = {"log": _WOOD_LOG, "wood": _WOOD_LOG, "leaves": _WOOD_LEAVES,
                   "stairs": _WOOD_STAIRS, "fence": _WOOD_FENCE,
                   "fence_gate": _WOOD_GATE, "door": _WOOD_DOOR}[rest]
            return per[species]

    # Material-matched stairs / slabs / walls.
    if name.endswith("_stairs"):
        mat = name[: -len("_stairs")]
        return _STONE_STAIRS.get(mat, (53, 0))
    if name.endswith("_slab"):
        mat = name[: -len("_slab")]
        if mat in _STONE_SLAB:
            return (44, _STONE_SLAB[mat])
        return (44, 0)
    if name.endswith("_wall"):
        return (139, 0)
    if name.endswith("_pane"):
        return (102, 0)
    if name.endswith("_fence"):
        return (85, 0)
    if name.endswith("_fence_gate"):
        return (107, 0)
    if name.endswith("_door"):
        return (64, 0)
    if name.endswith("_trapdoor"):
        return (96, 0)
    if name.endswith("_button"):
        return (77, 0)
    if name.endswith("_pressure_plate"):
        return (70, 0)
    if name.endswith("_sign"):
        return (63, 0)
    if name.endswith("_sapling") or name.endswith("_propagule"):
        return (6, 0)
    if name.endswith("_leaves"):
        return (18, 0)
    if name.endswith("_log") or name.endswith("_wood") or name.endswith("_stem") or name.endswith("_hyphae"):
        return (17, 0)
    if name.endswith("_planks") or name.endswith("_mosaic"):
        return (5, 0)
    if name.endswith("_banner"):
        return (35, 0)
    if name.endswith("_bed"):
        return (35, 14)
    if name.endswith("_carpet"):
        return (171, 0)
    if name.endswith("_ore"):
        return (1, 0)
    if name.endswith("_bricks") or name.endswith("_brick"):
        return (98, 0)

    for sub, target in _FAMILIES:
        if sub in name:
            return target
    return None


def build_token_lut(tok2block_path: Path | str = DEFAULT_TOK2BLOCK,
                    verbose: bool = False) -> np.ndarray:
    """LUT ``(n_tokens, 2)`` int32: text2mc token -> legacy (block_id, block_data).

    Unknown names map to stone ``(1, 0)``; air variants to ``(0, 0)``.
    """
    tok2block = json.loads(Path(tok2block_path).read_text())
    n = max(int(k) for k in tok2block) + 1
    lut = np.full((n, 2), (1, 0), dtype=np.int32)
    unmapped = []
    for k, state in tok2block.items():
        mapped = remap_name(state)
        if mapped is None:
            unmapped.append(_base_name(state))
        else:
            lut[int(k)] = mapped
    if verbose:
        uniq = sorted(set(unmapped))
        print(f"[block_remap] {n - len(unmapped)}/{n} states mapped; "
              f"{len(uniq)} unique base names fell back to stone")
        if uniq:
            print("  fallbacks:", ", ".join(uniq[:40]) + ("…" if len(uniq) > 40 else ""))
    return lut


def remap_token_array(tokens: np.ndarray, lut: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized remap of a text2mc token array -> (block_ids, block_data)."""
    t = np.clip(tokens, 0, len(lut) - 1)
    mapped = lut[t]
    return mapped[..., 0].astype(np.int32), mapped[..., 1].astype(np.int32)
