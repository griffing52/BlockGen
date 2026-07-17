"""Map legacy ``(block_id, block_data)`` pairs onto modern Minecraft block states.

Every model in this repo speaks the classic pre-flattening vocabulary: a numeric
block id plus a data value (see ``blockgen.utils.data.Structure``). Minecraft
1.21 speaks namespaced block states (``minecraft:oak_planks``). This module is
the bridge, and it is the only place that knowledge lives — the mod receives
ready-made state strings and just parses them.

Resolution order for each pair:

1. ``_PAIR_TABLE`` — ``(id, data)`` entries, for the families where the data value
   carries real meaning (dye colors, wood species, slab variants).
2. ``_OVERRIDES_ID`` — whole-id entries, for families where data is noise (stairs,
   doors) or the display name is ambiguous (id 2, "Grass").
3. ``_BY_NAME`` — keyed on the legacy *display name*, so it holds for every data
   value. Most fixes belong here: the corpora put arbitrary texture-variant indices
   in ``block_data``, so ``(26, 9)`` and ``(26, 10)`` are both just "Bed", and an
   ``(id, data)``-keyed override would fix one and silently miss the other.
4. ``_FAMILIES`` — regex rewrites over the display name, compressing the wood/color
   families ("Spruce Wood Plank" -> ``spruce_planks``).
5. ``snake(display_name)`` — the fallback, correct for ~74% of the vocabulary
   ("Mossy Stone Bricks" -> ``mossy_stone_bricks``).

**Traps.** A snake-cased display name can be a real-but-wrong modern block, so a
name existing in the registry is not evidence the mapping is right. Legacy id 2
("Grass") is the grass *block*, but ``grass`` was a plant in 1.19 and became
``short_grass`` in 1.20.3; legacy 31:1 ("Tall Grass") is the *short* one-block
plant, while modern ``tall_grass`` is the two-block plant. Both are pinned in
``_OVERRIDES``. Run ``scripts/export_blockmap.py`` to regenerate and re-validate
the table against a real block registry report.

**Orientation is not recoverable.** The corpora derive ``block_data`` from
GrabCraft texture-variant indices, not true legacy metadata: every stairs id in
the piece vocab carries exactly one data value (53 -> {2}, 114 -> {7}), where
real metadata would spread over 0..7. So stairs/logs/doors get default states
rather than fabricated facings. Colors (wool, concrete, terracotta: data 0-15)
*are* faithful and are mapped exactly.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, Optional, Tuple

from blockgen.tokenizers.standard_vocab import STANDARD_VOCAB

IdData = Tuple[int, int]

AIR = "minecraft:air"

# Dye order shared by wool / carpet / stained glass / terracotta / concrete.
_COLORS = [
    "white", "orange", "magenta", "light_blue", "yellow", "lime", "pink",
    "gray", "light_gray", "cyan", "purple", "blue", "brown", "green", "red",
    "black",
]
_WOODS = ["oak", "spruce", "birch", "jungle", "acacia", "dark_oak"]

# Legacy stone_slab (44) / double_stone_slab (43) data -> modern slab block.
# Follows the official flattening: double_stone_slab[variant=stone] became
# smooth_stone_slab[type=double], NOT stone_slab.
_STONE_SLABS = [
    "smooth_stone_slab", "sandstone_slab", "oak_slab", "cobblestone_slab",
    "brick_slab", "stone_brick_slab", "nether_brick_slab", "quartz_slab",
]


# --- explicit overrides -----------------------------------------------------
# Keyed by (id, data) first, then by bare id (any data). Values are modern block
# state strings without the namespace; ``minecraft:`` is added on output.
_OVERRIDES_PAIR: Dict[IdData, str] = {
    # id 31's display names collide with id 2's ("Grass"), so it cannot be keyed by
    # name and its data values genuinely differ.
    (31, 0): "dead_bush",           # "Dead Shrub"
    (31, 1): "short_grass",         # one-block plant; renamed from grass in 1.20.3
    (31, 2): "fern",
}

_OVERRIDES_ID: Dict[int, str] = {
    # "Grass" means the grass BLOCK for id 2 and the short plant for id 31 -- the one
    # display name that is genuinely ambiguous, so both are pinned by id.
    2: "grass_block",
    31: "short_grass",              # any data value not listed above
    # id 36 (the piston moving-block) has no STANDARD_VOCAB entry at all, so there is
    # no display name to key on and it would otherwise fall back to solid stone. It is
    # a technical block: air means the decoder drops it.
    36: "air",
    # Doors: legacy stored upper/lower halves in data; place the lower half and
    # let the mod's placement pass leave the (broken) upper half alone.
    64: "oak_door", 193: "spruce_door", 194: "birch_door", 195: "jungle_door",
    196: "acacia_door", 197: "dark_oak_door",
    # Stairs: data is a texture-variant index here, so facing is not recoverable.
    53: "oak_stairs", 134: "spruce_stairs", 135: "birch_stairs",
    136: "jungle_stairs", 163: "acacia_stairs", 164: "dark_oak_stairs",
}

# Keyed on the legacy display name, so one entry covers every data value that name
# appears under. This is where nearly all hand-fixes belong -- see the module doc.
_BY_NAME: Dict[str, str] = {
    # Fluids: legacy split flowing/still; modern has one block per fluid.
    "Flowing Water": "water", "Still Water": "water",
    "Flowing Lava": "lava", "Still Lava": "lava",
    # Traps: the snake-cased name is a real but different modern block.
    "Double Tallgrass": "tall_grass",   # the two-block plant
    "Large Fern": "large_fern",
    # Renames / spelling drift.
    "Lapis Lazuli Ore": "lapis_ore", "Lapis Lazuli Block": "lapis_block",
    "Moss Stone": "mossy_cobblestone",
    "Monster Spawner": "spawner",
    "Sugar Canes": "sugar_cane",
    "Jack o'Lantern": "jack_o_lantern",
    "Cake Block": "cake",
    "Melon Block": "melon",
    "Vines": "vine",
    "Nether Brick": "nether_bricks",
    "Enchantment Table": "enchanting_table",
    "Pillar Quartz Block": "quartz_pillar",
    "Hay Bale": "hay_block",
    "Hardened Clay": "terracotta",
    "Block of Coal": "coal_block",
    "Beetroot Block": "beetroots",
    "Grass Path": "dirt_path",          # renamed in 1.17
    "Red Nether Brick": "red_nether_bricks",
    "Daylight Sensor": "daylight_detector",
    "Weighted Pressure Plate (light)": "light_weighted_pressure_plate",
    "Weighted Pressure Plate (heavy)": "heavy_weighted_pressure_plate",
    "Mob Head": "skeleton_skull",
    "Bed": "red_bed",
    "Wall-mounted Banner": "white_wall_banner",
    "Standing Sign Block": "oak_sign",
    "Wall-mounted Sign Block": "oak_wall_sign",
    "Iron Door Block": "iron_door",
    "Wooden Pressure Plate": "oak_pressure_plate",
    "Wooden Trapdoor": "oak_trapdoor",
    "Wooden Button": "oak_button",
    "Wooden Slab": "oak_slab",
    "Double Wooden Slab": "oak_slab[type=double]",
    "Stone Slab": "smooth_stone_slab",  # the flattening sent 44:0 to smooth_stone
    "Double Stone Slab": "smooth_stone_slab[type=double]",
    "Purpur Double Slab": "purpur_slab[type=double]",
    "Double Red Sandstone Slab": "red_sandstone_slab[type=double]",
    # Powered / lit variants: legacy used separate ids, modern uses properties.
    "Burning Furnace": "furnace[lit=true]",
    "Redstone Torch (off)": "redstone_torch[lit=false]",
    "Redstone Torch (on)": "redstone_torch",
    "Redstone Repeater Block (off)": "repeater",
    "Redstone Repeater Block (on)": "repeater[powered=true]",
    "Redstone Lamp (inactive)": "redstone_lamp",
    "Redstone Lamp (active)": "redstone_lamp[lit=true]",
    "Redstone Comparator (inactive)": "comparator",
    "Redstone Comparator (active)": "comparator[powered=true]",
    "Wheat Crops": "wheat[age=7]",
    # Infested ("monster egg") stone family, legacy 97:0-5.
    "Stone Monster Egg": "infested_stone",
    "Cobblestone Monster Egg": "infested_cobblestone",
    "Stone Brick Monster Egg": "infested_stone_bricks",
    "Mossy Stone Brick Monster Egg": "infested_mossy_stone_bricks",
    "Cracked Stone Brick Monster Egg": "infested_cracked_stone_bricks",
    "Chiseled Stone Brick Monster Egg": "infested_chiseled_stone_bricks",
}


def _color_family(bid: int, pattern: str) -> Dict[IdData, str]:
    return {(bid, d): pattern.format(color=c) for d, c in enumerate(_COLORS)}


def _build_pair_table() -> Dict[IdData, str]:
    t: Dict[IdData, str] = {}
    # Colored families: data 0-15 is a genuine dye index.
    t.update(_color_family(35, "{color}_wool"))
    t.update(_color_family(95, "{color}_stained_glass"))
    t.update(_color_family(160, "{color}_stained_glass_pane"))
    t.update(_color_family(159, "{color}_terracotta"))       # "Hardened Clay"
    t.update(_color_family(171, "{color}_carpet"))
    t.update(_color_family(251, "{color}_concrete"))
    t.update(_color_family(252, "{color}_concrete_powder"))
    # Wood species families (data 0-5 = species; higher bits are noise here).
    for d, w in enumerate(_WOODS):
        t[(5, d)] = f"{w}_planks"
        t[(6, d)] = f"{w}_sapling"
        t[(126, d)] = f"{w}_slab"
        t[(125, d)] = f"{w}_slab[type=double]"
    # Logs (17/162) and leaves (18/161) are deliberately NOT tabulated here: the
    # corpus uses stray data values (17:4, 162:9) that a 0..3 species table would
    # miss, and the "X Wood"/"X Leaves" display names already name the species
    # unambiguously. See the _FAMILIES "X Wood" -> x_log rule.
    # Stone slab variants.
    for d, name in enumerate(_STONE_SLABS):
        t[(44, d)] = name
        t[(43, d)] = f"{name}[type=double]"
    # Explicit overrides win over the family tables.
    t.update(_OVERRIDES_PAIR)
    return t


_PAIR_TABLE = _build_pair_table()

# Display-name rewrites, applied when no table entry matched. Ordered.
_FAMILIES: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"^(.*) Wood Plank$"), "{0}_planks"),
    (re.compile(r"^(.*) Wood Stairs$"), "{0}_stairs"),
    (re.compile(r"^Double (.*) Wood Slab$"), "{0}_slab[type=double]"),
    (re.compile(r"^(.*) Wood Slab$"), "{0}_slab"),
    (re.compile(r"^(.*) Door Block$"), "{0}_door"),
    (re.compile(r"^(.*) Hardened Clay$"), "{0}_terracotta"),
    # Legacy had no six-sided bark blocks, so "Oak Wood" is a LOG, not the modern
    # `oak_wood`. Without this, 17:4 and 162:9 silently map to real-but-wrong bark.
    (re.compile(r"^(.*) Wood$"), "{0}_log"),
)


def _snake(s: str) -> str:
    s = s.lower().replace("'", "")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def legacy_display_name(block_id: int, block_data: int) -> Optional[str]:
    """The classic display name, e.g. ``(35, 14) -> 'Red Wool'``."""
    raw = STANDARD_VOCAB.get(f"{block_id}:{block_data}") or \
        STANDARD_VOCAB.get(str(block_id))
    if not raw:
        return None
    m = re.match(r"^(.*)\(.*\)$", raw)
    return (m.group(1) if m else raw).strip()


@lru_cache(maxsize=4096)
def modern_state(block_id: int, block_data: int) -> str:
    """Map a legacy pair to a modern block state string (``minecraft:`` prefixed).

    Always returns something placeable; unmappable blocks degrade to stone rather
    than raising, so a single odd voxel can never abort a generation. Use
    ``is_fallback`` to measure how often that happens.
    """
    block_id, block_data = int(block_id), int(block_data)
    if block_id == 0:
        return AIR

    hit = _PAIR_TABLE.get((block_id, block_data))
    if hit is None:
        hit = _OVERRIDES_ID.get(block_id)
    if hit is not None:
        return f"minecraft:{hit}"

    name = legacy_display_name(block_id, block_data)
    if name:
        if name in _BY_NAME:
            return f"minecraft:{_BY_NAME[name]}"
        for pattern, template in _FAMILIES:
            m = pattern.match(name)
            if m:
                return f"minecraft:{template.format(_snake(m.group(1)))}"
        return f"minecraft:{_snake(name)}"
    return "minecraft:stone"


def is_fallback(block_id: int, block_data: int) -> bool:
    """True when the pair had no vocabulary entry at all and degraded to stone."""
    if block_id == 0:
        return False
    if (block_id, block_data) in _PAIR_TABLE or block_id in _OVERRIDES_ID:
        return False
    return not legacy_display_name(block_id, block_data)


def block_name(state: str) -> str:
    """``'minecraft:oak_slab[type=double]'`` -> ``'oak_slab'``."""
    return state.split("[", 1)[0].split(":", 1)[-1]
