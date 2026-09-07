"""The small part of the ontology that has to be written down.

Two kinds of fact live here, and neither can be mined:

**Physical properties** are rules of the game, not statistics. Whether oak planks
burn is true of Minecraft, not of our corpus, and no amount of looking at houses
will reveal it. They are written as *rules over block names* rather than as 70
hand-typed rows: a rule is auditable ("everything wooden is flammable"), it
covers palette entries added later, and it cannot silently disagree with itself
the way a long hand table does.

**Placement grammar** is derived from repo code rather than authored at all --
:mod:`blockgen.agentic.blockstate` already knows which families accept
``facing``/``axis``/``type``, because it is the module that turns those
properties into legacy metadata bits. Restating it here by hand would be a second
source of truth that drifts; asking the module is a fact about what the executor
will actually honour.

Marked ``authored`` and ``derived`` in the schema respectively, so an ablation
can strip them and ask what the *mined* fields are worth on their own.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from blockgen.agentic.blockstate import STAIRS_IDS, try_resolve_block

_LOG_IDS = frozenset({17, 162})
_SLAB_IDS = frozenset({44, 126, 182, 205})

#: ``(pattern, flammable)``. First match wins; the default is not flammable.
_FLAMMABLE: Tuple[Tuple[str, bool], ...] = (
    (r"stripped_|_log$|_wood$|planks|fence|_door$|trapdoor|bookshelf|"
     r"crafting_table|_wool$|carpet|leaves|hay_block|ladder|_sign$|scaffolding", True),
)

#: Blocks that fall. Sand and gravel are the whole list in our palette; concrete
#: *powder* falls but plain concrete does not, hence the negative lookahead.
_GRAVITY = re.compile(r"^(sand|red_sand|gravel|.*_concrete_powder|anvil)$")

#: Light level, for the handful of palette entries that emit any.
_LIGHT: Dict[str, int] = {
    "torch": 14, "lantern": 15, "glowstone": 15, "sea_lantern": 15,
    "redstone_torch": 7, "jack_o_lantern": 15, "beacon": 15, "end_rod": 14,
    "shroomlight": 15, "campfire": 15, "magma_block": 3, "fire": 15,
    "lava": 15,
}

#: Things that do not fill their cube. A wall built out of these has holes in it,
#: which is the single most common way an agentic build fails its coherence check.
_NOT_FULL = re.compile(
    r"_stairs$|_slab$|_fence$|_fence_gate$|_wall$|_pane$|_door$|trapdoor|"
    r"ladder|torch|lantern|_sign$|carpet|button|lever|_bars$|chain|"
    r"leaves|_bed$|flower_pot|rail|vine|sapling|grass$|fern|_plate$")


def _base(name: str) -> str:
    return name.strip().lower().split(":", 1)[-1].split("[", 1)[0]


def flammable(name: str) -> bool:
    base = _base(name)
    return any(bool(re.search(pattern, base)) for pattern, value in _FLAMMABLE if value)


def gravity(name: str) -> bool:
    return bool(_GRAVITY.match(_base(name)))


def light_level(name: str) -> int:
    return _LIGHT.get(_base(name), 0)


def full_block(name: str) -> bool:
    """False for anything that leaves gaps -- panes, fences, stairs, slabs."""
    return not _NOT_FULL.search(_base(name))


def state_properties(name: str) -> List[str]:
    """Which state properties the executor will actually honour for this block.

    Derived from :mod:`blockgen.agentic.blockstate`: a property this list does not
    mention parses fine and then does nothing, which is exactly the kind of
    silent no-op the ontology exists to prevent.
    """
    pair = try_resolve_block(_base(name))
    if pair is None:
        return []
    block_id, _ = pair
    if block_id in STAIRS_IDS:
        return ["facing", "half"]
    if block_id in _LOG_IDS:
        return ["axis"]
    if block_id in _SLAB_IDS:
        return ["type"]
    return []


def physical_attrs(name: str) -> Dict[str, object]:
    """The authored + derived block of one part's attributes."""
    tags: List[str] = []
    if flammable(name):
        tags.append("flammable")
    if gravity(name):
        tags.append("falls")
    if light_level(name):
        tags.append(f"light{light_level(name)}")
    if not full_block(name):
        tags.append("not-full")
    states = state_properties(name)
    return {"physical.notes": tags,
            "physical.states": ",".join(states) if states else None}


__all__ = ["flammable", "full_block", "gravity", "light_level", "physical_attrs",
           "state_properties"]
