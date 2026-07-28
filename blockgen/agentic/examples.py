"""In-context example programs for the agentic track.

The zero-shot baseline in ``scripts/zeroshot_brickgpt.py`` found what
VoxelCodeBench reports: a frontier model emits *valid-looking* output and
spatially incoherent builds. One real example in the prompt is the cheapest fix
for both format and grounding, so the agent supports a ``n_examples`` knob and
these are what it draws from.

They are hand-written rather than mined from the corpus on purpose: we have no
inverse compiler from a voxel build back to a command program, and a
one-line-per-voxel "example" would teach exactly the habit the DSL exists to
avoid. Each example is a *demonstration of technique* — footprint first, then
walls, then openings, then roof, then detail — which is the part that transfers.

:func:`select_examples` does keyword-overlap retrieval today. It is the seam where
a real retriever (CLIP text embedding over captions, as in
``blockgen/labeling/embed_conditions.py``) drops in without touching the agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence
import json
import re


@dataclass(frozen=True)
class Example:
    caption: str
    program: str
    tags: Sequence[str] = ()

    def render(self) -> str:
        return f"Build: {self.caption}\n\n{self.program.strip()}"


COTTAGE = Example(
    caption="Small oak cottage with a cobblestone base, glass windows and a "
            "steep gabled roof.",
    tags=("house", "cottage", "small", "oak", "wood", "gable", "cabin"),
    program="""\
# 1. footprint: a cobblestone plinth one block wider than the house
fill cobblestone 0 0 0 11 0 9
# 2. shell: walls on top of the plinth, floor level y=1
walls oak_planks 1 1 1 10 4 8 thickness=1
fill oak_planks 1 1 1 10 1 8            # interior floor
# 3. corner posts read as timber framing
line oak_log 1 1 1 1 4 1
line oak_log 10 1 1 10 4 1
line oak_log 1 1 8 1 4 8
line oak_log 10 1 8 10 4 8
# 4. openings: cut, then dress
clear 5 1 1 6 3 1
set oak_door 5 1 1
set oak_door 6 1 1
fill glass_pane 3 2 1 3 3 1
fill glass_pane 8 2 1 8 3 1
fill glass_pane 1 2 4 1 3 5
fill glass_pane 10 2 4 10 3 5
# 5. roof: ceiling, then a gable with an overhang for the eaves
fill oak_planks 1 5 1 10 5 8
gable oak_stairs 1 5 1 10 5 8 axis=x overhang=1
# 6. detail
set torch 4 3 0
set torch 7 3 0
fill cobblestone 9 1 9 9 7 9           # chimney
""",
)

STONE_TOWER = Example(
    caption="Round stone watchtower with a battlemented top and a lantern.",
    tags=("tower", "castle", "stone", "round", "watchtower", "keep", "medieval"),
    program="""\
# 1. round shaft: hollow cylinder, solid base ring for the foundation
cylinder cobblestone 8 0 8 5 1
cylinder cobblestone 8 1 8 5 14 hollow=true
# 2. floors every 5 blocks
cylinder oak_planks 8 6 8 4 1
cylinder oak_planks 8 11 8 4 1
# 3. arrow slits, cut through the shaft on all four sides
clear 3 4 8 3 5 8
clear 13 4 8 13 5 8
clear 8 4 3 8 5 3
clear 8 4 13 8 5 13
# 4. battlements: a ring, then remove every other block
cylinder stone_bricks 8 15 8 5 2 hollow=true
clear 3 16 7 3 16 9
clear 13 16 7 13 16 9
clear 7 16 3 9 16 3
clear 7 16 13 9 16 13
# 5. light, resting on the top floor (nothing floats)
set glowstone 8 12 8
set lantern 8 13 8
""",
)

MODERN_HOUSE = Example(
    caption="Two-storey modern house with white concrete walls, a flat roof and "
            "large glass panels.",
    tags=("modern", "contemporary", "house", "concrete", "glass", "flat", "two-story"),
    program="""\
# 1. slab foundation and ground floor shell
fill gray_concrete 0 0 0 15 0 11
walls white_concrete 0 1 0 15 4 11
fill white_concrete 0 5 0 15 5 11        # floor slab between storeys
# 2. glazed south face: cut the wall out and glaze the opening
clear 2 1 0 13 4 0
fill glass 2 1 0 13 4 0
# 3. upper storey, set back to make a terrace
walls white_concrete 0 6 3 15 9 11
fill white_concrete 0 10 3 15 10 11      # flat roof
clear 3 6 3 12 9 3
fill glass 3 6 3 12 9 3
# 4. terrace rail and roof trim
fill glass_pane 0 6 0 15 6 2
replace any white_terracotta 0 10 3 15 10 11
# 5. entrance
clear 7 1 0 8 3 0
set oak_door 7 1 0
set oak_door 8 1 0
fill sea_lantern 6 4 0 9 4 0
""",
)

TREE_GARDEN = Example(
    caption="A small garden with a tree, a fence and a stone path.",
    tags=("garden", "tree", "nature", "fence", "path", "park", "yard"),
    program="""\
# 1. ground
fill grass_block 0 0 0 15 0 15
fill gravel 7 0 0 8 0 15                 # path
# 2. tree: trunk then a leaf canopy
line oak_log 4 1 4 4 6 4
sphere oak_leaves 4 7 4 3
sphere oak_leaves 4 9 4 2
# 3. fence around the plot, with a gate on the path
box oak_fence 0 1 0 15 1 15
clear 7 1 0 8 1 0
set oak_fence_gate 7 1 0
set oak_fence_gate 8 1 0
# 4. flower bed
fill podzol 11 1 11 14 1 14
""",
)

EXAMPLES: List[Example] = [COTTAGE, STONE_TOWER, MODERN_HOUSE, TREE_GARDEN]

_WORD = re.compile(r"[a-z]+")


def select_examples(prompt: str, k: int = 1,
                    pool: Sequence[Example] = tuple(EXAMPLES)) -> List[Example]:
    """Pick the ``k`` examples whose tags best match the prompt.

    Ties break towards the first example in the pool, so ``k=1`` on an
    unrecognizable prompt deterministically yields the cottage — the most generic
    demonstration.
    """
    if k <= 0:
        return []
    words = set(_WORD.findall(prompt.lower()))
    scored = [(-sum(1 for t in ex.tags if t in words), i, ex)
              for i, ex in enumerate(pool)]
    scored.sort()
    return [ex for _, _, ex in scored[:k]]


def load_examples(path: Path | str) -> List[Example]:
    """Load extra examples from JSON ``[{caption, program, tags?}, …]``.

    Lets an experiment swap in a curated example set (say, programs a previous run
    produced and a human kept) without editing this file.
    """
    blob = json.loads(Path(path).read_text())
    return [Example(caption=e["caption"], program=e["program"],
                    tags=tuple(e.get("tags", ()))) for e in blob]


__all__ = ["EXAMPLES", "Example", "load_examples", "select_examples"]
