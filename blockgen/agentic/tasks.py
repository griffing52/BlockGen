"""Prompt sets for agentic experiments.

Text conditioning is the point of this track, so *which* text we condition on is
an experimental variable, not a detail. Four sources, all addressable by name:

``short``     terse prompts ("a brick house") — the hard case, minimal grounding
``detailed``  hand-written rich prompts in the style of our VLM captions
``large``     multi-part / long-horizon requests (a village, a castle complex) —
              the regime the per-voxel tracks cannot reach at all
``captions:k`` real captions from the labeled house corpus
              (``data/minecraft/labels/houses_32_captions.json``), caption index
              ``k``: 0 is the short tag, 1-2 are progressively richer descriptions

The last one is what makes the detail ablation honest: ``captions:0`` and
``captions:2`` describe *the same builds* at different levels of detail, so the
comparison isolates prompt richness instead of confounding it with subject matter.
It also lets an agentic build be compared against the real build it was described
from — the same held-out captions the conditioned neural tracks are scored on.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

CAPTIONS_PATH = Path("data/minecraft/labels/houses_32_captions.json")

SHORT_PROMPTS: List[str] = [
    "a brick house",
    "a wooden cabin",
    "a stone tower",
    "a modern house",
    "a small church",
    "a windmill",
    "a farmhouse with a barn",
    "a lighthouse",
    "a Japanese pagoda",
    "a desert hut",
    "a treehouse",
    "a bridge over a river",
]

DETAILED_PROMPTS: List[str] = [
    "Two-story suburban house with a steep gabled wooden roof, a front porch with "
    "posts, sandstone walls and a stone chimney on the right side.",
    "Small medieval cottage: cobblestone base, oak timber framing with white plaster "
    "infill, a steep dark roof, one door and two shuttered windows.",
    "Round stone watchtower three storeys tall with arrow slits, a battlemented top, "
    "and a wooden door at the base.",
    "Modern flat-roofed villa with white concrete walls, a full-height glass front, "
    "a set-back upper storey and a terrace with a glass railing.",
    "A red brick village church with a tall square bell tower, arched stained-glass "
    "windows, and a pitched slate roof.",
    "Wooden windmill on a stone base with four sails, a small door and a shingled cap.",
    "Rustic farmhouse with a red barn beside it, hay bales, a wooden fence enclosing "
    "the yard, and a dirt path leading to the door.",
    "White and red striped lighthouse on a rocky base with a glass lantern room and a "
    "railed gallery at the top.",
    "Three-tier Japanese pagoda with upturned dark roofs, red wooden columns, a stone "
    "foundation and a small garden with a tree.",
    "Sandstone desert dwelling with a flat roof, small square windows, an awning over "
    "the entrance and palm-like planting beside it.",
    "Oak treehouse built into a large tree, with a plank platform, rope-style ladder, "
    "a railed walkway and a small pitched roof.",
    "Stone arch bridge spanning a water channel, with cobblestone railings, lanterns "
    "along the deck and steps at both ends.",
]

LARGE_PROMPTS: List[str] = [
    "A village square: three different houses around a central well, connected by "
    "gravel paths, with fences, trees and lanterns.",
    "A castle with four corner towers, a curtain wall with battlements, a gatehouse "
    "with a portcullis opening, and a keep in the middle.",
    "A small farm complex: a two-storey farmhouse, a barn, a fenced animal pen, crop "
    "fields in rows, and a dirt track connecting them.",
    "A harbour front: a stone quay, two wooden piers over the water, a warehouse, a "
    "crane, and a lighthouse at the end of the breakwater.",
    "A monastery: a cloister of columns around a courtyard garden, a chapel with a "
    "tall roof, and a bell tower.",
    "A modern city block: three towers of different heights with glass facades, a "
    "plaza with planters between them, and a low shopfront row along one side.",
]

BUILTIN_SETS: Dict[str, List[str]] = {
    "short": SHORT_PROMPTS,
    "detailed": DETAILED_PROMPTS,
    "large": LARGE_PROMPTS,
}


def load_caption_prompts(n: int, caption_index: int = 0, seed: int = 0,
                         path: Path | str = CAPTIONS_PATH) -> List[str]:
    """Sample ``n`` real captions at detail level ``caption_index``.

    Deterministic given ``seed``, and the *same* seed picks the same builds at any
    caption index — that is what makes the detail ablation a paired comparison.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"caption file {path} not found; build it with "
            f"`python -m blockgen.labeling.build_captions` (see docs/labeling.md)")
    blob = json.loads(path.read_text())
    keys = sorted(k for k, v in blob.items() if len(v) > caption_index)
    picks = random.Random(seed).sample(keys, min(n, len(keys)))
    return [blob[k][caption_index] for k in sorted(picks)]


def load_prompts(source: str, n: int = 8, seed: int = 0) -> List[str]:
    """Resolve a prompt-set spec to a list of prompts.

    ``"short"`` / ``"detailed"`` / ``"large"`` (built-in), ``"captions:2"`` (real
    captions at detail level 2), or a path to a ``.txt`` (one prompt per line) or
    ``.json`` (list of strings) file.
    """
    name, _, arg = source.partition(":")
    name = name.strip().lower()
    if name in BUILTIN_SETS:
        pool = BUILTIN_SETS[name]
        return list(pool[:n]) if n <= len(pool) else list(pool)
    if name in ("captions", "caption"):
        return load_caption_prompts(n, int(arg or 0), seed=seed)
    path = Path(source)
    if path.is_file():
        if path.suffix == ".json":
            blob = json.loads(path.read_text())
            prompts = blob if isinstance(blob, list) else list(blob.values())
        else:
            prompts = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        return [str(p) for p in prompts[:n]]
    raise ValueError(
        f"unknown prompt source '{source}'; use one of {sorted(BUILTIN_SETS)}, "
        f"'captions:<index>', or a path to a .txt/.json file")


def describe_sources() -> str:
    lines = [f"  {k:<10} {len(v)} prompts" for k, v in BUILTIN_SETS.items()]
    lines.append("  captions:<i>  real corpus captions at detail level i (0=short tag)")
    lines.append("  <path>.txt|.json  your own prompts")
    return "\n".join(lines)


__all__ = ["BUILTIN_SETS", "DETAILED_PROMPTS", "LARGE_PROMPTS", "SHORT_PROMPTS",
           "describe_sources", "load_caption_prompts", "load_prompts"]
