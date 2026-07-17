"""Template captions from existing structure metadata.

Free, instant captions derived from what each corpus already carries
(``Structure.metadata``: corpus/category/title/url) plus features computed from
the voxels (dims, block count, dominant materials).  These are the baseline
half of the hybrid labeling plan; VLM captions (``vlm_captions.py``) supply the
higher-quality half.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List

import numpy as np

from blockgen.tokenizers.standard_vocab import STANDARD_VOCAB
from blockgen.utils.data import Structure, _token_for

_JUNK_PATTERNS = (
    re.compile(r"\bminecraftt?\b", re.IGNORECASE),
    re.compile(r"[#*_~\[\]{}|]+"),
    re.compile(r"\b(?:v|ver|version)\s*\d+(?:\.\d+)*\b", re.IGNORECASE),
    re.compile(r"\b(?:wip|updated?|final|new)\b[!.]*$", re.IGNORECASE),
)
_WS = re.compile(r"\s+")

# Blocks that are scenery/ground rather than the build itself.
_SCENERY = {"Grass", "Dirt", "Sand", "Gravel", "Water", "Stationary Water",
            "Lava", "Stationary Lava", "Snow", "Ice", "Tall Grass", "Dandelion",
            "Poppy", "Oak Leaves", "Bedrock", "Farmland", "Mycelium"}


def clean_title(title: str) -> str:
    """Strip counters, markup junk and 'Minecraft' noise from scraped titles."""
    t = title or ""
    for pat in _JUNK_PATTERNS:
        t = pat.sub(" ", t)
    t = re.sub(r"!{2,}", "!", t)
    t = re.sub(r"\d+\s*[xX]\s*\d+(\s*[xX]\s*\d+)?", " ", t)  # "16x16" size tags
    t = _WS.sub(" ", t).strip(" -_.|,")
    return t


def _material_name(token: str) -> str:
    """Vocab token ("id" / "id:data") -> short human material name."""
    name = STANDARD_VOCAB.get(token, "")
    name = name.split("(", 1)[0].strip()
    return name


def dominant_materials(structure: Structure, k: int = 3) -> List[str]:
    """Most common non-scenery block names, most frequent first."""
    occ = structure.occupied_mask
    ids = structure.block_ids[occ]
    datas = structure.block_data[occ]
    counts: Counter = Counter()
    for bid, bdata in zip(ids.tolist(), datas.tolist()):
        name = _material_name(_token_for(int(bid), int(bdata)))
        if name and name not in _SCENERY:
            counts[name] += 1
    return [name for name, _ in counts.most_common(k)]


def _size_word(structure: Structure) -> str:
    n = int(structure.occupied_mask.sum())
    if n < 300:
        return "small"
    if n < 1200:
        return "medium-sized"
    return "large"


def _height_phrase(structure: Structure) -> str:
    height = structure.shape[1]
    if height <= 6:
        return "single-story"
    if height <= 12:
        return "two-story"
    return "tall"


def _category_phrase(category: str) -> str:
    """GrabCraft subcategory slug -> noun phrase ("medieval-houses" -> "medieval house")."""
    words = (category or "").replace("-", " ").replace("_", " ").strip().lower()
    if not words:
        return ""
    if words.endswith("es") and not words.endswith("houses"):
        words = words[:-2]
    elif words.endswith("houses"):
        words = words[:-1]
    elif words.endswith("s"):
        words = words[:-1]
    return words


def _materials_phrase(mats: List[str]) -> str:
    low = [m.lower() for m in mats[:2]]
    if not low:
        return ""
    if len(low) == 1:
        return f"built mainly from {low[0]}"
    return f"built from {low[0]} and {low[1]}"


def template_captions(structure: Structure, k: int = 4, seed: int = 0) -> List[str]:
    """Generate up to ``k`` caption variants from metadata + voxel features."""
    meta = structure.metadata or {}
    corpus = meta.get("corpus", "")
    title = clean_title(meta.get("title", ""))
    category = _category_phrase(meta.get("category", ""))
    mats = dominant_materials(structure)
    size = _size_word(structure)
    height = _height_phrase(structure)
    mat_phrase = _materials_phrase(mats)

    kind = category or "house"
    captions: List[str] = []
    if title:
        captions.append(f"{title}, a Minecraft {kind} build")
        if mat_phrase:
            captions.append(f"A {size} {kind} {mat_phrase}: {title}")
        captions.append(f"A Minecraft build of {title.lower() if title.isupper() else title}")
    captions.append(f"A {size} {height} Minecraft {kind}"
                    + (f" {mat_phrase}" if mat_phrase else ""))
    if mat_phrase:
        captions.append(f"A {height} {kind} {mat_phrase} in Minecraft")
    captions.append(f"A {size} voxel {kind} structure")

    # dedupe, keep order, trim/pad to k
    seen = set()
    out = []
    for c in captions:
        c = _WS.sub(" ", c).strip()
        if c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    rng = np.random.default_rng(seed)
    while len(out) < k:
        out.append(out[int(rng.integers(len(out)))])
    return out[:k]
