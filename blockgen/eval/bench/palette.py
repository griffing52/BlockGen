"""Block-palette statistics: what a build is made of, and in what proportions.

Three granularities, because the right one depends on the question:

* ``pair``   -- raw ``(block_id, block_data)``. The literal palette, orientation
                included: "Oak Stairs facing north" and "...facing south" are
                distinct. 667 distinct pairs on the house corpus.
* ``exact``  -- the ``STANDARD_VOCAB`` token, i.e. the display-name granularity
                ("Oak Wood Stairs"). Orientation is collapsed, species is not.
                296 distinct tokens on the house corpus, 100% vocabulary
                coverage. This is the default.
* ``family`` -- material class ("wood_oak", "cobblestone", "glass"). ~30 buckets.

Why ``family`` is needed at all: an exact-token histogram over a 296-symbol
alphabet estimated from one 60-block sample is almost all zeros, so JSD
saturates near its bound and stops discriminating. Collapsing form and colour
onto material puts the estimate back on a support the sample size can actually
cover.

**Two traps this module is written to avoid.**

1. JSD is computed *macro*: each structure's histogram is normalized first, then
   averaged. Pooling raw counts lets a single 4000-block cathedral define the
   corpus palette and drown out a hundred small houses.
2. The symbol set is fixed from the reference (train) side and shared by every
   arm. An arm inventing new blocks does not get a private alphabet; those blocks
   land in the reported ``palette_oov_frac`` instead of silently vanishing.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Dict, Hashable, List, Sequence, Tuple

import numpy as np

from blockgen.eval.bench import stats
from blockgen.tokenizers.standard_vocab import STANDARD_VOCAB
from blockgen.utils.data import Structure, _token_for

LEVELS = ("pair", "exact", "family")
FAMILY_OTHER = "other"

# Species checked longest-first so "dark oak" never matches the "oak" rule.
_SPECIES = ("dark oak", "spruce", "birch", "jungle", "acacia", "oak")

# Ordered (pattern, family). First match wins, so the table reads as a priority
# list: plants before wood (Oak Leaves is foliage, not lumber), specific stone
# types before generic stone, materials before the functional catch-alls.
_RULES: Tuple[Tuple[str, str], ...] = (
    (r"leaves|sapling|flower|tulip|orchid|allium|daisy|dandelion|poppy|rose|"
     r"azure bluet|oxeye|lilac|peony|sunflower|bluebell|"
     r"grass|fern|vine|cactus|mushroom|wheat|carrot|potato|melon|pumpkin|"
     r"sugar cane|lily|beetroot|cocoa|nether wart|dead bush", "plant"),
    (r"wool|carpet", "wool"),
    (r"glass", "glass"),
    (r"hay|bale|straw", "plant"),
    (r"nether brick", "nether_brick"),
    (r"stone brick", "stone_brick"),
    (r"end stone|endstone|purpur", "end_stone"),
    (r"prismarine|sea lantern", "prismarine"),
    (r"sandstone", "sandstone"),
    (r"quartz", "quartz"),
    (r"brick", "brick"),
    (r"cobblestone|mossy cobble", "cobblestone"),
    (r"obsidian", "obsidian"),
    (r"bedrock", "bedrock"),
    (r"netherrack|soul sand|magma", "nether"),
    (r"clay|terracotta", "clay"),
    (r"concrete", "concrete"),
    (r"snow|ice", "snow"),
    (r"water|lava", "liquid"),
    (r"iron|gold|diamond|emerald|lapis|redstone|coal|quartz ore|nether star",
     "metal_gem"),
    (r"torch|lamp|lantern|glowstone|beacon|fire", "light"),
    (r"dirt|podzol|mycelium|farmland|soil|path", "dirt"),
    (r"gravel|sand", "sand"),
    (r"andesite|diorite|granite|cobble|stone", "stone"),
    # `fence` is safe to treat as lumber here: the only non-wooden fences are
    # nether brick and iron, and both match an earlier rule.
    (r"wood|plank|log|fence", "wood"),    # generic wood, species resolved below
    (r"door|trapdoor|ladder|chest|furnace|bed\b|bookshelf|crafting|anvil|sign|"
     r"banner|gate|button|pressure plate|lever|rail|hopper|dispenser|piston|"
     r"jukebox|note block|cauldron|brewing|enchant|table|frame|pot|barrier",
     "utility"),
)
_COMPILED = tuple((re.compile(p), fam) for p, fam in _RULES)


def _display_name(block_id: int, block_data: int) -> str:
    """Human block name from STANDARD_VOCAB, e.g. "Oak Wood Stairs"."""
    return STANDARD_VOCAB.get(_token_for(int(block_id), int(block_data)), "").split(
        "(", 1)[0].strip()


@lru_cache(maxsize=8192)
def block_family(block_id: int, block_data: int) -> str:
    """Material family for a legacy ``(id, data)`` pair.

    Derived from the display name rather than from ``block_remap``, which maps
    modern-name -> legacy and therefore cannot answer this direction. Wooden
    blocks additionally carry their species ("wood_oak"), because species is the
    dominant visual difference between two otherwise identical builds.
    """
    name = _display_name(block_id, block_data).lower()
    if not name:
        return FAMILY_OTHER
    for pattern, family in _COMPILED:
        if pattern.search(name):
            if family != "wood":
                return family
            for sp in _SPECIES:
                if sp in name:
                    return f"wood_{sp.replace(' ', '_')}"
            return "wood_other"
    return FAMILY_OTHER


def block_key(block_id: int, block_data: int, level: str = "exact") -> Hashable:
    if level == "pair":
        return (int(block_id), int(block_data))
    if level == "exact":
        return _display_name(block_id, block_data) or FAMILY_OTHER
    if level == "family":
        return block_family(block_id, block_data)
    raise ValueError(f"level must be one of {LEVELS}, got {level!r}")


def palette_hist(s: Structure, level: str = "exact") -> Dict[Hashable, int]:
    """Block counts for one structure, air excluded."""
    occ = s.occupied_mask
    counts: Counter = Counter()
    ids, datas = s.block_ids[occ], s.block_data[occ]
    # Bincount over unique pairs first: a 4000-block structure has ~20 distinct
    # blocks, so this turns 4000 lookups into 20.
    pairs, n = np.unique(np.stack([ids, datas], 1), axis=0, return_counts=True)
    for (bid, bd), c in zip(pairs.tolist(), n.tolist()):
        counts[block_key(bid, bd, level)] += int(c)
    return dict(counts)


def palette_size(s: Structure, level: str = "exact") -> int:
    return len(palette_hist(s, level))


def vocabulary(structures: Sequence[Structure], level: str = "exact") -> List[Hashable]:
    """Sorted symbol set over a reference corpus. Fix this from train, once."""
    seen: set = set()
    for s in structures:
        seen.update(palette_hist(s, level))
    return sorted(seen, key=str)


def palette_matrix(
    structures: Sequence[Structure],
    level: str = "exact",
    keys: Sequence[Hashable] | None = None,
) -> Tuple[np.ndarray, List[Hashable], np.ndarray]:
    """Counts ``[N, K]`` over a fixed symbol set.

    Returns ``(counts, keys, oov_counts)``. Symbols outside ``keys`` are *not*
    folded into a bucket -- they are returned separately so the caller reports
    them as `palette_oov_frac` rather than pretending they did not occur.
    """
    keys = list(keys) if keys is not None else vocabulary(structures, level)
    index = {k: i for i, k in enumerate(keys)}
    counts = np.zeros((len(structures), len(keys)), dtype=np.float64)
    oov = np.zeros(len(structures), dtype=np.float64)
    for i, s in enumerate(structures):
        for k, c in palette_hist(s, level).items():
            j = index.get(k)
            if j is None:
                oov[i] += c
            else:
                counts[i, j] = c
    return counts, keys, oov


def macro_distribution(counts: np.ndarray) -> np.ndarray:
    """Mean of per-structure normalized histograms (see module docstring)."""
    if counts.size == 0 or counts.shape[0] == 0:
        return np.zeros(counts.shape[1] if counts.ndim == 2 else 0)
    totals = counts.sum(1, keepdims=True)
    safe = np.where(totals > 0, totals, 1.0)
    return (counts / safe).mean(0)


def union_vocabulary(a: Sequence[Structure], b: Sequence[Structure],
                     level: str = "exact") -> List[Hashable]:
    """Symbols occurring in either set.

    Divergences *must* be computed on the union. Scoring an arm on the
    reference's alphabet alone makes every unknown block invisible: a generator
    emitting nothing but blocks the corpus never uses would produce an all-zero
    histogram, which renormalizes to the reference distribution and reports JSD
    0.0 -- a perfect palette match for a build made of the wrong materials
    entirely. (`palette_oov_frac` still measures novelty against the *reference*
    alphabet; that is a different question and keeps its own denominator.)
    """
    return sorted(set(vocabulary(a, level)) | set(vocabulary(b, level)), key=str)


def palette_jsd(
    generated: Sequence[Structure],
    reference: Sequence[Structure],
    level: str = "exact",
    keys: Sequence[Hashable] | None = None,
) -> float:
    keys = list(keys) if keys is not None else union_vocabulary(generated, reference, level)
    g, _, _ = palette_matrix(generated, level, keys)
    r, _, _ = palette_matrix(reference, level, keys)
    return stats.jsd(macro_distribution(g), macro_distribution(r))


def presence_matrix(counts: np.ndarray) -> np.ndarray:
    """Binary "this build uses this material" matrix, for co-occurrence."""
    return (counts > 0).astype(np.float64)


def cooccurrence_from_presence(presence: np.ndarray) -> np.ndarray:
    return presence.T @ presence


def jsd_from_counts(gen_counts: np.ndarray, ref_macro: np.ndarray) -> float:
    """JSD given a precomputed count matrix and reference distribution.

    Exists so bootstrap resampling can draw *rows* of an already-built matrix
    instead of re-deriving every histogram (including the reference's) on each
    of hundreds of draws. Same value as `palette_jsd` over the same alphabet.
    """
    return stats.jsd(macro_distribution(gen_counts), ref_macro)


def palette_oov_frac(structures: Sequence[Structure], keys: Sequence[Hashable],
                     level: str = "exact") -> float:
    """Fraction of blocks whose symbol is absent from the reference vocabulary."""
    counts, _, oov = palette_matrix(structures, level, keys)
    total = counts.sum() + oov.sum()
    return float(oov.sum() / total) if total > 0 else 0.0


def family_other_frac(structures: Sequence[Structure]) -> float:
    """Fraction of blocks the family map failed to classify.

    The correctness gate for `_RULES`: measured at 0.4% on the real training
    split. A regression here means the rule table has drifted from the corpus.
    """
    total = other = 0.0
    for s in structures:
        for k, c in palette_hist(s, "family").items():
            total += c
            if k == FAMILY_OTHER:
                other += c
    return float(other / total) if total > 0 else 0.0


def cooccurrence(structures: Sequence[Structure], level: str = "family",
                 keys: Sequence[Hashable] | None = None,
                 topk: int = 48) -> np.ndarray:
    """Symmetric ``[K, K]`` matrix of how often two materials share a build.

    Restricted to the ``topk`` most common symbols: over the full alphabet the
    matrix is almost entirely zero, and JSD between two near-zero matrices
    saturates instead of discriminating. Presence-based, not count-based, so one
    enormous wall cannot dominate.
    """
    keys = list(keys) if keys is not None else vocabulary(structures, level)
    counts, keys, _ = palette_matrix(structures, level, keys)
    if counts.shape[1] > topk:
        order = np.argsort(-counts.sum(0))[:topk]
        counts = counts[:, order]
    present = (counts > 0).astype(np.float64)
    return present.T @ present


def cooccurrence_jsd(generated: Sequence[Structure], reference: Sequence[Structure],
                     level: str = "family", topk: int = 48) -> float:
    """JSD between co-occurrence patterns over the union alphabet.

    Top-K is chosen by reference frequency (the reference defines what "common"
    means), but the alphabet itself is the union -- see `union_vocabulary` for
    why scoring on the reference's symbols alone hides a wrong palette.
    """
    keys = union_vocabulary(generated, reference, level)
    counts, _, _ = palette_matrix(reference, level, keys)
    if len(keys) > topk:
        keys = [keys[i] for i in np.argsort(-counts.sum(0))[:topk]]
    g = cooccurrence(generated, level, keys, topk=len(keys))
    r = cooccurrence(reference, level, keys, topk=len(keys))
    return stats.jsd(g.ravel(), r.ravel())
