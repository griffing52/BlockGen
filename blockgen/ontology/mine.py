"""Measure what blocks actually *do* in the corpus.

This is the half of the ontology that can carry information a frontier model
does not already have. A model knows what oak planks look like; it does not know
that in *this* corpus they sit at 0.42 of build height, run 4.3 blocks
horizontally for every 1.6 vertically, and touch oak logs 9x more often than
chance. Those are the numbers this module produces.

Everything here is measured from :class:`~blockgen.utils.data.Structure` lists --
the same objects the benchmark scores -- so a claim made from the ontology and a
claim made from the bench are about the same corpus.

Design notes
------------
**Statistics live on the display-name symbol, not the raw ``(id, data)`` pair.**
``palette.block_key(..., level="exact")`` is the granularity where "Oak Wood
Stairs" is one thing regardless of which way it faces, which is what an ontology
entry should mean. The mapping is a precomputed 4096-entry lookup table, so
turning a voxel grid into a symbol grid is one fancy-index rather than a Python
loop over blocks.

**Every statistic is accumulated per structure with numpy, never per voxel in
Python.** The house corpus is 2661 builds; a per-voxel loop is minutes, the array
formulation is seconds.

**Neighbour affinity is normalized PMI over building materials.** The most
common neighbour of *everything* is the most common block, so a raw top-k list is
the same list for every entry and tells the model nothing; plain PMI overcorrects
the other way and fills the list with trivia (a jukebox that appears twice, both
times against oak). Normalized PMI (``pmi / -log2 p(a,b)``, bounded to [-1, 1])
damps the rare-item bias, and candidates are restricted to partners with
``min_support`` placements of their own, so what comes back is "what this is
built against more than chance" rather than "what freak block happened to touch
it".

**Sparse counts are withheld, not smoothed.** A block seen 12 times gets its
count reported and its derived tags suppressed (``min_support``). Inventing a
role for a block from a dozen placements would put noise in a prompt and call it
knowledge.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from blockgen.eval.bench.palette import block_family, block_key
from blockgen.utils.data import Structure

#: Legacy ids are a byte and metadata a nibble, so ``id << 4 | data`` is a dense
#: 12-bit code and every per-code accumulator is a 4096-entry array.
NCODE = 4096
AIR_SYMBOL = "air"

#: Height bins used for ``height_profile``. Five is enough to separate
#: foundation / lower wall / upper wall / eaves / roof and few enough that a
#: block with a few hundred placements has a non-degenerate histogram.
N_HEIGHT_BINS = 5


@lru_cache(maxsize=1)
def _symbol_table() -> Tuple[np.ndarray, Tuple[str, ...], Tuple[str, ...]]:
    """``(lut, symbols, families)``: code -> symbol index, and per-symbol names.

    Built once for the whole 12-bit code space (4096 ``block_key`` calls, ~30 ms)
    so that the hot path is ``lut[code_grid]``.
    """
    symbols: List[str] = [AIR_SYMBOL]
    families: List[str] = [AIR_SYMBOL]
    index: Dict[str, int] = {AIR_SYMBOL: 0}
    lut = np.zeros(NCODE, dtype=np.int32)
    for code in range(NCODE):
        block_id, data = code >> 4, code & 15
        if block_id == 0:
            continue
        name = str(block_key(block_id, data, "exact"))
        slot = index.get(name)
        if slot is None:
            slot = len(symbols)
            index[name] = slot
            symbols.append(name)
            families.append(block_family(block_id, data))
        lut[code] = slot
    return lut, tuple(symbols), tuple(families)


def symbol_names() -> Tuple[str, ...]:
    return _symbol_table()[1]


def symbol_index() -> Dict[str, int]:
    return {name: i for i, name in enumerate(_symbol_table()[1])}


def code_of(block_id: int, block_data: int) -> int:
    return (int(block_id) << 4) | (int(block_data) & 15)


def symbol_for(block_id: int, block_data: int) -> str:
    lut, symbols, _ = _symbol_table()
    return symbols[lut[code_of(block_id, block_data) % NCODE]]


@dataclass
class BlockStats:
    """Everything measured about one symbol. Raw numbers only -- the mapping from
    these to prompt-facing tags ("pillar", "roof") lives in
    :mod:`blockgen.ontology.minecraft`, so the thresholds are visible in one
    place and this module stays a measurement."""

    symbol: str
    family: str
    count: int = 0                       # placements corpus-wide
    n_builds: int = 0                    # builds containing it at least once
    share: float = 0.0                   # count / all placements
    builds_frac: float = 0.0             # n_builds / n_builds_total
    height_mean: float = 0.0             # mean normalized height in its build
    height_profile: List[float] = field(default_factory=list)
    vertical_run: float = 0.0            # mean run length along y
    horizontal_run: float = 0.0          # mean run length along x/z
    support_frac: float = 0.0            # fraction with something solid below
    exposure: float = 0.0                # mean air-facing sides, 0..6
    self_affinity: float = 0.0           # fraction of its faces touching itself
    #: ``(symbol, npmi, share_of_this_block's_faces, pair_count)``, best first.
    neighbors: List[Tuple[str, float, float, int]] = field(default_factory=list)
    category_lift: List[Tuple[str, float, int]] = field(default_factory=list)

    @property
    def anisotropy(self) -> float:
        """>1 means it runs vertically (posts, chimneys); <1 horizontally
        (floors, courses). The single most role-diagnostic number here."""
        return self.vertical_run / self.horizontal_run if self.horizontal_run else 0.0

    def to_json(self) -> dict:
        d = {k: getattr(self, k) for k in
             ("symbol", "family", "count", "n_builds", "share", "builds_frac",
              "height_mean", "height_profile", "vertical_run", "horizontal_run",
              "support_frac", "exposure", "self_affinity")}
        d["anisotropy"] = self.anisotropy
        d["neighbors"] = [list(n) for n in self.neighbors]
        d["category_lift"] = [list(c) for c in self.category_lift]
        return d


@dataclass
class CorpusStats:
    """The mined corpus: per-symbol statistics plus what they were measured on."""

    blocks: Dict[str, BlockStats]
    n_builds: int
    n_placements: int
    corpus: str = ""
    categories: Dict[str, int] = field(default_factory=dict)

    def get(self, symbol: str) -> Optional[BlockStats]:
        return self.blocks.get(symbol)

    def top(self, k: int = 20) -> List[BlockStats]:
        return sorted(self.blocks.values(), key=lambda b: -b.count)[:k]

    def to_json(self) -> dict:
        return {"corpus": self.corpus, "n_builds": self.n_builds,
                "n_placements": self.n_placements, "categories": self.categories,
                "blocks": {k: v.to_json() for k, v in self.blocks.items()}}


# --- per-structure accumulation --------------------------------------------
def _runs(sym: np.ndarray, axis: int, run_sum: np.ndarray, run_n: np.ndarray,
          n_sym: int) -> None:
    """Accumulate contiguous same-symbol run lengths along one axis.

    Rows are flattened with a ``-1`` sentinel between them so one ``np.diff``
    finds every boundary in the volume at once; without the sentinel a run would
    wrap from the end of one column into the start of the next.
    """
    a = np.moveaxis(sym, axis, -1)
    rows = a.reshape(-1, a.shape[-1])
    sentinel = np.full((rows.shape[0], 1), -1, dtype=np.int64)
    flat = np.concatenate([rows.astype(np.int64), sentinel], axis=1).ravel()
    change = np.flatnonzero(np.diff(flat)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [flat.size]))
    values, lengths = flat[starts], (ends - starts)
    keep = values > 0                      # drop air (0) and the sentinel (-1)
    if not keep.any():
        return
    run_sum += np.bincount(values[keep], weights=lengths[keep], minlength=n_sym)
    run_n += np.bincount(values[keep], minlength=n_sym)


def _adjacency(sym: np.ndarray, pairs: Counter, n_sym: int) -> None:
    """Count unordered adjacent (symbol, symbol) face pairs over 6-connectivity."""
    for axis in (0, 1, 2):
        a = np.moveaxis(sym, axis, 0)
        lhs, rhs = a[:-1], a[1:]
        both = (lhs > 0) & (rhs > 0)
        if not both.any():
            continue
        x, y = lhs[both].astype(np.int64), rhs[both].astype(np.int64)
        lo, hi = np.minimum(x, y), np.maximum(x, y)
        combined, counts = np.unique(lo * n_sym + hi, return_counts=True)
        for key, count in zip(combined.tolist(), counts.tolist()):
            pairs[key] += count


def _air_faces(sym: np.ndarray) -> np.ndarray:
    """Per-voxel count of the 6 neighbours that are air (outside counts as air)."""
    padded = np.zeros(np.array(sym.shape) + 2, dtype=sym.dtype)
    padded[1:-1, 1:-1, 1:-1] = sym
    out = np.zeros(sym.shape, dtype=np.int16)
    for axis in range(3):
        for delta in (0, 2):
            index = [slice(1, -1)] * 3
            index[axis] = slice(delta, delta + sym.shape[axis])
            out += (padded[tuple(index)] == 0).astype(np.int16)
    return out


def mine_corpus(structures: Sequence[Structure],
                categories: Optional[Sequence[str]] = None,
                *, corpus: str = "", min_support: int = 200,
                min_pair_count: int = 30, top_neighbors: int = 5,
                min_partner_count: Optional[int] = None,
                progress_every: int = 0) -> CorpusStats:
    """Measure every block statistic in one pass over ``structures``.

    ``min_support`` gates both the derived tags and (via ``min_partner_count``,
    which defaults to it) which symbols may appear in a neighbour list.

    ``categories`` is an optional per-structure label (GrabCraft's category, say)
    used for the style lift; when omitted it is read from
    ``structure.metadata["category"]``.
    """
    lut, symbols, families = _symbol_table()
    n_sym = len(symbols)

    count = np.zeros(n_sym, dtype=np.int64)
    builds = np.zeros(n_sym, dtype=np.int64)
    height_sum = np.zeros(n_sym, dtype=np.float64)
    height_hist = np.zeros(n_sym * N_HEIGHT_BINS, dtype=np.float64)
    vrun_sum, vrun_n = np.zeros(n_sym), np.zeros(n_sym)
    hrun_sum, hrun_n = np.zeros(n_sym), np.zeros(n_sym)
    supported = np.zeros(n_sym, dtype=np.float64)
    air_face_sum = np.zeros(n_sym, dtype=np.float64)
    pairs: Counter = Counter()
    cat_present: Dict[str, Counter] = defaultdict(Counter)
    cat_builds: Counter = Counter()

    n_builds = 0
    for i, structure in enumerate(structures):
        if progress_every and i and i % progress_every == 0:
            print(f"[ontology] mined {i}/{len(structures)} builds", flush=True)
        ids = np.asarray(structure.block_ids)
        data = np.asarray(structure.block_data)
        occupied = ids != structure.air_block_id
        if not occupied.any():
            continue
        n_builds += 1
        code = ((ids.astype(np.int64) << 4) | (data.astype(np.int64) & 15))
        # Ids outside the legacy byte range cannot be indexed into the LUT and
        # are not blocks we have a symbol for; treat them as air rather than
        # wrapping them onto an unrelated code.
        code = np.where(occupied & (code < NCODE) & (code >= 0), code, 0)
        sym = lut[code]

        present = np.unique(sym)
        present = present[present > 0]
        builds[present] += 1
        occ = sym > 0
        flat_sym = sym[occ]
        count += np.bincount(flat_sym, minlength=n_sym)

        # Height, normalized inside this build's occupied bbox: a block's role is
        # relative to the structure it is in, not to the canvas it was cropped to.
        ys = np.nonzero(occ)[1]
        y_lo, y_hi = int(ys.min()), int(ys.max())
        span = max(1, y_hi - y_lo)
        y_norm = (ys - y_lo) / span
        height_sum += np.bincount(flat_sym, weights=y_norm, minlength=n_sym)
        bins = np.clip((y_norm * N_HEIGHT_BINS).astype(np.int64), 0, N_HEIGHT_BINS - 1)
        height_hist += np.bincount(flat_sym * N_HEIGHT_BINS + bins,
                                   minlength=n_sym * N_HEIGHT_BINS)

        # Support: something solid directly below, with the ground plane counting
        # as supported (a foundation is not floating).
        below = np.zeros_like(occ)
        below[:, 1:, :] = occ[:, :-1, :]
        below[:, 0, :] = True
        supported += np.bincount(flat_sym, weights=below[occ].astype(np.float64),
                                 minlength=n_sym)
        air_face_sum += np.bincount(flat_sym,
                                    weights=_air_faces(sym)[occ].astype(np.float64),
                                    minlength=n_sym)

        _runs(sym, 1, vrun_sum, vrun_n, n_sym)
        _runs(sym, 0, hrun_sum, hrun_n, n_sym)
        _runs(sym, 2, hrun_sum, hrun_n, n_sym)
        _adjacency(sym, pairs, n_sym)

        label = (categories[i] if categories is not None and i < len(categories)
                 else str((structure.metadata or {}).get("category", "")))
        label = (label or "").strip()
        if label:
            cat_builds[label] += 1
            for s in present.tolist():
                cat_present[label][s] += 1

    total = int(count.sum())
    # PMI over *ordered* face pairs: a self-pair is two ordered pairs, which keeps
    # the marginals consistent and makes log2(p(a,b) / p(a)p(b)) the plain thing.
    ordered = np.zeros(n_sym, dtype=np.float64)
    for key, n in pairs.items():
        a, b = divmod(key, n_sym)
        ordered[a] += n if a == b else n
        ordered[b] += n if a != b else n
    ordered_total = float(ordered.sum()) or 1.0
    marginal = ordered / ordered_total

    partner_floor = min_support if min_partner_count is None else min_partner_count
    neighbors: Dict[int, List[Tuple[str, float, float, int]]] = defaultdict(list)
    self_affinity = np.zeros(n_sym, dtype=np.float64)
    for key, n in pairs.items():
        a, b = divmod(key, n_sym)
        if a == b:
            if ordered[a]:
                self_affinity[a] = 2.0 * n / ordered[a]
            continue
        if n < min_pair_count:
            continue
        joint = (2.0 * n) / ordered_total       # both orderings
        denom = marginal[a] * marginal[b]
        if denom <= 0 or joint <= 0:
            continue
        npmi = math.log2(joint / denom) / -math.log2(joint)
        if count[b] >= partner_floor:
            neighbors[a].append((symbols[b], float(npmi),
                                 float(n / ordered[a]) if ordered[a] else 0.0, int(n)))
        if count[a] >= partner_floor:
            neighbors[b].append((symbols[a], float(npmi),
                                 float(n / ordered[b]) if ordered[b] else 0.0, int(n)))

    n_builds_total = max(1, n_builds)
    blocks: Dict[str, BlockStats] = {}
    for s in range(1, n_sym):
        if count[s] == 0:
            continue
        hist = height_hist[s * N_HEIGHT_BINS:(s + 1) * N_HEIGHT_BINS]
        hist = (hist / hist.sum()).tolist() if hist.sum() else [0.0] * N_HEIGHT_BINS
        lift: List[Tuple[str, float, int]] = []
        base = builds[s] / n_builds_total
        if base > 0 and count[s] >= min_support:
            for label, n_cat in cat_builds.items():
                if n_cat < 10:
                    continue
                rate = cat_present[label][s] / n_cat
                if rate > 0:
                    lift.append((label, float(rate / base), int(cat_present[label][s])))
            lift.sort(key=lambda t: -t[1])
            lift = lift[:3]
        blocks[symbols[s]] = BlockStats(
            symbol=symbols[s], family=families[s], count=int(count[s]),
            n_builds=int(builds[s]), share=float(count[s] / total) if total else 0.0,
            builds_frac=float(builds[s] / n_builds_total),
            height_mean=float(height_sum[s] / count[s]),
            height_profile=[round(v, 4) for v in hist],
            vertical_run=float(vrun_sum[s] / vrun_n[s]) if vrun_n[s] else 0.0,
            horizontal_run=float(hrun_sum[s] / hrun_n[s]) if hrun_n[s] else 0.0,
            support_frac=float(supported[s] / count[s]),
            exposure=float(air_face_sum[s] / count[s]),
            self_affinity=float(self_affinity[s]),
            neighbors=sorted(neighbors.get(s, []), key=lambda t: -t[1])[:top_neighbors],
            category_lift=lift)
    return CorpusStats(blocks=blocks, n_builds=n_builds, n_placements=total,
                       corpus=corpus,
                       categories={k: int(v) for k, v in cat_builds.most_common()})


__all__ = ["AIR_SYMBOL", "BlockStats", "CorpusStats", "N_HEIGHT_BINS", "code_of",
           "mine_corpus", "symbol_for", "symbol_index", "symbol_names"]
