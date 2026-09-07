"""Curation-parameter playground: what a threshold is actually throwing away.

Why this exists
---------------
`curation.houses.build_house_dataset` bakes nine numeric gates into a cache that
takes minutes to rebuild, and the only feedback it ever gives is one `Counter`
printed at the end of the run. Nobody tunes a threshold that way -- they accept
whatever shipped, which is how `min_largest_component_frac = 0.55` has gone
unexamined since T3b. This runs the *same* gates as a preview: the real
`quality_filter` over a real `Curator`, nothing written to disk, plus a handful
of example `build_id`s per drop reason so "is 0.55 discarding good houses?" gets
answered by looking at six builds instead of by argument.

Two things worth knowing before reading a number off this page
--------------------------------------------------------------
**The corpora are already curated.** `houses_32` is the *output* of these gates,
so at the shipped thresholds essentially nothing drops. The preview's honest
question is therefore marginal: what does *tightening* a gate cost, on builds
that survived it once. The pooled pre-filter candidates (3,493 for `houses_32`)
are not cached anywhere and re-pooling them means re-reading three corpora, far
too slow to sit behind a slider; the manifest's shipped drop counts are returned
as `shipped_drops` so the page can show the real pass against the marginal one.

**The sample is strided, not the first N.** Corpus caches are written
corpus-by-corpus, so `structures[:400]` of `houses_32` is 400 GrabCraft builds
and 0 from 3dcraft/text2mc -- and the enclosed-air gate is *only* applied to the
latter two, so a contiguous sample makes the single largest drop reason
(`no_interior`, 675 of 826 at build time) structurally impossible to see. This is
the same contiguity bug T23e records for the ladder's noise floor. A stride keeps
the corpus mix of the whole cache and the original indices, so the `build_id`s
still address the same builds the rest of the lab shows.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Sequence, Tuple

from blockgen.curation.curate import Curator, compute_features
from blockgen.curation.houses import load_house_structures, quality_filter
from blockgen.utils.data import Structure

#: Gate order matches the rule order inside `quality_filter`, which matters:
#: the filter stops at the *first* failing rule, so a build that is both too
#: small and fragmented is counted only under `too_few_blocks`. The page shows
#: the gates in firing order so that shadowing is legible rather than surprising.
GATE_ORDER: Tuple[str, ...] = (
    "min_blocks", "min_height", "min_footprint", "min_density", "max_density",
    "min_block_types", "max_dominant_frac", "min_largest_component_frac",
    "min_enclosed_air",
)

_SIG = inspect.signature(quality_filter)
#: Read from the signature rather than retyped, so a change to `quality_filter`
#: cannot silently desync this page from the pipeline it is previewing.
DEFAULTS: Dict[str, Any] = {k: _SIG.parameters[k].default for k in GATE_ORDER}
#: One deliberate divergence: `quality_filter` defaults `min_enclosed_air=0`
#: (gate off) but every shipped cache was built through `build_house_dataset`,
#: which passes 8. Previewing with 0 would hide the gate that does most of the
#: work, so the pipeline's value is the one pre-filled here.
DEFAULTS["min_enclosed_air"] = 8

#: Slider bounds for the page. Ranges bracket the shipped value generously in
#: both directions -- the point of the tool is to walk a threshold until builds
#: you would have kept start falling out, which needs room on both sides.
PARAM_SPEC: List[Dict[str, Any]] = [
    {"name": "min_blocks", "label": "min blocks", "min": 0, "max": 600, "step": 10,
     "reason": "too_few_blocks", "help": "Fragments and single walls."},
    {"name": "min_height", "label": "min height", "min": 1, "max": 20, "step": 1,
     "reason": "too_flat", "help": "Floors, roads, pixel art lying flat."},
    {"name": "min_footprint", "label": "min footprint", "min": 0, "max": 200, "step": 4,
     "reason": "footprint_too_small", "help": "x·z area of the cropped bbox."},
    {"name": "min_density", "label": "min density", "min": 0.0, "max": 0.4, "step": 0.005,
     "reason": "too_sparse", "help": "Blocks / bbox volume. Low = scaffolding, trees."},
    {"name": "max_density", "label": "max density", "min": 0.3, "max": 1.0, "step": 0.01,
     "reason": "solid_blob", "help": "High = a solid lump, not a building."},
    {"name": "min_block_types", "label": "min block types", "min": 1, "max": 12, "step": 1,
     "reason": "too_few_materials", "help": "Distinct resource names."},
    {"name": "max_dominant_frac", "label": "max dominant frac", "min": 0.3, "max": 1.0,
     "step": 0.01, "reason": "single_material_blob",
     "help": "Share held by the commonest block."},
    {"name": "min_largest_component_frac", "label": "min largest component",
     "min": 0.0, "max": 1.0, "step": 0.01, "reason": "fragmented",
     "help": "6-connected largest component / all blocks."},
    {"name": "min_enclosed_air", "label": "min enclosed air", "min": 0, "max": 200, "step": 1,
     "reason": "no_interior",
     "help": "Interior air voxels unreachable from the bbox face. Applied to "
             "3dcraft/text2mc only -- GrabCraft is category-labeled ground truth "
             "and 17% of its real houses have open interiors."},
]

#: Human-readable gloss per drop reason, so the page never shows a bare key.
REASON_HELP: Dict[str, str] = {p["reason"]: p["help"] for p in PARAM_SPEC}

#: Corpora the enclosed-air gate applies to, mirrored from `quality_filter` for
#: display; not tunable here because it is a curation policy, not a threshold.
ENCLOSED_AIR_CORPORA: Tuple[str, ...] = tuple(
    _SIG.parameters["enclosed_air_corpora"].default)

_MAX_EXAMPLES = 8

# --- loading ---------------------------------------------------------------
# Caching decision: features, not structures, are the cost here -- `compute_features`
# runs a pure-Python 6-connected flood fill per build (~1.5 ms each, 4 s for all
# of houses_32) while loading the .npz takes 0.13 s. So the cache is keyed by
# dataset and holds {global_index: feature_row}, filled lazily and never evicted:
# a slider drag re-runs only the threshold comparisons, which are microseconds.
# Structures are held alongside because the enclosed-air gate needs the voxels.
# One process, one researcher, a few thousand rows -- bounded by construction.
_STRUCTS: Dict[str, List[Structure]] = {}
_FEATURES: Dict[str, Dict[int, dict]] = {}
_MANIFESTS: Dict[str, dict] = {}


def _corpus_name(dataset: str) -> str:
    """`corpus:houses_32`, `corpus/houses_32` and `houses_32` all mean one cache."""
    return dataset.split(":")[-1].split("/")[-1]


def _load(dataset: str) -> Tuple[List[Structure], dict]:
    """Structures for a dataset id, preferring the lab's own catalog.

    Going through `catalog.load_builds` when it is importable keeps indices --
    and therefore `build_id`s and thumbnails -- identical to what `/api/builds`
    serves. The direct corpus load is the fallback so this module stays testable
    on its own (`python -c "from tools.lab import gates; gates.preview()"`).
    """
    if dataset in _STRUCTS:
        return _STRUCTS[dataset], _MANIFESTS.get(dataset, {})

    structures: Optional[List[Structure]] = None
    manifest: dict = {}
    try:
        from tools.lab import catalog  # imported lazily: written in parallel
        structures = list(catalog.load_builds(dataset))
    except Exception:
        structures = None
    name = _corpus_name(dataset)
    if name.startswith("houses_"):
        # Loaded even when the catalog already gave us the builds: the manifest
        # carries the drop counts from the pass that *built* this cache, which is
        # the only honest reference point for a preview run over its output.
        try:
            cached, manifest = load_house_structures(int(name.split("_")[-1]))
            structures = structures or list(cached)
        except (FileNotFoundError, ValueError):
            if not structures:
                raise
    if not structures:
        raise FileNotFoundError(f"no loader for dataset {dataset!r}")

    _STRUCTS[dataset] = structures
    _FEATURES.setdefault(dataset, {})
    _MANIFESTS[dataset] = manifest or {}
    return structures, _MANIFESTS[dataset]


def _features_for(dataset: str, structures: Sequence[Structure],
                  picks: Sequence[int]) -> List[dict]:
    """Feature rows for `picks`, computing only the ones not already memoized."""
    memo = _FEATURES.setdefault(dataset, {})
    missing = [g for g in picks if g not in memo]
    if missing:
        for g, row in zip(missing, compute_features([structures[g] for g in missing])):
            memo[g] = row
    return [memo[g] for g in picks]


def _sample(n_total: int, limit: int) -> List[int]:
    """Evenly strided indices -- see the module docstring on why not `[:limit]`."""
    if limit <= 0 or limit >= n_total:
        return list(range(n_total))
    step = n_total / float(limit)
    return sorted({min(n_total - 1, int(i * step)) for i in range(limit)})


def _coerce(name: str, value: Any) -> Any:
    """Query strings arrive as text; keep each gate's declared type."""
    default = DEFAULTS[name]
    if value is None or value == "":
        return default
    try:
        return float(value) if isinstance(default, float) else int(float(value))
    except (TypeError, ValueError):
        return default


def _build_id(dataset: str, index: int) -> str:
    try:
        from tools.lab import catalog
        return catalog.make_build_id(dataset, index)
    except Exception:
        return f"{dataset}:{index}"  # the contract's own definition


def _empty(dataset: str, params: Dict[str, Any], note: str) -> Dict[str, Any]:
    """A missing corpus is the normal state on a fresh clone, not an error."""
    return {"dataset": dataset, "kept": 0, "dropped": 0, "reasons": {}, "examples": {},
            "params": params, "n_scanned": 0, "n_total": 0, "sampled": "none",
            "defaults": dict(DEFAULTS), "spec": PARAM_SPEC, "reason_help": REASON_HELP,
            "enclosed_air_corpora": list(ENCLOSED_AIR_CORPORA),
            "shipped_drops": {}, "note": note}


# --- the preview ------------------------------------------------------------
def preview(dataset: str = "houses_32", limit: int = 400, **params: Any) -> Dict[str, Any]:
    """Run `quality_filter` at the given thresholds without building anything.

    Returns kept/dropped counts, the per-reason breakdown, and up to
    `_MAX_EXAMPLES` `build_id`s per reason -- the examples are the product here;
    the counts alone would just be a slower version of the build log.
    """
    gates = {k: _coerce(k, params.get(k, DEFAULTS[k])) for k in GATE_ORDER}
    try:
        limit = max(0, int(limit))
    except (TypeError, ValueError):
        limit = 400

    try:
        structures, manifest = _load(dataset)
    except FileNotFoundError as exc:
        return _empty(dataset, gates, str(exc))
    except Exception as exc:  # a corrupt cache must not take the page down
        return _empty(dataset, gates, f"could not load {dataset!r}: {exc}")
    if not structures:
        return _empty(dataset, gates, f"{dataset!r} contains no builds.")

    picks = _sample(len(structures), limit)
    rows = _features_for(dataset, structures, picks)
    sel = [structures[g] for g in picks]

    # A fresh Curator per call: `mark_remove` mutates `decisions`, and a shared
    # one would accumulate the marks of every threshold ever previewed. Nothing
    # here calls `save_decisions`, so no curation state reaches disk.
    cur = Curator(structures=sel, features=rows)
    try:
        _, reasons = quality_filter(cur, **gates)
    except Exception as exc:
        return _empty(dataset, gates, f"quality_filter failed: {exc}")

    # `quality_filter` reports only counts, so the per-build attribution is read
    # back out of the decisions it wrote, keyed the way `Curator.mark` keys them.
    local_by_key: Dict[str, int] = {}
    for local, s in enumerate(sel):
        local_by_key.setdefault(s.source_path or f"#{local}", local)
    examples: Dict[str, List[str]] = {}
    attributed = 0
    for key, decision in cur.decisions.items():
        local = local_by_key.get(key)
        if local is None or decision.get("decision") != "remove":
            continue
        attributed += 1
        bucket = examples.setdefault(decision.get("reason") or "?", [])
        if len(bucket) < _MAX_EXAMPLES:
            bucket.append(_build_id(dataset, picks[local]))

    dropped = int(sum(reasons.values()))
    notes: List[str] = []
    if picks and len(picks) < len(structures):
        notes.append(f"strided sample of {len(picks)} across all {len(structures)} "
                     f"builds (not the first {len(picks)}, which would be one corpus)")
    if manifest.get("report", {}).get("quality_drops"):
        pooled = manifest["report"].get("pooled")
        notes.append(f"{_corpus_name(dataset)} is already the output of these gates "
                     f"(pooled {pooled} → {manifest['report'].get('final')}), so drops "
                     f"here are what *tightening* a threshold would cost")
    if attributed < dropped:
        # Two builds sharing a source_path collapse to one decisions key. Counts
        # stay right (they come from the Counter); only the examples thin out.
        notes.append(f"{dropped - attributed} drops share a source path with another "
                     f"build and have no example thumbnail")

    return {
        "dataset": dataset,
        "kept": len(picks) - dropped,
        "dropped": dropped,
        "reasons": {k: int(v) for k, v in reasons.most_common()},
        "examples": examples,
        "params": gates,
        "n_scanned": len(picks),
        "n_total": len(structures),
        "sampled": "all" if len(picks) == len(structures) else "stride",
        "defaults": dict(DEFAULTS),
        "spec": PARAM_SPEC,
        "reason_help": REASON_HELP,
        "enclosed_air_corpora": list(ENCLOSED_AIR_CORPORA),
        "shipped_drops": dict(manifest.get("report", {}).get("quality_drops", {})),
        "note": "; ".join(notes) or None,
    }


if __name__ == "__main__":  # smoke check: `python -m tools.lab.gates`
    import json
    import sys

    out = preview(*(sys.argv[1:2] or ["houses_32"]))
    out.pop("spec", None)
    print(json.dumps(out, indent=2)[:4000])
    strict = preview(min_blocks=300, min_largest_component_frac=0.95, max_dominant_frac=0.6)
    print("strict:", strict["kept"], "kept /", strict["dropped"], "dropped",
          strict["reasons"])
    print("unknown dataset:", preview(dataset="nope_1")["note"])
