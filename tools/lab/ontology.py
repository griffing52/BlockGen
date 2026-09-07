"""The lab's read side for block ontologies (`blockgen.ontology`).

The ontology is the first artifact in this repo that is *authored by a
measurement* and then *read by a model*, which makes it the first one where "is
this any good?" cannot be answered by a scalar. Three questions need eyes:

1. **Is the number right?** Oak stairs are tagged ``roof``; is that because they
   sit at 0.61 of build height, and is 0.61 believable? Every word in the table
   is backed by a raw statistic, and the detail panel shows both.
2. **Does the colour match the block?** The catalog's colours are measured from
   the same texture PNGs the renderer draws, so the swatch route serves those
   exact pixels next to the hex value. A mismatch is a bug you can only see.
3. **What is it costing?** The table goes into a system prompt that is re-sent
   every round. The prompt view renders the *exact* string the agent will send,
   with its size, so the token bill is a number on screen rather than a guess.

Read-only, like every other lab module: it opens JSON the batch pipeline wrote
(``python -m blockgen.ontology``) and never edits a catalog. Rebuilding one is
the CLI's job, and a tool that could silently rewrite the ontology an experiment
was run against would make old runs unreproducible.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ONTOLOGY_DIRS = (Path("data/ontology"), Path("outputs/ontology"))

#: Variants the page can ask for. ``mined`` is the catalog as built; the others
#: are derived on the fly by `blockgen.ontology`, so there is nothing to rebuild
#: when you want to look at the control.
VARIANTS = ("mined", "shuffled", "stats")


def _catalog_paths() -> List[Path]:
    out: List[Path] = []
    for root in ONTOLOGY_DIRS:
        if root.is_dir():
            out.extend(sorted(p for p in root.glob("*.json") if p.is_file()))
    return out


def _id_for(path: Path) -> str:
    return path.stem


def list_catalogs() -> List[dict]:
    """Every ontology JSON on disk, with enough metadata to pick one.

    Reads each file (they are a few hundred KB) rather than stat-ing them: the
    page needs the corpus and build count to say *what* a catalog is, and a
    picker that lists opaque filenames is a picker nobody uses.
    """
    rows: List[dict] = []
    for path in _catalog_paths():
        try:
            catalog = _load(str(path), path.stat().st_mtime)
        except Exception as exc:                            # noqa: BLE001
            rows.append({"id": _id_for(path), "path": str(path), "error":
                         f"{type(exc).__name__}: {exc}"})
            continue
        meta = catalog.meta or {}
        rows.append({
            "id": _id_for(path), "path": str(path), "name": catalog.name,
            "domain": catalog.domain, "n": len(catalog),
            "corpus": meta.get("corpus", ""), "n_builds": meta.get("n_builds"),
            "n_placements": meta.get("n_placements"),
            "n_with_tags": meta.get("n_with_tags"),
            "when": time.strftime("%Y-%m-%d %H:%M",
                                  time.localtime(path.stat().st_mtime)),
        })
    return rows


@lru_cache(maxsize=8)
def _load(path: str, mtime: float) -> Any:
    """Parse one catalog. Keyed by mtime so a rebuild is picked up live -- the
    normal workflow is "re-mine, refresh the page"."""
    from blockgen.ontology.schema import Catalog

    return Catalog.load(path)


def _resolve(catalog_id: Optional[str]) -> Path:
    paths = _catalog_paths()
    if not paths:
        raise FileNotFoundError(
            "no ontology found; build one with `python -m blockgen.ontology`")
    if not catalog_id:
        return paths[0]
    for path in paths:
        if _id_for(path) == catalog_id:
            return path
    raise FileNotFoundError(f"no ontology named {catalog_id!r}")


def get_catalog(catalog_id: Optional[str] = None, variant: str = "mined",
                seed: int = 0) -> Any:
    """One catalog, in one variant. ``shuffled``/``stats`` are derived here so the
    page can show the control without anything being written to disk."""
    path = _resolve(catalog_id)
    catalog = _load(str(path), path.stat().st_mtime)
    if variant in ("", "mined", None):
        return catalog
    if variant == "shuffled":
        return catalog.shuffled(seed)
    if variant == "stats":
        return catalog.subset_by_source(("mined",))
    raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")


def _row(part: Any, fields: Sequence[str]) -> dict:
    return {"id": part.id, "category": part.category, "support": part.support,
            "notes": part.notes,
            "attrs": {name: part.attrs.get(name) for name in fields}}


def overview(catalog_id: Optional[str] = None, variant: str = "mined",
             seed: int = 0) -> dict:
    """Everything the index table needs in one request.

    One payload rather than a row-per-request: 70 parts is ~60 KB of JSON over
    loopback, and a table that sorts and filters client-side cannot do either if
    it only holds the page it fetched.
    """
    catalog = get_catalog(catalog_id, variant, seed)
    prompt = prompt_text(catalog)
    schema = [s.to_json() for s in catalog.schema]
    fields = [s["name"] for s in schema]
    return {
        "catalog": {"id": catalog_id or _id_for(_resolve(catalog_id)),
                    "name": catalog.name, "domain": catalog.domain,
                    "variant": variant, "meta": catalog.meta, "n": len(catalog)},
        "schema": schema,
        "prompt_fields": [s["name"] for s in schema if s["prompt"]],
        "parts": [_row(p, fields) for p in
                  sorted(catalog.parts.values(), key=lambda p: p.id)],
        "prompt": {"chars": len(prompt), "tokens": len(prompt) // 4},
        "sources": sorted({s["source"] for s in schema}),
    }


def part_detail(part_id: str, catalog_id: Optional[str] = None,
                variant: str = "mined", seed: int = 0) -> dict:
    """One part, with its raw evidence and its neighbours resolved for display."""
    catalog = get_catalog(catalog_id, variant, seed)
    part = catalog.get(part_id)
    if part is None:
        raise KeyError(part_id)
    by_source: Dict[str, List[dict]] = {}
    for spec in catalog.schema:
        value = part.attrs.get(spec.name)
        if value is None:
            continue
        by_source.setdefault(spec.source, []).append(
            {"name": spec.name, "label": spec.label, "value": value,
             "kind": spec.kind, "describe": spec.describe, "prompt": spec.prompt})
    neighbors = []
    for entry in (part.attrs.get("stats.neighbors") or []):
        symbol, npmi, share, count = (list(entry) + [None] * 4)[:4]
        neighbors.append({"symbol": symbol, "npmi": npmi, "share": share,
                          "count": count, "in_palette": symbol in _palette_symbols()})
    return {"id": part.id, "category": part.category, "support": part.support,
            "notes": part.notes, "nested": part.nested(), "by_source": by_source,
            "neighbors": neighbors,
            "swatch": f"/api/ontology/swatch/{part.id}"}


@lru_cache(maxsize=1)
def _palette_symbols() -> Dict[str, str]:
    """``corpus symbol -> palette name``, for marking which neighbours are
    sayable in a program. A neighbour outside the palette is real evidence but
    not a name the model may type, and the page should show the difference."""
    try:
        from blockgen.ontology.minecraft import palette_symbols

        return {symbol: name for name, symbol in palette_symbols().items()}
    except Exception:                                        # noqa: BLE001
        return {}


def prompt_text(catalog: Any, fields: Sequence[str] = ()) -> str:
    """The exact string the agent puts in its system prompt."""
    from blockgen.ontology.minecraft import prompt_block

    return prompt_block(catalog, fields)


def prompt_view(catalog_id: Optional[str] = None, variant: str = "mined",
                fields: Sequence[str] = (), seed: int = 0) -> dict:
    catalog = get_catalog(catalog_id, variant, seed)
    text = prompt_text(catalog, fields)
    return {"text": text, "chars": len(text), "tokens": len(text) // 4,
            "fields": list(fields) or
                      [s.name for s in catalog.schema if s.prompt]}


def swatch(part_id: str) -> Tuple[bytes, str]:
    """The block's own texture tile, as PNG bytes.

    Served from ``data/textures/vanilla`` -- the same pixels the catalog measured
    its colour from and the same ones the renderer draws, so "does the swatch
    match the hex" is a real check rather than a decorative one.
    """
    from blockgen.agentic.blockstate import resolve_block
    from blockgen.ontology.appearance import texture_path

    path = texture_path(*resolve_block(part_id))
    if path is None:
        raise FileNotFoundError(f"no texture on disk for {part_id!r}")
    return path.read_bytes(), "image/png"


__all__ = ["ONTOLOGY_DIRS", "VARIANTS", "get_catalog", "list_catalogs", "overview",
           "part_detail", "prompt_text", "prompt_view", "swatch"]
