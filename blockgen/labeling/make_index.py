"""Build the per-structure index the VLM labeler consumes, from a cache manifest.

``blockgen.labeling.vlm_captions`` reads an ``index.json`` mapping each render id
``h{i:05d}`` to metadata (``title``/``category`` for the prompt hint, ``split``
to skip export-filtered rows). The MinecraftACE exporter emits one, but we don't
need a full export just to caption a cache — this derives the same index straight
from a ``*_manifest.json`` (row ``i`` == manifest item ``i`` == render id
``h{i:05d}``), so any rendered cache is immediately labelable.

Usage:
    python -m blockgen.labeling.make_index \
        --manifest data/minecraft/cache/houses_48_manifest.json \
        --out data/minecraftace/houses_48/index.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

_HINT_KEYS = ("title", "category", "corpus", "url")


def index_from_manifest(manifest_path: str) -> Dict[str, dict]:
    """Manifest -> ``{h{i:05d}: {title, category, corpus, url}}`` (cache order)."""
    manifest = json.loads(open(manifest_path).read())
    return {
        f"h{i:05d}": {k: item.get(k, "") for k in _HINT_KEYS}
        for i, item in enumerate(manifest["items"])
    }


def add_materials(index: Dict[str, dict], cache_path: str, k: int = 6) -> None:
    """Attach a top-k dominant-material list to each index entry, from the cache.

    Feeding the ground-truth block histogram to the VLM (idea, ideas.md) lets it name
    materials correctly ("is it a dirt house or a wood house") — a render alone can't
    reliably reveal exact block types. Mutates ``index`` in place.
    """
    from blockgen.curation.houses import load_structures_from_cache
    from blockgen.labeling.templates import dominant_materials
    structs, _ = load_structures_from_cache(cache_path)
    for i, s in enumerate(structs):
        sid = f"h{i:05d}"
        if sid in index:
            index[sid]["materials"] = dominant_materials(s, k=k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="*_manifest.json from the cache")
    parser.add_argument("--out", required=True, help="index.json for the VLM labeler")
    parser.add_argument("--cache", default=None,
                        help="cache .npz; if given, add dominant-material lists for the "
                             "VLM prompt (block-stats grounding)")
    args = parser.parse_args()

    index = index_from_manifest(args.manifest)
    if args.cache:
        add_materials(index, args.cache)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(index, f, indent=1)
    print(f"wrote index for {len(index)} structures -> {args.out}")


if __name__ == "__main__":
    main()
