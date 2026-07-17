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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="*_manifest.json from the cache")
    parser.add_argument("--out", required=True, help="index.json for the VLM labeler")
    args = parser.parse_args()

    index = index_from_manifest(args.manifest)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(index, f, indent=1)
    print(f"wrote index for {len(index)} structures -> {args.out}")


if __name__ == "__main__":
    main()
