"""Merge template + VLM labels into the final caption file (+ a labels sidecar).

Produces two outputs from the VLM JSONL (``vlm_captions.py``) + template captions:

* ``<out>``                  ``{id: [4 captions]}`` — the format
  ``blockgen.export.minecraftace --captions`` links in (LegoACE text_count=4).
* ``<out stem>_labels.json`` ``{id: {is_build, category, flag_reason}}`` — the
  coherence filter + category for curation (drop / review flagged builds).

Captions priority: VLM captions first (higher quality), then template captions,
deduped, padded/trimmed to exactly 4. The VLM ``category`` (when present) also
enriches the templated caption so all four agree on the build type.

Usage:
    python -m blockgen.labeling.build_captions \
        --index data/minecraftace/houses_48/index.json \
        --vlm data/minecraft/labels/houses_48_vlm.jsonl \
        --cache data/minecraft/cache/houses_48.npz \
        --out data/minecraft/labels/houses_48_captions.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from blockgen.labeling.templates import template_captions
from blockgen.labeling.vlm_captions import load_done


def _labels_path(out_path: str) -> str:
    if out_path.endswith("_captions.json"):
        return out_path[: -len("_captions.json")] + "_labels.json"
    return out_path.rsplit(".", 1)[0] + "_labels.json"


def build_captions(index_path: str, vlm_jsonl: str | None, out_path: str,
                   max_dim: int = 32, cache: str | None = None, k: int = 4) -> dict:
    with open(index_path) as f:
        index = json.load(f)
    vlm = load_done(vlm_jsonl) if vlm_jsonl else {}

    if cache is not None:
        from blockgen.curation.houses import load_structures_from_cache
        structures, _ = load_structures_from_cache(cache)
    else:
        from blockgen.curation.houses import load_house_structures
        structures, _ = load_house_structures(max_dim=max_dim)

    captions: dict = {}
    labels: dict = {}
    n_with_vlm = 0
    flagged = Counter()
    for i, s in enumerate(structures):
        sid = f"h{i:05d}"
        if index.get(sid, {}).get("split") == "filtered":
            continue
        rec = vlm.get(sid, {})
        vlm_caps = list(rec.get("captions", []))
        if vlm_caps:
            n_with_vlm += 1
        # Let the VLM's category drive the templated caption too, when we have it.
        if rec.get("category"):
            s.metadata["category"] = rec["category"]

        # Guaranteed short-tag caption first (how users actually type prompts), then
        # the VLM's detailed captions, then template fallbacks — mixed granularity.
        short = [rec["short_tag"]] if rec.get("short_tag") else []
        pool = short + vlm_caps + template_captions(s, k=k, seed=i)
        seen = set()
        merged = []
        for c in pool:
            key = c.strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged.append(c.strip())
        while len(merged) < k:
            merged.append(merged[len(merged) % max(1, len(merged))])
        captions[sid] = merged[:k]

        if rec:
            is_build = rec.get("is_build", True)
            labels[sid] = {"is_build": is_build,
                           "category": rec.get("category", ""),
                           "flag_reason": rec.get("flag_reason", "")}
            if not is_build:
                flagged[rec.get("flag_reason", "") or "unspecified"] += 1

    with open(out_path, "w") as f:
        json.dump(captions, f, indent=1)
    labels_path = _labels_path(out_path)
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=1)
    print(f"wrote {len(captions)} caption sets ({n_with_vlm} with VLM) -> {out_path}")
    print(f"wrote {len(labels)} labels -> {labels_path}; "
          f"flagged not-a-build: {sum(flagged.values())} "
          f"{dict(flagged.most_common(6))}")
    return captions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--vlm", default=None, help="VLM labels JSONL (optional)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-dim", type=int, default=32,
                        help="houses cache max_dim (ignored if --cache given)")
    parser.add_argument("--cache", default=None,
                        help="any structure cache .npz (whole-corpus etc.)")
    args = parser.parse_args()
    build_captions(args.index, args.vlm, args.out, max_dim=args.max_dim, cache=args.cache)


if __name__ == "__main__":
    main()
