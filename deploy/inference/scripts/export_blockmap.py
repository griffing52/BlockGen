"""Validate the legacy->modern block map against a real Minecraft block registry.

The map in ``blockgen_server.blockmap`` is hand-built, so it needs checking against
ground truth rather than eyeballing. Minecraft's own server jar can emit that
ground truth::

    java -DbundlerMainClass=net.minecraft.data.Main -jar server.jar --reports
    # -> generated/reports/blocks.json  (every block + its legal property values)

This script maps every ``(block_id, block_data)`` pair in a model's vocabulary and
asserts each result is a block that actually exists, with properties that actually
exist and values that are actually legal. A mapping that names a real block but
the *wrong* one is not detectable here — that is what the ``_BY_NAME`` comments
in ``blockmap.py`` are for — but this catches every typo, rename, and stale id.

**Pass every vocabulary you serve, not just one.** Vocabularies built from different
corpora use different ``block_data`` values for the same blocks — the exported
``houses_32_bpe`` vocab and the rebuilt ``native_bpe`` vocab share 352 of 422 pairs
and disagree on 70. Validating only the first left ``(26, 10)`` -> ``minecraft:bed``
(not a real block) to be found by Minecraft at runtime instead. With no ``--vocab``
the script checks every vocab named in ``models.json``, which is the right default.

Usage:
    python scripts/export_blockmap.py --report path/to/blocks.json      # all models
    python scripts/export_blockmap.py --report ... --vocab a.json --vocab b.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from blockgen_server.blockmap import is_fallback, legacy_display_name, modern_state


def parse_state(state: str) -> Tuple[str, Dict[str, str]]:
    """``'minecraft:oak_slab[type=double]'`` -> ``('minecraft:oak_slab', {...})``."""
    m = re.match(r"^([^\[]+)(?:\[(.*)\])?$", state)
    if not m:
        raise ValueError(f"unparseable state: {state!r}")
    name, props = m.group(1), {}
    if m.group(2):
        for part in m.group(2).split(","):
            k, _, v = part.partition("=")
            props[k.strip()] = v.strip()
    return name, props


def validate(state: str, registry: dict) -> List[str]:
    """Return a list of problems with *state*; empty means valid."""
    name, props = parse_state(state)
    entry = registry.get(name)
    if entry is None:
        return [f"no such block: {name}"]
    legal = entry.get("properties", {})
    problems = []
    for k, v in props.items():
        if k not in legal:
            problems.append(f"{name}: no property {k!r} (has {sorted(legal)})")
        elif v not in legal[k]:
            problems.append(f"{name}: {k}={v!r} not in {legal[k]}")
    return problems


def vocabs_from_models(models_json: str) -> List[str]:
    """Every vocab file referenced by models.json, resolved against the repo root."""
    root = os.path.join(os.path.dirname(models_json), "..", "..")
    blob = json.load(open(models_json))
    out = []
    for entry in blob["models"].values():
        for key in ("piece_vocab", "block_vocab"):
            rel = entry.get(key)
            if rel:
                p = rel if os.path.isabs(rel) else os.path.join(root, rel)
                if os.path.exists(p):
                    out.append(p)
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True, help="vanilla generated blocks.json")
    ap.add_argument("--vocab", action="append", default=None,
                    help="vocab json; repeatable. Default: every vocab in models.json")
    ap.add_argument("--models", default=os.path.join(os.path.dirname(__file__), "..",
                                                     "models.json"))
    ap.add_argument("--out", default=None, help="write the resolved map here")
    args = ap.parse_args()

    registry = json.load(open(args.report))
    vocab_paths = args.vocab or vocabs_from_models(args.models)
    if not vocab_paths:
        print("no vocabularies to check")
        sys.exit(1)

    table, problems, fallbacks = {}, [], []
    pairs = set()
    for vp in vocab_paths:
        blob = json.load(open(vp))
        vpairs = [tuple(p) for p in blob["block_index_to_pair"]]
        pairs.update(vpairs)
        print(f"  {len(vpairs):4d} pairs  {os.path.relpath(vp)}")

    for bid, data in sorted(pairs):
        state = modern_state(bid, data)
        table[f"{bid}:{data}"] = state
        problems.extend([f"{bid}:{data} ({legacy_display_name(bid, data)}) -> {p}"
                         for p in validate(state, registry)])
        if is_fallback(bid, data):
            fallbacks.append(f"{bid}:{data}")

    print(f"\nvocabularies     : {len(vocab_paths)}")
    print(f"distinct pairs   : {len(pairs)}")
    print(f"distinct blocks  : {len(set(v.split('[')[0] for v in table.values()))}")
    print(f"fallback to stone: {len(fallbacks)} {fallbacks or ''}")
    print(f"invalid states   : {len(problems)}")
    for p in problems:
        print(f"  ! {p}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"_comment": "generated by scripts/export_blockmap.py",
                       "map": table}, f, indent=1, sort_keys=True)
        print(f"-> {args.out}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
