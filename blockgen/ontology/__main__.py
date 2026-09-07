"""Build the block ontology from a corpus cache.

    python -m blockgen.ontology                         # houses_32 -> data/ontology/
    python -m blockgen.ontology --corpus data/minecraft/cache/all_32.npz
    python -m blockgen.ontology --print                 # show the prompt table
"""

from __future__ import annotations

import argparse
from pathlib import Path

from blockgen.ontology.minecraft import (DEFAULT_CORPUS, DEFAULT_OUT, MIN_SUPPORT,
                                         mine_and_build, prompt_block)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m blockgen.ontology",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS, help="structure cache (.npz)")
    ap.add_argument("--out", default=None, help="output JSON (default: data/ontology/)")
    ap.add_argument("--limit", type=int, default=0, help="mine only the first N builds")
    ap.add_argument("--min-support", type=int, default=MIN_SUPPORT,
                    help="placements required before a block's tags are trusted")
    ap.add_argument("--print", action="store_true", dest="show",
                    help="print the prompt table and its size")
    args = ap.parse_args(argv)

    catalog = mine_and_build(args.corpus, limit=args.limit,
                             min_support=args.min_support, progress_every=1000)
    stem = Path(args.corpus).stem + (f"_first{args.limit}" if args.limit else "")
    out = Path(args.out) if args.out else DEFAULT_OUT / f"minecraft_{stem}.json"
    catalog.save(out)

    meta = catalog.meta
    table = prompt_block(catalog)
    if args.show:
        print(table)
    print(f"[ontology] {len(catalog)} parts, {meta['n_with_tags']} with mined tags, "
          f"from {meta['n_builds']} builds / {meta['n_placements']:,} placements")
    print(f"[ontology] prompt block: {len(table):,} chars (~{len(table) // 4:,} tokens)")
    print(f"[ontology] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
