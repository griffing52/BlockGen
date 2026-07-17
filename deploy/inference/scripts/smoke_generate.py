"""Generate from a registered model without Minecraft in the loop.

This is the check that a registry entry is actually servable end-to-end: weights
load, the vocab agrees with them, tokens stream, and blocks decode to real modern
block states. It also prints the stats worth comparing against the training run's
``rows.json`` -- for a rebuilt vocabulary, plausible block counts and a sane palette
are the only evidence the *patterns* are right (sizes matching proves nothing).

Usage:
    python scripts/smoke_generate.py --model native_bpe --seed 0
    python scripts/smoke_generate.py --model text_cond --prompt "a small stone house"
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from blockgen_server.backends import GenerateRequest
from blockgen_server.registry import Registry

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--list", action="store_true", help="list models and exit")
    args = ap.parse_args()

    reg = Registry(HERE / "models.json", REPO)
    if args.list:
        for row in reg.describe():
            mark = "*" if row["default"] else " "
            status = "ok" if row["available"] else row.get("unavailable_reason", "?")
            print(f"{mark} {row['name']:16s} {row['kind']:14s} [{status}]")
            print(f"    {row['description']}")
        return

    t0 = time.time()
    backend = reg.get(args.model)
    print(f"loaded {backend.spec.name} in {time.time() - t0:.1f}s "
          f"(kind={backend.spec.kind}, supports_text={backend.supports_text})")
    if args.prompt and not backend.supports_text:
        print(f"WARNING: {backend.spec.name} ignores prompts; sampling unconditionally")

    req = GenerateRequest(prompt=args.prompt, seed=args.seed,
                          temperature=args.temperature, top_k=args.top_k,
                          cfg_scale=args.cfg, max_tokens=args.max_tokens)
    t0 = time.time()
    blocks, batches = [], 0
    for batch in backend.stream(req):
        blocks.extend(batch)
        batches += 1
        if batches == 1:
            print(f"first blocks after {time.time() - t0:.2f}s")
    dt = time.time() - t0

    if not blocks:
        print("NO BLOCKS GENERATED")
        sys.exit(1)

    xs = [b.x for b in blocks]; ys = [b.y for b in blocks]; zs = [b.z for b in blocks]
    palette = collections.Counter(b.state for b in blocks)
    print(f"\n{len(blocks)} blocks in {dt:.1f}s ({len(blocks)/dt:.0f} blocks/s), "
          f"{batches} decode steps")
    print(f"bbox: x[{min(xs)}..{max(xs)}] y[{min(ys)}..{max(ys)}] z[{min(zs)}..{max(zs)}]")
    print(f"distinct states: {len(palette)}")
    print("top palette:")
    for state, n in palette.most_common(8):
        print(f"  {n:6d}  {state}")


if __name__ == "__main__":
    main()
