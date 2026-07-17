"""Is the perceptual metric trustworthy? Test it where we already know the answer.

A metric that cannot see damage we have already photographed (F16: canon-16 deletes a
pagoda's entire roof, 86.5% of blocks) has no business adjudicating anything subtler.

Four probes against one fixed real-native reference set:
  1. real native (held-out)  -> CMMD should be ~0. Floor / noise level.
  2. real @ canon-16         -> MUST be much higher. This is F16, quantified.
  3. real @ canon-8          -> higher still. Monotone in damage = the metric ranks.
  4. shuffled-material real  -> geometry intact, palette destroyed. Isolates whether
                                the metric sees materials at all (nn_iou barely does).

If 1 < 2 < 3 with a clear margin, the metric is usable for T20's arms.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from blockgen.curation.houses import load_house_structures
from blockgen.eval.perceptual import (DEFAULT_PROMPT, DEFAULT_VIEWS, clip_image_features,
                                      clip_text_features, cmmd, render_views)
from blockgen.experiments_gen import canonicalize
from blockgen.utils.data import Structure


def shuffle_materials(s: Structure, rng) -> Structure:
    """Keep the shape, permute which block type sits where. Palette destroyed."""
    c = s.crop_to_non_air()
    occ = c.block_ids != c.air_block_id
    bi, bd = c.block_ids.copy(), c.block_data.copy()
    idx = np.argwhere(occ)
    vals = [(bi[x, y, z], bd[x, y, z]) for x, y, z in idx]
    rng.shuffle(vals)
    for (x, y, z), (a, b) in zip(idx, vals):
        bi[x, y, z], bd[x, y, z] = a, b
    return Structure(block_ids=bi, block_data=bd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32, help="builds per probe set")
    ap.add_argument("--px", type=int, default=224)
    ap.add_argument("--views", type=int, default=2, help="views per build")
    ap.add_argument("--out", default="outputs/analysis/perceptual_validation.json")
    args = ap.parse_args()

    views = DEFAULT_VIEWS[: args.views]
    rng = np.random.default_rng(0)
    structs, _ = load_house_structures(32)
    structs = [s.crop_to_non_air() for s in structs]
    # only builds the decimator actually touches, so probes 2/3 mean something
    pool = [s for s in structs if max(s.shape) > 16 and 300 <= int((s.block_ids != s.air_block_id).sum()) <= 1500]
    rng.shuffle(pool)
    ref = pool[: args.n]                  # reference (real, native)
    probe = pool[args.n: 2 * args.n]      # disjoint held-out real, native

    print(f"reference {len(ref)} / probe {len(probe)} builds, {len(views)} views, {args.px}px\n")

    def feats(name, sset):
        print(f"  [{name}]", flush=True)
        return clip_image_features(render_views(sset, views, args.px, verbose=False))

    f_ref = feats("reference: real native", ref)
    probes = {
        "real native (held-out)": probe,
        "real @ canon-16": canonicalize(probe, 16),
        "real @ canon-8": canonicalize(probe, 8),
        "real, materials shuffled": [shuffle_materials(s, rng) for s in probe],
    }
    t = clip_text_features([DEFAULT_PROMPT])

    rows = {}
    for name, sset in probes.items():
        f = feats(name, sset)
        rows[name] = {"cmmd": round(cmmd(f, f_ref), 3),
                      "clip_text": round(float((f @ t.T).mean()), 4)}

    print(f"\n{'probe':<28}{'CMMD vs real':>14}{'CLIP text':>12}")
    print("-" * 54)
    for k, v in rows.items():
        print(f"{k:<28}{v['cmmd']:>14.3f}{v['clip_text']:>12.4f}")
    print("-" * 54)
    print("CMMD: lower = looks more like real native houses. CLIP text: higher = more house-like.")

    ok = rows["real native (held-out)"]["cmmd"] < rows["real @ canon-16"]["cmmd"] < rows["real @ canon-8"]["cmmd"]
    print(f"\nmonotone in decimation damage (native < canon16 < canon8): {ok}")
    print("  -> metric ranks known damage correctly; usable for T20's arms" if ok
          else "  -> METRIC FAILS its sanity check. Do not use it to adjudicate anything.")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"n": args.n, "views": len(views), "px": args.px,
               "prompt": DEFAULT_PROMPT, "rows": rows, "monotone": ok},
              open(args.out, "w"), indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
