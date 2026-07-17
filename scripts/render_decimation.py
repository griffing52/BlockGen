"""The money shot for T17: the SAME real houses, native 32 vs canon-16.

No model involved. If the canon-16 row looks as bad as the model's samples, the
decimator is the ceiling and the model is faithful. Real textures.
"""
import argparse, os
import numpy as np
from blockgen.curation.houses import load_house_structures
from blockgen.experiments_gen import canonicalize
from blockgen.renderer.grid import save_grid


def occ(s):
    return int((s.block_ids != s.air_block_id).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--out", default="outputs/figures/decimation_native_vs_canon16.png")
    args = ap.parse_args()

    structs, _ = load_house_structures(32)
    structs = [s.crop_to_non_air() for s in structs]
    # pick recognisable houses that the decimator actually touches
    cands = [s for s in structs if max(s.shape) > 16 and 400 <= occ(s) <= 1400]
    print(f"{len(cands)} candidates (max_dim>16, 400-1400 blocks)")
    pick = cands[:args.cols]

    canon = canonicalize(pick, 16)
    tiles = list(pick) + list(canon)          # row 1 = native, row 2 = canon-16
    for a, b in zip(pick, canon):
        print(f"  {occ(a):5d} blocks (native {max(a.shape)}³) -> {occ(b):4d} blocks "
              f"(canon-16)   retained {occ(b)/occ(a)*100:4.1f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_grid(tiles, args.out, cols=args.cols, tile_px=320)
    print(f"\ntop row = real native 32, bottom row = SAME builds at canon-16 -> {args.out}")


if __name__ == "__main__":
    main()
