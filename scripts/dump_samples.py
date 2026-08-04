"""Convert generated samples into the standard structure-cache `.npz`.

`python -m blockgen.eval.bench` deliberately consumes one input format and one
only: the cache written by `blockgen.curation.houses.save_house_cache`. That
keeps the benchmark decoupled from every generator -- it never imports a model,
loads a checkpoint, or calls a provider API -- so it survives changes to any
track. The agentic track already writes this format
(`blockgen.agentic.report.save_structures`); the neural tracks do not, and this
script closes that gap.

Accepted inputs (exactly one):

    --npy-dir DIR       a directory of per-sample .npy block-id grids
                        (the `sample-*.npy` layout)
    --keyed-npz FILE    one .npz holding `s0`, `s1`, ... block-id grids
                        (what `scripts/train_attach_corpus.py` writes)
    --cache FILE        an existing structure cache, to subset or resample

Examples:

    python scripts/dump_samples.py --keyed-npz outputs/run_x/arm/samples.npz \\
        --out-dir outputs/run_x/arm --name arm --max-dim 32
    python scripts/dump_samples.py --cache data/minecraft/cache/houses_32.npz \\
        --n 64 --seed 0 --out-dir /tmp/probe --name subset --max-dim 32
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from blockgen.curation.houses import load_structures_from_cache, save_house_cache
from blockgen.utils.data import Structure


def _as_structure(ids: np.ndarray, data: np.ndarray | None = None) -> Structure:
    ids = np.asarray(ids)
    if ids.ndim != 3:
        raise ValueError(f"expected a 3-D block-id grid, got shape {ids.shape}")
    if data is None:
        data = np.zeros_like(ids)
    return Structure(block_ids=ids.astype(np.int32),
                     block_data=np.asarray(data).astype(np.int32))


def from_npy_dir(path: Path, pattern: str = "*.npy") -> List[Structure]:
    out: List[Structure] = []
    for f in sorted(path.glob(pattern)):
        arr = np.load(f, allow_pickle=False)
        out.append(_as_structure(arr))
    if not out:
        raise SystemExit(f"no files matching {pattern!r} in {path}")
    return out


def from_keyed_npz(path: Path) -> List[Structure]:
    """`{"s0": grid, "s1": grid, ...}`, ordered numerically not lexically."""
    with np.load(path, allow_pickle=False) as z:
        keys = sorted(z.files, key=lambda k: (len(k), k))
        if not keys:
            raise SystemExit(f"{path} contains no arrays")
        # Paired block_ids/block_data layout, if present.
        if "block_ids" in z.files and "block_data" in z.files:
            ids, data = z["block_ids"], z["block_data"]
            return [_as_structure(i, d) for i, d in zip(ids, data)]
        return [_as_structure(z[k]) for k in keys]


def from_cache(path: Path) -> List[Structure]:
    structs, _ = load_structures_from_cache(str(path))
    return structs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--npy-dir")
    src.add_argument("--keyed-npz")
    src.add_argument("--cache")
    ap.add_argument("--pattern", default="*.npy", help="glob for --npy-dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="samples")
    ap.add_argument("--max-dim", type=int, default=32,
                    help="recorded in the filename and manifest; no resampling")
    ap.add_argument("--n", type=int, default=None, help="cap the number of samples")
    ap.add_argument("--seed", type=int, default=0, help="seed for the --n subset")
    ap.add_argument("--crop", action="store_true",
                    help="crop each structure to its non-air bounding box")
    args = ap.parse_args()

    if args.npy_dir:
        structs = from_npy_dir(Path(args.npy_dir), args.pattern)
        source = args.npy_dir
    elif args.keyed_npz:
        structs = from_keyed_npz(Path(args.keyed_npz))
        source = args.keyed_npz
    else:
        structs = from_cache(Path(args.cache))
        source = args.cache

    if args.crop:
        structs = [s.crop_to_non_air() for s in structs]

    n_before = len(structs)
    if args.n is not None and args.n < len(structs):
        idx = np.random.default_rng(args.seed).permutation(len(structs))[:args.n]
        structs = [structs[i] for i in sorted(idx.tolist())]

    n_empty = sum(1 for s in structs if not s.occupied_mask.any())
    if n_empty:
        print(f"note: {n_empty}/{len(structs)} samples are empty "
              f"(kept; the benchmark reports them as n_empty)")

    path = save_house_cache(
        structs, args.max_dim, cache_dir=args.out_dir, name=args.name,
        report={"source": source, "n_source": n_before, "n_written": len(structs),
                "n_empty": n_empty, "seed": args.seed, "cropped": bool(args.crop)})
    print(f"bench with:\n  python -m blockgen.eval.bench --arms ar/{args.name}:{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
