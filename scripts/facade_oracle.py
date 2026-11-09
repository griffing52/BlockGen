"""The 2.5D ceiling: real houses reduced to what their facades show, on the bench.

A generator that emits facades and stops can at best reproduce the visible
shell of a real house. This writes that shell for every test-split house as three
control rungs, `real@facades4` (the four sides, what a facade-only model would
hand back), `real@facades5` (the sides plus the roof plan) and `real@facades6`
(every face, so the floor plate comes back too), in the arm cache
format every other arm uses, then runs the benchmark on them beside the usual
controls. The scorecard then says, pillar by pillar, what the bench charges for
"no interior and no roof" before any facade model exists.

    PYTHONPATH=. python scripts/facade_oracle.py            # write arms + run
    PYTHONPATH=. python scripts/facade_oracle.py --no-run   # arms only
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

from blockgen.corpus import facade as fc
from blockgen.curation.houses import save_house_cache
from blockgen.eval.bench import splits

ARMS = {
    "real@facades4": fc.SIDES,
    "real@facades5": fc.SIDES_TOP,
    "real@facades6": fc.ALL_FACES,
}
NAME = "facade oracle: real houses reduced to their visible shell"
NOTE = ("Test-split houses lifted from their face images (blockgen.corpus.facade): "
        "facades4 keeps what the four side faces see, facades5 adds the roof plan, "
        "facades6 adds the underside. The ceiling of any facade-only generator; "
        "nothing inside survives. --n equals the test split so every arm has the same n.")


def write_arms(corpus: str, seed: int, out_dir: str, max_dim: int) -> dict:
    split = splits.load_split(corpus, seed)
    test = [s for s in splits.split_structures(split, "test") if s.occupied_mask.any()]
    paths = {}
    for name, faces in ARMS.items():
        shells = [fc.shell(s, faces) for s in test]
        kept = np.array([sh.occupied_mask.sum() / s.occupied_mask.sum()
                         for sh, s in zip(shells, test)])
        report = {"faces": list(faces), "drawn_from": f"split:{corpus}:test",
                  "blocks_kept_median": float(np.median(kept)),
                  "blocks_kept_mean": float(kept.mean())}
        paths[name] = save_house_cache(shells, max_dim, cache_dir=out_dir,
                                       report=report, name=name.replace("@", "_"))
        print(f"[facade] {name}: {len(shells)} builds, median {np.median(kept):.0%} "
              f"of blocks kept", flush=True)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", default="houses_32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-dim", type=int, default=32)
    ap.add_argument("--out-dir", default="outputs/bench_arms")
    ap.add_argument("--n", type=int, default=399,
                    help="control-arm n; the oracle arms always hold the whole test split")
    ap.add_argument("--tier", default="both")
    ap.add_argument("--no-run", action="store_true", help="write the arms, print the command")
    args = ap.parse_args()

    paths = write_arms(args.corpus, args.seed, args.out_dir, args.max_dim)
    cmd = ["python", "-m", "blockgen.eval.bench", "--tier", args.tier, "--n", str(args.n),
           "--corpus", args.corpus, "--seed", str(args.seed),
           "--name", NAME, "--note", NOTE, "--arms",
           *(f"control/{name}:{path}" for name, path in paths.items())]
    print("[facade] " + shlex.join(cmd), flush=True)
    if args.no_run:
        return 0
    env = {**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", ".") or "."}
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
