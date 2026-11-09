"""Facade baselines on the bench: the floor of the 2D-first track, as arms.

Fits `blockgen.corpus.facade_baselines` on the houses_32 train split, writes each
baseline as an arm cache under `outputs/bench_arms/`, and runs the benchmark on
them beside the six-face oracle rung (`real@facades6`, the ceiling of any
facade generator) and the usual controls. Extra arms (a trained facade model's
samples) go on the same card with `--arm track/name=path.npz`.

    PYTHONPATH=. python scripts/facade_baselines.py [--n 399] [--faces all|sides] [--no-run]
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

from blockgen.corpus import facade_baselines as fb
from blockgen.corpus.shards import FACE_SETS
from blockgen.curation.houses import save_house_cache
from blockgen.eval.bench import splits

NAME = "facade baselines: unigram, markov and patchwork faces lifted into real boxes"
NOTE = ("Non-learned facade generators fitted on the train split (blockgen.corpus.facade_baselines), "
        "scored beside real@facades6, the ceiling of a facade generator. --n equals the test split.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", default="houses_32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=399, help="builds per baseline and control-arm n")
    ap.add_argument("--faces", choices=sorted(FACE_SETS), default="all")
    ap.add_argument("--out-dir", default="outputs/bench_arms")
    ap.add_argument("--arm", action="append", default=[], help="extra track/name=path.npz arms")
    ap.add_argument("--tier", default="both")
    ap.add_argument("--no-run", action="store_true")
    args = ap.parse_args()

    split = splits.load_split(args.corpus, args.seed)
    train = [s for s in splits.split_structures(split, "train") if s.occupied_mask.any()]
    faces = FACE_SETS[args.faces]
    arms = {}
    for name, structs in fb.arms(train, args.n, seed=args.seed, faces=faces).items():
        report = {"recipe": fb.RECIPES[name], "faces": list(faces), "drawn_from": f"split:{args.corpus}:train",
                  "seed": args.seed}
        arms[f"baseline/{name}"] = save_house_cache(structs, 32, cache_dir=args.out_dir, report=report, name=name)
        print(f"[facade] {name}: {len(structs)} builds", flush=True)
    oracle = os.path.join(args.out_dir, "real_facades6_32.npz")
    if os.path.exists(oracle):
        arms["control/real@facades6"] = oracle
    for spec in args.arm:
        name, _, path = spec.partition("=")
        arms[name] = path
    cmd = ["python", "-m", "blockgen.eval.bench", "--tier", args.tier, "--n", str(args.n),
           "--corpus", args.corpus, "--seed", str(args.seed), "--name", NAME, "--note", NOTE,
           "--arms", *(f"{k}:{v}" for k, v in arms.items())]
    print("[facade] " + shlex.join(cmd), flush=True)
    if args.no_run:
        return 0
    env = {**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", ".") or "."}
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
