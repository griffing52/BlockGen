"""Colour-free learned 3D encoders on interior-inclusive points, through the paper's gates.

The head-to-head (`bench_headtohead.py`) feeds Uni3D-B and ULIP-2 PointBERT clouds that
carry block colour, so both fail only the geometry-tier invariance gate (G6): material
rungs move them. This run asks whether the encoders, once they cannot read colour, pass
G1, G2, G6, G7 and G8 on the same ladder, and how they order the n=256 leaderboard.

Rows (each with the colour row recomputed on the same points for reference):
    {uni3d,ulip2}_{all,vol}_kid        xyz + block colour (as the head-to-head)
    {uni3d,ulip2}_{all,vol}_nocol_kid  xyz + constant grey 0.4 (only
                                       geometry remains)
`all` = points on every exposed block face (sealed rooms included); `vol` = points
uniform inside occupied voxels. Protocol, ladder, board and scoring are
`bench_headtohead.run_ladder` / `run_board` unchanged; only `candidate_rows` is
replaced, so the rows are scored by literally the same code.

    flock <gpu.lock> .venv/bin/python scripts/uni3d_colorfree_gates.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_headtohead as h  # noqa: E402
from blockgen.eval.bench import distances as D  # noqa: E402
from blockgen.eval.bench import geometry as geom  # noqa: E402
from blockgen.eval.bench import pc_encoders as pe  # noqa: E402
from blockgen.utils.runs import new_run_dir  # noqa: E402

GREY = 0.4


def _cpu(job):
    i, s, seed = job
    rng = np.random.default_rng([seed, i])
    c = s.crop_to_non_air()
    m = h.mp.structure_mesh(c)
    return i, {"all": h.mp.surface_samples(m, h.K_ENC, rng),
               "vol": h.volume_samples(c, h.K_ENC, rng)}


def candidate_rows(structs, ref_ix, sets, n, *, seed, device, workers, emd, k_cd=None,
                   log=print):
    N = len(structs)
    rows, cost = {}, {}
    std = geom.Standardizer.fit(geom.descriptor([structs[i] for i in ref_ix]))
    rows["vkid"] = (lambda r, p: geom.geom_kid(p, r), geom.geom_features(structs, std))
    items = {}
    with ProcessPoolExecutor(workers) as ex:
        for i, f in ex.map(_cpu, [(i, s, seed) for i, s in enumerate(structs)], chunksize=8):
            items[i] = f
    clouds = {s: np.stack([items[i][s] for i in range(N)]) for s in ("all", "vol")}
    del items
    import torch
    for name in ("uni3d", "ulip2"):
        enc = pe.ENCODERS[name](device)
        for s in ("all", "vol"):
            for col in (True, False):
                t1 = time.time()
                C = clouds[s] if col else np.concatenate(
                    [clouds[s][..., :3], np.full_like(clouds[s][..., 3:6], GREY)], -1)
                Z = pe.embed(enc, list(C))
                key = f"{name}_{s}_kid" if col else f"{name}_{s}_nocol_kid"
                rows[key] = (lambda r, p: D.kid(p, r), Z)
                cost[key] = time.time() - t1
                log(f"  [cf] {key} {cost[key]:.0f}s, "
                    f"GPU peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
        del enc
        torch.cuda.empty_cache()
    return rows, cost


class Args:
    seed = 0
    n_ladder = 128
    n_board = 256
    reps = 48
    ci_reps = 100
    k_cd_board = 1024
    workers = 8
    emd = False
    device = "cuda"


def main():
    h.candidate_rows = candidate_rows
    geom.set_descriptor_version("v1")
    run = new_run_dir("uni3d_colorfree_gates")
    shutil.copy(__file__, run / Path(__file__).name)
    logf = open(run / "log.txt", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    args = Args()
    out = {"grey": GREY, "protocol": "bench_headtohead ladder + board, seed 0"}
    out["ladder"] = h.run_ladder(args, log)
    (run / "results.json").write_text(json.dumps(out, indent=1, default=float))
    out["board"] = h.run_board(args, log)
    (run / "results.json").write_text(json.dumps(out, indent=1, default=float))
    h.samples_png(run, 0)
    log(f"-> {run}")


if __name__ == "__main__":
    main()
