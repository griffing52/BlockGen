"""The render and voxel rows of `run_mesh_probe_3d.py` alone: DINOv2 KID on 4 exterior
views, DINOv2 KID on 4 cutaway views, and V-KID, on the same split, probe draw and
voxeliser seeds, with the same `probe3d.score` (noise floor, 200 paired 80% subsample
CIs). No point clouds, so it runs in minutes where the full script takes hours on CPU.

    GPU_LOCK=... PYTHONPATH=. .venv/bin/python scripts/run_mesh_probe_render_rows.py \
        --objects 600 --n 170 --seed 1 --name mesh_render_rows_568_s1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from blockgen.eval.bench import distances as D          # noqa: E402
from blockgen.eval.bench import features as ft           # noqa: E402
from blockgen.eval.bench import geometry as geom         # noqa: E402
from blockgen.eval.bench import mesh_probe as mp         # noqa: E402
from blockgen.eval.bench import probe3d as p3            # noqa: E402
from blockgen.utils.runs import new_run_dir              # noqa: E402
from run_mesh_probe import PROBES, make_variant         # noqa: E402
from run_mesh_probe_3d import ROOT, strict_objects      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", type=int, default=512)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--n-ref", type=int, default=None)
    ap.add_argument("--reps", type=int, default=48)
    ap.add_argument("--ci-reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda",
                    help="cpu: DINOv2 on CPU; run with PYOPENGL_PLATFORM=osmesa for no GPU at all")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    recs = strict_objects(a.objects)
    N = len(recs)
    order = np.random.default_rng(1000 + a.seed).permutation(N)
    recs = [recs[i] for i in order]
    n_ref = a.n_ref or N // 2
    pool = list(range(n_ref, N))
    n = min(a.n, len(pool) - 1)
    probe_src = set(pool[:n])
    run = new_run_dir(a.name or f"mesh_render_rows_s{a.seed}")
    print(f"[rows] {N} objects: ref {n_ref}, pool {len(pool)}, n={n}, seed {a.seed} -> {run}",
          flush=True)
    variants_of = {oi: ("real",) + (PROBES if oi in probe_src else ()) for oi in range(N)}
    keys = [(oi, v) for oi in range(N) for v in variants_of[oi]]
    feats = {"dino4": [], "dino_cut4": []}
    structs = []
    import contextlib
    with (p3.gpu_lock() if a.device != "cpu" else contextlib.nullcontext()):
        for oi, r in enumerate(recs):
            m, ex = mp.load_npz(str(ROOT / "meshes" / f"{r['uid']}.npz"), r["uid"])
            g = mp.grid_for(m, mp.HIDDEN_R)
            for vi, v in enumerate(variants_of[oi]):
                vm = make_variant(v, m, ex["visible"], ex["hidden"], g, np.random.default_rng(oi))
                v4 = mp.render(vm, mp.VIEWS4)
                c4 = mp.render(vm, mp.VIEWS4, cut=True)
                feats["dino4"].append(ft.pooled(ft.embed_images(v4, "dinov2b", a.device, a.batch)[None])[0])
                feats["dino_cut4"].append(ft.pooled(ft.embed_images(c4, "dinov2b", a.device, a.batch)[None])[0])
                # seeded as the full script's voxeliser, which draws the point clouds
                # from the same generator first, so the voxel sample is not identical
                structs.append(mp.voxelize(vm, mp.VOXEL_R, np.random.default_rng([a.seed, oi, vi])))
            if (oi + 1) % 8 == 0:
                # pyrender keeps per-scene GL state alive in the cached renderer; under
                # osmesa that is host RAM (~28 MB per item), so recycle the context
                for r_ in mp._RENDERERS.values():
                    r_.delete()
                mp._RENDERERS.clear()
            if (oi + 1) % 64 == 0:
                print(f"  {oi + 1}/{N}", flush=True)
        ft._MODELS.clear()
    key_ix = {k: i for i, k in enumerate(keys)}
    ref_ix = np.array([key_ix[(i, "real")] for i in range(n_ref)])
    null_ix = np.array([key_ix[(i, "real")] for i in pool])
    probe_list = sorted(probe_src)
    rung_ix = {"real_heldout": np.array([key_ix[(i, "real")] for i in probe_list])}
    for v in PROBES:
        rung_ix[v] = np.array([key_ix[(i, v)] for i in probe_list])
    table = {}

    def add(name, fn, X):
        table[name] = (fn, X[ref_ix], {k: X[v] for k, v in rung_ix.items()}, X[null_ix])

    for fam in ("dino4", "dino_cut4"):
        add(f"{fam}_kid", lambda r, p: D.kid(p, r), np.stack(feats[fam]).astype(np.float64))
    desc = geom.descriptor(structs)
    add("vkid_geom", lambda r, p: D.kid(p, r), geom.Standardizer.fit(desc[ref_ix])(desc))
    res = p3.score(table, PROBES, n, a.reps, a.ci_reps, np.random.default_rng(a.seed + 2))
    (run / "results.json").write_text(json.dumps(
        {"n_objects": N, "n_ref": n_ref, "n_pool": len(pool), "n": n, "seed": a.seed,
         "reps": a.reps, "ci_reps": a.ci_reps, "results": res,
         "device": a.device, "gl": __import__("os").environ.get("PYOPENGL_PLATFORM"),
         "note": "render and voxel rows of run_mesh_probe_3d.py only"}, indent=2) + "\n")
    lines = [f"# Mesh probe render rows, seed {a.seed}, n={n}, ref={n_ref}, pool={len(pool)}", ""]
    lines += p3.table_md(res, PROBES, ("_kid",))
    (run / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
