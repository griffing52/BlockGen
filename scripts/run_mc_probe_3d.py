"""The 3D-native rows on the Minecraft Fill-Interior ladder (notes §65).

Same rows and samplers as the Objaverse probe (`probe3d`): Uni3D-B and ULIP-2
KID/FD, chamfer and exact-EMD MMD / 1-COV / 1-NNA, each on points sampled from

    all   every exposed block face (faces bordering any air, sealed rooms included)
    vis   exposed faces visible from some exterior viewpoint (ray cast on the
          block-face mesh, `mesh_probe.visible_faces`)
    cut   exposed faces visible in the four cutaway views

of the block-face mesh (`mesh_probe.structure_mesh`: every face of an occupied
voxel that borders air, coloured with the block's base colour, the voxel
suite's frame: bbox-centred, divided by the 32 canvas, no per-shape rescale).
For comparison it also carries T67's two voxel-centre chamfer rows
(`pointcloud.points`: `volume` = every occupied voxel centre, `surface` =
voxels with any face on any air), at 1,024 points.

Data: houses_32 split seed 0 (houses32-v1), the ladder's usable subset; the
reference is every usable val build, probes are the first n of a seeded
shuffle of the usable test builds and the null pool is all of them (as
`ladder.run_ladder`). `--seed` changes the shuffle and every sampler draw.
Arms: Fill-Interior (`solidify`), Hollow-Shell (`hollow`), and controls
canon16, chunk_delete_40, occ_noise_5, jitter_columns, material_shuffle,
rot90_1 (an invariance: should not move).

    GPU_LOCK=/path/gpu.lock .venv/bin/python scripts/run_mc_probe_3d.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from blockgen.eval.bench import mesh_probe as mp
from blockgen.eval.bench import pointcloud as pc
from blockgen.eval.bench import probe3d as p3
from blockgen.eval.bench import probes as pb
from blockgen.eval.bench import splits
from blockgen.utils.runs import new_run_dir

RUNGS = ("solidify", "hollow", "canon16", "chunk_delete_40", "occ_noise_5",
         "jitter_columns", "material_shuffle", "rot90_1")
RAW_ROWS = ("_kid", "_mmd")


def make_rung(name: str, s, rng):
    if name == "solidify":
        return pb.solidify(s)
    if name == "hollow":
        return pb.hollow(s)
    if name == "canon16":
        return pb.canon([s], 16)[0]
    if name == "chunk_delete_40":
        return pb.chunk_delete(s, 0.40, rng)
    if name == "occ_noise_5":
        return pb.occupancy_noise(s, 0.05, rng)
    if name == "jitter_columns":
        return pb.jitter_columns(s, 1, rng)
    if name == "material_shuffle":
        return pb.shuffle_materials(s, rng)
    if name == "rot90_1":
        return pb.rotate_y(s, 1)
    raise ValueError(name)


def cpu_item(job):
    i, s, seed = job
    rng = np.random.default_rng([seed, i])
    m = mp.structure_mesh(s)
    masks = mp.sampler_masks(m)
    a = m.face_areas
    return i, {
        "clouds": {k: mp.surface_samples(m, p3.K_ENC, rng, mask=mk) for k, mk in masks.items()},
        "area_frac": {k: (1.0 if mk is None else float(a[mk].sum() / max(a.sum(), 1e-12)))
                      for k, mk in masks.items()},
        "vox_volume": pc.points(s, "volume", 1024, rng),
        "vox_surface": pc.points(s, "surface", 1024, rng),
        "n_faces": int(len(m.faces)), "n_vox": int(s.occupied_mask.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", default="houses_32")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--reps", type=int, default=48)
    ap.add_argument("--ci-reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--emd-workers", type=int, default=16)
    ap.add_argument("--emd-from", default=None,
                    help="emd.npz of an earlier run with the same arguments and seed")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: cap ref and pool")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)

    split = splits.load_split(args.corpus, args.split_seed)

    def usable(s):            # the subset `ladder.main` / `pointcloud.main` use
        return max(s.shape) > 16 and 300 <= int(s.occupied_mask.sum()) <= 4000

    ref = [s.crop_to_non_air() for s in splits.split_structures(split, "val", predicate=usable)]
    test = [s.crop_to_non_air() for s in splits.split_structures(split, "test", predicate=usable)]
    rng = np.random.default_rng(args.seed)
    ref = [ref[i] for i in rng.permutation(len(ref))]
    test = [test[i] for i in rng.permutation(len(test))]
    if args.limit:
        ref, test = ref[:args.limit], test[:args.limit]
    n = min(args.n, len(test) - 1)
    probe_src = test[:n]
    run = new_run_dir(args.name or f"mc_probe3d_s{args.seed}")
    log(f"[mc3d] {args.corpus} split {args.split_seed}: ref {len(ref)} val, pool {len(test)} "
        f"test, n={n}, seed {args.seed} -> {run}")

    structs: List = list(ref)
    ref_ix = np.arange(len(ref))
    null_ix = np.arange(len(structs), len(structs) + len(test))
    structs.extend(test)
    rung_ix = {"real_heldout": null_ix[:n]}
    rrng = np.random.default_rng(args.seed + 7)
    for name in RUNGS:
        rung_ix[name] = np.arange(len(structs), len(structs) + n)
        structs.extend(make_rung(name, s, rrng) for s in probe_src)
    N = len(structs)
    log(f"[mc3d] {N} structures")

    t0 = time.time()
    feats: Dict[int, dict] = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for k, (i, f) in enumerate(ex.map(cpu_item, [(i, s, args.seed)
                                                     for i, s in enumerate(structs)],
                                          chunksize=4)):
            feats[i] = f
            if (k + 1) % 200 == 0:
                log(f"  [cpu] {k + 1}/{N}, {time.time() - t0:.0f}s")
    items = [feats[i] for i in range(N)]

    clouds = {s: np.stack([it["clouds"][s] for it in items]) for s in mp.SAMPLERS}
    for it in items:
        del it["clouds"]
    sets = {**rung_ix, "null": null_ix}
    emd = p3.emd_dms(clouds, ref_ix, sets, n, args.emd_workers, log,
                     cache=args.emd_from or str(run / "emd.npz"))

    # samples.png: exterior and cutaway renders of the first probe builds per arm
    # one lock acquisition for every GPU step (the inner `gpu_lock`s are no-ops):
    # re-queueing on the shared lock between steps can cost hours
    outer_lock = p3.gpu_lock()
    outer_lock.__enter__()
    from PIL import Image, ImageDraw
    show = ("real_heldout", "solidify", "hollow", "chunk_delete_40", "canon16")
    with p3.gpu_lock():
        rows_im = []
        for j in range(3):
            ims = []
            for name in show:
                m = mp.structure_mesh(structs[rung_ix[name][j]])
                ims.append(mp.render(m, mp.VIEWS4[:1])[0])
            for name in ("real_heldout", "solidify", "hollow"):
                m = mp.structure_mesh(structs[rung_ix[name][j]])
                ims.append(mp.render(m, mp.VIEWS4[:1], cut=True)[0])
            rows_im.append(np.concatenate(ims, 1))
    grid = np.concatenate(rows_im, 0)
    header = Image.new("RGB", (grid.shape[1], 22), "white")
    dr = ImageDraw.Draw(header)
    for j, c in enumerate(list(show) + ["cut:real", "cut:solidify", "cut:hollow"]):
        dr.text((j * 224 + 6, 4), c, fill="black")
    Image.fromarray(np.concatenate([np.asarray(header), grid], 0)).save(run / "samples.png")

    rows = p3.rows_3d(clouds, ref_ix, sets, n, args.device, emd=emd, log=log)
    dms = rows.pop("_dms")[1]
    # T67's voxel-centre rows on the same draw
    need = p3.pe.needed_pairs(N, ref_ix, sets, n)
    pairs = np.argwhere(np.triu(need | need.T, 1))
    with p3.gpu_lock():
        for k in ("vox_volume", "vox_surface"):
            cl = np.stack([it[k] for it in items]).astype(np.float32)
            dms["cd_" + k] = p3._matrix(N, pairs, p3.chamfer_pairs(cl, pairs, args.device))
    col = np.arange(N).reshape(-1, 1)
    for k in ("cd_vox_volume", "cd_vox_surface"):
        for mname, fn in pc.metric_table({k: dms[k]}).items():
            rows[mname] = (fn, col)
    np.savez_compressed(run / "embeddings.npz",
                        **{k: X for k, (fn, X) in rows.items() if k.endswith("_kid")},
                        ref_ix=ref_ix, null_ix=null_ix,
                        **{f"rung_{k}": v for k, v in rung_ix.items()})

    outer_lock.__exit__(None, None, None)
    table = {name: (fn, X[ref_ix], {k: X[v] for k, v in rung_ix.items()}, X[null_ix])
             for name, (fn, X) in rows.items()}
    srng = np.random.default_rng(args.seed + 2)
    results = p3.score(table, RUNGS, n, args.reps, args.ci_reps, srng, log=log)

    area = {r: {s: float(np.median([items[i]["area_frac"][s] for i in ix]))
                for s in mp.SAMPLERS} for r, ix in rung_ix.items()}
    vox = {r: float(np.median([items[i]["n_vox"] / max(items[j]["n_vox"], 1)
                               for i, j in zip(ix, rung_ix["real_heldout"])]))
           for r, ix in rung_ix.items()}
    summary = {"corpus": args.corpus, "split_seed": args.split_seed, "seed": args.seed,
               "n_ref": len(ref), "n_pool": len(test), "n": n, "reps": args.reps,
               "ci_reps": args.ci_reps,
               "k": {"encoders": p3.K_ENC, "chamfer": p3.K_CD, "emd": p3.K_EMD, "vox": 1024},
               "median_sampler_area_frac": area, "median_voxel_ratio": vox,
               "results": results}
    (run / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [f"# 3D-native rows on the Minecraft ladder ({args.corpus}, split "
             f"{args.split_seed}), seed {args.seed}, n={n}, ref={len(ref)}, pool={len(test)}",
             "", "z = (rung - real held-out) / noise-floor sd [95% paired subsample CI]; "
             "(raw difference) on KID (MMD^2) and MMD rows. Samplers on the block-face mesh: "
             "all = every exposed face, vis = exterior-visible faces, cut = cutaway-visible "
             "faces; cd_vox_* = T67 voxel-centre clouds.", ""]
    lines += p3.table_md(results, RUNGS, RAW_ROWS)
    lines += ["", "Median sampler area fraction: " + "; ".join(
        f"{r}: " + ", ".join(f"{s} {area[r][s]:.2f}" for s in mp.SAMPLERS) for r in area),
        "Median voxel ratio vs real: " + ", ".join(f"{r} {v:.2f}" for r, v in vox.items())]
    (run / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {run}")


if __name__ == "__main__":
    main()
