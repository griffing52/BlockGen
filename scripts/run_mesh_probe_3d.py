"""Paper-scale mesh interior probe: render rows plus the 3D-native rows.

The pilot (`run_mesh_probe.py`, notes §64) at n=64 on 256 objects, scaled to
the strict multi-room set (`mesh_probe.STRICT_RULE`, notes §65) at n=128, with
the standard 3D-native metrics added on three samplers (`probe3d`):
Uni3D-B and ULIP-2 PointBERT KID/FD, chamfer and exact EMD MMD / 1-COV / 1-NNA,
each on all faces, exterior-visible faces and cutaway-visible faces.

Arms: real held-out, Strip-Interior, Fill-Interior (identical exterior), and
three exterior positive controls (vertex noise, a missing wall, decimation).
Split: the fixed object set (first `--objects` of a seed-0 shuffle of the strict
uids) is re-split per `--seed` into reference (half) and pool (half); probes are
the first n of the pool; the noise floor draws n real objects from the pool.

Phases: CPU (variants, ray-cast samplers, voxels; process pool), GPU (renders,
DINOv2/CLIP, point encoders, chamfer; under the shared GPU flock `$GPU_LOCK`),
CPU (EMD, V-KID, scoring).

    GPU_LOCK=/path/gpu.lock .venv/bin/python scripts/run_mesh_probe_3d.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from blockgen.eval.bench import distances as D
from blockgen.eval.bench import features as ft
from blockgen.eval.bench import mesh_probe as mp
from blockgen.eval.bench import pointcloud as pc
from blockgen.eval.bench import probe3d as p3
from blockgen.utils.runs import new_run_dir
from run_mesh_probe import PROBES, Embedder, make_variant

ROOT = Path("data/external/objaverse_interior")
FIGURE_LICENSES = ("by", "by-sa", "cc0")
RAW_ROWS = ("_kid", "_mmd")


def strict_objects(n_objects: int) -> List[dict]:
    man = {}
    for line in (ROOT / "manifest.jsonl").read_text().splitlines():
        r = json.loads(line)
        man[r["uid"]] = r
    st = [json.loads(l) for l in (ROOT / "interior_stats.jsonl").read_text().splitlines()]
    ok = sorted({s["uid"] for s in st if s["strict"]
                 and (ROOT / "meshes" / f"{s['uid']}.npz").exists()})
    stats = {s["uid"]: s for s in st}
    order = np.random.default_rng(0).permutation(len(ok))
    uids = [ok[i] for i in order][:n_objects]
    return [{**man[u], "interior": stats[u]} for u in uids]


def cpu_object(job):
    """Every CPU-side feature of one object's variants."""
    uid, variants, seed, oi = job
    m, extra = mp.load_npz(str(ROOT / "meshes" / f"{uid}.npz"), uid)
    vis0, hid = extra["visible"], extra["hidden"]
    g = mp.grid_for(m, mp.HIDDEN_R)
    out = {}
    for vi, vname in enumerate(variants):
        vm = make_variant(vname, m, vis0, hid, g, np.random.default_rng(oi))
        rng = np.random.default_rng([seed, oi, vi])
        masks = mp.sampler_masks(vm)
        f = {"clouds": {s: mp.surface_samples(vm, p3.K_ENC, rng, mask=mk)
                        for s, mk in masks.items()}}
        a = vm.face_areas
        f["area_frac"] = {s: (1.0 if mk is None else float(a[mk].sum() / max(a.sum(), 1e-12)))
                          for s, mk in masks.items()}
        s_ = mp.voxelize(vm, mp.VOXEL_R, rng)
        f["_struct"] = s_
        f["pts_vol"] = pc.points(s_, "volume", 1024, rng)
        f["n_vox"] = int(s_.occupied_mask.sum())
        f["n_faces"] = int(len(vm.faces))
        out[vname] = f
    return oi, out


def gpu_object(m_real, vis, hid, g, oi, variants, emb, rng):
    out = {}
    for vname in variants:
        vm = make_variant(vname, m_real, vis, hid, g, np.random.default_rng(oi))
        v4, depth_pts = mp.render(vm, mp.VIEWS4, depth=True)
        v8 = mp.render(vm, mp.VIEWS8)
        v16 = mp.render(vm, mp.VIEWS16)
        c4 = mp.render(vm, mp.VIEWS4, cut=True)
        f = {"_v4": v4, "_c4": c4}
        d4, cl4 = emb(v4, "dinov2b"), emb(v4, "clipL")
        f["dino4"], f["clip4"] = ft.pooled(d4[None])[0], ft.pooled(cl4[None])[0]
        f["clipscore"] = float(100.0 * (cl4 @ emb.text).mean())
        f["dino8"] = ft.pooled(emb(v8, "dinov2b")[None])[0]
        f["dino16"] = ft.pooled(emb(v16, "dinov2b")[None])[0]
        f["dino_cut4"] = ft.pooled(emb(c4, "dinov2b")[None])[0]
        f["clip_cut4"] = ft.pooled(emb(c4, "clipL")[None])[0]
        rp = np.concatenate(depth_pts) if depth_pts else np.zeros((0, 3), np.float32)
        f["pts_render"] = (rp[rng.choice(len(rp), 1024, replace=len(rp) < 1024)] if len(rp)
                           else np.zeros((1024, 3), np.float32))
        out[vname] = f
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--objects", type=int, default=512)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--n-ref", type=int, default=None, help="default: half the objects")
    ap.add_argument("--reps", type=int, default=48)
    ap.add_argument("--ci-reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--emd-workers", type=int, default=16)
    ap.add_argument("--emd-from", default=None,
                    help="emd.npz of an earlier run with the same arguments and seed")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    recs = strict_objects(args.objects)
    N = len(recs)
    order = np.random.default_rng(1000 + args.seed).permutation(N)
    recs = [recs[i] for i in order]
    n_ref = args.n_ref or N // 2
    pool = list(range(n_ref, N))
    n = min(args.n, len(pool) - 1)
    probe_src = set(pool[:n])
    run = new_run_dir(args.name or f"mesh_probe3d_s{args.seed}")
    log = lambda s: print(s, flush=True)
    log(f"[mesh3d] {N} strict objects: ref {n_ref}, pool {len(pool)}, n={n}, seed {args.seed}"
        f" -> {run}")
    variants_of = {oi: ("real",) + (PROBES if oi in probe_src else ()) for oi in range(N)}

    # --- phase 1: CPU --------------------------------------------------------------
    t0 = time.time()
    cpu: Dict[int, dict] = {}
    jobs = [(r["uid"], variants_of[oi], args.seed, oi) for oi, r in enumerate(recs)]
    with ProcessPoolExecutor(args.workers) as ex:
        for k, (oi, out) in enumerate(ex.map(cpu_object, jobs, chunksize=2)):
            cpu[oi] = out
            if (k + 1) % 32 == 0:
                log(f"  [cpu] {k + 1}/{N} objects, {time.time() - t0:.0f}s")

    # index sets (fixed by the variant lists alone), then EMD while the GPU is busy
    keys = [(oi, v) for oi in range(N) for v in variants_of[oi]]
    key_ix = {k: i for i, k in enumerate(keys)}
    ref_ix = np.array([key_ix[(i, "real")] for i in range(n_ref)])
    null_ix = np.array([key_ix[(i, "real")] for i in pool])
    probe_list = sorted(probe_src)
    rung_ix = {"real_heldout": np.array([key_ix[(i, "real")] for i in probe_list])}
    for v in PROBES:
        rung_ix[v] = np.array([key_ix[(i, v)] for i in probe_list])
    sets = {**rung_ix, "null": null_ix}
    clouds = {s: np.stack([cpu[oi][v]["clouds"][s] for oi, v in keys]) for s in mp.SAMPLERS}
    for oi, v in keys:
        del cpu[oi][v]["clouds"]
    emd = p3.emd_dms(clouds, ref_ix, sets, n, args.emd_workers, log,
                     cache=args.emd_from or str(run / "emd.npz"))

    # --- phase 2: GPU renders + image embeddings ------------------------------------
    # one lock acquisition for every GPU step (the inner `gpu_lock`s are no-ops):
    # re-queueing on the shared lock between steps can cost hours
    outer_lock = p3.gpu_lock()
    outer_lock.__enter__()
    gpu: Dict[int, dict] = {}
    sample_rows, sample_names = [], []
    rng = np.random.default_rng(args.seed + 1)
    with p3.gpu_lock():
        t0 = time.time()
        emb = Embedder(args.device)
        for oi, r in enumerate(recs):
            m, extra = mp.load_npz(str(ROOT / "meshes" / f"{r['uid']}.npz"), r["uid"])
            g = mp.grid_for(m, mp.HIDDEN_R)
            out = gpu_object(m, extra["visible"], extra["hidden"], g, oi, variants_of[oi],
                             emb, rng)
            if (oi in probe_src and len(sample_rows) < 6
                    and r["license"] in FIGURE_LICENSES):
                ims = [out[v]["_v4"][0] for v in ("real",) + PROBES]
                ims += [out[v]["_c4"][0] for v in ("real", "strip", "fill")]
                sample_rows.append(np.concatenate(ims, 1))
                sample_names.append(f"{r['name'][:40]} ({r['license']}, {r['uid'][:8]})")
            exts = {v: mp.exterior_delta(out["real"]["_v4"], out[v]["_v4"])
                    for v in variants_of[oi] if v != "real"}
            for v in out:
                out[v].pop("_v4")
                out[v].pop("_c4")
            out["_ext"] = exts
            gpu[oi] = out
            if (oi + 1) % 32 == 0:
                log(f"  [gpu] {oi + 1}/{N} objects, {time.time() - t0:.0f}s")
        del emb
        import torch
        ft._MODELS.clear()
        torch.cuda.empty_cache()

    # samples.png first, so a crash later still leaves the picture
    from PIL import Image, ImageDraw
    if sample_rows:
        grid = np.concatenate(sample_rows, 0)
        W = grid.shape[1]
        header = Image.new("RGB", (W, 22), "white")
        dr = ImageDraw.Draw(header)
        cols = ["real", "strip", "fill", "noise", "wall", "decim", "cut:real", "cut:strip",
                "cut:fill"]
        for j, c in enumerate(cols):
            dr.text((j * 224 + 6, 4), c, fill="black")
        im = Image.fromarray(np.concatenate([np.asarray(header), grid], 0))
        dr = ImageDraw.Draw(im)
        for i, s in enumerate(sample_names):
            dr.text((6, 22 + i * 224 + 4), s, fill="black")
        im.save(run / "samples.png")

    # --- flat item list -------------------------------------------------------------
    items: List[dict] = [{**cpu[oi][v], **gpu[oi][v]} for oi, v in keys]

    table = {}

    def add(name, fn, X):
        table[name] = (fn, X[ref_ix], {k: X[v] for k, v in rung_ix.items()}, X[null_ix])

    for fam in ("dino4", "clip4", "dino8", "dino16", "dino_cut4", "clip_cut4"):
        X = np.stack([it[fam] for it in items]).astype(np.float64)
        add(f"{fam}_kid", lambda r, p: D.kid(p, r), X)
        if fam in ("dino4", "clip4"):
            add(f"{fam}_fd", p3.FastFD(), X)
    cs = np.array([it["clipscore"] for it in items])[:, None]
    add("clipscore_drop", lambda r, p: -float(p.mean()), cs)

    from blockgen.eval.bench import geometry as geom
    log("[mesh3d] voxel descriptors")
    desc = geom.descriptor([it["_struct"] for it in items])
    G = geom.Standardizer.fit(desc[ref_ix])(desc)
    add("vkid_geom", lambda r, p: D.kid(p, r), G)

    # --- 3D-native rows --------------------------------------------------------------
    rows = p3.rows_3d(clouds, ref_ix, sets, n, args.device, emd=emd, log=log)
    dms = rows.pop("_dms")[1]
    for name, (fn, X) in rows.items():
        add(name, fn, X)
    np.savez_compressed(run / "embeddings.npz",
                        **{k: X for k, (fn, X) in rows.items() if k.endswith("_kid")},
                        ref_ix=ref_ix, null_ix=null_ix,
                        **{f"rung_{k}": v for k, v in rung_ix.items()})
    # the pilot's remaining point rows: render-depth clouds, voxel centres, IoU
    with p3.gpu_lock():
        for fam in ("pts_render", "pts_vol"):
            cl = np.stack([it[fam] for it in items]).astype(np.float32)
            dms["cd_" + fam[4:]] = pc.chamfer_matrix(cl, args.device, block=4)
        dms["iou_vox"] = pc.iou_matrix(pc.grids([it["_struct"] for it in items]), args.device)
    col = np.arange(len(items)).reshape(-1, 1)
    for k in ("cd_render", "cd_vol", "iou_vox"):
        for mname, fn in pc.metric_table({k: dms[k]}).items():
            add(mname, fn, col)

    outer_lock.__exit__(None, None, None)
    # --- score -------------------------------------------------------------------------
    srng = np.random.default_rng(args.seed + 2)
    results = p3.score(table, PROBES, n, args.reps, args.ci_reps, srng, log=log)

    # paired per-object distances / NN gap (the "CD to ground truth" reading)
    paired = {}
    for dname, dm in dms.items():
        nn_gap = float(np.nanmin(dm[np.ix_(rung_ix["real_heldout"], ref_ix)], 1).mean())
        paired[dname] = {"nn_gap": nn_gap}
        for v in PROBES:
            d = np.array([dm[a, b] for a, b in zip(rung_ix["real_heldout"], rung_ix[v])])
            paired[dname][v] = float(np.nanmean(d) / nn_gap) if nn_gap > 0 else float("nan")

    probe_recs = [recs[i] for i in probe_list]
    ext = {v: [gpu[i]["_ext"][v] for i in probe_list] for v in PROBES}
    ratio = lambda key: {v: float(np.median([cpu[i][v][key] / max(cpu[i]["real"][key], 1)
                                             for i in probe_list])) for v in PROBES}
    area = {v: {s: float(np.median([cpu[i][v]["area_frac"][s] for i in probe_list]))
                for s in mp.SAMPLERS} for v in ("real",) + PROBES}
    lic_all = Counter(r["license"] for r in recs)
    summary = {
        "n_objects": N, "n_ref": n_ref, "n_pool": len(pool), "n": n, "seed": args.seed,
        "reps": args.reps, "ci_reps": args.ci_reps, "strict_rule": mp.STRICT_RULE,
        "k": {"encoders": p3.K_ENC, "chamfer": p3.K_CD, "emd": p3.K_EMD},
        "uids": {"ref": [recs[i]["uid"] for i in range(n_ref)],
                 "pool": [recs[i]["uid"] for i in pool],
                 "probe": [r["uid"] for r in probe_recs]},
        "objects": [{"uid": r["uid"], "name": r["name"], "license": r["license"],
                     "user": r["user"], "uri": r["uri"], "interior": r["interior"]}
                    for r in recs],
        "license_tally": dict(lic_all),
        "license_tally_probe": dict(Counter(r["license"] for r in probe_recs)),
        "samples_png": sample_names,
        "exterior_pixel_delta_mean": {v: float(np.mean(x)) for v, x in ext.items()},
        "exterior_pixel_delta_max": {v: float(np.max(x)) for v, x in ext.items()},
        "median_voxel_ratio": ratio("n_vox"), "median_face_ratio": ratio("n_faces"),
        "median_sampler_area_frac": area,
        "results": results, "paired_over_nn_gap": paired,
    }
    (run / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [f"# Mesh interior probe, paper scale (Objaverse strict multi-room), "
             f"seed {args.seed}, n={n}, ref={n_ref}, pool={len(pool)}", "",
             "z = (probe - real held-out) / noise-floor sd [95% paired subsample CI]; "
             "(raw difference) on KID (MMD^2) and MMD rows. Samplers: all = all faces, "
             "vis = exterior-visible faces, cut = cutaway-visible faces.", ""]
    lines += p3.table_md(results, PROBES, RAW_ROWS)
    lines += ["", "Exterior pixel delta (0-255): " + ", ".join(
        f"{v} {summary['exterior_pixel_delta_mean'][v]:.2f}" for v in PROBES),
        "Median sampler area fraction: " + "; ".join(
        f"{v}: " + ", ".join(f"{s} {area[v][s]:.2f}" for s in mp.SAMPLERS) for v in area),
        f"Licences (all {N}): {dict(lic_all)}"]
    (run / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {run}")


if __name__ == "__main__":
    main()
