"""V-KID against the structural metrics a 3D reviewer would reach for, judged alike.

The render metrics are blind to interiors by construction, so beating them is not
the claim a geometry pillar has to earn. The claim is that V-KID is a *valid*
structural metric and at least as good as the obvious structural alternatives.
This script puts every one of those alternatives through the same three tests
V-KID passed:

1. **Probes** (ladder draw): Fill-Interior and Hollow-Shell must separate;
   Material-Shuffle and Monochrome must not move a geometry metric.
2. **Blocking gates** (`ladder.score_rungs`, literally the code that gates the
   leaderboard's metrics): damage ordering, noise ordering, invariance,
   self-consistency, n-stability. Geometry-family rungs (`GEOMETRY_RUNGS`) for
   every row; the colour-reading encoders are also judged on the render-family
   rungs, so no row fails only because it reads colour.
3. **Leaderboard ordering** (n=256 arms of the leaderboard of record, rebuilt
   with `ablate_view_facade_section.build_arms`, which checks build hashes):
   Spearman with V-KID and with MV-KID, and with the GPT-4o exterior
   Bradley-Terry ranking where arms overlap.

Rows. Distances are pairwise matrices on the entries the set statistics read
(`pc_encoders.needed_pairs`); each feeds MMD / 1-COV / 1-NNA (`pointcloud`).

    vkid             V-KID, the bench's 77-d descriptor (v1), whitened on the reference
    occ_kid          KID on the occupancy grid itself: bbox-centred 32^3, 4^3 average-pooled (512-d)
    iou_corner       1 - IoU, grids placed corner to corner (T67's voxel row)
    iou_d4           1 - max IoU over the 8 yaw/mirror variants, grids bbox-centred
    cd_vol           chamfer on 1,024 points at occupied-voxel centres (T67)
    cd_all           chamfer on 2,048 points on every exposed block face, sealed rooms included
    emd_all          exact EMD on 512 such points (ladder only: O(n^2) Hungarian)
    {uni3d,ulip2}_{all,vol}_kid   encoder KID; `all` = every exposed face, `vol` =
                     points uniform inside occupied voxels; both carry block colour

Protocol: houses_32 split houses_32.s0.70-15-15.v1, seed 0. Ladder: the ladder
subset (`ladder.main`'s `usable`), val reference, n=128 per rung, 48 noise reps.
Board: 399 val reference, 399 test calibration, n=256 arms, noise floor = sd of
48 fresh real draws (the leaderboard's rule, via the view-ablation scorer).

    GPU_LOCK=/path/gpu.lock .venv/bin/python scripts/bench_headtohead.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from blockgen.eval.bench import distances as D
from blockgen.eval.bench import geometry as geom
from blockgen.eval.bench import ladder as L
from blockgen.eval.bench import mesh_probe as mp
from blockgen.eval.bench import pc_encoders as pe
from blockgen.eval.bench import pointcloud as pc
from blockgen.eval.bench import probe3d as p3
from blockgen.eval.bench import probes as pb
from blockgen.eval.bench import splits
from blockgen.utils.runs import new_run_dir

CANVAS = 32
POOL = 4
K_ENC, K_CD, K_EMD, K_VOX = p3.K_ENC, p3.K_CD, p3.K_EMD, 1024
VIEW_RUN = Path("outputs/run_20261104_213620_view_facade_section")
VLM_RUN = Path("outputs/run_20261102_214026_vlm_judge")
#: GPT-4o exterior Bradley-Terry arm name -> leaderboard arm (analysis.md)
VLM_MAP = {"Real": "real_test", "Fill-Interior": "real@solidify",
           "Patch-Stitch": "patchwork", "Gabled-House": "gabled_house",
           "Material-Shuffle": "real_shuffled_materials", "Monochrome": "real@monochrome",
           "Hollow-Box": "shell_box", "Voxel-AR (clean)": "canon_oriented",
           "Voxel-AR (contaminated)": "native_oriented"}

LADDER_PROBES = ("solidify", "hollow", "material_shuffle", "monochrome",
                 "chunk_delete_40", "occ_noise_5", "canon16", "rot90_1", "mirror_x")


def _view_module():
    spec = importlib.util.spec_from_file_location(
        "ablate_view_facade_section",
        Path(__file__).resolve().parent / "ablate_view_facade_section.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- per-structure CPU features ------------------------------------------------------
def _centred(occ: np.ndarray) -> np.ndarray:
    g = np.zeros((CANVAS,) * 3, bool)
    sh = [min(a, CANVAS) for a in occ.shape]
    o = [(CANVAS - a) // 2 for a in sh]
    g[o[0]:o[0] + sh[0], o[1]:o[1] + sh[1], o[2]:o[2] + sh[2]] = occ[:sh[0], :sh[1], :sh[2]]
    return g


def _d4(occ: np.ndarray) -> List[np.ndarray]:
    out = []
    for m in (occ, occ[::-1]):
        for k in range(4):
            out.append(np.rot90(m, k, axes=(0, 2)))
    return out


def volume_samples(s, k: int, rng: np.random.Generator) -> np.ndarray:
    """`(k, 6)` xyz+rgb: uniform points inside occupied voxels, in `structure_mesh`'s frame."""
    c = s.crop_to_non_air()
    occ = c.occupied_mask
    cells = np.argwhere(occ)
    if len(cells) == 0:
        return np.concatenate([np.zeros((k, 3), np.float32), np.full((k, 3), 0.5, np.float32)], 1)
    pick = cells[rng.integers(0, len(cells), size=k)]
    xyz = (pick + rng.random((k, 3)) - np.asarray(occ.shape) / 2.0) / float(CANVAS)
    col = c.facecolors()[..., :3]
    rgb = np.clip(col[pick[:, 0], pick[:, 1], pick[:, 2]], 0, 1)
    return np.concatenate([xyz, rgb], 1).astype(np.float32)


def cpu_item(job):
    i, s, seed = job
    rng = np.random.default_rng([seed, i])
    c = s.crop_to_non_air()
    occ = c.occupied_mask
    m = mp.structure_mesh(c)
    cent = _centred(occ)
    pooled = cent.reshape(CANVAS // POOL, POOL, CANVAS // POOL, POOL,
                          CANVAS // POOL, POOL).mean((1, 3, 5)).reshape(-1)
    return i, {
        "all": mp.surface_samples(m, K_ENC, rng),
        "vol": volume_samples(c, K_ENC, rng),
        "vox": pc.points(c, "volume", K_VOX, rng),
        "occ_pool": pooled.astype(np.float32),
        "d4": np.stack([_centred(v) for v in _d4(occ)]).reshape(8, -1),
    }


# --- the candidate rows ---------------------------------------------------------------
def candidate_rows(structs: Sequence, ref_ix: np.ndarray, sets: Dict[str, np.ndarray], n: int,
                   *, seed: int, device: str, workers: int, emd: bool, k_cd: int = K_CD,
                   log=print
                   ) -> Tuple[Dict[str, Tuple[Callable, np.ndarray]], Dict[str, float]]:
    """`{row: (fn(ref, probe), X)}` over every structure, and wall time per family."""
    N = len(structs)
    cost: Dict[str, float] = {}
    rows: Dict[str, Tuple[Callable, np.ndarray]] = {}
    col = np.arange(N).reshape(-1, 1)

    t0 = time.time()
    std = geom.Standardizer.fit(geom.descriptor([structs[i] for i in ref_ix]))
    rows["vkid"] = (lambda r, p: geom.geom_kid(p, r), geom.geom_features(structs, std))
    cost["vkid (CPU)"] = time.time() - t0
    log(f"  [h2h] vkid {cost['vkid (CPU)']:.0f}s")

    t0 = time.time()
    items: Dict[int, dict] = {}
    with ProcessPoolExecutor(workers) as ex:
        for k, (i, f) in enumerate(ex.map(cpu_item, [(i, s, seed) for i, s in enumerate(structs)],
                                          chunksize=8)):
            items[i] = f
            if (k + 1) % 500 == 0:
                log(f"  [cpu] {k + 1}/{N}, {time.time() - t0:.0f}s")
    t_sample = time.time() - t0
    log(f"  [h2h] sampling {t_sample:.0f}s")
    occ_pool = np.stack([items[i]["occ_pool"] for i in range(N)]).astype(np.float64)
    rows["occ_kid"] = (lambda r, p: D.kid(p, r), occ_pool)
    clouds = {s: np.stack([items[i][s] for i in range(N)]) for s in ("all", "vol")}
    vox = np.stack([items[i]["vox"] for i in range(N)]).astype(np.float32)
    d4 = np.stack([items[i]["d4"] for i in range(N)])                   # (N, 8, 32768) bool
    del items
    cost["point/grid sampling, all rows (CPU)"] = t_sample

    need = pe.needed_pairs(N, ref_ix, sets, n)
    pairs = np.argwhere(np.triu(need | need.T, 1))
    log(f"  [h2h] {N} items, {len(pairs)} distance pairs")
    dms: Dict[str, np.ndarray] = {}
    if emd:
        t0 = time.time()
        dms["emd_all"] = pe.emd_matrix(np.ascontiguousarray(clouds["all"][:, :K_EMD, :3]), need,
                                       workers=workers)
        cost["emd_all (CPU)"] = time.time() - t0
        log(f"  [h2h] emd_all {cost['emd_all (CPU)']:.0f}s")

    with p3.gpu_lock():
        import torch
        t0 = time.time()
        dms["iou_corner"] = pc.iou_matrix(pc.grids(structs), device)
        cost["iou_corner (GPU)"] = time.time() - t0

        t0 = time.time()
        X = torch.as_tensor(d4[:, 0], dtype=torch.float16, device=device)          # (N, V)
        size = X.float().sum(1)
        best = torch.zeros((N, N), dtype=torch.float32, device=device)
        for k in range(8):
            Vk = torch.as_tensor(d4[:, k], dtype=torch.float16, device=device)
            for a in range(0, N, 1024):
                inter = (Vk[a:a + 1024] @ X.T).float()
                best[a:a + 1024] = torch.maximum(best[a:a + 1024], inter)
            del Vk
        best = torch.maximum(best, best.T)
        union = size[:, None] + size[None, :] - best
        dm = (1.0 - best / union.clamp_min(1.0)).cpu().numpy()
        np.fill_diagonal(dm, 0.0)
        dms["iou_d4"] = dm
        del X, best, union
        torch.cuda.empty_cache()
        cost["iou_d4 (GPU)"] = time.time() - t0
        log(f"  [h2h] iou {time.time() - t0:.0f}s")

        t0 = time.time()
        dms["cd_vol"] = p3._matrix(N, pairs, p3.chamfer_pairs(vox, pairs, device))
        cost["cd_vol (GPU)"] = time.time() - t0
        t0 = time.time()
        dms["cd_all"] = p3._matrix(N, pairs, p3.chamfer_pairs(
            np.ascontiguousarray(clouds["all"][:, :k_cd, :3]), pairs, device))
        cost[f"cd_all k={k_cd} (GPU)"] = time.time() - t0
        log(f"  [h2h] chamfer done")

        for name in ("uni3d", "ulip2"):
            t0 = time.time()
            enc = pe.ENCODERS[name](device)
            for s in ("all", "vol"):
                t1 = time.time()
                Z = pe.embed(enc, list(clouds[s]))
                rows[f"{name}_{s}_kid"] = (lambda r, p: D.kid(p, r), Z)
                cost[f"{name}_{s}_kid (GPU)"] = time.time() - t1
                log(f"  [h2h] {name}_{s} {time.time() - t1:.0f}s")
            del enc
            torch.cuda.empty_cache()

    for mname, fn in pc.metric_table(dms).items():
        rows[mname] = (fn, col)
    return rows, cost


# --- part 1+2: ladder -----------------------------------------------------------------
def run_ladder(args, log) -> dict:
    split = splits.load_split("houses_32", args.seed)

    def usable(s):
        return max(s.shape) > 16 and 300 <= int(s.occupied_mask.sum()) <= 4000

    ref_all = splits.split_structures(split, "val", predicate=usable)
    rng = np.random.default_rng(args.seed)
    ref = [ref_all[i].crop_to_non_air() for i in rng.permutation(len(ref_all))[:256]]
    probe = [s.crop_to_non_air() for s in splits.split_structures(split, "test", predicate=usable)]
    order = np.random.default_rng(args.seed).permutation(len(probe))
    probe = [probe[i] for i in order]
    n = min(args.n_ladder, len(probe) - 1)
    src = probe[:n]
    suite = pb.probe_suite(src, np.random.default_rng(args.seed))
    suite["real_heldout"] = src

    structs: List = list(ref)
    ref_ix = np.arange(len(ref))
    sets: Dict[str, np.ndarray] = {}
    for name, ss in suite.items():
        sets[name] = np.arange(len(structs), len(structs) + len(ss))
        structs.extend(ss)
    sets["null"] = np.arange(len(structs), len(structs) + len(probe))
    structs.extend(probe)
    log(f"[ladder] ref {len(ref)}, pool {len(probe)}, n={n}, {len(structs)} structures")

    rows, cost = candidate_rows(structs, ref_ix, sets, n, seed=args.seed, device=args.device,
                                workers=args.workers, emd=args.emd, log=log)
    out = {"n": n, "n_ref": len(ref), "n_pool": len(probe), "cost_s": cost, "rows": {}}
    for name, (fn, X) in rows.items():
        res = {}
        specs = {"geometry": L.GEOMETRY_RUNGS}
        if "uni3d" in name or "ulip2" in name:
            specs["render"] = L.RENDER_RUNGS
        for sname, spec in specs.items():
            r = L.score_rungs({name: fn}, X[ref_ix], {k: X[v] for k, v in sets.items()
                                                    if k != "null"},
                              X[sets["null"]], spec, backbone="h2h", view=name,
                              reps=args.reps, rng=np.random.default_rng(args.seed + 3))
            sd = r.noise[name]["sd"]
            base = r.scores[name]["real_heldout"]
            res[sname] = {
                "gates": r.gates[name], "passed": all(r.gates[name].values()),
                "failed": [g for g, v in r.gates[name].items() if not v],
                "sd": sd, "scores": r.scores[name],
                "z": {k: ((r.scores[name][k] - base) / sd if sd and np.isfinite(sd) and sd > 0
                          else float("nan")) for k in LADDER_PROBES
                      if k in r.scores[name]},
            }
        out["rows"][name] = res
        g = res["geometry"]
        log(f"  {name:28s} solidify {g['z']['solidify']:+8.1f} hollow {g['z']['hollow']:+7.1f} "
            f"shuffle {g['z']['material_shuffle']:+6.1f}  gates {'PASS' if g['passed'] else g['failed']}")
    return out


# --- part 3: leaderboard ordering -----------------------------------------------------
def run_board(args, log) -> dict:
    vm = _view_module()
    n = args.n_board
    ref, calib, arms, check = vm.build_arms(args.seed, n)
    log(f"[board] ref {len(ref)} calib {len(calib)} arms {len(arms)}, same builds: "
        f"{all(check.values()) if check else 'unchecked'}")
    structs: List = list(ref)
    ref_ix = np.arange(len(ref))
    sets: Dict[str, np.ndarray] = {}
    for name, ss in {"calib": calib, **arms}.items():
        sets[name] = np.arange(len(structs), len(structs) + len(ss))
        structs.extend(ss)
    rows, cost = candidate_rows(structs, ref_ix, sets, n, seed=args.seed, device=args.device,
                                workers=args.workers, emd=False, k_cd=args.k_cd_board, log=log)
    vres = json.loads((VIEW_RUN / "results.json").read_text())["results"]
    vr = vres.get(str(n), vres["256"])
    ranked = [a for a in arms if a not in ("real_test",) + vm.INVARIANCE]
    mv = [vr["corner (paper)"][a]["sep"] for a in ranked]
    vk = [vr["V-KID (77-d)"][a]["sep"] for a in ranked]
    bt = _vlm_bt()
    out = {"n": n, "arms": ranked, "same_builds": check, "cost_s": cost, "rows": {},
           "mv_kid": dict(zip(ranked, mv)), "v_kid_view_run": dict(zip(ranked, vk))}
    for name, (fn, X) in rows.items():
        res = vm.score_space(X[ref_ix], X[sets["calib"]], {a: X[sets[a]] for a in arms}, n,
                             args.seed, kidfn=lambda a, r, fn=fn: fn(r, a), ci_reps=args.ci_reps)
        z = [res[a]["sep"] for a in ranked]
        both = [a for a in ranked + ["real_test"] if a in bt]
        zz = {**{a: res[a]["sep"] for a in ranked}, "real_test": 0.0}
        out["rows"][name] = {
            "sep": {a: res[a]["sep"] for a in arms if a != "real_test"},
            "ci": {a: res[a]["ci"] for a in arms if a != "real_test"},
            "sd": res["_sd"],
            "rho_vkid": vm.spearman(z, vk), "rho_mvkid": vm.spearman(z, mv),
            "rho_vlm_bt": (vm.spearman([-zz[a] for a in both], [bt[a] for a in both])
                           if len(both) >= 4 else None),
            "vlm_arms": both,
        }
        r = out["rows"][name]
        log(f"  {name:28s} rho V-KID {r['rho_vkid']:+.2f}  MV-KID {r['rho_mvkid']:+.2f}  "
            f"solidify {r['sep']['real@solidify']:+.1f}")
    return out


def _vlm_bt() -> Dict[str, float]:
    """GPT-4o exterior Bradley-Terry strengths, from the system-level table of analysis.md."""
    out: Dict[str, float] = {}
    for line in (VLM_RUN / "analysis.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 6 and cells[0] in VLM_MAP:
            try:
                out[VLM_MAP[cells[0]]] = float(cells[1])
            except ValueError:
                pass
    return out


# --- samples.png -----------------------------------------------------------------------
def samples_png(run: Path, seed: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    split = splits.load_split("houses_32", seed)
    test = [s.crop_to_non_air() for s in splits.split_structures(split, "test")]
    s = max(test[:40], key=lambda t: int(t.occupied_mask.sum()) if max(t.shape) <= 32 else 0)
    arms = {"real": s, "Fill-Interior": pb.solidify(s), "Hollow-Shell": pb.hollow(s)}
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(3, 4, figsize=(12, 9))
    for r, (name, t) in enumerate(arms.items()):
        occ = t.occupied_mask
        y = occ.shape[1] // 3
        ax[r, 0].imshow(occ[:, y, :].T, cmap="Greys", origin="lower")
        ax[r, 0].set_title(f"{name}: occupancy slice y={y}", fontsize=8)
        for c, (lab, cl) in enumerate((("all-face points", mp.surface_samples(mp.structure_mesh(t), 6000, rng)),
                                       ("volume points", volume_samples(t, 6000, rng)))):
            yy = (y - occ.shape[1] / 2.0) / CANVAS
            m = np.abs(cl[:, 1] - (yy + 0.5 / CANVAS)) < 0.5 / CANVAS
            ax[r, 1 + c].scatter(cl[m, 0], cl[m, 2], s=1, c=cl[m, 3:6])
            ax[r, 1 + c].set_title(f"{lab}, same slice", fontsize=8)
            ax[r, 1 + c].set_aspect("equal")
        g = _centred(occ).reshape(CANVAS // POOL, POOL, CANVAS // POOL, POOL,
                                  CANVAS // POOL, POOL).mean((1, 3, 5))
        ax[r, 3].imshow(g[:, y // POOL, :].T, cmap="Greys", origin="lower", vmin=0, vmax=1)
        ax[r, 3].set_title("occ_kid input (4^3 pooled)", fontsize=8)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(run / "samples.png", dpi=110)
    (run / "samples.txt").write_text("rows: real, Fill-Interior, Hollow-Shell (one 3D-Craft test "
                                     "build); columns: occupancy slice, all-face samples in the "
                                     "slice, volume samples in the slice, pooled occupancy\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--part", choices=("ladder", "board", "both"), default="both")
    ap.add_argument("--n-ladder", type=int, default=128)
    ap.add_argument("--n-board", type=int, default=256)
    ap.add_argument("--reps", type=int, default=48)
    ap.add_argument("--ci-reps", type=int, default=100)
    ap.add_argument("--k-cd-board", type=int, default=1024,
                    help="chamfer points on the n=256 board (2,048 on the ladder)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-emd", dest="emd", action="store_false")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    geom.set_descriptor_version("v1")
    run = Path(args.out) if args.out else new_run_dir("bench_headtohead")
    run.mkdir(parents=True, exist_ok=True)
    log_f = open(run / "log.txt", "a")

    def log(s):
        print(s, flush=True)
        log_f.write(s + "\n")
        log_f.flush()

    log(f"[h2h] -> {run}")
    samples_png(run, args.seed)
    out = {"args": vars(args)}
    if args.part in ("ladder", "both"):
        out["ladder"] = run_ladder(args, log)
        (run / "results.json").write_text(json.dumps(out, indent=1, default=float))
    if args.part in ("board", "both"):
        out["board"] = run_board(args, log)
        (run / "results.json").write_text(json.dumps(out, indent=1, default=float))
    log(f"-> {run}")


if __name__ == "__main__":
    main()
