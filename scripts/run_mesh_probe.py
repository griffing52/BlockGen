"""Mesh-domain interior probe: are mesh-generation metrics blind to interiors?

Runs Strip-Interior and Fill-Interior (identical exterior) plus three exterior
positive controls (vertex noise, a missing wall, 5% decimation) on Objaverse
buildings selected by `scripts/fetch_objaverse_interior.py`, and scores every
rung with the metrics text-to-3D and shape-generation papers report:

    render   multi-view KID / FD with DINOv2-B and CLIP-L on 4 exterior views
             (also DINOv2 on 8 and 16 views, and on 4 cutaway views), CLIP score
             against a fixed prompt
    points   chamfer MMD / 1-COV / 1-NNA on clouds sampled from all faces,
             from exterior-visible faces only, and back-projected from the
             4-view depth buffers
    voxel    chamfer on occupied-voxel centres (cd_volume), 1-IoU, and
             V-KID (`geometry.descriptor`, i.e. geom_kid) on a 32-per-unit grid

Every separation is (metric(probe) - metric(real held-out)) / sd, where sd is
the metric's own noise floor (`ladder.noise_floor`: n real held-out objects
drawn from the pool, scored against the same reference). Statistics are the
voxel suite's own (`distances`, `pointcloud`, `ladder`); nothing is reimplemented.

    .venv/bin/python scripts/run_mesh_probe.py            # all accepted objects
    .venv/bin/python scripts/run_mesh_probe.py --limit 48 # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from blockgen.eval.bench import distances as D
from blockgen.eval.bench import features as ft
from blockgen.eval.bench import ladder as L
from blockgen.eval.bench import mesh_probe as mp
from blockgen.eval.bench import pointcloud as pc
from blockgen.utils.runs import new_run_dir

ROOT = Path("data/external/objaverse_interior")
PROMPT = "a 3D model of a building"
PROBES = ("strip", "fill", "noise", "wall", "decim")
IDENTICAL_EXTERIOR = ("strip", "fill")


def load_objects(limit: int | None, seed: int):
    recs = [json.loads(l) for l in (ROOT / "manifest.jsonl").read_text().splitlines()]
    recs = [r for r in recs if r.get("accepted") and (ROOT / "meshes" / f"{r['uid']}.npz").exists()]
    recs.sort(key=lambda r: r["uid"])
    order = np.random.default_rng(seed).permutation(len(recs))
    recs = [recs[i] for i in order]
    return recs[:limit] if limit else recs


def make_variant(name: str, m, vis, hid, g, rng):
    if name == "real":
        return m
    if name == "strip":
        return mp.strip(m, vis)
    if name == "fill":
        return mp.fill(m, vis, hid, g)
    if name == "noise":
        return mp.vertex_noise(m, 0.005, rng)
    if name == "wall":
        return mp.wall_cut(m, 0.3)
    if name == "decim":
        return mp.decimate(m, 0.05)
    raise ValueError(name)


class Embedder:
    def __init__(self, device: str):
        import torch
        from transformers import CLIPTokenizer
        self.device = device
        model, _ = ft._load_backbone("clipL", device)
        tok = CLIPTokenizer.from_pretrained(ft.BACKBONES["clipL"][0])
        with torch.no_grad():
            t = model.get_text_features(**tok([PROMPT], return_tensors="pt").to(device))
            self.text = torch.nn.functional.normalize(t, dim=-1).float().cpu().numpy()[0]

    def __call__(self, imgs: List[np.ndarray], backbone: str) -> np.ndarray:
        return ft.embed_images(imgs, backbone, self.device, batch=16)


def item_features(m, emb: Embedder, k: int, rng: np.random.Generator) -> Dict:
    """Everything every metric needs for one mesh."""
    out: Dict = {}
    v4, depth_pts = mp.render(m, mp.VIEWS4, depth=True)
    v8 = mp.render(m, mp.VIEWS8)
    v16 = mp.render(m, mp.VIEWS16)
    c4 = mp.render(m, mp.VIEWS4, cut=True)
    out["_v4"], out["_c4"] = v4, c4
    d4 = emb(v4, "dinov2b")
    cl4 = emb(v4, "clipL")
    out["dino4"] = ft.pooled(d4[None])[0]
    out["clip4"] = ft.pooled(cl4[None])[0]
    out["clipscore"] = float(100.0 * (cl4 @ emb.text).mean())
    out["dino8"] = ft.pooled(emb(v8, "dinov2b")[None])[0]
    out["dino16"] = ft.pooled(emb(v16, "dinov2b")[None])[0]
    out["dino_cut4"] = ft.pooled(emb(c4, "dinov2b")[None])[0]
    out["clip_cut4"] = ft.pooled(emb(c4, "clipL")[None])[0]
    # point clouds
    out["pts_all"] = mp.surface_points(m, k, rng)
    vis = mp.visible_faces(m)
    out["pts_ext"] = mp.surface_points(m, k, rng, mask=vis)
    rp = np.concatenate(depth_pts) if depth_pts else np.zeros((0, 3), np.float32)
    out["pts_render"] = (rp[rng.choice(len(rp), k, replace=len(rp) < k)] if len(rp)
                         else np.zeros((k, 3), np.float32))
    s = mp.voxelize(m, mp.VOXEL_R, rng)
    out["_struct"] = s
    out["pts_vol"] = pc.points(s, "volume", k, rng)
    out["n_vox"] = int(s.occupied_mask.sum())
    out["n_faces"] = int(len(m.faces))
    out["ext_area_frac"] = float(m.face_areas[vis].sum() / max(m.face_areas.sum(), 1e-12))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-ref", type=int, default=None, help="default: half the objects")
    ap.add_argument("--n", type=int, default=None, help="probe n; default half the pool")
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=48)
    ap.add_argument("--ci-reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--name", default="mesh_probe")
    args = ap.parse_args()

    recs = load_objects(args.limit, args.seed)
    N = len(recs)
    n_ref = args.n_ref or N // 2
    pool = list(range(n_ref, N))
    n = args.n or len(pool) // 2
    probe_src = pool[:n]
    run = new_run_dir(args.name)
    print(f"[mesh-probe] {N} objects: ref {n_ref}, pool {len(pool)}, probes n={n} -> {run}",
          flush=True)

    emb = Embedder(args.device)
    rng = np.random.default_rng(args.seed + 1)
    # one entry per unique (object, variant)
    items: List[Dict] = []
    key_ix: Dict[tuple, int] = {}
    stats_rows = []
    sample_rows = []
    t0 = time.time()
    for oi, r in enumerate(recs):
        m, extra = mp.load_npz(str(ROOT / "meshes" / f"{r['uid']}.npz"), r["uid"])
        vis, hid = extra["visible"], extra["hidden"]
        g = mp.grid_for(m, mp.HIDDEN_R)
        assert tuple(g.shape) == tuple(hid.shape), (g.shape, hid.shape)
        variants = ("real",) + (PROBES if oi in probe_src else ())
        feats = {}
        for vname in variants:
            vm = make_variant(vname, m, vis, hid, g, np.random.default_rng(oi))
            f = item_features(vm, emb, args.k, rng)
            feats[vname] = f
            key_ix[(oi, vname)] = len(items)
            items.append({k_: v for k_, v in f.items() if k_ not in ("_v4", "_c4")})
        if oi in probe_src:
            row = {"uid": r["uid"], "license": r["license"]}
            for vname in PROBES:
                row[f"ext_delta_{vname}"] = mp.exterior_delta(feats["real"]["_v4"],
                                                              feats[vname]["_v4"])
                row[f"n_vox_{vname}"] = feats[vname]["n_vox"]
                row[f"n_faces_{vname}"] = feats[vname]["n_faces"]
            row["n_vox_real"] = feats["real"]["n_vox"]
            row["n_faces_real"] = feats["real"]["n_faces"]
            row["ext_area_frac_real"] = feats["real"]["ext_area_frac"]
            stats_rows.append(row)
            if len(sample_rows) < 6:
                ims = [feats[v]["_v4"][0] for v in ("real",) + PROBES]
                ims += [feats[v]["_c4"][0] for v in ("real", "strip", "fill")]
                sample_rows.append(np.concatenate(ims, 1))
        if (oi + 1) % 16 == 0:
            print(f"  {oi + 1}/{N} objects, {len(items)} items, {time.time() - t0:.0f}s",
                  flush=True)

    # samples.png first, so a crash later still leaves the picture
    from PIL import Image, ImageDraw
    if sample_rows:
        grid = np.concatenate(sample_rows, 0)
        header = Image.new("RGB", (grid.shape[1], 22), "white")
        dr = ImageDraw.Draw(header)
        cols = ["real", "strip", "fill", "noise", "wall", "decim", "cut:real",
                "cut:strip", "cut:fill"]
        for j, c in enumerate(cols):
            dr.text((j * 224 + 6, 4), c, fill="black")
        Image.fromarray(np.concatenate([np.asarray(header), grid], 0)).save(run / "samples.png")

    # --- index sets ---------------------------------------------------------------
    ref_ix = np.array([key_ix[(i, "real")] for i in range(n_ref)])
    null_ix = np.array([key_ix[(i, "real")] for i in pool])
    rung_ix = {"real_heldout": np.array([key_ix[(i, "real")] for i in probe_src])}
    for v in PROBES:
        rung_ix[v] = np.array([key_ix[(i, v)] for i in probe_src])

    # --- metric table: name -> (fn(ref, probe), ref_feats, {rung: feats}, null_feats)
    table = {}

    def add_vec(name, fam, fn):
        X = np.stack([it[fam] for it in items]).astype(np.float64)
        table[name] = (fn, X[ref_ix], {k_: X[v] for k_, v in rung_ix.items()}, X[null_ix])

    for fam in ("dino4", "clip4", "dino8", "dino16", "dino_cut4", "clip_cut4"):
        add_vec(f"{fam}_kid", fam, lambda r, p: D.kid(p, r))
    for fam in ("dino4", "clip4"):
        add_vec(f"{fam}_fd", fam, lambda r, p: D.fd(p, r))
    cs = np.array([it["clipscore"] for it in items])[:, None]
    table["clipscore_drop"] = (lambda r, p: -float(p.mean()), cs[ref_ix],
                               {k_: cs[v] for k_, v in rung_ix.items()}, cs[null_ix])

    # V-KID: the voxel suite's descriptor, whitened on the reference only
    from blockgen.eval.bench import geometry as geom
    print("[mesh-probe] voxel descriptors", flush=True)
    desc = geom.descriptor([it["_struct"] for it in items])
    std = geom.Standardizer.fit(desc[ref_ix])
    G = std(desc)
    table["vkid_geom"] = (lambda r, p: D.kid(p, r), G[ref_ix],
                          {k_: G[v] for k_, v in rung_ix.items()}, G[null_ix])

    # point-cloud distance matrices over every item
    dev = args.device
    dms = {}
    for fam in ("pts_all", "pts_ext", "pts_render", "pts_vol"):
        clouds = np.stack([it[fam] for it in items]).astype(np.float32)
        dms["cd_" + fam[4:]] = pc.chamfer_matrix(clouds, dev, block=4)
        print(f"  [cd_{fam[4:]}] done", flush=True)
    dms["iou_vox"] = pc.iou_matrix(pc.grids([it["_struct"] for it in items]), dev)
    col = lambda a: a.reshape(-1, 1)
    for mname, fn in pc.metric_table(dms).items():
        table[mname] = (fn, col(ref_ix), {k_: col(v) for k_, v in rung_ix.items()},
                        col(null_ix))

    # --- score -----------------------------------------------------------------------
    srng = np.random.default_rng(args.seed + 2)
    results = {}
    for mname, (fn, R, P, Nl) in table.items():
        scores = {k_: float(fn(R, f)) for k_, f in P.items()}
        mean, sd, sem = L.noise_floor(fn, R, Nl, min(n, len(Nl) - 1), args.reps, srng)
        base = scores["real_heldout"]
        sep, ci = {}, {}
        m_sub = int(round(0.8 * n))
        for v in PROBES:
            sep[v] = (scores[v] - base) / sd if sd > 0 else float("nan")
            draws = []
            for _ in range(args.ci_reps):
                idx = srng.choice(n, m_sub, replace=False)
                draws.append((fn(R, P[v][idx]) - fn(R, P["real_heldout"][idx])) / sd)
            ci[v] = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
        results[mname] = {"scores": scores, "noise_mean": mean, "noise_sd": sd,
                          "noise_sem": sem, "sep": sep, "sep_ci95": ci,
                          "real_vs_null_z": (base - mean) / sd if sd > 0 else float("nan")}
        print(f"  {mname:28s} sd={sd:.4g}  " + "  ".join(f"{v}={sep[v]:+.1f}" for v in PROBES),
              flush=True)

    # --- paired, per-object distances (the "CD to ground truth" reading) ---------
    # Mean d(real_i, probe_i) over the probe objects, in units of the mean
    # distance from a real held-out object to its nearest *other* real object in
    # the reference: 1.0 means the corruption moves an object as far as the gap
    # between two different buildings.
    paired = {}
    for dname, dm in dms.items():
        nn_gap = float(dm[np.ix_(rung_ix["real_heldout"], ref_ix)].min(1).mean())
        paired[dname] = {"nn_gap": nn_gap}
        for v in PROBES:
            d = np.array([dm[a, b] for a, b in zip(rung_ix["real_heldout"], rung_ix[v])])
            paired[dname][v] = float(d.mean() / nn_gap) if nn_gap > 0 else float("nan")

    # --- write -----------------------------------------------------------------------
    ext = {v: [r_[f"ext_delta_{v}"] for r_ in stats_rows] for v in PROBES}
    vox_ratio = {v: float(np.median([r_[f"n_vox_{v}"] / max(r_["n_vox_real"], 1)
                                     for r_ in stats_rows])) for v in PROBES}
    face_ratio = {v: float(np.median([r_[f"n_faces_{v}"] / max(r_["n_faces_real"], 1)
                                      for r_ in stats_rows])) for v in PROBES}
    summary = {
        "n_objects": N, "n_ref": n_ref, "n_pool": len(pool), "n": n, "k": args.k,
        "reps": args.reps, "prompt": PROMPT, "seed": args.seed,
        "uids": {"ref": [recs[i]["uid"] for i in range(n_ref)],
                 "pool": [recs[i]["uid"] for i in pool],
                 "probe": [recs[i]["uid"] for i in probe_src]},
        "licenses": {r_["uid"]: r_["license"] for r_ in recs},
        "exterior_pixel_delta_mean": {v: float(np.mean(x)) for v, x in ext.items()},
        "exterior_pixel_delta_max": {v: float(np.max(x)) for v, x in ext.items()},
        "median_voxel_ratio": vox_ratio, "median_face_ratio": face_ratio,
        "results": results, "paired_over_nn_gap": paired,
    }
    (run / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (run / "per_object.jsonl").open("w") as fh:
        for r_ in stats_rows:
            fh.write(json.dumps(r_) + "\n")
    lines = [f"# Mesh interior probe (Objaverse buildings), n={n}, ref={n_ref}, "
             f"pool={len(pool)}", "",
             "Separation = (probe - real held-out) / noise-floor sd; [95% paired "
             "subsample CI]. strip/fill keep the exterior identical.", "",
             "| metric | " + " | ".join(PROBES) + " | sd |", "|" + "---|" * (len(PROBES) + 2)]
    for mname, r_ in results.items():
        cells = [f"{r_['sep'][v]:+.1f} [{r_['sep_ci95'][v][0]:+.1f}, {r_['sep_ci95'][v][1]:+.1f}]"
                 for v in PROBES]
        lines.append(f"| {mname} | " + " | ".join(cells) + f" | {r_['noise_sd']:.3g} |")
    lines += ["", "Exterior pixel delta (0-255, mean over objects x 4 views): "
              + ", ".join(f"{v} {summary['exterior_pixel_delta_mean'][v]:.2f}" for v in PROBES),
              "Median voxel ratio vs real: " + ", ".join(f"{v} {x:.2f}" for v, x in vox_ratio.items()),
              "", "Paired d(real_i, probe_i) / mean NN gap to reference:",
              "| distance | " + " | ".join(PROBES) + " |", "|" + "---|" * (len(PROBES) + 1)]
    lines += [f"| {d} | " + " | ".join(f"{paired[d][v]:.3f}" for v in PROBES) + " |"
              for d in paired]
    lines += ["", "Median face ratio vs real: " + ", ".join(f"{v} {x:.2f}" for v, x in face_ratio.items())]
    (run / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {run}")


if __name__ == "__main__":
    main()
