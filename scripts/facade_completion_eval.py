"""Completion-from-given-faces, re-scored per house (CPU).

For each oracle run (the real front face, or front and right faces, of every held-out test house
forced verbatim; the model writes the rest), compute per house
  shell IoU   voxel IoU of the lifted completion with the real house's six-face shell
  block agr.  share of voxels both occupy that carry the same block id
with 95% percentile bootstrap intervals over houses, under two lifts: the default (faces lifted
in decode order, later faces overwrite) and given-first priority (the given faces are lifted last,
so a disagreeing model face can never overwrite a real one).

Retrieval baseline: for each test house, the training house whose given face(s) match best
(same face size; fewest cells differing in mask, depth or block) is returned as its own shell.

    PYTHONPATH=. python scripts/facade_completion_eval.py --out outputs/analysis/faces_completion.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from blockgen.corpus import facade as fc  # noqa: E402
from blockgen.eval.bench import splits  # noqa: E402

R = "outputs/run_20261028_125240_houses32_facades_joint_d4_20m"
RUNS = {
    ("front", 1.0, "grammar"): "oracle_front_none_t1.0",
    ("front", 1.0, "cross"): "oracle_front_constrain_t1.0",
    ("front", 0.7, "grammar"): "oracle_front_none_t0.7",
    ("front", 0.7, "cross"): "oracle_front_constrain_t0.7",
    ("front,right", 0.7, "grammar"): "oracle_front_right_none_t0.7",
    ("front,right", 0.7, "cross"): "oracle_front_right_constrain_t0.7",
}
GIVEN = {"front": ("-z",), "front,right": ("-z", "+x")}
ORDER = ("-z", "+x", "+z", "-x", "+y", "-y")


def load_sets(path):
    z = np.load(path)
    n = 1 + max(int(k.split("_")[0]) for k in z.files)
    out = []
    for i in range(n):
        X, Y = z[f"{i}_-z_depth"].shape
        Z = z[f"{i}_+x_depth"].shape[0]
        shape = (X, Y, Z)
        out.append((shape, {f: fc.Facade(f, z[f"{i}_{f}_depth"], z[f"{i}_{f}_ids"], z[f"{i}_{f}_data"],
                                         shape["xyz".index(f[1])]) for f in ORDER if f"{i}_{f}_depth" in z.files}))
    return out


def iou_agree(a_ids, b_ids):
    """IoU of occupancy and block agreement on the shared voxels; canvases aligned at the origin."""
    shp = tuple(max(x, y) for x, y in zip(a_ids.shape, b_ids.shape))
    A = np.zeros(shp, a_ids.dtype); A[tuple(slice(0, s) for s in a_ids.shape)] = a_ids
    B = np.zeros(shp, b_ids.dtype); B[tuple(slice(0, s) for s in b_ids.shape)] = b_ids
    oa, ob = A != 0, B != 0
    inter = oa & ob
    u = (oa | ob).sum()
    return (inter.sum() / u if u else 0.0), (float((A[inter] == B[inter]).mean()) if inter.any() else 0.0)


def boot(x, n=1000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(len(x), size=len(x))].mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summarize(vals):
    return {"mean": float(np.mean(vals)), "ci": boot(vals)}


def face_dist(F, G):
    if F.depth.shape != G.depth.shape:
        return np.inf
    same = (F.mask == G.mask) & ((~F.mask) | ((F.depth == G.depth) & (F.ids == G.ids)))
    return 1.0 - same.mean()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/analysis/faces_completion.json")
    a = ap.parse_args(argv)
    split = splits.load_split("houses_32", 0)
    test = [s.crop_to_non_air() for s in splits.split_structures(split, "test")]
    test = [s for s in test if max(s.shape) <= 32 and s.occupied_mask.any()]
    shells = [fc.shell(s, fc.ALL_FACES) for s in test]
    real_faces = [fc.project_all(s, fc.ALL_FACES) for s in shells]
    res = {}
    for (given, T, dec), d in RUNS.items():
        sets = load_sets(f"{R}/{d}/faces.npz")
        n = min(len(sets), len(shells))
        g_names = GIVEN[given]
        rows = {"default": ([], []), "given_first": ([], [])}
        for i in range(n):
            shape, faces = sets[i]
            for mode in rows:
                order = list(ORDER)
                if mode == "given_first":
                    order = [f for f in ORDER if f not in g_names] + list(g_names)
                faces_o = [faces[f] for f in order if f in faces]
                for f in g_names:                         # the real face, exactly as forced
                    if mode == "given_first":
                        faces_o[[x.face for x in faces_o].index(f)] = real_faces[i][f]
                lifted = fc.lift(faces_o, shape)
                iou, agr = iou_agree(lifted.block_ids, shells[i].block_ids)
                rows[mode][0].append(iou); rows[mode][1].append(agr)
        key = f"{given}|T={T}|{dec}"
        res[key] = {m: {"shell_iou": summarize(v[0]), "block_agree": summarize(v[1]), "n": n} for m, v in rows.items()}
        print(key, {m: round(res[key][m]["shell_iou"]["mean"], 3) for m in rows}, flush=True)

    # retrieval baseline over the training houses
    train = [s.crop_to_non_air() for s in splits.split_structures(split, "train")]
    train = [s for s in train if max(s.shape) <= 32 and s.occupied_mask.any()]
    tr_shells = [fc.shell(s, fc.ALL_FACES) for s in train]
    tr_faces = [fc.project_all(s, ("-z", "+x")) for s in tr_shells]
    for given, g_names in GIVEN.items():
        ious, agrs = [], []
        for i, rf in enumerate(real_faces):
            best, bd = None, np.inf
            for j, tf in enumerate(tr_faces):
                dsum = sum(face_dist(rf[f], tf[f]) for f in g_names)
                if dsum < bd:
                    bd, best = dsum, j
            if best is None:                               # no training house shares the face size
                ious.append(0.0); agrs.append(0.0)
                continue
            iou, agr = iou_agree(tr_shells[best].block_ids, shells[i].block_ids)
            ious.append(iou); agrs.append(agr)
        key = f"{given}|retrieval"
        res[key] = {"shell_iou": summarize(ious), "block_agree": summarize(agrs), "n": len(ious),
                    "no_match": int(sum(1 for x in ious if x == 0.0))}
        print(key, round(res[key]["shell_iou"]["mean"], 3), "no match", res[key]["no_match"], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
