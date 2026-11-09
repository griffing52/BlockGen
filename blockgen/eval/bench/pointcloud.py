"""The volumetric row of the ladder: standard point-cloud metrics on the same rungs.

The obvious objection to `geom_kid` is that a 3D reviewer would never have used
an image metric in the first place: the shape-generation literature scores
generated sets with MMD, coverage and 1-NN accuracy under chamfer distance
(Achlioptas et al. 2018, Yang et al. 2019), and every one of those sees inside a
shape if the points do. This module runs exactly that suite through the same
rungs, gates and noise floors as every other metric in `ladder.py`, so the
answer is a table rather than a paragraph.

Three distances, each a pairwise matrix over every structure the ladder touches:

    cd_volume   chamfer on points at the centres of *all* occupied voxels
    cd_surface  chamfer on points at occupied voxels with an exposed face only
    iou         1 - IoU of the two occupancy grids placed corner to corner

and three set statistics per distance, oriented so larger means further from
the reference: `mmd` (mean over reference shapes of the distance to the nearest
probe), `one_minus_cov` (reference shapes no probe is nearest to) and `one_nna`
(leave-one-out 1-NN accuracy between an equal-sized reference draw and the
probes; 0.5 is indistinguishable, 1.0 is fully separated).

Points are voxel centres shifted to the bounding-box centre and divided by the
canvas side, a single global scale. No per-shape normalisation: a metric that
rescales each build to a unit box cannot see that a build is the wrong size,
and the geometry tier does. Clouds are subsampled to `k` points without
replacement, or with replacement when a build has fewer.

The two chamfer variants are the point: surface sampling is what every released
voxel generator's chamfer uses, and Appendix K of the draft argues it is exactly
invariant to filling the interior. `cd_volume` is what a reviewer would reach for
to fix that. The ladder says what each one sees and what each one passes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from blockgen.eval.bench import ladder as L
from blockgen.eval.bench import probes as pb
from blockgen.utils.data import Structure

BACKBONE = "pointcloud"
CANVAS = 32.0
DISTANCES = ("cd_volume", "cd_surface", "iou")
STATS = ("mmd", "one_minus_cov", "one_nna")


# --- clouds ------------------------------------------------------------------
def surface_mask(occ: np.ndarray) -> np.ndarray:
    """Occupied voxels with at least one face on air (any air, not only exterior)."""
    occ = np.asarray(occ, bool)
    p = np.pad(occ, 1)
    exposed = np.zeros_like(occ)
    for ax in range(3):
        for d in (-1, 1):
            nb = np.roll(p, d, axis=ax)[1:-1, 1:-1, 1:-1]
            exposed |= occ & ~nb
    return exposed


def points(s: Structure, mode: str, k: int, rng: np.random.Generator) -> np.ndarray:
    """`(k, 3)` float32 cloud for one build; see the module docstring."""
    occ = s.crop_to_non_air().occupied_mask
    if mode == "surface":
        occ = surface_mask(occ)
    xyz = np.argwhere(occ).astype(np.float32)
    if len(xyz) == 0:
        return np.zeros((k, 3), np.float32)
    centre = (np.asarray(occ.shape, np.float32) - 1.0) / 2.0
    xyz = (xyz - centre) / CANVAS
    idx = rng.choice(len(xyz), size=k, replace=len(xyz) < k)
    return xyz[idx]


def grids(structures: Sequence[Structure], side: int = int(CANVAS)) -> np.ndarray:
    """`(N, side^3)` bool: each crop placed at the corner of a fixed canvas."""
    out = np.zeros((len(structures), side, side, side), bool)
    for i, s in enumerate(structures):
        occ = s.crop_to_non_air().occupied_mask
        sl = tuple(slice(0, min(a, side)) for a in occ.shape)
        out[i][sl] = occ[sl]
    return out.reshape(len(structures), -1)


# --- pairwise distances -------------------------------------------------------
def chamfer_matrix(clouds: np.ndarray, device: str = "cuda", block: int = 8) -> np.ndarray:
    """`(N, N)` symmetric chamfer distance (mean squared nearest-neighbour
    distance in both directions, the PointFlow convention) between every pair."""
    import torch
    x = torch.as_tensor(clouds, dtype=torch.float32, device=device)  # (N, k, 3)
    n = x.shape[0]
    out = torch.zeros((n, n), dtype=torch.float32, device=device)
    for i in range(0, n, block):
        a = x[i:i + block]                                            # (A, k, 3)
        for j in range(0, n, block):
            b = x[j:j + block]                                        # (B, k, 3)
            d = torch.cdist(a[:, None], b[None]).square_()            # (A, B, k, k)
            cd = d.amin(3).mean(2) + d.amin(2).mean(2)
            out[i:i + block, j:j + block] = cd
    out = 0.5 * (out + out.T)
    out.fill_diagonal_(0.0)
    return out.cpu().numpy()


def iou_matrix(g: np.ndarray, device: str = "cuda") -> np.ndarray:
    """`(N, N)` of 1 - IoU between occupancy grids."""
    import torch
    x = torch.as_tensor(g, dtype=torch.float32, device=device)
    inter = x @ x.T
    size = x.sum(1)
    union = size[:, None] + size[None, :] - inter
    iou = torch.where(union > 0, inter / union.clamp_min(1.0), torch.ones_like(inter))
    return (1.0 - iou).cpu().numpy()


# --- set statistics over a distance matrix ------------------------------------
def _ix(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).reshape(-1).astype(np.int64)


def mmd(dm: np.ndarray, ref: np.ndarray, probe: np.ndarray) -> float:
    sub = dm[np.ix_(_ix(ref), _ix(probe))]
    return float(sub.min(1).mean())


def one_minus_cov(dm: np.ndarray, ref: np.ndarray, probe: np.ndarray) -> float:
    sub = dm[np.ix_(_ix(ref), _ix(probe))]
    matched = np.unique(sub.argmin(0))
    return float(1.0 - len(matched) / sub.shape[0])


def one_nna(dm: np.ndarray, ref: np.ndarray, probe: np.ndarray) -> float:
    """Leave-one-out 1-NN accuracy on an equal-sized reference draw plus the
    probes. The reference draw is the first `len(probe)` indices of `ref`,
    which `run` shuffles once, so every rung meets the same reference shapes."""
    r, p = _ix(ref), _ix(probe)
    r = r[:len(p)]
    both = np.concatenate([r, p])
    label = np.concatenate([np.zeros(len(r), bool), np.ones(len(p), bool)])
    sub = dm[np.ix_(both, both)].copy()
    np.fill_diagonal(sub, np.inf)
    pred = label[sub.argmin(1)]
    return float((pred == label).mean())


def metric_table(dm_by_name: Dict[str, np.ndarray]
                 ) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    table = {}
    for dname, dm in dm_by_name.items():
        for sname, fn in (("mmd", mmd), ("one_minus_cov", one_minus_cov), ("one_nna", one_nna)):
            table[f"{dname}_{sname}"] = (lambda r, p, dm=dm, fn=fn: fn(dm, r, p))
    return table


# --- the ladder ----------------------------------------------------------------
def view_key(k: int) -> str:
    return f"k{k}_canvas{int(CANVAS)}"


def run(reference: Sequence[Structure], probe_pool: Sequence[Structure], n: int = 64,
        k: int = 1024, reps: int = 48, seed: int = 0, device: str = "cuda",
        verbose: bool = True) -> L.LadderResult:
    """Same reference, probe draw, rungs, gates and noise floor as the other
    two families; only the features differ, and here a feature is an index into
    a pairwise distance matrix over every structure the ladder scores."""
    rng = np.random.default_rng(seed)
    ref_pool = [s.crop_to_non_air() for s in reference]
    all_probe = [s.crop_to_non_air() for s in probe_pool]
    order = np.random.default_rng(seed).permutation(len(all_probe))
    all_probe = [all_probe[i] for i in order]
    probe_src = all_probe[:n]
    null_pool = all_probe
    suite = pb.probe_suite(probe_src, np.random.default_rng(seed))
    suite["real_heldout"] = probe_src

    # One flat list; every set the ladder scores is a slice of it.
    structs: List[Structure] = list(ref_pool)
    ref_idx = np.arange(len(structs))
    probe_idx: Dict[str, np.ndarray] = {}
    for name, ss in suite.items():
        probe_idx[name] = np.arange(len(structs), len(structs) + len(ss))
        structs.extend(ss)
    null_idx = np.arange(len(structs), len(structs) + len(null_pool))
    structs.extend(null_pool)
    if verbose:
        print(f"[pointcloud] reference {len(ref_pool)}, probes {len(probe_src)}, "
              f"null pool {len(null_pool)}, {len(structs)} structures, k={k}", flush=True)

    prng = np.random.default_rng(seed + 1)
    dms: Dict[str, np.ndarray] = {}
    for mode in ("volume", "surface"):
        clouds = np.stack([points(s, mode, k, prng) for s in structs])
        dms[f"cd_{mode}"] = chamfer_matrix(clouds, device)
        if verbose:
            print(f"  [cd_{mode}] done", flush=True)
    dms["iou"] = iou_matrix(grids(structs), device)
    if verbose:
        print("  [iou] done", flush=True)

    # `score_rungs` indexes `null_feats[idx]`, so features are (N, 1) index columns.
    col = lambda a: a.reshape(-1, 1)
    result = L.score_rungs(metric_table(dms), col(ref_idx),
                           {k_: col(v) for k_, v in probe_idx.items()}, col(null_idx),
                           L.GEOMETRY_RUNGS, backbone=BACKBONE, view=view_key(k),
                           reps=reps, rng=rng)
    return result


def main() -> None:
    from blockgen.eval.bench import splits
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", default="houses_32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=64, help="builds per probe rung")
    ap.add_argument("--n-ref", type=int, default=256)
    ap.add_argument("--k", type=int, default=1024, help="points per cloud")
    ap.add_argument("--reps", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-root", default=str(L.LADDER_ROOT))
    args = ap.parse_args()

    split = splits.load_split(args.corpus, args.seed)

    def usable(s: Structure) -> bool:          # the same subset `ladder.main` uses
        return max(s.shape) > 16 and 300 <= int(s.occupied_mask.sum()) <= 4000

    ref_all = splits.split_structures(split, "val", predicate=usable)
    rng = np.random.default_rng(args.seed)
    ref = [ref_all[i] for i in rng.permutation(len(ref_all))[:args.n_ref]]
    probe = splits.split_structures(split, "test", predicate=usable)
    result = run(ref, probe, n=args.n, k=args.k, reps=args.reps, seed=args.seed,
                 device=args.device)
    path = L.save(result, root=args.out_root)
    print()
    print(L.render(result))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
