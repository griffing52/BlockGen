"""Evaluation of the two-stage generator on full test houses.

Every arm maps a test house's (coarse condition, inventory) to a realized build on the
32^3 grid. Metrics per house (means over houses):

* ``placed_frac``: realized blocks over the inventory size; ``complete``: every block placed;
  ``single``: the realized build is one face-connected component holding every block;
  ``connectivity``: largest component over the inventory size.
* ``coarse_iou`` / ``coarse_l1``: coarse occupancy IoU and count L1 (over N) against the condition.
* ``voxel_iou`` and ``typed_f1`` (Dice of (cell, type)) against the real house in the
  shared origin-normalized frame.
* ``glue_f1`` (and P, R): F1 of the multisets of typed glued face pairs
  ``(type on the +a side, type on the -a side, a)``, as in eval_faces.
* ``slot_l1``: share of blocks whose (coarse cell, type) slot differs from the real house
  (the stage-1 error; 0 for oracle arms whose builds are complete).
* ``support``: share of blocks resting on the ground row or on a block below (real: reported).
* Diversity (separate table): for the first ``n_div`` houses, ``k_div`` independent
  samples; mean pairwise voxel IoU and typed F1 between samples (lower = more diverse),
  and the same samples' mean voxel IoU to the real house.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .cells import (S, House, assignment, assignment_l1, cell_of, house_coarse, inventory, prior_cell_dist,
                    sample_assignment)
from .coarse import Coarse, coarsen


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def to_structure(coords, types, house: House):
    """Realize (coords, types) with each type's most common raw block in ``house``."""
    from .data import graph_to_structure
    rep: Dict[int, Counter] = {}
    for t, r in zip(np.asarray(house.types).tolist(), np.asarray(house.raw).tolist()):
        rep.setdefault(int(t), Counter())[tuple(r)] += 1
    m = {t: c.most_common(1)[0][0] for t, c in rep.items()}
    coords = np.asarray(coords, np.int64).reshape(-1, 3)
    blocks = [[(0, 0, 0, *m.get(int(t), (1, 0)))] for t in np.asarray(types).tolist()]
    return graph_to_structure(coords, blocks)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _grid(coords, types):
    g = np.zeros((S, S, S), np.int64)
    c = np.asarray(coords, np.int64).reshape(-1, 3)
    if len(c):
        g[c[:, 0], c[:, 1], c[:, 2]] = np.asarray(types, np.int64) + 1
    return g


def glue_counts(g: np.ndarray, T: int) -> np.ndarray:
    """Bincount over (t_plus, t_minus, a) of glued face pairs of a type grid (0 empty)."""
    out = []
    for a in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[a], hi[a] = slice(0, -1), slice(1, None)
        x, y = g[tuple(lo)], g[tuple(hi)]
        m = (x > 0) & (y > 0)
        out.append(np.bincount((x[m] - 1) * T + (y[m] - 1), minlength=T * T))
    return np.concatenate(out)


def shift_overlap(g: np.ndarray, gr: np.ndarray, r: int = 2) -> tuple:
    """Best voxel IoU over translations in [-r, r]^3 (no wraparound), and the typed F1 at
    that translation. Separates "right structure, shifted by a few cells" from "wrong"."""
    A, B = g > 0, gr > 0
    na, nb = A.sum(), B.sum()
    best = (-1.0, 0.0)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dz in range(-r, r + 1):
                sa = tuple(slice(max(0, -d), S - max(0, d)) for d in (dx, dy, dz))
                sb = tuple(slice(max(0, d), S - max(0, -d)) for d in (dx, dy, dz))
                ga, gb = g[sa], gr[sb]
                inter = int(((ga > 0) & (gb > 0)).sum())
                iou = inter / max(na + nb - inter, 1)
                if iou > best[0]:
                    best = (iou, float(2 * ((ga == gb) & (ga > 0)).sum() / max(na + nb, 1)))
    return best


def build_metrics(coords, types, h: House, cond: Coarse, T: int, A_true: np.ndarray) -> dict:
    from scipy.ndimage import label
    coords = np.asarray(coords, np.int64).reshape(-1, 3)
    types = np.asarray(types, np.int64)
    N = len(h)
    n = len(coords)
    g = _grid(coords, types)
    gr = _grid(h.coords, h.types)
    out = {"placed_frac": n / N, "complete": float(n == N)}
    if n == 0:
        return {**out, "single": 0.0, "connectivity": 0.0, "shift_iou": 0.0, "shift_typed_f1": 0.0, "coarse_iou": 0.0, "coarse_l1": 1.0, "voxel_iou": 0.0,
                "typed_f1": 0.0, "glue_p": 0.0, "glue_r": 0.0, "glue_f1": 0.0, "slot_l1": 1.0, "support": 0.0}
    lab, k = label(g > 0)
    big = np.bincount(lab.ravel())[1:].max() if k else 0
    out["connectivity"] = big / N
    out["single"] = float(n == N and k == 1)
    c = coarsen(coords, types, cond.f, size=S)
    out["coarse_iou"] = float((c.occ & cond.occ).sum() / max((c.occ | cond.occ).sum(), 1))
    out["coarse_l1"] = float(np.abs(c.count - cond.count).sum() / N)
    A, B = g > 0, gr > 0
    out["voxel_iou"] = float((A & B).sum() / max((A | B).sum(), 1))
    out["typed_f1"] = float(2 * ((g == gr) & A).sum() / (A.sum() + B.sum()))
    out["shift_iou"], out["shift_typed_f1"] = shift_overlap(g, gr)
    pg, pr = glue_counts(g, T), glue_counts(gr, T)
    inter = np.minimum(pg, pr).sum()
    p, r = inter / max(pg.sum(), 1), inter / max(pr.sum(), 1)
    out.update(glue_p=float(p), glue_r=float(r), glue_f1=float(2 * p * r / max(p + r, 1e-9)))
    Ag = np.zeros_like(A_true)
    np.add.at(Ag, (cell_of(coords, cond.f), types), 1)
    out["slot_l1"] = assignment_l1(A_true, Ag) if n == N else float(np.abs(A_true - Ag).sum() / (2 * N))
    below = np.zeros_like(A)
    below[:, 1:] = A[:, :-1]
    ymin = coords[:, 1].min()
    sup = A & (below | (np.arange(S)[None, :, None] == ymin))
    out["support"] = float(sup.sum() / A.sum())
    return out


def pair_similarity(a, b) -> dict:
    ga, gb = _grid(*a), _grid(*b)
    A, B = ga > 0, gb > 0
    return {"iou": float((A & B).sum() / max((A | B).sum(), 1)),
            "typed_f1": float(2 * ((ga == gb) & A).sum() / max(A.sum() + B.sum(), 1))}


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def upsample_assignment(A: np.ndarray, f: int, rng) -> tuple:
    """Each slot's blocks at random free fine cells of its coarse cell (no model)."""
    G = S // f
    loc = np.stack(np.meshgrid(*[np.arange(f)] * 3, indexing="ij"), -1).reshape(-1, 3)
    C, T_ = [], []
    for c in np.nonzero(A.sum(1))[0]:
        ts = np.repeat(np.arange(A.shape[1]), A[c])
        spots = loc[rng.permutation(len(loc))[:len(ts)]] + np.array([c // (G * G), (c // G) % G, c % G]) * f
        C.append(spots); T_.append(ts[rng.permutation(len(ts))])
    return np.concatenate(C), np.concatenate(T_)


class CoarseRetriever:
    def __init__(self, train: Sequence[House], f: int, T: int):
        self.train = train
        cs = [house_coarse(h, f, T) for h in train]
        self.C = np.stack([c.count.ravel() for c in cs]).astype(np.float32)
        self.D = np.stack([c.dom.ravel() for c in cs])

    def __call__(self, cond: Coarse) -> House:
        c, d = cond.count.ravel().astype(np.float32), cond.dom.ravel()
        dist = np.abs(self.C - c).sum(1) + ((self.D != d) & (c > 0)).sum(1)
        return self.train[int(dist.argmin())]


# --------------------------------------------------------------------------- #
# stage 1 report
# --------------------------------------------------------------------------- #
def stage1_report(model, ds, f, device, seed=0, k=3, n=None):
    """NLL and sampled-assignment L1 on the test houses for the model and the heuristic
    prior; pairwise L1 between samples. Also returns figure rows (stage-1 assignment
    placed at random positions inside cells, and the same for the prior)."""
    from .stage1 import nll_of, stage1_items, stage1_probs
    T = ds["vocab"].n_types
    hs = ds["test"][:n] if n else ds["test"]
    items = stage1_items(hs, f, T)
    P = stage1_probs(model, items, device)
    rng = np.random.default_rng(seed)
    rep = {"n": len(items), "f": f}
    for name, getP in (("model", lambda i: P[i]), ("prior", lambda i: prior_cell_dist(items[i]["coarse"], items[i]["inv"]))):
        nll, l1, pw, l1g = [], [], [], []
        for i, it in enumerate(items):
            Pi = getP(i)
            Pn = Pi / Pi.sum(1, keepdims=True)
            nll.append(nll_of(Pn, it["A"]))
            sm = [sample_assignment(Pi, it["count"], it["inv"], rng) for _ in range(k)]
            l1 += [assignment_l1(it["A"], s) for s in sm]
            pw += [assignment_l1(sm[x], sm[y]) for x in range(k) for y in range(x + 1, k)]
            l1g.append(assignment_l1(it["A"], sample_assignment(Pi, it["count"], it["inv"], rng, greedy=True)))
        rep[name] = {"nll_per_block": float(np.mean(nll)), "slot_l1_sample": float(np.mean(l1)),
                     "slot_l1_greedy": float(np.mean(l1g)), "pairwise_l1_between_samples": float(np.mean(pw))}
    rows = {}
    for name in ("prior", "model"):
        r = []
        for i, (h, it) in enumerate(zip(hs[:8], items[:8])):
            Pi = P[i] if name == "model" else prior_cell_dist(it["coarse"], it["inv"])
            Ai = sample_assignment(Pi, it["count"], it["inv"], rng)
            r.append((*upsample_assignment(Ai, f, rng), h))
        rows[f"{name} assign. at random pos."] = r
    return rep, rows


# --------------------------------------------------------------------------- #
# main evaluation
# --------------------------------------------------------------------------- #
DEFAULT_ARMS = ("s1_grow", "s1_grow_t1", "oracle_grow", "prior_grow", "s1_random", "oracle_random",
                "upsample", "s1_upsample", "retrieval", "s1_merge", "oracle_merge", "s1_merge_random",
                "oracle_merge_random")


def _agg(rows: List[dict], idx=None) -> dict:
    rows = [rows[i] for i in idx] if idx is not None else rows
    keys = sorted({k for r in rows for k in r})
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys} | {"n": len(rows)}


def _blank():
    from blockgen.utils.data import Structure
    return Structure(block_ids=np.zeros((1, 1, 1), np.int32), block_data=np.zeros((1, 1, 1), np.int32))


def save_figure(samples: Dict[str, list], path, tile_px: int = 160):
    """eval.save_samples with empty structures drawn as blank tiles."""
    from PIL import Image, ImageDraw
    from blockgen.renderer.grid import default_tile_fn, render_grid
    base = default_tile_fn()

    def tile(s, px):
        if not np.asarray(s.block_ids).any():
            return np.full((px, px, 3), 255, np.uint8)
        return base(s, px)
    rows = []
    for arm, structs in samples.items():
        if not structs:
            continue
        img = render_grid(structs, cols=len(structs), tile_px=tile_px, verbose=False, tile_fn=tile)
        canvas = Image.new("RGB", (img.shape[1] + 150, img.shape[0]), (255, 255, 255))
        canvas.paste(Image.fromarray(img), (150, 0))
        ImageDraw.Draw(canvas).text((6, img.shape[0] // 2 - 6), arm, fill=(0, 0, 0))
        rows.append(canvas)
    out = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), (255, 255, 255))
    y = 0
    for r in rows:
        out.paste(r, (0, y)); y += r.height
    out.save(path)


def evaluate_cells(run: Path, ds, a, device):
    tag = getattr(a, "tag", "") or ""
    from .coarse import coarse_structure, upsample_baseline
    from .grow_cells import grow_decode, load_grow
    from .stage1 import load_stage1, stage1_items, stage1_probs
    f, T = a.f, ds["vocab"].n_types
    hs = ds["test"][:a.n]
    items = stage1_items(hs, f, T)
    conds = [it["coarse"] for it in items]
    A_true = [it["A"] for it in items]
    s1 = load_stage1(Path(a.stage1) / "stage1.pt", device)
    assert s1.cfg.f == f
    P = stage1_probs(s1, items, device)
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    A_s1 = [sample_assignment(p, it["count"], it["inv"], rng) for p, it in zip(P, items)]
    A_prior = [sample_assignment(prior_cell_dist(it["coarse"], it["inv"]), it["count"], it["inv"], rng)
               for it in items]
    print(f"[stage1] sampled assignments in {time.time() - t0:.0f}s", flush=True)
    grow = load_grow(Path(a.grow) / "grow.pt", device) if a.grow else None
    merges = {}
    for mr in a.merge:
        from .merge_cells import load_merge
        cfgj = json.loads((Path(mr) / "config.json").read_text())
        key = "merge_dagger" if cfgj["args"].get("dagger") else "merge"
        merges[key] = load_merge(Path(mr) / "merge.pt", device)
    small = [i for i, h in enumerate(hs) if len(h) <= a.merge_max_n]
    arms = list(a.arms or DEFAULT_ARMS)
    if "merge_dagger" in merges and a.arms is None:
        arms += ["s1_merge_dagger", "oracle_merge_dagger"]
    res = {"n": len(hs), "f": f, "steps": a.steps, "merge_subset_n": len(small),
           "merge_max_n": a.merge_max_n, "seed": a.seed, "arms": {}, "arms_merge_subset": {}}
    rows_all, shows = {}, {"condition": [coarse_structure(c) for c in conds[:8]],
                           "real": [to_structure(h.coords, h.types, h) for h in hs[:8]]}
    retr = None
    for arm in arms:
        t0 = time.time()
        sub = list(range(len(hs)))
        extra = {}
        if "merge" in arm:
            key = "merge_dagger" if arm.endswith("dagger") else "merge"
            pol = "random" if arm.endswith("random") else "model"
            if key not in merges and pol == "model":
                continue
            if pol == "random" and not merges:
                continue
            from .merge_cells import merge_cells_decode
            Asg = A_s1 if arm.startswith("s1") else A_true
            sub = small
            outs = merge_cells_decode(merges.get(key) if pol == "model" else None, [hs[i] for i in sub],
                                      [Asg[i] for i in sub], device, f, policy=pol, seed=a.seed, steps=a.steps)
            gens = [(o["coords"], o["types"]) for o in outs]
            extra["merge_complete"] = [float(o["complete"]) for o in outs]
            extra["clusters"] = [o["n_clusters"] for o in outs]
            extra["rejected"] = [o["rejected"] for o in outs]
            extra["dropped"] = [o["dropped"] / len(hs[i]) for o, i in zip(outs, sub)]
        elif arm.endswith(("grow", "grow_t1", "random", "grow_k1")):
            # "_k1": one block per model call (no parallel placement)
            Asg = {"s1": A_s1, "oracle": A_true, "prior": A_prior}[arm.split("_")[0]]
            pol = "random" if arm.endswith("random") else "model"
            if pol == "model" and grow is None:
                continue
            outs = grow_decode(grow if pol == "model" else None, hs, Asg, conds, device, f,
                               temperature=1.0 if arm.endswith("t1") else 0.0,
                               steps=10 ** 9 if arm.endswith("k1") else a.steps, seed=a.seed, policy=pol)
            gens = [(o["coords"], o["types"]) for o in outs]
            extra["n_seeds"] = [o["n_seeds"] for o in outs]
        elif arm == "upsample":
            gens = []
            for k, (h, c) in enumerate(zip(hs, conds)):
                u = upsample_baseline(np.asarray(h.types, np.int64), c, seed=a.seed + k)
                gens.append((u["pos"][u["placed"]], np.asarray(h.types, np.int64)[u["placed"]]))
        elif arm == "s1_upsample":
            r2 = np.random.default_rng(a.seed + 1)
            gens = [upsample_assignment(A, f, r2) for A in A_s1]
        elif arm == "retrieval":
            if retr is None:
                retr = CoarseRetriever(ds["train"], f, T)
            got = [retr(c) for c in conds]
            gens = [(g.coords, g.types) for g in got]
        else:
            raise ValueError(arm)
        rows = []
        for j, (k, (c, t)) in enumerate(zip(sub, gens)):
            m = build_metrics(c, t, hs[k], conds[k], T, A_true[k])
            if arm == "retrieval":           # a real, complete house of its own size
                m.update(placed_frac=1.0, complete=1.0, single=1.0, connectivity=1.0,
                         inventory_l1=float(np.abs(inventory(got[k].types, T) - inventory(hs[k].types, T)).sum()
                                            / (2 * max(len(hs[k]), len(got[k])))))
            for kk, v in extra.items():
                m[kk] = v[j]
            rows.append(m)
        rows_all[arm] = {"idx": sub, "rows": rows}
        res["arms"][arm] = _agg(rows) | {"sec": time.time() - t0}
        pos = {k: j for j, k in enumerate(sub)}
        res["arms_merge_subset"][arm] = _agg(rows, [pos[k] for k in small if k in pos])
        print(f"[arm] {arm}: " + json.dumps({k: round(v, 3) for k, v in res["arms"][arm].items()}), flush=True)
        shows[arm] = [to_structure(*gens[pos[k]], (got[k] if arm == "retrieval" else hs[k])) if k in pos
                      else _blank() for k in range(min(8, len(hs)))]     # blank: house not in the arm's subset
        (run / f"results{tag}.json").write_text(json.dumps(res, indent=1))
    # diversity: k samples for the same input
    nd = min(a.n_div, len(hs))
    div = {}
    for arm in ("s1_grow_t1", "oracle_grow_t1", "s1_grow", "s1_random") if nd > 0 else ():
        if "grow" in arm and grow is None:
            continue
        t0 = time.time()
        samples = []
        for r in range(a.k_div):
            rr = np.random.default_rng(a.seed + 100 + r)
            if arm.startswith("s1"):
                Asg = [sample_assignment(P[i], items[i]["count"], items[i]["inv"], rr) for i in range(nd)]
            else:
                Asg = A_true[:nd]
            outs = grow_decode(None if arm.endswith("random") else grow, hs[:nd], Asg, conds[:nd], device, f,
                               temperature=1.0 if arm.endswith("t1") else 0.0, steps=a.steps,
                               seed=a.seed + 1000 * (r + 1), policy="random" if arm.endswith("random") else "model")
            samples.append([(o["coords"], o["types"]) for o in outs])
        pw_iou, pw_tf, real_iou = [], [], []
        for i in range(nd):
            for x in range(a.k_div):
                gr = (h_ := hs[i]).coords, h_.types
                real_iou.append(pair_similarity(samples[x][i], gr)["iou"])
                for y in range(x + 1, a.k_div):
                    s = pair_similarity(samples[x][i], samples[y][i])
                    pw_iou.append(s["iou"]); pw_tf.append(s["typed_f1"])
        div[arm] = {"n": nd, "k": a.k_div, "pairwise_iou": float(np.mean(pw_iou)),
                    "pairwise_typed_f1": float(np.mean(pw_tf)), "iou_to_real": float(np.mean(real_iou)),
                    "sec": time.time() - t0}
        print(f"[div] {arm}: {div[arm]}", flush=True)
        if arm == "s1_grow_t1":
            for r in range(1, a.k_div):
                shows[f"s1_grow_t1 sample {r + 1}"] = [to_structure(*samples[r][k], hs[k]) for k in range(min(8, nd))]
    res["diversity"] = div
    (run / f"results{tag}.json").write_text(json.dumps(res, indent=1))
    (run / f"rows{tag}.json").write_text(json.dumps(rows_all))
    save_figure(shows, run / "samples.png")
    print(f"[done] {run}/results{tag}.json samples.png", flush=True)
    return res
