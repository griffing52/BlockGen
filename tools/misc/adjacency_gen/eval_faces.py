"""Evaluation of the face model against the coarse-shape baselines.

Every arm turns a held-out (inventory, coarse condition) pair into realized blocks in
the window frame. Metrics per chunk (means over the test chunks, W1 over the shape
statistics):

* ``placed_frac``: share of the inventory in the realized build (realizability).
  ``collisions``: blocks dropped because their cell was taken. ``connectivity``: largest
  face-connected component of the realized build over the inventory size.
* ``coarse_iou``: IoU between the realized build's coarse occupancy and the condition's;
  ``coarse_l1``: sum of |count differences| over the inventory size; ``out_frac``: share
  of realized blocks outside the window.
* ``voxel_iou`` and ``typed_f1`` (Dice of (cell, type)) against the real chunk in the
  window frame, and ``aligned_iou`` (T50's yaw- and shift-invariant IoU). The condition
  pins the shape, so these are now meaningful, but the task is still multimodal:
  ``self_iou`` (two temperature-1 samples of the same input) measures how far apart two
  plausible answers are.
* ``glue_f1``: F1 of the typed glued face pairs ``(type +a side, type -a side, a)``
  against the real chunk (same-type blocks are exchangeable, so multisets are compared).
* ``exposed_err``: |exposed-face fraction realized - real|.
* shape stats of the realized build (fill, support, height, voxels) and their W1 to the
  real chunks.

Frame-free arms (one-shot, random tree, the T50 model) are translated to the shift that
best matches the coarse condition (:func:`decode_faces.align_to_condition`); that uses the
condition, which favours them, and it is also applied to the unconditioned model's
one-shot output. Sequential arms place their seed at an absolute cell and are not moved.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .coarse import Coarse, coarsen, upsample_baseline
from .data import components, graph_to_structure, touching
from .decode_faces import align_to_condition, oneshot_realize, oneshot_scores, sequential_decode
from .eval import aligned_iou, shape_stats
from .faces import exposed_faces, multiset_f1, typed_glue

SEQ_ARMS = ("seq_check", "seq_check_t1", "seq_nocheck", "seq_check_cmask", "seq_random_check",
            "seq_random_cmask")
MERGE_ARMS = ("merge_check", "merge_check_t1", "merge_nocheck", "merge_random_check")
# merge arms with "noext" in the name keep the overlap check but drop the window-extent test
ONESHOT_ARMS = ("oneshot_check", "oneshot_nocheck")
BASE_ARMS = ("upsample", "retrieval_counts", "retrieval_coarse", "random_tree_check")


def realized_metrics(coords: np.ndarray, types: np.ndarray, n_inv: int, it: dict, cond: Coarse,
                     size: int = 8) -> dict:
    coords = np.asarray(coords, np.int64).reshape(-1, 3)
    types = np.asarray(types, np.int64)
    real_c, real_t = it["coords"], it["types"]
    M = len(coords)
    out = {"placed_frac": M / n_inv}
    if M == 0:
        return {**out, "connectivity": 0.0, "coarse_iou": 0.0, "coarse_l1": 1.0, "out_frac": 0.0,
                "voxel_iou": 0.0, "typed_f1": 0.0, "aligned_iou": 0.0, "glue_f1": 0.0,
                "glue_p": 0.0, "glue_r": 0.0, "exposed_err": 1.0, "n_vox": 0, "height": 0,
                "fill": 0.0, "support": 0.0}
    cc = components(touching(coords, [np.zeros((1, 3), np.int64)] * M))
    out["connectivity"] = max(len(c) for c in cc) / n_inv
    c = coarsen(coords, types, cond.f, size=size)
    out["coarse_iou"] = float((c.occ & cond.occ).sum() / max((c.occ | cond.occ).sum(), 1))
    out["coarse_l1"] = float(np.abs(c.count - cond.count).sum() / n_inv)
    out["out_frac"] = float(1 - np.all((coords >= 0) & (coords < size), 1).mean())
    A = {tuple(v) for v in coords.tolist()}
    B = {tuple(v) for v in real_c.tolist()}
    out["voxel_iou"] = len(A & B) / len(A | B)
    At = {(tuple(v), int(t)) for v, t in zip(coords.tolist(), types)}
    Bt = {(tuple(v), int(t)) for v, t in zip(real_c.tolist(), real_t)}
    out["typed_f1"] = 2 * len(At & Bt) / (len(At) + len(Bt))
    out["aligned_iou"] = aligned_iou(real_c, coords)
    p, r, f1 = multiset_f1(typed_glue(coords, types), typed_glue(real_c, real_t))
    out.update(glue_p=p, glue_r=r, glue_f1=f1)
    out["exposed_err"] = abs(float(exposed_faces(coords).mean()) - float(it["exposed"].mean()))
    out.update(shape_stats(coords))
    return out


def _raw(it_graph, idx):
    return [it_graph.blocks[i] for i in idx]


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
class Retriever:
    def __init__(self, train_items: Sequence[dict], train_graphs, n_types: int, f: int, vocab, size: int = 8):
        from .coarse import graph_coarse
        self.graphs = train_graphs
        self.items = train_items
        self.n_types = n_types
        self.M = np.zeros((len(train_items), n_types), np.float32)
        for k, it in enumerate(train_items):
            np.add.at(self.M[k], it["types"], 1)
        cs = [graph_coarse(g, f, vocab, size) for g in train_graphs]
        self.C = np.stack([c.count.ravel() for c in cs]).astype(np.float32)
        self.D = np.stack([c.dom.ravel() for c in cs])

    def by_counts(self, it) -> int:
        v = np.bincount(it["types"], minlength=self.n_types).astype(np.float32)
        return int(np.abs(self.M - v).sum(1).argmin())

    def by_coarse(self, cond: Coarse) -> int:
        c, d = cond.count.ravel().astype(np.float32), cond.dom.ravel()
        dist = np.abs(self.C - c).sum(1) + ((self.D != d) & (c > 0)).sum(1)
        return int(dist.argmin())


def t50_runner(t50_run: Path, our_vocab, device):
    """The T50 voxel model (block types only, no positions) on our inventories, with its
    check decoder. Returns fn(types) -> (pos, keep). Trained on 6^3 windows with 20..80
    blocks, so inventories above 80 are out of its training range."""
    from .decode import decode, model_scorer
    from .train import ShapeIndex, load_dataset, load_model
    ds = load_dataset(t50_run / "dataset.pkl")
    v50 = ds["vocab"]
    sidx = ShapeIndex(v50)
    model = load_model(t50_run, device)
    scorer = model_scorer(model, sidx.type_valid(), device)
    tmap = np.array([v50.token_to_type.get(n, 0) for n in our_vocab.names])

    def run(types, seed=0):
        t = tmap[types]
        r = decode(t, v50, scorer, steps=24, check=True, seed=seed)
        comp = r["comp"]
        vals, cnts = np.unique(comp, return_counts=True)
        keep = comp == vals[np.argmax(cnts)]
        return r["pos"], keep
    return run


def diversity(model, items, conds, device, n_items: int = 16, k: int = 4, seed: int = 0, bs: int = 32,
              size: int = 8) -> dict:
    """Sample k builds (merge decoder, check, temperature 1) for the first n_items held-out
    inventories; mean pairwise voxel IoU between samples after alignment to the condition,
    the pose-invariant aligned IoU, and the same against the real chunk for scale."""
    from .merge import merge_decode
    sub = list(items[:n_items])
    builds = [[] for _ in sub]
    for r in range(k):
        outs = merge_decode(model, sub, device, check=True, temperature=1.0, seed=seed + 1000 * (r + 1), bs=bs,
                            size=size)
        for q, (o, c) in enumerate(zip(outs, conds)):
            vals, cnts = np.unique(o["comp"], return_counts=True)
            keep = (o["comp"] == vals[np.argmax(cnts)]) & ~o["conflict"]
            builds[q].append(align_to_condition(o["pos"], keep, c, size=size)[keep])
    pw, pa, rv = [], [], []
    for q, bl in enumerate(builds):
        for x in range(k):
            A = {tuple(v) for v in bl[x].tolist()}
            R = {tuple(v) for v in sub[q]["coords"].tolist()}
            rv.append(len(A & R) / max(len(A | R), 1))
            for y in range(x + 1, k):
                B = {tuple(v) for v in bl[y].tolist()}
                pw.append(len(A & B) / max(len(A | B), 1))
                pa.append(aligned_iou(bl[x], bl[y]))
    return {"n_items": len(sub), "k": k, "pairwise_iou": float(np.mean(pw)),
            "pairwise_aligned_iou": float(np.mean(pa)), "iou_to_real": float(np.mean(rv)),
            "distinct_frac": float(np.mean([p < 0.999 for p in pw]))}


# --------------------------------------------------------------------------- #
# main evaluation
# --------------------------------------------------------------------------- #
def exposed_head_metrics(scores, items, train_items) -> dict:
    """Accuracy of the exposed head (empty prefix) against a (type, direction) majority
    prior fitted on train chunks."""
    tab = {}
    for it in train_items:
        for t, e in zip(it["types"], it["exposed"]):
            r = tab.setdefault(int(t), np.zeros((2, 6)))
            r[0] += e; r[1] += 1
    ok = okp = tot = 0
    for s, it in zip(scores, items):
        pred = s["ex"] > 0
        ok += int((pred == it["exposed"]).sum())
        pri = np.stack([(tab[int(t)][0] / tab[int(t)][1] > 0.5) if int(t) in tab else np.ones(6, bool)
                        for t in it["types"]])
        okp += int((pri == it["exposed"]).sum())
        tot += it["exposed"].size
    return {"exposed_acc": ok / tot, "exposed_acc_prior": okp / tot,
            "exposed_base_rate": float(np.mean(np.concatenate([it["exposed"].ravel() for it in items])))}


def evaluate_faces(run: Path, model, test_graphs, test_items, train_graphs, train_items, vocab, device,
                   arms: Sequence[str], f_eval: int, seed: int = 0, n_show: int = 8,
                   t50_run: Optional[Path] = None, tag: str = "", decode_bs: int = 32) -> dict:
    from scipy.stats import wasserstein_distance
    from .decode import random_tree
    from .train import ShapeIndex
    conds = [it["coarse"] for it in test_items]
    assert all(c.f == f_eval for c in conds)
    size = conds[0].G * conds[0].f
    real_stats = [shape_stats(it["coords"]) for it in test_items]
    res = {"n": len(test_items), "f_eval": f_eval, "real": {k: float(np.mean([s[k] for s in real_stats]))
                                                              for k in real_stats[0]}}
    samples = {"condition": [], "real": []}
    from .coarse import coarse_structure
    for g, c in zip(test_graphs[:n_show], conds[:n_show]):
        samples["condition"].append(coarse_structure(c))
        samples["real"].append(graph_to_structure(g.coords, g.blocks))
    retr = None
    vt = ShapeIndex(vocab).type_valid()
    cache = {}
    if model is not None:
        sc = oneshot_scores(model, test_items, device, bs=decode_bs)
        res["exposed_head"] = exposed_head_metrics(sc, test_items, train_items[:4000])
        cache["scores"] = sc
    t50 = None
    for arm in arms:
        t0 = time.time()
        rows, shows, extra = [], [], Counter()
        if arm.startswith("merge"):
            from .merge import merge_decode
            outs = merge_decode(None if "random" in arm else model, test_items, device,
                                check="nocheck" not in arm, temperature=1.0 if "t1" in arm else 0.0,
                                seed=seed, policy="random" if "random" in arm else "model", bs=decode_bs,
                                size=size, extent="noext" not in arm, cmask="cmask" in arm)
            if "t1" in arm:
                res[arm + "_diversity"] = diversity(model, test_items, conds, device, seed=seed, bs=decode_bs,
                                                    size=size)
                print(f"[div] {res[arm + '_diversity']}", flush=True)
        if arm.startswith("seq"):
            outs = sequential_decode(None if "random" in arm else model, test_items, device,
                                     check="nocheck" not in arm, temperature=1.0 if arm.endswith("t1") else 0.0,
                                     seed=seed, policy="random" if "random" in arm else "model",
                                     cmask="cmask" in arm)
            if arm.endswith("t1"):
                outs2 = sequential_decode(model, test_items, device, check=True, temperature=1.0,
                                          seed=seed + 7919)
        for k, (g, it, cond) in enumerate(zip(test_graphs, test_items, conds)):
            N = len(g)
            idx = None
            if arm.startswith("seq"):
                o = outs[k]
                keep = o["placed"] & ~o["conflict"]
                pos = o["pos"]
                extra["collisions"] += int(o["conflict"].sum())
                extra["unplaced"] += int((~o["placed"]).sum())
            elif arm.startswith("merge"):
                o = outs[k]
                vals, cnts = np.unique(o["comp"], return_counts=True)
                keep = (o["comp"] == vals[np.argmax(cnts)]) & ~o["conflict"]
                pos = align_to_condition(o["pos"], keep, cond, size=size)
                extra["collisions"] += int(o["conflict"].sum())
                extra["rejected_merges"] += o["rejected"]
                extra["complete"] += int(o["merges"] == N - 1)
            elif arm.startswith("oneshot"):
                s = cache["scores"][k]
                r = oneshot_realize(s["X"], s["ex"], check=arm.endswith("_check"))
                pos = align_to_condition(r["pos"], r["keep"], cond, size=size)
                keep = r["keep"]
                extra["collisions"] += int(r["conflict"].sum())
                extra["rejected_merges"] += r["rejected"]
                pm = Counter((int(it["types"][i]), int(it["types"][j]), a) for a, i, j, _ in r["matches"])
                extra["match_f1_raw"] += multiset_f1(pm, typed_glue(it["coords"], it["types"]))[2]
            elif arm == "upsample":
                u = upsample_baseline(it["types"], cond, seed=seed + k)
                pos, keep = u["pos"], u["placed"]
            elif arm == "random_tree_check":
                r = random_tree(it["types"], vocab, vt, check=True, seed=seed + k)
                keep = ~r["conflict"]
                pos = align_to_condition(r["pos"], keep, cond, size=size)
                extra["collisions"] += int(r["conflict"].sum())
            elif arm == "t50_check":
                if t50 is None:
                    t50 = t50_runner(t50_run, vocab, device)
                p0, keep = t50(it["types"], seed=seed + k)
                pos = align_to_condition(p0, keep, cond, size=size)
            elif arm.startswith("retrieval"):
                if retr is None:
                    retr = Retriever(train_items, train_graphs, vocab.n_types, f_eval, vocab, size)
                ti = retr.by_counts(it) if arm == "retrieval_counts" else retr.by_coarse(cond)
                tg = train_graphs[ti]
                m = realized_metrics(tg.coords, tg.types, N, it, cond, size=size)
                m["placed_frac"] = 1.0
                m["connectivity"] = 1.0
                m["inventory_l1"] = int(np.abs(np.bincount(tg.types, minlength=vocab.n_types)
                                               - np.bincount(g.types, minlength=vocab.n_types)).sum())
                rows.append(m)
                if k < n_show:
                    shows.append(graph_to_structure(tg.coords, tg.blocks))
                continue
            else:
                raise ValueError(arm)
            idx = np.nonzero(keep)[0]
            m = realized_metrics(pos[idx], it["types"][idx], N, it, cond, size=size)
            if arm.startswith("seq") and arm.endswith("t1"):
                o2 = outs2[k]
                A = {tuple(v) for v in pos[idx].tolist()}
                B = {tuple(v) for v in o2["pos"][o2["placed"]].tolist()}
                m["self_iou"] = len(A & B) / max(len(A | B), 1)
            rows.append(m)
            if k < n_show:
                shows.append(graph_to_structure(pos, g.blocks, keep=list(idx)))
        keys = sorted({kk for r in rows for kk in r})
        out = {kk: float(np.nanmean([r.get(kk, np.nan) for r in rows])) for kk in keys}
        for kk in ("n_vox", "height", "fill", "support"):
            out[f"w1_{kk}"] = float(wasserstein_distance([r[kk] for r in rows], [s[kk] for s in real_stats]))
        out.update({f"{kk}_per_chunk": v / len(rows) for kk, v in extra.items()})
        out["sec"] = time.time() - t0
        res[arm] = out
        samples[arm] = shows
        print(f"[gen] {arm}: " + json.dumps({kk: round(v, 3) for kk, v in out.items()}), flush=True)
    (run / f"results{tag}.json").write_text(json.dumps(res, indent=1))
    from .eval import save_samples
    save_samples(samples, run / f"samples{tag}.png")
    print(f"[done] {run}/results{tag}.json samples{tag}.png", flush=True)
    return res
