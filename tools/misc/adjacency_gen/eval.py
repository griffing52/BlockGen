"""Generation metrics, baselines and the samples.png figure.

Every arm turns a held-out node multiset into placed nodes (``pos``), a component id
per node (``comp``) and a conflict flag per node. Metrics:

* realizability: fraction of nodes placed without a voxel collision.
* connectivity: fraction of nodes in the largest component (of the committed edges).
* collisions: count of colliding nodes per graph.
* implied-edge consistency (closure): of the face-touching pairs in the realized
  largest component, the fraction whose label the scorer itself predicted at its last
  look while the pair was undecided (``implied_consistency``: argmax over all labels;
  ``implied_dir``: argmax over the offset labels only; ``implied_p``: mean probability
  of the realized label). Geometry fixes these pairs; this asks whether the model
  agreed.
* edge F1 vs the original, by node identity. Same-type nodes are exchangeable, so this
  is a lower bound; ``typed_edge_jaccard`` (Jaccard of the multisets of
  ``(type_i, type_j, offset)`` edges) is the permutation-invariant counterpart and
  ``aligned_iou`` (best voxel IoU over 4 yaw rotations and small shifts) the
  geometric one. All three are weak under multimodality: a perfect sampler of
  p(build | multiset) still scores low against the one real chunk.
* shape stats on the realized largest component, compared to the real chunks as a
  1-D Wasserstein distance: voxel count, height, bbox fill, support (fraction of
  voxels resting on a voxel or on the bottom layer), and per-type relative height
  error against the original chunk with the same multiset.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

from .data import Graph, NodeVocab, edge_labels, graph_to_structure, touching
from .decode import TypePrior, decode, model_scorer, random_tree, realize


def _cells(pos, shapes, keep) -> np.ndarray:
    out = set()
    for i in keep:
        for v in (np.asarray(shapes[i]) + pos[i]).tolist():
            out.add(tuple(v))
    return np.array(sorted(out)) if out else np.zeros((0, 3), np.int64)


def aligned_iou(a: np.ndarray, b: np.ndarray, shift: int = 2) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    A = {tuple(v) for v in (a - a.min(0)).tolist()}
    best = 0.0
    for r in range(4):
        bb = b.copy()
        for _ in range(r):                      # yaw: (x, z) -> (z, -x)
            bb = np.stack([bb[:, 2], bb[:, 1], -bb[:, 0]], 1)
        bb = bb - bb.min(0)
        for dx in range(-shift, shift + 1):
            for dy in range(-1, 2):
                for dz in range(-shift, shift + 1):
                    Bs = {(x + dx, y + dy, z + dz) for x, y, z in bb.tolist()}
                    inter = len(A & Bs)
                    best = max(best, inter / (len(A) + len(Bs) - inter))
    return best


def shape_stats(cells: np.ndarray) -> dict:
    if len(cells) == 0:
        return {"n_vox": 0, "height": 0, "fill": 0.0, "support": 0.0}
    mn, mx = cells.min(0), cells.max(0)
    ext = mx - mn + 1
    S = {tuple(v) for v in cells.tolist()}
    sup = sum(1 for x, y, z in S if y == mn[1] or (x, y - 1, z) in S)
    return {"n_vox": len(cells), "height": int(ext[1]), "fill": len(cells) / float(np.prod(ext)),
            "support": sup / len(S)}


def type_height_err(types, pos_gen, keep_gen, pos_real) -> float:
    """Mean over types (node-weighted) of |mean rel. height gen - real|."""
    def rel(pos, keep):
        y = pos[keep, 1].astype(float)
        h = max(y.max() - y.min(), 1.0)
        out: Dict[int, list] = {}
        for i, yy in zip(keep, (y - y.min()) / h):
            out.setdefault(int(types[i]), []).append(yy)
        return {k: np.mean(v) for k, v in out.items()}
    if len(keep_gen) < 2:
        return float("nan")
    g, r = rel(pos_gen, list(keep_gen)), rel(pos_real, list(range(len(types))))
    w = Counter(int(types[i]) for i in keep_gen)
    return float(sum(w[k] * abs(g[k] - r[k]) for k in g) / sum(w.values()))


def typed_edges(types, lab) -> Counter:
    c = Counter()
    N = len(types)
    for i in range(N):
        for j in range(N):
            if lab[i, j] > 0 and (types[i], i) < (types[j], j):
                c[(int(types[i]), int(types[j]), int(lab[i, j]))] += 1
    return c


def jaccard(a: Counter, b: Counter) -> float:
    keys = a.keys() | b.keys()
    mx = sum(max(a[k], b[k]) for k in keys)
    return sum(min(a[k], b[k]) for k in keys) / mx if mx else 1.0


def graph_metrics(g: Graph, vocab: NodeVocab, pos, comp, conflict, pred=None) -> dict:
    """All per-graph metrics for one realized arm output."""
    shapes = [vocab.shapes[int(t)] for t in g.types]
    N = len(g)
    lc = Counter(comp.tolist()).most_common(1)[0][0]
    keep = [i for i in range(N) if comp[i] == lc and not conflict[i]]
    real_lab, _ = edge_labels(g.coords, shapes, vocab.space)
    # realized labels among kept nodes
    gl = np.zeros((N, N), dtype=np.int64)
    adj = touching(pos[keep], [shapes[i] for i in keep])
    for a, b in zip(*np.nonzero(adj)):
        u, v = keep[a], keep[b]
        gl[u, v] = max(vocab.space.cls(pos[v] - pos[u]), 0)
    tp = int(((gl > 0) & (real_lab > 0)).sum()); fp = int(((gl > 0) & (real_lab == 0)).sum())
    fn = int(((gl == 0) & (real_lab > 0)).sum())
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    out = {"N": N, "realizability": 1 - float(conflict.mean()),
           "connectivity": Counter(comp.tolist()).most_common(1)[0][1] / N,
           "collisions": int(conflict.sum()),
           "edge_f1_identity": 2 * pr * rc / max(pr + rc, 1e-9),
           "typed_edge_jaccard": jaccard(typed_edges(g.types, gl), typed_edges(g.types, real_lab)),
           "aligned_iou": aligned_iou(_cells(g.coords, shapes, range(N)), _cells(pos, shapes, keep)),
           "type_height_err": type_height_err(g.types, pos, keep, g.coords)}
    if pred is not None:
        m = gl > 0
        if m.any():
            lp = pred[m]                                   # [E,K+1] last log-probs
            y = gl[m]
            out["implied_consistency"] = float((lp.argmax(-1) == y).mean())
            out["implied_dir"] = float((lp[:, 1:].argmax(-1) + 1 == y).mean())
            out["implied_p"] = float(np.exp(lp[np.arange(len(y)), y]).mean())
    out.update(shape_stats(_cells(pos, shapes, keep)))
    out["_keep"] = keep
    return out


def arm_outputs(arm: str, g: Graph, vocab: NodeVocab, ctx: dict, seed: int = 0) -> dict:
    """Run one arm on one held-out multiset; returns pos/comp/conflict(/pred)."""
    shapes = [vocab.shapes[int(t)] for t in g.types]
    N = len(g)
    if arm.startswith("model") or arm.startswith("prior"):
        check = arm.endswith("check") and not arm.endswith("nocheck")
        scorer = ctx["model_scorer"] if arm.startswith("model") else ctx["prior_scorer"]
        temp = ctx.get("prior_temp", 1.0) if arm.startswith("prior") else ctx.get("temp", 0.0)
        grow = arm.endswith("grow")
        check = check or grow
        res = decode(g.types, vocab, scorer, steps=ctx.get("steps", 24), check=check,
                     temperature=temp, seed=seed, grow=grow)
        if check:
            return {"pos": res["pos"], "comp": res["comp"], "conflict": np.zeros(N, bool),
                    "pred": res["pred"], "stats": res["stats"]}
        r = realize(res["labels"], res["conf"], shapes, vocab.space)
        return {"pos": r["pos"], "comp": r["comp"], "conflict": r["conflict"], "pred": res["pred"],
                "stats": {**res["stats"], "edge_ok": r["edge_ok"]}}
    if arm in ("random_tree", "random_tree_check"):
        r = random_tree(g.types, vocab, ctx["vt"], check=arm.endswith("check"), seed=seed)
        return {**r, "pred": None}
    if arm == "retrieval":
        tg = ctx["retrieve"](g)
        return {"graph": tg}
    raise ValueError(arm)


def make_retriever(train_graphs: Sequence[Graph], n_types: int):
    M = np.zeros((len(train_graphs), n_types), np.float32)
    for k, g in enumerate(train_graphs):
        np.add.at(M[k], g.types, 1)

    def retrieve(g: Graph) -> Graph:
        v = np.bincount(g.types, minlength=n_types).astype(np.float32)
        return train_graphs[int(np.abs(M - v).sum(1).argmin())]
    return retrieve


def evaluate_generation(graphs: Sequence[Graph], vocab: NodeVocab, arms: Sequence[str], ctx: dict,
                        seed: int = 0, keep_samples: int = 8) -> dict:
    """Per-arm mean metrics + shape-stat distance to the real chunks; also returns the
    first ``keep_samples`` realized Structures per arm for the figure."""
    from scipy.stats import wasserstein_distance
    real_stats = [shape_stats(_cells(g.coords, [vocab.shapes[int(t)] for t in g.types], range(len(g))))
                  for g in graphs]
    out = {"real": {k: float(np.mean([s[k] for s in real_stats])) for k in real_stats[0]}}
    samples = {"real": [graph_to_structure(g.coords, g.blocks) for g in graphs[:keep_samples]]}
    for arm in arms:
        rows, structs, extra = [], [], Counter()
        for k, g in enumerate(graphs):
            o = arm_outputs(arm, g, vocab, ctx, seed=seed + k)
            if arm == "retrieval":
                tg = o["graph"]
                sh = [vocab.shapes[int(t)] for t in tg.types]
                cells = _cells(tg.coords, sh, range(len(tg)))
                lab, _ = edge_labels(tg.coords, sh, vocab.space)
                real_lab, _ = edge_labels(g.coords, [vocab.shapes[int(t)] for t in g.types], vocab.space)
                m = {"N": len(g), "realizability": 1.0, "connectivity": 1.0, "collisions": 0,
                     "typed_edge_jaccard": jaccard(typed_edges(tg.types, lab), typed_edges(g.types, real_lab)),
                     "aligned_iou": aligned_iou(_cells(g.coords, [vocab.shapes[int(t)] for t in g.types],
                                                       range(len(g))), cells),
                     "multiset_l1": int(np.abs(np.bincount(tg.types, minlength=vocab.n_types)
                                               - np.bincount(g.types, minlength=vocab.n_types)).sum())}
                m.update(shape_stats(cells))
                if k < keep_samples:
                    structs.append(graph_to_structure(tg.coords, tg.blocks))
            else:
                m = graph_metrics(g, vocab, o["pos"], o["comp"], o["conflict"], o.get("pred"))
                for kk, v in (o.get("stats") or {}).items():
                    extra[kk] += v
                if k < keep_samples:
                    structs.append(graph_to_structure(o["pos"], g.blocks, keep=m["_keep"]))
                m.pop("_keep")
            rows.append(m)
        keys = [k for k in rows[0] if not k.startswith("_")]
        res = {k: float(np.nanmean([r.get(k, np.nan) for r in rows])) for k in keys}
        for k in ("n_vox", "height", "fill", "support"):
            res[f"w1_{k}"] = float(wasserstein_distance([r[k] for r in rows], [s[k] for s in real_stats]))
        res["decode_stats_per_graph"] = {k: v / len(graphs) for k, v in extra.items()}
        out[arm] = res
        samples[arm] = structs
        print(f"[gen] {arm}: " + json.dumps({k: round(v, 3) for k, v in res.items()
                                             if isinstance(v, float)}), flush=True)
    return {"metrics": out, "samples": samples}


def prior_masked_metrics(prior: TypePrior, items, vocab, rate: float, seed: int = 0) -> dict:
    """The type-pair prior on the masked-edge task (ignores observed edges)."""
    rng = np.random.default_rng(seed)
    from .train import average_precision
    tp = fp = fn = dok = dn = 0
    cache = {}
    sc, ys = [], []
    for it in items:
        t, lab = it["types"], it["labels"]
        n = len(t)
        m = np.triu(rng.random((n, n)) < rate, 1)
        for i, j in zip(*np.nonzero(m)):
            key = (int(t[i]), int(t[j]))
            if key not in cache:
                lp = prior.logp(*key)
                cache[key] = (int(np.argmax(lp)), 1 - float(np.exp(lp[0])))
            (p, pe), y = cache[key], lab[i, j]
            sc.append(pe); ys.append(y > 0)
            tp += p > 0 and y > 0; fp += p > 0 and y == 0; fn += p == 0 and y > 0
            if p > 0 and y > 0:
                dn += 1; dok += p == y
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"rate": rate, "edge_f1": 2 * pr * rc / max(pr + rc, 1e-9), "edge_prec": pr, "edge_rec": rc,
            "dir_acc": dok / max(dn, 1),
            "edge_ap": average_precision(np.array(sc), np.array(ys))}


def save_samples(samples: Dict[str, list], path, tile_px: int = 160):
    """One labelled row per arm."""
    from PIL import Image, ImageDraw
    from blockgen.renderer.grid import render_grid
    rows = []
    label_w = 150
    for arm, structs in samples.items():
        if not structs:
            continue
        img = render_grid(structs, cols=len(structs), tile_px=tile_px, verbose=False)
        canvas = Image.new("RGB", (img.shape[1] + label_w, img.shape[0]), (255, 255, 255))
        canvas.paste(Image.fromarray(img), (label_w, 0))
        ImageDraw.Draw(canvas).text((6, img.shape[0] // 2 - 6), arm, fill=(0, 0, 0))
        rows.append(canvas)
    W = max(r.width for r in rows)
    out = Image.new("RGB", (W, sum(r.height for r in rows)), (255, 255, 255))
    y = 0
    for r in rows:
        out.paste(r, (0, y)); y += r.height
    out.save(path)
    return path
