"""Phase-0 gate + ordering bake-off (implementation_plan.md §7).

Two jobs, both training-free, both cheap:

**1. Round-trip gate.** Every ordering must satisfy encode -> decode -> occupancy
IoU == 1.0 on real corpus builds. This gates everything downstream in the plan.
Also reports the op-length distribution, which is the number that decides whether
"train on the full 40k" survives contact with p90 max-dim 256.

**2. The bake-off.** ``implementation_plan.md`` §2 asserts one canonical order
(bottom-center BFS) but notes.md §8/T11 measured BFS-from-ground as the *worst*
non-broken arm for raster AR. The plan argues the finding does not transfer. That
argument is untested and load-bearing, so we test it before spending GPU, on two
training-free proxies:

  * **bits-per-op** -- fit an order-k context model over each ordering's op stream
    and measure compressibility. Lower = the next op is more predictable given
    local context = easier to learn. This is a direct proxy for learnability and
    would plausibly have *predicted* T11's BFS failure from data alone.
  * **human-order agreement** -- 3D-Craft ships real human placement order
    (``corpora.load_3dcraft_order``), the only ground truth in the building for
    "how do people actually build". Measures rank correlation between each
    synthetic ordering's voxel sequence and the human one.

Neither proxy is the real objective (sample quality is), so this ranks candidates
and de-risks the assumption; it does not replace the training run.

Usage:
    python -m scripts.ordering_bakeoff --limit 300
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from blockgen.curation.houses import load_structures_from_cache
from blockgen.utils.attach_order import (ORDERINGS, OP_ATTACH, OP_CLOSE, OP_SEED,
                                         Op, attach_ops_to_structure, choose_seed,
                                         roundtrip_iou, structure_to_attach_ops)
from blockgen.utils.runs import new_run_dir


def op_key(op: Op) -> tuple:
    """Symbol identity for the entropy model: what the model would have to predict."""
    if op.kind == OP_CLOSE:
        return (OP_CLOSE, op.direction)
    return (op.kind, op.piece, op.direction)


def bits_per_op(streams: List[List[Op]], context: int = 2) -> Dict[str, float]:
    """Order-k context-model bits/op with add-alpha smoothing, train/test split.

    Held out so a richer context cannot win by memorizing. This is the
    learnability proxy: how many bits does the next op cost given the previous k?
    """
    split = max(1, int(0.8 * len(streams)))
    train, test = streams[:split], streams[split:] or streams[:1]

    counts: Dict[tuple, Counter] = defaultdict(Counter)
    vocab = set()
    for s in train:
        keys = [op_key(o) for o in s]
        vocab.update(keys)
        for i, k in enumerate(keys):
            ctx = tuple(keys[max(0, i - context):i])
            counts[ctx][k] += 1
    V = max(1, len(vocab))
    alpha = 0.1

    total_bits = 0.0
    total_ops = 0
    for s in test:
        keys = [op_key(o) for o in s]
        for i, k in enumerate(keys):
            ctx = tuple(keys[max(0, i - context):i])
            c = counts.get(ctx)
            # Back off to shorter contexts when the full one is unseen.
            while c is None and ctx:
                ctx = ctx[1:]
                c = counts.get(ctx)
            if c is None:
                p = 1.0 / V
            else:
                p = (c.get(k, 0) + alpha) / (sum(c.values()) + alpha * V)
            total_bits += -math.log2(max(p, 1e-12))
            total_ops += 1
    return {"bits_per_op": total_bits / max(1, total_ops), "n_ops": total_ops}


def human_agreement(limit: int = 200) -> Dict[str, float]:
    """Spearman correlation between each synthetic ordering and human build order.

    3D-Craft gives per-house chronological placement. We encode the same house
    under each ordering, extract the voxel visit sequence, and correlate ranks over
    the voxels both sequences contain.
    """
    from blockgen.utils.corpora import DEFAULT_3DCRAFT, load_3dcraft_order
    from blockgen.utils.attach_order import _connected_components

    root = Path(DEFAULT_3DCRAFT)
    if not root.exists():
        return {"error": f"3dcraft not found at {root}"}
    dirs = sorted([d for d in root.iterdir() if d.is_dir()])[:limit]

    scores: Dict[str, List[float]] = defaultdict(list)
    used = 0
    for d in dirs:
        try:
            human = load_3dcraft_order(d)
        except Exception:
            continue
        if len(human) < 40:
            continue
        coords = [c for c, _ in human]
        lo = np.min(np.array(coords), axis=0)
        local = [tuple(int(v) for v in (np.array(c) - lo)) for c in coords]
        dim = tuple(int(v) + 1 for v in np.max(np.array(local), axis=0))
        if max(dim) > 96:
            continue
        ids = np.zeros(dim, dtype=np.int32)
        for (x, y, z), (bid, _bd) in zip(local, [b for _, b in human]):
            ids[x, y, z] = bid if bid != 0 else 1

        from blockgen.utils.data import Structure
        s = Structure(block_ids=ids, block_data=np.zeros_like(ids))
        human_rank = {c: i for i, c in enumerate(local)}

        for name in ORDERINGS:
            try:
                seq = _visit_sequence(s, name)
            except Exception:
                continue
            pairs = [(human_rank[c], i) for i, c in enumerate(seq) if c in human_rank]
            if len(pairs) < 20:
                continue
            a = np.array([p[0] for p in pairs], dtype=float)
            b = np.array([p[1] for p in pairs], dtype=float)
            ra, rb = _rankdata(a), _rankdata(b)
            if ra.std() == 0 or rb.std() == 0:
                continue
            scores[name].append(float(np.corrcoef(ra, rb)[0, 1]))
        used += 1

    out = {k: float(np.mean(v)) for k, v in scores.items() if v}
    out["_n_houses"] = used
    return out


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = a.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    return ranks


def _visit_sequence(structure, ordering: str) -> List[tuple]:
    """The order in which voxels are PLACED under ``ordering`` (cropped frame)."""
    import heapq
    from blockgen.utils.attach_order import (LIFO_ORDERINGS, OrderingContext,
                                             PORT_DIRECTIONS, _in_bounds)

    s = structure.crop_to_non_air()
    occ = s.occupied_mask
    prio = ORDERINGS[ordering]
    lifo = ordering in LIFO_ORDERINGS

    comps = _rt_components(occ)
    comps.sort(key=len, reverse=True)
    comp = comps[0]
    comp_occ = np.zeros_like(occ)
    for c in comp:
        comp_occ[c] = True
    seed = choose_seed(comp_occ)
    ctx = OrderingContext(seed=seed)

    placed = {seed}
    seq = [seed]
    frontier: List[tuple] = []
    counter = 0
    visited = set()

    def push(v):
        nonlocal counter
        for d, (dx, dy, dz) in enumerate(PORT_DIRECTIONS):
            child = (v[0] + dx, v[1] + dy, v[2] + dz)
            if (v, d) in visited or child in placed:
                continue
            visited.add((v, d))
            heapq.heappush(frontier, (prio(v, d, child, ctx),
                                      -counter if lifo else counter, v, d))
            counter += 1

    push(seed)
    while frontier:
        _p, _c, parent, d = heapq.heappop(frontier)
        dx, dy, dz = PORT_DIRECTIONS[d]
        child = (parent[0] + dx, parent[1] + dy, parent[2] + dz)
        if child in placed:
            continue
        if _in_bounds(child, occ) and occ[child]:
            placed.add(child)
            seq.append(child)
            push(child)
    return seq


def _rt_components(occ):
    from blockgen.utils.attach_order import _connected_components
    return _connected_components(occ)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/minecraft/cache/all_32.npz")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--context", type=int, default=2)
    ap.add_argument("--human-limit", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.out) if args.out else new_run_dir("ordering_bakeoff")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bakeoff] run dir: {run_dir}", flush=True)

    structures, _manifest = load_structures_from_cache(args.cache)
    structures = structures[:args.limit]
    print(f"[bakeoff] {len(structures)} structures from {args.cache}", flush=True)

    results: Dict[str, dict] = {}
    for name in ORDERINGS:
        t0 = time.time()
        ious, lens, attach, close, cov = [], [], [], [], []
        streams: List[List[Op]] = []
        fails = 0
        for s in structures:
            try:
                ops, info = structure_to_attach_ops(s, ordering=name)
                iou = roundtrip_iou(s, ordering=name)
            except Exception:
                fails += 1
                continue
            ious.append(iou)
            lens.append(info["n_ops"])
            attach.append(info["n_attach"])
            close.append(info["n_close"])
            cov.append(info["coverage"])
            streams.append(ops)

        ent = bits_per_op(streams, context=args.context)
        results[name] = {
            "roundtrip_iou_mean": float(np.mean(ious)) if ious else 0.0,
            "roundtrip_iou_min": float(np.min(ious)) if ious else 0.0,
            "roundtrip_pass_rate": float(np.mean([i > 0.9999 for i in ious])) if ious else 0.0,
            "ops_mean": float(np.mean(lens)) if lens else 0.0,
            "ops_p50": float(np.percentile(lens, 50)) if lens else 0.0,
            "ops_p90": float(np.percentile(lens, 90)) if lens else 0.0,
            "ops_max": int(np.max(lens)) if lens else 0,
            "attach_frac": float(np.sum(attach) / max(1, np.sum(attach) + np.sum(close))),
            "coverage_mean": float(np.mean(cov)) if cov else 0.0,
            "bits_per_op": ent["bits_per_op"],
            "n_fail": fails,
            "secs": round(time.time() - t0, 1),
        }
        r = results[name]
        print(f"[bakeoff] {name:20s} iou={r['roundtrip_iou_mean']:.4f} "
              f"pass={r['roundtrip_pass_rate']:.3f} bits/op={r['bits_per_op']:.4f} "
              f"ops_p50={r['ops_p50']:.0f} cov={r['coverage_mean']:.3f} "
              f"({r['secs']}s)", flush=True)

    print("[bakeoff] human-order agreement (3D-Craft)...", flush=True)
    human = human_agreement(limit=args.human_limit)
    print(f"[bakeoff] human agreement: {human}", flush=True)

    payload = {"orderings": results, "human_agreement": human,
               "n_structures": len(structures), "cache": args.cache,
               "context": args.context}
    (run_dir / "bakeoff.json").write_text(json.dumps(payload, indent=2))
    print(f"[bakeoff] wrote {run_dir / 'bakeoff.json'}", flush=True)


if __name__ == "__main__":
    main()
