"""The f=4 oracle test of the cascade-and-AR hybrid (notes §74 option a, §89 follow-up).

A ``pooled4`` facade model draws a face after its own level f=4. Given the *real*
level f=4 of each test front face, how much of the face does it recover, against
the naive upsample of that level (the copy) and the fixed-pooling cascade's
completion (T88)? And does it draw windows, which the cascade never did?

    python scripts/facade_hybrid_oracle.py --run <pooled4 run> [--n 399] [--temperatures 1.0,0.7]
                                           [--face -z] [--batch 32] [--out DIR]

Writes ``oracle.md``, ``summary.json`` and ``sheet.png`` (real, copy, hybrid from the
real f=4 at each temperature, hybrid from the box) into ``--out`` (default
``<run>/hybrid_oracle_<face>``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blockgen.corpus import facade as fc
from blockgen.eval.facade_stats import face_agreement
from blockgen.train.sample_constrained import face_size, generate_faces_constrained
from facade_cascade import STAT_COLS, agreement_table, draw_sheet, stats_table   # noqa: E402
from facade_constrained import test_houses                                        # noqa: E402

GLASS = {20, 95, 102, 160}          # glass, stained glass, glass pane, stained pane: the window blocks
F = 4


def load(run: str, ckpt: str | None):
    from blockgen.train.__main__ import selected_ckpt
    from blockgen.train.trainer import load_run_checkpoint
    ckpt = ckpt or selected_ckpt(run)
    model, tokenizer, cfg, info = load_run_checkpoint(run, ckpt)
    assert tokenizer.kind == "facades" and tokenizer.layout == "pooled4", (tokenizer.kind, tokenizer.layout)
    return model, tokenizer, cfg, info, ckpt


def window_stats(faces: List[fc.Facade]) -> Dict[str, float]:
    n = [int(np.isin(f.ids, list(GLASS))[f.mask].sum()) for f in faces]
    return {"glass_cells": float(np.mean(n)), "faces_with_glass": float(np.mean([x > 0 for x in n]))}


def empty_like(real: fc.Facade) -> fc.Facade:
    return fc.Facade(face=real.face, depth=np.full(real.depth.shape, fc.EMPTY, np.int16),
                     ids=np.zeros(real.depth.shape, np.int16), data=np.zeros(real.depth.shape, np.int16),
                     extent=real.extent)


def draw(model, tokenizer, cfg, real: List[fc.Facade], shapes, pooled, T: float, top_k: int, seed: int,
         batch: int, given: bool) -> List[fc.Facade]:
    plans = [[(r.face, *pooled[i].depth.shape), (r.face, *r.depth.shape)] for i, r in enumerate(real)]
    giv = [{0: pooled[i]} for i in range(len(real))] if given else None
    seqs, _ = generate_faces_constrained(model, tokenizer, cfg.sample_tags, plans, shapes, T, top_k, seed,
                                         batch=batch, constrain=False, given=giv)
    out = []
    for seq, r in zip(seqs, real):
        faces = tokenizer.decode_faces(seq)[0]
        f = faces[1] if len(faces) > 1 and faces[1].depth.shape == r.depth.shape else None
        out.append(f if f is not None else empty_like(r))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=399)
    ap.add_argument("--face", default="-z")
    ap.add_argument("--temperatures", default="1.0,0.7")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--corpus", default="houses_32")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    model, tokenizer, cfg, info, ckpt = load(a.run, a.ckpt)
    top_k = cfg.top_k if a.top_k is None else a.top_k
    out_dir = a.out or os.path.join(a.run, f"hybrid_oracle_{a.face}")
    os.makedirs(out_dir, exist_ok=True)
    houses = test_houses(a.corpus, a.split_seed, a.n, tokenizer.max_dim)
    real = [fc.project(h, a.face) for h in houses]
    shapes = [tuple(int(x) for x in h.shape) for h in houses]
    pooled = [fc.pool(r, F, "any") for r in real]
    arms: Dict[str, List[fc.Facade]] = {"upsample f4": [fc.upsample(p, F, r.depth.shape) for p, r in zip(pooled, real)]}
    minutes = {}
    for T in [float(x) for x in a.temperatures.split(",")]:
        t0 = time.time()
        arms[f"hybrid from real f4, T={T}"] = draw(model, tokenizer, cfg, real, shapes, pooled, T, top_k, a.seed, a.batch, True)
        minutes[f"given T={T}"] = (time.time() - t0) / 60
    t0 = time.time()
    arms["hybrid from the box, T=1.0"] = draw(model, tokenizer, cfg, real, shapes, pooled, 1.0, top_k, a.seed, a.batch, False)
    minutes["free T=1.0"] = (time.time() - t0) / 60
    agree = agreement_table(real, arms)
    stats = stats_table({"real": real, **arms})
    windows = {k: window_stats(v) for k, v in {"real": real, **arms}.items()}
    # the coarse level drawn freely: does the model's own f=4 look like the real one?
    free_coarse = []
    summary = {"run": a.run, "checkpoint": ckpt, "step": info["step"], "face": a.face, "n": len(real),
               "top_k": top_k, "minutes": minutes, "agreement": agree, "stats": stats, "windows": windows}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    keys = ["mask_iou", "mask_precision", "mask_recall", "depth_exact", "depth_mae", "block_agree"]
    L = [f"# f=4 oracle of the pooled4 hybrid: face {a.face}, n={len(real)}, step {info['step']}", "",
         "## Agreement with the real face (means over faces)", "",
         "| arm | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for k, row in agree.items():
        L.append(f"| {k} | " + " | ".join(f"{row[c]:.3f}" for c in keys) + " |")
    L += ["", "## Face statistics (means; W1 = distance of the per-face distribution to real)", "",
          "| arm | " + " | ".join(STAT_COLS) + " | glass cells | faces with glass | empty faces |",
          "|---|" + "---|" * (len(STAT_COLS) + 3)]
    for k, row in stats.items():
        L.append(f"| {k} | " + " | ".join(f"{row[c]:.3f}" for c in STAT_COLS)
                 + f" | {windows[k]['glass_cells']:.2f} | {windows[k]['faces_with_glass']:.3f} | {row['empty_faces']:.3f} |")
    L += ["", "| arm | " + " | ".join(f"W1 {c}" for c in STAT_COLS) + " |", "|---|" + "---|" * len(STAT_COLS)]
    for k, row in stats.items():
        if k != "real":
            L.append(f"| {k} | " + " | ".join(f"{row.get('w1_' + c, float('nan')):.3f}" for c in STAT_COLS) + " |")
    L += ["", f"minutes: {json.dumps({k: round(v, 1) for k, v in minutes.items()})}"]
    with open(os.path.join(out_dir, "oracle.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    labels = ["real", "upsample f4"] + [k for k in arms if k != "upsample f4"]
    rows = [[real[i]] + [arms[k][i] for k in labels[1:]] for i in range(min(12, len(real)))]
    Image.fromarray(draw_sheet(rows, labels)).save(os.path.join(out_dir, "sheet.png"))
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
