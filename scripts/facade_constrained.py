"""Cross-face constrained decoding for the joint facade model (research.md H.7 lever, T68 next step).

``sample`` draws ``n`` builds from a trained joint facade run with the same real boxes
``facade_builds`` would draw, under one of two modes: ``none`` (grammar enforced, faces
otherwise free; the control) or ``constrain`` (every face masked by what the earlier
faces committed, see ``blockgen.corpus.facade_constraints``). It writes the structure
cache the bench scores (``samples_step<n>_48.npz``), the faces (``faces.npz``), the
sheets, and ``summary.json`` with the reprojection disagreement, the share of cells
earlier faces determined, face statistics and 3D components.

``oracle`` completes real test houses: the faces named by ``--given`` are forced verbatim
(the real front face), the rest are drawn under the mode, and every drawn face is
scored against the real one (``face_agreement``), the lifted shell against the real
shell. It isolates what the constraint buys when the context is right.

    PYTHONPATH=. python scripts/facade_constrained.py sample --run <joint run> --mode constrain --n 399
    PYTHONPATH=. python scripts/facade_constrained.py oracle --run <joint run> --given -z --mode constrain
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Sequence

import numpy as np

from blockgen.corpus import facade as fc
from blockgen.corpus import facade_constraints as fcc
from blockgen.eval.bench import splits
from blockgen.eval.facade_stats import face_agreement, face_stats
from blockgen.train.sample_constrained import decode_builds, face_size, generate_faces_constrained
from blockgen.utils.runs import new_run_dir

MODES = ("none", "constrain")


def load(run: str, ckpt: str | None):
    from blockgen.train.__main__ import selected_ckpt
    from blockgen.train.trainer import load_run_checkpoint
    ckpt = ckpt or selected_ckpt(run)
    model, tokenizer, cfg, info = load_run_checkpoint(run, ckpt)
    assert tokenizer.kind == "facades" and tokenizer.layout == "joint", (tokenizer.kind, tokenizer.layout)
    return model, tokenizer, cfg, info, ckpt


def face_order(cfg, tokenizer) -> List[str]:
    from blockgen.train.sample import _train_dims
    dims, faces = _train_dims(cfg, tokenizer)
    return dims, list(faces or fc.ALL_FACES)


def save_faces(per_build: List[Dict[str, fc.Facade]], path: str) -> None:
    out = {}
    for i, faces in enumerate(per_build):
        for name, f in faces.items():
            key = f"{i}_{name}"
            out[key + "_depth"] = f.depth.astype(np.int16)
            out[key + "_ids"] = f.ids.astype(np.int16)
            out[key + "_data"] = f.data.astype(np.int16)
    np.savez_compressed(path, **out)


def summarise(builds, per_build, shapes) -> dict:
    rep = [fcc.reprojection(f, s) for f, s in zip(per_build, shapes) if f]
    cons = [fcc.consistent(f, s, order=list(f)) for f, s in zip(per_build, shapes) if f]
    st = [face_stats(b) for b in builds if b.occupied_mask.any()]
    keys = ["face_comps", "face_holes", "face_jumps", "face_lone", "comps3d", "lcc", "dust_blocks"]
    out = {k: float(np.nanmean([s[k] for s in st])) for k in keys}
    out.update({f"reproj_{k}": float(np.nanmean([r[k] for r in rep])) for k in ("cell_disagree", "hidden", "filled_in")})
    out["consistent_frac"] = float(np.mean(cons)) if cons else float("nan")
    out["n_empty"] = int(sum(not b.occupied_mask.any() for b in builds))
    out["blocks_median"] = float(np.median([int(b.occupied_mask.sum()) for b in builds]))
    return out


def write_sheets(builds, per_build, out_dir: str) -> None:
    from blockgen.train.sample import _facade_sheet
    _facade_sheet(per_build, out_dir)
    try:
        from blockgen.renderer.grid import save_grid
        save_grid(builds[:64], os.path.join(out_dir, "samples.png"), cols=8)
    except Exception as e:                                   # the renderer needs a display or EGL; never fatal
        print(f"[constrained] render failed: {e}", flush=True)


def cmd_sample(a) -> None:
    model, tokenizer, cfg, info, ckpt = load(a.run, a.ckpt)
    dims, faces = face_order(cfg, tokenizer)
    seed = cfg.seed if a.seed is None else a.seed
    T = cfg.temperature if a.temperature is None else a.temperature
    top_k = cfg.top_k if a.top_k is None else a.top_k
    rng = np.random.default_rng(seed)
    boxes = dims[rng.integers(len(dims), size=a.n)]           # the draw facade_builds makes under this seed
    shapes = [tuple(int(x) for x in b) for b in boxes]
    plans = [[(f, *face_size(f, s)) for f in faces] for s in shapes]
    out_dir = a.out or os.path.join(a.run, f"constrained_{a.mode}_t{T}_n{a.n}")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    seqs, info_gen = generate_faces_constrained(model, tokenizer, cfg.sample_tags, plans, shapes, T, top_k, seed,
                                                batch=a.batch, constrain=(a.mode == "constrain"))
    minutes = (time.time() - t0) / 60
    builds, per_build = decode_builds(seqs, shapes, tokenizer, corpus=f"{cfg.name}_{a.mode}")
    from blockgen.curation.houses import save_house_cache
    report = {"run": a.run, "step": info["step"], "n": a.n, "seed": seed, "temperature": T, "top_k": top_k,
              "tags": cfg.sample_tags, "model": cfg.name, "arch": cfg.arch, "kind": tokenizer.kind,
              "checkpoint": {"file": ckpt, "step": info["step"]}, "decode": f"facade_constrained:{a.mode}",
              "trained_on_signals": list(cfg.trained_on_signals or []), "train_split": cfg.split_name()}
    keep = [b.crop_to_non_air() for b in builds if b.occupied_mask.any()]
    path = save_house_cache(keep, tokenizer.max_dim, cache_dir=out_dir, name=f"samples_step{a.n}", report=report)
    save_faces(per_build, os.path.join(out_dir, "faces.npz"))
    summary = summarise(builds, per_build, shapes)
    summary.update({"mode": a.mode, "n": a.n, "seed": seed, "temperature": T, "top_k": top_k, "minutes": minutes,
                    "forced_cells": info_gen["forced_cells"], "cells": info_gen["cells"],
                    "forced_frac": info_gen["forced_cells"] / max(info_gen["cells"], 1), "npz": path, "faces": faces})
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    write_sheets(builds, per_build, out_dir)
    print(json.dumps(summary, indent=1), flush=True)


def test_houses(corpus: str, seed: int, n: int | None, max_dim: int):
    split = splits.load_split(corpus, seed)
    structs = [s.crop_to_non_air() for s in splits.split_structures(split, "test")]
    structs = [s for s in structs if max(s.shape) <= max_dim and s.occupied_mask.any()]
    return structs[:n] if n else structs


def cmd_oracle(a) -> None:
    model, tokenizer, cfg, info, ckpt = load(a.run, a.ckpt)
    _, faces = face_order(cfg, tokenizer)
    T = cfg.temperature if a.temperature is None else a.temperature
    top_k = cfg.top_k if a.top_k is None else a.top_k
    given_names = [g.strip() for g in a.given.split(",") if g.strip()]
    houses = test_houses(a.corpus, a.split_seed, a.n, tokenizer.max_dim)
    shapes = [s.shape for s in houses]
    real = [fc.project_all(s, fc.ALL_FACES) for s in houses]
    plans = [[(f, *face_size(f, s)) for f in faces] for s in shapes]
    given = [{g: r[g] for g in given_names} for r in real]
    out_dir = a.out or os.path.join(a.run, f"oracle_{'_'.join(g.replace('-', 'm').replace('+', 'p') for g in given_names)}_{a.mode}_t{T}_n{len(houses)}")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    seqs, info_gen = generate_faces_constrained(model, tokenizer, cfg.sample_tags, plans, shapes, T, top_k, a.seed,
                                                batch=a.batch, constrain=(a.mode == "constrain"), given=given)
    minutes = (time.time() - t0) / 60
    builds, per_build = decode_builds(seqs, shapes, tokenizer, corpus=f"{cfg.name}_oracle_{a.mode}")
    drawn = [f for f in faces if f not in given_names]
    per_face: Dict[str, Dict[str, List[float]]] = {f: {} for f in drawn}
    shell_iou, shell_block = [], []
    for r, g, b, s, h in zip(real, per_build, builds, shapes, houses):
        for f in drawn:
            if f in g and g[f].depth.shape == r[f].depth.shape:
                for k, v in face_agreement(r[f], g[f]).items():
                    per_face[f].setdefault(k, []).append(v)
        real_shell = fc.lift(r, s)
        a_, b_ = real_shell.occupied_mask, b.occupied_mask
        both = a_ & b_
        shell_iou.append(both.sum() / max((a_ | b_).sum(), 1))
        shell_block.append(float(((real_shell.block_ids == b.block_ids) & (real_shell.block_data == b.block_data))[both].mean())
                           if both.any() else 0.0)
    table = {f: {k: float(np.nanmean(v)) for k, v in d.items()} for f, d in per_face.items()}
    allk = sorted({k for d in table.values() for k in d})
    table["mean"] = {k: float(np.nanmean([d[k] for d in table.values() if k in d])) for k in allk}
    summary = summarise(builds, per_build, shapes)
    summary.update({"mode": a.mode, "given": given_names, "drawn": drawn, "n": len(houses), "seed": a.seed,
                    "temperature": T, "top_k": top_k, "minutes": minutes, "forced_cells": info_gen["forced_cells"],
                    "cells": info_gen["cells"], "forced_frac": info_gen["forced_cells"] / max(info_gen["cells"], 1),
                    "shell_iou": float(np.mean(shell_iou)), "shell_block_agree": float(np.mean(shell_block)),
                    "per_face": table})
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    save_faces(per_build, os.path.join(out_dir, "faces.npz"))
    write_sheets(builds, per_build, out_dir)
    lines = [f"# Oracle completion: given {given_names}, mode {a.mode}, T={T}, n={len(houses)}", "",
             f"shell IoU {summary['shell_iou']:.3f}, shell block agree {summary['shell_block_agree']:.3f}, "
             f"reprojection disagree {summary['reproj_cell_disagree']:.4f}, forced cells {summary['forced_frac']:.3f}, "
             f"comps3d {summary['comps3d']:.1f}", "", "| face | " + " | ".join(allk) + " |", "|---|" + "---|" * len(allk)]
    for f, d in table.items():
        lines.append(f"| {f} | " + " | ".join(f"{d.get(k, float('nan')):.3f}" for k in allk) + " |")
    with open(os.path.join(out_dir, "oracle.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def cmd_bon(a) -> None:
    """Best-of-k under the constrained decoder: ``n * k`` boxes drawn as the facade sampler draws
    them (the first ``n`` are the run's own sample boxes), each group of ``k`` consecutive draws
    keeps the reward's favourite, as ``blockgen.posttrain.bon`` does for the free sampler."""
    from blockgen.curation.houses import save_house_cache
    from blockgen.posttrain import bon as bon_mod
    from blockgen.posttrain import rewards as rw
    model, tokenizer, cfg, info, ckpt = load(a.run, a.ckpt)
    dims, faces = face_order(cfg, tokenizer)
    seed = cfg.seed if a.seed is None else a.seed
    T = cfg.temperature if a.temperature is None else a.temperature
    top_k = cfg.top_k if a.top_k is None else a.top_k
    rng = np.random.default_rng(seed)
    boxes = dims[rng.integers(len(dims), size=a.n * a.k)]
    shapes = [tuple(int(x) for x in b) for b in boxes]
    plans = [[(f, *face_size(f, s)) for f in faces] for s in shapes]
    out_dir = a.out or os.path.join(a.run, f"constrained_bon_{a.reward}_k{a.k}_{a.mode}_t{T}_n{a.n}")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    seqs, info_gen = generate_faces_constrained(model, tokenizer, cfg.sample_tags, plans, shapes, T, top_k, seed,
                                                batch=a.batch, constrain=(a.mode == "constrain"))
    minutes = (time.time() - t0) / 60
    builds, per_build = decode_builds(seqs, shapes, tokenizer, corpus=f"{cfg.name}_{a.mode}_bon")
    reward = rw.get(a.reward)
    nonempty = np.array([b.occupied_mask.any() for b in builds])
    scores = np.full(len(builds), -np.inf)
    if nonempty.any():
        scores[nonempty] = reward.score([b for b, ok in zip(builds, nonempty) if ok])
    kept_idx, rows = bon_mod.select(scores, a.k)
    kept = [builds[i].crop_to_non_air() for i in kept_idx if nonempty[i]]
    own = rw.signals_of(reward)
    report = {"run": a.run, "step": info["step"], "n": a.n, "k": a.k, "seed": seed, "temperature": T, "top_k": top_k,
              "tags": cfg.sample_tags, "model": cfg.name, "arch": cfg.arch, "kind": tokenizer.kind,
              "checkpoint": {"file": ckpt, "step": info["step"]}, "decode": f"facade_constrained:{a.mode}",
              "method": "bon", "reward": reward.name, "reward_weights": {reward.name: 1.0},
              "trained_on_signals": [x for x in (cfg.trained_on_signals or []) if x not in own] + own,
              "train_split": cfg.split_name(), "n_requested": a.n * a.k, "n_drawn": int(nonempty.sum()),
              "n_kept": len(kept), "score_all_mean": float(np.mean(scores[nonempty])) if nonempty.any() else None,
              "score_kept_mean": float(np.mean([scores[i] for i in kept_idx if nonempty[i]])) if kept else None}
    path = save_house_cache(kept, tokenizer.max_dim, cache_dir=out_dir, name="bon", report=report)
    try:
        import pandas as pd
        pd.DataFrame(rows, columns=["group", "index", "score", "kept"]).to_parquet(os.path.join(out_dir, "scores.parquet"), index=False)
    except Exception as e:                                   # the parquet is a convenience
        print(f"[constrained] scores.parquet skipped: {e}", flush=True)
    kept_faces = [per_build[i] for i in kept_idx]
    kept_shapes = [shapes[i] for i in kept_idx]
    summary = summarise([builds[i] for i in kept_idx], kept_faces, kept_shapes)
    summary.update({"mode": a.mode, "n": a.n, "k": a.k, "reward": reward.name, "seed": seed, "temperature": T,
                    "top_k": top_k, "minutes": minutes, "forced_cells": info_gen["forced_cells"], "cells": info_gen["cells"],
                    "forced_frac": info_gen["forced_cells"] / max(info_gen["cells"], 1), "npz": path,
                    "score_all_mean": report["score_all_mean"], "score_kept_mean": report["score_kept_mean"]})
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    save_faces(kept_faces, os.path.join(out_dir, "faces.npz"))
    write_sheets([builds[i] for i in kept_idx], kept_faces, out_dir)
    print(json.dumps(summary, indent=1), flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sample", "oracle", "bon"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--ckpt", default=None)
        p.add_argument("--mode", choices=MODES, default="constrain")
        p.add_argument("--n", type=int, default=399)
        p.add_argument("--temperature", type=float, default=None)
        p.add_argument("--top-k", type=int, default=None)
        p.add_argument("--batch", type=int, default=16)
        p.add_argument("--out", default=None)
    for name in ("sample", "bon"):
        sub.choices[name].add_argument("--seed", type=int, default=None, help="default: the run's seed (same boxes as its samples)")
    b = sub.choices["bon"]
    b.add_argument("--k", type=int, default=4)
    b.add_argument("--reward", default="face_solidity")
    o = sub.choices["oracle"]
    o.add_argument("--given", default="-z", help="comma list of real faces to force, e.g. -z or -z,+x")
    o.add_argument("--corpus", default="houses_32")
    o.add_argument("--split-seed", type=int, default=0)
    o.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    {"sample": cmd_sample, "oracle": cmd_oracle, "bon": cmd_bon}[a.cmd](a)


if __name__ == "__main__":
    main()
