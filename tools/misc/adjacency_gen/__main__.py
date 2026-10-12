"""CLI: build data, train, evaluate, render.

    .venv/bin/python -m tools.misc.adjacency_gen run --kind voxel --minutes 35
    .venv/bin/python -m tools.misc.adjacency_gen run --kind piece --minutes 35
    .venv/bin/python -m tools.misc.adjacency_gen eval --run outputs/run_..._adjacency_gen_voxel
    .venv/bin/python -m tools.misc.adjacency_gen run --kind voxel --smoke     # 2-minute check

Face formulation with coarse-shape conditioning (README, "Face nodes"):

    .venv/bin/python -m tools.misc.adjacency_gen faces-run --f 2 --minutes 30
    .venv/bin/python -m tools.misc.adjacency_gen faces-eval --run outputs/run_..._adjacency_faces_f2

Two-stage full-house generator (README, "Full houses"; cells_cli.py):

    .venv/bin/python -m tools.misc.adjacency_gen cells-stage1 --f 4 --minutes 12
    .venv/bin/python -m tools.misc.adjacency_gen cells-grow --f 4 --minutes 40
    .venv/bin/python -m tools.misc.adjacency_gen cells-merge --f 4 --minutes 40 [--dagger]
    .venv/bin/python -m tools.misc.adjacency_gen cells-eval --f 4 --stage1 <run> --grow <run> --merge <run>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from .data import build_dataset
from .decode import TypePrior, decode, model_scorer
from .eval import evaluate_generation, make_retriever, prior_masked_metrics, save_samples
from .model import ModelConfig
from .train import (ShapeIndex, load_dataset, load_model, masked_metrics, pick_device, prepare,
                    save_dataset, train)

RATES = (0.25, 0.5, 0.75, 1.0)
ARMS = ("model_check", "model_nocheck", "model_check_t1", "model_grow", "prior_check", "random_tree",
        "random_tree_check", "retrieval")


def free_run(model, ds, vocab, sidx, device, n=8):
    model.eval()
    sc = model_scorer(model, sidx.type_valid(), device)
    con, edges = [], []
    for g in ds["val"][:n]:
        r = decode(g.types, vocab, sc, steps=16, check=True)
        from collections import Counter
        con.append(Counter(r["comp"].tolist()).most_common(1)[0][1] / len(g))
        edges.append(int((r["labels"] > 0).sum() // 2) / len(g))
    return {"connectivity": float(np.mean(con)), "edges_per_node": float(np.mean(edges))}


def evaluate(run: Path, ds, model, sidx, device, n_gen: int, steps: int, seed: int = 0,
             arms=None, tag: str = "") -> dict:
    vocab = ds["vocab"]
    test_items = prepare(ds["test"], vocab, sidx)
    train_items_small = [ds["train"][i] for i in
                         np.random.default_rng(0).choice(len(ds["train"]), min(4000, len(ds["train"])),
                                                         replace=False)]
    t0 = time.time()
    prior = TypePrior(train_items_small, vocab, sidx.type_valid())
    print(f"[prior] fitted on {len(train_items_small)} train chunks in {time.time() - t0:.0f}s", flush=True)
    res = {"dataset": ds["stats"], "masked": {}, "masked_prior": {}}
    for r in RATES:
        res["masked"][str(r)] = masked_metrics(model, sidx, test_items, device, r, seed=seed)
        res["masked_prior"][str(r)] = prior_masked_metrics(prior, test_items[:400], vocab, r, seed=seed)
        print(f"[masked] rate {r}: model {res['masked'][str(r)]}  prior {res['masked_prior'][str(r)]}",
              flush=True)
    ctx = {"model_scorer": model_scorer(model, sidx.type_valid(), device),
           "prior_scorer": prior.scorer(), "vt": sidx.type_valid(), "steps": steps,
           "retrieve": make_retriever(ds["train"], vocab.n_types)}
    gen_graphs = ds["test"][:n_gen]
    gens = {}
    samples = None
    for arm in arms or ARMS:
        c = dict(ctx)
        name = arm
        if arm == "model_check_t1":          # sampled decode: temperature 1 + noisy order
            c["temp"] = 1.0
            name, arm = "model_check_t1", "model_check"
        t0 = time.time()
        g = evaluate_generation(gen_graphs, vocab, [arm], c, seed=seed)
        gens.setdefault("real", g["metrics"]["real"])
        gens[name] = g["metrics"][arm]
        gens[name]["sec_per_graph"] = (time.time() - t0) / len(gen_graphs)
        if samples is None:
            samples = {"real": g["samples"]["real"]}
        samples[name] = g["samples"][arm]
    res["generation"] = gens
    res["n_gen"] = len(gen_graphs)
    save_samples(samples, run / f"samples{tag}.png")
    (run / f"results{tag}.json").write_text(json.dumps(res, indent=1))
    print(f"[done] {run}/results{tag}.json, samples{tag}.png", flush=True)
    return res


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1].startswith("cells-"):
        from .cells_cli import main as cells_main     # two-stage full-house generator
        return cells_main(sys.argv[1:])
    ap = argparse.ArgumentParser(prog="adjacency_gen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--kind", choices=["voxel", "piece"], required=True)
    r.add_argument("--minutes", type=float, default=35)
    r.add_argument("--per-build", type=int, default=24)
    r.add_argument("--n-train-builds", type=int, default=None)
    r.add_argument("--data-from", default=None, help="reuse dataset.pkl from an earlier run")
    r.add_argument("--bs", type=int, default=32)
    r.add_argument("--layers", type=int, default=6)
    r.add_argument("--d", type=int, default=192)
    r.add_argument("--rnf", type=int, default=8)
    r.add_argument("--B", type=int, default=4, help="piece offset cube half-width")
    r.add_argument("--n-gen", type=int, default=200)
    r.add_argument("--steps", type=int, default=24)
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--name", default=None)
    r.add_argument("--max-gb", type=float, default=3.5)
    r.add_argument("--prefix-frac", type=float, default=0.0,
                   help="share of training graphs with a growth-state (connected prefix) mask")
    e = sub.add_parser("eval")
    e.add_argument("--run", required=True)
    e.add_argument("--n-gen", type=int, default=200)
    e.add_argument("--steps", type=int, default=24)
    e.add_argument("--max-gb", type=float, default=3.5)
    e.add_argument("--arms", nargs="+", default=None)
    e.add_argument("--tag", default="", help="suffix for results/samples files")
    fr = sub.add_parser("faces-run", help="train + evaluate the face model")
    fr.add_argument("--f", type=int, default=2, help="coarse factor (0 = no conditioning)")
    fr.add_argument("--oracle", action="store_true", help="give each block its true f=2 coarse cell")
    fr.add_argument("--houses", action="store_true",
                    help="full houses_32 builds (64..512 blocks, 32^3 frame) instead of 8^3 windows")
    fr.add_argument("--decode-bs", type=int, default=32)
    fr.add_argument("--merge", action="store_true", help="merge formulation (merge.py) instead of seed growth")
    fr.add_argument("--no-residual", action="store_true",
                    help="coarse tokens do not see the partial build's per-cell counts (v1 runs)")
    fr.add_argument("--minutes", type=float, default=30)
    fr.add_argument("--per-build", type=int, default=24)
    fr.add_argument("--n-train-builds", type=int, default=None)
    fr.add_argument("--data-from", default=None, help="reuse dataset.pkl from an earlier faces run")
    fr.add_argument("--bs", type=int, default=16)
    fr.add_argument("--layers", type=int, default=6)
    fr.add_argument("--d", type=int, default=192)
    fr.add_argument("--n-gen", type=int, default=200)
    fr.add_argument("--arms", nargs="+", default=None)
    fr.add_argument("--t50-run", default="outputs/run_20261007_111458_adjacency_gen_voxel")
    fr.add_argument("--smoke", action="store_true")
    fr.add_argument("--name", default=None)
    fr.add_argument("--max-gb", type=float, default=3.5)
    fe = sub.add_parser("faces-eval", help="re-evaluate a trained face model")
    fe.add_argument("--run", required=True)
    fe.add_argument("--n-gen", type=int, default=200)
    fe.add_argument("--arms", nargs="+", default=None)
    fe.add_argument("--t50-run", default="outputs/run_20261007_111458_adjacency_gen_voxel")
    fe.add_argument("--tag", default="")
    fe.add_argument("--max-gb", type=float, default=3.5)
    a = ap.parse_args()
    if a.cmd.startswith("faces"):
        return faces_main(a)

    from blockgen.utils.runs import new_run_dir
    device = pick_device(a.max_gb)
    print(f"[device] {device}", flush=True)
    if a.cmd == "run":
        if a.smoke:
            a.minutes, a.n_train_builds, a.per_build, a.n_gen = 1.5, 60, 6, 12
        run = new_run_dir(f"adjacency_gen_{a.name or a.kind}{'_smoke' if a.smoke else ''}")
        print(f"[run] {run}", flush=True)
        t0 = time.time()
        if a.data_from:
            ds = load_dataset(Path(a.data_from) / "dataset.pkl")
        else:
            ds = build_dataset(a.kind, n_train_builds=a.n_train_builds, per_build_train=a.per_build,
                               B=1 if a.kind == "voxel" else a.B,
                               n_eval_builds=40 if a.smoke else 200)
        save_dataset(ds, run / "dataset.pkl")
        print(f"[data] {ds['stats']} in {time.time() - t0:.0f}s", flush=True)
        vocab = ds["vocab"]
        sidx = ShapeIndex(vocab)
        train_items = prepare(ds["train"], vocab, sidx)
        val_items = prepare(ds["val"], vocab, sidx)
        cfg = ModelConfig(n_types=vocab.n_types, B=vocab.space.B, d=a.d, layers=a.layers, rnf=a.rnf)
        (run / "config.json").write_text(json.dumps({"args": vars(a), "model": cfg.to_dict()}, indent=1))
        bs = a.bs if a.kind == "voxel" else max(8, a.bs // 2)
        model = train(train_items, val_items, vocab, sidx, run, device, cfg, minutes=a.minutes, bs=bs,
                      eval_every=300 if a.smoke else 1500,
                      free_run_fn=lambda m: free_run(m, ds, vocab, sidx, device),
                      prefix_frac=a.prefix_frac)
        model.eval()
        evaluate(run, ds, model, sidx, device, a.n_gen, a.steps)
    else:
        run = Path(a.run)
        ds = load_dataset(run / "dataset.pkl")
        sidx = ShapeIndex(ds["vocab"])
        model = load_model(run, device)
        evaluate(run, ds, model, sidx, device, a.n_gen, a.steps, arms=a.arms, tag=a.tag)


def faces_main(a):
    from blockgen.utils.runs import new_run_dir
    from .coarse import build_window_dataset
    from .eval_faces import BASE_ARMS, ONESHOT_ARMS, SEQ_ARMS, evaluate_faces
    from .model_faces import FaceConfig
    from .train_faces import (load_face_model, load_window_dataset, prepare_faces, save_window_dataset,
                              train_faces)
    device = pick_device(a.max_gb)
    print(f"[device] {device}", flush=True)
    if a.cmd == "faces-run":
        if a.smoke:
            a.minutes, a.n_train_builds, a.per_build, a.n_gen = 1.0, 60, 6, 12
        name = a.name or (("merge_" if a.merge else "") + ("oracle" if a.oracle else (f"f{a.f}" if a.f else "nocond")))
        run = new_run_dir(f"adjacency_faces_{name}{'_smoke' if a.smoke else ''}")
        print(f"[run] {run}", flush=True)
        t0 = time.time()
        if a.data_from:
            ds = load_window_dataset(Path(a.data_from) / "dataset.pkl")
        elif a.houses:
            from .coarse import build_house_dataset
            ds = build_house_dataset()
        else:
            ds = build_window_dataset(n_train_builds=a.n_train_builds, per_build_train=a.per_build,
                                      n_eval_builds=40 if a.smoke else 399)
        save_window_dataset(ds, run / "dataset.pkl")
        print(f"[data] {ds['stats']} in {time.time() - t0:.0f}s", flush=True)
        f = 2 if a.oracle and not a.f else a.f
        cfg = FaceConfig(n_types=ds["vocab"].n_types, f=f, oracle=a.oracle, d=a.d, layers=a.layers,
                         size=int(ds["stats"].get("window", 8)),
                         residual=bool(f) and not a.no_residual and not a.merge, merge=a.merge)
        (run / "config.json").write_text(json.dumps({"args": vars(a), "model": cfg.to_dict()}, indent=1))
        tr = prepare_faces(ds["train"], ds["vocab"], cfg.f, size=cfg.size)
        va = prepare_faces(ds["val"], ds["vocab"], cfg.f, size=cfg.size)
        model = train_faces(tr, va, cfg, run, device, minutes=a.minutes, bs=a.bs,
                            eval_every=200 if a.smoke else 1500)
    else:
        run = Path(a.run)
        ds = load_window_dataset(run / "dataset.pkl")
        model = load_face_model(run, device)
        cfg = model.cfg
        tr = prepare_faces(ds["train"], ds["vocab"], cfg.f, size=cfg.size)
    f_eval = cfg.f or 2
    te_g = ds["test"][:a.n_gen]
    te = prepare_faces(te_g, ds["vocab"], cfg.f, size=cfg.size)
    from .eval_faces import MERGE_ARMS
    arms = a.arms or ((list(MERGE_ARMS) if cfg.merge else list(SEQ_ARMS)) + list(ONESHOT_ARMS) + list(BASE_ARMS))
    t50 = Path(a.t50_run) if a.t50_run else None
    evaluate_faces(run, model, te_g, te, ds["train"], tr, ds["vocab"], device, arms, f_eval,
                   t50_run=t50, tag=getattr(a, "tag", ""), decode_bs=getattr(a, "decode_bs", 32))


if __name__ == "__main__":
    main()
