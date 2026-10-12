"""CLI: ``python -m tools.misc.attention_graphs {llm,ar2,summary,figures} [options]``.

Examples::

    export HF_HOME=~/hf_cache HF_HUB_OFFLINE=1
    .venv/bin/python -m tools.misc.attention_graphs ar2 --out outputs/run_X_attn_adjacency
    .venv/bin/python -m tools.misc.attention_graphs llm --models Qwen/Qwen2.5-1.5B \\
        --out outputs/run_X_attn_adjacency

Both write ``<out>/<model>/results.json``, per-head CSVs and figures, then rebuild
``<out>/summary.md`` from every ``results.json`` under ``<out>``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from . import viz  # noqa: E402
from .experiment import (Arm, probe_test_baselines, run_probes, summarize,  # noqa: E402
                         write_head_csv)
from .extract import extract, head_scores, pick_device  # noqa: E402

LLM_ARMS = (("absolute", "raster"), ("absolute", "shuffle"), ("relative", "bfs"),
            ("relative", "random"))


# ---------------------------------------------------------------- figures

def arm_figures(name: str, summary: dict, out: Path, short: str, scene, heads: np.ndarray,
                token_att=None, spans=None, metric: str = "auc") -> None:
    """For item 0 of an arm: the head selected by ``metric`` vs the true graph, every head
    of its layer, and its token-level attention."""
    l, h = summary["heads"][metric]["head"]
    tag = name.replace("/", "_")
    # Row i spreads its mass over i earlier lines; scale by i so 1.0 = uniform and late
    # rows stay visible (AUROC is rank-based per build and is unaffected by display).
    lift = np.arange(heads.shape[-1])[:, None]
    viz.versus_truth([(f"L{l}H{h}", heads[l, h] * lift, np.tril(scene.adjacency, -1))],
                     out / f"{tag}_best_head_vs_truth.png",
                     f"{short} {name}: head L{l}H{h} (held-out {metric} "
                     f"{summary['heads'][metric]['heldout']:.3f}) vs true adjacency\n"
                     f"row i = a block, columns = earlier blocks j < i; "
                     f"score = attention x i (1 = uniform)")
    viz.head_grid(heads[l] * lift, out / f"{tag}_head_grid_L{l}.png",
                  f"{short} {name}: layer {l}, every head (node attention x i)")
    if token_att is not None:
        viz.token_heatmap(token_att[l, h], spans, out / f"{tag}_tokens_L{l}H{h}.png",
                          f"{short} {name} L{l}H{h}: token attention, first 20 lines")


def llm_example_figures(model, tok, out: Path, short: str, sums: Dict[str, dict],
                        seed: int = 0, size: int = 5) -> None:
    """Re-run item 0 of every LLM arm and draw its figures (cheap: one forward each)."""
    from .scenes import house_chunks, lego_scenes
    from .serialize import serialize

    first = {"house": house_chunks(1, size=size, seed=seed)[0],
             "lego": lego_scenes(1, n_bricks=24, seed=seed)[0]}
    for name, s in sums.items():
        fmt, order = ("absolute", name.split("/")[1]) if name.startswith("lego/") else name.split("/")
        scene = first["lego" if name.startswith("lego/") else "house"]
        ser = serialize(scene, tok, order=order, seed=0, fmt=fmt)
        ex = extract(model, ser.input_ids, ser.spans, keep_tokens=True)
        heads, _ = head_scores(ex)
        arm_figures(name, s, out, short, ser.scene, heads, ex.token_att, ser.spans)
        (out / f"{name.replace('/', '_')}_example.txt").write_text(ser.text)


def ar2_example_figures(model, builds, out: Path, short: str, sums: Dict[str, dict]) -> None:
    """Item 0 of the query-Z and mean arms, head chosen by hard-pair stratified AUROC."""
    from .ar2 import extract_ar2, tokens_for_order

    scene, toks = builds[0]
    for order in ("raster", "shuffle"):
        t2, perm = tokens_for_order(toks, order, seed=0)
        exs = extract_ar2(model, t2, keep_tokens=True)
        for sl in ("Z", "mean"):
            name = f"{order}/q{sl}"
            heads, _ = head_scores(exs[sl])
            arm_figures(name, sums[name], out, short, scene.subset(perm), heads,
                        exs[sl].token_att, [(1 + 4 * i, 5 + 4 * i) for i in range(len(scene))],
                        metric="hard_strat")


HEATMAPS = (("auc", "AUROC"), ("strat", "distance-stratified AUROC"),
            ("hard", "AUROC on pairs sharing 2 coordinates"),
            ("hard_strat", "hard-pair AUROC, distance-stratified"),
            ("unstated_auc", "AUROC on contacts the text never states"))


def model_figures(out: Path, short: str) -> None:
    """Layer x head heatmaps and probe curves, from ``head_scores.csv`` + ``results.json``.

    Also reachable as ``python -m tools.misc.attention_graphs figures <run_dir>``.
    """
    sums = {s["name"]: s for s in json.loads((out / "results.json").read_text())["arms"]}
    with open(out / "head_scores.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["head"] != "rollout"]
    for metric, label in HEATMAPS:
        grids, notes = {}, {}
        for name, s in sums.items():
            vals = [(int(r["layer"]), int(r["head"]), r.get(f"{metric}_mean", ""))
                    for r in rows if r["arm"] == name]
            if not vals or all(v in ("", "nan") for *_, v in vals):
                continue
            g = np.full((max(v[0] for v in vals) + 1, max(v[1] for v in vals) + 1), np.nan)
            for l, h, v in vals:
                g[l, h] = float(v) if v else np.nan
            grids[name] = g
            ref = "tree_hops" if metric == "unstated_auc" else "recency"
            notes[name] = f"{ref} {s['baselines'][ref].get(metric, float('nan')):.3f}"
        if grids:
            viz.auroc_heatmaps(grids, notes, out / f"heatmap_{metric}.png",
                               f"{short}: per-head {label} (mean over items)", label=label)
    curves = {name: {int(l): v["strat"] for l, v in s["probe"]["distance"].items()}
              for name, s in sums.items() if "probe" in s}
    if curves:
        viz.probe_curves(curves, {}, out / "probe_by_layer.png",
                         f"{short}: distance probe on frozen node states, held-out builds",
                         ylabel="held-out distance-stratified AUROC")


def _finish(arms: List[Arm], sums: List[dict], out: Path, short: str, meta: dict) -> None:
    write_head_csv(arms, out / "head_scores.csv")
    (out / "results.json").write_text(json.dumps({"meta": meta, "arms": sums}, indent=1))
    model_figures(out, short)


# ---------------------------------------------------------------- LLM

def run_llm(a) -> None:
    from .extract import load_model
    from .scenes import house_chunks, lego_scenes
    from .serialize import serialize

    scenes = house_chunks(a.n, size=a.size, seed=a.seed)
    conditions = [(scenes, fmt, order, f"{fmt}/{order}") for fmt, order in LLM_ARMS]
    if a.lego:
        bricks = lego_scenes(a.n, n_bricks=24, seed=a.seed)
        conditions += [(bricks, "absolute", o, f"lego/{o}") for o in ("raster", "shuffle")]
    for name in a.models:
        device = a.device or pick_device()
        short = name.split("/")[-1]
        print(f"== {name} on {device}, {len(scenes)} chunks", flush=True)
        model, tok = load_model(name, device=device)
        cfg = getattr(model.config, "text_config", model.config)
        n_layers = cfg.num_hidden_layers
        layers = sorted(set(range(0, n_layers + 1, a.probe_every)) | {n_layers}) if a.probe_every else []
        out = Path(a.out) / short
        out.mkdir(parents=True, exist_ok=True)
        arms, sums = [], []
        for items, fmt, order, arm_name in conditions:
            arm = Arm(arm_name)
            for k, scene in enumerate(items):
                ser = serialize(scene, tok, order=order, seed=k, fmt=fmt)
                ex = extract(model, ser.input_ids, ser.spans, query=a.query, key=a.key,
                             hidden_layers=layers)
                heads, roll = head_scores(ex)
                arm.add(ser.scene, heads, roll, ser.parents, ex.hidden)
            s = summarize(arm)
            if layers:
                s["probe"] = run_probes(arm, steps=a.probe_steps)
                s["probe_test_baselines"] = probe_test_baselines(arm)
            arm.hidden = {}
            arms.append(arm)
            sums.append(s)
            print(f"  {arm.name}: best head AUROC {s['heads']['auc']['heldout']:.3f}, "
                  f"strat {s['heads']['strat']['heldout']:.3f}", flush=True)
        _finish(arms, sums, out, short, {"model": name, "device": device, "n": len(scenes),
                                         "chunk": a.size, "query": a.query, "key": a.key,
                                         "seed": a.seed})
        llm_example_figures(model, tok, out, short, {x["name"]: x for x in sums}, a.seed, a.size)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        write_summary(Path(a.out))


# ---------------------------------------------------------------- voxel AR

def run_ar2(a) -> None:
    from .ar2 import DEFAULT_RUN, SLOTS, extract_ar2, load_ar2, tokens_for_order, val_builds

    run_dir = Path(a.ckpt) if a.ckpt else DEFAULT_RUN
    device = a.device or pick_device(min_free_gb=2.0)
    model, vocab = load_ar2(run_dir, device=device)
    builds = val_builds(vocab, a.n, seed=a.seed)
    n_layers = len(model.blocks)
    short = f"ar2_{run_dir.name}"
    out = Path(a.out) / short
    out.mkdir(parents=True, exist_ok=True)
    print(f"== {run_dir} on {device}: {len(builds)} val builds, median "
          f"{int(np.median([len(s) for s, _ in builds]))} blocks", flush=True)
    arms, sums = [], []
    for order in ("raster", "shuffle"):
        slot_arms: Dict[str, Arm] = {sl: Arm(f"{order}/q{sl}") for sl in SLOTS + ("mean",)}
        for k, (scene, toks) in enumerate(builds):
            t2, perm = tokens_for_order(toks, order, seed=k)
            sc = scene.subset(perm)
            exs = extract_ar2(model, t2, hidden_layers=range(n_layers + 1))
            for sl, ex in exs.items():
                heads, roll = head_scores(ex)
                slot_arms[sl].add(sc, heads, roll, hidden=ex.hidden if sl == "mean" else None)
        for sl, arm in slot_arms.items():
            s = summarize(arm)
            if sl == "mean":
                s["probe"] = run_probes(arm, steps=a.probe_steps)
                s["probe_test_baselines"] = probe_test_baselines(arm)
            arm.hidden = {}
            arms.append(arm)
            sums.append(s)
            print(f"  {arm.name}: best head AUROC {s['heads']['auc']['heldout']:.3f}, "
                  f"strat {s['heads']['strat']['heldout']:.3f}, "
                  f"hard {s['heads']['hard']['heldout']:.3f}", flush=True)
    _finish(arms, sums, out, short, {"model": str(run_dir), "device": device, "n": len(builds),
                                     "seed": a.seed})
    ar2_example_figures(model, builds, out, short, {x["name"]: x for x in sums})
    write_summary(Path(a.out))


def redraw_examples(model_dir: Path) -> None:
    res = json.loads((model_dir / "results.json").read_text())
    meta, sums = res["meta"], {x["name"]: x for x in res["arms"]}
    for old in model_dir.glob("*_best_head_vs_truth.png"):
        old.unlink()
    for pattern in ("*_head_grid_L*.png", "*_tokens_L*H*.png"):
        for old in model_dir.glob(pattern):
            old.unlink()
    device = pick_device(min_free_gb=4.0)
    if model_dir.name.startswith("ar2_"):
        from .ar2 import load_ar2, val_builds
        model, vocab = load_ar2(Path(meta["model"]), device=device)
        ar2_example_figures(model, val_builds(vocab, 1, seed=meta.get("seed", 0)), model_dir,
                            model_dir.name, sums)
    else:
        from .extract import load_model
        model, tok = load_model(meta["model"], device=device)
        llm_example_figures(model, tok, model_dir, model_dir.name, sums,
                            meta.get("seed", 0), meta.get("chunk", 5))
    del model
    gc.collect()


# ---------------------------------------------------------------- summary.md

def _f(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def write_summary(out: Path) -> None:
    lines = ["# Attention as adjacency: run summary", "",
             "All AUROCs are over causal candidate pairs `j < i` (T28 protocol). `strat` = "
             "distance-stratified AUROC (recency removed). `hard` = AUROC on pairs sharing "
             "two of x/y/z (coordinate-token matching removed). Heads are selected on even "
             "items and reported on odd items; baselines are reported on the same odd items.", ""]
    for res_path in sorted(out.glob("*/results.json")):
        res = json.loads(res_path.read_text())
        meta = res["meta"]
        lines += [f"## {res_path.parent.name}", "", f"`{json.dumps(meta)}`", "",
                  "| arm | nodes | edge rate | recency auc / strat / hard / hard_strat | "
                  "coord_match auc / strat | coord_L1 auc | best head auc | best head strat | "
                  "best head hard | best head hard_strat | rollout auc |",
                  "|---|--:|--:|---|---|---|---|---|---|---|--:|"]
        for s in res["arms"]:
            b, hd = s["baselines"], s["heads"]
            hh = lambda m: f"{_f(hd[m]['heldout'])} (L{hd[m]['head'][0]}H{hd[m]['head'][1]})"  # noqa: E731
            lines.append(
                f"| {s['name']} | {s['median_nodes']} | {s['edge_rate']:.3f} | "
                f"{_f(b['recency']['auc'])} / {_f(b['recency']['strat'])} / "
                f"{_f(b['recency']['hard'])} / {_f(b['recency']['hard_strat'])} | "
                f"{_f(b['coord_match']['auc'])} / {_f(b['coord_match']['strat'])} | "
                f"{_f(b['coord_L1']['auc'])} | "
                f"{hh('auc')} | {hh('strat')} | {hh('hard')} | {hh('hard_strat')} | "
                f"{_f(s['rollout']['auc'])} |")
        rel = [s for s in res["arms"] if "parent_top1" in s["heads"]]
        if rel:
            lines += ["", "Coordinate-free (relative) format: pointer to the stated parent, and "
                          "contacts the text never states:", "",
                      "| arm | parent top-1: best head / recency | unstated auc / strat: "
                      "best head | recency | tree_hops | same_name |",
                      "|---|---|---|---|---|---|"]
            for s in rel:
                hd, b = s["heads"], s["baselines"]
                u = lambda d: f"{_f(d['unstated_auc'])} / {_f(d['unstated_strat'])}"  # noqa: E731
                lines.append(
                    f"| {s['name']} | {_f(hd['parent_top1']['heldout'])} / "
                    f"{_f(b['recency']['parent_top1'])} | {_f(hd['unstated_auc']['heldout'])} / "
                    f"{_f(hd['unstated_strat']['heldout'])} | {u(b['recency'])} | "
                    f"{u(b['tree_hops'])} | {u(b['same_name'])} |")
            lines += ["", "In the relative format no coordinates are printed, so `coord_match` "
                          "and `coord_L1` above are oracles there, not text baselines."]
        probed = [s for s in res["arms"] if "probe" in s]
        if probed:
            lines += ["", "Trained edge probes on frozen node states (held-out builds; "
                          "best layer by strat):", "",
                      "| arm | probe | layer | auc | strat | hard | hard_strat | "
                      "recency auc / hard (same items) | coord_match auc (same items) |",
                      "|---|---|--:|---|---|---|---|---|---|"]
            for s in probed:
                tb = s["probe_test_baselines"]
                for kind, by_layer in s["probe"].items():
                    l, v = max(by_layer.items(), key=lambda kv: kv[1]["strat"])
                    lines.append(f"| {s['name']} | {kind} | {l} | {_f(v['auc'])} | "
                                 f"{_f(v['strat'])} | {_f(v.get('hard'))} | "
                                 f"{_f(v.get('hard_strat'))} | {_f(tb['recency']['auc'])} / "
                                 f"{_f(tb['recency'].get('hard'))} | "
                                 f"{_f(tb['coord_match']['auc'])} |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description="attention-as-adjacency experiments")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("llm", "ar2"):
        q = sub.add_parser(name)
        q.add_argument("--out", default=None,
                       help="run dir (default: outputs/run_<stamp>_attn_adjacency)")
        q.add_argument("--n", type=int, default=48)
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--probe-steps", type=int, default=400)
        q.add_argument("--device", default=None)
    llm = sub.choices["llm"]
    llm.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-1.5B"])
    llm.add_argument("--size", type=int, default=5, help="house chunk edge length")
    llm.add_argument("--query", default="mean", choices=["mean", "last"])
    llm.add_argument("--key", default="sum", choices=["sum", "mean"])
    llm.add_argument("--probe-every", type=int, default=4, help="0 disables the probe")
    llm.add_argument("--lego", action="store_true", help="also run LEGO stacks (stud graph)")
    sub.choices["ar2"].add_argument("--ckpt", default=None,
                                    help="flat-token run dir (model.pt + block_vocab.json)")
    sub.add_parser("summary").add_argument("out")
    fig = sub.add_parser("figures", help="redraw figures from a finished run dir")
    fig.add_argument("out")
    fig.add_argument("--examples", action="store_true",
                     help="also redraw the per-arm example figures (loads each model)")
    a = p.parse_args()
    if a.cmd in ("summary", "figures"):
        if a.cmd == "figures":
            for res in sorted(Path(a.out).glob("*/results.json")):
                model_figures(res.parent, res.parent.name)
                if a.examples:
                    redraw_examples(res.parent)
        write_summary(Path(a.out))
        return
    if a.out is None:
        from blockgen.utils.runs import new_run_dir
        a.out = str(new_run_dir("attn_adjacency"))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    {"llm": run_llm, "ar2": run_ar2}[a.cmd](a)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
