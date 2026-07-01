"""Curated-subset experiment runner: figures + retraining + novelty, all to disk.

Given the metadata-labeled cache, this:
  1. carves the dataset into named, coherent **subsets** (houses, pixel art,
     redstone, material-variant groups, ...),
  2. renders **contact-sheet figures** that show off each subset,
  3. **retrains** the three generative tracks on the subsets they can actually
     handle (the variable-size constraint means token tracks need compact builds,
     while diffusion's fixed 24^3 grid eats dense houses), and
  4. runs the **novelty** eval per trained model and renders sample-vs-neighbor
     comparison grids.

Everything for a run lands under ``outputs/run_<stamp>/`` with the exact configs,
loss curves, samples, metrics, and a ``run_notes.md`` so runs are reproducible and
comparable. Invoke::

    .venv/bin/python -m blockgen.experiments --stamp $(date +%Y%m%d_%H%M%S)

Honest framing for the paper: training a *type* (e.g. houses) is the "learn one
object class, then expand" milestone. Which track can learn which subset is itself
a finding about the size/footprint challenge, so we record it rather than hide it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from blockgen.curation.curate import Curator  # noqa: E402
from blockgen.eval.novelty import (  # noqa: E402
    evaluate_novelty,
    render_comparison_grid,
    summary_row,
)
from blockgen.renderer.render import render_schem  # noqa: E402
from blockgen.utils.data import Structure  # noqa: E402
from blockgen.utils.serialize import build_block_vocab  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "outputs"

# Categories that hold actual buildings/objects (vs. flat art / wiring maps).
STRUCT_CATS = ["Land Structure Map", "Air Structure Map", "Other Map", "Complex Map"]


# --------------------------------------------------------------------------- #
# Subsets
# --------------------------------------------------------------------------- #
def build_subsets(lab: Curator) -> Dict[str, dict]:
    """Return name -> {view, desc, filter} for the curated subsets we show off."""
    subs: Dict[str, dict] = {}

    def add(name, view, desc, flt):
        subs[name] = {"view": view, "desc": desc, "filter": flt, "n": len(view)}

    add(
        "houses",
        lab.search("house").filter(category_in=STRUCT_CATS).filter(
            min_blocks=60, max_components=3, min_block_types=3
        ),
        "Buildings: search 'house' within structure categories, connected & multi-material.",
        "search('house') & cat in STRUCT_CATS & min_blocks>=60 & comps<=3 & types>=3",
    )
    add(
        "pixel_art",
        lab.filter(category_in=["Pixel Art Map"]).filter(min_blocks=30),
        "Pixel/flat art: a visually distinct, compact, token-tractable type.",
        "cat == 'Pixel Art Map' & min_blocks>=30",
    )
    add(
        "redstone",
        lab.filter(category_in=["Redstone Device Map"]).filter(min_blocks=40, max_blocks=255),
        "Redstone devices, capped to AR-tractable size (<=255 blocks).",
        "cat == 'Redstone Device Map' & 40<=blocks<=255",
    )
    add(
        "towers",
        lab.search("tower").filter(category_in=STRUCT_CATS).filter(min_blocks=40),
        "Tall builds: 'tower' within structure categories.",
        "search('tower') & cat in STRUCT_CATS & min_blocks>=40",
    )
    add(
        "trees",
        lab.search("tree").filter(min_blocks=30, max_components=4),
        "Organic builds: 'tree'.",
        "search('tree') & min_blocks>=30 & comps<=4",
    )
    add(
        "popular",
        lab.filter(min_diamonds=10).filter(min_blocks=30),
        "Crowd-validated seed set: >=10 diamonds.",
        "diamonds>=10 & min_blocks>=30",
    )
    return subs


def _structs(view: Curator) -> List[Structure]:
    return [view.structures[i] for i in view.indices]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _render_grid(structures, titles, path, cols=6, figscale=2.3, suptitle=None):
    """Robust contact sheet of structures (skips ones that fail to render)."""
    n = len(structures)
    if n == 0:
        return None
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(figscale * cols, figscale * rows))
    for k, s in enumerate(structures):
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        try:
            render_schem(s, ax=ax, max_dim=24, show=False)
        except Exception as e:  # noqa: BLE001
            ax.text(0.5, 0.5, 0.5, f"render err\n{type(e).__name__}", fontsize=6)
        ax.set_title(titles[k][:34], fontsize=7)
        ax.set_xlabel("");  ax.set_ylabel("");  ax.set_zlabel("")
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def render_subset_figures(lab: Curator, subsets: Dict[str, dict], figdir: Path,
                          per_sheet: int = 18, seed: int = 0) -> Dict[str, str]:
    """One contact sheet per subset + a material-variant figure. Returns name->path."""
    figdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    out: Dict[str, str] = {}

    for name, info in subsets.items():
        view = info["view"]
        idx = list(view.indices)
        if not idx:
            continue
        pick = rng.choice(len(idx), size=min(per_sheet, len(idx)), replace=False)
        chosen = [idx[i] for i in sorted(pick)]
        structs = [lab.structures[i] for i in chosen]
        titles = []
        for i in chosen:
            r = lab.features[i]
            t = (r["title"] or r["path"])[:24]
            titles.append(f"{t}\n{r['n_blocks']}blk d{r['diamonds']}")
        p = figdir / f"subset_{name}.png"
        _render_grid(structs, titles, p,
                     suptitle=f"{name}  (n={info['n']})  —  {info['desc']}")
        out[name] = str(p)
        print(f"[fig] {name:10s} n={info['n']:4d} -> {p.name}", flush=True)

    # Material-variant groups: same shape, different materials -> KEEP.
    try:
        variants = lab.find_variant_groups(iou_threshold=0.9)
        for gi, g in enumerate(variants[:3]):
            structs = [lab.structures[i] for i in g[:6]]
            titles = []
            for i in g[:6]:
                r = lab.features[i]
                titles.append(f"{(r['title'] or r['path'])[:18]}\n{r['dominant_block'].split(':')[-1]}")
            p = figdir / f"variant_group_{gi}.png"
            _render_grid(structs, titles, p, cols=6,
                         suptitle=f"Material-variant group {gi}: same shape, different blocks (KEEP all)")
            out[f"variant_group_{gi}"] = str(p)
            print(f"[fig] variant_group_{gi} size={len(g)} -> {p.name}", flush=True)
    except Exception:  # noqa: BLE001
        print("[fig] variant groups failed:\n" + traceback.format_exc(), flush=True)

    return out


# --------------------------------------------------------------------------- #
# Training one (subset, track)
# --------------------------------------------------------------------------- #
def _loss_curve(history: dict, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for k, v in history.items():
        if isinstance(v, list) and v:
            ax.plot(v, label=k)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def train_track(subset_name: str, track: str, structs: List[Structure],
                outdir: Path, epochs: int, n_samples: int = 16,
                device: str = "cuda") -> Optional[dict]:
    """Train one track on one subset; write configs/history/samples/novelty. Return metrics row."""
    run = outdir / f"{subset_name}__{track}"
    run.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"[{subset_name}/{track}] {m}", flush=True)  # noqa: E731
    t0 = time.time()

    vocab = build_block_vocab(structs, max_dim=24)
    try:
        if track == "ar":
            from blockgen.training.train_ar import (ARTrainConfig, sample_structures,
                                                    train_ar, build_sequences)
            cfg = ARTrainConfig(epochs=epochs, device=device)
            n_fit = len(build_sequences(structs, vocab, cfg.max_seq_len))
            log(f"{len(structs)} structs, {n_fit} fit seq<= {cfg.max_seq_len}; vocab={vocab.vocab_size}")
            if n_fit < 8:
                log("too few sequences fit; skipping"); return None
            model, history = train_ar(structs, vocab, cfg)
            samples = sample_structures(model, vocab, num_samples=n_samples,
                                        temperature=1.0, top_k=40)
            cfg_d = asdict(cfg)

        elif track == "diffusion":
            from blockgen.training.train_diffusion import (DiffusionTrainConfig,
                                                           calibrate_air_bias,
                                                           sample_structures, train_diffusion)
            cfg = DiffusionTrainConfig(epochs=epochs, device=device)
            log(f"{len(structs)} structs; grid={cfg.grid}; classes={vocab.num_blocks + 1}")
            model, history = train_diffusion(structs, vocab, cfg)
            med_occ = float(np.median([s.crop_to_non_air().occupied_mask.sum() for s in structs]))
            bias = calibrate_air_bias(model, vocab, target_occupancy=med_occ,
                                      grid=cfg.grid, steps=cfg.sample_steps, device=device)
            log(f"calibrated air_bias={bias} (target occ {med_occ:.0f})")
            samples = sample_structures(model, vocab, num_samples=n_samples, grid=cfg.grid,
                                        steps=cfg.sample_steps, air_bias=bias, device=device)
            cfg_d = asdict(cfg); cfg_d["air_bias"] = bias

        elif track == "graph":
            from blockgen.training.train_graph import (GraphTrainConfig, sample_structures,
                                                       train_graph)
            from blockgen.training.train_ar import build_sequences
            cfg = GraphTrainConfig(epochs=epochs, device=device)
            n_fit = len(build_sequences(structs, vocab, cfg.max_seq_len))
            log(f"{len(structs)} structs, {n_fit} fit seq; vocab={vocab.vocab_size}")
            if n_fit < 8:
                log("too few sequences fit; skipping"); return None
            model, history = train_graph(structs, vocab, cfg)
            samples = sample_structures(model, vocab, num_samples=n_samples,
                                        temperature=1.0, top_k=40, device=device)
            cfg_d = asdict(cfg)
        else:
            raise ValueError(track)
    except Exception:  # noqa: BLE001
        (run / "ERROR.txt").write_text(traceback.format_exc())
        log("FAILED:\n" + traceback.format_exc())
        return None

    train_min = (time.time() - t0) / 60.0

    # Novelty eval: compare samples to the SUBSET they trained on (memorization check).
    report = evaluate_novelty(samples, structs, vocab, grid=24, k=3)
    row = summary_row(f"{subset_name}/{track}", report)
    row["n_train"] = len(structs); row["n_samples"] = len(samples)
    row["train_min"] = round(train_min, 2); row["final_loss"] = round(history["loss"][-1], 4)
    row["median_sample_occ"] = float(np.median(
        [s.crop_to_non_air().occupied_mask.sum() for s in samples]))

    # Persist everything.
    (run / "config.json").write_text(json.dumps(
        {"subset": subset_name, "track": track, "config": cfg_d,
         "vocab_size": vocab.vocab_size, "n_blocks_vocab": vocab.num_blocks}, indent=2))
    (run / "history.json").write_text(json.dumps(history, indent=2))
    (run / "novelty.json").write_text(json.dumps(row, indent=2))
    _loss_curve(history, run / "loss.png", f"{subset_name}/{track} loss")

    titles = [f"sample#{i}\nocc={int(s.crop_to_non_air().occupied_mask.sum())}"
              for i, s in enumerate(samples)]
    _render_grid(samples, titles, run / "samples.png",
                 suptitle=f"{subset_name}/{track} — generated samples")
    try:
        fig = render_comparison_grid(samples, structs, report, num_rows=4, k=3,
                                     title=f"{subset_name}/{track}: sample vs nearest training neighbors")
        fig.savefig(run / "comparison.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    except Exception:  # noqa: BLE001
        (run / "comparison_ERROR.txt").write_text(traceback.format_exc())

    log(f"done in {train_min:.1f} min  nn_iou={row['mean_nn_iou']} dup={row['duplicate_rate']} "
        f"div={row['diversity']} valid={row['validity_rate']}")
    return row


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
# Which tracks to train on which subset (only feasible pairings).
TRAIN_PLAN = [
    ("pixel_art", ["ar", "diffusion", "graph"]),  # all-tracks comparison (compact)
    ("houses", ["diffusion"]),                     # headline dense build (token tracks can't fit)
    ("redstone", ["ar"]),                          # token track on its best-supported type
]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Curated-subset figures + retraining + novelty.")
    ap.add_argument("--stamp", default="manual", help="run id (use $(date +%Y%m%d_%H%M%S))")
    ap.add_argument("--epochs-ar", type=int, default=60)
    ap.add_argument("--epochs-diff", type=int, default=80)
    ap.add_argument("--epochs-graph", type=int, default=60)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--quick", action="store_true", help="2 epochs each (timing probe)")
    args = ap.parse_args()

    ep = {"ar": args.epochs_ar, "diffusion": args.epochs_diff, "graph": args.epochs_graph}
    if args.quick:
        ep = {k: 2 for k in ep}

    outdir = OUTPUTS / f"run_{args.stamp}"
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    (outdir / "models").mkdir(parents=True, exist_ok=True)
    print(f"=== run {args.stamp} -> {outdir} ===", flush=True)

    lab = Curator.from_labeled_cache(max_dim=24)
    subsets = build_subsets(lab)

    # Reproducibility manifest.
    manifest = {
        "stamp": args.stamp, "git_sha": _git_sha(),
        "cache": "tf_small_24 (labeled, 5866)", "epochs": ep,
        "n_samples": args.samples, "train_plan": [list(p) for p in TRAIN_PLAN],
        "subsets": {n: {"n": i["n"], "filter": i["filter"], "desc": i["desc"]}
                    for n, i in subsets.items()},
    }
    (outdir / "config.json").write_text(json.dumps(manifest, indent=2))

    # 1) Figures.
    fig_paths = render_subset_figures(lab, subsets, outdir / "figures")
    (outdir / "figures" / "index.json").write_text(json.dumps(fig_paths, indent=2))

    if args.figures_only:
        print("figures-only: done", flush=True); return

    # 2) Retraining + novelty.
    rows: List[dict] = []
    for subset_name, tracks in TRAIN_PLAN:
        structs = _structs(subsets[subset_name]["view"])
        if len(structs) < 8:
            print(f"[skip] {subset_name}: only {len(structs)} structs", flush=True); continue
        for track in tracks:
            row = train_track(subset_name, track, structs, outdir / "models",
                              epochs=ep[track], n_samples=args.samples)
            if row:
                rows.append(row)
                # Persist incrementally so a crash still leaves partial results.
                (outdir / "metrics.json").write_text(json.dumps(rows, indent=2))

    # 3) Metrics table.
    if rows:
        cols = ["track", "n_train", "n_samples", "final_loss", "mean_nn_iou",
                "duplicate_rate", "diversity", "validity_rate", "mean_block_agreement",
                "median_sample_occ", "train_min"]
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        (outdir / "metrics.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
    print(f"=== run {args.stamp} complete ===", flush=True)


if __name__ == "__main__":
    main()
