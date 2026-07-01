"""Train our best models on the GrabCraft dataset (clean, category-labeled builds).

GrabCraft gives one coherent build *type* per category (e.g. 724 medieval-houses), which
is exactly the "learn one object class" supervision we want. We train the models that won
on Minecraft-schematics houses:

  * AR on scale-normalized (canonical N³) builds — our best house generator,
  * diffusion at 24³ (handles dense builds; MaskGIT vs flow-matching sampler),
  * diffusion at N³ for a direct AR-vs-diffusion head-to-head on identical data,

and render the full evidence: a data contact sheet, generated-sample sheets,
sample-vs-nearest-neighbor novelty grids, and generation-process filmstrips.

    .venv/bin/python -m blockgen.experiments_grabcraft --stamp $(date +%Y%m%d_%H%M%S) --category medieval-houses
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from blockgen.curation.curate import Curator  # noqa: E402
from blockgen.eval.novelty import (  # noqa: E402
    evaluate_novelty,
    render_comparison_grid,
    summary_row,
)
from blockgen.experiments import _render_grid  # noqa: E402
from blockgen.experiments_gen import (  # noqa: E402
    _ar_progressive_film,
    _calib_sampler,
    _grid_film,
    ar_feasible,
    canonicalize,
)
from blockgen.utils.data import Structure  # noqa: E402
from blockgen.utils.serialize import build_block_vocab, grid_to_structure  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "outputs"

HOUSE_CLASSES = ["medieval-houses", "modern-houses", "wooden-houses", "stone-houses",
                 "quartz-houses", "survival-houses", "starter-houses", "brick-houses",
                 "tree-houses"]


def _occ(s: Structure) -> int:
    return int(s.crop_to_non_air().occupied_mask.sum())


def _samples_sheet(samples, path, suptitle):
    titles = [f"#{i}\nocc={_occ(s)}" for i, s in enumerate(samples)]
    _render_grid(samples, titles, path, suptitle=suptitle)


def _novelty(samples, train, vocab, grid, run: Path, tag: str) -> dict:
    rep = evaluate_novelty(samples, train, vocab, grid=grid, k=3)
    row = summary_row(tag, rep)
    row["median_sample_occ"] = float(np.median([_occ(s) for s in samples]))
    try:
        fig = render_comparison_grid(samples, train, rep, num_rows=4, k=3,
                                     title=f"{tag}: sample vs nearest training neighbors")
        fig.savefig(run / "comparison.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    except Exception:  # noqa: BLE001
        (run / "comparison_ERROR.txt").write_text(traceback.format_exc())
    return row


# --------------------------------------------------------------------------- #
def train_ar_canon(structs, outdir: Path, dim: int, ar_seq: int, epochs: int,
                   n_samples: int, device: str, batch: int = 8) -> Optional[dict]:
    from blockgen.training.train_ar import ARTrainConfig, train_ar
    from blockgen.training.train_ar import sample_structures as ar_sample
    run = outdir / f"ar_canon{dim}"; run.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"[AR{dim}] {m}", flush=True)  # noqa: E731
    canon = canonicalize(structs, dim)
    vocab = build_block_vocab(canon, max_dim=dim)
    shared = ar_feasible(canon, vocab, ar_seq)
    log(f"{len(canon)} canonical, {len(shared)} fit AR seq<= {ar_seq}; vocab={vocab.vocab_size}")
    if len(shared) < 16:
        log("too few fit; skipping"); return None
    try:
        cfg = ARTrainConfig(max_seq_len=ar_seq, epochs=epochs, batch_size=batch, device=device)
        model, hist = train_ar(shared, vocab, cfg)
        torch.save(model.state_dict(), run / "model.pt")
        samples = ar_sample(model, vocab, num_samples=n_samples, temperature=1.0, top_k=40)
        _samples_sheet(samples, run / "samples.png", f"GrabCraft AR (canonical {dim}³) — generated")
        _ar_progressive_film(model, vocab, run / "film_ar_progressive.png", dim, ar_seq, device)
        row = _novelty(samples, shared, vocab, dim, run, f"grabcraft/ar{dim}")
        row["final_loss"] = round(hist["loss"][-1], 4); row["n_train"] = len(shared)
        (run / "history.json").write_text(json.dumps(hist, indent=2))
        (run / "novelty.json").write_text(json.dumps(row, indent=2))
        log(f"nn_iou={row['mean_nn_iou']} valid={row['validity_rate']} div={row['diversity']} "
            f"occ={row['median_sample_occ']:.0f}")
        return row
    except Exception:  # noqa: BLE001
        (run / "ERROR.txt").write_text(traceback.format_exc()); log("FAILED"); return None


def train_diffusion_at(structs, outdir: Path, grid: int, epochs: int, n_samples: int,
                       device: str, with_flow: bool) -> List[dict]:
    from blockgen.models.voxel_diffusion import sample_grids, sample_grids_flow
    from blockgen.training.train_diffusion import (DiffusionTrainConfig,
                                                   calibrate_air_bias, train_diffusion)
    run = outdir / f"diffusion{grid}"; run.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"[DIFF{grid}] {m}", flush=True)  # noqa: E731
    vocab = build_block_vocab(structs, max_dim=grid)
    rows: List[dict] = []
    try:
        # bigger grids need a smaller batch to fit in GPU memory
        bsz = 8 if grid >= 32 else 16
        cfg = DiffusionTrainConfig(grid=grid, epochs=epochs, batch_size=bsz, device=device)
        model, hist = train_diffusion(structs, vocab, cfg)
        torch.save(model.state_dict(), run / "model.pt")
        (run / "history.json").write_text(json.dumps(hist, indent=2))
        med = float(np.median([_occ(s) for s in structs]))
        steps = cfg.sample_steps
        samplers = [("maskgit", sample_grids)]
        if with_flow:
            samplers.append(("flow", sample_grids_flow))
        for name, fn in samplers:
            cands = [-8, -6, -4, -3, -2, -1, 0, 1, 2] if name == "flow" else None
            bias = _calib_sampler(fn, model, grid, med, steps, device,
                                  candidates=cands) if cands else \
                calibrate_air_bias(model, vocab, target_occupancy=med, grid=grid,
                                   steps=steps, device=device)
            x, traj = fn(model, grid, n_samples, steps=steps, device=device,
                         air_bias=bias, return_trajectory=True)
            samples = [grid_to_structure(g.cpu().numpy(), vocab) for g in x]
            sub = run / name; sub.mkdir(exist_ok=True)
            _samples_sheet(samples, sub / "samples.png", f"GrabCraft diffusion-{name} {grid}³")
            picks = sorted(set(np.linspace(0, steps - 1, 6).astype(int).tolist()))
            _grid_film(traj, vocab, 0, picks, sub / "film.png",
                       f"Diffusion ({name}) generation — GrabCraft {grid}³")
            row = _novelty(samples, structs, vocab, grid, sub, f"grabcraft/diffusion{grid}-{name}")
            row["final_loss"] = round(hist["loss"][-1], 4); row["n_train"] = len(structs)
            (sub / "novelty.json").write_text(json.dumps(row, indent=2))
            rows.append(row)
            log(f"{name}: nn_iou={row['mean_nn_iou']} valid={row['validity_rate']} "
                f"occ={row['median_sample_occ']:.0f} (target {med:.0f})")
    except Exception:  # noqa: BLE001
        (run / "ERROR.txt").write_text(traceback.format_exc()); log("FAILED")
    return rows


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default="manual")
    ap.add_argument("--category", default="medieval-houses")
    ap.add_argument("--categories", default=None, help="comma-separated category list (overrides --category)")
    ap.add_argument("--house-classes", action="store_true", help="train on all *-houses categories")
    ap.add_argument("--max-dim", type=int, default=24, help="which gc_small_<dim> cache to load")
    ap.add_argument("--dedup", action="store_true", help="drop exact-duplicate builds (keep material variants)")
    ap.add_argument("--min-blocks", type=int, default=40)
    ap.add_argument("--canon-dim", type=int, default=12)
    ap.add_argument("--ar-seq", type=int, default=1600)
    ap.add_argument("--ar-batch", type=int, default=8)
    ap.add_argument("--diff-grid", type=int, default=24, help="native diffusion grid (dense track)")
    ap.add_argument("--epochs-ar", type=int, default=100)
    ap.add_argument("--epochs-diff24", type=int, default=150)
    ap.add_argument("--epochs-diff-canon", type=int, default=120)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.epochs_ar = args.epochs_diff24 = args.epochs_diff_canon = 3

    if args.house_classes:
        cats = HOUSE_CLASSES
    elif args.categories:
        cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    else:
        cats = [args.category]
    label = "house-classes" if args.house_classes else "+".join(cats)[:40]

    outdir = OUTPUTS / f"run_{args.stamp}_grabcraft"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"=== grabcraft [{label}] {args.stamp} -> {outdir} ===", flush=True)
    t0 = time.time()

    gc = Curator.from_grabcraft_cache(max_dim=args.max_dim)
    view = gc.filter(category_in=cats).filter(min_blocks=args.min_blocks)
    structs = [gc.structures[i] for i in view.indices]
    from collections import Counter
    per_cat = Counter(gc.features[i]["category"] for i in view.indices)
    print(f"[{label}] {len(structs)} builds (>= {args.min_blocks} blocks) from cache dim {args.max_dim}",
          flush=True)
    print("  per category:", dict(per_cat.most_common()), flush=True)
    if len(structs) < 16:
        print("too few; aborting", flush=True); return

    n_dropped = 0
    if args.dedup:
        sub = Curator.from_structures(structs, max_dim=args.max_dim)
        n_dropped = sub.dedupe_keep_variants(iou_threshold=0.95)
        structs = sub.apply()
        print(f"[dedup] dropped {n_dropped} exact-duplicate builds -> {len(structs)} kept", flush=True)

    (outdir / "config.json").write_text(json.dumps({
        "stamp": args.stamp, "source": "grabcraft", "categories": cats, "label": label,
        "cache_max_dim": args.max_dim, "n_builds": len(structs), "dedup_dropped": n_dropped,
        "per_category": dict(per_cat), "canon_dim": args.canon_dim, "ar_seq": args.ar_seq,
        "diff_grid": args.diff_grid,
        "epochs": {"ar": args.epochs_ar, "diff_native": args.epochs_diff24,
                   "diff_canon": args.epochs_diff_canon}}, indent=2))

    # data contact sheet (show off the dataset)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(structs), size=min(18, len(structs)), replace=False)
    titles = [f"{_occ(structs[i])}blk" for i in pick]
    _render_grid([structs[i] for i in pick], titles, outdir / "data_sheet.png",
                 suptitle=f"GrabCraft — {label} (n={len(structs)}, cache dim {args.max_dim})")

    rows: List[dict] = []
    # headline: AR on canonical builds (our best)
    r = train_ar_canon(structs, outdir, args.canon_dim, args.ar_seq, args.epochs_ar,
                       args.samples, "cuda", batch=args.ar_batch)
    if r:
        rows.append(r)
    # dense-fidelity diffusion + flow sampler study (native grid)
    rows += train_diffusion_at(structs, outdir, args.diff_grid, args.epochs_diff24,
                               args.samples, "cuda", with_flow=True)
    # head-to-head diffusion at canonical dim (vs AR)
    canon = canonicalize(structs, args.canon_dim)
    shared = ar_feasible(canon, build_block_vocab(canon, max_dim=args.canon_dim), args.ar_seq)
    rows += train_diffusion_at(shared, outdir, args.canon_dim, args.epochs_diff_canon,
                               args.samples, "cuda", with_flow=False)

    (outdir / "metrics.json").write_text(json.dumps(rows, indent=2))
    if rows:
        cols = ["track", "n_train", "final_loss", "mean_nn_iou", "duplicate_rate",
                "diversity", "validity_rate", "mean_block_agreement", "median_sample_occ"]
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        (outdir / "metrics.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
    print(f"=== done in {(time.time() - t0) / 60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
