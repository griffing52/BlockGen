"""Overnight experiment battery: cohesion + data levers across datasets & models.

Sweeps the ideas that target our two open problems — *samples look like the type
but aren't cohesive*, and *the labeled subsets are small* — over three datasets:

  datasets: gc-houses-large (footprint>=16), combined-houses (GrabCraft + schematics
            cache), gc-vehicles (cars/transportation from the dim-24 cache)
  methods : AR-flat (baseline), AR-cluster (3D-BPE), AR-cluster @higher-res,
            AR-conditioned (category token), diffusion (native grid),
            graph-VAE (PyG block+port graph encoder -> latent -> GRU token decoder)
  levers  : D4 augmentation (train only, 8x), held-out val split for honest novelty,
            connectivity/validity gate (LCC repair) reported as an ablation on AR.

Every experiment is guarded (writes ERROR.txt and continues), every artifact lands
under ``outputs/run_<stamp>_overnight/<dataset>/<method>/``, and a master
``leaderboard.md`` aggregates dataset x method. Novelty is reported both vs the
*train* set (duplicate_rate = memorization) and vs the held-out *val* set
(val_nn_iou = does it resemble unseen real builds), with the real val-vs-train IoU
printed as the reference for "how close is a genuine new build".

    .venv/bin/python -m blockgen.experiments_overnight --stamp $(date +%Y%m%d_%H%M%S)
    .venv/bin/python -m blockgen.experiments_overnight --stamp smoke --quick   # wiring test
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from blockgen.curation.curate import Curator  # noqa: E402
from blockgen.eval.novelty import (evaluate_novelty, render_comparison_grid,  # noqa: E402
                                    summary_row)
from blockgen.eval.validity import repair_lcc  # noqa: E402
from blockgen.experiments import _render_grid  # noqa: E402
from blockgen.experiments_gen import canonicalize  # noqa: E402
from blockgen.training.train_ar import ARTrainConfig  # noqa: E402
from blockgen.utils.augment import augment_with_labels  # noqa: E402
from blockgen.utils.data import Structure  # noqa: E402
from blockgen.utils.serialize import (build_block_vocab, grid_to_structure,  # noqa: E402
                                       structure_to_tokens)

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "outputs"

HOUSE_CLASSES = ["medieval-houses", "modern-houses", "wooden-houses", "stone-houses",
                 "quartz-houses", "survival-houses", "starter-houses", "brick-houses",
                 "tree-houses"]
VEHICLE_CLASSES = ["working-vehicles", "other-transportation", "planes",
                   "emergency-vehicles", "boats", "family-cars", "sports-cars",
                   "cars", "buses", "spaceships"]
STRUCT_CATS = ["Land Structure Map", "Air Structure Map", "Other Map", "Complex Map"]


def _occ(s: Structure) -> int:
    return int(s.crop_to_non_air().occupied_mask.sum())


# --------------------------------------------------------------------------- #
# dataset builders  -> (structs, labels, categories_present)
# --------------------------------------------------------------------------- #
def ds_gc_houses_large(min_dim: int) -> Tuple[List[Structure], List[str], List[str]]:
    gc = Curator.from_grabcraft_cache(max_dim=32)
    v = gc.filter(category_in=HOUSE_CLASSES, min_dim=min_dim, min_blocks=60)
    structs = [gc.structures[i] for i in v.indices]
    labels = [gc.features[i]["category"] for i in v.indices]
    return structs, labels, [c for c in HOUSE_CLASSES if c in set(labels)]


def ds_combined_houses(min_dim: int) -> Tuple[List[Structure], List[str], List[str]]:
    structs, labels, cats = ds_gc_houses_large(min_dim)
    try:
        lab = Curator.from_labeled_cache(max_dim=24)
        v = lab.search("house").filter(category_in=STRUCT_CATS).filter(
            min_blocks=60, max_components=3, min_block_types=3)
        for i in v.indices:
            structs.append(lab.structures[i]); labels.append("schematics-house")
        if any(lab_ == "schematics-house" for lab_ in labels):
            cats = cats + ["schematics-house"]
    except Exception:  # noqa: BLE001
        print("[combined] labeled cache unavailable; GrabCraft houses only", flush=True)
    return structs, labels, cats


def ds_gc_vehicles(min_blocks: int) -> Tuple[List[Structure], List[str], List[str]]:
    gc = Curator.from_grabcraft_cache(max_dim=24)
    v = gc.filter(category_in=VEHICLE_CLASSES, min_blocks=min_blocks)
    structs = [gc.structures[i] for i in v.indices]
    labels = [gc.features[i]["category"] for i in v.indices]
    return structs, labels, [c for c in VEHICLE_CLASSES if c in set(labels)]


# --------------------------------------------------------------------------- #
# evaluation helper
# --------------------------------------------------------------------------- #
def _eval_row(tag: str, samples: List[Structure], train: List[Structure],
              val: List[Structure], vocab, grid: int, run: Path,
              gated: bool = True) -> dict:
    rep = evaluate_novelty(samples, train, vocab, grid=grid, k=3)
    row = summary_row(tag, rep)                     # metrics vs TRAIN (dup=memorization)
    row["median_sample_occ"] = float(np.median([_occ(s) for s in samples]))
    if val:
        row["val_nn_iou"] = round(evaluate_novelty(samples, val, vocab, grid=grid, k=3).mean_nn_iou, 3)
    if gated:
        rep_g = evaluate_novelty(repair_lcc(samples), train, vocab, grid=grid, k=3)
        row["validity_gated"] = round(rep_g.validity_rate, 3)
        row["nn_iou_gated"] = round(rep_g.mean_nn_iou, 3)
    try:
        fig = render_comparison_grid(samples, train, rep, num_rows=4, k=3,
                                     title=f"{tag}: sample vs nearest train neighbors")
        fig.savefig(run / "comparison.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    except Exception:  # noqa: BLE001
        (run / "comparison_ERROR.txt").write_text(traceback.format_exc())
    titles = [f"#{i} occ={_occ(s)}" for i, s in enumerate(samples)]
    try:
        _render_grid(samples, titles, run / "samples.png", suptitle=tag)
    except Exception:  # noqa: BLE001
        pass
    (run / "novelty.json").write_text(json.dumps(row, indent=2))
    return row


def _log(ds: str, m: str) -> None:
    print(f"[{ds}] {m}", flush=True)


# --------------------------------------------------------------------------- #
# experiments (each returns a row or None; never raises)
# --------------------------------------------------------------------------- #
def exp_flat_ar(fit, ref, val, out: Path, dim: int, seq: int, epochs: int, samples: int,
                batch: int) -> Optional[dict]:
    from blockgen.experiments_gen import ar_feasible
    from blockgen.training.train_ar import sample_structures as ar_sample
    from blockgen.training.train_ar import train_ar
    run = out / "ar_flat"; run.mkdir(parents=True, exist_ok=True)
    try:
        vocab = build_block_vocab(ref + val, max_dim=dim)
        shared = ar_feasible(fit, vocab, seq)
        _log(out.name, f"flat-AR: {len(shared)}/{len(fit)} fit seq<={seq}; vocab={vocab.vocab_size}")
        if len(shared) < 16:
            (run / "SKIP.txt").write_text("too few fit AR seq"); return None
        cfg = ARTrainConfig(max_seq_len=seq, epochs=epochs, batch_size=batch, device="cuda")
        model, hist = train_ar(shared, vocab, cfg)
        torch.save(model.state_dict(), run / "model.pt")
        s = ar_sample(model, vocab, num_samples=samples, temperature=1.0, top_k=40)
        row = _eval_row("ar_flat", s, ref, val, vocab, dim, run)
        row["final_loss"] = round(hist["loss"][-1], 4); row["n_train"] = len(shared)
        _log(out.name, f"flat-AR done: nn_iou={row['mean_nn_iou']} val_nn={row.get('val_nn_iou')} "
             f"dup={row['duplicate_rate']} valid={row['validity_rate']}->{row.get('validity_gated')}")
        return row
    except Exception:  # noqa: BLE001
        (run / "ERROR.txt").write_text(traceback.format_exc()); _log(out.name, "flat-AR FAILED"); return None


def exp_cluster_ar(fit, ref, val, out: Path, dim: int, seq: int, epochs: int, samples: int,
                   batch: int, n_merges: int, tag: str = "ar_cluster") -> Optional[dict]:
    from blockgen.tokenizers.cluster_bpe import build_cluster_sequences, learn_clusters
    from blockgen.training.train_ar_ext import sample_cluster_structures, train_from_sequences
    run = out / tag; run.mkdir(parents=True, exist_ok=True)
    try:
        vocab = build_block_vocab(ref + val, max_dim=dim)  # for novelty grid only
        cv = learn_clusters(fit, max_dim=dim, n_merges=n_merges, max_corpus=400)
        seqs, kept = build_cluster_sequences(fit, cv, seq)
        _log(out.name, f"{tag}@{dim}: {len(kept)}/{len(fit)} fit; pieces={cv.n_pieces} "
             f"vocab={cv.vocab_size} median_len={int(np.median([len(s) for s in seqs])) if seqs else 0}")
        if len(seqs) < 16:
            (run / "SKIP.txt").write_text("too few fit cluster seq"); return None
        cfg = ARTrainConfig(max_seq_len=seq, epochs=epochs, batch_size=batch, device="cuda")
        model, hist = train_from_sequences(seqs, cv.vocab_size, cfg)
        torch.save(model.state_dict(), run / "model.pt")
        (run / "cluster_meta.json").write_text(json.dumps(
            {"n_pieces": cv.n_pieces, "num_blocks": cv.num_blocks, "n_merges": len(cv.merges),
             "median_seq_len": int(np.median([len(s) for s in seqs]))}, indent=2))
        s = sample_cluster_structures(model, cv, num_samples=samples, temperature=1.0, top_k=40)
        row = _eval_row(tag, s, ref, val, vocab, dim, run)
        row["final_loss"] = round(hist["loss"][-1], 4); row["n_train"] = len(seqs)
        _log(out.name, f"{tag} done: nn_iou={row['mean_nn_iou']} val_nn={row.get('val_nn_iou')} "
             f"dup={row['duplicate_rate']} valid={row['validity_rate']}->{row.get('validity_gated')}")
        return row
    except Exception:  # noqa: BLE001
        (run / "ERROR.txt").write_text(traceback.format_exc()); _log(out.name, f"{tag} FAILED"); return None


def exp_conditioned_ar(fit, fit_labels, ref, val, out: Path, dim: int, seq: int,
                       epochs: int, samples: int, batch: int, categories: List[str]) -> Optional[dict]:
    from blockgen.training.train_ar_ext import (build_conditioned_sequences,
                                                sample_conditioned_structures, train_from_sequences)
    run = out / "ar_conditioned"; run.mkdir(parents=True, exist_ok=True)
    try:
        if len(categories) < 2:
            (run / "SKIP.txt").write_text("single category; conditioning n/a"); return None
        vocab = build_block_vocab(ref + val, max_dim=dim)
        seqs, kept_labels = build_conditioned_sequences(fit, fit_labels, vocab, categories, seq)
        _log(out.name, f"cond-AR: {len(seqs)}/{len(fit)} fit; {len(categories)} categories")
        if len(seqs) < 16:
            (run / "SKIP.txt").write_text("too few fit conditioned seq"); return None
        cfg = ARTrainConfig(max_seq_len=seq, epochs=epochs, batch_size=batch, device="cuda")
        model, hist = train_from_sequences(seqs, vocab.vocab_size + len(categories), cfg)
        torch.save(model.state_dict(), run / "model.pt")
        # sample a balanced spread across categories
        per = max(1, samples // len(categories))
        s: List[Structure] = []
        for ci in range(len(categories)):
            s += sample_conditioned_structures(model, vocab, ci, num_samples=per, temperature=1.0, top_k=40)
        s = s[:samples] if len(s) >= samples else s
        row = _eval_row("ar_conditioned", s, ref, val, vocab, dim, run)
        row["final_loss"] = round(hist["loss"][-1], 4); row["n_train"] = len(seqs)
        (run / "categories.json").write_text(json.dumps(categories, indent=2))
        _log(out.name, f"cond-AR done: nn_iou={row['mean_nn_iou']} val_nn={row.get('val_nn_iou')} "
             f"valid={row['validity_rate']}->{row.get('validity_gated')}")
        return row
    except Exception:  # noqa: BLE001
        (run / "ERROR.txt").write_text(traceback.format_exc()); _log(out.name, "cond-AR FAILED"); return None


def exp_diffusion(fit, ref, val, out: Path, grid: int, epochs: int, samples: int) -> Optional[dict]:
    from blockgen.models.voxel_diffusion import sample_grids
    from blockgen.training.train_diffusion import (DiffusionTrainConfig, calibrate_air_bias,
                                                   train_diffusion)
    run = out / f"diffusion{grid}"; run.mkdir(parents=True, exist_ok=True)
    try:
        vocab = build_block_vocab(ref + val, max_dim=grid)
        bsz = 8 if grid >= 32 else 16
        cfg = DiffusionTrainConfig(grid=grid, epochs=epochs, batch_size=bsz, device="cuda")
        model, hist = train_diffusion(fit, vocab, cfg)
        torch.save(model.state_dict(), run / "model.pt")
        med = float(np.median([_occ(s) for s in ref]))
        bias = calibrate_air_bias(model, vocab, target_occupancy=med, grid=grid,
                                  steps=cfg.sample_steps, device="cuda")
        x = sample_grids(model, grid, samples, steps=cfg.sample_steps, device="cuda", air_bias=bias)
        s = [grid_to_structure(g.cpu().numpy(), vocab) for g in x]
        row = _eval_row(f"diffusion{grid}", s, ref, val, vocab, grid, run, gated=False)
        row["final_loss"] = round(hist["loss"][-1], 4); row["n_train"] = len(fit)
        _log(out.name, f"diffusion{grid} done: nn_iou={row['mean_nn_iou']} val_nn={row.get('val_nn_iou')} "
             f"valid={row['validity_rate']} occ={row['median_sample_occ']:.0f} (target {med:.0f})")
        return row
    except Exception:  # noqa: BLE001
        (run / "ERROR.txt").write_text(traceback.format_exc()); _log(out.name, f"diffusion{grid} FAILED"); return None


def exp_graph_vae(fit, ref, val, out: Path, dim: int, seq: int, epochs: int,
                  samples: int, batch: int) -> Optional[dict]:
    from blockgen.experiments_gen import ar_feasible
    from blockgen.training.train_graph import (GraphTrainConfig,
                                               sample_structures as graph_sample,
                                               train_graph)
    run = out / "graph_vae"; run.mkdir(parents=True, exist_ok=True)
    try:
        vocab = build_block_vocab(ref + val, max_dim=dim)
        shared = ar_feasible(fit, vocab, seq)
        _log(out.name, f"graph-VAE: {len(shared)}/{len(fit)} fit seq<={seq}; vocab={vocab.vocab_size}")
        if len(shared) < 16:
            (run / "SKIP.txt").write_text("too few fit graph/token pairs"); return None
        cfg = GraphTrainConfig(max_seq_len=seq, epochs=epochs, batch_size=batch, device="cuda")
        model, hist = train_graph(shared, vocab, cfg)
        torch.save(model.state_dict(), run / "model.pt")
        s = graph_sample(model, vocab, num_samples=samples, max_len=seq,
                         temperature=1.0, top_k=40)
        row = _eval_row("graph_vae", s, ref, val, vocab, dim, run)
        row["final_loss"] = round(hist["loss"][-1], 4); row["n_train"] = len(shared)
        row["final_kl"] = round(hist["kl"][-1], 4)
        _log(out.name, f"graph-VAE done: nn_iou={row['mean_nn_iou']} val_nn={row.get('val_nn_iou')} "
             f"dup={row['duplicate_rate']} valid={row['validity_rate']}->{row.get('validity_gated')}")
        return row
    except Exception:  # noqa: BLE001
        (run / "ERROR.txt").write_text(traceback.format_exc()); _log(out.name, "graph-VAE FAILED"); return None


# --------------------------------------------------------------------------- #
def run_dataset(name: str, structs: List[Structure], labels: List[str],
                categories: List[str], outdir: Path, args) -> List[dict]:
    out = outdir / name; out.mkdir(parents=True, exist_ok=True)
    from collections import Counter
    _log(name, f"{len(structs)} builds; categories={dict(Counter(labels))}")
    if len(structs) < 32:
        (out / "SKIP.txt").write_text("dataset too small"); return []

    # dedup (keep material variants) on the raw set
    sub = Curator.from_structures(structs, max_dim=32)
    n_drop = sub.dedupe_keep_variants(iou_threshold=0.95)
    removed = set(sub.remove_list())
    pairs = [(structs[i], labels[i]) for i in range(len(structs))
             if (structs[i].source_path or f"#{i}") not in removed]
    structs, labels = [p[0] for p in pairs], [p[1] for p in pairs]
    _log(name, f"dedup dropped {n_drop} -> {len(structs)} kept")

    # held-out val split (on raw builds, BEFORE augmentation)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(structs))
    n_val = max(8, int(round(args.val_frac * len(structs))))
    val_idx, tr_idx = set(perm[:n_val].tolist()), perm[n_val:].tolist()
    train_raw = [structs[i] for i in tr_idx]; train_lab = [labels[i] for i in tr_idx]
    val = [structs[i] for i in val_idx]
    _log(name, f"split: {len(train_raw)} train / {len(val)} val")

    # D4 augmentation on the TRAIN set only (the val set stays untouched)
    if args.augment:
        train_aug, aug_lab = augment_with_labels(train_raw, train_lab)
        _log(name, f"D4 augment: {len(train_raw)} -> {len(train_aug)} fit builds")
    else:
        train_aug, aug_lab = train_raw, train_lab

    # canonicalize (scale-normalize). fit = augmented (for training); ref = distinct
    # real builds (for vocab + all novelty eval, so metrics compare to real builds,
    # not to rotated copies — and eval stays ~8x cheaper).
    dim = args.canon_dim
    fit_c = canonicalize(train_aug, dim)
    ref_c = canonicalize(train_raw, dim)
    val_c = canonicalize(val, dim)

    # data contact sheet (distinct real builds)
    pick = rng.choice(len(ref_c), size=min(18, len(ref_c)), replace=False)
    _render_grid([ref_c[i] for i in pick], [f"{_occ(ref_c[i])}blk" for i in pick],
                 out / "data_sheet.png", suptitle=f"{name} (train n={len(ref_c)} distinct, canon {dim})")

    # reference: how close is a genuine unseen build to the train set?
    vbase = None
    try:
        vv = build_block_vocab(ref_c + val_c, max_dim=dim)
        vbase = round(float(np.mean(evaluate_novelty(val_c, ref_c, vv, grid=dim, k=1).nn_iou)), 3)
        _log(name, f"val-vs-train baseline NN-IoU (real held-out builds): {vbase}")
    except Exception:  # noqa: BLE001
        pass

    (out / "config.json").write_text(json.dumps({
        "dataset": name, "n_fit": len(fit_c), "n_ref": len(ref_c), "n_val": len(val_c),
        "categories": categories, "augment": args.augment, "canon_dim": dim,
        "val_baseline_nn_iou": vbase}, indent=2))

    rows: List[dict] = []
    # 1. flat AR baseline
    r = exp_flat_ar(fit_c, ref_c, val_c, out, dim, args.ar_seq, args.epochs_ar,
                    args.samples, args.ar_batch)
    if r: rows.append(r)
    # 2. cluster (3D-BPE) AR at the same resolution -> direct comparison to flat
    r = exp_cluster_ar(fit_c, ref_c, val_c, out, dim, args.ar_seq, args.epochs_ar,
                       args.samples, args.ar_batch, args.n_merges)
    if r: rows.append(r)
    # 3. cluster AR at higher resolution -> showcase the "reach" that BPE unlocks
    if args.hi_dim > dim:
        r = exp_cluster_ar(canonicalize(train_aug, args.hi_dim), canonicalize(train_raw, args.hi_dim),
                           canonicalize(val, args.hi_dim), out, args.hi_dim, args.ar_seq,
                           args.epochs_ar, args.samples, args.ar_batch, args.n_merges,
                           tag=f"ar_cluster_hi{args.hi_dim}")
        if r: rows.append(r)
    # 4. category-conditioned AR (one model, any class)
    r = exp_conditioned_ar(fit_c, aug_lab, ref_c, val_c, out, dim, args.ar_seq, args.epochs_ar,
                           args.samples, args.ar_batch, categories)
    if r: rows.append(r)
    # 5. diffusion at the native grid (dense track; trains on augmented full-res)
    r = exp_diffusion(train_aug, train_raw, val, out, args.diff_grid, args.epochs_diff, args.samples)
    if r: rows.append(r)
    # 6. graph VAE (block+port graph -> latent -> token decoder; transfer-relevant track)
    r = exp_graph_vae(fit_c, ref_c, val_c, out, dim, args.ar_seq, args.epochs_graph,
                      args.samples, args.ar_batch)
    if r: rows.append(r)

    for r in rows:
        r["dataset"] = name; r["val_baseline_nn_iou"] = vbase
    (out / "metrics.json").write_text(json.dumps(rows, indent=2))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default="manual")
    ap.add_argument("--datasets", default="gc-houses-large,combined-houses,gc-vehicles",
                    help="comma list of: gc-houses-large, combined-houses, gc-vehicles")
    ap.add_argument("--house-min-dim", type=int, default=16, help="footprint floor for 'large' houses")
    ap.add_argument("--vehicle-min-blocks", type=int, default=55)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--augment", action="store_true", default=True)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--canon-dim", type=int, default=16)
    ap.add_argument("--hi-dim", type=int, default=20, help="higher canonical res for the BPE reach test (0=off)")
    ap.add_argument("--ar-seq", type=int, default=1600)
    ap.add_argument("--ar-batch", type=int, default=8)
    ap.add_argument("--n-merges", type=int, default=250)
    ap.add_argument("--diff-grid", type=int, default=32)
    ap.add_argument("--epochs-ar", type=int, default=90)
    ap.add_argument("--epochs-diff", type=int, default=130)
    ap.add_argument("--epochs-graph", type=int, default=60)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.epochs_ar = args.epochs_diff = args.epochs_graph = 3
        args.n_merges = 40; args.samples = 4; args.diff_grid = 16; args.hi_dim = 18

    outdir = OUTPUTS / f"run_{args.stamp}_overnight"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"=== overnight battery {args.stamp} -> {outdir} ===", flush=True)
    print(f"config: {vars(args)}", flush=True)
    t0 = time.time()

    builders = {
        "gc-houses-large": lambda: ds_gc_houses_large(args.house_min_dim),
        "combined-houses": lambda: ds_combined_houses(args.house_min_dim),
        "gc-vehicles": lambda: ds_gc_vehicles(args.vehicle_min_blocks),
    }
    wanted = [d.strip() for d in args.datasets.split(",") if d.strip()]

    all_rows: List[dict] = []
    for name in wanted:
        if name not in builders:
            print(f"unknown dataset '{name}', skipping", flush=True); continue
        try:
            structs, labels, cats = builders[name]()
            all_rows += run_dataset(name, structs, labels, cats, outdir, args)
        except Exception:  # noqa: BLE001
            (outdir / f"{name}_ERROR.txt").write_text(traceback.format_exc())
            print(f"dataset {name} FAILED:\n{traceback.format_exc()}", flush=True)
        # write the leaderboard after every dataset so partial results survive
        _write_leaderboard(all_rows, outdir)
    print(f"=== done in {(time.time() - t0) / 60:.1f} min ===", flush=True)


def _write_leaderboard(rows: List[dict], outdir: Path) -> None:
    (outdir / "leaderboard.json").write_text(json.dumps(rows, indent=2))
    if not rows:
        return
    cols = ["dataset", "track", "n_train", "final_loss", "mean_nn_iou", "duplicate_rate",
            "val_nn_iou", "val_baseline_nn_iou", "diversity", "validity_rate",
            "validity_gated", "mean_block_agreement", "median_sample_occ"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    (outdir / "leaderboard.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
