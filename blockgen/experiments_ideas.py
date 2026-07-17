"""Ideas battery: ablations for the deep-researched quality levers (research.md).

Four arm groups on one dataset (default: gc-houses-large), sharing the overnight
battery's prep (dedup -> val split -> D4 augment -> canonicalize) and eval
(`_eval_row`: train-NN-IoU, duplicate_rate, val_nn_iou, validity raw/gated):

  A. **Positional encodings** (flat AR): learned | sin | rope | alibi | phase4.
     `learned` re-baselines the custom-attention implementation against the
     stock model from the overnight run.
  B. **Ordering + constrained decoding**: BFS-from-ground training order,
     sampled unconstrained AND with in-loop 6-adjacency logit gating; plus
     constrained sampling from the raster model (train-test-mismatch control).
  C. **Diffusion samplers** (one model, four inference rules): maskgit | flow |
     remask (ReMDM-lite) | stratified (octant-spread commits).
  D. **Two-stage occupancy->materials** (Scaffold recipe + generated occupancy)
     vs the single-stage diffusion row from group C.

    .venv/bin/python -m blockgen.experiments_ideas --stamp $(date +%Y%m%d_%H%M%S)
    .venv/bin/python -m blockgen.experiments_ideas --stamp smoke --quick
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
import numpy as np  # noqa: E402
import torch  # noqa: E402

from blockgen.curation.curate import Curator  # noqa: E402
from blockgen.eval.novelty import evaluate_novelty  # noqa: E402
from blockgen.experiments_gen import ar_feasible, canonicalize  # noqa: E402
from blockgen.experiments_overnight import (OUTPUTS, _eval_row, _log,  # noqa: E402
                                            _render_grid, _write_leaderboard, _occ,
                                            ds_combined_houses, ds_gc_houses_large,
                                            ds_gc_vehicles)
from blockgen.training.train_ar import ARTrainConfig, build_sequences  # noqa: E402
from blockgen.training.train_ar import sample_structures as ar_sample  # noqa: E402
from blockgen.utils.augment import augment_with_labels  # noqa: E402
from blockgen.utils.serialize import EOS_TOKEN, build_block_vocab  # noqa: E402


def prepare(name, structs, labels, args):
    """Overnight-battery prep: dedup -> split -> augment -> canonicalize."""
    sub = Curator.from_structures(structs, max_dim=32)
    n_drop = sub.dedupe_keep_variants(iou_threshold=0.95)
    removed = set(sub.remove_list())
    pairs = [(structs[i], labels[i]) for i in range(len(structs))
             if (structs[i].source_path or f"#{i}") not in removed]
    structs, labels = [p[0] for p in pairs], [p[1] for p in pairs]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(structs))
    n_val = max(8, int(round(args.val_frac * len(structs))))
    val_idx, tr_idx = set(perm[:n_val].tolist()), perm[n_val:].tolist()
    train_raw = [structs[i] for i in tr_idx]
    val = [structs[i] for i in val_idx]
    train_aug, _ = augment_with_labels(train_raw, [labels[i] for i in tr_idx]) \
        if args.augment else (train_raw, None)
    dim = args.canon_dim
    fit_c, ref_c, val_c = canonicalize(train_aug, dim), canonicalize(train_raw, dim), canonicalize(val, dim)
    _log(name, f"dedup -{n_drop}; split {len(train_raw)}/{len(val)}; fit {len(fit_c)} @ {dim}")
    vv = build_block_vocab(ref_c + val_c, max_dim=dim)
    vbase = round(float(np.mean(evaluate_novelty(val_c, ref_c, vv, grid=dim, k=1).nn_iou)), 3)
    _log(name, f"val-vs-train baseline NN-IoU: {vbase}")
    return fit_c, ref_c, val_c, train_aug, train_raw, val, vbase


# --------------------------------------------------------------------------- #
# pause/resume support: each arm persists its row to <run>/novelty.json; on a
# rerun with the same --stamp, finished arms are loaded from disk and skipped
# (so `scripts/pause_ideas.sh` can kill the process to free ALL GPU memory and
# `scripts/resume_ideas.sh` restarts losing at most the interrupted arm).
def _done_row(run: Path) -> Optional[dict]:
    f = run / "novelty.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            return None
    return None


def _save_row(run: Path, row: dict) -> None:
    (run / "novelty.json").write_text(json.dumps(row, indent=2))


def exp_pe(fit, ref, val, out: Path, dim: int, args, pe: str) -> Optional[dict]:
    from blockgen.training.train_ar_ext import train_from_sequences
    tag = f"ar_pe_{pe}"
    run = out / tag; run.mkdir(parents=True, exist_ok=True)
    row = _done_row(run)
    if row is not None:
        _log(out.name, f"{tag}: already done, skipping"); return row
    try:
        vocab = build_block_vocab(ref + val, max_dim=dim)
        seqs = build_sequences(ar_feasible(fit, vocab, args.ar_seq), vocab, args.ar_seq)
        cfg = ARTrainConfig(max_seq_len=args.ar_seq, epochs=args.epochs_ar,
                            batch_size=args.ar_batch, device="cuda")
        model, hist = train_from_sequences(seqs, vocab.vocab_size, cfg, pe=pe)
        torch.save(model.state_dict(), run / "model.pt")
        s = ar_sample(model, vocab, num_samples=args.samples, temperature=1.0, top_k=40)
        row = _eval_row(tag, s, ref, val, vocab, dim, run)
        row["final_loss"] = round(hist["loss"][-1], 4); row["n_train"] = len(seqs)
        _save_row(run, row)
        _log(out.name, f"{tag} done: nn_iou={row['mean_nn_iou']} val_nn={row.get('val_nn_iou')} "
             f"valid={row['validity_rate']}->{row.get('validity_gated')}")
        return row
    except Exception:  # noqa: BLE001
        (run / "ERROR.txt").write_text(traceback.format_exc()); _log(out.name, f"{tag} FAILED"); return None


def _make_ar(vocab_size: int, cfg: ARTrainConfig, state_path: Path):
    """Rebuild a stock AR model from a checkpoint (for resume without retraining)."""
    from blockgen.models.voxel_transformer_ar import VoxelTransformerAR
    model = VoxelTransformerAR(
        vocab_size=vocab_size, max_seq_len=cfg.max_seq_len, d_model=cfg.d_model,
        nhead=cfg.nhead, num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout)
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    return model.to("cuda" if torch.cuda.is_available() else "cpu")


def exp_ordering(fit, ref, val, out: Path, dim: int, args) -> List[dict]:
    """BFS-order training; unconstrained + adjacency-constrained sampling arms."""
    from blockgen.training.constrained_decode import sample_constrained_structures
    from blockgen.training.train_ar_ext import train_from_sequences
    from blockgen.utils.ordering import build_bfs_sequences
    rows: List[dict] = []
    vocab = build_block_vocab(ref + val, max_dim=dim)
    cfg = ARTrainConfig(max_seq_len=args.ar_seq, epochs=args.epochs_ar,
                        batch_size=args.ar_batch, device="cuda")

    run = out / "ar_bfs"; run.mkdir(parents=True, exist_ok=True)
    run2 = out / "ar_bfs_constrained"; run2.mkdir(parents=True, exist_ok=True)
    r, r2 = _done_row(run), _done_row(run2)
    try:
        if r is not None and r2 is not None:
            _log(out.name, "ar_bfs + constrained: already done, skipping")
            rows += [r, r2]
        else:
            seqs = build_bfs_sequences(fit, vocab, args.ar_seq)
            if r is not None and (run / "model.pt").exists():
                _log(out.name, "ar_bfs: row done, reloading model for constrained arm")
                model = _make_ar(vocab.vocab_size, cfg, run / "model.pt")
                rows.append(r)
            else:
                _log(out.name, f"ar_bfs: {len(seqs)} BFS-ordered sequences")
                model, hist = train_from_sequences(seqs, vocab.vocab_size, cfg)
                torch.save(model.state_dict(), run / "model.pt")
                s = ar_sample(model, vocab, num_samples=args.samples, temperature=1.0, top_k=40)
                r = _eval_row("ar_bfs", s, ref, val, vocab, dim, run)
                r["final_loss"] = round(hist["loss"][-1], 4); r["n_train"] = len(seqs)
                _save_row(run, r); rows.append(r)
                _log(out.name, f"ar_bfs done: valid={r['validity_rate']} nn_iou={r['mean_nn_iou']}")
            s2 = sample_constrained_structures(model, vocab, num_samples=args.samples,
                                               temperature=1.0, top_k=40)
            r2 = _eval_row("ar_bfs_constrained", s2, ref, val, vocab, dim, run2, gated=False)
            r2["n_train"] = len(seqs)
            _save_row(run2, r2); rows.append(r2)
            _log(out.name, f"ar_bfs_constrained done: valid={r2['validity_rate']} "
                 f"nn_iou={r2['mean_nn_iou']} val_nn={r2.get('val_nn_iou')}")
    except Exception:  # noqa: BLE001
        (run / "ERROR.txt").write_text(traceback.format_exc()); _log(out.name, "ar_bfs FAILED")

    # control: constrained decoding on a raster-order model (train-test mismatch)
    run3 = out / "ar_raster_constrained"; run3.mkdir(parents=True, exist_ok=True)
    r3 = _done_row(run3)
    if r3 is not None:
        _log(out.name, "ar_raster_constrained: already done, skipping")
        rows.append(r3); return rows
    try:
        seqs = build_sequences(ar_feasible(fit, vocab, args.ar_seq), vocab, args.ar_seq)
        if (run3 / "model.pt").exists():
            model = _make_ar(vocab.vocab_size, cfg, run3 / "model.pt")
        else:
            model, _ = train_from_sequences(seqs, vocab.vocab_size, cfg)
            torch.save(model.state_dict(), run3 / "model.pt")
        s3 = sample_constrained_structures(model, vocab, num_samples=args.samples,
                                           temperature=1.0, top_k=40)
        r3 = _eval_row("ar_raster_constrained", s3, ref, val, vocab, dim, run3, gated=False)
        r3["n_train"] = len(seqs)
        _save_row(run3, r3); rows.append(r3)
        _log(out.name, f"ar_raster_constrained done: valid={r3['validity_rate']} nn_iou={r3['mean_nn_iou']}")
    except Exception:  # noqa: BLE001
        (run3 / "ERROR.txt").write_text(traceback.format_exc()); _log(out.name, "ar_raster_constrained FAILED")
    return rows


def exp_samplers(train_aug, train_raw, val, out: Path, args) -> List[dict]:
    """One diffusion model, four inference rules."""
    from blockgen.models.voxel_diffusion import (sample_grids, sample_grids_flow,
                                                 sample_grids_remask, sample_grids_stratified)
    from blockgen.training.train_diffusion import (DiffusionTrainConfig, calibrate_air_bias,
                                                   train_diffusion)
    from blockgen.utils.serialize import grid_to_structure
    rows: List[dict] = []
    grid = args.diff_grid
    run0 = out / f"diff{grid}_shared"; run0.mkdir(parents=True, exist_ok=True)
    done = {n: _done_row(out / f"diff{grid}_{n}")
            for n in ("maskgit", "flow", "remask", "stratified")}
    if all(v is not None for v in done.values()):
        _log(out.name, f"diff{grid}_*: all sampler arms done, skipping")
        return list(done.values())
    try:
        vocab = build_block_vocab(train_raw + val, max_dim=grid)
        cfg = DiffusionTrainConfig(grid=grid, epochs=args.epochs_diff,
                                   batch_size=8 if grid >= 32 else 16, device="cuda")
        if (run0 / "model.pt").exists():
            from blockgen.models.voxel_diffusion import VoxelUNet3D
            from blockgen.utils.serialize import num_grid_classes
            _log(out.name, f"diff{grid}: reloading shared model from checkpoint")
            model = VoxelUNet3D(num_classes=num_grid_classes(vocab),
                                base_channels=cfg.base_channels)
            model.load_state_dict(torch.load(run0 / "model.pt", map_location="cpu"))
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
            hist = {"loss": [float("nan")]}
        else:
            model, hist = train_diffusion(train_aug, vocab, cfg)
            torch.save(model.state_dict(), run0 / "model.pt")
        med = float(np.median([_occ(s) for s in train_raw]))
        bias = calibrate_air_bias(model, vocab, target_occupancy=med, grid=grid,
                                  steps=cfg.sample_steps, device="cuda")
        samplers = {
            "maskgit": lambda: sample_grids(model, grid, args.samples, steps=cfg.sample_steps,
                                            device="cuda", air_bias=bias),
            "flow": lambda: sample_grids_flow(model, grid, args.samples, steps=cfg.sample_steps,
                                              device="cuda", air_bias=bias),
            "remask": lambda: sample_grids_remask(model, grid, args.samples,
                                                  steps=max(cfg.sample_steps, 16),
                                                  device="cuda", air_bias=bias),
            "stratified": lambda: sample_grids_stratified(model, grid, args.samples,
                                                          steps=cfg.sample_steps,
                                                          device="cuda", air_bias=bias),
        }
        for name, fn in samplers.items():
            tag = f"diff{grid}_{name}"
            run = out / tag; run.mkdir(parents=True, exist_ok=True)
            if done[name] is not None:
                _log(out.name, f"{tag}: already done, skipping"); rows.append(done[name]); continue
            try:
                s = [grid_to_structure(g.cpu().numpy(), vocab) for g in fn()]
                r = _eval_row(tag, s, train_raw, val, vocab, grid, run, gated=False)
                r["final_loss"] = round(hist["loss"][-1], 4); r["n_train"] = len(train_aug)
                _save_row(run, r); rows.append(r)
                _log(out.name, f"{tag} done: nn_iou={r['mean_nn_iou']} val_nn={r.get('val_nn_iou')} "
                     f"valid={r['validity_rate']} occ={r['median_sample_occ']:.0f}")
            except Exception:  # noqa: BLE001
                (run / "ERROR.txt").write_text(traceback.format_exc()); _log(out.name, f"{tag} FAILED")
    except Exception:  # noqa: BLE001
        (run0 / "ERROR.txt").write_text(traceback.format_exc()); _log(out.name, "diff_shared FAILED")
    return rows


def exp_twostage(train_aug, train_raw, val, out: Path, args) -> Optional[dict]:
    from blockgen.training.train_diffusion import calibrate_air_bias
    from blockgen.training.train_twostage import (TwoStageConfig, sample_twostage,
                                                  train_materials, train_occupancy)
    grid = args.diff_grid
    tag = f"twostage{grid}"
    run = out / tag; run.mkdir(parents=True, exist_ok=True)
    row = _done_row(run)
    if row is not None:
        _log(out.name, f"{tag}: already done, skipping"); return row
    try:
        vocab = build_block_vocab(train_raw + val, max_dim=grid)
        cfg = TwoStageConfig(grid=grid, epochs_occ=args.epochs_diff, epochs_mat=args.epochs_diff,
                             batch_size=8 if grid >= 32 else 16, device="cuda")
        from blockgen.models.voxel_diffusion import VoxelUNet3D
        from blockgen.utils.serialize import num_grid_classes
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if (run / "occ_model.pt").exists():
            _log(out.name, f"{tag}: reloading occupancy model")
            occ_model = VoxelUNet3D(num_classes=2, base_channels=cfg.base_channels)
            occ_model.load_state_dict(torch.load(run / "occ_model.pt", map_location="cpu"))
            occ_model = occ_model.to(dev); h1 = {"loss": [float("nan")]}
        else:
            occ_model, h1 = train_occupancy(train_aug, vocab, cfg)
            torch.save(occ_model.state_dict(), run / "occ_model.pt")
        if (run / "mat_model.pt").exists():
            _log(out.name, f"{tag}: reloading material model")
            mat_model = VoxelUNet3D(num_classes=num_grid_classes(vocab),
                                    base_channels=cfg.base_channels)
            mat_model.load_state_dict(torch.load(run / "mat_model.pt", map_location="cpu"))
            mat_model = mat_model.to(dev); h2 = {"loss": [float("nan")]}
        else:
            mat_model, h2 = train_materials(train_aug, vocab, cfg)
            torch.save(mat_model.state_dict(), run / "mat_model.pt")
        med = float(np.median([_occ(s) for s in train_raw]))
        bias = calibrate_air_bias(occ_model, vocab, target_occupancy=med, grid=grid,
                                  steps=cfg.sample_steps, device="cuda")
        s = sample_twostage(occ_model, mat_model, vocab, grid, args.samples,
                            steps=cfg.sample_steps, occ_air_bias=bias, device="cuda")
        row = _eval_row(tag, s, train_raw, val, vocab, grid, run, gated=False)
        row["final_loss"] = round(h2["loss"][-1], 4); row["n_train"] = len(train_aug)
        row["occ_final_loss"] = round(h1["loss"][-1], 4)
        _save_row(run, row)
        _log(out.name, f"{tag} done: nn_iou={row['mean_nn_iou']} val_nn={row.get('val_nn_iou')} "
             f"valid={row['validity_rate']} block_agree={row['mean_block_agreement']} "
             f"occ={row['median_sample_occ']:.0f} (target {med:.0f})")
        return row
    except Exception:  # noqa: BLE001
        (run / "ERROR.txt").write_text(traceback.format_exc()); _log(out.name, f"{tag} FAILED"); return None


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default="manual")
    ap.add_argument("--config", default=None,
                    help="YAML config (path or name under configs/); its values "
                         "become defaults, explicit CLI flags still override")
    ap.add_argument("--dataset", default="gc-houses-large",
                    choices=["gc-houses-large", "combined-houses", "gc-vehicles"])
    ap.add_argument("--groups", default="pe,ordering,samplers,twostage",
                    help="comma list of: pe, ordering, samplers, twostage")
    ap.add_argument("--pe-arms", default="learned,sin,rope,alibi,phase4")
    ap.add_argument("--house-min-dim", type=int, default=16)
    ap.add_argument("--vehicle-min-blocks", type=int, default=55)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--augment", action="store_true", default=True)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--canon-dim", type=int, default=16)
    ap.add_argument("--ar-seq", type=int, default=1600)
    ap.add_argument("--ar-batch", type=int, default=8)
    ap.add_argument("--diff-grid", type=int, default=32)
    ap.add_argument("--epochs-ar", type=int, default=90)
    ap.add_argument("--epochs-diff", type=int, default=110)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.config:
        from blockgen.config import load_config, apply_to_parser
        apply_to_parser(ap, load_config(args.config))
        args = ap.parse_args()  # re-parse so CLI flags still win over the config
    if args.quick:
        args.epochs_ar = args.epochs_diff = 3
        args.samples = 4; args.diff_grid = 16
        args.pe_arms = "rope,phase4"

    outdir = OUTPUTS / f"run_{args.stamp}_ideas"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"=== ideas battery {args.stamp} -> {outdir} ===", flush=True)
    print(f"config: {vars(args)}", flush=True)
    t0 = time.time()

    builders = {
        "gc-houses-large": lambda: ds_gc_houses_large(args.house_min_dim),
        "combined-houses": lambda: ds_combined_houses(args.house_min_dim),
        "gc-vehicles": lambda: ds_gc_vehicles(args.vehicle_min_blocks),
    }
    structs, labels, _cats = builders[args.dataset]()
    out = outdir / args.dataset; out.mkdir(parents=True, exist_ok=True)
    fit_c, ref_c, val_c, train_aug, train_raw, val, vbase = prepare(args.dataset, structs, labels, args)

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    rows: List[dict] = []
    if "pe" in groups:
        for pe in [p.strip() for p in args.pe_arms.split(",") if p.strip()]:
            r = exp_pe(fit_c, ref_c, val_c, out, args.canon_dim, args, pe)
            if r: rows.append(r)
            _write_leaderboard([dict(r, dataset=args.dataset, val_baseline_nn_iou=vbase)
                                for r in rows], outdir)
    if "ordering" in groups:
        rows += exp_ordering(fit_c, ref_c, val_c, out, args.canon_dim, args)
        _write_leaderboard([dict(r, dataset=args.dataset, val_baseline_nn_iou=vbase)
                            for r in rows], outdir)
    if "samplers" in groups:
        rows += exp_samplers(train_aug, train_raw, val, out, args)
        _write_leaderboard([dict(r, dataset=args.dataset, val_baseline_nn_iou=vbase)
                            for r in rows], outdir)
    if "twostage" in groups:
        r = exp_twostage(train_aug, train_raw, val, out, args)
        if r: rows.append(r)

    final = [dict(r, dataset=args.dataset, val_baseline_nn_iou=vbase) for r in rows]
    (out / "metrics.json").write_text(json.dumps(final, indent=2))
    _write_leaderboard(final, outdir)
    print(f"=== done in {(time.time() - t0) / 60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
