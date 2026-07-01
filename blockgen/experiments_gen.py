"""How each model *generates*: process filmstrips + AR-vs-diffusion-vs-flow on houses.

Two studies, written to ``outputs/run_<stamp>_gen/``:

G1 (fidelity, 24^3 houses) — train the masked-diffusion net once, then sample it two
   ways and film the process:
     * MaskGIT (confidence-ranked unmasking)         -> ``film_diffusion_maskgit.png``
     * discrete flow matching (rate-driven unmasking) -> ``film_diffusion_flow.png``
   plus a novelty table comparing the two *samplers on the same weights*.

G2 (the AR-on-houses test, canonical N^3) — houses are too big to tokenize at 24^3
   (median 1086 blocks). **Scale normalization** (downsample to a fixed N^3) is what
   unlocks the token tracks, so we canonicalize, then race **AR vs diffusion** on the
   identical canonical set and film AR's progressive voxel build-up
   (``film_ar_progressive.png``). This answers "does AR work best on houses?" and
   "do we need different normalization?" with one experiment.

Run::

    .venv/bin/python -m blockgen.experiments_gen --stamp $(date +%Y%m%d_%H%M%S)
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
from blockgen.eval.novelty import evaluate_novelty, summary_row  # noqa: E402
from blockgen.renderer.render import render_schem  # noqa: E402
from blockgen.utils.data import Structure  # noqa: E402
from blockgen.utils.serialize import (  # noqa: E402
    BOS_TOKEN,
    EOS_TOKEN,
    build_block_vocab,
    grid_to_structure,
    structure_to_tokens,
    tokens_to_structure,
)

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "outputs"
STRUCT_CATS = ["Land Structure Map", "Air Structure Map", "Other Map", "Complex Map"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def houses_subset(lab: Curator) -> List[Structure]:
    v = lab.search("house").filter(category_in=STRUCT_CATS).filter(
        min_blocks=60, max_components=3, min_block_types=3)
    return [lab.structures[i] for i in v.indices]


def canonicalize(structs: List[Structure], dim: int) -> List[Structure]:
    """Scale-normalize: crop, downsample so max axis <= dim, recrop. Fixed footprint scale."""
    out = []
    for s in structs:
        c = s.crop_to_non_air()
        if max(c.shape) > dim:
            c = c.downsample(max_dim=dim).crop_to_non_air()
        out.append(c)
    return out


def ar_feasible(structs: List[Structure], vocab, max_seq_len: int) -> List[Structure]:
    keep = []
    for s in structs:
        try:
            if 2 < len(structure_to_tokens(s, vocab)) <= max_seq_len:
                keep.append(s)
        except ValueError:
            continue
    return keep


def _render_strip(structs, titles, path, suptitle=None, figscale=2.4):
    n = len(structs)
    fig = plt.figure(figsize=(figscale * n, figscale + 0.6))
    for k, s in enumerate(structs):
        ax = fig.add_subplot(1, n, k + 1, projection="3d")
        try:
            render_schem(s, ax=ax, max_dim=24, show=False)
        except Exception:  # noqa: BLE001
            ax.text(0.5, 0.5, 0.5, "empty", fontsize=8)
        ax.set_title(titles[k], fontsize=9)
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def _calib_sampler(fn, model, grid, target, steps, device, candidates, probe=6):
    """Pick the air_bias whose median sampled occupancy best matches a target, for `fn`."""
    best_b, best_e = 0.0, float("inf")
    for b in candidates:
        x = fn(model, grid, probe, steps=steps, device=device, air_bias=b)
        med = float(np.median([(g > 0).sum().item() for g in x]))
        e = abs(med - target)
        if e < best_e:
            best_e, best_b = e, b
    return best_b


def _grid_film(traj, vocab, idx, picks, path, suptitle):
    """Filmstrip from a diffusion trajectory (list of (B,D,D,D) grids) for sample idx."""
    structs, titles = [], []
    steps = len(traj)
    for p in picks:
        g = traj[p][idx].cpu().numpy()
        structs.append(grid_to_structure(g, vocab))
        occ = int((g > 0).sum())
        titles.append(f"step {p + 1}/{steps}\nocc={occ}")
    _render_strip(structs, titles, path, suptitle)


# --------------------------------------------------------------------------- #
# G1 — diffusion sampler study (24^3) + filmstrips
# --------------------------------------------------------------------------- #
def study_diffusion(structs, outdir: Path, epochs: int, grid: int = 24,
                    n_samples: int = 16, device: str = "cuda") -> List[dict]:
    from blockgen.models.voxel_diffusion import sample_grids, sample_grids_flow
    from blockgen.training.train_diffusion import (DiffusionTrainConfig,
                                                   calibrate_air_bias, train_diffusion)
    log = lambda m: print(f"[G1/diff{grid}] {m}", flush=True)  # noqa: E731
    vocab = build_block_vocab(structs, max_dim=grid)
    cfg = DiffusionTrainConfig(grid=grid, epochs=epochs, device=device)
    model, history = train_diffusion(structs, vocab, cfg)
    torch.save(model.state_dict(), outdir / f"diffusion_{grid}.pt")

    med_occ = float(np.median([s.crop_to_non_air().occupied_mask.sum() for s in structs]))
    bias = calibrate_air_bias(model, vocab, target_occupancy=med_occ, grid=grid,
                              steps=cfg.sample_steps, device=device)
    log(f"air_bias={bias} target_occ={med_occ:.0f}")
    steps = cfg.sample_steps

    rows = []
    for name, fn in [("maskgit", sample_grids), ("flow", sample_grids_flow)]:
        # Each sampler controls density differently (MaskGIT commits confident=often-air
        # voxels first; flow reveals random voxels by rate), so calibrate air_bias per
        # sampler for a fair occupancy match. Flow over-fills -> needs negative bias.
        s_bias = bias
        if name == "flow":
            s_bias = _calib_sampler(fn, model, grid, med_occ, steps, device,
                                    candidates=[-8, -6, -4, -3, -2, -1, 0, 1, 2])
            log(f"flow air_bias={s_bias}")
        x, traj = fn(model, grid, n_samples, steps=steps, device=device,
                     air_bias=s_bias, return_trajectory=True)
        samples = [grid_to_structure(g.cpu().numpy(), vocab) for g in x]
        rep = evaluate_novelty(samples, structs, vocab, grid=grid, k=3)
        row = summary_row(f"houses{grid}/diffusion-{name}", rep)
        row["median_sample_occ"] = float(np.median(
            [s.crop_to_non_air().occupied_mask.sum() for s in samples]))
        rows.append(row)
        picks = sorted(set(np.linspace(0, steps - 1, 6).astype(int).tolist()))
        _grid_film(traj, vocab, 0, picks, outdir / f"film_diffusion_{name}.png",
                   f"Diffusion ({name}) generation process — houses {grid}³  (all-MASK → build)")
        log(f"{name}: nn_iou={row['mean_nn_iou']} valid={row['validity_rate']} "
            f"div={row['diversity']} occ={row['median_sample_occ']:.0f}")
    (outdir / "diffusion_history.json").write_text(json.dumps(history, indent=2))
    return rows


# --------------------------------------------------------------------------- #
# G2 — AR vs diffusion on canonical houses + AR progressive film
# --------------------------------------------------------------------------- #
def study_ar_vs_diffusion(canon: List[Structure], outdir: Path, dim: int,
                          ar_seq: int, epochs_ar: int, epochs_diff: int,
                          n_samples: int = 16, device: str = "cuda") -> List[dict]:
    from blockgen.training.train_ar import ARTrainConfig, train_ar
    from blockgen.training.train_ar import sample_structures as ar_sample
    from blockgen.training.train_diffusion import (DiffusionTrainConfig,
                                                   calibrate_air_bias, train_diffusion)
    from blockgen.training.train_diffusion import sample_structures as diff_sample
    log = lambda m: print(f"[G2/canon{dim}] {m}", flush=True)  # noqa: E731

    vocab = build_block_vocab(canon, max_dim=dim)  # shared vocab -> fair IoU
    shared = ar_feasible(canon, vocab, ar_seq)
    log(f"{len(canon)} canonical houses, {len(shared)} fit AR seq<= {ar_seq}; vocab={vocab.vocab_size}")
    rows = []

    # --- AR ---
    try:
        cfg = ARTrainConfig(max_seq_len=ar_seq, epochs=epochs_ar, batch_size=8,
                            d_model=256, num_layers=6, device=device)
        model, hist = train_ar(shared, vocab, cfg)
        torch.save(model.state_dict(), outdir / f"ar_{dim}.pt")
        ar_samples = ar_sample(model, vocab, num_samples=n_samples, temperature=1.0, top_k=40)
        rep = evaluate_novelty(ar_samples, shared, vocab, grid=dim, k=3)
        row = summary_row(f"houses{dim}/ar", rep)
        row["final_loss"] = round(hist["loss"][-1], 4)
        row["median_sample_occ"] = float(np.median(
            [s.crop_to_non_air().occupied_mask.sum() for s in ar_samples]))
        rows.append(row)
        (outdir / "ar_history.json").write_text(json.dumps(hist, indent=2))
        log(f"AR: nn_iou={row['mean_nn_iou']} valid={row['validity_rate']} "
            f"div={row['diversity']} occ={row['median_sample_occ']:.0f}")
        _ar_progressive_film(model, vocab, outdir / "film_ar_progressive.png", dim, ar_seq, device)
    except Exception:  # noqa: BLE001
        (outdir / "ar_ERROR.txt").write_text(traceback.format_exc())
        log("AR FAILED:\n" + traceback.format_exc())

    # --- diffusion on the SAME canonical set ---
    try:
        cfg = DiffusionTrainConfig(grid=dim, epochs=epochs_diff, device=device)
        model, hist = train_diffusion(shared, vocab, cfg)
        torch.save(model.state_dict(), outdir / f"canon_diffusion_{dim}.pt")
        med = float(np.median([s.crop_to_non_air().occupied_mask.sum() for s in shared]))
        bias = calibrate_air_bias(model, vocab, target_occupancy=med, grid=dim,
                                  steps=cfg.sample_steps, device=device)
        d_samples = diff_sample(model, vocab, num_samples=n_samples, grid=dim,
                                steps=cfg.sample_steps, air_bias=bias, device=device)
        rep = evaluate_novelty(d_samples, shared, vocab, grid=dim, k=3)
        row = summary_row(f"houses{dim}/diffusion", rep)
        row["final_loss"] = round(hist["loss"][-1], 4)
        row["median_sample_occ"] = float(np.median(
            [s.crop_to_non_air().occupied_mask.sum() for s in d_samples]))
        rows.append(row)
        log(f"diffusion: nn_iou={row['mean_nn_iou']} valid={row['validity_rate']} "
            f"div={row['diversity']} occ={row['median_sample_occ']:.0f}")
    except Exception:  # noqa: BLE001
        (outdir / "canon_diff_ERROR.txt").write_text(traceback.format_exc())
        log("DIFF FAILED:\n" + traceback.format_exc())
    return rows


@torch.no_grad()
def _ar_progressive_film(model, vocab, path: Path, dim: int, ar_seq: int, device: str):
    """Film the AR build: decode increasing prefixes of one generated token stream."""
    model.eval()
    toks = model.generate(bos_token_id=BOS_TOKEN, eos_token_id=EOS_TOKEN,
                          max_new_tokens=ar_seq - 1, temperature=1.0, top_k=40).tolist()[0]
    # voxel boundaries are every 4 tokens after BOS; show cumulative placement.
    body = [t for t in toks if t not in (BOS_TOKEN, EOS_TOKEN)]
    n_vox = len(body) // 4
    if n_vox < 4:
        return
    structs, titles = [], []
    for frac in [0.15, 0.3, 0.5, 0.7, 0.85, 1.0]:
        k = max(1, int(n_vox * frac))
        prefix = [BOS_TOKEN] + body[: 4 * k] + [EOS_TOKEN]
        s = tokens_to_structure(prefix, vocab)
        structs.append(s)
        titles.append(f"{int(frac * 100)}% tokens\n{k}/{n_vox} voxels")
    _render_strip(structs, titles, path,
                  f"AR generation process — houses {dim}³  (voxels emitted left→right)")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default="manual")
    ap.add_argument("--canon-dim", type=int, default=12)
    ap.add_argument("--ar-seq", type=int, default=1600)
    ap.add_argument("--epochs-diff24", type=int, default=150)
    ap.add_argument("--epochs-ar", type=int, default=80)
    ap.add_argument("--epochs-diff-canon", type=int, default=120)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--skip-g1", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.epochs_diff24 = args.epochs_ar = args.epochs_diff_canon = 3

    outdir = OUTPUTS / f"run_{args.stamp}_gen"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"=== gen study {args.stamp} -> {outdir} ===", flush=True)
    t0 = time.time()

    lab = Curator.from_labeled_cache(max_dim=24)
    houses = houses_subset(lab)
    print(f"houses: {len(houses)}", flush=True)

    manifest = {"stamp": args.stamp, "n_houses": len(houses),
                "canon_dim": args.canon_dim, "ar_seq": args.ar_seq,
                "epochs": {"diff24": args.epochs_diff24, "ar": args.epochs_ar,
                           "diff_canon": args.epochs_diff_canon}}
    (outdir / "config.json").write_text(json.dumps(manifest, indent=2))

    rows: List[dict] = []
    if not args.skip_g1:
        try:
            rows += study_diffusion(houses, outdir, args.epochs_diff24, grid=24,
                                    n_samples=args.samples)
        except Exception:  # noqa: BLE001
            (outdir / "g1_ERROR.txt").write_text(traceback.format_exc())
            print("G1 FAILED:\n" + traceback.format_exc(), flush=True)

    canon = canonicalize(houses, args.canon_dim)
    rows += study_ar_vs_diffusion(canon, outdir, args.canon_dim, args.ar_seq,
                                  args.epochs_ar, args.epochs_diff_canon, n_samples=args.samples)

    (outdir / "metrics.json").write_text(json.dumps(rows, indent=2))
    if rows:
        cols = ["track", "final_loss", "mean_nn_iou", "duplicate_rate", "diversity",
                "validity_rate", "mean_block_agreement", "median_sample_occ"]
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        (outdir / "metrics.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
    print(f"=== done in {(time.time() - t0) / 60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
