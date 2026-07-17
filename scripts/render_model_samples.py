"""Textured renders of GENERATED samples from the ideas-battery checkpoints.

Rebuilds the exact gc-houses-large prep from run_20260704_001222_ideas (same
defaults, same rng seed), reloads chosen checkpoints, samples fresh structures,
and renders dense textured grids next to a real-data reference grid at the
same canonical scale.

    .venv/bin/python scripts/render_model_samples.py --samples 48
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

RUN = REPO / "outputs" / "run_20260704_001222_ideas" / "gc-houses-large"
FIGS = REPO / "outputs" / "figures"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--tile-px", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from blockgen.experiments_ideas import prepare
    from blockgen.experiments_overnight import ds_gc_houses_large
    from blockgen.renderer.grid import save_grid, _tile_textured
    from blockgen.training.train_ar import ARTrainConfig
    from blockgen.training.train_ar import sample_structures as ar_sample
    from blockgen.training.constrained_decode import sample_constrained_structures
    from blockgen.utils.serialize import build_block_vocab

    # Battery run defaults (cmd.txt only overrode epochs, which don't matter here).
    run_args = SimpleNamespace(val_frac=0.15, augment=True, canon_dim=16,
                               ar_seq=1600, samples=args.samples)
    torch.manual_seed(args.seed)

    structs, labels, _ = ds_gc_houses_large(16)
    fit_c, ref_c, val_c, train_aug, train_raw, val, _ = prepare(
        "gc-houses-large", structs, labels, run_args)
    dim = run_args.canon_dim
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Real-data reference grid at the same canonical scale as the AR samples.
    rng = np.random.default_rng(args.seed)
    ref_pick = [ref_c[i] for i in rng.permutation(len(ref_c))[:args.samples]]
    save_grid(ref_pick, str(FIGS / "samples_real_canon16.png"),
              cols=args.cols, tile_px=args.tile_px, tile_fn=_tile_textured)

    # --- AR arms (canon 16 vocab) -------------------------------------------
    vocab = build_block_vocab(ref_c + val_c, max_dim=dim)
    cfg = ARTrainConfig(max_seq_len=run_args.ar_seq, device=dev)

    def _stock_ar(path: Path):
        from blockgen.models.voxel_transformer_ar import VoxelTransformerAR
        m = VoxelTransformerAR(vocab_size=vocab.vocab_size, max_seq_len=cfg.max_seq_len,
                               d_model=cfg.d_model, nhead=cfg.nhead,
                               num_layers=cfg.num_layers,
                               dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        return m.to(dev).eval()

    def _pe_ar(path: Path, pe: str):
        from blockgen.models.voxel_transformer_ar2 import VoxelTransformerAR2
        m = VoxelTransformerAR2(pe=pe, vocab_size=vocab.vocab_size,
                                max_seq_len=cfg.max_seq_len, d_model=cfg.d_model,
                                nhead=cfg.nhead, num_layers=cfg.num_layers,
                                dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        return m.to(dev).eval()

    t0 = time.time()
    model = _stock_ar(RUN / "ar_raster_constrained" / "model.pt")
    s = sample_constrained_structures(model, vocab, num_samples=args.samples,
                                      temperature=1.0, top_k=40)
    print(f"ar_raster_constrained: {len(s)} samples in {time.time()-t0:.0f}s", flush=True)
    save_grid(s, str(FIGS / "samples_ar_raster_constrained.png"),
              cols=args.cols, tile_px=args.tile_px, tile_fn=_tile_textured)
    del model

    t0 = time.time()
    model = _pe_ar(RUN / "ar_pe_phase4" / "model.pt", pe="phase4")
    s = ar_sample(model, vocab, num_samples=args.samples, temperature=1.0, top_k=40)
    print(f"ar_pe_phase4: {len(s)} samples in {time.time()-t0:.0f}s", flush=True)
    save_grid(s, str(FIGS / "samples_ar_pe_phase4.png"),
              cols=args.cols, tile_px=args.tile_px, tile_fn=_tile_textured)
    del model

    # --- diffusion baseline (grid 32 vocab) ---------------------------------
    from blockgen.models.voxel_diffusion import VoxelUNet3D, sample_grids
    from blockgen.training.train_diffusion import DiffusionTrainConfig, calibrate_air_bias
    from blockgen.utils.serialize import grid_to_structure, num_grid_classes

    grid = 32
    dvocab = build_block_vocab(train_raw + val, max_dim=grid)
    dcfg = DiffusionTrainConfig(grid=grid, device=dev)
    dmodel = VoxelUNet3D(num_classes=num_grid_classes(dvocab),
                         base_channels=dcfg.base_channels)
    dmodel.load_state_dict(torch.load(RUN / "diff32_shared" / "model.pt",
                                      map_location="cpu"))
    dmodel = dmodel.to(dev).eval()
    med = float(np.median([int(x.occupied_mask.sum()) for x in train_raw]))
    bias = calibrate_air_bias(dmodel, dvocab, target_occupancy=med, grid=grid,
                              steps=dcfg.sample_steps, device=dev)
    t0 = time.time()
    grids = sample_grids(dmodel, grid, args.samples, steps=dcfg.sample_steps,
                         device=dev, air_bias=bias)
    s = [grid_to_structure(g.cpu().numpy(), dvocab) for g in grids]
    s = [x for x in s if int(x.occupied_mask.sum()) > 0]
    print(f"diff32_maskgit: {len(s)} non-empty samples in {time.time()-t0:.0f}s", flush=True)
    save_grid(s, str(FIGS / "samples_diff32_maskgit.png"),
              cols=args.cols, tile_px=args.tile_px, tile_fn=_tile_textured)

    print("DONE-MODEL-SAMPLE-GRIDS", flush=True)


if __name__ == "__main__":
    main()
