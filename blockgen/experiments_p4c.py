"""phase4 PE + in-loop adjacency constraint — the two T11 winners, combined.

Both pieces already exist: the trained ``ar_pe_phase4`` checkpoint from the
ideas battery and inference-time constrained decoding. No training needed —
rebuild the battery's exact prep/vocab, load the checkpoint, sample with the
constraint, and evaluate with the standard protocol. Row lands next to the T11
arms in ``outputs/run_20260704_001222_ideas/gc-houses-large/``.

T11 references (gc-houses-large): ar_pe_phase4 val_nn 0.405 / validity 0.375;
ar_raster_constrained val_nn 0.383 / validity 1.0. Hypothesis: combo ≈ 0.40+
val_nn at validity 1.0.

    .venv/bin/python -m blockgen.experiments_p4c
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from blockgen.experiments_ideas import prepare
from blockgen.experiments_overnight import OUTPUTS, _eval_row, _log, ds_gc_houses_large
from blockgen.training.constrained_decode import sample_constrained_structures
from blockgen.training.train_ar import ARTrainConfig
from blockgen.utils.serialize import build_block_vocab

RUN = OUTPUTS / "run_20260704_001222_ideas" / "gc-houses-large"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    run_args = SimpleNamespace(val_frac=0.15, augment=True, canon_dim=16,
                               ar_seq=1600, samples=a.samples)
    structs, labels, _ = ds_gc_houses_large(16)
    fit_c, ref_c, val_c, *_ , vbase = prepare("gc-houses-large", structs, labels, run_args)
    dim = run_args.canon_dim
    vocab = build_block_vocab(ref_c + val_c, max_dim=dim)
    cfg = ARTrainConfig(max_seq_len=run_args.ar_seq)

    from blockgen.models.voxel_transformer_ar2 import VoxelTransformerAR2
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = VoxelTransformerAR2(pe="phase4", vocab_size=vocab.vocab_size,
                                max_seq_len=cfg.max_seq_len, d_model=cfg.d_model,
                                nhead=cfg.nhead, num_layers=cfg.num_layers,
                                dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout)
    model.load_state_dict(torch.load(RUN / "ar_pe_phase4" / "model.pt", map_location="cpu"))
    model = model.to(dev).eval()

    tag = "ar_phase4_constrained"
    run = RUN / tag
    run.mkdir(parents=True, exist_ok=True)
    s = sample_constrained_structures(model, vocab, num_samples=a.samples,
                                      temperature=1.0, top_k=40)
    row = _eval_row(tag, s, ref_c, val_c, vocab, dim, run, gated=False)
    row["val_baseline_nn_iou"] = vbase
    (run / "novelty.json").write_text(json.dumps(row, indent=2))
    _log("gc-houses-large",
         f"{tag}: val_nn={row.get('val_nn_iou')} nn_iou={row['mean_nn_iou']} "
         f"valid={row['validity_rate']} dup={row['duplicate_rate']} "
         f"(refs: phase4 0.405/0.375, raster_constrained 0.383/1.0)")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
