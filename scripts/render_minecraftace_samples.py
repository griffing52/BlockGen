"""Decode, render and evaluate MinecraftACE samples.

Loads the raw ``sample-*.npy`` record files written by
``libs/MinecraftACE/inference/infer_uncondition_minecraft.py``, decodes them to
``Structure``s, renders a textured mosaic next to a real-data reference grid,
and reports novelty/validity against the houses cache.

Usage:
    python scripts/render_minecraftace_samples.py \
        --samples-dir outputs/minecraftace/uncond-samples \
        --out outputs/figures/minecraftace
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from blockgen.curation.houses import load_house_structures
from blockgen.eval.novelty import evaluate_novelty
from blockgen.eval.validity import n_components
from blockgen.export.minecraftace import (load_piece_vocab, piece_records_to_structure,
                                          records_to_structure, vocab_from_dat_dict)
from blockgen.renderer.grid import save_grid


def load_sample_structures(samples_dir: str, max_dim: int = 32):
    samples_dir = Path(samples_dir)
    dat_dict_path = samples_dir / "dat_dict.json"
    if not dat_dict_path.exists():
        raise FileNotFoundError(f"{dat_dict_path} missing (copied by the inference script)")

    piece_vocab_path = samples_dir / "piece_vocab.json"
    if piece_vocab_path.exists():
        # 3D-BPE arm: records are (anchor, piece_id); expand learned patterns.
        # Eval needs a block-level vocab — rebuild from the atomic pieces.
        from blockgen.utils.data import _token_for
        from blockgen.utils.serialize import BlockVocab
        cv = load_piece_vocab(str(piece_vocab_path))
        decode = lambda rec: piece_records_to_structure(rec, cv)
        tokens = [_token_for(*p) for p in cv.block_index_to_pair]
        vocab = BlockVocab(max_dim=max_dim,
                           block_token_to_id={t: i for i, t in enumerate(tokens)},
                           id_to_block_token=tokens,
                           block_index_to_pair=list(cv.block_index_to_pair))
    else:
        # picks up the sibling pair_dict.json for exact (id, data) decoding
        vocab = vocab_from_dat_dict(str(dat_dict_path), max_dim=max_dim)
        decode = lambda rec: records_to_structure(rec, vocab)

    structures = []
    for path in sorted(samples_dir.glob("sample-*.npy")):
        structures.append(decode(np.load(path)))
    return structures, vocab


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument("--out", default="outputs/figures/minecraftace")
    parser.add_argument("--max-dim", type=int, default=32)
    parser.add_argument("--grid-cols", type=int, default=8)
    parser.add_argument("--num-ref", type=int, default=32,
                        help="number of real houses in the reference grid")
    parser.add_argument("--eval-grid", type=int, default=32)
    parser.add_argument("--tag", default="uncond",
                        help="filename prefix for the output figures")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, vocab = load_sample_structures(args.samples_dir, max_dim=args.max_dim)
    print(f"loaded {len(samples)} samples")

    houses, _ = load_house_structures(max_dim=args.max_dim)

    sample_grid = out_dir / f"{args.tag}_samples.png"
    ref_grid = out_dir / f"{args.tag}_real_ref.png"
    save_grid(samples, str(sample_grid), cols=args.grid_cols)
    save_grid(houses[: args.num_ref], str(ref_grid), cols=args.grid_cols)
    print(f"grids -> {sample_grid}, {ref_grid}")

    report = evaluate_novelty(samples, houses, vocab, grid=args.eval_grid)
    comps = [n_components(s) for s in samples]
    blocks = [int(s.occupied_mask.sum()) for s in samples]

    summary = {
        "n_samples": len(samples),
        "mean_nn_iou": round(report.mean_nn_iou, 4),
        "duplicate_rate": round(report.duplicate_rate, 4),
        "diversity": round(report.diversity, 4),
        "validity_rate_occupancy": round(report.validity_rate, 4),
        "block_agreement_mean": round(float(np.mean(report.block_agreement)), 4),
        "connected_rate": round(float(np.mean([c == 1 for c in comps])), 4),
        "components_median": int(np.median(comps)),
        "blocks_median": int(np.median(blocks)),
        "blocks_min_max": [int(min(blocks)), int(max(blocks))],
    }
    print(json.dumps(summary, indent=2))
    with open(out_dir / f"{args.tag}_eval.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
