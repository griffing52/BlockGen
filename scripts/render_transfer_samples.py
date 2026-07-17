"""Textured renders of GENERATED samples from the transfer-run checkpoints.

Reconstructs the exact ``run_20260708_070807_transfer`` setup (houses_32 split +
cross-corpus pool + shared legacy vocab, all deterministic at seed 0), reloads
each saved checkpoint, samples fresh structures with the same plain sampler that
produced each arm's headline val_nn, and renders dense textured grids in the
style of ``outputs/figures/samples_ar_pe_phase4.png``.

Arms rendered (one figure each):
  scratch            -> samples_transfer_scratch.png            (houses-only control)
  pretrain_zeroshot  -> samples_transfer_pretrain_zeroshot.png  (pool ckpt, no finetune)
  finetune           -> samples_transfer_finetune.png           (pool ckpt + houses)

    .venv/bin/python scripts/render_transfer_samples.py --samples 48
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

RUN = REPO / "outputs" / "run_20260708_070807_transfer"
FIGS = REPO / "outputs" / "figures"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--tile-px", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    # Must mirror the transfer run's cmd.txt so the reconstructed vocab is identical.
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--canon-dim", type=int, default=16)
    ap.add_argument("--ar-seq", type=int, default=1600)
    args = ap.parse_args()

    from blockgen.curation.curate import Curator
    from blockgen.experiments_gen import canonicalize
    from blockgen.experiments_transfer import _make_model, _split_houses, load_pool
    from blockgen.renderer.grid import _tile_textured, save_grid
    from blockgen.training.train_ar import ARTrainConfig
    from blockgen.training.train_ar import sample_structures as ar_sample
    from blockgen.utils.augment import augment_with_labels
    from blockgen.utils.serialize import build_block_vocab

    torch.manual_seed(args.seed)
    dim = args.canon_dim
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # --- reconstruct the exact training vocab (seed 0, same pipeline) ---------
    h_train, h_val = _split_houses(args.seed, args.val_frac)
    ref_c = canonicalize(h_train, dim)
    val_c = canonicalize(h_val, dim)
    pool = load_pool(exclude_sources={s.source_path or "" for s in h_val})
    cur = Curator.from_structures(pool, max_dim=dim)
    cur.decisions.clear()
    cur.dedupe_keep_variants(iou_threshold=0.95)
    pool = cur.apply()
    p_aug, _ = augment_with_labels(pool, ["p"] * len(pool))
    pool_c = canonicalize(p_aug, dim)
    vocab = build_block_vocab(pool_c + ref_c + val_c, max_dim=dim)
    print(f"reconstructed vocab: {vocab.vocab_size}", flush=True)

    cfg = ARTrainConfig(max_seq_len=args.ar_seq, device=dev)

    arms = [
        ("scratch", RUN / "scratch" / "model.pt", "samples_transfer_scratch.png"),
        ("pretrain_zeroshot", RUN / "pretrain" / "model.pt",
         "samples_transfer_pretrain_zeroshot.png"),
        ("finetune", RUN / "finetune" / "model.pt", "samples_transfer_finetune.png"),
    ]
    for tag, ckpt, out_name in arms:
        if not ckpt.exists():
            print(f"{tag}: checkpoint missing ({ckpt}); skipping", flush=True)
            continue
        t0 = time.time()
        model = _make_model(vocab.vocab_size, cfg, ckpt).eval()
        s = ar_sample(model, vocab, num_samples=args.samples, temperature=1.0, top_k=40)
        print(f"{tag}: {len(s)} samples in {time.time() - t0:.0f}s", flush=True)
        save_grid(s, str(FIGS / out_name), cols=args.cols, tile_px=args.tile_px,
                  tile_fn=_tile_textured)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    print("DONE-TRANSFER-SAMPLE-GRIDS", flush=True)


if __name__ == "__main__":
    main()
