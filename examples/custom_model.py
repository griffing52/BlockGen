"""End-to-end example: define a model, train it, sample, render, evaluate.

Runnable as-is. It is deliberately small (an LSTM, a few epochs, a few samples)
so the whole loop finishes in a couple of minutes on one GPU -- the point is the
*wiring*, not the result. Swap `TinyLSTM` for your architecture and raise the
epochs.

    .venv/bin/python examples/custom_model.py --epochs 3 --samples 8
    .venv/bin/python -m blockgen.eval.bench --arms ar/my_model:<the npz it prints>

Every stage below is the same code the real batteries use, so anything that
works here works in `blockgen/experiments_*.py`.

See docs/custom-model.md for the narrative version.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


# --- 1. The model -----------------------------------------------------------
class TinyLSTM(nn.Module):
    """The whole contract a custom model has to satisfy.

    * ``forward(input_ids, pad_mask=None) -> logits [B, L, vocab_size]``
      `pad_mask` is True at padded positions. Ignoring it is fine for an RNN;
      attention models must not attend across it.
    * ``.vocab_size`` and ``.max_seq_len`` attributes.

    Nothing else. The trainer, the sampler, the constrained decoder and the whole
    evaluation path are representation-agnostic and only ever call the above.
    """

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int = 256,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor,
                pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        h, _ = self.lstm(self.embed(input_ids))
        return self.lm_head(self.norm(h))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-dim", type=int, default=16,
                    help="canonical grid; 16 keeps sequences short for a demo")
    ap.add_argument("--seq", type=int, default=1600)
    ap.add_argument("--limit", type=int, default=400, help="training builds")
    ap.add_argument("--name", default="my_model")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from blockgen.eval.bench import splits
    from blockgen.utils.runs import new_run_dir

    run = new_run_dir(args.name)
    print(f"[run] {run}")

    # --- 2. Data: the canonical split, so numbers compare to every other arm ---
    split = splits.load_split("houses_32", seed=0)
    train = splits.split_structures(split, "train")[:args.limit]
    print(f"[data] {len(train)} training builds "
          f"(split {split.key()}, {split.sizes})")

    # --- 3. Tokenize -------------------------------------------------------
    # The vocab must be built from the TRAINING side only and saved beside the
    # checkpoint: token ids are meaningless without it, and a checkpoint with a
    # missing or mismatched vocab decodes to the wrong blocks.
    from blockgen.utils.serialize import build_block_vocab, save_block_vocab
    from blockgen.training.train_ar import build_sequences

    vocab = build_block_vocab(train, max_dim=args.max_dim)
    save_block_vocab(vocab, str(run / "block_vocab.json"))
    seqs = build_sequences([s.downsample(args.max_dim) for s in train], vocab, args.seq)
    print(f"[tokens] {len(seqs)}/{len(train)} builds fit in {args.seq} tokens; "
          f"vocab={vocab.vocab_size}; median len="
          f"{int(np.median([len(s) for s in seqs])) if seqs else 0}")
    if not seqs:
        raise SystemExit("no sequences fit -- raise --seq or lower --max-dim")

    # --- 4. Train ----------------------------------------------------------
    from blockgen.training.train_ar import ARTrainConfig
    from blockgen.training.train_ar_ext import train_from_sequences

    cfg = ARTrainConfig(max_seq_len=args.seq, epochs=args.epochs, batch_size=8,
                        device=args.device)
    model = TinyLSTM(vocab_size=vocab.vocab_size, max_seq_len=args.seq)
    t0 = time.time()
    model, hist = train_from_sequences(seqs, vocab.vocab_size, cfg, model=model)
    torch.save(model.state_dict(), run / "model.pt")
    print(f"[train] {args.epochs} epochs in {(time.time() - t0) / 60:.1f} min; "
          f"final loss {hist['loss'][-1]:.4f}")

    # --- 5. Sample ---------------------------------------------------------
    from blockgen.training.train_ar import BOS_TOKEN, EOS_TOKEN
    from blockgen.training.train_ar_ext import generate_from_prefix
    from blockgen.utils.serialize import tokens_to_structure

    model.eval()
    samples = []
    for i in range(args.samples):
        toks = generate_from_prefix(model, [BOS_TOKEN], EOS_TOKEN,
                                    max_new_tokens=args.seq - 1,
                                    temperature=1.0, top_k=40)
        try:
            samples.append(tokens_to_structure(toks, vocab))
        except Exception as exc:                       # malformed stream
            print(f"  sample {i}: undecodable ({type(exc).__name__})")
    occ = [int(s.occupied_mask.sum()) for s in samples]
    print(f"[sample] {len(samples)}/{args.samples} decoded; "
          f"median blocks {int(np.median(occ)) if occ else 0}")
    if not samples:
        raise SystemExit("nothing decoded -- train longer before evaluating")

    # --- 6. Render ---------------------------------------------------------
    from blockgen.renderer.grid import save_grid
    try:
        save_grid(samples, str(run / "samples.png"), cols=4, tile_px=256)
        print(f"[render] {run / 'samples.png'}")
    except Exception as exc:                            # headless/EGL is optional
        print(f"[render] skipped ({type(exc).__name__}: {exc})")

    # --- 7. Save for the benchmark ----------------------------------------
    from blockgen.curation.houses import save_house_cache
    path = save_house_cache([s.crop_to_non_air() for s in samples], args.max_dim,
                            cache_dir=str(run), name=args.name,
                            report={"model": "TinyLSTM", "epochs": args.epochs})

    print("\nnext:")
    print(f"  .venv/bin/python -m blockgen.eval.bench --arms ar/{args.name}:{path}")
    print(f"  # add --tier both once the ladder has been run for your backbone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
