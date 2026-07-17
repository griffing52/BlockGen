# Reproducing T18 (`native_bpe` / `canon16_flat`)

The exact command behind `outputs/run_20260715_065938_native`, which is what the
Minecraft demo serves (`deploy/inference/models.json`).

## The command

```bash
cd /path/to/BlockGen
python -m blockgen.experiments_native --epochs 60
```

That is the whole thing. Every other value in the run's `cmd.txt` is a default:
`samples 16, batch 8, merges 256, seq_flat 1600, seq_bpe 5480, seed 0,
val_frac 0.15, arms [flat, bpe]`. Only `--epochs` was changed (the default is 90).

Explicitly, if you prefer nothing implicit:

```bash
python -m blockgen.experiments_native \
    --epochs 60 --samples 16 --batch 8 --merges 256 \
    --seq-flat 1600 --seq-bpe 5480 --seed 0 --val-frac 0.15 \
    --arms flat bpe
```

Both arms come from this one command; they are two arms of one experiment and are
only comparable because they share the split, augmentation, epochs and sampler.

* **Cost**: ~4 h on one GPU — arm A (`canon16_flat`) 29.7 min, arm B (`native_bpe`)
  209.9 min, plus ~7 min of BPE learning and the novelty eval.
* **Output**: `outputs/run_<stamp>_native/` with `cmd.txt`, `rows.json`, and a
  directory per arm (`model.pt`, vocab, `samples.png`, `comparison.png`,
  `novelty.json`).
* **Smoke test first** (~5 min, 2 epochs / 24 merges — trains a *useless* model, only
  checks the pipeline runs):

  ```bash
  python -m blockgen.experiments_native --quick
  ```

  Note `--quick` writes to the same `outputs/run_<stamp>_native/` layout with
  identically-shaped checkpoints. `run_20260715_062404_native` is one of these; it is
  a 2-epoch smoke, not a result. Pass `--stamp` to label such runs.

## Determinism

Everything upstream of training is seeded, so the data and vocabulary reproduce
bit-for-bit: `_split_houses(seed, val_frac)` uses a seeded permutation,
`augment_with_labels` enumerates the full D4 orbit (no sampling), and
`learn_clusters` seeds its own 400-build subsample and breaks merge ties by insertion
order. That is what made the lost vocabularies recoverable (notes.md §18).

Training itself is *not* bit-reproducible: `torch.manual_seed(args.seed)` is set, but
the DataLoader shuffles and bf16 autocast (`ARTrainConfig.amp`) is on. Expect the
headline numbers to move by ~0.01. Set `amp=False` to reproduce pre-2026-07-15 runs
exactly.

## What it produced

| metric | native_bpe (arm B) | canon16_flat (arm A) |
|---|---|---|
| val_nn_iou / baseline | 0.304 / 0.403 = **75%** | 0.368 / 0.442 = **83%** |
| validity / train validity | 0.188 / 0.595 = **31.5%** | 0.312 / 0.427 = **73%** |
| duplicate rate | 0.0 | 0.0 |
| diversity | 0.897 | 0.865 |
| median blocks | 942 | 217 |
| final loss | 0.287 | 0.174 |
| train time | 209.9 min | 29.7 min |
| params | 5,456,841 | 5,065,907 |

Compare **ratios, not raw nn_iou**: the arms score on different grids (32³ vs 16³) and
those numbers are not commensurable. `val_baseline_nn_iou` — how close a real held-out
build is to train at that grid — is the per-arm normalizer.

## Pipeline

`main()` (line 128) → `arm_flat` (line 79) / `arm_bpe` (line 100), all in
`blockgen/experiments_native.py`.

1. `load_house_structures(max_dim=32)` — `blockgen/curation/houses.py`
2. `_split_houses(seed=0, val_frac=0.15)` — `blockgen/experiments_transfer.py:117`
3. `augment_with_labels(h_train, ...)` — `blockgen/utils/augment.py:61`, D4 orbit,
   deduped, **train only**
4. `canonicalize(h_aug, 32)` — at dim 32 this crops but never downsamples (that is the
   point of arm B; T17 showed canon-16 destroys 86.5% of blocks)
5. `learn_clusters(fit, max_dim=32, n_merges=256, max_corpus=400)` —
   `blockgen/tokenizers/cluster_bpe.py:171`. 422 atomic block classes + 256 merges =
   **678 pieces**; `vocab_size = 3 + 32 + 678 = 713`
6. `build_cluster_sequences(fit, cv, 5480)` → 17,163 sequences, median length 1,638
7. `train_from_sequences(seqs, 713, cfg, pe="phase4")` —
   `blockgen/training/train_ar_ext.py:26`

Tokens are `[BOS, (X,Y,Z,PIECE)*, EOS]`; PAD=0, BOS=1, EOS=2, coords `3..34` (0-based),
pieces `35..712`; pieces emitted in `(y, z, x)` anchor order (`cluster_bpe.py:237`).

Model: `VoxelTransformerAR2`, `pe="phase4"`, d_model 256 / 8 heads / 6 layers / ff
1024 / dropout 0.1. Optimizer AdamW at a **flat 3e-4 (no schedule)**, batch 8, grad
clip 1.0, bf16 autocast, cross-entropy with `ignore_index=PAD`.

## Before you launch a variant

* **The vocab is now saved automatically** (`piece_vocab.json` / `block_vocab.json`,
  written before training so a killed run still leaves one). Runs before 2026-07-16
  do not have it; recover with `deploy/inference/scripts/rebuild_native_vocab.py`.
* **Serve it** by adding a `deploy/inference/models.json` entry pointing at the
  checkpoint and its vocab — no code changes if the `kind` already exists.
* **Unused levers** (see notes.md §9): no LR schedule at all; factored piece
  embeddings (`VoxelTransformerAR2(piece_factors=..., piece_offset=...)`) are wired
  but off; constrained decoding cannot gate `ClusterVocab` piece tokens yet, and
  validity (0.188) is arm B's weakest number; `merges=256` and `max_corpus=400` (BPE
  stats come from only 400 of the 17k builds) are both untuned.
