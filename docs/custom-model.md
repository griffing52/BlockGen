# Bring your own model

The shared trainer (`blockgen/train/`) takes a new architecture through one
seam: an `Arch` record in `blockgen/train/archs.py`. Register it, select it
with `arch: <name>`, and the loop, the sampler, the benchmark and the
leaderboard are yours. This page is the contract and the five commands.
[Add a model](add-a-model.md) is the same route with every step spelled out,
including a complete example module; [Training pipeline](pipeline.md) §5 is
the trainer reference.

!!! tip "Cheaper option first"
    If the idea is a positional encoding, an embedding or a decode-time
    constraint, you do not need a new module. `VoxelTransformerAR2`
    (`blockgen/models/voxel_transformer_ar2.py`) takes `pe=` variants, and its
    extra knobs (`geo_bias`, `piece_factors`, `semantic_embedding`) reach it
    through the `model:` dict of any config without a code change. Write a new
    module only when the architecture is the idea.

---

## 1. The contract

An `Arch` is four callables and a label:

```python
from blockgen.train.archs import Arch, register, loss_ar2, sample_ar2, constructor_kwargs

def build_mine(cfg, tokenizer):
    from blockgen.models.mine import Mine
    fixed = dict(vocab_size=tokenizer.vocab_size, max_seq_len=cfg.max_seq_len, d_model=cfg.d_model)
    return Mine(**constructor_kwargs(Mine, fixed, dict(cfg.model or {}), "mine"))

register(Arch(name="mine", build=build_mine, loss=loss_ar2, sample=sample_ar2, track="ar"))
```

- `build(cfg, tokenizer)` returns the module. `constructor_kwargs` merges the
  config-derived arguments with `cfg.model` and raises on a key the constructor
  does not take or one that duplicates a config field.
- `loss(model, batch)` receives `(input_ids, targets, pad_mask)` on the device
  and returns the mean loss per non-pad target token. `loss_ar2` is next-token
  cross-entropy with `ignore_index=PAD_TOKEN`; keep it for any token model.
- `sample(model, tokenizer, cfg, n, seed=None)` returns `(structures, extra)`.
  `sample_ar2` is the shared sampler, dispatched on the tokenizer kind.
- `track` is the benchmark track prefix of the arms it produces, `ar` here.

The module itself needs two things and may add a third:

```python
class Mine(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model=256):
        super().__init__()
        self.max_seq_len = max_seq_len                       # (1) required attribute
        ...

    def forward(self, input_ids, pad_mask=None):             # (2) required signature
        ...
        return logits                                        # [B, L, vocab_size]
```

- `input_ids` is `[B, L]` int64, right-padded with `PAD_TOKEN`; `pad_mask` is
  bool `[B, L]`, True at padded positions. An attention model must not attend
  across it. Return logits, not probabilities and not a loss.
- `max_seq_len` is the longest sequence the model accepts. `blockgen/train/data.py`
  drops longer sequences (never truncates) and the sampler caps its budget by it.
- (3) Optional: `forward(step, past=past, return_cache=True) -> (logits, past)`
  for KV-cached decoding. `blockgen/train/sample.py::supports_cache` reads the
  signature and the model's own `supports_cache()`; without it the sampler
  re-runs the whole prefix every step (`Decoder`), slower and otherwise
  identical. `pe=alibi` takes this path.

The model is causal by construction (position `t` predicts `t+1`); the trainer
adds no mask. Put the class in `blockgen/models/<name>.py` and the registration
in `blockgen/train/archs.py`. An unregistered name fails at the trainer's first
line: `KeyError: "unknown arch 'mine'; known: ['ar2']"`.

---

## 2. Data

The shards are already built: the pinned house split as `sft-houses32`
(1,863 train builds) and `eval-holdout` (798 val and test builds), voxel
tokenizer `60684456e598` (507 tokens), facade tokenizers `a9546c1f2f94`
(`per_face`) and `a4ca45ac1fe5` (`joint`, plus the `-d4` sibling). A model on
these tokens needs no data work. `python -m blockgen.corpus.shards show sft-houses32`
says what is on disk; [Data](data.md) §5 and §6 cover rebuilding.

Use the pinned split rather than rolling your own. It is group-aware:
GrabCraft ships sibling builds ("American Middle Class House 10 / 22 / 9"), and
a random row split leaks a family into 35.8% of val (T23 in `results.md`).

---

## 3. Smoke, train

```bash
python -m blockgen.train --config configs/train/smoke.yaml arch=mine
```

60 steps, 16 samples (`max(n_samples, bench_n)`), a fast-tier card on them,
about two minutes.
The run dir `outputs/run_<stamp>_smoke` holds `run.json`, `metrics.jsonl`,
`best.pt`, `last.pt`, `samples.png`, the sample cache, `bench/scorecard.json`
and `summary.json`, and one line went into `outputs/registry.jsonl`.

```bash
python -m blockgen.train --config configs/train/houses32-scratch-matched.yaml \
    arch=mine name=mine_s0 tokenizer=60684456e598
```

Three rules: no dropout change at `init_from` (refused, T59), `select=best`
(the default; the selected checkpoint is what gets sampled and registered),
and pin `tokenizer:` (`shards.find` otherwise takes the newest shard by
mtime). Extra constructor knobs go in `model:` as a YAML mapping or a JSON
object on the command line (`model='{"hidden": 1024}'`).

---

## 4. Sample

```bash
python -m blockgen.train --sample outputs/run_<stamp>_mine_s0 --n 399 [--ckpt best.pt] [--out-dir D]
```

Writes `samples_step399_<dim>.npz` plus `_manifest.json` whose `report`
carries `model`, `arch`, `kind`, `checkpoint {file, step}`, `trained_on_signals`
and `train_split`. The board ranks only `train_split: canonical` (or
`protocol`), which the trainer writes when `train == [sft-houses32]`; a
mixture of clean tiers has to set `train_split=canonical` itself.

Generation uses the KV cache when the model offers one. The fallback is a full
re-forward per step: quadratic in sequence length, so budget for it on a
model without a cache (`sample_max_blocks` caps the tokens per build).

---

## 5. Bench

```bash
# FAST tier: CPU, seconds. Coherence, palette, novelty, geom_kid, the geometry
# scalars and the traversal (player walk) pillar; no rendering.
python -m blockgen.eval.bench --arms ar/mine_s0:<npz> --n 128 --name "mine_s0 fast"

# BOTH tiers: adds the rendered appearance pillar (mv_dino_kid). The official run.
python -m blockgen.eval.bench --tier both --n 399 --baselines --name "mine_s0, n=399" \
    --arms ar/mine_s0:<npz>
```

A fresh clone builds two gitignored caches first, the geometry ladder
(`python -m blockgen.eval.bench.ladder --family geometry --n 64 --n-ref 256`)
and the DINOv2 feature cache plus render ladder
(`python -m blockgen.eval.bench.features --backbone dinov2b`, then
`... ladder --family render --n 64 --n-ref 256`).

Real-data controls are added automatically: `real_test` (the floor),
`real_shuffled_materials`, `real@monochrome`, `real@solidify`,
`real@single_mode` and `train_verbatim` (the memorisation detector). The
decimation rungs `real@canon16` and `real@canon8` live on the ladder, not among
the controls. Three rules for reading the card:

- **n.** 128 is the floor the protocol ranks at (`protocol.py`, `min_n=128`);
  256 or 399 is what the repo quotes. Below 16 the intervals are not
  interpretable and BlockScore marks the arm unranked. Equal n is required for
  any comparison; the board groups by n.
- **Coherence is distance-to-real, not "higher is better."** About 66% of real
  val houses are one connected component. An arm at 1.0 is as far from the
  data as one at 0.2.
- **Novelty sits on the same row as realism.** `train_verbatim` scores KID
  0.015, second only to real data, and is caught by `dino_nn_percentile = 0.000`.

Then look: `python -m tools.lab --open`, `/runs` for the card, `/leaderboard`
for the ranking across cards. The full route, including the human study and
the research record, is [Add a model](add-a-model.md) steps 7 and 8.

---

## Failure modes worth knowing

| Symptom | Cause |
|---|---|
| `KeyError: unknown arch` at start | The `register(Arch(...))` line is missing from `blockgen/train/archs.py`. |
| `unknown model option(s)` at start | A `model:` key the constructor does not take; the message lists the ones it does. |
| `init_from ... does not match the model` | Different arch or preset than the checkpoint; `allow_partial_init=true` if that is intended. |
| `shards disagree on tokenizer` | Mixed tiers built under different hashes; pin `tokenizer:` and rebuild the odd one. |
| `[sample] ... has no KV cache` and sampling is slow | The fallback path. Implement `forward(step, past=, return_cache=True)` or lower `sample_max_blocks`. |
| Arm on the board under *provisional* | Fast tier, n below 128, or `train_split` not `canonical`; the row says which. |
| Great KID, near-zero novelty percentile | The model is reciting its training set. |
| Renderer raises headless | EGL unavailable. `save_grid` falls back to matplotlib; the trainer retries once and never fails a run on it. |

---

## Appendix: the legacy loop (pre-protocol)

Before the shared trainer, a model was trained by handing an instance to
`blockgen.training.train_ar_ext.train_from_sequences`. That loop still exists,
Tracks B, C and F still use their own trainers beside it, and
`examples/custom_model.py` still runs it end to end with a two-layer LSTM:

```bash
PYTHONPATH=. python examples/custom_model.py --epochs 3 --samples 6
```

Its contract is three things: `forward(input_ids, pad_mask=None) -> logits`,
`.vocab_size` and `.max_seq_len` (`tests/test_custom_model.py` pins it).

```python
from blockgen.eval.bench import splits
from blockgen.utils.serialize import build_block_vocab, save_block_vocab, tokens_to_structure
from blockgen.training.train_ar import ARTrainConfig, BOS_TOKEN, EOS_TOKEN, build_sequences
from blockgen.training.train_ar_ext import train_from_sequences, generate_from_prefix

split = splits.load_split("houses_32", seed=0)
train = splits.split_structures(split, "train")
vocab = build_block_vocab(train, max_dim=16)                    # train side only
save_block_vocab(vocab, str(run / "block_vocab.json"))          # a token id means nothing without it
seqs = build_sequences([s.downsample(16) for s in train], vocab, max_seq_len=1600)

cfg = ARTrainConfig(max_seq_len=1600, epochs=60, batch_size=8, device="cuda")
model = TinyLSTM(vocab_size=vocab.vocab_size, max_seq_len=1600)
model, hist = train_from_sequences(seqs, vocab.vocab_size, cfg, model=model)

toks = generate_from_prefix(model.eval(), [BOS_TOKEN], EOS_TOKEN, max_new_tokens=1600, temperature=1.0, top_k=40)
structure = tokens_to_structure(toks, vocab)
```

What differs from the shared trainer, and why the legacy loop is the appendix:

- **A vocabulary per run.** `build_block_vocab` on the train side, saved next
  to the checkpoint. Two vocabs can share a size and disagree on every entry,
  so a mismatch decodes silently to the wrong blocks. The shared trainer has
  one tokenizer per shard, hashed, and refuses a mismatch.
- **Silent drops.** `build_sequences` drops builds over `max_seq_len`; print the
  kept fraction (231 of 400 at 1,600 tokens in the example). The shared trainer
  prints the drop count at start.
- **No cache, no registry.** `generate_from_prefix` re-runs the prefix each
  step; nothing is written to `outputs/registry.jsonl`; `model=` is mutually
  exclusive with `pe=`.
- **Entering the benchmark.** Write the samples with
  `blockgen.curation.houses.save_house_cache([s.crop_to_non_air() for s in samples], 16, cache_dir=str(run), name="my_model")`
  and bench the npz as in section 5. A legacy arm's manifest has no
  `train_split` unless you put one in the `report`, so it stays provisional on
  the board until it does. `scripts/sample_to_npz.py --train-split canonical`
  is the route for a checkpoint served in `deploy/inference/models.json`.
- **Constrained decoding** (`blockgen/training/constrained_decode.py`) masks
  the next voxel to the 6-neighbourhood of placed voxels, connectivity 1.0 by
  construction. Report it as a separate arm: it is a decode-time filter, not
  something the model learned. The shared sampler has no logits-mask hook.
