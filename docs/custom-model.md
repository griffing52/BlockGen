# Bring your own model

End to end: define an architecture, train it, sample from it, render the samples,
and score it against every other track on the same benchmark.

Everything below is in one runnable file — read it alongside this page:

```bash
.venv/bin/python examples/custom_model.py --epochs 3 --samples 6
```

That finishes in about a minute and trains a deliberately **useless** model (a
2-layer LSTM for 3 epochs). It exists to prove the wiring, not to generate
anything good. Real output from that exact command:

```
[data] 400 training builds (split houses_32.s0.70-15-15.v1, {'train': 1863, 'val': 399, 'test': 399})
[tokens] 231/400 builds fit in 1600 tokens; vocab=315; median len=890
[train] 3 epochs in 0.1 min; final loss 2.2499
[sample] 6/6 decoded; median blocks 45
[render] outputs/run_.../samples.png
saved 6 houses -> outputs/run_.../my_model_16.npz
```

!!! tip "Cheaper option first"
    If your idea is a **positional encoding, an embedding, or a decode-time
    constraint**, you do not need a new model. Add a `pe=` variant to
    `VoxelTransformerAR2` (`blockgen/models/voxel_transformer_ar2.py`) or a new
    masker next to `training/constrained_decode.py`. Everything downstream is
    representation-agnostic and you inherit it for free. Write a new module only
    when the *architecture* is the idea.

---

## 1. The contract

A model needs three things. That is the entire interface the trainer, the
sampler, the constrained decoder and the evaluation path use.

```python
class TinyLSTM(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model=256, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size       # (1) required attribute
        self.max_seq_len = max_seq_len     # (2) required attribute
        self.embed = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=num_layers, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, pad_mask=None):   # (3) required signature
        h, _ = self.lstm(self.embed(input_ids))
        return self.lm_head(self.norm(h))          # logits [B, L, vocab_size]
```

- `input_ids` is `[B, L]` int64. `pad_mask` is `[B, L]` bool, **True at padded
  positions**. An RNN can ignore it; an attention model must not attend across
  it, or padding leaks between builds in a batch.
- Return **logits**, not probabilities, and not a loss. The trainer applies
  `cross_entropy` with `ignore_index=PAD_TOKEN`.
- The model is causal by construction here (position `t` predicts `t+1`). If you
  use attention, apply a causal mask yourself — the trainer does not add one.

Put the finished class in `blockgen/models/` next to the others.

---

## 2. Data — use the canonical split

```python
from blockgen.eval.bench import splits

split = splits.load_split("houses_32", seed=0)
train = splits.split_structures(split, "train")
val   = splits.split_structures(split, "val")
```

Use this split rather than rolling your own. It is **group-aware**: GrabCraft
ships sibling builds ("American Middle Class House 10 / 22 / 9") and a random row
split scatters them across train and val. Measured, that leaks a build family
into 35.8% of val, which inflates your held-out scores and deflates every
memorization number simultaneously. See [the benchmark](benchmark.md).

---

## 3. Tokenize — and save the vocab

```python
from blockgen.utils.serialize import build_block_vocab, save_block_vocab
from blockgen.training.train_ar import build_sequences

vocab = build_block_vocab(train, max_dim=16)          # TRAIN side only
save_block_vocab(vocab, str(run / "block_vocab.json"))   # <- do not skip
seqs = build_sequences([s.downsample(16) for s in train], vocab, max_seq_len=1600)
```

Two things that bite:

- **Build the vocab from the training side only.** Including val leaks the block
  palette of builds you are about to be scored on.
- **Save it next to the checkpoint.** A token id means nothing without the vocab
  that produced it, and two vocabs can share a size while disagreeing on every
  entry — so a mismatch decodes silently to the wrong blocks rather than
  crashing. This has already cost the project a run
  (`scripts/rebuild_native_vocab.py` exists because one checkpoint shipped
  without its vocab).

`build_sequences` silently drops builds longer than `max_seq_len`. Print the
kept fraction — in the example above only **231 of 400** builds fit in 1600
tokens, and a 42% drop rate changes what you are modelling.

---

## 4. Train

Pass your instance to the stock loop; you inherit AMP, gradient clipping,
padding and the loss.

```python
from blockgen.training.train_ar import ARTrainConfig
from blockgen.training.train_ar_ext import train_from_sequences

cfg = ARTrainConfig(max_seq_len=1600, epochs=60, batch_size=8, device="cuda")
model = TinyLSTM(vocab_size=vocab.vocab_size, max_seq_len=1600)
model, hist = train_from_sequences(seqs, vocab.vocab_size, cfg, model=model)
torch.save(model.state_dict(), run / "model.pt")
```

`model=` is mutually exclusive with `pe=` / `semantic_embedding=`, which are
instructions for building the *stock* model. It cross-checks `model.vocab_size`
against the vocab the sequences were built from and refuses a mismatch.

Write outputs with `utils.runs.new_run_dir(name)` → `outputs/run_<stamp>_<name>/`,
so runs sort newest-first and nothing collides.

---

## 5. Sample

```python
from blockgen.training.train_ar import BOS_TOKEN, EOS_TOKEN
from blockgen.training.train_ar_ext import generate_from_prefix
from blockgen.utils.serialize import tokens_to_structure

model.eval()
toks = generate_from_prefix(model, [BOS_TOKEN], EOS_TOKEN,
                            max_new_tokens=1600, temperature=1.0, top_k=40)
structure = tokens_to_structure(toks, vocab)
```

`tokens_to_structure` raises on a malformed stream, which is normal for an
undertrained model — catch it per sample and report the decode rate rather than
letting one bad stream kill the run.

Generation has **no KV cache** (`generate_from_prefix` re-runs the whole prefix
each step), so cost is quadratic in sequence length. The native BPE arm is ~10 s
per build at a ~1750-token median. Budget for it: 64 samples took 10.4 minutes.

Optional: `training/constrained_decode.py` masks the next-voxel logits to the
6-neighbourhood of already-placed voxels, giving connectivity 1.0 by
construction. Report it as a **separate arm** — it is a decode-time filter, not
something the model learned, and mixing the two makes the gain unattributable.

---

## 6. Render

```python
from blockgen.renderer.grid import save_grid
save_grid(samples, str(run / "samples.png"), cols=4, tile_px=256)
```

Real Minecraft textures via headless EGL/pyrender, falling back to matplotlib
voxels if EGL is unavailable. **Render samples for every run** — the project has
three recorded cases where a metric moved the wrong way and only the pictures
caught it (T17/T20).

For one structure at a chosen angle:

```python
from blockgen.renderer.textured import render_structure
from blockgen.renderer.textures import load_face_textures

img = render_structure(s, px=512, azim_deg=45, elev_deg=30,
                       face_textures=load_face_textures())   # (512,512,4) uint8
```

Wrap rendering in `try/except` inside a training battery — a headless failure
should not lose you a finished model. Contact sheets of the *corpus* come from
`python -m blockgen.renderer.grid --houses 32 --rows 8 --cols 12`.

---

## 7. Evaluate

The benchmark reads **structure caches only** — it never imports your model — so
first write one:

```python
from blockgen.curation.houses import save_house_cache
path = save_house_cache([s.crop_to_non_air() for s in samples], 16,
                        cache_dir=str(run), name="my_model")
```

Then score it against the same reference every other track uses:

```bash
# FAST tier: coherence, palette, novelty, dimensions. CPU, seconds.
.venv/bin/python -m blockgen.eval.bench --arms ar/my_model:outputs/run_.../my_model_16.npz

# FULL tier: adds MV-DINO-KID, density/coverage, DINO memorization. Needs a
# warm feature cache (~4 min once) and a ladder run for your backbone.
.venv/bin/python -m blockgen.eval.bench.features --backbone dinov2b
.venv/bin/python -m blockgen.eval.bench.ladder --n 64
.venv/bin/python -m blockgen.eval.bench --tier both --n 256 --arms ar/my_model:<npz>
```

Real-data controls (`real_test`, `real@canon16`, `train_verbatim`, …) are added
automatically, and you want them: they are the floor and the known-damage rungs
your numbers are read against. Then:

```bash
.venv/bin/python scripts/bench_report.py --run outputs/run_<stamp>_bench
```

Reading the result — see [the benchmark page](benchmark.md) for the full rules,
but three that catch people out:

- **`n = 16` cannot rank arms.** KID's noise floor there is 0.04 against a
  real→canon-16 gap of 0.18. Use n ≥ 256 for anything you will quote; n = 64 is
  the reportable floor.
- **Coherence is distance-to-real, not "higher is better."** Only ~66% of real
  val houses are a single connected component. An arm at 1.0 is as far from the
  data as one at 0.2.
- **Check novelty on the same row as realism.** A model that memorizes scores
  near-perfect realism: `train_verbatim` (literal training copies) reads KID
  0.015, second only to real data, and is caught *only* by
  `dino_nn_percentile = 0.000`.

Already have a checkpoint served in `deploy/inference/models.json`? Skip steps
5–7 and sample straight into the benchmark format:

```bash
.venv/bin/python scripts/sample_to_npz.py --model native_oriented --n 256 \
    --out-dir outputs/bench_arms
```

---

## 8. Make it permanent

- **Model** → `blockgen/models/your_model.py`.
- **Battery** → a `blockgen/experiments_<name>.py` following the pattern in
  [Architecture](architecture.md#writing-your-own-experiment): guard each arm with
  `try/except` → `ERROR.txt`/`SKIP.txt` and continue, write `novelty.json` per arm
  (its presence is the resume marker), append to `leaderboard.md`.
- **Config** → `configs/experiments/<name>.yaml`, invoked as `--config <name>`.
- **Serving** (optional) → add a `kind` to
  `deploy/inference/blockgen_server/backends.py` and an entry to `models.json`
  naming the checkpoint, its vocab, and the architecture constants. The loader
  cross-checks vocab size against `lm_head`, which catches a swapped file but not
  a same-size mismatch — so name the right vocab.
- **Record the run** in `notes.md` and `results.md`. Those are the living research
  record; these docs summarize them and do not replace them.

---

## Failure modes worth knowing

| Symptom | Cause |
|---|---|
| Samples decode to the wrong blocks | Vocab does not match the checkpoint. Rebuild from the run's `block_vocab.json` / `piece_vocab.json`. |
| `tokens_to_structure` raises constantly | Undertrained model emitting malformed streams. Expected early; report the decode rate. |
| Far fewer sequences than builds | `build_sequences` dropped everything over `max_seq_len`. Raise `--seq` or lower `--max-dim`. |
| Validity 1.0 and suspiciously good | Constrained decoding is on. That is a filter, not learning — report it as its own arm. |
| Benchmark refuses a metric | The ladder rejected it for your backbone/view. Working as intended; see `outputs/analysis/ladder/`. |
| Great KID, near-zero novelty percentile | The model is reciting its training set. |
| Renderer raises headless | EGL unavailable. `save_grid` falls back to matplotlib; never let it kill a training run. |
