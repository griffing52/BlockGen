# Add a model, end to end

One walkthrough from an empty editor to a row on the leaderboard: environment,
data, a new architecture in `blockgen/models/`, its registration in
`blockgen/train/archs.py`, the smoke run, the real run, sampling, the official
benchmark, the board, the human study, and the research record. Every command
is the one the code runs today. [Training pipeline](pipeline.md) §5 is the
reference for the trainer; this page is the route.

The example architecture below (`TinyMLP`) is deliberately small. It trained
one smoke run and sampled through the shared sampler as written here on
2026-10-31; nothing else about it is interesting.

---

## 0. Environment and tests

Use the conda `python` on `PATH` (Python 3.13, torch 2.7 + CUDA 12.8,
pandas 3.0). The project `.venv/` has no pandas, and `blockgen.corpus` imports
it, so the corpus and trainer CLIs do not run from `.venv/bin/python`.

```bash
python -c "import torch, pandas; print(torch.__version__, torch.cuda.is_available(), pandas.__version__)"
PYTHONPATH=. python -m pytest tests/corpus tests/train -q
```

61 tests. One of them, `tests/train/test_train.py::test_smoke_run_writes_the_run_artifacts`,
is marked `slow`: it trains `configs/train/smoke.yaml` on the GPU for about
two minutes. On a shared GPU run the rest with `-m "not slow"` (60 tests, two
seconds) and let step 3 below be the smoke run.

---

## 1. Data: tiers and shards

A training run reads shards. A shard is a tier (an id list with a sha256,
`data/minecraft/corpus/tiers/<name>.json`) pushed through a tokenizer into
`data/minecraft/corpus/shards/<tier>/<tokenizer hash>[-<augment>]/`
(`blockgen/corpus/shards.py`). The pinned house split is two tiers:
`sft-houses32` (the 1,863 train builds) and `eval-holdout` (the 798 val and
test builds); every SFT run trains on the first and validates on the second.
[Data](data.md) §5 and §6 explain the layer; this is what a fresh clone runs.

```bash
# Build the two split-backed tiers (needs the houses_32 cache in the manifest).
python -m blockgen.corpus.tiers build sft-houses32 eval-holdout

# Voxel shards for both. The house tiers are whole builds, hence --max-blocks 8192.
python -m blockgen.corpus.shards build sft-houses32 --max-blocks 8192
python -m blockgen.corpus.shards build eval-holdout --max-blocks 8192

# Check what landed.
python -m blockgen.corpus.tiers show sft-houses32
python -m blockgen.corpus.shards show sft-houses32
```

What to expect on disk:

| Object | Hash | Size |
|---|---|---|
| tier `sft-houses32` | `cd526aaaeeb6...` | 1,863 builds |
| tier `eval-holdout` | `0f23acfed16f...` | 798 builds |
| voxel shard (507 tokens) | `60684456e598` | 1,863 and 798 sequences |
| facade shard, `per_face` layout (514 tokens) | `a9546c1f2f94` | 12,684 and 5,172 sequences |
| facade shard, `joint` layout (514 tokens) | `a4ca45ac1fe5` | 2,114 and 862 sequences |
| facade `joint` with D4 augmentation (train only) | `a4ca45ac1fe5-d4` | 16,880 sequences |

The tokenizer hash is what a config pins with `tokenizer:`; the three facade
variants come from `--kind facades [--layout per_face|joint] [--augment d4]`.
If `tiers build` raises `EmptyTierError`, the `houses_32` cache is missing from
the manifest: run
`python -m blockgen.corpus.manifest curated --caches all_32 houses_32 houses_48`
and build the tiers again. `scripts/corpus_refresh.sh --dry-run` prints the
whole manifest-to-shards order. A shard opened after its tier was rebuilt
warns `StaleShardWarning`; rebuild the shard.

A model on the existing tokens needs nothing more. A new token kind is a
different project: it touches `blockgen/corpus/tokenizer.py` (`KINDS`),
`shards.py`, `blockgen/train/config.py`, `trainer.py` and `sample.py`.

---

## 2. Model: the file, the contract, the registration

### The contract

`blockgen/train/archs.py` is the only seam. The step loop knows a model through
one `Arch` record, `Arch(name, build, loss, sample, track)`:

- `build(cfg, tokenizer) -> nn.Module`
- `loss(model, batch) -> Tensor`, where `batch` is `(input_ids, targets, pad_mask)`
  already on the device, and the result is the mean loss per non-pad target
  token (`loss_ar2` is next-token cross-entropy with `ignore_index=PAD_TOKEN`)
- `sample(model, tokenizer, cfg, n, seed=None) -> (List[Structure], extra)`
  (`sample_ar2` is the shared next-token sampler)
- `track`: the benchmark track prefix of its arms, `ar` for every autoregressive family

The module that `build` returns must satisfy, from the `archs.py` docstring:

- `model(input_ids, pad_mask=None) -> logits`. `input_ids` is `[B, L]` int64,
  right-padded with `PAD_TOKEN`; `pad_mask` is bool `[B, L]`, True where padded;
  `logits` is `[B, L, vocab_size]`. Training and validation call only this form.
- `model.max_seq_len`: the longest sequence it accepts. `blockgen/train/data.py`
  drops longer sequences (it never truncates) and the sampler caps its token
  budget by it.
- Optional, for fast sampling: `forward(step, past=past, return_cache=True) -> (logits, past)`.
  `blockgen/train/sample.py::supports_cache` reads the signature and the model's
  own `supports_cache()`; a model without the keywords is driven by a full
  re-forward per step (`Decoder`), slower and otherwise identical.

The model is causal by construction: position `t` predicts `t+1`, and the
trainer adds no mask. An attention model masks itself and must not attend
across `pad_mask`.

### The file

`blockgen/models/tiny_mlp.py`, complete. Each position sees the running mean of
the embeddings up to itself, so the model is causal with no attention and no
positional encoding. It implements only `forward(input_ids, pad_mask=None)`
and relies on the sampler's cache-optional fallback.

```python
"""A minimal architecture for the shared trainer: the smallest module that honours
the training half of the model contract (blockgen/train/archs.py).
"""
from __future__ import annotations

import torch
from torch import nn


class TinyMLP(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int = 256, hidden: int = 512):
        super().__init__()
        self.max_seq_len = max_seq_len            # the batchers and the sampler read this
        self.embed = nn.Embedding(vocab_size, d_model)
        self.mlp = nn.Sequential(nn.Linear(2 * d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embed(input_ids)                                   # [B, L, d]
        if pad_mask is not None:
            x = x.masked_fill(pad_mask[..., None], 0.0)             # padding adds nothing to the prefix
        pos = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype)[None, :, None]
        prefix = x.cumsum(dim=1) / pos                              # causal: position t sees tokens <= t
        h = self.norm(x + self.mlp(torch.cat([x, prefix], dim=-1)))
        return self.head(h)                                         # logits [B, L, vocab_size]
```

### The registration

Add a `build_` function and one `register` line to `blockgen/train/archs.py`,
next to `build_ar2`. The import of the model class stays inside `build`, as it
does for `ar2`, so importing the registry stays cheap.

```python
def build_tiny_mlp(cfg: TrainConfig, tokenizer: CorpusTokenizer) -> nn.Module:
    from blockgen.models.tiny_mlp import TinyMLP
    fixed = dict(vocab_size=tokenizer.vocab_size, max_seq_len=cfg.max_seq_len, d_model=cfg.d_model)
    return TinyMLP(**constructor_kwargs(TinyMLP, fixed, dict(cfg.model or {}), "tiny_mlp"))


register(Arch(name="tiny_mlp", build=build_tiny_mlp, loss=loss_ar2, sample=sample_ar2, track="ar"))
```

`arch: tiny_mlp` in a YAML (or `arch=tiny_mlp` on the command line) now
selects it. Before the registration exists, the trainer stops at its first
line, before any shard is opened:

```
KeyError: "unknown arch 'tiny_mlp'; known: ['ar2']"
```

### Extra knobs: the `model:` dict

`TrainConfig.model` is a dict that `constructor_kwargs` merges into the
constructor call. `hidden` above is such a knob:

```bash
python -m blockgen.train --config configs/train/smoke.yaml arch=tiny_mlp model='{"hidden": 1024}'
```

A key the constructor does not take raises with the list it does take; a key
that duplicates a config field (`d_model`, `max_seq_len`) raises and tells you
to set it at the top level. Both happen before any weights are allocated. On
the command line the value must be a JSON object. In a YAML it is a mapping:

```yaml
arch: tiny_mlp
model:
  hidden: 1024
```

---

## 3. Smoke

```bash
python -m blockgen.train --config configs/train/smoke.yaml arch=tiny_mlp
```

`smoke.yaml` is 60 steps, an eval every 30, then 16 samples at the end: the
trainer draws `max(n_samples, bench_n)` builds (the end-of-run
`sample_and_render` call in `blockgen/train/trainer.py`), scores the fast tier
on them and renders all 16 into
`samples.png`. With the GPU it takes about two minutes (`[bench]` is 24 of
those seconds). With `device=cpu bench=false` the same command draws only
`n_samples` (four) and skips the card; the example above trained and sampled
that way in 25 seconds. The run directory is
`outputs/run_<YYYYMMDD_HHMMSS>_smoke` and must contain:

```
run.json                       config, meta (tokenizer hash, tier hashes, git sha, arch, track), params
metrics.jsonl                  one line per log / eval event
best.pt  last.pt  ckpt_0000030.pt
samples.png                    the 16 builds, rendered (every run renders its samples)
samples_step60_48_manifest.json  samples_step60_48.npz      the structure cache the bench scored
bench/scorecard.json  bench/scorecard.md  bench/examples_28.npz  bench.log
summary.json                   the registry record; one more line in outputs/registry.jsonl
```

Three lines in the log are the check. `[train] ... 0.7M params (tiny_mlp), vocab 507`
says the arch was built from the shard tokenizer. `[sample] TinyMLP has no KV cache; sampling by full re-forward per step`
says the fallback engaged. `[eval] all holdout (step 60) eval-holdout ... (n=736)`
says the selected checkpoint was scored on every holdout sequence that fits
the 8,200-token context. The smoke card is a progress number only: a fast-tier
card at n=16 appears on the board under *skipped* with the reasons
"tier fast measures no appearance pillar" and "real_test calibrated on 16 builds".

---

## 4. Train

Two starting points on the house split, both early-stopped on holdout NLL:

```bash
# From a pretrained checkpoint (ar2 only: the M2 pretraining runs are ar2 at dropout 0).
python -m blockgen.train --config configs/train/houses32-sft-5m.yaml \
    name=tiny_mlp_sft_s0 init_from=outputs/run_20261014_125922_m2_pretrain_s0/last.pt

# From scratch, compute-matched to the pretrain + SFT pair (36,000 steps).
python -m blockgen.train --config configs/train/houses32-scratch-matched.yaml \
    arch=tiny_mlp name=tiny_mlp_scratch_s0 tokenizer=60684456e598
```

A new architecture has no pretrained checkpoint, so the first real run is the
scratch one. `init_from` loads weights only: its keys must match the model
(`allow_partial_init=true` accepts a partial match and records the counts), it
must have been trained under the same tokenizer hash, and it refuses a dropout
change. Three rules the configs encode and the code enforces:

1. **No dropout change at `init_from`.** `trainer.check_init_compat` refuses
   it. A 5M model pretrained at dropout 0 and fine-tuned at 0.1 lost its
   pretraining before the first update (T59). `houses32-sft-5m.yaml` sets
   `dropout: 0.0` for this reason; `allow_dropout_change=true` is the override
   for when you mean it.
2. **`select=best`.** The default. Every `eval_every` steps the first
   `eval_seqs` holdout sequences are scored and `best.pt` follows the first
   `val` tier; at the end the selected weights are what gets sampled,
   benchmarked and registered (`selected_ckpt` in the record). `last.pt` keeps
   the final weights. `select=last` is the matched-compute comparison without
   early stopping.
3. **Pin `tokenizer:`.** `shards.find` without a hash picks the newest shard
   of the kind by mtime. Every shard a run mixes must share one hash
   (`data.open_shards`), and `init_from` compares the checkpoint's hash with
   it. The voxel hash is `60684456e598`; the facade configs pin theirs.

Other things worth knowing before a long run: `resume=<run_dir>` restores the
full state from `last.pt` and does not reload the config; `train: [a:2, b:1]`
mixes tiers by weight; sequences over `max_seq_len` are dropped, and the
`[data]` line at start says how many (238 of the 1,863 houses at 8,200).
Facade models use the same loop with `kind: facades` and their own shard
hash (`configs/train/houses32-facades*.yaml`).

---

## 5. Sample

The end-of-run cache has `max(n_samples, bench_n)` builds. The board wants more,
and every board entry so far came from this route:

```bash
python -m blockgen.train --sample outputs/run_<stamp>_tiny_mlp_scratch_s0 --n 399 \
    [--ckpt best.pt] [--out-dir outputs/run_<stamp>_tiny_mlp_scratch_s0/samples399] \
    [temperature=1.0 top_k=40 seed=0]
```

It loads `best.pt` if it exists, else `last.pt`, applies the `key=value`
overrides with the YAML coercion, and writes `samples_step399_<dim>.npz` plus
`_manifest.json` (and a `samples.png`) under `--out-dir`, default the run dir.
The cache stem carries the sample count, so it never overwrites the trainer's
own `samples_step<step>` cache. The manifest `report` is what the board reads:

```json
{"model": "tiny_mlp_scratch_s0", "arch": "tiny_mlp", "kind": "voxels",
 "checkpoint": {"file": "best.pt", "step": <selected step>}, "trained_on_signals": [],
 "train_split": "canonical", "n": 399, "seed": 0, "temperature": 1.0, "top_k": 40, ...}
```

**`report.train_split` must be `canonical`** (or `protocol`;
`protocol.CLEAN_TRAIN_SPLITS`). The trainer writes `cfg.split_name()`:
`canonical` when `train == [sft-houses32]`, otherwise the joined tier list,
unless the run set `train_split` explicitly. A mixture run whose tiers all
derive from the protocol's train split (an RFT tier of the model's own samples
plus `sft-houses32`) is clean and must say so: `train_split=canonical` on its
command line. A run on `anneal` or `pretrain-*` is not clean and stays
provisional, which is the board doing its job.

---

## 6. Bench

```bash
python -m blockgen.eval.bench --tier both --n 399 --baselines \
    --name "tiny_mlp scratch s0, n=399" \
    --arms ar/tiny_mlp_scratch_s0:outputs/run_<stamp>_tiny_mlp_scratch_s0/samples399/samples_step399_48.npz
```

The arm name after `ar/` is the model identity the board collapses runs under;
use the run name. `--n 399` sizes the controls to the whole test split; the
file-backed arm is as big as its npz, so sample 399 too. `--baselines` adds
the procedural floor. The run writes `outputs/run_<stamp>_bench_<slug>/{scorecard.json, scorecard.md, examples_<dim>.npz}`
and prints the lab URL.

A fresh clone has to build two gitignored caches before its first `--tier both`
run (`.gitignore` ignores `outputs/` and `data/minecraft/features`):

```bash
# 1. The geometry ladder: outputs/analysis/ladder/geom__patch2_orb55_t5_h8_s12.json.
#    Without it the fast tier scores geom_kid as skipped ("no geometry ladder").
python -m blockgen.eval.bench.ladder --family geometry --n 64 --n-ref 256

# 2. The DINOv2 feature cache under data/minecraft/features/houses_32/ and the
#    render ladder (dinov2b__...json). The full tier raises without the ladder.
python -m blockgen.eval.bench.features --backbone dinov2b
python -m blockgen.eval.bench.ladder --family render --n 64 --n-ref 256
```

What *ranked* requires under `houses32-v1` (`blockgen/eval/bench/protocol.py`):
corpus `houses_32`, split `houses_32.s0.70-15-15.v1`, seed 0, tier `both`,
the `patch2_orb55_t5_h8_s12` descriptor, `real_test` and the arm at n >= 128,
a real reference of at least 256, and a clean `report.train_split`. Any miss
puts the arm under *provisional* with the reason. 128 is the floor; 256 or 399
is what the repo quotes ([Evaluation suite](benchmark.md), *Sample sizes*).
BlockScore is only comparable at equal n, and the board groups by n rather
than mixing them. The first trainer arm ranked this way is `m2b_sft_s1` at
n=256 (`outputs/run_20261031_031101_bench_voxel_post-training_best-of-4_and`).

---

## 7. Board and humans

```bash
python -m tools.lab --open          # http://127.0.0.1:8765
```

`/leaderboard` ranks every card on disk that matches the protocol, one row per
model, with chips for each distinct n and a banner when a view mixes them;
`/leaderboard?n=399` and `/api/standings?n=399` bookmark one group. `/runs`
shows one card with the builds beside its numbers. The lab discovers both
`outputs/*/scorecard.json` and the trainer's nested
`outputs/*/bench/scorecard.json` (card id `<run>/bench`); a lab started before
2026-10-31 needs a restart to see the nested cards and the n groups.

The human study runs on the phone queue. `python -m tools.lab --phone` binds
every interface and prints the `/phone` URL (no auth: only on a network you
trust). Which arms it serves is pinned by id in `configs/lab/phone.json`, not
globbed, because `outputs/run_*` is full of smoke runs. Add the arm to the
`compare` pools when its bench run is the one being written up. The id is the
npz path under `outputs/` with `/` as `__` and no suffix
(`tools/lab/catalog.py::_arm_id`):

```json
"arm:run_<stamp>_tiny_mlp_scratch_s0__samples399__samples_step399_48"
```

The id also appears in the lab's dataset picker and in `/api/datasets`.
Pairs land in `outputs/lab/lab.db`; `scripts/human_study_analyze.py` fits the
Bradley-Terry table.

---

## 8. Record

Three places, in the same session as the run.

1. **Registry.** The trainer appended one line to `outputs/registry.jsonl`
   (also `summary.json` in the run dir): `time, run, name, stage, arch, kind,
   layout, augment, preset, params, seed, dropout, lr, pe, model, steps,
   tokens, minutes, train, val, train_split, tokenizer, tiers, git, config,
   init_from, parent_run, init_missing, init_unexpected, trained_on_signals,
   train_loss, val_nll, val_nll_final, val_nll_all, eval_seqs, selected_step,
   selected_ckpt, select, best, bench_n, bench, card, samples_png, note`
   (`run_record` in `blockgen/train/trainer.py`; `registry.append` prepends
   `time`). `bench_n` is absent from lines written before 2026-10-31, so
   readers treat it as optional. Quote `val_nll_all` (the selected weights on
   every holdout sequence that fits) as the holdout NLL.
2. **`results.md`.** A new `## T<next>. <title> (<date>)` entry: the config, the
   run dirs, the bench card dir, n, and the table in the unit of that card.
   The last entry today is T78.
3. **`notes.md`.** A `## §<next>.` section: what was tried, what it showed, what
   is proven or disproven. The last section today is §62.

The docs summarise those two files and do not replace them.

---

## Appendix A: legacy arms

Arms that did not come through `blockgen.train` enter at step 6 as any
`.npz` + `_manifest.json` pair written by
`blockgen.curation.houses.save_house_cache`. What differs is who records
`train_split`:

```bash
# A checkpoint served in deploy/inference/models.json (the piece-token arms).
PYTHONPATH=. python scripts/sample_to_npz.py --model native_oriented --n 399 \
    --out-dir outputs/bench_arms --name native_oriented --train-split canonical

# Per-sample .npy or a keyed .npz from an older battery.
PYTHONPATH=. python scripts/dump_samples.py --keyed-npz outputs/run_x/arm/samples.npz \
    --out-dir outputs/run_x/arm --name arm --max-dim 32 --crop
```

`sample_to_npz.py` records `--train-split`, falling back to the model's
`train_split` in `models.json`; `dump_samples.py` writes a report with no
`train_split`, so its arm is provisional until the manifest says `canonical`.
Agentic arms (`blockgen/agentic/core.py`) record `train_only` or `none` and
the suite's oracle arms `oracle_train`, all provisional by design. The facade
ceilings and floors (`scripts/facade_oracle.py`, `scripts/facade_baselines.py`)
record `drawn_from` and no split. Best-of-N arms (`blockgen.posttrain.bon`)
carry the parent run's split and add `trained_on_signals`, which the board
flags without unranking.

## Appendix B: non-autoregressive models

The registry covers the loop, not the data shape. What a diffusion or graph
family gets for free: `build` with `cfg.model` knobs, `loss` on whatever it
returns, `sample` with its own decoding (the trainer draws through
`Arch.sample`, so a sampler that never emits tokens is fine), its own `track`
prefix on the bench, checkpoints, selection, the registry line, `--sample`,
and the bench and lab, which read structures only.

What it does not get: the batch. `blockgen/train/data.py::MixedBatches` yields
`(input_ids, targets, pad_mask)` as shifted token sequences under a token
budget, and `evaluate` weights the loss by non-pad targets. A dense-grid or
graph collate is a second seam, not written yet; until it exists, a family
that needs one trains through its legacy trainer in `blockgen/training/`
(`train_diffusion.py`, `train_graph.py`, `train_pick_n_place.py`) and enters
the pipeline at step 6 as an arm. Constrained decoding
(`blockgen/training/constrained_decode.py`) is also legacy-only:
`blockgen/train/sample.py` has no logits-mask hook.
