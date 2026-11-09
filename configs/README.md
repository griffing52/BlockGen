# Configs

Three families live here. `corpus/` and `train/` are the pipeline
(`docs/pipeline.md`); the rest are the legacy experiment batteries. Every YAML
is loaded by `blockgen/config.py`: a flat mapping, `extends:` for composition,
and a typo'd key fails loudly. Commands in this file run under the conda
`python` on PATH; the legacy batteries also run under `.venv/bin/python`, the
corpus layer does not (no pandas there).

| Folder | Loaded by | What one file pins |
|---|---|---|
| `corpus/` | `python -m blockgen.corpus.tiers build` | a tier: a filter over the manifest; schema in `corpus/README.md` |
| `train/` | `python -m blockgen.train --config` | a run: every `TrainConfig` field below |
| `lab/` | `python -m tools.lab --phone` | the phone queue (`phone.json`) |
| `datasets/`, `training/`, `models/`, `agentic/`, `experiments/` | `blockgen.experiments_ideas`, `experiments_overnight`, `experiments_agentic` | the legacy batteries (last section) |

## Composition

`extends:` lists bases (string or list), each a bare name or a path relative to
the file, then to `configs/`. Bases merge left to right; the file's own keys
override them. Cycles error out. A `description:` key is dropped at load.
Abridged from `train/houses32-sft-5m.yaml` (the file also sets `warmup: 200`,
`eval_every: 500` and `ckpt_every: 4000`):

```yaml
extends: [base]
name: houses32_sft_5m
stage: sft
train: [sft-houses32]
val: [eval-holdout]
steps: 16000
lr: 1.0e-4
dropout: 0.0
bench_n: 256
```

## `train/`: the TrainConfig schema

`blockgen/train/config.py` defines `TrainConfig`; a YAML may set any field and
nothing else (`load` raises `ValueError` on an unknown key). `key=value` on the
command line overrides the YAML with the same coercion: ints, floats and bools
by the field's type, `List[str]` as a comma list (`train=rft-x:1,sft-houses32:1`)
or a JSON list, dict fields as a JSON object (`model='{"geo_bias": true}'`),
`none` for `None`. Defaults below are the dataclass defaults; `base.yaml`, which
every run config extends, overrides the ones marked.

```
python -m blockgen.train --config configs/train/<name>.yaml [key=value ...]
```

| Field | Default | `base.yaml` | Meaning |
|---|---|---|---|
| `name` | `run` | | run name; the run dir is `outputs/run_<stamp>_<name>` |
| `stage` | `pretrain` | | `pretrain`, `anneal`, `sft`, `posttrain`: a registry label |
| `preset` | `None` | `5m` | one of `PRESETS` (`5m`, `20m`, `60m`, `150m`, `300m`); sets `d_model`, `nhead`, `layers`, `ff` |
| `arch` | `ar2` | | the registered `Arch` in `blockgen/train/archs.py` that builds, scores and samples the model |
| `model` | `{}` | | extra constructor kwargs for the arch; an unknown key or one duplicating a field raises |
| `d_model` | `256` | | |
| `nhead` | `8` | | |
| `layers` | `6` | | |
| `ff` | `1024` | | feed-forward width |
| `dropout` | `0.0` | | changing it at `init_from` is refused (T59) |
| `pe` | `phase4` | `phase4` | positional encoding of `VoxelTransformerAR2`; `alibi` decodes without a KV cache |
| `max_seq_len` | `8200` | `8200` | context; longer shard sequences are dropped, never truncated |
| `train` | `[sft-houses32]` | | tiers, each `tier` or `tier:weight` (relative sampling weight) |
| `val` | `[eval-holdout]` | | validation tiers; `best.pt` follows the first |
| `tokenizer` | `None` | | shard tokenizer hash; `None` takes the newest shard of the first train tier, so pin it when a tier has several shards |
| `kind` | `voxels` | | `voxels` or `facades`: which shards and which sampler |
| `augment` | `""` | | `""` or `d4`: train on the `<hash>-d4` shard; validation stays plain |
| `train_split` | `None` | | provenance the bench ranks by; `None` records `canonical` iff `train == [sft-houses32]`, else the joined tier list |
| `trained_on_signals` | `[]` | | rewards a post-training stage optimised, `name[@weight]` tokens |
| `batch_tokens` | `32768` | `32768` | padded tokens per micro-batch |
| `grad_accum` | `1` | `1` | |
| `seed` | `0` | | |
| `steps` | `2000` | | optimizer steps |
| `lr` | `3e-4` | `3.0e-4` | peak learning rate |
| `min_lr_frac` | `0.1` | `0.1` | the decay ends at `min_lr_frac * lr` |
| `warmup` | `100` | `100` | linear warmup steps |
| `decay_frac` | `0.2` | `0.2` | last fraction of steps in the linear decay phase (WSD) |
| `weight_decay` | `0.1` | `0.1` | on 2-D parameters other than embeddings |
| `beta1` | `0.9` | | |
| `beta2` | `0.95` | | |
| `clip` | `1.0` | | gradient-norm clip |
| `amp` | `True` | | bf16 autocast on CUDA |
| `compile` | `False` | | `torch.compile`; checkpoints are saved unwrapped |
| `log_every` | `20` | | steps between `metrics.jsonl` lines |
| `eval_every` | `500` | `500` | steps between validation passes |
| `eval_seqs` | `256` | `256` | holdout sequences per periodic pass (the first that fit the context) |
| `final_eval_all` | `True` | | after selection, score the selected weights on every holdout sequence that fits (`val_nll_all`) |
| `ckpt_every` | `1000` | `1000` | steps between `ckpt_<step>.pt` and `last.pt` saves |
| `sample_every` | `0` | | periodic sampling; `0` samples only at the end |
| `n_samples` | `16` | `16` | builds rendered to `samples.png`; the end-of-run draw is `max(n_samples, bench_n)` when `bench` is on, else `n_samples` (`blockgen/train/trainer.py:376`) |
| `sample_tags` | `{src: curated, q: 4, size: 32, win: whole}` | | prefix tags at sampling |
| `sample_max_blocks` | `2048` | | token budget cap per sampled build |
| `sample_batch` | `64` | | builds per sampling batch (facades cap at 32); lower on a shared GPU |
| `temperature` | `1.0` | | |
| `top_k` | `40` | | |
| `bench` | `True` | | run `blockgen.eval.bench` on the end-of-run samples |
| `bench_tier` | `fast` | | `fast`, `full` or `both` |
| `bench_n` | `128` | `64` | samples benchmarked; the M2, M2b and RFT configs set 256 |
| `bench_corpus` | `houses_32` | `houses_32` | the bench's reference corpus |
| `select` | `best` | | `best` (lowest first-tier NLL), `last`, or a registered hook; `reward` comes from `blockgen.posttrain` |
| `init_from` | `None` | | checkpoint path, weights only: a new stage from a finished run |
| `allow_dropout_change` | `False` | | accept an `init_from` checkpoint trained at another dropout |
| `allow_partial_init` | `False` | | accept an `init_from` checkpoint whose keys do not all match; counts go to the registry |
| `resume` | `None` | | run dir: continue the same run from `last.pt` with full state; the config is not reloaded |
| `out` | `outputs` | | parent of the run dir |
| `device` | `cuda` | | |
| `note` | `""` | | free text copied into the registry |
| `config_path` | `None` | | set by `load` to the YAML used, for the registry |

Configs in `train/` today: `base`, `smoke`, `smoke-facades` (wiring checks),
`anneal-pretrain-5m` (the pretraining pilot), `houses32-scratch`,
`houses32-scratch-matched` and `houses32-sft-5m` (the M2b comparison, T60),
`houses32-facades`, `-joint`, `-joint-20m`, `-joint-d4`, `-joint-d4-20m` (the
facade track, T66 and T68), `rft-voxel-5m` and `rft-facades-joint-d4-20m`
(post-training, `docs/pipeline.md` §7). Nothing names `pretrain-all` or
`pretrain-clean` yet.

## `corpus/`

One YAML per tier; keys `description`, `holdout_split`, `holdout_parts`,
`sources`, `filter`, `dedup`, `exclude`. Full schema, the seven tiers and the
build order are in `corpus/README.md`.

## The legacy batteries

Pass `--config <name-or-path>` to a battery entry point. A bare name resolves to
a file anywhere under `configs/` (`ideas-full` is `experiments/ideas-full.yaml`):

```bash
python -m blockgen.experiments_ideas     --config ideas-full     --stamp 20260704_x
python -m blockgen.experiments_overnight --config overnight-full --stamp 20260704_y
```

The config supplies defaults; a flag passed on the command line still wins, so a
one-off tweak needs no new file:

```bash
python -m blockgen.experiments_ideas --config ideas-full --epochs-ar 5
```

The launcher scripts forward extra flags (`./scripts/run_ideas.sh --config
ideas-pe-only`), and the agentic battery takes `--config` the same way, its keys
being that runner's flags (`arms`, `prompts`, `n`, `provider`, `canvas`):

```bash
python -m blockgen.experiments_agentic --config agentic-scaffolding
python -m blockgen.experiments_agentic --config agentic-detail --prompts captions:2
```

Keys are argparse dest names (`epochs_ar`, `diff_grid`, `pe_arms`). The folders
group fragments by concern and are not YAML nesting; a merged config is one flat
mapping.

| Folder | What it pins |
|---|---|
| `datasets/` | data-prep knobs (min dim and blocks, val split, augment, canon) |
| `training/` | epoch, batch and sequence budgets for AR and diffusion |
| `models/` | model-variant sweeps (PE arm lists); not the size presets, which are `PRESETS` in `blockgen/train/config.py` |
| `agentic/` | Track E loop knobs (canvas, token budget, reasoning effort) |
| `experiments/` | top-level runs that `extends:` the above and set selectors |

The dataset selector (`dataset:` for ideas, `datasets:` for overnight) lives in
the experiment file, since the two entry points name that flag differently. A
new sweep of an existing kind is an `experiments/*.yaml` that extends existing
fragments; a new reusable knob set is a fragment under the matching folder.
