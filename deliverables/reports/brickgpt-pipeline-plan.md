# BrickNet-style LLM through the whole pipeline: plan (2026-11-04)

Scope: take the Qwen3-0.6B + LoRA text-walk model in `libs/BrickNet` (the BrickNet recipe,
Kulits and Schmid CVPR 2026; not the ICCV 2025 BrickGPT, which is the LEGO arm of T83)
through pretrain on world crops, SFT on `sft-houses32`, post-training and a `houses32-v1`
card at n=256. Read-only investigation; nothing was launched.

## 0. Headline finding: the SFT readouts are an adapter-stacking bug, not a model failure

`train.py sft --pt_adapter A` merges A into the base weights and then trains a new LoRA on
top (`libs/BrickNet/scripts/train.py::load_model`). The saved SFT adapter is therefore only
the delta over base+A. `repro/run_mc_sft_eval.sh` and `repro/run_mc_eval.sh` call
`generate.py --lora <one adapter>`, which loads that delta onto raw Qwen3:

| readout | what was loaded | what it should be | result |
|---|---|---|---|
| `mc_sft_base_val256` | Qwen3 + sft | Qwen3 + mc_pt_base + sft | English prose, 0% parsable |
| `mc_sft_legoinit_val256` | Qwen3 + sft | Qwen3 + BrickNet-0.6B-PT + mc_pt_legoinit + sft | `a = a` loops, 0% parsable |
| `mc_pt_legoinit_{ns,as}` | Qwen3 + mc_pt_legoinit | Qwen3 + BrickNet-0.6B-PT + mc_pt_legoinit | 3.1% / 2.0% parsable |

The training itself worked: SFT val NLL fell to 0.649 (base init) and 0.653 (LEGO init)
against the PT adapters' 0.676 / 0.686. `generate.py` already accepts repeated `--lora`
(merged in order), so the fix is one flag per script, e.g.
`--lora repro/runs/mc_pt_base/adapter --lora repro/runs/mc_sft_base/adapter`.
The T69 note that the LEGO-init readout "crashed on a node labelled An" is the same bug.
The `mc_pt_base` readout (one adapter, nothing merged under it) is valid.

## 1. What exists today

**Format** (`libs/BrickNet/repro/mc_to_bricknet.py`). A sequence from a corpus shard is
decoded to cells, split into 6-connected components, and each component becomes a random
spanning-tree walk: a node line `<label> <block name>[ | <data>]`, then an edge line
`<parent label> face <dir>` (the child sits on that face of the parent). Labels are
bijective base 26, identity is positional. 9.43 Qwen3 tokens per block on val walks.
Round trip exact on 2,500 walks. SFT pairs a caption (p50 24 tokens) with the walk; loss
on the walk only. The converter reads any tier's shard (`Shard(find(tier))`), so a new
tier needs no new converter code.

**Data built** (`data/minecraft/bricknet/`, T69):

| file | tier | selection | walks |
|---|---|---|--:|
| `paths_pt16` (training pool) | `pretrain-clean` (not `anneal-worldcrop`) | 320k random sequences of at most 600 blocks, components of 16+ | 524,542, median 59 blocks, ~600M tokens |
| `paths_sft` + captions | `sft-houses32` | largest component, at most 800 blocks: 927 of 1,863 houses, 4 captions each | 3,708 |
| `paths_val` + captions | `eval-holdout` | same cut: 494 of 798 | 1,976 |

So pretraining was on small fragments of the broad cleaned tier, and SFT saw only the
smaller half of the houses (largest component only).

**Runs** (`libs/BrickNet/repro/runs/`): `mc_pt_base` (6,000 iters x 32 seqs, 10.6 h, val
NLL 0.677), `mc_pt_legoinit` (same, init merged from `kulits/BrickNet-0.6B-PT`, 11.4 h,
0.686: no transfer, T69), `mc_sft_base` and `mc_sft_legoinit` (2,000 iters x 8, max_len
8192, grad ckpt, 4.0 h and 2.8 h, val NLL 0.649 / 0.653). Each has `adapter/` plus the
last three checkpoints. Logs `runs/*.log`, queue logs `mc_chain.log`, `mc_chain2.log`.

**T69 readout, `mc_pt_base`, 256 unconditional 100-block samples:** nucleus Conn 99.6,
Coll 53.0, 99.2% parsable, 33.2% parsable and collision-free (48% of those are lines);
ancestral 91.1 / 36.9 / 77.7% / 9.0%. Samples are fragments and blobs, not houses.

**Bench:** never scored. `mc_eval.py` writes a `<grid>.npz` of `ids_i`/`data_i` keys,
which is not the bench cache format (`curation.houses.save_house_cache`, object arrays
`block_ids`/`block_data` plus a `_manifest.json` report). No run has a manifest,
`train_split`, `summary.json`, registry line or run-dir `samples.png`.

## 2. What "whole pipeline" means, step by step

Integration decision first. A registry `Arch` (`blockgen/train/archs.py`) is the wrong
seam: the contract is `model(input_ids, pad_mask) -> logits` over the shard
`CorpusTokenizer`, batched by `MixedBatches`, run under the conda Python 3.13 / torch 2.7
env. BrickNet needs Qwen's tokenizer, PEFT, liger-kernel 0.6.5 and transformers 4.57 in
its own venv. Wrapping it as an Arch means a second tokenizer path, a second collate and a
cross-venv import: a fork disguised as a registration. Recommended: **a separate track
that shares tiers, split, rewards, `bon.select`, the bench and the record formats**, with
one driver in the conda env that shells into the BrickNet venv for the two GPU steps. That
is exactly the "enters at step 6 as an arm" route of `docs/add-a-model.md` Appendix B,
plus the run-dir and registry conventions.

Proposed driver: `scripts/bricknet_pipeline.py` (new, conda python), subcommands
`data | pt | sft | sample | bon | rft`, each writing into one run dir from
`utils/runs.new_run_dir` (`outputs/run_<stamp>_bricknet_<stage>_<name>`):
`run.json` (config, git sha, tier names and sha256, parent run, adapter stack),
`metrics.jsonl` (parsed from the HF trainer log), `adapter/`, `samples.png`,
`samples_n256_32.npz` + `_manifest.json`, `summary.json`, and one
`outputs/registry.jsonl` line via `blockgen/train/registry.append` with `arch: bricknet`,
`stage`, `tiers`, `init_from`, `trained_on_signals`, `select: best`, `val_nll`. The
adapter stack (every adapter to merge, in order) must be stored in `run.json`, which
removes the bug in section 0 by construction.

| step | code that exists | what to write | notes |
|---|---|---|---|
| (a) PT on `anneal-worldcrop` | converter reads any tier; `run_mc_pt.sh`; patched `train.py` | `data` subcommand: build walks from the `anneal-worldcrop` shard (being built now, `shards/anneal-worldcrop/60684456e598`) into `data/minecraft/bricknet/anneal-worldcrop_<sha12>/`; raise `--max_len` to 20480 with `--grad_ckpt`; write tier hash into `run.json` | windows are long, see section 3; full LoRA r32 (recipe) not full fine-tune, which would not fit 16 GB with Adam |
| (b) SFT on `sft-houses32`, early stop on `eval-holdout` | `run_mc_sft.sh`; `train.py` evals every N and keeps checkpoints | rebuild SFT/val walks without the 800-block cut (all 1,863 / 798, max_len 20480); keep the best-eval checkpoint (`select=best`, pipeline rule) instead of the last; eval on a fixed 256-pair val subset to keep each eval under 10 min | the HF Trainer can do `load_best_model_at_end`; needs a flag in the patched `train.py` |
| (c) best-of-4 under `lcc26`, optional RFT | `blockgen/posttrain/rewards.py` (`lcc26`), `bon.select`, `rft` via `shards.from_structures` | `bon`: draw 4 per prompt, realise, score with `rewards.get("lcc26")`, `bon.select(scores, 4)`, record `trained_on_signals=[lcc26]` and `scores.parquet`. `rft`: kept builds to a tier through `shards.from_structures`, then the same converter, then one more `train.py sft` mixed with the SFT pairs (captions: the prompt that produced each kept build) | `bon.run` itself calls the native sampler, so only its `select` and reward objects are reused, not its CLI |
| (d) 256 samples to bench npz | `generate.py` (stacked `--lora`), `parse_path`, `cells_to_arrays` | `sample`: realise with collision count, crop to 32 and centre, `save_house_cache`, manifest report `{model, arch: bricknet, checkpoint, adapters, train_split: canonical, trained_on_signals, n, seed, temperature, top_k, top_p, prompts}`, render `samples.png` with `blockgen.renderer.grid.save_grid` | unparsable or empty draws: count them in the report, do not silently drop (bench arm size must stay 256, so draw until 256 valid or report the shortfall) |
| (e) card | `python -m blockgen.eval.bench --tier both --n 256 --baselines --arms llm/<run name>:<npz>` | nothing | track prefix is free text (`runner.py` rpartition); `llm/` keeps it apart from `ar/` |

`train_split`: Qwen3 was pretrained on an outside web corpus and the PT tier excludes the
holdout, which is the same footing as M2b (pretrain on `anneal`, SFT on `sft-houses32`,
ranked). Recording `canonical` is defensible; the card text should say the base is an
external LLM.

Prompting is a decision. The SFT model is caption-conditioned. The 256 prompts in
`data/minecraft/bricknet/prompts_val256.jsonl` come from `eval-holdout` (val + test), so
some prompts describe test houses: that is a prompt-matched arm, like T83's BrickGPT arm,
not an unconditional one. Options: captions of training houses (clean, unconditional in
distribution), or test captions (labelled prompt-matched).

## 3. Cost on one shared 16 GB GPU

**Tokens per build in this format** (9.43 tokens per block):

| set | blocks p50 / p90 / p99 | tokens p50 / p90 / p99 |
|---|---|---|
| `sft-houses32` sequences | 891 / 1,711 / 2,002 | 8.4k / 16.1k / 18.9k |
| `eval-holdout` sequences | 694 / 1,605 / 1,987 | 6.5k / 15.1k / 18.7k |
| `anneal` world-crop windows (proxy for `anneal-worldcrop`, shard cap 2,048 blocks) | 1,336 / 1,902 / 2,040 | 12.6k / 17.9k / 19.2k |

One pass over the ~1.2M `anneal-worldcrop` windows is about 1.6B blocks, ~15B tokens:
far beyond budget, so PT is a sample of windows, not an epoch.

**Throughput measured in the logs.** PT (`mc_pt_base`): 192k walks, ~221M tokens in 10.6 h,
about 21M tokens/h (5.8k tok/s), micro-batch 1, no checkpointing, max_len 6400. SFT:
~74M tokens in 2.8 to 4.0 h with checkpointing at max_len 8192, 18 to 26M tokens/h. For
12k-token windows assume 15 to 20M tokens/h (attention grows, checkpointing needed).

| PT budget | windows | GPU hours |
|---|--:|--:|
| 550M tokens (matched to the 5M/20M runs) | ~44k | 28 to 37 |
| 220M tokens (matched to the existing pilot) | ~17k | 11 to 15 |
| 100M tokens | ~8k | 5 to 7 |

SFT on all 1,863 houses x 4 captions: ~64M tokens per epoch; 2,000 iterations x 8 pairs
(~2 epochs, 138M tokens) is 6 to 9 h including evals.

**Sampling** with HF `generate` (Qwen3 dynamic cache): the SFT readout produced 256 x 4,096
tokens in about 1 h at batch 8, roughly 290 tok/s or 28 ms per decode step. A full house
needs 8 to 16k tokens and a batch runs to its longest member, so 256 houses take 3 to 5 h
and best-of-4's 1,024 draws 12 to 20 h. KV cache is ~114 KB per token (28 layers, 8 KV
heads, dim 128, bf16), so batch 8 at 16k tokens is ~15 GB: batch 4 to 6 is the ceiling.
A vLLM install in a separate venv, with the adapters merged into one checkpoint
(`generate.py --model <merged>` already supports this), would plausibly cut this 5 to 10x;
not measured.

## 4. Risks

- **Context length.** A whole 32^3 house is up to ~19k tokens; Qwen3-0.6B accepts 32k, so
  it fits, but every cost above scales with it, and the existing 800-block cut biased SFT
  towards small houses (geom_kid will see the size shift). Largest-component-only walks
  also drop detached parts (lanterns, fences, floating roofs), which `lcc26` then rewards.
- **Invalid outputs.** Collisions (a node placed on an occupied cell) and unparsable lines;
  the PT pilot was 33% collision-free at 100 blocks, and the rate falls with length. A
  collision mask in a logits processor (forbid `face <dir>` edges into occupied cells)
  would make every draw valid; it is a decoding deviation and must be labelled. Truncation
  at `max_new_tokens` yields partial houses; set it from the p99 (20k), not 4,096.
- **Degenerate PT samples.** T69's fragments were lines and blobs; windows from
  `anneal-worldcrop` are larger and more built than `paths_pt16`, which is the point of the
  new PT, but nothing guarantees structure.
- **Memory.** PT at max_len 6400 without checkpointing was 11.7 GiB; at 20k it needs
  `--grad_ckpt` (2.9 to 3.7 GiB measured at 6.4 to 8k). torch.compile recompiles per width:
  keep `--no_compile`. The GPU is shared: one job at a time, and chain scripts must not
  wait on a pgrep pattern (T69 stall).
- **Licensing.** Qwen3 weights Apache 2.0, BrickNet code MIT, world crops the same footing as
  every other pipeline run. The LEGO adapter is not needed (T69: no transfer), which keeps
  the BrickNet dataset licence out of the Minecraft arm.
- **Comparability.** BlockScore only at equal n with the n=256 board; `select=best`; record
  `lcc26` in `trained_on_signals` for the best-of-4 and RFT arms.

## 5. Recommended minimal first run

Stage 0, about 4 to 5 GPU hours, no new training: fix the adapter stack in the two readout
scripts, sample 256 full houses from the existing `mc_sft_base` (stack `mc_pt_base` +
`mc_sft_base`, `max_new_tokens` 20480, train-house or test captions per the decision),
convert to a bench cache and card it at n=256. This answers whether the existing model is
worth more compute before 30+ GPU hours go into PT.

Stage 1 if stage 0 shows houses: PT 220M tokens on `anneal-worldcrop` windows (11 to 15 h),
SFT on all houses with best-checkpoint selection (6 to 9 h), sample and card (3 to 5 h):
about 25 GPU hours. Best-of-4 `lcc26` adds 12 to 20 h at HF speed, or about 2 to 4 h with
vLLM.

## 6. Decisions for the user

1. Allow edits under `libs/BrickNet/repro` (readout fix, best-checkpoint flag, max_len) and
   a new `scripts/bricknet_pipeline.py`, or keep everything outside `libs/`.
2. Prompts for the card: training-house captions (clean) or test captions (prompt-matched).
3. Collision-masked decoding: allowed as a labelled arm, or raw sampling only.
4. PT budget: 100M, 220M or 550M tokens; and whether windows longer than 20k tokens are
   dropped or cropped.
5. Install vLLM in its own venv for sampling (needed for best-of-4 and RFT to be practical).
6. `train_split: canonical` for a Qwen3-based arm, or a distinct `external` label.
