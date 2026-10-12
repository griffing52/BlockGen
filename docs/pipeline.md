# The training pipeline: pretrain, fine-tune, post-train

Design document, written 2026-10-11. This is the plan for turning the repo's
collection of per-experiment training scripts into one stack that can pretrain on
the world-crop corpus, fine-tune on curated builds, post-train against rewards and
preferences, and tell us at every stage whether the extra work paid off. It
borrows the shape of the open LLM stacks (OLMo 3, nanochat, TRL) at the scale of
one 16 GB GPU and models of 5M to 300M parameters.

Companion pages: `workflow.md` (the train, sample, evaluate, view route the stack
must keep producing artifacts for), `worldcrop.md` (where the corpus comes from),
`benchmark.md` (BlockScore, the fixed yardstick), and `research.md` §G (the
literature behind the choices here).

---

## 1. Why a pipeline, not another script

Today each model family has its own training loop (`blockgen/training/train_*.py`,
`scripts/train_*.py`, `blockgen/experiments_*.py`). Data is loaded whole into
memory from the curated caches. None of them can read the 850k world crops, none
save optimizer state or resume, none record a git sha or a tokenizer hash, and
evaluation happens by hand after the fact. That was fine for 2.6k houses and
5M-parameter models. It is not fine for a corpus of a million builds, a scaling
sweep, or a post-training stage that has to start from a known checkpoint and be
compared against it.

The LLM analogy the plan follows:

| LLM stage | BlockGen equivalent | What we learn from it |
|---|---|---|
| Pretraining on web text | Pretrain on all world crops, dedup hard, filter soft, tag quality | Does scale buy cohesion and validity? |
| Mid-training / anneal | Decay phase on curated houses plus high-quality crops | How much of the gain is the mix, not the volume? |
| SFT | Curated captioned houses, text and image conditioning | Conditioning fidelity on top of a pretrained prior |
| RLVR / GRPO | Validity, support, cohesion, novelty as verifiable rewards | Does training beat search (best-of-N)? |
| Preference tuning | DPO or SimPO on lab 2AFC pairs and reward-ranked pairs | Human preference as the last pass |
| Distillation | 300M teacher into the deployable 20M model | Keeps the live `/gen` demo fast |

---

## 2. Layout

```
blockgen/
  corpus/        manifest, dedup, tiers, tokenized shards        (new)
  tokenizers/    block_tokenizer, cluster_bpe v2, versioned artifacts
  train/         one step-based trainer, config tree, presets    (new)
  posttrain/     rewards, rejection-sampling FT, GRPO, DPO, BoN  (new)
  eval/bench/    BlockScore (unchanged), plus a periodic hook
  models/        unchanged; registered by name in presets
outputs/
  run_<stamp>_<name>/   config.json, run.json (lineage), metrics.jsonl,
                        ckpt/{last,best}.pt, samples.png, scorecard.json
  registry.jsonl        one line per run, appended by the trainer
configs/
  corpus/*.yaml         tier definitions (filters + hash of the member ids)
  train/*.yaml          stage configs: pretrain, anneal, sft, rft, grpo, dpo
  models/*.yaml         size presets d5m, d20m, d60m, d150m, d300m
```

The existing `blockgen/config.py` (`extends:` plus flat overrides) stays as the
loader. The new part is a typed config tree with sections `model`, `data`,
`optim`, `sched`, `eval`, `run`, so a stage is a YAML file and a run is a YAML
file plus overrides.

---

## 3. Corpus layer

### 3.1 Manifest

One table over every build we own, curated or cropped. Built once per month
directory by a resumable worker, stored as Parquet under
`data/minecraft/corpus/manifest/`.

Columns: `id`, `source` (grabcraft, 3dcraft, text2mc, worldcrop), `path`,
`month`, `world_id`, `title`, `author`, `bbox` (dx, dy, dz), `n_blocks`, `fill`,
`n_ids`, `p_built` (gate), `cut`, `oversize`, `canon_hash`, `dedup_cluster`,
`split_group`, plus the 295 gate features as a float16 block so a fragment
detector or any new tag is a model over the table rather than a pass over npz
files. Scoring the corpus took about three hours on 12 workers, so the manifest
build is an overnight job and is run month by month as the ingest finishes.

### 3.2 Deduplication

Worlds are re-uploaded across months and builds are copied within worlds. The LLM
literature is consistent that dedup matters more than aggressive quality
filtering, and our own 31% GrabCraft memorisation says novelty must be measured
after dedup, not before.

- Exact: `canon_hash` of the crop after translation and D4 normalisation.
- Near: MinHash over shingles of (block id, relative offset) triples; cluster at
  Jaccard above 0.8. One representative per cluster goes to training.
- Splits are group-aware on `world_id` and `dedup_cluster`, so no test build shares
  a world or a cluster with a training build. This extends
  `eval/bench/splits.py` (v1) rather than replacing it; the pinned
  `houses_32`/`houses_48` splits stay as they are.

### 3.3 Tiers

A tier is a query over the manifest plus a hash of the resulting id list, stored
in `configs/corpus/<tier>.yaml`. Changing the filter changes the hash, and the
hash goes into every checkpoint.

| Tier | Filter | Role |
|---|---|---|
| `pretrain-all` | everything, deduped | the LLM-style broad corpus, terrain included |
| `pretrain-clean` | `p_built >= 0.5`, not `cut` | ablation: does soft filtering help? |
| `anneal` | curated houses plus crops with `p_built >= 0.9` and `n_blocks >= 500` | the decay-phase mix |
| `sft-houses` | the captioned `houses_32`/`houses_48` caches | conditioning |
| `eval-holdout` | pinned test splits | never trained on |

The gate rejects stay in `pretrain-all` on purpose. Terrain is a shape class the
model should know, and the quality tag below lets us sample away from it.

### 3.4 Tags, not deletions

Every training sequence starts with a short prefix of tags: `<src:archive|curated>`,
`<q:0..4>` (bucket of `p_built`, later of a fragment score), `<size:32|48|64|128>`,
and `<cat:...>` when a category is known. This is the quality-conditioning pattern
from Korbak et al. and MeCo, and the micro-conditioning trick from SDXL. At
sampling time we condition on `<q:4>`, optionally with classifier-free guidance
against `<q:0>`. During the anneal the tags are dropped (MeCo cooldown) so the
final model does not need them.

### 3.5 Windows and shards

Crops range from a bed to a castle. Rule: a crop with at most 2,048 blocks is one
sequence; a larger crop contributes random 32³ or 48³ windows, resampled each
epoch, which is what Dream-Cubed does with whole maps. Sequences are tokenised
once per tokenizer version into uint16 memmap shards under
`data/minecraft/corpus/shards/<tier>/<tokenizer_hash>/`, with an index of
(offset, length, size bucket, tags) for length-bucketed batching.

---

## 4. Tokenizers and the comparison arms

Every arm trains on the same tiers and is scored by the same BlockScore protocol.
Each arm is one tokenization story.

| Arm | Tokens | Status | Needs |
|---|---|---|---|
| A. Sparse raster AR | `(x, y, z, id)` per block, current `VoxelTransformerAR2` | trains today | shards, prefix tags, 8k context |
| B. Raster AR with 3D-BPE v2 | pieces from `cluster_bpe` | v1 trains; pieces are poor (T18) | learn on 50k builds, PMI merge scoring, shape-factored merges, versioned artifact |
| C. Masked discrete diffusion | dense grid with a MASK class, 2³ patches at 32³ | `VoxelUNet3D` at canon 12-20 | transformer backbone shared with A, 4,096-token grid |
| D. LLM baseline | one text line per block (BrickGPT format) on Qwen2.5-Coder-1.5B, LoRA | `scripts/train_llm_brickgpt.py` | same tiers rendered as text, rejection sampling at decode |

Arm A gets the full stack first because it is closest to ready. Arm B is the
project's native bet and is where tokenizer versioning matters most; the deploy
README already records a silent vocabulary mismatch. Arm C is where the strongest
published Minecraft results live (Dream-Cubed, Scaffold Diffusion) and gives
inpainting for free. Arm D is the reviewer-expected baseline.

A tokenizer is an artifact with a hash. The hash is written into `run.json` and
into the checkpoint, and the trainer refuses to load a checkpoint against a
different tokenizer.

---

## 5. The trainer

One module, `blockgen/train/trainer.py`, used by every stage. The stage is the
config, not the code.

- Step-based, not epoch-based. Warmup plus cosine or warmup-stable-decay (WSD),
  since WSD is what makes the anneal a natural phase.
- bf16 autocast, gradient accumulation, `torch.compile`, flex-attention as today.
- Full-state checkpoints: model, optimizer, step, RNG, data cursor, tokenizer hash,
  tier hash, git sha. Resume is `--resume <run_dir>`.
- Periodic validation NLL on each tier's held-out slice, so the pretraining curve
  and the curated-data curve are both visible.
- Periodic sampling: 64 builds to `samples.png` and `samples_textured.png`, which
  keeps the repo rule that every run renders samples.
- End-of-run evaluation: sample 128, run the fast BlockScore tier, write
  `scorecard.json` into the run dir. The lab's `/runs` and `/leaderboard` pick it
  up with no extra step.
- Metrics to `metrics.jsonl` always; wandb optional behind a flag.
- Size presets with a single depth dial as in nanochat: roughly 5M, 20M, 60M,
  150M, 300M parameters. With 2k-token sequences and bf16, 300M is the ceiling for
  16 GB with accumulation.

Compute check: a million crops at about 1k tokens each is 1B tokens per epoch. A
60M model at a sustained 100k tokens per second sees one epoch in about three
hours, so a Chinchilla-sized pretraining run (20 tokens per parameter) is a
night, not a week. Scaling sweeps over the five presets at matched tokens fit in
a weekend.

---

## 6. Stages

1. **Pretrain**: tier `pretrain-all` (and `pretrain-clean` as the ablation), tags
   on, WSD stable phase.
2. **Anneal**: same run continued into the decay phase on tier `anneal`, tags
   dropped. This is the Llama 3 and MiniCPM move and is cheap to ablate by
   annealing on `pretrain-all` instead.
3. **SFT**: `sft-houses` with the caption and image conditions already produced by
   `blockgen/labeling/`. Prune easy examples (Llama 4) if the set is large enough
   to afford it.
4. **Post-train**, in cost order (section 7).
5. **Distil**: on-policy distillation from the best large checkpoint into the
   deployable size. Keeps `deploy/inference` fast.

Every stage records its parent run in `run.json`, so the registry is a lineage
tree and a fine-tune can always be traced to its pretrain, tier and tokenizer.

---

## 7. Post-training

Rewards live in `blockgen/posttrain/rewards.py` and wrap signals the repo already
computes: adjacency validity, floating-block fraction and support, connectivity,
cohesion, the fast BlockScore pillars, a novelty penalty from nearest-neighbour
retrieval against the training tier, SigLIP or MineCLIP text alignment on the
textured renders, and the agentic VLM judge. Rule learned from the LEGO RL papers
(PVPO's "physhack"): never train on validity alone; always pair a validity term
with a geometric or semantic term.

Ladder, cheapest first, each one a baseline for the next:

1. **Best-of-N with a verifier at inference.** No training. This is the number any
   trained method has to beat, and the Rollback-Free versus LegoGPT comparison is
   the published precedent reviewers will expect.
2. **Rejection-sampling fine-tuning.** Sample, keep what passes validity and a
   BlockScore threshold, fine-tune on it. One script, hours of compute.
3. **GRPO with verifiable rewards.** Dr. GRPO or GSPO loss, groups of 4 to 8,
   KL to the SFT reference, intermediate rewards on partial builds for credit
   assignment over long sequences (the VAR RL and Hi-GRPO lesson). Feasible at
   up to 100M parameters because rollouts are cheap and the verifier is Python.
4. **Preference tuning.** DPO or SimPO on lab 2AFC pairs (30 today, so this
   stage waits for more labelling) and on reward-ranked synthetic pairs; DRO
   style single-sample reward if pairs stay scarce.
5. **Merge and distil.** Average per-reward checkpoints (WARP), distil into the
   deployable model.

---

## 8. Tracking progress

The question the user asked is whether pretraining is worth it. The stack
answers it with three fixed measurements, taken the same way for every run:

- Validation NLL on the curated hold-out, as a function of training tokens and
  parameters. This is the scaling-law plot and is free.
- BlockScore fast tier on 128 samples from the final checkpoint, written as a
  scorecard, ranked on the lab leaderboard under the pinned protocol.
- For conditioned models, `retrieval_acc` on the captioned houses.

The decisive experiment is matched compute, three seeds: SFT from scratch versus
pretrain plus anneal plus SFT, on `houses32-v1`. T12 ran the small version of
this (16k pool, 5M model, one seed) and found nothing; the hypothesis is that
the pool was too small and the model too small to show transfer.

`outputs/registry.jsonl` is the machine-readable log: one line per run with name,
stage, parent, tier hash, tokenizer hash, preset, steps, final NLL, scorecard
path. `results.md` keeps the narrative T-numbers; the registry is what the lab
and the scaling plots read.

---

## 9. Milestones

| | Deliverable | Depends on |
|---|---|---|
| M0 | Manifest, dedup, tiers, tokenizer artifact, shards for arm A | runs now against finished months |
| M1 | Trainer, presets, registry, auto-eval; reproduces the native `houses_32` arm | M0 |
| M2 | Pretraining pilot: 20M and 60M on `pretrain-all` vs `pretrain-clean`, anneal, SFT; the decisive matched-compute comparison | M1, ingest complete |
| M3 | Post-training ladder steps 1 to 3 on the best M2 checkpoint | M2 |
| M4 | Arms B, C, D on the same tiers | M1 |
| M5 | 2AFC collection in the lab, preference tuning, distillation | M3 |

M0 and M1 are no-regret and do not depend on any open decision. M2 is where the
project learns whether the corpus pays for itself.

---

## 10. Decisions taken and open

Taken in this document: one shared trainer rather than adopting an external
stack (our sequences, renderers and metrics are too specific for TRL or
torchtitan to save time); arm A first; rejects kept and tagged rather than
filtered; windows for large crops; the fast BlockScore tier as the automatic
end-of-run eval.

Open, for discussion:

- Whether to add wandb now or stay with `metrics.jsonl` plus the lab.
- The window sizes and the block-count cutoff for whole-crop sequences.
- Whether arm C should share the transformer backbone with arm A (cleanest
  ablation) or keep the U-Net.
- How much labelling effort to put into 2AFC before preference tuning.
