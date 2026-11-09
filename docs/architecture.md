# Architecture: the codebase map

This page is the operator's map of the repo: how data flows from raw corpora to a card
on the leaderboard, what every package does, and where a new model plugs in. Concepts
live in [Models](models.md) and [Representations](representations.md); this page is about
the code. It uses **Track** for a generator family (Track A to F, [Models](models.md))
and **Arm** for a tokenisation story in the shared trainer (Arm A to E,
[Training pipeline](pipeline.md) §4).

## The pipeline at a glance

```
raw sources ──► manifest ──► dedup + kinds ──► tiers ──► shards ──► blockgen.train ──► run dir
 worldcrop,      corpus/      corpus/           corpus/   corpus/     train/              outputs/run_*/
 caches,         manifest.py  dedup.py          tiers.py  shards.py   trainer.py          {best,last}.pt
 templates                    kinds.py                    tokenizer.py                    samples.png
                                                                        │                 bench/scorecard.json
                                                                        ▼
                       leaderboard ◄── bench ◄── structure cache (npz + _manifest.json)
                       tools/lab       eval/bench/   train/sample.py, posttrain/bon.py,
                                                     agentic/report.py, legacy samplers
```

Every arm enters the benchmark as a structure cache, so a legacy trainer, an LLM program,
a procedural baseline or a best-of-N filter over a trained model are scored by the same
code. The legacy batteries (`blockgen/experiments_*.py`) walk their own prep, train and
evaluate loop and write `novelty.json` per arm ([Experiments](experiments.md)); they
predate the corpus layer and still produce the T10 to T24 arms.

## Package layout

| Path | Role |
|---|---|
| `blockgen/data/` | raw corpus readers: schematics, tfrecords, GrabCraft, and `worldcrop/` (archive.org world saves to crops, with the terrain gate) |
| `blockgen/curation/` | `Curator` and the cross-corpus house caches (`houses.py`, `corpus.py`) |
| `blockgen/corpus/` | the corpus layer: manifest, canonical hashes and MinHash, dedup, kinds, tiers, the `CorpusTokenizer`, shards, facades, representation registry, templates ([Data](data.md)) |
| `blockgen/train/` | the shared trainer: `config.py` (flat `TrainConfig`, presets), `archs.py` (architecture registry), `data.py` (shard batching), `trainer.py`, `sample.py`, `registry.py` ([Training pipeline](pipeline.md)) |
| `blockgen/posttrain/` | rewards with the pillar refusal, best-of-N (`bon.py`), rejection-sampling FT (`rft.py`), the `select=reward` hook |
| `blockgen/utils/` | `Structure` (`data.py`), `serialize.py` (legacy token streams and grids), corpora loaders, `block_remap.py`, `schem.py`, orderings, `growth_order.py` |
| `blockgen/tokenizers/` | `standard_vocab.py`, `cluster_bpe.py` (3D-BPE pieces), `piece_factors.py`; `block_tokenizer.py` is a dead stub |
| `blockgen/models/` | the model zoo (below) |
| `blockgen/training/` | the legacy trainers, one per family, plus `constrained_decode.py` |
| `blockgen/agentic/` | Track E: DSL, canvas, providers, agent loop, recipes, scoring ([Agentic](agentic.md)) |
| `blockgen/ontology/` | what each block is, mined from the corpus ([Block ontology](ontology.md)) |
| `blockgen/eval/` | `novelty.py`, `validity.py`, `traversal.py` (player walk), `facade_stats.py`, and `bench/` (the benchmark) |
| `blockgen/labeling/` | renders, captions, frozen embeddings ([Labeling](labeling.md)) |
| `blockgen/export/` | converters to external layouts ([MinecraftACE](minecraftace.md)) |
| `blockgen/renderer/` | textured EGL renderer, texture atlas, mosaic grids |
| `blockgen/experiments_*.py` | the legacy run drivers |
| `blockgen/config.py` + `configs/` | `configs/{datasets,experiments,models,training,agentic}` for the legacy loader; `configs/corpus/` tier yamls; `configs/train/` trainer yamls |
| `scripts/` | launch helpers, figure scripts, `m2_seeds.sh` / `m2b_seeds.sh`, `corpus_refresh.sh`, facade oracle and baselines |
| `tools/lab/` | BlockLab ([BlockLab](lab.md)) |
| `deploy/` | the live `/gen` demo: inference server and Minecraft mod |
| `libs/` | vendored MinecraftACE / LegoACE, BrickNet, BrickAgent |
| `legogen/` | Blender 4.2 LDraw renderer for the LEGO corpora |
| `outputs/` | run artefacts (gitignored); `outputs/registry.jsonl` is the run index |

## Data layer

Two generations coexist.

- **Caches** (`data/minecraft/cache/*.npz` + `_manifest.json`): fixed-max-dim
  `(block_ids, block_data)` volumes. Builders: `blockgen/curation/houses.py`
  (`houses_32` = 2,661, `houses_48` = 3,088), `blockgen/curation/corpus.py` (`all_32`),
  `blockgen/data/tfrecord_dataset.py`, `blockgen/data/grabcraft_dataset.py`,
  `blockgen/data/build_cache.py`. `load_house_structures(max_dim)` is the entry point
  every legacy experiment and the benchmark's reference use. External corpora load through
  `blockgen/utils/corpora.py` (3D-Craft, text2mc `.h5`, text2mc `.schem`).
- **The corpus layer** (`blockgen/corpus/`): one Parquet manifest over 927k builds
  (world crops, the three caches, game templates), `dedup.parquet` and `kinds.parquet`
  sidecars, tiers as hashed id lists, and uint16 token shards per tier and tokenizer. The
  trainer reads shards only. [Data](data.md) walks it end to end.

The benchmark's split is `data/minecraft/splits/houses_32.s0.70-15-15.v1.json`
(`blockgen/eval/bench/splits.py`); the `sft-houses32` and `eval-holdout` tiers are its
train and val+test sides.

## Representation layer

`blockgen/utils/data.py::Structure` is the universal unit: paired `block_ids` and
`block_data` arrays plus metadata, with `crop_to_non_air()` and `occupied_mask`; the D4
symmetries live in `blockgen/utils/augment.py`. Representations in use:

1. **Flat token stream.** Legacy: `blockgen/utils/serialize.py` (`BlockVocab` per run,
   `build_block_vocab`, `structure_to_grid`). Shared trainer:
   `blockgen/corpus/tokenizer.py::CorpusTokenizer` kind `voxels`, a fixed 507-token
   vocabulary (hash `60684456e598`) with prefix tags `src, q, size, win`.
2. **Facades.** `blockgen/corpus/facade.py`: six `(depth, id)` face images, `project`
   and `lift`; tokenizer kind `facades` in `per_face` or `joint` layout
   ([Representations](representations.md) §6).
3. **3D-BPE pieces.** `blockgen/tokenizers/cluster_bpe.py`: `learn_clusters` merges
   frequent adjacent-voxel patterns into a `ClusterVocab`. Legacy only; not a shard kind.
4. **Canonical grid.** Scale normalisation to `N³` for diffusion and the canon-12/16 AR
   regimes.
5. **Block-and-port graph.** PyG graphs for the graph VAE.
6. **Growth stream.** `blockgen/utils/growth_order.py` for Track F.

`blockgen/corpus/reps.py` registers `voxels` and `facades` as `encode`, `decode`,
`picture` for the lab and the figure scripts; it is not yet linked to the tokenizer's
`KINDS`.

## Model zoo (`blockgen/models/`)

| Class | File | Params | What it is |
|---|---|---|---|
| `VoxelTransformerAR2` | `voxel_transformer_ar2.py` | 5.5M (5m), 26.8M (20m) | PE-pluggable causal transformer, `pe` in {learned, sin, rope, alibi, phase4, none}; the only arch in `blockgen/train/archs.py` |
| `VoxelTransformerAR` | `voxel_transformer_ar.py` | ~5M | v1 baseline, learned absolute PE; equals AR2 `pe=learned` |
| `CondVoxelAR2` | `voxel_transformer_cond.py` | ~6M | AR2 plus prefix conditioning and CFG (T15/T16) |
| `ResampledCondVoxelAR2` | `condition_resampler.py` | ~6M | Q-Former-lite resampler over the full SigLIP sequence; `scripts/train_cond_resampler.py` |
| `GeoAttentionBias` | `geo_attention_bias.py` | | relative-offset attention bias for AR2 (`geo_bias`); null to mildly negative (T29) |
| `FactoredPieceEmbedding` | `factored_embedding.py` | | `(family, variant)` embeddings; unused since T20 |
| `SemanticTokenEmbedding` | `semantic_embedding.py` | | idea #7 prior; tested negative |
| `VoxelUNet3D` | `voxel_diffusion.py` | 5 to 15M | masked discrete diffusion 3D-UNet, four samplers |
| `LargePyGGraphGenerator` | `large_pyg_graph_generator.py` | few M | graph VAE |
| `VoxelPortGNN` | `voxel_port_gnn.py` | 1 to 2M | starter GNN, unused |
| `PickAndPlace` | `pick_n_place.py` | 5.4M | Track F picker and placer heads ([Pick-and-place](pick-and-place.md)) |

### `VoxelTransformerAR2` and the phase4 PE

AR2 hand-rolls its attention block on `F.scaled_dot_product_attention` so the positional
encoding is a constructor argument. **phase4** decomposes each position in the
`(X, Y, Z, BLOCK)` stream into a phase embedding (`pos % 4`, which slot) and a record-index
embedding (`pos // 4`, which record). The grammar is given rather than rediscovered, and
tail positions stop starving. The T11 ablation put phase4 first and ALiBi last; the
coordinates travel as tokens, so the PE's job is grammar, not geometry (`research.md` §B,
`results.md` T11). AR2's extra knobs (`geo_bias`, `piece_factors`, `semantic_embedding`)
reach the constructor through `TrainConfig.model`, a dict of keyword arguments.

### The model contract for the shared trainer

`blockgen/train/archs.py` is the seam. An `Arch` is `build(cfg, tokenizer)`,
`loss(model, batch)`, `sample(model, tokenizer, cfg, n, seed=None)` and a benchmark
`track`. A model the loop can drive exposes `forward(input_ids, pad_mask=None) -> logits`
and `.max_seq_len`; for fast sampling it may add
`forward(step, past=past, return_cache=True) -> (logits, past)`, and a model without a
cache (`pe=alibi` is one) is sampled by a full re-forward per step. Registering a new
family is one `register(Arch(...))` plus `arch: <name>` in a YAML
([Bring your own model](custom-model.md)).

## Training layer

**Shared trainer** (`blockgen/train/`): `python -m blockgen.train --config
configs/train/<name>.yaml [key=value ...]`. One step-based loop, warmup-stable-decay
schedule, token-budget length-bucketed batches, bf16, `init_from=<ckpt>` for a new stage
(refuses a tokenizer-hash or dropout mismatch), `resume=<run_dir>` for the same run,
`select` in {best, last, reward}, end-of-run sampling and benchmark on `bench_tier`, and a
registry line. `python -m blockgen.train --sample <run_dir> --n 399` writes the structure
cache the benchmark ranks. Details: [Training pipeline](pipeline.md) §5.

**Legacy trainers** (`blockgen/training/`), one per family, still the only route for
Tracks B, C, F and conditioned models:

| Module | Trains | Notes |
|---|---|---|
| `train_ar.py`, `train_ar_ext.py` | flat AR, any token stream | `train_from_sequences(..., pe=...)` selects AR2; cluster-BPE and category-token conditioning |
| `train_diffusion.py` | `VoxelUNet3D` | `calibrate_air_bias`, the most fragile knob in the codebase |
| `train_twostage.py` | occupancy then materials | did not beat single-stage (T11) |
| `train_graph.py` | graph VAE | recon CE + KL |
| `train_conditioned.py` | `CondVoxelAR2` | consumes the labeling pipeline's embeddings |
| `train_pick_n_place.py` | `PickAndPlace` | reports `place_lift` against the legal-face chance baseline |
| `constrained_decode.py` | inference only | in-loop 6-adjacency constrained sampling, validity 1.0 by construction (T11) |

**Post-training** (`blockgen/posttrain/`): `python -m blockgen.posttrain.bon --run <run>
--n 256 --k 8 --reward face_solidity` samples `n*k`, keeps the top of each group by a
reward and writes an arm with `trained_on_signals`; `python -m blockgen.posttrain.rft
--run <run> --reward lcc26 --n 2048 --keep-frac 0.25` writes the kept builds as a derived
tier (`blockgen/corpus/shards.py::from_structures`) and prints the fine-tuning command.
`rewards.py` refuses any reward that is a BlockScore pillar input
([Training pipeline](pipeline.md) §7).

## Evaluation (`blockgen/eval/`)

`novelty.py::evaluate_novelty` is the shared novelty report (`mean_nn_iou`,
`duplicate_rate`, `diversity`, `validity_rate`, `block_agreement`, comparison grids);
`validity.py` supplies `n_components`, `repair_lcc` and `gated_sample`;
`traversal.py::traversal_stats` walks a two-block player through a build (`enterable`,
standing reached, floors reached; T63); `facade_stats.py::face_stats` measures relief
jumps, holes, lone cells and dust on the six faces.

`blockgen/eval/bench/` is the benchmark ([Evaluation suite](benchmark.md)). The legacy
eval modules are never modified, so T1 to T22 stay reproducible.

| module | role |
|---|---|
| `splits.py` | the pinned group-aware split |
| `topology.py` | connectivity, floating mass, sealed interior, as distance to real |
| `geometry.py` | the 80-dim D4-invariant descriptor, `geom_kid`, geometry scalars; the only metric that sees interiors |
| `palette.py` / `stats.py` | palette JSD; bootstrap and subsample CIs |
| `features.py` / `distances.py` | render + DINOv2 cache; KID, MMD, FD, PRDC |
| `pointcloud.py` | the volumetric row: MMD, coverage, 1-NNA under chamfer and IoU (T67) |
| `probes.py` / `ladder.py` | corruption rungs and the gate a metric must pass before it ranks |
| `compare.py` | paired head-to-head tests, Holm-corrected |
| `composite.py` | `BlockScore`: worst pillar in real-sample spreads, with DQ gates |
| `protocol.py` | `houses32-v1`: tier both, min n 128, the descriptor pin, `train_split` rules |
| `fast.py` / `full.py` / `runner.py` | the two tiers and the CLI `python -m blockgen.eval.bench` |

## Rendering (`blockgen/renderer/`)

`textured.py` is the headless EGL/pyrender renderer with real block textures
(`textures.py` maps tokens to tiles); `grid.py::save_grid` draws mosaic sheets and falls
back to matplotlib voxels without EGL. Never import it beside osmesa/pyrender-CPU in one
process, which is why MinecraftACE sample rendering happens on the blockgen side.
`blockgen/labeling/render_views.py` writes the 4-view export for captioning and DINOv2.
`legogen/renderer/` renders LDraw with Blender.

## Config system

Two loaders. `blockgen/config.py` with `configs/{datasets,experiments,models,training,agentic}`
is the legacy layered YAML (`extends:` chains flattened onto a battery's argparse
destinations). `blockgen/train/config.py::TrainConfig` is the flat typed config of the
shared trainer, loaded from `configs/train/*.yaml` with `extends:` and `key=value`
overrides; an unknown YAML key raises. Tier definitions live in `configs/corpus/*.yaml`
(`configs/corpus/README.md`).

## Where a new model plugs in

The runnable version is [Bring your own model](custom-model.md). The short form:

1. Write `blockgen/models/<name>.py` to the contract above and register an `Arch` in
   `blockgen/train/archs.py`.
2. `python -m blockgen.train --config configs/train/smoke.yaml arch=<name>` (60 steps;
   the run dir must contain `run.json`, `metrics.jsonl`, `samples.png`, `summary.json`,
   `bench/scorecard.json`).
3. Train from `configs/train/houses32-sft-5m.yaml` or from scratch; pin `tokenizer:`.
4. `python -m blockgen.train --sample <run> --n 399`, then
   `python -m blockgen.eval.bench --tier both --n 399 --name "..." --arms ar/<name>:<npz>`.
5. `python -m tools.lab --open` and read `/leaderboard`; record the run in `results.md`
   and `notes.md`, which remain the research record the docs summarise.

A new decode-time constraint belongs beside `constrained_decode.py` for the legacy
samplers; `blockgen/train/sample.py` has no logits-mask hook yet. A new token kind touches
`blockgen/corpus/tokenizer.py` (`KINDS`), `shards.py`, `blockgen/train/config.py`,
`trainer.py` and `sample.py`.
