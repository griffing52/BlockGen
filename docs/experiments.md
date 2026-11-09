# Experiments & outputs

Two generations of run driver write under `outputs/run_<stamp>_<name>/` (gitignored).
The legacy batteries, `blockgen.experiments_*`, are self-contained: each writes a config
manifest (git sha + all train configs), figures, per-arm `novelty.json`, samples, a
`leaderboard.md` and a `run_notes.md`, and resumes from the per-arm `novelty.json`. From
T57 onward every neural arm goes through the shared trainer, `python -m blockgen.train`,
whose run directory is described below and whose samples are scored by
`python -m blockgen.eval.bench` like any other arm ([Pipeline](pipeline.md),
[Evaluation suite](benchmark.md)).

## Current batteries (the ones behind the headline results)

| Runner | What it establishes | Results |
|---|---|---|
| `experiments_ideas` | PE variants · BFS/raster ± adjacency constraint · samplers · two-stage | T11, phase4 + constrained decoding |
| `experiments_overnight` | 3 datasets × 6 tracks: conditioning, 3D-BPE, more-data | T10, conditioning is the best lever |
| `experiments_p4c` | phase4 × constrained combo (the two proven winners) | do-not-stack check |
| `experiments_transfer` | pool-pretrain → finetune on `houses_32` | T12, no in-domain transfer gain |
| `experiments_agentic` | Track E: scaffolding arms (plan / example / repair / critique) × prompt sets | T22, programs beat per-voxel tokens for scale |
| `blockgen.agentic` (recipe harness) | staged, dream-loop and describe methods on the `houses32-v1` prompts | T41 to T51, dream ties compute-matched classic |
| `blockgen.train` with `scripts/m2_seeds.sh`, `scripts/m2b_seeds.sh` | matched-compute scratch vs pretrain on `anneal` then SFT, three seeds, 5M | T57 to T60, pretraining wins every seed |
| `blockgen.train` with `configs/train/houses32-facades*.yaml` | the facade track: per-face and joint layouts, 5M and 20M, D4 augmentation | T65 to T68, the facade oracle and the complete card |
| `scripts/posttrain_first_pass.sh` | best-of-N and rejection-sampling FT on the two models that exist | voxel card `run_20261031_031101_bench_voxel_post-training_best-of-4_and` landed 2026-10-31; the facade half stopped after its best-of-4 step and has no card yet; when it lands it follows the same `outputs/run_*_bench_*post-training*` pattern |

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
.venv/bin/python -m blockgen.experiments_ideas    --stamp $STAMP      # PE/order/sampler/2-stage ablations
.venv/bin/python -m blockgen.experiments_overnight --stamp $STAMP     # cohesion + data battery
.venv/bin/python -m blockgen.experiments_p4c      --stamp $STAMP      # phase4 + constrained
.venv/bin/python -m blockgen.experiments_transfer --stamp $STAMP      # cross-medium transfer
.venv/bin/python -m blockgen.experiments_transfer --stamp smoke --quick   # fast wiring check

# Track E (agentic): no GPU, no stamp; it makes its own run dir and costs API tokens
.venv/bin/python -m blockgen.experiments_agentic --config agentic-scaffolding
.venv/bin/python -m blockgen.experiments_agentic --quick --provider mock   # offline wiring check
```

The agentic battery is the one runner that does **not** train anything: it spends API
calls instead of GPU hours, caches every response on disk so re-runs are free and
deterministic, and reports cost per arm. See [Agentic generation](agentic.md).

Parameterized runs read `configs/{datasets,experiments,models,training}/*.yaml` via
`blockgen/config.py` (e.g. `--config ideas-full`). Launch helpers: `scripts/run_ideas.sh`,
`scripts/{pause,resume}_ideas.sh`, `scripts/run_overnight.sh`, these poll `nvidia-smi`
until enough VRAM is free (`FREE_MB`, default 2500), launch under `nohup`, and persist
the exact invocation to `cmd.txt`; pause/resume kills to free the GPU and replays the
same stamp (finished arms are skipped via their on-disk `novelty.json`).

## The shared protocol (what makes a number "honest")

Every battery arm runs the same prep → eval discipline; when you read a
`leaderboard.md`, this is what produced each row:

1. **Dedup** near-identical builds (occupancy IoU ≥ 0.95, variant-keeping).
2. **Held-out val split** (~15%), taken **before** augmentation.
3. **D4 augmentation** (8 symmetries), train split only.
4. **Canonicalize** (default `canon_dim=16` miniatures for AR; native grid for
   diffusion) and serialize per track.
5. Train → sample → evaluate against **train** (`nn_iou`, `duplicate_rate`, the
   memorization check) and against **val** (`val_nn`, the generalization number),
   next to a **real-build baseline** (what a real held-out house scores vs train:
   ≈0.48 houses, ≈0.64 vehicles). Read `val_nn` as a fraction of that ceiling.
6. Validity is reported raw and (where noted) after the LCC-repair or
   adjacency-constraint gate.

Each arm is crash-guarded (`ERROR.txt`/`SKIP.txt` and the battery continues), and its
`novelty.json` doubles as the resume marker.

## Battery anatomy: every arm, what it tests

### `experiments_overnight` (→ T10): datasets × methods

3 dataset builders: `gc-houses-large` (GrabCraft houses, footprint ≥16, ≥60 blocks),
`combined-houses` (GrabCraft + labeled-crawl houses), `gc-vehicles` (GrabCraft
transportation, ≥55 blocks), each × 6 methods:

| Arm | What it is | What it established (T10) |
|---|---|---|
| `ar_flat` | baseline flat-token AR | more data → raw validity 3× (0.188 → 0.562 gc → combined) |
| `ar_cluster` | 3D-BPE piece tokens | anti-memorization: dup 0 on vehicles where flat hits 0.125 |
| `ar_cluster_hi20` | BPE at higher canonical res | memorizes harder on small homogeneous data (dup 0.188) |
| `ar_conditioned` | category-token prefix AR (`[BOS, CAT, …]`) | **best all-around**: val_nn 0.435 = 93% of baseline |
| `diffusion32` | native-grid UNet | air-bias miscalibration at 32³ (occ 2577 vs 1461 target) |
| `graph_vae` | graph latent VAE | trails AR but never memorizes; scales with data |

Defaults: `epochs_ar=90, epochs_diff=130, epochs_graph=60, ar_seq=1600, n_merges=250`.

### `experiments_ideas` (→ T11): quality-lever ablations on `gc-houses-large`

Four groups, one dataset, resumable per-arm:

- **A. Positional encoding** (flat AR2): `ar_pe_{learned,sin,rope,alibi,phase4}`.
  Winner: **phase4** (grammar-aware `pos%4` phase + `pos//4` block-index embeddings,
  mechanics in [Architecture](architecture.md#voxeltransformerar2-and-the-phase4-pe)),
  val_nn 0.405. RoPE ≈ learned; ALiBi worst.
- **B. Ordering × constrained decoding**: `ar_bfs`, `ar_bfs_constrained`,
  `ar_raster_constrained`. Winner: **raster + in-loop 6-adjacency constraint**,
  validity 1.0 by construction at ~no quality cost; BFS ordering *hurts*.
- **C. Diffusion samplers** (one trained 32³ net, four inference rules):
  `diff32_{maskgit,flow,remask,stratified}`. Stratified best, remask broken; no
  sampler rescues an uncalibrated net.
- **D. Two-stage** (`twostage32`): occupancy diffusion → material diffusion. Best
  diffusion *shape* but ~9× occupancy overfill and near-random materials.

### `experiments_p4c`: composing the two T11 winners

No training, loads the ideas battery's trained `ar_pe_phase4` checkpoint, samples
**with** the adjacency constraint, writes into the existing ideas run dir. Result:
constrained-phase4 ≈ constrained-raster (val_nn 0.382 vs 0.383, both validity 1.0),
**the winners don't stack**; the constraint absorbs the phase4 edge.

### `experiments_transfer` (→ T12): pool-pretrain → finetune

Arms: `scratch` (houses_32 only), `pretrain` (16,383-build / 113k-sequence pool of
all five corpora, houses val excluded by `source_path`; + zero-shot eval),
`finetune` (pretrain checkpoint → houses at 0.3× lr). Every arm evaluated plain and
adjacency-constrained; shared legacy vocab. Result: **no in-domain transfer gain**
(finetune 0.379 vs scratch 0.405 despite 22% lower train loss), compression ≠
generation; pool pretraining is re-scoped to the data-poor LEGO regime.

## Reading a run directory

```
outputs/run_<stamp>_<battery>/
  config.json / cmd.txt          # git sha, resolved config, exact invocation
  <dataset>/<arm>/
    novelty.json                 # the metrics row (also the resume marker)
    samples/ · figures · ckpt    # arm artifacts (what render_*_samples.py reloads)
    ERROR.txt | SKIP.txt         # only if the arm failed/was skipped
  leaderboard.md                 # cross-arm table
  run_notes.md
```

`novelty.json` fields follow the [metrics glossary](models.md#evaluation-blockgenevalnoveltypy)
plus `val_nn` and the baseline rows described above.

### A trainer run directory

```
outputs/run_<stamp>_<name>/             # python -m blockgen.train --config ... name=<name>
  run.json                              # the resolved TrainConfig plus meta (git sha, tiers, tokenizer hash)
  metrics.jsonl                         # one line per log step
  last.pt  best.pt  ckpt_<step>.pt      # step zero-padded to seven digits, ckpt_0004000.pt; best = lowest holdout NLL on the first val tier (select=best)
  samples.png                           # the rendered end-of-run sample grid (every run renders one)
  samples_step<step>_<dim>.npz + _manifest.json   # the structure cache the bench scored; report.train_split set
  bench/scorecard.json, scorecard.md, examples_<dim>.npz   # the fast-tier card at bench_n (bench_tier in the config)
  summary.json                          # the registry record, also appended to outputs/registry.jsonl
  samples<N>/, bon_<reward>_k<k>_n<n>/  # later: --sample --out-dir and blockgen.posttrain.bon outputs
```

`outputs/registry.jsonl` holds one line per finished run (name, stage, arch, preset, params,
tiers and their hashes, `train_split`, `trained_on_signals`, `val_nll`, `val_nll_all`,
`eval_seqs`, the selected checkpoint, the bench result and the card path). The lab joins
`run.json`, `summary.json` and that line onto the run's `bench/` card
(`tools/lab/catalog.train_run`). At the recorded settings (`bench_tier: fast` from
`blockgen/train/config.py:72`, `bench_n: 64` from `configs/train/base.yaml:16`) a trainer card
fails `houses32-v1` on tier and on n, so it is never ranked; the board entry is a separate
`--sample --n 399` plus `python -m blockgen.eval.bench --tier both`.

## T-series index (which run produced which result)

Full tables live in the root `results.md` and `notes.md`
(the living records). Map from result id → what/where:

| T# | What | Runner / location | Run dir |
|---|---|---|---|
| T4–T5 | three-track novelty on curated subsets | `experiments` | `run_20260630_220830` |
| T6 | AR vs diffusion vs flow, filmstrips | `experiments_gen` | `run_20260630_232913_gen` |
| T7 | embedding analysis (birch≉oak) | `experiments_analysis` | `outputs/analysis/` |
| T8–T9 | GrabCraft categories; dedup kills memorization | `experiments_grabcraft` | `run_20260701_*` |
| T10 | cohesion + data battery (18 arms) | `experiments_overnight` | `run_20260702_022207_overnight` |
| T11 | PE/ordering/sampler/two-stage ablations | `experiments_ideas` | `run_20260704_001222_ideas` |
| (T11) | phase4 × constraint combo | `experiments_p4c` | (writes into the T11 dir) |
| T12 | pool-pretrain → finetune | `experiments_transfer` | `run_20260708_070807_transfer` |
| T13–T14 | 92M GPT-2 port, voxel vs 3D-BPE | [MinecraftACE](minecraftace.md) | `outputs/minecraftace/` |
| T15–T16 | image/text conditioning + CFG | `train_conditioned` ([docs](minecraftace.md#conditioning-t15t16-the-recipe-re-hosted-on-blockgen-models)) | `outputs/cond/` |
| T17 | pipeline audit: the canon-16 decimator destroys the training target | `scripts/render_decimation.py` | analysis only |
| T18 | 3D-BPE vocabulary audit: pieces are material patches, not parts | tokenizer audit, `notes.md` | analysis only |
| T19 | minecraft-schematics.com label recovery | `blockgen.data.recover_ms_labels` | `data/raw` |
| T20 | native resolution: the decimator thesis is not supported | `blockgen.experiments_native` | `run_20260715_065938_native` |
| T21 | attachment/growth phase 0 and 1: representation works, generation does not | `scripts/train_attach_corpus.py`, `scripts/ordering_bakeoff.py`, `scripts/attach_*.py` | `run_20260721_071702_attach_growth_ordering`, `run_20260721_074238_attach_growth_seq8192`, `run_20260721_080100_attach_growth_seq16384` |
| T22 | agentic track, first live measurements | `blockgen.agentic`, `scripts/run_agentic.py` | `run_20260728_050025_agentic_live_smoke`, `run_20260728_084343_agentic_showcase_detailed`, `run_20260728_084550_agentic_showcase_short` |
| T23 | the evaluation suite, validated cross-track | `blockgen.eval.bench`, `blockgen.eval.bench.ladder` | `run_20260803_033522_bench_compare`, `run_20260816_081955_bench_pnp` |
| T24 | pick-and-place: learned placement | `scripts/train_pick_n_place.py` | `run_20260816_081506_pnp_fixed` |
| T25 | the realism metric was structurally blind | `blockgen.eval.bench` (`probes.solidify`), `bench.ladder` | `run_20260831_*_bench` |
| T26 | dose-response, baselines, humans | `scripts/bench_doseresponse.py`, `scripts/human_study_export.py`, `scripts/human_study_analyze.py` | `run_20260831_*_bench`, `run_20260906_*_bench` |
| T27 | block ontology, a measured material reference | `python -m blockgen.ontology` | `data/ontology/*.json` |
| T28 | attention as adjacency: dominated by a subtraction | `scripts/probe_llm_attention.py` | `run_20260906_221420_attn_adjacency_probe` |
| T29 | geometry injected into attention: a null | `scripts/train_llm_geobias.py`, `scripts/sample_geobias.py`, `scripts/plot_geobias.py` | `run_20260906_222749_llm_geobias_control`, `run_20260906_232253_llm_geobias_bias`, `run_20260907_220119_bench_LLM_geometric_attention_bias_control_vs` |
| T31 | `native_oriented` at n=128: not a rankable number | `scripts/sample_to_npz.py`, `blockgen.eval.bench` | `run_20260910_183610_bench_native_oriented_at_n_128_vs_LLM_arms`, `run_20260916_233528_bench_native_oriented_at_n_128_vs_LLM_arms` |
| T32 | the clean-split retrain: the margin was the contamination | `blockgen.eval.bench.splits`, `scripts/sample_to_npz.py --train-split canonical` | `run_20260917_000431_canon_{flat,oriented,unoriented}_native`, `run_20260917_043807_bench_canonical-split_native_arms_oriented_vs` |
| T33 | what the interior probe applies to, across structure types | `blockgen.eval.bench.probes`, `blockgen.eval.bench.geometry` | analysis over `data/lego/stabletext2brick` and the corpora |
| T34 | LEGO replication of the blindness | `blockgen.eval.bench.lego` | `run_20260918_200852_bench_lego_interior_blindness_replication` |
| T35 | the noise-floor unit is n-dependent | `blockgen.eval.bench.ladder` | `outputs/analysis/ladder/` |
| T36, T37 | the blindness is not exact render invariance; the cause is the view protocol | `scripts/ablate_view_protocol.py` | `run_20260918_205019_view_protocol_ablation` |
| T38 | the residual is wobble, not partial detection | `scripts/figs/rescore_flat_opaque.py` | `outputs/figstage/rescore_flat_opaque.json` |
| T39 | the descriptor is not a size detector | `scripts/vkid_ablation.py` | `outputs/figstage/vkid_ablation.json` |
| T40 | the rater study: three pre-registered predictions hold | `scripts/human_study_serve.py`, `scripts/human_study_analyze.py` | `run_20260923_human_v5` |
| T41 to T50 | agentic methods suite, staged and dream methods, local Gemma, attention-as-adjacency revisited, nodes-to-adjacency | `blockgen.agentic`, `scripts/probe_llm_attention.py`, `tools/misc/adjacency_gen/` | `run_202610{03,04,05,06,07}_*` as named in each entry |
| T51 (world crops) | world-crop pilot: 400 worlds to 2,651 crops | `blockgen.data.worldcrop`, `scripts/worldcrop_ingest.sh` | `data/minecraft/worldcrop/201407/` |
| T51 (agentic) | dream vs compute-matched classic, dream-caption, on `houses32-v1` | `blockgen.agentic`, `blockgen.eval.bench` | `run_20261007_092654_bench_houses32_{classic,dream}`, `run_20261007_145754_bench_houses32_dream_caption`, `run_20261007_185345_bench_agentic_dream_vs_compute-matched` |
| T52, T53 | face nodes and the merge decoder; cell assignment then connections | `tools/misc/adjacency_gen/` | `run_20261007_1716*_adjacency_faces_*`, `run_20261008_0*_adjacency_cells_*` |
| T54 | world-crop terrain gate, four rounds | `blockgen/data/worldcrop/gate.py` | `data/minecraft/worldcrop/` |
| T55 | corpus manifest and dedup over 926k builds | `blockgen.corpus.manifest`, `blockgen.corpus.dedup` | `data/minecraft/corpus/` |
| T56 | structure templates from the client jar | `blockgen.corpus.templates` | `data/minecraft/templates/` |
| T57 | first runs through the shared trainer | `blockgen.train`, `configs/train/smoke.yaml`, `houses32-scratch.yaml` | `run_20261014_101448_houses32_scratch_5m` |
| T58 | pretrain on `anneal` then SFT against scratch, 5M | `blockgen.train`, `anneal-pretrain-5m.yaml`, `houses32-sft-5m.yaml` | `run_20261014_110921_houses32_sft_5m`, `run_20261014_113512_houses32_scratch_5m_v2`, `run_20261014_115238_houses32_sft_5m_v2` |
| T59 | M2 pilot, three seeds: a dropout fault and the protocol that replaces it | `scripts/m2_seeds.sh` | `run_20261014_*_m2_{scratch,pretrain,sft}_s{0,1,2}` |
| T60 | M2b, three seeds, early stopped: pretraining wins every seed | `scripts/m2b_seeds.sh` | `run_20261015_225212_m2b_sft_s0` through `run_20261016_013050_m2b_scratch_s2` |
| T61, T62, T69 | BrickNet Table 2 reproduction; BrickAgent/BrickBench; BrickNet recipe on Minecraft walks | `libs/BrickNet` (`libs/BrickNet/repro/mc_to_bricknet.py`), `libs/BrickAgent` | `outputs/figures/bricknet_repro/` |
| T63 | traversability: can a player walk into it? | `scripts/bench_traversal.py`, `blockgen.eval.traversal` | `run_20261023_004317_bench_traversal` |
| T64 | leaderboard re-scored with room scalars and a traversal pillar | `blockgen.eval.bench` | `run_20261023_011035_bench_traversal_pillar_leaderboard_re-score` |
| T65 | the facade oracle: real houses reduced to their visible shell | `scripts/facade_oracle.py`, `blockgen.corpus.facade` | `run_20261023_204241_bench_facade_oracle_real_houses_reduced_to` |
| T66 | the facade track: baselines, two facade models, the ceiling | `scripts/facade_baselines.py`, `blockgen.train` with `houses32-facades.yaml`, `houses32-facades-joint.yaml` | `run_20261026_153838_houses32_facades_5m`, `run_20261026_155541_houses32_facades_joint_5m`, `run_20261026_160934_bench_facade_track_baselines_the_per-face_and` |
| T67 | the volumetric row: MMD, coverage, 1-NNA on the ladder | `blockgen.eval.bench.pointcloud` | `outputs/analysis/ladder/pointcloud__k1024_canvas32.json` |
| T68 | the facade track at scale: 20M, D4, the complete card | `blockgen.train` with `houses32-facades-joint-20m.yaml`, `houses32-facades-joint-d4*.yaml`, `scripts/facade_sheets.py` | `run_20261028_110415_houses32_facades_joint_20m`, `run_20261028_123627_houses32_facades_joint_d4_5m`, `run_20261028_125240_houses32_facades_joint_d4_20m`, `run_20261028_143416_bench_facade_track_baselines_five_facade` |

`results.md` carries two entries numbered T51 (the world-crop pilot and the agentic
bench); the table keeps both under that number.

### Rendering model samples

Fresh samples from saved checkpoints, rendered with the real-texture pipeline (rebuilds
each run's exact prep/vocab, reloads checkpoints, samples, renders textured grids):

```bash
.venv/bin/python scripts/render_model_samples.py    --samples 48   # ideas-battery checkpoints
.venv/bin/python scripts/render_transfer_samples.py --samples 48   # transfer-run checkpoints
```

Outputs land in `outputs/figures/` (e.g. `samples_ar_raster_constrained.png`,
`samples_ar_pe_phase4.png`, `samples_transfer_finetune.png`).

### LEGO dataset sheets

LDraw/OMR structures render via a separate Blender 4.2 package (`legogen/renderer/`):

```bash
.venv/bin/python legogen/renderer/ldraw_grid.py --rows 8 --cols 12 --band 15 160 \
    --out outputs/figures/lego_data/omr_grid.png
```

---

## Legacy runners (early single-track studies)

The following predate the batteries above but remain reproducible and back some
`results.md` milestones.

## `blockgen.experiments`: subsets + retraining + novelty

```bash
.venv/bin/python -m blockgen.experiments --stamp $(date +%Y%m%d_%H%M%S) \
    --epochs-ar 100 --epochs-diff 150 --epochs-graph 100 --samples 16
```

Trains only the **feasible** (subset, track) pairings (dense builds → diffusion; compact
builds → all tracks), then evaluates each model against the subset it trained on.

```
outputs/run_<stamp>/
  config.json                     # git sha, subset filters, epochs
  figures/subset_*.png            # curated subset contact sheets
  figures/variant_group_*.png     # same shape, different materials
  models/<subset>__<track>/
    config.json  history.json  loss.png  samples.png  comparison.png  novelty.json
  metrics.md                      # cross-model table
  run_notes.md
```

Flags: `--figures-only` (skip training), `--quick` (2 epochs, timing probe).

## `blockgen.experiments_gen`: generation process + AR-vs-diffusion + flow matching

```bash
.venv/bin/python -m blockgen.experiments_gen --stamp $(date +%Y%m%d_%H%M%S) \
    --canon-dim 12 --ar-seq 1600 --epochs-diff24 150 --epochs-ar 80 --epochs-diff-canon 120
```

- **G1**, trains one diffusion net on 24³ houses, then films **MaskGIT** vs **flow
  matching** generation and compares the two samplers.
- **G2**, scale-normalizes houses to a canonical `N³`, then **races AR vs diffusion** on
  the identical set and films AR's progressive build.

```
outputs/run_<stamp>_gen/
  film_diffusion_maskgit.png  film_diffusion_flow.png   # generation filmstrips
  film_ar_progressive.png     compare_ar_vs_diffusion.png
  diffusion_24.pt  ar_12.pt  canon_diffusion_12.pt       # checkpoints (re-sample without retraining)
  metrics.md  run_notes.md
```

## `blockgen.experiments_grabcraft`: best models on a GrabCraft category

```bash
.venv/bin/python -m blockgen.experiments_grabcraft --stamp $(date +%Y%m%d_%H%M%S) \
    --category medieval-houses --canon-dim 12 --ar-seq 1600 \
    --epochs-ar 100 --epochs-diff24 150 --epochs-diff-canon 120
```

Trains our best models (AR on canonical builds + diffusion at 24³/N³) on one clean
GrabCraft category and renders a data sheet, generated-sample sheets, novelty grids, and
generation filmstrips under `outputs/run_<stamp>_grabcraft/`.

## `blockgen.experiments_analysis`: representation & embedding figures

```bash
.venv/bin/python -m blockgen.experiments_analysis   # -> outputs/analysis/*.png
```

Emits `tokenization_methods.png` (token stream vs grid vs graph, and where air lives) and
`embedding_analysis.png` (wood/wool embedding similarity from a saved AR checkpoint).

## Reproducibility notes

- The legacy batteries in this page were run with `.venv/bin/python`, and the command
  blocks above keep that. Everything from the corpus layer onward (`blockgen.corpus`,
  `blockgen.train`, `blockgen.posttrain`, `tools.lab`) runs with the conda `python` on
  `PATH`, which has torch and pandas; `.venv` has no pandas. See [Scripts](scripts.md).
- Runs are seeded only loosely; occupancy is sampling-sensitive, so treat single-run
  numbers as directional. The living records are `notes.md` (methods/ledger) and
  `results.md` (tables), at the repo root.
