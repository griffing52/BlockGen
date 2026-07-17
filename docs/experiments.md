# Experiments & outputs

Every `blockgen.experiments_*` module is a self-contained run driver that writes a run
directory under `outputs/run_<stamp>_<name>/` (gitignored) with a config manifest (git sha
+ all train configs), figures, per-arm `novelty.json`, samples, a `leaderboard.md`, and a
`run_notes.md`. All are **resumable** from their per-arm `novelty.json`.

## Current batteries (the ones behind the headline results)

| Runner | What it establishes | Results |
|---|---|---|
| `experiments_ideas` | PE variants · BFS/raster ± adjacency constraint · samplers · two-stage | T11 — phase4 + constrained decoding |
| `experiments_overnight` | 3 datasets × 6 tracks: conditioning, 3D-BPE, more-data | T10 — conditioning is the best lever |
| `experiments_p4c` | phase4 × constrained combo (the two proven winners) | do-not-stack check |
| `experiments_transfer` | pool-pretrain → finetune on `houses_32` | T12 — no in-domain transfer gain |

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
.venv/bin/python -m blockgen.experiments_ideas    --stamp $STAMP      # PE/order/sampler/2-stage ablations
.venv/bin/python -m blockgen.experiments_overnight --stamp $STAMP     # cohesion + data battery
.venv/bin/python -m blockgen.experiments_p4c      --stamp $STAMP      # phase4 + constrained
.venv/bin/python -m blockgen.experiments_transfer --stamp $STAMP      # cross-medium transfer
.venv/bin/python -m blockgen.experiments_transfer --stamp smoke --quick   # fast wiring check
```

Parameterized runs read `configs/{datasets,experiments,models,training}/*.yaml` via
`blockgen/config.py` (e.g. `--config ideas-full`). Launch helpers: `scripts/run_ideas.sh`,
`scripts/{pause,resume}_ideas.sh`, `scripts/run_overnight.sh` — these poll `nvidia-smi`
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
5. Train → sample → evaluate against **train** (`nn_iou`, `duplicate_rate` — the
   memorization check) and against **val** (`val_nn` — the generalization number),
   next to a **real-build baseline** (what a real held-out house scores vs train:
   ≈0.48 houses, ≈0.64 vehicles). Read `val_nn` as a fraction of that ceiling.
6. Validity is reported raw and (where noted) after the LCC-repair or
   adjacency-constraint gate.

Each arm is crash-guarded (`ERROR.txt`/`SKIP.txt` and the battery continues), and its
`novelty.json` doubles as the resume marker.

## Battery anatomy — every arm, what it tests

### `experiments_overnight` (→ T10): datasets × methods

3 dataset builders — `gc-houses-large` (GrabCraft houses, footprint ≥16, ≥60 blocks),
`combined-houses` (GrabCraft + labeled-crawl houses), `gc-vehicles` (GrabCraft
transportation, ≥55 blocks) — each × 6 methods:

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
  Winner: **phase4** (grammar-aware `pos%4` phase + `pos//4` block-index embeddings —
  mechanics in [Architecture](architecture.md#voxeltransformerar2-and-the-phase4-pe-the-best-arm)),
  val_nn 0.405. RoPE ≈ learned; ALiBi worst.
- **B. Ordering × constrained decoding**: `ar_bfs`, `ar_bfs_constrained`,
  `ar_raster_constrained`. Winner: **raster + in-loop 6-adjacency constraint** —
  validity 1.0 by construction at ~no quality cost; BFS ordering *hurts*.
- **C. Diffusion samplers** (one trained 32³ net, four inference rules):
  `diff32_{maskgit,flow,remask,stratified}`. Stratified best, remask broken; no
  sampler rescues an uncalibrated net.
- **D. Two-stage** (`twostage32`): occupancy diffusion → material diffusion. Best
  diffusion *shape* but ~9× occupancy overfill and near-random materials.

### `experiments_p4c`: composing the two T11 winners

No training — loads the ideas battery's trained `ar_pe_phase4` checkpoint, samples
**with** the adjacency constraint, writes into the existing ideas run dir. Result:
constrained-phase4 ≈ constrained-raster (val_nn 0.382 vs 0.383, both validity 1.0) —
**the winners don't stack**; the constraint absorbs the phase4 edge.

### `experiments_transfer` (→ T12): pool-pretrain → finetune

Arms: `scratch` (houses_32 only), `pretrain` (16,383-build / 113k-sequence pool of
all five corpora, houses val excluded by `source_path`; + zero-shot eval),
`finetune` (pretrain checkpoint → houses at 0.3× lr). Every arm evaluated plain and
adjacency-constrained; shared legacy vocab. Result: **no in-domain transfer gain**
(finetune 0.379 vs scratch 0.405 despite 22% lower train loss) — compression ≠
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

## T-series index (which run produced which result)

Full tables live in [Results](results.md) and the root `results.md`/`notes.md`
(the living records). Map from result id → what/where:

| T# | What | Runner / location | Run dir |
|---|---|---|---|
| T4–T5 | three-track novelty on curated subsets | `experiments` | `run_20260630_220830` |
| T6 | AR vs diffusion vs flow, filmstrips | `experiments_gen` | `run_20260630_232913_gen` |
| T7 | embedding analysis (birch≉oak) | `experiments_analysis` | `outputs/analysis/` |
| T8–T9 | GrabCraft categories; dedup kills memorization | `experiments_grabcraft` | `run_20260701_*` |
| T10 | cohesion + data battery (18 arms) | `experiments_overnight` | `run_20260702_022207_overnight` |
| T11 | PE/ordering/sampler/two-stage ablations | `experiments_ideas` | `run_20260704_001222_ideas` |
| — | phase4 × constraint combo | `experiments_p4c` | (writes into the T11 dir) |
| T12 | pool-pretrain → finetune | `experiments_transfer` | `run_20260708_070807_transfer` |
| T13–T14 | 92M GPT-2 port, voxel vs 3D-BPE | [MinecraftACE](minecraftace.md) | `outputs/minecraftace/` |
| T15–T16 | image/text conditioning + CFG | `train_conditioned` ([docs](minecraftace.md#conditioning-t15t16-the-recipe-re-hosted-on-blockgen-models)) | `outputs/cond/` |

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
[Results](results.md) milestones.

## `blockgen.experiments` — subsets + retraining + novelty

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

Flags: `--figures-only` (skip training), `--quick` (2 epochs — timing probe).

## `blockgen.experiments_gen` — generation process + AR-vs-diffusion + flow matching

```bash
.venv/bin/python -m blockgen.experiments_gen --stamp $(date +%Y%m%d_%H%M%S) \
    --canon-dim 12 --ar-seq 1600 --epochs-diff24 150 --epochs-ar 80 --epochs-diff-canon 120
```

- **G1** — trains one diffusion net on 24³ houses, then films **MaskGIT** vs **flow
  matching** generation and compares the two samplers.
- **G2** — scale-normalizes houses to a canonical `N³`, then **races AR vs diffusion** on
  the identical set and films AR's progressive build.

```
outputs/run_<stamp>_gen/
  film_diffusion_maskgit.png  film_diffusion_flow.png   # generation filmstrips
  film_ar_progressive.png     compare_ar_vs_diffusion.png
  diffusion_24.pt  ar_12.pt  canon_diffusion_12.pt       # checkpoints (re-sample without retraining)
  metrics.md  run_notes.md
```

## `blockgen.experiments_grabcraft` — best models on a GrabCraft category

```bash
.venv/bin/python -m blockgen.experiments_grabcraft --stamp $(date +%Y%m%d_%H%M%S) \
    --category medieval-houses --canon-dim 12 --ar-seq 1600 \
    --epochs-ar 100 --epochs-diff24 150 --epochs-diff-canon 120
```

Trains our best models (AR on canonical builds + diffusion at 24³/N³) on one clean
GrabCraft category and renders a data sheet, generated-sample sheets, novelty grids, and
generation filmstrips under `outputs/run_<stamp>_grabcraft/`.

## `blockgen.experiments_analysis` — representation & embedding figures

```bash
.venv/bin/python -m blockgen.experiments_analysis   # -> outputs/analysis/*.png
```

Emits `tokenization_methods.png` (token stream vs grid vs graph, and where air lives) and
`embedding_analysis.png` (wood/wool embedding similarity from a saved AR checkpoint).

## Reproducibility notes

- Always use `.venv/bin/python`.
- Runs are seeded only loosely; occupancy is sampling-sensitive, so treat single-run
  numbers as directional. The living records are `notes.md` (methods/ledger) and
  `results.md` (tables), at the repo root.
