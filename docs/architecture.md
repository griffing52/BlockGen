# Architecture — the codebase map

This page is the operator's map of the whole repo: how data flows from raw corpora to
figures, what every package does, and where to plug in your own models and experiments.
Concepts (why three tracks, what the metrics mean) live in
[Models](models.md)/[Representations](representations.md); this page is about the code.

## The pipeline at a glance

```
raw corpora ──► caches (npz) ──► curation ──► prep protocol ──► tokenize ──► train
 (5 sources)     data/cache/      Curator /     dedup → split     one of 4     one of 7
                                  houses.py     → augment →       reprs        trainers
                                                canonicalize
                                                                      │
      figures ◄── render ◄── evaluate ◄── sample ◄───────────────────┘
      outputs/    renderer/   eval/        per-model samplers
                  grid.py     novelty.py   (+ constrained_decode)
```

Every experiment battery (`blockgen/experiments_*.py`) is a driver that walks this
pipeline end-to-end for a set of *arms* and writes one run directory — see
[Experiments](experiments.md) for the batteries and their arms.

## Package layout

| Path | Role |
|---|---|
| `blockgen/data/` | raw-corpus → cache builders (schematics, tfrecords, GrabCraft scrape) |
| `blockgen/curation/` | interactive `Curator` + the cross-corpus house cache builder |
| `blockgen/utils/` | `Structure`, corpora loaders, block remapping, orderings |
| `blockgen/tokenizers/` | standard block vocab, per-voxel serialization, 3D-BPE |
| `blockgen/models/` | the model zoo (below) |
| `blockgen/training/` | one trainer module per model family + constrained decoding |
| `blockgen/eval/` | novelty report + connectivity/validity |
| `blockgen/labeling/` | renders → captions → frozen embeddings ([Labeling](labeling.md)) |
| `blockgen/export/` | converters to external layouts ([MinecraftACE](minecraftace.md)) |
| `blockgen/renderer/` | textured EGL renderer + mosaic grids |
| `blockgen/experiments_*.py` | the run drivers / batteries |
| `blockgen/config.py` + `configs/` | layered YAML config system |
| `scripts/` | launch helpers + figure/render scripts |
| `libs/` | vendored MinecraftACE / LegoACE (own docs page) |
| `legogen/` | Blender 4.2 LDraw renderer for the LEGO corpora |
| `outputs/` | all run artifacts (gitignored) |

## Data layer

- **Caches** (`data/cache/*.npz`): fixed-max-dim arrays of `(block_ids, block_data)`
  volumes + a `_meta.json` sidecar. Builders: `data/tfrecord_dataset.py` (labeled
  PlanetMinecraft crawl — the preferred cache, 100% metadata),
  `data/build_cache.py` (legacy `data/raw` schematics), `data/grabcraft_dataset.py`
  (scraped GrabCraft JSON with exact `(id,data)` and categories).
- **External corpora** (`utils/corpora.py`): `load_3dcraft` (2.5k houses + human
  placement-order traces), `load_text2mc` (11k builds, remapped via
  `utils/block_remap.py`).
- **Curation**: `curation/curate.py::Curator` is the interactive filter/search tool
  (see [Data & curation](data-and-curation.md)); `curation/houses.py::
  load_house_structures(max_dim=…)` is the *programmatic* entry point every current
  experiment uses — the pooled, quality-gated, cross-corpus house cache
  (`houses_32` = 2,661 builds; enclosed-air "house-ness" gate, corpus-calibrated).

## Representation layer

`utils/data.py::Structure` is the universal unit: paired `block_ids`/`block_data`
int arrays + metadata, with `crop_to_non_air()`, `occupied_mask`, D4 symmetry ops.
Four interchangeable representations feed the models:

1. **Flat token stream** (`tokenizers/serialize.py`): `[BOS,(X,Y,Z,BLOCK)*,EOS]`,
   coordinates as tokens, raster order `(y,z,x)` → bottom-up builds. `BlockVocab`
   built by `build_block_vocab`; `STANDARD_VOCAB` (~441 entries) names the classes.
2. **3D-BPE piece stream** (`tokenizers/cluster_bpe.py`): `learn_clusters` greedily
   merges frequent adjacent-voxel patterns into multi-voxel *pieces*
   (`ClusterVocab`); `build_cluster_sequences` re-serializes structures as piece
   placements. ~2× shorter sequences, anti-memorization (T10), LegoACE brick-library
   analog.
3. **Canonical grid**: scale-normalization (downsample to `N³`) for the diffusion
   track and for the "miniature" AR regimes (canon-12/16 — where all the best
   results live).
4. **Block+port graph**: PyG graphs for the graph-VAE track.

## Model zoo (`blockgen/models/`)

| Class | File | Params | What it is |
|---|---|---|---|
| `VoxelTransformerAR` | `voxel_transformer_ar.py` | ~5–6M | baseline causal transformer, learned absolute PE |
| `VoxelTransformerAR2` | `voxel_transformer_ar2.py` | ~5–6M | PE-pluggable rewrite — **the current flagship** |
| `CondVoxelAR2` | `voxel_transformer_cond.py` | ~6M | AR2 + prefix conditioning + CFG (T15/T16) |
| `VoxelUNet3D` | `voxel_diffusion.py` | ~5–15M | masked discrete diffusion 3D-UNet, 4 samplers |
| `LargePyGGraphGenerator` | `large_pyg_graph_generator.py` | few M | graph VAE (TransformerConv → latent → GRU decoder) |
| `VoxelPortGNN` | `voxel_port_gnn.py` | ~1–2M | starter block+port GNN (legacy) |

Shared AR defaults (`ARTrainConfig`): `d_model=256, nhead=8, num_layers=6,
dim_feedforward=1024, dropout=0.1`.

### `VoxelTransformerAR2` and the phase4 PE (the best arm)

AR2 hand-rolls its attention block on `F.scaled_dot_product_attention` (stock
`nn.TransformerEncoderLayer` exposes neither q/k rotation nor per-head biases), which
makes the positional encoding a constructor argument: `pe ∈ {learned, sin, rope,
alibi, phase4, none}`.

**phase4** is the grammar-aware scheme and the T11 ablation winner (val_nn 0.405 =
84% of the real-build baseline). Instead of one embedding per absolute position, it
decomposes the position of each token in the `(X,Y,Z,BLOCK)` stream into:

- a **phase embedding** `nn.Embedding(4, d_model)` indexed by `pos % 4` — *which slot
  of the record am I* (X, Y, Z, or BLOCK);
- a **block-index embedding** `nn.Embedding(max_seq/4+1, d_model)` indexed by
  `pos // 4` — *which record am I in*.

The period-4 grammar is thereby *given* to the model rather than rediscovered, and
tail positions no longer starve (with plain learned PE, only the largest builds ever
reach the tail indices, so those rows train on a handful of examples). The ablation
also showed generic relative PE is **not** a win here (RoPE ≈ learned, ALiBi worst) —
our coordinates travel as tokens, not positions, so the PE's job is grammar, not
geometry. Motivation and the full ablation: `research.md` §B, `results.md` T11.

`CondVoxelAR2` subclasses AR2 (phase4) and adds LegoACE-style conditioning: frozen
DINOv2/CLIP embeddings → `cond_proj` linear → prefix tokens (+ prefix position
embeddings), a learned `null_cond` trained by 10% condition-dropout, and
classifier-free guidance in `generate_cond(cfg_scale=…)`.

### Diffusion track

`VoxelUNet3D` (compact 3D-UNet, `base_channels=48`, GroupNorm/SiLU, sinusoidal time
embedding) trained MaskGIT/D3PM-absorbing-style by `train_diffusion.py` (mask a
fraction of voxels, cross-entropy on the masked ones, air down-weighted). Four
samplers on one trained net: `sample_grids` (MaskGIT confidence order),
`sample_grids_flow` (discrete flow matching), `sample_grids_remask` (ReMDM-lite —
currently broken, T11), `sample_grids_stratified`. `calibrate_air_bias` fits the
per-sampler air bias — the single most fragile knob in the codebase (a scalar can't
fit mixed-size corpora; every bad diffusion result in the ledger traces back to it).

## Training layer (`blockgen/training/`)

| Module | Trains | Notes |
|---|---|---|
| `train_ar.py` | flat AR | `build_sequences` / `train_ar` / `sample_structures` |
| `train_ar_ext.py` | any token stream | `train_from_sequences(…, pe=…)` selects AR2; also cluster-BPE sampling and category-token conditioning (`[BOS, CAT, …]`) |
| `train_diffusion.py` | UNet3D | `DiffusionTrainConfig`, `build_grids`, `calibrate_air_bias` |
| `train_twostage.py` | occupancy → materials | Scaffold-recipe two-stage (didn't beat single-stage, T11) |
| `train_graph.py` | graph VAE | recon CE + KL, `beta=1e-3` |
| `train_conditioned.py` | `CondVoxelAR2` | full CLI (see [Labeling](labeling.md) for the data it consumes) |
| `constrained_decode.py` | — (inference only) | **in-loop 6-adjacency constrained sampling**: per-axis factorized coordinate masking keeps every new voxel adjacent to the built structure; EOS gated by `min_blocks`. Validity 1.0 by construction, quality-neutral in raster order (T11) |

## Evaluation (`blockgen/eval/`)

`novelty.py::evaluate_novelty(samples, train_set, vocab, grid=…)` produces the shared
report: `mean_nn_iou`, `duplicate_rate` (≥0.95 IoU), `diversity`, occupancy
`validity_rate`, `block_agreement`, plus comparison grids (each sample beside its
nearest training neighbors — the visual novelty proof). The batteries extend it with
`val_nn` (IoU to *held-out* builds — the honest generalization number) and a
real-val-vs-train baseline (what a *real* unseen house scores: ≈0.48 for houses —
sample scores are read as a fraction of this ceiling). `validity.py` supplies
`n_components` and the `repair_lcc` gate (keep largest connected component).

## Rendering (`blockgen/renderer/`)

- `textured.py` — headless **EGL/pyrender** renderer with real Minecraft textures
  (`textures.py` maps vocab tokens → texture tiles). Never import alongside
  osmesa/pyrender-CPU in the same process (context clash — this is why MinecraftACE
  sample rendering happens on the blockgen side).
- `grid.py::save_grid` — mosaic contact sheets; falls back to matplotlib voxels when
  EGL is unavailable. CLI: `python -m blockgen.renderer.grid --houses 32 …`.
- `labeling/render_views.py` — the 4-view per-structure export for captioning and
  DINOv2 ([Labeling](labeling.md)).
- `legogen/renderer/` — separate Blender 4.2 + ImportLDraw package for LDraw/OMR
  sheets (LEGO corpora only).

## Config system

`blockgen/config.py` + `configs/{datasets,experiments,models,training}/*.yaml`:
layered YAML with an `extends:` chain, flattened onto the argparse destinations of
whichever battery you launch — so `--config ideas-full` is shorthand for a bundle of
CLI flags, and any explicit flag still wins. Launch helpers under `scripts/`
(`run_overnight.sh`, `run_ideas.sh`) add a GPU-wait gate (poll `nvidia-smi` until
`FREE_MB` are free), `nohup` detachment, and persist the exact invocation to
`cmd.txt` for resume; `pause_ideas.sh`/`resume_ideas.sh` kill-to-free-VRAM and replay
the same stamp (finished arms skip via their on-disk `novelty.json`).

## Writing your own experiment

The pattern every battery follows — copy it:

1. **Prep** (the honest-novelty protocol, in this order): dedup (IoU 0.95,
   variant-keeping) → held-out val split (~15%, *before* augmentation) → D4
   augmentation (8×, train only) → canonicalize/serialize.
2. **Train** via the matching `training/` module; save the checkpoint and the exact
   vocab/prep needed to decode samples later (the render scripts rebuild prep from
   this).
3. **Sample** (optionally through `constrained_decode`), **evaluate** with
   `evaluate_novelty` + `n_components`, and write `novelty.json` into the arm's
   directory — its presence is the resume marker.
4. Guard each arm (`try/except` → `ERROR.txt`/`SKIP.txt`, continue), append to
   `leaderboard.md`, and record the run in `notes.md`/`results.md` (the living
   research record — the docs summarize it, they don't replace it).

Model-side, the cheapest place to add a new AR idea is a new `pe=` variant or head on
`VoxelTransformerAR2` (everything downstream — trainer, samplers, constraint, eval —
is representation-agnostic); a new decode-time constraint belongs next to
`constrained_decode.py` on the native track or as a `LogitsProcessor` on the
[MinecraftACE](minecraftace.md) track.
