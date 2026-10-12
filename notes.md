# BlockGen — Research Notes

Living lab notebook for the project: reproducibility, design decisions, and the
running ledger of **proven** vs **disproven** ideas. Companion files:
[`results.md`](results.md) (tables / figures / ablations), [`research.md`](research.md)
(literature + strategy), [`roadmap.md`](roadmap.md) (current plan → LEGO paper thesis).
Keep this honest — dead ends are as valuable to record as wins.

_Last updated: 2026-09-19._

---

## 1. Goal & framing

Generative model for **structured / constrained 3D block data**, text-free
(LegoGPT-style). Milestones: (1) Minecraft voxel structures → (2) "tokenized"
cluster/lego-piece structures → (3) other mediums (electronics netlists).
Core deliverable for the paper: **generate novel structures and prove novelty**
(nearest-neighbor comparison, not memorization).

Central technical challenge: **variable size / footprint**. Handled three ways,
all from one cropped `Structure` + shared tokenizer (see §4).

---

## 2. Environment (reproducibility)

- Python 3.13, project venv at `.venv/` (use `.venv/bin/python`, **not** system/conda — `nbtschematic` etc. only in the venv).
- torch 2.10 + cu128, torch_geometric 2.7, nbtschematic 0.2.1, nbtlib 1.12.1, matplotlib 3.10.8.
- GPU: RTX 5070 Ti (16 GB).
- **Not available:** pandas, sklearn, scipy, ipywidgets, nbformat, tensorflow. (Curation/eval are numpy + matplotlib + torch only; tfrecords parsed by hand — no TF.)
- Jupyter CWD = the notebook's own directory → always anchor data paths to repo root (the cache builders do this via `Path(__file__).resolve().parents[2]`).
- **Docs:** browsable MkDocs site under `docs/` (`mkdocs.yml`, Material theme). Build with `pip install mkdocs-material && mkdocs serve`. Covers overview / data / representations / models / experiments / results / LEGO roadmap. `site/` is gitignored.

---

## 3. Data pipeline (reproducibility)

### 3.1 Sources
- `data/raw/*.schematic` — 10,963 files, numeric ids 1..18837. A **separate, drifted** download. Filenames map to **no** metadata.
- `data/more/minecraft-schematics-dataset/` — the crawl:
  - `fullSchematics.json` — 69,363 records keyed by planetminecraft `url` (title, subtitle/category, tags, description, views, downloads, diamondCount, favorites, comments).
  - `schematics/*.tfrecords` — 36,290 records, each `(url, schematicData)` in JSON order. **This is the metadata join key.**

### 3.2 Two caches (both gitignored under `data/cache/`)
- **Legacy / unlabeled:** `python -m blockgen.data.build_cache --max-dim 24`
  → `small_24.npz`. Scans `data/raw`, crops, filters (`max_dim≤24`, `8≤blocks≤4096`). 2,892 kept, 184 bad. **No metadata.**
- **Labeled (preferred):** `python -m blockgen.data.tfrecord_dataset --max-dim 24`
  → `tf_small_24.npz` + `tf_small_24_meta.json`. Decodes schematics straight from the tfrecords; carries the `url` in `source_path`. **5,866 kept, 100% with metadata.**

### 3.3 Decode path for tfrecord bytes
`SchematicFile.from_fileobj(io.BytesIO(gzip.decompress(schematicData)))` →
`Structure.from_schematic`. ~Half of records are non-classic formats
(`.schem`/`.nbt`/zip) and raise `KeyError: 'Blocks'`; these are skipped (safe).

### 3.4 External public corpora (pool-pretraining; loaders in `utils/corpora.py`)
- **3D-Craft / CraftAssist houses** (`data/3d_craft/houses/`, fetched 2026-07-03):
  **2,586 crowdsourced houses**; `schematic.npy` is `(Y,Z,X,2)` uint8 (id,data) —
  loader transposes to our XYZ. `placed.json` = **human build order** (verified: order
  length == occupancy) → natural connected ordering for the AR track (VoxelCNN signal).
  `load_3dcraft(min_blocks=60)` → ~2.5k. Same corpus as "CraftAssist houses" — one cite.
- **text2mc Kaggle dump** (fetched 2026-07-03 → symlink `data/text2mc/`, 17 GB): **11,092
  builds** as `.h5` uint16 **token** arrays over a 3,717-entry modern block-state vocab
  (`tok2block.json`; air = tokens 102/576/3352 — NOT 0). `load_text2mc` maps air→0,
  block→token+1 (shape pretraining; materials need a tok2block→family remap). Builds are
  large (median max-dim ~112, occ ~46k — whole projects): filter by `max_dim` or
  downsample; size index in `data/text2mc_index.json` — max-dim distribution: ≤24 → 937,
  ≤32 → 1,393, ≤48 → 2,814, ≤64 → 4,213, ≤96 → 6,168. CSV has PlanetMinecraft
  PAGE_URL + TAGS — **same source site as our schematics crawl → dedup before pooling.**
  Pool-pretraining estimate: GrabCraft 3,952 + schematics ~5.9k + 3D-Craft ~2.5k +
  text2mc ≤64-downsampled ~4.2k ≈ **~16k builds** before cross-corpus dedup.
- **2026-07-17 — unlocked the 28,235 raw `.schem`** (the gap between text2mc's "11k
  processed" and its README's "~40,000 builds"; author never finished `.schem`→`.h5`).
  Sponge/WorldEdit format (NBT `Palette` + varint `BlockData`, YZX) — `nbtschematic` can't
  read it. `blockgen/utils/schem.py` decodes it (vectorized LEB128) and `remap_name` maps
  palette strings → legacy vocab directly (~97-98%). `corpora.load_text2mc_schem` (general,
  legacy space, corpus=`text2mc_schem`) + `houses.load_text2mc_schem_houses` (house-tagged
  subset from CSV `PROCESSED_PATHS`, wired into `build_house_dataset`). nbtlib ~80-200ms/file
  → house loader uses a 1.5MB size prefilter to skip giant world exports. **Yield is
  strongly scale-dependent** (builds are large): house shards kept = **8 @32³** (2 with
  rooms) vs **348 @48³** (145 with rooms) — negligible at 32³, ~doubles text2mc's `houses_48`
  share at 48³. Free TAGS → coarse category via `labeling/categorize.py`.
- Bibliography for all of this: **`references.md`** (per-paper relevance notes).

### 3.5 Consolidated data & LABEL inventory (2026-07-06 — the conditioning menu)

Every corpus, its cache, and exactly which labels it carries (what a
conditioning/finetuning scheme can draw on):

| # | Corpus | Cached | Cache / loader | Labels & signals |
|---|---|--:|---|---|
| 1 | **PM schematics crawl** (tfrecords) | 5,866 @24 | `tf_small_24.npz` · `Curator.from_labeled_cache` | **title, description, tags** (free text); **category** (15 PM "map types": Land Structure, Redstone Device, 3D Art, …); **popularity** (views, downloads, diamonds, favorites, comments); url |
| 2 | **GrabCraft scrape** | 3,952 @24 / 3,061 @32 / 5,621 @48 | `gc_small_<dim>.npz` · `Curator.from_grabcraft_cache` | **title**; **subcategory** (113, e.g. medieval-houses; → 5 top-level via `subcat_to_toplevel.json`: BUILDINGS/STATUES/TRANSPORTATION/PIXEL ART/OUTDOORS); views; exact `(id,data)` palettes |
| 3 | **3D-Craft / CraftAssist** | 2,586 dirs (load-time filter) | `data/3d_craft/houses/` · `corpora.load_3dcraft` | **no text labels** (all "house"); **human build ORDER** per house (`placed.json`, `load_3dcraft_order`) — unique action-sequence signal |
| 4 | **text2mc** (PM dump) | 11,092 h5 + 28,235 `.schem` | kagglehub symlink · `corpora.load_text2mc` (h5) / `corpora.load_text2mc_schem` (schem, via `utils/schem.py`) → legacy vocab | **PM page URL + TAGS** via CSV (37,989 rows; h5 = `batch_*_<row>.h5` → row, schem = `PROCESSED_PATHS`); normalized category via `labeling/categorize.py`; caveat: builds are *chunks* of multi-build pages → page labels weak per-chunk |
| 5 | **Legacy `data/raw`** | 2,892 @24 | `small_24.npz` · `Curator.from_cache` | **none** (metadata join disproven, §8) — shape-only pool data |
| 6 | **Curated houses** (derived, 1+2+3+4) | 2,661 @32 / 2,971 @48 | `houses_<dim>.npz` · `curation.houses.load_house_structures` | **corpus** provenance + **category** + title/url in manifest; all quality-gated & deduped |

**Label taxonomy across corpora** (for a unified conditioning vocab):
- *Category*: two taxonomies — PM map-types (corpus 1) and GrabCraft subcats (2);
  map into one small shared ontology (house / castle / vehicle / statue / pixel-art /
  redstone / outdoors / other) + per-corpus fine token; 3/4/5 mostly unlabeled → UNK.
- *Free text*: titles + tags + descriptions on 1, 2, 4 (~20k builds) — enough for
  keyword-token or later CLIP/MineCLIP-style text conditioning.
- *Popularity*: scalars on 1 (and views on 2) — quality-weighted sampling or a
  "high-quality" conditioning bit (`auto_mark_reliable` already thresholds this).
- *Build order*: 3D-Craft only — natural AR curriculum / ordering supervision.
- *Derived features* (all corpora, `compute_features`): size, density, palette,
  dominant material, enclosed-air — free conditioning signals (e.g. size buckets).
- *Provenance*: corpus id itself (absorbs domain shift when pooling).

---

## 4. Representation / tokenizer (`blockgen/utils/serialize.py`)
- One unified vocab: specials `PAD=0, BOS=1, EOS=2`, then per-axis coordinate tokens, then block-class tokens. On the legacy subset: `vocab_size=454`, `num_blocks=427`.
- `structure_to_tokens`: raster `(y,z,x)` order, `[BOS, (X,Y,Z,BLOCK)*, EOS]`. Round-trip exact; **all ids asserted `< vocab_size`** (this killed a CUDA device-side assert during AR sampling).
- `structure_to_grid`: crop + center-pad into fixed `grid³` (for diffusion + as the translation-tolerant common rep for NN eval).
- Legacy block ids are unsigned bytes read as int8 → normalize with `& 0xFF`.

---

## 5. Methods so far (three comparison tracks)
All share the tokenizer + the novelty eval. No text embeddings.
- **A — AR token transformer** (`voxel_transformer_ar.py`): EOS-terminated stream; variable size is intrinsic. LegoGPT-aligned.
- **B — Masked discrete diffusion 3D-UNet** (`voxel_diffusion.py`): fixed 24³ grid, MaskGIT/D3PM-style iterative unmask.
- **C — Graph latent VAE** (`large_pyg_graph_generator.py`): block+port PyG graph → latent → token decoder; size-agnostic.
- **Eval** (`blockgen/eval/novelty.py`): occupancy-IoU nearest neighbors, duplicate rate, inter-sample diversity, validity (single connected component), block agreement.

---

## 6. Curation (`blockgen/curation/curate.py`)
`Curator` over a cache: per-structure features (dims, n_blocks, density, footprint,
height, n_block_types, dominant_block, connectivity, **palette_sig**), plus, on the
labeled cache, metadata (title, category, tags, description, popularity).
- Filter/sort/`search`; `group_by_similarity` (GPU IoU, 47s→1.6s); feature k-means.
- **Variant-aware dedup:** `find_exact_duplicates` (same shape **+** palette → drop extras) vs `find_variant_groups` (same shape, **different** materials → KEEP). `dedupe_keep_variants()` does the safe pass.
- `auto_mark_reliable()` by popularity; decisions persist to `data/cache/curation_decisions.json` keyed by source path/url.
- Notebook: `notebooks/data_curation.ipynb`.

### 6.1 Unified curated HOUSE dataset (`blockgen/curation/houses.py`, built 2026-07-06)

One clean cross-corpus house cache on the **shared legacy (id, data) vocab**:

    .venv/bin/python -m blockgen.curation.houses --max-dim 32   # -> data/cache/houses_32.npz (+_manifest.json)

- **Sources pooled:** GrabCraft `*-houses` subcategories; **all** of 3D-Craft
  (houses by construction); text2mc builds whose PlanetMinecraft URL/tags match
  house keywords (`house|cottage|cabin|villa|manor|…`).
- **text2mc unification:** `utils/block_remap.py` maps the modern 3,717-state
  vocab → legacy (id,data) at family fidelity (colors/woods/stairs parsed;
  3,611/3,717 states mapped, rest → stone; barrier/light → air). text2mc chunks
  are world *cuts* with terrain — `_strip_ground()` removes consecutive bottom
  layers that are ≥60% terrain blocks and ≥45% filled (308/410 needed it).
- **Quality gate** (first-failing-rule reported): `n_blocks≥80, height≥4,
  footprint≥16, 0.03≤density≤0.85, types≥3, dominant_frac≤0.92,
  largest_component≥0.55`. Then variant-aware exact dedup (IoU .95 + palette).
- **Known caveat (text2mc):** its h5 chunks are pieces of multi-build scene
  pages, so a "house"-tagged page can contribute a non-house chunk from the same
  scene (a parked truck, a tank) that passes the gates. Small slice (34 @ 32³ /
  180 @ 48³) — worth a quick manual pass in the Curator notebook before big runs.
- **Enclosed-air "house-ness" gate** (`enclosed_air_count`): interior air voxels
  unreachable from the bbox boundary (vectorized flood fill). Calibrated
  2026-07-06: 17% of ground-truth GrabCraft houses have open interiors → gate
  (≥8 voxels) applies **only to 3dcraft/text2mc**, where it kills the junk a
  visual grid actually showed (trees, a truck, roof fragments): 31% of 3D-Craft
  and 54% of text2mc fail it.
- **Results:** `houses_32` = **2,661** (1,360 gc / 1,267 3dc / 34 t2mc; biggest
  drop = no-interior 675), `houses_48` = **2,971** (1,410 / 1,381 / 180; needed
  `gc_small_48` rebuild first). vs the old labeled-cache `houses` subset (714):
  ~4× more clean houses. Load via
  `curation.houses.load_house_structures(max_dim)`; per-item corpus/category/
  url/title in the manifest. Showcase grids: `outputs/figures/houses_*.png`
  (textured renderer, §13).

---

## 6b. Curated-subset experiments (`blockgen/experiments.py`)

A single reproducible runner does **figures → retraining → novelty**, writing every
artifact under `outputs/run_<stamp>/` (config manifest with git sha + all train
configs, per-subset contact sheets, per-(subset,track) loss curve / samples /
comparison grid / novelty json, and a `metrics.md` table). Invoke:

    .venv/bin/python -m blockgen.experiments --stamp $(date +%Y%m%d_%H%M%S) \
        --epochs-ar 100 --epochs-diff 150 --epochs-graph 100

**Named subsets** (from the labeled cache, via `Curator`):
- `houses` — `search("house")` ∩ structure categories ∩ `min_blocks≥60, comps≤3, types≥3` → **714**.
- `pixel_art` — category `Pixel Art Map`, `min_blocks≥30` → ~124 (compact, all tracks fit).
- `redstone` — category `Redstone Device Map`, `40≤blocks≤255` → ~484 (token-tractable).
- `towers` / `trees` / `popular(≥10 diamonds)` — shown as figures.
- material-**variant groups** (`find_variant_groups`) — same shape, different blocks → KEEP.

**Key feasibility finding (the size/footprint challenge, made concrete):** the token
tracks (A AR, C graph) are bounded by `max_seq_len` (≈ `4·n_blocks` tokens). At
`max_seq_len=1024` only **41 / 714 houses** fit, but **all 714** fit diffusion's fixed
24³ grid. So *which track can learn which subset is itself a result*: dense building
types (houses, med 1086 blocks) are a **diffusion** job; compact types (pixel art med
200, capped redstone) are where AR/graph are trainable. The run trains the feasible
pairings only:
- `pixel_art` → **AR + diffusion + graph** (apples-to-apples, all compact-tractable),
- `houses` → **diffusion** (headline dense build),
- `redstone` → **AR** (token track on its best-supported coherent type).

Each model's novelty is evaluated against **the subset it trained on** (so NN-IoU
answers "did it memorize *these* houses"). Per-subset `BlockVocab` is rebuilt from the
subset, keeping the vocab tight and the model focused.

## 6c. Overnight cohesion + data battery (`blockgen/experiments_overnight.py`)

New unified runner attacking the two live problems — *samples look like the type but
aren't cohesive*, and *the labeled subsets are small* — as one sweep of
**dataset × method**, every experiment guarded (writes `ERROR.txt`/`SKIP.txt` and
continues) with a master `leaderboard.md`. Launch when the GPU is free via
`scripts/run_overnight.sh` (waits for free VRAM, then `nohup`s the module).

**Datasets** (builders return `(structs, labels, categories)`):
- `gc-houses-large` — 9 GrabCraft house classes, footprint `min_dim≥16` (drops the
  ~375 smallest of 1,373; the "larger houses are nicer" filter). max-axis distribution:
  ≥16 = 998, ≥18 = 833, ≥20 = 659.
- `combined-houses` — GrabCraft houses **+** schematics-cache houses (`search("house")`
  ∩ struct cats) merged for a bigger pool; extra class token `schematics-house`.
- `gc-vehicles` — cars/transportation from the **dim-24** cache (far better represented
  there: working-vehicles 182, other-transportation 308, planes 94, cars/sports/family
  ~65, boats, buses, spaceships) — a second object class to test generality.

**Methods** (per dataset): `ar_flat` (baseline), `ar_cluster` (3D-BPE, §9.10),
`ar_cluster_hi<N>` (BPE at higher canonical res — the "reach" test), `ar_conditioned`
(category token, one model any class), `diffusion<grid>` (dense track), `graph_vae`
(Track C: block+port PyG graph → Gaussian latent → GRU token decoder, `--epochs-graph`;
same canonicalized fit/ref/val protocol so it's directly comparable to the AR rows).

**Two methodology fixes baked in** (both were open TODOs, notes §9.9b):
1. **Held-out val split** (`val_frac=0.15`, split *before* augmentation). Novelty is
   reported vs **train** (`duplicate_rate` = memorization) **and** vs **val**
   (`val_nn_iou` = does a sample resemble an *unseen real* build), with the real
   **val-vs-train baseline NN-IoU** printed as the reference for "how close is a genuine
   new build" (~0.64 on vehicles). This is the honest memorization measurement T8's 0.31
   scare demanded.
2. **Fit vs eval sets are separated**: models train on the augmented set; vocab + all
   novelty eval use the *distinct* real builds (`ref`), so metrics compare to real builds
   not rotated copies — and eval stays ~8× cheaper (augmentation multiplies every
   per-structure Python loop, which was the runner's first bottleneck).

## 6d. New building blocks (all wiring-validated on a smoke run 2026-07-02)
- **D4 augmentation** (`blockgen/utils/augment.py`): the 4 vertical-axis rotations ×
  horizontal mirror = up to **8×** data, gravity-safe (no vertical flip). Occupancy is
  exactly preserved under rotation (verified). Applied to **train only**. Caveat: block
  *orientation* in `block_data` (stairs/logs) isn't remapped — fine for our occupancy-IoU
  metrics; canonicalize later. On vehicles: 677 → 5,380 fit builds.
- **3D-BPE cluster tokenizer** (`blockgen/tokenizers/cluster_bpe.py`): BPE over the
  6-connectivity graph — greedily merge the most frequent adjacent `(pieceA, pieceB,
  offset)` into a rigid connected "piece", replay merges to tokenize
  `[BOS,(X,Y,Z,PIECE)*,EOS]`. **Air is never a token** (pieces are occupied-only offset
  sets; gaps = absence) → the LEGO-piece-native rep. Round-trip **IoU = 1.000**; cuts
  sequence length ~40–75% (houses flat 1092 → cluster 638 at 60 merges; more merges =
  shorter), which is what lets AR reach higher resolution. Reuses `VoxelTransformerAR`
  via a generic `train_from_sequences` (`training/train_ar_ext.py`), no model change.
- **Category-conditioned AR** (`training/train_ar_ext.py`): prepend a class token
  `[BOS, CAT_k, …]`; one model samples any class via `sample_conditioned_structures`.
  Category ids sit just above the block vocab.
- **Connectivity/validity gate** (`blockgen/eval/validity.py`): `largest_component`
  keeps the biggest 6-connected component (a cheap *repair*); `gated_sample` oversamples
  + repairs + rejects tiny cores. Reported as an ablation: every AR row carries
  `validity_rate` (raw) **and** `validity_gated`. Smoke: validity **0.5→1.0** (flat),
  **0.0→1.0** (cluster) after LCC repair — the gate is the cheapest visible-cohesion win.

## 6e. Ideas ablation battery (`blockgen/experiments_ideas.py`) — built 2026-07-03

Implements the research.md §B levers as ablation arms (CPU-wiring-validated; GPU smoke
queued behind the overnight run; launch full via `scripts/run_ideas.sh`, same GPU-wait
pattern). Groups (default dataset `gc-houses-large`, same prep/eval as §6c):
- **pe** — flat AR × {learned, sin, rope, alibi, phase4} via the new PE-pluggable
  `models/voxel_transformer_ar2.py` (hand-rolled causal attention; `learned` re-baselines
  the implementation against the stock model). `phase4` = learned (pos mod 4) phase +
  (pos div 4) block-index embeddings — encodes the (X,Y,Z,BLOCK) grammar directly.
- **ordering** — BFS-from-ground training order (`utils/ordering.py`) sampled
  unconstrained (`ar_bfs`) and with **in-loop 6-adjacency logit gating**
  (`ar_bfs_constrained`, `training/constrained_decode.py`: connectivity by construction,
  exact per-axis factorization of the constraint) + `ar_raster_constrained` mismatch
  control. Upgrades the §9.3 LCC repair gate to LegoGPT-style in-loop enforcement.
- **samplers** — ONE diffusion model, four inference rules: maskgit | flow |
  **remask** (ReMDM-lite error correction, `sample_grids_remask`) | **stratified**
  (octant-spread commits ≈ Halton de-clustering, `sample_grids_stratified`).
- **twostage** — occupancy→materials factoring (`training/train_twostage.py`): binary
  occupancy diffusion + occupancy-clamped material MaskGIT (loss only on masked occupied
  voxels — zero air-imbalance pressure). Scaffold recipe + the generated-occupancy stage
  they left as future work. Watch `mean_block_agreement` vs single-stage `diff32_maskgit`.

## 7. Proven ideas ✅
- **Grammar-aware PE (phase4) is a small honest win** (T11: val_nn **0.405** vs learned
  0.395, best of all 5 PE arms, dup 0). Injecting the (X,Y,Z,BLOCK) period-4 grammar via
  pos%4 phase + pos//4 block-index embeddings beats making the model learn it. Keep it;
  it's cheap and stacks with conditioning. [2026-07-04]
- **In-loop adjacency constraint = validity 1.0 at ~no quality cost, with RASTER order**
  (T11: ar_raster_constrained validity 1.0, nn_iou 0.428 ≈ unconstrained band). This is
  the hard-connectivity knob; compose it on top of phase4. Upgrades §9.3 post-hoc LCC
  repair to LegoGPT-style enforcement. **BFS order must NOT be used** (see §8). [2026-07-04]
- **Two-stage factoring lifts diffusion *shape* fidelity** (T11: twostage32 nn_iou 0.382 =
  best of the diffusion family vs single-stage 0.21–0.26). The occupancy/material split
  helps geometry — but only geometry (see §8 for the material-stage failure). [2026-07-04]
- **Category conditioning lifts raw cohesion on every dataset** (T10: best-in-dataset raw
  validity gc-houses 0.333, vehicles 0.7; best val_nn overall on combined 0.435 = **93% of
  the real-build baseline** with dup 0). One class token, no architecture change. [2026-07-03]
- **3D-BPE cluster tokens resist memorization** (T10 vehicles: cluster dup **0.0** & best
  quality nn_iou 0.598 while flat memorizes at 0.125 and cluster@20 at 0.188). Chunk
  emission generalizes rather than copies. [2026-07-03]
- **More data → cohesion**: gc→combined houses lifts flat-AR raw validity 0.188→**0.562**
  and graph-VAE nn_iou 0.31→0.388 (T10) → pool-pretraining (§3.4, ~16k builds) is the
  next data move. [2026-07-03]
- **Honest-novelty protocol works**: D4-aug + val-split kept all 12 houses rows at dup 0
  while val_nn reached 85-93% of the val-baseline — generation ≠ memorization,
  quantified. [2026-07-03]
- **Build the cache from tfrecords, not `data/raw`.** Gives 100%-labeled data (see §8 for why the alternative failed). [2026-06-30]
- **Content-hash join is drift-proof but useless here** — *as a verification it proved* the raw files don't correspond to the metadata. Right tool, decisive negative result.
- **Asserting token ids `< vocab_size`** eliminates the CUDA device-side assert in AR sampling.
- **Down-weighting the air class** + **stochastic MaskGIT sampling** + **`air_bias` calibration** are all required for diffusion to produce non-empty, varied grids (see §8).
- **GPU IoU matrix** for similarity grouping (torch matmul) — ~30× faster than numpy at N≈2.9k.
- **`(id, data)` palette signature** distinguishes material/color variations of the same shape (resource-location alone does **not** — oak vs spruce planks share `minecraft:planks`).
- **Curated, single-category, labeled subsets are extractable** (714 "houses") — enables the "learn one object type, then expand" plan.
- **Diffusion learns a dense build *type* without memorizing** (run `20260630_220830`): trained on 714 houses, samples reproduce the house gestalt (grass→walls→roof) at occ 950 vs 1086, NN-IoU 0.38–0.49 to *distinct* houses, duplicate-rate 0. Fixed 24³ grid is the right tool for builds the token tracks can't ingest. [2026-06-30]
- **Per-track feasibility is dataset-dependent and must be matched to the subset:** token tracks (AR/graph) only ingest compact builds (`seq≈4·n_blocks ≤ max_seq_len`); diffusion's fixed grid takes any. Don't force one track on all subsets.
- **AR is our best house generator once scale-normalized** (run `…_232913_gen`): on identical canonical 12³ houses, AR NN-IoU **0.568** vs diffusion 0.369, matches target occupancy (165 vs 60), and generates *bottom-up* (foundation→walls→roof) — the spatial autoregression is a good building prior. This is the LegoGPT-aligned track → lead the LEGO demo with it. [2026-06-30]
- **Scale normalization (downsample to fixed N³) unlocks the token tracks on dense builds** — 654/714 houses fit AR at N=12 vs 41/714 at N=24. Lossy but enabling. [2026-06-30]
- **Flow-matching sampling reaches realistic occupancy where MaskGIT collapses** — on the same trained net, MaskGIT under-fills (occ 96 vs target 1086, dumps a blob at the last step) while the rate-driven flow sampler grows occupancy smoothly to 1017. Reusable on the existing absorbing-diffusion weights; needs its own (negative) `air_bias`. [2026-06-30] *(Sampler behavior is dataset-dependent: on GrabCraft the opposite held — MaskGIT held occupancy, flow under-shot. Calibrate per run.)*
- **Cleaner data → much better AR generation** (run `…_grabcraft`, GrabCraft medieval-houses, 724): AR NN-IoU 0.568→**0.727** and block-agreement 0.17→**0.483** vs Minecraft-schematics houses — crisp, palette-correct medieval houses. Category-labeled datasets are worth the scrape. [2026-07-01]
- **GrabCraft pipeline exists and mirrors the tfrecord one** — `blockgen/data/grabcraft_{scraper,dataset}.py` → `gc_small_<dim>.npz`; load via `Curator.from_grabcraft_cache()`. Blocks decode to exact `(id,data)` from each entry's `texture` field (`<id>_<data>.png`). `gc_small_32` = 3,061 builds; 9 house classes = 1,373.
- **Scaling data + resolution kills AR memorization** (run `…_061656_grabcraft`): medieval-only canon-12 had `dup_rate=0.31` (verbatim copies); all-9-house-classes at canon-16, dim-32, deduped, fewer epochs → **`dup_rate=0.00`** on every model, NN-IoU 0.47–0.51 vs distinct neighbors. Dedup dropped only 6 → the earlier dups were *model* memorization, not raw copies. Fix = more/diverse data + higher canonical res (12→16, less downsample collapse) + fewer epochs. [2026-07-01]
- **Larger native grid + more data ~doubles validity** — diffusion validity 0.31→**0.56** at 32³/16³; the best connectivity we've measured. Bigger `max_dim` cap (chosen 32: the 85% knee of the house size distribution) is worth it.
- **Choose the cache `max_dim` from the size distribution.** House builds by max-axis: ≤24=65%, ≤32=85%, ≤40=92%, ≤64=98%. 32 is the knee (2× data vs 24, still cheap for 32³ diffusion). Raise `max_blocks` too (was 4096) or dense builds get dropped.

- **phase4 + constraint combo: constraint dominates, PE gain doesn't stack**
  (2026-07-08, `experiments_p4c.py`, 32 samples on the T11 checkpoint):
  ar_phase4_constrained val_nn **0.382** / validity **1.0** / dup 0 ≈
  ar_raster_constrained (0.383/1.0). The unconstrained phase4 edge (0.405 vs
  0.395) is absorbed by in-loop gating — the constraint costs ~0.02 val_nn
  regardless of PE. Keep phase4 (free, principled) but don't claim stacking;
  the validity-1.0 row is the paper-relevant one. Row saved next to T11 arms.

## 8. Disproven / dead ends ❌
- **KV cache is NOT the sampling bottleneck — per-step launch overhead is** (2026-07-15).
  Implemented + verified (identical token ids vs uncached across all 6 PEs), but the
  speedup is only **1.09x @400 tok / 1.20x @1600 / 1.50x @3200**, not the ~800x a FLOP
  argument predicts. The tell: **cached ms/step is FLAT at ~11.5–12.3 ms regardless of
  context length**. At 5M params processing one token, attention arithmetic is
  microseconds; the rest is Python + CUDA kernel launches, and you cannot optimize away
  work that was never the cost. The cache's apparent gain grows with length only because
  the *uncached* side degrades quadratically. **The real 16x is batching**:
  `train_ar.sample_structures` loops `for _ in range(num_samples)` at **batch 1**, so 16
  samples = 16x the sequential steps; a 5M model on one token barely occupies the GPU, so
  B=16 costs ~the same wall clock as B=1. Needs per-sequence EOS/done-tracking. Cache is
  kept (free, correct, composes with batching, and it fixed two latent bugs: `rope.rotate`
  used `cos[:L]` and `sin` used `sin_table[:L]`, both zero-indexed — any cached decode
  would have silently applied position-0 encodings at position t). [2026-07-15]
- **Pool-pretrain → finetune gives NO in-domain quality gain at our scale** (T12,
  2026-07-08): finetune val_nn 0.379 < scratch 0.405 despite train loss 0.142 vs
  0.181 — the 113k-sequence pool prior buys compression, not generation quality,
  when the target set (houses_32, 2.3k train) is already sufficient. Caveats: 1
  seed / 16 samples (±0.03 band), one LR. **Do not** claim "pretraining helps" for
  data-rich targets; the cross-medium (Minecraft→LEGO) case is a different,
  data-POOR regime and stays open — and T12's zero-shot 0.365 shows the pool
  model itself is competent. Also: houses_32 scratch = 0.405 val_nn matches the
  best T11 arm → the curated cache trains as well as gc-houses-large. [2026-07-08]
  **[2026-07-15 — mechanism found, T17]:** all 16,383 pooled builds went through the
  same `canon_dim=16` strided decimator, which destroys 86.5% of blocks and halves
  connectivity on the 63% of builds it touches. The negative result stands as
  reported, but "more data does not help *this pipeline*" is the honest scope — it
  is **not** evidence that data scaling fails in general. Re-test after the
  native-resolution run.
- **"Same-type BPE merging would help" — already happens, no change needed**
  (T18, 2026-07-15): 253/256 learned pieces (98.8%) are a run of exactly one block
  token; `_canon_pair` keys on `(piece_a, piece_b, delta)` and frequency does it
  implicitly. The real defect is elsewhere: greedy *count* maximization grows
  **rectangles** (16 distinct shapes ↑D4 across 256 slots; no corners, no slopes),
  and on real builds it barely merges at all (1.2–1.9 voxels/piece). [2026-07-15]
- **Generic relative PE (RoPE/ALiBi) does NOT beat learned-absolute here** (T11: RoPE
  val_nn 0.391 ≈ learned 0.395; **ALiBi worst arm at 0.346**). Our coords travel as tokens,
  not positions, so there's no length-extrapolation regime for ALiBi's distance bias to
  help — it just costs capacity. Only grammar-aware phase4 wins (§7). Contra Scaffold's
  big PE swing. [2026-07-04]
- **BFS-from-ground token order hurts** (T11: ar_bfs val_nn 0.337 worst non-broken arm;
  bfs_constrained 0.397 < raster_constrained 0.428). The locality-preserving order is
  *harder* to learn than raster scan and drags down the constrained arm. Use raster +
  constraint (§7), not BFS. [2026-07-04]
- **No sampler rescues an uncalibrated 32³ diffusion model** (T11: off ONE trained model,
  occ swings 2 → 830 → 7716 by schedule alone, best sampler stratified nn_iou 0.259 ≪ AR
  0.44). **remask is broken** — the ReMDM re-mask/progress-gate interaction erases the
  structure (occ 2); lower remask_frac or raise the 0.75 gate before reuse. The occupancy
  problem is the model's, not the sampler's. [2026-07-04]
- **Two-stage does NOT fix material scrambling** (T11: twostage32 block_agree 0.028 ≈
  single-stage ~0.03–0.11; occ overfilled ~9× to 12968). The material MaskGIT stayed
  near-random and the occupancy stage over-generated. Factoring helps shape (§7) but the
  material stage needs rework (stronger conditioning on the occupancy footprint / more
  epochs / loss reweighting). [2026-07-04]
- **"Cluster tokens automatically cohere" — no** (T10): on houses@16³ cluster-AR raw
  validity is 0.0 (big pieces placed disconnected) and quality trails flat (0.378 vs
  0.485). BPE's win is generalization + reach, not connectivity — pair it with the
  §6e in-loop adjacency constraint. [2026-07-03]
- **Diffusion-32 occupancy calibration is fragile on heterogeneous pools** (T10): houses
  overshoot 2577 vs 1461 target, combined-houses collapse (nn_iou 0.202). Single
  air-bias scalar can't fit a mixed-size corpus; §6e two-stage + sampler arms are the
  fix; diffusion-16 (T9) stays the dense track's best configuration. [2026-07-03]
- **90 epochs overfits small homogeneous classes** (T10 vehicles: flat final_loss 0.07 →
  dup 0.125). Add early stopping on val NLL (§6e bundle). [2026-07-03]
- ~~**`data/raw` filename → metadata by index offset.**~~ **⚠ THIS DEAD END WAS WRONG —
  CORRECTED 2026-07-15.** The original experiment was sound but aimed at the wrong target:
  it joined `data/raw` against the **PlanetMinecraft tfrecord** metadata, which correctly
  failed (file `1.schematic` ≠ record 0 by any offset; content-hash match only 5.0%) —
  *because these files were never from PlanetMinecraft*. They come from `data/download.py`,
  which walks **minecraft-schematics.com** ids 1824→19000; the files are named `<id>.schematic`
  (verified range 1..18905, 12,364 id-named files). **The filename IS the m-s.com schematic
  id**, and every file re-links to `/schematic/<id>/` carrying category + theme + size +
  title + author + rating. The 5% hash overlap is just two unrelated crawls sharing popular
  builds. → `data/raw` is **labeled after all**; see §17 and results.md T19.
- **Inverted MaskGIT schedule** (original `_keep_masked_fraction` unmasked everything at step 0) → all-air collapse. Fixed to `cos(0.5·π·progress)`, decreasing 1→0.
- **Argmax diffusion sampling** from an all-mask start → identical samples. Replaced with categorical + Gumbel-perturbed confidence.
- **PyG `DataLoader` with tokens stored on `Data`** → `RuntimeError: sizes must match`. Replaced with plain `DataLoader` + custom collate (`Batch.from_data_list` + manual token pad).
- **`torch.gelu`** → use `nn.functional.gelu`.
- **MaskGIT confidence sampling on a *well-trained* absorbing-diffusion net** → under-fills to near-empty then dumps a blob at the last step (it commits high-confidence air first). Not a code bug — structural. Mitigate with flow-matching sampling (§9.7) or an occupancy prior. [run `…_232913_gen`]

## 9. Open / current ideas 🔬 (priority order)

> **0. ~~THE GATING RUN~~ — RAN 2026-07-15, NEGATIVE (→ results.md T20).**
> native32+BPE **75.4%** of its baseline vs canon16+flat **83.3%**. The intervention LOST;
> the **decimator thesis is not supported**. T17's measurement stands (canon-16 really does
> destroy 86.5% of blocks, F16); the *inference* that fixing it helps generation does not.
> **But the run cannot answer the question**: it moved resolution AND tokenizer, and BPE is
> a known loser on houses (T10) whose failure it reproduced exactly. Arm B was also underfit
> (loss 0.287 still falling vs the control's converged 0.174).
>
> **Now blocking (§9.18): a perceptual metric.** nn_iou has now disagreed with the eye
> three times in one day — it said 84% on rubble (T17) and "worse" on the first samples
> with real roofs and windows (T20). Its cross-grid comparability is not defensible even
> as a ratio: 83% of a coarse blob-matching score ≠ 75% of a fine detail-matching one.
> **Nothing downstream is adjudicable until this exists.**
>
> Then: (a) **native+flat @ seq 8192** (88.7% fit — unaffordable when T20 was designed,
> cheap once the SDPA fix landed; this de-confounds resolution from tokenizer and is the
> run that *should* have been designed); (b) arm B to val plateau; (c) BPE-aware
> constrained decoding (`sample_constrained_structures` is BlockVocab-only, so T11's
> validity-1.0 knob sat unused in T20).

1. ~~**Factored block embeddings**~~ **→ IMPLEMENTED 2026-07-15 at the piece level**
   (`tokenizers/piece_factors.py`, `models/factored_embedding.py`, opt-in via
   `VoxelTransformerAR2(piece_factors=…, piece_offset=…)`). `E[piece] = E_shape[↑D4] +
   E_rot + E_family + E_variant` subsumes the atomic `(family, variant)` split and adds
   rotation sharing. 678 pieces → 17 shapes / 8 rots / 239 families / 16 variants;
   embedding rows 713 → 315. **Awaiting the ablation vs flat at matched budget** — run it
   *after* §9.0, not folded into it.
   *Original evidence, still the motivation:* (a) block-class agreement 0.007–0.12 (shape
   learned, materials interchanged); (b) **direct** — trained AR block embeddings
   (`outputs/analysis/embedding_analysis.png`) show wood/wool variants near-orthogonal
   (within-family cos-sim 0.006/0.008 vs random 0.002). Flat embeddings never discover
   birch≈oak. (c) T18: 41% of merge slots are same-shape/same-family duplicates.
   *Resolved caveat:* `data` also encodes orientation (stairs/logs/doors) — we
   **factor the embedding instead of canonicalizing the tokens**, so no block-data
   rotation table is needed and a bad decomposition costs sharing, never correctness.
   A vertical vs horizontal log lands at "same family, different variant" — which is true.
2. **Decouple shape from palette** — generate geometry in an abstract palette, paint materials with a second head. Reframes "same house, different wood" as one shape sample × palette draw.
3. **Validity gate** — connectivity (and later support/gravity) check with rejection + rollback during AR sampling. Current samples fragment (validity ≈ 0); this is the biggest visible-quality lever. *Evidence:* run `20260630_220830` validity 0.06–0.25; AR/redstone samples swing from coherent (NN-IoU 0.62) to scattered voxel clouds (0.07) — connectivity is unconstrained.
4. **Category conditioning** — prepend a class token (metadata enables this) or train per-category. Generating "a house" ≫ easier than "any of 15 map types."
5. Scale `max_dim` / sequence length once the narrow case works.
6. `block2vec` co-occurrence init — only if 1–2 insufficient.
7. **Flow-matching as the default diffusion sampler** (implemented: `sample_grids_flow`). Holds occupancy where MaskGIT collapses; next: continuous-time corrector steps (re-mask + resample) and a learned/length-conditioned reveal schedule. Compare against an occupancy-prior MaskGIT.
8. **Delta-coordinate AR tokens** — emit each voxel's coords relative to the previous, to shorten sequences (reach 16–24³) and bias toward locality. Pairs with the §11 normalization story.
9. **Control AR memorization** (surfaced on GrabCraft: `dup_rate` 0.31, some samples NN-IoU=1.00). Levers: (a) **dedup the training set** first (`find_exact_duplicates`/`dedupe_keep_variants`); (b) evaluate novelty on a **held-out val split**, not train; (c) higher canonical resolution so downsampling stops collapsing distinct builds; (d) regularize / early-stop; (e) larger, more diverse subset. This is the current top open problem for the AR track. **(a,b,e implemented** in the overnight battery §6c: dedup + D4 augmentation + held-out val split + `val_nn_iou`/baseline reporting; (c) higher canonical res unlocked by BPE.)
10. **3D-BPE cluster tokens** (`cluster_bpe.py`, §6d) — the cohesion + sequence-length
   attack: occupied-only connected "pieces" as tokens so AR emits chunks (can't produce
   single-voxel noise) and sequences shrink enough to raise resolution. Supersedes the
   delta-coordinate idea (§9.8/§11.5) as the primary reach lever; delta-coords remain a
   cheaper fallback. **Wiring done; effect on cohesion/novelty is the headline of the
   pending overnight run.**

11. **Pool-pretrain → labeled finetune** (queued 2026-07-06; unblocked by
    `block_remap` putting all corpora on one vocab). Pretrain the AR (phase4 PE)
    on the full cross-corpus pool (~14k builds unlabeled: caches 1–5 in §3.5),
    then finetune on curated labeled houses (`houses_32` / gc-houses-large) with
    conditioning tokens. Tests the classic data-scaling bet on our exact stack;
    "more data → cohesion" (§7) says it should win. Compare: scratch vs
    pretrained at equal finetune epochs.
12. **Multi-slot conditioning + label dropout** (extends proven §9.4 single class
    token). Prefix = `[corpus] [category] [size-bucket]` (+optional style token =
    dominant-material family), each slot independently dropped to UNK with
    p≈0.1–0.3 during training (classifier-free-guidance style) so ONE model
    serves unconditional + any-slot-conditional generation across the
    heterogeneous label coverage of §3.5. At sample time: sweep categories for
    "different kinds of outputs" figures.
13. **Build-order supervision from 3D-Craft** — train the AR on human placement
    order (`load_3dcraft_order`) instead of raster; VoxelCNN showed this is a
    strong prior. Cheap ablation vs raster on the same houses. (Distinct from
    the disproven synthetic BFS order, §8 — this order is *human*.)
14. **phase4 + in-loop constraint combo** — the two T11 winners are orthogonal
    and have never been run together (constrained arm used stock learned PE).
15. **PMI / WordPiece merge scoring** (T18, 2026-07-15). Score merges by
    `count(ab) / (count(a)·count(b))` instead of raw `count(ab)` at
    `cluster_bpe.py:201`. Greedy count-maximization on a voxel grid always grows
    **rectangles** — the most frequent pair is forever "extend the rectangle" — which
    is why 256 slots encode only 16 shapes and the sheet has no corners or slopes.
    Planks-next-to-planks is high-count but low-PMI (planks are everywhere, so the
    adjacency is unremarkable); log-above-log is rarer but *specific*. PMI is exactly
    the "these go together more than chance" statistic that "meaningful part" means.
    One-line change; **eval is F14/F15** — rerun the sheets and see whether corners and
    pillars appear. Forks the paper: if they do, research.md §D's "connectivity-native
    tokenization" claim survives; if not, 3D-BPE is honestly a *compression* scheme and
    the novelty moves to the factored grammar (§9.1).
16. **Shape-factored merges** (T18). Learn merges over *shapes* (material-agnostic) and
    apply to any material, rather than over `(shape × material)`. Attacks the coverage
    problem no `n_merges` fixes: 422 block types × 6 directions ≈ 2,500 candidate
    same-type adjacencies vs 256 slots, so any material outside the global top-256 never
    merges and stays atomic (measured: **1.2–1.9 voxels/piece** on real builds). One rule
    ("vertical run of 3") would cover all 422 materials at once. Tokenizer change → do
    after §9.1's embedding-level ablation reports.
17. **Block-data rotation table** — the artifact three things need: correct D4
    augmentation (currently 7/8 of augmented training data has wrong-facing
    logs/stairs/doors — documented and tolerated at `utils/augment.py:12-16`, justified
    against occupancy metrics but then charged to **block-agreement**, which has been
    stuck at 0.03–0.15 since T5), token-level rotation canonicalization (§9.16), and any
    future orientation-aware curation. Not needed for §9.1 (deliberately).

> **Implemented since 2026-07-02** (was open, now wiring-validated, awaiting full run):
> §9.3 validity gate (→ `eval/validity.py`, LCC repair + `validity_gated`), §9.4 category
> conditioning (→ `train_ar_ext.py`), §9.9 memorization controls (val split + fit/ref
> separation), §9.10 3D-BPE. Still open: §9.1/§9.2 factored `(family,variant)` embeddings.

> **Deep-research sweep 2026-07-02 → `research.md`** (landscape / quality levers /
> web-knowledge rewards). Headlines: (a) a **text-serialized LLM baseline** (LoRA
> LLaMA-3.2-1B on `block x y z` lines, the exact LegoGPT recipe) is reviewer-expected and
> fits our GPU — planned track D; (b) **occupancy→materials factoring** (Scaffold
> Diffusion) directly targets our block-scrambling failure — their AR baseline reproduces
> it, and their PE ablation (learned vs 3D-sinusoidal, PPL 29→1.8) demands an audit of our
> tracks; (c) **adjacency-gated AR decoding + BFS-from-ground order** upgrades §9.3 from
> post-hoc repair to in-loop enforcement; (d) **MineCLIP** (640k YouTube MC clips, open
> weights) is a domain-matched reward for GRPO/DPO post-training ("more house-like" from
> web knowledge), with public corpora (text2mc ~11-40k, 3D-Craft 2.5k) for pool-pretraining
> first. Validates §9.1/§9.2 (Scaffold evidence) and the 3D-BPE axis (BrickAnything's tree
> tokenization is the nearest published relative).

---

## 10. Related work (for the paper) & novelty
- **LegoGPT** — text-free AR brick generation + physics-validity rejection/rollback. We share the AR + validity stance; differ in (a) Minecraft voxels→cluster tokens, (b) **three-way comparison** (AR vs discrete diffusion vs graph VAE) on one shared rep, (c) explicit material-variant-aware data curation.
- **MaskGIT / D3PM** — discrete/absorbing-state diffusion (our Track B).
- **block2vec** — co-occurrence block embeddings (candidate for §9.6).
- **Working novelty angles:** unified `Structure`↔tokenizer across 3 generative families with one NN-novelty protocol; metadata-driven curation that *preserves* material variations; a path from single-voxel tokens → piece/cluster tokens → electronics netlists that reuses the same serialization + eval.

> TODO as we go: pin exact related-work citations; quantify novelty vs LegoGPT (different medium, multi-model comparison, curation methodology).

---

## 11. Normalization methods (`blockgen/experiments_gen.py`)

The variable-size problem *is* a normalization problem. Houses (median 1086 blocks)
can't be tokenized at 24³ (median ~4346 tokens ≫ `max_seq_len`), which is why §6b
trained them diffusion-only. Levers we have / are trying:

1. **Translation normalization** — `crop_to_non_air()` + center-pad. Already universal;
   makes NN-IoU translation-tolerant. ✅ done.
2. **Scale normalization (canonical resolution)** — downsample each build so
   `max_axis ≤ N`, fixing the footprint scale and *bounding token length*. This is what
   unlocks the token tracks on houses: at N=12, **654/714 houses** fit AR `seq≤1600`
   (vs 41/714 at N=24). Cost: coarser geometry. This is the §6c experiment (G2).
3. **Class/air normalization** — air down-weighting in the loss + per-sampler `air_bias`
   calibration. Critical, and **sampler-specific**: MaskGIT commits its most-confident
   (often air) voxels first so density self-regulates, but the flow sampler reveals
   *random* voxels by rate and over-fills — it needs its own (negative) `air_bias`
   (e.g. −4 vs MaskGIT's ~0 at matched occupancy). [run `…_232913_gen`]
4. **Palette normalization** — factor `(family, variant)` so woods/wools share statistics
   (§9.1). Not a size lever but the materials-generalization lever.
5. **Coordinate normalization (future)** — emit coords relative to the previous voxel
   (deltas) instead of absolute, to shorten/standardize the AR stream and bias toward
   local structure. Untried; promising for both compression and locality.

**Resolved:** AR (with scale normalization) **beats** diffusion on houses — NN-IoU 0.568
vs 0.369 on identical canonical 12³ houses (run `…_232913_gen`, T6a). Lead with AR.

**New normalization to-do (priority):** delta-coordinate encoding (§11.5) to let AR reach
16–24³ without the 12³ fidelity loss; and an explicit occupancy/length prior so MaskGIT
diffusion stops under-filling (or default its sampling to flow matching).

## 12. Generality plan → LEGO & electronics (the funding-demo north star)

**Why this design generalizes.** Nothing above is Minecraft-specific *in principle*: a
build is (a) a set of typed placements with coordinates/orientation and (b) a
connectivity graph. Both target media are the same shape of object:

| medium | "block" | "coords" | "variant/material" | connectivity |
|---|---|---|---|---|
| Minecraft (now) | block id | voxel x,y,z | data value (wood/wool/orient) | 6-neighbour adjacency |
| **LEGO** | part id (LDraw) | stud grid x,y,z | colour code | stud ↔ anti-stud mate |
| electronics | component | board x,y (+layer) | value/footprint | net (pin↔pin) |

So the three reps port directly: **Track A** voxel-token → *piece-token*
`[BOS, (X,Y,Z, PART, ORIENT, COLOR)*, EOS]`; **Track C** block+port graph → LEGO stud
graph / SPICE-style netlist; the **novelty eval** (canonical-grid IoU + duplicate /
diversity) is medium-agnostic. The validity gate generalizes from 6-connectivity →
LEGO stud-mate feasibility + static stability (exactly LegoGPT's physics-rejection idea).

**Concrete LEGO demo plan (pitch-ready milestones):**
1. **Data** — LDraw/LDR official parts library + a model corpus (OMR — Official Model
   Repository; Rebrickable sets; or LEGO-provided data if they engage). Parse LDR
   (part, 3×4 transform, colour) → reuse the `Structure`/graph schema with a `Piece`
   type. *Ask of LEGO: curated build data + part metadata.*
2. **Tokenizer/normalizer** — snap to the LEGO stud grid (built-in scale normalization);
   canonicalize orientation to the 24 axis-aligned rotations; delta-coordinate encoding
   (§11.5) to keep sequences short.
3. **Model** — start with Track A AR (LegoGPT-aligned, our strongest token result) on a
   single set-type ("learn one model class, then expand", the milestone we already
   demonstrate for houses); add the **stud-mate + stability validity gate**.
4. **Eval/Proof** — novelty grid (sample vs nearest real set) + **buildability rate**
   (fraction of samples that are physically connectable & stable) — the LEGO-relevant
   headline metric, and our novelty story already works.
5. **Pitch framing** — "text-free generative design that produces *new, buildable* sets,
   with provable novelty (not memorization) and a curation pipeline that preserves
   colour/part variants." Reuse F2-style figures with LEGO renders.

> Sequencing for the demo: lock the houses result (best so far) → port the AR tokenizer
> to LDraw on one set category → validity gate → buildability metric. Each step reuses
> existing modules; the risk is data access (hence the LEGO ask) and the validity gate.

---

## 13. Rendering for figures (researched 2026-07-06)

Three tiers, replacing/augmenting the matplotlib voxel plot (`renderer/render.py`):

1. **Dense dataset grids (fast, real textures):** `renderer/textured.py` —
   pure-Python textured renderer, WORKING (built 2026-07-06): exposed-face
   culling → per-structure texture atlas (face shading baked: top 1.0 / N-S .85
   / E-W .70 / bottom .55) → pyrender **EGL headless**, orthographic, alpha bg.
   ~0.5 s for a small house, ~2.7 s/tile average over 32³ houses (mesh build is
   the python-loop bottleneck, not the GPU). `renderer/textures.py` maps legacy
   (id,data) → modern texture names (~97% vocab coverage after adding
   plant/flower tiles; rest get flat-color fallback tiles). Textures fetched by
   `python -m blockgen.renderer.textures --fetch` (vanilla jar via Mojang
   piston-meta → `data/textures/vanilla/`, gitignored — Mojang copyright, do
   NOT commit or redistribute; for the paper, optionally re-render with the
   Faithful pack via `--pack`, credit required). Grid mosaics:
   `python -m blockgen.renderer.grid --houses 32 --rows 8 --cols 12 --out …`
   (`--corpus`, `--matplotlib` fallback, `--order largest`).
   **Model-sample figures** (reload checkpoints → sample fresh → textured grid):
   `scripts/render_model_samples.py` (ideas-battery checkpoints) and
   `scripts/render_transfer_samples.py` (transfer-run checkpoints; reconstructs the
   run's exact seed-0 pool vocab before reloading). Both `--samples 48` → `outputs/figures/`.
2. **Quality single renders:** `mcrender` (pip) = Mineways (via Wine) + Blender
   headless; real game geometry+textures, isometric, ~15–60 s/img. Needs
   `apt install wine blender` + Mineways download; voxel→world via `amulet-core`.
   This is the GDMC-papers pipeline. Not yet wired in.
3. **Hero figures:** Chunky (Java path tracer, headless CLI, JSON scenes) on the
   same generated worlds — minutes/img, for 2–6 showcase renders.

**MineRL rejected** for rendering: RL env not a renderer (Java 8, xvfb 2–3×
slowdown + NVIDIA GL conflicts, low-res agent-POV frames, structure placement
via setblock handlers). Related repos (voxelcnn, text2mc-dataprocessor) have no
reusable render pipeline — in-game screenshots only — so textured renders are a
figure-quality edge over the baselines.

## 14. MinecraftACE — LegoACE port (built 2026-07-10)

**Goal:** replicate LegoACE's (VAST-AI, SIGGRAPH Asia 2025, DOI
10.1145/3757377.3763881) native per-brick tokenization + AR training on our
Minecraft data, inside `libs/MinecraftACE` (copy of their released code).
Their scheme: 5 tokens/brick `(x,y,z,rot,type)`, flat vocab
`BOS=0 | 1..pos_range coords (shared bank) | 48 rots | 9,314 types | EOS`,
bricks sorted `lexsort((z,x,y))` (y-primary bottom-up), coords normalized
`c - min + 1`, GPT-2 (uncond) / LLaMA (conditioned) decoder, format-grammar
logits masking (`cur_len % 5`), CLIP-text + DINOv2 4-view conditioning, no
physics constraints (~82% connected). Minecraft blocks are axis-aligned unit
voxels → **4 tokens/block** `(x,y,z,type)` and `%4` masking; no rotation bank.

**Integration = converter, not rewrite** (`blockgen/export/minecraftace.py`):
exports `houses_32` → LegoACE dataset layout (`data/minecraftace/houses_32/`:
dat_dict/pair_dict/rot_dict JSONs, `{train,val,test}_dataset.json`,
`tokens/<split>/<id>_moved.npy` raw `(N,4)` records, `index.json`).
Split = deterministic md5(source_path), corpus-stratified, 2127/114/120.
Vocab: pos_range=32, 422 block classes → **vocab_size 456**. Sequence-length
rule: n_positions=8192, filter n_blocks ≤ 2047 → keeps 2361/2661 (88.7%);
p50=745 blocks (~3k tok), p90=2156 (~8.6k), max kept seq 8146. Filter, don't
truncate (truncated build + EOS teaches "partial = complete").
Round-trip verified token-exact (block_ids exact; data nibble collapses where
the vocab merges variants — same lossiness as serialize.py).

**MinecraftACE-side changes (minimal):**
- `dataset/MinecraftTokenDataset.py` — new; uses the "+1" type-offset
  convention (matches MVNpzDataset/textDataset/logits processor). NOTE their
  `SingleTokenDataset` has a no-+1 offset bug (raw type 0 collides with max
  coord token) — never train with the stock class. + `LengthBucketSampler`
  (shuffled length-sorted batches; big padding win, lengths vary 8..2047 blocks).
- `model/logitsprocessor.py` — added `DynamicRangeMaskingProcessor4` (%4
  grammar, EOS only at record boundary, BOS never; verified stepwise on a real
  sequence over 7,385 steps).
- `train/train_unconditional.py` — `--dataset_name` Minecraft path,
  `--n_positions`, `--attn_implementation sdpa` (5070 Ti sm_120 has no
  flash-attn wheels), `use_cache=False` in the training forward (KV-cache
  memory waste; kept out of config so checkpoints still generate fast),
  fixed `persistent_workers` crash at num_workers=0.
- `model/gpt2.py` — forward absorbs `**kwargs` (transformers 4.57's generate
  passes `cache_position`). transformers must be **<5** (5.x drops
  `add_cross_attention` from PretrainedConfig → their GPT2 subclass breaks;
  pinned 4.57.6 in .venv).
- `inference/infer_uncondition_minecraft.py` — generate with %4 processor
  (their stock `infer_uncondition.py` uses NO grammar processor + retry loop),
  decode to raw records, save `sample-*.npy` + dat/pair dicts. No LDR/Blender.
- `dataset/MinecraftConditionedDataset.py` — MV (pre-rendered 4-view PNGs; NO
  on-the-fly mesh render, no pyrender/osmesa import — clashes with our EGL) and
  text (`text_count=4`) datasets for phases 3/4. Known deviation: no
  sub-assembly-crop augmentation (LegoACE re-renders partial assemblies).

**Decode/eval loop:** `scripts/render_minecraftace_samples.py` — records →
Structure (`records_to_structure` + pair_dict) → `save_grid` textured mosaics +
`evaluate_novelty` + component-count validity. Smoke-validated end-to-end
(68-step model → noise blobs, 52 comps median, as expected).

**Run 1 (COMPLETE 2026-07-10, → results.md T13):** uncond GPT-2 12L/768 (91.7M,
pos-emb 8192), houses_32, batch 1 × accum 16, bf16, sdpa, lr 1e-4 cosine,
**15 epochs** (~1.3 h; deliberate pipeline-test budget), length-bucketed,
per-epoch val loss (added a val loop to their script — it has none), checkpoints
every 500 steps. ~2.2 s/step, 11.2/15.4 GB (batch 2 OOMs).
Val 3.98 → 0.66, still falling at end (no overfit within budget). Samples =
coherent ground planes (sequence-prefix learning; y-first order + 3D-Craft grass
aprons dominate), no walls/EOS yet, ~78% repeated coords. Pipeline validated
end-to-end; longer run + curation tweak are the follow-ups (T13 "Next").
Gotchas fixed along the way: their script never saved a FINAL checkpoint (only
checkpointing_steps multiples — patched); generation max_length must be clamped
to n_positions (device-side assert otherwise — patched in infer script).

## 15. Auto-labeling pipeline (`blockgen/labeling/`, built 2026-07-10)

For image- and text-conditioned generation (LegoACE phases 3/4 + beyond):
- `render_views.py` — 4 views/structure (azim 45/135/225/315, elev 30, 512px,
  white bg — VLM+DINOv2-friendly), resumable, `--workers N` (spawn, one EGL ctx
  per proc; 3 workers coexist fine with training). Sweep of houses_32
  (2,661 × 4 = 10,644 tiles) → `outputs/renders/houses_32/` in flight.
- `templates.py` — free captions from metadata: grabcraft title+category,
  3dcraft (title-less) from dims/materials/height features, text2mc from tags.
- `vlm_captions.py` — Cap3D-style: 4 views + metadata hint → OpenAI API
  (gpt-5-mini default, Batch API = 50% off, strict json_schema output,
  detail=low images, resumable JSONL, `--only-missing-titles` option; key from
  repo .env). Sync spot-check gave accurate architectural captions; full
  2,361-structure batch launched 2026-07-10 (~$2–4) →
  `data/minecraft/labels/houses_32_vlm.jsonl`.
- `build_captions.py` — merge VLM-first + templates → exactly 4 captions/id →
  `data/minecraft/labels/houses_32_captions.json` (template-only version built,
  2,361 ids). Re-export with `--captions`/`--images-dir` patches the dataset
  JSONs in place (no re-tokenization).

**Run 2 — 3D-BPE arm (COMPLETE 2026-07-10, → results.md T14):** `--tokenizer
bpe` in the converter: records = `(x,y,z,piece_id)` over 678 pieces (422 atomic
+ 256 merges via `cluster_bpe.learn_clusters` on the train split; piece vocab
serialized to `<name>_piece_vocab.json`, decode expands patterns). 2× shorter
sequences → +257 more builds pass the length filter. Matched 15-epoch budget:
nn_iou 0.111→0.199, first nonzero raw connectivity (0.047), components 27→10,
and samples show real vertical structure (facades, courtyard, towers) where the
voxel arm produced only ground planes. **BPE pieces adopted as the default
representation for this track** (LegoACE part-library analog, learned instead
of physical). Air is never tokenized in either arm (occupied-only records, like
LegoACE brick lists). LegoACE itself uses NO learned tokenization — fixed
9,314-part vocab; our BPE is the learned counterpart and a paper-worthy delta.

**§14 addendum — linkage audit (2026-07-13).** Re-traced the whole port after the
T13/T14 grids raised "is something mis-wired?" concerns. Verdict: **no linkage
bug** — export/dataset/processor/decode conventions are mutually consistent
(+1 type offset applied in `MinecraftTokenDataset`, inverted in
`decode_to_records`; loss mask `attention_mask[:, :-1]` + the EOS-masked-False
dataset convention correctly trains the EOS target and excludes padding), and
the `*_real_ref.png` grids (real houses through the identical decode+render
path) render perfectly. The bad samples are the documented 15-epoch budget:
~2.0k optimizer steps for a from-scratch 91.7M GPT-2 vs LegoACE's 100–500
epochs × 4–8 GPUs on 55k models; val loss still falling at cutoff. No-EOS
symptom (median = 2047-record cap) compounds undertraining with `top_k=10`
(EOS must reach top-10 against the 32-coord bank at a record boundary to be
sampleable at all). Follow-ups unchanged: train to val plateau, then sampler
sweep (temperature/top_k/checkpoint) + duplicate-coordinate masking processor.
Full operator docs now at `docs/minecraftace.md` (export → train → sample →
render/eval, reproduction commands, extension points, env gotchas).

## 16. Conditioned AR on our stack — LegoACE recipe, our models (2026-07-10, → results.md T15)

After T13/T14 showed the ported 92M GPT-2 underperforms at our data scale, the
conditioning phases moved back into blockgen: **CondVoxelAR2**
(`models/voxel_transformer_cond.py`) = VoxelTransformerAR2 (phase4) + LegoACE's
conditioning recipe — frozen-encoder embeds → `nn.Linear` → prefix tokens
(+ learned prefix-position emb), learned null-cond trained by 10% cond-dropout,
CFG at sampling (`generate_cond`, batched, per-sample EOS tracking).
Data: 3D-BPE piece sequences over ALL houses_32 (max_seq_len 5480 = p95),
conditions precomputed once by `labeling/embed_conditions.py` (DINOv2-base CLS
×4 views; CLIP pooled ×4 captions) → `houses_32_cond_embeds.npz` (50 MB).
Train: `training/train_conditioned.py --cond {image,text}` (bf16 autocast,
cosine, per-epoch val, best/last ckpts). Eval:
`scripts/sample_conditioned.py` — held-out val conditions, CFG, paired-vs-
shuffled fidelity (occupancy IoU + block-palette cosine), target-vs-sample grids.

Result (T15): conditioning works **palette-first** — image arm palette sim
0.301 paired vs 0.201 shuffled (+50%), text +20%; geometry transfer weak
(image +36% IoU over chance, text none). Image > text (pooled CLIP token is a
thin channel; captions homogeneous). EOS rate 0.3–0.5. 36 min/arm at 5.7M
params — the whole conditioning loop now iterates ~20× faster than the
MinecraftACE port. Next levers in T15.

**§16 addendum (2026-07-11, → results.md T16):** the T15-vs-earlier quality gap
was the *regime*, not conditioning: T11/T12's good samples were canon-16³
miniatures (1,600-token full builds, 60–90 ep). Added `--repr voxel
--canon-dim 16` to `train_conditioned.py` (T12-identical tokens via
serialize.py; block vocab saved per-run for decode) and reran both arms:
T12-look samples, image palette fidelity +112% over shuffled (2× the 32³ gap),
EOS 0.8+. Text metric saturated by caption homogeneity though renders visibly
follow captions. Decode/eval unified in `scripts/sample_conditioned.py::
load_run_data` (both reprs). Ladder now: 16³ conditioned ✓ → add adjacency
constraint to `generate_cond` → scale 24³/32³ with longer training.

## 18. Minecraft deployment — live mod + inference server (2026-07-15)

`deploy/inference` (WebSocket server) + `deploy/mod` (Fabric 1.21.1) stream a
build into Minecraft block-by-block as it samples: `/gen`, `/gen <text>`,
`/model` to switch. Server owns everything model-specific (sampling, token
decode, legacy→modern block mapping); the mod just places what arrives. Adding a
checkpoint = one `models.json` entry. See `deploy/README.md`.

Three findings worth keeping, all of which cost real debugging time:

**(a) `CondVoxelAR2.forward` was broken and nothing caught it.** Adding KV-cache
support made `_CausalBlock.forward` return `(x, kv)`, but the cond subclass still
did `h = blk(h, bias)` → `AttributeError` on any forward with ≥2 layers. So
`generate_cond` and `scripts/sample_conditioned.py` were dead on arrival for
every T15/T16 cond run since that change. Fixed (`h, _ = blk(h, bias)`).
The subclass duplicates the parent's forward instead of reusing it, so parent
changes silently desync — worth collapsing if we touch it again.

**(b) The experiment scripts did not save the vocabulary they train against.**
*(Fixed 2026-07-16: both arms now call `save_piece_vocab`/`save_block_vocab` before
training starts, so even a killed run leaves a loadable vocab. Reproduce command and
full pipeline: `docs/reproduce-native.md`.)*
`experiments_native.arm_bpe` writes `cluster_meta.json` (counts only, no
`ClusterVocab`); `arm_flat` builds its `BlockVocab` inline and drops it. A token
id is meaningless without its vocab, so *both T18 checkpoints were unloadable*.
Recovered by re-deriving from the seeded pipeline
(`deploy/inference/scripts/rebuild_native_vocab.py`); both match their
checkpoint's `lm_head` (713 / 435). **The rebuilt BPE vocab is NOT the exported
`houses_32_bpe` one** — same 678 pieces (structural: 422 blocks + 256 merges),
different patterns/md5, because it was learned over the augmented train split.
Swapping them decodes to noise while looking healthy. New BPE runs should call
`save_piece_vocab` at train time; sizes matching proves nothing about patterns —
verify by sampling (native_bpe seed 0 → 1,752 blocks, coherent spruce/stone-brick
medieval palette, matching rows.json's ~942 median occ).

**(b2) `--quick` smoke runs are indistinguishable from real ones by inspection.**
`run_20260715_062404_native` is a 2-epoch/24-merge `--quick` smoke; it writes the same
`outputs/run_<stamp>_native/<arm>/model.pt` layout with *identically shaped* weights
(canon16_flat: 435 lm_head, 5,065,907 params either way). It has no `rows.json`, which
is the only tell. I briefly served that checkpoint as canon16_flat and every integrity
check passed — the flat `BlockVocab` depends only on (seed, val_frac), so it is
byte-identical across runs and the vocab/lm_head cross-check cannot catch a
wrong-checkpoint swap. Sampling tells them apart instantly (2-epoch: 115 blocks, 8
states, 87 of them acacia_planks; 60-epoch: 244 blocks, 20 states, coherent palette).
Use `--stamp` to label smokes.

**(c) Orientation is absent from the corpora, not just from the models.** The
`block_data` values are GrabCraft *texture-variant* indices, not legacy metadata:
every stairs id in the piece vocab carries exactly one data value (53→{2},
114→{7}, 163→{7}) where true metadata would spread 0..7; same for logs (17→
{1,2,3,4}) and the recurring `data=11` oak marker in 6/126/175. Colors (0–15 on
wool/concrete/terracotta) *are* faithful. So no facing information exists to
learn — every stairs block we place is default-facing, and no amount of model
work fixes it. If oriented builds matter for the paper's figures, that is a
**data** problem (re-scrape preserving metadata, or infer facing from neighbours
in a post-pass), not a modelling one.

Block mapping (`deploy/inference/blockgen_server/blockmap.py`): all 422 legacy
pairs → 398 distinct modern blocks, 0 fallbacks, validated against a real 1.21.1
registry dump (`scripts/export_blockmap.py`). Traps that a name-similarity map
gets wrong-but-plausible: legacy 2 "Grass" is `grass_block` (and modern `grass`
was renamed `short_grass` in 1.20.3), 31:1 "Tall Grass" is the *short* plant, and
"Oak Wood" (17:4/162:9) is a **log** — legacy had no bark blocks, so mapping it
to `oak_wood` silently yields the wrong block.

*(Correction, see §19: the §18(c) claim that "orientation is absent from the corpora"
is WRONG. Orientation IS in the raw grabcraft/3dcraft data; the tokenizer discarded it.
Fixed with the `oriented` vocab flag.)*

## 19. Full-corpus conditioned run + orientation fix (2026-07-20)

Built the whole-corpus conditioned-generation pipeline (the data-scaling lever,
research.md §B.8) and trained a native-32³ text-conditioned model on it. See
ideas.md for the approach ratings; deliverables/references.bib (193 entries) now
covers the full related-work set incl. LegoACE, BrickGPT, the adapter/graph-LLM/
3D-CLIP/TAG/attachment lines.

**Corpus (`blockgen/curation/corpus.py` → `data/minecraft/cache/all_32.npz`):**
all four corpora pooled UNFILTERED (no house filter) → quality_filter → dedupe →
**4,712 builds** (grabcraft 2,612 all-categories, 3dcraft 1,902, text2mc 198). NOT
the ~40k raw: text2mc builds are big (median maxdim 96, p90 256); only ~1,400 h5 +
~600 schem fit a native 32³ box. **The box is the binding constraint** — 40k needs
either decimation (vetoed) or a bigger box (fights native_bpe seq length). This is
what motivates the attachment/growth model (ideas.md, §below).

**Pipeline stages, all corpus-agnostic now:** render_views --cache (18,848 imgs) →
make_index --cache (adds top-6 block histogram) → vlm_captions (gpt-5-mini batch,
now feeds block-stats + emits a guaranteed short_tag; ~$5-11, 100% is_build) →
build_captions → embed_conditions --cache --text-encoder siglip (SigLIP text seqs
+ DINOv2). Model: `scripts/train_cond_resampler.py` (ResampledCondVoxelAR2 = #6
Q-Former-lite resampler over the full SigLIP token sequence, not a pooled prefix).

**Fixed an O(n²) cache-load bug** in `load_structures_from_cache` (indexed the NpzFile
inside the loop → re-decompressed the whole object array every row): minutes → 0.1s.
Silently taxed every run, houses included.

**Forced multi-block merges** (`cluster_bpe.learn_clusters force_families`): beds (26),
doors (64/71/193-197), double-plants (175) → single piece tokens (269 forced merges on
the oriented corpus). Also fixes the mod's broken door/bed placement.

**§18(c) was WRONG — orientation IS in the data.** The claim "no facing info exists"
was about the *tokenizer*, not the data. The raw corpus (grabcraft + 3dcraft) has full
stairs facings (block_data 0-7) and log axes (0-11); only text2mc collapsed them. What
threw orientation away was `_token_for`: it only keeps a data value if "id:data" is in
STANDARD_VOCAB, which lists one texture-variant per block, so real facings collapsed to
one token. **Fix:** opt-in `oriented=True` flag threaded through `_token_for` (new
`_ORIENTATION_IDS` = stairs/logs/doors/trapdoors/slabs) → `build_block_vocab` →
`BlockVocab`/`ClusterVocab` (persisted in save/load). Oriented corpus vocab: 427→655
blocks, 774→1,215 vocab; stairs now 8 facings, logs 16 values. The conditioned trainer
doesn't augment so it's clean; **augmented (unconditioned) runs still need the D4
block_data rotation table (§17)** or they get wrong-facing blocks — required for the
attachment model too.

**Attachment/growth model (next flagship, ideas.md #8/#5, research.md E.2).** No box:
seed at bottom-center, BFS across the bottom plane then climb, attach pieces to open
faces (= the ports in `graph_data.py`), pose derived from the connection, per-face
CLOSE = boundary (air ≡ EOS), collision-check-and-resample during sampling. Solves
train-on-everything (any size, no decimation). MVP plan: single-voxel pieces,
unconditioned, over the full 40k. Related: VoxelCNN/3D-Craft, BrickAnything (tree
tokens), SolidGen, GCPN/GraphAF/JT-VAE (valency≈collision + resample), GraphRNN/DiGress.

## 20. Attachment/growth Phase 0 — extractor + ordering bake-off (2026-07-21)

Results/tables: `results.md` T21. This section is methods + repro.

**New modules.**
- `blockgen/utils/attach_order.py` — `structure_to_attach_ops` / `attach_ops_to_structure`
  / `roundtrip_iou`. Ops are `SEED(piece)`, `ATTACH(piece, direction)`, `CLOSE(direction)`;
  direction indexes `graph_data.PORT_DIRECTIONS`. **No coordinate is ever emitted** — the
  parent voxel is implicit in frontier position, which the decoder reconstructs by
  replaying the same ordering. Frontier is a heap keyed by `(priority, tiebreak)`.
- `blockgen/utils/attach_vocab.py` — `AttachVocab`, the op↔token-id bridge. Layout:
  `0` BOS, `1` EOS, `2+d` CLOSE, `8+p` SEED, `8+P+p*6+d` ATTACH. Vocab is a function of
  the pieces the corpus actually uses (195 on 4k builds → 1,373 tokens).
- `scripts/ordering_bakeoff.py` — Phase-0 gate + training-free ordering comparison.
- `scripts/train_attach_corpus.py` — Phase-1 trainer, one arm per ordering.

**The op stream is a token sequence, so Phase 1 needs no graph encoder.** This is the
shortcut that made the MVP land in one session: a plain causal transformer over op tokens
*is* the autoregressive attachment model, so `VoxelTransformerAR2` +
`train_ar_ext.train_from_sequences` are reused unchanged (`pe="sin"` keeps the fused
`is_causal` SDPA path, which is what makes seq 4096 affordable). All geometry lives in the
decoder. The GNN encoder of `implementation_plan.md` §3 is a Phase-2 capacity upgrade, not
a prerequisite.

**ORDERING IS A PARAMETER, NOT A CONSTANT.** Every ordering is a priority function in
`attach_order.ORDERINGS`; adding one is ~4 lines. Shipped: `bfs_bottom_center` (the plan's
canonical order), `layered_raster`, `radial`, `dfs` (LIFO frontier, see `LIFO_ORDERINGS`).

**Constraint discovered the hard way — decode-availability.** A priority function may only
read what the decoder also has: face coordinates and the partial structure. Reading
ground-truth occupancy desyncs the frontier and round-trip IoU collapses to 0.23. Priorities
are also written **relative to the component seed** so they are shift-invariant (encode
works in the cropped build frame, decode in a padded working grid). If you add an ordering,
the round-trip test is the guardrail — it catches this immediately.

**Multi-component policy (the §9 open question), decided.** `multi_component="largest"` is
the corpus default: keep the largest 6-connected component, drop the rest — costs 0.4% of
voxels. `"reseed"` (one SEED per component) is available but **geometry-lossy BY
CONSTRUCTION**: the op stream encodes intra-component connectivity only, so nothing records
where disconnected components sit *relative to each other*. `"reject"` raises. This is a
real representational limit, not an implementation gap.

**Repro.**
```bash
# Phase-0 gate + ordering bake-off (CPU, ~5 min for 1200 builds)
.venv/bin/python -m scripts.ordering_bakeoff --limit 1200 --human-limit 400

# Phase-1: one arm per ordering
.venv/bin/python -m scripts.train_attach_corpus \
  --limit 4000 --max-seq-len 4096 --epochs 24 --batch-size 4 \
  --orderings bfs_bottom_center,layered_raster,radial,dfs
```
Note `.venv/bin/python` — `torch_geometric` is only in the venv, and `attach_order`
imports `graph_data` for `PORT_DIRECTIONS` (single source of truth for the 6 directions).

**Gotchas.**
- `AttachVocab` is saved **before** training starts (`attach_vocab.json` in the arm dir) —
  §18(b) burned two unloadable checkpoints on exactly this.
- Sampling runs **unmasked** (SEED is not banned after position 0) so the reported validity
  is *learned*, not filtered. Turning that mask on is the filter arm of the §10 ablation;
  do not turn it on silently or the headline validity becomes an artifact.
- Op count is ~2.91/voxel, so `max_seq_len` is the binding constraint, not the box:
  4096 fits ~70% of `all_32` builds. Raising it to 8192 reaches ~91% and is the cheapest
  next lever on "train on everything".

**Addendum (same day, after the first Phase-1 arms).**

- **Sparse decoder (bug fix, and a design correction).** `attach_ops_to_structure`
  originally materialized into a preallocated `256³` numpy array. A free-running model
  grows past any such volume: large overruns raised `IndexError` (killing samples), and —
  far worse — **small negative indices silently wrapped around**, scattering phantom
  blocks on the opposite face and manufacturing fake disconnected components. This
  corrupted the first arm's validity numbers (reported 0.154, actually ~1.0). The decoder
  is now a **dict of cells materialized and cropped at the end**, so decode coordinates
  are genuinely unbounded. A fixed working grid was the last hidden box in the pipeline —
  removing it is on-thesis, not just a bug fix. Round-trip re-verified: 1.0 on synthetic
  cases and on 120 corpus builds × 4 orderings, 0 failures.
- **Sampling temperature dominates sample quality.** T=1.0 → filaments (thickness 2.99);
  T≈0.5 → solid (4.75 vs real-build 4.14). Diagnose with `scripts/attach_diagnose.py`
  (prints a temperature sweep and a CALIBRATION vs STRUCTURAL verdict); re-sample trained
  arms with `scripts/attach_resample.py --temp 0.6`. **Always report thickness next to
  loss** — held-out NLL was excellent (1.077 bits/op) while samples were 1-voxel tendrils,
  so NLL alone is not a usable quality signal for this representation.
- **The n-gram proxy screens well; it does not separate near-ties.** Final trained order
  was `layered_raster` 1.0723 < `bfs_bottom_center` 1.0773 < `dfs` 1.5116 < `radial`
  2.9989. The proxy recovered this **exactly except for an adjacent swap of the top two,
  which differ by 0.5%** — so a ~5-minute CPU screen predicted the result of four GPU
  runs. It does not predict achievable loss (proxy values run ~2× the trained ones).
  Screen candidate orderings with it; settle near-ties by training.
- **What actually matters in an ordering: finish a layer before climbing.** The two
  gravity/layer-structured orderings tie and both beat `dfs` (+41%) and `radial` (+180%).
  Locality per se is *not* the property — `radial` is maximally local and is the worst arm
  by a wide margin.
- **New scripts**: `scripts/attach_diagnose.py` (temperature sweep + verdict),
  `scripts/attach_resample.py` (re-sample + render arms at a chosen temperature),
  `scripts/attach_report.py` (cross-run table incl. **bits/build**, the only
  representation-agnostic description-length metric — `bits/op` is not comparable across
  tokenizers).
- **`max_seq_len` is a data filter, not just a compute knob.** Over 600 `all_32` builds:
  unfiltered occ_p50 **1,058**, but only 65% pass seq≤4096 and those have occ_p50 **755**
  (thickness unchanged, 4.22 vs 4.21). The filter discards *large* builds specifically, so
  a seq-4096 arm trains on a systematically smaller distribution than the corpus and its
  generated size must be judged against **755, not ~1,000**. Report the *filtered*
  reference statistics in every arm, or the model looks worse than it is — and the "no
  box" claim looks stronger than it is (a volume constraint was traded for a length
  constraint with the same selection pressure).

**Addendum 2 — the metric that overturned the ordering conclusion.**

- **Run the perceptual metric before believing ANY ordering/quality claim.**
  `scripts/attach_perceptual.py` (CMMD + CLIP over textured renders, shared real
  reference, plus a **real-vs-real floor** so the numbers have a scale). At n=24 the CMMD
  ranking came out an **exact inversion** of the held-out-loss ranking (ρ = −1.0):
  `radial` is worst by loss (2.9989 bits/op) and best perceptually (CMMD 0.980 vs floor
  0.469; CLIP 0.2675 vs floor 0.2774), while `layered_raster` is best by loss and worst
  perceptually. **bits/op, human-order agreement, and val NLL all agreed with each other
  and all three disagreed with the eye** — the T20 lesson recurring in a new
  representation. Confirmation at n=64 × 3 seeds is the gate before this is load-bearing;
  always pass `--seed` so replicates coexist rather than overwrite.
- **Always emit a real-vs-real floor with CMMD.** Without it a CMMD of 0.98 is
  uninterpretable; against a 0.469 floor it means "about 2× the best achievable at this
  n". The floor is computed by splitting the reference set in half.
- **Sequence length was the binding constraint on sample size, confirmed.** 4096 → 8192,
  sole variable, matched sampler/decoder/temperature: generated **median** occupancy
  242 → 710 (2.9×), thickness preserved, validity 1.00, loss −12.6%. Retention
  70.2% → 88.4% → 92.3% @16384 → **100%** of the 300-build reference sample at 16384.
- **Design lesson (a run I wasted).** The seq-16384 arm dropped `batch_size` 2→1 and
  `epochs` 24→20 to fit the context, so it varies three things and cannot test the
  sequence-length hypothesis at all. When raising `max_seq_len`, hold the effective batch
  fixed with **gradient accumulation** and keep epochs constant — otherwise the arm is
  uninterpretable.
- **Report the MEDIAN of generated occupancy.** The distribution is right-skewed:
  seq-8192 @T=0.5 is median 710 but mean 1,272. `attach_resample` reports p50 and
  `attach_diagnose` reports the mean — do not quote them against each other.

**Addendum 3 — Phase-1 verdict: the op-token shortcut fails, and why.**

- **`scripts/attach_prefix_test.py` — the decisive diagnostic.** Teacher-force the first K
  ops of a *real* build, then let the model continue. Drift predicts a real prefix holds
  the model on-distribution; blindness predicts it does not help. Result: a real prefix
  makes things **worse**, and the model's close-rate climbs monotonically with prefix
  length (0.659 → 0.799 → 0.869) while the **ground-truth** continuation close-rate is
  flat (0.606 → 0.601). Handed more real structure, the model shuts the frontier down
  faster. **VERDICT: BLINDNESS.** It has learned the op-frequency marginals (free-run
  close-rate 0.659 vs true 0.606; attach-rate 0.337 vs real 0.346) but cannot read its own
  op history as geometry.
  *Always compute the ground-truth null for a position-dependent statistic* — the raw
  close-rate climb looks like correct late-build behaviour until you check that the true
  rate is flat.
- **Consequence: `implementation_plan.md` §3's graph/state encoder is a PREREQUISITE.**
  The Phase-1 MVP deliberately skipped it (a plain causal transformer over op tokens, so
  the whole AR stack could be reused). That shortcut is what fails: with no view of the
  partial structure, per-face decisions are made blind. Two earlier claims in this session
  that the encoder was unnecessary both rested on the buggy `thickness` metric and are
  withdrawn. Phase 2 = encoder over the placed structure with frontier faces as nodes;
  everything else built here (extractor, vocab, orderings, diagnostics, perceptual eval)
  plugs in unchanged.
- **Three gates for every future arm on this track:**
  1. Open `samples_*.png`. Three scalars (thickness, occupancy, CMMD) each moved the
     "right" way while the renders showed lines, plates and spheres.
  2. Use **corrected** thickness (zero-padded, not `np.roll`) and quote CMMD as a multiple
     of a real-vs-real floor computed at the **same** n (the floor moves 0.469 @n=24 →
     0.100 @n=64).
  3. Treat **NLL as adversarial**: `bits/op` correlates +1.000 with output solidity across
     orderings because a 1-voxel line is the most compressible op stream. The loss-minimising
     output is degenerate, so low loss is evidence of collapse, not quality.

## 21. LLM-baseline track — BrickGPT-style text→build (2026-07-22)

The "Track D" LLM baseline from `research.md`: does a pretrained LLM, prompted or
LoRA-finetuned, generate our houses from a caption? Three conditions, all sharing one
caption→block-lines format, name map, parser, and TEXTURED render so they compare
apples-to-apples (`scripts/train_llm_brickgpt.py`, `scripts/zeroshot_brickgpt.py`).

**Format (v1, per-voxel).** One line `<block_name> <x> <y> <z>` per occupied voxel,
`(y,z,x)` raster order (mirrors `serialize.structure_to_tokens`). `build_name_map`
gives a reversible readable-name ↔ `(id,data)` map (238 names over the houses corpus).
Completion-only loss, exactly like BrickGPT trains bricks conditioned on the prompt.

**Results.**

| Condition | Model | Parse rate | Coherence |
|---|---|---|---|
| zero-shot | gpt-5-mini (no example) | 0.26 | mostly flat floors + partial shells |
| one-shot | gpt-5-mini (1 in-ctx build) | 0.29 | same; median blocks *dropped* 78→40 |
| **finetuned** | **LoRA Qwen2.5-Coder-1.5B** | **0.87** | several read as houses (walls+windows+roof) |

- **Finetuning wins decisively** — 3× the parse rate and the only condition producing
  house-shaped output. Confirms the BrickGPT thesis: a small finetuned model beats a
  big *prompted* one on this structured spatial task (VoxelCodeBench's finding).
- Frontier prompting emits *valid-looking* blocks but *spatially incoherent* builds;
  ~75% of gpt-5-mini's blocks are dropped as out-of-vocab (it uses names outside our
  238). One example barely helped (0.26→0.29) — one build doesn't teach the vocab.
- **gpt-5 reasoning-token trap:** gpt-5* are reasoning models; reasoning tokens count
  against `max_completion_tokens`, so a small budget → EMPTY content. Fix:
  `reasoning_effort="low"` + large budget (16k). This is a serialization task, not a
  reasoning one.

**Recipe notes that mattered (v1).** Per-voxel costs ~10.7 tokens/block, so a median
house is ~7.8k tokens and only **77/2661** builds fit a 2048-token cap (3 fit 1024).
20 epochs (not 5) needed so the diluted prompt→first-block transition token accumulates
enough signal to learn to *start* the format; effective batch 4 (not 16) for enough
opt-steps on the tiny set; micro-batch 1 + chunked CE loss to keep the 152k-vocab logits
inside a shared 16 GB GPU. Failure modes remaining: a few builds collapse to a flat slab
or single stick; smallest builds train best (least first-token dilution).

**v2 — serialize PIECES, not voxels (`scripts/train_llm_pieces.py`).** Emit one line per
3D-BPE piece (`<piece_name> <x> <y> <z>`) reusing the AR track's cached piece vocab
(`data/minecraftace/houses_32_bpe`, 678 pieces). Piece name = `<majority_block>[_k]` —
grounded in caption vocab, reversible (name→piece id→pattern placed at anchor).
Serialize↔parse verified EXACT on 7/7 builds. Token scope (`scripts/scope_piece_tokens.py`):

| | tokens/block | median/build | fit ≤2048 | fit ≤4096 |
|---|---|---|---|---|
| voxel (v1) | 10.65 | 7765 | 77 | 462 |
| piece (v2) | 6.49 | 5026 | **308** | 1043 |

Per-block savings is only 1.64× (piece *names* are multi-token; unmerged voxels stay
atomic), but the decisive win is **4× the training data at the same cap (77→308 builds)**
— the binding lever on a tiny dataset. Full run: 20-epoch LoRA, same recipe as v1,
`outputs/run_20260722_072022_llm_pieces` (297 builds fit; some houses lack captions).

**v2 outcome (2026-07-22) — parse ✅, coherence ✗ (honest mixed result).**

| | parse | data | coherence |
|---|---|---|---|
| voxel v1 | 0.87 | 73 | ~5 good houses, ~40% collapse |
| piece v2 | **0.99** | **297** | ~6 good houses, ~40% collapse |

- **Improved:** parse 0.87→0.99 (pieces are a more structured, reliably-formatted target)
  and the 4× data landed as scoped. Both stated v2 goals met.
- **Did NOT improve:** geometric coherence. ~40% of val samples collapse to a flat slab,
  a rail, or a stick (the "wool tree on a single trunk" caption produced literally a
  single pole). The good ones (walls+roof+foundation) are on par with v1's best, not better.
- **Why pieces didn't fix it — two real reasons.** (1) The extra data is also *harder*:
  v1 trained on the 73 SMALLEST builds (~180 blocks median); v2's 297 reach ~400 blocks,
  diluting the format-initiation signal (v1's own "smallest builds train best"). (2) A wrong
  piece has a bigger **blast radius** — it stamps a whole multi-voxel pattern, so one error
  becomes a floating plank-run/rail, not a single stray voxel. The collapses are the model
  emitting a degenerate low-entropy piece stream. Val loss also climbed 0.37→0.41 (epochs
  8–20), mild overfit on the tiny set.
- **Reading.** Token efficiency alone does NOT buy coherence on a small heterogeneous set.
  Same lesson as §20: a flat token stream with no view of the partial structure places
  blind. Pieces remain the right target for *scale* (4× data, 0.99 parse) and are what ports
  to LEGO, but the coherence lever is elsewhere. Candidate next controls: (a) hold build-size
  fixed (rerun v2 on only ≤200-block builds) to isolate the format effect from the data-shift
  confound; (b) bigger pieces (more merges → fewer lines); (c) the on-thesis fix — a spatial
  encoder over the partial build, not more/denser tokens.

---

## 22. Agentic track (Track E) — the LLM writes the build *program* (2026-07-28)

`blockgen/agentic/`, docs in `docs/agentic.md`. §21's LLM baseline asked "can a model
emit our voxels?"; this asks the prior question — **is one-token-per-voxel the right
output format at all?** Track E replaces it: a frontier LLM emits a short program in a
WorldEdit-flavored command language, and an executor runs it onto a voxel canvas.

**Why the format change is the point.** §21 measured 10.65 tokens *per block*, so a
median house is ~7.8k tokens and only 77/2661 builds fit a 2k cap. A command is ~9
tokens and places tens–hundreds of blocks (measured: **24.9 blocks/command**). The
consequences are structural, not incremental: builds are no longer context-bound,
canvas size becomes a run-time argument instead of a training decision, and text/image
conditioning is free with the base model (no labeled corpus, no encoder, no training).

**Architecture (each layer independently swappable — that was the design constraint).**

| Module | Role |
|---|---|
| `blockstate.py` | modern names (+ `[facing=]`/`[axis=]`/`[type=]` states) → legacy `(id,data)`; state bits match `deploy/…/blockmap.py`, so a facing survives into a live server |
| `canvas.py` | bounded voxel buffer, clipping + change counting in ONE place; `structure_to_canvas` seeds it from a real build (the *editing* seam) |
| `dsl.py` | 14 commands via a `@register` decorator; parser + executor are separate |
| `providers.py` | `LLMProvider` interface; OpenAI / Gemini / Anthropic / scripted, disk response cache, cost accounting |
| `prompts.py` | system/plan/build/repair/critique — **the command reference is generated from the registry**, so prompts can't drift from the language |
| `examples.py` | hand-written in-context programs + a retrieval hook (keyword now, CLIP later) |
| `tasks.py` | prompt sets: `short`, `detailed`, `large`, `captions:k` (real corpus captions) |
| `agent.py` | the loop: plan → generate → execute → repair(N) → critique(M) |
| `report.py` | run artifacts; structure cache in the **standard** `.npz` format |

**Decisions that turned out to matter.**

- **Parse ≠ execute.** Syntax errors are reported with line numbers before anything
  runs; a failing *command* is skipped while the rest of the program executes. A
  program with 3 bad lines out of 80 still yields a build plus an exact fix-list. This
  is what makes the repair loop possible at all — the feedback is symbolic and precise,
  unlike "your voxel cloud is wrong".
- **Zero-change commands are warnings.** Almost always a coordinate bug, and the
  cheapest useful signal to hand back.
- **Unknown block = error, not a stone fallback.** A silent substitution would hide
  palette drift inside a grey blob and make the metrics lie.
- **Rollback on regression.** If a repair/critique round returns an empty build, the
  previous build is kept. A loop that destroys a good build is worse than no loop.
- **Response caching by request hash.** LLM sampling is the expensive
  non-reproducible step; cached re-runs are free *and* byte-identical, so re-rendering
  or adding a metric doesn't re-roll the experiment.
- **Stepped roofs need risers** (found by the connectivity metric, not by eye). A
  gable that steps up-and-in touches only diagonally, so a roof built the obvious way
  scored as 6 disconnected components under the repo's 6-connectivity validity notion.
  `gable … riser=true` (default) closes each step's vertical face. Regression-tested.

**Reasoning-token trap** (inherited from §21): `gpt-5*` bill thinking against the
completion budget → small cap = empty content. Default budget 16k;
`--reasoning-effort low` is the cheap setting for what is mostly a serialization task.

**How to run.**

```bash
.venv/bin/pip install -e '.[agentic]'          # keys go in .env
.venv/bin/python -m blockgen.experiments_agentic --config agentic-scaffolding
.venv/bin/python -m blockgen.experiments_agentic --quick --provider mock   # offline
.venv/bin/python scripts/run_agentic.py "a small oak cottage" --plan --examples 1 \
    --repair-rounds 1 --critique-rounds 1
.venv/bin/python scripts/run_agentic.py --list-commands   # what the model is told
```

Arms: `zeroshot | oneshot | plan | repair | critique | full` (all see the same
prompts). Configs: `configs/experiments/agentic-{scaffolding,detail,large}.yaml`.
The detail ablation is **paired** — `captions:0` and `captions:2` describe the same
builds at different richness under the same seed.

**Validated so far.** 57 tests (`tests/test_agentic_{dsl,agent}.py`) covering block
resolution, parsing, every command, the failure paths, the whole agent loop through
the scripted provider, the cache, and the run artifacts — all offline. One live
build: see results.md T22.

**Open (the honest gaps).**
1. **No novelty number yet.** NN-IoU vs `houses_32` is the natural next measurement,
   and the one that decides whether these builds are "new" in the paper's sense.
2. **No inverse compiler** (build → program), so there's no supervised signal here and
   no mined examples; the in-context examples are hand-written technique demos.
3. The scaffolding arms are **wired but not yet measured** at n>1 — plan/example/repair/
   critique each cost calls and none has earned its keep yet.
4. Language limits: single clipboard slot, no variables/loops/functions. Those are the
   next primitives if programs start hitting repetition limits.
5. The comparison to §21 must be made **at equal cost** (tokens and dollars per
   coherent build), not on parse rate.

### 22b. Serving Track E in Minecraft (2026-07-29)

The agentic track is now a servable *kind* in `deploy/inference` alongside the trained
checkpoints, so `/gen <anything>` in game runs the LLM→program→execute loop.

**Model groups.** An agentic entry is a *provider*, not a checkpoint, so listing every
API model as its own registry row would bury the four trained models under a wall of
names. An entry may now declare `"models": [...]`; it shows as **one row** in `/model`
and members resolve on demand as `agentic:<model>` (cached after first use). The mod's
`/model` argument became a greedy string, which is what lets `/model agentic list` and
`/model agentic gemini-3.5-flash` parse as two words instead of forcing a colon.
Vendor is inferred from the model name (`gpt-*`→openai, `gemini*`→gemini,
`claude*`→anthropic); explicit `vendor:model` always wins. `strict_models: false`
lets a model released after the list was written through anyway.

**Two entries, differing only in loop:** `agentic` (1 example + 1 repair, 48³) and
`agentic_plus` (plan + 2 repairs + visual critique, 64³). Every loop knob is a JSON
field, so a new preset is an entry, not code.

**Streaming is execution, not generation.** The neural backends stream because
sampling is incremental. The agentic backend has the whole program before it places
anything, so it *replays* it one command at a time (`ProgramRunner(..., track_voxels=
True)`), one message batch per command, labelled with the command. In world you watch
the foundation, then walls, then the doorway being cut, then the roof — cleared voxels
stream as `minecraft:air` so openings really open. The LLM wait (25–70 s) is up front
and unavoidable; only the replay is paced.

**Protocol compatibility.** `step` and `stats` ride as *optional fields on existing
message types* (`blocks`, `done`) rather than as new types — the mod's dispatcher
errors on an unknown type, so a new type would have broken every mod built before
this change.

**Cost is reported in chat**, per build: provider, commands, tokens, dollars. A model
missing from `providers.PRICES` reports **cost unknown**, never `$0.00` — printing
zero for a paid call reads as free. (Found immediately: `gemini-3.5-flash` is newer
than the price table.)

**Two defects this work surfaced, both fixed.**
1. *Every sample of a prompt was identical.* LLM sampling is not seedable and the
   response cache keys on the request, so `--n 4` returned one build four times. Fixed
   with a variation rider in the request (`build(..., seed=k)`), which both asks for a
   different design and changes the cache key.
2. *Roof stairs all faced east.* `gable` placed one facing for both slopes, which
   reads as a staircase. Now the two slopes get opposing legacy facing bits (an
   explicit `[facing=]` in the program still wins).

**Verified live.** `agentic:gemini-3.5-flash` over the real WebSocket: 2,915 blocks,
48 commands, 0 failed, 65 s (`gpt-5-mini` runs ~35 s). The Fabric mod compiles clean
against JDK 21. 85 server-side tests, all offline via `agentic:mock`.

**Architecture decision — the server stays in `deploy/`, for now.** The question came
up whether `blockgen_server` should move into the `blockgen` package. Recommendation:
**no for the transport, yes eventually for the domain layer.**
- The FastAPI/WebSocket server is a *deployment artifact*: it version-locks with the
  Fabric mod, carries its own deps (fastapi/uvicorn), and nothing in the research
  pipeline imports it. Folding it into the library would put a web framework in the
  dependency path of every experiment.
- But `blockmap.py` (legacy→modern block states) is *domain* logic, and it is now the
  inverse of `blockgen/agentic/blockstate.py` (modern→legacy). **Two inverse mappings
  maintained in different trees is the real risk here** — the bit conventions are
  currently kept in sync by comment and by test, not by construction. If a third
  consumer appears, move `blockmap` into `blockgen/utils/` and have the server import
  it, rather than moving the server.
- The cheap fix available today, unrelated to where code lives: `blockgen_server` is
  not installable (`run_server.sh` sets `PYTHONPATH`, tests do `sys.path.insert`).
  Adding it to `pyproject` as a package with a `deploy` extra would remove those
  hacks without moving a line.

## §23. The evaluation suite (`blockgen/eval/bench`, 2026-08-03)

Motivation: T17/T20 retracted `nn_iou` and T21 retracted `cmmd`, both *after* they
had been used to adjudicate arms. The suite's organizing rule is therefore that a
metric must pass a corruption ladder before it is allowed to rank anything, and
`full.py` mechanically refuses to report one that has not.

Design decisions worth remembering, and the measurements behind them:

- **Novelty is co-equal with realism, on the same row.** `train_verbatim` (literal
  training copies) scores MV-DINO-KID 0.015 — second only to real data — and is
  caught only by `dino_nn_percentile = 0.000`. Realism-without-novelty has a
  trivial winning strategy, and the AR baseline already exploits it (31% dupes).
- **Views are pooled per structure, not flattened.** `perceptual.render_views`
  treats 4 views as 4 samples; measured, that biases the KID null to +8e-5 at n=16
  (2σ) while pooling gives 0.000. Pooling also makes the bootstrap unit the
  structure, which every estimator here already assumes.
- **Coherence is distance-to-real.** 66% of real val houses are single-component
  natively (43% at canon-16), so `lcc_ratio = 1.0` is as wrong as 0.2. The
  scorecard type has no representation for a bare rate.
- **Validity gates ≠ sensitivity.** Ordering/invariance/self-consistency failures
  mean the metric measures the wrong thing → null. "Cannot resolve a 40% cut at
  n=64" is a power limit → recorded, still reported. Conflating them either
  disqualifies good metrics or launders bad ones.
- **The renderer fits the camera to the bbox**, so absolute size is invisible to
  every image-space metric. Consequences: cross-resolution comparison is legal
  (the point of T20); block-count and bbox distributions must live permanently in
  the voxel tier; and an *end-slab* deletion probe is a no-op, which is why
  `chunk_delete` cuts strictly interior.
- **`perceptual.py` is deliberately not patched** despite `cmmd` being the biased
  estimator with a σ=10 kernel that is nearly linear on unit-norm features
  (median heuristic says 0.70). Patching it would silently invalidate T20/T21;
  the ladder scores it as a labelled legacy metric instead.

Three bugs the tests/ladder caught in the suite's own code, all of the
silently-wrong-number kind: palette JSD returned **0.0** (perfect) when every
generated block was outside the reference vocabulary (fixed: union alphabet);
material agreement matched substrings, so "ice" fired on "a nice building" (fixed:
word boundaries); and the ladder's probe set was category-biased because corpus
order is contiguous by category, inflating every noise floor ~45%.

Cost is not a constraint: 21 ms/view to render, 2.4 ms/image to embed, so the full
2661-build corpus caches in ~4 min / 25 MB and a 128-sample arm scores in ~54 s.
The FAST/FULL split is about trust and dependencies, not compute.

### §23.1 First cross-track comparison (T23d)

Both tracks fail in opposite directions, and the split is clean enough to steer by:

- **native_oriented** learned the corpus *palette* (JSD 0.054 vs agentic's 0.228)
  but not its *structure*: interior volume 0.005 against a real 0.114, and the
  highest floating-block fraction in the table (0.155 vs 0.112). Its KID (0.187)
  is well clear of the real floor (−0.001).
- **agentic** matches real connectivity exactly (0.983) with the least floating
  mass, and wins realism (KID 0.107) — but its palette is 16× further from real,
  which is what a small hand-written DSL palette buys. n=12, so provisional.

Neither memorizes (dup 0.000, novelty percentile at/above the real floor), so the
31% duplication of the earlier AR baseline is not a property of the current arms.

The actionable item is shared: **nothing builds interiors.** `enclosed_air_ratio`
is a FAST-tier metric, so this was findable without a GPU and should gate future
runs rather than being discovered at eval time. Corpus curation already dropped
675 builds for `no_interior` — the models are not learning the property the
curation was selecting for.

Sampling a served checkpoint into the bench format is `scripts/sample_to_npz.py`,
which reads architecture from `deploy/inference/models.json` rather than
re-declaring d_model/layers/pe. Generation has no KV cache
(`train_ar_ext.generate_from_prefix` re-runs the prefix each step), so native
sampling is ~10 s/build at a ~1750-token median — 64 builds took 10.4 min. That
cost, not the eval, is what makes n>=256 a scheduling decision.

## §24. Pick-and-place: learned placement over a growth graph (2026-08-16)

Track: `blockgen/models/pick_n_place.py`, `utils/growth_order.py`,
`training/train_pick_n_place.py`, `scripts/train_pick_n_place.py`.

    Picker : G     -> P(V)    which piece next (or STOP)
    Placer : G x V -> P(E)    which open face to attach it to

**Why this and not T21 again.** T21 built the same growth process with placement
*implicit* — `attach_order`'s frontier heap decided where each piece went and the
model only chose the piece. The verdict was blindness, not drift: the model
matched op-frequency marginals almost exactly while its close-rate *climbed*
0.693 → 0.838 → 0.896 the more real structure it was handed, against a flat
ground-truth 0.606. It could not read its own op history as geometry.
`implementation_plan.md` §3 concluded a state encoder over the placed structure
is a prerequisite. This is that encoder, plus a head that makes "where" a learned
decision.

**Representation (`growth_order.py`).** A build is nodes in placement order; node
0 is the seed, every later node names an earlier parent and one of six faces.
Phase 0 gate, same as T21's: replay walks parent/direction from the seed and
never reads the stored coordinates — **round-trip IoU exactly 1.000 (min 1.000)**
across bfs/layered/dfs on 200 real builds, block retention 0.994 (the loss is
non-largest components, dropped on purpose and reported separately).

One thing that bit immediately and is worth remembering: the first round-trip
read 0.779 because the *check* compared the replay against the full original
structure while encoding had deliberately dropped minor components. Comparing
like with like is `sequence_to_reference`. A representation gate that measures
two things at once will report a failure that is not there — or hide one that is.

**Legality is precomputed, not searched.** A cell is occupied exactly when its
node index is ≤ t, so storing the node that eventually fills each neighbour cell
turns the whole time-varying mask into a comparison against t:
`legal(i,d,t) = (i<t) and not (0 <= nbr[i,d] < t)`. Vectorized over all steps, no
per-step set membership. Verified: 11,940/11,940 ground-truth placements legal,
seed row empty.

**Connection encoding instead of positional encoding** (the design question that
started this). Node inputs carry only what was knowable at placement — own piece,
arrival direction, parent's piece — and geometry enters *attention* as a learned
bias on the clamped relative 3D offset between every pair of placed nodes. An
edge is the special case `offset == unit direction`; the same table also covers
two-apart and diagonal. Encoding a node's final 6-neighbourhood as an input
feature would leak nodes placed later, inflate training accuracy, and evaporate
at sampling time — the leak is tested against (`test_encoder_is_causal`).

**The placer is a pointer, not an n² map.** Candidates are open faces (≤6N, and
mostly illegal), so query = current state ⊕ chosen piece, keys = (node,
direction). Mask with −inf **before** softmax; masking after and renormalizing is
numerically worse and makes the CE target inconsistent with what is sampled.

**Read `place_lift`, not `place_acc`.** The mask already removes most candidates,
so accuracy has a floor of 1/n_legal. Reporting the ratio is what distinguishes
"learned where things go" from "the mask did it". At 3 epochs on 200 builds:
place_acc 0.421 against chance 0.014 — **29.6× lift** — so the placer is doing
real work early. That number, not the loss, is the one to watch.

**Known MVP limits.** Sampling re-encodes per step (O(N²) per build); §3's cached
incremental message passing is the fix and is a sampling-time concern only.
Training is one encoder pass per build via causal masking. `legal` is
[B,N,N,6] so max_nodes drives memory — 256 is comfortable, and the corpus median
is ~900 nodes, so most builds train on a truncated (still connected, still valid)
prefix.

### §24.1 First result, and a bug that only free-running generation could see

30 epochs, 1,863 builds, max_nodes 192, 1.17M→13M params, **1.6 min** on one GPU.

| | before fix | after fix |
|---|---|---|
| val place_acc | 0.715 | 0.713 |
| place_lift | 84.0x | 83.8x |
| **median blocks generated** | **30** | **187** |

Teacher-forced metrics moved by 0.002. Generation went from 30 blocks to 187
(training length 192). The cause was a train/inference skew in one line:
`node_features` normalized the step index by the *current* sequence length, which
in training is the padded batch max (192) and during generation is however many
nodes exist so far. Three nodes in, every node read as "end of build", the picker
had learned STOP-at-1.0, and it stopped almost immediately.

**The lesson is the one T21 paid for.** Every teacher-forced number was excellent
and *stayed* excellent with the bug in place — pick/place loss, accuracy, and an
84x lift over the legal-face baseline all looked like a working model. Only the
free-running median-block count showed it. Any metric computed with the ground
truth fed in is blind to this whole class of failure, so a growth model needs at
least one free-running number in its training log, every run. `place_lift` tells
you the placer works; `median blocks` tells you the *loop* works, and they are
independent claims.

Guarded now by `test_prefix_features_match_full_sequence` — a prefix must encode
identically inside a longer sequence, which is the general invariant. Anything
computed from the current length violates it.

Qualitatively (`outputs/run_20260816_081506_pnp_fixed/samples.png`): solid,
connected, material-stratified massing — floors, walls, glass panes, doors, grass
sitting on dirt. Not houses. But T21's collapse modes were 1-voxel filaments,
flat plates and balls, and none of those appear, which is the first evidence that
the state encoder addresses the blindness it was prescribed for.

### §24.2 The prefix test, and a confound in my own test

Ran T21's blindness diagnostic against pick-and-place. First attempt, at
`max_nodes=192`:

| prefix | K | continued | true rem | cont_ratio | STOP |
|---|---|---|---|---|---|
| 0.00 | 0 | 187.9 | 192.0 | 0.979 | 0.54 |
| 0.10 | 19 | 169.2 | 173.0 | 0.978 | 0.50 |
| 0.25 | 48 | 140.0 | 144.0 | 0.972 | 0.58 |
| 0.50 | 96 | 93.7 | 96.0 | 0.976 | 0.42 |

Flat and high — which reads as a clean "not blind", and **is not evidence of
anything**. The corpus median is 623 nodes, so at a 192 cap every held-out build
is truncated to exactly 192 and `true_remaining` is `192 - K` for all of them.
The model emits ~188 nodes regardless. `cont_ratio = (188-K)/(192-K) ≈ 0.98` is
arithmetic, and a model that ignored the prefix entirely would score identically.
Only **7 of 399** test builds are naturally shorter than 192, so the pool had no
length variance to detect anything with.

The diagnostic has to ask a question a fixed-output model cannot fake: does the
model continue *further* for builds that genuinely had further to go? That is
`length_corr`, the correlation between continuation length and true remaining
length, over builds of differing size. The script now restricts to untruncated
builds, reports the correlation, and refuses a verdict when the pool's length
std is degenerate.

Worth noting the shape of this mistake, because it is the same shape as the one
in §24.1 and as T21's: **a number that looks like a result but is determined by
the setup.** T21's `bits/op` rewarded collapse; the palette JSD returned 0.0 for
a fully out-of-vocabulary generation; teacher-forced accuracy was blind to
STOP-at-1.0. In each case the metric was fine and the *denominator* was wrong.
Ask what a null model scores before believing what the real one scores.

Re-running at `max_nodes=384`, where a usable fraction of builds fit under the
cap. That also attacks the size confound in T24c: 185 blocks against a real 904
made KID mostly a truncation measurement.

### §24.3 Blindness is a property of the training data here

Same model, same hyperparameters, opposite prefix-test verdicts:

| trained on | \|cont_ratio-1\| by prefix | verdict |
|---|---|---|
| all 1,863 builds (all truncated at the cap) | 0.237 → 0.272 → 0.418 | error grows → blind |
| the 258 builds that fit under the cap | 0.213 → 0.103 | error halves → uses the prefix |

The mechanism is STOP's meaning. At `max_nodes=384` only 13.8% of builds finish
naturally, so for the rest the last step is an arbitrary cut. Train on all of
them and STOP means "hit the cap": the model emits a fixed ~350-node budget and
ignores the prefix. Suppress STOP on truncated builds — correct in principle —
and it becomes 0.036% of pick targets and the model stops emitting it at all
(STOP rate 0.00). Train only on complete builds and STOP means "finished": STOP
returns at 0.62 and the prefix pulls the model toward the right length.

Cost: 258 builds instead of 1,863, and `place_lift` 139.8x → 51.8x. So the tension
is real and it is about data, not architecture — **at any cap the memory budget
allows, most builds cannot teach an ending.** The unblock is a cap above the
corpus median (~623), which needs the candidate list to be sparse rather than
`[B,N,N,6]`.

Two measurement errors on the way, both mine, both the same species as §24.2:

* The first prefix test had no length variance in the pool, so cont_ratio was
  arithmetic (see §24.2).
* The verdict rule read *any* decline in cont_ratio as blindness. But the
  complete-only model free-runs 21% too long and lands 10% short with half a real
  prefix — the prefix pulling it toward correct, scored as failure. The statistic
  has to be distance from correct, |ratio-1|, because T21's blindness was
  undershooting and this baseline overshoots.

Running tally of "the metric was fine, the denominator was wrong" in this project:
bits/op rewarding collapse (T21), palette JSD returning 0.0 for fully OOV output,
teacher-forced accuracy blind to STOP-at-1.0, cont_ratio with no variance, and now
a verdict rule with the wrong sign convention. It is the dominant failure mode
here, and it is always cheap to catch by asking what a null model scores.

## §24.4. Documenting the model forced a formulation audit (2026-08-17)

Wrote `docs/pick-and-place.md` — paradigm, representation, architecture, the
tensor contract, how edge faces are computed, training, and a formal statement of
the factorization at the end. Nav entry, MathJax via `pymdownx.arithmatex`, and
cross-links from `models.md` / `representations.md` / `architecture.md` /
`index.md`. Writing the input/output contract down is what surfaced the finding.

**The placer's cross-entropy target is not identifiable.** Up to six open faces
can point at the same empty cell; all of them produce the identical structure, and
the loss calls one correct. Mean multiplicity 1.99 over 56,223 real placements, so
a model that correctly treats them as equivalent caps at `place_acc` 0.568 — and
we measure 0.729. Being *above* the ceiling is the tell: the surplus is the model
learning BFS's tie-break rule. That is a property of `structure_to_growth`, not of
houses, and it means `place_acc` is not comparable across orderings today. The fix
is a `logsumexp` over each fibre of `φ(i,d) = c_i + e_d` before the CE — the policy
is unchanged, only the label becomes well-defined.

Same species as the running tally: the metric was fine, the *label* was wrong.
That now makes six.

Two smaller ones. `pick_acc` had no baseline (most-frequent-block scores 0.189 vs
our 0.607; unigram entropy 3.717 nats vs `pick_loss` 1.578 — the picker is fine,
we just could not say so). And the height feature is absolute `y/32`, which only
means the same thing at train and sample time because `_seed_index` picks the
lowest voxel; generation is seed-relative and can descend into negative `y` it
never trained on. Unmeasured.

**And a correction.** Re-ran the prefix test on the complete-only checkpoint at
n=24 instead of n=12: verdict moved from `USES THE PREFIX` to `NOT BLIND BUT NOT
READING LENGTH`, `length_corr` ≈ 0 for both training sets. §24.3's positive
reading was noise at n=12. The surviving claim is that `--complete-only` removes
the blindness *signature* (error stops growing with prefix length), not that the
model reads length. `length_corr` was the stable statistic and `cont_ratio` was
not; quote the former.

---

## §25. The realism metric was measuring the facade (2026-08-31)

The suite's own rule is that a metric must rank known damage before it may rank
anything. §23 applied that to the metrics but never asked the prior question:
**is there damage the whole tier cannot see?** There is, and it is the damage
this project most needs to detect.

`probes.solidify` fills every enclosed air cell with the build's own dominant
material. Silhouette untouched, palette untouched, a median 31% more blocks, and
every room in the corpus gone. MV-DINO-KID scores it **0.000 [−0.002, 0.004]**
against a held-out-real floor of **−0.001 [−0.004, 0.003]** — the intervals
overlap. The camera is outside the building, so the tier is reporting on the
facade, and this is exactly the failure mode T23d hand-caught in
`native_oriented` (enclosed air 0.005 vs a real 0.114) with a single scalar and
no distribution behind it.

The fix is a second realism tier rather than a replacement, because the blindness
runs both ways: `geom_kid` reads occupancy only, so `material_shuffle` and
`monochrome` move it by *exactly zero* — bit-identical, not approximately. Two
complementary halves, each gated on the rungs it can see. That meant the ladder
had to learn that metrics come in families (`RungSpec`): gating an occupancy
metric on material noise would fail it for doing what it was built to do, so its
noise rung moves blocks instead of retyping them, and material corruptions become
part of its *invariance* set — the blindness asserted rather than tolerated.

**Design choices worth remembering.**

- **Reuse the estimator, change only the features.** `geom_kid` is
  `distances.kid` itself, not a reimplementation, so a disagreement between the
  two realism numbers is a fact about what the features see and never about how
  the distance was computed.
- **Invariance by construction, verified bit-for-bit.** Local 2×2×2 patterns are
  canonicalised onto D4 orbits (256 → 55) and every scalar commutes with the
  group, so the ladder's G6 is satisfied by design. The tests assert equality,
  not closeness; an approximate check would pass a descriptor that had quietly
  started reading orientation.
- **Ablate, or the metric is a black box.** Dropping the block count moves
  `solidify` from 92.0 to 89.8 sd, which is what kills the obvious objection
  ("it's a size detector"). No single block carries the metric.
- **It runs on CPU.** That is not a convenience. The render tier is the half that
  can break silently, and during this session it did.

**Three defects, and the shape they share.** Two are the running tally again —
the metric was fine, the denominator was wrong — and this time in the ladder's
own gates:

1. `sd(real)` was estimated from a pool that *excluded* the probe set, so the
   "fresh real draw" G7 tests was not a member of its own null; and overlapping
   draws carry a finite-population factor. `geom_kid` failed G7 at 3.5 sd for
   noise that was mismeasured, while separating `solidify` at 67 sd.
2. G8 compared a difference of *means* against the spread of a *single draw* —
   larger by `sqrt(reps)`. Against the standard error of the difference,
   `legacy_cmmd`'s null mean halving with every doubling of n scores 14.8 se
   instead of passing.
3. The head-to-head p-value was a bootstrap sign-count, floored at `1/n_rep`.
   Holm over 55 pairs multiplies 0.005 to 0.275, so the first leaderboard put
   held-out real and a monochrome corpus in the same "not separated" group.

That is four denominators-and-floors in one session, which is a pattern rather
than a run of bad luck: **whenever a criterion is a ratio, write down what the
numerator and denominator are each estimating and check they are the same
population.** Three of the four were caught only because something downstream
refused to make sense — a metric that obviously worked failing a gate, a
leaderboard with one group.

**The renderer failed open.** `render_views` turned any exception into a blank
white frame. A PyOpenGL/Python mismatch made every render raise; every feature
collapsed to one vector and every arm scored KID −4e-13, *better than real*. And
because `load_or_build` verifies a render canary before trusting its cache, the
all-white canary would have mismatched and the 30 MB feature cache would have
been rebuilt from blank frames on the next run. The canary was designed to catch
a changed texture pack and instead became the mechanism that would have destroyed
the cache. Fail-open plus cache-invalidation-on-mismatch is a worse combination
than either alone; `render_views` now raises past a 2% failure rate.

**On aggregating.** BlockScore is the *max* over pillars, not a mean, and
novelty/diversity are disqualification gates rather than terms. Both follow from
T23c: if realism can be traded against novelty, a copier wins; if pillars can be
averaged, "looks right from outside" buys "is built wrong inside", which is the
one trade the render tier specifically invites. There are no weights, deliberately
— a weighted sum is where a benchmark's authors put their thumb. And the
aggregate is validated the way the metrics are: it must be *shown* to reject the
degenerate strategies, and on its first run it failed to, twice.

---

## §26. Baselines, dose-response, and a registered prediction (2026-08-31)

§25 showed the render tier is blind to one corruption. This is the follow-up
work a benchmark paper needs: does the suite *rank*, does it have a floor, and
does it agree with people.

**Dose-response is the test the ladder does not run.** The ladder asks whether a
metric separates damaged from real — a binary question — and a metric can pass
every gate by firing on an artifact while being flat over the range that actually
separates two submissions. Four graded axes fix that. Three of them come out
perfectly monotone for both metrics over 105–226 noise floors. The fourth is T25
restated as a curve: `mv_dino_kid` across the whole solidify range moves less
than one draw of its own noise. "Flat" is a much stronger statement than "failed
to resolve the endpoint", and it only exists because the axis was graded.

Designing the dose axis took three attempts, which is the lesson worth keeping:
**a graded probe is only useful if the grading is calibrated.** Random speckle
reads as noise rather than as lost interior. Filling whole rooms largest-first
put 71% of the damage in the first step. Filling inward from the walls,
nearest-first, gives an exactly linear interior axis. Two of the three would have
produced a plausible-looking table with no information in it.

**The quadratic finding was not something I went looking for.** The mixture axis
looked oddly compressed, and `MMD²(αP + (1−α)Q, P) = (1−α)²MMD²(Q,P)` explains it
exactly — a quadratic fits 14× better than a linear. The practical form is
alarming for a leaderboard: an arm that fails on a quarter of its builds scores
31% of the way to fully-failing. Since generators fail on *subsets* of prompts,
this is the common case, not the corner case. `sqrt(KID)` recovers the failure
rate to 0.022 absolute. Reported as a companion readout rather than swapped in,
because it inherits the ordering gates but not the ladder's noise-floor units.

**Baselines change what the numbers mean.** Before this there was no answer to
"is 0.188 good". Now: every learned arm beats every procedural baseline on
BlockScore, `pick_n_place` only barely clears a pile of random real patches, and
on `geom_kid` alone it loses to it. None of that was sayable last week.

**`patchwork` was built to attack `geom_kid` and mostly failed to.** Every patch
is verbatim real, so the local pattern statistics are near-perfect, and it does
buy a better geometry score than coherent-but-simple builds get — but only about
1.2× better than its appearance score, against the render tier's ~580× blindness
to `solidify`. Building the attack and reporting that it underperformed is worth
more than not building it.

**The registered prediction.** Both metrics rank `patchwork` significantly above
`gabled_house`. The renders say that is backwards. The likely mechanism is that
both are distribution distances and `gabled_house` is *too regular* to belong to
the real distribution — wall fraction 0.977 against a real 0.657, perfect
symmetry, a hollow shell — while `patchwork` inherits real texture wholesale. So
the metrics may be rewarding statistically house-like over recognisably a house.
Writing the prediction down *before* collecting human data is the difference
between a finding and a post-hoc story, and it costs nothing to do.

**On the human study design.** The instinct is to have people rate builds and
correlate with the metric. That does not work here: KID and `geom_kid` are
two-sample statistics with no per-build value. The instinct's opposite — show
people two *grids* and ask which set looks more real — matches the metric's level
but asks people to eyeball a distribution, which they are bad at, and yields one
datum per rater per pair. Pairwise over builds, pooled to the arm pair, gets both:
an easy judgement and the right level. Bradley-Terry rather than win rate because
each session sees a random subset, so coverage is uneven by construction and win
rate would encode who an arm happened to be drawn against.

Exclusion rule, Bradley-Terry, then agreement — fixed in that order, in advance,
so no step can be tuned once the results are visible. The pipeline was tested on
simulated raters including two answering at random; it dropped exactly those two
and recovered the injected ordering.

**On the dataset tree, and a flat list that was arithmetically wrong.** The lab
listed all 35 datasets flat and summed them into a "builds" stat. That stat read
~178,000 for ~96,000 builds, because `raw:all` *is* the five raw corpora it sat
beside and each `split:houses_32:*` is part of the corpus above it. Nesting fixes
this by construction rather than by a special case: a node's builds are counted
once, at the node, and containment is exactly the relation the tree draws.

The rule that kept the tree honest was to draw only edges recorded on disk —
`rawcorpora.SPECS` for the union, `splits.load_split` for the splits, directory
containment for the arms. The tempting fifth edge, raw corpus → curated corpus,
is *also* recorded (per-row `corpus` in each cache's manifest), but `houses_32`
draws from three sources at once, so as a tree edge it would force a choice
between lying about two of them and duplicating the node. Multi-parent
provenance is not a tree, and pretending otherwise is where a browsing tool
starts misinforming. It renders as an annotation beside the node instead, next
to the curation drop counts, since "2,661 kept" only means something against
"3,493 pooled".

**Branches as rules, not copies.** A saved subset stores `{parent, rule}` and
resolves to parent indices on read. The alternative — materialise a new `.npz` —
costs 50 MB per experiment and, worse, goes stale silently when the parent is
rebuilt. Two modes, and the distinction matters more than it looks: a *live*
rule ("everything I labelled good") re-runs on read and therefore grows as you
label, which is right while curating and wrong for an experiment. A *frozen*
rule stores the resolved list. An eval set that quietly grows between two runs
makes those runs incomparable, so the paper cites frozen ones.

Two bugs worth recording. Resolving a subset needs its parent's size; the
obvious way to get that is `catalog.get_dataset`, and the catalog resolves
subsets while building its own listing — so the first version recursed until the
stack died, but only once a subset existed on disk. It passed every isolated
test. The fix threads a lookup map down instead of calling back up. Second, the
tree double-counted subsets after the catalog also started returning them: two
nodes for one branch, and the totals stat was wrong again in the same way it had
just been fixed. Both are now pinned by tests.

Validation earns its place here for one specific failure: `{"corpsu": [...]}` as
a filter key would keep everything, look completely plausible, and never be
questioned. Unknown keys are an error.

---

## §27. Block ontology — measuring what a material *is* (2026-09-06)

Track E's model sees `blockstate.PALETTE`: 70 names, alphabetical, nothing
attached. `blockgen/ontology` is that list plus what each block *is*, and the
question it exists to answer is narrow: **does a grounded ontology beat the
model's own priors?**

### Design decisions, and what changed my mind

**Mine it, do not write it.** The first sketch of this was an authored JSON
(`{"primary_color": "#735135", "palette_style": ["rustic","medieval"], ...}`).
Authored attributes mostly restate what a frontier model already believes, so
they can only test prompt structure, not knowledge. The fields that can carry
*new* information are the ones the corpus knows and the model cannot:
per-block height profile, run-length anisotropy, face exposure, adjacency
affinity, category lift. Provenance is therefore a first-class field
(`mined` / `asset` / `authored` / `derived`), and `subset_by_source()` makes the
split an ablation instead of a claim.

**The container is domain-agnostic from day one.** `schema.py` knows about
`Part`/`AttributeSpec`/`Catalog` and nothing about blocks; `minecraft.py` is the
first backend. LEGO and schematics are the stated next domains and both have
catalogs of thousands of parts the model cannot be assumed to know — the transfer
claim has to be an implementation detail, not a fork.

**Affinity: raw co-occurrence → PMI → NPMI over building materials.** Raw
co-occurrence gives the same top-k for every block (the most common neighbour of
everything is the most common block). Plain PMI overcorrects into trivia —
"oak_planks pairs with Jukebox", from two placements. NPMI plus a support floor on
the *partner* gives species-consistent lists (`spruce_planks → spruce_stairs ·
spruce_slab · spruce_fence`) that survive being read out loud.

**The vocabulary is closed.** Neighbours outside the palette are dropped, not
renamed. A `pairs with` entry the DSL cannot resolve would teach the model a name
that fails to parse — knowledge converted into per-line errors.

**Numbers → words in one visible place.** `height_mean = 0.61` becomes `roof`
because of a cutoff in `minecraft.py`, next to its reasoning. Below `min_support`
(200 placements) the numbers are still reported and the words are withheld: a role
inferred from a dozen placements is noise wearing a label.

### A bug the lab caught immediately

Oak leaves measured **grey**. The vanilla texture ships greyscale and is tinted at
draw time (`renderer.textures.TINTS`), so a catalog reading raw pixels contradicts
every render in the repo. Fixed by applying the renderer's own multipliers. This
is exactly what the swatch column on the lab's Ontology page is for — the hex sits
next to the real texture tile, and a mismatch is only visible.

### The control is the experiment

An ontology can help for two reasons: it carries information, or it is 3k tokens
of plausible structure. Those look identical in a results table, so every catalog
ships `Catalog.shuffled(seed)` — the same table with attributes permuted onto the
wrong blocks, token-matched to the character. `ont_mined` beating `ont_none` is
not a result; `ont_mined` beating `ont_shuffled` is.

The n=12 pilot (T27) returned a clean null with `ont_mined` ≈ `ont_shuffled` on
every metric, *including* the palette-JSD improvement both showed over `ont_none`.
Without the control that improvement reads as the ontology working.

### Where it plugs in

| seam | how |
|---|---|
| prompt | `AgentConfig.ontology` ∈ `none/mined/shuffled/stats`; the table replaces `_palette_block`, so the arm pays for knowledge, not a second copy of the vocabulary |
| arms | `ont_none/ont_mined/ont_shuffled/ont_stats` in `experiments_agentic.ARMS`, `configs/experiments/agentic-ontology.yaml` |
| one build | `scripts/run_agentic.py --ontology mined` |
| live demo | `deploy/.../backends.py` reads `"ontology"` off a `models.json` entry; absent = unchanged |
| lab | `/ontology` page — provenance per column, raw statistic under every word, real texture swatches, and the exact prompt string with its token count |
| tests | `tests/test_ontology.py` (a 4×4×4 structure whose every statistic is derivable by hand), `tests/lab/test_ontology_api.py` |

Derived data, so it is gitignored like every other cache: rebuild with
`python -m blockgen.ontology` (~1 s).

### Open

* n ≥ 128/arm before any of this means anything.
* Implementation #2, and the one I would bet on: the ontology as a **validator**
  inside the repair loop (directional block placed without a facing, `not-full`
  block used as a wall, gravity block unsupported, adjacency the corpus never
  contains) — feedback into a loop that already works, at zero prompt cost.
* Every arm uses ~10 distinct blocks per build against the corpus's 21.5. That gap
  is larger than anything the ablation moved.

### §27.1 Renderer fidelity audit (2026-09-06)

Prompted by an observation while looking at ontology swatches: redstone reads as
wool in our renders. Two separate defects, measured over `houses_32`
(2.78M placements) — filed as `todo.md` R1/R2/R3.

| defect | share of placements | what it is |
|---|--:|---|
| **wrong shape** | **27.5%** (765k, 162 types) | `textured.build_mesh` emits six axis-aligned faces per voxel and nothing else, so stairs, slabs, fences, panes, doors, torches, carpets and plants are drawn as full cubes wearing the right texture. Top: oak stairs 2.8%, oak slab 2.4%, spruce stairs 2.3%, oak fence 1.4%, glass pane 1.4% |
| **flat colour** | 1.33% (37k, 102 types) | no `FACE_TEXTURES` entry, so `_solid_texture` paints `_color_for` — a uniform block that looks like wool. Redstone wire/block/torch, rails, levers, buttons, pistons, beds, signs, glazed terracotta, crops |

The second is small *here only because houses are not redstone builds* — the full
corpus has 1,287 "Redstone Device Map" structures (T2) where it is the dominant
material. It is also the cheap one: a lookup table, not geometry.

The first is the one that should worry us. Every render the eval scores, the
human study shows, and the visual-critique loop feeds back is drawing a quarter
of its blocks as the wrong solid — and it does so *systematically*, which is the
same shape of problem as T25 (a realism metric that could not see through the
facade). It does not invalidate `geom_kid` (that tier reads geometry, not
pixels), but any claim resting on rendered images inherits it.

---

## §28. Probing "attention as an adjacency matrix" (2026-09-06)

The proposal under test was to take a pretrained LLM, run it over a serialized build,
and read its self-attention matrix as the connection graph — structure for free, no
training. `scripts/probe_llm_attention.py` measures it; T28 has the numbers. What
follows is the design, because most of the effort went into making the question
answerable rather than into answering it.

### The direction of the arrow

Worth stating first, because it decided what to build. The graph-transformer line this
idea gestures at (Graphormer; both systems named GraphGPT) *injects* known structure
**into** attention as a bias or mask. Nobody reads an adjacency matrix out. The nearest
real precedent for extraction is the "attention heads capture syntactic dependencies"
probing work, whose own conclusion was that heads correlate noisily and a supervised
probe on hidden states does better — and syntax is at least in the pretraining
distribution, which voxel adjacency is not. So the probe was built to answer the
extraction question, and T29's training run answers the injection one.

### Making it a pointer question, not a heatmap

The useful framing is that attention used as an output distribution over existing nodes
*is* a pointer network. So the probe asks precisely what the placer in
`blockgen/models/pick_n_place.py` asks: for line `i`, rank the previous lines `j < i` by
how likely they are to be the piece it attaches to. That makes AUC the natural score,
makes the candidate set exactly the causal prefix, and makes the result directly
comparable to a trained pointer head rather than to a picture.

### Three controls, and only one of them was obvious

**Recency was the control I planned for.** Lines are emitted in `(y,z,x)` raster order,
so `-(i-j)` alone scores 0.831. Any head has to beat that before it has said anything.

**Distance stratification was the control that made recency interpretable.** Beating
0.831 is not the same as knowing something position does not. Computing AUC *within*
sequence-distance bins and pooling collapses recency to 0.557 — by construction, which
is the point: in that view the floor is chance, and the best head's 0.800 is a real
statement about what it knows beyond order.

**The control I did not plan for is the one that decided the result.** The serialization
prints `<piece_name> <x> <y> <z>`. The coordinates are *on the page*. So before asking
what 1.5B parameters extract, ask what subtraction extracts: `-||a_i - a_j||_1` over the
two anchors the lines literally spell out scores **0.980 / 0.969**, against the best
head's 0.884 / 0.801 and a trained hidden-state probe's 0.914 / 0.857. I added this
baseline late, while writing up a result I was about to report as "attention carries
real adjacency signal". It does — and it is a lossy re-derivation of its own input.
Without this row the table reads as a mild positive.

The corollary is that `coord_match` (how many of x/y/z are the identical token, AUC
0.881) lands inside the best head's confidence interval, which is what a copying head
matching coordinate strings would look like, and not what a geometry head would.

### Choices inside the pooling that could have faked a result

**Prompt tokens are dropped and rows renormalized over completion lines.** The BOS
attention sink absorbs a large share of every row's mass; leaving it in makes rows
incomparable across builds of different lengths, and the pooled AUC would then partly
measure how long each build is.

**Query tokens are mean-pooled, key tokens summed.** A line is several tokens; summing
on the key side is the total mass placed on that line (what "points at it" means), and
averaging on the query side keeps lines with more tokens from counting more.

**Head selection is on a train half of the builds, reported on the test half.** Picking
the best of 336 heads on the same data it is scored on is a guaranteed overstatement.
Here it happened not to matter — L14 H0 wins on both halves in both arms, and the
oracle-on-test number is quoted alongside — but that is a fact about this run, not a
reason to skip the split.

**Truncation keeps whole-line prefixes.** 90 of the 120 held-out builds are ones the
finetune never saw *because they blew the 2048-token cap*. Cutting them at a line
boundary leaves a valid partial structure in raster order, so the adjacency ground truth
over the retained pieces stays exactly correct and the held-out set triples.

**Ties are handled with average ranks.** The recency baseline is massively tied
(every pair at the same `i-j`), and ordinal ranks would have quietly inflated or
deflated it depending on sort order.

### What the layer sweep added

The probe was run at layer 14 and at the final layer. The adjacency information peaks
mid-stack (0.914) and is down to 0.807 by layer 28 — the layer the LM head reads to emit
the next token. Even in the best case for the proposal, the signal is weakest exactly
where generation would consume it.

### What it cost

One script, two GPU-hours, one afternoon. The idea was worth a day of measurement and
would not have been worth a month of building, which is the entire argument for
probing before implementing.

## §29. Run identity, provenance and example builds (`bench/2`, 2026-09-06)

Eleven scorecards on disk, every one of them in a directory called
`run_<stamp>_bench`, and the only way to tell them apart is to open them. On the
largest of them, thirteen of sixteen rows say nothing about themselves except
`source: "in-memory"`. That is the whole motivation: the benchmark measured well
and recorded almost nothing about *what* it had measured, so its own output was
illegible three weeks later. `bench/2` fills the fields that already existed and
adds three that did not.

**What a run now records.** `--name` (free text, also names the directory through
a word-boundary slug), `--note`, `started_at`/`finished_at`, `git_branch` and
`git_dirty` beside the sha, `host`, `device` as *requested*, `argv` as data and a
`rerun` string built from it by `shlex.join`. `run.cmd` stays and is marked
legacy: its `argv[0]` is an absolute path to `__main__.py`, so it was never
copy-pasteable, and the claim that `rerun` is gets tested by re-parsing it
through the parser that wrote it — which is the only reason `build_parser()` was
factored out of `main()`. Per arm: `kind`, `origin`, `source_run_id`,
`structures_sha`, and a `provenance` block that is a manifest read for npz arms
and a stated recipe for in-process ones. The context block is now literally
`BenchContext.to_json()` rather than a hand-built dict beside it, which is how
four values the arms *were* scored with (`palette_level`, `dup_threshold`,
`n_ref`, `sizes`) started being reported at all.

**Why identity stayed in the scorecard.** The tempting shape was a sidecar
`run.json` plus a `blockgen/utils/runcard.py` writer, shared by every run in the
repo. I did not build it, and the reason is that this increment has exactly one
writer and one reader: a second memoized read path, a validator, a synthesizer
and a CLI would all be bought before anything needed them, and the scorecard is
already the file every consumer opens. The argument *for* the sidecar is real and
is about a different thing — a **training** run needs identity and will never
emit a scorecard — so the vocabulary a later lift-out would use (`kind`,
`artifacts`, `origin`, `status`, `created_at`) is reserved in `docs/benchmark.md`
so that move is a rename rather than a redesign. The trigger to build it is the
first trainer that wants a leaderboard row.

**The example builds, and why they are not a PNG.** Every arm leaves `k` builds
behind in one `examples_<max_dim>.npz` + manifest in the run directory. A contact
sheet would have been simpler and is unusable: no lab route serves a file out of
a run directory (`_static` resolves only under `tools/lab/static/`), and the only
image routes are keyed by `build_id`. An npz + manifest pair, on the other hand,
is *already* a dataset — the lab's existing arm glob finds it, the existing
content-addressed render cache draws it, Curate opens it — so the eval writes
identities and renders nothing, which also keeps the renderer out of the fast
tier. Measured cost: 52 KB for 64 rows on the smoke run, ~200 KB for a real
sixteen-arm run. Rows are `dataclasses.replace` copies with metadata set
explicitly, because `real@single_mode` is one `Structure` repeated *n* times and
that object is simultaneously in `real_test` and in the shared `test` split the
lab serves; an in-place caption would have corrupted three datasets at once for
the life of the process.

**A schema version nothing branches on.** `bench/1` is four shapes on disk, not
one, so a migration keyed on the version string would be wrong about eight of the
eleven cards before it ran. `SCHEMA_VERSION` is advisory: it drives one amber
banner for a card from the future, and every other decision is made on the
presence of a section or a key, in `tools/lab/cards.migrate` alone. Nothing on
disk is rewritten, no directory is renamed, and a legacy card is read with its
gaps *named* on the page rather than defaulted — a fabricated value is
indistinguishable on screen from a measured one.

**Seams left open, deliberately.** No standardization of train/render/view (only
the eval verb is standardized; `write_run_examples` is track-agnostic, so any
writer can drop an `examples_*.npz` into its own run dir and get the strip for
free). No shipped reference run under `benchmark/reference/` and no `promote.py`
— a committed scorecard has to be re-promoted on every metric change, and the CI
value was bought for ~5% of the cost by `tests/fixtures/scorecards/`. No
submission format. No guarded run-dir file route (that is where a later hero
image or training curve would go). No run-to-run Δ column: BlockScore is quoted
in each run's own measured noise floor, so the comparability check has to be
built before the column is. No stratified example policy — the page says "a
sample, not a best-of" rather than pretending eight uniform draws catch a
15%-failure mode. And no retention policy for the one dataset node each bench run
now adds; it wants doing in one pass with `prerender --kinds` and the frozen
contract's kind list.

---

## §30. Injecting geometry into attention, and a parameterization that ate the experiment (2026-09-07)

T28 killed reading adjacency *out* of a pretrained LLM's attention. §29 is the other
direction — the one Graphormer and the graph-transformer line actually take — and it
returned a null. The null is fine. What is worth recording is that the null was caused by
a choice I made for a good reason and did not check the consequences of.

### The design, and the constraint that shaped it

Everything about Track D v2 is held fixed and one term is added: a learned scalar on the
attention scores, indexed by the relative 3D offset between the pieces two token
positions refer to. Zero-initialized so the arms start numerically identical; delivered
through a prepared `{"full_attention": ...}` mask, which transformers 4.57 uses verbatim
when `attention_mask` is a dict.

The constraint is causality, and it is the whole story. A line reads
`<piece_name> <x> <y> <z>`, so the anchor of the line being *written* does not exist until
its coordinates have been emitted. The query side therefore cannot use it. I made the
reference the anchor of the last **completed** line, which is always available at
generation time, and gave keys inside the query's own line a separate scalar so the
current anchor cannot leak backward through them. Both properties were verified
numerically before training, along with the dict-mask path matching HF's default causal
mask bit-for-bit and gradients reaching the table.

That reasoning is correct and the implementation does what it says. It is also what
broke the experiment.

### Reading the table back is what caught it

**364 of 729 offset buckets were never updated — exactly zero.** The reference is the
previous line's anchor, lines come out in `(y,z,x)` raster order, so `dy ≥ 0` always and
the lower axes are sign-constrained whenever the higher ones tie. Three of the six
6-adjacent offsets, `-x`/`-y`/`-z`, are unreachable *by construction*. The bias could not
express "my neighbour below me" if it wanted to.

What it did learn: `+x` at +0.146 against a −0.168 mean over the reachable buckets. That
is "attend one step back along the raster run" — recency in different clothes, and T28
had already measured recency as the weak predictor. So the arm spent 20 epochs learning a
worse version of something the model already had.

I would not have found this from the loss curves. −0.0021 best val loss and +0.09 parse
rate look like a small, plausible, publishable-adjacent win, and the sample sheet looks
mildly better if you want it to. Plotting the parameter table was an afterthought added
because the figure script needed a second panel.

### Why the arms still look nearly identical

Worth being explicit, since a shared bias sounds weak and is: one scalar per offset added
at every layer and head. A null here bounds a shared-scalar bias, not per-head structure.
Between that and the dead half of the table, this run does not test the hypothesis it was
built to test. It tests an under-parameterized, half-masked version of it.

### The three gates from §20 applied here, and one that should be added

The §20 gates all fired usefully. **Open the renders**: 3/8 vs 4/8 degenerate samples,
against 0/30 for real builds, median 161 blocks against a real 699 — the arms are both
far from the data and indistinguishable from each other at n=8. **Quote a floor**: the
control reproduces Track D v2's 0.41 final val loss, so the comparison is a comparison.
**Treat loss as adversarial**: the loss moved the "right" way while nothing else did.

The gate this run adds: **read the learned parameters of any structural prior you
introduce, before believing its loss.** A structural inductive bias has an interpretable
parameterization almost by definition; if it cannot be read out, that is a reason to
distrust the arm, and if it can, it costs one plot.

### The next run, specified — and one claim in it was wrong

Emit coordinates before the piece name (`"<x> <y> <z> <piece_name>"`) so the *current*
anchor is causally available at the moment the piece type is predicted, putting the
geometry where the load-bearing prediction is. It needs its own matched control because
the serialization changes. Per-layer or per-head tables are the capacity axis after that,
and need per-layer injection rather than the shared mask.

**What I got wrong here (corrected 2026-09-16).** I also wrote that the reordering "makes
all 729 buckets reachable". It does not, and the arithmetic was available the day I wrote
it. The dead half is a property of the **emission order**, not of the reference point.
Under `(y,z,x)` raster order every earlier piece has `dy > 0`, or `dy = 0 ∧ dz > 0`, or
`dy = dz = 0 ∧ dx > 0`; clamped to ±4 that is exactly **364** buckets, plus the one where
the key *is* the reference piece — 365, which is precisely what the trained table updated.
Swapping the reference from `a_{i-1}` to `a_i` leaves `dy ≥ 0` untouched, because both are
anchors of pieces that precede or equal the query in the same monotonic order.

I had the 365 in front of me and reached for the explanation I had just been thinking
about (the reference point) instead of the one that actually predicts the number. The
check that caught it is four lines: enumerate the offsets raster order can produce and
count them. It reproduces 365 exactly.

So the real lever for coverage is a **non-monotonic emission order** — BFS/growth, which
this repo already has and already likes (`ar_bfs`, T11). Coords-first is still worth doing
for its own reason (the frame of the geometry), but it is not the reachability fix, and a
run justified on reachability grounds would have come back null for the same reason this
one did.

## §31. Two pages, because "which arm won" and "which model is best" are different questions (2026-09-07)

The leaderboard page ranked *arms inside one card*, which is what the bench
measures, and was named after a question the bench never answers. Split it:
`/runs` is one experiment in full (controls, calibration, provenance, examples),
`/leaderboard` is cross-run, one row per model, over every run of one protocol.

The split forced the real work. BlockScore's unit is a spread of the `real_test`
control **scored in the same run**, so cross-run ranking needs a comparability
rule. `blockgen/eval/bench/protocol.py` pins one: corpus, split, tier, seed,
`min_n` per arm, `min_ref`, and -- the one that is easy to miss -- `min_n` on the
calibration arm itself.

**That last check earned its place empirically.** Scoring `native_oriented`
against a 32-build `real_test` gave 87.14 -> 19.05. Same model, same corpus, same
split, same tier; a 4.5x "improvement" produced entirely by shrinking the ruler.
Re-run properly at n=128 it reproduced **87.14427315448478** against the older
card's **87.14427315440719** -- ten significant figures. So the protocol is doing
exactly what it claims, and a board without the control check would have
published the 19.05.

That float agreement then caused its own bug: `_better` compared scores with `<`,
so 8e-11 of dust chose which run represented the model, and it chose the older
card, which predates example builds -- the row silently lost its renders. Ties
within 1e-9 now go to the newer run.

Seven of sixteen runs match `houses32-v1`. **Zero submissions qualify**: every
model on disk was sampled below `min_n` (n=64, 16, 12). Rather than lower the bar
to fill the board, the page has a second table -- Provisional -- ordered but
explicitly not ranked, each row carrying the requirement it missed. Getting a
real ranked row needs re-sampling a model at n>=128, which is a sampling run, not
an eval change.

Also added `Slate`: a protocol may pin the prompts every arm renders so the
board's column *i* is one comparison. `houses32-v1` is unconditional and pins
none, so `Slate.aligned` is False and the strips are labelled as slots, not
matched items -- you cannot ask two unconditional models for the same build, and
a grid implying otherwise is worse than no grid.

---

## §32. Putting the LLM track on the paved path, and the hole it fell through (2026-09-07)

`docs/workflow.md` defines four verbs — train, sample, evaluate, view. The LLM arms
stopped at "train": they wrote a contact sheet and nothing the benchmark could read.
Wiring them up took one new script and turned a null into a mild negative (T29a). It
also found a gap in the leaderboard's guarantee, which is the part worth recording.

### The sampler, and the check that earned its place

`scripts/sample_geobias.py` is workflow step 2: run dir in, structure cache plus
`_manifest.json` with a `report` block out. Two things in it were not obvious.

**KV-cached decoding, and how to verify it.** The training-time sampler re-runs the whole
prefix each step (~45 s/build), which is fine for eight builds and impossible for 128.
The cached path builds the bias row incrementally, so there are now two implementations
of one function and the obvious guard is to decode greedily both ways and require
identical tokens.

That guard failed, and the failure was informative twice over. First it caught a real
bug: the cached path used the *advanced* line index as the query's line, but the query is
the token just emitted, which still belongs to the previous line when it carried the
newline. Second, after fixing that, it *still* failed — and this time the check was
wrong, not the code. Batched prefill and single-token decode run different bf16 kernels,
so their logits differ by ~0.6 median **with the bias switched off entirely**. The
control had passed exact-match only because its top-1/top-2 gap never closed; the bias
arm hit an exact tie between two coordinate digits and flipped a token.

The fix was not a hand-picked tolerance. `--verify` now asserts (1) the decode-time bias
row equals the training-time row exactly, at every step — the thing that can actually be
wrong — and (2) the cached path's logit drift is no worse than the same model's drift
with no bias at all, measured in the same process. The tolerance is the model's own
numerical floor rather than a number I chose, which is the same move as quoting BlockScore
in real-sample spreads.

**`--n` has to count kept builds.** It counted attempts, generations that parse to zero
blocks were dropped, and the control landed at 119 — eleven short of the protocol's 128,
which would have unranked it for a reason having nothing to do with the model. Now the
loop runs until it has `n`, and `--append` resumes the prompt cursor from the previous
pass's recorded attempt count, because a top-up that restarts the cursor re-draws the
same captions under the same seeds and silently duplicates the builds it was meant to
extend.

### The hole: the protocol checked the evaluation, not the training

Both arms scored on `houses_32`, the pinned split, both tiers, n=128 — and were **ranked**.
They should never have been: Track D's 90/10 shuffle predates the pinned split, and 53 of
its 267 training builds are in the canonical *test* set. I measured that before sampling
and assumed the board would catch it. It did not, because `match_run` validates the
*evaluation* — corpus, split, tier, n — and nothing looked at where the model trained.

The evidence was one field away the whole time: the arm's own `report.train_split` reads
`"trackd"`, and the page already renders that provenance block. So `match_arm` now checks
it, and an arm has to *say* it trained on the protocol's split to be ranked.

Two decisions inside that. **Silence is not consent** — an arm with a report but no
recorded `train_split` is unranked rather than assumed clean, because a contaminated arm
is exactly the one that will not know it is. And **controls are exempt**: `real_test` and
the procedural baselines carry no report, and requiring one would disqualify the ruler.
The check immediately flagged an existing arm (`native_oriented`) for not recording a
split; that arm was already unranked for `n=64`, so no ranked row was lost.

This is the same shape as T25 and T26g: a measurement that looked fine because the thing
it failed to check was invisible on the page. The protocol's own docstring argues a
protocol should be "a fact, checkable after the run from the card alone" — training
provenance is such a fact, and it was simply not among the ones being checked.

### What changed for future runs

`train_llm_geobias.py --split canonical` is now the default and draws train/val from the
pinned split, so new runs are born eligible. `--split trackd` reproduces the T29 pair and
is documented as permanently unrankable. The T29 arms keep their `/runs` page, where they
belong — the control-vs-bias comparison is internally matched and valid on its own terms;
what it cannot do is sit on a board next to models that did not train on the test set.

### Still open

The oversize-build rate (26/128 and 39/128 over 32, largest 253) is recorded but not
gated. It is real model output and filtering it would flatter the arm, but a build with a
253-voxel span is not a house by any reading, and `max_dim_w1` is the only place the
scorecard notices. Whether that belongs in the coherence pillar rather than in dataset
stats is a live question.

---

## §33. The best number in the project was scored on its own training data (2026-09-10)

`native_oriented` has been the headline learned arm since T23d — BlockScore 87.14, best
of everything that is not real data. Scoring it at n=128 against the LLM arms confirmed
it wins by a wide margin, and in the process established that the margin is not currently
interpretable.

### How it surfaced

Not from a suspicion about the model. The provenance gate added in §32 requires an arm to
declare the split it trained on, so wiring `native_oriented` through
`scripts/sample_to_npz.py` meant answering the question "what split was this?" for the
first time. Its config says `val_frac: 0.15`, its trainer calls
`_split_houses(seed, val_frac)` — a fresh `default_rng(0).permutation` over the whole
corpus — and the pinned split did not exist until 2026-08-03, three weeks after the
checkpoint. Reproducing that permutation and intersecting it with the canonical split
gives the number: **346 of 399 canonical test builds, 87%.**

That took four minutes and would have been worth doing in July. The gate did not detect
the contamination; it just made "which split?" a question that had to be answered out
loud, and the answer was immediate.

### Why this is not a retraction

The instinct is to strike 87.14 from the record. That would be wrong in the other
direction, because the memorization gates *do not fire*: duplicate rate 0.000, and
`dino_nn_percentile` 0.495 against a real floor of 0.539 — slightly closer to training
data than real held-out builds, which is directionally what contamination predicts, but
a rounding error next to `train_verbatim`'s +10.8 spreads. The model has a final train
loss of 0.311 and was still falling; it is underfit, and an underfit model extracts less
from a memorized example than a fitted one does.

So the defensible claim is narrow and worth stating precisely: **the margin cannot be
separated from the head start.** Not "the number is fake" and not "the number is fine".
The eval suite is built to tell those apart and here it cannot, because the arm was
trained before the split it is being scored against existed.

The same applies to `native_bpe`, `canon16_flat` and `text_cond`. All four now carry
`train_split` in `deploy/inference/models.json`, so the sampler records it without the
caller having to remember, and the board says *why* an arm is unranked rather than only
that it is.

### A comparison that stayed open

`native_bpe` and `native_oriented` are two runs, and I had assumed the oriented one simply
superseded the other. On the one metric where they are directly compared, `native_bpe`
wins: ungated validity 0.188 vs 0.062. It has never been benchmarked. "Oriented is
better" is an assumption this project has been carrying, not a result — and since both
are equally contaminated, settling it needs two clean retrains, not a rescore.

### What broke, and what it says about running on a shared box

`real_test` OOMed twice: an unrelated job on the same GPU held 12.6 GB. Every other arm
scored fine on the remaining 1.3 GB, so the card has complete realism columns and **no
BlockScore at all** — the composite needs the control arm because that arm's spread *is*
the unit. The raw KID columns are unaffected and comparable across runs, which is how the
result is reported here, and `llm_control`'s `geom_kid` reproducing to three decimals
(15.006 in both runs) is the check that says so.

Worth noting for the paper's repro section: a benchmark whose ranking silently disappears
when one control fails is behaving correctly — better a missing composite than one
calibrated against whichever arms happened to survive.

### Forward

`blockgen/experiments_native.py --split canonical` is the new default and draws from the
pinned split; `--split legacy` reproduces the July runs and logs a warning that they can
never be ranked. `_split_houses` keeps its behaviour and gains a docstring naming the 87%,
because the July numbers still have to reproduce.


## §34. The retrain that removed our own result, and a paper that is about the ruler (2026-09-17)

Two things happened today, and they are the same thing.

### The retrain

§33 priced a clean retrain at ~1.9 h and said the honest statement was that
`native_oriented`'s margin "cannot currently be separated from its head start."
It can now, and the answer is the uncomfortable one: **the margin was the head
start.** `canon_oriented` on the canonical split ties with both LLM arms in Holm
group e (results.md T32). The claim this project has carried since T23d — that
the native token track beats the prompted-LLM baselines — is gone.

Worth being precise about what is *not* gone. The native track is still
geometry-limited where the LLM arms are appearance-limited; they fail in
different directions and a single scalar would have hidden that. And the clean
model is still well clear of every procedural baseline. What died is the
head-to-head ranking claim, not the track.

The second finding is the one I would not have predicted: `canon_unoriented` is
**DQ'd for memorization at duplicate rate 0.000.** A duplicate counter passes it;
`dino_nn_percentile` (0.369 against a 0.540 real floor) does not. The arm never
copies a build outright, it just lives systematically nearer its training set
than real held-out data does. That is exactly the failure the retrieval
percentile was adopted for, and it is the first time it has fired on something
other than the `train_verbatim` control.

### The decimation controls are gone

Policy, set today: `real@canon16` / `real@canon8` are no longer leaderboard
controls (`fast.control_arms`), the C3/C4 composite checks over them are
removed, and every "worse than N%-decimated real data" claim is struck from
results.md and notes.md. They survive as **ladder rungs**, where the only
question asked of them is whether a metric orders a known corruption correctly.

The reasoning is that decimation changes *resolution*, not *quality* — it makes
a smaller building, not a worse one — so ranking a generator against it was
never a meaningful comparison, and the sentence it kept producing ("our model is
worse than 86%-deleted real data") sounds damning while meaning nothing. Tests
still 396/396; the guarded checks degrade to 6 from 9 rather than erroring.

### The paper

Drafted in `deliverables/paper/iclr/` — builds clean, 16 pp. The framing that
survived contact with the evidence is **not** the generator paper. It is:

> render-and-embed metrics read a surface, a building is not its surface, here
> is the probe that proves it, here is the tier that sees it, here is what an
> aggregate over two complementary blindnesses has to look like, and here is the
> ranking of our own flagship model that this suite deleted.

T32 is the closing section rather than a footnote, which is the right use of it:
a benchmark's credibility comes from having cost its authors something.

**The one blocking item is human data** (§7 is design-only, marked `\UNMEASURED`;
`\draftfalse` makes any leftover TODO a build error). Plan in the paper README.
The highest-value trial in the study is not in the pre-registration:
`real@solidify` vs `real_test` put to humans. If raters separate what the render
metric cannot, the central claim stops being a statistic about a statistic.
Add it explicitly and say it was added.

### §34.1 Prior-art sweep, and two things in the draft that were wrong (2026-09-18)

Ran a full novelty check before committing to the framing. Both central claims
survive; two errors did not.

**The 31% sentence was wrong.** "An arm failing on a quarter of its prompts reads
only 31% as bad as one failing on all of them" conflates 0.043/0.546 = **7.9%**
with measured-vs-linear = 0.31. Different quantities, and the wrong one was in
the abstract. This is the sort of thing a referee recomputes in the first five
minutes.

**The quadratic identity is Bińkowski's, not ours** — Appendix B.2 of the KID
paper we already cite. Worse, Appendix D.3 of the same paper proves FID is *not*
polynomial in the mixture weight, so generalising the claim to FID would have
been false. Demoted to "we make explicit", attributed, scoped to squared IPMs.
The √-reporting *recommendation* still looks novel; that is now the claim.

**Dream-Cubed states our hypothesis in prose, and it was in our own bib.** Their
limitations section: render-FID "struggles to assess internal structure (e.g.,
whether a building has a coherent interior)"; their human study has raters
agreeing with that FID only 54.3% of the time. We had it in §2 as "reports a
render-based FID." The fix is to lead with it, not bury it: the field's largest
study names the limitation and ranks with the metric anyway, and we make it
falsifiable. The paper is stronger this way, and the 54.3% is free corroboration.

**Next experiment, and it is cheap:** a `Hollow-Shell` probe (erode to the surface
layer). Our probe fills interiors; the field's actual failure is hollow shells.
Same machinery, and showing both directions invisible kills the "artifact of one
probe" objection.

The honest verdict to carry forward: **the interior claim is novel as a measured
result, not as an observation.** Say that in those words rather than implying
nobody had noticed.

### §34.2 The human study's positive control does not work, and why (2026-09-18)

Tried to build the condition that would let raters *see* an interior, so the
`Real` vs `Fill-Interior` pair could show humans succeeding where the metric
fails. Measured four quadrant cuts and twelve half-space cuts (both horizontal
axes, three depths, both directions) on the six held-out builds with the most
interior volume.

Best result: **half-space, z axis, upper half removed — 4.25 mean absolute pixel
difference against an exterior baseline of 1.42.** Three times the baseline, and
still only 4.25/255. Too faint to carry a positive control: at chance we could
not tell "humans cannot see interiors" from "the cutaway was too subtle", which
makes the condition uninterpretable in both directions. Removed from the study.

The reason is worth keeping: cutting a *render* still shows wall faces of the
same materials. What reads instantly is the **2D cross-section** from Figure 1 —
orange rooms against grey solid — because it is a schematic, not a picture. But
that changes the rater's task to judging floorplans, which needs its own pilot.

So the study ships exterior-only, and the interior pair is reported as the
**informational-blindness** result: if humans are also at chance on the views the
metric is given, the interior is absent from the input and no encoder could
recover it. That is a better claim than the one originally pre-registered, and it
is the one the instrument can actually support.

`render_cutaway` and `--cutaway` are kept in `scripts/human_study_export.py` for
whoever builds the cross-section version; the measured numbers are here so nobody
re-derives them.

---

### §34.3 The cutaway works, and the blindness is the view protocol (2026-09-18)

The reviewer objection §34 invites is "that is a DINOv2-at-4-views artifact, not a
property of render-and-embed metrics". Ablated the render pipeline and re-measured
`solidify` under each condition — same probe, same unbiased KID, same 256 held-out
builds against 512 reference builds, each condition scored against **its own**
noise floor (`ladder.noise_floor`, 48 fresh real draws, finite-population
corrected). Run: `outputs/run_20260918_205019_view_protocol_ablation/`,
driver `scripts/ablate_view_protocol.py`.

Separation of `solidify` from held-out real, in that condition's noise floors:

| exterior conditions | sd | | cutaway conditions | sd |
|---|---|---|---|---|
| DINOv2-B, 4 ortho (the paper's) | +1.1 | | facing-quadrant cut | **+9.1** |
| same, best single view (no pooling) | +1.4 | | roof off (upper half deleted) | **+20.3** |
| CLIP ViT-B/16 | +0.7 | | four horizontal 1-voxel sections | **+56.5** |
| CLIP ViT-B/32 | +1.7 | | four vertical 1-voxel sections | **+14.1** |
| ResNet-50 (ImageNet) | +1.0 | | | |
| Inception-v3 pool3 | +0.7 | | `geom_kid` (voxel descriptor) | **+120.8** |
| DINOv2-B, 8 ortho | +1.8 | | | |
| DINOv2-B, 16 ortho | +1.6 | | | |
| DINOv2-B, 4 perspective | +0.9 | | | |

Nothing on the left clears 2 sd; everything on the right clears 9. **So the claim
is about the view protocol, not about DINOv2 and not about the view budget.** Three
pretraining regimes (self-supervised ViT, language-supervised ViT, label-supervised
CNN), four times the cameras (16 views: 8 azimuths at elev 30, plus rings at 75 and
-45, so roof and underside are both covered), perspective instead of ortho, and
per-view scoring instead of mean pooling all land inside the noise. The information
is absent from the input, not mishandled by the features — and it comes back the
moment the views carry it. That is the strong form of the result, and it is now
falsifiable rather than asserted: the positive control exists and fires.

**§34.2's negative result was about the wrong measurement, not about cutaways.**
That entry rejected the cutaway because the best half-space cut moved only 4.25/255
mean absolute pixel difference. But mean pixel difference is not the metric — KID
on DINO features is, and the *same* quadrant cut §34.2 called too faint separates
`solidify` at 9.1 noise floors here. What §34.2 got right is that a cut render still
shows walls of the same material: the cross-section slab it recommended (a
one-voxel slice viewed face-on, air reading as background) is the strongest
condition by a factor of four, at +56.5.

Mechanics, in `blockgen/eval/bench/features.py`: `ViewConfig.cuts` takes one cut
spec per view (`half:<axis>:<lo|hi>`, `slab:<axis>:<pct>:<thick>`,
`quad:<lo|hi>:<lo|hi>`), applied before rendering, and the cut is applied to the
*reference too* — a cutaway is a view protocol, not a corruption, so every arm sees
the same one. Default `ViewConfig` behaviour and its cache keys are untouched.
`cut_empty_rate` reports how often a cut would empty a build and fall back to the
uncut one (0.0–0.5% here); a blank frame scores as a perfect match, so that number
has to be visible. `views.png` in the run dir is the eyeball check: exterior real
and exterior solidified are indistinguishable, the cross-sections are not.

One caveat worth keeping: this tests *whether* the interior is recoverable, not
whether a cutaway protocol is a good metric. Cross-section views throw away the
facade, so a metric built on them would be blind in the opposite direction. The
honest next question is whether an exterior+section protocol is dominated by
`geom_kid`, which sees the same damage at +120.8 sd for the cost of no renders at
all.

### §34.4 Correction to §34.2 — I measured the wrong thing (2026-09-18)

§34.2 concluded the cutaway positive control "does not work", on the grounds
that the best cut moved only **4.25/255 mean absolute pixel difference** against
a 1.42 baseline. That conclusion is wrong, and the reason is worth keeping.

**The metric is not pixels. It is KID over DINOv2 features.** Mean absolute
pixel difference is a proxy I chose, and it is the wrong one: a small,
spatially-concentrated change in the right region moves the feature distribution
far more than its pixel average suggests. Measured properly
(`outputs/run_20260918_205019_view_protocol_ablation/`), the *same* quadrant cut
I rejected separates `solidify` at **+9.1 noise floors**, and the cross-section
slab that entry recommended but did not test is **+56.5** — against +1.1 for the
uncut baseline.

So the positive control works, and works well. What actually failed was my
measurement of it.

Two consequences. The ablation below is the strong form of the paper's claim.
And the human study's cutaway condition was removed for a reason that does not
hold: if it is reinstated it should use the **floorplan slab**, not the quadrant
cut, since that is both the strongest signal and the most legible to a rater.

### §34.5 View-protocol ablation — it is not the encoder (2026-09-18)

n=256 probes vs 512 reference, noise floor from 48 fresh real draws, paired
bootstrap. Nothing computed on exterior views clears 2 noise floors:

| condition | separation | sees interior? |
|---|--:|---|
| dinov2b / 4 ortho (the paper's) | +1.1 [0.8, 1.4] | no |
| CLIP ViT-B/16 | +0.7 | no |
| CLIP ViT-B/32 | +1.7 | no |
| ResNet-50 (ImageNet-supervised) | +1.0 | no |
| Inception-v3 pool3 (the classic FID backbone) | +0.7 | no |
| dinov2b / **8** ortho | +1.8 | no |
| dinov2b / **16** ortho | **+1.6** | no |
| dinov2b / 4 perspective | +0.9 | no |
| single view, no mean-pooling | +1.4 | no |
| **cut: facing quadrant** | **+9.1** | **yes** |
| **cut: roof off** | **+20.3** | **yes** |
| **cut: floorplan slabs** | **+56.5** | **yes** |
| `geom_kid` (no renders at all) | +120.8 | yes |

Three supervision regimes — self-supervised ViT, language-supervised ViT,
label-supervised CNN — land in the same 0.7–1.7 band. **Sixteen views covering
roof and underside is *lower* than four.** The same encoder and the same
statistic separate the same probe at 9–57 floors the moment the renders show the
inside.

**The interior is absent from the input, not mishandled by the features.** That
is the claim the paper should make, and it is now measured rather than argued.

Caveat kept: this shows the interior is *recoverable*, not that a cutaway makes
a good metric — a section view discards the facade, and `geom_kid` sees the same
damage at +120.8 with no rendering at all.

---

## §35. Two gaps closed before submission: a legitimate bad pool, and the converse probe (2026-09-18)

Two things in §26 were open, and both were the kind of open that a reviewer
finds rather than the author.

**The mixture ladder was mixing the wrong thing.** Its "bad" pool was canon-8
decimated real data, and this project had already decided decimation is not a
quality axis — it changes a build's size, not its goodness. A canon-8 house has
47 blocks against a real 870. So every coefficient in the quadratic table was
keyed to a pool we had disavowed in our own control set. Re-run against
`uniform_random` and `shell_box` at n=128, the *conclusion* survives and gets
stronger: the best-fit exponent is 1.99 against the identity's 2 (the disavowed
pool gave 1.74), and `sqrt(KID)` recovers the failure rate to 0.011 against 0.124
raw. The *numbers* all move — 0.043/0.546 becomes 0.110/1.718, and "a quarter-bad
arm reads 7.9% of fully bad" becomes 6.4%, which is closer to the 6.25% the
identity predicts than the published figure was.

The lesson is the same one T25 taught in a different costume: **a probe's pool is
part of its claim.** We had already written down why decimation is not a quality
reference, in `fast.control_arms`, and the dose script kept using it for six
weeks anyway because nothing connected the two files. The re-run cost twenty
minutes of GPU.

Also worth keeping: the "14× better than linear" ratio is not a measurement. It
is a ratio to a residual that is already at the sampling floor, and it swings
from 23× to 127× between two draws of the same mixture while the exponent stays
at 1.99. Report the exponent.

**The interior probe only pushed one way.** `solidify` *adds* hidden mass, and a
reviewer can say the render metric is blind to that because a filled shell is
still a shell. So `probes.hollow` was built to delete it instead: keep only
blocks with a face on air that reaches the outside. It removes a median 13.4% of
a real build, 98.4% of builds, and leaves the exterior air region *exactly*
invariant — checked 128/128 — so the rendered surface is identical block for
block. MV-DINO-KID moves +0.4 noise floors (interval overlapping real);
`geom_kid` moves +156. Blind in both directions.

The measurement that makes it unambiguous is the pixel one: hollowing changes
0.585/255 of the image against a re-render floor of 0.0000, so the information
*is* in the input and the encoder throws it away. Without that number, "the
metric does not move" and "the picture does not change" are the same table.

Runs: `outputs/run_20260918_203311_doseresponse_uniform_random-shell_box-canon8`,
`outputs/run_20260918_203913_..._mix1`, `outputs/run_20260918_203947_bench_hollow`
(each with a `FINDINGS.md`), plus `run_20260918_202500_doseresponse_canon8_n64_repro`,
which reproduces the published canon-8 numbers through the refactored script so
that a changed number is a changed measurement and not a changed formula.

## §35b. The significance test was miscalibrated, and the power control did not survive (2026-09-19)

`compare.paired_delta` estimated the SE of a difference as `draws.std(ddof=1)`
over 80%-subsample replicates drawn **without replacement**. That is the spread
of a subsample statistic around the full-sample value, not the SE of the
full-sample statistic; m-out-of-n subsampling needs a rescale first, and none was
applied — three modules away, `ladder._noise_floor` applies exactly such a
correction, which is what made this a bug rather than a convention.

Measured type-I error on exact nulls (two arms from one distribution, `kid`,
d=32, n=96 vs 240 reference):

| null regime | old test | permutation (new default) |
|---|---|---|
| arms **=** reference | 0.170 / 0.078 | 0.050 / 0.011 |
| arms **≠** reference | **0.360 / 0.225** | 0.049 / 0.011 |

(`P(p<0.05) / P(p<0.01)`; permutation pooled over 6–18k trials.) The originally
recorded 2.1x was measured in the gentler regime. In the regime a leaderboard
lives in — two generators, neither distributed like the reference — the old test
rejected **7.2x** too often at α=0.05, and Holm over 153 pairs operates near
3e-4 where it is worse.

**No rescale fixes it**, which is why the fix is a permutation test and not a
factor. Conditional on the reference, the arm term's influence is
`u(x) − v(x) = E_P k(x,X') − mean_j k(x,ref_j)`: non-zero when the arm differs
from the reference (non-degenerate U-statistic, `Var = O(1/n)`, factor
`sqrt(m/(n−m))` = 2) and zero when it matches (degenerate, `Var = O(1/n²)`,
factor `m/(n−m)` = 4). Measured against a Monte-Carlo truth the required factor
is 1.31 in one regime and 2.06 in the other, and across the 153 pairs of the real
table the permutation-to-subsample SE ratio ranges **0.30 to 11.9**. The default
is now an exact label-permutation test against the fixed full reference, with the
Gram matrix formed once per pair and every replicate read off it by index
(`distances.kid_gram`, ~300x) — 10k replicates of a 128-vs-128 pair in 0.1 s.
`--sig-method subsample` reproduces old cards and nothing else.

**Recomputed on the run of record** (`run_20260917_043807_bench_canonical-split…`,
18 arms, 153 pairs; replayed twice, `geom_kid` bit-identical to the card,
`mv_dino_kid` to ~1% under GPU re-embedding):

- `real@solidify` vs `real_test` on `mv_dino_kid`: p 0.61 → **0.77**. The
  interior-blindness headline **holds**, and the conservative direction means it
  could only have got safer.
- `train_verbatim` vs `real_test` on `mv_dino_kid`: Holm 0.023 → **0.050 / 0.056**.
  **The power control fails**, and exactly on the boundary: in r1 it clears its own
  Holm threshold by 4 parts in 10 000 (p 4.1650e-3 vs 4.1667e-3) and is refused
  only because the step-down stopped one rank earlier on an unrelated pair that
  missed by 0.6%; in r2 it misses its own threshold by 12%. Re-embedding the same
  builds moves the raw p by 12%, i.e. 300x the r1 margin. It does clear the
  120-pair family of today's smaller control roster (0.039 / 0.042). "The test is
  not underpowered: at the same sample size it separates verbatim copies at
  Δ=0.015 and gives them a group of their own" (intro, appendix,
  `docs/benchmark.md`) is no longer supportable. The dose-response power argument
  is independent and untouched; the fix is n=256.
- `canon_oriented` vs `llm_control` on `geom_kid`: p 0.89 → **0.93**. The case
  study's tie **holds**.
- `native_oriented` (contaminated) on `geom_kid`: separated from `llm_bias`
  (Holm 8e-9 → **5.5e-3**, holds) but **not** from `llm_control`
  (3.4e-5 → **0.079**). It now also ties with `canon_oriented` (0.48), so "the
  contaminated arm occupies a significance group of its own" is gone. Both
  comparisons survive on `mv_dino_kid`.

Lost in total: 3 of 140 separated pairs on `geom_kid` (1 gained), 6 of 146 on
`mv_dino_kid` — concentrated entirely among submissions and near-real controls.

**The letters were also not a compact letter display.** `_letter_groups` gave
each arm one letter by greedy first-fit (a `letters` list built for multi-letter
output, defeated by a `break`), so "does not share a letter" implied nothing:
"not separated" is not transitive. It is now the maximal cliques of the
"not separated" graph, so sharing a letter and being unseparated are the same
statement in both directions — and under the corrected verdicts 4–5 of the 18 arms
need more than one letter, which is the honest shape of these tables.

Full write-up, per-pair numbers and reproduction:
`outputs/run_20260919_022424_significance_recompute/FINDINGS.md`.

---

## §36. The agentic methods suite: a harness for dream-and-build methods (2026-10-03)

**What.** Track E grew from one loop (`BuildAgent`) into a harness for agentic
methods (`blockgen/agentic/{core,recipe,kit,episode,suite,...}.py`, CLI
`python -m blockgen.agentic`, guide `docs/agentic-suite.md`). A method is a short
Python function over an `Episode` (ask, try, render, dream, score, best); a YAML
recipe picks one choice per axis (LLM per role, image generator, renderer, action
space, contexts, scorers, accept policy, budget). The headline method is the dream
loop: generate an image of the target (iso, back, interior cutaway, floor plan),
build, render the build at the dream's camera, compare, iterate, with a dream critic
that can regenerate the dream before building and re-imagine it when the build
stalls.

**Why this shape.** Three designs were proposed (P1 Episode + plain recipes, P2 a
CaP-X style gym, P3 a dream grid). The skeleton is P1, because control flow such as
redream-on-stall, voting and early stopping is ordinary `if`/`for` in a function,
whereas a mega-stage with flags becomes unreadable by the third option. Grafted on:
P3's cache modes, a ledger with `would_cost_usd`, `recipe_hash`, `estimate`, a shared
dream cache and the variants sheet; P2's `dream_id` re-anchoring (scores measured
against different dream versions are never compared), the feedback-mode knob (raw
images vs verbal differences, the CaP-X question), the spend gate, the `legacy`
wrapper and scorer calibration against distractors.

**Legacy untouched, on purpose.** `agent.py`, `canvas.py`, `dsl.py`, `prompts.py`,
`examples.py` and `deploy/**` are byte-identical. The live `/gen` path, the
published scaffolding and ontology numbers and the response cache all depend on
BuildAgent's exact request bytes. Method `legacy` wraps BuildAgent; method `classic`
re-implements it natively and is parity-tested on request fingerprints for every arm
x critique mode x keep_history (480 combinations through the real kit path). One
legacy bug is fixed only in the new world: `Canvas.clone` drops the clipboard, so a
patch-mode `paste` of an earlier copy fails in BuildAgent and works in
`MinecraftWorld`.

**Single IR.** Every action space (WorldEdit DSL, `<block> x y z` lines, sandboxed
Python, fenced mixed) lowers to canonical DSL executed by the legacy executor, so
replay, reports and deploy stay shared and a new representation is one class.

**Cost discipline.** Development is offline (`--mock` swaps every backend for a free
twin; `estimate` prices a run as the real models would, with zero network calls).
A cache miss on a paying backend needs `--allow-spend`; ledgers check caps before
each call; `read_only` reruns provably cost $0; dreams sit in one content-addressed
cache shared by every arm, so holding the dream spec fixed isolates loop variance
from dream variance. Upper-bound estimates (gpt-5-mini + gpt-image-1-mini low, every
repair round counted): classic-matched $0.051, dream $0.058, dream-refine $0.113,
dream-multiview $0.254 per build. Per-build `cost_usd` keeps the legacy full-price
meaning; `paid_usd` is what was paid (cache hits free).

**Integration fixes made while wiring it together** (each small): `kit.build_kit`
passes the canvas size to action spaces that take it and expands `turnaround`;
`Episode.try_` maps python / mixed execution issues back to reply lines; `mock:render`
draws conditioned dreams (back, interior, plan) from the same house as their
reference; `--mock` arms whose dreams are train-split renders are tagged
`train_split: oracle_train`; arms without scorers are rendered after the build so
their contact sheets are not blank; `experiments_agentic` merges arm overrides into
its defaults instead of passing keywords twice.

**Open risks.** (1) The dream to render domain gap: image models draw pictures, not
renders; the in-loop scorers are calibrated only on oracle dreams so far
(n=32 vs 31 distractors, top-1: siglipB 1.00, clipB16 0.97, color_lab 1.00, pixel
0.88, silhouette 0.25; `outputs/run_20261003_130834_calibrate`). (2) gpt-image cameras are approximate
(`view_search` absorbs part of it). (3) Claude Code CLI flags drift between versions;
the version is in the cache key. (4) `IMAGE_PRICES` and the reference surcharge were
not re-verified online.

**Next.** (1) `calibrate` the scorers on a handful of real dreams (`--dreams-dir`).
(2) The `dream-v1` pilot at n=12: 9 arms (classic-matched, dream, dream-refine,
dream-oracle, dream_vdm, and a 4-arm `max_iters` x `dreams` sweep), 108 builds,
$7.08 by the upper-bound estimate; actual spend is lower because dream-cache hits are
shared across arms. (3) If the dream arms beat
`classic-matched` and `best-of-4` at matched `would_cost_usd`, scale to n >= 256 on
free backends (`dream-local`: Claude Code + SDXL-turbo). Out of scope and left as
follow-ups: skill mining, an ensemble arm, a deploy `method` backend kind, and turning
BuildAgent into an adapter over `classic`.

**Review fixes, perception and scoring (2026-10-03).** (1) `dream_palette` named the
wrong materials: candidates were unshaded mean texture colours while two thirds of
an iso view is side faces at 0.85 / 0.70 brightness (magenta_terracotta x 0.79 is
purple_terracotta), and k-means ran on raw texels (bricks split into mortar + brick).
Candidates are now matched at every renderer face shade and the dream is box-filtered
to about block scale (48 px) and eroded one cell before clustering. On oracle
single-material 16^3 boxes the true block is first for 21 of 24 materials and in
the top 4 for all 24 (was first for
about 4 of 12); grey stones (stone_bricks, cobblestone, andesite) remain mutually
ambiguous by mean colour. (2) `foreground_mask` modelled the dream backdrop as one
median colour: it now fits a smooth quadratic backdrop field, median-filters before
thresholding, grows the backdrop into soft contact shadows in the lower half of the
build, and uses an exact-colour backdrop (tol 1, 3x3 opening) for renders, so snow
roofs stay foreground. Limitation: a white roof seen from above on a near-white
noisy backdrop is still lost (contrast below noise). (3) `texture_sheet{blocks:
used}` names only same-texture blocks (no more quartz slab as stone_slab) and is
rejected on an empty world, since contexts run before the first build turn. (4) A
backbone already loaded on one device is reused there instead of crashing an arm
that asks for another. (5) The `material` scorer is the bench's
`material_agreement` verbatim; it now warns and tags its note.

## §37. Staged methods: plan first, then build room by room (2026-10-03)

The first real dream pilot (results T42) said the single-image loop is the wrong
unit. A whole house is too much to change in one turn, and the photo-like dream and
the flat render differ mostly in style, so a rewrite moves the score by noise (embed
0.56 to 0.61, most rounds rejected). Conditioning a floor plan or a cutaway on the
iso exterior image makes the image model redraw the exterior, because an edit keeps
its reference's composition and camera.

Three staged methods follow from this. `blueprint` asks for a typed floor plan
(JSON, repaired deterministically and idempotently), builds the shell from it for
free, dreams the exterior as an edit of a render of our own massing, then dreams and
furnishes each room inside its own region, scored on a cutaway of that room only.
Writes outside a room are reverted and reported to the model. `floors` changes only
the unit (a storey). `turnaround` draws all four views in one 2x2 sheet so they
show one building, and assigns tiles to cameras by silhouette after the first build.
Every image arm has a control in `staged-v1`: no images (`blueprint-nodream` vs
`classic_long` at matched calls), exterior only, shuffled room dreams, LLM shell,
image-made plan, per-view turnaround. Interiors are measured offline with a new
`interior` scorer that no loop optimizes.

Status: mock-verified only. Integration fixes from reading the mock contact sheets:
an exterior door above the ground floor (a door whose far room was dropped by the
room limit) now becomes a window in repair; unit renders exist for every attempt,
including the roof and final columns and the image-free control, so the `u_<room>`
rows are never blank under the selected build; and the mock answers the room limit,
the LLM shell, storey walls and the LLM roof plausibly. Next: a small paid
`staged-v1` pilot after `estimate` ($17.33 upper bound for n=8), then compare
`blueprint` vs `blueprint-nodream` vs `classic_long` on BlockScore, `interior` and
cost.

## §38. Player scale and planning in words (2026-10-03)

Reading the pilot prompts raised two questions. Nothing told the model how big a
player is, so a house is sized by eye and its interior can be a doll's house
(1-block doorways, 2-block ceilings, 2 x 2 rooms) while the iso view looks fine. And
the staged methods plan by drawing (a dreamed floor plan) or by writing JSON
directly, while a model may plan better if it first reasons about the house in
prose and only then commits to coordinates.

What shipped (docs/agentic-suite.md section 5.5): a `scale` context with the
numbers a Minecraft builder uses (player 2 tall x 1 wide, doors 1 x 2, at least 3 of
clear ceiling so 4-5 floor-to-floor, rooms at least 3 x 3 inside, corridors 1-2
wide, windows from 1 above the floor, stairs with 2 of air over every step,
furniture 1 block, a 2-3 room cottage 10-16 per side)
plus a canvas note (simplify rather than shrink). It is system text, so every method
and action space gets it; the plan turn, the blueprint planner turns and the dream
style read the same switch, and the image cue sizes the openings without drawing a
person. It is on in the native dream, blueprint, floors and turnaround recipes and
in `classic-matched` (appended to the legacy prompts so the control keeps them), off
in `classic-full`, legacy and the representation arms. Controls `dream-noscale` and
`blueprint-noscale` are exactly the pre-change recipes (same hashes).

Planning in text: `dream-plan` reads the dream into blocks before the first program
(footprint, storeys, parts with coordinate ranges, door and windows), and
`blueprint-text-plan` (`plan_source: text`) writes a prose design brief (rooms and
sizes, shared walls, how a player walks from the front door to every room, storey
heights) and then transcribes it to the blueprint JSON. The blueprint repair now
drops rooms under 5 x 5 including walls (was 4 x 4, a 2 x 2 interior). A heuristic
`scale_*` diagnostic (headroom over covered floor, door-like openings by height)
is reported for every build and selected on by nothing.

Status: mock-verified only. Next: `battery scale-plan-v1` (estimate $11.34 upper
bound for n=8). Read `dream` vs `dream-noscale` on `scale_headroom3`,
`scale_door_ok` and `interior` first, then on BlockScore: the hypothesis is that
scale fixes interiors the iso score barely sees, so an unchanged BlockScore is not a
null result. Then `blueprint-text-plan` vs `blueprint` vs `blueprint-image-plan` on
repair notes per plan and `interior`.

## §39. Describing the dream in words: a spec before, a caption after (2026-10-05)

The dream loop hands the builder a picture and nothing else, and a builder reads a
picture loosely: it miscounts storeys, guesses a block from a colour, and cannot see
the back. The describe stage puts the building into words on either side of the
image (docs/agentic-suite.md section 5.6, `describe.py`). `before` has a planner
write the request out as a structured spec (footprint, storeys and clear heights,
roof, materials per slot as palette blocks, openings, features, what to omit); the
dream is drawn from it and the builder reads it. `after` has the vision judge
caption what the image actually shows, in the same schema plus block dimensions,
per-face notes and an `uncertain` list, and the builder and the match critic read
it as `TARGET DESCRIPTION`. `both` does the two and tells the builder that the image
and its caption win where they disagree with the spec, since the image is scored.

The control decides whether a gain means anything: `dream-caption-shuffled` makes
the same calls but shows another prompt's caption from the same arm (a library
caption for the arm's first prompt). If `dream-caption` beats `dream` but ties the
shuffled arm, the gain is extra text, not description. Recipes `dream-describe-before`,
`dream-caption`, `dream-describe-both`, `dream-caption-shuffled`; battery
`describe-v1` with `dream` and `classic-matched`. The caption arms reuse `dream`'s
images and `dream-describe-both` reuses `dream-describe-before`'s, so each pair
differs only in the text.

Status: mock-verified only. Next: `battery describe-v1` (estimate $3.17 upper bound
for n=8, cap $12). Read `dream-caption` vs `dream-caption-shuffled` first, then the
`describe_disagreement` count in `dream-describe-both` (how often the image model
ignores a spec).

## §40. The grid context: vertical planes and which face the camera sees (2026-10-05)

In pilot_diverse_v0 every windmill build that did not copy its dream closely laid its
sails flat in the x-z plane. Floors, walls and roofs came out right because they are
the shapes the commands make naturally; a flat vertical feature is the one case where
"y is up" is not enough, since it must also hold one horizontal axis fixed.

Rendering a marker build (walls on x=max and z=max, a pillar at the origin) through the
textured renderer settled the camera orientation: the `iso` dream camera sits at the
high-x, high-z corner, so the image's left face is the z=max side, its right face the
x=max side, and the x=0 and z=0 sides are hidden. The in-context DSL example puts its
door at z=1, which that camera never shows, and nothing told the builder which side a
TARGET face is. The `grid` context (`context.py`, `VIEW_SIDES`, opt-in) states both, and
`dream-grid` is `dream` plus it. `tests/agentic/test_grid.py` re-renders the `iso` case
so the table cannot silently go stale if the views change.

## §41. Attention-as-adjacency toolkit (2026-10-07)

`tools/misc/attention_graphs/` extends T28's probe into a reusable tool. It extracts
per-layer, per-head attention from any HF causal LM or from the repo's voxel AR
checkpoints, pools tokens to block nodes, and plots head grids and attention against the
true adjacency. Every head is scored under three controls: recency (distance-stratified
AUROC), coordinate copying (hard pairs, which share two of x/y/z) and both together
(`hard_strat`). It supports three serializations (absolute `x y z block`, a coordinate-free
parent-plus-direction format and a LEGO stud-graph toy) in raster or shuffled order, plus
trained distance and bilinear edge probes on frozen states. Rerun with
`PYTHONPATH=. HF_HOME=~/hf_cache HF_HUB_OFFLINE=1 .venv/bin/python -m tools.misc.attention_graphs {ar2|llm --lego --models ...|figures --examples} --out <run>`.
The tests live in `tests/misc/attention_graphs/` (17 tests, CPU, tiny random models).
Results are in `results.md` T49 and the literature review is in
`tools/misc/attention_graphs/RESEARCH.md`.

## §42. Adjacency-from-a-block-list prototype (2026-10-07)

`tools/misc/adjacency_gen/` holds the prototype. Its model is a permutation-equivariant
transformer over block or piece types, with a random identity feature per node and tied
antisymmetric direction logits. It is trained by masked edge modelling and decoded
MaskGIT-style, with an optional realizability check (BFS placement, collision rejection).
Decoding ranks pairs by their best direction label, because a plain argmax always picks
"no contact". Run `.venv/bin/python -m tools.misc.adjacency_gen run --kind voxel|piece --minutes 35`,
then `eval --run <dir> --arms model_grow --tag _grow`. The tests live in
`tests/misc/adjacency_gen/` (13, CPU). Results are in `results.md` T50.


## §43. World-crop pilot: mining structures out of archived worlds (2026-10-07)

The data question of the day was scale and releasability: we hold ~50k community builds,
want far more for pretraining, and need a licence-clean subset to publish. A survey
(`research.md §F`) found that no map host exposes a per-map licence except CurseForge, that
creators own their builds under the EULA, and that the single largest ready corpus is the
Minecraft Archive Project on archive.org (~175k worlds, 2011–2018, ~3.6 TB, no licence).
Decision: two tiers, an internal pretraining corpus mined from worlds and a small
licence-manifest release tier (a few thousand builds, usable as a test set).

The pilot (`blockgen/data/worldcrop/`, manual in `docs/worldcrop.md`) streams worlds
straight out of the monthly archive.org zips over HTTP range requests, so nothing large
touches disk. A dependency-light Anvil reader (nbtlib + numpy; Amulet is not free) decodes
McRegion, numeric Anvil and 1.13+ palette chunks into sparse 16³ sections. Detection has
two routes: superflat worlds diff against their preset layer stack; terrain worlds seed
components from STRONG block ids (planks, stairs, glass, wool, redstone, …), grow through
WEAK ids (cobble, logs, clay, obsidian, …), and mask vanilla structures from the
`data/*.dat` bounding boxes. Three filters mattered in practice: a *flat* label is rejected
when ores appear (converted worlds lie about their generator), stone/sandstone must never be
WEAK (one world grew a 3M-voxel "structure" through bedrock-to-surface stone), and families
of many small look-alike components are dropped (modded ores and replaced-ore server maps
read as vanilla stairs and fence gates). The text2mc cutter we already hold did none of
this and is chunk-granular; this one returns tight component bounding boxes with a
membership mask, so downstream can choose "structure only" or "structure in context".

Numbers for 400 archives of 2014-07: 2,651 crops from 366 of 381 Java worlds, median 3 per
world, median largest side 26, 1,562 fitting 32³ and 2,386 fitting 128³, plus 30 whole
schematics recovered from PMC "download schematic" zips. Throughput is 12.5 s per archive
per core (archive fetch, unrar, scan, label), ~8 min wall on 12 workers; the whole archive
is about two days. The sample sheet is a genuinely diverse corpus: houses, towers, arenas,
pixel art, redstone contraptions, statues, skyblock islands. The visible false positive is
WorldPainter terrain (snow and ice mountains made of snow blocks), which is the first
target for a learned gate trained on our labelled houses. Full accounting in `results.md`
T51.

## §43. Face-node adjacency generation (2026-10-07)

`tools/misc/adjacency_gen/` adds the face formulation (`faces.py`, `coarse.py`,
`model_faces.py`). It has three decoders: grow from a seed with a marginalized loss
(`decode_faces.py`), connection-by-connection merging (`merge.py`), and one-shot
Sinkhorn/Hungarian matching. Commands: `faces-run --f {0,2,4} [--oracle] [--merge]
--minutes 32 --bs 32` and `faces-eval --run <dir> --arms ... --tag ...`. The tests (28,
CPU) include a guard that incremental merge-state features equal a from-scratch rebuild.
Results are in `results.md` T52.

## §44. Cell assignment, then connections, on full houses (2026-10-08)

Two-stage graph generation (T53).
- Stage 1 samples a per-cell type-count assignment, using an IPF sampler with exact margins.
- Stage 2 builds within that frame: either a grow decoder over type slots (U-Net, local) or merge-in-cells.
- Commands: `python -m tools.misc.adjacency_gen cells-stage1|cells-grow|cells-merge [--dagger]|cells-eval`.

Lesson: the frame has to be fine enough to pin geometry. At f=4 every learned arm ties retrieval on voxel IoU (about 0.32), even with the oracle assignment. Merge-in-cells stops collapsing into lines but deadlocks, since there is no repair move. Next: f=2 hierarchical assignment, or per-cell occupancy patterns.

Prompt coherence (T51 addendum): CLIP-L retrieval of the build's own prompt among 32. All agentic arms land between dream (0.13) and the real ceiling (0.22), with no significant differences. GrabCraft titles are weak prompts; switch to houses_32 captions or a VLM judge.

## §45. World-crop P1: re-tiling, provenance, and the terrain gate (2026-10-11)

**Ingest.** The full-archive pass is 46 months in: 93.8k worlds with crops, 882k crops,
810k saved (the rest oversize), 7.5 GB. Three defects fixed on the way: archive.org 500s
on month open (retry), OOM-killed workers aborting a month (5 GB per-worker cap plus pool
rebuild), and builds split across tile edges (re-tile around any component touching an
edge; 26 merged, 23 flagged `cut` on a 110-world re-run, everything else identical).

**Provenance.** Zero worlds had a title because `resources.ndjson` carries no referer for
downloads. The per-thread finding-aid pages inside each zip join 99.4% of archives to a
title, author, URL, date and description with one 8 MB range request per month;
`provenance.py --rejoin` rewrote the finished months and the lab shows titles.

**Gate.** Four rounds, each decided by a held-out set of 192 random crops I labelled from
numbered sheets (106 built, 65 fragment, 8 terrain, 13 unsure after correcting nine
labels on inspection; farm plots and a stronghold library had been marked terrain).

| version | negatives | terrain rejected (of 8) | built rejected (of 106) |
|---|---|---|---|
| v1 | 4.4k natural chunk patches | 4 | 0 |
| v2 | + 4k synthetic solid plates/mounds | 4 | 1 |
| v3 | + 4k synthetic surface skins | 4 | 2 |
| v4 | + 10k crops mined by composition (≤3 ids, ≥97% terrain family) | 8 | 3 |

Why v1-v3 failed: the detector masks a crop to its *component*, so an edited mountain is
not a solid mound but a 1-8 block skin of snow block or clay covering 13-45% of the
columns, with fill 0.01-0.26. Natural patches and even synthetic skins were too regular;
real mined examples closed the gap. The three lost builds are a sandstone tower, a long
stone road and a wood/sponge plate; no fragment is rejected, which is the point: the gate
removes terrain only. Cross-validated AUC on the training mix is 0.9998 and meaningless
(sources are trivially separable); the 192-crop set is the number that counts.

**Corpus after the gate.** 746k of 810k saved crops pass at p ≥ 0.5 (92%); 56k rejected,
8k borderline. By size: 433k fit 32³, 532k fit 48³, 682k fit 128³. The reject sheet is
plates, dirt pillars, snow skins and flat-world slabs with roughly one built piece in
ten; the pass set still carries the fragment class (about a third of crops by the hand
labels), so "useful" today is on the order of 400k, and a fragment detector is the next
lever. Sheets: `data/minecraft/worldcrop/gate/sheet_{pass,reject,border}.png`.
