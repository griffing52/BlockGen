# BlockGen — Research Notes

Living lab notebook for the project: reproducibility, design decisions, and the
running ledger of **proven** vs **disproven** ideas. Companion files:
[`results.md`](results.md) (tables / figures / ablations), [`research.md`](research.md)
(literature + strategy), [`roadmap.md`](roadmap.md) (current plan → LEGO paper thesis).
Keep this honest — dead ends are as valuable to record as wins.

_Last updated: 2026-07-06._

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
