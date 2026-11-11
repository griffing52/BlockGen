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

> Sections 7 to 9 stop at 2026-07-15. The complete list of everything tried, with verdicts through T88, is [`tried.md`](tried.md).
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

## §46. Corpus layer: manifest, dedup, tiers, and the village leak (2026-10-12)

First milestone of the pipeline plan (`docs/pipeline.md`). `blockgen/corpus/` now holds
one Parquet manifest over every build we own (915,853 world crops from 47 months plus
10,461 curated rows from the all_32 / houses_32 / houses_48 caches), with geometry,
provenance, gate score, a canonical hash that folds translation and the eight
horizontal symmetries, a 128-value MinHash signature, and the 295 gate features as a
float16 sidecar. The build is resumable per month, skips the month the ingest is
writing, and gate-scores any crop that has no score, so the manifest replaces the
separate scoring pass. 926k crops took 70 minutes on 6 workers next to the ingest.

**Dedup.** Exact matching folds 216k rows; MinHash LSH (16 bands of 8, accept at
estimated Jaccard ≥ 0.8, builds under 20 blocks excluded) folds another 61k, so 29.9%
of the corpus is a duplicate of something else and 648,917 clusters remain. Duplicate
rate falls with size (38% under 50 blocks, 24% above 2k), as expected if small
structures repeat and large ones are unique. 164 curated builds also occur as world
crops, which is the cross-source leak the hold-out exclusion exists for.

**The village leak.** Sorting clusters by the number of distinct worlds they occur in
exposed a detector blind spot: the 24 most widespread clusters, each in 430 to 4,870
worlds, are all vanilla village pieces (lamp posts on gravel paths, wells, farm plots,
huts, the butcher, the church, a desert temple corner;
`data/minecraft/corpus/sheet_top_clusters.png`). The vanilla-structure mask in the
detector covers strongholds and mineshafts but not villages. Recurrence is the cheap
fix: 6,444 clusters occur in five or more worlds, covering 139,866 rows (15% of the
corpus). They are kept in the manifest as a `vanilla` tier (a negative set for the gate,
and a natural "generated vs built" probe) and excluded from `pretrain-clean` and
`anneal` by the `cluster_worlds < 5` filter. `pretrain-all` keeps one representative per
cluster, which is 6k builds of 648k and harmless.

**Tiers** (`configs/corpus/*.yaml`, hash of the member ids in
`data/minecraft/corpus/tiers/<name>.json`):

| tier | builds | blocks | fit 32³ |
|---|---:|---:|---:|
| pretrain-all | 647,838 | 10.7B | 358k |
| pretrain-clean (p ≥ 0.5, not recurrent) | 602,406 | 6.3B | 340k |
| anneal (curated ≥ 80 blocks, or p ≥ 0.9 and ≥ 500 blocks) | 281,240 | 5.7B | 66k |
| eval-holdout (houses_32 val+test) | 798 | 0.7M | 798 |
| vanilla (recurrent, one per cluster) | 6,444 | 138M | 3.9k |

One correction to the plan's compute estimate: the corpus is 10.7B blocks, not the
1B tokens assumed from a 1k-block average, because large crops dominate the block
count. Windows over large crops (plan §3.6) are therefore the sampling policy, not an
option, and an "epoch" will be defined in windows, not in builds.

### §46.1 Mineshafts, and kinds as a sidecar (2026-10-13)

The user noticed mineshafts throughout the corpus. Recurrence cannot catch them: the
generator randomises corridor layout, so no two shafts hash alike. Composition can.
The manifest sidecar already stores the per-id histogram of member voxels, and a
mineshaft is made of six things: planks, fences, rails, cobwebs, torches, chests.
Three contact sheets of 72 random crops each set the thresholds. At a mineshaft-material
share of 0.9 or more with at least one rail or cobweb present, 70 of 72 are shafts (the
two exceptions are a dungeon fused to a shaft and a player's rail line). Between 0.5
and 0.9 the sheet is half shafts and half player work that happens to use rails and
fences (roller coasters, docks, pens), so that band is tagged `mixed` and kept. Below
0.5 it is player work. Planks and fences alone without the tell (0.9 share, no rail or
cobweb) are mostly docks, frames and boxes with perhaps one in six being a bare
support pillar, so they stay `build`.

`blockgen/corpus/kinds.py` writes `kind` in {build, mixed, village, mineshaft} and
`mine_share` as a sidecar; `tiers._frame` merges it like `dedup.parquet` and configs
filter on it. Over 926,314 rows: build 714,398 (77.1%), village 116,483 (12.6%),
mineshaft 85,376 (9.2%), mixed 10,057 (1.1%). Tiers after the rebuild: pretrain-all
unchanged at 647,838 (kinds are tags there); pretrain-clean 602,406 to 547,789; anneal
281,240 to 274,413, which means 6,827 mineshafts with p_built at or above 0.9 and at
least 500 blocks had been sitting in the decay-phase mix; vanilla 6,444 to 61,130 once
it also holds one representative per mineshaft cluster.

Why the mask misses them: 58% of worlds carry no `data/*.dat` structure files at all,
and the village rows are not concentrated in those worlds (62% vs 58%), so pieces also
leak in worlds that have the boxes. The detector stays as it is; both leaks are now
caught downstream, where the fix is a column and not a re-ingest.

The user's position, which this adopts: vanilla content is a known-good class, not
rubbish. It stays in `pretrain-all` once per cluster with a `<src:vanilla>` tag, and
whether it belongs in the anneal mix becomes a config choice. The same reasoning led
to the game's own structure templates as a source (§47).

## §47. The game's own structure templates as a source (2026-10-13)

Following §46.1, the user asked for procedurally generated structures that are known
good, at larger scales than the village fragments the crops contain. The archived
worlds cannot supply them: the detector masks the big generated structures on purpose,
and pre-1.14 villages have no large piece. The game itself can. Since 1.14 the client
jar ships every jigsaw structure piece as an NBT template under
`data/minecraft/structure/`, and the fabric-loom cache from the deploy mod already
holds the 1.21.1 client jar. `blockgen/corpus/templates.py` reads the 1,180 templates
straight from the zip, maps modern block states onto the legacy vocabulary through
`block_remap` (396 distinct states; the twenty it did not know, mostly jigsaw markers
and potted plants, get explicit overrides, and a jigsaw block becomes its
`final_state`), writes one npz per piece under `data/minecraft/templates/`, and adds
them to the manifest as `source == "templates"` with ids `tpl:<family>/<name>` and the
family as `route`. 1,113 pieces have blocks; 1,007 survive the 20-block floor and dedup
(mirrored and rotated variants fold together under the canonical hash).

Families: village 483 (plains, desert, savanna, taiga, snowy, plus zombie variants),
trial chambers 170, bastion 167, trail ruins 84, woodland mansion rooms 73, ancient
city 58, underwater ruins 48, end city 20, shipwreck 20, fossils 30, ruined portals
13, pillager outpost 11, igloo 3. Median piece is 7 by 7 by 8 at 308 blocks; the
largest is a bastion bridge base at 14k. Two caveats on the sheets
(`data/minecraft/corpus/sheet_templates_random.png`, `sheet_templates_largest.png`):
the trial chamber, bastion and ancient city pieces are interiors carved into solid
stone, so from outside they are boxes, and village pieces carry a slab of ground
under them like our world crops do. Both are what the game places, and both are
tagged by family, so a tier can drop them with `route not in [...]` later.

Added to the sources of `pretrain-all`, `pretrain-clean` and `anneal` (the anneal
rule for non-worldcrop sources, 80 blocks or more, applies), and as a tier of their
own, `templates`, for probes. Licence: Mojang assets train an internal model and are
never released or used as a benchmark reference; the tier description says so.

Still missing, and the natural next step for known-good large structures: the
code-generated ones (woodland mansion as a whole, ocean monument, stronghold, nether
fortress, desert and jungle temples, mineshafts as positives). Those need a world.
The deploy server in `deploy/mod/run` can generate a few seeds, locate each kind, and
the chunk NBT records every structure start's bounding box, so cropping them is the
same anvil read the world-crop pipeline already does. Deferred until the shard
builder exists.

BlockLab now shows all of this. Subsets gained a `tier` rule
(`tools/lab/subsets.py`) that resolves a materialised corpus tier to positions in
the raw world-crop index by file path, so a stale index drops rather than
mislabels; `outputs/lab/subsets/tier-{anneal,pretrain-clean,vanilla}.json` are
seeded random 600s of each tier, and the templates are a raw corpus of their own
under Raw corpora.

## §48. M1: shards and the shared trainer (2026-10-14)

The second milestone of `docs/pipeline.md`. Everything a run needs now lives in two
packages and three commands.

**Tokenizer as an artifact.** `blockgen/corpus/tokenizer.py` fixes one vocabulary for
every tier: the three specials, 48 shared coordinate tokens, the 441 block classes the
standard vocabulary can produce plus `<unk>`, and 15 prefix tags in four groups
(`src` archive/curated/vanilla/templates, `q` 0..4 from `p_built`, `size` 16/24/32/48,
`win` whole/window), 507 tokens in all. It is not derived from a dataset, which is the
point: the old trainers built a vocabulary per run, so no two checkpoints agreed on
what token 57 meant. The sha of the saved JSON (`60684456e598` for the default) names
the shard directory and goes into every checkpoint; the oriented variant is 926 tokens.

**Shards.** `python -m blockgen.corpus.shards build <tier>` writes
`data/minecraft/corpus/shards/<tier>/<tokenizer>/` with every sequence back to back as
uint16, an offsets array, a meta table and a spec. A build that fits 48³ with at most
2,048 blocks is one `win:whole` sequence; a bigger one yields up to 32 windows centred on
random occupied voxels, retried at edges 48, 32, 24, 16 until a window holds at most
2,048 blocks, and dropped if a 16³ window is still too dense. The 2,048 cap is the sparse
raster arm's budget (8,198 tokens), not a property of the data; the diffusion arm will
take dense windows from the same manifest. Window choice is seeded by the build id, so
a shard is reproducible. The anneal tier sharded in 5.5 minutes: 275k builds into
1,241,999 sequences, 6.57B tokens, 1.64B blocks, 13 GB; windows outnumber whole
builds nine to one and a third of them are 16³, which says how dense the big world
crops are. The pinned houses_32 train split is its own tier, `sft-houses32`, 1,863
whole builds sharded without a cap so the scratch baseline sees the same houses every
old houses32 arm saw; `eval-holdout` is the 798 val plus test houses, also whole.

**Trainer.** `python -m blockgen.train --config configs/train/<name>.yaml [key=value]`.
One flat typed config with presets 5m to 300m, YAML `extends`, and command-line
overrides. Token-budget batches mixed across shards by weight (a buffer is sorted by
length and cut at 32k padded tokens, so padding stays small without a fixed batch
size), warmup-stable-decay schedule, bf16 autocast, gradient accumulation, AdamW with
decay only on matrices. Checkpoints carry model, optimiser, step, tokens seen, the
config, the tokenizer hash, the sha of every tier, the git sha, the sampler state and
both RNG states, so `resume=<run dir>` continues bit for bit and `init_from=<ckpt>`
starts a new stage from old weights and refuses a tokenizer mismatch. Every run writes
`metrics.jsonl`, validation NLL per tier, `samples.png`, a sample cache the benchmark
reads, the fast-tier scorecard under `bench/`, a `summary.json`, and one line in
`outputs/registry.jsonl` with the BlockScore and its worst pillar.

Smoke test (`configs/train/smoke.yaml`, 60 steps): the 5.5M model runs at 420 to 560k
tokens per second on the 5070 Ti, holdout NLL 3.31 after two million tokens, sixteen
samples rendered and scored in 16 seconds, resume verified after fixing one bug (RNG
state tensors come back on the GPU and must be moved before restore).

**Disk.** The anneal shard alone is 13 GB and 13 GB are free. The pretraining tiers
cannot be sharded until space is cleared; the candidates are listed in the status of
2026-10-14 (Kaggle and HF caches, conda's unused packages, the journal).

## §49. The phone queue: labelling in spare minutes (2026-10-14)

The question was what to do with short idle stretches on a phone that adds up to
something the project needs. The answer is not a new kind of data. Two things
in this repo are gated on human judgements that nobody has had the sitting-down
time to make: preference tuning (pipeline stage 7.4) waits on 2AFC pairs, of
which the store held 30, and the curated house caches have never been checked by
a person, so the nine numeric gates stand in for "is this a house". Puzzle-play
traces, the other idea, already exist at a scale one person cannot match (H-ARC
has attempts from 1,700 people over all 800 ARC tasks; ARCTraj has about 10,000
trajectories), and neither reports a model that improved by training on them.

So the lab grew a one-thumb page, `/phone`, and a server-side queue behind it
(`tools/lab/phone.py`, configured by `configs/lab/phone.json`). The page asks
for the next few tasks, shows one, records the answer through the existing
`/api/label` and `/api/compare` routes plus a new `/api/caption`, and asks again.
A judgement made on the phone is indistinguishable from one made at the desk.

Three task kinds, in the order they matter:

- **compare** (default 60% of the mix): a forced-choice pair. Two tags.
  `more_real` is the human-study question, real test houses against the
  canonical bench arms. `better_build` is the preference signal, arm against
  arm. Pairs are drawn to spread appearances across builds (four candidates,
  take the least seen) and an identical pair is never asked twice under one
  tag. Sides are randomized on the server.
- **curate** (40%): good, bad or unsure on one build, tap to turn it through
  the four views, swipe left or right as a shortcut. Each dataset is walked in
  a fixed seeded shuffle and anything already labelled is skipped, so progress
  is simply what is in the labels table and a session resumes by itself. One
  dataset finishes before the next starts, so a half-done pass is a complete
  label set over a prefix rather than a thin scatter. First `houses_32` and
  `houses_48` (the SFT set), then `3dcraft` and `grabcraft`.
- **caption**: a one-line description of a build, with word chips so a caption
  can be two taps and a word. Not in the mix because typing is the slow part;
  it has its own mode. Builds already marked good come first.

Mechanics that matter on a phone: every action has a 1.8 s undo window before
the write is sent (comparisons are append-only, so this is the only undo they
can have); writes go through a persistent outbox that retries when the network
comes back and is flushed with `sendBeacon` on leaving the page; the next two
tasks' images are preloaded so a cold render (about 2.3 s for a view the cache
has not seen, 3 ms for one it has) never shows as a grey box. `python -m
tools.lab --phone` binds every interface and prints the LAN and Tailscale URLs;
there is still no auth, which is a trusted-network decision the flag makes
explicit.

The arms in the config are pinned by id rather than globbed over `outputs/run_*`:
that tree is full of mock, smoke and pilot runs whose builds are not worth a
human second. Captions go to their own table, not `notes`: a note is what is
wrong with a build and a caption is what it is, and the second is destined for
the SFT conditioning set while the first must never be.

What the yield should be, if the first sessions hold up: a 2AFC trial is two to
four seconds, so a quarter hour is 150 to 250 pairs and a month of most days is
several thousand, past the few hundred diverse pairs the DPO literature treats
as workable and past the n>=256 the human study needs. The curate walk over the
two house caches is 5,749 builds; at a second or two each that is a few hours
of waiting-room time spread over weeks.

### §48.1 First scratch-vs-pretrain result, and the truncation bug (2026-10-14)

With M1 in place the first question could be asked the same afternoon: does
pretraining on the world crops help a 5M model on houses_32? T58 says yes on
likelihood (holdout NLL 0.461 from pretrain then SFT, against 0.532 from scratch, 13%
lower), with the pretrain-only model already within 0.014 of scratch on houses it never
saw. The samples agree with the likelihood. The benchmark did not, and tracing why found
a trainer bug worth recording: sequences longer than the context were truncated rather
than dropped, so they lost their EOS, and the model learned that one build in eight never
ends. Both runs sampled to the token budget (median length 7,596 and 8,198 tokens of a
possible 8,198) and the geometry pillar punished the oversize builds. The fix drops
over-length sequences in training and evaluation. The lesson generalises: in a format
where the sequence boundary is the building boundary, truncation is not a loss of data,
it is a corrupted label.

Three things to carry into M2. First, a pretrained 5M model samples windows, as it
should: the pretrain-only sheet is fragments and terrain slices in archive palettes, and
only SFT turns the `src:curated` tag into whole houses. Second, 591M tokens is 9% of
the anneal shard; the pretraining curve was still falling at the end. Third, the
decisive experiment needs matched compute (scratch for 817M tokens, not 226M) and
three seeds; this pilot is one seed and unmatched, and is logged as a pilot.

### §48.2 v2, batched sampling, and the M2 pilot launched (2026-10-14)

With truncation fixed the ordering holds and widens: pretrain then SFT reaches holdout
NLL 0.426 against 0.543 from scratch (22% lower) and BlockScore 95.5 against 138.7,
with sheets to match (T58). The scorecard's 63-point swing for scratch between v1 and v2
is the reminder from the eval-suite work that the geometry pillar needs n of 256 or
more; the likelihood ordering never moved. Sampling 256 builds per run was the cost
that made that impractical, so `train/sample.py` now decodes in batches through the
model's KV cache with per-row EOS tracking: 64 samples in 191 s against roughly 7 s
each one at a time, so 256 is 13 minutes rather than 30. `load_run()` in
`train/trainer.py` rebuilds model, tokenizer and config from a run directory.

The M2 pilot is launched as `blockgen-m2` (`scripts/m2_seeds.sh`): for seeds 0, 1, 2,
a matched-compute scratch run (28,000 steps, the pretraining and SFT budgets added), a
20,000-step pretrain on anneal, and an 8,000-step SFT from it, each benchmarked at
n=256. About six hours of GPU. Still the 5M size; 20M and 60M follow once disk allows
the pretraining shards.

### §48.3 M2 read: dropout wrecked two of three SFT arms, and the early-stopping protocol (2026-10-15)

The M2 pilot finished overnight and the first read of the nine registry lines was that
pretraining buys 3% in NLL and loses on the scorecard (T59). Three faults in the
protocol, not in the corpus, sit under that reading. The SFT config turned dropout on at
0.1 for a model pretrained at 0, and two of the three pretrained models collapsed to
near-uniform predictions before the first SFT update (train loss 5.6 at step 20; eval
mode on the same batches 0.6). That is a known property of transformers trained without
dropout, and the trainer now refuses `init_from` across a dropout change. The matched
scratch run overfits houses_32 by step 15k (156 epochs by 28k) and the final-step
comparison pits an overfitted scratch model against an unconverged SFT model, so the
trainer keeps `best.pt` and, with `select: best`, benchmarks and registers the
early-stopped checkpoint. And the scorecard unit shrinks with n, so the n=64 numbers of
T57 and T58 are on a different scale from these at n=256; only equal-n comparisons mean
anything, which is one more reason to pin n=256 and never quote a smaller one.

The one valid seed is the preview: pretrain then SFT 0.375 against scratch at its best
0.372, with the SFT curve still falling. M2b reruns SFT (16k steps, dropout 0) and the
matched scratch (36k, dropout 0) under early stopping from the same three pretrain
checkpoints; about six hours. The shard builds for `pretrain-clean` and `pretrain-all`
run alongside now that the disk has room (98 GB free this morning), so the 20M runs can
start on the full tiers as soon as M2b reports.

Two smaller observations for the pretraining design. The pretrain-only models, sampled
with `win:whole` tags, produce window-like fragments (43 to 80 connected components),
which says the whole/window tag alone does not switch the model into whole-build mode at
this size and the anneal mix needs more whole builds, or the SFT stage is what teaches
closure. And the pretrain seeds differ by 0.12 nats in holdout NLL after the same 20k
steps (0.458 to 0.579) with non-monotone curves, because 591M tokens is 9% of the anneal
shard and each seed sees a different ninth; holdout NLL on curated houses is not the
pretraining objective, so the curve is a weak early-stopping signal for that stage.

## §50. BrickNet reproduction, and what its data would add (2026-10-20)

BrickNet (Kulits and Schmid, CVPR 2026) is the closest published relative of our
Track C: an LLM (Qwen3 instruct, 0.6B to 14B, LoRA r=32 on all projections) fine-tuned
on build sequences serialised as spanning trees over a typed connectivity graph (stud,
hinge, axle, ball, fixed), each step naming the parent part, the new part and colour,
and the connector pair with a quantised joint (degrees, LDU slide). No coordinates are
ever emitted; the realiser recovers poses from the tree. There is no rollback or
constrained decoding; collision checking is used only to filter training walks and to
score generations. The dataset (253,623 pretraining graphs from "publicly available
LDraw structures online", 67,185 captioned SFT models, a 512-model val split) is gated
behind a request form; the ten LoRA adapters and the `bricknet` package (MIT) are
public. The clone lives in `libs/BrickNet` with its own venv, pinned to package
v0.1.0 because v0.1.1 (2026-10-01) replaced the inset-mesh triangle collider with convex
decompositions and the README says the paper numbers need v0.1.0.

What is reproducible today without the data is Table 2 (unconditional validity from
the pretrained adapters) and, qualitatively, text conditioning from the SFT adapters.
Table 3 needs the 512 val captions and their render pipeline plus three large
image-text scorers; Table 4 needs the val paths. The 0.6B pretrained adapter, 100-part
sequences, scored by the v0.1.0 collider (results.md T61): nucleus sampling gives a
mean connected prefix of 94.5 steps against the paper's 94.1 and a collision-free
prefix of 13.6 against 16.0; ancestral sampling gives 41.6 against 45.0 and 10.7
against 11.0. Three of four numbers are inside the sampling interval at our n; the
nucleus collision number is 15% low and the gap is outside the interval, so either a
decoding detail differs (dtype, the top-k/top-p order in HF generate) or the paper's
2^16 draws land differently; it does not change the paper's reading, that collisions,
not connectivity, are the binding failure, with almost no 100-part sample
collision-free under either sampler. The whole run was about two hours on the 16 GB
card at batch 32; batch 64 runs out of memory because Qwen3 uses a dynamic cache.

The data question. Our OMR scrape (1,819 model files, 1,470 sets) goes through
BrickNet's parser once the MPD submodels are flattened (`repro/omr_to_bricknet.py`;
subfiles ending in `.dat` are embedded part definitions and must be leaves). 80% of
placements are in BrickNet's 14,583-part vocabulary; the missing fifth is almost all
primitives and LDCad-generated flexible parts (hoses, strings, `ls`-series shortcuts),
the same tail our shadow-library coverage note already excludes. BrickNet's connector
labelling finds 2.3 edges per part, 80% stud, 17% axle, the rest hinge, fixed and
ball. Their collision-free walker places 84% of a component on average and completes
60% of components of four or more parts; the rest hit inset-mesh collisions that real
sets contain (tight Technic fits, pre-bent parts). Under BrickNet's own SFT filter (4 to
100 parts, collision-free) OMR yields 3,391 complete components from 1,805 models, so
OMR is about 5% of BrickNet-SFT and 1% of BrickNet-PT by model count, and it is
official-set data where theirs is mostly MOCs. OMR is therefore not a substitute for
their release; it is a licence-clean, attributable subset we can publish paths for, and
the likeliest overlap is that OMR is already inside their PT pool, which the perplexity
check below tests.

Two checks beyond Table 2. The pretrained 0.6B model's per-token perplexity on 300
OMR walks of 4 to 100 parts (our flattening, their collision-free walker) is 2.35
against the 1.33 they report on their held-out val paths. A gap that large says OMR is
not sitting in their pretraining pool as we sampled it, with the caveat that their val
walks come from their sampler settings and ours from one default walk per component,
so part of the gap may be walker mismatch rather than distribution shift; the test
to settle it is their val paths once the data arrives.

The caption-conditioned 0.6B adapter run on eight short captions of ours, two draws
each, with the authors' own inference code produced five of sixteen sequences that begin
at node `b` and are unparsable from step one (on the 512 val captions, written in
Gemini's "This LEGO model depicts..." register, this never happens: all 512 start at `a`,
so it is a prompt-distribution effect, see §50.1). The model's next-token distribution after
the caption and newline puts `a` first and `b` second, so this is the model's
behaviour, not a prompt-formatting slip (a merged `.\n` tokenisation makes it worse;
checked and reverted). The paper regenerates unparsable sequences before scoring
Table 3, so its P_inv of 0.256 at 0.6B is on survivors. A local `--force_root` flag
appends the root id `a` to the prompt, the same trick their unconditional mode already
uses; with it 11 of 16 parse fully and 4 of 16 are collision-free, and the realised
valid prefixes (outputs/figures/bricknet_repro/sft0.6B_smoke_8captions.png) are
caption-shaped: the car has four wheels on a chassis, the spaceship and robot draws are
blocky bodies with limbs, the towers are tall stacks, the castle wall is a red
crenellated slab. Unconditional 100-part samples
(outputs/figures/bricknet_repro/pt0.6B_nucleus_20samples.png) are recognisably
vehicles, baseplated scenes and figures with the collisions left in.

What this means for us. BrickNet has done, for LEGO, the typed-connection-token bet
from §E.2, with a far larger private-ish corpus; our open-data angle narrows to the
licence-clean OMR subset plus whatever their release permits. The reusable pieces are
their path grammar and realiser (a Minecraft block is the degenerate stud-only case),
their collision scorer as a validity pillar for a LEGO BlockScore, and the Qwen3 +
LoRA recipe as the LLM-baseline for track D. When the dataset arrives: run Table 4
perplexity and the Table 3 render metrics on val, then the 0.6B retrain
(`scripts/train.py pt`, 250k iters at batch 32, several days on one 16 GB card, so
likely a reduced-iteration check rather than the full recipe).

## §51. BrickAgent and BrickBench: the LEGO twin of Track E (2026-10-20)

BrickAgent (Kulits, Xu, Jones, Schmid, Wu; arXiv 2610.12452, "BrickBench: Evaluating
Agentic Brick Design") is the environment behind BrickBench, a benchmark in which a
coding agent, given a text prompt, writes a Python program that assembles a LEGO model
through BrickNet's connector system. The API is small: `Assembly`, `add` (place at a
pose), `attach` (place by an exact connector fit, with yaw, flip, slide, orient), `find`
and `describe` for part discovery, and `check`, which raises on collisions (BrickNet's
convex colliders), inexact or unauthored contacts, inventory violations, and
instability (each connected component simulated as a rigid body under gravity in
PyBullet, 3 LDU displacement limit). There is no learned model in the environment; the
agent is the model. This is exactly our Track E (an LLM writing a WorldEdit-style
build program against a validator), done for LEGO with a far stricter physical check
and a published benchmark.

The benchmark is 300 prompts in three settings (Model, at most 400 parts; Set, 400 to
4,000 parts; Alt-Build, only the 783 pieces of set 10698), each prompt decomposed into
a DSG-style question graph (1,719, 2,369 and 1,182 questions). Scoring is Valid
(verify.py: part limit, collisions, stability), VQA (Gemma 4 31B answers the question
graph from eight renders, a question counts only if its dependencies hold), and two
Bradley-Terry ELOs from pairwise VLM judgments, Align (to the prompt) and Design
(against the standard of an official set), plus list-price cost. Eleven frontier
agents ran with a 300-turn budget (Claude Opus 5 and 5.5 in Claude Code, the rest in
Codex CLI). Headline numbers, overall: GPT-6 Astra 1.00 valid, 0.954 VQA, ELO 1297
at $4.06 per assembly; GPT-6.1 Sol 1293 at $0.87; Claude Opus 5.5 1249 at $6.33;
Claude Opus 5 1101 at $16.47; GPT-5.6 Luna 898 at $0.75; GLM 5.3 Flash 752 at
$0.54. Five agents are valid on every prompt. The findings that matter for us: VQA
barely separates agents (0.95 to 0.59) while Design ELO spreads them by 550 points,
so prompt adherence and validity saturate and the headroom is design quality; the
environment ablation shows that without BrickAgent, alignment and design scores hold
up but validity collapses (Astra 1.00 to 0.40, Luna to below 0.01 with 156 collisions
per build), so the validator is what makes the output buildable, not the agent; Design
ELO agrees with 102 human raters on 34 of 36 agent pairs (Kendall tau 0.89); and in a
LEGO Turing test five raters picked the human-designed model 323 times out of 360, no
agent passing more than one time in five. BrickNet-14B appears as a data-driven
baseline in the Model setting at 0.26 valid, 0.180 VQA, ELO 731 (Align 485, Design
952), and BrickGPT at 1.00 valid, 0.024 VQA, ELO 338; a frontier agent with no
task-specific training beats both and lands within 0.02 of the held-out ground truth
on BrickNet's own PE, SigLIP 2 and VQAScore protocol (Luna 0.313, 0.818, 0.732 against
GT 0.315, 0.826, 0.748). That is the agentic-beats-trained result we were heading
toward in Track E, published first for LEGO.

Set up in `libs/BrickAgent` (MIT) with its own venv because it needs bricknet 0.1.1
and the convex colliders (1 GB, `.bricknet_data`), separate from the 0.1.0 pin used for
the BrickNet reproduction. The README wall builds and checks in 0.1 s; a floating brick
is rejected by the stability sim with the offending component named. To prove the
loop I played the agent on Model task 004 (a short lord in a red hat and tunic on an
enormous throne, boots above the floor, a red carpet up three steps): 53 parts, four
check iterations (tiles overlapping because a 1x2 covers two studs, rotated plates
extending to negative x from their first hole, a hand inside the throne back, the
seated leg's facing set by the yaw relative to a rotated stud's roll), then verify.py
valid in 1.7 s (results.md T62, render in `outputs/figures/bricknet_repro/`). The
two judges (VQA and ELO) need Gemma 4 31B at about 62 GB, so they do not run here; a
scored pilot needs a cloud GPU or the organisers' submission form (zip of at most 10
MB, `scripts/check_submission.py`).

What to take. For the Minecraft side, BrickBench is the protocol our Track E
evaluation should mirror: question graphs per prompt rather than a single VLM score,
pairwise Design ELO validated against humans, a validity gate that is mechanical, and
an environment ablation. For the LEGO side, running Claude Code on BrickBench is the
cheapest LEGO experiment we have (no training, about $5 per assembly at the Opus
price point, $0.87 for GPT-6.1 Sol), and the open question the paper leaves is the
one we care about: whether a trained generator can supply the design quality the
agents lack, since the human-designed models still win the Turing test by nine to
one.

### §50.1 The dataset arrived: Table 4 reproduces, and what the val split looks like (2026-10-20)

The gated files came through the same evening (`data/lego/bricknet/`: graphs, captions,
and the sft and val walks; the 11 GB pretraining walks and the mesh tarballs we already
hold were skipped for disk). Table 4 at 0.6B reproduces to the third digit under
train.py's own protocol: 1.318 against 1.331 unconditional on 1,000 first-round val
walks, 1.297 against 1.298 caption-conditioned on the 444 pairs. That pins the
released adapters to the paper and turns the OMR perplexity of 2.35 into a clean
statement: our OMR walks are far from their training distribution, so OMR is not in
their pool, or at least not as we sample it. The val split is 512 models with one
Gemini caption each ("This LEGO model depicts a minifigure dressed as an 18th-century
soldier...") and about 18 pre-sampled walks per model across five sampling rounds,
averaging 38 parts and all complete. P_inv (Table 3) on the 512 captions with natural stopping lands at 0.28 by the
placement-weighted reading and 0.20 by the per-sample reading against the paper's 0.256,
whose definition is not pinned down in the text; 90% of samples parse fully and 26% are
collision-free (results.md T61). The first pass used the unconditional recipe's forced
100-part length and is discarded.

## §52. Traversability: a two-block player walks the build (2026-10-23)

The interior measure says whether a build has rooms, not whether anyone can get
into them. The agentic pilots (§38) produce doll's houses: one-block doorways,
two-block ceilings, stairs with no headroom, all invisible from the isometric
view. `blockgen/eval/traversal.py` is the player's-eye measure: a 1 x 2 body is
set down anywhere outside the building and flood-filled over everything it can
walk to. Rules a player obeys: through doors, trapdoors, gates, carpets, torches
and plants (`PASSABLE_IDS`); up one block with a jump, which needs a block of
room over the head; up stairs and slabs without a jump (`STEP_IDS`), so a
staircase under a two-high ceiling works as it does in the game; down any drop;
up ladders, vines and water (`CLIMBABLE_IDS`). Fences and walls stop it.

Three decisions that moved the real-house numbers, in order:

1. **Seed from exterior air, not from the ground ring.** Seeding on the ground
   around the footprint gave 67% of real houses enterable. The failures were the
   site, not the house: a crop that cut away the hill (Gothic Brewery sits on a
   four-block terrace), a yard wall whose gate is outside the box (Stone Lodge,
   Small White Suburban House), an outside stair that starts in mid-air. Seeding
   from every standing cell in exterior air (not interior, not an open aperture)
   tests the building and lifted the rate to 84%.
2. **Stand the build on ground.** The bench's interior recipe sees a floorless
   ground-floor room as open through the bottom of the box. `interior_masks(occ,
   ground=True)` prepends a solid layer; the module uses it throughout, the
   `ground=False` path stays pinned to `geometry.interior_volumes` by test.
3. **Start in doorways that hold a door block too.** 3D-Craft crops often leave
   the front door two blocks up a sheer foundation with nothing to stand on
   outside it. A door is where the builder meant you to come in; seeding on door
   blocks (not on empty doorways) took the rate to 95%. The remaining 5% are
   open structures with next to no standing room and one tower whose ladder stops
   four blocks short of its room.

Two probes make the measure falsifiable. `seal_openings` bricks up every doorway
and window and turns every door, trapdoor and gate to wall (interior volume
unchanged, enterable 0.95 -> 0.02). `lower_ceilings` fills the top layer of every
room (exterior untouched, headroom3 0.94 -> 0.56, two-high rooms vanish). The
bench's `hollow` leaves the walk alone, as it must.

Known limits, recorded rather than fixed: the 3x3x3 closing in the interior
recipe eats cavities thinner than three blocks when they have an opening, so a
two-high room with a door is no room under either measure; exterior-air seeding
lets a balcony door count as a way in; `entrances` counts clusters of entry cells,
so a wide open ground floor counts many.

Script `scripts/bench_traversal.py` (pool, probes, every cached arm, walkthrough
GIFs with a quadrant cut: lime floor walked, light blue floor still to reach,
red stand-in at the wavefront). Results in results.md T63. Tests:
`tests/eval/test_traversal.py` (22). Nothing in `blockgen/eval/bench` changed.

## §53. Traversability wired into the bench: room scalars in the descriptor, a traversal pillar in BlockScore (2026-10-23)

Requested after §52. Three places changed in `blockgen/eval/bench`, which had
been frozen since the paper draft, plus the protocol.

1. **Descriptor** (`geometry.GEOMETRY_METRICS`, 77 -> 80 dims). Three scalars
   from the *blind* player (`traversal.walk_scalars`): `walk_standing`, the share
   of interior air a two-block body can stand in; `walk_floors`, storeys with a
   floor patch; `walk_headroom3`, the share of standing cells with a third free
   block above. Occupancy only, so the descriptor keeps its exact D4 invariance
   and exact blindness to material (both still tested bit-for-bit, and the
   geometry ladder re-validated on the new key `geom__patch2_orb55_t5_h8_s12`:
   every gate PASS, material shuffle / monochrome / rotations exactly at the real
   baseline). Reachability is deliberately *not* in the descriptor. The first
   draft put the blind walk's reach in, and an occupancy-only walk reads 70% of
   real houses as locked because doors are blocks; a generator that leaves its
   doorways open would then have been penalised for being more enterable than the
   corpus. Room goes in the descriptor, route goes in the pillar.
2. **Scorecard block and pillar.** Every FAST card carries `traversal`
   (`interior_standing_reached`, `enterable`, `floors_reached`, `headroom3`
   against the real distribution, same shape as `geometry_scalars`), and
   `composite.STRUCTURE_BLOCKS` includes it, so it is an RMS pillar in real-spread
   units and enters the max. `validate()` now accepts `traversal` as a tier that
   can see material shuffle (shuffled doors end up in the roof). Markdown cards
   render `geometry_scalars` and `traversal` beside `coherence`; the lab names
   `traversal` as missing on older cards.
3. **Protocol.** `Protocol.descriptor` pins the descriptor key; `houses32-v1`
   pins `patch2_orb55_t5_h8_s12`. Without it the board's best-per-model selection
   kept the 77-dim BlockScores (136.7 for `canon_oriented`) over the new ones
   (141.8). New cards stamp `context.geom_descriptor`; the first re-score card
   predates the stamp and is recognised by the `walk_standing` key in its arms.
   Every earlier card is unranked with the reason "re-score with the current
   bench". The protocol module's one-protocol rule is kept; the id did not change,
   its meaning did, and the doc says so.

Re-score: `run_20261023_011035_bench_traversal_pillar_leaderboard_re-score`
(`--tier both --n 128 --baselines`, the five leaderboard arms plus the three T51
agentic arms; self-validation 6/6). Results in results.md T64. Tests: 410 pass
across eval and lab (`tests/eval/test_traversal.py` 27, new cases in
test_geometry, test_composite, test_fast, test_protocol; standings and card
fixtures stamped as current cards).

What the pillar can and cannot do on this board: the KID pillars are in
noise-floor units and run to hundreds, the scalar blocks are RMS of per-build
W1 in real-spread units and run to a few. The traversal pillar therefore ranks
arms among the scalar blocks (`llm_control` 2.6, `dream` 1.3, `classic-matched`
0.8) but never becomes the worst pillar while `geom_kid` sits at 114 to 192. What
moved the ranking was the descriptor: `geom_kid` rose for the LLM arms (15.0 ->
16.9, 15.9 -> 17.4) and fell for the agentic arms (17.7 -> 15.7, 19.6 -> 17.9,
19.9 -> 18.4), and `dream-caption` now ranks above `llm_control`.

## §54. Facades: the build as six face images, and the 2.5D ceiling (2026-10-23)

The question that started this was whether rendering the six faces of a build,
instead of the four 45-degree views, would read the outer shape better. With the
textured renderer it would read it worse: the camera is orthographic and the
shading is flat, so face-on the side faces of every protrusion are edge-on and
vanish, and a flush wall and a wall with one-block relief render pixel for pixel
the same. The oblique views are the only reason relief is visible in RGB at all.
What the question was really after is a geometry signal, and it is cheaper to
take it straight from the voxels: a depth map per face.

A scratch probe (medians, n 120 to 200, no intervals) on axis-aligned depth maps
put the two failure families on opposite sides of real on one axis. Real facades
have 63% of neighbouring columns flush, 24% at one-block relief, 7% at jumps of
three or more; the LLM arms halve the one-block relief and more than double the
deep jumps (sticks and gaps) and their roofs are flat slabs; the agentic arms
are too smooth everywhere (dream-caption 80% flush, 2% deep jumps, mean roof
relief 0.40 against real 0.81). Not yet a bench quantity; candidate ladder rungs
when the descriptor next moves.

The representation that came out of it is `blockgen/corpus/facade.py`. A face
image looks along one axis and stores, per column, the depth to the first block
and that block's id and data, `-1` where see-through. Faces are named by axis
and end (`-z`, `+x`, `+z`, `-x` are the sides in walking order, `+y` the roof
plan, `-y` the underside); every image is indexed `(u, v)` in increasing
structure coordinates, so adjacent faces agree on shared edges and a rotation
about y relabels faces and flips `u`, nothing more (tested). `lift` places a set
of faces into one canvas, later faces overwriting earlier where generated faces
disagree; `shell` is `lift(project_all(crop))` and is a fixed point of itself
and a subset of the bench's `hollow` probe (tested, `tests/corpus/test_facade.py`,
8 cases). Purpose: a facade is a 2D grid of `(depth, id)` tokens, the format a
sequence model lays out reliably and the one the LLM arms never got to try; a set
of facades lifts into the 32^3 canvas the benchmark scores, so a facade model is
an arm on the same board, not a side project.

Before any model, the ceiling. `scripts/facade_oracle.py` reduces every
test-split house to its visible shell and writes three control rungs:
`real@facades4` (the four sides; median 56% of blocks kept), `real@facades5`
(sides plus roof plan; 65%) and `real@facades6` (every face, so the floor plate
returns). With four sides a flat roof disappears, because no side ray reaches
its upper surface; the roof plan restores it; without the underside the floor's
interior cells are gone and every room opens downward. Nothing inside survives
in any of them. These are what a facade-only generator can at best hand back,
scored with `--tier both` beside the usual controls. A first run left the
controls at n=128 against oracle arms of 399 (the runner's `--n` subsamples
controls only), so the reported run uses `--n 399` and every arm has the same n.
Results in results.md T65. The generator path, when it comes, is the existing
`blockgen.train` stack with a facade token kind in `blockgen/corpus/tokenizer.py`
and all four sides of every house as independent samples (decided 2026-10-23).

## §55. The facade token kind, its baselines and the first facade model (2026-10-26)

The 2D-first generator (decided 2026-10-23: representation first, depth plus
block per column, the existing trainer, all sides of every house as samples) is
now a token kind. `CorpusTokenizer.default(kind="facades")` adds two tag groups
to the shared vocabulary, `face` (the six faces) and `cell` (`empty`), 514
tokens in all, and writes a face as `[BOS, src, q, size, win, face, U, V, cell*,
EOS]`: `U` and `V` are coordinate tokens, a cell is `cell:empty` where the column
is see-through and `(depth, block)` otherwise, with depth a coordinate token, in
scan order (v outer, u inner). The voxel tokenizer's hash is untouched
(`60684456e598`, tested), because the kind is only written into the artifact when
it is not the default. `python -m blockgen.corpus.shards build <tier> --kind
facades --faces all` writes one sequence per face into its own hash directory
(`a9546c1f2f94`) and `find` now tells kinds apart by `spec.json`, so a voxel run
cannot pick up a facade shard by being newest; `TrainConfig.kind` names the kind
and the trainer refuses a mismatch. The houses tier sharded to 12,684 faces and
5.2M tokens (median face a few hundred tokens); the holdout to 5,172.

Roof and underside are in the shard too, not just the four sides, because the
format is the same and the face tag tells the model which it is; without them a
sampled build has no roof and no floor, which T65 priced at 200 BlockScore
points each. `--faces sides` keeps the four-sides variant available.

**Sampling a build from a facade model.** A build is six sequences. The sampler
draws a real bounding box from the training shard's meta (the `-z` rows carry
`u`, `v`, `extent`), prompts the model with each face's tag and size, generates
the six faces in one batch and lifts them into the box
(`blockgen.train.sample.facade_builds`). The model is never asked to agree with
itself on size, only on content, and disagreement between faces at a shared
voxel is settled by face order. Every run also writes `facades.png`, the face
sheets of its first eight samples. `python -m blockgen.train --sample <run_dir>
--n 399` samples a finished run into an arm cache.

**Baselines for the facade track.** `blockgen/corpus/facade_baselines.py`,
fitted on the train split only and kept out of `bench.baselines` so that
`--baselines` keeps its meaning: `facade_unigram` (every cell from the train
cell distribution of that face type: the floor), `facade_markov` (every cell
given the cells to its left and below, backing off to the unigram: the smallest
model that knows a window has a sill) and `facade_patchwork` (every face a real
train face of that type from a different house, resampled to the box: the
facade twin of `patchwork`). Each lifts into a real box.
`scripts/facade_baselines.py` writes them as arms and runs the bench beside
`real@facades6`, the ceiling. On a sheet the three read as expected: noise,
horizontal streaks, real walls that do not belong together.

**First model.** `configs/train/houses32-facades.yaml`, the 5M preset, 8,000
steps on the facade shard with dropout 0.1 and `select=best`; the smoke config
verified the path end to end first (60 steps, samples lifted, fast tier scored).
Results in results.md T66.

## §56. Faces conditioned on each other: the joint layout (2026-10-26)

The first facade model (§55) sampled each face of a build on its own, so the
six faces of one house disagreed on everything the box does not fix: one wall
oak, the next cyan, a roof that belongs to neither. The fix is the layout, not
the model. `CorpusTokenizer.default(kind="facades", layout="joint")` writes every
face of a build into one sequence, `[BOS, tags, (face, U, V, cell*)+, EOS]`, in
the shard's face order: the four sides in walking order, so each wall is written
in sight of the one it shares a corner with, then the roof plan, then the
underside. The vocabulary is unchanged; the layout is written into the artifact
only when it is `joint`, so the two layouts have different hashes
(`a9546c1f2f94` per face, `a4ca45ac1fe5` joint) and different shards, and both
configs now pin their hash rather than trusting "newest". The houses tier is
2,114 joint sequences, median 2,380 tokens, none over the 8,200 context.

**Sampling with forced headers.** `blockgen.train.sample.generate_faces` keeps
the KV-cached batch loop and adds one rule per row: the model writes cells, and
every face header (tag, U, V) is forced. When a face has received its `U*V`
cells, or the model ends it early with an EOS or a face tag of its own, the next
header is injected and generation continues; after the last face an EOS is
forced. So the six faces of a sample always describe one real box, drawn from
the shard's meta as before, and the model's only job is content. `decode_faces`
splits a sequence at its face tags, and `decode_face` is its first face, so the
per-face layout decodes through the same code. Tested with a babbling stand-in
model that emits random cells and the odd early EOS: every plan comes back with
its faces in order and at size.

The per-face model stays as the ablation: same data, same tokenizer vocabulary,
same box sampling, faces independent. Results for both, beside the facade
baselines and the six-face ceiling, in results.md T66.

## §57. The joint facade model at 20M: longer and bigger is not the lever (2026-10-28)

`configs/train/houses32-facades-joint-20m.yaml` extends the 5M joint config with the
20m preset (d_model 512, 8 layers) and three times the steps, 24,000, since the 5M
model's holdout NLL was still falling at its last eval. Run
`outputs/run_20261028_110415_houses32_facades_joint_20m`, 54 minutes, 734M tokens.

The holdout curve answers the question in the first quarter of the run: 0.97 at step
2,000 (the 5M model read 1.25 there), 0.81 at step 5,000, a minimum of 0.806 at step
6,000, then a climb without pause to 1.57 at step 23,000 while the train loss fell to
0.05. The joint shard holds 2,114 sequences, about 5M tokens an epoch, so step 6,000 is
already 35 epochs and the 20M model has memorised the training houses by then. Its best
checkpoint is slightly worse than the 5M model's final one (0.806 against 0.783), and
`select: best` samples and scores step 6,000. Fast bench at n=64: BlockScore 190.0
against the 5M joint model's 189; the n=399 card is T68.

Two consequences. First, the facade track is data-bound, not capacity-bound: a bigger
model converges faster to the same floor and then overfits, which is the regime where
pretraining or augmentation pays and more parameters do not. Second, the representation
has a free augmentation that the voxel kind does not: a D4 rotation of a house is a
relabelling of its faces plus a flip of `u` (facade.py docstring; the rotation test in
`tests/corpus/test_facade.py` is the proof), so the eight symmetries of the square turn
2,114 sequences into 16,912 at zero cost and with no new blocks invented. Mirrors are
not quite free semantically (text on signs, asymmetric stairs) but houses in this
corpus are not oriented, and the `oriented` tag already records that the canvas has no
preferred facing. That is the next facade run, before any bigger preset.

## §58. The volumetric row: standard point-cloud metrics on the ladder (2026-10-28)

The objection a 3D reviewer raises first against the geometry tier (research.md I.3,
item 1) is that nobody in shape generation would score voxel builds with an image
metric; they would sample points and run MMD, coverage and 1-NN accuracy under chamfer
distance (Achlioptas 2018, PointFlow 2019), and those see inside a shape if the points
do. The draft answers this in a paragraph (Appendix K: exterior-surface samples are
exactly invariant under Fill-Interior). `blockgen/eval/bench/pointcloud.py` turns the
paragraph into a ladder row by running exactly that suite through `ladder.score_rungs`,
the same rungs, gates and noise floor as every other metric.

Three pairwise distances over every structure the ladder scores (reference, nineteen
rungs, null pool; 1,709 builds at n=64): chamfer on 1,024 points at the centres of all
occupied voxels (`cd_volume`), chamfer on 1,024 points at voxels with any exposed face,
cavity walls included (`cd_surface`, the leaky variant Appendix K says does see the
probe per pair), and 1 - IoU of the occupancy grids placed corner to corner (`iou`).
Points are centred on the bounding box and divided by the canvas side, no per-shape
normalisation, so size stays visible. Per distance: MMD (mean over reference shapes of
the nearest probe), 1 - coverage, and leave-one-out 1-NN accuracy against an equal-size
reference draw fixed once per run. Chamfer matrices on the GPU beside the training run,
a few minutes in all; the module has four tests. Result file
`outputs/analysis/ladder/pointcloud__k1024_canvas32.json`, written up as T67. The
reading in one line: the standard suite resolves decimation and a 40 percent deletion,
is within two noise floors of real on Fill-Interior, Hollow, occupancy noise and column
jitter, and every one of its nine statistics fails a blocking gate. Not yet included:
EMD (needs an assignment per pair, 2.9M pairs), a learned 3D encoder (Uni3D or
PointBERT KID), and n=128 (running).

## §59. D4 augmentation of the facade shard (2026-10-28)

§57 said the facade track is data-bound and that the representation has a free
augmentation. Implemented as a shard variant rather than a loader option, so a run's
data is still one immutable directory: `python -m blockgen.corpus.shards build
sft-houses32 --kind facades --layout joint --augment d4` writes
`shards/sft-houses32/a4ca45ac1fe5-d4/`, beside the plain shard of the same tokenizer,
with `augment: d4` in its spec. `_encode_one` walks the eight D4 elements of each volume
through `blockgen.utils.augment._d4_arrays` (which already remaps stair and log facings),
drops exact duplicates so a symmetric house is not over-represented, and tags each
sequence `aug` 0 to 7 with 0 the identity. `find(tier, hash, kind, augment)` resolves
the suffix; `TrainConfig.augment` is applied to the training tiers only, so
`eval-holdout` is never augmented and the holdout NLL stays comparable with every
earlier facade run. One test in `tests/corpus/test_facade_tokens.py` checks that the
eight sequences lift to the eight D4 images of the original occupancy and that the
identity comes first. The sampler still reads its boxes from the plain shard, which has
the same box distribution up to swapping X and Z.

The houses shard: 2,114 joint sequences become 16,880 (32 collapsed), 40.7M tokens,
median length 2,384, built in seven seconds. `configs/train/houses32-facades-joint-d4.yaml`
is the 5M joint config on it at 16,000 steps (524M tokens, 13 epochs of the augmented
shard against the 46 the plain 5M run saw of the plain one). Run and card in T68.

The 5M run on the augmented shard (`outputs/run_20261028_123627_houses32_facades_joint_d4_5m`,
16,000 steps, 14 minutes) reaches a holdout NLL of 0.589, still falling at the last eval
(0.603, 0.598, 0.592, 0.589), against 0.783 for the plain 5M joint run and 0.806 for the
plain 20M. The same model with eight symmetries of the same houses is 0.19 nats better
on unaugmented held-out houses: the gap the 20M run showed was a data gap. Next, in
parallel with its card: the 20M preset on the augmented shard
(`configs/train/houses32-facades-joint-d4-20m.yaml`), 24,000 steps.

The 20M preset on the augmented shard (`outputs/run_20261028_125240_houses32_facades_joint_d4_20m`,
24,000 steps, 73 minutes beside a bench card) reaches 0.479 at step 18,000 and then sits
on a plateau between 0.48 and 0.49 to the end, no climb: eight times the data turned the
20M model from a memoriser at step 6,000 into a model that is still fitting at step
18,000. The facade likelihood ladder now reads 0.783 (5M plain), 0.806 (20M plain), 0.589
(5M D4), 0.479 (20M D4). Card with all five models in T68.

The card (T68) does not follow the likelihood ladder. Plain 20M 1035 < D4 20M 1082 < D4 5M
1113 < plain 5M 1128, every one decided by `geom_kid`, on which the D4 models gain nothing
(the two 20M models share a Holm group). D4 buys everything else: `mv_dino_kid` 0.468,
the only learned value below patchwork; enterable 0.23 to 0.35; memorization percentile
0.48 at held-out real where the plain models sit at 0.61 to 0.65; components 60 against 63
to 75. Reading: the geometry pillar for a facade arm is held by faces that do not meet at
shared edges and by the empty interior, and more data on the same representation does not
touch either. Next levers are an edge-consistency constraint at decode time and the
interior second stage, not training.

## §60. Where the pieces come from: face sheets against real, and the temperature (2026-10-28)

The user's goal for the facade track is solid builds; the learned arms lift into 60 to 75
coherence components where the ceiling lifts into 13. `scripts/facade_sheets.py` draws
the six-face sheet of eight builds per arm, real test houses in the same format as the
generated ones, into `outputs/analysis/facade_sheets/` (`real_vs_generated.png` puts real,
20M D4 and 20M plain side by side), and splits the fragmentation into two measurable causes:
within a face (2D components of the face mask, holes inside the filled hull, lone cells,
and the relief jump rate, the fraction of adjacent filled cells whose depths differ by two
or more) and in 3D (26-connected components, largest-component fraction, dust pieces under
five blocks). Faces are re-projected from the lifted shell, so for patchwork and the
generated arms they are the shell's faces, not the drawn ones.

The sheet is unambiguous. Real faces are structure: repeated windows, bands, a symmetric
roof plan, one depth per wall with a few steps. Generated faces at the training
temperature (1.0, top-k 40) are texture: the right palette and silhouette, speckled depth
everywhere, no windows. The numbers say the same. Face masks are nearly as connected as
real (1.32 components per face against 1.22, holes 2 percent against 1 percent) but the
relief jumps three times as often (0.29 against 0.10), and in 3D the shell is 6.8
26-connected pieces against 1.7 real, with 1 percent of blocks in dust. Against the bench's
60 components (6-connectivity), 6.8 at 26-connectivity means most of the "pieces" touch at
an edge or corner: the model draws neighbouring cells at depths that differ by one or two,
so they share no face. The fragmentation is relief noise inside the faces, not faces that
miss each other at the corners.

Temperature is a large part of it. Resampling the 20M D4 model at 0.7 and 0.5 (64 builds
each, `python -m blockgen.train --sample <run> --n 64 --out-dir ... temperature=0.5`):

| temperature | jump rate | 2D comps/face | 3D comps (26) | lcc | dust pieces | empty builds |
|---|--:|--:|--:|--:|--:|--:|
| real | 0.097 | 1.22 | 1.7 | 0.987 | 0.6 | 0 |
| 1.0 | 0.288 | 1.32 | 6.8 | 0.971 | 4.9 | 0 of 399 |
| 0.7 | 0.229 | 1.65 | 4.7 | 0.945 | 3.1 | 1 of 64 |
| 0.5 | 0.151 | 1.12 | 2.1 | 0.956 | 0.8 | 13 of 64 |

At 0.5 the surviving builds are nearly as solid as real on every face statistic and the
sheet shows houses with windows and a roof plan a reader would accept, but a fifth of the
draws are empty and some of the rest are a flat plate, which is the mode of the training
distribution (ground). The model has learned solid structure; it is not what it samples
at temperature 1. Three things follow. The bench's facade cards were all sampled at 1.0,
so the learned arms were scored on their noisiest decode and a temperature sweep belongs
on the card before any training lever. The relief jump rate is the solidity measure the
user asked for and is a candidate rung (§54 relief scalars). And the roof-plan-only model
(the user's suggestion, a `top` face set in `shards.py` plus a per-face run) is the clean
way to watch it: one face per sample, viewable as an image, with the same statistics.

## §61. BrickNet's recipe on the Minecraft corpus: pipeline and pilot launch (2026-10-26 to 2026-10-30)

The question from the user: what does it take to retrain BrickNet on our tokenized
Minecraft sets? The short answer is that the tokenized shards cannot be fed to their
trainer, because BrickNet trains Qwen3 on text, not on a block vocabulary. What transfers
is the recipe (Qwen3 + LoRA r32 on spanning-tree walks of a typed connectivity graph) and
the evaluation (actions before the first grammar failure or collision). So the shards are
decoded back to structures and re-rendered as BrickNet-style walk text, and their
`scripts/train.py` runs on that.

**Grammar.** A Minecraft block is BrickNet's degenerate part: one "face" connector per
side, no rotation. A build becomes a 6-connectivity graph and a walk is a random spanning
tree of one component: a random root, then repeatedly a uniformly random open edge from
the placed set to an unplaced neighbour. Text is one node line and one edge line per block,
`<label> <block name>[ | <data>]` and `<parent label> face <direction>`, labels decorative
and identity positional exactly as in `bricknet.tree`. The data value appears only when
non-zero; 32 names (stained clay, stained glass, slabs, stone variants) need it to pick the
variant, the rest do not, and the suffix costs two tokens on every block. Measured cost on
the holdout houses: 9.4 Qwen3 tokens per block (11.4 with the suffix always on; the
BrickGPT `name x y z` line is 11.7). The parser mirrors theirs: a grammar error stops the
walk, a collision (two nodes on one cell) is counted and the walk continues. Round-trip on
2,500 walks is exact and the realised holdout houses render as the originals
(`outputs/figures/bricknet_repro/mc_val_truth_20.png`). Code:
`libs/BrickNet/repro/mc_to_bricknet.py` (build, roundtrip, score) and `repro/mc_eval.py`
(score + realise + textured grid + npz for BlockScore).

**Data.** `data/minecraft/bricknet/`. Val: the 494 eval-holdout houses with at most 800
blocks (of 798), four walks each paired with the four houses_32 captions by sample_index,
largest component only (1,976 walks, 931k blocks). SFT: the same cut of sft-houses32, 927
of 1,863 houses, 3,708 walks. PT: one walk per component of a 320k-sequence random subset
of pretrain-clean capped at 600 blocks; 1.32M walks, of which 524,542 components of at
least 16 blocks are the training pool (64M blocks, about 600M tokens, median 59 blocks and
90th percentile 344 per walk). The 800-block cap on houses is the SFT context: at 9.4
tokens per block their 2,816-token SFT window holds a 300-block build, so SFT here will run
at about 8k tokens. The 600-block PT cap keeps a walk under the 6,400-token PT window.

**Trainer on one 16 GB card.** `scripts/train.py` is patched (original kept as
`train.py.orig`) with `--grad_accum`, `--max_len`, `--train_iters`, `--warmup_iters`,
`--eval_interval`, `--global_batch_size`, `--lr`, `--no_compile`, `--grad_ckpt`,
`--val_samples`, all defaulting to the recipe. liger-kernel 0.8 refuses transformers 4.57,
so 0.6.5 is pinned. Smoke tests at max_len 6400 on the val walks: micro-batch 1 without
checkpointing peaks at 11.7 GiB and runs at 0.53 s per sequence (about 8k tokens/s);
gradient checkpointing lets micro-batch 2 fit in 3.7 GiB but costs 0.79 s per sequence;
torch.compile recompiled on every padded width (16 steps in 374 s) and is off. The paper's
250k iterations at batch 32 are weeks at this rate; the pilot is 6,000 iterations at the
recipe's batch 32 (grad_accum 32), warmup 300, eval every 500 on 256 val walks, about 192k
walks or 37% of the pool. Launched 2026-10-30 23:10 as `repro/runs/mc_pt_base`
(`repro/run_mc_pt.sh mc_pt_base 6000`). The second arm, same data and budget but
initialised from the public LEGO adapter (`--pt_adapter kulits/BrickNet-0.6B-PT`), is the
transfer test from §50 and follows when the card frees up; the two cannot share it.

**What the pilot reads out.** Trainer eval loss on the val walks is the Table 4 per-token
NLL; `mc_eval.py` on 256 unconditional 100-block samples (nucleus 20/0.95 and ancestral,
`--stop_after_newlines 199`) is Table 2; the realised volumes go through BlockScore next to
arms A to D. The honest comparison for the LEGO-init arm is the base-init arm at equal
steps, not BrickNet's LEGO numbers.

**Pilot result (2026-11-02, T69).** Val NLL 0.677 per token at 6,000 steps and still
falling; nucleus samples 99% parsable with 53 actions before the first collision, but the
collision-free set is half straight lines and the rest are single-material blobs. Two
readings. First, the random-spanning-tree walk makes a Minecraft build a long-range
problem: a house's wall is a run of identical blocks, and a random tree walks through it
in an order that carries no plan, whereas a LEGO part's name already encodes a shape. The
measure that would separate "learned the walk syntax" from "learned buildings" is the
BlockScore of the realised volumes, which the npz beside each grid now allows; it was not
run because the samples are visibly not buildings. Second, the pool is the wrong
distribution for unconditional readouts: 524k fragments of pretrain-clean windows, median
59 blocks, mostly ground and terrain slabs, so the unconditional mode is a slab or a line.
The SFT stage on captioned whole houses is the stage that matters for us, and the pilot
was run so that stage starts from a walk-literate adapter. The LEGO-init twin runs next;
if it reaches the same NLL in fewer steps the connector-token pretraining transfers.

**Twin and SFT (2026-11-03 to 11-05, T69).** It does not: the LEGO-init run is 0.01 nats
behind at every checkpoint and no better at sampling. SFT on the 3,708 captioned house
walks takes the base arm to 0.649 and produces compact, house-sized, house-palette
samples that end on their own, but they are clumps with a median of 22 cell revisits
each. The walk grammar lets a block be placed onto an occupied cell (BrickNet's
collision), and nothing in 16k SFT pairs teaches the model to track occupancy across a
200-step random tree. That is the same lesson as BrickNet's own 16-step collision-free
horizon at 0.6B, made worse by walks that are three times longer and carry no part-shape
cue. Two lessons for the paper. The LLM-on-walks recipe is a validity story only with a
rollback or occupancy mask at decode time, which BrickNet declines to use and which our
raster arms do not need. And transfer from LEGO connector pretraining to Minecraft face
walks is null at this budget, which bounds what a shared typed-connection prior across
media (§E.2) can be expected to give for free. One process lesson: an adapter trained on
a merged model must be evaluated on the same stack; two sessions caught that a day late.

**Clean LEGO-init SFT arm (2026-11-05).** The confounded LEGO-init SFT was rerun with
both parent adapters merged (`mc_sft_legoinit2`, T69). It lands at eval loss 0.654 against
the base arm's 0.649, the same 0.005 gap as after pretraining, and its conditioned
samples match the base arm on every readout metric within noise (93.8% vs 91.8% parsable,
2.0% vs 1.2% collision-free, Coll 31 vs 32, same clump shapes). The transfer question is
closed as null at this budget, now on both the loss and the samples. The chain waiter for
this run also hung: it was launched with a relative log path from a shell whose working
directory later changed, so `grep -q` on a missing file never matched. Waiters should
take absolute paths.

## §62. The pipeline made pluggable (2026-10-30)

The question from the user: where does a new model plug into the pipeline, and why has no
`blockgen.train` run ever reached the leaderboard? The code is the record; this section
is what changed and why.

**What the audit found.** One architecture: `build_model` in `blockgen/train/trainer.py`
raised for any `arch` but `ar2`, and the loss, the collate, the sampler and the bench
track were AR-specific in four files. The model contract was written down nowhere:
`forward(input_ids, pad_mask=None)` for training plus `forward(step, past=..., return_cache=True)`
and `.max_seq_len` for the sampler, which is why `pe=alibi` trained and then crashed at
its first sampling call. Trainer cards were invisible: the trainer writes
`<run>/bench/scorecard.json` and `tools/lab/catalog.py` globbed `outputs/*/scorecard.json`
one level deep, so at the 2026-10-30 audit the lab saw the 35 top-level cards and none of
the 27 nested trainer cards (the count grows with every smoke run: 36 and 38 on
2026-10-31).
Equal n was enforced nowhere: `houses32-v1` checks `min_n` 128 per arm
(`blockgen/eval/bench/protocol.py`) and the board ranked `canon_oriented` at n=128 beside
five facade arms at n=399. There was no post-training module; `stage: posttrain` was a
label in `blockgen/train/config.py`. The whole new pipeline was untracked in git
(`blockgen/corpus`, `blockgen/train`, `configs/corpus`, `configs/train`, `tests/corpus`,
`tests/train`, all `??`), so the `git_sha` every `run.json` records names a commit that
contains none of the code that ran. And a bug in the frozen bench: `runner.py` sets
`ctx.derived["geom_descriptor"]` at line 303 and replaces `ctx.derived` wholesale at
line 364, so no card on disk carries the descriptor stamp; `protocol.run_descriptor`
reads the descriptor by presence of `walk_standing` in the scalars instead. The bench is
frozen, so the fallback stays the read and the bug stays recorded. Two smaller findings
went straight into results.md: the holdout NLL every T58 to T60 table quotes is over the
first 256 fitting sequences (`eval_seqs: 256` in `configs/train/base.yaml`), not 736, and
the board's floor is 128, not 256.

**What landed.** The registry: `blockgen/train/archs.py` defines
`Arch(name, build, loss, sample, track)`, `ARCHS`, `register` and `get`, with `ar2`
registered behaviour-preserving; the trainer calls `archs.get(cfg.arch)` for build and
loss and `sample_and_render` draws through `Arch.sample`. A model that wants the trainer
implements `forward(input_ids, pad_mask=None) -> logits`; `supports_cache` in
`blockgen/train/sample.py` decides up front whether the KV-cache path or a full re-forward
drives decoding, so an architecture without `past` samples, and so does `pe=alibi`.
`sample_structures(model, tokenizer, cfg, n, seed)` in the same file is the one sampling
entry the rewards use, and the structure-cache report now carries `model`, `arch`,
`kind`, `checkpoint {file, step}`, `trained_on_signals` and `train_split`. New
`TrainConfig` fields in `blockgen/train/config.py`: `model` (constructor kwargs for the
arch; unknown YAML keys now raise), `train_split`, `trained_on_signals`, `final_eval_all`
(default on: the last pass scores every holdout sequence that fits, recorded as
`val_nll_all` and `eval_seqs` in the registry line), `bench_tier`, `bench_n` 128 for a
bare config while `base.yaml` keeps 64, and `allow_partial_init` (`init_from` refuses a key
mismatch by default and records `init_missing`/`init_unexpected`). On the corpus side,
`from_structures` and the `from-npz` subcommand in `blockgen/corpus/shards.py` turn a
sample npz into a derived tier (`tiers/<tier>.json` with `kind: derived`, the five-file
shard under `shards/<tier>/<hash>/`), raise and leave nothing behind when every structure
is skipped, and `StaleShardWarning` fires when a shard's `tier_sha256` no longer matches
its tier json. `EmptyTierError` in `blockgen/corpus/tiers.py` stops a split-backed tier
from materialising to zero rows, which `manifest curated` used to make easy by omitting
`houses_32` from its default caches (`blockgen/corpus/manifest.py` now defaults to
`all_32 houses_32 houses_48`). `scripts/corpus_refresh.sh` is the driver the layer never
had, manifest build to shards in order, validated with `--dry-run` only. The lab:
`SCORECARD_GLOBS` in `tools/lab/catalog.py` looks two levels deep, a trainer card has the
id `<run>/bench` and is attributed to its training run from `run.json`, `summary.json`
and `outputs/registry.jsonl`; `tools/lab/standings.py` carries `n` on every row with
`n_groups` and `default_n`, and `/leaderboard` opens on one n with a chip per group and a
banner when a view mixes n. The filter hides, it never unranks; the protocol is untouched.
Post-training: `blockgen/posttrain/` with `rewards.py` (`face_solidity`, `lcc26`,
`novelty_penalty`, weighted composites; the six pillar names exist as placeholders that
refuse to score), `bon.py` (best-of-N arm with `scores.parquet`, prints its `--arms`
string), `rft.py` (keeps the top fraction, writes the `rft-<name>-<reward>` tier through
`from_structures`, prints the `blockgen.train` command), `select.py` (`select=reward`
registered with the trainer on package import), and the two recipes
`configs/train/rft-voxel-5m.yaml` and `configs/train/rft-facades-joint-d4-20m.yaml`. The
rule every rung is held to, from `blockgen/posttrain/__init__.py`:

> A post-trained arm may be rewarded only on signals that are not BlockScore pillar
> inputs. The trained-on signal is reported as a method-internal metric, never as a
> BlockScore gain. Evaluation is the full BlockScore at n equal to the control n under
> houses32-v1 with the real@ controls, and claims are made only for pillars the reward
> did not touch. Because 26-connected connectivity correlates with the 6-connected
> coherence pillar, movement on coherence is pre-registered as expected transfer, not
> independent evidence.

`rewards.get` enforces the first sentence and `trained_on_signals` travels with the
samples, so the lab flags the arm the way it flags oracle arms. The face statistics of
§60 moved out of the script into `blockgen/eval/facade_stats.py` so a reward and the
sheets score the same numbers; `scripts/facade_sheets.py` imports them.

**Tests and the smoke.** `tests/corpus`, `tests/train`, `tests/lab` and `tests/posttrain`
pass together, 400 tests (37, 24, 313, 26), and `tests/eval` passes its 510; the GPU
smoke test is marked slow and runs under a scratch `outputs/` and registry. The CLI
smoke, `outputs/run_20261031_024716_smoke`: 60 steps, holdout NLL 3.3005 on the
256-sequence subset and 3.3869 over all 736 fitting sequences, 16 samples, a fast card at
n=16 (BlockScore 47.36, which at n=16 is not a number), registry line 40 with every new
field. `--sample --n 8` writes a report whose `checkpoint.step` is the real step (60);
the npz file name still carries the sample count (`samples_step8_48.npz`, named so at
`blockgen/train/__main__.py:32` so it never overwrites the trainer's own end-of-run cache
of the same step). `python -m blockgen.posttrain.bon --n 4 --k 2 --reward lcc26` drew 8
in 6 seconds and raised the mean from 0.2008 to 0.2402; `rft --n 8 --keep-frac 0.5` wrote
a four-sequence derived tier that `shards.find` resolved without a staleness warning and
printed a train command that `config.load` accepts; both artefacts were removed. A lab
on an ephemeral port saw 73 cards (the 2026-10-30 lab saw 35; every smoke run adds a
nested card), and listed the smoke
card under skipped with the protocol's own reasons (fast tier, n 16 under 128), which is
the right verdict. The user's lab on 8765 predates all of this and will show the trainer
cards only after a restart.

**World crops: what the sort still owes.** The gate (`blockgen/data/worldcrop/gate.py`)
decides built against terrain, and its positives are the curated caches, which is why
`q` is 4 for every curated row. The 192 hand-labelled crops (`gate/eval_labels.jsonl`,
research.md §F and docs/worldcrop.md) are 106 built, 65 fragments, 8 terrain and 13
unsure (research.md §F carries an older labelling of the same file), so the larger noise
class is fragments, which the gate does not see. The plan,
not built: keep the terrain gate as it is, score fragments separately, and carry that
score as a quality sidecar merged by the tiers the way `dedup.parquet` and
`kinds.parquet` are, which widens the `q` tag that docs/pipeline.md §3.4 defines as the
built-against-terrain gate score only. Changing `q` changes every pretraining sequence, so it belongs
before the next big shard build, beside the 41 months (2009 to 2012, 2013 01/02/04/05,
2018) that have `crops.jsonl` but no manifest table.

**Decisions.** Taken: the pipeline's interpreter is the conda `python` (torch plus pandas;
`.venv` lacks pandas and the corpus CLIs need it). The lab glob was widened rather than
the card moved, so `summary.json` and registry paths did not change. Equal n is a view
rule in the lab, chips and a banner, and `houses32-v1` stays frozen at `min_n` 128 with
256 or 399 the quotable convention. The bench stays frozen, so the stamp bug is read
around, not patched. `init_from` is strict by default. `bench_n` defaults to 128 for a
bare config and every recorded config keeps the value it pinned. The honesty rule is
enforced in code, not prose. Best-of-N and rejection-sampling fine-tuning are the first
two rungs; GRPO, DPO and merge/distil are docstring only; both recipes are written and
nothing has been launched. Left to the user: commit the pipeline (every directory above
is still `??`, and until then no run is reproducible from a commit), and whether to
re-manifest the 41 unmanifested months, which changes every tier sha and orphans the
lineage of the shards and checkpoints on disk.

Two things the first post-training pass taught about the shared GPU (2026-10-31). The
sampler had no batch knob, so best-of-N on the 20M facade model at the default batch of
32 ran out of memory beside an 11 GB tenant; `TrainConfig.sample_batch` (and `--batch` on
`blockgen.posttrain.bon` and `rft`) now sets it, and the voxel sampler needs it too, since
voxel sequences run to 8,200 tokens and their cache is larger than the facade one. At batch
8 the facade model draws 0.16 builds per second, so best-of-4 at n=399 is 2.7 hours; plan
post-training draws in hours, not minutes. And `scripts/posttrain_first_pass.sh` is run from
a frozen copy: bash reads a script incrementally, and editing the file under a running copy
broke the first chain after its best-of-N step had finished.

First rung result (2026-10-31, T70): temperature 0.7 alone takes the 20M D4 facade model
from 1082 to 637 and below patchwork, solidify and shuffled materials, `geom_kid` 17.8
against the ceiling's 13.8; best-of-4 on `face_solidity` takes the raw decode to 871 and is
the lever that moves enterability (0.35 to 0.49) at a novelty cost (nearest-train IoU 0.30
to 0.37). §60's reading of the face sheets was right: the speckle was sampling noise, and
every facade card before this one scored the noisiest decode. Next: best-of-k at 0.7, and
the fine-tune retry at a quarter micro-batch.

Voxel half (T71): best-of-4 on `lcc26` takes `m2b_sft_s1` from 386 to 340 at n=256, `geom_kid`
18.6 to 16.3 and alone in its Holm group; the rejection-sampling fine-tune scored 395 because
`select=best` on holdout NLL picked step 500 of 4,000 (training on the model's own filtered
samples lowers holdout likelihood from the first eval), so the card scored a 500-step nudge.
Rerun under `select=reward` and with a larger kept set before judging rung 2.

Facade RFT (T72): 1082 to 994 at temperature 1.0, tied with patchwork on `geom_kid`, most
enterable facade arm on any card (0.53), memorization back at real; again `select=best` on NLL
chose step 500 of 4,000. Before calling rung 2 weaker than rung 1, rerun both RFT models with
`select=reward` and sample them at 0.7.

**Second pass (results T73, 2026-11-01): select=reward at temperature 0.7.** Facades: the
reward selector kept step 500 again (face_solidity 0.860 vs 0.845 at step 4,000), so the
new arm is the T72 weights at T=0.7, and it scores 553 against 637 for the raw model at the
same temperature and 493 for the facade ceiling; `geom_kid` 15.4 shares a Holm group with
the ceiling (13.8), components 17.6, enclosed air 0.048. The fine-tune and the temperature
compose. Voxels: the selector kept step 4,000 (lcc26 0.995), but the card's arm was drawn by
`--sample`, which defaulted to best.pt, so the 549 arm is step 500 at T=0.7 (fixed
2026-11-02: `--sample` now follows summary.json selected_ckpt). That arm fills (walls 1.45
blocks, surface-to-volume 1.08 vs real 2.06), worse than raw (386) and raw at T=0.7 (431);
temperature itself hurts voxels. A `--sample` that silently scores a different checkpoint
from the one the run selected is the class of bug the registry cannot catch. Lesson: selecting on the trained-on reward finds the
checkpoint that games it most, so on voxels the selector should use a held-out signal, and
lcc26 needs a counterweight to filling. Open cell: voxel step 4,000 at T=1.0.

**Step 4,000 scored (results T74, 2026-11-02).** The reward-selected step-4,000 voxel
fine-tune at temperature 1.0 scores 359: better than step 500 (395) and raw (386), behind
best-of-4 (340), none of the four separated after Holm at n=256. At 0.7 it scores 478, so
temperature 0.7 hurts every voxel model and helps every facade model; T73's "the selector
games lcc26 by filling" was that temperature effect on the wrong checkpoint. select=reward
picked correctly on both tracks.

## §63. The pooled-face probe: what each level of a fixed hierarchy keeps (2026-11-02)

Step 1 of the 2D next-scale plan (research.md H.7): before any model, pool real
faces to coarser grids and look. The user asked for the roof plan; the front face
ran beside it because the roof plan's silhouette turned out to be nearly a rectangle
and the front is where the silhouette carries information.

**What was built.** `facade.pool(f, factor, rule)` in `blockgen/corpus/facade.py`
reduces a face to ceil(U/f) x ceil(V/f) cells, depths still in voxels and the
extent unchanged, so the coarse face still describes the same box; this is the
level a next-scale model would see or be asked to draw. Three rules: `any`
(filled if any child is, nearest child depth, most common child block; dilating),
`majority` (filled if at least half the in-bounds children are, mode depth and
block; thin features vanish, T17's decimation by vote) and `center` (the centre
child, strided decimation, the T17 control). `facade.upsample` repeats a coarse
face back to a shape, the naive decode of a level. A first version counted the
padding cells that make a face a multiple of f as children, so a 20-wide face
pooled by 16 was judged against 12 empty cells and a third of roof plans went
empty under `majority`; the far-edge blocks now count only in-bounds children
(`tests/corpus/test_facade_pool.py`, 7 cases). `scripts/facade_pool_probe.py`
runs the test split through every rule at f = 2, 4, 8, 16, compares the
upsampled level with the original (mask IoU, precision, recall; depth exactness
and error on cells filled in both; block agreement) and reports the face
statistics of §60 on the upsampled level (relief jump rate, 8-connected
components, holes), then draws a sheet per rule: one house per row, levels as
columns, at the face's own size. Runs `run_20261102_163427_facade_pool_probe_roof`
and `run_20261102_163441_facade_pool_probe_front`, 399 faces each, seconds.
Numbers in results.md T75.

**Reading the sheets.** On the roof plan f = 2 is the real thing with the trim
softened; f = 4 keeps the ridge, the wings, a courtyard, an L-shaped footprint and
every material region; f = 8 keeps the footprint, the roof-against-ground split and
the dominant material; f = 16 is a material-coloured box. The roof plan's mask is
not informative: the silhouette from above is the footprint plus whatever ground
plate the build carries, filling a median 86 percent of its box, so the level's
content is depth (the slope and the ridge) and block. On the front face f = 2 keeps
windows as single cells, doors and the roof outline; at f = 4 the windows are gone
into the wall and what is left is storeys, roof line, ground band and the dominant
material; f = 8 is the wall, roof and ground bands; f = 16 a box. The one
see-through frame house in the sheet (row 3) is the thin-feature dilemma in one
row: `any` dilates it into a solid slab, `majority` keeps its posts to f = 8 and
loses it at 16.

**What this corrects in H.7.** The window rhythm does not live at 8 x 8 on these
faces; windows are one or two cells wide and survive only at f <= 2. The levels on
a 32-cell face read: f = 16 box and material, f = 8 bands, f = 4 storeys, roof line
and footprint shape, f = 2 openings and trim, f = 1 the relief. A residual
hierarchy would put the periodic structure at its two finest levels, which is also
where the within-level independence bites; that is the thing step 2 has to show it
can draw.

**What the numbers add.** `any` keeps every real cell (recall 1) and its mask IoU
on the front face falls only to 0.70 at f = 8, while `majority` and `center` lose
recall to 0.90 and 0.83 at f = 8; the three agree on depth and block within a few
points. The jump rate falls with pooling (0.115 real, 0.05 at f = 4, 0.02 at f = 8),
so a coarse level is smoother than a face, not noisier, and components go to 1.00:
every pooled level is one piece. Depth exactness is the quantity a level loses
first: 0.74 at f = 2, 0.52 to 0.64 at f = 4, with a mean error of one block at
f = 4, which is slope quantisation (a pitched roof crosses four depths inside one
4 x 4 cell), not a failure. Block agreement 0.78 at f = 2 and 0.67 at f = 4 says a
third of cells at f = 4 sit under a different material than their coarse cell's
vote: trims, window frames, the second material of a wall. That is the residual a
fine level would carry.

**Decision for step 2.** Pool with `any` for the mask and the nearest depth (so the
coarse level is the silhouette a camera sees, with no feature deleted) and the
mode for the block, which is what `any` does; `majority` is the control. The
cascade's levels are f = 16, 8, 4, 2, 1 on the box, which is the H.7 schedule
without 1 x 1.

## §64. Interior blindness off Minecraft: the mesh probe on Objaverse buildings (2026-11-02)

The mock CVPR review's main weakness was "a Minecraft curiosity". This pilot asks
whether the Fill-Interior blindness holds on ordinary textured triangle meshes,
scored with the metrics mesh and text-to-3D papers report. New files only:
`blockgen/eval/bench/mesh_probe.py` (load, visibility, probes, renderer, voxel
bridge), `scripts/fetch_objaverse_interior.py` (selection), `scripts/run_mesh_probe.py`
(metrics; statistics are `distances.kid/fd`, `pointcloud.chamfer_matrix/mmd/...`,
`geometry.descriptor`, `ladder.noise_floor`, unchanged).

**Data.** Objaverse-1.0 annotations keyword-filtered on the name (house, building,
cabin, apartment, interior, room, ...; furniture, props, scans excluded; 2k to 300k
faces, no animation): 17,765 candidates, the 1,388 with interior/rooms/bedroom
words first. 1,152 downloaded (8.3 GB transferred, every GLB deleted after
processing; 325 MB of npz caches kept in `data/external/objaverse_interior/`).
Each GLB is flattened to one mesh, per-face colour sampled from its texture,
centred, longest side 1, y up. *Visible* faces: first hit of some orthographic ray
(320 x 320 grid) from 64 Fibonacci directions plus the 4/8/16 evaluation cameras.
*Hidden* volume: empty cells of a 48-per-unit grid whose rays toward every one of
those directions hit the mesh. Accept iff hidden face area >= 25% and hidden volume
>= 3% of the box: 261 accepted (23%), median hidden area 43%, hidden volume 18%.
The first 256 by a seeded shuffle are used. By name about 65% houses/buildings,
35% single rooms or interiors (sealed boxes with furniture), a handful of oddities
(a car with an interior, a diorama). Licences: 182 CC-BY, 69 BY-NC-SA, 3 BY-SA,
2 BY-NC, per object in `manifest.jsonl` and `results.json`.

**Probes.** Strip-Interior deletes every non-visible face (median 10% of faces
survive). Fill-Interior is the stripped shell plus a voxel solid over the hidden
volume, eroded one cell so it cannot reach the shell, in the object's dominant
colour (voxel count x1.37). Both leave the exterior renders at a mean 0.03/255
pixel difference (max 0.16; faces smaller than a ray-grid cell are lost).
Positive controls: welded vertex jitter sigma 0.5% (3.5/255), the outer 30% +x
slab of faces deleted (a missing wall, 8.8/255), quadric decimation to 5%
(11.1/255).

**Metrics.** Renderer: pyrender EGL, opaque, double-sided, flat-shaded per-face
colour, orthographic, one key light plus ambient, white background, 224 px, framed
on the mesh's own box (the voxel renderer's protocol). DINOv2-B and CLIP-L KID on
4 views (az 45/135/225/315, el 30), DINOv2 on 8 and 16 views, DINOv2 and CLIP on
4 cutaway views (near plane through the box centre), FD on 4 views, CLIP score
against "a 3D model of a building". Chamfer MMD / 1-COV / 1-NNA (k = 1,024) on
clouds from all faces, from visible faces only (visibility recomputed on every
variant), and back-projected from the 4-view depth buffers (P-FID-style clouds,
chamfer rather than PointNet++ features). Voxel row: the mesh voxelised at 32 per
unit (surface cells plus Fill's solid), then `cd_volume`, 1-IoU and V-KID
(`geom_kid`, whitened on the reference). Split: 128 reference, 128 pool, n = 64
probes; noise floor = sd of the metric over 48 real n-draws from the pool; 95%
intervals from 200 paired 80% subsamples of the probe objects.

**Reading.** Every exterior render metric is blind to both probes (0.0 to 0.2
floors at 4, 8 and 16 views, CLIP score included), while the same metrics
separate noise and decimation at 10 to 42 floors. Exterior-surface and
render-derived chamfer are blind too (|sep| <= 0.3). V-KID separates Fill at 51
and Strip at 12. Cutaway renders see both (6 to 12). All-face chamfer is not
blind to Strip, but only just: MMD +2.8, 1-NNA +1.8, against -0.1 and +1.1 for
Fill, because Fill's solid sits under the shell and its faces look like the
interior walls they replaced. The standard point-cloud suite is weak on the
positive controls as well (0.5% noise does not move it; decimation 1 to 2), so its
near-blindness is partly a power limit of nearest-neighbour set statistics at
n = 64, not only an interior effect. Results T76.

Not done: EMD, PointNet++/ULIP/Uni3D features, a paired real-vs-real resample
floor for the per-object chamfers, an interior-only sampler. 3D-FRONT and
Structured3D need a licence agreement; not requested.

**Facade best-of-4 at 0.7 and voxels at 20M (results T77, T78, 2026-11-02).** Facade
best-of-4 at temperature 0.7 scores 483, on the real-faces ceiling (493; `geom_kid` 13.5 vs
13.8, same Holm group): the facade generator has caught up with its representation on
geometry, and the remaining gap is the lift's empty interior plus appearance. Voxels at 20M
(M2b recipe, size only): holdout NLL 0.420 to 0.286, board 386 to 346, tied with 5M
best-of-4 (340). The 20M SFT overfits fast (select=best kept step 1,500 of 16,000), and both
20M runs ran out of memory in the end-of-run draw at sample_batch 64 after training, losing
summary.json and the registry rows; 20M configs now set sample_batch 16. Two other sessions
wrote T75 and T76 on the same day, so these are T77 and T78.

The missing 20M best-of-4 arm ran on 2026-11-03 (two hours of sampling for 1,024 draws) and
the card was rescored: 341, level with 5M best-of-4 (340) and 20M raw (346). It wins
appearance outright (`mv_dino_kid` 0.288, alone in its Holm group) but not geometry
(`geom_kid` 16.3, group `c` with both). Size and selection each buy the same geometry step
from 386 to about 340 and do not stack, so the voxel geometry floor needs a different lever.

## §65. The fixed-pooling face cascade: built and scheduled (2026-11-02)

Step 2 of research.md H.7, the first model of the 2D next-scale arm, written today and
scheduled to train at 07:00 Pacific on 2026-11-03 (`scripts/facade_cascade_morning.sh`,
a user systemd timer `facade-cascade-0703`; the GPU is held tonight by the BrickNet Qwen
pretrain, so the script also waits for 9 GB free before starting). Nothing trained yet.

**What it is.** The NCM/VAR recipe with the learned tokenizer taken out. A face is the
stack of its `facade.pool` levels at f = 16, 8, 4, 2, 1 on a 32 x 32 canvas
(`blockgen.corpus.facade_levels`): level f is a (32/f)^2 grid whose cells hold a depth
class (0 see-through, 1 + d filled) and a block class (a train vocabulary of (id, data)
keys, 512 kept of 793 on side faces, index 0 unknown). Cells outside the face's U x V
rectangle are flagged and never predicted. One bidirectional transformer
(`blockgen.models.face_cascade.FaceCascade`, 5M: d=256, 6 layers; 20M: d=512, 8 layers)
draws level f from level 2f: every cell sees its parent's (depth, block), its own value
when revealed, the in-box flag, its canvas position and the level, plus three global
tokens for the box (U, V, extent). Two heads, depth and block; cells of a level are drawn
conditionally independently, which is the property under test. Training mixes the five
levels (one level per batch, uniform), corrupts a random fraction (up to 0.3) of the
parent's cells with random values as NCM does, and masks the level being drawn
MaskGIT-style (half the batches fully masked, the rest a uniform ratio), loss on the
masked in-box cells. So one checkpoint supports both the plain five-pass cascade
(`steps=1`) and refinement within a level (`steps=4`, confidence-ordered reveal on a
cosine schedule). Side faces of the train split (7,452 faces, mirrored along u: 14,904),
AdamW 3e-4, cosine, batch 96, 30k steps, early stopping on validation NLL of the front
faces (patience 8 evals), bf16.

**What the morning run reports** (`scripts/facade_cascade.py eval`, per run dir):
teacher-forced NLL per level in bits per in-box cell on the 399 test front faces;
box-conditioned samples from the real test boxes at (steps, T) in {1, 4} x {1.0, 0.7},
scored with the sheet statistics (jump rate, components, holes, lone cells, fill, distinct
depths and blocks) and the Wasserstein distance of each statistic's distribution to the
real faces, beside the front faces of the existing arms projected from their builds
(patchwork, markov, per-face 5M AR, joint 20M D4 AR, the RFT at T=0.7); oracle-coarse
completion from the real f=4 and f=2 levels, agreement with the real face against the
naive upsample of T75; `samples.png` (one box per row, the cascade's levels as columns,
the real face of the box last), `samples_s4.png` and `completion.png`. The comparison
rows already exist from the CPU smoke test: front-face jump rate real 0.115 (n=399), RFT
T=0.7 0.143, joint 20M D4 0.308, per-face 5M 0.341, patchwork 0.275 (resampling to the
box adds jumps), markov 0.519.

**The battery.** (1) 5M, `any` pooling; (2) 5M, `majority` (the T75 control that deletes
thin features); (3) 5M, `any`, no parent corruption; (4) 20M, `any`. Each about 15 to 45
minutes by the FLOP count; the whole chain under three hours once the GPU is free.

**Pre-registered reading.** Levels f=16 to f=4 should be easy (the probe found them
smooth and one piece): NLL there near zero, samples with real storeys and roof lines. The
test is f=2 and f=1, where windows and trim live and where within-level independence
bites: the plain cascade should show a jump rate above real and speckled trim; if steps=4
pulls the jump rate toward 0.1 without emptying the face, refinement is doing what
temperature did for the AR arm without losing fill. Completion from f=4 should beat the
upsample on depth exactness (0.52 at f=4 on the front, T75) and block agreement (0.56)
while keeping mask IoU near 0.76; if it does not, the fine levels are not learning the
residual and the learned tokenizer (step 3) is premature. Tests:
`tests/corpus/test_facade_levels.py`, `tests/test_face_cascade.py`.

## §66. A published generator on the bench: Dream-Cubed as the external arm (2026-11-02)

The mock CVPR review said the metric paper never scored a published model. Dream-Cubed
(Merino, Earle, Iwai, Togelius, Cetin; Sakana AI and NYU; arXiv:2604.22847, NeurIPS 2026
poster) is the closest published Minecraft generator with public weights, so it is the first
external arm.

**What it is.** A 279M-parameter 3D DiT trained with masked discrete diffusion (MD4) or with
DDPM on block embeddings. It generates 32^3 terrain chunks conditioned on one of 15 biome
classes (one of them `village`), plus inpainting and outpainting to larger worlds. It is not
text-conditioned. The data is 1.12.2 procedural worlds: about 1.8M natural chunks, plus
178k village chunks located with Pyubiomes and cut at stride 4. A second family adds six
human-authored city maps by Timenth_Buildings, with each map as its own class. The training
data contains no GrabCraft, 3D-Craft or text2mc builds.

**Licences.** The `dream-cubed/dream-cubed-models` weights (MD4/DDPM natural, balanced and
village-boosted) are Apache-2.0, and so is the `SakanaAI/DreamCubed2M` data and mappings repo.
The human-map models (`dream-cubed/dream-cubed-human-models`) and the `DreamCubedHuman` data
are CC-BY-NC-4.0. The GitHub code repo has no LICENSE file, and the arXiv paper is
CC BY-NC-SA 4.0. We run the code for evaluation and do not redistribute it.

**Which checkpoint.** We use `MD4_p4_boosted` conditioned on `village` with the config's own
settings: CFG 4.0 and 1000 reverse steps. It is the only Apache-2.0 checkpoint whose class
produces buildings, and patch 4 makes it cheap. The paper reports about 25 s per chunk on an
H100, against about 2.5 min for patch 2. Here it ran at 17.8 s per chunk at batch 16 with a
2.46 GiB peak on the shared GPU, so 288 chunks took 85 min. The human-map DDPM is patch 2
and NC-licensed; 256 chunks would take more than 10 GPU-hours here, so it is not run.

**Integration.** `libs/DreamCubed` holds the clone, a 14 MB overlay venv that reuses the
main torch through a `.pth` and adds `einops` and `ema_pytorch`, and 1.1 GB of weights.
`scripts/ext/dreamcubed_generate.py` builds DiT3D plus the MD4 sampler directly and stubs
wandb, pyvista and the renderer imports. It also works around an upstream bug: the converter
reads `assets/block_types.json`, which the repo does not ship. `scripts/ext/dreamcubed_to_npz.py`
writes the arms.

**Turning a chunk into a build.** A Dream-Cubed sample is a village on terrain (see
`samples_raw_chunks.png`). Scored whole, it is disqualified for mode collapse, because every
sample is a 32-wide slab of stone and dirt. The ranked arm `dreamcubed_32` is extracted as
follows:

- keep the constructed blocks;
- count sandstone only where no terrain block lies above it in its column, since desert
  sub-surface is sandstone under sand;
- select the 26-connected component with the most non-log blocks, so bare trees never win;
- crop that component to its bounding box;
- drop chunks whose component has fewer than 80 blocks.

Yield is 284 of 288. The extraction is ours and favours the model. The raw arm is kept beside
it so the cost of the choice is visible. Two vocabulary losses come from Dream-Cubed itself:
its slab and stair tokens carry no material (mapped to stone slab and oak stairs), and its
names are 1.12 Bukkit materials without variants (PLANKS, LOG and WOOL map to data 0). Both
push the palette pillar up. No rescaling: chunks are already 32^3.

**What the builds are.** The builds are vanilla 1.12 village pieces: cobblestone-and-plank
houses with oak-stair roofs, small huts, the library, the blacksmith with its stone-slab rim,
wells and lamp posts, and fragments of desert and savanna villages. They are small, with a
median of 278 blocks and a 9x8x9 bounding box, against 745 blocks and 16x12x15 in houses_32.
Results are in results.md T79.

## §67. The VLM judge: an exterior-view observer with the same null space (2026-11-02)

The VLM-judge arm replaces the removed 2AFC rater study. gpt-4o-2024-11-20, temperature 0,
both presentation orders, pre-registered plan (prereg.json, plan sha on file). Only 3D-Craft
builds and probes derived from them reach the API. The LLM-FT arms are excluded (user
decision). The plan has no human-study conditions or rater join: an earlier plan reused the
rater stimuli, was stopped at 430 calls (run_20261102_173248) and superseded. Run
`outputs/run_20261102_214026_vlm_judge`, 2,288 trials, all valid, $4.31.

Every pre-registered probe prediction holds. P(prefers real), cluster-bootstrap CI:
Fill-Interior on the metric's exterior views 0.53 [0.50, 0.56] (chance); roof-off cutaway
0.70 [0.64, 0.75]; quadrant cutaway 0.63 [0.53, 0.72]; Hollow-Shell exterior 0.52
[0.49, 0.54]; Hollow-Shell roof-off 0.73 [0.66, 0.80]; real vs uniform random 1.00. On
exterior views the judge picks the first slot 93% of the time and the two orders agree on
only 11–13% of pairs, so the preference is position, not content. The 89% of houses whose
fill is visible through a window do not move it (0.53 changed vs 0.50 unchanged), so for
this observer window leakage does not break the null.

Gabled-House vs Patch-Stitch: the judge prefers Patch-Stitch 0.72 (P(Gabled) 0.28
[0.21, 0.35]), the ordering MV-KID and V-KID give. The pre-registered prediction (Gabled
above) fails.

System level (Bradley–Terry over 11 arms) is a weak ranker. Real is 6th; Fill-Interior and
both Voxel-AR arms rank above it. Agreement with MV-KID: Spearman 0.76; with BlockScore:
0.57; with V-KID: 0.16. The judge reads exterior views, so it shares the render metric's
null space. Read it as a third exterior observer, not as ground truth. The correlation also
compares different subsets (judge: 3D-Craft only; metrics: full scored sets). Recompute it
against the n=256 leaderboard when that lands (`--analyze-only`, with RUN_OF_RECORD
pointed at it).


## §68. The 3D-native rows: Uni3D, ULIP-2, chamfer and EMD under three samplers, on Objaverse and Minecraft (2026-11-03)

§64 left the point-cloud row at chamfer on 1,024 points and asked for EMD and learned 3D
encoders. This section adds both on both domains, at two seeds, with every point metric
run under three samplers, so the table can say whether a metric's blindness follows what
its points observe. Results are T81.

**Samplers** (`mesh_probe.SAMPLERS`, `sampler_masks`, `surface_samples`). *all*: area-weighted
samples over every face. *vis*: area-weighted over the faces that are the first hit of an
exterior orthographic ray (64 Fibonacci directions plus the 20 evaluation cameras, 320^2 rays
each; the set Strip-Interior keeps). *cut*: area-weighted over the faces hit first by rays that
start on the plane through the box centre in the four cutaway views (what `render(cut=True)`
shows). Each point carries its face colour. Minecraft builds go through
`mesh_probe.structure_mesh`: one quad per block face bordering air (exterior or sealed), block
base colour, the voxel suite's frame (bbox centre, /32, no rescale). On a hollow 8^3 cube the
vis sampler keeps exactly the 64% outer area (test).

**Rows** (`blockgen/eval/bench/probe3d.py`, `pc_encoders.py`). Uni3D-B (BAAI/Uni3D
`modelzoo/uni3d-b`, 178 MB; EVA02-B blocks via timm 1.0.30 installed out of tree in
`data/external/pylibs`, every weight loads) and ULIP-2 PointBERT (SFXX/ulip, 10k xyzrgb,
402 MB; strict load), both pure torch (greedy FPS from index 0 and topk kNN, as the repos;
no pointnet2_ops), 10k points, each repo's eval preprocessing (unit sphere; Uni3D swaps y/z
for its z-up OpenShape frame). KID and FD on the L2-normalised CLIP-space embeddings; FD uses
`probe3d.FastFD` (one eigendecomposition of the reference, identical value to `distances.fd`,
about 100x faster, needed for the CIs at d = 1,024/1,280). Text-shape score (the Uni3D-score /
CLIP-score analogue) against "a 3D model of a building": the text towers of OpenCLIP
EVA02-E-14-plus and ViT-bigG-14 were range-read from the safetensors (1.4 GB and 2.8 GB, RAM
only, nothing kept; `scripts/pc_text_embed.py`); sanity: building vs house cos 0.93, vs chair
0.77. Chamfer (PointFlow convention) on 2,048 points; EMD *exact*: optimal assignment
(`linear_sum_assignment`) between equal 512-point subsamples, mean matched distance. Exact
because Sinkhorn at 2,048 points (50 eps-scaling steps) took 47 ms/pair on the shared GPU and
left marginals off by 30-50%, and the ladders need 0.12M (Objaverse) to 0.42M (Minecraft)
pairs per sampler; distances are computed only for the pairs MMD/COV/1-NNA read
(`pc_encoders.needed_pairs`). Set statistics, noise floors (z = (arm - real held-out) / sd of
48 real n-draws, finite-population corrected) and 200 paired 80% subsample CIs are the
suite's own. Raw MMD^2 (KID) and raw MMD sit beside every z in the run tables. PointNet FPD
(Shu 2019) not run: no maintained checkpoint for the colour-free 2,048-point PointNet.

**Objaverse at paper scale.** Strict rule (`mesh_probe.STRICT_RULE`, `interior_stats`): hidden
area >= 30%, hidden volume >= 5%, >= 2 sealed spaces (6-connected hidden components of at
least max(27 cells, 0.2% of the box)), >= 3% of the area on hidden faces two or more cells from
the shell. 139 of the pilot's 261 pass. The fetch then took interior-worded candidates (name,
tags or description) by GLB size ascending, capped at 8 MB (small files return 124 accepted/GB
against 12/GB at 10-20 MB). The disk budget was cut mid-run to 6 GB of new data in total, so it
stopped at 5.4 GB transferred (3,936 candidates seen, 609 loose-accepted, **290 strict**), not
512. Design therefore 130 reference / 160 pool / **n = 96**, not 128. Medians of the 290: 5
sealed spaces, 47% hidden area, 24% hidden volume. About 45 of the 290 are not buildings
(appliances, furniture, cars, matched through tags); re-scoring the encoder KID rows on the
233 building-named objects alone (ref 104, n 77) changes no verdict (vis: |z| <= 0.4 on both
probes at both seeds; all/cut: 2 to 15). Licences: 213 CC-BY, 3 CC-BY-SA, 65 BY-NC-SA, 9 BY-NC;
samples.png shows only CC-BY/BY-SA objects (named on the image). Exterior pixel delta of
Strip/Fill 0.04/255.

**Reading.** Sensitivity follows the sampler, on both domains, for every 3D-native metric.
Under *vis* every row is blind to Strip and Fill (all |z| <= 1.0 at both seeds; Hollow and
Fill-Interior on Minecraft <= 1.6), while the encoder rows under the same sampler separate the
exterior controls at 5 to 188 floors (chamfer and EMD see only the missing wall on Objaverse,
and canon16 / 40% deletion on Minecraft). Under *all* and *cut* the learned encoders see the interior: Uni3D KID Strip
+4.6/+11.5 and Fill +8.5/+14.1 (all), Minecraft Fill-Interior +21.7/+22.6 (all) and
+37.0/+30.4 (cut); ULIP-2 is weaker (Objaverse 2 to 11, Minecraft Fill 3 to 10). Chamfer and
EMD set statistics stay weak even when they see the points: Strip MMD +3 to +4 (all), Fill
-0.6 to +1.4 on Objaverse (Fill's solid sits under the shell and stands in for the interior
walls it replaces), Minecraft Fill +4 to +8 (all/cut), Hollow <= 1.8. The text-shape scores are
blind under every sampler (|z| <= 2.7 on interior arms) and weak on the exterior controls
too (mostly |z| < 5, at most 9), like CLIP score. The voxel-centre rows of T67 (`cd_vox_volume/surface`) give Minecraft
Fill-Interior -3.7/-2.8 and +2.2/+2.9: the volume cloud moves the wrong way because adding mass
spreads the 1,024 samples over more cells. Pose: Uni3D moves +1.1 to +1.9 under a 90 degree
rotation (it is not rotation invariant), ULIP-2 0.1.

**Which sampler the published metrics use by default.** Uni3D, ULIP-2 (OpenShape / ULIP
triplets), PointFlow and Achlioptas MMD/COV/1-NNA, and Shu's FPD all sample points uniformly
by area over the whole mesh surface, i.e. *all* here: the setting where a learned encoder can
see a sealed interior, if the generator emits one. Pipelines that first make a mesh watertight
(Occupancy Networks style ShapeNet preprocessing) or sample from rendered depth (P-FID-style,
`cd_render`) are *vis* and inherit the render metrics' null space exactly. Generated meshes from
surface-fitted or SDS generators are exterior shells, so on them *all* and *vis* coincide and
the learned 3D metric is blind in practice too. The paper's claim, stated with this table:
a metric inherits the null space of what its sampler observes.

**Seeds.** Seed changes the split (Objaverse), the probe draw (Minecraft), and every sampler
draw. Objaverse 275 cells: Pearson 0.91, Spearman 0.91, blind/not-blind (|z| < 2) verdict
agrees on 96%; 3D-native rows 0.89 / 95%; interior arms only 0.985. Minecraft 336 cells:
Pearson 0.98, Spearman 0.93, verdict 93% (interior arms 0.98). Disagreements are cells near
|z| = 2. The vis-sampler verdict (blind) holds in every cell at both seeds.

Runs: `outputs/run_20261103_004358_mesh_probe3d_s0`, `outputs/run_20261103_030030_mesh_probe3d_s1`,
`outputs/run_20261103_004358_mc_probe3d_s0`, `outputs/run_20261103_004358_mc_probe3d_s1`
(each with table.md, results.json, samples.png, embeddings.npz; seeds.md/seeds.json from
`scripts/summarize_probe3d.py`). Scripts `scripts/run_mesh_probe_3d.py`,
`scripts/run_mc_probe_3d.py`, `scripts/pc_text_embed.py`; tests `tests/eval/test_probe3d.py`.
GPU work ran under the shared flock (`GPU_LOCK`), one acquisition per run.

## §69. The n=256 leaderboard of record under the 0925 protocol (2026-11-02 to 2026-11-03)

The paper's own protocol asks for n >= 256 and its leaderboard (`run_20260925_002631`, the
run of record) ran at n=128. This reruns that board at n=256 under exactly 0925's protocol,
drops the two LLM arms (user decision), and adds the external Dream-Cubed arms (§66).

**Protocol pin.** Since 0925 the bench changed in two ways that alter what it computes. The
geometric descriptor gained three occupancy scalars (`walk_standing`, `walk_floors`,
`walk_headroom3`, 2026-10-23), so V-KID went from 77-d to 80-d, and BlockScore gained a sixth
`traversal` pillar fed by a new `traversal` block in the fast tier. Both stay the default
(`v2`). A pinned `v1` restores 0925: `geometry.DESCRIPTOR_VERSIONS` /
`set_descriptor_version`, the bench flag `--descriptor v1|v2`, and the stamp
`context.descriptor_version`, `descriptor_dim`, `geom_descriptor`. Under v1 the scalar panel is
the original eight, `ladder.geom_view_key()` returns the bare `patch2_orb55_t5_h8` (so the geom
gate reads the 0925 ladder file), `composite.structure_blocks()` drops `traversal`, and the fast
tier skips the traversal report entirely, which also keeps the shared bootstrap `rng` stream
where 0925 left it. Every other diff from 7e24d05 to the working tree was checked and computes
nothing different on these arms: decimation controls were already gone from 0925's arm list,
head-to-head was already the permutation test, the renderer cull fix and canary predate 0925's
dirty tree, `match_arm`'s train-split check and the protocol descriptor check change ranking
*status* in the lab only, and `C3`'s `NEEDS` tuple only matters when a traversal pillar exists.
One bug fixed on the way: `runner.main` reassigned `ctx.derived` after stamping
`geom_descriptor`, so that stamp (added 2026-10-23) never reached a card; it now updates the
dict. New flag `--save-features` writes per-build `geom_kid`/`mv_dino_kid` features
(`features.npz`, about 15 MB) so rank intervals can be bootstrapped after the fact.

**Validation.** `run_20261102_214725_bench_validation_0925_re-score_under_the_v1`: 0925's argv
verbatim plus `--descriptor v1`. All 560 recorded metric values match 0925. geom_kid, every
fast-tier number, the noise floors, ladder pass/fail, all six BlockScore self-checks and both
head-to-head letter groupings are bit-identical. The DINO path differs by GPU float noise only:
max |d mv_dino_kid| = 7.4e-7, max |d BlockScore| = 2.8e-4 noise floors.

**Samples.** The AR npz files held 128 builds. `scripts/sample_to_npz.py`'s loop (seed 0,
T=1.0, top-k 40, same checkpoints) was run in resumable chunks with the torch CPU+CUDA RNG state
carried across chunks so the GPU lock could be released between them. Rows 0-127 of every new
file are byte-identical to the old file (128/128 for all three arms), so each n=256 arm is the
0925 arm plus 128 more draws from the same stream: `outputs/bench_arms/{canon_oriented,
canon_unoriented,native_oriented}_n256_32.npz`. Dream-Cubed files were truncated to their first
256 builds (`dreamcubed_32_n256.npz`, `dreamcubed_raw_32_n256.npz`) because `--n` does not cap
file-backed arms.

**Result.** `run_20261103_090315_bench_n_256_leaderboard_of_record` (results.md T82;
`rank_intervals.md` from `scripts/bench_rank_intervals.py`). Self-validation passes 6/6. Raw
KIDs are n-stable (canon_oriented geom_kid 14.90 -> 15.06, mv_dino_kid 0.245 -> 0.254), but the
measured noise sd halves (geom_kid 0.110 -> 0.053, mv_dino_kid 0.00289 -> 0.00161), so every
pillar and every BlockScore roughly doubles: canon_oriented 136.73 -> 285.22, Solidify 201.80 ->
444.36. Separations in noise floors are not comparable across n; only ranks and ratios carry
over. Three things move beyond the doubling. (1) **canon_unoriented is no longer
disqualified.** Its memorization gate was marginal at n=128 (+3.0 spreads against a 3.0 bar);
at n=256 its retrieval percentile moves 0.382 -> 0.403 and real_test's 0.534 -> 0.507, the
gate floors the spread at `min_spread`=0.05, and it reads +2.1, so it ranks fifth at 345.10.
(2) **Dream-Cubed ranks first among generators** at 165.71 (appearance-limited; geometry
146.64), ahead of native_oriented (229.29) and canon_oriented (285.22), all with degenerate
rank intervals. Its train split is external, so the lab's protocol check leaves it unranked.
dreamcubed_raw is disqualified as intended (mode collapse, +24.0 spreads). (3) Among the
controls, Material-Shuffle, Patch-Stitch and Solidify become a three-way tie (ranks 6-8,
adjacent flip probabilities 0.19 and 0.20), and Copy-Noise and Gabled swap places in a tie
(flip 0.47). Solidify still shares a significance group with real on mv_dino_kid (`a`), and
its appearance pillar is +0.35 against +444.36 on geometry (ratio 1256; 0925: +0.14 and 1449).

**Dream-Cubed size versus shape.** Its builds are small (median 272 blocks against about 745
for houses_32). Refitting the 77-d standardiser without the `log1p(n_blocks)` column moves its
geom_kid only from 7.67 to 7.17 (145 -> 132 of that variant's own noise floors), and scoring
it against the 111 reference builds inside its own block-count IQR (155-462) raises geom_kid
to 8.57. Size is at most about 7% of its geometric gap; the rest is shape. canon_oriented,
for comparison: 14.90 full, 15.18 without size.

**Caveat on n=256 noise.** `ladder.noise_floor` draws 256 of the 399 test builds per
replicate and divides out the finite-population factor (1 - n/N = 0.36). That correction is
exact for a mean and leading-order for a U-statistic, and at n/N = 0.64 it is doing real
work, so the n=256 noise sd (and with it every separation) carries more model error than
at n=128. Ranks do not depend on it.

**Not done.** The lab's `houses32-v1` protocol pins the v2 descriptor (`_s12`), so it marks
this run `official: false` for the descriptor reason. Whether the paper's board should be
v1 (this run) or v2 is the user's call. No `.tex` and no `RUN_OF_RECORD` change; the run is
registered as `1103_n256` in `scripts/audit_claims.py`.

## §70. A published generator on the LEGO suite: BrickGPT on StableText2Brick's held-out test split (2026-11-02 to 2026-11-03)

**Why.** The mock CVPR review said the metric paper never scores a published model. BrickGPT
(Pun et al., ICCV 2025, arXiv 2505.05469, MIT code, MIT LoRA weights, MIT dataset) is the
cheapest one: its training corpus, StableText2Brick (ST2B), is already the paper's second
domain (`bench/lego.py`, T34).

**What the LEGO suite is.** `bench/lego.py` voxelizes each ST2B brick list (`hxw (x,y,z)`,
20^3, z up) into a Minecraft `Structure` (y up), one wool colour per brick footprint, so the
whole bench stack (V-KID `geom_kid`, MV-DINO-KID over 4 ortho views, coherence, geometry
scalars, traversal, novelty, the real controls and BlockScore) runs unchanged. The paper's
corpus `lego_void_20` is NOT a fair reference for BrickGPT: it pools ST2B train+test (BrickGPT
trained on train) and keeps only structures with a sealed void (44%). The smallest honest fix,
taken here: `lego.py --splits test` (new flag) builds `lego_st2b_test_20`, all 4,785 test
structures, no void filter (`--no-void-filter --min-blocks 1 --limit 0`). ST2B test objects
are disjoint from train objects (0 of 2,851 shared). The bench's own group split cuts it
3,349/718/718; the reference is 512 of the 718 val.

**Prompts.** 256 structures drawn uniformly (seed 0) from the bench's *test* partition, so
the prompted objects are in neither the reference nor BrickGPT's training data, and the
category mix matches the corpus (table 89, chair 43, car 30, sofa 25, ...). One of each
structure's 5 captions, seeded; prompt `i` uses generation seed `1000+i`.
`prompts.jsonl` in the run dir.

**Install.** `libs/BrickGPT` (official repo, commit 9c53c7c, MIT). The venv is 175 MB: a 3.13
venv whose `.pth` points at the main `.venv` (torch 2.10, transformers 4.57.6, peft 0.19.1)
plus `gurobipy` and `pyscipopt` (an earlier standalone venv pulled a second torch; deleted,
and its 12.6 GB of uv cache removed). The adapter's base `meta-llama/Llama-3.2-1B-Instruct`
is gated (403 for our token), so the base is the `unsloth/Llama-3.2-1B-Instruct` mirror (same
2,471,645,608-byte safetensors; Meta hides the hash, so size is the only check; the Llama 3.2
Community License permits redistribution).

**Generation** (`scripts/brickgpt_generate.py`). Default `BrickGPTConfig` (temperature 0.6,
+0.01 per repeated rejection, 500 rejections per brick, 100 rollbacks, logit masking) with
three forced deviations: bf16 (to fit the shared card), the base-weight mirror, and
`use_gurobi=False`, i.e. the repo's documented connectivity rollback instead of the physics
rollback, because the pip Gurobi licence refuses every ST2B structure down to 20 bricks
(about 22 variables per brick against a 2,000 cap). Plus a 1,800 s wall-clock cap per prompt
(not in BrickGPT): 3 prompts hit it (idx 5, 17, 67) and count as failures. Throughput on a
contended 16 GB card: median 11.1 s, mean 89.9 s per structure (40/h), peak 3.1 GB allocated.
Median rollbacks 1 (mean 6.5), 62% of structures roll back at least once, 1.2% hit the
100-rollback cap. 2.02 rejected brick proposals per accepted brick (67% of proposals): 76%
`already_rejected` resamples, 24% collisions, 33 not-in-library, 11 out-of-bounds. Wall time
was ~15 h, almost all of it GPU-lock queueing and OOM retries caused by unlocked jobs.

**Stability without Gurobi** (`scripts/brickgpt_scip_stability.py`). BrickGPT's own
`stability_analysis.py`, loaded with three mechanical rewrites onto SCIP (Apache-2.0): `.X`
reads to `getVal`, `y == abs_(x)` to `y >= ±x`, `m == max_(f)` to `m >= f` (exact at the
optimum: both enter the minimised objective with positive weight); the bilinear
complementarity constraints stay. Validated against the Gurobi scores ST2B ships per
structure (note: the dataset field is the score s, 1 = stable, not the heat map): 240/240
solved test structures agree on stable, mean per-structure score difference 4.6e-7. All
reference structures are stable, so the unstable side was checked synthetically: a balanced
T is stable, a tall 2x4 stack hung off a 1x8 cantilever on one stud is unstable (s = 0 on the
pillar). About 6% of structures time out at 300 s or hit a SCIP LP error; these are reported
as unsolved, never as unstable.

**Results** (run `outputs/run_20261102_220228_brickgpt_lego/`, T83). BrickGPT's own metrics:
valid 98.8% (253/256; the 3 failures are the timeouts), ground-connected 98.8% of valid,
physically stable 95.8% of solved (230/240; bounds over valid 90.9 to 96.0%), mean stability
0.995, mean min-brick stability 0.896. The paper reports 100% / 98.8% / 0.996 / 0.915 with
Gurobi rollback and 100% / 24.0% / 0.947 / 0.228 with no rollback. Our run sits close to the
full method, and the ~3-point stable gap is what swapping the physics rollback for the
connectivity rollback should cost (connected but cantilevered structures pass the
connectivity check). Median 11 s vs their 40.8 s is not comparable (connectivity checks are
nearly free; ours ran in bf16 on a shared card).

On the suite (n=253, ref 512, v2 80-d descriptor): `geom_kid` places BrickGPT at **+52.1**
noise floors and MV-DINO-KID at **+11.8**; the solidified real control sits at +40.5 and
−0.09 on the same run. BlockScore 52.1 (worst pillar geometry), ranked behind real@solidify
(40.5). The brick-footprint mix (the "palette" pillar here) is +40.8: BrickGPT's brick-type
distribution differs from ST2B's (JSD 0.009). Coherence says why geometry is far: largest
component 0.980 vs 0.999 (more separate ground-standing pieces), and sealed interior 0.011
vs 0.034 of real, i.e. a third of the hollow volume real ST2B objects enclose. Not
disqualified on novelty (NN-IoU 0.589 vs real 0.630; DINO NN percentile 0.558). The prompt-
matched held-out structures score like real (BlockScore 0.40). Pinned to the paper's v1 77-d
descriptor (`bench_v1/`) the numbers barely move: BrickGPT +51.5, solidify +36.4.

**Read.** The suite separates a published model from real data on both tiers, so it is not
only a ladder of synthetic damage; the geometric tier separates it 4.4x further than the
render tier, and part of that gap is interior (sealed-void deficit) that the render tier
cannot see by construction. The solidify control replicates on this unfiltered, test-only
corpus too (render −0.09, geometry +40.5). Open for the user: whether this goes in the main
text (one row plus one sentence in §sec:lego) or the supplement, and whether to rerun the
generation with a Gurobi academic licence (UCLA qualifies) to match BrickGPT's default
rollback exactly; the outputs can be re-scored with `scripts/brickgpt_validity.py` either way.

## §71. Nearest neighbours under the two feature spaces: a double dissociation (2026-11-03)

Run `outputs/run_20261103_185719_nn_geom_vs_dino`, reading the n=256 leaderboard's saved
features (same whitening and normalisation the KIDs use). Probes are index-aligned with
real_test, so each altered house's own source is known.

- **Fill-Interior (solidify):** in DINO space the solidified house's nearest real build is
  its own source 100% of the time (median source rank 0 of 256). In V-KID space it is the
  source only 26% of the time (median rank 17.5).
- **Material-Shuffle / Monochrome mirror it:** V-KID retrieves the source 100%; DINO only
  24–26% (median rank 7–9). Each space is blind exactly where the other sees.
- **The two spaces disagree on who is similar.** For held-out real builds, the nearest
  reference house is the same in both spaces only 3.9% of the time (chance 0.25%); top-5
  overlap is 10% (chance 1.25%). Generated arms show the same picture (top-5 overlap 3–8%).
- **NN distance is not a quality signal in V-KID space.** Solidify (0.68), Hollow-Box (0.36)
  and raw Dream-Cubed chunks (0.57) sit *closer* to their nearest real build than real builds
  sit to each other (ratio 1.0), because they collapse into dense regions of the reference.
  This is the same failure that keeps the NN-based fidelity metrics (precision, density)
  from passing the gates; distributional KID is needed.

## §74. The cascade ran: what the levels learned and what to do with it (2026-11-03)

The §65 battery fired at 07:00 Pacific and finished by 07:27; the 20M run ran out of
memory at batch 96 and was rerun at 48; two dropout runs and a per-level checkpoint
selection were added in the morning after reading the curves. Numbers in results T88.

**Three things the curves and sheets say.**

1. Overfitting starts at the coarse end. The f=16 level (four cells per face, the box's
   material) is best at step 1,000 in every run and climbs 4 to 5 bits after; f=8 and f=4
   follow. With 7.5k faces the box-to-material map is memorised, and the total NLL hides it
   because f=1 holds 1,024 of the 1,364 cells. Dropout 0.3 helps the fine levels (f1 2.97
   to 2.76 bits) and not the coarse ones. Added `best_f<f>.pt` per level and a
   `per_level` cascade that runs each level on its own selected checkpoint; it changes the
   samples little, because the coarse levels' failure is not the checkpoint.
2. The depth residual is learned, the block residual is not. Oracle completion from the
   real f=4 at the mode beats the copy on depth exactness (0.58 vs 0.52) and sharpens roof
   lines, but loses on block agreement (0.52 vs 0.56) and never places a window. At
   temperature 1 the fine levels speckle. This is the H.3 prediction with the split made
   explicit: silhouette is a per-cell function of the parent, a window is a joint pattern
   that conditionally independent cells cannot agree on.
3. The face statistics are now known to be gameable by smoothness. The cascade with four
   passes per level matches real on jump rate (0.114 vs 0.115), components, holes and
   fill, better than every AR arm, and the faces are blobs. T60/T68 read those statistics
   as evidence of fragmentation; they are, but their converse is not evidence of houses.
   Any facade card from here needs a distribution metric on face images (a KID on a small
   face feature, or the per-face AR's NLL of the generated faces) beside them.

**Smaller results.** Parent corruption (NCM's code corruption, rate up to 0.3) costs 0.07
bits teacher-forced and does nothing visible for samples at this scale; `majority` pooling
completes worse; 20M buys 0.18 bits over 5M and the same blobs. Training: 5M in 9 to 13
minutes, 20M in 15 at batch 48. GPU sharing: a second training job beside the 20M dies at
step 0 because the 20M's peak is above what nvidia-smi samples between steps; the queue
has to wait for the process, not for free memory.

**What to do with it.** The cascade earns its place for what it does well, the silhouette
and the box's material in four cheap passes, and the fine levels should not be asked to
draw windows independently. Three options, in the order I would try them:

a. Coarse by cascade, fine by AR. Use the cascade's f=4 (or f=2) level as a forced prefix
   to the existing per-face AR model (T66's 5M, retrained with the pooled level serialised
   before the face), so the sequence model draws the trim and openings with its neighbours
   in context and the cascade supplies the gestalt. The oracle test is free: give the real
   f=4 to the prefixed AR and measure block agreement and window counts against the copy.
   This is the T53 two-stage with the cell assignment learned instead of hand-built.
b. Make the fine levels dependent. A small AR or MaskGIT-with-many-steps head at f<=2
   only (the 16 x 16 and 32 x 32 grids), keeping parallel draws for f>=4. Cheaper than (a)
   per face but a new sampler.
c. More faces. The coarse levels memorise 7.5k boxes; the anneal tier (docs/pipeline.md)
   has tens of thousands of builds and the roof plans are a second face type. Worth doing
   under (a) or (b) regardless, since a scaled cascade alone would still speckle.

The learned residual VQ (step 3 of H.7) is deferred: its failure mode is the same
independence, and nothing above needs it. Six faces jointly and the lift to the bench
wait for a fine stage that draws windows.

## §72. The CVPR paper regenerated from the n=256 leaderboard of record (2026-11-04)

User decisions (2026-11-04): the leaderboard of record is
`run_20261103_090315_bench_n_256_leaderboard_of_record` (§69); everything else (ladder,
dose-response, gates, probes, case study) stays on the 0925 protocol run. Dream-Cubed
renders and LEGO renders may appear; BrickGPT gets a main-text row and a supplementary
reproduction table. Backup of the pre-edit paper: `deliverables/paper/cvpr27/.backup_20261104_pre_n256/`.

**Audit registry.** `scripts/audit_claims.py` now has two run roles. `RUN_OF_RECORD = "0925"`
binds every claim reading `card`; `LEADERBOARD_RUN = "1103_n256"` binds every claim reading
`lb` / `lbc` / `lbdir`. The role is read off each claim's closure; a claim reading both raises.
The TeX cross-check takes each role's alternative values only from runs other than that
role's own run, and percent numerals are never matched against pillar cells (they were
chance hits). New flags `--leaderboard-run`, `--paper cvpr27` (short name). Readiness reads
the leaderboard run for the n >= 256 check. 679 claims, all OK; 0 stale.

**The two failing claims, re-derived** from `outputs/figstage/contiguous_draw_check.json`: the
contiguous draw inflates the noise floor 10.88x (paper said 10.4x), shuffled sd 0.00294
(paper said 0.00305; the protocol run's floor is 0.00289). Paper now prints 10.9x, 0.0320
vs 0.0029, centre +0.0410.

**Stale numbers the audit could not see** and that were fixed by hand: the dose prose and
tab:dose quoted run_20260831/0918 runs, not the registered 0925 dose runs (render interior
range 0.4/1.1 -> 0.48/0.34; mixture/occupancy ranges 121-143 -> 185 and 549 at n=128;
Spearman +0.300/+0.900 -> +0.700 at both n); tab:blindness printed 0917 values for
Material-Shuffle (0.674 -> 0.667), Monochrome (0.930 -> 0.859) and Train-Copy (0.015 ->
0.017); tab:casestudy letters (d/def/de/e/fgh/f/j over 153 pairs) are now the 0925 groups
(c/cd/cd/d/e/e/h over 120 pairs); the sqrt readout is 6.0% (0.105 vs 1.740), exponent
2.05, MAE 0.013/0.020 vs 0.128/0.131; the rescore prose quoted +0.26/+0.03/-0.16 while
its own table read +0.119/+0.026/-0.069; the 16-view sentence was inverted (16 views read
+1.6, more than 4 views' +1.1, both under two floors).

**New derived numbers.** `scripts/dreamcubed_size_check.py` reproduces §69's size check
exactly and writes `outputs/figstage/dreamcubed_size.json` (7.67 -> 7.17 without the size
column, 8.57 against the 111 reference builds in its IQR 155-462; reference median 675
blocks). `scripts/vlm_vs_leaderboard.py` recomputes the judge's Bradley-Terry correlation
against the n=256 board without touching the judge run: Spearman MV-KID 0.74, BlockScore
0.54, V-KID 0.20 (n=128: 0.76, 0.57, 0.16); real still sixth. The NN figure
(`scripts/figs/nn_dissociation.py`) found that V-KID sends filled houses to a few hub
builds: 68 distinct neighbours over 256 queries, the top four taking 69.9%.

**Dream-Cubed wording.** A template-reproduction audit is pending (coordinator,
2026-11-04): the novelty gate checks only houses_32 train, not the ~483 vanilla village
templates. The paper states the row's numbers, rank 2, the extraction rule and the 284/288
yield, says "no near-copies of the houses_32 training split", and carries a `\TODO` at
the first mention in 6_results and in app:dreamcubed.

**Figures** (CPU renders, OSMesa, `BLOCKGEN_FIG_RENDER=cpu`): fig_samples (n=256 run,
Dream-Cubed in, LLM arms out), fig4 (Holm letters parsed from the 0925 scorecard), fig2
(registered 0925 dose run), fig3 (n=256), fig6 (explicit 0918 LEGO ladder, no fallback;
no 0925 LEGO ladder exists), fig7 (3D-Craft only: test rows 249, 138, 383), new
fig_nn_dissociation. Written to figures_new/ and copied into figures/.

**Dream-Cubed audit and paper candidates (results T86, 2026-11-04).** Dream-Cubed's builds are
repeated vanilla village pieces: 59% have a near-twin in their own set (real 0.8%), and the
vanilla templates alone score about 200 and rank 3rd, ahead of every BlockGen model. Our
novelty gates miss repetition within a set. The CVPR session holds its Dream-Cubed rank claim.
Six candidate-paper reports are in `deliverables/reports/`, all 2 of 5 as standalone papers
today. The pretraining report finds that the `anneal` pretraining tier holds 1,834 of the 1,863
SFT houses, so T60's transfer claim is not yet separated from extra epochs on the SFT set. The
next experiment is a world-crop-only pretraining arm against a scratch arm tuned on holdout.

## §73. Dream-Cubed's builds are mostly repeated vanilla village pieces (2026-11-04)

Audit by the peer agentic-suite session, CPU only: `outputs/analysis/dreamcubed_audit/README.md`.
Arm: the 284 extracted builds behind T82. Real control: houses_32 test split.

- **Within-set near-twins** (direct test, no reference needed). Median IoU to the nearest
  other build: Dream-Cubed 0.848, real 0.426. Builds with a twin at IoU>=0.8: 59.2% vs 0.8%;
  at >=0.9: 34.5% vs 0%. Clusters at 0.8: 160 of 284 vs 255 of 256. Largest cluster: 28 copies
  of one 117-block hut. At matched block count (n=98 each): 0.878 vs 0.493, near-twins 73% vs 0%.
- **Template IoU** against 299 1.21 village templates: median 0.52 vs real 0.32; none >=0.9.
  Dream-Cubed trained on 1.12.2 worlds, so 1.21 templates are a proxy and understate copying.
- **Composition:** 45% houses with a room, 36% roofless enclosures, 19% fragments (real 99% houses).
- **Template-control arm** (1.21 village templates through Dream-Cubed's own extraction, T82
  protocol, CPU fast tier): geom_kid 10.25 to 10.67, geometry pillar 195 to 203, mv_dino_kid
  ~0.17, BlockScore ~195 to 203, rank 3 of 13. It beats every BlockGen arm. Pillars use the
  n=256 floor at n=174/183, so they are approximate.
- **Gate gap.** `voxel_dup_rate` checks only against the houses_32 train split (0.004), and
  `voxel_diversity` is a within-set mean (0.866 vs 0.898). Neither catches a set of ~160
  template-like modes. A within-set near-duplicate gate would. That is a bench change and
  needs the user's approval.

Consequence for the paper: "Dream-Cubed ranks 2nd" is not a quality claim. The paper's
Dream-Cubed text keeps its `\TODO` until the user picks a handling.

**20M SFT schedule (results T87, 2026-11-04).** A decayed 3k-step SFT reaches the same best
holdout as the 16k run (0.286), so the early overfit is not a learning-rate artefact. Mixing
`anneal` 1:1 into SFT keeps the holdout falling to the last step (0.263 at 6k) and gives the
best voxel board so far (332, `geom_kid` 15.9, the first voxel arm Holm-separated from 5M raw),
at a cost in appearance. Next: a longer mix and a world-crop-only mix, because `anneal` holds
the SFT houses.

**canon_oriented on the current card (results T93, 2026-11-05).** The legacy 3D-BPE oriented
model scores 297, ahead of every new-pipeline voxel arm (best: the 20M mix at 332). It ties the mix
on `geom_kid` (14.2 against 15.9, same Holm group) and wins on appearance (157), enterable (0.61
against 0.28) and floating components, at a slightly low nn percentile (0.43). The 5M
world-crop oriented against unoriented chain isolates the orientation lever.

**Orientation in the pipeline (results T94, 2026-11-05).** With world-crop-only 5M pretraining,
the oriented tokenizer ties the unoriented one on `geom_kid` and loses on appearance (best-of-4
351 against 358 on the board, but `mv_dino_kid` 0.368 against 0.310, Holm-separated). So
canon_oriented's lead comes from 3D-BPE or its houses-only schedule, not from orientation.
World-crop-only pretraining beats the old `anneal` pretraining at 5M raw (361 against 386).

**3D-BPE in the pipeline (results T96, 2026-11-05).** The same world-crop recipe with the pipeline
3D-BPE tokenizer scores 501 raw and 491 best-of-4 against 361 and 358 for voxels: `geom_kid`
24.1 against 17.3, Holm-separated. Pieces keep canon_oriented's side benefits (enterable 0.31
against 0.15, nn percentile 0.53 against 0.68) but not its geometry, so the lead points at the
legacy schedule (60 epochs on houses only, D4, dropout 0.1). The D4 pieces pair tests it.

**Long 20M SFT (results T97, 2026-11-05).** The 20M SFT and world-crop mix run to 16k steps picks
step 8000 and scores 333, the same as the 6k mix (332, same Holm group). More SFT steps do not
move the bench, and the world-crop tier halves enterable (0.15 against 0.28).

**The legacy schedule on pipeline 3D-BPE (results T98, 2026-11-05).** Pieces from scratch with D4,
dropout 0.1 and 60k steps reach 425 and match canon_oriented's enterable (0.60 against 0.61) and
nearly its novelty, but `geom_kid` stays at 20.4 against 14.2 (voxels 17.3). Starting the same run
from the world-crop pieces pretrain is worse (476). Left to explain the geometry gap: the oriented
legacy tokenizer (1,079 pieces) and the legacy batch of 8 sequences.

**First category-conditioned models (results T99, 2026-11-05).** A 5M voxel model trained from
scratch on 11 classes follows the class token: mean probe accuracy 0.33 against a real-build ceiling
of 0.69 (shuffled 0.09), best on tree, vehicle, pixel art and aircraft. Its house samples are as good
as the houses-only model's (355 at n=255 against 361, better appearance). Starting from the world-crop
pretrain halves conditioning (0.16) while lowering NLL, so the pretrained prior drowns out the class
token.

**BrickNet on the house card (results T100, 2026-11-05).** With the corrected readout (256 of 260
draws parse, collisions kept first node wins), mc_sft_base scores 640: worse appearance than shuffled
materials, no interiors (enterable 0.01), every build one solid clump. The node format is learned,
house shapes are not, which agrees with T69.

**Equal-n re-scores (results T101 and T102, 2026-11-06).** The legacy 3D-BPE arms at n=256 under the
current descriptor: canon_oriented 297, canon_unoriented 362, native_oriented 247 (contaminated).
Oriented beats unoriented on geom_kid by 3.1, Holm-separated, so orientation matters with the legacy
piece vocabulary even though it does not with pipeline voxels (T94). The cat scratch arm, redrawn to
256 with the top-up sampler, scores 355, the same as at n=255; every Holm group of T99 holds.
The facade arms on their first 256 builds (T103) keep every n=399 order; best-of-4 at T=0.7
scores 290, beside canon_oriented's 297 (geom_kid 13.9 against 14.2, better appearance, worse
traversal), with no paired test between the two cards.
On one card (T104) the two tie on geometry (shared Holm group with the facade ceiling); the facade
arm wins appearance, canon_oriented wins palette, coherence and traversal. About 12% of the facade
arm's builds are single-material boxes (canon 0.8%, real 1%), a T=0.7 effect that geometry and
appearance barely see.

Style-cluster subset (T110). The houses_32 style clusters (scripts/cluster_houses.py) are a continuum
with clean centres; both_k6 c1 is medieval cottages. Growing it with anneal builds inside the cluster
radius gives 1,881 training builds. Fine-tuning the 5M pretrain on that set, alone or mixed 1:1 with
world crops, scores worse against the cluster's own held-out builds than the all-houses mix (geom_kid
21.2 and 19.9 against 16.7; mv_dino_kid 1.14 and 1.24 against 1.09), and none of the three models
produces a build the clusterer places in the cottage cluster. The narrower data moved the palette, not
the structure: the generator's error lies off the style manifold, so subsetting by style does not
remove it. The pretrain-all pool assignment (larger cluster) is running; T110 suggests it will not
change this unless the extra data fixes structure.

canon_oriented recreated (T111). Rerunning the Sep 17 legacy command at two torch seeds (split pinned)
gives 313 and 319 against the original 297, all in one Holm group on both primaries. The best model is
a reproducible recipe, not a lucky checkpoint, and seed 1 has the best appearance of any learned arm (129).
World-crop pretraining in front of the same recipe (T112) gives 255, the best learned arm on every headline
column (geom_kid 12.2, mv_dino_kid 0.203, appearance 125). It reverses T98, where pretraining hurt pieces under the
pipeline tokenizer and trainer, so the legacy trainer and oriented vocabulary are what make the pretraining
transfer. One seed; Holm separates it only from the weakest scratch seed.
The second pretrained seed (T113) scores 358, the worst of the five legacy arms, and the two pretrained
seeds separate from each other on geometry. Pretraining widens the seed spread (255 to 358 against 297 to 319)
without moving the mean (306 against 310), so T112's gain was a seed effect. Geometry at this scale moves by
5 units between seeds; legacy-recipe comparisons need three seeds per arm.
Ladder P1 (T116): the pipeline with every legacy setting a config can express (dropout 0.1, constant
lr 3e-4, beta2 0.999, weight decay 0.01, last checkpoint, 104k steps) scores 338 against its defaults' 425,
ahead of the legacy unoriented arm (362) and tied with canon_oriented on both primaries. The trainer half of
the gap was optimizer and schedule settings; tags, token-budget batching and 2-D-only decay do not hold it
back. P2 adds oriented pieces.
The same legacy recipe at 20M (T118, d512 x 8, 27M parameters) scores 190 against 297 to 319 at 5M,
beating every 5M arm on geometry by 5 to 6 units, past the seed spread, and leading appearance and mv_dino_kid.
It sits closer to its training data (dino_nn_percentile 0.36 against real 0.51) without voxel duplicates.
Scale, not pretraining, is what moves this recipe. The second 20M seed (T124) replicates the geometry
(8.1 against 9.1, same Holm group) but fails the dino_nn_percentile memorization gate (0.351; seed 0's 0.359
passed), so 20M sits on the gate's edge: proximity to training data, no voxel copies.
The agentic and LLM one-shot arms at n=256 on 256 distinct prompts (T125, $15.66) all fail the twin gate
(29 to 41% near-copies, real 0.8%): the LLMs reuse one template across different requests, so T51's twins
were not a repeated-prompt artifact. Ungated, dream-caption ties the 5M learned arms on geometry and beats them
on mv_dino_kid (0.223); one-shot and classic-matched trail every learned arm.
Ladder P2 (T119): P1's knobs on oriented pieces (`learn_pieces --oriented`, 78a45c76538c) score 330, tied with
canon_oriented on both primaries and equal on appearance and mv_dino_kid. The pipeline gap is closed: trainer
settings plus oriented pieces account for all of it. The pipeline defaults (WSD, beta2 0.95, wd 0.1, dropout 0,
select=best) should change to these for houses_32 SFT.
Category conditioning on the legacy recipe (T121): class token in BOS's slot, sft-cat fit32 with D4 and an
oriented piece vocabulary learned on it. Probe accuracy 0.53 (ceiling 0.69) against T99's 0.33, every class
better, margin 13.5 against 8.0. Its house-token samples score 272 beside canon_oriented's 297 in one Holm
group, so mixing in ten other classes costs houses nothing and pulls training-set proximity to real's level.

## §74b. Hollow-Shell rerun under the post-fix renderer (2026-11-04)

Replaces the unregistered pre-fix run `run_20260918_203947_bench_hollow`.
`scripts/bench_hollow.py --descriptor v1` (77-d, 0925 protocol), seed 0, 399-build reference.

| run | n | MV-KID hollow | MV-KID solidify | V-KID hollow | V-KID solidify |
|---|--:|--:|--:|--:|--:|
| `run_20261104_131723_bench_hollow_v1_n128` | 128 | -0.02 | +0.19 | +156.0 | +286.7 |
| `run_20261104_131806_bench_hollow_v1_n256` | 256 | -0.02 | +0.21 | +231.1 | +401.7 |

(Units: noise floors above the real_test draw.) The geometry numbers at n=128 are identical to
the old run, as expected, because they don't depend on the renderer. The render number moves from +0.4 to
-0.02, deeper inside real's interval. Hollowing removes a median of 14.5% of blocks (98% of builds
are affected). This script's floor (separation above a real_test draw) differs from the
leaderboard's, so compare these numbers within the run only. Example: solidify at n=256 is 401.7
here and 444.4 on the leaderboard.

## §75. LEGO in-domain ladder rerun under the post-fix renderer (2026-11-04)

Fig. 6 (`fig:seven`) and `tab:seven` read the LEGO ladder of `run_20260918_200852_bench_lego_interior_blindness_replication`,
which predates the 09-25 transparency-cull fix. Rerun unchanged, with the 77-d v1 descriptor, into
`outputs/run_20261104_212706_bench_lego_ladder_postfix` (log, samples.png: real / Fill-Interior / Hollow-Shell):

    python -c "geometry.set_descriptor_version('v1'); ladder.main()" --corpus lego_void_20 \
        --family both --n 128 --n-ref 256 --no-legacy --out-root <run>/ladder

The render scores are identical to the 09-18 ladder to a maximum relative difference of 5e-6. The geometry
ladder JSON matches byte for byte. Every `tab:seven` cell is unchanged:

| statistic | Fill-Interior | 20% deleted | Material-Shuffle |
|---|--:|--:|--:|
| KID | 2.0 | 12.3 | 132.3 |
| MMD-RBF | 2.1 | 13.9 | 148.8 |
| Fréchet | 0.5 | 10.8 | 72.4 |
| 1-coverage | 0.2 | 7.7 | 19.8 |
| 1-density | 0.1 | 7.2 | 14.0 |
| 1-precision | -0.4 | 9.0 | 28.0 |
| 1-recall | 0.4 | 1.4 | 30.7 |

The gate verdicts are also unchanged: kid and 1-density pass; mmd_rbf fails G7, fd / 1-coverage / 1-recall fail G8, and 1-precision fails G2.
This is the expected result. LEGO builds are all wool, which is opaque, so the transparency cull cannot change a
LEGO render. `fig6` and `audit_claims` now point at the new run, and fig6 was regenerated with the svg added.
The BrickGPT scoring (`run_20261102_220228_brickgpt_lego`) was run after the fix on the same wool-only corpus, so
it does not need rescoring.

## §76. Facade and cross-section views for MV-KID (2026-11-04)

Run `outputs/run_20261104_213620_view_facade_section` (script
`scripts/ablate_view_facade_section.py`). This answers two reviewer questions: would
architectural views see the interior, and would cross-sections see it? The setup is the
leaderboard of record exactly:
- every arm reproduces its `structures_sha` from the leaderboard of record;
- 399 val builds as reference, DINOv2-B/14, mean-pool, unbiased KID;
- noise floors are the sd of 48 real draws from the 399 test builds.

Protocols:
- **corner** (paper): 4 corner views.
- **A-sides**: 4 side elevations at el 0.
- **A-render**: the 4 elevations plus a roof plan.
- **A-2.5D**: the five facade images (first block + depth, `blockgen.corpus.facade`), with no
  renderer.
- **B-mid**: one-voxel mid sections of x, z, y, face-on.
- **B-9**: sections at 25/50/75% of each axis.
- **B-cut**: half the build removed along x, z, y, seen at the cut.

n=256, noise floors above real_test:

| arm | corner | A-sides | A-render | A-2.5D | B-mid | B-9 | B-cut | V-KID |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Fill-Interior | +0.29 | +4.45 | +3.62 | 0 (exact) | +71.1 | +67.5 | +14.3 | +444.4 |
| Hollow-Shell | +0.03 | +0.82 | +0.60 | 0 (exact) | +39.7 | +42.8 | +5.3 | +265.4 |
| Material-Shuffle | +346 | +548 | +610 | +355 | +226 | +292 | +415 | 0 |
| Monochrome | +439 | +1146 | +1309 | +314 | +1080 | +1314 | +925 | 0 |
| rot90 / mirror | -0.24 / -0.15 | +0.04 / +0.14 | +0.08 / +0.06 | +0.10 / +0.01 | +0.41 / -0.40 | +0.49 / -0.26 | -0.25 / +0.01 | 0 / 0 |
| Spearman vs corner (11 arms) | 1 | 0.97 | 0.97 | 0.96 | 0.93 | 0.93 | 0.95 | 0.27 |
| Spearman vs V-KID | 0.27 | 0.10 | 0.10 | 0.28 | 0.07 | 0.07 | 0.05 | 1 |

Findings:
- **The 2.5D facade is exactly blind.** The facade images of Fill-Interior and Hollow-Shell
  are bit-identical to real for 256/256 builds, so the features are identical and the
  separation is exactly 0. This is the proposition with no renderer in the way: the facade
  representation counts glass as opaque.
- **Rendered elevations leak a little more than corner views.** Fill-Interior reads +4.45 at
  n=256 (+2.29 at n=128), over the 3-floor bar at n=256. Hollow-Shell is +0.8. This is window
  leakage seen face-on, still 100 times below V-KID's 444. It is consistent with "attenuated,
  not invisible" (89% of test houses leak through a window).
- **Cross-sections see the interior**, but at 1/6 of V-KID: B-mid +71 / +40, B-cut +14 / +5.
  Their arm ordering stays an appearance ordering: Spearman 0.93 with corner, 0.07 with V-KID.
  Monochrome and shell_box dominate. Showing the inside to an image encoder recovers the
  signal, but the metric stays texture-led.
- **Invariance:** every protocol is within about 0.5 floors under yaw-90 and mirror. Axis-aligned
  pooled views are invariant in practice.
- **Calibration check:**
  - V-KID column: reproduces the leaderboard exactly (444.36; sd 5.33e-2).
  - Corner MV-KID: Fill-Interior reads +0.29 here vs 0.354 on the card. This run's noise sd is
    1.96e-3 vs 1.61e-3, because the card draws its MV-KID floor from the RNG stream after the
    geometry floor.
  - n=128 column: uses the first 128 builds of each n=256 arm, which equal the 0925 n=128
    draws (V-KID Fill-Interior 201.80, same as the paper).
- **Paper impact:**
  - No headline claim changes.
  - "Up to 16 views" now has a caveat: face-on elevations cross 3 floors at n=256 through
    windows.
  - Cross-sections join cutaways as positive controls.
- Draft: `appendix_draft.tex` in the run dir.

## §77. Minecraft render ladder rerun under the post-fix renderer (2026-11-04)

`outputs/run_20261104_221808_bench_mc_ladder_postfix` (`python -m blockgen.eval.bench.ladder
--family render --n 64`, same settings as the 2026-08-31 file in `outputs/analysis/ladder`, which
the bench still reads and was left untouched). Unbiased KID and MMD-RBF still pass every gate,
and the other metrics still fail, so no leaderboard ranking changes. One verdict moved:
1-recall now passes n-stability but fails noise ordering (G2) and invariance (G6), where the
Aug-31 run had it failing n-stability only. The paper now cites the post-fix file: `tab:ladder-render`, the
intro and the method sentence (recall moved from the "fails n-stability" list to the "fails
monotonicity" list), and `audit_claims.LADDER`. Cell values moved by up to ~10% (monochrome,
precision on material shuffle), as expected since glass now renders correctly.

## §78. A within-set near-twin gate, and what it does to the leaderboard (2026-11-04)

User decisions 8a and 9 (2026-11-04): keep Dream-Cubed's row, add the vanilla-template
control row, and add a gate for builds that repeat each other within a set.

**The gate.** `blockgen/eval/bench/twins.py`. For each build, its nearest other build in the
same set, at the best occupancy IoU over the 8 yaw/mirror variants and every integer
translation (exact, scipy FFT cross-correlation, CPU). Pairs that cannot reach the threshold
by size alone are skipped, and exact repeats are keyed without an FFT. The statistic is
`voxel_twin_rate`, the share of builds with a twin at IoU >= 0.8. It is computed in the fast
tier with its own bootstrap generator, so no other number on a pinned-protocol card moves.
The gate in `composite.GATES` uses the duplicate gate's rule, fixed before Dream-Cubed was
scored: it fires at more than 3 spreads above the real control of the same run, with the
spread floored at 0.02. At n=256 the bar is 0.0078 + 3 x 0.02 = 0.068. Cards without the
metric are not gated. Tests: `tests/eval/test_twins.py` (rotated, mirrored and shifted
copy; brute-force IoU check; repeats; gate fires; old cards skipped). 842 eval and lab tests
pass.

**Re-score of the leaderboard of record.** `scripts/bench_twin_rescore.py` writes
`outputs/run_20261104_214146_bench_twin_gate_rescore`. Arms are rebuilt and verified
against the card's `structures_sha`; only the `voxel_twin_*` metrics are new.

| arm | twin rate | median NN IoU | status |
|---|--:|--:|---|
| real_test (and its material probes) | 0.008 | 0.42 | ok |
| real@solidify | 0.066 | 0.62 | ok (z 2.9, just under the bar) |
| train_verbatim | 0.023 | 0.43 | DQ (memorization, as before) |
| canon/native AR arms | 0.000 | 0.39 | ok |
| patchwork, uniform_random, train_copy_noise | 0.000 | | ok |
| gabled_house | 0.207 | 0.66 | **DQ +10.0** |
| shell_box | 0.246 | 0.70 | **DQ +11.9** |
| dreamcubed | 0.598 | 0.85 | **DQ +29.5** |
| dreamcubed_raw | 0.711 | | DQ (diversity, as before) |

Null of fresh real draws (n=256, 5 draws each): test 0.006, val 0.008, train 0.019 (max
0.031). New ranks: real 1, native_oriented 2 (contaminated, unranked under the protocol),
canon_oriented 3, canon_unoriented 4, shuffle / patchwork / solidify tie at 5–7,
monochrome 8, train_copy_noise 9, uniform_random 10. Self-validation passes (6 of 6).
samples.png shows the top twin pairs per arm. The gabled and shell_box twins are the same
shape in different materials, because the gate reads occupancy only.

Caveat: solidify sits at z 2.93, just under the bar of 3. Filled builds are more alike
than hollow ones. A larger n or a different seed could move it across.

**Vanilla-template control on the GPU path.**
`outputs/run_20261104_213125_bench_vanilla_template_control`: the same protocol as the
leaderboard of record. The noise floors are identical, and Dream-Cubed reproduces 165.71
exactly.

| arm | n | MV-KID raw | V-KID raw | appearance | geometry | ungated BlockScore | twins | status |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| vanilla_village_houses | 174 | 0.178 | 10.67 | 110.21 | 202.95 | 202.95 | 0.672 | DQ +33.2 |
| vanilla_village_all | 183 | 0.176 | 10.25 | 108.63 | 194.98 | 194.98 | 0.683 | DQ +33.8 |

Ungated, the templates would rank ahead of every model we trained (native 229.29, canon
285.22). The peer's CPU replica numbers (195 and 203) hold.

**Paper.**
- `6_results.tex`: the Dream-Cubed paragraph is rewritten and its `\TODO` removed.
- The leaderboard table:
  - Dream-Cubed, the templates, gabled and hollowbox move into the DQ block, with their
    twin rates and pillars shown.
  - The ranks are renumbered.
  - The caption defines "twins".
  - The pillar sentence now names Patch-Stitch rather than Dream-Cubed.
- `4_method.tex`: one sentence on the within-set test.
- `X_appendix.tex`:
  - In app:dreamcubed, the `\TODO` is removed and two paragraphs are added: the gate and
    the audit numbers.
  - The tab:n256 rank column is updated, plus its note.
  - The smallest-score sentence now reads 229.29.
- The audit registers the claims: 778 of 778 match, 0 not found.

## §79. Twin gate: probes exempt, real-data validation test (2026-11-04)

The user decided that diagnostic probes of real builds (`composite.is_probe_arm`: `real@*` and
`real_shuffled_materials`) are exempt from the within-set twin gate. `real@single_mode` is the
gates' positive control and stays judged. Fill-Interior had passed at 2.93 spreads against a
bar of 3, because filling interiors makes real houses more alike in shape. No leaderboard row
changes. New tests in `tests/eval/test_twins.py`:
- the exemption;
- a slow real-data check. Three fresh n=128 test draws pass the bar that the val split
  calibrates. Sixteen real builds, each copied at 4 rotations x 2, are disqualified.

Paper upkeep in the same pass:
- seven of the nine unregistered main-text numbers are now in the audit (785 claims);
- the main text is back to 8 pages after the twin-gate text, with Dream-Cubed and the
  conclusion trimmed;
- the private reading notes in the `.bib` files (`note=`) are renamed to `annote=`, so the
  bibliography stops printing them.
Still unverified: the window-leak "median 1.75% of pixels" (T37, 2026-09-20), which may predate
the transparency-cull fix.

## §80. V-KID against the structural metrics a reviewer would name (2026-11-05)

Run `outputs/run_20261104_235615_bench_headtohead` (`scripts/bench_headtohead.py`).
Every alternative goes through the same tests V-KID passed:
- **Ladder:** houses_32 ladder subset, n=128, 48 noise reps, and `ladder.score_rungs`, the code that
  gates the leaderboard. Geometry-family rungs apply to all rows. The colour-reading encoders are
  also judged on the render-family rungs.
- **Board:** the n=256 arms of the leaderboard of record. Build hashes are checked, and the
  leaderboard's noise-floor rule is used.

The V-KID row reproduces the leaderboard exactly (+444.4; Spearman 1.00 with the view run's V-KID).

| row | Fill (ladder n=128) | Hollow | Shuffle | Monochrome | rot90 | gates | Fill (board n=256) | rho V-KID | rho MV-KID |
|---|--:|--:|--:|--:|--:|---|--:|--:|--:|
| **V-KID** | **+111.4** | **+71.0** | 0 | 0 | 0 | **PASS** | **+444.4** | 1 | 0.27 |
| occupancy KID (4^3-pooled grid) | +48.4 | +13.1 | 0 | 0 | +1.5 | G2, G6 | +121.2 | 0.83 | 0.23 |
| Uni3D KID, all faces | +26.6 | +10.4 | +22.4 | +99.7 | +1.5 | G6 (both specs) | +31.8 | 0.51 | 0.85 |
| Uni3D KID, volume points | +37.2 | +10.6 | +6.7 | +105.5 | +0.9 | G6 | +108.4 | 0.59 | 0.78 |
| ULIP-2 KID, all faces | +4.6 | +1.8 | +29.1 | +101.8 | -0.1 | G6 geom / G2 render | +12.0 | 0.42 | 0.87 |
| ULIP-2 KID, volume points | +11.9 | +9.0 | +9.7 | +62.8 | +0.2 | G6 geom / G2 render | +43.2 | 0.51 | 0.85 |
| chamfer MMD, all faces | +7.6 | +1.4 | -0.1 | -0.1 | -0.4 | G2, G8 | +8.9 | 0.64 | 0.27 |
| chamfer 1-NNA, all faces | +2.9 | +1.1 | 0 | -0.6 | +0.5 | G2, G6 | +4.0 | 0.76 | 0.43 |
| EMD MMD, all faces | +3.2 | +0.3 | -0.2 | -0.3 | -0.6 | G2, G8 | (ladder only) | | |
| EMD 1-NNA, all faces | +1.6 | +0.1 | -0.5 | -0.2 | -0.1 | G2 | | | |
| chamfer MMD, voxel centres | **-3.4** | +2.8 | +0.1 | +0.1 | -0.2 | G2, G8 | -3.5 | 0.53 | 0.54 |
| IoU MMD, D4-aligned | **-9.0** | +7.7 | 0 | 0 | 0 | G8 | -5.3 | 0.36 | 0.35 |
| IoU 1-NNA, D4-aligned | +2.7 | +2.4 | 0 | 0 | +0.2 | G2 | +5.9 | 0.75 | 0.28 |
| IoU MMD, corner (T67) | **-11.5** | +8.4 | 0 | 0 | +2.6 | G6, G8 | -7.4 | 0.52 | 0.46 |

All values are noise floors above held-out real. COV rows fail at least two gates each; see results.json.

**What it shows**
1. **Only V-KID passes every blocking gate.** It also separates both interior probes most strongly:
   2.3× the next row on Fill-Interior (occupancy KID) and 5× on Hollow-Shell.
2. **The 3D encoders see the interior once their points include it,** but they are not geometry
   metrics:
   - Material-Shuffle and Monochrome move them by +7 to +106 floors.
   - Uni3D is not yaw-invariant: rot90 moves it +1.5 floors at n=128 and +5.7 at n=256 on the volume
     samples.
   - Their arm ordering follows MV-KID (0.78-0.87), not V-KID.
3. **The voxel-generation suite (chamfer/EMD/IoU MMD, COV, 1-NNA) never separates the interior probes
   by more than single digits, and none of it passes the gates.** All of it fails noise ordering or
   n-stability. Nearest-neighbour MMD under IoU or volume chamfer rates Fill-Interior *closer* to real
   than real itself (-9 to -11.5 floors at n=128): a filled house overlaps its neighbours more. The
   field's standard voxel metric would therefore reward a generator for filling its rooms.
4. **Occupancy KID** (same estimator as V-KID, raw grid as features) is the strongest alternative.
   - It separates Fill-Interior (+48 / +121).
   - Its board ordering agrees with V-KID (0.83).
   - It fails noise ordering, and its rot90 reading sits at the edge of the invariance bar
     (+1.5 / +2.5) because a centred grid is not yaw-invariant.
   - So the descriptor, not the estimator, is what V-KID adds.
5. **Cost.** V-KID takes 15-18 s on the CPU for ~3-4k builds; the encoders take 25-50 s of GPU each
   after sampling; chamfer takes 2-3 min of GPU; exact EMD took 32 min on 16 CPU cores (ladder only).

**Caveats**
- The gates and V-KID come from the same project. The gates are borrowed from the image-metric
  literature, but V-KID's descriptor was chosen while they existed.
- The GPT-4o exterior Bradley-Terry ranking agrees with no structural row (|rho| <= 0.42 over 9 arms).
  It judges exteriors, so it is not a validity test here.

**Paper claim this supports:** V-KID is the only structural metric tested (14 rows from four families)
that passes every validation gate, and it separates both interior probes most strongly. The standard
voxel-set metrics either miss the probes or score the filled corpus as more realistic than real data.

## §81. ldraw-nova, and an inspect-and-repair stage for Track E (2026-11-05)

The user found ldraw-nova (github.com/anteloc/ldraw-nova; AGPL-3.0, 520 stars, five weeks
old, no paper, no metrics). It is not a learned model: a frontier LLM writes a plan JSON and
a Python generator that emits an LDraw .mpd, around a tool loop of brief, retrieve (curated
knowledge base plus example and part search), build, inspect (headless renders, collision,
gap and contact checks), repair, deliver. That loop is Track E's loop, so there was nothing
to port or benchmark as a model; what Track E lacked were two of its stages.

**What was missing.** `turns.repair` reports what the executor saw (failed lines, no-op
commands) and nothing about what was built. The only signal about the built geometry was
the render critique, which costs images. And the only knowledge the builder gets is the
fixed in-context example.

**Built.** `blockgen/agentic/inspection.py`: eight deterministic checks on the voxel
world, each a pass or a failure with severity and a fix hint, rendered as a text block
(`turns.inspect`): grounded, connected (with the position of the largest stray piece),
supported (nothing below in the column and no supported neighbour beside), interior
(enclosed air; a solid mass named separately), roofed (floor area under open sky),
enterable (the `eval/traversal` player walk; unreached storeys), headroom (3 clear blocks),
openings (glazing or an aperture over a sill), plus size and built. `steps.inspect_repair`
runs rounds until the build is clean at `clean_at` severity; method `inspect` = classic
with inspect rounds between repair and critique. Scorer `inspection` = share of applicable
checks passed, so `accept: improve_valid` and `select: best` keep the build that passes
most. `MethodResult.metrics()` now reports `inspect_pass` and `inspect_enterable` for every
method. `lessons` context: `configs/agentic/lessons.yaml`, thirteen hand-written rules with
tags, the `always` ones plus the k best tag matches per request, and a shuffled
token-matched control. Recipes `inspect`, `inspect-critique`, `inspect-lessons`,
`inspect-lessons-shuffled`, `classic-lessons`; battery `inspect-v1` (six arms, n=12,
estimate $3.70 upper bound). Docs: agentic-suite.md §5.7. Tests: 15 in
`tests/agentic/test_inspection.py`; the agentic suite passes (1246).

**Calibration on real houses.** The first support rule (nothing below in the column)
flagged 44% of 400 real houses_32 builds above 15%, because corpus crops often lack their
floor so every roof block over a room "hangs". The rule now forgives a block with a
supported horizontal neighbour (p90 18%, threshold 25%, 4% of real houses exceed it).
Stray pieces: 21% of real houses have some, 6% above 2%, 3% above 5%, so `connected` is a
note under 2%, a warning to 5%, an error above. Real houses: mean score 0.934, 55% clean at
warning.

**What the inspector says about the T51 arms** (the n=133 builds behind the agentic bench
row, results T95): the agentic arms pass every structural check that real houses fail
(supported, connected, roofed) and fail the one real houses pass: enterable 33% (classic-
matched), 54% (dream), 57% (dream-caption) against 17% for real. Headroom follows the
dream arms (7%, 13%). So the inspector's value for this loop is almost entirely the door
check, which is the traversability gap T63 measured, and the pilot's first number to read
is `inspect_enterable` per arm.

**Trap.** `suite.estimate` injects `stop_at=inf` and `patience=10**6` into any method whose
signature has those names (dream's score threshold). The inspect method's severity knob
was first called `stop_at` and every estimate build errored; it is `clean_at`.

**Not done.** The paid pilot (`battery inspect-v1 --allow-spend --budget-usd 6`) waits for
authorization. Running ldraw-nova itself as an external LEGO generator on the BrickBench
task is deferred past the CVPR deadline (research §E.1 addendum).

## §82. V-KID robustness: feature families, random subsets, hand-design-free descriptors, global terms (2026-11-06)

Run `outputs/run_20261106_193358_vkid_robustness` (`scripts/vkid_robustness.py`, CPU only, about 3 h of which 2.8 h is board CIs). It uses the
§80 protocols: ladder n=128 with `ladder.score_rungs` and geometry rungs, and the n=256 leaderboard
of record (build hashes match). Every row is whitened on the reference and scored by the same KID.
rho is Spearman over the 9 structural board arms against full V-KID. Full table: `outputs/run_20261106_193358_vkid_robustness/results.md`.
Draft supp section and main-text sentence: `outputs/run_20261106_193358_vkid_robustness/supp_vkid_robustness.tex`.

| descriptor | dims | gates | Fill (board) | Hollow (board) | Patch-Stitch | rho |
|---|--:|---|--:|--:|--:|--:|
| full V-KID | 77 | PASS | 444.4 | 265.4 | 368.2 | 1.00 |
| leave one of 7 families out (range) | 22-76 | 7/7 PASS | 310-457 | 146-270 | 327-375 | 0.95-1.00 |
| random 50% subsets, median [min] | 38 | 20/20 PASS | 351 [283] | 221 [133] | 317 [230] | 0.97 [0.88] |
| random 25% subsets, median [min] | 19 | 17/20 PASS | 298 [179] | 153 [56] | 246 [178] | 0.95 [0.85] |
| random 3D CNN, 3 weight draws | 224 | 3/3 PASS | 339-437 | 224-297 | 278-310 | 0.73-0.80 |
| hashed raw 3^3 patterns | 256 | PASS | 256.7 | 208.9 | 326.0 | 0.75 |
| multi-scale 3^3 neighbour counts | 168 | PASS | 359.3 | 166.3 | 514.9 | 0.80 |
| V-KID + LCC ratio, component count | 79 | PASS | 454.6 | 271.4 | 366.1 | 1.00 |
| V-KID + 2x,4x pooled orbit histograms | 187 | PASS | 485.3 | 253.4 | 354.1 | 1.00 |
| LCC terms alone | 2 | PASS | 0.0 | 0.1 | 258.4 | -0.20 |

Families are pattern (55), thickness (5 bins + mean + p90), height (8 bins + entropy), surface
(surface/volume, wall fraction), symmetry, interior (sealed, open), size. Generic rows are
f(x) = mean over D4 of phi(gx), with phi a fixed generic map. A unit test confirms exact invariance
(CNN to 2e-7).

**What it shows**
1. **No feature family carries V-KID.** Removing any family keeps every gate. Removing thickness costs
   the most Fill-Interior (444 to 310). Removing pattern orbits costs the most Hollow-Shell (265 to 146).
   Single families alone mostly fail G2: height, surface, symmetry, interior and size do; pattern and
   thickness alone pass. The descriptor works because its families are redundant, not because one
   feature targets one probe.
2. **A descriptor with no hand design passes every gate.** All six generic rows pass. They separate
   Fill-Interior by 257-437 board floors (V-KID 444) and 105-175 ladder floors (V-KID 111).
   - §80's occupancy KID failed G2 and G6. Orbit averaging plus local pooling is the difference.
   - So the paper can say the result comes from observing the volume, and V-KID is one valid
     instantiation.
3. **Generic rows agree on probes and baselines but not on the learned arms.** Over the 5
   probe/baseline arms rho is 0.8-1.0. Over the 4 learned arms it is -0.4 for the CNNs, which rank
   Dream-Cubed behind canon/native. The hashed 3^3 row agrees with V-KID (1.0). Rankings of close
   learned arms depend on the descriptor; probe verdicts do not.
4. **Global terms do not single out Patch-Stitch.** Appended to V-KID they keep every gate, but
   Patch-Stitch stays 4th of 9 and its ratio to Fill-Interior stays 0.73-0.81 (V-KID 0.83).
   - LCC alone sees Patch-Stitch (258) and not the interior probes (0). It also scores
     canon_oriented at 308, so learned arms fragment too.
   - At 2 of 79 dims it is diluted in the joint kernel.
   - Multi-scale neighbour counts are the most Patch-Stitch-sensitive full descriptor (515, ratio 1.43).
   - Reported only; the paper descriptor is unchanged.

**Caveats**
- LCC alone passes every blocking gate while separating no interior probe. The gates test validity,
  not coverage, so passing them is not evidence of sensitivity.
- Board CIs for the random subsets use 20 resamples rather than 100.

**Paper claim this supports:** V-KID's interior result survives the removal of any feature family and
of random half of its dimensions. Generic, orbit-averaged random features pass the same gates, so the
result comes from reading the volume, not from feature engineering. Global coherence is a separate
axis, which we state as scope.

## §83. The blindness on real generators: 40 generators, two rankings (2026-11-06)

Run `outputs/run_20261106_193513_render_blind_rank_flips` (results T106). Every house generator
scored against the 399-build val reference under the post-fix renderer was pooled: 22 autoregressive
voxel arms, 10 facade arms, 5 LLM program arms, canon_oriented / canon_unoriented, BrickNet (40;
35 at n=256). V-KID was recomputed exactly (v1 77-d of the leaderboard of record, and v2), and it
matches every card to 1e-12. MV-KID is each card's value and interval, and its replicates are
independent, which makes MV separations conservative. Pairs are Holm-corrected over 780.

- **The rankings diverge far beyond noise.** Spearman MV vs V-KID v1 0.50 [0.42, 0.54]; v2 0.65.
  Each metric against a resample of itself: 0.98 to 0.99. Within the 24 autoregressive voxel arms the
  v1 value is 0.37.
- **76 of the 388 pairs that both metrics separate are ordered oppositely** (v2: 46 of 405).
  MV prefers a facade arm in 38 of them.
- **Facade arms are shells.** Median hidden-block share is 0.5% (real 15.6%, voxel arms 12.5%).
  They lose 7.7 places on average from the MV ranking to the V-KID ranking. LLM program arms are
  also near-shells but move up, so V-KID is not a hidden-block counter.
- **Key pair (samples.png).** `rft_facades_joint_d4_20m_face_solidity` vs `m2b_20m_sftmix_s1`:
  - MV difference -3.1 [-20.6, +14.5], a tie;
  - V-KID difference 292.9 [268.1, 317.2];
  - hidden blocks 0.2% vs 12.6%.
  The cutaways show walls around an empty floor.
- **Facade ceiling.** real@facades6 is MV +8.1, rank 1 of 41, and V-KID +310.6, rank 19.
- **Probe on output.** Hollowing the 20M voxel arm's outputs:
  - V-KID +147.5, rank 5 to 29;
  - on the same 128 builds, MV-KID (osmesa, all arms rendered alike) +0.15 floors and V-KID +117.1;
    for canon_oriented, +0.18 and +139.2.
  Hollowing the over-solid T=0.7 arm moves V-KID 24.7 closer to real.
- **34 design decisions.** 13 agree, 0 opposite, 12 MV-only, 3 V-KID-only, 6 neither. For example,
  3D-BPE pieces are V-KID +162.6 and MV n.s.; D4 at 20M facade is MV -52.4 and V-KID n.s.

Caveats:
- One project's generators.
- v2 halves the reversal count.
- The best facade arm (T=0.7) has the lowest V-KID of the pool despite an empty interior.
- Exact paired DINO features were stopped after 23 arms to free the GPU lock. On those arms
  MV-KID matches the cards to 1.6e-5. The cache is resumable (`extract_features.py`, then `analyze.py`).

Draft paper text (intro pre-emption, Sec. 6.4 paragraph, supp section `app:rankflips`):
`draft_text.tex` in the run dir.

## §84. Gate provenance, threshold dates, threshold robustness (2026-11-06)

Run `outputs/run_20261106_193109_gate_provenance` (results T107). Answers the circularity charge
(V-KID and the gates come from one project).

**Provenance.** Every gate restates a published criterion: ordering (Heusel 2017 App., Binkowski
2018, O'Bray 2022, Thompson 2022, CMMD), invariance (Borji 2019/2022, Kynkäänniemi 2023, Parmar
2022, CheckList), self-consistency (Naeem 2020, unbiased KID, C2ST/1-NNA null), n-stability
(Binkowski, Chong & Forsyth, CMMD, FLD), memorization (Meehan 2020, Alaa 2022, FLD, Stein 2023,
Somepalli 2023), twins (FLD's approximate-copy test). Three new bib entries in the run dir's
`gate_provenance.bib` (ribeiro2020checklist, borji2022newdevelopments, somepalli2023replication).

**Dates.** Bars set 2026-08-04 (a9eb431, T23), before any voxel metric. V-KID enters 2026-08-31
with two denominator corrections; the noise-floor one flipped V-KID's G7 from fail (3.5 sd) to
pass. Gate code frozen 2026-09-07 (16b95f8; later diffs touch only naming and a CLI flag), with the
v1 descriptor and the novelty bars. T92 ran 58 days later on that code.

**Robustness.** CPU rerun of V-KID, occupancy KID and six IoU rows reproduces T92 (2e-6, same
gates) and records G7/G8 statistics. The GPU rerun OOMed after its 97-min EMD stage against an
external 14.6 GB process, so chamfer/EMD/encoder rows keep their T92 G7/G8 verdicts.
- V-KID is the only geometry pass at every ordering tolerance <= 0 and every G6 bar 0.5-5, and at
  every G7 bar >= 2 and G8 bar 1-5. It passes all 360 of 480 joint settings with G7 >= 2.
- G7 bar 1 rejects everything, V-KID included (1.22 sd).
- The sensitive gate is G2's strict inequality: alternatives fail it by ties of 0.2-0.7 floors.
  A tolerance of 0.5 floor admits EMD and IoU-D4 1-NNA (Fill +1.6/+2.7); adding a G6 bar of 2
  admits occupancy KID (+48.4).
- Encoders fail geometry G6 by 63-106 floors and IoU MMD fails G8 by 34-36 SE, so no threshold
  rescues them.
- Novelty bars: positive controls, Dream-Cubed and shell_box are DQ in all 9 settings
  (z 2/3/5 x floor 0.5/1/2). The loosest corner readmits gabled_house. z=2 first DQs our own
  canon AR arms.

Drafts for the lead (not integrated): `draft_gate_provenance.tex` (supp section app:gateprov +
one main-text sentence for §4.3), figure = `samples.png`.
Open: verify Xu et al. 2018's transformation-robustness experiment in the PDF body (the abstract
lists only discriminability, mode dropping/collapse and overfitting).

## §85. The leak remeasured under the current renderer, the opaque rescore at n=256, and the leak dose curve (2026-11-06)

Run `outputs/run_20261106_193345_leak_opaque_dose` (scripts in the run dir: `leak_render.py`,
`leak_score.py`, `determinism.py`, `make_samples.py`, `build_results.py`; results T108). The paper's
"91% of views, median 1.75% of pixels, median max channel 212/255" came from the T37 census
(2026-09-20), before the transparency-aware face cull. Remeasured on all 399 test houses x 4 views
through `features.render_views` on EGL, with three counterfactual pipelines: *translucent_fixed*
(one texture atlas for the whole experiment, so no UV moves when interior faces disappear),
*opaque_perbuild* (glass alpha forced to 255, per-build atlas: the atlas artefact alone) and
*opaque_fixed* (both off: the lemma's setting). Reference = 399 val houses, DINOv2-B on the four
corner views, n = 256, 48 noise draws, the draw of `scripts/figs/rescore_flat_opaque.py`.

| Fill-Interior, 1,596 views | views altered | houses | median px (altered) | median max ch |
|---|--:|--:|--:|--:|
| deployed | 90.9% | 92.2% | 1.23% | 128 |
| translucency only (atlas fixed) | 58.7% | 67.2% | 0.21% | 107 |
| atlas artefact only (glass opaque) | 91.9% | 93.5% | 0.99% | 79 |
| glass opaque, atlas fixed, EGL | 11.1% | 20.1% | 0.008% | 1 (max 2) |
| glass opaque, atlas fixed, CPU (osmesa) | 0 / 1,596 | 0 | - | - |

- **91% stands; 1.75% and 212/255 do not.** Under the current renderer the median altered view
  changes 1.23% of its pixels at a median max channel of 128/255. "89% of houses leak through a
  window" becomes 67% once the atlas artefact is removed (92% with it).
- **The opaque setting is exact.** With glass opaque and a fixed atlas, Fill-Interior and
  Hollow-Shell are bit-identical in all 1,596 views on the CPU rasterizer. MV-KID at n = 256 reads
  -0.00004 (Fill) and -0.00002 (Hollow) floors on the GPU path; the GPU rasterizer is not
  bit-reproducible (the same house rendered twice changes up to 9 px by 1/255 in 6 of 24 houses),
  which is the whole residual. Deployed: +0.29 / +0.03; translucency only: +0.40 / +0.13.
- **Dose curve.** Leak = share of a house's silhouette pixels (4 views) changed with the atlas
  fixed. Bins: no leak (131 houses), then quartiles (67 each; top quartile median 4.4%, max 40%).
  Each house's pooled feature moves with its leak (Spearman 0.96; median shift 0 / 0.004 / 0.026 /
  0.062 / 0.106 against a median real-to-reference nearest-neighbour distance of 0.45), yet the
  per-bin separation is +0.00, -0.00, +0.03, +0.40, -0.66 floors (deployed renderer: -0.08 to
  +0.27, then -1.36 in the top quartile). The leak reaches single images and does not reach the
  set statistic; the leakiest filled houses move toward the reference rather than away.
- **LEGO** (lego_void_20, wool only): deployed Fill-Interior -0.011 floors (the paper's -0.01),
  74% of views altered, all of it the atlas artefact; with the atlas fixed -0.00004.
- **osmesa is not the bench renderer.** It returns alpha 255 everywhere, so the white-background
  composite of `features.render_views` becomes black and translucent pixels blend over black; its
  census (kept in `osmesa_census/`) reads 56% of views altered. Features for the bench must come
  from EGL. The bit-identity statement uses osmesa only because identity is a property of the
  geometry and the atlas, not of the background.
- Silhouette: the outline of a build cannot change (sealed cells project inside their shell), but
  the alpha mask can: with the atlas fixed, 8% of houses have a changed pixel where the real render
  shows background, a sightline through two windows that the fill now blocks. The paper's "the
  silhouette is bit-identical" needs that qualifier.

Audit: `audit_claims.patch` in the run dir adds `added_claims_leak_postfix_20261106` (15 claims, all
pass against the run); not applied, because the TeX still prints 1.75% and 212. `draft_latex.tex`
has the replacement paragraphs for §6.2 and app:hull. samples.png: four 3D-Craft test houses.

## §86. The Objaverse mesh probe at 568 strict objects, and LEGO's sample size (2026-11-07)

The strict Objaverse set stopped at 290 because of a 6 GB budget (§68). The fetch resumed on
2026-11-06 (`fetch_objaverse_interior.py --strict-target 600 --max-gb 9`) and ran out of candidates at
**568 strict objects** (1,219 accepted, about 6.5 GB transferred, GLBs deleted). The 278 new objects
include the small-file tier that is not interior-worded, so the building share needs the §68 name
audit again. Design 284 reference / 284 pool / n = 170, seed 0, run
`outputs/run_20261106_210006_mesh_probe3d_568_s0` (results T109). The shared GPU had under 1 GB free
(a vLLM server holds 14.6 GB), so `scripts/run_mesh_probe_3d_noclip.py` runs the paper script with
DINOv2 on CUDA at batch 8 and Uni3D, ULIP-2 and chamfer on the CPU, dropping CLIP-L and the 8/16-view
rows. On the CPU the run takes about 12 hours, mostly chamfer pairs (40 to 50 min per sampler) and the
FD rows' 200 subsample intervals (about 45 min per row); its results.json was not written at the time
of this note, so the numbers below are the scoring log plus intervals rescored from embeddings.npz.

| seed 0 | Strip | Fill | at n=96 (290) |
|---|--:|--:|--:|
| DINOv2 4 ext. views KID | +0.0 | +0.0 | +0.0 / +0.0 |
| DINOv2 4 cutaways KID | +17.2 | +32.5 | +10.9 / +13.5 |
| V-KID | +65.8 | +258.3 | +23.7 / +73.8 |
| Uni3D vis KID [95% CI] | -0.5 [-0.9, -0.2] | -0.4 [-1.0, -0.0] | +0.4 / +0.3 |
| ULIP-2 vis KID | -0.0 [-0.1, +0.1] | -0.1 [-0.3, -0.0] | -0.0 / +0.1 |
| Uni3D all / cut KID | +14.3 / +19.1 | +14.2 / +20.6 | +4.6 / +4.3, +8.5 / +6.3 |

The blindness holds at 1.8x the probe n and 2.2x the reference set, and every interior observer
separates further. Exterior controls stay large (DINOv2: noise +55.6, wall +10.7, decimation +61.8).

Seed 1 at this scale: the full script ran out of GPU memory (the seed-0 process kept a 322 MB CUDA
context while scoring). A render-rows-only script, `scripts/run_mesh_probe_render_rows.py` (DINOv2
ext/cut and V-KID, same split, `probe3d.score`), was started on the CPU under osmesa, where
`mesh_probe.render` leaks about 28 MB of host RAM per item through pyrender's cached renderer; the
script now recycles the renderer every 8 objects (RSS flat at about 0.8 GB, against 2.7 GB after 40
objects without). The CPU runs `run_20261107_071437_mesh_render_rows_568_s{0,1}_b` were still going
at the time of this note; `run_20261107_051031_mesh_render_rows_568_s1` died of the leak.

**LEGO.** The LEGO arms in Table~lego are already at n = 256 (ShapeNet void corpus, 450 test) and
n = 253 (StableText2Brick test). The leak run (§85) adds that LEGO is the opaque regime: wool is opaque,
and with the atlas fixed Fill-Interior reads -0.00004 floors, so the paper's -0.01 is the atlas
artefact.

## §87. A functional ground truth for the 76 reversed pairs: it sides with MV-KID (2026-11-08)

Run `outputs/run_20261108_204723_functional_groundtruth` (results.md there; T114). Asked by
critic_3dvision W4/Q1 and critic_ac_clarity W2/E1: on the 76 pairs where MV-KID and V-KID v1
disagree with both significant (§83), which one is right? No humans, no paid API, CPU only.

**Measures** (`functional.py`, ~5 ms per build), from the material-aware traversal player set
down outside: enterable, reached interior floor cells, reached rooms (interior-air components,
doorways closed so they split rooms, >= 6 standing cells), roof (>= 50% of the hole-filled plan
sheltered by a block with two free cells under it), largest 6-connected component share. Real
val means 0.94 / 80 cells / 1.36 / 0.81 / 0.98. V-KID's 77 features explain per-build ridge
R^2 0.29 to 0.62 (related through its interior-volume ratios, not the same). FD = mean of
per-measure W1 to val over real between-build spread; variants FD_noise, FD_kid,
FD_occ (occupancy only, no door semantics). B = 1000, shared reference resample.

**Validity.** Real ranks first of 41 under all four FDs (0.09 [0.08, 0.19]; best generator
classic-matched 0.52). FILL-INTERIOR 1.21 (enterable 0.94 -> 0.20), HOLLOW-SHELL 0.60, facade
ceiling 1.22. Interior-aware, so not a third exterior observer.

**Result: functional sides with MV-KID.** Spearman over 40: FD vs MV-KID +0.67 [0.41, 0.82],
vs V-KID v1 +0.16 [-0.20, 0.45], difference -0.50 [-0.84, -0.16] (v2: +0.38). Reversed pairs:
V-KID side 16/76 = 21% [12.5, 31.9]; non-facade pairs 1/38 (2.6%); facade pairs 15/38 (39%).
Variants: FD_noise 8%, FD_kid 26%, FD_occ 24% (non-facade 1/38). Leave-one-generator-out
12-25%. Per measure: enterable, floor, rooms each side with V-KID on 2/76; roof 7; lcc 36.

**Why.** Voxel AR generators clutter their interiors: median 22% enterable (11-60%), median
3.4 reached interior floor cells vs real 80. V-KID's local statistics (thickness, patterns,
sealed air) read the clutter as realism. Deleting hidden blocks from m2b_20m_sftmix_s1 output:
V-KID +147 floors (worse), FD 1.32 -> 0.90 (better); canon_oriented 0.56 -> 0.40. So the §83
reading of that 147-floor cost as interior damage does not hold functionally.

**What changes for the paper.** The probes stand (FILL/HOLLOW are functional damage MV-KID
cannot see). The claim "V-KID orders the reversed pairs correctly" does not: on generators a
functional check agrees with the render ranking. Draft sentence + supp subsection in the
run's `draft.tex` state this as measured and list the contradicted claims. Natural next step:
a functional pillar (or FD itself) reported beside both tiers; V-KID v2's walk scalars move
its agreement to +0.38, so a reachability term in the descriptor is worth testing.

## §88. Statistical methodology audit for the CVPR critique: FPC, effect size, exact tests, TOST, kernels (2026-11-08)

Run `outputs/run_20261108_205428_stats_fixes` (results.md, results.json, drafts.tex, samples.png).
Answers critic_metrics.md W1/W2/W3/W5 from cached features (plus one GPU render of Hollow/Grow DINO
features on the leaderboard's 256 builds; also completed the 40-generator DINO cache in
`run_20261106_193513_render_blind_rank_flips/feat_cache`).

- **FPC claim refuted in direction.** Regime-matched simulation (reference + 1,464-build population
  from train, non-overlapping n-sets as truth): the bench floor `sd/sqrt(1-f)` overstates the true
  fresh-draw sd by 5-12% (truth/bench 0.88-0.95 at n=256, 0.89-0.93 at n=128); the critic's `(1-f)`
  would overstate it ~1.8x. The `(1-f)^2` argument misses the zeta2/N first-order term of the pool's
  own kernel matrix. On the real pool: card V-KID floor 0.0533 vs corrected 0.0482 (card 11% high),
  MV 0.00161 vs 0.00165. Corrected headlines: Fill MV 0.34, V-KID 491 (was 444). The 48-draw MC
  error ([0.043, 0.067] for V-KID n=256) is larger than the FPC bias and explains the 422 vs 346
  Material-Shuffle discrepancy (two floors, same raw KID). No V-KID/MV-KID gate or ranking changes.
- **Effect size** sqrt(KID)/sqrt(KID_RandomVoxels): Fill MV 0.03 vs V 0.78 (27x, not 1,270x);
  BrickGPT 2.2x (not 4.4x); elevations 17x (not 100x); Patch-Stitch reverses (geometry 0.71 >
  appearance 0.63). Detection-n: V-KID n=8 on Fill/Hollow/Grow; MV-KID never reaches 80% on
  Fill/Hollow up to 256.
- **Inference.** Cards' significance groups are exact permutation (verified); main §5 "paired
  bootstrap" is wrong text; rank intervals are unrescaled 80% subsamples (rescale x2 adds no tie);
  the 40-generator study used the discredited subsample-normal test. Exact permutation (20k, Holm
  780): **40 of 308** both-separated pairs reverse (was 76 of 388), rho 0.50 [0.37, 0.58]; n=256 only
  24 of 255; v2 29 of 308. Generator-bootstrap rho CI [0.20, 0.72]. Rank shift vs hidden blocks
  rho -0.26 (p 0.10). (§87/T114's "76 reversed pairs" set should be rebased on the 40.)
- **TOST** (margin 2 floors): Fill MV +0.35, 90% CI [-0.28, 0.98], Hollow +0.04 [-0.25, 0.33]:
  equivalence established. Paired swap test Fill p=0.055; the probe-vs-source paired MMD shows a
  tiny detected render shift (e 0.018, glass leak).
- **Kernel.** Linear term = 99.8% of MV-KID, 98.8-99.5% of V-KID: both are mean-difference tests.
  Linear/RBF/no-clip/no-whitening V-KID all keep Fill and Hollow at hundreds of floors and pass all
  geometry-ladder gates; whitened DINO keeps Fill at 0.5.

## §89. Cross-face constrained decoding: faces that cannot contradict each other (2026-11-09)

T68 left the facade track's geometry pillar held by two things no training lever touched:
faces that disagree where they meet, so the lifted shell falls into 60 to 75 pieces against
13.5 for real faces, and the empty interior. Nothing in the pipeline enforced agreement. The
joint layout conditions each face on the ones before it through next-token cross-entropy
alone, `facade.lift` resolves a disagreement by letting the later face overwrite, and the
`face_solidity` reward is per face. This section adds the constraint at decode time, the
same move as T11's adjacency-constrained decoding: a consistent face set by construction, at
no training cost, with the model free wherever the earlier faces left a choice.

**The rule** (`blockgen/corpus/facade_constraints.py`). A `ShellState` over the box records
what the faces drawn so far committed: every cell's first hit is a *known* voxel carrying its
block token, and the voxels in front of it (or the whole column of a see-through cell) are
*empty*. For a cell of a later face it returns the depths still allowed and the block each
depth forces. A first hit cannot sit on a voxel another ray saw through, cannot lie behind a
voxel another face filled, and when it lands on a known voxel it carries that block; the cell
may be see-through only if no known voxel lies on its ray. A cell whose first open voxel is
already known has exactly one answer and is written without sampling. Everything else is the
model's: it may still put a block in front of a known voxel where no ray has passed (the
one-block-thick wall a side face sees edge-on, the eave the front face hid), and it draws
freely where nothing is known. Real face sets satisfy the rule in every order (test), and
lifting a consistent set and projecting back reproduces every cell, which gives the metric
`reprojection`: the share of cells that come back different (zero for real shells), plus the
share of filled cells hidden behind another face's block.

**The sampler** (`blockgen/train/sample_constrained.py`) mirrors `sample.generate_faces` for
the joint layout, forces the face headers and the final EOS, enforces the grammar at every
position (a cell is `empty` or a depth inside the box then a block, so a face is never ended
early), and in `constrain` mode masks the logits with the state. `none` mode is the matched
control: same grammar, no state. `given` faces are forced verbatim and committed, which is the
oracle: the real front face in context, the model drawing the rest.

**Battery** (`scripts/facade_constrained.py`, `scripts/facade_constrained_battery.sh`, queued
behind another session's GPU job; log `outputs/logs/facade_constrained.log`), all on the 20M D4
joint model of T68 at its selected step 18,000, n=399 on the same real boxes its T68 samples
drew (seed 0): grammar control and constrained at T=1.0 and T=0.7; oracle completion of the
399 test houses from the real front face at both temperatures and both modes, and from front
plus right at T=0.7; then one card with the four decode arms beside the T68 raw, T70 T=0.7,
T77 best-of-4 and T73 RFT arms, patchwork and the six-face ceiling. CPU smoke on two builds:
22.5 percent of cells determined by earlier faces, reprojection disagreement 0, one 26-connected
piece in the oracle completion. Pre-registered reading: the constraint should cut `n_components`
toward the ceiling's 13.5 and move `geom_kid`; if it does not, the extra pieces are not edge
disagreement but the faces' own fragmentation, and the lever is the interior stage.

**Results (2026-11-09, results T117).** The prediction held. Free decoding never produces a
consistent face set (0 of 399 at T=1.0, 7 of 393 at T=0.7; 28 to 39 percent of cells come back
different after the lift), and under the rule a fifth to a quarter of all cells are determined by
the earlier faces and written without sampling. On the card the constrained arm cuts the lifted
shell's components from 62.7 to 29.5 at T=1.0 and from 33.7 to 12.9 at T=0.7, the ceiling's 13.5;
`geom_kid` moves 29.6 to 27.0 and 19.0 to 16.5, both Holm-separated from the matched grammar
control (p 0.033), and `mv_dino_kid` 0.496 to 0.331 and 0.299 to 0.166, the latter level with
best-of-4. BlockScore 1057 to 963 and 679 to 589; the grammar control is not separated from the
raw sampler, so the effect is the cross-face rule alone. T68's extra pieces were edge disagreement.
The constrained arm sits between the T70 temperature arm (637) and the RFT arm (553); best-of-4
(483) and the ceiling (493) remain ahead, and the remaining `geom_kid` gap is thinness and size
(541 blocks against 660, `wall_frac` 0.65 against 0.74, `walk_floors` 0.45 against 0.87), not
connectivity (`lcc_ratio` 0.81 for both).

The rule's one cost is that a see-through cell is absolute. The front face is drawn first, so each
of its holes empties a whole column for the five faces after it, and a sparse front face leaves a
skeleton where the control drew a cluster: median blocks 686 to 509, `enterable` 0.316 to 0.285,
and a few builds on the sheets reduced to two beams or a row of dots. Three things follow. The
lever is orthogonal to selection and to the fine-tune, so constrained best-of-4 and constrained
decoding of the RFT model should stack. A relaxed rule that forbids first-hit contradictions but
lets a later face fill a column an earlier face saw through (re-projecting that earlier cell)
would remove the skeletons at the price of exact consistency. And since the first face dictates,
the face order is a training choice worth revisiting: the top face carries the footprint, the
most informative silhouette, and nothing in the joint layout requires the front face to go first.
The oracle completions OOM'd at batch 32 beside another session's bench job
(`scripts/facade_constrained_oracle_rerun.sh` reruns them at batch 8, log
`outputs/logs/facade_constrained_oracle.log`); their reading is appended below when they land.

**Oracle reading (appended 2026-11-09).** From the real front face the constrained completion
reaches shell IoU 0.55 against 0.36 for the grammar control (front plus right: 0.72 against
0.47), with a quarter to a third of the drawn cells written by the rule. The gain is where the
front face determines things: the back face's mask (IoU 1.0 by construction) and the side faces'
depth (0.29 to 0.74 exact), which is the footprint's extent. The back face's depth (0.27 exact,
2.7 cells off) and the block identity of free cells (0.37 to 0.51 agreement) stay the model's
problem. Table in results T117.

**Stacking results (appended 2026-11-09, results T122).** The three stacked arms ran 13:05 to 15:17 UTC
(`outputs/run_20261109_144658_bench_facade_constrained_decoding_stacked`). Constrained best-of-4 on the
base model is the best learned facade arm on appearance (`mv_dino_kid` 0.098 against best-of-4's 0.149,
its own Holm group) and coherence (components 7.7, below the six-face ceiling's 13.5; `enclosed_air`
0.083 against real's 0.114), and ties best-of-4 and the ceiling on `geom_kid` (13.9, 13.5, 13.8, one
group), so its BlockScore 497 against 483 is noise inside a pillar that has reached the representation's
ceiling. Selection also removes most of the see-through cost (618 blocks against 541): the most solid
of four candidates is the one whose front face has no holes to empty columns with. Constrained decoding
of the RFT model ties it on BlockScore (552 against 553) and improves appearance (0.182 to 0.147) and
components (17.6 to 10.4): the fine-tune had already bought most of the shell. Constrained best-of-4 on
the RFT model is disqualified for self-duplication (twin rate 0.271 against real's 0.010): three levers
pushing on `face_solidity` (fine-tune, rule, selection) saturate the reward (37% of kept builds at 0.99
or above, against 1.8% under one lever and 7.3% under two) and the optimum at the plateau is a filled
single-material box, which the sheet shows in three cells of four. Conclusions: `face_solidity` is
spent as a reward and needs a counterweight before another lever is stacked on it; the geometry pillar
is at the six-face ceiling for the best arms, so the facade track's BlockScore is now bounded by the
empty interior, and the next lever for the number is an interior stage, not another face lever. The
relaxed see-through rule and the face order stay open but matter less now that selection removes the
skeletons.

**Shell reference (2026-11-11, T122 addendum).** Scoring the same arms against the six-face shells of the
reference houses (standardizer refit on shells) splits the geometry gap in two: representation (real
houses 18.2 from their own shells; ceiling 13.8 from real) and generator (constrained best-of-4 4.0,
constrained 5.4, best-of-4 8.2, grammar 11.3). The order of constrained decoding and best-of-4 flips
between the two references because free decoding's face disagreement fills volume the full-house
reference reads as interior. Keep both columns on facade cards: the shell column for developing the
generator, the full-house column for the board.

## §90. The f=4 oracle of the cascade-and-AR hybrid: a per-face AR that reads its own coarse level first (2026-11-09)

§74 closed the fixed-pooling cascade with a recommendation: let the cascade supply the gestalt
and a sequence model draw the trim and openings with its neighbours in context, and test it
for free by giving the real f=4 level to a prefixed AR. The cascade completing from the real
f=4 reached block agreement 0.515 against the copy's 0.556 and drew no window (T88); whether
that is the hierarchy's fault or the parallel fine level's is the question the oracle settles.

**The layout.** `pooled4` is a third facade layout (`blockgen/corpus/tokenizer.py`): one face
per sequence, preceded by its own level f=4 under the `any` rule (`facade.pool`), in the same
cell grammar: `[BOS, tags, face, U/4, V/4, cell*, face, U, V, cell*, EOS]`. The shard is the
per-face shard with the prefix (12,684 train sequences, 5,172 holdout, tokenizer
`4c4fd9c6cb48`, built in a second). The model is T66's per-face 5M recipe unchanged
(`configs/train/houses32-facades-pooled4.yaml`, 8,000 steps, about five minutes), so the
holdout NLL on the face cells is the only training number that changes. The sampler is §89's
grammar-enforcing decoder with the coarse level passed as a dictated plan entry
(`given={0: pooled}`); without it the model draws its own coarse level and then the face,
which is the hybrid as a generator and is what `facade_builds` does for this layout.

**The test** (`scripts/facade_hybrid_oracle.py`, `scripts/facade_hybrid_chain.sh`, queued
before the constrained stack of T117's follow-up): the 399 test front faces, the real f=4 of
each given, the face drawn at T=1.0 and T=0.7; against the copy (upsample of the real f=4)
and T88's cascade completion (`mode from f4`: mask IoU 0.754, depth exact 0.581, block
agreement 0.515, 4.4 distinct blocks per face, holes 0.001). Face statistics with W1 to real,
and a window count: glass, stained glass and pane cells per face, faces with any. Pre-registered
reading: the hybrid should beat the copy on block agreement (the cascade did not) and draw
windows at a rate near the real faces'; the depth exactness should land near the cascade's
0.58 or above. If block agreement stays below the copy with the real gestalt in hand, the
coarse level is not what the fine model needs and the hierarchy idea closes for faces; if it
clears the copy, option (a) of §74 is live and the next step is the cascade's own f=4 as the
prefix instead of the real one.

**Results (2026-11-09, results T120).** The bar was not cleared. Given the real f=4, block
agreement is 0.28 at T=1.0 and 0.33 at T=0.7 against the copy's 0.556 and the cascade's 0.515;
mask IoU 0.62 against 0.76, and the model leaves a quarter of the dictated silhouette undrawn.
Its faces given the real level agree with the real face barely more than its faces drawn from
the box (IoU 0.617 against 0.567): the prefix sets palette and size, not shape. What it does
give is windows (69 percent of faces, real 74, cascade 0) and face statistics close to real
(W1 fill fraction 0.02, distinct blocks 9.9 per face). The cascade is faithful and blind to
detail, the AR detailed and deaf to the gestalt, and context alone does not join them. Option
(a) closes as a free lever. What remains of the hierarchy idea for faces is a decode-time
constraint tying the fine face to its coarse level (T117's move one level up), which fixes
the mask and not the blocks; the block identity of a cell is the number no facade arm has
moved, oracle or not (§89's joint oracle: 0.37 to 0.57 given one to two real faces).

## §91. The functional ground truth and the design decisions on the exact test: 33 of 40 side with MV-KID (2026-11-09)

Run `outputs/run_20261109_152704_rankflips_functional_rebase` (results.md there; T123). CPU only.
§87 judged the 76 reversed pairs of the subsampling test, which §88 showed anti-conservative; this
rebases it on §88's 40 exact-permutation pairs (39 of them are in the old 76). Same FD, same bootstrap
(reproduced bit-for-bit, 16/76 re-asserted).

- **Pairs.** Walk-through sides with MV-KID on 33 of 40, V-KID 7 (17.5% [7, 33]); every FD difference
  significant. Non-facade 0 of 18; facade 7 of 22, and 5 of those 7 are BrickNet (0.8% enterable)
  preferred over a facade arm, i.e. a choice between two unenterable families. Variants: 2 / 9 / 8 of
  40; enterable, floor and rooms alone side with V-KID on 0 of 40. LOGO 6-24%.
- **Rank agreement** unchanged (FD vs MV 0.67, vs V 0.16); vs the max of the two z's 0.17
  [-0.18, 0.45] because geometry is the worse pillar for 37 of 40. BlockScore inherits V-KID's ordering
  on generators.
- **34 decisions under the exact test** (Holm 34): agree 6 (was 13), MV-only 14 (12), V-only 1 (3),
  neither 13 (6), opposite-sign 0. The surviving V-only decision (3D-BPE pieces, +162.6 V-KID floors) is
  one FD reverses (1.34 vs 1.67).
- **Worked examples** survive at the Holm floor 0.039; fig:rankflips stays valid.
- **For the paper:** drafts in `deliverables/paper/cvpr27/drafts_20261109/` (rankflips.tex with a new
  app:functional subsection, stats.tex, limitation_line.tex, audit_claims_additions.py, README.md).
  Claim to state: detection of interior change is not ranking of interior function. The probes hold;
  "V-KID orders generators better" does not.

---

## §92. Round-2 review experiments: a function pillar does not fix the ranking; V-KID is a detector (2026-11-11)

Round-2 critiques (`deliverables/paper/cvpr27/critiques/round2/`, rigor, skeptic, content) converge on
one point: the walk-through FD sides with MV-KID on generators, so V-KID and BlockScore do not rank
function. Eight CPU experiments (results T126 to T133; drafts in
`deliverables/paper/cvpr27/drafts_20261111_round2/`, claims in its `audit_claims_additions.py`,
61/61 hold).

- **Function pillar (T126).** FD in noise floors (sd 0.019 over 200 real draws) never sets the
  maximum (0/40 generators in z, 2/40 in e), so BS+F = BS: ρ with FD 0.17, 4/40 reversed pairs.
  Guards against circularity: measure split, door-free criterion, leave-one-measure-out, build
  split. Held out, the function pillar predicts no better than MV-KID (LOMO mean 0.41 vs 0.42), and
  any maximum that contains V-KID sits below zero (−0.11 to −0.18). Weighting function up only helps
  once it binds everywhere, i.e. once BlockScore is the function pillar. Hollow-Shell is not damage
  under reachability (z 0.7); its FD is connectivity alone. Lesson: the leaderboard's problem is
  V-KID's veto, not a missing pillar. Report function as a column beside the board.
- **Effect-size plane (T127).** Every generator lies above the diagonal (V e > MV e); FD colour
  does not follow height on the V axis. Figures with live text in the drafts dir.
- **BrickGPT 2.2 (T128).** The number does not survive: interval [1.8, 3.2] with the card's anchor
  (OSMesa CPU render, self-consistent), 1.6 to 3.2 over single-build anchors, 3.2 and 1.6 under two
  new anchors. The direction does (ratio > 1 everywhere). OSMesa does not reproduce EGL pixels
  (cosine 0.87 to 0.96), so CPU features must never be mixed with cached EGL ones.
- **Seed variance (T129).** σ_seed 13 MV / 36 V floors from three recipes (df 4; the V term rests on
  the world-crop pair). With it, MV-significant decisions 20 → 10, V 7 → 2; 28/40 reversed pairs
  clear the 95% seed band. The faces "seeds" card is sampling seeds, not training seeds.
- **Leak bins (T130).** 0.96 is the atlas-fixed number; deployed 0.66. Original bin intervals were
  the unrescaled 80% subsample. Paired test: no bin survives Holm; top quartile −1.5 floors toward
  real, p 0.012, Holm 0.060.
- **Rank intervals (T131).** Rescale + floor MC error: VOXEL-AR-U ties MATERIAL-SHUFFLE, MONOCHROME
  ties TRAIN-COPY+NOISE; flips 0.29/0.41 replace 0.20/0.21. Ungated Dream-Cubed is 2–3 (z), 2 (e2).
- **Gate audit (T132).** S6 blocking removes every render metric (both domains) and the 2-d
  connected-parts metric; V-KID class untouched. Dropping G1 changes 1 of 112 verdicts.
- **Fill inference (T133).** Paired swap p 0.057, unpaired 0.87; canonical paired fixed-reference
  CI [−0.23, +0.94] floors, TOST equivalent; direct paired MMD detects (e 0.018, p < 1e-4).

Process note: `pkill -f <path>` from a Bash call matches the calling shell's own command line and
kills it; use the PID.

## §93. Reviewer-weakness queue, 2026-11-11: judge position, worst pillar across n, the twin gate, colour-free 3D encoders, Objaverse seed 1

CPU items first, no API calls, sec/ untouched; drafts in `deliverables/paper/cvpr27/drafts_20261111_experiments/`
(item1..item5 .tex, `audit_claims_additions.py`, 45 claims pass).

- **Judge position (T134).** `run_20261111_015408_vlm_judge_debias`. Exterior pairs are decided by slot: the two
  orders agree on 12% (31 of 256 pairs) against 54% roof off. On content-decided pairs P(real) 0.68 [0.50, 0.84]
  exterior (p 0.07), 0.89 [0.83, 0.95] roof off; BT with a position term 0.59 [0.50, 0.70] vs 0.73. The contrast
  survives; the exterior number is "uninformative, leaning real", not a clean null. System BT with a position term:
  ranks unchanged, real 6th; Spearman MV 0.77, BS 0.59, V 0.17.
- **Worst pillar across n (T135).** `run_20261111_015636_blockscore_pillar_stability`. 0 of 13 shared arms change
  worst pillar between n=128 and 256; Spearman 0.98. Effect-size BlockScore (max of app and geo, sqrt MMD^2 / RV):
  Spearman 0.96 (n=256) / 0.90 (n=128) with the z form; submissions ordered identically. Five-pillar effect size
  lets palette dominate (0.61).
- **Twin gate (T136).** `run_20261111_015843_twin_gate_audit`. 0 of 24 real held-out draws over the bar (max 0.023).
  6 of 40 rank-flip generators gated: 3 agentic LLM, llm_control, and **two trained voxel-AR arms at T=0.7
  (0.070, 0.109)**, which contradicts the paper's "every model we trained reads 0.000". Provenance: rule (3 spreads,
  0.02) is the duplicate gate's from 2026-09-07; the statistic and its IoU 0.8 were chosen after the Dream-Cubed
  audit (§73). At IoU 0.95 Dream-Cubed still reads 0.19 vs real <= 0.016. Ungated, Dream-Cubed ranks 2 (first
  submission). The library's global FFT grid needs tens of GB on the LLM arms (crops up to 253 wide);
  the audit uses a per-pair grid (identical on dream, checked).
- **Colour-free encoders (T137).** `run_20261111_024010_uni3d_colorfree_gates`. Grey points: ULIP-2 passes G1,
  G2, G6, G7, G8 on all-face and volume points (Fill +6.0 / +16.7 at n=128, V-KID +111.4); Uni3D fails G6 on yaw
  (1.2-1.7 floors). Colour-free rows follow V-KID on the board (rho 0.72-0.81) not MV-KID (0.45-0.49). This
  **contradicts "only V-KID passes every gate"** (6_results l.79-80, X_headtohead).
- **Objaverse seed 1 (T138).** Rerun `scripts/run_mesh_probe_3d_cpuonly.py` (no GPU: OSMesa + CPU DINOv2) from
  seed 1's emd.npz. The first two attempts died: an OOM while a peer job held the RAM, then pyrender leaking
  ~1 GB/min under OSMesa (every uploaded mesh kept until the renderer is deleted); the wrapper now drops the
  renderer per object (RSS flat at 1.7 GB). OSMesa and EGL renders differ by 1.1/255 mean (3% of pixels > 8,
  edges), so seed 1's render rows are not bit-comparable to seed 0's. Run dir `run_20261111_022342_mesh_probe3d_568_s1cpu`.
