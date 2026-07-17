# BlockGen — Data Sources

Single reference for every corpus we have on disk, across both media. Numbers are
measured from the local fetches (Minecraft: various; LEGO: fetch of 2026-07-07). Per-track
loaders, curation, and the paper's results tables live in `notes.md §3`, the mkdocs
[Data & curation](docs/data-and-curation.md) page, and `results.md` (T1–T3b); this file is
the consolidated *inventory* — what each source is, how big, how licensed, what it labels,
and what a sample looks like.

Raw corpora are gitignored under `data/` (Minecraft) and `data/lego/` (LEGO). Sample sheets
referenced below are rendered to `outputs/figures/mc_data/` and `outputs/figures/lego_data/`
(also gitignored — regenerate with the commands in each section).

---

## 0. At a glance

### Minecraft (voxel builds — legacy `(id, data)` vocab)

| Corpus | Path | Size | License | Labels |
|---|---|--:|---|---|
| **GrabCraft** | `data/minecraft/grabcraft/` | 6,560 builds | site user-content (research use) | 5 top-level / 113 sub categories, title, dims, tags, views; **exact `(id,data)`** |
| **3D-Craft** | `data/minecraft/3d_craft/` | 2,537 houses | research (Meta AI CraftAssist) | single class ("house") + timestamped placement sequences |
| **text2mc** | `data/minecraft/text2mc/` | 11,092 h5 (of ~38k) | Kaggle/CC (research) | free-text tags on some builds; modern namespaced blocks |
| **tfrecord crawl** | `data/minecraft/more/` | 36,290 NBT / 69,363 meta | PlanetMinecraft user-content | 15 map categories, title/views/downloads/diamonds/tags |
| Legacy schematics | `data/minecraft/raw/` | 12,366 `.schematic` | mixed (drifted crawl) | none reliable — filenames ≠ metadata |
| Block vocab ref | `data/minecraft/block_ids.csv` | 933 block types | minecraft-data (MIT) | id/name/material/hardness/light/… |

### LEGO (brick assemblies — LDraw geometry)

| Corpus | Path | Size | License | Labels |
|---|---|--:|---|---|
| **OMR** (Open Model Repository) | `data/lego/omr/` | 1,463 sets → 1,819 models, 651,017 placements | **CC BY 4.0** (per file) | set id → title; full color + 3×3 rotation + part ref per placement |
| **StableText2Brick** | `data/lego/stabletext2brick/` | ~42k train + test | **MIT** | text captions + per-brick stability scores + ShapeNet category |
| LDraw parts library | `data/lego/ldraw/` | 33,362 part `.dat` | **CC BY 2.0** | part geometry (primitives + meshes) |
| LDCad Shadow Library | `data/lego/shadow/` | 4,251 files | **CC BY-SA 4.0** | connectivity metas (studs/anti-studs/clips/axles/pins) |

> **Unique parts/blocks.** Minecraft draws from a fixed ~933-entry block vocabulary
> (`block_ids.csv`); the curated house set actually uses a small legacy `(id,data)` palette.
> LEGO OMR uses **5,173 unique part types** (LegoACE's unreleased LegoVerse: 9,314 across 55k
> private models); StableText2Brick restricts to **8 cuboid brick types** (`1x1`…`2x6`).

---

## 1. Minecraft corpora

### 1.1 GrabCraft — the category-labeled workhorse

[grabcraft.com](https://www.grabcraft.com) organizes community builds into a clean human
category tree, which is exactly the per-type supervision we want. Scraped by
`blockgen/data/grabcraft_scraper.py` (resumable) → `blockgen/data/grabcraft_dataset.py`.

- **6,560 builds** across **87 category folders / 113 sub-categories / 5 top-levels:**

  | Top level | n | | Top level | n |
  |---|--:|---|---|--:|
  | Buildings | 3,744 | | Statues | 303 |
  | Transportation | 1,317 | | Pixel Art | 286 |
  | Outdoors | 910 | | **Total** | **6,560** |

- **Blocks decode to an exact `(id, data)`** — every block entry carries a
  `"<legacy_id>_<data>.png"` texture field (pre-flattening numeric id + data), so no fuzzy
  name matching (oak vs spruce planks are distinguished). Lines up with `STANDARD_VOCAB`.
- **Per-build metadata:** title, width/height/depth, tags, views.
- GrabCraft-only cropped caches (`build_cache`): `gc_small_24` 3,952 kept · `gc_small_32`
  3,061 · `gc_small_48` 5,621 (of 6,560 scanned; the rest exceed the dim cap).

*Example* — `american-middle-class-house-10.json` (`brick-houses`): **30×14×25**, **2,348
blocks**, tags `[middle class house, american house, american suburban home, house, building]`.

### 1.2 3D-Craft — house-building action traces

Meta AI's CraftAssist 3D-Craft corpus: each house is a human build recorded as a
*placement sequence*, not just a final grid. Loader in `blockgen/utils/corpora.py`.

- **2,537 houses**, each a `houses/<id>/` dir with:
  - `schematic.npy` — final `(y, z, x, id/meta)` 4D array (note the y-up axis order),
  - `placed.json` — `(tick, player, xyz, (id, meta), 'P'|'B')` place/break events,
  - `stats.json` — size, block count, time spent.
- Single implicit class ("house"); the temporal traces are unique to this corpus (usable
  for order/curriculum experiments — see `utils/ordering.py`).

*Example* — `house`, **21×11×24**, **1,100 blocks** (after our pool crop).

### 1.3 text2mc — modern namespaced builds

HF/Kaggle `shauncomino/minecraft-builds-dataset` (a UCF senior project). Modern
*flattened* block names remapped to our legacy `(id, data)` vocab via
`blockgen/utils/block_remap.py` (**97% state coverage**; world cuts ground-stripped).

- **11,092 processed `.h5` builds** (integer grids) with a **3,717-entry** `tok2block.json`
  name map, loaded by `corpora.load_text2mc`.
- **28,235 raw `.schem` builds** the dataset author never converted (the gap between the
  "11k processed" and the "~40,000 builds" advertised — see the dump's `README.txt`).
  These are **Sponge/WorldEdit schematics** (NBT `Palette` + varint `BlockData`, YZX) that
  the legacy `nbtschematic` loader can't read; `blockgen/utils/schem.py` decodes them and
  `corpora.load_text2mc_schem` maps their block-state names straight to our legacy vocab
  (**~97–98% palette coverage by name**, higher by voxel). Together the two loaders realize
  the full **~39k**-build corpus. `processed_build_dataframe.csv` (**37,989** rows) links each
  build to its `.schem` shards (`PROCESSED_PATHS`) and its PlanetMinecraft page + tags.
- Every build carries free-text **tags** (`TAGS` column) — a coarse, human-assigned
  category we get for free. `blockgen/labeling/categorize.py` normalizes them into a
  build *type* + *style* (e.g. `medieval castle`) for captions and conditioning.

*Example* — `house`, **32×24×28**, **4,972 blocks** (post-remap, in the 48³ pool).

### 1.4 tfrecord crawl — the labeled PlanetMinecraft set

A PlanetMinecraft crawl under `data/minecraft/more/`: `fullSchematics.json` = **69,363**
metadata records; `schematics/*.tfrecords` = **36,290** records pairing a `url` with
gzipped-NBT `schematicData` (the metadata join key). Parsed by hand (no TensorFlow) in
`blockgen/data/tfrecord_dataset.py`.

- Labeled cache `tf_small_24` keeps **5,866** structures (cropped, `max_dim ≤ 24`,
  `8 ≤ blocks ≤ 4096`), **100% with metadata**, across **15 categories:**

  | Category | n | | Category | n |
  |---|--:|---|---|--:|
  | Land Structure | 1,723 | | Pixel Art | 125 |
  | Redstone Device | 1,287 | | Piston | 120 |
  | 3D Art | 1,264 | | Environment | 101 |
  | Other | 477 | | Water Structure | 78 |
  | Air Structure | 427 | | Challenge / Minecart | 51 / 50 |
  | Complex | 126 | | Underground / Nether / Music | 19 / 9 / 9 |

- Metadata includes title, subtitle/category, tags, views, downloads, diamonds — the
  popularity signals used for the "reliable seed set."

### 1.5 Legacy schematics & block vocab

- `data/minecraft/raw/*.schematic` — **12,366** files from an older download that has
  **drifted** from its metadata (file 1 ≠ record 0 by any offset; only ~5% content-match the
  tfrecords). Kept for volume only; **not used for labeling**. Old `small_24` cache: 2,892.
- `data/minecraft/block_ids.csv` — **933** block types (id, name, displayName, material,
  hardness, light, bounding box, …). The canonical Minecraft block reference vocabulary.

### 1.6 Curated house dataset (the training target)

`blockgen/curation/houses.py` pools GrabCraft + 3D-Craft + text2mc (`.h5`) + text2mc_schem
(`.schem`) into a shared legacy vocab, then applies a per-corpus enclosed-air "house-ness"
gate + variant-aware dedup. This is what the models actually train on. Load via
`load_house_structures(max_dim)`.

| Cache | Pooled | Quality drops | Exact dups | **Final** | GrabCraft | 3D-Craft | text2mc `.h5` | text2mc `.schem` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `houses_32` | 3,493 | 826 | 6 | **2,661** | 1,360 | 1,267 | 34 | — |
| `houses_48` | 4,502 | 1,381 | 33 | **3,088** | 1,410 | 1,381 | 180 | **117** |

> `houses_48` rebuilt 2026-07-17 with the new **`text2mc_schem`** source (the 28k raw
> `.schem`, decoded via `utils/schem.py`). It added **117** clean room-bearing houses (of 348
> pooled) — lifting text2mc's total house contribution from 180 → **297** (+65%) and the pool
> from 2,971 → 3,088; every other corpus reproduced identically. The payoff is **strongly
> scale-dependent** (schem builds are mostly large multi-shard world exports, so the `max_dim`
> cap binds hard): only **8** survive at 32³ vs **348** at 48³, so `houses_32` was left
> unchanged. Rebuild either with `python -m blockgen.curation.houses --max-dim {32,48}`. The
> other win is the general `.schem` corpus (`corpora.load_text2mc_schem`, ~28k builds) for
> occupancy/shape pretraining at larger volumes, where size isn't the binding constraint.

Top drop reason is **no-interior** (675 at 32³) — the enclosed-air gate (≥8 interior air
voxels unreachable from the bbox boundary) rejects trees, vehicles, roof fragments. It is
corpus-calibrated: 17% of ground-truth GrabCraft houses have open interiors (so GrabCraft is
exempt) vs 31% of 3D-Craft and 54% of text2mc. This yields **~3.7–4.2× more clean houses**
than the old 714-house labeled subset. Full accounting: `results.md` T3b.

**Dataset sheets** (real Minecraft textures, headless pyrender; `blockgen/renderer/grid.py`):

![Combined curated house grid (140 samples, 32³)](outputs/figures/mc_data/houses_32_grid_140.png)
*140-sample combined `houses_32` grid, real-texture render.*

Per-corpus splits of the same cache make the source mix legible:

| GrabCraft | 3D-Craft | text2mc |
|---|---|---|
| ![](outputs/figures/mc_data/houses_32_grabcraft.png) | ![](outputs/figures/mc_data/houses_32_3dcraft.png) | ![](outputs/figures/mc_data/houses_32_text2mc.png) |

![48³ curated house grid](outputs/figures/mc_data/houses_48_grid.png)
*`houses_48` (48³) grid — larger builds admitted by the bigger cap.*

Regenerate:
```bash
.venv/bin/python -m blockgen.renderer.grid --houses 32 --corpus grabcraft --rows 8 --cols 12 \
    --out outputs/figures/mc_data/houses_32_grabcraft.png
```

---

## 2. LEGO corpora

All LEGO corpora fetched 2026-07-07 (roadmap Phase 0). See `data/lego/README.md` for the
fetch commands and `research.md §E` for the typed-connection thesis. Rendered with a
separate Blender 4.2 + ImportLDraw package (`legogen/renderer/`).

### 2.1 OMR — Open Model Repository (license-clean core)

Official LEGO sets as LDraw `.mpd` — every part with color, position, 3×3 rotation matrix,
and part reference. Scraped resumably by `python -m blockgen.data.fetch_omr`.

- **1,463 sets → 1,819 model files** (363 MB; 7 dead links).
- **651,017 official-part placements**; parts/model median **156**, mean **447**, p90 **1,107**.
- **5,173 unique part types** — the diverse, *non-cuboid* vocabulary the thesis targets
  (tires, slopes, curved panels, Technic gears/axles, minifigs, BrickHeadz, architecture).
- **Connectivity coverage** (joining the shadow library transitively through primitives):
  **91.1% of placements, 88.4% of unique parts.** Uncovered tail = hoses/ropes/stickers/
  electric exotica → restrict the generation vocab to covered parts, keep >91% of the corpus.
- **License: CC BY 4.0 per file** (`0 !LICENSE …` required by the OMR spec).

*Example* — set `10014-1` → "LDraw.org Official Model Repository - Caboose" (`10014-1.mpd`).

**Dataset sheets** (Blender, `legogen/renderer/ldraw_grid.py`) — same white-background,
orthographic-isometric aesthetic as the Minecraft `mc_data` sheets:

![OMR dataset grid — 96 small/medium sets](outputs/figures/lego_data/omr_grid.png)
*`omr_grid.png` — 96 small/medium official sets (part-ref band 15–160).*

![OMR showcase — 24 larger part-diverse sets](outputs/figures/lego_data/omr_showcase.png)
*`omr_showcase.png` — 24 larger, part-diverse sets (band 200–700), rendered bigger.*

Regenerate:
```bash
.venv/bin/python legogen/renderer/ldraw_grid.py --rows 8 --cols 12 --band 15 160 \
    --out outputs/figures/lego_data/omr_grid.png
```

### 2.2 StableText2Brick — BrickGPT/LegoGPT pretraining volume

HF `AvaLovelace/StableText2Brick` — the BrickGPT corpus: cuboid-brick structures as text
brick lists + captions + per-brick stability. Cuboid-only, but big and eval-comparable.

- **~42k train (6 parquet shards × 7,100) + test**, 43 MB.
- **8 cuboid brick types** (`1x1`, `1x2`, `2x2`, `2x4`, `2x6`, … as `HxW`).
- Each structure carries **multiple text captions**, **per-brick stability scores**, and a
  ShapeNet **category id** (e.g. `04379243` = table).
- **License: MIT.**
- Format is `hxw (x,y,z)` bricks, *not* LDraw — not yet rendered (needs an `hxw`→LDraw
  part converter into the `legogen` pipeline).

*Example* — category `04379243` (table), caption *"Small square table with open rectangular
legs."*, bricks `2x2 (15,17,0)`, `2x6 (15,11,0)`, `2x6 (15,5,0)`, `2x2 (1,17,0)`, …

### 2.3 LDraw parts library + LDCad Shadow Library (geometry + connectivity)

- **LDraw parts** (`data/lego/ldraw/`) — the official part library, **33,362 part `.dat`
  files** (geometry as primitives + meshes). Source ldraw.org `complete.zip`. **CC BY 2.0.**
- **LDCad Shadow Library** (`data/lego/shadow/`) — connectivity metadata NOT in base LDraw:
  **4,251** patch files mirroring the part library with `!LDCAD SNAP_*` metas (studs,
  anti-studs, clips, axles, pins). Source github.com/RolandMelkert/LDCadShadowLibrary.
  **CC BY-SA 4.0.** Always join this to get connection points; its coverage is a subset of
  parts → drives the "restrict to covered parts" decision above.

> LegoACE's LegoVerse (55k models, 9,314 part types) is **not** publicly released, so OMR +
> StableText2Brick are our license-clean core.

---

## 3. Where this feeds the pipeline

- **Minecraft training** → curated `houses_32` / `houses_48` (§1.6); category conditioning
  from GrabCraft/tfrecord labels; see `results.md` T-series.
- **LEGO track** → OMR (typed-connection AR, restricted to shadow-covered parts) +
  StableText2Brick (cuboid pretraining volume, BrickGPT-comparable eval). See `roadmap.md`
  and `research.md §E`.

**Related docs:** `data/lego/README.md` (LEGO fetch + connectivity stats) ·
`docs/data-and-curation.md` (curator API) · `notes.md §3` (loaders) · `results.md` T1–T3b
(dataset tables) · F10 / F13 (these figures).
