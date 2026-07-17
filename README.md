# BlockGen

Generative models for discrete 3D **buildable** structures — Minecraft voxel builds today,
LEGO brick assemblies next. The research thesis: *guaranteed-buildable generation with
diverse parts, trained on open data, transferred across media, with honest novelty
accounting.* See `roadmap.md` for the paper plan and `research.md` for the field survey.

This README is the practical guide: how the repo is laid out, how to run each model, and
how to render structures. For the research narrative, results tables, and idea backlog see
the [Documentation map](#documentation-map) at the bottom.

---

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # base package (declares pyyaml; numpy/torch/h5py assumed present)
pip install -e ".[render]"       # textured rendering: pyrender, trimesh, PyOpenGL, pillow
pip install -e ".[gnn]"          # graph track: torch, torch-geometric
```

- Python 3.10+, Linux. Rendering is headless via EGL (works on the RTX 5070 Ti box).
- CUDA is used automatically when available; everything falls back to CPU.

---

## 2. Data layout

All corpora live under `data/`, split by medium. Raw caches are gitignored; the loaders
below produce in-memory `Structure` objects (see §4).

```
data/
  minecraft/
    cache/            legacy NBT cache (build_cache.py)          — 1.12-era (id,data) vocab
    grabcraft/        GrabCraft category-labeled builds          — preferred labeled set
    3d_craft/         3D-Craft house corpus
    text2mc/          text2mc modern builds (h5) + tok2block.json — remapped to legacy ids
    more/             tfrecord-derived labeled crawl (npz caches)
    raw/              raw schematics/NBT
    text2mc_index.json
  lego/               LEGO corpora (LDraw MPD) — see data/lego/README.md
    omr/              Open Model Repository: 1,819 models / 651k placements (CC BY / CCAL)
    stabletext2brick/ StableText2Brick ~42k (HF, MIT)
    ldraw/            LDraw official parts library (33,362 parts)
    shadow/           LDCad Shadow Library — SNAP connection metas (CC BY-SA 4.0)
  textures/
    vanilla/          extracted Minecraft block textures (fetch below)
```

**Curated Minecraft house caches** (the training targets), built by
`blockgen/curation/houses.py` with a per-corpus enclosed-air "house-ness" gate:

```bash
.venv/bin/python -m blockgen.curation.houses          # builds houses_32.npz / houses_48.npz
```
`load_house_structures(max_dim=32)` → `houses_32` (2,661 builds). See `notes.md §6.1`.

**Fetch textures / LEGO data:**
```bash
.venv/bin/python -m blockgen.renderer.textures --fetch     # vanilla block textures → data/textures/vanilla/
.venv/bin/python -m blockgen.data.fetch_omr                # resumable OMR LEGO scraper → data/lego/omr/
```

---

## 3. Package map

```
blockgen/
  data/         corpus loaders: build_cache, grabcraft_dataset, tfrecord_dataset, fetch_omr
  utils/        serialize (Structure↔tokens↔grid), data (Structure), augment (D4),
                block_remap (text2mc→legacy), corpora (3D-Craft/text2mc), ordering, graph_data
  curation/     curate (Curator: dedup/quality), houses (cross-corpus house dataset)
  tokenizers/   block_tokenizer, standard_vocab, cluster_bpe (3D-BPE anti-memorization)
  models/       voxel_transformer_ar (Track A), voxel_transformer_ar2 (phase4/rope/alibi PE),
                voxel_diffusion (Track B UNet3D), large_pyg_graph_generator (Track C), voxel_port_gnn
  training/     train_ar (+ _ext), train_diffusion, train_graph, train_twostage,
                constrained_decode (adjacency-constrained sampler)
  eval/         novelty (NN-IoU + diversity + duplicate rate), validity (connectivity, LCC repair)
  renderer/     textured (pyrender/EGL real textures), textures (atlas + fetch),
                grid (dense sample grids), render (legacy matplotlib)
  experiments_*.py   reproducible run drivers (see §6)
```

---

## 4. Representation & tokenizer (`blockgen/utils/serialize.py`)

- **`Structure`** (`utils/data.py`): `block_ids` + `block_data` int arrays (legacy Minecraft
  `(id, data)` vocab) with `occupied_mask`, `shape`, `crop_to_non_air()`.
- **AR token stream**: `[BOS, (X, Y, Z, BLOCK)*, EOS]` — variable size, no fixed grid. Built by
  `build_block_vocab(structs, max_dim)` → `BlockVocab`; `build_sequences(...)` emits training seqs.
- **Canonical grid**: `structure_to_grid` / `grid_to_structure` center-pad to a fixed `dim³`
  cube for the diffusion track.
- **Augmentation** (`utils/augment.py`): D4 symmetry group for honest-novelty eval.

---

## 5. Models & methods

| Track | Class | Spec (defaults) | Notes |
|---|---|---|---|
| **A. AR transformer** | `VoxelTransformerAR` | d_model 256, 8 heads, 6 layers, ff 1024, seq 1024 | (X,Y,Z,BLOCK) token stream |
| **A′. AR + grammar PE** | `VoxelTransformerAR2` | same, `pe ∈ {learned, sin, phase4, rope, alibi}` | **phase4** = pos%4 phase + pos//4 block-index embeddings |
| **B. Discrete diffusion** | `VoxelUNet3D` | 48 base ch, grid 24–32, 18 sample steps | MaskGIT-style; air class down-weighted |
| **C. Graph VAE** | `LargePyGGraphGenerator` | block+6-port graph | decodes shared token stream |

**Proven methodology (see `notes.md §7`, `results.md`):**
- **Adjacency-constrained decoding** (`training/constrained_decode.py`): masks the next-voxel
  logits to the 6-neighborhood of placed voxels → **validity 1.0 by construction**. This is the
  single biggest quality lever and *dominates* the phase4 PE gain (they don't stack — §7).
- **Category conditioning** = best cohesion lever (overnight battery T10).
- **3D-BPE cluster tokens** (`tokenizers/cluster_bpe.py`) = anti-memorization.
- **Honest novelty**: D4 augmentation + held-out val NN-IoU + duplicate rate (`eval/novelty.py`).
- **Cross-medium transfer**: pool-pretrain→finetune gives *no in-domain gain* when the target is
  data-rich (T12) — the transfer claim is reframed for the data-POOR LEGO regime.

---

## 6. Reproducing experiments

Each `experiments_*.py` is a self-contained run driver writing to `outputs/run_<stamp>_<name>/`
(with `metrics.json`, `leaderboard.md`, per-arm `novelty.json`, `samples.png`). All are
resumable from their per-arm `novelty.json`.

```bash
# Ideas ablation battery (PE variants, BPE, conditioning, constrained decode)
.venv/bin/python -m blockgen.experiments_ideas --stamp $(date +%Y%m%d_%H%M%S)

# Overnight cohesion + data battery
.venv/bin/python -m blockgen.experiments_overnight --stamp $(date +%Y%m%d_%H%M%S)

# phase4 + constrained combo (two proven winners)
.venv/bin/python -m blockgen.experiments_p4c --stamp $(date +%Y%m%d_%H%M%S)

# Pool-pretrain → finetune on houses_32 (cross-medium transfer evidence)
.venv/bin/python -m blockgen.experiments_transfer --stamp $(date +%Y%m%d_%H%M%S)
.venv/bin/python -m blockgen.experiments_transfer --stamp smoke --quick   # fast wiring check
```

Configs for parameterized runs live in `configs/{datasets,experiments,models,training}/*.yaml`
(loaded via `blockgen/config.py`).

---

## 7. Rendering structures

Two renderers. **Use the textured one for figures.**

**Single structure → PNG** (real Minecraft textures, headless EGL, ortho by default):
```bash
.venv/bin/python -m blockgen.renderer.textured \
    --cache data/minecraft/cache/houses_32.npz --index 0 --out outputs/figures/one.png --px 512
```
```python
from blockgen.renderer.textured import render_structure
img = render_structure(structure, px=512, azim_deg=45, elev_deg=30, ortho=True)  # (px,px,4) RGBA
```
Baked face shading (top 1.0 / N–S 0.85 / E–W 0.70 / bottom 0.55); exposed-face culling; ~0.5–3 s/img.
~97% block-face texture coverage with a solid-color fallback.

**Dense sample grid** (`blockgen/renderer/grid.py`) — textured by default, `--matplotlib` to force fallback:
```bash
# curated house cache, filtered to one corpus
.venv/bin/python -m blockgen.renderer.grid --houses 32 --corpus grabcraft --rows 8 --cols 12 \
    --out outputs/figures/houses_32_grabcraft.png
# any .npz structure cache
.venv/bin/python -m blockgen.renderer.grid --cache <path.npz> --rows 6 --cols 8 --out grid.png
```

**Model sample figures** (rebuild vocab, reload checkpoints, sample fresh, render textured grids):
```bash
.venv/bin/python scripts/render_model_samples.py    --samples 48   # ideas-battery checkpoints
.venv/bin/python scripts/render_transfer_samples.py --samples 48   # transfer-run checkpoints
```
Outputs land in `outputs/figures/` (e.g. `samples_ar_pe_phase4.png`,
`samples_transfer_{scratch,pretrain_zeroshot,finetune}.png`).

**LEGO / LDraw structures** use a separate package, `legogen/renderer/` (Blender 4.2 +
ImportLDraw — the canonical LDraw pipeline), kept apart from `blockgen` for now. It renders
dense dataset sheets of the OMR corpus to `outputs/figures/lego_data/`:
```bash
.venv/bin/python legogen/renderer/ldraw_grid.py --rows 8 --cols 12 --band 15 160 \
    --out outputs/figures/lego_data/omr_grid.png
```
See `legogen/renderer/README.md`.

---

## 8. Evaluation

`eval/novelty.py` reports per-run: `mean_nn_iou` (train), `val_nn_iou` (held-out — the honest
number), `diversity`, `duplicate_rate`, `validity_rate` (+ `validity_gated`). `eval/validity.py`
gives connectivity (`n_components`), largest-connected-component repair (`repair_lcc`), and
`gated_sample` (resample until valid). Lower val NN-IoU + high diversity + zero duplicates =
genuinely new structures, not memorized copies.

---

## Documentation map

| File | Contents |
|---|---|
| `README.md` | this file — how to use the codebase |
| `data_sources.md` | **consolidated data inventory**: every MC + LEGO corpus — counts, licenses, labels, sample sheets |
| `notes.md` | research notes: data pipeline, methods, proven ideas §7, dead ends §8, idea backlog §9, rendering §13 |
| `results.md` | every run's metrics tables (T-series) + figure index |
| `research.md` | field survey; §E = post-LegoGPT LEGO landscape & the typed-connection bet |
| `roadmap.md` | paper thesis + 3-phase plan (LEGO data → typed-connection AR → cross-medium transfer) |
| `references.md` | paper bibliography |
| `data/lego/README.md` | LEGO corpora, licenses, connectivity-coverage stats |
| `configs/README.md` | experiment config schema |
```
