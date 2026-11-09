# BlockGen

Generative models for discrete 3D buildable structures: Minecraft voxel builds today, LEGO
brick assemblies next. The research thesis: guaranteed-buildable generation with diverse
parts, trained on open data, transferred across media, with honest novelty accounting.
The plan is `research.md` §I; the field survey is the rest of `research.md`.

This README is the practical guide: how the repo is laid out, how to run each part, and
where the documentation lives. The MkDocs site under `docs/` (start at `docs/index.md`)
has the long form.

---

## 1. Setup

The pipeline's interpreter is the conda `python` on `PATH` (torch 2.7 + CUDA 12.8,
pandas 3.0). The corpus layer and the lab need pandas; the project `.venv` (torch 2.10)
has no pandas, so every command below says `python`. Two exceptions still run under
`.venv/bin/python`: the legacy batteries that need `nbtschematic`, and the MinecraftACE
trainers under `libs/MinecraftACE/train/`, which need `diffusers` (`docs/minecraftace.md`).

```bash
pip install -e .                 # base package
pip install -e ".[render]"       # textured rendering: pyrender, trimesh, PyOpenGL, pillow
pip install -e ".[gnn]"          # graph track: torch-geometric
python -c "import torch, pandas; print(torch.__version__, torch.cuda.is_available())"
```

Linux, one 16 GB GPU (RTX 5070 Ti in development), headless rendering through EGL.
Tests: `PYTHONPATH=. python -m pytest tests/corpus tests/train tests/posttrain -q`.

---

## 2. Data layout

```
data/
  minecraft/
    cache/            curated caches: houses_32.npz, houses_48.npz, all_32.npz (+ _manifest.json)
    grabcraft/        GrabCraft category-labelled builds
    3d_craft/         3D-Craft house corpus with placement traces
    text2mc/          text2mc .h5 builds and 28k raw .schem
    more/             tfrecord-derived labelled crawl
    raw/              legacy schematics (drifted metadata)
    templates/        1.21.1 structure templates (internal only)
    worldcrop/        <month>/crops.jsonl + crops/*.npz, gate/, logs/   (pretraining only)
    corpus/           manifest/*.parquet, dedup.parquet, kinds.parquet, tiers/, shards/
    splits/           houses_32.s0.70-15-15.v1.json (the pinned benchmark split)
  lego/               OMR, StableText2Brick, LDraw, Shadow Library (data/lego/README.md)
  textures/vanilla/   extracted block textures
```

`docs/data.md` follows a build from source to shard. The short version:

```bash
python -m blockgen.curation.houses --max-dim 32                   # houses_32.npz
python -m blockgen.corpus.manifest curated --caches all_32 houses_32 houses_48
python -m blockgen.corpus.dedup && python -m blockgen.corpus.kinds
python -m blockgen.corpus.tiers build                             # configs/corpus/*.yaml -> tiers/
python -m blockgen.corpus.shards build sft-houses32 --max-blocks 8192
python -m blockgen.corpus.shards build eval-holdout --max-blocks 8192
bash scripts/corpus_refresh.sh --dry-run                          # the whole chain, in order
python -m blockgen.renderer.textures --fetch                      # block textures
```

---

## 3. Package map

```
blockgen/
  data/         corpus readers (grabcraft, tfrecord, legacy cache) and worldcrop/ (archived
                worlds -> crops, terrain gate)
  curation/     houses (cross-corpus house caches), corpus (all_32), curate (Curator)
  corpus/       manifest, canon + dedup, kinds, tiers, tokenizer (voxels | facades), shards,
                facade, reps, templates                                  -> docs/data.md
  train/        the shared trainer: config, archs (registry), data, trainer, sample, registry
                                                                         -> docs/pipeline.md
  posttrain/    rewards (pillar refusal), bon (best-of-N), rft (rejection-sampling FT), select
  utils/        Structure, serialize (legacy tokens/grids), corpora, block_remap, schem, growth_order
  tokenizers/   standard_vocab, cluster_bpe (3D-BPE pieces), piece_factors
  models/       voxel_transformer_ar2 (Track A), voxel_transformer_cond + condition_resampler,
                voxel_diffusion (B), large_pyg_graph_generator (C), pick_n_place (F), ...
  training/     legacy trainers per family + constrained_decode
  eval/         novelty, validity, traversal (player walk), facade_stats, bench/ (BlockScore,
                ladder, protocol, pointcloud)                             -> docs/benchmark.md
  agentic/      Track E: an LLM writes the build program                 -> docs/agentic.md
  ontology/     what each block is, mined from the corpus                -> docs/ontology.md
  labeling/     renders -> captions -> frozen embeddings                 -> docs/labeling.md
  renderer/     textured (EGL), textures, grid
  export/       converters (MinecraftACE layout)
  experiments_*.py   legacy run drivers (T10 to T24)
tools/lab/      BlockLab, the local app                                  -> docs/lab.md
configs/        corpus/ (tiers), train/ (trainer stages), and the legacy datasets/experiments/models/training/agentic
scripts/        m2b_seeds.sh, corpus_refresh.sh, facade_oracle.py, facade_baselines.py, figure scripts
deploy/         live /gen demo: inference server + Minecraft mod
libs/           MinecraftACE (LegoACE port), BrickNet, BrickAgent
```

Two letter schemes: a **Track** is a generator family (A autoregressive, B diffusion,
C graph, D LLM baseline, E agentic, F pick-and-place; `docs/models.md`); an **Arm** is a
tokenisation story in the shared trainer (A sparse raster, B 3D-BPE, C dense grid, D text
lines, E facades; `docs/pipeline.md` §4).

---

## 4. Representation

- `Structure` (`blockgen/utils/data.py`): `block_ids` + `block_data` arrays in the legacy
  `(id, data)` vocabulary, `occupied_mask`, `crop_to_non_air()`.
- `CorpusTokenizer` (`blockgen/corpus/tokenizer.py`): kind `voxels` writes
  `[BOS, src, q, size, win, (X, Y, Z, BLOCK)*, EOS]` over a fixed 507-token vocabulary
  (hash `60684456e598`); kind `facades` writes six `(depth, id)` face images per build,
  one sequence per face or all faces jointly. A shard is a tier under one tokenizer hash.
- Legacy streams and grids: `blockgen/utils/serialize.py` (`BlockVocab` per run,
  `structure_to_grid`); 3D-BPE pieces: `blockgen/tokenizers/cluster_bpe.py`.
- D4 symmetry: `blockgen/utils/augment.py`; shard-level `--augment d4`.

---

## 5. Training

```bash
python -m blockgen.train --config configs/train/smoke.yaml                 # 60-step wiring check (GPU, ~2 min)
python -m blockgen.train --config configs/train/anneal-pretrain-5m.yaml     # pretrain on the anneal tier
python -m blockgen.train --config configs/train/houses32-sft-5m.yaml init_from=outputs/<pretrain run>/last.pt
python -m blockgen.train --config configs/train/houses32-facades-joint-d4-20m.yaml
python -m blockgen.train --sample outputs/<run> --n 399                    # structure cache for the bench
bash scripts/m2b_seeds.sh                                                  # the three-seed pretrain-vs-scratch recipe (T60)
```

Runs land in `outputs/run_<stamp>_<name>/` with `run.json`, `metrics.jsonl`, `best.pt`,
`last.pt`, `samples.png`, `summary.json` and `bench/scorecard.json`, plus one line in
`outputs/registry.jsonl`. Rules that cost a run before: no dropout change at `init_from`
(T59), `select=best` on holdout NLL, pin `tokenizer:` in the YAML.

Post-training:

```bash
python -m blockgen.posttrain.bon --run outputs/<run> --n 256 --k 8 --reward face_solidity
python -m blockgen.posttrain.rft --run outputs/<run> --reward lcc26 --n 2048 --keep-frac 0.25
```

Rewards are `face_solidity`, `lcc26`, `novelty_penalty` or a `--weights` composite; a
BlockScore pillar input is refused as a reward. Legacy batteries still produce Tracks B,
C and F (`python -m blockgen.experiments_ideas --stamp ...`, `docs/experiments.md`) and the
agentic track runs through `python -m blockgen.experiments_agentic --config
agentic-scaffolding` (API calls; `--provider mock` is offline).

---

## 6. Evaluation

```bash
python -m blockgen.eval.bench --tier both --n 399 --name "my arm" --arms ar/my_arm:outputs/<run>/samples_step399_48.npz
python -m tools.lab --open          # http://127.0.0.1:8765/leaderboard
```

`blockgen/eval/bench` ranks every arm under protocol `houses32-v1`: tier `both`, at least
128 samples per arm (256 or 399 by convention), real controls (`real_test`,
`real@solidify`, `real@monochrome`, ...) and procedural baselines added automatically.
`BlockScore` is the worst pillar in real-sample spreads; the novelty gates disqualify a
memoriser. Always `--name` a run. The lab reads `outputs/*/scorecard.json` and
`outputs/*/bench/scorecard.json`, so a trainer run's own card is on `/runs`. Full
description: `docs/benchmark.md`; human 2AFC and labels: `python -m tools.lab --phone`.

---

## 7. Rendering

```bash
python -m blockgen.renderer.textured --cache data/minecraft/cache/houses_32.npz --index 0 \
    --out outputs/figures/one.png --px 512
python -m blockgen.renderer.grid --houses 32 --corpus grabcraft --rows 8 --cols 12 \
    --out outputs/figures/houses_32_grabcraft.png
python -m blockgen.renderer.grid --cache <path.npz> --rows 6 --cols 8 --out grid.png
```

Headless EGL with real block textures; matplotlib voxels as the fallback. LEGO sheets use
`legogen/renderer/ldraw_grid.py` (Blender 4.2 + ImportLDraw).

---

## Documentation map

| File | Contents |
|---|---|
| `docs/index.md` | site home: what's here, naming, headline results |
| `docs/data.md` | sources, curation, manifest, dedup and kinds, tiers, shards |
| `docs/pipeline.md` | the shared trainer, stages, post-training, the registry |
| `docs/models.md` | the tracks and their best numbers |
| `docs/architecture.md` | the codebase map and the model contract |
| `docs/representations.md` | token streams, grids, graphs, growth, facades |
| `docs/benchmark.md` | BlockScore, the ladder, the protocol |
| `docs/lab.md` | BlockLab |
| `docs/agentic.md`, `docs/agentic-suite.md` | Track E |
| `docs/minecraftace.md` | the LegoACE port |
| `docs/getting-started.md`, `docs/workflow.md` | environment, data, the train / sample / bench / view route |
| `docs/custom-model.md` | the model contract |
| `docs/add-a-model.md` | adding a model end to end, from an empty editor to a board row |
| `docs/status.md` | generated matrix: every option used, unused, planned or stale (`scripts/status_matrix.py`) |
| `docs/experiments.md`, `docs/scripts.md` | the T-series index with run dirs; every script and CLI |
| `data_sources.md` | per-corpus inventory with licences and sheets |
| `notes.md` | research notes, the methods ledger |
| `results.md` | every run's tables (T-series) |
| `research.md` | field survey; §F data sourcing, §G the training stack, §I the paper plan |
| `references.md` | bibliography |
| `configs/corpus/README.md`, `configs/README.md` | config schemas |
