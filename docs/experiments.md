# Experiments & outputs

Two reproducible runners. Both write a self-contained run directory under `outputs/`
(gitignored) with a config manifest (git sha + all train configs), figures, per-model
samples/metrics, and a `run_notes.md`.

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
