# Supplementary code

Code and recorded runs for the paper. It contains the evaluation suite, the
corruption probes, the procedural baselines, the pinned corpus splits, the
human-study instrument with every rater response, and the recorded runs that
every number in the paper is read from. Nothing is transcribed by hand: the
figures read the runs, and a registered check re-derives each number.

```
blockgen/eval/bench/    the suite: runner, tiers, probes, baselines, ladder, splits, scorecards
blockgen/curation/      corpus rebuild (houses.py) and quality gates
blockgen/renderer/      headless textured renderer used by the render tier
legogen/renderer/       LDraw parsing and voxelization for the LEGO domain
scripts/audit_claims.py every number in the paper, recomputed from the runs below
scripts/paper_figures.py, scripts/figs/   the figures
scripts/human_study_*.py                  export, serve and analyse the rater study
tests/eval/             unit tests for the suite
data/minecraft/splits/  the pinned group-aware splits (index lists)
data/minecraft/cache/   corpus manifests: source URL, title and size per build, no content
outputs/                the recorded runs (scorecards, dose curves, ablation, human study)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -e .        # audit, tests, FAST tier, recorded-run figures
pip install -r requirements-full.txt        # adds the render tier, the LEGO loader, the corpus rebuild
pip install --no-deps pyrender==0.1.45      # pins an older PyOpenGL; its other needs are in the file above
```

Python 3.10 or newer, Linux. Rendering is headless through EGL. A CUDA device
is used when present and everything falls back to CPU.

## Check the paper's numbers

```bash
python scripts/audit_claims.py              # --verbose shows each derivation
```

Every registered claim is recomputed from a recorded run under `outputs/`.
Claims marked `CORP` read the corpus itself, which this release does not
contain (see *What is not included*); they are checked once the corpus is
rebuilt. The run of record for the case study is
`outputs/run_20260917_043807_bench_canonical-split_native_arms_oriented_vs/scorecard.json`.
Its `run` block records the command, seed, split hash, host and commit.

## Tests

```bash
python -m pytest tests/eval -m "not slow"   # ~20 s, no corpus, no GPU
python -m pytest tests/eval                 # adds tests that render or embed (full requirements)
```

Two of the slow tests read the corpus (`test_palette.py::test_family_map_covers_the_real_corpus`,
`test_splits.py::test_real_corpus_split_has_no_leakage`) and fail until it is rebuilt.

## What runs from the recorded runs alone

These need neither the corpus nor a GPU:

```bash
python scripts/paper_figures.py fig2 fig3 fig4      # dose-response, pillars, case study
python scripts/figs/ladder.py                       # the validation ladder
python scripts/human_study_analyze.py \
    --key outputs/run_20260923_human_v5/answer_key.json \
    --responses outputs/run_20260923_human_v5/rater_*.json \
    --scorecard outputs/run_20260917_043807_bench_canonical-split_native_arms_oriented_vs/scorecard.json
```

Everything else reads structures and needs the corpus below.

## Rebuilding the corpus

The Minecraft corpus pools three public sources. We redistribute none of their
content; the manifests in `data/minecraft/cache/` list every build's source
URL, title, size and a hash of its voxel content, and the curation is
deterministic, so the corpus rebuilds from the sources and can be checked
build by build.

1. **3D-Craft** (CraftAssist house corpus, CC BY-NC 4.0). The house data is
   linked from https://github.com/facebookresearch/voxelcnn. Unpack it so that
   each house is `data/minecraft/3d_craft/houses/<house>/schematic.npy`.
2. **GrabCraft.** `python -m blockgen.data.grabcraft_scraper --category houses`
   writes one JSON per build to `data/minecraft/grabcraft/raw/` (polite,
   resumable; the site is live, so the set can drift from the manifest), then
   `python -m blockgen.data.grabcraft_dataset --max-dim 32` builds the
   intermediate GrabCraft cache the curation reads.
3. **text2mc** (PlanetMinecraft builds, https://github.com/shauncomino/text2mc-dataprocessor).
   Place `processed_build_dataframe.csv`, `tok2block.json` and
   `processed_builds/*.h5` under `data/minecraft/text2mc/`. The per-file
   shape index the curation reads is included as `data/minecraft/text2mc_index.json`.

Then curate and pin. `houses_32` pools exactly these three sources:

```bash
python -m blockgen.curation.houses --max-dim 32 --include grabcraft 3dcraft text2mc
python -m blockgen.eval.bench.splits --corpus houses_32 --seed 0 --show
```

The split is loaded from `data/minecraft/splits/houses_32.s0.70-15-15.v1.json`.
It carries a fingerprint of the manifest it was cut from (source, title and
category of every build, with local paths in portable form), and the loader
refuses a corpus whose fingerprint differs rather than silently re-cutting the
split. If your rebuild is refused, diff your manifest against
`data/minecraft/cache/houses_32_manifest.json`: `source_path` says which
upstream builds changed and `content_sha` whether a build's voxels did. The
scorecards under `outputs/` cite the split by its earlier fingerprint, kept in
the split file as `source_sha_legacy`.

## Running the suite

```bash
# FAST tier: structural metrics only. Seconds, no rendering.
python -m blockgen.eval.bench --n 128 --baselines --name "fast"

# Validation ladder, both metric families
python -m blockgen.eval.bench.ladder --family geometry --n 64 --n-ref 256
python -m blockgen.eval.bench.features --backbone dinov2b        # warms the feature cache
python -m blockgen.eval.bench.ladder --family render --n 64 --n-ref 256

# FULL tier on the arms scored in the paper (the run of record's command is in its scorecard)
python -m blockgen.eval.bench --tier both --n 128 --baselines --name "canonical" \
    --arms ar/canon_oriented:outputs/bench_arms/canon_oriented_32.npz \
           ar/canon_unoriented:outputs/bench_arms/canon_unoriented_32.npz \
           ar/native_oriented:outputs/bench_arms/native_oriented_32.npz \
           llm/llm_control:outputs/bench_arms/llm_control_32.npz \
           llm/llm_bias:outputs/bench_arms/llm_bias_32.npz

# Dose-response and the view-protocol ablation
python scripts/bench_doseresponse.py --n 128
python scripts/ablate_view_protocol.py --n 256 --n-ref 512
```

Each run writes `outputs/run_<timestamp>_<name>/scorecard.{json,md}`.
Controls are re-scored in every run and never cached across runs, because the
aggregate's unit is the spread of a control scored under the same protocol.

## Figures

Numbered as in the paper. "recorded" means the script reads only files in
`outputs/`; "corpus" means it reads structures and needs the rebuilt corpus;
"render" additionally needs `pyrender` and the textures.

| paper figure | file | script | needs |
|---|---|---|---|
| 1 teaser | `fig1_teaser` | `scripts/paper_figures.py fig1` | corpus, render |
| 2 ladder | `fig10_ladder` | `scripts/figs/ladder.py` | recorded |
| 3 cutaway | `fig9_cutaway` | `scripts/figs/cutaway.py` | corpus, render |
| 4 ablation | `fig5_ablation` | `scripts/paper_figures.py fig5` | recorded |
| 5 contamination | `fig_contamination` | `scripts/figs/contamination.py` | corpus |
| 6 LEGO OMR | `fig8_lego_omr` | `scripts/figs/lego_ldraw.py` | OMR models, Blender (`BLENDER=/path/to/blender`) |
| 7 influence | `fig11_influence` | `scripts/build_influence.py`, then `scripts/figs/influence_pair.py` | corpus |
| 8 pipeline | `fig_pipeline` | `scripts/figs/pipeline.py` | corpus, render |
| 9 probe gallery | `fig7_probe_gallery` | `scripts/paper_figures.py fig7` | corpus, render |
| 10 dose-response | `fig2_dose_response` | `scripts/paper_figures.py fig2` | recorded |
| 11 seven statistics | `fig6_seven_statistics` | `scripts/paper_figures.py fig6` | recorded |
| 12 pillars | `fig3_pillars` | `scripts/paper_figures.py fig3` | recorded |
| 13 case study | `fig4_casestudy` | `scripts/paper_figures.py fig4` | recorded |
| 14 descriptor | `fig_descriptor` | `scripts/figs/descriptor.py` | corpus |
| 15 samples | `fig_samples` | `scripts/figs/samples.py` | corpus, render |

`scripts/figs/*` write to `outputs/figstage/`; `paper_figures.py` writes to
`deliverables/paper/iclr/figures/`. The small JSON files under
`outputs/figstage/` are written by the check scripts of the same name
(`contiguous_draw_check.py`, `interior_counts.py`, `math_hull_check.py`,
`rescore_flat_opaque.py`, `influence_pair.py`) and by `scripts/vkid_ablation.py`;
they are what the audit reads. The converse-probe run is
`outputs/run_20260918_203947_bench_hollow/` (`scripts/bench_hollow.py`).

The render tier and the figures use the game's block textures, which are not
redistributed. `python -m blockgen.renderer.textures --fetch` downloads the
current client from the publisher and extracts them to `data/textures/vanilla/`.
Without the pack the renderer falls back to a flat colour per block, which
renders but does not reproduce the paper's render-tier numbers.

## The LEGO domain

```bash
python -m blockgen.eval.bench.lego --limit 3000            # StableText2Brick -> lego_void_20.npz
python -m blockgen.eval.bench --corpus lego_void_20 --tier both --n 256
python -m blockgen.data.fetch_omr                          # the OMR models, for the LEGO figure only
```

The loader reads the StableText2Brick parquet files (Hugging Face dataset
`AvaLovelace/StableText2Brick`, MIT) from `data/lego/stabletext2brick/data/`.
The pinned split is in `data/minecraft/splits/`.

## Human study

`outputs/run_20260923_human_v5/` holds the instrument (`study.html`, self
contained), the answer key, all twenty rater responses and the analysis the
paper reports. To rebuild the instrument from the corpus:

```bash
python scripts/human_study_export.py --per-arm 24 --sessions 40 --out outputs/run_<name>
python scripts/human_study_serve.py --run outputs/run_<name>     # serves it, collects rater_<session>.json
```

## What is not included

- **Corpus content.** GrabCraft permits sharing blueprints only by link and
  PlanetMinecraft prohibits redistributing site data. The release carries the
  curation code, the manifests and the split, not the structures.
- **Textures.** The game's assets belong to their publisher.
- **Model weights and prompts.** The scored arms are included as their sample
  files under `outputs/bench_arms/`, so the case study re-scores from them. The
  models that produced the samples are not part of this paper's contribution.
- **The lab.** An interactive viewer for runs and leaderboards is not needed
  to reproduce any number and is left out.

## License

GPL-3.0. Third-party data keep their own terms: 3D-Craft is CC BY-NC 4.0, the
LDraw OMR is CC BY 2.0, StableText2Brick is MIT.
