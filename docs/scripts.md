# Scripts

Everything in `scripts/` is a runnable entry point, and the pipeline packages
(`blockgen.corpus`, `blockgen.train`, `blockgen.posttrain`, `blockgen.eval.bench`)
expose theirs as `python -m` modules. Library code lives in `blockgen/`; these
are the things you *run*. Only the first three groups are part of the
[standard workflow](workflow.md). The rest are the one-off instruments that
produced particular results, kept because a probe worth running once is usually
worth running again.

Most scripts take `--help`. The growth and pick-and-place scripts (`attach_*.py`,
`pnp_prefix_test.py`, `ordering_bakeoff.py`, `train_attach_corpus.py`,
`train_pick_n_place.py`) import `torch_geometric` through `blockgen.utils.graph_data`
and run only under `.venv/bin/python`. `scope_piece_tokens.py`, `vkid_ablation.py`,
`facade_sheets.py` and the `figs/` modules have no argparse; running them is running
the computation. Most scripts write into `outputs/run_<stamp>_<name>/` via
`blockgen.utils.runs.new_run_dir`.

**Interpreter.** The commands below assume the conda `python` on `PATH`
(`/home/griffing52/miniconda3/bin/python`, torch 2.7 and pandas 3.0). The repo
also carries `.venv/bin/python`, which the legacy batteries were run with; it has
no pandas, so every `blockgen.corpus` command fails there with
`ModuleNotFoundError`. The conda interpreter has no `torch_geometric`, so the
graph scripts named above fail there the same way. Module commands (`python -m ...`)
find the package from the repo root on their own; script paths (`python scripts/x.py`)
need `PYTHONPATH=.`.

## Corpus layer

Module CLIs over the manifest, in the order each one reads the one before it.
The schemas are in [Pipeline](pipeline.md) §3.

| command | what it does |
|---|---|
| `python -m blockgen.corpus.manifest build [--workers 6] [--months 201407 ...] [--limit N]` | one manifest parquet per world-crop month under `data/minecraft/corpus/manifest/`; resumable per month |
| `python -m blockgen.corpus.manifest curated [--caches all_32 houses_32 houses_48]` | manifest rows for the curated caches, default `all_32 houses_32 houses_48`; `houses_32` is the cache the `sft-houses32` and `eval-holdout` tiers index, so leaving it out empties both tiers (`blockgen/corpus/manifest.py:305-306`) |
| `python -m blockgen.corpus.manifest summary` | row counts per source |
| `python -m blockgen.corpus.templates [--jar PATH] [--workers 6]` | the 1.21.1 client jar's structure templates as `data/minecraft/templates/<family>/<name>.npz` plus `templates.parquet` (T56) |
| `python -m blockgen.corpus.dedup [--bands 16] [--jaccard 0.8] [--min-blocks 20]` | exact and MinHash duplicate clusters, `dedup.parquet` (T55) |
| `python -m blockgen.corpus.kinds` | `build` / `village` / `mineshaft` / `mixed` per row, `kinds.parquet`. **Takes no arguments and parses none: `--help` runs the full pass** (`blockgen/corpus/kinds.py:79`) |
| `python -m blockgen.corpus.tiers build [names ...]` | materialise `configs/corpus/<name>.yaml` to `tiers/<name>.{json,ids.txt}`; all configs when no names; raises `EmptyTierError` for an empty split-backed tier |
| `python -m blockgen.corpus.tiers show <name>` | counts and hash of a built tier |
| `python -m blockgen.corpus.shards build <tier> [--kind voxels\|facades] [--layout per_face\|joint] [--faces all\|sides\|sides_top] [--augment d4] [--oriented] [--max-blocks N] [--max-windows N] [--workers N]` | tokenise a tier into `shards/<tier>/<tokenizer hash>[-d4]/` |
| `python -m blockgen.corpus.shards show <tier> [--hash H]` | what a shard holds |
| `python -m blockgen.corpus.shards from-npz <npz> --tier <name> [--kind ...] [--layout ...] [--tokenizer HASH] [--tags k=v ...] [--source NAME] [--max-blocks N]` | a derived tier plus shard from a structure cache (model samples a reward kept); the tier name may not collide with a `configs/corpus/*.yaml` name |

Two shell drivers sit on top:

| script | what it does |
|---|---|
| `corpus_refresh.sh [--dry-run] [--workers N]` | the whole order, `manifest build` through `shards build`, with `set -euo pipefail`; the shard stage rebuilds every (tier, tokenizer) pair that has a shard on disk. Validated with `--dry-run` only |
| `worldcrop_ingest.sh [WORKERS] [MONTH ...]`, `worldcrop_resume.sh`, `worldcrop_retry.sh`, `worldcrop_pause.sh pause\|resume\|status` | stream Minecraft Archive Project months through the world-crop extractor, resume the full-archive pass, re-run failed months, pause the workers without losing progress. See [World crops](worldcrop.md) |

```bash
PYTHONPATH=. bash scripts/corpus_refresh.sh --dry-run      # print the command list
python -m blockgen.corpus.tiers build sft-houses32 eval-holdout
python -m blockgen.corpus.shards build sft-houses32 --max-blocks 8192
```

## Training and post-training

| command | what it does |
|---|---|
| `python -m blockgen.train --config configs/train/<x>.yaml [key=value ...]` | train one run; any `TrainConfig` field is an override (`seed=1 name=x init_from=<run>/last.pt train=a:1,b:1 tokenizer=<hash> select=best`); an unknown YAML key is an error |
| `python -m blockgen.train --sample <run_dir> --n 399 [--ckpt best.pt] [--out-dir D] [temperature=0.7 ...]` | sample a finished run into the structure cache the bench scores, `samples_step<n>_<dim>.npz`, with `report.train_split` set from the config; the route every trainer entry on the board took (the legacy `canon_oriented` row came through `scripts/sample_to_npz.py`) |
| `python -m blockgen.posttrain.bon --run <run_dir> --n 256 --k 8 --reward face_solidity\|lcc26\|novelty_penalty [--weights name:w,...] [--temperature T] [--top-k K] [--batch B] [--ckpt best.pt] [--out-dir D] [--seed S]` | best-of-N with a verifier: `n*k` draws, keep the reward's favourite of every `k`; writes `<run>/bon_<reward>_k<k>_n<n>/bon_<dim>.npz`, `scores.parquet`, `report.json`, and prints the `--arms` spec |
| `python -m blockgen.posttrain.rft --run <run_dir> --reward <r> --n 2048 --keep-frac 0.25 [...]` | rejection-sampling fine-tuning: best-of-N, then `shards.from_structures` into the tier `rft-<run name>-<reward>`, then the printed `python -m blockgen.train` command that fine-tunes the parent on it mixed one-to-one with `sft-houses32` |

Configs: `configs/train/base.yaml` and the recipes `smoke.yaml`,
`smoke-facades.yaml`, `houses32-scratch.yaml`, `houses32-scratch-matched.yaml`,
`anneal-pretrain-5m.yaml`, `houses32-sft-5m.yaml`, `houses32-facades*.yaml`,
`rft-voxel-5m.yaml`, `rft-facades-joint-d4-20m.yaml`. The stage rules and the
run-directory layout are in [Pipeline](pipeline.md) §5.

| script | what it does |
|---|---|
| `m2_seeds.sh` | the M2 pilot (T59): per seed, matched scratch, pretrain on `anneal`, SFT from that checkpoint; `SEEDS="0 1 2"` by default, logs under `outputs/logs/` |
| `m2b_seeds.sh` | the M2b rerun (T60): reuses the M2 pretrain checkpoints, SFT at dropout 0 with `select=best`, matched scratch |
| `posttrain_first_pass.sh` | the first pass through the post-training ladder on the two models that exist: best-of-4 and RFT on `lcc26` for the voxel M2b SFT model, on `face_solidity` for the joint facade D4 model, each ending in a `--tier both` card. `PART=voxel\|facade\|both`, `BATCH_V`, `BATCH_F`; run detached |
| `facade_sheets.py [name=path ...]` | real against generated face sheets, and where the fragmentation lives, faces or edges; imports `face_stats` from `blockgen.eval.facade_stats`. No argparse: arguments are `name=path` pairs (`scripts/facade_sheets.py:31`), so `--help` is rejected; the default arms are the `ARMS` dict at `:10-15` and output goes to `outputs/analysis/facade_sheets/` |

```bash
python -m blockgen.train --config configs/train/smoke.yaml            # 60 steps, bench included
python -m blockgen.train --sample outputs/run_<stamp>_<name> --n 399
python -m blockgen.posttrain.bon --run outputs/run_<stamp>_<name> --n 256 --k 4 --reward lcc26
```

## Pipeline: the paved path

| script | what it does |
|---|---|
| `sample_to_npz.py --train-split canonical ...` | Sample a served legacy checkpoint into the standard structure-cache `.npz`; `--train-split` is what the board ranks on |
| `dump_samples.py` | Convert generated samples (`.npy` dir, keyed `.npz`, existing cache) into that same format; records no `train_split`, so the arm is unranked until the manifest says so |
| `sample_geobias.py` | Sample a `train_llm_geobias.py` run into a benchmark arm |

These are the seam between "a model" and "an evaluation arm": all write the
`.npz` + `_manifest.json` pair the benchmark and BlockLab understand, and all
record a provenance `report` (model, checkpoint, seed, temperature, top-k) that
survives into the leaderboard row. `python -m blockgen.train --sample` is the
same seam for trainer runs.

## Legacy training scripts

The arms that predate `blockgen.train`, each with its own loop.

| script | what it does |
|---|---|
| `train_uncond_corpus.py` | Unconditioned `native_bpe` on the full corpus, the diagnostic run |
| `train_cond_resampler.py` | Text-conditioned piece AR with a resampled conditioning channel (idea #6) |
| `train_pick_n_place.py` | The pick-and-place growth model; trains, samples, renders and dumps for the bench |
| `train_attach_corpus.py` | Attachment/growth AR, one arm per ordering |
| `train_llm_pieces.py` | BrickGPT-style text to build over 3D-BPE *pieces* rather than voxels |
| `train_llm_brickgpt.py` | BrickGPT/LegoGPT-style text to build with a LoRA-finetuned LLM |
| `train_llm_geobias.py` | Track D plus a structural attention bias: inject adjacency instead of extracting it |

## Evaluation, reporting and the human study

| script | what it does |
|---|---|
| `bench_report.py` | A compact cross-arm table from one or more scorecards |
| `bench_doseresponse.py` | Does the benchmark reward *more realistic* output, or only detect artifacts? (T26) |
| `bench_hollow.py` | The converse probe: is the render tier blind to hollowing as well as filling? |
| `bench_traversal.py` | The traversability probe run as a bench: real, sealed-opening controls and every generated arm, with the walk rendered (T63). `blockgen.eval.traversal` has no CLI of its own |
| `facade_oracle.py` | Real houses reduced to their visible shell, `real@facades4/5/6`, as control rungs on a card (T65) |
| `facade_baselines.py [--n 399] [--faces all\|sides] [--no-run]` | The facade baselines as arms under `outputs/bench_arms/`, scored beside the oracle (T66) |
| `ablate_view_protocol.py` | Is the interior blindness a DINOv2 artifact or a property of the view protocol? (T37) |
| `vkid_ablation.py` | What in the geometric descriptor detects interior filling, and is it counting blocks? (T39) |
| `build_influence.py` | Which individual builds push an arm away from real: KID's exact per-build decomposition |
| `curation_label_sheet.py` | Blind labelling sheet: does KID's influence tail contain worse builds? Pre-registered |
| `human_study_export.py` | Export a 2AFC study, stimuli plus a self-contained page |
| `human_study_serve.py --run <dir>` | Serve that page and collect responses straight to disk, one file per session |
| `human_study_analyze.py` | Exclusion, Bradley-Terry, metric-human agreement; `--lab-db outputs/lab/lab.db` reads BlockLab's compares |
| `validate_perceptual.py` | Is the perceptual metric trustworthy? Tested where the answer is known |
| `score_arms_perceptual.py` | Adjudicate the T20 arms with the perceptual metric |
| `eval_mv_conditioning.py` | Does multi-view conditioning carry structure-*specific* information? |
| `sample_conditioned.py` | Sample a conditioned run and evaluate fidelity |

`human_study_export.py` strips arm labels from the page and asserts it by grep: a
condition visible in the DOM is a condition the rater can read.

## Paper and status

| script | what it does |
|---|---|
| `paper_figures.py [--all \| fig1 ...]` | Every figure for the submission, read out of `outputs/` at run time; `deliverables/paper/iclr/figures/*.pdf` plus a PNG each |
| `paper_readiness.py [--verbose]` | Is the paper submittable: unmeasured claims, missing figures, runs below the protocol's `min_n`, unverified citations. Exit 0 only when every blocking check passes |
| `audit_claims.py` | Recompute every numeric claim in the paper from the scorecard or run that produced it |
| `status_matrix.py [--date D] [--out -] [--check]` | Generate `docs/status.md` from tier configs, shard specs, `PRESETS`, `ARCHS`, train configs, the registry and `configs/status_notes.yaml`; `--check` exits 1 when the page is stale |
| `figs/*.py` | One module per paper figure (`armnames`, `contamination`, `cutaway`, `descriptor`, `four_views`, `influence_pair`, `interior_counts`, `ladder`, `lego_ldraw`, `pipeline`, `samples`, `rescore_flat_opaque`), the `math_*_check.py` audits of the BlockScore and mixture identities, `figstyle.py` for one palette, and `proposal_figures.py` for the Illustrator-editable SVG set under `deliverables/figures/` |

## Rendering and figures

All of these go through `blockgen.renderer.textured`, so they show real block
textures rather than coloured cubes.

| script | what it does |
|---|---|
| `render_model_samples.py` | Textured sheets from the ideas-battery checkpoints |
| `render_native_samples.py` | Sheets for the §9.0 native-resolution arms (T20) |
| `render_transfer_samples.py` | Sheets from the transfer-run checkpoints |
| `render_cond_samples.py` | A conditioned run's samples, with prompts |
| `render_cond_showcase.py` | Input to output showcase grids (T15) |
| `render_mv_comparison.py` | Input-vs-output sheet for MinecraftACE multi-view samples |
| `render_minecraftace_samples.py` | Decode, render and evaluate MinecraftACE samples |
| `render_decimation.py` | The same real houses at native 32 vs canon-16 (T17) |
| `render_generation_video.py` | Animate AR generation block-by-block to GIF + MP4 |
| `attach_growth_video.py` | Animate growth generation, highlighting the face being decided |

## Probes and diagnostics

The scripts that answer a question rather than produce an artifact. Several of
these returned **negative** results, which is why they still exist; see
`results.md` at the repo root.

| script | question |
|---|---|
| `probe_llm_attention.py` | Does a pretrained LLM's attention carry 3D adjacency? (T28, T49) |
| `plot_attn_probe.py` | Figures for that probe |
| `plot_geobias.py` | Figures for the geometry-bias control-vs-bias pair (T29) |
| `attach_prefix_test.py` | Does the growth model collapse because it cannot *see*, or because it drifts? |
| `pnp_prefix_test.py` | The same question for pick-and-place: can it read a real partial build? |
| `attach_diagnose.py` | Diagnose the filament failure mode |
| `attach_perceptual.py` | Perceptual scoring of the growth arms |
| `attach_resample.py` | Re-sample trained growth arms at a chosen temperature |
| `attach_report.py` | Morning report over growth runs |
| `ordering_bakeoff.py` | Phase-0 gate plus the ordering bake-off |
| `run_semantic_prior.py` | Does a semantic block-embedding prior beat a learned table? (idea #7) |
| `scope_piece_tokens.py` | Token savings of piece serialization vs per-voxel |
| `zeroshot_brickgpt.py` | Zero-shot baseline: prompt a frontier LLM for a build |
| `vlm_price_probe.py` | Price full-corpus VLM captioning from a small live sample |
| `run_agentic.py` | Single-prompt entry point for the agentic track |

## Launch helpers

Shell wrappers for the legacy batteries. Each polls `nvidia-smi` until `FREE_MB`
(default 2500) of VRAM is free, launches under `nohup`, and writes the exact
invocation to `cmd.txt` in the run directory.

| script | what it does |
|---|---|
| `run_overnight.sh [flags]` | `blockgen.experiments_overnight` once the GPU is free (T10) |
| `run_ideas.sh [flags]` | `blockgen.experiments_ideas` (T11); epoch budgets from `EPOCHS_AR` and friends only when set |
| `pause_ideas.sh` | stop the ideas battery outright so its VRAM is released; finished arms are on disk |
| `resume_ideas.sh [stamp]` | replay the same stamp; arms with a `novelty.json` are skipped |

## Module entry points

Library code with a CLI attached:

```bash
python -m blockgen.corpus.manifest|templates|dedup|kinds|tiers|shards   # the corpus layer, above
python -m blockgen.train --config ... | --sample ...                      # the shared trainer
python -m blockgen.posttrain.bon | rft                                    # post-training rungs 1 and 2
python -m blockgen.eval.bench                 # score arms  -> scorecard.json
python -m blockgen.eval.bench.splits --show   # inspect or rebuild the canonical split
python -m blockgen.eval.bench.ladder          # validate the metrics themselves
python -m blockgen.eval.bench.features        # warm the DINOv2 feature cache
python -m blockgen.eval.bench.pointcloud --n 64 --k 1024   # the volumetric row on the ladder (T67)
python -m blockgen.eval.bench.lego            # StableText2Brick as a second domain (T34)
python -m blockgen.agentic                    # the agentic track's recipe harness
python -m blockgen.ontology                   # the block catalog
python -m tools.lab [--open] [--phone]        # BlockLab, and its phone queue
python -m tools.lab.prerender                 # warm the thumbnail cache
python -m tools.lab.rawcorpora --hash         # index the raw corpora
```
