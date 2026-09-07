# Scripts

Everything in `scripts/` is a runnable entry point with a `--help`. Library code
lives in `blockgen/`; these are the things you *run*. They fall into five groups,
and only the first is part of the [standard workflow](workflow.md) — the rest are
the one-off instruments that produced particular results, kept because a probe
worth running once is usually worth running again.

Every script takes `--help`, and most write into `outputs/run_<stamp>_<name>/`
via `blockgen.utils.runs.new_run_dir`.

## Pipeline — the paved path

| script | what it does |
|---|---|
| `sample_to_npz.py` | Sample a served checkpoint into the standard structure-cache `.npz` |
| `dump_samples.py` | Convert generated samples (`.npy` dir, keyed `.npz`, existing cache) into that same format |

These two are the seam between "a model" and "an evaluation arm": both write the
`.npz` + `_manifest.json` pair the benchmark and BlockLab understand, and both
record a provenance `report` (model, checkpoint, seed, temperature, top-k) that
survives into the leaderboard row.

## Training

| script | what it does |
|---|---|
| `train_uncond_corpus.py` | Unconditioned `native_bpe` on the full corpus — the diagnostic run |
| `train_cond_resampler.py` | Text-conditioned piece AR with a resampled conditioning channel (idea #6) |
| `train_pick_n_place.py` | The pick-and-place growth model; trains, samples, renders and dumps for the bench |
| `train_attach_corpus.py` | Attachment/growth AR, one arm per ordering |
| `train_llm_pieces.py` | BrickGPT-style text→build over 3D-BPE *pieces* rather than voxels |
| `train_llm_brickgpt.py` | BrickGPT/LegoGPT-style text→build with a LoRA-finetuned LLM |
| `train_llm_geobias.py` | Track D plus a structural attention bias — inject adjacency instead of extracting it |

## Evaluation, reporting and the human study

| script | what it does |
|---|---|
| `bench_report.py` | A compact cross-arm table from one or more scorecards |
| `bench_doseresponse.py` | Does the benchmark reward *more realistic* output, or only detect artifacts? |
| `human_study_export.py` | Export a 2AFC study — stimuli plus a self-contained page |
| `human_study_analyze.py` | Exclusion → Bradley-Terry → metric-human agreement |
| `validate_perceptual.py` | Is the perceptual metric trustworthy? Tested where the answer is known |
| `score_arms_perceptual.py` | Adjudicate the T20 arms with the perceptual metric |
| `eval_mv_conditioning.py` | Does multi-view conditioning carry structure-*specific* information? |
| `sample_conditioned.py` | Sample a conditioned run and evaluate fidelity |

`human_study_export.py` strips arm labels from the page and asserts it by grep: a
condition visible in the DOM is a condition the rater can read.

## Rendering and figures

All of these go through `blockgen.renderer.textured`, so they show real block
textures rather than coloured cubes.

| script | what it does |
|---|---|
| `render_model_samples.py` | Textured sheets from the ideas-battery checkpoints |
| `render_native_samples.py` | Sheets for the §9.0 native-resolution arms (T20) |
| `render_transfer_samples.py` | Sheets from the transfer-run checkpoints |
| `render_cond_samples.py` | A conditioned run's samples, with prompts |
| `render_cond_showcase.py` | Input → output showcase grids (T15) |
| `render_mv_comparison.py` | Input-vs-output sheet for MinecraftACE multi-view samples |
| `render_minecraftace_samples.py` | Decode, render and evaluate MinecraftACE samples |
| `render_decimation.py` | The same real houses at native 32 vs canon-16 (T17) |
| `render_generation_video.py` | Animate AR generation block-by-block to GIF + MP4 |
| `attach_growth_video.py` | Animate growth generation, highlighting the face being decided |

## Probes and diagnostics

The scripts that answer a question rather than produce an artifact. Several of
these returned **negative** results, which is why they still exist — see
[Results](results.md).

| script | question |
|---|---|
| `probe_llm_attention.py` | Does a pretrained LLM's attention carry 3D adjacency? |
| `plot_attn_probe.py` | Figures for that probe |
| `plot_geobias.py` | Figures for the geometry-bias control-vs-bias pair |
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

## Module entry points

Some tools are `python -m` modules rather than scripts, because they are library
code with a CLI attached:

```bash
python -m blockgen.eval.bench                 # score arms  -> scorecard.json
python -m blockgen.eval.bench.splits --show   # inspect or rebuild the canonical split
python -m blockgen.eval.bench.ladder          # validate the metrics themselves
python -m blockgen.eval.bench.features        # warm the DINOv2 feature cache
python -m blockgen.ontology                   # the block catalog
python -m tools.lab                           # BlockLab
python -m tools.lab.prerender                 # warm the thumbnail cache
python -m tools.lab.rawcorpora --hash         # index the raw corpora
```
