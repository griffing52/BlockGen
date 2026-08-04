# Evaluation suite

`blockgen/eval/bench` scores any generation track against one fixed real
reference, so arms from different tracks and different voxel resolutions land in
the same table.

## Quick start

```bash
# 1. Inspect the canonical split (built and cached on first use)
python -m blockgen.eval.bench.splits --corpus houses_32 --seed 0 --show

# 2. FAST tier: no GPU, no rendering, seconds
python -m blockgen.eval.bench --n 128

# 3. Warm the feature cache (~4 min, 25 MB), then validate the metrics (~4 min)
python -m blockgen.eval.bench.features --backbone dinov2b
python -m blockgen.eval.bench.ladder --n 64 --n-ref 256

# 4. FULL tier
python -m blockgen.eval.bench --tier both --n 256 \
    --arms agentic/oneshot:outputs/run_.../oneshot_48.npz
```

Neural arms that dump per-sample `.npy` need one conversion step first:

```bash
python scripts/dump_samples.py --keyed-npz outputs/run_x/arm/samples.npz \
    --out-dir outputs/run_x/arm --name arm --max-dim 32 --crop
```

## What it reports

| pillar | leads | notes |
|---|---|---|
| Realism | `mv_dino_kid` | unbiased polynomial-kernel MMD on pooled DINOv2 features |
| | `mv_dino_mmd_rbf` | second opinion, RBF kernel, bandwidth frozen on the reference |
| Novelty | `dino_nn_percentile` | resolution-independent; low = memorized |
| | `voxel_nn_iou_mean`, `*_dup_rate` | delegated to `eval/novelty.py`, unchanged |
| Coherence | `lcc_ratio`, `enclosed_air_ratio`, `floating_block_frac` | reported as **distance to real** |
| Dataset stats | `palette_jsd_{exact,family}`, `palette_cooccur_jsd` | plus block-count / bbox W1 |
| Faithfulness | `retrieval_acc` (chance 1/32) | `mv_clipscore` and material agreement alongside |
| Cost | `usd_per_coherent_build` | agentic only; never combined with GPU-hours |

## Rules the code enforces

**Every metric carries an interval and a direction.** `scorecard.Metric` cannot
represent a bare float, so a coherence rate can never be printed without the real
value beside it. That matters because only ~66% of *real* val houses are a single
connected component — an arm at 1.0 is as far from the data as one at 0.2.

**No metric reports before it passes the ladder.** `ladder.py` scores each metric
on rungs of known damage (decimation, block noise, interior deletion, material
shuffle) and on invariances that must not move it (rotation, mirroring). Validity
failures are blocking and surface as `null` naming the failed gate; sensitivity
limits ("cannot resolve a 40% cut at n=64") are recorded but never suppress a
value, because those are answered by more samples.

**Real data is scored as an arm.** `real_test` is the floor, `real@canon16` and
`real@canon8` are known-damage rungs, `train_verbatim` proves the memorization
detector fires. They run through the identical code path as any model.

**Novelty sits on the same row as realism.** `train_verbatim` scores MV-DINO-KID
0.015 — second only to real data — and is caught only by `dino_nn_percentile`.

## Sample sizes

KID's noise floor is 0.005 at n=64 and 0.04 at n=16, against a real→canon-16 gap
of 0.18. Use **n ≥ 256** for headline numbers; n = 64 is the reportable floor;
n = 16 (the current `--samples` default) cannot rank arms.

## Caches

* `data/minecraft/splits/` — the canonical split, tracked in git.
* `data/minecraft/features/<corpus>/<split>/<view>/<backbone>.npz` — not tracked;
  rebuilt in minutes. Invalidates on structure-set change, view change, backbone
  change, or a `render_canary_sha` mismatch (a changed texture pack would
  otherwise poison every cached number invisibly).
* `outputs/analysis/ladder/<backbone>__<view>.json` — validation verdicts.

## Relationship to the older metrics

`eval/novelty.py`, `eval/validity.py` and `eval/perceptual.py` are **not
modified**, so every number in results.md T1–T22 stays reproducible. `bench`
reuses `evaluate_novelty` (pinned bit-for-bit by a test) and scores
`perceptual.cmmd` in the ladder as a labelled legacy metric — where it fails
n-stability, because it is the biased MMD estimator despite its docstring. See
results.md T23.
