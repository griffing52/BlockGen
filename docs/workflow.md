# The workflow: train, sample, evaluate, view

Four verbs, one paved path through them, and a way out of the path at every step.

```
   train  ->  sample  ->  evaluate  ->  view
   a run      to .npz     a scorecard   in BlockLab
```

This page is the default route, the one that produces artifacts every other
tool in the repo already understands. Most steps are optional. The sections
marked *Other ways in* exist because half the useful work in this project came
from skipping a stage. What matters is the shape of what you hand to the next
stage, not which script produced it.

If you only read one thing: give every run a name, and the rest of the
pipeline stays legible three weeks later. Use the conda `python` on `PATH`
([Getting started](getting-started.md)).

---

## The short version

```bash
# 1. train (the shared trainer; sft-houses32 in, eval-holdout out, a card at the end)
python -m blockgen.train --config configs/train/houses32-scratch-matched.yaml name=mine_s0

# 2. sample the selected checkpoint into an evaluation arm of board size
python -m blockgen.train --sample outputs/run_<stamp>_mine_s0 --n 399

# 3. score it against controls, baselines and the real held-out set, both tiers
python -m blockgen.eval.bench --tier both --n 399 --baselines \
    --name "mine_s0, n=399" \
    --arms ar/mine_s0:outputs/run_<stamp>_mine_s0/samples_step399_48.npz

# 4. look at it
python -m tools.lab --open        # /runs for this card, /leaderboard across cards
```

Step 3 prints the URL for step 4. [Add a model](add-a-model.md) is this route
with a new architecture plugged in at step 1.

---

## 1. Train

The shared trainer is `python -m blockgen.train --config configs/train/<name>.yaml [key=value ...]`
(`blockgen/train/`, [Training pipeline](pipeline.md) §5). It reads tokenised
shards, runs one step-based loop under a warmup-stable-decay schedule, selects
a checkpoint on holdout NLL (`select=best`), renders its samples, scores them
on `bench_tier` (fast by default) and appends one line to
`outputs/registry.jsonl`. A later stage is a new run with
`init_from=<checkpoint>`; the same run continues with `resume=<run_dir>`.
Every run writes to

```
outputs/run_<YYYYMMDD_HHMMSS>_<name>/
```

created by `blockgen.utils.runs.new_run_dir(name)`. The lab's discovery, the
hub's provenance tree and every `run_*` glob key off that shape. The run dir
holds `run.json`, `metrics.jsonl`, `best.pt`, `last.pt`, `samples.png`, the
end-of-run sample cache, `bench/scorecard.json` and `summary.json`.

Rendered samples are a standing rule, and it is scar tissue: T21 shipped a
metric improvement that the renders would have shown was nonsense. Look at the
builds before writing any number down. The trainer renders at the end of every
run (`samples.png`, and `facades.png` for facade models).

**Other ways in.** The legacy trainers in `blockgen/training/` and the
batteries in `blockgen/experiments*.py` and `scripts/train_*.py` still train
Tracks B, C, F and the conditioned models; they write to the same run-dir
shape. You do not need a training run at all to use everything downstream: a
procedural generator, an LLM writing build programs (`blockgen/agentic/`), a
hand-built set or another project's output all enter at step 2 on equal
terms. [Models](models.md) lists the tracks that already exist.

## 2. Sample to an arm

The benchmark is generator-blind: it never imports a model, loads a checkpoint
or calls a provider. Its only input format is a structure cache, an `.npz` plus
a sibling `_manifest.json`, written by
`blockgen.curation.houses.save_house_cache`.

```bash
# the shared trainer: best.pt if it exists, else last.pt; overrides use the YAML coercion
python -m blockgen.train --sample outputs/run_<stamp>_mine_s0 --n 399 \
    [--ckpt best.pt] [--out-dir D] [temperature=1.0 top_k=40 seed=0]

# a checkpoint served in deploy/inference/models.json (legacy piece-token arms)
PYTHONPATH=. python scripts/sample_to_npz.py --model native_oriented --n 399 \
    --out-dir outputs/bench_arms --name native_oriented --train-split canonical

# per-sample .npy or a keyed .npz you already have
PYTHONPATH=. python scripts/dump_samples.py --keyed-npz outputs/run_x/arm/samples.npz \
    --out-dir outputs/run_x/arm --name arm --max-dim 32 --crop
```

The trainer's end-of-run cache has only `max(n_samples, bench_n)` builds, so
`--sample --n 399` is the step that makes an arm of board size. It writes
`samples_step399_<dim>.npz` under `--out-dir` (default the run dir), named by
its sample count so it never overwrites the trainer's own cache.

Every writer puts a `report` block in the manifest. The board reads one key of
it above all others, `train_split`, because a protocol pins the evaluation and
that turned out not to be enough: an arm also has to have trained on the
protocol's own train split, or the board ranks a model against builds it was
fitted on. `report.train_split` must be `canonical` (or `protocol`) for a row
to rank; anything else, including a missing value, is listed as provisional
with the reason (`blockgen/eval/bench/protocol.py::match_arm`).

| Writer | What it records in `report.train_split` | On the board |
|---|---|---|
| `blockgen.train` (end of run and `--sample`) | `cfg.split_name()`: `canonical` when `train == [sft-houses32]`, else the joined tier list, unless the run set `train_split` itself | ranked when canonical; a mixture of clean tiers must pass `train_split=canonical` |
| `scripts/sample_to_npz.py` | `--train-split`, else the model's `train_split` in `models.json`, else empty | ranked when canonical |
| `scripts/dump_samples.py` | nothing | provisional until the manifest says `canonical` |
| `blockgen/agentic` (`core.py`) | `train_only` or `none`; the suite's oracle arms `oracle_train` | provisional by design |
| `blockgen.posttrain.bon` | the parent run's split, plus `trained_on_signals` | as the parent; the signals are flagged, not unranked |
| `scripts/facade_oracle.py`, `scripts/facade_baselines.py` | nothing (`drawn_from` instead) | provisional; they are the ceiling and the floor |

`dump_samples.py` and the BoN CLI print the exact `--arms` string to paste into
step 3.

**Other ways in.** Any `.npz` + `_manifest.json` pair works, however you made
it. Sample size matters more than it looks: 128 per arm is the protocol floor,
256 or 399 is what gets quoted, and equal n across arms is required for a
comparison ([Evaluation suite](benchmark.md), *Sample sizes*).

## 3. Evaluate

```bash
python -m blockgen.eval.bench --tier both --n 399 --baselines --name "what this run is for" \
    --arms ar/mine_s0:<npz>
```

The run writes `scorecard.json`, `scorecard.md` and, unless you pass
`--examples 0`, an `examples_<dim>.npz` holding a few builds from every arm,
controls included, so the lab has something to show for each row.

Three flags carry most of the meaning:

- `--tier fast` is CPU-only and takes seconds; it ranks on structure
  (coherence, palette, novelty, `geom_kid`, the geometry scalars, traversal).
  `--tier both` adds the rendered `appearance` pillar and needs a GPU; the
  official protocol is `both`.
- `--n` sets how many builds the controls and baselines use. It does not cap
  arms loaded from a file, which are as big as their `.npz`. Keep the two
  equal.
- `--name` names both the card and the directory. Skip it and you get
  `run_<stamp>_bench`, which is what this whole workflow exists to stop.

Controls and procedural baselines (`--baselines`) are added automatically, and
the run self-validates: real data scores about 0, the memorising and
mode-collapsed arms are disqualified, damage is ordered. If those checks fail
the ruler bent and the ranking should not be quoted; the card says so and so
does the page.

A fresh clone builds two gitignored caches before its first `--tier both` run:
the geometry ladder, `python -m blockgen.eval.bench.ladder --family geometry --n 64 --n-ref 256`,
and the DINOv2 feature cache plus render ladder,
`python -m blockgen.eval.bench.features --backbone dinov2b` then
`python -m blockgen.eval.bench.ladder --family render --n 64 --n-ref 256`.

**Promote to the board.** A trainer run's own card (`<run>/bench/scorecard.json`)
is a fast-tier card at `bench_n` builds, a progress number. The board row is
steps 2 and 3 at `--n 399` and `--tier both`, with `--arms ar/<run name>:<npz>`
so the row collapses onto the model's name. The first trainer arm ranked this
way is `m2b_sft_s1` at n=256
(`outputs/run_20261031_031101_bench_voxel_post-training_best-of-4_and`).

**Other ways in.** `--no-controls` scores arms alone (faster, and unrankable:
BlockScore needs the control to have a unit). `blockgen.eval.bench.ladder`
validates the metrics themselves against synthetic corruption without scoring
any model, and `blockgen.eval.bench.splits` inspects or rebuilds the canonical
split. For scoring one thing quickly, the fast tier alone is often enough.

## 4. Render and view

Rendering is not a separate stage you run. It happens where it is needed, and
there is one renderer:

- `blockgen.renderer.textured.render_structure` turns a structure into an image
  with real block textures. It holds one offscreen EGL context, which is
  thread-affine, so all render work in the lab is drained by a single worker.
- The trainer and the legacy batteries write their own `samples.png` sheets.
- BlockLab renders on demand into a content-addressed cache keyed on the
  build's voxels, so the same build reached through two datasets renders once.

```bash
python -m tools.lab --open
```

- `/runs`: one card in full, identity, provenance, the calibration panel, every
  arm including controls, example renders per row. For a trainer card the page
  also shows the training run it came from (`run.json`, `summary.json`, the
  registry line).
- `/leaderboard`: across cards, one row per model, ranked over every card
  matching a pinned protocol. Rows carry their n; chips select one n group and
  `?n=399` bookmarks it; a banner says when a view mixes n.
- `/curate`, `/compare`, `/ontology`, `/phone`: the rest of the
  instrument. `python -m tools.lab --phone` serves the one-thumb queue that
  feeds the human study; the arms it shows are pinned in `configs/lab/phone.json`.

The lab discovers `outputs/*/scorecard.json` and the trainer's nested
`outputs/*/bench/scorecard.json`. A lab started before 2026-10-31 needs a
restart to see the nested cards. See [BlockLab](lab.md) for all of it.

Images are rendered lazily, so the first load of a fresh card renders its
example builds one at a time. To warm them up front, point the prerenderer at
the run directory:

```bash
python -m tools.lab.prerender --runs outputs/run_<stamp>_bench_<name>
```

An arm that failed during scoring contributes no examples and so has no
images; that is a missing arm, fixed by re-running it.

**Other ways in.** `scripts/render_model_samples.py` and friends produce
figures for the paper. Hero images go through mcrender/Chunky, not through
this pipeline.

---

## What makes a run comparable

Scores are calibrated within a run: every pillar is expressed in spreads of
the `real_test` control scored in that same run. Two numbers from two runs are
on one ladder only if both ran the same corpus, split, tier, seed and sample
sizes. That set is pinned as a protocol (`blockgen/eval/bench/protocol.py`,
`houses32-v1`: corpus `houses_32`, the committed 70/15/15 split, tier `both`,
at least 128 builds per arm, seed 0, the `patch2_orb55_t5_h8_s12` descriptor),
and the leaderboard ranks only cards that match one.

This is not a formality. Scoring one model against a 32-build control instead
of a 128-build one moved its BlockScore from 87.14 to 19.05, same model, same
corpus, same split, same tier. The ruler changed, not the model. The same
spread shrinks with n, so an arm at n=128 and one at n=399 are in different
units even under one protocol; the board groups them rather than mixing them.

A protocol pins the evaluation, and the arm's `report.train_split` pins the
training provenance (section 2). The failure this prevents is recorded: a pair
of LLM arms whose training set contained 53 of the 399 canonical test builds
scored on the canonical split at n=128 and were ranked, with `train_split`
sitting unread in the provenance block the page already displayed
(notes.md §32).

You are not obliged to run the official protocol. A fast-tier run on a
different corpus simply appears on `/runs` rather than on the ranked board,
and the leaderboard says why it was excluded rather than dropping it silently.

## Where things go

| stage | writes | who reads it |
|---|---|---|
| train | `outputs/run_<stamp>_<name>/`: checkpoints, `run.json`, `samples.png`, `bench/scorecard.json`, `summary.json`; one line in `outputs/registry.jsonl` | you, the lab's `/runs`, the registry readers |
| sample | `samples_step<n>_<dim>.npz` + `_manifest.json` (or `<name>_<dim>.npz` from the legacy writers) | the bench, the lab |
| evaluate | `scorecard.json`, `scorecard.md`, `examples_<dim>.npz` under `outputs/run_<stamp>_bench_<slug>/` | the lab, `scripts/bench_report.py` |
| view | `outputs/lab/` only; never an artifact it did not create | you |

## Reproducing someone else's run

Every card records `rerun`: a copy-pasteable command rebuilt from the
arguments as data, so a path with a space survives. Paste it and you get the
same run. The older `cmd` field is kept for eyeball habit but is not
re-runnable; its `argv[0]` is an absolute path to `__main__.py`. Cards also
record the git sha, the branch, whether the tree was dirty, the host and the
split fingerprint. A dirty tree is flagged rather than hidden: a sha that does
not describe the code that ran is worse than no sha. A training run records
the same in `run.json` (`meta.git`, the tokenizer hash and the tier hashes).

## Related

- [Getting started](getting-started.md): environment and data first
- [Add a model](add-a-model.md): this route with a new architecture, every step spelled out
- [Bring your own model](custom-model.md): the model contract and the five commands
- [Training pipeline](pipeline.md): the corpus layer and the trainer in full
- [Evaluation suite](benchmark.md): what every metric means, and its validation
- [BlockLab](lab.md): the instrument in full
- [Experiments & outputs](experiments.md): what the batteries do
