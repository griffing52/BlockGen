# The workflow — train, evaluate, render, view

Four verbs, one paved path through them, and a way out of the path at every step.

```
   train  ──▶  sample  ──▶  evaluate  ──▶  view
   a model     to .npz      a scorecard    in BlockLab
```

This page is the **default route**: the one that produces artifacts every other
tool in the repo already understands. It is not the only route, and most steps
are optional — the sections marked *Other ways in* exist because half the useful
work in this project has come from skipping a stage. What matters is the shape of
what you hand to the next stage, not which script produced it.

If you only read one thing: **give every run a `--name`**, and the rest of the
pipeline stays legible three weeks later.

---

## The short version

```bash
# 1. train something (any track; this is the smallest real one)
python examples/custom_model.py --epochs 3 --samples 6

# 2. turn a checkpoint into an evaluation arm
python scripts/sample_to_npz.py --model native_oriented --n 128 \
    --out-dir outputs/bench_arms --name native_oriented

# 3. score it against controls, baselines and the real held-out set
python -m blockgen.eval.bench --tier both --n 128 \
    --name "BPE oriented, n=128" \
    --arms ar/native_oriented:outputs/bench_arms/native_oriented_32.npz

# 4. look at it
python -m tools.lab --open        # /runs for this run, /leaderboard across runs
```

Step 3 prints the URL for step 4.

---

## 1. Train

Training entry points live in `scripts/train_*.py` and `blockgen/experiments*.py`.
Every one of them writes to a timestamped run directory:

```
outputs/run_<YYYYMMDD_HHMMSS>_<name>/
```

created by `blockgen.utils.runs.new_run_dir(name)`. Use it rather than inventing
a path — the lab's discovery, the hub's provenance tree and every `run_*` glob in
the repo key off that shape.

What a training run should leave behind, at minimum: the checkpoint, the config
it ran with, and **rendered samples**. That last one is a standing rule, and it is
scar tissue: T21 shipped a metric improvement that the renders would have shown
was nonsense. Look at the builds before writing any number down.

**Other ways in.** You do not need a training run at all to use everything
downstream. An arm is just a cache of structures — procedural generators, an LLM
writing build programs (`blockgen/agentic/`), a hand-built set, or another
project's output all enter the pipeline at step 2 on equal terms. See
[Bring your own model](custom-model.md) for the smallest end-to-end example, and
[Models](models.md) for the tracks that already exist.

## 2. Sample to an arm

The benchmark is **generator-blind**: it never imports a model, loads a
checkpoint, or calls a provider. Its only input format is a structure cache — an
`.npz` plus a sibling `_manifest.json`, written by
`blockgen.curation.houses.save_house_cache`.

```bash
# from a registered model
python scripts/sample_to_npz.py --model native_oriented --n 128 \
    --temperature 1.0 --top-k 40 --out-dir outputs/bench_arms --name native_oriented

# from per-sample .npy or a keyed .npz you already have
python scripts/dump_samples.py --keyed-npz outputs/run_x/arm/samples.npz \
    --out-dir outputs/run_x/arm --name arm --max-dim 32 --crop
```

Both write a `report` block into the manifest — model, checkpoint, seed,
temperature, top-k — and the benchmark now **keeps** it, so a leaderboard row can
say which checkpoint produced it and how. Nothing forces you to fill that block,
but an arm without one is a row that cannot explain itself later.

`dump_samples.py` prints the exact `--arms` string to paste into step 3.

**Other ways in.** Any `.npz` + `_manifest.json` pair works, however you made it.
Sampling `n` matters more than it looks: see *Sample sizes* in
[Evaluation suite](benchmark.md), and note that the official protocol wants at
least 128 builds per arm.

## 3. Evaluate

```bash
python -m blockgen.eval.bench --tier both --n 128 --name "what this run is for"
```

The run writes `scorecard.json`, `scorecard.md`, and — unless you pass
`--examples 0` — an `examples_<dim>.npz` holding a few real builds from *every*
arm, controls included, so the lab has something to show for each row.

Three flags carry most of the meaning:

- `--tier fast` is CPU-only and takes seconds; it already ranks on structure.
  `--tier both` adds the rendered `appearance` pillar and needs a GPU.
- `--n` sets how many builds the controls and baselines use. It does **not** cap
  arms loaded from a file — those are however big their `.npz` is.
- `--name` names both the card and the directory. Skip it and you get
  `run_<stamp>_bench`, which is what this whole workflow exists to stop.

Controls and procedural baselines are added automatically, and the run
self-validates: it checks that real data scores ~0, that the memorizing and
mode-collapsed arms are disqualified, and that damage is ordered. **If those
checks fail, the ruler bent and the ranking should not be quoted** — the card
says so and so does the page.

**Other ways in.** `--no-controls` scores arms alone (faster, and unrankable —
BlockScore needs the control to have a unit). `--baselines` adds the procedural
floor. `blockgen.eval.bench.ladder` validates the metrics themselves against
synthetic corruption without scoring any model, and `blockgen.eval.bench.splits`
inspects or rebuilds the canonical split. For scoring one thing quickly, the fast
tier alone is often enough.

## 4. Render and view

Rendering is not a separate stage you run — it happens where it is needed, and
there is one renderer:

- `blockgen.renderer.textured.render_structure` turns a structure into an image
  with real block textures. It holds **one** offscreen EGL context, which is
  thread-affine, so all render work in the lab is drained by a single worker.
  A locking scheme that lets the context migrate silently returns blank tiles.
- Training scripts write their own `samples.png` contact sheets.
- BlockLab renders on demand into a content-addressed cache, keyed on the build's
  voxels — so the same build reached through two datasets renders once, and
  browsing an arm in Curate warms the leaderboard.

```bash
python -m tools.lab --open
```

- **`/runs`** — one experiment in full: identity, provenance, the calibration
  panel, every arm including controls, with example renders per row.
- **`/leaderboard`** — across runs: one row per model, ranked, over every run
  matching a pinned protocol.
- **`/curate`, `/compare`, `/gates`, `/ontology`** — the rest of the instrument.

See [BlockLab](lab.md) for all of it.

**Other ways in.** `python -m tools.lab.prerender` warms thumbnails in batch.
`scripts/render_model_samples.py` and friends produce figures for the paper.
Hero images go through mcrender/Chunky, not through this pipeline.

---

## What makes a run comparable

Scores are calibrated **within a run** — every pillar is expressed in spreads of
the `real_test` control scored in that same run. Two numbers from two runs are
only on one ladder if both ran the same corpus, split, tier, seed and sample
sizes. That set is pinned as a **protocol**
(`blockgen/eval/bench/protocol.py`), and the leaderboard ranks only runs that
match one.

This is not a formality. Scoring one model against a 32-build control instead of
a 128-build one moved its BlockScore from 87.14 to 19.05 — same model, same
corpus, same split, same tier. The ruler changed, not the model.

You are not obliged to run the official protocol. A fast-tier run on a different
corpus is a perfectly good experiment; it simply appears on `/runs` rather than
on the ranked board, and the leaderboard says why it was excluded rather than
dropping it silently.

## Where things go

| stage | writes | who reads it |
|---|---|---|
| train | `outputs/run_<stamp>_<name>/` — checkpoint, config, `samples.png` | you, the hub |
| sample | `<name>_<dim>.npz` + `_manifest.json` | the bench, the lab |
| evaluate | `scorecard.json`, `scorecard.md`, `examples_<dim>.npz` | the lab, `bench_report.py` |
| view | `outputs/lab/` only — never an artifact it did not create | you |

## Reproducing someone else's run

Every card records `rerun`: a copy-pasteable command rebuilt from the arguments
as data, so a path with a space survives. Paste it and you get the same run.

The older `cmd` field is kept for eyeball habit but is **not** re-runnable — its
`argv[0]` is an absolute path to `__main__.py`. Cards also record the git sha,
the branch, whether the tree was dirty, the host, and the split fingerprint.

A dirty tree is flagged rather than hidden: a sha that does not describe the code
that ran is worse than no sha.

## Related

- [Getting started](getting-started.md) — environment and data caches first
- [Evaluation suite](benchmark.md) — what every metric means, and its validation
- [BlockLab](lab.md) — the instrument in full
- [Bring your own model](custom-model.md) — the smallest end-to-end example
- [Experiments & outputs](experiments.md) — what the batteries do
