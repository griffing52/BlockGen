# Evaluation suite

`blockgen/eval/bench` scores any generation track against one fixed real
reference, so arms from different tracks and different voxel resolutions land in
the same table.

## Quick start

```bash
# 1. Inspect the canonical split (built and cached on first use)
python -m blockgen.eval.bench.splits --corpus houses_32 --seed 0 --show

# 2. Validate the structural metrics (CPU only, ~1 min)
python -m blockgen.eval.bench.ladder --family geometry --n 64 --n-ref 256

# 3. FAST tier: no GPU, no rendering, seconds. Already ranks on structure.
#    --name is what makes the run directory (and the leaderboard row) legible.
python -m blockgen.eval.bench --n 128 --name "first fast run"

# 4. Warm the feature cache (~4 min, 30 MB), then validate the render metrics
python -m blockgen.eval.bench.features --backbone dinov2b
python -m blockgen.eval.bench.ladder --family render --n 64 --n-ref 256

# 5. FULL tier
python -m blockgen.eval.bench --tier both --n 256 \
    --name "oneshot vs controls" --note "n=256, agentic arm from run_..." \
    --arms agentic/oneshot:outputs/run_.../oneshot_48.npz

# 6. Read it: the runner prints this link, and the lab serves it
python -m tools.lab --open      # http://127.0.0.1:8765/leaderboard?run=<dir>
```

Steps 2–3 need no GPU, no renderer and no network, so the structural half of the
benchmark runs in CI and on a reviewer's laptop. That is not a convenience: the
render half is the half that can break silently, and it did, see *A renderer
that failed open* below.

Neural arms that dump per-sample `.npy` need one conversion step first:

```bash
python scripts/dump_samples.py --keyed-npz outputs/run_x/arm/samples.npz \
    --out-dir outputs/run_x/arm --name arm --max-dim 32 --crop
```

## What it reports

| block | leads | notes | BlockScore role |
|---|---|---|---|
| Realism (appearance) | `mv_dino_kid` | unbiased polynomial-kernel MMD on pooled DINOv2 features; FULL tier only | pillar `appearance` |
| | `mv_dino_mmd_rbf` | second opinion, RBF kernel, bandwidth frozen on the reference | none |
| **Realism (structure)** | **`geom_kid`** | **the same estimator on D4-invariant voxel descriptors; no GPU** | pillar `geometry` |
| Novelty | `dino_nn_percentile` | resolution-independent; low = memorized | gate: DQ when more than 3.0 real spreads **below** real (smallest spread counted 0.05) |
| | `voxel_dup_rate` | delegated to `eval/novelty.py`, unchanged | gate: DQ when more than 3.0 spreads **above** real (smallest spread 0.02) |
| | `voxel_diversity` | mean pairwise IoU within the arm | gate: DQ when more than 10.0 spreads **below** real (smallest spread 0.02) |
| | `voxel_nn_iou_mean`, `dino_dup_rate` | reported, not gated | none |
| Coherence | `lcc_ratio`, `enclosed_air_ratio`, `floating_block_frac` | reported as **distance to real** (`w1_norm`) | pillar `coherence` (RMS over the block) |
| **Structure** (`geometry_scalars`) | **`thickness_*`, `wall_frac`, `yaw_symmetry`, `interior_ratio_open`**, and since 2026-10-23 `walk_standing`, `walk_floors`, `walk_headroom3` | **also distance to real** | pillar `geometry_scalars` (RMS) |
| **Traversal** | `interior_standing_reached`, `enterable`, `floors_reached`, `headroom3` | the player walk (`blockgen/eval/traversal.py`), distance to real; FAST tier; cards from 2026-10-23 | pillar `traversal` (RMS) |
| Dataset stats | `palette_jsd_{exact,family}`, `palette_cooccur_jsd` | plus block-count / bbox W1 | pillar `palette` = `palette_jsd_exact` |
| Faithfulness | `retrieval_acc` (chance 1/32) | `mv_clipscore` and material agreement alongside; needs `ArmSpec.prompts`, which `parse_arm` never sets (`runner.py:70-76`), so the CLI cannot reach it | none |
| Cost | `usd_per_coherent_build` | agentic only; never combined with GPU-hours | none |
| Volumetric (ladder only) | `cd_volume`, `cd_surface`, `iou`, each as `mmd`, `one_minus_cov`, `one_nna` | `blockgen/eval/bench/pointcloud.py` runs the point-cloud suite on the ladder's rungs; written to `outputs/analysis/ladder/pointcloud__k1024_canvas32.json`, never to a scorecard | none |
| **Aggregate** | **`BlockScore`** | **worst pillar in real-sample spreads, with cheat gates** | the max over the six pillars |

The pillar map is `composite.PILLARS` and `composite.STRUCTURE_BLOCKS`
(`blockgen/eval/bench/composite.py:53-68`); the three gates with their
thresholds are `composite.GATES` (`composite.py:82-98`). A gate is a z-score
against the `real_test` control, not a fixed cutoff, and the smallest spread
column is the floor that keeps a zero-width control interval from disabling
the gate.

## Run identity, provenance and examples

A `bench/1` card recorded five things about the run that produced it: `dir`,
`git_sha`, `cmd`, `tier`, `elapsed_s`. Eleven of them are on disk here, every one
of them in a directory called `run_<stamp>_bench`, and the only way to tell them
apart is to open them. Worse, on the largest card thirteen of sixteen rows said
nothing about themselves except `source: "in-memory"`, the arms were built in
process, scored, and forgotten. `bench/2` is the fix, and it is purely additive:
identity, per-arm provenance, and a few real builds from every arm.

### Naming a run

```bash
python -m blockgen.eval.bench --tier fast --n 128 \
    --name "ontology arms v2" \
    --note "mined vs shuffled ontology, seed 0" \
    --examples 8
```

```
outputs/run_20260906_190029_bench_ontology_arms_v2/
    scorecard.json               # identity + context + every arm's metrics
    scorecard.md                 # the same run, rendered
    examples_48.npz              # k builds from EVERY arm, controls included
    examples_48_manifest.json
```

`--name` does two jobs with one flag: it goes onto the card verbatim (spaces and
all) and it names the directory, through `runner.run_slug`, non-alphanumerics to
`_`, truncated to 40 chars **on a word boundary**, because a naive `[:40]` turns
"ontology arms vs agentic ontology probe" into `..._bench_ontology_arms_vs_agentic_ontol`,
which reads as a typo rather than as an ellipsis. The `bench_` prefix is kept in
front of the slug so every `outputs/run_*_bench*` glob written in anyone's notes
keeps matching. `--note` is one line of display-only context. `--examples 8` is
the default; `--examples 0` disables the sibling npz. A run with no `--name`
still works and prints a one-line reminder at the end, because that is the flag
whose absence you only regret three weeks later.

**The name is never an id.** It is free text, it is not unique, four cards on
disk derive the identical label `native_oriented vs agentic_oneshot vs pick_n_place`,
and nothing may key on it: not a path segment, not a URL parameter, not a dict
key, not an `<option value>`. The id doing those jobs is the directory name, and
it stays the directory name, so `?run=` bookmarks keep resolving.

### The `run` block

`scorecard.RUN_KEYS` declares it. The runner builds `card.run` *from* that tuple
and raises if it holds a key the tuple does not, so the contract and the writer
are one edit rather than two that drift; the golden fixture in
`tests/fixtures/scorecards/` is pinned against the same constant.

| key | type | written from | what it is for |
|---|---|---|---|
| `dir` | str | `--out` or `utils.runs.new_run_dir` | **not identity**, informational only |
| `name` | str | `--name`, verbatim | the run in words; display only |
| `note` | str | `--note` | one line of context |
| `git_sha` | str\|null | `scorecard.git_info()` | short sha |
| `git_branch` | str\|null | `git rev-parse --abbrev-ref HEAD` | branch |
| `git_dirty` | bool\|null | `git status --porcelain` non-empty | `null` means *could not ask*, not *clean* |
| `started_at` | str | `iso_utc(t0)`, beside `t0` | so a run that crashes late still carries it |
| `finished_at` | str | `iso_utc(finished)` | written beside `elapsed_s` |
| `cmd` | str | `" ".join(sys.argv)` | legacy; kept for eyeball habit |
| `rerun` | str | `shlex.join` over `argv` | the copy-pasteable command |
| `argv` | list[str] | `sys.argv[1:]`, as data | so an argument with a space survives |
| `tier` | str | `--tier` | `fast` \| `full` \| `both` |
| `host` | str | `socket.gethostname()` | which machine |
| `device` | str | `--device`, as **requested** | `cuda` here with no GPU is a fact about the invocation |
| `elapsed_s` | float | wall clock | |

Four more keys are present *by construction*, never by version:
`blockscore` and `blockscore_validation` when controls ran and `real_test`
scored, `head_to_head` when more than one arm was scored without
`--no-head-to-head`, and `examples` when at least one arm's builds were written.
`RUN_KEYS_OPTIONAL` names them so a reader can tell "absent" from "forgotten".

**`rerun` is copy-pasteable and `cmd` is not.** `cmd` is `sys.argv` joined, so
its `argv[0]` is an absolute path to the module file, on the smoke run,
`/home/.../blockgen/eval/bench/__main__.py --tier fast --n 24 ... --name smoke run`,
which is not a command, and whose unquoted `smoke run` would parse as two
arguments if it were. `rerun` is built by `shlex.join` from `argv` kept as data:
`python -m blockgen.eval.bench --tier fast --n 24 --min-n 8 --no-head-to-head --name 'smoke run' ...`.
That claim is testable and is tested: `build_parser()` was factored out of
`main()` for no other reason than to let a test re-parse the string a run wrote
and assert it round-trips.

**`run.dir` is not identity.** Readers use `path.parent.name`. The two oldest
cards carry `"dir": "/home/.../scratchpad/bench_compare"` and
`".../scratchpad/bench_pnp"`, absolute paths outside `outputs/` entirely, from
runs written with `--out`, which is exactly why the directory name, and only
the directory name, is the run id. `dir` is kept because things read it, and
because it is the one record of an `--out` that pointed somewhere else.

### Where an arm came from

Each arm's `meta` gains five keys, and none of the existing ones move.
`meta.source` in particular stays byte-identical (`arm.npz or "in-memory"`),
because the join from an arm to a browsable dataset is exact string equality on
it (`catalog.source_dataset_id`), and so are the leaderboard's "open in Curate"
link and the drill panel's *source npz* row.

| key | what it answers |
|---|---|
| `kind` | `submission` \| `control` \| `baseline`, derived from `track` at write time by `ArmSpec.kind`, the single definition of the rule, so a card read on its own groups its rows by reading one field instead of re-deriving `track not in {control, baseline}` in three places across two languages |
| `origin` | `npz` \| `in_memory`, "nothing was ever written" and "the dataset join failed" used to render as the same sentence |
| `source_run_id` | the first `run_*` segment of the checkpoint path (else of `source`), or `null`, the edge back to the run that produced this arm's inputs |
| `structures_sha` | `features.structures_sha` over the **scored** structures: a pure hash of block ids, no renderer and no GPU, and the only cross-run arm identity that will ever exist |
| `provenance` | a dict, never `null`, whose `writer` says what is knowable |

For an npz-backed arm, `ArmSpec.provenance()` reports the two paths, the
manifest's `count`/`max_dim`, its mtime as `written_at`, the manifest's `report`
**verbatim**, and a promoted allowlist lifted out of that report:

```python
PROMOTED = ("model", "checkpoint", "seed", "temperature", "top_k",
            "arm", "epochs", "source")
```

The allowlist is closed on purpose, and it is closed over four report shapes that
share no key but `report` itself:

| writer | keys in its `report` | promoted |
|---|---|---|
| `scripts/sample_to_npz.py:146-149` | model, checkpoint, n, n_empty, seed, temperature, top_k, train_split | model, checkpoint, seed, temperature, top_k |
| `scripts/dump_samples.py:120` | source, n_source, n_written, n_empty, seed, cropped | source, seed |
| `blockgen/agentic/report.py:108` | arm, n_requested, n_nonempty | arm |
| the pick-and-place manifest (`outputs/bench_arms/pick_n_place_32_manifest.json`) | model, epochs | model, epochs |
| `blockgen/train/sample.py:346-352` (`python -m blockgen.train --sample`) | run, step, n, n_empty, seed, temperature, top_k, tags, model, arch, kind, checkpoint `{file, step}`, trained_on_signals, train_split | model, checkpoint, seed, temperature, top_k |
| `blockgen/posttrain/bon.py:65-75` | run, name, checkpoint (path), train_split, model (dict), tokenizer, kind, seed, temperature, top_k, tags, parent_signals, plus the BoN fields (method, k, n, reward, trained_on_signals) | model, checkpoint, seed, temperature, top_k |

`train_split`, `trained_on_signals` and `checkpoint.step` are not promoted; the
lab reads them one level down, under `provenance.report`
(`tools/lab/standings.py:115-150`).

Promotion is a convenience for whoever renders the page; `report` is the record.
A key nobody promotes is not lost, it is simply one level deeper.

`provenance()` is deliberately **self-sufficient**: if `load()` has not run it
reads the `_manifest.json` sibling itself. The first version depended on
`score_fast` calling `load()` before it built the meta block, an invariant
nothing in the type stated, and one that `score_full` and the examples writer
each break by calling `load()` on their own. It never raises: a missing manifest
is a fact about the arm, not a reason to lose a two-hour scoring run.

### The control recipes live in one place

An in-memory arm has no manifest to read, so `fast.control_arms` and the baseline
arms attach a `provenance_override` naming their builder, the split they were
drawn from, `n_requested`, `seed`, and a one-sentence **recipe**:

```
real@solidify: every SEALED air cell of a held-out test build filled with that
build's own dominant material; a median 33% more blocks over 94% of builds, the
silhouette untouched, the control that tells the two realism tiers apart, since
at n=128 MV-DINO-KID cannot separate it from real and geom_kid puts it 202
spreads away. Note "sealed": air reachable from outside through a doorway
survives, so this is a 33% mass increase the render tier cannot see, NOT the
deletion of every room.
```

That prose already existed. It sat in the `control_arms` docstring, where no
reader of a scorecard ever saw it. It now lives in `fast.CONTROL_RECIPES` (one
entry per rung) and `baselines.RECIPES` (one per baseline), **and the docstring
points at the dict instead of restating it**, so there is exactly one copy to
keep true. The lab imports both dicts as a read-time fallback, which is what
gives the eleven legacy cards a recipe on every control row they never recorded.

### Example builds are identities, not images

Every arm now leaves `k` of its builds behind, in one `examples_<max_dim>.npz`
plus its manifest, in the run directory. The obvious way to do this is to render
a contact sheet next to `scorecard.json`, and it does not work. The chain of
reasons is worth writing down, because each link is load-bearing:

* **No lab route serves a file out of a run directory.** `api._static` resolves
  only under `tools/lab/static/`, with a traversal prefix check. A PNG dropped in
  a run directory is a file nothing can fetch.
* **The only image routes are build-id keyed**: `/api/thumb/<build_id>` and
  `/api/view/<build_id>/<k>`. Adding `/api/runfile/<run>/<name>` would open a
  traversal surface over a directory full of checkpoints and bypass the
  content-addressed render cache, the blank-frame guard and the single-thread
  discipline, to buy a thumbnail the render pipeline already produces. So an
  example has to be expressible as a **build id**, not as a path.
* **An npz + manifest pair is already a dataset.** The lab admits a dataset iff
  `<x>.npz` and `<x>_manifest.json` both exist, and it already rglobs
  `outputs/run_*/**/*.npz`. Writing that pair, with
  `curation.houses.save_house_cache`, so there is no new IO code, makes the
  run's examples browsable, thumbnailable and openable in Curate with **zero**
  discovery changes. The eval side records a path; the id is derived in the lab
  alone, so the two cannot drift.
* **The eval renders nothing.** One thread-affine EGL context lives in the lab
  and is drained by a single worker. Keeping the renderer out of
  `blockgen/eval/bench` is also what keeps the fast tier runnable on a reviewer's
  laptop.

Cost, measured: the smoke run's pair is 51 KB for 64 rows (8 arms × k=8) at
28³; at the 1.2 KB/build of a real 32³ arm cache
(`bench_arms/native_oriented_32.npz` is 77 KB for 64 builds), a sixteen-arm run
is **≈200 KB**. That is the whole price of every row on the board having pictures.

Selection is a seeded permutation (`ctx.rng(9)`, salt 9 being unused elsewhere),
sorted ascending so the strip reads in source order. It is uniform, and the page
says so: **a seeded sample, explicitly not a best-of.** A stratified policy,
by worst-pillar contribution, say, is a body change to one function, and is not
in this increment.

Rows are written as **copies** with metadata set explicitly
(`dataclasses.replace`, never in place). Two independent reasons, both measured:
probe and baseline builders construct structures with no metadata at all, so
without this most of a card's rows would be blank-captioned thumbnails; and
`real@single_mode` is one `Structure` repeated *n* times, an object that is
simultaneously row 0 of `real_test` and a member of the shared `test` split the
lab serves, an in-place edit would corrupt three datasets at once for the life
of the process.

Warm the thumbnails before a demo, rather than watching them trickle in:

```bash
python -m tools.lab.prerender --datasets arm:<run>__examples_48 --px 96 --workers 8
```

Two things are deliberately absent. There is no `examples.png`, the paper path
is `tools/lab/prerender.py` + Curate + mcrender/Chunky, and a contact sheet the
UI cannot fetch is residue. And if `--out` points outside `outputs/run_*/`, the
examples are still written (the file is the artifact) but nothing discovers them;
the runner says so in a warning rather than recording build ids that resolve to
nothing.

### Schema versions

`SCHEMA_VERSION` is `"bench/2"`, and **nothing branches on the string.** It is
stamped on the card and read back for exactly one purpose: the lab copies it into
`compat.version` and shows an amber "written by a newer BlockGen" banner when its
major exceeds what the reader understands, while still rendering whatever parses.
Every other decision a reader makes is on the *presence* of a section or a key,
and `tools/lab/cards.migrate` is the one place that dispatch happens, called
inside the scorecard read, before the memo insert, returning a new top-level dict
that shares `arms` by identity so a normalisation can never write into the cache.

The reason is on disk. `bench/1` is not one shape; it is four:

| shape | cards | what distinguishes it |
|---|---|---|
| **A** | `run_20260803_033522_bench_compare`, `run_20260816_081955_bench_pnp` | no `blockscore`, has `fidelity`, no `geometry_scalars`, absolute `run.dir` |
| **B** | `run_20260831_195140/195352/195452_bench` | `blockscore` + `geometry_scalars`, all-control, no `head_to_head` |
| **C** | `run_20260831_195547/202936/204112/225029_bench` | `blockscore` + `fidelity` + `geometry_scalars`, some with `head_to_head` |
| **D** | `run_20260906_185828/190029_bench` | `cost`, no `fidelity` |

A migration keyed on the version *string* would have been wrong about eight of
those eleven cards before it ran. `bench/2` is therefore `bench/1` **plus keys**
(and cards written from 2026-10-23 carry one more block, `traversal`, which older
cards name as missing and render without):
nothing removed, renamed, or re-typed, `Metric.to_json()` untouched, `meta.source`
byte-identical. That rule is enforced by what would break: `scripts/bench_report.py`
reads `card["arms"]` and `entry["gen"]["mean"]` bare and is the one consumer that
hard-crashes rather than degrading.

Nothing on disk is ever rewritten: no backfill, no renamed directory. A legacy
card is read, its gaps are *named* on the page, and every table that parses is
drawn. Hand-adding a `run.name` to an old `scorecard.json` is picked up on the
next page load, that is the documented escape hatch, and it is deliberately not
automated.

**Reserved, not implemented.** A training run needs identity too and will never
emit a scorecard, so the vocabulary a future universal run manifest would use,
`kind`, `artifacts`, `origin`, `status`, `created_at`, is reserved here rather
than invented twice. When that file exists, lifting the `run` block out should be
a rename, not a redesign.

## Two realism tiers, because one of them is blind

Every realism number in this suite used to be computed on *renders*. That is what
makes a 16³ arm and a 32³ arm comparable at all, and it is why the render tier
exists, but it buys comparability by discarding everything a camera outside the
build cannot reach. The cost is measurable and it is not small.

`probes.solidify` fills every enclosed air cell: a median **31%** more blocks,
every room in the corpus deleted, silhouette untouched. Scored at n=128 against
the same 399 held-out real builds:

| arm | `mv_dino_kid` ↓ | `geom_kid` ↓ |
|---|---|---|
| `real_test` (floor) | −0.001 [−0.004, 0.003] | −0.09 [−0.22, 0.04] |
| **`real@solidify`** | **0.000 [−0.002, 0.004]** | **22.1 [20.7, 23.7]** |
| `real_shuffled_materials` | 0.673 [0.642, 0.710] | *−0.09, bit-identical to real* |
| `real@monochrome` | 0.930 [0.903, 0.962] | *−0.09, bit-identical to real* |

The intervals in the first two rows **overlap**. A generator that emits solid
blobs shaped like houses is, to the render-space realism metric, indistinguishable
from one that emits houses. That is not hypothetical: T23d measured
`native_oriented` at an enclosed-air ratio of 0.005 against a real 0.114 while it
scored respectably on KID.

**And it is blind in the other direction too.** `probes.hollow` is `solidify`
run backwards: it deletes every block a camera cannot see, keeping only the
surface layer (a block survives iff one of its six faces borders air that is
6-connected to the outside). The exterior region, and therefore every rendered
face, is *exactly* invariant, checked on all 128 builds. That is the failure
mode the wider 3D field ships: a shell with nothing inside it.

| arm | `mv_dino_kid` ↓ | `geom_kid` ↓ | blocks |
|---|---|---|---|
| `real_test` (floor) | +0.004 [−0.001, +0.012] | +0.13 [−0.07, +0.42] | 870 |
| **`real@hollow`** | **+0.006 [−0.000, +0.012]** | **+13.3 [12.6, 14.1]** | **−13.4% (98.4% of builds)** |
| `real@solidify` | +0.007 [+0.002, +0.014] | +24.4 [23.1, 26.0] | +34.7% (96.1%) |

Hollowing moves the render metric **+0.4 noise floors** and the structural metric
**+156**. It is not that the pixels are unchanged, hollowing moves 0.585/255 of
the image against a re-render floor of 0.0000, because interiors are visible
through doors and windows, the encoder discards it. Measured by
`scripts/bench_hollow.py --n 128`
(`outputs/run_20260918_203947_bench_hollow`).

The last two rows of the first table are the converse, and they are why this is a
second tier rather than a replacement. `geom_kid` reads occupancy only, so a material permutation
moves it by *exactly zero*, the two tiers are complementary halves, and the
ladder gates each on the rungs it is built to see (`GEOMETRY_RUNGS` swaps the
material-noise triple for an occupancy-noise one and puts material corruptions in
its *invariance* set).

### The geometric descriptor

80 dimensions per build, all **exactly** invariant to the D4 yaw group (four
rotations about the vertical axis, and the mirror) by construction rather than by
luck, local patterns are canonicalised onto orbit representatives and every
scalar commutes with the group. Verified bit-for-bit in
`tests/eval/test_geometry.py`.

* **55** D4-canonical 2×2×2 occupancy-pattern frequencies (256 raw patterns → 55
  orbits; the all-empty orbit is dropped)
* **5** taxicab wall-thickness bins
* **8** normalised height-profile bins
* **12** scalars: thickness mean/p90, surface-to-volume, wall fraction, yaw
  symmetry, sealed and aperture-aware interior, height entropy, and since
  2026-10-23 the player's room (the share of interior air a two-block body can
  stand in, the number of storeys with a floor, the share of standing cells with
  a third free block above; see "Traversability" below), plus `log1p(blocks)`

The room scalars read occupancy only, so the material-shuffle column below stays
exactly zero; whether a body can *get in* depends on which blocks are doors and is
the separate `traversal` block. The geometry ladder was re-validated on the
80-dimensional descriptor (`outputs/analysis/ladder/geom__patch2_orb55_t5_h8_s12.json`,
every gate passing, invariance rungs bit-identical to real). The ablation table
below was measured on the 77-dimensional descriptor before the room scalars.

Which part does the work, in noise-floor sd at n=64:

| descriptor | dim | canon16 | solidify | occ_noise_10 | jitter_cols | mat-shuffle |
|---|---|---|---|---|---|---|
| full | 77 | 51.2 | 92.0 | 125.4 | 147.6 | **0.0** |
| − pattern block | 22 | 74.8 | 103.8 | 58.7 | 93.0 | **0.0** |
| − scalars | 68 | 31.4 | 76.7 | 116.2 | 136.2 | **0.0** |
| pattern block only | 55 | 30.6 | 51.4 | 107.9 | 123.8 | **0.0** |
| − block count | 76 | 42.0 | 89.8 | 122.7 | 144.2 | **0.0** |

No component explains the metric on its own, and dropping the block count changes
almost nothing (92.0 → 89.8 on `solidify`), so this is not a size detector
wearing a costume. The material-shuffle column is exactly zero in every variant,
which is the invariance claim restated as an ablation.

### The volumetric row: what the standard 3D suite sees

The objection to a handcrafted descriptor is that shape generation already has
metrics that read volume: MMD, coverage and 1-NN accuracy under chamfer distance
on sampled points. `blockgen/eval/bench/pointcloud.py` runs exactly that suite,
plus the same three statistics under 1 - IoU of the occupancy grids, through
`ladder.score_rungs`, so it meets the same rungs, gates and noise floor as every
other metric:

```bash
python -m blockgen.eval.bench.pointcloud --n 64 --k 1024
```

Points are the centres of all occupied voxels (`cd_volume`) or of voxels with an
exposed face (`cd_surface`), centred and divided by the canvas side with no
per-shape normalisation. Measured at n=64 (results T67,
`outputs/analysis/ladder/pointcloud__k1024_canvas32.json`): every one of the
nine statistics places Fill-Interior within three noise floors of real, and the
two MMDs place it on the near side (-2.7 sd under volumetric chamfer, -8.6 under
IoU: a filled house is nearer to every reference than its hollow original), while
`geom_kid` separates it at 71 sd. Hollow, occupancy noise and column jitter are
within 2.3 sd for all nine. What point clouds resolve is decimation and large
deletions, and all nine fail a blocking gate (eight the occupancy-noise ordering,
IoU-MMD n-stability), so the suite would admit none of them to rank anything. The
row is the measured form of the surface-invariance argument: a metric that
averages nearest-neighbour distances is dominated by surface agreement, whichever
points it samples.

### Interiors are undercounted by the sealed-air measure

`enclosed_air_ratio` counts only *fully sealed* air, so a room with an open
doorway counts as no room at all. `geometry.interior_volumes` also closes the
solid, sealing openings up to two blocks wide, and reports the air that becomes
enclosed only then. Measured on 200 held-out real houses:

* **9.0%** have no sealed interior at all, and **94.4% of those do have a room**;
* apertures account for **20.5%** of all real interior volume.

Every earlier statement in this project that generated builds "have no interiors"
was made with the sealed-only counter, and understates real houses by that much.

### Traversability: can a player walk in?

The interior measure says whether a build has rooms. It does not say whether
anyone can get into them, and the agentic pilots produce exactly that failure: a
house that reads well from outside with a one-block doorway, a two-block ceiling
and a staircase with no headroom. `blockgen.eval.traversal` answers the question
a player would ask, with the rules a player obeys. A two-block body is set down
anywhere outside the building (on the ground, a porch, an outside stair, or in a
doorway holding a door block) and floods everywhere it can walk: through doors,
trapdoors and gates, up one block at a time, up stairs and slabs without a jump,
down any drop, up ladders and vines. The build stands on solid ground, as it does
in the game, so a crop that took the floor away still has its ground-floor rooms.

Per build it reports `interior_standing_reached` (the share of interior cells a
player could stand in that a player can reach from outside; no interior counts as
zero), `floors` and `floors_reached` (storeys with a floor patch, and storeys
reached, so a staircase that does not work shows up), `entrances` (doorways
actually used), `headroom3` (the share of interior standing cells with a third
free block above) and `enterable`. It runs in milliseconds per build.

`scripts/bench_traversal.py` scores the test pool, two probes of its own and the
bench's interior probes, and every cached generated arm, then renders the walk
as a GIF with a quadrant cut away (`walkthrough_<i>.gif`, a four-frame strip,
and `samples.png`). Lime floor is where the player has been, light blue is
interior floor still to reach, red is the two-block stand-in at the wavefront.

| arm | enterable | walkable | floors reached / floors | headroom3 |
|---|--:|--:|--:|--:|
| `real_test` (n=128) | 0.95 | 0.88 | 1.67 / 1.84 | 0.94 |
| `real@seal_openings` | 0.02 | 0.01 | 0.03 / 1.98 | 0.94 |
| `real@lower_ceilings` | 0.82 | 0.73 | 1.22 / 1.47 | 0.56 |
| `real@solidify` | 0.19 | 0.17 | 0.29 / 0.32 | 1.00 |
| `real@hollow` | 0.97 | 0.86 | 1.20 / 1.45 | 0.98 |
| `canon_oriented` | 0.59 | 0.51 | 0.70 / 0.87 | 1.00 |
| `llm_control` | 0.02 | 0.02 | 0.02 / 0.02 | 0.88 |
| `dream-caption` (n=133) | 0.51 | 0.44 | 0.53 / 1.16 | 1.00 |
| `classic-matched` (n=133) | 0.68 | 0.62 | 0.77 / 1.17 | 1.00 |

`seal_openings` bricks up every doorway and window and turns every door to wall:
interior volume is unchanged and nobody gets in, so the interior ratio does not
move and the walk goes to zero. `lower_ceilings` fills the top layer of every
room: the exterior is untouched, rooms that were two high are gone to a player,
and `headroom3` drops from 0.94 to 0.56. `hollow` leaves the walk alone, as it
must, because it deletes only blocks no camera and no player can reach. The 5%
of real houses that are not enterable are open structures with next to no
standing room (a windmill, pavilions) and a tower whose ladder stops four blocks
short of its room.

Two cautions. The walk is seeded from any exterior standing cell, so it tests
the building and not the site: a crop that cut away the hill, left a yard wall's
gate outside the box, or started an outside stair in mid-air does not count
against the house, and a balcony door counts as a way in. And the interior
recipe's 3x3x3 closing eats cavities thinner than three blocks in any direction
when they have an opening, so a two-high room with a door is not interior under
either measure; it is a doll's house under this one and no room under the other.

Run `python scripts/bench_traversal.py --n 128 --gif 2` (with the repo on
`PYTHONPATH`); the run above is `outputs/run_20261023_004317_bench_traversal`.

**Wired into the bench.** Every FAST-tier scorecard now carries a `traversal`
block (the four player readings above against the real distribution, in the same
distance-to-real shape as `coherence` and `geometry_scalars`), and BlockScore
reads it as its own pillar, combined by RMS within the block and entering the
max like every other pillar. The geometric descriptor carries the material-blind
part (standing room, storeys, headroom) so `geom_kid` sees a doll's house too.
Results of the first re-score are in results.md T64.

## BlockScore: one column to rank on

```bash
python -m blockgen.eval.bench --tier both --n 256 --arms ...
```

A competition needs a single sorted column, and that is a hazard: T23c measured a
model that does nothing but recite its training set scoring **second best in the
table** on MV-DINO-KID. Realism alone has a trivial winning strategy. So:

**Novelty and diversity are gates, not terms.** A term can be bought, accept a
poor novelty score, pay for it with realism, and the arithmetic allows it. A gate
cannot. Copying the training set or emitting one build repeatedly is
*disqualified*, and the scorecard names the gate.

**Pillars combine by worst case, never by average.** BlockScore is the **maximum**
distance-from-real across pillars, so every pillar is a veto and "looks right from
outside" cannot buy "is built wrong inside", the trade the render tier
specifically invites.

The pillars are `appearance` (MV-DINO-KID), `geometry` (geom_kid), `palette`
(exact palette JSD), and three blocks of distance-to-real scalars combined by RMS
within the block: `coherence`, `geometry_scalars`, and since 2026-10-23
`traversal`, the player's reading (can a two-block body get in, how much of the
standing room does it reach, how many storeys, is there headroom). A house nobody
can enter loses on `traversal` however it renders.

**Units are real-sample spreads.** Each pillar is `(value − real) / sd(real)`,
where `sd(real)` is measured *in the run* as the spread of a fresh held-out real
sample of the same size against the same reference. 0 means indistinguishable
from real; there are no weights to tune, which is deliberate, a weighted sum is
where a benchmark's authors put their thumb.

**The aggregate is validated like a metric.** `composite.validate()` scores the
known-degenerate control arms and asserts real wins and every cheat loses; a run
whose self-validation fails prints a warning telling you not to quote the
ranking. The checks are not decorative, they caught two real defects on their
first run (a memorisation gate silently disabled by a zero-width control
interval, and a blindness check applied to a tier that cannot see it).

## Protocols: what makes two runs comparable

BlockScore is calibrated **within a run**: every pillar is `(value - real) /
sd(real)`, where `real` is the `real_test` control scored in that same run, on
that corpus, at that `n`. That is what lets the number be read without hand-set
weights, and it is exactly why two scores from two runs are not automatically on
one ladder.

**The descriptor is part of the ruler (2026-10-23).** `houses32-v1` now pins the
geometric descriptor (`descriptor="patch2_orb55_t5_h8_s12"`, the key the geometry
ladder is stored under). The panel gained the player's room scalars and BlockScore
a `traversal` pillar on that date, so a card scored before it carries a BlockScore
in a different unit under the same name; the board unranks it with that reason
("re-score with the current bench") rather than letting `select=best` pick an
old number over a new one. Cards do not currently carry the stamp:
`runner.py:303` sets `ctx.derived["geom_descriptor"]` and `runner.py:364` then
assigns `ctx.derived` a fresh dict, so the key is gone before the card is
serialised, and no card on disk has `context.geom_descriptor`. The lab
re-derives the descriptor by presence (`protocol.run_descriptor`,
`protocol.py:159-175`): an arm with `walk_standing` in its `geometry_scalars`
was scored by the current descriptor, anything else by the legacy one. Read the
`run.protocol` block on a card as what the runner believed before scoring
(`runner.py:435`), and the board's re-derived verdict as the one that counts.
The re-score of every ranked arm is
`run_20261023_011035_bench_traversal_pillar_leaderboard_re-score`.

`blockgen/eval/bench/protocol.py` pins the conditions under which they are:

| field | `houses32-v1` | why it changes what a score means |
|---|---|---|
| `corpus` | `houses_32` | a different corpus is a different distribution |
| `split_key` | `houses_32.s0.70-15-15.v1` | a different held-out set is a different target |
| `tier` | `both` | `fast` measures no `appearance` pillar at all |
| `min_n` | 128 | per arm, and **also checked against `real_test`** |
| `min_ref` | 256 | the real sample the distances are measured against |
| `seed` | 0 | a different split seed is a different held-out set |
| `descriptor` | `patch2_orb55_t5_h8_s12` | a different geometric descriptor is a different `geom_kid`, and so a different BlockScore unit |

`min_n` applying to the calibration arm is the one that is easy to miss and
expensive to get wrong. Measured on the cards on disk: `native_oriented` scores
**87.14** against a 128-build `real_test` and **19.05** against a 32-build one,
same model, corpus, split and tier. The submission did not change; the ruler did.
A protocol that checked only the submission's `n` would have published that as a
4.5× improvement.

`--n` caps controls and baselines but **not** file-backed arms, so one run
legitimately holds a 128-build control beside a 64-build submission. That is why
the run-level and arm-level checks are separate: the run can be official while
one of its arms is unranked, and the leaderboard says which rather than letting
the arm vanish.

The runner records its verdict in `run.protocol`
(`{"id", "official", "reasons"}`), but **the lab re-derives it on every page
load** rather than trusting it. Tightening a protocol therefore takes effect
immediately instead of invalidating every card on disk, and cards written before
protocols existed still get a verdict.

### What `train_split` means

`match_arm` (`protocol.py:262-297`) also checks training provenance. A
generated arm's `report.train_split` must be one of
`CLEAN_TRAIN_SPLITS = ("canonical", "protocol")` (`protocol.py:260`) to be
ranked. A missing value unranks the arm with "training split not recorded";
any other value unranks it with the split named. Controls and baselines carry
no `report` and skip the check. The value is written by whatever produced the
arm's npz, and the writers disagree:

| writer | what it records |
|---|---|
| `python -m blockgen.train` (`blockgen/train/sample.py:346-352`) | `TrainConfig.split_name()` (`blockgen/train/config.py:106-111`): the explicit `train_split:` config field, else `canonical` when `train == [sft-houses32]`, else the comma-joined tier list. Also `model`, `arch`, `kind`, `checkpoint {file, step}`, `trained_on_signals`. |
| `python -m blockgen.posttrain.bon` (`blockgen/posttrain/bon.py:65-75`) | the parent run's `split_name()`, its `model` dict, the checkpoint path, `parent_signals` |
| `scripts/sample_to_npz.py --train-split <value>` (`:109-114`) | the flag verbatim, else `models.json`'s `train_split` for the model |
| `scripts/dump_samples.py` (`:118-121`) | nothing; an arm converted this way is unranked until the manifest is edited by hand |
| `blockgen/agentic/core.py:1053` | `train_only` when the method read the corpus, else `none`; `oracle_train` for the oracle arms (`suite.py:136`) |

A pretraining-only or mixture run through the trainer is therefore unranked by
construction until `train_split: canonical` is set in its config, and the board
shows the reason rather than hiding the row.

### The slate: when a strip of renders is a comparison

A protocol may also pin a `Slate`: the fixed prompts every arm should render, so
column *i* of the leaderboard is the same request on every row. `pick_slate`
selects in **slate order**, matches case- and whitespace-insensitively, and
**skips** a prompt the arm produced nothing for rather than substituting another
build, a hole says "no build for this prompt", which is true, where a
substitution would put the wrong build under that label.

For an unconditional protocol there is nothing to align: you cannot ask two
unconditional models for the same build. The slate degrades to *k* slots under a
pinned seed, `Slate.aligned` is `False`, and the page is required to say so.
`houses32-v1` is unconditional, so today every strip is slots.

## Baselines: the floor a leaderboard needs

```bash
python -m blockgen.eval.bench --tier both --n 128 --baselines --arms ...
```

"MV-DINO-KID 0.188" means nothing until you know what a hand-written box scores.
Five non-learned arms, seeded and fitted on the **train split only**, so a
baseline never competes with an advantage a submitted model lacks:

| baseline | what it is |
|---|---|
| `uniform_random` | random voxels at real density, size and palette. The floor. |
| `shell_box` | hollow box: floor, four walls, flat roof. |
| `patchwork` | 8³ patches lifted from *different* real builds and tiled. |
| `gabled_house` | shell box + pitched roof + gable ends + door + windows. |
| `train_copy_noise` | a training build with 10% of its blocks displaced. |

`gabled_house` is the bar that matters: it is what a competent afternoon of rules
produces, and a benchmark on which a learned model cannot beat it is not yet
measuring what a learned model is for.

`patchwork` is an **attack, not a baseline**. Every patch is verbatim real, so
the local 2×2×2 pattern statistics that do most of the work in `geom_kid` are
near-perfect while the global structure is nonsense. Where it lands is the honest
statement of the descriptor's limitation, and it is reported rather than avoided.

## Dose-response: does a better build score better?

```bash
python scripts/bench_doseresponse.py --n 128 --bad-pool uniform_random shell_box
```

The ladder asks a binary question: can this metric tell damaged from real. A
leaderboard needs the harder one, does the number move smoothly and in the right
direction as quality changes? A metric can pass every gate by firing on an
artifact while being flat across the range that separates two submissions.

Four axes with known ground-truth ordering (`solidify`, `occ_noise`, `canon`,
`mixture`). On three of them both realism metrics are perfectly monotone
(Spearman +1.000) across 105–226 noise floors. On `solidify`, `mv_dino_kid`
scores **+0.300, non-monotone, dynamic range 0 noise floors**, its whole
response to progressively deleting every interior is smaller than one draw of
its own sampling noise, against `geom_kid` at +1.000 over 133.

`mixture` is the axis that matters most for ranking, because it is how
generators actually fail: some good builds, some bad. Both metrics are monotone
in the fraction of an arm that is real, which is the direct answer to "does this
benchmark reward better output".

**What the bad half of a mixture is.** `--bad-pool` names arms from
`bench.baselines` (`uniform_random`, `shell_box`), which are bad along a *quality*
axis at fixed resolution. The original axis mixed real builds with their own
canon-8 decimations, which this suite no longer treats as a quality reference for
the reason `fast.control_arms` gives, decimation changes a build's size, not its
quality, and a canon-8 house has 47 blocks against a real 870. `canon8` remains
selectable to reproduce the older numbers; every mixture axis carries the pool in
its name and in the run directory's.

### But the response is quadratic, so read `sqrt`

`MMD²(αP + (1−α)Q, P) = (1−α)²·MMD²(Q,P)`. A squared-MMD metric therefore
responds to the *fraction* of bad samples quadratically, and it does: against the
`uniform_random` mixture at n=128 the best-fit exponent is **1.99** and the
quadratic's residual is 0.008 against the linear's 1.064. **An arm that fails on
a quarter of its builds looks 25% as bad as one that fails on all of them** on a
linear reading, and reaches only 6.4% of the fully-bad value.

`sqrt(KID)` undoes it, it recovers the true bad fraction to a mean absolute
error of **0.011** (`mv_dino_kid`) and **0.022** (`geom_kid`), against 0.124 and
0.135 for the raw values. Every squared-MMD metric in
the scorecard now carries `mmd` beside it. Read `mmd` for "what fraction of this
arm is bad"; read the metric for "is this the right distribution".

## Human study: the agreement number a metric paper needs

```bash
python scripts/human_study_export.py --per-arm 12 --per-pair 8 --arms name:path.npz
python scripts/human_study_analyze.py --key <run>/answer_key.json \
    --responses responses.json --scorecard <run>/scorecard.json
```

The ladder proves the metrics rank *synthetic corruption*. It does not prove they
agree with people, and that is the first thing asked of a new metric.

**Why the task is pairwise over builds, not over sets.** The metrics are
two-sample statistics, so there is no per-build number to correlate with a
per-build rating. Asking raters to compare two *grids* would match the metric's
level but asks people to eyeball a distribution, which they are poor at. So
raters compare individual builds, an easy, reliable judgement, and the trials
are pooled to the **arm pair**, which is the level the metric speaks at.

The analysis order is fixed in advance so no step can see the next: exclude
sessions below 75% on real-vs-noise catch trials, fit **Bradley-Terry** arm
strengths (raw win rate depends on which opponents an arm was drawn against, and
coverage is uneven by design), then report pairwise agreement on humanly-decided
pairs and Spearman against the BT scores. Pairs where humans are near chance are
reported but excluded from agreement, scoring a metric on coin flips drives any
metric toward 50%.

The export writes a self-contained page with the arm labels stripped, and the
analysis is tested end to end against simulated raters, including inattentive
ones it must exclude.

## Head-to-head: overlapping intervals are not a tie

Marginal intervals are not a comparison. Two arms are separable when their
difference exceeds `sqrt(h_a² + h_b²)`; non-overlap of their intervals demands
`h_a + h_b`. For equal widths that is 2h against 1.41h, **the overlap heuristic
needs the difference to be 41% larger before it will call anything.**

`compare.rank_table` tests every pair directly, with both arms scored against the
same reference in every replicate, a shared subsample for the interval, the full
reference for the test, and applies Holm–Bonferroni across
the family (with 7 arms there are 21 tests, and the chance of one spurious
"significant" result at α=0.05 is about two in three). The letters are a valid
compact letter display, maximal cliques of the "not separated" graph, so an arm
may hold several, which means both readings hold: arms sharing a letter were not
separated, and arms sharing **no** letter were.

Two flags shape the table. `--n-rep` (default 200, `runner.py:259`) is the
number of 80% subsample draws behind each pair's **interval**; it has no effect
on the p-value. `--sig-method permutation --n-perm 10000` (`runner.py:260-271`)
is the test itself, and `--n-perm` has to clear the Holm family, as the third
bullet below explains.

What it looks like on a real run (n=128 controls, 399-build reference; arms
sharing a letter are not separated at α=0.05 after Holm):

| arm | `mv_dino_kid` | grp | `geom_kid` | grp |
|---|---|---|---|---|
| `real_test` | −0.001 | **a** | −0.095 | **a** |
| **`real@solidify`** | 0.000 | **a** | **22.03** | **f** |
| `real_shuffled_materials` | 0.676 | f | −0.095 | **a** |
| `real@monochrome` | 0.930 | g | −0.095 | **a** |
| `train_verbatim` | 0.015 | b | 0.960 | b |
| `agentic_oneshot` | 0.107 | c | 8.52 | c |
| `real@canon16` | 0.110 | c | 6.57 | c |
| `native_oriented` | 0.188 | d | 9.46 | d |
| `real@canon8` | 0.484 | e | 18.58 | e |
| `pick_n_place` | 0.533 | e | 24.06 | f |
| `real@single_mode` | 1.805 | h | 41.72 | g |

`real@solidify` shares group **a** with held-out real on the render metric, and on
`geom_kid` the same arm is separated from it at Δ = −22.13 [−24.08, −20.72].

> **The table above predates the 2026-09-19 calibration fix, and its letters are
> the old test's.** Two things about it are now known to be wrong: the p-values
> came from a miscalibrated test (see the second bullet), and the letters came
> from a greedy one-letter-per-arm assignment that is not a valid compact letter
> display, so "has a group of its own" was never a supported reading of it. The
> claim this paragraph used to make from that table, that the same test at the
> same n separates `train_verbatim` at Δ = 0.015, so the solidify tie is not a
> power failure, **does not survive the fix**: recomputed on the canonical
> 18-arm run, `train_verbatim` vs `real_test` on `mv_dino_kid` lands at an
> adjusted p of 0.05–0.06 and is no longer separated. The solidify tie itself is
> unaffected, and in the safe direction: a conservative test cannot manufacture a
> tie. See `outputs/run_20260919_022424_significance_recompute/FINDINGS.md`.

Three details that are easy to get wrong, and were:

* **The p-value cannot be a sign-count.** A bootstrap sign-balance p is floored
  at `1/n_rep`; Holm over the 55 pairs of an 11-arm run multiplies 0.005 to
  0.275, so *nothing* is separable however large the difference. The first run
  put held-out real and a monochrome corpus in the same group. The sign-balance
  value is kept only as `p_boot`.
* **Nor can it be a studentized subsample spread.** The version that replaced the
  sign-count divided Δ by the standard deviation of the 80%-subsample replicates
  and called that a standard error. It is not one, an m-out-of-n subsample needs
  a rescale before its spread becomes an SE, and the test rejected 3.4x too
  often at α=0.05 with the arms near the reference and 7.2x too often with the
  arms away from it, which is where a leaderboard's generator-vs-generator pairs
  sit. No single rescale fixes both, because the statistic is a degenerate
  U-statistic in the first case and a non-degenerate one in the second. The
  default is now an exact label **permutation** test (`--sig-method`, `--n-perm`),
  measured at nominal in both regimes; `--sig-method subsample` reproduces old
  cards and nothing else. An exact p is floored at `1/(n_perm+1)`, so `--n-perm`
  has to clear the Holm family: at 18 arms, 10k bottoms out at an adjusted 0.015.
* **The pairing is the correct null, not a power gain.** Measured across four
  reference and arm sizes it moved the interval width by under 10% and not
  consistently in one direction, the reference's contribution to an unbiased
  MMD is a term the difference already cancels analytically. That cancellation
  *is* worth having for a different reason: `distances.kid_arm_term` skips the
  reference self-kernel entirely, which is exact and takes a 66-pair table from
  tens of minutes to about one.

## Rules the code enforces

**Every metric carries an interval and a direction.** `scorecard.Metric` cannot
represent a bare float, so a coherence rate can never be printed without the real
value beside it. That matters because only ~66% of *real* val houses are a single
connected component, an arm at 1.0 is as far from the data as one at 0.2.

**No metric reports before it passes the ladder.** `ladder.py` scores each metric
on rungs of known damage (decimation, block noise, interior deletion, material
shuffle) and on invariances that must not move it (rotation, mirroring). Validity
failures are blocking and surface as `null` naming the failed gate; sensitivity
limits ("cannot resolve a 40% cut at n=64") are recorded but never suppress a
value, because those are answered by more samples.

**Decimation is a ladder rung, not a leaderboard arm** (policy set 2026-09-17).
`probes.canon` still builds `canon16`/`canon8` for the metric-validation ladder,
where the only question asked of them is whether a metric orders a known
corruption correctly. They are no longer scored as controls on the leaderboard.
A coarser grid is a change of *resolution*, not a loss of *quality*, it yields a
smaller building, not a worse one, so a row ranking a generator against it
invites "this model is worse than 86%-deleted real data", which sounds damning
and means nothing. Scorecards written before this date carry the two rows; the
raw numbers in them are unaffected, but do not read them as a quality bar.

**Real data is scored as an arm.** `real_test` is the floor, `train_verbatim` proves the memorization
detector fires. They run through the identical code path as any model.

**The `context` block *is* the object the arms were scored against.**
`card.context` is `BenchContext.to_json()`, re-serialized once after the last
late-measured value lands on it, not a hand-built dict assembled beside it. The
parallel dict it replaced silently dropped four values the arms really were
scored with (`palette_level`, `dup_threshold`, `n_ref`, `sizes`), and a context
that is not the context is worse than no context at all. The same rule is why
`text_backbone` appears only inside `context.backbone`, on the full tier: a
fast-tier card must not claim a backbone it never loaded.

**A failed example write costs the strip, never the run.** Each arm's pick is
guarded on its own and the whole write is guarded again; a failure becomes a
`card.warnings` entry. The scorecard is written last and unconditionally, and the
per-arm row indices are merged onto it in the same place the npz is produced, so
the card either names a file that exists or says nothing at all.

**Novelty sits on the same row as realism.** `train_verbatim` scores MV-DINO-KID
0.015, second only to real data, and is caught only by `dino_nn_percentile`.

**A metric is judged only on damage it can see.** `RungSpec` says which rungs
gate which family. Gating the occupancy-only geometry metrics on material noise
would fail them for doing exactly what they were built to do, so their noise rung
moves blocks instead of retyping them, and material corruptions join their
*invariance* set, a positive statement of the blindness rather than a hole in
the ladder.

## Corrections this suite has had to make to itself

**A renderer that failed open.** `render_views` used to substitute a blank white
frame for any exception. A PyOpenGL/Python version mismatch then made every
`render_structure` call raise: every image came back white, every DINO feature
collapsed to one vector, and every arm scored a KID of −4e-13, i.e. *better than
real*. Worse, `load_or_build` verifies a `render_canary_sha` before trusting its
cache, so the all-white canary would have failed to match and the 30 MB feature
cache would have been silently rebuilt from blank frames. It now raises past a 2%
failure rate. (Repairing PyOpenGL reproduced the stored canary bit-for-bit, so no
cached number was affected.)

**The noise floor had the wrong denominator, twice.** `sd(real)` is the unit every
gate is stated in, and it was estimated from repeated subsamples of a pool that
*excluded* the probe set. Two errors: the "fresh real draw" G7 tests was not a
member of its own null distribution, and overlapping draws carry a finite-
population factor that shrinks their spread. Corrected, `geom_kid` moved from
failing G7 at 3.5 sd to passing, a metric that separates `solidify` at 68 sd had
been rejected for noise that was mismeasured.

**G8 was testing a difference of means against the spread of a single draw**,
larger by `sqrt(reps)`, so it had almost no power. It let `legacy_cmmd` pass
despite that estimator's null mean halving every time n doubles (0.173, 0.100,
0.058, 0.039 at n = 16/32/64/128). Tested against the standard error of the
difference, the same drift scores **14.8 se** and is rejected. `fd` and
`one_minus_coverage` fail the same way, which is what the earlier verdicts said
for weaker reasons.

## Sample sizes

One table, four thresholds, and the code that enforces each:

| n per arm | what it buys | enforced where |
|---|---|---|
| below 16 | nothing: the card warns "intervals are not interpretable" and `BlockScore` marks the arm `unranked` | `--min-n` default 16 (`runner.py:244`, `scorecard.py:329`); `fast.py:214-216`; `composite.py:237-243` |
| 16 to 127 | a readable card for a smoke test or a pilot; never a board row | `houses32-v1` `min_n=128` (`protocol.py:146`), checked per arm and against `real_test` (`protocol.py:244, 284`) |
| 128 | rankable under `houses32-v1` | the board (`tools/lab/standings.py`) and `match_arm` |
| 256 or 399 | quotable: 256 is the paper's floor for a headline number, 399 is the whole test split and what every facade card used (T66, T68) | convention, results.md T59 onwards; `--n-ref` default 512 and `min_ref=256` (`runner.py:243`, `protocol.py:106`) cover the reference |

**Equal n is required.** `sd(real)` shrinks with n, so a BlockScore at n=128
and one at n=399 are in different units even under one protocol (T35). The
protocol does not enforce this; the lab groups the board by n, opens on one
group and puts an amber banner over any view that mixes them
(`tools/lab/standings.py:249-290`, `/api/standings?n=`). Compare two arms only
inside one group, and re-sample the smaller one rather than reading across.
KID's noise floor is 0.005 at n=64 and 0.04 at n=16, against a real-to-canon-16
gap of 0.18, which is where the 16 came from.

## Caches

Two of these are gitignored and gate a tier, so a fresh clone has to build them
before its first official run.

* `data/minecraft/splits/`: the canonical split, tracked in git
  (`.gitignore:13` re-includes it).
* `data/minecraft/features/<corpus>/<split>/<view>/<backbone>.npz`: not tracked
  (`.gitignore:6`); rebuilt in minutes by `python -m blockgen.eval.bench.features`.
  Invalidates on structure-set change, view change, backbone change, or a
  `render_canary_sha` mismatch (a changed texture pack would otherwise poison
  every cached number invisibly). The FULL tier raises without it
  (`full.py:76`).
* `outputs/analysis/ladder/<backbone>__<view>.json`: validation verdicts, one
  file per metric family, under `outputs/`, which `.gitignore:27` ignores
  wholesale. The three on disk today:
    * `geom__patch2_orb55_t5_h8_s12.json`, the geometry ladder. Its key is
      `ladder.geom_view_key()` (`ladder.py:451-458`) and the `_s12` records the
      scalar count, so the room scalars of 2026-10-23 produced a new file
      beside the old `geom__patch2_orb55_t5_h8.json`. Without it the FAST tier
      scores `geom_kid` as skipped with the reason "no geometry ladder; run
      `bench.ladder --family geometry`" (`fast.py:363`), and `houses32-v1` pins
      this exact descriptor, so build it first:
      `python -m blockgen.eval.bench.ladder --family geometry --n 64 --n-ref 256`.
    * `dinov2b__v4_az45-135-225-315_el30-30-30-30_px224_ortho.json`, the render
      ladder for the FULL tier (`python -m blockgen.eval.bench.ladder --family render`).
    * `pointcloud__k1024_canvas32.json`, the volumetric row (T67), read by
      nothing in the runner.

## Frozen surfaces

These are the things a change to this suite must not move. Each has a test or a
consumer that breaks, and the reason is recorded next to it.

| surface | where | why it is frozen |
|---|---|---|
| `eval/novelty.py`, `eval/validity.py`, `eval/perceptual.py` | `blockgen/eval/` | every number in results.md T1 to T22 reproduces from them; `tests/eval/test_fast.py:47` pins `bench` against `evaluate_novelty` bit for bit |
| `meta.source` | `fast.py:220-227`, exactly `arm.npz or "in-memory"` | five consumers join on the literal string: the lab's dataset join, `scripts/bench_report.py`, `scripts/human_study_analyze.py`, the Curate link, the thumbnail fallback |
| the committed split | `data/minecraft/splits/`, `houses_32.s0.70-15-15.v1` | every protocol number is against this test set; a new split is a new protocol id |
| `houses32-v1` | `protocol.py:140-154` | the only protocol; its fields are the unit of every ranked BlockScore. Add a second protocol beside it, never edit this one |
| the lab HTTP contract | the docstring of `tools/lab/__init__.py` ("Frozen here so the pages and the server can be built against one description") | every page is built against it; additions are fine, renames are not |
| the build-id grammar | `"<dataset_id>:<index>"`, pinned literally in `tests/lab/test_arm_ids.py` | build ids are the primary key of `outputs/lab/lab.db`; a changed slug orphans every label, note and 2AFC comparison |
| `probes.shuffle_materials` | `blockgen/eval/bench/probes.py:38`, a copy of the function in `scripts/validate_perceptual.py` | the script's recorded output is a published result; `tests/eval/test_probes.py` pins the two copies to identical output at a fixed seed |
| the golden cards | `tests/fixtures/scorecards/` | the only committed scorecards; the README forbids hand-made cards, so a reader change is tested against real shapes |
| `bench/2` as `bench/1` plus keys | `scorecard.SCHEMA_VERSION` | nothing removed, renamed or re-typed; `scripts/bench_report.py` reads `card["arms"]` bare and is the consumer that hard-crashes |

## Relationship to the older metrics

`eval/novelty.py`, `eval/validity.py` and `eval/perceptual.py` are **not
modified**, so every number in results.md T1–T22 stays reproducible. `bench`
reuses `evaluate_novelty` (pinned bit-for-bit by a test) and scores
`perceptual.cmmd` in the ladder as a labelled legacy metric, where it fails
n-stability, because it is the biased MMD estimator despite its docstring. See
results.md T23.
