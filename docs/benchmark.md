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
render half is the half that can break silently, and it did — see *A renderer
that failed open* below.

Neural arms that dump per-sample `.npy` need one conversion step first:

```bash
python scripts/dump_samples.py --keyed-npz outputs/run_x/arm/samples.npz \
    --out-dir outputs/run_x/arm --name arm --max-dim 32 --crop
```

## What it reports

| pillar | leads | notes |
|---|---|---|
| Realism (appearance) | `mv_dino_kid` | unbiased polynomial-kernel MMD on pooled DINOv2 features |
| | `mv_dino_mmd_rbf` | second opinion, RBF kernel, bandwidth frozen on the reference |
| **Realism (structure)** | **`geom_kid`** | **the same estimator on D4-invariant voxel descriptors; no GPU** |
| Novelty | `dino_nn_percentile` | resolution-independent; low = memorized |
| | `voxel_nn_iou_mean`, `*_dup_rate` | delegated to `eval/novelty.py`, unchanged |
| Coherence | `lcc_ratio`, `enclosed_air_ratio`, `floating_block_frac` | reported as **distance to real** |
| **Structure** | **`thickness_*`, `wall_frac`, `yaw_symmetry`, `interior_ratio_open`** | **also distance to real** |
| Dataset stats | `palette_jsd_{exact,family}`, `palette_cooccur_jsd` | plus block-count / bbox W1 |
| Faithfulness | `retrieval_acc` (chance 1/32) | `mv_clipscore` and material agreement alongside |
| Cost | `usd_per_coherent_build` | agentic only; never combined with GPU-hours |
| **Aggregate** | **`BlockScore`** | **worst pillar in real-sample spreads, with cheat gates** |

## Run identity, provenance and examples

A `bench/1` card recorded five things about the run that produced it: `dir`,
`git_sha`, `cmd`, `tier`, `elapsed_s`. Eleven of them are on disk here, every one
of them in a directory called `run_<stamp>_bench`, and the only way to tell them
apart is to open them. Worse, on the largest card thirteen of sixteen rows said
nothing about themselves except `source: "in-memory"` — the arms were built in
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
all) and it names the directory, through `runner.run_slug` — non-alphanumerics to
`_`, truncated to 40 chars **on a word boundary**, because a naive `[:40]` turns
"ontology arms vs agentic ontology probe" into `..._bench_ontology_arms_vs_agentic_ontol`,
which reads as a typo rather than as an ellipsis. The `bench_` prefix is kept in
front of the slug so every `outputs/run_*_bench*` glob written in anyone's notes
keeps matching. `--note` is one line of display-only context. `--examples 8` is
the default; `--examples 0` disables the sibling npz. A run with no `--name`
still works and prints a one-line reminder at the end, because that is the flag
whose absence you only regret three weeks later.

**The name is never an id.** It is free text, it is not unique — four cards on
disk derive the identical label `native_oriented vs agentic_oneshot vs pick_n_place` —
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
| `dir` | str | `--out` or `utils.runs.new_run_dir` | **not identity** — informational only |
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
its `argv[0]` is an absolute path to the module file — on the smoke run,
`/home/.../blockgen/eval/bench/__main__.py --tier fast --n 24 ... --name smoke run`,
which is not a command, and whose unquoted `smoke run` would parse as two
arguments if it were. `rerun` is built by `shlex.join` from `argv` kept as data:
`python -m blockgen.eval.bench --tier fast --n 24 --min-n 8 --no-head-to-head --name 'smoke run' ...`.
That claim is testable and is tested: `build_parser()` was factored out of
`main()` for no other reason than to let a test re-parse the string a run wrote
and assert it round-trips.

**`run.dir` is not identity.** Readers use `path.parent.name`. The two oldest
cards carry `"dir": "/home/.../scratchpad/bench_compare"` and
`".../scratchpad/bench_pnp"` — absolute paths outside `outputs/` entirely, from
runs written with `--out` — which is exactly why the directory name, and only
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
| `kind` | `submission` \| `control` \| `baseline` — derived from `track` at write time by `ArmSpec.kind`, the single definition of the rule, so a card read on its own groups its rows by reading one field instead of re-deriving `track not in {control, baseline}` in three places across two languages |
| `origin` | `npz` \| `in_memory` — "nothing was ever written" and "the dataset join failed" used to render as the same sentence |
| `source_run_id` | the first `run_*` segment of the checkpoint path (else of `source`), or `null` — the edge back to the run that produced this arm's inputs |
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
| `scripts/sample_to_npz.py:115-119` | model, checkpoint, n, n_empty, seed, temperature, top_k | model, checkpoint, seed, temperature, top_k |
| `scripts/dump_samples.py:120` | source, n_source, n_written, n_empty, seed, cropped | source, seed |
| `blockgen/agentic/report.py:54-71` | arm, n_requested, n_nonempty | arm |
| the pick-and-place manifest (`outputs/bench_arms/pick_n_place_32_manifest.json`) | model, epochs | model, epochs |

Promotion is a convenience for whoever renders the page; `report` is the record.
A key nobody promotes is not lost, it is simply one level deeper.

`provenance()` is deliberately **self-sufficient**: if `load()` has not run it
reads the `_manifest.json` sibling itself. The first version depended on
`score_fast` calling `load()` before it built the meta block — an invariant
nothing in the type stated, and one that `score_full` and the examples writer
each break by calling `load()` on their own. It never raises: a missing manifest
is a fact about the arm, not a reason to lose a two-hour scoring run.

### The control recipes live in one place

An in-memory arm has no manifest to read, so `fast.control_arms` and the baseline
arms attach a `provenance_override` naming their builder, the split they were
drawn from, `n_requested`, `seed` — and a one-sentence **recipe**:

```
real@solidify: every enclosed air cell of a held-out test build filled; a median
31% more blocks, every room deleted, the silhouette untouched — the control that
tells the two realism tiers apart, since at n=128 MV-DINO-KID cannot separate it
from real and geom_kid puts it 205 spreads away.
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
* **The only image routes are build-id keyed** — `/api/thumb/<build_id>` and
  `/api/view/<build_id>/<k>`. Adding `/api/runfile/<run>/<name>` would open a
  traversal surface over a directory full of checkpoints and bypass the
  content-addressed render cache, the blank-frame guard and the single-thread
  discipline, to buy a thumbnail the render pipeline already produces. So an
  example has to be expressible as a **build id**, not as a path.
* **An npz + manifest pair is already a dataset.** The lab admits a dataset iff
  `<x>.npz` and `<x>_manifest.json` both exist, and it already rglobs
  `outputs/run_*/**/*.npz`. Writing that pair — with
  `curation.houses.save_house_cache`, so there is no new IO code — makes the
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
says so: **a seeded sample, explicitly not a best-of.** A stratified policy —
by worst-pillar contribution, say — is a body change to one function, and is not
in this increment.

Rows are written as **copies** with metadata set explicitly
(`dataclasses.replace`, never in place). Two independent reasons, both measured:
probe and baseline builders construct structures with no metadata at all, so
without this most of a card's rows would be blank-captioned thumbnails; and
`real@single_mode` is one `Structure` repeated *n* times, an object that is
simultaneously row 0 of `real_test` and a member of the shared `test` split the
lab serves — an in-place edit would corrupt three datasets at once for the life
of the process.

Warm the thumbnails before a demo, rather than watching them trickle in:

```bash
python -m tools.lab.prerender --datasets arm:<run>__examples_48 --px 96 --workers 8
```

Two things are deliberately absent. There is no `examples.png` — the paper path
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
and `tools/lab/cards.migrate` is the one place that dispatch happens — called
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
those eleven cards before it ran. `bench/2` is therefore `bench/1` **plus keys**:
nothing removed, renamed, or re-typed, `Metric.to_json()` untouched, `meta.source`
byte-identical. That rule is enforced by what would break — `scripts/bench_report.py`
reads `card["arms"]` and `entry["gen"]["mean"]` bare and is the one consumer that
hard-crashes rather than degrading.

Nothing on disk is ever rewritten: no backfill, no renamed directory. A legacy
card is read, its gaps are *named* on the page, and every table that parses is
drawn. Hand-adding a `run.name` to an old `scorecard.json` is picked up on the
next page load — that is the documented escape hatch, and it is deliberately not
automated.

**Reserved, not implemented.** A training run needs identity too and will never
emit a scorecard, so the vocabulary a future universal run manifest would use —
`kind`, `artifacts`, `origin`, `status`, `created_at` — is reserved here rather
than invented twice. When that file exists, lifting the `run` block out should be
a rename, not a redesign.

## Two realism tiers, because one of them is blind

Every realism number in this suite used to be computed on *renders*. That is what
makes a 16³ arm and a 32³ arm comparable at all, and it is why the render tier
exists — but it buys comparability by discarding everything a camera outside the
build cannot reach. The cost is measurable and it is not small.

`probes.solidify` fills every enclosed air cell: a median **31%** more blocks,
every room in the corpus deleted, silhouette untouched. Scored at n=128 against
the same 399 held-out real builds:

| arm | `mv_dino_kid` ↓ | `geom_kid` ↓ |
|---|---|---|
| `real_test` (floor) | −0.001 [−0.004, 0.003] | −0.09 [−0.22, 0.04] |
| **`real@solidify`** | **0.000 [−0.002, 0.004]** | **22.1 [20.7, 23.7]** |
| `real_shuffled_materials` | 0.673 [0.642, 0.710] | *−0.09 — bit-identical to real* |
| `real@monochrome` | 0.930 [0.903, 0.962] | *−0.09 — bit-identical to real* |

The intervals in the first two rows **overlap**. A generator that emits solid
blobs shaped like houses is, to the render-space realism metric, indistinguishable
from one that emits houses. That is not hypothetical: T23d measured
`native_oriented` at an enclosed-air ratio of 0.005 against a real 0.114 while it
scored respectably on KID.

The last two rows are the converse, and they are why this is a second tier rather
than a replacement. `geom_kid` reads occupancy only, so a material permutation
moves it by *exactly zero* — the two tiers are complementary halves, and the
ladder gates each on the rungs it is built to see (`GEOMETRY_RUNGS` swaps the
material-noise triple for an occupancy-noise one and puts material corruptions in
its *invariance* set).

### The geometric descriptor

77 dimensions per build, all **exactly** invariant to the D4 yaw group (four
rotations about the vertical axis, and the mirror) by construction rather than by
luck — local patterns are canonicalised onto orbit representatives and every
scalar commutes with the group. Verified bit-for-bit in
`tests/eval/test_geometry.py`.

* **55** D4-canonical 2×2×2 occupancy-pattern frequencies (256 raw patterns → 55
  orbits; the all-empty orbit is dropped)
* **5** taxicab wall-thickness bins
* **8** normalised height-profile bins
* **9** scalars: thickness mean/p90, surface-to-volume, wall fraction, yaw
  symmetry, sealed and aperture-aware interior, height entropy, `log1p(blocks)`

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

### Interiors are undercounted by the sealed-air measure

`enclosed_air_ratio` counts only *fully sealed* air, so a room with an open
doorway counts as no room at all. `geometry.interior_volumes` also closes the
solid — sealing openings up to two blocks wide — and reports the air that becomes
enclosed only then. Measured on 200 held-out real houses:

* **9.0%** have no sealed interior at all, and **94.4% of those do have a room**;
* apertures account for **20.5%** of all real interior volume.

Every earlier statement in this project that generated builds "have no interiors"
was made with the sealed-only counter, and understates real houses by that much.

## BlockScore — one column to rank on

```bash
python -m blockgen.eval.bench --tier both --n 256 --arms ...
```

A competition needs a single sorted column, and that is a hazard: T23c measured a
model that does nothing but recite its training set scoring **second best in the
table** on MV-DINO-KID. Realism alone has a trivial winning strategy. So:

**Novelty and diversity are gates, not terms.** A term can be bought — accept a
poor novelty score, pay for it with realism, and the arithmetic allows it. A gate
cannot. Copying the training set or emitting one build repeatedly is
*disqualified*, and the scorecard names the gate.

**Pillars combine by worst case, never by average.** BlockScore is the **maximum**
distance-from-real across pillars, so every pillar is a veto and "looks right from
outside" cannot buy "is built wrong inside" — the trade the render tier
specifically invites.

**Units are real-sample spreads.** Each pillar is `(value − real) / sd(real)`,
where `sd(real)` is measured *in the run* as the spread of a fresh held-out real
sample of the same size against the same reference. 0 means indistinguishable
from real; there are no weights to tune, which is deliberate — a weighted sum is
where a benchmark's authors put their thumb.

**The aggregate is validated like a metric.** `composite.validate()` scores the
known-degenerate control arms and asserts real wins and every cheat loses; a run
whose self-validation fails prints a warning telling you not to quote the
ranking. The checks are not decorative — they caught two real defects on their
first run (a memorisation gate silently disabled by a zero-width control
interval, and a blindness check applied to a tier that cannot see it).

## Protocols — what makes two runs comparable

BlockScore is calibrated **within a run**: every pillar is `(value - real) /
sd(real)`, where `real` is the `real_test` control scored in that same run, on
that corpus, at that `n`. That is what lets the number be read without hand-set
weights — and it is exactly why two scores from two runs are not automatically on
one ladder.

`blockgen/eval/bench/protocol.py` pins the conditions under which they are:

| field | `houses32-v1` | why it changes what a score means |
|---|---|---|
| `corpus` | `houses_32` | a different corpus is a different distribution |
| `split_key` | `houses_32.s0.70-15-15.v1` | a different held-out set is a different target |
| `tier` | `both` | `fast` measures no `appearance` pillar at all |
| `min_n` | 128 | per arm, and **also checked against `real_test`** |
| `min_ref` | 256 | the real sample the distances are measured against |
| `seed` | 0 | a different split seed is a different held-out set |

`min_n` applying to the calibration arm is the one that is easy to miss and
expensive to get wrong. Measured on the cards on disk: `native_oriented` scores
**87.14** against a 128-build `real_test` and **19.05** against a 32-build one —
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

### The slate — when a strip of renders is a comparison

A protocol may also pin a `Slate`: the fixed prompts every arm should render, so
column *i* of the leaderboard is the same request on every row. `pick_slate`
selects in **slate order**, matches case- and whitespace-insensitively, and
**skips** a prompt the arm produced nothing for rather than substituting another
build — a hole says "no build for this prompt", which is true, where a
substitution would put the wrong build under that label.

For an unconditional protocol there is nothing to align: you cannot ask two
unconditional models for the same build. The slate degrades to *k* slots under a
pinned seed, `Slate.aligned` is `False`, and the page is required to say so.
`houses32-v1` is unconditional, so today every strip is slots.

## Baselines — the floor a leaderboard needs

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

## Dose-response — does a better build score better?

```bash
python scripts/bench_doseresponse.py --n 64
```

The ladder asks a binary question: can this metric tell damaged from real. A
leaderboard needs the harder one — does the number move smoothly and in the right
direction as quality changes? A metric can pass every gate by firing on an
artifact while being flat across the range that separates two submissions.

Four axes with known ground-truth ordering (`solidify`, `occ_noise`, `canon`,
`mixture`). On three of them both realism metrics are perfectly monotone
(Spearman +1.000) across 105–226 noise floors. On `solidify`, `mv_dino_kid`
scores **+0.300, non-monotone, dynamic range 0 noise floors** — its whole
response to progressively deleting every interior is smaller than one draw of
its own sampling noise — against `geom_kid` at +1.000 over 133.

`mixture` is the axis that matters most for ranking, because it is how
generators actually fail: some good builds, some bad. Both metrics are monotone
in the fraction of an arm that is real, which is the direct answer to "does this
benchmark reward better output".

### But the response is quadratic, so read `sqrt`

`MMD²(αP + (1−α)Q, P) = (1−α)²·MMD²(Q,P)`. A squared-MMD metric therefore
responds to the *fraction* of bad samples quadratically, and it does: fitted
against the mixture ladder, the quadratic beats the linear by 14× in absolute
error. **An arm that fails on a quarter of its builds looks 31% as bad as one
that fails on all of them.**

`sqrt(KID)` undoes it — it recovers the true bad fraction to a mean absolute
error of **0.022**, against 0.119 for the raw value. Every squared-MMD metric in
the scorecard now carries `mmd` beside it. Read `mmd` for "what fraction of this
arm is bad"; read the metric for "is this the right distribution".

## Human study — the agreement number a metric paper needs

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
raters compare individual builds — an easy, reliable judgement — and the trials
are pooled to the **arm pair**, which is the level the metric speaks at.

The analysis order is fixed in advance so no step can see the next: exclude
sessions below 75% on real-vs-noise catch trials, fit **Bradley-Terry** arm
strengths (raw win rate depends on which opponents an arm was drawn against, and
coverage is uneven by design), then report pairwise agreement on humanly-decided
pairs and Spearman against the BT scores. Pairs where humans are near chance are
reported but excluded from agreement — scoring a metric on coin flips drives any
metric toward 50%.

The export writes a self-contained page with the arm labels stripped, and the
analysis is tested end to end against simulated raters, including inattentive
ones it must exclude.

## Head-to-head — overlapping intervals are not a tie

Marginal intervals are not a comparison. Two arms are separable when their
difference exceeds `sqrt(h_a² + h_b²)`; non-overlap of their intervals demands
`h_a + h_b`. For equal widths that is 2h against 1.41h — **the overlap heuristic
needs the difference to be 41% larger before it will call anything.**

`compare.rank_table` tests every pair directly, with both arms scored against the
same reference subsample in each replicate, and applies Holm–Bonferroni across
the family (with 7 arms there are 21 tests, and the chance of one spurious
"significant" result at α=0.05 is about two in three). Arms sharing a group
letter were not separated.

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

`real@solidify` shares group **a** with held-out real on the render metric. That
is not a power failure — the same test at the same n separates `train_verbatim`
at a difference of 0.015 and gives it a group of its own. The render tier
resolves 0.015 and cannot resolve the deletion of every interior in the corpus;
on `geom_kid` the same arm is group **f**, Δ = −22.13 [−24.08, −20.72].

Two details that are easy to get wrong, and were:

* **The p-value cannot be a sign-count.** A bootstrap sign-balance p is floored
  at `1/n_rep`; Holm over the 55 pairs of an 11-arm run multiplies 0.005 to
  0.275, so *nothing* is separable however large the difference. The first run
  put held-out real and a monochrome corpus in the same group. The reported `p`
  is studentized (`|Δ| / se`, normal reference) and is not floored; the
  sign-balance value is kept alongside as `p_boot`.
* **The pairing is the correct null, not a power gain.** Measured across four
  reference and arm sizes it moved the interval width by under 10% and not
  consistently in one direction — the reference's contribution to an unbiased
  MMD is a term the difference already cancels analytically. That cancellation
  *is* worth having for a different reason: `distances.kid_arm_term` skips the
  reference self-kernel entirely, which is exact and takes a 66-pair table from
  tens of minutes to about one.

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

**The `context` block *is* the object the arms were scored against.**
`card.context` is `BenchContext.to_json()`, re-serialized once after the last
late-measured value lands on it — not a hand-built dict assembled beside it. The
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
0.015 — second only to real data — and is caught only by `dino_nn_percentile`.

**A metric is judged only on damage it can see.** `RungSpec` says which rungs
gate which family. Gating the occupancy-only geometry metrics on material noise
would fail them for doing exactly what they were built to do, so their noise rung
moves blocks instead of retyping them, and material corruptions join their
*invariance* set — a positive statement of the blindness rather than a hole in
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
failing G7 at 3.5 sd to passing — a metric that separates `solidify` at 68 sd had
been rejected for noise that was mismeasured.

**G8 was testing a difference of means against the spread of a single draw** —
larger by `sqrt(reps)`, so it had almost no power. It let `legacy_cmmd` pass
despite that estimator's null mean halving every time n doubles (0.173, 0.100,
0.058, 0.039 at n = 16/32/64/128). Tested against the standard error of the
difference, the same drift scores **14.8 se** and is rejected. `fd` and
`one_minus_coverage` fail the same way, which is what the earlier verdicts said
for weaker reasons.

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
* `outputs/analysis/ladder/<backbone>__<view>.json` — validation verdicts, one
  file per metric family (`dinov2b__v4_...json`, `geom__patch2_orb55_t5_h8.json`).
  The geometry file's key records the descriptor's shape, so changing the
  descriptor invalidates its ladder instead of silently reusing it.

## Relationship to the older metrics

`eval/novelty.py`, `eval/validity.py` and `eval/perceptual.py` are **not
modified**, so every number in results.md T1–T22 stays reproducible. `bench`
reuses `evaluate_novelty` (pinned bit-for-bit by a test) and scores
`perceptual.cmmd` in the ladder as a labelled legacy metric — where it fails
n-stability, because it is the biased MMD estimator despite its docstring. See
results.md T23.
