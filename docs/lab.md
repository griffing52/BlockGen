# BlockLab: looking at builds, not just scoring them

Every number this project produces comes from a batch script that runs once and
prints a table. That is fine for a result and useless for the two things that
actually consume time: **looking** at builds, and **deciding** about them. T21
left a standing lesson, *look at the renders before writing any number down*,
and until now the only tool for that was rendering a contact sheet and squinting.

BlockLab is the missing half: a small always-on local app over the artifacts the
repo already writes. It reads; the batch pipeline still owns writing.

```bash
python -m tools.lab            # http://127.0.0.1:8765
python -m tools.lab --open     # ...and launch a browser
```

No install step. Standard library only: `http.server` plus `sqlite3` plus what
`blockgen` already imports. FastAPI lives in `deploy/inference/requirements.txt`
and is deliberately *not* used here: a tool meant to be re-run for months should
not acquire a dependency that can rot.

## The eight pages

| page | what it is for |
|---|---|
| **Hub** | the dataset tree, what exists, what came from what, and where to branch |
| **Curate** | keyboard-first triage of a corpus, the main event |
| **Leaderboard** | the cross-run board: one row per model, over every run of one pinned protocol, grouped by `n` |
| **Runs** | a single benchmark run made legible, with drill-down to the actual builds |
| **Compare** | rerunnable 2AFC, feeding the same Bradley-Terry analysis as the study |
| **Gates** | move a curation threshold and *see what it throws away* |
| **Ontology** | the block catalog a model reads, every word next to the number behind it |
| **Phone** (`/phone`) | the one-thumb queue: the server decides what to judge next, the page asks, shows, records |

### Hub: the dataset tree

Datasets nest by **derivation**: a child came out of its parent. Every edge is
read off disk, never guessed from a name.

| edge | where it is recorded |
|---|---|
| `raw:all` → the five raw corpora | `rawcorpora.SPECS` defines the union |
| corpus → `train`/`val`/`test` | `splits.load_split` assigns the rows |
| `outputs/` → run directory → its arms | the arm's `.npz` sits inside the run dir |
| any node → a saved branch | the branch's own rule |

This is not cosmetic. The flat list it replaced showed `raw:all` beside the five
corpora it *is*, and each split beside the corpus it partitions, so its total
counted ~96,000 builds as ~178,000. In the tree a node's builds are counted once,
at the node.

`outputs/` and each run directory are **containers**, they hold no builds of
their own, only children, so they cannot be curated or branched and are not
counted as collections. Every sampling run ever done lives under `outputs/`, and
most are one-build smoke tests; at top level they read as peers of `houses_32`,
which they are not. Inside, `bench_arms` leads because the scorecards cite it,
then run dirs newest-first, plain reverse name order, since they are named
`run_<YYYYMMDD_HHMMSS>_<name>`.

**Provenance is an annotation, not an edge.** `houses_32` draws 1,360 builds
from GrabCraft, 1,267 from 3D-Craft and 34 from text2mc, known exactly, per
row, from its manifest. But that makes it a child of three parents at once, and
a tree edge would force a choice between lying about two of them and duplicating
the node. It renders beside the corpus instead, next to what curation dropped
(`houses_32`: 675 no-interior, 52 single-material blobs, …), because "2,661
kept" only means something against "3,493 pooled".

### Branches (subsets)

**Branch** on any node makes a named subset. A subset is not a copy, it is
`{parent, rule}`, a few hundred bytes in `outputs/lab/subsets/*.json`, resolved
to parent indices on read. Materialising instead would mean a new 50 MB `.npz`
per experiment, and the point of a branch is that you make a dozen while
deciding what the corpus should be. Branches nest: `houses_32 → grabcraft only →
random 256` composes index maps without loading twice.

| filter | resolved from |
|---|---|
| labels / exclude labels | the lab's SQLite store |
| source corpus, category, title | the corpus manifest |
| min/max blocks, fits-in-cube | the corpus manifest |
| random sample + seed | seeded, sorted, reproducible across machines |

Two modes, because they answer different questions:

- **live** re-runs its rule on read, so "everything I labelled good" grows as you
  label. That is what you want while curating.
- **frozen** stores the resolved index list. That is what an experiment cites,
  a live rule is not a reproducible sample, and an eval set that quietly grows
  between two runs makes those runs incomparable.

The dialog refuses to freeze a list you have not previewed, and any edit to a
rule invalidates the preview, so the frozen set is always the one whose size you
were shown.

Rules are validated on save: an unknown filter key is an error rather than a
wider selection. `{"corpsu": ["grabcraft"]}` would otherwise keep all 2,661
builds, look entirely plausible, and never be questioned again.

A field filter on a parent with no manifest is refused above 20,000 builds
rather than opening 59,387 files to read metadata, filter by label, or branch a
smaller parent first.

### Curate

A dataset picker, a label filter, live counts, and a lazy grid. It is
keyboard-first because throughput is the whole point when most of what you are
looking at is bad:

| key | action |
|---|---|
| `←→↑↓` / `hjkl` | move the selection |
| `g` / `b` / `u` | good / bad / unsure (same key again unlabels) |
| `x` | clear |
| `n` | open and focus the note |
| `Enter` / `Esc` | detail drawer with all four views / close |
| `⌘`/`Ctrl`+`Enter` | save note (also saves on blur) |

The drawer has a "read as" switch when the library knows more than one
representation of a build (`blockgen/corpus/reps.py`): `voxels` is the four
orbit renders, `facades` is the six face images of (depth, block), sides in
walking order and then roof plan and underside, darker where a column sits
deeper. The pictures come from `/api/rep/<name>/<build_id>` and share the render
cache, and a representation added to the registry appears here with no change
to the lab.

Labels apply optimistically, the tile updates, then the POST goes out, and a
failure rolls the tile *and* the counters back rather than leaving the screen
lying.

### Gates

Sliders over the nine real `curation.houses.quality_filter` thresholds, with
thumbnails of example builds dropped by each reason. Seeing what a threshold
discards is the thing that makes the decision possible instead of guesswork.

Two details that took measurement to get right:

* **The preview samples by stride, not by prefix.** Corpus caches are written
  corpus-by-corpus, so `houses_32[:400]` is 400/400 GrabCraft, and the
  enclosed-air gate applies only to 3dcraft/text2mc, so `no_interior` (675 of
  826 shipped drops) could never have fired. This is the same contiguity bug
  T23e records for the ladder, in a new place.
* **`min_enclosed_air` is pre-filled at 8, not at `quality_filter`'s default of
  0**, because every shipped cache was built through `build_house_dataset`,
  whose default is 8. Pre-filling the signature default would have shown a gate
  that never fires.

Note only `min_blocks` and `min_enclosed_air` are plumbed through
`build_house_dataset`; the other seven are hard-coded there, and the page says
so rather than emitting a call that would silently ignore them.

### Compare: 2AFC you can re-run

Two builds side by side, pick the better one, keyboard-driven (`←`/`→`, `=` for
a tie, `u` to undo). The point is not a score; it is that the *same* instrument
that runs a human study also runs on you, at any moment, over any two datasets,
so "is this arm actually better" is a question you can answer in five minutes
instead of scheduling.

Arm labels are stripped from the page. That is enforced, not intended: the study
export asserts it by grep, because a condition visible in the DOM is a condition
the rater can read, and a 2AFC with a visible label measures nothing.

Judgements land in `outputs/lab/lab.db` and are exported by
`/api/export?what=compares` into the **same Bradley-Terry fit** the formal study
uses (`scripts/human_study_analyze.py`). A session you run yourself is therefore
directly comparable to a rater's, same estimator, same exclusion rules, which
is what makes a quick sanity check worth anything.

Bradley-Terry rather than raw win rate, for the reason it always is: win rate
depends on which opponents an arm happened to be drawn against, and coverage is
never even when each session sees a random subset.

The `/phone` page (below) posts to the same `/api/compare` route with the same
`tag`, so a pair judged on a phone and a pair judged here are one table in
`lab.db`. The export also carries captions: `/api/export?what=captions` returns
the one-line descriptions typed on either page (`tools/lab/api.py:656-662`).

### Leaderboard: which model is best

The cross-run board. One row per **model**, ranked by BlockScore, over every run
that satisfies a pinned protocol. This is a different question from the one a
scorecard answers, and conflating them was the original sin of the old single
page: the bench measures *arms within one run*, and "which model is best" is a
comparison the bench never makes.

It cannot be made naively, either. BlockScore's unit is a spread of the
`real_test` control **scored in the same run**, so two scores only sit on one
ladder when both runs used the same corpus, split, tier, seed, reference size and
control size. `blockgen/eval/bench/protocol.py` writes that down as a
`Protocol`, and `houses32-v1` is the one the board ranks.

How much that matters is measurable rather than theoretical: `native_oriented`
scores **87.14** against a 128-build `real_test` and **19.05** against a
32-build one, same model, same corpus, same split, same tier. A board that
ignored the calibration arm's size would have shown that as a 4.5× improvement.
This is why `min_n` is checked against the *control* as well as the submission,
and why a run with a small control is excluded rather than quietly averaged in.

**Which cards it reads.** `catalog.SCORECARD_GLOBS` (`tools/lab/catalog.py:673`)
is two patterns: `outputs/*/scorecard.json`, the card a bench run writes at the
top of its directory, and `outputs/*/bench/scorecard.json`, the card the
trainer's end-of-run bench writes under a training run. Both are attributed to
the run directory they sit under, so a nested card is sorted and stamped by
`run_<stamp>_<name>`, never by `bench`. Its id is its directory relative to
`outputs/`, so a nested card is `run_20261014_101448_houses32_scratch_5m/bench`
in the picker, in `?run=` and in the API path, URL-encoded as
`/api/scorecard/run_..._5m%2Fbench` (`catalog.scorecard_run_id`,
`catalog.py:716-739`). Before the second pattern existed every trainer card was
invisible to the board. **A lab process started before 2026-10-30 runs the
one-level glob: restart it to see nested cards, the `n` chips and the
training-run box.**

Five things the page does that are worth knowing before reading it:

- **Runs that do not match are excluded with their reason**, listed under "Runs
  excluded from this board". A board silently covering half the data reads
  exactly like a board covering all of it. A fast-tier card, or one whose
  `real_test` control is under 128 builds, lands here (`houses32-v1` wants tier
  `both` and `min_n` 128, `blockgen/eval/bench/protocol.py:141-154`); the
  trainer's own `bench/` card is fast-tier at `bench_n`, so a training run
  reaches the ranked table only through a separate `--tier both` bench on its
  samples (see [Pipeline](pipeline.md)).
- **Ranked and Provisional are separate tables.** Provisional rows are ordered
  but explicitly not ranked, and each carries the requirement it missed. Today
  that is `train_split` in every case: an arm whose `report.train_split` is not
  `canonical` or `protocol` is provisional with the split named
  (`protocol.CLEAN_TRAIN_SPLITS`, `protocol.py:260`). Lowering the bar to fill a
  board is the failure `composite.py` was written against, and hiding the
  near-misses is unhelpful, so both tables are shown.
- **The board is grouped by `n`, and a mixed view says so.** BlockScore is
  comparable only at equal `n`: a pillar is a bootstrap distance whose spread
  shrinks with the sample, so an n=399 arm and an n=128 arm are not two readings
  on one ruler even inside one protocol (T35). Every row carries its `n`, one
  chip per distinct `n` plus **all** sits above the tables, the page opens on
  `default_n` (the largest `n` with at least two arms,
  `tools/lab/standings.py:262-269`), `?n=` bookmarks a chip, and an amber
  banner covers any view that mixes sample sizes. Grouping never unranks: the
  chips only decide which rows sit side by side, and `n_groups` keeps counting
  every arm so a filter cannot pass for the whole board.
- **A model appearing in several official runs is one row**, at its best score,
  with every appearance listed in the drill-down. Identity is `provenance.model`
  where the card has it, else a *string* `provenance.report.model`, else the arm
  name (`standings.model_id`, `standings.py:78-97`). A dict `report.model`, the
  architecture block a post-training arm writes, describes a model and does not
  name one, so it is shown as `model_label` and never used as identity.
- **It shows what an arm trained on, and flags reward-trained arms.** Each row
  lifts `model`, `checkpoint` (always a path; a `{file, step}` dict from the
  trainer is unwrapped and the step shown as `checkpoint_step`), `train_split`
  and `trained_on_signals` out of `provenance` or `provenance.report`
  (`standings.training`, `standings.py:115-152`). A non-empty
  `trained_on_signals`, the rewards a BoN or RFT stage optimised, earns an amber
  **trained on signals** pill with the signal names in its tooltip and the same
  left border a provisional or oracle row gets. The pill is a flag, not a
  verdict: only `protocol.match_arm` decides ranking, and a model that trained
  against a signal the bench also measures has to be read with that in mind.

The board on 2026-10-31 (`standings.board()` over 74 cards, 38 of them nested)
matches seven runs and ranks arms in three groups: the five facade models at
n=399, the M2b voxel model (`m2b_sft_s1`, best score from the 2026-10-31 post-training
card `run_20261031_031101_bench_voxel_post-training_best-of-4_and`) at n=256,
`canon_oriented` at n=128. The six provisional rows are all `train_split`.

The drill-down strip is labelled according to whether the protocol can actually
align it. With pinned prompts, column *i* is the same request on every row and
the strip compares directly. Without them, every protocol today, the columns
are *slots*: same count, same seed, same order, **different builds**. The page
says so, because a grid that implies a comparison it cannot support is worse than
no grid.

### Runs: what one experiment measured

Reads **every `outputs/*/scorecard.json` and `outputs/*/bench/scorecard.json`**
(`catalog.SCORECARD_GLOBS`), every directory, not `outputs/run_*_bench/`. Two of
the cards on disk (`..._bench_compare`, `..._bench_pnp`) do not match that
narrower pattern and are read anyway, which is the intended behaviour: a run
directory is whatever `--out` said it was. The nested pattern is the trainer's
card, and its id is `<run>/bench`; a `?run=<run>` bookmark to the training run
also resolves it (`catalog.load_scorecard`, `catalog.py:998`). The picker label
shows the card's `n`, and the meta chips repeat it.

Disqualified and unranked arms are pulled out of the numeric ordering and show
their reason, the whole design of BlockScore is that a DQ is a *refusal to
rank*, not a bad score. Gated metrics render as a dash with the failing gate named,
never as a blank that reads like zero, and every metric shows its direction,
because `coherence` is `distance_to_real` and showing it as "higher is better" is
a misreading the suite is explicitly built to prevent.

Nine sections, in the order a run is actually read:

| # | section | what the reader does with it |
|---|---|---|
| 1 | **Run picker** | choose a run: `2026-08-31 22:50 · native_oriented vs agentic_oneshot vs pick_n_place · 3 submissions + 13 reference · houses_32 · both`, date first, counts split into results and instrumentation |
| 2 | **Title + note** | see, in words, what this run was asking |
| 3 | **Meta chips + Re-run** | check corpus, split, n, host, device, sha, and copy the command that reproduces it |
| 4 | **Warnings + compat banner** | find out before reading a number whether to trust it, and what this card is too old to show |
| 5 | **Calibration panel** | confirm the ruler stayed straight: every self-validation check by name, pass or fail |
| 6 | **BlockScore table** | rank the submissions, with the calibration rows folded away |
| 7 | **Drill-down** | click a row: what the arm *is*, where it came from, and what its builds look like |
| 8 | **Per-metric tables** | read the pillar that produced the rank, with intervals and directions |
| 9 | **Head-to-head** | see which differences are actually resolved, a shared letter means *not separated*, not *equal* |

**The picker label is display only, and it is never an id.** Three tiers, most
honest first: `run.name` if the run was named; else its *submission* arm names,
`"a vs b vs c"` and `+N` past the third, because an unnamed run is remembered as
"the one comparing X and Y"; else the run id itself, the all-control cards have
no submissions at all, and inventing a name for them would be worse than showing
the directory. The label collides freely, four cards on disk derive the
identical `native_oriented vs agentic_oneshot vs pick_n_place`, and two more
collide with each other, and that has to stay fine: the `<option value>`, the
`?run=` parameter and the API path segment are all still the directory name, so
bookmarks keep resolving. Making the label unique by construction would quietly
turn it into a second id.

**Ordering is by the `run_<YYYYMMDD_HHMMSS>_` stamp in the directory name, not by
`st_mtime`.** File modification time is not a property of a run: a `git
checkout`, an rsync, a `touch`, or a second writer dropping a file into a run
directory silently reordered the picker, so "newest" was whichever card the
filesystem had been poked at last. Every run directory on disk parses from its
own name, so the ordering costs no JSON reads at all; `st_mtime` remains the
fallback for a
directory that does not follow the convention, which is the one case where it is
genuinely the best guess available.

**The Kind column and the controls toggle.** Every row is `submission`,
`control` or `baseline`, one field, written by the bench and derived in exactly
one place for the cards that predate it. Controls are **hidden by default**, and
this is the one place on the page where something true is off screen, so it is
worth stating why: a real card is sixteen rows of which thirteen are damaged
copies of real builds whose only job is to prove the ruler still bends the right
way. Ranking them beside the three arms someone actually submitted buries the
result under its own instrumentation. Baselines are never hidden, a submission
that loses to `gabled_house` must see that in the same table, and `real_test` is
pinned above the ranked block with a `floor` pill and no rank number, because the
zero point is not a competitor. The count is stated three times (the checkbox
label, the note under the board, the Calibration panel) so nobody can come away
thinking nine rows is all that ran.

**The Calibration panel is what makes hiding them honest.** It lists every
`blockscore_validation` check with a pass/fail pill whether or not any failed,
plus how many controls ran and how many are hidden. The red *"Do not quote this
ranking"* banner above it is unchanged and is not replaced by it: a failure has
to be loud, and a pass has to be visible. On the all-control cards, whose
entire point was calibration, this panel is finally where that run's result
lives.

**Clicking an arm shows what it is, then what it looks like.** The identity block
is never empty: an npz-backed arm shows its `models.json` prose, checkpoint,
sampling settings, a link back to the run that trained it (when that run left a
scorecard of its own), and its `structures_sha`; an in-process arm shows its
builder, the split it was drawn from, its seed, and the one-sentence recipe that
says what the damage actually is. A card too old to carry provenance says exactly
that, rather than showing an empty box. The identity block also lists the arm's
`model`, `checkpoint` (with its step), `train_split` and `trained_on_signals`
when the report carries them, with the same amber pill the board uses.

**Scored by training run.** A nested card sits beside the run that produced it,
and the page joins the three sidecars that run left: `run.json` (the config),
`summary.json` (the registry record) and the matching line of
`outputs/registry.jsonl`, the last line for that run winning
(`catalog.train_run`, `catalog.py:813`). The box names the run, its stage, arch
and preset, steps, parameter count, the train and val tiers, the split, the git
sha, the collapsed `run.json` config and the signals pill. A bench-only run has
no sidecars and no box, which is how the page tells "this card was scored by a
training run" from "this card is a bench run".

Below it, the builds, in three tiers:

1. **This run's own example builds.** From `bench/2` on, the bench writes `k`
   builds of *every* arm, controls included, into one `examples_<max_dim>.npz`
   in the run directory. Captioned, linked into Curate, and labelled *"8 of 64,
   drawn with seed 0, a sample, not a best-of"*. This is the tier that covers
   the in-process arms, which are thirteen of sixteen rows on a real card.
2. **The arm's own npz, browsable as a dataset.** The pre-existing path. The
   `meta.source` → dataset join is still exact string equality, but it now
   happens once in Python (`catalog.source_dataset_id`) and arrives as an id, so
   the page looks it up instead of scanning. Legacy cards with an npz on disk
   land here and behave exactly as they always did; `real_test` falls back to the
   held-out `test` split it was drawn from.
3. **Nothing to browse**, and the two reasons are told apart. An arm whose
   `source` names a file that no dataset resolves to says the npz has moved or
   been deleted; an in-process arm says it was built from the corpus at run time
   and points at the recipe above. Both suggest `--examples 8`.

**Why build ids and not a PNG.** A contact sheet written next to
`scorecard.json` would be a file this server cannot serve: `_static` resolves
only under `tools/lab/static/`, and the only image routes are keyed by
`build_id`. An `.npz` + `_manifest.json` pair, on the other hand, is already a
dataset, discovered by the same glob as every arm, rendered through the same
single-threaded content-addressed cache, openable in Curate. So the bench writes
identities and the lab renders them, lazily; a build that exists both in
`bench_arms/` and in a run's examples renders exactly once, and browsing it in
Curate warms the leaderboard.

### Phone: the one-thumb queue

```bash
python -m tools.lab --phone      # binds every interface, prints http://<lan or tailscale ip>:8765/phone
```

The desktop pages hand you a dataset and a grid. A phone in a waiting room
wants the opposite: open the page, have the next thing already decided, do it
in one tap, come back later to the thing after that. `/phone`
(`tools/lab/static/phone.html`) is that page, and the decisions live on the
server in `tools/lab/phone.py`, driven by `configs/lab/phone.json`.

Three task kinds, served by `GET /api/phone/next?mode=mix|compare|curate|caption&n=6&exclude=<ids,>`:

| kind | what the page asks | what it writes |
|---|---|---|
| `compare` | a forced-choice pair under a tagged question (`more_real`: "which looks more like a real Minecraft house"; `better_build`) | `POST /api/compare` with the tag; the same table the Compare page and the formal study use |
| `curate` | good / bad / unsure on one build of a configured dataset | `POST /api/label` |
| `caption` | one line of text, the way you would ask `/gen` for it | `POST /api/caption` (a new `captions` table in `lab.db`, 500 characters at most) |

`mix` draws `compare` and `curate` at the weights in the config (0.6 / 0.4
today); captions are deliberately not in the mix, because typing is the slow
part on a phone, and have their own mode. `exclude` is the build ids the page
is still holding, queued or inside its undo window (1.8 s, `phone.html`), so
two batches never overlap. Curate walks each dataset in a fixed seeded shuffle
and skips anything already labelled, so progress is simply what is in the
labels table and a session resumes by itself; `caption` prefers builds already
marked `good` and skips `bad` (`prefer_label`, `skip_label`). Pools are
resolved against the catalog at call time and memoized for 45 s
(`phone._POOL_TTL_S`), so a bench run that lands while the server is up joins
the rotation without a restart. `/api/phone/summary` is what drives the mode
chips (a mode with nothing left is shown as done) and `/api/phone/stats?since=`
gives totals and "today" in the rater's local midnight.

**Arms are pinned by id in `configs/lab/phone.json`, never globbed over
`outputs/run_*`.** That directory is full of mock, smoke and pilot runs whose
builds are not worth a human second. The pools today are `split:houses_32:test`
against `arm:bench_arms__*_32` plus three agentic bench arms for the compares,
`corpus:houses_32` and `corpus:houses_48` for curation and captions, and the raw
3D-Craft and GrabCraft corpora for a "usable house build" pass. Add an arm's id
when its bench run is the one being written up; no trainer checkpoint is pinned
yet, so every compare recorded so far is off-policy for the trainer's models.

The page is a PWA (`phone.webmanifest`) so it can live on a home screen, writes
are queued rather than awaited so a tap feels instant on flaky wifi, and
`?mode=caption` opens straight into captions. **There is no auth**: `--phone`
rebinds `--host` to `0.0.0.0` and prints the LAN and Tailscale addresses
(`tools/lab/__main__.py:96-100, 136-143`); anyone on that network can label.

### Ontology

The [block ontology](ontology.md) is the first artifact in the repo that is
*authored by a measurement* and then *read by a model*, which makes it the first
one where "is this any good?" cannot be answered by a scalar. Three questions need
eyes, and the page is laid out around them.

**Is the number right?** Oak stairs are tagged `roof`. That is a threshold applied
to a measured mean height of 0.61, so clicking the row shows the word, the
statistic under it, and the rest of the raw measurements (`anisotropy`,
`exposure`, `support_frac`, run lengths) that decided the other tags. Every column
header carries its provenance, `mined`, `asset`, `authored`, `derived`, because
"measured over 2.78M placements" and "someone typed it" must never look alike.

**Does the colour match the block?** Every row draws the block's real texture tile
next to the hex the catalog measured from those exact pixels. A mismatch is a bug
you can only see. (This is how the foliage-tint bug was caught: oak leaves
measured *grey*, because the vanilla texture ships greyscale and is tinted at draw
time, the catalog now applies the renderer's own multipliers.)

**What is it costing?** The table goes into a system prompt that is re-sent every
round. The **Prompt** tab renders the exact string the agent will send, with its
character and token count, so the bill is a number on screen rather than a guess.

The variant switch is the experiment in miniature: `shuffled` is the token-matched
control with every attribute permuted onto the wrong block. Flipping to it and
finding the table still looks completely plausible is the fastest way to
understand why that control has to exist.

Catalogs are read from `data/ontology/*.json`; `shuffled` and `stats` are derived
on the fly, so there is nothing to rebuild to look at the control. The page is
read-only like the rest of the lab, a tool that could silently rewrite the
ontology an experiment was run against would make old runs unreproducible. Build
one with `python -m blockgen.ontology`.

## What it can and cannot see

Datasets are discovered from **`.npz` caches**, which is what the batch pipeline
writes. On a full checkout that is 48 nodes / 96,103 builds, counted once each:

| kind | what | count |
|---|---|---|
| raw | the uncurated source corpora, indexed lazily | 5, plus their union at 59,387 |
| corpus | every cache in `data/minecraft/cache` | 9, from `houses_32` (2,661) up to `tf_small_24` (5,866) |
| split | the canonical group-aware split | train 1,863 / val 399 / test 399 |
| arm | generated samples under `outputs/` (`catalog.ARM_ROOTS = ("bench_arms", "run_*")`) | 13 arm caches under `outputs/bench_arms/` today, plus every `.npz` + `_manifest.json` pair under a run directory |
| subset | saved branches | whatever you have made |

Corpora sort first and arms of 8 builds or fewer are flagged `smoke run` and
sink to the bottom, discovery order originally buried the nine real corpora
under seventeen one-to-eight-build smoke runs, which made the picker look like
it had found the wrong thing.

A `bench/2` run adds **one** node of its own: the `examples_<max_dim>.npz` it
writes beside its scorecard, noted `bench examples · <run>` and grouped under
that run in the tree. One file per run, not one per arm, a per-arm file would
add a dozen rows to a dataset list that is rebuilt on every call and opens a
zipfile per entry. It is exempt from the `smoke run` flag, because a
single-arm run's eight examples are a benchmark artifact and not a smoke test,
and calling them one would be a lie about the only thing on the page that shows
what a control looks like. In the flat picker it sorts to the *top* of the arm
block rather than the bottom, since that list sorts by size and 128 example rows
outrank a 64-build arm cache: honest, mildly annoying, and left alone, a
retention and grouping policy for these nodes belongs in one pass with
`prerender --kinds` and the kind list in the frozen contract.

**Two cache schemas, and only one reader for them.** `data/minecraft/cache`
holds both, told apart by their keys:

| schema | keys | written by | datasets |
|---|---|---|---|
| house | `block_ids, block_data, corpus, sources` | `curation.houses.save_house_cache` | `houses_*`, `all_*` |
| build | `block_ids, block_data, shapes, urls`\|`paths` | `data.build_cache` | `gc_small_*`, `small_*`, `tf_small_*` |

Neither existing reader covers both: `houses.load_structures_from_cache` raises
`KeyError: 'sources'` on the second, and `build_cache.load_cached_structures` is
hardcoded to `small_{dim}.npz` so it cannot open `gc_small_32.npz` either.
`catalog._read_npz_structures` dispatches on the keys. This mattered: five of
the nine corpora rendered as placeholder tiles, which looked exactly like a
broken renderer and was actually a swallowed `KeyError`.

Titles for the build-cache corpora come from the `*_meta.json` sidecar, keyed by
the same url the npz stores. Without it those are thousands of anonymous tiles.
`small_24` has no sidecar and legitimately has no titles: `data_sources.md`
records that corpus as "none reliable, filenames ≠ metadata".

### The raw corpora

The caches above are the pipeline's **output**, already filtered, cropped and
deduped. Browsing them tells you nothing about what was thrown away, and "I see
a lot of bad builds" is a question about the input. So `rawcorpora` indexes the
**sources** as well:

| dataset | n | what |
|---|---|---|
| `raw:text2mc_schem` | 28,235 | the `.schem` the dataset author never converted |
| `raw:text2mc_h5` | 11,092 | token grids, remapped to legacy `(id,data)` on load |
| `raw:legacy_raw` | 10,963 | the drifted crawl; filenames are not metadata |
| `raw:grabcraft` | 6,560 | category-labeled, exact `(id,data)` |
| `raw:3dcraft` | 2,537 | human build traces, single class |
| **`raw:all`** | **59,387** | every source concatenated |

Nothing here is curated and the junk is all still in it, which is the point.

**Everything is lazy.** The repo's own loaders (`corpora.load_3dcraft`,
`load_text2mc`, `load_grabcraft_structures`) return a fully materialised list,
right for a training run, wrong for a browser, because 59,387 builds will not fit
in memory and nobody wants to wait for all of them to look at twelve. A corpus
here is a *list of paths*, walked once (0.6 s) and cached to
`outputs/lab/index/`, and `LazyBuilds` reads a build only when it is asked for.
Discovery reads the cached index length and never touches a build.

The per-item readers are the repo's own, never reimplemented:
`grabcraft_dataset.structure_from_artifact`, `schem.schem_to_legacy`,
`Structure.from_schematic_path`, `block_remap.remap_token_array`.

Three things worth knowing:

* **text2mc h5 stores its own token ids, not legacy `(id, data)`.** Its grids
  index a 3,717-entry block-state vocabulary; handed to the renderer unremapped
  they paint every build in whatever legacy blocks those integers collide with,
  plausible-looking and completely wrong. `build_token_lut` +
  `remap_token_array` is applied on load, the same path `curation.houses` uses.
  Reading it at all needs `h5py`, which was not installed here.
* **A raw scrape is partly broken, and that is shown rather than hidden.**
  text2mc's own index records 2,700+ entries with zero blocks. An unreadable
  entry becomes an empty structure (so a grid does not crash on `None.shape`)
  tagged `unreadable`, and renders as an empty tile, which is true.
* **`data/minecraft/more` is empty on this machine.** `data_sources.md`
  documents a 36,290-record tfrecord crawl there; the directory exists and holds
  nothing, so that corpus is absent rather than broken, and is not registered.

Still not indexed: `data/lego` (1.2 GB) and `data/minecraftace` (4 GB), different
media, and each needs its own reader.

### Overlap: the counts double-count, heavily

The nine caches plus the three splits list **39,016 rows but hold 16,734
distinct builds**: 57% is repetition. `split:houses_32:*` is exactly
`houses_32` re-partitioned; `all_32` contains `houses_32`; `gc_small_32` is the
GrabCraft slice of both. Read the per-dataset counts as "what this set contains",
never as "how much data we have".

`raw:all` is a concatenation and is **not deduped** either, text2mc and the
legacy crawl both scraped PlanetMinecraft, so they overlap by construction, and
`data_sources.md` warns to dedup before pooling. To measure it:

```bash
python -m tools.lab.rawcorpora --hash --report   # one pass, ~20 min for 59k
```

That caches a content hash per build (blocks and data, cropped, metadata
ignored, so two records of one build with different scraped titles are one
build) and prints per-corpus distinct counts plus pairwise sharing.

## Running it: the CLI

```bash
python -m tools.lab                     # http://127.0.0.1:8765
python -m tools.lab --port 8790         # somewhere else
python -m tools.lab --port 0            # any free port, chosen by the OS
python -m tools.lab --open              # ...and launch a browser
python -m tools.lab --host 0.0.0.0      # see the warning below
python -m tools.lab --phone             # 0.0.0.0 plus the /phone URL for LAN and Tailscale
```

`--host` defaults to loopback and should usually stay there: **there is no
authentication on this thing**, and it will happily serve, label and delete
subsets for anyone who can reach the port.

### Why the port is fixed

8765 is a constant, not an auto-assigned port, because it is quoted in places
that cannot follow it: the bench prints `view: http://127.0.0.1:8765/...` at the
end of every run (`runner.LAB_PORT`), and this documentation prints it too. A
port that moved per launch would falsify a link the reader is being told to
click, and would break every bookmark.

So "port in use" almost always means *you already have BlockLab open*, not *pick
another port*, and it is treated that way:

```
$ python -m tools.lab
[lab] BlockLab is already running on http://127.0.0.1:8765/, using that one (nothing new started).
[lab] to run a second instance anyway: python -m tools.lab --port 0
```

That exits 0, starts nothing, and honours `--open` against the server that is
already there. It only says this when the thing holding the port answers
`/api/datasets` as a BlockLab; anything else gets the plain bind error and a
non-zero exit.

`--port 0` is the escape hatch when you genuinely want a second instance, a
throwaway one against a different checkout, say. Both instances share
`outputs/lab/lab.db`, which SQLite handles, but two windows labelling the same
corpus is rarely what anyone means to do.

If `--open` prints `could not open a browser`, the server is still up, open the
printed URL yourself. In a VS Code terminal `BROWSER` points at a helper that
opens URLs over a Unix socket, tmux panes inherit that variable, and the socket
dies with the window that created it; `BROWSER= python -m tools.lab --open` uses
the normal handler instead.

Two companion commands do batch work the server deliberately does not:

```bash
# Warm the thumbnail cache so a page is instant on first open rather than
# rendering 60 tiles on one thread while you wait.
python -m tools.lab.prerender --datasets arm:<id> --px 96 --workers 8

# Index the raw corpora (expensive, one pass, cached under outputs/lab/index/).
python -m tools.lab.rawcorpora --hash --report
```

## The HTTP surface

Small and flat, and the module docstring of `tools/lab/__init__.py` is the
contract of record. Everything returns JSON except the image routes.

| method | route | returns |
|---|---|---|
| GET | `/api/datasets` | every browsable dataset: arms, corpora, splits, raw, subsets |
| GET | `/api/tree` | the same, arranged as the hub's provenance tree |
| GET | `/api/builds?dataset=&limit=` | a page of builds with their stats |
| GET | `/api/build/<build_id>` | one build: features, geometry, metadata |
| GET | `/api/thumb/<build_id>?px=` | one rendered thumbnail (JPEG) |
| GET | `/api/view/<build_id>/<k>` | orbit view *k*, the four angles the metrics embed |
| GET | `/api/scorecards` | every benchmark run, newest first; rows carry `run` (`<run>` or `<run>/bench`), `run_dir`, `n`, `train_run` |
| GET | `/api/scorecard/<run>` | one card, plus `arms_index`, `compat`, `run_dir`, `train_run`; a nested id is URL-encoded, `run_..._5m%2Fbench` |
| GET | `/api/standings?protocol=&n=` | the cross-run board for one protocol; `n=` keeps only the arms scored at that integer sample size (anything else is 400) and the reply carries `n_groups`, `default_n`, `n_filter`, `n_hidden`, `mixed_n`; rows carry `n`, `model_label`, `checkpoint`, `checkpoint_step`, `train_split`, `trained_on_signals` |
| GET | `/api/protocols` | the pinned protocols, and which is default |
| GET | `/api/labels` | labels recorded so far |
| GET | `/api/subsets` | saved branches |
| GET | `/api/curation/preview` | what a set of gate thresholds would drop |
| GET | `/api/ontology`, `/ontology/catalogs`, `/ontology/prompt` | the block catalog |
| GET | `/api/ontology/part/<id>`, `/ontology/swatch/<id>` | one entry, one colour chip |
| GET | `/api/phone/next?mode=&n=&exclude=` | the next tasks for the phone page |
| GET | `/api/phone/summary`, `/api/phone/stats?since=` | what the queue has left; totals and today's count |
| GET | `/api/export?what=labels\|notes\|captions\|compares` | the seam out to the batch pipeline |
| POST | `/api/label`, `/api/note`, `/api/compare`, `/api/caption` | record a decision |
| POST | `/api/subset`, `/api/subset/preview`, `/api/subset/delete` | manage branches |

A build is addressed by `build_id` = `"<dataset_id>:<row>"`, and dataset ids
legitimately contain colons (`split:houses_32:test`), so the id is split from the
**right**. These ids are the primary key of `outputs/lab/lab.db`: a change in how
they are derived orphans every label, note and comparison ever recorded, which is
why `tests/lab/test_arm_ids.py` pins them literally.

## What it writes

The lab never writes an artifact it did not create. Everything it owns is under
`outputs/lab/`:

| path | what |
|---|---|
| `outputs/lab/lab.db` | SQLite: labels, notes, captions, 2AFC comparisons |
| `outputs/lab/index/` | cached raw-corpus indexes (`rawcorpora`) |
| `outputs/lab/thumbs/` | the content-addressed render cache |
| `outputs/lab/subsets/` | saved branches, one JSON per subset |

Indexes and thumbnails rebuild themselves, so deleting those is free. `lab.db`
and `subsets/` hold decisions you cannot regenerate, back those up if you have
been labelling.

## Design rules

**Decisions are proposals, never writes.** Labels and notes live in
`outputs/lab/lab.db` and are never written back into a corpus cache. Turning a
curation decision into a dataset stays the batch pipeline's job;
`/api/export?what=labels|notes|captions|compares` is the seam.

**Renders are serialized and cached.** `renderer.textured` holds a single EGL
context and is not thread-safe: `notes.md` records a context clash from mixing
pyrender backends in one process. Every render goes through one lock into a
content-addressed disk cache, so a thumbnail is computed once ever (cold ~530
ms, warm 0.06 ms) and the same build reached through two datasets shares one
entry. A failed render returns a *striped* placeholder, not a flat tile, and is
not cached, a flat tile is precisely what a dead renderer emits (T25c).

**It lives outside `blockgen/`.** This is an instrument, not part of the library
under test; nothing in `blockgen/` may import it. It sits in `tools/` rather
than `deploy/`, which is the shipped Minecraft demo with its own dependency file
and release cadence.

The full HTTP contract is the module docstring of `tools/lab/__init__.py`.

## Feeding the human study

Comparisons made in the Compare page, or on the phone, land in the same
Bradley-Terry fit as the batch 2AFC study:

```bash
python scripts/human_study_analyze.py --lab-db outputs/lab/lab.db
```

The lab log is the cleaner of the two instruments: it records the winning
`build_id`, and a build id carries its dataset, so there is no answer key to join
against and nothing that can drift. What it does *not* have is catch trials, a
self-collected log has no guard against an inattentive rater, and the analysis
says so out loud rather than quietly reporting the same confidence.
