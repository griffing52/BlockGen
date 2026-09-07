# BlockLab — looking at builds, not just scoring them

Every number this project produces comes from a batch script that runs once and
prints a table. That is fine for a result and useless for the two things that
actually consume time: **looking** at builds, and **deciding** about them. T21
left a standing lesson — *look at the renders before writing any number down* —
and until now the only tool for that was rendering a contact sheet and squinting.

BlockLab is the missing half: a small always-on local app over the artifacts the
repo already writes. It reads; the batch pipeline still owns writing.

```bash
python -m tools.lab            # http://127.0.0.1:8765
python -m tools.lab --open     # ...and launch a browser
```

No install step. Standard library only — `http.server` plus `sqlite3` plus what
`blockgen` already imports. FastAPI lives in `deploy/inference/requirements.txt`
and is deliberately *not* used here: a tool meant to be re-run for months should
not acquire a dependency that can rot.

## The seven pages

| page | what it is for |
|---|---|
| **Hub** | the dataset tree — what exists, what came from what, and where to branch |
| **Curate** | keyboard-first triage of a corpus — the main event |
| **Leaderboard** | the cross-run board: one row per model, over every run of one pinned protocol |
| **Runs** | a single benchmark run made legible, with drill-down to the actual builds |
| **Compare** | rerunnable 2AFC, feeding the same Bradley-Terry analysis as the study |
| **Gates** | move a curation threshold and *see what it throws away* |
| **Ontology** | the block catalog a model reads — every word next to the number behind it |

### Hub — the dataset tree

Datasets nest by **derivation**: a child came out of its parent. Every edge is
read off disk, never guessed from a name.

| edge | where it is recorded |
|---|---|
| `raw:all` → the five raw corpora | `rawcorpora.SPECS` defines the union |
| corpus → `train`/`val`/`test` | `splits.load_split` assigns the rows |
| run directory → its arms | the arm's `.npz` sits inside the run dir |
| any node → a saved branch | the branch's own rule |

This is not cosmetic. The flat list it replaced showed `raw:all` beside the five
corpora it *is*, and each split beside the corpus it partitions, so its total
counted ~96,000 builds as ~178,000. In the tree a node's builds are counted once,
at the node.

**Provenance is an annotation, not an edge.** `houses_32` draws 1,360 builds
from GrabCraft, 1,267 from 3D-Craft and 34 from text2mc — known exactly, per
row, from its manifest. But that makes it a child of three parents at once, and
a tree edge would force a choice between lying about two of them and duplicating
the node. It renders beside the corpus instead, next to what curation dropped
(`houses_32`: 675 no-interior, 52 single-material blobs, …), because "2,661
kept" only means something against "3,493 pooled".

### Branches (subsets)

**Branch** on any node makes a named subset. A subset is not a copy — it is
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
- **frozen** stores the resolved index list. That is what an experiment cites —
  a live rule is not a reproducible sample, and an eval set that quietly grows
  between two runs makes those runs incomparable.

The dialog refuses to freeze a list you have not previewed, and any edit to a
rule invalidates the preview, so the frozen set is always the one whose size you
were shown.

Rules are validated on save: an unknown filter key is an error rather than a
wider selection. `{"corpsu": ["grabcraft"]}` would otherwise keep all 2,661
builds, look entirely plausible, and never be questioned again.

A field filter on a parent with no manifest is refused above 20,000 builds
rather than opening 59,387 files to read metadata — filter by label, or branch a
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

Labels apply optimistically — the tile updates, then the POST goes out, and a
failure rolls the tile *and* the counters back rather than leaving the screen
lying.

### Gates

Sliders over the nine real `curation.houses.quality_filter` thresholds, with
thumbnails of example builds dropped by each reason. Seeing what a threshold
discards is the thing that makes the decision possible instead of guesswork.

Two details that took measurement to get right:

* **The preview samples by stride, not by prefix.** Corpus caches are written
  corpus-by-corpus, so `houses_32[:400]` is 400/400 GrabCraft — and the
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

### Leaderboard — which model is best

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
32-build one — same model, same corpus, same split, same tier. A board that
ignored the calibration arm's size would have shown that as a 4.5× improvement.
This is why `min_n` is checked against the *control* as well as the submission,
and why a run with a small control is excluded rather than quietly averaged in.

Three things the page does that are worth knowing before reading it:

- **Runs that do not match are excluded with their reason**, listed under "Runs
  excluded from this board". A board silently covering half the data reads
  exactly like a board covering all of it.
- **Ranked and Provisional are separate tables.** Every submission on disk today
  is below `houses32-v1`'s `min_n` of 128 (n=64, 16, 12), so ranking only what
  qualifies gives an empty board and lowering `min_n` to fill it gives a board
  whose intervals mean nothing. Provisional rows are ordered but explicitly not
  ranked, and each carries the requirement it missed.
- **A model appearing in several official runs is one row**, at its best score,
  with every appearance listed in the drill-down. Identity is `provenance.model`
  where the card has it, so the same checkpoint under two `--arms` labels
  collapses correctly.

The drill-down strip is labelled according to whether the protocol can actually
align it. With pinned prompts, column *i* is the same request on every row and
the strip compares directly. Without them — every protocol today — the columns
are *slots*: same count, same seed, same order, **different builds**. The page
says so, because a grid that implies a comparison it cannot support is worse than
no grid.

### Runs — what one experiment measured

Reads **every `outputs/*/scorecard.json`** — every directory, not
`outputs/run_*_bench/`. Two of the cards on disk (`..._bench_compare`,
`..._bench_pnp`) do not match that narrower pattern and are read anyway, which is
the intended behaviour: a run directory is whatever `--out` said it was.

Disqualified and unranked arms are pulled out of the numeric ordering and show
their reason — the whole design of BlockScore is that a DQ is a *refusal to
rank*, not a bad score. Gated metrics render as "—" with the failing gate named,
never as a blank that reads like zero, and every metric shows its direction,
because `coherence` is `distance_to_real` and showing it as "higher is better" is
a misreading the suite is explicitly built to prevent.

Nine sections, in the order a run is actually read:

| # | section | what the reader does with it |
|---|---|---|
| 1 | **Run picker** | choose a run: `2026-08-31 22:50 · native_oriented vs agentic_oneshot vs pick_n_place · 3 submissions + 13 reference · houses_32 · both` — date first, counts split into results and instrumentation |
| 2 | **Title + note** | see, in words, what this run was asking |
| 3 | **Meta chips + Re-run** | check corpus, split, n, host, device, sha — and copy the command that reproduces it |
| 4 | **Warnings + compat banner** | find out before reading a number whether to trust it, and what this card is too old to show |
| 5 | **Calibration panel** | confirm the ruler stayed straight: every self-validation check by name, pass or fail |
| 6 | **BlockScore table** | rank the submissions, with the calibration rows folded away |
| 7 | **Drill-down** | click a row: what the arm *is*, where it came from, and what its builds look like |
| 8 | **Per-metric tables** | read the pillar that produced the rank, with intervals and directions |
| 9 | **Head-to-head** | see which differences are actually resolved — a shared letter means *not separated*, not *equal* |

**The picker label is display only, and it is never an id.** Three tiers, most
honest first: `run.name` if the run was named; else its *submission* arm names,
`"a vs b vs c"` and `+N` past the third, because an unnamed run is remembered as
"the one comparing X and Y"; else the run id itself — the all-control cards have
no submissions at all, and inventing a name for them would be worse than showing
the directory. The label collides freely — four cards on disk derive the
identical `native_oriented vs agentic_oneshot vs pick_n_place`, and two more
collide with each other — and that has to stay fine: the `<option value>`, the
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
`control` or `baseline` — one field, written by the bench and derived in exactly
one place for the cards that predate it. Controls are **hidden by default**, and
this is the one place on the page where something true is off screen, so it is
worth stating why: a real card is sixteen rows of which thirteen are damaged
copies of real builds whose only job is to prove the ruler still bends the right
way. Ranking them beside the three arms someone actually submitted buries the
result under its own instrumentation. Baselines are never hidden — a submission
that loses to `gabled_house` must see that in the same table — and `real_test` is
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
that, rather than showing an empty box.

Below it, the builds, in three tiers:

1. **This run's own example builds.** From `bench/2` on, the bench writes `k`
   builds of *every* arm — controls included — into one `examples_<max_dim>.npz`
   in the run directory. Captioned, linked into Curate, and labelled *"8 of 64,
   drawn with seed 0 — a sample, not a best-of"*. This is the tier that covers
   the in-process arms, which are thirteen of sixteen rows on a real card.
2. **The arm's own npz, browsable as a dataset.** The pre-existing path. The
   `meta.source` → dataset join is still exact string equality, but it now
   happens once in Python (`catalog.source_dataset_id`) and arrives as an id, so
   the page looks it up instead of scanning. Legacy cards with an npz on disk
   land here and behave exactly as they always did; `real_test` falls back to the
   held-out `test` split it was drawn from.
3. **Nothing to browse** — and the two reasons are told apart. An arm whose
   `source` names a file that no dataset resolves to says the npz has moved or
   been deleted; an in-process arm says it was built from the corpus at run time
   and points at the recipe above. Both suggest `--examples 8`.

**Why build ids and not a PNG.** A contact sheet written next to
`scorecard.json` would be a file this server cannot serve: `_static` resolves
only under `tools/lab/static/`, and the only image routes are keyed by
`build_id`. An `.npz` + `_manifest.json` pair, on the other hand, is already a
dataset — discovered by the same glob as every arm, rendered through the same
single-threaded content-addressed cache, openable in Curate. So the bench writes
identities and the lab renders them, lazily; a build that exists both in
`bench_arms/` and in a run's examples renders exactly once, and browsing it in
Curate warms the leaderboard.

### Ontology

The [block ontology](ontology.md) is the first artifact in the repo that is
*authored by a measurement* and then *read by a model*, which makes it the first
one where "is this any good?" cannot be answered by a scalar. Three questions need
eyes, and the page is laid out around them.

**Is the number right?** Oak stairs are tagged `roof`. That is a threshold applied
to a measured mean height of 0.61 — so clicking the row shows the word, the
statistic under it, and the rest of the raw measurements (`anisotropy`,
`exposure`, `support_frac`, run lengths) that decided the other tags. Every column
header carries its provenance — `mined`, `asset`, `authored`, `derived` — because
"measured over 2.78M placements" and "someone typed it" must never look alike.

**Does the colour match the block?** Every row draws the block's real texture tile
next to the hex the catalog measured from those exact pixels. A mismatch is a bug
you can only see. (This is how the foliage-tint bug was caught: oak leaves
measured *grey*, because the vanilla texture ships greyscale and is tinted at draw
time — the catalog now applies the renderer's own multipliers.)

**What is it costing?** The table goes into a system prompt that is re-sent every
round. The **Prompt** tab renders the exact string the agent will send, with its
character and token count, so the bill is a number on screen rather than a guess.

The variant switch is the experiment in miniature: `shuffled` is the token-matched
control with every attribute permuted onto the wrong block. Flipping to it and
finding the table still looks completely plausible is the fastest way to
understand why that control has to exist.

Catalogs are read from `data/ontology/*.json`; `shuffled` and `stats` are derived
on the fly, so there is nothing to rebuild to look at the control. The page is
read-only like the rest of the lab — a tool that could silently rewrite the
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
| arm | generated samples under `outputs/` | 3 benchmark arms plus 14 smoke runs |
| subset | saved branches | whatever you have made |

Corpora sort first and arms of 8 builds or fewer are flagged `smoke run` and
sink to the bottom — discovery order originally buried the nine real corpora
under seventeen one-to-eight-build smoke runs, which made the picker look like
it had found the wrong thing.

A `bench/2` run adds **one** node of its own: the `examples_<max_dim>.npz` it
writes beside its scorecard, noted `bench examples · <run>` and grouped under
that run in the tree. One file per run, not one per arm — a per-arm file would
add a dozen rows to a dataset list that is rebuilt on every call and opens a
zipfile per entry. It is exempt from the `smoke run` flag, because a
single-arm run's eight examples are a benchmark artifact and not a smoke test,
and calling them one would be a lie about the only thing on the page that shows
what a control looks like. In the flat picker it sorts to the *top* of the arm
block rather than the bottom, since that list sorts by size and 128 example rows
outrank a 64-build arm cache: honest, mildly annoying, and left alone — a
retention and grouping policy for these nodes belongs in one pass with
`prerender --kinds` and the kind list in the frozen contract.

**Two cache schemas, and only one reader for them.** `data/minecraft/cache`
holds both, told apart by their keys:

| schema | keys | written by | datasets |
|---|---|---|---|
| house | `block_ids, block_data, corpus, sources` | `curation.houses.save_house_cache` | `houses_*`, `all_*` |
| build | `block_ids, block_data, shapes, urls`\|`paths` | `data.build_cache` | `gc_small_*`, `small_*`, `tf_small_*` |

Neither existing reader covers both — `houses.load_structures_from_cache` raises
`KeyError: 'sources'` on the second, and `build_cache.load_cached_structures` is
hardcoded to `small_{dim}.npz` so it cannot open `gc_small_32.npz` either.
`catalog._read_npz_structures` dispatches on the keys. This mattered: five of
the nine corpora rendered as placeholder tiles, which looked exactly like a
broken renderer and was actually a swallowed `KeyError`.

Titles for the build-cache corpora come from the `*_meta.json` sidecar, keyed by
the same url the npz stores. Without it those are thousands of anonymous tiles.
`small_24` has no sidecar and legitimately has no titles — `data_sources.md`
records that corpus as "none reliable — filenames ≠ metadata".

### The raw corpora

The caches above are the pipeline's **output** — already filtered, cropped and
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
`load_text2mc`, `load_grabcraft_structures`) return a fully materialised list —
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
  they paint every build in whatever legacy blocks those integers collide with —
  plausible-looking and completely wrong. `build_token_lut` +
  `remap_token_array` is applied on load, the same path `curation.houses` uses.
  Reading it at all needs `h5py`, which was not installed here.
* **A raw scrape is partly broken, and that is shown rather than hidden.**
  text2mc's own index records 2,700+ entries with zero blocks. An unreadable
  entry becomes an empty structure (so a grid does not crash on `None.shape`)
  tagged `unreadable`, and renders as an empty tile — which is true.
* **`data/minecraft/more` is empty on this machine.** `data_sources.md`
  documents a 36,290-record tfrecord crawl there; the directory exists and holds
  nothing, so that corpus is absent rather than broken, and is not registered.

Still not indexed: `data/lego` (1.2 GB) and `data/minecraftace` (4 GB) — different
media, and each needs its own reader.

### Overlap: the counts double-count, heavily

The nine caches plus the three splits list **39,016 rows but hold 16,734
distinct builds** — 57% is repetition. `split:houses_32:*` is exactly
`houses_32` re-partitioned; `all_32` contains `houses_32`; `gc_small_32` is the
GrabCraft slice of both. Read the per-dataset counts as "what this set contains",
never as "how much data we have".

`raw:all` is a concatenation and is **not deduped** either — text2mc and the
legacy crawl both scraped PlanetMinecraft, so they overlap by construction, and
`data_sources.md` warns to dedup before pooling. To measure it:

```bash
python -m tools.lab.rawcorpora --hash --report   # one pass, ~20 min for 59k
```

That caches a content hash per build (blocks and data, cropped — metadata
ignored, so two records of one build with different scraped titles are one
build) and prints per-corpus distinct counts plus pairwise sharing.

## Design rules

**Decisions are proposals, never writes.** Labels and notes live in
`outputs/lab/lab.db` and are never written back into a corpus cache. Turning a
curation decision into a dataset stays the batch pipeline's job;
`/api/export?what=labels|notes|compares` is the seam.

**Renders are serialized and cached.** `renderer.textured` holds a single EGL
context and is not thread-safe — `notes.md` records a context clash from mixing
pyrender backends in one process. Every render goes through one lock into a
content-addressed disk cache, so a thumbnail is computed once ever (cold ~530
ms, warm 0.06 ms) and the same build reached through two datasets shares one
entry. A failed render returns a *striped* placeholder, not a flat tile, and is
not cached — a flat tile is precisely what a dead renderer emits (T25c).

**It lives outside `blockgen/`.** This is an instrument, not part of the library
under test; nothing in `blockgen/` may import it. It sits in `tools/` rather
than `deploy/`, which is the shipped Minecraft demo with its own dependency file
and release cadence.

The full HTTP contract is the module docstring of `tools/lab/__init__.py`.

## Feeding the human study

Comparisons made in the Compare page land in the same Bradley-Terry fit as the
batch 2AFC study:

```bash
python scripts/human_study_analyze.py --lab-db outputs/lab/lab.db
```

The lab log is the cleaner of the two instruments: it records the winning
`build_id`, and a build id carries its dataset, so there is no answer key to join
against and nothing that can drift. What it does *not* have is catch trials — a
self-collected log has no guard against an inattentive rater, and the analysis
says so out loud rather than quietly reporting the same confidence.
