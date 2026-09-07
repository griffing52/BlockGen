# BlockGen — Results

Tables, figures, and ablations backing the paper. Methods/repro live in
[`notes.md`](notes.md). Each result notes **which cache / subset** it ran on and
**when**, because the dataset is being migrated (legacy `data/raw` → labeled
tfrecord cache) and numbers are not comparable across caches.

_Last updated: 2026-09-07._

Legend: **Legacy** = `small_24.npz` (from `data/raw`, no metadata). **Labeled** =
`tf_small_24.npz` (from tfrecords, 100% metadata).

---

## T1. Dataset & cache statistics

| Cache | Source | Scanned | Kept | Skipped bad | Too big (>24) | Too small | Metadata |
|---|---|--:|--:|--:|--:|--:|--:|
| Legacy `small_24` | `data/raw/*.schematic` | 10,963 | 2,892 | 184 | — | — | 0% |
| **Labeled `tf_small_24`** | tfrecords (36,290 rec) | 36,290 | **5,866** | 2,688 | 27,433 | 303 | **100%** |

Filters: cropped to non-air bbox, `max_dim ≤ 24`, `8 ≤ occupied ≤ 4096`.
Note the dominant skip is "too big" — the 24³ cap keeps only compact builds.

## T2. Category distribution (Labeled, 5,866 structures)

| Category | n | | Category | n |
|---|--:|---|---|--:|
| Land Structure Map | 1,723 | | Piston Map | 120 |
| Redstone Device Map | 1,287 | | Environment / Landscaping | 101 |
| 3D Art Map | 1,264 | | Water Structure Map | 78 |
| Other Map | 477 | | Challenge / Adventure | 51 |
| Air Structure Map | 427 | | Minecart Map | 50 |
| Complex Map | 126 | | Underground Structure | 19 |
| Pixel Art Map | 125 | | Nether / Music | 9 / 9 |

Top dominant materials (557-slice sample): wool, stone, planks, dirt, cobblestone, grass, log, sand.

## T3. Curation outcomes (Labeled, 5,866)

| Quantity | Value | Notes |
|---|--:|---|
| Buildable "houses" subset | **714** | `search("house")` ∩ structure categories ∩ `min_blocks≥60, comps≤3, types≥3` |
| Exact-duplicate groups | 27 | same shape **and** palette; 29 droppable extras |
| **Material-variant groups (KEEP)** | **58** | same shape, different materials — preserved, not dropped |
| Reliable seed set (popularity) | 1,532 | ≥10 diamonds **or** ≥100 downloads |

→ Figure F1.

## T3b. Unified curated HOUSE dataset (2026-07-06, `blockgen/curation/houses.py`)

Cross-corpus pool → shared legacy vocab (text2mc remapped via `utils/block_remap.py`,
97% state coverage; ground-stripping for world cuts) → quality gate → variant-aware dedup.

| Cache | Pooled | Quality drops | Exact dups | **Final** | GrabCraft | 3D-Craft | text2mc |
|---|--:|--:|--:|--:|--:|--:|--:|
| `houses_32` | 3,493 | 826 | 6 | **2,661** | 1,360 | 1,267 | 34 |
| `houses_48` | 4,154 | 1,159 | 24 | **2,971** | 1,410 | 1,381 | 180 |

Drop reasons (32³): **no-interior 675** (enclosed-air gate, 3dcraft/text2mc only),
monotype blob 52, fragmented 37, too flat 29, <3 materials 22, too sparse 9.
The enclosed-air gate (≥8 interior air voxels unreachable from the bbox boundary)
was calibrated on the corpora: 17% of *ground-truth* GrabCraft houses have open
interiors (so GrabCraft is exempt), vs 31% of 3D-Craft and 54% of text2mc — and a
pre-gate visual grid showed exactly that junk (trees, a truck, roof fragments).
Load: `curation.houses.load_house_structures(max_dim)`.
Compare to the old labeled-cache `houses` subset (714): **~3.7–4.2× more clean houses.**

---

## T29. Injecting geometry into attention — a null, and the reason is in the parameterization (2026-09-07)

The complementary experiment to T28: instead of reading adjacency out of attention, put
it in. `scripts/train_llm_geobias.py` holds Track D v2 exactly fixed — same 3D-BPE piece
vocabulary, same `"<piece_name> <x> <y> <z>"` lines, same LoRA (r32/α16 on q_proj,v_proj),
same optimizer/schedule, same seed, same 2048-token filter (it rebuilds the identical
**297/2661** split, train 267 / val 30) — and adds, in the `bias` arm only, a learned
additive attention bias indexed by the relative 3D offset between the pieces two token
positions refer to. Zero-initialized, so at step 0 the arms are numerically identical;
injected through a prepared `{"full_attention": ...}` mask, so it is shared across all
28 layers and 12 heads.

Causality is the binding constraint: a line's anchor is unknown until its coordinates are
emitted, so the query-side reference is the anchor of the last **completed** line, and
keys inside the query's own line get a separate learned scalar. Verified numerically that
the same-line block is exactly that scalar (no coordinate leaks backward), that the
dict-mask path reproduces HF's default causal mask bit-for-bit, and that the zero-init
arm equals the control.

| | control (= Track D v2) | + geometric bias |
|---|--:|--:|
| best val loss | 0.3683 @ep8 | **0.3662** @ep8 |
| final val loss (ep20) | 0.4090 | 0.4048 |
| mean parse rate | 0.903 | **0.994** |
| degenerate samples (flat or stick) | 4/8 | 3/8 |
| median blocks / sample | 169 | 161 |
| real held-out builds, same measures | 0/30 degenerate | median 699 blocks |

The control's 0.409 final val loss reproduces Track D v2's reported 0.41, so this is a
faithful re-run and not a different experiment wearing the same name.

**Verdict: null.** −0.0021 best val loss is not a result at one seed, the parse-rate gain
lands in a metric that was already 0.99 in the original run, and the collapse mode is
untouched: 3/8 vs 4/8 samples are still a flat slab or a stick against 0/30 for real
builds, and the median sample is a quarter the size of a real house. n=8 samples cannot
separate 3/8 from 4/8.

**The diagnostic is worth more than the number.** Reading the learned table back
(`geobias_table.png`) shows **364 of 729 offset buckets were never updated** — exactly
zero. The reference point is the *previous* line's anchor and lines are emitted in
`(y,z,x)` raster order, so `dy ≥ 0` always, and `dz`/`dx` are sign-constrained whenever
the higher axes tie. Three of the six 6-adjacent offsets — `-x`, `-y`, `-z` — are
**unreachable by construction**:

| offset | learned bias |
|---|--:|
| `+x` | **+0.146** |
| `+y` | −0.096 |
| `+z` | −0.106 |
| `-x`, `-y`, `-z` | +0.000 (never visited) |
| zero offset | +0.101 |
| all other reachable buckets (mean) | −0.168 |
| specials | prompt −0.180 · same-line +0.104 · no-ref +0.173 |

So the arm's whole learned signal is "attend to the piece one step back along the raster
run" — a reparameterization of recency, which T28 already showed is the weak predictor.
The bias never got the chance to express neighbourhood.

**The fix this implies, and the next run.** The query reference has to be the *current*
piece's anchor, not the previous one's, which requires emitting coordinates before the
piece name (`"<x> <y> <z> <piece_name>"`) so the anchor is causally available at the
moment the piece type is predicted. That change also makes all 729 buckets reachable. It
needs its own matched control on the reordered format, since the serialization changes.

Secondary caveat, stated in the script: the bias is shared across all layers and heads,
so this null bounds a **shared-scalar** bias, not per-head structure.

Figures: `geobias_loss.png`, `geobias_table.png`, and `samples_textured.png` per arm in
`outputs/run_20260906_222749_llm_geobias_control/` and
`outputs/run_20260906_232253_llm_geobias_bias/`.

## T28. "Attention as an adjacency matrix" — measured, and dominated by a subtraction (2026-09-06)

The proposal: skip training a structure model, run a pretrained LLM over a serialized
build, read its self-attention `A[i, j]` as the probability that piece `i` connects to
piece `j`, and get the structural prior for free. `scripts/probe_llm_attention.py` tests
it directly — for every held-out house the true answer is known, so each of the
`28 x 12 = 336` heads is scored by how well it ranks the truly 6-adjacent previous
pieces above the non-adjacent ones (AUC over the causal candidate set `j < i`, which is
exactly the pointer-network question "which placed piece does this one attach to").

Setup: Track D's piece serialization and split, held fixed. 120 held-out builds — the
30 Track D val builds plus 90 the finetune never saw at all — median 171 pieces, 2.89%
of candidate pairs are true edges. Best head chosen on a train half of the builds,
reported on the test half (n=60). Adapter = `run_20260722_072022_llm_pieces`.

| predictor | AUC ↑ | distance-stratified AUC ↑ |
|---|--:|--:|
| chance | 0.500 | 0.500 |
| recency `-(i-j)` | 0.831 | 0.557 |
| base Qwen2.5-Coder-1.5B, best head (L14 H0) | 0.874 | 0.800 |
| **+ LoRA (Track D), best head** (L14 H0) | **0.884** | 0.801 |
| base, trained bilinear probe on hidden states (L14) | 0.895 | 0.825 |
| + LoRA, trained bilinear probe on hidden states (L14) | 0.914 | 0.857 |
| + LoRA, same probe at the **final** layer (L28) | 0.807 | 0.717 |
| surface text: # of x/y/z fields that are the same token | 0.881 | 0.832 |
| **surface text: L1 distance between the two printed anchors** | **0.980** | **0.969** |

The stratified column computes AUC *within* sequence-distance bins and pools them. Lines
are emitted in `(y,z,x)` raster order, so `i-j` alone is a strong predictor; holding it
fixed collapses recency to 0.557 and leaves only what a predictor knows beyond position.

**The heads are not at chance — and it does not matter.** A single head in the
*un-finetuned* model reaches 0.874 (0.800 stratified), well clear of the recency floor.
But `-||a_i - a_j||_1`, three lines of numpy over the coordinates the serialization
literally prints on each line, scores **0.980 / 0.969** with no model at all. The
attention head recovers a lossy fraction of information that is already sitting in the
prompt in plain text. There is signal; there is nothing to *steal*.

**What the head is probably doing.** Its raw AUC (0.884) is inside the CI of
`coord_match` (0.881), the count of x/y/z fields that are the identical token — the
signature of a copying/induction head matching coordinate strings, not a head that
represents 3D neighbourhood.

**Finetuning does not create an adjacency head.** 20 epochs of LoRA on this exact
serialization moved the best head by +0.010 raw and **+0.001 stratified**, and the mean
over all 336 heads by +0.006. The same layer, the same head (L14 H0) wins before and
after.

**A trained probe beats raw attention, at the layer generation does not use.** The
supervised bilinear probe wins at every arm (0.914 vs 0.884), replicating the standing
result of the attention-probing literature. Its best layer is 14 of 28; by the final
layer — the one the LM head actually reads to emit the next token — it has fallen to
0.807 / 0.717. The adjacency information is mid-stack and partly gone by the output.

Figures: `head_auc.png` (all 336 heads, both arms), `probe_summary.png` (the table
above), `example_build.png` (one build's true adjacency vs the winning head vs recency
vs the printed coordinates) in
`outputs/run_20260906_221420_attn_adjacency_probe/`.

**Verdict: NEGATIVE for the proposal as stated**, and negative for a specific reason
worth keeping — not "attention knows nothing", but "attention knows a degraded copy of
the input, and is weakest exactly where the model would consume it". The direction the
graph-transformer literature actually runs — *inject* structure into attention — is
tested separately in T29.

## T27. Block ontology — a *measured* material reference for the agentic model (2026-09-06)

Track E hands the model a list of 70 block names and nothing else; everything it
believes about those materials is its own prior. `blockgen/ontology` attaches
knowledge to the names — colour and surface measured from the shipped textures,
placement behaviour and material affinity mined from `houses_32` (2,661 builds /
2.78M placements, ~1 s), game rules written once — and renders it as a table that
**replaces** the bare palette list in the system prompt (1,278 → 4,258 tokens).

Full design: [`docs/ontology.md`](docs/ontology.md). What the corpus said:

| block | layer | form | pairs with (NPMI, palette-closed) |
|---|---|---|---|
| `oak_planks` | mid | plate | oak_stairs · oak_log · torch |
| `oak_stairs` | roof | trim | oak_planks · oak_slab · oak_fence |
| `spruce_log` | mid | post | spruce_planks · spruce_stairs |
| `grass_block` | ground | plate | dirt · coarse_dirt · oak_leaves |

### The pilot (run `20260906_184620_ontology_pilot`, gpt-5-mini, 12 detailed prompts/arm, 48³, $0.22)

Four arms, all `oneshot` with one knob moved. `ont_shuffled` is the control: the
same table with every attribute permuted onto the wrong block, **token-matched to
`ont_mined` to the character**.

| arm | blocks | coherence_rate | cmd_ok | geom_kid ↓ | palette_jsd_family ↓ | BlockScore ↓ |
|---|--:|--:|--:|--:|--:|--:|
| `ont_none` | 1,926 | 0.92 | 0.997 | 6.67 [5.21, 8.71] | 0.243 [0.168, 0.317] | 172 |
| `ont_mined` | 1,623 | 0.83 | 1.000 | 7.92 [6.35, 10.36] | 0.193 [0.104, 0.281] | 205 |
| `ont_shuffled` | 2,213 | 0.67 | 1.000 | 8.13 [6.39, 10.50] | 0.186 [0.110, 0.261] | 210 |
| `ont_stats` | 1,989 | 1.00 | 0.999 | 9.55 [7.85, 12.35] | 0.211 [0.130, 0.291] | 247 |

Bench: `--tier fast --corpus houses_48` (size-matched to the 48³ canvas; scoring
these against `houses_32` inflates every geometry pillar and was re-run).

**Verdict: no effect, and the run is underpowered by design.** Every interval
overlaps every other. n = 12 is a plumbing and cost check, not a result — the
suite needs n ≥ 256 (T23/T26), and at n = 12 a one-build swing moves
`coherence_rate` by 0.08.

The one thing the pilot does settle is **why the control is mandatory**:
`ont_mined` and `ont_shuffled` are indistinguishable on every metric, and both
move `palette_jsd_family` toward the corpus relative to `ont_none` by about the
same amount. Whatever the table did here, it did not require the attributes to be
attached to the right blocks. Read without the control, `ont_mined`'s
0.243 → 0.193 palette improvement would have been reported as the ontology
working.

Material mix, mean per-build family share (top corpus families):

| family | corpus | `ont_none` | `ont_mined` | `ont_shuffled` |
|---|--:|--:|--:|--:|
| wood_oak | 16.6% | 15.4% | 10.2% | 11.3% |
| wood_spruce | 9.2% | 1.7% | **9.4%** | 6.2% |
| stone | 6.2% | 9.4% | **6.5%** | 7.1% |
| cobblestone | 5.6% | 4.2% | **6.6%** | 13.7% |
| stone_brick | 5.1% | 13.8% | 12.2% | 8.5% |

`ont_mined` lands closest on three of the five, and is the only arm that uses
spruce at corpus rate. Suggestive, not significant at this n.

**The renders and `geom_kid` disagree, which is itself a datum.** Looking at
`samples_ont_none.png` next to `samples_ont_mined.png`: the `ont_none` church is a
roof with a wall under it, its pagoda is a green mass, its arch bridge is a grey
lump. The `ont_mined` versions of the same three prompts have a bell tower with
arched windows, a tiered pagoda, and a bridge with a visible arch and lanterns
along the deck. `geom_kid` ranks `ont_none` *better* (6.67 vs 7.92, overlapping).
At n = 12 both readings are noise, but this is exactly the kind of case the
lab's 2AFC page exists to settle — and exactly the failure mode T25 found in the
render tier. Do not resolve it by argument; run the pairs.

**An arm-independent finding worth more than the ablation:** every agentic arm
uses ~10 distinct blocks per build against the corpus's **21.5**. Track E builds
are materially half as rich as real ones regardless of what it is told about
materials, and no metric in the current scorecard penalizes that directly
(`palette_size_w1` is in `dataset_stats`, not a pillar).

Next: n ≥ 128 per arm on the two live arms (`ont_none`, `ont_mined`) plus the
control, ~$2; and the second implementation — the ontology as a *validator* in the
repair loop rather than context, which costs no prompt budget at all.

## T26. Does the benchmark reward better builds? Dose-response, baselines, humans (2026-08-31)

> ### Bottom line
> On three of four dose axes both realism metrics are **perfectly monotone**
> (Spearman +1.000) across 105–226 noise floors. On the fourth — progressively
> filling interiors — `mv_dino_kid` has Spearman **+0.300, is not monotone, and
> its total dynamic range is 0 noise floors**, while `geom_kid` is +1.000 over
> 133. Separately: both metrics are **quadratic in the fraction of an arm that
> is bad**, so a 25%-failure arm scores only 31% of the way to fully-bad —
> reporting `sqrt(KID)` recovers the failure rate to a mean error of 0.022.
> Five procedural baselines and a 2AFC human study are now in place.

**Run:** `python scripts/bench_doseresponse.py --n 64` ·
`python -m blockgen.eval.bench --tier both --baselines …` ·
`python scripts/human_study_export.py`

### T26a. Dose-response — the question the ladder does not ask

The ladder asks whether a metric can tell damaged from real. A leaderboard needs
more: does the number move *smoothly and in the right direction* as quality
changes? A metric can pass every gate by firing on an artifact while being flat
across the range that separates two submitted models.

n=64 per rung, 399-build reference. `range (sd)` is best-to-worst span in units
of the metric's own noise floor.

| axis | metric | Spearman | monotone | range (sd) |
|---|---|---|---|---|
| **solidify** (interior filled, 0→100%) | `geom_kid` | **+1.000** | yes | **133** |
| | `mv_dino_kid` | **+0.300** | **no** | **0** |
| occupancy noise (0→20% blocks moved) | `geom_kid` | +1.000 | yes | 226 |
| | `mv_dino_kid` | +1.000 | yes | 143 |
| decimation (32³→8³) | `geom_kid` | +1.000 | yes | 105 |
| | `mv_dino_kid` | +1.000 | yes | 121 |
| **mixture** (fraction of arm that is real) | `geom_kid` | +1.000 | yes | 105 |
| | `mv_dino_kid` | +1.000 | yes | 121 |

Two readings. **The suite does reward better output**: on the mixture axis —
the one that most resembles how a generator actually fails, some good builds and
some bad — both metrics are perfectly monotone in the fraction of the arm that
is real. And **the solidify row is T25 restated as a curve rather than a point**:
the render metric's response to progressively deleting every interior in the
corpus is smaller, end to end, than one draw of its own sampling noise. It is
not weakly sensitive; it is flat.

The `partial_solidify` probe fills interior cells **inward from the walls**,
nearest-first, which gives an exactly calibrated dose (interior ratio 0.1175 →
0.0941 → 0.0708 → 0.0475 → 0.0242). Two earlier designs were rejected: random
speckle is detected as noise rather than as lost interior, and filling whole
rooms largest-first made the axis useless — one dominant pocket is typically
more than a quarter of the interior, so `frac=0.25` already removed 71% of it.

### T26b. Both metrics are quadratic in the failure rate

`MMD²(αP + (1−α)Q, P) = (1−α)²·MMD²(Q,P)`, so a squared-MMD metric responds to
the *fraction* of bad samples quadratically. Measured on the mixture ladder:

| bad fraction | `mv_dino_kid` | quadratic | linear | measured / linear |
|---|---|---|---|---|
| 0.25 | 0.043 | 0.036 | 0.138 | **0.31** |
| 0.50 | 0.149 | 0.138 | 0.274 | 0.54 |
| 0.75 | 0.304 | 0.308 | 0.410 | 0.74 |
| 1.00 | 0.546 | 0.546 | 0.546 | 1.00 |

Total absolute error against a quadratic 0.023, against a linear 0.325 — a 14×
better fit. **An arm that fails on a quarter of its builds looks 31% as bad as
one that fails on all of them.** That is the wrong shape for ranking generators,
which fail on subsets of prompts rather than uniformly.

The fix is free: `MMD = sqrt(MMD²)` is *linear* in the contaminated fraction.
`sqrt(KID)/sqrt(KID_worst)` recovers the true bad fraction to a mean absolute
error of **0.022** (`mv_dino_kid`) and **0.028** (`geom_kid`), against 0.119 and
0.142 for the raw values — five times better. The scorecard now carries `mmd`
beside every squared-MMD metric as a companion readout. It is a monotone
transform, so it inherits the ordering gates but not the ladder's noise-floor
units; read it for "what fraction of this arm is bad", read the metric itself
for "is this the right distribution".

**Which runs actually carry it.** The readout was added *after* the T26e run was
launched, so that scorecard does not contain an `mmd` key — the process had
already imported the old module. Verified separately on
`outputs/run_mmd_check_bench` (FAST tier, n=96): `real_test` geom_kid −0.145 →
`mmd` 0.000 (negative KID clamps to zero, as it should), `real@canon8` 17.721 →
4.210, `real@solidify` 23.542 → 4.852. Any run from this commit onward carries
it; T26e's does not. Caught by a reviewer of this entry, not by the author.

### T26c. Procedural baselines — the floor a leaderboard needs

Five non-learned arms, seeded, fitted on the **train split only** (a baseline
that had seen val or test would compete with an advantage no submitted model
has). Rendered and eyeballed before any number was written down, per T21's
standing lesson.

| baseline | what it is | blocks | wall_frac | thickness | interior | lcc |
|---|---|---|---|---|---|---|
| `uniform_random` | random voxels at real density | 1123 | 0.063 | 1.01 | 0.003 | 0.44 |
| `shell_box` | hollow box, floor + walls + flat roof | 1783 | 1.000 | 1.00 | 0.666 | 1.00 |
| `patchwork` | 8³ patches from *different* real builds | 1704 | 0.551 | 1.07 | 0.012 | 0.93 |
| `gabled_house` | box + pitched roof + door + windows | 1497 | 0.977 | 1.00 | 0.592 | 1.00 |
| `train_copy_noise` | a train build, 10% of blocks displaced | 1227 | 0.402 | 1.05 | 0.038 | 0.98 |
| **real** | | 1411 | 0.657 | 1.08 | 0.115 | 1.00 |

`gabled_house` is the bar that matters — it is what a competent afternoon of
hand-written rules produces, and the renders confirm it reads as a house. Note
it and `shell_box` are *too* hollow (0.59, 0.67 against a real 0.115): an empty
shell is as far from a real house as a solid blob, in the other direction, which
is the whole reason coherence is reported as distance-to-real.

`patchwork` is an **attack, not a baseline**. Every patch is lifted verbatim from
a real house, so the 2×2×2 pattern statistics that do most of the work in
`geom_kid` are near-perfect while the global structure is nonsense. Its position
in the leaderboard is the honest statement of the descriptor's limitation.

### T26d. The human study

Metric-vs-human agreement is the first thing a reviewer asks of a new metric, and
the ladder cannot supply it — it proves the metrics rank *synthetic corruption*,
not that they agree with people.

**Design.** `geom_kid` and `mv_dino_kid` are two-sample statistics: they score a
distribution, so there is no per-build number to correlate with a per-build
rating. Showing raters a grid from arm A beside a grid from arm B would match
the metric's level but asks people to eyeball a distribution, which they are poor
at. So raters compare **individual builds** (easy, reliable) and the judgements
are **pooled to the arm pair** (the level the metric speaks at). 9 arms → 36
pairs, 288 trials, 40 per session, sides randomized per trial.

Analysis is fixed in advance, in this order: exclude sessions below 75% on
real-vs-noise catch trials; fit **Bradley-Terry** arm strengths (not raw win
rate, which depends on which opponents an arm happened to be drawn against, and
coverage is always uneven when each session sees a random subset); then report
pairwise agreement on humanly-decided pairs plus Spearman against the BT scores.

The pipeline is **tested end to end on simulated raters**: 14 sessions, 2 of them
answering at random, and it dropped exactly those two on catch trials and
recovered the injected quality ordering. Arm labels are stripped from the page —
verified by grep — so the condition cannot be read off the source.

Instrument: `scripts/human_study_export.py` (stimuli + self-contained page) and
`scripts/human_study_analyze.py` (exclusion → Bradley-Terry → agreement).

### T26e. The full leaderboard, with the floor in it

n=128 controls and baselines, 399-build reference, 16 arms, self-validation 9/9.

| # | arm | BlockScore ↓ | worst pillar | appearance | geometry | palette |
|---|---|---|---|---|---|---|
| 1 | `real_test` | **0.00** | — | +0.00 | +0.00 | +0.00 |
| 2 | `real@canon16` | 60.80 | geometry | +36.05 | +60.80 | +0.88 |
| 3 | **`native_oriented`** | **87.14** | geometry | +62.38 | +87.14 | +6.28 |
| 4 | `real@canon8` | 170.35 | geometry | +159.06 | +170.35 | +4.46 |
| 5 | `real@solidify` | 201.80 | geometry | **+0.35** | **+201.80** | +2.51 |
| 6 | **`pick_n_place`** | **220.29** | geometry | +174.13 | +220.29 | +35.10 |
| 7 | `real_shuffled_materials` | 220.53 | appearance | **+220.53** | **+0.00** | +0.00 |
| 8 | `patchwork` | 224.63 | appearance | **+224.63** | **+183.03** | −2.97 |
| 9 | `gabled_house` | 300.15 | geometry | +291.87 | +300.15 | +25.61 |
| 10 | `train_copy_noise` | 302.73 | geometry | +145.03 | +302.73 | +1.11 |
| 11 | `real@monochrome` | 304.62 | appearance | +304.62 | +0.00 | +79.63 |
| 12 | `shell_box` | 464.49 | appearance | +464.49 | +312.39 | +24.91 |
| 13 | `uniform_random` | 562.77 | appearance | +562.77 | +344.94 | +23.70 |
| — | **`agentic_oneshot`** | UNRANKED (n=12) | — | +35.19 | +78.56 | +35.44 |
| — | `real@single_mode` | **DQ** mode collapse | — | — | — | — |
| — | `train_verbatim` | **DQ** memorization | — | — | — | — |

**All three learned arms beat all five procedural baselines on BlockScore** —
which is the first time this project has been able to say that, because until now
there was no floor to say it against. `pick_n_place` (220) clears `patchwork`
(225) by very little, and on `geom_kid` alone it is actually *worse*
(24.06 vs 19.98) — consistent with T24e, where every sample hits the 384-node cap.

**The two blind spots are visible in one table, and they are complementary.**
`real@solidify` reads +0.35 on appearance and +201.80 on geometry; `patchwork`
reads +224.63 on appearance and +183.03 on geometry. Each tier is fooled by the
corruption aimed at it and rescued by the other. The asymmetry is worth noting
honestly: the render tier's blindness to `solidify` is near-total (a factor of
~580 between the pillars), whereas the attack on `geom_kid` only moves it about
1.2× — local-pattern matching buys `patchwork` a better geometry score than
genuinely coherent but simple builds get, but nothing close to a pass.

`patchwork`'s palette score is **−2.97**, i.e. *closer to real than the held-out
real set is*, which is exactly right: its blocks are literally real blocks.

### T26f. A pre-registered prediction for the human study

Both metrics rank **`patchwork` significantly above `gabled_house`** — separate
Holm-corrected groups on both (`geom_kid` e vs g; `mv_dino_kid` g vs h). The
renders say `gabled_house` is a recognisable house with a pitched roof and
windows, and `patchwork` is a pile of real fragments that is globally incoherent.

**Prediction, recorded before any human data is collected: human raters will
reverse this pair.** If they do, it localises a real limitation — both metrics
are distribution distances, and `gabled_house` is *too regular* to sit in the
real distribution (wall fraction 0.977 against a real 0.657, perfect bilateral
symmetry, an empty shell interior of 0.592 against 0.115) while `patchwork`
inherits real texture statistics wholesale. The metrics would then be rewarding
*statistically* house-like over *recognisably* a house.

If humans agree with the metrics instead, that is a finding too, and a more
comfortable one. Either way the pair is the highest-information comparison in the
study, and it is registered here rather than selected afterwards.

<!--HUMAN-RESULTS-->

---

## T25. The benchmark's realism metric was structurally blind (2026-08-31)

> ### Bottom line
> Filling every enclosed air cell in the held-out corpus — a median **31% more
> blocks**, every room deleted, silhouette untouched — leaves **MV-DINO-KID
> unable to separate it from held-out real** (0.000 vs −0.001; same group after a
> paired Holm-corrected test that *does* separate `train_verbatim` at a
> difference of 0.015). A generator that emits solid blobs shaped like houses is
> invisible to the primary realism metric, and it is not a power problem. `blockgen/eval/bench/geometry.py`
> adds **`geom_kid`**, the same unbiased estimator on D4-invariant voxel
> descriptors, which puts the same arm **202 real-sample spreads** from real,
> passes every ladder gate, and needs no GPU. Three defects were found and fixed
> along the way, two of them in the ladder's own denominators.

**Run:** `python -m blockgen.eval.bench.ladder --family geometry` ·
`python -m blockgen.eval.bench --tier both --n 128 --n-ref 399 --arms …`

### T25a. The blindness, measured

`probes.solidify` fills enclosed air with the build's own dominant material, so
it is not detectable as a palette anomaly either. On 200 held-out real houses it
adds a median **31%** blocks (mean 37%) and affects **91%** of builds. Scored at
n=128 against the same 399 val builds:

| arm | `mv_dino_kid` ↓ | `geom_kid` ↓ |
|---|---|---|
| `real_test` (floor) | −0.001 [−0.004, 0.003] | −0.09 [−0.22, 0.04] |
| **`real@solidify`** | **0.000 [−0.002, 0.004]** | **22.1 [20.7, 23.7]** |
| `real@canon16` | 0.109 | 6.57 |
| `real_shuffled_materials` | 0.673 | **−0.09** — bit-identical to real |
| `real@monochrome` | 0.930 | **−0.09** — bit-identical to real |

The bottom two rows are why this is a second tier and not a replacement:
`geom_kid` reads occupancy only, so a material permutation moves it by *exactly*
zero. The tiers are complementary halves, and the ladder gates each on the rungs
it can see — `GEOMETRY_RUNGS` replaces the material-noise triple with an
occupancy-noise triple and moves material corruptions into its *invariance* set.

This was not a hypothetical risk. T23d recorded `native_oriented` at an
enclosed-air ratio of 0.005 against a real 0.114 — essentially solid — while it
scored respectably on KID, and the only thing that caught it was a hand-picked
scalar with no distributional metric behind it.

### T25b. The geometry ladder (n=64, CPU, ~1 min)

| metric | real | canon16 | canon8 | solidify | occ_noise_10 | jitter_cols | mat-shuffle | rot90 | sd | validity |
|---|---|---|---|---|---|---|---|---|---|---|
| **geom_kid** | 0.555 | 13.46 | 21.11 | **23.74** | 32.18 | 37.77 | *0.555* | *0.555* | 0.347 | **PASS** |
| geom_mmd_rbf | 4.32 | 112.8 | 183.6 | 217.1 | 286.8 | 348.7 | *4.32* | *4.32* | 2.44 | **PASS** |

Italic cells are bit-identical to the real column — exact invariance, not
approximate. For comparison, the render tier on the same rung: `kid` scores
`solidify` at **0.6 sd** and does not carry the `S6_resolves_solidify` flag;
`geom_kid` scores it at **67 sd**.

**Which part of the descriptor does the work** (noise-floor sd, n=64):

| descriptor | dim | canon16 | solidify | occ_noise_10 | jitter_cols | mat-shuffle |
|---|---|---|---|---|---|---|
| full | 77 | 51.2 | 92.0 | 125.4 | 147.6 | **0.0** |
| − pattern block | 22 | 74.8 | 103.8 | 58.7 | 93.0 | **0.0** |
| − scalars | 68 | 31.4 | 76.7 | 116.2 | 136.2 | **0.0** |
| pattern block only | 55 | 30.6 | 51.4 | 107.9 | 123.8 | **0.0** |
| − block count | 76 | 42.0 | 89.8 | 122.7 | 144.2 | **0.0** |

No component carries it alone, and removing the block count barely moves
`solidify` (92.0 → 89.8), so this is not a size detector in disguise.

### T25c. Three defects, two of them in denominators

**1. The renderer failed open — and would have destroyed the feature cache.**
`features.render_views` substituted a blank white frame for any exception. A
PyOpenGL/Python version mismatch made every `render_structure` call raise: every
image came back white, every DINO feature collapsed to one vector, and every arm
scored **KID = −4e-13**, i.e. better than real. Worse, `load_or_build` checks a
`render_canary_sha` before trusting its cache, so the all-white canary would have
mismatched and the 30 MB cache would have been **silently rebuilt from blank
frames**. Now raises past a 2% failure rate. Repairing PyOpenGL (3.1.0 → 3.1.7)
reproduced the stored canary `058d907f0d4bbf54` bit-for-bit, so no previously
recorded number is affected.

**2. `sd(real)` — the unit every gate is stated in — was mismeasured twice.** It
came from repeated subsamples of a pool that *excluded* the probe set, so the
"fresh real draw" G7 tests was not a member of its own null; and overlapping
draws carry a finite-population factor that shrinks their spread. Corrected,
`geom_kid` moved from failing G7 at 3.5 sd to passing. A metric separating
`solidify` at 67 sd had been rejected for noise that was measured wrong.

**3. G8 tested a difference of means against the spread of a single draw** —
larger by `sqrt(reps)`, so it had almost no power, and it passed `legacy_cmmd`.
Against the standard error of the difference, the same estimator's documented
1/n decay is rejected decisively:

| metric | n=16 | n=32 | n=64 | n=128 | verdict |
|---|---|---|---|---|---|
| `kid` | 0.0038 | 0.0053 | 0.0043 | 0.0029 | stable |
| `legacy_cmmd` | 0.1733 | 0.0999 | 0.0583 | 0.0389 | **drifts 14.8 se** |
| `fd` | 0.1994 | 0.1443 | 0.1078 | 0.0759 | **drifts 16.6 se** |
| `one_minus_coverage` | 0.7409 | 0.5640 | 0.3311 | 0.1396 | **drifts 16.0 se** |

`legacy_cmmd`'s null mean halves every time n doubles, exactly as the estimator's
bias predicts. **Correction to T23a:** the earlier fd / coverage / cmmd G8
failures were right, but for a weaker reason than stated; `one_minus_recall` now
also fails G8, and `one_minus_coverage` fails G8 rather than G6.

### T25d. Interiors were undercounted on real data too

`enclosed_air_ratio` counts only *fully sealed* air, so a room with an open
doorway counts as no room. `geometry.interior_volumes` also closes the solid
(26-connected, sealing openings up to two blocks wide) and reports the air that
becomes enclosed only then. Calibration on a hollow 9-cube with a 343-cell room:
a 1×1, 1×2 or 2×2 opening recovers all 343; a 3×3 hole or a missing wall recovers
0 (correctly an open structure, not a room). On 200 held-out real houses:

* **9.0% have no sealed interior at all — and 94.4% of those do have a room.**
* Apertures account for **20.5% of all real interior volume.**

Every earlier statement in this project that generated builds "have no interiors"
was made with the sealed-only counter and understates real houses by that much.

### T25e. BlockScore — a leaderboard the known cheats lose

Realism alone has a trivial winning strategy: T23c measured `train_verbatim`
scoring second best in the table. So novelty and diversity are **disqualification
gates**, not terms, and pillars combine by **worst case**, never by average — a
mean would let an arm buy a bad pillar with a good one, and the trade the render
tier invites is precisely the dangerous one. Units are real-sample spreads,
calibrated in-run against `real_test`; there are no weights to tune.

The aggregate is validated like a metric: `composite.validate()` asserts real
wins and every degenerate control loses, and a run that fails prints a warning
telling you not to quote the ranking. **It caught two real defects on its first
run** — a memorization gate silently disabled because the control's duplicate
rate is exactly 0 with a zero-width interval (so `z` was NaN and
`train_verbatim`, at duplicate rate 1.000, sailed through), and a blindness check
being applied to a tier that cannot see it.

**Cross-track, n=128 controls, 399-build reference, 8.5 min wall (both tiers,
both head-to-head tables). Self-validation 9/9.**

| # | arm | BlockScore ↓ | worst pillar | appearance | geometry | palette | coherence | geom scalars |
|---|---|---|---|---|---|---|---|---|
| 1 | `real_test` | **0.00** | — | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| 2 | `real@canon16` | 60.80 | geometry | +36.26 | +60.80 | +0.88 | +0.79 | +0.68 |
| 3 | **`native_oriented`** | **87.14** | geometry | +61.87 | +87.14 | +6.28 | +0.50 | +0.37 |
| 4 | `real@canon8` | 170.35 | geometry | +158.77 | +170.35 | +4.46 | +1.77 | +1.75 |
| 5 | `real@solidify` | 201.80 | geometry | **+0.30** | **+201.80** | +2.51 | +0.31 | +2.73 |
| 6 | **`pick_n_place`** | **220.29** | geometry | +174.62 | +220.29 | +35.10 | +0.50 | +0.92 |
| 7 | `real_shuffled_materials` | 221.37 | appearance | **+221.37** | **+0.00** | +0.00 | +0.00 | +0.00 |
| 8 | `real@monochrome` | 304.57 | appearance | +304.57 | +0.00 | +79.63 | +0.00 | +0.00 |
| — | **`agentic_oneshot`** | UNRANKED (n=12) | — | +35.15 | +78.56 | +35.44 | +0.39 | +0.91 |
| — | `real@single_mode` | **DQ** mode collapse (+45.1) | — | — | — | — | — | — |
| — | `train_verbatim` | **DQ** memorization (+10.8) | — | — | — | — | — | — |

Rows 5 and 7 are the two blindnesses side by side: `solidify` is +0.30 on
appearance and +201.80 on geometry; `shuffled_materials` is +221.37 on appearance
and **exactly** +0.00 on geometry. Neither tier alone ranks both correctly, and
the max-over-pillars rule is what makes the aggregate catch both.

Cross-track ordering is unchanged from T23d where they overlap — agentic ahead of
`native_oriented` on both realism metrics — with `pick_n_place` last, consistent
with T24/T24e (every sample hits the 384-node cap, so its builds are truncated).
`agentic_oneshot` stays **unranked at n=12**; regenerate at n ≥ 128 before
quoting it (≈ $0.07).



### T25f. Head-to-head, and the p-value that made it vacuous

Marginal intervals are not a comparison: non-overlap demands a difference 41%
larger than significance does. `compare.rank_table` tests every pair with the
reference draw shared and applies Holm across the family.

The first run put **every arm, from held-out real to a monochrome corpus, in one
"not separated" group.** A bootstrap sign-balance p is floored at `1/n_rep`, and
Holm over the 55 pairs of an 11-arm run multiplies 0.005 to 0.275 — nothing can
reach 0.05 however large the difference. The reported `p` is now studentized and
is not floored; `p_boot` is kept alongside.

Corrected, the two tiers give the cleanest statement of T25a. Arms sharing a
letter are not separated at α = 0.05 after Holm:

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

**`real@solidify` shares group `a` with held-out real on the render metric.** This
is not a power failure: the same test at the same n separates `train_verbatim` at
a difference of 0.015 and gives it its own group. The render tier resolves a
0.015 difference and cannot resolve the deletion of every interior in the corpus.
On `geom_kid` the same arm is group `f`, Δ = −22.13 [−24.08, −20.72], p ≈ 0.

Two things measured and *not* claimed: pairing the reference changes interval
width by under 10% and inconsistently in sign, so it is the correct null rather
than a power gain — but the term it cancels analytically is the expensive one,
and `distances.kid_arm_term` skips it exactly, taking a 66-pair table from tens
of minutes to about one.

---

## T24. Pick-and-place — learned placement, first test (2026-08-16)

> ### Bottom line
> The mechanism works and T21's collapse modes are gone; the model is **not yet
> competitive** on realism. Picker/placer train in **1.6 min** to a placer that is
> **84× above the legal-face chance baseline** (val 0.713 vs 0.0085). Generated
> builds are solid, connected, material-stratified massing — no filaments, no
> plates, no balls. But KID 0.532 is worse than every existing track, and the
> dominant reason is size: 185 blocks against a real 904, because training
> truncates at `max_nodes=192`.
>
> One bug is worth the entry on its own: a train/inference skew that **no
> teacher-forced metric could see**. See T24b.

Code: `blockgen/models/pick_n_place.py`, `utils/growth_order.py`,
`training/train_pick_n_place.py`, `scripts/train_pick_n_place.py`.
Run: `outputs/run_20260816_081506_pnp_fixed` · 30 epochs, 1,863 builds, 5.4M params.

    Picker : G     -> P(V)    which piece next (or STOP)
    Placer : G x V -> P(E)    which open face to attach it to

This is `implementation_plan.md` §3's state encoder. T21 ran the same growth
process with placement decided by a hand-written frontier heap and failed with a
specific diagnosis — the model could not read its own op history as geometry.
Here placement is learned, and geometry enters attention as a bias on the clamped
relative 3D offset between placed nodes.

### T24a. Phase 0 — representation

Round-trip **IoU exactly 1.000 (min 1.000)** across bfs/layered/dfs on 200 real
builds; replay walks parent/direction from the seed and never reads the stored
coordinates. Block retention 0.994 (loss is non-largest components, dropped
deliberately). Legality mask verified against ground truth: **11,940/11,940**
true placements legal, seed row empty.

### T24b. A bug only free-running generation could see

| | before fix | after fix |
|---|---|---|
| val place_acc | 0.715 | 0.713 |
| place_lift | 84.0× | 83.8× |
| **median blocks generated** | **30** | **187** |

`node_features` normalized the step index by the *current* sequence length — the
padded batch max (192) in training, but the current build length during
generation. Three nodes in, every node read as "end of build"; the picker had
learned STOP-at-1.0 and stopped almost immediately.

**Every teacher-forced number stayed excellent with the bug in place.** Loss,
accuracy and an 84× lift over chance all looked like a working model. Only the
free-running block count moved — by 6×. Any metric computed with ground truth fed
in is blind to this entire class of failure, which is precisely why T21's
teacher-forced numbers could not be trusted. **A growth model needs at least one
free-running number in its training log, every run.** Guarded by
`test_prefix_features_match_full_sequence`.

### T24c. Against the other tracks (n=128 ref, same split)

| arm | n | KID ↓ | recall ↑ | novelty ↑ | connected | interior | floating | blocks |
|---|---|---|---|---|---|---|---|---|
| real_test | 128 | −0.001 | 0.882 | 0.536 | 0.977 | 0.106 | 0.112 | 1003 |
| agentic | 12 | **0.107** | 0.992 | 0.673 | 0.983 | 0.021 | 0.017 | 1699 |
| native_oriented | 64 | 0.192 | 0.799 | 0.525 | 0.955 | 0.005 | 0.155 | 1082 |
| **pick_n_place** | 16 | 0.532 | 0.075 | 0.604 | **1.000** | 0.000 | **0.000** | **185** |
| *real@canon8* | 128 | *0.485* | *0.115* | *0.724* | *0.826* | *0.002* | *0.293* | *47* |

Read this carefully:

1. **`connected 1.000` and `floating 0.000` are structural guarantees, not
   achievements.** Growth cannot emit a disconnected or floating voxel. Real
   builds score 0.977/0.112, so the model is *further from the data* than the
   other tracks on both — the same "validity by construction is a filter, not
   learning" caveat this project already applies to constrained decoding.
2. **KID 0.532 is dominated by size.** At 185 blocks it sits beside `real@canon8`
   (47 blocks, KID 0.485). Until `max_nodes` reaches real build sizes (~904
   median) this number is measuring truncation.
3. **It does not memorize** (dup 0.000, novelty 0.604) but `diversity` 0.836 is
   the lowest of any arm, which is worth watching.
4. **No track builds interiors.** pick-and-place is 0.000, joining native (0.005)
   and agentic (0.021) against a real 0.114.

### T24d. The prefix test — blindness is caused by the *training data*, not the model

T21's diagnostic, run against pick-and-place. Teacher-force the first K nodes of
a real held-out build, let the model continue, and ask how close the continuation
is to what the build actually had left. The statistic is **|cont_ratio − 1|**,
distance from correct.

| model | trained on | \|ratio−1\| by prefix | verdict |
|---|---|---|---|
| `pnp_384` | all 1,863 builds (every one truncated at the cap) | 0.237 → 0.272 → **0.418** | error **grows** → blind |
| `pnp_complete` | 258 builds that fit under the cap | 0.213 → **0.103** | error **halves** → uses the prefix |

Same architecture, same hyperparameters, opposite verdicts. The difference is
what STOP was allowed to mean.

**Two measurement mistakes were made getting here, both worth the entry.**

1. *No variance in the pool.* The first run at `max_nodes=192` reported
   cont_ratio 0.979/0.978/0.972/0.976 — flat, high, and meaningless. The corpus
   median is 623 nodes, so every held-out build truncated to exactly 192,
   `true_remaining` was `192 − K` for all of them, and the model emitted ~188
   every time. `(188−K)/(192−K) ≈ 0.98` is arithmetic; a model ignoring the
   prefix scores identically. Only **7 of 399** test builds are naturally under
   192. The script now restricts to untruncated builds and refuses a verdict when
   length variance is degenerate.
2. *The wrong direction of error.* The rule initially read any decline in
   cont_ratio as blindness. But `pnp_complete` free-runs 21% **too long** and
   drops to 10% too short when given half a real build — the prefix pulling it
   toward the correct length, scored as a failure. T21's blindness was shutting
   down *below* target, so the error has to grow, not the ratio fall.

**Mechanism.** With `max_nodes=384`, `complete_frac` is 0.138 — 86% of builds are
cut off arbitrarily, so STOP never marks a real ending. Suppressing the STOP
target on truncated builds (correct in principle) made it 0.036% of pick targets
and the model stopped emitting it entirely (STOP rate **0.00**, every build ran
to the cap). Training only on complete builds restores it (STOP 0.62) and yields
a model that uses the prefix — at the cost of 7× less data, so `place_lift` falls
139.8× → 51.8×.

That is the real tension, and it is a data problem, not an architecture problem:
**at any cap the corpus can afford, most builds cannot teach an ending.** The fix
is a cap above the corpus median (~623), which needs the sparse-candidate
refactor since `legal` is `[B,N,N,6]`.

Caveat: 12 builds and 2 prefix points. Directionally clear, statistically thin.

Next, in order: sparse candidate list so `max_nodes` can reach ~1024 (unblocks
both the size confound and the STOP signal), re-run the prefix test at n≥32
builds, then n≥256 samples before quoting KID.

### T24e. Formulation audit — the placer's label is not identifiable (2026-08-17)

Written up in full at `docs/pick-and-place.md`. Four measurements, all on
`houses_32` train/val under the canonical split.

**1. The placer is trained on an unidentifiable target.** Several open faces can
point at the *same* empty cell, so placing block `v` at cell `c` from parent A or
from parent B yields a byte-identical structure — but cross-entropy names one
correct and scores the rest as errors. Over 56,223 real placements:

| faces aiming at the target cell | 1 | 2 | 3 | ≥4 |
|---|---|---|---|---|
| share | 19.8% | **61.6%** | 18.2% | 0.4% |

Mean multiplicity **1.99** ⇒ a model that treats tied faces as equivalent and
breaks ties at random tops out at `place_acc` **0.568**. Measured val `place_acc`
is **0.729**, *above* that ceiling — so part of the reported accuracy is the model
reproducing **BFS's tie-break convention**, an artifact of the serializer rather
than a property of houses. Fix: marginalize the loss over cells (`logsumexp` the
face logits within each `φ⁻¹(c)` fibre) so the label becomes well-defined and
`place_acc` becomes comparable across orderings. Not yet implemented.

**2. `pick_acc` never had a baseline.** `place_acc` is reported against its chance
floor, `pick_acc` bare. On val (143,356 nodes, 190 pieces): most-frequent-block =
**0.189** vs measured **0.607** (3.2×); unigram entropy **3.717 nats** vs measured
`pick_loss` **1.578**. The picker is genuinely learning — previously unquotable.
`evaluate()` should emit both, as it does `place_chance`.

**3. Uncapped corpus scale** (400 train builds, largest component):

| | p50 | p90 | max |
|---|---|---|---|
| nodes per build | **1,069** | 2,611 | 7,673 |
| open faces at completion | 1,980 | 4,980 | 9,072 |
| Σₜ open-faces(t) | 1.1 M | 6.5 M | 32 M |

Σₜ Fₜ ≈ N² exactly (median ratio 1.00 — Fₜ *is* the partial build's surface area),
which prices the "make every candidate connection a token" idea at 1.1 M tokens
per median house. Dead. Persistent face tokens cost 6N ≈ 6.4k (linear, but 7× the
sequence for the same content). A **typed** connection token is tractable: 10,842
distinct `(parent_piece, dir, child_piece)` triples, 1,386 covering 90%, and it
prunes the pointer's candidate set from a median 1,928 open faces → 286 (direction
fixed) → **27** (typed). It cannot replace the pointer, only shrink it.

**4. The window that makes the refactor free.** Recency gap `t − parent`, i.e. how
far back the true parent is:

| ordering | gap p50 | parent within last K |
|---|---|---|
| **bfs** | 122 | K=512 → **100.0%** (bounded by BFS layer width) |
| dfs | 1 | K=32 → 98.5%, but the tail reaches 4,289 |
| layered | 86 | K=256 → 71.9% |

So under BFS, `[B,N,N,6]` → `[B,N,512,6]` loses **zero** reachable targets and
makes placer memory linear in N. That is the sparse-candidate refactor, now with a
measured K. (Measured on 60 builds / 60,305 placements — re-check on the full
split before hard-coding 512.)

**Correction to T24d.** Re-running the prefix test on the same `--complete-only`
checkpoint at **n=24** (was n=12) moves `abs(cont_ratio−1)` from `0.213 → 0.103`
to `0.169 → 0.105 → 0.129`, and the verdict from `USES THE PREFIX` to `NOT BLIND
BUT NOT READING LENGTH`. `length_corr` is ≈0 for **both** training sets (−0.05
all-builds, −0.02 complete-only). The defensible claim is narrower than T24d's:
`--complete-only` removes the *blindness signature* (the error stops growing with
prefix length) but does **not** make the model read build length. The n=12
optimistic reading did not survive doubling the sample.

---

## T23. Evaluation suite — a validated cross-track benchmark (2026-08-03)

> ### Bottom line
> `blockgen/eval/bench` scores any track from one `.npz` of structures against one
> fixed real reference. **MV-DINO-KID passes every validity gate** and is the new
> primary realism metric. The legacy `perceptual.cmmd` **fails n-stability**, so
> every CMMD number in T20/T21 is comparable only at fixed n. Fréchet distance
> and PRDC precision/density/coverage also fail and are reported as `null`.
> First cross-track numbers (T23d): **agentic beats native_oriented on realism**
> (KID 0.107 vs 0.187, non-overlapping), **neither memorizes**, and **neither
> builds interiors** — enclosed-air 0.021 and 0.005 against a real 0.114.

**Run:** `python -m blockgen.eval.bench --tier both --arms <track>/<name>:<path.npz>`
· ladder: `python -m blockgen.eval.bench.ladder` · split: `…bench.splits --show`

### T23a. The metric-validation ladder (dinov2b, 4 views @224px ortho, n=64)

Every metric is scored on rungs of known damage, in units of its own noise floor
(the spread of real-vs-reference draws). Validity gates are blocking; sensitivity
is recorded separately because *how much* damage a metric resolves is a question
about n, not about whether the metric is measuring the right thing.

| metric | real | canon16 | canon8 | noise10 | del40 | mat-shuffle | rot90 | sd | validity |
|---|---|---|---|---|---|---|---|---|---|
| **kid** | 0.102 | 0.278 | 0.626 | 0.224 | 0.140 | 0.897 | 0.109 | 0.005 | **PASS** |
| **mmd_rbf** | 30.3 | 95.7 | 206.5 | 69.2 | 46.2 | 269.7 | 31.5 | 1.97 | **PASS** |
| **one_minus_recall** | 0.222 | 0.734 | 0.992 | 0.516 | 0.371 | 0.952 | 0.246 | 0.022 | **PASS** |
| fd | 0.135 | 0.201 | 0.320 | 0.172 | 0.149 | 0.399 | 0.137 | 0.003 | FAIL G8 n-stability |
| one_minus_coverage | 0.528 | 0.746 | 0.940 | 0.637 | 0.617 | 0.968 | 0.512 | 0.038 | FAIL G6 invariance |
| one_minus_density | 0.131 | 0.478 | 0.891 | 0.331 | 0.338 | 0.928 | 0.094 | 0.074 | FAIL G2 noise-order |
| one_minus_precision | 0.141 | 0.203 | 0.609 | 0.188 | 0.094 | 0.734 | 0.109 | 0.025 | FAIL G2 noise-order |
| `legacy_cmmd` | 0.308 | 0.745 | 1.628 | 0.615 | 0.400 | 2.326 | 0.326 | 0.017 | FAIL G8 n-stability |

Three things this settles:

1. **KID reads appearance, not solidity.** Material shuffle preserves geometry
   exactly and scores *worst of all rungs* (0.897 vs 0.102 real). That is the
   specific failure T21 recorded for CMMD (r = −0.993 vs thickness).
2. **`perceptual.cmmd` is the biased MMD estimator**, despite its docstring. Its
   within-set means include the diagonal, so its null on two disjoint real halves
   decays as ~1/n — measured 0.289 (n=16), 0.071 (n=64), 0.014 (n=400). The
   unbiased KID here measures 0.000 at every n. `perceptual.py` is deliberately
   **not patched**, so T20/T21 stay reproducible; the finding is recorded instead.
   (Note `cmmd(x, x)` on an *identical* array is exactly 0.0 — the bias only shows
   across two disjoint samples.)
3. **Precision/recall are unstable at n=64**, as expected: precision and density
   fail monotonicity in block noise. Recall passes; coverage fails invariance.

**Sample size.** KID's noise sd is 0.005 at n=64 and 0.04 at n=16, while the
real→canon-16 gap is 0.18. Runs default to `--samples 16`
(`experiments_overnight.py:396`), which resolves only differences >0.15.
**Headline numbers need n ≥ 256; n = 64 is the reportable floor; n = 16 cannot
rank arms.**

### T23b. FAST tier — real-data controls (n=150, ref = 399 val builds)

| arm | lcc_ratio | enclosed_air | palette JSD (exact) | nn_iou | dup_rate |
|---|---|---|---|---|---|
| real_test (floor) | 0.978 (real 0.980) | 0.094 (real 0.115) | 0.092 | 0.371 | 0.007 |
| real@canon16 | 0.893 — 1.63 sd | 0.042 — 0.50 sd | 0.099 | 0.367 | 0.000 |
| real@canon8 | 0.816 — 3.74 sd | 0.000 — 0.84 sd | 0.295 | 0.314 | 0.000 |
| real_shuffled_materials | *identical to floor* | *identical* | *identical* | 0.363 | 0.000 |
| real@monochrome | *identical to floor* | *identical* | **0.901** | 0.363 | 0.000 |
| train_verbatim | 0.981 | 0.101 | 0.196 | **1.000** | **1.000** |

`train_verbatim` confirms the memorization detector fires perfectly on literal
copies. `real_shuffled_materials` is bit-identical to the floor on every palette
column — voxel palette statistics are **provably blind** to arrangement, which is
what the render tier is for. `real@monochrome` is the converse.

**Coherence is reported as distance-to-real, never "higher is better."** Only 66%
of real val houses are single-component at native resolution (43% at canon-16), so
an arm scoring `lcc_ratio = 1.0` is as far from the data as one scoring 0.2. The
scorecard schema has no slot for a bare coherence rate.

### T23c. FULL tier — why novelty is a co-equal pillar, not a footnote

Run: `--tier both --n 128 --n-ref 399`, 54 s wall after the feature cache is warm.
Ladder-passing metrics only; `fd`, coverage, density and precision are emitted as
`null` with their failing gate named.

| arm | MV-DINO-KID ↓ | dino_nn_percentile ↑ | dino_dup_rate ↓ |
|---|---|---|---|
| real_test (floor) | −0.001 [−0.004, 0.003] | 0.539 | 0.02 |
| real@canon16 | 0.109 [0.095, 0.129] | 0.541 | 0.01 |
| real@canon8 | 0.485 [0.462, 0.512] | 0.723 | 0.00 |
| real_shuffled_materials | 0.674 [0.643, 0.711] | 0.795 | 0.00 |
| real@monochrome | 0.930 [0.903, 0.962] | 0.949 | 0.00 |
| **train_verbatim** | **0.015** [0.006, 0.024] | **0.000** | **1.00** |
| agentic `oneshot_detailed` (n=8) | 0.065 [0.001, 0.158] | 0.734 | 0.00 |

**`train_verbatim` is the whole argument in one row.** A model that does nothing
but recite its training set scores second-best in the table on realism — 0.015,
closer to real than any corruption — and is caught *only* by the novelty columns.
Any suite that reports realism without novelty on the same row has a trivial
winning strategy. This is not hypothetical here: the AR baseline duplicates ~31%
of training builds at IoU ≥ 0.95 (T5).

Track E's KID point estimate (0.065) sits below canon-16, i.e. its builds look
closer to real houses than decimated real houses do — but at n=8 the interval
spans [0.001, 0.158] and the run carries a `min_n` warning. **Regenerate Track E at
n ≥ 128 before quoting this.** At ~$0.0005/build that is roughly $0.07.

### T23d. First cross-track comparison — native AR vs agentic vs real

`--tier both --n 128 --n-ref 399`, all arms scored against the same 399 held-out
real builds. Native samples generated fresh from the served checkpoint via
`scripts/sample_to_npz.py --model native_oriented --n 64` (10.4 min, 0 empty,
median 890 blocks); agentic pooled from both showcase runs (n=12).

| arm | n | KID ↓ | novelty ↑ | dup ↓ | palette JSD ↓ | recall ↑ |
|---|---|---|---|---|---|---|
| real_test (floor) | 128 | **−0.001** [−0.004, 0.003] | 0.537 | 0.000 | 0.014 | 0.882 |
| `train_verbatim` | 128 | 0.015 [0.006, 0.023] | **0.000** | **1.000** | 0.016 | 0.845 |
| **agentic** oneshot | 12 | **0.107** [0.071, 0.152] | 0.673 | 0.000 | 0.228 | 0.992 |
| real@canon16 | 128 | 0.110 [0.095, 0.129] | 0.541 | 0.000 | 0.018 | 0.684 |
| **native_oriented** | 64 | 0.187 [0.155, 0.219] | 0.524 | 0.000 | 0.054 | 0.805 |
| real@canon8 | 128 | 0.485 | 0.723 | 0.000 | 0.024 | 0.113 |
| real_shuffled_materials | 128 | 0.668 | 0.792 | 0.000 | 0.014 | 0.188 |

Coherence — generated (real in parentheses):

| arm | connected | interior | floating | blocks |
|---|---|---|---|---|
| real_test | 0.977 (0.983) | **0.106** (0.114) | 0.112 (0.103) | 1003 (904) |
| agentic | 0.983 (0.983) | 0.021 (0.114) | 0.017 (0.103) | 1699 (904) |
| native_oriented | 0.955 (0.983) | **0.005** (0.114) | 0.155 (0.103) | 1082 (904) |

1. **Agentic beats native on realism with non-overlapping intervals** (0.107
   [0.071, 0.152] vs 0.187 [0.155, 0.219]). Provisional: agentic is n=12, below
   the `min_n` guard, and the run carries that warning.
2. **Native's samples are further from real houses than 86%-decimated real houses
   are** — KID 0.187 vs canon-16's 0.110.
3. **Neither memorizes.** Both at dup-rate 0.000 with novelty percentiles at or
   above the real floor (0.524, 0.673 vs 0.537). The detector is not simply
   insensitive: `train_verbatim` reads 0.000 / 1.000.
4. **They fail in opposite directions.** Native learned the corpus palette (JSD
   0.054 vs agentic's 0.228) but builds almost solid blobs (interior 0.005 vs
   real 0.114) with the most floating mass here (0.155 vs 0.112). Agentic matches
   real connectivity exactly and has the least floating mass, but its palette is
   16× further from real — expected from a small hand-written DSL palette.
   **Neither track builds interiors**, which the FAST tier alone would have found.

Regenerate: `python scripts/bench_report.py --run <run_dir>`.

### T23e. Split hygiene

The canonical split is **group-aware**: GrabCraft ships sibling builds ("American
Middle Class House 10 / 22 / 9") that a random row split scatters across train and
val. Measured: **35.8% of val items would share a build family with train** under a
random split, which inflates the held-out floor and deflates every memorization
number at once. Group-aware: 0%. 2054 groups over 2661 builds, 70/15/15, tracked at
`data/minecraft/splits/`.

The same contiguity bug bit the ladder itself — taking the first n test builds gave
the probe set and null pool different category mixes and inflated every metric's
noise floor by ~45% (KID sd 0.009 → 0.005 once shuffled).

---

## T22. Agentic track (E) — LLM-written build programs, first live measurements (2026-07-28)

> ### Bottom line
> **The format change delivers the scale it was designed for: 13 live builds,
> 1,140–1,979 mean blocks, 12/13 non-empty and single-component-dominant, 1 failed
> command out of ~430, ~$0.004 per build.** The *scaffolding* ablation (does planning /
> examples / repair / critique help?) has NOT been run — every number below is the
> `oneshot` arm or a single `full`-arm build, so nothing here attributes quality to any
> stage of the loop. Novelty vs the corpus is also still unmeasured.

Track E (`blockgen/agentic/`, notes §22): a frontier LLM emits a WorldEdit-style build
*program*; an executor runs it onto a voxel canvas. Compare against T-series §21, where
the same family of models emitted one line per voxel.

**Live single build.** `gpt-5-mini`, 32³ canvas, plan + 1 in-context example + 1 repair
round + 1 visual-critique round, `--reasoning-effort low`. Run:
`outputs/run_20260728_050025_agentic_live_smoke/`.

| | value |
|---|---|
| prompt | "a small oak cottage with a cobblestone chimney and a steep gabled roof" |
| blocks | **1,070** |
| connected components | **1** (largest-component fraction 1.00) |
| commands / failed / no-op | 43 / **0** / **0** |
| blocks per command | 24.9 |
| clipped writes (outside canvas) | 0 |
| tokens (prompt / completion) | 6,407 / 3,313 |
| cost / wall-clock | **$0.0082** / 35 s |
| rounds kept | generate (1,065 blocks) → critique1 (1,070) |

The render (`samples.png`) reads as a cottage: cobblestone plinth, oak walls, glazed
windows, a steep gabled roof, chimney.

**Prompt-conditioned sample sheets (n=12, one API call each).** `oneshot` arm — one
in-context example, no planning, no repair, no critique — `gpt-5-mini`, 48³ canvas,
`--reasoning-effort low`. Each build is titled with the exact conditioning text in
`docs/assets/agentic_samples_{detailed,short}.png`.

| prompt set | n | mean blocks | coherence | cmd success | failed cmds | mean $/build | total |
|---|--:|--:|--:|--:|--:|--:|--:|
| `detailed` | 8 | **1,979** | 0.875 | 0.996 | 1 / 334 | $0.0043 | $0.034 |
| `short` | 4 | 1,140 | **1.000** | 1.000 | 0 / 124 | $0.0036 | $0.015 |

Per build (`detailed`): suburban house 1,214 (1 comp) · medieval cottage 736 (1) ·
watchtower 1,432 (2) · modern villa 1,078 (1) · brick church 2,782 (7, LCC 0.98) ·
windmill 2,206 (1) · farmhouse+barn 3,337 (5, LCC 0.89) · lighthouse 3,045 (4, LCC 0.92).
Runs: `outputs/run_20260728_084343_agentic_showcase_detailed/`,
`outputs/run_20260728_084550_agentic_showcase_short/`.

**Two readings, one honest and one cautionary.**
- *Detail buys specificity, not just size.* The terse prompts produce generically
  correct buildings; the detailed prompts produce the named features — four sails and a
  shingled cap on the windmill, red/white stripes and a railed gallery on the
  lighthouse, a set-back upper storey and glass front on the villa. This is the
  text-conditioning payoff with no labeled corpus and no training. It is *not yet* the
  paired detail ablation (different subjects in each set), which is what
  `agentic-detail.yaml` exists to run.
- *Fragmentation scales with scene complexity.* Coherence is 1.00 on the four
  single-object terse prompts but 0.875 on the detailed set, and every multi-part
  request (farmhouse **+ barn + fence**, church **+ bell tower**, lighthouse **+ rocky
  base**) is where the extra components appear — detached landscaping and separated
  annexes. LCC stays ≥0.89, so the main mass is intact; the debris is decoration placed
  off the structure. That is the concrete failure mode for the repair/critique arms to
  attack, and the reason the `large` prompt set is a real test rather than a demo.

**Against the per-voxel LLM baseline (§21 / T-series, same prompted-model regime):**

| | per-voxel (§21) | program (T22) |
|---|---|---|
| prompted `gpt-5-mini` parse rate | 0.26 (zero-shot), 0.29 (one-shot) | **command success 1.00** |
| typical build | ~150–200 blocks, "spatially incoherent" | 1,070 blocks, 1 component |
| tokens per block | 10.65 | ~3.1 (3,313 completion tokens / 1,070 blocks) |
| builds fitting a 2k-token cap | 77 / 2,661 | not a constraint |

**Caveats — do not over-read this row.**
- **One model, one arm, no seed replication.** 13 builds total (12 `oneshot` + 1
  `full`), all `gpt-5-mini`, one sample per prompt. The arms exist (`zeroshot … full`)
  and none has been compared, so *no* claim is made about whether planning, examples,
  repair or critique help — nor about model choice.
- **The two prompt sets are not a paired comparison.** `detailed` and `short` describe
  different subjects, so the detail contrast above is illustrative, not measured. The
  paired version (`captions:0` vs `captions:2`, same builds, same seed) is unrun.
- **Parse rate and command-success rate are not the same metric.** The §21 number
  counts voxel lines that survived a vocabulary check; here every emitted command
  executed. The comparable quantity is cost per coherent build, not either rate.
- **Novelty is unmeasured.** NN-IoU against `houses_32` has not been run for this
  track, so "novel" is not claimed — only "coherent" and "large".
- Coherence here is the same 6-connectivity notion the neural tracks report
  (`blockgen/eval/validity.py`), so the component count *is* comparable to T10/T11.
- The DSL's `gable` primitive was fixed during this work (stepped roofs were
  diagonally-connected → 6 components); builds made before that fix are not comparable.

**Offline validation.** 57 tests over the DSL, the agent loop, the response cache and
the run artifacts run with no API key (scripted provider):
`.venv/bin/python -m pytest tests/test_agentic_dsl.py tests/test_agentic_agent.py`.

**Next measurements (in priority order).** (1) `--config agentic-scaffolding` — the
5-arm ablation at n=12; the specific hypothesis to test is whether repair/critique fix
the multi-part fragmentation above. (2) NN-IoU novelty vs `houses_32`. (3) The paired
prompt-detail ablation (`captions:0` vs `captions:2`). (4) `--config agentic-large` —
does the long-horizon claim hold on a 96³ canvas?

---

## T21. Attachment/growth Phase 0 + Phase 1 — representation works, generation does not (2026-07-21)

> ### Bottom line
> **The representation is correct and validated. The model does not generate buildings.**
> Phase 0 passed cleanly; Phase 1 trains, is connectivity-valid, and collapses to
> degenerate geometry (1-voxel lines, flat plates, or spheres depending on ordering).
> Three separate scalars said otherwise and all three were wrong.
>
> **Trust these:**
> - Round-trip encode→decode IoU **exactly 1.0**, 4 orderings × 1,200 builds, 0 failures.
> - **Learned validity 1.00 with exactly one SEED**, sampled **unmasked**, every arm, every
>   temperature, every seq length. A component count, so immune to the metric bugs below.
>   This is the §10 learned-vs-filtered ablation and the learned side wins.
> - **Sequence length causes occupancy** (matched single-variable test, 242→710 median).
>   Retention 70.2% @4096 → 88.4% @8192 → 100% @16384.
> - **`bits/op` is a near-perfect proxy for output solidity (ρ=+1.000 vs thickness), not
>   for quality.** A 1-voxel line is maximally compressible, so NLL *rewards* collapse.
>
> **Do not trust / retracted during this run:**
> `validity 0.154` (decoder wrap-around artifact) · unfiltered occupancy reference ·
> "proxy only ranks, doesn't predict" (it predicts well) · "NLL anti-correlates with size"
> (rejected by 4th arm) · **"radial is best-looking"** (it makes *balls*; CMMD is a solidity
> detector, r=−0.993) · **"temperature fixes the filaments"** (rested on a buggy thickness
> metric; corrected, no ordering reaches real solidity) · seq-16384 arm (I varied batch +
> epochs alongside context — confounded) · every `thickness` number (np.roll wrapped;
> inflated up to +79%) · every "temperature" number (silently also top-k=32).
>
> **The collapse is BLINDNESS, not drift — the §3 state encoder is REQUIRED.**
> `scripts/attach_prefix_test.py` teacher-forces the first K ops of a *real* build and
> lets the model continue. Under drift, a real prefix should hold it on-distribution;
> instead a real prefix makes it **worse**, and the longer the prefix the harder it
> shuts down:
>
> | prefix | thickness | occupancy | model close-rate | **true** close-rate |
> |---|---|---|---|---|
> | *real reference* | *4.24* | *1286* | — | — |
> | 0.00 (free-run) | 3.37 | 321 | 0.659 | 0.606 |
> | 0.10 | 3.02 | 164 | **0.799** | 0.606 |
> | 0.25 | 3.35 | 278 | **0.869** | 0.601 |
>
> The obvious confound — "late ops are naturally CLOSE-heavy in BFS order" — is **ruled
> out**: the ground-truth continuation close-rate is *flat* (0.606→0.601) across the same
> prefixes. So the model's climb to 0.87 is failure, not correct behaviour. Handed more
> real structure to continue, it closes the frontier down faster. Its *marginal* statistics
> are meanwhile near-exact (free-run close-rate 0.659 vs true 0.606; earlier arm 0.337
> attach-rate vs real 0.346) — it has learned the op-frequency distribution but **cannot
> read its own op history as geometry.** That is exactly hypothesis (b).
>
> **Replicated at n=28** (2.8× the original sample, after fixing a crash where corpus-tail
> builds contained pieces absent from the arm's `--vocab-limit` vocab — 3 such builds are
> now skipped and counted rather than silently avoided):
>
> | prefix | thickness | occupancy | model close-rate | **true** close-rate |
> |---|---|---|---|---|
> | *reference (this pool)* | *3.93* | *896* | — | — |
> | 0.00 (free-run) | 3.28 | 475 | 0.693 | ~0.606 |
> | 0.25 | **3.07** | **244** | **0.838** | ~0.601 |
> | 0.50 | 3.67 | 528 | **0.896** | ~0.632 |
>
> Same direction, same magnitude, and now **monotone**: the model's close-rate climbs
> 0.693 → 0.838 → 0.896 as it is handed *more* real structure, while the ground-truth
> continuation rate stays flat (0.606 → 0.632). Thickness never reaches the 3.93 reference
> at any prefix length. The more of a real build the model is shown, the harder it shuts
> the frontier down — the signature of a model that cannot interpret what it is being
> shown. **VERDICT: BLINDNESS, at n=28.** The finding does not rest on the original
> 10-build sample.
>
> Consequence: `implementation_plan.md` §3's graph/state encoder over the placed structure
> is a **prerequisite for Phase 1 working at all**, not the "capacity upgrade" this entry
> earlier claimed. Two prior arguments that it was unnecessary (the filament temperature
> verdict, the size/seq-length result) both rested on the buggy thickness metric and are
> withdrawn.

Run `20260721_071258_ordering_bakeoff` (1,200 builds from `all_32.npz`; human agreement
over 379 3D-Craft houses). New code: `blockgen/utils/attach_order.py` (op extractor +
decoder, ordering pluggable), `blockgen/utils/attach_vocab.py` (op↔token bridge),
`scripts/ordering_bakeoff.py`. Training-free — no GPU.

**Phase-0 gate PASSED.** Round-trip (encode → decode → occupancy IoU) is **1.0000 on
every ordering, every build, 0 failures**, and coverage is **0.996** (the `largest`
multi-component policy discards 0.4% of voxels). `implementation_plan.md` §7's gate for
everything downstream is met.

**The ordering bake-off — the plan's canonical order wins both *proxies*.** (Trained, the
top two swap by 0.5%; see the Phase-1 table below. The layer-structure conclusion survives
either way.)
`implementation_plan.md` §2 asserts bottom-center BFS while notes.md §8/T11 measured
BFS-from-ground as the *worst* non-broken arm for raster AR; the plan waves this off
("the order **is** the generative process"). That argument was untested and load-bearing,
so it was tested before spending GPU:

| ordering | bits/op ↓ | human agreement ↑ | round-trip IoU | ops p50 |
|---|---|---|---|---|
| **bfs_bottom_center** (the plan's) | **2.3244** | **0.3518** | 1.0000 | 3,344 |
| layered_raster | 2.5101 | 0.3516 | 1.0000 | 3,344 |
| dfs | 2.6921 | 0.1255 | 1.0000 | 3,344 |
| radial | 4.5217 | 0.2031 | 1.0000 | 3,344 |

*bits/op* = held-out order-2 context-model compressibility of the op stream (learnability
proxy: how many bits does the next op cost given local context). *human agreement* =
Spearman ρ between the ordering's voxel visit sequence and **real human placement order**
from 3D-Craft (`corpora.load_3dcraft_order`) — the only ground truth available for "how do
people actually build", and the untested notes.md idea #13.

Reads: (a) **T11's anti-BFS finding does NOT transfer** — the plan's reasoning was right.
The distinction that matters is coordinate-emission order (where BFS lost) vs generative
order (where it wins). (b) `radial` is decisively worst at ~2× the bits/op — locality
alone is not the property that matters; **gravity/layer structure is**. (c) `dfs` is
competitive on compressibility but least human-like, so the two proxies are measuring
genuinely different things and should not be collapsed. (d) bits/op and human agreement
**agree on the winner**, which is the strongest form this evidence could take — though
see the ⚠️ perceptual watch item below: both proxies may be ranking *learnability* rather
than sample quality, and `dfs`/`radial` behave very differently once sampled.

**Three findings the plan does not contain:**

1. **The decode-availability constraint.** An ordering must be a pure function of what the
   *decoder* also knows — face coordinates and the partial structure — and may NOT read
   ground-truth occupancy. Two natural candidates (`support_first`: "prefer children
   resting on occupied cells"; `shell_first`: "exterior before interior") scored the
   frontier against the *final* build at encode time but the *partial* build at decode
   time, so the heap desynced and round-trip IoU fell to **0.23**. This rules out any
   ordering defined over properties of the finished structure, and is why the shipped set
   is all static. State-dependent orderings need a pop-time re-evaluating frontier
   (stale-key problem) — deferred, not impossible.
2. **The op stream is MORE compact than raster**: **2.91 ops/voxel** vs raster's 4
   tokens/voxel, despite 65% of ops being `CLOSE`. The representation is not paying a
   sequence-length penalty for dropping coordinates — it is *saving* one.
   Still long in absolute terms: ops p50 3,344 / p90 7,640. Measured retention on the
   4,000-build training pool: **70.2%** at seq 4096, **88.4%** at 8192 (a 400-build
   pilot estimated 60%/91.5%; the trainer's figures supersede it). The box is dissolved
   in the *representation*; a length budget re-enters through the transformer — and that
   budget turned out to be the binding constraint on sample size (see the seq-8192 result).
3. **Validity is not 1.0 *by construction* — but it is ~1.0 *learned*.** Pose derivation
   makes any single component connected, yet nothing structurally stops the model emitting
   a second `SEED`, which would start a new component. Banning `SEED` after position 0
   would force validity to 1.0 as a decode-time **filter** (the §10 artifact confound), so
   all sampling here runs **unmasked**. Empirically the trained model emits **exactly one
   SEED in 36/36 samples across six temperatures** (`seeds_mean` 1.00, `components_mean`
   1.00) — so the connectivity claim is *earned*, not filtered. This is the DiGress-style
   learned-vs-filtered ablation, and it comes out favourably.

   ⚠️ **Correction.** An earlier version of this entry reported learned validity 0.154 /
   2.23 components from the first arm. That was a **decoder bug, not a model property**:
   the decoder wrote into a preallocated 256³ array, and a free-running model can grow
   past it. Large overruns raised `IndexError` (samples dropped), but small negative
   indices **silently wrapped around** numpy-style and scattered phantom blocks on the
   opposite face of the grid — manufacturing disconnected components out of nothing. The
   decoder is now **sparse** (dict of cells, materialized and cropped at the end), which
   both fixes the artifact and removes the last fixed-size volume in the pipeline. Any
   validity/component number produced before this fix is void.

**Phase-1 headline: sampling temperature, not model capacity, decides whether the
growth model builds volumes or filaments.** The first arm had an excellent held-out loss
(1.077 bits/op) yet its T=1.0 samples were 1-voxel-wide tendrils. Diagnosis
(`scripts/attach_diagnose.py`, arm `bfs_bottom_center`, 6 samples/temp) — *thickness* =
mean occupied 6-neighbour count, the statistic that separates a filament (~2) from a wall
(~4) from a solid (~5+):

| T | attach rate | thickness | occupancy | components | seeds |
|---|---|---|---|---|---|
| **real builds** | **0.337** | **4.14** | **997** | 1 | 1 |
| 0.40 | 0.287 | 5.58 | 271 | 1.00 | 1.00 |
| 0.50 | 0.318 | 4.75 | 451 | 1.00 | 1.00 |
| 0.60 | 0.334 | 3.86 | 437 | 1.00 | 1.00 |
| 0.85 | 0.320 | 3.30 | 534 | 1.00 | 1.00 |
| 1.00 | 0.317 | 2.99 | 516 | 1.00 | 1.00 |

Reads: (a) ~~**Verdict CALIBRATION, not capacity.**~~ **❌ RETRACTED — every thickness
figure in this table is inflated** (the `np.roll` wrap bug, §(1)/(4) above), and the
inflation is largest for exactly the thin structures at issue. Corrected,
`bfs_bottom_center` at T=0.5 is **3.30 vs a 4.06 reference**, not "5.11 vs 4.14" — it never
reached real solidity at any temperature. The accompanying claim that the §3 graph encoder
"was tested and not supported" is likewise withdrawn: the prefix diagnostic later returned
**BLINDNESS**, making that encoder a prerequisite (see the verdict box at the top of T21).
Temperature does move thickness — the sweep's *relative* ordering survives — but it does
not fix the failure. (b) **Held-out NLL is a poor guide to sample quality in this
representation**, and every future arm must report **corrected** thickness alongside loss.
The mechanism is not merely compounding error: `bits/op` correlates **+1.000** with output
solidity across orderings, because a 1-voxel line is the most compressible op stream — the
loss-minimising output is degenerate (§5).

(c) **Temperature moves along a thickness↔size frontier, and the real-build point is off
it.** Best-thickness T=0.5 gives (4.75, occ 451); best-size T=0.85 gives (3.30, occ 534).
No setting reaches real-build (4.14, 755) jointly — the model can match or *exceed* real
solidity, but not at real size. Size is therefore a genuine model/data deficiency, not a
sampling artifact.

(d) ⚠️ **The `max_seq_len` filter is a silent selection effect — it halves median build
size.** Over 600 corpus builds: all builds occ_p50 **1,058**; builds passing the seq-4096
filter the trainer applies, occ_p50 **755** (65% kept, thickness unchanged at 4.2). So the
correct reference for a seq-4096 arm is **755, not ~1,000** — an earlier draft of this
entry compared against the unfiltered figure and overstated the gap. More importantly,
**"the box is dissolved" is only half-true at this setting**: a 32³ *volume* constraint was
replaced by a *length* constraint that preferentially discards large builds, which is the
same selection pressure in a new coordinate system. Whatever the representation permits,
the trained model still never sees the big builds.

**Prediction to check against the queued seq-8192 arm** (stated in advance): raising
`max_seq_len` 4096 → 8192 should raise the retained fraction (~70% → ~91%), raise the
*training* occ_p50 toward 1,058, and therefore raise generated occupancy above 451 — with
thickness roughly unchanged, since thickness was already temperature-controllable and is
invariant to the filter (4.22 vs 4.21). If generated size does **not** move, the cause is
the model, not the data filter.

> ### ✅ PREDICTION CONFIRMED on all four components — the size gap was the DATA FILTER, not model capacity
>
> Run `20260721_074238_attach_growth_seq8192`, `bfs_bottom_center`, **matched T=0.5,
> matched sampler, fixed sparse decoder** — the only variable is `max_seq_len`:
>
> | | seq 4096 | seq 8192 | predicted |
> |---|---|---|---|
> | builds retained | 70.2% | **88.4%** | ~91% ✓ |
> | trainable ref occ_p50 | 806 | **1,120** | toward 1,058 ✓ |
> | **generated occupancy** | 242 | **710** | ">451" ✓ (**2.9×**) |
> | generated thickness | 5.11 | 4.76 | ~unchanged ✓ (ref 4.21) |
> | val loss (bits/op) | 1.0773 | **0.9416** | — (−12.6%) |
> | learned validity / seeds | 1.00 / 1.00 | **1.00 / 1.00** | — |
>
> Generated size **nearly tripled** while thickness stayed at the real-build reference and
> validity stayed at 1.00. In *relative* terms the arm also closed on its own target:
> 242/806 = 30% → 710/1,120 = **63%**. So the Phase-1 "builds solid but small" deficiency
> was substantially an artifact of the sequence-length filter silently selecting small
> builds — exactly the mechanism flagged in (d) — and **not** a capacity limit needing the
> §3 graph encoder.
>
> This is the "dissolve the box" thesis earning its keep in the right way: the binding
> constraint was the length budget, and relaxing it improved generation directly. **Next
> lever: seq 16384**, queued as run `attach_growth_seq16384`. Report retention,
> trainable-reference occupancy, and generated occupancy together every time — the three
> move as one.
>
> **Expect diminishing returns, and the measured retention says so.** Actual seq-16384
> retention is **92.3%** (3,692/4,000), not the ~97% predicted, and the training median
> moves only 2,182 → 2,284 ops. So 8192→16384 buys **+3.9 points** of retention against
> 4096→8192's **+18.2**. Revised prediction, stated before the result: generated median
> occupancy should rise above 710 but by *far* less than the previous 2.9×, and the lever
> is close to exhausted — the residual gap to the 1,120–1,228 reference will need
> something other than sequence length (capacity, epochs, or the §3 encoder).
>
> ⚠️ **The seq-16384 arm is CONFOUNDED and does not test this — my error.** To fit the
> longer context I dropped `batch_size` 2→1 and `epochs` 24→20 in the same run, so it
> varies three things at once, not one. Its result (val **1.0375** bits/op, worse than
> seq-8192's 0.9416; in-run occ_p50 373 ≈ seq-8192's 372) is therefore **uninterpretable
> as a sequence-length result** — the loss regression is at least as likely to come from
> the shorter, smaller-batch training. Retention 92.3% is the only clean number in it.
> A valid test needs batch and epochs held fixed at the seq-8192 settings (gradient
> accumulation to keep the effective batch at 2). The confirmed 4096→8192 result above is
> unaffected: that comparison held ordering, temperature, sampler, decoder, batch and
> epochs constant, with `max_seq_len` the sole variable.
>
> **Attach-rate calibration also lands.** The seq-8192 arm at T=0.5 samples at
> `attach_rate` **0.337** against a real-build **0.346** — within 3%, versus 0.298 for the
> seq-4096 arm. More trainable data calibrated the per-face CLOSE/ATTACH decision, which
> is the mechanism by which size recovered.
>
> ⚠️ **Measurement caveat — generated occupancy is right-skewed, report the median.**
> The same arm at the same temperature gives **median 710** (`attach_resample`, n=16) but
> **mean 1,272** (`attach_diagnose`, n=6): a few very large samples drag the mean above
> even the reference median. The 4096→8192 comparison above is **median-to-median** and so
> is internally valid, but the two scripts report different statistics and must not be
> quoted against each other. Prefer the median; quoting the mean would inflate this result
> into "exceeds real builds", which the median does not support.

**Sample quality by ordering, fixed sparse decoder, T=0.5** (16 samples/arm;
`scripts/attach_resample.py`). Reference row is the **trainable** slice (seq≤4096), which
is the distribution these arms actually saw:

| ordering | bits/op | thickness | occupancy | validity (unmasked) | seeds |
|---|---|---|---|---|---|
| bfs_bottom_center | 1.0773 | **5.11** | 242 | **1.00** | 1.00 |
| radial | 2.9989 | 4.44 | 287 | **1.00** | 1.00 |
| layered_raster | **1.0723** | 3.94 | 324 | **1.00** | 1.00 |
| dfs | 1.5116 | 3.55 | **1354** | **1.00** | 1.00 |
| *reference (trainable)* | — | *4.20* | *806* | *1* | *1* |

Replicated at T=0.6 (same ranking, same conclusions): `bfs_bottom_center` 4.77/314,
`radial` 4.46/296, `layered_raster` 3.69/366, `dfs` 3.85/**1334**.

1. **Learned validity is 1.00 across every ordering, with exactly one SEED in every
   sample.** Holds over **8 configurations** (4 orderings × T∈{0.5, 0.6}), plus the six
   temperatures of the diagnose sweep and the seq-8192 arm — every sample generated after
   the decoder fix, without exception. This is the cleanest result of the run:
   connectivity is *earned*, not filtered, and it is robust to both the linearization and
   the sampler. It is the §10 learned-vs-filtered ablation, and the learned side wins
   outright.
2. **A thickness↔size tradeoff runs across orderings, monotonically** (5.11/242 →
   4.44/287 → 3.94/324 → 3.55/1354). No arm reaches the reference joint point
   (4.20, 806): `bfs_bottom_center` beats reference thickness at ~30% of reference size;
   `dfs` beats reference size (1354 > 806) at 85% of reference thickness.
3. ❌ **Hypothesis rejected: NLL does not anti-correlate with build size.** An interim
   read of two arms (bfs small + low loss, dfs large + higher loss) suggested easier-to-
   model orderings terminate early and build small. The full four-arm data kills it:
   `radial` has by far the **worst** loss (2.9989) and still builds **small** (287). `dfs`
   is a lone outlier, not a trend — plausibly because depth-first growth extends a tendril
   instead of closing out a layer, so its frontier survives far longer. Recorded because
   the interim version was stated out loud before the data was complete.

### 🚨 READ THIS FIRST — the renders say no arm produces buildings, and the `thickness` metric was buggy

Two problems found by *looking at the samples*, after all the numeric tables below were
written. They do not invalidate the validity or sequence-length results, but they change
what the quality numbers mean.

**(1) `thickness` was computed with `np.roll`, which WRAPS.** Builds are cropped to their
bounding box, so boundary voxels always sit on the array faces and wrapped around to count
each other as neighbours. Verified: two voxels at opposite corners scored **1.00** instead
of 0.0; a 1-voxel line spanning the box scored ~3.9 instead of ~1.8. **Every thickness
figure in this entry is inflated**, and the inflation is *largest for exactly the thin,
box-spanning structures the metric was supposed to catch* — so it systematically hid the
failure mode it existed to detect. Fixed (zero-padding; `scripts/attach_diagnose.thickness`)
and re-verified against known shapes: line 1.88, solid cube 5.40, disjoint voxels 0.00.
Corrected real-build reference: **4.06** (was 4.21 — small for dense real builds, large for
thin generated ones). Re-measurement of the arms is running.

**(2) The samples are not houses — and the two "best" arms fail in opposite ways.**
Looking at `samples_T05.png`:
- **`layered_raster`** (best loss, worst CMMD) emits **straight 1-voxel diagonal lines**.
  Its reported thickness of 3.94 was almost entirely the wrap-around artifact.
- **`radial`** (worst loss, best CMMD) emits **solid blobs and literal spheres**. That is
  mechanistically unsurprising in hindsight: `radial`'s priority is squared distance from
  the seed, so its frontier expands as a spherical shell and a model that keeps attaching
  fills a ball. **The ordering imposes the shape.**

So CMMD's preference for `radial` over `layered_raster` is *directionally correct* — a
solid volume really is closer to a Minecraft build than a 1-voxel line — but "best" here
means **least bad**, not good. **No arm produces house-like structure.** Phase 1 is a
working representation with a trained model that does not yet generate buildings, and any
claim below phrased as an ordering "winning" should be read in that light.

**(3) The seq-8192 "2.9× size" win is real numerically but NOT qualitatively.** Its
`samples_T05.png` shows mostly **flat slabs and long thin diagonal sheets** — the extra
occupancy went into *bigger plates*, not into structure. One sample (#3) is a genuine
foundation outline with walls and a grass floor, which is the most building-like output
produced all night, but it is 1 of 16. So occupancy misled in exactly the way thickness
did: **a scalar went up while the thing it was standing in for did not.** The
sequence-length → occupancy causal claim stands (it was a matched single-variable test);
the implication "therefore better builds" does not.

**(4) Corrected thickness — re-measured, T=0.5, n=16/arm.** The inflation was severe and
*selective*, hitting the thin arms hardest, which is why it hid the failure:

| ordering | thickness (as reported below) | **corrected** | inflation | occupancy | renders show |
|---|---|---|---|---|---|
| radial | 4.44 | **4.31** | +3% | 272 | solid spheres / blobs |
| dfs | 3.55 | **3.44** | +3% | 1799 | sprawling thin sheets |
| bfs_bottom_center | 5.11 | **3.30** | **+55%** | 294 | flat plates, thin wedges |
| layered_raster | 3.94 | **2.20** | **+79%** | 369 | 1-voxel diagonal lines |
| **real builds** | 4.21 | **4.06** | +4% | 806 | — |

The corrected metric now agrees with the renders: `layered_raster` at **2.20** is barely
above a pure 1-voxel line (1.88), and `radial` at 4.31 is genuinely solid. Consequences:

- ❌ **"Temperature fixes the filaments" is RETRACTED.** That verdict rested on
  `bfs_bottom_center` scoring 5.11 at T=0.5 vs a 4.14 reference — i.e. "exceeds real
  solidity". Corrected, it is **3.30 vs 4.06**, comfortably *below* reference. Temperature
  moves thickness (the sweep's relative ordering is still real), but **no temperature
  reaches real-build solidity for any layer-structured ordering.** The filament problem was
  never solved; the metric only made it look solved.
- The one arm at real-build thickness (`radial`, 4.31) achieves it by generating **balls**,
  which its own ordering geometrically imposes. Matching a scalar is not matching the shape.

**(5) The "inversion" is not mysterious — it has a mechanism, and CMMD is a solidity
detector.** With corrected thickness in hand, the four scalars line up almost perfectly:

| relationship | Spearman | note |
|---|---|---|
| bits/op vs corrected thickness | **+1.000** | perfectly monotone |
| corrected thickness vs CMMD | −0.800 (**Pearson r = −0.993**) | near-linear |
| bits/op vs CMMD | −0.800 | the reported "inversion" |

Read jointly: **`bits/op` is a near-perfect proxy for how solid an ordering's output is, and
CMMD is a near-perfect (r = −0.993) proxy for the same thing.** They "disagree" only in
sign. The mechanism is straightforward once seen — *a 1-voxel line is an extremely
compressible op stream*. `layered_raster` attains the best loss because the model finds a
**degenerate low-entropy mode** (emit a line) and that mode genuinely minimizes NLL;
`radial` has the worst loss because solid volumes require long, varied, hard-to-predict
ATTACH runs. So the ordering ranking by loss is really a ranking of *how easy it is to
cheat*, and CMMD is measuring solidity rather than anything architectural. Neither is
evidence about house-ness, and the "loss disagrees with the eye" framing overstates it:
both metrics agree with each other and with thickness; **none of the three measures whether
the output is a building.**

**(6) The CMMD floor is strongly sample-size dependent — the n=24 reading flattered every
arm.** Floor (real-vs-real) is **0.469 at n=24/ref=48** but **0.100 at n=64/ref=96**. So
in floor-multiples: `radial` 0.980→**10.7×** floor (not the "≈2× floor, essentially
realistic" claimed earlier), `bfs` 28.9×, `dfs` 29.9×, `layered_raster` 48.4×. **All four
arms are far from the real distribution**; `radial` is least-bad by a wide margin but is
nowhere near real builds. Always report CMMD as a multiple of a floor computed at the
*same* n, never as an absolute.

**(7) Every "temperature" number in this entry is actually temperature × top-k-32.**
`VoxelTransformerAR2.generate` defaults to `top_k=32` and no sampling call here overrode
it. Constant across arms, so the *comparisons* stand — but the "CALIBRATION not capacity"
framing is confounded: top-k truncation independently suppresses the distribution tail,
and the tail is where varied ATTACH patterns live. A model pushed toward a degenerate
low-entropy mode by top-k is not the same finding as one that is intrinsically
mis-calibrated. Re-run the temperature sweep with `top_k=None` before treating any
calibration claim as settled.

**Standing lesson for this track: look at `samples_*.png` BEFORE writing any number down.**
Tonight three separate scalars (thickness, occupancy, CMMD) each moved in the "right"
direction while the renders showed lines, plates, and spheres respectively. The render is
the ground truth; the scalars are hypotheses about it.

### 🔴 The perceptual metric INVERTS the loss ranking — the bake-off measured learnability, not quality

**Confirmed at n=64 × 3 seeds.** The table below is the original n=24 reading; the
replication (n=64/arm, 96-build reference, seeds 1–3) reproduces the ranking tightly and
supersedes it:

| ordering | CMMD s1 | s2 | s3 | mean | **× floor** | CLIP (mean) |
|---|---|---|---|---|---|---|
| radial | 1.071 | 1.073 | 1.172 | **1.105** | **10.9×** | 0.266 |
| dfs | 2.993 | 2.823 | 2.980 | 2.932 | 28.9× | 0.198 |
| bfs_bottom_center | 2.888 | 2.880 | 3.079 | 2.949 | 29.1× | 0.209 |
| layered_raster | 4.840 | 4.879 | 4.860 | 4.860 | 47.9× | 0.172 |
| **real-vs-real floor** | 0.100 | 0.105 | 0.102 | **0.102** | 1× | 0.275 |

Seed-to-seed spread is ≤0.20 (`layered_raster` ≤0.04, floor ≤0.005), so the ranking is not
noise — though `dfs` and `bfs_bottom_center` are a statistical tie (2.932 vs 2.949, well
inside their own seed spread) and must not be ordered against each other.
But note the **floor multiples**: every arm is 10–47× the real-vs-real floor, i.e. **all
four are far outside the real distribution** — `radial` is merely least-bad, and its
advantage comes from producing solid *balls* (§(2)/(5)), not buildings. The n=24 floor of
0.469 flattered every arm by ~4.5×; always compute the floor at the same n.

Original n=24 reading, `scripts/attach_perceptual.py`, vs a shared 48-build real reference
(drawn from the trainable slice), T=0.5, fixed decoder:

| ordering | bits/op (rank) | **CMMD** ↓ (rank) | **CLIP** ↑ | occupancy | thickness |
|---|---|---|---|---|---|
| radial | 2.9989 (**4**) | **0.980** (**1**) | **0.2675** | 287 | 4.44 |
| dfs | 1.5116 (3) | 2.164 (2) | 0.2051 | 1354 | 3.55 |
| bfs_bottom_center | 1.0773 (2) | 2.470 (3) | 0.2184 | 242 | 5.11 |
| layered_raster | **1.0723** (**1**) | 4.533 (**4**) | 0.1755 | 324 | 3.94 |
| **real-vs-real floor** | — | **0.469** | **0.2774** | — | — |

**The CMMD ranking is an exact inversion of the loss ranking (Spearman ρ = −1.0).** The
best-loss arm (`layered_raster`) is the worst-looking; the worst-loss arm (`radial`) is
the best-looking, at **CMMD 0.980 against a floor of 0.469** — roughly 2× the best
achievable at this sample size — and **CLIP 0.2675 against a floor of 0.2774**, i.e.
essentially at the ceiling for prompt agreement.

**This overturns the ordering conclusion stated earlier in this entry.** The reads above —
"gravity/layer structure is what matters, locality is not", justified by `radial` being
worst at ~2× the bits/op — hold *only for learnability*. Judged by the metric T20 says is
the only adjudicating one, the conclusion reverses: **`radial`'s locality-driven growth
produces the most realistic-looking builds**, and the layer-structured orderings that are
easiest to model produce the least realistic ones. Both bake-off proxies (order-2 bits/op
*and* human-order agreement, which agreed with each other) therefore rank **learnability**,
and neither is a proxy for sample quality. The cheap CPU screen remains useful for
predicting trained *loss* — it did that well — but it must not be used to choose an
ordering for quality.

This is the T20 lesson recurring in a new representation, and it is exactly why
`eval/perceptual.py` was built: three metrics (bits/op, human agreement, val NLL) agreed
with each other and all three disagreed with the eye.

**Caveats before this is load-bearing.** n=24 samples vs a 48-build reference is small for
CMMD (the module targets small-n, but this is at the low end); single seed; one temperature.
The inversion is large and monotone rather than marginal, which argues against noise, but
**confirm at n≥64 with 2–3 seeds before putting it in the paper.** Note also the ranking is
not explained by the obvious confounds: it does not track occupancy (`radial` 287 vs
`bfs` 242 are similar in size but 2.5× apart in CMMD) nor distance-from-reference
thickness (`layered_raster` is near-reference at 3.94 yet scores worst).

**The raster baseline to beat, and how to compare fairly.** From
`run_20260715_065938_native/rows.json` (`native_bpe` arm) and
`run_20260721_013119_native_oriented_native`:

| | native_bpe (T18) | native_bpe oriented (2026-07-21) |
|---|---|---|
| validity_rate (**ungated**) | **0.188** | **0.062** |
| validity_gated | 1.0 (filter) | 1.0 (filter) |
| median_sample_occ | 942 | 750 |
| final **train** loss | 0.287 (underfit, still falling) | 0.311 |

The clean comparison is **ungated raster validity (0.188) vs unmasked growth validity**:
neither side uses a filter, so it isolates what the *representation* buys. If growth
lands well above 0.188 without a mask, that is the connectivity claim earned rather than
filtered — and it is the number `implementation_plan.md` §6 should have specified instead
of "≈1.0 by construction".

Two caveats that must travel with any such comparison: (a) the raster runs trained on the
augmented **houses** corpus (n_train 16.9–17.2k) while the growth arms train on
`all_32` (4,712 builds, 70% of which fit seq 4096) — **different data, so this is
indicative, not a controlled ablation**; (b) `native_bpe` logs *train* loss only, no
held-out NLL, so bits/build cannot be compared against it without a re-run. A controlled
version needs both models on one corpus with one eval — that is the flagship ablation of
§6 and is not yet run.

**Phase-1 trained result — all four arms** (run `20260721_071702_attach_growth_ordering`;
identical corpus, budget, vocab construction and seq cap, only the linearization differs;
2,808/4,000 builds fit seq 4096 in every arm):

| ordering | **trained** bits/op ↓ | proxy bits/op | proxy rank | trained rank |
|---|---|---|---|---|
| layered_raster | **1.0723** | 2.5101 | 2 | **1** |
| bfs_bottom_center | 1.0773 | 2.3244 | 1 | 2 |
| dfs | 1.5116 | 2.6921 | 3 | 3 |
| radial | 2.9989 | 4.5217 | 4 | 4 |

**The training-free proxy substantially works.** It recovers the trained ranking exactly
except for an adjacent swap of the top two, which are separated by **0.5%** (1.0723 vs
1.0773) — i.e. the only thing it got "wrong" is a near-tie. It correctly identified
`radial` as far worse (borne out at 2.8×) and `dfs` as mid-pack. So a ~5-minute CPU
screen predicted the outcome of four GPU training runs. Use it to *screen* candidate
orderings before spending GPU; do not use it to separate near-ties, and note it does not
predict achievable loss (proxy values are ~2× the trained ones throughout).

**Substantive read on ordering.** The two gravity/layer-structured orderings are
statistically indistinguishable and both clearly beat `dfs` (+41%) and `radial` (+180%).
So the property that matters is **finishing a horizontal layer before climbing** — not
locality per se (`radial` is maximally local and worst), and not depth-first tendril
growth. `implementation_plan.md` §2's instinct is vindicated at the level of *layer
structure*; its specific choice between BFS-in-plane and raster-in-plane is not resolved
by this evidence and does not appear to matter.

---

## T20. §9.0 native resolution — **NEGATIVE. The decimator thesis is NOT supported** (2026-07-15)

**Pre-committed read** (stated before the run): compare each arm's `val_nn` as a fraction
of **its own** `val_baseline_nn_iou`, because nn_iou is grid-dependent. T12 scratch = 84.4%.

`blockgen/experiments_native.py`, 60 ep (matching T12's `--epochs-finetune 60`), batch 8,
256 merges, 16 samples, seed 0, houses_32 split 2,262 train / 399 val, D4-augmented to
18,052. Everything else held: same cache, lr, sampler; factored embeddings OFF.

| arm | val_nn | baseline | **ratio** | valid | v/train | dup | occ | final_loss | n_train | min |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **canon16_flat** (control) | 0.368 | 0.442 | **83.3%** | 0.312 | **0.732** | 0.0 | 218 | 0.174 | 12,728 | 30 |
| **native_bpe** (intervention) | 0.304 | 0.403 | **75.4%** | 0.188 | 0.315 | 0.0 | 942 | 0.287 | 17,163 | 210 |

**Verdict: the intervention lost by ~8 points.** Native 32³ + 3D-BPE did not beat canon-16
+ flat. The control reproduces T12 well (83.3% vs 84.4%), so the harness is sound.

**What this does and does not overturn.** T17's *measurement* stands — canon-16 destroys
86.5% of blocks and halves connectivity (F16 is a photograph of it). What is **not**
supported is the *inference* that repairing it improves generation. Different claims.

**Why the run cannot answer the real question (design error, acknowledged):**
1. **Confounded with a known-bad variable.** Arm B moved resolution AND tokenizer. T10
   already found cluster-AR trails flat *on houses* (0.378 vs 0.485, "big pieces placed
   disconnected, raw validity 0"). Arm B reproduces that ordering exactly (0.304 vs 0.368,
   validity 0.188, occ 942 vs a real 745 median, one sample at occ=**2**). The loss may be
   entirely the tokenizer, with resolution's effect invisible underneath. T18 gives the
   mechanism: pieces are flat plates up to 96 voxels; one bad anchor = a floating slab.
   The de-confounding arm (**native + flat @ seq 8192**, 88.7% fit) was unaffordable when
   this was designed and became cheap once the SDPA fix landed — it should have been
   re-planned then.
2. **Arm B is underfit.** Loss 0.287 and still falling vs the control's converged 0.174;
   210 min vs 30. Same 5M params against 2.5× the sequence and 4.6× the content. Cf.
   T13/T14, where the 91.7M model was stopped with val loss still dropping.
3. **The metric may not be cross-grid comparable — including via the ratio.** nn_iou is
   occupancy-IoU shape matching. At grid 16 every house is a coarse blob and blobs overlap
   generously; at grid 32 houses are distinct and matching one is harder. **83% of a
   blob-matching score ≠ 75% of a detail-matching score**, and dividing by `val_baseline`
   (0.442 vs 0.403) does not obviously repair that. This undermines the pre-committed
   normalization itself.

**The eye disagrees with the metric — for the THIRD time today, and that is the finding.**
Arm B's samples show the first genuine architecture in this project (pitched roofs, walls,
window openings; `native_bpe/samples.png`) against the control's house-shaped blobs. **This
is deliberately NOT used to overturn the result**: arm B renders at native 32³ with 4.6× the
voxels, so it looks more detailed regardless of model quality. But note the pattern —
T17: val_nn said 84% while samples were rubble. T20: val_nn says worse while samples look
better. The common factor is nn_iou.

**Blocking next step:** a **perceptual metric** (Render-FID / CLIP-on-renders, research.md
§A). Recommended 2026-07-15 as a nice-to-have; it is now the blocker — arm A vs arm B is
not adjudicable without a number that tracks what we see, and the factored-embedding A/B,
PMI merges, and any DPO all inherit this.

**Then:** (a) native+flat @ 8192 to de-confound; (b) arm B to val plateau (its number may
mean "underfit", not "native is worse"); (c) BPE-aware constrained decoding —
`sample_constrained_structures` is BlockVocab-only and cannot gate piece tokens, so arm B
ran unconstrained while T11's validity-1.0 knob sat unused.

**Artifacts:** `outputs/run_20260715_065938_native/` (rows.json, per-arm model.pt,
samples.png, comparison.png).

---

## T19. minecraft-schematics.com label recovery — `data/raw` was labeled all along (2026-07-15)

**The reversal.** `notes.md` §8 retired `data/raw` as unlabeled ("filename → metadata:
verified false"). That test joined the files against the **PlanetMinecraft tfrecord**
metadata — the wrong corpus. `data/download.py` walks **minecraft-schematics.com** ids
1824→19000, and the files are named `<id>.schematic`. Verified: **12,364 id-named files,
range 1..18905** (10,963 `.schematic`, 848 `.schem`, 553 `.litematic`). The filename *is*
the m-s.com schematic id. The 5% content-hash overlap with the tfrecords is just two
unrelated crawls sharing popular builds.

**Route.** The live site is behind Cloudflare and **403s plain HTTP regardless of
user-agent** (`robots.txt` is 200 but **empty** — nothing disallowed; the block is a WAF,
not a stated policy). We read the **Internet Archive** instead — no circumvention:

| quantity | value |
|---|--:|
| local corpus (distinct ids) | 12,366 |
| ids with a 200-status snapshot (CDX) | 21,553 |
| **overlap with local corpus** | **11,842 (95.8%)** |
| archived ids never downloaded | 9,711 |
| snapshot years | 2012–2026, weighted recent (3,911 in 2025) |

**Fields recovered** (smoke test 25/25, all present): `title`, `category`, `theme`, `size`,
`file_format`, `submitted_by`, `posted_on`, `rating`; `downloads` parses on newer snapshots
only. Example: `#10055 → Pixel Art / Modern / Small`, `#3 → Arenas / Medieval / Medium`.

**No new *files* from this route:** the archive has **zero** captures of
`/schematic/*/download*` — Wayback holds pages, never bytes. New schematics would require
driving a real browser past Cloudflare (which is what `data/download.py` does, one page at
a time). Note we already hold ~63% of the site, and T17 says the binding constraint is the
decimator, not build count — so more files is the lower-leverage half.

**Status:** full run in flight (`python -m blockgen.data.recover_ms_labels`). ~27% of
fetches fail as archive.org throttles the IP (diagnosed: `Connection refused`, not parse
failures) — the script is **resumable**, so a second pass at lower `--workers` fills the
gaps.

**Output is QUARANTINED** at `data/minecraft/ms_labels/labels.jsonl`, deliberately **not**
merged into `houses_32`: that cache is the reference for the T11/T12 val_nn 0.405 baseline,
and changing the data underneath it would make §9.0's native-resolution comparison
uninterpretable. Merge after that run reports.

**Why it matters:** category+theme is free, exact, human-authored supervision, and T10
found category conditioning is our best cohesion lever (raw validity best-in-dataset on
gc-houses 0.333 and vehicles 0.7; best val_nn overall on combined 0.435). Prefer it to VLM
captions for conditioning — captions cost money, are inferred, and captioning 12k builds
that curation will cut to ~3k is mostly wasted spend. Caption the survivors, not the pool.

**Licensing:** schematics remain copyright their submitters; this recovers factual
metadata (category/theme/author/date) for research indexing only.

---

## T17. Pipeline audit — the canon-16 decimator is destroying the training target (2026-07-15)

**Question:** unconditioned samples look bad to the eye while `val_nn` says 84% of the
real-build baseline. Which one is lying?

**Answer: neither — they measure different objects.** `val_nn` is computed in the same
degraded space the model trains in, so it scores 84% of a ceiling that is itself rubble.

`Structure.downsample` (`utils/data.py:224`) is strided decimation,
`block_ids[::s, ::s, ::s]` — no pooling, no majority vote. Minecraft houses are almost
entirely **1-voxel-thick** walls, roofs and floors, so a feature survives only if its
index has the surviving parity: decimation deletes walls by coin flip and leaves the
debris. Measured on the real `houses_32` cache through the exact `canonicalize()` the
T11/T12 winning arm uses (n=2,661; 1,685 actually decimated). **No model involved.**

| metric | real 32³ | canon-16 (all) | canon-16 (decimated only) |
|---|--:|--:|--:|
| median blocks | 745 | 236 | 1,083 → **146** |
| median components | 1 | 2 | 1 → **3** |
| single-component rate | 0.592 | 0.428 | 0.534 → **0.277** |
| LCC > 0.95 rate | 0.906 | 0.705 | 0.895 → 0.577 |

**Reading:**
1. **86.5% of blocks destroyed** on the 63% of the cache it touches; connectivity halved.
2. **The model is faithful, not failing.** T11 `ar_pe_phase4` raw validity 0.375 sits at
   its training distribution (canon-16 single-component 0.428). Reading validity against
   an implied 1.0 was the error. (T11 used gc-houses-large, so this is indicative rather
   than an exact pairing — see the doc correction below.)
3. **`houses_32` is only 59.2% single-component at NATIVE resolution.** Validity can never
   approach 1.0 by learning — the curation gate admits detached fences/trees/lanterns.
   The validity denominator was wrong independently of decimation.
4. **Retires the T12 mystery** (§8 "pool-pretrain gives no gain"): all 16,383 pooled builds
   went through the same decimator. Data scaling cannot pay until the model sees a house.
5. Block-agreement is **not** explained by this — decimation is nearest-neighbour, so
   surviving voxels keep true materials. T7's near-orthogonal embeddings remain the
   better-evidenced cause (→ §9.1, now implemented; see T18).

**Native resolution is feasible today** — the sequence-budget premise for downsampling is
false. Fraction of builds fitting `max_seq_len` (4 tok/block flat; 3D-BPE ~2× per T14):

| config | fits 1600 | fits 5480 | fits 8192 |
|---|--:|--:|--:|
| houses_32 native + flat | 16.3% | 76.7% | 88.7% |
| **houses_32 native + 3D-BPE** | 53.4% | **94.8%** | 98.7% |
| houses_48 canon-32 + 3D-BPE | 57.4% | 96.4% | 100% |
| *current: canon-16 + flat* | *70.9%* | *99.8%* | *100%* |

The current regime keeps 70.9% of builds and destroys most of their blocks; native + BPE
keeps **94.8%** at a `max_seq_len` T15 already ran (5,480, 5.7M params, 36 min/arm).

**Fixes landed (2026-07-15):**
- `eval/novelty.py` — `NoveltyReport.train_validity_rate` (same measure on real training
  builds) + `validity_vs_train`; both in `summary_row`. Mirrors the `val_baseline_nn_iou`
  idiom: score against real data, not perfection. Connectivity now uses
  `scipy.ndimage.label` (verified identical to the old pure-Python DFS on
  slab/two-islands/empty; ~100× faster, which is what makes the per-arm ceiling affordable).
- AR loop throughput (`train_ar.py`, `train_ar_ext.py`, `experiments_transfer.py`): bf16
  autocast + on-device loss accumulation (a per-step `.item()` synced the GPU every
  iteration) + `set_to_none` + `pin_memory`. **2.62× measured** at the T11 config
  (23.21 → 8.86 s/epoch), loss trajectory unchanged (1.4711 → 1.4612). `amp=False`
  reproduces pre-2026-07-15 runs exactly. `batch_size` left at 8 deliberately — it is the
  biggest remaining lever but changes the optimization trajectory and would confound the
  comparison against 0.405.

**Open (needs GPU):** retrain `ar_pe_phase4` at native 32³ + 3D-BPE + constrained decoding
vs the 0.405 baseline. Until it runs, "the unconditioned model isn't good" is untested.

### Doc corrections found by this audit

| claim | where | actual |
|---|---|---|
| T11 trained on the curated cross-corpus set | results.md T11 narrative | **GrabCraft only** — `configs/experiments/ideas-full.yaml:9` → `ds_gc_houses_large` → `Curator.from_grabcraft_cache(max_dim=32)`. Zero text2mc, zero 3D-Craft. |
| open interiors: 17% GrabCraft / 54% text2mc | data_sources.md:142 | **25.2% / 67.9%** (measured) |
| `data/raw` unlabeled, "filenames ≠ metadata" | data_sources.md:26, notes.md §8 | **False** — filenames are minecraft-schematics.com ids (see T19) |
| text2mc yields only 34/11,092 = a bug | (working assumption) | **Not a bug.** 11,092 was never the denominator: 3,668 are house-tagged, 410 size-plausible. The `max_dim` cap dominates (text2mc are world cuts, median max-dim 96 vs a 32³ target). `houses_48` lifts it 34 → 180 for free. |

---

## T18. 3D-BPE vocabulary audit — pieces are material patches, not parts (2026-07-15)

256 merges learned on `houses_32` (`tokenizers/cluster_bpe.py`, seed 0, `max_corpus=400`),
then measured and rendered. Motivated by the question "do the learned tokens mean anything?"

| quantity | value |
|---|--:|
| single block-token pieces | **253/256 (98.8%)** |
| single block-family pieces | 253/256 (98.8%) |
| pure-terrain pieces | 36/256 (14.1%) |
| piece size (voxels) | median 4, mean 5.0, max 96 |
| **distinct shapes ↑D4** | **16** |
| D4+family duplicate slots | **105/256 (41.0%)** |

**Reading:**
1. **Same-type merging already happens implicitly** — 98.8% of pieces are a run of exactly
   one block token. `_canon_pair` keys on `(piece_a, piece_b, delta)` and frequency does
   the rest; an explicit same-type bias would change nothing.
2. **The vocabulary is architectural, not terrain** (14.1% terrain — a predicted
   grass-apron takeover did *not* happen). Top families: oak planks (296 cells), dirt,
   grass, wooden slab, cobblestone, **log**, stone brick, quartz, sandstone, stained clay.
3. **The pieces are rectangles.** Canonicalized up to D4, 256 slots encode **16 shapes**.
   The contact sheet (F14) shows flat plates and solid cuboids, a handful of 1×N beams,
   **exactly two vertical pillars**, and zero wall corners / L-shapes / roof slopes —
   against `cluster_bpe.py`'s own docstring promise of "a wall corner … a roof-slope unit".
   Cause is structural: greedy maximization of raw adjacency *count* on a voxel grid always
   grows rectangles, because the most frequent pair is always "extend the rectangle".
4. **On real builds it barely merges at all** (F15, exploded views): **1.2–1.9 voxels per
   piece** (205→110, 300→171, 344→289, 323→245). Most voxels stay atomic. The arithmetic:
   422 block types × 6 directions ≈ 2,500 candidate same-type adjacencies vs **256 merge
   slots** — any material outside the global top-256 never merges.
5. **Corrects T14.** That run reported "2× compression, records/blocks 0.49"; measured here
   the ratio is **0.54–0.84 and strongly build-dependent**. T14's +79% nn_iou was
   attributed to that compression, so the claim needs a range, not a point.
6. **Explains T10's cluster-AR failure** ("big learned pieces get placed disconnected,
   raw validity 0"): the largest piece is a **96-voxel flat plate** — one wrong anchor
   deposits 96 misplaced voxels as a floating slab.
7. **Reframes the §D novelty claim.** research.md leads with "connectivity-native
   tokenization"; this figure supports the *compression* half, not the *semantics* half.

**Implication:** the win available is **factoring**, not same-type biasing. Pieces are
already single-material runs, so the vocabulary decomposes as
`(shape ↑D4) × rotation × family × variant` — and 41% of slots are paying the cross
product. Factored, ~16 shapes + 8 rotations + families span what 256 material-specific
merges structurally cannot, and each factor trains on ~8× more examples.

**Landed (2026-07-15):** `tokenizers/piece_factors.py` (`build_piece_factors(cv)` →
shape/rot/family/variant indices via D4 canonicalization) + `models/factored_embedding.py`
(`FactoredPieceEmbedding`), wired **opt-in** to `VoxelTransformerAR2` via
`piece_factors=`/`piece_offset=` so it stays a clean ablation. Measured on the 678-piece
vocab: `678 pieces → 17 shapes, 8 rots, 239 families, 16 variants`; embedding rows
**713 → 315**; params 5,208,521 → 5,106,633. Gradients verified flowing to all four tables.

**Why factor the embedding rather than canonicalize the tokens:** a log's axis and a
stair's facing live in `block_data`, so token-level rotation canonicalization requires a
correct block-data rotation table — and getting it wrong silently emits wrong-facing
blocks (the tradeoff `utils/augment.py:12-16` already documents and tolerates for D4
augmentation). Factoring the embedding leaves the token stream untouched: a bad
decomposition costs parameter sharing, never correctness.

**Next, in order:** (a) train the factored arm vs flat at matched budget; (b) **PMI /
WordPiece merge scoring** — score by `count(ab)/(count(a)·count(b))` instead of
`count(ab)` at `cluster_bpe.py:201`, so "planks next to planks" (high count, low PMI,
planks are everywhere) loses to "log above log" (rarer but *specific*); the F14/F15 sheets
are the eval — do corners and pillars appear? (c) shape-factored *merges* (material-
agnostic merge rules applying to all 422 materials at once), which attacks the coverage
problem in #4 that no amount of `n_merges` fixes.

---

## T16. Conditioning in the T12 regime (canon 16³, per-voxel, 60 ep) — the regime was the gap (2026-07-11)

**Question (user):** why did T15's conditioned samples look so much worse than
the T11/T12 uncond runs? **Answer:** T11/T12 trained on 16³-downsampled
miniatures (canon_dim=16, ar_seq=1600, 60–90 ep, curated houses; T12 finetune
median sample = 275 blocks); T15 trained at native 32³ (≤5,480 tokens) for 40
epochs on all houses_32 with unconstrained CFG sampling — four hard variables
changed at once. T16 reruns both conditioned arms in the exact T12 regime
(`--repr voxel --canon-dim 16 --max-seq-len 1600`, 60 ep, 1,884 builds fit,
vocab 435; only conditioning added). **7 s/epoch → ~7 min/arm.**

**Val loss:** image 2.95 → **0.741**; text 2.96 → **0.761** (plateaued ~ep 45).

| metric (16 held-out val conds, CFG 3) | image c16 | text c16 | image 32³ (T15) |
|---|--:|--:|--:|
| palette sim paired / shuffled | **0.478 / 0.226 (+112%)** | 0.309 / 0.300 | 0.301 / 0.201 (+50%) |
| EOS rate | 0.81 | 0.88 | 0.31 |
| blocks median | 231 | 239 | 692 |
| components median | 17 | 8 | 64 |

**Reading:**
1. **Sample quality is back to the familiar T12 look** — compact builds on
   tidy grass plinths, wall/roof massing, T12-scale occupancy (231–275) — now
   *steerable*. Best examples: "longhouse with thatched roof, cobblestone
   walls, oak log supports" → a longhouse-shaped wooden roof plane over
   cobblestone walls with oak supports; image-conditioned A-frame lodge → two
   compact wooden houses with stone foundations.
2. **Image palette fidelity doubled** vs 32³ (paired/shuffled gap +50% → +112%).
3. **Text metric saturates** (0.309 vs 0.300) despite visible caption-material
   matching in renders — val captions are homogeneous medieval wood/stone
   houses, so the shuffled floor is already palette-correct; per-instance
   discrimination needs richer text prefixes (CLIP token sequences) or more
   varied categories (vehicles/statues arms).
4. Remaining gap to T12's *constrained/gated* figures: connectivity (0–6%
   raw). Compose T11 in-loop adjacency constraint with CFG next.

**Figures:** `outputs/cond/showcase_{image,text}.png` (input | 2 samples per
row), `outputs/figures/cond_c16/*`. Runs: `outputs/cond/{image,text}_run_c16/`.

---

## T15. LegoACE-style conditioning on OUR stack — image (DINOv2) & text (CLIP) prefix runs (2026-07-10)

**Setup (pivot after T13/T14):** the LegoACE *conditioning recipe* (frozen
encoder → linear projection → prefix tokens → cond-dropout → CFG) ported onto
our proven backbone instead of their 92M GPT-2/LLaMA: **CondVoxelAR2** =
VoxelTransformerAR2 (phase4 PE, T11 winner) + cond prefix + learned null-cond,
**5.7M params**, over 3D-BPE piece sequences (T14 winner). Trained on **all
houses_32** (2,530/2,661 fit max_seq_len 5480 = p95 of piece lengths; +
learned-null CFG branch via 10% cond-dropout). Conditions precomputed once
(`labeling/embed_conditions.py`): DINOv2-base CLS ×4 views (prefix 4×768),
CLIP ViT-B/32 pooled caption embeds (prefix 1×512; 3 VLM + 1 template caption
per build). 40 epochs, batch 8, lr 3e-4 cosine — **~36 min per arm** (vs ~1.5 h
for the 92M MinecraftACE runs). Deviation from LegoACE: CLS/pooled embeds
(4 / 1 prefix tokens), not full patch/token sequences (their 257×4) — right
scale for a 5.7M decoder.

**Val loss:** image 2.996 → **1.238**; text 2.978 → **1.250** (40 ep, smooth,
plateau at cosine end — well-fit).

**Conditioning fidelity (16 held-out val conditions, CFG 3.0, paired vs
shuffled pairing = chance floor):**

| metric | image arm | text arm |
|---|--:|--:|
| palette sim paired / shuffled | **0.301 / 0.201 (+50%)** | 0.229 / 0.191 (+20%) |
| occupancy IoU paired / shuffled | 0.083 / 0.061 (+36%) | 0.080 / 0.088 (none) |
| EOS rate | 0.31 | 0.50 |
| components median | 64 | 31 |

**Reading:**
1. **Conditioning works, palette-first.** Samples visibly adopt their target's
   materials (black-roof target → black+brown sample; brick → red brick; stone
   tower → grey stone; white-roof → white slabs) — confirmed by the paired-vs-
   shuffled palette gap. Geometry transfer is weaker (image arm +36% IoU over
   chance; text arm none) — expected at 5.7M/36 min with pooled embeddings.
2. **Image > text**, matching LegoACE's own ordering; a single pooled CLIP
   token is a thin channel, and val captions are homogeneous ("medieval house
   with gabled roof…") so there's little discriminative signal to exploit.
3. **EOS now works** (0.3–0.5 vs ~0 for the 92M MinecraftACE runs at similar
   budget) — small model + short piece sequences learn termination fast.
4. Structures still blobby/fragmented (connected 0%) — apply the T11
   constrained-decoding knob (in-loop adjacency masking) and/or longer training;
   both are orthogonal to conditioning.

**Next levers, in expected-value order:** (a) richer prefixes (DINOv2 patch
tokens / CLIP token sequence); (b) bigger decoder (d512/12L still ≪ 92M);
(c) adjacency-constrained sampling from T11 composed with CFG; (d) CFG-scale
sweep; (e) block-agreement-aware fidelity metrics.

**Figures:** `outputs/figures/cond/{image,text}_cfg3_target_vs_sample.png`
(rows alternate conditioning target, sample). Code: `models/voxel_transformer_cond.py`,
`training/train_conditioned.py`, `labeling/embed_conditions.py`,
`scripts/sample_conditioned.py`.

---

## T14. MinecraftACE × 3D-BPE pieces — matched 15-epoch comparison vs T13 (2026-07-10)

**Motivation:** LegoACE's per-record compression comes from its physical part
library (one 2×4-brick token = 8 cells); a direct Minecraft port loses that
(block types are materials, not shapes). Our 3D-BPE cluster tokenizer
(`cluster_bpe.py`) is the *learned* analog: `--tokenizer bpe` in the converter
emits `(x, y, z, piece_id)` with 678 pieces (422 atomic + 256 merges learned on
the train split only, replayed deterministically). Same 4-token grammar, same
`%4` masking, vocab 712. Round-trip token-exact.

**Data effect:** 2× sequence compression (median 745 blocks → 418 pieces;
records/blocks ratio 0.49) → only 43 builds filtered (vs 300) → **+257 training
builds** (2,357 train). Same model/budget as T13 (GPT-2 91.7M, 15 ep, batch 1 ×
accum 16, lr 1e-4 cosine, sdpa bf16). Final ckpt-2220 (final-save patch worked).

**Val loss:** 4.31 → 1.14 (piece-token CE; NOT comparable to T13's 0.66 —
different vocab & sequence length; compare at sample level).

**Samples (64, final ckpt, same eval as T13):**

| metric | T13 voxel | **T14 3D-BPE** | Δ |
|---|--:|--:|---|
| mean_nn_iou | 0.111 | **0.199** | +79% |
| connected_rate (raw) | 0.000 | **0.047** | first nonzero |
| components median | 27 | **10** | −63% |
| diversity | 0.904 | 0.963 | + |
| duplicate_rate | 0.0 | 0.0 | = |
| unique blocks median | 458 | 474 (max 2,444) | pieces expand past the record cap |

**Qualitative (the headline):** T13 produced only flat ground planes; T14
produces **vertical structure at the same budget** — multi-story wooden facades
with window openings, a walled courtyard, stone masses/towers, a tree with
foliage, glass frames. The 2× shorter sequences let the model get past the
terrain prefix into walls/roofs within 15 epochs, and every emitted token being
a connected multi-voxel chunk kills the single-voxel noise mode.

**Still missing at this budget:** EOS (62/64 hit the 2,047-record cap),
connectivity 4.7% ≪ LegoACE's converged ~82%. Same prescription as T13: train
to val plateau (val still falling at 1.14).

**Verdict: 3D-BPE pieces are the right representation for the MinecraftACE
track** — adopt as default; voxel arm kept as ablation baseline.

**Figures:** `outputs/figures/minecraftace/uncond_bpe2220_{samples,real_ref}.png`.

---

## T13. MinecraftACE (LegoACE port) — uncond pipeline test on houses_32 (2026-07-10, ~1.3 h train)

**Setup:** LegoACE native tokenization ported to Minecraft (notes.md §14): 4-token
records `(x,y,z,type)`, vocab 456, GPT-2 12L/768 (91.7M, n_positions 8192, sdpa,
bf16), houses_32 export (2,127 train / 114 val, n_blocks ≤ 2047), batch 1 × accum
16, lr 1e-4 cosine, **15 epochs** (deliberate short pipeline-validation budget),
`%4` grammar masking at sampling (top_k 10, top_p 0.95). Eval on checkpoint-1500
(epoch ~11.3; script didn't save a final checkpoint — now fixed).

**Val loss (per-token CE):** 3.98 → 3.22 → 2.73 → 2.36 → 1.96 → 1.50 → 1.25 →
1.05 → 0.92 → 0.83 → 0.77 → 0.74 → 0.71 → 0.68 → **0.66** (epoch 14) — still
decreasing at end; **no overfit inside this budget** (more epochs warranted).

**Samples (64, ckpt-1500, grid=32 eval vs full houses cache):**

| metric | value | T11 best AR (ref) |
|---|--:|--:|
| mean_nn_iou | 0.111 | 0.454 |
| duplicate_rate | 0.0 | 0.0 |
| diversity | 0.904 | — |
| connected_rate (raw, no constraint) | 0.0 | 0.062 |
| components median | 27 | — |
| unique blocks median (of 2,047 emitted) | 458 | ~211 occ |

**Reading (this is a pipeline test, not a converged model):**
1. **Pipeline validated end-to-end** — export → train → grammar-constrained
   generate → decode → render → eval all work; every generated sequence parses
   as clean quadruples by construction.
2. **The model learned the sequence prefix first**, as AR predicts: samples are
   coherent *ground planes* (grass terrain slabs w/ correct dirt-under-grass
   layering, sandstone floors, one wooden hull) — the y-first raster order means
   layer-0 terrain dominates every training prefix. 53% of train is 3D-Craft,
   whose "houses" carry big grass aprons → the dominant generation mode.
3. **Not yet learned:** vertical structure (walls/roofs), EOS (63/64 ran to the
   2,047-block cap; 1 natural stop), coordinate non-repetition (~78% of emitted
   records repeat coords → 458 unique of 2,047).
4. vs LegoACE's reported ~82% raw connectivity: not comparable at this budget;
   revisit after a converged run.

**Figures:** `outputs/figures/minecraftace/uncond_ckpt1500_{samples,real_ref}.png`,
mid-run `midrun_ckpt1000_samples.png` (repetition-loop phase at epoch 7.5).

**Next:** (a) train to val-loss plateau (40–100 ep; val still falling at 0.66);
(b) consider GrabCraft-only or terrain-cropped curation to kill the grass-apron
mode; (c) MV-image (DINOv2) and text (CLIP) conditioning — data fully ready
(renders + VLM captions linked); (d) compare vs T11 small-AR baselines at
matched compute.

---

## T12. Pool-pretrain → finetune on houses_32 — run `20260708_070807_transfer` (11.4 h)

Pool = 16,383 builds / 113,049 aug sequences (labeled crawl + legacy + GrabCraft +
3D-Craft + remapped text2mc), houses_32 **val excluded from pool by source**; phase4
AR @ canon-16; scratch vs pretrain(30ep)→finetune(60ep, lr×0.3); 16 samples/arm.

| arm | val_nn | val_nn (constrained) | validity | dup | occ | final loss |
|---|--:|--:|--:|--:|--:|--:|
| **scratch (control)** | **0.405** | 0.378 | 0.5→1.0 gated / 1.0 constr | 0 | 230 | 0.181 |
| pretrain zero-shot | 0.365 | 0.301 | 0.375 / 1.0 | 0 | 145 | — |
| pretrain→finetune | 0.379 | 0.327 | 0.375 / 1.0 | 0 | 275 | **0.142** |

**Verdict: NO in-domain transfer gain** — finetune trails scratch on val-NN despite 22%
lower train loss (compression ≠ generation; pool also double-exposes train-house
sources, yet dup stays 0). Caveats: 1 seed, 16 samples (±~0.03 noise band), single
finetune LR, 30 pretrain epochs. **Positives:** houses_32-scratch matches the best T11
val_nn (0.405) on a cleaner dataset — curated cache validated; zero-shot pool model
reaches 0.365 without any house-specific training.

## A1. Ablation — discrete diffusion (Track B) sampling

What it took to stop the diffusion model collapsing to all-air / identical
samples (Legacy 896-subset). Each row is a fix; "result" is the qualitative effect.

| Change | Before | After |
|---|---|---|
| MaskGIT keep-mask schedule | `1 − cos(...)` (unmasked everything at step 0) → all-air | `cos(0.5·π·progress)`, decreasing 1→0 → progressive unmask |
| Decode rule | argmax from all-mask → identical samples | categorical + Gumbel-perturbed confidence → varied |
| Loss class weighting | uniform CE → predicts air everywhere | down-weight air (`air_weight=0.05`) |
| Occupancy control | seed-sensitive, empty/blob | `calibrate_air_bias` to match training median occupancy |

> TODO: turn this into a quantitative ablation (occupancy & validity vs each toggle).

---

## T11. Ideas ablations — run `20260704_001222_ideas` (COMPLETE, 15.4 h, 13/13 arms)

Ablation arms for the research.md levers (notes §6e), gc-houses-large: **pe**
(learned/sin/rope/alibi/phase4 on flat AR), **ordering** (BFS order ± in-loop adjacency
constraint vs raster), **samplers** (one diffusion model × maskgit/flow/remask/stratified),
**twostage** (occupancy→materials vs single-stage). Launch: `scripts/run_ideas.sh`
(or `--config ideas-full`). Real-build baseline `val_baseline_nn_iou = 0.48`,
occupancy target ≈ 1461 voxels. **All AR arms dup_rate 0.0 (zero memorization).**

| track | nn_iou | val_nn | validity raw→gated | block_agree | occ | note |
|---|---|---|---|---|---|---|
| **ar_pe_phase4** | **0.456** | **0.405** | 0.375→1.0 | 0.111 | 285 | **best arm** — grammar-aware PE |
| ar_pe_rope | 0.454 | 0.391 | 0.062→1.0 | 0.092 | 211 | ties phase4 on nn_iou, worse val_nn |
| ar_pe_learned | 0.442 | 0.395 | 0.312→1.0 | 0.061 | 239 | baseline PE |
| ar_pe_sin | 0.434 | 0.388 | 0.375→1.0 | 0.097 | 276 | ≈ learned |
| ar_pe_alibi | 0.391 | 0.346 | 0.375→1.0 | 0.038 | 254 | **weakest PE** |
| ar_raster_constrained | 0.428 | 0.383 | **1.0** (by constr.) | 0.085 | 235 | validity floor, ~quality-neutral |
| ar_bfs_constrained | 0.397 | 0.366 | **1.0** (by constr.) | 0.049 | 247 | BFS order *costs* quality |
| ar_bfs | 0.392 | 0.337 | 0.062→1.0 | 0.107 | 235 | BFS alone = loser |
| diff32_stratified | 0.259 | 0.218 | 0.0 | 0.023 | 830 | best sampler (de-clustered) |
| diff32_maskgit | 0.217 | 0.181 | 0.0 | 0.027 | 778 | underfills |
| diff32_flow | 0.212 | 0.224 | 0.0 | 0.004 | 7716 | overfills ~5× |
| diff32_remask | 0.004 | 0.003 | 0.25* | 0.0 | 2 | **broken** — erases structure |
| twostage32 | 0.382 | 0.335 | 0.0 | 0.028 | 12968 | best diffusion-family shape; overfills ~9×, material stage failed |

*remask validity 0.25 is meaningless — a ~2-voxel grid trivially passes connectivity.

### Hypothesis verdicts

1. **Relative/structured PE beats learned-absolute** — *partly*. Generic relative
   (RoPE/ALiBi) does **not** win: RoPE ≈ learned on val_nn (0.391 vs 0.395), ALiBi is
   the worst arm (0.346). Only **grammar-aware phase4** wins, and modestly (val_nn
   0.405 vs 0.395, +2.5%). Contrast Scaffold's big PE swing → our coords travel as
   tokens, not positions, so PE is a weak lever here.
2. **In-loop constraint → validity 1.0 at little cost** — *confirmed for the constraint,
   refuted for BFS order*. Adjacency constraint gives validity 1.0 by construction;
   **raster**+constraint is near-quality-neutral (nn_iou 0.428, val_nn 0.383). BFS order
   *hurts* (bfs_constrained 0.397 < raster_constrained 0.428; bfs alone worst val_nn 0.337).
   Recipe: raster + constraint, not BFS.
3. **Smarter sampler cuts hole artifacts** — *refuted at this scale*. Off one weak 32³
   model, occ swings 2 → 830 → 7716 by schedule alone with no fidelity recovery
   (best sampler stratified 0.259 ≪ AR 0.44). remask is unstable (erases structure).
   The occupancy problem is the model's, not the sampler's.
4. **Two-stage fixes material scrambling** — *refuted*. block_agree 0.028 ≈ single-stage
   (~0.03–0.11); the material stage stayed near-random. Factoring *did* lift shape
   fidelity (nn_iou 0.382, best of the diffusion family) but the occupancy stage
   overfilled ~9× (occ 12968) and materials never learned.

**Headline:** AR still dominates (best arm val_nn 0.405 = 84% of the 0.48 real-build
baseline, zero duplicates). phase4 is a small, honest win worth keeping; raster+constraint
is the validity-guarantee knob (compose them). Diffusion at 32³ remains uncalibrated —
neither smarter samplers nor two-stage factoring closed the gap to AR.

---

## T10. Cohesion + data battery — run `20260702_022207_overnight` (COMPLETE, 28.1 h)

**3 datasets × 6 tracks, 18/18 rows, zero errors** (notes §6c/§6d). Honest-novelty
protocol: dedup → val split → D4 augment (train only) → eval vs distinct real builds;
`val_baseline_nn_iou` = how close a *real* unseen build is to train (houses ≈0.48,
vehicles ≈0.64). Full table: `outputs/run_20260702_022207_overnight/leaderboard.md`.

| dataset | track | nn_iou | dup ↓ | val_nn | valid raw→gated | blk-agree | occ |
|---|---|--:|--:|--:|---|--:|--:|
| gc-houses | ar_flat | 0.485 | 0.0 | **0.412** | 0.188→1.0 | 0.142 | 207 |
| gc-houses | ar_cluster | 0.378 | 0.0 | 0.334 | 0.0→0.94 | 0.053 | 302 |
| gc-houses | ar_cluster_hi20 | 0.379 | 0.0 | 0.329 | 0.188→1.0 | 0.075 | 284 |
| gc-houses | ar_conditioned | 0.461 | 0.0 | 0.402 | **0.333**→1.0 | **0.15** | 312 |
| gc-houses | diffusion32 | 0.368 | 0.0 | 0.33 | 0.0 | 0.097 | 2577 (1461!) |
| gc-houses | graph_vae | 0.31 | 0.0 | 0.277 | 0.062→1.0 | 0.047 | 167 |
| combined | ar_flat | 0.475 | 0.0 | 0.408 | **0.562**→1.0 | 0.087 | 273 |
| combined | ar_cluster | 0.426 | 0.0 | 0.395 | 0.25→1.0 | 0.063 | 260 |
| combined | ar_cluster_hi20 | 0.432 | 0.0 | 0.381 | 0.125→1.0 | 0.065 | 287 |
| combined | **ar_conditioned** | **0.481** | 0.0 | **0.435** | 0.4→1.0 | 0.052 | 232 |
| combined | diffusion32 | 0.202 | 0.0 | 0.123 | 0.0 | 0.102 | 987 (1311) |
| combined | graph_vae | 0.388 | 0.0 | 0.364 | 0.0→1.0 | 0.028 | 154 |
| vehicles | ar_flat | 0.568 | **0.125** | 0.446 | 0.625→1.0 | 0.264 | 113 |
| vehicles | **ar_cluster** | **0.598** | **0.0** | **0.46** | 0.562→1.0 | 0.215 | 121 |
| vehicles | ar_cluster_hi20 | 0.525 | **0.188** | 0.396 | 0.5→1.0 | 0.249 | 146 |
| vehicles | ar_conditioned | 0.553 | 0.1 | 0.376 | **0.7**→1.0 | 0.151 | 76 |
| vehicles | diffusion32 | 0.446 | 0.0 | 0.38 | 0.25 | 0.019 | 86 (145) |
| vehicles | graph_vae | 0.391 | 0.0 | 0.3 | 0.188→1.0 | 0.044 | 102 |

**Hypothesis verdicts:**
1. **3D-BPE cluster tokens: object-type-dependent, anti-memorization confirmed.** On
   vehicles cluster-AR is the best track (nn_iou 0.598, val_nn 0.46) and — key — stays
   `dup 0.0` while flat memorizes (0.125) and hi20 memorizes harder (0.188). On houses
   at 16³ it *trails* flat (0.378 vs 0.485): big learned pieces get placed disconnected
   (raw validity 0). Chunk emission generalizes; it doesn't automatically cohere.
2. **BPE "reach" partially supported:** cluster@20³ matches cluster@16³ quality on houses
   (0.379/0.432 vs 0.378/0.426) where flat can't fit — but no quality *gain*, and on
   vehicles higher res = more memorization.
3. **Generalization is real:** best tracks hit **85-93% of the val-baseline** resemblance
   (combined ar_conditioned val_nn 0.435 vs 0.47 baseline = 93%) with dup 0 — the honest
   version of the memorization story T8 demanded.
4. **Aug + val-split kept houses dup at 0.0 everywhere**; vehicles at 90 epochs
   (final_loss 0.07) show overfit-memorization on flat/hi20/cond → next: early stopping
   on val NLL (§6e bundle), or fewer epochs on small homogeneous classes.
5. **Category conditioning = best all-around lever:** raw validity best-in-dataset on
   gc-houses (0.333) and vehicles (0.7), best val_nn overall on combined (0.435).
6. **More data works:** gc→combined lifts flat raw validity 0.188→**0.562** (3×),
   cluster 0.0→0.25, graph_vae nn_iou 0.31→0.388. Strong support for pool-pretraining
   (§3.4 corpora, ~16k builds).
7. **Graph-VAE trails AR everywhere** (0.31-0.39 nn_iou) but never memorizes, has top
   diversity, and scales with data — keep as the transfer-native track, not the flagship.
8. **Diffusion-32 is the weak point:** occupancy calibration fragile (2577 vs 1461 target
   on houses; collapse on combined 0.202), validity ≤0.25. The §6e sampler + two-stage
   arms target exactly this; diffusion-16 (T9) remains its best configuration.

---

## T9. All house classes, larger dim, deduped (GrabCraft) — run `20260701_061656_grabcraft`

Scaled up from one class to **9 house classes** and raised the cache cap to `max_dim=32`
(size-distribution knee: 24→65%, **32→85%**, 40→92% of house builds). Trained on **1,367**
deduped builds (`gc_small_32`). Directly fixes T8's memorization.

| model | n_train | NN-IoU | **dup ↓** | diversity | **validity ↑** | occ (target) |
|---|--:|--:|--:|--:|--:|--:|
| AR (canonical 16³) | 1009 | 0.47 | **0.00** | 0.815 | 0.312 | 303 |
| diffusion-32 MaskGIT | 1367 | 0.493 | **0.00** | 0.546 | **0.438** | 1495 (1158) |
| diffusion-16 MaskGIT | 1009 | 0.508 | **0.00** | 0.556 | **0.562** | 328 (233) |
| diffusion-32 flow | 1367 | 0.177 | 0.00 | 0.939 | 0.00 | 406 (1158) |

**Best overall run.** **Memorization eliminated** (`dup 0` everywhere vs T8's 0.31 — no more
NN-IoU=1.00 copies; Fig F9) via more/diverse data + higher canonical resolution (12→16) +
fewer epochs; dedup dropped only 6 (so T8's dups were *model* memorization, not raw copies).
**Validity ~doubled** (up to **0.562**, our best). Block-agreement fell to 0.066 — 9 diverse
palettes make exact materials harder (→ factored embeddings). Artifacts:
`outputs/run_20260701_061656_grabcraft/`.

---

## T8. Best models on **GrabCraft** medieval-houses — run `20260701_053401_grabcraft`

First run on the new **GrabCraft** dataset (category-labeled, cleaner). Trained our best
models on **724 medieval-houses**. Artifacts + checkpoints under
`outputs/run_20260701_053401_grabcraft/`; runner `blockgen/experiments_grabcraft.py`.

| model | n_train | NN-IoU | **dup ↓** | diversity | validity | **blk-agree** | occ (target) |
|---|--:|--:|--:|--:|--:|--:|--:|
| **AR (canonical 12³)** | 674 | **0.727** | **0.312** | 0.783 | 0.25 | **0.483** | 154 (151) |
| diffusion-24 MaskGIT | 724 | 0.46 | 0.00 | 0.616 | 0.312 | 0.035 | 1248 (914) |
| diffusion-24 flow | 724 | 0.17 | 0.00 | 0.943 | 0.00 | 0.014 | 396 (914) |
| diffusion-12 MaskGIT | 674 | 0.473 | 0.00 | 0.557 | 0.188 | 0.026 | 82 (151) |

**Read.** The cleaner dataset markedly improved AR: NN-IoU 0.568→**0.727** and
block-agreement 0.17→**0.483** (crisp, palette-correct medieval houses — grass→wood→purple
roof; Fig F8). **⚠ But AR now memorizes ~31%** (`dup=0.312`; some samples are NN-IoU=1.00
verbatim copies) — homogeneous style + 12³ downsampling collapse + possible raw
near-duplicates. **Diffusion is the novelty-safe complement** (dup 0, blobbier, MaskGIT held
occupancy at 24³). → Mitigations: dedup training set, regularize/early-stop AR on a val
split, raise canonical resolution. See run `run_notes.md`.

---

## T7. Representation & embedding analysis — `outputs/analysis/`

Answers the "what does the model actually see / learn" questions.
- **Tokenization methods** (`tokenization_methods.png`): the same structure as AR token
  stream vs fixed grid vs graph. **Air is only a token in the grid rep** (class 0, +MASK);
  the AR/graph reps emit occupied voxels only.
- **Learned embeddings** (`embedding_analysis.png`, from `ar_12.pt`): wood-plank and
  wool-colour variants are **near-orthogonal** — within-family cosine similarity
  **0.006 (wood) / 0.008 (wool)** vs a random baseline **0.002**, and the PCA shows
  families fully intermixed. ⇒ Flat embeddings do **not** learn birch≈oak; each `(id,data)`
  is an independent atom. Direct motivation for **factored `(family,variant)` embeddings**
  (notes §9.1). → Fig F6, F7.

---

## T6. How models generate + AR-vs-diffusion + flow matching — run `20260630_232913_gen`

Two studies on the **houses** subset. Runner `blockgen/experiments_gen.py`; artifacts +
checkpoints under `outputs/run_20260630_232913_gen/` (`run_notes.md`).

**T6a — AR vs diffusion on identical canonical 12³ houses** (scale-normalized so AR can
ingest them: median 1086→185 blocks, 654/714 fit AR `seq≤1600`):

| model | NN-IoU ↑ | validity ↑ | diversity | blk-agree | median occ (target 185) | loss |
|---|--:|--:|--:|--:|--:|--:|
| **AR** | **0.568** | 0.375 | 0.779 | **0.170** | **165** | 0.114 |
| diffusion | 0.369 | 0.375 | 0.902 | 0.138 | 60 (under-fills) | 0.254 |

→ **AR wins on houses** once scale-normalized (confirms the hypothesis). AR samples are
consistently house-shaped; diffusion at 12³ is mixed (some blobs, two empty). → Fig F5.

**T6b — diffusion sampler study** (one trained 24³ net, two inference rules):

| sampler | NN-IoU | validity | median occ (target 1086) |
|---|--:|--:|--:|
| MaskGIT (confidence top-k) | 0.266 | 0.062 | 96 — **under-fills** |
| flow matching (rate-driven) | 0.228 | 0.000 | 1017 — **holds density** |

→ A well-trained absorbing-diffusion net is so confident about air that **MaskGIT commits
air and dumps a blob at the final step** (occ trace 0→3→4→5→401); **flow matching** reveals
voxels at a steady rate and grows occupancy smoothly to target (25→…→1123). Real sampler
difference; flow needs its own negative `air_bias`. → Fig F4.

**Key takeaways for the paper/LEGO pitch:** AR (LegoGPT-aligned) is our best house
generator and generates *bottom-up like a builder*; scale normalization is the unlock;
absorbing-diffusion+MaskGIT is fragile to occupancy.

---

## T5. Curated-subset retraining + novelty (Labeled) — run `20260630_220830`

The first training pass on **curated, labeled subsets**. Each model is evaluated
against *the subset it trained on* (so NN-IoU = "did it memorize *these* builds").
Artifacts + configs under `outputs/run_20260630_220830/` (`run_notes.md`, per-model
`config.json`/`comparison.png`). Runner: `blockgen/experiments.py`. 100–150 epochs.

| (subset) track | n_train | final loss | mean NN-IoU | dup ↓ | diversity ↑ | validity ↑ | blk-agree | med occ |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| (pixel_art) AR | 124 | 0.218 | 0.406 | 0.00 | 0.898 | 0.062 | 0.081 | 158 |
| (pixel_art) diffusion | 124 | 0.039 | 0.101 | 0.00 | 0.991 | 0.000 | 0.007 | 60 |
| (pixel_art) graph VAE | 124 | 0.362 | 0.107 | 0.00 | 0.972 | 0.125 | 0.007 | 35 |
| **(houses) diffusion** | 714 | 0.167 | **0.483** | 0.00 | 0.713 | **0.250** | 0.065 | 950 |
| (redstone) AR | 484 | 0.136 | 0.392 | 0.00 | 0.877 | 0.062 | 0.120 | 159 |

**Read.** `dup_rate = 0` for every model ⇒ **no memorization**; NN-IoU ≈ 0.4–0.5 means
samples *look like the type* without copying any build. Headline:
**houses/diffusion** reproduces the house gestalt (grass pad → walls → colored roof) at
near-correct occupancy (950 vs 1086) with NN-IoU 0.38–0.49 to *distinct* real houses —
the "learn one object type" milestone (→ Figure F2a). Token tracks (AR/graph) learn
compact types but **fragment** (validity ≤ 0.13) — motivates the validity gate.
**Block-class agreement is low everywhere** (≤ 0.12): shape is learned, exact materials
are not — motivates factored `(family, variant)` embeddings.

> Note: which track can train which subset is itself a result — token tracks can't
> ingest dense houses (41/714 fit `seq≤1024`), diffusion takes all 714. See notes §6b.

---

## T4. Preliminary three-track novelty  ⚠️ SUPERSEDED by T5

Ran on the **Legacy** 896-structure subset (seq ≤ 1024), short training runs.
**Superseded by T5** (curated labeled subsets). Kept here as the protocol demo and
sanity baseline only.

| Track | Median occ. | mean NN-IoU ↓ | duplicate rate ↓ | diversity ↑ | validity ↑ |
|---|--:|--:|--:|--:|--:|
| A: AR transformer | 142 | 0.26 | 0.00 | 0.96 | ~0 |
| B: diffusion | 94 | 0.50 | 0.00 | 0.74 | ~0 |
| C: graph VAE | 49 | 0.17 | 0.00 | 0.99 | ~0 |

Self-check: a training structure scores NN-IoU = 1.000 against the training set ✓
(eval is correct). **Read:** low NN-IoU + 0 duplicates + high diversity ⇒ genuinely
novel, *not* memorized. **Caveat:** validity ≈ 0 — samples fragment; motivates the
connectivity/validity gate (notes §9.3).

---

## Figures (rendered in notebooks; export to `figures/` for the paper)

> Full per-corpus data inventory (counts, licenses, labels, sample sheets for every
> Minecraft and LEGO source) lives in **`data_sources.md`**.


- **F1** — Curated subset contact sheets (houses / pixel_art / redstone / towers / trees /
  popular) + material-variant groups (same shape, different blocks). →
  `outputs/run_20260630_220830/figures/subset_*.png`, `variant_group_*.png`.
- **F2** — Per-track: generated sample (left) vs top-k nearest training neighbors (right) —
  the visual novelty proof. → `outputs/run_20260630_220830/models/*/comparison.png`.
  - **F2a (headline)** — `houses__diffusion/comparison.png`: generated house gestalts vs
    distinct real houses (NN-IoU 0.38–0.49).
- **F3** — NN-IoU distribution per track (histogram). _TODO._
- **F4** — Generation-process filmstrips: AR (bottom-up build), MaskGIT (stall→blob),
  flow matching (smooth fill). → `outputs/run_20260630_232913_gen/film_*.png`.
- **F5 (headline)** — Canonical 12³ houses: AR vs diffusion samples side-by-side (AR makes
  consistent houses; diffusion blobs/empties). → `…_232913_gen/compare_ar_vs_diffusion.png`.
- **F6** — Tokenization/representation methods (token stream / grid / graph + air handling).
  → `outputs/analysis/tokenization_methods.png`.
- **F7** — Learned block-embedding similarity (wood/wool near-orthogonal; PCA intermixed).
  → `outputs/analysis/embedding_analysis.png`.

- **F8** — GrabCraft medieval-houses: dataset sheet + AR samples vs nearest neighbors (crisp
  purple-roof houses; note the NN-IoU=1.00 memorized rows). →
  `outputs/run_20260701_053401_grabcraft/{data_sheet,ar_canon12/comparison}.png`.
- **F9** — All-9-house-classes (dim 32, deduped): AR samples vs neighbors — novel, diverse
  houses, **no IoU=1.00 copies** (contrast F8). →
  `outputs/run_20260701_061656_grabcraft/ar_canon16/comparison.png`.
- **F11 (model samples, textured)** — Fresh samples from the ideas-battery checkpoints
  rendered with the real-texture pipeline, 48 each, same canonical scale as training:
  real canon-16 reference vs `ar_raster_constrained` (validity 1.0 — visibly single-
  component, house-gestalt massing) vs `ar_pe_phase4` (best val-NN; similar quality,
  occasional floaters) vs `diff32_maskgit` baseline (scattered translucent blobs —
  matches its validity 0). → `outputs/figures/samples_{real_canon16,ar_raster_constrained,
  ar_pe_phase4,diff32_maskgit}.png`. Regenerate: `.venv/bin/python
  scripts/render_model_samples.py --samples 48` (rebuilds the battery's exact prep/vocab,
  reloads checkpoints, samples fresh, renders textured).
- **F10 (dataset showcase)** — Dense grids of the unified curated house dataset,
  rendered with **real Minecraft textures** (headless pyrender pipeline, notes §13):
  140-sample combined grid + per-corpus grids (grabcraft / 3dcraft / text2mc) + 48³ grid.
  → `outputs/figures/houses_32_grid_140.png`, `houses_32_{grabcraft,3dcraft,text2mc}.png`,
  `houses_48_grid.png`. Regenerate: `python -m blockgen.renderer.grid --houses 32 …`.
- **F12 (transfer samples, textured)** — Fresh samples from the three
  `run_20260708_070807_transfer` checkpoints (T12) rendered with the real-texture
  pipeline, 48 each: `scratch` (houses-only control, best val-NN 0.405) vs
  `pretrain_zeroshot` (pool checkpoint, no finetune — noisier, sparser massing) vs
  `finetune` (pool ckpt + houses, lowest train loss but no in-domain val gain). Visual
  read matches the negative transfer result: finetune ≈ scratch, zero-shot weaker. →
  `outputs/figures/samples_transfer_{scratch,pretrain_zeroshot,finetune}.png`. Regenerate:
  `.venv/bin/python scripts/render_transfer_samples.py --samples 48` (reconstructs the
  run's exact seed-0 vocab from the cross-corpus pool, reloads each checkpoint, samples
  fresh with the same plain sampler that produced each arm's headline val_nn).

- **F16 (headline — what the decimator does, textured)** — six real `houses_32` builds at
  **native** resolution (top row) and **the same six** after `canon_dim=16` (bottom row).
  No model involved. Retention **8.4–16.5%**. The clearest case is the pagoda (col 3): 667 →
  **56** blocks, the one-voxel-thick tiered roof **vanishes entirely**, and four corner
  lanterns are left floating in mid-air — the parity-deletion mechanism of T17, visible.
  Col 1's cabin loses its roof and gains a hole through the wall; cols 5–6 are roof segments
  floating over gaps. **Compare the bottom row to any uncond sample sheet** (e.g.
  `run_20260715_065938_native/canon16_flat/samples.png`): the model's blobby, holed,
  fragmented output *is* a faithful imitation of this. Also explains
  `train_validity_rate=0.427` with no statistics needed — four floating blocks are four
  extra components. → `outputs/figures/decimation_native_vs_canon16.png`. Regenerate:
  `.venv/bin/python scripts/render_decimation.py --cols 6`. **This is the figure for the paper's data
  section.** See T17.
- **F14 (3D-BPE piece vocabulary, textured)** — the 96 largest learned pieces from a
  256-merge vocab on `houses_32`, real textures, sorted by voxel count. **The honest read:
  flat plates and solid cuboids, a handful of 1×N beams, exactly two vertical pillars, and
  zero wall corners / L-shapes / roof slopes** — despite `cluster_bpe.py`'s docstring
  promising "a wall corner … a roof-slope unit". This is the eval for the PMI-merge
  experiment (notes §9.15): rerun and see whether corners and pillars appear. →
  `outputs/figures/bpe_pieces.png`. See T18.
- **F15 (exploded token views)** — four real builds, each shown intact (left) and cracked
  apart into the pieces it tokenizes to (right), displaced radially from the centroid.
  Shows the **1.2–1.9 voxels/piece** result at a glance: most voxels stay atomic, so the
  tidy slabs of F14 are the *vocabulary*, not what a build actually becomes. →
  `outputs/figures/bpe_exploded.png`. See T18.
- **F13 (LEGO dataset showcase, Blender)** — Dense grids of the **OMR** LEGO corpus
  (`data/lego/omr`) rendered with real LDraw geometry + LEGO colours via **Blender 4.2 +
  ImportLDraw** (`legogen/renderer/`, kept separate from `blockgen`): `omr_grid.png` = 96
  small/medium official sets (part-ref band 15–160), `omr_showcase.png` = 24 larger,
  part-diverse sets (band 200–700) rendered bigger. Both show the diverse **non-cuboid
  parts** the thesis targets — tires, slopes, curved panels, Technic gears/axles, minifigs,
  BrickHeadz, architecture. Uniform orthographic-isometric framing on white, the Minecraft
  `mc_data` sheet aesthetic. → `outputs/figures/lego_data/omr_{grid,showcase}.png`.
  Regenerate: `.venv/bin/python legogen/renderer/ldraw_grid.py --rows 8 --cols 12 --band
  15 160 --out outputs/figures/lego_data/omr_grid.png` (see `legogen/renderer/README.md`).
  _StableText2Brick (cuboid `hxw` format) not yet rendered — needs an hxw→LDraw converter._

> Browsable docs: `mkdocs serve` (site under `docs/`, config `mkdocs.yml`) — overview,
> models, representations (with F6/F7), experiments, the pick-and-place track
> (`docs/pick-and-place.md` — architecture, I/O contract, formulation audit, and the
> math), and the LEGO roadmap.

---

## Open result slots (fill as we run)
- [x] Re-run T4 on the **labeled** curated subsets (→ **T5**, run `20260630_220830`).
- [ ] Ablation: factored `(family,variant)` embeddings vs flat — rare-block generalization, val loss, sample quality.
- [ ] Ablation: validity gate on/off — validity_rate, diversity cost, reject rate.
- [ ] Category-conditional vs unconditional — per-category validity & novelty.
- [x] Distributional metric (FID-analogue over occupancy/feature space) train vs samples
      (→ **T23**, MV-DINO-KID, validated against known damage before use).
- [ ] Regenerate every headline arm at n ≥ 256 (Track E at n ≥ 128, ~$0.07) — at the
      current `--samples 16` default no arm comparison is resolvable below a 0.15 KID gap.
- [ ] Score the neural AR arms through `bench` (needs `scripts/dump_samples.py` per arm).
- [ ] Human 2AFC study + metric-vs-human rank correlation, to justify the suite for the paper.

### T26g — BlockLab dataset tree and branches (tooling)

The hub's flat dataset list was replaced by a derivation tree. It also fixed an
arithmetic error: summing all 35 rows reported **~178,000 builds where 96,103
exist**, because `raw:all` is the five raw corpora listed beside it and each
split is part of its corpus. Counts are now per node.

| root | builds | children |
|---|---|---|
| `raw:all` | 59,387 | 5 raw corpora (text2mc_schem 28,235 · text2mc_h5 11,092 · legacy 10,963 · grabcraft 6,560 · 3dcraft 2,537) |
| 9 corpus caches | 36,545 | `houses_32` carries its 3 splits |
| 13 run dirs | 171 | 17 arms |

Every edge is read off disk. Corpus provenance (`houses_32` ← grabcraft 1,360 ·
3dcraft 1,267 · text2mc 34, from the manifest's per-row `corpus`) is a
**multi-parent** relation and is rendered as an annotation, not an edge.

**Branches.** Any node can be subset by rule — labels, source corpus, category,
title, block count, fits-in-cube, seeded random sample — stored as
`{parent, rule}` JSON, resolved to parent indices on read, and nestable. `live`
rules re-run on read (they grow as you label); `frozen` rules store the resolved
index list and are what an experiment should cite. Verified end to end: a
grabcraft-only branch of `houses_32` resolves to 1,360, a random-256 branch of
*that* resolves to 256 all-grabcraft builds, and thumbnails render through the
branch like any other dataset.
