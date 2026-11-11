# BlockGen — Results

Tables, figures, and ablations backing the paper. Methods/repro live in
[`notes.md`](notes.md). Each result notes **which cache / subset** it ran on and
**when**, because the dataset is being migrated (legacy `data/raw` → labeled
tfrecord cache) and numbers are not comparable across caches.

_Last updated: 2026-11-09._ The one-line index of every experiment and its verdict is [`tried.md`](tried.md).

> **Policy, 2026-09-17 — decimated real data is not a quality reference.**
> `real@canon16` / `real@canon8` have been removed as leaderboard controls
> (`blockgen/eval/bench/fast.py`) and every claim of the form "worse than
> N%-decimated real data" has been struck from this file. Requantising a build
> to a coarser grid changes its *resolution*, not its *quality*: it makes a
> smaller building, not a worse one, so ranking a generator against it was never
> a meaningful comparison. The rungs survive inside the metric-validation ladder,
> where the only question asked of them is whether a metric orders a known
> corruption correctly. **Tables below that predate this date still show the two
> rows** — their numbers are correct and unchanged; ignore their position in the
> ranking.

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

## T32. The clean-split retrain — the margin was the contamination (2026-09-17)

> ### Bottom line
> Retrained on the canonical group-aware split, the native track **loses its
> separation from the LLM arms entirely**: `canon_oriented` lands in Holm group
> **e** alongside `llm_control` (15.006) and `llm_bias` (15.869), where the
> contaminated `native_oriented` had group **d** to itself. The unoriented clean
> retrain is **disqualified for memorization** (+3.4 spreads) at duplicate rate
> 0.000. "Native beats the LLM baselines" does not survive T31's correction.

**Runs:** `outputs/run_20260917_000431_canon_{oriented,unoriented,flat}_native`
(60 ep, `--split canonical`, seed 0) · scored in
`outputs/run_20260917_043807_bench_canonical-split_native_arms_oriented_vs/`
(n=128, ref 399, self-validation 9/9).

| arm | geom_kid ↓ | grp | mv_dino_kid ↓ | dino_nn_pct | BlockScore ↓ | status |
|---|--:|:-:|--:|--:|--:|---|
| `real_test` | −0.095 | a | −0.001 | 0.540 | 0.00 | floor |
| `native_oriented` *(contaminated)* | 11.624 | **d** | 0.194 | 0.496 | 106.87 | unranked (T31) |
| **`canon_oriented`** *(clean)* | **14.898** | **e** | 0.243 | 0.430 | **136.73** | ranked |
| `llm_control` | 15.006 | **e** | 0.444 | 0.744 | 145.60 | ranked |
| `llm_bias` | 15.869 | **e** | 0.566 | 0.792 | 185.73 | ranked |
| `canon_unoriented` *(clean)* | 19.570 | f | 0.311 | 0.369 | — | **DQ memorization (+3.4)** |

**Three readings.**

1. **The margin was the head start.** T31 reported `native_oriented` a full group
   clear of both LLM arms and flagged that the margin could not be separated from
   87% test contamination. It now can: the same architecture, same
   hyperparameters, clean split, is *tied* with both. The gap was contamination,
   not capability. BlockScore moves 106.87 → 136.73 in the same direction.

2. **The memorization gate fired on a clean retrain, and a duplicate counter
   would have missed it.** `canon_unoriented` has duplicate rate **0.000** and is
   still DQ'd, on `dino_nn_percentile` 0.369 against a real floor of 0.540 — it
   sits systematically closer to its training data than real held-out builds do,
   without copying any build outright. This is the retrieval-percentile
   formulation earning its place over a duplicate count.

3. **Oriented beats unoriented on a clean split** — 14.898 vs 19.570, separate
   groups (e vs f). That settles the direction of the long-standing assumption,
   though not the original `native_bpe` vs `native_oriented` question, which was
   between two *contaminated* checkpoints and still needs its own clean pair.

`canon_oriented`'s own `dino_nn_percentile` is 0.430 against the 0.540 floor —
not DQ'd, but leaning the same way as its disqualified sibling. Worth watching on
the n≥256 re-run rather than treating as clean.

**Still open:** every arm here is at n=128, the protocol `min_n` floor; 256 or 399 is the quotable convention (B1).

## T31. `native_oriented` at n=128 — BlockScore 106.87, and not a rankable number (2026-09-10, scored 2026-09-16)

The question was whether the native BPE track beats the LLM arms. It does, decisively —
and measuring it surfaced that the number everyone has been quoting is scored on data the
model trained on.

**Which native model.** `native_bpe` (2026-07-15) and `native_oriented` (2026-07-21) are
two different runs, not one. `native_oriented` is the benchmarked one; `native_bpe` has
**never been through the bench**, and on the one metric where they are compared head to
head it is *ahead* (ungated validity 0.188 vs 0.062, results.md:1684). That comparison is
still open.

**Sampled** with `scripts/sample_to_npz.py --model native_oriented --n 128` (17.8 min, 0
empty, median 887 blocks against a real 904 — four times the LLM arms' ~206). Scored in
`outputs/run_20260910_183610_bench_native_oriented_at_n_128_vs_LLM_arms/`.

| arm | n | `geom_kid` ↓ | grp | `mv_dino_kid` ↓ | grp | nn_iou | dup |
|---|--:|--:|:-:|--:|:-:|--:|--:|
| `real@canon16` | 128 | 6.572 | c | 0.110 | c | — | — |
| **`native_oriented`** | 128 | **11.624** | **d** | **0.196** | **d** | 0.350 | 0.000 |
| `llm_control` | 128 | 15.006 | e | 0.443 | e | 0.257 | 0.000 |
| `llm_bias` | 128 | 15.869 | e | 0.566 | f | 0.242 | 0.000 |
| `real@canon8` | 128 | 18.584 | f | 0.485 | e | — | — |
| `patchwork` | 128 | 19.976 | f | 0.683 | g | 0.278 | 0.000 |

Holm-corrected head-to-head: `native_oriented` beats `llm_control` by −0.247
[−0.285, −0.209] and `llm_bias` by −0.370 [−0.424, −0.314], both p ≈ 0. It is a full
group clear of both on each metric. `llm_control`'s `geom_kid` reproduces the T29a run to three decimals (15.006),
which is the cross-run consistency check that makes these comparable.

**BlockScore, on the re-run** (`run_20260916_233528_...`, self-validation 9/9; the first
attempt's `real_test` control OOMed against an unrelated job and produced no composite):

| # | arm | BlockScore ↓ | worst pillar | appearance | geometry |
|---|---|--:|---|--:|--:|
| 1 | `real_test` | 0.00 | — | +0.00 | +0.00 |
| 2 | `real@canon16` | 60.80 | geometry | +36.14 | +60.80 |
| 3 | **`native_oriented`** | **106.87** | geometry | +64.50 | **+106.87** |
| 4 | `llm_control` | 145.40 | appearance | **+145.40** | +137.72 |
| 5 | `real@canon8` | 170.35 | geometry | +159.03 | +170.35 |
| 6 | `llm_bias` | 185.83 | appearance | +185.83 | +145.58 |
| 9 | `patchwork` | 223.24 | appearance | +223.24 | +183.03 |

**106.87, not the 87.14 on record.** T26e scored `native_oriented` from **64** builds;
this is the same checkpoint at the protocol's **128**. Nothing about the model changed —
KID is biased by sample size, and the arm's own `n` moves it. The 87.14 that has been
quoted as this project's best learned number is a 64-build reading, and 106.87 is the
one the protocol asks for.

**The two tracks fail in opposite directions**, which the pillar columns make plain:
`native_oriented` is geometry-limited (+106.87 geometry against +64.50 appearance) while
both LLM arms are appearance-limited (+145.40 against +137.72). They are not the same
model at different quality levels.

**The catch, measured.** `native_oriented` predates the pinned split (created 2026-08-03)
and trained through `_split_houses(seed=0, val_frac=0.15)` — a fresh permutation over the
whole corpus. Overlap with the canonical split, computed exactly:

> **346 of the 399 canonical test builds (87%) are in its training set**, plus 323 of 399
> canonical val.

So its long-standing **BlockScore 87.14** (T26e) — the best learned number this project
has — is measured almost entirely on builds it was fitted on, and the same is true of
`native_bpe`, `canon16_flat` and `text_cond`. All four now record that provenance in
`deploy/inference/models.json` (`train_split: "pre-split-val_frac0.15-seed0"`), so the
board states the reason rather than only the verdict.

**What keeps this from being a simple retraction.** The memorization gates do not fire:
duplicate rate 0.000, and `dino_nn_percentile` 0.495 against a real-held-out floor of
0.539. It is *slightly* closer to training data than real builds are — directionally what
87% contamination predicts — but nowhere near the `train_verbatim` DQ (+10.8 spreads). It
is a small underfit model (final train loss 0.311, still falling), so the inflation is
plausibly modest. Plausibly is not measured, and the honest statement is that
`native_oriented`'s margin over the LLM arms **cannot currently be separated from its
head start**.

**Fix, priced.** Retrain on the canonical split: `train_min` was 113.4, so ~1.9 h, plus
~18 min to sample 128 and ~11 min to bench. `blockgen/experiments_native.py --split
canonical` is now the default and does exactly this; `--split legacy` reproduces the July
runs and warns that it can never be ranked.

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

**The fix this implies, and the next run.** The query reference should be the *current*
piece's anchor, not the previous one's, which requires emitting coordinates before the
piece name (`"<x> <y> <z> <piece_name>"`) so the anchor is causally available at the
moment the piece type is predicted — the geometry then answers "what goes *here*"
rather than "what goes near where I just was". It needs its own matched control on the
reordered format, since the serialization changes.

> **Correction (2026-09-16).** This section originally also claimed the reordering
> "makes all 729 buckets reachable". It does not. The dead half of the table is caused
> by the **emission order**, not by the reference point. Under `(y,z,x)` raster order
> every earlier piece satisfies `dy > 0`, or `dy = 0 ∧ dz > 0`, or `dy = dz = 0 ∧ dx > 0`,
> which admits exactly **364** clamped buckets, plus one for the case where the key *is*
> the reference piece — **365**, precisely the number the trained table updated. Both
> candidate reference anchors (`a_i` and `a_{i-1}`) are subject to the same constraint,
> so reordering the line changes *which* frame the geometry is expressed in and nothing
> about coverage. Unlocking the other half needs a non-monotonic emission order —
> BFS/growth, which this repo already has (`ar_bfs`, T11).

Secondary caveat, stated in the script: the bias is shared across all layers and heads,
so this null bounds a **shared-scalar** bias, not per-head structure.

Figures: `geobias_loss.png`, `geobias_table.png`, and `samples_textured.png` per arm in
`outputs/run_20260906_222749_llm_geobias_control/` and
`outputs/run_20260906_232253_llm_geobias_bias/`.

### T29a. The same pair through the benchmark at n=128 — the null becomes a mild negative (2026-09-07)

T29 read the arms from 8 samples and a val-loss delta. Run through
`blockgen.eval.bench` at the protocol's n=128, they separate, and not in the
direction the loss suggested.

Arms sampled with `scripts/sample_geobias.py` (KV-cached; `--n` counts *kept* builds).
Prompts come from the canonical `houses_32` test split **minus the 53 captions each run
trained on** — prompting a model with a build it memorized measures recall, not
generation. 128 builds each, mean parse 0.939 / 0.940, 11 / 9 empty generations
discarded. Card: `outputs/run_20260907_220119_bench_LLM_geometric_attention_bias_control_vs/`,
self-validation **all 9 checks pass**.

| arm | BlockScore ↓ | worst pillar | geom_kid ↓ | mv_dino_kid ↓ | voxel_diversity | mean max_dim |
|---|--:|---|--:|--:|--:|--:|
| `real_test` | 0.00 | — | −0.095 | −0.001 | 0.902 | 19.2 |
| **`llm_control`** | **145.55** | appearance | 15.006 `d` | 0.444 `d` | 0.927 | 28.4 |
| **`llm_bias`** | **186.18** | appearance | 15.869 `d` | 0.568 `e` | 0.941 | 35.4 |
| `patchwork` (best baseline) | 223.51 | appearance | 19.976 | 0.682 `f` | 0.897 | — |
| `gabled_house` | 300.15 | geometry | 32.818 | 0.891 `g` | 0.917 | — |
| `uniform_random` | 562.76 | appearance | 37.729 | 1.718 `i` | 0.915 | — |

Letters are Holm-corrected head-to-head groups.

**Both arms clear every procedural baseline** — the first time the LLM track has been
scored against the floor at all. `llm_control` ranks 3rd of 13 scored arms, behind only
`real_test` and `real@canon16`.

**The bias arm is significantly worse, on appearance only.** On `geom_kid` the two share
group `d` — geometry does not separate them. On `mv_dino_kid` they are groups `d` vs `e`,
Δ = −0.124 [−0.180, −0.052], p_holm = 0.0005. So the −0.0021 val-loss advantage T29
recorded does not survive contact with the benchmark, and the §20 rule holds again:
**loss moved the right way while quality moved the other way.**

**The mechanism is runaway coordinates.** The parser accepts anchors up to 255, and both
models emit builds far outside the corpus's 32³ scale — `llm_control` 26/128 builds over
32 (largest 253), `llm_bias` **39/128** (largest 251). Mean `max_dim` is 19.2 for real
builds, 28.4 for the control, 35.4 for the bias arm. The bias arm's extra 50% of runaway
builds is a rendered-appearance penalty and nothing else, which is exactly the pillar it
loses on. Oversize builds are **not** filtered — they are real model output, and dropping
them would flatter the arm — but the rate is now recorded in every arm's report.

Both arms remain far from the data on the thing that matters most: mean `n_blocks` 211
and 206 against a real 904.

**Neither arm is leaderboard-eligible**, and the board now says so — see §32.

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

Track E's KID point estimate is 0.065 — but at n=8 the interval
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
2. **Native's KID is 0.187**, an interval clear of the real floor (−0.001) and of
   the agentic arm below it.
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
split is part of its corpus. Counts are now per node, and `outputs/` plus each
run dir are containers that hold no builds of their own.

| root | builds | children |
|---|---|---|
| `raw:all` | 59,387 | 5 raw corpora (text2mc_schem 28,235 · text2mc_h5 11,092 · legacy 10,963 · grabcraft 6,560 · 3dcraft 2,537) |
| 9 corpus caches | 36,545 | `houses_32` carries its 3 splits |
| `outputs/` | 1,131 | 21 run dirs → 30 arms, `bench_arms` first then newest-first |

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

## T33. What the interior probe applies to — measured across structure types (2026-09-18)

> ### Bottom line
> The predictor is **enclosed volume**, not "rooms". Across 1,800 labelled
> builds the probe is strong on dwellings (medieval house 85% affected, median
> +22.5% blocks; house 73%/+27.3%) and collapses on solid sculptural forms
> (statue 43% affected, **median +0.0%**). But the same measurement on LEGO
> bricks finds cars 94%, bottles 92%, cameras 97% — none of them room-based.
> The corruption needs a structure that *encloses space*, which is common in
> architecture, vehicles, containers and shells, and absent in solid masses.

Method: `scipy.ndimage.binary_fill_holes` on the cropped occupancy grid, i.e.
exactly what `probes.solidify` fills. n=1,800 sampled from `all_32` (4,712
VLM-categorised builds); categories with n<15 omitted.

| category | n | % with any void | median +blocks | mean +blocks |
|---|--:|--:|--:|--:|
| medieval house | 221 | **85%** | **22.5%** | 27.2% |
| blacksmith | 23 | 83% | 8.5% | 13.1% |
| wooden house | 19 | 79% | 5.3% | 11.2% |
| modern house | 74 | 74% | 14.8% | 22.2% |
| house | 797 | 73% | **27.3%** | 30.7% |
| treehouse | 16 | 62% | 0.6% | 6.5% |
| tower | 37 | 62% | 3.1% | 16.5% |
| statue | 42 | **43%** | **0.0%** | 7.0% |
| **all** | 1,800 | 71% | 15.9% | — |

**Three readings.**

1. **The gradient is real and it tracks enclosure, not category.** A treehouse
   is a dwelling and scores 0.6% median; a statue is not and scores 0.0%. Both
   are mostly solid or open mass. Meanwhile `house` and `medieval house` — which
   enclose rooms — are where the probe bites hardest.

2. **It is not confined to buildings.** On `data/lego/stabletext2brick`
   (47,389 structures, MIT), 48.7% contain enclosed voids, and the per-category
   rates are camera 97%, car 94%, bottle 92%, bus 90%, tower 77%, sofa 74%.
   These are objects, not rooms. Conversely `ShapeNetVox32` is **already
   flood-filled** — mean enclosed voids 0.0 across twelve of thirteen categories
   — so the probe is a literal no-op there and must not be used.

3. **Only one part of the suite depends on this.** `solidify` and the
   `enclosed_air_ratio` coherence column need enclosed volume. The geometry and
   appearance tiers, the validation ladder, the memorization and mode-collapse
   gates, and BlockScore's aggregation rule are all indifferent to it. Two
   things *are* domain-specific and would need redesign elsewhere: the procedural
   baselines (`gabled_house`, `shell_box` are house-shaped) and the palette
   pillar (degenerate on single-material corpora such as ModelNet40).

**Consequence for the paper.** The scope sentence should be *"structures that
enclose space"*, evidenced by this table plus the LEGO rates — not *"buildings"*,
which the statue and treehouse rows contradict, and not *"3D generation"*, which
ShapeNetVox32 contradicts. This is also the empirical backing for the
visual-hull framing: the general claim is that a render-and-embed pipeline
scores only the visible boundary, and this table says where that boundary hides
enough to matter.

## T34. LEGO replication — the blindness is not one metric, and not one domain (2026-09-18)

> ### Bottom line
> On a second domain (47,389 LEGO brick structures, 20³, MIT-licensed) with a
> second representation and a second literature, the result replicates **more
> completely than on houses**: MV-KID puts solidified structures at **−0.01
> noise floors** — a point estimate *below* the real floor — while `geom_kid`
> puts them at **+118.6**. And the in-domain ladder generalises the claim from
> one metric to a class: **all seven render-space statistics fail to resolve
> `solidify`, and all seven resolve every other corruption.**

**Run:** `outputs/run_20260918_200852_bench_lego_interior_blindness_replication/`
(n=256 arms, 450-build val reference, dinov2b 4 ortho views @224px, seed 0).
Loader `blockgen/eval/bench/lego.py`, 18 tests in `tests/eval/test_lego.py`.

### T34a. The arm table (verified against the scorecard)

| arm | `mv_dino_kid` ×1000 | sep | `geom_kid` ×1000 | sep |
|---|--:|--:|--:|--:|
| `real_test` (floor) | 3.244 | +0.00 | 371.1 | +0.0 |
| **`real@solidify`** | **3.217** | **−0.01** | **20440.7** | **+118.6** |
| `real_shuffled_materials` | 315.9 | **+140.0** | 371.1 | *+0.0 bit-identical* |
| `real@monochrome` | 1534.6 | **+685.8** | 371.1 | *+0.0 bit-identical* |
| `train_verbatim` | 5.896 | +1.19 | 186.4 | −1.1 |

In-run noise floors: `mv_dino_kid` 0.00223, `geom_kid` 0.16928.

**The control is inside the table.** The render metric is not asleep on this
corpus: at the same n, on the same builds, it separates a material permutation
at **+140** floors and a monochrome repaint at **+686**. It cannot separate the
interior fill at all. The geometric tier's complementary blindness replicates
exactly too — bit-identical under both material corruptions.

Against houses, in the same units:

| domain | n | `real@solidify` appearance | geometry |
|---|--:|--:|--:|
| `houses_32` | 128 | +0.38 | +201.8 |
| **`lego_void_20`** | **256** | **−0.01** | **+118.6** |

> ⚠️ **These columns are NOT directly comparable, and an earlier version of this
> table implied they were.** The noise-floor unit is n-dependent: measured on the
> same corruption at two sample sizes, `geom_kid`'s noise floor *halves* when n
> doubles (0.1732 → 0.0848, ratio 0.49), so a separation in noise floors grows
> roughly **linearly in n** (133 → 287, ratio 2.16). Normalised to a common
> n=128, LEGO's geometry separation is ≈**59**, not 118.6 — so the effect is
> about **3.4× weaker** on LEGO than on houses, not 1.7× as the raw figures
> suggest. The qualitative result (render ≈ 0, geometry enormous) holds in both
> domains and at both n; the numeric comparison does not survive without
> normalising. See T35.

### T34b. It is the render *space*, not the metric — all seven fail

The in-domain ladder (n=128) scores every render-tier metric on the same rungs,
each in its own noise floor:

| metric | `solidify` | `chunk_delete_20` | `occ_noise_10` | `material_shuffle` |
|---|--:|--:|--:|--:|
| `kid` | **2.0** | 12.3 | 22.3 | 132.3 |
| `mmd_rbf` | **2.1** | 13.9 | 26.7 | 148.8 |
| `fd` (Fréchet) | **0.5** | 10.8 | 10.8 | 72.4 |
| `1−coverage` | **0.2** | 7.7 | 7.5 | 19.8 |
| `1−density` | **0.1** | 7.2 | 5.3 | 14.0 |
| `1−precision` | **−0.4** | 9.0 | 2.5 | 28.0 |
| `1−recall` | **0.4** | 1.4 | 6.6 | 30.7 |

**`S6_resolves_solidify` is `False` for all seven, while `S4` (material
shuffle) and `S5` (40% chunk deletion) are `True` for all seven.** Deleting a
fifth of the structure is resolved at 7–14 floors; filling its interior is not
resolved by any statistic in the family.

This is the claim the paper should make. Not "DINOv2-KID is blind" — an
encoder-and-statistic complaint a reviewer can dismiss as a bad choice of
either — but **the render representation discards interior structure, so every
statistic computed on it inherits the blindness**. Two domains, two
representations, seven statistics.

### T34c. Honest caveats

- Damage is 3× smaller than on houses (+10.3% vs +31% median blocks), yet the
  appearance separation went to *zero* rather than shrinking proportionally.
- The corpus is selected on having an interior (44.2% of 47,389 have ≥8 sealed
  cells) — the same selection logic as the curated house corpus.
- **The renders are not identical**: 78% of 384 views change, by a median 0.40%
  of pixels. The damage does reach the image; the feature distribution ignores
  it. So this is not a rendering artifact.
- One backbone and one view set. Varying the *statistic* is done here; varying
  the *encoder* is the separate ablation.
- All arms are real data or controlled corruptions — no generative arm is
  scored on this domain.
- Every brick is wool, so `palette_jsd_family` is degenerate here.
- `mmd_rbf` fails G7 in-domain though it passes on houses; recorded, not hidden.


## T35. The noise-floor unit is n-dependent (2026-09-19)

> ### Bottom line
> A separation quoted in "noise floors" is a **detectability-at-this-n**
> statement, not a scale-free quality scale. Measured on the identical
> corruption at two sample sizes: `geom_kid`'s noise floor is 0.1732 at n=64 and
> 0.0848 at n=128 — a ratio of **0.49**, i.e. it falls as 1/n. The signal is
> constant, so the separation grows **linearly in n**: 132.9 → 286.7, ratio
> **2.16**.

| metric | noise sd n=64 | noise sd n=128 | sd ratio | separation n=64 | n=128 | sep ratio |
|---|--:|--:|--:|--:|--:|--:|
| `geom_kid` | 0.17322 | 0.08476 | **0.49** | 132.9 | 286.7 | **2.16** |
| `mv_dino_kid` | 0.00449 | 0.00331 | 0.74 | 0.4 | 1.1 | 2.84 |

**Three consequences, all of which required corrections.**

1. **Every separation figure must carry its n.** "202 noise floors" means "202 at
   n=128"; the same measurement at n=256 would read roughly 400. The abstract
   and §4 now state the n.

2. **Cross-domain comparisons must be normalised.** The houses (n=128) and LEGO
   (n=256) figures were tabulated side by side in the first version of T34 as
   though commensurable. They are not — see the correction there.

3. **"Regenerate at n≥256 changes no qualitative conclusion" is true only of
   orderings.** Every headline number roughly doubles. The claim in the paper's
   limitations section was right about rankings and misleading about magnitudes;
   it now says so.

`mv_dino_kid` scales less cleanly (0.74 rather than 0.49) because its
"separation" is a ratio of two quantities both near zero — which is itself the
finding, and the reason the render row should be read as "indistinguishable"
rather than as a number.

## T36. The blindness is not exact render invariance (2026-09-19)

> **SUPERSEDED BY T37 (2026-09-20).** The headline conclusion of this entry is
> wrong. Under an *opaque* renderer interior fill **is** exact render
> invariance, and provably so. The 86% figure here is also a small-sample draw
> (the full split gives 91.4%), and the cause is mostly a texture-atlas
> artefact in `build_mesh` rather than interiors showing through openings.
> Read T37 before using anything below.

Measured on 60 held-out builds × 4 views (240 renders), real vs `solidify`:

| quantity | value |
|---|--:|
| bit-identical renders | **33/240 = 13.8%** |
| views that change | **207/240 = 86.2%** |
| of changed views, pixels differing | median **1.74%**, mean 2.34%, max 20.7% |
| max channel difference where it changes | median **218/255** |

**This matters for how the claim is phrased.** If `solidify` preserved every
pixel, the blindness would be a theorem: any function of the renders would be
exactly invariant and no encoder could ever see it. It does not. Interior mass
leaks through doorways and windows into a small fraction of pixels — but where
it leaks, it leaks hard: a median maximum channel difference of **218/255**, i.e.
those pixels are almost completely different.

So the honest statement is the stronger-sounding one made precise:

> The corruption is plainly visible in the image — it alters most views, and
> alters them by nearly the full dynamic range in the pixels it touches. Every
> render-space statistic we test still fails to resolve it, while resolving
> corruptions of comparable and smaller magnitude.

That is *not* "the signal is mathematically absent". It is "a large, spatially
concentrated signal is present and every statistic in this family ignores it",
which is a claim about the statistics and the pooling, not about information
theory. The cutaway result (T-ablation) is what shows the signal becomes
resolvable once it occupies more of the frame.

**Correction to earlier phrasing.** §4 said the information "reaches the encoder
and is discarded rather than being absent", citing a mean pixel difference of
0.784/255. That mean is correct but misleading on its own: averaged over the
whole frame it sounds negligible, when in fact it is a near-saturated change
confined to ~2% of pixels. Both numbers now appear together.

## T37. T36 was right about the numbers and wrong about the cause (2026-09-20)

T36 concluded that interior fill is "not exact render invariance" because
`solidify` alters ~86% of renders. Two independent measurements this session
say the numbers were a small-sample draw and the *interpretation* was wrong.

**The numbers, remeasured.** On the whole test split (399 builds, 1,596 views,
`features.render_views` with its own texture pack): **91.4%** of views change
(T36's 86% came from a 60-build subsample; the first-60 slice reproduces 82%),
median **1.75%** of pixels, median max channel **212/255**. A second agent at
1,024 pairs got 91.4% / 1.67% / 211. Every changed pixel is inside the
silhouette, which is bit-identical in all 1,596 views.

**The cause, which T36 got wrong.** Counterfactual texture libraries separate
the hypotheses (1,024 pairs each):

| renderer | bit-identical |
|---|--:|
| opaque + flat tiles | **100.0%** |
| flat tiles, translucency on | 62.5% |
| textured, translucency off | 8.1% |
| deployed | 8.6% |

So under an **opaque** renderer `solidify` is *exactly* invisible — which is a
theorem, not a measurement: filling sealed air cannot change an exterior view,
because any ray reaching a filled cell must first cross an opaque face
bordering exterior air. Verified 1,024/1,024, and surface invariance verified
exhaustively (0 violations / 399 builds, for `hollow` too).

The deployed 91% is **two leaks, only one of them real**:

1. *Translucency* — Minecraft glass in front of a filled void does show the
   fill. Real, and worth 37% of views on its own.
2. *A texture-atlas artefact in `build_mesh`* — tile indices are assigned in
   face-iteration order over **all** exposed faces including interior ones, so
   deleting interior faces changes the tile count, the atlas layout and every UV
   rect, with a `GL_NEAREST` sampler. Worth 92% of views by itself. Confirmed
   structurally in the source and empirically ("atlas unchanged" predicts
   bit-identity at 98.5% with translucency off).

T36's sentence "because interiors show through doorways and windows" is half
wrong: doorway-reachable air is **not sealed**, so `solidify` never fills it.
Windows (translucency) are a channel; doorways cannot be.

**Where the surviving signal dies:** DINOv2 patch tokens that a changed pixel
touches move 0.062 against a between-build scale of 0.517 (12%); CLS 0.011 vs
0.277 (3.9%); 4-view pooled 0.0053 vs 0.268 (2.0%). Attenuation is at
aggregation. Stated as measurement; the "encoder is invariant by training"
story is contradicted by the patch numbers.

**Consequence for the paper, and it is a strengthening.** The claim moves from
"the corruption reaches the image and is discarded" to "under an opaque
renderer no render-and-embed metric *can* separate it, and the little our
renderer does leak is discarded too." Abstract, §6.3 and a new
`app:hull` appendix updated; six claims registered in `audit_claims.py`.

**Open, and cheap:** re-score MV-KID's `solidify` arm under flat+opaque
textures. Part of the residual +0.38 noise floors may be the atlas artefact, in
which case the measured blindness is *deeper* than reported. Nobody owns this.

Also flagged: `outputs/run_20260918_205019_view_protocol_ablation/FINDINGS.md`
says "the information is absent from the input", which contradicts both T36 and
this entry; it should read "absent under an opaque renderer; present but heavily
diluted under ours".

## T38. The residual is wobble, not partial detection (2026-09-20)

Requested follow-up to T37: how much of MV-KID's +0.38 noise floors on
`solidify` is the texture-atlas artefact rather than the corruption? Rescored
the same arms, the same pool draw (`default_rng(seed+1)`) and the same estimator
through four render pipelines, changing nothing but the texture pack.
`scripts/figs/rescore_flat_opaque.py`, artifact
`outputs/figstage/rescore_flat_opaque.json`. Reproduced **exactly** across two
independent runs (identical to six decimals).

| renderer | kid_real | kid_solidify | floor sd | separation | features identical |
|---|--:|--:|--:|--:|:--:|
| opaque + flat tiles | −0.001578 | −0.001578 | 0.002388 | **+0.000** | **yes** |
| flat tiles, translucency on | −0.001689 | −0.002078 | 0.002385 | **−0.163** | no |
| textured, translucency off | −0.001546 | −0.001457 | 0.003497 | **+0.026** | no |
| deployed | −0.001060 | −0.000127 | 0.003581 | **+0.260** | no |

**The lemma is confirmed end to end.** Under the opaque flat renderer the
features are bit-identical (max |Δ| exactly 0.0) so the separation is an
identity, not a small number. This is the occlusion lemma of T37 pushed through
the real bench path rather than through a render comparison.

**The more useful finding: the residual is wobble.** Changing only the texture
pack — which changes nothing about the geometry being probed — moves the
separation from +0.26 to +0.03 to −0.16. That range is wider than the value
itself and it *crosses zero*: with transmission only, the solidified corpus
reads **closer to real than held-out real does**. Neither leak alone reproduces
the deployed +0.26, so they are not additive either.

So the answer to "is the residual the artefact?" is neither yes nor no: there is
no stable residual to attribute. The render metric is flat on this axis, not
weakly sensitive to it, which is what the paper already argues from the
dose-response — this is an independent confirmation by a different route.

**Caveat, and it is the T35/T37 caveat again:** the deployed row here reads
+0.260 where the bench run of record reads +0.38 for the same quantity. The two
estimate their own floors (0.00358 vs 0.00306) from their own draws. Both are
well under one noise floor; neither is transferable to the other.

Written into `app:hull` with the table; five claims registered in
`audit_claims.py`, two of them at **zero tolerance** because "exactly invisible"
and "bit-identical features" are identities and an approximate match would mean
the proof or the implementation is wrong.

## T39. V-KID is not a size detector, and the proof is a mass-matched control (2026-09-21)

Two reviewer objections to a handcrafted 77-d descriptor, answered by
measurement. `scripts/vkid_ablation.py`, artifact
`outputs/figstage/vkid_ablation.json`, n=128, reps=48, seed 0.

### 1. Which components carry the result?

Each variant scored in **its own** noise floor, measured in the same run.

| descriptor variant | dims | solidify | grow |
|---|--:|--:|--:|
| full | 77 | **201.8** | 214.5 |
| pattern only | 55 | 113.9 | 201.3 |
| thickness only | 5 | 158.7 | 75.7 |
| height only | 8 | 8.0 | 28.9 |
| scalars only | 9 | 164.5 | 118.1 |
| without pattern | 22 | 198.0 | 120.9 |
| without thickness | 72 | 176.4 | 216.6 |
| without height | 69 | 199.6 | 213.9 |
| without scalars | 68 | 171.6 | 206.7 |
| **without `interior_ratio_sealed`** | 76 | **197.1** | 211.7 |
| **without `log1p(n_blocks)`** | 76 | **202.2** | 214.9 |
| **without both** | 75 | **197.3** | 211.9 |

**The two suspect scalars carry none of it.** `interior_ratio_sealed` measures
the very thing the probe destroys and `log1p(n_blocks)` is pure size; deleting
either changes the separation by under 3%, and deleting both leaves 197.3 of
201.8. The result is distributed: no single block reaches the full value (best
is the scalar panel at 164.5) and removing any one block leaves at least 171.6.

The blocks are also not redundant. Thickness and the scalars respond more to
`solidify` (158.7 and 164.5 against 75.7 and 118.1); pattern and height respond
more to `grow` (201.3 and 28.9 against 113.9 and 8.0). They measure different
things.

### 2. The mass-matched control, and the double dissociation

`Grow-Shell` dilates the solid **outward** until it has added exactly the number
of blocks `solidify` adds to that build. Matched to within one block on
**128/128 builds**, median 196. Same builds, same material, same mass. The only
difference is whether the new mass lands inside or outside.

| arm | blocks added | render | geometry |
|---|--:|--:|--:|
| `real@solidify` | +196 | **+0.26** | **201.8** |
| `real@grow` | +196 | **+102.27** | 214.5 |

**A metric that counted blocks could not produce this table, because the block
count is the one thing the two arms share.** The render metric resolves the
matched control at 102 noise floors and the probe at 0.26. This is the
interior-blindness claim with the confound removed by construction rather than
by argument, and it is the strongest single piece of evidence in the paper:
mass, material and build are held fixed, and only the location of the change
varies.

Resolves the `\TODO` in §6.4 that asked whether V-KID is a disguised size
detector. Eight claims registered in `audit_claims.py`.

## T40. The rater study: all three pre-registered predictions hold (2026-09-23)

`outputs/run_20260923_human_v5`, 20 sessions, 960 judgements, 19 kept (one
dropped at 0.67 catch accuracy). Textured stimuli, 3D-Craft only, arm labels
stripped, sides randomized. Predictions and their rationale were fixed in
`PRIORITY_PAIRS` before any rater saw the page.

| pair | result | 95% CI | verdict |
|---|--:|---|---|
| `real` vs `real@solidify`, **exterior** | 43/84 = **0.512** | [0.407, 0.616] | contains chance |
| `real#cut` vs `real@solidify#cut`, **cutaway** | 69/90 = **0.767** | [0.669, 0.842] | excludes chance |
| `gabled_house` vs `patchwork` | 93/93 = **1.000** | [0.960, 1.000] | unanimous |
| `real` vs `real@hollow` | 20/33 = 0.606 | [0.437, 0.753] | contains chance |
| catch (`real` vs `uniform_random`) | 113/114 = 0.991 | [0.952, 0.998] | raters attending |

**1. The blindness is informational, not a defect of the metric.** People shown
the exterior renders cannot separate a real build from its solidified copy
either: 0.512, interval centred on chance. The metric is not failing to read
something a person reads. The information is not in the picture.

**2. The information is there to be had, and the view is what withholds it.**
The same raters, the same builds, a view that exposes the interior: 0.767,
interval excluding chance. This is the human counterpart of the cutaway
ablation, and it is what turns "nobody can tell" into "nobody can tell *from
this view*".

**3. The metrics and people disagree, on a pair chosen in advance.** Both
metrics rank `patchwork` above `gabled_house` (BlockScore 223.0 against 300.2;
`geom_kid` 20.0 against 32.8). Raters preferred `gabled_house` **93 times out of
93**. `patchwork` buys good local statistics with globally incoherent structure,
and it fools both tiers while fooling no one.

**Caveat on the Bradley-Terry column.** It puts `gabled_house` first, above
`real`, which is an artifact of the schedule rather than a finding: BT strength
depends on which opponents an arm met, and `gabled_house` met `patchwork` 93
times and beat it every time. Report the per-pair rates, not the BT ranking.

**Power.** The headline pair has n=84 and an interval of about +/-0.10, which
rules out a human preference stronger than roughly 0.62 but does not establish
equivalence; a TOST against a stated margin would. The weakest cell is
`real@hollow` at n=33. Four claims registered in `audit_claims.py`.

## T41. Agentic methods suite: verified offline, no science results yet (2026-10-03)

**There are no live-API results in this entry.** Everything below was run with mock
backends (`--mock`: an offline LLM that returns one fixed house, train-split renders
as dreams) or in tests. It verifies the harness, not any method.

What was verified:

- `python -m blockgen.agentic battery smoke-mock --mock` runs all 17 shipped recipes
  on 2 prompts in about 10 s, with no truncation, writing per-arm npz + manifests
  (provenance incl. `recipe_hash`, `train_split`, `git_commit`), `samples.png`,
  per-build `contact.png`, `contact_<arm>.png`, `variants.png`, `metrics.json` with
  `cost_known` and `would_cost_usd` (`outputs/run_20261003_124201_smoke_mock`).
- `run --recipe dream-multiview --prompts short:2 --mock --renderer textured`
  (`outputs/run_20261003_124055_suite_dream-multiview_mock`): five dreams per build
  (iso, back, interior, plan, mood), with the conditioned views showing the same house,
  matched renders per view, 3 attempts, best column selected.
- The npz files bench unchanged: `python -m blockgen.eval.bench --tier fast` on the
  mock dream-multiview and classic-matched arms
  (`outputs/run_20261003_124312_bench_agentic_suite_mock_smoke`). Both arms are
  disqualified for mode collapse, which is the correct verdict for a mock builder
  that returns one house for every prompt.
- `estimate` (zero network calls, upper bound with every repair round counted,
  completion = 0.3 x max_tokens) gives, per build: classic-matched $0.051, dream
  $0.058, dream-refine $0.113, dream-multiview $0.254. The whole n=12 `dream-v1`
  battery (9 arms, 108 builds) estimates at $7.08.
- A real backend without `--allow-spend` stops with `SpendNotAllowed` before any
  network call.
- Legacy parity: native `classic` reproduces BuildAgent's provider requests, final
  program, blocks and stages for every legacy arm x critique mode x keep_history x 3
  prompts x 4 reply scripts, through the real kit path; the `legacy` method is
  BuildAgent itself. `tests/agentic/test_classic_parity.py` +
  `test_legacy_parity.py`: 101 passed; the untouched legacy
  `tests/test_agentic_agent.py` + `test_agentic_dsl.py`: 57 passed.
- Scorer calibration on oracle dreams (`calibrate --n 32 --seed 0`, 31 distractors,
  chance 0.03, textured renderer; `outputs/run_20261003_130834_calibrate`), top-1:
  `embed` siglipB 1.00, clipB16 0.97, `color_lab` 1.00, `pixel` 0.88, `silhouette`
  0.25. These are ceilings (no dream-to-render domain gap).

Placeholder for the first battery (`dream-v1`), to fill after it runs. PILOT for any
n < 256; `dream-oracle` (`oracle_train`) is a ceiling and never ranked.

| arm | n | BlockScore | geom_kid | faithfulness | dream_gain | would_cost_usd | cost_known |
|---|---|---|---|---|---|---|---|
| classic-matched | | | | | | | |
| dream | | | | | | | |
| dream-refine | | | | | | | |
| dream_vdm | | | | | | | |
| dream-oracle (ceiling) | | | | | | | |

Review fix, scorer recalibration (2026-10-03, after the segmentation change in notes
§36): `calibrate --n 32 --seed 0 --renderer textured`, top-1 `silhouette` 0.28 (was
0.25), `color_lab` 1.00, `pixel` 0.88. `embed` baselines were not re-measured; renders
change only where near-white blocks touch the backdrop.

## T42. Staged agentic methods, and what the first dream pilot showed (2026-10-03)

**First real dream pilot** (`outputs/run_20261003_185128_pilot_dream_v1`; dream,
dream-refine, dream-multiview, classic-matched x 2 prompts; PILOT, no rankings). The
gpt-image-1-mini dreams are good Minecraft iso targets and the builds follow the
dream's shape. Four problems:

1. The match loop barely improves: most match rounds are rejected, and the embed
   similarity between the photo-like dream and the flat render stays in a narrow
   0.56 to 0.61 band, so most of the distance is style, not build error.
2. The `plan` dream was just another exterior (the image model ignores "top-down
   floor plan" when edited from the iso exterior), and the interior cutaway dream
   was weak.
3. On the contact sheet the column header showed the dream score while selection
   used `final_score` (dream-multiview), so the boxed best looked lower than an
   accepted neighbour.
4. `dream` and `dream-refine` were identical: the critic accepted dream v0 and every
   later request hit the shared cache (correct for a paired comparison).

**Staged methods, mock-verified only.** `blueprint` (typed plan, deterministic
shell, massing-anchored exterior dream, rooms built one by one in their own regions
against their own dreams, roof, exterior pass), `floors` (storey by storey, bottom
up) and `turnaround` (one 2x2 sheet split into four view targets) respond to these
findings; docs/agentic-suite.md section 5.4. Fixes 3 and 4 shipped with them
(`sel` header line and `score.json` selection; recipe field `salt`). No real-API run
of any staged method exists, so there are no results for them yet.

- `battery staged-v1 --mock --prompts short:2`: all 14 arms, 28 builds, no
  truncation (`outputs/run_20261003_201629_staged_v1_mock`). The offline `interior`
  rescore separates the mock arms as intended (blueprint 0.68, blueprint-nodream 0.56,
  classic and dream arms 0.36), but mock builds say nothing about quality.
- `run --recipe blueprint,floors,turnaround,blueprint-nodream --prompts short:2
  --mock` (`outputs/run_20261003_201611_staged_mock`), contact sheets inspected.
- `estimate`, upper bound per build: blueprint $0.23, floors $0.16, turnaround $0.064,
  dream $0.058, classic-matched $0.051; the whole `staged-v1` battery (8 prompts, 14
  arms) $17.33.
- Tests: `tests/agentic` 1107 passed, 4 skipped; legacy agent, DSL and deploy
  backend tests pass; legacy files untouched.

| arm | n | BlockScore | geom_kid | interior | dream_gain | would_cost_usd | n_llm_calls |
|---|---|---|---|---|---|---|---|
| (staged-v1, to fill after a paid run) | | | | | | | |

## T43. Player scale and text planning: wired and mock-verified, no results yet (2026-10-03)

No real-API run; every number below is from the offline mock and says nothing about
quality. What exists (notes §38, docs/agentic-suite.md section 5.5): the `scale`
context (rules in blocks for a 2-tall player, canvas note, dream cue), the
dream-reading plan turn (`dream-plan`), the prose-brief blueprint plan
(`blueprint-text-plan`, `plan_source: text`), the controls `dream-noscale` and
`blueprint-noscale`, blueprint rooms of at least 5 x 5 including walls, and the
heuristic `scale_*` metrics in every suite build's metrics.

- Recipe hashes changed on purpose: `dream` 6974c014229e (was fc8963b57243),
  `dream-multiview` 69253ce81bcf (was ce2a800873a3), `classic-matched` 5557761f11dd
  (was 59523f8f03dd), and every recipe extending them, plus blueprint, floors,
  turnaround, best-of-4 and dream-texpalette. `dream-noscale` and
  `blueprint-noscale` carry the old `dream` and `blueprint` hashes. `classic-full`,
  `dsl-classic` and the legacy arms are unchanged.
- `run --recipe dream-plan,blueprint-text-plan,dream-noscale --prompts short:2
  --mock` (`outputs/run_20261003_233413_suite_dream-plan_blueprint-text-plan_dream-noscale_mock`):
  6 builds, no truncation, `plan.txt` and `plan_brief.txt` in each trace. Prompt dump
  of one build per new recipe: `outputs/prompt_review/prompts_scale_plan.md` (since
  regenerated from the follow-up rerun below).
- `battery scale-plan-v1 --mock --prompts short:2`
  (`outputs/run_20261003_233804_scale_plan_v1_mock`): all 9 arms, 18 builds, no
  truncation. The dream-family pairs reuse images as designed (`dream-plan` 0 new
  images, shared with `dream`; `dream_scale_textonly` shared with `dream-noscale`).
  `blueprint-text-plan` also showed 0 new images, but only because the mock gives it
  the same blueprint as `blueprint`; in a real run its prose-brief blueprint differs,
  so its massing render, rooms and dream requests differ and it pays for its own
  dreams.
- `estimate --battery scale-plan-v1`: 72 builds, 888 LLM calls, 200 images, $11.34
  upper bound (no cache sharing assumed), cap $25. Only the dream-family pairs
  save on images, so the real cost sits close to the bound.
- Tests: `tests/agentic` plus the legacy agent, DSL and deploy backend tests, 1244
  passed, 4 skipped; legacy files untouched.
- Review follow-up (same day, prompts changed, recipe hashes unchanged): the
  blueprint scale paragraph now carries only blueprint-unit facts and makes a hall or
  corridor a room at least 5 wide (a 3-wide corridor rect was dropped by the
  repair); the rules give room sizes as open floor, replace "a small house 7-12 per
  side" with "a 2-3 room cottage is about 10-16 per side", and ask for 2 blocks of
  air over stair steps (was 3); the canvas note offers blueprint and floors their own
  2-storey cap; `dream-plan`'s scale line keeps the TARGET's proportions and sizes
  them by its doors and storeys; the JSON turn states the unit once. The `scale_*`
  heuristic drops sloped low cells (attic eaves, overhangs: a gabled hut went from
  0.71 to 1.0 headroom3, a gabled 2-high-ceiling hut still reads 0.3), counts one
  doorway per opening and not a corridor's cells, needs a lintel (no crenels) and a
  non-furniture floor (no bed under a window); the `scale` scorer gives a covered
  build with no opening a door term of 0. Mock rerun
  `outputs/run_20261003_235423_suite_dream-plan_blueprint-text-plan_dream-noscale_mock`
  (6 builds, no truncation); `prompts_scale_plan.md` regenerated from it. Tests:
  1253 passed, 4 skipped.

| arm | n | BlockScore | interior | scale_headroom3 | scale_door_ok | would_cost_usd | n_llm_calls |
|---|---|---|---|---|---|---|---|
| (scale-plan-v1, to fill after a paid run) | | | | | | | |

## T44. First live looks at staged and dream methods on varied prompts (2026-10-04)

Two small paid pilots, gpt-5-mini plus gpt-image-1-mini (low), 32^3 canvas, $1.94 total.
n = 2 and n = 4 per arm: these are visual impressions, not measurements, and the
benchmark needs n >= 128 per arm before any comparison means anything.

**pilot_staged_v0** (`outputs/run_20261003_235637_pilot_staged_v0`, cabin + tower,
blueprint / floors / turnaround / blueprint-nodream / classic-matched, $0.78). Every
build completed. In blueprint, the LLM plan gave a two-room cabin, each room got its own
interior dream, and the builder furnished both rooms from them; the roof stage raised
the dream score from 0.38 to 0.66. Player-scale diagnostics favour the plan-driven arms
(blueprint: headroom3 0.90, door_ok 1.00; classic-matched: 0.61, 0.00). The final
exterior pass damaged finished walls, and `select: best` returned that rejected attempt
because the dream-independent selector preferred it. Fixed by `select: best_accepted`,
now used by blueprint and floors.

**pilot_diverse_v0** (`outputs/run_20261004_004446_pilot_diverse_v0`, three-tier pagoda,
lighthouse on rocks, modern two-storey glass house, Dutch windmill; dream / dream-plan /
blueprint / classic-matched, $1.16):

- `dream` follows its image most visibly: three red and dark tiers on the pagoda, red
  and white bands on the lighthouse, glass bands on the modern house, sails on the
  windmill.
- `dream-plan` (a text plan read from the dream) did not look better than `dream` on
  any of the four, and was worse on the pagoda and windmill.
- `blueprint` turns every request into a house: its rooms-and-shell plan put a box house
  under the lighthouse tower and the windmill, and a box under the pagoda roof, because
  the exterior dream is conditioned on the shell render. It suits houses, not towers,
  mills or monuments.
- `classic-matched` (no image) produced a credible pagoda and lighthouse, so the dream's
  advantage is in following a specific look, not in basic recognisability.

Next: n >= 12 on houses for blueprint vs dream vs classic-matched, and a shape-type gate
(house vs tower/landmark) that routes non-houses away from blueprint.

## T45. Describe stage (spec before, caption after): wired and mock-verified, no results yet (2026-10-05)

No real-API run; every number below is from the offline mock or the no-network
estimate and says nothing about quality (notes §39, docs/agentic-suite.md section 5.6).

- New params on `dream`, `turnaround`, `blueprint` and `floors`: `describe`
  (none, before, after, both), `caption_format`, `expand_into_image`, `show_spec`,
  `caption_keys`, `caption_in_match`, `caption_control` (none, shuffled). Default
  `none` sends byte-identical requests; every existing recipe keeps its hash
  (pinned-hash tests pass).
- `run --recipe dream-describe-before,dream-caption,dream-describe-both,dream-caption-shuffled
  --prompts short:2 --mock`
  (`outputs/run_20261005_110007_suite_dream-describe-before_dream-caption_dream-describe-both_drea_mock`):
  8 builds, no truncation, `describe.md`, `dreams/spec.json` and
  `dreams/iso_v0.caption.json` in each trace. `dream-describe-both` drew 0 new images
  (shared with `dream-describe-before`), `dream-caption-shuffled` 0 (shared with
  `dream-caption`). The shuffled arm's second prompt showed the first prompt's
  caption, its first a library caption. Prompt dump of one `dream-describe-both`
  build: `outputs/prompt_review/prompts_describe.md`.
- `estimate --recipe dream,dream-describe-before,dream-caption,dream-describe-both,dream-caption-shuffled
  --prompts /tmp/diverse_prompts.txt` (4 prompts, upper bound):

| arm | builds | llm_calls | images | est_paid_usd | est_full_usd | per build |
|---|---|---|---|---|---|---|
| dream | 4 | 20 | 0 | 0.2115 | 0.2331 | 0.0583 |
| dream-describe-before | 4 | 24 | 4 | 0.2753 | 0.2753 | 0.0688 |
| dream-caption | 4 | 24 | 0 | 0.2527 | 0.2743 | 0.0686 |
| dream-describe-both | 4 | 28 | 4 | 0.3167 | 0.3167 | 0.0792 |
| dream-caption-shuffled | 4 | 24 | 0 | 0.2527 | 0.2742 | 0.0686 |

  The caption arms and `dream` find their images in the cache from
  pilot_diverse_v0; the spec arms need new ones (their dream prompt changes).
  `estimate --battery describe-v1` (8 detailed prompts, 6 arms): 48 builds, 280 LLM
  calls, 40 images, $3.17 upper bound, cap $12.
- `battery smoke-mock --mock` (`outputs/run_20261005_110203_smoke_mock`): all 30
  recipes including the four new ones, 60 builds, none truncated.
- Tests: `tests/agentic` (new `test_describe.py`, 31 tests) plus the legacy agent,
  DSL and deploy backend tests; legacy files untouched.

| arm | n | BlockScore | in-loop score | describe_disagreement / build | would_cost_usd |
|---|---|---|---|---|---|
| (describe-v1, to fill after a paid run) | | | | | |

## T46. Describe/caption and grid: first live look (2026-10-05)

`outputs/run_20261005_111819_pilot_describe_v0`: the four varied prompts of T44 (pagoda,
lighthouse, modern house, windmill), arms dream / dream-describe-before / dream-caption /
dream-describe-both / dream-caption-shuffled / dream-grid, $0.84. n = 4 per arm, so
these are impressions.

- `dream-caption` looked best and had the highest in-loop dream score (best 0.63 vs 0.53
  for dream, gain +0.10). It is the only arm whose windmill ended with vertical sails:
  its first program laid the arms flat (y fixed at 9), and the match loop, with the
  caption naming "four large diagonal sail arms" and a per-face description, turned them
  upright. Its builds are smaller (652 blocks on average vs 1283).
- `dream-describe-before` was worst. The expanded spec changes the image prompt, so it
  produced different dreams (the pagoda lost its red), and the builds drifted further
  from them (dream score best 0.45).
- `dream-describe-both`: mixed; an upright but plain lighthouse, a tilted windmill.
- `dream-caption-shuffled`: the lighthouse came out as a red box house, as expected when
  the caption describes a different building; the pagoda was unaffected.
- `dream-grid` did not fix the sails. Its program spans the sail cloth over x and z at
  y = 12 to 14 (`fill white_wool 6 12 18 26 14 20`, `fill white_wool 15 12 6 17 14 30`),
  directly against the rule it was given, while correctly putting the hub on the z=max
  face the rule pointed it to. A text rule is not enough for vertical planes; a
  plane-explicit primitive or a validator in the repair loop is the next thing to try.

Total live spend for T44 to T46: $2.78.

## T47. Fully local, text-only: Gemma 4 12B on the existing vLLM server (2026-10-05)

`outputs/run_20261005_130141_local_gemma12b_blueprint` and
`outputs/run_20261005_132342_local_gemma12b_classic`: the six prompts of T44 and T46
(cabin, tower, pagoda, lighthouse, modern house, windmill), gemma-4-12b-it QAT w4a16
served by vLLM with image input disabled, `gen.thinking=false`, `gen.vision=false`
(renders replaced by a text placeholder), $0, no image generator.

- `blueprint-nodream` (room by room): 6/6 builds, 74 LLM calls, 22 min. Every build is
  a gable-roofed box house, including the tower, pagoda, lighthouse and windmill (the
  windmill gets two bare poles). Player-scale diagnostics are good (headroom3 0.83,
  door_ok 0.90), but builds fragment more (5.8 components on average, coherence 0.83).
- `classic` (one program, symbolic repair only, no render critique): 6/6 builds, one
  call each, 91 s total, coherence 1.0. It reads more of the requests than the staged
  arm: the lighthouse is a stone column with a lantern room on a stepped base, and the
  windmill a round tower with a cap and one sail stub. The pagoda and the tower are
  still a gable house and a red-roofed box, and doors are rarely player height
  (door_ok 0.23).
- Compared with gpt-5-mini on the same prompts (T44, T46), the local builds are
  simpler and generic: no tiers on the pagoda, no bands on the lighthouse.

Without thinking the 12B model writes valid programs (every command parsed in both
arms), but it does not plan distinctive shapes. Thinking on was unusable at a 4000-token
cap. Not tried yet: thinking with a large budget, vision on (needs a server restart),
and a local image generator.

Addendum (2026-10-06), thinking on: `outputs/run_20261006_010147_local_gemma12b_classic_think`,
the same six prompts and `classic` arm with `gen.thinking=true`, `gen.max_tokens=12000`.
Every call finished with a program after 3.4k to 6.9k tokens of reasoning (about 1 to
1.5 min each), so the T47 failure was the 4000-token cap, not the model. Thinking changes
what the model attempts more than how well it executes: the tower is now tall and narrow
(it was a box), the pagoda is three stepped tiers (it was a gable house, though now a solid
pyramid with no building under it), the lighthouse a column with a glass lantern on a
rocky base, and the windmill has four upright sails. Execution got worse: the cabin roof
is broken and coherence fell from 1.00 to 0.78.

## T48. The dream loop fully local: Gemma 4 12B (vision) + SDXL-Turbo on CPU (2026-10-06)

`outputs/run_20261006_020248_local_gemma12b_dream_v2`: `dream-gemma-local` vs
`dream-caption-gemma-local` on the six prompts of T47, $0. LLM: gemma-4-12b-it QAT via
vLLM with image input on (at most 4 images a request, so `gen.max_images: 4` keeps the
newest), thinking on, 12k tokens. Dreams: SDXL-Turbo, 4 steps, bf16 on CPU (3.7 s an
image after a 19 s load), `compact` dream style. 12/12 builds, no truncations. The first
attempt (`run_20261006_012558_local_gemma12b_dream`) truncated 8 of 12 builds on the
server's image limit, which is what `max_images` fixes.

- SDXL-Turbo dreams are recognisable and detailed but ignore "plain white background":
  every one has grass, and the lighthouse and windmill sit in water and terrain. The
  windmill dream is two structures. Dream scores are correspondingly low (best 0.36 and
  0.35 against 0.53 for gpt-image-1-mini in T46), since the scene, not the build, fills
  most of the image.
- Plain `dream` copies the scenery: grass platforms under the cabin, tower and windmill,
  and a pool of water around the lighthouse. Following the dream literally includes its
  backdrop.
- With the caption (`surroundings_to_omit` lists the terrain), the builds drop the
  scenery: coherence 1.00 against 0.83, and headroom3 0.96. Both arms get the
  lighthouse right (white shaft, red lantern) and the tower's red roof; the caption arm's
  pagoda is stepped tiers. Neither windmill has sails.

Next for the local path: cut the dream's background before it is shown or scored
(foreground segmentation to white), since SDXL-Turbo cannot take a negative prompt.

## T49. Attention as adjacency, revisited: the trained voxel model has a neighbour head (2026-10-07)

Follow-up to T28, built in `tools/misc/attention_graphs/` (visualizer plus scoring; the
AUROC and distance-stratified AUROC are imported from `scripts/probe_llm_attention.py`, so
the numbers are on T28's scale). Run: `outputs/run_20261006_231900_attn_adjacency/`
(`summary.md`, per-model `results.json`, `head_scores.csv`, figures). Literature memo:
`tools/misc/attention_graphs/RESEARCH.md`. T28 left three questions open. This run
answers them with one more control: **hard pairs**, block pairs that share two of x/y/z,
where matching identical coordinate tokens scores exactly 0.5. `hard_strat` removes the
recency and the copy shortcut together. Heads are chosen on half the items and reported
on the other half.

| question | best head | shortcut baseline (same items) | distance probe, held-out |
|---|---|---|---|
| voxel AR (`canon16_flat`, 60 val builds), raster, query on the `Z` token | L4H2: strat **0.982**, hard_strat **0.871** | recency hard_strat 0.488 | hard_strat 0.975 (layer 3) |
| same model, shuffled block order | 0.597 hard_strat | recency 0.503 | hard_strat 0.964 (layer 2) |
| Qwen2.5-1.5B, absolute `x y z block` lines, shuffled | L8H3: auc 0.882, hard_strat 0.664 | coord_match auc 0.939 | hard_strat 0.917 |
| Qwen2.5-1.5B, coordinate-free (parent index + direction), random growth order | stated parent top-1 0.335 | recency 0.268 | hard_strat 0.655 |

**The voxel AR model learned a neighbour head; the frozen LLMs did not.** In the voxel
model's own training order, head L4H2 ranks true 6-neighbours far above both shortcuts.
Its attention peaks on the block directly below (32% of argmaxes) or one row behind
(27%). Both targets sit a variable number of tokens back, so the head locates them by
content, not by a fixed offset. The sequence does contain coordinate tokens, and the
model-free `coord_L1` oracle still scores 1.000, so this is a learned "differs by one"
relation over coordinate embeddings rather than geometry recovered from nothing. One head
also covers one neighbour direction, not the full graph (AP 0.55 at a 1.7% edge rate). The
head fails when block order is shuffled, which this raster-only model never saw in
training. A linear distance probe on the same frozen states stays at 0.96 to 0.99 in both
orders: the representation encodes adjacency even where the head does not use it.

**On a frozen LLM, T28 stands.** With coordinates printed, the best shuffled-order head
is a copy head: 81% of its attention lands on identical earlier tokens, against 7.9% for
uniform attention. On hard pairs it scores 0.66. With coordinates removed (the
coordinate-free format), the information is simply not in the input. On contacts the text
never states, the best head scores 0.660 stratified, level with the same-block-name
baseline (0.654). Qwen2.5-Coder-1.5B is within noise of the base model on every arm. LEGO
stud graphs behave like the absolute format.

**What this means for native graph generation.** An adjacency-reading head can emerge
from training on builds alone, and adjacency is linearly readable mid-stack. A frozen
model gives nothing to steal, so the edge decision has to be trained into attention. The
next experiment is a coordinate-free parent-pointer serialization for the voxel AR model,
with one head supervised to point at the parent (LISA-style; Strubell et al. 2018) and
masked to legal faces, compared against the absolute format. Closest prior work: PolyGen
(Nash et al. 2020, pointer attention over earlier vertices), LISA, and LegoGPT/LegoACE for
the domain.

Caveats: one AR checkpoint and one seed; 24 report items per LLM arm, so differences
below about 0.03 are noise. Gemma-3-4b-it was not run (no free GPU memory).

## T50. Nodes in, adjacency out: a block list with no positions, decoded to a build (2026-10-07)

Prototype in `tools/misc/adjacency_gen/` (README has full tables; literature memo
`RESEARCH.md`). The model sees only block (or 3D-BPE piece) types plus a random per-node
identity feature: no coordinates, and no positional encoding. It predicts, for every
pair, "no contact" or a direction/offset label, with the logits tied so +x one way is -x
the other way. It is trained with random edge masking. Generation starts from a held-out
block list with every edge masked and commits the most confident edges, optionally with a
realizability check: nodes are placed by BFS, and an edge that would put two nodes in one
cell is rejected. Data: 6x6x6 house windows (voxel nodes, median 58 blocks) and 8x8x8 piece
chunks (about 46 pieces). One seed, 35 minutes of training per model, 200 held-out lists.
Runs: `outputs/run_20261007_111458_adjacency_gen_voxel`,
`outputs/run_20261007_112531_adjacency_gen_piece` (plus a negative `_voxel_prefix`).

**Infilling works; generation from nothing is carried by the check.** At mask rate 0.25
the voxel model reaches edge F1 0.85 with 0.99 direction accuracy (piece: 0.69 / 0.89).
At full mask it reaches AP 0.15 against 0.11 for a block-type prior (piece: 0.21 vs 0.15),
so on its own it knows little beyond which types touch.

Generation, distance to real chunks (W1 on fill and support; lower is better):

| arm | voxel fill / support | piece fill / support | collisions per build (voxel) |
|---|---|---|---|
| model + check | 0.092 / 0.048 | 0.110 / 0.047 | 0 |
| model, no check | 0.069 / 0.023 (on only 62% of nodes placed) | 0.091 / 0.061 | 23.4 |
| type prior + check | 0.266 / 0.248 | 0.258 / 0.266 | 0 |
| random tree + check | 0.198 / 0.178 | 0.214 / 0.202 | 0 |
| retrieval (nearest training block counts) | 0.059 / 0.039 | 0.093 / 0.059 | 0 |
| piece, grow decoding (one growing component) | | **0.037 / 0.012** | 0 |

Without the check, 34 to 39% of nodes collide and only about 70% end up connected. With
it, the model beats the type prior and random trees by a wide margin on every shape
statistic. That is the model's contribution, since all three share the same decoder.
Retrieval still wins for voxel nodes, because it returns real chunks. For pieces,
grow decoding beats retrieval. The samples match the numbers: model-plus-check voxel
builds are coherent fragments (a brick wall under a plank roof, layered snow on dark oak,
a terrace on a stone pillar), random trees are blobs, and the type prior gives tangles.
Piece grow samples are the best looking (walls with roof overhangs, a door frame on a
grass ledge), with some plank rods too long. Implied-edge consistency is low (0.03): the
model predicts a tree-like subset of the contacts its own realized build contains.
Training half the batches on grown-prefix masks made every arm worse and produced
single-block filaments.

**Reading.** As an infilling model the idea works. As a generator, the geometric check
is the decoder rather than a repair step. This points to the memo's recommended
formulation: an inventory-conditioned spanning-tree decoder (choose a remaining block plus
an attach face, masked to legal faces; T24's pick_n_place restricted to the inventory),
which merges with T49's supervised pointer head. Next steps there are several seeds,
larger chunks and geom_kid.


## T51. World-crop pilot: 400 archived worlds → 2,651 structure crops (2026-10-07)

Source: Minecraft Archive Project `Worlds_201407` (archive.org), 400 of 1,774 archives,
evenly strided. Pipeline `blockgen/data/worldcrop/run.py --month 201407 --limit 400
--workers 12`; outputs `data/minecraft/worldcrop/201407/`.

| Stage | Count |
|---|--:|
| Archives processed | 399 |
| Java worlds read / Bedrock / unreadable or non-world | 381 / 4 / 17 |
| Chunks decoded | 911,224 |
| Worlds routed flat / heuristic (flat rejected by ore check) | 213 / 168 (41) |
| Candidate components dropped as ore-like families | 4,740 (95% in 10 worlds) |
| **Crops kept** (flat / heuristic / whole schematic) | **2,651** (1,015 / 1,606 / 30) |
| Saved `.npz` / oversize (>320 blocks, indexed only) | 2,557 / 94 |
| Worlds with ≥1 crop | 366 |
| Crops per world, median / p90 | 3 / 14 |
| Largest side, median / p90 | 26 / 128 |
| Fit 32³ / 48³ / 64³ / 128³ | 1,562 / 1,909 / 2,068 / 2,386 |
| Worker-seconds total / per archive | 5,266 / 12.5 |
| Wall time, 12 workers | ~8 min |
| On-disk size of crops + indexes | 31 MB |

Extrapolation: ~1,800 archives per month × 89 months ≈ 160k archives ≈ 1.0M crops of which
~0.6M fit 32³, in ~2 days on 24 cores and ~12 GB of compressed crops.

Caveats: no class labels yet (provenance titles/descriptions are free text); WorldPainter
terrain passes as built; nether/end skipped; 1.13+ palette decoding untested on real data;
corpus is pretraining-only (no licence).

## T51. Agentic arms on the houses32-v1 benchmark: dream vs compute-matched classic, plus dream-caption (2026-10-07)

The first agentic arms run at the protocol's n. Prompts are the GrabCraft titles of the
houses32-v1 test split, 133 of them, phrased as `<title> (a Minecraft <category>)`
(`configs/agentic/prompts/houses32_test_grabcraft.txt`). All arms use gpt-5-mini, and
the dream arms use gpt-image-1-mini. dream and classic-matched ran on all 133 prompts.
dream-caption ran on a seeded random 100 of them (`..._shuffled.txt`), which is what the
remaining budget allowed. Its n is below `min_n` = 128, so it is scored but not
comparable under the protocol. Runs:
`outputs/run_20261007_092654_bench_houses32_{dream,classic}`,
`outputs/run_20261007_145754_bench_houses32_dream_caption`. Bench:
`outputs/run_20261007_185345_bench_agentic_dream_vs_compute-matched` (`--tier both
--n 133 --baselines`). Spend was $12.18 in total, including $1.56 from a first attempt
that was killed when the session ended; the rerun replayed those builds from the cache.

| arm | n | BlockScore ↓ (worst pillar) | appearance | geometry | geom_kid ↓ | mv_dino_kid ↓ | voxel_diversity (real 0.902) | dup | $/build |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| dream-caption | 100 | **165.0** | **82.6** | **165.0** | **16.9** | **0.215** | 0.795 | 0 | 0.048 |
| dream | 133 | 191.3 | 131.4 | 191.3 | 19.6 | 0.342 | 0.786 | 0 | 0.040 |
| classic-matched | 133 | 194.4 | 154.6 | 194.4 | 19.9 | 0.403 | 0.833 | 0 | 0.045 |
| real@solidify (control) | 133 | 213.4 | 0.2 | 213.4 | 21.9 | -0.001 | 0.818 | 0 | |
| patchwork (baseline) | 133 | 271.4 | 271.4 | 195.2 | 20.0 | 0.708 | 0.897 | 0 | |
| gabled_house (baseline) | 133 | 318.5 | 315.4 | 318.5 | 32.7 | 0.824 | 0.919 | 0 | |

**Dreaming alone does not beat the compute-matched control at this n.** dream and
classic-matched share a significance group on both distribution metrics (geom_kid "cd" /
"cd", mv_dino_kid "d" / "de", Holm-corrected paired tests). The thumbnails suggest
dream is more varied in material and closer to its prompt (classic draws almost every
prompt as the same oak gable cottage). That shows up only as a better appearance pillar
(131 vs 155) and a non-significant mv_dino_kid edge. Geometrically the dream builds are
*less* diverse (0.786 vs 0.833). The practical checks favour the control: door at player
scale 73% vs 52%, headroom 93% vs 87%, and one connected piece 97.7% vs 94.7%.

**dream-caption is the best agentic arm, and the caption is what moves it.** It beats
dream (+0.128 mv_dino_kid, p_holm 0.009) and classic-matched (+0.188, p_holm 0.009) on
the render metric. It has the best appearance pillar (82.6) and the best BlockScore.
That repeats the T46 pilot and the local T48 run at the protocol's prompts. Caveats: n=100
on a different random subset, so it is not protocol-comparable, and its geom_kid lead
over the other two is not significant.

**All three agentic arms beat every procedural baseline and the real@solidify control**
on BlockScore, with geometry as the worst pillar for each. None memorizes (dup 0;
dino_nn_percentile 0.48 to 0.59, real 0.54). The interior remains the gap:
enclosed_air_ratio is 0.04 against a real 0.11.

**Next:** run dream-caption on all 133 prompts (about $5) so it becomes
protocol-comparable. Wire prompts into the fidelity pillar (bench WP8), because the
"dream follows the prompt" claim is currently measured only by eye.

## T52. Face nodes with coarse-shape conditioning; the connection-by-connection merge decoder (2026-10-07)

Follow-up to T50 in `tools/misc/adjacency_gen/` (README section "Face nodes with
coarse-shape conditioning"). Each block becomes 6 face tokens: type, face direction and
a random per-block identity. The output is three N×N per-axis glue matrices plus an
exposed/buried logit per face. That makes the face matrix symmetric, lets +x pair only
with -x, and prevents a block gluing to itself, all by construction. The condition is one
token per coarse cell (count and dominant type) with a 3D position, in a joint
transformer. Data: 8³ windows of 40 to 120 blocks (median 89); the block list still
comes from the real chunk. 200 held-out items, one seed, about 32 minutes of training
per model. Runs: `outputs/run_20261007_1716*_adjacency_faces_{f2,f4,oracle}_resid`,
`..._163543_..._nocond`, `..._17*_adjacency_faces_merge_{f2,f4,oracle,nocond}`.

| arm (f=2 unless noted) | coarse IoU | voxel IoU | typed F1 | glue F1 |
|---|--:|--:|--:|--:|
| grow, check | 0.49 | 0.25 | 0.16 | 0.51 |
| grow, check + coarse mask | 0.97 | 0.48 | 0.24 | **0.54** |
| random policy, check + coarse mask | 0.96 | 0.45 | 0.18 | 0.38 |
| naive upsampling | 1.00 | 0.43 | **0.33** | 0.47 |
| retrieval by coarse grid | 0.76 | **0.50** | 0.11 | 0.13 |
| grow, no conditioning | 0.26 | 0.12 | 0.07 | 0.46 |
| grow, oracle per-block cell + mask | 0.99 | **0.55** | **0.40** | **0.69** |
| merge, check + window-extent test | 0.37 | 0.17 | 0.09 | 0.44 |
| merge, overlap check only | 0.22 | 0.04 | 0.01 | 0.33 |
| merge, random policy | 0.30 | 0.11 | 0.06 | 0.33 |
| one-shot matching (Sinkhorn + Hungarian) | 0.26 | 0.07 | 0.04 | 0.12 |

**The grow decoder does real work, but the condition does the shaping as a mask.** Under
identical masks the model beats a random policy on glue F1 (0.54 vs 0.38) and on
support, and it beats naive upsampling on voxel IoU and glue F1. Upsampling keeps the
best typed F1. Unmasked, the model follows the condition only weakly (coarse IoU 0.49,
against 0.26 unconditioned). Telling each block its coarse cell is the largest lever in
every column. Samples: grow plus mask is closest to the real chunks; upsampling and the
random policy give the right outline filled with speckle; one-shot gives rods and slabs.

**The merge decoder (inventory known, one glued pair per step, clusters merge) does not
work as trained.** Free-running samples are one-block-thick lines with only the overlap
check, and flat sheets once a window-extent test is added. A random policy scores higher
on voxel IoU. The debug found no train/inference skew: a T24b-style guard test confirms
that incremental state features equal a from-scratch rebuild. It found two structural
causes instead. First, the early training target is nearly uninformative. With every
block alone, 43% of all candidate pairs count as correct, because any block of a type can
stand in for another of that type. The teacher-forced accuracy (0.77 to 0.93) therefore
mostly measures which types and axes go together, and greedy decoding over-picks the x
axis (56% vs 35% in real data). Second, free-running states drift off distribution:
components grow elongated (2.96 vs 2.0 in teacher forcing), and extending a line never
collides. Merged clusters also have no shared frame, so the coarse condition cannot act
as a mask (as a hard mask, only 1% of builds complete). Training on the decoder's own
rollouts was not tried. Diversity: two temperature-1 grow samples of one input overlap
at voxel IoU 0.29 (0.43 with the oracle).

**Reading.** Two findings point to the same fix. The oracle cell index is the biggest
lever, and merging fails for lack of a frame. So: first assign each block to a coarse
cell (a set-to-cell assignment, sampled), then resolve connections within and between
neighbouring cells. Every cluster then has an absolute frame from the start. The early
target becomes informative, and the coarse condition can constrain every step.
Same-type interchangeability suggests predicting connections between type slots rather
than particular inventory items.

## T53. Full houses: cell assignment, then connections (2026-10-08)

**Question.** T52's strongest lever was telling each block its coarse cell. Does a two-stage generator, which first samples a block-to-cell assignment and then connects blocks inside that frame, carry from 8x8x8 windows to full houses?

**Setup.** houses_32 v1 split, every house with no size cap: 1,863 train (7,452 with yaw rotations), 399 test, 155 to 7,202 blocks (median 623). Coarse condition at f=4.
- Stage 1: a transformer over coarse cells plus an inventory token (1.9M params) predicts per-cell type distributions as a correction to a heuristic prior. IPF then turns them into an integer assignment that matches both the cell counts and the inventory exactly.
- Stage 2 grow: T52's grow decoder over type slots, as a 3D U-Net with a coarse-cell transformer at the bottleneck (3.6M params). An action is "a free cell next to the build, a slot type", restricted to cells that still hold that slot.
- Stage 2 merge: connections added one at a time with all blocks known. Each cluster carries a box of allowed translations that keeps its members in their cells. One representative per slot cuts the share of correct first-step candidates from 43% (T52) to about 13%. A variant trains on its own rollouts (DAgger).
- One seed. Grow trained 40 min, merge 40 min per arm, stage 1 10 to 12 min; up to four jobs shared the GPU. Code is in `tools/misc/adjacency_gen/` (cells.py, stage1.py, grow_cells.py, merge_cells.py, eval_cells.py); 40 tests pass.

**Results (f=4, 399 test houses).** "Single" means one connected piece holding every block. Shift IoU is the best voxel IoU over shifts of up to 2 cells.

| arm | single | voxel IoU | shift IoU | typed F1 | glue F1 |
|---|--:|--:|--:|--:|--:|
| stage 1 + grow | 0.75 | 0.322 | 0.364 | 0.196 | 0.580 |
| oracle assignment + grow | 0.76 | 0.326 | 0.369 | 0.245 | 0.654 |
| heuristic assignment + grow | 0.76 | 0.315 | 0.356 | 0.204 | 0.592 |
| random policy (stage 1 / oracle assignment) | 0.55 / 0.52 | 0.284 / 0.283 | 0.296 / 0.294 | 0.148 / 0.184 | 0.420 / 0.504 |
| naive upsampling | 0 | 0.246 | 0.254 | 0.155 | 0.406 |
| retrieval | 1 | 0.320 | 0.381 | 0.049 | 0.096 |

Merge, on the 297 houses with at most 1,024 blocks:
- Teacher-forced: 0.7% (stage 1) and 2.7% (oracle) of builds finish as one cluster, with about 6 clusters left and 8 to 9% of blocks dropped. Voxel IoU is 0.27.
- Own-rollout training: 0% finish and 22% of blocks are dropped.
- Grow on the same houses always completes, with voxel IoU 0.31.

**Reading.**
- The stage 2 model works locally. Given the same assignment, it beats a random policy by about 0.15 glue F1 and keeps 75% of builds in one piece against about 53%.
- It does not recover house geometry beyond the condition. Voxel IoU is about 0.32 for every learned arm, the same as retrieval. A 4^3 cell holds up to 64 blocks, so f=4 pins far less than T52's f=2 condition on windows (0.48 to 0.55).
- The oracle assignment adds glue F1 (+0.07) but almost no IoU, because knowing which types go in a cell does not say where they go inside it.
- The learned stage 1 beats the heuristic on its own target (1.53 vs 1.89 nats per block), but not after building.
- Samples have the right outline and materials, with porous volumes and no clean walls or roof slopes (`run_20261008_020520_adjacency_cells_eval_f4/samples.png`).
- f=8 is worse (grow IoU 0.22 to 0.23).
- Merge no longer collapses into lines, but it deadlocks: clusters cannot join under the box, overlap and position checks. Training on its own rollouts made it worse.

**Next.** A finer frame: f=2 on full houses with hierarchical stage 1 (f=4 to f=2), or per-cell occupancy patterns. Score with geom_kid and a retrieval-novelty check at n>=256 with more seeds. For merge, add a split/repair move, or accept any merge consistent with the condition.

Runs: `outputs/run_20261008_0{10026,11130,12320}_adjacency_cells_stage1_*`, `run_20261008_010024_adjacency_cells_grow_f4`, `run_20261008_014046_adjacency_cells_grow_f8`, `run_20261008_010618_adjacency_cells_merge_f4`, `run_20261008_011351_adjacency_cells_merge_dagger_f4`, `run_20261008_020520_adjacency_cells_eval_f4` (main), `run_20261008_022124_adjacency_cells_eval_f8`, `run_20261008_014701_adjacency_cells_eval_f4_k1`.

**T51 addendum (prompt coherence, n=133).** The dream-caption arm was completed to n=133 (`outputs/run_20261008_004051_bench_houses32_dream_caption133`, $1.30 paid; session total $13.48). Prompt coherence was scored with `bench/faithfulness` primitives from a scratch script, so the bench files were not modified. The test asks whether CLIP-L ranks a build's own prompt first among 32 (chance 0.031).

| | top-1 | MRR | material agreement |
|---|--:|--:|--:|
| real builds (ceiling) | 0.218 | 0.349 | 0.35 |
| classic-matched | 0.188 | 0.319 | 0.40 |
| dream-caption | 0.158 | 0.294 | 0.17 |
| dream | 0.128 | 0.286 | 0.33 |

No paired difference is significant (dream minus classic MRR: -0.033 [-0.083, +0.015]). SigLIP-B sits at chance even on real builds. GrabCraft titles are weak prompts. Material agreement applies to only 20% of prompts.

**T51 addendum 2 (bench rerun at n=133, 2026-10-08).** The bench was rescored with dream-caption at full n (`outputs/run_20261008_025647_bench_agentic_dream_vs_compute-matched`). All three arms now meet min_n.

| arm | geom_kid | mv_dino_kid |
|---|--:|--:|
| dream-caption | 17.65 | 0.217 |
| dream | 19.62 | 0.342 |
| classic-matched | 19.94 | 0.403 |

The dream-caption lead on mv_dino_kid holds at qualifying n: +0.125 vs dream and +0.186 vs classic-matched, p_holm 0.009 for both. On geom_kid the three agentic arms share a Holm group and are not separated.

## T54. World-crop terrain gate: four rounds against a 192-crop hand-labelled hold-out (2026-10-11)

Setting: `blockgen/data/worldcrop/gate.py`, HistGradientBoosting on 295 hand features
(id histogram, family fractions, heightmap/column statistics, exposure, enclosure,
symmetry, planarity, vertical profile). Positives: 9,391 labelled builds (GrabCraft 6,560,
3D-Craft 2,534, text2mc houses 297). Hold-out: 192 random saved crops labelled from
numbered sheets (106 built / 65 fragment / 8 terrain / 13 unsure), never trained on.

| version | added negatives | n_neg | terrain rejected /8 | built rejected /106 | fragment rejected /65 |
|---|---|---:|---:|---:|---:|
| v1 | natural chunk patches (30 worlds, 2014-07) | 4,439 | 4 | 0 | 0 |
| v2 | + synthetic solid plates / mounds / ridges / pillars | 8,439 | 4 | 1 | 0 |
| v3 | + synthetic patchy surface skins | 12,439 | 4 | 2 | 0 |
| v4 | + 10,000 crops mined by composition (≤3 ids, ≥97% terrain family) | 22,439 | **8** | 3 | 0 |

Corpus scoring with v4 (810,487 saved crops, 46 months): 746,077 pass at p ≥ 0.5 (92.1%),
56,329 reject, 7,944 in [0.35, 0.65). Passing crops that fit 32³ / 48³ / 64³ / 128³:
433k / 532k / 586k / 682k; 328k passing crops have ≥ 500 blocks. Reject sheet is terrain
with about one built piece in ten (spawn boxes, a lighthouse on rock, decorative plates).
Fragments are untouched by design and remain about a third of the passing set.

## T55. Corpus manifest and dedup over 926k builds (2026-10-12)

Setting: `blockgen/corpus/{manifest,dedup,tiers}.py` over 915,853 saved world crops
(47 months) and 10,461 curated cache rows. Canonical hash folds translation and the
eight horizontal symmetries; MinHash with 128 hashes, LSH 16×8, accept at estimated
Jaccard ≥ 0.8, builds under 20 blocks excluded from the near pass.

| step | rows folded | clusters |
|---|---:|---:|
| exact (canonical hash) | 216,038 | |
| near (MinHash/LSH) | 61,359 | |
| total | 277,397 (29.9%) | 648,917 |

Duplicate rate by block count: <50: 38.2%, 50-200: 34.9%, 200-500: 30.5%, 500-2k: 26.1%,
>2k: 23.5%. Largest cluster 9,323 rows across 2,908 worlds. Clusters present in ≥ 5
distinct worlds: 6,444, holding 139,866 rows (15.1%); the top 24 by world count are all
vanilla village pieces (sheet `data/minecraft/corpus/sheet_top_clusters.png`), so
recurrence across worlds is used as the generated-content filter. 164 curated builds
have world-crop duplicates.

Tiers after dedup and recurrence filtering: pretrain-all 647,838 builds (10.7B blocks,
358k fit 32³); pretrain-clean 602,406 (6.3B, 340k); anneal 281,240 (5.7B, 66k);
eval-holdout 798; vanilla 6,444. Wall time: manifest 70 min on 6 workers, dedup 26 s.

### T55.1 Kinds over the manifest (2026-10-13)

Rule: mineshaft = share of {planks, fence, rail, powered/detector rail, cobweb, torch,
chest, monster egg} among member voxels at or above 0.9 and a rail or cobweb present;
mixed = share in [0.5, 0.9) with the tell; village = dedup cluster in 5+ distinct
worlds. Hand check on random 72-crop sheets: mineshaft 70/72, mixed about half shafts,
below 0.5 all player work. Sheets: `data/minecraft/corpus/sheet_mineshaft.png`,
`sheet_mineshaft_mixed.png`.

| kind | rows | share |
|---|--:|--:|
| build | 714,398 | 77.1% |
| village | 116,483 | 12.6% |
| mineshaft | 85,376 | 9.2% |
| mixed | 10,057 | 1.1% |

Tiers rebuilt with `kind` filters (builds / blocks / fit 32³): pretrain-all 647,838 /
10.67B / 357,985 (unchanged); pretrain-clean 547,789 / 6.30B / 297,659 (was 602,406);
anneal 274,413 / 5.66B / 65,171 (was 281,240, 6,827 mineshafts removed); vanilla
61,130 / 160M / 46,503 (was 6,444, now includes mineshaft representatives).

## T56. Structure templates from the 1.21.1 client jar (2026-10-13)

`python -m blockgen.corpus.templates`: 1,180 templates, 1,113 with blocks, 1,007 in
the `templates` tier after the 20-block floor and dedup (0.5M blocks, 991 fit 32³).
State mapping: 396 distinct block states, 0 fall back to stone after overrides.
Tiers with templates added as a source (builds / blocks / fit 32³): pretrain-all
648,845 / 10.67B / 358,976; pretrain-clean 548,732 / 6.30B / 298,586; anneal
275,200 / 5.66B / 65,942. Sheets: `data/minecraft/corpus/sheet_templates_random.png`,
`sheet_templates_largest.png`.

| family | templates | note |
|---|--:|---|
| village | 483 | five biomes plus zombie variants; houses, paths, decor, pens |
| trial_chambers | 170 | interiors in solid stone |
| bastion | 167 | interiors; largest pieces |
| trail_ruins | 84 | |
| woodland_mansion | 73 | rooms, not the whole mansion |
| ancient_city | 58 | interiors |
| underwater_ruin | 48 | |
| end_city, shipwreck | 20 each | |
| fossil, nether_fossils | 30 | |
| ruined_portal, pillager_outpost, igloo | 27 | |

## T57. First runs through the shared trainer: smoke and the houses_32 scratch baseline (2026-10-14)

`python -m blockgen.train --config configs/train/houses32-scratch.yaml`: 5.5M parameters
(d 256, 6 layers, phase4 PE), token-budget batches of 32k, 8,000 steps, 226M tokens,
22.8 minutes on the 5070 Ti including evaluation and sampling. Train split
`sft-houses32` (1,863 whole builds), validation `eval-holdout` (798). Run dir
`outputs/run_20261014_101448_houses32_scratch_5m`, line 3 of `outputs/registry.jsonl`
(lines 1 and 2 are the two smoke runs of 2026-10-14).

| step | holdout NLL / token |
|--:|--:|
| 1,000 | 1.700 |
| 2,000 | 1.147 |
| 4,000 | 0.72 |
| 6,000 | 0.616 |
| 8,000 (final) | 0.532 |

Fast-tier scorecard on 64 samples (median 1,260 blocks against a real 904; 0 empty):

| arm | BlockScore | worst pillar | geom_kid | nn_iou | dup | components |
|---|--:|---|--:|--:|--:|--:|
| houses32_scratch_5m | 75.75 | geometry 75.8 | 12.5 [11.5, 13.8] | 0.338 | 0.00 | 17.1 (real 2.7) |
| real@solidify (control) | 130.94 | geometry | 21.8 | | | |
| real@monochrome (control) | 46.52 | palette | -0.2 | | | |

Status `ok`, so the number is rankable; the run neither memorises (train_verbatim is
disqualified by the protocol, this arm has duplicate rate 0 and nearest-neighbour IoU
0.34) nor collapses. Its failure is fragmentation: 17 components per build against 2.7
real, and 65% of components floating, which the samples sheet shows as layered plates
and scattered pieces. That is the number the pretrain-then-SFT partner has to beat.

Smoke run (`configs/train/smoke.yaml`, 60 steps, 1.7M tokens): holdout NLL 3.31,
BlockScore 45.9 on 16 samples (too few to rank), there only to prove the loop, the
sampler, the benchmark call and resume work.

## T58. Pretrain on anneal, then SFT, against scratch: the first comparison, 5M (2026-10-14)

Three registry lines, all 5.5M parameters, same tokenizer, same holdout.
`anneal_pretrain_5m`: 20,000 steps on the anneal shard (591M tokens, 31.6 min).
`houses32_sft_5m`: 8,000 steps on `sft-houses32` from that checkpoint, lr 1e-4.
`houses32_scratch_5m` is T57.

| run | tokens | holdout NLL | BlockScore (n=64) | worst pillar | median blocks | median length |
|---|--:|--:|--:|---|--:|--:|
| scratch | 226M | 0.532 | 75.75 | geometry 75.8 | 1,260 | 7,596 |
| pretrain only | 591M | 0.546 | 101.58 | geometry 101.6 | 1,093 | 5,168 |
| pretrain + SFT | 591M + 226M | **0.461** | 80.50 | geometry 80.5 | 1,757 | 8,198 |

Pretraining on the world-crop anneal tier gets within 0.014 nats of the scratch model on
houses it never trained on, and SFT from it lands 13% below scratch in holdout NLL, the
first direct evidence that the archive corpus transfers. The samples
(`outputs/run_20261014_110921_houses32_sft_5m/samples.png`) are visibly more house-like
than scratch: floors, walls and roofs instead of plates and scattered pieces. The
scorecard disagrees, and the reason is a bug, not the model. Median sample length for the
SFT run is 8,198 tokens, which is the generation budget: samples did not end, they were
cut. Cause: 238 of the 1,863 training houses (12.8%) exceed the 8,200-token context and
the trainer truncated them, so one sequence in eight taught the model that a long build
never emits EOS. Samples at 1,757 blocks against a real 904 pay for that in the geometry
pillar. The scratch run suffered the same (7,596 median length). Fix in
`blockgen/train/data.py`: over-length sequences are dropped, never truncated, for both
training and evaluation batches; the holdout loses 62 of 798 builds to the same rule, so
v2 NLLs are over the first 256 of those 736 builds (`eval_seqs: 256` in
`configs/train/base.yaml`; `eval_batches` in `blockgen/train/data.py` takes the first
`n_seqs` sequences that fit, in fixed order) and are reported as their own rows. Every
trainer run recorded through T68 scored that 256-sequence subset; runs from 2026-10-31
on with `final_eval_all` on (the default) also score every fitting sequence in the final
pass, written as `val_nll_all` and `eval_seqs` in `outputs/registry.jsonl`; the first
such line is registry line 29, `run_20261031_014123_smoke`. Reruns of scratch and SFT
are queued as `*_v2`.

**v2, truncation fixed** (over-length sequences dropped: 238 of 1,863 training houses, 62
of 798 holdout; NLL over the first 256 of the remaining 736):

| run | holdout NLL (256 of 736) | BlockScore (n=64) | geometry | median blocks | median length |
|---|--:|--:|--:|--:|--:|
| scratch v2 | 0.543 | 138.68 | 138.7 | 1,156 | 4,914 |
| pretrain + SFT v2 | **0.426** | **95.50** | 95.5 | 1,454 | 6,990 |

Samples now end on their own (median length well under the 8,198 budget). On both the
likelihood (22% lower) and the scorecard (43 points lower), pretrain then SFT beats
scratch at matched SFT steps, and the sheets agree:
`outputs/run_20261014_115238_houses32_sft_5m_v2/samples.png` has walls, pitched roofs
and towers where `run_20261014_113512_houses32_scratch_5m_v2` has blobs.

Two cautions. The scorecard moved by 63 points for scratch between v1 and v2 for a
change that made the samples better by eye and shorter, which says that at n=64 and one
seed the geometry pillar is not a stable number (eval-suite memory: n of 256 or more).
The likelihood ordering is the same in v1 and v2 and is the result to trust at this
size. And the pretrained model saw 3.6 times the tokens; the matched-compute,
three-seed version is the M2 experiment, not this one.

## T59. M2 pilot at 5M, three seeds, n=256: a dropout fault, and the protocol that replaces it (2026-10-15)

Nine registry lines from `scripts/m2_seeds.sh` (unit `blockgen-m2`, 2026-10-14 12:22 to
18:50). Per seed: `m2_scratch` is 28,000 steps on `sft-houses32` (842M tokens, the
pretraining and SFT budgets added), `m2_pretrain` is 20,000 steps on the anneal shard
(591M tokens), `m2_sft` is 8,000 steps on `sft-houses32` from that checkpoint (241M
tokens, lr 1e-4). All 5.5M parameters, holdout NLL over the first 256 of the 736
`eval-holdout` builds that fit the context (`eval_seqs: 256`), scorecard at n=256 on `houses_32.s0.70-15-15.v1`.

| seed | scratch NLL, final (best, step) | pretrain NLL | pretrain + SFT NLL | BlockScore scratch | pretrain | pretrain + SFT |
|---|--:|--:|--:|--:|--:|--:|
| 0 | 0.483 (0.416, 17k) | 0.579 | 0.529 | 268.7 | 424.2 | 285.1 |
| 1 | 0.471 (0.400, 16k) | 0.530 | 0.461 | 264.9 | 352.9 | 287.3 |
| 2 | 0.452 (0.372, 15k) | 0.458 | **0.375** | 219.4 | 392.0 | 285.3 |
| mean | 0.469 (0.396) | 0.522 | 0.455 | 251.0 | 389.7 | 285.9 |

Read at face value the table says pretraining buys 3% in final NLL and loses on the
scorecard. It cannot be read at face value, for three reasons that each change the
protocol.

**The SFT runs for seeds 0 and 1 did not start from the pretrained model.** Their
first logged training loss (step 20) is 5.65 and 5.47 nats; uniform over the 507-token
vocabulary is 6.23. Seed 2 started at 1.71. The weights loaded correctly (zero missing
or unexpected keys). The cause is dropout: the SFT config set 0.1, the pretraining
config 0. Replayed from the saved checkpoints, the pretrained seed-0 model scores 0.55
to 0.71 nats on SFT batches in eval mode and 5.6 to 5.7 in train mode with dropout
on; seed 2 scores 0.35 to 0.46 and 1.6 to 2.0. A model trained without dropout is not
robust to it, and for two of three seeds the pretraining was gone before the first
update. Their SFT lines (0.529, 0.461) are what an 8,000-step scratch run at lr 1e-4
gets (0.543 in T58), which is what they were. The one valid seed shows what the
comparison is about: pretrain then SFT 0.375 against the scratch run's best 0.372,
with SFT still falling at its last eval (0.38, 0.38, 0.37) and the scratch curve past
its minimum. Fixes: `check_init_compat` in `train/trainer.py` refuses `init_from`
across a dropout change unless `allow_dropout_change=true`, and the SFT config keeps
the pretraining dropout of 0.

**Matched compute without early stopping penalises the arm that converges first.**
Every scratch run bottoms out between steps 15,000 and 17,000 and then climbs 0.07 to
0.08 nats by step 28,000: 28,000 steps on 1,625 eligible houses (5.4M tokens an epoch)
is 156 epochs, and the final train loss of 0.13 against holdout 0.47 is memorisation.
The final-step numbers compare an overfitted scratch model with an unconverged SFT
model. The trainer now tracks the best holdout NLL on the first validation tier,
writes `best.pt`, and with `select: best` (the default) samples, benchmarks and
registers that checkpoint; the registry records final and best, and the selected step.

**The scorecard's unit depends on n.** The same control arms score `real@solidify`
geometry 131 at n=64 (T58) and 444 at n=256, `real@monochrome` palette 46.5 and
125.0. A pillar is measured in real-sample spreads and the spread narrows with n, so
scores are comparable only at equal n. The leaderboard protocol (`houses32-v1` in
`blockgen/eval/bench/protocol.py`) ranks from `min_n` 128, and 256 or 399 is the
quotable convention; the n=64 numbers in
T57 and T58 are not on the same scale as these. At n=256 the ordering is scratch
(219 to 269) ahead of pretrain + SFT (285 to 287) ahead of pretrain alone (353 to
424), all on geometry. Novelty says why scratch leads: its nearest-training-neighbour
IoU is 0.41 to 0.42 against 0.375 for real held-out builds and 0.36 to 0.39 for the
SFT models, so 156 epochs put it closer to the training set than real data is, which
the realism pillar rewards until the duplicate gate fires (dup rate 0.000 on all
arms). Pretrain alone sampled with `win:whole` tags produces window-like fragments: 43
to 80 connected components against a real 2.7.

**Protocol M2b** (`scripts/m2b_seeds.sh`, unit `blockgen-m2b`, launched 2026-10-15
23:00): the three M2 pretrain checkpoints are reused unchanged (dropout 0 throughout).
SFT runs 16,000 steps at dropout 0; scratch runs the matched 36,000 steps at dropout 0;
both arms are early-stopped on holdout NLL and benchmarked at n=256 from the selected
checkpoint. The first M2b SFT line (seed 0) is at 0.490 by step 2,000 against 0.60 at
the same step in M2. Results in T60.

## T60. M2b at 5M, three seeds, early stopped: pretraining wins every seed on likelihood and on the board (2026-10-16, written 2026-10-28)

Six registry lines from `scripts/m2b_seeds.sh` (unit `blockgen-m2b`, 2026-10-15 22:52 to
2026-10-16 01:30), the protocol T59 set out. Per seed: `m2b_sft` is 16,000 steps on
`sft-houses32` (481M tokens, lr 1e-4) from the M2 pretrain checkpoint of the same seed
(20,000 steps on the anneal shard, 591M tokens, T59), dropout 0 throughout so the
checkpoint survives its first SFT batch; `m2b_scratch` is the matched 36,000 steps on
`sft-houses32` (1.08B tokens), also at dropout 0. Both arms select the best holdout
checkpoint (`select: best`), and that checkpoint is sampled and scored at n=256 on
`houses_32.s0.70-15-15.v1`. The holdout NLL is over the first 256 of the 736 `eval-holdout`
builds that fit the context (`eval_seqs: 256`). The cards were scored before the T64 descriptor change, so these six
arms are not on the current board until re-scored.

| seed | pretrain NLL | pretrain + SFT NLL (best step) | scratch NLL (best step) | BlockScore pretrain + SFT | scratch | geom_kid SFT / scratch | nn IoU SFT / scratch |
|---|--:|--:|--:|--:|--:|--:|--:|
| 0 | 0.579 | 0.479 (8,500) | 0.565 (8,000) | 374.8 | 502.1 | 19.8 / 26.6 | 0.33 / 0.28 |
| 1 | 0.530 | 0.420 (5,500) | 0.555 (8,000) | 323.2 | 444.9 | 17.1 / 23.6 | 0.35 / 0.31 |
| 2 | 0.458 | 0.350 (3,000) | 0.509 (10,000) | 324.1 | 333.7 | 17.1 / 17.6 | 0.31 / 0.36 |
| mean | 0.522 | **0.416** | 0.543 | **340.7** | 426.9 | 18.0 / 22.6 | 0.33 / 0.32 |

Runs: `outputs/run_20261015_225212_m2b_sft_s0`, `run_20261015_231328_m2b_scratch_s0`,
`run_20261015_235516_m2b_sft_s1`, `run_20261016_002201_m2b_scratch_s1`,
`run_20261016_010438_m2b_sft_s2`, `run_20261016_013050_m2b_scratch_s2`, each with its
`samples.png` and `bench/scorecard.md`.

**Within the protocol the result is clean.** Pretraining then SFT beats scratch on every
seed on both measures: 0.13 nats on holdout NLL (0.09, 0.14, 0.16) and 86 BlockScore
points (127, 122, 10), with geometry the worst pillar everywhere. SFT lowers the
pretrained model's NLL by 0.10 to 0.11 nats on each seed, and does it early: the
selected SFT checkpoints are at 3,000 to 8,500 steps (90M to 256M SFT tokens), so the
pretrained arm reached its best with a fifth to a half of the fine-tuning budget and a
total of 0.68B to 0.85B tokens against the scratch arm's 1.08B. The palette pillar tells
the same story (10 to 18 for SFT against 18 to 23 for scratch), and neither arm trips the
novelty gate (duplicate rate 0.000; nearest-training-neighbour IoU 0.31 to 0.36 for both,
against 0.375 for real held-out builds).

**The scratch arm overfits, and pretraining is the regulariser.** Every scratch run
bottoms out between steps 8,000 and 10,000 and then climbs without pause, to 1.34, 1.68
and 1.81 nats at step 36,000, with a train loss at the selected step of 0.35 to 0.45
against 0.51 to 0.57 held out. The SFT runs sit on a plateau: seed 2 is within 0.02 nats
of its minimum from step 2,000 to step 16,000, and its train loss at the selected step
(0.325) is 0.025 under its holdout (0.350). Seven epochs of world-crop pretraining do
what 156 epochs of houses cannot, which is to keep a 5M model from memorising 1,625
houses.

**What the comparison still owes.** The M2 scratch arm (T59) ran at dropout 0.1 and
reached a best holdout NLL of 0.416, 0.400 and 0.372 at steps 15,000 to 17,000; the M2b
scratch arm at dropout 0 reaches only 0.565, 0.555 and 0.509. Removing dropout from the
scratch arm, which the protocol did so that both arms were trained identically, cost it
0.14 nats, and against the regularised M2 scratch numbers the SFT arm wins one seed of three
(seed 1: 0.420 vs 0.400 is a loss by 0.02; seed 2: 0.350 vs 0.372 a win by 0.02; seed 0:
0.479 vs 0.416 a loss by 0.06). So the honest statement today is: at matched tokens and matched
regularisation, pretraining wins every seed; against the best scratch regularisation
tried so far, it is at best a tie on likelihood (one seed of three, mean 0.416 against
0.396). The comparison the paper needs is pretrain +
SFT against a scratch arm whose dropout and weight decay are chosen on holdout, both
early stopped, at 5M and 20M, three seeds each (research.md I.5). The M2 scratch
BlockScores (219 to 269) are not the counterpart on the board: they were sampled from
the final, overfitted checkpoint, whose nearest-neighbour IoU of 0.41 to 0.42 the
realism pillar rewards until the duplicate gate fires, which T59 already set out.

## T61. BrickNet Table 2 reproduction from the public 0.6B adapter, and OMR through BrickNet's pipeline (2026-10-20)

**Setup.** `libs/BrickNet`, package `bricknet==0.1.0` (paper collider), Qwen3-0.6B +
`kulits/BrickNet-0.6B-PT` merged, bf16, `scripts/generate.py` with
`--stop_after_newlines 199` (100 parts), EOS suppressed, temperature 1, seed 0. Nucleus
= top-k 20, top-p 0.95; ancestral = no truncation. Scored with `bricknet score`
(parse validity + autoregressive inset-mesh collision). Metric = mean number of
successful steps before the first invalidating step; Conn counts parse/connectivity
failures only, Coll counts the first of parse failure or collision. Intervals are 95%
normal. Paper used 2^16 samples. Driver: `libs/BrickNet/repro/run_table2.sh`,
metrics `repro/table2.py`.

| sampler | n | Conn ours | Conn paper | Coll ours | Coll paper | fully parsable | parsable and collision-free |
|---|---|---|---|---|---|---|---|
| nucleus | 512 | 94.5 ± 1.5 | 94.1 | 13.6 ± 1.1 | 16.0 | 85.7% | 0.6% |
| ancestral | 256 | 41.6 ± 3.9 | 45.0 | 10.7 ± 1.2 | 11.0 | 11.7% | 0.0% |

Three of four cells reproduce within the interval; nucleus Coll is 15% low (outside the
interval). First-collision histogram under nucleus sampling is broad and unimodal
around steps 5 to 12, so the shortfall is not a handful of early failures. Wall clock:
512 nucleus samples 105 min at batch 32 on the 16 GB card (about 4 s per 100-part
sample); batch 64 OOMs.

**OMR through BrickNet.** `repro/omr_to_bricknet.py --n 0` flattens all 1,819 OMR
`.mpd` files (MPD submodels inlined with composed transforms; `.dat` subfiles treated as
leaves) and parses them with `bricknet.parse_ldr`: 1,811 parse; 796,886 placements, of
which 638,906 (80.2%) are in the BrickNet vocabulary. The unknown tail is primitives
and LDCad flex parts (`4-4cyli`, `166`, `u9218`, `77`, `box4o8a`, `ls`-series). Graph
stats: parts per model median 137, mean 353; 2.30 edges per part; edge families stud
1,184,809, axle 244,801, fixed 20,824, hinge 19,978, ball 1,211; components per model
median 4; largest component holds 82% of parts on average (median 91%); 52.5% of
models have a largest component of 100+ parts. `repro/sample_omr.py` (one
collision-free walk per component, BrickNet's own sampler, error-tolerant per model):
7,584 paths from 1,805 models; 5,629 components of 4+ parts, of which 4,507 are in the
4 to 100 range and 1,122 exceed 100; the walker places 84.5% of a component on average
and completes 60.2%; 3,391 complete components in the 4 to 100 range (72,363 parts).
For scale, BrickNet-SFT has 67,185 models (1.77M parts) and BrickNet-PT 253,623 graphs
(38.8M parts, 125.8M edges). Paths: `libs/BrickNet/repro/omr_all_paths.jsonl`.

**Perplexity proxy (Table 4 protocol, our data).** PT-0.6B per-token perplexity on
300 OMR walks (4 to 100 parts, path + EOS, bf16): **2.353** (280,575 tokens); paper
val 1.331. **Caption-conditioned smoke (8 captions × 2, nucleus, seed 0).** Authors'
inference code: 8/16 fully parsable, 5/16 unparsable at step 1 (sequence starts at
`b`), mean steps before failure 11.4. With local `--force_root`: 11/16 fully parsable,
4/16 collision-free, mean steps before failure 8.8. Renders:
`outputs/figures/bricknet_repro/`.

## T62. BrickAgent environment check and one BrickBench task end to end (2026-10-20)

**Environment.** `libs/BrickAgent/.venv` (brickagent 0.1.0, bricknet 0.1.1, convex
colliders in `libs/BrickAgent/.bricknet_data`). README wall (plate 2x8 + 4 courses of
2xN bricks): `check` passes in 0.1 s. Two stacked 2x4 bricks with the upper one placed
5 plates too high: rejected, "2 connected components ... unstable", the moving
component named.

**BrickBench Model task 004** ("A short, black-haired lord in a red hat and red tunic
sits on an enormous throne. His boots hang far above the floor. A long red carpet
leads up three steps to the seat."), built by Claude in this session through the
`brickagent` API (`libs/BrickAgent/work/model-004/build.py`): plate 8x16 floor, two
rows of red 1x2 tiles, three plate-stepped stairs with carpet tiles, a three-plate
dais, a seat of two 2x4 bricks, a nine-brick 1x6 back, 1x2 armrests, and a seated
minifig (legs by their back hole on the seat studs, hips, red torso, arms down,
head, red fez). `check`: 53 parts, 0 collision pairs, 1 connected component, stable.
`scripts/verify.py --split model`: **valid**, 53 parts, 16 distinct, 52 authored
attachments, 210 native connection edges, 1.66 s. Four check iterations were needed;
all four failures were API-convention mistakes (tile footprint, rotated-part extent,
facing yaw relative to a rotated stud's roll, arm swing) that `check`'s messages
located exactly. VQA and ELO not scored (Gemma 4 31B judge, about 62 GB). Render:
`outputs/figures/bricknet_repro/brickbench_model-004_claude.png`.

**Table 4 on the released val split (data received 2026-10-20, `libs/BrickNet/repro/table4.py`).**
Protocol as `scripts/train.py`: pt = 1,000 first-round val walks picked with rng 0, path + EOS,
loss on every token; sft = the 444 (caption, first-round walk) pairs, caption + "\n" + path +
EOS, loss on path + EOS. bf16, Qwen3-0.6B + released adapters.

| model | ours | paper |
|---|---|---|
| PT 0.6B, unconditional | **1.318** (910,771 tokens) | 1.331 |
| PT + SFT 0.6B, caption-conditioned | **1.297** (296,676 tokens) | 1.298 |

Both cells reproduce to the third digit, so the released adapters are the paper's and the OMR
perplexity of 2.353 above is a real distribution gap, not a protocol artefact.

**P_inv, first attempt (invalid protocol).** 512 val captions, 1 sample each, but with
`--stop_after_newlines 199` carried over from the unconditional recipe, which suppresses EOS
and forces 100-part sequences (val walks average 38 parts): 53.3% parsable, mean steps before
failure 10.8, invalid placements over all placements 0.603. Not comparable to the paper's
0.256; rerun with natural EOS below.

**P_inv, natural stopping (`repro/run_pinv.sh`, 512 val captions × 1, nucleus, seed 0,
max 2,816 new tokens, batch 16).** 462/512 fully parsable (90.2%), 5 unparsable at step 1
(1.0%; every sample starts at root `a`), mean 31.6 parts (median 19; val walks average 38),
135/512 parsable and collision-free (26.4%), mean steps before first failure 9.7. The paper
does not define P_inv precisely, so three readings: invalid placements over all placements
(parse-failed tail counted) **0.318**; colliding placements over placements among fully
parsable samples **0.279**; per-sample mean collision fraction among parsable samples
**0.197**. Paper 0.6B: 0.256, which the second and third readings bracket. The 30% root-skip
seen with our eight hand-written captions (above) does not occur on val-style captions.

## T63. Traversability: can a player walk into it? Real 95%, agentic 50 to 68%, LLM arms 2% (2026-10-23)

> ### Bottom line
> A two-block player set down outside the build and flood-filled through
> everything it can walk reaches the interior of **95%** of real houses and
> **88%** of their standing room. The LLM piece-serialization arms admit a player
> **2%** of the time, the native AR arms **58 to 59%**, the agentic recipes
> **50 to 68%**, with `classic-matched` ahead of both dream recipes. The
> leaderboard's geometry pillar puts `llm_control` above every agentic arm; the
> measure a player cares about disagrees 25 to 1.

**Run:** `outputs/run_20261023_004317_bench_traversal` (`scripts/bench_traversal.py
--n 128 --gif 2`, `houses_32.s0.70-15-15.v1`, seed 0). Method in notes.md §52 and
docs/benchmark.md ("Traversability: can a player walk in?"). Code
`blockgen/eval/traversal.py`, 22 tests, nothing in `blockgen/eval/bench` changed.

`enterable` is the share of builds a player can get into at all; `walkable` is the
mean share of each build's interior standing cells a player can reach, with no
interior counting as zero; `floors` are storeys with a floor patch and `reached`
those with a reached cell; `headroom3` is the share of interior standing cells with
a third free block above.

| arm | n | enterable | walkable | median | no interior | floors reached / floors | headroom3 | entrances | blocks |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `real_test` | 128 | **0.95** | **0.88** | 1.00 | 0.02 | 1.67 / 1.84 | 0.94 | 3.30 | 596 |
| `real@seal_openings` | 128 | 0.02 | 0.01 | 0.00 | 0.02 | 0.03 / 1.98 | 0.94 | 0.05 | 600 |
| `real@lower_ceilings` | 128 | 0.82 | 0.73 | 1.00 | 0.05 | 1.22 / 1.47 | **0.56** | 2.80 | 664 |
| `real@solidify` | 128 | 0.19 | 0.17 | 0.00 | 0.79 | 0.29 / 0.32 | 1.00 | 1.10 | 818 |
| `real@hollow` | 128 | 0.97 | 0.86 | 1.00 | 0.00 | 1.20 / 1.45 | 0.98 | 2.70 | 502 |
| `canon_oriented` | 128 | 0.59 | 0.51 | 0.66 | 0.24 | 0.70 / 0.87 | 1.00 | 2.18 | 1101 |
| `native_oriented` | 128 | 0.58 | 0.52 | 0.66 | 0.34 | 0.60 / 0.76 | 1.00 | 1.78 | 888 |
| `llm_control` | 128 | **0.02** | 0.02 | 0.00 | 0.97 | 0.02 / 0.02 | 0.88 | 0.02 | 205 |
| `llm_bias` | 128 | 0.05 | 0.05 | 0.00 | 0.95 | 0.05 / 0.05 | 1.00 | 0.07 | 190 |
| `dream-caption` | 133 | 0.51 | 0.44 | 0.38 | 0.15 | 0.53 / 1.16 | 1.00 | 0.74 | 628 |
| `dream` | 133 | 0.50 | 0.45 | 0.00 | 0.14 | 0.51 / 1.06 | 1.00 | 0.67 | 828 |
| `classic-matched` | 133 | **0.68** | **0.62** | 1.00 | 0.14 | 0.77 / 1.17 | 1.00 | 2.07 | 1093 |

Real floor intervals (80% subsamples): enterable 0.95 [0.93, 0.97], walkable 0.88
[0.86, 0.90].

**The probes behave.** `seal_openings` (every doorway and window bricked, every
door, trapdoor and gate turned to wall; interior volume unchanged) takes real
houses from 0.95 to 0.02 enterable with the block count moving by 4 blocks at the
median. `lower_ceilings` (top layer of every room filled; exterior untouched)
leaves 82% enterable but drops `headroom3` from 0.94 to 0.56 and `walkable` to
0.73: rooms that were two high are gone to a player. `hollow` leaves the walk
alone (0.97 / 0.86), as it must, because it deletes only blocks no camera and no
player can reach. `solidify` removes nearly everything to walk (79% of builds
have no standing room left); the 19% still enterable keep a doorway-reachable
room the fill does not touch.

**What it says about the arms.** The two LLM arms build sticks and slabs with no
standing room in 95 to 97% of samples; nothing a player can enter. The native
arms have rooms in two thirds of samples and a player gets into most of those,
but with 0.7 to 0.9 floors per build against a real 1.8. The agentic recipes have
standing room in 85% of samples and a player enters half to two thirds of them;
`classic-matched` (0.68 / 0.62) is ahead of `dream` (0.50 / 0.45) and
`dream-caption` (0.51 / 0.44), the reverse of their BlockScore order, and all
three leave upstairs rooms unreached (0.5 to 0.8 floors reached of 1.1 to 1.2).
Every generated arm has `headroom3` at 1.00: the player-scale prompt (§38) made
ceilings tall, and the failure is doors and stairs, not height.

**Against the leaderboard.** BlockScore ranks `llm_control` (145, worst pillar
appearance) above `dream-caption` (165), `dream` (191) and `classic-matched`
(194), all of whose worst pillar is geometry. On `geom_kid` the LLM fragments
score 15.0 against 17.7 to 19.9 for the agentic houses; on `mv_dino_kid` the
order is already the other way (0.43 against 0.22 to 0.40). So the ordering that
reads wrong comes from the geometric descriptor, not from the render tier, and
the player's measure sides with the render tier here. The descriptor is a
distribution distance over pattern, thickness and height histograms; thin
fragments of real materials sit closer to real houses in that space than
program-written cuboid rooms do. Traversability is not in the descriptor and not
in BlockScore; wiring it in is a change to `blockgen/eval/bench` and is left as a
decision.

**Caveats.** Seeding from any exterior standing cell tests the building and not
the site (a balcony door counts as a way in); the interior recipe's 3x3x3 closing
drops two-high rooms that have an opening, so a doll's house with a door reads as
"no interior" rather than "unreachable interior"; `entrances` counts clusters of
entry cells and is inflated by open ground floors. The 5% of real houses not
entered are open structures with next to no standing room and one tower whose
ladder stops four blocks short of its room.

Figures: `samples.png` (uncut house beside its finished walk, quadrant cut away),
`walkthrough_115.gif` (Japanese House, five floors reached, 24 frames) and
`walkthrough_043.gif` (Medieval Rustic Mansion, three floors), each with a
four-frame `_strip.png`.

## T64. Leaderboard re-scored with room scalars in the descriptor and a traversal pillar (2026-10-23)

> ### Bottom line
> With the player's room (standing share, storeys, headroom) in the geometric
> descriptor, `geom_kid` moves the right way for every arm: up for the LLM
> fragments, down for the agentic houses. `dream-caption` (155.8) now ranks
> above `llm_control` (167.9), where it was below. The new `traversal` pillar
> orders the arms as the player does (`llm_control` 2.6 spreads from real,
> `dream` 1.3, `classic-matched` 0.8) but is never the worst pillar: BlockScore
> is still decided by `geom_kid` for every learned arm.

**Run:** `outputs/run_20261023_011035_bench_traversal_pillar_leaderboard_re-score`
(`python -m blockgen.eval.bench --tier both --n 128 --baselines`, the five
leaderboard arms of 2026-09-25 plus the three T51 agentic arms; same split, same
samples; self-validation 6/6). Method in notes.md §53; the descriptor, pillar and
protocol changes are in docs/benchmark.md. This is the only card the board ranks
now: `houses32-v1` pins the descriptor, and every earlier card is unranked with
the reason "re-score with the current bench".

| # | arm | BlockScore ↓ | worst | appearance | geometry | palette | coherence | geom. scalars | traversal |
|---|---|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | `real_test` | 0.00 | — | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| unranked (card rank 2; contaminated, T31) | `native_oriented` | 114.05 | geometry | 69.76 | 114.05 | 2.75 | 0.44 | 0.56 | 1.04 |
| 3 | **`canon_oriented`** | **141.82** | geometry | 85.31 | 141.82 | 2.51 | 0.48 | 0.54 | 1.02 |
| 4 | `dream-caption` | 155.83 | geometry | 75.84 | 155.83 | 31.58 | 0.41 | 0.45 | 1.24 |
| 5 | `llm_control` | 167.92 | geometry | 149.19 | 167.92 | 17.02 | 2.48 | 1.74 | **2.59** |
| 6 | `dream` | 177.23 | geometry | 119.12 | 177.23 | 31.64 | 0.33 | 0.50 | 1.26 |
| 7 | `classic-matched` | 181.86 | geometry | 140.19 | 181.86 | 30.52 | 0.23 | 0.41 | 0.78 |
| 8 | `llm_bias` | 191.88 | appearance | 191.88 | 172.51 | 16.48 | 2.74 | 1.84 | 2.49 |
| 9 | `real@solidify` | 214.57 | geometry | 0.14 | 214.57 | 2.51 | 0.31 | 2.49 | 2.05 |
| 10 | `real_shuffled_materials` | 231.53 | appearance | 231.53 | 0.00 | 0.00 | 0.00 | 0.00 | 0.41 |
| 11 | `patchwork` | 246.15 | appearance | 246.15 | 186.18 | −2.97 | 1.28 | 0.81 | 1.42 |
| 12 | `real@monochrome` | 297.94 | appearance | 297.94 | 0.00 | 79.63 | 0.00 | 0.00 | 2.24 |
| 13 | `train_copy_noise` | 310.52 | geometry | 153.54 | 310.52 | 1.11 | 1.22 | 0.74 | 0.35 |
| 14 | `gabled_house` | 312.75 | geometry | 287.97 | 312.75 | 25.61 | 0.33 | 1.24 | 0.45 |
| 15 | `shell_box` | 476.22 | appearance | 476.22 | 323.73 | 24.91 | 1.20 | 1.71 | 2.65 |
| 16 | `uniform_random` | 603.10 | appearance | 603.10 | 357.93 | 23.70 | 15.71 | 2.37 | 2.34 |
| — | `canon_unoriented` | DQ | memorization (+3.1 spreads) | | | | | | |
| — | `train_verbatim`, `real@single_mode` | DQ | memorization / mode collapse | | | | | | |

**`geom_kid`, before and after the room scalars** (same builds, same reference):

| arm | 77-dim (T32 / T51) | 80-dim (this run) |
|---|--:|--:|
| `real_test` | −0.095 | −0.090 |
| `canon_oriented` | 14.898 | 14.291 |
| `llm_control` | 15.006 | **16.937** |
| `llm_bias` | 15.869 | **17.402** |
| `dream-caption` | 17.650 | **15.712** |
| `dream` | 19.620 | **17.882** |
| `classic-matched` | 19.937 | **18.351** |

Three scalars out of eighty moved the two families by about two units in
opposite directions, which is what T63 predicted: the LLM fragments have no
standing room, no storeys and no headroom, and the agentic houses have real
rooms. `canon_oriented`'s 136.73 of T32 is now 141.82 for the same reason in the
other direction: two thirds of its samples have standing room, a third do not.

**The traversal pillar, per arm** (generated mean / real mean, W1 in real spreads):

| arm | enterable | walkable | floors reached | headroom3 | pillar (RMS) |
|---|--:|--:|--:|--:|--:|
| `real_test` | 0.94 / 0.94 | 0.84 / 0.87 | 1.62 / 1.57 | 0.82 / 0.84 | 0.00 |
| `canon_oriented` | 0.59 (1.6) | 0.51 (1.3) | 0.70 (0.8) | 0.85 (0.3) | 1.02 |
| `llm_control` | 0.02 (4.1) | 0.02 (3.2) | 0.02 (1.4) | 0.77 (0.5) | 2.59 |
| `dream-caption` | 0.51 (1.9) | 0.44 (1.6) | 0.53 (0.9) | 0.85 (0.3) | 1.24 |
| `dream` | 0.50 (2.0) | 0.45 (1.6) | 0.51 (1.0) | 0.92 (0.4) | 1.26 |
| `classic-matched` | 0.68 (1.2) | 0.62 (1.0) | 0.77 (0.7) | 0.97 (0.5) | 0.78 |
| `real@solidify` | 0.21 (3.2) | 0.18 (2.6) | 0.31 (1.1) | 0.96 (0.5) | 2.05 |
| `real_shuffled_materials` | 0.80 (0.6) | 0.68 (0.7) | 1.41 (0.2) | 0.80 (0.2) | 0.41 |
| `shell_box` | 0.00 (4.1) | 0.00 (3.2) | 0.00 (1.4) | 1.00 (0.6) | 2.65 |
| `gabled_house` | 1.00 (0.2) | 1.00 (0.5) | 1.00 (0.7) | 1.00 (0.6) | 0.45 |

The pillar does what T63 measured, in the bench's units, and the units are the
point to read carefully. KID pillars are quoted in noise-floor spreads of a
set-level statistic and run to hundreds; the scalar blocks are the RMS of
per-build W1 distances in real-sample spreads and run to a few. Under the max,
`traversal` can only decide BlockScore when `geom_kid` and `mv_dino_kid` are
within a few spreads of real, which no learned arm is. It is a veto against a
future generator that matches the distributions and still builds houses nobody
can enter (`shell_box` and `real@solidify` show the shape of that case); today
it is a column that ranks the arms the way a player would, beside a column that
ranks them the way a kernel does.

**Protocol consequence.** `houses32-v1` now pins `descriptor="patch2_orb55_t5_h8_s12"`.
T31/T32's 106.87 and 136.73, T51's agentic BlockScores and every number on the
board before today were measured with the 77-dim descriptor and are not
comparable with this table; the board unranks those cards with that reason. The
`agentic_oneshot` (n=12) and `pick_n_place` (n=16) provisional rows are off the
board until re-scored, and were below `min_n` in any case.

## T65. The facade oracle: real houses reduced to their visible shell (2026-10-23)

Run `run_20261023_204241_bench_facade_oracle_real_houses_reduced_to`
(`--tier both --n 399`, self-validation 6/6). Three control rungs made by
`scripts/facade_oracle.py` from `blockgen.corpus.facade` (notes.md §54): every
test-split house projected to face images and lifted back, keeping only what a
camera at infinity along each axis sees. `facades4` is the four sides (what a
facade-only generator hands back), `facades5` adds the roof plan, `facades6`
adds the underside. Median share of blocks kept: 56%, 65%, 79%. Nothing inside
survives in any of them. All arms n=399 (a first run at `--n 128` left the
controls smaller than the oracle arms and is superseded).

| arm | BlockScore | worst pillar | geom_kid | mv_dino_kid | components | thickness mean | sealed interior | walk standing (descr.) | enterable | standing reached |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| real_test | 0.00 | appearance | -0.07 | 0.000 | 2.7 | 1.04 | 0.114 | 1.80 | 0.95 | 0.87 |
| real@facades6 | 492.7 | geometry | 13.76 | 0.001 | 13.5 | 1.00 | 0.069 | 1.79 | 0.73 | 0.69 |
| real@facades5 | 700.5 | geometry | 19.60 | 0.005 | 18.6 | 1.00 | 0.000 | 1.85 | 0.57 | 0.52 |
| real@solidify | 846.3 | geometry | 23.70 | 0.000 | 2.3 | 1.56 | 0.000 | 1.82 | 0.18 | 0.16 |
| real@facades4 | 858.0 | geometry | 24.02 | 0.076 | 23.5 | 1.00 | 0.000 | 1.90 | 0.15 | 0.14 |

(`components` is coherence `n_components`, real 2.7; `sealed interior` is
`interior_ratio_sealed`; `enterable` and `standing reached` are the traversal
block; BlockScore pillars in spreads: facades6 appearance +0.96, coherence
+1.96, traversal +0.69; facades4 appearance +78.9, coherence +4.34, traversal
+2.27.)

**The render tier cannot see the shell.** MV-DINO-KID is 0.001 for the six-face
shell and 0.005 with the roof plan, indistinguishable from real at this n, and
the appearance pillar sits at one spread. The four-side shell alone is caught
(0.076, 79 spreads) only because its roofs fall apart: no side ray reaches the
upper surface of a flat roof, so the roof survives as a ring of eaves and the
build splits into 23.5 pieces on average. This is `hollow` read from the
generator's side: a facade model that gets the faces right has beaten the
appearance pillar.

**Geometry catches it in every version, and it is the worst pillar every time.**
Thickness collapses to exactly one block everywhere (real mean 1.04, p90 1.16),
the sealed interior share goes to zero without the floor and to 0.069 with it,
and `geom_kid` ranks the rungs by how much of the house is missing: 13.8, 19.6,
24.0 against solidify's 23.7. The facade descriptor room scalars (T64) are near
real for all three (`walk_standing` 1.79 to 1.90 against 1.80), which is
correct: a shell has the same standing room as the house it came from.

**Traversal reads the floor.** Without the underside every room opens downward
and the ground layer the player model stands on is the only floor, so the
four-side shell is 15% enterable and solidify-like (14% standing reached). The
roof plan recovers it to 57%, the underside to 73% against real's 95%; the rest
are rooms whose floor is not the bottom face of the build (upper storeys,
porches) and interior floors the faces never see.

**What this fixes for the facade track.** A 2.5D arm's ceiling is `facades6`
at BlockScore 493, decided by geometry, and that ceiling is reached by faces
alone; the roof plan is worth 200 BlockScore points and the underside another
200. Interiors remain out of reach by construction, which is the gap the second
stage (a plan, or an interior fill) has to close. Facade arms should be reported
against these rungs, not against real.

## T66. The facade track: baselines, two facade models and the ceiling (2026-10-26)

Run `run_20261026_160934_bench_facade_track_baselines_the_per-face_and`
(`--tier both --n 399`, self-validation 6/6, every arm n=399). The three
facade baselines of notes.md §55 (fitted on the train split, lifted into real
train boxes), the per-face facade model (§55, faces sampled independently), the
joint facade model (§56, every face of a build in one sequence, each face
conditioned on the ones before it) and `real@facades6`, the ceiling of a facade
generator (T65). Both models are the 5M preset, 8,000 steps, about seven minutes
each; holdout NLL 0.67 per face and 0.78 joint, both still falling.

| arm | BlockScore | worst pillar | geom_kid | mv_dino_kid | components | enterable | nn percentile |
|---|--:|---|--:|--:|--:|--:|--:|
| real_test | 0 | appearance | -0.07 | 0.000 | 2.7 | 0.95 | |
| real@facades6 (ceiling) | 493 | geometry | 13.8 | 0.001 | 13.5 | 0.73 | 0.51 |
| facade_patchwork | 990 | geometry | 27.7 | 0.538 | 26.5 | 0.23 | 0.48 |
| houses32_facades_joint_5m | 1128 | geometry | 31.6 | 0.633 | 71.5 | 0.25 | 0.65 |
| houses32_facades_5m | 1205 | geometry | 33.8 | 0.702 | 74.7 | 0.24 | 0.50 |
| facade_markov | 1335 | geometry | 37.4 | 1.141 | 144.7 | 0.08 | 0.75 |
| facade_unigram | 2208 | appearance | 39.8 | 2.119 | 332.6 | 0.67 | 0.87 |

(`components` is coherence `n_components`; `nn percentile` is the memorization
gate `dino_nn_percentile`, 0.5 is held-out real; no arm is disqualified.)

**The ordering is the predicted one.** Unigram below markov below the learned
models below patchwork below the ceiling, on BlockScore and on both KIDs, with
every gap outside the intervals. Conditioning the faces on each other moves the
model from 1205 to 1128 and improves `geom_kid` (33.8 to 31.6) and
`mv_dino_kid` (0.70 to 0.63) at the same training budget, and the face sheets
show why: the per-face model gives every wall of a build a different material,
the joint model keeps one palette and style across all six faces. The joint
model's palette pillar is worse (17.9 against 12.6 spreads), which is the same
effect seen from the other side: a build in one material has a narrower palette
than a real house.

**What the learned models still lose to.** `facade_patchwork`, six real faces
from six different houses, beats both models on every pillar. Its faces are real
and the models' are not yet: a 5M model trained for seven minutes draws walls
with the right bands and materials but not the right windows, and the lifted
shells fall into 70 pieces where the ceiling falls into 13, because faces that
disagree at a shared edge do not meet. Enterability is the same for every facade
generator, about 23%, against 73% for the ceiling: the faces do not yet put a
door where the floor is.

**What the track has established.** A facade generator is now an arm on the
same board as everything else, with a floor, a ladder of baselines and a ceiling
measured under the same protocol; the bench reproduces the intended order of all
of them; and the first learned result is that the layout, not the model, decided
the first improvement. The next levers, in order: train longer (both NLLs still
falling), a bigger preset, and the ceiling's own gap, interiors, which no facade
model can close and the second stage has to.

## T67. The volumetric row: MMD, coverage and 1-NNA under chamfer and IoU on the ladder (2026-10-28)

`PYTHONPATH=. python -m blockgen.eval.bench.pointcloud --n 64` (notes §58). Same
reference (248 usable val builds), probe draw (64 usable test builds, seed 0), rungs and
noise floor as `geom__patch2_orb55_t5_h8_s12.json`, the geometry ladder of record at the
same n. Separation from `real_heldout` in units of each metric's own noise-floor sd, the
unit every gate is stated in; the geometry tier's two metrics first for comparison.

| metric | canon16 | canon8 | chunk_del_40 | solidify | hollow | occ_noise_5 | occ_noise_10 | jitter | mat_shuffle | rot90 | validity |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| geom_kid | 46.6 | 65.5 | 31.0 | **70.9** | 42.2 | 57.4 | 92.4 | 109.2 | 0.0 | 0.0 | PASS |
| geom_mmd_rbf | 58.5 | 84.8 | 39.0 | **96.0** | 52.5 | 71.2 | 121.3 | 148.5 | 0.0 | 0.0 | PASS |
| cd_volume MMD | 32.0 | 123.5 | 7.2 | -2.7 | 2.3 | -0.7 | -0.9 | -0.9 | 0.0 | 0.9 | FAIL G2 |
| cd_volume 1-COV | 10.8 | 13.7 | 0.0 | 0.0 | -0.6 | 0.0 | -1.0 | 0.3 | -0.6 | -1.0 | FAIL G2 |
| cd_volume 1-NNA | 10.7 | 11.3 | 5.4 | 1.2 | 1.0 | -0.5 | 0.5 | -0.2 | 0.0 | 1.2 | FAIL G2 |
| cd_surface MMD | 32.3 | 125.0 | 7.1 | 1.8 | 2.0 | -1.0 | -1.2 | -1.0 | -0.1 | 0.6 | FAIL G2 |
| cd_surface 1-COV | 11.4 | 15.1 | 0.0 | -0.7 | -0.3 | 0.3 | -0.3 | 0.3 | 0.3 | -1.3 | FAIL G2 |
| cd_surface 1-NNA | 10.6 | 11.7 | 5.4 | 0.7 | 1.3 | 0.4 | 0.5 | 0.4 | 0.4 | 1.1 | FAIL G2 |
| iou MMD | 23.3 | 46.5 | 6.8 | **-8.6** | 6.0 | 1.4 | 3.0 | 7.4 | 0.0 | 1.0 | FAIL G8 |
| iou 1-COV | 13.1 | 16.2 | 0.8 | 2.0 | 0.4 | 0.4 | 0.4 | -0.4 | 0.0 | 1.6 | FAIL G2 |
| iou 1-NNA | 8.1 | 8.5 | 2.1 | 0.4 | 1.3 | -0.3 | -0.6 | 0.8 | 0.0 | 1.0 | FAIL G2 |

Raw values for the headline pair: `cd_volume` MMD reads 0.00602 on real held-out builds,
0.00552 on Fill-Interior and 0.00644 on Hollow, noise sd 0.00018; `cd_surface` MMD 0.00612,
0.00645, 0.00649, sd 0.00018. 1-NNA sits at 0.48 to 0.53 for real, Fill-Interior and
Hollow alike (0.5 is indistinguishable), and reaches 0.97 to 1.00 only on the
decimation rungs.

**The standard 3D suite is as blind to the interior as the render tier.** Every
point-cloud statistic places Fill-Interior within three noise floors of real, and the
two MMDs place it on the wrong side: filling a house moves it *closer* to the reference
under volumetric chamfer (-2.7 sd) and under IoU (-8.6 sd). The mechanism is the
metric's, not a bug. Chamfer averages squared nearest-neighbour distances; the points
Fill-Interior adds sit one or two blocks from a wall, nearer than the typical mismatch
between two different houses (about 2.5 blocks RMS at this scale), so the mean falls.
A filled grid likewise overlaps every reference grid at least as much as its hollow
original does. The cavity-wall sampler that Appendix K says sees the probe per pair
reads +1.8 sd at the set level. Hollow, occupancy noise at 5 and 10 percent, and column
jitter, which the geometry tier separates at 42 to 148 sd, are within 2.3 sd for every
point-cloud statistic. What the suite does resolve is what point clouds are built for:
decimation (8 to 125 sd) and a 40 percent deletion (5 to 7 sd for MMD and 1-NNA).

**None of the nine would be admitted to rank anything.** Eight fail G2, the occupancy
noise ordering (1 percent, 5 percent, 10 percent of blocks moved must read as
increasing damage; chamfer and IoU read them as flat), and `iou` MMD fails G8,
n-stability. The gates were written for render metrics and the geometry tier and were
not tuned here; the point-cloud suite was run through them unchanged.

**Caveats, for the paper's table.** n=64 (n=128 is running); 1,024 points per cloud,
PointFlow's squared-distance chamfer, no per-shape normalisation; 1-NNA against an
equal-size reference draw fixed once per run. Not yet run: EMD (an assignment per pair
across 2.9M pairs), a KID on a pretrained 3D encoder (Uni3D or PointBERT), and the
leaderboard arms themselves under these statistics. The row answers the objection as
stated; the learned-encoder column is the remaining one a reviewer can ask for.

**At n=128 the picture holds** (`--n 128 --out-root outputs/analysis/ladder_n128`, 2,925
structures). The decimation separations roughly double, as a test statistic should
(cd_volume MMD canon8 123 to 196 sd), while Fill-Interior stays where it was: -3.9 sd
under volumetric chamfer MMD, +2.7 under cavity-wall chamfer MMD, -11.4 under IoU MMD,
0.8 to 2.3 for every 1-NNA and coverage statistic. Hollow reads 0.0 to 3.5 across the
nine, occupancy noise at 10 percent -1.1 to 5.0. The IoU statistics now also fail G6:
corner-aligned grids are not rotation invariant and a 90 degree turn moves them 2.6 to
2.7 sd. Doubling the sample buys the standard suite resolution on what it already saw
and nothing on the interior.

## T68. The facade track at scale: 20M, D4 augmentation and the complete card (2026-10-28)

Run `run_20261028_143416_bench_facade_track_baselines_five_facade` (`--tier both
--n 399`, self-validation 6/6, every arm n=399, 80 minutes). The T66 arms plus three
new joint facade models: the 20M preset on the plain shard (notes.md §57, 24,000
steps, best checkpoint at step 6,000), the 5M preset on the D4-augmented shard (§59,
16,000 steps) and the 20M preset on the D4 shard (§59, 24,000 steps). All four
joint models use `select=best` on the unaugmented holdout. A four-model card without
the 20M D4 arm, `run_20261028_131522_bench_facade_track_baselines_four_facade`, ran
first and is superseded; every shared number agrees to the quoted digits.

| arm | NLL | BlockScore | appearance | geom_kid | mv_dino_kid | components | enterable | nn pct |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| real_test | | 0 | 0 | -0.07 | 0.000 | 2.7 | 0.95 | 0.50 |
| real@facades6 (ceiling) | | 493 | 1 | 13.8 | 0.001 | 13.5 | 0.73 | 0.51 |
| facade_patchwork | | 990 | 561 | 27.7 | 0.538 | 26.5 | 0.23 | 0.48 |
| joint 20M plain | 0.806 | 1035 | 566 | 29.0 | 0.543 | 63.1 | 0.23 | 0.63 |
| joint 20M D4 | 0.479 | 1082 | 487 | 30.3 | 0.468 | 60.0 | 0.35 | 0.48 |
| joint 5M D4 | 0.589 | 1113 | 591 | 31.2 | 0.567 | 70.0 | 0.32 | 0.61 |
| joint 5M plain | 0.783 | 1128 | 660 | 31.6 | 0.633 | 71.5 | 0.25 | 0.65 |
| per-face 5M | 0.67/face | 1205 | 731 | 33.8 | 0.702 | 74.7 | 0.24 | 0.50 |
| facade_markov | | 1335 | 1189 | 37.4 | 1.141 | 144.7 | 0.08 | 0.75 |
| facade_unigram | | 2208 | 2208 | 39.8 | 2.119 | 332.6 | 0.67 | 0.87 |

(`NLL` is holdout nats per token on the unaugmented holdout at the selected
checkpoint; `appearance` is that pillar in spreads; every BlockScore except
unigram's is the geometry pillar; `components` is coherence `n_components`;
`nn pct` is the memorization gate, 0.5 is held-out real; no arm is disqualified.)

**Likelihood and the card do not move together.** The likelihood ladder is
monotone in data and size: 0.806 (20M plain), 0.783 (5M plain), 0.589 (5M D4),
0.479 (20M D4), a 0.33 nat gain from eight symmetries of the same 2,114 houses.
The card does not follow it. The plain 20M model, the worst of the four on
likelihood and a memoriser by step 6,000 (§57), has the best BlockScore of any
learned arm, 1035, because BlockScore is the geometry pillar and its `geom_kid`
(29.0) is the lowest; the Holm groups put it with patchwork (`ef` against `e`),
the only learned model the baseline does not separate on geometry. The 20M D4
model, the best of the four on likelihood by 0.33 nats, sits second at 1082: its
`geom_kid` is 30.3, which the Holm groups do not separate from the plain 20M (`fg`
against `ef`) but which the leaderboard, a point estimate, ranks behind it. On
every other reading it is the best learned arm on the board. Its `mv_dino_kid` of
0.468 is the only learned value below patchwork (group `c`, alone; patchwork and
the plain 20M share `d`; paired delta 0.075 against the plain 20M, Holm p =
0.0004), its appearance pillar is 487 spreads against patchwork's 561, its shells fall into
60 pieces against 63 to 75 for the others, it is the only arm whose memorization
percentile sits at held-out real (0.48, where the plain models sit at 0.61 to
0.65, closer to train), and it is the most enterable learned arm at 35 percent.

**What the D4 augmentation bought, and what it did not.** On the likelihood it is
the lever §57 asked for: the 20M model goes from memorising at step 6,000 to still
fitting at 18,000. On the card it buys appearance (`mv_dino_kid` 0.633 to 0.567 at
5M, 0.543 to 0.468 at 20M), traversal (enterable 0.25 to 0.32 at 5M, 0.23 to 0.35
at 20M; the plain models never moved it off the baselines' 0.23) and novelty, and
it buys nothing on `geom_kid`: 31.6 to 31.2 at 5M is inside the group, and at 20M it
costs 1.3. The augmented models also drift on the palette pillar (18 to 25 spreads
against 13 for the per-face model and 6 for patchwork), with `palette_jsd_exact`
0.185 at 20M D4 against 0.123 for the per-face model: a model that has seen every
house from eight sides commits harder to one material per build, and real houses
do not. The one geometry statistic it moves is the one the ceiling itself is held
by, `n_components`, from 71.5 to 60.0; the ceiling is 13.5. Faces that disagree at
a shared edge still do not meet, and that, not the walls themselves, is what
`geom_kid` is reading.

**Patchwork still leads every learned model** on BlockScore (990 against 1035),
by geometry alone: six real faces from six houses have real windows, doors and
edge rows, and lift into 26 pieces. The learned models have overtaken it on
appearance (20M D4) and tied it on geometry at the group level (20M plain), and no
learned model has yet matched it on `n_components`. The gap to the ceiling is
unchanged by any of this: 493 against 990 to 1205, held by `geom_kid` 13.8 and by
enterability 0.73, both of which no facade generator reaches because the interior
behind the faces is empty.

**What the track has established.** Four joint models and two levers on one card:
size without data memorises and does not help the card; data (eight symmetries)
helps likelihood, appearance, traversal and novelty and leaves geometry where it
was; and the geometry pillar, which is BlockScore for every facade arm, is held by
edge agreement between faces and by the empty interior, neither of which more
training on the same representation addresses. The next levers are therefore not
training levers: an edge-consistency term or decoding constraint so adjacent faces
share their corner column, and the second stage that fills the interior the
ceiling (T65) says is missing.

## T69. BrickNet recipe on Minecraft walks: data built, trainer fitted to 16 GB, pilot launched (2026-10-30)

Setup in notes.md §61. Files in `data/minecraft/bricknet/`, code in `libs/BrickNet/repro/`.

| file | source tier | builds | walks | blocks | note |
|---|---|--:|--:|--:|---|
| paths_val + captions_val | eval-holdout, ≤800 blocks, largest component | 494 of 798 | 1,976 | 930,792 | 4 walks per house, one per caption |
| paths_sft + captions_sft | sft-houses32, same cut | 927 of 1,863 | 3,708 | 1,813,356 | |
| paths_pt | pretrain-clean, 320k random seqs ≤600 blocks, components ≥4 | 320,000 | 1,322,324 | 69,550,049 | median 10 blocks per walk |
| paths_pt16 (training pool) | components ≥16 blocks | | 524,542 | 63,935,261 | median 59, p90 344, max 600; ≈600M tokens |

Grammar cost on val walks: 9.43 Qwen3 tokens per block (walk p50 4,418 tokens, p90 6,823,
max 8,038); captions p50 24 tokens. Round-trip 2,500 walks exact; ground-truth realisation
renders correctly (`outputs/figures/bricknet_repro/mc_val_truth_20.png`).

Trainer smoke (0.6B, LoRA r32, max_len 6400, val walks, 16 steps of 4 sequences):

| config | peak GPU | time per sequence |
|---|--:|--:|
| micro-batch 1, no compile | 11.67 GiB | 0.53 s |
| micro-batch 1, torch.compile | 10.70 GiB | 23 s (recompiles per width) |
| micro-batch 2, gradient checkpointing | 3.70 GiB | 0.79 s |

liger-kernel pinned to 0.6.5 (0.8.x requires transformers ≥5). Pilot `mc_pt_base`: 6,000
iterations, global batch 32 via 32 accumulated micro-batches, warmup 300, lr 5e-4 cosine to
1%, eval every 500 on 256 val walks; launched 2026-10-30 23:10, 11.2 GiB resident, 7.9 s per
iteration (ETA 2026-10-31 midday). Queued behind it (`repro/run_mc_chain.sh`): its Table 2
readout (`repro/run_mc_eval.sh`, 256 samples nucleus and ancestral), then `mc_pt_legoinit`
(same budget, initialised from `kulits/BrickNet-0.6B-PT`) and its readout.

**Pilot outcome (2026-11-02).** Finished on schedule (13.2 h). Validation per-token NLL on
256 holdout-house walks, still falling at the end:

| step | 500 | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 |
|---|--:|--:|--:|--:|--:|--:|--:|
| eval loss | 0.762 | 0.732 | 0.707 | 0.695 | 0.685 | 0.678 | 0.677 |

Perplexity 1.97 (BrickNet-0.6B-PT on its own val: 1.32; different medium, grammar and
budget, so not a comparison). Table-2-style readout, 256 unconditional 100-block samples
per mode (`repro/run_mc_eval.sh mc_pt_base`, `mc_eval.py`):

| sampler | Conn | Coll | fully parsable | parsable and collision-free | collisions p50/p90 | compact (max dim ≤20) | compact and collision-free |
|---|--:|--:|--:|--:|--:|--:|--:|
| nucleus 20/0.95 | 99.6±0.6 | 53.0±4.4 | 99.2% | 33.2% | 4/16 | 37% | 2% |
| ancestral | 91.1±2.5 | 36.9±3.3 | 77.7% | 9.0% | 7/17 | 67% | 5% |

The grammar is learned (nucleus samples are 99% parsable; BrickNet-0.6B gets 86%) and the
collision-free action count is three times BrickNet's 16 on LEGO, but the collision-free
samples are dominated by degenerate walks: 48% of nucleus collision-free samples are a
line (second bounding-box dimension ≤2), median bounding box 29×8×1. Samples with
collisions are blobby single-material fragments, not buildings
(`outputs/figures/bricknet_repro/mc_pt_base_ns_20.png` collision-free,
`mc_pt_base_ns_all_20.png` all). Read: 192k random-walk fragments of mostly terrain-like
windows teach the walk syntax and local material continuity, not building structure.
`mc_pt_legoinit` (LEGO-adapter init, same budget) launched 2026-11-02 17:10 after the
queue stalled for a day (notes §61); the comparison at equal steps is the transfer test.
The chain stalled because its wait loop's pgrep pattern matched the shell that launched
it; rewritten without the wait.

**LEGO-init twin (2026-11-03).** `mc_pt_legoinit`, same data and budget, LoRA merged from
`kulits/BrickNet-0.6B-PT` before the new adapter. Val NLL at every 500-step eval, base
first:

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 5000 | 5500 | 6000 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| base init | 0.761 | 0.732 | 0.717 | 0.707 | 0.701 | 0.694 | 0.690 | 0.684 | 0.681 | 0.678 | 0.676 | 0.676 |
| LEGO init | 0.770 | 0.745 | 0.728 | 0.719 | 0.710 | 0.704 | 0.700 | 0.695 | 0.691 | 0.688 | 0.687 | 0.686 |

The LEGO-pretrained start is behind at every checkpoint by about 0.01 nats per token and
never catches up: no transfer from LEGO connector walks to Minecraft face walks at this
budget, if anything a small penalty from the merged adapter. Its sampling readout crashed
on a sample whose node label was the word "An" (English leaking from the LEGO adapter's
caption training; the parser raised the wrong exception class, fixed) and is being
re-run at the head of the SFT queue.

**SFT queue (launched 2026-11-04 00:50, `repro/run_mc_chain2.sh`).** For each PT adapter:
`repro/run_mc_sft.sh` runs the SFT recipe (lr 2.5e-4, alpha 64, global batch 8, loss on
the walk only) for 2,000 iterations (16k caption-walk pairs, 4.3 epochs of the 3,708) at
max_len 8192 with gradient checkpointing, eval every 500 on the 1,976 val pairs; then
`repro/run_mc_sft_eval.sh` samples one walk per val caption for 256 houses (natural EOS,
4,096 new tokens, nucleus) and scores/renders it.

**SFT training (2026-11-04).** Both runs finished (about 2.5 h each). Val NLL on the walk
tokens of the 1,976 holdout caption-walk pairs:

| step | 500 | 1000 | 1500 | 2000 |
|---|--:|--:|--:|--:|
| SFT from base-init PT | 0.657 | 0.651 | 0.649 | 0.649 |
| SFT from LEGO-init PT | 0.672 | 0.659 | 0.654 | 0.653 |

Captions buy 0.027 nats per token over the unconditional PT model (0.676 to 0.649), and
the LEGO-init gap persists through SFT.

**Readout bug (found 2026-11-05 by the voxel-pipeline session).** The first sampling
readouts of `mc_pt_legoinit` and both SFT runs loaded only the run's own adapter onto
plain Qwen3, whereas train.py had merged the parent adapter(s) into the base before
training it; the samples were garbage (0 to 3% parsable) and are quarantined in
`repro/gen/invalid_single_adapter/`. `run_mc_eval.sh` and `run_mc_sft_eval.sh` now take
the base adapters as trailing arguments and stack them in training order. The training
eval losses above were computed on the merged model and are unaffected. Re-run
(`repro/run_mc_chain3.sh`, 2026-11-05) results. Unconditional 100-block readout of the
LEGO-init PT run, base-init rows repeated for comparison:

| run | sampler | Conn | Coll | fully parsable | collision-free |
|---|---|--:|--:|--:|--:|
| base init | nucleus | 99.6±0.6 | 53.0±4.4 | 99.2% | 33.2% |
| LEGO init | nucleus | 100.0±0.1 | 45.3±4.3 | 99.6% | 26.2% |
| base init | ancestral | 91.1±2.5 | 36.9±3.3 | 77.7% | 9.0% |
| LEGO init | ancestral | 91.4±2.6 | 35.2±3.1 | 79.7% | 7.4% |

Caption-conditioned readout of `mc_sft_base` (stack mc_pt_base then mc_sft_base; one
nucleus sample per holdout caption, 256 houses, natural EOS, 4,096-token budget):

| metric | value |
|---|--:|
| fully parsable | 91.8% |
| blocks per sample, p50 / p90 | 216 / 372 |
| ended with EOS inside the budget | 95% |
| collisions per sample, p50 / p90 | 22 / 54 |
| collision-free samples | 1.2% |
| bounding box p50 | 18×13×8 |
| max dimension ≤32 | 93% |
| actions before first collision (Coll) | 32.1±2.3 |

The conditioned samples are compact and house-sized with a house palette (planks,
cobblestone, logs, leaves, glass) but are amorphous clumps, not buildings
(`outputs/figures/bricknet_repro/mc_sft_base_val_20.png`, realised first-node-wins), and
almost every walk revisits a cell. For the voxel bench they are realised with collisions
kept (`mc_eval.py --all`).

`mc_sft_legoinit` is confounded: chain2 passed only the Minecraft delta as `--pt_adapter`,
so its SFT ran on plain Qwen3 plus a delta trained relative to the LEGO-merged model (the
eval loss 0.653 above is that confounded arm). Its readout with the three-adapter stack
produced empty walks (median 0 nodes) and is discarded.

**Clean LEGO-init SFT arm `mc_sft_legoinit2`** (2026-11-05, `repro/run_mc_chain4.sh`;
`--pt_adapter kulits/BrickNet-0.6B-PT --pt_adapter repro/runs/mc_pt_legoinit/adapter`,
same 2,000 iters, peak 2.9 GiB). Eval loss at 500/1000/1500/2000:
0.663, 0.657, 0.654, 0.654 (base arm 0.657, 0.651, 0.649, 0.649): the LEGO-init stack
stays 0.005 nats/token behind the base arm through SFT, the same gap as after PT. Readout
with the three-adapter stack (kulits PT, mc_pt_legoinit, mc_sft_legoinit2), same protocol as
the base arm; both columns recomputed with one script (bounding box from the realised
volumes, "ended inside budget" = under 4,090 Qwen3 tokens):

| metric | base init | LEGO init |
|---|--:|--:|
| fully parsable | 91.8% | 93.8% |
| nodes per sample, p50 / p90 | 216 / 372 | 212 / 336 |
| ended inside the 4,096-token budget | 91% | 95% |
| collisions per sample, p50 / p90 | 22 / 54 | 27 / 51 |
| collision-free samples | 1.2% | 2.0% |
| actions before first collision (Coll) | 32.1±2.3 | 31.2±1.9 |
| actions before first grammar failure (Conn) | 235.9±11.8 | 225.2±10.3 |
| realised bounding box p50 | 14×9×15 | 15×8×15 |
| max dimension ≤32 | 94% | 95% |

Every difference is inside the standard error of the readout. The LEGO-init samples
(`outputs/figures/bricknet_repro/mc_sft_legoinit2_val_20.png`) are the same compact
house-palette clumps as the base arm: single-material plank or cobblestone blobs with
stray arms, no walls, roofs or openings. Verdict for the transfer question: null on eval
loss (0.686 vs 0.676 after PT, 0.654 vs 0.649 after SFT) and null on every sampling
metric, so LEGO connector pretraining buys nothing for Minecraft face walks at a 220M-token
PT budget plus 2k SFT steps. Closed.

## T70. Post-training, first rung, facades: temperature and best-of-N against the raw 20M D4 model (2026-10-31)

Run `run_20261031_080537_bench_facade_post-training_best-of-4_on_face` (`--tier both
--n 399`, self-validation 6/6). The raw 20M D4 joint facade model of T68 beside two
inference-time arms drawn from the same checkpoint: the same sampler at temperature 0.7
(395 builds, four empty draws dropped), and best-of-4 under the `face_solidity` reward of
`blockgen/posttrain/rewards.py` (1,596 draws at temperature 1.0, the best of each four kept,
399 builds; reward mean 0.64 over all draws, 0.79 over the kept). Patchwork and the six-face
ceiling as before. The reward is not a BlockScore pillar input; by the honesty rule
(`blockgen/posttrain/__init__.py`) its own statistics are method-internal, movement on
coherence is pre-registered transfer, and the claims below are about the pillars the reward
did not touch.

| arm | BlockScore | appearance | geom_kid | mv_dino_kid | components | lcc | enterable | nn pct |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| real@facades6 (ceiling) | 493 | 1 | 13.8 | 0.001 | 13.5 | 0.81 | 0.73 | 0.51 |
| 20M D4, temperature 0.7 | 637 | 280 | 17.8 | 0.268 | 29.8 | 0.84 | 0.31 | 0.51 |
| 20M D4, best-of-4 face_solidity | 871 | 313 | 24.4 | 0.300 | 27.6 | 0.93 | 0.49 | 0.42 |
| facade_patchwork | 990 | 561 | 27.7 | 0.538 | 26.5 | 0.90 | 0.23 | 0.48 |
| 20M D4, raw (T68) | 1082 | 487 | 30.3 | 0.468 | 60.0 | 0.81 | 0.35 | 0.48 |

(`lcc` is coherence `lcc_ratio`; other columns as in T68. Holm groups on `geom_kid`: ceiling
c, temperature d, best-of-4 e, patchwork f, raw g, every pair separated; on `mv_dino_kid` the
two post-training arms share a group and both are separated from raw and from patchwork.)

**Temperature is the largest single lever the facade track has found.** The same
checkpoint sampled at 0.7 instead of 1.0 goes from 1082 to 637, from the worst learned
facade arm on the T68 card to the first learned arm of any kind below patchwork, below
solidify (846) and below the shuffled-materials control (710), with `geom_kid` 17.8 against
the ceiling's 13.8 and 30.3 for the raw decode. Every pillar moves the same way:
appearance 487 to 280, components 60 to 30, palette the exception (25 to 44 spreads: a
cooler decode commits harder to one material, the drift T68 saw with D4 made larger).
Memorization stays at held-out real (0.51) and the diversity gate is untouched (0.928
against 0.919 raw). The cost is four empty draws in 399 and the palette pillar. Notes.md §60
predicted this from the face sheets: the depth speckle that fragmented the shells was
sampling noise, and T66 and T68 scored every facade model on its noisiest decode.

**Best-of-4 on face solidity is the second lever and moves different things.** It takes
the raw decode from 1082 to 871, below patchwork, with `geom_kid` 24.4 and appearance 313,
and it is the arm that moves the pillars temperature does not: enterable 0.35 to 0.49
(patchwork 0.23, ceiling 0.73), `lcc_ratio` 0.81 to 0.93 (above the ceiling's 0.81, which
is held down by the ceiling's own detached roof pieces), `wall_frac` 0.44 to 0.54. Its
components (27.6) are the pre-registered transfer and are not claimed. It pays with novelty:
nearest-train IoU 0.295 to 0.369 and memorization percentile 0.478 to 0.420, both still
inside the gates, and the direction to watch when k grows. The two levers are orthogonal
enough that the next card should carry their product: best-of-k at temperature 0.7.

**Rejection-sampling fine-tuning did not run on this card.** The derived tier was written
(399 kept builds, 0.76M tokens, `rft-houses32_facades_joint_d4_20m-face_solidity`), but the
20M model would not train at the default micro-batch of 32,768 tokens beside an 11 GB tenant
on the GPU; the retry at a quarter of the micro-batch with four accumulation steps is
queued and gets its own entry. The card and the voxel first rung (M2b seed 1, `lcc26`)
are in `scripts/posttrain_first_pass.sh`.

## T71. Post-training, first rung, voxels: best-of-4 and rejection-sampling FT on the M2b model (2026-10-31)

Run `run_20261031_105428_bench_voxel_post-training_best-of-4_and` (`--tier both --n 256
--baselines`, self-validation 6/6). The parent is `m2b_sft_s1` (T60: 5M, pretrained on
anneal then SFT, dropout 0, holdout NLL 0.420), scored here from its own 256 end-of-run
samples at temperature 1.0. Two post-training arms from the same checkpoint: best-of-4
under `lcc26` (the 26-connected largest-component fraction, `blockgen/posttrain/rewards.py`;
1,024 draws, 256 kept), and rejection-sampling fine-tuning, `rft-voxel-5m.yaml` from the
parent's `best.pt` on the 256 kept builds mixed 1:1 with `sft-houses32` for 4,000 steps at
lr 5e-5, dropout 0, `select=best` on holdout NLL, 256 samples at temperature 1.0. The reward
is not a pillar input; coherence movement is pre-registered transfer. An earlier card of the
same name (`run_20261031_031101_...`) holds the raw arm alone after the best-of-N step ran out
of GPU memory at the default sampling batch; it is superseded.

| arm | BlockScore | appearance | geom_kid | mv_dino_kid | components | lcc | enterable | nn IoU | nn pct |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| m2b_sft_s1, best-of-4 lcc26 | 340 | 213 | 16.3 | 0.344 | 4.9 | 0.98 | 0.21 | 0.395 | 0.61 |
| m2b_sft_s1, raw | 386 | 254 | 18.6 | 0.410 | 11.3 | 0.94 | 0.20 | 0.353 | 0.64 |
| m2b_sft_s1, RFT lcc26 | 395 | 221 | 19.0 | 0.357 | 9.5 | 0.96 | 0.26 | 0.379 | 0.62 |
| patchwork | 432 | 432 | 18.5 | 0.696 | 6.6 | 0.87 | 0.43 | 0.282 | 0.55 |
| real@solidify | 482 | 0 | | | | | 0.18 | | |

(`nn IoU` is the novelty gate `voxel_nn_iou_mean`, real 0.375; `nn pct` the memorization
gate, 0.5 is held-out real. Holm groups on `geom_kid`: best-of-4 alone in c; patchwork, raw
and RFT share d. On `mv_dino_kid`: best-of-4 c, RFT cd, raw de, patchwork f.)

**Best-of-4 on connectivity is a real move on the voxel model too.** 386 to 340, with
`geom_kid` 18.6 to 16.3 (separated from every other learned arm), appearance 254 to 213 and
`mv_dino_kid` 0.410 to 0.344. The pre-registered transfer is large (components 11.3 to 4.9,
`lcc_ratio` 0.94 to 0.98, now at real's 0.98) and is not claimed. The arm is the first voxel
arm of the new pipeline to rank on the board, and under the current descriptor it sits
below patchwork, solidify and shuffled materials. The novelty gates move toward the training
set as they did for facades (nearest-train IoU 0.353 to 0.395, real 0.375; diversity 0.881
to 0.847), inside the gates at k=4.

**Rejection-sampling fine-tuning did not beat the inference-time rung at this budget.**
395 against 386 raw and 340 best-of-4; `geom_kid` 19.0 is in the raw arm's group. It moved
appearance (254 to 221, `mv_dino_kid` 0.357, between raw and best-of-4), components (11.3 to
9.5) and traversal (enterable 0.20 to 0.26, headroom 0.47 to 0.57), but not the geometry
pillar that decides BlockScore. The selector is the confound: `select=best` on holdout NLL
chose step 500 of 4,000 (0.419, the parent's 0.420), because fine-tuning on the model's own
filtered samples lowers holdout likelihood from the first eval on, so the card scores a
500-step nudge. The `reward` selector (`blockgen/posttrain/select.py`) exists for exactly
this and was not used here; the rerun under it, and a larger kept set (k=4 at n=1,024
rather than 256), are the next voxel experiments. Rung 1 is the baseline rung 2 has to beat,
and on both tracks it has set that bar higher than the raw model by 40 to 440 BlockScore.

## T72. Post-training, second rung, facades: rejection-sampling FT on face solidity (2026-10-31)

Run `run_20261031_121039_bench_facade_post-training_rejection-sampling` (`--tier both
--n 399`, self-validation 6/6), the T70 arms plus the fine-tuned model:
`rft-facades-joint-d4-20m.yaml` from the 20M D4 parent's `best.pt` on the 399 best-of-4
builds of T70 mixed 1:1 with the plain joint SFT shard, 4,000 steps at lr 5e-5, dropout 0.1,
`batch_tokens 8192 x grad_accum 4` (the default micro-batch ran out of memory beside another
GPU tenant; the effective batch is unchanged), `select=best` on holdout NLL, sampled at
temperature 1.0. Run `outputs/run_20261031_114825_rft_facades_joint_d4_20m_face_solidity`.

| arm | BlockScore | appearance | geom_kid | mv_dino_kid | components | lcc | enterable | nn pct |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 20M D4, temperature 0.7 (T70) | 637 | 280 | 17.8 | 0.268 | 29.8 | 0.84 | 0.31 | 0.51 |
| 20M D4, best-of-4 (T70) | 871 | 313 | 24.4 | 0.300 | 27.6 | 0.93 | 0.49 | 0.42 |
| facade_patchwork | 990 | 561 | 27.7 | 0.538 | 26.5 | 0.90 | 0.23 | 0.48 |
| 20M D4, RFT face_solidity | 994 | 396 | 27.8 | 0.380 | 43.4 | 0.89 | 0.53 | 0.51 |
| 20M D4, raw (T68) | 1082 | 487 | 30.3 | 0.468 | 60.0 | 0.81 | 0.35 | 0.48 |

(Holm groups on `geom_kid`: RFT shares f with patchwork, separated from raw g and from
best-of-4 e; on `mv_dino_kid` RFT is alone in d, between best-of-4 and raw.)

**The fine-tune moves every pillar the right way and keeps less of the gain than best-of-4
does.** 1082 to 994, `geom_kid` 30.3 to 27.8 (tied with patchwork), appearance 487 to 396,
and it is now the most enterable facade arm on any card at 0.53 (raw 0.35, best-of-4 0.49,
ceiling 0.73), with memorization back at held-out real (0.51) where best-of-4 drifted to
0.42. The same selector confound as the voxel run (T71) caps it: holdout NLL picked step 500
of 4,000 (0.460 on the periodic subset, 0.488 on all 862 fitting holdout sequences), so the
card scores a short nudge of a model sampled at its noisiest temperature. A model trained to
produce what best-of-4 selects should be judged where best-of-4 is strongest, and the reward
selector (`select=reward`) plus a temperature-0.7 draw from the same checkpoint are the two
cheap reruns that separate "the fine-tune did not learn it" from "we scored the wrong
checkpoint at the wrong temperature".

**Where the post-training ladder stands after three cards.** On both tracks the
inference-time rung beat the raw model (facades 1082 to 637 by temperature and to 871 by
best-of-4; voxels 386 to 340 by best-of-4), and the fine-tuning rung improved on the raw
model at the raw temperature but not on best-of-N. That is the expected order for a first
pass, and the two fixes are known: select by reward, and sample at the temperature the
first rung found.

## T73. Post-training, second pass: both fine-tunes selected by reward and sampled at temperature 0.7 (2026-11-01)

Both rejection-sampling fine-tunes of T71 and T72 rerun with the same data, steps, lr and
seed, changing only the checkpoint selector (`select=reward`: the final weights and the
likelihood-best checkpoint each draw 64 builds at the config's temperature, and the set with
the higher mean trained-on reward wins) and the decode (temperature 0.7 for the selector,
the end-of-run samples and the card arm). Recipe `scripts/posttrain_rft_reward_t07.sh`; a raw
voxel arm at 0.7 was added so temperature and fine-tune separate on both tracks.

**Facades.** Card `run_20261101_160512_bench_facade_rung_2_rerun_rejection-sampling`
(`--tier both --n 399`, self-validation 6/6). The selector kept `best.pt` (step 500, mean
face_solidity 0.860 against 0.845 for step 4,000), which is the checkpoint T72 already
scored, so the new arm differs from the T72 RFT only in temperature, and from the raw T=0.7
arm only in the fine-tune. Run `outputs/run_20261101_154120_rft_facades_joint_d4_20m_face_solidity_rsel_t07`.

| arm | BlockScore | appearance | geom_kid | mv_dino_kid | components | lcc | enterable | nn pct |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| real@facades6 (ceiling) | 493 | 1 | 13.8 | 0.001 | 13.5 | 0.81 | 0.73 | 0.51 |
| **RFT face_solidity, reward-selected, T=0.7** | **553** | 190 | 15.4 | 0.182 | 17.6 | 0.94 | 0.35 | 0.58 |
| 20M D4, T=0.7 (T70) | 637 | 280 | 17.8 | 0.268 | 29.8 | 0.84 | 0.31 | 0.51 |
| 20M D4, best-of-4 (T70) | 871 | 313 | 24.4 | 0.300 | 27.6 | 0.93 | 0.49 | 0.42 |
| facade_patchwork | 990 | 561 | 27.7 | 0.538 | 26.5 | 0.90 | 0.23 | 0.48 |
| RFT face_solidity, NLL-selected, T=1.0 (T72) | 994 | 396 | 27.8 | 0.380 | 43.4 | 0.89 | 0.53 | 0.51 |
| 20M D4, raw (T68) | 1082 | 487 | 30.3 | 0.468 | 60.0 | 0.81 | 0.35 | 0.48 |

(Holm groups on `geom_kid`: the new arm is `cd`, sharing `c` with the real@facades6
ceiling and `d` with raw T=0.7; the paired gap to raw T=0.7 is 2.4 [1.3, 3.2], p_adj 0.10.
On `mv_dino_kid` the new arm is alone in `c`, separated from raw T=0.7, 0.086 [0.073, 0.101],
p_adj 0.008.)

**The fine-tune and the temperature compose on facades, and the result is the first
learned facade arm the card cannot separate from the facade ceiling on geometry.** 1082 to
553: temperature alone gives 637, the fine-tune at matched temperature takes it to 553, and
the same weights at T=1.0 scored 994, so most of what T72 looked like it lacked was the
decode. Components per build halve again (29.8 to 17.6, ceiling 13.5), the largest component
holds 94% of blocks, appearance drops 280 to 190, and the enclosed-air ratio rises from
0.004 to 0.048 (ceiling 0.069), so the fine-tuned faces start to close rooms. Two costs:
enterable stays at the raw T=0.7 level (0.35; the same weights reach 0.53 at T=1.0), and
the nearest-neighbour percentile moves away from training data (0.58), which is the
direction of novelty, not memorization. The `geom_kid` win over raw T=0.7 is a trend after
Holm; the appearance win is significant. face_solidity is the trained-on signal, so the
claim rests on the untouched pillars: geometry, appearance and traversal.

**Voxels.** Card `run_20261101_143155_bench_voxel_rung_2_rerun_rejection-sampling`
(`--tier both --n 256 --baselines`, self-validation 6/6). The selector kept `last.pt`
(step 4,000, mean lcc26 0.995 against 0.992 for step 500), but the card's arm was drawn by
`python -m blockgen.train --sample`, which then defaulted to `best.pt` (step 500) whatever the
selector chose; the default now follows `summary.json`'s `selected_ckpt`. **Correction
(2026-11-02): the voxel arm below is step 500 at T=0.7, not step 4,000.** Step 4,000 at 1.0
and 0.7 is on the follow-up card. Run `outputs/run_20261101_134001_rft_m2b_sft_s1_lcc26_rsel_t07`.

| arm | BlockScore | appearance | geom_kid | mv_dino_kid | components | lcc | thickness | enterable |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| M2b, best-of-4 lcc26 (T71) | 340 | 213 | 16.3 | 0.344 | 4.9 | 0.977 | 1.15 | 0.21 |
| M2b, raw T=1.0 (T71) | 386 | 254 | 18.6 | 0.410 | 11.3 | 0.939 | 1.12 | 0.20 |
| RFT lcc26, NLL-selected step 500, T=1.0 (T71) | 395 | 221 | 19.0 | 0.357 | 9.5 | 0.956 | 1.19 | 0.26 |
| M2b, raw T=0.7 | 431 | 374 | 20.7 | 0.604 | 6.5 | 0.980 | 1.30 | 0.11 |
| patchwork | 432 | 432 | 18.5 | 0.696 | 6.6 | 0.873 | 1.04 | 0.43 |
| **RFT lcc26, step 500, T=0.7** | **549** | 327 | 26.4 | 0.527 | 4.7 | 0.983 | 1.45 | 0.12 |

(Real held-out: thickness 1.05, components 2.7, lcc 0.983, enterable 0.95. Holm groups on
`geom_kid`: the new arm is alone in `g` with real@solidify, worse than raw T=0.7 by 5.7
[4.8, 6.8], p_adj 0.012.)

**On voxels temperature 0.7 points the other way, with or without the fine-tune.**
Temperature 0.7 hurts the raw voxel model (386 to 431: walls thicken from 1.12 to 1.30
blocks, surface-to-volume falls 1.96 to 1.34, enterable halves), and it hurts the step-500
fine-tune more (395 at 1.0 to 549 at 0.7): lcc26 at real's 0.983 and the fewest components
on the card (4.7), with walls 1.45 blocks thick, surface-to-volume 1.08 against real's 2.06,
and `geom_kid` next to real@solidify. Lower temperature fills, and a model fine-tuned on
lcc26-selected builds fills further. The pre-registered transfer from connectivity to
coherence holds (coherence pillar 0.79 raw to 0.69), and it is bought with geometry, which
BlockScore, as the worst pillar, refuses to trade. Whether the selector's step 4,000 also
fills is the follow-up card's question.

**Where the ladder stands.** Facades: raw 1082, T=0.7 637, RFT plus T=0.7 553, ceiling 493;
post-training closed 90% of the raw-to-ceiling gap, with the fine-tune's share significant on
appearance and a trend on geometry. Voxels: best-of-4 340 is still the best arm, every
fine-tune or temperature change scored so far has made geometry worse, and lcc26 rewards filling; the next
voxel reward needs a counterweight to filling that is not itself a BlockScore input, or the
selector should score a held-out signal rather than the trained-on reward.

## T74. Post-training, voxels: the reward-selected step-4,000 fine-tune at temperature 1.0 and 0.7 (2026-11-02)

Card `run_20261102_113421_bench_voxel_rung_2_step-4000_reward-selected` (`--tier both
--n 256 --baselines`, self-validation 6/6). The cell T73 meant to measure and did not (see
its correction): the `last.pt` weights (step 4,000) of
`outputs/run_20261101_134001_rft_m2b_sft_s1_lcc26_rsel_t07`, the checkpoint `select=reward`
chose, drawn with `--ckpt last.pt` at temperature 1.0 and 0.7. Recipe
`scripts/posttrain_voxel_step4000.sh`. Every other voxel arm of T71 and T73 is on the card at
the same n.

| arm | BlockScore | appearance | geom_kid | mv_dino_kid | components | lcc | thickness | enterable | nn pct |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| M2b, best-of-4 lcc26 (T71) | 340 | 213 | 16.3 | 0.344 | 4.9 | 0.977 | 1.15 | 0.21 | 0.61 |
| **RFT lcc26, step 4,000, T=1.0** | **359** | 225 | 17.2 | 0.363 | 6.9 | 0.973 | 1.16 | 0.27 | 0.58 |
| M2b, raw (T71) | 386 | 254 | 18.6 | 0.410 | 11.3 | 0.939 | 1.12 | 0.20 | 0.64 |
| RFT lcc26, step 500, T=1.0 (T71) | 395 | 221 | 19.0 | 0.357 | 9.5 | 0.956 | 1.19 | 0.26 | 0.62 |
| M2b, raw T=0.7 (T73) | 431 | 374 | 20.7 | 0.604 | 6.5 | 0.980 | 1.30 | 0.11 | 0.86 |
| RFT lcc26, step 4,000, T=0.7 | 478 | 263 | 23.0 | 0.425 | 6.0 | 0.984 | 1.35 | 0.14 | 0.81 |
| RFT lcc26, step 500, T=0.7 (T73) | 549 | 327 | 26.4 | 0.527 | 4.7 | 0.983 | 1.45 | 0.12 | 0.84 |

(Real held-out: thickness 1.05, components 2.7, lcc 0.983, enterable 0.95. Holm groups on
`geom_kid`: best-of-4 `c`, step 4,000 `cd`, raw `cd`, step 500 `de`; no pair among the
four T=1.0 arms survives Holm. On `mv_dino_kid` step 4,000 shares `cd` with best-of-4 and
step 500, raw is `de`.)

**The reward selector chose well, and the longer fine-tune is the best voxel fine-tune.** At
temperature 1.0 step 4,000 scores 359 against 395 for step 500, the checkpoint holdout NLL
preferred: components fall from 9.5 to 6.9 and the largest component from 0.956 to 0.973
of blocks, with wall thickness and enterability unchanged (1.16, 0.27) and memorization at
its least on the card (0.58). It closes 59% of the gap from raw (386) to best-of-4 (340)
without the four-fold sampling cost, but the card cannot separate any of the four T=1.0 arms
after Holm, so the ordering best-of-4, step 4,000, raw, step 500 is a trend. Temperature 0.7
hurts every voxel model it touches, step 4,000 included (359 to 478, walls 1.16 to 1.35),
so on voxels the fine-tune should be sampled at 1.0, and T73's reading that the
reward-selected checkpoint fills was a temperature effect on the wrong checkpoint.

**Where the ladder stands.** Facades: post-training plus temperature 0.7 is the best
learned arm (553, ceiling 493, T73). Voxels: best-of-4 (340) and the reward-selected
fine-tune at 1.0 (359) lead the raw model (386) by a trend, and temperature 0.7 hurts. The
two tracks want opposite temperatures, so every post-training arm should be scored at its
track's temperature, and a voxel arm at n=256 cannot resolve gaps of this size; the next
voxel comparisons should be quoted at n=399.

## T75. The pooled-face probe: a fixed coarse-to-fine hierarchy of real faces, by level (2026-11-02)

Runs `run_20261102_163427_facade_pool_probe_roof` (`+y`) and
`run_20261102_163441_facade_pool_probe_front` (`-z`), `scripts/facade_pool_probe.py`
(notes.md §63), every test-split house of `houses_32` with a block on that face,
n=399 each. A face is pooled by f under a rule (`facade.pool`), repeated back to its
size, and compared with the original; the face statistics of §60 are then read on
the pooled level. Means over faces. `depth exact` and `block agree` are on cells
filled in both; `jumps` is the relief jump rate; `comps` 8-connected components of
the mask; `empty` the share of faces with no cell left.

Roof plan (`+y`):

| rule | f | mask IoU | recall | precision | depth exact | depth MAE | block agree | jumps | comps | empty |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| real | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00 | 1.000 | 0.095 | 1.38 | 0.000 |
| any | 2 | 0.922 | 1.000 | 0.922 | 0.711 | 0.63 | 0.779 | 0.107 | 1.13 | 0.000 |
| any | 4 | 0.859 | 1.000 | 0.859 | 0.489 | 1.59 | 0.678 | 0.063 | 1.02 | 0.000 |
| any | 8 | 0.826 | 1.000 | 0.826 | 0.357 | 2.68 | 0.587 | 0.032 | 1.00 | 0.000 |
| any | 16 | 0.817 | 1.000 | 0.817 | 0.278 | 3.55 | 0.509 | 0.012 | 1.00 | 0.000 |
| majority | 2 | 0.932 | 0.992 | 0.938 | 0.747 | 0.55 | 0.778 | 0.103 | 1.12 | 0.000 |
| majority | 4 | 0.891 | 0.959 | 0.921 | 0.613 | 1.08 | 0.673 | 0.063 | 1.04 | 0.000 |
| majority | 8 | 0.848 | 0.946 | 0.885 | 0.521 | 1.73 | 0.583 | 0.029 | 1.01 | 0.000 |
| majority | 16 | 0.806 | 0.928 | 0.827 | 0.441 | 2.26 | 0.491 | 0.005 | 0.96 | 0.035 |
| center | 2 | 0.921 | 0.959 | 0.955 | 0.723 | 0.59 | 0.749 | 0.104 | 1.13 | 0.000 |
| center | 4 | 0.874 | 0.935 | 0.923 | 0.567 | 1.11 | 0.620 | 0.065 | 1.04 | 0.000 |
| center | 8 | 0.821 | 0.907 | 0.885 | 0.458 | 1.79 | 0.506 | 0.035 | 1.01 | 0.000 |
| center | 16 | 0.774 | 0.895 | 0.800 | 0.344 | 2.61 | 0.389 | 0.010 | 0.94 | 0.060 |

Front face (`-z`):

| rule | f | mask IoU | recall | precision | depth exact | depth MAE | block agree | jumps | comps | empty |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| real | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00 | 1.000 | 0.115 | 1.14 | 0.000 |
| any | 2 | 0.858 | 1.000 | 0.858 | 0.737 | 0.62 | 0.696 | 0.090 | 1.04 | 0.000 |
| any | 4 | 0.761 | 1.000 | 0.761 | 0.520 | 1.41 | 0.556 | 0.051 | 1.01 | 0.000 |
| any | 8 | 0.710 | 1.000 | 0.710 | 0.354 | 2.28 | 0.466 | 0.026 | 1.00 | 0.000 |
| any | 16 | 0.695 | 1.000 | 0.695 | 0.261 | 2.97 | 0.412 | 0.007 | 1.00 | 0.000 |
| majority | 2 | 0.875 | 0.986 | 0.886 | 0.766 | 0.55 | 0.691 | 0.086 | 1.04 | 0.000 |
| majority | 4 | 0.803 | 0.931 | 0.852 | 0.640 | 0.98 | 0.544 | 0.045 | 1.03 | 0.000 |
| majority | 8 | 0.741 | 0.901 | 0.805 | 0.561 | 1.30 | 0.456 | 0.019 | 1.01 | 0.000 |
| majority | 16 | 0.676 | 0.886 | 0.708 | 0.469 | 1.63 | 0.383 | 0.003 | 0.94 | 0.060 |
| center | 2 | 0.851 | 0.922 | 0.915 | 0.731 | 0.62 | 0.644 | 0.089 | 1.08 | 0.000 |
| center | 4 | 0.765 | 0.871 | 0.862 | 0.585 | 1.06 | 0.464 | 0.051 | 1.06 | 0.000 |
| center | 8 | 0.696 | 0.831 | 0.806 | 0.498 | 1.41 | 0.360 | 0.022 | 1.01 | 0.000 |
| center | 16 | 0.574 | 0.756 | 0.614 | 0.293 | 2.12 | 0.229 | 0.002 | 0.82 | 0.180 |

Faces fill a median 86 percent (roof plan) and 71 percent (front) of their box and
hold a median 6 to 7 distinct depths and 7 to 10 distinct blocks.

**The hierarchy exists in the data, and its levels read as the sheets show**
(`sheet_any.png`, `sheet_majority.png` in each run). On a 32-cell face: f = 16 is
the box and its dominant material; f = 8 the wall, roof and ground bands; f = 4
storeys, the roof line and the footprint shape (ridge, wings, courtyards on the
roof plan); f = 2 openings and trim; f = 1 the relief. Windows and doors are one
or two cells wide and survive only to f = 2, so the periodic structure of a facade
is a fine-level feature, not the 8 x 8 feature research.md H.7 guessed.

**Pooling does not create the failure modes the models have.** Every pooled level
is smoother than the face (jump rate falls from 0.10 to 0.11 real to 0.05 at f = 4
and 0.02 at f = 8) and one piece (components 1.00 to 1.04 from f = 4 up). The
generated faces of T68 fragment and speckle; their coarse levels would not.

**What a level loses first is depth, then block, then mask.** Depth exactness is
0.71 to 0.77 at f = 2 and 0.49 to 0.64 at f = 4 with a mean error of one block,
which is a pitched roof crossing four depths inside one 4 x 4 cell. Block
agreement is 0.78 at f = 2 and 0.56 to 0.68 at f = 4: trims, frames and second
wall materials sit under a different material than their cell's vote. The mask is
the last to go, and on the roof plan it hardly goes at all, because the silhouette
from above is the footprint plus the ground plate. For a residual hierarchy this
says the finest two levels carry openings, trim and relief; the coarse levels carry
the shape and the palette.

**Rules.** `any` keeps every real cell at every level (recall 1.000) and pays in
precision, 0.70 on the front at f = 8; `majority` and `center` delete thin
features, losing a tenth to a fifth of the cells by f = 8 and emptying 6 to 18
percent of front faces at f = 16. The three agree on depth and block within a few
points. `any` is the rule for step 2, the cascade, with `majority` as the control.

## T76. Mesh interior probe: Objaverse buildings with hidden interiors, n=64 (2026-11-02)

Run `outputs/run_20261102_173929_mesh_probe` (`scripts/run_mesh_probe.py`, notes.md
§64; smoke run `run_20261102_172754_mesh_probe_smoke`, n=4). 256 Objaverse
building/room meshes with >= 25% of face area invisible from every exterior
direction: 128 reference, 128 pool, 64 probed. Separation = (probe - real
held-out) / noise-floor sd, [95% paired-subsample interval]. Strip and Fill keep
the exterior renders identical to 0.03/255 mean pixel difference.

| metric | Strip | Fill | noise 0.5% | missing wall | decim 5% |
|---|--:|--:|--:|--:|--:|
| MV-KID DINOv2, 4 views | +0.0 [-0.0, +0.1] | +0.0 [-0.0, +0.1] | +28.3 | +7.0 | +24.5 |
| MV-KID CLIP-L, 4 views | +0.2 [+0.1, +0.3] | +0.2 [+0.1, +0.3] | +15.6 | +4.5 | +10.4 |
| MV-KID DINOv2, 8 views | +0.0 [-0.0, +0.1] | +0.0 [-0.0, +0.1] | +42.1 | +7.9 | +29.4 |
| MV-KID DINOv2, 16 views | +0.0 [-0.0, +0.1] | +0.0 [-0.0, +0.1] | +29.5 | +6.9 | +22.3 |
| MV-FD DINOv2 / CLIP-L | +0.0 / +0.0 | +0.0 / +0.0 | +9.0 / +8.1 | +1.7 / +1.6 | +8.1 / +6.3 |
| CLIP score (drop) | -0.0 [-0.1, +0.0] | -0.0 [-0.1, +0.1] | -0.3 | +0.3 | +1.7 |
| cutaway KID DINOv2 | +9.0 [+6.1, +11.8] | +10.2 [+7.4, +13.4] | +32.9 | +5.9 | +22.5 |
| cutaway KID CLIP-L | +5.5 [+3.8, +7.5] | +11.7 [+9.1, +14.4] | +24.7 | +4.9 | +12.1 |
| CD all faces, MMD | +2.8 [+2.3, +3.7] | -0.1 [-1.0, +0.3] | +0.3 | +5.8 | +2.1 |
| CD all faces, 1-NNA | +1.8 [+1.2, +2.9] | +1.1 [+0.2, +1.9] | +0.6 | +5.3 | +0.8 |
| CD visible faces, MMD | -0.2 [-0.3, -0.0] | +0.0 [-0.2, +0.2] | -0.3 | +4.7 | +1.2 |
| CD visible faces, 1-NNA | +0.0 [-0.3, +0.2] | +0.1 [-0.3, +0.5] | -0.1 | +4.0 | +0.7 |
| CD render depth, MMD | +0.0 [-0.1, +0.1] | -0.0 [-0.1, +0.1] | +0.2 | +3.6 | +1.0 |
| CD render depth, 1-NNA | -0.1 [-0.3, +0.2] | +0.0 [-0.3, +0.6] | -0.4 | +3.8 | +1.0 |
| CD voxel volume, MMD | +1.9 [+1.5, +2.7] | -1.8 [-2.7, -1.4] | +0.2 | +1.0 | +0.3 |
| 1-IoU voxel, MMD | +1.7 [+1.3, +1.9] | -1.8 [-2.6, -1.6] | +0.4 | +1.7 | +1.6 |
| **V-KID (geom_kid)** | **+11.9 [+10.4, +13.3]** | **+50.8 [+46.4, +56.1]** | +12.1 | +8.6 | +14.0 |

1-COV rows are within 3 floors everywhere and omitted (full 25-row table in the
run's `table.md`). Voxel count vs real: Strip x0.86, Fill x1.37.

**The blindness transfers off Minecraft.** Every exterior render metric, every
backbone and view count, CLIP score, and chamfer on exterior or render-derived
points sits within 0.3 floors of real under both probes, while the same metrics
resolve the exterior controls at 4 to 42 floors. V-KID on the voxelised mesh
separates Fill at 51 and Strip at 12; cutaway renders see both at 6 to 12. The one
partial exception is chamfer on points from *all* faces, which sees Strip at 2 to
3 floors and Fill at about 1: interior walls disappearing is visible to a sampler
that reaches them, but nearest-neighbour set statistics at n = 64 resolve it
weakly, and they are equally weak on the exterior controls (0.5% noise does not
move them), so part of that is power rather than interior sensitivity.

## T77. Post-training, facades: best-of-4 at temperature 0.7 reaches the facade ceiling on geometry (2026-11-02)

Card `run_20261102_131429_bench_facade_rung_1_and_2_at_temperature_0_7` (`--tier both
--n 399`, self-validation 6/6). Best-of-4 on face_solidity from the 20M D4 parent, drawn at
temperature 0.7 (1,578 drawn, 395 kept, mean face_solidity 0.625 to 0.861; recipe
`scripts/posttrain_facade_bon_t07.sh`), against every facade post-training arm.

| arm | BlockScore | appearance | geom_kid | mv_dino_kid | components | lcc | enterable | nn pct |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **best-of-4, T=0.7** | **483** | 155 | 13.5 | 0.149 | 15.9 | 0.95 | 0.37 | 0.49 |
| real@facades6 (ceiling) | 493 | 1 | 13.8 | 0.001 | 13.5 | 0.81 | 0.73 | 0.51 |
| RFT step 500, T=0.7 (T73) | 553 | 190 | 15.4 | 0.182 | 17.6 | 0.94 | 0.35 | 0.58 |
| raw, T=0.7 (T70) | 637 | 280 | 17.8 | 0.268 | 29.8 | 0.84 | 0.31 | 0.51 |
| best-of-4, T=1.0 (T70) | 871 | 313 | 24.4 | 0.300 | 27.6 | 0.93 | 0.49 | 0.42 |
| raw, T=1.0 (T68) | 1082 | 487 | 30.3 | 0.468 | 60.0 | 0.81 | 0.35 | 0.48 |

(Holm groups on `geom_kid`: best-of-4 at 0.7 and the ceiling share `c`, the fine-tune is `cd`,
raw at 0.7 `d`; best-of-4 beats raw at 0.7 by 4.3 [3.5, 5.2], p_adj 0.009, and leads the
fine-tune by 1.9, p_adj 0.20. On `mv_dino_kid` best-of-4 beats the fine-tune, 0.033
[0.022, 0.042], p_adj 0.009.)

**Best-of-4 and temperature 0.7 compose, and the result sits on the facade ceiling.** The
geometry pillar is 483 against 493 for real faces lifted the same way, indistinguishable
after Holm, so on `geom_kid` the facade generator has caught up with its own
representation: what still separates it from real houses is the lift (empty interiors), not
the faces. Appearance is where the gap remains (155 against 1), and it is the best
appearance of any learned facade arm. Best-of-4 beats the fine-tune at matched temperature
on appearance and by a trend on geometry, without memorization (0.49, at real), so on
facades the inference-time rung is still ahead of the fine-tuning rung, as it is on voxels
(T74). Enterable stays at the 0.7 level (0.37, ceiling 0.73). The further gains on this
track have to come from the interior, since a facade arm can no longer move much on
`geom_kid` relative to the ceiling.

## T78. Voxels at 20M: the M2b recipe with only the model size changed (2026-11-02)

Card `run_20261102_161948_bench_voxel_M2b_at_20M_against_5M_raw_and` (`--tier both --n 256
--baselines`, self-validation 6/6). `anneal-pretrain-20m.yaml` (20,000 steps on `anneal`,
547M tokens, 49 minutes) then `houses32-sft-20m.yaml` (16,000 steps, lr 1e-4, dropout 0,
`select=best`), seed 1, the same shard, schedule and seed as `m2b_sft_s1`; runs
`outputs/run_20261102_142710_m2_20m_pretrain_s1` and `outputs/run_20261102_151738_m2b_20m_sft_s1`.
Recipe `scripts/m2_20m_voxel.sh`.

Both runs finished training and then ran out of memory in the end-of-run draw (64 builds per
batch at 20M), so neither wrote `summary.json` or a registry row and the in-run card is
missing; the configs now set `sample_batch: 16`. The 256 samples were drawn afterwards from
`best.pt` (step 1,500). The best-of-4 arm failed for the same missing `summary.json`; it was
rerun on 2026-11-03 (1,024 draws in 7,424 s at `--batch 16`, kept 256 by `lcc26`) and the
whole card rescored as `run_20261103_101858_bench_voxel_M2b_at_20M_against_5M_raw_and`
(self-validation 6/6, the other arms unchanged).

| arm | BlockScore | appearance | geom_kid | mv_dino_kid | components | lcc | enterable | nn pct | holdout NLL |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 5M, best-of-4 lcc26 (T71) | 340 | 213 | 16.3 | 0.344 | 4.9 | 0.977 | 0.21 | 0.61 | |
| **20M, best-of-4 lcc26** | **341** | 178 | 16.3 | 0.288 | 3.8\* | 0.988\* | 0.30 | 0.50 | |
| **20M, raw** | **346** | 215 | 16.6 | 0.347 | 8.6 | 0.949 | 0.26 | 0.58 | 0.286 |
| 5M, raw (T60, T71) | 386 | 254 | 18.6 | 0.410 | 11.3 | 0.939 | 0.20 | 0.64 | 0.420 |

(Holm groups on `geom_kid`: 5M best-of-4 `c`, 20M `cd`, 5M raw `d`; 20M against 5M raw 1.9
[1.1, 2.8], p_adj 0.09. On `mv_dino_kid` 20M and 5M best-of-4 share `c`, and 20M beats 5M raw
by 0.063 [0.041, 0.083], p_adj 0.016.)

**Four times the parameters buys what best-of-4 bought, at a quarter of the sampling cost.**
Holdout NLL falls from 0.420 to 0.286 at the selected checkpoint, and on the board the 20M
model ties 5M best-of-4 (346 against 340, same Holm groups on both realism metrics): it
beats 5M raw on appearance after Holm and by a trend on geometry, with fewer pieces per
build (8.6 against 11.3), more enterable builds (0.26) and less memorization (0.58). The SFT
stage peaks early and then overfits hard: `select=best` kept step 1,500 of 16,000 (holdout
0.286; 0.797 at the end), so a 20M SFT wants a far shorter schedule or the pretraining shard
mixed in.

\* Components and lcc are what `lcc26` selected on, so for the best-of-4 rows they are
method-internal and carry no claim (honesty rule, `blockgen/posttrain/__init__.py`).

**Best-of-4 on the 20M model moves appearance, not geometry.** On the rescored card it ranks
341, level with 5M best-of-4 (340) and 20M raw (346). Its `geom_kid` (16.3) shares Holm group
`c` with 5M best-of-4 and does not separate from 20M raw. On `mv_dino_kid` it is alone in
the best group: 0.288 against 0.347 for 20M raw (0.059 [0.040, 0.077], p_adj 0.013) and
0.344 for 5M best-of-4 (0.056 [0.034, 0.076], p_adj 0.012), so its appearance pillar falls
from 215 to 178. Geometry is the worst pillar for every voxel arm here, and at about 340 it
does not move with either model size or best-of-4 once one of them is in place. It is also
the most enterable arm (0.30) and the least memorizing (nn percentile 0.50), but builds get
smaller (max-dim W1 2.96 against 0.64 raw), the same shrinkage connectivity selection gave
at 5M. The voxel geometry floor near 340 is the thing to attack next, not by stacking
selection on size.

## T79. Dream-Cubed, the first published generator on the bench: fast-tier smoke (2026-11-02)

Generation run `outputs/run_20261102_220927_dreamcubed_village_md4p4` (notes.md §66). It
used Dream-Cubed `MD4_p4_boosted` (Apache-2.0) with class `village`, CFG 4.0, 1000 steps and
seed 0, and produced 288 chunks at 17.8 s per chunk on the shared GPU. Bench run
`outputs/run_20261102_233900_bench_dreamcubed-smoke` used the fast tier, `--n 256` and CPU.
The controls are at n=256. File-backed arms are not capped, so the external arms are at
284 and 288.

| arm | n | geom_kid | NN-IoU to train | dup rate | diversity | BlockScore (worst pillar) |
|---|--:|--:|--:|--:|--:|--:|
| real_test | 256 | -0.14 | 0.375 | 0.008 | 0.898 | 0.00 |
| dreamcubed (extracted) | 284 | 8.63 [8.16, 9.19] | 0.420 [0.400, 0.440] | 0.004 | 0.867 | **181.3** (geometry) |
| dreamcubed_raw (whole chunk) | 288 | 36.46 | 0.454 | 0.000 | 0.426 | DQ (mode collapse) |

The other `dreamcubed` pillars are palette +87.9, coherence +0.47, geometry scalars +0.67 and
traversal +1.68. Its exact-palette JSD is 0.645 against 0.072 for real, and 35.6% of its
builds are enterable against 94.9% for real.

**Dream-Cubed is not memorising our corpus.** Its mean NN-IoU to the houses_32 train split
is 0.42, within 0.05 of held-out real. Its duplicate rate is 0.4%, one build of 284, below
real_test's 0.8%. Its training data shares no source with GrabCraft, 3D-Craft or text2mc.
Vanilla village houses do occur on GrabCraft, so a small overlap was possible; the
duplicate rate rules out a large one.

**Geometry is the worst pillar.** The builds are vanilla village pieces at about half the
corpus's size, with a mean of 453 blocks and a mean max dimension of 14 against 904 and 19.
They also have little enclosed space: the sealed-interior ratio is 0.016 against 0.114.
Fragments of desert and savanna villages that are not houses add to this. The palette pillar
partly reflects our conversion, because Dream-Cubed's material-free slab and stair tokens
collapse to one material each. Scored whole, the chunks are disqualified for mode collapse,
since every chunk is a 32-wide slab of terrain. This is why the extraction step exists.

**Placement.** This is a fast-tier smoke run, not a leaderboard row: the controls are at
n=256 while the arm is at 284. Read only as a rough guide, the geometry pillar sits in the
same range as our own voxel arms. In the 2026-11-02 v1 re-score (a different run, n≈251),
canon_oriented scored 136.7 and native_oriented 106.9. The ranked number has to come from
the leaderboard run, where all arms share n.

## T80. VLM judge (gpt-4o), 3D-Craft only, pre-registered (2026-11-02)

Run `outputs/run_20261102_214026_vlm_judge`, 2,288 trials, $4.31. See notes.md §67.

| condition | view | n | P(real) | CI | prediction met |
|---|---|--:|--:|---|---|
| Fill-Interior | exterior | 256 | 0.53 | [0.50, 0.56] | chance: yes |
| Fill-Interior | roof off | 256 | 0.70 | [0.64, 0.75] | above: yes |
| Fill-Interior | quadrant cut | 128 | 0.63 | [0.53, 0.72] | above: yes |
| Hollow-Shell | exterior | 256 | 0.52 | [0.49, 0.54] | chance: yes |
| Hollow-Shell | roof off | 128 | 0.73 | [0.66, 0.80] | above: yes |
| vs uniform random | exterior | 128 | 1.00 | [1.00, 1.00] | yes |
| Gabled vs Patch-Stitch | exterior | 256 | P(Gabled) 0.28 | [0.21, 0.35] | no (judge agrees with metrics) |

System Bradley–Terry vs metrics (11 arms): Spearman MV-KID 0.76, BlockScore 0.57, V-KID 0.16.
Real ranks 6th of 11.

## T81. 3D-native rows (Uni3D, ULIP-2, chamfer, exact EMD) under three samplers: Objaverse strict multi-room and the Minecraft ladder, two seeds (2026-11-03)

Method and reading in notes §68. Cells: z seed 0 / z seed 1, where z is the null-standardised separation (arm minus real held-out, over the sd of 48 real n-draws); the raw difference at seed 0 (MMD^2 for KID rows, raw MMD for MMD rows) in brackets. Samplers: all = every face, vis = exterior-ray-visible faces, cut = cutaway-visible faces. Chamfer uses 2,048 points; EMD is exact assignment on 512 points; the encoders use 10k xyz+rgb points.

**Objaverse**: 290 strict multi-room objects (the 6 GB budget allowed 290, not 512); ref 130, pool 160, n = 96. Runs `outputs/run_20261103_004358_mesh_probe3d_s0`, `outputs/run_20261103_030030_mesh_probe3d_s1`. Seed agreement: Pearson 0.91 over 275 cells, blind-verdict agreement 96%. Licences: 213 BY, 3 BY-SA, 65 BY-NC-SA, 9 BY-NC.

| metric | strip | fill | noise | wall | decim | sd (s0) |
|---|---|---|---|---|---|---|
| dino4_kid | +0.0 / +0.0 (+7.3e-05) | +0.0 / +0.0 (+0.00015) | +34.7 / +66.6 (+0.29) | +9.5 / +15.0 (+0.079) | +21.3 / +29.0 (+0.18) | 0.00833 |
| clip4_kid | +0.0 / +0.1 (+5.5e-05) | +0.0 / +0.1 (+0.00013) | +18.8 / +21.2 (+0.097) | +8.4 / +8.4 (+0.044) | +13.6 / +13.3 (+0.07) | 0.00517 |
| dino16_kid | -0.0 / +0.0 (-0.00041) | -0.0 / +0.0 (-0.00032) | +40.3 / +72.0 (+0.38) | +8.5 / +12.1 (+0.079) | +22.3 / +27.8 (+0.21) | 0.00931 |
| dino_cut4_kid | +10.9 / +9.0 (+0.071) | +13.5 / +13.7 (+0.089) | +43.9 / +32.7 (+0.29) | +7.0 / +6.2 (+0.046) | +22.3 / +14.6 (+0.15) | 0.00655 |
| clip_cut4_kid | +4.9 / +6.4 (+0.017) | +13.0 / +16.2 (+0.045) | +30.2 / +31.5 (+0.1) | +7.9 / +8.7 (+0.028) | +11.9 / +11.5 (+0.041) | 0.00347 |
| clipscore_drop | +0.1 / -0.1 | +0.0 / -0.1 | +0.1 / -0.6 | +1.0 / +1.8 | +0.0 / -0.5 | 0.284 |
| vkid_geom | +23.7 / +26.5 | +73.8 / +118.4 | +18.1 / +31.8 | +16.2 / +13.4 | +12.0 / +12.4 | 0.289 |
| uni3d_all_kid | +4.6 / +11.5 (+0.051) | +8.5 / +14.1 (+0.094) | +13.7 / +34.0 (+0.15) | +9.8 / +15.4 (+0.11) | +4.3 / +6.7 (+0.047) | 0.011 |
| uni3d_all_fd | +5.6 / +6.8 | +8.7 / +8.1 | +10.6 / +15.5 | +6.3 / +7.1 | +5.0 / +5.0 | 0.0101 |
| uni3d_all_textscore_drop | -0.2 / -2.7 | +1.7 / +0.5 | +1.2 / +0.5 | -1.7 / -3.1 | -2.2 / -2.1 | 0.538 |
| uni3d_vis_kid | +0.4 / -0.2 (+0.0035) | +0.3 / +0.0 (+0.0026) | +9.5 / +20.7 (+0.095) | +16.5 / +18.5 (+0.17) | +7.9 / +8.3 (+0.08) | 0.01 |
| uni3d_vis_fd | +0.3 / -0.4 | +0.3 / +0.2 | +7.6 / +9.9 | +10.2 / +9.3 | +6.7 / +6.5 | 0.0103 |
| uni3d_vis_textscore_drop | +0.6 / -0.3 | +0.5 / -0.9 | +0.2 / +0.6 | -2.1 / -1.9 | -2.9 / -0.6 | 0.457 |
| uni3d_cut_kid | +4.3 / +8.4 (+0.053) | +6.3 / +12.4 (+0.078) | +3.7 / +13.2 (+0.045) | +5.6 / +10.0 (+0.069) | +1.9 / +2.9 (+0.023) | 0.0124 |
| uni3d_cut_fd | +5.2 / +5.1 | +6.5 / +5.3 | +3.7 / +5.3 | +3.9 / +3.9 | +2.6 / +2.9 | 0.0111 |
| uni3d_cut_textscore_drop | +1.6 / -0.3 | +2.5 / +0.9 | +1.4 / +0.7 | -0.1 / -1.3 | -0.4 / -0.1 | 0.544 |
| ulip2_all_kid | +2.4 / +4.2 (+0.02) | +3.3 / +8.4 (+0.028) | +5.0 / +19.6 (+0.041) | +12.5 / +29.7 (+0.1) | +7.8 / +14.0 (+0.065) | 0.00836 |
| ulip2_all_fd | +2.8 / +3.1 | +2.3 / +3.1 | +2.5 / +5.3 | +5.9 / +7.5 | +4.4 / +5.3 | 0.0145 |
| ulip2_all_textscore_drop | +0.8 / +0.4 | -0.6 / -0.6 | -0.7 / -0.5 | +0.9 / +0.9 | -0.4 / -0.3 | 0.675 |
| ulip2_vis_kid | -0.0 / -0.3 (-0.00013) | +0.1 / -0.1 (+0.00046) | +6.5 / +16.3 (+0.05) | +15.2 / +22.4 (+0.12) | +9.3 / +10.9 (+0.071) | 0.0077 |
| ulip2_vis_fd | +0.0 / -0.1 | +0.3 / +0.2 | +2.7 / +7.4 | +6.4 / +8.2 | +4.3 / +6.4 | 0.0149 |
| ulip2_vis_textscore_drop | +0.1 / -0.2 | +0.0 / +0.1 | -1.2 / -0.3 | -0.2 / +0.6 | -1.1 / -0.7 | 0.737 |
| ulip2_cut_kid | +4.2 / +4.7 (+0.034) | +4.9 / +10.8 (+0.039) | +3.5 / +11.5 (+0.028) | +8.4 / +18.8 (+0.067) | +5.1 / +6.0 (+0.04) | 0.00796 |
| ulip2_cut_fd | +4.0 / +2.3 | +2.8 / +3.0 | +2.7 / +3.9 | +5.4 / +4.5 | +4.0 / +3.7 | 0.0109 |
| ulip2_cut_textscore_drop | +1.3 / +0.4 | -0.7 / -0.3 | +0.3 / +0.6 | +0.4 / +0.5 | -0.1 / +0.0 | 0.542 |
| cd_all_mmd | +3.4 / +3.9 (+0.00094) | -0.6 / -0.4 (-0.00015) | -0.3 / +0.0 (-9.3e-05) | +9.9 / +10.9 (+0.0027) | +0.2 / +1.4 (+6.9e-05) | 0.000277 |
| cd_all_one_minus_cov | +0.5 / +0.3 | +0.7 / +1.4 | +0.0 / +1.1 | +3.3 / +3.5 | -0.7 / -0.5 | 0.0326 |
| cd_all_one_nna | +0.6 / +1.2 | +0.7 / +0.3 | -0.2 / -0.4 | +5.3 / +8.4 | -0.7 / +0.0 | 0.0473 |
| cd_vis_mmd | -0.1 / +0.2 (-2.5e-05) | -0.0 / +0.1 (-1.6e-05) | -0.6 / -0.6 (-0.00021) | +7.9 / +10.5 (+0.0028) | -0.9 / +0.3 (-0.00031) | 0.000352 |
| cd_vis_one_minus_cov | +0.0 / -0.2 | -0.2 / -0.2 | -1.6 / -0.5 | +1.2 / +1.2 | -0.4 / -1.0 | 0.0384 |
| cd_vis_one_nna | +0.1 / -0.4 | -0.1 / -0.2 | +0.0 / +0.1 | +7.6 / +7.5 | -0.7 / +0.6 | 0.039 |
| cd_cut_mmd | +2.8 / +2.9 (+0.00077) | -0.7 / -0.1 (-0.0002) | -0.2 / +0.1 (-5.5e-05) | +8.8 / +7.7 (+0.0024) | +0.2 / +0.9 (+6.5e-05) | 0.000273 |
| cd_cut_one_minus_cov | +0.7 / +1.6 | +1.0 / +0.8 | +0.2 / +1.1 | +2.2 / +2.1 | -1.5 / +0.3 | 0.0317 |
| cd_cut_one_nna | +1.3 / +1.8 | +0.7 / -0.1 | -0.2 / +0.2 | +5.8 / +6.2 | -0.9 / +0.3 | 0.0425 |
| emd_all_mmd | +3.4 / +3.8 (+0.0059) | +0.7 / +0.2 (+0.0013) | +3.7 / +2.3 (+0.0064) | +19.6 / +20.5 (+0.034) | +0.9 / +1.7 (+0.0015) | 0.00171 |
| emd_all_one_minus_cov | +1.2 / +1.0 | +0.9 / -0.4 | -1.2 / -0.6 | +7.9 / +5.2 | -1.5 / -0.8 | 0.0263 |
| emd_all_one_nna | +2.5 / +3.6 | +1.4 / +1.2 | -0.1 / +0.2 | +7.6 / +9.0 | +0.0 / +0.8 | 0.0493 |
| emd_vis_mmd | -0.0 / +0.2 (-3.1e-05) | -0.2 / +0.3 (-0.00042) | +0.5 / +0.9 (+0.0011) | +11.6 / +9.9 (+0.026) | +0.4 / +0.6 (+0.0008) | 0.00226 |
| emd_vis_one_minus_cov | -0.4 / -0.8 | -0.2 / -1.0 | -0.2 / -1.0 | +4.0 / +2.8 | -0.7 / -1.6 | 0.035 |
| emd_vis_one_nna | +0.6 / +0.0 | +0.9 / +0.8 | +0.1 / -0.9 | +6.8 / +8.3 | -0.3 / -0.8 | 0.0483 |
| emd_cut_mmd | +2.2 / +3.3 (+0.0042) | -0.1 / -0.1 (-0.00027) | +0.6 / +0.8 (+0.0011) | +13.2 / +15.2 (+0.025) | +0.5 / +1.0 (+0.001) | 0.00189 |
| emd_cut_one_minus_cov | +2.0 / +0.4 | +0.7 / -0.2 | +1.7 / +0.2 | +6.2 / +4.8 | -1.0 / -0.2 | 0.0312 |
| emd_cut_one_nna | +2.4 / +1.6 | +1.1 / -0.7 | +0.2 / +0.0 | +7.4 / +6.3 | -0.2 / -0.5 | 0.042 |
| cd_render_mmd | +0.1 / +0.0 (+2.6e-05) | +0.1 / +0.1 (+2.9e-05) | +0.0 / +0.0 (+1e-05) | +9.4 / +8.5 (+0.0039) | +0.1 / +1.0 (+4.3e-05) | 0.000418 |
| cd_vol_mmd | +3.1 / +2.3 (+0.00079) | -1.5 / -0.4 (-0.00038) | +0.7 / +0.7 (+0.00019) | +5.0 / +4.0 (+0.0013) | +0.3 / +1.0 (+7.8e-05) | 0.000257 |
| iou_vox_mmd | +3.0 / +2.4 (+0.042) | -1.7 / -1.4 (-0.023) | +1.0 / +1.0 (+0.015) | +2.7 / +2.0 (+0.038) | +1.0 / +1.2 (+0.014) | 0.0142 |

**Minecraft** (houses_32 split 0, usable subset): ref 248 val, pool 245 test, n = 128. Runs `outputs/run_20261103_004358_mc_probe3d_s0`, `_s1`. Seed agreement: Pearson 0.98 over 336 cells, verdict 93%.

| metric | solidify | hollow | canon16 | chunk_delete_40 | occ_noise_5 | jitter_columns | material_shuffle | rot90_1 | sd (s0) |
|---|---|---|---|---|---|---|---|---|---|
| uni3d_all_kid | +21.7 / +22.6 (+0.08) | +8.5 / +9.6 (+0.031) | +154.8 / +186.0 (+0.57) | +50.9 / +65.4 (+0.19) | +51.4 / +62.5 (+0.19) | +161.2 / +177.5 (+0.59) | +16.5 / +18.9 (+0.061) | +1.6 / +1.1 (+0.0058) | 0.00367 |
| uni3d_all_fd | +9.9 / +11.1 | +5.0 / +5.9 | +52.4 / +63.1 | +25.5 / +29.8 | +17.2 / +19.4 | +54.3 / +59.1 | +6.5 / +6.9 | -0.0 / +0.2 | 0.00499 |
| uni3d_all_textscore_drop | +2.7 / +1.7 | +0.1 / -0.0 | +3.4 / +2.7 | +4.4 / +3.0 | -0.0 / -1.6 | -3.5 / -4.8 | -1.9 / -2.0 | +0.2 / -0.2 | 0.308 |
| uni3d_vis_kid | -0.1 / +0.4 (-0.00022) | +0.1 / +0.4 (+0.00028) | +128.2 / +117.1 (+0.45) | +46.0 / +43.1 (+0.16) | +88.4 / +74.4 (+0.31) | +187.6 / +150.0 (+0.67) | +19.0 / +16.0 (+0.068) | +1.6 / +0.8 (+0.0055) | 0.00355 |
| uni3d_vis_fd | +0.3 / +0.5 | +0.1 / +0.3 | +50.1 / +45.6 | +29.0 / +24.0 | +37.0 / +30.5 | +75.7 / +60.9 | +8.6 / +6.6 | +0.6 / +0.3 | 0.0042 |
| uni3d_vis_textscore_drop | -0.1 / +0.4 | +0.3 / +0.7 | -0.8 / +0.8 | +1.5 / +1.3 | -2.7 / -4.7 | -6.4 / -6.8 | +0.8 / +1.2 | +1.0 / +2.4 | 0.27 |
| uni3d_cut_kid | +37.0 / +30.4 (+0.11) | +13.5 / +10.4 (+0.041) | +132.3 / +117.9 (+0.4) | +43.6 / +49.8 (+0.13) | +49.7 / +44.2 (+0.15) | +151.0 / +116.8 (+0.46) | +19.1 / +15.8 (+0.058) | +1.9 / +1.8 (+0.0059) | 0.00306 |
| uni3d_cut_fd | +20.2 / +21.3 | +7.7 / +6.7 | +49.4 / +54.6 | +25.7 / +31.2 | +16.4 / +17.7 | +51.0 / +48.9 | +8.1 / +7.1 | -0.7 / -0.8 | 0.00396 |
| uni3d_cut_textscore_drop | +0.5 / +1.6 | +1.7 / +2.2 | -2.0 / -0.7 | -2.0 / -2.3 | +4.8 / +3.8 | +3.3 / +2.8 | -0.1 / -0.6 | -1.1 / -0.2 | 0.242 |
| ulip2_all_kid | +4.0 / +3.2 (+0.016) | +2.1 / +1.9 (+0.0081) | +70.2 / +63.0 (+0.28) | +49.1 / +48.1 (+0.19) | +40.7 / +38.4 (+0.16) | +96.2 / +84.3 (+0.38) | +33.6 / +27.0 (+0.13) | +0.1 / +0.1 (+0.00026) | 0.00392 |
| ulip2_all_fd | +3.6 / +3.7 | +1.8 / +2.2 | +31.2 / +36.0 | +26.1 / +31.4 | +16.4 / +19.6 | +40.2 / +46.5 | +13.2 / +13.8 | +0.1 / -0.1 | 0.0057 |
| ulip2_all_textscore_drop | +2.6 / +2.0 | +1.3 / +0.7 | +1.2 / +0.2 | +7.0 / +4.8 | +0.7 / +0.9 | +3.3 / +2.8 | -2.9 / -2.8 | +0.2 / -0.4 | 0.281 |
| ulip2_vis_kid | -0.1 / +0.1 (-0.00047) | -0.1 / +0.0 (-0.00063) | +62.1 / +60.8 (+0.26) | +41.0 / +45.0 (+0.17) | +50.1 / +52.4 (+0.21) | +100.9 / +97.1 (+0.43) | +30.6 / +26.6 (+0.13) | -0.1 / +0.0 (-0.0005) | 0.00426 |
| ulip2_vis_fd | -0.1 / +0.2 | -0.2 / +0.1 | +28.3 / +28.3 | +22.4 / +24.1 | +20.8 / +21.7 | +42.2 / +42.4 | +12.2 / +11.4 | -0.0 / -0.1 | 0.00594 |
| ulip2_vis_textscore_drop | -0.1 / +0.1 | +0.0 / +0.3 | -1.0 / -1.0 | +3.3 / +2.2 | -2.4 / -1.6 | +0.3 / +0.0 | -1.6 / -1.6 | +0.2 / +0.3 | 0.379 |
| ulip2_cut_kid | +9.8 / +6.4 (+0.041) | +2.2 / +1.3 (+0.0093) | +52.5 / +41.8 (+0.22) | +34.3 / +30.8 (+0.15) | +29.4 / +23.2 (+0.12) | +85.9 / +64.1 (+0.36) | +22.9 / +15.0 (+0.097) | +0.1 / -0.0 (+0.00061) | 0.00424 |
| ulip2_cut_fd | +8.0 / +7.8 | +2.7 / +3.0 | +26.4 / +28.2 | +21.9 / +25.1 | +13.7 / +14.9 | +42.9 / +44.1 | +10.4 / +9.2 | +0.0 / +0.1 | 0.0053 |
| ulip2_cut_textscore_drop | -0.1 / +0.2 | +1.6 / +1.3 | -2.1 / -3.3 | +2.7 / +2.4 | +2.8 / +4.1 | +9.4 / +10.0 | -0.4 / -0.5 | -0.3 / -0.7 | 0.264 |
| cd_all_mmd | +6.3 / +8.0 (+0.00057) | +1.1 / +1.7 (+0.0001) | +55.0 / +66.2 (+0.005) | +13.1 / +15.9 (+0.0012) | -0.6 / -0.1 (-5.1e-05) | -2.7 / -2.7 (-0.00024) | -0.0 / +0.2 (-2.5e-06) | -0.1 / +0.9 (-9.3e-06) | 9.07e-05 |
| cd_all_one_minus_cov | +0.0 / +0.6 | -0.6 / -0.2 | +12.1 / +12.7 | -1.2 / +1.0 | -0.6 / -0.4 | +0.6 / +0.6 | -0.2 / -0.2 | -1.0 / -0.2 | 0.0197 |
| cd_all_one_nna | +3.3 / +1.6 | +1.2 / +0.6 | +11.8 / +12.3 | +4.6 / +5.1 | +1.4 / +1.2 | -0.4 / -0.8 | -0.8 / -0.7 | +1.1 / -0.8 | 0.0388 |
| cd_vis_mmd | -0.1 / -0.1 (-9.8e-06) | -0.4 / -0.3 (-3.8e-05) | +64.3 / +75.2 (+0.0067) | +8.0 / +9.5 (+0.00084) | -2.5 / -2.9 (-0.00026) | -5.0 / -5.4 (-0.00052) | -0.2 / -0.1 (-2.3e-05) | -0.3 / +1.3 (-2.7e-05) | 0.000104 |
| cd_vis_one_minus_cov | +0.0 / +0.2 | -0.4 / -0.2 | +14.0 / +15.6 | +0.6 / +1.0 | +0.6 / +1.2 | -1.0 / +2.0 | +0.0 / -0.6 | +0.0 / +0.6 | 0.0205 |
| cd_vis_one_nna | +0.0 / -0.1 | +0.0 / -0.3 | +15.2 / +14.1 | +5.1 / +6.5 | +4.2 / +2.2 | +2.3 / +0.5 | -0.5 / -0.4 | +1.5 / +1.2 | 0.0305 |
| cd_cut_mmd | +7.8 / +6.9 (+0.00059) | +1.8 / +1.4 (+0.00013) | +55.6 / +51.2 (+0.0042) | +14.9 / +14.0 (+0.0011) | -0.4 / -0.7 (-2.8e-05) | -2.6 / -2.0 (-0.0002) | +0.0 / -0.2 (+2e-06) | -0.6 / -0.3 (-4.6e-05) | 7.53e-05 |
| cd_cut_one_minus_cov | +0.0 / -0.2 | -0.7 / -1.1 | +12.7 / +12.2 | -1.6 / +0.2 | -0.7 / +0.0 | -0.4 / +0.0 | -0.4 / -0.2 | +0.0 / -0.4 | 0.0182 |
| cd_cut_one_nna | +5.4 / +4.2 | +2.3 / +0.6 | +14.6 / +12.3 | +6.4 / +6.1 | +1.6 / +1.3 | -0.2 / +0.3 | +0.6 / -0.1 | +0.9 / +0.8 | 0.0319 |
| emd_all_mmd | +4.5 / +3.6 (+0.0035) | +1.0 / -0.0 (+0.00081) | +36.5 / +31.8 (+0.029) | +11.2 / +8.8 (+0.0087) | +0.8 / -0.4 (+0.00063) | +0.2 / -0.0 (+0.00016) | -0.3 / -0.7 (-0.00021) | +0.9 / +0.5 (+0.0007) | 0.000782 |
| emd_all_one_minus_cov | +0.8 / +0.6 | +0.2 / -1.3 | +13.0 / +11.5 | -2.3 / -0.4 | -0.6 / -0.9 | -0.4 / -0.2 | -0.8 / -0.7 | -0.6 / -1.3 | 0.0192 |
| emd_all_one_nna | +4.1 / +1.6 | +0.8 / +0.4 | +11.9 / +11.0 | +2.3 / +2.0 | +0.0 / +0.5 | +0.1 / +0.1 | +0.1 / -0.2 | +0.1 / -0.6 | 0.0359 |
| emd_vis_mmd | +0.0 / +0.6 (+3.8e-05) | +0.0 / +0.1 (+1.4e-05) | +40.7 / +45.9 (+0.034) | +10.5 / +11.1 (+0.0088) | +1.3 / +0.9 (+0.0011) | +0.6 / +0.7 (+0.00053) | -0.0 / +0.1 (-1.4e-05) | +0.6 / +0.6 (+0.00047) | 0.000841 |
| emd_vis_one_minus_cov | +0.0 / -0.2 | +0.0 / +0.7 | +12.4 / +16.6 | -1.1 / +0.7 | +0.2 / +0.5 | -1.7 / -0.5 | -0.8 / +0.9 | -0.2 / +0.5 | 0.0211 |
| emd_vis_one_nna | +0.3 / +0.4 | -0.5 / +1.6 | +12.9 / +12.7 | +2.8 / +3.2 | +1.2 / +2.1 | +1.3 / +1.7 | +0.3 / -0.7 | +1.2 / +2.2 | 0.0357 |
| emd_cut_mmd | +3.6 / +3.9 (+0.0032) | +0.4 / +0.4 (+0.00038) | +27.0 / +31.6 (+0.024) | +8.9 / +9.5 (+0.008) | +0.2 / +0.2 (+0.00019) | -0.3 / -0.0 (-0.00023) | +0.1 / -0.7 (+4.6e-05) | +0.6 / +0.8 (+0.00054) | 0.000891 |
| emd_cut_one_minus_cov | -0.5 / +0.2 | -0.2 / -0.9 | +9.8 / +13.0 | -1.0 / -2.6 | -0.7 / -1.3 | -0.5 / -0.4 | -0.5 / -1.3 | +0.0 / +0.0 | 0.0238 |
| emd_cut_one_nna | +4.5 / +4.0 | +0.2 / +1.7 | +11.3 / +12.8 | +2.8 / +4.3 | +0.2 / +0.5 | -0.7 / +1.7 | -0.5 / -0.4 | -0.1 / +3.2 | 0.0337 |
| cd_vox_volume_mmd | -3.7 / -2.8 (-0.00035) | +3.3 / +2.8 (+0.00031) | +60.9 / +47.8 (+0.0057) | +16.4 / +13.5 (+0.0015) | -0.7 / -0.6 (-7e-05) | -2.2 / -1.1 (-0.00021) | -0.2 / +0.1 (-1.8e-05) | +0.2 / +0.7 (+2.1e-05) | 9.39e-05 |
| cd_vox_volume_one_nna | +1.9 / +0.3 | +3.4 / +1.3 | +16.8 / +11.5 | +8.7 / +6.3 | +0.5 / +0.4 | +1.3 / +0.9 | +0.3 / +0.4 | +2.2 / +0.8 | 0.0302 |
| cd_vox_surface_mmd | +2.2 / +2.9 (+0.00026) | +2.4 / +2.6 (+0.00028) | +49.0 / +49.2 (+0.0058) | +13.2 / +13.7 (+0.0016) | -0.6 / -0.2 (-7.3e-05) | -1.7 / -1.2 (-0.0002) | -0.2 / -0.1 (-2e-05) | +0.1 / +0.4 (+1.7e-05) | 0.000118 |
| cd_vox_surface_one_nna | +2.0 / +1.7 | +1.8 / +1.0 | +13.8 / +12.6 | +6.6 / +7.1 | +0.4 / +0.6 | +0.5 / +1.3 | +0.4 / +0.6 | +1.6 / +1.3 | 0.0362 |

Under vis, every 3D-native row is blind to the interior arms at both seeds (|z| <= 1.0 on Objaverse, <= 1.6 on Minecraft). Under all and cut, Uni3D sees them (Objaverse 4 to 14, Minecraft Fill-Interior 22 to 37) and ULIP-2 sees them more weakly. Chamfer and EMD set statistics stay at or below about 8 even when their points include the interior.

## T82. The n=256 leaderboard of record, 0925 protocol, 77-d V-KID, no LLM arms (2026-11-03)

Run `outputs/run_20261103_090315_bench_n_256_leaderboard_of_record` (`--tier both --n 256
--baselines --descriptor v1 --save-features`; notes.md §69). Protocol pin validated first by
`run_20261102_214725_bench_validation_0925_re-score_under_the_v1`, which reproduces
`run_20260925_002631` exactly (bit-identical geom_kid, fast tier, ladder, self-checks and
head-to-head groups; mv_dino_kid within 7e-7, BlockScore within 3e-4). AR arms regenerated to
256 from the same checkpoints and seed; rows 0-127 are byte-identical to the 0925 files.
Self-validation 6/6. Rank interval = 95% over 1,000 m-out-of-n replicates (m=0.8n, reference
subsample shared per replicate; `rank_intervals.md`).

| arm | geom_kid n128 → n256 | mv_dino_kid n128 → n256 | BlockScore n128 → n256 | rank n128 → n256 | rank interval |
|---|--:|--:|--:|--:|--:|
| real_test | -0.095 → -0.143 | -0.002 → 0.001 | 0.00 → 0.00 | 1 → 1 | 1-1 |
| dreamcubed (ext, new) | — → 7.67 | — → 0.268 | — → **165.71** (appearance) | — → 2 | 2-2 |
| native_oriented (contaminated) | 11.62 → 12.08 | 0.200 → 0.208 | 106.87 → 229.29 | 2 → 3 | 3-3 |
| canon_oriented | 14.90 → 15.06 | 0.245 → 0.254 | 136.73 → 285.22 | 3 → 4 | 4-4 |
| canon_unoriented | 19.57 → 18.25 | 0.317 → 0.295 | DQ (+3.0) → 345.10 | DQ → 5 | 5-5 |
| real_shuffled_materials | -0.095 → -0.143 | 0.667 → 0.681 | 231.52 → 422.43 | 7 → 6 | 6-7 (tie) |
| patchwork | 19.98 → 19.48 | 0.709 → 0.696 | 246.15 → 431.70 | 8 → 7 | 6-8 (tie) |
| real@solidify | 22.03 → 23.54 | -0.001 → 0.002 | 201.80 → 444.36 | 6 → 8 | 6-8 (tie) |
| real@monochrome | -0.095 → -0.143 | 0.859 → 0.863 | 297.95 → 535.36 | 9 → 9 | 9-9 |
| train_copy_noise | 33.10 → 32.86 | 0.442 → 0.439 | 302.73 → 619.13 | 11 → 10 | 10-11 (tie) |
| gabled_house | 32.82 → 32.96 | 0.830 → 0.864 | 300.15 → 620.98 | 10 → 11 | 10-11 (tie) |
| shell_box | 34.16 → 34.28 | 1.373 → 1.392 | 476.22 → 864.00 | 12 → 12 | 12-12 |
| uniform_random | 37.73 → 38.63 | 1.740 → 1.752 | 603.09 → 1087.06 | 13 → 13 | 13-13 |
| real@single_mode, train_verbatim | | | DQ → DQ | | |
| dreamcubed_raw (ext, new) | — → 38.68 | — → 0.641 | — → DQ (mode collapse +24.0) | | |
| llm_control, llm_bias | 15.01, 15.87 → dropped | 0.429, 0.552 → dropped | 149.19, 191.88 → dropped | 4, 5 → — | |

**Units double, ranks mostly hold.** The noise sd halves at n=256 (geom_kid 0.110 → 0.053,
mv_dino_kid 0.00289 → 0.00161), so every BlockScore roughly doubles while the raw KIDs barely
move. canon_unoriented's marginal memorization DQ does not survive n=256 (+2.1 spreads).
Dream-Cubed is the best generator on the board; dropping its size term moves its geom_kid
only 7.67 → 7.17, so its geometric gap is shape, not size. Solidify: appearance +0.35,
geometry +444.36 (ratio 1256), still in real's mv_dino_kid group. Patch-Stitch
appearance/geometry 1.17. The lab marks the run `official: false` because `houses32-v1` pins
the 80-d (v2) descriptor.

## T83. BrickGPT (published, ICCV 2025) on the LEGO suite, held-out ST2B test split, n=253 (2026-11-03)

**Run:** `outputs/run_20261102_220228_brickgpt_lego/` (generation, `validity_summary.json`,
`gen_summary.json`, `samples.png`), bench cards in `bench/` (v2, 80-d) and `bench_v1/` (v1,
77-d). Corpus `lego_st2b_test_20` (ST2B test split only, 4,785, no void filter), split
3,349/718/718, reference 512 val, dinov2b 4 ortho views @224. 256 prompts from the bench test
partition; 253 produced a structure (3 hit the 1,800 s cap). Details and deviations in
notes §70.

**Suite (v2 card; separations in in-run noise floors: geom_kid 0.0986, mv_dino_kid 0.00156).**

| arm | geom_kid | sep | mv_dino_kid | sep | BlockScore (worst) |
|---|--:|--:|--:|--:|---|
| real_test (floor) | 0.110 | 0 | 0.001 | 0 | 0.00 |
| real_prompt_matched | −0.049 | −1.6 | −0.002 | −1.6 | 0.40 (palette) |
| real@solidify | 4.106 | **+40.5** | 0.000 | **−0.09** | 40.53 (geometry) |
| **brickgpt** | **5.251** | **+52.1** | **0.019** | **+11.8** | **52.14 (geometry)** |
| real_shuffled_materials | 0.110 | 0 | 0.314 | +201.4 | 201.36 (appearance) |
| real@monochrome | 0.110 | 0 | 1.280 | +821.6 | 4449.55 (palette) |

v1 descriptor: brickgpt +51.5, solidify +36.4 (BlockScore 51.48 / 36.41). BrickGPT coherence:
lcc_ratio 0.980 (real 0.999), sealed interior 0.011 (real 0.034), brick-type JSD 0.009
(palette pillar +40.8). Novelty: NN-IoU 0.589 (real 0.630), not DQ.

**BrickGPT's own metrics, reproduction check.**

| | valid | stable | mean stab. | min stab. | median time |
|---|--:|--:|--:|--:|--:|
| paper, BrickGPT (Gurobi rollback) | 100% | 98.8% | 0.996 | 0.915 | 40.8 s |
| paper, w/o rollback | 100% | 24.0% | 0.947 | 0.228 | n/a |
| **ours (connectivity rollback, bf16)** | **98.8%** | **95.8%** of solved (230/240) | **0.995** | **0.896** | 11.1 s |
| held-out real (same 256) | 100% | 100% (SCIP and shipped Gurobi agree 240/240) | 0.997 | 0.953 | |

Stability solved with SCIP on BrickGPT's own formulation (`scripts/brickgpt_scip_stability.py`,
validated against the dataset's Gurobi scores to 4.6e-7). Ground-connected: 98.8% of valid.
Rollbacks median 1, 1.2% at the 100 cap; 2.02 rejections per accepted brick.

## T84. Nearest-neighbour double dissociation, V-KID vs MV-KID feature spaces (2026-11-03)

Run `outputs/run_20261103_185719_nn_geom_vs_dino` (notes.md §71). Source retrieval among the 256 real builds:

| probe | DINO: NN is own source | V-KID: NN is own source |
|---|--:|--:|
| Fill-Interior | 1.00 | 0.26 (median rank 17.5) |
| Material-Shuffle | 0.26 (rank 7) | 1.00 |
| Monochrome | 0.24 (rank 9) | 1.00 |

Same nearest real house in both spaces: 3.9% for real builds (chance 0.25%).

## T88. The fixed-pooling face cascade on front faces: silhouettes learned, windows not (2026-11-03)

Step 2 of research.md H.7 (notes §65 for the design, §74 for the reading). One bidirectional
transformer draws level f of a face from its pooled level 2f, f = 16, 8, 4, 2, 1 on a 32 x 32
canvas, every cell of a level in parallel (`scripts/facade_cascade.py`,
`blockgen/models/face_cascade.py`, `blockgen/corpus/facade_levels.py`). Trained on the
side faces of the houses_32 train split mirrored along u (14,904 faces), checkpoint
selected on validation NLL of the front faces, scored on the 399 test front faces. Six
runs, 9 to 15 minutes each on the 5070 Ti.

**Teacher-forced NLL, bits per in-box cell on the test front faces** (the block term over
filled cells spread onto all cells; `total` weights levels by cell count, so f=1 dominates):

| run | select step | f16 | f8 | f4 | f2 | f1 | total |
|---|--:|--:|--:|--:|--:|--:|--:|
| 5M, `any` | 6,000 | 11.12 | 6.14 | 4.78 | 3.84 | 2.97 | 3.346 |
| 5M, `majority` (control) | 5,000 | 10.14 | 5.05 | 4.34 | 4.03 | 3.05 | 3.391 |
| 5M, `any`, no parent corruption | 4,000 | 10.15 | 5.49 | 4.48 | 3.76 | 2.95 | 3.275 |
| 5M, `any`, dropout 0.1 | 6,000 | 10.56 | 5.53 | 4.36 | 3.67 | 2.87 | 3.197 |
| 5M, `any`, dropout 0.3 | 10,000 | 10.79 | 5.40 | 4.38 | 3.58 | 2.76 | 3.104 |
| 20M, `any` (batch 48) | 6,000 | 10.08 | 5.38 | 4.36 | 3.63 | 2.85 | 3.170 |
| 5M, dropout 0.3, each level at its own best step | 1k/4k/6k/6k/15k | 8.69 | 5.05 | 4.20 | 3.58 | 2.76 | |

Every run overfits from the coarse end: the f=16 validation NLL is lowest at step 1,000 and
climbs by 4 to 5 bits while the total is still improving (train f=16 reaches 2.2 bits
against 13.6 on validation in the first run). Dropout 0.3 moves the total's best step
from 6,000 to 10,000 and buys 0.24 bits; 20M buys 0.18 at twice the time; the NCM parent
corruption costs 0.07 bits of teacher-forced NLL and does not change the samples below.
`majority` pooling is not comparable level by level (different targets) and completes
worse (components 1.83 from f=4 against 1.29).

**Front faces drawn from the real test boxes, face statistics (n = 399; T75 columns, W1 =
Wasserstein distance of the per-face distribution to real).** The AR rows are the front
faces of those arms' builds, projected.

| arm | jumps | comps | holes | fill | distinct depths | distinct blocks | W1 jumps | W1 comps |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| real test faces | 0.115 | 1.14 | 0.007 | 0.69 | 7.6 | 12.6 | | |
| cascade 5M, 1 pass per level, T=1.0 | 0.211 | 1.27 | 0.015 | 0.81 | 10.2 | 38.4 | 0.096 | 0.130 |
| cascade 5M, 4 passes per level, T=1.0 | 0.114 | 1.15 | 0.004 | 0.81 | 7.7 | 16.4 | 0.011 | 0.025 |
| cascade 5M, 1 pass, T=0.7 | 0.105 | 1.13 | 0.004 | 0.84 | 7.1 | 13.2 | 0.011 | 0.005 |
| cascade 5M d0.3, 4 passes, T=1.0 | 0.068 | 1.14 | 0.004 | 0.77 | 5.6 | 10.5 | 0.047 | 0.015 |
| cascade 20M, 4 passes, T=1.0 | 0.090 | 1.07 | 0.003 | 0.86 | 6.5 | 12.2 | 0.025 | 0.063 |
| per-face 5M AR (T66) | 0.341 | 1.03 | 0.020 | 0.96 | 12.2 | 21.8 | 0.227 | 0.110 |
| joint 20M D4 AR (T68) | 0.308 | 1.34 | 0.023 | 0.89 | 11.7 | 10.7 | 0.193 | 0.206 |
| joint 20M D4 RFT, T=0.7 (T73) | 0.143 | 1.14 | 0.005 | 0.93 | 6.4 | 5.6 | 0.041 | 0.103 |
| patchwork | 0.275 | 1.04 | 0.007 | 0.96 | 11.1 | 31.5 | 0.160 | 0.100 |

No cascade arm empties a face at any setting. Four passes per level (MaskGIT-style
refinement) take the jump rate from 0.21 to 0.11 at the same fill, which is what
temperature 0.7 did for the AR arm at the price of 13/64 empty builds (§60); here fill
does not move. By these statistics the cascade is the closest arm to real. By eye
(`samples.png`, `samples_s4.png` in each run dir) it is not: the faces are one or two
materials in blobs with a ground strip, no storeys, no window rhythm, no door. The face
statistics measure smoothness and connectedness, and a blob has both; this is the first
arm that matches them without looking like a house, so they are necessary, not sufficient.

**Oracle-coarse completion** (the real level given, the finer levels drawn; `mode` is the
cell-wise argmax, `sample` is temperature 1.0; `upsample` is the T75 copy). 5M dropout 0.3:

| from | mask IoU | depth exact | depth MAE | block agree | jumps |
|---|--:|--:|--:|--:|--:|
| f=4, upsample (copy) | 0.761 | 0.520 | 1.41 | 0.556 | 0.051 |
| f=4, mode | 0.754 | 0.581 | 1.05 | 0.515 | 0.052 |
| f=4, sample | 0.732 | 0.488 | 1.22 | 0.367 | 0.154 |
| f=2, upsample (copy) | 0.858 | 0.737 | 0.62 | 0.696 | 0.090 |
| f=2, mode | 0.853 | 0.783 | 0.47 | 0.659 | 0.083 |
| f=2, sample | 0.837 | 0.693 | 0.67 | 0.523 | 0.147 |

The mode beats the copy on depth (exactness 0.58 against 0.52 from f=4, mean error 1.05
against 1.41 blocks) and `completion.png` shows why: gables, ridges and the staircase of a
pitched roof come back sharp from the 4 x 4 blocks. It loses to the copy on blocks (0.515
against 0.556): the model's best guess for a cell's material is its parent's, and when it
departs from that it is more often wrong than right. Windows and doors, which T75 placed
at f <= 2, never appear in any completion or sample. Sampling at temperature 1.0 is worse
than the copy on every column: the fine levels' spread is speckle, not trim.

**Reading.** The depth residual of the hierarchy is learnable with 7.5k faces and the
block residual is not, which is the within-level-independence prediction of H.3 and §65
with the split made explicit: a roof line is a per-cell function of the parent's
silhouette; a window is a joint pattern over a neighbourhood that independent cells
cannot agree on, so the sampler either copies the parent (mode) or speckles (sample). The
coarse levels memorise the train boxes within a thousand steps, so what the cascade
offers from the box is a material and a blob. The pre-registered gate (beat the copy on
depth and block from f=4) is half met, and the learned residual tokenizer (step 3) would
not change what fails: a VQ code at f=2 is still drawn independently of its neighbours.
Next in notes §74.

Run dirs: `outputs/run_20261103_150008_facade_cascade_5m`, `_150944_..._5m_majority`,
`_151845_..._5m_nocorrupt`, `_185205_facade_cascade_20m`, `_190741_..._5m_d01`,
`_191823_..._5m_d03`; chain log `outputs/logs/facade_cascade_morning.log` (the 20M at
batch 96 ran out of memory there and was rerun at batch 48).

## T85. Paper regeneration checks on the n=256 board (2026-11-04)

notes.md §72. Derived numbers produced while porting the CVPR paper to the n=256 leaderboard.

| quantity | value | source |
|---|--:|---|
| Dream-Cubed raw V-KID, full / no size column / size-matched reference | 7.67 / 7.17 / 8.57 | `outputs/figstage/dreamcubed_size.json` |
| size share of Dream-Cubed's V-KID, upper bound | 6.5% | same |
| canon_oriented, native_oriented without size column | 15.06 -> 15.31, 12.08 -> 12.35 | same |
| VLM BT Spearman vs n=256 MV-KID / BlockScore / V-KID | 0.74 / 0.54 / 0.20 | `outputs/figstage/vlm_vs_leaderboard.json` |
| contiguous-draw floor inflation (re-derived) | 10.88x (sd 0.0320 vs 0.00294) | `outputs/figstage/contiguous_draw_check.json` |
| 25%-bad arm, fraction of fully bad (render) | 0.060 | 0925 dose run |
| V-KID NN hubs for Fill-Interior | 68 distinct NNs / 256; top 4 take 69.9% | `outputs/figstage/nn_dissociation.json` |

Audit: 679 registered claims, all match; 0 stale; main text 8 pages.

## T86. Dream-Cubed audit: repeated vanilla village pieces, and a template-only control that ranks 3rd (2026-11-04)

`outputs/analysis/dreamcubed_audit/` (README.md, CPU only, bench files unchanged). Arms: the
284 extracted Dream-Cubed builds of T79/T82, 399 real test builds, and the 1.21.1 jar village
templates of T56 (299 after removing terrain the same way the extraction does). Dream-Cubed
trained on 1.12.2 worlds, whose villages come from code and not from these template files,
so the template IoU is a stand-in reference and a lower bound on copying. The within-set
test needs no reference.

| measure | Dream-Cubed (284) | real test |
|---|--:|--:|
| median best IoU to a village template (8 symmetries, all translations) | 0.52 | 0.32 |
| builds with template IoU >= 0.5 / >= 0.8 | 53% / 4.2% | 5% / 0% |
| median IoU to the nearest other build in the same set | 0.848 | 0.426 |
| builds with a near-twin, IoU >= 0.8 / >= 0.9 | 59% / 34.5% | 0.8% / 0% |
| clusters at IoU 0.8 (largest) | 160 (28) | 255 of 256 (2) |
| houses with a room / roofless enclosures / fragments | 45.4% / 35.9% / 18.7% | 99.2% houses |

At matched block count (98 builds each) the near-twin rate is 73% against 0%, so the
repetition is not a size effect. The largest cluster is 28 copies of one 117-block hut.

The template-control arm passes the vanilla village templates through Dream-Cubed's own
extraction and the fast tier of the T82 protocol, which reproduces Dream-Cubed's T82 values
exactly (geom_kid 7.672, geometry pillar 146.64):

| arm | n | geom_kid | geometry pillar | mv_dino_kid (CPU render) |
|---|--:|--:|--:|--:|
| Dream-Cubed | 256 | 7.67 | 146.6 | 0.260 |
| vanilla village templates, all | 183 | 10.25 | 195.0 | 0.165 |
| vanilla village templates, houses only | 174 | 10.67 | 203 | 0.168 |
| native_oriented / canon_oriented (T82) | 256 | 12.08 / 15.06 | 229 / 287 | 0.208 / 0.254 |

Pillars at n=174 and 183 use the n=256 noise floor and are approximate.

**Dream-Cubed's rank 2 measures how close vanilla village geometry sits to houses_32, not house
generation.** An unlearned set of vanilla templates scores about 195 to 203, ranks 3rd of 13 and
beats every BlockGen model. Both template arms and Dream-Cubed pass the duplicate gate, which
only compares against the houses_32 train split (0.004), and the diversity gate, which is a
within-set mean (0.866 against 0.898). A within-set near-duplicate gate would catch the 59%
near-twin rate. Proposing it is a change to the frozen bench and waits on authorisation. The
CVPR session (blockgen-03) has the numbers and holds the Dream-Cubed ranking claims until it
decides how to frame them.

## T87. 20M voxel SFT, decayed short schedule and pretraining mix: the mix is the first arm still improving at the end (2026-11-04)

Card `run_20261104_033716_bench_voxel_20M_SFT_short_and_mixed_against` (`--tier both --n 256
--baselines`, self-validation 6/6). Both runs start from the T78 pretrain checkpoint
(`outputs/run_20261102_142710_m2_20m_pretrain_s1/last.pt`), seed 1, dropout 0, select=best,
`houses32-sft-20m.yaml` with overrides:

- `outputs/run_20261104_004730_m2b_20m_sft3k_s1`: `steps=3000`, so the cosine schedule decays
  inside the window where T78 was still improving.
- `outputs/run_20261104_020102_m2b_20m_sftmix_s1`: `steps=6000 train=sft-houses32:1,anneal:1`,
  which gives the same SFT tokens as the 3k run with the pretraining tier mixed in.

Recipe: the scratchpad copy `sft20m_short.sh` (log `outputs/logs/sft20m_short.log`).

| arm | BlockScore | appearance | geom_kid | mv_dino_kid | holdout NLL, best (step) | all-holdout NLL | components | enterable | nn pct |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **20M SFT + anneal mix, 6k** | **332** | 238 | **15.9** | 0.384 | **0.263 (6,000, last)** | **0.280** | 10.4 | 0.28 | 0.66 |
| 20M SFT, 3k decayed | 338 | 251 | 16.2 | 0.405 | 0.286 (1,750) | 0.304 | 8.4 | 0.29 | 0.62 |
| 5M best-of-4 lcc26 (T71) | 340 | 213 | 16.3 | 0.344 | | | 4.9 | 0.21 | 0.61 |
| 20M SFT, 16k (T78) | 346 | 215 | 16.6 | 0.347 | 0.286 (1,500) | 0.305 | 8.6 | 0.26 | 0.58 |
| 5M raw (T60) | 386 | 254 | 18.6 | 0.410 | | | 11.3 | 0.20 | 0.64 |

Holm groups on `geom_kid`: mix `c`, 3k `cd`, 5M best-of-4 `cd`, 20M 16k `cde`, 5M raw `de`.
Only mix against 5M raw separates (2.61 [1.88, 3.40], p_adj 0.043). On `mv_dino_kid` the order
reverses: 5M best-of-4 `c`, 20M 16k `cd`, mix `cde`, 3k `de`, and 3k is worse than 5M
best-of-4 (0.061 [0.040, 0.082], p_adj 0.036).

**Shortening the schedule does not help; mixing the pretraining tier back in does.** The
decayed 3k run reaches the same best holdout NLL as the 16k run (0.286 at step 1,750 against
step 1,500), so T78's early peak was not caused by the learning rate still being near its peak. The
mix run holds the holdout curve down instead: it is at its lowest at the last step (0.263,
still falling 0.002 per 250 steps), its all-holdout NLL is 0.280 against 0.305, and it is the
first voxel arm whose `geom_kid` separates from 5M raw after Holm. It also memorizes the least
(nn pct 0.66, diversity 0.908). The cost is appearance: both new arms lose `mv_dino_kid`
against the 16k 20M model, so the board gain (332 against 346) is inside the `geom_kid` ties
among 20M arms, and geometry stays the worst pillar. Two caveats carry over: one seed, and the
`anneal` tier contains 1,834 of the 1,863 SFT houses (deliverables/reports/worldcrop-pretraining.md),
so the mix is partly more passes over the SFT set. Next: run the mix longer, since it had not
converged, and run a world-crop-only mix to separate the two effects.

## T89. LEGO in-domain ladder, post-fix renderer rerun (2026-11-04)

`outputs/run_20261104_212706_bench_lego_ladder_postfix` reproduces the 09-18 LEGO ladder (fig:seven, tab:seven)
to a maximum relative difference of 5e-6 on every render statistic and rung. The geometry ladder is identical
and the gate verdicts are unchanged. All seven render statistics stay at or below 2.1 noise floors on
Fill-Interior. Wool is opaque, so the 09-25 transparency cull cannot affect LEGO renders. Details are in notes §75.

## T90. MV-KID with facade and cross-section views (2026-11-04)

`outputs/run_20261104_213620_view_facade_section`, n=256, leaderboard-of-record builds and
reference. Units: noise floors above real_test. Details are in notes §76.

| protocol | Fill-Interior | Hollow-Shell | Shuffle | Monochrome | rot90 | Spearman vs corner / V-KID |
|---|--:|--:|--:|--:|--:|--:|
| corner (paper) | +0.29 | +0.03 | +346 | +439 | -0.24 | 1 / 0.27 |
| A-sides (4 elevations) | +4.45 | +0.82 | +548 | +1146 | +0.04 | 0.97 / 0.10 |
| A-render (+ roof plan) | +3.62 | +0.60 | +610 | +1309 | +0.08 | 0.97 / 0.10 |
| A-2.5D facade images | 0 (bit-identical) | 0 (bit-identical) | +355 | +314 | +0.10 | 0.96 / 0.28 |
| B-mid (3 mid sections) | +71.1 | +39.7 | +226 | +1080 | +0.41 | 0.93 / 0.07 |
| B-9 (9 sections) | +67.5 | +42.8 | +292 | +1314 | +0.49 | 0.93 / 0.07 |
| B-cut (half removed) | +14.3 | +5.3 | +415 | +925 | -0.25 | 0.95 / 0.05 |
| V-KID (77-d) | +444.4 | +265.4 | 0 | 0 | 0 | 0.27 / 1 |

## T91. Within-set near-twin gate; Dream-Cubed and the vanilla templates are disqualified (2026-11-04)

The gate (notes §78) counts the share of builds with a near-copy in the same set: D4 and
translation occupancy IoU >= 0.8. It uses the duplicate gate's rule: more than 3 spreads
above real, with the spread floored at 0.02. At n=256 the bar is 0.068.

| arm (n=256 unless noted) | twin rate | gate |
|---|--:|---|
| held-out real | 0.008 | pass |
| real@solidify | 0.066 | pass (z 2.9) |
| AR models (3 arms) | 0.000 | pass |
| gabled_house / shell_box | 0.207 / 0.246 | DQ +10.0 / +11.9 |
| Dream-Cubed | 0.598 | DQ +29.5 |
| vanilla village templates (n=174 / 183) | 0.672 / 0.683 | DQ +33.2 / +33.8 |

The real null over fresh draws has means of 0.006 to 0.019 and a maximum of 0.031.
Ungated, the templates score BlockScore 202.95 / 194.98 on the GPU path, ahead of every
BlockGen model. Runs:
- `outputs/run_20261104_214146_bench_twin_gate_rescore` (re-scored leaderboard, rank
  intervals);
- `outputs/run_20261104_213125_bench_vanilla_template_control`.


## T92. V-KID against structural alternatives: probes, gates, ordering (2026-11-05)

Run `outputs/run_20261104_235615_bench_headtohead`; notes §80. Values are noise floors above
held-out real. The ladder uses n=128 (houses_32 ladder subset, `ladder.score_rungs`); the board
uses the n=256 leaderboard-of-record arms, with build hashes checked.

| row | Fill | Hollow | Shuffle | Mono | gates | Fill @256 | rho V-KID | rho MV-KID |
|---|--:|--:|--:|--:|---|--:|--:|--:|
| V-KID | +111.4 | +71.0 | 0 | 0 | PASS | +444.4 | 1.00 | 0.27 |
| occupancy KID | +48.4 | +13.1 | 0 | 0 | G2, G6 | +121.2 | 0.83 | 0.23 |
| Uni3D KID all-face | +26.6 | +10.4 | +22.4 | +99.7 | G6 | +31.8 | 0.51 | 0.85 |
| Uni3D KID volume | +37.2 | +10.6 | +6.7 | +105.5 | G6 | +108.4 | 0.59 | 0.78 |
| ULIP-2 KID all-face | +4.6 | +1.8 | +29.1 | +101.8 | G6/G2 | +12.0 | 0.42 | 0.87 |
| ULIP-2 KID volume | +11.9 | +9.0 | +9.7 | +62.8 | G6/G2 | +43.2 | 0.51 | 0.85 |
| chamfer MMD all-face | +7.6 | +1.4 | -0.1 | -0.1 | G2, G8 | +8.9 | 0.64 | 0.27 |
| EMD MMD all-face | +3.2 | +0.3 | -0.2 | -0.3 | G2, G8 | n/a | | |
| chamfer MMD voxel centres | -3.4 | +2.8 | +0.1 | +0.1 | G2, G8 | -3.5 | 0.53 | 0.54 |
| IoU MMD D4-aligned | -9.0 | +7.7 | 0 | 0 | G8 | -5.3 | 0.36 | 0.35 |
| IoU 1-NNA D4-aligned | +2.7 | +2.4 | 0 | 0 | G2 | +5.9 | 0.75 | 0.28 |

V-KID is the only row that passes every gate. The voxel-set metrics rate Fill-Interior closer to real
than real is.

## T93. canon_oriented on the current card: the legacy 3D-BPE model still beats every new-pipeline voxel arm (2026-11-05)

Card `run_20261104_235313_bench_canon_oriented_against_the_new_pipeline` (`--tier both --n 256
--baselines`). canon_oriented is the legacy native_bpe model (3D-BPE, 256 merges, oriented,
5.7M params, 60 epochs on houses only), scored from its T32/T82 samples
`outputs/bench_arms/canon_oriented_n256_32.npz`; the other arms are the T60, T71, T78 and T87
caches. Recipe: scratchpad copy `card_canon.sh` (log `outputs/logs/card_canon.log`).

| arm | BlockScore | appearance | geom_kid | mv_dino_kid | components | enterable | nn pct |
|---|--:|--:|--:|--:|--:|--:|--:|
| **canon_oriented (legacy 3D-BPE)** | **297** | **157** | **14.2** | **0.254** | **7.9** | **0.61** | 0.43 |
| 20M SFT + anneal mix (T87) | 332 | 238 | 15.9 | 0.384 | 10.4 | 0.28 | 0.66 |
| 5M best-of-4 lcc26 (T71) | 340 | 213 | 16.3 | 0.344 | | | |
| 20M best-of-4 lcc26 (T78) | 341 | 178 | 16.3 | 0.288 | | | |
| 20M SFT, 16k (T78) | 346 | 215 | 16.6 | 0.347 | | | |
| 5M raw (T60) | 386 | 254 | 18.6 | 0.410 | | | |

Holm groups on `geom_kid`: canon_oriented `c`, mix `c`, both best-of-4 arms `cd`, 20M 16k `cde`,
5M raw `de`; canon_oriented separates only from 5M raw (-4.31 [-5.21, -3.20], p_adj 0.014). On
`mv_dino_kid` canon_oriented `c` shares a group only with the 20M best-of-4 and separates from
every other model arm (against the mix -0.130, p_adj 0.014).

**The best model is still the oldest one, and the pipeline has not yet closed the gap.** The
lead is largest where nothing was tuned: appearance (157 against 178 or worse), enterable 0.61
against 0.28, and floating components. Geometry ties the mix. Two caveats: canon_oriented has
the lowest nn percentile of the model arms (0.43, under the real 0.51, so it sits a little
closer to training data than held-out real does; no gate fires), and it differs from the
pipeline arms in three ways at once (tokenizer, orientation, training data and schedule). The
5M oriented against unoriented world-crop chain (`oriented_5m.sh`, started 01:30 UTC) isolates
orientation; tokenizer hashes unoriented 60684456e598, oriented 0e6ace66bcdf.

## T94. 5M world-crop-only pretraining, oriented against unoriented: orientation is not the lever (2026-11-05)

Card `run_20261105_050900_bench_5M_world-crop_pretraining_oriented` (`--tier both --n 256
--baselines`). Both arms are 5M ar2, seed 1, pretrained 20k steps on the new `anneal-worldcrop`
tier (270,166 world crops, SFT houses and holdout excluded), then SFT 12k steps on
`sft-houses32:1,anneal-worldcrop:1` from `last.pt`, then best-of-4 lcc26. They differ only in
the tokenizer: unoriented 60684456e598 (vocab 507), oriented 0e6ace66bcdf (vocab 926; its anneal-worldcrop shard was deleted 2026-11-05 for disk space, rebuild with `shards build anneal-worldcrop --oriented`; same 1,863
SFT houses, split into 2,114 rows because oriented sequences are longer). Holdout NLL is per
token, so it does not compare across the two tokenizers. Recipe: scratchpad `oriented_5m.sh`
(log `outputs/logs/oriented_5m.log`).

| arm | BlockScore | appearance | geom_kid | mv_dino_kid |
|---|--:|--:|--:|--:|
| canon_oriented (T93) | 297 | 157 | 14.2 | 0.254 |
| 20M SFT + anneal mix (T87) | 332 | 238 | 15.9 | 0.384 |
| 5M best-of-4 lcc26 (T71) | 340 | 213 | 16.3 | 0.344 |
| world-crop 5M oriented, best-of-4 | 351 | 228 | 16.8 | 0.368 |
| world-crop 5M unoriented, best-of-4 | 358 | 192 | 17.2 | 0.310 |
| world-crop 5M unoriented | 361 | 238 | 17.3 | 0.384 |
| world-crop 5M oriented | 373 | 277 | 17.9 | 0.447 |
| 5M raw (T60, `anneal` pretraining) | 386 | 254 | 18.6 | 0.410 |

Holm groups on `geom_kid`: all four world-crop arms share `de` (oriented best-of-4 `cde`), so
orientation does not move geometry. On `mv_dino_kid` the unoriented arms are better and
separated: best-of-4 0.310 against 0.368 (p_adj 0.030), and unoriented best-of-4 against oriented raw
(p_adj 0.017). Oriented arms also raise the within-set twin rate a little (0.023 and 0.035
against real 0.008, no gate fires).

**Orientation does not explain canon_oriented's lead.** Inside the pipeline it costs appearance
and leaves geometry tied, so the 297 against roughly 350 gap must come from the rest of the
legacy recipe: 3D-BPE tokens and 60 epochs on houses only. World-crop-only pretraining
does help a little over the old `anneal` pretraining at 5M (raw 361 against 386, same Holm group on
`geom_kid`), and best-of-4 again buys appearance, not geometry. One seed. Next: put 3D-BPE through the pipeline
(a native_bpe tokenizer on the world-crop tier), and the long 20M world-crop mix.

## T95. Inspection checks on the T51 agentic arms and real houses (2026-11-05)

`blockgen/agentic/inspection.py` run on the n=133 builds of the T51 bench arms and on 400
real `houses_32` builds (seed 1). Failure rate per check at warning severity or worse;
`openings`, `grounded`, `built` and `size` are 0% everywhere and omitted. Notes §81.

| arm | n | mean score | clean at warning | connected | supported | interior | roofed | enterable | headroom |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| classic-matched | 133 | 0.952 | 68% | 0% | 0% | 0% | 6% | 33% | 1% |
| dream | 133 | 0.942 | 53% | 0% | 0% | 0% | 2% | 54% | 7% |
| dream-caption | 133 | 0.933 | 48% | 0% | 0% | 2% | 0% | 57% | 13% |
| real houses | 400 | 0.934 | 55% | 6% | 14% | 0% | 10% | 17% | 9% |

Reading: the agentic arms are structurally sound and not enterable; real houses are the
reverse (their failures are crop artefacts: missing floors and detached yard pieces). The
`inspect` method's job is the enterable column. Measurement only; no new builds.

## T96. 5M world-crop pretraining with 3D-BPE pieces against voxels: pieces lose geometry (2026-11-05)

Card `run_20261105_132412_bench_5M_world-crop_pretraining_3D-BPE_pieces` (`--tier both --n 256
--baselines`). Arm p repeats the T94 unoriented recipe exactly (5M ar2, seed 1, pretrain 20k
steps on `anneal-worldcrop`, SFT 12k steps on `sft-houses32:1,anneal-worldcrop:1` from
`last.pt`, best-of-4 lcc26) with the pipeline 3D-BPE tokenizer `pieces` cdce56584ebf (256 merges
learned on sft-houses32, unoriented, vocab 811) in place of voxels 60684456e598. Runs
`run_20261105_101818_wc5m_pretrain_p_s1` and `run_20261105_112220_wc5m_sftmix_p_s1`
(select=best chose the last step, holdout 0.637 per token, still falling; per-token NLL does not
compare across tokenizers). Recipe: scratchpad `pieces_5m.sh` (log `outputs/logs/pieces_5m.log`).

| arm | BlockScore | appearance | geom_kid | mv_dino_kid | enterable | dino nn pct |
|---|--:|--:|--:|--:|--:|--:|
| canon_oriented (T93) | 297 | 157 | 14.2 | 0.254 | 0.61 | 0.43 |
| 20M SFT + anneal mix (T87) | 332 | 238 | 15.9 | 0.384 | 0.28 | 0.66 |
| world-crop 5M voxels, best-of-4 (T94) | 358 | 192 | 17.2 | 0.310 | 0.15 | 0.62 |
| world-crop 5M voxels (T94) | 361 | 238 | 17.3 | 0.384 | 0.15 | 0.68 |
| world-crop 5M pieces, best-of-4 | 491 | 244 | 23.6 | 0.394 | 0.37 | 0.47 |
| world-crop 5M pieces | 501 | 260 | 24.1 | 0.419 | 0.31 | 0.53 |

Paired Holm tests: voxels beat pieces on `geom_kid` by 6.3 to 6.9 in every raw/best-of-4 pairing,
and on `mv_dino_kid` for both best-of-4 comparisons (all p_adj 0.014, the floor of the
permutation test). Pieces are better on traversal (enterable 0.31 against 0.15) and further from
the training set (nn percentile 0.53 against 0.68), the two places canon_oriented also leads.

**3D-BPE alone does not reproduce canon_oriented.** Under the pipeline schedule (world-crop
pretraining, a 1:1 mix, 12k SFT steps) pieces lose geometry by a wide margin, so the 297 must
come from the legacy schedule (60 epochs on houses only, D4 augmentation, dropout 0.1) or from
that schedule interacting with pieces. Two caveats: the pipeline tokenizer is not byte-identical
to the legacy one (unoriented, first-wins id fix, merges learned on sft-houses32), and the
pieces SFT was still improving at its last step. The D4 pair (`pieces_d4.sh`: pieces from
scratch with D4 at 60k steps, and the same from the pieces pretrain) is the direct test. One seed.

## T97. 20M SFT + world-crop mix run to 16k steps: no gain over 6k (2026-11-05)

Card `run_20261105_154732_bench_20M_SFT_world-crop_mix_16k` (`--tier both --n 256 --baselines`).
`run_20261105_091034_m2b_20m_sftwc16k_s1` continues the T87 question (holdout still falling at
6k): 20M ar2, seed 1, from the 20M pretrain, `train=sft-houses32:1,anneal-worldcrop:1`, 16k steps,
voxels 60684456e598. select=best chose step 8000 (holdout 0.2667 against 0.2832 at 16k; the T87
6k mix reached 0.263 on the older `anneal` tier). Its end-of-run draw OOMed beside another
session's job, so the 256 samples were redrawn from `best.pt` (scratchpad
`sft20m_long_resample.sh`).

| arm | BlockScore | appearance | geom_kid | mv_dino_kid | enterable | dino nn pct |
|---|--:|--:|--:|--:|--:|--:|
| canon_oriented (T93) | 297 | 157 | 14.2 | 0.254 | 0.61 | 0.43 |
| 20M SFT + anneal mix, 6k (T87) | 332 | 238 | 15.9 | 0.384 | 0.28 | 0.66 |
| 20M SFT + world-crop mix, 16k (best step 8000) | 333 | 239 | 15.9 | 0.386 | 0.15 | 0.66 |
| world-crop 5M voxels, best-of-4 (T94) | 358 | 192 | 17.2 | 0.310 | 0.15 | 0.62 |

Holm groups on `geom_kid`: canon_oriented `c`, both 20M mixes `cd`, the 5M world-crop arms `de`.
The 16k arm is not separated from the 6k mix on either primary.

**Longer SFT saturates.** The holdout NLL kept falling only to step 8000 and the bench does not
move at all (333 against 332), so the 20M voxel recipe sits at about 332 however long it trains.
Enterable drops (0.15 against 0.28), which matches the world-crop tier, not the step count (the 5M
world-crop arms also read 0.15). The remaining gap to canon_oriented is in the schedule, which
the D4 pieces pair tests. One seed.

## T98. 3D-BPE with D4 at 60k steps, scratch against world-crop pretraining: traversal and novelty reproduce, geometry does not (2026-11-05)

Card `run_20261105_181623_bench_3D-BPE_with_D4_scratch_against` (`--tier both --n 256 --baselines`).
Both arms use the pipeline pieces tokenizer cdce56584ebf on the D4 shard `sft-houses32/cdce56584ebf-d4`
(14,868 sequences), 5M ar2 (d_model 256, 6 layers), seed 1, 60k steps of 32,768 tokens:
- `run_20261105_154811_p5m_d4scratch_s1`: from scratch, dropout 0.1 (`houses32-scratch-matched.yaml`),
  the legacy canon_oriented schedule. select=best step 51,000 (holdout 0.544).
- `run_20261105_172651_p5m_d4pre_s1`: from the T96 pieces pretrain `last.pt`, dropout 0
  (`houses32-sft-5m.yaml`). select=best step 22,000 (holdout 0.585), so it overfits early.

Sampling matches canon_oriented (T=1.0, top_k 40). Recipe: scratchpad `pieces_d4.sh` (log
`outputs/logs/pieces_d4.log`).

| arm | BlockScore | appearance | geom_kid | mv_dino_kid | enterable | dino nn pct |
|---|--:|--:|--:|--:|--:|--:|
| canon_oriented (T93) | 297 | 157 | 14.2 | 0.254 | 0.61 | 0.43 |
| world-crop 5M voxels (T94) | 361 | 238 | 17.3 | 0.384 | 0.15 | 0.68 |
| pieces D4 scratch, 60k | 425 | 209 | 20.4 | 0.338 | 0.60 | 0.48 |
| pieces D4 from pieces pretrain | 476 | 208 | 22.9 | 0.336 | 0.27 | 0.47 |
| world-crop 5M pieces (T96) | 501 | 260 | 24.1 | 0.419 | 0.31 | 0.53 |

Holm groups on `geom_kid`: canon_oriented `c`, voxels `d`, D4 scratch `ef`, D4 pretrain `fg`,
world-crop pieces `g`. On `mv_dino_kid`: canon_oriented `c`, both D4 arms `d`, voxels `de`.

**The legacy schedule reproduces canon_oriented's traversal and novelty but not its geometry.**
D4 scratch matches enterable (0.60 against 0.61) and nearly matches novelty, and it is the best
pieces arm (425 against 501), but its `geom_kid` (20.4) is still worse than plain world-crop voxels
(17.3) and far from canon_oriented (14.2). Pretraining on world crops hurts pieces here (476, and
enterable falls back to 0.27), the opposite of voxels. What still differs from canon_oriented:
its tokenizer is oriented (627 block states, 1,079 pieces after 256 merges against 811 here), it
was trained with batches of 8 sequences (about 104k smaller steps against 60k steps of 32k tokens),
and its merges were learned on its own D4 training set. One seed.

## T99. First category-conditioned models: scratch follows the class token, world-crop pretraining weakens it (2026-11-05)

Two 5M ar2 voxel models on the category tokenizer voxels-cat 116e68927d45 (the class token takes
BOS's slot), trained on `sft-cat` (11 classes plus `none`, notes on the data in
`deliverables/reports/category-conditioning.md`), seed 1:
- `run_20261105_181801_cat5m_scratch_s1` (`cat-sft-5m-scratch.yaml`, 36k steps): select=best step 17,000
  (holdout 0.551).
- `run_20261105_185826_cat5m_pt_s1` (`cat-sft-5m-pt.yaml`, 16k steps) from the T94 unoriented
  world-crop pretrain `last.pt`, vocab extended by copying BOS: select=best step 16,000 (holdout 0.527).

Conditioned eval (`python -m blockgen.eval.conditioned`, 128 samples per class, sizes drawn from the
class, against `eval-cat`). Probe ceiling is the probe's accuracy on the real eval builds of the class.
The mean leaves out church, which folded into `none` but keeps its vocab token and has no eval builds
(fixed in `conditioned.py` 2026-11-05; the chain's first printout averaged it in as 0).

| class | eval builds | ceiling | scratch probe | pretrain probe | scratch geom_kid | pretrain geom_kid |
|---|--:|--:|--:|--:|--:|--:|
| tree | 41 | 0.95 | 0.78 | 0.14 | 4.2 | 10.7 |
| vehicle | 112 | 0.74 | 0.58 | 0.52 | 4.3 | 5.0 |
| pixelart | 35 | 1.00 | 0.56 | 0.27 | 8.0 | 17.6 |
| aircraft | 41 | 0.63 | 0.39 | 0.12 | 5.1 | 5.3 |
| house | 128 | 0.77 | 0.34 | 0.10 | 18.7 | 19.3 |
| statue | 47 | 0.75 | 0.27 | 0.06 | 15.0 | 17.3 |
| farm | 36 | 0.56 | 0.20 | 0.24 | 14.5 | 13.8 |
| ship | 32 | 0.63 | 0.20 | 0.10 | 7.1 | 5.6 |
| shop | 69 | 0.45 | 0.20 | 0.09 | 21.3 | 21.2 |
| castle | 32 | 0.53 | 0.10 | 0.07 | 16.7 | 13.9 |
| tower | 35 | 0.51 | 0.05 | 0.02 | 20.5 | 19.7 |
| **mean (11)** | | **0.69** | **0.33** | **0.16** | 12.3 | 13.6 |

Mean margin (KID to the other classes minus KID to its own class): scratch 8.0, pretrain 3.1. The
shuffled-label control reads 0.09 (chance 0.09).

Houses card `run_20261105_212258_bench_category-conditioned_5M_cat_house` (`--tier both --n 256
--baselines`, the end-of-run draws with the house token):

| arm | n | BlockScore | appearance | geom_kid | mv_dino_kid | enterable | dino nn pct |
|---|--:|--:|--:|--:|--:|--:|--:|
| canon_oriented (T93) | 256 | 297 | 157 | 14.2 | 0.254 | 0.61 | 0.43 |
| cat 5M from world-crop pretrain | 256 | 349 | 220 | 16.8 | 0.355 | 0.23 | 0.69 |
| cat 5M scratch | 255 | 355 | 201 | 17.0 | 0.324 | 0.20 | 0.61 |
| world-crop 5M voxels, houses only (T94) | 256 | 361 | 238 | 17.3 | 0.384 | 0.15 | 0.68 |

The scratch arm dropped one empty build, so its BlockScore is at n=255 and not strictly comparable
(KIDs are n-stable). Holm groups on `geom_kid`: canon_oriented `c`, cat pretrain `cd`, cat scratch and
voxels `d`; on `mv_dino_kid`: cat scratch `d`, cat pretrain `de`, voxels `e`.

**Conditioning works from scratch and is cheap for houses.** The scratch model reaches about half the
real-build probe ceiling, and the shape classes the probe can tell apart (tree, vehicle, pixel art,
aircraft) come out close to their real class on geometry. Training on 11 other classes costs houses
nothing: both cat models tie or beat the houses-only voxel arm, and the scratch model's house
appearance is better (0.324 against 0.384). World-crop pretraining lowers holdout NLL but halves
conditioning (probe 0.16, margin 3.1): the pretrained prior ignores the new class token, the same
direction as pieces in T98. Tower, castle and shop barely separate from houses, which matches their
low ceilings. One seed. Next candidates: class-balanced sampling, a longer warm start or higher
learning rate for the class rows when starting from a pretrain, and the world-crop labels.

## T100. BrickNet stage 0: the corrected readout of mc_sft_base on the house card (2026-11-05)

Card `run_20261105_225739_bench_BrickNet_stage_0_readout` (`--tier both --n 256 --baselines`).
Sample run `run_20261105_212201_bricknet_sample_mc_sft_base` (`scripts/bricknet_pipeline.py sample
--model mc_sft_base --n 256 --seed 0 --max_new_tokens 20480 --batch_size 4`): Qwen3-0.6B with the
mc_pt_base then mc_sft_base LoRA stack (blockgen-21's runs, libs/BrickNet/repro), prompted with
training-house captions, T=1.0, top_k 20, top_p 0.95. 260 draws for 256 valid (4 unparsable, 0
empty), 93 min. 255 of 256 builds had node collisions, kept first node wins; 22 needed the 32^3
crop (x/z centred, y from the bottom). `train_split` canonical, `base_model` Qwen3-0.6B (external
pretraining). The first launch failed on a relative-path bug in the driver (fixed 2026-11-05).
Recipe: scratchpad `bricknet_stage0b.sh`.

| arm | BlockScore | worst pillar | appearance | geom_kid | mv_dino_kid | enterable | lcc | dino nn pct |
|---|--:|---|--:|--:|--:|--:|--:|--:|
| canon_oriented (T93) | 297 | geometry | 157 | 14.2 | 0.254 | 0.61 | 0.98 | 0.43 |
| 20M SFT + anneal mix (T87) | 332 | geometry | 238 | 15.9 | 0.384 | 0.28 | 0.94 | 0.66 |
| real_shuffled_materials (control) | 422 | appearance | 422 | 0 | 0.681 | | | |
| **BrickNet mc_sft_base** | **640** | appearance | 640 | 25.2 | 1.032 | 0.01 | 1.00 | 0.78 |
| train_copy_noise (baseline) | 647 | geometry | 272 | 31.2 | 0.439 | | | |

**BrickNet's Minecraft readout is far from houses.** It ranks below every real-build control and
just above train_copy_noise: a single connected clump per build (lcc 1.00, one component), no
interior (enterable 0.008, real 0.95) and appearance worse than shuffled materials (mv_dino_kid
1.03). The samples sheet shows irregular solid blobs and branching limbs in house palettes, never
walls, floors or roofs. This matches blockgen-21's eval-loss verdict (T69: no LEGO-to-Minecraft
transfer, and the merged model learns the node format but not the shapes). One seed, one model; the
LEGO-init arm ties mc_sft_base on eval loss and was not drawn at n=256.

## T101. Legacy 3D-BPE arms at n=256 under the current descriptor: orientation separates on geometry (2026-11-06)

Card `run_20261105_225412_bench_legacy_3D-BPE_arms_at_n_256` (`--tier both --n 256 --baselines`),
on the n=256 caches of T82 (`outputs/bench_arms/{canon_oriented,canon_unoriented,native_oriented}_n256_32.npz`),
so the houses32-v1 board has these arms at n=256 instead of the T64 n=128 rows. Recipe: scratchpad
`card_legacy256.sh`.

| arm | BlockScore | worst pillar | appearance | geom_kid | Holm (geom) | mv_dino_kid | enterable | dino nn pct |
|---|--:|---|--:|--:|---|--:|--:|--:|
| native_oriented (contaminated, T31) | 247 | geometry | 129 | 11.8 | c | 0.208 | 0.59 | 0.49 |
| canon_oriented | 297 | geometry | 157 | 14.2 | c | 0.254 | 0.61 | 0.43 |
| canon_unoriented | 362 | geometry | 183 | 17.3 | d | 0.295 | 0.63 | 0.40 |

**Orientation helps the legacy 3D-BPE recipe.** canon_oriented against canon_unoriented: geom_kid
-3.10 [-4.07, -2.01], p (Holm) 0.023; mv_dino_kid groups overlap (cd and d). Traversal is the
same for both (enterable 0.61 and 0.63). T82 showed the same gap under the v1 descriptor (15.06
against 18.25); the v2 descriptor keeps it and the paired test now separates it. Inside the
pipeline's voxel tokens orientation does not move geometry (T94), so the effect needs the
oriented piece vocabulary, not orientation alone. canon_oriented's 297 matches T93. native_oriented
stays unranked (its training split overlaps the test houses).

## T102. Category-conditioned 5M house card at n=256: the scratch arm redrawn with the top-up sampler (2026-11-06)

Card `run_20261106_000559_bench_category-conditioned_5M_cat_house_n_256` (`--tier both --n 256
--baselines`). The scratch arm of T99 was redrawn with the new top-up sampler (`--sample --n 256`,
step 17000, cat house, seed 1, T=1.0, top_k 40; one empty draw replaced in round 1, 256 builds),
written to `outputs/run_20261105_181801_cat5m_scratch_s1/samples256`. The other arms reuse their
T99 caches. Recipe: scratchpad `cat_scratch256.sh`.

| arm | n | BlockScore | worst pillar | appearance | geom_kid | Holm (geom) | mv_dino_kid | Holm (app) | enterable |
|---|--:|--:|---|--:|--:|---|--:|---|--:|
| canon_oriented | 256 | 297 | geometry | 157 | 14.2 | c | 0.254 | c | 0.61 |
| cat pretrain (cat5m_pt_s1) | 256 | 349 | geometry | 220 | 16.8 | cd | 0.355 | de | 0.23 |
| cat scratch (cat5m_scratch_s1) | 256 | 355 | geometry | 200 | 17.1 | d | 0.324 | d | 0.20 |
| voxels, world-crop mix (wc5m_sftmix_u_s1) | 256 | 361 | geometry | 238 | 17.3 | d | 0.384 | e | 0.15 |

**The n=255 reading holds.** The scratch arm scores 355 at n=256, the same as at n=255, and every
Holm group of T99 is unchanged. Both cat models tie the houses-only voxel arm on geometry; the
scratch model beats it on appearance (mv_dino_kid -0.061, p (Holm) 0.011). canon_oriented still
separates from both (geom_kid -2.81 against scratch, p (Holm) 0.014). All four arms now sit at the
board's n.

## T103. Facade arms at n=256: best-of-4 at T=0.7 reaches canon_oriented (2026-11-06)

Card `run_20261105_225345_bench_facade_arms_at_n_256` (`--tier both --n 256 --baselines`). Every
facade arm of T68, T70 to T73 and T77 re-scored on the first 256 builds of its n=399 cache
(`outputs/bench_arms/n256`, reports carry `prefix_of` and `n_source`); draws are i.i.d., so the
prefix is a fair sample. Recipe: scratchpad `card_facade256.sh`. The paired step over 25 arms took
about 3 h beside another session's CPU load.

| arm | BlockScore | worst pillar | appearance | geom_kid | Holm (geom) | mv_dino_kid | enterable | dino nn pct |
|---|--:|---|--:|--:|---|--:|--:|--:|
| BoN face_solidity k4, T=0.7 (T77) | 290 | geometry | 99 | 13.9 | c | 0.160 | 0.35 | 0.48 |
| RFT face_solidity, T=0.7 (T73) | 320 | geometry | 109 | 15.3 | cde | 0.177 | 0.36 | 0.58 |
| real@facades6 (ceiling) | 328 | geometry | 8 | 15.7 | cd | 0.014 | 0.70 | 0.46 |
| joint D4 20M, T=0.7 (T70) | 387 | geometry | 175 | 18.6 | de | 0.283 | 0.29 | 0.51 |
| BoN face_solidity k4, T=1.0 (T70) | 513 | geometry | 183 | 24.7 | fg | | | |
| facade_patchwork (baseline) | 570 | geometry | 327 | 27.4 | gh | | | |
| RFT face_solidity, T=1.0 (T72) | 599 | geometry | 241 | 28.8 | hi | | | |
| joint 20M, T=1.0 | 620 | geometry | 343 | 29.9 | hi | | | |
| joint D4 20M, T=1.0 (T68) | 631 | geometry | 291 | 30.4 | i | | | |
| joint D4 5M | 650 | geometry | 356 | 31.3 | ij | | | |
| joint 5M | 657 | geometry | 395 | 31.6 | ij | | | |
| independent faces 5M | 714 | geometry | 442 | 34.4 | l | | | |

**The order of record holds at n=256, and the best facade arm now meets the voxel best.** Every
pairwise order of the n=399 cards is kept. Best-of-4 at T=0.7 scores 290 against canon_oriented's
297 (T101): geom_kid 13.9 against 14.2, appearance much better (0.160 against 0.254), traversal
worse (enterable 0.35 against 0.61). The two were scored on different cards, so the geometry gap
has no paired test. It ties the facade ceiling on geometry (Holm c against cd), as at n=399 (T77).
Its reward, face_solidity, is not a pillar signal, but best-of-4 costs four draws per build.

## T104. Facade best-of-4 at T=0.7 against canon_oriented on one card, and the single-material boxes (2026-11-06)

Card `run_20261106_141254_bench_facade_best-of-4_T0_7_against_canon` (`--tier both --n 256
--baselines`), the T103 and T101 caches of the two arms plus real@facades6. Recipe: scratchpad
`card_pair256.sh`. Box count: scratchpad `grey_boxes.py`, output `grey_boxes.json`.

| arm | BlockScore | appearance | geometry | palette | coherence | traversal | geom_kid | mv_dino_kid | palette_jsd_exact | bbox_fill_w1 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| facade BoN face_solidity k4, T=0.7 | 290 | 99 | 290 | 30.6 | 1.01 | 1.70 | 13.9 (c) | 0.160 (c) | 0.271 | 0.072 |
| canon_oriented | 297 | 157 | 297 | 6.1 | 0.48 | 0.99 | 14.2 (c) | 0.254 (d) | 0.112 | 0.022 |
| real@facades6 (ceiling) | 328 | 8 | 328 | 3.1 | 2.02 | 0.77 | 15.7 (c) | 0.014 (b) | | |

**Geometry ties, appearance goes to the facade arm, every other pillar to canon_oriented.** On
`geom_kid` the two arms and the facade ceiling share Holm group c (no pair separates). On
`mv_dino_kid` the facade arm is better by 0.095 (p (Holm) 0.009). canon_oriented is better on
palette (6.1 against 30.6), coherence and traversal; BlockScore takes only the worst pillar, which
is geometry for both, so the 290 against 297 ordering is a tie.

**One facade build in eight is a single-material box.** Share of builds with at least 90% of their
blocks one material (n=256; real test n=399):

| arm | at least 90% | at least 70% |
|---|--:|--:|
| real test | 1.0% | 9.8% |
| real@facades6 | 2.3% | 6.2% |
| canon_oriented | 0.8% | 9.4% |
| facade raw, T=1.0 | 5.5% | 15.2% |
| facade raw, T=0.7 | 17.6% | 24.2% |
| facade RFT, T=0.7 | 17.2% | 23.0% |
| facade BoN k4, T=0.7 | 11.7% | 19.1% |

The boxes come from the lower temperature (raw 5.5% to 17.6%); best-of-4 lowers them, so
face_solidity does not select for them. On the sheet's first 32 builds the 90% rule found 5 of the 7
boxes picked by eye (the two missed are stone or brick with glass). The palette pillar and
bbox_fill_w1 register them; geometry and appearance barely do, which is a blind spot to state
with the metric.

## T105. V-KID robustness: leave-family-out, random subsets, generic descriptors, global terms (2026-11-06)

Run `outputs/run_20261106_193358_vkid_robustness` (`scripts/vkid_robustness.py`). Protocols are those of §80: ladder n=128 gates, and the
n=256 leaderboard of record with build hashes matched. Details are in notes §82.

| descriptor | dims | gates (ladder) | Fill (board) | Hollow (board) | Patch-Stitch | rho vs V-KID |
|---|--:|---|--:|--:|--:|--:|
| full V-KID | 77 | PASS | 444.4 | 265.4 | 368.2 | 1.00 |
| leave-one-family-out, 7 rows | 22-76 | 7/7 PASS | 310-457 | 146-270 | 327-375 | 0.95-1.00 |
| random 50%, 20 draws, median [min] | 38 | 20/20 | 351 [283] | 221 [133] | 317 [230] | 0.97 [0.88] |
| random 25%, 20 draws, median [min] | 19 | 17/20 | 298 [179] | 153 [56] | 246 [178] | 0.95 [0.85] |
| generic (3 random CNNs, hashed 3^3, neighbour counts, CNN+counts) | 168-392 | 6/6 PASS | 257-437 | 166-297 | 278-515 | 0.73-0.80 |
| V-KID + global terms (LCC / pooled orbits) | 79-189 | 4/4 PASS | 455-549 | 253-302 | 354-432 (rank 4/9) | 1.00 |

**V-KID's result does not depend on its features.** Every family can be removed, and every random half
of the dimensions passes the gates. Hand-design-free D4-orbit-averaged descriptors also pass every
gate and separate both interior probes. They agree with V-KID on the probes and baselines, and
disagree on the order of the four learned arms. Global terms leave Patch-Stitch 4th of 9.

## T106. The blindness on real generators: MV-KID and V-KID rankings of 40 generators (2026-11-06)

Run `outputs/run_20261106_193513_render_blind_rank_flips` (notes §83). The generators are every house
arm on the 399-val reference under the post-fix renderer. V-KID was recomputed exactly; MV-KID is the
card values. 2000 subsampling replicates, Holm over all pairs.

| set | k | Spearman MV vs V-KID v1 | v2 | self MV / V | pairs both sig. | opposite |
|---|--:|---|---|---|--:|--:|
| all generators | 40 | 0.50 [0.42, 0.54] | 0.65 [0.58, 0.68] | 0.99 / 0.98 | 388 | **76** |
| n=256 only | 35 | 0.60 [0.51, 0.63] | 0.68 [0.62, 0.71] | 0.99 / 0.99 | 326 | 55 |
| autoregressive voxel | 24 | 0.37 [0.21, 0.45] | 0.47 [0.36, 0.55] | 0.98 / 0.97 | 100 | 24 |
| facade | 10 | 0.99 | 0.99 | 1.00 / 0.99 | 36 | 0 |

| arm | MV z (rank /41) | V-KID v1 z (rank /41) | hidden blocks |
|---|--:|--:|--:|
| real@facades6 (perfect facade generator) | +8.1 (1) | +310.6 (19) | 0% |
| m2b_20m_sftmix_s1 (best voxel) | +238.1 (21) | +277.9 (5) | 12.6% |
| same outputs, hollowed | +0.15 vs arm at n=128 (osmesa) | +425.3 (29); +117.1 vs arm at n=128 | 0% |
| rft_facades_joint_d4_20m_face_solidity | +241.2 | +570.8 | 0.2% |

**MV-KID ties the facade RFT arm with the best voxel arm (-3.1 [-20.6, +14.5]); V-KID separates them
by 292.9 floors.** In 38 of the 76 reversed pairs the arm MV prefers is a facade arm, and facade arms
are shells (median 0.5% hidden blocks, against 15.6% for real). Over 34 design decisions: 13 agree,
0 are opposite, 12 are MV-only and 3 are V-KID-only.

Caveats:
- MV intervals are treated as independent, which is conservative.
- v2 gives 46 reversals.
- The exact paired-DINO rerun is pending.

## T107. Gate provenance and threshold robustness of the T92 verdicts (2026-11-06)

Run `outputs/run_20261106_193109_gate_provenance` (notes §84).

- **Dates:** bars set 2026-08-04 (a9eb431), V-KID added 2026-08-31, gate code frozen 2026-09-07
  (16b95f8), head-to-head 2026-11-04/05.
- **Robustness:** the sweep covers ordering tol -1..2 sd, G6 bar 0.5-5, G7 bar 1-5 and G8 bar 1-5.
  - V-KID is the sole geometry pass at every tol <= 0 and every G6 bar.
  - V-KID passes all 360 of the 480 settings with G7 >= 2.
  - A G7 bar of 1 rejects every row.
  - A looser G2 (tol >= 0.5) admits EMD and IoU-D4 1-NNA (Fill +1.6/+2.7).
  - Adding a G6 bar of 2 admits occupancy KID (+48.4).
  - No threshold admits the encoders (G6 63-106 floors) or IoU MMD (G8 34-36 SE).
- **Novelty gates:** across z 2/3/5 and floors x0.5/1/2, the positive controls, Dream-Cubed and
  shell_box are DQ in all 9 settings. The loosest corner readmits gabled_house. z=2 first DQs our
  own canon arms.
- **Coverage:** the G7/G8 statistics are rerun for 8 of 21 rows; the chamfer, EMD and encoder rows
  are held at their T92 verdicts.

## T108. Leak census, opaque-renderer MV-KID at n=256, leak dose curve (2026-11-06)

Run `outputs/run_20261106_193345_leak_opaque_dose`; details notes §85. houses_32 test (399 x 4 views),
val reference, DINOv2-B corner views, n = 256.

| pipeline | views altered (Fill) | median px | median max ch | MV-KID Fill | MV-KID Hollow |
|---|--:|--:|--:|--:|--:|
| deployed | 90.9% | 1.23% | 128 | +0.29 | +0.03 |
| translucency only (atlas fixed) | 58.7% | 0.21% | 107 | +0.40 | +0.13 |
| glass opaque, atlas fixed (GPU) | 11.1% | 0.008% | 1 | -0.00004 | -0.00002 |
| glass opaque, atlas fixed (CPU) | 0 / 1,596 | - | - | (identical renders) | (identical) |

Dose (bins by per-house leak, atlas fixed): separation 0.00 / -0.00 / +0.03 / +0.40 / -0.66 floors from
no leak to the top quartile (median leak 4.4% of the silhouette), while per-house feature shift
tracks the leak (Spearman 0.96). LEGO: deployed -0.011, atlas fixed -0.00004.
Paper numbers that change: 1.75% -> 1.23%, 212/255 -> 128/255, "89% of houses" -> 67%
(translucency) / 92% (deployed); app:hull deployed row 8.6% / 1.67% / 211 -> 9.1% / 1.23% / 128.

## T109. Objaverse mesh probe at 568 strict objects, seed 0, n = 170 (2026-11-07)

Run `outputs/run_20261106_210006_mesh_probe3d_568_s0` (results.md; notes §86). 284 ref / 284 pool, n = 170
(was 290 objects, 130 / 160, n = 96).

| row | Strip | Fill | noise | wall | decimate |
|---|--:|--:|--:|--:|--:|
| DINOv2 4 ext. KID | +0.0 | +0.0 | +55.6 | +10.7 | +61.8 |
| DINOv2 4 cut KID | +17.2 | +32.5 | +126.1 | +21.2 | +129.5 |
| V-KID | +65.8 | +258.3 | +69.8 | +43.2 | +64.7 |
| Uni3D vis KID | -0.5 [-0.9, -0.2] | -0.4 [-1.0, -0.0] | +38.6 | +37.8 | +44.3 |
| ULIP-2 vis KID | -0.0 [-0.1, +0.1] | -0.1 [-0.3, -0.0] | +14.2 | +18.4 | +32.7 |

Blindness holds at the larger n; interior separations grow (V-KID Fill 73.8 -> 258.3). Seed 1 and the
chamfer/EMD rows with intervals pending (script still scoring at the time of this entry).

## T110. Style-cluster fine-tune: cottage cluster against the all-houses mix, n = 256 (2026-11-07)

Run `outputs/run_20261107_113816_cluster_eval_cottage` (scripts/cluster_eval.py; notes, style-cluster paragraph).
Cluster: houses_32 both_k6 c1 (medieval cottages, `outputs/run_20261106_141808_cluster_houses`), grown with
1,519 anneal builds inside the cluster radius (`outputs/run_20261106_162105_cluster_assign_anneal32`):
sft-cottage 1,881 builds (9859ff1c7efa), eval-cottage 114 val+test builds (reference). All arms init from
wc5m_pretrain_u_s1, 12k steps, lr 1e-4, dropout 0, select best, tokenizer 60684456e598. KID x1000 against the
cluster's 114 held-out builds, 95% interval from 200 resamples of the arm; cluster scale, comparable only in this table.
Style hit = share of builds the cluster k-means places in c1 within its 90th-percentile radius.

| arm | geom_kid | mv_dino_kid | style hit | eval-cottage NLL | eval-holdout NLL |
|---|--:|--:|--:|--:|--:|
| real cottage train | 0.32 [0.28, 0.72] | 0.008 | 0.895 | | |
| real houses test, all styles | 3.0 [2.5, 3.8] | 0.446 | 0.074 | | |
| all-houses mix (wc5m_sftmix_u_s1) | **16.7** [16.1, 17.5] | **1.085** | 0.0 | | |
| cottage + world-crop mix (cot5m_mix_s1) | 19.9 [19.2, 20.9] | 1.239 | 0.0 | 0.4635 | 0.4806 |
| cottage only (cot5m_sft_s1) | 21.2 [20.3, 22.3] | 1.140 | 0.0 | 0.4638 | 0.5543 |

Training on the style subset made the model worse at that style on both metrics, and no arm produces a single
build the clusterer calls a cottage. Without the radius, the nearest centre of every generated build is a
catch-all cluster (c0, c2 or c5): base 55/22/20%, mix 47/18/34%, cottage-only 25/21/54%, c1 0% in all three
(real reference 100% c1). Median distance to the nearest centre is 0.92 to 0.94 for the models against 0.84
for the real reference. Samples (cot5m_sft_s1/samples.png) are wood-and-cobble clumps: the palette moved
towards the style, the structure did not. Narrowing the data does not fix structure; the generator's error
is off the style manifold, not between styles. Cottage-only also loses 0.07 nats on the general holdout.
Pool follow-up: assigning all 358,976 anneal + pretrain-all builds that fit 32
(`outputs/run_20261107_183120_cluster_assign_anneal_pretrain-all32`) grows cottage only from 1,519 to 1,720
accepted builds (both_k6 c1; dino_k6 c1 1,885). The extra pretrain-all crops fall into the catch-all clusters
(both_k6 c0 264,883), so a larger cottage set is not available and the cluster track stops here.

## T111. canon_oriented recreated at two seeds: the legacy recipe reproduces (2026-11-07)

Card `run_20261107_222928_bench_canon_oriented_recreated_at_seeds_0_and` (`--tier both --n 256 --baselines`).
The Sep 17 command rerun unchanged (`python -m blockgen.experiments_native --arms bpe --oriented --epochs 60
--batch 8 --merges 256 --split canonical`) through a scratchpad wrapper that pins the split seed to 0, so the
second torch seed does not move val/test builds into train. Runs `run_20261107_130322_recanon_s0_native` and the
seed 1 sibling; tokenizer identical to the original (1,079 pieces, 13,890 of 14,868 D4 builds fit, median 1,814).
256 samples each via `scripts/sample_to_npz.py` (seed 0, T=1.0, top_k 40), as for the original. Recipe: scratchpad
`recanon/recanon.sh` (log `outputs/logs/recanon.log`). Final train loss 0.315 (original), 0.319, 0.326.

| arm | BlockScore | appearance | geom_kid | mv_dino_kid | enterable | components | dino nn pct |
|---|--:|--:|--:|--:|--:|--:|--:|
| canon_oriented (Sep 17) | 297 | 157 | 14.2 | 0.254 | 0.61 | 7.9 | 0.43 |
| recreation, seed 0 | 313 | 151 | 15.0 | 0.245 | 0.66 | 7.7 | 0.46 |
| recreation, seed 1 | 319 | 129 | 15.3 | 0.208 | 0.56 | 7.3 | 0.41 |

All three share Holm group `c` on both `geom_kid` and `mv_dino_kid`. No gate fires.

**The legacy recipe reproduces; 297 is the lucky end of a 297 to 319 seed range.** Every recreation stays ahead
of every new-pipeline voxel arm (best 332, T87), and seed 1 has the best appearance of any learned arm (129).
The gap to the pipeline is the recipe, not a fortunate checkpoint. The world-crop-pretrained version of the same
recipe (scratchpad `recanon/legacy_pre.py`) is scored next.

## T112. canon_oriented recipe with world-crop pretraining: new best learned arm, 255 (2026-11-08)

Card `run_20261107_234331_bench_canon_oriented_recipe_with_world-crop` (`--tier both --n 256 --baselines`).
Run `run_20261107_181022_recanon_wcpre_s0` (scratchpad `recanon/legacy_pre.py`, log `outputs/logs/wcpre.log`):
canon_oriented's own piece vocabulary tokenizes the 61,365 anneal-worldcrop builds that fit 32; 54,308 kept
(dropped: >10% voxels outside the vocab, or longer than 5480; none voxel-equal to a val/test house), 123M tokens,
median 2,018. Legacy trainer throughout (AR2 phase4, d256 x6, dropout 0.1, batch 8, lr 3e-4 constant): 8
epochs on the crops (loss 0.417), then the unchanged T111 house stage (D4, 60 epochs) from those weights
(final loss 0.268 against 0.319 from scratch). Seed 0; samples as T111.

| arm | BlockScore | appearance | geom_kid | mv_dino_kid | enterable | dino nn pct |
|---|--:|--:|--:|--:|--:|--:|
| **legacy recipe + world-crop pretraining** | **255** | **125** | **12.2** | **0.203** | 0.65 | 0.41 |
| canon_oriented (Sep 17) | 297 | 157 | 14.2 | 0.254 | 0.61 | 0.43 |
| recreation, seed 0 (T111) | 313 | 151 | 15.0 | 0.245 | 0.66 | 0.46 |
| recreation, seed 1 (T111) | 319 | 129 | 15.3 | 0.208 | 0.56 | 0.41 |

Holm groups on `geom_kid`: pretrained `c`, canon_oriented and seed 0 `cd`, seed 1 `d`; the pretrained arm
separates from seed 1 (-3.08 [-3.92, -2.10], p_adj 0.021) but not from canon_oriented or seed 0. On
`mv_dino_kid` all four share `c`. No gate fires (exact-duplicate rate 0; nn percentile 0.41, in the range of
the scratch arms).

**World-crop pretraining helps the legacy recipe, the opposite of T98.** It is the best learned arm on every
headline column (BlockScore 255, appearance 125, geom_kid 12.2, mv_dino_kid 0.203), and against the scratch seed
range (297 to 319) it is better by 40 to 60. Only the seed 1 comparison clears Holm on its own, so a second
pretrained seed is needed before the gain is claimed against canon_oriented itself. What T98 changed that this
run did not: the pipeline tokenizer (unoriented, merges on sft-houses32) and the pipeline trainer (32k-token
steps, dropout 0 at fine-tune). One seed.

## T113. Second seed of the world-crop-pretrained legacy recipe: 358, so T112's 255 does not hold (2026-11-08)

Card `run_20261108_050420_bench_canon_oriented_recipe_with_world-crop` (`--tier both --n 256 --baselines`).
`run_20261108_010628_recanon_wcpre_s1` repeats T112 at torch seed 1 (same 54,308 tokenized crops, 8 pretraining
epochs, 60 house epochs; final house loss 0.275 against 0.268); samples as T111 (seed 0, T=1.0, top_k 40).
Recipe: scratchpad `recanon/wcpre_s1.sh` (log `outputs/logs/wcpre_s1.log`).

| arm | BlockScore | appearance | geom_kid | mv_dino_kid | enterable | palette |
|---|--:|--:|--:|--:|--:|--:|
| pretrained, seed 0 (T112) | 255 | 125 | 12.2 | 0.203 | 0.65 | 6.9 |
| canon_oriented (Sep 17) | 297 | 157 | 14.2 | 0.254 | 0.61 | 6.1 |
| scratch recreation, seed 0 (T111) | 313 | 151 | 15.0 | 0.245 | 0.66 | 7.4 |
| scratch recreation, seed 1 (T111) | 319 | 129 | 15.3 | 0.208 | 0.56 | 6.9 |
| **pretrained, seed 1** | **358** | 146 | **17.2** | 0.236 | 0.60 | 10.2 |

Holm groups on `geom_kid`: pretrained s0 `c`, canon_oriented and scratch s0 `cd`, scratch s1 `d`, pretrained
s1 `de`. The two pretrained seeds separate from each other (-4.97 [-5.97, -3.97], p_adj 0.012). On
`mv_dino_kid` all five share `c`. No gate fires on a model arm.

**Pretraining widens the seed spread instead of shifting it.** Scratch seeds span 297 to 319 (mean 310);
pretrained seeds span 255 to 358 (mean 306). T112's lead was a good seed, and the pretrained recipe is not yet
better than the scratch one on average. What holds across all five: the legacy recipe beats every pipeline arm
on `mv_dino_kid` (0.203 to 0.254 against 0.31 or worse), and its geometry ranges from 12.2 to 17.2, so a single
seed decides geometry rankings at this scale. Claims about this recipe need at least three seeds per arm.

## T114. Functional ground truth on the 76 reversed pairs: sides with MV-KID (2026-11-08)

Run `outputs/run_20261108_204723_functional_groundtruth` (notes §87). Five player-based measures
(enterable, reached interior floor, reached rooms, roof, largest component), per build, on the
40 generators of T106/§83, real and probes; FD = mean per-measure W1 to the 399 val builds in
units of real between-build spread; B = 1000.

| arm | FD [95%] | MV-KID z | V-KID v1 z |
|---|---|--:|--:|
| held-out real (n=256) | **0.09** [0.08, 0.19] | 0 | 0 |
| FILL-INTERIOR | 1.21 [1.11, 1.32] | 0.4 | 444 |
| HOLLOW-SHELL (real) | 0.60 [0.52, 0.73] | | |
| real@facades6 | 1.22 [1.07, 1.38] | 8.1 | 311 |
| classic-matched (best generator) | 0.52 [0.43, 0.65] | 250 | 377 |
| m2b_20m_sftmix_s1 (V-KID's best voxel arm) | 1.32 [1.20, 1.43] | 238 | 278 |
| same @hollow | 0.90 | +0.15 vs arm | 425 |

| test | MV-KID | V-KID v1 |
|---|---|---|
| Spearman with FD, 40 generators | **+0.67** [0.41, 0.82] | +0.16 [-0.20, 0.45] |
| Spearman with FD, 30 non-facade | +0.60 [0.27, 0.80] | -0.19 [-0.52, 0.14] |
| reversed pairs FD sides with, all 76 | **60** | 16 (21% [12.5, 31.9]) |
| non-facade 38 / facade 38 | 37 / 23 | 1 / 15 |

Real ranks first of 41 under FD, FD_noise, FD_kid and occupancy-only FD_occ; the V-KID share
of reversed pairs is 8%, 26%, 24% under the variants. Per-build ridge R^2 of the measures on
V-KID's 77 features: 0.29 to 0.62. Verdict: the functional test detects the interior probes
MV-KID misses, but on real generators it agrees with the render ranking; V-KID rewards the
cluttered, unenterable interiors of voxel AR models (median 22% enterable vs real 94%).

## T115. Statistics audit: floor FPC, n-free effect size, exact 40-generator tests, TOST, kernel ablation (2026-11-08)

`outputs/run_20261108_205428_stats_fixes` (notes §88). CPU on cached features; Hollow/Grow DINO
rendered once.

| question | old | new |
|---|---|---|
| floor FPC, truth / bench (non-overlapping sim) | critic: up to 1/1.67 | 0.88-0.95 (n=256), floor overstated 5-12% |
| V-KID Fill-Interior n=256 (corrected floor) | 444.36 | 491 |
| MV-KID Fill-Interior n=256 | 0.35 | 0.34 |
| Fill, V/MV ratio | 1,270x (z) | 27x (effect size, anchor Random-Voxels) |
| BrickGPT geometric/render gap | 4.4x (z) | 2.2x (effect size) |
| 40 generators, opposite & both separated | 76 of 388 (subsample test) | 40 of 308 (exact permutation, Holm) |
| rho, 95% | 0.50 [0.42, 0.54] | 0.50 [0.37, 0.58] sampling; [0.20, 0.72] over generators |
| Fill / Hollow equivalence, MV-KID, margin 2 floors | not tested | TOST equivalent, 90% CI [-0.28, 0.98] / [-0.25, 0.33] |
| rank-interval rescale (x2) | unrescaled | no new ties |
| cubic KID linear share | | 99.8% MV, 98.8-99.5% V; dissociation holds under linear/RBF/no-whitening |

Verdict: the floor is slightly conservative, not anti-conservative; the cross-metric z ratios must
be replaced by effect sizes; the reversal count halves under the calibrated test.

## T116. Ladder P1: the pipeline with the legacy trainer's knobs scores 338, from 425 (2026-11-09)

Card `run_20261109_040307_bench_ladder_P1_pipeline_with_legacy_knobs` (`--tier both --n 256 --baselines`).
`run_20261108_202534_p5m_d4legacyknobs_s1` is T98's `p5m_d4scratch_s1` (5M, unoriented pieces `cdce56584ebf`,
D4) with every legacy setting a config can express: `dropout=0.1 select=last warmup=0 decay_frac=0 lr=3e-4
beta2=0.999 weight_decay=0.01 max_seq_len=5480 steps=104000` (constant lr, last checkpoint; 808 sequences
over 5,480 tokens dropped). Still the pipeline's: tag tokens, token-budget batches (16,384), decay on 2-D
weights only, unoriented pieces. Final eval-holdout loss 0.604. Recipe: scratchpad `recanon/ladder_p1.sh`
(log `outputs/logs/ladder_p1.log`).

| arm | BlockScore | appearance | geom_kid | mv_dino_kid | palette |
|---|--:|--:|--:|--:|--:|
| canon_oriented (legacy, oriented) | 297 | 157 | 14.2 | 0.254 | 6.1 |
| **pipeline + legacy knobs (P1)** | **338** | 176 | **16.2** | 0.285 | 4.4 |
| canon_unoriented (legacy, unoriented) | 362 | 183 | 17.3 | 0.295 | 7.8 |
| pipeline defaults (T98) | 425 | 209 | 20.4 | 0.338 | 11.3 |

Holm groups on `geom_kid`: canon_oriented `c`, P1 `cd`, canon_unoriented `de`, T98 `f`. P1 beats T98
(-4.20 [-5.17, -3.17], p_adj 0.011) and ties both legacy arms. On `mv_dino_kid` P1 shares `cd` with both
legacy arms. No gate fires on a model arm.

**The trainer half of the gap is config, and it closes.** The legacy settings move the pipeline 87 points,
past the legacy unoriented arm it was matched against (362), so nothing code-level is needed for the
unoriented comparison. The remaining 41 points to canon_oriented are within T101's 65-point orientation effect; P2
(the same knobs on oriented pieces `78a45c76538c`) tests whether orientation closes them. One seed.

## T117. Cross-face constrained decoding: consistent faces by construction, 679 to 589 at T=0.7, components at the facade ceiling (2026-11-09)

**Setup.** The 20M D4 joint facade model of T68 (selected step 18,000) sampled n=399 on the T68 boxes
(seed 0, top-k 40) under two decoders at two temperatures. The grammar control forces the face headers
and the final EOS and makes every cell `empty` or an in-box depth followed by a block, with no state
across faces. The constrained decoder adds the cross-face rule of notes §89 (`blockgen/corpus/facade_constraints.py`,
`blockgen/train/sample_constrained.py`): a `ShellState` over the box records each drawn cell's first hit as a
known voxel and the voxels in front of it as empty; a later face's cell may report its first hit only at a
depth not already seen through and not behind a known voxel, it carries the known block when it lands on one,
it may be see-through only when no known voxel lies on its ray, and a cell whose first open voxel is already
known is written without sampling. Everything the earlier faces left open stays the model's, including a block
in front of a known voxel where no ray has passed. Faces are drawn in the shard's order (front, right, back,
left, top, bottom). One card put the four decode arms beside the T68 raw, T70 T=0.7, T77 best-of-4 and T73
RFT arms, patchwork and the six-face ceiling:
`outputs/run_20261109_050232_bench_facade_cross-face_constrained_decoding`. Samples, face sheets and
`summary.json` per arm under `outputs/run_20261028_125240_houses32_facades_joint_d4_20m/constrained_<mode>_t<T>_n399`.
Decoding took 32 to 36 minutes per arm at batch 32; the constraint costs about four minutes.

**Self-consistency** (sampler's own measure: lift the six faces, project back, count cells that differ).

| arm | consistent face sets | cells changed on reprojection | filled cells hidden | see-through cells filled in | cells written by the rule | median blocks |
|---|---|---|---|---|---|---|
| grammar, T=1.0 | 0.000 | 0.386 | 0.298 | 0.587 | 0 | 771 |
| constrained, T=1.0 | 1.000 | 0.000 | 0.000 | 0.000 | 0.253 | 606 |
| grammar, T=0.7 | 0.018 | 0.281 | 0.214 | 0.384 | 0 | 686 |
| constrained, T=0.7 | 1.000 | 0.000 | 0.000 | 0.000 | 0.214 | 509 |

Free decoding almost never produces a consistent face set: three cells in ten come back different after
the lift, and more than half of the see-through cells at T=1.0 are filled by some other face. Under the
constraint a fifth to a quarter of all cells have exactly one legal answer and are written without
sampling.

**The card** (n=399 unless noted; the T=0.7 arms lose the handful of all-air draws).

| arm | BlockScore | geom_kid | mv_dino_kid | n_components (real 2.7) | lcc_ratio (real 0.98) | n_blocks (real 904) | enterable (real 0.95) |
|---|---|---|---|---|---|---|---|
| T68 raw, T=1.0 | 1082 | 30.3 | 0.468 | 60.0 | 0.811 | 785 | 0.348 |
| grammar, T=1.0 | 1057 | 29.6 | 0.496 | 62.7 | 0.813 | 806 | 0.446 |
| constrained, T=1.0 | 963 | 27.0 | 0.331 | 29.5 | 0.794 | 624 | 0.331 |
| T70 raw, T=0.7 | 637 | 17.8 | 0.268 | 29.8 | 0.836 | 713 | 0.309 |
| grammar, T=0.7 | 679 | 19.0 | 0.299 | 33.7 | 0.810 | 689 | 0.316 |
| constrained, T=0.7 | 589 | 16.5 | 0.166 | 12.9 | 0.812 | 541 | 0.285 |
| T73 RFT + selection, T=0.7 | 553 | 15.4 | 0.182 | 17.6 | 0.937 | 707 | 0.352 |
| T77 best-of-4, T=0.7 | 483 | 13.5 | 0.149 | 15.9 | 0.948 | 697 | 0.365 |
| patchwork | 990 | 27.7 | 0.538 | 26.5 | 0.898 | 1028 | 0.231 |
| real@facades6 (ceiling) | 493 | 13.8 | 0.001 | 13.5 | 0.814 | 660 | 0.729 |

Holm-corrected paired tests (10,000 permutations): the constrained arm is separated from its grammar
control on `geom_kid` at both temperatures (T=1.0: 27.0 vs 29.6, p_Holm 0.033; T=0.7: 16.5 vs 19.0,
p_Holm 0.033) and on `mv_dino_kid` at both (groups f vs gh, cd vs ef). At T=0.7 the constrained arm
shares a `geom_kid` group with RFT (d, cd) and is separated from best-of-4 and the ceiling (c); on
`mv_dino_kid` it shares a group with best-of-4 (cd, c). The grammar control is not separated from the raw
sampler at either temperature, so the grammar itself changes nothing and the whole effect is the
cross-face rule.

**Reading.** The pre-registered prediction of notes §89 held: the constraint cuts the lifted shell's
components from 34 to 13 at T=0.7, the six-face ceiling's 13.5, and moves `geom_kid` by 2.5 at both
temperatures. The extra pieces of T68 were edge disagreement between faces, not fragmentation inside
them, and removing them is worth 90 BlockScore points at T=0.7 (679 to 589) and 94 at T=1.0, at no
training cost and four minutes of decode time. The appearance gain is larger still: `mv_dino_kid` 0.299
to 0.166, level with best-of-4, because a shell whose faces agree renders as one house. The sheets
show the same thing: on the same boxes and seed, the constrained builds have roofs that sit on their
walls and far fewer stray blocks in the air.

The constraint does not close the gap to the ceiling (16.5 vs 13.8 on `geom_kid`) or to best-of-4
(483), and the card says where the rest sits: the constrained shells are thinner and smaller
(541 blocks against the ceiling's 660, `wall_frac` 0.65 against 0.74, `walk_floors` 0.45 against
0.87, yaw symmetry 0.57 against 0.67) while `lcc_ratio` stays at 0.81, which is also the ceiling's
value and so is not what separates them. The rule has one cost, visible in the median block count
(686 to 509) and in the sheets: a see-through cell is absolute, so a front face with holes empties
whole columns for the five faces after it, and a sparse front face sentences the build to a
skeleton (two beams, a row of dots) where the control drew a dense cluster. `enterable` drops a
little (0.316 to 0.285) for the same reason. The front face dictates because it is drawn first; the
order is the shard's training order, so a different order needs a retrained model.

**What follows.** The lever is orthogonal to selection and to the fine-tune, so the next arms are
constrained best-of-4 at T=0.7 and constrained decoding of the T73 RFT model, which should combine
the component fix with their solidity. The see-through cost suggests a relaxed rule that forbids
first-hit contradictions but lets a later face fill a column an earlier face saw through, re-projecting
the earlier face's cell. The oracle completions from the real front face (and front plus right),
which measure how much of a real shell the model recovers when the dictating face is correct, OOM'd
at batch 32 beside another session's bench job and are rerunning at batch 8; their numbers are
appended to notes §89 when they land.

## T118. canon_oriented's recipe at 20M: 190, the best learned arm by a margin past seed noise (2026-11-09)

Card `run_20261109_074328_bench_canon_oriented_recipe_at_20M` (`--tier both --n 256 --baselines`).
`run_20261108_202210_recanon20m_s0_native` is T111's command (legacy `experiments_native`, oriented 3D-BPE,
D4, 60 epochs, batch 8, constant lr 3e-4, split pinned to seed 0) with the model widened to d512 x 8, ff 2048
(27M parameters against 5M), torch seed 0. Final train loss 0.112 against 0.319/0.326 for the 5M seeds.
Samples as T111 (T=1.0, top_k 40, 256). Recipe: scratchpad `recanon/legacy20m.sh` (log `outputs/logs/legacy20m.log`).

| arm | BlockScore | appearance | geom_kid | mv_dino_kid | palette | dino_nn_percentile | voxel_nn_iou |
|---|--:|--:|--:|--:|--:|--:|--:|
| **20M, seed 0** | **190** | **98** | **9.1** | **0.159** | 4.5 | 0.359 | 0.390 |
| canon_oriented (5M, Sep 17) | 297 | 157 | 14.2 | 0.254 | 6.1 | 0.433 | 0.360 |
| 5M recreation, seed 0 (T111) | 313 | 151 | 15.0 | 0.245 | 7.4 | 0.464 | 0.361 |
| 5M recreation, seed 1 (T111) | 319 | 129 | 15.3 | 0.208 | 6.9 | 0.412 | 0.366 |
| real test | 0 | 0 | -0.1 | 0.001 | 0 | 0.508 | 0.375 |

Holm on `geom_kid`: 20M beats canon_oriented (-5.18 [-6.04, -4.14]), recanon s0 (-5.95) and s1 (-6.22), all
p_adj 0.011. On `mv_dino_kid` 20M is alone in group `c` (5M arms `cd`/`d`). No gate fires on a model arm.

**Scale is the lever the legacy recipe was missing.** The 20M model gains 5 to 6 geom units over every 5M arm,
past the 5-unit seed spread T113 measured, and leads on appearance, palette and both primaries. Samples
(`native_bpe/samples_textured.png`) show roofs, stone bases, windows and furnished interiors. The cost is
proximity to training data: `dino_nn_percentile` falls to 0.359 (real 0.508, 5M 0.41 to 0.46) and train loss
is a third of the 5M one, but there are no voxel duplicates and nearest-neighbour IoU (0.390) is at real's
level (0.375). One seed; T112's 255 that did not hold is the warning. Next: a second 20M seed.

## T119. Ladder P2: oriented pieces under the legacy knobs score 330, tied with canon_oriented (2026-11-09)

Card `run_20261109_075008_bench_ladder_P2_oriented_pieces_with_legacy` (`--tier both --n 256 --baselines`).
`run_20261109_052337_p5m_d4oriknobs_s1` is T116's P1 with the tokenizer swapped for oriented pieces
`78a45c76538c` (`learn_pieces --oriented`, 213 forced + 256 merges, D4 shard `sft-houses32/78a45c76538c-d4`);
same knobs, 104k steps. Final eval-holdout loss 0.621 (not comparable to P1's 0.604: different vocabulary).
Recipe: scratchpad `recanon/ladder_p2.sh` (log `outputs/logs/ladder_p2.log`).

| arm | BlockScore | appearance | geom_kid | mv_dino_kid | palette |
|---|--:|--:|--:|--:|--:|
| canon_oriented (legacy, oriented) | 297 | 157 | 14.2 | 0.254 | 6.1 |
| **pipeline + legacy knobs, oriented pieces (P2)** | **330** | **156** | 15.8 | **0.253** | 9.2 |
| pipeline + legacy knobs, unoriented pieces (P1) | 338 | 176 | 16.2 | 0.285 | 4.4 |
| canon_unoriented (legacy, unoriented) | 362 | 183 | 17.3 | 0.295 | 7.8 |
| pipeline defaults (T98) | 425 | 209 | 20.4 | 0.338 | 11.3 |

Holm on `geom_kid`: canon_oriented, P2 and P1 share `c`; canon_unoriented `cd`; T98 `d`. On `mv_dino_kid`
P2 and canon_oriented share `c`, P1 and canon_unoriented `cd`. P2 beats T98 on both primaries (p_adj 0.012).

**The pipeline now matches the legacy recipe.** Orientation moves appearance and `mv_dino_kid` to
canon_oriented's level (156 against 157; 0.253 against 0.254), and geometry sits 1.6 units behind, inside one
Holm group and inside the 5-unit seed spread. The 33-point BlockScore gap is one seed's worth. The ladder's
answer: the pipeline's deficit was trainer settings (T116) plus unoriented pieces (this entry); tags,
token-budget batching and 2-D-only decay cost nothing measurable. Pipeline runs at 20M with these knobs and
oriented pieces are the comparison to T118.

**Oracle completions (appended 2026-11-09, batch 8).** The 399 real test houses, the real front face (or
front plus right) forced verbatim, the model drawing the remaining faces; agreement of each drawn face
with the real one, and IoU of the lifted shell with the real shell. Means over the drawn faces.

| given | mode | T | shell IoU | shell block agree | drawn-face mask IoU | depth exact | block agree | cells written by the rule | components |
|---|---|---|---|---|---|---|---|---|---|
| front | grammar | 1.0 | 0.347 | 0.505 | 0.687 | 0.324 | 0.331 | 0 | 8.0 |
| front | constrained | 1.0 | 0.509 | 0.581 | 0.849 | 0.601 | 0.468 | 0.285 | 3.1 |
| front | grammar | 0.7 | 0.362 | 0.552 | 0.665 | 0.354 | 0.373 | 0 | 5.5 |
| front | constrained | 0.7 | 0.553 | 0.601 | 0.869 | 0.639 | 0.505 | 0.282 | 2.3 |
| front + right | grammar | 0.7 | 0.469 | 0.630 | 0.705 | 0.394 | 0.414 | 0 | 4.8 |
| front + right | constrained | 0.7 | 0.716 | 0.668 | 0.955 | 0.744 | 0.571 | 0.314 | 2.1 |

Given the real front face, the constraint lifts the completed shell's IoU with the real shell from
0.36 to 0.55, and with front plus right from 0.47 to 0.72. The per-face table says why: the back
face's mask is fixed by the front face (every filled front column has a back hit, every see-through
column has none), so its mask IoU is 1.0 by construction, and the side faces' depth, which is mostly
the footprint's extent, goes from 0.29 exact to 0.74. What the rule leaves to the model stays hard:
the back face's depth (0.27 exact either way, a 2.7 cell mean error) and the block identity of every
free cell (0.37 to 0.51). So the constraint recovers the silhouette and the footprint from a correct
front face; materials and the relief of the far side are the model's and do not improve much.

## T120. The f=4 oracle of the cascade-and-AR hybrid: the fine model ignores its coarse level (2026-11-09)

**Setup** (notes §90). A per-face facade model that reads its own level f=4 before the face: the
`pooled4` layout, `[BOS, tags, face, U/4, V/4, cell*, face, U, V, cell*, EOS]`, trained with T66's
per-face 5M recipe unchanged (8,000 steps, five minutes, `outputs/run_20261109_125654_houses32_facades_pooled4_5m`,
holdout NLL 0.72 over face and prefix cells). The oracle gives the real f=4 (`any` pooling) of each of
the 399 test front faces and lets the model draw the face at T=1.0 and T=0.7; the copy is the upsample
of that level, the cascade row is T88's completion from the same level. `hybrid from the box` is the
model drawing its own coarse level first, the two-stage AR as a generator.

| arm | mask IoU | mask recall | depth exact | block agree | distinct blocks / face (real 12.6) | faces with glass (real 0.74) | W1 fill_frac | W1 n_blocks |
|---|---|---|---|---|---|---|---|---|
| copy (upsample f4) | 0.761 | 1.000 | 0.520 | 0.556 | 4.4 | 0.21 | 0.208 | 8.2 |
| cascade, mode from f4 (T88) | 0.754 | 0.864 | 0.581 | 0.515 | 4.4 | 0 | | |
| hybrid from real f4, T=1.0 | 0.617 | 0.762 | 0.381 | 0.284 | 9.9 | 0.69 | 0.022 | 2.7 |
| hybrid from real f4, T=0.7 | 0.574 | 0.704 | 0.403 | 0.333 | 4.5 | 0.37 | 0.051 | 8.1 |
| hybrid from the box, T=1.0 | 0.567 | 0.739 | 0.175 | 0.022 | 9.9 | 0.69 | 0.027 | 2.7 |

**Reading.** The pre-registered bar was block agreement above the copy's 0.556, which the cascade
missed at 0.515. The hybrid misses it by much more: 0.28 to 0.33, with mask IoU 0.62 against the
copy's 0.76 and a quarter of the dictated silhouette left undrawn (recall 0.76). With the real
gestalt in its context the fine model does not reproduce it; its faces given the real f=4 agree with
the real face barely more than its faces drawn from the box (mask IoU 0.617 against 0.567), so the
prefix carries palette and size and little else, and the sheet shows the per-face model's usual
speckled faces under a coarse level they do not follow. What the AR fine stage does deliver is the
distribution: windows on 69 percent of faces against the cascade's none (real 74 percent), 9.9
distinct blocks per face against the cascade's 4.4 (real 12.6), and W1 to real on fill fraction and
block count a tenth of the copy's. So the two stages have complementary failures, the cascade
faithful to the gestalt and blind to detail, the AR rich in detail and deaf to the gestalt, and
putting the gestalt in the AR's context does not join them. Under the rule set in §90 this closes
option (a) of §74 for faces as a free lever; the fix it points to is a decode-time constraint that
holds the fine face to its coarse level (empty coarse blocks empty, filled ones filled, the move of
T117 one level up), which would buy mask fidelity but not block agreement, the number that failed.
The trainer's own fast-tier card at n=64 scored the free two-stage model 220, not comparable with the
n=399 facade cards.

## T121. Category conditioning on the legacy recipe: probe 0.53 against 0.33, and its houses tie canon_oriented (2026-11-09)

`run_20261109_091009_legacycat_s0` (scratchpad `recanon/legacy_cat.py`, chain `recanon/legacy_cat.sh`, log
`outputs/logs/legacy_cat.log`): T111's legacy recipe (5M AR2 phase4, oriented 3D-BPE learned on this data,
256 merges, D4, batch 8, constant lr 3e-4, dropout 0.1, 22 epochs) on `sft-cat` builds that fit 32 (4,621,
11 classes plus `none`; 36,188 after D4, 34,057 within 5,480 tokens). The class token replaces BOS, as in
T99. Final train loss 0.427. Samples at T=1.0, top_k 40, batched 16 (full re-forward, no KV cache): 128 per
class, 256 for house. Sheets: `sheets/overview.png` (8 per class, one class per row, sampling order) and
`sheets/<class>.png` (24 each).

Conditioned eval (`python -m blockgen.eval.conditioned --n 128 --boot 200`, T99 protocol, same probe and
`eval-cat` reference):

| class | ceiling | T99 pipeline probe | **legacy probe** | T99 geom_kid | **legacy geom_kid** |
|---|--:|--:|--:|--:|--:|
| tree | 0.95 | 0.78 | **0.95** | 4.2 | **1.9** |
| vehicle | 0.74 | 0.58 | **0.85** | 4.3 | **0.8** |
| pixelart | 1.00 | 0.56 | **0.74** | 8.0 | **1.8** |
| aircraft | 0.63 | 0.39 | **0.51** | 5.1 | **3.0** |
| house | 0.77 | 0.34 | **0.44** | 18.7 | **12.2** |
| statue | 0.75 | 0.27 | **0.62** | 15.0 | **7.4** |
| farm | 0.56 | 0.20 | **0.36** | 14.5 | **8.0** |
| ship | 0.63 | 0.20 | **0.38** | 7.1 | **2.6** |
| shop | 0.45 | 0.20 | **0.42** | 21.3 | **12.5** |
| castle | 0.53 | 0.10 | **0.20** | 16.7 | **10.1** |
| tower | 0.51 | 0.05 | **0.31** | 20.5 | **11.0** |
| **mean (11)** | **0.69** | 0.33 | **0.53** | 12.3 | **6.5** |

Mean margin (KID to the other classes minus KID to its own): 13.5 against T99's 8.0. Every class improves
on both columns. Copy check (scratchpad, D4 train builds of the class): no sample equals a training build
in any of pixelart, tree, vehicle and house; same-shape IoU >= 0.8 for 0 to 1.6%. The pixelart row repeats one
yellow character, a mode of the class, not a copy.

Houses card `run_20261109_125713_bench_category-conditioned_legacy_recipe` (`--tier both --n 256
--baselines`, the 256 house-token samples):

| arm | BlockScore | appearance | geom_kid | mv_dino_kid | palette | enterable | dino nn pct |
|---|--:|--:|--:|--:|--:|--:|--:|
| **category model, house token** | **272** | **128** | **13.0** | **0.207** | 15.0 | 0.51 | 0.52 |
| canon_oriented | 297 | 157 | 14.2 | 0.254 | 6.1 | 0.61 | 0.43 |
| recanon s0 (T111) | 313 | 151 | 15.0 | 0.245 | 7.4 | 0.66 | 0.46 |

One Holm group (`c`) on both primaries. No gate fires.

**The legacy recipe makes category conditioning work, and training on 11 classes costs houses nothing.**
The pipeline's category model (T99) scored 355 on houses; this one scores 272 and sits in canon_oriented's
group, with its training-set proximity at real's level (0.52 against 0.51) where the houses-only models sit
closer (0.43 to 0.46). Its weak spots are palette (15.0, the worst of the three) and traversal (enterable
0.51). One seed. Next: the 20M version, since scale was the large lever on houses (T118).

## T122. Constrained decoding stacked with best-of-4 and the RFT model: appearance to 0.098, geometry at the ceiling, and a reward that saturates (2026-11-09)

**Setup.** The cross-face constrained decoder of T117 applied where T117 said it should stack: under
best-of-4 selection on `face_solidity` (k=4, `scripts/facade_constrained.py bon`), on the T73 RFT model
(`run_20261101_154120_rft_facades_joint_d4_20m_face_solidity_rsel_t07`), and both at once. All arms
T=0.7, top-k 40, seed 0, the T68 boxes, n=399, batch 32. One card put the three new arms beside the T117
grammar control and constrained arm, the T77 best-of-4 and T73 RFT arms, patchwork and the six-face
ceiling: `outputs/run_20261109_144658_bench_facade_constrained_decoding_stacked`. Per-arm outputs
(`summary.json`, `scores.parquet` for the best-of-4 arms, `samples.png`, `facades.png`, `faces.npz`) under
`outputs/run_20261028_125240_houses32_facades_joint_d4_20m/constrained_bon_face_solidity_k4_t0.7_n399`
and `<RFT run>/constrained_{constrain,bon_face_solidity_k4}_t0.7_n399`. Wall time 56, 9 and 36 minutes
for the three arms, the card 30 minutes; the whole chain ran 13:05 to 15:17 UTC.

**Sampler's own measures.** All three arms are consistent face sets by construction (reprojection
changes no cell), with 21% to 23% of the drawn cells written by the rule. The reward distributions
of the three best-of-4 arms are the part that matters:

| best-of-4 arm | `face_solidity`, all 1596 candidates (mean / median) | kept 399 (mean) | kept at 0.99 or above | median blocks kept |
|---|---|---|---|---|
| T77 base, free decoding | 0.625 / 0.744 | 0.861 | 0.018 | 697 |
| base, constrained | 0.690 / 0.835 | 0.914 | 0.073 | 550 |
| RFT, constrained | 0.871 / 0.911 | 0.971 | 0.371 | 496 |

**The card** (n=399; the base arms lose the handful of all-air draws; real values in the header).

| arm, T=0.7 | BlockScore | geom_kid | mv_dino_kid | n_components (2.7) | lcc_ratio (0.98) | n_blocks (904) | enclosed_air (0.114) | enterable (0.95) | twin rate (0.010) |
|---|---|---|---|---|---|---|---|---|---|
| grammar control (T117) | 679 | 19.0 | 0.299 | 33.7 | 0.810 | 689 | 0.008 | 0.316 | 0.000 |
| constrained (T117) | 589 | 16.5 | 0.166 | 12.9 | 0.812 | 541 | 0.016 | 0.285 | 0.015 |
| best-of-4 (T77) | 483 | 13.5 | 0.149 | 15.9 | 0.948 | 697 | 0.015 | 0.365 | 0.010 |
| **constrained best-of-4** | 497 | 13.9 | **0.098** | **7.7** | 0.922 | 618 | 0.083 | 0.316 | 0.005 |
| RFT + selection (T73) | 553 | 15.4 | 0.182 | 17.6 | 0.937 | 707 | 0.048 | 0.352 | 0.028 |
| **constrained RFT** | 552 | 15.4 | 0.147 | 10.4 | 0.902 | 621 | 0.079 | 0.323 | 0.038 |
| **constrained best-of-4 on RFT** | DQ | 14.8 | 0.304 | 4.3 | 0.971 | 555 | 0.236 | 0.251 | **0.271** |
| real@facades6 (ceiling) | 493 | 13.8 | 0.001 | 13.5 | 0.814 | 660 | 0.069 | 0.729 | 0.010 |

Holm-corrected paired permutation tests (10,000 permutations). On `geom_kid` the constrained best-of-4
arm, the T77 best-of-4 arm and the ceiling share group c; the constrained RFT arm, the RFT arm and the
disqualified arm sit in cd; the constrained arm alone in de and the grammar control in e. On
`mv_dino_kid` the constrained best-of-4 arm is alone in group c (0.098), separated from best-of-4 and
constrained RFT (d, 0.149 and 0.147), which are separated from RFT (e, 0.182); the grammar control and
the disqualified arm share f (0.299 and 0.304).

**Reading.**

*Constrained best-of-4 on the base model is the best learned facade arm on appearance and coherence,
and ties the ceiling on geometry.* `mv_dino_kid` 0.098 is a third below best-of-4 (0.149) and in a Holm
group of its own; `n_components` 7.7 is the first learned facade arm below the six-face ceiling's 13.5,
`enclosed_air` 0.083 is the closest any facade arm has come to real's 0.114, and the see-through cost
of T117 is mostly gone (618 blocks against 541 for the constraint alone), because selection on face
solidity prefers the candidates whose front face has no holes to empty columns with. Its BlockScore,
497 against best-of-4's 483, is the geometry pillar, where the two arms and the ceiling are one Holm
group: the facade geometry pillar is at its ceiling for all three, and nothing drawn as six faces can
pass 493 on BlockScore while the interior stays empty. The remaining gap is the ceiling's own, held by
`enterable` 0.73 and `walk_floors`, not by the faces.

*Constrained decoding of the RFT model adds appearance, not geometry.* BlockScore 552 against 553, the
same `geom_kid`, components 17.6 to 10.4 and `mv_dino_kid` 0.182 to 0.147 (separated). The fine-tune
already bought most of what the rule gives on the shell, and the rule removes what edge disagreement
the fine-tune left.

*Constrained best-of-4 on the RFT model is disqualified for self-duplication, and the samples say why.*
The twin rate (pairs within the set at IoU 0.8 or above) is 0.271 against real's 0.010, thirteen spreads
out; `mv_dino_kid` falls back to the grammar control's level (0.304); `enclosed_air` is double real's
and `bbox_fill` 0.455 against 0.288. The sheet is monoliths: solid single-material cuboids, grass and
dirt layer cakes, blank stone and sandstone blocks, with a real-looking house in about one cell in four.
The reward table shows the mechanism. `face_solidity` is a fraction of filled face cells, and three
levers push the same way: the RFT model was fine-tuned on it, the constraint forces a fifth of the cells
to agree with faces already drawn, and selection takes the most solid of four. Under one lever 1.8% of
kept builds are at 0.99 or above, under two 7.3%, under three 37%: the reward saturates, and at its
ceiling the optimum is a filled box, which every model draws the same way. The two-lever stacks stay
on the right side of the gate (twin rates 0.005 and 0.038) because neither reaches the plateau.

**What follows.** Two conclusions for the facade track. First, `face_solidity` is spent as a reward:
any further stacking needs a counterweight (an enclosed-air or interior term, palette entropy, or
the novelty gate itself) before another solidity lever is added; a selection reward that saturates
on a filled box was always going to end here once the base distribution got solid enough. Second,
the geometry pillar is at the six-face ceiling for the best arms, so the facade track's BlockScore
is now bounded by the representation, not the generator, and the next lever for the number is an
interior stage (floors and openings behind the shell) rather than another face lever. The relaxed
see-through rule and the face order of T117 remain open but are worth less than they were, since
selection already removes the skeletons.

## T123. Functional ground truth and design decisions rebased on the exact 40-pair test (2026-11-09)

`outputs/run_20261109_152704_rankflips_functional_rebase` (notes §91). CPU, cached features.

| quantity | on the subsampling test (T114/T106) | on the exact test |
|---|---|---|
| reversed pairs judged | 76 | 40 |
| walk-through sides with MV-KID / V-KID | 60 / 16 (21%) | **33 / 7 (17.5%, CP [7, 33])** |
| non-facade pairs, V-KID side | 1 of 38 | **0 of 18** |
| facade pairs, V-KID side | 15 of 38 | 7 of 22 (5 = BrickNet over facade) |
| FD_noise / FD_kid / FD_occ, V-KID side | 6 / 20 / 18 of 76 | 2 / 9 / 8 of 40 |
| Spearman FD vs MV / V / max of two | 0.67 / 0.16 / not computed | 0.67 / 0.16 / 0.17 [-0.18, 0.45] |
| 34 decisions: agree / MV-only / V-only / neither | 13 / 12 / 3 / 6 | 6 / 14 / 1 / 13 |
| worked examples, V-KID Holm p | < 1e-100 (normal) | 0.039 (permutation floor) |

Verdict: the functional result strengthens under the calibrated test; V-KID detects interior change
but does not rank interior function on generators.

## T124. A second 20M seed: geometry replicates (8.1 against 9.1), but it fails the memorization gate (2026-11-10)

Card `run_20261109_224721_bench_canon_oriented_recipe_at_20M_second_seed` (`--tier both --n 256 --baselines`).
`run_20261109_144738_recanon20m_s1_native` is T118's command with torch seed 1 (split still pinned to seed 0).
Final train loss 0.103 (seed 0: 0.112). Samples as T118 (T=1.0, top_k 40, 256, sample seed 0). Recipe:
scratchpad `recanon/legacy20m_s1.sh` (log `outputs/logs/legacy20m_s1.log`).

| arm | BlockScore | geom_kid | mv_dino_kid | dino_nn_percentile | voxel_nn_iou | voxel_diversity | dino_dup_rate |
|---|--:|--:|--:|--:|--:|--:|--:|
| 20M, seed 0 (T118) | 190 | 9.1 [8.1, 10.2] | 0.159 | 0.359 | 0.390 | 0.878 | 0.000 |
| **20M, seed 1** | **DQ** | **8.1 [7.3, 8.9]** | **0.157** | **0.351** | 0.405 | 0.866 | 0.004 |
| canon_oriented (5M) | 297 | 14.2 | 0.254 | 0.433 | 0.360 | 0.885 | 0.000 |
| 5M recreation, seed 0 / 1 | 313 / 319 | 15.0 / 15.3 | 0.245 / 0.208 | 0.464 / 0.412 | 0.361 / 0.366 | 0.885 / 0.886 | 0.000 |
| real test | 0 | -0.1 | 0.001 | 0.508 | 0.375 | 0.898 | 0.008 |

Seed 1 is disqualified by the `dino_nn_percentile` gate (memorization: closer to training data than held-out real,
+3.1 spreads). Seed 0's 0.359 passed; seed 1's 0.351 does not, so the gate threshold sits between the two seeds.
Holm on `geom_kid`: the two 20M seeds share group `c`, both ahead of every 5M arm (seed 1 vs canon_oriented
-6.18, vs recanon s0 -6.95, all p_adj 0.012). Same on `mv_dino_kid` (group `c`).

**The 20M quality gain replicates; the recipe sits on the memorization boundary.** Geometry and appearance at 20M
are a seed-stable 5 to 7 units better than 5M, so T118 was not a lucky seed. But both seeds sit at the gate's edge
(nn percentile 0.35 to 0.36, real 0.51, 5M 0.41 to 0.46): the extra capacity buys realism partly by moving toward
the training set. There are still no voxel duplicates and nn IoU (0.39 to 0.41) is near real's 0.375, so this is
proximity, not copying. T118's 190 stands on the card but is one gated seed of two. Candidate fixes, untested:
an earlier checkpoint (train loss 0.10 is a third of 5M's), dropout above 0.1, or fewer epochs.

## T125. Agentic and LLM one-shot arms at n=256 with distinct prompts: all four fail the twin gate (2026-11-10)

Card `run_20261110_095530_bench_agentic_and_LLM_one-shot_arms_at_n_256` (`--tier both --n 256 --baselines`).
Each agentic arm is 256 builds from 256 distinct prompts: T51's 133 test GrabCraft prompts
(`houses32_test_grabcraft.txt`) plus 123 new ones from val GrabCraft titles
(`configs/agentic/prompts/houses32_val_grabcraft.txt`, disjoint from test, same format). dream-caption, dream and
classic-matched reuse T51's test builds and add `run_20261109_165950_bench_houses32_agentic_val`; `llm-oneshot`
(recipe `legacy-oneshot`: gpt-5-mini, one example, no plan, repair or critique) is new on both halves
(`run_20261109_130512_bench_houses32_oneshot_test`, `run_20261109_145342_bench_houses32_oneshot_val`).
Spend: one-shot $1.39 + $1.31, agentic val $12.50, plus $0.46 on a stopped first chain; **$15.66 total** against the
$25 + $10 caps. Recipe: scratchpad `recanon/agentic256.sh` (log `outputs/logs/agentic256.log`).

| arm | BlockScore | geom_kid | mv_dino_kid | voxel_twin_rate | voxel_diversity | dino_nn_percentile |
|---|--:|--:|--:|--:|--:|--:|
| 20M legacy, seed 0 (T118) | 190 | 9.1 | 0.159 | 0.008 | 0.878 | 0.359 |
| canon_oriented | 297 | 14.2 | 0.254 | 0.000 | 0.885 | 0.433 |
| pipeline P2 (T119) | 330 | 15.8 | 0.253 | 0.000 | 0.888 | 0.461 |
| dream-caption | DQ | 15.2 | 0.223 | 0.285 | 0.777 | 0.463 |
| dream | DQ | 17.4 | 0.349 | 0.309 | 0.784 | 0.490 |
| classic-matched | DQ | 18.6 | 0.416 | 0.410 | 0.821 | 0.582 |
| llm-oneshot | DQ | 18.9 | 0.434 | 0.406 | 0.836 | 0.599 |
| real test | 0 | -0.1 | 0.001 | 0.008 | 0.898 | 0.508 |

Every LLM arm is disqualified on `voxel_twin_rate` (near-copies at IoU >= 0.8 inside the set; +13.9 to +20.1
spreads). Holm on `geom_kid`: dream-caption (`de`) ties canon_oriented and P2 (`d`); dream `ef`; classic-matched and
one-shot `f`, behind every learned arm. On `mv_dino_kid` dream-caption (0.223, alone in `d`) beats both 5M arms
(`e`) and trails only the 20M model (`c`).

**Distinct prompts do not remove the twins: the LLM arms repeat themselves across different requests.** T51's
n=133 rates (0.21 to 0.32) were not an artifact of repeated prompts; with 256 different titles, 29 to 41% of builds
still have a near-copy in the set (real 0.8%), and diversity is 0.78 to 0.84 against real's 0.90. The LLMs reuse
one house template per style. The dream loop halves the gap to real on appearance (one-shot 0.434 to dream-caption
0.223) and gets geometry to canon_oriented's level, but adds no diversity (dream-caption's twin rate 0.285 vs
one-shot 0.406, both far past the gate). Ungated, dream-caption would sit beside the 5M learned arms, behind 20M.
Untested: a diversity rider (`--variation`) or temperature to break the template.

## T126. A function pillar for BlockScore: it never binds, and held out it does not beat MV-KID (2026-11-11)

`outputs/run_20261111_021115_function_pillar` (notes §92). FD (five walk-through measures, W1 / robust
spread) as a pillar: z = (FD − FD_REAL) / 0.019 (sd of 200 real-vs-ref draws, n = 256); e = fraction
of the way to Random-Voxels (FD 4.09). Spearman over 40 generators / agreement on the 40 reversed pairs.

| variant | vs FD (circular) | LOMO mean ρ | build split ρ | build+measure split ρ |
|---|---|--:|--:|--:|
| MV-KID | 0.67 / 33 | 0.42 | 0.66 | 0.43 |
| V-KID | 0.16 / 7 | −0.14 | 0.15 | 0.23 |
| BS2 = max(MV, V) | 0.17 / 4 | −0.11 | 0.17 | 0.22 |
| function pillar | 1.00 / 40 | 0.41 | 0.97 | 0.30 |
| BS+F (z) | 0.17 / 4 | −0.11 | 0.17 | 0.22 |
| e3 = max(MV, V, F) e | 0.19 / 7 | −0.18 | 0.18 | 0.39 |

Function is the worst pillar for 0/40 (z), 2/40 (e). Leaderboard order unchanged (draft Table 4 with
function columns in the drafts dir). Probes: Fill z 58.6, Hollow z 26.5 but reach-only 0.7 (enterable
0.95 = real), Shuffle/Monochrome occupancy-only 0.0. Weight sweep: F binds on 35/40 at w = 4, held-out ρ
then 0.39 to 0.41.

## T127. Main-figure data: MV-KID vs V-KID effect size per leaderboard arm (2026-11-11)

`outputs/run_20261111_021721_main_figure_effect_sizes`. e with half-sampling 95% CIs (B = 400), FD
colour; Fill 0.029 / 0.78, Hollow 0.023 / 0.60, VOXEL-AR clean 0.38 / 0.62, null 95th pct MV 0.032.
Figures `fig_effect_plane.*`, `fig_rank_scatter_40.*` (PDF, SVG with live text, PNG) in the drafts dir.

## T128. BrickGPT effect-size ratio: "2.2" depends on the anchor; the direction holds (2026-11-11)

`outputs/run_20261111_022236_brickgpt_ratio_robustness`. V-KID exact; MV-KID via OSMesa CPU render (all
arms, one protocol) and cached EGL split features.

| anchor | ratio e_V / e_MV [95%] |
|---|---|
| SINGLE-MODE pool[0], card (EGL) | 2.18 |
| SINGLE-MODE pool[0], OSMesa | 2.37 [1.82, 3.22] |
| RANDOM-VOXELS (LEGO), OSMesa | 3.17 [2.46, 4.31] |
| CHUNK-DELETE-20, OSMesa | 1.58 [1.07, 2.31] |
| SINGLE-MODE from each of 253 builds (EGL) | median 2.28, 95% [1.64, 3.23], min 1.53 |

## T129. Training-seed variance: half the significant design decisions are inside seed noise (2026-11-11)

`outputs/run_20261111_022723_training_seed_variance`. Seeds: recanon_s0/s1 + canon_oriented,
recanon20m_s0/s1, recanon_wcpre_s0/s1 (n = 256). σ_total 13.0 MV floors [7.8, 37.4], 36.4 V floors
[21.8, 104.7] (df 4); sampling-only control 6.2 / 15.6. Decisions (Holm over 34): MV-significant
20 → 10, V 7 → 2; verdicts agree 6 → 2, MV-only 14 → 8, V-only 1 → 0. Reversed pairs beyond the 95%
seed band on both metrics: 28/40 (Bonferroni-780: 4). Without the world-crop pair σ_V = 11.0 and all
V decisions stand.

## T130. Leak bins under the deployed renderer, paired test (2026-11-11)

`outputs/run_20261111_023229_leak_bins_deployed`. Spearman leak vs DINO shift: deployed 0.656
(0.767 with deployed leak), atlas fixed 0.959. Bins (atlas-fixed leak, deployed features), z [half-
sampling 95%], paired p, Holm: none −0.10 [−0.50, 0.28]; q1 +0.02; q2 +0.13; q3 +0.29 [−0.46, 0.95]
p 0.11; q4 −1.49 [−3.31, 0.37] p 0.012, Holm 0.060. Binned by deployed leak: all |z| ≤ 0.26, p ≥ 0.15.

## T131. Table 4 rank intervals with the m-of-n rescale and floor Monte-Carlo error (2026-11-11)

`outputs/run_20261111_024303_rank_intervals_rescaled`. z ranks (rescaled + floor MC): REAL 1,
VOXEL-AR contaminated 2, clean 3, VOXEL-AR-U 4–5, MATERIAL-SHUFFLE 4–7, PATCH-STITCH 5–7, FILL 5–8,
MONOCHROME 7–9, TRAIN-COPY+NOISE 8–9, RANDOM-VOXELS 10. Flips Shuffle/Patch 0.29, Patch/Fill 0.41
(unrescaled 0.19/0.19). Floor MC factor 95% [0.81, 1.16]. e2: VOXEL-AR clean ties Shuffle (3–4).
Ungated: Dream-Cubed 2–3 (z), 2 (e2).

## T132. Gate audit from stored ladders: S6 blocking, G1 dropped, diversity bar, twin margin (2026-11-11)

`outputs/run_20261111_024632_gate_audit_stored`. S6 (Fill > 3 sd) blocking: Minecraft render passes
KID, MMD-RBF → none; LEGO render KID, 1−density → none; geometry V-KID class unchanged; ablation set
60 → 59 (connected parts alone drops). Without G1: 1 of 112 verdicts changes (rand25_17). Diversity bar
10 since commit 16b95f8 (2026-09-07). Fill twin rate 0.0664 [0.039, 0.098] vs bar 0.0678.

## T133. Fill-Interior under MV-KID: one test, one interval (2026-11-11)

`outputs/run_20261111_024856_fill_interior_inference`. Δ = +0.35 floors; paired swap p 0.057; unpaired
label (Holm family) 0.87. 95% CI paired fixed ref [−0.23, +0.94]; random ref [−0.40, +1.10]; unpaired SE
[−3.44, +4.15]. TOST ±2 floors: equivalent with either paired SE, not with the unpaired SE. Direct paired
MMD vs source: e 0.018, p < 1e-4 (Hollow 0.007, p 0.002). e vs ref 0.029, Hoeffding [0.00, 0.058]
(half-sampling [0.00, 0.08]).

## T134. GPT-4o judge with position removed (2026-11-11)

`outputs/run_20261111_015408_vlm_judge_debias` (no new calls; records of `run_20261102_214026_vlm_judge`). House-cluster bootstrap 95%.

| pair | view | agree | raw | order-consistent (pairs) | BT + position |
|---|---|--:|--:|---|---|
| real vs Fill | exterior | 0.13 | 0.53 | 0.71 [0.46, 0.92] (17) | 0.61 [0.48, 0.78] |
| real vs Hollow | exterior | 0.11 | 0.52 | 0.64 [0.38, 0.90] (14) | 0.57 [0.45, 0.71] |
| both | exterior | 0.12 | 0.52 | 0.68 [0.50, 0.84] (31), p 0.07 | 0.59 [0.50, 0.70] |
| real vs Fill | roof off | 0.54 | 0.70 | 0.87 [0.78, 0.94] (69) | 0.70 [0.65, 0.76] |
| real vs Hollow | roof off | 0.53 | 0.73 | 0.94 [0.85, 1.00] (34) | 0.80 [0.72, 0.90] |
| both | roof off | 0.54 | 0.71 | 0.89 [0.83, 0.95] (103) | 0.73 [0.68, 0.78] |

System BT + position: ranks unchanged (real 6th of 11), Spearman MV 0.77, BS 0.59, V 0.17. Weakens "at chance" to "uninformative, leaning real"; the cutaway contrast holds.

## T135. BlockScore worst pillar at n=128 vs n=256; effect-size BlockScore (2026-11-11)

`outputs/run_20261111_015636_blockscore_pillar_stability`. Worst pillar: 0 of 13 shared arms change. Spearman BS(128) vs BS(256) 0.98. Effect-size BlockScore (max of appearance and geometry as fractions of Random-Voxels): Spearman with z BlockScore 0.96 (n=256), 0.90 (n=128); n=128 vs 256 0.99; submissions 0.56 / 0.62 / 0.69 / 0.71 in z order. Worst pillar differs from z for Patch-Stitch, Hollow-Box, Single-Mode, Dream-Cubed (geometry in e). Five pillars in e: Spearman 0.61, palette dominates material/parametric controls.

## T136. Near-twin gate audit (2026-11-11)

`outputs/run_20261111_015843_twin_gate_audit`. Bar 0.068 (n=256). Real held-out draws (12 at n=128, 12 at n=256, test/val/test+val): mean 0.003 / 0.008, max 0.023, 0 over bar. 40 rank-flip generators: 6 gated (classic-matched 0.316, dream 0.301, dream-caption 0.211, rft_m2b_sft_s1_lcc26_rsel_t0.7 0.109, llm_control 0.070, m2b_sft_s1_t0.7 0.070); 22 of 35 learned at 0, max 0.109. IoU 0.95: Dream-Cubed 0.191, templates 0.638, real and learned <= 0.016. Board without the gate: Dream-Cubed 2 (165.71), contaminated AR 3, gabled 11, shell_box 12. **Contradicts "every model we trained reads 0.000"** (two T=0.7 voxel-AR arms exceed the bar). The bar/floor predate Dream-Cubed (2026-09-07); the statistic and IoU 0.8 postdate its audit.

## T137. Colour-free Uni3D / ULIP-2 on interior-inclusive points through the gates (2026-11-11)

`outputs/run_20261111_024010_uni3d_colorfree_gates` (bench_headtohead ladder + board; colour rows reproduce tab:headtohead).

| row | Fill | Hollow | Shuffle | Mono | max rot | gates | board Fill | rho V | rho MV |
|---|--:|--:|--:|--:|--:|---|--:|--:|--:|
| V-KID | +111.4 | +71.0 | 0 | 0 | 0 | pass | +444.4 | 1 | 0.27 |
| Uni3D all, grey | +20.6 | +10.3 | +0.6 | -0.2 | 1.2 | G6 | +30.4 | 0.75 | 0.49 |
| Uni3D vol, grey | +43.5 | +11.8 | +0.2 | +0.0 | 1.7 | G6 | +102.2 | 0.81 | 0.45 |
| ULIP-2 all, grey | +6.0 | +3.3 | -0.0 | +0.0 | 0.2 | **pass** | +13.5 | 0.72 | 0.45 |
| ULIP-2 vol, grey | +16.7 | +11.0 | -0.2 | -0.6 | 0.6 | **pass** | +57.8 | 0.73 | 0.47 |

**Contradicts "only V-KID passes every gate"**: colour-free ULIP-2 passes on both samplers, at 8 to 33 times smaller Fill separation on the board.

## T138. Objaverse 568-object probe, seed 1, CPU-only rerun (2026-11-11, running)

`outputs/run_20261111_022342_mesh_probe3d_568_s1cpu`, `scripts/run_mesh_probe_3d_cpuonly.py --objects 600 --n 170 --seed 1 --device cpu --workers 4 --emd-workers 8 --emd-from outputs/run_20261106_231005_mesh_probe3d_568_s1/emd.npz`. Same design as T109 (seed 0); OSMesa renderer and CPU DINOv2, so no GPU. Voxel rows: unit longest side, 32 cells per unit (surface cells + Fill's solid points), visibility on a 48-per-unit grid. A waiter (`scripts/mesh_probe_seed_compare.py`) writes `seed_compare.md` (s0 vs s1 per row) as soon as `embeddings.npz` exists (3D-encoder rows) and again when `results.json` lands (all rows). Seed 0 took about a day end to end on this machine.

**Addendum (2026-11-11): the same arms against the shells of the real houses.** The geometry
descriptor sees interiors (sealed and open interior ratios, the player's room, the 2x2x2 occupancy
patterns of a filled volume), and a facade arm has none by construction. To separate the generator's
error from the representation's, `scripts/facade_shell_reference.py` scores each arm's descriptor
against the card's reference set reduced to its six-face shell (`facade.shell`, all faces), with the
standardizer refit on the shells. Same estimator, same arms, same n; the two columns are on
different scales and only order within a column is comparable.

| arm, T=0.7 | geom_kid against real houses (the card) | geom_kid against their six-face shells |
|---|---|---|
| real test houses | -0.07 | 18.20 |
| real@facades6 (ceiling) | 13.76 | -0.08 |
| patchwork | 27.72 | 17.14 |
| grammar control | 19.01 | 11.26 |
| constrained | 16.47 | 5.39 |
| best-of-4 | 13.50 | 8.16 |
| constrained best-of-4 | 13.88 | 3.95 |
| RFT | 15.45 | 9.08 |
| constrained RFT | 15.44 | 4.34 |
| constrained best-of-4 on RFT | 14.79 | 11.42 |

Three things the card could not show. The representation gap is the larger of the two: real houses
are 18 from their own shells, and the ceiling is 14 from real houses, while the best generator is 4
from the shells. The order of the arms changes: against real houses best-of-4 beats constrained
decoding (13.5 against 16.5), against the shells constrained decoding beats best-of-4 (5.4 against
8.2), and the same holds for RFT against constrained RFT (9.1 against 4.3). Free decoding's face
disagreement fills volume behind the faces that the full-house reference reads as interior, so the
card rewarded facade arms for an artefact and charged the consistent shells for being hollow.
And the disqualified arm is bad geometry as well as duplicated geometry: 11.4 against the shells,
the grammar control's level. Reading: for developing the facade generator the shell column is the
right number, and it says the generator's remaining shell error is about a third of what it was
before the rule; for the leaderboard the full-house column stays the number, because the goal is
a solid enterable house and the ceiling arm exists to keep the representation's cost visible.
