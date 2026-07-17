# BlockGen — Results

Tables, figures, and ablations backing the paper. Methods/repro live in
[`notes.md`](notes.md). Each result notes **which cache / subset** it ran on and
**when**, because the dataset is being migrated (legacy `data/raw` → labeled
tfrecord cache) and numbers are not comparable across caches.

_Last updated: 2026-07-02._

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
> models, representations (with F6/F7), experiments, and the LEGO roadmap.

---

## Open result slots (fill as we run)
- [x] Re-run T4 on the **labeled** curated subsets (→ **T5**, run `20260630_220830`).
- [ ] Ablation: factored `(family,variant)` embeddings vs flat — rare-block generalization, val loss, sample quality.
- [ ] Ablation: validity gate on/off — validity_rate, diversity cost, reject rate.
- [ ] Category-conditional vs unconditional — per-category validity & novelty.
- [ ] Distributional metric (FID-analogue over occupancy/feature space) train vs samples.
