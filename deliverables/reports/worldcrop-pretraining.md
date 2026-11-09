# World-crop pretraining for block generators

Internal technical report, 2026-11-04. Built from results.md (T51, T54 to T60, T78, T82), notes.md (§43, §45, §46, §48), research.md (§F, §I.5, §I.7), docs/pipeline.md, the corpus tier files and the run directories named below. No new experiments were run for this report. One count (the SFT-in-pretraining overlap in §3.3) was computed for this report from `data/minecraft/corpus/dedup.parquet` and the tier id lists.

## 1. Thesis

A terrain-gated, deduplicated corpus of about 650k structures cropped from archived Minecraft worlds is a pretraining source for block generators: pretraining on it and then fine-tuning on 1,863 curated houses lowers held-out likelihood and improves the geometry of samples against training on the houses alone.

## 2. Contributions

- **A world-cropping pipeline over the Minecraft Archive Project.** HTTP range reads of archive.org month zips, flat-world and heuristic routes, ore-family and vanilla-structure filters, re-tiling at tile edges (T51, research.md F.6, F.7). Pilot cost is 12.5 s per archive; 915,853 crops from 47 months are on disk (T55).
- **A learned terrain gate and a corpus layer.** A 295-feature gradient-boosted gate that rejects 8 of 8 terrain crops and 3 of 106 built crops on a 192-crop hand-labelled hold-out (T54, v4), followed by canonical-hash and MinHash dedup that folds 29.9% of 926k rows and exposes vanilla villages and mineshafts by cross-world recurrence and composition (T55, T55.1). Tiers: pretrain-all 648,845, pretrain-clean 548,732, anneal 275,200 builds (T56).
- **A controlled transfer result.** At 5M parameters, three seeds, early stopping and n=256 scoring, pretrain then SFT beats a scratch arm trained identically on every seed, on holdout NLL and on BlockScore (T60). At 20M, one seed, holdout NLL falls to 0.286 (T78).
- **Two protocol findings that apply to any pretrain-then-fine-tune study on small 3D corpora.** A dropout change at `init_from` erases the pretrained model before the first update (T59), and matched-compute comparisons without early stopping reward the arm that memorises (T59).

## 3. Evidence on disk

### 3.1 Solid (three seeds, n=256, early stopped)

| T | comparison | numbers as written | runs |
|---|---|---|---|
| T60 | 5M pretrain + SFT vs 5M scratch, dropout 0 both | holdout NLL 0.479 / 0.420 / 0.350 (mean **0.416**) vs 0.565 / 0.555 / 0.509 (mean 0.543); BlockScore 374.8 / 323.2 / 324.1 (mean **340.7**) vs 502.1 / 444.9 / 333.7 (mean 426.9); geom_kid mean 18.0 vs 22.6; dup rate 0.000 both | `outputs/run_20261015_225212_m2b_sft_s0`, `run_20261015_231328_m2b_scratch_s0`, `run_20261015_235516_m2b_sft_s1`, `run_20261016_002201_m2b_scratch_s1`, `run_20261016_010438_m2b_sft_s2`, `run_20261016_013050_m2b_scratch_s2` |
| T60 | token efficiency | selected SFT checkpoints at 3,000 to 8,500 steps; 0.68B to 0.85B total tokens against scratch 1.08B | same |
| T60 | regularisation effect | scratch climbs to 1.34, 1.68, 1.81 nats by step 36,000; SFT seed 2 within 0.02 nats of its minimum from step 2,000 to 16,000 | same |
| T54 | terrain gate v4 | terrain rejected 8/8, built rejected 3/106, fragment rejected 0/65 (192-crop hold-out) | `blockgen/data/worldcrop/gate.py` |
| T55 | dedup | 277,397 rows folded (29.9%), 648,917 clusters; top 24 cross-world clusters all vanilla village pieces | `data/minecraft/corpus/dedup.parquet`, `sheet_top_clusters.png` |

### 3.2 Single-seed, pilot, or confounded

| T | result | why it is weak |
|---|---|---|
| T58 | v2: pretrain + SFT NLL 0.426 vs scratch 0.543; BlockScore 95.50 vs 138.68 | one seed, n=64, pretrained arm saw 3.6 times the tokens; scratch BlockScore moved 63 points between v1 and v2 |
| T59 | M2 pilot | invalid: seeds 0 and 1 lost their pretraining to a dropout switch; only seed 2 is a valid pair (0.375 vs scratch best 0.372, a tie) |
| T60 | **the open issue** | the M2 scratch arm at dropout 0.1 reached best NLL 0.416 / 0.400 / 0.372 (mean 0.396). Against it, pretrain + SFT (mean 0.416) wins one seed of three. On likelihood the result is a tie against the best-regularised scratch arm tried so far |
| T60 | board comparability | the six M2b cards predate the T64 descriptor change and are not on the current board |
| T78 | 20M, seed 1 only | holdout NLL 0.286 (step 1,500 of 16,000; 0.797 at the end); BlockScore 346 raw vs 386 for 5M raw; 20M vs 5M raw on geom_kid 1.9 [1.1, 2.8], p_adj 0.09 (trend only). No 20M scratch arm exists. The 20M pretrained model alone scores holdout 0.353 before any SFT (`outputs/run_20261102_142710_m2_20m_pretrain_s1/metrics.jsonl`, step 20,000), already below every 5M SFT seed |
| pending | 20M short SFT (3,000 steps), `outputs/logs/sft20m_short.log`, run `outputs/run_20261104_004730_m2b_20m_sft3k_s1` | training finished: select=best step 1,750, holdout 0.2857 (first 256), 0.3040 over all 736. This reproduces T78 (0.286) and does not improve it. No samples, `summary.json` or scorecard on disk yet. The pretrain-mix SFT arm has no run directory yet |

### 3.3 A confound the paper has to remove

The `anneal` tier that every pretraining run used is not world crops alone. Its `by_source` count is worldcrop 270,192, grabcraft 2,294, 3dcraft 1,484, templates 787, text2mc 309, text2mc_schem 134 (`data/minecraft/corpus/tiers/anneal.json`). Joining the tier id lists through the dedup clusters, **1,834 of the 1,863 `sft-houses32` training houses have a cluster representative inside `anneal`**, so the pretrained arm has already seen the SFT set (mostly through the `all_32` cache copies) for about seven epochs. The 798 `eval-holdout` builds have zero clusters in `anneal`, so there is no test leakage. The transfer claim as stated (world crops transfer) is therefore not yet isolated from "more epochs on the curated houses plus 3k other curated builds". A world-crop-only pretraining arm is the first ablation (§5).

## 4. Novelty and closest prior work

- **text2mc** (UCF 2024; references.md) already cut 19,394 `.schem` crops out of 9,138 PlanetMinecraft world downloads with a chunk-granular `InhabitedTime` plus natural-block-list rule and DBSCAN, with no vanilla-structure masking (research.md F.1). It is the direct precedent for world cropping and the baseline the gate must beat on the same worlds. That comparison has not been run.
- **Dream-Cubed** (Merino et al., arXiv 2604.22847; NeurIPS 2026 per research.md I.7) trains diffusion on about 359k 32³ human chunks from six maps by one builder plus 1.67M natural chunks, CC BY-NC (research.md F.2). It is the scale precedent and the strongest external generator on our own board: BlockScore 165.71 on the n=256 leaderboard of record (T82), against 340 to 386 for our 5M and 20M voxel arms on their cards (T78). The cards use different descriptors and are not directly comparable, but a reviewer will put the two numbers side by side. Dream-Cubed does not report a pretrain-then-fine-tune transfer study on curated houses (as recorded in references.md; not re-verified against the paper for this report).
- **The pretrain, anneal, SFT recipe** is the standard LLM stack (docs/pipeline.md). The novelty is its application to a mined 3D block corpus with a learned built-vs-terrain gate and recurrence-based removal of procedurally generated content. research.md F.4 records that no prior work has a learned built-structure detector for world saves; this was a web survey on 2026-10-07 and has not been rechecked since.
- **Not verified:** whether any 2026 paper pretrains a block or voxel generator on MAP or MCGS data. research.md I.7 lists no such paper, but its literature pass targeted evaluation, not data.

## 5. Gaps a reviewer would demand

Costs assume the shared 16 GB 5070 Ti and the measured times: 5M pretrain 20,000 steps in 31.6 min (T58), 5M SFT or scratch about 3 min per 1,000 steps (T57), 20M pretrain 49 min and 20M SFT 3,000 steps in 12.7 min (T78, log), 20M n=256 sampling about 30 min (scaled from 1,024 draws in 7,424 s, T78).

| # | experiment | cost | what sinks the claim |
|---|---|---|---|
| 1 | **Tuned scratch baseline**: dropout {0, 0.1, 0.2} by weight decay {0.01, 0.1}, chosen on a validation split separate from `eval-holdout`, three seeds, early stopped; pretrain + SFT given the same search (needs pretraining at the matching dropout, T59 rule) | 5M: about 15 to 20 GPU-hours; 20M: about 10 more | tuned scratch matches pretrain + SFT within seed spread on NLL and geom_kid at 5M (T60 already shows a likelihood tie against dropout 0.1) |
| 2 | **World-crop-only anneal**: rebuild `anneal` with `source == 'worldcrop'` and no curated clusters, three seeds at 5M | about 3 GPU-hours plus a shard rebuild (CPU, under an hour) | the gain shrinks to zero once the curated builds and SFT-house duplicates are out of pretraining (§3.3) |
| 3 | **20M, three seeds, both arms**, with the short SFT schedule | about 5 GPU-hours plus three tuned 20M scratch runs | 20M scratch reaches 0.29 holdout NLL; the gap closes with scale |
| 4 | **Corpus ablations**: pretrain-all vs pretrain-clean vs anneal (gate and dedup value), and text2mc crops or Dream-Cubed human chunks as an alternative pretraining corpus at matched tokens | about 6 GPU-hours for the tiers; Dream-Cubed adds a 133 GB download | the ungated or the Dream-Cubed corpus transfers as well, so the gate and the archive are not the reason |
| 5 | **Token scaling curve**: SFT from the 5k, 10k, 15k, 20k pretraining checkpoints already saved for 20M | about 1 GPU-hour plus scoring | flat curve: pretraining length does not matter, so the effect is regularisation, not knowledge |
| 6 | **Re-score all arms on one card** at n=256 under the current pinned descriptor, with Holm groups | about 2 GPU-hours of sampling, CPU scoring | pretrain + SFT and tuned scratch share a Holm group on geom_kid |
| 7 | Gate vs text2mc's `world2vec` on the same worlds, precision on a labelled sheet | CPU, a day of labelling | text2mc's rule matches the gate's precision |

## 6. Venue fit and timeline

- **CVPR 2027 main (Nov 16, enrollment Nov 10): not realistic.** Twelve days remain, the GPU is shared with the evaluation paper, gaps 1 to 3 are each a confound a reviewer finds on the first read, and the evaluation paper needs the same author time. research.md I.2 reached the same verdict on 2026-10-28.
- **Dual-submission risk with the evaluation paper.** The cvpr27 sections do not mention pretraining, M2b or the archive corpus, and the T82 leaderboard of record carries no M2b arm. The overlap is the benchmark itself: this paper scores with BlockScore, geom_kid and the houses32-v1 protocol introduced there. That is citation, not duplication, as long as this paper does not re-present the metric validation. If the evaluation paper later adds pretrained arms to its board, the arms should be cited from there and not reported as new here.
- **CVPR 2027 workshop:** plausible fit once gaps 1 and 2 are run (deadlines are typically March to April; not checked for 2027).
- **ICCV 2027 (deadline estimated early March 2027, unconfirmed):** realistic only if gaps 1 to 5 come back positive at 5M and 20M. A data-and-pretraining paper with an unreleasable corpus is a weaker main-track pitch than a method paper.
- **NeurIPS 2027 Datasets and Benchmarks (~May 2027, unconfirmed): poor fit.** The corpus cannot be released (MAP has no rights grant, research.md F.3), and D&B reviews the data. A fit exists only if a CurseForge release tier with per-item licences is built (F.3 recommendation 2).
- **arXiv technical report now:** reasonable for the pipeline and gate (T51, T54 to T56) as a citable artefact, but the transfer section should wait for gaps 1 and 2. Publishing the 5M result with the dropout-0 scratch arm invites the exact objection in T60.

Release story under F.3: code, manifest schema, tier filters, gate weights and model weights; no block grids from MAP.

## 7. Paper outline (8 pages)

1. **Introduction** (1 p): curated block corpora are thousands of builds; archived worlds hold hundreds of thousands; the question is whether mined structures transfer.
2. **Related work** (0.75 p): text2mc, Dream-Cubed, 3D-Craft, MineAnyBuild, LLM data pipelines (dedup, quality gates, annealing).
3. **Mining structures from worlds** (1.5 p): routes, filters, terrain gate, dedup and recurrence, kinds, tiers, yield table.
4. **Training protocol** (0.75 p): tokenizer, tiers, pretrain then SFT, early stopping, the dropout and matched-compute findings.
5. **Transfer results** (2 p): 5M and 20M, three seeds, tuned scratch, NLL and BlockScore with Holm groups.
6. **Ablations** (1.25 p): world-crop-only vs mixed anneal, gate and dedup tiers, alternative corpora, token scaling.
7. **Licensing, limitations, release** (0.5 p): F.3 statement, what is released.
8. **Conclusion** (0.25 p).

Figures:
1. Pipeline schematic with the yield funnel (926k crops, 29.9% duplicates, gate, tiers) and a sample sheet of gated crops beside rejected terrain and village clusters.
2. Holdout NLL and BlockScore against total tokens for scratch (tuned) and pretrain + SFT at 5M and 20M, three seeds, with the scratch overfitting curve visible.
3. Matched sample sheets: tuned scratch vs pretrain + SFT at 20M (T58 sheets show walls and roofs against blobs; this needs regenerating on the final arms).

## 8. Verdict

**Strength 2 of 5.** The pipeline and corpus are solid engineering with clean numbers, and the T60 result is clean within its protocol. The transfer claim is confounded twice: the scratch arm was not tuned (a dropout 0.1 scratch arm ties on likelihood) and the pretraining tier contains 1,834 of the 1,863 SFT houses plus about 3k other curated builds. The single most important next experiment is the **world-crop-only pretraining arm against a holdout-tuned scratch arm, three seeds at 5M**, about one GPU-day; if pretraining still wins there, the paper exists.
