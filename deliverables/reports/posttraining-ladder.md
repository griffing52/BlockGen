# Post-training block generators under an honesty rule

Internal technical report, 2026-11-04. Evidence: results.md T60, T68, T70 to T74, T77, T78; notes.md §62 to §64; docs/pipeline.md §7; research.md §G, §I. No new experiments.

**Thesis.** When a block generator's verifier is barred from the metric's own inputs, the inference-time rung carries most of the gain. Best-of-N on a non-pillar reward is the strongest single rung. A reward-selected rejection-sampling fine-tune recovers part of that gain without the k-fold sampling cost. Decode temperature is a lever as large as either one, and its sign flips between representations. At the voxel geometry floor, model size and selection buy the same step and do not stack.

## 1. Contributions

- **An honesty protocol for post-training against a composite metric, enforced in code.** Rewards that feed a BlockScore pillar are refused by `rewards.get`. The trained-on signals travel with the samples (`trained_on_signals`), and claims are made only on pillars the reward did not touch. Transfer from a correlated signal (26-connected to 6-connected connectivity) is pre-registered and not claimed (`blockgen/posttrain/__init__.py`, docs/pipeline.md §7).
- **The first two rungs of the ladder, measured on two representations.** Best-of-4 and RFT were run on a voxel AR model (5M and 20M) and on a six-face facade model (20M D4). On facades, best-of-4 at temperature 0.7 matches the representation's own ceiling on geometry (T77).
- **Checkpoint selection is the confound in RFT.** Selecting on holdout NLL picks a 500-step nudge, because fine-tuning on self-filtered samples raises holdout loss from the first eval (T71, T72). Selecting on the trained-on reward (`select=reward`) picked the better checkpoint on both tracks (T73, T74).
- **Size and selection are substitutes on voxel geometry.** At 5M, best-of-4 takes BlockScore from 386 to 340. At 20M, the raw model scores 346 and best-of-4 scores 341, and the extra gain lands on appearance only (T78).

## 2. Evidence on disk

BlockScore is in noise-floor units that scale with n (T82: units roughly double from n=128 to n=256). Facade cards (n=399) and voxel cards (n=256) are therefore not comparable with each other. Within a card, all arms share n, the reference and the descriptor. Lower is better throughout.

### 2.1 Facade track (parent `outputs/run_20261028_125240_houses32_facades_joint_d4_20m`, 20M D4, one seed, n=399)

| T | arm | BlockScore | geom_kid | mv_dino_kid | enterable | nn pct | card |
|---|---|--:|--:|--:|--:|--:|---|
| T68 | raw, T=1.0 | 1082 | 30.3 | 0.468 | 0.35 | 0.48 | (T68 card) |
| T70 | raw, T=0.7 | 637 | 17.8 | 0.268 | 0.31 | 0.51 | `run_20261031_080537_bench_facade_post-training_best-of-4_on_face` |
| T70 | best-of-4 face_solidity, T=1.0 | 871 | 24.4 | 0.300 | 0.49 | 0.42 | same |
| T72 | RFT, NLL-selected (step 500), T=1.0 | 994 | 27.8 | 0.380 | 0.53 | 0.51 | `run_20261031_121039_bench_facade_post-training_rejection-sampling` |
| T73 | RFT, reward-selected (step 500), T=0.7 | 553 | 15.4 | 0.182 | 0.35 | 0.58 | `run_20261101_160512_bench_facade_rung_2_rerun_rejection-sampling` |
| T77 | **best-of-4, T=0.7** | **483** | 13.5 | 0.149 | 0.37 | 0.49 | `run_20261102_131429_bench_facade_rung_1_and_2_at_temperature_0_7` |
| T77 | real@facades6 (ceiling) | 493 | 13.8 | 0.001 | 0.73 | 0.51 | same |

### 2.2 Voxel track (parents `outputs/run_20261015_235516_m2b_sft_s1` at 5M and `outputs/run_20261102_151738_m2b_20m_sft_s1` at 20M, seed 1 only, n=256)

| T | arm | BlockScore | geom_kid | mv_dino_kid | enterable | nn pct | card |
|---|---|--:|--:|--:|--:|--:|---|
| T71 | 5M raw, T=1.0 | 386 | 18.6 | 0.410 | 0.20 | 0.64 | `run_20261031_105428_bench_voxel_post-training_best-of-4_and` |
| T71 | 5M best-of-4 lcc26 | 340 | 16.3 | 0.344 | 0.21 | 0.61 | same |
| T71 | 5M RFT, NLL-selected (step 500) | 395 | 19.0 | 0.357 | 0.26 | 0.62 | same |
| T73 | 5M raw, T=0.7 | 431 | 20.7 | 0.604 | 0.11 | 0.86 | `run_20261101_143155_bench_voxel_rung_2_rerun_rejection-sampling` |
| T74 | 5M RFT, reward-selected (step 4,000), T=1.0 | 359 | 17.2 | 0.363 | 0.27 | 0.58 | `run_20261102_113421_bench_voxel_rung_2_step-4000_reward-selected` |
| T74 | same, T=0.7 | 478 | 23.0 | 0.425 | 0.14 | 0.81 | same |
| T78 | 20M raw | 346 | 16.6 | 0.347 | 0.26 | 0.58 | `run_20261103_101858_bench_voxel_M2b_at_20M_against_5M_raw_and` |
| T78 | 20M best-of-4 lcc26 | 341 | 16.3 | 0.288 | 0.30 | 0.50 | same |

### 2.3 Solid and pilot

What holds within a card, after Holm correction on paired bootstrap:
- Facade best-of-4 at T=0.7 shares the ceiling's `geom_kid` group and beats raw T=0.7 by 4.3 [3.5, 5.2], p_adj 0.009 (T77).
- Facade RFT at T=0.7 beats raw T=0.7 on `mv_dino_kid`, 0.086 [0.073, 0.101], p_adj 0.008 (T73).
- Temperature 0.7 is significant on both facade pillars, with every Holm pair separated (T70). On voxels it is significantly worse on geometry: the T73 arm loses by 5.7 [4.8, 6.8], p_adj 0.012.
- 5M best-of-4 sits alone in its voxel `geom_kid` group (T71).
- 20M best-of-4 beats 20M raw on `mv_dino_kid`, 0.059 [0.040, 0.077], p_adj 0.013 (T78).

What is single-seed or a trend:
- Every arm comes from one parent seed per track.
- The four voxel arms at T=1.0 (raw, best-of-4, RFT step 500, RFT step 4,000) are not separated after Holm at n=256 (T74). The ordering is a trend.
- The headline "size and selection do not stack" rests on 20M raw versus 5M raw on `geom_kid`: 1.9 [1.1, 2.8], p_adj 0.09, one seed (T78).
- The facade RFT versus raw gap at T=0.7 on `geom_kid` has p_adj 0.10 (T73).
- The 20M SFT was selected at step 1,500 of 16,000 and overfits hard after that (T78). A short-schedule rerun, `outputs/run_20261104_004730_m2b_20m_sft3k_s1`, is on disk with no summary and no card.

Caveats a reviewer will find:
- Best-of-4 on `lcc26` shrinks builds (max-dim W1 2.96 against 0.64 raw at 20M, T78). The geometry gain may be partly a size effect. T82 checked this for Dream-Cubed by dropping the size term; nobody has run the same check here.
- On facades, best-of-4 at T=1.0 drifts toward memorization (nn pct 0.42, T70). At T=0.7 it returns to 0.49 (T77).
- Facade arms reach a facade ceiling, not real houses. Real faces lifted to a shell still score 493, so a facade arm "at the ceiling" still has empty interiors. That is the failure the evaluation paper is about.
- The 5M raw parent scores 386 on the T71 card and 323.2 on the T60 card. The difference is the T64 descriptor change. Post-training numbers are therefore comparable only on the T70 to T78 cards.

## 3. Novelty and closest prior work

None of the techniques is new. Rejection-sampling fine-tuning is "the first move in every report" of the 2025 to 2026 LLM stacks (research.md §G.2). Best-of-N with a verifier is the standard test-time baseline. The closest domain work:

- **BrickGPT / LegoGPT** (2505.05469, ICCV 2025): brick-level rejection sampling plus physics rollback at inference.
- **Rollback-Free Brick Generation** (2605.06947): moves validity into GRPO training with assembly-level rewards.
- **PVPO** (2606.07602): names the "PhysHack" failure, where a model trained on validity alone produces degenerate builds. This is the closest prior statement of the problem the honesty rule addresses.
- **BrickAnything** (2605.26182): best-of-N, then reward-weighted DPO.
- **Mesh-RFT** (2505.16761), **DSO** (2503.22677), and "Are We Ready for RL in Text-to-3D?" (2512.10949, CVPR 2026) for 3D generally.
- **Dream-Cubed** (2604.22847), the Minecraft diffusion baseline. It reports no post-training stage that this repo has verified.

What may be new:
- Applying the rung to Minecraft block generators.
- The explicit honesty protocol: pillar-disjoint rewards, claims restricted to untouched pillars, pre-registered transfer.
- The size-versus-selection substitution measured on a geometry metric.
- The representation-dependent sign of temperature.

Flags:
- I could not verify that no prior work states a pillar-disjoint reward rule. Reward-hacking analyses (2409.06493, 2601.03468, references.md §7) are adjacent and unread here.
- Test-time-compute scaling papers that trade model size against best-of-N (the LLM literature on this exchange rate) are not in references.md. A reviewer will cite them, and they must be found and read before any claim about the size and selection trade.
- research.md marks several arXiv ids with †, meaning recalled and not fetched (DPO, Diffusion-DPO, Flow-GRPO). They need checking.
- The rung beyond RFT, GRPO, has no code (docs/pipeline.md §7). Rollback-Free and PVPO already publish GRPO on bricks. A "post-training ladder" paper without rung 3 is the weaker half of those papers.

## 4. Gaps a reviewer would demand

| # | experiment | rough cost | result that sinks the claim |
|---|---|---|---|
| 1 | **Seed replication of the voxel 2x2** (5M/20M x raw/best-of-4) at n=399. 5M seeds 0 and 2 exist (T60); 20M needs two more pretrain+SFT seeds. | 20M pretrain 49 min + short SFT, about 1.5 h per seed; 20M best-of-4 at 1,596 draws about 3.2 h per seed (7.25 s per draw, T78); 5M sampling under 1 h per seed. About 1.5 GPU-days total. | 20M+best-of-4 separates from both 20M raw and 5M best-of-4 on `geom_kid` across seeds (the effects stack), or the 5M best-of-4 gain does not replicate. |
| 2 | **best-of-k sweep**, k = 1, 2, 4, 8, 16, at 5M and 20M, plotted against sampling FLOPs | Mostly sampling; k=16 at 20M is about 13 h per seed. 1 to 2 GPU-days. | The curves cross nowhere, or k=8 at 5M passes 20M raw by a wide margin. Then "substitutes" becomes "selection dominates". |
| 3 | **Size-free and shrinkage controls**: rescore the best-of-4 arms without the size term and with size-matched real references | CPU, under an hour per card | The `geom_kid` gain disappears without the size term, meaning `lcc26` selection bought geometry by picking smaller builds. |
| 4 | **Pillar-reward counter-example**: best-of-4 on a pillar input (geometry scalars), scored with the full card and a held-out human 2AFC | Sampling only (reuse the T71 draws: rescoring 1,024 voxel draws is CPU), plus about 200 2AFC judgements on the existing phone queue | Pillar-rewarded selection does not inflate its pillar beyond human preference. The rule would then be unnecessary, and contribution 1 loses its motivation. |
| 5 | **Human 2AFC: best-of-4 vs raw** on both tracks | No GPU; rater time | Raters do not prefer best-of-4, so the BlockScore gain is metric-internal. |
| 6 | **Temperature sweep** 0.5 to 1.2 on both parents, with best-of-4 at each | Facade best-of-4 at n=399 is 2.7 h per point (notes §62); about 2 GPU-days for 5 points | Best-of-4 adds nothing beyond the best temperature, which would make selection just sharpening. T77 already argues against this on facades with one seed. |
| 7 | **GRPO rung** (groups of 4 to 8, KL to SFT reference, research.md §G.6) | Implementation about 1 week; runs 1 to 3 GPU-days at 5M | GRPO beats best-of-4 easily. That reframes the paper as "the cheap rungs are a floor", which is a weaker finding. |
| 8 | **Second facade parent seed** | 20M D4 facade training plus 3 sampling arms at 2.7 h each. About 1 GPU-day. | Best-of-4 at T=0.7 leaves the ceiling's Holm group. |

## 5. Venue fit and timeline

- **CVPR 2027 main (Nov 16, 12 days):** not realistic. Gaps 1 to 3 alone need 3 to 4 GPU-days on a GPU the evaluation paper also needs. The technique novelty is low for a main track, and research.md §I.2 already ruled that only the evaluation paper reaches the bar.
- **Workshop (CVPR 2027 workshops, deadlines typically March, unverified):** plausible as part of a recipe paper. It is weak alone.
- **ICCV 2027 (about early March 2027, unconfirmed):** realistic only as a combined paper. The combination is world-crop pretraining (research.md §I.5, T60) plus this ladder, with a GRPO rung, three seeds, and the size-by-selection exchange-rate figure. That combination is a coherent "LLM-style training stack for block generators" paper.
- **NeurIPS 2027 D&B (about May):** poor fit. This is a methods result, not a dataset or benchmark. NeurIPS main is a better slot for the combined paper if ICCV slips.
- **arXiv now:** premature. Every arm is one seed. Post after gap 1, ideally after the evaluation paper is public, so the metric can be cited by name.

**Overlap with the evaluation paper.** The CVPR draft contains no post-training, facade or M2b results; a grep of `deliverables/paper/cvpr27/sec/*.tex` finds none, and research.md §I.3 keeps them out. The overlap is the instrument itself: BlockScore, V-KID, the pillars and the gates. The honesty rule is defined in terms of the pillars, so this paper cannot claim the metric as a contribution. It must cite the evaluation paper, as an anonymous concurrent submission while that paper is under review. The dual-submission risk is low as long as the post-training rows never enter the CVPR paper as "scored arms", which research.md §I.3 permits. That door should stay shut.

## 6. Paper outline (8 pages, as a section of or companion to the pretraining paper)

1. **Introduction** (0.75 p): post-training against a composite metric invites gaming; we state a rule and measure what survives it.
2. **Related work** (0.75 p): LLM post-training stacks, brick RL (BrickGPT, Rollback-Free, PVPO, BrickAnything), 3D RL, test-time compute scaling.
3. **Setup** (1 p): two representations (voxel AR, six-face facades), the pretrain-then-SFT parents, BlockScore pillars, and the honesty rule with its code enforcement.
4. **Rung 1: best-of-N and temperature** (1.5 p): the facade ceiling result (T77), the opposite temperature signs (T70, T73, T74), and the k sweep.
5. **Rung 2: RFT and checkpoint selection** (1.25 p): NLL selection picks the wrong checkpoint (T71, T72), and reward selection fixes it (T73, T74).
6. **Size versus selection** (1.25 p): the 2x2 over three seeds and the exchange-rate curve (T78 extended).
7. **Rung 3: GRPO** (0.75 p): to be run.
8. **Limitations and conclusion** (0.75 p): facade ceiling is not real, reward shrinkage, one dataset (houses32).

Figures:
1. **Ladder plot.** BlockScore and `geom_kid` per arm for both tracks, with the ceiling and real lines and Holm groups as brackets.
2. **Exchange-rate plot.** `geom_kid` against sampling FLOPs for best-of-k at 5M and 20M, three seeds, with the substitution point marked.
3. **Sample sheet.** Raw, best-of-4 and RFT at each track's temperature beside the ceiling, with one cutaway column to show the facade interiors stay empty.

## 7. Verdict

**Strength 2 of 5 as a standalone paper; 3 of 5 as a section of the pretraining paper.** Every number is one seed. The voxel ordering at T=1.0 does not survive Holm correction. The techniques are standard, and brick GRPO already exists. The sound contributions are the honesty protocol, the checkpoint-selection lesson and the facade-ceiling result. Recommendation: fold this into the world-crop pretraining paper as its post-training section, aimed at ICCV 2027.

**Most important next experiment:** the three-seed voxel 2x2 (5M/20M x raw/best-of-4) at n=399 with the size-free rescore (gaps 1 and 3, about 1.5 GPU-days). It either confirms or kills "size and selection do not stack", which is the only finding here that is not already in the literature.
