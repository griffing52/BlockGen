# Facades First: Six-Face Depth Images as a Generative Representation for Block Structures

Internal technical report, 2026-11-04. Sources: results.md T65, T66, T68, T70, T72, T73, T74, T75, T77, T78, T88 (the face-cascade entry dated 2026-11-03), T82; notes.md §54 to §60, §63, §65, §74; research.md §H.7, §I.6, §I.7.

**Thesis.** A block structure written as six axis-aligned (depth, block-id) face images and lifted back to voxels is a representation a small sequence model learns well: with D4 augmentation, temperature 0.7 and best-of-4 selection, a 20M joint facade model reaches the geometry of real faces lifted the same way (BlockScore 483 against the ceiling's 493, same Holm group on `geom_kid`, T77). The cost of the representation is fixed and measured: every lifted build has an empty interior.

## 1. Contributions

1. **The representation and its oracle.** Six face images per build (four sides, roof plan, underside), each cell `empty` or `(depth, block)`, with an exact `project`/`lift` pair (`blockgen/corpus/facade.py`). The facade oracle (T65) fixes the ceiling of any facade generator before a model is trained: `real@facades6` 493, `real@facades5` 700, `real@facades4` 858, all held by geometry.
2. **A full baseline ladder on one board.** Unigram 2208 > markov 1335 > per-face 1205 > joint 1128 > patchwork 990 > ceiling 493 (T66), the ordering predicted in advance and every gap outside its interval.
3. **What moves a facade generator, by lever.** Joint layout over per-face (1205 to 1128, T66); D4 augmentation buys likelihood (0.806 to 0.479 nats), appearance, traversal and novelty but not `geom_kid` (T68); temperature 0.7 is the largest single lever (1082 to 637, T70); best-of-4 on `face_solidity` and temperature compose to 483 (T77); rejection-sampling fine-tuning reaches 553 and stays behind best-of-4 (T73).
4. **A measured coarse-to-fine hierarchy of faces** (T75) and a fixed-pooling cascade that learns the depth residual and not the block residual (T88): silhouettes and roof lines come back sharp, windows never appear, and face smoothness statistics are shown to be gameable by blobs.

## 2. Evidence on disk

All facade cards are `--tier both --n 399` (the whole `houses_32` test split), self-validation 6/6, descriptor `patch2_orb55_t5_h8`. Every learned facade arm is one training seed.

| T | claim | key numbers (as written in results.md) | run |
|---|---|---|---|
| T65 | facade ceiling, render tier blind to shells | facades6 492.7, facades5 700.5, facades4 858.0, solidify 846.3; MV-DINO-KID 0.001 (facades6); enterable 0.73 / 0.57 / 0.15 vs real 0.95 | `outputs/run_20261023_204241_bench_facade_oracle_real_houses_reduced_to` |
| T66 | baseline ladder, joint beats per-face | 2208 / 1335 / 1205 / 1128 / 990 / 493; components 71.5 (joint) vs 13.5 (ceiling) | `outputs/run_20261026_160934_bench_facade_track_baselines_the_per-face_and` |
| T68 | D4 helps NLL and appearance, not geometry | NLL 0.806 / 0.783 / 0.589 / 0.479; BlockScore 20M plain 1035, 20M D4 1082; D4 `mv_dino_kid` 0.468 vs patchwork 0.538, Holm p = 0.0004 | `outputs/run_20261028_143416_bench_facade_track_baselines_five_facade` |
| T70 | temperature and best-of-4 are separate levers | raw 1082, T=0.7 637, best-of-4 T=1.0 871; every `geom_kid` pair Holm-separated | `outputs/run_20261031_080537_bench_facade_post-training_best-of-4_on_face` |
| T72 | RFT at T=1.0 | 994, ties patchwork on `geom_kid`; enterable 0.53 | `outputs/run_20261031_121039_bench_facade_post-training_rejection-sampling` |
| T73 | RFT plus T=0.7 | 553; `geom_kid` gap to raw T=0.7 2.4 [1.3, 3.2], p_adj 0.10 (trend); `mv_dino_kid` gap 0.086, p_adj 0.008 | `outputs/run_20261101_160512_bench_facade_rung_2_rerun_rejection-sampling` |
| T77 | best-of-4 at T=0.7 on the ceiling | 483 vs 493; `geom_kid` 13.5 vs 13.8, same Holm group `c`; beats raw T=0.7 by 4.3 [3.5, 5.2], p_adj 0.009; appearance 155; enterable 0.37; nn pct 0.49 | `outputs/run_20261102_131429_bench_facade_rung_1_and_2_at_temperature_0_7` |
| T75 | the face hierarchy exists in real data | windows survive only to f = 2; pooled levels are one piece from f = 4 up | `outputs/run_20261102_163427_facade_pool_probe_roof`, `..._163441_..._front` |
| T88 | fixed-pooling cascade on front faces | completion from real f=4: depth exact 0.581 vs copy 0.520, block agree 0.515 vs copy 0.556; samples match face statistics (jumps 0.114 vs real 0.115) and read as blobs | `outputs/run_20261103_150008_facade_cascade_5m` and five siblings |

**Solid.** Within-card comparisons at n=399 with paired intervals and Holm groups: the T66 ladder ordering, the T70 temperature and best-of-4 effects, the T77 tie with the ceiling on `geom_kid`, and the appearance gains of D4 (T68) and of best-of-4 over RFT (T77, p_adj 0.009). The T65 oracle is deterministic given the test split.

**Single-seed or pilot.** Every learned facade arm is one seed of one model; no seed variance exists for the 483. The RFT `geom_kid` gains are trends after Holm (T73 p_adj 0.10; T77 best-of-4 over RFT p_adj 0.20). The cascade (T88) is a pilot on one face type, scored by face statistics now known to be gameable, and never lifted to the bench.

### Comparison with the voxel AR (about 340): not comparable as quoted

The voxel best arm is BlockScore 340 (M2b 5M best-of-4 `lcc26`, T71, T74; 20M best-of-4 341, T78). The facade 483 and the voxel 340 are not on one scale, for four reasons.

1. **Units depend on n.** BlockScore is in noise-floor spreads, and the spread shrinks with n (T82: every BlockScore roughly doubles from n=128 to n=256). Facade cards are at n=399, voxel cards at n=256. On these cards one `geom_kid` unit is worth about 36 BlockScore points on the facade card (492.7 / 13.76) and about 21 on the T74 voxel card (340 / 16.3). The facade 483 corresponds to roughly 280 in voxel-card units.
2. **Raw `geom_kid` is the fairer number, and it favours facades.** On the same descriptor (`patch2_orb55_t5_h8`, T74 and T77 cards) facade best-of-4 at T=0.7 is 13.5 against voxel best-of-4 16.3, and `mv_dino_kid` 0.149 against 0.344. T78's 20M voxel numbers are on the v2 80-d descriptor and are not on this scale. The reference sets also differ (399 against a 256 subsample), so the gap carries no interval.
3. **The facade sampler receives a real bounding box for free.** `facade_builds` (`blockgen/train/sample.py`) draws each build's (X, Y, Z) from the train split's boxes; the voxel AR samples its own size. The size term moves Dream-Cubed's `geom_kid` by 0.5 (7.67 to 7.17, T82), which is a lower bound on what a free box is worth, not a measurement.
4. **A hollow shell scores well on exactly the metrics the evaluation paper calls blind.** `real@facades6` (13.8) already sits below the best voxel AR (16.3) on `geom_kid`, and MV-DINO-KID cannot see the shell at all (0.001, T65). A claim that facades beat voxels rests on the interior-insensitive part of the board, which is the evaluation paper's own finding turned against the generator.

The defensible statement today is relative: facade generators close 98% of the gap from raw (1082) to their own ceiling (493) on one card. Facade against voxel needs one card, one n, one descriptor and a box-conditioned voxel control.

## 3. Novelty and closest prior work

- **Six-face (depth, block) for block structures** is unoccupied in the literature found (research.md §I.7). General precedents are multi-view depth diffusion (MVDD) and 2.5D-latent 3D generators; UniRecGen argues the opposite position (research.md §I.6). None of the three is in references.md; their citations and claims are **not verified** and need lookup before any writing.
- **Next-scale lineage** for the cascade: VAR (2404.02905), NCM (2610.14162), SAR3D (2411.16856), 3D-WAG (2411.19037), MARS (2502.11390), MaskGIT (CVPR 2022), all in references.md. references.md states the hierarchy is "not novel in 3D by itself"; the facade port is a host, not a contribution.
- **Block-structure generators to position against:** Dream-Cubed (2604.22847), Scaffold Diffusion (2509.00062), MineAnyBuild, Li's regime map (FDG 2026, same 32^3 canvas). Dream-Cubed is the strongest external arm on the board (165.71 at n=256, T82), but it generates terrain with village structures, so it is a reference row for the benchmark and not a like-for-like house generator; a facade paper should not frame itself against it.
- **Post-training levers** (temperature, best-of-N, rejection-sampling FT) are standard; their composition on facades is a result, not a method.

## 4. Gaps a reviewer would demand

| gap | cost | result that sinks the claim |
|---|---|---|
| **Matched voxel comparison**: facade and voxel arms on one card at n=399, current protocol (`_s12` descriptor; the T77 scorecard flags its BlockScore unit as out of date), plus a voxel AR prompted with the real box | rescore: about 2 GPU-h; voxel 256 to 399 draws and best-of-4: about 3 GPU-h; box-conditioned voxel SFT at 5M: about 2 GPU-h | box-conditioned voxel AR matches facade `geom_kid` (13.5) |
| **Seeds**: three seeds of 20M D4, each with T=0.7 best-of-4 | 3 x (training about 1 h + 1,600 draws + card 80 min): about 8 GPU-h | seed spread on `geom_kid` exceeds the 0.3 gap to the ceiling, or a seed falls outside group `c` |
| **Interior stage**: fill the lifted shell (voxel model conditioned on the shell, or plan plus fill) | weeks; new model and data path (research.md §H.7, §I.6 "a separate month") | filled builds score worse than the hollow shell on `geom_kid`, or sealed interior stays near 0 |
| **Interior-aware rows on facade arms**: volumetric IoU MMD (T67), 3D-native rows (T81), thickness and sealed interior | CPU, under 2 h per card | facade arms fall below voxel AR on IoU-based rows (expected, since thickness is exactly 1) |
| **Edge consistency**: shared-edge agreement rate between adjacent faces; a decoding constraint on corner columns | CPU metric, about 1 day of code; constraint about 2 days | components stay near 16 against ceiling 13.5 with no gain on `geom_kid` |
| **Face-image distribution metric** (KID on face features), since T88 shows face statistics are gameable | about 1 day code, CPU | ranking of facade arms changes under it |
| **Human 2AFC** on renders (the phone queue in BlockLab exists, never run) | no GPU, about 1 day of rater time | raters prefer voxel AR builds, which the render tier's shell blindness predicts it cannot detect |
| **Generalization beyond houses_32** (48^3 cache, other categories) | about 4 GPU-h plus data build | ceiling tie does not hold on a second corpus |

## 5. Venue fit and timeline

- **CVPR 2027 main (Nov 16, enrollment Nov 10): not realistic.** Twelve days, one seed, no matched voxel card, no interior stage, and the same two authors are submitting the evaluation paper to the same venue on the same benchmark.
- **Dual-submission and overlap risk.** The evaluation paper's Hollow rung and its interior-blindness finding (cvpr27 §`sec:hollow`) are the same fact as T65's render-tier result on `real@facades6`. A facade paper that reports T65 as a contribution, or that wins on render KID, duplicates and undercuts the evaluation paper. The facade paper must cite the evaluation paper for the benchmark and the blindness result and claim only the generator.
- **Workshop (CVPR 2027 workshops, deadlines around March): good fit** once seeds and the matched card exist. The relative-to-ceiling result plus lever ablations is a complete 4-page story.
- **ICCV 2027 (around early March): realistic only with the interior stage working** and the matched comparison favouring facades. Without the interior, the review will read "hollow shells".
- **NeurIPS 2027 D&B (around May): poor fit.** This is a method, not a dataset or benchmark.
- **arXiv technical report now: not advised.** It would expose the benchmark and its numbers while the evaluation paper is under anonymous review. Post after the CVPR decision, or fold T65 into the evaluation paper's appendix.

## 6. Paper outline (8 pages)

1. **Introduction** (1 p): block structures as images; the 2.5D bet and its fixed cost, the interior.
2. **Related work** (0.75 p): 2.5D and multi-view depth generation, next-scale models, block-structure generators.
3. **Representation** (1 p): six faces, project and lift, D4 symmetry as face relabelling, joint layout.
4. **The ceiling** (0.75 p): the facade oracle ladder, what each face is worth (roof plan 200 points, underside 200).
5. **Generators and levers** (1.5 p): baselines, joint AR, D4, temperature, best-of-N, RFT; one table, lever by lever.
6. **Results against voxels** (1.25 p): matched card, box-conditioned control, interior-aware rows, seeds.
7. **Interior stage** (1 p): shell to solid build, or stated as the limit if not built.
8. **Limitations and conclusion** (0.75 p): empty interiors, edge disagreement, windows at fine levels (T88).

Figures:
1. Representation: a house, its six face images, the lifted shell, and a cutaway showing the empty interior.
2. The lever ladder: BlockScore and `geom_kid` from raw 1082 to 483 with the ceiling and the voxel arms in matched units.
3. Face sheets: real, patchwork, raw T=1.0 and best-of-4 T=0.7 faces side by side with their lifted renders.

## 7. Verdict

**Strength 2/5.** The within-track result (483 against a 493 ceiling, n=399, Holm-tied) is clean and the lever story is complete, but it is one seed, it is relative to a ceiling that is hollow by construction, the facade-versus-voxel comparison is not yet made on one scale, and its favourable reading leans on the interior-blind metrics the evaluation paper exposes.

**Most important next experiment:** one card at n=399 under the current protocol with the T77 facade arm (three seeds), the voxel best-of-4 arm and a box-conditioned voxel AR, scored on `geom_kid` plus the volumetric IoU rows. If the facade arm still leads on geometry there, the interior stage is worth its month.
