# BlockGen: everything tried, with verdicts

One line per experiment or approach, grouped by theme and in date order inside each group.
Every line cites the results.md table (T-number) or notes.md section that holds the numbers,
so a verdict here can be checked there. Updated 2026-11-05, through T95.

Verdict marks: **WIN** the lever worked and is kept · **NULL** no measurable effect or a tie ·
**NEG** the idea lost or was disproven · **FIND** a measurement that changed what we do ·
**TOOL** infrastructure or a protocol that other rows depend on · **OPEN** built or launched,
result not yet in results.md · **SUP** superseded or retracted.

Two T-numbers were used twice by parallel sessions and are left as they stand in results.md:
T51 (world-crop pilot, then agentic arms) and T79 (Dream-Cubed smoke). The face cascade,
first filed as T79, is now T88 (notes §74, first filed as §66).

## 1. Data and corpora

| mark | what we tried | outcome | where |
|---|---|---|---|
| TOOL | PlanetMinecraft tfrecord cache (`tf_small_24`), 5,866 labelled builds | 100% metadata; the 24³ cap drops 27k as too big | T1, T2 |
| TOOL | Curation: house subset, exact dedup, material-variant groups kept | 714 houses, 58 variant groups preserved | T3 |
| TOOL | Cross-corpus house caches `houses_32` / `houses_48` on one vocab | 2,661 / 2,971 builds; enclosed-air gate drops 675 | T3b |
| NEG then SUP | `data/raw` declared unlabelled by filename-offset test | Wrong corpus joined; files are minecraft-schematics.com ids, labelled after all | notes §8, T19 |
| FIND | Cache `max_dim` from the size distribution | 32 is the knee (85% of houses) | notes §7 |
| TOOL | text2mc raw `.schem` unlock (28k) | 11k to 39k builds available | memory, notes §3 |
| WIN | GrabCraft category-labelled scrape | Cleaner data: AR NN-IoU 0.57 to 0.73, block agreement 0.17 to 0.48 | T8 |
| TOOL | World-crop pilot: 400 archived worlds | 2,651 crops in 8 min; extrapolates to about 1M | T51 (first) |
| WIN | Learned terrain gate (GBT, 295 features), four rounds | Rejects 8/8 terrain, keeps 106/106 built on the hand-labelled hold-out | T54 |
| TOOL | Corpus manifest, exact and MinHash near-dedup over 926k builds | 216k exact folds; 29.9% of the corpus is a near-duplicate; village leak found | T55, notes §46 |
| TOOL | 1.21.1 jar structure templates as a source | 1,007 pieces; internal use only | T56 |
| TOOL | Tiers (pretrain-all / clean / anneal / sft / holdout) and tokenised shards | Feed the shared trainer | notes §48 |
| FIND | `anneal` tier contains 1,834 of the 1,863 SFT houses | Pretrain-then-SFT comparisons partly measure extra passes over SFT data | T87, worldcrop report |

## 2. Representation and tokenisation

| mark | what we tried | outcome | where |
|---|---|---|---|
| FIND | What the model sees: token stream vs grid vs graph; learned embeddings | Wood and wool variants near-orthogonal (cos 0.006); flat embeddings never learn birch≈oak | T7 |
| WIN | 3D-BPE cluster tokens | Resist memorisation (dup 0.0 where flat copies) and halve sequences | T10, T14 |
| NEG | "Cluster tokens cohere by themselves" | Raw validity 0.0 at 16³; BPE buys reach, not connectivity | T10 |
| FIND | 3D-BPE vocabulary audit | 253/256 pieces are one material; greedy counts grow rectangles only | T18 |
| FIND | Pipeline audit of the canon-16 decimator | Strided decimation destroys 86.5% of blocks on 63% of builds | T17 |
| NEG | Native 32³ + BPE to escape the decimator (the "gating run") | 75.4% of baseline vs 83.3% for canon16+flat; resolution and tokeniser moved together | T20 |
| OPEN | Factored piece embeddings (shape + rot + family + variant) | Implemented 2026-07-15; matched-budget ablation never run | notes §9.1 |
| NEG | Semantic node-embedding prior (frozen CLIP table, ideas #7) | Lowered block agreement 0.076 to 0.052 and validity; single seed | ideas.md Status |
| OPEN | PMI merge scoring, shape-factored merges, block-data rotation table | Proposed after T18; not built | notes §9.15 to §9.17 |
| WIN | Oriented vocab (stairs facings, log axes kept) | 427 to 655 blocks; fixes mod door placement | notes §19 |
| TOOL | Scale normalisation (downsample to fixed N³) to fit token tracks | 654/714 houses fit AR at 12³ vs 41 at 24³; lossy | notes §11 |

## 3. Autoregressive voxel generators and training levers

| mark | what we tried | outcome | where |
|---|---|---|---|
| SUP | Three-track novelty on the legacy 896 subset | Protocol demo only | T4 |
| FIND | Curated-subset retraining, novelty vs own training set | AR memorises small classes; diffusion learns the gestalt without copying | T5 |
| WIN | AR vs diffusion on identical canonical 12³ houses | AR NN-IoU 0.568 vs 0.369, builds bottom-up | T6 |
| FIND | AR on 724 medieval houses | Best quality yet and 31% verbatim copies | T8 |
| WIN | More classes, 32 cap, dedup, fewer epochs | dup 0.31 to 0.00 on every model | T9 |
| WIN | Category conditioning (one class token) | Best cohesion lever on every dataset | T10 |
| WIN | More data (gc to combined houses) | Flat-AR raw validity 0.19 to 0.56 | T10 |
| WIN | Honest-novelty protocol (dedup, val split, D4 train-only, NN-IoU, dup rate) | 12/12 rows at dup 0 with val_nn 85 to 93% of baseline | T10 |
| WIN | Grammar-aware phase4 positional encoding | Best of five PE arms, 0.405 vs learned 0.395 | T11 |
| NEG | RoPE / ALiBi | RoPE ties learned, ALiBi worst arm | T11 |
| WIN | In-loop adjacency-constrained decoding, raster order | Validity 1.0 at about no quality cost | T11 |
| NEG | BFS-from-ground token order | Worst non-broken arm; drags the constrained arm down | T11 |
| NULL | phase4 + constraint stacked | Constraint dominates; PE edge absorbed | notes §7 |
| NEG | Pool-pretrain (16k) then finetune on `houses_32`, canon-16 | Finetune 0.379 < scratch 0.405; the decimator was in the loop | T12, T17 |
| TOOL | LegoACE port (MinecraftACE, GPT-2 92M, 4-token records) | Pipeline validated; flat ground planes at 15 epochs | T13 |
| WIN | MinecraftACE with 3D-BPE pieces, matched budget | First vertical structure; NN-IoU +79%, components 27 to 10 | T14 |
| NEG then FIND | LegoACE-style conditioning (DINOv2 / CLIP prefix, CFG) at native 32³ | Looked worse than unconditioned; the regime (resolution, epochs, CFG) was the gap, not conditioning | T15, T16 |
| OPEN | Conditioning-channel upgrade (Q-Former-lite resampler, ideas #6) on the full captioned corpus | Trained (`outputs/cond/`), paired-vs-shuffled fidelity never written up | notes §19, ideas.md |
| NEG | KV cache as the sampling speed-up | 1.1 to 1.5×, not 800×; launch overhead dominates; batching is the lever | notes §8 |
| TOOL | Shared trainer (`blockgen.train`), WSD schedule, registry, end-of-run bench | 5M scratch baseline in 23 min | T57 |
| WIN | Pretrain on world-crop `anneal` then SFT, 5M | 22% lower holdout NLL than scratch once truncation bug fixed | T58 |
| FIND | M2 three-seed pilot | Dropout change at init_from invalidated it; protocol rewritten | T59 |
| WIN | M2b: pretrain then SFT vs matched scratch, 3 seeds, early stopped | Wins every seed: 0.416 vs 0.543 nats, 341 vs 427 BlockScore | T60 |
| FIND | Against the best-regularised scratch (dropout 0.1) | Likelihood is a tie at best; the honest comparison still owed | T60 |
| WIN | Same recipe at 20M | NLL 0.420 to 0.286; ties 5M best-of-4 on the board; SFT overfits from step 1,500 | T78 |
| NULL | 20M SFT with a 3k decayed schedule | Same best NLL as 16k | T87 |
| WIN | 20M SFT with the anneal tier mixed 1:1 | Still improving at the end (0.263); first voxel arm to separate from 5M raw on geometry | T87 |

## 4. Masked discrete diffusion (Track B)

| mark | what we tried | outcome | where |
|---|---|---|---|
| TOOL | Fixes to stop all-air collapse: keep-mask schedule, Gumbel confidence, air down-weighting, `air_bias` | Required for any non-empty, varied grid | A1, notes §8 |
| WIN | Flow-matching sampler on the trained absorbing net | Holds occupancy where MaskGIT dumps a blob; dataset-dependent | T6, notes §7 |
| WIN | Two-stage occupancy then materials | Best diffusion shape fidelity (nn_iou 0.38 vs 0.21 to 0.26) | T11 |
| NEG | Two-stage for material scrambling | block_agree 0.028, same as single-stage; occupancy 9× overfilled | T11 |
| NEG | Sampler sweep (maskgit / flow / remask / stratified) on one 32³ model | No sampler rescues it; remask broken; occupancy swings 2 to 7,716 | T11 |
| NEG | Single air-bias scalar on heterogeneous pools | Overshoots or collapses; diffusion-16 stays the dense track's best | T10 |

## 5. Graph, attachment and adjacency generators

| mark | what we tried | outcome | where |
|---|---|---|---|
| NEG | Attachment/growth AR (no box, attach to open faces), Phase 0 + 1 | Representation validated; generation collapses to lines, plates or spheres | T21 |
| NULL | Pick-and-place: learned picker and placer over a growth graph | Mechanism works (84× chance), builds solid but small; KID worse than every track | T24 |
| NULL | Attention as an adjacency matrix in a pretrained LLM | Signal is a coordinate-token subtraction | T28 |
| NULL | Geometry bias injected into attention (Track D v2 + relative-offset bias) | Null; parameterisation ate the experiment | T29 |
| FIND | Attention-as-adjacency on our trained voxel model | A neighbour head exists (hard-pair AUROC above chance) | T49 |
| OPEN | Nodes in, adjacency out (block list, no positions) | Prototype decodes 8³ windows; coarse condition does the shaping | T50 |
| NEG | Connection-by-connection merge decoder | Lines and sheets; random policy scores higher; no frame | T52 |
| NULL | Cell assignment then connections on full houses, f=4 | Ties retrieval on voxel IoU (0.32); merge deadlocks; f=8 worse | T53 |
| OPEN | Finer frame f=2 with hierarchical stage 1 | Proposed, not run | T53 |

## 6. LLM baseline (Track D) and contamination

| mark | what we tried | outcome | where |
|---|---|---|---|
| TOOL | BrickGPT-style LoRA LLM on `block x y z` lines (`llm_control`, `llm_bias`) | Baseline arms on the board | notes §21, §32 |
| FIND | `native_oriented` at n=128, BlockScore 106.87 | Scored on its own training data | T31, notes §33 |
| FIND | Clean-split retrain | Native loses its separation from the LLM arms; unoriented retrain DQ for memorisation | T32 |
| FIND | LLM arms under the traversal probe | A player enters 2% of LLM builds | T63 |
| SUP | LLM arms on the leaderboard of record | Dropped by user decision at n=256 | T82 |

## 7. Agentic programs (Track E)

| mark | what we tried | outcome | where |
|---|---|---|---|
| WIN | LLM writes a WorldEdit-style build program | 1,000+ block coherent houses for under a cent | T22 |
| OPEN | Scaffolding ablation (plan / example / repair / critique) | Wired, never run | T22 |
| NULL | Block ontology as prompt context (`ont_none` / `ont_mined` / `ont_shuffled`), n=12 | Every interval overlaps; needs n≥128 | T27 |
| TOOL | Recipe harness: staged, dream-loop, describe methods, mock-verified | 17 recipes, cached and deterministic | T41, T43, T45 |
| FIND | First dream pilot | Match loop barely improves; distance is style, not build error | T42 |
| FIND | Staged and dream pilots, n=2 to 4 | Impressions only | T44, T46 |
| FIND | Fully local Gemma 4 12B, text only | Valid programs, generic shapes; thinking changes what it attempts | T47 |
| FIND | Fully local dream loop (Gemma 12B vision + SDXL-Turbo on CPU) | Caption removes copied scenery; $0 | T48 |
| NULL | Dream vs compute-matched classic, n=133 | Same significance group on both metrics | T51 (second) |
| WIN | dream-caption | Best agentic arm; beats dream and classic on mv_dino_kid, p 0.009; prompt coherence not separable | T51 (second) |
| FIND | Inspection checks on the T51 arms vs real houses | Agentic arms pass every structural check and fail `enterable` 33 to 57% (real 17%); the door is the gap | T95, notes §81 |
| TOOL | ldraw-nova-style inspect-and-repair (`inspect` method, `inspection` scorer) and `lessons` context, with shuffled control | Built and mock-verified; calibrated on 400 real houses (mean 0.93); battery `inspect-v1` estimated at $3.70 | notes §81, docs §5.7 |
| OPEN | `inspect-v1` battery: symbolic inspection vs render critique, with and without retrieved lessons | Not yet paid-run | notes §81 |
| FIND | Agentic arms under the traversal probe | 50 to 68% enterable vs real 95% | T63 |

## 8. The facade track (six face images and a lift)

| mark | what we tried | outcome | where |
|---|---|---|---|
| FIND | Facade oracle: real houses reduced to the visible shell | Ceiling for any facade generator: 493 where real is 0 | T65 |
| WIN | Joint six-face layout vs per-face | 1128 vs 1205; one palette across faces | T66 |
| FIND | Patchwork (six real faces from six houses) | Still beats every learned facade model on BlockScore | T66, T68 |
| NEG | 20M on the plain shard | Memorises by step 6,000; best geom_kid anyway | T68, notes §57 |
| WIN | D4 augmentation of the facade shard | 0.33 nats at 20M; appearance, traversal, novelty; nothing on geom_kid | T68 |
| FIND | Where the geometry pillar is held | Edge disagreement between faces and the empty interior | T68 |
| WIN | Cross-face constrained decoding (ShellState logit mask: no first hit behind or through what earlier faces committed; determined cells written, not sampled) | Face sets consistent by construction; components 33.7 to 12.9 at T=0.7 (ceiling 13.5); geom_kid 19.0 to 16.5 and mv_dino_kid 0.30 to 0.17, Holm-separated from the grammar control; 679 to 589; no training | T117, notes §89 |
| FIND | Grammar-only decoding (headers and EOS forced, cells well formed) | Not separated from the raw sampler; the facade gain is the cross-face rule alone | T117 |
| FIND | Cost of the hard see-through rule | A hole in the front face empties the column for every later face: median blocks 686 to 509, skeletal builds; relaxed rule and face order are the follow-ups | T117 |
| FIND | Oracle completion from the real front face, constrained vs grammar | Shell IoU 0.36 to 0.55 (front + right: 0.47 to 0.72); the rule recovers silhouette and footprint, the far face depth and block identity stay the model's | T117 |
| WIN | Constrained decoding stacked with best-of-4 (`face_solidity`, k=4, base model) | geom_kid 13.9 = best-of-4 13.5 = ceiling 13.8 (one Holm group), mv_dino_kid 0.149 to 0.098 (own group, best learned facade arm), components 15.9 to 7.7 (below the ceiling), see-through cost mostly gone; BlockScore 497 vs 483 is the geometry pillar at its ceiling | T122, notes §89 |
| FIND | Constrained decoding of the RFT model | BlockScore 552 vs 553, geom tie, mv_dino_kid 0.182 to 0.147 separated, components 17.6 to 10.4; the fine-tune had already bought most of the shell | T122 |
| NEG | Constrained best-of-4 on the RFT model (three solidity levers stacked) | DQ for self-duplication: twin rate 0.27 (real 0.01); 37% of kept builds at face_solidity >= 0.99 (one lever 1.8%, two 7.3%); solid single-material monoliths, mv_dino_kid back to 0.304; the reward saturates and its optimum is a filled box | T122 |
| FIND | Facade geometry pillar at the six-face ceiling (constrained best-of-4, best-of-4, ceiling in one group) | BlockScore of the facade track is bounded by the empty interior, not the generator; next lever = interior stage, and a counterweighted reward before more selection | T122 |
| FIND | Pooled-face probe (fixed hierarchy f=16 to 1 on real faces) | Windows survive only to f=2; pooled levels smoother than real | T75 |
| NEG | Fixed-pooling face cascade (NCM/VAR recipe, six runs) | Depth residual learned, block residual not; no window ever; coarse levels memorise; blobs match face statistics better than any AR arm | T88, notes §74 |
| FIND | Face statistics as a quality signal | Gameable by smoothness; a distribution metric on faces is needed | T88 |
| NEG | Coarse by cascade, fine by AR: per-face AR with its own f=4 level as prefix (`pooled4` layout), real f=4 given | Block agreement 0.28 to 0.33 vs copy 0.556 (cascade 0.515), mask IoU 0.62 vs 0.76; prefix sets palette, not shape; draws windows (0.69 of faces, cascade 0) | T120, notes §90 |
| OPEN | Edge-consistency decoding; interior second stage | Named as the levers after T68; not built | T68 |

## 9. Post-training (best-of-N and rejection-sampling fine-tuning)

| mark | what we tried | outcome | where |
|---|---|---|---|
| WIN | Best-of-4 on `face_solidity`, facades, T=1.0 | 1082 to 871 | T70 |
| WIN | Temperature 0.7 alone, facades | 1082 to 637 | T70 |
| WIN | Best-of-4 on `lcc26`, voxels | 386 to 340, geom_kid separated from every learned arm | T71 |
| NULL | RFT on `lcc26`, NLL-selected, voxels | 395; selector kept step 500 | T71 |
| NULL | RFT on `face_solidity`, NLL-selected, T=1.0 | 994, inside raw's group | T72 |
| WIN | RFT + reward selection + T=0.7, facades | 553; not separable from the facade ceiling on geometry; fine-tune share significant on appearance only | T73 |
| NEG | Temperature 0.7 on voxels, with or without the fine-tune | Fills walls (1.12 to 1.30 to 1.45 thick); geometry worse | T73, T74 |
| FIND | `lcc26` rewards filling | Next voxel reward needs a counterweight that is not a pillar input | T73 |
| WIN | Best-of-4 at T=0.7, facades | 483, at the facade ceiling on geometry | T77 |
| NULL | Best-of-4 on the 20M voxel model | Moves appearance, not geometry; the voxel geometry floor sits near 340 | T78 |
| TOOL | Honesty rule: rewards refuse BlockScore pillar inputs; `select=reward` | Trained-on signals carry no claim | notes §62 |

## 10. Evaluation: metrics, their validation, and what they miss

| mark | what we tried | outcome | where |
|---|---|---|---|
| TOOL | Cross-track bench with a corruption ladder; MV-DINO-KID as primary realism | CMMD, Fréchet, PRDC fail validity gates | T23 |
| FIND | Fill every enclosed air cell (`real@solidify`) | Invisible to MV-DINO-KID; `geom_kid` sees it | T25 |
| FIND | Dose-response, baselines, registered prediction | geom_kid monotone on all four axes; MV-KID flat on interior fill | T26 |
| FIND | Which structure types the interior probe applies to | Enclosed volume, not rooms; LEGO cars 94% | T33 |
| FIND | LEGO replication (47k bricks, 20³) | All seven render-space statistics blind to solidify | T34 |
| FIND | Noise-floor unit is n-dependent | Separation grows linearly in n; always state n | T35 |
| SUP | "Blindness is not exact render invariance" | Retracted: under an opaque renderer it is invariance; 86% was a texture-atlas artefact | T36, T37 |
| FIND | Residual after the atlas fix | Wobble, not partial detection | T38 |
| FIND | V-KID descriptor ablation, mass-matched control | Not a size detector | T39 |
| WIN | Human rater study, 20 sessions, pre-registered | All three predictions hold; exterior solidify at chance (0.512) | T40 |
| WIN | Traversability probe (two-block player walks in) | Real 95%, agentic 50 to 68%, LLM 2% | T63 |
| WIN | Room scalars in the descriptor and a traversal pillar | Arms re-order the way the player does | T64 |
| FIND | Standard point-cloud suite (MMD, coverage, 1-NNA; chamfer and IoU) | As blind to interiors as the render tier | T67 |
| FIND | Mesh interior probe on Objaverse buildings | Strip and fill keep exterior renders identical; render metrics at 0 | T76 |
| FIND | VLM judge (gpt-4o), pre-registered | At chance on exterior views, above chance with the roof off | T80 |
| FIND | 3D-native rows (Uni3D, ULIP-2, chamfer, exact EMD), three samplers, two seeds | Strip and fill at z≈0 on every row | T81 |
| TOOL | n=256 leaderboard of record, 0925 protocol pinned | Units double, ranks hold; Dream-Cubed is the best generator | T82 |
| FIND | Nearest-neighbour double dissociation | DINO space finds the source of a filled build, V-KID finds the source of a reshuffled one | T84 |
| TOOL | Paper regeneration checks | Size share of Dream-Cubed's V-KID at most 6.5% | T85 |
| WIN | V-KID robustness: leave-family-out, random subsets, generic orbit-averaged descriptors, global terms | Every family and 20/20 random halves pass; random 3D CNN and raw 3^3 rows pass every gate; global terms leave Patch-Stitch 4th of 9 | T105, notes §82 |
| FIND | Functional ground truth (player enterability, reached floor/rooms, roof, connectivity) on the 76 MV-KID/V-KID reversed pairs | Real ranks first and FILL/HOLLOW are caught, but it sides with MV-KID on 60 of 76 (V-KID 21%; non-facade 1 of 38); Spearman with MV-KID 0.67 vs V-KID 0.16 | T114, notes §87 |
| FIND | Functional ground truth rebased on the 40 exact-permutation reversed pairs; 34 decisions re-tested exactly | Sides with MV-KID 33 of 40 (non-facade 18 of 18); decisions agree 6, MV-only 14, V-only 1 (which FD reverses) | T123, notes §91 |
| TOOL | BlockLab, phone queue, run identity, leaderboard protocol | Curation, 2AFC, standings | notes §29, §31, §49 |

## 11. External generators and the LEGO twin

| mark | what we tried | outcome | where |
|---|---|---|---|
| WIN | BrickNet Table 2 reproduced from the public 0.6B adapter; OMR through its pipeline | Reproduces | T61 |
| TOOL | BrickAgent environment, one BrickBench task end to end | Verified; judges need 62 GB | T62 |
| NEG | BrickNet recipe on Minecraft walks, LEGO-init vs base-init pretrain | LEGO start 0.01 nats behind at every checkpoint, no transfer | T69 |
| NEG | BrickNet SFT on caption-walk pairs, base vs clean LEGO-init stack | Compact house-palette clumps, 92-94% parsable, 1-2% collision-free, arms tie within noise | T69 |
| FIND | Dream-Cubed (published MD4 village model) on the bench | Best generator on the board at n=256; raw chunks DQ for mode collapse | T79, T82 |
| FIND | Dream-Cubed audit | Mostly repeated vanilla village pieces; a template-only control ranks 3rd | T86 |
| FIND | BrickGPT (ICCV 2025) on the LEGO suite, held-out test split | geom_kid +52 noise floors, worse than solidify; own stability metric reproduced | T83 |

## 12. Ideas rated and never run

From ideas.md (2026-07-19) and research.md: own structure encoder for conditioning (#1);
structure-to-VLM-token translator (#4); native graph-generation mode on a VLM (#5);
typed-connection AR for LEGO with validity by construction (#8, research §E.2); DPO or
GRPO with a VLM or MineCLIP reward (#9, research §C; only best-of-N and RFT exist); the
learned residual VQ for the cascade (research §H.4, deferred after T88); the 3D NCM arm
(research §H); build-order supervision from 3D-Craft (notes §9.13); delta-coordinate
tokens (§9.8); multi-slot conditioning with label dropout (§9.12).

## 13. Standing open items

- Scratch arm with dropout and weight decay chosen on holdout, both arms early stopped, 5M
  and 20M, three seeds (research §I.5; T60 owes it).
- Ontology arms at n≥128 (T27); scaffolding ablation (T22); agentic novelty vs corpus.
- Face-image distribution metric (T88); edge-consistency decoding and an interior stage (T68).
- Run the 20M anneal mix longer and a world-crop-only mix (T87).
- Commit the untracked pipeline, facade and cascade code.
