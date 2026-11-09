# BlockPlayground: one protocol for every way of making a block structure

Internal technical report, 2026-11-04. Candidate paper topic: the whole suite as one artifact.
Built only from evidence on disk; every number is quoted from results.md (T-numbers), notes.md
(§) or research.md as written there.

**Thesis.** Block-structure generators from different paradigms (trained voxel and piece
autoregressive models, facade models, post-trained models, LLM-written build programs, published
external generators and procedural baselines) can be ranked on one board only when the board pins
the ruler: a fixed split, a calibration arm scored in the same run, a minimum n, and gates that
disqualify memorization and collapse; BlockPlayground is that board plus the tools to train,
inspect, label and play what it ranks.

## 1. Contributions a platform paper would claim

- **A comparability protocol for cross-run leaderboards.** `houses32-v1` (blockgen/eval/bench/protocol.py)
  pins corpus, group-aware split, tier, seed, descriptor, `min_n=128` per arm and on the calibration
  arm, `min_ref=256`. The motivating number: the same `native_oriented` checkpoint scores 87.14
  against a 128-build `real_test` and 19.05 against a 32-build one (leaderboard-protocol memo, 2026-09-07).
  A protocol re-score reproduces an earlier card bit-identically (T82 validation run).
- **A cross-paradigm board of record.** One n=256 card ranks procedural baselines, degraded-real
  controls, three in-house AR arms and one external published generator with bootstrap rank
  intervals (T82); a sister LEGO card scores BrickGPT (T83). Agentic LLM arms and post-trained
  models are scored on the same protocol at n=128 to 133 (T51, T64, T73, T74, T77, T78).
- **BlockLab**, a dependency-free local app (tools/lab, eight pages: hub, curate, gates, compare/2AFC,
  leaderboard, runs, ontology, phone queue) that turns scorecards into inspectable builds and
  collects curation labels and 2AFC pairs. On disk as of 2026-11-04: 452 curation labels,
  617 compares (306 `more_real`, 311 `better_build`), 0 captions (outputs/lab/lab.db).
- **A shared trainer, arch registry and live serving path.** `python -m blockgen.train` with an
  arch registry (blockgen/train/archs.py), post-training rungs (blockgen/posttrain), and a Fabric
  1.21.1 mod plus WebSocket server that streams any registered model or agentic recipe into a live
  game (deploy/README.md; verified end to end: 473 blocks placed in 1.2 s, undo reverted 461).

## 2. Evidence on disk

### Solid (protocol n, controls in-run, intervals or multiple seeds)

| T | result | run path | status |
|---|---|---|---|
| T82 | n=256 board of record: real 0.00, dreamcubed 165.71, native_oriented (contaminated) 229.29, canon_oriented 285.22, canon_unoriented 345.10, patchwork 431.70, gabled_house 620.98, uniform_random 1087.06; rank intervals 2-2, 3-3, 4-4, 5-5 for the generators | outputs/run_20261103_090315_bench_n_256_leaderboard_of_record | n=256, 1,000 replicates, self-validation 6/6; marked `official: false` because the pin names the 80-d descriptor and this run uses 77-d |
| T82 (validation) | re-score reproduces run_20260925_002631: bit-identical geom_kid, mv_dino_kid within 7e-7 | outputs/run_20261102_214725_bench_validation_0925_re-score_under_the_v1 | reproducibility of the protocol itself |
| T26 | dose-response monotone (Spearman +1.000) on 3 of 4 axes for both tiers; five procedural baselines, every learned arm beats all five | see T26 | ladder, CPU |
| T60 | pretrain + SFT beats scratch on every seed at 5M: holdout NLL 0.479, 0.420, 0.350 vs 0.565, 0.555, 0.509; BlockScore 375, 323, 324 vs 502, 445, 334 | six M2b scorecards, outputs/registry.jsonl | three seeds, n=256; shows the trainer and board working together |
| T83 | BrickGPT on held-out ST2B test, n=253: geom +52.1, render +11.8, BlockScore 52.14; stability reproduction 95.8% of solved vs paper 98.8% | outputs/run_20261102_220228_brickgpt_lego | external model, licence-clean LEGO domain |
| T64 | traversal pillar orders arms as a player does (llm_control 2.59, dream 1.26, classic-matched 0.78) but never decides BlockScore | outputs/run_20261023_011035_bench_traversal_pillar_leaderboard_re-score | n=128, one card |

### Single-seed, pilot, or not reportable

| T | result | caveat |
|---|---|---|
| T51 | dream-caption 165.0, dream 191.3, classic-matched 194.4 (BlockScore); dream and classic share a significance group; dream-caption beats both on render KID (p_holm 0.009) | one run each, dream-caption at n=100 below `min_n`; descriptor since changed (T64) |
| T73, T74, T77, T78 | post-training: facade 553 (RFT, T=0.7), 483 (BoN, T=0.7) against a facade ceiling of 493; voxel 359 (RFT step 4,000) not Holm-separated from raw 386 at n=256; 20M voxel 346 | one seed each |
| T79 | Dream-Cubed fast-tier smoke, extracted arm 181.3 | smoke; superseded by T82 row |
| T40 | rater study (19 raters, 960 judgements): exterior 0.512, cutaway 0.767, gabled_house over patchwork 93/93 | removed from all papers: no IRB determination; cannot be reported |
| lab.db | 617 phone/compare judgements | collected by the authors; not a human study; same IRB constraint for any outside raters |
| deploy | one verified console transcript | a demo, not a measurement |

### The Dream-Cubed row needs a caveat before any paper leans on it

The co-author's concern holds up on the evidence. Dream-Cubed generates 32³ terrain chunks
conditioned on a biome class; the only Apache-2.0 class that produces buildings is `village`
(notes §66). The builds are vanilla 1.12 village pieces, median 278 blocks in a 9x8x9 box against
745 blocks and 16x12x15 for houses_32 (notes §66); sealed-interior ratio 0.016 against 0.114
(T79). The house is cut out of the chunk by an extraction step written by us that "favours the
model" (notes §66), and the whole chunk is disqualified for mode collapse (T82, +24.0). Dream-Cubed's
second place on T82 therefore measures how close Minecraft's own village templates are to
houses_32, filtered through our extraction, more than it measures a generator. The repo already
holds the direct control: 483 village templates from the 1.21.1 jar (T56). Scoring those templates
as an arm (CPU, minutes) decides the question. If the templates alone rank at or above Dream-Cubed,
its row is a template-prior reference, not a competitor. Its duplicate rate (0.4%) and diversity
(0.867 against real 0.898, T79) show it is not emitting one house, but a small template set can
still produce varied fragments. A platform paper should list it as an out-of-task external
reference. The evaluation paper session in deliverables/paper/cvpr27 should hear this before the
row is framed as "the best published generator".

## 3. Novelty and closest prior work

| work | what it is | how BlockPlayground differs |
|---|---|---|
| MC-Bench / MineBench (references.md §2; research.md table) | frontier LLMs emit code/JSON builds, human pairwise Glicko voting | closest platform precedent. It ranks one paradigm (prompted LLMs) by human vote; ours ranks six paradigms by a validated metric with in-run controls. It has human votes at scale; we have none reportable |
| GDMC Settlement Challenge (Salge et al., 1803.09853) | annual competition, human 0-10 rubric over world edits | human-judged, settlement scale, procedural entrants |
| MineAnyBuild (NeurIPS 2025 D&B, research.md §I.7) | Minecraft building benchmark in the D&B track | shows the venue accepts Minecraft benchmarks; its exact protocol and metrics are not recorded in the repo (unverified) |
| Dream-Cubed (2604.22847, NeurIPS 2026) | 3D diffusion on 2M chunks, render FID plus human study | a generator with an evaluation, not a board |
| BrickBench (2610.12452), BrickNet (CVPR 2026) | LEGO agentic benchmark; LEGO generator | LEGO twins; the LEGO half of our board overlaps them |
| CaP-X (cited in the agentic-suite memo as the harness style) | recipe harness for agents | design influence only; full citation not in references.md (unverified) |

What appears unoccupied (research.md §I.7): pairing geometry-space and render-space KID on one
block board, a traversability pillar, a calibration ladder of degraded real builds. All three are
the evaluation paper's contributions, which is the central problem for this topic (section 5).
The new part a platform paper can claim is narrower: the comparability protocol, the
cross-paradigm board, and the tooling. Those read as engineering unless the board yields a finding.

## 4. Gaps a reviewer would demand

| # | gap | cost | result that sinks the claim |
|---|---|---|---|
| 1 | **Releasable data.** houses_32 is 51% scraped GrabCraft, 48% 3D-Craft without a licence file, 1% text2mc; no redistribution grant (eval-suite memo). A D&B reviewer requires the board to be reproducible by outsiders. Options: run the full board on licence-clean LEGO (ST2B MIT, OMR CC BY 4.0), release reference features plus a fetch script for Minecraft, or a CurseForge-licensed set (research.md §F) | LEGO board: 1 to 2 GPU-days of arm sampling plus CPU scoring; CurseForge: weeks of licensing work | rankings on the clean set disagree with houses32-v1, or no clean Minecraft set exists |
| 2 | **Village-template control** for Dream-Cubed (T56 templates as an arm) | CPU, under an hour | templates match or beat Dream-Cubed, removing the only external Minecraft row |
| 3 | **Equal-n, multi-seed arms for every paradigm.** Agentic arms at n=256 under the 80-d descriptor; three seeds for facade and post-trained arms | agentic about $25 API (T51 spent $12.18 at n=133); facade and voxel seeds about 1 GPU-day each | paradigm ranks cross inside their rank intervals, so the board orders nothing |
| 4 | **Human agreement with IRB.** The phone queue exists; outside raters need an IRB determination first (user decision 2026-11-02) | 2 to 4 weeks for the determination; rater time cheap | board ranks disagree with people beyond the known Patch-Stitch reversal (T40, 93/93) |
| 5 | **An outside submission.** Every arm so far was integrated by the authors; Dream-Cubed needed an upstream bug workaround and a custom extraction (notes §66) | one external collaborator, weeks | integration needs author hand work, so the platform is a lab notebook, not a platform |
| 6 | **Text faithfulness pillar.** The mod serves `/gen <text>`, but `ArmSpec.prompts` is never set, so faithfulness is unscored (eval-suite memo) | 2 to 3 engineering days plus a CLIP or VLM pass | a playground for text-to-build that cannot score the text |
| 7 | **Reproducibility hygiene.** The trainer and corpus pipeline are untracked in git; `runner.py:364` drops the descriptor stamp (pipeline-standardisation memo) | hours, needs the user's go-ahead | released code differs from what produced the board |

## 5. Venue fit, timeline, and overlap with the evaluation paper

**Can the whole suite go in one paper?** Not at CVPR, and not without weakening the evaluation
paper. The scope decision of 2026-09-20 (SCOPE.md, iclr-2027-scope memo) still holds: inside the
evaluation paper, the weak models are evidence that the suite catches failures (memorization near
31%, an 87% test-contaminated arm, one-block-thick LLM output). Promoted to contributions, the same
facts invite a generative-modelling review against BrickGPT and Dream-Cubed, which the in-house
models lose. A platform paper that "shows off all the approaches" should report them as baselines,
not methods, and focus its finding on one or two cross-paradigm comparisons that the board makes
possible: geometry decides BlockScore for every learned arm (T64), and pretraining helps at every
seed (T60).

**Dual-submission risk.** BlockScore, the gates, the ladder and the houses32-v1 board are the
evaluation paper's apparatus. Submitting a platform paper that re-presents them while the CVPR
paper is under review (decisions 2027-02-25) is a dual-submission violation at CVPR and at most
venues. After acceptance, the platform paper must cite the evaluation paper and claim only the
protocol, the cross-paradigm board, the licence-clean set, the human data and the tools. After a
rejection, the two collide again at ICCV 2027 (~early March, unconfirmed).

| venue | fit | verdict |
|---|---|---|
| CVPR 2027 main (Nov 16) | no D&B track; 12 days; direct overlap with the evaluation paper | not realistic; do not attempt |
| arXiv technical report now | arXiv is allowed under CVPR rules, but a report naming the authors and the board de-anonymizes the evaluation paper | wait; ship the playground as the evaluation paper's anonymized supplementary release (make_release.py) instead |
| CVPR 2027 workshop (deadlines ~March, unconfirmed) | a 4-page non-archival demo or tools paper | possible after the CVPR decision, if non-archival |
| ICCV 2027 (~early March) | a vision main track wants a finding, not a platform | poor fit |
| NeurIPS 2027 D&B (~mid May, unconfirmed) | the venue for benchmark-plus-harness papers (research.md §I, iclr-2027-scope memo); precedent MineAnyBuild | **best fit**, conditional on gaps 1, 3 and 4 |
| demo tracks: SIGGRAPH Real-Time Live, CHI Interactivity, IEEE CoG, FDG (dates unverified) | the live mod and the dream-loop narration are demo material | a short demo of the mod plus lab, separate from the D&B paper |

## 6. Paper outline (8 pages)

1. **Introduction** (1.0). Block generators come from six paradigms and no board compares them; a board without a pinned ruler ranks the ruler.
2. **Related work** (0.75). MC-Bench, GDMC, MineAnyBuild, Dream-Cubed, BrickBench; the evaluation paper as the metric source.
3. **The protocol** (1.0). Split, in-run calibration arm, `min_n`, gates, rank intervals; the 87.14 versus 19.05 demonstration and the bit-identical re-score.
4. **The arms** (1.0). Procedural baselines, degraded-real controls, AR, facade, post-trained, agentic, external; the submission contract (npz plus manifest, models.json entry).
5. **The board** (1.75). Minecraft and LEGO cards at n=256 with rank intervals and pillar breakdown; what decides each arm.
6. **Tools** (1.0). BlockLab pages, phone queue, trainer registry, live serving; what each one caught.
7. **Human agreement** (0.75). IRB-approved rater study over board pairs.
8. **Limitations and release** (0.75). Data licensing, what is released, maintenance plan.

Figures: (1) system diagram from corpus to trainer, sampler, bench, lab and live mod; (2) the
cross-paradigm board, one row per arm with rank interval, worst pillar and an example strip;
(3) the ruler figure: one model's BlockScore against calibration-set size, with the pinned
protocol's value marked.

## 7. Verdict

**Strength: 2 of 5** as a paper today; 3 of 5 for NeurIPS 2027 D&B if the board is rebuilt on
licence-clean data with equal-n arms and an IRB-approved rater study. The tooling is real and
unusually well instrumented, but most of its scientific content belongs to the evaluation paper,
its human data cannot be reported, its only external Minecraft row is a template-prior suspect,
and its Minecraft data cannot be released.

**Most important next experiment:** the licence-clean board, the LEGO suite at n=256 with
BrickGPT, the degraded-real LEGO controls, procedural baselines and any LEGO generator retrained through the shared trainer (T69 BrickNet recipe), under a pinned protocol. Run the
village-template control (gap 2) first, because it costs minutes and changes how the evaluation
paper frames Dream-Cubed.
