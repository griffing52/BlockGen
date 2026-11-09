# Imagine, then build: LLM-written build programs with an image-model dream loop

Internal technical report, 2026-11-04. Topic slug `agentic-dream-build` (Track E).
Built only from evidence on disk: results.md T22, T27, T42, T44 to T48, T51, T63, T64,
T80; notes.md §22, §27, §36, §39, §52; research.md §I.6, §I.7; references.md.

## 1. Thesis

**Candidate claim:** when a language model writes a WorldEdit-style build program, an
imagined target image plus a VLM caption of that image moves the build distribution
toward real houses in render space, while the image alone does nothing a
compute-matched no-image loop cannot, and neither closes the gap a player measures
(getting in, reaching upper floors).

The claim the evidence supports today is narrower: **"dreaming is a null on geometry;
the caption moves the render metric; the no-dream control is the most walkable."**
That is a workshop result (research.md §I.6), not a main-track paper.

Naming note. The `dream` arms here are the image-imagination loop. They are unrelated to
**Dream-Cubed** (Sakana, 2604.22847), the published diffusion generator now on the
evaluation paper's board (T79, T82). Dream-Cubed's village checkpoint is class
conditioned, not text conditioned, and produces terrain chunks with vanilla village
pieces (T79: mean 453 blocks against 904 real, sealed-interior ratio 0.016 against
0.114, whole chunks disqualified for mode collapse). It is not a baseline for
prompt-to-build agents and should not be presented as one.

## 2. Contributions a paper would claim

1. **A recipe harness for agentic block building** (`blockgen/agentic`, notes §36): a
   method is a plain function over an Episode (ask, try, render, dream, score, best);
   a YAML recipe fixes LLM per role, image generator, action space (DSL, block lines,
   sandboxed Python, mixed, all lowered to one canonical DSL), contexts, scorers and
   accept policy. A shared content-addressed dream cache holds the dream fixed across
   arms, and a ledger prices every arm before spending. 34 recipes exist in
   `configs/agentic/recipes`.
2. **A compute-matched control for imagination.** `classic-matched` runs the same number
   of render critiques without a dream, so "dreaming helps" is tested against "more
   loop iterations help". At n=133 the dream gain over this control is not significant
   on either distribution metric (T51).
3. **Caption grounding as the effective ingredient.** `dream-caption` (a VLM describes the
   dream, including `surroundings_to_omit`) beats both `dream` and `classic-matched` on
   multi-view DINO KID (p_holm 0.009 each, T51), and in the local Gemma path the
   caption is what strips imagined terrain from the build (T48).
4. **A shuffled-attribute control for knowledge injection.** The block ontology arm
   (T27) shows that a grounded material table and the same table with attributes
   permuted onto the wrong blocks move palette JSD by the same amount: the control
   that prevents a false positive.

## 3. Evidence on disk

All paid arms use gpt-5-mini; dream arms use gpt-image-1-mini. Prompts are the 133
GrabCraft titles of the houses32-v1 test split (`configs/agentic/prompts/houses32_test_grabcraft.txt`).
Every agentic result is **one seed**.

### 3a. Protocol-scale results (n >= 128, one seed)

| T | result | numbers as written | run |
|---|---|---|---|
| T51 | BlockScore, worst pillar geometry for all three | dream-caption 165.0 (n=100), dream 191.3, classic-matched 194.4 (n=133); real@solidify 213.4 | `outputs/run_20261007_185345_bench_agentic_dream_vs_compute-matched` |
| T51 | render KID (mv_dino_kid) | dream-caption 0.215, dream 0.342, classic-matched 0.403; dream-caption beats both, p_holm 0.009; dream vs classic same group | same |
| T51 | geom_kid | 16.9 / 19.6 / 19.9; no pair significant | same |
| T51 | diversity, memorization | voxel_diversity 0.795 / 0.786 / 0.833 (real 0.902); dup 0 for all; dino_nn_percentile 0.48 to 0.59 (real 0.54) | same |
| T51 | practical checks | "favour the control: door at player scale 73% vs 52%, headroom 93% vs 87%, and one connected piece 97.7% vs 94.7%" | same |
| T51 | interior | enclosed_air_ratio 0.04 against real 0.11 | same |
| T63 | traversability, enterable / walkable | classic-matched 0.68 / 0.62, dream 0.50 / 0.45, dream-caption 0.51 / 0.44; real 0.95 / 0.88; LLM serialization arms 0.02 to 0.05 | `outputs/run_20261023_004317_bench_traversal` |
| T63 | floors reached / floors | 0.77/1.17, 0.51/1.06, 0.53/1.16; real 1.67/1.84 | same |
| T64 | re-scored board (80-d descriptor, traversal pillar) | dream-caption 155.83 (rank 4), dream 177.23 (6), classic-matched 181.86 (7); canon_oriented 141.82 (3); traversal pillar 1.24 / 1.26 / 0.78 | `outputs/run_20261023_011035_bench_traversal_pillar_leaderboard_re-score` |
| T51 | cost | $0.048 / $0.040 / $0.045 per build; $12.18 total | runs `outputs/run_20261007_092654_bench_houses32_{dream,classic}`, `outputs/run_20261007_145754_bench_houses32_dream_caption` |

What is solid: the three-arm ordering on mv_dino_kid (Holm-corrected paired tests at
n=100 to 133), the dream versus classic tie on both KIDs, and the T63 traversal gap,
which uses a calibrated probe (seal_openings takes real from 0.95 to 0.02 enterable).
What is weak: one seed, one LLM, one image model; dream-caption ran on a seeded
random 100 of the 133 prompts, below `min_n` = 128, so it is "scored but not
comparable under the protocol" (T51). The T82 leaderboard of record (n=256) carries no
agentic arm, and its units differ (BlockScores roughly double at n=256), so agentic
numbers cannot be placed on the current board.

### 3b. Pilots (n <= 12, impressions only)

| T | arms, n | what it showed |
|---|---|---|
| T22 | oneshot, n=13 | 1,140 to 1,979 mean blocks, 1 failed command of ~430, ~$0.004/build; scaffolding ablation never run |
| T27 | ontology none/mined/shuffled/stats, n=12, $0.22 | every interval overlaps; mined equals shuffled on every metric |
| T42, T44 | dream variants, n <= 4 | match loop barely improves (embed similarity 0.56 to 0.61); blueprint turns every prompt into a box house |
| T46 | six caption/describe arms, n=4, $0.84 | dream-caption best in-loop score (0.63 vs 0.53); describe-before worst; text grid rule did not make windmill sails vertical |
| T47, T48 | local Gemma 4 12B, SDXL-Turbo, n=6, $0 | free path works; caption removes imagined terrain (coherence 1.00 vs 0.83) |
| T80 | gpt-4o VLM judge, 2,288 trials, $4.31 | **no agentic arm in it.** Instrument only: judge P(real) at chance on exteriors, system ranks Spearman 0.76 with MV-KID |

A qualitative point from T51 that a reviewer will ask about: "classic draws almost every
prompt as the same oak gable cottage". Set-level voxel_diversity (0.833, the highest of
the three) does not see this, because diversity is measured over the set and not per
prompt. The same weakness applies to Dream-Cubed's village pieces on the evaluation
paper's board.

## 4. Novelty and closest prior work

- **MC-Bench / MineBench** (mcbench.ai, minebench.ai, references.md): frontier LLMs emit
  code or JSON blocks for Minecraft prompts, ranked by human pairwise Glicko votes. This
  is the closest prior and it undercuts "LLM writes Minecraft build programs" as a
  novelty claim. It is a live leaderboard with no published ablations, so the
  dream/caption/compute-matched ablation remains open.
- **BrickBench / BrickAgent** (2610.12452): coding agents write Python against a LEGO
  connector API with a mechanical validator; 300 prompts, VQA and pairwise VLM ELO
  (human-validated tau 0.89); agents beat trained generators. This is the LEGO twin and
  it published the "agentic beats trained" framing first. Our environment check ran one
  task end to end (T62).
- **BrickWright** (2610.31651): named in research.md §I.6 as a second LEGO twin. It has
  **no entry in references.md** and no notes section; its content is unverified.
- **MineAnyBuild** (NeurIPS 2025 D&B): cited in research.md §I.7 as a must-cite; its
  agent-planning content is not summarized anywhere in the repo and needs a read
  before any novelty sentence.
- **DreamCraft** (2404.15538): per-prompt CLIP-guided optimization of block NeRFs; a
  contrast, not a program method. **ShapeAssembly** (2009.08026): program prior over
  cuboids.
- Not in references.md and not verified here: LLM-to-Blender or LLM-to-CAD code agents
  with visual feedback (3D-GPT, SceneCraft, LL3M, CAD code-verification loops). A
  reviewer from CV will name these; the related-work pass must cover them.

Defensible novelty after the read: (a) the compute-matched imagination control, (b) the
finding that the caption, not the image, carries the render gain, (c) evaluating
program-written builds with interior and traversal measures that render metrics and
VLM judges miss. Item (c) belongs to the evaluation paper.

## 5. Gaps a reviewer would demand

API prices from T51 ($0.040 to $0.048 per build) and T80 ($0.0019 per judge trial).
**Budget remaining: $1.52 of $15.** Every row below except the free ones needs a new
spend authorization from the authors.

| gap | cost | result that sinks the claim |
|---|---|---|
| dream-caption on all 133 prompts (protocol-comparable) | ~$1.6 for the 33 missing (T51 estimated ~$5 with retries) | mv_dino_kid lead over dream shrinks into one significance group |
| caption controls at n=133: `dream-caption-shuffled`, and a no-image arm with a text caption of the prompt | ~$13 | shuffled caption scores as well as the true caption (then the gain is caption length or style words, not grounding) |
| n=256 and a second seed for the three main arms (needs 123 more prompts, e.g. val titles) | ~$35 to $50 | ordering flips between seeds |
| second LLM family (Claude or Gemini) on three arms, n=133 | ~$30 to $60; free for Gemma local but GPU hours on the shared card | caption gain absent for a stronger model (a gpt-5-mini artifact) |
| `best-of-4` at matched `would_cost_usd` | ~$25 | best-of-4 matches dream-caption, so the gain is sampling, not imagination |
| scaffolding ablation (zeroshot/oneshot/plan/repair/critique) at n=133 | ~$10 to $30 | no stage earns its cost (then the harness is engineering, not a finding) |
| prompt fidelity: paired build vs its titled real house, CLIP text score | CPU, free | dream-caption no closer to its own prompt than classic |
| per-prompt diversity (k samples per prompt) for the cottage failure | ~$20 for 32 prompts x 4 samples x 3 arms | classic per-prompt diversity equals dream (then "dream follows the prompt" is anecdote) |
| human 2AFC, dream-caption vs classic-matched, phone queue (§49) | free, ~2 to 3 author-hours for 300 pairs | humans prefer classic or are at chance |
| VLM judge pairs on the agentic arms (T80 protocol) | ~$2 | judge disagrees with mv_dino_kid ordering |
| ontology at n >= 128 with shuffled control | ~$15 | mined equals shuffled again (expected; report as a negative) |

Total for a credible paper: roughly $150 to $250 of API and a few author-hours. No
training and almost no GPU.

## 6. Venue fit and timeline

- **CVPR 2027 main (Nov 16, enrol Nov 10): not realistic.** Twelve days, $1.52 of
  budget, one seed, one LLM, and the strongest result is a render-KID gain at n=100.
  It also competes for the same author-hours as the evaluation paper.
- **Dual-submission risk.** The evaluation paper (`deliverables/paper/cvpr27`) contains
  no agentic arm today (only Dream-Cubed), so result tables do not overlap. The risk is
  in the instruments: BlockScore, the traversal probe and the VLM judge are that paper's
  contributions. An agentic paper must cite them as prior or concurrent work and claim
  only the generation findings. If the agentic arms are later added to the evaluation
  paper's board (for example as the "external generator" row), this paper loses its T63
  and T64 results to it. The authors should decide which paper owns T51/T63 before
  either is submitted.
- **CVPR 2027 workshop (deadlines typically March to April): good fit.** A four to
  eight page "dreaming is a null, captions are not" note with the controls in §5.
- **ICCV 2027 main (~early March): possible** only if the second LLM, n=256 with two
  seeds and the human 2AFC all land and the caption effect survives. Two to four weeks
  of work after Nov 16.
- **NeurIPS 2027 D&B (~May): weak fit** as a method; fits only if reframed as a
  Minecraft agentic benchmark, which BrickBench already occupies for LEGO.
- **arXiv technical report now: not recommended** before the evaluation paper is
  submitted; it would publish the bench before its own paper and complicate anonymity.

## 7. Paper outline (8 pages)

1. Introduction (1.0): programs instead of voxels; the question is whether imagination
   helps, measured against a compute-matched loop.
2. Related work (0.75): MC-Bench, BrickBench, BrickWright, MineAnyBuild, code-writing 3D
   agents, Dream-Cubed and trained voxel generators.
3. Harness (1.0): Episode, recipes, single IR, dream cache, ledger, controls.
4. Methods (1.0): classic-matched, dream, dream-caption, shuffled caption, best-of-4.
5. Evaluation (0.75): bench pillars, traversal probe, fidelity, human 2AFC (cite the
   evaluation paper).
6. Results (2.0): main table at n=256, two seeds, two LLMs; caption ablation;
   traversal; per-prompt diversity; cost.
7. Failure analysis (0.75): oak cottage collapse, imagined terrain, flat sails,
   unreachable upper floors.
8. Limitations and conclusion (0.25), statements (0.5).

Figures:
1. Pipeline: prompt to dream to caption to program to render, with the match loop.
2. One prompt row per arm (classic-matched, dream, dream-caption, shuffled caption)
   beside its dream image and the real titled house.
3. Scatter of mv_dino_kid against traversal enterable per arm with real and the AR
   arms, showing the render gain and the walkability loss move in opposite directions.

## 8. Verdict

**Strength 2 / 5.** One seed, one LLM, the headline arm below protocol n, the closest
prior (MC-Bench, BrickBench) already public, and the one significant gain is on the
render metric the evaluation paper shows is blind to interiors.

**Most important next experiment:** the caption control at protocol n:
`dream-caption`, `dream-caption-shuffled` and `dream` on all 133 prompts (about $15),
then two seeds. If the shuffled caption matches the true caption, the paper has no
claim.
