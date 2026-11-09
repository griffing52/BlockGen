# LEGO transfer: the evaluation suite on brick assemblies

Internal technical report, 2026-11-04. Built only from evidence on disk (results.md, notes.md,
research.md, references.md). Topic slug: `lego-transfer`.

## 1. Title and thesis

**Working title.** *Inside the Bricks: Render Metrics Cannot See Interior Structure in
Text-to-LEGO Generation.*

**Thesis.** The interior blindness of render-and-embed metrics is a property of the render
representation, not of Minecraft: on LEGO brick assemblies every render-space statistic fails
to resolve a filled interior while a geometric statistic resolves it, and a published
text-to-brick generator (BrickGPT) is separated from held-out real data 4.4 times further by
geometry than by renders.

## 2. Contributions

- A second-domain replication of the interior-blindness result on 47,389 StableText2Brick
  (ST2B) structures: Fill-Interior moves MV-DINO-KID by −0.01 noise floors and `geom_kid` by
  +118.6 at n=256, and all seven render-space statistics fail to resolve it while resolving
  every other corruption (T34).
- The first scoring of a published LEGO generator under a dual-tier protocol: BrickGPT on the
  held-out ST2B test split, n=253, with its own validity and stability metrics reproduced on an
  open solver (T83, notes §70).
- Reproductions of BrickNet (CVPR 2026) Tables 2 and 4 from the public adapters and a working
  BrickAgent/BrickBench environment (T61, T62): infrastructure, not findings.
- A typed-connection token representation for diverse parts (research.md §E.2). **Proposal
  only; nothing is built or measured.**

## 3. Evidence on disk

| claim | entry | run path | n | seeds | status |
|---|---|---|--:|--:|---|
| LEGO Fill-Interior: render −0.01, geometry +118.6 noise floors; shuffled materials +140.0, monochrome +685.8 on render, geometry bit-identical | T34a | `outputs/run_20260918_200852_bench_lego_interior_blindness_replication/` (rerun `run_20260925_005517_...`) | 256, ref 450 | 1 (seed 0) | solid within run; single seed, one backbone (dinov2b, 4 ortho views) |
| All 7 render statistics fail `solidify` (kid 2.0, mmd_rbf 2.1, fd 0.5, ...), all resolve `chunk_delete_20` (7 to 14 floors) | T34b | same | 128 | 1 | solid as an ordering; `mmd_rbf` fails gate G7 in-domain |
| Normalised to n=128, LEGO geometry separation ≈59, about 3.4× weaker than houses | T34 correction, T35 | same | 128 vs 256 | 1 | the noise-floor unit is n-dependent; cross-domain magnitudes need normalising |
| BrickGPT: `geom_kid` +52.1, MV-DINO-KID +11.8; real@solidify +40.5 / −0.09; BlockScore 52.14 (geometry) | T83 | `outputs/run_20261102_220228_brickgpt_lego/` (`bench/` v2, `bench_v1/` v1) | 253, ref 512 | 1 generation pass (per-prompt seeds 1000+i) | single run; v1 descriptor agrees (+51.5 / +36.4) |
| BrickGPT coherence: lcc_ratio 0.980 (real 0.999), sealed interior 0.011 (real 0.034) | T83 | same | 253 | 1 | descriptive, no interval reported |
| BrickGPT own metrics: valid 98.8%, stable 95.8% of solved (230/240), mean stab. 0.995 vs paper 100% / 98.8% / 0.996 | T83 | same | 256 | 1 | deviations: bf16, connectivity rollback instead of Gurobi physics rollback, 1,800 s cap |
| BrickNet Table 2: Conn 94.5 ± 1.5 (paper 94.1), Coll 13.6 ± 1.1 (paper 16.0, outside interval) | T61 | `libs/BrickNet/repro/gen/pt0.6B_ns_n512*.jsonl` | 512 / 256 | 1 | reproduction, 3 of 4 cells in interval |
| BrickNet Table 4 perplexity 1.318 / 1.297 (paper 1.331 / 1.298) | T62 | `libs/BrickNet/repro/table4.py` | 1,000 / 444 walks | n/a | exact reproduction |
| BrickNet P_inv, 512 val captions: 90.2% parsable, 26.4% collision-free | T62 | `libs/BrickNet/repro/gen/sft0.6B_val_eos.jsonl` | 512 | 1 | metric definition ambiguous; three readings bracket the paper's 0.256 |
| BrickBench Model task 004 built valid by an agent | T62 | `libs/BrickAgent/work/model-004/` | 1 task | n/a | anecdote; VQA/ELO not scored (judge needs ~62 GB) |
| LEGO-to-Minecraft transfer: LEGO-initialised BrickNet behind base init by ≈0.01 nats/token at every checkpoint | T69 | `libs/BrickNet/repro/runs/` | 256 val walks | 1 | **negative**; SFT readout still pending |

**Solid.** The qualitative replication (render ≈ 0, geometry large, controls inside the table)
and the BrickGPT dual-tier gap. Both survive a descriptor change (v1 vs v2).

**Single-seed or pilot.** Every LEGO number above is one seed. No LEGO arm has a bootstrap over
reference draws, a second encoder, or a human judgement. BrickGPT is the only generator scored,
under a non-default rollback. The suite only reads 20³ cuboid ST2B brick lists (`bench/lego.py`
voxelises each footprint as one wool colour), so the "palette" pillar is a brick-type mix and
diverse-part LEGO (OMR, BrickNet's LDraw) is not covered at all.

## 4. Novelty and closest prior work

- **BrickGPT** (Pun et al., ICCV 2025, arXiv 2505.05469) evaluates validity, stability and
  CLIP-style text alignment; it reports no distributional realism metric on interiors.
- **BrickNet** (Kulits and Schmid, CVPR 2026, arXiv 2604.22984): LM over graph-backed
  connectivity programs with constrained decoding. This occupies most of the typed-connection
  idea of research.md §E.2, which was written on 2026-07-07 before BrickNet was in the
  bibliography. The §E "open intersection" claim should be treated as **taken** unless a
  specific difference (validity closed by construction rather than by checking) is shown.
- **BrickBench / BrickAgent** (arXiv 2610.12452): agentic text-to-LEGO with Valid / VQA / Design
  ELO; agents beat BrickNet-14B and BrickGPT. Its evaluation is render and VLM based, so the
  blindness argument applies to it directly. This is the strongest hook for a LEGO-specific
  evaluation paper.
- **LegoACE** (SIGGRAPH Asia 2025), **BrickAnything** (2605.26182), **BC-Bench**
  (2606.05445), **Rollback-Free Stable Brick Generation** (2605.06947), **LEGO-Eval**
  (VLM judges near-zero human agreement on fine 3D; cited in research.md §E.1 without an arXiv
  id in references.md: **unverified**).
- research.md §I.6 lists "LEGO reproductions and pick-and-place (T13, T24, T61, T62): no new
  result". That assessment stands; T83 is the one new LEGO result since.

Not verified here: whether any 2026 LEGO paper reports a geometry-space KID or an interior
probe. The §I.7 novelty scan covered block structures generally, not LEGO specifically.

## 5. Gaps a reviewer would demand

| gap | cost | result that sinks the claim |
|---|---|---|
| A second published LEGO generator on the suite. BrickNet's 512 val-caption samples already exist (`sft0.6B_val_eos.jsonl`); needs an LDraw-to-voxel path and a BrickNet-val reference | 1 to 2 days CPU engineering, minutes to score; reference needs the gated BrickNet data (received 2026-10-20) | BrickNet's geometry/render ratio near 1, or render separating it further than geometry: the 4.4× gap becomes BrickGPT-specific |
| Diverse-part ladder (OMR or BrickNet LDraw with Fill-Interior, chunk delete, material shuffle) | 2 to 3 days CPU (voxeliser, probes), under 1 GPU hour for DINO renders | render tier resolves Fill-Interior on detailed parts (windows, slopes, Technic gaps show the inside) |
| Seeds and intervals on T34 and T83 (3 to 5 reference and probe seeds, rank intervals) | under 2 GPU hours | render separation of solidify rising above the G-gates on some seed |
| BrickGPT with the Gurobi physics rollback (academic licence; UCLA qualifies) | ~4 to 15 h wall on the shared card (median 11 s, mean 90 s per structure) | none for the suite claim; removes the "not the published method" objection |
| Human 2AFC on LEGO (BrickGPT vs held-out real, solidify pairs with cutaways) | phone queue exists; 1 to 2 days of rater time, no GPU | people cannot tell BrickGPT from real and do not care about interiors, so the geometry gap is unmotivated for this domain |
| BrickBench agents scored on the suite (re-using their released builds if any) | depends on release; Design ELO judges need ~62 GB (cloud) | agent builds pass geometry as well as render, removing the "VLM benchmarks miss interiors" argument |
| Typed-connection model (§E.2) | 3 to 6 GPU weeks plus a month of engineering | already published in substance by BrickNet; not a 2026 item |

## 6. Venue fit and timeline

**Dual-submission overlap is decisive.** T34 and T83 are already in the CVPR 2027 evaluation
paper: §6.3 "Three domains" in the main text (`tab:lego`, BrickGPT row +11.8 / +51.5) and a
dedicated supplementary section `app:brickgpt`. A separate LEGO paper built from the same
evidence would substantially overlap a paper under review from Nov 16 2026 to Feb 25 2027.
CVPR's dual-submission rule forbids that for any venue with overlapping review periods, and
after acceptance the material counts as prior publication.

- **CVPR 2027 main (Nov 16):** not realistic as a separate paper and not permissible with the
  current evidence. Keep LEGO as the evaluation paper's second domain.
- **CVPR 2027 workshop:** same overlap problem if it reuses T34/T83; non-archival workshops are
  the only option, and they add little.
- **ICCV 2027 (~early March, unconfirmed):** possible only as a *new* paper after the CVPR
  decision: a diverse-part LEGO benchmark (OMR/BrickNet LDraw ladder, BrickGPT, BrickNet,
  BrickBench agents, human study) that cites the CVPR paper for the method. Roughly 3 to 5
  weeks of work, mostly CPU and engineering.
- **NeurIPS 2027 D&B (~mid May, unconfirmed):** the best fit for that extended LEGO benchmark,
  with a release of the voxelised references, probes and scorecards (ST2B is MIT, OMR is CC BY
  4.0 / CCAL 2.0, so a LEGO release is licence-clean, unlike the Minecraft corpus).
- **arXiv now:** no. The LEGO content would appear in the evaluation paper's arXiv version
  anyway.

## 7. Paper outline (for the post-CVPR extension, 8 pages)

1. Introduction (1.0): LEGO generators are judged by renders and VLMs; interiors decide whether
   a model is buildable and solid where it should be hollow.
2. Related work (0.75): BrickGPT, LegoACE, BrickAnything, BrickNet, BrickBench, BC-Bench,
   LEGO-Eval; render and VLM evaluation in 3D.
3. Representations and probes (1.0): ST2B cuboid voxelisation, LDraw part voxelisation, the
   probe ladder (Fill-Interior, chunk delete, material shuffle, monochrome).
4. Metric audit (1.5): seven render statistics, `geom_kid`, a volumetric 3D row, gates, across
   cuboid and diverse-part corpora with seeds.
5. Generators under the suite (1.5): BrickGPT, BrickNet, BrickBench agents; validity,
   stability, both tiers, sealed-interior and connectivity scalars.
6. Human agreement (0.75): 2AFC with exterior and cutaway views.
7. Discussion and release (0.5).
8. Conclusion (0.25); remaining 0.75 for figures and tables beyond those inside sections.

**Figures.** (a) A LEGO structure and its Fill-Interior twin: identical exterior renders, the
cutaway and the `geom_kid` / MV-KID bars side by side. (b) The ladder heatmap: rungs × metrics,
cuboid and diverse-part corpora, cells in noise floors at a stated n. (c) Generator scatter:
render separation against geometry separation for BrickGPT, BrickNet and agents, with real
controls, plus sample renders.

## 8. Verdict

**Strength as a standalone paper: 2 of 5.** The LEGO evidence is real but single-seed, has one
scored generator, and is already committed to the evaluation paper, which it strengthens as its
second domain. The reproductions (T61, T62) and the LEGO-to-Minecraft transfer (T69, negative)
add no publishable finding, and the typed-connection bet is largely pre-empted by BrickNet.
Keep it as a section of the evaluation paper now; revisit as a diverse-part LEGO benchmark for
ICCV 2027 or NeurIPS 2027 D&B after the CVPR decision.

**Most important next experiment:** score BrickNet's existing 512 caption-conditioned samples
(`libs/BrickNet/repro/gen/sft0.6B_val_eos.jsonl`) on the suite through an LDraw voxeliser. It is
CPU-only, uses data already on disk, gives the evaluation paper a second published LEGO
generator, and is the first test of whether the geometry-over-render gap holds beyond BrickGPT.
