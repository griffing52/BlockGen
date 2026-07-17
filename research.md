# Deep-research: landscape, quality levers, web-knowledge rewards (2026-07-02)

Three parallel literature sweeps (existing systems / quality techniques / vision-model
rewards), condensed. Full URLs inline. Feeds notes.md §9-§10 and the paper's related work.

---

## A. Landscape of existing systems (what we benchmark against)

| System | Year/Venue | Representation | Model | Data | Public? |
|---|---|---|---|---|---|
| **LegoGPT/BrickGPT** (CMU) | ICCV 2025 (Marr Prize) | text brick lines `hxw (x,y,z)`, 20³ grid | fine-tuned LLaMA-3.2-1B + validity check + PhysX rollback | StableText2Lego 47k structs | code+data+weights |
| Rollback-Free Brick Gen | 2026, arXiv 2605.06947 | brick tokens | RL post-training (collision/connectivity/interlock rewards), no test-time rollback | StableText2Lego-class | yes |
| **BrickAnything** | 2026, arXiv 2605.26182 | **structure-aware tree tokenization** (attachment relations) — closest relative of our 3D-BPE | geometry-conditioned AR + DPO-style alignment + constrained decoding | point-cloud→brick | arXiv |
| **BrickNet** (Inria/ENS) | CVPR 2026, arXiv 2604.22984 | **graph-backed connectivity programs** — precedent for our Track C | LM over build-sequence programs | 100k+ LDraw objects | yes |
| PVPO LEGO post-training | 2026, arXiv 2606.07602 | LLM brick text | RL (physical feasibility + voxel-geometry rewards); names **"PhysHack"** failure | subset of LEGO data | arXiv |
| TreeSBA | ECCV 2024 | BFS LEGO-tree | tree-transformer from multi-view images | synthetic | yes |
| Brick-by-Brick | NeurIPS 2021 | unit-brick actions | RL assembly from images | — | yes |
| ShapeAssembly | SIGGRAPH Asia 2020 | cuboid DSL programs | hierarchical seq VAE | PartNet | yes |
| World-GAN | CoG 2021 | voxels + block2vec | 3D SinGAN | single-example | yes; **TPKL-Div metric** |
| DreamCraft | FDG 2024 | quantized NeRF over block types | CLIP-guided per-prompt optimization (hours/sample) | none | yes |
| **Scaffold Diffusion** | NeurIPS 2025 WS, arXiv 2509.00062 | sparse occupied-voxel tokens — closest relative of our Track B | masked discrete diffusion (MDLM/DiT), occupancy-conditioned | 1,432 3D-Craft houses | yes |
| **Dream-Cubed** (Sakana) | 2026, arXiv 2604.22847 | dense cube tokens | large 3D diffusion study; **Render-FID** + human study | 2M+ MC chunks | code+weights+data |
| VoxelCNN / **3D-Craft** (FAIR) | ICCV 2019 | voxels + human build order | order-aware CNN | **2.5k ordered MC houses** | yes |
| text2mc | 2024 | dense voxel .h5 | VAE | **~11k builds (Kaggle, up to 40k)** | yes |
| MarioGPT | NeurIPS 2023 | text-serialized tiles | fine-tuned DistilGPT-2 | Mario levels | yes |
| LLaMA-Mesh (NVIDIA) | 2024 | raw OBJ text (64-bin coords) | SFT LLaMA-3-8B | mesh corpus | yes |
| MC-Bench / MineBench | 2024-26 live | prompt→code/JSON blocks | zero-shot frontier LLMs, human Glicko voting | benchmark | yes |
| GDMC settlement challenge | annual | world edits | procedural/search | — | human 0-10 rubric |

Key URLs: LegoGPT https://arxiv.org/abs/2505.05469 · BrickAnything https://arxiv.org/abs/2605.26182 ·
BrickNet https://arxiv.org/abs/2604.22984 · Scaffold https://arxiv.org/abs/2509.00062 ·
Dream-Cubed https://arxiv.org/abs/2604.22847 · 3D-Craft https://github.com/facebookresearch/voxelcnn ·
text2mc https://www.kaggle.com/datasets/shauncomino/minecraft-builds-dataset

### The text-serialized-LLM baseline (benchmark plan)
Strong precedent — **reviewer-expected**: LegoGPT (LLaMA-3.2-1B on brick text), MarioGPT,
LLaMA-Mesh, SuperVoxelGPT (2605.29655), BrickNet all fine-tune pretrained LMs on
serialized structures. Feasibility on our 5070 Ti 16GB: LoRA on LLaMA-3.2-1B <5 GB VRAM,
1-4 h for ~3 epochs on ~30M tokens; QLoRA fits up to ~14B. A 16-20³ build ≈1.5-7k tokens
as `block x y z` lines (needs block-ID indexing to stay compact). **Recommended track D:**
(1) LLaMA-3.2-1B-Instruct LoRA (exact LegoGPT base → comparability), (2) Qwen3-4B QLoRA
(upper bound of "does web knowledge help"), (3) optional MineBench-style zero-shot
frontier-LLM JSON emission.

### Metrics to add for comparability
Render-FID (Dream-Cubed), CLIP score on renders ("a Minecraft medieval house"),
floating-block/support-stability rate (voxel analogue of LegoGPT % stable),
TPKL-Div + block histograms (World-GAN), COV/MMD/1-NNA over IoU distance, small human
pairwise study; memorization z-score per arXiv 2512.23628 (our val-baseline protocol maps
directly). Our validity+dup+val-split suite is already more rigorous than Scaffold/VoxelCNN
— a claimable contribution.

---

## B. Quality levers, ranked (evidence → our failure mode: right gestalt, incoherent up close)

1. **Factor occupancy → materials** (HIGHEST). Scaffold Diffusion: occupancy-conditioned
   MDLM over only occupied voxels gives coherent houses; **their AR-over-materials baseline
   collapses to few block types — literally our block-agreement 0.03-0.13 failure**; dense
   grid + inverse-class-frequency loss also fails. Their ablation: learned positional
   embeddings PPL 29.05 vs **3D sinusoidal PPL 1.79** → audit our PEs. They leave
   "generate the occupancy too" as future work → doing both stages is claimable. XCube
   (CVPR 2024) uses the same factoring at scale. Cost ~2-4 days on our MaskGIT code.
2. **Constrained AR decoding**: mask logits to positions 6-adjacent to existing structure
   (+ ground layer), pair with **BFS-from-ground training order** (so the constraint is
   true in training too; VoxelCNN shows humans build connected/local), plus
   LegoGPT-style validity-check + rollback-resume. Replaces post-hoc LCC repair with
   in-loop enforcement (later tokens condition on valid context). Inference-only, ~1 day.
3. **Masked-diffusion sampling upgrades** (free, inference-only): Gumbel-noise confidence
   sampling with annealed temperature (pure argmax-confidence is known-worse), **Halton
   scheduler** (ICLR 2025, de-clusters unmask positions; arXiv 2503.17076), **ReMDM**
   remasking (arXiv 2503.00307) so committed errors are revisitable; more steps (we're
   small). Token-Critic (ECCV 2022) as learned gate if needed.
4. **Two-level pyramid** coarse→fine (PDD, ECCV 2024 oral, arXiv 2311.12085; explicitly a
   small-data play with cross-dataset coarse transfer). 4³/8³ occupancy → full-res refine.
5. **Patch-VQ latent tokens vs 3D-BPE** (AutoSDF P-VQ-VAE, MeshGPT): one focused ablation
   arm; use FSQ/binary quantization (codebook collapse risk at our data size). Frames the
   paper axis: raw coords vs handcrafted 3D-BPE vs learned codes. BPT (~75% mesh token
   compression, CVPR 2025) validates compressive tokenization as a contribution axis.
6. **Discrete classifier-free guidance** for the conditioned tracks (Schiff et al., ICLR
   2025, https://discrete-diffusion-guidance.github.io/): 10-20% condition dropout at
   train, guide logits at sample; ramp guidance late (early high guidance hurts masked
   models). Sweep w∈{1,2,3,5}.
7. **Token ordering**: raster is a strong default; **bottom-up raster / BFS order** is the
   right pairing with #2; RAR-style random-order-anneal as regularization. Hilbert: skip.
8. **Small-data bundle**: EMA (Scaffold uses 0.9999), dropout 0.1-0.3 + wd 0.05-0.1 +
   early stop on val NLL; ≤4 epochs of repeats ≈ fresh data (arXiv 2305.16264) justifies
   D4×multi-epoch; **pool-pretrain one model on everything (all GrabCraft + 3D-Craft +
   text2mc ≈ 15-45k builds, category token), fine-tune per subset — likely the biggest
   single small-data win we haven't tried.**

Cross-cutting checks: (i) PE audit all tracks (learned vs 3D sin-cos); (ii) never generate
air tokens (consistent with our BPE design); (iii) AR-over-dense-materials collapses at
this scale — factor or constrain, don't add capacity.

---

## C. Web-knowledge rewards / RL (the "discriminator that has seen a million houses" idea)

**Verdict: sound and now-published direction (2026 brick-RL papers do validity rewards;
none do visual-semantic rewards on Minecraft builds — open niche), but data scaling comes
first: every RL result assumes a competent base policy.**

- **Reward models**: **MineCLIP** (MineDojo, 640k YouTube MC clip-text pairs, open
  weights) — shown to agree with human judges on "built a house with a pool"-type
  building tasks; expects 16-frame POV video → render a camera orbit with MC textures.
  Generic CLIP works on Minecraft-style renders (DreamCraft evidence) with augmentations
  (PureCLIPNeRF: augmentations are the main defense vs adversarial samples). VLM pairwise
  judging (GPTEval3D protocol): use pairwise + position-swap, not absolute scores.
- **Algorithms**: GRPO on AR generators (AR-GRPO 2508.06924; STAGE 2509.25027 adds
  KL/entropy stabilizers), DDPO/DPOK for diffusion (~20k reward queries for visible
  gains), **d1/diffu-GRPO** (2504.12216) for masked discrete diffusion, offline
  **DPO with VLM-judged pairs** (Diffusion-DPO 69% human-preferred) as the stable/cheap
  variant. Nabla-R2D3 (NeurIPS 2025): RL-aligning 3D generators with 2D rewards on
  renders — our exact shape, in mesh domain.
- **Failure modes to design around**: reward hacking (keep KL-to-SFT + early stop),
  diversity collapse (entropy bonus; watch our duplicate_rate — RL will push it up),
  **PhysHack** (validity satisfied, semantics degrade → pair visual reward with
  programmatic checks and vice versa), CLIP domain gap (use MC textures / MineCLIP).
- **Renderers**: fast non-differentiable is enough for GRPO/DPO — trimesh/pyrender cube
  meshes with texture-averaged block colors, ~10-50 ms/frame; mcrender (Mineways+Blender)
  for pretty figures; nvdiffrast only if we ever do differentiable-reward backprop.

**Ranked pipelines for us**
0. **Data scaling first**: pretrain on text2mc (11-40k) + 3D-Craft (2.5k) + CraftAssist
   (filtered) with Cap3D-style VLM auto-labeling (Cap3D captions judged better than
   human) → fine-tune on curated GrabCraft. Strongest evidence, lowest risk, likely fixes
   cohesion more than any reward.
1. **GRPO + reward ensemble** (w1·CLIP-category + w2·MineCLIP-orbit + w3·programmatic
   validity/support): 20-50k generations ≈ 1-3 GPU-days on a 10-50M AR model. Best
   gain-per-effort for "more house-like".
2. **Offline DPO with VLM pairwise judgments** (5-20k pairs, rubric prompts): stabler,
   <$100 API or local Qwen2.5-VL; hours of training.
3. Differentiable reward backprop (DRaFT-style, 25× sample-efficient) via Gumbel-softmax +
   nvdiffrast — only if 1-2 plateau (adversarial-CLIP risk, heavy engineering).
4. SDS/DreamCraft-style offline data amplification, filtered by MineCLIP → extra training
   data (turns per-sample optimization into one-time data cost).

Key URLs: MineCLIP https://github.com/MineDojo/MineCLIP · DDPO https://arxiv.org/abs/2305.13301 ·
d1 https://arxiv.org/abs/2504.12216 · AR-GRPO https://arxiv.org/abs/2508.06924 ·
STAGE https://arxiv.org/abs/2509.25027 · Nabla-R2D3 https://arxiv.org/abs/2506.15684 ·
DreamCraft https://arxiv.org/abs/2404.15538 · Cap3D https://arxiv.org/abs/2306.07279 ·
GPTEval3D https://arxiv.org/abs/2401.04092 · PVPO https://arxiv.org/abs/2606.07602

---

## D. Synthesis: the best-novel-approach bet

No published system combines (a) **connectivity-native tokenization** (our 3D-BPE pieces /
block+port graph), (b) **occupancy→materials factoring with a generated occupancy stage**
(Scaffold's stated future work), (c) **constraint-aware decoding** (adjacency gating +
rollback), and (d) **post-training against a domain-matched visual reward (MineCLIP) +
programmatic validity ensemble** — on **small curated data with honest memorization
accounting**. The 2026 brick-RL line (PVPO, Rollback-Free) proves RL-on-structures but
uses only physics/geometry rewards; DreamCraft proves CLIP-on-blocks but per-instance;
Scaffold proves factoring but no occupancy generation and no RL. The stack above is
individually de-risked by strong precedent at every step and jointly novel.

Suggested order (each step paper-able alone):
week 1 free wins (PE audit, adjacency-gated decoding + BFS order, Gumbel/Halton/ReMDM
sampling) → week 2 two-stage occupancy→materials + LLM-baseline track D → week 3 pooled
pretraining (text2mc/3D-Craft) → week 4 DPO-or-GRPO with MineCLIP+validity ensemble.

---

## E. Deep research 2026-07-07: LEGO endgame, the post-LegoGPT landscape, and our bet

Verified-claims sweep (adversarial 3-vote protocol; key LDraw claims hand-verified).
Question: novelty space for buildable-structure generation targeting a top venue,
esp. LEGO with NON-cuboid parts.

### E.1 The field moved fast — what's now taken

- **BrickGPT (né LegoGPT)**, CMU — **ICCV 2025 Best Paper (Marr Prize)**: LoRA
  LLaMA-3.2-1B, next-brick TEXT tokens, in-loop validity check + physics rollback →
  "first physically stable brick assembly from text". **8 cuboid brick types, 20³ grid**;
  slopes/tiles explicitly future work. Released StableText2Brick (47k structures / 28k
  objects, captions). https://arxiv.org/abs/2505.05469
- **LegoACE**, SIGGRAPH Asia 2025 (ZJU+VAST+Tsinghua+KAUST+HKU): **native per-brick
  tokens** (pos+rot+type, 5 tok/brick, 48 rotations) on a LLaMA decoder — explicitly
  rejects text serialization ("semantic gap … limits diversity/fidelity"). Frozen **CLIP
  text + DINOv2 multi-view conditioning, plus DPO** (⇒ our planned recipe is published).
  **LegoVerse: 55k models / 9,314 brick types** incl. wheels/doors/windows; code+models
  released (dataset source/license in supplementary — CHECK). **No hard constraints:
  82% connected rate; guaranteed buildability conceded as future work.**
  https://dl.acm.org/doi/10.1145/3757377.3763881
- **BrickAnything** (arXiv 2605.26182, May 2026): point-cloud-conditioned AR with
  **structure-aware TREE tokenization** (BFS over attachment graph, parent-relative
  attachment tokens) + **best-of-N → reward-weighted DPO** (IoU+Chamfer+StableLego
  stability) + constrained decoding/rollback (rollbacks 6.75→0.42, %stable 76→83.4).
  **Still 8 cuboid types, 20³.**
- **BC-Bench / Brick-Composer** (arXiv 2606.05445, June 2026): first MLLM benchmark for
  DIVERSE-brick assembly (LDraw, SE(3) poses). Frontier MLLMs **<1%** strict step
  success zero-shot; physically-grounded finetuning → ~15%; ≤42% of steps of a full
  object. Their real data: **102 files** (copyright bottleneck).
- Minecraft side: **Scaffold Diffusion** landed at a NeurIPS-25 **workshop** (SPIGM) —
  Minecraft-only discrete diffusion without stronger eval is workshop-tier. Its claim
  "AR collapses at >98% sparsity" is directly rebuttable by our phase4+constrained AR.
  **VoxelCodeBench** (FAIR, Apr 2026) = the text/code-encoding wave: LLMs write Python
  to build voxel scenes; finds *executable ≠ spatially correct*. **No standard
  structure-generation benchmark exists** — everyone uses ad-hoc metrics.
- Judge-guided 3D: published for CONTINUOUS 3D (VLM critics, de-biased VLM-judge
  protocols, 2026); **LEGO-Eval caution: naive VLM judges ≈ zero human agreement on
  fine-grained 3D** — tool-augmented judging required. Judge-guided *discrete* remains
  open but adjacent-published.

### E.2 The open intersection (our bet)

**Hard-guaranteed validity × diverse non-cuboid parts.** The guarantee line (BrickGPT,
BrickAnything) is stuck at 8 cuboid types; the diverse-parts line (LegoACE, 9,314 types)
has no guarantees and names them future work; the MLLM line barely functions. Nobody has
constraint-guaranteed decoding over a rich part vocabulary — and in-loop constrained
decoding is our proven strength (validity 1.0 on Minecraft).

**The enabling data artifact exists**: the **LDCad Shadow Library** (GitHub) adds
machine-readable `!LDCAD SNAP_*` connectivity metas (studs, anti-studs, clips, axles,
pins) to official LDraw parts — connection semantics do NOT have to be inferred from
mesh geometry. Caveats (hand-verified): covers "a growing subset" of the 18k+ part
library (restrict vocab to covered parts — still ≫8); page says all-rights-reserved
(fine as train-time tooling; don't redistribute; confirm GitHub terms).

**Representation claim that beats "tokens + checker": typed-connection tokens.** Emit
each part as (part-type, parent-connector, own-connector, discrete orientation) — pose
*derived* from the connection, so invalid placements are *inexpressible* (validity by
construction, not by rollback). BrickAnything's tree tokens still carry offsets and
still need rollback (0.42/model); ours would be closed under buildability. This is the
phase4 philosophy (bake the grammar in) elevated to the connection graph — one
coherent story from our Minecraft results to LEGO. Collision still needs a fast check
(voxelized part-occupancy proxies); hinges/continuous angles deferred (discrete
rotation set first, like LegoACE's 48).

**Cross-medium transfer as the ML-venue framing**: pretrain on ~16k Minecraft builds
(connection grammar = 6-adjacency, trivial case), transfer to LEGO's typed-connector
grammar — "one constrained-assembly formulation, two mediums", plus our honest-novelty
protocol as the eval story and a proposed benchmark (the field demonstrably lacks one).

### E.3 LEGO data feasibility verdict — YES, with care

- **StableText2Brick** 47k (cuboid, captions) — public, immediate pretraining fodder.
- **LegoVerse** 55k / 9,314 types — **NOT released** (hand-verified 2026-07-07: project
  page offers paper/supp/code only; no dataset link, no sourcing statement, no data
  license). Their headline advantage is a private-data moat → results not reproducible,
  not fairly comparable; the open-data slot in this field is EMPTY.
- **LDraw OMR**: official sets, full part+pose (`1 color x y z rot part.dat`), every
  file **CC BY 4.0** (verified: `0 !LICENSE Licensed under CC BY 4.0`) — the
  license-clean core, includes Technic; size ~low thousands of sets.
- **Rebrickable dumps**: metadata/BOM only (no poses) — joins, not geometry.
  **BrickLink Studio/MOC files**: copyright-constrained (BC-Bench managed only 102).
- **Connectivity**: LDCad shadow library (above). Part geometry: LDraw primitives+mesh.

**Fetched 2026-07-07 → `data/lego/` (see its README; Minecraft moved to
`data/minecraft/` with compat symlinks):** LDraw complete library (33,362 part
`.dat`); shadow library GitHub clone — **CC BY-SA 4.0 in-repo** (supersedes the
website's all-rights-reserved note), 3,334 part patches, 2,859 with `SNAP_CYL/GEN`
metas (typed + gendered + positioned connectors — the typed-connection-token
ingredients; some common parts get connectivity via template includes, resolve in
parser); StableText2Brick from HF `AvaLovelace/StableText2Brick` — **MIT**, ~42k
train + test parquet, text brick lists + captions + per-brick stability; OMR via
`blockgen/data/fetch_omr.py` (resumable scraper; ~25 sets/page + alternates).
License nuance: older OMR files carry **CCAL 2.0** rather than CC BY 4.0 — both
attribution-redistributable; keep per-file license from the header in the manifest.

### E.4 Expected issues

Rotation/articulation DOF (hinges, turntables) — start discrete; mesh-level collision
cost — voxel proxies; shadow-library gaps — vocab restriction + report coverage;
stability for clips/axles has no StableLego analogue — empirical (PyBullet) not
guaranteed, claim *connection validity* guarantee only; 16 GB GPU — small decoders fine
(5 tok/brick ⇒ 55k models is tractable); **speed of the field** — 3 major papers in 12
months, stake the typed-connection claim early (workshop paper / arXiv preprint).

### E.5 Priority call (2026-07-07)

1. **LEGO data + connection grammar first** (gating item): fetch OMR + StableText2Brick
   (+LegoVerse if released), parse LDraw + shadow library → typed-connection graphs.
   We just built exactly this muscle (multi-corpus curation + unified vocab).
2. In parallel, cheap Minecraft runs that feed the story: phase4+constrained combo;
   pool-pretrain → finetune (doubles as the transfer-learning evidence).
3. Model bet: typed-connection AR (phase4-style grammar PE over connection tuples).
4. VLM/MineCLIP judge + DPO: keep as quality tooling — no longer a novelty claim
   (LegoACE, BrickAnything published it).
