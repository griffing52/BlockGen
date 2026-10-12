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


Generative Modeling of 3D Modular Structures: Leveraging Attention Mechanisms as Implicit Adjacency Matrices for LEGO and Minecraft TopologiesThe pursuit of generative artificial intelligence capable of synthesizing complex, physically viable three-dimensional modular structures represents a significant frontier in both machine learning and computational geometry. Distinct from continuous 3D mesh generation, the synthesis of discrete modular environments—such as LEGO brick assemblies and Minecraft voxel worlds—requires models to simultaneously respect global semantic coherence and strict, local topological constraints. Traditional generative approaches that rely on coordinate regression or dense voxel grids frequently fail in these discrete domains due to accumulating spatial errors and the persistent violation of rigid physical rules.The hypothesis under investigation posits that the attention matrices inherent in pretrained Transformer architectures can be systematically repurposed or finetuned to act as implicit adjacency matrices, thereby treating discrete structural components as interconnected nodes within a relational graph. This approach theoretically circumvents the severe limitations of Euclidean coordinate generation by directing the model to learn the topological connectivity of components rather than their absolute positions in space. An exhaustive analysis of recent advancements in graph neural networks, graph-augmented language models, structure-aware tokenization, and physics-aware decoding frameworks reveals that this theoretical framework is not only feasible but aligns seamlessly with the cutting edge of contemporary Graph Transformer research.The Taxonomy of Modular Environments: Voxel Grids versus Relational GraphsTo construct a unified generative model capable of synthesizing both LEGO structures and Minecraft environments, it is essential to first resolve the fundamental geometric distinctions between the two mediums. While both environments rely heavily on the assembly of discrete primitives, their spatial logic dictates vastly different representational requirements.Minecraft structures are inherently defined by an absolute, uniformly spaced Cartesian coordinate system. The world space is a discrete three-dimensional grid where each coordinate, heavily constrained along the vertical axis, contains exactly one block type or atmospheric void. Prior generative models for Minecraft, such as World-GAN, have treated this environment similarly to three-dimensional image generation, utilizing Generative Adversarial Networks or diffusion models to generate continuous RGB values that are subsequently mapped back to the nearest categorical block identifiers. Because the grid is regular and isotropic, adjacency is implicitly defined by Euclidean proximity; a block is adjacent only to its six orthogonal neighbors, and there are zero degrees of rotational freedom.Conversely, the generative modeling of LEGO assembly shatters the assumptions of a simple voxel grid. While early experimental datasets for LEGO generation artificially restricted models to discrete grids and single brick types to simplify the computational load, real-world human designs encompass thousands of distinct parts with diverse, typed connections. Generating complex LEGO structures, such as a biological organism or a wheeled vehicle, requires tracking the six-degrees-of-freedom continuous-discrete pose of each element. In the digital modeling standard LDraw, positional units are expressed in LDraw Units, where one unit equals approximately 0.4 millimeters, and standard connection points like a stud have a diameter of six units. Attempting to autoregressively predict absolute three-dimensional coordinates for such parts leads to a rapid degradation of precision. As the generative sequence progresses, minute floating-point errors accumulate along shifted rotational frames, inevitably causing structural collisions, disconnected floating bricks, and the complete failure of the generation.To unify these disparate modalities into a single foundation model, absolute coordinate representation must be abandoned entirely in favor of topological graphs. In this unified framework, an object is modeled as a directed, typed connectivity graph. Nodes represent individual components, while node attributes consist of categorical tokens identifying the part type, dimensions, and visual metadata. Edges represent the local attachment relations or physical connections. Under this abstraction, a Minecraft block is simply a highly constrained structural node possessing exactly six orthogonal connection ports with rigid alignment, rendering it a specialized, simplified subset of the broader, more complex LEGO graph grammar.FeatureMinecraft Spatial ParadigmLEGO LDraw Spatial ParadigmUnified Graph Representation StrategySpatial ModelDiscrete Cartesian GridContinuous-Discrete 6-DoF PoseRelational Graph (Nodes = Parts, Edges = Connections)PrimitivesIsotropic, uniform cubesAnisotropic, multi-scale meshesFeature-attributed node tokensConnectivityOrthogonal planar facesStuds, hinges, axles, pins, clipsTyped edge relationships governing local transformsGenerative RiskVoxel artifacting and aliasingCumulative coordinate driftTopological cycles, unbuildable sub-graphsAttention Mechanisms as Implicit Adjacency MatricesThe core proposition—that the attention mechanism of a pretrained model can serve as an adjacency matrix for three-dimensional component nodes—is structurally sound and actively supported by current research investigating the internal mechanics of Large Language Models. Standard language models process tokens in a one-dimensional sequence, but when tasked with graph data, their internal mechanics naturally attempt to reconstruct non-Euclidean topologies.Recent forensic analyses of Large Language Models processing serialized graph data reveal a critical phenomenon: these models spontaneously reconstruct the graph's topology internally. This reconstruction is evidenced by a distinct "sawtooth" pattern emerging within their attention maps, which directly aligns with the token-level adjacency matrix of the input graph. However, in standard autoregressive Transformers, this structural understanding is often diluted by the "attention sink" phenomenon, a dynamic where the model's anisotropic bias forces an overwhelming amount of attention toward initial tokens, inadvertently suppressing the topology-aware local aggregation required for advanced spatial reasoning.To leverage this spontaneous topological mapping for generative assembly, researchers have developed frameworks to explicitly inject graph-aware attention biases directly into the attention modules. The standard self-attention mechanism computes the attention matrix by applying a scaled dot-product between query and key projections. When treating modular components as nodes, this attention matrix must be mathematically coerced to represent the physical adjacency matrix. Architectures such as Graphormer and its derivatives achieve this by introducing spatial and structural encodings directly into the attention formulation. If a function represents the spatial relation, such as the shortest path distance on the structural graph between two nodes, and another variable represents the typed edge feature, the attention matrix is modified by adding learnable scalar biases indexed by the spatial distance and edge-encoding biases.By initializing these bias parameters and finetuning the model on vast physical structure datasets, the attention matrix explicitly mimics and learns the adjacency matrix, allowing the model to attend intensely to structurally connected bricks rather than just sequentially preceding language tokens. Furthermore, frameworks like GraphGPT demonstrate how higher-order graph information can be injected into this process. By taking the self-attention matrix, treating it explicitly as an adjacency matrix, and injecting higher-order structural information via Taylor expansions, models can learn complex multi-hop structural dependencies. For a generated structure, this means the model understands not just the immediate block it is currently attaching, but the ongoing structural integrity and geometric trajectory of the entire connected branch.To resolve the suppression caused by attention sinks, training-free inference interventions such as StructuraL Attention SHarpening have been proposed. This methodology redistributes attention away from generic initial tokens and sharpens the attention weights strictly along the learned adjacency matrices, forcing the model to heavily weight structurally connected precursor nodes when predicting the next modular placement.Graph Serialization and Structure-Aware TokenizationTransformers mandate sequential input, requiring that the three-dimensional connectivity graph be flattened into a linear format. However, naively listing the nodes or spatial coordinates destroys the relational logic necessary for physical assembly. Recent breakthroughs in generative brick assembly solve this by defining a compact parametrization of structure that represents spatial relationships between parts exclusively through a graph of their connectivity.Arbitrary structures are serialized as a spanning tree over a typed connectivity graph. Instead of representing a sequence of absolute coordinates, the serialization process utilizes structure-aware tree tokenization. In this sophisticated tokenization scheme, components are organized according to local attachment relations rather than global spatial ordering. A token sequence is constructed where each generative step defines the parent node in the existing assembly to which the new component attaches, the specific part token identifying the type of the new component, and the connection token detailing the exact physical connection and discrete geometric offset.By serializing the connectivity graph into a spanning tree, each generated edge becomes a discrete instruction that can be executed deterministically to recover the exact spatial transforms. This effectively bypasses the accumulation of floating-point errors. For complex structures with diverse connections, the tokenization must account for different families of connections. For instance, a basic stud connection allows one degree of rotational freedom, requiring the model to generate a rotational yaw parameter. Hinge connections must encode boolean flip variables alongside rotation, while axle connections require parametrizing the specific slide along the geometric axis. By treating three-dimensional assembly as a specialized language of these connections, the model learns the structured, relational logic of graphs.Integrating Three-Dimensional Relative Positional EncodingWhile topological distance and node hops are critical for ensuring valid connections, physical three-dimensional space cannot be entirely ignored. Relying solely on topological connectivity can lead to structures that intersect themselves across different branches of the spanning tree, as the model might place two distinct, topologically distant branches into the same physical coordinate space. To ensure overall geometric fidelity and prevent self-intersections, the model must simultaneously possess an awareness of Euclidean geometry.To resolve this, the proposed architecture must integrate three-dimensional Relative Positional Encoding into the multi-head self-attention mechanisms. By computing the relative spatial vector between the centroids of any two generated components, the attention mechanism is provided with continuous spatial context. This relative offset is encoded and infused into the attention calculations alongside the topological graph biases. This dual-encoding approach—combining discrete topological adjacency biases with Euclidean relative positional encodings—ensures the model fundamentally understands both the mechanics of how the components are physically connected and where they reside relative to one another in the global coordinate space.Physics-Aware Decoding and Computational SimulationGenerating a sequence of tokens is only half the challenge in modeling physical reality; ensuring that the generated graph is physically stable, buildable, and semantically meaningful is a notorious hurdle. Autoregressive language models will inevitably occasionally predict a token that constitutes a physically impossible connection, such as attempting a stud connection where the target surface is flat, or generating a piece that clips through the existing structure. Relying purely on the model's learned distribution is insufficient for the strict, unforgiving rules of physical interaction.State-of-the-art systems implement an explicit algorithmic loop during the inference decoding phase, a process termed validity-constrained decoding or physics-aware rollback. At each generation step, the proposed structural token is mapped back into a lightweight computational simulator or a rigid-body adjacency checker. To perform highly accurate collision detection, the system must expand non-convex meshes to check for volume intersections. Because real-world plastic deforms slightly when pressed together, systems apply techniques to inset mesh faces by fractions of a unit to account for minute deformations, preventing the collision checker from producing false positives.If an unstable, unconnected, or intersecting component is detected by the simulator, the generative process immediately halts. A rollback mechanism triggers, reverting the sequence to the parent component in the tree tokenization structure, pruning the infeasible token prediction from the probability distribution, and forcing the model to resample an alternative structural continuation. This neuro-symbolic approach guarantees that every output is mathematically buildable. Without this parent-aware rollback mechanism, the probability of generating a flawless, multi-part object drops exponentially with sequence length, as single errors compound fatally.Beyond simple geometric connectivity, a model must understand structural statics. A structure may be mathematically connected but physically unstable under gravitational load. Recent developments include physics-based simulators designed explicitly for interlocking brick assemblies, which introduce compact force-based mechanics models for snap-fit connections. These simulators solve the resulting internal force distributions using structured convex quadratic programs, enabling real-time, high-fidelity simulation of assembly and structural collapse. Integrating structural stability solvers into the generative loop allows for deep analysis of whether a generated topology will sustain its own weight.Mitigating the Misalignment of Physics and SemanticsOptimizing a generative model to strictly follow physical rules introduces secondary challenges, primarily the misalignment between physical validity and semantic intent. A critical failure mode identified in reinforcement learning for modular assembly is the phenomenon where models discover adversarial shortcuts to maximize their physical reward signals.When models are optimized heavily or exclusively on physical validity and stability rewards, they learn that generating a solid block of bricks is highly stable and physically flawless, completely ignoring the semantic intent of the user prompt. For example, if tasked with generating a jar or an animal, the model might produce a dense, unidentifiable cuboid. Models exhibiting this phenomenon may achieve near-perfect physical validity metrics, yet their structural and semantic alignment scores plummet.To prevent the model from collapsing into trivial, highly stable geometries, the reward function must heavily incorporate multi-modal semantic valuation. During post-training, filtering datasets to preserve domain diversity and utilizing visual grounding ensures that the topological graphs generated by the attention matrix actually resemble their intended real-world counterparts. By comparing generation outputs against Vision-Language Models, researchers can utilize Direct Preference Optimization to align the model. In this setup, preference pairs are constructed where the model is explicitly taught to favor generation trajectories that maximize both geometric fidelity to the prompt and the physical buildability of the structure.Evaluation MetricQwen-based Validity FocusVision-Language Semantics Focus (Direct Preference Optimization)Physical Validity@40.93 - 1.00 (Near Perfect)0.85 - 0.90 (Highly Stable)Gemma@4 (Semantics)0.39 (Poor Alignment)0.63 (Strong Alignment)MiMo@4 (Semantics)0.68 (Moderate Alignment)0.84 (Strong Alignment)Primary Failure ModeFeatureless, highly dense cubesOccasional minor overhang instabilityImplementation Strategy and Feasibility AssessmentEvaluating the core proposition—building a generative model for discrete structures utilizing pretrained attention as adjacency matrices—yields a conclusion of high theoretical viability and practical feasibility. The necessary mathematical, representational, and architectural components have recently been validated across disparate domains of artificial intelligence research. To actualize this system, an integrated development pipeline must be established, synthesizing massive dataset curation, architectural modification, and reinforcement alignment.The foundation requires the curation of a massive data engine. Developers must extract structures from extensive collections, such as the BrickNet dataset which encompasses hundreds of thousands of human-designed samples, tens of thousands of part variants, and millions of placed components. Concurrently, Minecraft schematic databases must be parsed, employing voxel extraction to capture robust environments. All objects must be abstracted into topological graphs and serialized into sequence data using structure-aware tree tokenization.The architectural phase involves selecting a robust open-source autoregressive language model or Vision-Language Model equipped with a deep context window. The attention mechanisms must be explicitly modified to introduce spatial encoding biases, thereby directly injecting graph topology into the attention calculations and forcing the attention weights to function as learnable adjacency matrices. Furthermore, the network must be equipped to handle three-dimensional relative positional encodings to maintain global geometric awareness.Training proceeds in two distinct phases. Unconditional pretraining on the vast corpus of spanning trees teaches the model the fundamental syntax of structural connectivity and physical topologies. Following this, the model undergoes supervised fine-tuning and Direct Preference Optimization. By utilizing a joint reward function that measures semantic adherence via a visual evaluator and structural stability via a physics simulator, the training process effectively mitigates the risk of mode collapse and semantic misalignment.Finally, the deployment phase necessitates the integration of an inference wrapper that acts as a localized physics engine. As the model autoregressively predicts each node and edge token, the wrapper attempts to execute the instruction within a simulated environment. If a collision or impossible geometry is detected, the wrapper prunes the token, triggers a parent-aware rollback, and forces the model to sample the next highest probability branch, guaranteeing physical viability.ConclusionThe conceptualization of modeling modular three-dimensional structures as relational graphs, and governing their generative synthesis by adapting Transformer attention mechanisms into implicit adjacency matrices, represents a highly sophisticated and fully realizable approach to artificial intelligence-driven geometric synthesis. By discarding absolute Euclidean coordinate regression in favor of structure-aware tree tokenization, the persistent issue of spatial drift and cumulative coordinate error is permanently eliminated.The inherent properties of self-attention in large language models naturally incline toward topological mapping, a trait that can be explicitly harnessed and amplified via spatial encodings and attention sharpening interventions. When coupled with a rigid, physics-aware rollback mechanism during inference, this graph-theoretic architecture promises the ability to reliably synthesize vast, complex, and physically stable Minecraft architectures and LEGO assemblies from natural language or visual prompts. As the boundaries between discrete modular geometry and language modeling continue to blur, this paradigm stands to become the definitive standard for the procedural and generative synthesis of modular physical environments.

## F. Data sourcing at scale: more builds, a releasable subset, and world-crop pretraining (2026-10-07)

Question: where can we get far more Minecraft builds than the ~50k we hold, which of it
could we legally release, and is "download tons of worlds, crop out the human-made parts"
a viable pretraining corpus? Four web surveys (datasets, map hosts, extraction tooling,
licensing) plus a local audit. Verified facts are marked [V]; estimates [I].

### F.1 What we hold, and one thing we had not noticed

Inventory is in `data_sources.md`. Two audit findings:

- **text2mc already did the world-crop approach.** `processed_build_dataframe.csv` has
  37,989 rows over 25,600 PlanetMinecraft pages. 9,138 rows are `mca` (world downloads,
  .zip/.rar archives, 130 GB total, median 1.7 MB) that its `world2vec.py` cut into
  **19,394 `.schem` crops**; a further 3,251 rows are standalone schematics yielding 4,982.
  So half of the "40k builds" are machine-cut world fragments, which is why they are large
  and ground-stripped. Their cutter [V, github.com/shauncomino/text2mc-dataprocessor]:
  anvil-parser, skip chunks with low `InhabitedTime`, find the surface section, mark a
  chunk as "build" if the 4 sections above the surface contain any block outside
  `natural_blocks.txt`, DBSCAN(eps=5, min_samples=5) over chunk coordinates, export each
  cluster. Chunk-granular, no vanilla-structure masking. It is a working baseline we
  can measure against.
- The raw PlanetMinecraft tfrecord crawl (`data/minecraft/more/`) is **no longer on
  disk**; only the `tf_small_24` cache survives. The upstream is rom1504's
  minecraft-schematics-dataset on GitLab (tfrecords + `schematicsWithFinalUrl.json`).
- Disk: **98 GB free** of 914 GB. Any world-scale ingest has to stream (download, crop,
  delete) or go to a new drive.

### F.2 Sources, ranked by usable volume

| Source | Volume | Format | Access | Per-item licence | Verdict |
|---|--:|---|---|---|---|
| **Minecraft Archive Project (MAP)** worlds, archive.org, Leonard Richardson, 2011–2018 [V] | 89 `Worlds_YYYYMM` zips, ~3.6 TB, ~162k archive files / 175k worlds from PMC, Minecraft Forum, MediaFire, Dropbox | world saves + schematics, pre-1.13 numeric ids mostly | plain HTTP from IA; each zip has an HTML finding aid | none (archival deposit, no rights grant) | **Largest ready corpus by far.** Pretraining only; not redistributable |
| **MCGS** (Minecraft Geologic Survey, same author) [V, mcgs.crummy.com] | 12 GB → 50 GB: a ~1% chunk sample of all 175,345 MAP worlds, re-saved in 1.8 Anvil, chunks scored by entity/sign/chest density, `manifest.json` per world with sign/book text and original coordinates | Anvil 1.8 | IA download | none | **Ideal pilot**: one download, every world represented, biased toward inhabited chunks |
| CurseForge Worlds [V] | 6,452 projects | world zips | REST API (key), `allowModDistribution` flag, `downloadUrl` null when author opts out | author-chosen licence dropdown (CC variants, ARR, custom); API ToS forbids caching/redistribution of API data | **Only host with machine-readable licence + distribution consent** → the release tier |
| PlanetMinecraft projects | likely 10^5+ maps [I] | schematics hosted on PMC; worlds often external mirrors | no API, Cloudflare, robots disallows `/api/`; ToS §5/§7/§9 forbid systematic extraction and redistribution [V, 2020 terms] | **no licence field for maps** (only skins/texture packs have the CC chooser) [V] | Already mined twice (rom1504, text2mc). Not for release; re-crawl is a ToS problem |
| Hermitcraft season worlds [V] | ~18 worlds (vanilla S1–S10, modded S1–S8) | world saves, hundreds of dense expert builds each | direct download | none stated | High-quality pretraining; ask the Hermits for permission (Dream-Cubed model) |
| 2b2t.place + IA 2b2t dumps [V] | 15 TB compressed (80 TB raw), older 300–700 GB items | custom `.zvcr` + Anvil | torrent | none | Too big, griefed, offensive content; skip |
| minecraft-schematics.com | ~28k items [I from id counter] | .schematic/.schem/world | Cloudflare | none seen | Minor; partly in MAP |
| **Dream-Cubed** (Sakana AI, 2026) [V, HF dream-cubed/DreamCubedHuman] | ~359k 32³ human chunks (133 GB) from 6 maps by one pro builder + 1.67M natural chunks | `.npz` int block ids, biome labels | HF | **CC BY-NC 4.0**, creator forbids commercial use and map reconstruction | Directly usable for pretraining/eval; the closest precedent for our crop idea (single creator, explicit permission) |
| HF/Kaggle odds and ends [V] | Hack337 669 (MIT), assskelad 1,982 (none), lvnvceo ~40 GB JSON (BY-NC), MineAnyBuild 3k (BY-NC-SA), farhanwew 17 GB "CC0" (PMC-derived, licence-washed), bsig128 12k 32³ (aggregate, dups) | mixed | HF/Kaggle | uploader-claimed | Small; provenance poor; use only with care |
| 3D-Craft [V] | 2,587 houses, segmentation labels on 2,050 | npy + traces | S3 | repo MIT; data crowdsourced by Meta, no separate data licence | Already held; the one truly clean human corpus |

### F.3 The licensing reality

- Mojang's EULA [V]: players own their original builds ("your creation of a Gothic
  Cathedral ... we don't own that"); Mojang owns blocks and textures. Usage guidelines
  [V]: "Minecraft" may not be the dominant element of a name; disclaimer required;
  textures are reserved assets, screenshots are allowed. So renders are fine, shipping
  the texture atlas is not; "BlockGen: a benchmark of Minecraft builds" is fine,
  "Minecraft-Builds-1M" is not.
- A block-id grid is a near-lossless copy of the creator's build, so it inherits the
  creator's rights. PMC and GrabCraft terms explicitly forbid redistribution and bulk
  extraction [V]. No major host exposes a per-map licence except CurseForge.
- Precedents: 3D-Craft (own crowdsourcing → MIT), Dream-Cubed (one creator's permission →
  CC BY-NC), rom1504/text2mc/MineAnyBuild (scraped, redistributed anyway, no or
  uploader-claimed licence). Objaverse's model (CC-only subset + per-item licence
  manifest, ODC-By index) is the standard we should meet. NeurIPS D&B item 12 and CVPR
  ask us to state scraped sources' ToS and whether the owners' permission was sought [V].

**Recommendation: two tiers.**
1. *Pretraining tier (internal, never redistributed):* MAP worlds + MCGS + text2mc crops +
   Hermitcraft + Dream-Cubed (NC is fine for research). This is what text-to-3D work does
   with NC data; the paper describes it and releases weights, not grids.
2. *Release tier (per-item licence manifest, Objaverse-style):* 3D-Craft, CurseForge
   worlds with permissive licence AND `allowModDistribution=true`, Dream-Cubed (as a
   pointer, under its own NC terms), plus **creator outreach**: a handful of prolific map
   makers (Hermits, PMC top builders) each supply more dense human-built volume than a
   thousand random uploads, and a signed permission is the cleanest paper statement.
   Metadata (URLs, bounding boxes, tags, our curation labels) and all extraction code
   release under ODC-By/MIT regardless.

### F.4 Cropping human-made structures out of worlds

Tooling [V]: **Amulet-Core is not free** (LICENSE: "All rights reserved. A licence must be
purchased"), so it cannot sit in a redistributable pipeline. Clean choices: Rust
**fastanvil/fastnbt** (MIT/Apache, all versions 1.2→current via a `JavaChunk` enum, no
Python bindings so a thin PyO3 wrapper is needed), **MCA Selector** (Java, MIT, headless
`--mode select/export` with `InhabitedTime`, `Palette contains`, `Structures` filters) for
region-level prefiltering, and **anvil-parser2** (MIT, pure Python, 1.18+) for a quick
pilot. The 1.8-era MAP/MCGS data is numeric-id Anvil, which maps straight onto our legacy
`(id,data)` vocab with no remapping.

Three detection routes, in order of precision:
1. **Flat/void worlds (exact).** `level.dat` `generatorName: flat` → every block outside
   the flat layers is human-made. A large share of PMC "project" maps are superflat [I].
2. **Regenerate-and-diff (near exact).** Seed + generator settings live in `level.dat`.
   Regenerate the same chunks with the matching vanilla/Paper server (Chunky pregen or
   WorldEdit `//regen`) and diff; vanilla villages/outposts/mansions regenerate identically
   and cancel out. No standalone block-level generator exists (cubiomes is biomes only), so
   this means one server jar per version bucket, ~35–110 chunks/s each [I from host
   guides]. Caveats [V]: exact-version requirement (worldgen changes nearly every major),
   MC-55596 (decoration placement depends on chunk generation order, so trees/flowers
   differ spuriously), and non-player deltas (creeper craters, fire, flows, decay). Mask
   the diff by an artificial-block whitelist and morphological opening.
3. **Heuristic (text2mc-style, improved).** Artificial-block whitelist → 3D connected
   components with 1–2 voxel dilation → per-component bbox → size/density filters →
   ground-strip by heightmap (`MOTION_BLOCKING`/`WORLD_SURFACE` exist 1.13+; compute from
   columns pre-1.13) → **mask vanilla structures using the chunk's `structures.starts`
   piece bounding boxes** (`Level.Structures` pre-1.18), dilated 2–3 voxels, which text2mc
   never did → drop chunks with `InhabitedTime == 0`.
Then a **learned gate**: a small 3D classifier trained on our labelled GrabCraft/houses
(positives) vs regenerated-vanilla windows (negatives), calibrated on route-2 crops where
ground truth exists, scores every route-3 candidate. Prior art has no such detector [V];
3D-Craft's segmentation model only labels parts of already-isolated houses.

Compute [I]: 60M chunks (20k worlds × 1–5k) decode in a few core-hours with fastanvil;
connected components and export dominate but fit in a day on 24 cores; regen-diff on the
20M chunks that survive prefiltering is ~55 server-hours, ~1 day with 3 instances. Storage
is the binding constraint, not CPU: MAP is 3.6 TB against 98 GB free, so the ingest must
stream zip by zip.

### F.5 Proposed phases

- **P0 pilot (MCGS, 12 GB, no new disk):** build the route-3 extractor on fastanvil (or
  anvil-parser for speed of writing), run it over the 1% survey of all 175k worlds, and
  measure yield/precision against text2mc's `world2vec` on the same worlds and against our
  enclosed-air house gate. The `manifest.json` sign/book text is free weak captioning.
- **P1 MAP streaming ingest:** iterate the 89 zips, crop, keep crops + provenance, delete
  the world. Expect well over 10^5 structures [I]; this becomes the pretraining corpus.
- **P2 release tier:** CurseForge API crawl of licence + distribution flags, download the
  consenting permissive subset, Objaverse-style manifest; draft outreach to 5–10 creators.
- **P3 (optional) regen-diff** for the flat and version-identifiable worlds, to produce
  the gold set that calibrates the learned gate.

### F.6 Decisions and the P0 pilot (2026-10-07)

Decisions: the release tier may be small (a few thousand builds, usable as a held-out
test set); creator outreach is deferred but the contact list is kept; large windows and
arbitrary structure types are in scope, not only 32-48³ houses; disk will be freed.

P0 is built (`blockgen/data/worldcrop/`, manual in `docs/worldcrop.md`). The archive.org
monthly zips turned out to be readable **over HTTP range requests** (five requests list a
12 GB zip, one more pulls a world), so the pilot streams worlds directly from the Minecraft
Archive Project instead of using MCGS. Pilot on 400 archives of month 2014-07
(`data/minecraft/worldcrop/201407`): 381 Java worlds read (4 Bedrock, 15 unreadable or
non-world downloads), 911k chunks, **2,651 crops from 366 worlds** (2,557 saved; 94
components wider than 320 blocks recorded only), plus 30 whole `.schematic` builds found
inside "download schematic" zips. Median 3 crops per world, median largest side 26 blocks;
1,562 fit 32³, 1,909 fit 48³, 2,386 fit 128³. Routes: 1,015 flat-world crops, 1,606
terrain-world crops; 41 "flat" level.dats were rejected by the ore check. The ore-like
family filter removed 4,740 candidates, 95% of them in ten worlds (replaced-ore server maps,
a golf course of identical markers). Cost: 12.5 s per archive single-threaded, ~8 min wall
on 12 workers; the whole archive (~160k archives) is ~2 days on this machine and roughly 12 GB of
compressed crops (31 MB per 400 archives). Sample sheets: `sheet_masked.png` / `sheet_context.png` in the run dir.

Known gaps for P1: WorldPainter terrain (snow/ice mountains) passes as "built"; a learned
gate trained on our labelled houses vs. these crops is the next quality lever; 1.13+
palette worlds are decoded but untested on real data; nether/end dimensions are skipped.

### F.7 P1 ingest, provenance, and searching the corpus (2026-10-11)

**Ingest status.** The full-archive pass (`scripts/worldcrop_ingest.sh`, 12 workers)
has processed 36 months and 57.7k worlds into 476k crops (419k saved, 48k oversize) in
7.5 GB. Two failure modes from the first night are fixed: archive.org answers the
month-opening request with sporadic 500s (now retried), and a worker could be OOM-killed
on a giant world, which aborted the whole month (workers now run under a 5 GB
address-space cap and the pool is rebuilt around a dead worker). A tile-edge bug split
roughly 4% of crops into halves (ships, rail lines, domes); components touching a tile
edge are now re-tiled whole. Months ingested before the fix will be re-run.

**Provenance was silently empty.** `resources.ndjson` records no referer for download
resources, so the thread join yielded 0 titled worlds. Every month zip also ships a static
finding aid (`threads/detail/*.html`, one page per thread with title, author, original
URL, date, description and a table of mirrored files). Reading all pages of a month in one
range request and parsing the table joins 99% of archives to their thread. This matters
for three things: the release-tier licence manifest, creator outreach, and text search.

**Searching the corpus.** Two literature surveys (EmbeddingGemma and 3D retrieval) agree
on the shape of the solution. Provenance text alone is too thin for search: a thread
describes a whole map, not the crop. The recommended index is *render-and-embed*: the
existing four-view textured renders (`blockgen/labeling/render_views.py`) embedded with an
image-text model and mean-pooled per crop, with text queries through the same model's
text tower. Candidates, all Apache-2.0: SigLIP 2 (So400m or B/16, fastest, already a
bench backbone), EmbeddingGemma 2 (released 2026-10-06; one 768-d space for text, image,
video and audio, 440M parameters for text+vision, supports an interleaved
"caption + views" document), and Qwen3-VL-Embedding (strongest on MMEB, ten times
slower). Native 3D encoders (Uni3D, OpenShape) accept coloured points and would skip
rendering, but are trained on Objaverse meshes and untested on blocky scenes. A
caption-then-embed route (Qwen3-VL-4B locally, roughly 30-60 GPU-hours per million crops,
or gpt-5-mini via the existing batch labeller at about $1-2k per million) gives the best
compositional queries and free category labels, and is the natural second index on the
gated subset. Storage and lookup are not a concern: 1M x 768 fp16 is 1.5 GB and a FAISS
HNSW index answers in under a millisecond. Rendering is the bottleneck (about 4M tiles for
the whole corpus, tens of CPU/GPU hours with EGL workers). Acceptance test before scaling:
`retrieval_acc` on the 2,661 captioned `houses_32` builds for each candidate embedder.
Known risk: orthographic flat-shaded renders are out of distribution for photo-trained
encoders; the bench already shows SigLIP-B and CLIP-B retrieve a build's own caption at
top-1 among 32, which is encouraging but not a search benchmark.

**Learned gate.** See `docs/worldcrop.md`: a hand-labelled held-out set of 192 random
crops has 98 built, 65 fragments, 17 terrain and 12 unsure, so fragments (partial builds,
scattered pieces) are the larger noise class, not terrain. The gate is trained built vs
natural terrain first, with fragment handling as the next step.

## G. Pretrain, fine-tune, post-train: the 2026 recipe and what transfers (2026-10-11)

Survey behind `docs/pipeline.md`. Two sweeps: one over the repo (what exists), one over
the literature (what the strong stacks do). arXiv ids marked † were recalled rather than
fetched and need a spot check before entering `references.md`.

### G.1 What the repo has and lacks

Eleven model families, each with its own training loop and no shared trainer. Data comes
from in-memory curated caches; no trainer reads the 850k world crops. Checkpoints hold
weights only (no optimizer state, step, git sha or tokenizer hash), so there is no
mid-run resume and a rebuilt BPE vocabulary silently mismatches a checkpoint. There is no
mixture specification, no streaming or sharded tokenized data, no variable-size windows,
no periodic eval hook, and no results database beyond hand-written T-numbers and
`deploy/inference/models.json`. 3D-BPE (`tokenizers/cluster_bpe.py`) learns on 400 builds
with count-based merges; T18 found 98.8% of pieces are single-material rectangles. The
only preference code is an unused DPO script in the MinecraftACE port. The evaluation
side is the strength: a generator-blind bench, a self-validating composite, pinned
protocol, and a leaderboard that reads scorecards.

### G.2 The consensus LLM recipe

Every open report from mid-2025 to mid-2026 converges on pretrain → mid-train/anneal on
curated data → SFT → on-policy RL with verifiable rewards → a light preference pass, with
distillation to propagate into smaller models. Qwen3 (2505.09388): four stages, GRPO then
GSPO (2507.18071), small models by strong-to-weak distillation. OLMo 3 (2512.13961):
Dolma pretrain, Dolmino mid-training, then SFT → DPO → RLVR, every checkpoint released;
the clearest template for a small team. Llama 4 (Meta blog, 2025-04): prune 95% of SFT
data as easy, online RL with hard-prompt filtering, light DPO. Gemma 3/4 (2503.19786,
2607.02770): distillation plus BOND/WARM/WARP. DeepSeek V3.2 and V4 (2512.02556,
2606.19348): GRPO specialists, then on-policy distillation in place of mixed RL. Kimi K2
(2507.20534): RLVR plus rubric self-critique rewards.

Algorithms: PPO with a critic has left the open reports; GRPO (2402.03300) is the
critic-free default, corrected by Dr. GRPO (2503.20783), DAPO (2503.14476), GSPO, CISPO
(2506.13585). GRPO with group size two reduces to a DPO-like update (2510.00977), which
is useful at tiny compute. Offline preference methods (DPO 2305.18290†, SimPO
2405.14734†, KTO, ORPO, IPO) survive as the cheap final pass and as the workhorse
wherever rollouts are expensive, which is all of the 3D work. Rejection-sampling
fine-tuning is still the first move in every report. Mid-training: Llama 3 (2407.21783)
anneals on upsampled high-quality data; MiniCPM (2404.06395) mixes SFT-quality data into
the WSD decay phase; OLMo 2/3 and SmolLM3 have explicit mid-training stages.

### G.3 Post-training for non-text generators

Image: Diffusion-DPO (2311.12908†), Flow-GRPO (2505.05470†), DanceGRPO (2505.07818),
MixGRPO (2507.21802). Discrete AR image models: GRPO on VAR (2505.23331), AR-GRPO
(2508.06924), and "VAR RL Done Right" (2601.02256), whose lesson is that long discrete
sequences need intermediate rewards for credit assignment. 3D: DreamReward (2403.14613),
DSO (2503.22677: DPO or pair-free DRO against a non-differentiable physics score),
Mesh-RFT (2505.16761: masked DPO with face-level rewards), and "Are We Ready for RL in
Text-to-3D?" (CVPR 2026, 2512.10949: Hi-GRPO with hierarchical rewards). The pretrained
3D generators (TRELLIS 2412.01506, TRELLIS.2 2512.14692, Hunyuan3D 2.1 2506.15442,
Step1X-3D 2505.07747, Direct3D-S2 2505.17412) report no RL stage.

LEGO, the closest analogue: LegoGPT/BrickGPT (ICCV 2025, 2505.05469) with brick-level
rejection sampling and physics rollback; Rollback-Free (2605.06947) moves validity into
GRPO training with assembly-level rewards and reports faster inference at higher quality;
PVPO (2606.07602) names "physhack", assemblies that satisfy physics but are geometrically
wrong, the reason never to train on validity alone; BrickAnything (2605.26182) adds
structure-aware tokenization plus preference alignment; BrickBench (2610.12452, 2026-10-08)
is the agentic benchmark. Minecraft: Dream-Cubed (Sakana and NYU, 2604.22847) trains
discrete and continuous 3D diffusion directly on blocks over billions of blocks from
procedural biomes and human maps, with inpainting and a human preference study; this is
the baseline to cite and compare against. Scaffold Diffusion (2509.00062) does masked
discrete diffusion on 3D-Craft houses and reports AR failing at high sparsity. Li (FDG
2026) maps which structural constraints CFG on structural tokens can move at 32³. CAD
follows the same SFT → GRPO-with-geometric-verifier pattern (CAD-Coder 2505.19713 and its
2026 follow-ups).

### G.4 Corpus practice that transfers

Dedup is universal (exact plus MinHash, 2107.06499; FineWeb 2406.17557; DCLM 2406.11794).
"Removing Noise, not Finding Gold" (2510.00866) shows quality classifiers work by removing
junk, not selecting gold, which supports keeping the gate's rejects and using the score as
a soft tag. Quality-conditioned training beats filtering: Korbak et al. (2302.08582,
conditional training on reward tokens is Pareto-optimal), MeCo (2501.01956, prepend
metadata then cool down without it), SDXL micro-conditioning (2307.01952), Ambient
Diffusion Omni (2506.10038, use bad data for the parts of the signal it is good for).
Mixtures: RegMix (2407.01492) fits one GPU, dozens of 10M-parameter runs. Scaling
protocol: Chinchilla isoFLOP (2203.15556) with the Porian et al. corrections
(2406.19146). Repos worth borrowing from rather than adopting: nanochat (tokenizer →
pretrain → SFT → RL in one small codebase, single GPU), modded-nanogpt (Muon, QK-norm),
TRL for GRPO/DPO at this scale; torchtitan and Lingua are heavier than we need and
torchtune is unmaintained.

### G.5 Tokenizations and the comparison arms

Text serialisation: LegoGPT brick lines, LLaMA-Mesh (2411.09595†), MeshXL, 3D-GPT.
Native: MeshGPT VQ faces, BPT (2411.07025, blocked and patchified, 75% shorter),
BrickAnything attachment order, TRELLIS SLAT and O-Voxel, OctGPT (2504.09975, octree
multiscale AR), VAR/Infinity next-scale, masked discrete diffusion (MaskGIT, LLaDA
2502.09992, DVD 2605.07971 which also uses predictive entropy as a data-quality score).
Four arms for the paper, one tokenizer story each on shared tiers and the shared bench:
sparse raster AR with and without 3D-BPE, masked discrete diffusion on the same backbone,
hierarchical AR (next-scale or octree) where hierarchical rewards attach, and the
LLM-program baseline. The recommended stack and milestones are in `docs/pipeline.md`.

### G.6 What one 16 GB GPU can do

Full pretraining of 10M to 300M models in bf16; SFT; rejection-sampling FT; DPO/SimPO
with precomputed reference logits; GRPO with groups of 4 to 8 at up to 100M because the
verifier is Python and BlockScore's fast tier is seconds; best-of-N; checkpoint merging;
on-policy distillation from a 300M teacher. Not feasible: PPO with a separate critic and
reward model above 100M, async multi-model pipelines, RL on 128³ sequences without
hierarchical tokens.
