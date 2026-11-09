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
- Agentic LEGO, unpublished (2026-11-05 addendum): **ldraw-nova** (github.com/anteloc/ldraw-nova,
  AGPL-3.0, 520 stars, no paper, no metrics) is an LLM-agent toolchain, not a learned model:
  brief, retrieve (curated knowledge base + example search), build (plan JSON -> Python
  generator -> LDraw .mpd), inspect (renders, collision/gap/contact checks), repair, deliver.
  A third agentic twin of Track E after BrickWright and BrickBench. The two pieces Track E
  lacked, symbolic inspection as a repair signal and retrieved lessons, are now the `inspect`
  method and the `lessons` context (docs/agentic-suite.md §5.7, notes §81). Running it as an
  external LEGO generator on the BrickBench task is deferred past the CVPR deadline.

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

## H. Next-Concept Modeling as a hierarchical arm: what it is, what it would take (2026-10-23)

Prompted by Kumbong, Khare, Mlauzi, Woen, Ding, Buchanan, Arora, Mirhoseini, Ré,
"Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation", arXiv 2610.14162
(2026-10-09), code https://github.com/HazyResearch/ncm (MIT, evaluation only). The
question asked: can a model of this shape learn the hierarchy of a build, adding detail
at each level? Short answer: yes in principle, it is next-scale prediction (VAR) with a
learned residual hierarchy, and 3D is the easier domain for it. But its measured gains
are speed over diffusion, not quality over AR, so the case for it here rests on what fast
sampling and a learned coarse level unlock (post-training, hierarchical rewards,
level-wise control), not on beating arm A at BlockScore. Details below, then the plan.

### H.1 What NCM is, precisely

Two stages, both small.

**Tokenizer: a multiscale residual VQ-VAE.** Encoder E maps a length-L token sequence to
latents z in R^{L x d} (for most datasets the encoder is just an embedding lookup; the
LM1B one has 6 blocks). K levels with lengths l_1 < ... < l_K = L. Level k pools the
current residual down to l_k positions (F.interpolate area, i.e. average pooling; level 1
uses a learned strided conv initialised as average pooling), snaps each position to the
nearest entry of a per-level codebook, lifts the codes back to length L (linear
interpolation plus a small per-level residual conv "Phi"), adds the result to the running
reconstruction ẑ, and subtracts it from the residual. Decoder is one transformer block
over ẑ^(K) producing per-position token logits. Loss is reconstruction CE plus commitment
(beta 0.25) plus a codebook term; codebooks move by EMA (0.95 to 0.99) with dead-code
restart. The ingredient that makes coarse levels meaningful is nested dropout: with
probability 0.5 the decoder is fed ẑ truncated at a random level k, so early levels must
carry the gist on their own. Reported exact-sequence reconstruction 98.8 to 100%.
Settings: K = 12 for L = 128 (linear schedule), K = 32 for L = 1024, codebooks 4096 to
8192, d = 384; Sudoku uses K = 12 over L = 89 with 16-entry codebooks and the schedule
1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89.

**Generator: a transformer that predicts one level at a time, all positions in parallel.**
p(r_{1:K}) = prod_k prod_i p(r_k^i | ẑ^(k-1)). The input at level k is ẑ^(k-1) resized to
l_k positions and projected, plus a level embedding; RoPE restarts at 0 within each
level. Attention is block-diagonal: a level attends only to itself and the prompt prefix,
never to earlier levels' tokens, which enter only through ẑ. This was their best variant
(Gen PPL 15.4 vs 19.6 for block-causal on TinyStories) and it is what makes sampling K
forward passes. Training is one pass over all levels with teacher forcing, with the
conditioning codes corrupted: a fraction of input codes (mean 0.5, decreasing linearly
from coarse to fine) is replaced by uniform random codes from the same codebook while the
targets stay clean. Loss weight per level l_k^(-alpha); alpha in [0, 1] made no
difference, 1.5 hurt. Conditional tasks use a prefix-LM mask and classifier-free guidance
whose weight ramps from 0 at the coarsest level to the set value at the finest.
Test-time scaling is MBR: sample N, keep the one most similar to the rest.

**Numbers worth holding onto.** LM1B Gen PPL 93 at 12 NFE vs MDLM 115 and DUO 97 at 128
NFE. OpenWebText 83 at 32 NFE vs DUO 78 and MDLM 105 at 1024, while a plain AR model
samples at 35. Sudoku: beats both AR and diffusion baselines at every clue count with
fewer function evaluations (the one place NCM wins on quality, and it is a global
constraint-satisfaction task). Generators are about 100M parameters; tokenizers 15 to
144M; everything trained on one node, timed on one H100. Appendix H probes the hierarchy:
fixing the three coarsest of 12 levels and resampling the rest keeps a story's theme and
changes names, objects and wording. Stated limitations: fixed K and schedule regardless
of input, tokenizer trained separately from the generator, artifacts at level boundaries.

**What is and is not released.** The repo has the quantizer (`models/quant.py`, with
EMA, k-means init, dead-code restart, nested dropout, learned level-1 sampling), the
VQ-VAE wrapper, the generator (`models/next_concept.py`, with masks, KV cache, CFG,
sampling), checkpoints and configs. There is no training loop, no loss, and no corruption
code. Algorithm 1 is a page, so this is a day of writing, but it is ours to write.

### H.2 Lineage: this is VAR for text, and 3D already has VAR

NCM's contribution relative to VAR (Tian et al., NeurIPS 2024, 2404.02905) is making
next-scale prediction work where there is no natural pooling: learned level-1 sampling,
nested dropout so coarse codes carry meaning, corruption for robustness, block-diagonal
attention. Voxels do have natural pooling, and the 3D field already has the
corresponding models: SAR3D (2411.16856, multiscale 3D VQ-VAE on triplanes, next-scale
AR, 0.8 s per object), G3PT (cross-scale VQ, residual across scales), 3D-WAG (2411.19037,
wavelet token maps, BMVC 2025), OctGPT (2504.09975, already in references.md §4), MARS
(2502.11390, level-of-detail VQ for meshes). None of them is on voxel structures of
discrete block classes, and none measures the hierarchy with a generator-blind bench, but
"NCM on Minecraft" is not itself a contribution. What NCM adds that those lack and that
we should keep: the residual VQ with nested dropout (so coarse levels decode to something
on their own, rather than being a fixed pooling), code corruption, and the
within-level-independence recipe with a measured attention ablation.

This arm is already on the plan: research.md §G.5 lists "hierarchical AR (next-scale or
octree) where hierarchical rewards attach" as the fourth tokenization arm, and
docs/pipeline.md §4 arm C (dense grid, 2³ patches at 32³, 4,096 tokens) is the data path
it shares. T53 was a hand-built two-stage coarse-to-fine (cell assignment at f=4, then
connections) and tied retrieval; its lesson, that the frame must be fine enough to pin
geometry, becomes here "the hierarchy must reach at least 16³".

### H.3 Mapping onto builds: the assumptions to make explicit

1. **Dense canvas, not the sparse raster.** NCM's levels are lengths of one sequence
   with pooling between neighbours. In arm A's `(x, y, z, id)` list, neighbouring tokens
   are fields of different blocks and position i has no spatial meaning, so pooling it is
   not the user's "add detail at the next level". The natural host is the fixed 32³
   canvas of `houses_32` (arm C): pool in 3D with trilinear/area interpolation over
   (D, H, W), levels are resolutions. Running the released 1D recipe on our token shards
   as text is still worth one cheap run as a control (text has no spatial order either
   and it worked), but the prediction is that its coarse codes are "which y-bands are
   dense", not massing.
2. **Latent pooling is not T17's decimator.** T17 showed strided decimation of discrete
   ids destroys 86.5% of blocks because walls are one voxel thick. NCM pools continuous
   embeddings and keeps the residual, and the decoder is trained for exact
   reconstruction, so a one-voxel wall becomes a half-strength latent at 16³ plus a
   correction at 32³. Whether the coarse levels alone decode to a connected massing is an
   empirical question and the first thing to measure (H.4 step 1).
3. **Air dominance.** A house is a median 745 blocks in 32,768 cells. The codebook will
   spend most of its capacity on air and the recon CE will look excellent while occupied
   voxels are wrong. Arm C needed air down-weighting to produce anything (notes §8). The
   tokenizer must be judged on occupied-voxel exactness and build-exact rate, and the
   recon loss probably needs the same class weighting.
4. **Within-level independence is our failure mode by construction.** Every position at
   a level is sampled independently given the coarser reconstruction. "Right gestalt,
   incoherent up close" (§B) is exactly the error this invites at the finest level: a
   wall whose patches disagree. Mitigations, in the order to try: the paper's corruption
   training (mean 0.5), the residual structure itself (the finest level only adds a small
   correction if the tokenizer put the wall at 16³), best-of-N selected by connectivity
   or the fast tier, and as an ablation two to four MaskGIT iterations inside the finest
   level, reusing arm C's sampler. The pure recipe goes first so the result is about NCM.
5. **Finest level is 2³ patches, not voxels.** Block-diagonal attention costs sum of
   l_k², so 32³ = 32,768 positions at the finest level is a billion-entry attention per
   head; 16³ patches of 2³ is 4,096, the arm-C budget, and the tokenizer's conv decoder
   unpatchifies to 32³ with per-voxel class logits. Levels 1³, 2³, 4³, 8³, 16³ give 4,681
   tokens in total and K = 5 forward passes per sample. VAR uses a denser schedule
   (1, 2, 3, 4, 5, 6, 8, 10, 13, 16); the level-schedule ablation is cheap and the paper
   says it matters little in 1D. Anisotropic levels (coarser in y, houses are wide) are a
   second-round option.
6. **The tokenizer is a trained artifact with a hash.** pipeline.md's rule that a
   tokenizer is a versioned artifact whose hash a checkpoint refuses to mismatch applies
   with more force: the hash must cover the VQ-VAE weights, and every generator run names
   its tokenizer. The tokenizer should train on the anneal tier (world-crop windows at
   32³), not on 1,863 houses, both because a VQ-VAE at that size overfits and because that
   is what the pretrain tier is for; the generator then follows the pretrain -> SFT
   schedule like arm A.
7. **Conditioning comes free, and lands where geometry is decided.** Prefix-LM over our
   tags (src, q, size, win) and CLIP/DINO tokens, CFG ramped by level. T15's finding was
   that palette transfers and geometry does not; in NCM the coarse levels are where
   geometry lives and CFG is weakest there by design (ramp from 0). Testing the ramp
   direction is a one-flag ablation with a real hypothesis behind it.
8. **Quality expectation, stated before running.** On text NCM sits in the
   diffusion class (83 vs AR 35 Gen PPL). On our data AR beats diffusion on houses (T6,
   0.568 vs 0.369 NN-IoU). The prior is therefore that NCM loses to arm A on material
   coherence and fine geometry at matched parameters, and the Sudoku result is the one
   reason to expect gains on global constraints: connectivity, enclosure, traversability
   (T63), the pillars where arm A and the LLM arms are weakest. Pre-register that split.
9. **Speed is the lever that matters here, even though it is not our inference
   bottleneck.** Five forward passes per build instead of up to 8,200 makes the
   expensive stages of docs/pipeline.md §7 affordable: best-of-N and MBR at sampling,
   20 to 50k generations for DPO data, GRPO groups with a Python verifier (§G.6 said RL on
   dense sequences needs hierarchical tokens). Rewards can also attach per level: a
   massing reward at 4³ or 8³ before any material is placed. That is the story the arm
   would carry in the paper, not the architecture.

### H.4 What to do, in order

1. **Tokenizer-only study (one to two days, no generator).** A 3D multiscale residual
   VQ-VAE at 32³ with the schedule in H.3.5, conv encoder/decoder, EMA codebooks with
   restart, nested dropout 0.5, air-weighted CE. Train on anneal windows, hold out
   `houses_32` test. Report: occupied-voxel exactness, build-exact rate, codebook
   utilisation per level, and a `samples.png` whose columns are the truncated
   reconstructions ẑ^(1) ... ẑ^(K) of real houses. That figure answers the user's
   question directly: if level 3 is a connected massing and level 5 adds windows and
   trims, the hierarchy exists before anything is generated. Gate: at least 99% occupied
   exactness and 90% build-exact on the holdout, or the generator is not worth training.
   Run the connectivity and traversal probes on the coarse-only decodes too; a hierarchy
   whose coarse levels are porous cannot host a massing reward.
2. **Generator, unconditional, houses_32 (two to three days).** Block-diagonal transformer
   at the 5M and 20M presets, corruption mean 0.5, alpha 0, pretrain on anneal then SFT
   as arm A's M2b protocol (select = best, no dropout change at init_from). Bench at
   n >= 256 on houses32-v1 against arm A's T59 numbers and arm C. Add the raw and the
   best-of-8 arms separately, compute-matched, since MBR is part of the method's claim.
3. **Ablations that separate NCM from VAR.** Fixed average-pool hierarchy without VQ
   residual learning (VAR-style) vs learned residual; nested dropout off; corruption off;
   block-causal attention; two-step MaskGIT refinement at the finest level; level
   schedule. Each is a flag, each a 5M run.
4. **Conditioning and control.** Prefix tags plus CLIP tokens with CFG ramp up and
   ramp down; the Appendix H probe on builds (fix levels 1 to 2, resample 3 to 5: same
   house, different details; swap coarse codes between two houses). Rendered as a grid,
   this is a figure the paper does not have today.
5. **Only then post-training.** Best-of-N by fast tier, DPO on 2AFC pairs from the lab,
   per-level massing rewards. This is where the arm pays for itself or does not.

### H.5 Implementation questions to settle before writing code

- Own implementation in `blockgen/models` against the shared trainer, borrowing the EMA,
  restart, k-means init and nested-dropout logic from `quant.py` (MIT), vs vendoring the
  repo under `libs/` and adding a training loop. Recommendation: own implementation. The
  quantizer is 1D (B, L, C) with 1D interpolation and would need a 3D rewrite anyway, the
  trainer's checkpoint, hash, registry and bench hooks are what make a result comparable,
  and `trainer.build_model` currently hard-codes `VoxelTransformerAR2` and token shards,
  so a dense-grid data path has to be added (arm C needs it too; pipeline §10's open
  question about arm C sharing the backbone is answered by this arm sharing it).
- Patch size and finest level: 2³ patches to 16³ (recommended) vs voxel-level with a
  windowed finest level vs 4³ patches to 8³ (too coarse by T53).
- Codebook size per level: 4096 as in text, or 512 to 1024 given air dominance and the
  small number of distinct house-scale configurations; whether to use FSQ/BSQ as OctGPT
  does to avoid collapse instead of EMA plus restart.
- Nested dropout cutoff distribution (uniform over levels in the code; LM1B used 0.7
  full / 0.3 truncated) and whether coarse-only decodes should be pushed toward
  connectivity with an auxiliary loss (no, not in the first round).
- Corruption schedule and alpha: transfer the paper's values, ablate later.
- Canvas: 32³ only for the first round; 48³ and world-crop windows need their own
  tokenizer or a window-tiled finest level, and are a second tokenizer version.
- Augmentation: yaw rotations and mirrors are free on a conv tokenizer and should match
  what arm A sees.
- Selection and compute matching for MBR/best-of-N arms: which selector (fast tier,
  connectivity, geom descriptor similarity as the paper's "most similar to the rest"), and
  how the bench card records the N.
- What "n" means for the tokenizer hash: weights plus config plus the anneal tier hash.

### H.6 Reservations, plainly

The paper's win is 10 to 30x faster than diffusion at diffusion quality, and quality
well below AR on text. The hierarchy is real (Appendix H) and the Sudoku result is the
one hint that it helps global constraints, which is our weakest pillar. The 3D version is
prior art several times over, so the architecture cannot be the contribution; the
contribution would be hierarchy-attached rewards, level-wise control, and bench evidence
on a dense structured domain, which is what §G.5 already reserved the fourth arm for. No
training code is released. The tokenizer-only study in H.4.1 costs under two days and
settles whether the coarse levels of a learned residual hierarchy are massings or noise,
which is the question actually asked. Start there.

### H.7 The 2D port: facades as the first host (2026-10-30)

The facade track (notes §54 to §60, T65 to T68) makes a face an image already: a
(U, V) grid of cells, each `empty` or `(depth, block)`, up to 32 x 32, six per build,
lifted into the canvas the bench scores. NCM for images is VAR with NCM's deltas
(residual per-level codebooks, nested dropout, block-diagonal attention, code
corruption, level-ramped CFG), and a face is a categorical image with three heads
(mask, depth, block) instead of RGB. So the port is: 2D area pooling over cell
embeddings, levels 1, 2, 4, 8, 16, 32 (K = 6, finest at the cell, 1,024 positions,
no patching), a conv or one-block decoder with three categorical heads, and the
generator predicting all six faces of a build inside each level block so the faces
see each other at every scale. The box stays a forced prefix (size tags, as the joint
sampler does today) and cells outside it are masked, so the six faces always describe
one real box. Decoding is codes -> decoder -> argmax per head -> `Facade` -> `lift`
-> bench, all of which exists; the only new code is the model and a dense face loader
(a third `kind`). That is why this is the cheaper host: the 3D arm needs the dense
32^3 data path and the air problem (H.3.3); the face's empty cells are outside the
silhouette, not 97 percent of the canvas.

What the hierarchy would mean on a facade, checkable by eye from truncated decodes:
1 x 1 and 2 x 2 the box and the dominant material, 4 x 4 storeys and the roof line,
8 x 8 the window rhythm and bands, 16 and 32 the windows, doors, trim and the depth
relief. Those sheets are the whole first experiment, and they are images, so no
renderer is in the loop. *Measured 2026-11-02 (T75, notes §63) with fixed pooling:*
on a 32-cell face f = 16 is the box and material, f = 8 the bands, f = 4 storeys,
roof line and footprint shape, f = 2 openings and trim, f = 1 the relief. Windows
are one or two cells wide and survive only to f = 2, so the periodic structure is a
fine-level feature, not an 8 x 8 one; pooled levels are smoother and more connected
than real faces, so a coarse level does not host the models' fragmentation. Rule for
the cascade: `any` (dilating, recall 1), `majority` as the control. *Cascade trained
2026-11-03 (T88, notes §74):* the depth residual is learned (completion from the real
f = 4 beats the copy on depth exactness, roof lines come back sharp) and the block
residual is not (block agreement below the copy, no window ever drawn); the coarse
levels memorise the 7.5k train boxes within a thousand steps; samples from the box are
smooth one-material blobs that match the face statistics better than any AR arm, so
those statistics are gameable by smoothness. The learned residual VQ is deferred;
the next move is the cascade's coarse level as a prefix to the per-face AR, which draws
the trim with its neighbours in context (notes §74).

Why it targets the measured failures rather than only being cheaper:
- **Relief noise (§60).** The 20M D4 model at temperature 1 draws speckled depth (jump
  rate 0.29 vs real 0.10) and fragments into 6.8 pieces at 26-connectivity; at 0.5 the
  faces are solid but a fifth of the draws are empty, the ground-plate mode. In a
  coarse-to-fine model "what and where" (coarse levels, sampled at full temperature for
  diversity) and "texture" (fine levels, which can run cold) are separate passes. The
  testable claim: a per-level temperature schedule gets temperature 0.5's solidity
  without its empty builds. AR has no such knob.
- **Faces that do not meet (T68).** `geom_kid` for every facade arm is held by
  `n_components` (60 to 75 against the ceiling's 13.5) because faces disagree at shared
  edges. With all six faces in one level block, corner depths and materials are settled
  at 2 x 2 and 4 x 4 before any cell is drawn, and attention within the block crosses
  faces at every scale. Measure with the 26-connected shell components and a direct
  shared-edge agreement rate.
- **Windows and doors.** Patchwork beats every learned model on geometry because real
  faces have real windows. Periodic structure is what the 8 x 8 level codes; the Sudoku
  result is the one hint that the hierarchy is good at global regularities. Enterability
  (0.35 at best vs ceiling 0.73) needs a door at the floor, which is a coarse-level
  decision.
- **Interiors.** Unchanged; the ceiling stays 493. Only the 3D hierarchy reaches inside,
  and the 2D and 3D arms are the same code with 2D or 3D interpolation, which is the
  bridge: the facade arm is the correctness check for the volume arm.

Simplest first steps, as in H.4 but in 2D:
1. One face type, the roof plan or the front, as the user suggested in §60. Pool real
   faces to 16, 8, 4, 2 with majority rules on (mask, depth, block), decode naively, and
   look: does 8 x 8 read as a facade? Half a day, no model.
2. Fixed-pooling cascade on that face type: a small 2D U-Net or 1,024-token transformer
   that predicts level k + 1 from level k with corrupted input, trained on all levels,
   sampled in five passes. Score with `facade_sheets.py` statistics (jump rate, 2D
   components, holes) against the per-face 5M AR at matched budget. D4 gives eight
   faces per face for free.
3. Learned residual VQ on the same face type, then six faces jointly, with the lifted
   builds on the T68 card at n = 399 against the 20M D4 joint model (1082), patchwork
   (990) and the ceiling (493). Report raw and best-of-N separately.
4. Then H.4 in 3D with the same code.

Data: 2,114 houses x 6 faces x 8 symmetries is about 100k face images at 32 x 32 or
smaller, enough for a small VQ-VAE; the anneal tier's crops add faces by the million if
the tokenizer wants them. Compute: six faces x 1,365 positions is 8,190 tokens per build
with block-diagonal attention, so a 5M to 20M generator trains in minutes to an hour.
Code to start from: VAR (FoundationVision/VAR, MIT) releases the multi-scale VQ-VAE and
the training loop that NCM does not; NCM's deltas are small on top of it.

## I. CVPR 2027: which papers, and what makes them worthy (2026-10-28)

Four independent reads on the same day: a readiness audit of the unsent ICLR draft and
its planning docs, a cold review of the draft by a reader who was not allowed to see
notes.md or results.md, an inventory of every other track for second-paper candidates,
and a web check of the CVPR 2027 call and the 2025 to 2026 literature. This section is
their synthesis. Each report's full text is in the session transcript; the claims below
are the ones at least two reads agreed on or that come with a source.

### I.1 The calendar

CVPR 2027 (Seattle, June 20 to 25; call posted 2026-09-03,
https://cvpr.thecvf.com/Conferences/2027/CallForPapers and /AuthorGuidelines):

| what | when (AoE) |
|---|---|
| paper enrollment, author list frozen, OpenReview profiles required | Nov 10, 2026 |
| full paper | Nov 16, 2026 |
| supplementary | Nov 23, 2026 |
| reviews out / rebuttal | Jan 25 to Feb 1, 2027 |
| decisions | Feb 25, 2027 |

Eight pages including figures and tables; references and the new mandatory section
titled "LLM Usage Statement" (after the main text, before the references; disclose any
generative AI use in research or writing, or state none; missing it is desk reject) do
not count. Double blind, arXiv allowed, all authors join the reviewer pool. There is no
datasets-and-benchmarks track; "data synthesis, datasets and benchmarks" is a topic.
Anything online after 2026-09-15 is contemporaneous work. 3DV 2027 has already closed
(Aug 28). The first realistic fallback after a February reject is ICCV 2027 (Hong Kong,
October; deadline estimated early March 2027, unconfirmed), then NeurIPS 2027 Datasets
and Benchmarks (expected mid May 2027, unconfirmed). SIGGRAPH and ICML 2027 fall before
the CVPR decision, so they are alternatives, not fallbacks.

Nineteen days from today to the full-paper deadline, on one shared 16 GB GPU.

### I.2 Verdict

One paper is realistic for the CVPR main conference: the evaluation paper. Nothing else
in the repo reaches a main-conference bar in nineteen days, and the inventory read said
so plainly. The strongest second candidate, world-crop pretraining, has its decisive
experiment already run and unreported (the M2b three-seed result, never written as a
results entry) and is the right workshop paper to draft in parallel. Everything else is
a scored arm or a supplementary table for the evaluation paper, or not yet a paper.

### I.3 The evaluation paper: what it is and what holds it at borderline

The draft on disk is not a "benchmark" paper; its headline is the blindness result.
Render-and-embed metrics (multi-view DINOv2 KID and six other render-space statistics)
cannot see whether a structure has an inside: Fill-Interior adds a median 33 percent of
blocks and moves MV-KID a fraction of a noise floor while the geometry KID moves by
about two hundred, cutaway views recover the signal with the same encoders, and a
mass-matched Grow-Shell control shows the render metric does see mass that lands
outside. BlockScore, the gates, the ladder, the human 2AFC and the leaderboard are the
supporting apparatus. The cold reviewer rated it 3 of 5 at confidence 4 and said the
finding is "real, cleanly demonstrated and worth publishing".

Both the cold review and the audit put the same two gaps first:

1. **No standard volumetric 3D metric on the ladder.** PointFlow's MMD-CD, COV and 1-NNA
   on point clouds sampled from all occupied voxels, voxel IoU, and KID on a pretrained
   3D encoder would all see Fill-Interior. The paper answers this only in prose. Without
   that table the second contribution reads as "a handcrafted descriptor whose
   thickness bin was designed to see the probe beats an image metric". This is the first
   paragraph of every 3D reviewer's report.
2. **No external published generator is scored.** Four in-house arms, two of which the
   paper itself calls weak baselines. Candidates with released samples or weights:
   Dream-Cubed (NeurIPS 2026, the closest competitor, render FID plus a human study on
   Minecraft worlds), Scaffold Diffusion (3D-Craft houses, qualitative eval only), the
   3D-Craft VoxelCNN, BrickGPT on the LEGO side.

Then, in order of damage:

3. The human agreement the protocol pre-registers (Bradley-Terry strengths, Spearman rho
   against each metric over all pairs) is never reported; what is reported is three pairs
   and one 93-of-93 reversal, which says both metrics disagree with people. Twenty raters
   is thin. Appendix H still describes a schedule that is not the one run.
4. The headline separation is quoted at 0.14, 0.4, 0.38, 1.0, 1.1, 2.0 and 0.26 in
   different places, and the Section 6.2 prose contradicts Table 12 by a factor of ten.
   If Table 12 is right the probe is weakly but significantly detected under the
   deployed translucent renderer, so "invisible" holds exactly only for an opaque
   renderer. One run of record, one n, one number, and the claim softened to "two orders
   of magnitude less sensitive, exactly invisible under opaque rendering".
5. The noise-floor unit scales with n and moved from 202 to 287 between two identical
   runs; it is a test statistic quoted as an effect size to three figures. Report an
   n-free effect size beside it.
6. The leaderboard sits at n=128 against the paper's own minimum of 256, and still
   carries the legacy subsample p-values rather than the permutation test the text
   describes.
7. BlockScore is five pillars in the text, three in practice (Appendix M). Define all
   five in the method or present it as the max over the ones that decide.
8. Figure numbers disagree with Table 2 (Figs 12 and 15), Fig 2 says eight rungs and
   draws six, Table 11 is captioned in-domain but is the LEGO ladder, three TODOs and a
   visible margin comment survive, main text is ten single-column pages against eight
   two-column, no mainstream 3D generation related work (TRELLIS, Hunyuan3D, CLAY,
   AssetGen, OccNet, SparseFlex), and the LEGO licence reads "unresolved" in one place
   and "MIT" in another.

The release report's open items (Grow-Shell has no probe in code, the bench_hollow run
and the seed sweep have no released source, requirements-full.txt broken) are the
reproducibility half of the same list.

### I.4 The nineteen-day plan for the evaluation paper

Experiments, all of them on existing tooling and none needing more than GPU minutes
per item except the first:

- Leaderboard of record at n=256 with the 80-dimensional descriptor, the traversal
  pillar, the permutation test and the procedural baselines. The LLM arms need new
  samples. About one day. Closes items 4, 6 and the T64 provenance in one run.
- Volumetric baseline row: MMD-CD, COV, 1-NNA on one point per occupied voxel, voxel IoU
  nearest-neighbour, and KID on one pretrained 3D encoder, run on the same rungs and
  through the same gates. One day. Turns item 1 into a table; if a standard metric
  passes the gates and sees the probe, say so, the paper is then about the audit and
  the ladder rather than about the descriptor, which is the stronger paper anyway.
- One external generator's released samples through the suite, even unranked. One to
  two days if samples can be obtained; start the request on day one. Item 2.
- Human study: fit the pre-registered Bradley-Terry and rho on the data that exists,
  with bootstrap intervals over items and raters, and a TOST on the exterior pair. Two
  hours. More raters only if time remains.
- Implement the Grow-Shell probe in probes.py and regenerate its table from code; add
  the twelve-seed interval for the headline. Half a day.

Writing, overlapping:

- Port to the CVPR template at eight pages, appendix to the supplement. Two days.
- Visual-hull reframe and a mainstream 3D generation related-work paragraph. Half a day,
  the best value per hour in the list.
- Fix every stale number and caption in item 8; expand method text to match code (KID
  kernel and scaling, pooling, gate and DQ thresholds); state the n-dependence of the
  unit in every caption and drop three-figure precision. One day.
- Remove the self-doubt and process sentences the cold review listed (ten quotes, mostly
  in the appendix: "and it did not", "was mismeasured twice", "rejected 3.4x too often",
  "They are weak baselines").
- The LLM Usage Statement, the data and terms-of-service statement, the human-subjects
  and compensation statement, the author list and reviewer commitments before Nov 10.

What the new work contributes to it:

- Traversability (T63, T64) goes in as a third falsifiable probe family and as the
  honest answer to "the geometric tier is also easily fooled": the descriptor ranked
  one-block sticks above room-building houses until room scalars were added, and the
  paper should say it audited its own geometric tier and fixed it. Not a new pillar
  claim; the pillar never decides BlockScore.
- The facade oracle (T65) is one row and one sentence: a facade-perfect generator keeps
  79 percent of the blocks and scores 0.001 on the render pillar. It is the blindness
  result stated from the generator's side.
- The facade generators (T66), the agentic methods (T51) and the M2b pretraining result
  stay out, as scored arms at most. SCOPE.md's argument against folding models in holds.

### I.5 Second paper: world-crop pretraining, as a workshop submission

Thesis: a terrain-gated, deduplicated corpus of about 650k structures cropped from
archived Minecraft worlds transfers to curated house generation at matched compute.
Evidence already on disk: the world-crop pipeline (T51), the learned terrain gate (T54),
the 926k manifest with 30 percent duplicates (T55), the jar templates (T56), the shared
trainer (T57, T58), the dropout and early-stopping diagnosis (T59), and the unwritten
M2b result: 5M parameters, three seeds, early stopped, n=256, holdout NLL 0.479, 0.420,
0.350 for pretrain plus SFT against 0.565, 0.555, 0.509 from scratch, BlockScore 375,
323, 324 against 502, 445, 334, pretraining winning every seed on both measures.

Missing for a credible paper: a 20M run (three seeds, about two GPU days, queued behind
the facade job), a scratch arm regularised on holdout rather than left to overfit 1,863
houses for 156 epochs, one pretrain-all versus pretrain-clean ablation, re-scoring the
M2b arms under the current descriptor so they are board comparable, and a scaling
figure. The corpus cannot be released (MAP has no licence, F.3), so the release is code,
manifest schema and weights, with the F.3 licensing statement in the paper. Three
figures: the pipeline and yield table, NLL and BlockScore against tokens for scratch
versus pretrain plus SFT at 5M and 20M, and a sample sheet. First step costs nothing:
write T60 from outputs/registry.jsonl and the six M2b scorecards.

### I.6 The rest, honestly

- **Agentic dream-and-build (Track E).** Dream-caption beats dream and compute-matched
  classic on the render KID at n=133 (T51) and all three tie on geometry; classic beats
  both dream arms on traversability (T63). "Dreaming is a null on geometry; captions
  move the render metric" is a workshop result. Needs a human 2AFC (the phone queue
  exists, never run), stage ablations and a second LLM. One to two weeks, 50 to 150
  dollars of API, no GPU. The fallback if the GPU cannot be shared. The literature read
  found no 2025 to 2026 paper where an LLM writes WorldEdit-style build programs, so
  the territory is open; BrickWright (2610.31651) and BrickBench (2610.12452) are the
  LEGO twins and published the agentic-beats-trained framing first.
- **Facade generation.** The six-face (depth, block) representation with a lift appears
  to be new for block structures (MVDD and 2.5D-latent papers are the general
  precedent; UniRecGen argues the opposite side and should be cited). But one seed,
  seven minutes of training, and six real faces from six different houses still beat
  every learned model on every pillar (T66). Workshop only if the 20M model beats
  patchwork; the interior stage is a separate month.
- **Traversability alone** is a four-page note; it belongs in the evaluation paper.
  Nearest neighbours: SceneTeract (2603.29798, reachability and clearance for indoor
  scenes), LMBuild (2610.04292, functional affordance for part assemblies), SceneEval
  (WACV 2026 navigability). None measures whether a player can enter a generated
  building.
- **Graph and attention work** (T49 to T53): ties and negatives plus one nice one-seed
  analysis. Not yet. **LEGO reproductions and pick-and-place** (T13, T24, T61, T62): no
  new result. **BlockLab and the live mod**: demo track material.

### I.7 Novelty, as of today

Unoccupied in the literature found: a block-structure benchmark pairing geometry-space
and render-space KID; a traversability pillar on generated buildings; a ladder of
degraded real builds as the calibration instrument (nearest analogue is the face-drop
regime for meshes in 2606.18451); the six-face facade representation. Cite prominently:
Dream-Cubed (2604.22847, NeurIPS 2026), MineAnyBuild (NeurIPS 2025 D&B), MineCEraft
(2608.28884), Scaffold Diffusion (2509.00062), Li's regime map (FDG 2026, same 32 cubed
canvas and architecture family), SceneTeract, LMBuild, BrickBench, BrickNet (CVPR 2026).
Dream-Cubed is the one to engage directly: it validated a render FID with a human study
on the same game, and the blindness result is exactly the thing that validation would
not catch.
