# References & related work (for the paper) — maintained 2026-07-03

One entry per work: citation info, link, **relevance to us** (cite-where note).
Sections mirror the likely paper structure. Sourced from the 2026-07 deep-research
sweep (`research.md`) + earlier notes. ★ = must-cite / closest relatives.

## 1. Brick/LEGO generation (the closest published line)

- ★ **LegoGPT / BrickGPT** — Pun*, Deng*, Liu* et al., "Generating Physically Stable and
  Buildable LEGO Designs from Text," ICCV 2025 (reported Marr Prize).
  https://arxiv.org/abs/2505.05469 · code https://github.com/AvaLovelace1/BrickGPT ·
  data https://huggingface.co/datasets/AvaLovelace/StableText2Lego
  *Relevance:* the anchor related work. Fine-tunes LLaMA-3.2-1B on plain-text brick lines
  `hxw (x,y,z)` (20³ grid, 8 brick types); inference-time validity check + PhysX stability
  + brick rollback (24%→98.8% stable). **Our planned track D uses its exact base model for
  comparability.** We differ: Minecraft voxels, 3D-BPE cluster tokens, three-way model
  comparison, small curated data + honest memorization eval. Cite in: intro, related work,
  method (validity gate lineage), baselines. TODO: pull exact training compute from PDF.
- ★ **BrickAnything** — arXiv 2605.26182 (2026). Structure-aware **tree tokenization**
  (attachment relations, not spatial order) + geometry-conditioned AR + DPO alignment +
  constrained decoding/rollback. *Relevance:* nearest relative of our 3D-BPE cluster
  tokens — theirs is attachment-tree, ours is merged rigid pieces; must discuss contrast.
- ★ **BrickNet** — Kulits & Schmid, CVPR 2026, arXiv 2604.22984.
  https://kulits.github.io/BrickNet/ — LM over **graph-backed connectivity programs**;
  100k+ LDraw objects released. *Relevance:* precedent + data source for our Track C
  (block+port graph) and the LEGO-transfer north star.
- **Rollback-Free Stable Brick Generation** — arXiv 2605.06947 (2026). RL post-training
  with assembly-level rewards (collision/connectivity/interlocking/shape) removes
  test-time rollback. *Relevance:* validity-as-training-signal (vs our inference gate);
  motivates our GRPO plan.
- **PVPO** — "Sample-Efficient Post-Training for LEGO Spatial-Physics Reasoning,"
  arXiv 2606.07602 (2026). RL with physical-feasibility + voxel-geometry rewards; names
  the **"PhysHack"** failure (valid but degenerate). *Relevance:* reward-design caveat for
  our RL stage — pair programmatic + visual rewards.
- **StableLego** — arXiv 2402.10711 (2024). Force-based per-brick stability analysis
  (used by LegoGPT). *Relevance:* the stability metric lineage; our voxel analogue is the
  floating-block/support rate.
- **TreeSBA** — ECCV 2024, arXiv 2407.15648. BFS LEGO-tree transformer from multi-view
  images. *Relevance:* connected build-order precedent.
- **Brick-by-Brick** — NeurIPS 2021, arXiv 2110.15481. RL sequential brick assembly.
  *Relevance:* early RL-for-assembly; related-work only.
- **Image2Lego** — arXiv 2108.08477 (2021). Octree AE + legolization. *Relevance:*
  voxel→LEGO conversion pipeline for our transfer demo.
- **Brick-Composer / BC-Bench** — arXiv 2606.05445 (2026). MLLM diverse-brick assembly
  benchmark. *Relevance:* related work; eval-protocol reference.

## 2. Minecraft / voxel generation

- ★ **Scaffold Diffusion** — Jung, NeurIPS 2025 SPIGM WS, arXiv 2509.00062.
  https://github.com/jsjung00/scaffold-diffusion — masked discrete diffusion (MDLM/DiT)
  over **occupied-voxel tokens only**, occupancy-conditioned, 1,432 3D-Craft houses, 32³.
  *Relevance:* closest relative of our diffusion track AND the strongest evidence for our
  two-stage occupancy→materials factoring (their AR baseline reproduces our
  block-scrambling failure; their PE ablation—learned PPL 29.05 vs 3D-sinusoidal
  1.79—motivates our PE arms). They generate materials given GT occupancy; **we add the
  generated occupancy stage (their stated future work).** Note: they declined quantitative
  sample metrics — our eval suite is more rigorous (claimable).
- ★ **Dream-Cubed** — Sakana AI (Merino et al.), arXiv 2604.22847 (2026). Large 3D
  diffusion study on 2M+ Minecraft chunks; **Render-FID** metric + human study; code,
  weights, data public. *Relevance:* scale counterpoint (we are the data-efficient
  regime); adopt Render-FID for comparability.
- ★ **VoxelCNN / 3D-Craft** — Chen et al., ICCV 2019.
  https://github.com/facebookresearch/voxelcnn — order-aware next-block prediction on
  **2,586 crowdsourced houses with full human build order** (the corpus now in
  `data/3d_craft/`, loader `blockgen/utils/corpora.py`). *Relevance:* data source for
  pool-pretraining; evidence humans build in local connected order (supports our
  BFS-order + adjacency-constrained decoding); their ACC@k protocol if we claim ordering.
- **World-GAN** — Awiszus et al., IEEE CoG 2021, arXiv 2106.10155. 3D SinGAN with
  block2vec embeddings; **TPKL-Div** metric. *Relevance:* Minecraft-native distribution
  metric to adopt; block2vec relates to our factored-embedding idea (§9.1).
- **DreamCraft** — Earle et al., FDG 2024, arXiv 2404.15538. CLIP-guided **quantized NeRF
  over block types**, per-prompt optimization. *Relevance:* proof CLIP scores
  Minecraft-style renders sensibly; soft→discrete annealing recipe; contrast: per-instance
  optimization vs our trained generators.
- **text2mc** — UCF 2024. Kaggle dump
  https://www.kaggle.com/datasets/shauncomino/minecraft-builds-dataset (~11k processed
  builds, .h5; loader in `corpora.py`). *Relevance:* pretraining corpus.
- **CraftAssist** — Gray et al., arXiv 1907.08584 (2019). The crowdsourcing platform
  behind the 3D-Craft houses. *Relevance:* data provenance citation.
- **GDMC Settlement Challenge** — Salge et al., arXiv 1803.09853. Human-judged 0-10
  rubric. *Relevance:* human-eval protocol option.
- **T2BM** — arXiv 2406.08751 (2024). Prompted GPT-3.5/4 building generation.
  *Relevance:* zero-shot LLM baseline precedent.
- **MC-Bench / MineBench** — mcbench.ai / minebench.ai (2024-26). Frontier LLMs emit
  code/JSON blocks, human pairwise Glicko voting. *Relevance:* zero-shot-LLM comparison +
  pairwise human-vote protocol.
- **NCA Minecraft artefacts** — Sudhakaran et al., arXiv 2103.08737 (2021). Neural
  cellular automata growing builds. *Relevance:* related-work contrast (local rules =
  inherent local coherence, but one model per artifact).

## 3. Pretrained-LM / text-serialized 3D (track D lineage)

- ★ **LLaMA-Mesh** — NVIDIA, arXiv 2411.09595 (2024). LLaMA-3-8B SFT on raw OBJ text
  (64-bin coords). *Relevance:* "LLM world knowledge transfers to serialized 3D" — the
  thesis our track D tests on blocks.
- **MarioGPT** — Sudhakaran et al., NeurIPS 2023, arXiv 2302.05981. DistilGPT-2 on
  text-serialized Mario levels. *Relevance:* canonical PCG precedent for pretrained-LM
  fine-tuning on tile grids.
- **SuperVoxelGPT** — arXiv 2605.29655 (2026). LM over learned supervoxel tokens.
  *Relevance:* learned-tokenizer contrast to our handcrafted 3D-BPE.

## 4. Tokenization & AR generation of 3D structure

- ★ **MeshGPT** — Siddiqui et al., CVPR 2024, arXiv 2311.15475. Learned geometric VQ
  vocabulary + GPT; per-category ShapeNet scale (= our data scale). *Relevance:* learned
  codes beat raw coordinate tokens — motivates our patch-VQ ablation arm vs 3D-BPE.
- **AutoSDF** — Mittal et al., CVPR 2022, arXiv 2203.09516. **Patch-wise VQ-VAE**
  (independent 8³ patch encode, joint decode) + randomized-order AR. *Relevance:* the
  P-VQ-VAE recipe for our learned-tokenizer arm; random-order training as augmentation.
- **BPT (Blocked & Patchified Tokenization)** — arXiv 2411.07025, CVPR 2025. ~75% mesh
  token compression enabling better AR. *Relevance:* compressive tokenization as a
  contribution axis — validates the 3D-BPE line; cite next to AMT/MeshAnythingV2
  (arXiv 2408.02555, ~50%).
- **OctGPT** — SIGGRAPH 2025, arXiv 2504.09975. Multiscale octree AR, binary spherical
  quantization. *Relevance:* hierarchical AR + quantization-collapse fix (FSQ/BSQ) for the
  VQ arm.
- **Octree adaptive tokenization** — arXiv 2504.02817 (2025). *Relevance:* variable-length
  3D token budgets; related work for BPE compression framing.
- **RAR** — Yu et al., ICCV 2025. Random-order→raster annealing improves AR image gen
  (FID 3.08→2.18). *Relevance:* ordering-as-regularization for our ordering arm.
- **"From Sequential to Spatial"** — arXiv 2512.24639 (2025). Scan-order ablations;
  raster wins among fixed orders. *Relevance:* justifies raster default + our BFS
  hypothesis being about *constraint match*, not order per se.
- **ShapeAssembly** — Jones et al., SIGGRAPH Asia 2020, arXiv 2009.08026. Program-VAE
  over cuboid DSL. *Relevance:* structured/program alternative in related work.
- **PolyGen** — Nash et al., ICML 2020. n-gon mesh AR with canonical ordering.
  *Relevance:* canonical-order lineage (brief cite).

## 5. Discrete diffusion & sampling

- ★ **MaskGIT** — Chang et al., CVPR 2022. Masked generative transformer, confidence
  schedule. *Relevance:* our track B sampler's basis.
- **D3PM** — Austin et al., NeurIPS 2021. Discrete diffusion framework (absorbing state).
  *Relevance:* theoretical basis of our masked formulation.
- **MDLM** — Sahoo/Shi et al. 2024 (simplified masked diffusion LMs). *Relevance:*
  modern masked-diffusion formulation Scaffold builds on.
- ★ **Halton Scheduler for MaskGIT** — Besnier et al., ICLR 2025, arXiv 2503.17076.
  Confidence selection clusters spatially → non-recoverable errors; quasi-random position
  spreading fixes it. *Relevance:* direct motivation for our `stratified` sampler arm.
- ★ **ReMDM** — arXiv 2503.00307 (2025). Principled remasking for pretrained MDLMs
  (revisit committed tokens). *Relevance:* our `remask` sampler arm.
- **Token-Critic** — Lezama et al., ECCV 2022, arXiv 2209.04439. Learned resample
  selector for MaskGIT. *Relevance:* backup if remask/stratified insufficient; "learned
  validity gate" framing.
- **"[MASK] is All You Need" / ReDDiT** — arXiv 2412.06787 / 2505.19656. Gumbel-noise +
  temperature ablations in confidence sampling. *Relevance:* justifies our Gumbel-annealed
  sampler (already implemented) and its tuning.
- **Discrete Flow Matching** — Gat et al., NeurIPS 2024, arXiv 2407.15595. *Relevance:*
  our `flow` sampler; wins mainly at high NFE (why we didn't switch frameworks).
- **Simple Guidance for Discrete Diffusion (D-CFG)** — Schiff et al., ICLR 2025.
  https://discrete-diffusion-guidance.github.io/ *Relevance:* the CFG recipe for our
  conditioned tracks; pair with late-stage guidance ramp (arXiv 2507.08965, 2506.10971).
- **Analog Bits (self-conditioning)** — Chen et al., ICLR 2023, arXiv 2208.04202.
  *Relevance:* only if we try continuous-relaxation diffusion; footnote.
- **Pyramid Discrete Diffusion (PDD)** — Liu et al., ECCV 2024 oral, arXiv 2311.12085.
  Scale-cascaded discrete diffusion, explicit small-data motivation + cross-dataset
  coarse transfer. *Relevance:* our two-level pyramid plan; small-data transfer trick.
- **XCube** — NVIDIA, CVPR 2024 Highlight, arXiv 2312.03806. Sparse-voxel latent
  hierarchy; structure-then-attributes. *Relevance:* factoring-at-scale precedent.
- **SemCity** — CVPR 2024, arXiv 2403.07773. *Relevance:* geometry/semantics separation
  precedent (brief).

## 6. Positional encodings (PE ablation arm)

- **RoFormer/RoPE** — Su et al., 2021, arXiv 2104.09864. *Relevance:* standard relative
  PE; our `rope` arm.
- **ALiBi** — Press et al., ICLR 2022, arXiv 2108.12409. Linear attention biases; train
  short test long. *Relevance:* our `alibi` arm; learned-absolute weakness evidence.
- **PE & length generalization** — Kazemnejad et al., NeurIPS 2023, arXiv 2305.19466.
  ALiBi/NoPE > RoPE for length extrapolation in small decoders. *Relevance:* motivates
  including `none`; our regime is exactly small-decoder.
- (+ Scaffold Diffusion's learned-vs-3D-sinusoidal ablation, §2 above — the in-domain PE
  evidence.)

## 7. RL / reward-model post-training & web-knowledge signals

- ★ **MineCLIP / MineDojo** — Fan et al., NeurIPS 2022 (Outstanding Paper),
  arXiv 2206.08853. https://github.com/MineDojo/MineCLIP — video-text CLIP on 640k
  YouTube Minecraft clips; agrees with human judges on creative building tasks.
  *Relevance:* our domain-matched reward model; render 16-frame orbits to match input.
- **DDPO** — Black et al., 2023, arXiv 2305.13301. RL for diffusion; ~20k reward queries
  for visible gains; used LLaVA-as-reward. *Relevance:* budget calibration + VLM-reward
  precedent.
- **DPOK** — Fan et al., 2023, arXiv 2305.16381. KL-regularized RL. *Relevance:*
  KL-to-reference as reward-hacking defense.
- **Diffusion-DPO** — Wallace et al., 2023, arXiv 2311.12908. Offline preference pairs.
  *Relevance:* our cheap/stable pipeline 2.
- **AR-GRPO** — arXiv 2508.06924 (2025); **STAGE** — arXiv 2509.25027 (2025). GRPO for AR
  image generation; KL/entropy stabilizers. *Relevance:* the algorithm + stabilizer recipe
  for our AR model.
- **d1 / diffu-GRPO** — arXiv 2504.12216 (2025). Policy-gradient RL for masked diffusion
  LMs. *Relevance:* the RL recipe for our diffusion track.
- **DRaFT** (arXiv 2309.17400) / **AlignProp** (arXiv 2310.03739). Differentiable reward
  backprop, ~25× sample-efficient. *Relevance:* pipeline 3 (only if 1-2 plateau).
- **Nabla-R2D3** — NeurIPS 2025, arXiv 2506.15684. RL-aligning 3D generators with 2D
  rewards on renders. *Relevance:* the exact shape of our plan, in mesh domain.
- **DreamReward** (arXiv 2403.14613) / **DreamDPO** (arXiv 2502.04370). RLHF/DPO for
  text-to-3D. *Relevance:* 3D preference-optimization lineage.
- **DreamFields** (arXiv 2112.01455), **CLIP-Mesh** (arXiv 2203.13333), **PureCLIPNeRF**
  (arXiv 2209.15172), **Dream3D** (CVPR 2023). CLIP-guided 3D; augmentations as the
  defense vs adversarial generations. *Relevance:* CLIP-guidance caveats; PureCLIPNeRF =
  CLIP-on-voxel-grids evidence.
- **GPTEval3D** — CVPR 2024, arXiv 2401.04092. VLM pairwise judging of 3D renders,
  human-aligned. *Relevance:* our VLM-judge protocol (pairwise + position swap).
- **Reward hacking / diversity collapse** — arXiv 2409.06493, 2601.03468 (analysis),
  HRF arXiv 2410.08315, GARDO arXiv 2512.24138. *Relevance:* failure-mode design; watch
  duplicate_rate under RL.
- **Cap3D** — Luo et al., NeurIPS 2023, arXiv 2306.07279. Render→BLIP2→CLIP-filter→LLM
  captioning of 3D assets; beats crowdsourced captions. *Relevance:* our VLM auto-labeling
  pipeline for unlabeled corpora.
- **Phidias** — ICLR 2025, arXiv 2409.11406. Retrieval-augmented 3D diffusion.
  *Relevance:* RAG alternative to RL (pipeline option b).
- **Scaling Data-Constrained LMs** — Muennighoff et al., NeurIPS 2023, arXiv 2305.16264.
  ≤4 epochs of repeats ≈ fresh data. *Relevance:* justifies multi-epoch training on
  small data + D4 augmentation.

## 8. Evaluation & memorization

- **Memorization in 3D shape generation** — arXiv 2512.23628 (2025). NN-retrieval +
  Mann-Whitney z-score between gen→train and val→train distances. *Relevance:* formalizes
  our val-baseline protocol; report the z-score for the paper.
- **Render-FID** — via Dream-Cubed (§2). **TPKL-Div** — via World-GAN (§2).
  **CLIP score** — via LegoGPT (§1). **COV/MMD/1-NNA** — standard shape-gen suite
  (Achlioptas et al. 2018 lineage) over occupancy-IoU distance. *Relevance:* metrics to
  add for comparability (research.md §A).

## 9. Datasets (public corpora)

| corpus | size | format | status here |
|---|---|---|---|
| GrabCraft scrape (ours) | 3,952 @ dim≤32 cache | tfrecords + cache | primary labeled set |
| Schematics cache (ours) | ~3k labeled | cache | combined-houses arm |
| **3D-Craft / CraftAssist houses** | 2,586 houses + build order | (Y,Z,X,2) npy + placed.json | **fetched → `data/3d_craft/`, loader `utils/corpora.py`** |
| **text2mc Kaggle dump** | ~11k builds | .h5 voxel arrays | **downloading via kagglehub; loader ready** |
| StableText2Lego | 47k LEGO structs + captions | HF dataset | for the LEGO-transfer phase |
| BrickNet LDraw corpus | 100k+ objects | LDraw programs | for the LEGO-transfer phase |
| Objaverse(-XL/-Mix) | 800k-10M (900k voxelized) | meshes/voxels | optional pretraining reach |
