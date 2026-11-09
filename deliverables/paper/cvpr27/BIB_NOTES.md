# BIB_NOTES: references_new.bib

`references_new.bib` holds 53 new entries for the CVPR 2027 port. None of them repeats a key, arXiv ID or title from `deliverables/paper/iclr/references.bib`; a script checked all three. A BibTeX test run with `\nocite{*}` processed every entry with no warnings. Each entry has a `% verified:` comment. Metadata comes from the arXiv API. Published venues were checked against Crossref, OpenAlex, OpenReview, PMLR, the ACL Anthology or HAL.

Venue notes:
- Kept as arXiv preprints because no published venue was found: Xu 2018, T3Bench, 3DGen-Bench, Point-E, Hunyuan3D 2.0, SceneTeract, Word2Minecraft, STABLE, BrickBench, MultiWorldBench.
- Hi3DEval (NeurIPS 2025) and MineAnyBuild (NeurIPS 2025 D&B) were confirmed only through OpenAlex. MineDojo's "Datasets and Benchmarks Track" label is a reasonable assumption that was not checked separately. Check the proceedings pages before camera-ready.
- Barthet et al. first appeared as IEEE ToG early access in 2022 and was printed in vol. 15(4), 2023. The entry uses 2023.
- The GDMC first-year report (2103.14950) has a published version, KI 34(1) 2020. It is a different paper from the existing `salge2018gdmc` (1803.09853).
- STABLE: the arXiv title is "Rollback-Free Stable Brick Structures Generation". The repository and model are named STABLE.

## Entries

| key | title | verified URL | status |
|---|---|---|---|
| `xu2018empirical` | An empirical study on evaluation metrics of generative adversarial networks | https://arxiv.org/abs/1806.07755 | arXiv preprint |
| `thompson2022evaluation` | On Evaluation Metrics for Graph Generative Models | https://arxiv.org/abs/2201.09871 | published: International Conference on Learning Representations (ICLR) |
| `stein2023exposing` | Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models | https://arxiv.org/abs/2306.04675 | published: Advances in Neural Information Processing Systems (NeurIPS) |
| `kynkaanniemi2023imagenet` | The Role of ImageNet Classes in Fréchet Inception Distance | https://arxiv.org/abs/2203.06026 | published: International Conference on Learning Representations (ICLR) |
| `parmar2022aliased` | On Aliased Resizing and Surprising Subtleties in GAN Evaluation | https://arxiv.org/abs/2104.11222 | published: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) |
| `sajjadi2018assessing` | Assessing Generative Models via Precision and Recall | https://arxiv.org/abs/1806.00035 | published: Advances in Neural Information Processing Systems (NeurIPS) |
| `alaa2022faithful` | How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models | https://arxiv.org/abs/2102.08921 | published: International Conference on Machine Learning (ICML) |
| `jiralerspong2023fld` | Feature Likelihood Divergence: Evaluating the Generalization of Generative Models Using Samples | https://arxiv.org/abs/2302.04440 | published: Advances in Neural Information Processing Systems (NeurIPS) |
| `sai2021perturbation` | Perturbation CheckLists for Evaluating NLG Evaluation Metrics | https://arxiv.org/abs/2109.05771 | published: Conference on Empirical Methods in Natural Language Processing (EMNLP) |
| `karpinska2022demetr` | DEMETR: Diagnosing Evaluation Metrics for Translation | https://arxiv.org/abs/2210.13746 | published: Conference on Empirical Methods in Natural Language Processing (EMNLP) |
| `kocmi2021ship` | To Ship or Not to Ship: An Extensive Evaluation of Automatic Metrics for Machine Translation | https://arxiv.org/abs/2107.10821 | published: Proceedings of the Sixth Conference on Machine Translation (WMT) |
| `sutherland2017optimizedmmd` | Generative Models and Model Criticism via Optimized Maximum Mean Discrepancy | https://arxiv.org/abs/1611.04488 | published: International Conference on Learning Representations (ICLR) |
| `bounliphone2016relative` | A Test of Relative Similarity For Model Selection in Generative Models | https://arxiv.org/abs/1511.04581 | published: International Conference on Learning Representations (ICLR) |
| `lopezpaz2017c2st` | Revisiting Classifier Two-Sample Tests | https://arxiv.org/abs/1610.06545 | published: International Conference on Learning Representations (ICLR) |
| `adebayo2018sanity` | Sanity Checks for Saliency Maps | https://arxiv.org/abs/1810.03292 | published: Advances in Neural Information Processing Systems (NeurIPS) |
| `he2023t3bench` | T$^3$Bench: Benchmarking Current Progress in Text-to-3D Generation | https://arxiv.org/abs/2310.02977 | arXiv preprint |
| `wu2024gpteval3d` | GPT-4V(ision) is a Human-Aligned Evaluator for Text-to-3D Generation | https://arxiv.org/abs/2401.04092 | published: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) |
| `zhang2025gen3dbench` | 3DGen-Bench: Comprehensive Benchmark Suite for 3D Generative Models | https://arxiv.org/abs/2503.21745 | arXiv preprint |
| `duggal2025eval3d` | Eval3D: Interpretable and Fine-grained Evaluation for 3D Generation | https://arxiv.org/abs/2504.18509 | published: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) |
| `zhang2025mate3d` | Benchmarking and Learning Multi-Dimensional Quality Evaluator for Text-to-3D Generation | https://arxiv.org/abs/2412.11170 | published: IEEE/CVF International Conference on Computer Vision (ICCV) |
| `zhang2025hi3deval` | Hi3DEval: Advancing 3D Generation Evaluation with Hierarchical Validity | https://arxiv.org/abs/2508.05609 | published: Advances in Neural Information Processing Systems (NeurIPS) |
| `nichol2022pointe` | Point-E: A System for Generating 3D Point Clouds from Complex Prompts | https://arxiv.org/abs/2212.08751 | arXiv preprint |
| `yang2019pointflow` | PointFlow: 3D Point Cloud Generation with Continuous Normalizing Flows | https://arxiv.org/abs/1906.12320 | published: IEEE/CVF International Conference on Computer Vision (ICCV) |
| `shu2019treegan` | 3D Point Cloud Generative Adversarial Network Based on Tree Structured Graph Convolutions | https://arxiv.org/abs/1905.06292 | published: IEEE/CVF International Conference on Computer Vision (ICCV) |
| `zeng2022lion` | LION: Latent Point Diffusion Models for 3D Shape Generation | https://arxiv.org/abs/2210.06978 | published: Advances in Neural Information Processing Systems (NeurIPS) |
| `xue2023ulip` | ULIP: Learning a Unified Representation of Language, Images, and Point Clouds for 3D Understanding | https://arxiv.org/abs/2212.05171 | published: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) |
| `zhou2024uni3d` | Uni3D: Exploring Unified 3D Representation at Scale | https://arxiv.org/abs/2310.06773 | published: International Conference on Learning Representations (ICLR) |
| `jain2022dreamfields` | Zero-Shot Text-Guided Object Generation with Dream Fields | https://arxiv.org/abs/2112.01455 | published: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) |
| `gao2022get3d` | GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images | https://arxiv.org/abs/2209.11163 | published: Advances in Neural Information Processing Systems (NeurIPS) |
| `xiang2025trellis` | Structured 3D Latents for Scalable and Versatile 3D Generation | https://arxiv.org/abs/2412.01506 | published: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) |
| `zhao2025hunyuan3d` | Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation | https://arxiv.org/abs/2501.12202 | arXiv preprint |
| `paschalidou2021atiss` | ATISS: Autoregressive Transformers for Indoor Scene Synthesis | https://arxiv.org/abs/2110.03675 | published: Advances in Neural Information Processing Systems (NeurIPS) |
| `tang2024diffuscene` | DiffuScene: Denoising Diffusion Models for Generative Indoor Scene Synthesis | https://arxiv.org/abs/2303.14207 | published: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) |
| `feng2023layoutgpt` | LayoutGPT: Compositional Visual Planning and Generation with Large Language Models | https://arxiv.org/abs/2305.15393 | published: Advances in Neural Information Processing Systems (NeurIPS) |
| `yang2024holodeck` | Holodeck: Language Guided Generation of 3D Embodied AI Environments | https://arxiv.org/abs/2312.09067 | published: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) |
| `yang2024physcene` | PhyScene: Physically Interactable 3D Scene Synthesis for Embodied AI | https://arxiv.org/abs/2404.09465 | published: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) |
| `tam2026sceneeval` | SceneEval: Evaluating Semantic Coherence in Text-Conditioned 3D Indoor Scene Synthesis | https://arxiv.org/abs/2503.14756 | published: IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) |
| `maillard2026sceneteract` | SceneTeract: Probing and Improving Agent-Aware Activity Reasoning in 3D Indoor Scenes | https://arxiv.org/abs/2603.29798 | arXiv preprint |
| `fan2022minedojo` | MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge | https://arxiv.org/abs/2206.08853 | published: Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track |
| `huang2025word2minecraft` | Word2Minecraft: Generating 3D Game Levels through Large Language Models | https://arxiv.org/abs/2503.16536 | arXiv preprint |
| `hu2024t2bm` | 3D Building Generation in Minecraft via Large Language Models | https://arxiv.org/abs/2406.08751 | published: IEEE Conference on Games (CoG) |
| `barthet2023openended` | Open-Ended Evolution for Minecraft Building Generation | https://arxiv.org/abs/2209.03108 | published: IEEE Transactions on Games |
| `jiang2022controllable` | Learning Controllable 3D Level Generators | https://arxiv.org/abs/2206.13623 | published: International Conference on the Foundations of Digital Games (FDG) |
| `khalifa2020pcgrl` | PCGRL: Procedural Content Generation via Reinforcement Learning | https://arxiv.org/abs/2001.09212 | published: AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment (AIIDE) |
| `salge2020gdmcreport` | The AI Settlement Generation Challenge in Minecraft: First Year Report | https://arxiv.org/abs/2103.14950 | published: KI - K{\"u}nstliche Intelligenz |
| `wei2025mineanybuild` | MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents | https://arxiv.org/abs/2505.20148 | published: Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track |
| `xu2026stable` | Rollback-Free Stable Brick Structures Generation | https://arxiv.org/abs/2605.06947 | arXiv preprint |
| `kulits2026brickbench` | BrickBench: Evaluating Agentic Brick Design | https://arxiv.org/abs/2610.12452 | arXiv preprint |
| `xu2026multiworldbench` | MultiWorldBench: Do Independently Controlled Views Describe One Shared World? | https://arxiv.org/abs/2610.11723 | arXiv preprint |
| `morozov2021selfsupervised` | On Self-Supervised Image Representations for GAN Evaluation | https://openreview.net/forum?id=NeRdBeTionN | published: ICLR 2021 (spotlight); no arXiv version |
| `renz2019failure` | On Failure Modes in Molecule Generation and Optimization | https://doi.org/10.1016/j.ddtec.2020.09.003 | published: Drug Discovery Today: Technologies 32-33 (2019); ID 1912.05910 was wrong |
| `smith2010expressive` | Analyzing the Expressive Range of a Level Generator | https://doi.org/10.1145/1814256.1814260 | published: PCGames workshop 2010 (ACM) |
| `comino2025text2mc` | Minecraft Builds Dataset (text2mc) | https://www.kaggle.com/datasets/shauncomino/minecraft-builds-dataset | @misc Kaggle; no paper found |

## Dropped or not added

| requested | reason |
|---|---|
| Achlioptas et al. 1707.02392 | already in the ICLR bib as `achlioptas2018pointclouds` |
| World-GAN 2106.10155 | already present as `awiszus2021worldgan` |
| Hervé & Salge 2107.02457 | already present as `herve2021pcgmetrics` |
| BrickAnything 2605.26182 | already present as `ni2026brickanything` (arXiv entry, matches) |
| Renz et al. as 1912.05910 | that ID is "CORE: Automatic Molecule Optimization Using Copy & Refine Strategy" (Fu et al., AAAI 2020). Renz et al. has no arXiv version. Added from the journal DOI instead (`renz2019failure`). |
| text2mc "UCF" attribution | `references.md` says UCF 2024, but no source confirmed it. The entry cites only the Kaggle dataset and its owner. |

No other ID resolved to the wrong paper.

## Checks on existing keys (report only; the ICLR bib was not edited)

1. **`luan2026rendermetric` (TGE, AAAI 2026, 2512.01380).** TGE is a *full-reference* metric. It scores an input mesh against a reference mesh using a PointNet++ variant. Its input is the mesh's own vertex set with per-vertex RGB, reduced by farthest-point sampling and ball query. It does not sample only the exterior or visible surface, and it does not sample points uniformly over faces. It uses whatever vertices the mesh has, interior surfaces included. The paper's motivation does criticise renders for "incomplete structural coverage and sensitivity to viewpoint choices". Our text says it "propose[s] surface-point sampling" and that a structure and its interior-filled version get "identical surface-point representations". That holds only if the build is meshed as an exterior shell. A hollow house meshed with its inner wall faces would get different vertices once filled. Suggested rewording: "replaces renders with points sampled from the mesh; for voxel builds meshed as exterior shells this remains blind to interior fill". Also note that TGE is paired (reference-based), not distributional.
2. **`jayasumana2024cmmd` for "the biased MMD estimator".** This is inaccurate as worded. The paper's Eq. (4) is the **unbiased** U-statistic, and its Table 1 promotes CMMD as an "unbiased estimator". However, the official code (`google-research/cmmd/distance.py`, `mmd()`) implements the **biased/minimum-variance V-statistic** (its docstring says so, citing Gretton et al. Eq. 5). Options: cite `gretton2012mmd` for the biased estimator, or say "the biased MMD estimator used in the CMMD reference implementation \citep{jayasumana2024cmmd}".
3. **`earle2024dreamcraft` as a "Minecraft world model".** Not accurate. DreamCraft trains quantized NeRFs, one per text prompt, to generate functional 3D artifacts in Minecraft. It is text-guided generation, not a world model that predicts dynamics. `2_related.tex:20` cites it twice: as a voxel generator, which is fine, and as a "Minecraft world model", which is wrong. Candidates for the world-model slot: `merino2026dreamcubed` (if accurate) or `xu2026multiworldbench` (a benchmark *of* multiplayer Minecraft world models).
