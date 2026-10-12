# Nodes in, adjacency out: research memo

Idea under review: train a model whose input is a list of nodes and whose output is an
adjacency matrix, with (1) voxel blocks and (2) 3D-BPE pieces (`blockgen/tokenizers/cluster_bpe.py`)
as nodes. Builds on T24 (`pick_n_place`), T28 and T49 in `results.md`, and on
`blockgen/utils/attach_order.py`. Companion memo for the attention line:
`tools/misc/attention_graphs/RESEARCH.md`.

## Summary

The starting framing is right on its two main points and needs three corrections.

**Confirmed.** (a) With coordinates in the input the task is trivial: T28's model-free
`coord_L1` scored 0.980 AUC and T49's oracle 1.000. The useful version takes nodes without
coordinates. (b) Many builds share one multiset, so the output needs a generative model
over edges, not a regressor.

**Correction 1: for voxels the adjacency matrix is redundant and constrained.** A voxel
build's 6-adjacency graph is an *induced* subgraph of the Z^3 lattice: two blocks in
face-sharing cells are always adjacent. So the edge set is a function of the positions,
and a direction-labelled spanning tree (N-1 edges) already determines everything, up to
translation. That is exactly `attach_order.py`'s `SEED / ATTACH(piece, dir)` encoding.
Emitting all N^2 entries costs quadratic output for no extra information, and it adds
a third realizability condition on top of the two in the brief: besides cycle
consistency and injectivity, every lattice-adjacent pair must be marked as an edge
(*closure*). One-shot models will break all three.

**Correction 2: equivariant set-to-graph cannot even average.** Nodes of the same block
type are exchangeable (a median house has hundreds of `oak_planks`). A deterministic
permutation-equivariant map such as Set2Graph sends identical inputs to identical rows,
so it cannot output an asymmetric graph among identical nodes at all. Symmetry has to
be broken by noise (random node features, Sato et al. 2021; the noise in diffusion) or
by an ordering (autoregression). The training target is also only defined up to
relabelling same-type nodes, so a per-entry loss needs a matching step or a canonical
order.

**Correction 3: binary edges are not a usable output.** Without direction labels,
realization becomes unit-length lattice embedding. For graphs in general that is
NP-complete (Bhatt & Cosmadakis 1987), and lattice HP folding, a close relative, is
NP-complete too (Berger & Leighton 1998). It is also ambiguous: a fence or a 1-wide
pillar embeds as any self-avoiding walk. So edges must carry direction labels (voxels)
or offsets (pieces, LEGO studs). With labels, realization is a linear-time BFS.

**Recommended formulation.** Use inventory-conditioned autoregressive decoding of a
direction-labelled spanning tree, with legality masks. In practice that is
`pick_n_place` with the picker restricted to the remaining inventory and the placer
trained with T24e's cell-marginalized loss. This makes realizability hold by
construction, and it fixes T24d's STOP problem for free: generation ends when the
inventory is empty. Run a sparse one-shot or diffusion arm next to it with
maximum-spanning-tree repair, as the comparison the reviewers will ask for.

**Novelty: modest.** Set-to-graph (Serviansky et al. 2020), labelled LEGO graph
generation (Thompson et al. 2020) and budget-constrained brick assembly (Ahn et al.,
TMLR) cover the pieces. The paper-worthy version is house-scale (1k+ nodes)
inventory-conditioned construction with exact realizability, plus a controlled
"positions from bonds vs bonds from positions" comparison on the lattice (the
MiDi-vs-EDM question, transplanted).

## 1. Formulation choices

### 1.1 Node features: with or without coordinates

| choice | buys | failure modes |
|---|---|---|
| full coordinates | nothing; adjacency is a lookup | T28/T49: any learner just learns `L1 == 1` |
| no coordinates (type only) | the graph *is* the structure; a clean "inventory to build" task | maximal exchangeability (Correction 2); high multimodality |
| coarse features (y-layer, or a role such as wall, roof, floor) | breaks most of the symmetry; the y-layer alone fixes support order | leaks geometry; if the features come from the real build, conditioning on them is half the answer, so they must be generated too |
| random IDs or Laplacian-free noise features | makes equivariant one-shot models expressive enough (Sato et al. 2021) | they are noise, not information; samples vary with the seed (that is the point) |

Recommendation: type only, plus random node features for any non-autoregressive arm. A
coarse-feature arm (type plus y-layer) is a useful ablation: it measures how much of
the difficulty is vertical layout.

### 1.2 Edge labels

| label | realization | notes |
|---|---|---|
| binary | NP-hard and ambiguous (Correction 3) | reject |
| 6 face directions (voxels) | BFS from any root: `pos[child] = pos[parent] + dir` | cycle consistency means the direction vectors sum to zero around every cycle (equivalently, a potential exists); checkable in O(N+E) |
| relative offset `(dx,dy,dz)` (pieces) | same BFS on anchors | the offset vocabulary depends on the piece pair; contact must be checked (two pieces at an offset may overlap or not touch) |
| stud offset plus orientation (LEGO) | same | this is Thompson et al.'s labelled-edge setup |

### 1.3 Decoding

- **One-shot set-to-graph** (Set2Graph; SPECTRE for one-shot generation; EDP-GNN). It is
  fast, but with no symmetry breaking it is degenerate (Correction 2). Even with noise
  features, independent per-edge outputs break cycle consistency, because each edge is
  decided without knowing the others. It needs repair (1.4). The scale is N^2 unless
  candidates are restricted.
- **Autoregressive edges** (GraphRNN, DGMG, GRAN, BiGG; graph-as-sequence transformers
  such as G2PT and AutoGraph). An ordering breaks the symmetry. Tracking the partial
  embedding lets every step be masked to legal faces, so all three realizability
  conditions hold by construction. The costs are order sensitivity (`attach_order.py`
  makes ordering a parameter for this reason) and exposure bias, which only
  free-running evaluation can see (T24b). BiGG shows that sparse autoregressive
  generation scales to O((n+m) log n).
- **Discrete diffusion or flow over edges** (DiGress, GDSS, DeFoG). The noise breaks the
  symmetry and the denoiser is permutation-equivariant. Dense versions stop around a few
  hundred nodes. Sparse variants exist: EDGE, which removes edges as noise and
  regenerates them on a few active nodes per step, and SparseDiff, which reaches about
  1,000 nodes. Neither enforces lattice consistency, so a projection step is needed.
- **Masked iterative decoding** (MaskGIT-style). Commit the most confident edges, re-mask
  the rest, and re-predict. This sits naturally next to constrained decoding: after each
  round, realize the committed forest and mask any candidate that now conflicts. It is a
  cheap middle ground between autoregressive and diffusion.

### 1.4 Enforcing realizability

1. **Constrained decoding** (autoregressive only). Keep `pos[]` and an occupancy hash, and
   allow only `ATTACH` to an empty cell. Injectivity and cycle consistency are then
   guaranteed. Closure is implied, because edges are *derived* from the realized
   positions rather than predicted. `pick_n_place` already does this (its legality mask
   matched 11,940 of 11,940 true placements).
2. **Projection** (one-shot or diffusion). Take the maximum-weight spanning tree on the
   edge logits, realize it by BFS, resolve collisions by dropping the lower-scored
   subtree or re-attaching it greedily, then recompute the edges from positions. This
   throws away every non-tree prediction, which shows how much of a dense adjacency is
   wasted.
3. **Post-hoc repair** (constraint solving or local search over conflicting edges). It is
   the most flexible and the least honest: as with T24c's `connected = 1.000`, validity
   by construction is a filter, not learning. Report the pre-repair rate.

### 1.5 Where the node list comes from at generation time

- **Given** (conditional task). Inventory-conditioned construction ("build a house from
  this chest") is a real capability in Minecraft survival and in LEGO (build from the set
  you own). Ahn et al. study a budget-constrained version for LEGO.
- **A separate multiset model.** A count vector over block types is a small object (an AR
  model over `(type, count)` pairs, or a Dirichlet-multinomial fit). Do **not** sample the
  training multisets: that leaks build identity and inflates the memorization gates.
- **Joint generation.** This is what `pick_n_place` already does (the picker emits the next
  type). The inventory-conditioned model and the joint model share a trunk, so both can
  be trained by dropping the inventory prefix some of the time (classifier-free style).

A concrete side benefit: with the inventory given, the build length is known, so no STOP
token is needed. T24d showed that STOP is the weak point. At any cap the corpus could
afford, most builds could not teach an ending, and suppressing STOP made the model never
stop. Inventory conditioning removes that failure entirely. The benchmark's palette
pillar becomes a structural guarantee, though, and must be flagged as such (see 3).

## 2. Voxel nodes vs 3D-BPE piece nodes

| | voxel nodes | 3D-BPE piece nodes |
|---|---|---|
| N per house | p50 1,069, p90 2,611, max 7,673 (T24e, largest component) | only 1.2 to 1.9 voxels per piece on real builds (T18), so N drops about 1.5x; T28's held-out set had a median of 171 pieces, at Track D's resolution |
| edge semantics | 6 directions, degree at most 6, closed-form realization | offset between anchors; contact depends on both shapes; degree unbounded |
| label vocabulary | 6 | per piece-pair offset set; factor as `shape × rot × family` (T18's `piece_factors.py`) and predict the offset relative to a contact face |
| exchangeability | extreme (few types, huge counts) | lower (pieces carry shape), but T18 found 98.8% of pieces are single-material rectangles, so it is still high |
| collision check | one cell | multi-cell overlap; one wrong offset on a 96-voxel plate deposits a floating slab (the T10 failure) |
| ambiguity of the target | none: one graph per build | the BPE segmentation is canonical for real builds, but a generated piece graph need not be a canonical tokenization; many piece graphs realize the same voxels, so the likelihood is spread over segmentations |
| port to LEGO | weak | direct: pieces are bricks and offsets are stud connections |

Read: voxels are the right first test because realization is exact and cheap and the
code exists (`attach_order.py`, `pick_n_place`). Pieces are the right *second* test,
because they are the LEGO rehearsal, but the 1.5x compression does not buy scale. The
case for pieces is semantic (typed connections, the `(parent_piece, dir, child_piece)`
triples that T24e measured shrink the candidate set from 1,928 to 27), not N.

## 3. Evaluation

1. **Edge F1 on held-out graphs: report it, but do not rank on it.** It is weak for three
   reasons. First, it is ill-defined under exchangeability: scoring needs a type-preserving
   node matching, which is graph matching (hard in general; use a Hungarian match on
   realized positions after D4 and translation alignment as an approximation). Second,
   under multimodality a perfect sampler of p(build | inventory) scores low F1 against
   the one real build. Third, for voxels F1 can be computed after realization anyway,
   where it reduces to an aligned IoU. A better teacher-forced number is `place_acc` with
   the cell-marginalized label, against its chance floor (T24e).
2. **Realizability rate before repair.** Cycle-consistency violations per cycle-closing
   edge, the collision rate, and the closure-violation rate. These numbers are 0 by
   construction for constrained AR, so report them for the one-shot and diffusion arms,
   and report the AR mask's rejection mass as its analogue.
3. **Connectivity and coherence** (`lcc_ratio`, `floating_block_frac`), as *distance to
   real*, never as raw maxima (T24c: 1.000 connected is further from real than 0.977).
4. **The benchmark on realized builds** (`docs/benchmark.md`): `geom_kid` (co-primary,
   no GPU) and `mv_dino_kid`, the novelty and diversity gates, the structure pillar
   (`thickness_*`, `wall_frac`, `interior_ratio_open`) and BlockScore, at n of at least
   256. Two adjustments are needed. (a) When the inventory is taken from held-out builds,
   `palette_jsd_*` and the block-count W1 are matched by construction: exclude them from
   BlockScore or mark them as guaranteed, as T24c did for connectivity. (b) Add an
   *inventory-matched* score: for each held-out inventory, `geom_kid` of the samples
   against the reference, plus the nearest-real IoU, to check the model is not
   retrieving a training house with a similar multiset.
5. **A free-running number in every training log** (the T24b rule): blocks used, the
   fraction of the inventory consumed, and the realizability rate.
6. **A must-run baseline: the multiset nearest neighbour.** Retrieve the training build
   whose type histogram is closest to the inventory. If that already scores well on
   `geom_kid`, the inventory carries most of the information and the model must beat
   retrieval on novelty, not on realism.

## 4. Novelty verdict

A reviewer would cite these first:

- **Thompson et al. 2020 (DGMLG)**: LEGO structures as directed graphs with labelled
  connection edges, generated autoregressively (DGMG-style). This is the "graph IS the
  structure" idea, already applied to bricks, at small scale.
- **Serviansky et al. 2020 (Set2Graph)**: the set-to-graph function class, with
  universality results.
- **Ahn et al. (budget-aware sequential brick assembly, TMLR)**: brick assembly that
  respects a limited number and type of bricks, which is the inventory-conditioned
  setting.
- Then: PolyGen (pointer attention to earlier vertices, i.e. faces as edges over a
  generated node set), DiGress/DeFoG for the generic machinery, LegoGPT/BrickGPT and
  LegoACE as the domain state of the art (both emit poses, not graphs), and MiDi vs EDM
  for the bonds/positions question. On that question, MiDi reports that generating bonds
  jointly beats deriving them from EDM's distances (92% vs 6% stable on GEOM-DRUGS).

Adjacent literatures to cite briefly: chemistry's "bonds from positions" (EDM, xyz2mol)
vs "positions from bonds" (GeoMol, and distance geometry in general, Liberti et al.
2014); scene-graph-to-layout (Johnson et al. 2018, Graph-to-3D), where the graph is
given and the layout generated, the inverse of this task; part graphs (StructureNet);
and assembly planning (Kim et al. 2020, Tian et al. 2022).

The generic claim "nodes in, adjacency out" is not novel. Three things would be:

1. **Scale with exact realizability.** House-scale (median about 1k, p90 about 2.6k
   nodes) labelled-lattice graph generation, where every sample is realizable by
   construction. Prior lattice and LEGO graph work stays in the tens to low hundreds of
   nodes.
2. **Inventory-conditioned construction as a task**, with an evaluation protocol
   (held-out inventories, a retrieval baseline, palette pillars marked as guaranteed) and
   the STOP-free length control that it gives.
3. **A controlled representation study**: positions-first (raster AR, with edges derived)
   vs labelled-edges-first (tree AR, with positions derived) vs one-shot edges with
   repair, all at matched compute on the same split and scored on the benchmark. This is
   the cleanest version of the MiDi/EDM debate outside chemistry, and the lattice makes
   realizability exactly checkable.

The current paper is metric-only (CVPR 2027), so this is a models-track contribution and
a later paper. Its strongest form is (2) plus (3), with the voxel-to-piece-to-LEGO ladder
as the generalization story.

## 5. Relation to the attention line (T28, T49)

- **T49's planned next experiment is this idea's autoregressive arm.** It plans a
  coordinate-free parent-pointer serialization for the voxel AR model, with one head
  supervised to point at the parent (LISA-style) and masked to legal faces. That *is*
  "labelled spanning tree, decoded autoregressively", minus the inventory. Merge them:
  run the supervised pointer head as arm A, and the same model with an inventory prefix
  as arm B. One experiment answers both questions.
- **The supervised pointer head beats reading attention.** T28 found a frozen LLM's heads
  carry only a lossy copy of printed coordinates. T49 found a trained voxel model grows a
  neighbour head, but one head covers one direction (AP 0.55), and the head fails under
  shuffled order while a linear probe on the same states still reads adjacency at 0.96 to
  0.99. So adjacency is in the representation, and a *trained* bilinear or pointer head
  (Dozat and Manning style) is the reliable reader. That is what the edge decoder here
  should be.
- **What it competes with.** The dense one-shot arm competes with the pointer head on
  efficiency but not on validity. T49's probe result only says edges are readable *when
  coordinates are in the input*, which is the trivial case (1.1). It is no evidence that
  a coordinate-free one-shot decoder will work.
- **What it absorbs from T24.** The sparse-candidate refactor (BFS keeps the true parent
  within the last 512 placements in 100% of cases), the cell-marginalized placer loss, and
  `test_prefix_features_match_full_sequence` all carry over unchanged. The inventory
  version needs one more guard: the remaining-inventory feature must be computed the same
  way under teacher forcing and in free running (the T24b skew, again).

## 6. Suggested first runs (cheapest first)

1. A multiset nearest-neighbour baseline on `houses_32` (an hour, no training). How much
   does the inventory determine?
2. `pick_n_place` plus an inventory prefix and inventory-masked picking, with sparse
   candidates and `max_nodes` of at least 1,024; n of at least 256 on held-out
   inventories, against unconditional `pick_n_place`.
3. A one-shot arm: an equivariant set transformer with random node features and per-pair
   6-way-plus-none logits on candidate pairs, with MST repair. Report the pre-repair
   realizability rate.
4. Pieces: the same as run 2 over the factored piece vocabulary, with typed offsets.

## 7. Follow-up (2026-10-08): cell assignment, then connections, on full houses

T52 found that a per-block coarse cell was the largest lever and that merge decoding failed
without a frame. The third prototype (README, "Full houses") makes the cell explicit and
generates whole houses_32 builds in two stages.

* **Stage 1 is a contingency-table problem.** The cell assignment is an integer table whose
  row sums (blocks per coarse cell) and column sums (the inventory) are both fixed. Sampling
  integer tables with fixed margins is classical (iterative proportional fitting, Deming and
  Stephan 1940). We let a transformer predict per-cell type distributions, scale them with
  IPF to the right expected margins, and draw integer counts slot by slot, which gives exact
  margins and a real distribution over assignments. The learned part helps only modestly: per-block NLL drops
  from 1.89 to 1.53 nats against a dominant-type prior and the slot error from 0.34 to 0.30,
  and the gain does not carry over to the built houses.
* **Stage 2 over type slots** follows the inventory-conditioned spanning-tree decoder of
  section 1.3, with the inventory now split by cell. Interchangeable blocks of one (cell,
  type) slot collapse into one action, so the target is no longer diluted (12.7% of the
  first-step merge candidates of a sample house are correct, against T52's 43%). The grow
  decoder needs a local encoder at this scale: a 3D U-Net (Ronneberger et al. 2015) at fine
  resolution with a transformer over the coarse cells at the bottleneck.
* **Training on the decoder's own states** is the standard remedy for exposure bias
  (scheduled sampling, Bengio et al. 2015; DAgger, Ross et al. 2011). It needs an expert for
  off-distribution states. Ours keeps the real house as the reference and credits only
  merges that extend clusters consistent with it. Under that expert, own-rollout training
  made the merge decoder worse (23 clusters left instead of 6), which suggests the expert
  must also reward repairs, or accept any completion consistent with the condition, for
  DAgger to help a decoder that can paint itself into a corner.
* **Result in one line.** Inside a cell frame the grow decoder learns local contacts (glue F1
  0.65 vs 0.50 for a random policy with the same assignment) but recovers no geometry beyond
  the condition (voxel IoU 0.33, equal to retrieval) at f = 4; merging in cells no longer
  collapses to lines but deadlocks with about 6 clusters left. The resolution of the frame,
  not the stage-2 model, looks like the binding constraint.

## References

- Ahn, S., Kim, J., Cho, M., Park, J. Budget-Aware Sequential Brick Assembly with Efficient Constraint Satisfaction. TMLR. https://arxiv.org/abs/2210.01021
- Bengio, S., Vinyals, O., Jaitly, N., Shazeer, N. Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. NeurIPS 2015. https://arxiv.org/abs/1506.03099
- Berger, B., Leighton, T. Protein Folding in the Hydrophobic-Hydrophilic (HP) Model is NP-Complete. Journal of Computational Biology 5(1):27-40, 1998.
- Bhatt, S. N., Cosmadakis, S. S. The Complexity of Minimizing Wire Lengths in VLSI Layouts. Information Processing Letters, 1987.
- Chang, H., Zhang, H., Jiang, L., Liu, C., Freeman, W. T. MaskGIT: Masked Generative Image Transformer. CVPR 2022. https://arxiv.org/abs/2202.04200
- Chen, D., Krimmel, M., Borgwardt, K. Flatten Graphs as Sequences: Transformers are Scalable Graph Generators (AutoGraph). NeurIPS 2025. https://arxiv.org/abs/2502.02216
- Chen, X., He, J., Han, X., Liu, L.-P. Efficient and Degree-Guided Graph Generation via Discrete Diffusion Modeling (EDGE). ICML 2023. https://arxiv.org/abs/2305.04111
- Chen, X., Wang, Y., He, J., Du, Y., Hassoun, S., Xu, X., Liu, L.-P. Graph Generative Pre-trained Transformer (G2PT). ICML 2025. https://arxiv.org/abs/2501.01073
- Dai, H., Nazi, A., Li, Y., Dai, B., Schuurmans, D. Scalable Deep Generative Modeling for Sparse Graphs (BiGG). ICML 2020. https://arxiv.org/abs/2006.15502
- Deming, W. E., Stephan, F. F. On a Least Squares Adjustment of a Sampled Frequency Table When the Expected Marginal Totals are Known. Annals of Mathematical Statistics 11(4):427-444, 1940.
- Dhamo, H., Manhardt, F., Navab, N., Tombari, F. Graph-to-3D: End-to-End Generation and Manipulation of 3D Scenes Using Scene Graphs. ICCV 2021. https://arxiv.org/abs/2108.08841
- Dozat, T., Manning, C. D. Deep Biaffine Attention for Neural Dependency Parsing. ICLR 2017. https://arxiv.org/abs/1611.01734
- Ganea, O.-E., Pattanaik, L., Coley, C. W., Barzilay, R., Jensen, K. F., Green, W. H., Jaakkola, T. S. GeoMol: Torsional Geometric Generation of Molecular 3D Conformer Ensembles. NeurIPS 2021. https://arxiv.org/abs/2106.07802
- Hoogeboom, E., Satorras, V. G., Vignac, C., Welling, M. Equivariant Diffusion for Molecule Generation in 3D (EDM). ICML 2022. https://arxiv.org/abs/2203.17003
- Jo, J., Lee, S., Hwang, S. J. Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations (GDSS). ICML 2022. https://arxiv.org/abs/2202.02514
- Johnson, J., Gupta, A., Fei-Fei, L. Image Generation from Scene Graphs. CVPR 2018. https://arxiv.org/abs/1804.01622
- Kim, J., Chung, H., Lee, J., Cho, M., Park, J. Combinatorial 3D Shape Generation via Sequential Assembly. NeurIPS 2020 Workshop on ML for Engineering Modeling, Simulation and Design. https://arxiv.org/abs/2004.07414
- Kim, Y., Kim, W. Y. Universal Structure Conversion Method for Organic Molecules: From Atomic Connectivity to Three-Dimensional Geometry (xyz2mol). Bulletin of the Korean Chemical Society 36(7):1769-1777, 2015.
- Li, Y., Vinyals, O., Dyer, C., Pascanu, R., Battaglia, P. Learning Deep Generative Models of Graphs (DGMG). arXiv 2018. https://arxiv.org/abs/1803.03324
- Liao, R. et al. Efficient Graph Generation with Graph Recurrent Attention Networks (GRAN). NeurIPS 2019. https://arxiv.org/abs/1910.00760
- Liberti, L., Lavor, C., Maculan, N., Mucherino, A. Euclidean Distance Geometry and Applications. SIAM Review 56(1):3-69, 2014. https://arxiv.org/abs/1205.0349
- Martinkus, K., Loukas, A., Perraudin, N., Wattenhofer, R. SPECTRE: Spectral Conditioning Helps to Overcome the Expressivity Limits of One-shot Graph Generators. ICML 2022. https://proceedings.mlr.press/v162/martinkus22a.html
- Mo, K., Guerrero, P., Yi, L., Su, H., Wonka, P., Mitra, N., Guibas, L. J. StructureNet: Hierarchical Graph Networks for 3D Shape Generation. SIGGRAPH Asia 2019 (ACM TOG 38(6)). https://arxiv.org/abs/1908.00575
- Nash, C., Ganin, Y., Eslami, S. M. A., Battaglia, P. PolyGen: An Autoregressive Generative Model of 3D Meshes. ICML 2020. https://arxiv.org/abs/2002.10880
- Niu, C., Song, Y., Song, J., Zhao, S., Grover, A., Ermon, S. Permutation Invariant Graph Generation via Score-Based Generative Modeling (EDP-GNN). AISTATS 2020. https://arxiv.org/abs/2003.00638
- Pun, A., Deng, K., Liu, R., Ramanan, D., Liu, C., Zhu, J.-Y. Generating Physically Stable and Buildable Brick Structures from Text (LegoGPT / BrickGPT). ICCV 2025. https://arxiv.org/abs/2505.05469
- Qin, Y., Madeira, M., Thanou, D., Frossard, P. DeFoG: Discrete Flow Matching for Graph Generation. ICML 2025. https://arxiv.org/abs/2410.04263
- Qin, Y., Vignac, C., Frossard, P. SparseDiff: Sparse Discrete Diffusion for Scalable Graph Generation. TMLR 2025. https://arxiv.org/abs/2311.02142
- Ronneberger, O., Fischer, P., Brox, T. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015. https://arxiv.org/abs/1505.04597
- Ross, S., Gordon, G., Bagnell, D. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger). AISTATS 2011. https://arxiv.org/abs/1011.0686
- Sato, R., Yamada, M., Kashima, H. Random Features Strengthen Graph Neural Networks. SDM 2021. https://arxiv.org/abs/2002.03155
- Serviansky, H., Segol, N., Shlomi, J., Cranmer, K., Gross, E., Maron, H., Lipman, Y. Set2Graph: Learning Graphs From Sets. NeurIPS 2020. https://arxiv.org/abs/2002.08772
- Strubell, E., Verga, P., Andor, D., Weiss, D., McCallum, A. Linguistically-Informed Self-Attention for Semantic Role Labeling (LISA). EMNLP 2018. https://aclanthology.org/D18-1548/
- Thompson, R., Ghalebi, E., DeVries, T., Taylor, G. W. Building LEGO Using Deep Generative Models of Graphs. NeurIPS 2020 Workshop on ML for Engineering Modeling, Simulation and Design. https://arxiv.org/abs/2012.11543
- Tian, Y., Xu, J., Li, Y., Luo, J., Sueda, S., Li, H., Willis, K. D. D., Matusik, W. Assemble Them All: Physics-Based Planning for Generalizable Assembly by Disassembly. SIGGRAPH Asia 2022 (ACM TOG). https://arxiv.org/abs/2211.03977
- Vignac, C., Krawczuk, I., Siraudin, A., Wang, B., Cevher, V., Frossard, P. DiGress: Discrete Denoising Diffusion for Graph Generation. ICLR 2023. https://arxiv.org/abs/2209.14734
- Vignac, C., Osman, N., Toni, L., Frossard, P. MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation. ECML PKDD 2023. https://arxiv.org/abs/2302.09048
- Xu, H. et al. LegoACE: Autoregressive Construction Engine for Expressive LEGO Assemblies. SIGGRAPH Asia 2025. https://doi.org/10.1145/3757377.3763881
- You, J., Ying, R., Ren, X., Hamilton, W., Leskovec, J. GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models. ICML 2018. https://arxiv.org/abs/1802.08773
