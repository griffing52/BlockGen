# Attention as an adjacency matrix: literature and plan

Research memo, 2026-10-06. Question: can the learned self-attention of an LLM or
transformer be used as the adjacency matrix of a generated structure, so that a
Minecraft or LEGO generator produces a native graph without training a graph model?

## Summary

Attention does carry relational structure, and the literature has measured it for a
decade. But two results already in this repo bound the cheap versions of the idea.
T28 (`results.md`, `notes.md` §28) read raw attention from Qwen2.5-Coder-1.5B over
serialized houses: the best head reached AUC 0.874 for 6-adjacency, a trained bilinear
probe 0.914, and the L1 distance between the coordinates printed in the text reached
0.980 with no model. T29 injected a geometric bias into attention and was null to
mildly negative. The repo also already has a pointer-style generator (`pick_n_place`,
T24) whose placer is attention over open faces.

The conclusion of this memo: a zero-shot read-out is either trivial (the structure is
printed in the serialization, so arithmetic beats the model) or impossible (the
structure is not in the input). The only regime with something new to say is the one
where an attention head is *trained to make the edge decision during generation*, and
the serialization is coordinate-free, so that the head is the sole carrier of
structure. That version is a credible paper, but its mechanism has strong precedents
(PolyGen, Pointer Networks, LISA), and the contribution must be framed as a
connection-native serialization for assemblies, not as a free adjacency hack.

## 1. Evidence that attention encodes structure

**Syntax in heads.** Raganato and Tiedemann (2018) and Clark et al. (2019) found
individual BERT and MT encoder heads that attend to syntactic relations (direct objects,
determiners, coreference) well above positional baselines, and Clark et al. showed that a
small combination of heads parses at a respectable level. Htut et al. (2019) extracted
dependency trees from every head (max attention and maximum spanning tree) and found
*specialist* heads for particular relation types but *no generalist head* that beats a
right-branching baseline by much. Kim et al. (2020) induced constituency trees from
attention distances without training.

**Probes beat raw attention.** Hewitt and Manning (2019) showed that a linear
transformation of hidden states recovers parse-tree distances, so syntax is encoded in
the representation geometry even where no single head displays it. This "the probe on
hidden states beats the attention map" pattern is the most replicated finding in the
area, and T28 reproduced it on voxel adjacency (0.914 probe vs 0.884 head).

**Attention flow.** Abnar and Zuidema (2020) showed single-layer attention is a poor
proxy for routing because residual streams mix tokens, and proposed rollout and flow,
which any read-out should also report.

**Caveats.** Jain and Wallace (2019) showed attention weights often disagree with
gradient attributions and can be permuted without changing predictions; Wiegreffe and
Pinter (2019) answered that attention can be a faithful explanation under some
conditions, but not by default. For our purposes the lesson is narrower: an attention
pattern correlating with adjacency does not mean the model uses it as adjacency.

**Transformers as GNNs.** Joshi (2020; technical version 2025) frames self-attention as
message passing on a complete graph weighted by attention: the intuition behind the
question, but it describes soft, dense routing, not a sparse predicted graph.

**Graphs and relations out of LLM attention (2020 to 2026).**
- Wang, Liu and Song (2020), MAMA: beam search over the attention matrices of BERT and
  GPT-2 extracts (head, relation, tail) triples from text with no fine-tuning; the
  attention is used as a relational score, then mapped to a schema.
- Schwaller et al. (2021), RXNMapper: an ALBERT trained without supervision on reaction
  SMILES has a head whose attention *is* the atom mapping between reactants and products,
  read out directly at 99.4% full-reaction accuracy. This is the strongest existing case
  of an unsupervised attention head serving as a graph correspondence, and it is in a
  domain (chemistry tokens) closer to ours than natural language syntax.
- Wang et al. (2024), ALPINE: on path-finding, a transformer provably and empirically
  stores the adjacency matrix (and a partial reachability matrix) in its weights.
- Guan et al. (2025): LLMs reading serialized graphs attend to node text but fail to
  model inter-node relations; attention over nodes does not match structural patterns.
- Liu et al. (2026), SLASH: LLMs show a "sawtooth" attention pattern that aligns with
  the token-level adjacency of a serialized graph, and attention sinks suppress it.
- Loveland et al. (2026, KDD): RoPE turns graph linearization into distance-dependent
  attention decay, so graph neighbours far apart in the sequence are under-attended;
  they propose an inference-time bias toward graph neighbours.

Net reading: pretrained attention reflects structure *explicit in the text* (copying,
matching identifiers) and is distorted by position (recency, RoPE decay, sinks), which is
T28's diagnosis: the best head matched coordinate strings, it did not model 3D.

## 2. Attention or pairwise scores used as edges in a generator

**Pointer mechanisms.** Pointer Networks (Vinyals et al., 2015) use the attention
distribution over input positions as the output. PolyGen (Nash et al., 2020) generates
meshes with a transformer whose face model emits *pointers to previously generated
vertices*: each face index is an attention distribution over vertex embeddings. This is
precisely "attention is the incidence structure" in an autoregressive 3D generator, and
it is the closest mechanism-level prior art. Pointer Graph Networks (Velickovic et al.,
2020) let each node point to another node and supervise the pointers to build a dynamic
graph.

**Supervising a head to be a graph.** LISA (Strubell et al., 2018) trains one attention
head to attend to each token's syntactic parent, so the head's attention is a dependency
tree, and uses it downstream. This is option (c) below, already done for syntax.

**Bilinear edge heads.** The deep biaffine parser (Dozat and Manning, 2017) scores every
(dependent, head) pair with a bilinear form on encoder states, which is a trained
attention-like map read as a tree. DETR-family scene-graph models score relations
between object queries: Relationformer (Shit et al., 2022) adds a relation token and
predicts edges between object-token pairs; RelTR (Cong et al., 2023) uses coupled
subject and object queries. These are set predictors with an explicit pairwise head.

**Graph generators.** GraphRNN (You et al., 2018) emits each node's adjacency vector to
previous nodes autoregressively; GRAN (Liao et al., 2019) generates blocks of adjacency
rows with attention-based GNNs; DiGress (Vignac et al., 2023) runs discrete diffusion over
node and edge types. Recent work makes the plain decoder-only transformer the graph
generator by serializing: G2PT (Chen et al., 2025) emits a node set then an edge set of
(index, index) tokens; AutoGraph (Chen, Krimmel and Borgwardt, 2025) flattens graphs into
sequences whose prefixes are induced subgraphs, with length linear in edges. In these the
edge is a *token that names a node index*, not an attention read-out, which costs
sequence length and forces the model to learn index arithmetic.

**Injection.** Graphormer (Ying et al., 2021) and TokenGT (Kim et al., 2022) put known
structure into attention; they consume graphs rather than generate them.

## 3. Structures, LEGO and Minecraft

- Thompson et al. (2020): LEGO assemblies as graphs (bricks as nodes, connections as
  edges) generated with a deep generative model of graphs. The direct ancestor of the
  "native graph generation" framing for LEGO.
- Chung et al. (2021), Brick-by-Brick: sequential brick assembly by RL with an action
  validity network; connection rules define the action space.
- Huang et al. (2020): part assembly via dynamic graph learning (relation reasoning over
  a part graph, pose prediction).
- LegoGPT / BrickGPT (Pun et al., 2025, ICCV): fine-tunes LLaMA-3.2-1B on text lines of
  brick dimensions and *absolute* (x,y,z) coordinates; validity check and physics-based
  rollback give stability. Connectivity is implicit in coordinates.
- LegoACE (Xu et al., 2025, SIGGRAPH Asia): a 32-layer Llama-style transformer
  generating assemblies as sequences of placed bricks with image or text conditioning.
- Minecraft: 3D-Craft / VoxelCNN (Chen et al., 2019) predicts the next block in human
  build order; text2mc provides a large build corpus.

No published LEGO or Minecraft generator found here uses an attention head as the
connection decision; LegoGPT and LegoACE carry structure through coordinates, which T28
showed makes attention redundant.

## 4. What "hacking attention into an adjacency matrix" could mean

Mechanical facts that apply to all options:
- Softmax attention is row-stochastic: each row sums to one, so it cannot say "no edge"
  or "three edges" on its own. Fixes: a learned null key (or use the sink deliberately),
  sigmoid attention for multi-label rows, or a separate degree/stop decision.
- It is not symmetric. For an undirected graph, either symmetrize, or record each edge
  once at its later endpoint (natural under a causal mask).
- The causal mask makes it lower-triangular. In growth order this is a feature: every
  new node attaches to at least one earlier node, so "edges to the past" covers the
  whole graph if rows are multi-label.
- Heads mix position and content. Recency (raster order makes `-(i-j)` score 0.831 in
  T28), RoPE decay and sinks all produce attention that looks structural. Every number
  needs a recency baseline and distance-stratified AUC.
- A node spans several tokens (4 in the AR2 `(X,Y,Z,PIECE)` grammar, more in LLM text),
  so key mass must be summed per node and query pooled; one-token-per-node
  serializations avoid this.

Ranked by feasibility (most feasible first), with what is already known here:

**(b) Light probe on frozen hidden states.** A bilinear or MLP head on layer-l states,
Hewitt-Manning style. Buys: the most information for the least training; tells you
where in the stack structure lives. Failure: it is an analysis tool, not a generator;
T28 found the signal peaks mid-stack (L14, 0.914) and decays by the output layer (0.807).
Measure: AUROC, average precision (positives are about 3% of candidate pairs, so AP is
the more honest number), distance-stratified AUC. **Done (T28): beaten by printed-coordinate
arithmetic.**

**(a) Zero-shot read-out of raw attention from a frozen LLM.** Buys: nothing to train.
Failure: the dilemma above. If the serialization prints coordinates, a subtraction wins
(0.980); if it does not, the structure is absent from the input and no head can recover
it. Measure: per-head AUROC and AP, head chosen on a train split, plus recency, rollout,
and coordinate baselines. **Done (T28): negative.** Only worth re-running on the repo's
own trained models (see plan step 1), not on more LLMs.

**(d) Pointer-style generation, attention argmax is the edge.** Each new node token's
query attends over previous node keys; the distribution (with masking) is sampled as the
attachment. Pose is derived from parent pose plus connection type, so no coordinates are
emitted. Buys: graph generation that is native by construction; sequence length N
rather than 4N (AR2) or N+E (G2PT); validity by construction, since illegal attachments
(occupied faces, collisions, unstable studs) can be masked to minus infinity before the
softmax. Failure: one parent per step is a tree; extra edges (cycles, a brick on two
others) need a multi-label sigmoid head or are recovered geometrically after placement
(for voxels and studs, all contacts follow from poses, so only the spanning tree needs
predicting). Exposure bias: teacher-forced pointer accuracy says little (T24b). **Partly
done: `pick_n_place` (T24) is this with a separate encoder and pointer head; placer val
accuracy 0.713 vs 0.0085 chance, perfectly connected outputs, but size-capped at 185
blocks and worst on KID.**

**(c) Fine-tune a transformer so a designated head equals the adjacency.** LISA-style
auxiliary cross-entropy (or BCE for sigmoid rows) on one head's attention logits. Buys:
the merge of (d) with a pretrained LLM, so the head that routes information is also the
head that decides edges, and the rest of the model is free to use it. Failure: the
supervised head may become a pure output head that the residual stream ignores (check
by ablating it); pretrained position bias fights the target; fused attention kernels do
not return logits, so the head needs a custom path (FlexAttention `score_mod` or a
separate q/k projection on that head). Measure: parent top-1 and AP on held-out builds,
then free-running validity and quality.

Practical ranking for generation: (d) plus (c) together is the only version that
produces a generator; (a) and (b) are diagnostics.

## 5. Experiment plan for this repo, cheapest first

**Step 0 (done).** T28 (zero-shot and probe, Qwen2.5-Coder-1.5B, Track D serialization),
T29 (geometric bias injection), T24 (`pick_n_place` pointer placer). Do not repeat these.

**Step 1: probe our own voxel AR transformers (hours, no training).** Run the T28
protocol on `VoxelTransformerAR2` checkpoints in `outputs/` (for example the
`native_oriented` and `canon_*_native` runs), with and without the geometric bias. Query
on each block's PIECE (or Z) token, keys summed over the four tokens of each earlier
block. Metrics: per-head AUROC and AP for 6-adjacency against previous blocks,
distance-stratified AUC, attention rollout, and the same three baselines (recency,
coordinate-token match, coordinate L1). Question answered: does a model trained only on
blocks grow a geometric head that beats coordinate matching? A positive here would be the
first evidence that attention learned adjacency rather than copied it. Memory is trivial
(small models), and `output_attentions` is not available on the SDPA path, so recompute
q and k per layer with hooks.

**Step 2: coordinate-free serialization study (one to two days).** Write a serializer
emitting `(piece, connection_type)` per node in BFS growth order (T24's measurement:
under BFS every parent lies within the last 512 nodes, so a windowed key set loses
nothing). Train small causal transformers, three arms of equal size: (i) attention argmax
of an unsupervised designated head, (ii) LISA-supervised head, (iii) separate Dozat
bilinear head on final states (the `pick_n_place` style control). Metrics: parent top-1,
AP, free-running connectivity and floating fraction, block count distribution, `geom_kid`
and BlockScore at n of at least 256 against `houses32-v1`. The hypothesis worth a paper is
(ii) matching (iii) at lower cost and with the edge head inside the routing.

**Step 3: the LLM version (two to four days, about 6 GB).** LoRA on Qwen2.5-1.5B (bf16
weights about 3.1 GB, gradient checkpointing, sequence at most 2048) with the step-2
serialization rendered as text, one head in a mid layer (L14 was the best layer in T28)
supervised as the attach pointer, implemented via an extra q/k projection reading that
layer's input so the fused kernel stays intact. Compare against Track D (coordinates
printed) on the same split: parse rate, connectivity, `geom_kid`, memorization rate. The
question is whether pretraining helps a pointer head at all, given T28's evidence that
pretrained heads are position- and copy-driven. Gemma-3-4B will not train in 6 GB.

**Step 4: LEGO (about a week).** Convert StableText2Lego (LegoGPT data) to brick graphs
where connection type is (stud offset, rotation). Mask the pointer by collision and by
LegoGPT-style stability analysis. Metrics: valid-and-stable rate without rollback, number
of rollbacks needed, and LegoGPT's own evaluation for comparability.

## 6. Verdict

**Novelty: moderate, and only in the strong form.** "Read the adjacency out of a frozen
LLM" is neither new (Clark et al., Htut et al., MAMA, RXNMapper) nor promising here
(T28 measured it negative for a structural reason). "Make one attention head the edge
decoder of a generator" is mechanistically PolyGen plus LISA. What is genuinely open is
the combination this repo is positioned to test: a **coordinate-free, connection-native
assembly language in which a pretrained LLM's own supervised attention head is the
attachment pointer, pose is derived from typed connections, and validity (collision,
stability) is enforced by masking that head's keys**. That gives graph generation with no
GNN, no edge tokens, sequence length N, and validity by construction, and it directly
answers why LegoGPT needs rollback (its coordinates let it propose unsupported bricks).
It is `ideas.md` idea 8 (typed-connection AR) with the pointer inside the LLM.

**The paper claim that would survive review:** on Minecraft houses and LEGO, an
attention-pointer serialization matches or beats coordinate serializations (Track D,
LegoGPT format) on quality and validity at a fraction of the sequence length, and the
pointer can live inside a pretrained LLM head without an extra network. The ablation
that makes it a finding rather than a system: supervised head versus separate bilinear
head versus coordinates, with free-running metrics.

**Closest prior work a reviewer would cite against it:**
1. PolyGen (Nash et al., 2020): autoregressive transformer whose outputs are pointers to
   earlier generated elements in a 3D object.
2. LISA (Strubell et al., 2018): one attention head supervised to equal a parse tree.
3. LegoGPT (Pun et al., 2025) and LegoACE (Xu et al., 2025): LLM-style autoregressive
   brick generation, the domain baselines.
Also likely: G2PT and AutoGraph (graphs as transformer sequences), Pointer Networks, and
Thompson et al. (LEGO as graph generation).

## References

- Abnar, S., Zuidema, W. Quantifying Attention Flow in Transformers. ACL 2020. https://arxiv.org/abs/2005.00928
- Carion, N. et al. End-to-End Object Detection with Transformers (DETR). ECCV 2020. https://arxiv.org/abs/2005.12872
- Chen, D., Krimmel, M., Borgwardt, K. Flatten Graphs as Sequences: Transformers are Scalable Graph Generators (AutoGraph). NeurIPS 2025. https://arxiv.org/abs/2502.02216
- Chen, X., Wang, Y., He, J., Du, Y., Hassoun, S., Xu, X., Liu, L.-P. Graph Generative Pre-trained Transformer (G2PT). ICML 2025. https://arxiv.org/abs/2501.01073
- Chen, Z. et al. Order-Aware Generative Modeling Using the 3D-Craft Dataset. ICCV 2019. https://github.com/facebookresearch/voxelcnn
- Chung, H., Kim, J., Knyazev, B., Lee, J., Taylor, G. W., Park, J., Cho, M. Brick-by-Brick: Combinatorial Construction with Deep Reinforcement Learning. NeurIPS 2021. https://arxiv.org/abs/2110.15481
- Clark, K., Khandelwal, U., Levy, O., Manning, C. D. What Does BERT Look At? An Analysis of BERT's Attention. BlackboxNLP 2019. https://arxiv.org/abs/1906.04341
- Cong, Y., Yang, M. Y., Rosenhahn, B. RelTR: Relation Transformer for Scene Graph Generation. IEEE TPAMI 2023. https://arxiv.org/abs/2201.11460
- Dozat, T., Manning, C. D. Deep Biaffine Attention for Neural Dependency Parsing. ICLR 2017. https://arxiv.org/abs/1611.01734
- Guan, Z., Wu, L., Zhao, H., He, M., Fan, J. Attention Mechanisms Perspective: Exploring LLM Processing of Graph-Structured Data. ICML 2025. https://arxiv.org/abs/2505.02130
- Hewitt, J., Manning, C. D. A Structural Probe for Finding Syntax in Word Representations. NAACL 2019. https://aclanthology.org/N19-1419/
- Htut, P. M., Phang, J., Bordia, S., Bowman, S. R. Do Attention Heads in BERT Track Syntactic Dependencies? arXiv 2019. https://arxiv.org/abs/1911.12246
- Huang, J., Zhan, G., Fan, Q., Mo, K., Shao, L., Chen, B., Guibas, L., Dong, H. Generative 3D Part Assembly via Dynamic Graph Learning. NeurIPS 2020. https://arxiv.org/abs/2006.07793
- Jain, S., Wallace, B. C. Attention is not Explanation. NAACL 2019. https://arxiv.org/abs/1902.10186
- Joshi, C. K. Transformers are Graph Neural Networks. The Gradient, 2020; technical version arXiv 2025. https://arxiv.org/abs/2506.22084
- Kim, J., Nguyen, T. D., Min, S., Cho, S., Lee, M., Lee, H., Hong, S. Pure Transformers are Powerful Graph Learners (TokenGT). NeurIPS 2022. https://arxiv.org/abs/2207.02505
- Kim, T., Choi, J., Edmiston, D., Lee, S. Are Pre-trained Language Models Aware of Phrases? Simple but Strong Baselines for Grammar Induction. ICLR 2020. https://arxiv.org/abs/2002.00737
- Liao, R. et al. Efficient Graph Generation with Graph Recurrent Attention Networks (GRAN). NeurIPS 2019. https://arxiv.org/abs/1910.00760
- Liu, Y., Lu, B., Wang, X., Zhou, C., Jin, M. SLASH the Sink: Sharpening Structural Attention Inside LLMs. arXiv 2026. https://arxiv.org/abs/2605.10503
- Loveland, D., Trivedi, P., Weinstein, A., Huang, E. W., Koutra, D. Formalizing and Mitigating Structural Distortion in LLM Attention for Graph Reasoning. KDD 2026. https://arxiv.org/abs/2606.15633
- Nash, C., Ganin, Y., Eslami, S. M. A., Battaglia, P. PolyGen: An Autoregressive Generative Model of 3D Meshes. ICML 2020. https://arxiv.org/abs/2002.10880
- Pun, A., Deng, K., Liu, R., Ramanan, D., Liu, C., Zhu, J.-Y. Generating Physically Stable and Buildable Brick Structures from Text (LegoGPT / BrickGPT). ICCV 2025. https://arxiv.org/abs/2505.05469
- Raganato, A., Tiedemann, J. An Analysis of Encoder Representations in Transformer-Based Machine Translation. BlackboxNLP 2018. https://aclanthology.org/W18-5431/
- Schwaller, P., Hoover, B., Reymond, J.-L., Strobelt, H., Laino, T. Extraction of Organic Chemistry Grammar from Unsupervised Learning of Chemical Reactions (RXNMapper). Science Advances 7(15), 2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC8026122
- Shit, S., Koner, R., Wittmann, B. et al. Relationformer: A Unified Framework for Image-to-Graph Generation. ECCV 2022. https://arxiv.org/abs/2203.10202
- Strubell, E., Verga, P., Andor, D., Weiss, D., McCallum, A. Linguistically-Informed Self-Attention for Semantic Role Labeling (LISA). EMNLP 2018. https://aclanthology.org/D18-1548/
- Thompson, R., Ghalebi, E., DeVries, T., Taylor, G. W. Building LEGO Using Deep Generative Models of Graphs. NeurIPS 2020 Workshop on ML for Engineering Modeling, Simulation and Design. https://arxiv.org/abs/2012.11543
- Velickovic, P., Buesing, L., Overlan, M., Pascanu, R., Vinyals, O., Blundell, C. Pointer Graph Networks. NeurIPS 2020. https://arxiv.org/abs/2006.06380
- Vignac, C., Krawczuk, I., Siraudin, A., Wang, B., Cevher, V., Frossard, P. DiGress: Discrete Denoising Diffusion for Graph Generation. ICLR 2023. https://arxiv.org/abs/2209.14734
- Vinyals, O., Fortunato, M., Jaitly, N. Pointer Networks. NeurIPS 2015. https://arxiv.org/abs/1506.03134
- Wang, C., Liu, X., Song, D. Language Models are Open Knowledge Graphs (MAMA). arXiv 2020. https://arxiv.org/abs/2010.11967
- Wang, S. et al. ALPINE: Unveiling the Planning Capability of Autoregressive Learning in Language Models. NeurIPS 2024. https://arxiv.org/abs/2405.09220
- Wiegreffe, S., Pinter, Y. Attention is not not Explanation. EMNLP-IJCNLP 2019. https://arxiv.org/abs/1908.04626
- Xu, H. et al. LegoACE: Autoregressive Construction Engine for Expressive LEGO Assemblies. SIGGRAPH Asia 2025. https://doi.org/10.1145/3757377.3763881
- Ying, C. et al. Do Transformers Really Perform Bad for Graph Representation? (Graphormer). NeurIPS 2021. https://arxiv.org/abs/2106.05234
- You, J., Ying, R., Ren, X., Hamilton, W., Leskovec, J. GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models. ICML 2018. https://arxiv.org/abs/1802.08773
