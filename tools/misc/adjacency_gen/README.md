# adjacency_gen: nodes in, directed adjacency out

Prototype of "a model whose input is a list of nodes and whose output is an adjacency
matrix", on small `houses_32` chunks, with two node vocabularies: single blocks
(voxel nodes) and 3D-BPE pieces (piece nodes). Background: results.md T24, T28, T49,
and the literature memo `RESEARCH.md` in this folder.

## What the model sees and outputs

* **Input**: a multiset of node types. No coordinates and no positional encoding. Per
  node: a type embedding, two order-free set statistics (log of the number of
  same-type nodes in the set, log of the set size), and an 8-d random node feature
  (RNF, drawn fresh on every forward pass). RNF is required: without it every node of a
  type gets the identical embedding, so an all-masked set produces identical rows and
  cannot express any asymmetric graph (RESEARCH.md, correction 2). The encoder is
  equivariant in distribution; with RNF switched off it is exactly equivariant, which
  is what the test checks.
* **Output**: for every ordered pair, a label in `{none} + offsets`. The offset is
  `anchor_j - anchor_i` in a cube `[-B, B]^3`. For voxels `B = 1`, and a shape-validity
  mask leaves exactly the 6 face directions, so this is the 7-class label from the
  brief. For pieces `B = 4` (728 offsets), masked per pair to the offsets at which the
  two piece shapes touch without overlapping. Piece shape is a property of the type,
  so this mask leaks nothing positional. 0.5% of piece contacts (5,900 of 1.21M) have
  an offset outside the cube and are labelled "none".
* **Antisymmetry is exact**: raw scores `f(i,j,k)` are tied as
  `z(i,j,k) = f(i,j,k) + f(j,i,opp(k))`, so label(i,j) = +x iff label(j,i) = -x.
* **Conditioning on observed edges**: each pair has a state code (masked, observed
  none, observed offset, self). Its embedding (factorized over state, dx, dy, dz)
  enters every layer as a per-head attention bias and as a relation value added to the
  attention output (Graphormer bias plus Shaw-style relative values). The pair head is
  an MLP on `A h_i + B h_j + rel_ij`. Size: 6 layers, d = 192, about 3M parameters.
* **Training**: masked edge modelling. Per graph, the mask rate is 1.0 with
  probability 0.2, otherwise U(0.05, 1). Cross-entropy is computed on masked pairs.

## Decoding (decode.py)

MaskGIT-style, with one deviation from the brief. **Committing pairs by plain argmax
does not work**: a given pair is an edge with probability of about 3/N, so "none"
wins every pair and nothing connects (seen in the first smoke run). The decoder
instead ranks each undecided pair by its best *offset* label and commits the most
confident ones on a cosine-like schedule (24 forward passes).

* **With the check** (`model_check`): nodes live in rigid components. An edge that
  would make two nodes share a cell is banned for that pair, and the pair is
  re-scored. A merge fixes every pair across the two components by geometry, so
  voxel closure holds by construction. Decoding stops when everything is one
  component (training chunks are connected by construction).
* **Without the check** (`model_nocheck`): the decoder commits the model's expected
  edge count, `round(sum P(edge))` (at least N - 1), then places nodes by BFS
  over a confidence-ordered spanning forest. Colliding nodes are dropped and counted.
* `model_check_t1` uses temperature-1 sampling of labels plus Gumbel-noised commit
  order. `model_grow` adds the check and, after the first edge, only allows pairs with
  one end in the grown component. This is the RESEARCH.md recommendation (an
  inventory-conditioned spanning tree) run on the same model, without retraining.

## Data

`houses_32` with the grouped split `houses_32.s0.70-15-15.v1` (train builds give train
chunks, test builds give test chunks, so no build leaks).

| variant | window | nodes per chunk (median) | train chunks | test chunks |
|---|---|---|---|---|
| voxel | 6^3, largest component, 20 to 80 blocks | 58 | 34,864 | 382 |
| piece | 8^3 re-tokenized with the houses_32 BPE merges, largest touching component, 12 to 64 pieces | 45 (about 94 voxels) | 15,109 | 263 |

Piece windows are larger in space, so the two variants are compared on rates and on
distance to their own real chunks, not head to head on raw numbers.

## Results (single seed, 35 min of training each, n = 200 test multisets for generation)

Runs: `outputs/run_20261007_111458_adjacency_gen_voxel`,
`outputs/run_20261007_112531_adjacency_gen_piece`,
`outputs/run_20261007_115713_adjacency_gen_voxel_prefix`. Each has `results.json`,
`samples.png` (one row per arm, with the same 8 held-out multisets in every column),
and `results_grow.json` / `samples_grow.png` for the grow decoder.

### (a) Masked edge prediction, one forward pass

Columns: edge-existence F1 at argmax, average precision of P(edge), direction
accuracy on true edges predicted as edges. The type-pair prior's AP is the
block-type-only baseline.

| mask rate | voxel F1 | voxel AP | voxel dir | piece F1 | piece AP | piece dir | prior AP (vox / piece) |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.848 | 0.928 | 0.988 | 0.686 | 0.806 | 0.893 | 0.112 / 0.151 |
| 0.50 | 0.427 | 0.560 | 0.967 | 0.336 | 0.503 | 0.847 | 0.111 / 0.152 |
| 0.75 | 0.063 | 0.230 | 0.929 | 0.092 | 0.281 | 0.762 | 0.110 / 0.152 |
| 1.00 | 0.004 | 0.152 | 0.359 | 0.026 | 0.207 | 0.582 | 0.111 / 0.152 |

Edge base rates are 0.056 (voxel) and 0.074 (piece). When some edges are observed, the
model infers the missing ones from geometry very well. At full mask it is only
slightly above the type prior, which is the expected multimodality: at full mask, no
fixed answer exists for a single pair.

### (b) Generation from a held-out multiset, all edges masked

Realizability is the fraction of nodes placed without collision. Connectivity is the
largest-component fraction. Edge F1 is by node identity; same-type nodes are
exchangeable and no matching is done, so it is a lower bound. Typed-edge J is the
Jaccard of `(type_i, type_j, offset)` edge multisets, the permutation-invariant
version. Aligned IoU is the best voxel IoU against the original over 4 yaws and small
shifts. All three are weak under multimodality: a perfect sampler still scores low
against the one real chunk. W1 is the 1-D Wasserstein distance of the realized
largest component's stat to the real chunks (lower is better).

Voxel nodes (real: fill 0.428, support 0.761, height 5.0):

| arm | realiz. | conn. | coll./graph | edge F1 | typed-edge J | IoU | fill | support | W1 fill | W1 support | W1 height |
|---|---|---|---|---|---|---|---|---|---|---|---|
| model_check | 1.00 | 1.00 | 0 | 0.116 | 0.204 | 0.322 | 0.478 | 0.809 | 0.092 | 0.048 | 1.23 |
| model_check_t1 | 1.00 | 1.00 | 0 | 0.094 | 0.217 | 0.326 | 0.350 | 0.703 | 0.078 | 0.065 | 0.86 |
| model_grow | 1.00 | 1.00 | 0 | 0.115 | 0.204 | 0.393 | 0.750 | 0.929 | 0.322 | 0.168 | 0.68 |
| model_nocheck | 0.62 | 0.69 | 23.4 | 0.055 | 0.077 | 0.230 | (0.452) | (0.769) | (0.069) | (0.023) | (1.45) |
| prior_check | 1.00 | 1.00 | 0 | 0.076 | 0.187 | 0.239 | 0.162 | 0.515 | 0.266 | 0.248 | 1.65 |
| random_tree | 0.56 | 1.00 | 26.3 | 0.045 | 0.110 | 0.265 | 0.266 | 0.585 | 0.163 | 0.185 | 0.64 |
| random_tree_check | 1.00 | 1.00 | 0 | 0.061 | 0.159 | 0.294 | 0.230 | 0.597 | 0.198 | 0.178 | 1.40 |
| retrieval (nearest train multiset) | 1.00 | 1.00 | 0 | n/a | 0.300 | 0.385 | 0.456 | 0.744 | 0.059 | 0.039 | 0.66 |

Piece nodes (real: fill 0.385, support 0.741, height 6.4):

| arm | realiz. | conn. | coll./graph | edge F1 | typed-edge J | IoU | fill | support | W1 fill | W1 support | W1 height |
|---|---|---|---|---|---|---|---|---|---|---|---|
| model_check | 1.00 | 1.00 | 0 | 0.169 | 0.130 | 0.290 | 0.275 | 0.693 | 0.110 | 0.047 | 0.71 |
| model_check_t1 | 1.00 | 1.00 | 0 | 0.136 | 0.112 | 0.255 | 0.230 | 0.627 | 0.156 | 0.114 | 0.98 |
| model_grow | 1.00 | 1.00 | 0 | 0.170 | 0.123 | 0.331 | 0.353 | 0.745 | **0.037** | **0.012** | 0.90 |
| model_nocheck | 0.66 | 0.74 | 16.8 | 0.098 | 0.060 | 0.248 | (0.295) | (0.684) | (0.091) | (0.061) | (1.49) |
| prior_check | 1.00 | 1.00 | 0 | 0.106 | 0.100 | 0.192 | 0.127 | 0.475 | 0.258 | 0.266 | 2.17 |
| random_tree | 0.60 | 1.00 | 19.3 | 0.062 | 0.039 | 0.216 | 0.203 | 0.530 | 0.182 | 0.212 | 0.92 |
| random_tree_check | 1.00 | 1.00 | 0 | 0.090 | 0.055 | 0.224 | 0.171 | 0.539 | 0.214 | 0.202 | 1.30 |
| retrieval | 1.00 | 1.00 | 0 | n/a | 0.111 | 0.270 | 0.471 | 0.757 | 0.093 | 0.059 | 1.71 |

Parenthesised stats are on the surviving largest conflict-free component, which holds
only about 30% of the voxels (17 of 58, 42 of 94), so they are not comparable.
Retrieval returns a *different* multiset (mean L1 distance 25 voxel nodes, 35 piece
nodes), real geometry included, so its shape stats are those of real data by
construction.

**With versus without the realizability check.** Without it, 34 to 39% of nodes
collide, only 60% of committed direction edges hold after BFS placement (`edge_ok`
0.61 / 0.60), and the largest component is about 70% of the set. With it, collisions
are 0 by construction, but the greedy decoder hits many bans (272 rejected edges per
voxel graph at temperature 0, 19 at temperature 1). The model's one-shot proposals are
locally plausible and globally inconsistent. The check is not an optional repair;
it is the decoder.

**Implied-edge consistency.** Among face-touching pairs in the realized voxel build,
the model's best offset label matched the realized direction 89% of the time
(`implied_dir`). But it gave the pair "edge" over "none" only about 3% of the time
(`implied_consistency`). Closure comes from the check, not from the model.

**Prefix-mask training hurts (`voxel_prefix`).** Training half the batches on
growth-state masks (a connected prefix observed, the rest masked) made every
generation arm worse. W1 fill went from 0.092 to 0.184 for check decoding and from
0.322 to 0.234 for grow, and the samples became long one-block filaments, despite 22k
training steps against 13k (the GPU was less contended).

### What the samples look like

* Voxel `model_check`: coherent fragments with the right material roles. One sample
  is a brick wall under a planks roof slab, one a dark-oak base under a snow layer, one
  a pink/green terrace on a stone pillar. Some samples are a single solid box, and a few
  have stray one-block arms. These samples look clearly more like houses than the
  random trees (lumpy blobs) or the type prior (sprawling, branching tangles).
* Voxel `model_grow`: compact, solid cuboids with plausible vertical material layering
  (snow on top, grass over cobble, a planks wall with a door). The cuboids are too dense
  (fill 0.75 against 0.43).
* Piece `model_check`: recognisable parts (a pink beam with a green end post, grass
  platforms with dark-oak posts), but long plank rods often stick out at odd angles.
  Piece `model_grow` is the best looking arm. It produced a brick wall with a planks
  roof overhang, a grass ledge with a dark-oak door frame, and a stone-and-planks
  corner. Its fill and support match the real chunks (W1 0.037 / 0.012), but some
  rods are still too long.
* `samples.png` in each run folder shows the same 8 test multisets for every arm.

## Honest read

* Masked edge infilling works well. With a quarter of the pairs hidden, voxel edge F1
  is 0.85 and direction accuracy is 0.99. Generating from nothing but the multiset is
  a different problem. At full mask the per-pair posterior is close to the type prior
  (AP 0.15 against 0.11), so all the structure has to come from the iterative decode
  plus the realizability check.
* With the check, the model clearly beats both the random-tree and the type-prior
  baselines on every shape statistic, for both node types. Retrieval of the nearest
  training multiset is still as good or better on voxels (it returns real
  chunks). On pieces, grow decoding beats retrieval on fill and support.
* Piece nodes did not make the masked task easier (lower AP and direction accuracy,
  728 labels per pair instead of 6). In generation, piece grow decoding is the best
  arm. That supports RESEARCH.md: a coarser vocabulary plus spanning-tree growth.
* Caveats: one seed, 35 minutes of training (the masked AP was still rising), n = 200,
  and chunk-scale only (6^3 or 8^3).

## Next

1. Train for the growth decoder the right way: a proper inventory-conditioned
   autoregressive spanning tree (pick from remaining inventory plus attach face, with
   the cell-marginalized loss of T24e). The cheap prefix-mask version above was
   negative.
2. Drop full-N^2 output: predict edges only between candidate pairs that the current
   partial geometry allows. That is the redundancy RESEARCH.md calls correction 1.
3. Seeds, longer training, and the benchmark's `geom_kid` on realized chunks, with
   an inventory-matched novelty check against retrieval.

## Commands

    .venv/bin/python -m tools.misc.adjacency_gen run --kind voxel --minutes 35
    .venv/bin/python -m tools.misc.adjacency_gen run --kind piece --minutes 35 --max-gb 2.5
    .venv/bin/python -m tools.misc.adjacency_gen run --kind voxel --name voxel_prefix --prefix-frac 0.5 \
        --data-from outputs/run_20261007_111458_adjacency_gen_voxel
    .venv/bin/python -m tools.misc.adjacency_gen eval --run <run dir> --arms model_grow --tag _grow
    .venv/bin/python -m tools.misc.adjacency_gen run --kind voxel --smoke      # about 3 minutes
    .venv/bin/python -m pytest -q tests/misc/adjacency_gen

The piece run reused a dataset built separately
(`outputs/run_20261007_111628_adjacency_gen_piece_data/dataset.pkl`, about 2 minutes on
6 workers) through `--data-from`. Without it, `run --kind piece` builds the dataset
itself. The GPU is capped by `--max-gb` (via `set_per_process_memory_fraction`), and
the code falls back to CPU when less than 5 GB is free.

Files: `data.py` (chunks, BPE re-tokenization, label space, validity table),
`model.py` (equivariant encoder and tied edge head), `train.py` (masking, loss,
masked metrics), `decode.py` (MaskGIT decode, check, grow, BFS realization,
random-tree and type-prior baselines), `eval.py` (generation metrics, retrieval, figure).

## Face nodes with coarse-shape conditioning (2026-10-07, second prototype)

T50 showed that a block-type multiset alone is too ambiguous: at full mask the pair
model barely beats a type prior and the collision check does the decoding. This
prototype changes the nodes to faces and adds a coarse shape as the condition.

### Formulation (faces.py, coarse.py, model_faces.py)

* **Nodes are faces.** Each block contributes 6 face tokens: block type, face direction
  and a random per-block identity vector (fresh every forward pass, shared by the 6 faces,
  so faces know their block and identical blocks are distinguishable).
* **Edges glue a face to a face.** The 6N x 6N symmetric face matrix has only three
  non-zero blocks, so the model outputs three N x N axis matrices,
  `X_a[i, j]` = "block i's +a face is glued to block j's -a face". Symmetry,
  complementarity (+x only with -x) and "never within one block" (masked diagonal) hold by
  construction; "at most one partner per face" is a row/column constraint on each `X_a`.
  An exposed/buried logit per face is the auxiliary output and doubles as the dust-bin
  score. The glue loss is a dual softmax with a dust bin (rows and columns of each `X_a`).
* **Coarse condition.** The 8^3 window is downsampled by `f`: per coarse cell, the block
  count and the dominant type. These are extra tokens with a Fourier 3D position (the same
  encoder as placed-block positions), in one joint transformer with the face tokens.
  `f = 2` gives a 4^3 grid and matches the canon16 decimation of the 32^3 houses (window
  origins are even); `f = 4` gives 2^3. With `residual` (the "_resid" runs), coarse tokens
  also see how many blocks the partial build already has in each cell.
* **Data.** 8^3 windows of houses_32, largest face-connected component, **40 to 120 blocks**
  (median 89), window origins on multiples of 2; grouped split v1 (18,871 train chunks,
  474 test chunks; generation on the first 200). Model: 6 layers, d = 192, 2.9M params,
  30 to 32 minutes per run, one seed.
* **The inventory (the block list) still comes from the real chunk**, as in T50. A full
  generator would need it from somewhere else, for example per-coarse-cell block counts from
  a coarse model.

### Decoders

* **(A) Grow from a seed (decode_faces.sequential_decode).** A seed head places one block
  at an absolute cell, then each step picks (open face of the placed build, unplaced block
  attached by its complementary face), which fixes the new block's cell. Placed blocks
  get their positions as features; unplaced ones do not. Check: the target cell must be
  empty and inside the window; every face touching the new cell is glued automatically.
  No check: a face is open until used by an explicit attachment, collisions are dropped.
  Training is teacher forcing on random valid attachment orders of real chunks with the
  loss marginalized over every correct (face, block) choice (all faces that point at a
  correct cell, T24e, and any unplaced block of the right type). Faces into occupied or
  out-of-window cells stay in the softmax as negatives, so "no check" tests what the
  model learned. `cmask` arms also use the condition as a hard mask (attach only into
  coarse cells still short of their count); `seq_random_*` arms replace the model by a
  uniform choice under the same masks.
* **(B) Merge (merge.py), the user's formulation.** All blocks start as singleton
  components; each step glues one complementary face pair across two components, which
  fixes their relative pose. With the check a merge whose rigid bodies overlap, or whose
  bounding box exceeds the window (`extent`), is rejected and the pair banned; contacts
  implied by the merge are glued automatically. Exactly N - 1 merges, no stop token.
  Features: random per-component vector, log component size, offset to the component
  anchor, face state within the component. Teacher forcing on random Kruskal orders of the
  real contact edges; the target is every true cross-component connection (same-type
  singletons exchangeable), loss marginalized by logsumexp. `merge_*_cmask` also rejects a
  merge after which the component cannot be translated into the window within the coarse
  counts.
* **(C) One-shot matching (decode_faces.oneshot_realize).** One pass with nothing placed;
  per axis a log-domain Sinkhorn with dust bin, Hungarian, keep the plan's expected number of
  glued faces; realize by Kruskal + collision check (or decode.realize without it). The MAP
  of the augmented problem leaves nearly every face in its bin, which is why the expected
  count is used.

Frame-free outputs (one-shot, merge, random tree, T50) are translated to the shift that best
matches the coarse condition before scoring; the grow decoder is scored where it put the
blocks. Baselines: naive coarse upsampling (each coarse cell gets its count of inventory
blocks, dominant type first, at random positions inside the cell), retrieval of the nearest
training chunk by block counts and by coarse grid, random tree + check, and the T50 voxel
model with its check decoder (trained on 6^3 windows with up to 80 blocks).

### Results (n = 200 held-out chunks, single seed)

Columns: coarse IoU with the condition, voxel IoU and typed F1 (cell and type) against the
real chunk, glued-face-pair F1 (typed multisets), W1 of fill and support to real chunks
(real fill 0.35, support 0.74). Grow and merge arms use `f = 2` unless noted.

| arm | coarse IoU | voxel IoU | typed F1 | glue F1 | W1 fill / support |
|---|---|---|---|---|---|
| grow, check (f = 2, resid) | 0.49 | 0.25 | 0.16 | 0.51 | 0.080 / 0.044 |
| grow, check, T = 1 | 0.52 | 0.27 | 0.17 | 0.52 | 0.068 / 0.058 |
| grow, no check (22 collisions/chunk, 76% placed) | 0.42 | 0.20 | 0.14 | 0.44 | 0.163 / 0.094 |
| grow, check + coarse mask | **0.97** | 0.48 | 0.24 | **0.54** | 0.034 / 0.034 |
| random policy, check | 0.22 | 0.10 | 0.05 | 0.39 | 0.045 / 0.054 |
| random policy, check + coarse mask | 0.96 | 0.45 | 0.18 | 0.38 | 0.030 / 0.071 |
| grow, check, f = 2 without resid feature | 0.39 | 0.21 | 0.13 | 0.50 | 0.077 / 0.065 |
| grow, check, f = 4 (resid) | 0.68 (f = 4 grid) | 0.19 | 0.11 | 0.49 | 0.072 / 0.065 |
| grow, check + coarse mask, f = 4 | 0.96 | 0.27 | 0.15 | 0.50 | 0.053 / 0.040 |
| grow, check, no conditioning | 0.26 | 0.12 | 0.07 | 0.46 | 0.080 / 0.065 |
| grow, check + f = 2 mask, no-cond model | 0.89 | 0.42 | 0.19 | 0.52 | 0.038 / 0.031 |
| grow, check, oracle cell index | 0.75 | 0.45 | 0.36 | **0.70** | 0.013 / 0.026 |
| grow, check + coarse mask, oracle | 0.99 | **0.55** | **0.40** | 0.69 | 0.010 / 0.009 |
| merge, check + extent (f = 2) | 0.37 | 0.17 | 0.09 | 0.44 | 0.057 / 0.065 |
| merge, check + extent, T = 1 | 0.35 | 0.15 | 0.08 | 0.45 | 0.091 / 0.182 |
| merge, overlap check only (no extent) | 0.22 | 0.04 | 0.01 | 0.33 | 0.213 / 0.157 |
| merge, random policy, check + extent | 0.30 | 0.11 | 0.06 | 0.33 | 0.164 / 0.211 |
| merge, no conditioning | 0.36 | 0.16 | 0.08 | 0.45 | 0.045 / 0.047 |
| merge, oracle cell index | 0.44 | 0.17 | 0.13 | 0.62 | 0.089 / 0.147 |
| merge, check + coarse mask (f = 2; 38% of blocks joined, 1% complete) | 0.52 | 0.21 | 0.11 | 0.27 | 0.168 / 0.094 |
| merge, check + coarse mask, oracle (47% joined) | 0.66 | 0.27 | 0.21 | 0.41 | 0.045 / 0.069 |
| one-shot Sinkhorn + check (largest comp. 14% of blocks) | 0.26 | 0.07 | 0.04 | 0.12 | 0.341 / 0.185 |
| one-shot, no check | 0.26 | 0.07 | 0.05 | 0.13 | 0.330 / 0.178 |
| naive coarse upsampling (f = 2) | 1.00 | 0.43 | **0.33** | 0.47 | 0.051 / 0.144 |
| naive coarse upsampling (f = 4) | 1.00 | 0.23 | 0.14 | 0.32 | 0.110 / 0.286 |
| retrieval, block counts | 0.26 | 0.14 | 0.09 | 0.42 | 0.058 / 0.037 |
| retrieval, coarse grid (f = 2) | 0.76 | **0.50** | 0.11 | 0.13 | 0.057 / 0.019 |
| random tree + check | 0.36 | 0.14 | 0.08 | 0.38 | 0.127 / 0.143 |
| T50 voxel model + check | 0.45 | 0.20 | 0.11 | 0.51 | 0.102 / 0.066 |

Other numbers. Exposed-face accuracy of the head with nothing placed: 0.66 (f = 2), 0.63
(no conditioning), 0.80 (oracle), against 0.60 for a (type, direction) majority prior.
Raw one-shot matching F1 before realization: 0.26 (0.57 with the oracle). Multimodality
and diversity: two temperature-1 grow samples of the same input overlap with voxel IoU 0.29
(f = 2), 0.16 (no conditioning), 0.43 (oracle). For the merge decoder (4 samples on 16
inputs, check + extent) pairwise voxel IoU between samples is 0.12 (0.23 pose-invariant),
every sample distinct, and each sample's IoU to the real chunk is 0.13.

**The check.** Grow without the check loses a quarter of the blocks to collisions (22 per
chunk at f = 2) and drops glue F1 from 0.51 to 0.44, but the surviving blocks still sit in
roughly the right coarse cells. One-shot is almost unaffected by the check because its
matching yields many small components (the largest holds about 14% of the blocks). For the
merge decoder the window-extent test does most of the work: without it every sample
becomes a long one-block line.

### Debugging the merge decoder

The first merge evaluation (overlap check only) collapsed to long straight lines
(occasionally an L), far below the random policy, while teacher-forced pointer accuracy
was 0.72 to 0.93. Findings:

* **No feature skew.** A guard test (`test_merge_features_incremental_equal_scratch`)
  replays decoded connections and checks that the partition, anchor offsets and face
  states the decoder feeds equal those rebuilt from scratch from the same connection set.
  Candidates and scoring match training (all cross-component pairs of `X_a`, same axis
  sign, the j-side component moves).
* **The teacher-forced target is nearly uninformative early.** With every block a
  singleton, 43% of all candidate pairs are true connections (any same-type singleton may
  play any real block), so the high pointer accuracy mostly measures type and axis
  compatibility. The greedy first pick is true 96% of the time, yet greedy picks x-axis
  merges 56% of the time against 35% in real data.
* **Exposure bias with a line attractor.** In teacher forcing, components are compact real
  pieces (largest component at k = N/2: max extent / size^(1/3) = 2.0); in free running they
  are elongated (3.0), a state the model never saw, and extending a line end never collides,
  so overlap checking cannot stop it. The extent test turns lines into flat sheets (see
  `samples_ext.png`), which is better than random but still far from the grow decoder.
* **Frame-free merging cannot use the condition.** Components have no absolute frame, so
  f = 2, f = 4 and no conditioning give the same merge results. Using the condition as a
  hard mask (a component must still fit the coarse counts after a merge) deadlocks: wrongly
  assembled clusters cannot be joined within the counts, so only 1% of builds complete and
  the largest component holds 38% of the blocks (47% with the oracle).

Not tried for lack of time: training on the decoder's own rollouts (scheduled sampling or
DAgger-style mixed states), which is the direct fix for the exposure bias.

**Full houses (paused).** A merge run on whole houses_32 builds (largest component, 64 to
512 blocks, median 370, 496 train builds with 4 yaw rotations, f = 4) trained to a
teacher-forced pointer accuracy of about 0.8 (`outputs/run_20261007_183327_adjacency_faces_merge_houses_f4`,
checkpoint only), but was stopped before evaluation once the window-scale merge decoder
was shown to collapse; no full-house generation numbers exist.

### What the samples look like

* Grow + check (`run_..._f2_resid/samples.png`): coherent pieces with the right material
  roles (planks walls with dark-oak posts, a grass floor, pink beams over cobble), but the
  global shape often departs from the condition (a vertical wall where the condition has a
  horizontal slab).
* Grow + check + coarse mask: the closest to the real chunks; the white/dark-oak slab and the
  pink beam frames reappear.
* Random policy + coarse mask and naive upsampling: the right outline filled with speckled,
  disconnected-looking material noise.
* One-shot: single rods or small slabs (the largest matched component).
* Merge: lines without the extent test; flat sheets and slabs with it; temperature 1 gives
  layered piles.

### Verdict

With a coarse shape, the face model does real work in the grow decoder. Under identical
masks (check + coarse mask) the model beats the random policy on glue F1 (0.54 vs 0.38),
typed F1 (0.24 vs 0.18), voxel IoU (0.48 vs 0.45) and support W1 (0.034 vs 0.071), and it
beats naive upsampling on voxel IoU and glue F1, while upsampling keeps the best typed F1
(it puts each cell's dominant type exactly in that cell). Without the mask the model uses
the condition only weakly (coarse IoU 0.49 against 0.26 unconditioned), so most of the
coarse consistency comes from the condition used as a constraint. The oracle cell index
lifts glue F1 to 0.70, so per-block placement information, not more model, is the main
remaining lever. The merge formulation, as trained here, does not work: its quality comes
from the extent check, and it needs training on its own rollouts.

### Commands

    .venv/bin/python -m tools.misc.adjacency_gen faces-run --f 2 --name f2_resid --minutes 32 --bs 32 --max-gb 1.5
    .venv/bin/python -m tools.misc.adjacency_gen faces-run --f 4 --name f4_resid --minutes 32 --bs 32 --max-gb 1.5
    .venv/bin/python -m tools.misc.adjacency_gen faces-run --f 0 --minutes 32 --bs 32 --max-gb 1.5
    .venv/bin/python -m tools.misc.adjacency_gen faces-run --f 2 --oracle --name oracle_resid --minutes 32 --bs 32
    .venv/bin/python -m tools.misc.adjacency_gen faces-run --merge --f 2 --minutes 30 --bs 32   # also --f 0, --f 4, --oracle
    .venv/bin/python -m tools.misc.adjacency_gen faces-eval --run <run dir> --arms merge_check merge_check_t1 merge_random_check --tag _ext
    .venv/bin/python -m tools.misc.adjacency_gen faces-run --houses --merge --f 4 --bs 1 --decode-bs 1 --n-gen 40
    .venv/bin/python -m pytest -q tests/misc/adjacency_gen

`--data-from <run dir>` reuses a run's `dataset.pkl`; `--arms` picks arms (`t50_check` needs
the T50 voxel run). Runs: `outputs/run_20261007_171614_adjacency_faces_f2_resid`,
`..._171629_..._f4_resid`, `..._171644_..._oracle_resid`, `..._163543_..._nocond` (plus
`results_cmask.json`), `..._163503_..._f2` and `..._163523_..._f4` (without the residual
feature), `..._174923_..._merge_f2`, `..._174943_..._merge_nocond`, `..._175803_..._merge_f4`,
`..._175823_..._merge_oracle` (each with `results_ext.json`; merge f2 and oracle with
`eval_cmask.log`: the merge coarse-mask arm deadlocks, and its random-policy control was
stopped unfinished because each rejected merge runs the coarse-fit test). Each run folder has `samples*.png` with rows condition, real, then one
row per arm.

## Full houses: cell assignment, then connections (2026-10-08, third prototype)

T52 left two pointers: telling each block its coarse cell was the largest lever, and the
merge decoder failed because clusters had no frame. This prototype generates **whole
houses_32 builds** in two stages. Stage 1 samples a cell assignment (how many blocks of
each type go in each coarse cell); stage 2 builds the house inside that frame. Stage 2 never
handles individual inventory items: "an oak plank in cell c" is a *type slot*, and the
blocks of one slot are interchangeable.

### Data (cells.py)

Every house of the v1 split, largest face-connected component, shifted to the origin of a
32^3 grid, **no size cap**: 1,863 train houses (7,452 with the 4 yaw rotations), 399 val,
399 test. Test houses have 155 to 7,202 blocks (median 623; train median 794, max 7,673).
Vocabulary: the 255 most frequent unoriented block tokens plus `<unk>`. The condition at
factor f (grid G = 32 / f) is `coarse.coarsen` (count and dominant type per coarse cell);
the inventory (block multiset) comes from the real house, as in T50 and T52. The cell
assignment `A[c, t]` has the condition counts as row sums and the inventory as column sums.

### Stage 1: cell assignment (stage1.py)

* **Model.** A transformer (4 layers, d = 192, 1.9M params) over the coarse cells plus one
  inventory token (`log1p` of the type counts). Output per cell: logits over block types,
  as a residual on top of a heuristic prior (global inventory frequencies plus a learned
  bonus for the cell's dominant type), with types absent from the inventory masked. Loss:
  NLL of every real block's type under its cell's distribution.
* **Sampling (cells.sample_assignment).** IPF scales the per-cell distributions into an
  expected table with the right margins; then every slot (one per block) is visited in a
  random order interleaved across cells and draws a type proportional to the still unused
  part of its row, among types with inventory left. Every sample matches both margins
  exactly, and one (condition, inventory) pair gives many assignments.
* **Baseline.** The same sampler on the heuristic prior alone (dominant type weight 4).

### Stage 2, main arm: grow in cells (grow_cells.py)

* **Action.** One step attaches one slot to the build: a type t and a free fine cell q that
  is face-adjacent to the build and lies in a coarse cell with `rem[c(q), t] > 0`. This is
  T52's oracle grow decoder over type slots: "open face of p, unplaced block j" becomes
  "free cell next to the build, slot type", which merges every face pointing at the same
  cell and every interchangeable block of the slot into one action. The check (empty
  target, inside the grid) and the coarse mask hold by construction. The first block, and
  a new seed when no attachment is legal while slots remain (`n_seeds`), may use any free
  cell of a coarse cell with slots, so every build places every block.
* **Model (GrowNet, 3.6M params).** Houses have up to 7,673 blocks (46k face tokens), so
  attention is not over blocks. A 3D U-Net over the 32^3 grid of placed types (the two fine
  levels see about one coarse cell and its 26 neighbours) meets, at the 8^3 bottleneck, a
  4-layer transformer over coarse cells whose tokens carry the condition, the remaining
  slot counts per type and the cell position; the slot counts are also broadcast to the
  fine grid. Score of (q, t) = `<u(h_q), e_t> + b_t`.
* **Training.** Teacher forcing on prefixes of random growth orders of real houses (Dijkstra
  from a random seed with Exp(1) edge weights, i.e. first-passage percolation, the same law
  as T52's uniform random frontier edge), with the true assignment. Loss: `-log` of the total
  probability of the correct actions (q holds a real block of type t) over the legal set.
* **Decoding.** One model call places up to `K = ceil(N / 256)` blocks at distinct frontier
  cells (top-K of the scores, Gumbel-perturbed at T = 1), skipping picks whose slot ran out
  within the call. `_k1` arms place one block per call.

### Stage 2, second arm: merge in cells (merge_cells.py)

The user's formulation with a frame: all blocks are instantiated from the assignment, every
block starts as its own cluster, and one step glues block i's +a face to block j's -a face.

* Every cluster has a **translation box**: the translations that keep each member inside
  its own coarse cell (a singleton's box is its cell). A merge intersects the boxes and is
  rejected if the intersection is empty, so clusters have a position from step one and the
  condition constrains every step. Candidates: j's cell equals i's cell or the next cell
  along +a. Checks: no overlap inside the merged cluster; a box that shrinks to one point
  fixes the cluster in the grid, and fixed clusters may not overlap. Every check only gets
  stricter as clusters grow, so rejected candidates are banned for good. When no candidate
  is left, clusters are laid out (fixed ones at their point, others largest first at a free
  translation of their box; blocks that still collide are dropped).
* **Type slots.** Singletons of one (cell, type) slot get identical features (no random
  identity), so one representative per slot enters the candidate set. With all blocks
  alone, 12.7% of the legal candidates of a sample house are correct (T52: 43%).
* **Model (MergeNet, 6.4M params).** Per-block features (type, cell position, estimated
  absolute position = box centre + offset, box extent, log cluster size, a random cluster
  vector, face state inside the cluster, log number of singletons in the slot), then 4
  rounds of [MLP; mean over the cluster; mean over the coarse cell, a 3x3x3 convolution over
  the coarse grid and a transformer layer over cells with the condition tokens]. Pair score
  `<q_a(h_i), k_a(h_j)> / sqrt(dk) + b_a`.
* **Targets.** A multi-block cluster is consistent if some translation in its box puts every
  member on a real block of its type; that maps its members to real blocks. A real contact
  (u, v, a) is realized by the cluster members mapped to u and v or, if no consistent
  cluster claims them, by the representative singleton of their slot; it is a correct
  candidate if both ends are realized in different clusters. On teacher-forced states
  (random Kruskal prefixes of the real contact graph) this is exactly "real cross-cluster
  contacts, singletons exchangeable within their slot" (tested); on the decoder's own
  states it still names the merges that extend consistent clusters.
* **Training on own rollouts (`--dagger`).** Half of every batch comes from a buffer of the
  decoder's own states: every 150 steps, 8 training houses are decoded (T = 1, oracle
  assignment, 64 rounds) and snapshotted at 3 random rounds. Trained from scratch with the
  same 40-minute budget as the teacher-forced run (rollouts took 4.4 of the 40 minutes).
  Merge arms are trained and evaluated on houses with at most 1,024 blocks (4,636 of 7,452
  train items; 297 of 399 test houses).

### Results (test split, single seed 0, f = 4 unless noted)

Columns: `single` = the build is one face-connected component holding every block
(`complete`, every block placed, is 1.00 for every grow and upsampling arm by construction);
voxel IoU and typed F1 (Dice of (cell, type)) against the real house in the shared frame;
`shift IoU` = best voxel IoU over translations within +-2 cells; glue F1 = F1 of the typed
glued face pairs (as in T52); slot L1 = share of blocks in a different (coarse cell, type)
slot than the real house. Every grow and upsampling arm matches the condition exactly
(coarse IoU 1.0, count L1 0).

All 399 test houses:

| arm | single | voxel IoU | shift IoU | typed F1 | glue F1 | slot L1 |
|---|--:|--:|--:|--:|--:|--:|
| stage 1 + grow (greedy) | 0.75 | 0.322 | 0.364 | 0.196 | 0.580 | 0.298 |
| stage 1 + grow, T = 1 | 0.75 | 0.321 | 0.358 | 0.192 | 0.566 | 0.298 |
| **oracle assignment + grow** (upper bound) | 0.76 | **0.326** | 0.369 | **0.245** | **0.654** | 0 |
| heuristic-prior assignment + grow | 0.76 | 0.315 | 0.356 | 0.204 | 0.592 | 0.337 |
| random policy, stage-1 assignment | 0.55 | 0.284 | 0.296 | 0.148 | 0.420 | 0.298 |
| random policy, oracle assignment | 0.52 | 0.283 | 0.294 | 0.184 | 0.504 | 0 |
| naive coarse upsampling | 0 (largest comp. 54%) | 0.246 | 0.254 | 0.155 | 0.406 | 0.352 |
| stage-1 assignment at random positions | 0 (55%) | 0.245 | 0.255 | 0.129 | 0.348 | 0.298 |
| retrieval, nearest train house by coarse grid | 1 | 0.320 | **0.381** | 0.049 | 0.096 | 0.854 |

Retrieval returns a real, complete house, but with another inventory (inventory L1 0.78)
and only 0.76 coarse IoU with the condition.

Merge subset (297 test houses with at most 1,024 blocks; grow rows on the same houses):

| arm | one cluster | clusters left | blocks dropped | voxel IoU | typed F1 | glue F1 |
|---|--:|--:|--:|--:|--:|--:|
| stage 1 + grow | 0.76 (single) | | 0 | **0.306** | 0.196 | 0.585 |
| oracle + grow | 0.79 (single) | | 0 | **0.310** | **0.240** | **0.652** |
| stage 1 + merge | 0.007 | 6.0 | 8.3% | 0.267 | 0.155 | 0.570 |
| oracle + merge | 0.027 | 5.9 | 8.9% | 0.263 | 0.192 | 0.625 |
| stage 1 + merge, random policy | 0 | 13.2 | 10.3% | 0.262 | 0.153 | 0.423 |
| oracle + merge, random policy | 0 | 12.7 | 9.8% | 0.264 | 0.184 | 0.498 |
| stage 1 + merge, own-rollout training | 0 | 22.9 | 22.2% | 0.237 | 0.144 | 0.494 |
| oracle + merge, own-rollout training | 0 | 23.7 | 23.0% | 0.232 | 0.174 | 0.550 |

f = 8 (4^3 coarse grid, cells of 512), all 399 houses:

| arm | single | voxel IoU | shift IoU | typed F1 | glue F1 | slot L1 |
|---|--:|--:|--:|--:|--:|--:|
| stage 1 + grow | 0.83 | 0.223 | 0.279 | 0.118 | 0.579 | 0.192 |
| stage 1 + grow, T = 1 | 0.82 | 0.229 | 0.280 | 0.119 | 0.567 | 0.192 |
| oracle + grow | 0.82 | 0.227 | 0.280 | **0.133** | **0.599** | 0 |
| heuristic prior + grow | 0.83 | 0.221 | 0.276 | 0.121 | 0.582 | 0.236 |
| random policy, stage 1 / oracle | 0.50 / 0.57 | 0.205 / 0.204 | 0.240 | 0.087 / 0.099 | 0.397 / 0.431 | |
| naive upsampling | 0 (31%) | 0.164 | 0.180 | 0.075 | 0.279 | 0.285 |
| retrieval | 1 | **0.277** | **0.343** | 0.038 | 0.081 | 0.871 |

Stage 1 alone (399 test houses): per-block NLL 1.53 nats (heuristic prior 1.89) at f = 4
and 1.77 (2.31) at f = 8; slot L1 of a sampled assignment 0.299 (0.337) at f = 4 and 0.192
(0.236) at f = 8; two samples for the same input differ in 12.5% of slots at f = 4. Stage 1
overfits after about 1,000 steps (1,863 distinct houses); the checkpoint with the best
validation NLL is kept. A smaller, more regularized variant with mirror augmentation
(`stage1_f4_reg`) gives the same test numbers (NLL 1.535, slot L1 0.296).

Diversity (first 64 test houses, 3 samples each; pairwise voxel IoU between samples, then
mean IoU of a sample to the real house): stage 1 + grow at T = 1: 0.435 / 0.406 (typed F1
between samples 0.25); oracle + grow at T = 1: 0.448 / 0.416; sampled stage 1 with greedy
grow: 0.473 / 0.410; random policy: 0.331 / 0.325. At f = 8: 0.342 / 0.298 for stage 1 +
grow at T = 1. Samples of one input differ from each other about as much as each differs
from the real house.

Parallel placement does not cost quality: on the first 64 test houses, oracle + grow gives
voxel IoU 0.418 and glue F1 0.587 with `K = ceil(N / 256)` and 0.416 / 0.593 with one block
per call (8.4 times slower: 690 s against 82 s); stage 1 + grow 0.412 / 0.510 against
0.411 / 0.512.

Training (one seed, GPU shared by up to four of these jobs at once, so step counts are
lower than a solo run would get): stage 1, 12 min (best at step 1,000 of 18,000) at f = 4
and 10 min at f = 8; grow, 40 min each (22,143 steps at f = 4, 25,289 at f = 8, batch 16,
1.9 GB), teacher-forced top-1 validation accuracy 0.95 / 0.92; merge, 40 min each (13,789
teacher-forced steps, batch 8, validation top-1 0.77; own-rollout run 15,451 steps, 0.80 on
the same teacher-forced validation states). Evaluation of all f = 4 arms: 45 minutes (f = 8: 18 minutes).

### What the samples look like

* **Grow (stage 1 or oracle).** The outline and the material zoning follow the condition
  closely: pink and green roof material on top of the cherry houses, plank walls, cobble
  bases, the white and dark-oak bands of the long house. The volumes are porous and
  irregular: no clean wall planes or roof slopes, scattered single blocks inside, and
  occasional rods sticking out of the outline (one oracle sample has a long plank beam).
  The oracle assignment gives visibly cleaner material regions than stage 1.
* **Random policy and upsampling** under the same assignment: the right outline filled with
  speckle; upsampling also leaves floating blocks (largest component about half the house).
* **f = 8 grow**: sparser, more "architectural" fragments (plank floors, wall pieces, posts
  and beams, an open frame) with the right materials, but nothing closes into a house.
* **Merge in cells**: no more one-block lines (T52's failure); chunky, porous blobs with the
  right materials per region, several loose clusters, and a few dropped blocks. Own-rollout
  training makes them more speckled.
* **Retrieval**: clean real houses that match the coarse shape loosely and the materials
  not at all.

### Reading and failure analysis

* **The stage-2 model does real work under a fixed frame.** With the same assignment, grow
  beats the random policy by about 0.15 glue F1 (0.654 vs 0.504 oracle, 0.580 vs 0.420 stage
  1), 0.05 to 0.06 typed F1 and 0.04 voxel IoU, and it keeps 75% of builds in one piece against
  about 53%. Glue F1 is the metric where it is strongest: local contacts are learned.
* **Geometry is not recovered beyond what the condition gives.** Voxel IoU is 0.32 for every
  learned arm, about the same as retrieval (0.32), and the best translation only lifts it
  to 0.36. At f = 4 a coarse cell holds up to 64 blocks, so the condition pins far less than
  T52's f = 2 condition on 8^3 windows (cells of 8), where the same decoder reached 0.48 to
  0.55. The sample pictures agree: right volume and materials, wrong internal structure.
* **The oracle assignment is a smaller lever than T52's oracle cell.** It adds 0.07 glue F1
  and 0.05 typed F1 but almost no voxel IoU (0.326 vs 0.322). Knowing which types go in each
  cell says little about where they go inside it.
* **The learned stage 1 does not help downstream.** It lowers the slot error (0.298 vs 0.337
  for the heuristic prior), yet prior + grow has slightly higher glue F1 (0.592 vs 0.580)
  and typed F1 (0.204 vs 0.196). The prior concentrates each cell on its dominant type,
  which yields homogeneous regions whose contacts are common; the model's samples spread
  types more. Stage 1 also overfits quickly on 1,863 houses.
* **Merge in cells works locally and deadlocks globally.** Its glue F1 is close to grow on
  the same houses (0.625 vs 0.652 with the oracle) and well above its random policy (0.498),
  so the frame fixed T52's line collapse. But only 0.7 to 2.7% of builds end as one
  cluster: about 6 clusters remain whose every remaining candidate overlaps, empties a box
  or hits a fixed cluster (6,400 to 6,600 rejected candidates per house), and 8 to 9% of blocks
  are dropped at layout. Grow cannot deadlock, because each placement is checked against
  the realized grid at once instead of against loose, partly placed clusters.
* **Training on its own rollouts made merging worse** (23 clusters left, 22% dropped). A
  plausible reason: the consistent-cluster target gives no credit to any merge that touches
  a wrong cluster, so on its own states the model learns to avoid joining those clusters
  rather than to repair them; the snapshots also came from 64-round rollouts (4 times more
  merges per call than the evaluation decoder). Not tested further.

### Next

The most promising step is a finer frame for the grow decoder: f = 2 on full houses (16^3
coarse grid, cells of 8), where T52's window results suggest the condition starts to pin
geometry, with stage 1 run at f = 4 and refined to f = 2 hierarchically, or stage 2 given a
per-cell occupancy pattern instead of type counts only. Second, scoring with the benchmark's
`geom_kid` and an inventory-matched novelty check against retrieval, with n of at least 256
and more seeds. For the merge arm, a repair move (split a cluster) or a target that credits
any merge consistent with the condition, rather than with the one real house, is the
direct fix for the deadlock.

### Commands

    .venv/bin/python -m tools.misc.adjacency_gen cells-stage1 --f 4 --minutes 12 --bs 32
    .venv/bin/python -m tools.misc.adjacency_gen cells-stage1 --f 4 --minutes 10 --d 128 --heads 4 --layers 3 \
        --dropout 0.3 --wd 0.05 --lr 2e-4 --mirror --eval-every 250 --name stage1_f4_reg
    .venv/bin/python -m tools.misc.adjacency_gen cells-stage1 --f 8 --minutes 10 --eval-every 250
    .venv/bin/python -m tools.misc.adjacency_gen cells-grow --f 4 --minutes 40 --bs 16       # also --f 8
    .venv/bin/python -m tools.misc.adjacency_gen cells-merge --f 4 --minutes 40 --bs 8 --max-n 1024 [--dagger]
    .venv/bin/python -m tools.misc.adjacency_gen cells-eval --f 4 --stage1 <stage1 run> --grow <grow run> \
        --merge <merge run> <merge dagger run>
    .venv/bin/python -m tools.misc.adjacency_gen cells-eval --f 4 ... --n 64 --n-div 0 \
        --arms oracle_grow oracle_grow_k1 s1_grow s1_grow_k1 --name eval_f4_k1
    .venv/bin/python -m pytest -q tests/misc/adjacency_gen

Runs: `outputs/run_20261008_010026_adjacency_cells_stage1_f4`, `..._011130_..._stage1_f4_reg`,
`..._012320_..._stage1_f8`, `..._010024_..._grow_f4`, `..._014046_..._grow_f8`,
`..._010618_..._merge_f4`, `..._011351_..._merge_dagger_f4`, evaluations
`..._020520_..._eval_f4` (results.json, rows.json with per-house metrics; its samples.png
was redrawn on the first 8 test houses with the same models and seeds, results_fig8.json and
config_fig8.json; config.json was restored from the launch command),
`..._eval_f8`, `..._014701_..._eval_f4_k1`. Every folder has samples.png: condition, real,
then one row per arm (merge rows leave houses above 1,024 blocks blank).

Files: `cells.py` (data, assignments, IPF sampler, heuristic prior), `stage1.py`,
`grow_cells.py`, `merge_cells.py`, `eval_cells.py` (metrics, baselines, figure),
`cells_cli.py` (the `cells-*` commands); tests in `tests/misc/adjacency_gen/test_cells.py`.
