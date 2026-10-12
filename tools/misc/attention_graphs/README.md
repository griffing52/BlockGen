# attention_graphs: can self-attention act as an adjacency matrix?

The question: when a transformer reads a serialised build (one block per line), does
some attention head already behave like the build's adjacency matrix, so that a
generator could emit a native graph (blocks plus "which earlier block do I touch")
without training a graph model?

This package visualises attention as a block-by-block matrix, scores every head
against the true contact graph, and runs the controls that decide whether a high
score means geometry or a shortcut. The literature memo is `RESEARCH.md`.

## What T28 already established, and what is new here

T28 (`results.md`, `scripts/probe_llm_attention.py`) ran the basic experiment on
Qwen2.5-Coder-1.5B with Track D `piece x y z` lines in raster order over whole houses:
best head AUROC 0.874 (0.800 distance-stratified), recency 0.831, trained bilinear
probe 0.895 to 0.914, but the L1 distance between the printed coordinates scored 0.980
with no model. The head mostly copies coordinate strings.

New here:

1. **A visualiser** (it did not exist): token-level and block-level attention heatmaps,
   all heads of a layer, and the selected head next to the true adjacency.
2. **The repo's own voxel AR transformer** (`VoxelTransformerAR2`, canon16_flat
   checkpoint, canonical val builds it never trained on). The question T28 could not
   answer: does a model trained only on houses grow a head that ranks 3D neighbours
   above sequence-distance and coordinate-matching controls?
3. **A coordinate-free serialisation** for the LLM (each line names a parent line and a
   face direction, no xyz), against absolute coordinates.
4. **Shuffled-order controls** for every format, and a new **hard-pair** metric that
   removes the coordinate-copying shortcut (below).
5. A trained edge probe (distance and bilinear) on frozen per-block states, layer by
   layer, on held-out builds, under every condition.

## Metrics and controls

All scores follow T28 so numbers are comparable: for every causal candidate pair
`(i, j)` with `j < i`, block `i`'s attention to block `j` (query tokens of line `i`
averaged, key tokens of line `j` summed, each row renormalised over earlier lines) ranks
"is `j` a 6-neighbour of `i`?". The AUROC and stratified AUROC functions are imported
from T28's script.

| metric | removes | meaning |
|---|---|---|
| `auc` | nothing | raw AUROC over all causal pairs |
| `strat` | recency | AUROC inside bins of sequence distance `i - j`, pooled (T28) |
| `hard` | coordinate matching | AUROC only over pairs that share exactly two of x/y/z. Matching identical coordinate tokens scores exactly 0.5 here; separating them requires knowing that the third coordinate differs by one |
| `hard_strat` | both | `hard`, stratified by sequence distance. The strictest test |

Baselines on the same items: `recency` (`-(i-j)`), `coord_match` (number of identical
coordinates, T28's copy signature), `coord_L1` (T28's 0.980 subtraction; an oracle when
coordinates are not printed), `same_name`, and for the relative format `tree_hops`
(path length in the stated parent tree, readable from the text). Heads are selected on
even items and reported on odd items. Probes train on 60% of items and test on the rest
(different builds).

## Results

Run: `outputs/run_20261006_231900_attn_adjacency/` (`summary.md` has every number).

### 1. The repo's voxel AR transformer (60 canonical val builds, median 218 blocks)

`[BOS, (X, Y, Z, BLOCK)*, EOS]`, 6 layers x 8 heads. A node is one block's four tokens;
`qZ` means the query is the block's `Z` token (the moment its position is complete).

| order | query | recency auc / hard_strat | best head auc | strat | hard | hard_strat | probe (distance) strat / hard_strat |
|---|---|---|---|---|---|---|---|
| raster | Z | 0.923 / 0.488 | 0.975 | **0.982** (L4H2) | 0.839 | **0.871** (L4H2) | 0.976 / 0.975 (layer 3, mean query) |
| raster | mean | 0.923 / 0.488 | 0.966 | 0.964 | 0.826 | 0.795 | |
| shuffle | Z | 0.496 / 0.503 | 0.816 | 0.823 | 0.586 | 0.597 | 0.978 / 0.964 (layer 2, mean query) |
| shuffle | mean | 0.496 / 0.503 | 0.762 | 0.774 | 0.546 | 0.561 | |

`coord_match` 0.975 auc (0.5 on hard pairs by construction), `coord_L1` 1.000.

**This is the first positive in the repo for learned geometry in attention.** In the
order it was trained on, head L4H2 queried from the `Z` token ranks true neighbours at
0.982 with recency removed, and at 0.871 on hard pairs with recency removed, where both
shortcuts sit at chance. Where its attention peaks (argmax over earlier blocks, 6
builds): the block directly below (`dy = -1`) 32% of the time, the block one row behind
(`dz = -1`) 27%, two below 16%. Both targets sit a variable number of lines back
(row and layer lengths differ), so the head finds them by content, not by a fixed
offset (`raster_qZ_best_head_vs_truth.png` shows the two bands).
It does not survive a shuffled order (0.597 hard_strat), but shuffled sequences are
out of distribution for a model trained only on raster order, so that is a statement
about this head, not about the representation: a distance probe on the same frozen
states reaches 0.96 to 0.99 hard_strat at layers 0 to 2 under both orders (shuffled,
it then declines to 0.77 by the last layer). The
learned coordinate embeddings already lie on a number line, so "differs by one" is
linear in the representation before any attention happens. Two caveats: the AR
sequence still contains coordinate tokens, so `coord_L1` (1.000) beats every head; and
one head covers one neighbour direction, not the full graph (AP 0.55 at a 1.7% edge
rate).

### 2. Qwen2.5-1.5B on house chunks (48 chunks of 5x5x5, median 36 blocks, 9.3% edges)

| format / order | recency auc | coord_match auc | best head auc | strat | hard | hard_strat | probe (distance) auc / hard_strat |
|---|---|---|---|---|---|---|---|
| absolute / raster | 0.853 | 0.939 | 0.940 (L14H0) | 0.902 | 0.850 | 0.734 | 0.950 / 0.925 (L16) |
| absolute / shuffle | 0.506 | 0.939 | 0.882 (L8H3) | 0.897 | 0.649 | 0.664 | 0.922 / 0.917 (L8) |
| relative / bfs | 0.813 | (oracle) | 0.920 (L5H11) | 0.841 | 0.873 | 0.813 | 0.839 / 0.784 (L8) |
| relative / random | 0.797 | (oracle) | 0.849 (L14H0) | 0.765 | 0.794 | 0.671 | 0.772 / 0.655 (L0) |
| LEGO / raster (stud graph) | 0.770 | 0.564 | 0.811 (L5H11) | 0.671 | | | 0.924 (L4) |
| LEGO / shuffle | 0.518 | 0.564 | 0.593 | 0.627 | | | 0.902 (L4) |

Relative format, does attention follow the pointer the text states, and find contacts
it does not state?

| order | parent top-1: best head / recency | unstated contacts auc / strat: best head | recency | tree_hops | same_name |
|---|---|---|---|---|---|
| bfs | 0.590 / 0.038 | 0.869 / 0.740 | 0.822 / 0.603 | 0.605 / 0.575 | 0.623 / 0.624 |
| random | 0.335 / 0.268 | 0.784 / 0.660 | 0.753 / 0.537 | 0.657 / 0.595 | 0.647 / 0.654 |

Qwen2.5-Coder-1.5B (T28's model) is within noise of the base model on every arm:
absolute/shuffle best head 0.871 auc, 0.653 hard_strat (`coord_match` 0.939); distance
probe 0.923 hard_strat (L8); relative/random parent top-1 0.451 (recency 0.268),
unstated contacts 0.678 stratified (`same_name` 0.654); LEGO shuffle head 0.590, probe
0.888.

Reading:

* **Absolute coordinates, shuffled (the key control).** The best head (L8H3) scores
  0.882, below the zero-model `coord_match` baseline (0.939), and only 0.649 on hard
  pairs. From a digit token it puts 81% of its mass on earlier tokens with the
  identical id (7.9% would be uniform; 6 chunks): a copy head, as T28 concluded. T28's diagnosis replicates on
  small chunks and under shuffling.
* **The probe beats every head and survives shuffling** (0.917 hard_strat at L8): the
  LLM's residual stream knows that "3" and "4" are adjacent numbers, no head reads it
  out. This is the standard probe-beats-attention result, now with both shortcuts
  removed.
* **Coordinate-free text.** With no coordinates printed, the heads do not reliably
  follow the pointer the text spells out (`on p dir`): parent top-1 0.59 in BFS order
  and 0.34 in random growth order (recency 0.27). The unstated contacts score 0.66
  stratified under random growth, level with the `same_name` baseline (0.654). The
  probe is weak too (0.655 hard_strat). A frozen LLM does not compose relative moves
  into geometry, so the information is genuinely absent rather than hidden.
* **LEGO** behaves like the absolute format: the head collapses under shuffling
  (0.593), the probe does not (0.902).

### Verdict

Zero-shot "attention as adjacency" from a frozen LLM stays negative: wherever a head
looks like an adjacency matrix, a model-free reading of the same text matches or beats
it, and without coordinates in the text nothing is there to read. The repo's own AR
model is different in kind: trained only on builds, it grew a neighbour head that beats
both shortcuts in its training order, and its states make adjacency linearly readable
in any order. Attention is therefore a plausible place to put the edge decision, but
it has to be trained there; it will not be found for free.

**Next step toward native graph generation** (matches `RESEARCH.md` options c and d):
train the AR model with a coordinate-free, parent-pointer serialisation and supervise
one head (LISA style) to put its mass on the parent block, with the pointer distribution
masked to legal faces. The evidence here says (i) such a head can emerge in this
architecture (L4H2), (ii) the needed information is linear in the states from layer 0,
so a single supervised head or a bilinear pointer on the states is enough, and (iii)
the pretrained LLM will not supply it zero-shot, so the LLM variant needs fine-tuning.
A cheaper intermediate check: re-run section 1 on a checkpoint trained in random growth
order, to see whether the neighbour head becomes order-invariant.

## Figures (per model directory)

| file | shows |
|---|---|
| `heatmap_{auc,strat,hard,hard_strat}.png` | layer x head score per arm, diverging around chance 0.5; panel title gives the best head and the recency baseline |
| `heatmap_unstated_auc.png` | LLM relative format: AUROC on contacts the text never states |
| `<arm>_best_head_vs_truth.png` | item 0: selected head's block-level attention (row-renormalised, times row index so 1 = uniform) next to the true lower-triangular adjacency (first 64 blocks) |
| `<arm>_head_grid_L<l>.png` | every head of that layer for the same item |
| `<arm>_tokens_L<l>H<h>.png` | token-level attention of the selected head, first 20 lines, line boundaries drawn |
| `probe_by_layer.png` | held-out distance-stratified AUROC of the distance probe at each layer |
| `<arm>_example.txt` | the exact serialised text of item 0 |
| `head_scores.csv`, `results.json` | every per-head mean, best heads, baselines, probe results |

## Usage

```bash
export HF_HOME=~/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=.
RUN=outputs/run_$(date +%Y%m%d_%H%M%S)_attn_adjacency
.venv/bin/python -m tools.misc.attention_graphs ar2 --n 60 --out $RUN
.venv/bin/python -m tools.misc.attention_graphs llm --n 48 --lego \
    --models Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-Coder-1.5B --out $RUN
.venv/bin/python -m tools.misc.attention_graphs figures --examples $RUN   # redraw
.venv/bin/python -m pytest tests/misc/attention_graphs                  # ~20 s, CPU
```

The GPU is shared: `pick_device` uses CUDA only if 4 GB (LLM) or 2 GB (AR) are free.
Without `--out` the run goes to `outputs/run_<stamp>_attn_adjacency`. `ar2 --ckpt` takes
any flat-token run dir with `model.pt` and `block_vocab.json`.

## Modules

| module | role |
|---|---|
| `scenes.py` | house chunks from `houses_32.npz` (largest connected component), toy structures, LEGO stacks with stud graphs |
| `serialize.py` | absolute (`x y z block`) and relative (`k block on p dir`) text, orders, exact per-line token spans |
| `extract.py` | eager-attention LLM loading, `[L, H, T, T]` extraction, token-to-node pooling (mean/last query, sum/mean key), row renormalisation, rollout, symmetrisation |
| `ar2.py` | `VoxelTransformerAR2` loading, attention recomputed from each block's `qkv` (the fused kernel returns none), per-slot queries |
| `adjacency.py` | ground truth (face, stud, tree hops), baselines, AUROC / stratified / AP / hard-pair metrics |
| `probe.py` | Hewitt-Manning distance probe and low-rank bilinear probe on frozen node states |
| `experiment.py` | per-arm accumulation, held-out head selection, probes, CSV |
| `viz.py` | all figures |

Limitations: one AR checkpoint (no geometric-bias or BPE-piece variants), small chunks
for the LLM (whole houses are T28's regime), one seed, and n of 24 report items per arm,
so differences under about 0.03 are not meaningful.
