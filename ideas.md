# Ideas: paths to strong text/image → structure generation

Rating of every generation approach we've considered, scored for the specific goal
of **strong conditioned (text/image → structure) generation**. Companion to
`research.md` (landscape + quality levers) and `notes.md` (run log). Written
2026-07-19 from a five-thread literature sweep; load-bearing citations inline.

## The organizing distinction: encoder vs decoder

Almost every idea below reduces to one split that the literature treats as a hard
regularity:

- **Reading/aligning structure into a shared semantic space** (a "SigLIP for
  structure"; semantic node embeddings) is an *encoder* problem — well-supported and
  cheap at our data scale.
- **Autoregressing structure out** is a *decoder* problem, and the aligned/semantic
  representation does **not** save you here: generation always needs a decoder trained
  on the structure's own tokens with a reconstruction objective.

Every "generate from the shared space" system (ImageBind→DALL·E-2, Point-Bind→
CLIP-Forge, Michelangelo, Shap-E, CLIP-Sculptor) uses the aligned embedding *only as a
condition* and generates with a *separate* native-token decoder. Michelangelo
(NeurIPS 2023) names the pattern **"alignment-before-generation."** Contrastive
embeddings are a lossy semantic bottleneck (unCLIP: the decoder must invent "details
absent from the CLIP representation"), so they are not invertible to a voxel grid or a
connection graph. **Alignment gives a front-end, never a generator.**

## The LegoACE puzzle (why 45k samples looked so good)

LegoACE is *native tokens on a pretrained LLaMA decoder* + frozen CLIP-text **and
DINOv2 multi-view** conditioning + **DPO**. The "world knowledge" is in the backbone
and the rich multi-view image conditioning, not the tokenization. That also explains
the reproduction difficulty: quality rides on (a) the pretrained decoder, (b)
multi-view image conditioning far richer than a pooled text token, (c) DPO preference
data papers underspecify, (d) a private dataset (LegoVerse, unreleased). Takeaway: the
reproducible core of the LegoACE recipe on our stack is **multi-view aligned
conditioning (idea 1) + preference post-training (idea 9)** — ideas we can build.

## Ratings

Scale Low / Med / High. **Compute** = cost (Low is good). **Payoff** = expected gain
for text/image → structure specifically.

| # | Idea | Novelty | Feasibility | Compute | Payoff | Role |
|---|---|---|---|---|---|---|
| 1 | Own structure encoder (SigLIP/DINOv3-style) → conditioned AR decode | Low–Med | High | Med | **High** | Front-end |
| 2 | Finetune LLaMA on serialized structure (track D) | Low | High | Med | Med | Baseline |
| 3 | Code/DSL interface (`fill`, `sphere`…) via LLM | Med | Med | Med | Med–High | Dark horse |
| 4 | Structure↔VLM-token translator (encode in / decode out) | Med | Med | Med–High | Med–High | Front-end+ |
| 5 | Native graph-gen mode on a VLM (nodes+edges+tags) | **High** | Low–Med | Med–High | Med (uncertain) | Flagship bet |
| 6 | Conditioning-channel upgrade on our decoder (multi-token cross-attn / Q-Former) | Low | High | **Low** | **High** | Quick win |
| 7 | Semantic node-embedding prior (frozen text/img embeddings as the token table) | Low–Med | High | **Very Low** | Med | Quick win |
| 8 | Typed-connection AR + constrained decoding (E.2) | **High** | Med | Low–Med | Med | Flagship bet |
| 9 | Data scaling + DPO / VLM-reward (research.md C) | Med | Med–High | Med–High | **High** | Quality lever |

### Per-idea rationale

1. **Own encoder → decode.** The established "alignment-before-generation" recipe
   (CLIP-Forge CVPR 2022; Michelangelo NeurIPS 2023): align a structure tower into a
   frozen CLIP/SigLIP space via rendered views, then decode from the text/image
   embedding. Trainable at our scale (ULIP ~52k, CLIP2Point 52,460, ReCon ~51k all
   work by distilling into a *frozen* space). Strong for the goal; not novel.
2. **LLaMA serialized (track D).** Reviewer-expected, proven (LegoGPT, MarioGPT,
   LLaMA-Mesh), QLoRA-cheap. Won't beat our from-scratch decoder on unconditioned
   quality; keep as the baseline, not the bet.
3. **Code/DSL.** Underrated. Massive sequence compression (a house = dozens of
   `fill`/loops vs thousands of voxels) and it leans on the *code* pretraining prior
   (CADmium: code-LLMs beat general LLMs on DSLs). Risks: must synthesize training
   *programs* from existing builds (inverse/library-learning, nontrivial), and
   "executable ≠ spatially correct" (VoxelCodeBench). Best for regular architecture,
   weak for organic shape.
4. **Structure↔VLM translator.** The connector pattern (MetaQueries, GILL): a learned
   connector reads a frozen VLM's hidden states → our decoder; the VLM is the
   conditioning brain. Overlaps idea 1; higher compute (running a VLM). Output side
   still needs the native decoder.
5. **Native graph-gen on a VLM.** Most novel — the native-graph-*output*-vs-text
   ablation is genuinely unrun (graph soft-tokens beat text for *reading* graphs,
   GraphToken +73pp, but nobody generates structure with them). Highest risk: no
   precedent, conditioning still needs the front-end. Paper bet, not a results bet.
   The "nodes with tags to reason about connections" part = idea 7, foldable in.
6. **Conditioning-channel upgrade.** `CondVoxelAR2` conditions on *one pooled CLIP
   token → one prefix vector* — the textbook weak link (Point-E/Shap-E "struggle with
   intricate prompts"). Multi-token cross-attention / a small Q-Former is days of work
   and directly targets the measured T15 failure ("palette transfers, geometry
   doesn't"). Cheapest high-payoff move.
7. **Semantic node prior.** Frozen text/image embeddings as the block/piece token
   table (wood≈spruce; "this component is used for X"). Well-supported by the
   text-attributed-graph literature (OFA, ZeroG, UniGraph — but all *discriminative*);
   attacks our block-agreement 0.03–0.13 collapse. Cheapest experiment of all.
8. **Typed-connection AR.** Pose derived from connection + constrained decoding so
   invalid edges are inexpressible. High novelty (validity-by-construction over rich
   parts is the open intersection, research.md E.2). Mainly a validity/representation
   win; conditioning is separate. Caveat: "valid by construction" is often really a
   *decoding-time* artifact (BrickGPT 98.8%→24% stable without rollback).
9. **Data scaling + DPO.** Our own research.md says data scaling likely fixes cohesion
   more than any reward; LegoACE's DPO was a major quality lever. Highest compute
   (20–50k generations, GPU-days) but high payoff. Probably the single biggest quality
   jump available.

## Two recommended sequences

**Novelty and the goal diverge.** The novel arms (5, 8) mostly buy validity/diversity
and still need a conditioning front-end; the things that most improve text/image →
structure are the un-novel ones (6, 1, 9). So pick the objective.

**For strong conditioned results (the stated goal):** `7 → 6 → 1 → 9`, cheapest first.
- **7 + 6 (days each):** semantic node prior + fix the conditioning channel on the
  *existing* decoder; measure with the paired-vs-shuffled IoU + palette metric. This
  isolates whether conditioning is weak because of the *channel* (bet: mostly yes).
- **1 (1–2 wks):** the aligned multi-view structure encoder as the richer front-end.
- **9 (GPU-days):** DPO / VLM-reward for the quality jump.
- This sequence *is* the reproducible core of the LegoACE recipe on our stack.

**For the paper:** **8 is the flagship** (typed-connection validity-by-construction
over a rich part vocabulary + the native-output-vs-text ablation nobody has run), with
5 as the higher-risk stretch, and 2 as the required baseline.

**Best combined bet:** run the front-end cluster (7→6→1→9) as the results engine and 8
as the novelty arm in parallel — they share the decoder and don't compete.

## Zero-shot new node types — the honest verdict

Using semantic node embeddings as a **placement prior** (idea 7) is well-supported.
Using them for **zero-shot generation of unseen node types** is aspirational: no
buildable-domain generator emits a truly unseen type (LegoGPT: 8 fixed types), and
token models cannot emit a symbol with no learned embedding (Akyürek & Andreas, ACL
2021). The tractable reframe is **open-set *selection* over an inference-time
catalog** — pointer / constrained decoding (GENRE reports near-parity seen-vs-unseen,
64.4 vs 63.2) or predict-embedding→nearest-neighbor decode (TIGER cold-start) —
budgeting for the generalization drop on never-seen entries (ZeroGR 63.5→23.8 Hit@10).

## Status

- **[tested — NEGATIVE] 7 — semantic node-embedding prior.**
  `blockgen/models/semantic_embedding.py` (`SemanticTokenEmbedding` + CLIP text-embedding
  builder), wired into `VoxelTransformerAR2` via `semantic_embedding=`. Tests:
  `tests/test_semantic_embedding.py` (6/6). A/B harness: `scripts/run_semantic_prior.py`.
  The frozen CLIP space *is* meaningful (wood planks cluster; furnace near fire/torch/iron
  — function, not just appearance), **but the 60-epoch A/B did not pan out**
  (`outputs/run_semantic_prior_60ep`):

  | metric | baseline (learned) | semantic prior |
  |---|---|---|
  | block_agreement (target) | **0.076** | 0.052 |
  | validity_rate | **0.438** | 0.312 |
  | val_nn_iou | 0.375 | **0.393** |
  | mean_nn_iou | 0.423 | **0.438** |

  The prior *lowered* the material-coherence metric it was meant to raise, and hurt
  validity, while marginally helping novelty. Caveats: single seed; n=16 samples
  (validity/block-agreement are noisy); **input-side only** (output head is still a
  learned softmax, so material *selection* may not benefit) — tying the output head to
  the semantic vectors is the untested variant; and this measured *unconditioned* quality,
  not the *conditioning* payoff #7 was ranked for. Verdict: not worth more investment as
  the input-only variant on this evidence; the conditioning path (#6/#1) is higher-value.
- **[training] 6 — conditioning-channel upgrade.** `blockgen/models/condition_resampler.py`
  (`ConditionResampler` Q-Former-lite + `ResampledCondVoxelAR2`), via a behavior-preserving
  `_make_prefix` refactor of `CondVoxelAR2` (existing cond parity tests still 8/8; per-sample
  `cond_drop` added for CFG). Tests: `tests/test_condition_resampler.py` (6/6).
  - Sequence-level conditions built: `labeling/embed_conditions.py` now emits
    `text_token_embeds` [N,4,L,512] + `text_token_mask` (CLIP `last_hidden_state`; opt-in
    DINOv2 patch tokens) → `data/minecraft/labels/houses_32_cond_embeds_seq.npz`.
  - Trainer: `scripts/train_cond_resampler.py` (mirrors `train_conditioned.py` text arm;
    renders `samples.png` per run). Full pipeline smoke-verified end-to-end.
  - **Real 40-epoch run training now** (`outputs/cond/text_resampler`, n_prefix=16). Compare
    its conditioning fidelity (paired-vs-shuffled IoU + palette) against the pooled-prefix
    `outputs/cond/text_run` via `scripts/sample_conditioned.py`.
- Others unstarted. See sequences above for order.
