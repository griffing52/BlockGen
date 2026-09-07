# Implementation Plan — Attachment / Growth Autoregressive Model

_Design date: 2026-07-20. Companion to `ideas.md` (#5/#8), `research.md` §E.2
("typed-connection tokens"), and `notes.md` §19 (the run that motivated it).
This is an engineering plan only — no code is written here._

---

## 1. Overview & motivation

Every generator we have shipped tokenizes a build inside a **fixed box**: raster
`[BOS, (X,Y,Z,PIECE)*, EOS]` (`serialize.py`, `cluster_bpe.py`) or a dense grid
(`voxel_diffusion.py`). The box is the binding constraint. `notes.md` §19 records
the cost precisely: the pooled corpus `all_32.npz` has only **4,712 builds**, not
the ~40k available, because text2mc builds are large (median max-dim 96, p90 256)
and only the ones that fit a native 32³ box survive. Growing the box fights
`native_bpe` sequence length; decimating destroys 86.5% of blocks (§9.0 / T17).

**The attachment/growth model dissolves the box.** Instead of emitting absolute
coordinates into a grid, it **grows** a structure: seed one piece, then
autoregressively attach new pieces to *open faces* of already-placed pieces. Pose
is **derived** from the connection (which face + orientation), so absolute
coordinates are never emitted and a placement that does not connect is
inexpressible — **connectivity is guaranteed by construction**, the one thing the
in-loop adjacency gate (`training/constrained_decode.py`, `eval/validity.py`) had
to enforce after the fact. There is **no air token**: a face is either filled or
`CLOSE`d (a boundary); air ≡ a per-face CLOSE. Generation ends when the frontier of
open faces is empty. Because nothing is boxed, structures grow to any size — the
model can train on **all ~40k corpus builds regardless of size**, and data scaling
is the single most reliable cohesion lever we have measured (`notes.md` §7, T10).

This is the paper's flagship: **validity-by-construction over a rich part
vocabulary + the native-connection-output-vs-text ablation nobody has run**
(`ideas.md` #8, `research.md` §E.2). Related work to cite (all in
`deliverables/references.bib`): VoxelCNN/3D-Craft (order-aware next-block, humans
build bottom-up + connected), BrickAnything (BFS attachment-tree, parent-relative
tokens), SolidGen (pointer nets indexing prior geometry), GCPN / GraphAF / JT-VAE
(sequential graph construction with valency checks + resampling — the analog of our
collision-check-and-resample), GraphRNN / GRAN / DiGress (autoregressive/diffusion
graph generation order), TreeSBA & Brick-by-Brick (LEGO assembly sequences).

---

## 2. Representation / tokenization spec

**Substrate already exists.** `blockgen/utils/graph_data.py` already models each
voxel as a block node owning **six port nodes** (one per axis face, `PORT_DIRECTIONS`
= ±X,±Y,±Z with `OPPOSITE_PORT`) joined by typed `EDGE_PORT_TO_PORT` edges. The
"open faces" ARE these ports. We reuse this schema verbatim as the state
representation and add an *emission order* over it.

**Attachment-op sequence.** A build linearizes as a sequence of ops, each op = a
decision about **one target open face** taken from a FIFO frontier:

- `SEED(piece)` — the very first op; place `piece` at the bottom-center anchor.
- `ATTACH(piece, face_orient)` — attach `piece` to the current target face; the
  new piece's mating face is determined by `face_orient` (which of the new piece's
  faces docks, plus its discrete rotation for multi-voxel/oriented pieces).
- `CLOSE` — the target face is a boundary (no attachment). This is the local EOS /
  the air replacement.

Pose is derived: given the target face (a port on an existing piece, with a known
world position + outward direction) and the chosen `face_orient`, the new piece's
world position is `neighbor = parent_pos + PORT_DIRECTIONS[dir]` for single-voxel
pieces, generalized to the piece's offset set for BPE pieces. **No X/Y/Z tokens are
ever emitted.**

**Canonical order (the user's refinement).** Deterministic, gravity-aware,
foundations-first (VoxelCNN evidence + `notes.md`: humans build bottom-up +
connected):

1. **Seed** = the occupied voxel nearest the **bottom-center** of the cropped
   build: minimum `y`, then the voxel closest to the `(x,z)` centroid of that
   bottom layer (Euclidean, ties broken by `(x, z)` lexicographic).
2. **BFS across the bottom-most x,z plane first** (all `y == y_min` voxels reachable
   through 6-adjacency within the plane), **then climb vertically** layer by layer.
   Concretely: BFS over the port graph with a priority that exhausts the current
   `y` before any face that raises `y`. This gives "finish the foundation, then go
   up".
3. Within a layer, expand faces in a fixed order: horizontal faces (+X,-X,+Z,-Z)
   before the +Y face, each in `PORT_DIRECTIONS` index order.
4. **Face-visit order is the frontier FIFO order**; for each visited face emit
   exactly one op (`ATTACH` or `CLOSE`). A face already filled by a previously
   placed piece is skipped (it is an internal, already-satisfied edge).

This order is a *pure function of the occupancy* (no randomness), so a build
tokenizes identically every time — the property `cluster_bpe.py` relies on. Reuse
`utils/ordering.py` (BFS-from-ground already exists) as the starting point; note
`notes.md` §8 found *unconstrained* BFS token order hurt raster AR — that finding is
about coordinate emission and does not apply here, because in the growth model the
order **is** the generative process, not a scan pattern the model must invert.

> **TESTED 2026-07-21 (results.md T21) — the argument above HOLDS.** It was
> load-bearing and unverified, so it was measured before spending GPU. Across 1,200
> builds, `bfs_bottom_center` wins **both** training-free proxies: lowest held-out
> bits/op (2.324 vs 2.510 `layered_raster`, 2.692 `dfs`, 4.522 `radial`) *and*
> highest agreement with real human placement order from 3D-Craft (ρ=0.352). T11's
> anti-BFS result does not transfer. `radial` is decisively worst at ~2× the bits/op,
> so **locality alone is not the property that matters — gravity/layer structure is.**
> Ordering nonetheless remains a *parameter* (`attach_order.ORDERINGS`), not a
> constant, because these are proxies and sample quality is the real objective.
>
> **New hard constraint — decode-availability.** An ordering may only read what the
> *decoder* also has: face coordinates and the partial structure. It may NOT read
> ground-truth occupancy. Two natural candidates (`support_first`, `shell_first`)
> scored the frontier against the *final* build at encode time and the *partial*
> build at decode time; the frontier desynced and round-trip IoU fell to 0.23. Any
> ordering defined over properties of the finished structure is therefore invalid
> unless the frontier re-evaluates priorities at pop time (stale-key problem).
> Priorities must also be written relative to the component seed, so they are
> shift-invariant across the encode and decode coordinate frames.

**Face indexing.** Six canonical directions from `graph_data.PORT_DIRECTIONS`;
`OPPOSITE_PORT` gives the mating face. Each face carries its **own embedding** (see
§3) so the model reasons about connection type and where to connect.

**Per-face CLOSE.** Absence of attachment = CLOSE = boundary; there is no air token
anywhere in the vocabulary. A build is complete when the frontier is empty (every
open face has been either attached-through or CLOSEd).

**Multi-voxel BPE pieces.** A `cluster_bpe.ClusterVocab` piece is an occupied-only
offset set (`patterns`, already reanchored). Its **open faces** = the set of
(voxel, direction) port positions on its boundary that are not internal to the
piece. When such a piece is placed, all of its boundary faces enter the frontier;
its internal faces never do. `face_orient` for a multi-voxel piece selects (a) which
boundary face docks to the target and (b) the piece's discrete rotation (the D4 set,
`utils/augment.py`). The forced multi-block merges (`cluster_bpe.MULTI_BLOCK_IDS`:
beds/doors/double-plants) already guarantee those objects are one piece with a
correct multi-face footprint.

**Deterministic tie-breaking.** Frontier is a FIFO; seed selection, layer priority,
and per-face direction order are all fixed above. The full linearization is
reproducible and round-trips (parse ops → place pieces → occupancy == input), which
is the acceptance test for the extractor (§4).

---

## 3. Model architecture

**State encoder — recommendation: incremental sparse encoder over placed pieces.**
Two options:

- **(A) Graph/point transformer over placed pieces** — reuse the PyG substrate:
  `LargePyGGraphGenerator`'s `TransformerConv` encoder stack
  (`models/large_pyg_graph_generator.py`) or `VoxelPortGNN`'s `SAGEConv` stack over
  the block+port graph from `structure_to_pyg_data`. Natural fit: the frontier faces
  are literally nodes.
- **(B) Sparse 3D CNN over occupancy** — denser spatial context, but re-voxelizes
  each step and does not expose faces as first-class objects.

**Recommend (A)**, because faces are already nodes and the head predicts per-face,
and because it is size-agnostic (no box — consistent with the whole point). Start
from `VoxelPortGNN` (smaller, `SAGEConv`) for the MVP; graduate to the
`TransformerConv` stack for capacity.

**Face-embedding scheme.** Each open face is embedded from: (i) its parent piece
token embedding (reuse `models/factored_embedding.FactoredPieceEmbedding` or the
plain `nn.Embedding` used in `voxel_transformer_ar2.py`), (ii) a learned
**per-direction** embedding (6-way, cf. `VoxelPortGNN.port_direction_embedding`),
(iii) its position in the growth order / height, and (iv) a pooled summary of the
whole current partial structure. Each step is conditioned on **the whole current
partial structure + the current piece + the specific target face** — exactly the
face node's contextualized embedding after message passing, concatenated with the
graph-pooled vector (`global_mean_pool` + `global_max_pool`, as in
`large_pyg_graph_generator.encode_graph`).

**Prediction head.** From the target-face embedding, a softmax over
`{CLOSE} ∪ {(piece, face_orient)}`. Factor it to keep it tractable:
`P(CLOSE) `, else `P(piece) · P(face_orient | piece)`. `piece` ranges over the
`ClusterVocab` pieces (single-voxel in the MVP); `face_orient` ranges over the
piece's dockable faces × discrete rotations. Mask illegal `face_orient` given the
piece (only faces that can physically mate an outward-facing parent port).

**Incremental / cached encoding (critical, avoids O(N²)).** Re-encoding the whole
growing graph every step is O(N) per step → O(N²) per build. Mitigations, in order:

1. **Locality window** — recompute messages only for nodes within k hops of the
   just-placed piece; freeze the rest (attachment only perturbs the local
   neighborhood). This is the cheap 80% win.
2. **Cached node states + incremental message passing** — keep per-node hidden
   states; on `ATTACH`, add the new piece's nodes and run message passing only over
   the dirty frontier (mirrors the KV-cache philosophy in
   `voxel_transformer_ar2.py`, whose lesson (`notes.md` §8) is that per-step launch
   overhead, not FLOPs, dominates — so **batch builds** and keep steps cheap).
3. **Teacher-forcing shortcut for training** — at train time the full final graph is
   known, so encode it once and predict all ops in parallel with a causal mask over
   the growth order (no incremental recompute during training at all). Incremental
   encoding is a *sampling-time* concern only.

---

## 4. Training

**Data prep — canonical attachment-order extractor.** New module
`blockgen/utils/attach_order.py`: `structure_to_attach_ops(Structure, ClusterVocab)
→ List[Op]` implementing §2's canonical order over the piece labeling produced by
`cluster_bpe`'s merge replay (`_atomic_labeling` + `_apply_merge`). Round-trip test:
ops → placement → occupancy IoU == 1.0 (the bar `cluster_bpe` already meets). Run
over the full pooled corpus via `curation/corpus.py` (`all_32.npz` loader
`load_structures_from_cache`) — and, crucially, **without the 32³ box filter**, so
the ~40k builds all qualify (this is the whole point; large builds simply produce
longer op sequences).

**Oriented vocab + D4 rotation table.** Use the `oriented=True` vocab
(`notes.md` §19: stairs 8 facings, logs 16). **Blocking prerequisite for
augmentation:** `notes.md` §17/§19 — the D4 augmenter (`utils/augment.py`) does NOT
rotate `block_data`, so augmenting the oriented vocab corrupts stair/log/door
facings. Either (a) train un-augmented first (like `train_uncond_corpus.py`, which
deliberately skips D4), or (b) build the **block_data rotation table** (§17) so D4
augmentation is orientation-correct. The growth model needs this table for augmented
runs — schedule it as a Phase-2 dependency, not an MVP blocker.

**Teacher forcing.** Encode the final graph once; predict every op from its
target-face embedding under the causal growth-order mask. Loss = cross-entropy over
the `{CLOSE, (piece, face_orient)}` head at every face-visit, summed per build,
mean over builds. Optionally down-weight `CLOSE` if the class is imbalanced (many
boundary faces) — but note this is *not* the air-imbalance problem, since CLOSE is a
per-face binary within a small head, not a dominant grid class.

**Config reuse.** `ARTrainConfig` (`training/train_ar.py`) for optimizer/schedule;
`train_ar_ext.train_from_sequences` as the loop template; bf16 autocast, cosine LR,
per-epoch val NLL + early stop (the small-data bundle, `research.md` §B.8). Save the
`ClusterVocab` next to the checkpoint (`export.minecraftace.save_piece_vocab`) —
`notes.md` §18(b) burned two unloadable checkpoints on exactly this.

---

## 5. Sampling

Greedy/temperature autoregressive growth (`generate`-style loop mirroring
`voxel_transformer_ar2.generate`):

1. **Init** — `SEED(piece)` sampled from the marginal; place at origin; push its open
   faces onto the frontier FIFO.
2. **Loop** — pop the next target face; encode incrementally (§3); sample from the
   `{CLOSE, (piece, face_orient)}` head.
3. **Viability / collision check (the user's refinement, the GCPN/JT-VAE analog)** —
   before committing an `ATTACH`, verify the proposed piece's derived voxels (a) lie
   in no already-occupied cell and (b) do not otherwise violate legality. Reuse a
   fast voxel-occupancy set (a Python `set`/hash grid of placed cells; `research.md`
   §E.4 "voxel proxies"). If invalid → **resample** (up to N tries) → else force
   `CLOSE` that face. This is BrickGPT-style rollback compressed to a per-face
   decision.
4. **Frontier update** — on `ATTACH`, push the new piece's still-open boundary faces;
   any face that now abuts an existing piece is auto-satisfied (removed, not
   emitted). On `CLOSE`, just drop the face.
5. **Termination** — frontier empty ⇒ done. Add a hard cap on total pieces
   (size-blowup guard) mirroring `generate`'s `max_new_tokens`.

Decode ops → `Structure` (place each piece's `ClusterVocab.patterns` at derived
anchors, exactly like `cluster_tokens_to_structure`). Then render via the project
standard: `eval/cond_render.textured_prompt_grid` + `utils/runs.new_run_dir`.

---

## 6. Evaluation

Reuse the existing suite unchanged (it is representation-agnostic — operates on
`Structure`/occupancy):

- **Novelty / memorization** — `eval/novelty.evaluate_novelty` (occupancy-IoU NN,
  `duplicate_rate`, inter-sample diversity, block agreement) vs **train and a
  held-out val split**, with the val-vs-train baseline printed (`notes.md` §6c).
- **Validity** — `eval/validity` (`largest_component`, connected-component count).
  ~~**Expected headline: validity ≈ 1.0 by construction**, no gate needed~~ —
  **CORRECTED 2026-07-21 (T21): validity is NOT 1.0 by construction.** Pose derivation
  makes any single component connected, but nothing stops the model emitting a
  **second `SEED`**, which starts a new component — multi-component samples are
  exactly multi-SEED samples. Banning `SEED` after position 0 forces validity to 1.0
  as a pure decode-time **filter**, i.e. precisely the artifact confound of §10. So
  Phase-1 arms sample **unmasked** and report *learned* validity; the masked number is
  the filter arm of the ablation and must never be reported as the headline alone.
- **Figures** — textured EGL renders via `eval/cond_render.textured_prompt_grid`
  (project render standard, `notes.md` §13).
- **The flagship ablation — native-connection-output vs raster-text** — same corpus,
  same eval, compare growth-model samples against `train_uncond_corpus.py`'s
  `native_bpe` raster AR: validity, novelty, cohesion, and (the unrun claim)
  *whether growth trains on the full 40k where the box tops out at 4,712*.

---

## 7. Phased milestones (checklist)

- [ ] **Phase 0 — extractor + round-trip.** `utils/attach_order.py`
      (`structure_to_attach_ops` / `attach_ops_to_structure`); canonical order per §2;
      round-trip IoU == 1.0 on `all_32` and on a handful of >32³ builds. Deliverable:
      passing round-trip test over the full pooled corpus, op-length distribution
      reported. **Gate for everything downstream.**
- [ ] **Phase 1 — MVP: single-voxel pieces, unconditioned, full corpus.** Atomic
      `ClusterVocab` (no merges), `VoxelPortGNN` encoder + per-face head, teacher
      forcing, un-augmented. Train on the **whole ~40k** (no box filter — the headline
      capability). Deliverable: a trained checkpoint, textured sample grid, novelty +
      validity vs `native_bpe`, and the "trains on 40k vs 4,712" claim substantiated.
- [ ] **Phase 2 — BPE multi-voxel pieces.** Swap atomic vocab for the learned
      `ClusterVocab` (with forced multi-block merges); multi-face bookkeeping + D4
      rotation of `face_orient`; build the block_data rotation table (§17) so
      augmentation is orientation-correct. Deliverable: shorter op sequences,
      cohesion/novelty delta vs Phase 1.
- [ ] **Phase 3 — typed connections.** Promote `face_orient` to typed connectors
      (the LEGO stud/anti-stud generalization, `research.md` §E.2); mask by
      connector compatibility. On Minecraft the type is trivial (6-adjacency), so this
      phase is the *scaffolding* that makes the LEGO transfer drop-in. Deliverable:
      typed-connection head + compatibility mask, validity still 1.0.
- [ ] **Phase 4 — add conditioning.** Plug the existing SigLIP→resampler stack
      (`models/condition_resampler.ConditionResampler`, `ResampledCondVoxelAR2`
      pattern) as a prefix/cross-attention condition on the per-face head — unchanged,
      because conditioning is orthogonal to how structure is emitted. Deliverable:
      text/image-conditioned growth, paired-vs-shuffled fidelity via
      `eval/cond_render.render_cond_run`.

---

## 8. File / module layout

New modules (reusing existing code named inline):

- `blockgen/utils/attach_order.py` — the extractor/detokenizer. Reuses
  `cluster_bpe._atomic_labeling` / `_apply_merge` (piece labeling), `graph_data`
  (`PORT_DIRECTIONS`, `OPPOSITE_PORT`), `utils/ordering.py` (BFS-from-ground),
  `Structure.crop_to_non_air`.
- `blockgen/models/attach_growth.py` — the growth model. Reuses `VoxelPortGNN` /
  `LargePyGGraphGenerator` encoders, `factored_embedding.FactoredPieceEmbedding`,
  and the incremental-encode/cache idea from `voxel_transformer_ar2.py`.
- `blockgen/training/train_attach.py` — trainer. Reuses `ARTrainConfig`
  (`train_ar.py`), the `train_ar_ext.train_from_sequences` loop shape, `utils/runs`,
  `export.minecraftace.save_piece_vocab`.
- `blockgen/eval/` — **no new files**; reuse `novelty.py`, `validity.py`,
  `cond_render.py`.
- `scripts/train_attach_corpus.py` — driver mirroring `scripts/train_uncond_corpus.py`
  (corpus load, vocab save, train, textured samples), pointed at the **unfiltered**
  corpus.
- Conditioning (Phase 4) reuses `models/condition_resampler.py`,
  `labeling/embed_conditions.py`, `data/minecraft/labels/*` — no changes.

---

## 9. Risks & open questions

- **Incremental-encoding cost.** Naive re-encode is O(N²)/build; the k-hop locality
  window + cached node states (§3) must land early or sampling large builds is
  intractable. `notes.md` §8's lesson: batch builds and minimize per-step Python /
  kernel-launch overhead — the FLOP argument is not the real cost.
- **Collision ≠ global coherence.** Local viability guarantees connectivity and
  no-overlap, not that the *gestalt* is a house — exactly `ideas.md` #8's caveat and
  BrickGPT's PhysHack. Validity-by-construction is a floor, not the quality win;
  conditioning + data scale still carry cohesion.
- **Ordering determinism at scale.** The seed/BFS/tie-break rules must be a strict
  function of occupancy across all corpora (grabcraft/3dcraft/text2mc), including
  disconnected components (multiple seeds? reject in curation?). Decide the
  multi-component policy in Phase 0.
- **Memory / sequence length.** Removing the box means op sequences for big text2mc
  builds can be very long (p90 max-dim 256). Teacher forcing over the full graph
  needs a length budget / gradient checkpointing; may need to keep a max-pieces cap
  for the largest builds even though the *representation* has none.

  > **CONFIRMED 2026-07-21 (T21) — and it is the binding constraint on sample quality,
  > not just on memory.** `max_seq_len` is a *data filter* that removes large builds
  > specifically: retention 70.2% @4096 → 88.4% @8192 → 92.3% @16384, and the trainable
  > slice's median occupancy moves 806 → 1,120 with it. Raising 4096 → 8192 (matched
  > ordering, temperature, sampler and decoder) **nearly tripled generated median
  > occupancy, 242 → 710**, at preserved thickness and validity 1.00, and improved loss
  > 12.6%. So "the box is dissolved" is only true of the *representation* — a length
  > budget reintroduces the same selection pressure, and relaxing it is currently the
  > highest-leverage knob in the whole pipeline. Two consequences for this plan: (1)
  > always report retention + trainable-reference occupancy alongside any sample metric,
  > or the model is judged against a distribution it never saw; (2) ~~the §3 encoder is
  > motivated by sampling cost, not sample quality~~ — **WRONG, retracted same day.** That
  > claim rested on a buggy `thickness` metric (`np.roll` wrapped, inflating thin
  > structures up to +79%); corrected, neither the filament problem nor the size problem
  > was actually solved. The prefix diagnostic (results.md T21) then returned **BLINDNESS**:
  > handed a real prefix the model shuts the frontier down *faster* (close-rate 0.659 →
  > 0.869 against a flat ground-truth 0.606), so it cannot read its op history as geometry.
  > **The §3 encoder is a prerequisite for Phase 1 working at all.** The seq-length →
  > occupancy result above still stands on its own (matched single-variable test); it just
  > does not imply better *structure* — the extra occupancy went into larger flat plates.
- **Multi-voxel face bookkeeping.** Enumerating a BPE piece's boundary faces under
  all D4 rotations, and correctly auto-satisfying internal faces on placement, is
  fiddly and the most bug-prone part of Phase 2 — round-trip IoU is the guardrail.
- **Orientation data (`notes.md` §17/§19).** Oriented vocab needs the block_data D4
  rotation table before any augmented run; without it, augmentation teaches
  wrong-facing stairs/logs. Un-augmented training sidesteps this for the MVP.

---

## 10. Research update — circuit / typed-graph findings (2026-07-21)

A circuit/netlist-generation literature sweep sharpened three points for this model
(see also `hierarchical_plan.md`, which wraps this growth model as its composition layer):

- **Pin/terminal-level nodes are validated as SOTA.** AnalogGenie (ICLR 2025) generates
  circuits as an autoregressive traversal over a **pin-level** graph (nodes = device
  terminals, not whole devices) and hits **93.2% valid** vs 57–68% for whole-component
  baselines — because pin-level nodes remove connection ambiguity. This is exactly the
  6-ports-per-block substrate already in `graph_data.py` (`PORT_DIRECTIONS`), and it is
  the project's own prior idea (the `graph_data.py` netlist docstring), externally
  confirmed. Keep faces/ports as the attachment nodes, not whole blocks/pieces.
- **A checker-in-the-loop is REQUIRED, not optional.** Every domain shows a valid typed
  graph ≠ a working result: VerilogEval (syntax 60% vs functional 43.5%), LegoGPT
  (100%-valid but 24%-stable without physics rollback), AnalogCoder (SPICE-in-the-loop
  self-correction). Phase-1 sampling must run the collision/occupancy oracle + a
  rollback/repair step; do not trust connectivity-by-construction alone (it guarantees
  the graph is connected, not that the build is coherent/stable).
- **Isolate LEARNED validity from FILTER validity (the artifact confound).** Typed rules
  raise validity mostly as rejection/masking (GraphAF 68→100% via a valency mask;
  grammar-constrained decoding is 100% by definition) — reviewers discount this. To claim
  the model *learned* the connection grammar, run a DiGress-style ablation: measure
  validity of the growth model with the collision mask OFF. That number is the real
  evidence, and it's the ablation to bake into Phase 1's eval.

Anchor citations on the verified set (AnalogGenie, LaMAGIC, DiGress, LegoGPT, VerilogEval,
Holodeck); several 2026 preprints (incl. `bricknet2026` currently in `references.bib`)
are UNVERIFIED — keep them off load-bearing claims.
