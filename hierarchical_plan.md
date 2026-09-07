# North-Star Design: One Typed-Connection-Graph Generator Across Domains

_Design date: 2026-07-21. A thesis document, not an engineering spec. Synthesizes two
literature sweeps (`research.md`, `ideas.md`) with the repo's existing substrate
(`blockgen/utils/graph_data.py`, `models/condition_resampler.py`). Companion to
`implementation_plan.md` (the attachment/growth model, which is the composer layer of
this architecture). Citation keys refer to `deliverables/references.bib`; relatives not
yet in the bib are named with venue and flagged **[add to bib]**._

---

## Thesis

**ONE typed-connection-graph generative model, with per-part semantic text descriptions
that enable LLM reasoning, transferring across Minecraft ↔ LEGO ↔ electronics.**

Structure is represented not as a coordinate grid but as a graph of *described parts*
joined at *typed connection points*. The repo already commits to this representation:
`graph_data.py` models every Minecraft voxel as a block node owning **six typed port
nodes** (`PORT_DIRECTIONS` = ±X, ±Y, ±Z, with `OPPOSITE_PORT` giving the mating face)
connected only through `EDGE_PORT_TO_PORT` edges. That is, structurally, a **netlist**:
components (blocks/parts) with pins (ports) wired pin-to-pin. The pin/port-level
typed-connection idea is therefore the project's **own prior commitment**, expressed in
code well before any external precedent. AnalogGenie (ICLR 2025, pin-level analog-circuit
graphs, 93.2% valid topologies) is *validation that the representation generalizes to a
physically distinct domain* — it is confirmation, not the source, of the bet.

---

## 1. Motivation: why a typed-connection graph of described parts

Every generator this repo has shipped tokenizes a build inside a **fixed box** — raster
`[BOS, (X,Y,Z,PIECE)*, EOS]` or a dense voxel grid (`voxel_diffusion.py`). The box is the
binding constraint: `notes.md` §19 records that the pooled corpus caps at **4,712 builds**
(not the ~40k available) purely because larger builds do not fit a native 32³ box. Beyond
the data problem, a coordinate grid encodes *where every atom is* but nothing about *what
the pieces are* or *how they attach* — precisely the information needed to generalize.

The generalization target is **out-of-distribution scale and elaboration**: castles,
cathedrals, whole planets — builds far larger and more intricate than anything in a
16k-house training set. A coordinate model cannot reach these; it has no mechanism to
compose. A part-graph can, because the OOD claim is deliberately narrow: **novel
arrangements of known parts, not novel part types.** Walls, towers, roofs, gates, arches
already exist (in distribution) in the house corpus. A castle is a *new arrangement* of
those known parts. Recombining a fixed vocabulary into unseen configurations is the
regime where compositional models provably extend beyond their training distribution
(SCAN \cite{scan2017}, COGS \cite{cogs2020}, CFQ \cite{cfq2019}; compositional skill
emerges multiplicatively, Okawa et al. \cite{okawa2023}). Generating a genuinely *unseen
part type* is the hard frontier and is explicitly **out of scope** — `ideas.md`'s honest
verdict is that no buildable-domain generator emits a truly unseen type, and token models
cannot emit a symbol with no learned embedding (Akyürek & Andreas \cite{akyurek2021}).

The compositional plan — *which* parts, and *how* they connect — is exactly what a
web-scale LLM knows and a 16k-house model does not. "A castle has an outer curtain wall
with corner towers, a gatehouse on the south face, and a keep at the center" is world
knowledge, not geometry to be learned from scratch. This is the motivating division of
labor for the architecture below.

## 2. The three-layer architecture

**Layer A — LLM symbolic planner.** From a prompt, an LLM emits a *part-graph*:
```
{ parts:       [ {id, text_description, role, approx_size}, ... ],
  attachments: [ (partA.port  →  partB.port), ... ] }
```
It supplies **what** parts and **how** they connect *semantically*, drawing on web-scale
knowledge. Critically, **the LLM does not do spatial layout or geometry.** This is not a
convenience — it is a finding. Architect (NeurIPS 2024) **[add to bib]** shows LLM
planners can *lose* to diffusion on spatial arrangement; VoxelCodeBench (FAIR 2026) and
3D-PreMise **[add to bib]** show LLMs emit *valid symbolic plans with wrong geometry*
("executable ≠ spatially correct", echoed in `ideas.md` #3 and `research.md` §E.1). The
planner's output is therefore *purely relational* — a described, connected part-graph
with no coordinates — and is produced under **structured / grammar-constrained decoding**
(grammar-constrained decoding \cite{gcd2023}, Outlines \cite{outlines2023},
XGrammar \cite{xgrammar2024}) so the emitted JSON is always a well-formed graph.

**Layer B — native part generator.** Each part's `text_description` is realized by a
native block generator. **The repo already has this component:** the SigLIP→resampler
stack (`models/condition_resampler.py`: `ConditionResampler` Q-Former-lite +
`ResampledCondVoxelAR2`) *is* a text-description → native-blocks generator, trained in
`notes.md` §19. Each part is **in-distribution** — walls, towers, and roofs are the very
things the house corpus contains — so this layer works within the regime the model is
competent in, rather than being asked to imagine a whole castle at once.

**Layer C — checked attachment composer.** Parts are attached at their **typed ports**
(the pin/terminal-level nodes — the six ports/block already in `graph_data.py`, the
AnalogGenie generalization to typed studs/pins). Pose is **derived from the connection**,
not emitted, so connectivity holds **by construction** — the exact property
`implementation_plan.md`'s growth model guarantees (pose = `parent_pos +
PORT_DIRECTIONS[dir]`; a non-connecting placement is inexpressible). But a valid *graph*
is not a working *result*: LegoGPT \cite{legogpt2025} needs physics rollback (98.8% →
24% stable without it, `ideas.md` #8); AnalogCoder **[add to bib]** keeps SPICE in the
loop; VerilogEval **[add to bib]** documents the functional-vs-syntax gap. So Layer C
**requires** a validity / physics / collision checker-in-the-loop with repair or
rollback (the GCPN/JT-VAE collision-check-and-resample of `implementation_plan.md` §5,
generalizing LegoGPT-style rollback to a per-attachment decision).

The three layers map onto a robustly supported pattern: **planner + grounded executor
beats end-to-end.** SayCan and Text2Motion **[add to bib]** report 82% vs 13% for
planner-plus-grounded-executor over an end-to-end policy; the scene-generation line
(Holodeck, SceneCraft, LayoutGPT **[add to bib]**; LayoutNUWA \cite{layoutnuwa2023})
runs LLM-plan → module-fill. This architecture is that pattern instantiated for
buildable structure, with the typed-connection graph as the interface between layers.

## 3. Why the typed/semantic structure helps — an honest two-mechanism split

The claim "structure improves validity" hides two very different mechanisms, and
reviewers treat them very differently.

**(a) Typed connection rules as a rejection/masking filter.** Much of the headline
validity gain is a decoding-time *filter*, not learning. GraphAF \cite{graphaf2020}
goes 68% → 100% valency validity via a hard valency **mask**; LegoGPT \cite{legogpt2025}
goes 37% → 99% via physics **rollback**; grammar-constrained decoding \cite{gcd2023}
enforces syntax at sample time. `ideas.md` #8 flags this bluntly ("valid by construction
is often really a decoding-time artifact"). Reviewers discount filter-based validity: it
demonstrates the constraint machinery works, not that the model learned anything.

**(b) Genuine learning / quality gains.** The defensible contribution is the model
*internalizing* the grammar and the semantics lifting quality. DiGress \cite{digress2022}
learns ~99% molecular validity with **no hard valency rules** — it absorbed the
connection grammar from data. Description-based retrieval produced a *measured* OOD gain
in Holodeck-style scene work (20.4% vs 4.11% **[add to bib]**, verify figure before use).
Text conditioning lifts sample quality across the board (`ideas.md` #1/#6; the §19 run).
To claim (b) credibly, it must be **isolated with a DiGress-style ablation**: train the
composer with the typed-connection grammar *learned from data* and compare against the
same model with the grammar enforced only as a decode-time mask. The delta between
"learned validity" and "masked validity" is the real, reviewer-proof result.

**The genuinely novel axis** is per-part **semantic text descriptions on a structured
generator.** This is essentially absent from academic work — only commercial flux.ai uses
component datasheets as descriptions, and unmeasured. A structured part-graph generator
whose nodes carry natural-language descriptions (rather than bare type IDs) is defensibly
new, and it is what lets Layer A's world knowledge flow into Layer B.

## 4. Cross-domain transfer — the ML-venue story

The strongest novelty claim is domain span. **No learned generator crosses physically
distinct domains.** UniMoMo **[add to bib]** is chemistry-only; DiGress \cite{digress2022}
is a general *formulation* but is trained per-dataset; the LEGO line
(LegoGPT \cite{legogpt2025}, BrickAnything \cite{brickanything2026},
LegoACE \cite{legoace2025}) is LEGO-only; AnalogGenie **[add to bib]** is circuits-only.

The bet: **one formulation** spanning Minecraft → LEGO → electronics, with the connection
type varying while the representation and model do not.
- **Minecraft** — 6-adjacency; connection type is *trivial* (a face touches a face). This
  is the easy case the repo already implements (`graph_data.py`), and the base training
  domain (~16k+ builds).
- **LEGO** — *typed* studs / anti-studs / clips / axles / pins. The connectivity metadata
  exists machine-readable in the LDCad shadow library (`research.md` §E.2/E.3, fetched to
  `data/lego/`), so types need not be inferred from mesh geometry.
- **Electronics** — *typed* pins / terminals; components + pins = the netlist that
  `graph_data.py` is already structurally isomorphic to, and that AnalogGenie validates
  at 93.2% valid.

The **semantic part description is the cross-domain bridge**: "load-bearing wall",
"2×4 brick", "NPN transistor" are all natural-language part descriptions over a
typed-connection graph. "One constrained-assembly formulation, three physically distinct
mediums, unified by semantic part descriptions" is the claim an ML venue rewards, and it
subsumes the weaker "two mediums" framing of `research.md` §E.2.

## 5. Getting reusable parts — the hardest open problem

The architecture presumes a **vocabulary of reusable parts**. Obtaining one for Minecraft
is the central open problem; **no Minecraft part-decomposition paper exists**, which makes
solving it a clear contribution. Options, in increasing autonomy:

1. **Supervised** — PartNet-style labeled part segmentation **[add to bib]**. Cleanest
   labels, but no Minecraft equivalent exists; would require annotation.
2. **Unsupervised co-segmentation** — BAE-Net-style branched autoencoder part discovery
   **[add to bib]**. No labels, but noisier boundaries.
3. **Combinatorial motif mining** — gSpan-style frequent-subgraph mining **[add to bib]**
   over the block+port graphs to surface recurring motifs (a wall stripe, a window frame).
4. **Coarse 3D-BPE (cheapest, already in-repo)** — run the repo's `cluster_bpe`
   (`ClusterVocab`) at a **coarser merge scale** so pieces grow from voxel-sized to
   *wall-sized*. The machinery, the round-trip guarantee, and the forced multi-block
   merges (beds/doors/double-plants, `notes.md` §19) already exist; only the merge budget
   changes. This is the fastest route to a first part vocabulary and the natural Phase-2
   feed into `implementation_plan.md`.

Part-boundary/interface definition is a first-class research risk (§8), not a detail.

## 6. Relationship to the attachment model (`implementation_plan.md`)

`implementation_plan.md` specifies the **attachment/growth autoregressive model**: seed a
piece, then attach pieces to open faces (the `graph_data.py` ports), pose derived from the
connection, per-face `CLOSE` = boundary, collision-check-and-resample at sampling. **That
model _is_ Layer C** — the checked attachment composer — operating at the piece level.

This document is the **higher-level hierarchical wrapper** around it: Layer A (LLM
planner) supplies a part-graph the growth model does not currently receive; Layer B
(native part generator) fills each planned part; Layer C (the growth model) composes them
with checking. Where the growth model grows a single build bottom-up from primitives, the
hierarchical plan grows a *castle* from *described, LLM-planned parts*. The two are
complementary and share substrate (ports, the collision checker, the conditioning stack).
**Cross-reference, do not duplicate:** tokenization, canonical order, incremental
encoding, and the phased composer milestones live in `implementation_plan.md`.

## 7. Closest relatives

- **AnalogGenie** (ICLR 2025) **[add to bib]** — pin-level typed-connection graph,
  93.2% valid topologies. The single best **template**: it is this repo's port-graph
  idea, realized in circuits, at scale. Validation of the representation, not its origin.
- **LaMAGIC** (ICML 2024) **[add to bib]** — a *structured* circuit representation beats
  naive text serialization, **ablated**. Direct support for §3's "structure over text".
- **LegoGPT / BrickGPT** \cite{legogpt2025} — native piece generation + physical
  attachment + physics check + rollback. The Layer B+C precedent (and §3(a)'s filter
  caveat).
- **StructureNet / GRASS** **[add to bib]**, **ShapeAssembly** \cite{shapeassembly2020} —
  hierarchical part graphs; ShapeAssembly has an explicit `attach` DSL — the closest
  prior to a typed-attachment part program.
- **Holodeck / SceneCraft / LayoutGPT** **[add to bib]**; **LayoutNUWA**
  \cite{layoutnuwa2023} — LLM-plan → module-fill, the Layer A+B pattern.
- **SayCan / Text2Motion** **[add to bib]** — planner + grounded executor beats
  end-to-end (82% vs 13%); the architectural justification for §2.
- **DiGress** \cite{digress2022} — *learned* graph validity with no hard rules; the
  ablation target for §3(b).
- **BrickAnything** \cite{brickanything2026}, **BrickNet** \cite{bricknet2026} — the
  attachment-tree / graph-program LEGO line.

**Verification caveat.** Several 2026 preprints are **UNVERIFIED** — BrickNet
(`bricknet2026`, arXiv 2604.22984) is flagged in `research.md`, and multiple "add to bib"
figures above are from a fast-moving literature. Keep all of them **off load-bearing
claims** until independently confirmed; do not let an unverified number anchor a result.

## 8. Risks

1. **Plan ↔ geometry grounding gap.** The LLM plan is relational; realizing it in space
   is where systems fail (VoxelCodeBench, Architect). Mitigation: a *grounded executor*
   (Layers B/C) plus the required checker — never LLM-only geometry.
2. **Part-boundary / interface definition.** What counts as "one part", and what its
   ports are, is underdetermined and pervasive (§5). Invest early; a wrong interface
   contaminates every downstream layer.
3. **Attachment coherence / seams.** Connectivity-by-construction guarantees pieces
   touch, not that the *gestalt* reads as a castle (BrickGPT's "PhysHack"; `ideas.md`
   #8; `implementation_plan.md` §9). Conditioning + data scale still carry cohesion.
4. **The artifact confound.** Validity may be a decode-time filter, not learning (§3).
   The DiGress-style learned-vs-filtered ablation is mandatory, not optional — it is what
   separates a reviewer-proof result from a discounted one.

## 9. Phased MVP

**Phase 0 — hand-defined parts, LLM-planned, rule-attached (prove the loop).**
Start with a **hand-defined part vocabulary**: `wall`, `tower`, `roof`, `gate`. Condition
the *existing* generator (`condition_resampler.py`) to produce each on demand. Have an LLM
plan a **castle as a part-graph in JSON** (Layer A, grammar-constrained). Attach parts
with **simple typed rules + a collision check** (a thin Layer C). Success criterion: a
**recognizable castle emerges from parts the model can actually make** — proving the
three-layer loop end-to-end before any hard research is attempted. Reuse the existing
representation-agnostic eval (`eval/novelty.py`, `eval/validity.py`, textured renders).

**Phase 1 — learned part decomposition.** Replace hand-defined parts with discovered ones
(§5), starting from **coarse `cluster_bpe`** because it is in-repo and round-trip-checked;
graduate to co-segmentation / motif mining if boundaries are poor.

**Phase 2 — the typed-connection composer.** Fold in `implementation_plan.md`'s
attachment/growth model as the full Layer C (typed ports, pose-derived placement,
collision-check-and-resample), and run the **learned-vs-filtered validity ablation** (§3b)
that makes the quality claim defensible.

**Phase 3 — cross-domain transfer.** Carry the same formulation to LEGO's typed
studs/pins (data already fetched, `research.md` §E.3) and, as the stretch, to a circuit
netlist — cashing the §4 novelty claim with the semantic part description as the bridge.

The through-line: **prove the loop with parts the model can make (Phase 0), then earn each
harder capability** — learned parts, the typed composer with its honest ablation, and
finally the cross-domain span that is the strongest claim.
