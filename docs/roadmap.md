# Roadmap — LEGO, electronics, and improving quality

## Why the design generalizes

A build is (a) typed placements with coordinates/orientation and (b) a connectivity
graph. Both target media are the same shape of object, so the three
[representations](representations.md) port directly:

| medium | "block" | "coords" | variant / material | connectivity |
|---|---|---|---|---|
| Minecraft (now) | block id | voxel x,y,z | data value (wood/wool/orient) | 6-neighbour adjacency |
| **LEGO** | part id (LDraw) | stud grid x,y,z | colour code | stud ↔ anti-stud mate |
| electronics | component | board x,y (+layer) | value / footprint | net (pin ↔ pin) |

The AR token format generalizes to *piece tokens*
`[BOS,(X,Y,Z,PART,ORIENT,COLOR)*,EOS]`; the block+port graph generalizes to LEGO stud
graphs / SPICE-style netlists; the novelty eval is medium-agnostic.

## What's closest to generating LEGO structures today

**Track A (autoregressive tokens).** It's LegoGPT-aligned, is our **best house generator**
(NN-IoU 0.568), and its token stream maps most cleanly onto LEGO piece tokens. Track C
(graph) is the closest *connectivity* model (studs = ports) and is the natural second
step for enforcing buildability. Lead the LEGO demo with **AR + a validity gate**.

## LEGO demo plan (pitch-ready milestones)

1. **Data** — LDraw parts library + a model corpus (OMR / Rebrickable, or LEGO-provided
   data). Parse LDR (part, 3×4 transform, colour) into the existing `Structure`/graph
   schema with a `Piece` type. *Ask of LEGO: curated build data + part metadata.*
2. **Tokenizer / normalizer** — snap to the LEGO stud grid (built-in scale normalization);
   canonicalize orientation to the 24 axis-aligned rotations; delta-coordinate encoding to
   keep sequences short.
3. **Model** — AR on one set category ("learn one model class, then expand" — the milestone
   already demonstrated for houses) + a **stud-mate + stability validity gate**.
4. **Proof** — novelty grid (sample vs nearest real set) + **buildability rate** (fraction
   physically connectable & stable) — the LEGO-relevant headline metric.
5. **Framing** — "text-free generative design that produces *new, buildable* sets, with
   provable novelty and variant-preserving curation."

## Improving generation quality

Concrete experiments, in rough priority order (details/evidence in the repo `notes.md`):

1. **Factored `(family, variant)` block embeddings.** Measured: flat embeddings don't
   learn birch≈oak (within-family cos-sim ≈ random). Share an `E_family[id]` across all
   woods/wools + a small `E_variant[data]`. Expect better material agreement and rare-block
   generalization. *Caveat:* `data` also encodes orientation for stairs/logs — split it.
2. **Validity gate** (biggest visible-quality lever). Connectivity now, support/gravity
   later, with rejection + rollback during AR sampling. Samples currently fragment
   (validity ≤ 0.38).
3. **Decouple shape from palette.** Generate geometry in an abstract palette, then paint
   materials with a second head — reframes "same house, different wood" as one shape ×
   one palette draw.
4. **Delta-coordinate AR tokens.** Shorten sequences so AR reaches 16–24³ without the 12³
   fidelity loss, and bias toward local structure.
5. **Flow-matching as the default diffusion sampler** (implemented) + continuous-time
   corrector steps (re-mask & resample) + a length/occupancy-conditioned reveal schedule,
   to fix MaskGIT under-fill.
6. **Category / class conditioning.** Prepend a class token (metadata enables it) or train
   per-category — "a house" is far easier than "any of 15 map types".
7. **Scale up** `max_dim` / sequence length once the narrow case is solid.
8. **`block2vec` co-occurrence init** — only if 1 & 3 are insufficient.

## Related work & novelty

- **LegoGPT** — text-free AR brick generation + physics-validity rejection. We share the
  AR + validity stance; differ in (a) Minecraft voxels → cluster/piece tokens, (b) a
  **three-way comparison** (AR vs discrete diffusion vs graph VAE) on one shared rep,
  (c) explicit **material-variant-aware curation**.
- **MaskGIT / D3PM** — discrete/absorbing-state diffusion (Track B).
- **Discrete flow matching** — our alternate sampler.
- **block2vec** — co-occurrence block embeddings (candidate for factored embeddings).
