# Paper structure and abstract (draft for the Nov 10 registration)

Written 2026-11-04 against the new evidence (n=256 leaderboard, VLM judge, 3D-native
rows, BrickGPT, Dream-Cubed, NN dissociation). Numbers are from the leaderboard of record
`run_20261103_090315_bench_n_256_leaderboard_of_record` unless marked. This file is a
proposal only: nothing in `sec/*.tex` was changed for it.

## The one-sentence thesis

A metric can only rank what its observations retain, and every standard way of observing a
generated 3D structure (exterior renders, visible-surface points, an image judge) drops the
interior, which is the part that makes a building, a LEGO model or a scene usable.

The paper used to be "render metrics can't see Minecraft interiors". The new evidence makes it
**"observation-limited evaluation"**: render metrics, 3D point-cloud metrics and a VLM judge all
fail the same probe for the same reason, in three domains. That answers the mock review's
"true by construction, never shown to matter". It matters because the field's strongest
alternatives fail too, and because it changes how published generators rank.

## Title options

1. **Judged by the Cover: Observation-Limited Evaluation of 3D Generative Models** (keeps your
   book-cover opener as the title instead of the first sentence)
2. What the Camera Cannot See: Interior Blindness in 3D Generation Metrics
3. A Metric Ranks Only What It Observes: Validating Evaluation for 3D Structure Generation

## Contributions (3 bullets)

1. **Observation-limited evaluation, made testable.** *Complement probes* alter a structure while
   keeping every exterior-visible face bit-identical (Fill-Interior, Hollow-Shell), or keep the
   geometry and change only appearance (Material-Shuffle, Monochrome). Under an opaque renderer no
   exterior-view metric can separate them (proposition, with its hypotheses stated).
2. **The blindness holds for every observer the field uses.** Seven render statistics, five image
   encoders and up to 16 views; Uni3D, ULIP-2, Chamfer and EMD on visible-surface points; a GPT-4o
   judge. Across Minecraft, LEGO and Objaverse meshes. Changing only the observation (cutaways,
   interior-face sampling) recovers the signal, so the cause is the observation, not the encoder.
3. **A remedy that is validated before it ranks.** V-KID, a CPU-only voxel kernel distance with the
   complementary blind spot. A protocol that admits a metric to a leaderboard only after it passes
   ordering, invariance and n-stability gates; Fréchet distance and the biased MMD fail them. A
   leaderboard that scores published generators: BrickGPT on LEGO, and Dream-Cubed, which the within-set twin gate disqualifies.

## Section plan (8 pages)

| § | section | pages | content | status |
|---|---|---|---|---|
| — | Fig. 1 teaser | 0.4 | real house vs Fill-Interior: identical exterior renders, different cutaways; metric strip (MV-KID 0.35, Uni3D-vis ≈0, GPT-4o chance, V-KID 444) | new concept; needs you |
| 1 | Introduction | 1.0 | interiors matter → field evaluates by observing exteriors → Dream-Cubed's own limitation quote → probe and the headline → it holds for every observer → the remedy and the 3 bullets | rewrite paragraphs 1, 3, 5 |
| 2 | Related work | 0.6 | render-and-embed; 3D-native metrics (now *tested*, not just cited); metric auditing (Xu 2018, Thompson 2022, borrowed gates); VLM judges (GPTEval3D) | add VLM + 3D-native positioning |
| 3 | Observation-limited evaluation | 0.6 | observation function, null space, complement probes, the proposition | mostly exists (3_background) |
| 4 | Metric validation | 1.0 | probes, MV-/V-KID, gates, aggregation (worst pillar), disqualification | exists |
| 5 | Setup | 0.5 | corpora (houses_32, LEGO ST2B test, Objaverse 290), arms incl. Dream-Cubed and BrickGPT, n=256 | update |
| 6.1 | Every observer misses the interior | 1.2 | **headline table: observers × probes** (render KID / Uni3D / ULIP-2 / CD-EMD / GPT-4o / V-KID × Fill-Interior, Hollow-Shell, Material-Shuffle, cutaway, controls) | new; agent drafting the pieces |
| 6.2 | It is the observation, not the encoder | 0.8 | cutaways, all-face vs visible sampling, NN double dissociation (100% vs 26% source retrieval), window leakage (89% leak, still at chance) | partly exists |
| 6.3 | Three domains | 0.5 | LEGO (incl. BrickGPT replication of solidify), Objaverse meshes | exists in parts |
| 6.4 | Complementary blindness forces the aggregation; the leaderboard | 1.0 | leaderboard at n=256 with Dream-Cubed (caveat pending the template audit) and BrickGPT on LEGO; rank intervals; raw MMD² beside noise floors | agent updating |
| 6.5 | Limitations | 0.3 | atlas/window leakage, judge is one model, Objaverse n=96, gates borrowed | rewrite |
| 7 | Conclusion | 0.2 | | |
| — | LLM Usage Statement | (free) | | yours to attest |

Cut from the main text (to the supplement): case study (87% contamination), probe gallery, pillars
figure, full encoder ablation, full mesh tables, VLM protocol, BrickGPT reproduction, Dream-Cubed
extraction, NN figure.

## Abstract, version A (recommended: observation-limited framing, about 230 words)

> Generative models of 3D structure are ranked by metrics that observe only part of what they
> score: rendered exterior views, points sampled from visible surfaces, or a judge shown images. A
> metric cannot rank what its observations discard, and for buildings, LEGO models and indoor
> scenes the discarded part is often the interior, which decides whether a structure can be
> entered, assembled or navigated. We make this limit measurable with complement probes that alter
> a structure while keeping every exterior-visible face identical. Filling the sealed rooms of real
> Minecraft houses adds a median 33% more blocks, yet a multi-view DINOv2 kernel distance places the
> filled corpus 0.35 noise floors (units of the metric's own real-versus-real sampling spread) from
> real data, in the same significance group. The blindness holds for seven render statistics, five
> image encoders and up to sixteen views, for Uni3D and ULIP-2 point-cloud encoders and Chamfer and
> earth mover's distances on visible-surface points, and for a GPT-4o judge, across Minecraft, LEGO
> and Objaverse meshes. Changing only the observation, by cutting the views open or sampling
> interior faces, recovers the signal. We introduce V-KID, a CPU-only voxel metric that separates
> the probe by 444 noise floors and is blind exactly where render metrics see, and a protocol that
> admits a metric to a leaderboard only after it passes gates on ordering, invariance and
> sample-size stability, which Fréchet distance fails. Applied to the published generator
> BrickGPT, the protocol finds a geometric gap 4.4 times its render gap.

## Abstract, version B (closer to the current draft: Minecraft-led, about 200 words)

> Generative models of 3D structure are evaluated by rendering exterior views, embedding them with
> a pretrained encoder, and measuring a distribution distance to real data. Such a protocol cannot
> see whether a structure has an inside, yet interiors are what make a building walkable, a LEGO
> model buildable and a scene navigable. We measure the gap with a probe that fills every sealed
> room of real Minecraft houses with the house's own material: a median 33% more blocks and an
> unchanged exterior. A multi-view DINOv2 kernel distance places the filled corpus 0.35 noise
> floors from real data, inside its significance group, while a kernel distance on voxel
> descriptors (V-KID) places it at 444. The same probe defeats seven render statistics, five
> encoders, sixteen views, the Uni3D and ULIP-2 point-cloud encoders on visible surfaces, and a
> GPT-4o judge, on Minecraft, LEGO and Objaverse; cutaway views recover it with the same encoder.
> We pair V-KID with a validation protocol that admits a metric to a leaderboard only after gates on
> ordering, invariance and sample-size stability, and use it to score published generators,
> BrickGPT and Dream-Cubed.

## Claims in the abstracts, and where each is backed

| claim | number | source |
|---|---|---|
| median +33% blocks | 33% | existing paper claim (audit_claims registry) |
| MV-DINO-KID on Fill-Interior | 0.35 noise floors, significance group `a` | n=256 run, appearance pillar 0.3537 |
| V-KID on Fill-Interior | 444 | n=256 run, geometry pillar 444.36 |
| 7 statistics, 5 encoders, 16 views | | 0925-protocol ablation runs (existing) |
| Uni3D / ULIP-2 / CD / EMD blind on visible surfaces | \|z\| ≤ 1.0 Objaverse, ≤ 1.6 Minecraft | notes §68, results T81 |
| GPT-4o at chance | 0.53 [0.50, 0.56] | notes §67, results T80 |
| LEGO | render −0.09, geometry +40.5 | notes §70, results T83 |
| interior faces / cutaways recover it | Uni3D all-face 4.6–22; cutaway KID 9–57 | §68; existing ablation |
| Fréchet fails n-stability | | existing gate results |
| ~~Dream-Cubed closest to real among generators~~ | rank 2 of 13 | **DROPPED; Dream-Cubed is now DQ by the within-set twin gate** (notes §78, run_20261104_214146_bench_twin_gate_rescore; audit 2026-11-04, `outputs/analysis/dreamcubed_audit/README.md`): 59% of its builds have a near-twin in the set (IoU>=0.8) vs 0.8% for real; largest cluster = 28 copies of one hut; 36% roofless enclosures, 19% fragments; unlearned 1.21 vanilla village templates through the same extraction score BlockScore ~195-203 (rank 3, above every BlockGen arm). Both bench gates miss it. |
| BrickGPT geometric gap 4.4× render gap | 52.1 vs 11.8 | T83 |

**Caution on 0.35 vs 444.** Both numbers are in noise floors at n=256. They can't be compared
with the old 0.38 / 202, which were at n=128 under the pre-fix renderer (run 0917). The paper
should define a noise floor once (the null-standardized z-score) and print raw MMD² beside it in
the table.

## Decisions for you before Nov 10

1. Framing: version A (observation-limited, three domains, every observer) or version B
   (Minecraft-led). I recommend A: it is the version the new evidence supports, and it answers
   the reviewer.
2. Title (options above).
3. Fig. 1 concept: the metric strip above, or your own.
4. Whether "five image encoders" and "sixteen views" stay in the abstract (they are older
   0925-protocol results; still valid, but the abstract is long).
