# BlockGen — running task list

One table, priority-ordered. Anything with a number attached cites where the
number came from, so a priority can be argued with rather than guessed at.

**P0** blocks a claim we intend to make · **P1** wanted for the paper ·
**P2** quality / correctness, not blocking · **P3** parked idea

Areas: `render` `bench` `agentic` `ontology` `lab` `deploy` `data` `model`

| id | pri | area | task | why / evidence |
|---|---|---|---|---|
| R1 | **P1** | render | **Non-cube blocks are drawn as full cubes.** `renderer/textured.py::build_mesh` emits six axis-aligned faces per voxel and nothing else, so stairs, slabs, fences, panes, doors, torches, carpets and plants all render as solid blocks with the right texture on the wrong shape. | **27.5% of `houses_32` placements** (765k / 2.78M, 162 block types) are non-cube blocks. Top offenders: oak stairs 2.8%, oak slab 2.4%, spruce stairs 2.3%, oak fence 1.4%, glass pane 1.4%. This is a *systematic* distortion of every render the eval and the human study look at — the same class of problem as T25. |
| R2 | **P2** | render | **Unmapped blocks fall back to a flat colour cube**, which reads as wool. `FACE_TEXTURES` has no entry, so `_solid_texture` paints `_color_for`. Redstone wire (55), redstone block (152), redstone torch (75/76), rails, levers, buttons, pistons, beds, signs, glazed terracotta, crops. | **1.33% of `houses_32` placements** (37k, 102 types) — small here *because houses are not redstone builds*. The full corpus has 1,287 "Redstone Device Map" structures (T2), where this is the dominant material. Cheap to fix: it is a lookup table, not geometry. |
| R3 | **P2** | render | Torches, levers and wall-mounted things need *attachment* geometry, not just a smaller box — a torch on a wall is not a torch on the floor. Split from R1 because it needs the block-state bits, which the corpus only partly carries. | Called out as the hard case; do it after R1's slab/stair/pane shapes land. |
| O1 | **P0** | ontology | Re-run the ontology arms at **n ≥ 128** (`ont_none`, `ont_mined`, `ont_shuffled`), ~$2. | T27 pilot at n=12 is a plumbing check; every interval overlaps. Nothing about the ontology can be claimed until this runs. |
| O2 | **P1** | ontology | Implementation #2: the ontology as a **validator in the repair loop** (directional block with no facing, `not-full` block used as a wall, unsupported gravity block, adjacency the corpus never contains) instead of as context. | Costs zero prompt budget, reuses a loop that already works, and is the piece that transfers cleanly to LEGO stud/anti-stud legality and schematic pin rules. |
| O3 | **P2** | ontology | Authored-ontology arm (`ont_authored`), ideally LLM-generated offline so it is automatic rather than 70 hand-typed rows. | Completes the provenance ablation: mined vs priors-written-down, not just mined vs nothing. |
| B1 | **P0** | bench | Regenerate every headline arm at n ≥ 256 (Track E at n ≥ 128) and score them together. | Standing item from T26; no arm comparison in the repo is currently resolvable. |
| B2 | **P1** | bench | Human 2AFC on `ont_none` vs `ont_mined` pairs. | The renders and `geom_kid` **disagree** on these two arms (T27): the eye prefers `ont_mined`, the metric slightly prefers `ont_none`. That is a metric-validation question, and the lab's Compare page already feeds the same Bradley-Terry analysis. |
| B3 | **P2** | bench | `palette_size` is in `dataset_stats`, not a pillar — nothing penalizes a materially impoverished build. | Every agentic arm uses ~10 distinct blocks per build against the corpus's 21.5 (T27). A build can be half as rich as real and lose nothing. |
| L1 | **P2** | lab | Open `/ontology` in a real browser and click through it. | The page's JS was written and reviewed but never executed — there is no headless browser in this environment. |
| A1 | **P1** | agentic | Skills / tool calling in the agentic approach *(inherited)*. | The MCP / palette-API implementation: knowledge pulled on demand instead of pasted into the prompt. Prerequisite: `providers.py` has no tool-call path yet. |
| A2 | **P2** | agentic | Add notes about scale to the prompt — e.g. a player is two blocks tall *(inherited)*. | Cheap prompt fix for buildings that are the wrong size for their doors. |
| A3 | **P2** | agentic | Novelty of agentic builds against the corpus is still unmeasured. | Listed as a known limit in `docs/agentic.md`; the `.npz` output already drops into `eval/novelty.py`. |
| D1 | **P1** | deploy | Native gen/mod: doors always break when placed; beds are not always a full bed block — probably placing over an existing bed *(inherited)*. | Two-part blocks need their partner placed in the same tick; this is the same block-state gap as R3. |
| M1 | **P3** | model | Graph LLM *(inherited)*. | Parked. |

*(inherited)* items are from the previous free-form `todo.md`, kept verbatim in
meaning; their priorities are a first guess and are the ones most worth arguing
with.
