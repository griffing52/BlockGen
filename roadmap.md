# BlockGen → BuildGen: Roadmap (2026-07-07)

The paper thesis, one sentence: **guaranteed-buildable generation with diverse
(non-cuboid) parts, trained on open data, transferred from an abundant medium
(Minecraft), evaluated with honest novelty accounting** — every clause missing from
all three incumbent LEGO papers. Full cited landscape: `research.md §E`.

---

## 1. Where we are (done, validated)

**Data.** Four corpora unified onto one legacy vocab (`utils/block_remap.py`, 97%
state coverage): PM schematics crawl (5,866 labeled), GrabCraft (5,621 @48, category
labels), 3D-Craft (2,586 houses + human build order), text2mc (11k). Curated
cross-corpus house sets: `houses_32` = 2,661 / `houses_48` = 2,971 (quality gates +
enclosed-air "house-ness" + variant-aware dedup). Full label inventory: notes.md §3.5.

**Models / proven ablations** (notes.md §7): phase4 grammar-aware PE = best PE;
in-loop adjacency-constrained decoding = validity 1.0 at no quality cost; 3D-BPE
cluster tokens resist memorization; category conditioning = best cohesion lever;
more data → cohesion (validity 3×); two-stage diffusion helps shape only. Honest-
novelty protocol (D4 aug + val split + NN-IoU + dup rate) — none of the LEGO papers
have an equivalent.

**Infra.** Real-texture headless renderer (~0.5–3 s/img) + dense grid mosaics +
checkpoint→samples→textured-grid script; curation/eval tooling; 16 GB RTX 5070 Ti.

## 2. What others have done (verified 2026-07-07, research.md §E)

| Work | Venue | Representation | Parts | Guarantee | Data |
|---|---|---|---|---|---|
| BrickGPT (LegoGPT) | **ICCV'25 Best Paper** | text tokens + physics rollback | **8 cuboid** | stability ✓ | StableText2Brick 47k **released** |
| LegoACE | SIGGRAPH Asia'25 | native brick tokens, CLIP/DINOv2 + DPO | **9,314 types** | **none** (82% connected) | LegoVerse 55k **PRIVATE** |
| BrickAnything | arXiv 5/26 | tree tokens + reward-DPO + rollback | **8 cuboid** | validity ✓ | — |
| BC-Bench | arXiv 6/26 | MLLM step assembly | diverse | — (<1%→15% success) | 102 files (copyright) |

Also: Minecraft-only discrete diffusion = workshop-tier (Scaffold Diffusion →
NeurIPS SPIGM WS); LLM-code-gen wave (VoxelCodeBench: executable ≠ spatially
correct); **no standard benchmark exists**; judge-guided generation published for
continuous 3D only; naive VLM judges ≈ zero human agreement on fine-grained 3D.

**The open intersection: guaranteed validity × diverse parts.** Plus: open data
(SOTA's dataset is private), cross-medium transfer (unclaimed), novelty accounting
(absent), assemblable build *sequences* (untouched).

## 3. The plan

### Phase 0 — now (gating + cheap parallel wins)
1. **LEGO data pipeline** (the gating item; we just built this muscle):
   - Fetch LDraw **OMR** (official sets, part+pose, per-file CC BY 4.0) +
     **StableText2Brick** (47k, public) + LDraw part library + **LDCad Shadow
     Library** (machine-readable stud/clip/axle/pin connectors; subset coverage).
   - Parse MPD/LDraw → (part, color, pos, rot) lists; join shadow-lib connectors →
     **typed-connection graphs**; report: #models, #parts covered by connectors,
     % models fully-covered (vocab = covered subset).
2. **Minecraft runs that feed the paper** (launch immediately, they're cheap):
   - phase4 + constrained decoding combo (the two proven winners, never combined).
   - Pool-pretrain (~14k builds) → labeled finetune on `houses_32` — this *is* the
     transfer-learning evidence, not just a quality lever.

### Phase 1 — the core bet
3. **Typed-connection tokenizer + AR**: emit (part-type, parent-connector,
   own-connector, discrete orientation); pose *derived* from connection ⇒ invalid
   placements inexpressible (validity by construction — stronger than
   BrickAnything's rollback). Grammar-aware PE = phase4 generalized to the
   connection tuple. Collision via voxelized part proxies. Train on OMR +
   StableText2Brick.
4. **Open benchmark** ("BuildBench"): released parser + data recipe + metrics
   (connection validity, empirical stability via PyBullet, novelty protocol,
   render-CLIP fidelity) across LEGO + Minecraft. Converts LegoVerse's non-release
   into our differentiator.

### Phase 2 — the ML-venue framing + stretch
5. **Cross-medium transfer**: Minecraft pretraining (6-adjacency = trivial
   connector grammar) → LEGO finetune; data-efficiency curves vs from-scratch.
6. **Assemblable sequences** (stretch): guarantee an insertion-feasible build
   order; 3D-Craft human order supervises the Minecraft side.
7. Tooling (not novelty claims): VLM/MineCLIP judge best-of-N → DPO; LDraw
   rendering for figures (extend our pyrender path with LDraw meshes).

**Stake the claim early**: arXiv/workshop preprint as soon as Phase 1 shows
validity-guaranteed diverse-parts generation working at all.

### Risks (research.md §E.4)
Shadow-lib coverage gaps (restrict vocab, report); hinges/continuous angles
(defer; discrete rotations first); no stability solver for clips/axles (claim
connection-validity guarantee, stability empirical); field speed (3 major papers
in 12 months) → publish early.
