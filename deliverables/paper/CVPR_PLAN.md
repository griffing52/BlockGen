# CVPR 2027 plan: who does what

Written 2026-11-02 from a 9-agent review (mock review, claims audit, format/policy,
writing/figures, related work, three benchmark surveys). Deadlines (AoE):
**Nov 10** enrollment + abstract + frozen author list, **Nov 16** paper,
**Nov 23** supplementary. Mock review verdict today: Weak Reject, because "true by
construction, never shown to matter".

Done today: human study removed from paper + audit scripts (no IRB); backup of the
pre-cut sections in `iclr/.backup_20261102_pre_human_cut/`.

---

## Part 1 — Only you (decide, write, think, sign)

### Admin (today–Nov 10)
- [ ] **Today:** every co-author's OpenReview profile complete (moderation can take
      2 weeks without an institutional email); check reciprocal-reviewer eligibility.
- [ ] Confirm in the OpenReview console there is no live ICLR 2027 submission.
- [ ] Ask the UCLA compliance office about the corpus release (3D-Craft CC BY-NC,
      text2mc, GrabCraft). Until answered, the paper says "will be released" only for
      code/probes/split, not data. `9_statements.tex:49` ("settled") must change.
- [ ] Nov 10: final author list, title, abstract registered.

### Decisions that unblock agents (by Nov 4)
- [ ] **Run of record.** Paper prints `run_20260917_043807`; the audit checks
      `run_20260925_002631_..._fixed_transparency_cull`. Recommend 0925 (fixed
      renderer). Every table/number gets regenerated from your choice.
- [ ] **External generator(s) to fund:** Dream-Cubed chunks (closest competitor,
      weights public, highest value) and/or BrickGPT on the LEGO ladder (cheapest).
- [ ] **Approve the VLM-judge full run** once the agent reports its cost estimate.
- [ ] Request a 3D-FRONT license? (decide after the Objaverse mesh pilot reports)
- [ ] Case study (87% contamination retrain): restore a short version or cut it
      entirely. It is commented out but still referenced.
- [ ] Atlas bug: fix and re-render (cleaner headline) or disclose in main text.

### Claims and framing (think hard; this decides accept vs reject)
- [ ] **Headline wording.** The render metric is *weakly sensitive* (~1–2 noise
      floors, CIs exclude zero), not invisible. Pick the phrasing you will defend:
      "within ~2 noise floors vs 9–57 for cutaways".
- [ ] **Framing:** "observation-limited evaluation in 3D generation" (any metric that
      reads renders or surface samples inherits the view's null space) vs. the current
      Minecraft framing. Title follows.
- [ ] **Lemma's role:** short proposition with its missing hypotheses (unit opaque
      cubes, deterministic renderer); the headline is empirical attenuation, since the
      deployed renderer leaks.
- [ ] **Novelty boundaries, stated honestly:** gates are borrowed ideas (Heusel, Xu
      2018, Thompson 2022, O'Bray) used as *blocking admission criteria* (new);
      World-GAN's TPKL-Div is a volumetric precursor of V-KID; walkability exists
      (3D PCGRL) but not as a calibrated distributional pillar.
- [ ] "Noise floors" = null-standardized z-score; adopt that definition + report
      raw MMD² beside it.
- [ ] Response to Patch-Stitch outranking Gabled (now argued from geometry scalars
      only, `6_results.tex` Limitations).

### Write yourself
- [ ] One-sentence thesis + 3–4 contribution bullets (draft in the writing review).
- [ ] Abstract (~200 words; draft exists, rewrite after the run-of-record decision).
- [ ] Intro first paragraph (drop the proverb opener) and the Fig. 1 concept.
- [ ] Related-work positioning paragraph: borrowed vs new.
- [ ] Limitations paragraph.
- [ ] LLM Usage Statement: it is your attestation, so make it true (e.g. "every
      numeral re-derived" is currently false).
- [ ] Clear your own `\com{}` notes (`1_intro:21,24,35`, `2_related:20`,
      `3_background:32`, `5_setup:19`).

### Final pass (Nov 13–16)
- [ ] Read the CVPR build end to end as Reviewer 2.
- [ ] Anonymity sign-off (incl. `4_method.tex:76` "checkpoint from prior work").
- [ ] Submit Nov 16; supplementary Nov 23.

---

## Part 2 — Delegate to agents

GPU note: one 16 GB GPU, ~11 GB held by another job. Run GPU items one at a time.

### Running now
- [ ] **Mesh probe (step 9):** Objaverse buildings with interiors; Strip-/Fill-Interior
      vs MV-FID/KID, CLIP, surface vs all-face point clouds, V-KID. Writes
      `outputs/run_*_mesh_probe`.
- [ ] **VLM judge:** GPTEval3D-style 2AFC (exterior, cutaway, converse, catch,
      Gabled/Patch-Stitch, system-level Bradley–Terry vs BlockScore). Smoke test
      only; waits for your cost approval.

### Mechanical (start now, no decision needed)
- [ ] Port to the **CVPR 2027 author kit** (`cvpr/` is the 2026 kit); macros from
      `iclr/main.tex:52-115`; `ieeenat_fullname` bib; full-width teaser + tables;
      44 appendix refs → "Supp. Sec. X" (`xr`); supplementary as a separate PDF;
      LLM Usage Statement placed before refs; remove ICLR statements/styles.
- [ ] Fix broken refs (`sec:casestudy`, `sec:casestudy:llm`), "MV- and K-KID"
      header (`4_method.tex:36`), encoder count 4 vs 5, the "four public sources"
      that lists three, LEGO "publish no pictures" vs `fig:lego`.
- [ ] **Audit tooling:** `audit_claims.py` greps the TeX for each registered value;
      `paper_readiness.py` rebuilds before checking refs, scans `\com{}`, checks the
      leaderboard run's n (not the newest run).
- [ ] **Bibliography:** add ~40 entries with verified arXiv IDs from the surveys
      (Xu 2018, Thompson 2022, Stein 2023, Kynkäänniemi 2023, clean-fid, T3Bench,
      GPTEval3D, 3DGen-Bench, Eval3D, MATE-3D, Hi3DEval, Point-E, PointFlow, LION,
      Uni3D, ATISS, DiffuScene, PhyScene, SceneEval, World-GAN, 3D PCGRL, T2BM,
      Word2Minecraft, MineCLIP, Hervé & Salge, Sutherland, Lopez-Paz C2ST, ...);
      un-comment `deutsch2021statistical`, `mathur2020tangled`, `card2020power`;
      re-check `luan2026rendermetric` and the CMMD citation; text2mc dataset entry.
- [ ] **Figures:** strip repo identifiers (`geom-kid`, `voxel_diversity`), one font
      family, simplify the pipeline figure, reconcile the cutaway pixel value
      (0.89 / 3.9 / 0.8 per 255), fix the Fig. 1 caption, build the new Fig. 1 with
      the metric strip (after your concept), cut the probe gallery and pillars figure.
- [ ] Style-brief sweep (hedges, process narration, one name per concept).
- [ ] Draft the benchmark-comparison table (survey draft exists) for your review.

### After the run-of-record decision
- [ ] Regenerate the leaderboard, dose-response and all prose numbers from one run;
      fix the inverted 16-view sentence (`6_results.tex:56`), the √ readout
      (6.0% not 6.4%), the rescore prose (`X_appendix.tex:806`).
- [ ] Leaderboard at **n ≥ 256**, with rank intervals + pairwise flip probabilities
      from the existing bootstrap; mark ties.

### Experiments (comparison rows on the Fill-Interior ladder)
- [ ] Point-cloud suite, surface vs voxel-centre sampling (`pointcloud.py` exists;
      first verify which sampling T67 used). CPU.
- [ ] MV-FID (Inception) + CLIP-FD/CMMD rows beside DINOv2 MV-KID. Cheap GPU.
- [ ] TPKL-Div (World-GAN, ~40 lines numpy). CPU.
- [ ] Validity gates (BrickGPT/T2BM: % valid, connected, stable) and Hervé–Salge
      scalars, showing existing validity checks are interior-blind too. CPU.
- [ ] Uni3D score row. GPU.
- [ ] External generators: Dream-Cubed chunks and/or BrickGPT (after your pick).
- [ ] Discriminability: C2ST/1-NN AUROC row + minimum-detectable-corruption vs n.
- [ ] Quantified gate thresholds (Spearman ρ with CI; |z|<1 for invariance) with
      multiple-comparison correction; kernel-bandwidth sensitivity of rankings.
- [ ] Atlas-bug fix + re-render (if you choose to fix it).

### Research
- [ ] Check MineBench/VoxelBench arena API terms (external builds + human ranks).
- [ ] Verify the ~22 survey arXiv IDs given from memory.
- [ ] Nov 13: arXiv concurrency re-check (closest so far: Luan et al. 2512.01380).
