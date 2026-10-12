# ICLR submission — working draft

**Title:** Render-Based Metrics Cannot See Inside a Building: Structural
Blindness in the Evaluation of 3D Structure Generation

```bash
cd deliverables/paper/iclr && latexmk -pdf main.tex
.venv/bin/python scripts/paper_readiness.py            # what is still blocking
```

Builds clean: no undefined references, no undefined citations, all 26 citations
verified against arXiv / the official proceedings on 2026-09-18.

## Naming

Every name a reviewer sees is defined once, as a macro in `main.tex`, and never
falls back to a repository identifier. `Table~\ref{tab:arms}` in §3 defines all
sixteen arms before any of them is used.

| paper name | was | what it is |
|---|---|---|
| `MV-KID` | `mv_dino_kid` | multi-view render KID (DINOv2-B, 4 ortho views) |
| `V-KID` | `geom_kid` | KID on D4-invariant voxel descriptors |
| `Fill-Interior` | `real@solidify` | the probe |
| `Material-Shuffle`, `Monochrome`, `Train-Copy`, `Single-Mode` | `real@*`, `train_verbatim` | probes |
| `Patch-Stitch`, `Gabled-House`, `Hollow-Box`, `Random-Voxels` | `patchwork`, … | baselines |
| `Voxel-AR`, `Voxel-AR-U`, `LLM-FT`, `LLM-FT-Geo` | `canon_oriented`, … | submissions |

Section headings are standard noun phrases (Introduction / Related Work /
Benchmark Design / … / Limitations / Conclusion) rather than sentences.

## Page budget — the open structural problem

ICLR limits main text; references and appendix do not count. Main text is
currently **12 pages** (down from 16). Already moved to the appendix: the
metric-validation ladder table, the baselines table, the dose table and the
case-study table (both duplicated figures — the same numbers twice), the
interior-counting method, the gate-defect audit, and the full human protocol.

Remaining ~3 pages, in the order I would cut them:

1. **§5 is still the longest section** (p.6–8). The `Patch-Stitch` discussion and
   the self-validation paragraph can each lose half their length.
2. **§6 case study** (p.8–10) can be one page. The "why this is not a
   retraction" argument is important but currently three paragraphs of what
   could be one.
3. **§2 Related Work** is dense but can lose ~15 lines.
4. **§1** can lose the last contribution bullet if it is stated in §6 anyway.

Do *not* cut: Figure 1, the blindness result, the two-blindness table, or the
limitations section.

## The claim

Render-and-embed metrics (Render-FID / multi-view KID) read a structure's
*surface* and are blind to whether it has an interior. We prove it with a probe,
localise it with a dose-response curve, build the tier that sees it, show that
tier is blind in the complementary direction, and combine them into an aggregate
that the known degenerate strategies lose. The closing case study is the
benchmark deleting our own headline result.

**This is a benchmark/evaluation paper, not a generative-modelling paper.** We do
not claim a SOTA generator, and §6 says so explicitly.

## Status by section

| file | state |
|---|---|
| `sec/0_abstract.tex` | drafted, all numbers sourced |
| `sec/1_intro.tex` | drafted |
| `sec/2_related.tex` | drafted; bib entries marked `NEEDS-VERIFY` must be checked |
| `sec/3_setting.tex` | drafted — corpus, splits, ladder, three self-found defects |
| `sec/4_blindness.tex` | **drafted — the core result** |
| `sec/5_leaderboard.tex` | drafted — BlockScore, baselines, the quadratic/sqrt result |
| `sec/6_casestudy.tex` | drafted — contamination → clean retrain → ranking collapse |
| `sec/7_human.tex` | **design only. No data. Marked `\UNMEASURED`.** |
| `sec/8_discussion.tex` | drafted — limitations are real, not decorative |
| `sec/X_appendix.tex` | corpus, descriptor ablation, repro, three negative results |

Every number traces to `results.md` or a scorecard in `outputs/`. Nothing is
invented. `\TODO{}` and `\UNMEASURED{}` render in red; setting `\draftfalse` in
`main.tex` turns any remaining one into a **build error**, so an unmeasured
claim cannot ship by accident.

## What the prior-art check changed (2026-09-18)

A full novelty sweep ran against the central claims. **Neither claim is dead, but
two things in the draft were wrong and are now fixed.**

**1. An arithmetic error, in the abstract.** The draft said an arm failing on a
quarter of its prompts "reads only 31% as bad as one failing on all of them."
That conflated two quantities: 0.043/0.546 = **7.9%** is how bad it reads; 0.31
is the same number against a *linear* scale (the last column of the table). A
referee recomputes this first. Corrected everywhere.

**2. The quadratic identity is not ours.** \citet{binkowski2018kid} — the KID
paper we already cite — states it in Appendix B.2, and both they and Heusel et
al. already plot their metrics against a contamination fraction. Appendix D.3 of
the same paper proves FID is *not* polynomial in the mixture weight, so any
"and FID" generalisation would have been false. The claim is demoted from "we
show" to "we make explicit", attributed, and scoped to squared IPMs. **The
√-reporting recommendation appears genuinely novel** and is what we now claim.

**3. Dream-Cubed already states our hypothesis in prose** — and it was in our own
bibliography, described only as "reports a render-based FID." Their limitations
section says render-FID "struggles to assess internal structure (e.g., whether a
building has a coherent interior)," and their human study finds raters agreeing
with that FID only 54.3% of the time. This does not scoop the measurement — they
have no probe, no dose-response, no null and no alternative — but burying it
would have looked like we missed it. The intro now **leads** with their caveat:
the field's largest study flags the limitation and ranks models with the metric
anyway; we make it falsifiable and show it is worse than the caveat suggests.
Their 54.3% is free corroboration.

**Verdict: the central claim is novel as a measured result, not as an
observation.** That is a defensible and honest position, and the paper now says
so in those words.

**4. A cheap experiment that makes the result much harder to attack.** Our probe
fills interiors. The field's *actual* failure mode is the converse — hollow
shells, which is what surface-fitted mesh and implicit generators emit. A
`Hollow-Shell` probe (erode each build to its surface layer) uses the same
machinery and would show both directions are invisible. Flagged as a `\TODO` in
§8. I would do this before the second domain.

**5. Citations added, and deliberately blocked.** Ten must-cite/must-distinguish
works were added (O'Bray ICLR 2022 is the highest-risk reviewer; Borji's
desiderata predate our gates; Chong & Forsyth for n-stability; Bai et al. for
FID gamed by memorization in a real competition). They carry `NEEDS-VERIFY`, so
`paper_readiness.py` now blocks until they get the same title/author check the
first 26 received. Two have unresolved given names.

## Blocking work, in priority order

### P0 — human data (§7)

The single item that decides whether this is submittable. A metrics paper
without human agreement gets rejected; the ladder proves the metrics rank
*synthetic corruption*, not that they agree with people.

Everything except the raters exists: instrument, analysis, pre-registration,
and an end-to-end validation on simulated raters (14 sessions, 2 answering at
random — it excluded exactly those two and recovered the injected ordering).

**Collection plan** (~2 evenings of your time, ~30 min per rater):

```bash
# 1. Generate the stimuli + the self-contained page
python scripts/human_study_export.py --per-arm 12 --out outputs/run_<stamp>_human/

# 2. Raters work in BlockLab's Compare page (labels stripped, enforced by grep)
python -m tools.lab --open      # → /compare

# 3. Analyse: exclusion → Bradley-Terry → agreement
python scripts/human_study_analyze.py --lab-db outputs/lab/lab.db \
    --scorecard outputs/run_20260917_043807_bench_*/scorecard.json
```

Target: **8–10 raters naive to the hypothesis, 40 trials each, ≥320 usable
trials after catch-trial exclusion.**

Four things to get right, because they are what a reviewer checks:

1. **Naive raters.** Do not tell them the hypothesis, do not tell them the
   `patchwork` vs `gabled_house` prediction, and do not rate a full session
   yourself — your own sessions are a sanity check, not data. Report how many
   sessions were run and how many excluded.
2. **Use the batch export, not only the lab log.** The lab's Compare page has no
   catch trials; the exported study does. Without catch trials the exclusion
   rule in the pre-registration cannot be applied, and that rule is the
   paper's answer to "how do you know they were paying attention".
3. **Prioritise the pairs that carry information.** With ~10 raters you cannot
   cover all 36 pairs evenly. Weight sampling toward: (a) `patchwork` vs
   `gabled_house` — the pre-registered prediction; (b) `real@solidify` vs
   `real_test` — *the paper's central claim, put to humans*: if raters can tell
   solidified from real while the metric cannot, that is the blindness
   demonstrated on people rather than on a statistic, and it belongs in §4, not
   §7; (c) `canon_oriented` vs `llm_control` — the pair the metrics call a tie.
4. **Report inter-rater agreement**, not only the Bradley-Terry fit.

> The `real@solidify` vs `real_test` pair is worth calling out separately: it is
> currently the highest-value trial in the whole study and it is *not* in the
> pre-registered set. Add it explicitly and say it was added, rather than
> letting it look selected after the fact.

### P0 — the second-encoder ablation (§8)

A reviewer will ask whether the blindness is DINOv2-at-4-views or
render-and-embed in general. Needs: CLIP and an ImageNet CNN as alternative
encoders; 8 and 16 views; perspective as well as orthographic. Add a
**cutaway/cross-section view condition** as the positive control — we expect it
*would* resolve `solidify`, which turns the mechanism from asserted into shown.

Cheap (feature extraction only, no training) and it converts the paper's main
claim from "with this encoder" to "with this class of protocol".

### P1 — n ≥ 256

The protocol's own `min_n` is 256; the reported runs are at 128. No qualitative
conclusion changes — the effects are tens of noise floors wide — but the paper
should not ask for something it does not do.

### P0 — re-run the mixture ladder with a non-decimation bad pool

**Found while drawing fig2.** `scripts/bench_doseresponse.py:59,124` builds the
mixture axis — and the quadratic/√ analysis that is one of the paper's headline
contributions — by mixing real builds with **canon-8 decimated** ones as the
"bad" distribution. Since decimation is no longer something we treat as a
quality reference, the coefficients in the paper's Table 5 are keyed to a pool
we have disavowed.

The algebra `MMD²(αP+(1−α)Q,P) = (1−α)²MMD²(Q,P)` is distribution-independent,
so the *conclusion* is not at risk — but the numbers are. Re-run with
`uniform_random` or `shell_box` as the bad pool. A `\TODO` is in place at that
table in `sec/5_leaderboard.tex`.

### P1 — a second domain (§8)

Replicate the `solidify` probe on LDraw/brick data (`legogen/`, already in the
repo). Does not need a generator, only the probe and the two metrics. This is
what upgrades the claim from "Minecraft houses" to "3D structure".

### Figures — done, regenerable

```bash
python scripts/paper_figures.py --all      # fig1 needs the project venv (EGL)
```

Every figure reads its numbers out of `outputs/` at draw time, so a new bench
run redraws the paper rather than desynchronising from it.

| figure | what it shows |
|---|---|
| `fig1_teaser` | real vs solidified: renders + cross-sections + both scores. **The paper in one image** — §1, page 1. |
| `fig2_dose_response` | both metrics in units of their own noise floor; the interior panel is flat |
| `fig3_pillars` | the two blindnesses as mirror images across the pillars |
| `fig4_casestudy` | contaminated → clean retrain moves a Holm group |

Palette is CVD-validated (worst adjacent pair ΔE 24.7 protan / 33.6 normal) and
colour follows the entity — `geom_kid` is blue in every figure, `mv_dino_kid`
orange in every figure.

Still missing: a figure for the human results (§7), once there are any.

## Venue note — superseded

See `DECISION.md`. Short version: ICLR 2027 abstract is due **tonight**
(Sep 18 AOE), the author list locks permanently at that deadline, withdrawal is
free until Sep 25, and CVPR 2027 is not a usable fallback. NeurIPS Evaluations &
Datasets (~May 2027) is.

## Old venue note

ICLR 2027's deadline is imminent and the human data does not exist. Two honest
options: submit to **NeurIPS Datasets & Benchmarks** (arguably the better home
for this paper than ICLR main track), or take the extra cycle and use it for the
human study, the encoder ablation and the second domain. The LaTeX here is ICLR
format; the content ports unchanged.

## Before submitting

- [ ] `\draftfalse` in `main.tex` → build must succeed with zero `\TODO`
- [ ] Verify every `NEEDS-VERIFY` entry in `references.bib` against the real record
- [ ] Main text ≤ 9 pages (currently 16 pp including appendix + refs — trim)
- [ ] Anonymized code/data release URL in `app:repro`
- [ ] Confirm no de-anonymizing text; `\iclrfinalcopy` stays commented
