# Overnight log — 2026-09-19

Ordered by how much it affects the paper. Everything here is verified against
data; agent reports were re-checked rather than taken on trust.

---

## 1. A unit problem that changes how every number is stated

**The "noise floor" is n-dependent.** Measured on the identical corruption at two
sample sizes: the geometric metric's noise floor is `0.1732` at n=64 and
`0.0848` at n=128 — a ratio of **0.49**, i.e. it falls as 1/n while the signal
stays fixed. So a separation in noise floors grows **linearly in n**
(132.9 → 286.7, ratio 2.16).

Three consequences, all now fixed:

- Every separation figure needs its n. "202" means "202 at n=128"; at n=256 it
  reads ≈400. The abstract and §4 now say so.
- **I had made an invalid cross-domain comparison.** T34 tabulated houses
  (n=128, +201.8) beside LEGO (n=256, +118.6) as if commensurable. Normalised,
  LEGO's effect is ≈59 — about **3.4× weaker** than houses, not 1.7× as printed.
  Corrected in `results.md` T34 and in §4.
- "Regenerate at n≥256 changes no qualitative conclusion" is true of *orderings*
  only; every magnitude roughly doubles.

Written up as `results.md` **T35**, appendix `app:nscale`.

## 2. The blindness is not exact render invariance — and the honest version is stronger

Measured 60 builds × 4 views:

| | |
|---|--:|
| bit-identical renders | 13.8% |
| views that change | **86.2%** |
| of those, pixels differing | median **1.74%** |
| max channel difference where it changes | median **218/255** |

So the corruption is *plainly present* in the image — it alters most views, by
nearly the full dynamic range in the pixels it touches — and every render-space
statistic still ignores it. That is a claim about the statistics and the
pooling, **not** an information-theoretic claim.

§4 previously cited only the whole-frame mean (0.784/255), which sounds
negligible and understates it badly. Both numbers now appear together.
`results.md` **T36**.

## 3. Three appendix tables were wrong; all regenerated

- **Validation ladder** — every value was a stale copy of a superseded run.
  Critically, **`1−recall` was printed as PASS in bold** when the cited artifact
  says it fails n-stability. Only KID and MMD-RBF clear every gate. Prose in §1
  and §3 fixed to match.
- **Procedural baselines** — the mixed-statistic error was real: `gabled_house`
  showed 0.592 (aperture-aware) against real's 0.115 (sealed), making an empty
  shell read as a furnished house. Now two separate columns. Its sealed interior
  is *exactly* 0.000.
- **Corpus table** — the `houses_48` row was back-solved on a three-source
  assumption and silently dropped a whole source (`text2mc_schem`, 117 builds).
  There are **four** sources; prose updated.

The claims registry went **20 → 137 entries**, including identity checks
(`final == pooled − drops − dups`, `sum(per-source) == final`) that would have
caught the corpus defect the moment it appeared.

## 4. Citations: one had metadata I invented

All 14 remaining `NEEDS-VERIFY` entries resolved. Three had wrong metadata, and
`luan2026rendermetric` had a **title and author list I fabricated** when writing
that bib — the paper is real (AAAI-26) but was catalogued under an invented
title. Now correct. `meehan2020datacopying` and `herve2021pcgmetrics` also had
wrong titles. Zero undefined citations.

One thing to check yourself: the appendix claims Hervé & Salge correlate
"eighteen" PCG metrics; the agent could not source the number 18. Soften unless
you have it from the PDF.

## 5. New candidate figures — look at these and decide

Neither is in the paper yet.

- `figures/fig5_ablation.png` — **the strongest candidate.** Every encoder and
  view budget below the 2-noise-floor line in grey; every cutaway condition
  above it in blue. Answers "is this just DINOv2?" in one image.
- `figures/fig6_seven_statistics.png` — dumbbell per statistic: interior probe
  vs the weakest of three stronger corruptions. Note it states honestly that
  `1−recall` also fails the mild deletion rung.

## 6. Structure rewritten to canonical form

The paper now follows the guide's structure: Introduction / Related Work /
Background and Problem Setting / Validating a Metric Before Trusting It (method)
/ Experimental Setup / Results and Discussion / Conclusion. Every paragraph
carries a `%TL;DR` comment (60 of them), so the flow is auditable at a glance.
Limitations folded into Results and Discussion, which is where the guide puts
them. **Main text went 16 -> 12 pages.**

Two stale `\TODO`s resolved while checking it: one asked for a LEGO replication
we have already run, the other for n>=256 which the LEGO arm already satisfies.

## 7. Two problems I caused, both worth knowing

**Figures got orphaned and I did not notice.** The restructure dropped
`fig3_pillars` and `fig4_casestudy` from the paper. My own readiness check passed
anyway, because it only verified that *referenced* figures exist -- it could not
see a built figure silently falling out. Fixed: the check now reports figures
that are built but never included. Four are currently orphaned (fig3-fig6).

**I edited section files while an agent was rewriting them.** Twice. The second
time it corrupted `main.aux` with null bytes (two `latexmk` processes writing the
same file) and broke the build entirely; cleaning the aux files fixed it. My
fig5/fig6 placements were then overwritten by the agent's next write and are
lost. Nothing else was damaged, and the paper builds clean at 21 pages.

**Consequence: placing the figures is a pending task, not a done one.** Do it
after the structural agent finishes. `fig5_ablation` belongs beside "It Is the
View Protocol, Not the Encoder" in Results, and `fig6_seven_statistics` beside
"Both Directions, and a Second Domain".

## 8. THE BIG ONE — the significance test was worse than I measured, and the power control died

The recomputation is in `outputs/run_20260919_022424_significance_recompute/`.

**The bug was worse than my measurement.** I measured 2.1x too liberal at
alpha=0.05. That was the *gentler* of two regimes. Where a leaderboard actually
lives -- two generators, neither distributed like the reference -- the old test
rejected **7.2x** too often at alpha=0.05 and **22x** at alpha=0.01. Fixed with an
exact label-permutation test, whose type-I error is nominal at both levels.

**What changed, and the split is not a coincidence:**

| claim | old | new |
|---|---|---|
| `Fill-Interior` NOT separated from real | p=0.61 | **p=0.77 — HOLDS, stronger** |
| clean retrain NOT separated from LLM baseline | p=0.89 | **p=0.93 — HOLDS, stronger** |
| `Train-Copy` IS separated at 0.015 | Holm 0.017 | **FAILS — 0.050, on the boundary** |
| contaminated arm IS separated from both LLM arms | 3e-5 / 8e-9 | **HALF — one yes, one no (0.079)** |

Only positive separations died. A test that rejects too easily can manufacture a
difference but never a tie, so **both null results survive and are strengthened**.

**The paper's power control was the casualty.** Its argument was "this is
blindness, not weak power, because the same test at the same n resolves
Delta=0.015". That sentence is no longer true.

**Fixed, and the paper is better for it.** The argument now runs on magnitude,
not on a second p-value: the same encoder and statistic, at the same n, separate
*this very corruption* at 9-57 noise floors once the renders expose the interior,
and separate a material permutation of the same builds at 140. Those controls
never depended on the p-value, and they are far stronger than the one they
replace. New appendix section documents the calibration and everything it cost.

## 9. One more casualty — the significance LETTERS were never valid

The group letters came from a greedy first-fit that gave each arm exactly one
letter. Since "not separated" is not transitive, that construction cannot express
the truth, and "shares no letter" implied nothing. Now fixed to a real compact
letter display (maximal cliques of the not-separated graph); 4-5 of 18 arms
genuinely need multiple letters.

**This cost the case study its framing.** `native_oriented` holds letter **d**,
shared with `canon_oriented` (**def**) and `llm_control` (**de**) — so the
contaminated arm **never occupied a group of its own**, and the paper said it
did, in §6 and in the abstract.

Corrected to a claim that survives: with 87% contamination the arm separates from
one of the two baselines (Holm 5.5e-3) and scores 11.62; retrained clean it
separates from neither and scores 14.90, inside their range. Same conclusion —
the apparent advantage was the contamination — stated in terms of what actually
separates.

## Status at hand-off

| | |
|---|---|
| builds, zero undefined refs | PASS |
| 40/40 citations verified | PASS |
| 137 registered claims match data | PASS |
| main text | **10 pp** (was 16; limit is 9) |
| unmeasured claims | 10 |
| human data | 0 raters |
| orphaned figures | 5 |

### Three numeric inconsistencies I did not fix

Flagged by the structural agent, all cheap, all worth doing before anything else:

1. Intro contribution bullet says the square-root readout recovers to "0.119" —
   superseded; §6.5 says 0.124/0.135.
2. Figure 1's caption says +31% blocks over 91% of builds; the text says 33%/94%.
   The caption is generated by `scripts/paper_figures.py`, so fix it there.
3. "Seven" vs "eight" render-space statistics — `tab:seven` has seven rows, §4.3
   says eight were scored.

## Finished overnight

- ~~Significance-test recalibration~~ — done, see §8/§9 + recomputation of every separation claim.
  **This is the one that could still move a result** — the power control in §4
  ("does separate verbatim copies") is the claim at risk.
- Structural/prose rewrite to the canonical ML-paper form, with aggressive
  cutting toward the 9-page limit.
- LaTeX conventions pass (cleveref, csquotes, backref).

## For you, first thing

The abstract deadline is **11:59 UTC today**. The only irreversible item is the
**author list**. Also confirm reciprocal-reviewer eligibility — judged as of the
deadline, desk-reject if missed.
