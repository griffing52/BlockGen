# Scope, page triage, and what ships where

Written 2026-09-20, in answer to: *should this be one paper with everything —
benchmark, models, full suite — or not?*

---

## 1. The recommendation: one paper, and it is the metric paper

**Do not fold the models in.** Not because there is no time, though there isn't;
because it would make the paper *worse*, and it would do so in a way reviewers
punish hard.

The paper currently has one claim, and it is unusually strong for a benchmark
paper: **render-and-embed metrics cannot see whether a structure has an inside,
and under an opaque renderer they provably never can.** That claim is carried by
an impossibility lemma, two domains, seven render-space statistics, four
encoders, view budgets to sixteen, a positive control that recovers the signal
(cutaways), and a case study where applying the suite overturned this project's
own headline result. A reviewer can hold that in one sentence.

Adding a models contribution attacks it from three directions:

1. **The models are this project's weakest work, and the paper already says so
   — as evidence.** The AR model memorized ~31% of training builds; the
   best-scoring arm turned out to be 87% test-contaminated; the prompted-LLM
   baselines emit structures one block thick (`llm_control` 100×1×1,
   `llm_bias` 5×1×251). Inside the metric paper these are *findings*: they are
   what a validated suite catches. Promoted to contributions, the same facts
   become a generative-modelling paper whose models memorize, leak and emit
   sticks. The identical evidence flips sign depending on the frame.
2. **It changes who reviews it and against what bar.** "Here is a metric the
   field uses and here is why it is broken" is judged against measurement
   quality. "Here is our generator" is judged against BrickGPT, DreamCubed and
   the rest. The first bar we clear; the second we do not, today.
3. **There is no room.** Main text is 12 pages against a hard 9 *before* any of
   the seven new figures are placed. A models section is another 2–3.

The models stay in the paper as what they already are: the **arms the suite
scores** and the **subject of the case study**. That is not a demotion — the
case study is the strongest section in the paper for a benchmarks audience,
precisely because the model being audited is ours and the audit cost us the
result.

**What this defers.** A models paper is a real paper and this work sets it up
well: a pinned group-aware split, a validated metric, a provenance gate, and a
leaderboard with honest baselines. It is a better second paper than it is a
third of a first one.

---

## 2. The page problem

**Resolved 2026-09-24.** Main text now ends on **page 10**, the target the author set.
The cuts that got it there, in order of size:

| move | pages |
|---|---|
| `fig:pipeline`, `fig:lego`, `fig:influence`, `fig:contamination` to the appendix | ~1.6 |
| §6 prose 3,724 -> ~2,450 words | ~1.6 |
| the squared-statistic subsection to `app:failrate`, leaving one clause in §6.4 | ~0.2 |
| every main-text caption cut to at most five typeset lines | ~0.5 |
| `tab:leaderboard` from nine columns to six, caption 165 -> 60 words | ~0.3 |
| §1, §2, §4, §5 compressed ~20% each | ~0.9 |

Main text keeps three figures (`fig:teaser`, `fig:ladder`, `fig:cutaway`) and two tables
(`tab:arms`, `tab:leaderboard`). `fig:contamination` went to the appendix rather than
shrinking because it stops being legible below 0.90\linewidth, and there was no page
budget for it at that width.

The section below is the earlier triage, written against a 9-page limit and a 12-page
draft. Keep it for the reasoning; the numbers in it are stale.

## 2b. The page problem, as first triaged

Main text **ends on page 12**; the limit is **9** and overrunning is a stated
desk-reject condition. Current layout:

| pp | contents |
|---|---|
| 1–2 | abstract, intro, Figure 1 |
| 3 | related work |
| 4 | background, method §4.1 |
| 5 | gates §4.3, BlockScore §4.4 |
| 6 | setup, results §6.1, `tab:arms` |
| 7 | view ablation §6.2, `tab:seven`, `fig:ablation` |
| 8 | both directions §6.3, `tab:ablation` |
| 9 | `fig:lego` (nearly a full page) |
| 10 | complementarity §6.4, `tab:leaderboard`, failure rate §6.5, case study §6.6 |
| 11 | human eval §6.7, limitations §6.8 |
| 12 | conclusion |

**And none of `four_views`, `cutaway`, `ladder`, `pipeline`, `descriptor`,
`contamination` or `samples` is placed yet.** So the real problem is not "cut
three pages", it is "cut three pages *and* find room for the figures that carry
the argument".

### The resolution: figures replace prose and tables, they do not accompany them

This is a visual claim about a visual failure. Every one of the new figures was
built to *be* an argument, not to illustrate one, so each can displace the text
that currently makes the same point more slowly.

| place | put in | take out | net |
|---|---|---|---|
| §4.2 | **`pipeline`** | the prose defining $\phi = \mathcal{E}\circ R$ twice (§3 and §4.2), and the V-KID dimension list | ≈ −0.3 pp |
| §6.1 | **`ladder`** | `tab:seven` (8 rows × 4 cols) and the paragraph reading it aloud | ≈ −0.2 pp |
| §6.2 | **`cutaway`** | `tab:ablation` and half of §6.2's prose | ≈ −0.3 pp |
| §6.6 | **`contamination`** | the two paragraphs enumerating 346/399, 53/399, and the before/after scores | ≈ −0.2 pp |
| §6.3 | `fig:lego` at `0.85\linewidth` | — | ≈ −0.2 pp |

To the appendix, in full, with main-text pointers: **`four_views`** (its job is
rebuttal, and the cutaway carries the same argument *plus* the fix),
**`descriptor`**, **`samples`**, and the existing `fig:gallery`, `fig:dose`,
`fig:seven`, `fig:pillars`, `fig:casestudy`.

Prose cuts that owe nothing to figures, in the order I would make them:

1. **§6.8 limitations → appendix, leaving three sentences.** Currently ~0.6 pp.
2. **§2 related work → 0.75 pp** by merging the three render-critique paragraphs
   into one compare-and-contrast paragraph.
3. **§4.4's BlockScore justification → appendix** (`app:blockscore` already
   holds the measurements); keep the rule and the one-sentence reason.
4. **§6.5 failure rate → 0.4 pp.** The identity is not ours; the recommendation
   is one sentence and one number.

Estimated landing: **9.0–9.3 pp**, which is close enough that the last trim is a
typesetting pass rather than an argument cut.

---

## 3. What ships where

### Main paper (9 pp)
The claim, the proof sketch, the probe, the two metrics, the gates, the
leaderboard, the case study, the human study, and five figures:
`fig:teaser`, `pipeline`, `ladder`, `cutaway`, `contamination` (+ `fig:lego`).

### Appendix (unlimited)
- `app:hull` — the lemmas, the four-renderer table, **and the disclosure of our
  own atlas artefact**. This one is not optional: we report a renderer defect
  that inflates a number we quote.
- `app:blockscore` — the max-of-$k$ null, the two pillars whose unit is not a
  noise floor.
- `app:nscale` — the unit is $n$-dependent *and estimated*; the 201.8 vs 286.7
  reconciliation.
- `app:significance` — the calibration fix and what it changed.
- `app:data` — provenance, licensing, what we do and do not redistribute.
- `app:interior`, `app:corpus`, `app:descriptor`, `app:negative`, `app:repro`.
- The remaining figures listed above.

### Code and data release
- The suite, probes, baselines, gates, and the pinned split as index lists.
- **Not** the corpus: GrabCraft forbids pictures of blueprints, text2mc's builds
  come from PlanetMinecraft. Ship the rebuild script, source URLs and hashes.
- The LDraw → voxel exporter (`legogen/renderer/ldraw_voxel.py`), which is new
  and independently useful.
- `scripts/audit_claims.py`, which is the honest centrepiece of the release:
  every number in the paper recomputed from a recorded run.

### Explicitly out, and say so in limitations
Generative models as a contribution; the agentic/Track-E work; the ontology
pilot; MinecraftACE; the deploy/mod demo; anything about generation quality.

---

## 4. Open items that gate submission

| item | owner | state |
|---|---|---|
| ≥8 raters on `run_20260919_human_v3` | **you** | 0 collected |
| institutional determination (release + human subjects) | **you** | not started |
| page count 12 → 9 | me | plan above |
| 10 remaining `\TODO`/`\UNMEASURED` | me | in progress |
| anonymized code URL + commit | you, at submission | — |
| commit everything (no commit reproduces the paper) | you to approve | 50+ files dirty |
