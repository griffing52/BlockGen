# Audit — what I checked, what I found, what you must decide

Three things here: how to re-run the checks yourself, what my own audit found,
and the abstract question that is due tomorrow.

```bash
.venv/bin/python scripts/audit_claims.py --verbose   # every number vs its data
.venv/bin/python scripts/paper_readiness.py          # submission blockers
.venv/bin/python -m pytest tests/ -q                 # ~909 tests
```

---

## 1. The abstract — decide this tonight

It is now **366 words** (was 443; typical is 250–350). It is registered
tomorrow and, per ICLR's rules, the final version must stay close to what you
register — so the framing locks even though the text can be edited until the
25th.

**The one substantive change I made**: the old abstract led with
"$0.000$ vs a real floor of $-0.001$". Those are raw KID values, and they
understate the result — a reader has no idea whether $0.001$ is a lot. It now
leads with the pair that means something:

> the render metric places the probe **0.38 noise floors** from real; the
> geometric metric places it **202**.

Same measurement, same run, stated in the unit the paper defines. 0.38 is
"under half its own sampling noise", which a reader can evaluate immediately.
Both verified against the run of record (`audit_claims.py`).

**What I did not change, and you should sanity-check:** the abstract claims the
suite is released. That is a promise about licensing you have not yet verified
(see §4 below). If the corpus cannot be redistributed, that sentence must
change before the paper deadline — not before the abstract.

A backup of the previous version is at `sec/0_abstract.tex.bak`.

---

## 2. What I audited myself, and the one error I found

**`scripts/audit_claims.py` is new and is the durable part of this.** It is a
registry of 19 numeric claims, each with the code that recomputes it from a
scorecard. All 19 currently pass. Adding a number to the paper means adding it
there; a claim nobody can recompute is a claim nobody can check.

### The error: a noise-floor unit that is only defined within a run

I could not reconcile the "202 noise floors" figure — the sources said 202, 205,
133 and 67 in different places. Resolved: **202 is correct**, and the others are
different quantities.

The trap is that KID's sampling noise falls with $n$, so a separation quoted in
noise floors is only meaningful when the difference and the noise floor come
from the **same run at the same $n$**. Dividing the $n{=}128$ difference by the
$n{=}64$ noise floor gives 128, which is meaningless and looks plausible. The
audit script now enforces this: `Card.sep()` reads both numerator and
denominator from one scorecard, and the module docstring explains why.

Two leftovers to fix, both cosmetic but both wrong in print:

| where | says | should say |
|---|---|---|
| `blockgen/eval/bench/fast.py` (solidify recipe string) | 205 spreads | 202 |
| `results.md` T25b | "67 sd" | correct, but it is the $n{=}64$ ladder — label it |

### What I verified as correct

- Every abstract number against `run_20260917_043807_bench_...`.
- The exact-zero invariance claims (material shuffle, monochrome) are
  *bit-identical*, not approximately zero — the paper says "exactly" and it is.
- The case study's numbers: clean retrain 14.898, LLM baseline 15.006,
  contaminated arm 11.624. The *values* are right; the **interpretation was
  not** — see §3a, the comparators are themselves contaminated.
- The cross-run consistency check (LLM baseline reproducing to three decimals).
- The quadratic ratio (0.079, not the 0.31 an earlier draft printed).

---

## 3. VERIFIED AUDIT FINDINGS — act on these

Two audits returned. I re-verified every critical finding myself rather than
relying on the reports; what follows is what I could confirm by running code.

### 3a. FIXED — the LLM comparators are also contaminated (was undisclosed)

Both prompted-LLM arms were fine-tuned on a $90/10$ shuffle that puts **53 of
399 canonical test builds (13%)** in their training data. The repo says so in
`protocol.py`, `train_llm_geobias.py` and `tests/eval/test_protocol.py`. The
paper did not.

Worse: in the run of record **no arm declares a training split at all** (every
one `<absent>`), and the run is still stamped `official: true, reasons: []`.
The provenance gate §6 credits **never fired** — it was written afterwards, in
response to this audit.

Fixed: new §6 subsection discloses it. Two consequences, opposite directions —
the within-architecture finding (same model, clean split, moves group) is
untouched, but the cross-track "tie" is no longer a clean claim. Note the bias
direction *flatters the baselines*: clean LLM arms would score worse, so the
clean native arm is if anything under-credited. The paper now says "not
separated" rather than "tied".

### 3b. FIXED — "every room is deleted" was false

Verified on the calibration fixture: `solidify` fills only **sealed** air. A
room with a 1x1 or 2x2 doorway survives completely — the probe is a literal
no-op on it. On 200 real builds the sealed share of interior volume has median
$100\%$ but mean $85.5\%$, and $6\%$ of builds are untouched. Abstract and
§4 now say "fills every *sealed* void, median $+33\%$ blocks, $94\%$ of builds
altered", with a paragraph stating what the probe does not reach.

### 3c. OPEN, IMPORTANT — the significance test is anti-conservative

Measured directly against `paired_delta` on an exact null:

| level | measured type-I error | nominal | inflation |
|---|--:|--:|--:|
| $\alpha=0.05$ | 0.107 | 0.05 | **2.1x** |
| $\alpha=0.01$ | 0.067 | 0.01 | **6.7x** |

Cause: it subsamples $0.8n$ *without replacement* and uses that spread directly
as the SE, with no subsampling rescale — while `ladder._noise_floor` applies
exactly such a correction, which is what makes this a bug rather than a choice.
Holm over 153 pairs operates near $\alpha=3\times10^{-4}$, where it is worst.

**Direction of harm, and why the headline is safe:** a claim that two arms are
*not* separated is conservative under a liberal test. Interior blindness is of
that form. **Every claim of separation is at risk**, including the abstract's
power control ("does separate verbatim training copies").

`tests/eval/test_compare_calibration.py` now pins this measurement so a fix or a
regression is visible.

**What I could not settle.** I tried three times to recompute the specific
`real_test` vs `train_verbatim` comparison from the cached DINOv2 features and
failed to reproduce the bench's pipeline — `real_test` should be $\approx 0$ and
my reconstruction gives $0.0197$ even after matching the mean-pool and the L2
normalisation. I am not reporting a verdict on that control from a pipeline I
could not reproduce. **The right fix is to add a permutation option inside
`compare.py` and re-run the bench**, not to reconstruct it externally.

### 3d. FIXED — an appendix table with no source

The descriptor ablation had no code, no script and no recorded run anywhere in
the repo. Removed rather than reprinted; the §4 claim that leaned on it ("not a
size detector in disguise") is now a `\TODO`.

### 3e. FIXED — verbatim copies rank third, not second

`solidify` beats them on the render metric, which is the paper's own point.

### 3f. STILL OPEN — from the code audit, not yet acted on

- **Three appendix tables carry numbers matching no recorded run**: the render
  ladder (every value differs from the cited artifact), the procedural-baselines
  table, and the corpus `houses_48` row. Regenerate or remove.
- **`1-recall` is marked PASS but the cited ladder artifact fails it** on
  n-stability; `results.md` already recorded that correction.
- **Nothing is committed.** Nine modules are uncommitted and all three audit
  scripts are untracked; `git_dirty: true` in the run of record. No commit
  reproduces the paper.
- **The split can silently re-randomise** and keep its name; `split_sha` is
  recorded and never checked by any consumer.
- **`blockgen/eval/perceptual.py` still has the blank-frame failure** that was
  fixed only in `bench/features.py`; three scripts import it.
- The "202" has **±15% seed sensitivity** (185–274 across 12 seeds) — do not
  quote three significant figures.

### 3g. NEW RESULT — the LEGO replication, and it upgrades the claim

**Verified against the scorecard myself.** On 47,389 MIT-licensed LEGO brick
structures (second domain, second representation, second literature), at n=256:

| | `mv_dino_kid` | `geom_kid` |
|---|--:|--:|
| `real@solidify` separation | **−0.01 floors** | **+118.6 floors** |
| (control) material shuffle | **+140.0** | +0.0 bit-identical |
| (control) monochrome | **+685.8** | +0.0 bit-identical |

The point estimate is *below* the real floor. The control is inside the table:
the render metric is wide awake on this corpus — it resolves a material
permutation at 140 floors — and cannot see the interior fill at all.

**And the in-domain ladder generalises the claim from one metric to a class.**
All seven render-space statistics — KID, MMD-RBF, Fréchet, coverage, density,
precision, recall — have `S6_resolves_solidify = False`, while all seven resolve
material shuffle and 40% chunk deletion:

| metric | solidify | chunk_del_20 | mat_shuffle |
|---|--:|--:|--:|
| kid | **2.0** | 12.3 | 132.3 |
| fd | **0.5** | 10.8 | 72.4 |
| 1−density | **0.1** | 7.2 | 14.0 |
| 1−precision | **−0.4** | 9.0 | 28.0 |

**This is a different and much stronger paper.** Not "DINOv2-KID is blind" — an
encoder-and-statistic complaint a reviewer dismisses as a bad choice of either —
but **the render representation discards interior structure, so every statistic
computed on it inherits the blindness.** Two domains, two representations, seven
statistics. It also partly pre-empts the encoder ablation: that varies the
encoder, this varies the statistic.

Written up as `results.md` T34. **Not yet in the paper** — it deserves to be in
§4 and the abstract, and that is a framing decision.

## 4. What the remaining subagents are doing (results pending)

Five ran in parallel while you were away. **I have not seen their results yet**
and will not summarize them until I do.

| agent | question |
|---|---|
| math audit | Is the mixture identity right as stated? Does `sqrt()` of a possibly-negative unbiased KID make sense? Are "significance groups" from pairwise tests transitive, or does the letter notation mislead? Is a max over pillars with different sampling distributions well-defined? |
| code-vs-claims | Every number in the paper traced to code. Silent-failure hunt. Which claims no test covers. |
| LEGO replication | Does the blindness reproduce on a second domain and a second representation? |
| encoder ablation | Is this DINOv2-at-4-views, or every render-and-embed protocol? Plus the cutaway positive control. |
| mixture + hollow probe | Re-derive the quadratic coefficients on a legitimate bad pool; test the converse (hollowing) probe. |

The two that can **change the paper's claims** are the encoder ablation (if more
views recover the interior, the claim weakens to "the standard protocol is
under-viewed") and the mixture re-run (the current coefficients come from a
decimation pool the project has disavowed).

---

## 5. What only you can decide

1. **Author list — irreversible tomorrow.** See `DECISION.md`.
2. **Reciprocal reviewing** — eligibility is judged as of the abstract deadline;
   missing it is a desk reject.
3. **Read `sec/9_statements.tex`** (the AI use statement). It is mandatory, and
   a misstatement there is a Code of Ethics matter. I wrote it; you have to own
   it.
4. **Licensing.** The corpus pools three third-party scrapes. The paper promises
   a release. Verify the upstream terms permit redistribution.
5. **Page budget.** 13 pages against a hard 9 — a stated desk-reject condition.
   References and appendix are unlimited, so this is moving, not cutting. The
   plan is in `README.md`; the remaining calls are which arguments to compress.

---

## 6. How to review the paper itself

Read in this order, which is the order a reviewer will:

1. `main.pdf` Figure 1 — if the teaser does not land, nothing else matters.
2. The abstract, against §1's claims.
3. §4 (the blindness) — this is the paper. Check that the significance
   statement is a *test*, not interval overlap, and that the dose-response
   distinguishes "flat" from "weak".
4. §6 (the case study) — the strongest section for a benchmark venue, because
   NeurIPS E&D's reviewer guidance literally asks "does it shift conclusions if
   applied to existing benchmarks?"
5. §8 limitations — I wrote these to be real rather than decorative. If any
   reads as hedging you would not defend, cut it.

Things I would push back on if I were reviewing:

- **"Visual-hull blindness" is not yet the framing.** The paper still says
  "interior". Mainstream mesh generators cannot commit this corruption (their
  data is solidified before training), so "interior deletion" reads as
  voxel-specific. The reframe is free and in `RUNBOOK.md §3c`.
- **n=128 against a protocol that asks for 256.** Defensible, but a reviewer
  will notice the paper does not follow its own rule.
- **The human study is one pilot session so far**, and it is yours, so it is not
  data.
