# Runbook — the manual work only you can do

Everything automatable is done. This is the list of things that need a human,
in the order that unblocks the paper fastest. Each item says what to do, how
long it takes, and what it produces in the paper.

---

## 0. STOP — read this before sending the study to anyone

Two design problems surfaced after the study was generated. Both are cheap to
fix now and expensive to discover after your raters have gone home.

### 0a. The interior pair: run it, but as the *exterior* condition only

Raters see one exterior render per build — the same view the metric sees. On
`Real` vs `Fill-Interior` they will very likely be at chance too.

**That is a result, not a failure.** If humans cannot separate them from the
views the metric is given, the blindness is **informational**: the interior is
absent from the input, so no encoder could recover it. That is a stronger
mechanistic claim than "DINOv2 is a weak encoder", and it is the claim §4 should
make.

**What you must not say** is "humans can tell and the metric cannot." As built,
they cannot either, and the pre-registration originally implied otherwise. That
wording is now corrected in `sec/7_human.tex`.

**CORRECTION (later the same night): the positive control does work.** I first
rejected it on mean pixel difference (4.25/255), which is the wrong measurement
— the metric is KID over DINOv2 features, not pixels. Measured properly, the
quadrant cut separates `Fill-Interior` at **+9.1 noise floors** and a floorplan
slab at **+56.5**, against +1.1 uncut. If you reinstate the cutaway condition,
use the **floorplan slab**: strongest signal and most legible to a rater.

**Recommendation: run the exterior-only study now.** It is publishable as the
informational-blindness result. The cross-section condition is a good second
experiment, not a blocker for this one. The implementation is in
`scripts/human_study_export.py::render_cutaway` with `--cutaway`, kept for
whoever picks it up, and its measured weakness is documented above so nobody
re-derives it.

### 0b. The pre-registered headline statistic is out of date

`sec/7_human.tex` pre-registers **Spearman ρ against Bradley–Terry scores** as
the headline metric-vs-human number. That is the older convention. WMT24
dropped Pearson entirely in favour of **tie-calibrated pairwise accuracy**,
and Deutsch et al. (TACL 2021) show the significance tests used with correlation
have near-zero power at realistic levels — and that resampling items alone turns
a nominal 95% interval into an actual 68% one.

With **9 arms** we are well inside the range where Mathur et al. found
correlation unstable. Change the pre-registration **before** collecting, not
after: lead with tie-calibrated pairwise ranking accuracy, bootstrap over
**both items and raters**, and keep Spearman-vs-BT as a secondary check.

---

## 1. Run the rater study  ·  P0

**Use `outputs/run_20260921_human_v4/`.** 24 builds per arm, 48 trials per
session, 40 distinct sessions, 6 catch trials each.

Do not use v2 or v3. Both are missing the study's own positive control: the
`--cutaway` flag raised `NameError` on an undefined `src`, so every earlier
export silently omitted the `real#cut` vs `real@solidify#cut` arms. Without them
a chance result on the exterior pair is uninterpretable, because nothing
distinguishes "the corruption is invisible" from "the task is too hard". v4 has
18 trials on each.

| pair | trials | what it buys |
|---|--:|---|
| `real` vs `real@solidify` | 18 | the headline. Chance here means the blindness is informational |
| `real#cut` vs `real@solidify#cut` | 18 | **the positive control.** Above chance here localises the blindness to the view protocol |
| `gabled_house` vs `patchwork` | 18 | the pre-registered prediction: both metrics rank patchwork higher, we predict raters reverse it |
| `real` vs `uniform_random` | 15 | catch pool; 6 per session, fixed quota |

Cutaway arms are only ever shown against other cutaway arms, so the view is
never the most salient difference on screen. Verified: 0 mixed trials.

```bash
.venv/bin/python scripts/human_study_serve.py --run outputs/run_20260921_human_v4
```

### How many raters, and what each buys

Power to detect a real human preference on one pair, from an exact binomial
test. `p` is the true fraction preferring the better arm; `0.50` is chance:

| raters | judgements on the key pair | p=0.65 | p=0.70 | p=0.75 | p=0.80 |
|--:|--:|--:|--:|--:|--:|
| 8 | 34 | 31% | 55% | 79% | 94% |
| **12** | **51** | 58% | **84%** | 96% | 100% |
| 16 | 67 | 70% | 92% | 99% | 100% |
| 20 | 80 | 72% | 94% | 100% | 100% |

**12 raters is the point of diminishing returns.** It reaches conventional 80%
power for a moderate preference (p=0.70); going to 20 buys almost nothing at
that effect size and does not rescue a weak one (p=0.65 stays underpowered
throughout — detecting that needs ~150 judgements, i.e. ~35 raters).

**More raters beats longer sessions** for the same total judgements, because
answers from one person are correlated: the analysis bootstraps over raters as
well as items, so ten people × 48 trials carries more information than five ×
96. Session length is capped by attention, not statistics.

### Assign sessions explicitly

`study.html?s=0` through `?s=39`. Without `?s=`, the page picks at random and
two raters can collide — with 12 raters drawing from 40, that is about a 1-in-6
chance, and it wastes a session. Keep a list of who got which number.

**The study is generated and verified.** It lives at:

```
outputs/run_20260918_human_v2/study.html      ← send this file to raters
outputs/run_20260918_human_v2/answer_key.json ← keep private, never send
```

It is a single self-contained HTML file (5.3 MB, images embedded). It works
from a file:// path, an email attachment, or any static host. No server.

### 1a. Working remotely — serve it, don't open the file

`study.html` is 2.5 MB of embedded images; VS Code will not open it in an editor
tab, and you do not want it to. Serve it instead:

```bash
.venv/bin/python scripts/human_study_serve.py --run outputs/run_20260921_human_v4
```

It prints `http://127.0.0.1:8777/`. **VS Code Remote forwards that port
automatically** — the notification offers "Open in Browser", or look under the
PORTS tab. Without VS Code, forward it yourself:

```bash
ssh -L 8777:127.0.0.1:8777 <this-host>     # then open http://127.0.0.1:8777/
```

The server does more than display the page:

- **Responses save straight to disk** as `rater_<session>.json` in the run
  directory — the format `human_study_analyze.py --responses` already reads. No
  copy-paste, no JSON mangled by a mail client, no rater who closes the tab and
  loses forty judgements. This is the single biggest source of loss in a
  multi-rater study and it is now gone.
- `http://127.0.0.1:8777/status` shows who has submitted and how many trials
  each gave, so you can chase stragglers.
- **`answer_key.json` is never served** (verified: it 404s). A rater who pokes
  at the server cannot read the conditions.
- Submissions never overwrite: two people handed the same session produce
  `rater_s3.json` and `rater_s3_1.json`.

### 1a-bis. Getting it to your raters

Three options, in the order I would try them:

1. **Each rater on their own machine, via the file.** `study.html` is entirely
   self-contained, so email it, or drop it in Drive/Dropbox and share the link.
   It works from a `file://` path with no server — the page falls back to
   showing the JSON for them to send back. Simplest, but you are back to
   copy-paste.
2. **Everyone hits your server.** Best data-handling, needs the port reachable.
   `--host 0.0.0.0` exposes it on your network; a tunnel (`cloudflared tunnel
   --url http://localhost:8777`) exposes it publicly. **Read the warning in the
   script's docstring first: there is no authentication.** The stimuli are
   served to anyone who can reach it. For a short window with colleagues that is
   usually acceptable; decide deliberately rather than by default.
3. **University web space.** If `math.ucla.edu` gives you public `~/public_html`,
   copying `study.html` there is the cleanest public option — but responses go
   back to copy-paste, since there is nothing to POST to.

If in doubt: option 1 for raters outside your network, option 2 for anyone who
can reach the host.

### 1b. Before you send it — 30 seconds

```bash
.venv/bin/python -m pytest tests/test_human_study.py -q     # 8 checks
```

These verify the things that would silently ruin the study: that every session
has its full catch-trial quota, that no arm label reaches the page, that sides
are balanced, and that 10 raters cover all 28 pairs.

### 1c. Recruit 8–12 people

They need no domain knowledge. They must be **naive to the hypothesis** — do
not explain the project, do not mention interiors, do not mention that some
builds have been altered. Send them this and nothing more:

> Hi — I need ~20 minutes of pattern-matching help for a research project.
>
> Open the attached file in a browser. You'll see two Minecraft-style buildings
> side by side, 40 times. Each time, click the one that looks more like a
> **real building someone actually made**. Go with your gut; there's no trick
> and there's no right answer I'm hoping for. When you're done the page gives
> you a block of text — paste it back to me.
>
> Please do it in one sitting, on a laptop or desktop rather than a phone.

Assign each person a distinct session with `study.html?s=0`, `?s=1`, … through
`?s=39`. This is optional — without it the page picks one at random — but it
guarantees even coverage and lets you tell who has finished.

**Do not do a full session yourself and include it as data.** You know the
hypothesis and the pre-registered prediction. Rate a session if you want a feel
for the task, then discard it. Reviewers ask about this.

### 1d. Collect and analyse

If you used the server, the files are already there. If raters sent JSON back,
save each as `outputs/run_20260918_human_v2/rater_<name>.json`. Either way:

```bash
.venv/bin/python scripts/human_study_analyze.py \
    --key outputs/run_20260918_human_v2/answer_key.json \
    --responses outputs/run_20260918_human_v2/rater_*.json \
    --scorecard outputs/run_20260917_043807_bench_canonical-split_native_arms_oriented_vs/scorecard.json
```

It prints the exclusion count, a Bradley–Terry ranking, and per-metric
agreement — i.e. §7's results table, ready to paste.

### What the numbers will be, and what each one means

Three pairs are deliberately oversampled. With 10 raters you get roughly:

| pair | judgements | what it decides |
|---|---|---|
| `Real` vs `Fill-Interior` | ~33 | **the informational-blindness test — see §0a.** On exterior views humans are expected to be at chance *too*, which shows the interior is absent from the input rather than mishandled by the encoder. Only the cutaway condition can show humans separating them. |
| `Patch-Stitch` vs `Gabled-House` | ~35 | the pre-registered prediction. Both metrics rank patchwork higher; we predicted raters reverse it. Either outcome is reportable. |
| `Voxel-AR` vs `LLM-FT` | ~20 | the pair the metrics call a statistical tie |

The exclusion rule is 5 of 6 catch trials correct. A rater answering at random
survives that **10.9%** of the time — the analyser prints this, and it should
go in the paper, because it is the honest statement of how well the screen
works. It tolerates one misclick by a genuine rater.

> **One caveat to write down.** `Real` vs `Fill-Interior` was added to the
> priority set *after* the original pre-registration, on the grounds that it is
> the most informative pair available. Say so in the paper. A pre-registration
> you amend in the open is stronger than one you quietly pad.

---

## 2. The encoder / view-budget ablation  ·  P0  ·  ~2 h, no training

A reviewer will ask whether the blindness belongs to DINOv2-at-4-views or to
render-and-embed as a class. This is feature extraction only — no model
training — and it is the difference between a narrow result and a general one.

Four conditions, all scoring the same `solidify` probe:

| condition | why |
|---|---|
| CLIP ViT-B/16 instead of DINOv2 | is it this encoder? |
| an ImageNet CNN (ResNet-50) | is it self-supervised features specifically? |
| 8 and 16 views instead of 4 | is it the view budget? |
| **a cutaway / cross-section view** | **the positive control** |

The cutaway is the important one. We expect it *would* resolve `solidify`, and
showing that turns the mechanism from asserted into demonstrated: the metric is
not broken, it is being shown the wrong pictures.

I have not written this because it needs a decision from you: extending
`blockgen/eval/bench/features.py` to take a pluggable encoder is a ~100-line
change and I would rather you sanity-check the interface than discover it is
wrong after the runs. Say the word and I will build it.

---

## 3. Re-run the mixture ladder  ·  P0  ·  ~20 min

`scripts/bench_doseresponse.py:59,124` uses decimated real data as its "bad"
pool, including for the quadratic/√ analysis. Since decimation is no longer a
quality reference, those coefficients need re-deriving against
`uniform_random` or `shell_box`. The algebra is distribution-independent, so
the conclusion is not at risk — only the numbers in Table 5.

---

## 3b. Second domain: LEGO — measured, viable, half a day  ·  P1

The dataset search came back with a result better than expected, and one trap.

**Use `data/lego/stabletext2brick/` — it is already on disk.** 47,389 brick
structures on a 20³ grid, MIT-licensed, and measured directly:

- **48.7%** of structures contain at least one enclosed void
- `blockgen.eval.bench.probes.solidify` runs on them **unchanged**
- median **+9.5% blocks** conditional on having a void (vs. +31% for our houses)
- filtering to the hollow-rich categories (car, bus, bottle, camera, tower,
  sofa, …) gives **17,684 structures** — comfortably past the protocol's
  `min_ref = 256`

Weaker effect than Minecraft, but real, and the payoff is larger than the
number suggests: **BrickGPT evaluates with CLIP score and DINOv2 cosine on
rendered images** — precisely the metric class we indict. Demonstrating the
blindness there hits a live SOTA protocol rather than a strawman.

Work: a ~15-line regex rasterizer from the brick text format into `Structure`,
then map brick types to block ids and the existing renderer and bench run as-is.

**The trap: do not use ShapeNetVox32.** It was downloaded and measured — mean
enclosed voids is **0.0** across twelve of thirteen categories, because the
standard voxelization is already flood-filled. The fill probe is a literal no-op
there and a referee who checks will catch it. (ModelNet40 has much larger
effects — dresser void/occupied 1.56 — but ships with *no license at all*, so it
would have to be a download script, never redistributed.)

## 3c. Reframe the claim: visual-hull blindness  ·  P0, free

This is the highest-value item on the list and costs only writing.

Mainstream mesh and SDF generators cannot commit our corruption, because their
training data was solidified before they ever saw it (OccNet → mesh-fusion →
ManifoldPlus → CLAY, Hunyuan3D). So "interior deletion" reads as a voxel-domain
curiosity. But the *mechanism* is fully general and already documented:

- **SparseFlex §3.3 states it verbatim**: "standard rendering supervision
  typically focuses only on the visible surface, neglecting the interior of the
  shape" — and their fix is to move the near clipping plane inside to render
  cross-sections, i.e. our cutaway ablation.
- **Meta's AssetGen §6.2** removes "interior faces that are invisible from any
  exterior viewpoint" with a 512-viewpoint pass — an unscoreable failure at
  production scale.
- TRELLIS.2 now advertises "fully-enclosed surfaces"; RoSI opens on models
  "missing their interior structures". The field is moving toward interiors.

**Claim visual-hull blindness**: a render-and-embed pipeline scores only the
visible boundary, so sealed rooms, internal faces, double shells and occluded
concavities are all unscored. That survives contact with mainstream 3D
generation, where "interior deletion" does not — and it converts the "one
domain" objection into "the only domain where the corruption is *measurable*,
with AssetGen as evidence it bites elsewhere."

`references.bib` currently contains **none** of the mainstream 3D-gen line
(Trellis, Hunyuan3D, CLAY, SparseFlex, AssetGen, OccNet). A 3D-vision reviewer
will look for exactly those.

## 4. Regenerate headline arms at n ≥ 256  ·  P1  ·  ~1 h mostly unattended

The protocol's own `min_n` is 256; everything reported is at 128. No
qualitative conclusion changes, but the paper should not ask for something it
does not do.

---

## 5. Decisions only you can make

- **Author list, affiliation, acknowledgements.**
- **Code/data release.** The paper promises a release. Decide the license and
  whether the corpus can be redistributed — it pools three third-party scrapes,
  and their terms are not all the same. This is a real blocker for a
  benchmark paper and is worth an hour of checking before submission.
- **Ethics / rater compensation statement.** If raters are unpaid colleagues,
  say so. Some venues require a statement either way.
- **Venue.** See `README.md`.

---

## Where each piece lands in the paper

| you do | fills |
|---|---|
| §1 rater study | `sec/7_human.tex` (currently `\UNMEASURED`), plus a result in §4 |
| §2 encoder ablation | `sec/8_discussion.tex` limitation → promoted to a result |
| §3 mixture re-run | Table 5 in `sec/5_leaderboard.tex` |
| §4 n ≥ 256 | every table, unchanged conclusions |

Run `.venv/bin/python scripts/paper_readiness.py` at any point for the current
state of all of this.
