# Supplement restructure recommendation (CVPR 2027)

Reviewed 2026-11-07 against `supp.pdf` built 2026-11-06 23:32 (29 pages, 27 sections A..AA)
and the main text `sec/0_abstract.tex` .. `sec/7_conclusion.tex`, `sec/teaser.tex`,
`sec/9_statements.tex`. No `.tex` file was edited.

## 0. Summary

| | current | proposed |
|---|---|---|
| top-level sections | 27 (A..Z, AA; needs `alphalph`) | 13 (A..M; plain `\Alph` again) |
| pages | 29 (about 18.5 text + about 10 pages of single figures stranded at the end) | about 18 to 20 (lean path to about 15 in Sec. 5) |
| figures | 15 | 10 (cut 5 that duplicate a table) |
| tables | 18 + 3 inline | 15 + 2 inline (cut `tab:blindness`; corpus table shrinks) |

The biggest single page win is mechanical: every `figure*` sits in "Additional Figures"
(Y) and floats one per page to pp. 19 to 29. Moving each figure into the section it backs,
and putting `\FloatBarrier` (package `placeins`) at each `\section`, recovers about 4 to 5
pages before any prose is cut.

## 1. Inventory of the current supplement

Page = page where the section heading lands in the current PDF. "Main refs" = number of
`\suppsec/\supptab/\suppfig` references from the main text (incl. teaser and statements)
to the section or to a float inside it.

| sec | title | label | pp. | main refs (from) | claim it backs | reviewer would read? |
|---|---|---|---|---|---|---|
| A | Corpus Construction and Provenance | `app:corpus` | 1 to 2 (0.6) | 1 (5_setup:10) | 2,661 houses after gate + dedup | skim; yes for the count. `houses_48` / `text2mc_schem` row is not used by any run |
| B | Descriptor Ablation | `app:descriptor` (+`tab:descriptor-ablation`) | 1 to 2 (0.8) | 1 (6_results:266) | size term and sealed-interior scalar do not carry V-KID | yes. Duplicates L's leave-one-family-out at n=256; Grow-Shell paragraph duplicates main 6_results nearly verbatim |
| C | Reproducibility | `app:repro` | 1 to 2 (0.5) | 2 (5_setup:21, :59) | calibration draw must be shuffled; LLM arms off the board | the calibration bullet only. Contains a `\TODO`, a process note ("requirement post-dates that run"), and **10.9x vs main's 10.4x** |
| D | Negative Results | `app:negative` | 2 (0.5) | 0 | attention-as-adjacency and relative-3D bias nulls for code LLMs | no. Off-topic for a metric paper |
| E | Counting Interiors | `app:interior` | 2 to 3 (0.6) | 1 (4_method:16) | aperture-aware counter; plus p-value floor and dose-axis design | partly. **Does not contain the 85.5% / median 100% sealed-share number the main text cites it for.** p-value floor paragraph is process history |
| F | Defects Found in Our Own Gates | `app:defects` | 3 (0.4) | 1 (4_method:81) | gates need auditing | skim. Overlaps M's "when the thresholds were set" (same two denominator fixes) |
| G | Full Tables (+G.1 Dose-Response, G.2 Metrics `sec:metrics`, G.3 Metric Validation) | `app:tables` | 3 to 6 (3.0) | 4 to `app:tables`, 1 to `sec:metrics`, 1 to `tab:arms-full` | grab bag: n=128 blindness table, dose table, case-study table, full arms table, metric defs, render ladder, baselines | yes, but it is a bucket. The four main refs to `app:tables` mean four different things (see Sec. 4) |
| H | Extended Related Work | `app:related` | 6 (0.4) | 0 | Obray, Borji, Caytuiro, Luan, Herve | skim. Obray/Borji also appear in M's prior-use table |
| I | Full View-Protocol Ablation | `app:ablation` (+`fig:ablation`, `tab:seven`, `tab:ablation`) | 6 to 9 (1.0) | 3 to `tab:ablation`, 2 to `tab:seven` | encoders/views/cutaways all at 0.0 to 0.2; cutaways 6.7 to 55.8 | yes. `tab:seven` is misfiled here (it is the statistics ladder, not views). `fig:ablation` duplicates `tab:ablation` and main `fig:cutaway` |
| J | Facade and cross-section views | `app:views25d` | 6 to 7 (0.7) | 2 (6_results:110, :311) | face-on elevations +4.4 via windows; sections see interior via appearance | yes. Refers to "the proposition of main sec:background", which does not exist |
| K | V-KID Against Structural Alternatives | `app:headtohead` | 7 to 9 (1.3) | 1 (6_results:79) | of 14 structural metrics only V-KID passes every gate | yes, a 3D reviewer's first question |
| L | Descriptor Robustness and Generic Descriptors | `app:vkid-robust` | 9 to 10 (1.2) | 1 (6_results:268) | no family carries V-KID; random CNN descriptors pass | yes, answers "hand-designed to catch the probe" |
| M | Where the Gates Come From | `app:gateprov` | 10 to 11 (1.3) | 1 (4_method:49) | gates restate prior criteria; thresholds predate V-KID; sweep | prior-use table and sweep: yes. Dated commit log: reads as process narration |
| N | What a Render Protocol Provably Cannot See | `app:hull` | 11 to 13 (2.0) | 4 (1_intro:26, 6_results:125, :133, :304) | opaque-render impossibility; translucency/atlas leak | yes, the theory backing |
| O | Reading a Squared Statistic | `app:failrate` | 13 (0.5) | 2 (6_results:288, 9_statements:19) | report sqrt(MMD^2) for failure rate | skim. First paragraph repeats the main text verbatim |
| P | What BlockScore's Maximum Is | `app:blockscore` | 13 (0.4) | 1 (4_method:69) | positive null of a max; two pillars in coarser units | yes, short |
| Q | The Noise-Floor Unit Depends on n | `app:nscale` | 13 to 14 (1.0) | 3 (3_background:34, 6_results:146, :317) | separations scale with n and with reference size | yes for the table; the 201.8 vs 286.7 decomposition is defensive |
| R | Calibration of the Head-to-Head Test | `app:significance` | 14 (0.6) | 0 (4 supp-internal) | permutation test replaces anti-conservative test | skim. Written as history ("our first implementation") |
| S | Case Study: Contamination | `app:casestudy` | 15 (0.5) | 2 (5_setup:58, 6_results:294) | split declaration caught 87% contamination | yes |
| T | The n=256 Leaderboard and the n=128 Protocol Run | `app:n256` (+`tab:n256`) | 15 to 16 (0.8) | 0 (2 internal) | ranks stable across n | table yes; bit-identical reproduction narrative belongs in the repo |
| U | Dream-Cubed | `app:dreamcubed` | 16 to 17 (1.0) | 2 (4_method:79, 6_results:254) | extraction rule, twin gate, templates control | yes, a published model being disqualified must be defended |
| V | BrickGPT on the LEGO Suite | `app:brickgpt` | 17 (0.6) | 1 (6_results:155) | BrickGPT run reproduces its validity/stability | yes |
| W | The VLM Judge | `app:vlm` | 17 (0.5) | 2 (6_results:86, `tab:vlm` 6_results:17) | GPT-4o at chance on exterior views | yes |
| X | 3D-Native Metrics and Their Samplers | `app:probe3d` | 17 to 18 (1.0) | 1 + 2 to `tab:probe3d` | point-cloud metrics inherit sampler null space | yes |
| Y | Additional Figures | `app:figures` | 18 to 29 (about 10 float pages) | 0 to the section; 7 to its figures | holds `fig:nn, ladder, contamination, lego, influence, pipeline, gallery, dose, seven, pillars, casestudy` | the figures yes, the section no: dissolve |
| Z | Data Provenance and Release | `app:data` (+`fig:descriptor`, `fig:samples`) | 18 to 19 (0.8) | 2 (teaser, 5_setup:71) | licences, what is and is not redistributed | yes, CVPR wants licences stated. `fig:descriptor` and `fig:samples` are misfiled here |
| AA | Reproducibility Statement | `app:repro-statement` | 19 (0.3) | 0 | released artifacts, scorecards | merge; contains a `\TODO` |

Unreferenced from anywhere (main or supp): `app:negative`, `app:related`, `app:figures`,
`app:ablation` (section label), `app:repro-statement`, `lem:matblind`, `eq:mixture`,
**`fig:samples`** (the every-arm gallery, which reviewers do look at: give it a reference).

## 2. Judgement against CVPR norms

- **Too long and not organized around the paper.** 27 sections in roughly the order the
  experiments were run. A reviewer looking up "Supp. Sec. G" finds a bucket of seven
  unrelated tables. Strong CVPR supplements mirror the main paper's sections; here that is
  Method (probes, metrics, gates, aggregation), Setup, and Results 6.1 to 6.4.
- **Process history in the PDF.** Several passages narrate how the project got here rather
  than what the reader needs: D (code-LLM attention nulls), E's p-value-floor paragraph and
  rejected dose designs, F, G's "What is not a control" (decimation removed), R's "our first
  implementation", T's bit-identical reproduction narrative, M's commit dates, C's "the
  requirement post-dates that run", A's `houses_48`/`text2mc_schem` cache that no run uses.
  These belong in the release README / CHANGELOG. Keep at most one sentence each where a
  number in the paper depends on the fix (F's denominator correction, R's permutation test).
- **Defending against objections the main text does not raise.**
  - D Negative Results: main text never claims anything about injecting structure into
    sequence models. Cut.
  - G "What is not a control" + `tab:dose` caption's dropped resolution axis: decimation is
    never mentioned in the main text. Cut.
  - Q's 201.8 vs 286.7 decomposition and twelve-seed sweep: main text never quotes 286.7.
    Keep one sentence ("floor estimated from 48 draws, about 10% relative SE; twelve seeds
    span 185 to 274") and cut the derivation.
  - T's suite-version flag and "traversal pillar": the main text has five pillars and no
    traversal. Cut; it invites a question the paper does not answer.
  - M is borderline: main 4_method does raise "threshold predates V-KID", so keep the
    prior-use table and the threshold sweep, compress the dated history to two sentences.
  - B and L both answer "hand-designed descriptor"; the main text raises it, so keep, but as
    one section.
- **Duplicated content.**
  - Grow-Shell paragraph: main 6_results (lines 270 to 278) and supp B (lines 65 to 72), same
    numbers. Keep the table column, delete the supp prose.
  - Descriptor ablation at n=128 (B) vs leave-one-family-out at n=256 (L): same experiment
    at two n. Keep both tables only if the main text keeps its n=128 numbers (197.3 / 201.8);
    see Sec. 4 item 9 for the simpler option.
  - O first paragraph repeats main 6_results:286 to 289 verbatim.
  - G.2 last paragraph and G.3 first two paragraphs repeat main 3_background (noise floor)
    and 4_method 4.3 (gates are blocking).
  - G "Every control is re-scored in every run" repeats main 5_setup:62 and C bullet 1 and
    AA. Three copies in supp plus one in main.
  - C bullet 3 (split declaration) repeats main 4_method:80 and S.
  - `tab:blindness` (n=128 protocol run) is superseded by main `tab:leaderboard` (n=256) and
    `tab:n256`. Its significance discussion repeats R.
  - `fig:pillars` plots two columns of main `tab:leaderboard`.
  - `fig:casestudy` plots `tab:casestudy`; `fig:contamination` panel B shows the same.
  - `fig:ablation` plots `tab:ablation`, and main `fig:cutaway` already shows it.
  - `fig:seven` plots `tab:seven`, **but its caption says LEGO while `tab:seven` says
    "in-domain"**: either they are different data (then the caption of one is wrong) or a
    duplicate. Keep `tab:seven` (main cites it twice), cut the figure.
  - `fig:dose` plots `tab:dose`. Keep the table (main cites the ranges); figure optional.
  - Obray and Borji appear in both H and M's prior-use table.
- **Where reproducibility / data / licence info should live.** CVPR has no reproducibility
  slot in the main paper and the review form asks about it, so a single short supplement
  section is right: released artifacts, compute and runtime, licences of every corpus and
  model, the redistribution policy, and the Mojang attribution (this one is a condition of
  the usage guidelines and must stay in the PDF). Everything finer (per-source drop counts,
  cache manifests, canary hashes, scorecard schema, commit log, decoding caps, the
  bit-identical reproduction of the protocol run) goes to the release README, with one
  sentence pointing to it.

## 3. Proposed outline (13 sections, mirroring the main paper)

Page estimates assume figures are placed with their section and `\FloatBarrier` is used.

| new | title | mirrors | contents (old -> new) | labels carried | est. pp. |
|---|---|---|---|---|---|
| A | Metrics and the Voxel Descriptor | main 4.2 | G.2 Metrics (defs, D4 orbits, whitening/clipping); `fig:pipeline` (from Y); `fig:descriptor` (from Z). **Add:** list of the 9 global scalars, the kernel and its parameters for both metrics, renderer and camera spec | `sec:metrics` (promote to section label), `fig:pipeline`, `fig:descriptor` | 1.25 |
| B | Probes and Counting Interiors | main 4.1 | E (closing counter, calibration cube, 25.6%); E's dose-axis design (two sentences, rejected designs cut); `fig:gallery` (from Y, single column). **Add:** the sealed-share statistic (median 100%, mean 85.5%) the main text cites here | `app:interior`, `fig:gallery` | 0.75 |
| C | Validation Gates | main 4.3 | C.1 render ladder: `tab:ladder-render`, `tab:seven` (from I), `fig:ladder` (from Y). C.2 baselines: `tab:baselines` + "Why coherence is reported as distance to real" (from G.3). C.3 prior use and thresholds: M (prior-use table, two-sentence dating, sweep + `fig:gatesweep`) with H's Obray/Borji/Herve paragraphs folded in as prior-use text. C.4 auditing the gates: F condensed to one paragraph (the denominator fix and the fail-open renderer), deduplicated against C.3 | new `app:gates`; `app:gateprov` on C.3; `app:defects` on C.4; `tab:ladder-render`, `tab:seven`, `fig:ladder`, `tab:baselines`, `tab:gateprov`, `fig:gatesweep` | 2.25 |
| D | Aggregation and Statistical Reading | main 3, 4.4 | D.1 BlockScore's maximum (P). D.2 the noise-floor unit and n (Q table + rules + reference-size caveat; decomposition cut to one sentence) with `tab:n256` from T (table + one paragraph; reproduction narrative cut). D.3 significance test (R rewritten as method: permutation test, nominal type-I, one sentence on what changed; E's p-value floor as one sentence). D.4 squared statistics (O minus its first paragraph) | new `app:stats`; `app:blockscore` D.1; `app:nscale` D.2; `app:n256` + `tab:n256` D.2; `app:significance` D.3; `app:failrate` + `eq:mixture` D.4 | 2.0 |
| E | Corpus, Split and Arms | main 5 | E.1 corpus: A with the `houses_32` row only, drop reasons in one line, per-source gate calibration. E.2 arms: `tab:arms-full`. **Add:** a short paragraph on VoxAR (params, tokenizer, training data, steps) since it is scored. E.3 calibration draw and run pinning: C bullets 1 and 2 (fix 10.4 vs 10.9) | new `app:setup`; `app:corpus` E.1; `tab:arms-full` E.2; `app:repro` E.3 | 1.25 |
| F | Dose-Response | main 6.1 | `tab:dose` (caption: drop resolution-axis remark); `fig:dose` optional | `app:tables` (section label, now meaning "dose tables"), `tab:dose`, (`fig:dose`) | 0.5 |
| G | View Protocols | main 6.2 | G.1 encoder/view/cutaway ablation: `tab:ablation` (from I; drop `fig:ablation`). G.2 facade and cross-section views: J (fix the "proposition of sec:background" pointer) | `app:ablation` G; `tab:ablation`; `app:views25d` + `tab:views25d` G.2 | 1.25 |
| H | What a Render Protocol Provably Cannot See | main 6.2 | N unchanged except trimming the DINOv2 attenuation paragraph to two sentences | `app:hull`, `lem:*` | 1.75 |
| I | Other Observers: Point Clouds, Structural Metrics, a VLM Judge | main 6.1 | I.1 X (3D-native metrics and samplers, `tab:probe3d`). I.2 K (head-to-head, `tab:headtohead`). I.3 W (VLM judge, `tab:vlm`). **Add:** the judge prompt text | `app:probe3d` I (section); `tab:probe3d`; `app:headtohead` + `tab:headtohead` I.2; `app:vlm` + `tab:vlm` I.3 | 2.75 |
| J | Robustness of V-KID | main 6.4 | J.1 suspect terms and Grow-Shell: B's table only (prose cut, main has it). J.2 L (families, random subsets, generic descriptors, global coherence). `fig:nn`, `fig:influence` (from Y) | `app:vkid-robust` J (section); `app:descriptor` + `tab:descriptor-ablation` J.1; `tab:vkid-robust`; `fig:nn`; `fig:influence` | 2.0 |
| K | LEGO and BrickGPT | main 6.3 | V + `fig:lego` (from Y) | `app:brickgpt` K (section); `tab:brickgpt-validity`; `fig:lego` | 1.0 |
| L | Leaderboard Arms: Dream-Cubed and the Contamination Case Study | main 6.4 | L.1 U (keep extraction rule, novelty, twin gate, templates control; compress "What the builds are" and "Size" to one paragraph). L.2 S + `tab:casestudy` + `fig:contamination`; the LLM-arm decoding cap from C as one sentence. L.3 `fig:samples` gallery (from Z), now referenced | new `app:leaderboard`; `app:dreamcubed` L.1; `app:casestudy`, `tab:casestudy`, `fig:contamination` L.2; `fig:samples` L.3 | 2.5 |
| M | Reproducibility, Data and Licences | CVPR checklist | AA + Z merged and shortened: released artifacts, compute/hardware/runtime (fill the TODO), licence of each corpus/model, redistribution policy, 3D-Craft-only depiction rule, Mojang attribution verbatim. Pointer to release README for the rest | `app:data` M (section); `app:repro-statement` (keep as alias on the same section, or drop) | 1.0 |

**Total: about 20 pages** (sum 20.25; with tighter figure sizing about 18).

### Cut entirely (move to repo/notes)

| current | what | where it goes |
|---|---|---|
| D Negative Results | code-LLM attention and relative-3D bias nulls | `notes.md` / a future modelling paper |
| G `tab:blindness` + "Significance, not interval overlap" | n=128 protocol-run blindness table | superseded by main `tab:leaderboard` and `tab:n256`; significance point lives in D.3 |
| G "What is not a control" | decimation control removed | release CHANGELOG |
| G "Every control is re-scored" paragraph | third copy | main 5_setup already says it |
| G.3 opening paragraphs | restatement of main 4.3 | none |
| H as a section | extended related work | folded into C.3 prior use; Caytuiro/Luan paragraphs move to main related if space, else cut (Luan is already discussed in H's hull section) |
| Y as a section | figure bucket | figures distributed |
| `fig:pillars`, `fig:casestudy`, `fig:ablation`, `fig:seven` (and optionally `fig:dose`) | plots of tables already printed | none |
| A's `houses_48` row, `text2mc_schem` paragraph | unused cache | release README |
| C TODO list, scorecard description, bullet 3 | reproducibility minutiae / duplicate | M (one sentence) and README |
| T bit-identical reproduction, descriptor version flag, traversal pillar | process | README |
| R narrative of the first implementation and rescale factors | process | one sentence in D.3; details in README |
| M dated commit history | process | two sentences in C.3; full log in README |
| E p-value floor paragraph, rejected dose designs | process | one sentence each in D.3 / B |
| Z long Mojang/3D-Craft licence reasoning | licence analysis | keep the conclusions + required attribution in M; reasoning in release LICENSE notes |

## 4. Label and reference changes (apply mechanically)

### 4a. Main-text references that must be re-pointed

`app:tables` currently means four different things. Re-point each, then reuse the label for
the dose section (F).

| # | file:line | current | change to | why |
|---|---|---|---|---|
| 1 | `sec/4_method.tex:51` | `\suppsec{app:tables}` ("settles three things") | `\supptab{tab:ladder-render}` | the three findings are the render ladder |
| 2 | `sec/6_results.tex:55` | `\suppsec{app:tables}` (interior-axis ranges 0.48/0.34, 133/287) | `\supptab{tab:dose}` | dose table |
| 3 | `sec/6_results.tex:198` | `\suppsec{app:tables}` in `tab:leaderboard` caption ("coarser in-run scale") | `\suppsec{app:blockscore}` | that is P's second paragraph |
| 4 | `sec/6_results.tex:309` | `\suppsec{app:tables}` (Patch-Stitch / global coherence) | `\suppsec{app:vkid-robust}` (global-coherence paragraph) or `\supptab{tab:baselines}` | coherence evidence is in L |
| 5 | `sec/5_setup.tex:59` | `\suppsec{app:repro}` (LLM arms not on the leaderboard) | `\suppsec{app:casestudy}` | decoding cap moves into L.2 |
| 6 | `sec/5_setup.tex:21` | `\suppsec{app:repro}` with "10.4x" | keep ref (label moves to E.3); **reconcile 10.4x vs supp 10.9x** | number mismatch |
| 7 | `sec/4_method.tex:16` | `\suppsec{app:interior}` for "median 100%, mean 85.5%" | keep ref; **add the statistic to B** | supp currently lacks it |
| 8 | `sec/4_method.tex:38` | `\suppsec{sec:metrics}` | keep (now a section, A) | none |
| 9 | `sec/6_results.tex:266` | `\suppsec{app:descriptor}` with n=128 values 197.3/201.8 | keep ref (label on J.1). Optional simplification: cite the n=256 rows of `tab:vkid-robust` instead (without size 440.9, without interior 421.8 of 444.4) and then drop `tab:descriptor-ablation` | removes the two-n duplication |
| 10 | (none) | `fig:samples` unreferenced | add `\suppfig{fig:samples}` to the `tab:leaderboard` caption or 6_results:200 | reviewers look for a gallery |

All other main-text refs keep working unchanged once the labels sit where Sec. 3 puts them:
`app:hull, app:nscale, app:failrate, app:blockscore, app:gateprov, app:defects, app:corpus,
app:data, app:casestudy, app:dreamcubed, app:brickgpt, app:vlm, app:probe3d, app:headtohead,
app:views25d, app:vkid-robust, app:interior, app:descriptor, tab:ablation, tab:seven,
tab:probe3d, tab:vlm, tab:arms-full, fig:pipeline, fig:ladder, fig:contamination, fig:lego,
fig:nn, fig:influence`. Their letters change (e.g. `app:hull` N -> H), which is automatic
via `xr-hyper` after rebuilding supp then main (Makefile order).

### 4b. Labels to add, move, or remove

- **Add** section labels: `app:gates` (C), `app:stats` (D), `app:setup` (E), `app:leaderboard` (L).
- **Promote** `sec:metrics` from `\subsection` to `\section` (A).
- **Move** (label travels with its text, now as a `\subsection`): `app:gateprov` -> C.3,
  `app:defects` -> C.4, `app:blockscore` -> D.1, `app:nscale` and `app:n256` -> D.2,
  `app:significance` -> D.3, `app:failrate` -> D.4, `app:corpus` -> E.1, `app:repro` -> E.3,
  `app:views25d` -> G.2, `app:headtohead` -> I.2, `app:vlm` -> I.3, `app:descriptor` -> J.1,
  `app:dreamcubed` -> L.1, `app:casestudy` -> L.2.
- **Keep as section labels**: `app:interior` (B), `app:tables` (F, dose), `app:ablation` (G),
  `app:hull` (H), `app:probe3d` (I), `app:vkid-robust` (J), `app:brickgpt` (K), `app:data` (M).
- **Remove**: `app:negative`, `app:related`, `app:figures`, `tab:blindness`, `fig:pillars`,
  `fig:casestudy`, `fig:ablation`, `fig:seven`, optionally `fig:dose`; `app:repro-statement`
  can be dropped (nothing references it) or kept on M.

### 4c. Supplement-internal references to fix after the move

- `X_appendix.tex:276-277, 286`: text about `tab:blindness` goes with the table.
- `X_appendix.tex:293-296` (Dose-Response intro) refs `fig:dose`, `tab:casestudy`,
  `fig:casestudy`: rewrite to reference only what remains in F.
- `X_appendix.tex:462-466` (Metric Validation intro) refs `fig:seven`, `fig:gallery`,
  `fig:pillars`, `fig:descriptor`: delete the paragraph (each float is now in its own section).
- `X_appendix.tex:589` refs `fig:ablation`: delete.
- `X_appendix.tex:307-308` (`tab:dose` caption) ref to `\mainsec{sec:setting}` for the
  resolution axis: delete the sentence.
- `X_appendix.tex:1503` (`fig:casestudy` caption) goes with the figure.
- `X_views25d.tex:48` "the proposition of `\mainsec{sec:background}`": no such proposition;
  change to `Lemma~\ref{lem:surface}`.
- `X_reproducibility.tex:10` "Sec.~\ref{app:repro} lists these": rewrite once merged into M.
- `X_appendix.tex:146` "Appendix~\ref{app:casestudy}": moves into L.2 itself, drop.
- `supp.tex`: with 13 sections, revert `\AlphAlph` to `\Alph` and drop `alphalph`; add
  `\usepackage{placeins}` and `\FloatBarrier` before each `\section`; remove the separate
  `\input{sec/X_reproducibility}` once merged into M.
- Mixed citation style inside supp: some refs read "Appendix~\ref{...}", others "Sec.~";
  CVPR supplement convention is "Sec." Normalize.

## 5. Lean path to about 15 pages (if wanted)

In order of least loss: drop `fig:dose` (-0.4); drop `tab:descriptor-ablation` after item 9
(-0.4); move the Objaverse half of `tab:probe3d` to keep only `vis` and `cut` rows (-0.5);
compress H's renderer-ablation tables to one table (-0.4); cut `fig:gallery` since the teaser
and `fig:samples` show the probe (-0.4); shorten L.1 Dream-Cubed to extraction rule + twin
gate (-0.5); shorten C.3 to the prior-use table and the sweep figure (-0.5).

## 6. Genuinely missing, or inconsistent with the main text

1. **Main 4_method:16 cites `app:interior` for "sealed share median 100%, mean 85.5%";
   the supplement never states it.** Add to B.
2. **10.4x (main 5_setup:21) vs 10.9x (supp `app:repro`)** for the unshuffled calibration
   draw. One is stale.
3. **Main 6_results:130 says "What leaks is translucency, not doorways"**, but supp
   `app:hull` says the atlas-layout artifact alone alters 92% of views with translucency
   off, and "the corruption's own reach is the 28% that survives flattening the textures".
   The main-text sentence and the 91% figure need a qualifier pointing at `app:hull`, or a
   reviewer who reads H will see the main text overstate translucency.
4. `fig:seven` caption says LEGO, `tab:seven` says in-domain: resolve before cutting either.
5. `X_views25d.tex:48` points to a non-existent proposition in main sec:background.
6. Reproducibility content the CVPR form will ask for and the supplement lacks: compute
   (hardware, GPU-hours, wall-clock per run), anonymized code URL and commit (two `\TODO`s),
   the institutional determination `\TODO` in Z.
7. Method details needed to reimplement V-KID and MV-KID that are not stated anywhere: the
   kernel actually used on each feature set (polynomial degree/offset or RBF bandwidth),
   the list of the 9 global scalars, the render spec (camera azimuth/elevation of the 4
   orthographic views, lighting, texture pack), and the 6- vs 26-connectivity used to define
   sealed air for Fill-Interior.
8. VoxAR (a scored submission and the contaminated arm) has one line in `tab:arms-full`; a
   short paragraph on size, tokenizer, data and training budget is expected.
9. The GPT-4o judge's prompt text and the pre-registered predictions are referred to but not
   printed; add them to I.3 (a few lines).
10. `fig:samples` is unreferenced.
