# Proposed fixes for the live \com / \TODO notes (2026-11-04)

None of these is applied. Line numbers are as of the restructure on 2026-11-04. `make submit`
fails on any note left in place, so each needs a decision.

## Comments (\com)

### 1. sec/1_intro.tex:15, `\com{What does this mean haha? These two last lines}`
The sentence is "…a solidified corpus and its genuine, real data counterpart *share a
significance group* under a paired, Holm-corrected comparison over all arm pairs. The same
encoder and the same statistic separate a material permutation of the same builds at
+422.43 noise floors."

Proposed replacement for both sentences:
> Scored at $n{=}256$ against $399$ held-out real builds, the filled corpus sits $0.35$
> noise floors from real data, where one noise floor is the metric's own spread between
> two draws of real data, and no significance test over all arm pairs separates the two
> \citep{holm1979simple}. The same encoder and the same statistic place a copy of the same
> builds with their materials shuffled at $+422.43$ (Table~\ref{tab:leaderboard}).

### 2. sec/1_intro.tex:18, `\com{weakish? Hard to follow--maybe too much bloat, …}`
The paragraph repeats the cutaway result and the opaque-renderer theorem. Both are now in
§6.2. Proposed replacement, which cuts it to three sentences:
> The failure is in what the observation contains, not in what the encoder computes.
> Changing only the cameras so that the renders show the interior separates the same probe
> at $6.7$ to $55.8$ noise floors with the same encoder and statistic, and under an opaque
> renderer no exterior-view metric can separate it at all (\S\ref{sec:ablation},
> \suppsec{app:hull}).

### 3. sec/1_intro.tex:29, `\com{what does this line mean}` on "A metric should therefore earn the right to rank."
Proposed replacement for the opening sentence:
> We therefore require a metric to pass tests before it may rank models.

The rest of the paragraph stays.

### 4. sec/2_related.tex:20, `\com{review the flow here}`
This opening sentence is unattached: the next paragraph starts again with Dream-Cubed.
Proposed fix: delete the opening sentence and start the first paragraph with
> Generators of discrete 3D structure, text-to-brick systems
> \citep{pun2025brickgpt,kulits2026bricknet,ni2026brickanything}, voxel generators
> \citep{chen2019voxelcnn,jung2025scaffold,awiszus2021worldgan} and Minecraft world models
> \citep{earle2024dreamcraft,merino2026dreamcubed}, evaluate their outputs through rendered
> views. Dream-Cubed, for example, ranks generated worlds with render-based FID, …

This also fixes a citation error: `earle2024dreamcraft` appears in both the voxel list and
the world-model list.

### 5. sec/3_background.tex:34, `\com{check all appendix refs}`
Checked on 2026-11-04 after the restructure. Building with `make` gives 0 undefined references in
main.log and supp.log. Every `\suppsec`, `\supptab` and `\suppfig` resolves into supp.aux,
including the new `app:casestudy` and `fig:ladder`. **Proposed fix: delete the comment.**

### 6. sec/5_setup.tex:19, `\com{what is this next line?}` on "The calibration draw needs the same care…"
Proposed replacement:
> The real draw that sets the noise floor needs the same care: taking the first $n$ test
> builds instead of a random sample inflates the measured noise floor by
> $\mathbf{10.4\times}$, because consecutive builds in the source files are siblings
> (\suppsec{app:repro}).

Before applying, check that "consecutive builds are siblings" is the reason
`app:repro` gives. If it is not, drop the "because" clause.

## TODOs

### 7. sec/6_results.tex:245 and sec/X_appendix.tex:1150, the Dream-Cubed template audit
**Owned by the gate agent.** The audit is done (outputs/analysis/dreamcubed_audit/README.md,
notes §73), and the user chose to add a vanilla-template control row and the near-twin rate
(option 8a). Not touched here.

### 8. sec/X_reproducibility.tex:16 and sec/X_appendix.tex:113, the anonymized code URL, commit, hardware and wall-clock
Only the authors can supply these. Fill them in at submission (supplement deadline Nov 23).
For the main-paper deadline, the reproducibility TODO can be replaced by
> Code and the pinned split will be released; an anonymized copy accompanies the
> supplementary material.

That holds only if the anonymized release (`make_release.py` → `../blockgen-release`)
is uploaded with the supplement.

### 9. sec/X_appendix.tex:1516, the institutional determination for the corpus release
This is for the authors and the UCLA research-compliance office. Until the determination is
in, a truthful placeholder that does not block the build is
> We release no upstream build content; whether the derived corpus may be released is
> pending review by the authors' institution, and the release will follow its determination.
