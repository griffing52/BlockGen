# Supplement material moved out of the PDF

Restructure of 2026-11-07 (plan: SUPP_RESTRUCTURE.md, 13-section version). Each block below is
verbatim LaTeX from the pre-restructure sources, which are also kept whole in
`.backup_20261107_pre_supp_restructure/`. Line numbers refer to those backup files.
Destination for each: the release README / CHANGELOG, or notes.md.

## 1. Corpus: the fourth source and the 48^3 cache

From: `X_appendix.tex:4-11, :28, :42-44 (old Sec. A, Corpus Construction and Provenance)`

Why moved: No reported run uses `houses_48` or `text2mc_schem`. The new Sec. E.1 keeps the `houses_32` row and one line of drop reasons.

```latex
\emph{Four} public sources are pooled onto a shared legacy block vocabulary
(the text2mc dump is remapped with $97\%$ block-state coverage; world cuts are
ground-stripped), passed through a quality gate, and deduplicated variant-aware.
The fourth source, \texttt{text2mc\_schem}, is the set of raw \texttt{.schem}
files in the text2mc release that its authors never converted. It entered the
pool after the $32^3$ cache was built, so that cache draws on three sources and
only the $48^3$ cache carries all four. The main text (\mainsec{sec:setting}) describes the three that build
\texttt{houses\_32}, the corpus every reported run uses.
% ...
\texttt{houses\_48} & 4{,}502 & 1{,}381 & 33 & \textbf{3{,}088} & 1{,}410 & 1{,}381 & 180 & 117 \\
% ...
\texttt{text2mc\_schem} is gated on the same grounds as the rest of the text2mc
release and is not calibrated separately. What the gate removes from the gated
sources is trees, a truck and roof fragments, read off a pre-gate visual grid.
```

## 2. Descriptor ablation: Grow-Shell prose

From: `X_appendix.tex:65-72 (old Sec. B)`

Why moved: Duplicates main 6_results (Grow-Shell paragraph, same numbers). The table column stays in Sec. J.1.

```latex
\growshell{} settles the mass question directly. It dilates the solid outward
until it has added exactly the number of blocks \solidify{} adds to that build,
which it achieves to within one block on $128$ of $128$ builds at a median of
$196$ added blocks. Same builds, same material, same mass, and the only
difference is whether the new mass lands inside or outside. The two tiers then
separate completely: \mvkid{} reads $+0.26$ noise floors on \solidify{} and
$+102.27$ on \growshell{}, while \vkid{} reads $201.8$ and $214.5$. A descriptor
that counted blocks could not produce that pair of rows.
```

## 3. Reproducibility: TODO, controls-re-scored bullet, split-declaration bullet

From: `X_appendix.tex:113-130, :138-142 (old Sec. C)`

Why moved: Bullet 1 repeats main 5_setup; bullet 3 repeats main 4_method and Sec. L.2. The 87.14 / 19.05 control-size example is preserved here. The TODO is merged into Sec. M.

```latex
\TODO{Fill in at submission: anonymized code URL, exact commit, the pinned split
files, the feature-cache render canary hash, hardware, and per-run wall-clock.
The suite writes a scorecard carrying corpus, split, $n$, reference size, seed,
host, device and git SHA for every run, so this table is a transcription rather
than a reconstruction.}

Two properties of the protocol matter for reproducibility:

\begin{itemize}[leftmargin=*,itemsep=1pt,topsep=2pt]
\item \textbf{Controls are re-scored in every run, never cached across runs.}
  \blockscore{}'s unit is the spread of the real control \emph{scored in the
  same run}, so two scores sit on one scale only when corpus, split, tier, seed,
  reference size and control size all match. We pin these as a named protocol
  and refuse to rank runs that do not satisfy it, listing each excluded run with
  its reason. The size of the control alone moves the score: the same model on
  the same corpus, split and tier scores $87.14$ against a $128$-build control
  and $19.05$ against a $32$-build one. A board that ignored the control's size
  would read that as a $4.5\times$ improvement.
% ...
\item \textbf{Every arm must declare its training split} to be ranked, and an
  arm that cannot is reported unranked with that as the stated reason. No arm in
  the run reported here declares one (Fig.~\ref{fig:contamination}). The
  requirement post-dates that run, so it is the fix rather than a gate that
  fired.
```

## 4. Negative Results (whole section)

From: `X_appendix.tex:151-178 (old Sec. D, label app:negative)`

Why moved: Off-topic for a metric paper; nothing in the main text references it.

```latex
\section{Negative Results}
\label{app:negative}

Three experiments that did not work. Each bounds a claim a reader might
otherwise expect us to make about injecting structure into sequence models.

\paragraph{Attention is not a free adjacency prior.} A pretrained code LLM's
self-attention over a serialized build recovers less adjacency information than
the coordinates already printed in the prompt. Scoring all $336$ heads by how
well they rank the truly $6$-adjacent previous pieces (AUC over the causal
candidate set), the best head reaches $0.874$ in the base model and $0.884$
after task fine-tuning, well above a recency floor. But $-\|a_i - a_j\|_1$,
three lines of arithmetic over the coordinates the serialization already prints
on every line, scores $0.980$. A trained bilinear probe does better, $0.914$ at
layer $14$ of $28$, and falls to $0.807$ by the final layer, which is the layer
generation actually reads. There is signal; there is nothing to steal.

\paragraph{A shared relative-{3D} attention bias is a null, for a reason in the
parameterization.} Adding a learned additive attention bias indexed by the
relative {3D} offset between pieces, zero-initialized and otherwise holding the
fine-tune fixed, moved best validation loss by $-0.0021$ at one seed and did not
touch the collapse mode. The learned table explains why: $364$ of $729$ offset
buckets are \emph{never updated}. Under raster emission order every earlier
piece is sign-constrained, so half the offset space is unreachable by
construction, and the arm's entire learned signal is a reparameterization of
recency. Unlocking the rest requires a non-monotonic emission order. The
reference anchor is not the constraint: the dead half of the offset space
follows from monotonic emission alone.
```

## 5. Counting interiors: the bootstrap p-value floor and the rejected dose designs

From: `X_appendix.tex:200-207, :212-216 (old Sec. E)`

Why moved: Process history; one sentence of each remains (Sec. D.4 and Sec. B).

```latex
\paragraph{The bootstrap $p$-value floor.}
A bootstrap sign-balance $p$-value is floored at $1/n_{\text{rep}}$, and Holm
over the $55$ pairs of an $11$-arm run multiplies $0.005$ to $0.275$. Nothing
can then reach $\alpha=0.05$ however large the true difference, which put
\emph{every} arm of an earlier head-to-head, from held-out real to a monochrome
corpus, in one ``not separated'' group. The reported $p$ is now studentized and
unfloored. The failure mode is silent: it produces a table in which nothing is
significant, which reads like a careful result.
% ...
0.0475 \to 0.0242$). Two earlier designs were rejected. Random interior speckle
is detected as \emph{noise} rather than as lost interior. Filling whole rooms
largest-first makes the axis useless, because one dominant pocket is typically
more than a quarter of the total interior, so a nominal $25\%$ dose already
removes $71\%$ of it.
```

## 6. Defects Found in Our Own Gates (full text)

From: `X_appendix.tex:218-245 (old Sec. F)`

Why moved: Condensed to one paragraph in Sec. C.4 (label app:defects kept).

```latex
\section{Defects Found in Our Own Gates}
\label{app:defects}

These are the class of error a validation ladder exists to prevent, and it did
not.

\paragraph{Two defects in our own gates.} First,
$\mathrm{sd}(\text{real})$, the unit every gate is stated in, was mismeasured
twice. It was estimated from repeated subsamples of a pool that \emph{excluded}
the probe set, so the fresh real draw being tested was not a member of its own
null. Overlapping draws also carry a finite-population factor that shrinks their
spread. Corrected, a metric that separates the solidify probe at $67$ s.d.\ passes
the gate it had failed, having been rejected for noise that was measured wrong.
Second, the $n$-stability gate tested a difference of means against the spread
of a \emph{single} draw, larger by $\sqrt{\text{reps}}$, and therefore had almost
no power. Against the standard error of the difference, the biased estimator's
$1/n$ decay is rejected at $14.8$ standard errors.

\paragraph{A third defect, in the renderer.} A feature extractor that fails
open produces numbers that look excellent and cannot be caught downstream. Ours
substituted a blank white frame for any rendering exception. Under a library
version mismatch every render raised, every image came back white, every feature
collapsed to one vector, and every arm scored KID $=-4\times 10^{-13}$, better
than real. The feature cache is validated by a render canary, so the all-white
canary would have mismatched and the cache would have been silently rebuilt from
blank frames. The renderer now raises past a $2\%$ failure rate. Repairing the
library reproduced the stored canary hash bit-for-bit, so no previously recorded
number is affected.
```

## 7. Full Tables: tab:blindness and its significance discussion

From: `X_appendix.tex:253-288 (old Sec. G)`

Why moved: The n=128 protocol-run blindness table is superseded by main tab:leaderboard (n=256) and tab:n256; the significance point lives in Sec. D.4.

```latex
\begin{table*}[t]
\centering
\small
\caption{Realism at $n{=}128$ against $399$ held-out real builds, with
bootstrap $95\%$ intervals. \textbf{\solidify{} is indistinguishable from
held-out real on the render metric} and $202$ spreads away on the geometric one.
The bottom two rows are the converse blindness: \vkid{} reads occupancy only,
so a material permutation moves it by \emph{exactly} zero (values are
bit-identical to the real row, not merely close).}
\label{tab:blindness}
\begin{tabular}{lcc}
\toprule
arm & \mvkid{} $\downarrow$ & \vkid{} $\downarrow$ \\
\midrule
\realheld{} (floor)       & $-0.002$ $[-0.004, 0.002]$ & $-0.09$ $[-0.22, 0.03]$ \\
\textbf{\solidify{}}   & $\mathbf{-0.001}$ $[-0.004, 0.002]$ & $\mathbf{22.03}$ $[20.65, 23.70]$ \\
\matshuffle{}& $0.667$ $[0.636, 0.706]$ & \emph{$-0.09$ (bit-identical)} \\
\monochrome{}          & $0.859$ $[0.833, 0.890]$ & \emph{$-0.09$ (bit-identical)} \\
\traincopy{}          & $0.017$ $[0.008, 0.026]$ & $0.96$ $[0.60, 1.51]$ \\
\bottomrule
\end{tabular}
\end{table*}

Table~\ref{tab:blindness} is the result, from the protocol run. The solidified corpus reads
$-0.001$ against a real floor of $-0.002$.

\paragraph{Significance, not interval overlap.}
Marginal confidence intervals are not a comparison, because non-overlap demands
a difference roughly $41\%$ larger than significance does. We therefore test every
arm pair with the reference draw shared and apply Holm correction
\citep{holm1979simple} across the family. Under that test
\solidify{} and \realheld{} \textbf{share a significance
group} on \mvkid{}. The same test at the same $n$ assigns
\traincopy{} its own group at a difference of $0.019$, a separation that sits on the $\alpha{=}0.05$ boundary after Holm once the test is correctly calibrated (Appendix~\ref{app:significance}), which is why the main text argues the point by magnitude instead. On \vkid{}
the same solidified arm is four groups away, $\Delta = -22.13$
$[-23.74, -20.83]$, $p \approx 0$.
```

## 8. Full Tables: 'Every control is re-scored' and 'What is not a control' (full text)

From: `X_appendix.tex:399-414 (old Sec. G)`

Why moved: The first paragraph repeats main 5_setup. 'What is not a control' is kept in Sec. E.2 as a two-sentence policy statement (paper_readiness.py requires the phrase); the full reasoning is here.

```latex
Every control is re-scored in every run rather than cached, because a distance
means nothing until a floor has been measured under \emph{this} run's corpus,
split, $n$ and reference size. Importing a floor across runs ranks arms against
conditions they were never measured under.

\paragraph{What is not a control.} Earlier versions of this suite
included real data requantized to a coarser grid as a ranked control. We have
removed it. Decimation changes a structure's \emph{resolution}, not its
\emph{quality}, yielding a smaller building rather than a worse one, so a
leaderboard row for it invites the comparison ``this model is worse than
$86\%$-deleted real data,'' which is not a meaningful statement about quality.
It is removed from the validation ladder for the same reason and not merely from
the leaderboard: a ladder rung asks whether a metric orders a \emph{severity}
axis correctly, and decimation is not one, so a metric that ranks it any
particular way has demonstrated nothing. No result in this paper rests on a
decimation rung.
```

## 9. Metric Validation: opening paragraphs

From: `X_appendix.tex:460-474 (old Sec. G.3)`

Why moved: Float pointers made obsolete by placing each float in its own section; the second paragraph restates main 4.3.

```latex
\subsection{Metric Validation}

Table~\ref{tab:baselines} reports the procedural baselines against real held-out data,
Figure~\ref{fig:seven} plots Table~\ref{tab:seven}, Figure~\ref{fig:gallery} shows both
probe directions on three builds,
Figure~\ref{fig:pillars} plots the two pillars against each other, and
Figure~\ref{fig:descriptor} walks through the geometric descriptor.

Every candidate metric is scored on rungs of known damage, in units of its own
noise floor (the spread of real-vs-reference draws at the same $n$). Validity
gates are \emph{blocking}: a metric that fails one is reported as \emph{gated}
with the failing gate named, and may not appear in any ranking. Sensitivity, how
\emph{much} damage a metric resolves, is recorded separately, because
that is a question about $n$, not about whether the metric measures the right
thing.
```

## 10. Extended Related Work (whole section)

From: `X_appendix.tex:551-585 (old Sec. H, label app:related)`

Why moved: Obray, Borji and Herve are folded into Sec. C.3 as prior-use text; Luan is discussed in Sec. H. Caytuiro and the full Luan paragraph are here; they can move to main related work if space allows.

```latex
\section{Extended Related Work}
\label{app:related}

\paragraph{\citet{obray2022graphmetrics}.} Perturbs graphs at graded severity
and scores a metric on whether it responds monotonically, with Pearson
correlation against perturbation degree as the pass criterion. The procedure
selects among metric hyperparameters; it never disqualifies a metric, has no
invariance axis, and does not test stability in the sample size. Our ladder is a
blocking precondition on ranking and adds both.

\paragraph{\citet{borji2019prosandcons}.} Enumerates desiderata for generative
metrics including sensitivity to distortions, invariance to
semantically-meaningless transforms, and low sample complexity, which are our ordering,
invariance and $n$-stability gates respectively. The contribution here is
operationalizing them as a gate that runs in every evaluation rather than as a
list.

\paragraph{\citet{caytuiro2025symmetry}.} Audits {3D} generators for symmetry
violation, with the same rhetorical structure as this paper one property over.
The contrast is the point: a broken symmetry is visible in any exterior view, so
a render-and-embed metric can in principle see it. An interior cannot be seen
from outside at all, which is why the failure is structural rather than a matter
of encoder quality.

\paragraph{\citet{luan2026rendermetric}.} Criticizes rendering-based {3D}
evaluation on the grounds of incomplete structural coverage and viewpoint
sensitivity, and replaces renders with sampled surface points. This removes
rendering without removing the blindness: a surface point set describes the same
boundary. We read this as support: two independent critiques of the render
pipeline, neither of whose fixes reaches the interior.

\paragraph{\citet{herve2021pcgmetrics}.} Correlates eighteen procedural-content
metrics against GDMC jury scores. The only prior meta-evaluation of metrics in
this domain, and the route by which a human-validated version of this suite
would proceed.
```

## 11. View-protocol ablation: fig:ablation

From: `X_appendix.tex:587-602 (old Sec. I)`

Why moved: The figure plots tab:ablation and main fig:cutaway already shows it. The PDF figures/fig5_ablation.pdf stays on disk.

```latex
\section{Full View-Protocol Ablation}

Figure~\ref{fig:ablation} plots every condition in Table~\ref{tab:ablation} on one
axis.
\label{app:ablation}

\begin{figure*}[t]
\centering
\includegraphics[width=0.9\linewidth]{figures/fig5_ablation}
\caption{\textbf{The blindness belongs to the view protocol, not the encoder.}
Separation of the interior-fill probe in each condition's own noise floors (log
scale), with paired bootstrap intervals. Gray: every encoder, view budget and
projection we test on exterior views, spanning $0.0$--$0.2$ floors. Blue: the
same encoder and the same statistic once the renders expose the interior.}
\label{fig:ablation}
\end{figure*}
```

## 12. Render-kernel section: DINOv2 attenuation paragraph (full text)

From: `X_appendix.tex:797-805 (old Sec. N)`

Why moved: Trimmed to two sentences in Sec. H.

```latex
%TL;DR of paragraph: where the surviving signal is lost, stated as measurement rather than theory.
Where the surviving signal dies is measurable but not, we think, provable. In
DINOv2 the patch tokens a changed pixel touches move by $0.062$ against a
between-build scale of $0.517$ ($12\%$); the CLS token moves $0.011$ against
$0.277$ ($3.9\%$); the four-view pooled embedding moves $0.0053$ against
$0.268$ ($2.0\%$). The signal enters the features and is attenuated by
aggregation. We report this as a measurement and decline the tempting story
that the encoder is invariant to the corruption by training, which the
patch-level numbers contradict.
```

## 13. Squared statistic: first paragraph

From: `X_appendix.tex:828-833 (old Sec. O)`

Why moved: Repeated the main text verbatim; the main text has since cut it (lead, 2026-11-07), so a one-sentence version now opens Sec. D.5.

```latex
Everything in the main text concerns what a metric can \emph{see}. How its value is
\emph{read} is a second issue, and it affects any leaderboard reporting KID or MMD over a
population of prompts. Because these statistics are squared, an arm that fails on a quarter
of its builds scores only $6.0\%$ of the way to fully bad, where a reader assuming
proportionality would infer a quarter of the damage. Reporting the square root recovers the failure rate to a mean absolute error of $0.013$
against $0.128$ for the raw value.
```

## 14. Noise-floor unit: the 201.8 vs 286.7 decomposition

From: `X_appendix.tex:946-969 (old Sec. Q)`

Why moved: Defensive; Sec. D.2 keeps one sentence (48 draws, about 10% relative SE, twelve seeds span 185 to 274).

```latex
\paragraph{The unit is estimated, not only $n$-dependent, and equal $n$ is not
enough.} The floor is itself a statistic: $48$ real-versus-reference draws,
carrying a relative standard error near $1/\sqrt{2 \cdot 47} \approx 10\%$.
Two of our own runs make the consequence concrete. At the same $n{=}128$, the
same corpus, the same pinned split, the same reference and the same commit, the
protocol run reports the interior-fill separation as
$\mathbf{201.8}$ and the converse-probe run reports it as $\mathbf{286.7}$.
The only differing inputs are two random seeds, one choosing which $128$ test
builds were drawn and one seeding the $48$ floor draws. The ratio decomposes
exactly into the two:
\[
\underbrace{\frac{286.7}{201.8}}_{1.42}
\;=\;
\underbrace{\frac{0.1097}{0.0848}}_{\text{floor estimate, } 1.29}
\;\times\;
\underbrace{\frac{24.430 - 0.128}{22.034 - (-0.095)}}_{\text{raw difference, } 1.10},
\]
that is, roughly $70\%$ a different estimate of the same floor and $30\%$ a
different draw of the builds. Two estimates of one floor landing $29\%$ apart is
a $2\sigma$ event for a standard deviation taken from $48$ draws, so the unit
carries a visible error bar. A separation should therefore be read to one or two
significant figures, with its run named. We quote $202$ throughout because it is
the protocol run's value, and the reader should understand it as \emph{about two
hundred}: a sweep over twelve seeds places it between $185$ and $274$.
```

## 15. Head-to-head calibration: narrative of the first implementation

From: `X_appendix.tex:999-1016 (old Sec. R)`

Why moved: Rewritten as method in Sec. D.4; the rescale factors and the history are here.

```latex
Our first implementation estimated the standard error of a paired difference by
subsampling $80\%$ of each arm without replacement and using the spread of those
draws directly. That is the variability of a subsample statistic around the
full-sample value, which becomes the standard error of the full-sample statistic
only after a rescale we did not apply. Measured on exact nulls, the resulting
test rejected $3.4\times$ too often at $\alpha{=}0.05$ when the arms match the
reference and $7.2\times$ too often when they do not.

%TL;DR of paragraph: no single rescale works because the statistic changes regime, so we use a permutation test.
No constant rescale repairs it. Conditional on a fixed reference, the difference
of kernel distances is a degenerate U-statistic when both arms match the
reference and a non-degenerate one when they do not, with required factors of
$1.31$ and $2.06$ respectively; across the $153$ pairs of one table the factor
ranges from $0.30$ to $11.9$. We therefore use an exact label-permutation test:
pool the two arms, re-split at the observed sizes, and recompute the difference
against the same fixed reference. Its null standard deviation tracks a
Monte-Carlo ground truth within $5\%$ in both regimes, and its type-I error is
nominal at both levels.
```

## 16. n=256 leaderboard: bit-identical reproduction, descriptor version flag, traversal pillar

From: `X_appendix.tex:1070-1080 (old Sec. T)`

Why moved: Process; the main text has five pillars and no traversal pillar.

```latex
The leaderboard of \maintab{tab:leaderboard} reruns the protocol run's board at the
protocol's own $n{=}256$, with the same reference, split, descriptor and seed. The suite's
current default descriptor adds three occupancy scalars and a sixth \emph{traversal}
pillar; a pinned version flag restores the protocol run's $77$-dimensional descriptor and
five pillars exactly. Rerun under that flag at the protocol run's own arguments, the suite
reproduces every recorded value of the protocol run: \vkid{}, every fast-tier number, the
noise floors, the validation ladder, all six self-checks and both head-to-head groupings
are bit-identical, and \mvkid{} differs by GPU float noise only (at most
$7.4\times10^{-7}$). The generator arms were then extended from $128$ to $256$ samples from
the same checkpoints, seed and sampling stream, so the first $128$ samples of every arm are
byte-identical to the protocol run's.
```

## 17. Dream-Cubed: 'What the builds are' and 'Size' (full text)

From: `X_appendix.tex:1182-1203 (old Sec. U)`

Why moved: Compressed to one paragraph in Sec. L.1; the templates control is kept in full.

```latex
\paragraph{What the builds are.} Of the $284$ extracted builds, $45\%$ are houses with a
room, $36\%$ are roofless enclosures and $19\%$ are fragments, against $99\%$ houses in
held-out real. Their median best IoU to the $299$ village templates of the current game
version is $0.52$, against $0.32$ for held-out real, and none reaches $0.9$. Dream-Cubed
trained on worlds of an older version whose villages are generated by code rather than
from template files, so this comparison understates reproduction of vanilla village
pieces; the within-set rate needs no reference. As a control, we pass the current templates
through Dream-Cubed's own extraction and score them under the leaderboard's protocol:
$174$ house templates survive the $80$-block floor ($183$ village templates in all). Their
pillars would give a \blockscore{} of $202.95$ ($194.98$ for all $183$), geometry-limited,
ahead of every model we trained, from a set that involves no learning. They repeat each
other as well ($67.2\%$ twins), and the near-twin gate disqualifies them at $+33.2$
spreads. Scored in the same run, Dream-Cubed reproduces its leaderboard values exactly.

\paragraph{Size.} The builds are village pieces at a median of $272$
blocks, against $675$ for the reference. Two checks bound what the size term contributes to
the geometric distance. Refitting the standardizer without the $\log(1+n_{\text{blocks}})$ column
moves raw \vkid{} from $7.67$ to $7.17$, so size carries at most $7\%$ of it. Scoring
against only the $111$ reference builds inside Dream-Cubed's own block-count interquartile
range ($155$--$462$) raises raw \vkid{} to $8.57$. The builds enclose little space
(sealed interior ratio $0.016$ against $0.114$ for real), and some are fragments of desert
and savanna villages rather than houses.
```

## 18. Figures cut as plots of printed tables: fig:seven, fig:pillars, fig:casestudy

From: `X_appendix.tex:1476-1507 (old Sec. Y)`

Why moved: fig:seven plots tab:seven (its caption said LEGO; tab:seven's caption said in-domain; the data, outputs/run_20261104_212706_bench_lego_ladder_postfix, is the LEGO ladder, so tab:seven's caption is now corrected to LEGO). fig:pillars plots two columns of main tab:leaderboard; fig:casestudy plots tab:casestudy. The PDFs stay in figures/.

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=0.86\linewidth]{figures/fig6_seven_statistics}
\caption{\textbf{Every render-space statistic shares the blindness} (LEGO,
$n{=}128$). Filled marker: the interior-fill probe. Open marker: the weakest of
three stronger corruptions. All seven resolve the stronger corruptions and none
resolves the interior fill. One minus recall also misses the milder
$20\%$-deletion rung at $1.4$ floors, a limit of that statistic rather than of
the probe.}
\label{fig:seven}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.88\linewidth]{figures/fig3_pillars}
\caption{\textbf{The two blindnesses are mirror images.} Distance from held-out
real on the two pillars of the $n{=}256$ leaderboard. \solidify{} is invisible to the render tier and far from
real on geometry; \matshuffle{} is the exact converse, at precisely zero on
geometry.}
\label{fig:pillars}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.9\linewidth]{figures/fig4_casestudy}
\caption{\textbf{The contamination case study} (protocol run, $n{=}128$). \vkid{} with
bootstrap intervals and the Holm groups of the permutation test
(Appendix~\ref{app:significance}). The contaminated arm (c) is not separated from its
clean retrain (cd), and the clean retrain is separated from neither language-model
baseline.}
\label{fig:casestudy}
\end{figure*}
```

## 19. Data provenance: long licence reasoning (full text)

From: `X_appendix.tex:1525-1562 (old Sec. Z)`

Why moved: Sec. M keeps the conclusions, the redistribution policy, the 3D-Craft-only depiction rule and the Mojang attribution verbatim.

```latex
%TL;DR of paragraph: published images are restricted by source, and the texture assets they use are permitted for non-commercial images but not redistributed.
Every real build shown in this paper comes from \textbf{3D-Craft only}; generated builds
are shown as generated. GrabCraft's terms name pictures of blueprints
as a prohibited form of distribution; 3D-Craft's builds were commissioned as a
research corpus and carry no prohibition on depiction. Their license needs
stating precisely, because it is not the one the CraftAssist code carries. The
CraftAssist codebase is MIT, but the house data ships as a tarball with no
license file, and the release that distributes it for the 3D-Craft paper,
\texttt{facebookresearch/voxelcnn}, is CC BY-NC 4.0. We therefore treat the
corpus as non-commercial research material requiring attribution, which is what
our use and this paper's figures are, and we state that restriction on the
release rather than inheriting the code's MIT terms. Figures render these builds
with Minecraft's own texture assets, which are Mojang's copyright. Mojang's
usage guidelines permit non-commercial use of images of the game, and this
paper is non-commercial research; we also reproduce the attribution they
require, and nothing here suggests official endorsement. Two limits follow and
we observe both. The released code and data contain \textbf{no texture files},
so a reader reproducing our figures supplies them from their own installation.
And the permission covers the images, not the assets: we publish renders, never
the pack. The alternative would be flat per-block colors, which need no
permission at all because they leave only third-party-authored geometry, and
the figure scripts still produce them on request; we render with textures
because they show what the metric is actually given. Minecraft features: \copyright{} Mojang Synergies
AB, included courtesy of Mojang AB. This is not an official Minecraft product and
is not approved by or associated with Mojang or Microsoft.

%TL;DR of paragraph: attribution is by URL because creator names were never captured, and we say so.
Builds are credited by source URL in the released manifests. The original scrape
did not capture uploader names, so we cannot credit individual creators, and we
record that as a limitation rather than an omission. Our LEGO evaluation corpus
is StableText2Brick \citep{pun2025brickgpt}, loaded from HuggingFace and not
redistributed. StableText2Brick and the BrickGPT code and weights are MIT-licensed, which
permits LEGO renders in this paper; the dataset derives from ShapeNetCore, whose terms
restrict use to non-commercial research. Dream-Cubed's village checkpoint and its data
repository are Apache-2.0; we run its code for evaluation and do not redistribute it.
\TODO{Insert the institutional determination for the corpus release before
submission. It is an open question for the university's research-compliance
office, not settled by the authors.}
```

## 20. Gate provenance: dated record of the thresholds (full text)

From: `X_gateprov.tex:49-64 ('When the thresholds were set')`

Why moved: Compressed to two sentences in Sec. C.3. Commit dates belong in the release CHANGELOG.

```latex
\paragraph{When the thresholds were set.} The repository history dates each
decision. The ordering gates, the $1$-floor invariance bar, the $2$-floor
self-consistency bar and the $2$-unit $n$-stability bar were committed on
2026-08-04 and validated on render metrics, four weeks before the first
voxel-descriptor metric existed. \vkid{} entered on 2026-08-31 together with
two corrections to the gates' denominators: the noise floor became the spread
of a fresh draw from a pool containing the probe set, with the
finite-population factor, and the $n$-stability test became a test of two means
against the standard error of their difference. The first correction applies
to every metric and is what moved \vkid{} from failing self-consistency at
$3.5$ floors to passing; the second is stricter and rejects the biased MMD
estimator at $14.8$ standard errors. The gate code has been unchanged since
2026-09-07, which also fixes the $77$-dimensional descriptor, and the
head-to-head ran on 2026-11-04 with that code, $58$ days later. The memorization
and diversity bars date from the same commit; the twin gate (2026-11-04) reuses
the duplicate gate's rule.
```

## 21. Reproducibility Statement (whole file)

From: `sec/X_reproducibility.tex (old Sec. AA, label app:repro-statement)`

Why moved: Merged into Sec. M; supp.tex no longer inputs the file (the file is left on disk).

```latex
% Moved from the ICLR main text (sec/9_statements.tex): CVPR has no
% reproducibility-statement slot in the main paper, so it lives in the supplementary.
\section{Reproducibility Statement}
\label{app:repro-statement}

%TL;DR of paragraph: everything needed to reproduce is released, and every number is traceable to a recorded run.
The evaluation suite, probes, procedural baselines and corpus split are
released. Every benchmark run writes a scorecard
recording corpus, split, sample size, reference size, seed, host, device and
commit, so each number in this paper is traceable to the run that produced it;
Sec.~\ref{app:repro} lists these. Two properties bear on replication in
particular: controls are re-scored in every run rather than cached across runs,
because the aggregate's unit is the spread of a control scored under the same
protocol; and every arm must declare the split it was trained on, with arms that
cannot reported as unranked (Fig.~\ref{fig:contamination}).
\TODO{Add the anonymized code URL and commit before submission, and insert the
institutional determination covering the release. The licensing question itself
is settled: Sec.~\ref{app:data} states what we redistribute and what we do
not, and the release contains no upstream content.}
```
