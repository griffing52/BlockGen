# `math_max`: the statistics of BlockScore, its unit, its letters, and its $1/n$

Agent `math_max`. Files I own: this one and `scripts/figs/math_max_check.py`.
Every number below is either quoted from a file I name or produced by that
script, which writes `outputs/figstage/math_max_check.json`.

```
.venv/bin/python scripts/figs/math_max_check.py         # a b c p d, ~20 min CPU
.venv/bin/python scripts/figs/math_max_check.py a b     # theory + measured null only
```

Sources used, all read-only: the run of record
`outputs/run_20260917_043807_bench_canonical-split_native_arms_oriented_vs/scorecard.json`;
`blockgen/eval/bench/{composite,compare,ladder,full,fast,topology,geometry}.py`;
the significance recomputation of 2026-09-19,
`outputs/run_20260919_022424_significance_recompute/{FINDINGS.md,recompute_r1.json}`.

---

## 0. Top matter: what changes at abstract level

**Nothing in the abstract changes, and no number in the paper moves.** I state
this first because the worry in the brief is real but its magnitude is small
against the effects the paper reports. The abstract quotes only *per-pillar*
quantities ($0.38$ and $>200$ noise floors on the appearance and geometry
statistics), never \blockscore{}, so the max-of-$k$ inflation cannot reach it.
The inflation I measure is $\approx 1.1$ noise floors; the smallest ranked
\blockscore{} on the card is $60.80$.

Four things do need to change in the body, in descending order of seriousness:

1. **The case-study table's significance letters and its caption sentence are
   wrong** (`sec/X_appendix.tex`, `tab:casestudy`): the letters are the *legacy*
   test's *greedy* display, and "the contaminated arm occupies a significance
   group of its own" is false under the corrected permutation test --- its
   letter is shared with two other arms. This was already flagged in
   `outputs/run_20260919_022424_significance_recompute/FINDINGS.md` §9.2; it is
   still in the `.tex`. §3 below gives drop-in replacement rows. This is the one
   item that is *wrong* rather than imprecise.
2. **"each in noise floors" is false for 2 of the 5 pillars** (`sec/4_method.tex`
   §\ref{sec:blockscore}). The `coherence` and `geometry_scalars` pillars are
   quoted in a per-build robust spread of the real data, not in a sampling noise
   floor; measured, their declared unit is $29\times$ and $52\times$ larger than
   their real-draw spread. They therefore cannot veto anything, which is the one
   property the max is sold on. §2.
3. **The null of \blockscore{} is $\approx 1.1$ floors, not $0$**, for a
   genuinely-real arm that is not itself the calibration control. The card's
   $0.00$ is an identity, not a measurement. The leaderboard caption in
   `sec/6_results.tex` already says "by construction", which is correct and
   good; what is missing is the number a fresh real arm would score. §1.
4. **One clause in `app:nscale` gives the wrong mechanism** for \mvkid{}'s
   shallower-than-$1/n$ floor. The $1/n$ claim itself is confirmed for \vkid{}
   at $-0.998$ with a 95% interval of $[-1.033, -0.966]$. §4.

| item | verdict |
|---|---|
| 1. max-of-$k$ null | **imprecise** (real bias, correctly disclosed at the table, never quantified) |
| 2. "noise floors" as a unit for a max | **imprecise**, and *wrong* for 2 of the 5 pillars |
| 3. significance groups | implementation **correct** since 2026-09-19; intro wording **correct**; appendix table **wrong** |
| 4. $\sim 1/n$ | **correct** for \vkid{}, **imprecise** for \mvkid{} (one clause states the wrong cause) |

---

## 1. A max of $k$ statistics does not have the null of one statistic

### 1.1 What the implementation actually does

`composite.Calibration.from_arm` takes the floor and the spread of every pillar
from **one arm of the run, `real_test`**, and `composite.leaderboard` then scores
*that same arm* against them. So for the control, and only for the control,

$$z_p \;=\; \frac{v_p(\text{real\_test}) - \mathrm{floor}_p}{\mathrm{sd}_p}
\;=\; \frac{v_p - v_p}{\mathrm{sd}_p} \;=\; 0 \quad\text{exactly, for every } p,$$

hence $\blockscore(\text{real\_test}) = \max_p 0 = 0$ with zero variance. The
card confirms it: `real_test` is row 1 at `0.00` with `+0.00` on all five
pillars. **The calibration does not absorb the max-of-$k$ inflation; it makes it
invisible by self-reference.** Two consequences:

* the top row of the leaderboard is not evidence that the aggregate is
  unbiased, and cannot be;
* `composite.validate`'s check `C1_real_is_near_zero` (`abs(real) < 3.0`) is
  **vacuous**: it compares $0.0$ to $3.0$ on every run. The card's
  "Self-validation: all checks pass (9 checks)" includes it.

$k = 5$ on this card (`appearance`, `geometry`, `palette` from `PILLARS`, plus
`coherence` and `geometry_scalars` from `STRUCTURE_BLOCKS`). $k$ is not fixed
across runs: a FAST-tier card has no `appearance` pillar, and `score_arm` drops a
pillar it cannot compute into `missing` rather than failing, so $k \in \{1..5\}$
in practice.

### 1.2 Derivation

Let $X_p^{A}$ be pillar $p$ measured on an arm and $X_p^{C}$ on the control, both
real draws of the same size, each with population sd $\sigma_p$ and pillar
correlation matrix $\rho$. Since the floor *is* $X_p^{C}$,

$$z_p = \frac{X_p^{A} - X_p^{C}}{\sigma_p}, \qquad
\mathbb{E}z_p = 0, \qquad \operatorname{Var} z_p = 2, \qquad
\operatorname{Corr}(z_p, z_q) = \rho_{pq}.$$

The per-pillar null is therefore $N(0,\sqrt{2})$, **not** $N(0,1)$: the floor is
itself one noisy draw, and that doubles the variance. Under the equicorrelated
model $z_p = \sqrt{2}\left(\sqrt{\rho}\,Z + \sqrt{1-\rho}\,\varepsilon_p\right)$
the max separates exactly, because $\mathbb{E}Z = 0$:

$$\boxed{\;\mathbb{E}\!\left[\blockscore \mid \text{real}\right]
= \sqrt{2}\,\sqrt{1-\rho}\;a_k, \qquad
a_k \equiv \mathbb{E}\!\left[\max_{i\le k} \varepsilon_i\right],\;}$$
$$\operatorname{sd}\!\left[\blockscore \mid \text{real}\right]
= \sqrt{2}\,\sqrt{\rho + (1-\rho)\,v_k}, \qquad
v_k \equiv \operatorname{Var}\!\left[\max_{i \le k}\varepsilon_i\right].$$

The brief's $\sqrt{2\ln k}$ is the correct *asymptote* but a poor approximation
at $k=5$: it overstates $a_5$ by 54%. Exact values (part `a` of the script,
`quad` on $x\,k\,\phi(x)\Phi(x)^{k-1}$, agreeing with $4\times10^5$-sample Monte
Carlo to 3 decimals):

| $k$ | $a_k$ | $\sqrt{2\ln k}$ | $\sqrt{v_k}$ | $\sqrt2\,a_k$ (i.e. $\rho=0$) |
|---|---|---|---|---|
| 2 | 0.5642 | 1.1774 | 0.826 | 0.798 |
| 3 | 0.8463 | 1.4823 | 0.748 | 1.197 |
| 4 | 1.0294 | 1.6651 | 0.702 | 1.456 |
| 5 | **1.1630** | **1.7941** | 0.669 | **1.645** |
| 10 | 1.5388 | 2.1460 | 0.588 | 2.176 |

So the qualitative worry is confirmed: a real arm should sit **above** zero, by
of order one noise floor, purely from the max --- with two caveats that soften
it, correlation ($\sqrt{1-\rho}$) and the fact that $a_k$ at small $k$ is about
half its asymptote.

### 1.3 Measurement

Part `b` of the script measures the null directly on the run of record's own
data, without simulating anything: two **disjoint** 128-build draws from the
399-build held-out test split play "arm" and "control"; all five pillars are
recomputed for each ($\mvkid$ from the cached DINOv2 features of the run's own
view key, $\vkid$ from the same whitened 77-d descriptor, palette JSD, and the
two RMS-of-`w1_norm` blocks); each is $z$-scored with the card's own floors and
spreads; the max is taken. 300 replicates. Faithfulness check: my recomputed
pillar means match the card's `real_test` values (`geom_kid` mean $-0.083$ vs the
card's $-0.0948$; `mv_dino_kid` $-0.0002$ vs $-0.0009$ --- the card's value is
one draw of the same quantity), and my measured `geom_kid` spread at $n=128$ is
$0.1025$ against the card's in-run measured $0.1097$ (7%).

Measured null of \blockscore{} on genuinely held-out real data, in the card's
declared units:

| | mean | median | sd | 90% interval | $P(>3)$ |
|---|---|---|---|---|---|
| arm vs a **fresh** real control | **1.080** | 0.930 | 0.893 | $[0.00, 2.80]$ | 0.033 |
| arm vs the **card's** control values | 0.920 | 0.808 | 0.743 | $[0.02, 2.21]$ | 0.017 |
| same, in honest (measured) per-pillar units | 1.392 | 1.327 | 0.931 | $[0.01, 2.93]$ | 0.043 |
| *the same arms under a **mean** aggregate* | 0.018 | --- | --- | --- | --- |

The max is right-skewed (skew $+0.89$); the largest of 300 null draws was
$4.49$. Derivation against measurement: a Gaussian model with the *measured*
covariance of $z$ gives $\mathbb{E}[\max] = 1.062$ against the measured $1.080$
(2%), so the Gaussian reasoning of §1.2 is adequate. Measured per-pillar
$\operatorname{sd}(z)$ is $1.36$ (appearance), $1.14$ (geometry), $1.29$
(palette) --- the predicted $\sqrt2 = 1.41$, reduced as expected because two
disjoint draws from a 399-build pool are not independent (finite-population
algebra predicts $1.34$; the true independent-draw null is therefore about 5%
*above* what I measure, i.e. $\approx 1.14$). Mean off-diagonal pillar
correlation is $+0.11$ (range $-0.03$ to $+0.37$), so the near-independent
column of the table above is the relevant one.

The offset depends on how many pillars a card happens to have, which matters
because the paper's own runs differ:

| pillars kept | mean null \blockscore{} |
|---|---|
| all 5 (full tier, the run of record) | 1.080 |
| 4, dropping `appearance` (FAST tier) | 0.785 |
| mean over all 3-subsets | 0.763 |
| mean over all 2-subsets | 0.515 |
| any single pillar | 0.018 |

### 1.4 Verdict

**VERDICT: the paper is imprecise, not wrong.** The bias is real and the
calibration does not absorb it. But (i) the paper never quotes a \blockscore{}
near the null --- the smallest ranked value is $60.80$, which is $22\,\sigma$
above the null's mean and $55\times$ its 95th percentile, so no ranking, no
comparison and no number in the paper changes; and (ii)
`sec/6_results.tex`'s leaderboard caption already discloses the mechanism
("\realheld{} sits at $0.00$ on every pillar **by construction**: it is the
control each pillar's spread is calibrated against, so it anchors the scale
rather than winning on it"), which is exactly right. What is missing is the
*number*, so that a reader cannot mistake "0" for "what real data scores". One
sentence fixes it.

Draft, to append to the \blockscore{} paragraph of `sec/4_method.tex`:

```latex
Two properties of a maximum have to be stated, because a reader will otherwise
read $0$ as the value real data takes. First, the control scores exactly $0$ on
every pillar \emph{by construction}: it is the arm each pillar's floor and
spread are taken from, so its \blockscore{} is an identity rather than a
measurement. Second, a maximum of $k$ noisy quantities is positive on average
even when nothing is wrong: with each pillar's floor itself estimated from one
real draw, a pillar's null is $N(0,\sqrt{2})$ rather than $N(0,1)$, and for $k$
pillars of inter-pillar correlation $\rho$ the null of the aggregate is
$\sqrt{2}\sqrt{1-\rho}\,a_k$ with $a_k = \mathbb{E}\max_{i \leq k} N(0,1)$
($a_5 = 1.16$). Measured on this run --- two disjoint $128$-build draws of
held-out real data scored against each other through the identical pipeline,
$300$ replicates --- a genuinely real arm scores $\mathbf{1.08}$ on average
(median $0.93$, $95$th percentile $2.80$), against $0.02$ for a mean-aggregated
score. That offset is the price of the veto, and it is $1.8\%$ of the smallest
ranked score in Table~\ref{tab:leaderboard}. A \blockscore{} should therefore be
read against a null of about one floor, not of zero.
```

---

## 2. Is "noise floors" a coherent unit for a max?

### 2.1 What the unit is, pillar by pillar

The three `PILLARS` are single distribution distances divided by a spread; the
two `STRUCTURE_BLOCKS` are an RMS over a family of `w1_norm` entries, whose
spread `Calibration.from_arm` sets to $1.0$ --- *"their unit is 1 by
construction"*. It is not. Tracing `topology.coherence_report` and
`geometry.geometry_report`: `w1_norm` $=$ $W_1$(arm, real) divided by
`stats.robust_spread(real)`, which is $\max(\mathrm{IQR}, \mathrm{sd})$ of the
**per-build** values on the reference. That is an effect-size unit (how far the
distributions are apart, in units of how much builds differ from each other),
not a sampling noise floor. Measured fresh-draw spreads of each pillar in the
unit the card divides by (script part `b`):

| pillar | declared unit | measured real-draw sd | declared / measured | $\operatorname{sd}(z)$ |
|---|---|---|---|---|
| appearance (\mvkid{}) | $0.003055$ (measured, 48 draws) | $0.003269$ | **0.93** | 1.36 |
| geometry (\vkid{}) | $0.109654$ (measured, 48 draws) | $0.106627$ | **1.03** | 1.14 |
| palette (JSD) | $0.009848$ (bootstrap CI) | $0.009347$ | **1.05** | 1.29 |
| coherence (RMS $W_1$) | $1.0$ (by fiat) | $0.034088$ | **29.3** | 0.040 |
| geometry scalars (RMS $W_1$) | $1.0$ (by fiat) | $0.019373$ | **51.6** | 0.026 |

Two findings here, one reassuring and one not.

* The reassuring one: the in-run *measured* noise floors are accurate to within
  7%, and the palette pillar's bootstrap CI --- which `Calibration`'s own
  docstring warns is the wrong quantity for a set-level statistic --- is
  accurate to 5% on this metric. The three live pillars really are in noise
  floors.
* The other: the two structure pillars are quoted in units $29\times$ and
  $52\times$ coarser than their own noise floor. They are consequently **not
  vetoes**. To win a max against a live pillar at, say, $+60$, a structure
  pillar would have to move $60$ of its declared units, i.e. $\approx 1\,800$
  and $3\,100$ of its actual noise floors. On the card, no arm's worst pillar is
  ever `coherence` or `geometry_scalars` --- for all 15 ranked arms it is
  `appearance` or `geometry` --- and the most damaged control in the run,
  \randomvox{}, reaches only $+15.71$ on coherence against $+562.83$ on
  appearance. The claim that "every pillar is a veto" holds for three pillars of
  five.

### 2.2 What "15" means

Precisely: *the worst of five quantities, each the difference between this arm's
pillar value and the control's, divided by that pillar's own scale, was $15$;
for the three live pillars that scale is one fresh-real-draw sd, for the other
two it is one per-build robust spread of the real data.* It is therefore:

* **not** a distance in the units of anything in particular, unless the winning
  pillar is named --- which the card and Table~\ref{tab:leaderboard} both do, via
  the `worst pillar` column. With that column the number is interpretable; on
  its own it is not;
* **not** convertible to a probability or a $z$: the null is $\max$-shaped
  ($\mathbb{E}=1.08$, sd $0.89$, skew $+0.89$), so $15$ is not "15 sd from the
  null", it is "well outside a null whose 95th percentile is $2.8$";
* **not** comparable across cards with different pillar counts or different
  correlations without a correction: measured, dropping the appearance pillar
  moves the null from $1.08$ to $0.785$; $\rho$ enters as $\sqrt{1-\rho}$;
* comparable **ordinally** within a run at fixed $n$, reference size and control
  size, which is what the paper uses it for, and which `app:repro` already pins
  as a protocol.

One more consequence of the "floor is itself one draw" algebra, which applies to
the *per-pillar* numbers the paper leans on rather than to the max: the
difference between an arm and the control has null sd $\sqrt2 \approx 1.41$ in
these units, not $1.0$. So \solidify{}'s $+0.38$ appearance reading is $0.27$ of
its own null sd, i.e. even deeper inside the null than "$0.38$ floors"
suggests. This is the conservative direction for the paper's headline.

### 2.3 Verdict

**VERDICT: imprecise, and wrong in one clause.** "Each in noise floors" is false
for `coherence` and `geometry_scalars`. The aggregate is ordinally meaningful
within a protocol and should be described that way rather than as a distance.

Draft replacement for the unit sentence in `sec/4_method.tex`, and a companion
sentence for `app:repro`:

```latex
\blockscore{} is the \textbf{maximum} over five pillars --- appearance,
geometry, palette, coherence, geometric scalars. The first three are
distribution distances expressed in noise floors calibrated in-run against the
held-out real control; the last two are Wasserstein distances on per-structure
scalars, expressed in the real data's own per-build spread, which is a coarser
unit --- measured, $29\times$ and $52\times$ a noise floor at $n{=}128$ --- so
in practice they report rather than veto. There are no weights to tune.
```

```latex
\item \textbf{\blockscore{} is ordinal, not a distance.} It is the worst pillar's
  standardized distance from the run's own real control, and it is only
  meaningful beside the name of that pillar, which the leaderboard prints. Two
  cards' aggregates are comparable when corpus, split, tier, seed, reference
  size, control size, $n$ \emph{and the set of pillars scored} all match: the
  null of a maximum grows with the number of pillars ($1.08$ floors at five
  pillars, $0.79$ at four, measured on held-out real data), so a card that could
  not compute a pillar is not on the same scale as one that could.
```

---

## 3. Significance groups are not equivalence classes

### 3.1 (a) What the implementation does

Two separate things were wrong here and one has been fixed:

* **The display.** Until 2026-09-19, `compare._letter_groups` assigned one
  letter per arm by greedy first-fit. Since "not separated" is not transitive,
  that display supports the forward reading (*share a letter $\Rightarrow$ not
  separated*) but **not** the reverse (*no shared letter $\Rightarrow$
  separated*). It now enumerates the maximal cliques of the not-separated graph
  (Bron--Kerbosch with a pivot) and an arm carries one letter per clique, which
  gives the biconditional. An uncomputable comparison counts as not-separated,
  which can only merge groups.
* **The test.** The card's $p$-values came from a miscalibrated subsampling SE
  (type-I error $7.2\times$ nominal in the two-generators regime) and were
  replaced by an exact label-permutation test. Both changes are documented in
  `outputs/run_20260919_022424_significance_recompute/FINDINGS.md`.

I reconstructed both displays from the 153 stored pairwise verdicts (script part
`d`) and reproduce the stored letter maps exactly for all four
(test $\times$ display) combinations. Non-transitivity is not hypothetical on
this card:

| metric | test | separated | maximal cliques | intransitive triples | arms with $>1$ letter |
|---|---|---|---|---|---|
| \vkid{} | legacy (the card) | 140/153 | 11 | 3 | 3 |
| \vkid{} | permutation | 138/153 | 12 | 7 | 4 |
| \mvkid{} | legacy (the card) | 146/153 | 13 | 2 | 1 |
| \mvkid{} | permutation | 140/153 | 12 | 6 | 5 |

An "intransitive triple" is $a \sim b$, $b \sim c$, $a \not\sim c$: exactly the
configuration no single-letter partition can display. The card's \vkid{} table
shows 18 single letters where the valid display needs `fg`, `fg` and `hi`.

### 3.2 (b) Could the paper's phrasing mislead?

Two phrasings, and they differ.

* `sec/1_intro.tex`: "the solidified corpus and genuinely held-out real data
  *share a significance group*." This is the **forward** reading, which the
  greedy display did support, and it survives every change: under the corrected
  permutation test the pair is not separated at $p_{\text{adj}} = 0.77$ (was
  $0.61$), both arms hold letter **a**, and the tie got *further* from
  significance. The direction is also the conservative one, as the brief says. I
  would still add five words, because "share a significance group" invites
  "statistically indistinguishable", and a non-rejection over 153 Holm-corrected
  pairs is weak evidence of sameness; the paper already does the right thing by
  arguing magnitude instead (\S\ref{sec:ablation}: $9$--$57$ floors with
  cutaways), so the fix is only to label the argument.
* `sec/X_appendix.tex`, `tab:casestudy` caption: "The contaminated arm
  **occupies a significance group of its own**." This is the **reverse**
  reading, and it is the sentence the corrected code's docstring names as
  unsupported. Worse, it is now also false on the merits: under the corrected
  test the contaminated arm's letter **d** is shared with \voxar{} (clean),
  whose letters are `def`, and with \llmft{}, whose letters are `de`. The
  letters printed in the table (`d`, `e`, `e`, `e`, `f`) are the legacy test's
  greedy letters.

The rest of that caption --- "its clean replacement ties with the baselines it
appeared to beat" --- survives: `canon_oriented` (`def`) shares `e` with
`llm_bias` (`e`) and `d`,`e` with `llm_control` (`de`), and the corrected
$p_{\text{adj}}$ for clean-vs-\llmft{} is $0.93$.

### 3.3 (c) Minimal honest phrasing

For the intro, one clause:

```latex
\ldots{}and in a paired, Holm-corrected \citep{holm1979simple} comparison over
all $153$ arm pairs no test separates the solidified corpus from genuinely
held-out real data ($p_{\text{adj}} = 0.77$). Failure to separate is not
evidence of equivalence, so we rest the claim on magnitude rather than on the
non-rejection:\ldots{}
```

For the case-study table, replace the caption sentence and the `group` column
(corrected \vkid{} letters, permutation test, Holm over the run's own 153 pairs;
source `recompute_r1.json` `metrics.geom_kid.groups_cld_permutation`, identical
in the independent replicate `r2`):

```latex
\caption{Retraining on a clean split collapses the margin. \vkid{} at
$n{=}128$, $399$-build reference, paired and Holm-corrected over all $153$ pairs
with an exact label-permutation test. Letters mark maximal sets of arms that no
test separated at $\alpha=0.05$; a shared letter means the pair was
\emph{not separated}, not that the arms are equivalent, and an arm carries
several letters when non-separation is not transitive. The contaminated arm is
not separated from either its own clean retrain or the stronger language-model
baseline, and the clean retrain ties with both baselines it appeared to beat.}
```

| row | current letter | corrected letters |
|---|---|---|
| \realheld{} (floor) | a | a |
| \traincopy{} | b | b |
| \voxar{}$^\dagger$ (contaminated) | **d** | **d** |
| \voxar{} (clean) | **e** | **def** |
| \llmft{} | **e** | **de** |
| \llmgeo{} | **e** | **e** |
| \voxarun{} (clean) | f | **fgh** |

(The contaminated row keeps the bare letter `d`; what changes is that `d` is no
longer unique to it, which is precisely why the "group of its own" sentence has
to go.)

If an equivalence claim is ever wanted rather than a failure to separate, it
needs an equivalence test (TOST) against a pre-registered margin --- for this
suite the natural margin is one noise floor. The paper does not make such a
claim and should not start; the magnitude argument is stronger and already
there.

### 3.4 Verdict

**VERDICT: implementation correct (since 2026-09-19); intro phrasing correct but
should name its own weakness; the appendix case-study table is wrong** --- its
letters are from the superseded test and the invalid display, and its "group of
its own" sentence is contradicted by the corrected verdicts. The headline's
direction is safe, as the brief anticipated.

---

## 4. The noise floor falls as $1/n$

### 4.1 Derivation

Write the reported statistic as the unbiased two-sample MMD$^2$ with a fixed
reference $Y = \{y_j\}_{j\le R}$ (the suite scores every arm against the same
reference, and never resamples it):

$$\widehat{M}(X) = \frac{1}{n(n-1)}\sum_{i \neq i'} k(x_i, x_{i'})
\;+\; \underbrace{\frac{1}{R(R-1)}\sum_{j\neq j'} k(y_j,y_{j'})}_{\text{constant given } Y}
\;-\; \frac{2}{nR}\sum_{i,j} k(x_i,y_j).$$

Conditional on $Y$, this is a degree-2 U-statistic in $X$ with kernel
$h(x,x') = k(x,x') - v(x) - v(x') + \text{const}$, $v(x) = R^{-1}\sum_j k(x,y_j)$.
Its Hoeffding decomposition gives

$$\operatorname{Var}\widehat{M} = \frac{4(n-2)}{n(n-1)}\zeta_1
+ \frac{2}{n(n-1)}\zeta_2
\;\approx\; \frac{4\zeta_1}{n} + \frac{2\zeta_2}{n^2},
\qquad \zeta_1 = \operatorname{Var}_P\!\big(u(X) - v(X)\big),$$

with $u(x) = \mathbb{E}_{X'\sim P}k(x,X')$ and $\zeta_2 = \operatorname{Var} h$.

* **$P = Q$ and $R = \infty$:** $v \equiv u$, so $\zeta_1 = 0$. The statistic is
  a **degenerate** U-statistic, $\operatorname{Var} = O(n^{-2})$, and
  $\operatorname{sd} = O(1/n)$. This is the paper's claim, and it is the right
  one for a real-vs-real null. (The null is then a weighted $\chi^2$ mixture,
  $\frac1n\sum_i \lambda_i(\chi^2_{1,i}-1)$, which is also why its tail is not
  normal --- relevant to `app:significance`, not to the scaling.)
* **$P = Q$, $R$ finite:** $u - v$ is a Monte-Carlo error of size
  $O(R^{-1/2})$, so $\zeta_1 = O(1/R)$ and
  $\operatorname{sd} \approx \sqrt{a/n^2 + b/(nR)}$. The $1/n$ law holds while
  $n \ll n^{*} \propto R$, and crosses over to $1/\sqrt{n}$ above it.
* **$P \neq Q$:** $\zeta_1$ is $O(1)$ and $\operatorname{sd} = \Theta(1/\sqrt n)$
  --- the regime the head-to-head recomputation independently ran into
  (`FINDINGS.md` §3, subsampling factors $4$ vs $2$).

So "$\sim 1/n$" is a statement about a *null with a large reference*, and the
paper's consequence (separations in floors grow $\propto n$, because the
numerator's expectation is $n$-free for an unbiased estimator) follows only in
that regime.

### 4.2 Measurement

Script parts `c` and `p`, on the run of record's own protocol: probe draws from
the 399-build held-out test split, reference the 399-build val split, spreads
corrected for the finite pool exactly as `ladder._noise_floor` does. 1000 draws
per $n$, bootstrap interval on the exponent over 400 resamples:

| metric | $\mathrm{sd}$ at $n = 16/32/64/128/192$ | exponent | 95% CI | $\mathrm{sd}(128)/\mathrm{sd}(64)$ |
|---|---|---|---|---|
| \vkid{} | 0.8727, 0.3962, 0.1986, 0.1025, 0.0726 | $\mathbf{-0.998}$ | $[-1.033, -0.966]$ | **0.516** |
| \mvkid{} | 0.02081, 0.00963, 0.00533, 0.00316, 0.00255 | $\mathbf{-0.842}$ | $[-0.871, -0.814]$ | 0.592 |

Both conditions are genuine nulls (mean at $n{=}128$: $-0.083$ and $-0.0002$).
**\vkid{}'s exponent is $1.00$ to within $\pm 0.035$, and $1/\sqrt n$ is
excluded by 14 interval widths.** The paper's own two-point ratio for \vkid{}
($0.49$, `app:nscale`) is reproduced here as $0.516$. \mvkid{}'s exponent is
significantly shallower than $1$ and significantly steeper than $\tfrac12$ ---
i.e. it sits *between* the two regimes, which is exactly the finite-reference
cross term.

Three controls confirm the mechanism rather than asserting it (script part `c`):

* shrinking the reference to a **random** $R = 200$ / $100$ keeps the null
  ($\mathbb{E}\approx 0$) and moves \vkid{}'s exponent from $-0.93$ to $-0.87$
  and $-0.84$, and raises the floor itself at $n{=}128$ from $0.104$ to $0.145$
  and $0.179$: the floor is a function of $R$ as well as of $n$;
* using the **first** $R$ rows of the split instead --- which turns out to be a
  biased subsample, $\mathbb{E}[\vkid] = +1.22$ and $+7.58$ --- gives exponents
  $-0.536$, $-0.507$ (\vkid{}) and $-0.521$, $-0.502$ (\mvkid{}): $1/\sqrt n$,
  to two decimals, in four independent measurements. That is the
  non-degenerate branch of the derivation, measured;
* a second $P\neq Q$ reading (train pool against the val reference, also not a
  null on this corpus, $n$ up to 512) gives $-0.56$ overall and $-0.48$ over
  $n \in [128, 512]$.

Two consequences worth stating in the paper:

1. "$202$ at $n{=}128$ becomes roughly $400$ at $n{=}256$" is **defensible for
   \vkid{}** --- the exponent is $-1.00$ over $n \in [16,192]$ with no detectable
   cross term --- but it is an extrapolation one rung past the measured range,
   and it requires the reference to stay at $399$.
2. The same rule must **not** be applied to \mvkid{}: at exponent $-0.84$ a
   separation in \mvkid{} floors grows like $n^{0.84}$, not $n$. The paper's
   $n{=}64 \to 128$ ratio table already shows this; only its explanation is
   wrong.

### 4.3 Verdict

**VERDICT: correct for \vkid{} --- measured $-0.998\,[-1.033,-0.966]$ --- and the
"degenerate U-statistic" attribution is right. Imprecise for \mvkid{}:** the
clause "\mvkid{} scales less cleanly ($0.74$ rather than $0.49$) because its
separation is a ratio of two quantities that are both near zero" explains a
property of the *separation* to account for a property of the *noise floor's
scaling*, which is a different quantity and does not follow. The cause is a
non-degenerate variance component from the fixed finite reference.

Draft replacement for that paragraph of `app:nscale`:

```latex
The exponent is worth measuring rather than inferring from two points. Fitting
$\mathrm{sd} \propto n^{-\alpha}$ over $n \in \{16,32,64,128,192\}$ at $1000$
draws per rung, on the same held-out pool and the same $399$-build reference the
run uses, gives $\alpha = 1.00$ $(95\%\ [0.97, 1.03])$ for \vkid{} --- the $1/n$
decay of a degenerate U-statistic, with $1/\sqrt{n}$ excluded --- and
$\alpha = 0.84$ $(95\%\ [0.81, 0.87])$ for \mvkid{}. The gap is not a defect of
the unit but the second term of the variance: conditional on a \emph{fixed}
reference of $R$ builds, $\operatorname{Var} \approx 4\zeta_1/n + 2\zeta_2/n^2$
with $\zeta_1 = O(1/R)$, so the $1/n$ law holds only while $n \ll n^{*}
\propto R$ and crosses over to $1/\sqrt{n}$ above it; $n^{*}$ is a property of
the kernel, and it is larger than our whole range for \vkid{} and of order
$60$ for \mvkid{}. Two controls fix the interpretation: shrinking the reference
to a random $100$ builds flattens \vkid{}'s exponent to $0.84$ and raises its
floor at $n{=}128$ by $74\%$, and making the reference genuinely mismatched
(so that $P \neq Q$ and $\zeta_1 = O(1)$) gives $0.51$--$0.54$ for both metrics,
i.e.\ exactly $1/\sqrt{n}$. Consequently a separation in \vkid{} floors grows
$\propto n$ and ``$202$'' means ``$202$ at $n{=}128$ against a $399$-build
reference'', while a separation in \mvkid{} floors grows only like $n^{0.84}$.
```

---

## 5. Recommendations for the suite (not implemented --- not my files)

Ordered by value; each is a small change in `blockgen/eval/bench/`.

1. **Make `C1_real_is_near_zero` mean something.** It currently compares $0.0$
   to $3.0$. Score a *second*, disjoint real draw (`real_val`, or a held-out
   half of `real_test`) as an ordinary arm and assert *its* \blockscore{} is
   below the measured null's upper quantile ($\approx 3.5$ at five pillars).
   That is the check the suite thinks it is making, and it would also put the
   null on every card.
2. **Give the structure pillars a real unit.** `Calibration.from_arm` sets their
   spread to $1.0$ "by construction"; the machinery to measure it already exists
   (`ladder.noise_floor` over held-out real draws, as `runner.measured_noise`
   does for the two kernel statistics). Measured here, the correction is
   $29\times$ and $52\times$ --- and it would make them able to veto. This is the
   one change that would alter numbers: the run of record's ranking is unchanged
   (no arm's structure pillars would overtake its live ones at these
   magnitudes), but arms near real would move.
3. **Record $k$ and the null on the card.** `BlockScore.to_json` already carries
   `missing_pillars`; adding the null offset for the pillar set actually scored
   would make two cards comparable instead of only orderable.
4. **Re-render the run of record's `scorecard.md` with the current
   `compare.py`** (`--sig-method subsample` reproduces the old card exactly, so
   nothing is lost), so that no table in the repository carries greedy letters.
   Every card written before 2026-09-19 has them.

## 6. Provenance of every number in this file

| number(s) | source |
|---|---|
| `real_test` \blockscore{} $=0.00$, all pillars $+0.00$; leaderboard values | `outputs/run_.../scorecard.md`, `scorecard.json` |
| floors, declared spreads, in-run measured noise sd | `scorecard.json` `context.noise_floor_sd` + `arms.real_test`, via `math_max_check.card_calibration()` |
| $a_k$, $\sqrt{v_k}$, $\mathbb{E}[\max]$ vs $\rho$ | script part `a` |
| measured null of \blockscore{}, per-pillar measured sd, correlations, null by pillar subset, Gaussian-model check | script part `b`, 300 disjoint draw pairs |
| exponents with intervals, floors vs $n$, $R$-sweep, $P\neq Q$ controls | script parts `p` and `c` |
| separated-pair counts, cliques, intransitive triples, corrected letters | script part `d`, reconstructed from `outputs/run_20260919_022424_significance_recompute/recompute_r1.json` (reproduces its stored letter maps exactly) |
| corrected $p_{\text{adj}}$ values ($0.77$, $0.93$, $0.079$) | `outputs/run_20260919_022424_significance_recompute/FINDINGS.md` §5 |

All of it is also in `outputs/figstage/math_max_check.json`.
