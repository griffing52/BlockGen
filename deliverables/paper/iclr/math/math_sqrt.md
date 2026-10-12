# `math_sqrt` — audit of contribution 4 (mixture identity, $\sqrt{\mathrm{KID}}$ readout)

## Does anything here change the abstract?

**No.** `sec/0_abstract.tex` makes no claim about the mixture identity, the quadratic
response, or the square-root readout, so nothing in this audit touches it.

**One number in the paper is wrong and must change: the intro bullet's `0.119`**
(`sec/1_intro.tex:91`). It is not reproducible from the run of record; it comes from a
superseded $n{=}64$ run whose bad pool the code itself records as *disavowed*. §6.5's
numbers (`0.011 / 0.124 / 0.022 / 0.135`) are all correct. Details in item 3.

Two further items need new text rather than corrections: the citation to Bińkowski et
al. Appendix B.2 points at a theorem that says something different from (and more
interesting than) what the paper attributes to it (items 1, 4), and the paper's claim
that Fréchet distance "is not polynomial in $\alpha$" is, as far as I can verify, not in
the cited paper — but it is true, and I supply the derivation and the measurement
(item 4).

Everything below was produced by `scripts/figs/math_sqrt_check.py` (owned by this agent;
run with `.venv/bin/python scripts/figs/math_sqrt_check.py`) or read out of the recorded
run JSONs named inline. No number in this file is an estimate from memory.

---

## Item 1 — the identity, its proof, and the direction of the mixture

### Result

The equation as printed in `sec/4_method.tex` (Eq. `eq:mixture`) is **correct**, and the
role of $\alpha$ in it is the *clean* fraction. It matches the code: in
`scripts/bench_doseresponse.py:ladders`, `mix(damaged, alpha)` keeps
`k = round(alpha * n)` **real** builds and the dose axis is recorded as `1 - alpha`, the
bad fraction. So equation, code and the surrounding prose ("linear in the contaminated
fraction" = linear in $1-\alpha$) all agree. There is no sign or role swap.

### Proof

Let $k$ be a fixed positive-definite kernel with RKHS $\mathcal{H}$ and canonical
feature map $\phi(x) = k(\cdot, x)$, and for a distribution $R$ with
$\mathbb{E}_R\sqrt{k(x,x)} < \infty$ let $\mu_R = \mathbb{E}_{x\sim R}\,\phi(x) \in
\mathcal{H}$ be its mean embedding (Bochner integral). Then
$\mathrm{MMD}^2(R,S) = \lVert \mu_R - \mu_S\rVert_{\mathcal{H}}^2$, which expanded by
the reproducing property is exactly Eq. `eq:mmd`.

Fix $\alpha \in [0,1]$ and let $R_\alpha = \alpha P + (1-\alpha) Q$. The mixture is a
measure, and integration is linear in the measure, so for every $f \in \mathcal{H}$,
$\mathbb{E}_{R_\alpha} f = \alpha\,\mathbb{E}_P f + (1-\alpha)\,\mathbb{E}_Q f$; taking
$f = \phi(\cdot)$ componentwise gives
$$\mu_{R_\alpha} = \alpha\,\mu_P + (1-\alpha)\,\mu_Q .$$
Hence $\mu_{R_\alpha} - \mu_P = (1-\alpha)(\mu_Q - \mu_P)$ and, by absolute homogeneity
of the norm,
$$\mathrm{MMD}^2(R_\alpha, P) = (1-\alpha)^2\,\lVert\mu_Q-\mu_P\rVert^2
= (1-\alpha)^2\,\mathrm{MMD}^2(Q,P). \qquad\blacksquare$$

Assumptions actually used: the kernel is **fixed** (not chosen from the data being
compared — see item 4 for why this bites in this codebase), the embeddings exist (true
for any bounded kernel; the suite's polynomial kernel on $\ell_2$-normalised features is
bounded), and $\alpha\in[0,1]$ so that $\lvert 1-\alpha\rvert = 1-\alpha$. Unbiasedness
transfers the identity to the estimator *in expectation* exactly: since
$\mathbb{E}[\widehat{\mathrm{MMD}}^2_u] = \mathrm{MMD}^2$ for i.i.d. draws, the
$(1-\alpha)^2$ law holds for the reported estimator's expectation. It does **not** hold
for the biased estimator, whose $O(1/n)$ diagonal term depends on $R_\alpha$ and so
adds an $\alpha$-dependent contribution that is not $(1-\alpha)^2$ — an extra argument
for the paper's §4.3 point that the biased estimator is unusable.

### Two imprecisions worth fixing

**(a) $\alpha$ is never defined in the paper.** The contamination literature almost
always writes the Huber form $(1-\varepsilon)P + \varepsilon Q$ with $\varepsilon$ the
*bad* fraction, and everything in the paragraphs around Eq. `eq:mixture` is phrased in
terms of the bad fraction ("an arm failing on a quarter of its builds", "the fraction of
an arm that is bad"). A reader who imports that convention reads $\alpha$ as the bad
fraction and gets the roles backwards, even though the equation stays true. One clause
fixes it.

**(b) The identity assumes the arm's good builds are distributed exactly as the
reference.** That is a real assumption, and it is where the measured exponent's
deviation from $2$ comes from. Writing the arm as $R_\beta = (1-\beta)G + \beta Q$ with
$G$ the generator's *good* component, $\beta$ the bad fraction and $P$ the reference,
$$\mathrm{MMD}^2(R_\beta,P) = \underbrace{\lVert\mu_G-\mu_P\rVert^2}_{\theta_0}
 + \beta\underbrace{2\langle \mu_G-\mu_P,\ \mu_Q-\mu_G\rangle}_{L}
 + \beta^2\underbrace{\lVert\mu_Q-\mu_G\rVert^2}_{\mathrm{MMD}^2(Q,G)} .$$
The response is an exact quadratic in $\beta$ for *any* generator, but the pure
$\beta^2$ law needs $\mu_G=\mu_P$ (i.e. $G=P$ for a characteristic kernel), which no
real generator satisfies. Consequences: $\sqrt{\mathrm{MMD}^2}$ is linear in $\beta$
only in the idealised case, and the cross term $L$ may have either sign, so a fitted
exponent can land on either side of 2.

Measured coefficients (exact, computed from the real feature pools, not fitted —
`part_e`/`part_d` of the check script; $\theta_0, L, Q_2$ in the suite's
`scale=1000` units):

| setting | $\theta_0$ | $L$ | $Q_2=\mathrm{MMD}^2(Q,G)$ | $\lvert L\rvert/Q_2$ | exponent of the exact curve |
|---|---|---|---|---|---|
| `geom_kid`, real `uniform_random` pool | 0.0240 | $-0.6496$ | 39.375 | 0.017 | **2.036** |
| `mv_dino_kid`, LEGO builds as the bad pool | 0.0023 | $+0.0248$ | 1.018 | 0.024 | **1.977** |
| `mv_dino_kid`, mean-shift surrogate ($L\approx0$ by construction) | 0.0023 | $-0.0003$ | 1.718 | 0.000 | **2.011** |

The recorded ladder fits are 1.99 (`mv_dino_kid`) and 2.16 (`geom_kid`) on
`mixture_uniform_random`. Both are now explained rather than merely tolerated: the
cross term shifts the exponent (negative $L$ pushes it above 2, which is the `geom_kid`
case) and the finite-sample term below does the rest.

### Bonus: the exact finite-sample correction for the mixture the ladder actually builds

The ladder does not draw the arm i.i.d. from the mixture; it puts **exactly**
$k=\beta n$ bad builds in the arm. For the unbiased estimator against a reference drawn
independently, in the idealised case $G=P$, the expectation is exactly
$$\mathbb{E}\big[\widehat{\mathrm{MMD}}^2_u\big]
= \mathrm{MMD}^2(Q,P)\Big(\beta^2 - \frac{\beta(1-\beta)}{n-1}\Big),$$
(derived by counting the $PP$, $QQ$ and $PQ$ pairs in the off-diagonal sum; verified to
$2.7\times10^{-15}$ against the general formula over random kernels in `part_f`). So a
stratified mixture reads slightly *below* $\beta^2$, by $2.4\%$ relative at $\beta=0.25,
n=128$, and a power fit to that curve returns an exponent of **2.011** at $n{=}128$ and
**2.023** at $n{=}64$ (`part_a`). This is small, but it means the ladder's "best-fit
exponent 1.99 against the identity's 2" is being compared against a prediction that is
itself 2.011 at that $n$ — worth one clause in the appendix, and it is *the* reason not
to over-interpret a fitted exponent of 1.99 as a deviation.

### VERDICT

**Correct but imprecise.** The equation, its direction, and the prose agree with each
other and with the code; no sign error. Two clauses are missing: $\alpha$ is undefined
(and the field's usual convention is the opposite one), and the identity's requirement
that the arm's clean component match the reference is unstated even though it is what
makes the measured exponent differ from 2. The proof is one line and should be in the
appendix rather than asserted, because the paper's appendix has room and the claim now
carries the extra quadratic-with-cross-term result, which is *not* in Bińkowski et al.

---

## Item 2a — the estimator can be negative; what the code does

### Result

`sqrt(max(v, 0.0))`. Three places, identical convention:

- `blockgen/eval/bench/scorecard.py:193` — `with_mmd`: `m.extra["mmd"] = float(np.sqrt(max(m.value, 0.0)))`
- `scripts/bench_doseresponse.py:183` — `root = np.sqrt(np.maximum(v, 0.0))` (and the
  denominator `np.sqrt(max(worst, 0.0))`)
- `scripts/bench_hollow.py:230` — `"mmd": float(np.sqrt(max(point, 0.0)))`

So: **clamped at zero. Never NaN, never a signed square root.** The paper's
recommendation is therefore well-defined as implemented, at every value the estimator
can take. The paper does not say this anywhere, and it should, because the clamp is not
a rare rescue:

Measured $P(\widehat{\mathrm{KID}}<0)$ for a *clean* arm of $n{=}128$ real builds
against the 399-build reference (400 Monte-Carlo draws, `part_b`/`part_c`):

| metric | $P(\widehat{\mathrm{KID}}^2 < 0)$ at $n{=}128$ |
|---|---|
| `mv_dino_kid` | **0.265** |
| `geom_kid` | **0.485** |

and across $n$ (`part_g`, 600 draws): `geom_kid` 0.56 / 0.49 / 0.49 / 0.41 and
`mv_dino_kid` 0.49 / 0.34 / 0.27 / 0.12 at $n = 32/64/128/256$. A leaderboard's top
entries sit exactly here, so the clamp fires on a quarter to a half of clean arms.

What the clamp costs, and what the paper should say:

1. The readout is non-negative by construction, so a printed `mmd = 0.000` is not
   evidence of a distributional match — it is the estimator having landed on the
   negative half.
2. The clamp is only *weakly* monotone: it collapses the whole negative half-line to a
   single point. `scorecard.with_mmd`'s docstring justifies the readout as "a monotone
   transform, so it inherits the ordering gates". Monotone, yes; **strictly** monotone,
   no — two arms both at the floor are tied by construction, so any ordering between
   near-real arms is destroyed by the readout even where the underlying estimator
   ordered them. For a readout whose stated purpose is "what fraction of this arm's
   builds are bad", that is acceptable; for ranking it is not, and the docstring's
   "companion readout, not a ranking metric" is the right call.
3. `with_mmd` attaches a point value and **no interval**, although the metric it
   decorates carries a CI (`sc.from_ci(... stats.subsample_ci ...)` in `fast.py:363`).
   Because $\sqrt{\cdot}$ is monotone, transforming the two CI endpoints is exact and
   free: $[\sqrt{\max(\ell,0)}, \sqrt{\max(u,0)}]$ is a valid interval for
   $\mathrm{MMD}$ at the same coverage. Recommending a readout with a known
   $n$-dependent floor (item 2b) while printing it without an interval is the one real
   gap here.

### VERDICT

**Correct as implemented, incomplete as written.** The recommendation is well-defined
(clamped), but the paper never states the clamp, never states that the negative event is
common rather than pathological, and prints the readout without the interval that a
monotone transform gives for free. One sentence in §4.4 and one CI change in
`with_mmd` (not mine to make) close it.

---

## Item 2b — the Jensen/delta-method bias, derived and measured

### Derivation

Write $\widehat{X}$ for the unbiased estimator, $\theta = \mathbb{E}\widehat X =
\mathrm{MMD}^2 > 0$, $\sigma^2 = \operatorname{Var}\widehat X$, $g(x)=\sqrt x$. With
$g'(\theta) = \tfrac{1}{2}\theta^{-1/2}$ and $g''(\theta) = -\tfrac14\theta^{-3/2}$,
$$\mathbb{E}\big[\sqrt{\widehat X}\big]
= \sqrt\theta - \frac{\sigma^2}{8\,\theta^{3/2}} + O\!\big(\mathbb{E}(\widehat X-\theta)^3\big),
\qquad
\operatorname{sd}\big(\sqrt{\widehat X}\big) \approx \frac{\sigma}{2\sqrt\theta},$$
so the **relative** downward bias is, to leading order,
$$\frac{\mathbb{E}\sqrt{\widehat X}}{\sqrt\theta} - 1 \;\approx\; -\frac{1}{8\,\mathrm{SNR}^2},
\qquad \mathrm{SNR} := \theta/\sigma,$$
i.e. it is governed entirely by how many of its own standard errors the metric sits
above zero. Jensen guarantees the sign ($\mathbb{E}\sqrt{\widehat X}\le\sqrt\theta$ for
the unclamped root); the expansion is valid only for $\mathrm{SNR}\gg1$ and is
meaningless at the floor, where the clamp — not the concavity — decides the answer.

At the floor the right calculation is not an expansion but a direct one. For a pure
null ($\theta=0$, $\widehat X \approx \sigma Z$ with $Z$ approximately standard),
$$\mathbb{E}\big[\sqrt{\widehat X^{+}}\big] = \sqrt{\sigma}\;\mathbb{E}\sqrt{Z^+}
= 0.41109\,\sqrt{\sigma},
\qquad \mathbb{E}\sqrt{Z^+} = \frac{2^{1/4}\Gamma(3/4)}{2\sqrt\pi} = 0.41109$$
(MC check 0.41088, `part_a`). Note the scale: the floor of the root readout is
$\sqrt\sigma$, **not** $\sigma$. In recovered-failure-rate units, dividing by
$\sqrt{\theta_{\text{worst}}}$, the floor is
$$\text{apparent bad fraction at the clean end} \;\approx\;
0.411\,\sqrt{\sigma/\theta_{\text{worst}}}\;=\;0.411/\sqrt{\mathrm{SNR}_{\text{worst}}} ,$$
which at the run of record is $0.411\sqrt{0.0033/1.7178} = 0.018$ (`mv_dino_kid`) and
$0.411\sqrt{0.0848/37.73} = 0.019$ (`geom_kid`) — and because $\sigma$ falls like $1/n$
for a degenerate U-statistic under the null, this floor falls only like $n^{-1/2}$.

### Measurement at the paper's $n$

Monte Carlo, $n{=}128$, fixed 399-build reference, 400 draws per rung, arm members drawn
i.i.d. from the empirical pools so that $\mathbb{E}\widehat X$ is known **exactly** (it
is computed in closed form from the pool Gram matrices, and matches the MC mean to
$\le 0.002$ at every rung — see the `E[X]` vs `MC` columns in the script's output).
`geom_kid` uses the real `uniform_random` bad pool; the `mv_dino_kid` row uses a
mean-shift surrogate calibrated so that $\mathrm{MMD}^2(Q,P) = 1.7178$, the run of
record's fully-bad value, since rendering a bad pool was out of scope.

| metric | $\beta$ | $\theta$ | $\sigma$ | SNR | $\sqrt\theta$ | $\mathbb{E}\sqrt{\widehat X^+}$ | bias | delta-method | rel. bias |
|---|---|---|---|---|---|---|---|---|---|
| `mv_dino_kid` | 0.00 | 0.0023 | 0.0032 | 0.73 | 0.0484 | 0.0407 | $-0.0077$ | (invalid) | $-16\%$ |
| | 0.25 | 0.1071 | 0.0041 | 26.4 | 0.3272 | 0.3276 | $+0.0003$ | $-0.0001$ | $+0.1\%$ |
| | 0.50 | 0.4283 | 0.0054 | 80.0 | 0.6544 | 0.6546 | $+0.0002$ | $-0.0000$ | $+0.03\%$ |
| | 1.00 | 1.7198 | 0.0086 | 199.8 | 1.3114 | 1.3115 | $+0.0001$ | $-0.0000$ | $+0.01\%$ |
| `geom_kid` | 0.00 | 0.0240 | 0.1271 | 0.19 | 0.1548 | 0.1612 | $+0.0064$ | (invalid) | $+4\%$ |
| | 0.25 | 2.2644 | 0.3646 | 6.2 | 1.5048 | 1.5014 | $-0.0034$ | $-0.0049$ | $-0.23\%$ |
| | 0.50 | 9.4655 | 0.5915 | 16.0 | 3.0766 | 3.0736 | $-0.0030$ | $-0.0015$ | $-0.10\%$ |
| | 1.00 | 38.7496 | 0.4779 | 81.1 | 6.2249 | 6.2248 | $-0.0002$ | $-0.0001$ | $-0.003\%$ |

The variance formula is confirmed to three digits wherever the SNR is informative: MC
$\operatorname{sd}(\sqrt{\widehat X})$ vs $\sigma/2\sqrt\theta$ is
$0.1200/0.1211$, $0.0960/0.0961$, $0.0790/0.0791$, $0.0384/0.0384$ for `geom_kid` at
$\beta = 0.25, 0.5, 0.75, 1$.

Null behaviour vs $n$ (`part_g`, 600 draws, clean arm against the same reference; the
last column uses the run of record's $\theta_{\text{worst}}$):

| metric | $n$ | $\sigma$ | $P(\widehat X<0)$ | $\mathbb{E}\sqrt{\widehat X^+}$ | apparent bad fraction |
|---|---|---|---|---|---|
| `mv_dino_kid` | 32 / 64 / 128 / 256 | 0.0101 / 0.0056 / 0.0033 / 0.0020 | 0.49 / 0.34 / 0.27 / 0.12 | 0.0443 / 0.0450 / 0.0400 / 0.0419 | **0.034 / 0.034 / 0.031 / 0.032** |
| `geom_kid` | 32 / 64 / 128 / 256 | 0.437 / 0.212 / 0.122 / 0.072 | 0.56 / 0.49 / 0.49 / 0.41 | 0.251 / 0.191 / 0.156 / 0.141 | **0.041 / 0.031 / 0.025 / 0.023** |

### Concrete answer to "does the bias threaten the recommendation?"

**No — but a different property of the same readout does, and the paper should state
that one instead.**

1. The Jensen bias is negligible wherever the readout is informative. At every rung with
   $\mathrm{SNR}\ge6$ the measured bias is under $0.25\%$ of $\sqrt\theta$, i.e. under
   **0.4 percentage points of recovered failure rate** — an order of magnitude below the
   MAE the paper quotes. The closed form $-1/(8\,\mathrm{SNR}^2)$ makes this checkable
   for any future arm: at 5 noise floors the bias is $0.5\%$, at 10 it is $0.1\%$.
2. Near the floor the sign of the error flips and the *clamp* dominates: `geom_kid`'s
   error at $\beta=0$ is $+0.0064$ (upward), not downward. So "biased downward" is the
   wrong one-line summary of the readout; it is biased downward where it is informative
   and biased **upward** at the floor.
3. The real effect at the floor is not bias but a **floor**: a perfectly clean arm reads
   as $3.1\%$ (`mv_dino_kid`) or $2.5\%$ (`geom_kid`) failed at $n{=}128$, and the
   recorded single draws in the run of record agree (recovered fraction at dose 0 is
   0.0458 and 0.0581). Two things set it, and at the paper's $n$ they are comparable:
   the estimator-noise part $0.411\sqrt{\sigma/\theta_{\text{worst}}} \approx 0.018$,
   which shrinks like $n^{-1/2}$, and $\sqrt{\theta_0/\theta_{\text{worst}}}$ — the
   *genuine* discrepancy between the reference split and the arm's clean builds
   ($\theta_0 = 0.0023$ and $0.0240$ measured), which does not shrink with $n$ at all.
   That is why the table above barely improves from $n{=}32$ to $n{=}256$.
4. The readout should therefore be quoted with a floor and never read below it:
   "$\sqrt{\mathrm{KID}}/\sqrt{\mathrm{KID}_{\text{worst}}} = 0.25$, floor $0.03$".
   And the CI should be carried through (item 2a).

Finally, a cross-check on the paper's own MAE numbers: the MAEs it reports are single
draws, so it is fair to ask what they are in expectation. Using $\mathbb{E}\sqrt{\widehat
X^+}$ at each rung, the MAE of the recovered fraction is **0.0066** (`mv_dino_kid`
surrogate), **0.0118** (`mv_dino_kid` with the LEGO bad pool) and **0.0088**
(`geom_kid`). The paper's 0.011 and 0.022 are consistent with that and if anything
slightly pessimistic, so the recommendation is not being flattered by a lucky draw. The
error is concentrated where item 2b says it should be: for the recorded `mv_dino_kid`
row the dose-0 rung alone contributes $0.0458$ of the $5\times0.011 = 0.055$ total
absolute error, i.e. **83% of the MAE comes from the clean end**, and the interior-only
MAE is 0.003 against 0.206 raw (both recorded in the run JSON as
`mae_sqrt_interior` / `mae_raw_interior`).

### VERDICT

**Correct, and under-stated in a way that is easy to fix.** The recommendation survives:
the sqrt bias is $<0.4$ points of recovered failure rate at every informative rung at
the paper's $n$, and the closed form $-1/(8\,\mathrm{SNR}^2)$ bounds it for anyone
else's $n$. But the paper's silence about the clean end is a real omission — the readout
has a floor of $\approx3$ points of apparent failure rate at $n{=}128$, dominated by a
term that does not shrink with $n$, and the honest framing is "linear in the
contaminated fraction above a stated floor", not "linear".

---

## Item 3 — the $0.022$ / $0.119$ / $0.124$ / $0.135$ inconsistency

### Recomputed from the recorded runs

Run of record for this claim:
`outputs/run_20260918_203311_doseresponse_uniform_random-shell_box-canon8/doseresponse.json`
($n{=}128$, `n_ref` 399, bad pools `uniform_random shell_box canon8`, `mix_seed` 0,
`split_key houses_32.s0.70-15-15.v1`, `git_sha 7e24d05`), axis
`mixture_fits.mixture_uniform_random` — the axis §6.5 names ("bad pool is `randomvox` at
$n{=}128$"):

| metric | `exponent_best_fit` | `mae_sqrt` | `mae_raw` | `mae_sqrt_interior` | `mae_raw_interior` |
|---|---|---|---|---|---|
| `mv_dino_kid` | **1.99** | **0.01103** | **0.12371** | 0.00312 | 0.20549 |
| `geom_kid` | 2.16 | **0.02234** | **0.13520** | 0.01787 | 0.22412 |

and the rung values for `mv_dino_kid` are `[0.0036, 0.1100, 0.4380, 0.9696, 1.7178]`, so
$0.1100/1.7178 = 6.40\%$.

**Everything in §6.5 checks out**: 1.99, "0.110 against 1.718", 6.4%, and
$0.011 / 0.022$ vs $0.124 / 0.135$ are all exactly these numbers, correctly attached to
the right metric.

**The intro's `0.119` is not in this run, or in any current run.** Recomputing
`mixture_fit` on every recorded dose run locates it uniquely:
`outputs/run_20260831_224723_doseresponse/doseresponse.json` — $n{=}64$, `git_sha
1939db3`, axis `mixture` — gives `mv_dino_kid` `mae_sqrt = 0.02241`, `mae_raw =
0.11871`, i.e. **"0.022 against 0.119"**. That run's bad pool is `canon8`, which
`scripts/bench_doseresponse.py`'s own module docstring records as retracted:

> "That pool was then disavowed as a quality reference … requantising real builds to a
> coarser grid produces a *smaller* building, not a worse one, so '25% of this arm is
> canon-8' is 25% of a resolution change and not 25% of a failure."

So the intro bullet is quoting a self-consistent pair from a **retracted pool at the
wrong $n$**, and the coincidence that `0.022` also happens to be the run of record's
`geom_kid` value is what has kept the inconsistency alive: the bullet says
$\sqrt{\mathrm{KID}}$ (the render metric), whose current value is **0.011**, while
`0.022` is V-KID's.

Plainly, which printed values are wrong:

- `sec/1_intro.tex:90-91` — "$\sqrt{\mathrm{KID}}$ recovers the failure rate to a mean
  absolute error of $0.022$ against $0.119$": **`0.119` is wrong** (unreproducible from
  the run of record; traceable only to the retracted `canon8` pool at $n{=}64$), and
  **`0.022` is right for the wrong metric** (it is V-KID's, not the render metric's).
- `sec/6_results.tex:165-167` — 0.011, 0.022, 0.124, 0.135: **all correct.**
- Also stale, though not paper text: `blockgen/eval/bench/scorecard.py:with_mmd`'s
  docstring repeats "0.022, against 0.119" and adds "an arm whose output is 25% canon-8
  scores only 31% of the way from real to fully-canon-8 on `mv_dino_kid`". The $n{=}64$
  `canon8` run gives 0.282, not 0.31, so that sentence is not reproducible from any
  recorded run either. Flagged, not mine to edit; a reviewer reading the released code
  will read it.

### Draft replacement for the intro bullet

```latex
follows is not, and reporting $\sqrt{\mathrm{KID}}$ recovers the failure rate to a mean
absolute error of $0.011$ against $0.124$.
```

(or, to match §6.5 without privileging one tier: `... to a mean absolute error of
$0.011$ (\mvkid{}) and $0.022$ (\vkid{}), against $0.124$ and $0.135$.` — but the
bullet's `$\sqrt{\mathrm{KID}}$` reads as the render metric, so the single pair
$0.011/0.124$ is the minimal fix.)

### VERDICT

**Wrong in the intro, correct in §6.5.** One number (`0.119`) must go, and the
surviving `0.022` must either change to `0.011` or be labelled as V-KID's. This is a
retracted-pool number that outlived its retraction, which is exactly the class of error
the paper's own §6.5 provenance discipline exists to catch.

---

## Item 4 — scope: squared IPMs yes, Fréchet no

### The precise condition, and its proof

The identity is not special to kernels. Let $N$ be any functional on signed measures
that is positively homogeneous, $N(c\mu) = \lvert c\rvert N(\mu)$, and define
$\gamma(R,S) := N(R-S)^2$. Then for $R_\alpha = \alpha S + (1-\alpha)Q$ with
$\alpha\in[0,1]$,
$$R_\alpha - S = (1-\alpha)(Q-S)
\;\Longrightarrow\;
\gamma(R_\alpha,S) = (1-\alpha)^2\,\gamma(Q,S). \qquad\blacksquare$$

So the exact condition is: **the statistic is a positively-homogeneous-degree-2
functional of the signed difference $R-S$ alone, with a witness class (kernel) that does
not depend on the samples being compared.** Every IPM qualifies, because
$\mathrm{IPM}_{\mathcal F}(R,S) = \sup_{f\in\mathcal F}\lvert\int f\,d(R-S)\rvert =
N_{\mathcal F}(R-S)$, and $N_{\mathcal F}$ is positively homogeneous (a seminorm when
$\mathcal F$ is symmetric and convex). MMD is the IPM of the RKHS unit ball and KID is
its square with a polynomial kernel, so both are covered; so is the suite's MMD-RBF,
Wasserstein-1, total variation, and anything else in the family. The paper's sentence
"holds for squared integral probability metrics" is therefore right, and the general
form above is the defensible way to say it.

**The sample-independence clause is not pedantry in this codebase.**
`distances.mmd_rbf(x, y, sigma=None)` sets the bandwidth by the median heuristic *on the
concatenation of the two samples*, which makes the kernel a function of $\alpha$ and
breaks the identity (it holds only approximately, and the error is uncontrolled). The
suite's main path freezes $\sigma$ on the reference (`full.py:84`, `ladder.py:89`,
`ladder.py:438`), which restores it exactly; the gated diagnostic variant at
`ladder.py:97` does not. If the scope sentence names MMD, it should name the frozen-
bandwidth condition too, in one clause.

### Why Fréchet distance is outside, with numbers

$\mathrm{FD}$ is a function of the first two moments of each argument *separately*, not
of $R-S$. For $R_\beta = (1-\beta)P + \beta Q$ (writing $\beta = 1-\alpha$ for the bad
fraction, $\Delta = m_Q - m_P$):
$$m_{R_\beta} = m_P + \beta\Delta,\qquad
\Sigma_{R_\beta} = (1-\beta)\Sigma_P + \beta\Sigma_Q + \beta(1-\beta)\,\Delta\Delta^{\!\top},$$
$$\mathrm{FD}(R_\beta,P) = \underbrace{\beta^2\lVert\Delta\rVert^2}_{\text{exactly }\beta^2}
+ \underbrace{\operatorname{Tr}\!\big(\Sigma_{R_\beta} + \Sigma_P
 - 2(\Sigma_{R_\beta}\Sigma_P)^{1/2}\big)}_{\text{a matrix square root of a matrix affine in }\beta}.$$
The mean term obeys the identity exactly; the Bures/covariance term is a matrix-square-
root functional of a matrix affine in $\beta$, which is not a polynomial in $\beta$ and,
crucially for the paper's purpose, **not a fixed power of $\beta$ either** — its
behaviour depends on the two covariances, so no single exponent (and hence no single
root) linearises it.

Measured on the exact population mixture moments, no sampling involved
(`part_e`; the mean part is reported separately so the split is visible):

| feature space / pair | best-fit exponent of FD vs $\beta$ | cov-part$/\beta$ at $\beta=0.02\to0.25$ | cov-part$/\beta^2$ over the same range |
|---|---|---|---|
| DINOv2 768-d, houses vs LEGO builds | **1.65** | 0.099 → 0.158 (nearly flat) | 4.95 → 0.63 |
| DINOv2 768-d, mean-shift pair ($\Sigma_Q=\Sigma_P$) | **1.04** | 0.358 → 0.304 | 17.9 → 1.22 |
| DINOv2 projected to 64-d (full-rank $\Sigma$) | **1.62** | 0.073 → 0.139 | 3.66 → 0.55 |
| geometric descriptor 77-d, houses vs `uniform_random` | **2.07** | 0.118 → 0.343 | 5.91 → 1.37 |

The exponent ranges from 1.04 to 2.07 across four real pairs in the paper's own feature
spaces. Two consequences the paper should draw:

1. Restricting the correction to KID is **right**, and the reason to state is "FD's
   response is not a fixed power of the contaminated fraction (measured 1.0–2.1)", not
   merely "not polynomial".
2. The corollary matters for anyone tempted to generalise: **$\sqrt{\mathrm{FID}}$ is
   not the analogous fix.** FD's covariance term is close to *linear* in $\beta$ near
   the clean end (the cov-part$/\beta$ column is nearly flat where cov-part$/\beta^2$
   diverges), so a square root would over-correct. And $\sqrt{\mathrm{FD}}$ is the
   2-Wasserstein (Bures) distance between the fitted Gaussians, not an unsquared IPM, so
   it is not the same object as $\sqrt{\mathrm{MMD}^2}$ in the first place.

### The citation needs re-pointing (please have a human confirm against the PDF)

I read the arXiv HTML of Bińkowski et al. (arXiv:1801.01401) through a summarising fetch
rather than the PDF, so treat this as a flag to verify, not a settled fact. What that
reading shows:

- **Appendix B.2 is titled "Estimator bias"** and contains **Theorem 3**: *"Let
  $\mathcal P$ be a class of distributions such that $\{(1-\alpha)\mathbb P_0 + \alpha
  \mathbb P_1 : 0\le\alpha\le1\}\subseteq\mathcal P$, where $\mathbb P_0\ne\mathbb P_1$
  are two fixed distributions. Let $\mathcal D$ be an IPM. There does not exist any
  estimator of $\mathcal D$ which is unbiased on $\mathcal P$."* The proof runs through
  the mixture: $R(\alpha) = \mathcal D((1-\alpha)\mathbb P_0+\alpha\mathbb P_1,
  \tfrac12\mathbb P_0+\tfrac12\mathbb P_1) = \lvert\tfrac12-\alpha\rvert\,\mathcal
  D(\mathbb P_0,\mathbb P_1)$, which *"is not a polynomial with any finite degree"*.
- **Appendix D is "FID estimator bias"** and is empirical (D.2, high-dimensional
  censored normals); I found **no** statement there that Fréchet distance is not
  polynomial in $\alpha$.
- **Appendix E, "Comparison of evaluation metrics' resilience to noise"**, is what
  supports the paper's other half-sentence, that they plot metrics against a
  contamination fraction.

So the paper's "they state it in their Appendix B.2" is *nearly* right and *usefully*
wrong. What B.2 states is the **linear** mixture law for the unsquared IPM — the same
mathematical fact, in the form the paper's own recommendation uses — and it states it in
order to prove a negative result. Two edits follow:

1. Attribute the quadratic identity to the linearity of the mean embedding (it needs no
   citation beyond `gretton2012mmd`) and cite B.2 for the linear IPM form, or cite
   Appendix E for the contamination plots.
2. **Cite Theorem 3 for the thing it actually says, because it strengthens the paper.**
   The paper recommends reporting an unsquared IPM, and Theorem 3 says no unbiased
   estimator of an unsquared IPM exists — at any $n$, for any estimator. That is the
   population-level statement of item 2b: the bias of $\sqrt{\widehat{\mathrm{MMD}^2}}$
   is not a small-sample nuisance to be apologised for, it is unavoidable in principle,
   and the correct presentation is therefore "$\sqrt{\mathrm{KID}}$ with a stated
   floor", never "$\sqrt{\mathrm{KID}}$, unbiased". A paper that recommends a readout
   *and* cites the theorem saying it cannot be unbiased is in a much stronger position
   than one that quietly takes a square root.
3. Drop or replace the "which \citet{binkowski2018kid} show is not polynomial in
   $\alpha$" clause, and stand the Fréchet exclusion on the decomposition and the
   measurement above.

### VERDICT

**Substantively correct, attribution imprecise.** The scope claim (squared IPMs yes,
Fréchet no) is true, and the underlying condition — a degree-2 positively homogeneous
functional of $R-S$ with a sample-independent kernel — is provable in one line and
should replace the current informal phrasing, with the frozen-bandwidth clause for
MMD-RBF. The Fréchet exclusion is right but its stated reason is attributed to a
theorem I cannot find in the cited paper; the honest and stronger version is the exact
mean/covariance decomposition plus the measured exponent range 1.0–2.1, and the
observation that $\sqrt{\mathrm{FID}}$ would over-correct. The B.2 citation should be
re-pointed to Theorem 3, whose actual content (no unbiased estimator of an unsquared
IPM) is a better argument for the paper's own caveat than anything the paper currently
says.

---

## Draft LaTeX (not inserted into the paper — for the orchestrator)

### D1. Replacement for the last paragraph of `sec/4_method.tex` (§4.4)

```latex
%TL;DR of paragraph: state the quadratic mixture identity, its scope, and the square-root fix we recommend.
A generator fails on a subset of prompts rather than uniformly, and for a squared-MMD
statistic the response to that mixture is quadratic: writing $\alpha$ for the fraction
of an arm that is drawn from the reference distribution $P$ and $1-\alpha$ for the
contaminated fraction drawn from a failure distribution $Q$,
\begin{equation}
\mathrm{MMD}^2\!\left(\alpha P + (1-\alpha)Q,\; P\right)
  = (1-\alpha)^2\,\mathrm{MMD}^2(Q, P),
\label{eq:mixture}
\end{equation}
a one-line consequence of the linearity of the mean embedding
(Appendix~\ref{app:mixture}). \textbf{This is not new}: the linear form for an
unsquared integral probability metric (IPM) is the engine of
\citeauthor{binkowski2018kid}'s \citeyearpar{binkowski2018kid} Theorem~3, and both they
and \citet{heusel2017fid} plot their metrics against a contamination fraction. Missing
is the recommendation, which we supply: report $\mathrm{MMD} = \sqrt{\mathrm{MMD}^2}$,
which is \emph{linear} in the contaminated fraction and free to compute. Two caveats we
state rather than hide. The identity needs the arm's non-failing component to be
distributed as the reference; for a generator whose good builds are merely close to
real, the response is still exactly quadratic in the failure rate but acquires a linear
term (Appendix~\ref{app:mixture}), which is why our measured exponents bracket $2$
rather than equalling it. And the unbiased estimator of Eq.~\ref{eq:mmd} is legitimately
negative on a real-versus-real draw --- at $n{=}128$ this happens on $27\%$ of clean
draws for \mvkid{} and $49\%$ for \vkid{} --- so we clamp at zero before taking the
root, and the readout therefore carries a floor of about three points of apparent
failure rate at our $n$, below which it must not be read.
The identity holds for any statistic that is a positively homogeneous degree-two
functional of the signed difference $R-P$ with a kernel fixed independently of the
samples: all squared IPMs, KID and MMD-RBF included, the latter only because we freeze
its bandwidth on the reference. It fails for Fr\'echet distance, whose mean term obeys
it exactly but whose covariance term is a matrix square root of a matrix affine in
$\alpha$ --- not a fixed power of the contaminated fraction (we measure exponents from
$1.04$ to $2.07$ across four feature pairs, Appendix~\ref{app:mixture}), so no single
root linearises it and we derive the correction for KID only.
```

### D2. New appendix subsection (`app:mixture`)

```latex
\subsection{The mixture identity, and what the square-root readout costs}
\label{app:mixture}

\paragraph{The identity.} Let $k$ be a fixed positive-definite kernel with RKHS
$\mathcal{H}$, and for a distribution $R$ with $\mathbb{E}_R\sqrt{k(x,x)}<\infty$ let
$\mu_R = \mathbb{E}_{x\sim R}\,k(\cdot,x)\in\mathcal{H}$, so that
$\mathrm{MMD}^2(R,S)=\lVert\mu_R-\mu_S\rVert^2_{\mathcal{H}}$ is Eq.~\ref{eq:mmd}.
Integration is linear in the measure, so $\mu_{\alpha P+(1-\alpha)Q} = \alpha\mu_P +
(1-\alpha)\mu_Q$ and $\mu_{R_\alpha}-\mu_P=(1-\alpha)(\mu_Q-\mu_P)$; squaring the norm
gives Eq.~\ref{eq:mixture}. More generally, any $\gamma(R,S)=N(R-S)^2$ with
$N(c\mu)=|c|N(\mu)$ satisfies it, which covers every squared IPM; it fails as soon as
the kernel is chosen from the samples being compared, as a per-arm median bandwidth
would.

\paragraph{A generator whose good builds are not exactly real.} Write the arm as
$R_\beta=(1-\beta)G+\beta Q$ with $\beta$ the failure rate and $G$ the non-failing
component. Then
\begin{equation}
\mathrm{MMD}^2(R_\beta,P) = \lVert\mu_G-\mu_P\rVert^2
 + 2\beta\,\langle\mu_G-\mu_P,\;\mu_Q-\mu_G\rangle
 + \beta^2\,\lVert\mu_Q-\mu_G\rVert^2,
\label{eq:mixture-general}
\end{equation}
exactly quadratic in $\beta$ for any generator, with a cross term that vanishes only
when $G=P$ and whose sign is unconstrained. Measured on our pools, the cross term is
$1.7\%$ of the quadratic coefficient for \vkid{} against \randomvox{} and pushes the
exponent of the exact response curve to $2.04$; the fits in \S\ref{sec:failrate}
bracket $2$ for this reason and not because the identity is approximate.

\paragraph{Fixed-count mixtures.} Our ladder places exactly $\beta n$ failures in each
arm rather than drawing the arm from the mixture, and for the unbiased estimator this is
not the same thing: in the idealised case $G=P$,
\begin{equation}
\mathbb{E}\big[\widehat{\mathrm{MMD}}^2_u\big]
 = \mathrm{MMD}^2(Q,P)\Big(\beta^2-\frac{\beta(1-\beta)}{n-1}\Big),
\end{equation}
so a stratified ladder reads $2.4\%$ below $\beta^2$ at $\beta=0.25$, $n=128$, and a
power fit to its own prediction returns $2.011$ rather than $2$.

\paragraph{What the root costs.} With $\theta=\mathrm{MMD}^2$ and
$\sigma^2=\operatorname{Var}\widehat{\mathrm{MMD}}^2_u$, the delta method gives
$\mathbb{E}\sqrt{\widehat\theta} \approx \sqrt\theta\,(1-\tfrac18\mathrm{SNR}^{-2})$ and
$\operatorname{sd}\sqrt{\widehat\theta}\approx\sigma/2\sqrt\theta$ with
$\mathrm{SNR}=\theta/\sigma$, so the downward bias is below a quarter of a percent
wherever the metric clears six noise floors; we measure it at $\le0.4$ points of
recovered failure rate at every informative rung at $n{=}128$. At the clean end the
expansion does not apply and the clamp decides the value: for a null draw
$\mathbb{E}\sqrt{\widehat\theta^{+}} = 0.411\sqrt\sigma$, so the recovered-fraction
readout has a floor near $0.411\sqrt{\sigma/\theta_{\text{worst}}}$ from sampling noise
plus $\sqrt{\theta_0/\theta_{\text{worst}}}$ from any real discrepancy between the
reference split and the arm's good builds --- $0.018$ and $0.037$ respectively for
\mvkid{} at our $n$, the second of which does not shrink with $n$. Consistently with
Theorem~3 of \citet{binkowski2018kid}, which rules out any unbiased estimator of an
unsquared IPM, we report the root with this floor rather than as an unbiased quantity.
```

### D3. `sec/6_results.tex` §6.5, optional clause after the MAE sentence

```latex
The error is concentrated at the clean end, where the readout has a floor: the
dose-$0$ rung alone contributes $83\%$ of \mvkid{}'s mean absolute error, and dropping
both anchored endpoints leaves $0.003$ against $0.206$.
```

---

## Provenance of every number above

| number(s) | source |
|---|---|
| 0.011, 0.124, 0.022, 0.135, 1.99, 2.16, rung values, interior MAEs | `outputs/run_20260918_203311_doseresponse_uniform_random-shell_box-canon8/doseresponse.json`, `mixture_fits.mixture_uniform_random` |
| 0.02241 / 0.11871 (the intro's "0.022 against 0.119") | `outputs/run_20260831_224723_doseresponse/doseresponse.json` axis `mixture` ($n{=}64$, `canon8` pool), re-fitted with `bench_doseresponse.mixture_fit` |
| 0.0215 / 0.1185 (same pool, re-run at `git 7e24d05`) | `outputs/run_20260918_202500_doseresponse_canon8_n64_repro/doseresponse.json` |
| noise floors 0.0033 (`mv_dino_kid`), 0.0848 (`geom_kid`) at $n{=}128$ | `context.noise_sd` of the run of record |
| $\theta_0$, $L$, $Q_2$, exact expectations, MC biases, $P(X<0)$, $n$-scaling, FD exponents | `scripts/figs/math_sqrt_check.py` (parts A–G); JSON dump at `<scratchpad>/math_sqrt_results.json` |
| clamp locations | `blockgen/eval/bench/scorecard.py:193`, `scripts/bench_doseresponse.py:183`, `scripts/bench_hollow.py:230` |
| frozen vs per-call RBF bandwidth | `blockgen/eval/bench/distances.py:145-158`, `full.py:84`, `ladder.py:89`, `ladder.py:97`, `ladder.py:438` |
| Bińkowski Theorem 3 / appendix titles | arXiv:1801.01401 HTML via `ar5iv`, read through a summarising fetch — **verify against the PDF before editing the citation** |

Simulation caveat, stated because the brief demands it: the `mv_dino_kid` rows with a
non-zero $\beta$ use a *surrogate* bad pool (LEGO-corpus DINOv2 features, and a
calibrated mean shift), because the real procedural bad pools have no cached renders and
this agent does not render. The null rows, every $\sigma$, every $P(X<0)$ and the whole
`geom_kid` mixture ladder use the real features and the real `uniform_random` pool, and
the exact expectations are closed-form from the pool Gram matrices rather than
estimated, so the bias figures do not depend on the surrogate's realism — only the
particular SNR values do, and those were calibrated to the run of record's own
$\mathrm{MMD}^2(Q,P) = 1.7178$.
