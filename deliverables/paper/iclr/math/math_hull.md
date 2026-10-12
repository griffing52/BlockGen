# How much of the interior blindness is provable?

`math_hull`, 2026-09-20. Every number below comes from
`outputs/figstage/math_hull_check.json`, written by
`.venv/bin/python scripts/figs/math_hull_check.py` (corpus `houses_32`, split
`houses_32.s0.70-15-15.v1`, seed 0, test split, $n=256$ builds $\times$ 4 views
$=1024$ (build, view) pairs unless stated). Nothing here is quoted from memory
or from another agent's report.

---

## 0. Answer first

**There is a real theorem, and it is not the one the paper currently implies.**

1. **Provable and worth stating.** For an *opaque* renderer, `solidify` is
   **exactly** invisible: not "hard to see", but the rendered pixels are
   identical, for every camera, every encoder, every statistic, and a human
   looking at the same images. Two lemmas give this (surface invariance +
   occlusion), and the code check is **1024/1024 (build, view) pairs
   bit-identical**.
2. **A correction the paper needs.** The deployed render path is *not* opaque
   and, worse, is not even a function of the visible surface. It has exactly two
   leaks, and I separated them by counterfactual:
   * **light through translucent blocks** (glass, leaves) — real interior
     information, present in **37.5%** of views;
   * **a texture-atlas repack** — an artefact of our mesher, present in
     **91.9%** of views, carrying no appearance information about the build.

   So the paper's "`solidify` alters 86% of renders … because interiors show
   through doorways and windows" is right in the count and **wrong in the
   attribution**: doorways cannot be a channel at all (air behind a doorway is
   not sealed, so `solidify` never fills it), and the 86% is dominated by the
   atlas artefact, not by the windows. The informative leak is **~38% of views**,
   not 86%.
3. **Not provable, stated as empirical.** Why DINOv2 + KID fails to resolve the
   visible part. I can measure it precisely (below) but I cannot prove it, and
   the write-up must say so.
4. **Recommendation.** ~0.75 page of appendix (two lemmas + one four-row table)
   plus a two-sentence rewrite of the §6 attribution sentence. Lemma A on its own
   is padding; it earns its place only as the second half of the occlusion
   argument. Details in §7.

---

## 1. Setup and notation (for the appendix)

A **structure** $x$ assigns to each cell $c\in\mathbb Z^3$ either a material
$m\in\mathcal B$ or air; $O(x)\subseteq\mathbb Z^3$ is its occupied set, finite.
A **view set** $V$ is a finite list of cameras. A **render pipeline**
$R_V:\mathcal X\to\mathcal I^{|V|}$ maps a structure to $|V|$ images; it is a
deterministic function (verified: `selfcheck.deterministic = true`). An
**encoder** is any map $\phi:\mathcal I\to\mathbb R^D$, a **pooling** any
$\pi:(\mathbb R^D)^{|V|}\to\mathbb R^{D'}$, and a **statistic** any function $T$
of the two multisets of pooled features. A *render-and-embed metric* is

$$M(\mathcal S,\mathcal S') \;=\; T\big(\{\pi(\phi(R_V(x)))\}_{x\in\mathcal S},\;
\{\pi(\phi(R_V(y)))\}_{y\in\mathcal S'}\big).$$

MV-KID is the case $\phi=$ DINOv2 CLS, $\pi=$ normalised mean, $T=$ unbiased
polynomial-kernel MMD$^2$.

---

## 2. Lemma A — the trivial, load-bearing one

LaTeX, ready to paste (the appendix will need `\usepackage{amsthm}` and
`\newtheorem{lemma}{Lemma}`; if the orchestrator prefers not to add the package,
`\paragraph{Lemma 1 (Render kernel).}\emph{…}` works verbatim with the same text):

```latex
\begin{lemma}[Render kernel]\label{lem:kernel}
Fix a deterministic render pipeline $R_V$. Define $x\sim_V x'$ iff
$R_V(x)=R_V(x')$. Then $\sim_V$ is an equivalence relation, and for every
encoder $\phi$, every pooling $\pi$ and every statistic $T$, the metric
$M$ of Eq.~\eqref{eq:randembed} is constant under replacing any build in
either sample by a $\sim_V$-equivalent build. In particular, if
$x_i\sim_V x_i'$ for $i=1,\dots,n$, then
$M(\{x_i\},\mathcal R)=M(\{x_i'\},\mathcal R)$ exactly, for every reference
set $\mathcal R$.
\end{lemma}

\begin{proof}
$\sim_V$ is the kernel of a function, hence an equivalence relation. The map
$x\mapsto \pi(\phi(R_V(x)))$ factors through the quotient $\mathcal X/\!\sim_V$,
because $R_V(x)=R_V(x')$ implies $\phi(R_V(x))_v=\phi(R_V(x'))_v$ for each view
$v$ and therefore equal pooled vectors. $T$ reads only those multisets, which
are then equal as multisets. \end{proof}
```

Two remarks worth one sentence each in the paper:

* The conclusion is **exact equality of the metric value**, not "no significant
  difference". No sample size, no encoder ablation, and no confidence interval
  can change it; the noise floor is irrelevant because the two numbers are the
  same number.
* It covers everything in the family at once: KID, FID, MMD, precision/recall,
  1-NN two-sample tests, CMMD, a learned critic, *and* a human judging the same
  images. That is the reply to "you picked a bad encoder".

**Code check (`lemma1_demo`).** A synthetic $7^3$ opaque box with a sealed
$3^3$ room (27 cells filled by `solidify`): renders bit-identical, and features
bit-identical for `dinov2b`, `clipB` and `resnet50` (max $|\Delta f| = 0.0$ for
all three). The class is non-empty and contains a `solidify` pair.

---

## 3. Lemmas B1/B2 — when `solidify` lands inside a class

This is where the content is. Write $A_{\mathrm{ext}}(x)$ for the air cells
6-connected to the outside of the bounding box (`probes.exterior_air`), and

$$F(x)=\{(c,d,m)\;:\; c\in O(x),\ c+d\in A_{\mathrm{ext}}(x),\ m=\text{material of }c\},$$

the **exterior face set** — the quads `renderer.textured._exposed_masks` emits
that a ray from outside can reach.

```latex
\begin{lemma}[Surface invariance]\label{lem:surface}
Let $x'=\solidify(x)$. Then $A_{\mathrm{ext}}(x')=A_{\mathrm{ext}}(x)$ and
$F(x')=F(x)$: the corruption changes neither the outside air region nor a single
exterior face, its position or its material.
\end{lemma}

\begin{proof}
$\solidify$ adds exactly the sealed air $S=\overline{O(x)}\setminus A_{\mathrm{ext}}(x)$,
so $O(x')=O(x)\cup S$ and the air of $x'$ is $A_{\mathrm{ext}}(x)$.
(i) No cell of $S$ is $6$-adjacent to $A_{\mathrm{ext}}(x)$: adjacency would put
it in the same connected component of air as the outside, contradicting
$c\notin A_{\mathrm{ext}}(x)$. (ii) Every cell of $A_{\mathrm{ext}}(x)$ is still
connected to the outside through cells of $A_{\mathrm{ext}}(x)$, none of which
were filled, so $A_{\mathrm{ext}}(x')=A_{\mathrm{ext}}(x)$.
(iii) $F(x)\subseteq F(x')$: each such face has an occupied cell of unchanged
material and a neighbour still in $A_{\mathrm{ext}}$. $F(x')\subseteq F(x)$: a
face of $x'$ not in $F(x)$ would need a cell of $S$ with a neighbour in
$A_{\mathrm{ext}}$, excluded by (i). \end{proof}

\begin{lemma}[Exact invisibility under an opaque renderer]\label{lem:occlusion}
Suppose the pipeline (a) colours each pixel from the nearest drawn surface along
its ray, (b) draws exactly the air-bordering faces of occupied cells, all
opaque, and (c) places every camera outside the bounding box. Then
$R_V(\solidify(x))=R_V(x)$ for every view set $V$, and by
Lemma~\ref{lem:kernel} no render-and-embed metric separates the two.
\end{lemma}

\begin{proof}
By Lemma~\ref{lem:surface} the two structures have the same exterior faces; the
only faces of $x$ absent from $x'$ are those bordering sealed air (the faces
$x'$ additionally lacks are interior to the filled region). It suffices that
every such face is occluded in $x$. Let a ray from a camera meet such a face at
$p$, at parameter $t_p$, and let $t^\ast=\inf\{t: \mathrm{ray}(t)\in
\mathrm{int}\,c,\ c\in O(x)\}$. For $t<t^\ast$ the ray lies in air, and that
sub-path is connected to the camera, which is outside the bounding box by (c);
hence it lies in $A_{\mathrm{ext}}(x)$. At $t^\ast$ the ray crosses a face $f$
of an occupied cell whose other side contains points of $A_{\mathrm{ext}}(x)$,
so $f\in F(x)$; by (b) $f$ is drawn and opaque. If $t^\ast<t_p$ the ray is
stopped before $p$ and $p$ is occluded. Equality $t^\ast=t_p$ forces $f$ to be
$p$'s own face, which borders sealed air and therefore is not in $F(x)$, a
contradiction except where the ray meets $p$ only along an edge or corner --- a
measure-zero set of rays. \end{proof}
```

Three things must be said out loud next to this lemma, or a referee will
puncture it:

* **Hypothesis (b) is false for the deployed renderer** (glass and leaves are
  alpha-blended), and the deployed renderer additionally is not a function of
  the drawn surface alone. §4 measures exactly how false.
* The measure-zero caveat is not hand-waving in practice: the check below is
  bit-identity over 1024 renders, so no tie-breaking pathology occurred.
* The same two lemmas apply verbatim to `hollow`, which deletes the cells that
  `solidify` adds. Code check: `surface_invariance` = **0 violations out of 399
  held-out builds for `solidify` and 0 out of 399 for `hollow`** (median 974
  exterior faces per build). Lemma B1's premise is not an idealisation; it holds
  on every build in the corpus.

---

## 4. Part B — the measured decomposition

The deployed pipeline can differ from the opaque ideal in only two ways, and I
isolated each with a counterfactual texture library (geometry, camera, shading
and materials untouched in all four):

| pipeline | what it removes | bit-identical | … given the build has a sealed pocket | changed views: pixels touched (median) | max channel diff (median) |
|---|---|--:|--:|--:|--:|
| `paper` (deployed) | nothing | **8.6%** | 3.5% | **1.67%** | **211**/255 |
| `opaque` (alpha $\to$ 255) | transmission | 8.1% | 3.0% | 0.99% | 78/255 |
| `flat` (2$\times$2 solid tiles) | the atlas repack | **62.5%** | 60.3% | 0.80% | 94/255 |
| `flat_opaque` | both | **100.0%** | **100.0%** | — | — |

$n=1024$ (build, view) pairs; 94.5% of builds have a sealed pocket at all;
median mass increase 34.7%. `flat` is the `flat_textures()` idiom the paper's
own figures use.

**Reading the table.**

* **`flat_opaque` = 100.0% bit-identical** is Lemma B2's code check: under its
  hypotheses the invariance is exact on every one of the 1024 pairs. The theorem
  is not vacuous.
* **The atlas repack.** `build_mesh` numbers atlas tiles in first-appearance
  order over *all* exposed faces — interior ones included — so deleting the
  faces that bordered a sealed pocket renumbers the atlas, shifts every tile's
  UV rect, and at `GL_NEAREST` flips the texel sampled at some pixels of
  surfaces that did not change. This is the dominant channel: with transparency
  disabled, **91.9%** of views still change. The mechanism is confirmed, not
  assumed: in the `opaque` pipeline the render is bit-identical **exactly when**
  the atlas is not repacked (agreement 98.5%; of 1024 pairs exactly 2 go the
  dangerous way — atlas intact yet the render moved — and those two move 1 and 4
  pixels respectively, by 1/255, which is rasteriser tie-breaking at a
  zero-width diagonal chink, the case 6-connectivity calls sealed. In the `flat`
  and `flat_opaque` pipelines that count is 0.)
* **Transmission through translucent blocks** is the only *informative* channel,
  and it is much smaller: `flat` leaves **62.5%** of views bit-identical, 57.4%
  of builds bit-identical in *all four* views. It is well predicted by geometry:
  in `flat`, pairs whose sealed pocket has no translucent block on its boundary
  are **94.0%** bit-identical ($n=548$), against **16.4%** when one is
  ($n=420$). (Not 100%, because glass elsewhere in the line of sight also
  transmits; the predicate is deliberately local.)
* **View budget does not rescue it.** At 8 and 16 views (128 builds) the
  per-pair bit-identical rate is 9.7% and 12.9%, and the fraction of builds
  identical in *every* view is 7.8% at 4, 8 and 16 views alike. Adding cameras
  adds views that change by ~1.7% of pixels each; it does not convert the
  exactly-invisible part into a visible one, because that part is invisible per
  ray, not per camera. (Caveat: the 4-view row is $n=256$ builds, the 8- and
  16-view rows $n=128$; they are not paired.)

### 4.1 Reconciliation with the three numbers in `sec/6_results.tex`

The paper (l. 121–123) states: `solidify` "alters $86\%$ of renders, touching a
median $1.74\%$ of pixels, and in those pixels the median maximum channel
difference is $\mathbf{218/255}$, because interiors show through doorways and
windows."

| claim | my measurement | verdict |
|---|---|---|
| alters 86% of renders | **91.4%** of 1024 pairs (paper's 86.2% was $n=240$) | **confirmed**, same ballpark; use one $n$ or the other, not both |
| median 1.74% of pixels | **1.67%** (median over the 936 changed views) | **confirmed** |
| median max channel diff 218/255 | **211/255** | **confirmed** |
| "because interiors show through doorways and windows" | **contradicted** | see below |

The attribution is wrong twice. (i) **Doorways cannot be a channel.** Air
reachable through a doorway is 6-connected to the outside, so it is not sealed
and `solidify` never fills it — `geometry.interior_volumes` already makes this
distinction, and `probes.solidify` inherits it. (ii) **Windows are a channel but
a minority one**: transmission alone alters 37.5% of views, while the atlas
repack alone alters 91.9%. An honest replacement sentence:

> The change does reach the image, but less than the raw count suggests: with an
> opaque renderer `solidify` is provably invisible (Lemma 3), and of the 91% of
> views our textured pipeline does alter, only ~38% are altered by light
> transmitted through glass; the rest is a texture-atlas artefact of the mesher.
> Where the corruption is genuinely visible it is stark — a median 1.7% of
> pixels at a median maximum channel difference of 211/255 — and every
> render-space statistic we test still discards it.

**Build 249 (the Figure 1 build) is consistent with this, with a caveat.** Under
the *flat* textures the paper's figures use, all four standard views are
bit-identical (mean $|\Delta| = 0.00/255$ on each) — the exactly-invisible
regime, exactly as the cutaway agent reported. Under the *textured* pipeline the
benchmark actually feeds DINOv2, the same four views differ by mean $|\Delta| =
4.29, 3.68, 4.94, 4.42$ /255. Same build, same cameras: the figure is in the
provable regime and the metric's input is not. If the paper shows build 249 and
says "pixel-identical", it should say *in the figure's flat-shaded rendering*,
or a referee who reruns the bench pipeline will get 4.3/255 and call it a
falsehood.

---

## 5. Part C — why the visible part is not separated (empirical, no theorem)

I could not prove anything here and I do not recommend trying. What I can do is
localise the loss, which is more useful than a weak theorem. DINOv2-base,
128 builds $\times$ 4 views, paper pipeline; cosine distances, medians:

| quantity | paper pipeline | `flat` (transmission only) |
|---|--:|--:|
| views altered | 91.2% | 39.1% |
| patch tokens touched (of 256) | 27.3% | 2.5% (mean) |
| token distance, touched tokens | 0.0618 | 0.0802 |
| token distance, untouched tokens | 0.0064 | $1.3\times10^{-8}$ |
| token distance, *different builds* | 0.517 | 0.497 |
| CLS distance, altered views | 0.0128 | 0.0049 |
| CLS distance, *different builds* | 0.277 | 0.238 |
| pooled 4-view distance (what KID reads) | 0.0053 | $6\times10^{-8}$ |
| pooled distance, *different builds* | 0.268 | 0.218 |

What this supports, stated as measurement:

* The change **is** in the features — it is not discarded by the encoder. The
  touched patch tokens move by 0.062, about **12%** of the between-build token
  scale, which is far above float noise.
* The loss happens at **aggregation**, in two stages: CLS over 256 tokens gives
  0.0128 (4.6% of the between-build CLS scale), and mean-pooling over four views
  gives 0.0053 (2.0%). A two-sample kernel statistic on vectors displaced by 2%
  of the inter-sample spread is exactly the regime where $n=128$ has no power —
  which is what the +1.1 noise floors in the view ablation says.
* Under `flat`, the pooled displacement is $6\times10^{-8}$, i.e. numerically
  zero for the majority of builds, because their renders are literally
  unchanged.

What this does **not** support, and must not be written: any claim that DINOv2
is "invariant by training" to interiors. I have no evidence for that and the
patch-token number argues against it.

**A caveat that cuts in the paper's favour.** Part of the deployed pipeline's
residual +1.1 noise floors may come from the atlas artefact rather than from
interior information, since the artefact is a deterministic (hash-like) function
of the interior. If so, the true blindness is slightly *deeper* than measured.
I did not run the ablation that would settle this (re-score MV-KID under the
`flat` pipeline, where the artefact is absent); it is one cheap run if the
orchestrator wants it.

---

## 6. Lemma D — the matching lemma (V-KID's provable blindness)

```latex
\begin{lemma}[Occupancy factorisation]\label{lem:vkid}
The \vkid{} descriptor factors through the occupancy set: $\Phi(x)=\Psi(O(x))$.
Consequently, for any bijection $\sigma$ of the occupied cells' materials
(in particular \matshuffle{}, which permutes the multiset of materials over
$O(x)$, and \monochrome{}, which collapses it), $\Phi(\sigma\cdot x)=\Phi(x)$
and \vkid{} returns exactly the same value. No sample size and no kernel choice
changes this.
\end{lemma}

\begin{proof}
Each block of $\Phi$ --- the $D_4$-canonical $2\times2\times2$ pattern histogram,
the taxicab thickness histogram, the height profile, and the scalar panel
(thickness moments, surface-to-volume, wall fraction, yaw symmetry, sealed and
aperture-aware interior ratios, height entropy, $\log(1+|O(x)|)$) --- is computed
from the boolean array $O(x)$ alone. A material relabelling of occupied cells
leaves $O(x)$ fixed, since no occupied cell is relabelled to air.
\end{proof}
```

**Code check (`D_matching_lemma`, 64 builds, reference = 399 val builds).**
Descriptors bit-identical under both arms (`max_abs_descriptor_diff = 0.0`), and
`geom_kid` returns the *same float* for real, `material_shuffle` and
`monochrome`: `0.02731528486288326` in all three cases. The table in §6 of the
paper prints `+0.00` for V-KID on MATERIAL-SHUFFLE as if it were a measurement;
it is an identity.

**The other half, and it is weaker on purpose.** The renders of the same 64
builds are **never** identical under material shuffle (0/256 views;
median 25.1% of pixels changed, whole-frame mean $|\Delta| = 13.5/255$) or under
monochrome (0/256; 32.6% of pixels, 15.3/255). So MATERIAL-SHUFFLE provably
escapes the render kernel $\sim_V$ — Lemma A does not apply to it — and
empirically MV-KID separates it (+220.83 floors on houses, +140.0 on LEGO, both
from the paper's own tables). The honest phrasing of "complementary blindness"
is therefore **asymmetric**:

* V-KID's blindness to materials is an **identity** (Lemma D).
* MV-KID's blindness to sealed interiors is an **identity only under an opaque
  renderer** (Lemma B2, exact on 100% of pairs there); under the deployed
  renderer it is a very strong empirical null (+1.1 floors at $n=256$, across
  four encoders, four projections and 4/8/16 views).

That is still both halves proved in the sense the brief wanted — one exactly,
one under an explicitly stated hypothesis that the paper's own figure pipeline
satisfies — and it is much harder to attack than "we measured both".

---

## 7. Proved vs. empirical, explicitly

**Proved (and verified in code):**

1. Lemma A: any statistic∘pooling∘encoder∘render is constant on $\sim_V$ classes.
   *Verified:* 3 encoders, exact feature equality on a synthetic pair.
2. Lemma B1: `solidify` (and `hollow`) preserve $A_{\mathrm{ext}}$ and the whole
   exterior face set. *Verified:* 0/399 violations, each direction.
3. Lemma B2: under (a) nearest-surface shading, (b) opaque air-bordering faces,
   (c) cameras outside the bounding box, `solidify` is exactly invisible.
   *Verified:* 1024/1024 pairs bit-identical in the `flat_opaque` pipeline.
4. Lemma D: the V-KID descriptor factors through occupancy, so material
   permutations leave it bit-identical. *Verified:* identical descriptors and an
   identical `geom_kid` float.

**Empirical (no proof, and the write-up must say so):**

5. The sizes of every regime in §4 — 8.6% / 62.5% / 100% bit-identical, 37.5%
   transmission, 91.9% atlas — are measurements on one corpus, one renderer and
   one view protocol.
6. The atlas repack as *the* second channel: established by a counterfactual and
   a 98.5% predicate agreement, not by a proof about OpenGL.
7. Everything in §5. In particular "the information survives to the patch tokens
   and dies in the pooling" is a description of two medians, not a theorem.
8. That MV-KID *does* separate material shuffle (an empirical +220.83).

**Explicitly not claimed:** that no encoder can separate `solidify` under the
deployed textured renderer. That is false — the pixels differ, so some function
of them differs. The theorem is about the opaque pipeline; the deployed one gets
an empirical null with a measured, mostly-artefactual leak.

---

## 8. Where it belongs, and how much space

**Recommended: one appendix subsection of ~0.75 page, plus two sentences in §6.**

* **Appendix** (new `app:invisible`, cross-referenced from the §6 sentence at
  l. 121): Lemmas A, B1, B2 with the proofs above (they are short — B2 is the
  only one over four lines), the four-row table of §4, and a three-sentence
  statement of which hypotheses the deployed renderer violates. Lemma D goes
  next to the V-KID descriptor definition instead, as a two-line remark — it is
  a one-liner and putting it beside the other lemmas overstates it.
* **Main text**: replace the current attribution clause in §6 with the two
  sentences drafted in §4.1 and point at the appendix. Do not put a theorem
  environment in the main text; the result reads as "this is provable, here is
  the pointer", which is the rhetorical payload, and a boxed trivial lemma in a
  6-page body invites the "padded theory section" reaction.
* **Do not** state Lemma A alone anywhere as a headline. On its own it is a
  one-line observation about function composition and a referee will say so. It
  is valuable only because Lemma B1+B2 show the class is non-trivial and
  contains the paper's own probe.
* **Optional figure** (`outputs/figstage/math_hull_leak.png`, already produced,
  3D-Craft build only so the licensing gate is satisfied): real | solidified |
  change mask | atlas-only change mask, showing that the paper-pipeline mask is
  a blob behind the glass *plus* one-texel seams everywhere, and the opaque mask
  is seams only. This is the most convincing single image for the correction in
  §4.1, but it is a defensive figure; include it only if space allows after the
  results figures.

**Is it too thin to state?** No — but it would have been if I had stopped at
Lemma A. The combination that earns the space is: an exactness theorem under a
stated hypothesis, a measurement of how far the deployed pipeline is from that
hypothesis, and a correction to a claim the paper currently makes wrongly. If
the orchestrator wants to cut, cut in this order: the figure, then Lemma A's
proof (keep the statement), then §5's table (keep two numbers in prose).

---

## 9. Things I found that are not mine to fix

Flagged for the orchestrator; I edited no `.tex`, `notes.md` or `results.md`.

1. **`sec/6_results.tex` l. 121–123** — "because interiors show through doorways
   and windows" is wrong (doorway air is never filled) and the 86% is dominated
   by a renderer artefact. Replacement drafted in §4.1.
2. **`outputs/run_20260918_205019_view_protocol_ablation/FINDINGS.md`** — the
   Verdict says "The information is absent from the input, not mishandled by the
   features". That contradicts T36 and my §5: in the deployed pipeline the
   information is *present* in the input (it moves the touched patch tokens by
   12% of the between-build scale) and is lost in aggregation. The sentence
   should read "absent from the input under an opaque renderer; present but
   heavily diluted under ours".
3. **`results.md` T36** — its 13.8% / 86.2% / 1.74% / 218 stand up ($n=240$ vs my
   $n=1024$: 8.6% / 91.4% / 1.67% / 211), but T36's interpretation ("a large,
   spatially concentrated signal is present") attributes the whole 86% to real
   signal. A follow-up entry splitting it 37.5% transmission / 91.9% atlas is
   warranted.
4. **Cheap run worth doing** (nobody owns it yet): re-score MV-KID's `solidify`
   arm under the `flat` texture pipeline. If the +1.1 noise floors fall toward
   0, the paper can say the residual was an artefact of our mesher and the
   blindness is deeper than reported.

---

## 10. Reproduction

```
.venv/bin/python scripts/figs/math_hull_check.py          # ~12 min, 1 GPU
# -> outputs/figstage/math_hull_check.json
# -> outputs/figstage/math_hull_leak.png   (3D-Craft build only)
```

The script pins itself to the real pipeline before measuring anything:
`selfcheck.matches_features_render_views = true` (its renders are bit-identical
to `blockgen/eval/bench/features.py::render_views`) and
`selfcheck.deterministic = true`. Keys: `lemma1_demo` (§2),
`surface_invariance` (§3), `B_{paper,opaque,flat,flat_opaque}_v4`,
`B_paper_v8`, `B_paper_v16` (§4), `C_tokens_{paper,flat}` (§5),
`D_matching_lemma` (§6).
