# drafts_20261109: splice plan

Drafts for the supplement's 40-generator section and the statistics fixes. No file under
`deliverables/paper/cvpr27/sec/` or `scripts/audit_claims.py` was edited. Line numbers refer to
`sec/X_appendix.tex` as of 2026-11-09 15:30. A consistency pass was editing it at the same time, so
locate each block by its anchor text, not by line number.

Sources: `outputs/run_20261108_205428_stats_fixes` (T115, notes §88),
`outputs/run_20261108_204723_functional_groundtruth` (T114, §87), and the recompute run
`outputs/run_20261109_152704_rankflips_functional_rebase` (T123, §91: `rebase.py`, `results.json`,
`samples.png`, `fig_detection.py`).

## Files

| file | goes to | action |
|---|---|---|
| `rankflips.tex` | `X_appendix.tex`, from `\subsection{The blindness on real generators}` (~1374) through the `\end{figure*}` that closes `\label{fig:rankflips}` (~1464) | **replace**. The `fig:samples` float right after it (~1466-1484) is **not** part of the block. Leave it in place. |
| `stats.tex` block A | end of `\subsection{Reading a squared statistic ...}` (`app:failrate`, ~503-539) | **append** paragraph + `tab:effect` |
| `stats.tex` block B | `\subsection{The head-to-head test}` (`app:significance`, ~474-501), after its 2nd paragraph | **append** paragraph + `fig:detection` (figure*) |
| `stats.tex` block C | `\subsection{The $n{=}256$ leaderboard ...}` (`app:n256`), last 4 sentences, from "The noise floor at $n{=}256$ is estimated from" to "The ranks do not depend on it." (~465-470) | **replace** those sentences with the paragraph |
| `stats.tex` block D | new `\subsection{Kernel and preprocessing}` (`app:kernel`), last in `\section{Robustness of \vkid{}}`, after `\input{sec/X_vkidrobust}` (~1104) and before the `fig:nn` float | **insert**. Optional one-line pointer at the end of `\paragraph{Estimator and kernel.}` in Sec. A (~36-43) |
| `limitation_line.tex` | `sec/6_results.tex`, `\subsection{Limitations}`, end of the first paragraph (after "... so interior appearance is an open measurement.") | **append** one sentence (~30 words) |
| `fig_detection.{pdf,svg,png}` | `figures/fig_detection.pdf` | **move** (stats block B includes `figures/fig_detection`) |
| `audit_claims_additions.py` | `scripts/audit_claims.py` | paste the 3 functions and register them (below) |
| `figures/`, `scratch/` | | delete after splicing. They hold the compile-test wrapper and a copy of the figure. |

`figures/rankflips_samples.png` **still fits**. Its pair (RFT facade vs `m2b_20m_sftmix_s1`) is still
tied under MV-KID (3.1 floors, Holm p = 1) and separated under V-KID (292.9 floors, Holm p = 0.039 at
the exact test's floor). The caption numbers (293, 0.2%, 12.6%, 15.6%) are unchanged. No new paper
figure is needed for the functional subsection. `samples.png` in the rebase run (a non-facade
reversed pair with walk-through labels) can become one if you want it, but it is not in figstyle yet.

## Labels

- Kept, same meaning: `app:rankflips`, `tab:rankflips`, `fig:rankflips`. Grep shows no `\ref`/`\suppsec`
  to them from main or supp except inside the block itself. `fig:samples` (referenced at ~1214) is
  untouched because it stays outside the block.
- New: `app:functional`, `tab:functional` (rankflips.tex); `tab:effect`, `fig:detection`, `app:kernel`,
  `tab:kernel` (stats.tex). None of these labels exists anywhere else in the paper.
- `limitation_line.tex` cites `\suppsec{app:functional}`, which resolves once rankflips.tex is spliced
  and supp is rebuilt (xr `S-` prefix).
- stats.tex cites `app:nscale`, `app:hull`, `app:significance` and rankflips.tex cites `app:significance`.
  All of these exist.

The fragments compile cleanly (no errors, no overfull boxes) in `scratch/test_wrapper.tex`. It
`\input`s `preamble.tex` with stub labels. The only warning is the undefined citation
`kulits2026bricknet`, because the wrapper has no bibliography.

## What changes in the paper's claims

1. **Reversals: 76 of 388 becomes 40 of 308** (exact label permutation, 20k perms, Holm over 780).
   ρ = 0.50 is unchanged. Its interval goes from [0.42, 0.54] to [0.37, 0.58] (half-sampling), and the
   generator / family-cluster bootstraps give [0.20, 0.72] / [0.02, 0.91], which the text now states.
   The n=256 row is 24 of 255, AR 7 of 58, facade 0 of 32, and the 80-d descriptor gives 29 of 308
   (was 46 of 405). The "self" column of the old table came from the subsampling replicates, so it
   is dropped.
2. **Functional ground truth on the 40:** the walk-through sides with MV-KID on **33**, V-KID on **7**
   (17.5%, CP [7, 33]). Every FD difference is significant. Non-facade pairs: **0 of 18**. Facade
   pairs: 7 of 22, and 5 of those 7 are BrickNet (0.8% enterable) over a facade arm. Over the 40
   generators, Spearman with FD is MV-KID 0.67, V-KID 0.16, max-of-two 0.17. The old 60/76 and 16/76
   are superseded.
3. **Design decisions (34, now on the exact test):** agree 6 (was 13), MV-only 14 (12), V-only 1 (3),
   neither 13 (6), opposite 0. Three statements in the old text no longer hold:
   - "the two metrics agree on 13";
   - "3 under V-KID alone";
   - "missed the tokenizer change that most damaged it". The one V-KID-only decision (3D-BPE
     pieces) is one the walk-through *prefers*.
4. **Hollowing generator output:** V-KID's +147.5 floors reads deletion of hidden blocks as damage,
   but the walk-through improves (FD 1.32 to 0.90). The new text no longer calls it damage and
   points to app:functional.
5. **Any main-text sentence implying V-KID orders generators more correctly than MV-KID** (e.g. the
   complementarity paragraph, if it still says so) contradicts T114/T123 and should go. The probe
   claims (Fill/Hollow) are unaffected and are in fact supported: both are functional damage.
6. stats.tex: the cross-metric ratios are effect sizes (Fill 27x, BrickGPT 2.2x, Patch-Stitch 1.1x
   on V/MV) and the stored floor is slightly conservative. The headline 444.36 / 0.35 stay as the
   card's values.

## Claim registry

Register in `build_claims()` after `added_claims_effect_sizes()`:

```python
claims += added_claims_rankflips_exact()
claims += added_claims_functional_rebase()
claims += added_claims_stats_supp()
```

202 claims, and all pass against the current run dirs
(`.venv/bin/python deliverables/paper/cvpr27/drafts_20261109/audit_claims_additions.py`).
The file imports `Claim`, `_load` and `ROOT` from `audit_claims` only for that standalone check. When
pasting, drop the import header and the `__main__` block.

### Obsolete claims to delete

In `added_claims_rankflips()` (reads `run_20261106_193513_render_blind_rank_flips/results.json`):

| claim | old value | replaced by |
|---|---|---|
| "Spearman CI low" | 0.42 | `exact: Spearman half-sampling CI low` 0.37 |
| "Spearman CI high" | 0.54 | `... CI high` 0.58 |
| "self-resample agreement (min of the two medians)" | 0.98 | dropped (column removed) |
| "pairs separated by both metrics (Holm)" | 388 | `exact: pairs separated by both metrics` 308 |
| "significant opposite-order pairs" | 76 | `exact: opposite-order pairs` 40 |

These stay as they are, because their values are unchanged and still printed: "generators pooled" 40,
"Spearman MV-KID vs V-KID" 0.50, and the facade-ceiling ranks 1 and 19. `exact: generators pooled`
duplicates the first, so drop one of the two.

There are no registered claims for T114's 60/76, 16/76 or 1/38, because draft.tex was never spliced.
Nothing to delete there.

Related claims elsewhere in `audit_claims.py` that the effect-size framing makes obsolete *if* the
consistency pass removes the numbers from the text. These belong to the main-text drafts in
`stats_fixes/drafts.tex`, not these files:

- line ~467 "pillar gap on the interior probe" 1270;
- line ~971 "BrickGPT geometric gap over render gap (abstract)" 4.4.

## Not included (deliberately)

- The stats run's rescaled rank intervals (stats_fixes/drafts.tex item 6) are left out. The draft
  quotes Material-Shuffle at "6-7 to 6-8", but the leaderboard prints 5-7. Check which board that
  refers to before splicing it.
- The corrected-floor headline values (491, 293, 230) are left out. The text keeps each run's stored
  floor and states that the correction changes no verdict or ranking.
