# Style brief for the rewrite pass

The paper says the right things and reads like a lab notebook. Fix the reading,
change nothing about the findings.

## The voice we want

This is the target, an abstract written in the style the author asked for:

> Standard multi-view evaluation metrics for 3D generation evaluate models by
> rendering exterior views, leaving them blind to internal structural properties
> like volumetric density, indoor spatial layout, and sub-component
> organization. In domain applications such as interactive media, game engine
> assets, and virtual environments, these internal mechanics are essential for
> playable spaces, collision boundaries, and asset optimization. Through
> controlled structural corruptions on modular architectures, we show that
> render-space metrics fail to distinguish solid interior-filled structures from
> authentic hollow assets, overlooking 33% median block shifts because the outer
> shell remains identical. To resolve this gap, we introduce a metric validation
> protocol that stress-tests 3D evaluation metrics against controlled structural
> degradations.

Read it for rhythm. It states a problem, says why the problem matters, shows
what was done, and says what it means. Sentences are ordinary length and carry
one idea each. Nothing is hedged and nothing is padded.

## Rules

1. **Cut em dashes.** They are everywhere and they are the main reason the prose
   lurches. Replace with a full stop, a comma, a colon, or a restructured
   sentence. A rare one is fine; one per page, not three per paragraph.
2. **One idea per sentence.** Break up sentences that carry a claim, its
   caveat, and its provenance at once. The caveat usually deserves its own
   sentence, or belongs in the appendix.
3. **Write as people who found something.** We ran the experiments and we know
   what they show. Delete self-doubt and throat-clearing. Ban these words and
   their relatives outright: *honest*, *honestly*, *we admit*, *we concede*,
   *arguably*, *it is worth noting*, *we would argue*, *in fairness*, *to be
   clear*, *rather than quietly*. Stating a limitation is fine. Apologising for
   it, or narrating our own process of discovering it, is not.
4. **No process narration.** The reader does not need to know that an audit
   found something, that an earlier draft said something else, that a number was
   corrected, or that we nearly made a mistake. State what is true. The
   appendix may record a correction where reproducibility demands it; the main
   text may not.
5. **Kill repetition.** "Render-and-embed sees only surfaces" appears many
   times. Say it once, well, then use the shorthand.
6. **No sudden shifts.** Each paragraph should follow from the one before it.
   Where a paragraph changes direction, give it a transition. Where two adjacent
   paragraphs say the same thing, merge them.
7. **Keep terminology fixed.** One name per concept. Do not alternate between
   synonyms for variety; that is a virtue in fiction and a defect here.
8. Passive voice only where the actor genuinely does not matter.
9. **Figures carry labels, not arguments.** An image may hold data labels, axis
   titles, ticks, numbers, panel letters, legend keys and group headers of at
   most about four words. A sentence of interpretation goes in the caption, or
   better, in the prose at the point the figure is cited. A caption identifies
   what is shown and stays under five typeset lines; it does not restate the
   body's argument.
10. **No repository identifier ever reaches a reader.** `solidify`, `real_test`,
   `native_oriented` and their kind are replaced by the paper's names in figure
   images as well as in prose. `scripts/figs/armnames.py` holds the map, and
   `main.tex` defines the macro for each.
11. **A case study reads as a case study**: the question it examines, the test
   run, the result, and how the result bears on the main claim. The subject is
   the method, not the authors.

## The target voice, in the author's words

Clear, punchy, narrative-driven minimalism. Sharp and economic: fewer words, no
corporate or academic fluff. Story-driven: every figure, equation and paragraph
moves the plot forward. Humble yet authoritative: groundbreaking results stated
matter-of-factly, without pompous jargon. Symmetrical: text, equations and
figures balanced in clean, predictable rhythm blocks.

## Hard constraints

- **Do not change any number, and do not add one.** Every numeral in the paper
  is registered in `scripts/audit_claims.py` and checked against a recorded run.
  If a sentence reads badly *because* of a number, rewrite the sentence around
  it. If you believe a number is wrong, say so in your report; do not edit it.
- **Keep every `%TL;DR of paragraph:` comment**, and update it when you rewrite
  the paragraph under it so the two still agree. If you merge two paragraphs,
  merge their TL;DRs.
- **Keep every `\label`, `\ref`, `\citep`, `\citet` and macro** (`\solidify{}`,
  `\mvkid{}`, `\vkid{}`, `\blockscore{}` and the rest). Never expand a macro
  into literal text.
- Do not touch `\TODO{}` or `\UNMEASURED{}` markers; they gate the build.
- Do not touch figures, tables, captions, or any file you do not own.
- `\citet` for "Smith et al. show", `\citep` for a parenthetical. No Random
  Capitalisation of ordinary nouns.

## Before you finish

Run `latexmk -pdf -interaction=nonstopmode main.tex` in
`deliverables/paper/iclr` and confirm it exits 0 with no `!` lines. Then run
`.venv/bin/python scripts/audit_claims.py` from the repo root and confirm every
claim still matches; if a claim now fails, you changed a number and must put it
back.

Report: what you changed, the em-dash count before and after, anything you found
that reads as a contradiction between sections, and any number you suspect.
