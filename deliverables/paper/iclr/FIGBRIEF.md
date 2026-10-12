# Brief for figure and math agents

## Hard rules (a past collision corrupted the LaTeX build; do not repeat it)

1. **Do NOT edit any file you do not own.** In particular: never touch
   `scripts/paper_figures.py`, `deliverables/paper/iclr/sec/*.tex`,
   `main.tex`, `*.bib`, `notes.md`, `results.md`, or another agent's files.
   Integration into the paper is done by the orchestrator, not by you.
2. Your figure generator goes in **`scripts/figs/<your_id>.py`** as a standalone
   script runnable as `.venv/bin/python scripts/figs/<your_id>.py`.
   It writes `outputs/figstage/<your_id>.png` AND `.pdf`.
3. Write your findings to **`deliverables/paper/iclr/math/<your_id>.md`**
   (math agents) or append a `## <your_id>` section to your own
   `outputs/figstage/<your_id>.md` (figure agents). One file, yours alone.
4. Use `.venv/bin/python`. Run your script and LOOK at the PNG you produced
   (Read tool renders images) before declaring it done. An unopened figure is
   an unverified figure.

## What may and may not appear inside a figure

A figure image carries **labels, not arguments**. Permitted: data labels, axis
titles, tick labels, numbers, panel letters, legend keys, colour keys, and group
headers of at most about four words. Not permitted: a sentence of
interpretation, a footer paragraph, an italic subtitle restating the caption, or
any editorial aside. Those belong in the LaTeX caption, or better, in the prose
at the point the figure is cited.

A caption **identifies what is shown** and stays under five typeset lines. It
does not restate the body's argument. Licensing and attribution lines are the
one exception: they are required and stay.

**No repository identifier ever reaches a reader.** `solidify`, `real_test`,
`native_oriented`, `geom_kid`, `patchwork` and their kind are display-mapped to
the paper's names through `scripts/figs/armnames.py`. Note that most of these
words are *also* function names, dict keys and scorecard fields in the same
file: classify every occurrence as IDENTIFIER or LABEL before changing one. A
blanket find-and-replace has broken a script here before.

When you remove text from a figure, **close the gap it leaves** by adjusting the
layout constants near it, so the figure gets shorter rather than emptier. Report
the PDF page size in points before and after: the paper is at its page limit and
a figure that grows is a regression. Removing a wide footer can also shrink the
tight bounding box, which makes LaTeX *enlarge* the figure at an unchanged
`\includegraphics` width; say so if it happens, and give the width that holds
the printed size.

Verify by rasterizing at print size (`pdftoppm -r 150 -scale-to-x <0.9*5.5*150>`)
and looking at it. Nothing may print below 6 pt.

## Figure conventions (match the existing figures)

- Flat per-block colours, never Minecraft texture assets:
  `from blockgen.renderer.textures import load_face_textures` with a fresh empty
  `pack_dir` -- see `flat_textures()` in `scripts/paper_figures.py` for the
  exact idiom. Copy that helper into your own file; do not import from it.
- **Licensing gate, non-negotiable.** Any *rendered build shown to a reader*
  must come from the 3D-Craft corpus. GrabCraft's terms forbid distributing
  blueprints "by taking pictures of the blueprints". Use the `_corpus_mask`
  idiom in `scripts/paper_figures.py` (copy it) to restrict to
  `corpus == "3dcraft"`. Aggregate statistics over the whole corpus are fine;
  pictures are not.
- Vector text, raster images: save PDF with `dpi=300`, fonts as text not paths.
- Colourblind-safe. No red/green as the only distinction. No rainbow ramps.
- Figure width is ICLR single column, 5.5in usable. Assume it will be printed
  at 0.9\linewidth. **Every label must be legible at that size** -- check it.
- Caption text is NOT your job beyond a one-paragraph draft in your .md.

## Honesty rules, which outrank making a nice picture

- Never invent a number. Every number on a figure comes from a scorecard or
  from code you ran, and your .md says which file it came from.
- Noise-floor separations are only comparable **within one run at one n**
  (the noise floor falls ~1/n). Never divide a difference from one run by a
  floor from another. See `scripts/audit_claims.py` `Card.sep()`.
- If the figure you were asked to make turns out to misrepresent the result,
  say so in your .md and propose the honest version instead. That is a
  success, not a failure.

## Context: what the paper claims

Render-and-embed metrics (render a structure from a few views, embed with a
pretrained encoder, take KID/FID) cannot distinguish real buildings from the
same buildings with every sealed interior void filled with the build's own
material (`solidify`, +33% median mass, 94% of builds altered, exterior
silhouette bit-identical). A kernel distance on D4-invariant voxel descriptors
(`geom_kid`, "V-KID") moves 202 noise floors on the same probe. The failure is
in the *view protocol*, not the encoder: the corruption does reach the image
(it alters 86% of views) and every statistic in the family discards it;
cutaway renders recover it with the same encoder and statistic.

Run of record: `outputs/run_20260917_043807_bench_*`. View ablation:
`outputs/run_20260918_205019_view_protocol_ablation`.
