# Fig. 1 assets

These files are for redrawing the teaser by hand in Illustrator. All of them come from
`scripts/figs/teaser.py --assets`, and the finished teaser comes from `--figure`
(`../fig1_teaser.{pdf,svg,png}`). Re-running the script rebuilds everything here from
the recorded runs.

**Build.** The houses_32 test split, row 249 (`paper_figures.TEASER_BUILD`). It is a
3D-Craft build, which is the only corpus whose builds may appear in figures.
- Real: 2,481 blocks, 28 × 15 × 24 (x, y, z).
- Fill-Interior (`filled`): 5,907 blocks, which is +138% for this build. The corpus median is +33%.
- Hollow-Shell (`hollow`): 2,006 blocks (−19%).

**Rendering.** The renders use the vanilla texture pack, the same images the metric is
given, through OSMesa (CPU) with an exact two-pass alpha.
- The background is transparent, and every image in a group is cropped to the same box,
  so real, filled and hollow stay at one scale and can be stacked directly.
- With flat block colours, real and filled are bit-identical in all four metric views
  (max channel difference 0).
- With textures, about 600 of 50k pixels per view differ (max 164/255, mean 0.8/255).
  The cause is the texture-atlas repacking when interior faces disappear, not the fill.
  The appendix on what a render protocol cannot see covers this. If a panel has to be
  pixel-identical, use flat colours (`BLOCKGEN_FLAT_FIGURES=1`).

## Files

`<arm>` is one of `real`, `filled` or `hollow`.

| path | what it shows | size |
|---|---|---|
| `exterior/<arm>_az{045,135,225,315}_el30.png` | the paper's four metric views (orthographic, elevation 30) | ~1200 px |
| `hero/<arm>_az030_el25.png`, `_az060_el25`, `_az045_el35` | isometric hero shots | ~2000 px |
| `cutaway_roofoff/<arm>_az{045,225}_el60.png` | lower half kept, roof off, seen from above at 60° | ~1600 px |
| `cutaway_quadrant/<arm>_az{045,225}_el30.png` | the quadrant facing the camera removed (the paper's quadrant cutaway) | ~1600 px |
| `floor_slabs/<arm>_slab{20,40,60,80}.png` | a one-voxel horizontal slab at 20/40/60/80% of the height (the paper's floor-plan protocol) | ~1400 px |
| `sections/<arm>_vertical_z6.svg` | **vector** vertical section through z=6, the plane with the most rooms. One `<rect>` per block, in two groups: `blocks` (#3f3f3f) and `interior_air` (#f7b48b, sealed air). Each group recolours in one click. | 12 px per block |
| `sections/<arm>_plan_y6.svg` | **vector** floor plan at y=6, seen from above, with the same two groups | 12 px per block |
| `plan_masks/<arm>_footprint_plan_y6.png` | top-down footprint (grey) with that floor's sealed air in peach | 24 px per block |
| `metric_strip.{svg,pdf,png}` | panel c on its own. The SVG text is live `<text>` (Liberation Sans, which is metrically Arial). | 2.4 × 1.75 in |
| `manifest.json` | build index, plane indices and the numbers drawn, as read from the runs | |

Colours follow `scripts/figs/figstyle.py`:
- render tier: orange #eb6834;
- geometry tier: blue #2a78d6;
- observers outside both tiers (Uni3D, GPT-4o): grey #7a7a7a;
- sealed air: peach #f7b48b;
- block: #3f3f3f.

## Numbers in panel c (read at draw time)

| observer | value | source |
|---|---|---|
| MV-DINO-KID, 4 exterior renders | +0.35 noise floors | appearance pillar of `real@solidify`, n=256 leaderboard of record (`run_20261103_090315_…`) |
| Uni3D KID, visible-surface points | −0.1 (seed 0), +0.4 (seed 1) | `uni3d_vis_kid`, `run_20261103_004358_mc_probe3d_s{0,1}`, n=128 |
| V-KID, voxels | +444 | geometry pillar of `real@solidify`, same leaderboard run |
| GPT-4o, exterior pairs | P(prefers real) = 0.53, 95% CI [0.50, 0.56], 128 pairs × 2 orders | `interior_ext`, `run_20261102_214026_vlm_judge`, pair-clustered bootstrap |

The Uni3D row is at n=128 with its own noise floor, so its value is not on the same
scale as the n=256 values; the caption says so. Under all-face sampling the same
encoder reads +21.7 (Appendix 3D-native rows), so panel c shows the observation and not the
encoder.

## Proposed LaTeX block (for whoever is restructuring the sections)

```latex
% sec/teaser.tex (input by main.tex inside \twocolumn[...], so it lands on page 1)
\begin{center}
  \centering
  \includegraphics[width=\textwidth]{figures/fig1_teaser}
  \captionsetup{type=figure}
  \caption{\textbf{A metric ranks only what it observes.}
  \textbf{(a)}~A real held-out house and its \solidify{} copy, in which every sealed
  room is filled with the house's own material ($+138\%$ blocks on this build; median
  $+33\%$ over the test split). Under flat colours their exterior renders are
  bit-identical in all four metric views.
  \textbf{(b)}~With the roof removed, the real house has rooms and the copy is solid.
  \textbf{(c)}~Distance of the filled corpus from real data under each observer, in
  noise floors (the shaded band is $\pm 2$). The multi-view DINOv2 KID on exterior
  renders and Uni3D on visible-surface points ($n{=}128$; seeds $-0.1$ and $+0.4$) stay
  inside the noise, and a GPT-4o judge shown exterior pairs prefers the real house in
  $0.53$ of trials ($95\%$ CI $[0.50, 0.56]$). \vkid{} on voxels places the same corpus
  $444$ noise floors away. The render and voxel values come from the $n{=}256$
  leaderboard (\S\ref{sec:complementary}); published builds come from 3D-Craft
  (\suppsec{app:data}).}
  \label{fig:teaser}
\end{center}
```

This replaces the body of `sec/teaser.tex`. The current caption there describes the old
2×2 teaser, so it no longer matches the image until this block goes in. The old
figure is in `.backup_20261104_pre_n256/figures/fig1_teaser.pdf`. Check
`sec:complementary` against the restructured sections. The figure is 6.875 × 1.95 in, set at `\textwidth`, so every
type size prints at its own value (7 pt minimum).
