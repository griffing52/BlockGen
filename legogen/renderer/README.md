# legogen/renderer

LEGO / LDraw rendering, kept **separate** from the `blockgen` package (the two PyPI
projects are not merged yet). Renders LDraw models to figure-quality PNGs and dense
dataset grids — the LEGO analogue of `blockgen/renderer/`.

## Why Blender + ImportLDraw (not the pyrender path)

LDraw models are trees of `.dat` primitive parts with per-line poses and an official
colour library. Rendering them correctly means resolving the whole part hierarchy,
materials, and LEGO-specific shading. The **ImportLDraw** Blender addon is the canonical
pipeline for this and produces real, product-quality LEGO renders (studs, tires, slopes,
minifigs, transparent parts). LegoACE's `utils/render.py` only renders *pre-converted*
`.glb` meshes, so it is not usable on raw LDraw here.

Requirements (already installed on this box):
- Blender 4.2.3 → `/home/griffing52/downloads/blender-4.2.3-linux-x64/blender`
- `io_scene_importldraw` addon → `~/.config/blender/4.2/scripts/addons/`
- LDraw parts library → `data/lego/ldraw/ldraw` (24k parts + `LDConfig.ldr` colours)

## Files

- **`blender_render.py`** — runs *inside* Blender. Reads a job JSON and renders each
  listed LDraw file to a transparent PNG tile with a fixed **orthographic isometric**
  camera framed to the model's bounding box (uniform framing across tiles), a key sun +
  soft world fill, Cycles (GPU, denoised) by default. Batches all tiles in one Blender
  session so the OptiX kernel compiles once. Per-model failures are caught and logged.
- **`ldraw_grid.py`** — driver run with normal Python (needs only numpy + pillow).
  Selects models from an OMR-style folder (filtered by part-reference count so tiles are
  legible), launches Blender as a subprocess, then composites the tiles onto a white
  canvas — the `blockgen/renderer/grid.py` aesthetic.

## Usage

```bash
# Dense dataset sheet of small/medium official sets (the one in the paper)
.venv/bin/python legogen/renderer/ldraw_grid.py --rows 8 --cols 12 --band 15 160 \
    --tile-px 256 --render-res 512 --out outputs/figures/lego_data/omr_grid.png

# Showcase sheet of larger, part-diverse sets, rendered bigger
.venv/bin/python legogen/renderer/ldraw_grid.py --rows 4 --cols 6 --band 200 700 \
    --tile-px 384 --render-res 640 --samples 64 --out outputs/figures/lego_data/omr_showcase.png

# Recompose the grid from already-rendered tiles (no Blender)
.venv/bin/python legogen/renderer/ldraw_grid.py --rows 8 --cols 12 --band 15 160 \
    --skip-render --out outputs/figures/lego_data/omr_grid.png
```

Key flags: `--band LO HI` (part-count window), `--random` (vs evenly-spread sampling),
`--engine EEVEE` (faster, slightly lower fidelity), `--azim/--elev/--margin` (camera),
`--colour-scheme {lgeo,ldraw,alt}`, `--look {normal,instructions}`.

Render one model by hand:
```bash
BL=/home/griffing52/downloads/blender-4.2.3-linux-x64/blender
$BL --background --python legogen/renderer/blender_render.py -- <job.json>
```

## Not yet covered

- **StableText2Brick** (`data/lego/stabletext2brick`) is the BrickGPT `hxw (x,y,z)`
  cuboid format, not LDraw. Rendering it needs an `hxw`→LDraw brick-part converter
  (standard bricks only) feeding the same `blender_render.py` pipeline. Deferred: it is
  cuboid-only (the regime our thesis moves beyond), so it is a lower-value figure.
```
