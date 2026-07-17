# Labeling & captioning — renders, VLM captions, condition embeddings

How every structure in `houses_32` got 4-view renders, 4 text captions, and frozen
image/text embeddings — the inputs behind the conditioned runs (T15/T16) and the
`--images-dir/--captions` fields of the [MinecraftACE export](minecraftace.md). Each
stage is a standalone, **resumable** CLI; this page gives the exact commands as run,
in order.

```
houses_32 cache ──► 1. render_views ──► h#####_view{0..3}.png
                          │
        index.json ──► 2a. vlm_captions (OpenAI batch)──► houses_32_vlm.jsonl
   (from the export)   2b. templates (free, in 3.) ──┐
                          │                          │
                    3. build_captions ◄──────────────┘ ──► houses_32_captions.json
                          │
                    4. embed_conditions ──► houses_32_cond_embeds.npz
                          │
        consumers:  train_conditioned (T15/T16) · export --images-dir/--captions
```

IDs are consistent everywhere: `h{index:05d}` where `index` is the position in the
houses cache — the same ids as `data/minecraftace/houses_32/index.json`. Run the
[export](minecraftace.md#stage-1-export-blockgen-legoace-layout) first if you don't
have that `index.json` yet.

## 1. Render 4 views per structure

```bash
.venv/bin/python -m blockgen.labeling.render_views --houses 32 \
    --out outputs/renders/houses_32 --px 512 --workers 6
```

`labeling/render_views.py` renders each structure from 4 fixed viewpoints (azimuths
45/135/225/315°, elevation 30°, 512px, **white background** — the format both the VLM
and DINOv2 consume) using the textured EGL renderer. Notes from the production sweep
(2,661 structures × 4 = 10,644 tiles):

- `--workers N` forks with one EGL context per process (spawn); 3 workers coexisted
  fine with a training run on the same GPU.
- Resumable: existing files are skipped unless `--overwrite`. `--limit N` for tests.

## 2. Captions

Two sources, merged in step 3. **Templates are free and instant; the VLM pass is the
quality half** (~$2–4 for the full set via the Batch API).

### 2a. VLM captions (what was actually run)

```bash
# spot-check a handful synchronously first
.venv/bin/python -m blockgen.labeling.vlm_captions \
    --renders outputs/renders/houses_32 \
    --index data/minecraftace/houses_32/index.json \
    --out data/minecraft/labels/houses_32_vlm.jsonl \
    --limit 8 --sync

# then the full batch run (2,361 unfiltered structures, ~$2–4, 2026-07-10)
.venv/bin/python -m blockgen.labeling.vlm_captions \
    --renders outputs/renders/houses_32 \
    --index data/minecraftace/houses_32/index.json \
    --out data/minecraft/labels/houses_32_vlm.jsonl
```

Mechanics of `labeling/vlm_captions.py` (Cap3D-style):

- **Model/API:** OpenAI `gpt-5-mini` by default (`--model` to change). Key read from
  the environment or the repo **`.env`** (`OPENAI_API_KEY=…` — the file already
  exists at the repo root, gitignored).
- **Request:** the 4 views (re-encoded PNG→JPEG q85, `detail: low` — keeps batch
  files under the upload limit) + a metadata hint (builder's title, category, flagged
  "may be noisy") + a system prompt asking for **3 diverse player-style captions**
  (structure type, roof/stories/towers, materials; never mention the white background
  or camera). Output is forced through a strict `json_schema` response format, so
  parsing never guesses.
- **Batch API by default** (50% cheaper, 24h window): requests are chunked ≤1,000 per
  job and ≤60MB per upload, submitted with retry/backoff, polled every 60s, results
  appended to the JSONL. `--sync` bypasses batching for small tests.
- **Resumable by design:** the output JSONL (`{"id": …, "captions": […]}` per line)
  is the progress ledger — reruns skip completed ids, so a killed batch run is
  restarted with the identical command. Failed/filtered individual requests are
  logged and skipped, not fatal.
- `--only-missing-titles` restricts to structures with no title metadata (the
  cheap-first strategy: template captions are strongest when a human title exists).

### 2b. Template captions (free fallback / augmentation)

`labeling/templates.py` composes captions from what the corpora already carry —
cleaned builder titles (junk/markup/"minecraft"/size-tag stripping), category,
computed features (dims, size word, dominant **non-scenery** materials via
`STANDARD_VOCAB` — grass/dirt/water etc. are excluded so captions describe the build,
not the ground). No CLI needed; step 3 calls it internally.

## 3. Merge into the final caption file

```bash
.venv/bin/python -m blockgen.labeling.build_captions \
    --index data/minecraftace/houses_32/index.json \
    --vlm data/minecraft/labels/houses_32_vlm.jsonl \
    --out data/minecraft/labels/houses_32_captions.json
```

`build_captions.py` produces `{id: [exactly 4 captions]}` — LegoACE's `text_count=4`
format. Priority: VLM captions first, then templates; case-insensitive dedup; padded
by cycling if fewer than 4 remain. Omit `--vlm` for a template-only file (that
version was built first, 2,361 ids, before the batch came back).

## 4. Precompute frozen condition embeddings

```bash
.venv/bin/python -m blockgen.labeling.embed_conditions \
    --renders outputs/renders/houses_32 \
    --captions data/minecraft/labels/houses_32_captions.json \
    --out data/minecraft/labels/houses_32_cond_embeds.npz
```

LegoACE conditions on frozen encoders; we precompute once so **training never loads
the encoders**:

- images → **DINOv2-base** CLS token per view → `image_embeds [N, 4, 768]`
- captions → **CLIP ViT-B/32** pooled text features → `text_embeds [N, 4, 512]`

Row `i` of the `.npz` == houses-cache index `i` == id `h{i:05d}`. The thin pooled
CLIP vector is a known limitation — text conditioning measurably trails image
conditioning (T15/T16), and caption homogeneity saturates the text-fidelity metric.

## 5. Consumers

- **Conditioned AR (T15/T16):** `blockgen.training.train_conditioned --cond
  {image,text}` reads the `.npz` + caption file, trains `CondVoxelAR2` with 10%
  condition-dropout; `scripts/sample_conditioned.py --cfg 3.0` samples with CFG and
  reports paired-vs-shuffled condition fidelity. Commands and the 16³-regime lesson:
  [MinecraftACE → Conditioning](minecraftace.md#conditioning-t15t16-the-recipe-re-hosted-on-blockgen-models).
- **MinecraftACE dataset JSONs:** re-run the export with `--images-dir
  outputs/renders/houses_32 --captions data/minecraft/labels/houses_32_captions.json`
  to patch `images`/`text` into `{train,val,test}_dataset.json` **without
  re-tokenizing** — the fields `MinecraftConditionedDataset` consumes if you run
  LegoACE's own conditioned trainers.

## Redoing this for a new dataset

The whole chain is dataset-agnostic given (a) a structure cache loader and (b) an
export `index.json` for ids/metadata: render (step 1 with your loader's `--houses`
equivalent — see `render_views.py --houses MAX_DIM`), caption (steps 2–3, swap the
paths), embed (step 4), point the consumers at the new files. Budget rule of thumb
from the houses run: ~$1 per ~700 structures with `gpt-5-mini` batch + low-detail
images; spot-check `--limit 8 --sync` before submitting a batch.
