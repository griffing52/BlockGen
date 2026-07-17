# MinecraftACE — the LegoACE port

This page is the operator's manual for the MinecraftACE track: what it is, how every
stage is wired, the exact commands to reproduce T13/T14 (and the conditioned T15/T16
follow-ups), and how to extend it with your own datasets, models, and experiments.

**What it is.** [LegoACE](https://doi.org/10.1145/3757377.3763881) (VAST-AI, SIGGRAPH
Asia 2025) is an autoregressive GPT-2/LLaMA transformer over native per-brick tokens.
`libs/MinecraftACE` is a copy of their released code, minimally modified to train on
our Minecraft data. The bridge between the two codebases is a **converter, not a
rewrite**: `blockgen/export/minecraftace.py` writes our curated caches into LegoACE's
dataset layout, and `scripts/render_minecraftace_samples.py` decodes their samples back
into blockgen `Structure`s for rendering and evaluation. Everything in between (model,
training loop, generation) is LegoACE's own machinery.

!!! warning "Reading the current samples (T13/T14): the pipeline is correct, the models are undertrained"
    The grids in `outputs/figures/minecraftace/` look bad — T13 (`uncond_ckpt1500`) is
    mostly flat grass planes with floating debris; T14 (`uncond_bpe2220`) has real
    massing but incohesive, fragmented blobs. **This is not a linkage bug.** The
    evidence:

    - The `*_real_ref.png` grids are training houses pushed through the **exact same**
      records → `Structure` → renderer path as the samples, and they render as clean
      houses. Export, token layout, decode, and rendering all round-trip correctly
      (the export was additionally verified token-exact at build time — see
      `notes.md` §14).
    - Both runs used a **deliberate 15-epoch pipeline-test budget** (~2,000 optimizer
      steps for a 91.7M GPT-2 from scratch). Val loss was still falling steeply at the
      end (voxel: 3.98 → 0.66; BPE: 4.31 → 1.14 — different vocabs, not comparable to
      each other). For reference, LegoACE's own recipes are 100–500 epochs on 4–8 GPUs
      over 55k models — two to three orders of magnitude more optimization.
    - The failure modes are textbook undertraining, not scrambled tokens: the voxel
      model learned only the **sequence prefix** (records are sorted bottom-to-top, so
      the prefix is literally the grass apron / ground layer); ~78% of emitted
      coordinates are repeats; EOS almost never fires (median sample = 2047 records =
      the hard cap). The BPE model at the same budget already produces walls, towers
      and courtyards because its sequences are ~2× shorter — same budget, more
      semantic progress. That ordering (BPE > voxel at equal budget) is itself the T14
      result.

    Two mechanical details compound the no-EOS symptom and are worth knowing before
    blaming a run: EOS appears once per ~3,000-token training sequence (weak gradient
    signal early), and at inference `top_k=10` means EOS is only *reachable* once its
    logit breaks into the top 10 against the 32 coordinate tokens at a record boundary.
    A converged model does this; a 15-epoch model does not.

## Architecture

- **Model:** `SingleTokenModel` (`libs/MinecraftACE/model/gpt2.py`) — a stock HF
  `GPT2Model` + untied linear LM head. Config `SingleTokenConfig`: 12 layers, 768 wide,
  12 heads → **91.7M params** at our vocab. We train it with `n_positions=8192`
  (learned absolute position embeddings sized at init — sequences can never exceed
  this) and `attn_implementation=sdpa`.
- **Loss:** custom masked cross-entropy in `gpt2.py::forward` — per-token CE multiplied
  by `attention_mask[:, :-1]`. Combined with the dataset convention below this trains
  every real token *including the EOS target* and excludes everything after it
  (padding). Note the `.mean()` divides by all positions including masked ones, so the
  absolute loss value scales with padding fraction — use length bucketing and compare
  val losses only within a run.
- **Generation:** HF `model.generate()` with a **format-grammar logits processor**
  (below), sampling with `temperature/top_k/top_p`. The grammar mask is applied first,
  then the sampling warpers.

## Tokenization: 4-token records

LegoACE serializes a brick as 5 tokens `(x, y, z, rotation, brick_type)`. Minecraft
blocks are axis-aligned unit voxels, so we drop the rotation bank: **4 tokens per
record** `(x, y, z, type)`, and the grammar masking runs on `cur_len % 4`.

Flat shared vocabulary (defined implicitly by `MinecraftTokenDataset` +
`DynamicRangeMaskingProcessor4`):

| Vocab id | Meaning |
|---|---|
| `0` | BOS (also the padding id in collate) |
| `1 .. pos_range` (32) | coordinate values — x, y and z **share one bank**; raw values are 1-based (`c - min + 1`) |
| `pos_range+1 .. pos_range+num_types` | block-type ids (`raw_type + pos_range + 1`) |
| `vocab_size - 1` | EOS |

`vocab_size = pos_range + num_types + 2` (houses_32 voxel: 32 + 422 + 2 = **456**;
houses_32_bpe: 32 + 678 + 2 = **712**).

Records are sorted `np.lexsort((z, x, y))` — **y-primary, bottom-to-top**, then x, then
z — exactly LegoACE's brick ordering, so generation is a bottom-up build. A structure
becomes `[BOS, (x,y,z,type)*N, EOS]`, i.e. `4·n_blocks + 2` tokens.

Two critical conventions (both bit us once; see `notes.md` §14):

- **The "+1" type offset** lives in `dataset/MinecraftTokenDataset.py` (`seq[:,3] +=
  pos_range + 1`), matching `MVNpzDataset`/`textDataset` and the logits processors.
  LegoACE's `SingleTokenDataset` has a no-+1 bug (raw type 0 collides with the max
  coordinate token) — **never train Minecraft data with the stock class**.
- **The `.npy` files on disk store raw records** (1-based coords, 0-based type ids).
  The offset is applied at load time and inverted at decode time
  (`decode_to_records`). Anything that touches token files directly must remember
  this.

### The two exported representations

- **`voxel`** (T13): one record per block. Median house = 745 blocks ≈ 3k tokens.
- **`bpe`** (T14, **adopted default**): one record per learned **3D-BPE piece** — the
  direct analog of LegoACE's multi-cell brick library (their 2×4 brick covers 8 cells
  with one token). Merges are learned with `tokenizers/cluster_bpe.py` on the **train
  split only** and replayed deterministically everywhere. 422 atomic + 256 merges = 678
  piece types; ~2× sequence compression (745 blocks → median 418 pieces), which also
  let +257 more builds fit under the record cap. The record's `(x,y,z)` is the piece's
  **anchor**; decode expands each piece's voxel pattern at its anchor
  (`piece_records_to_structure`). The piece vocab is saved as
  `<name>_piece_vocab.json` and must travel with the checkpoint's samples.

### Format-grammar constrained decoding

`model/logitsprocessor.py::DynamicRangeMaskingProcessor4` masks logits each step so only
syntactically valid tokens are reachable: steps `0,1,2 (mod 4)` → coordinate bank only;
step `3 (mod 4)` → type bank only; EOS allowed **only** at a record boundary; BOS never.
This guarantees well-formed sequences but does **not** prevent semantic errors —
duplicate coordinates, floating components, or missing EOS. (It was verified stepwise
against a real 7,385-step sequence.) LegoACE's stock `infer_uncondition.py` uses *no*
grammar processor plus a retry loop; our `infer_uncondition_minecraft.py` replaces that.

## Stage 1 — Export (`blockgen` → LegoACE layout)

```bash
# voxel records (T13 dataset)
python -m blockgen.export.minecraftace --out data/minecraftace/houses_32

# 3D-BPE piece records (T14 dataset)
python -m blockgen.export.minecraftace --out data/minecraftace/houses_32_bpe \
    --tokenizer bpe --n-merges 256

# later: patch image renders + captions into the JSONs without re-tokenizing
python -m blockgen.export.minecraftace --out data/minecraftace/houses_32 \
    --images-dir outputs/renders/houses_32 --captions outputs/captions/houses_32.json
```

Source data is the curated cross-corpus house cache (`curation/houses.py::
load_house_structures`, 2,661 builds at 32³). Key flags: `--max-dim 32` (sets
`pos_range`), `--max-blocks 2047` (structures with more records are **filtered, not
truncated** — a truncated build followed by EOS would teach "partial = complete"; the
cap keeps `4N+2 ≤ 8192 = n_positions`). Split is deterministic
(`md5(seed:source_path)`, corpus-stratified, 90/5/5 → 2127/114/120 voxel; BPE keeps
2,357 train thanks to compression).

Layout written under `--out`:

| File | Contents |
|---|---|
| `<name>_dat_dict.json` | block/piece token string → raw class id |
| `<name>_pair_dict.json` | block token → `[block_id, block_data]` (voxel mode; exact decode) |
| `<name>_piece_vocab.json` | 3D-BPE patterns + merges (bpe mode) |
| `<name>_rot_dict.json` | `{}` — layout compatibility only |
| `{train,val,test}_dataset.json` | id → `{tokens, images, text, n_blocks, source}` |
| `tokens/<split>/<id>_moved.npy` | `(N, 4)` int32 **raw** records |
| `index.json` | provenance + split for *all* structures (incl. filtered) |
| `export_report.json` | counts, vocab size, `max_seq_len`, filter stats |

Round-trip (`structure_to_records` ↔ `records_to_structure`) is token-exact on
`block_ids`; the data nibble collapses only where the vocab merges variants (same
lossiness as `serialize.py`).

## Stage 2 — Training

All MinecraftACE commands run **from `libs/MinecraftACE`** with `PYTHONPATH=.` and
`LEGOACE_DATA_ROOT` pointing at the export root:

```bash
cd libs/MinecraftACE
export PYTHONPATH=.
export LEGOACE_DATA_ROOT=../../data/minecraftace
```

The T13 run (swap `houses_32` → `houses_32_bpe` and the output dir for T14):

```bash
python train/train_unconditional.py \
    --dataset_name houses_32 \
    --output_dir ../../outputs/minecraftace/uncond-houses32 \
    --pos_range 32 --n_positions 8192 \
    --train_batch_size 1 --gradient_accumulation_steps 16 \
    --num_epochs 15 --learning_rate 1e-4 --lr_scheduler cosine \
    --mixed_precision bf16 --attn_implementation sdpa \
    --length_bucketing --validate_epochs 1 \
    --checkpointing_steps 500 --logger tensorboard \
    > ../../outputs/minecraftace/uncond-houses32.log 2>&1
```

What the flags mean and why they're set this way:

- `--dataset_name` switches the script into Minecraft mode (`MinecraftTokenDataset`,
  4-token collate). Without it you get LegoACE's `SingleTokenDataset` — see the +1
  warning above.
- `--train_batch_size 1 --gradient_accumulation_steps 16`: on the 5070 Ti (15.4 GB),
  batch 2 at 8k context **OOMs**; effective batch is 16. ~2.2 s/step, ~11 GB. The
  15-epoch budget = 1,995 optimizer steps ≈ 1.3 h.
- `--length_bucketing`: shuffled length-sorted batches (`LengthBucketSampler`).
  Lengths vary 8..2047 blocks, so this is a large padding/wall-clock win; it also
  keeps the padding-sensitive loss mean stable within a batch.
- `--attn_implementation sdpa`: no flash-attn wheels for sm_120; SDPA is fine.
- `--validate_epochs 1`: the per-epoch val loop (batch 1 over the 114-build val split)
  was added in the port — stock LegoACE has **no** validation. `val_loss` is the
  checkpoint-selection and overfit signal; it's logged both to the console log and
  tensorboard (`<output_dir>/logs`).
- Checkpoints land at `<output_dir>/checkpoint-<step>/transformer` (HF
  `save_pretrained` format via an accelerate hook) every 500 steps **plus a final
  save**. Resume with `--resume_from_checkpoint latest`.

Multi-GPU: wrap the same command in `accelerate launch --config_file
accelerate_config/{2,4,6,8}-gpu.yaml` (see `train/train_npz_mv.sh` for the pattern).

**Scaling reference (what "actually trained" means):** LegoACE's own scripts run 500
epochs × 4 GPUs × batch 6 (image-conditioned) or 100 epochs × 8 GPUs × batch 8 (text)
on ~55k models. Our T13/T14 val curves were still dropping at epoch 15 — before judging
any architecture change on this track, train to **val-loss plateau** (target: hundreds
of epochs / ≥50k optimizer steps at this data size, then pick the best-val checkpoint).

## Stage 3 — Inference / sampling

```bash
# still inside libs/MinecraftACE with PYTHONPATH and LEGOACE_DATA_ROOT set
python inference/infer_uncondition_minecraft.py \
    --ckpt_dir ../../outputs/minecraftace/uncond-houses32/checkpoint-1500/transformer \
    --dataset_name houses_32 --data_root "$LEGOACE_DATA_ROOT" \
    --save_dir ../../outputs/minecraftace/uncond-samples-ckpt1500 \
    --num_samples 64 --batch_size 8 \
    --temperature 1.0 --top_k 10 --top_p 0.95 --seed 0
```

Mechanics: starts every row from BOS, generates under the `%4` grammar processor with
`eos_token_id = pad_token_id = vocab_size-1`, caps at the model's `n_positions`
(→ 2047 records max), strips BOS, cuts at first EOS, drops any trailing partial record,
un-offsets the type column, and range-checks every record. Output: one raw
`sample-XXXX.npy` per sample, plus copies of `dat_dict.json` (and
`pair_dict.json`/`piece_vocab.json` when present) so the sample dir is
**self-describing** — the render script needs no other context. The console log prints
per-batch record counts; a median pinned at 2047 means EOS isn't firing (see the
warning box). The startup warning "attention mask is not set … pad token is same as eos
token" is benign — prompts are single BOS tokens, nothing is padded.

Sampling knobs worth ablating on a converged model: `--temperature` (LegoACE sweeps
0.6–1.0), `--top_k` (10 is aggressive against a 32-value coordinate bank; try 0 =
disabled with `top_p` only), and checkpoint choice (use the best-val one, not the
last).

## Stage 4 — Rendering & evaluation

Back at the repo root (blockgen environment):

```bash
python scripts/render_minecraftace_samples.py \
    --samples-dir outputs/minecraftace/uncond-samples-ckpt1500 \
    --out outputs/figures/minecraftace --tag uncond_ckpt1500
```

This auto-detects the representation (a `piece_vocab.json` in the samples dir → BPE
expansion, else voxel + `pair_dict.json` for exact `(id,data)` decode), then:

1. decodes every `sample-*.npy` to a `Structure`;
2. renders `<tag>_samples.png` (the samples) and `<tag>_real_ref.png` (real houses
   through the *same* pipeline — the built-in sanity check for the decode/render path)
   via `renderer/grid.py::save_grid` — real Minecraft textures through the headless
   EGL/pyrender renderer (`renderer/textured.py`), matplotlib-voxel fallback if EGL is
   unavailable;
3. writes `<tag>_eval.json`: the shared novelty report (`eval/novelty.py`:
   `mean_nn_iou` vs the houses cache, `duplicate_rate`, `diversity`,
   occupancy-validity, block agreement) plus `connected_rate` / `components_median`
   (`eval/validity.py::n_components`) and block-count stats.

Metric orientation, using T13 vs T14 as the yardstick: voxel@15ep scored
`mean_nn_iou 0.111, connected 0.00, components 27`; BPE@15ep scored `0.199, 0.047, 10`.
The best blockgen-native AR arm sits at `nn_iou ≈ 0.454` (T11) and real builds at
`≈ 0.48` — that is the gap left to close with real training budget.

## Conditioning (T15/T16) — the recipe, re-hosted on blockgen models

Image/text conditioning was **not** run inside `libs/MinecraftACE` (their LLaMA-based
conditioned trainers are present and untouched). Instead the LegoACE conditioning
recipe — frozen encoder → linear projection → prefix tokens, condition dropout,
classifier-free guidance — was reimplemented on our 5.7M `CondVoxelAR2`
(`models/voxel_transformer_cond.py`), which iterates ~20× faster than the 92M port:

```bash
# 1) 4-view renders (white bg, EGL textured renderer; ids match the export index)
python -m blockgen.labeling.render_views --houses 32 --out outputs/renders/houses_32 \
    --px 512 --workers 6
# 2) captions: template-based (templates.py) and/or VLM (vlm_captions.py, gpt-5-mini)
python -m blockgen.labeling.build_captions --index data/minecraftace/houses_32/index.json \
    --out outputs/captions/houses_32.json
# 3) frozen embeddings: DINOv2-base CLS x4 views (image), CLIP ViT-B/32 (text)
python -m blockgen.labeling.embed_conditions --renders outputs/renders/houses_32 \
    --captions outputs/captions/houses_32.json --out outputs/cond/embeddings
# 4) train — T16 regime (the one that works: 16³ canonical, 1600-token sequences)
python -m blockgen.training.train_conditioned --cond image --out outputs/cond/image_run_c16 \
    --repr voxel --canon-dim 16 --max-seq-len 1600 --epochs 60
# 5) sample with CFG + evaluate paired-vs-shuffled condition fidelity
python scripts/sample_conditioned.py --run outputs/cond/image_run_c16 --cfg 3.0
```

T16's headline: at the 16³ regime, image-conditioned palette fidelity is **+112%** over
shuffled conditions with EOS rate 0.8+ and T12-quality samples; at native 32³ (T15) the
same code conditions weakly — regime, not recipe, was the difference. Known deviation
from LegoACE: no sub-assembly-crop augmentation (they re-render partial assemblies).
`MinecraftConditionedDataset.py` exists in the port for eventually running their LLaMA
conditioned trainers on our exports (it consumes the `images`/`text` fields patched in
by the export `--images-dir/--captions` flags).

## Extending the track

- **New dataset:** point the export at any structure list — the only
  houses-specific line in `export()` is the `load_house_structures` call; swap in
  another cache/curation loader, pick `--out data/minecraftace/<name>`, and train with
  `--dataset_name <name>`. Keep `--max-dim` = your `pos_range` and mind the
  `4N+2 ≤ n_positions` rule.
- **New decode-time constraints:** subclass `LogitsProcessor` next to
  `DynamicRangeMaskingProcessor4` and append it to the `LogitsProcessorList` in
  `infer_uncondition_minecraft.py`. Natural first candidates: mask already-emitted
  coordinates (kills the duplicate problem by construction), or port blockgen's
  6-adjacency mask (`training/constrained_decode.py`) to the `%4` grammar — that
  combination is exactly what made validity 1.0 free on the native track (T11).
- **Bigger/smaller models:** `SingleTokenConfig(n_layer=…, n_embd=…, n_head=…)` in
  `train_unconditional.py`; everything else adapts. Vocab is derived from the dataset.
- **Eval anything:** any directory of `sample-*.npy` + `dat_dict.json` (+
  `pair_dict.json`/`piece_vocab.json`) is renderable/evaluable with
  `render_minecraftace_samples.py` — write your own sampler and keep that contract.

## Environment gotchas

- **`transformers < 5` required** (pinned 4.57.6): 5.x drops `add_cross_attention`
  from `PretrainedConfig`, breaking LegoACE's GPT-2 subclass. The port's
  `forward(**kwargs)` already absorbs 4.57's `cache_position`.
- **No flash-attn** on sm_120 (5070 Ti) — use the default `--attn_implementation sdpa`.
- **`PYTHONPATH=.` inside `libs/MinecraftACE`** — imports are top-level (`model.…`,
  `dataset.…`).
- **`LEGOACE_DATA_ROOT`** must contain the export dir named exactly like
  `--dataset_name`.
- **Don't import pyrender/osmesa inside MinecraftACE** — it clashes with our EGL
  renderer context; that's why the conditioned dataset consumes *pre-rendered* PNGs
  and why sample rendering lives on the blockgen side.
- Sequences longer than `n_positions` at generation trigger a device-side assert in
  the position-embedding lookup — the inference script clamps `max_length` for this
  reason; keep that clamp if you write a new sampler.
