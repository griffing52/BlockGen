# The training pipeline: pretrain, fine-tune, post-train

Reference for the pipeline as it exists on 2026-10-31. This page started as a
design document on 2026-10-11; it now describes the code in `blockgen/corpus`,
`blockgen/train` and `blockgen/posttrain` as it is, and says for every stage
whether it has run. Section numbers are unchanged from the design document, so
the references in `notes.md` §46 to §48 and in the module docstrings still
resolve. The stack borrows the shape of the open LLM stacks (OLMo 3, nanochat,
TRL) at the scale of one 16 GB GPU and models of 5M to 300M parameters.

Interpreter: every command on this page runs under the conda `python` on PATH
(Python 3.13, torch 2.7, pandas 3.0). `.venv/bin/python` has no pandas and
cannot run the corpus CLIs. Tests for the three packages:

```
PYTHONPATH=. python -m pytest tests/corpus tests/train tests/posttrain -q
```

87 tests; one of them (`tests/train/test_train.py::test_smoke_run_writes_the_run_artifacts`)
is marked `slow` and trains a 60-step model on the GPU.

Companion pages: `workflow.md` (the train, sample, evaluate, view route),
`worldcrop.md` (where the corpus comes from), `benchmark.md` (BlockScore, the
fixed yardstick), `lab.md` (the leaderboard that reads the run registry) and
`research.md` §G (the literature behind the choices here).

---

## 1. Why a pipeline, not another script

Until 2026-10-14 each model family had its own training loop
(`blockgen/training/train_*.py`, `scripts/train_*.py`, `blockgen/experiments_*.py`).
Data was loaded whole into memory from the curated caches. None of those loops
could read the world crops, save optimizer state, resume, or record a git sha
and a tokenizer hash, and evaluation happened by hand afterwards. That was fine
for 2.6k houses and 5M-parameter models. It does not work for a corpus of 649k
builds, a scaling sweep, or a post-training stage that must start from a known
checkpoint and be compared against it.

The LLM analogy the stack follows, with where each stage stands:

| LLM stage | BlockGen equivalent | Status |
|---|---|---|
| Pretraining on web text | Pretrain on world crops, dedup hard, filter soft, tag quality | run at 5M on the `anneal` tier (T58 to T60); `pretrain-all` and `pretrain-clean` are sharded and untrained |
| Mid-training / anneal | Decay phase on curated houses plus high-quality crops | planned; the trainer has no tier switch inside a run |
| SFT | Curated houses from a pretrained checkpoint | run (T60); unconditioned, no caption or image channel |
| RLVR / rejection sampling | Non-pillar rewards, best-of-N, rejection-sampling fine-tuning | code landed 2026-10-31 (`blockgen/posttrain`), no recorded run |
| Preference tuning | DPO or SimPO on lab 2AFC pairs | pairs being collected (311 `better_build` compares in `outputs/lab/lab.db`); no training code |
| Distillation | large teacher into the deployable 20M model | planned |

---

## 2. Layout

```
blockgen/
  corpus/      canon, manifest, dedup, kinds, templates, tiers,
               tokenizer, shards, facade, facade_baselines, reps
  train/       config (TrainConfig, PRESETS), archs (the Arch registry),
               data, trainer, sample, registry, __main__
  posttrain/   rewards, bon, rft, select
  eval/bench/  BlockScore, frozen; eval/facade_stats.py holds the face
               statistics the rewards read
  models/      voxel_transformer_ar2.py, the one architecture registered
outputs/
  run_<stamp>_<name>/   run.json, metrics.jsonl, last.pt, best.pt,
                        ckpt_<step>.pt, samples.png, facades.png (facade runs),
                        samples_step<N>_<dim>.npz + _manifest.json,
                        bench/scorecard.json, bench.log, summary.json
  registry.jsonl        one line per finished run, appended by the trainer
configs/
  corpus/*.yaml         tier definitions (configs/corpus/README.md)
  train/*.yaml          run configs, flat TrainConfig keys (configs/README.md)
data/minecraft/corpus/
  manifest/             per-source Parquet tables + float16 gate features
  dedup.parquet         cluster, is_rep, cluster_size, cluster_worlds
  kinds.parquet         kind, mine_share
  tiers/<name>.{json,ids.txt}
  shards/<tier>/<tokenizer hash>[-d4]/
```

Three facts about the layout that differ from the design document. Size
presets are the `PRESETS` dict in `blockgen/train/config.py`, not files under
`configs/models/`. Checkpoints sit at the run root, not under `ckpt/`. The
config is one flat dataclass, `TrainConfig`, loaded through the existing
`blockgen/config.py` (`extends:` plus overrides); an unknown YAML key raises
`ValueError` in `blockgen.train.config.load`, an unknown `key=value` on the
command line is a usage error.

---

## 3. Corpus layer

### 3.1 Manifest

One table over every build we own, curated or cropped, built once per source
and stored as Parquet under `data/minecraft/corpus/manifest/`
(`blockgen/corpus/manifest.py`). The columns are `COLUMNS` in that module:

`id, source, path, month, world_id, title, author, url, dx, dy, dz, n_blocks,
fill, n_ids, p_built, cut, oversize, route, canon_hash, sig`

`sig` is the 128 x uint32 MinHash signature as bytes; the 295 gate features
sit in a row-aligned `*_feats.npy` beside each table. `source` takes the values
`worldcrop`, `grabcraft`, `3dcraft`, `text2mc`, `text2mc_schem` and `templates`.
Ids are `cache:<corpus>:<i>` for the curated caches, `tpl:<family>/<name>` for
templates, and `wc:<month>_<world>_<i>` (the crop file stem) for world crops;
the crop path is the `path` column. Columns the design document
listed and the table does not have: `bbox` (it is `dx, dy, dz`), `dedup_cluster`
and `split_group` (the dedup columns live in `dedup.parquet`, the split in the
pinned split JSON).

The world-crop build is resumable per month and skips the month the ingest is
writing. 47 of the 88 ingested months have a table today; the rest (2009 to
2012, parts of 2013, 2018) have `crops.jsonl` and no manifest rows. Adding them
changes every tier hash, so that refresh is a decision (section 10), not a
chore.

### 3.2 Deduplication

Worlds are re-uploaded across months and builds are copied within worlds. The
LLM literature is consistent that dedup matters more than aggressive quality
filtering, and our own 31% GrabCraft memorisation says novelty must be measured
after dedup, not before.

- Exact: `canon_hash` of the crop after translation and D4 normalisation
  (`blockgen/corpus/canon.py`).
- Near: MinHash over shingles of (block id, relative offset) triples, LSH
  banding, clusters at Jaccard 0.8 (`blockgen/corpus/dedup.py`). The
  representative of a cluster is its highest `p_built` row.
- Recurrence: `cluster_worlds`, the number of distinct worlds a cluster occurs
  in. The 24 most widespread clusters (430 to 4,870 worlds each) are vanilla
  village pieces the detector's structure mask misses, so `cluster_worlds >= 5`
  is the generated-content filter (notes.md §46).
- Composition: mineshafts are randomised, so recurrence misses them. A crop
  whose member voxels are at least 90% planks, fences, rails, cobwebs, torches
  and chests, with a rail or cobweb present, is a mineshaft; 50 to 90% with the
  tell is `mixed`, kept and tagged. `blockgen/corpus/kinds.py` writes `kind` in
  {build, mixed, village, mineshaft} as a sidecar the tiers merge like
  `dedup.parquet`. Over the 927,427 rows of the current manifest (`python -m
  blockgen.corpus.manifest summary`, 2026-10-31): build 77.2%, village 12.6%,
  mineshaft 9.2%, mixed 1.1% (notes.md §46.1 reports the pre-templates count).
- Holdout exclusion, not a new split. The pinned `houses_32` split
  (`data/minecraft/splits/houses_32.s0.70-15-15.v1.json`) stays the split. A
  tier that lists `exclude: [eval-holdout]` drops every row whose `world_id` or
  dedup `cluster` appears in the holdout (`blockgen/corpus/tiers.py`,
  `materialise`), which closes the curated-in-worldcrop leak without changing
  the test set.

### 3.3 Tiers

A tier is a query over the manifest plus a hash of the resulting id list. The
query lives in `configs/corpus/<tier>.yaml` (schema in `configs/corpus/README.md`);
`python -m blockgen.corpus.tiers build` writes
`data/minecraft/corpus/tiers/<tier>.{json,ids.txt}`. The sha256 of the sorted
ids is the tier hash recorded in every checkpoint trained on it
(`trainer.tier_hashes`). Changing the filter changes the hash. A tier pinned to
a split (`holdout_split`) that resolves to zero rows raises `EmptyTierError`
instead of writing an empty tier; the message says whether the cache is missing
from the manifest or the tier's own filters removed every row.

The seven tiers on disk:

| Tier | Filter (short) | Builds | sha256 (12) | Role and status |
|---|---|---|---|---|
| `pretrain-all` | `n_blocks >= 20`, dedup, exclude holdout | 648,845 | `5cc35e5296fd` | broad corpus, terrain included; sharded, never trained on |
| `pretrain-clean` | plus `p_built >= 0.5`, not `cut`, `kind` in build/mixed | 548,732 | `ef6815d600fb` | soft-filtering ablation; sharded, never trained on |
| `anneal` | (non-worldcrop, `n_blocks >= 80`) or (`p_built >= 0.9`, `n_blocks >= 500`, not `cut`); `kind == build` | 275,200 | `49ca324a73cc` | the pretraining tier of every run so far (T58 to T60) |
| `sft-houses32` | train part of the pinned split | 1,863 | `cd526aaaeeb6` | SFT target and scratch baseline |
| `eval-holdout` | val + test parts of the pinned split | 798 | `0f23acfed16f` | validation NLL; never trained on; excluded from every other tier |
| `vanilla` | `kind` in village/mineshaft, one per cluster | 61,130 | `6df404ad2a2e` | gate negatives, `src:vanilla` source; no shard |
| `templates` | `source == templates`, `n_blocks >= 20` | 1,007 | `6a32c0e8d5d7` | 1.21.1 jigsaw pieces; internal only (Mojang assets); sharded, unused alone |

The gate rejects stay in `pretrain-all` on purpose. Terrain is a shape class the
model should know, and the quality tag lets us sample away from it.

### 3.4 Tags, not deletions

Every training sequence starts with four tag tokens, in this order
(`TAG_GROUPS` in `blockgen/corpus/tokenizer.py`):

| Group | Values | Set by |
|---|---|---|
| `src` | `archive`, `curated`, `vanilla`, `templates` | `shards._src_tag`: worldcrop is `archive`, or `vanilla` when its kind is village or mineshaft; templates is `templates`; every other source is `curated` |
| `q` | `0` to `4` | `int(p_built * 5)` capped at 4; rows without a gate score (the curated caches) get `4` |
| `size` | `16`, `24`, `32`, `48` | the sequence's edge: the tight box of a whole build, or the window edge |
| `win` | `whole`, `window` | whether the sequence is a whole build or a window of a bigger one |

The `facades` kind adds two groups, `face` and `cell`, after these four. There
is no `cat` tag and no tag dropping: the trainer feeds the tags on every
sequence of every stage, and sampling conditions on
`TrainConfig.sample_tags`, by default `src:curated q:4 size:32 win:whole`. The
`src` group has no spare slot, so a new source maps to `curated` until the
tokenizer is versioned again.

### 3.5 Commands

Built and tested: `blockgen/corpus/{canon,manifest,dedup,kinds,templates,tiers,tokenizer,shards,facade,facade_baselines,reps}.py`
and `tests/corpus/`. The full order, raw sources to shards:

```
python -m blockgen.corpus.manifest build --workers 6        # world-crop months, resumable, skips the in-progress month
python -m blockgen.corpus.manifest curated --caches all_32 houses_32 houses_48   # the default list; houses_32 is what sft-houses32 and eval-holdout index
python -m blockgen.corpus.manifest summary
python -m blockgen.corpus.templates                          # jar structure templates -> data/minecraft/templates + manifest/templates.parquet
python -m blockgen.corpus.dedup                              # exact + MinHash/LSH clusters -> corpus/dedup.parquet
python -m blockgen.corpus.kinds                              # build / mixed / village / mineshaft -> corpus/kinds.parquet (after dedup); takes no arguments, any invocation (even --help) rewrites the file
python -m blockgen.corpus.tiers build [names ...]            # every configs/corpus/*.yaml, or the named ones -> corpus/tiers/<name>.{json,ids.txt}
python -m blockgen.corpus.tiers show <tier>
python -m blockgen.corpus.tokenizer learn-pieces [--tier sft-houses32] [--n-merges 256] [--max-corpus 400] [--seed 0]   # -> corpus/tokenizers/<hash>.json
python -m blockgen.corpus.shards build <tier> [--max-blocks 2048] [--max-windows 32] [--workers 6] [--oriented] [--limit N]
                                              [--kind voxels|facades] [--faces all|sides|sides_top] [--layout per_face|joint] [--augment d4]
                                              [--tokenizer <hash>|<tokenizer.json>]   # required for pieces
python -m blockgen.corpus.shards show <tier> [--hash H] [--kind voxels|facades|pieces]
python -m blockgen.corpus.shards from-npz <npz> --tier <name> [--kind voxels|facades] [--layout per_face|joint] [--faces all|sides|sides_top]
                                              [--tokenizer <hash>|<tokenizer.json>] [--tags k=v ...] [--source NAME] [--max-blocks N] [--augment d4]
```

`scripts/corpus_refresh.sh [--dry-run] [--workers N]` runs that order end to
end and rebuilds every shard variant on disk today (the houses tiers at
`--max-blocks 8192`, the pretraining tiers at the default 2048, the per-face,
joint and joint-d4 facade shards). It stops at the tier stage if a split-backed
tier comes up empty. It has been run with `--dry-run` only.

The manifest build gate-scores any crop with no entry in its month's
`gate_scores.jsonl`, so new months need no separate scoring pass.

### 3.6 Windows and shards

A shard is built once per (tier, tokenizer) and memory-mapped by every run that
trains on it (`blockgen/corpus/shards.py`). The window rule, fixed at build
time: a build whose tight box fits 48³ and has at most `--max-blocks` blocks is
one sequence tagged `win:whole` with `size:32` or `size:48`. Anything larger
yields `ceil(n_blocks / max_blocks)` windows, capped at `--max-windows` (32),
each placed around a random occupied voxel with a jitter of up to a quarter
edge (`EDGES`, `_windows`); a window with more than `max_blocks` blocks is
retried at edge 32, 24 and 16 and dropped if still too dense.
Sequences shorter than `MIN_BLOCKS` (16) are dropped. Window choice is seeded
by the build id, so a shard is reproducible and the windows are the same every
epoch. The design document said windows would be resampled per epoch; they are
not.

Layout of `data/minecraft/corpus/shards/<tier>/<tokenizer hash>[-<augment>]/`:

```
tokens.u16      every sequence back to back, uint16
offsets.npy     int64 [n+1]: sequence i is tokens[offsets[i]:offsets[i+1]]
meta.parquet    seq, id, source, src, q, size, win, ox, oy, oz, n_blocks, n_tokens (+ face, u, v, extent, n_cells, aug)
tokenizer.json  the tokenizer artifact
spec.json       tier name + sha, tokenizer hash, kind, layout, faces, augment, parameters, counts
```

The shards on disk:

| Tier | Tokenizer | Kind / layout / augment | `max_blocks` | Sequences | Tokens |
|---|---|---|---|---|---|
| `pretrain-all` | `60684456e598` | voxels | 2048 | 1,792,538 | 8.00B |
| `pretrain-clean` | `60684456e598` | voxels | 2048 | 1,542,508 | 6.93B |
| `anneal` | `60684456e598` | voxels | 2048 | 1,241,999 | 6.57B |
| `templates` | `60684456e598` | voxels | 2048 | 1,021 | 1.37M |
| `sft-houses32` | `60684456e598` | voxels | 8192 | 1,863 | 8.22M |
| `eval-holdout` | `60684456e598` | voxels | 8192 | 798 | 2.92M |
| `sft-houses32` | `a9546c1f2f94` | facades, per_face | 2048 | 12,684 | 5.16M |
| `eval-holdout` | `a9546c1f2f94` | facades, per_face | 2048 | 5,172 | 2.00M |
| `sft-houses32` | `a4ca45ac1fe5` | facades, joint | 2048 | 2,114 | 5.09M |
| `sft-houses32` | `a4ca45ac1fe5-d4` | facades, joint, d4 | 2048 | 16,880 | 40.7M |
| `eval-holdout` | `a4ca45ac1fe5` | facades, joint | 2048 | 862 | 1.97M |
| `sft-houses32` | `cdce56584ebf` | pieces | 8192 | 1,863 | 4.22M |
| `eval-holdout` | `cdce56584ebf` | pieces | 8192 | 798 | 1.59M |
| `anneal-worldcrop` | `cdce56584ebf` | pieces | 2048 | 1,235,960 | 3.86B |

`Shard.__init__` compares `spec.tier_sha256` with the tier json's current
`sha256` and warns `StaleShardWarning` when the tier was rebuilt since; the
shard still opens. `find(tier, tok_hash, kind, augment)` resolves a shard by
hash, else the newest of its kind by mtime, which is why every facade config
pins `tokenizer:`.

Derived tiers. `shards.from_structures(structures, tier, tok, *, tags, source,
max_blocks, faces, augment)` writes the same five files for a list of
`Structure` objects that never passed through the manifest, which is how model
samples become training data (section 7). Each structure is encoded as a whole
manifest build of the same size would be, the tier json carries `kind: derived`
and the sha256 of the kept structures' content hashes, and a derived tier name
may not collide with a `configs/corpus/*.yaml` name. When every structure is
skipped nothing is written and `ValueError` names the skip counts. The
`from-npz` subcommand is the same function from the command line.

### 3.7 Facades

`blockgen/corpus/facade.py` gives every build a second, 2.5D reading: six face
images, one per axis and end, each storing per column the depth to the first
block and that block's id and data. The four sides are the facades, `+y` is the
roof plan. `lift` puts any set of faces back into a canvas, so a model that
emits facades produces builds the benchmark scores like any other; `shell`
(project then lift) is the visible shell of a real build and the ceiling of such
a model. `scripts/facade_oracle.py` writes that ceiling as three control rungs,
`real@facades4`, `real@facades5` and `real@facades6` (sides, plus roof plan, plus
underside), for the bench (results.md T65). `blockgen/corpus/reps.py` registers
`facades` beside `voxels` with `encode`, `decode` and `picture`, which is how the
lab shows a build's faces. `blockgen/corpus/facade_baselines.py` holds the
non-learned facade generators (`facade_unigram`, `facade_markov`,
`facade_patchwork`) the facade arms are measured against (T66).

The tokenizer's `facades` kind (`CorpusTokenizer.default(kind="facades")`)
writes one sequence per face, `[BOS, src, q, size, win, face:<f>, U, V, cell*,
EOS]`, with a cell either `cell:empty` or `(DEPTH, BLOCK)`. Its `joint` layout
(`layout="joint"`) puts every face of a build in one sequence, sides in walking
order, then roof plan and underside, so each face is generated in sight of the
ones before it. The two layouts have different hashes and different shards. A
run with `kind: facades` samples every face of a real bounding box drawn from
the first training shard's `meta.parquet` and lifts them
(`blockgen/train/sample.py`, `facade_builds`); the `--faces` set the shard was
built with travels in `spec.json` and is what the sampler emits.

### 3.8 Pieces (3D-BPE)

The `pieces` kind is arm B on tiers. `python -m blockgen.corpus.tokenizer
learn-pieces` runs the legacy `learn_clusters` (`blockgen/tokenizers/cluster_bpe.py`)
over a seeded sample of `--max-corpus` builds of a training tier (never a holdout
tier), on the voxel kind's fixed unoriented block classes: the forced
bed/door/double-plant merges, then `--n-merges` frequency merges. The merges are
stored in the artifact, so the hash names the piece vocabulary; the artifact is
written to `data/minecraft/corpus/tokenizers/<hash>.json` and every shard built
with it carries a copy. A sequence is `[BOS, src, q, size, win, (X, Y, Z,
PIECE)*, EOS]` with `X, Y, Z` the piece's anchor (min corner of its occupied
box), anchors in (y, z, x) order. A one-block piece is the voxel kind's block
token; learned pieces take ids after the tags, so coordinates, blocks and tags
keep their voxel ids. Encoding replays the merges on the build's blocks, so a
build tokenises the same whether or not it was in the fit sample; decoding
places each piece's pattern at its anchor (`cluster_tokens_to_structure`) and
equals the voxel kind's decode of the same build (`tests/corpus/test_pieces.py`).
Windows, `max_blocks` and `MIN_BLOCKS` are the voxel kind's; only the body
changes. Training and sampling need `kind=pieces tokenizer=<hash>`; the sampler
is the voxel one (`tokenizer.decode` dispatches on the kind), with no grammar
mask, like voxels.

The tokenizer on disk is `cdce56584ebf` (`sft-houses32`, 256 merges, 400
builds, seed 0): 48 forced and 256 learned merges, 304 pieces beyond the 441
block classes, 811 tokens. Median sequence length is 1,770 on `sft-houses32`
(voxels 3,226) and 1,490 on `eval-holdout` (voxels 2,652). Holdout NLL is per
token, so it does not compare across kinds; compare runs by BlockScore.

### 3.9 D4 augmentation

`--augment d4` writes a second shard beside the plain one (`<hash>-d4`) holding
the eight horizontal symmetries of every build (four rotations about y, each
with and without a mirror; `blockgen/utils/augment` remaps stair and log
facings), duplicates of a symmetric build collapsed and each sequence tagged
`aug` 0 to 7 in `meta.parquet`. `TrainConfig.augment: d4` trains on it;
validation tiers are always opened plain (`trainer.train` passes no augment to
`find` for `cfg.val`). The only d4 shard on disk is the joint facade shard of
`sft-houses32`, 16,880 sequences from 1,863 houses; at 20M it took holdout NLL
from 0.806 to 0.479 (T68, notes.md §59). The voxel kind accepts `--augment d4`
and has not been built or trained that way.

### 3.10 Category conditioning

The class-conditioned arm (scoping in `deliverables/reports/category-conditioning.md`)
adds one prompt word per build. Three pieces:

* **Labels** (`blockgen/corpus/labels.py`, `python -m blockgen.corpus.labels build`)
  write `corpus/labels.parquet` (`id, cat, cat_orig, cat_src, cat_raw, build_key,
  cat_split`) and `labels.json` (counts). A curated build is classified from its
  title, then its GrabCraft subcategory, its GrabCraft tags, its VLM category, and
  finally its cache (a houses cache row is a house), through one ordered keyword
  list; a build at most 2 blocks thick is `pixelart`. The eleven classes with at
  least 120 curated train builds are house, castle, tower, shop, farm, ship,
  vehicle, aircraft, statue, tree and pixelart; church folds into `none` for now.
  The non-house classes come from the GrabCraft 48 cache (`gc_small_48`), added as
  an auxiliary manifest (`manifest curated --aux`, read only by tiers that name it
  in `aux:`), so no existing tier changes when it is rebuilt.
* **Tiers** `sft-cat` (5,048 builds, `cat_split == 'train'`), `eval-cat` (608, 32 to
  128 per class, house drawn from `eval-holdout`, nothing any existing run trained
  on) and `sft-cat-wc` (5,278 world crops labelled by title, noisier, train only).
* **Tokenizer** `voxels-cat` `116e68927d45` (`CorpusTokenizer.default(cats=True)`,
  `shards build <tier> --cat`): the class token takes the place of BOS,
  `[CAT, src, q, size, win, body, EOS]`, so lengths and body positions (and the
  `phase4` encoding) are those of the base tokenizer; the 13 class ids are appended
  after the base vocabulary (507 to 520). `init_from` a base checkpoint grows its
  embedding and `lm_head` rows (`trainer.extend_vocab`), the new rows starting as
  copies of BOS, so the extended model computes exactly what the checkpoint did.

| Tier | Tokenizer | Kind | `max_blocks` | Sequences | Tokens |
|---|---|---|---|---|---|
| `sft-cat` | `116e68927d45` | voxels-cat | 8192 | 5,049 | 18.6M |
| `eval-cat` | `116e68927d45` | voxels-cat | 8192 | 608 | 2.2M |
| `eval-holdout` | `116e68927d45` | voxels-cat | 8192 | 798 | 2.9M |
| `sft-cat-wc` | `116e68927d45` | voxels-cat | 8192 | 5,375 | 41.6M |

Sampling one class: `python -m blockgen.train --sample <run> --n 128 --cat all`
(or `--cat castle,ship`) writes one cache per class and `cat_samples.json`;
`python -m blockgen.eval.conditioned --samples <run>/cat_samples.json` scores probe
accuracy, per-class `geom_kid` and the own-class margin against `eval-cat`.

---

## 4. Tokenizers and the comparison arms

Every arm is scored by the same BlockScore protocol. Each arm is one
tokenization story. "Arm" here means a tokenizer arm; the generator families in
`models.md` are "tracks".

| Arm | Tokens | Trainer | Status |
|---|---|---|---|
| A. Sparse raster AR | `[BOS, src, q, size, win, (X, Y, Z, BLOCK)*, EOS]`, 507 tokens, hash `60684456e598` | `blockgen.train`, `arch: ar2`, `kind: voxels` | every M1, M2 and M2b run (T57 to T60); the oriented variant (926 tokens, `--oriented`) has no shard |
| B. Raster AR with 3D-BPE pieces | legacy: `ClusterVocab` pieces, per-run vocabulary; tiers: `kind: pieces`, 811 tokens, `cdce56584ebf` (section 3.8) | legacy (`blockgen/experiments_native.py`); `blockgen.train`, `arch: ar2`, `kind: pieces` | best ranked learned arm on the board, `canon_oriented` (T64), is legacy; the pieces kind has shards and no run yet |
| C. Masked discrete diffusion | dense grid with a MASK class | legacy (`blockgen/training/train_diffusion.py`) | T10 and T11 only; no dense-grid collate or non-CE loss in `blockgen.train` |
| D. LLM baseline | one text line per block, Qwen2.5 LoRA | legacy (`scripts/train_llm_geobias.py`) | T64 `llm_control` 167.92 and `llm_bias` 191.88 at n=128 (card ranks 5 and 8), both provisional on the board because their `train_split` is `trackd` (`protocol.py` `CLEAN_TRAIN_SPLITS`); the T29 pair predates the protocol |
| E. Facade 2.5D | per-face (514 tokens, `a9546c1f2f94`) or joint (514 tokens, `a4ca45ac1fe5`) | `blockgen.train`, `arch: ar2`, `kind: facades` | T66 (per_face 5M, joint 5M), T68 (joint 20M, joint d4 5M and 20M); card of record `run_20261028_143416` at n=399 |

The trainer is arch-blind (section 5) but the loop, the collate and the sampler
are autoregressive: next-token cross-entropy on a shifted sequence, a KV-cached
or re-forward decode. Arms C and D stay on their legacy loops; arm B runs on both,
legacy for `canon_oriented` and the `pieces` kind on tiers.

A tokenizer is an artifact with a hash: the first 12 hex digits of the sha256
of its saved JSON. The hash names the shard directory, is written into
`run.json` and every checkpoint, and `init_from_checkpoint` refuses a
checkpoint whose tokenizer hash differs from the shards.

---

## 5. The trainer

One module, `blockgen/train/trainer.py`, used by every stage. The stage is the
config, not the code.

```
python -m blockgen.train --config configs/train/<name>.yaml [key=value ...]
python -m blockgen.train --sample <run_dir> --n 399 [--ckpt best.pt] [--out-dir D] [temperature=0.9 top_k=20 seed=1]
```

**The arch registry** (`blockgen/train/archs.py`). `Arch(name, build, loss,
sample, track)`; `ar2` is the only entry. `build(cfg, tokenizer)` returns the
module, `loss(model, batch)` turns `(input_ids, targets, pad_mask)` into the
mean loss per non-pad token, `sample(model, tokenizer, cfg, n, seed=None)`
returns `(structures, extra)`, and `track` is the benchmark track prefix
(`ar`). A new family registers its own `Arch` and is selected by `arch: <name>`.
`cfg.model` (a dict) reaches the constructor as keyword arguments, so `ar2`
exposes `geo_bias`, `piece_factors` and `semantic_embedding` without new
`TrainConfig` fields; a key the constructor does not take, or one that
duplicates a config field, raises before any weights are allocated.

**The model contract**, from the `archs.py` docstring:

- `model(input_ids, pad_mask=None) -> logits`, `input_ids` of shape `[B, L]`
  right-padded with `PAD_TOKEN`, `pad_mask` a bool `[B, L]` True where padded,
  `logits` of shape `[B, L, vocab_size]`. Training and validation call only this.
- `model.max_seq_len`: the longest sequence it accepts. The batchers drop longer
  sequences (never truncate) and the sampler caps its token budget by it.
- Optional: `model.forward(step, past=past, return_cache=True) -> (logits, past)`
  for KV-cached decoding. `sample.supports_cache(model)` checks the signature
  and the model's own `supports_cache()`; a model without it is decoded by a
  full re-forward per step (`pe=alibi` takes this path).

**Config.** `TrainConfig` in `blockgen/train/config.py`; every field and its
default is tabled in `configs/README.md`. The fields added since the M1
trainer:

| Field | Default | What it does |
|---|---|---|
| `model` | `{}` | extra constructor kwargs for the arch (JSON object on the command line) |
| `train_split` | `None` | the provenance the bench ranks by; `None` records `canonical` iff `train == [sft-houses32]`, else the joined tier list (`split_name()`) |
| `trained_on_signals` | `[]` | the rewards a post-training stage optimised (`name[@weight]` tokens); travels into the sample report and the registry |
| `final_eval_all` | `True` | after selection, score the selected weights on every holdout sequence that fits (`val_nll_all`); the periodic and final passes stay on `eval_seqs` |
| `bench_tier` | `fast` | `fast`, `full` or `both`, passed to `blockgen.eval.bench --tier` |
| `bench_n` | `128` | samples benchmarked at the end; `base.yaml` pins 64, the M2 and RFT configs 256 |
| `select` | `best` | `best`, `last`, or a registered hook; `reward` is registered by `blockgen.posttrain` |
| `allow_partial_init` | `False` | accept an `init_from` checkpoint whose keys do not all match; counts land in `init_missing` / `init_unexpected` |

**Schedule and mechanics.** Warmup-stable-decay only (`lr_at`: linear warmup,
flat, then linear decay over the last `decay_frac` of steps to `min_lr_frac`);
there is no cosine option. bf16 autocast, gradient accumulation, grad clipping,
AdamW with weight decay on 2-D parameters other than embeddings, token-budget
length-bucketed batches
over weighted tier mixtures (`blockgen/train/data.py`, `MixedBatches`). All
train shards must share one tokenizer hash. `compile: true` wraps the model in
`torch.compile`; checkpoints are saved unwrapped and `load_weights` strips the
`_orig_mod.` prefix, so compiled runs reload.

**Stages inside the loop.** `init_from=<ckpt>` loads weights only into a fresh
optimizer; it refuses a tokenizer-hash mismatch, a dropout change
(`allow_dropout_change`, T59) and a key mismatch (`allow_partial_init`).
`resume=<run_dir>` restores model, optimizer, step, data cursor, RNG and best
from `last.pt`; the config is not reloaded. `stage:` is a registry label.

**Validation and selection.** Every `eval_every` steps the first `eval_seqs`
(256) holdout sequences that fit the context are scored per tier; `best.pt`
follows the first `val` tier. At the end: a final pass on the same subset, then
`select_checkpoint` loads the chosen weights (`best`, `last`, or the hook), then
`final_eval_all` scores them on the whole tier (736 of the 798 holdout houses
fit 8,200 tokens). The selected model is what gets sampled, benchmarked and
registered; `last.pt` keeps the final weights.

**End of run.** `sample_and_render` draws `max(n_samples, bench_n)` builds when
`bench` is on, else `n_samples` (`blockgen/train/trainer.py:376`), through
`Arch.sample`, writes `samples.png` (and `facades.png` for facade runs)
and a structure cache `samples_step<step>_<dim>.npz` whose `_manifest.json`
report carries `model` (the run name), `arch`, `kind`, `checkpoint {file,
step}`, `trained_on_signals`, `train_split`, the tags, temperature and top_k.
`bench` then runs `python -m blockgen.eval.bench --arms <track>/<name>:<npz>
--tier <bench_tier> --n <bench_n> --corpus <bench_corpus> --out <run>/bench
--name train_<name> --no-head-to-head` in a subprocess; a bench failure is a
warning in `bench.log`, never a failed run.
The card at `<run>/bench/scorecard.json` is discovered by the lab
(`tools/lab/catalog.py`, `SCORECARD_GLOBS`). Under `houses32-v1` a run is
ranked only when its card is tier `both` at n >= 128 with a clean
`train_split`, so the in-run fast-tier card is a progress number and the route
to the board is `--sample <run> --n 399` followed by the bench CLI at
`--tier both`.

**`--sample`.** Loads `best.pt` if it exists, else `last.pt`, applies any
`key=value` overrides with the YAML coercion (so `bench=false` is False) and
writes `samples_step<N>_<dim>.npz` under `--out-dir`, where N is `--n` (the
sample count, so `--n 399` gives `samples_step399_48.npz`); the name never
collides with the trainer's own `samples_step<step>` cache. The report inside
carries the checkpoint's real step.

**Presets** (`PRESETS`): `5m` (d_model 256, 6 layers, 5.5M params), `20m` (512,
8, 26.8M), `60m` (768, 10), `150m` (1024, 12), `300m` (1280, 16). Only `5m` and
`20m` have been trained; `20m` on the facade kind only.

---

## 6. Stages

| Stage | How it is expressed | Status |
|---|---|---|
| Pretrain | `stage: pretrain`, fresh WSD run on a pretraining tier, tags on | run at 5M on `anneal` (`configs/train/anneal-pretrain-5m.yaml`, three seeds, T59 and T60); no run on `pretrain-all` or `pretrain-clean` and no config for them |
| Anneal | continue into the decay phase on the `anneal` tier | planned; `resume` continues the same tiers and the loop has no tier switch or tag dropping |
| SFT | `stage: sft`, `init_from=<pretrain>/last.pt`, `configs/train/houses32-sft-5m.yaml` | run (M2b, T60) against `houses32-scratch-matched.yaml`; unconditioned, no caption or image channel in the shards or the sampler |
| Post-train | `stage: posttrain`, section 7 | best-of-N and rejection-sampling code landed 2026-10-31; no recorded run |
| Distil | teacher to deployable size | planned |

Three rules from T59 and T60 hold for every stage: never change dropout at
`init_from` (the trainer refuses), `select=best` on both arms of a comparison,
and pin `tokenizer:` in any config whose tier has more than one shard.

Every stage records its parent in `run.json` (`meta.init_from.run`) and the
registry (`parent_run`), so the registry is a lineage tree.

The two chained recipes that produced the M2 and M2b runs are
`scripts/m2_seeds.sh` (per seed: matched scratch, pretrain on `anneal`, SFT
from that checkpoint) and `scripts/m2b_seeds.sh` (the rerun that reuses the M2
pretrain checkpoints and applies the T59 fixes: SFT at the pretraining dropout
for 16k steps, scratch at the matched 36k, both `select=best`). Each writes its
logs to `outputs/logs/`; `SEEDS="0 1 2"` is the default loop.

---

## 7. Post-training

`blockgen/posttrain/` implements the first two rungs of the ladder and the
rule every rung is held to. The package docstring states the rule:

> A post-trained arm may be rewarded only on signals that are not BlockScore
> pillar inputs. The trained-on signal is reported as a method-internal metric,
> never as a BlockScore gain. Evaluation is the full BlockScore at n equal to
> the control n under houses32-v1 with the real@ controls, and claims are made
> only for pillars the reward did not touch. Because 26-connected connectivity
> correlates with the 6-connected coherence pillar, movement on coherence is
> pre-registered as expected transfer, not independent evidence.

**Rewards** (`blockgen/posttrain/rewards.py`). A reward is `name`, `pillar` and
`score(structs) -> np.ndarray`, higher is better, CPU only. `REWARDS` holds:

| Name | Score | Pillar |
|---|---|---|
| `face_solidity` | `lcc - (face_jumps + face_holes + face_lone + dust_blocks)` from `blockgen/eval/facade_stats.py`; empty build scores -4 | no |
| `lcc26` | share of blocks in the largest 26-connected component | no |
| `novelty_penalty` | minus nearest-train occupancy IoU at grid 24, minus one more for a duplicate (IoU >= 0.95); reference is the houses32-v1 train split | no |
| `composite` | weighted sum of the above, via `--weights name:w,...`; name `composite-<a>[@w]-<b>[@w]` | inherited |
| `coherence`, `geometry_scalars`, `traversal`, `geom_kid`, `mv_dino_kid`, `palette_jsd_exact` | placeholders that cannot score | yes |

`rewards.get(name)` raises `PillarReward` for the six pillar inputs; the CLIs
offer only `CLI_REWARDS = (face_solidity, lcc26, novelty_penalty)` plus
`--weights`. The signals a run trained on are recorded as `name[@weight]`
tokens in `trained_on_signals`, which the lab flags on the board and
`select=reward` rebuilds the reward from.

**Rung 1, best-of-N** (`blockgen/posttrain/bon.py`):

```
python -m blockgen.posttrain.bon --run <run_dir> --n 256 --k 8 --reward face_solidity
                                 [--weights face_solidity:2,novelty_penalty:0.5] [--temperature T] [--top-k K]
                                 [--batch B] [--ckpt best.pt] [--out-dir D] [--seed S]
```

Draws `n * k` builds through `sample_structures`, keeps the reward's favourite
of every `k`, and writes `<run>/bon_<reward>_k<k>_n<n>/bon_<dim>.npz` (suffixed
`_2`, `_3` when the directory is taken) with `scores.parquet` (group, index,
score, kept) and `report.json`. The cache report carries the parent's
`train_split`, `model` (the arch dict), `checkpoint` (path), `method: bon`,
`k`, `n`, `reward`, `reward_weights`, the draw's seed and the parent's
`trained_on_signals` with the reward's signals appended. It prints the
`--arms bon/<name>_bon_<reward>_k<k>:<npz>` string for the bench CLI. No
training, so this is the floor every trained method must beat at the same `n`.

**Rung 2, rejection-sampling fine-tuning** (`blockgen/posttrain/rft.py`):

```
python -m blockgen.posttrain.rft --run <run_dir> --reward face_solidity --n 2048 --keep-frac 0.25
                                 [--weights ...] [--temperature T] [--top-k K] [--batch B] [--ckpt best.pt] [--out-dir D] [--seed S]
```

Runs best-of-N with `k = round(1 / keep_frac)` into `<run>/rft_<reward>_k<k>_n<groups>/`,
writes the kept builds as the derived tier `rft-<run name>-<reward>` through
`shards.from_structures` (tags `src`, `q`, `win` copied from the parent's
`sample_tags`, `size` inferred per build, faces and augment at their defaults),
and prints the training command:

```
python -m blockgen.train --config configs/train/rft-voxel-5m.yaml init_from=<run>/best.pt \
    train=rft-<name>-<reward>:1,sft-houses32:1 tokenizer=<hash> trained_on_signals=<reward> dropout=<parent> name=rft_<name>_<reward>
```

`configs/train/rft-voxel-5m.yaml` (tokenizer `60684456e598`, dropout 0.0) and
`configs/train/rft-facades-joint-d4-20m.yaml` (preset 20m, `kind: facades`,
tokenizer `a4ca45ac1fe5`, `augment: ""`, dropout 0.1) are the two stage
configs: `stage: posttrain`, 4,000 steps, lr 5e-5, `select: best`, `bench_n`
256. The RFT tier is written unaugmented, so the facade config mixes it with
the plain joint shard. The first targets are `outputs/run_20261015_235516_m2b_sft_s1`
(voxel, T60) and `outputs/run_20261028_125240_houses32_facades_joint_d4_20m`
(facade, T68); T68 names edge agreement and empty interiors as the facade
model's failures, which is what `face_solidity` scores.

**`select=reward`** (`blockgen/posttrain/select.py`). Registered with
`trainer.register_select("reward")` when `blockgen.posttrain` is imported,
which `trainer.resolve_select` does on demand. A run with `select=reward` and
`trained_on_signals=<reward>` samples `n_samples` builds from its final weights
and from `best.pt`, keeps the set with the higher mean reward, and registers
that checkpoint like any other.

**Rungs 3 to 5**, GRPO with verifiable rewards, DPO or SimPO on lab 2AFC pairs
and reward-ranked pairs, merge and distil, are described in the package
docstring and have no code. The 2AFC pairs are collected on the lab's `/phone`
page (`python -m tools.lab --phone`, queue in `configs/lab/phone.json`);
`better_build` is the preference signal, `more_real` the human-study signal.
No trainer checkpoint is pinned in that queue yet, so the pairs on disk are
off-policy for every trainer model.

End to end on the GPU, the ladder ran once on a 60-step smoke model on
2026-10-31 (train, `--sample --n 8`, `bon --n 4 --k 2`, `rft --n 8
--keep-frac 0.5`, the printed command loading through `config.load`); the
derived smoke tier was removed afterwards. No real post-training run exists.

---

## 8. Tracking progress

Three fixed measurements, taken the same way for every run:

- Validation NLL on `eval-holdout`: the periodic curve on the first 256
  fitting sequences (`val_nll`, `val_nll_final`) and the final pass of the
  selected weights on all 736 that fit (`val_nll_all`, with the counts in
  `eval_seqs`). Quote `val_nll_all` when comparing runs.
- BlockScore on the end-of-run samples at `bench_tier` and `bench_n`, written
  to `<run>/bench/scorecard.json`; the ranked number comes from the `--sample`
  plus `--tier both` route in section 5.
- For conditioned models, `retrieval_acc` on the captioned houses. No trainer
  model is conditioned yet.

`outputs/registry.jsonl` is the machine-readable log, 40 lines today, written
by `trainer.run_record` and also saved as `<run>/summary.json`. The fields:

`time, run, name, stage, arch, kind, layout (facades only), augment, preset,
params, seed, dropout, lr, pe, model, steps, tokens, minutes, train, val,
train_split, tokenizer, tiers {tier: sha256}, git, config, init_from,
parent_run, init_missing, init_unexpected, trained_on_signals, train_loss,
val_nll, val_nll_final, val_nll_all, eval_seqs {periodic, final {tier: n}},
selected_step, selected_ckpt, select, best {nll, step}, bench_n, bench {ok,
seconds, tier, blockscore, status, worst_pillar, pillars, disqualified,
unranked}, card, samples_png, note`

The 40 rows come in three generations. Rows written before 2026-10-31 (28
rows, including both 20m facade runs) carry only the M1 field set: `time, run,
name, stage, params, steps, tokens, minutes, train, val, tokenizer, tiers, git,
init_from, train_loss, val_nll, bench, note`, plus `best`, `selected_step` and
`val_nll_final` from 2026-10-15 on. Rows written on 2026-10-31 before 02:24
(ten rows) lack `init_missing` and `init_unexpected`, and the first five of
those also `samples_png`. `bench_n` is written by `run_record`
(`blockgen/train/trainer.py:266`) but appears on no row yet; the newest row
predates that change. The lab's readers treat every one of these fields as
optional. `results.md` keeps
the narrative T-numbers; the registry is what the lab (`tools/lab/catalog.py`,
`train_run`) and the scaling plots read.

---

## 9. Milestones

| | Deliverable | Status (2026-10-31) |
|---|---|---|
| M0 | Manifest, dedup, kinds, templates, tiers, tokenizer artifact, shards for arm A | done 2026-10-13 (notes.md §46, §47; T55, T56) |
| M1 | Trainer, presets, registry, auto-eval; reproduces the `houses_32` scratch arm | done 2026-10-14 (notes.md §48; T57) |
| M2 | Pretraining pilot, the decisive matched-compute comparison | 5M on `anneal`, three seeds: T58 first comparison, T59 pilot invalid (dropout fault), M2b done 2026-10-16 with early stopping (T60): pretrain plus SFT beats scratch on likelihood and on the board at every seed. The 20M and 60M runs on `pretrain-all` against `pretrain-clean` have shards and no configs or runs |
| M3 | Post-training ladder, rungs 1 to 3, on the best M2 checkpoint | rungs 1 and 2 implemented 2026-10-31; GRPO not started; no recorded run |
| M4 | Arms B, C, D, E on the same tiers | E done through the trainer (T66, T68); B, C, D stay on legacy loops (T64 has B and D on the board) |
| M5 | 2AFC collection, preference tuning, distillation | collection running (notes.md §49; 311 `better_build` compares); no preference-tuning or distillation code |

Also landed since the design document: the arch registry, `select` hooks,
`final_eval_all`, `from_structures`, `StaleShardWarning`, `EmptyTierError`,
`scripts/corpus_refresh.sh`, and leaderboard discovery of `<run>/bench` cards
(all 2026-10-31).

---

## 10. Decisions taken and open

Taken in the design document and still standing: one shared trainer rather
than an external stack (our sequences, renderers and metrics are too specific
for TRL or torchtitan to save time); arm A first; rejects kept and tagged rather
than filtered; windows for large crops; the fast BlockScore tier as the
automatic end-of-run number.

Taken on 2026-10-30:

- The pipeline's interpreter is the conda `python` on PATH (torch and pandas).
  `.venv` stays for the legacy batteries and does not run the corpus layer.
- The arch registry is AR-first: `Arch` covers build, loss and sample, the loop
  and collate stay next-token. A dense-grid collate and a non-CE loss for arm C
  are separate work if arm C is ever re-run on tiers.
- `canon_oriented` stays a legacy run: it cannot be `init_from` into the new
  trainer (per-run vocabulary). 3D-BPE itself is now the `pieces` kind (section
  3.8), learned once from `sft-houses32` unoriented, so a pieces run is trained
  from scratch on tiers.

Open, for the user:

- Commit the pipeline. `blockgen/corpus`, `blockgen/train`,
  `blockgen/posttrain`, `configs/corpus`, `configs/train`, `tests/corpus`,
  `tests/train` and `tests/posttrain` are untracked, so the `git` sha every
  checkpoint and registry line records names a commit that contains none of
  this code. Until they are committed no run is reproducible from its recorded
  sha.
- The `geom_descriptor` stamp. `blockgen/eval/bench/runner.py` sets
  `ctx.derived["geom_descriptor"]` at line 303 and replaces `ctx.derived`
  wholesale at line 364, so no card on disk carries the stamp and the verdict
  recorded in the card is computed before scoring. The lab re-derives the match
  through `protocol.walk_standing`, so the board is right; the card is not.
  `blockgen/eval/bench` is frozen, so the fix (`ctx.derived.update(...)` and
  moving the verdict after scoring) needs the user's sign-off.
