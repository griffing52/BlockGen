# Corpus tiers

Each YAML here is a tier: a filter over the corpus manifest
(`blockgen/corpus/manifest.py`), materialised by
`python -m blockgen.corpus.tiers build [names ...]` into
`data/minecraft/corpus/tiers/<name>.{json,ids.txt}`. The JSON carries `n`,
`n_blocks_total`, `fits32`, `fits48`, `by_source` and the sha256 of the sorted
member ids; that sha256 is the tier hash a checkpoint records (`run.json`
`meta.tiers`, registry `tiers`). Changing a filter changes the hash. See
`docs/pipeline.md` §3.3. Commands run under the conda `python` on PATH, the
interpreter with pandas.

## Schema

Every key is optional. `materialise` in `blockgen/corpus/tiers.py` applies them
in this order:

| Key | Type | What it does |
|---|---|---|
| `description` | string | copied into the tier JSON |
| `holdout_split` | path, repo-relative | pin the tier to a split file (`data/minecraft/splits/houses_32.s0.70-15-15.v1.json`). The tier is restricted to the ids `cache:<split corpus>:<i>` of the parts named below. A split-backed tier that resolves to zero rows raises `EmptyTierError` instead of writing an empty tier; the message says whether the cache is missing from the manifest (`python -m blockgen.corpus.manifest curated --caches houses_32`) or the tier's own filters removed every row |
| `holdout_parts` | list of `train`, `val`, `test` | which parts of `holdout_split` to take; default `[val, test]` |
| `sources` | list of manifest `source` values | keep rows whose `source` is listed: `worldcrop`, `grabcraft`, `3dcraft`, `text2mc`, `text2mc_schem`, `templates` |
| `filter` | pandas query string | over the manifest columns (`n_blocks`, `p_built`, `cut`, `oversize`, `route`, `dx`, `dy`, `dz`, `source`, `month`, `world_id`, ...) plus `max_dim` (largest of `dx, dy, dz`), the dedup sidecar (`is_rep`, `cluster`, `cluster_size`, `cluster_worlds`) and the kinds sidecar (`kind` in build / mixed / village / mineshaft, `mine_share`). The sidecars come from `python -m blockgen.corpus.dedup` and `python -m blockgen.corpus.kinds`; when a sidecar file is absent every row is its own representative with `kind == build` |
| `dedup` | bool, default `true` | keep one representative per duplicate cluster (`is_rep`) |
| `exclude` | list of tier names | drop every row whose `world_id` or dedup `cluster` appears in those tiers. `tiers build` materialises an excluded tier first, so order in the command line does not matter |
| `aux` | list of auxiliary cache names | also read `corpus/manifest_aux/curated_<name>.parquet` (written by `python -m blockgen.corpus.manifest curated --aux --caches <name>`); only tiers that list a cache see its rows. The labels sidecar (`corpus/labels.parquet`, `python -m blockgen.corpus.labels build`) adds `cat`, `cat_split` and `cat_src` to every frame |

`max_dim` is a derived column, not a key: filter on it with `"max_dim <= 32"`.

## The tiers

| File | Kind of tier | Role |
|---|---|---|
| `pretrain-all.yaml` | `sources` + `filter` + `dedup` + `exclude: [eval-holdout]` | every build, deduplicated, terrain kept and tagged |
| `pretrain-clean.yaml` | same shape, `p_built >= 0.5 and not cut and kind in ['build', 'mixed']` | the soft-filtering ablation |
| `anneal.yaml` | same shape, substantial confident crops plus every non-worldcrop build with 80+ blocks, `kind == 'build'` | the pretraining tier of every run so far |
| `sft-houses32.yaml` | `holdout_split` with `holdout_parts: [train]`, `dedup: false` | the SFT target: the pinned train split |
| `eval-holdout.yaml` | `holdout_split` with `holdout_parts: [val, test]`, `dedup: false` | validation; excluded from every other tier |
| `vanilla.yaml` | `sources: [worldcrop]`, `kind in ['village', 'mineshaft']` | generated content, one per cluster; no shard |
| `templates.yaml` | `sources: [templates]` | the game's own structure pieces; internal only |
| `sft-cat.yaml` | `aux: [gc_small_48]`, `filter: cat_split == 'train'`, `exclude: [eval-holdout, eval-cat]` | the category-conditioned SFT tier (`blockgen.corpus.labels`) |
| `eval-cat.yaml` | `aux: [gc_small_48]`, `filter: cat_split == 'eval'` | the class-balanced category eval set; never trained on |
| `sft-cat-wc.yaml` | `sources: [worldcrop]`, `filter: cat_split == 'wc_train'` | world crops labelled by world title; a low-weight mix |

Counts, hashes and shard status for each are tabled in `docs/pipeline.md` §3.3
and §3.6.

## Build order

```
python -m blockgen.corpus.manifest build --workers 6
python -m blockgen.corpus.manifest curated --caches all_32 houses_32 houses_48
python -m blockgen.corpus.templates
python -m blockgen.corpus.dedup
python -m blockgen.corpus.kinds        # no arguments; any invocation (even --help) rewrites corpus/kinds.parquet
python -m blockgen.corpus.tiers build
python -m blockgen.corpus.shards build <tier> [--kind facades --layout joint --augment d4]
```

The category tiers read labels that are drawn against the built `eval-holdout`,
`sft-houses32` and anneal tiers, so they come after them:

```
python -m blockgen.corpus.manifest curated --aux --caches gc_small_48
python -m blockgen.corpus.labels build
python -m blockgen.corpus.tiers build eval-cat sft-cat sft-cat-wc
python -m blockgen.corpus.shards build sft-cat --cat --max-blocks 8192      # and eval-cat, eval-holdout, sft-cat-wc
```

`scripts/corpus_refresh.sh [--dry-run]` runs the whole order. Rebuilding a tier
changes its sha; a shard built from the old sha warns `StaleShardWarning` when
opened, so rebuild the shard after the tier.

## Derived tiers

A tier can also be written from model samples without a YAML here:
`python -m blockgen.corpus.shards from-npz <npz> --tier <name>` or
`blockgen.corpus.shards.from_structures`. Its JSON carries `kind: derived` and
the sha256 of the kept structures' content. A derived tier name may not be the
name of a YAML in this directory. `python -m blockgen.posttrain.rft` writes one
named `rft-<run name>-<reward>`.
