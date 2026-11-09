# Category-conditioned generation: scoping and build

2026-11-05. Moves the training pipeline (`docs/pipeline.md`) from unconditional
houses to one prompt word per build. Unconditional houses saturate around
BlockScore 300 to 350; in July conditioning was the best cohesion lever (results.md
T10: raw validity 0.33, 0.4, 0.7 with category tokens); the world-crop pretraining
should pay off most where the target distribution is wide; and the live `/gen`
demo needs prompts. Part 1 is the data, Part 2 what was built. Nothing here has
been trained yet.

## Part 1. Scoping

### 1.1 Label sources

| Source | Builds | Label | Used |
|---|---|---|---|
| Corpus manifest, curated caches (`all_32`, `houses_32`, `houses_48`) | 10,445 rows, about 5,200 distinct builds | `route` = GrabCraft subcategory, `house` for 3D-Craft, text2mc tag phrase for text2mc | yes |
| GrabCraft 48 cache `gc_small_48` (not in the manifest before today) | 5,621 builds, max edge 48, 66 subcategories, title and 4 to 8 tags each | subcategory, title, tags | yes, as an auxiliary manifest |
| VLM labels `data/minecraft/labels/{all_32,houses_48}_labels.json` | 4,712 + 3,088 | one free-text `category` per build (2,096 "house", 590 "medieval house", 105 "statue", 74 "tower", ...) | yes, as a fallback |
| World-crop titles (`worldcrop` rows) | 915,853 crops, 106,398 worlds, 86% titled | the world's title | train-only side tier |
| 3D-Craft | 2,016 distinct builds | none beyond "house" (the collection instruction) | as house |
| text2mc CSV `TAGS` | 37,989 rows, 25,600 pages; by our rules: house 3,949, castle 1,444, church 685, tower 641, shop 571, vehicle 504, ... | page-level tags | no: only 529 text2mc builds are in the manifest (the house subset), the rest are undecoded world chunks with page-level labels |
| `tf_small_24` (PlanetMinecraft crawl) | 5,866 builds at max edge 24; by our rules house 1,039, vehicle 605, aircraft 450, statue 199, ... | title and tags | no: not in the manifest, 24-voxel cap, 1,287 redstone devices; a candidate for the next round |
| BrickNet captions (`data/minecraft/bricknet/captions_*.jsonl.xz`) | 1,861 SFT houses x 4 (3,708 SFT rows) | free text | no: all houses, so captions only split house subtypes |
| BlockLab rows (`tools/lab`) | the curated caches | `category` copied from the cache manifests | no new information |

One label per build comes from the first of: its title, its GrabCraft
subcategory (or the route as free text), its GrabCraft tags, its VLM category, and
finally its cache (a houses cache row is a house). A build at most 2 blocks thick
becomes `pixelart` whatever its words say. 93% of the 8,106 distinct curated builds
map to a class; titles decide 4,129, subcategories 3,024, the flatness rule 235,
tags 109, VLM 42, and 567 stay unmapped (`cat:none`: fountains, lanterns, roads,
gardens, bridges, ruins, the GrabCraft "other-190" bin).

Quality gates. Every curated build passes the world-crop gate (p_built >= 0.9 for
all but 3). The house curation gates (`curation.houses.quality_filter`) are house
specific: on the labelled train and eval builds they pass house 100%, castle,
farm, shop and tower 96 to 99%, ship 94%, aircraft 88%, vehicle 75%, statue 52%,
tree 42% and pixelart 12% (trees and statues fail `too_few_materials`, pixel art
`too_flat`). They are therefore not applied; the category tiers require only
`n_blocks >= 30` and a 48-voxel box.

### 1.2 Taxonomy

Minimum: at least 120 curated train builds per class after its eval draw (the
example 150 would drop ship at 124), plus 32 to 128 eval builds. Classes below it
fold into `cat:none` and still train. The vocabulary carries 13 classes so a class
can be promoted later without a new tokenizer hash.

| Class (prompt) | Raw labels mapped (examples) | Train (`sft-cat`) | Fits context | Eval (`eval-cat`) |
|---|---|---|---|---|
| house | all house subcategories, 3D-Craft, cottage, cabin, hut, mansion, treehouse, farmhouse | 1,741 (the houses_32 train split) | 1,535 | 128 (from eval-holdout) |
| vehicle | other-transportation, working, emergency, cars, buses, trains, tanks, wagons | 632 | 625 | 112 |
| shop | tavern, inn, bakery, blacksmith, market stall, restaurant, store | 385 | 296 | 69 |
| statue | statues, characters, superheroes, youtubers, animals, creatures | 269 | 248 | 47 |
| aircraft | planes, airships, spaceships, helicopters, jets | 232 | 225 | 41 |
| tree | trees, palms, giant mushrooms, ents | 231 | 218 | 41 |
| farm | farm buildings, barns, stables, windmills, watermills | 205 | 163 | 36 |
| pixelart | logos, emoji, anything 2 or fewer blocks thick | 199 | 199 | 35 |
| tower | towers, watchtowers, lighthouses, wizard towers | 198 | 159 | 35 |
| castle | castles, walls, gates, keeps, forts, barracks | 172 | 121 | 32 |
| ship | boats, sailing ships, warships, yachts | 124 | 120 | 32 |
| none | unmapped curated builds, and church (118, below the minimum) | 661 | 569 | 0 |
| total | | 5,049 sequences / 5,048 builds | 4,478 | 608 |

"Fits context" is the number of sequences within `max_seq_len` 8,200 that the
batcher keeps (it drops, never truncates). Spot checks of 14 train titles per
class read about 90 to 95% right; the known weak spots are castle (walls and
gates, a few military props) and statue (a dragon gate, a giant coffee cup).

Leftovers: church folds into none at 118 builds (temple, pagoda, shrine,
cathedral, chapel); bridge (45), ruin, garden, fountain and decoration bins are
too small for a class. 1,779 houses outside houses_32 (all_32, houses_48 and
GrabCraft 48 houses) are kept out (`extra`) so the house class is exactly the
pinned houses_32 train split.

### 1.3 World crops

A crop gets a class only from its world's title, only when the world has at most
3 crops in the manifest (a single-build world), at least 300 blocks, passes
`p_built >= 0.9`, not cut, `kind == build`, and the title names no map, server,
city or game (`labels.WC_STOP`). That yields 5,278 crops (`sft-cat-wc`): house
3,285, castle 366, shop 299, farm 259, tower 234, statue 232, aircraft 206, ship
193, vehicle 117, tree 56, pixelart 31.

They are noisy. The real-build probe of 1.5 agrees with the title class on 45% of
these crops overall, against its 69% on real eval builds; relative to the probe's
own per-class ceiling the agreement is about 50% for castle, farm, shop and tower,
80% for aircraft and 5% for tree (a "tree" world is rarely a tree). The
recommendation: world crops carry `cat:none` by default (that is what the
pretraining shard already is, unlabeled), and `sft-cat-wc` is at most a
low-weight mix (`train: [sft-cat:0.8, sft-cat-wc:0.2]`) to be tested against
`sft-cat` alone, never an eval source.

### 1.4 Split

* `eval-holdout` (houses_32 val and test) and every copy of its builds (same url
  or file basename, or same content hash) is never trained on; the 3D-Craft paths
  moved from `data/` to `data/minecraft/` between cache builds, so copies are
  matched on the basename, not the path.
* `eval-cat` draws per class 15% of its builds, at least 32 and at most 128,
  eval-holdout members first, then by a hash of the build key. The house class is
  128 eval-holdout houses. No eval build is a `sft-houses32` row or shares a
  duplicate cluster with `sft-houses32`, `anneal` or `anneal-worldcrop`.
* Checked on disk: `eval-cat` shares no id, url, content hash or cluster with
  `sft-cat`, `sft-cat-wc`, `sft-houses32`, `anneal` or `anneal-worldcrop` (one
  cluster with `pretrain-all`, which no current checkpoint used). 393 eval builds
  come only from the GrabCraft 48 cache, which the dedup sidecar has not seen, so
  near-duplicates of those in world crops are unchecked (exact content: none).

### 1.5 Conditioned evaluation (outside the frozen bench)

`blockgen/eval/conditioned.py` reuses the bench descriptor (v2, 80-d),
`Standardizer` and `geom_kid` unchanged:

* probe accuracy: logistic regression on [geometry descriptor, sqrt block-id
  histogram] fit on two thirds of each class's `sft-cat` builds, applied to the
  samples of each prompted class;
* per-class `geom_kid` against that class's `eval-cat` builds, whitened on the
  whole eval reference so the classes share a unit;
* margin: mean KID to the other classes minus KID to the own class.

Controls on real builds (n = 45 to 128 per class against 32 to 128 references):

| | probe acc | mean geom_kid | mean margin |
|---|---|---|---|
| real held-out builds as samples | 0.692 (ceiling 0.694, chance 0.091) | 0.21 (90% intervals straddle 0) | 17.97 |
| the same builds, classes shuffled | 0.089 | 9.39 | 0.13 |

Per-class probe ceilings on eval builds: pixelart 1.00, tree 0.95, house 0.77,
statue 0.74, vehicle 0.74, aircraft 0.63, ship 0.62, farm 0.56, castle 0.53, tower
0.51, shop 0.45. A generated class should be read against its own ceiling; the
building classes (shop, tower, castle, farm) confuse with house even for real
builds.

Feasible n: the eval side is fixed at 32 to 128 per class; sample n = 128 per
class (11 classes, 1,408 samples). On real builds the 90% KID intervals are about
1 to 1.7 wide at n_ref 32 to 41 and 0.3 to 0.9 wide at n_ref 112 to 128, against a
shuffled-class KID of 2 to 28, so per-class KID separates a right class from a
wrong one but not two good models that differ by less than about 1. For continuity the unconditional bench is run
unchanged on `cat:house` samples (the trainer's end-of-run bench does this, with
`sample_tags.cat: house`), and the house NLL on `eval-holdout` is directly
comparable with every houses32 run, because the conditioned sequences have the
same tokens after position 0.

### 1.6 Risks

* Imbalance: house is 34% of `sft-cat` and none 13%; vehicle is 5x ship. Uniform
  sampling over sequences will under-train the small classes; per-class reweighting
  needs one shard per class or a weighted sampler (not built).
* Label noise: title rules are about 90 to 95% right on curated builds; world-crop
  titles about half; castle and statue are the noisiest curated classes.
* Size and context: 11% of `sft-cat` sequences exceed 8,200 tokens and are
  dropped, 30% of castles and 20% of farms, shops and towers; the remaining
  castle class is 121 trainable builds.
* GrabCraft dominates every non-house class (one site, one style, many series:
  "Clash of Clans X 1..9"), so a class is partly a GrabCraft style.
* The probe is geometric and palette-based; it can be satisfied by a build with
  the right silhouette and blocks but no coherence. Read it with BlockScore.
* Leakage residuals as in 1.4.

## Part 2. What was built

### 2.1 Tokenizer `voxels-cat`, hash `116e68927d45`

`CorpusTokenizer.default(cats=True)` (any kind: `tok.with_cats()`; pieces and
facades checked in tests). The class token replaces BOS:

```
[CAT, src, q, size, win, (X, Y, Z, BLOCK)*, EOS]
```

Why there: `phase4` keys on absolute position (pos mod 4, pos div 4), and a sixth
prefix token would shift every body token's phase relative to what the
pretrained checkpoint learned. In BOS's slot, every sequence keeps its length
and every body token its position, house NLL stays comparable, and the class
token is a class-specific start token (the class-conditional GPT recipe). The 13
class ids sit after the whole base vocabulary (507 to 519), so no existing id
moves. Existing hashes are unchanged and asserted in
`tests/corpus/test_categories.py`: 60684456e598, 0e6ace66bcdf, cdce56584ebf,
a9546c1f2f94, a4ca45ac1fe5.

### 2.2 Tiers and shards

| Tier | Builds | sha (first 12) | Shard `116e68927d45` |
|---|---|---|---|
| `sft-cat` | 5,048 | 3c8b5c9058bc | 5,049 seqs, 18.6M tokens, 36 MB |
| `eval-cat` | 608 | 0736145782f4 | 608 seqs, 2.2M tokens, 4.2 MB |
| `eval-holdout` (unchanged tier) | 798 | 0f23acfed16f | 798 seqs, 2.9M tokens, 5.7 MB |
| `sft-cat-wc` | 5,278 | 1c3dfa9c188b | 5,375 seqs, 41.6M tokens, 80 MB |

Built at `--max-blocks 8192` like the houses shards. New disk: about 132 MB of
shards, 5.7 MB auxiliary manifest, 0.5 MB labels. The `anneal-worldcrop` shard was
not rebuilt with the new tokenizer: the init path below makes it unnecessary.

### 2.3 Pretrain init

The 5M world-crop checkpoint has vocab 507 on `60684456e598`.
`trainer.init_from_checkpoint` now accepts a checkpoint whose tokenizer is the
run's tokenizer minus the category group (`CorpusTokenizer.extends`), grows the
token embedding and the `lm_head` weight and bias to 520 rows with the new rows
copied from BOS, and records `meta.init_from.extended`. Since a class token sits
where BOS was and starts as BOS, the extended model's logits on every old token
equal the checkpoint's exactly (checked on the real checkpoint: max difference
0.0), and a 2-step CPU smoke SFT from it starts at loss 0.79 (uniform is 6.2). Any
other tokenizer mismatch is still refused.

### 2.4 Files

* `blockgen/corpus/labels.py` (new): rules, split, sidecar.
* `blockgen/corpus/manifest.py`: auxiliary manifests (`curated --aux`), GrabCraft
  cache items from `<name>_meta.json`, `load(aux=...)`.
* `blockgen/corpus/tiers.py`: `aux:` key, labels sidecar columns.
* `blockgen/corpus/tokenizer.py`: the category group.
* `blockgen/corpus/shards.py`: `--cat`, labels in `_tier_rows`, `cat` meta column,
  `find` prefers an unconditioned shard.
* `blockgen/train/trainer.py`: `extend_vocab`, init across the extension.
* `blockgen/train/sample.py`, `blockgen/train/__main__.py`: `--cat`, `--sizes`.
* `blockgen/eval/conditioned.py` (new).
* `configs/corpus/{sft-cat,eval-cat,sft-cat-wc}.yaml`,
  `configs/train/{cat-sft-5m-scratch,cat-sft-5m-pt}.yaml`.
* Tests: `tests/corpus/test_categories.py`, `tests/train/test_cat_init.py`,
  `tests/eval/test_conditioned.py`.
* Docs: `docs/pipeline.md` §3.10, `configs/corpus/README.md`.

### 2.5 Commands

```
# (a) 5M class-conditioned SFT from scratch (matched compute: 36k steps, select=best)
python -m blockgen.train --config configs/train/cat-sft-5m-scratch.yaml seed=1

# (b) the same from the 5M world-crop pretrain
python -m blockgen.train --config configs/train/cat-sft-5m-pt.yaml seed=1 \
    init_from=outputs/run_20261105_013018_wc5m_pretrain_u_s1/last.pt

# (c) per-class samples, then the conditioned eval
python -m blockgen.train --sample <run_dir> --n 128 --cat all
python -m blockgen.eval.conditioned --samples <run_dir>/cat_samples.json --boot 200
```

One class through the generic override also works:
`sample_tags='{"src":"curated","q":"4","size":"48","win":"whole","cat":"castle"}'`.
