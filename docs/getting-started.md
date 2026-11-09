# Getting started

## Environment

One interpreter: the conda `python` on `PATH` (Python 3.13, PyTorch 2.7 with
CUDA 12.8, pandas 3.0, `nbtschematic`, `matplotlib`). The corpus layer
(`blockgen/corpus`) imports pandas, and the project `.venv/` does not have it,
so the corpus and trainer commands below run from `python`, not
`.venv/bin/python`. The one thing `.venv/` still holds that conda does not is
`torch_geometric`, which only the legacy graph track imports
(`blockgen/training/train_graph.py` and the PyG models and loaders it uses:
`blockgen/models/voxel_port_gnn.py`, `blockgen/models/large_pyg_graph_generator.py`,
`blockgen/utils/graph_data.py`, `blockgen/utils/data_loader.py`). Every
command on these pages assumes the repo root as the working directory.
`blockgen` is not installed into conda, so anything run as a script path
(`python scripts/x.py`, `python examples/x.py`) wants `PYTHONPATH=.`;
`python -m blockgen...` commands, the corpus CLIs included, find the package
from the working directory and do not.

```bash
python -c "import torch, pandas; print(torch.__version__, torch.cuda.is_available(), pandas.__version__)"
PYTHONPATH=. python -m pytest tests/corpus tests/train -q -m "not slow"     # 60 tests, seconds
```

The GPU (an RTX 5070 Ti in dev, 16 GB) is used for training, sampling, the
render half of the benchmark and the DINOv2 feature cache. Everything on the
fast tier of the benchmark runs on the CPU.

## Data

Three layers, each built from the one before ([Data](data.md)):

1. **Caches** under `data/minecraft/cache/` (gitignored): the curated house
   caches `houses_32` (2,661 builds) and `houses_48`, the whole-corpus cache
   `all_32`, and the labelled crawl caches. The pinned benchmark split
   `data/minecraft/splits/houses_32.s0.70-15-15.v1.json` (train 1,863 / val 399
   / test 399) is tracked in git and defines every number in the eval suite.
2. **The corpus layer** under `data/minecraft/corpus/`: a manifest (one row per
   build, every source), dedup and kind sidecars, tiers (an id list with a
   sha256 per `configs/corpus/<name>.yaml`) and shards (a tier through a
   tokenizer, memory-mapped by the trainer).
3. **Shards the trainer reads.** For the house split:

```bash
python -m blockgen.corpus.manifest curated --caches all_32 houses_32 houses_48
python -m blockgen.corpus.tiers build sft-houses32 eval-holdout
python -m blockgen.corpus.shards build sft-houses32 --max-blocks 8192
python -m blockgen.corpus.shards build eval-holdout --max-blocks 8192
python -m blockgen.corpus.shards show sft-houses32      # 1,863 sequences, tokenizer 60684456e598
```

`scripts/corpus_refresh.sh --dry-run` prints the whole order from the raw
sources to every shard a config reads; `configs/corpus/README.md` lists the
seven tiers. The legacy labelled caches still build with
`python -m blockgen.data.tfrecord_dataset --max-dim 24` and are what the
`Curator` and the old batteries read.

## Train, sample, evaluate

```bash
# a two-minute end-to-end check of the trainer (60 steps, 16 samples, a fast-tier card)
python -m blockgen.train --config configs/train/smoke.yaml

# a real run on the house split, early-stopped on holdout NLL
python -m blockgen.train --config configs/train/houses32-scratch-matched.yaml name=mine_s0

# an arm of board size from the selected checkpoint, then the official benchmark
python -m blockgen.train --sample outputs/run_<stamp>_mine_s0 --n 399
python -m blockgen.eval.bench --tier both --n 399 --baselines --name "mine_s0, n=399" \
    --arms ar/mine_s0:outputs/run_<stamp>_mine_s0/samples_step399_48.npz
```

Every run writes `outputs/run_<YYYYMMDD_HHMMSS>_<name>/` with its config,
checkpoints, rendered samples and a benchmark card, and one line in
`outputs/registry.jsonl`. The full route is [Workflow](workflow.md); a new
architecture plugs in through `blockgen/train/archs.py`
([Add a model](add-a-model.md), [Bring your own model](custom-model.md)); the
card is read with [Evaluation suite](benchmark.md). A fresh clone builds the
benchmark's two gitignored caches before its first `--tier both` run, the
geometry ladder (`python -m blockgen.eval.bench.ladder --family geometry --n 64 --n-ref 256`)
and the DINOv2 feature cache (`python -m blockgen.eval.bench.features --backbone dinov2b`
followed by `python -m blockgen.eval.bench.ladder --family render --n 64 --n-ref 256`).

## Look at what you made

```bash
python -m tools.lab --open      # http://127.0.0.1:8765
```

BlockLab reads whatever is already under `outputs/`; there is no import step.
`/runs` shows one card with the actual builds beside its numbers,
`/leaderboard` ranks models across cards (grouped by n, with the protocol's
reasons for every unranked row), `/curate` is where corpus triage happens, and
`python -m tools.lab --phone` serves the one-thumb queue for the human study.
See [BlockLab](lab.md).

## Quickstart: curate and look (legacy caches)

```python
from blockgen.curation import Curator
lab = Curator.from_labeled_cache(max_dim=24)     # loads tf_small_24 from data/minecraft/cache
lab.summary()                                    # aggregate stats + top categories
houses = lab.search("house").filter(min_blocks=60, max_components=3, min_block_types=3)
fig = lab.contact_sheet(houses.indices[:18])     # a grid of thumbnails
```

The legacy batteries (`python -m blockgen.experiments --stamp $(date +%Y%m%d_%H%M%S) ...`,
`blockgen.experiments_gen`) write to the same `outputs/run_<stamp>_<name>/`
shape; [Experiments & outputs](experiments.md) lists them.

## Build these docs

```bash
pip install mkdocs-material     # pulls mkdocs too
mkdocs serve                    # live preview at http://127.0.0.1:8000
mkdocs build                    # static site into site/
```
