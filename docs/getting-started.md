# Getting started

## Environment

BlockGen uses a project virtualenv at `.venv/`. **Always** use `.venv/bin/python` —
system/conda Python lacks `nbtschematic` and other deps.

- Python 3.13, PyTorch 2.10 + CUDA 12.8, `torch_geometric` 2.7, `nbtschematic`,
  `matplotlib`.
- GPU used for training and the IoU similarity matrix (an RTX 5070 Ti in dev).
- Not available in the env: pandas, sklearn, scipy, tensorflow — curation/eval are
  numpy + matplotlib + torch only, and tfrecords are parsed by hand.

```bash
# sanity check
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Build the data caches

Two caches live under `data/cache/` (gitignored). The **labeled** cache is preferred —
it carries titles/categories/tags/popularity from the crawl.

```bash
# labeled cache from the tfrecords (100% metadata) -> tf_small_24.npz + _meta.json
.venv/bin/python -m blockgen.data.tfrecord_dataset --max-dim 24

# (legacy, unlabeled) cache from data/raw -> small_24.npz
.venv/bin/python -m blockgen.data.build_cache --max-dim 24
```

See [Data & curation](data-and-curation.md) for why the labeled cache exists.

## Quickstart: curate → look

```python
from blockgen.curation import Curator
lab = Curator.from_labeled_cache(max_dim=24)     # loads tf_small_24
lab.summary()                                    # aggregate stats + top categories
houses = lab.search("house").filter(min_blocks=60, max_components=3, min_block_types=3)
print(len(houses))                               # ~714
fig = lab.contact_sheet(houses.indices[:18])     # render a grid of thumbnails
```

## Quickstart: run the experiments

Everything writes to `outputs/run_<stamp>/` (configs + figures + metrics).

```bash
STAMP=$(date +%Y%m%d_%H%M%S)

# figures for each curated subset + retrain the feasible (subset, track) pairs
.venv/bin/python -m blockgen.experiments --stamp $STAMP \
    --epochs-ar 100 --epochs-diff 150 --epochs-graph 100

# "how each model generates" filmstrips + AR-vs-diffusion + flow matching (houses)
.venv/bin/python -m blockgen.experiments_gen --stamp $STAMP \
    --canon-dim 12 --ar-seq 1600 --epochs-diff24 150 --epochs-ar 80 --epochs-diff-canon 120
```

## Build these docs

```bash
pip install mkdocs-material     # pulls mkdocs too
mkdocs serve                    # live preview at http://127.0.0.1:8000
mkdocs build                    # static site into site/
```
