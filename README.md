# BlockGen
Generative AI for Voxel and Brick Structures (such as Minecraft or Lego)

## Generative experiments: comparing three approaches

`notebooks/experiments_compare.ipynb` trains and compares three generative models for
Minecraft voxel structures on a small subset (max-dim ≤ 24), sharing one tokenizer and
one nearest-neighbor novelty evaluation (no text embeddings, LegoGPT-style):

- **Track A** — autoregressive token transformer (`VoxelTransformerAR`): structures are
  serialized to an EOS-terminated `[BOS, (X,Y,Z,BLOCK)*, EOS]` stream, so variable size
  needs no fixed grid.
- **Track B** — masked discrete diffusion 3D-UNet (`VoxelUNet3D`) over a fixed canonical
  24³ grid (center-padded; air class down-weighted; MaskGIT-style iterative decoding).
- **Track C** — graph latent VAE (`LargePyGGraphGenerator`) over the block+port graph,
  decoding the same shared token stream.

```bash
# 1. Build the cached subset (skips ~184 corrupt files automatically)
python -m blockgen.data.build_cache --max-dim 24

# 2. Run notebooks/experiments_compare.ipynb end-to-end
```

Shared building blocks: `blockgen/data/build_cache.py` (filtered NBT cache),
`blockgen/utils/serialize.py` (Structure ↔ tokens ↔ canonical grid),
`blockgen/training/{train_ar,train_diffusion,train_graph}.py`,
`blockgen/eval/novelty.py` (occupancy-IoU nearest neighbors + comparison renders).

The novelty eval shows generated samples have low nearest-neighbor IoU to the training
set with high diversity — i.e. they are genuinely new structures, not memorized copies.

## Graph pipeline (torch_geometric)

Install optional graph dependencies:

```bash
pip install .[gnn]
```

Build a graph dataset from schematics:

```python
from blockgen.utils.graph_data import dataset_from_directory

dataset = dataset_from_directory("data/raw", max_dim=64)
graph = dataset[0]
```

Graph schema:
- 1 block node per voxel
- 6 child port nodes per block node (`+X, -X, +Y, -Y, +Z, -Z`)
- block nodes connect only to their own ports
- voxel-to-voxel adjacency is represented as port-to-port edges

Starter model:

```python
from blockgen.models import VoxelPortGNN

model = VoxelPortGNN()
out = model(graph)
```


source .venv/bin/activate

# PROBLEM
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
litemapy 0.11.0b0 requires nbtlib>=2.0.3, but you have nbtlib 1.12.1 which is incompatible.