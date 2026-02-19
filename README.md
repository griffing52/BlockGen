# BlockGen
Generative AI for Voxel and Brick Structures (such as Minecraft or Lego)

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