__all__ = ["VoxelPortGNN", "VoxelTransformerAR", "LargePyGGraphGenerator", "LargePyGGraphGeneratorConfig"]


def __getattr__(name):
	if name == "VoxelPortGNN":
		from blockgen.models.voxel_port_gnn import VoxelPortGNN

		return VoxelPortGNN
	if name == "VoxelTransformerAR":
		from blockgen.models.voxel_transformer_ar import VoxelTransformerAR

		return VoxelTransformerAR
	if name in {"LargePyGGraphGenerator", "LargePyGGraphGeneratorConfig"}:
		from blockgen.models.large_pyg_graph_generator import (
			LargePyGGraphGenerator,
			LargePyGGraphGeneratorConfig,
		)

		if name == "LargePyGGraphGenerator":
			return LargePyGGraphGenerator
		return LargePyGGraphGeneratorConfig
	raise AttributeError(f"module 'blockgen.models' has no attribute {name!r}")
