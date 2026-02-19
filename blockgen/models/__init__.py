__all__ = ["VoxelPortGNN"]


def __getattr__(name):
	if name == "VoxelPortGNN":
		from blockgen.models.voxel_port_gnn import VoxelPortGNN

		return VoxelPortGNN
	raise AttributeError(f"module 'blockgen.models' has no attribute {name!r}")
