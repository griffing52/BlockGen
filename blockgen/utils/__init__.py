__all__ = [
	"EDGE_BLOCK_TO_PORT",
	"EDGE_PORT_TO_PORT",
	"NODE_BLOCK",
	"NODE_PORT",
	"GraphBuildConfig",
	"SchematicGraphDataset",
	"dataset_from_directory",
	"dataset_from_list_file",
	"list_schematic_files",
	"structure_to_pyg_data",
]


def __getattr__(name):
	if name in __all__:
		from blockgen.utils import graph_data

		return getattr(graph_data, name)
	raise AttributeError(f"module 'blockgen.utils' has no attribute {name!r}")
