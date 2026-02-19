from nbtschematic import SchematicFile

def load_schematic(path):
    # nbt_data = nbtlib.load(path)
    # print(nbt_data) # Prints a human-readable representation of the NBT data
    schematic_file = SchematicFile.load(path)

    return schematic_file

    # print(f"Schematic shape: {schematic_file.shape}")
    # print(f"Number of blocks: {schematic_file.blocks.size}")

    # # Access a specific block's ID at a given coordinate (Y, Z, X)
    # # Note: Schematic files often use YZX order for coordinates
    # x_coord, y_coord, z_coord = 0, 0, 0
    # block_id = schematic_file.blocks[y_coord, z_coord, x_coord]
    # print(f"Block ID at ({y_coord}, {z_coord}, {x_coord}): {block_id}")

def get_file_list(path):
    with open(path, "r") as f:
        file_list = [line.strip() for line in f.readlines()]
    return file_list

def parse_data(path):
    file_list = get_file_list(path)

    for file in file_list:
        schematic = load_schematic(file)
        blocks = schematic.blocks.astype(int)
        data = schematic.data.astype(int)

        # Flatten arrays to 1D for vectorized processing
        blocks_flat = blocks.ravel()
        data_flat = data.ravel()

        pairs = set(zip(blocks_flat, data_flat))
        # vocab.update(f"{int(b)}" if d == 0 else f"{int(b)}:{int(d)}" for b,d in pairs)
# 
# TODO pad minecraft structures with End Of Block block (air). 

def schemToGraph(schem):
    from blockgen.utils.data import Structure
    from blockgen.utils.graph_data import structure_to_pyg_data

    structure = Structure.from_schematic(schem)
    return structure_to_pyg_data(structure)


def load_schematic_graph(path, *, include_air=False, crop_non_air=True, max_dim=None):
    """Load one schematic and convert it to a torch_geometric Data graph."""
    from blockgen.utils.data import Structure
    from blockgen.utils.graph_data import GraphBuildConfig, structure_to_pyg_data

    schematic = load_schematic(path)
    structure = Structure.from_schematic(schematic, source_path=path)

    config = GraphBuildConfig(
        include_air=include_air,
        crop_non_air=crop_non_air,
        max_dim=max_dim,
    )
    if config.crop_non_air:
        structure = structure.crop_to_non_air()
    if config.max_dim is not None:
        structure = structure.downsample(max_dim=config.max_dim)

    return structure_to_pyg_data(structure, include_air=config.include_air)


def make_graph_dataset(
    path,
    *,
    from_list_file=False,
    include_air=False,
    crop_non_air=True,
    max_dim=None,
):
    """Create a simple graph dataset from a directory or a list file."""
    from blockgen.utils.graph_data import dataset_from_directory, dataset_from_list_file

    if from_list_file:
        return dataset_from_list_file(
            path,
            include_air=include_air,
            crop_non_air=crop_non_air,
            max_dim=max_dim,
        )

    return dataset_from_directory(
        path,
        include_air=include_air,
        crop_non_air=crop_non_air,
        max_dim=max_dim,
    )

# def load_litematic(path):
#     schem = Schematic.load(path)

# def load_schem(path):
#     pass

# 17611.litematic
# 15650.schem
# load_schematic("data/Schematics/15650.schem")
# load_schematic("data/Schematics/1.schematic")
