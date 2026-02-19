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
    pass

# def load_litematic(path):
#     schem = Schematic.load(path)

# def load_schem(path):
#     pass

# 17611.litematic
# 15650.schem
# load_schematic("data/Schematics/15650.schem")
# load_schematic("data/Schematics/1.schematic")
