from nbtschematic import SchematicFile
# from litemapy import Schematic
# import nbtlib

    # Load an NBT file (schematics are typically gzipped NBT files)


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


# def load_litematic(path):
#     schem = Schematic.load(path)

# def load_schem(path):
#     pass

# 17611.litematic
# 15650.schem
# load_schematic("data/Schematics/15650.schem")
# load_schematic("data/Schematics/1.schematic")
