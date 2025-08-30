import os

# makes a list.txt file with all schematic file names
def create_schematic_list(directory):
    with open("data/list.txt", "w") as list_file:
        for filename in os.listdir(directory):
            if filename.endswith(".schematic"):
                list_file.write(f"{filename}\n")


create_schematic_list("data/raw")