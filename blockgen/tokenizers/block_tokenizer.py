from blockgen.utils.data_loader import load_schematic

# def tokenize_schematic(schematic_file):
#     # Tokenize the schematic file into a sequence of blocks
#     blocks = schematic_file.blocks
#     tokens = [block.id for block in blocks]
#     return tokens

def check_vocab(file_list):
    vocab = set()
    for file in file_list:
        schematic = load_schematic(file)
        for block in schematic.blocks:
            vocab.add(block.id)
    return vocab


def bpe(vocab, X):
    # Implement Byte Pair Encoding (BPE)
    
    pass


def block2vec():
    pass