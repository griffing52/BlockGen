from blockgen.utils.data_loader import load_schematic
from blockgen.tokenizers.standard_vocab import STANDARD_VOCAB
import numpy as np

# def tokenize_schematic(schematic_file):
#     # Tokenize the schematic file into a sequence of blocks
#     blocks = schematic_file.blocks
#     tokens = [block.id for block in blocks]
#     return tokens

def check_vocab(file_list):
    vocab = set()
    for file in file_list: 
        schematic = load_schematic(file)
        blocks = schematic.blocks.astype(int)
        data = schematic.data.astype(int)

        # Flatten arrays to 1D for vectorized processing
        blocks_flat = blocks.ravel()
        data_flat = data.ravel()

        pairs = set(zip(blocks_flat, data_flat))
        vocab.update(f"{int(b)}" if d == 0 else f"{int(b)}:{int(d)}" for b,d in pairs)
    return vocab

def token_to_block_name(token):
    a,b = token.split(":") if ":" in token else (token, "0")
    return STANDARD_VOCAB.get(token, STANDARD_VOCAB.get(a, "Unknown Block")+f":{b}")

def bpe(vocab, file_list):
    # Implement Byte Pair Encoding (BPE)
    vocab = set()
    
    return vocab

def block2vec():
    pass