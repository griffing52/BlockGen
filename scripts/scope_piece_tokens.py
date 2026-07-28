"""Measure the token savings of PIECE serialization vs per-voxel for the LLM.

For each house we serialize two ways and count Qwen tokens:
  * voxel: one `<name> <x> <y> <z>` line per occupied voxel (train_llm_brickgpt).
  * piece: one `<piece_name> <x> <y> <z>` line per 3D-BPE piece instance.

Reports the per-build token distribution and how many builds fit under 2048/4096
tokens each way -- the whole justification for the piece-level v2.

    .venv/bin/python scripts/scope_piece_tokens.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))  # for train_llm_brickgpt

import numpy as np

from blockgen.curation.houses import load_house_structures
from blockgen.export.minecraftace import load_piece_vocab
from blockgen.tokenizers.cluster_bpe import _apply_merge, _atomic_labeling
from blockgen.utils.serialize import BlockVocab
from blockgen.utils.data import _token_for
from train_llm_brickgpt import build_name_map, serialize_structure, _name_for_pair

VOCAB = "data/minecraftace/houses_32_bpe/houses_32_bpe_piece_vocab.json"


def build_piece_names(cv):
    """piece_id -> readable unique name `<majority_block>_<k>` (k per material)."""
    id_to_name, name_to_id = {}, {}
    counter: dict = {}
    for pid, pat in enumerate(cv.patterns):
        bidxs = [cell[3] for cell in pat]
        maj = max(set(bidxs), key=bidxs.count)
        bid, bdata = cv.block_index_to_pair[maj]
        base = _name_for_pair(bid, bdata)
        k = counter.get(base, 0)
        counter[base] = k + 1
        name = base if k == 0 else f"{base}_{k}"
        id_to_name[pid] = name
        name_to_id[name] = pid
    return id_to_name, name_to_id


def piece_instances(s, cv):
    base = BlockVocab(
        max_dim=cv.max_dim,
        block_token_to_id={_token_for(*p, oriented=cv.oriented): i
                           for i, p in enumerate(cv.block_index_to_pair)},
        id_to_block_token=[_token_for(*p, oriented=cv.oriented) for p in cv.block_index_to_pair],
        block_index_to_pair=cv.block_index_to_pair, oriented=cv.oriented)
    c = s.crop_to_non_air()
    if max(c.shape) > cv.max_dim:
        return None
    inst, owner, nid = _atomic_labeling(c, base)
    for pid_new, (pa, pb, delta) in enumerate(cv.merges, start=cv.num_blocks):
        nid = _apply_merge(inst, owner, nid, pa, pb, delta, pid_new)
    return sorted(inst.values(), key=lambda d: (d["anchor"][1], d["anchor"][2], d["anchor"][0]))


def serialize_pieces(s, cv, id_to_name):
    order = piece_instances(s, cv)
    if order is None:
        return None
    lines = []
    for d in order:
        ax, ay, az = d["anchor"]
        lines.append(f"{id_to_name[d['piece']]} {ax} {ay} {az}")
    return "\n".join(lines)


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")

    structures, _ = load_house_structures(max_dim=32)
    cv = load_piece_vocab(VOCAB)
    _, pair_to_name = build_name_map(structures)
    id_to_name, _ = build_piece_names(cv)
    print(f"[vocab] {cv.n_pieces} pieces ({cv.num_blocks} atomic, "
          f"{len(cv.merges)} merges), oriented={cv.oriented}")

    vox_tok, pie_tok, blocks = [], [], []
    for s in structures:
        occ = int(s.occupied_mask.sum())
        if occ == 0:
            continue
        vtxt = serialize_structure(s, pair_to_name)
        ptxt = serialize_pieces(s, cv, id_to_name)
        if ptxt is None:
            continue
        blocks.append(occ)
        vox_tok.append(len(tok(vtxt, add_special_tokens=False)["input_ids"]))
        pie_tok.append(len(tok(ptxt, add_special_tokens=False)["input_ids"]))

    vox_tok, pie_tok, blocks = map(np.array, (vox_tok, pie_tok, blocks))
    print(f"\n[{len(blocks)} houses]  blocks: median {np.median(blocks):.0f}, "
          f"max {blocks.max()}")
    for name, arr in (("voxel", vox_tok), ("piece", pie_tok)):
        print(f"\n{name}: tokens/build  median {np.median(arr):.0f}  "
              f"mean {arr.mean():.0f}  max {arr.max()}")
        print(f"   tokens/block  {arr.sum()/blocks.sum():.2f}")
        for cap in (1024, 2048, 4096):
            print(f"   fit <= {cap}: {(arr <= cap).sum()} builds")
    ratio = vox_tok.sum() / pie_tok.sum()
    print(f"\n[savings] piece uses {1/ratio:.2f}x the voxel tokens "
          f"(={ratio:.2f}x fewer)")


if __name__ == "__main__":
    main()
