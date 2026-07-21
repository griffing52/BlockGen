"""Export curated Minecraft caches into the MinecraftACE (LegoACE) dataset layout.

LegoACE consumes datasets as a directory of per-model token files plus JSON
manifests.  This module converts our ``Structure`` caches (e.g. ``houses_32``)
into that layout with the Minecraft-native **4-token record** ``(x, y, z, type)``
— LegoACE's 5-token brick record minus the rotation token, which axis-aligned
unit voxels don't need.

Record convention (raw, as stored in the ``.npy`` files):
    * coords: 1..max_dim  (translation-normalized ``c - min + 1``; 0 is BOS)
    * type:   0..num_blocks-1  (raw block-class index into the dat_dict)
    * row order: ``np.lexsort((z, x, y))`` — bottom-to-top layers (y primary),
      then x, then z; exactly LegoACE's brick ordering.

The dataset-side offset (``type += pos_range + 1``) is applied by
``MinecraftTokenDataset`` in libs/MinecraftACE, matching the LegoACE "+1"
convention used by MVNpzDataset/textDataset (NOT SingleTokenDataset's buggy
no-+1 variant).  Final sequence: ``[BOS=0, (x,y,z,type')*, EOS=vocab_size-1]``
with ``vocab_size = pos_range + num_blocks + 2``.

Output layout under ``--out`` (default ``data/minecraftace/houses_32``):
    <name>_dat_dict.json      block token string ("id" / "id:data") -> raw id
    <name>_pair_dict.json     block token string -> [block_id, block_data]
    <name>_rot_dict.json      {} (kept for LegoACE layout compatibility)
    {train,val,test}_dataset.json   id -> {tokens, images, text, n_blocks, source}
    tokens/{split}/<id>_moved.npy   (N, 4) int32 raw records
    index.json                id -> provenance/split (all structures, incl. filtered)
    export_report.json        counts, vocab size, filter stats

Re-running with ``--images-dir`` / ``--captions`` patches image paths and text
captions into the dataset JSONs without re-tokenizing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from blockgen.curation.houses import load_house_structures
from blockgen.tokenizers.cluster_bpe import ClusterVocab, learn_clusters
from blockgen.utils.data import Structure, _token_for
from blockgen.utils.serialize import BlockVocab, build_block_vocab

DEFAULT_OUT_DIR = os.path.join("data", "minecraftace", "houses_32")


def structure_id(index: int) -> str:
    return f"h{index:05d}"


# --- Structure <-> (N, 4) records -------------------------------------------
def structure_to_records(structure: Structure, vocab: BlockVocab) -> np.ndarray:
    """Serialize a structure to raw ``(x, y, z, type)`` records.

    Coordinates are 1-based (LegoACE ``c - min + 1``), types are raw block-class
    indices (0..num_blocks-1).  Unknown blocks are skipped, mirroring
    ``serialize.structure_to_tokens``.
    """
    s = structure.crop_to_non_air()
    if max(s.shape) > vocab.max_dim:
        raise ValueError(
            f"structure max dim {max(s.shape)} exceeds vocab.max_dim {vocab.max_dim}")

    coords = np.argwhere(s.occupied_mask)  # rows of (x, y, z), 0-based
    # LegoACE brick order: y (layer) primary, then x, then z.
    order = np.lexsort((coords[:, 2], coords[:, 0], coords[:, 1]))
    coords = coords[order]

    rows: List[Tuple[int, int, int, int]] = []
    for x, y, z in coords.tolist():
        tok = _token_for(int(s.block_ids[x, y, z]), int(s.block_data[x, y, z]))
        block_index = vocab.block_token_to_id.get(tok)
        if block_index is None:
            continue  # unknown block -> skip voxel rather than crash
        rows.append((x + 1, y + 1, z + 1, block_index))

    return np.asarray(rows, dtype=np.int32).reshape(-1, 4)


def records_to_structure(records: np.ndarray, vocab: BlockVocab) -> Structure:
    """Inverse of ``structure_to_records`` (raw records -> Structure)."""
    records = np.asarray(records, dtype=np.int64).reshape(-1, 4)
    valid = (
        (records[:, :3] >= 1).all(axis=1)
        & (records[:, :3] <= vocab.max_dim).all(axis=1)
        & (records[:, 3] >= 0) & (records[:, 3] < vocab.num_blocks)
    )
    records = records[valid]
    if len(records) == 0:
        return Structure(block_ids=np.zeros((1, 1, 1), np.int32),
                         block_data=np.zeros((1, 1, 1), np.int32))

    size = records[:, :3].max(axis=0)  # coords are 1-based
    block_ids = np.zeros(tuple(size), np.int32)
    block_data = np.zeros(tuple(size), np.int32)
    for x, y, z, t in records.tolist():
        bid, bdata = vocab.block_index_to_pair[t]
        block_ids[x - 1, y - 1, z - 1] = bid
        block_data[x - 1, y - 1, z - 1] = bdata
    return Structure(block_ids=block_ids, block_data=block_data)


def vocab_from_dat_dict(dat_dict_path: str, max_dim: int) -> BlockVocab:
    """Rebuild a ``BlockVocab`` from an exported dat_dict (+ sibling pair_dict)."""
    with open(dat_dict_path) as f:
        dat_dict: Dict[str, int] = json.load(f)
    dirname, basename = os.path.split(dat_dict_path)
    pair_path = os.path.join(dirname, basename.replace("dat_dict.json", "pair_dict.json"))
    pairs: Dict[str, List[int]] = {}
    if pair_path != dat_dict_path and os.path.exists(pair_path):
        with open(pair_path) as f:
            pairs = json.load(f)

    id_to_token = [tok for tok, _ in sorted(dat_dict.items(), key=lambda kv: kv[1])]

    def pair_for(tok: str) -> Tuple[int, int]:
        if tok in pairs:
            return int(pairs[tok][0]), int(pairs[tok][1])
        if ":" in tok:
            bid, bdata = tok.split(":", 1)
            return int(bid), int(bdata)
        return int(tok), 0

    return BlockVocab(
        max_dim=max_dim,
        block_token_to_id=dict(dat_dict),
        id_to_block_token=id_to_token,
        block_index_to_pair=[pair_for(tok) for tok in id_to_token],
    )


# --- 3D-BPE piece records (LegoACE "brick library" analog) -------------------
def structure_to_piece_records(structure: Structure, cv: ClusterVocab) -> np.ndarray:
    """Raw ``(x, y, z, piece_id)`` records — piece anchors 1-based, LegoACE order.

    One record covers a whole learned multi-voxel piece, the direct analog of
    LegoACE's multi-cell brick types (their 2x4 brick = 8 cells in one token).
    """
    from blockgen.tokenizers.cluster_bpe import _apply_merge, _atomic_labeling

    base = BlockVocab(
        max_dim=cv.max_dim,
        block_token_to_id={_token_for(*p): i for i, p in enumerate(cv.block_index_to_pair)},
        id_to_block_token=[_token_for(*p) for p in cv.block_index_to_pair],
        block_index_to_pair=cv.block_index_to_pair)
    c = structure.crop_to_non_air()
    if max(c.shape) > cv.max_dim:
        raise ValueError(f"structure max dim {max(c.shape)} exceeds max_dim {cv.max_dim}")
    inst, owner, nid = _atomic_labeling(c, base)
    for pid_new, (pa, pb, delta) in enumerate(cv.merges, start=cv.num_blocks):
        nid = _apply_merge(inst, owner, nid, pa, pb, delta, pid_new)

    anchors = np.array([d["anchor"] for d in inst.values()], dtype=np.int64)
    pieces = np.array([d["piece"] for d in inst.values()], dtype=np.int64)
    order = np.lexsort((anchors[:, 2], anchors[:, 0], anchors[:, 1]))  # y, x, z
    rows = np.concatenate([anchors[order] + 1, pieces[order, None]], axis=1)
    return rows.astype(np.int32).reshape(-1, 4)


def piece_records_to_structure(records: np.ndarray, cv: ClusterVocab) -> Structure:
    """Inverse: expand each piece's pattern at its (1-based) anchor."""
    records = np.asarray(records, dtype=np.int64).reshape(-1, 4)
    placed = []
    for x, y, z, pid in records.tolist():
        if not (1 <= x <= cv.max_dim and 1 <= y <= cv.max_dim and 1 <= z <= cv.max_dim
                and 0 <= pid < cv.n_pieces):
            continue
        for dx, dy, dz, bidx in cv.patterns[pid]:
            placed.append((x - 1 + dx, y - 1 + dy, z - 1 + dz, bidx))
    if not placed:
        return Structure(block_ids=np.zeros((1, 1, 1), np.int32),
                         block_data=np.zeros((1, 1, 1), np.int32))
    arr = np.array(placed, dtype=np.int64)
    size = arr[:, :3].max(axis=0) + 1
    bi = np.zeros(tuple(size), np.int32)
    bd = np.zeros(tuple(size), np.int32)
    for x, y, z, bidx in arr.tolist():
        rid, rdata = cv.block_index_to_pair[bidx]
        bi[x, y, z] = rid
        bd[x, y, z] = rdata
    return Structure(block_ids=bi, block_data=bd).crop_to_non_air()


def save_piece_vocab(cv: ClusterVocab, path: str) -> None:
    with open(path, "w") as f:
        json.dump({
            "max_dim": cv.max_dim,
            "num_blocks": cv.num_blocks,
            "patterns": [[list(cell) for cell in pat] for pat in cv.patterns],
            "merges": [[pa, pb, list(delta)] for pa, pb, delta in cv.merges],
            "block_index_to_pair": [list(p) for p in cv.block_index_to_pair],
            "oriented": cv.oriented,
        }, f)


def load_piece_vocab(path: str) -> ClusterVocab:
    with open(path) as f:
        blob = json.load(f)
    return ClusterVocab(
        max_dim=blob["max_dim"],
        patterns=[tuple(tuple(c) for c in pat) for pat in blob["patterns"]],
        block_index_to_pair=[tuple(p) for p in blob["block_index_to_pair"]],
        merges=[(pa, pb, tuple(delta)) for pa, pb, delta in blob["merges"]],
        num_blocks=blob["num_blocks"],
        oriented=blob.get("oriented", False),
    )


# --- splits ------------------------------------------------------------------
def split_indices(items: Sequence[dict], seed: int = 0,
                  ratios: Tuple[float, float, float] = (0.90, 0.05, 0.05),
                  ) -> Dict[int, str]:
    """Deterministic train/val/test assignment, stratified by corpus.

    Hash of ``source_path`` (fallback: index) keeps membership stable across
    re-exports even if the cache is re-filtered.
    """
    by_corpus: Dict[str, List[int]] = {}
    for i, item in enumerate(items):
        by_corpus.setdefault(item.get("corpus", "?"), []).append(i)

    def key(i: int) -> str:
        base = items[i].get("source_path") or str(i)
        return hashlib.md5(f"{seed}:{base}".encode()).hexdigest()

    assignment: Dict[int, str] = {}
    for corpus_indices in by_corpus.values():
        ordered = sorted(corpus_indices, key=key)
        n = len(ordered)
        n_val = max(1, round(n * ratios[1])) if n >= 3 else 0
        n_test = max(1, round(n * ratios[2])) if n >= 3 else 0
        for j, i in enumerate(ordered):
            if j < n - n_val - n_test:
                assignment[i] = "train"
            elif j < n - n_test:
                assignment[i] = "val"
            else:
                assignment[i] = "test"
    return assignment


# --- export ------------------------------------------------------------------
def export(out_dir: str = DEFAULT_OUT_DIR, max_dim: int = 32,
           max_blocks: int = 2047, seed: int = 0,
           images_dir: Optional[str] = None,
           captions_json: Optional[str] = None,
           tokenizer: str = "voxel", n_merges: int = 256) -> dict:
    """tokenizer="voxel": one record per block. tokenizer="bpe": one record per
    learned 3D-BPE piece (LegoACE brick-library analog); merges are learned on
    the train split only and replayed deterministically everywhere."""
    structures, manifest = load_house_structures(max_dim=max_dim)
    items = manifest["items"]
    name = os.path.basename(os.path.normpath(out_dir))

    captions: Dict[str, List[str]] = {}
    if captions_json:
        with open(captions_json) as f:
            captions = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    assignment = split_indices(items, seed=seed)

    if tokenizer == "bpe":
        train_structs = [s for i, s in enumerate(structures) if assignment[i] == "train"]
        cv = learn_clusters(train_structs, max_dim=max_dim, n_merges=n_merges)
        save_piece_vocab(cv, os.path.join(out_dir, f"{name}_piece_vocab.json"))
        num_classes = cv.n_pieces
        dat_dict = {f"piece_{i:05d}": i for i in range(cv.n_pieces)}
        to_records = lambda s: structure_to_piece_records(s, cv)
    else:
        # Vocab over the FULL cache (pre-filter) so it stays stable if max_blocks changes.
        vocab = build_block_vocab(structures, max_dim=max_dim)
        num_classes = vocab.num_blocks
        dat_dict = vocab.block_token_to_id
        to_records = lambda s: structure_to_records(s, vocab)
        with open(os.path.join(out_dir, f"{name}_pair_dict.json"), "w") as f:
            json.dump({tok: list(vocab.block_index_to_pair[i])
                       for tok, i in vocab.block_token_to_id.items()}, f, indent=1)

    with open(os.path.join(out_dir, f"{name}_dat_dict.json"), "w") as f:
        json.dump(dat_dict, f, indent=1)
    with open(os.path.join(out_dir, f"{name}_rot_dict.json"), "w") as f:
        json.dump({}, f)
    datasets: Dict[str, Dict[str, dict]] = {"train": {}, "val": {}, "test": {}}
    index: Dict[str, dict] = {}
    n_filtered = 0
    skipped_voxels = 0

    for i, (s, item) in enumerate(zip(structures, items)):
        sid = structure_id(i)
        split = assignment[i]
        records = to_records(s)
        n_blocks = len(records)  # records (pieces for bpe), not raw voxels
        if tokenizer == "voxel":
            skipped_voxels += int(s.occupied_mask.sum()) - n_blocks
        keep = 0 < n_blocks <= max_blocks

        index[sid] = {
            "source_path": item.get("source_path", ""),
            "corpus": item.get("corpus", ""), "category": item.get("category", ""),
            "title": item.get("title", ""), "url": item.get("url", ""),
            "n_blocks": n_blocks, "dims": item.get("dims", []),
            "split": split if keep else "filtered",
        }
        if not keep:
            n_filtered += 1
            continue

        tok_rel = os.path.join("tokens", split, f"{sid}_moved.npy")
        tok_path = os.path.join(out_dir, tok_rel)
        os.makedirs(os.path.dirname(tok_path), exist_ok=True)
        np.save(tok_path, records)

        image_paths = None
        if images_dir:
            candidates = [os.path.join(images_dir, f"{sid}_view{v}.png") for v in range(4)]
            if all(os.path.exists(p) for p in candidates):
                image_paths = [os.path.abspath(p) for p in candidates]

        datasets[split][sid] = {
            "tokens": tok_rel,
            "images": image_paths,
            "text": captions.get(sid),
            "n_blocks": n_blocks,
            "source": item.get("source_path", ""),
        }

    for split, entries in datasets.items():
        with open(os.path.join(out_dir, f"{split}_dataset.json"), "w") as f:
            json.dump(entries, f, indent=1)
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)

    report = {
        "name": name, "max_dim": max_dim, "max_blocks": max_blocks, "seed": seed,
        "tokenizer": tokenizer,
        "n_merges": n_merges if tokenizer == "bpe" else 0,
        "total": len(structures), "filtered_out": n_filtered,
        "skipped_unknown_voxels": skipped_voxels,
        "splits": {k: len(v) for k, v in datasets.items()},
        "splits_by_corpus": {
            split: dict(Counter(index[sid]["corpus"] for sid in entries))
            for split, entries in datasets.items()},
        "num_blocks_vocab": num_classes,
        "pos_range": max_dim,
        "vocab_size": max_dim + num_classes + 2,
        "max_seq_len": max(
            (e["n_blocks"] for split in datasets.values() for e in split.values()),
            default=0) * 4 + 2,
        "with_images": sum(1 for split in datasets.values()
                           for e in split.values() if e["images"]),
        "with_text": sum(1 for split in datasets.values()
                         for e in split.values() if e["text"]),
    }
    with open(os.path.join(out_dir, "export_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export houses cache to MinecraftACE layout.")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-dim", type=int, default=32)
    parser.add_argument("--max-blocks", type=int, default=2047,
                        help="drop structures with more blocks (4N+2 must fit n_positions)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--images-dir", default=None,
                        help="dir of <id>_view{0..3}.png renders to link into the dataset")
    parser.add_argument("--captions", default=None,
                        help="captions JSON {id: [4 strings]} to link into the dataset")
    parser.add_argument("--tokenizer", choices=["voxel", "bpe"], default="voxel",
                        help="voxel: one record per block; bpe: one record per "
                             "learned 3D-BPE piece (LegoACE brick-library analog)")
    parser.add_argument("--n-merges", type=int, default=256,
                        help="BPE merge steps (bpe tokenizer only)")
    args = parser.parse_args()
    report = export(out_dir=args.out, max_dim=args.max_dim, max_blocks=args.max_blocks,
                    seed=args.seed, images_dir=args.images_dir, captions_json=args.captions,
                    tokenizer=args.tokenizer, n_merges=args.n_merges)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
