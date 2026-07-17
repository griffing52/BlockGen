"""Sample from a conditioned run (train_conditioned.py) and evaluate fidelity.

Conditions on HELD-OUT val structures' embeddings, generates with CFG, decodes
pieces, and measures conditioning fidelity = occupancy IoU between each sample
and the structure whose image/caption conditioned it (grid-normalized, same as
eval.novelty). Renders sample-vs-target grids for eyeballing.

Usage:
    python scripts/sample_conditioned.py --run outputs/cond/image_run \
        --num 16 --cfg 3.0 --out outputs/figures/cond
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from blockgen.curation.houses import load_house_structures
from blockgen.eval.novelty import voxelize_occupancy
from blockgen.eval.validity import n_components
from blockgen.export.minecraftace import load_piece_vocab, split_indices
from blockgen.models.voxel_transformer_cond import CondVoxelAR2
from blockgen.renderer.grid import save_grid
from blockgen.training.train_conditioned import (BOS, EOS, NUM_SPECIAL, EMBEDS,
                                                 PIECE_VOCAB, build_sequences,
                                                 build_sequences_voxel)
from blockgen.utils.serialize import BlockVocab, tokens_to_structure as voxel_tokens_to_structure
from blockgen.utils.data import _token_for


def load_run_data(run: Path, cfg: dict, structures):
    """Returns (seqs, decode_fn, eval_vocab) for either representation."""
    if cfg.get("repr", "bpe") == "voxel":
        from blockgen.experiments_gen import canonicalize
        blob = json.loads((run / "block_vocab.json").read_text())
        vocab = BlockVocab(
            max_dim=blob["max_dim"],
            block_token_to_id=blob["block_token_to_id"],
            id_to_block_token=[t for t, _ in sorted(
                blob["block_token_to_id"].items(), key=lambda kv: kv[1])],
            block_index_to_pair=[tuple(p) for p in blob["block_index_to_pair"]])
        structures = canonicalize(structures, cfg["canon_dim"])
        seqs, _ = build_sequences_voxel(structures, vocab, cfg["max_seq_len"])
        return structures, seqs, (lambda t: voxel_tokens_to_structure(list(t), vocab)), vocab
    cv = load_piece_vocab(PIECE_VOCAB)
    seqs, _ = build_sequences(structures, cv, cfg["max_seq_len"])
    tokens_v = [_token_for(*p) for p in cv.block_index_to_pair]
    vocab = BlockVocab(max_dim=32,
                       block_token_to_id={t: i for i, t in enumerate(tokens_v)},
                       id_to_block_token=tokens_v,
                       block_index_to_pair=list(cv.block_index_to_pair))
    return structures, seqs, (lambda t: tokens_to_structure(t, cv)), vocab


def tokens_to_structure(tokens, cv):
    """[BOS,(X,Y,Z,PIECE)*,EOS] -> Structure via piece expansion."""
    from blockgen.export.minecraftace import piece_records_to_structure
    coord_off = NUM_SPECIAL
    piece_off = NUM_SPECIAL + cv.max_dim
    recs, buf = [], []
    for t in tokens:
        t = int(t)
        if t == EOS:
            break
        if t in (BOS, 0):
            continue
        if len(buf) < 3:
            if coord_off <= t < piece_off:
                buf.append(t - coord_off + 1)
            else:
                buf = []
        else:
            if piece_off <= t:
                recs.append(buf + [t - piece_off])
            buf = []
    return piece_records_to_structure(np.array(recs or [[1, 1, 1, 0]]), cv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--ckpt", default="best.pt")
    parser.add_argument("--num", type=int, default=16)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--max-new", type=int, default=2500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--out", default="outputs/figures/cond")
    args = parser.parse_args()

    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text())
    device = "cuda"

    structures, manifest = load_house_structures(max_dim=32)
    structures, seqs, decode, bv = load_run_data(run, cfg, structures)
    assignment = split_indices(manifest["items"], seed=0)
    val_idx = [i for i in range(len(seqs))
               if assignment[i] == "val" and seqs[i] is not None][: args.num]

    blob = np.load(EMBEDS)
    embeds = blob["image_embeds" if cfg["cond"] == "image" else "text_embeds"]
    if cfg["cond"] == "image":
        cond = torch.tensor(embeds[val_idx], device=device)          # (N,4,768)
    else:
        cond = torch.tensor(embeds[val_idx, 0][:, None], device=device)  # (N,1,512)

    model = CondVoxelAR2(
        cond_dim=cfg["cond_dim"], n_prefix=cfg["n_prefix"],
        vocab_size=cfg["vocab_size"], max_seq_len=cfg["max_seq_len"],
        d_model=cfg["d_model"], nhead=8, num_layers=cfg["layers"],
        dim_feedforward=4 * cfg["d_model"], pe=cfg["pe"]).to(device).eval()
    model.load_state_dict(torch.load(run / args.ckpt, map_location=device))

    toks = model.generate_cond(
        cond=cond, bos_token_id=BOS, eos_token_id=EOS,
        max_new_tokens=args.max_new, temperature=args.temperature,
        top_k=args.top_k, cfg_scale=args.cfg)

    samples = [decode(row.cpu().numpy()) for row in toks]
    targets = [structures[i] for i in val_idx]

    # conditioning fidelity: occupancy IoU sample vs its own conditioning target
    occ_s, _ = voxelize_occupancy(samples, 32, bv)
    occ_t, _ = voxelize_occupancy(targets, 32, bv)
    occ_s = occ_s.reshape(len(samples), -1).astype(bool)
    occ_t = occ_t.reshape(len(targets), -1).astype(bool)
    inter = (occ_s & occ_t).sum(axis=1)
    union = (occ_s | occ_t).sum(axis=1)
    pair_iou = inter / np.maximum(1, union)
    # chance floor: IoU vs a shuffled pairing
    perm = np.roll(np.arange(len(samples)), 1)
    inter_r = (occ_s & occ_t[perm]).sum(axis=1)
    union_r = (occ_s | occ_t[perm]).sum(axis=1)
    rand_iou = inter_r / np.maximum(1, union_r)

    # palette fidelity: cosine sim of block-type histograms, paired vs shuffled
    # (occupancy IoU is blind to materials; conditioning often shows up as
    # palette transfer first)
    def hist(s):
        h = np.zeros(len(bv.block_index_to_pair) + 1)
        occ = s.occupied_mask
        ids = s.block_ids[occ]
        datas = s.block_data[occ]
        lut = bv.block_token_to_id
        for bid, bdata in zip(ids.tolist(), datas.tolist()):
            h[lut.get(_token_for(int(bid), int(bdata)), len(lut))] += 1
        return h / max(1, h.sum())

    H_s = np.stack([hist(s) for s in samples])
    H_t = np.stack([hist(t) for t in targets])
    def cos(a, b):
        return (a * b).sum(axis=1) / np.maximum(
            1e-9, np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    pal_pair = cos(H_s, H_t)
    pal_rand = cos(H_s, H_t[perm])

    comps = [n_components(s) for s in samples]
    stops = [int(EOS in row) for row in toks.cpu().numpy()]
    summary = {
        "run": str(run), "ckpt": args.ckpt, "cond": cfg["cond"],
        "cfg_scale": args.cfg, "n": len(samples),
        "pair_iou_mean": round(float(pair_iou.mean()), 4),
        "shuffled_iou_mean": round(float(rand_iou.mean()), 4),
        "palette_sim_paired": round(float(pal_pair.mean()), 4),
        "palette_sim_shuffled": round(float(pal_rand.mean()), 4),
        "eos_rate": round(float(np.mean(stops)), 3),
        "connected_rate": round(float(np.mean([c == 1 for c in comps])), 3),
        "components_median": int(np.median(comps)),
        "blocks_median": int(np.median([int(s.occupied_mask.sum()) for s in samples])),
    }
    print(json.dumps(summary, indent=2))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{cfg['cond']}_cfg{args.cfg:g}"
    # interleave target/sample rows for visual comparison
    paired = []
    for t, s in zip(targets, samples):
        paired.extend([t, s])
    save_grid(paired, str(out / f"{tag}_target_vs_sample.png"), cols=8)
    with open(out / f"{tag}_eval.json", "w") as f:
        json.dump(summary, f, indent=2)
    if cfg["cond"] == "text":
        caps = json.load(open("data/minecraft/labels/houses_32_captions.json"))
        for k, i in enumerate(val_idx[:8]):
            print(f"[{k}] {caps[f'h{i:05d}'][0][:90]}")
    print(f"grid -> {out / f'{tag}_target_vs_sample.png'} (rows alternate target, sample)")


if __name__ == "__main__":
    main()
