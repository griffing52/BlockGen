"""Convert Dream-Cubed raw chunks into a bench arm (structure-cache `.npz`).

Input: the `.pt` written by `scripts/ext/dreamcubed_generate.py`, i.e. N 32^3
chunks over Dream-Cubed's 117-entry compressed vocabulary (1.12 Bukkit names,
plus material-free SLAB_* / STAIR_* shape tokens).

Two arms are written, because a terrain chunk is not a build and the choice of
how to turn one into a build is ours, not the model's:

* ``dreamcubed_32``     -- **structure extraction**. Keep the constructed blocks
  (BUILT below; no terrain, ores, liquids, foliage, crops or paths; sandstone
  only where no TERRAIN block lies above it in its column), take the
  26-connected component with the most non-log blocks, keep only that
  component, crop to its bounding box. Chunks whose
  component has < ``--min-blocks`` blocks are dropped and counted (the yield is
  reported, it is part of the result).
* ``dreamcubed_raw_32`` -- the whole chunk, air-cropped, no filtering. Shows what
  the model emits before our extraction; mostly terrain.

No rescaling: a chunk is already 32^3, so every crop fits max_dim 32 as is.

Vocabulary: each name maps to a legacy (block_id, data) pair (NAME_TO_LEGACY).
Lossy where Dream-Cubed is lossy: SLAB_* carries no material (-> stone slab 44),
STAIR_* carries no material (-> oak stairs 53); facing PX/NX/PY/NY -> legacy
data 0/1/2/3 (east/west/south/north, assuming Dream-Cubed's second horizontal
axis is z), TOP -> +4. Door upper halves get data 8.

    .venv/bin/python scripts/ext/dreamcubed_to_npz.py \\
        --raw outputs/run_..._dreamcubed_village_md4p4/raw_village.pt \\
        --run-dir outputs/run_..._dreamcubed_village_md4p4
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

NAME_TO_LEGACY = {
    "AIR": (0, 0), "STONE": (1, 0), "GRASS": (2, 0), "DIRT": (3, 0), "COBBLESTONE": (4, 0),
    "PLANKS": (5, 0), "BEDROCK": (7, 0), "FLOWING_WATER": (8, 0), "WATER": (9, 0),
    "FLOWING_LAVA": (10, 0), "LAVA": (11, 0), "SAND": (12, 0), "GRAVEL": (13, 0),
    "GOLD_ORE": (14, 0), "IRON_ORE": (15, 0), "COAL_ORE": (16, 0), "LOG": (17, 0),
    "LEAVES": (18, 0), "SPONGE": (19, 0), "LAPIS_ORE": (21, 0), "DISPENSER": (23, 0),
    "SANDSTONE": (24, 0), "BED": (26, 0), "STICKY_PISTON": (29, 0), "WEB": (30, 0),
    "TALLGRASS": (31, 1), "DEADBUSH": (32, 0), "WOOL": (35, 0), "YELLOW_FLOWER": (37, 0),
    "RED_FLOWER": (38, 0), "BROWN_MUSHROOM": (39, 0), "GOLD_BLOCK": (41, 0),
    "DOUBLE_STONE_SLAB": (43, 0), "TNT": (46, 0), "BOOKSHELF": (47, 0),
    "MOSSY_COBBLESTONE": (48, 0), "OBSIDIAN": (49, 0), "TORCH": (50, 5),
    "MOB_SPAWNER": (52, 0), "CHEST": (54, 2), "REDSTONE_WIRE": (55, 0), "DIAMOND_ORE": (56, 0),
    "CRAFTING_TABLE": (58, 0), "WHEAT": (59, 7), "FARMLAND": (60, 7), "FURNACE": (61, 2),
    "WOODEN_DOOR": (64, 0), "LADDER": (65, 2), "RAIL": (66, 0), "WALL_SIGN": (68, 2),
    "LEVER": (69, 5), "STONE_PRESSURE_PLATE": (70, 0), "IRON_DOOR": (71, 0),
    "WOODEN_PRESSURE_PLATE": (72, 0), "REDSTONE_ORE": (73, 0), "REDSTONE_TORCH": (76, 5),
    "STONE_BUTTON": (77, 0), "SNOW_LAYER": (78, 0), "ICE": (79, 0), "SNOW": (80, 0),
    "CACTUS": (81, 0), "CLAY": (82, 0), "REEDS": (83, 0), "FENCE": (85, 0), "PUMPKIN": (86, 0),
    "UNPOWERED_REPEATER": (93, 0), "TRAPDOOR": (96, 0), "MONSTER_EGG": (97, 0),
    "STONEBRICK": (98, 0), "BROWN_MUSHROOM_BLOCK": (99, 14), "RED_MUSHROOM_BLOCK": (100, 14),
    "IRON_BARS": (101, 0), "GLASS_PANE": (102, 0), "MELON_BLOCK": (103, 0), "VINE": (106, 0),
    "MYCELIUM": (110, 0), "WATERLILY": (111, 0), "BREWING_STAND": (117, 0), "CAULDRON": (118, 0),
    "END_PORTAL_FRAME": (120, 0), "COCOA": (127, 0), "EMERALD_ORE": (129, 0),
    "TRIPWIRE_HOOK": (131, 0), "TRIPWIRE": (132, 0), "COBBLESTONE_WALL": (139, 0),
    "FLOWER_POT": (140, 0), "CARROTS": (141, 7), "POTATOES": (142, 7), "ANVIL": (145, 0),
    "STAINED_HARDENED_CLAY": (159, 0), "LEAVES2": (161, 0), "LOG2": (162, 0),
    "PRISMARINE": (168, 0), "SEA_LANTERN": (169, 0), "CARPET": (171, 0), "HARDENED_CLAY": (172, 0),
    "PACKED_ICE": (174, 0), "DOUBLE_PLANT": (175, 2), "SPRUCE_FENCE": (188, 0),
    "BIRCH_FENCE": (189, 0), "DARK_OAK_FENCE": (191, 0), "ACACIA_FENCE": (192, 0),
    "SPRUCE_DOOR": (193, 0), "ACACIA_DOOR": (196, 0), "BEETROOTS": (207, 3), "GRASS_PATH": (208, 0),
    "BONE_BLOCK": (216, 0),
    "SLAB_BOTTOM": (44, 0), "SLAB_TOP": (44, 8),
    "STAIR_STRAIGHT_BOTTOM_PX": (53, 0), "STAIR_STRAIGHT_BOTTOM_NX": (53, 1),
    "STAIR_STRAIGHT_BOTTOM_PY": (53, 2), "STAIR_STRAIGHT_BOTTOM_NY": (53, 3),
    "STAIR_STRAIGHT_TOP_PX": (53, 4), "STAIR_STRAIGHT_TOP_NX": (53, 5),
    "STAIR_STRAIGHT_TOP_PY": (53, 6), "STAIR_STRAIGHT_TOP_NY": (53, 7),
}
DOORS = {64, 71, 193, 196}

# Constructed blocks. Everything else is terrain, ore, liquid, foliage, crop,
# path or mob/dungeon block. LOG/LOG2 count because 1.12 village houses use log
# pillars. Components are ranked by non-log blocks (see extract), so a bare
# tree is never chosen; a trunk survives only if it touches the house.
BUILT = {
    "PLANKS", "COBBLESTONE", "COBBLESTONE_WALL", "MOSSY_COBBLESTONE", "STONEBRICK",
    "DOUBLE_STONE_SLAB", "SANDSTONE", "LOG", "LOG2", "GLASS_PANE", "IRON_BARS",
    "FENCE", "SPRUCE_FENCE", "BIRCH_FENCE", "DARK_OAK_FENCE", "ACACIA_FENCE",
    "WOODEN_DOOR", "SPRUCE_DOOR", "ACACIA_DOOR", "IRON_DOOR", "TRAPDOOR", "LADDER",
    "TORCH", "WALL_SIGN", "BED", "BOOKSHELF", "CHEST", "CRAFTING_TABLE", "FURNACE",
    "CAULDRON", "BREWING_STAND", "ANVIL", "CARPET", "WOOL", "FLOWER_POT",
    "WOODEN_PRESSURE_PLATE", "STONE_PRESSURE_PLATE", "STONE_BUTTON", "LEVER",
    "HARDENED_CLAY", "STAINED_HARDENED_CLAY", "GOLD_BLOCK", "SEA_LANTERN", "PRISMARINE",
    "SLAB_BOTTOM", "SLAB_TOP",
    "STAIR_STRAIGHT_BOTTOM_PX", "STAIR_STRAIGHT_BOTTOM_NX", "STAIR_STRAIGHT_BOTTOM_PY",
    "STAIR_STRAIGHT_BOTTOM_NY", "STAIR_STRAIGHT_TOP_PX", "STAIR_STRAIGHT_TOP_NX",
    "STAIR_STRAIGHT_TOP_PY", "STAIR_STRAIGHT_TOP_NY",
}

TERRAIN = {
    "STONE", "GRASS", "DIRT", "SAND", "GRAVEL", "CLAY", "BEDROCK", "MYCELIUM", "SNOW",
    "ICE", "PACKED_ICE", "COAL_ORE", "IRON_ORE", "GOLD_ORE", "DIAMOND_ORE", "LAPIS_ORE",
    "REDSTONE_ORE", "EMERALD_ORE", "MONSTER_EGG",
}


def to_legacy(idx: np.ndarray, vocab: list) -> tuple:
    lut_id = np.array([NAME_TO_LEGACY[n][0] for n in vocab], np.int16)
    lut_dt = np.array([NAME_TO_LEGACY[n][1] for n in vocab], np.int16)
    ids, data = lut_id[idx], lut_dt[idx].copy()
    door = np.isin(ids, list(DOORS))
    upper = door[:, 1:, :] & (ids[:, 1:, :] == ids[:, :-1, :])
    data[:, 1:, :][upper] = 8
    return ids, data


def extract(idx: np.ndarray, vocab: list, min_blocks: int):
    from scipy import ndimage
    built_lut = np.array([n in BUILT for n in vocab])
    mask = built_lut[idx]
    # SANDSTONE is both desert-village wall material and desert sub-surface
    # terrain. Keep a sandstone voxel only if nothing natural (TERRAIN) lies
    # above it in its column: walls stand on the ground, bedrock sandstone sits
    # under sand. Axis 1 is y (up).
    terrain_lut = np.array([n in TERRAIN for n in vocab])
    terr = terrain_lut[idx]
    terrain_above = np.flip(np.cumsum(np.flip(terr, 1), 1), 1) - terr > 0
    ss = np.array([n == "SANDSTONE" for n in vocab])[idx]
    mask &= ~(ss & terrain_above)
    if not mask.any():
        return None
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3), bool))
    sizes = np.bincount(lab.ravel())[1:]
    # Rank components by their NON-log blocks, so a bare tree trunk (logs only)
    # never wins over a house; a component with no non-log block is rejected.
    log = np.array([n in ("LOG", "LOG2") for n in vocab])[idx]
    nonlog = np.bincount(lab[mask & ~log].ravel(), minlength=n + 1)[1:]
    k = int(np.argmax(nonlog)) + 1
    if nonlog[k - 1] == 0 or sizes[k - 1] < min_blocks:
        return None
    comp = lab == k
    xs, ys, zs = np.where(comp)
    sl = (slice(xs.min(), xs.max() + 1), slice(ys.min(), ys.max() + 1), slice(zs.min(), zs.max() + 1))
    return comp[sl], sl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default="outputs/bench_arms")
    ap.add_argument("--min-blocks", type=int, default=80, help="houses_32 curation uses 80")
    ap.add_argument("--name", default="dreamcubed")
    args = ap.parse_args()

    import torch
    from blockgen.curation.houses import save_house_cache
    from blockgen.utils.data import Structure

    raw = torch.load(args.raw, map_location="cpu", weights_only=False)
    idx_all = raw["indices"].numpy().astype(np.int64)
    vocab = raw["vocab"]
    missing = [n for n in vocab if n not in NAME_TO_LEGACY]
    if missing:
        raise SystemExit(f"unmapped Dream-Cubed names: {missing}")
    meta = raw["meta"]
    N = len(idx_all)

    extracted, whole, dropped = [], [], []
    name_counts, kept_counts = Counter(), Counter()
    for i, idx in enumerate(idx_all):
        ids, data = to_legacy(idx[None], vocab)
        ids, data = ids[0], data[0]
        u, c = np.unique(idx, return_counts=True)
        name_counts.update({vocab[a]: int(b) for a, b in zip(u, c)})
        src = f"dreamcubed:{meta['model']}:{raw['biome']}:seed{meta['seed']}:{i}"
        md = {"corpus": "dreamcubed", "category": raw["biome"], "title": f"dreamcubed_{i:04d}"}
        s = Structure(ids, data, source_path=src, metadata=dict(md)).crop_to_non_air() \
            if (ids != 0).any() else None
        if s is not None:
            whole.append(s)
        r = extract(idx, vocab, args.min_blocks)
        if r is None:
            dropped.append(i)
            continue
        comp, sl = r
        e_ids = np.where(comp, ids[sl], 0)
        e_dat = np.where(comp, data[sl], 0)
        kept_counts.update(Counter(vocab[a] for a in idx[sl][comp].ravel().tolist()))
        extracted.append(Structure(e_ids, e_dat, source_path=src, metadata=dict(md)))

    occ = [int(s.occupied_mask.sum()) for s in extracted]
    dims = np.array([s.shape for s in extracted]) if extracted else np.zeros((0, 3))
    report_common = {
        "generator": "Dream-Cubed (Merino, Earle, Iwai, Togelius, Cetin 2026; arXiv:2604.22847)",
        "repo": "https://github.com/SakanaAI/DreamCubed @ 84d9274 (no LICENSE file in repo)",
        "weights": f"https://huggingface.co/dream-cubed/dream-cubed-models/{meta['model']} (Apache-2.0)",
        "mappings": f"https://huggingface.co/SakanaAI/DreamCubed2M boosted_dataset/{meta['mappings']} (Apache-2.0)",
        "license": "weights + mappings Apache-2.0; code repo has no license file; outputs of an "
                   "Apache-2.0 model trained on procedural Minecraft worlds (Minecraft EULA applies to game assets)",
        "conditioning": f"biome class '{raw['biome']}' (no text; Dream-Cubed is not text-conditioned)",
        "prompts": [],
        "seed": meta["seed"], "reverse_steps": meta["reverse_steps"], "cond_scale": meta["cond_scale"],
        "n_generated": N, "train_split": "external (Dream-Cubed natural+village-boosted data; "
                                         "no GrabCraft/3D-Craft/text2mc)",
        "throughput_s_per_chunk": meta["elapsed_s"] / max(N, 1), "peak_gib": meta["peak_gib"],
        "vocab_map": "scripts/ext/dreamcubed_to_npz.py NAME_TO_LEGACY; slabs->stone slab 44, "
                     "stairs->oak stairs 53 (Dream-Cubed tokens carry no material)",
        "rescaling": "none (chunks are 32^3)",
    }
    rep_e = {**report_common, "model": f"{args.name}_32", "conversion": "structure extraction: BUILT blocks, "
             f"largest 26-connected component, bbox crop, min_blocks={args.min_blocks}",
             "n": len(extracted), "n_dropped": len(dropped), "yield": len(extracted) / max(N, 1),
             "median_blocks": int(np.median(occ)) if occ else 0,
             "median_dims": np.median(dims, 0).tolist() if len(dims) else []}
    rep_w = {**report_common, "model": f"{args.name}_raw_32", "conversion": "whole chunk, air-cropped",
             "n": len(whole)}
    p_e = save_house_cache(extracted, 32, cache_dir=args.out_dir, name=args.name, report=rep_e)
    p_w = save_house_cache(whole, 32, cache_dir=args.out_dir, name=f"{args.name}_raw", report=rep_w)

    run = Path(args.run_dir)
    stats = {"extracted": rep_e, "raw": rep_w, "dropped_indices": dropped,
             "name_freq_all": dict(name_counts.most_common()),
             "name_freq_extracted": dict(kept_counts.most_common())}
    (run / "conversion.json").write_text(json.dumps(stats, indent=1))
    try:
        from blockgen.renderer.grid import save_grid
        rng = np.random.default_rng(0)
        pick = [extracted[i] for i in sorted(rng.permutation(len(extracted))[:48])]
        save_grid(pick, str(run / "samples.png"), cols=8, tile_px=192)
        pick_w = [whole[i] for i in sorted(rng.permutation(len(whole))[:24])]
        save_grid(pick_w, str(run / "samples_raw_chunks.png"), cols=8, tile_px=192)
        import shutil
        shutil.copy(run / "samples.png", Path(args.out_dir) / f"{args.name}_32_samples.png")
    except Exception as exc:  # noqa: BLE001
        print(f"[render] skipped ({type(exc).__name__}: {exc})")
    print(f"[convert] {N} chunks -> {len(extracted)} extracted ({len(dropped)} dropped), "
          f"median blocks {rep_e['median_blocks']}, median dims {rep_e['median_dims']}")
    print(f"arms:\n  ext/{args.name}:{p_e}\n  ext/{args.name}_raw:{p_w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
