"""Loaders for external public corpora (pool-pretraining data; research.md §C.0).

* **3D-Craft / CraftAssist houses** (FAIR, ICCV 2019 "Order-Aware Generative
  Modeling Using the 3D-Craft Dataset"): 2,586 crowdsourced human-built houses.
  ``data/3d_craft/houses/<name>/schematic.npy`` is a ``(Y, Z, X, 2)`` uint8 array
  (last dim = block_id, block_data; verified: vertical axis has the smallest
  median). ``placed.json`` holds the human **build-order** action stream
  ``[tick, player?, [x, y, z], [block_id, data], 'P'|'B']`` — usable as a natural
  connected ordering for the AR track (VoxelCNN's key signal).
* **text2mc Kaggle dump** (shauncomino/minecraft-builds-dataset): ~11k builds as
  ``.h5`` voxel arrays (``load_text2mc``), **plus ~28k raw Sponge ``.schem``** the
  author never converted (``load_text2mc_schem``, decoded via ``utils/schem.py``) —
  together the dataset's advertised ~40k builds.

All load into the standard ``Structure`` (XYZ indexing, Y vertical), so all
curation, canonicalization, and tokenizers apply unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from blockgen.utils.data import Structure

REPO = Path(__file__).resolve().parents[2]
DEFAULT_3DCRAFT = REPO / "data" / "minecraft" / "3d_craft" / "houses"


def load_3dcraft(root: Path | str = DEFAULT_3DCRAFT, min_blocks: int = 60,
                 max_dim: Optional[int] = None, limit: Optional[int] = None,
                 verbose: bool = True) -> List[Structure]:
    """Load 3D-Craft houses as ``Structure``s ((Y,Z,X,2) -> XYZ transpose)."""
    root = Path(root)
    out: List[Structure] = []
    dirs = sorted(d for d in root.iterdir() if (d / "schematic.npy").exists())
    for d in dirs:
        try:
            sch = np.load(d / "schematic.npy")            # (Y, Z, X, 2)
        except Exception:  # noqa: BLE001
            continue
        if sch.ndim != 4 or sch.shape[-1] != 2:
            continue
        ids = sch[..., 0].transpose(2, 0, 1)              # -> (X, Y, Z)
        data = sch[..., 1].transpose(2, 0, 1)
        occ = int((ids > 0).sum())
        if occ < min_blocks:
            continue
        if max_dim is not None and max(ids.shape) > max_dim:
            continue
        out.append(Structure(block_ids=ids.astype(np.int32),
                             block_data=data.astype(np.int32),
                             source_path=str(d),
                             metadata={"corpus": "3dcraft", "category": "house"}))
        if limit and len(out) >= limit:
            break
    if verbose:
        print(f"[3dcraft] loaded {len(out)}/{len(dirs)} houses "
              f"(min_blocks={min_blocks}, max_dim={max_dim})", flush=True)
    return out


def load_3dcraft_order(house_dir: Path | str) -> List[Tuple[Tuple[int, int, int], Tuple[int, int]]]:
    """Human build order for one house: [(world (x,y,z), (block_id, data)), ...].

    Net placements only ('P' actions, minus later breaks of the same voxel) in
    original chronological order. World coords are NOT the schematic-local coords;
    subtract the min corner to localize.
    """
    actions = json.loads((Path(house_dir) / "placed.json").read_text())
    placed: dict = {}
    order: List[Tuple[int, int, int]] = []
    for act in actions:
        try:
            _tick, _pid, xyz, block, kind = act
        except ValueError:
            continue
        key = tuple(int(c) for c in xyz)
        if kind == "P":
            if key not in placed:
                order.append(key)
            placed[key] = (int(block[0]), int(block[1]))
        elif kind == "B" and key in placed:
            placed.pop(key)
            if key in order:
                order.remove(key)
    return [(k, placed[k]) for k in order if k in placed]


# text2mc token ids that mean "empty" (from tok2block.json):
# 102=minecraft:air, 576=cave_air, 3352=void_air
_TEXT2MC_AIR = (102, 576, 3352)


def load_text2mc(root: Path | str, min_blocks: int = 60, max_dim: Optional[int] = None,
                 limit: Optional[int] = None, verbose: bool = True) -> List[Structure]:
    """Load the text2mc Kaggle dump as ``Structure``s.

    Each ``processed_builds/*.h5`` holds one build as a 3D **uint16 token** array
    over the corpus's own 3,717-entry block-state vocabulary (``tok2block.json``;
    modern flattened states, NOT legacy (id, data)). Air/cave_air/void_air map to
    0; every other voxel stores ``token + 1`` in ``block_ids`` (shift avoids the
    token-0 collision) with ``block_data = 0``. Good for occupancy/shape
    pretraining; material use needs a tok2block remap to our families.
    ``processed_build_dataframe.csv`` carries PlanetMinecraft PAGE_URL + TAGS —
    same source site as our schematics crawl, so dedup before pooling.
    """
    import h5py
    root = Path(root)
    files = sorted(root.rglob("*.h5"))
    out: List[Structure] = []
    for f in files:
        try:
            with h5py.File(f, "r") as h:
                key = list(h.keys())[0]
                arr = np.asarray(h[key]).astype(np.int32)
        except Exception:  # noqa: BLE001
            continue
        if arr.ndim != 3:
            continue
        occ_mask = ~np.isin(arr, _TEXT2MC_AIR)
        occ = int(occ_mask.sum())
        if occ < min_blocks:
            continue
        if max_dim is not None and max(arr.shape) > max_dim:
            continue
        ids = np.where(occ_mask, arr + 1, 0)
        out.append(Structure(block_ids=ids,
                             block_data=np.zeros_like(ids, dtype=np.int32),
                             source_path=str(f),
                             metadata={"corpus": "text2mc"}))
        if limit and len(out) >= limit:
            break
    if verbose:
        print(f"[text2mc] loaded {len(out)}/{len(files)} builds "
              f"(min_blocks={min_blocks}, max_dim={max_dim})", flush=True)
    return out


def load_text2mc_schem(root: Path | str, min_blocks: int = 60,
                       max_dim: Optional[int] = None, limit: Optional[int] = None,
                       max_voxels: int = 8_000_000, max_bytes: Optional[int] = None,
                       verbose: bool = True) -> List[Structure]:
    """Load text2mc's raw Sponge ``.schem`` builds as legacy-vocab ``Structure``s.

    The dataset ships ~11k processed ``.h5`` (see :func:`load_text2mc`) **and**
    ~28k raw ``.schem`` its author never converted — the difference between the
    11k advertised as "processed" and the ~40k advertised total (``README.txt``).
    We decode the Sponge palette + varint ``BlockData`` (:mod:`blockgen.utils.schem`)
    and remap block-state names straight to our shared legacy ``(id, data)`` space
    via ``block_remap.remap_name``, so these pool directly with 3D-Craft/GrabCraft
    — no ``token + 1`` shift and no tok2block detour. ``processed_build_dataframe.csv``
    carries the PlanetMinecraft PAGE_URL + TAGS per build; dedupe against the
    ``.h5`` set (same source pages) before pooling.
    """
    from blockgen.utils.schem import schem_to_legacy
    root = Path(root)
    files = sorted(root.rglob("*.schem"))
    out: List[Structure] = []
    n_read = 0
    for f in files:
        res = schem_to_legacy(f, max_voxels=max_voxels, max_bytes=max_bytes)
        if res is None:
            continue
        n_read += 1
        ids, data = res
        occ = int((ids != 0).sum())
        if occ < min_blocks:
            continue
        if max_dim is not None and max(ids.shape) > max_dim:
            continue
        out.append(Structure(block_ids=ids, block_data=data, source_path=str(f),
                             metadata={"corpus": "text2mc_schem"}))
        if limit and len(out) >= limit:
            break
    if verbose:
        print(f"[text2mc_schem] loaded {len(out)}/{len(files)} builds "
              f"({n_read} decoded ok; min_blocks={min_blocks}, max_dim={max_dim})",
              flush=True)
    return out
