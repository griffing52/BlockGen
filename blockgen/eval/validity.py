"""Connectivity / validity gate for generated structures.

Our headline visible-quality problem is that samples *fragment*: the AR and graph
tracks emit voxels with no connectivity constraint, so a "house" comes out as a
house-shaped cloud of disconnected debris (validity_rate ~ 0.06-0.3). This module
gives two tractable levers, short of full LegoGPT-style rollback:

1. ``largest_component`` — keep only the biggest 6-connected occupied component of
   a structure (a cheap *repair*). Turns a debris cloud into its coherent core.
2. ``gated_sample`` — oversample, repair each to its largest component, and reject
   the ones whose core is too small, resampling to top up to ``num_samples``.

Both operate on the occupied mask (block-type agnostic), so they compose with any
generator. Reported alongside the raw (ungated) samples so the gate's effect on
validity / novelty / diversity is an explicit ablation.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np

from blockgen.utils.data import Structure

_NEIGHBORS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def _components(occ: np.ndarray) -> List[List[tuple]]:
    """Return 6-connected components of a boolean occupancy array, largest first."""
    coords = [tuple(c) for c in np.argwhere(occ).tolist()]
    occ_set = set(coords)
    seen: set = set()
    comps: List[List[tuple]] = []
    for start in coords:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = [start]
        while stack:
            x, y, z = stack.pop()
            for dx, dy, dz in _NEIGHBORS:
                p = (x + dx, y + dy, z + dz)
                if p in occ_set and p not in seen:
                    seen.add(p)
                    stack.append(p)
                    comp.append(p)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def n_components(s: Structure) -> int:
    return len(_components(s.crop_to_non_air().occupied_mask))


def largest_component(s: Structure) -> Structure:
    """Keep only the largest 6-connected occupied component (cropped)."""
    c = s.crop_to_non_air()
    occ = c.occupied_mask
    comps = _components(occ)
    if len(comps) <= 1:
        return c
    keep = set(comps[0])
    bi = c.block_ids.copy()
    bd = c.block_data.copy()
    for x, y, z in np.argwhere(occ).tolist():
        if (x, y, z) not in keep:
            bi[x, y, z] = c.air_block_id
            bd[x, y, z] = 0
    return Structure(block_ids=bi, block_data=bd, source_path=s.source_path,
                     air_block_id=c.air_block_id, metadata=dict(s.metadata)).crop_to_non_air()


def repair_lcc(structs: List[Structure]) -> List[Structure]:
    """Repair each structure to its largest connected component."""
    return [largest_component(s) for s in structs]


def gated_sample(sample_fn: Callable[[int], List[Structure]], num_samples: int,
                 min_blocks: int = 12, max_rounds: int = 6) -> List[Structure]:
    """Draw ``num_samples`` structures whose largest-component core has >= ``min_blocks``.

    ``sample_fn(n)`` returns a list of n raw samples. Each is repaired to its LCC;
    cores below ``min_blocks`` are rejected and we resample (up to ``max_rounds``).
    Whatever survives after the rounds is returned (padded with best-effort if short).
    """
    kept: List[Structure] = []
    pool: List[Structure] = []
    for _ in range(max_rounds):
        if len(kept) >= num_samples:
            break
        raw = sample_fn(num_samples)
        for s in raw:
            core = largest_component(s)
            occ = int(core.occupied_mask.sum())
            if occ >= min_blocks:
                kept.append(core)
            else:
                pool.append(core)
    if len(kept) < num_samples:
        pool.sort(key=lambda s: int(s.occupied_mask.sum()), reverse=True)
        kept.extend(pool[: num_samples - len(kept)])
    return kept[:num_samples]
