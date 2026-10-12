"""Corruption and invariance probes: structures whose quality we already know.

A metric is only trustworthy where it ranks damage we have independently
measured. F16 established that canon-16 decimation deletes 86.5% of blocks and
removes whole roofs, so a metric that cannot separate canon-16 from held-out
real is not measuring build quality, whatever else it correlates with.

`shuffle_materials` is a deliberate **copy** of the function in
`scripts/validate_perceptual.py` rather than an import. That script and its
recorded output in `outputs/analysis/perceptual_validation.json` are a published
result; importing from it would couple a frozen artifact to a module under
active development. `tests/eval/test_probes.py` pins the two implementations to
identical output at a fixed seed.

Note which probes each tier can see. Material shuffle permutes *where* blocks
sit while leaving the multiset untouched, so every voxel palette statistic is
blind to it by construction -- it is a negative control for the FAST tier and a
real probe only once rendering enters. `monochrome` is the converse.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Sequence

import numpy as np

from blockgen.utils.data import Structure

COBBLESTONE = (4, 0)


def canon(structures: Sequence[Structure], dim: int) -> List[Structure]:
    """Decimate to a maximum dimension (delegates to `Structure.downsample`)."""
    return [s.crop_to_non_air().downsample(dim) for s in structures]


def shuffle_materials(s: Structure, rng: np.random.Generator) -> Structure:
    """Keep the shape, permute which block type sits where. Palette destroyed.

    Copied verbatim from `scripts/validate_perceptual.py`; see module docstring.
    """
    c = s.crop_to_non_air()
    occ = c.block_ids != c.air_block_id
    bi, bd = c.block_ids.copy(), c.block_data.copy()
    idx = np.argwhere(occ)
    vals = [(bi[x, y, z], bd[x, y, z]) for x, y, z in idx]
    rng.shuffle(vals)
    for (x, y, z), (a, b) in zip(idx, vals):
        bi[x, y, z], bd[x, y, z] = a, b
    return Structure(block_ids=bi, block_data=bd)


def monochrome(s: Structure, block: tuple = COBBLESTONE) -> Structure:
    """Keep the shape, collapse every block to one material. Palette collapsed.

    The converse of `shuffle_materials`: geometry untouched, multiset destroyed.
    Any palette metric that cannot separate this from real is not working.
    """
    c = s.crop_to_non_air()
    occ = c.occupied_mask
    bi = np.where(occ, block[0], c.air_block_id).astype(np.int32)
    bd = np.where(occ, block[1], 0).astype(np.int32)
    return Structure(block_ids=bi, block_data=bd)


def block_noise(s: Structure, p: float, rng: np.random.Generator) -> Structure:
    """Retype a fraction `p` of occupied blocks to another block already present."""
    c = s.crop_to_non_air()
    occ = c.occupied_mask
    bi, bd = c.block_ids.copy(), c.block_data.copy()
    idx = np.argwhere(occ)
    if len(idx) == 0:
        return c
    present = np.unique(np.stack([bi[occ], bd[occ]], 1), axis=0)
    hit = rng.random(len(idx)) < p
    for (x, y, z), flip in zip(idx, hit):
        if flip:
            a, b = present[rng.integers(len(present))]
            bi[x, y, z], bd[x, y, z] = int(a), int(b)
    return Structure(block_ids=bi, block_data=bd)


def chunk_delete(s: Structure, frac: float, rng: np.random.Generator) -> Structure:
    """Delete a contiguous *interior* slab, keeping the bounding box intact.

    The slab is strictly interior on purpose. `render_structure` fits the camera
    to the bounding box, so a build that merely got smaller renders almost
    identically -- deleting an end slab is invisible to every image-space metric
    by construction, and the first version of this probe did exactly that, which
    made the whole ladder's deletion gate unpassable. Cutting through the middle
    keeps the silhouette and removes visible mass, which is the damage we
    actually want to detect.

    Falls back to an end slab only when the structure is too thin to have an
    interior along the chosen axis.
    """
    c = s.crop_to_non_air()
    bi, bd = c.block_ids.copy(), c.block_data.copy()
    # Prefer an axis with room for an interior cut.
    axes = [a for a in range(3) if c.shape[a] >= 5] or [int(rng.integers(3))]
    axis = int(axes[rng.integers(len(axes))])
    extent = c.shape[axis]
    width = max(1, int(round(frac * extent)))
    width = min(width, max(1, extent - 2))
    lo, hi = 1, max(1, extent - width - 1)
    start = int(rng.integers(lo, hi + 1)) if hi >= lo else 0
    sl: List[slice] = [slice(None)] * 3
    sl[axis] = slice(start, start + width)
    bi[tuple(sl)] = c.air_block_id
    bd[tuple(sl)] = 0
    return Structure(block_ids=bi, block_data=bd)


def rotate_y(s: Structure, k: int = 1) -> Structure:
    """Rotate k*90 degrees about the vertical axis. Quality is unchanged."""
    c = s.crop_to_non_air()
    return Structure(block_ids=np.rot90(c.block_ids, k, axes=(0, 2)).copy(),
                     block_data=np.rot90(c.block_data, k, axes=(0, 2)).copy())


def mirror_x(s: Structure) -> Structure:
    """Mirror along x. Quality is unchanged."""
    c = s.crop_to_non_air()
    return Structure(block_ids=c.block_ids[::-1].copy(),
                     block_data=c.block_data[::-1].copy())


def probe_suite(structures: Sequence[Structure],
                rng: np.random.Generator | None = None) -> Dict[str, List[Structure]]:
    """The standard ladder rungs, damage-ordered where damage is comparable."""
    rng = rng or np.random.default_rng(0)
    pool = [s.crop_to_non_air() for s in structures]
    return {
        "canon16": canon(pool, 16),
        "canon8": canon(pool, 8),
        "noise_1": [block_noise(s, 0.01, rng) for s in pool],
        "noise_5": [block_noise(s, 0.05, rng) for s in pool],
        "noise_10": [block_noise(s, 0.10, rng) for s in pool],
        # Two deletion sizes. The gate uses the larger one -- "can this metric
        # see a chunk missing at all" -- while the smaller is recorded ungated
        # as a sensitivity datapoint. Measured at n=64, no image-space metric
        # resolves a 20% interior cut at 3 sigma (KID moves 0.100 -> 0.123
        # against a 0.027 bar), which is worth knowing but is not grounds for
        # rejecting the metric.
        "chunk_delete_20": [chunk_delete(s, 0.20, rng) for s in pool],
        "chunk_delete_40": [chunk_delete(s, 0.40, rng) for s in pool],
        # Structural rungs. `solidify` is the render tier's measured blind
        # spot; the occupancy-noise triple is the geometric counterpart of the
        # material-noise triple above, and each tier is gated on the one it is
        # built to see (see `geometry` module docstring).
        "solidify": [solidify(s) for s in pool],
        # `hollow` is `solidify` run backwards -- both leave the rendered
        # surface intact, one adds every invisible block and the other deletes
        # every one. Recorded as a rung so the render tier's blind spot is
        # measured in both directions rather than once.
        "hollow": [hollow(s) for s in pool],
        "occ_noise_1": [occupancy_noise(s, 0.01, rng) for s in pool],
        "occ_noise_5": [occupancy_noise(s, 0.05, rng) for s in pool],
        "occ_noise_10": [occupancy_noise(s, 0.10, rng) for s in pool],
        "jitter_columns": [jitter_columns(s, 1, rng) for s in pool],
        "material_shuffle": [shuffle_materials(s, rng) for s in pool],
        "monochrome": [monochrome(s) for s in pool],
        "rot90_1": [rotate_y(s, 1) for s in pool],
        "rot90_2": [rotate_y(s, 2) for s in pool],
        "mirror_x": [mirror_x(s) for s in pool],
    }


def exterior_air(occ: np.ndarray) -> np.ndarray:
    """Air cells 6-connected to the outside of the build's bounding box.

    The region a camera is standing in, and therefore the region that decides
    which block faces exist in a render: `renderer.textured._exposed_masks`
    emits a quad for every face whose neighbour is air, and of those only the
    ones bordering *this* set can be reached by a ray from outside. Sealed
    pockets are what `solidify` fills; this is their complement.

    Face (6-) connectivity rather than 26-, matching `solidify`'s
    `binary_fill_holes` structure, because a cube mesh leaks no light through an
    edge- or corner-only opening: the aperture has zero area. Using 26 here
    would call diagonal chinks "outside" and make the two probes inverses of
    different sets.
    """
    from scipy import ndimage

    padded = np.pad(np.asarray(occ, dtype=bool), 1)
    lab, _ = ndimage.label(~padded,
                           structure=ndimage.generate_binary_structure(3, 1))
    return (lab == lab[0, 0, 0])[1:-1, 1:-1, 1:-1]


def hollow(s: Structure) -> Structure:
    """Erode to the visible surface layer: keep only blocks a camera can see.

    The converse of `solidify`, and the failure mode the wider 3D-generation
    field actually ships: surface-fitted mesh and implicit generators emit
    shells with nothing inside them. `solidify` adds every block a camera cannot
    see; this deletes every block a camera cannot see. If a render-and-embed
    metric is blind to interior *mass* it must be equally blind to the absence
    of it, and a probe in each direction is much harder to dismiss as an artifact
    of one construction.

    **"Surface layer" is defined by visibility, not by distance.** A block is
    kept iff at least one of its six faces borders `exterior_air` -- air
    6-connected to the outside of the bounding box. Three consequences, each of
    which is the reason for the choice:

    * The exterior region itself is *exactly* invariant. No kept block loses a
      face, and every cell this probe empties is one with no face on the
      exterior, so the emptied cells join the sealed-air region rather than
      extending the outside one (induction over the deleted set). The set of
      faces bordering the outside -- the rendered surface -- is therefore
      identical block for block, which is a stronger silhouette guarantee than
      `solidify` gives and makes the render-tier reading unambiguous.
    * The bounding box is preserved, so `render_structure`'s camera fit is
      unchanged and the probe is not detectable as a change of scale, the
      mistake `chunk_delete` documents.
    * Interiors reachable through a door or window are *not* emptied, because
      their air is genuinely exterior: what you can see through a window, you
      can see. That is the aperture accounting `geometry.interior_volumes`
      already established for real houses, and it makes this probe conservative
      by construction -- it deletes only mass that no view could resolve.

    A one-block-thick shell is a fixed point; a solid cube becomes its own
    surface. Palette is untouched on every surviving block, so the rung is not
    detectable as a material anomaly either.
    """
    from scipy import ndimage

    c = s.crop_to_non_air()
    occ = c.occupied_mask
    if not occ.any():
        return c
    struct = ndimage.generate_binary_structure(3, 1)
    # Dilating in padded space keeps blocks on the crop boundary -- they face the
    # outside air the crop removed, and deleting them would shrink the silhouette.
    ext = np.pad(exterior_air(occ), 1, constant_values=True)
    keep = occ & ndimage.binary_dilation(ext, struct)[1:-1, 1:-1, 1:-1]
    drop = occ & ~keep
    if not drop.any():
        return c
    bi, bd = c.block_ids.copy(), c.block_data.copy()
    bi[drop], bd[drop] = c.air_block_id, 0
    return Structure(block_ids=bi, block_data=bd)


def solidify(s: Structure) -> Structure:
    """Fill every SEALED air cell. Interior mass added, silhouette preserved.

    The probe that motivates the geometry tier. Because `render_structure` sees
    only the outside, filling the rooms is close to invisible to any image-space
    metric -- measured on 128 held-out builds, MV-DINO-KID cannot separate it
    from real at all (0.000 [-0.002, 0.004] against a real -0.001
    [-0.004, 0.003]), while it adds a median 33% to the block count across 94%
    of builds. Any metric that claims to measure build quality and cannot
    separate this rung is measuring the facade.

    `binary_fill_holes` reaches only air that is UNREACHABLE from the bounding
    box without passing through solid, so a room with a doorway to the outside
    survives the probe untouched: on 200 real houses the sealed share of
    interior volume has median 100% but mean 85.5%, and 6% of builds are not
    altered at all. The claim this rung supports is therefore "a 33% mass
    increase the render tier cannot see", NOT "every room is deleted" -- an
    overstatement that stood in this docstring, in the leaderboard recipe and
    in the paper until a fixture test on a hollow cube with a 1x1 doorway showed
    the probe adding exactly zero cells.

    Filled cells take the build's own most common material, so the probe is not
    detectable as a palette anomaly either.
    """
    from scipy import ndimage

    c = s.crop_to_non_air()
    occ = c.occupied_mask
    if not occ.any():
        return c
    filled = ndimage.binary_fill_holes(
        occ, structure=ndimage.generate_binary_structure(3, 1))
    add = filled & ~occ
    if not add.any():
        return c
    bi, bd = c.block_ids.copy(), c.block_data.copy()
    vals, counts = np.unique(bi[occ], return_counts=True)
    bi[add], bd[add] = vals[counts.argmax()], 0
    return Structure(block_ids=bi, block_data=bd)


def partial_solidify(s: Structure, frac: float,
                     rng: np.random.Generator) -> Structure:
    """Fill a fraction `frac` of the enclosed air cells. Graded `solidify`.

    The dose-response arm for structural damage. `solidify` is the frac=1.0
    endpoint; intermediate values let a metric be asked the harder question --
    not "can you see every room removed" but "does your number move smoothly as
    rooms are removed" -- which is what separates a metric that ranks quality
    from one that merely detects an artifact.

    Interior cells are filled **inward from the walls**, nearest-first, so a
    half-solidified build is a house whose walls have grown until half the
    interior volume is gone. Two alternatives were tried and rejected: random
    speckle is trivially detectable as noise rather than as lost interior, and
    filling whole rooms largest-first makes the dose axis useless -- one dominant
    pocket is typically more than a quarter of the interior, so `frac=0.25`
    already removed 71% of it (0.1175 -> 0.0337) and every larger dose was
    indistinguishable. Nearest-first fills exactly `frac` of the interior volume
    at every setting.
    """
    from scipy import ndimage

    c = s.crop_to_non_air()
    occ = c.occupied_mask
    if not occ.any() or frac <= 0:
        return c
    filled = ndimage.binary_fill_holes(
        occ, structure=ndimage.generate_binary_structure(3, 1))
    pockets = filled & ~occ
    if not pockets.any():
        return c
    # Depth of each interior cell below the wall it hides behind.
    depth = ndimage.distance_transform_cdt(pockets, metric="taxicab")
    idx = np.argwhere(pockets)
    d = depth[tuple(idx.T)]
    k = int(round(min(max(frac, 0.0), 1.0) * len(idx)))
    if k < 1:
        return c
    keep = idx[np.argsort(d, kind="stable")[:k]]
    add = np.zeros_like(pockets)
    add[tuple(keep.T)] = True
    bi, bd = c.block_ids.copy(), c.block_data.copy()
    vals, counts = np.unique(bi[occ], return_counts=True)
    bi[add], bd[add] = vals[counts.argmax()], 0
    return Structure(block_ids=bi, block_data=bd)


def occupancy_noise(s: Structure, p: float, rng: np.random.Generator) -> Structure:
    """Move a fraction `p` of blocks to random cells touching the build.

    The structural analogue of `block_noise`, and the reason the geometry tier
    needs its own noise rung. `block_noise` retypes blocks without moving them,
    so an occupancy-only metric is exactly blind to it -- ordering that rung
    would be asking the metric to see something it is designed not to see. This
    probe is the mirror image: block count and palette multiset are preserved
    exactly, and only the arrangement degrades.
    """
    c = s.crop_to_non_air()
    occ = c.occupied_mask
    n = int(occ.sum())
    k = int(round(p * n))
    if k < 1 or n == 0:
        return c

    # Destinations: air cells 6-adjacent to the solid, so displaced blocks stay
    # near the build. Scattering them anywhere in the bounding box would make
    # the rung trivially separable by block count per unit volume alone.
    nbr = np.zeros_like(occ)
    for axis in range(3):
        for shift in (1, -1):
            nbr |= np.roll(occ, shift, axis=axis)
    dest = np.argwhere(nbr & ~occ)
    src = np.argwhere(occ)
    if len(dest) == 0:
        return c
    k = min(k, len(dest), len(src))
    take = src[rng.permutation(len(src))[:k]]
    put = dest[rng.permutation(len(dest))[:k]]

    bi, bd = c.block_ids.copy(), c.block_data.copy()
    moved = [(int(bi[tuple(t)]), int(bd[tuple(t)])) for t in take]
    for t in take:
        bi[tuple(t)], bd[tuple(t)] = c.air_block_id, 0
    for d, (a, b) in zip(put, moved):
        bi[tuple(d)], bd[tuple(d)] = a, b
    return Structure(block_ids=bi, block_data=bd)


def jitter_columns(s: Structure, amp: int, rng: np.random.Generator) -> Structure:
    """Shift each vertical column by up to `amp` blocks. Planarity destroyed.

    Walls, floors and roof planes are broken into steps while the block count,
    the palette and the footprint are all preserved exactly. This is the rung
    `wall_frac` exists to catch.
    """
    c = s.crop_to_non_air()
    bi, bd = c.block_ids, c.block_data
    X, Y, Z = c.shape
    nbi = np.full_like(bi, c.air_block_id)
    nbd = np.zeros_like(bd)
    sh = rng.integers(-amp, amp + 1, size=(X, Z))
    for x in range(X):
        for z in range(Z):
            d = int(sh[x, z])
            src = slice(max(0, -d), Y - max(0, d))
            dst = slice(max(0, d), Y - max(0, -d))
            nbi[x, dst, z] = bi[x, src, z]
            nbd[x, dst, z] = bd[x, src, z]
    return Structure(block_ids=nbi, block_data=nbd)
