"""The renderer must cull against OPAQUE neighbours, not against occupancy.

The bug these tests pin: `_exposed_masks` used to ask only "is the neighbour
cell filled?". Under that rule a solid block behind a glass block had its face
culled and the glass's own inner face culled too, so looking through a window
you saw the BACKGROUND rather than what stands behind the pane. That made the
`solidify` probe -- which fills sealed air with the build's own dominant
material -- look like it had *removed* the interior through every window, the
exact opposite of the paper's claim that a window in front of a filled void
reveals the fill.
"""

from __future__ import annotations

import numpy as np
import pytest

from blockgen.renderer import textured
from blockgen.utils.data import Structure

PLANKS = (5, 0)
GLASS = (102, 0)
ICE = (79, 0)


def _lib(**alpha_by_key):
    """A minimal face-texture library: one 2x2 tile per block, chosen alpha."""
    lib = {}
    for key, alpha in alpha_by_key.items():
        bid, bdat = key_pair = _KEYS[key]
        tile = np.zeros((2, 2, 4), np.uint8)
        tile[..., :3] = 200
        tile[..., 3] = alpha
        lib[key_pair] = {"top": tile, "side": tile, "bottom": tile}
    return lib


_KEYS = {"planks": PLANKS, "glass": GLASS, "ice": ICE}


def _line(*blocks) -> Structure:
    """A 1x1xN column of blocks along z, air-padded at both ends."""
    ids = np.zeros((1, 1, len(blocks) + 2), np.int32)
    data = np.zeros_like(ids)
    for i, (bid, bdat) in enumerate(blocks):
        ids[0, 0, i + 1], data[0, 0, i + 1] = bid, bdat
    return Structure(block_ids=ids, block_data=data)


def _faces(structure, lib):
    """{(x, y, z, direction)} actually drawn, via the real mask code."""
    occupied = structure.occupied_mask
    masks = textured._exposed_masks(
        occupied, textured._cull_classes(structure, lib))
    return {
        (int(x), int(y), int(z), name)
        for name in masks
        for x, y, z in np.argwhere(masks[name])
    }


def test_solid_behind_glass_keeps_the_face_you_see_through_the_window():
    """glass | planks: the planks' face toward the glass is DRAWN.

    This is the whole bug. Minecraft culls against opaque cubes only, so the
    block behind a pane is what you see through it.
    """
    s = _line(GLASS, PLANKS)
    lib = _lib(glass=128, planks=255)
    faces = _faces(s, lib)
    assert (0, 0, 2, "north") in faces, "planks face toward the glass was culled"
    # ...and the glass's own inner face is culled, because planks ARE opaque.
    # Vanilla does the same; drawing both would put two coplanar quads in the
    # depth buffer for one seam.
    assert (0, 0, 1, "north") in faces      # outward face of the pane (toward air)
    assert (0, 0, 1, "south") not in faces  # inward face, hidden by the planks


def test_occupancy_only_cull_hides_it_the_old_behaviour():
    """Without cull classes the mask degrades to the old, wrong rule."""
    s = _line(GLASS, PLANKS)
    masks = textured._exposed_masks(s.occupied_mask)
    drawn = {(int(x), int(y), int(z), n) for n in masks
             for x, y, z in np.argwhere(masks[n])}
    assert (0, 0, 2, "north") not in drawn


def test_opaque_behind_opaque_is_still_culled():
    """The fix must not turn the renderer into an uncull: solid stays solid."""
    s = _line(PLANKS, PLANKS)
    faces = _faces(s, _lib(planks=255))
    assert (0, 0, 2, "north") not in faces
    assert (0, 0, 1, "south") not in faces
    assert len(faces) == 10  # two cubes, one shared seam gone from each side


def test_two_glass_blocks_of_one_kind_do_not_draw_their_shared_seam():
    """Vanilla omits the seam between like panes; so do we.

    Keeping it would be harmless to look at but would double the blended
    geometry of a window wall for nothing.
    """
    s = _line(GLASS, GLASS)
    faces = _faces(s, _lib(glass=128))
    assert (0, 0, 1, "north") in faces
    assert (0, 0, 2, "south") in faces
    assert (0, 0, 1, "south") not in faces
    assert (0, 0, 2, "north") not in faces


def test_two_different_transparent_blocks_do_draw_both_sides():
    """glass | ice: you can see the ice through the glass and vice versa."""
    s = _line(GLASS, ICE)
    faces = _faces(s, _lib(glass=128, ice=200))
    assert (0, 0, 1, "south") in faces
    assert (0, 0, 2, "north") in faces


def test_transparency_is_resolved_once_per_block_kind_not_per_voxel():
    """A big build must not re-inspect a texture per voxel."""
    ids = np.full((8, 8, 8), GLASS[0], np.int32)
    s = Structure(block_ids=ids, block_data=np.zeros_like(ids))

    seen = []

    class Counting(dict):
        def __getitem__(self, key):
            seen.append(key)
            return super().__getitem__(key)

    lib = Counting(_lib(glass=128))
    textured._cull_classes(s, lib)
    assert len(seen) == 1, f"looked up textures {len(seen)} times for one kind"


def test_a_sealed_fill_is_visible_through_a_window_end_to_end():
    """The paper's mechanism, rendered: a filled void shows through glass.

    A 5^3 shell of planks with one glass block in the +z wall. Hollow, the
    window shows the far interior wall; filled, it shows the fill. Either way
    it must NOT show the background, which is what the occupancy cull gave.
    """
    pyrender = pytest.importorskip("pyrender")
    ids = np.zeros((5, 5, 5), np.int32)
    ids[:, :, :] = PLANKS[0]
    ids[1:4, 1:4, 1:4] = 0                     # hollow it out
    ids[2, 2, 4] = GLASS[0]                    # one window in the +z wall
    hollow = Structure(block_ids=ids, block_data=np.zeros_like(ids))
    filled_ids = ids.copy()
    filled_ids[1:4, 1:4, 1:4] = PLANKS[0]
    filled = Structure(block_ids=filled_ids, block_data=np.zeros_like(ids))

    lib = _lib(glass=128, planks=255)
    # Hollow: the window opens on air, so the far interior wall (z=0) shows.
    masks = textured._exposed_masks(
        hollow.occupied_mask, textured._cull_classes(hollow, lib))
    assert masks["south"][2, 2, 0], "the far interior wall is not drawn"
    assert not hollow.occupied_mask[2, 2, 3]

    # Filled: the cell right behind the pane is now planks, and its face
    # toward the pane is drawn. Under the occupancy cull it was not, and the
    # window showed the background instead.
    classes = textured._cull_classes(filled, lib)
    masks = textured._exposed_masks(filled.occupied_mask, classes)
    assert masks["south"][2, 2, 3], "the fill behind the window is not drawn"
    old = textured._exposed_masks(filled.occupied_mask)
    assert not old["south"][2, 2, 3], "old cull should have hidden it"
