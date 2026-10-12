"""The LDraw rasterizer's two coverage rules: studs out, baseplates in.

`legogen/renderer/ldraw_voxel.py` counts a part's bounding box into a cell when
it covers more than half of it, which is what drops a brick's 4 LDU stud band
without dropping a plate's 8 LDU body. A 4 LDU baseplate body scores that same
0.5 at every alignment, so the threshold alone deleted every baseplate in the
OMR corpus and with it the floor of the ground-floor room. These tests pin both
halves of the fixed rule: the stud layer still goes, the baseplate stays, and it
stays as *one* layer sitting under the bricks rather than two layers overlapping
them.
"""

from __future__ import annotations

import numpy as np
import pytest

from legogen.renderer import ldraw_voxel as V

pytestmark = pytest.mark.skipif(
    not (V.LDRAW_LIB / "parts" / "3001.dat").is_file(),
    reason="LDraw parts library not present (data/lego/ldraw)")


@pytest.fixture(scope="module")
def geom() -> V.PartGeometry:
    return V.PartGeometry(V.LDrawLibrary())


#: The lattice phase a real build imposes on these parts: bricks stand on
#: multiples of 8 LDU up, and an even-footprint part's walls land on multiples of
#: 20 in x/z. `voxelize`'s "auto" phase reads the phase off the parts present, so
#: a baseplate voxelized *alone* aligns to its own 4 LDU body and never straddles
#: anything; the bug only exists on a model's lattice, which is what this pins.
MODEL_PHASE = (0.0, 0.0, 0.0)


def _place(part: str, up: float = 0.0) -> V.Placement:
    """One part, its origin `up` LDU above the model origin (LDraw y is down)."""
    return V.Placement(part=part, colour=4, matrix=np.eye(3),
                       offset=np.array([0.0, -up, 0.0]))


def test_known_part_sizes_unchanged(geom: V.PartGeometry) -> None:
    """The stud rule: every catalogue part still voxelizes to its own size."""
    ok, n = V.test_single_parts(V.LDrawLibrary(), geom, verbose=False)
    assert (ok, n) == (n, n)


@pytest.mark.parametrize("part", ["3857.dat", "3811.dat", "4186.dat"])
def test_thin_baseplate_survives_the_model_lattice(part: str,
                                                   geom: V.PartGeometry) -> None:
    """A 4 LDU body on the *model's* lattice: one layer, not zero and not two.

    On `MODEL_PHASE` the baseplate's box straddles a cell wall and scores 0.5 in
    both cells, which the bare threshold rejected outright.
    """
    pl = [_place(part)]
    vm = V.voxelize(pl, geom, phase=MODEL_PHASE, source=part)
    assert vm.rescued == 1
    assert vm.shape[1] == 1
    assert vm.occ.all()


def test_baseplate_lands_under_the_brick_not_through_it(
        geom: V.PartGeometry) -> None:
    """The tie breaks downwards, so the plate is a floor and not a mezzanine.

    A 2x4 brick on a 16x32 baseplate is 1 plate + 3 plates tall. Getting 3
    layers would mean the rescued baseplate landed in the brick's own first
    layer; getting 5 would mean its studs earned a layer of their own. Both part
    origins sit on the *top* of their own body, so a brick standing on the
    plate's top surface (up 0) has its origin 24 LDU up.
    """
    plate, brick = _place("3857.dat"), _place("3001.dat", up=24.0)
    vm = V.voxelize([plate, brick], geom, phase=MODEL_PHASE, source="plate+brick")
    assert vm.rescued == 1
    assert vm.shape[1] == 4
    floor = vm.occ[:, 0, :]
    assert floor.all(), "the baseplate layer is the model's floor, fully solid"
    above = vm.occ[:, 1:, :]
    assert above.sum() == 24, "the brick is 4x3x2 cells, its studs no layer"
    # The brick sits directly on the plate, in the layer above it.
    assert vm.occ[:, 1, :].sum() == 8


def test_rescue_does_not_move_a_part_that_already_fits(
        geom: V.PartGeometry) -> None:
    """A brick, a plate and a tile clear the volume test and are left alone.

    Each on the phase its own footprint sits on: even footprints on the 20 LDU
    walls (`MODEL_PHASE`), odd ones on `V.PHASE`, offset half a stud.
    """
    for part in ("3001.dat", "3020.dat", "3068b.dat"):
        vm = V.voxelize([_place(part)], geom, phase=MODEL_PHASE, source=part)
        assert vm.rescued == 0, part
    for part in ("3005.dat", "3024.dat", "3009.dat"):
        vm = V.voxelize([_place(part)], geom, phase=V.PHASE, source=part)
        assert vm.rescued == 0, part


def test_sealed_ground_floor_of_the_apple_tree_house(geom: V.PartGeometry) -> None:
    """The regression that started this: a room whose floor had vanished.

    `5891-1_Small-House.mpd` stands on a bright green `3857.dat`. With the
    baseplate dropped, the ground storey's air escaped through the bottom face of
    the bounding box as a single 1,276-cell component, and the interior probe
    called the whole storey outdoors.
    """
    src = V.OMR_DIR / "5891-1_Small-House.mpd"
    if not src.is_file():
        pytest.skip("OMR corpus not present (data/lego/omr)")
    lib = V.LDrawLibrary()
    placements, report = V.flatten_model(src, lib, geom)
    vm = V.voxelize(placements, geom, source=str(src), report=report)
    floor = vm.occ[:, 0, :]
    assert floor.mean() > 0.9, "the baseplate is back as the bottom layer"
    void = V.enclosed_air(vm.occ)
    # The storey is 22 plates tall; every layer of it now has sealed interior.
    assert (void[:, 1:22, :].sum(axis=(0, 2)) > 0).all()
    assert int(void.sum()) == 2508


def test_out_of_int32_colour_code_does_not_crash(geom: V.PartGeometry) -> None:
    """A malformed colour code survives the grid instead of overflowing it.

    `10220-1.mpd` puts the literal code 4294967295 (0xFFFFFFFF, "no colour set"
    from some editors) on six `97122.dat`. Those parts are thin enough that the
    volume test used to drop them, so the code never reached the colour array;
    once they are rescued it does, and an int32 array raises OverflowError.
    """
    src = V.OMR_DIR / "10220-1.mpd"
    if not src.is_file():
        pytest.skip("OMR corpus not present (data/lego/omr)")
    lib = V.LDrawLibrary()
    placements, report = V.flatten_model(src, lib, geom)
    assert any(p.colour == 4294967295 for p in placements), "fixture premise"
    vm = V.voxelize(placements, geom, source=str(src), report=report)
    assert vm.rescued > 0
    assert (vm.colour[vm.occ] >= 0).all(), "every filled cell got a colour"
    # 0xFFFFFFFF is neither in LDConfig nor a direct colour: documented grey.
    assert V.colour_rgba(4294967295, V.parse_colours()) == (128, 128, 128, 255)
