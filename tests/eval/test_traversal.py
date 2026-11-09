"""Traversability: a two-block player walking a build.

The fixtures are hand-built boxes, because the claims are about rules a player
obeys (step up one, no jump under a low ceiling unless the step is a stair,
doors open, walls do not) and each fixture isolates one rule.
"""

from __future__ import annotations

import numpy as np
import pytest

from blockgen.eval import traversal as tv
from blockgen.eval.bench import geometry as geom
from blockgen.eval.bench import probes as pb
from blockgen.utils.data import Structure

PLANKS, STAIRS, DOOR, STONE = 5, 53, 64, 1


def _struct(ids: np.ndarray) -> Structure:
    ids = ids.astype(np.int32)
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


def _box(n=9, height=None, door=True, door_block=False) -> Structure:
    """Hollow box with a 1x2 doorway in the x=0 wall (air, or a door block)."""
    h = height or n
    ids = np.zeros((n, h, n), dtype=np.int32)
    ids[0, :, :] = ids[-1, :, :] = PLANKS
    ids[:, :, 0] = ids[:, :, -1] = PLANKS
    ids[:, 0, :] = ids[:, -1, :] = PLANKS
    if door:
        ids[0, 1:3, n // 2] = DOOR if door_block else 0
    return _struct(ids)


def _two_storeys(step_block: int) -> Structure:
    """A 9-cube with a floor slab at y=4 and a three-step stair up to it.

    The steps rise along x at z=1, under the slab; the slab has a two-cell hole
    where the climber's head and last step go. With `STAIRS` each rise is half
    a block and needs no jump; with full `PLANKS` the second rise needs jump
    room where the slab is, and the player is stuck downstairs.
    """
    s = _box(9)
    ids = s.block_ids.copy()
    ids[1:-1, 4, 1:-1] = PLANKS            # upper floor
    ids[2, 4, 1] = ids[3, 4, 1] = 0        # the stairwell hole
    for k in (1, 2, 3):
        ids[k, k, 1] = step_block
    return _struct(ids)


def _house_rng() -> Structure:
    """The geometry test's caricature: hollow box, doorway, a window."""
    occ = np.ones((9, 9, 9), dtype=bool)
    occ[1:-1, 1:-1, 1:-1] = False
    occ[0, 1:3, 4] = False
    occ[4, 5, 0] = False
    return _struct(np.where(occ, PLANKS, 0))


# --- interior masks agree with the bench ------------------------------------
@pytest.mark.parametrize("s", [_box(), _box(door=False), _house_rng(), _two_storeys(STAIRS)])
def test_interior_masks_pin_to_geometry_interior_volumes(s):
    sealed, open_int = tv.interior_masks(s.occupied_mask)
    assert (int(sealed.sum()), int(open_int.sum())) == geom.interior_volumes(s.occupied_mask)


def test_ground_closes_a_floorless_room():
    """A box with no floor is no room to the bench and a room to a player."""
    s = _box(door=False)
    ids = s.block_ids.copy()
    ids[1:-1, 0, 1:-1] = 0                  # take the floor away, walls still touch the ground
    s = _struct(ids)
    assert geom.interior_volumes(s.occupied_mask) == (0, 0)
    sealed, open_int = tv.interior_masks(s.occupied_mask, ground=True)
    assert int(sealed.sum()) == 7 * 8 * 7   # the room, down to the ground
    r = tv.traversal_stats(_struct(np.where(ids == 0, 0, ids)))
    assert r.interior_standing == 49 and not r.enterable


# --- the walk ---------------------------------------------------------------
def test_air_doorway_admits_the_player():
    r = tv.traversal_stats(_box())
    assert r.enterable and r.interior_standing_reached == 1.0
    assert r.floors == 1 and r.floors_reached == 1 and r.entrances == 1
    assert r.headroom3 == 1.0


def test_door_block_is_walked_through():
    r = tv.traversal_stats(_box(door_block=True))
    assert r.enterable and r.interior_standing_reached == 1.0 and r.entrances == 1


def test_sealed_box_is_not_enterable_but_has_a_room():
    r = tv.traversal_stats(_box(door=False))
    assert not r.enterable and r.interior_standing_reached == 0.0
    assert r.interior_cells == 343 and r.interior_standing == 49
    assert r.floors == 1 and r.floors_reached == 0 and r.entrances == 0


def test_start_anywhere_reaches_the_sealed_room():
    s = _box(door=False)
    w = tv.walk(s, start="anywhere")
    assert int((w.feet() & tv.standing_mask(s)).sum()) == 49 + 0  # the room only; walls fill the rest


def test_dolls_house_has_no_standing_room():
    """Interior one block high: a room by volume, nowhere to stand. With a
    doorway the bench's closing recipe does not even count it as a room (a
    cavity thinner than three in any direction is eaten by the 3x3x3
    closing), so the walkable share is zero either way."""
    sealed = tv.traversal_stats(_box(height=3, door=False))
    assert sealed.interior_cells == 49 and sealed.interior_standing == 0
    assert not sealed.enterable and sealed.interior_standing_reached == 0.0
    doored = tv.traversal_stats(_box(height=3))
    assert doored.interior_cells == 0 and not doored.enterable
    assert doored.interior_standing_reached == 0.0


def test_stairs_climb_under_a_low_ceiling_and_full_blocks_do_not():
    up = tv.traversal_stats(_two_storeys(STAIRS))
    assert up.floors == 2 and up.floors_reached == 2
    stuck = tv.traversal_stats(_two_storeys(PLANKS))
    assert stuck.floors == 2 and stuck.floors_reached == 1
    assert 0.0 < stuck.interior_standing_reached < 1.0


def test_one_block_step_is_walked_with_headroom():
    """A raised floor one block up, open ceiling above: a plain jump."""
    s = _box(9)
    ids = s.block_ids.copy()
    ids[4:-1, 1, 1:-1] = PLANKS             # a platform one block high
    r = tv.traversal_stats(_struct(ids))
    assert r.interior_standing_reached == 1.0


def test_fence_is_a_wall():
    """A fence across the doorway stops a player (1.5 blocks, cannot be jumped)."""
    s = _box(9)
    ids = s.block_ids.copy()
    ids[0, 1, 4] = 85                       # oak fence in the door's lower cell
    r = tv.traversal_stats(_struct(ids))
    assert not r.enterable


def test_terrain_does_not_count_as_a_room():
    """A hollow box on a two-layer slab of dirt with a one-cell yard: the room
    is reached (the player is set down anywhere outside, including on the
    dirt), and the dirt adds no interior."""
    s = _box(9)
    ids = np.zeros((11, 11, 11), dtype=np.int32)
    ids[:, :2, :] = 3                       # dirt, two deep, with a yard around
    ids[1:-1, 2:, 1:-1] = s.block_ids
    r = tv.traversal_stats(_struct(ids))
    assert r.enterable and r.interior_standing_reached == 1.0
    assert r.interior_cells == 343


def test_a_door_two_blocks_up_a_sheer_wall_still_admits():
    """The crop took the ground: the front door sits on a two-high foundation
    with nothing to stand on outside it. A door block is where the builder meant
    you to come in, so the walk starts there; an empty doorway is not."""
    base = _box(9, door_block=True).block_ids
    ids = np.zeros((9, 11, 9), dtype=np.int32)
    ids[:, :2, :] = STONE                   # sheer foundation, full footprint
    ids[:, 2:, :] = base
    assert tv.traversal_stats(_struct(ids)).enterable
    ids[0, 3:5, 4] = 0                      # the same doorway, no door in it
    assert not tv.traversal_stats(_struct(ids)).enterable


# --- probes -----------------------------------------------------------------
def test_seal_openings_keeps_the_room_and_locks_the_door():
    s = _box()
    before = tv.traversal_stats(s)
    sealed = tv.seal_openings(s)
    after = tv.traversal_stats(sealed)
    assert before.enterable and not after.enterable
    assert after.interior_cells == before.interior_cells == 343
    # exactly the two doorway cells were filled, with the dominant material
    assert int(sealed.occupied_mask.sum()) - int(s.occupied_mask.sum()) == 2
    assert sealed.block_ids[0, 1, 4] == PLANKS and sealed.block_ids[0, 2, 4] == PLANKS


def test_seal_openings_turns_a_door_block_to_wall():
    s = _box(door_block=True)
    sealed = tv.seal_openings(s)
    assert int(sealed.occupied_mask.sum()) == int(s.occupied_mask.sum())
    assert not (sealed.block_ids == DOOR).any()
    assert not tv.traversal_stats(sealed).enterable


def test_seal_openings_leaves_a_sealed_box_alone():
    s = _box(door=False)
    assert np.array_equal(tv.seal_openings(s).block_ids, s.block_ids)


def test_lower_ceilings_takes_one_layer_and_kills_a_two_high_room():
    tall = tv.lower_ceilings(_box(9))
    r = tv.traversal_stats(tall)
    assert r.interior_cells == 343 - 49 and r.enterable
    low = tv.lower_ceilings(_box(height=4))   # interior two high -> one high
    r2 = tv.traversal_stats(low)
    assert r2.interior_standing == 0 and not r2.enterable


def test_bench_solidify_removes_everything_to_walk():
    r = tv.traversal_stats(pb.solidify(_box(door=False)))
    assert r.interior_cells == 0 and r.interior_standing_reached == 0.0 and not r.enterable


# --- summary and frames -----------------------------------------------------
def test_summarize_reports_distribution_and_build_fractions():
    sm = tv.summarize([_box(), _box(door=False), _box(height=3)])
    assert sm["builds"]["n"] == 3
    assert sm["builds"]["enterable_frac"] == pytest.approx(1 / 3)
    assert sm["builds"]["no_interior_frac"] == pytest.approx(1 / 3)
    assert sm["interior_standing_reached"]["n"] == 3
    assert sm["headroom3"]["n"] == 2          # undefined for the doll's house


def test_walkthrough_frames_paint_the_trail_and_the_player():
    s = _box()
    frames = list(tv.walkthrough_frames(s))
    w = tv.walk(s)
    assert len(frames) == w.max_dist + 1
    assert all(f.shape == s.shape for f in frames)
    last = frames[-1]
    # the wavefront starts on the ring outside the crop, so the first frame
    # may show no stand-in; the frame that enters the doorway must
    player = [((f.block_ids == tv.PLAYER_BLOCK[0]) & (f.block_data == tv.PLAYER_BLOCK[1])).sum()
              for f in frames]
    assert max(player) >= 2 and player[1] >= 2
    trail = (last.block_ids == tv.TRAIL_BLOCK[0]) & (last.block_data == tv.TRAIL_BLOCK[1])
    assert trail[1:-1, 0, 1:-1].all()         # every interior floor cell was stood on
    assert not (trail[0] | trail[-1] | trail[:, :, 0] | trail[:, :, -1]).any()
    unreached = (frames[0].block_ids == tv.UNREACHED_BLOCK[0]) & (frames[0].block_data == tv.UNREACHED_BLOCK[1])
    assert unreached[1:-1, 0, 1:-1].all()     # before the walk, every floor cell is still to find
    assert not ((last.block_ids == tv.UNREACHED_BLOCK[0]) & (last.block_data == tv.UNREACHED_BLOCK[1])).any()


# --- the blind player, what the geometric descriptor carries ----------------
def test_blind_walk_treats_a_door_block_as_wall_and_an_open_doorway_as_a_way_in():
    assert tv.traversal_stats(_box(door_block=True), blind=True).enterable is False
    assert tv.traversal_stats(_box(), blind=True).interior_standing_reached == 1.0


def test_blind_walk_needs_a_jump_on_stairs():
    """Under a low ceiling the half-step rule is what climbs; blind has none."""
    assert tv.traversal_stats(_two_storeys(STAIRS), blind=True).floors_reached == 1
    assert tv.traversal_stats(_two_storeys(STAIRS)).floors_reached == 2


def test_walk_scalars_are_exactly_blind_to_material_and_d4_invariant():
    s = _two_storeys(STAIRS)
    rng = np.random.default_rng(0)
    base = tv.walk_scalars(s)
    assert base == tv.walk_scalars(pb.shuffle_materials(s, rng))
    assert base == tv.walk_scalars(pb.monochrome(s))
    for k in (1, 2, 3):
        assert base == tv.walk_scalars(pb.rotate_y(s, k))
    assert base == tv.walk_scalars(pb.mirror_x(s))


def test_walk_scalars_are_finite_and_zero_without_standing_room():
    w = tv.walk_scalars(_box(height=3, door=False))
    assert w == {"walk_standing": 0.0, "walk_floors": 0.0, "walk_headroom3": 0.0}
    w = tv.walk_scalars(_box(door=False))      # a locked room is still a room
    assert w == {"walk_standing": pytest.approx(49 / 343), "walk_floors": 1.0,
                 "walk_headroom3": 1.0}
    assert tv.walk_scalars(_two_storeys(STAIRS))["walk_floors"] == 2.0


def test_traversal_report_has_the_distance_to_real_shape():
    real = [_box(), _box(door_block=True), _two_storeys(STAIRS), _box(7)]
    gen = [_box(door=False), _box(height=4), _box(door=False)]
    rep = tv.traversal_report(gen, real, n_boot=40)
    assert set(rep) == set(tv.TRAVERSAL_REPORT)
    e = rep["interior_standing_reached"]
    assert e["direction"] == "distance_to_real"
    assert e["real"]["mean"] == pytest.approx(1.0) and e["gen"]["mean"] == pytest.approx(0.0)
    assert e["w1"] == pytest.approx(1.0)
    # a reference table computed once can stand in for the real builds
    table = tv.traversal_table(real)
    assert tv.traversal_report(gen, [], n_boot=40, real_table=table)["enterable"]["w1"] == pytest.approx(
        rep["enterable"]["w1"])
