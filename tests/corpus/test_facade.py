import numpy as np
import pytest

from blockgen.corpus import facade as fc
from blockgen.eval.bench import probes
from blockgen.utils.data import Structure

STONE, PLANK, GLASS = 1, 5, 20


def _box(n=8, h=6, roof=True, pillar=False, window=True):
    ids = np.zeros((n, h, n), np.int32)
    ids[:, 0, :] = STONE                       # floor
    ids[0, :, :] = ids[-1, :, :] = PLANK       # walls
    ids[:, :, 0] = ids[:, :, -1] = PLANK
    if roof:
        ids[:, -1, :] = STONE
    if window:
        ids[0, 2, 3] = GLASS
    if pillar:
        ids[n // 2, 1:h - 1, n // 2] = STONE   # hidden inside
    data = np.zeros_like(ids)
    return Structure(block_ids=ids, block_data=data)


def test_depth_and_ids_follow_the_face_convention():
    s = Structure(block_ids=np.zeros((7, 9, 5), np.int32),
                  block_data=np.zeros((7, 9, 5), np.int32))
    s.block_ids[2, 5, 3] = GLASS
    s.block_data[2, 5, 3] = 4
    f = fc.project(s, "-z")                    # looks along z, u = x, v = y
    assert f.depth.shape == (7, 9)
    assert f.depth[2, 5] == 3 and f.ids[2, 5] == GLASS and f.data[2, 5] == 4
    assert (f.depth == fc.EMPTY).sum() == 7 * 9 - 1
    assert fc.project(s, "+z").depth[2, 5] == 5 - 1 - 3
    assert fc.project(s, "-x").depth[3, 5] == 2  # u = z, v = y
    assert fc.project(s, "+x").depth[3, 5] == 7 - 1 - 2
    top = fc.project(s, "+y")                  # u = x, v = z
    assert top.depth[2, 3] == 9 - 1 - 5
    assert fc.project(s, "-y").depth[2, 3] == 5


def test_lift_inverts_project_on_the_visible_shell():
    s = _box(pillar=True)
    back = fc.lift(fc.project_all(s, fc.ALL_FACES), s.shape)
    vis = fc.visible_mask(s.occupied_mask, fc.ALL_FACES)
    assert np.array_equal(back.block_ids != 0, vis)
    assert np.array_equal(back.block_ids[vis], s.block_ids[vis])
    # the pillar is hidden from every face and does not come back
    assert back.block_ids[4, 2, 4] == 0 and s.block_ids[4, 2, 4] == STONE
    # the hollow box itself is entirely visible
    plain = _box()
    assert np.array_equal(fc.shell(plain, fc.ALL_FACES).block_ids, plain.block_ids)


def test_sides_lose_a_flat_roof_and_the_top_face_restores_it():
    s = _box()
    four = fc.shell(s, fc.SIDES)
    five = fc.shell(s, fc.SIDES_TOP)
    assert four.shape == s.shape == five.shape
    roof = s.block_ids[1:-1, -1, 1:-1]
    assert (four.block_ids[1:-1, -1, 1:-1] == 0).all() and (roof == STONE).all()
    assert np.array_equal(five.block_ids[:, -1, :], s.block_ids[:, -1, :])
    # floor interior is seen by no side and not by the roof plan either
    assert (five.block_ids[1:-1, 0, 1:-1] == 0).all()


def test_shell_is_idempotent_and_inside_the_hollow_probe():
    rng = np.random.default_rng(3)
    ids = np.zeros((12, 10, 12), np.int32)
    ids[2:10, 0:7, 2:10] = rng.integers(1, 30, (8, 7, 8))
    ids[4:8, 2:5, 4:8] = 0                      # a room behind two-thick walls
    s = Structure(block_ids=ids, block_data=np.zeros_like(ids))
    sh = fc.shell(s, fc.ALL_FACES)
    again = fc.shell(sh, fc.ALL_FACES)
    assert np.array_equal(sh.block_ids, again.block_ids)
    hollow = probes.hollow(s)
    assert ((sh.block_ids != 0) <= (hollow.block_ids != 0)).all()
    assert sh.occupied_mask.sum() < s.occupied_mask.sum()


def test_a_rotation_about_y_only_relabels_faces():
    s = _box(window=True)
    s.block_ids[3, 3, 0] = GLASS                # break the symmetry
    r = Structure(block_ids=np.rot90(s.block_ids, 1, axes=(0, 2)),
                  block_data=np.zeros(s.shape, np.int32))
    originals = [f.depth for f in fc.project_all(s, fc.SIDES).values()]
    for f in fc.project_all(r, fc.SIDES).values():
        assert any(np.array_equal(f.depth, o) or np.array_equal(f.depth, o[::-1])
                   for o in originals), f.face
    # and the roof plan is the rotated roof plan
    top_s, top_r = fc.project(s, "+y").depth, fc.project(r, "+y").depth
    variants = [v for k in range(4) for v in (np.rot90(top_s, k), np.rot90(top_s, k)[::-1])]
    assert any(np.array_equal(top_r, v) for v in variants)


def test_lift_infers_shape_and_rejects_inconsistent_faces():
    s = _box(n=6, h=5)
    faces = fc.project_all(s, fc.SIDES_TOP)
    assert fc.lift(faces).shape == s.shape
    # one face image plus its extent already fixes every axis
    assert fc.lift([faces["-z"]]).shape == s.shape
    bad = fc.Facade(face="+y", depth=np.zeros((6, 7), np.int16),
                    ids=np.zeros((6, 7), np.int16), data=np.zeros((6, 7), np.int16),
                    extent=5)
    with pytest.raises(ValueError):
        fc.lift([faces["-z"], bad])


def test_later_faces_overwrite_earlier_ones():
    a = fc.Facade(face="-z", depth=np.zeros((1, 1), np.int16),
                  ids=np.full((1, 1), STONE, np.int16), data=np.zeros((1, 1), np.int16),
                  extent=1)
    b = fc.Facade(face="+z", depth=np.zeros((1, 1), np.int16),
                  ids=np.full((1, 1), PLANK, np.int16), data=np.zeros((1, 1), np.int16),
                  extent=1)
    assert fc.lift([a, b]).block_ids[0, 0, 0] == PLANK
    assert fc.lift([b, a]).block_ids[0, 0, 0] == STONE


def test_shell_keeps_metadata_and_bounding_box():
    s = _box()
    s.metadata["title"] = "box"
    padded = Structure(block_ids=np.pad(s.block_ids, 3), block_data=np.pad(s.block_data, 3),
                       metadata=dict(s.metadata))
    sh = fc.shell(padded)
    assert sh.shape == s.shape and sh.metadata["title"] == "box"


def test_face_picture_shades_depth_and_shares_corners():
    s = _box(n=6, h=4)
    s.block_ids[2, 2, 0] = 0                    # a recess in the -z wall: deeper
    s.block_ids[2, 2, 1] = PLANK
    faces = fc.project_all(s, fc.SIDES)
    pic = fc.face_picture(faces["-z"])
    assert pic.shape == (4, 6, 3) and pic.dtype == np.uint8
    flush, recessed = pic[4 - 1 - 2, 1], pic[4 - 1 - 2, 2]    # rows are y, drawn upward
    assert (recessed.astype(int) < flush.astype(int)).all()
    # walking around: the right column of one side is the left column of the next
    pics = [fc.face_picture(faces[k]) for k in fc.SIDES]
    for a, b in zip(pics, pics[1:] + pics[:1]):
        assert np.array_equal(a[:, -1], b[:, 0])
    sheet = fc.sheet(faces, scale=2, gap=1)
    assert sheet.shape == (4 * 2 + 2, 4 * 6 * 2 + 5, 3)
    both = fc.sheet(fc.project_all(s, fc.ALL_FACES), scale=1)
    assert both.shape[0] == 4 + 6 + 3                     # sides row, then plan row
