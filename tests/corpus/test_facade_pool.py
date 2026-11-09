import numpy as np
import pytest

from blockgen.corpus import facade as fc

STONE, PLANK, GLASS = 1, 5, 20


def _face(depth, ids, data=None):
    depth = np.asarray(depth, np.int16)
    ids = np.asarray(ids, np.int16)
    data = np.zeros_like(ids) if data is None else np.asarray(data, np.int16)
    return fc.Facade(face="-z", depth=depth, ids=ids, data=data, extent=8)


def test_factor_one_is_the_face_itself():
    f = _face([[0, 1], [2, -1]], [[STONE, STONE], [PLANK, 0]])
    assert fc.pool(f, 1, "any") is f


def test_any_keeps_one_cell_features_and_the_nearest_depth():
    depth = np.full((4, 4), -1); depth[1, 2] = 3; depth[1, 3] = 5
    ids = np.zeros((4, 4)); ids[1, 2] = GLASS; ids[1, 3] = PLANK
    p = fc.pool(_face(depth, ids), 2, "any")
    assert p.depth.shape == (2, 2)
    assert p.mask.tolist() == [[False, True], [False, False]]
    assert p.depth[0, 1] == 3                      # nearest surface of the block
    assert p.ids[0, 1] in (GLASS, PLANK)           # a tie; the mode picks the smaller
    assert p.extent == 8


def test_majority_drops_a_lone_cell_and_votes_on_depth_and_block():
    depth = np.full((4, 4), -1); depth[1, 2] = 3
    p = fc.pool(_face(depth, np.where(depth >= 0, GLASS, 0)), 2, "majority")
    assert not p.mask.any()
    depth = np.array([[2, 2], [2, 7]]); ids = np.array([[PLANK, PLANK], [PLANK, GLASS]])
    p = fc.pool(_face(depth, ids), 2, "majority")
    assert p.depth[0, 0] == 2 and p.ids[0, 0] == PLANK


def test_edge_blocks_count_only_in_bounds_children():
    # a 3 x 3 face pooled by 2: the far blocks hold 2, 2 and 1 real children
    depth = np.zeros((3, 3), np.int16); ids = np.full((3, 3), STONE, np.int16)
    maj = fc.pool(_face(depth, ids), 2, "majority")
    cen = fc.pool(_face(depth, ids), 2, "center")
    assert maj.mask.all() and cen.mask.all()


def test_data_survives_pooling():
    depth = np.zeros((2, 2), np.int16); ids = np.full((2, 2), PLANK, np.int16)
    data = np.full((2, 2), 3, np.int16)
    p = fc.pool(_face(depth, ids, data), 2, "any")
    assert p.ids[0, 0] == PLANK and p.data[0, 0] == 3


def test_upsample_repeats_and_crops_to_shape():
    depth = np.array([[0, 1]]); ids = np.array([[STONE, PLANK]])
    up = fc.upsample(_face(depth, ids), 2, (2, 3))
    assert up.depth.tolist() == [[0, 0, 1], [0, 0, 1]]
    assert up.ids[1, 2] == PLANK


def test_rejects_unknown_rule():
    with pytest.raises(ValueError):
        fc.pool(_face([[0]], [[STONE]]), 2, "mean")
