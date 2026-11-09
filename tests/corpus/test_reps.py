import numpy as np
import pytest

from blockgen.corpus import reps
from blockgen.utils.data import Structure


def _house():
    ids = np.zeros((9, 6, 7), np.int32)
    ids[1:8, 0, 1:6] = 1
    ids[1, 0:5, 1:6] = ids[7, 0:5, 1:6] = 5
    ids[1:8, 0:5, 1] = ids[1:8, 0:5, 5] = 5
    ids[1:8, 5, 1:6] = 1
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


def test_registry_names_and_lookup():
    assert reps.names()[0] == "voxels" and "facades" in reps.names()
    with pytest.raises(KeyError):
        reps.get("holograms")


def test_facades_round_trip_and_picture_without_a_renderer():
    r = reps.get("facades")
    assert not r.needs_renderer
    s = _house()
    enc = r.encode(s)
    back = r.decode(enc)
    assert back.shape == s.crop_to_non_air().shape
    assert np.array_equal(back.block_ids, s.crop_to_non_air().block_ids)   # a hollow box is all visible
    pic = reps.picture("facades", s, px=400)
    assert pic.dtype == np.uint8 and pic.ndim == 3 and pic.shape[2] == 3
    assert 200 <= pic.shape[1] <= 400


def test_voxels_is_the_identity_on_a_crop():
    r = reps.get("voxels")
    s = _house()
    assert r.needs_renderer
    assert r.decode(r.encode(s)).shape == s.crop_to_non_air().shape
