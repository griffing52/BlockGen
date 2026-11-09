"""Cross-face constraints: real faces are consistent under them, contradictions are caught,
and the constrained sampler returns consistent face sets from a model that knows nothing."""
import numpy as np
import pytest
import torch

from blockgen.corpus import facade as fc
from blockgen.corpus import facade_constraints as fcc
from blockgen.corpus.tokenizer import CorpusTokenizer
from blockgen.utils.data import Structure


def _house():
    ids = np.zeros((7, 5, 6), np.int32)
    ids[1:6, 0, 1:5] = 4                      # floor
    ids[1:6, 1:4, 1] = 5; ids[1:6, 1:4, 4] = 5  # front and back walls
    ids[1, 1:4, 1:5] = 5; ids[5, 1:4, 1:5] = 5  # side walls
    ids[3, 1:3, 1] = 0                        # a door in the front wall
    ids[2:5, 4, 2:4] = 17                     # a roof ridge
    ids[6, 2, 0] = 20                         # a loose block only one face sees first
    return Structure(block_ids=ids, block_data=np.zeros_like(ids))


def test_real_faces_are_consistent_and_reproject_exactly():
    s = _house()
    faces = fc.project_all(s, fc.ALL_FACES)
    assert fcc.consistent(faces, s.shape, order=fc.ALL_FACES)
    assert fcc.consistent(faces, s.shape, order=list(reversed(fc.ALL_FACES)))
    r = fcc.reprojection(faces, s.shape)
    assert r["cell_disagree"] == 0 and r["hidden"] == 0 and r["filled_in"] == 0


def test_contradictions_are_caught():
    s = _house()
    faces = fc.project_all(s, fc.ALL_FACES)
    f = faces["+x"]
    bad = fc.Facade(f.face, f.depth.copy(), f.ids.copy(), f.data.copy(), f.extent)
    filled = np.argwhere(bad.mask)
    u, v = filled[0]
    bad.depth[u, v] = min(bad.depth[u, v] + 2, bad.extent - 1)   # the first hit moved behind what -z saw
    faces2 = dict(faces); faces2["+x"] = bad
    assert not fcc.consistent(faces2, s.shape, order=fc.ALL_FACES) or fcc.reprojection(faces2, s.shape)["cell_disagree"] > 0


def test_allowed_blocks_skipping_past_a_known_voxel():
    st = fcc.ShellState((4, 3, 5))
    st.commit_cell("-z", 1, 1, 2, key=7)       # -z sees a block at z=2 in column x=1,y=1 (z=0,1 empty)
    allowed, forced, allow_empty = st.allowed("+z", 1, 1)   # +z looks back along the same column
    assert allow_empty is False
    # from +z, depth 0..2 are z=4,3,2: z=2 is the known block, z=3,4 are free; nothing behind z=2
    assert allowed.tolist() == [True, True, True, False, False]
    assert forced.tolist()[2] == 7 and forced[0] == fcc.FREE
    allowed_x, _, empty_x = st.allowed("-x", 1, 1)          # -x row y=1,z=1 passes the empty voxel at x=1
    assert empty_x is True and not allowed_x[1] and allowed_x[0] and allowed_x[2]
    assert st.determined("+z", 1, 1) is False
    st.commit_cell("+z", 1, 1, 0, key=9)                    # +z claims z=4 instead; now -x through z=4 ...
    assert st.known[1, 1, 4] == 9


class _Uniform(torch.nn.Module):
    """A model with no opinion: uniform logits, no cache, so only the masks shape the sample."""

    def __init__(self, vocab_size, max_seq_len=4000):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size

    def forward(self, tokens, pad_mask=None):
        return torch.zeros(tokens.shape[0], tokens.shape[1], self.vocab_size) + self.w


@pytest.mark.parametrize("constrain", [True, False])
def test_constrained_sampler_is_consistent_and_grammatical(constrain):
    from blockgen.train.sample_constrained import decode_builds, face_size, generate_faces_constrained
    tok = CorpusTokenizer.default(kind="facades", layout="joint")
    model = _Uniform(tok.vocab_size)
    tags = {"src": "curated", "q": "4", "size": "32", "win": "whole"}
    shapes = [(5, 4, 6), (3, 3, 3)]
    plans = [[(f, *face_size(f, s)) for f in fc.ALL_FACES] for s in shapes]
    seqs, info = generate_faces_constrained(model, tok, tags, plans, shapes, temperature=1.0, top_k=0, seed=0,
                                            batch=2, constrain=constrain)
    builds, per_build = decode_builds(seqs, shapes, tok)
    assert info["cells"] == sum(sum(U * V for _, U, V in p) for p in plans)
    for faces, shape, plan in zip(per_build, shapes, plans):
        assert set(faces) == set(fc.ALL_FACES)
        for f, U, V in plan:
            assert faces[f].depth.shape == (U, V)
            assert (faces[f].depth < shape[fc._parse(f)[0]]).all()
        if constrain:
            assert fcc.consistent(faces, shape, order=fc.ALL_FACES)
            assert fcc.reprojection(faces, shape)["cell_disagree"] == 0
    if constrain:
        assert info["forced_cells"] > 0


def test_given_face_is_kept_verbatim_and_respected():
    from blockgen.train.sample_constrained import decode_builds, face_size, generate_faces_constrained
    tok = CorpusTokenizer.default(kind="facades", layout="joint")
    model = _Uniform(tok.vocab_size)
    s = _house()
    real = fc.project_all(s, fc.ALL_FACES)
    tags = {"src": "curated", "q": "4", "size": "32", "win": "whole"}
    plans = [[(f, *face_size(f, s.shape)) for f in fc.ALL_FACES]]
    seqs, _ = generate_faces_constrained(model, tok, tags, plans, [s.shape], 1.0, 0, seed=1, batch=1,
                                         constrain=True, given=[{"-z": real["-z"]}])
    _, per_build = decode_builds(seqs, [s.shape], tok)
    g = per_build[0]["-z"]
    assert np.array_equal(g.depth, real["-z"].depth) and np.array_equal(g.ids, real["-z"].ids)
    assert fcc.consistent(per_build[0], s.shape, order=fc.ALL_FACES)
