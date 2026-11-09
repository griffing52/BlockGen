"""The face cascade model: shapes, the sampler's invariants, oracle-coarse starts."""
import numpy as np
import torch

from blockgen.corpus import facade_levels as fl
from blockgen.models import face_cascade as mc


def _model():
    torch.manual_seed(0)
    return mc.FaceCascade(n_blocks=9, dim=32, layers=1, heads=4)


def _inbox(boxes):
    return {f: torch.as_tensor(np.stack([fl.inbox_grid((int(b[0]), int(b[1])), f) for b in boxes])) for f in fl.FACTORS}


def test_forward_shapes_and_loss():
    m = _model()
    B, f = 3, 4
    G = fl.grid_size(f)
    pd, pb = m.none_parent(B, G, "cpu")
    od, ob = m.masked_own(B, G, "cpu")
    inbox = torch.ones((B, G, G), dtype=torch.bool)
    box = torch.tensor([[13, 10, 9]] * B)
    ld, lb = m(f, pd, pb, od, ob, inbox, box)
    assert ld.shape == (B, G, G, fl.N_DEPTH) and lb.shape == (B, G, G, 9)
    td = torch.randint(0, 10, (B, G, G))
    tb = torch.where(td > 0, torch.randint(1, 9, (B, G, G)), 0)
    loss, parts = mc.level_loss(m, f, pd, pb, od, ob, inbox, box, td, tb, inbox)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert parts["n_depth"] == B * G * G


def test_sample_level_respects_box_and_extent():
    m = _model()
    boxes = torch.tensor([[13, 10, 5], [32, 32, 32], [3, 2, 1]])
    f = 2
    G = fl.grid_size(f)
    inbox = _inbox(boxes.numpy())[f]
    pd, pb = m.none_parent(3, G, "cpu")
    gen = torch.Generator().manual_seed(0)
    for steps in (1, 3):
        d, b = mc.sample_level(m, f, pd, pb, inbox, boxes, steps=steps, generator=gen)
        assert d.shape == (3, G, G)
        assert not d[~inbox].any() and not b[~inbox].any()           # outside the box: see-through
        assert (d <= boxes[:, 2].view(3, 1, 1)).all()                # 1 + depth <= extent
        assert (b[d == 0] == 0).all() and (b[d > 0] != fl.UNK).all()  # blocks only where filled, never UNK
        assert (d[2] <= 1).all()                                       # extent 1 -> depth 0 only


def test_cascade_draws_every_level_and_honours_a_start():
    m = _model()
    boxes = torch.tensor([[13, 10, 9], [8, 8, 8]])
    inbox = _inbox(boxes.numpy())
    gen = torch.Generator().manual_seed(1)
    out = mc.cascade(m, boxes, inbox, steps=1, generator=gen)
    assert list(out) == list(fl.FACTORS)
    for f, (d, b) in out.items():
        assert d.shape == (2, fl.grid_size(f), fl.grid_size(f))
    fixed = (torch.full((2, 8, 8), 3), torch.full((2, 8, 8), 2))
    fixed = (torch.where(inbox[4], fixed[0], 0), torch.where(inbox[4], fixed[1], 0))
    out2 = mc.cascade(m, boxes, inbox, steps=1, generator=gen, start=(4, *fixed))
    assert set(out2) == {4, 2, 1}
    assert torch.equal(out2[4][0], fixed[0])


def test_sampling_is_reproducible_under_a_seed():
    m = _model()
    boxes = torch.tensor([[13, 10, 9]])
    inbox = _inbox(boxes.numpy())
    a = mc.cascade(m, boxes, inbox, steps=2, generator=torch.Generator().manual_seed(7))
    b = mc.cascade(m, boxes, inbox, steps=2, generator=torch.Generator().manual_seed(7))
    assert all(torch.equal(a[f][0], b[f][0]) for f in a)
