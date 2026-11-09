"""select=reward on a one-weight model: the trainer hook samples both checkpoints and keeps the reward's pick."""
import os

import numpy as np
import pytest
import torch
from torch import nn

from blockgen.posttrain import rewards as rw
from blockgen.posttrain import select as sel
from blockgen.train import trainer as tr
from blockgen.utils.data import Structure


class Knob(nn.Module):
    """A model whose only weight is the edge of the cube it 'samples'."""
    def __init__(self, edge: float):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(edge)))


class EdgeReward:
    name = "edge"
    pillar = False

    def score(self, structs):
        return np.array([s.shape[0] for s in structs], dtype=float)


def _fake_sample_structures(model, tokenizer, cfg, n, seed=None):
    e = int(round(float(model.weight.item())))
    ids = np.full((e, e, e), 5, np.int32)
    return [Structure(block_ids=ids, block_data=np.zeros_like(ids)) for _ in range(n)], {"lengths": [e] * n}


def _cfg():
    from blockgen.train.config import TrainConfig
    return TrainConfig(name="fake", train=["sft-houses32"], select="reward", trained_on_signals=["edge"], n_samples=2)


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(sel, "sample_structures", _fake_sample_structures)
    monkeypatch.setattr(rw, "REWARDS", {**rw.REWARDS, "edge": EdgeReward})


def test_reward_selector_is_registered_with_the_trainer():
    assert tr.SELECT_FNS["reward"] is sel.select_reward
    assert tr.resolve_select("reward") is sel.select_reward


def test_trainer_resolves_select_reward_in_a_fresh_interpreter():
    """What ``python -m blockgen.train ... select=reward`` does: nothing imports posttrain first."""
    import subprocess, sys
    code = ("from blockgen.train import trainer as tr; fn = tr.resolve_select('reward'); "
            "print(fn.__module__ + '.' + fn.__name__)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                         env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "blockgen.posttrain.select.select_reward"


def test_select_reward_loads_best_when_it_scores_higher(tmp_path, patched):
    model = Knob(3)                                                   # final weights sample 3-cubes
    torch.save({"model": Knob(9).state_dict(), "val": {"eval-holdout": 0.5}, "step": 100}, tmp_path / "best.pt")
    val, step = tr.select_checkpoint(model, str(tmp_path), _cfg(), {"nll": 0.5, "step": 100}, 200,
                                     {"eval-holdout": 0.6}, "cpu", tokenizer=None)
    assert (val, step) == ({"eval-holdout": 0.5}, 100) and model.weight.item() == 9.0


def test_select_reward_keeps_final_weights_when_they_win(tmp_path, patched):
    model = Knob(9)
    torch.save({"model": Knob(3).state_dict(), "val": {"eval-holdout": 0.5}, "step": 100}, tmp_path / "best.pt")
    val, step = sel.select_reward(model, str(tmp_path), _cfg(), {"nll": 0.5, "step": 100}, 200,
                                  {"eval-holdout": 0.6}, "cpu")
    assert (val, step) == ({"eval-holdout": 0.6}, 200) and model.weight.item() == 9.0
    assert model.training                                              # mode restored after sampling


def test_select_reward_without_a_best_checkpoint_or_signal(tmp_path, patched):
    model = Knob(4)
    val, step = sel.select_reward(model, str(tmp_path), _cfg(), {"nll": 1.0, "step": 200}, 200, {"eval-holdout": 0.7}, "cpu")
    assert (val, step) == ({"eval-holdout": 0.7}, 200) and not os.path.exists(tmp_path / "best.pt")
    cfg = _cfg(); cfg.trained_on_signals = []
    with pytest.raises(ValueError):
        sel.reward_for(cfg)
    cfg.trained_on_signals = ["coherence"]
    with pytest.raises(rw.PillarReward):                               # selection on a pillar is refused too
        sel.reward_for(cfg)


def test_select_by_reward_and_composite_of_signals(patched):
    cube = lambda e: Structure(block_ids=np.full((e, e, e), 5, np.int32), block_data=np.zeros((e, e, e), np.int32))
    assert sel.select_by_reward({"a.pt": [cube(3)], "b.pt": [cube(9), cube(2)], "c.pt": []}, EdgeReward()) == "b.pt"
    assert sel.select_by_reward({"a.pt": [cube(5)], "b.pt": [cube(5)]}, EdgeReward()) == "a.pt"
    cfg = _cfg(); cfg.trained_on_signals = ["edge", "lcc26"]
    r = sel.reward_for(cfg)
    assert r.name == "composite-edge-lcc26" and not r.pillar
    assert r.score([cube(4)])[0] == pytest.approx(4.0 + 1.0)
    cfg.trained_on_signals = ["edge@0.5", "lcc26@2"]            # weighted tokens rebuild the same composite
    r = sel.reward_for(cfg)
    assert r.name == "composite-edge@0.5-lcc26@2" and rw.signals_of(r) == cfg.trained_on_signals
    assert r.score([cube(4)])[0] == pytest.approx(0.5 * 4.0 + 2.0 * 1.0)
    cfg.trained_on_signals = ["edge@2"]                          # one weighted token is still a composite
    assert sel.reward_for(cfg).score([cube(4)])[0] == pytest.approx(8.0)
