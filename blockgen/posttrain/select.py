"""``select=reward``: keep the checkpoint whose samples the trained-on reward scores highest.

Registered with the trainer's ``SELECT_FNS`` hook under the name ``reward``
(``blockgen.train.trainer.register_select``), so a run that sets ``select=reward``
and ``trained_on_signals=<reward>`` has its final and ``best.pt`` weights both
sampled (``n_samples`` draws each, the run's seed) and keeps the set with the
higher mean reward. The hook follows ``select_checkpoint``: it loads the chosen
weights into ``model`` and returns their validation nll and step, so the trainer
samples, benchmarks and registers the selected checkpoint like any other.

Selecting on the reward is within the honesty rule because the reward is
non-pillar by construction (``rewards.get`` refuses pillar inputs); the
selection is recorded by the registry row's ``select`` and ``trained_on_signals``.
Importing this module is what registers the hook.
"""
from __future__ import annotations

import os
from typing import Dict, Sequence

import numpy as np
import torch

from blockgen.train import trainer as tr
from blockgen.train.sample import sample_structures
from blockgen.utils.data import Structure

from . import rewards as rw


def reward_for(cfg) -> rw.Reward:
    """The reward a run trained on, rebuilt from its ``trained_on_signals`` tokens
    (``name[@weight]``): one plain token is that reward, anything else the composite."""
    names = [str(x) for x in (cfg.trained_on_signals or []) if str(x)]
    if not names:
        raise ValueError("select=reward needs trained_on_signals=<reward> on the run")
    return rw.from_signals(names)


def mean_reward(structs: Sequence[Structure], reward: rw.Reward) -> float:
    """Mean score, or -inf for a checkpoint that produced nothing to score."""
    return float(np.mean(reward.score(structs))) if len(structs) else float("-inf")


def select_by_reward(candidates: Dict[str, Sequence[Structure]], reward: rw.Reward) -> str:
    """The candidate whose samples score highest on average; ties go to the first."""
    means = {c: mean_reward(s, reward) for c, s in candidates.items()}
    return max(means, key=lambda c: (means[c], -list(means).index(c)))


def _draw(model, tokenizer, cfg, n: int):
    was_training = model.training
    model.eval()
    try:
        return sample_structures(model, tokenizer, cfg, n, seed=cfg.seed)[0]
    finally:
        model.train(was_training)


@tr.register_select("reward")
def select_reward(model, run_dir: str, cfg, best: dict, step: int, final_nll: dict, device, tokenizer=None):
    """Trainer hook: final weights versus ``best.pt`` by mean reward on ``n_samples`` draws."""
    reward = reward_for(cfg)
    n = max(int(cfg.n_samples), 1)
    samples = {"last.pt": _draw(model, tokenizer, cfg, n)}
    vals = {"last.pt": (final_nll, step)}
    path = os.path.join(run_dir, "best.pt")
    if best.get("step") != step and os.path.exists(path):
        bk = torch.load(path, map_location=device, weights_only=False)
        final_sd = {k: v.detach().clone() for k, v in tr.unwrap(model).state_dict().items()}
        tr.load_weights(model, bk["model"])
        samples["best.pt"] = _draw(model, tokenizer, cfg, n)
        vals["best.pt"] = (bk["val"], best["step"])
        winner = select_by_reward(samples, reward)
        if winner == "last.pt":
            tr.load_weights(model, final_sd)
    else:
        winner = "last.pt"
    means = {c: mean_reward(s, reward) for c, s in samples.items()}
    print(f"[train] select=reward ({reward.name}, n={n}): "
          + ", ".join(f"{c} step {vals[c][1]} mean {means[c]:.4f}" for c in samples) + f" -> {winner}", flush=True)
    return vals[winner]
