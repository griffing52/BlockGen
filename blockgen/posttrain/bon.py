"""Best-of-N: sample ``n * k`` builds, keep the reward's favourite of every ``k``.

    python -m blockgen.posttrain.bon --run <run_dir> --n 256 --k 8 --reward face_solidity
                                     [--weights face_solidity:2,novelty_penalty:0.5]
                                     [--temperature T] [--top-k K] [--ckpt best.pt] [--out-dir D]

Rung one of the ladder (see the package docstring): no training, so it is the
floor every trained method has to beat at the same ``n``. The kept builds are a
structure cache the bench scores like any other arm, with the parent run's
``train_split``, ``model`` and ``checkpoint`` carried into the report and the
reward's signals appended to the parent's ``trained_on_signals``; ``scores.parquet``
beside it holds every candidate's score and whether it was kept. ``--weights``
selects on a weighted composite of rewards instead of ``--reward``; the pillar
inputs of BlockScore are not offered, because they cannot be scored here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from blockgen.train.sample import sample_structures
from blockgen.utils.data import Structure

from . import rewards as rw


def draw(model, tokenizer, cfg, n: int, seed: Optional[int] = None) -> Tuple[List[Structure], dict]:
    """``sample_structures`` behind one name the tests can monkeypatch on this module."""
    return sample_structures(model, tokenizer, cfg, n, seed)


def select(scores: Sequence[float], k: int) -> Tuple[List[int], List[dict]]:
    """Winner of every group of ``k`` consecutive candidates (ties to the first).

    Returns the kept indices and one row per candidate: group, index, score,
    kept. A short last group (the sampler dropped empty builds) still elects a
    winner, so the kept count is ``ceil(len(scores) / k)``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    kept, rows = [], []
    for g0 in range(0, len(scores), k):
        g = g0 // k
        best = g0 + int(np.argmax(scores[g0:g0 + k]))
        kept.append(best)
        for i in range(g0, min(g0 + k, len(scores))):
            rows.append({"group": g, "index": i, "score": float(scores[i]), "kept": i == best})
    return kept, rows


def parent_report(run_dir: str, cfg, ckpt: str, tokenizer) -> dict:
    """The fields a sample report carries from the run that produced the model.

    ``model`` is the architecture dict (``tools/lab/standings.model_label`` renders
    it as ``arch/preset d_model x layers``), ``checkpoint`` the full path so the
    bench can walk back to the ``run_`` directory, and ``train_split`` the same
    inference the trainer records. ``parent_signals`` are what the parent itself
    was trained on, so a best-of-N over an RFT model is not read as reward-free.
    """
    if hasattr(cfg, "split_name"):
        split = cfg.split_name()
    else:
        split = getattr(cfg, "train_split", None) or ("canonical" if list(cfg.train) == ["sft-houses32"] else ",".join(cfg.train))
    model = dict(getattr(cfg, "model", None) or {})
    model.update({k: getattr(cfg, k) for k in ("arch", "preset", "d_model", "nhead", "layers", "ff", "dropout", "pe")
                  if hasattr(cfg, k)})
    return {"run": run_dir, "name": cfg.name, "checkpoint": os.path.join(run_dir, ckpt), "train_split": split,
            "model": model, "tokenizer": tokenizer.hash, "kind": tokenizer.kind, "seed": cfg.seed,
            "temperature": cfg.temperature, "top_k": cfg.top_k, "tags": cfg.sample_tags,
            "parent_signals": [str(x) for x in (getattr(cfg, "trained_on_signals", None) or [])]}


def out_dir_for(run_dir: str, reward: str, k: int, n: int, out_dir: Optional[str] = None,
                method: str = "bon") -> str:
    """``<run>/<method>_<reward>_k<k>_n<n>``, suffixed ``_2``, ``_3``, ... when that
    directory already holds an arm, so a second draw never overwrites the first."""
    base = os.path.join(out_dir or run_dir, f"{method}_{reward}_k{k}_n{n}")
    out, i = base, 1
    while os.path.isdir(out) and os.listdir(out):
        i += 1
        out = f"{base}_{i}"
    return out


def arm_spec(cfg_name: str, reward: str, k: int, npz: str) -> str:
    """The ``--arms`` entry for ``python -m blockgen.eval.bench``."""
    return f"bon/{cfg_name}_bon_{reward}_k{k}:{npz}"


def run(run_dir: str, n: int, k: int, reward_name: str, temperature: Optional[float] = None,
        top_k: Optional[int] = None, ckpt: str = "best.pt", out_dir: Optional[str] = None,
        seed: Optional[int] = None, reward: Optional[rw.Reward] = None, loaded=None,
        method: str = "bon", batch: Optional[int] = None) -> dict:
    """Sample, score, keep, save. Returns the report with ``npz``, ``scores`` and ``arms`` added.

    ``loaded`` is ``(model, tokenizer, cfg)`` for a caller that already has them;
    otherwise ``blockgen.train.trainer.load_run`` opens ``ckpt`` of ``run_dir``.
    """
    import pandas as pd
    from blockgen.curation.houses import save_house_cache
    if loaded is None:
        from blockgen.train.trainer import load_run
        loaded = load_run(run_dir, ckpt)
    model, tokenizer, cfg = loaded
    if temperature is not None:
        cfg.temperature = float(temperature)
    if top_k is not None:
        cfg.top_k = int(top_k)
    if batch is not None:
        cfg.sample_batch = int(batch)
    reward = reward or rw.get(reward_name)
    out = out_dir_for(run_dir, reward.name, k, n, out_dir, method)
    os.makedirs(out, exist_ok=True)

    t0 = time.time()
    structs, extra = draw(model, tokenizer, cfg, n * k, seed)
    t_sample = time.time() - t0
    scores = reward.score(structs) if structs else np.zeros(0)
    kept_idx, rows = select(scores, k)
    kept = [structs[i] for i in kept_idx]
    t_total = time.time() - t0

    report = parent_report(run_dir, cfg, ckpt, tokenizer)
    report["seed"] = extra.get("seed", cfg.seed if seed is None else seed)   # the seed the draw used
    own = rw.signals_of(reward)
    signals = [x for x in report["parent_signals"] if x not in own] + own
    report.update({"method": "bon", "k": k, "n": n, "reward": reward.name, "trained_on_signals": signals,
                   "reward_weights": dict(getattr(reward, "weights", None) or {reward.name: 1.0}),
                   "n_requested": n * k, "n_drawn": len(structs), "n_kept": len(kept),
                   "score_kept_mean": float(np.mean(scores[kept_idx])) if kept_idx else None,
                   "score_all_mean": float(np.mean(scores)) if len(scores) else None,
                   "seconds_sample": round(t_sample, 1), "seconds": round(t_total, 1),
                   "sample_per_s": round(len(structs) / max(t_sample, 1e-9), 2)})
    if "lengths" in extra:
        report["median_tokens"] = float(np.median(extra["lengths"])) if extra["lengths"] else None
    pd.DataFrame(rows, columns=["group", "index", "score", "kept"]).to_parquet(os.path.join(out, "scores.parquet"), index=False)
    npz = save_house_cache(kept, tokenizer.max_dim, cache_dir=out, name="bon", report=report) if kept else None
    report["npz"] = npz
    report["scores"] = os.path.join(out, "scores.parquet")
    report["arms"] = arm_spec(cfg.name, reward.name, k, npz) if npz else None
    json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=1)
    print(f"[bon] {len(structs)} drawn in {t_sample:.0f}s ({report['sample_per_s']} builds/s), kept {len(kept)} of "
          f"{len(structs)} by {reward.name}: mean {report['score_all_mean']} -> {report['score_kept_mean']}", flush=True)
    if npz:
        print(f"[bon] --arms {report['arms']}", flush=True)
    return report


def reward_from_args(reward: Optional[str], weights: Optional[str]) -> rw.Reward:
    """``--weights`` wins over ``--reward``; neither means ``face_solidity``."""
    if weights:
        return rw.get("composite", weights=rw.parse_weights(weights))
    return rw.get(reward or "face_solidity")


def add_reward_args(ap: argparse.ArgumentParser, default: Optional[str] = "face_solidity") -> None:
    ap.add_argument("--reward", default=default, choices=list(rw.CLI_REWARDS))
    ap.add_argument("--weights", default=None, metavar="name:w,...",
                    help="select on a weighted composite of rewards instead of --reward")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=256, help="builds to keep (groups)")
    ap.add_argument("--k", type=int, default=8, help="candidates per kept build")
    add_reward_args(ap)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None, help="builds per sampling batch (TrainConfig.sample_batch); lower on a shared GPU")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    reward = reward_from_args(a.reward, a.weights)
    run(a.run, a.n, a.k, reward.name, a.temperature, a.top_k, a.ckpt, a.out_dir, a.seed, reward=reward, batch=a.batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
