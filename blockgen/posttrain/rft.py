"""Rejection-sampling fine-tuning: best-of-N selection, then a derived tier and the command to train on it.

    python -m blockgen.posttrain.rft --run <run_dir> --reward face_solidity --n 2048 --keep-frac 0.25
                                     [--weights face_solidity:2,novelty_penalty:0.5]
                                     [--temperature T] [--top-k K] [--ckpt best.pt] [--out-dir D]

``--n`` builds are drawn and ``keep-frac`` of them kept, as the winner of every
group of ``round(1 / keep_frac)``; the kept builds become the shard tier
``rft-<run name>-<reward>`` through ``blockgen.corpus.shards.from_structures``,
and the printed ``python -m blockgen.train`` command fine-tunes the parent on that
tier mixed one-to-one with the SFT tier, at the parent's dropout (T59). The
reward's signals are recorded in ``trained_on_signals`` (``name[@weight]`` tokens,
see ``rewards.signal_token``) so the honesty rule in the package docstring can be
checked on the registry row and ``select=reward`` can rebuild the same reward. Importing this module registers
``select=reward`` (``select.py``) with the trainer, for a run that wants the
checkpoint the reward prefers rather than the lowest-nll one.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional, Sequence

from blockgen.corpus.shards import from_structures
from blockgen.utils.data import Structure

from . import bon
from . import rewards as rw
from .select import select_by_reward  # noqa: F401  importing registers select=reward with the trainer

#: Which rft config fine-tunes which kind of model.
RFT_CONFIGS = {"voxels": "configs/train/rft-voxel-5m.yaml", "facades": "configs/train/rft-facades-joint-d4-20m.yaml"}

#: The sample-tag groups that are conditioning choices and travel onto the kept
#: builds. ``size`` is left out so ``from_structures`` infers it per build.
CONDITIONING_TAGS = ("src", "q", "win")


def tier_name(run_name: str, reward: str) -> str:
    return f"rft-{run_name}-{reward}"


def k_for(keep_frac: float) -> int:
    if not 0 < keep_frac <= 1:
        raise ValueError("keep-frac must be in (0, 1]")
    return max(1, int(round(1.0 / keep_frac)))


def train_command(run_dir: str, cfg, tier: str, tok_hash: str, reward: str, kind: str = "voxels",
                  ckpt: str = "best.pt", signals: Optional[Sequence[str]] = None) -> str:
    """The fine-tuning command, with the parent's dropout spelled out.

    ``reward`` is the reward's name (the tier and run name carry it); ``signals``
    are its ``trained_on_signals`` tokens, by default just the name. Both go on
    the command line as comma lists, which is the form ``config._coerce`` reads.
    """
    config = RFT_CONFIGS.get(kind, RFT_CONFIGS["voxels"])
    sig = ",".join(signals or [reward])
    return (f"python -m blockgen.train --config {config} init_from={os.path.join(run_dir, ckpt)} "
            f"train={tier}:1,sft-houses32:1 tokenizer={tok_hash} trained_on_signals={sig} "
            f"dropout={cfg.dropout} name=rft_{cfg.name}_{reward}")


def run(run_dir: str, reward_name: str, n: int = 2048, keep_frac: float = 0.25, temperature: Optional[float] = None,
        top_k: Optional[int] = None, ckpt: str = "best.pt", out_dir: Optional[str] = None,
        seed: Optional[int] = None, loaded=None, reward: Optional[rw.Reward] = None,
        batch: Optional[int] = None) -> dict:
    """Best-of-N, then the derived tier and the training command.

    The tier is written by ``from_structures`` with its defaults for ``faces``
    (all six) and ``augment`` (none), which is what the SFT facade shard the
    command mixes it with uses; a model trained on another face set would need
    its own call. Only the conditioning tag groups (``CONDITIONING_TAGS``) are
    copied from the parent's ``sample_tags``; ``size`` is inferred per build.
    """
    k = k_for(keep_frac)
    groups = max(1, n // k)
    reward = reward or rw.get(reward_name)
    if loaded is None:
        from blockgen.train.trainer import load_run
        loaded = load_run(run_dir, ckpt)
    model, tokenizer, cfg = loaded
    report = bon.run(run_dir, groups, k, reward_name, temperature, top_k, ckpt, out_dir, seed, reward=reward,
                     loaded=loaded, method="rft", batch=batch)
    tier = tier_name(cfg.name, reward.name)
    report.update({"method": "rft", "keep_frac": keep_frac, "tier": tier})
    if report["npz"] is None:
        print("[rft] nothing kept; no tier written", flush=True)
        return report
    from blockgen.curation.houses import load_structures_from_cache
    kept, _ = load_structures_from_cache(report["npz"])
    tags = {g: str(v) for g, v in dict(cfg.sample_tags).items() if g in CONDITIONING_TAGS}
    shard = from_structures(kept, tier=tier, tok=tokenizer, tags=tags, source=report["npz"])
    report["shard"] = shard
    report["train_command"] = train_command(run_dir, cfg, tier, tokenizer.hash, reward.name, tokenizer.kind, ckpt,
                                            signals=rw.signals_of(reward))
    print(f"[rft] tier {tier} -> {shard}", flush=True)
    print(f"[rft] {report['train_command']}", flush=True)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    bon.add_reward_args(ap, default=None)
    ap.add_argument("--n", type=int, default=2048, help="builds to draw")
    ap.add_argument("--keep-frac", type=float, default=0.25)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None, help="builds per sampling batch; lower on a shared GPU")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    if not (a.reward or a.weights):
        ap.error("one of --reward or --weights is required")
    reward = bon.reward_from_args(a.reward, a.weights)
    run(a.run, reward.name, a.n, a.keep_frac, a.temperature, a.top_k, a.ckpt, a.out_dir, a.seed, reward=reward, batch=a.batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
