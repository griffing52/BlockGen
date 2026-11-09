"""Post-training (docs/pipeline.md section 7): the ladder above SFT, cheapest rung first.

1. Best-of-N with a verifier at inference (``bon.py``). No training; the number
   every trained method has to beat.
2. Rejection-sampling fine-tuning (``rft.py``). Sample, keep what the reward
   ranks highest, write the kept builds as a derived tier and fine-tune on it
   mixed with the SFT tier.
3. GRPO with verifiable rewards: groups of 4 to 8, KL to the SFT reference.
4. Preference tuning: DPO or SimPO on lab 2AFC pairs and reward-ranked pairs.
5. Merge and distil: average per-reward checkpoints, distil into one model.

Rewards (``rewards.py``) are pure CPU functions of a list of structures. The
honesty rule every rung is held to:

    A post-trained arm may be rewarded only on signals that are not BlockScore
    pillar inputs. The trained-on signal is reported as a method-internal metric,
    never as a BlockScore gain. Evaluation is the full BlockScore at n equal to
    the control n under houses32-v1 with the real@ controls, and claims are made
    only for pillars the reward did not touch. Because 26-connected connectivity
    correlates with the 6-connected coherence pillar, movement on coherence is
    pre-registered as expected transfer, not independent evidence.

``rewards.get`` enforces the first sentence: a reward flagged ``pillar`` is
refused by default and cannot score even when constructed, and the signals a run
was trained on travel with its samples as ``trained_on_signals``. ``select.py``
registers ``select=reward`` with the trainer; importing this package does it,
which is how ``trainer.resolve_select`` finds the hook for a ``python -m
blockgen.train ... select=reward`` run.
"""
from . import select as _select  # noqa: F401  registers select=reward with the trainer
