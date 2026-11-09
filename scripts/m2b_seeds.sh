#!/usr/bin/env bash
# M2b at 5M (T59): the M2 pilot rerun with the two protocol fixes. SFT keeps the
# pretraining dropout (0) and runs 16k steps; scratch gets the matched 36k steps;
# both arms are early-stopped (select=best) on holdout NLL. The three pretrain
# checkpoints from M2 are reused as they are (they were trained at dropout 0).
set -u
cd "$(dirname "$0")/.."
for SEED in ${SEEDS:-0 1 2}; do
  PRE=$(grep -o "outputs/run_[0-9_]*_m2_pretrain_s$SEED" outputs/logs/m2_pretrain_s$SEED.log | head -1)
  python -m blockgen.train --config configs/train/houses32-sft-5m.yaml seed=$SEED name=m2b_sft_s$SEED init_from=$PRE/last.pt note="pretrain=$PRE" \
      > outputs/logs/m2b_sft_s$SEED.log 2>&1
  python -m blockgen.train --config configs/train/houses32-scratch-matched.yaml seed=$SEED name=m2b_scratch_s$SEED \
      > outputs/logs/m2b_scratch_s$SEED.log 2>&1
done
