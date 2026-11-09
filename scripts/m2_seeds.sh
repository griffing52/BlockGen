#!/usr/bin/env bash
# M2 pilot at 5M: matched-compute scratch vs pretrain+SFT, three seeds (docs/pipeline.md §9).
# Each seed: scratch for 28k steps; pretrain 20k steps on anneal; SFT 8k steps from it.
set -u
cd "$(dirname "$0")/.."
for SEED in ${SEEDS:-0 1 2}; do
  python -m blockgen.train --config configs/train/houses32-scratch-matched.yaml seed=$SEED name=m2_scratch_s$SEED \
      > outputs/logs/m2_scratch_s$SEED.log 2>&1
  python -m blockgen.train --config configs/train/anneal-pretrain-5m.yaml seed=$SEED name=m2_pretrain_s$SEED \
      > outputs/logs/m2_pretrain_s$SEED.log 2>&1
  PRE=$(grep -o "outputs/run_[0-9_]*_m2_pretrain_s$SEED" outputs/logs/m2_pretrain_s$SEED.log | head -1)
  python -m blockgen.train --config configs/train/houses32-sft-5m.yaml seed=$SEED name=m2_sft_s$SEED init_from=$PRE/last.pt note="pretrain=$PRE" \
      > outputs/logs/m2_sft_s$SEED.log 2>&1
done
