#!/usr/bin/env bash
# Launch the overnight experiment battery once the GPU is free.
#
# Waits until GPU memory use drops below $FREE_MB (another job finished), then runs
# blockgen.experiments_overnight with nohup so it survives a disconnect, logging to
# outputs/run_<stamp>_overnight/run.log. Override any knob via env, e.g.
#   FREE_MB=1500 EPOCHS_AR=120 ./scripts/run_overnight.sh
#   ./scripts/run_overnight.sh --datasets gc-houses-large   # pass extra flags through
set -euo pipefail
cd "$(dirname "$0")/.."

FREE_MB="${FREE_MB:-2500}"          # consider the GPU free when used memory < this
POLL_S="${POLL_S:-60}"
EPOCHS_AR="${EPOCHS_AR:-90}"
EPOCHS_DIFF="${EPOCHS_DIFF:-130}"
EPOCHS_GRAPH="${EPOCHS_GRAPH:-60}"

echo "waiting for GPU to free up (used < ${FREE_MB} MiB)..."
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "$(date '+%H:%M:%S')  GPU used=${used} MiB"
  [ "${used:-99999}" -lt "$FREE_MB" ] && break
  sleep "$POLL_S"
done

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="outputs/run_${STAMP}_overnight"
mkdir -p "$OUT"
echo "GPU free — launching battery -> $OUT/run.log"
nohup .venv/bin/python -m blockgen.experiments_overnight \
  --stamp "$STAMP" --epochs-ar "$EPOCHS_AR" --epochs-diff "$EPOCHS_DIFF" \
  --epochs-graph "$EPOCHS_GRAPH" "$@" \
  > "$OUT/run.log" 2>&1 &
echo "launched pid $! ; tail -f $OUT/run.log"
