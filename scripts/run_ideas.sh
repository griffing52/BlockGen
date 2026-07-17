#!/usr/bin/env bash
# Launch the ideas ablation battery once the GPU is free (queue it right after the
# overnight battery: both wait on VRAM, so starting this while that runs is safe).
#   FREE_MB=1500 EPOCHS_AR=120 ./scripts/run_ideas.sh
#   ./scripts/run_ideas.sh --groups pe,ordering      # pass extra flags through
set -euo pipefail
cd "$(dirname "$0")/.."

FREE_MB="${FREE_MB:-2500}"
POLL_S="${POLL_S:-60}"
# Epoch budgets are injected as explicit flags ONLY when set via env, so a
# --config <name> passthrough isn't silently overridden (explicit CLI flags win
# over config values). Unset -> the config (or the entry point default) decides.
EPOCH_FLAGS=()
[ -n "${EPOCHS_AR:-}" ]   && EPOCH_FLAGS+=(--epochs-ar "$EPOCHS_AR")
[ -n "${EPOCHS_DIFF:-}" ] && EPOCH_FLAGS+=(--epochs-diff "$EPOCHS_DIFF")

echo "waiting for GPU to free up (used < ${FREE_MB} MiB)..."
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "$(date '+%H:%M:%S')  GPU used=${used} MiB"
  [ "${used:-99999}" -lt "$FREE_MB" ] && break
  sleep "$POLL_S"
done

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT="outputs/run_${STAMP}_ideas"
mkdir -p "$OUT"
# persist the exact invocation so resume_ideas.sh can replay it (same stamp ->
# finished arms are skipped)
printf '%q ' .venv/bin/python -m blockgen.experiments_ideas \
  --stamp "$STAMP" ${EPOCH_FLAGS[@]+"${EPOCH_FLAGS[@]}"} "$@" > "$OUT/cmd.txt"
echo "GPU free — launching ideas battery -> $OUT/run.log"
nohup .venv/bin/python -m blockgen.experiments_ideas \
  --stamp "$STAMP" ${EPOCH_FLAGS[@]+"${EPOCH_FLAGS[@]}"} "$@" \
  >> "$OUT/run.log" 2>&1 &
echo "launched pid $! ; tail -f $OUT/run.log"
echo "pause (frees ALL GPU memory):  ./scripts/pause_ideas.sh"
echo "resume (skips finished arms):  ./scripts/resume_ideas.sh"
