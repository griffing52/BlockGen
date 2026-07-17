#!/usr/bin/env bash
# Resume the most recent (or given) ideas battery run. Waits for the GPU to be
# free, then replays the exact original command with the same stamp — arms whose
# results are already on disk are skipped, so only the interrupted arm re-runs.
#   ./scripts/resume_ideas.sh                     # latest run
#   ./scripts/resume_ideas.sh 20260704_090000     # specific stamp
set -euo pipefail
cd "$(dirname "$0")/.."

FREE_MB="${FREE_MB:-2500}"
POLL_S="${POLL_S:-30}"

if [ $# -ge 1 ]; then
  OUT="outputs/run_${1}_ideas"
else
  OUT=$(ls -dt outputs/run_*_ideas 2>/dev/null | head -1 || true)
fi
if [ -z "${OUT:-}" ] || [ ! -f "$OUT/cmd.txt" ]; then
  echo "no resumable run found (need $OUT/cmd.txt)"; exit 1
fi
if pgrep -f "[e]xperiments_ideas --stamp" >/dev/null; then
  echo "an ideas battery is already running"; exit 1
fi

echo "waiting for GPU to free up (used < ${FREE_MB} MiB)..."
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "${used:-99999}" -lt "$FREE_MB" ] && break
  echo "$(date '+%H:%M:%S')  GPU used=${used} MiB"
  sleep "$POLL_S"
done

echo "resuming: $(cat "$OUT/cmd.txt")"
nohup bash -c "$(cat "$OUT/cmd.txt")" >> "$OUT/run.log" 2>&1 &
echo "resumed pid $! ; tail -f $OUT/run.log"
