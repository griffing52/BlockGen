#!/usr/bin/env bash
# Pause the ideas battery by stopping the process entirely — this frees ALL GPU
# memory (a SIGSTOP'd process would keep its VRAM allocated, which is useless if
# you need the GPU for something else). Progress is safe: every finished arm has
# its results + model checkpoint on disk, and ./scripts/resume_ideas.sh restarts
# with the same stamp, skipping finished arms. You lose at most the arm that was
# mid-training.
set -euo pipefail

pids=$(pgrep -f "[e]xperiments_ideas --stamp" || true)
if [ -z "$pids" ]; then
  echo "no ideas battery running"
  exit 0
fi
kill $pids
sleep 3
pgrep -f "[e]xperiments_ideas --stamp" >/dev/null && kill -9 $pids 2>/dev/null || true
echo "paused (pid $pids stopped, GPU freed). Resume with: ./scripts/resume_ideas.sh"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
