#!/usr/bin/env bash
# The step-2 battery of research.md H.7: the fixed-pooling face cascade on front faces.
# Waits until the GPU has room (the BrickNet Qwen pretrain of 2026-11-02 holds 11 GB), runs a
# one-minute GPU smoke test, then four runs in order, each trained on the side faces of the
# houses_32 train split and scored on the 399 test front faces (stats.md, samples.png,
# completion.png in each run dir):
#   1. 5M, `any` pooling                 the main arm
#   2. 5M, `majority` pooling            the control from T75 (deletes thin features)
#   3. 5M, `any`, no parent corruption   the NCM corruption ablation
#   4. 20M, `any`                        capacity
# Scheduled for 07:00 America/Los_Angeles on 2026-11-03 with a user systemd timer
# (systemctl --user list-timers | grep facade-cascade). Log: outputs/logs/facade_cascade_morning.log.
set -uo pipefail
cd /home/griffing52/programming/gen/BlockGen
export PYTHONPATH=.
PY=.venv/bin/python
NEED_MB=${NEED_MB:-9000}
LOG=outputs/logs/facade_cascade_morning.log
mkdir -p outputs/logs
exec >>"$LOG" 2>&1
echo "== facade cascade morning battery =="; date

free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
waited=0
until [ "$(free_mb)" -ge "$NEED_MB" ]; do
  if [ $((waited % 1800)) -eq 0 ]; then echo "waiting for GPU: $(free_mb) MB free, need $NEED_MB ($(date))"; fi
  sleep 60; waited=$((waited + 60))
  if [ $waited -ge 43200 ]; then echo "gave up after 12 h waiting for the GPU"; echo "CASCADE CHAIN EXIT 99"; exit 99; fi
done
echo "GPU has $(free_mb) MB free; starting ($(date))"

echo "== smoke =="; date
$PY scripts/facade_cascade.py train --limit 40 --steps 20 --eval-every 10 --batch 8 --dim 32 --layers 1 --n 16 \
    --name smoke_cascade --out outputs/smoke/facade_cascade_gpu_smoke || { echo "SMOKE FAILED"; echo "CASCADE CHAIN EXIT 1"; exit 1; }

run() {  # name, args...
  local name=$1; shift
  echo "== $name =="; date
  $PY scripts/facade_cascade.py train --name "$name" "$@"
  echo "exit $? for $name"; date
}
run facade_cascade_5m            --preset 5m  --rule any
run facade_cascade_5m_majority   --preset 5m  --rule majority
run facade_cascade_5m_nocorrupt  --preset 5m  --rule any --corrupt 0
run facade_cascade_20m           --preset 20m --rule any --batch 48   # batch 96 runs out of memory on 16 GB
echo "CASCADE CHAIN EXIT 0"; date
