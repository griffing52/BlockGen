#!/usr/bin/env bash
# The f=4 oracle test of the cascade-and-AR hybrid (notes §74 option a), then the stacked
# constrained battery (scripts/facade_constrained_stack.sh). Waits for the GPU tenant WAIT_PID
# to exit and NEED_MB free, trains the pooled4 per-face model (5M, 8000 steps, about five
# minutes), runs the oracle on the 399 test front faces, then hands the GPU to the stack.
# Log: outputs/logs/facade_hybrid.log. Run detached from a copy.
set -uo pipefail
cd /home/griffing52/programming/gen/BlockGen
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
WAIT_PID=${WAIT_PID:-293001}
NEED_MB=${NEED_MB:-9000}
LOG=outputs/logs/facade_hybrid.log
mkdir -p outputs/logs
exec >>"$LOG" 2>&1
echo "== facade hybrid chain =="; date
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "tenant $WAIT_PID gone ($(date))"
waited=0
until [ "$(free_mb)" -ge "$NEED_MB" ]; do
  sleep 60; waited=$((waited + 60))
  if [ $waited -ge 43200 ]; then echo "gave up waiting for the GPU"; echo "FHYB CHAIN EXIT 99"; exit 99; fi
done
echo "GPU has $(free_mb) MB free; starting ($(date))"
echo "== train pooled4 5M =="; date
$PY -m blockgen.train --config configs/train/houses32-facades-pooled4.yaml
echo "exit $?"
RUN=$(ls -td outputs/run_*_houses32_facades_pooled4_5m | head -1)
echo "run $RUN"
echo "== hybrid oracle, front face =="; date
$PY scripts/facade_hybrid_oracle.py --run $RUN --n 399 --temperatures 1.0,0.7 --batch 32 --face=-z
echo "exit $?"
echo "FHYB CHAIN EXIT 0"; date
echo "== handing over to the stack =="
exec bash scripts/facade_constrained_stack.sh
