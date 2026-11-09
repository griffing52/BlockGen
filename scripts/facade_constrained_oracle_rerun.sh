#!/usr/bin/env bash
# Rerun of the six oracle completions from scripts/facade_constrained_battery.sh at a smaller
# batch: at B=32 the 32-box test houses OOM'd beside another session's bench job. Waits for
# the battery's card (WAIT_PID) to exit and for NEED_MB free, then runs the oracles at B=8.
set -uo pipefail
cd /home/griffing52/programming/gen/BlockGen
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
R=outputs/run_20261028_125240_houses32_facades_joint_d4_20m
WAIT_PID=${WAIT_PID:-214610}
NEED_MB=${NEED_MB:-7000}
B=${B:-8}
LOG=outputs/logs/facade_constrained_oracle.log
exec >>"$LOG" 2>&1
echo "== facade constrained oracle rerun (B=$B) =="; date
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
until [ "$(free_mb)" -ge "$NEED_MB" ]; do sleep 60; done
echo "GPU has $(free_mb) MB free; starting ($(date))"
S=scripts/facade_constrained.py
for T in 1.0 0.7; do
  for M in none constrain; do
    echo "== oracle -z $M T=$T =="; date
    $PY $S oracle --run $R --mode $M --temperature $T --given=-z --batch $B --out $R/oracle_front_${M}_t${T}
    echo "exit $?"
  done
done
for M in none constrain; do
  echo "== oracle -z,+x $M T=0.7 =="; date
  $PY $S oracle --run $R --mode $M --temperature 0.7 --given=-z,+x --batch $B --out $R/oracle_front_right_${M}_t0.7
  echo "exit $?"
done
echo "FCON ORACLE EXIT 0"; date
