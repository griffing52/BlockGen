#!/usr/bin/env bash
# Cross-face constrained decoding on the 20M D4 joint facade model (results T68 named it the
# next lever). Waits for the GPU tenant named in WAIT_PID (another session's run) to exit and
# for NEED_MB free, then:
#   1. four sample arms at n=399 on the T68 boxes: grammar-only control and constrained, at T=1.0 and T=0.7
#   2. oracle completions of the 399 test houses from the real front face (and from front + right),
#      control and constrained, at both temperatures
#   3. one facade card with the four new arms beside the T68 raw, T70 T=0.7, T77 best-of-4 and T73 RFT arms,
#      patchwork and the six-face ceiling
# Log: outputs/logs/facade_constrained.log. Run detached from a copy.
set -uo pipefail
cd /home/griffing52/programming/gen/BlockGen
export PYTHONPATH=.
PY=.venv/bin/python
R=outputs/run_20261028_125240_houses32_facades_joint_d4_20m
FR=outputs/run_20261101_154120_rft_facades_joint_d4_20m_face_solidity_rsel_t07
WAIT_PID=${WAIT_PID:-45321}
NEED_MB=${NEED_MB:-6000}
B=${B:-32}
LOG=outputs/logs/facade_constrained.log
mkdir -p outputs/logs
exec >>"$LOG" 2>&1
echo "== facade constrained battery =="; date

free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "tenant $WAIT_PID gone ($(date))"
waited=0
until [ "$(free_mb)" -ge "$NEED_MB" ]; do
  sleep 60; waited=$((waited + 60))
  if [ $waited -ge 43200 ]; then echo "gave up waiting for the GPU"; echo "FCON CHAIN EXIT 99"; exit 99; fi
done
echo "GPU has $(free_mb) MB free; starting ($(date))"

S=scripts/facade_constrained.py
for T in 1.0 0.7; do
  for M in none constrain; do
    echo "== sample $M T=$T =="; date
    $PY $S sample --run $R --mode $M --temperature $T --n 399 --batch $B --out $R/constrained_${M}_t${T}_n399
    echo "exit $?"
  done
done
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

npz_in() { ls "$1"/*.npz 2>/dev/null | grep -v manifest | grep -v faces | head -1; }
echo "== facade card =="; date
$PY -m blockgen.eval.bench --tier both --n 399 \
  --arms ar/houses32_facades_joint_d4_20m:$R/samples399/samples_step399_48.npz \
         ar/houses32_facades_joint_d4_20m_t0.7:$(npz_in $R/samples399_t0.7) \
         decode/joint_d4_20m_grammar:$(npz_in $R/constrained_none_t1.0_n399) \
         decode/joint_d4_20m_constrained:$(npz_in $R/constrained_constrain_t1.0_n399) \
         decode/joint_d4_20m_grammar_t0.7:$(npz_in $R/constrained_none_t0.7_n399) \
         decode/joint_d4_20m_constrained_t0.7:$(npz_in $R/constrained_constrain_t0.7_n399) \
         bon/houses32_facades_joint_d4_20m_bon_face_solidity_k4_t0.7:$(npz_in $R/t0.7/bon_face_solidity_k4_n399) \
         posttrain/rft_facades_joint_d4_20m_face_solidity_t0.7:$(npz_in $FR/samples399) \
         baseline/facade_patchwork:outputs/bench_arms/facade_patchwork_32.npz control/real@facades6:outputs/bench_arms/real_facades6_32.npz \
  --name "facade cross-face constrained decoding against free decoding and the post-training arms"
echo "FCON CHAIN EXIT $?"; date
