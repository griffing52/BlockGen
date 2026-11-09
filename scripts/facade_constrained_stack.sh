#!/usr/bin/env bash
# Stacks the cross-face constraint (results T117) on the two levers that beat it: best-of-4 on
# face_solidity at T=0.7 (T77) and the T73 RFT model. Waits for the GPU tenant WAIT_PID to exit
# and for NEED_MB free, then:
#   1. constrained best-of-4 on the base 20M D4 model (n=399, k=4, T=0.7)
#   2. constrained sampling of the RFT model (n=399, T=0.7)
#   3. constrained best-of-4 on the RFT model
#   4. one card: the three new arms beside T77 best-of-4, T73 RFT, constrained base (T117),
#      grammar control, patchwork and the six-face ceiling
# Log: outputs/logs/facade_constrained_stack.log. Run detached from a copy.
set -uo pipefail
cd /home/griffing52/programming/gen/BlockGen
export PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
R=outputs/run_20261028_125240_houses32_facades_joint_d4_20m
FR=outputs/run_20261101_154120_rft_facades_joint_d4_20m_face_solidity_rsel_t07
WAIT_PID=${WAIT_PID:-293001}
NEED_MB=${NEED_MB:-12000}
B=${B:-32}
LOG=outputs/logs/facade_constrained_stack.log
mkdir -p outputs/logs
exec >>"$LOG" 2>&1
echo "== facade constrained stack =="; date
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "tenant $WAIT_PID gone ($(date))"
waited=0
until [ "$(free_mb)" -ge "$NEED_MB" ]; do
  sleep 60; waited=$((waited + 60))
  if [ $waited -ge 43200 ]; then echo "gave up waiting for the GPU"; echo "FSTACK CHAIN EXIT 99"; exit 99; fi
done
echo "GPU has $(free_mb) MB free; starting ($(date))"
S=scripts/facade_constrained.py
echo "== constrained best-of-4, base, T=0.7 =="; date
$PY $S bon --run $R --mode constrain --temperature 0.7 --n 399 --k 4 --reward face_solidity --batch $B --out $R/constrained_bon_face_solidity_k4_t0.7_n399
echo "exit $?"
echo "== constrained sample, RFT, T=0.7 =="; date
$PY $S sample --run $FR --mode constrain --temperature 0.7 --n 399 --batch $B --out $FR/constrained_constrain_t0.7_n399
echo "exit $?"
echo "== constrained best-of-4, RFT, T=0.7 =="; date
$PY $S bon --run $FR --mode constrain --temperature 0.7 --n 399 --k 4 --reward face_solidity --batch $B --out $FR/constrained_bon_face_solidity_k4_t0.7_n399
echo "exit $?"
npz_in() { ls "$1"/*.npz 2>/dev/null | grep -v manifest | grep -v faces | head -1; }
echo "== facade card =="; date
$PY -m blockgen.eval.bench --tier both --n 399 \
  --arms decode/joint_d4_20m_grammar_t0.7:$(npz_in $R/constrained_none_t0.7_n399) \
         decode/joint_d4_20m_constrained_t0.7:$(npz_in $R/constrained_constrain_t0.7_n399) \
         bon/houses32_facades_joint_d4_20m_bon_face_solidity_k4_t0.7:$(npz_in $R/t0.7/bon_face_solidity_k4_n399) \
         bon/joint_d4_20m_constrained_bon_face_solidity_k4_t0.7:$(npz_in $R/constrained_bon_face_solidity_k4_t0.7_n399) \
         posttrain/rft_facades_joint_d4_20m_face_solidity_t0.7:$(npz_in $FR/samples399) \
         posttrain/rft_joint_d4_20m_constrained_t0.7:$(npz_in $FR/constrained_constrain_t0.7_n399) \
         posttrain/rft_joint_d4_20m_constrained_bon_k4_t0.7:$(npz_in $FR/constrained_bon_face_solidity_k4_t0.7_n399) \
         baseline/facade_patchwork:outputs/bench_arms/facade_patchwork_32.npz control/real@facades6:outputs/bench_arms/real_facades6_32.npz \
  --name "facade constrained decoding stacked with best-of-4 and the RFT model"
echo "FSTACK CHAIN EXIT $?"; date
