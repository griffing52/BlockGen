#!/usr/bin/env bash
# Best-of-4 on face_solidity at temperature 0.7 on the 20M D4 facade parent (results T70 ran it
# at 1.0), then a facade card holding every facade post-training arm at n=399. Waits for the
# GPU (voxel step-4000 sampling) and for the voxel card to finish before its own card.
# Run detached from a copy.
set -uo pipefail
export PYTHONPATH=.
B=${B:-16}
F=outputs/run_20261028_125240_houses32_facades_joint_d4_20m
FR=outputs/run_20261101_154120_rft_facades_joint_d4_20m_face_solidity_rsel_t07
VLOG=outputs/logs/posttrain_voxel_s4000.log
npz_in() { ls "$1"/*.npz 2>/dev/null | grep -v manifest | head -1; }
until grep -q "== voxel card ==\|S4000 CHAIN EXIT" $VLOG; do sleep 60; done
echo "== facade best-of-4 at T=0.7 =="; date
[ -z "$(npz_in $F/t0.7/bon_face_solidity_k4_n399)" ] && python -m blockgen.posttrain.bon --run $F --n 399 --k 4 \
    --reward face_solidity --temperature 0.7 --batch $B --out-dir $F/t0.7
until grep -q "S4000 CHAIN EXIT" $VLOG; do sleep 60; done
echo "== facade card =="; date
python -m blockgen.eval.bench --tier both --n 399 \
  --arms ar/houses32_facades_joint_d4_20m:$F/samples399/samples_step399_48.npz \
         ar/houses32_facades_joint_d4_20m_t0.7:$(npz_in $F/samples399_t0.7) \
         bon/houses32_facades_joint_d4_20m_bon_face_solidity_k4:$F/bon_face_solidity_k4_n399/bon_48.npz \
         bon/houses32_facades_joint_d4_20m_bon_face_solidity_k4_t0.7:$(npz_in $F/t0.7/bon_face_solidity_k4_n399) \
         posttrain/rft_facades_joint_d4_20m_face_solidity:$(npz_in outputs/run_20261031_114825_rft_facades_joint_d4_20m_face_solidity/samples399) \
         posttrain/rft_facades_joint_d4_20m_face_solidity_t0.7:$(npz_in $FR/samples399) \
         baseline/facade_patchwork:outputs/bench_arms/facade_patchwork_32.npz control/real@facades6:outputs/bench_arms/real_facades6_32.npz \
  --name "facade rung 1 and 2 at temperature 0.7: best-of-4 against rejection-sampling FT"
echo "FBON CHAIN EXIT $?"; date
