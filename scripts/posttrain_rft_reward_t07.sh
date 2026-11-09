#!/usr/bin/env bash
# Second pass on rung 2 (results.md T71, T72): both rejection-sampling fine-tunes rerun with
# select=reward and temperature 0.7. The reward selector samples the final weights and the
# likelihood-best checkpoint (n_samples draws each, at the config's temperature) and keeps the
# set with the higher mean trained-on reward; the end-of-run and card samples are also at 0.7.
# Same data, steps, lr and seed as the first pass, so the weights match it and only the
# checkpoint choice and the decode change. A raw voxel arm at 0.7 separates temperature from
# the fine-tune (the facade one exists from T70).
# Run detached from a copy: bash reads scripts incrementally.
set -uo pipefail
export PYTHONPATH=.
B=${B:-16}
V=outputs/run_20261015_235516_m2b_sft_s1
F=outputs/run_20261028_125240_houses32_facades_joint_d4_20m
npz_in() { ls "$1"/*.npz 2>/dev/null | grep -v manifest | head -1; }
newest() { ls -td outputs/run_*_"$1" 2>/dev/null | head -1; }
SEL="select=reward temperature=0.7 n_samples=64 sample_batch=$B"

echo "== voxel RFT, select=reward, T=0.7 =="; date
python -m blockgen.train --config configs/train/rft-voxel-5m.yaml init_from=$V/best.pt \
  train=rft-m2b_sft_s1-lcc26:1,sft-houses32:1 tokenizer=60684456e598 trained_on_signals=lcc26 dropout=0.0 \
  name=rft_m2b_sft_s1_lcc26_rsel_t07 $SEL
VR=$(newest rft_m2b_sft_s1_lcc26_rsel_t07)
python -m blockgen.train --sample $VR --n 256 --out-dir $VR/samples256 temperature=0.7 sample_batch=$B
echo "== voxel raw at T=0.7 =="; date
[ -z "$(npz_in $V/samples256_t0.7)" ] && python -m blockgen.train --sample $V --n 256 --out-dir $V/samples256_t0.7 temperature=0.7 sample_batch=$B
echo "== voxel card =="; date
python -m blockgen.eval.bench --tier both --n 256 --baselines \
  --arms ar/m2b_sft_s1:$V/samples_step5500_48.npz \
         ar/m2b_sft_s1_t0.7:$(npz_in $V/samples256_t0.7) \
         bon/m2b_sft_s1_bon_lcc26_k4:$(npz_in $V/bon_lcc26_k4_n256) \
         posttrain/rft_m2b_sft_s1_lcc26:$(npz_in $(newest rft_m2b_sft_s1_lcc26)/samples256) \
         posttrain/rft_m2b_sft_s1_lcc26_rsel_t0.7:$(npz_in $VR/samples256) \
  --name "voxel rung 2 rerun: rejection-sampling FT selected by reward and sampled at temperature 0.7"

echo "== facade RFT, select=reward, T=0.7 =="; date
python -m blockgen.train --config configs/train/rft-facades-joint-d4-20m.yaml init_from=$F/best.pt \
  train=rft-houses32_facades_joint_d4_20m-face_solidity:1,sft-houses32:1 trained_on_signals=face_solidity \
  name=rft_facades_joint_d4_20m_face_solidity_rsel_t07 batch_tokens=8192 grad_accum=4 $SEL
FR=$(newest rft_facades_joint_d4_20m_face_solidity_rsel_t07)
python -m blockgen.train --sample $FR --n 399 --out-dir $FR/samples399 temperature=0.7 sample_batch=$B
echo "== facade card =="; date
python -m blockgen.eval.bench --tier both --n 399 \
  --arms ar/houses32_facades_joint_d4_20m:$F/samples399/samples_step399_48.npz \
         ar/houses32_facades_joint_d4_20m_t0.7:$(npz_in $F/samples399_t0.7) \
         bon/houses32_facades_joint_d4_20m_bon_face_solidity_k4:$F/bon_face_solidity_k4_n399/bon_48.npz \
         posttrain/rft_facades_joint_d4_20m_face_solidity:$(npz_in $(newest rft_facades_joint_d4_20m_face_solidity)/samples399) \
         posttrain/rft_facades_joint_d4_20m_face_solidity_rsel_t0.7:$(npz_in $FR/samples399) \
         baseline/facade_patchwork:outputs/bench_arms/facade_patchwork_32.npz control/real@facades6:outputs/bench_arms/real_facades6_32.npz \
  --name "facade rung 2 rerun: rejection-sampling FT selected by reward and sampled at temperature 0.7"
echo "RSEL CHAIN EXIT $?"; date
