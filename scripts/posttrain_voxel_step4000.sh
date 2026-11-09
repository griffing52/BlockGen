#!/usr/bin/env bash
# The voxel cell T73 missed (results.md T73, correction): the reward-selected step-4,000
# weights of the select=reward rerun, drawn at temperature 1.0 and 0.7. T73's voxel arm was
# drawn by `--sample` before it followed summary.json's selected_ckpt, so it was step 500 at
# 0.7; it stays on this card under that name. Run detached from a copy.
set -uo pipefail
export PYTHONPATH=.
B=${B:-16}
V=outputs/run_20261015_235516_m2b_sft_s1
VR=outputs/run_20261101_134001_rft_m2b_sft_s1_lcc26_rsel_t07
npz_in() { ls "$1"/*.npz 2>/dev/null | grep -v manifest | head -1; }
for T in 1.0 0.7; do
  echo "== step 4000 at T=$T =="; date
  [ -z "$(npz_in $VR/samples256_s4000_t$T)" ] && python -m blockgen.train --sample $VR --ckpt last.pt --n 256 \
      --out-dir $VR/samples256_s4000_t$T temperature=$T sample_batch=$B
done
echo "== voxel card =="; date
python -m blockgen.eval.bench --tier both --n 256 --baselines \
  --arms ar/m2b_sft_s1:$V/samples_step5500_48.npz \
         ar/m2b_sft_s1_t0.7:$(npz_in $V/samples256_t0.7) \
         bon/m2b_sft_s1_bon_lcc26_k4:$(npz_in $V/bon_lcc26_k4_n256) \
         posttrain/rft_m2b_sft_s1_lcc26_s500:$(npz_in outputs/run_20261031_102252_rft_m2b_sft_s1_lcc26/samples256) \
         posttrain/rft_m2b_sft_s1_lcc26_s500_t0.7:$(npz_in $VR/samples256) \
         posttrain/rft_m2b_sft_s1_lcc26_s4000:$(npz_in $VR/samples256_s4000_t1.0) \
         posttrain/rft_m2b_sft_s1_lcc26_s4000_t0.7:$(npz_in $VR/samples256_s4000_t0.7) \
  --name "voxel rung 2: step-4000 reward-selected fine-tune at temperature 1.0 and 0.7"
echo "S4000 CHAIN EXIT $?"; date
