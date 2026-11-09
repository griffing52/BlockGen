#!/usr/bin/env bash
# M2 at 20M: the M2b voxel recipe (pretrain on anneal, SFT on houses32, select=best) at the
# 20m preset, seed 1, so the comparison against m2b_sft_s1 changes model size only. Then the
# same post-training floor as T71 (best-of-4 on lcc26) and a card at n=256 with the 5M arms.
# Waits for the voxel step-4000 card and the facade best-of-4 card to finish.
# Run detached from a copy.
set -uo pipefail
export PYTHONPATH=.
B=${B:-16}
V=outputs/run_20261015_235516_m2b_sft_s1
npz_in() { ls "$1"/*.npz 2>/dev/null | grep -v manifest | head -1; }
newest() { ls -td outputs/run_*_"$1" 2>/dev/null | head -1; }
until grep -q "S4000 CHAIN EXIT" outputs/logs/posttrain_voxel_s4000.log && grep -q "FBON CHAIN EXIT" outputs/logs/posttrain_facade_bon_t07.log; do sleep 120; done

echo "== pretrain 20m =="; date
[ -z "$(newest m2_20m_pretrain_s1)" ] && python -m blockgen.train --config configs/train/anneal-pretrain-20m.yaml seed=1 name=m2_20m_pretrain_s1
PRE=$(newest m2_20m_pretrain_s1)
[ -f "$PRE/last.pt" ] || { echo "no pretrain checkpoint in $PRE"; echo "M2_20M CHAIN EXIT 1"; exit 1; }

echo "== SFT 20m =="; date
python -m blockgen.train --config configs/train/houses32-sft-20m.yaml seed=1 name=m2b_20m_sft_s1 init_from=$PRE/last.pt note="pretrain=$PRE"
SFT=$(newest m2b_20m_sft_s1)

echo "== sample 256 and best-of-4 lcc26 =="; date
python -m blockgen.train --sample $SFT --n 256 --out-dir $SFT/samples256 sample_batch=$B
python -m blockgen.posttrain.bon --run $SFT --n 256 --k 4 --reward lcc26 --batch $B \
    --ckpt $(python -c "from blockgen.train.__main__ import selected_ckpt; print(selected_ckpt('$SFT'))")

echo "== voxel card =="; date
python -m blockgen.eval.bench --tier both --n 256 --baselines \
  --arms ar/m2b_sft_s1:$V/samples_step5500_48.npz \
         bon/m2b_sft_s1_bon_lcc26_k4:$(npz_in $V/bon_lcc26_k4_n256) \
         ar/m2b_20m_sft_s1:$(npz_in $SFT/samples256) \
         bon/m2b_20m_sft_s1_bon_lcc26_k4:$(npz_in $SFT/bon_lcc26_k4_n256) \
  --name "voxel M2b at 20M against 5M, raw and best-of-4"
echo "M2_20M CHAIN EXIT $?"; date
