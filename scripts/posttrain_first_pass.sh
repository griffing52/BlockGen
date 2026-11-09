#!/usr/bin/env bash
# First pass through the post-training ladder (docs/pipeline.md section 7) on the two
# models that exist: the voxel M2b SFT model (5M, seed 1) and the joint facade D4 model
# (20M). Rung 1, best-of-N with a verifier, and rung 2, rejection-sampling fine-tuning,
# each judged on a full card at the control n beside the raw model, a temperature arm
# where it is cheap, and the usual baselines and controls. Rewards are non-pillar
# signals (blockgen/posttrain): lcc26 for voxels, face_solidity for facades.
# Run detached: nohup bash scripts/posttrain_first_pass.sh > outputs/logs/posttrain_first_pass.log 2>&1 &
# Steps are idempotent: a finished best-of-N arm or temperature arm is reused. Never edit this file while it runs;
# bash reads a script incrementally and an edit under a running copy breaks it.
set -uo pipefail
export PYTHONPATH=.
PART=${PART:-both}             # voxel | facade | both
BATCH_F=${BATCH_F:-8}          # sampling batch; the GPU is shared and 32 does not fit beside an 11 GB tenant
BATCH_V=${BATCH_V:-8}          # voxel sequences run to 8,200 tokens, so the cache is larger than the facade one

V=outputs/run_20261015_235516_m2b_sft_s1
F=outputs/run_20261028_125240_houses32_facades_joint_d4_20m
F_RAW=$F/samples399/samples_step399_48.npz
V_RAW=$V/samples_step5500_48.npz
BASE_ARMS="baseline/facade_patchwork:outputs/bench_arms/facade_patchwork_32.npz control/real@facades6:outputs/bench_arms/real_facades6_32.npz"
newest() { ls -td outputs/run_*_"$1" 2>/dev/null | head -1; }
npz_in() { ls "$1"/*.npz 2>/dev/null | grep -v manifest | head -1; }

if [ "$PART" != facade ]; then
echo "== voxel: best-of-4 on lcc26, n=256 =="; date
V_BON=$(npz_in $V/bon_lcc26_k4_n256)
[ -z "$V_BON" ] && python -m blockgen.posttrain.bon --run $V --n 256 --k 4 --reward lcc26 --batch $BATCH_V
V_BON=$(npz_in $V/bon_lcc26_k4_n256)
echo "== voxel: rejection-sampling FT from the kept set =="; date
python -m blockgen.corpus.shards from-npz "$V_BON" --tier rft-m2b_sft_s1-lcc26 --kind voxels --tokenizer 60684456e598 --source bon \
 && python -m blockgen.train --config configs/train/rft-voxel-5m.yaml init_from=$V/best.pt train=rft-m2b_sft_s1-lcc26:1,sft-houses32:1 \
      tokenizer=60684456e598 trained_on_signals=lcc26 dropout=0.0 name=rft_m2b_sft_s1_lcc26 sample_batch=$BATCH_V
V_RFT=$(newest rft_m2b_sft_s1_lcc26)
V_ARMS="ar/m2b_sft_s1:$V_RAW bon/m2b_sft_s1_bon_lcc26_k4:$V_BON"
if [ -n "$V_RFT" ]; then
  python -m blockgen.train --sample $V_RFT --n 256 --out-dir $V_RFT/samples256 sample_batch=$BATCH_V
  V_RFT_NPZ=$(npz_in $V_RFT/samples256); [ -n "$V_RFT_NPZ" ] && V_ARMS="$V_ARMS posttrain/rft_m2b_sft_s1_lcc26:$V_RFT_NPZ"
fi
echo "== voxel card =="; date
python -m blockgen.eval.bench --tier both --n 256 --baselines --arms $V_ARMS \
  --name "voxel post-training: best-of-4 and rejection-sampling FT on 26-connected largest component against the M2b SFT seed 1 model"
fi
if [ "$PART" != voxel ]; then

echo "== facade: best-of-4 on face_solidity, n=399 =="; date
F_BON=$(npz_in $F/bon_face_solidity_k4_n399)
[ -z "$F_BON" ] && python -m blockgen.posttrain.bon --run $F --n 399 --k 4 --reward face_solidity --batch $BATCH_F
F_BON=$(npz_in $F/bon_face_solidity_k4_n399)
echo "== facade: temperature 0.7 arm =="; date
F_T07=$(npz_in $F/samples399_t0.7)
[ -z "$F_T07" ] && python -m blockgen.train --sample $F --n 399 --out-dir $F/samples399_t0.7 temperature=0.7 sample_batch=$BATCH_F
F_T07=$(npz_in $F/samples399_t0.7)
echo "== facade: rejection-sampling FT from the kept set =="; date
python -m blockgen.corpus.shards from-npz "$F_BON" --tier rft-houses32_facades_joint_d4_20m-face_solidity --kind facades --layout joint --tokenizer a4ca45ac1fe5 --source bon \
 && python -m blockgen.train --config configs/train/rft-facades-joint-d4-20m.yaml init_from=$F/best.pt \
      train=rft-houses32_facades_joint_d4_20m-face_solidity:1,sft-houses32:1 trained_on_signals=face_solidity \
      name=rft_facades_joint_d4_20m_face_solidity sample_batch=$BATCH_F
F_RFT=$(newest rft_facades_joint_d4_20m_face_solidity)
F_ARMS="ar/houses32_facades_joint_d4_20m:$F_RAW bon/houses32_facades_joint_d4_20m_bon_face_solidity_k4:$F_BON"
[ -n "$F_T07" ] && F_ARMS="$F_ARMS ar/houses32_facades_joint_d4_20m_t0.7:$F_T07"
if [ -n "$F_RFT" ]; then
  python -m blockgen.train --sample $F_RFT --n 399 --out-dir $F_RFT/samples399 sample_batch=$BATCH_F
  F_RFT_NPZ=$(npz_in $F_RFT/samples399); [ -n "$F_RFT_NPZ" ] && F_ARMS="$F_ARMS posttrain/rft_facades_joint_d4_20m_face_solidity:$F_RFT_NPZ"
fi
echo "== facade card =="; date
python -m blockgen.eval.bench --tier both --n 399 --arms $F_ARMS $BASE_ARMS \
  --name "facade post-training: best-of-4 on face solidity, temperature 0.7 and rejection-sampling FT against the raw 20M D4 model"
fi
echo "POSTTRAIN CHAIN EXIT $?"; date
