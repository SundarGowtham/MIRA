#!/usr/bin/env bash
# Stage 6 (optional): only if Stages 1-5 are done. Train sft-corpus-full, Gate R, n32 sets.
set -u
cd /users/gsundar/projects/MIRA
for s in S1_gateR_sft-e3c-full S2_sample S3_rl S4_sample_e1 S4_sample_ref S5_sample; do
  [ -f "runs/phase20/STAGE_${s}.done" ] || { echo "Stage 6 skipped: ${s} not done"; exit 0; }
done
bash scripts/phase20_stages/S1_train.sh sft-corpus-full || exit 1
bash scripts/phase20_stages/S1_gateR.sh sft-corpus-full || exit 1
bash scripts/phase20_stages/_sample.sh stage6; rc=$?; [ $rc -eq 3 ] && exit 0; exit $rc
