#!/usr/bin/env bash
# Stage 4 scoring (GPU ~1 h): G, E, F1 for sft-rssft-full -> E1-full@300 with
# the stage-4 step table. Needs the E1-full n32 sets (Block G counts) and the
# sft-rssft-full reference (P-20-3's "sampled" label); deferred until both pass.
set -u
cd /users/gsundar/projects/MIRA
export PYTHONPATH=. PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V="$HOME/envs/mira-vllm/bin/python"
for j in stage2 stage4_e1 stage4_ref; do
  "$V" research/phase20_verify_samples.py "$j" >/dev/null 2>&1 || { echo "[S4_score] $j not accepted yet -> deferred"; exit 3; }
done
[ -f results/phase20/p19_2b/location.json ] || { echo "[S4_score] waits for Stage 2b's Block E cache -> deferred"; exit 3; }
for b in G E F1; do
  uv run python -u research/phase20_p19.py --stage 4 --block "$b" 2>&1 | tee -a run_logs/phase20_p19_4.log
  [ "${PIPESTATUS[0]}" -eq 0 ] || exit 1
done
