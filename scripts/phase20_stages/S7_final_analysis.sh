#!/usr/bin/env bash
# Final analysis: every P-20 prediction and measurement (CPU), after all sample
# sets are accepted and Stage 4 scoring is done. Marker-resumable.
set -u
cd /users/gsundar/projects/MIRA
export PYTHONPATH=. PYTHONHASHSEED=0
V="$HOME/envs/mira-vllm/bin/python"
for j in stage2 stage4_e1 stage4_ref stage5; do
  "$V" research/phase20_verify_samples.py "$j" >/dev/null 2>&1 || { echo "[final] $j not accepted"; exit 1; }
done
[ -f results/phase20/p19_4/dimension.json ] || { echo "[final] Stage 4 scoring missing"; exit 1; }
# the SFT-side run (if it did not happen yet) is folded into the final run
bash scripts/phase20_cpu.sh final
