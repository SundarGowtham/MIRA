#!/usr/bin/env bash
# Phase 16: waits for E1's ASTRAL n=32 generation to finish (GPU held by
# e1_astral_gen in the main env), then immediately launches E1's held-out
# n=32 sampling (data_curation/phase16_e1_heldout_sampling.py, mira-vllm
# env) without requiring anyone to be watching. Self-contained chain so
# the next job starts the moment the GPU frees up.
set -u
cd /users/gsundar/projects/MIRA
NTFY_TOPIC="mira-g5x7k2-status"
ASTRAL_LOG="run_logs/phase16_e1_astral_gen.log"
HELDOUT_LOG="run_logs/phase16_e1_heldout_gen.log"

notify() { curl -s -m 10 -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true; }

notify "MIRA chain: waiting for E1 ASTRAL gen to finish before launching E1 held-out sampling"

until grep -q "EXIT_CODE" "$ASTRAL_LOG" 2>/dev/null; do
    sleep 20
done

RC=$(grep "EXIT_CODE" "$ASTRAL_LOG" | tail -1 | grep -oE '[0-9]+')
echo "=== E1 ASTRAL gen finished rc=${RC} $(date -u) ===" | tee -a "$HELDOUT_LOG"
notify "MIRA chain: E1 ASTRAL gen done (rc=${RC}). Launching E1 held-out n=32 sampling now."

source ~/envs/mira-vllm/bin/activate
python -u data_curation/phase16_e1_heldout_sampling.py 2>&1 | tee -a "$HELDOUT_LOG"
hc=${PIPESTATUS[0]}
echo "EXIT_CODE:${hc}" | tee -a "$HELDOUT_LOG"
notify "MIRA chain: E1 held-out sampling finished rc=${hc}"
