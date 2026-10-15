#!/usr/bin/env bash
# Waits for the support-growth-by-checkpoint GPU sampling job to finish,
# then runs the CPU-only analysis (bootstrap + hits/carbonate per
# checkpoint) without requiring anyone to be watching.
set -u
cd /users/gsundar/projects/MIRA
NTFY_TOPIC="mira-g5x7k2-status"
SAMPLING_LOG="run_logs/phase16_support_growth_trajectory_sampling.log"
ANALYSIS_LOG="run_logs/support_growth_by_checkpoint.log"

notify() { curl -s -m 10 -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true; }

notify "MIRA chain: waiting for support-growth-trajectory sampling to finish before running checkpoint analysis"

until grep -q "EXIT_CODE" "$SAMPLING_LOG" 2>/dev/null; do
    sleep 30
done

RC=$(grep "EXIT_CODE" "$SAMPLING_LOG" | tail -1 | grep -oE '[0-9]+')
echo "=== support-growth-trajectory sampling finished rc=${RC} $(date -u) ===" | tee -a "$ANALYSIS_LOG"
notify "MIRA chain: support-growth sampling done (rc=${RC}). Running checkpoint analysis now."

uv run python -u research/support_growth_by_checkpoint.py 2>&1 | tee -a "$ANALYSIS_LOG"
ec=${PIPESTATUS[0]}
echo "EXIT_CODE:${ec}" | tee -a "$ANALYSIS_LOG"
notify "MIRA chain: support-growth-by-checkpoint analysis finished rc=${ec}"
