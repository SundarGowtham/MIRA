#!/usr/bin/env bash
# Phase 16: waits for E3/E3c n=32 sampling to finish (GPU, mira-vllm env),
# then immediately runs the novelty-rate/support-growth evaluation
# (CPU, main env, thermo-backed validator calls) without requiring
# anyone to be watching.
set -u
cd /users/gsundar/projects/MIRA
NTFY_TOPIC="mira-g5x7k2-status"
SAMPLING_LOG="run_logs/phase16_e3_e3c_sampling.log"
EVAL_LOG="run_logs/phase16_e3_e3c_evaluation.log"

notify() { curl -s -m 10 -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true; }

notify "MIRA chain: waiting for E3/E3c sampling to finish before running novelty/support-growth evaluation"

until grep -q "EXIT_CODE" "$SAMPLING_LOG" 2>/dev/null; do
    sleep 20
done

RC=$(grep "EXIT_CODE" "$SAMPLING_LOG" | tail -1 | grep -oE '[0-9]+')
echo "=== E3/E3c sampling finished rc=${RC} $(date -u) ===" | tee -a "$EVAL_LOG"
notify "MIRA chain: E3/E3c sampling done (rc=${RC}). Running novelty-rate/support-growth evaluation now."

uv run python -u research/phase16_e3_e3c_evaluation.py 2>&1 | tee -a "$EVAL_LOG"
ec=${PIPESTATUS[0]}
echo "EXIT_CODE:${ec}" | tee -a "$EVAL_LOG"
notify "MIRA chain: E3/E3c novelty-rate/support-growth evaluation finished rc=${ec}"
