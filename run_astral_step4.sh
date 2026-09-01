#!/usr/bin/env bash
# ASTRAL Step 4 — does the model ever propose the good precursors?
# 35 targets x 8 samples, SFT checkpoint, closed-book. Resume-safe (per-target
# incremental writes), auto-restarting per the usual pattern.
set -u
cd /users/gsundar/projects/MIRA
mkdir -p run_logs

LOG="run_logs/astral_step4.log"
NTFY_TOPIC="mira-g5x7k2-status"

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; [ -f .env ] && source .env; set +a

notify() { curl -s -m 10 -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true; }

for attempt in $(seq 1 20); do
    echo "=== attempt ${attempt} $(date -u) ===" | tee -a "${LOG}"
    uv run python -u run_debug_and_analysis/astral_model_generations.py 2>&1 | tee -a "${LOG}"
    rc=${PIPESTATUS[0]}
    echo "=== attempt ${attempt} exited rc=${rc} $(date -u) ===" | tee -a "${LOG}"
    if [ "${rc}" -eq 0 ]; then
        echo "DONE $(date -u)" | tee -a "${LOG}"
        notify "MIRA ASTRAL step4: DONE (attempt ${attempt})"
        exit 0
    fi
    echo "rc=${rc} -- restarting in 30s (attempt $((attempt+1))/20)" | tee -a "${LOG}"
    sleep 30
done
notify "MIRA ASTRAL step4: FAILED after 20 attempts"
exit 1
