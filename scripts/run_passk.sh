#!/usr/bin/env bash
# pass@k n=200 runner (item 2, Claude's spec): k=16, ALL 200 held-out val
# targets, decision pair (sft, gdpo300) computed first. Resume-safe: reruns
# skip completed (target, model) cells, so the loop just re-invokes.
#
# Decision rule (set in advance): GDPO-300 pass@16 - SFT pass@16, paired
# across targets. Gap >= 5 pts -> expansion real, relaunch run 3. Gap
# closes -> RL premise fails, pivot to inventory-constrained difficulty.
#
#   Usage:  bash run_passk.sh
#   Logs:   run_logs/passk_n200.log ; ntfy on terminal states.
set -u
cd /users/gsundar/projects/MIRA
mkdir -p run_logs

LOG="run_logs/passk_n200.log"
NTFY_TOPIC="mira-g5x7k2-status"

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; [ -f .env ] && source .env; set +a

notify() { curl -s -m 10 -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true; }

for attempt in $(seq 1 10); do
    echo "=== attempt ${attempt} $(date -u) ===" | tee -a "${LOG}"
    uv run python probe_passk.py \
        --targets-per-stratum 999 --n-samples 16 \
        --out misc/passk_n200.json 2>&1 | tee -a "${LOG}"
    rc=${PIPESTATUS[0]}
    echo "=== attempt ${attempt} exited rc=${rc} $(date -u) ===" | tee -a "${LOG}"
    if [ "${rc}" -eq 0 ]; then
        echo "DONE $(date -u)" | tee -a "${LOG}"
        notify "MIRA passk_n200: DONE (attempt ${attempt})"
        exit 0
    fi
    sleep 30
done
notify "MIRA passk_n200: FAILED after 10 attempts"
exit 1
