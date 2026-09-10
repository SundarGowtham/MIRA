#!/usr/bin/env bash
# PHASE11_REVISED.md Step 2 measurement: parse rate, then ASTRAL n=32,
# sequential on one GPU. runs/sft-qlora-format-only/final is the checkpoint.
set -u
cd /users/gsundar/projects/MIRA
mkdir -p run_logs

LOG="run_logs/format_only_eval.log"
NTFY_TOPIC="mira-g5x7k2-status"

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; [ -f .env ] && source .env; set +a

notify() { curl -s -m 10 -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true; }

run_step() {
    local label="$1"; shift
    for attempt in $(seq 1 20); do
        echo "=== ${label} attempt ${attempt} $(date -u) ===" | tee -a "${LOG}"
        "$@" 2>&1 | tee -a "${LOG}"
        rc=${PIPESTATUS[0]}
        echo "=== ${label} attempt ${attempt} exited rc=${rc} $(date -u) ===" | tee -a "${LOG}"
        if [ "${rc}" -eq 0 ]; then
            echo "${label} DONE $(date -u)" | tee -a "${LOG}"
            notify "MIRA Phase11 Step2 ${label}: DONE (attempt ${attempt})"
            return 0
        fi
        echo "${label} rc=${rc} -- restarting in 30s (attempt $((attempt+1))/20)" | tee -a "${LOG}"
        sleep 30
    done
    notify "MIRA Phase11 Step2 ${label}: FAILED after 20 attempts"
    return 1
}

run_step "parse-rate" uv run python -u run_debug_and_analysis/format_only_parse_rate.py || exit 1
run_step "astral-n32" uv run python -u run_debug_and_analysis/astral_model_generations.py \
    --checkpoint runs/sft-qlora-format-only/final --tag format_only \
    --out misc/astral_gen_n32_format_only.json --n-samples 32 || exit 1

echo "ALL DONE $(date -u)" | tee -a "${LOG}"
notify "MIRA Phase11 Step2 format-only SFT evaluation: ALL DONE"
exit 0
