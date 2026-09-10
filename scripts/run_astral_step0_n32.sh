#!/usr/bin/env bash
# Phase 11 Step 0: high-resolution ASTRAL baseline, 32 samples/target,
# base + SFT + GDPO-300, sequential (one GPU), each resume-safe on its own
# output file, auto-restarting on failure. 35 x 32 x 3 = 3,360 generations.
set -u
cd /users/gsundar/projects/MIRA
mkdir -p run_logs

LOG="run_logs/astral_step0_n32.log"
NTFY_TOPIC="mira-g5x7k2-status"

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; [ -f .env ] && source .env; set +a

notify() { curl -s -m 10 -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true; }

run_model() {
    local checkpoint="$1" tag="$2" out="$3"
    for attempt in $(seq 1 20); do
        echo "=== ${tag} n32 attempt ${attempt} $(date -u) ===" | tee -a "${LOG}"
        uv run python -u run_debug_and_analysis/astral_model_generations.py \
            --checkpoint "${checkpoint}" --tag "${tag}" --out "${out}" \
            --n-samples 32 2>&1 | tee -a "${LOG}"
        rc=${PIPESTATUS[0]}
        echo "=== ${tag} n32 attempt ${attempt} exited rc=${rc} $(date -u) ===" | tee -a "${LOG}"
        if [ "${rc}" -eq 0 ]; then
            echo "${tag} n32 DONE $(date -u)" | tee -a "${LOG}"
            notify "MIRA Phase11 Step0 ${tag} (n=32): DONE (attempt ${attempt})"
            return 0
        fi
        echo "${tag} n32 rc=${rc} -- restarting in 30s (attempt $((attempt+1))/20)" | tee -a "${LOG}"
        sleep 30
    done
    notify "MIRA Phase11 Step0 ${tag} (n=32): FAILED after 20 attempts"
    return 1
}

run_model "runs/sft-qlora-sft-v3-2nd-rank16/final" "sft" "misc/astral_gen_n32_sft.json" || exit 1
run_model "base" "base" "misc/astral_gen_n32_base.json" || exit 1
run_model "runs/gdpo-qlora-gdpo-v3/checkpoint-300" "gdpo300" "misc/astral_gen_n32_gdpo300.json" || exit 1

echo "ALL DONE $(date -u)" | tee -a "${LOG}"
notify "MIRA Phase11 Step0 (n=32, 3 models): ALL DONE"
exit 0
