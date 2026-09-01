#!/usr/bin/env bash
# ASTRAL Step 4, 3-model comparison: base + GDPO-300 generation (SFT already
# done in misc/astral_model_generations.json, reused directly -- its matcher
# is the same code, verified correct, no need to regenerate 280 completions).
# Sequential (one GPU): base first, then GDPO-300. Each resume-safe on its
# own output file, auto-restarting on failure.
set -u
cd /users/gsundar/projects/MIRA
mkdir -p run_logs

LOG="run_logs/astral_three_model.log"
NTFY_TOPIC="mira-g5x7k2-status"

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; [ -f .env ] && source .env; set +a

notify() { curl -s -m 10 -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true; }

run_model() {
    local checkpoint="$1" tag="$2" out="$3"
    for attempt in $(seq 1 20); do
        echo "=== ${tag} attempt ${attempt} $(date -u) ===" | tee -a "${LOG}"
        uv run python -u run_debug_and_analysis/astral_model_generations.py \
            --checkpoint "${checkpoint}" --tag "${tag}" --out "${out}" 2>&1 | tee -a "${LOG}"
        rc=${PIPESTATUS[0]}
        echo "=== ${tag} attempt ${attempt} exited rc=${rc} $(date -u) ===" | tee -a "${LOG}"
        if [ "${rc}" -eq 0 ]; then
            echo "${tag} DONE $(date -u)" | tee -a "${LOG}"
            notify "MIRA ASTRAL ${tag}: DONE (attempt ${attempt})"
            return 0
        fi
        echo "${tag} rc=${rc} -- restarting in 30s (attempt $((attempt+1))/20)" | tee -a "${LOG}"
        sleep 30
    done
    notify "MIRA ASTRAL ${tag}: FAILED after 20 attempts"
    return 1
}

run_model "base" "base" "misc/astral_gen_base.json" || exit 1
run_model "runs/gdpo-qlora-gdpo-v3/checkpoint-300" "gdpo300" "misc/astral_gen_gdpo300.json" || exit 1

echo "ALL DONE $(date -u)" | tee -a "${LOG}"
notify "MIRA ASTRAL 3-model comparison: both base and gdpo300 DONE"
exit 0
