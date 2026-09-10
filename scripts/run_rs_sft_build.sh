#!/usr/bin/env bash
# PHASE11_REVISED.md Step 3: RS-SFT from base.
# Sample 8 routes/target from BASE Qwen3-8B (not SFT -- base has the good
# routes in support), score with the unmodified validator, keep survivors
# at bar=0.9 (data_curation/build_rs_sft_dataset.py's own documented
# convention), fine-tune from base on survivors. Resume-safe (per-target
# skip on rerun).
set -u
cd /users/gsundar/projects/MIRA
mkdir -p run_logs

LOG="run_logs/rs_sft_build.log"
NTFY_TOPIC="mira-g5x7k2-status"

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; [ -f .env ] && source .env; set +a

notify() { curl -s -m 10 -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true; }

for attempt in $(seq 1 30); do
    echo "=== rs-sft-build attempt ${attempt} $(date -u) ===" | tee -a "${LOG}"
    uv run python -u data_curation/build_rs_sft_dataset.py \
        --checkpoint base --model Qwen/Qwen3-8B \
        2>&1 | tee -a "${LOG}"
    rc=${PIPESTATUS[0]}
    echo "=== rs-sft-build attempt ${attempt} exited rc=${rc} $(date -u) ===" | tee -a "${LOG}"
    if [ "${rc}" -eq 0 ]; then
        echo "rs-sft-build DONE $(date -u)" | tee -a "${LOG}"
        notify "MIRA Phase11 Step3 RS-SFT build: DONE (attempt ${attempt})"
        exit 0
    fi
    echo "rs-sft-build rc=${rc} -- restarting in 30s (attempt $((attempt+1))/30)" | tee -a "${LOG}"
    sleep 30
done
notify "MIRA Phase11 Step3 RS-SFT build: FAILED after 30 attempts"
exit 1
