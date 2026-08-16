#!/usr/bin/env bash
# Auto-restarting GDPO/GRPO runner.
#   - Resumes from the latest runs/<run>/checkpoint-N after any failure
#     (OOM, kill, disconnect); first attempt initializes from the SFT final.
#   - Usage:
#       bash run_gdpo.sh                                  # GDPO, tag gdpo-v1, seed 42
#       bash run_gdpo.sh gdpo-v2 1337                     # GDPO, second seed
#       bash run_gdpo.sh grpo-v1 42 grpo                  # classic GRPO arm
#   - Logs everything to run_logs/gdpo_<tag>.log via tee.
set -u
cd /users/gsundar/projects/MIRA
mkdir -p run_logs

TAG="${1:-gdpo-v3}"
SEED="${2:-42}"
EXP="${3:-gdpo}"
AGG="${4:-sum_then_normalize}"   # only used by the grpo arm; gdpo forces normalize_then_sum
RUN_DIR="runs/${EXP}-qlora-${TAG}"
LOG="run_logs/gdpo_${TAG}.log"
SFT_INIT="runs/sft-qlora-sft-v3-2nd-rank16/final"

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; [ -f .env ] && source .env; set +a

for attempt in $(seq 1 50); do
    LATEST=$(ls -d "${RUN_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
    if [ -n "${LATEST}" ]; then
        INIT_FROM="${LATEST}"
    else
        INIT_FROM="${SFT_INIT}"
    fi
    echo "=== attempt ${attempt}: init_from=${INIT_FROM} $(date -u) ===" | tee -a "${LOG}"
    uv run python train.py "${EXP}" \
        --adapter qlora --model Qwen/Qwen3-8B \
        --data-dir data/rl --data-prefix rl \
        --lora-r 16 --lora-alpha 32 \
        --reward-aggregation "${AGG}" \
        --seed "${SEED}" --tag "${TAG}" \
        --init-from "${INIT_FROM}" 2>&1 | tee -a "${LOG}"
    rc=${PIPESTATUS[0]}
    echo "=== attempt ${attempt} exited rc=${rc} $(date -u) ===" | tee -a "${LOG}"
    if [ "${rc}" -eq 0 ]; then
        echo "DONE: training completed $(date -u)" | tee -a "${LOG}"
        exit 0
    fi
    echo "rc=${rc} — restarting in 60s (attempt $((attempt+1))/50)" | tee -a "${LOG}"
    sleep 60
done
echo "FAILED: gave up after 50 attempts $(date -u)" | tee -a "${LOG}"
exit 1
