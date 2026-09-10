#!/usr/bin/env bash
# Phase 12 GDPO runner — GDPO initialized from RS-SFT-from-base instead of
# full SFT. Same auto-restarting pattern as run_gdpo_run3.sh/run_gdpo_run4.sh.
#
# The ONLY variable vs Arm A is the init checkpoint: same beta=0.001,
# lr=1e-5, G=8, per-device batch 1 x accum 16, epsilon_high=5.0, closed-book,
# same data pipeline as run 3 (data/rl_run3), fixed 30-target probe every
# 50 steps, EvalModeGuard, checkpoints every 100 steps, unmodified validator
# (NOT the ranker -- it failed its external gate, misc/PHASE11_RESULTS.md).
#
# First attempt initializes from runs/sft-qlora-rs-sft-from-base/final with
# --fresh-restart (new init point -> new optimizer/scheduler state; a normal
# resume would try to reuse run-3/run-4's state). Restarts resume from the
# latest runs/gdpo-qlora-<tag>/checkpoint-N.
#
#   Usage:  bash run_gdpo_phase12.sh [tag] [seed]
#   Logs:   run_logs/gdpo_<tag>.log ; ntfy ping on terminal states.
set -u
cd /users/gsundar/projects/MIRA
mkdir -p run_logs

TAG="${1:-gdpo-phase12-rssft}"
SEED="${2:-42}"
EXP="gdpo"
RUN_DIR="runs/${EXP}-qlora-${TAG}"
LOG="run_logs/gdpo_${TAG}.log"
RS_SFT_INIT="runs/sft-qlora-rs-sft-from-base/final"
NTFY_TOPIC="mira-g5x7k2-status"

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; [ -f .env ] && source .env; set +a

notify() { curl -s -m 10 -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true; }

for attempt in $(seq 1 50); do
    LATEST=$(ls -d "${RUN_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
    if [ -n "${LATEST}" ]; then
        INIT_FROM="${LATEST}"; FRESH=""
    else
        INIT_FROM="${RS_SFT_INIT}"; FRESH="--fresh-restart"
    fi
    echo "=== attempt ${attempt}: init_from=${INIT_FROM} ${FRESH} $(date -u) ===" | tee -a "${LOG}"
    uv run python -u train.py "${EXP}" \
        --adapter qlora --model Qwen/Qwen3-8B \
        --data-dir data/rl_run3 --data-prefix rl3 \
        --lora-r 16 --lora-alpha 32 \
        --seed "${SEED}" --tag "${TAG}" \
        --init-from "${INIT_FROM}" ${FRESH} 2>&1 | tee -a "${LOG}"
    rc=${PIPESTATUS[0]}
    echo "=== attempt ${attempt} exited rc=${rc} $(date -u) ===" | tee -a "${LOG}"
    if [ "${rc}" -eq 0 ]; then
        echo "DONE: training completed $(date -u)" | tee -a "${LOG}"
        notify "MIRA ${TAG}: DONE (attempt ${attempt})"
        exit 0
    fi
    echo "rc=${rc} — restarting in 60s (attempt $((attempt+1))/50)" | tee -a "${LOG}"
    sleep 60
done
echo "FAILED: gave up after 50 attempts $(date -u)" | tee -a "${LOG}"
notify "MIRA ${TAG}: FAILED after 50 attempts"
exit 1
