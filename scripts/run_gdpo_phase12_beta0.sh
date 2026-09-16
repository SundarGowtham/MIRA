#!/usr/bin/env bash
# Phase 12 GDPO runner, relaunch after the KL-estimator blowup (2026-09-10).
# Identical to run_gdpo_phase12.sh except beta=0 (was 0.001).
#
# Original run (tag gdpo-phase12-rssft) killed at step 96: KL exploded
# 1.32 -> 97.2 -> 1.99e11 across steps 25/50/75 (k3 estimator exp(r)-r-1 is
# unbounded above; a single outlier token with r ~ 35-37 nats, most likely
# a fractional-formula numeral RS-SFT sharpened far from base's probability
# on, blows up the whole batch mean). loss was 99.99% KL penalty by step 75
# (beta*KL = 0.001*1.99e11 ~= 1.99e8 against a logged loss of 2.07e8), grad
# norm collapsed 616 -> 1.84 -> 0.072 (learning had stopped), reward
# declining 3.88 -> 3.80 -> 3.68 (not stable, as first read). See
# misc/PHASE12_RESULTS.md and CLAUDE.md finding 20.
#
# beta=0 is not a second variable against Arm A: Arm A's own beta*KL was
# order 2e-6 against a loss of order 0.03 (KL 0.001-0.003) -- numerically
# absent already. Standard practice (DAPO drops the KL term for long-
# horizon RLVR). Everything else identical to the original Phase 12 launch:
# RS-SFT-from-base init, validator scorer, data/rl_run3, same G/batch/accum/
# epsilon_high/checkpointing.
#
#   Usage:  bash run_gdpo_phase12_beta0.sh [tag] [seed]
#   Logs:   run_logs/gdpo_<tag>.log ; ntfy ping on terminal states.
set -u
cd /users/gsundar/projects/MIRA
mkdir -p run_logs

TAG="${1:-gdpo-phase12-rssft-beta0}"
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
    echo "=== attempt ${attempt}: init_from=${INIT_FROM} ${FRESH} beta=0.0 $(date -u) ===" | tee -a "${LOG}"
    uv run python -u train.py "${EXP}" \
        --adapter qlora --model Qwen/Qwen3-8B \
        --data-dir data/rl_run3 --data-prefix rl3 \
        --lora-r 16 --lora-alpha 32 \
        --kl-beta 0.0 \
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
