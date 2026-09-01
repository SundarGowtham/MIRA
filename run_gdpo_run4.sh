#!/usr/bin/env bash
# Run-4 GDPO runner — Arm B (RANKER_SPEC.md step 4 / ranker_fixes_instructions.md
# step 5). Same auto-restarting pattern as run_gdpo_run3.sh.
#
# The verifier is the ONLY thing that changes vs Arm A: same beta=0.001,
# lr=1e-5, G=8, per-device batch 1 accum 16, epsilon_high=5.0, closed-book,
# same data pipeline as run 3 (data/rl_run3: mid-band p-hat,
# gradeability-stable targets), fixed 30-target probe every 50 steps,
# EvalModeGuard, checkpoints every 100 steps. --scorer ranker swaps
# core/reward.py's SynthesisValidator for core/ranker.py's Ranker
# (5 active objectives post ranker_fixes_instructions.md: temperature_economy,
# step_economy, precursor_availability, volatility_risk, driving_force_margin
# -- phase_purity dropped, see core/ranker.py's OBJECTIVE_NAMES comment).
#
# Gate: misc/ranker_capacity_recheck.json showed 51.12% capacity on properly
# grouped archived data (up from the pre-fix probe's 48.7%), both comfortably
# clearing the pre-registered 40% bar -- this is what justifies training.
#
# First attempt initializes from the SFT checkpoint (NOT run 2's GDPO
# checkpoint -- both the reward vector AND the init point are new here) with
# --fresh-restart. Restarts resume from the latest
# runs/gdpo-qlora-gdpo-run4-ranker/checkpoint-N.
#
#   Usage:  bash run_gdpo_run4.sh [tag] [seed]
#   Logs:   run_logs/gdpo_<tag>.log ; ntfy ping on terminal states.
set -u
cd /users/gsundar/projects/MIRA
mkdir -p run_logs

TAG="${1:-gdpo-run4-ranker}"
SEED="${2:-42}"
EXP="gdpo"
RUN_DIR="runs/${EXP}-qlora-${TAG}"
LOG="run_logs/gdpo_${TAG}.log"
SFT_INIT="runs/sft-qlora-sft-v3-2nd-rank16/final"
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
        INIT_FROM="${SFT_INIT}"; FRESH="--fresh-restart"
    fi
    echo "=== attempt ${attempt}: init_from=${INIT_FROM} ${FRESH} $(date -u) ===" | tee -a "${LOG}"
    uv run python -u train.py "${EXP}" \
        --adapter qlora --model Qwen/Qwen3-8B \
        --data-dir data/rl_run3 --data-prefix rl3 \
        --lora-r 16 --lora-alpha 32 \
        --scorer ranker \
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
