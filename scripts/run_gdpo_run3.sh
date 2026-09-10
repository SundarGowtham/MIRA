#!/usr/bin/env bash
# Run-3 GDPO runner — auto-restarting, same pattern as run_gdpo.sh.
#
# Run-3 spec (gdpo_v4_next_steps_claude_recommendation.md §5 step 4):
#   5-channel reward (amount_accuracy, thermodynamic_favorable,
#   stoichiometry, chempot_atmosphere, operation_order), uniform weights;
#   beta=0.001, lr=1e-5; epsilon_high=5.0; closed-book prompts;
#   mid-band + gradeability-stable dataset (data/rl_run3, from
#   data_curation/build_rl_run3_dataset.py); fixed 30-target probe every
#   50 steps (eval_*, generations.jsonl) + per-check within-group std
#   (within_group_std/*) as the dead-channel diagnostics.
#
# First attempt initializes from run 2's checkpoint-300 with
# --fresh-restart (reward vector AND dataset changed -> new
# optimizer/scheduler; a normal resume would re-impose run 2's state).
# Restarts resume from the latest runs/gdpo-qlora-gdpo-v4/checkpoint-N.
#
#   Usage:  bash run_gdpo_run3.sh [tag] [seed]
#   Logs:   run_logs/gdpo_<tag>.log ; ntfy ping on terminal states.
set -u
cd /users/gsundar/projects/MIRA
mkdir -p run_logs

TAG="${1:-gdpo-v4}"
SEED="${2:-42}"
EXP="gdpo"
RUN_DIR="runs/${EXP}-qlora-${TAG}"
LOG="run_logs/gdpo_${TAG}.log"
RUN2_INIT="runs/gdpo-qlora-beta-ablation-probe/checkpoint-300"
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
        INIT_FROM="${RUN2_INIT}"; FRESH="--fresh-restart"
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
