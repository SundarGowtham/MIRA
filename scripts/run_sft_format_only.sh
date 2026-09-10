#!/usr/bin/env bash
# PHASE11_REVISED.md Step 2: format-only SFT from base.
# 300 examples (seeded sample of the full 1217-example sft_v3 train set),
# ~1 epoch (max-steps sized to batch=4 x accum=8=32 effective batch:
# 300/32 ~= 9.4 -> 10 steps), fresh LoRA from base (not continuing SFT).
# Goal: install the JSON schema without imprinting the corpus's chemistry
# preferences (which is what PHASE11_REVISED.md's finding 16 blames for the
# ASTRAL support collapse, base 10/35 -> SFT 1/35).
set -u
cd /users/gsundar/projects/MIRA
mkdir -p run_logs

LOG="run_logs/sft_format_only.log"
NTFY_TOPIC="mira-g5x7k2-status"

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; [ -f .env ] && source .env; set +a

notify() { curl -s -m 10 -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true; }

for attempt in $(seq 1 10); do
    echo "=== format-only-sft attempt ${attempt} $(date -u) ===" | tee -a "${LOG}"
    uv run python -u train.py sft --adapter qlora --model Qwen/Qwen3-8B \
        --data-dir data/sft_v3 --data-prefix format_only \
        --init-from base --max-steps 10 --tag format-only \
        2>&1 | tee -a "${LOG}"
    rc=${PIPESTATUS[0]}
    echo "=== format-only-sft attempt ${attempt} exited rc=${rc} $(date -u) ===" | tee -a "${LOG}"
    if [ "${rc}" -eq 0 ]; then
        echo "format-only-sft DONE $(date -u)" | tee -a "${LOG}"
        notify "MIRA Phase11 Step2 format-only SFT: DONE (attempt ${attempt})"
        exit 0
    fi
    echo "format-only-sft rc=${rc} -- restarting in 30s (attempt $((attempt+1))/10)" | tee -a "${LOG}"
    sleep 30
done
notify "MIRA Phase11 Step2 format-only SFT: FAILED after 10 attempts"
exit 1
