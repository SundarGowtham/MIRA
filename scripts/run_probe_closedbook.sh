#!/usr/bin/env bash
# Auto-restarting runner for the CLOSED-BOOK effective-support probe
# (Claude Step 3): SFT-v3 checkpoint, no PD context in prompts, full-thermo
# grading. 125 targets (25/stratum) x 24 samples at T=1.0; completions saved
# for RS-SFT/DPO reuse. Resumes from misc/support_probe_closedbook.json.
set -u
cd /users/gsundar/projects/MIRA
mkdir -p run_logs
LOG=run_logs/support_probe_closedbook.log
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for attempt in $(seq 1 50); do
    echo "=== attempt $attempt starting $(date -u) ===" | tee -a "$LOG"
    uv run python probe_effective_support.py \
        --checkpoint runs/sft-qlora-sft-v3-2nd-rank16/final \
        --model Qwen/Qwen3-8B \
        --targets-per-stratum 25 \
        --samples-per-target 24 \
        --temperature 1.0 \
        --closed-book \
        --save-completions \
        --out misc/support_probe_closedbook.json 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    echo "=== attempt $attempt exited rc=$rc $(date -u) ===" | tee -a "$LOG"
    if [ "$rc" -eq 0 ]; then
        echo "DONE: closed-book probe completed $(date -u)" | tee -a "$LOG"
        exit 0
    fi
    echo "rc=$rc — restarting in 60s (attempt $((attempt+1))/50)" | tee -a "$LOG"
    sleep 60
done
echo "FAILED: gave up after 50 attempts $(date -u)" | tee -a "$LOG"
exit 1
