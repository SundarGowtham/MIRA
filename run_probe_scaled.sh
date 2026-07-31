#!/usr/bin/env bash
# Auto-restarting runner for the SCALED effective-support probe:
#   ~2,000 targets (400/stratum) x 12 samples at T=0.9 (GRPO's training
#   temperature), completions + breakdowns stored for RS-SFT/DPO reuse and
#   the gradeability-stability analysis.
# Resumes from misc/support_probe_scaled.json after any failure.
set -u
cd /users/gsundar/projects/MIRA
mkdir -p run_logs
LOG=run_logs/support_probe_scaled.log
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for attempt in $(seq 1 50); do
    echo "=== attempt $attempt starting $(date -u) ===" | tee -a "$LOG"
    uv run python probe_effective_support.py \
        --checkpoint runs/sft-qlora-sft-v3-2nd-rank16/final \
        --model Qwen/Qwen3-8B \
        --targets-per-stratum 400 \
        --samples-per-target 12 \
        --temperature 0.9 \
        --save-completions \
        --out misc/support_probe_scaled.json 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    echo "=== attempt $attempt exited rc=$rc $(date -u) ===" | tee -a "$LOG"
    if [ "$rc" -eq 0 ]; then
        echo "DONE: scaled probe completed $(date -u)" | tee -a "$LOG"
        exit 0
    fi
    echo "rc=$rc — restarting in 60s (attempt $((attempt+1))/50)" | tee -a "$LOG"
    sleep 60
done
echo "FAILED: gave up after 50 attempts $(date -u)" | tee -a "$LOG"
exit 1
