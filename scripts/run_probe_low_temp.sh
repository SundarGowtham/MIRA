#!/usr/bin/env bash
# Low-temperature objective probe runner (misc/some_claude_files/low_temperature_objective_SPEC.md).
# Same pattern as run_gdpo_run3.sh / run_passk.sh: auto-restarting, tee'd log,
# ntfy ping on terminal states. Resume-safe via probe_hardening.py's own
# (condition, target) skip-list, written to --out after every cell.
#
# misc/hardening_low_temp.json is PRE-SEEDED with the 320 baseline records
# from the original 5-condition hardening.json run (same checkpoint/T/top_p/
# max_new_tokens defaults) -- baseline is a pure replicate with no new
# constraint, so those completions are statistically valid for this
# comparison and reusing them roughly halves the GPU time. Only the low_temp
# condition (40 targets x 8 samples = 320 completions) generates fresh.
# --reuse-pool-from locks the target pool to those same 40 targets.
#
#   Usage:  bash run_probe_low_temp.sh
#   Logs:   run_logs/probe_low_temp.log ; ntfy on terminal states.
set -u
cd /users/gsundar/projects/MIRA
mkdir -p run_logs

LOG="run_logs/probe_low_temp.log"
NTFY_TOPIC="mira-g5x7k2-status"
OUT="misc/hardening_low_temp.json"

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; [ -f .env ] && source .env; set +a

notify() { curl -s -m 10 -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true; }

for attempt in $(seq 1 20); do
    echo "=== attempt ${attempt} $(date -u) ===" | tee -a "${LOG}"
    uv run python -u probe_hardening.py \
        --conditions baseline low_temp \
        --reuse-pool-from misc/hardening.json \
        --n-targets 40 --samples 8 --seed 0 \
        --out "${OUT}" 2>&1 | tee -a "${LOG}"
    rc=${PIPESTATUS[0]}
    echo "=== attempt ${attempt} exited rc=${rc} $(date -u) ===" | tee -a "${LOG}"
    if [ "${rc}" -eq 0 ]; then
        echo "DONE $(date -u)" | tee -a "${LOG}"
        notify "MIRA low_temp probe: DONE (attempt ${attempt})"
        exit 0
    fi
    echo "rc=${rc} -- restarting in 60s (attempt $((attempt+1))/20)" | tee -a "${LOG}"
    sleep 60
done
notify "MIRA low_temp probe: FAILED after 20 attempts"
exit 1
