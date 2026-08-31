#!/usr/bin/env bash
# Low-temperature objective probe, ROUND 2 (misc/some_claude_files/post_low_temp_probe_steps.md).
# Same auto-restarting / tee'd-log / ntfy pattern as run_probe_low_temp.sh.
#
# Round 1 (misc/hardening_low_temp.json) got capacity 23.8%, just under the
# 25% floor -- CLAUDE.md finding 15. Round 2 fixes two measured weaknesses:
#   Fix A: temperature_economy was badly scaled (1/3 of completions piled at
#     0.0/1.0, no gradient). Tightened T_ref_margin 300->150, T_span 600->400.
#   Fix B: the soft preference alone only pulled mean reported T to 1056 C,
#     while an explicit ceiling (finding 9) forced 100% compliance at 993 C.
#     New low_temp_ceiling condition combines both: hard ceiling for the
#     behavioral pull, economy gradient banded to [lit_T, ceiling].
# Also: low_temp_inventory (round 1's low_temp_combined, renamed), and
# baseline is instrumented with max_T_reported this round (round 1 left it
# untracked, so there was no baseline T to compare against).
#
# Baseline is regenerated fresh this round (not reused from round 1) because
# the instrumentation fix means round-1 baseline records lack max_T_reported.
# --reuse-pool-from still locks the target pool to the same 40 targets used
# in every round for comparability.
#
#   Usage:  bash run_probe_low_temp2.sh
#   Logs:   run_logs/probe_low_temp2.log ; ntfy on terminal states.
set -u
cd /users/gsundar/projects/MIRA
mkdir -p run_logs

LOG="run_logs/probe_low_temp2.log"
NTFY_TOPIC="mira-g5x7k2-status"
OUT="misc/hardening_low_temp2.json"

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; [ -f .env ] && source .env; set +a

notify() { curl -s -m 10 -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true; }

for attempt in $(seq 1 30); do
    echo "=== attempt ${attempt} $(date -u) ===" | tee -a "${LOG}"
    uv run python -u probe_hardening.py \
        --conditions baseline low_temp low_temp_ceiling low_temp_inventory \
        --reuse-pool-from misc/hardening.json \
        --t-ref-margin 150 --t-span 400 \
        --n-targets 40 --samples 8 --seed 0 \
        --out "${OUT}" 2>&1 | tee -a "${LOG}"
    rc=${PIPESTATUS[0]}
    echo "=== attempt ${attempt} exited rc=${rc} $(date -u) ===" | tee -a "${LOG}"
    if [ "${rc}" -eq 0 ]; then
        echo "DONE $(date -u)" | tee -a "${LOG}"
        notify "MIRA low_temp probe round 2: DONE (attempt ${attempt})"
        exit 0
    fi
    echo "rc=${rc} -- restarting in 60s (attempt $((attempt+1))/30)" | tee -a "${LOG}"
    sleep 60
done
notify "MIRA low_temp probe round 2: FAILED after 30 attempts"
exit 1
