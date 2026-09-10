#!/usr/bin/env bash
# Interp probes A/B/C runner (CLAUDE.md "Active work / next steps" items 1-3),
# same pattern as run_passk.sh / run_gdpo_run3.sh: auto-restarting, tee'd log,
# ntfy ping on terminal states. Forward-pass-only (no generation, no
# sampling) over the 200 pass@k targets x 3 checkpoints; script itself is
# resume-safe via get_acts' on-disk npz cache (a restart skips any
# checkpoint whose activations are already cached and starts probing).
#
#   Usage:  bash run_interp_probes.sh
#   Logs:   run_logs/interp_probes.log ; ntfy on terminal states.
set -u
cd /users/gsundar/projects/MIRA
mkdir -p run_logs

LOG="run_logs/interp_probes.log"
NTFY_TOPIC="mira-g5x7k2-status"

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; [ -f .env ] && source .env; set +a

notify() { curl -s -m 10 -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true; }

for attempt in $(seq 1 10); do
    echo "=== attempt ${attempt} $(date -u) ===" | tee -a "${LOG}"
    uv run python -u run_debug_and_analysis/interp_probes.py --all 2>&1 | tee -a "${LOG}"
    rc=${PIPESTATUS[0]}
    echo "=== attempt ${attempt} exited rc=${rc} $(date -u) ===" | tee -a "${LOG}"
    if [ "${rc}" -eq 0 ]; then
        echo "DONE $(date -u)" | tee -a "${LOG}"
        notify "MIRA interp_probes: DONE (attempt ${attempt})"
        exit 0
    fi
    echo "rc=${rc} -- restarting in 30s (attempt $((attempt+1))/10)" | tee -a "${LOG}"
    sleep 30
done
notify "MIRA interp_probes: FAILED after 10 attempts"
exit 1
