#!/usr/bin/env bash
# Phase 18 single-block runner: tee to run_logs/, ntfy on start/finish.
# Usage: scripts/run_phase18_block.sh <label> <script.py> [args...]
set -u
cd /users/gsundar/projects/MIRA
NTFY_TOPIC="mira-g5x7k2-status"
LABEL="$1"; shift
SCRIPT="$1"; shift
LOG="run_logs/phase18_${LABEL}.log"
notify() { curl -s -m 10 -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true; }
export PYTHONPATH=. PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== phase18 ${LABEL} start $(date -u) ===" | tee -a "$LOG"
notify "MIRA Phase 18 ${LABEL}: started"
uv run python -u "$SCRIPT" "$@" 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "EXIT_CODE:${ec}" | tee -a "$LOG"
notify "MIRA Phase 18 ${LABEL}: finished rc=${ec}"
exit $ec
