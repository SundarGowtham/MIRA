#!/usr/bin/env bash
# Phase 20 CPU analyses (tmux). Each step runs once and leaves a marker in
# runs/phase20/CPU_<name>.done; a failed step pings and the rest continue.
#   bash scripts/phase20_cpu.sh sft     SFT-side halves (after Stage 2b)
#   bash scripts/phase20_cpu.sh final   everything (Stage 7)
set -u
cd /users/gsundar/projects/MIRA
export PYTHONPATH=. PYTHONHASHSEED=0
notify() { curl -s -m 10 -d "$1" "https://ntfy.sh/mira-g5x7k2-status" >/dev/null 2>&1 || true; }
LOG=run_logs/phase20_cpu_$1.log
run() {  # run <marker-name> <analysis args...>
  local name="$1"; shift
  [ -f "runs/phase20/CPU_${name}.done" ] && { echo "skip $name"; return 0; }
  echo "=== $name $(date -u) ===" | tee -a "$LOG"
  uv run python -u research/phase20_analysis.py "$@" 2>&1 | tee -a "$LOG"
  if [ "${PIPESTATUS[0]}" -eq 0 ]; then date -u > "runs/phase20/CPU_${name}.done"; else notify "MIRA phase20 CPU analysis $name FAILED"; FAILED=1; fi
}
FAILED=0
case "$1" in
  sft)
    run transport_2b transport --stage 2b
    run reach_2b reach --stage 2b
    run uniform_2b uniform --stage 2b
    run posterior_2b posterior --stage 2b
    run coverage_2b coverage --stage 2b
    run hits_2b hits --stage 2b
    run alkali_2b alkali --stage 2b
    notify "MIRA phase20: SFT-side CPU analyses finished (failed=$FAILED) -- report 2 due"
    ;;
  final)
    run transport_final transport --stage final
    run posterior_final posterior --stage final
    run support_final support
    run reach_final reach --stage final
    run location_final location
    run uniform_final uniform --stage final
    run coverage_final coverage
    run mmd_rssft mmd_rssft
    run hits_final hits
    run alkali_final alkali
    run tempfix tempfix
    run tempfix_blocks tempfix_blocks
    run weights weights
    run report report
    notify "MIRA phase20: final analyses finished (failed=$FAILED)"
    ;;
esac
exit $FAILED
