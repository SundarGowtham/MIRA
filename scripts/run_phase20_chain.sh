#!/usr/bin/env bash
# Phase 20 chain (docs/phases/PHASE20_PREREG.md). Resume-safe: each stage
# writes runs/phase20/STAGE_<name>.done and is skipped on rerun if the marker
# exists. ntfy ping at the end of every stage and on any failure. Stops (exit
# 1) only when a gate fails or a stage cannot complete.
#
# DEFERRED stages (exit code 3): a sampling stage reserved for RunPod
# (touch runs/phase20/RESERVE_<job>) whose files are not back yet, or a
# scoring stage whose input samples are not back yet. No marker is written and
# the chain moves on (a reserved Stage 2 never blocks Stage 3). Deferred
# stages are retried after every later stage, and at the end the chain polls
# them every 30 min until they complete. S7_final_analysis itself waits for
# every sample set to pass research/phase20_verify_samples.py.
#
# Each stage body lives in scripts/phase20_stages/<name>.sh, read fresh when
# the stage starts, so later stages (the analyses) can be written while
# earlier ones run. If a stage script does not exist yet, the chain waits for
# it (pinging every 2 h) instead of failing.
#
#   tmux new -d -s phase20_chain 'bash scripts/run_phase20_chain.sh'
set -u
cd /users/gsundar/projects/MIRA
mkdir -p run_logs runs/phase20
LOG=run_logs/phase20_chain.log
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
notify() { curl -s -m 10 -d "$1" "https://ntfy.sh/mira-g5x7k2-status" >/dev/null 2>&1 || true; }

STAGES=(
  S0_gates
  S1_train_sft-rssft-full S1_gateR_sft-rssft-full
  S1_train_sft-e3-full    S1_gateR_sft-e3-full
  S1_train_sft-e3c-full   S1_gateR_sft-e3c-full
  S2_sample
  S2b_score
  S3_rl
  S4_sample_e1
  S4_sample_ref
  S4_score
  S5_sample
  S6_optional
)
FINAL=S7_final_analysis
DEFERRED=()

# run_stage <name> [quiet] -> 0 done, 3 deferred; exits the chain on failure
run_stage() {
  local st="$1" quiet="${2:-}"
  local marker="runs/phase20/STAGE_${st}.done"
  [ -f "$marker" ] && { [ -z "$quiet" ] && echo "[chain] skip ${st} (done)" | tee -a "$LOG"; return 0; }
  local body="scripts/phase20_stages/${st%%_sft-*}.sh" arg="" waited=0
  [[ "$st" == *_sft-* ]] && arg="sft-${st#*_sft-}"
  while [ ! -f "$body" ]; do
    [ $((waited % 7200)) -eq 0 ] && notify "MIRA phase20: stage ${st} waiting for its script ${body}"
    sleep 300; waited=$((waited + 300))
  done
  echo "[chain] === ${st} start $(date -u) ===" | tee -a "$LOG"
  bash "$body" $arg 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 3 ]; then
    echo "[chain] ${st} DEFERRED $(date -u)" | tee -a "$LOG"
    [ -z "$quiet" ] && notify "MIRA phase20: stage ${st} deferred (inputs on RunPod); continuing"
    return 3
  fi
  if [ "$rc" -ne 0 ]; then
    echo "[chain] ${st} FAILED rc=${rc} $(date -u)" | tee -a "$LOG"
    notify "MIRA phase20: stage ${st} FAILED rc=${rc}. Chain stopped; see run_logs/phase20_chain.log"
    exit 1
  fi
  date -u > "$marker"
  echo "[chain] === ${st} done $(date -u) ===" | tee -a "$LOG"
  notify "MIRA phase20: stage ${st} done"
  return 0
}

retry_deferred() {
  local keep=() st
  for st in "${DEFERRED[@]}"; do
    run_stage "$st" quiet; [ $? -eq 3 ] && keep+=("$st")
  done
  DEFERRED=("${keep[@]}")
}

for st in "${STAGES[@]}"; do
  [ ${#DEFERRED[@]} -gt 0 ] && retry_deferred
  run_stage "$st"; [ $? -eq 3 ] && DEFERRED+=("$st")
done
waited=0
while [ ${#DEFERRED[@]} -gt 0 ]; do
  retry_deferred
  [ ${#DEFERRED[@]} -eq 0 ] && break
  [ $((waited % 7200)) -eq 0 ] && notify "MIRA phase20: waiting for RunPod results: ${DEFERRED[*]}"
  sleep 1800; waited=$((waited + 1800))
done
run_stage "$FINAL"
notify "MIRA phase20: chain COMPLETE"
