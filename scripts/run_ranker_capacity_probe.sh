#!/usr/bin/env bash
# Ranker capacity probe -- RANKER_SPEC.md step 3, THE STOP GATE for Arm B.
# Same auto-restarting / tee'd-log / ntfy pattern as the other probe runners.
#
# Paired against Arm A's validator baseline: same 40 targets (--reuse-pool-from
# misc/hardening.json), same seed, same SFT checkpoint, closed-book, baseline
# condition only (no prompt constraints -- the ranker itself is the only
# thing that changed). Rail-calibrated scales from
# misc/ranker_rail_calibration_v2.json (N_MAX 5->250 was the major fix;
# phase_purity was 100% dead at the spec's starting N_MAX).
#
# Pre-registered rule (RANKER_SPEC.md section 5):
#   >40%    -> proceed to run 4
#   25-40%  -> identify flat channels, fix scales, re-probe once, do not train
#   <25%    -> stop; two verifier generations and five/eight interventions
#              have failed to produce a rankable reward
#
#   Usage:  bash run_ranker_capacity_probe.sh
#   Logs:   run_logs/ranker_capacity_probe.log ; ntfy on terminal states.
set -u
cd /users/gsundar/projects/MIRA
mkdir -p run_logs

LOG="run_logs/ranker_capacity_probe.log"
NTFY_TOPIC="mira-g5x7k2-status"
OUT="misc/ranker_capacity_probe.json"

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; [ -f .env ] && source .env; set +a

notify() { curl -s -m 10 -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true; }

for attempt in $(seq 1 30); do
    echo "=== attempt ${attempt} $(date -u) ===" | tee -a "${LOG}"
    uv run python -u probe_hardening.py \
        --scorer ranker \
        --conditions baseline \
        --reuse-pool-from misc/hardening.json \
        --n-targets 40 --samples 8 --seed 0 \
        --out "${OUT}" 2>&1 | tee -a "${LOG}"
    rc=${PIPESTATUS[0]}
    echo "=== attempt ${attempt} exited rc=${rc} $(date -u) ===" | tee -a "${LOG}"
    if [ "${rc}" -eq 0 ]; then
        echo "DONE $(date -u)" | tee -a "${LOG}"
        notify "MIRA ranker capacity probe (STOP GATE): DONE (attempt ${attempt})"
        exit 0
    fi
    echo "rc=${rc} -- restarting in 60s (attempt $((attempt+1))/30)" | tee -a "${LOG}"
    sleep 60
done
notify "MIRA ranker capacity probe: FAILED after 30 attempts"
exit 1
