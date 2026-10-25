#!/usr/bin/env bash
# Phase 17 Block A (the real instrument/gate, per docs/phases/PHASE17_PREREG.md):
# log pi_hat_m(y|x) under the marginal and answer-only estimators, for 12
# models x 75 targets x U(x). Cost estimated first (results/phase17/
# block_a_forward_pass_cost_estimate.json, ~4-12h depending on safety
# margin, well under the pre-reg's 24h gate) before this was launched.
#
# research/phase17_block_a_forward_pass.py is itself resume-safe (skips
# any (model,target) pair already in results/phase17/logp_universe.jsonl
# on restart); this wraps it in the project's standard retry-on-crash loop.
set -u
cd /users/gsundar/projects/MIRA
mkdir -p run_logs results/phase17

LOG="run_logs/phase17_block_a_forward_pass.log"
NTFY_TOPIC="mira-g5x7k2-status"

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; [ -f .env ] && source .env; set +a

notify() { curl -s -m 10 -d "$1" "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null 2>&1 || true; }

notify "Phase17 Block A forward-pass (Gate G1 instrument): starting, estimated 4-12h."

for attempt in $(seq 1 20); do
    echo "=== forward_pass attempt ${attempt} $(date -u) ===" | tee -a "${LOG}"
    uv run python -u research/phase17_block_a_forward_pass.py 2>&1 | tee -a "${LOG}"
    rc=${PIPESTATUS[0]}
    echo "=== forward_pass attempt ${attempt} exited rc=${rc} $(date -u) ===" | tee -a "${LOG}"
    if [ "${rc}" -eq 0 ]; then
        notify "Phase17 Block A forward-pass: DONE (attempt ${attempt})."
        exit 0
    fi
    echo "forward_pass rc=${rc} -- restarting in 30s (attempt $((attempt+1))/20)" | tee -a "${LOG}"
    sleep 30
done
notify "Phase17 Block A forward-pass: FAILED after 20 attempts."
exit 1
