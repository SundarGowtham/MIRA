#!/usr/bin/env bash
# Analysis chain: temperature sweep -> PD attention -> pass@k baseline.
# Each stage retries up to 30x and resumes from its own incremental output.
# Logs: run_logs/<stage>.log (+ master). Phone ping via ntfy at the end.
set -u
cd /users/gsundar/projects/MIRA
mkdir -p run_logs
MASTER=run_logs/analysis_chain.log
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NTFY_TOPIC="mira-g5x7k2-status"

notify() { curl -s -m 10 -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true; }

run_stage() {
    local name="$1"; shift
    local log="run_logs/${name}.log"
    for attempt in $(seq 1 30); do
        echo "=== [$name] attempt $attempt $(date -u) ===" | tee -a "$MASTER" "$log"
        "$@" 2>&1 | tee -a "$log"
        rc=${PIPESTATUS[0]}
        echo "=== [$name] attempt $attempt rc=$rc $(date -u) ===" | tee -a "$MASTER" "$log"
        if [ "$rc" -eq 0 ]; then
            echo "=== [$name] STAGE DONE $(date -u) ===" | tee -a "$MASTER"
            return 0
        fi
        sleep 60
    done
    return 1
}

echo "=== analysis chain starting $(date -u) ===" | tee -a "$MASTER"
run_stage temp_diversity_sweep uv run python sweep_temperature_diversity.py \
    || { notify "MIRA chain FAILED at temp sweep (see run_logs/temp_diversity_sweep.log)"; exit 1; }
run_stage pd_attention uv run python analyze_pd_attention.py \
    || { notify "MIRA chain FAILED at pd attention"; exit 1; }
run_stage passk_baseline uv run python probe_passk.py \
    || { notify "MIRA chain FAILED at pass@k baseline"; exit 1; }
echo "=== analysis chain DONE $(date -u) ===" | tee -a "$MASTER"
notify "MIRA analysis chain DONE: temp sweep + PD attention + pass@k all complete"
exit 0
