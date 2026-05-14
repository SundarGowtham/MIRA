#!/bin/bash
# run_evals_sequential.sh
# Waits for the currently-running checkpoint-200 eval to finish,
# then runs 400 and 600 sequentially. Detects completion by reading
# the output JSON's "n_examples" field.

set -e   # exit on any error

PROJECT_DIR="$HOME/projects/MIRA"
cd "$PROJECT_DIR"

EXPECTED_N=811        # full test set size
POLL_INTERVAL=60      # seconds between completion checks
SAFETY_BUFFER=120     # extra seconds to wait after detection

# Returns 0 (success) when the given JSON exists AND has >= EXPECTED_N records
wait_for_eval() {
    local tag="$1"
    local json_path="eval_results/eval_${tag}_test.json"

    echo "[$(date +%H:%M:%S)] Waiting for $json_path ..."

    while true; do
        if [ -f "$json_path" ]; then
            n=$(python -c "
import json
try:
    with open('$json_path') as f:
        d = json.load(f)
    print(d.get('aggregate', {}).get('n_examples', 0))
except Exception:
    print(0)
" 2>/dev/null)
            if [ "$n" -ge "$EXPECTED_N" ]; then
                echo "[$(date +%H:%M:%S)]   $tag: complete ($n/$EXPECTED_N records)"
                # Final guard: process might still be writing — give it 2 min
                echo "[$(date +%H:%M:%S)]   Safety buffer: ${SAFETY_BUFFER}s"
                sleep "$SAFETY_BUFFER"
                return 0
            fi
            echo "[$(date +%H:%M:%S)]   $tag: $n/$EXPECTED_N records, waiting..."
        else
            echo "[$(date +%H:%M:%S)]   $tag: no JSON yet, waiting..."
        fi
        sleep "$POLL_INTERVAL"
    done
}

run_eval() {
    local ckpt="$1"
    local tag="$2"
    local logfile="$3"

    echo ""
    echo "============================================================"
    echo "[$(date +%H:%M:%S)] Starting eval: $tag"
    echo "============================================================"

    uv run python -u evaluate_batched.py \
        --checkpoint "$ckpt" \
        --model Qwen/Qwen3-8B \
        --tag "$tag" \
        --skip-thermo \
        --batch-size 8 \
        2>&1 | tee "$logfile"

    echo "[$(date +%H:%M:%S)] Finished eval: $tag"
}

# ----- Main sequence -----

echo "Sequential eval runner started at $(date)"
echo "Project: $PROJECT_DIR"
echo "Expected test set size: $EXPECTED_N"
echo ""

# Step 1: wait for the currently-running checkpoint-200 eval to finish
wait_for_eval "sft-grpo-step200"

# Step 2: run checkpoint-400
run_eval \
    "runs/grpo-qlora-stage2-stage2-grpo/checkpoint-400" \
    "sft-grpo-step400" \
    "runs/eval_sft_grpo_400.log"

# Step 3: run checkpoint-600
run_eval \
    "runs/grpo-qlora-stage2-stage2-grpo/checkpoint-600" \
    "sft-grpo-step600" \
    "runs/eval_sft_grpo_600.log"

echo ""
echo "============================================================"
echo "[$(date +%H:%M:%S)] All evals complete"
echo "============================================================"
ls -la eval_results/