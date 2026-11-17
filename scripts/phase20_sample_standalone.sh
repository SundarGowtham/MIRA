#!/usr/bin/env bash
# Phase 20: run ONE sampling stage on any machine (this one or a rented GPU).
#
#   Usage: bash scripts/phase20_sample_standalone.sh <stage>
#     stage2      the three new SFT models, ASTRAL + held-out, 32/target
#     stage4_ref  the sft-rssft-full reference, 232/target
#     stage4_e1   E1-full @100/@200/@300, 32/target
#     stage5      temperature-fix resamples of the four old models (ASTRAL)
#     stage6      sft-corpus-full, 32/target
#
# Needs: a clone of this repo; the adapter folders the stage lists
# (`--list-outputs` prints them) at the same relative paths; a Python with
# vllm==0.23.0 (scripts/phase20_vllm_requirements.txt). Set VLLM_PY to that
# python if it is not ~/envs/mira-vllm/bin/python. Base Qwen/Qwen3-8B is
# downloaded from Hugging Face on first use.
#
# Writes data/novelty/phase20/*.jsonl (resume-safe), each row carrying its
# sampling settings, then runs the same acceptance check the chain uses.
# Full RunPod procedure: docs/phases/PHASE20_RUNPOD.md.
set -u
STAGE="${1:?usage: phase20_sample_standalone.sh <stage2|stage4_ref|stage4_e1|stage5|stage6>}"
cd "$(dirname "$0")/.."
mkdir -p run_logs
PY="${VLLM_PY:-$HOME/envs/mira-vllm/bin/python}"
[ -x "$PY" ] || PY="$(command -v python)"
LOG="run_logs/phase20_sample_${STAGE}.log"
export PYTHONPATH=.
export PATH="$(dirname "$PY"):$PATH"   # vLLM needs the env's ninja on PATH

"$PY" -c "import vllm,sys; assert vllm.__version__=='0.23.0', vllm.__version__" \
    || { echo "need vllm==0.23.0 in $PY (see scripts/phase20_vllm_requirements.txt)"; exit 2; }
"$PY" data_curation/phase20_sampling.py --job "$STAGE" --list-outputs

for attempt in $(seq 1 10); do
    echo "=== ${STAGE} attempt ${attempt} $(date -u) ===" | tee -a "$LOG"
    "$PY" -u data_curation/phase20_sampling.py --job "$STAGE" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] && break
    echo "rc=${rc}, retrying in 30s" | tee -a "$LOG"; sleep 30
done
"$PY" research/phase20_verify_samples.py "$STAGE" 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
