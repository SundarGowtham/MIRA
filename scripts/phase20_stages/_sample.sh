#!/usr/bin/env bash
# Verify-or-sample one sampling job. Accept files already in place (e.g. from
# RunPod) if their recorded settings match the spec; if the job is reserved
# for RunPod and not yet returned, exit 3 (deferred); otherwise sample here.
set -u
JOB="$1"
cd /users/gsundar/projects/MIRA
export PYTHONPATH=.
VPY="$HOME/envs/mira-vllm/bin/python"
if "$VPY" research/phase20_verify_samples.py "$JOB"; then
  echo "[sample] ${JOB}: files present and accepted; skipping sampling"; exit 0
fi
if [ -f "runs/phase20/RESERVE_${JOB}" ]; then
  echo "[sample] ${JOB}: reserved for RunPod, not yet returned -> deferred"; exit 3
fi
bash scripts/phase20_sample_standalone.sh "$JOB"
