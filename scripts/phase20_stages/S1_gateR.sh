#!/usr/bin/env bash
# Gate R after each SFT run: R2 sampling (mira-vllm), then R1 + R2 check (main env).
set -u
TAG="$1"
cd /users/gsundar/projects/MIRA
export PYTHONPATH=.
"$HOME/envs/mira-vllm/bin/python" -u research/phase20_gate_r.py --tag "$TAG" --sample || exit 1
uv run python research/phase20_gate_r.py --tag "$TAG" --check
