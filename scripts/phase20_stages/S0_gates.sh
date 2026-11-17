#!/usr/bin/env bash
# Stage 0 gates: Gate T (masking) on every dataset, Gate M (memory) results
# present and passing for every dataset. Gate M already ran in Stage 0
# (results/phase20/gate_m_*.json); Gate T is re-run here so a data or code
# change is caught.
set -u
cd /users/gsundar/projects/MIRA
export PYTHONPATH=.
uv run python research/phase20_gate_t.py || exit 1
for k in rs_sft e3 e3c sft_v3; do
  uv run python -c "import json,sys; d=json.load(open('results/phase20/gate_m_$k.json')); sys.exit(0 if d['pass_'] else 1)" \
    || { echo "Gate M missing/failed for $k"; exit 1; }
done
uv run python tests/test_sft_masking.py || exit 1
