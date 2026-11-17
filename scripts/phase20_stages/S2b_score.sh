#!/usr/bin/env bash
# Stage 2b (GPU ~1 h): Phase 19 Blocks G, E, F1 unchanged on the new SFT models
# (research/phase20_p19.py --stage 2b), then start the SFT-side CPU analyses in
# the background. Deferred (exit 3) until the Stage 2 samples pass verification
# (they may be on RunPod); the chain retries it after later stages.
set -u
cd /users/gsundar/projects/MIRA
export PYTHONPATH=. PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$HOME/envs/mira-vllm/bin/python" research/phase20_verify_samples.py stage2 >/dev/null 2>&1 \
  || { echo "[S2b] Stage 2 samples not accepted yet -> deferred"; exit 3; }
for b in G E F1; do
  uv run python -u research/phase20_p19.py --stage 2b --block "$b" 2>&1 | tee -a run_logs/phase20_p19_2b.log
  [ "${PIPESTATUS[0]}" -eq 0 ] || exit 1
done
tmux new-session -d -s phase20_cpu_sft "bash scripts/phase20_cpu.sh sft" 2>/dev/null \
  || echo "[S2b] phase20_cpu_sft already running"
