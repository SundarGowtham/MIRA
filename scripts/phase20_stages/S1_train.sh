#!/usr/bin/env bash
# Stage 1: train one new SFT model with the --sft-full path. Same data files,
# init (Base), QLoRA, LoRA r16/a32, 3 epochs, lr 2e-4, seed 42 as the old run.
set -u
TAG="$1"
cd /users/gsundar/projects/MIRA
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; [ -f .env ] && source .env; set +a
case "$TAG" in
  sft-rssft-full)  DD=data/rs_sft; PF=rs_sft ;;
  sft-e3-full)     DD=data/e3;     PF=e3 ;;
  sft-e3c-full)    DD=data/e3c;    PF=e3c ;;
  sft-corpus-full) DD=data/sft_v3; PF=sft ;;
  *) echo "unknown tag $TAG"; exit 1 ;;
esac
OUT="runs/sft-qlora-${TAG}"
if [ -f "${OUT}/final/adapter_model.safetensors" ]; then echo "${OUT}/final exists; not retraining"; exit 0; fi
for attempt in 1 2 3; do
  echo "=== ${TAG} attempt ${attempt} $(date -u) ==="
  uv run python -u train.py sft --adapter qlora --model Qwen/Qwen3-8B \
      --data-dir "$DD" --data-prefix "$PF" --lora-r 16 --lora-alpha 32 \
      --seed 42 --tag "$TAG" --sft-full 2>&1 | tee -a "run_logs/phase20_train_${TAG}.log"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] && [ -f "${OUT}/final/adapter_model.safetensors" ] && exit 0
  echo "${TAG} rc=${rc}; retrying in 60s"; sleep 60
done
exit 1
