#!/usr/bin/env bash
# Stage 3: E1 rerun from sft-rssft-full. scripts/run_gdpo_e1.sh unchanged except
# RS_SFT_INIT and the tag; SAVE_TOTAL_LIMIT=1000 keeps every checkpoint across
# automatic restarts (no training-math change). Verified before launch.
set -u
cd /users/gsundar/projects/MIRA
export PYTHONPATH=.
TAG=gdpo-e1-ranker-rssft-full
export RS_SFT_INIT=runs/sft-qlora-sft-rssft-full/final SAVE_TOTAL_LIMIT=1000
[ -f "$RS_SFT_INIT/adapter_model.safetensors" ] || { echo "missing $RS_SFT_INIT"; exit 1; }
# pre-launch verification: the exact argv the launcher builds yields save_total_limit=1000
uv run python - <<PY || exit 1
import sys, inspect
sys.argv = ["train.py","gdpo","--adapter","qlora","--model","Qwen/Qwen3-8B","--data-dir","data/rl_run3",
            "--data-prefix","rl3","--lora-r","16","--lora-alpha","32","--kl-beta","0.0","--scorer","ranker",
            "--validator-version","2","--use-vllm","off","--max-steps","300","--save-total-limit","1000",
            "--seed","42","--tag","$TAG","--init-from","$RS_SFT_INIT","--fresh-restart"]
import train
a = train.parse_args()
assert a.save_total_limit == 1000, a.save_total_limit
import experiments.grpo as g
src = inspect.getsource(g)
assert 'save_total_limit=getattr(self.args, "save_total_limit", None) or 3' in src
print("save_total_limit verified: 1000 (all 3 checkpoints kept across restarts)")
PY
grep -q 'save-total-limit "${SAVE_TOTAL_LIMIT}"' scripts/run_gdpo_e1.sh || { echo "launcher lacks --save-total-limit"; exit 1; }
bash scripts/run_gdpo_e1.sh "$TAG" 42 || exit 1
for c in 100 200 300; do
  [ -f "runs/gdpo-qlora-${TAG}/checkpoint-${c}/adapter_model.safetensors" ] || { echo "checkpoint-${c} missing"; exit 1; }
done
