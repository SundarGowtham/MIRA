#!/usr/bin/env bash
# Phase 17 Block A GPU regeneration (Gowtham-approved, generation only,
# no training). Order per regular Claude: Phase12@300 ASTRAL, E1@300
# ASTRAL, GDPO-300 both sets (ASTRAL then held-out), Corpus SFT both
# sets (ASTRAL then held-out). ASTRAL legs match the ORIGINAL
# astral_model_generations.py config exactly (HF transformers, temp=1.0,
# top_p=0.95, max_new_tokens=8192, n=32/target, chunked 8 at a time) --
# this is a REPLICATION check, so the original pipeline must be used
# unchanged. Held-out legs use the project's separate, already-
# established held-out convention (vLLM, temp=0.9, mira-vllm env) via
# the new data_curation/phase17_block_a_heldout_sampling.py.
#
# New filenames only. Never swaps a regenerated number in for the
# original headline (findings 16/18/22/30 keep their original source
# files; this script never writes to them).
set -u
cd /users/gsundar/projects/MIRA
mkdir -p run_logs results/phase17

LOG="run_logs/phase17_block_a_regen.log"
NTFY_TOPIC="mira-g5x7k2-status"

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; [ -f .env ] && source .env; set +a

notify() { curl -s -m 10 -d "$1" "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null 2>&1 || true; }

run_astral_hf() {
    local checkpoint="$1" tag="$2" out="$3"
    for attempt in $(seq 1 20); do
        echo "=== ASTRAL ${tag} attempt ${attempt} $(date -u) ===" | tee -a "${LOG}"
        uv run python -u research/astral_model_generations.py \
            --checkpoint "${checkpoint}" --tag "${tag}" --out "${out}" \
            --n-samples 32 2>&1 | tee -a "${LOG}"
        rc=${PIPESTATUS[0]}
        echo "=== ASTRAL ${tag} attempt ${attempt} exited rc=${rc} $(date -u) ===" | tee -a "${LOG}"
        if [ "${rc}" -eq 0 ]; then
            notify "Phase17 BlockA ASTRAL ${tag}: DONE (attempt ${attempt})"
            return 0
        fi
        echo "ASTRAL ${tag} rc=${rc} -- restarting in 30s (attempt $((attempt+1))/20)" | tee -a "${LOG}"
        sleep 30
    done
    notify "Phase17 BlockA ASTRAL ${tag}: FAILED after 20 attempts"
    return 1
}

run_heldout_vllm() {
    local job="$1"
    for attempt in $(seq 1 20); do
        echo "=== held-out ${job} attempt ${attempt} $(date -u) ===" | tee -a "${LOG}"
        bash -c "source ~/envs/mira-vllm/bin/activate && PYTHONPATH=. python -u data_curation/phase17_block_a_heldout_sampling.py --job ${job}" 2>&1 | tee -a "${LOG}"
        rc=${PIPESTATUS[0]}
        echo "=== held-out ${job} attempt ${attempt} exited rc=${rc} $(date -u) ===" | tee -a "${LOG}"
        if [ "${rc}" -eq 0 ]; then
            notify "Phase17 BlockA held-out ${job}: DONE (attempt ${attempt})"
            return 0
        fi
        echo "held-out ${job} rc=${rc} -- restarting in 30s (attempt $((attempt+1))/20)" | tee -a "${LOG}"
        sleep 30
    done
    notify "Phase17 BlockA held-out ${job}: FAILED after 20 attempts"
    return 1
}

notify "Phase17 Block A regeneration: starting (6 legs, ~40h estimated)."

run_astral_hf "runs/gdpo-qlora-gdpo-phase12-rssft-beta0/checkpoint-300" "phase12_beta0_ckpt300_replication" \
    "results/phase17/block_a_astral_gen_n32_phase12_beta0_ckpt300_replication.json" || exit 1

run_astral_hf "runs/gdpo-qlora-gdpo-e1-ranker-rssft/final" "e1_replication" \
    "results/phase17/block_a_astral_gen_n32_e1_replication.json" || exit 1

run_astral_hf "runs/gdpo-qlora-gdpo-v3/checkpoint-300" "gdpo300_replication" \
    "results/phase17/block_a_astral_gen_n32_gdpo300_replication.json" || exit 1

run_heldout_vllm "gdpo300" || exit 1

run_astral_hf "runs/sft-qlora-sft-v3-2nd-rank16/final" "corpus_sft_replication" \
    "results/phase17/block_a_astral_gen_n32_corpus_sft_replication.json" || exit 1

run_heldout_vllm "corpus_sft" || exit 1

echo "ALL DONE $(date -u)" | tee -a "${LOG}"
notify "Phase17 Block A regeneration: ALL 6 LEGS DONE."
exit 0
