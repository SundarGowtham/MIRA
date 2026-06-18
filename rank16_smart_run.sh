# #!/bin/bash

# # Configuration
# TARGET_FILE="run_logs/v3/rank32/rank32_seed42_rerun.log"
# TARGET_WORD="Results at:"
# SCRIPT_LOG="rank16_evals_smart_run.log"

# # Setup logging function to handle clean redirection
# log_msg() {
#     echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$SCRIPT_LOG"
# }

# touch "$TARGET_FILE"
# log_msg "Starting active monitoring on '$TARGET_FILE'..."

# # Use Process Substitution to keep the loop in the main shell context
# while read -r line; do
#     if [[ "$line" =~ "$TARGET_WORD" ]]; then
#         log_msg "[FOUND] Target word '$TARGET_WORD' detected. Kicking off python evaluations..."
        
#         # Combined Environment Variables and Execution Block
#         export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        
#         log_msg "[EXEC] Starting Eval 1: seed1337"
#         uv run python -u evaluate_batched.py \
#             --checkpoint runs/sft-qlora-v3-rank16-seed1337/final \
#             --model Qwen/Qwen3-8B \
#             --data-dir data/sft --data-prefix sft \
#             --split test --tag v3-rank16-seed1337 --batch-size 4 \
#             --skip-thermo 2>&1 | tee run_logs/v3/rank16/rank16_seed1337_rerun.log

#         log_msg "[EXEC] Starting Eval 2: seed7"
#         uv run python -u evaluate_batched.py \
#             --checkpoint runs/sft-qlora-v3-rank16-seed7/final \
#             --model Qwen/Qwen3-8B \
#             --data-dir data/sft --data-prefix sft \
#             --split test --tag v3-rank16-seed7 \
#             --batch-size 4 \
#             --skip-thermo 2>&1 | tee run_logs/v3/rank16/rank16_seed7_rerun.log

#         log_msg "[SUCCESS] All evaluation steps completed."
#         break # Successfully breaks loop and terminates tail process below
#     fi
# done < <(tail -Fn0 "$TARGET_FILE")

# log_msg "Monitor script execution finished."
