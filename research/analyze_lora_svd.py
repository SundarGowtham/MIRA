import torch
from pathlib import Path
from safetensors.torch import load_file
import matplotlib.pyplot as plt
import argparse


# Point this to the checkpoint-200 directory you downloaded
# CHECKPOINT_DIR = Path("runs/grpo-qlora-grpo-sft-v2-qwen/checkpoint-200")

# def analyze_svd(args):

#     print(args.data_dir)


def analyze_svd(args):
    # PEFT usually saves adapters as adapter_model.safetensors

    CHECKPOINT_DIR = args.data_dir


    weights_path = CHECKPOINT_DIR / "adapter_model.safetensors"
    if not weights_path.exists():
        weights_path = CHECKPOINT_DIR / "adapter_model.bin" # Fallback for older PEFT
        if not weights_path.exists():
            print("Could not find adapter weights.")
            return
            
    print(f"Loading weights from {weights_path.name}...")
    if weights_path.suffix == ".safetensors":
        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location="cpu")

    # Group A and B matrices by their target module
    lora_modules = {}
    for key, tensor in state_dict.items():
        if "lora_A.weight" in key:
            base_name = key.replace(".lora_A.weight", "")
            if base_name not in lora_modules: lora_modules[base_name] = {}
            lora_modules[base_name]["A"] = tensor.float() # Cast to FP32 for SVD math
        elif "lora_B.weight" in key:
            base_name = key.replace(".lora_B.weight", "")
            if base_name not in lora_modules: lora_modules[base_name] = {}
            lora_modules[base_name]["B"] = tensor.float()

    print(f"Found {len(lora_modules)} LoRA modules. Computing SVD...")
    
    # We will aggregate the singular values across all layers
    # to see the average capacity utilization
    all_svs = []
    
    for name, matrices in lora_modules.items():
        if "A" in matrices and "B" in matrices:
            # Reconstruct the full weight update: ΔW = B * A
            delta_w = matrices["B"] @ matrices["A"]
            
            # Compute Singular Values
            U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)

            # TRUNCATE TO TOP 16 (since rank r=16, the rest are mathematically zero)
            S_top16 = S[:16]

            
            # Normalize them so the max value is 1.0 for easy comparison
            normalized_S = S_top16 / S_top16.max()
            all_svs.append(normalized_S)

    if not all_svs:
        print("No valid A/B matrix pairs found.")
        return

    # Average the singular values across all adapter layers
    avg_svs = torch.stack(all_svs).mean(dim=0)
    
    print("\n--- LoRA Capacity Utilization (r=16) ---")
    for i, val in enumerate(avg_svs):
        print(f"Rank {i+1:02d}: {val.item():.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))

    return p.parse_args()



if __name__ == "__main__":
    args = parse_args()
    analyze_svd(args) 