import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Headless backend for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import plotly.graph_objects as go


def main():
    parser = argparse.ArgumentParser(description="MIRA Manifold Visualization (Headless)")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default=None,
        help="Path to the model checkpoint directory (e.g., runs/grpo.../checkpoint-400)."
    )
    parser.add_argument(
        "--out-dir", 
        type=str, 
        default=".",
        help="Directory to save the output files."
    )
    args = parser.parse_args()

    # =====================================================================
    # CONFIGURATION & LOADING
    # =====================================================================
    MODEL_ID = "Qwen/Qwen3-8B"

    print("Loading model and tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )


    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="auto", 
        quantization_config=quant_config,
        output_hidden_states=True 
    )

    if args.data_dir and Path(args.data_dir).exists():
        print(f"Loading adapter from {args.data_dir}...")
        model.load_adapter(args.data_dir)
    elif args.data_dir:
        print(f"Warning: Checkpoint directory '{args.data_dir}' not found. Running base model.")
    else:
        print("No --data-dir provided. Running with base model weights.")

    model.eval()

    # =====================================================================
    # DATA COLLECTION
    # =====================================================================
    prompts = [
        "A standard solid state synthesis route requires heating to 1000 C.",
        "Another completely different text but we just want to track sequence position.",
        "To synthesize this material, mix the precursors thoroughly in an agate mortar.",
        "Calcination should be performed in an alumina crucible under ambient air.",
        "The mixture was heated at a ramp rate of 5 C per minute up to target.",
        "Sintering at high temperature ensures optimal phase purity and density.",
    ]

    print(f"Running forward passes across {len(prompts)} strings...")
    layer_to_inspect = -1 
    max_seq_len = 15

    all_hidden_states = []

    with torch.no_grad():
        for text in prompts:
            inputs = tok(text, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            
            layer_activations = outputs.hidden_states[layer_to_inspect][0] 
            
            if layer_activations.shape[0] >= max_seq_len:
                all_hidden_states.append(layer_activations[:max_seq_len].cpu().float().numpy())

    activations_np = np.array(all_hidden_states)
    mean_activations = activations_np.mean(axis=0)

    # =====================================================================
    # NATIVE PYTORCH PCA
    # =====================================================================
    print("Computing PCA dimensions natively via low-rank SVD...")
    X = torch.tensor(mean_activations, dtype=torch.float32)
    X_centered = X - X.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(X_centered, q=3)
    pca_result = torch.matmul(X_centered, V[:, :3]).numpy()

    # Calculate total variance explained
    var_explained = (S**2).sum() / (torch.var(X_centered, dim=0, correction=0).sum() * (X.shape[0] - 1))
    print(f"Projected shape: {pca_result.shape}")
    print(f"Estimated Explained Variance: {var_explained.item() * 100:.2f}%")

    # =====================================================================
    # EXPORT FILES
    # =====================================================================
    # Extract the checkpoint name to label the output files cleanly
    ckpt_name = Path(args.data_dir).name if args.data_dir else "base_model"
    
    # --- ADD DIRECTORY CREATION AND PATH MAPPING ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / f"mira_manifold_{ckpt_name}.csv"
    png_path = out_dir / f"mira_manifold_{ckpt_name}.png"
    html_path = out_dir / f"mira_manifold_{ckpt_name}.html"

    print(f"Exporting raw coordinates to {csv_path}...")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Token_Index", "PC1", "PC2", "PC3"])
        for i, row in enumerate(pca_result):
            writer.writerow([i, row[0], row[1], row[2]])

    print(f"Rendering static image to {png_path}...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], 
                         c=range(max_seq_len), cmap='viridis', s=50, depthshade=True)
    ax.plot(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], color='gray', alpha=0.5)

    ax.set_title(f"MIRA Topology (Layer {layer_to_inspect}) - {ckpt_name}")
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    fig.colorbar(scatter, label='Token Sequence Position', shrink=0.5)

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Generating interactive Plotly file to {html_path}...")
    fig_plotly = go.Figure(data=[go.Scatter3d(
        x=pca_result[:, 0], y=pca_result[:, 1], z=pca_result[:, 2],
        mode='lines+markers',
        marker=dict(size=7, color=list(range(max_seq_len)), colorscale='Viridis', opacity=0.9),
        line=dict(color='rgba(50, 50, 150, 0.6)', width=3)
    )])

    fig_plotly.update_layout(
        title=f"MIRA Topology (Layer {layer_to_inspect}) - {ckpt_name}",
        scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
        template="plotly_white"
    )
    fig_plotly.write_html(html_path)

    print("\nDONE. All files saved to current directory.")

if __name__ == "__main__":
    main()