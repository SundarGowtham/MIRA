import argparse
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from scipy.interpolate import splprep, splev
import plotly.graph_objects as go

def main():
    parser = argparse.ArgumentParser(description="MIRA Manifold Spline Fitter")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--layer", type=int, default=-5, help="Layer to analyze (e.g., -5 for layer 26)")
    args = parser.parse_args()

    MODEL_ID = "Qwen/Qwen3-8B"
    print(f"Loading tokenizer and model...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)


    # Re-apply the QLoRA quantization config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    
    # Keeping it fast: load in bfloat16 for this specific targeted pass
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="auto", 
        quantization_config=quant_config, 
        output_hidden_states=True
    )

    if args.data_dir and Path(args.data_dir).exists():
        print(f"Loading adapter from {args.data_dir}...")
        model.load_adapter(args.data_dir)
    model.eval()

    # 1. STERILE PROBES (Targeted Layer Only)
    print(f"Extracting temperature vectors for Layer {args.layer}...")
    temperatures = list(range(500, 1501, 10))
    base_prompt = "<think> The target material requires a standard solid-state route. We will mix the precursors and heat them in a furnace. </think>\n<operations>\n1. HeatingOperation | T="
    
    mean_activations = []
    with torch.no_grad():
        for t in temperatures:
            text = f"{base_prompt}{t}°C"
            inputs = tok(text, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            # Extract last token (-1)
            vec = outputs.hidden_states[args.layer][0, -1].cpu().float().numpy()
            mean_activations.append(vec)

    # 2. PCA PROJECTION
    X = torch.tensor(np.array(mean_activations), dtype=torch.float32)
    X_centered = X - X.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(X_centered, q=3)
    pca_result = torch.matmul(X_centered, V[:, :3]).numpy()

    # Normalize coordinates to [-1, 1]
    max_val = np.max(np.abs(pca_result))
    pts = pca_result / max_val
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # 3. FIT 3D B-SPLINE
    print("Fitting mathematical spline to the manifold...")
    # splprep creates a parameterized B-spline. s=0.05 allows slight smoothing.
    tck, u = splprep([x, y, z], s=0.05)
    
    # Generate 1000 points along the perfect mathematical curve
    u_fine = np.linspace(0, 1, 1000)
    x_fine, y_fine, z_fine = splev(u_fine, tck)

    # 4. CALCULATE "JITTER SCORE" (Mean Squared Error from Spline)
    # We compare the actual model points to the smoothed spline points
    x_smooth, y_smooth, z_smooth = splev(u, tck)
    mse = np.mean((x - x_smooth)**2 + (y - y_smooth)**2 + (z - z_smooth)**2)
    
    # Scale to a readable integer (lower = smoother)
    jitter_score = int(mse * 1000000)
    
    ckpt_name = Path(args.data_dir).name if args.data_dir else "base_model"
    print(f"\n{'='*40}")
    print(f"CHECKPOINT: {ckpt_name}")
    print(f"MANIFOLD JITTER SCORE: {jitter_score}")
    print(f"(Lower is better/smoother)")
    print(f"{'='*40}\n")

    # 5. VISUALIZE BOTH
    html_path = f"spline_{ckpt_name}_layer{args.layer}.html"
    
    fig = go.Figure()
    # Plot the raw model points
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(size=6, color=temperatures, colorscale='Viridis'),
        name='LLM Raw Concept'
    ))
    # Plot the mathematical perfect curve
    fig.add_trace(go.Scatter3d(
        x=x_fine, y=y_fine, z=z_fine, mode='lines',
        line=dict(color='rgba(200, 50, 50, 0.8)', width=4),
        name='Idealized Spline'
    ))

    fig.update_layout(
        title=f"MIRA Topology (Layer {args.layer}) | Jitter Score: {jitter_score}",
        template="plotly_white",
        scene=dict(xaxis_range=[-1, 1], yaxis_range=[-1, 1], zaxis_range=[-1, 1])
    )
    fig.write_html(html_path)
    print(f"Saved visualization to {html_path}")

if __name__ == "__main__":
    main()