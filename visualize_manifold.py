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

import pandas as pd
import plotly.express as px


import json
import re
from collections import defaultdict


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
    parser.add_argument(
        "--datasetpath", 
        type=Path,
        default=None,
        help="path to sft dataset"
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

    # # =====================================================================
    # # DATA COLLECTION (MIRA SFT_V2 DATASET)
    # # =====================================================================

    

    # # Dictionary: {1000: [tensor, tensor], 1300: [tensor, tensor, ...]}
    # temp_activations = defaultdict(list)
    
    # # Regex to capture exactly the number inside T=...°C
    # temp_pattern = re.compile(r'T=(\d{3,4})°C') 
    
    
    # data_path = args.datasetpath
    # if not data_path.exists():
    #     print(f"Error: Could not find {data_path}. Run this from project root.")
    #     return

    # print(f"Parsing {data_path} for temperature manifolds...")


    # # How many prompts to process (< 500 is enough)
    # layer_to_inspect = -1
    # MAX_PROMPTS = 300 
    # processed_count = 0

    # with open(data_path, 'r') as f:
    #     for line in f:
    #         if processed_count >= MAX_PROMPTS:
    #             break
                
    #         record = json.loads(line)
    #         # Feed the model the exact context it saw during training
    #         text = record["prompt"] + record["completion"]
            
    #         match = temp_pattern.search(text)
    #         if not match:
    #             continue # Skip routes without a HeatingOperation
                
    #         temp_val = int(match.group(1))
            
    #         # Find the character index where the number starts
    #         char_start = match.start(1)
            
    #         # HACK: Tokenize the text exactly up to the start of the number.
    #         # The length of this prefix tells us the token index of the temperature!
    #         prefix_tokens = tok.encode(text[:char_start], add_special_tokens=False)
    #         target_idx = len(prefix_tokens)
            
    #         # Run the forward pass
    #         inputs = tok(text, return_tensors="pt").to(model.device)
            
    #         with torch.no_grad():
    #             outputs = model(**inputs)
                
    #         # Extract the activation from the final layer at the exact temperature token
    #         layer_activations = outputs.hidden_states[layer_to_inspect][0]
            
    #         # Safety check to ensure we don't index out of bounds
    #         if target_idx < layer_activations.shape[0]:
    #             specific_activation = layer_activations[target_idx].cpu().float().numpy()
    #             temp_activations[temp_val].append(specific_activation)
    #             processed_count += 1
                
    #             if processed_count % 50 == 0:
    #                 print(f"Processed {processed_count} temperature vectors...")

    # print(f"Extracted {processed_count} valid temperature vectors across {len(temp_activations)} unique temperatures.")

    # # Average them and prepare for PCA
    # ordered_temps = sorted(temp_activations.keys())
    # mean_activations = []
    # labels = []

    # for t in ordered_temps:
    #     # We only plot temperatures that have at least 2 examples to ensure we 
    #     # are plotting the concept of the temperature, not the noise of a specific prompt
    #     if len(temp_activations[t]) >= 2:
    #         mean_vec = np.mean(temp_activations[t], axis=0)
    #         mean_activations.append(mean_vec)
    #         labels.append(t) # Save the actual temperature integer for coloring the plot

    # mean_activations = np.array(mean_activations)


    # =====================================================================
    # DATA COLLECTION (ALL LAYERS - STERILE PROBES)
    # =====================================================================
    print("Running sterile temperature probes across ALL layers...")

    temperatures = list(range(500, 1501, 10))
    base_prompt = "<think> The target material requires a standard solid-state route. We will mix the precursors and heat them in a furnace. </think>\n<operations>\n1. HeatingOperation | T="

    num_layers = len(model.config.encoder_layers) if hasattr(model.config, 'encoder_layers') else model.config.num_hidden_layers + 1
    temp_layer_acts = {l: [] for l in range(num_layers)}
    labels = []

    with torch.no_grad():
        for t in temperatures:
            text = f"{base_prompt}{t}°C"
            inputs = tok(text, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            
            for l, hidden_state in enumerate(outputs.hidden_states):
                # THE FIX: Extract the vector at the absolute LAST token (-1).
                # At this point, causal attention has processed the entire integer.
                vec = hidden_state[0, -1].cpu().float().numpy()
                temp_layer_acts[l].append(vec)
            
            labels.append(t)


    
    # =====================================================================
    # NATIVE PYTORCH PCA (GLOBAL FIXED AXES) (LOOPING OVER ALL LAYERS)
    # =====================================================================

    print("Computing a GLOBAL PCA coordinate system to lock the axes...")
    
    # Combine all vectors to find the universal principal components
    all_vecs = []
    for l in range(num_layers):
        all_vecs.extend(temp_layer_acts[l])
        
    X_global = torch.tensor(np.array(all_vecs), dtype=torch.float32)
    X_global_centered = X_global - X_global.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(X_global_centered, q=3)

    print("Projecting each layer into the fixed global space...")
    all_records = []
    global_max = 0

    for l in range(num_layers):
        if len(temp_layer_acts[l]) == 0:
            continue
            
        X_layer = torch.tensor(np.array(temp_layer_acts[l]), dtype=torch.float32)
        X_layer_centered = X_layer - X_layer.mean(dim=0, keepdim=True)
        
        # Project using the fixed GLOBAL right-singular vectors (V)
        pca_result = torch.matmul(X_layer_centered, V[:, :3]).numpy()
        
        # Track the absolute maximum coordinate to build a perfect bounding box later
        layer_max = np.max(np.abs(pca_result))
        if layer_max > global_max:
            global_max = layer_max

        for i, t in enumerate(labels):
            all_records.append({
                "Layer": l,
                "Temperature": t,
                "PC1": pca_result[i, 0],
                "PC2": pca_result[i, 1],
                "PC3": pca_result[i, 2]
            })

    df = pd.DataFrame(all_records)
    
    # Pad the bounding box by 10% so the manifold doesn't touch the walls
    bbox = float(global_max * 1.1)


    # =====================================================================
    # EXPORT FILES
    # =====================================================================
    ckpt_name = Path(args.data_dir).name if args.data_dir else "base_model"
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    html_path = out_dir / f"mira_manifold_evolution_{ckpt_name}.html"

    print(f"Generating interactive layer-by-layer animation to {html_path}...")
    
    fig_plotly = px.scatter_3d(
        df, 
        x='PC1', y='PC2', z='PC3',
        color='Temperature',
        animation_frame='Layer',
        title=f"MIRA Temperature Concept Evolution - {ckpt_name}",
        color_continuous_scale='Viridis',
        # Apply the mathematically locked bounding box
        range_x=[-bbox, bbox], range_y=[-bbox, bbox], range_z=[-bbox, bbox]
    )
    
    # THE FIX: mode='lines+markers' connects the dots sequentially!
    fig_plotly.update_traces(
        mode='lines+markers',
        marker=dict(size=5, opacity=0.9),
        line=dict(width=4)
    )
    
    fig_plotly.update_layout(template="plotly_white")
    fig_plotly.write_html(html_path)

    print(f"\nDONE. Saved animation to {html_path}")

if __name__ == "__main__":
    main()