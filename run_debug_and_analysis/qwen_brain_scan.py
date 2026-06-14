import torch
import numpy as np
import seaborn as sns

import matplotlib
matplotlib.use('Agg') # Forces headless rendering on your server
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen3-8B" 

# The prompt we want to analyze
PROMPT = "Precursors: BaCO3, TiO2. Target: BaTiO3. To balance the Ti, we need 1 "

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    device_map="auto", 
    output_attentions=True # CRITICAL: This exposes the "brain waves"
)

def generate_brain_scan():
    print(f"Analyzing prompt: '{PROMPT}'")
    
    # 1. Tokenize the input
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids[0]
    tokens = [tokenizer.decode([tok]) for tok in input_ids]
    
    # Find the index of the " TiO2" token in the prompt
    # We want to see which heads are looking at this specific chemical
    target_token_idx = -1
    for i, tok in enumerate(tokens):
        if "Ti" in tok:
            target_token_idx = i
            break
            
    if target_token_idx == -1:
        print("Could not find target token. Just using the middle token.")
        target_token_idx = len(tokens) // 2

    print(f"Tracking attention to token: '{tokens[target_token_idx]}'")

    # 2. Run the forward pass to get attention matrices
    with torch.no_grad():
        outputs = model(**inputs)
    
    # outputs.attentions is a tuple of (Layers), each containing (Batch, Heads, SeqLen, SeqLen)
    attentions = outputs.attentions
    
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    
    # 3. Build the Brain Scan Matrix (Layer vs. Head)
    # We are looking at the VERY LAST token predicting the next word, 
    # and seeing how intensely it "looks back" at the target precursor.
    brain_scan_matrix = np.zeros((num_heads, num_layers))
    
    for layer in range(num_layers):
        for head in range(num_heads):
            # Attention from the LAST token to the TARGET precursor token
            attn_score = attentions[layer][0, head, -1, target_token_idx].item()
            brain_scan_matrix[head, layer] = attn_score

    # 4. Render the Heat Map
    print("Rendering visualization...")
    plt.style.use("dark_background") # Makes it look like a glowing scan
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Use "magma" or "inferno" for that high-contrast heat/scan aesthetic
    sns.heatmap(
        brain_scan_matrix, 
        cmap="magma", 
        linewidths=0.5, 
        linecolor='black',
        cbar_kws={'label': f'Attention Intensity to "{tokens[target_token_idx]}"'}
    )

    plt.title(f"Qwen Element Tracking Circuit\n(Which heads activate to find '{tokens[target_token_idx]}')", fontsize=18, pad=20, color='white')
    plt.xlabel("Network Layer (Depth)", fontsize=14, color='white')
    plt.ylabel("Attention Head", fontsize=14, color='white')
    
    # Clean up the axes
    ax.invert_yaxis()
    plt.tight_layout()
    
    # Save the artifact
    output_file = "qwen_stoichiometry_circuit.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"\nSUCCESS: Brain scan saved as {output_file}")
    
    # If running interactively, show it
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    generate_brain_scan()