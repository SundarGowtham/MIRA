import torch
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg') # Headless rendering for your server
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")


from peft import PeftModel # <-- NEW: Required for QLoRA checkpoints
import torch


# --- CONFIGURATION ---
PROMPT = "Reaction 1: BaTiO3 requires BaCO3. Reaction 2: SrTiO3 requires SrCO3. Reaction 3: BaTiO3 requires"



BASE_MODEL_ID = "Qwen/Qwen3-8B" 
ADAPTER_PATH = "runs/grpo-qlora-grpo-sft-v2-checkpoint200-qwen/checkpoint-2400"

print(f"Loading Base Model: {BASE_MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)




# 1. Load the underlying Base Model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, 
    device_map="auto", 
    torch_dtype=torch.bfloat16,
    output_attentions=True
)

# 2. Inject your GRPO trained weights on top
print(f"Injecting GRPO Checkpoint: {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)



def generate_induction_scan():
    print(f"Analyzing prompt: '{PROMPT}'")
    
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids[0]
    
    # Qwen tokenizer sometimes groups things weirdly, so we clean the token strings for the plot
    tokens = [tokenizer.decode([tok]).replace('\n', '\\n') for tok in input_ids]
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    attentions = outputs.attentions
    num_layers = len(attentions)
    
    # Induction heads typically live in the middle-to-late layers. 
    # We will look at Layer 18 (out of 24 for the 0.5B model)
    target_layer = int(num_layers * 0.75) 
    
    # We want the attention from the VERY LAST token (the one predicting the next word)
    # We average across all heads in this layer to capture the dominant network focus
    # last_token_attn = attentions[target_layer][0, :, -1, :].mean(dim=0).cpu().numpy()
    last_token_attn = attentions[target_layer][0, :, -1, :].mean(dim=0).float().cpu().numpy()
    
    # --- RENDER THE VISUALIZATION ---
    print("Rendering Induction Circuit visualization...")
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create a horizontal bar chart where the length/color represents attention intensity
    y_pos = np.arange(len(tokens))
    
    # Use a vibrant colormap for the bars
    colors = plt.cm.magma(last_token_attn / max(last_token_attn))
    
    bars = ax.barh(y_pos, last_token_attn, color=colors, edgecolor='black', height=0.8)
    
    # Formatting the aesthetics to look like an advanced diagnostic UI
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{i}: '{t}'" for i, t in enumerate(tokens)], fontsize=14, fontfamily='monospace')
    ax.invert_yaxis() # Read top-to-bottom
    
    # Hide the x-axis numbers, keep the visual clean
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_color('#444444')
    
    plt.title(f"In-Context Learning (Induction Circuit)\nWhat is the model looking at to predict the next word?", 
              fontsize=18, pad=20, color='white', fontweight='bold')
    plt.ylabel("Prompt Tokens", fontsize=14, color='#AAAAAA', labelpad=15)
    
    # Add a visual "Target" line showing where we expect the model to look
    target_idx = [i for i, t in enumerate(tokens) if "Ba" in t and i < 10][1] # Find the first BaCO3 token
    ax.axhline(y=target_idx, color='#00FFCC', linestyle='--', alpha=0.5, zorder=0)
    ax.text(max(last_token_attn)*0.7, target_idx - 0.5, "Induction Target", color='#00FFCC', fontsize=12, fontweight='bold')

    plt.tight_layout()
    
    output_file = "qwen_induction_circuit_grpo2400.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='#0D0D0D')
    print(f"\nSUCCESS: Brain scan saved as {output_file}")

if __name__ == "__main__":
    generate_induction_scan()