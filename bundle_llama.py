import torch
from transformers import AutoModelForCausalLM, AutoConfig
import os
from tqdm import tqdm

# Constants
MODEL_ID = "unsloth/llama-3-8b"
CACHE_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache"
OUT_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/llama3_bundled_ffn.bin"

def bundle():
    print(f"Loading {MODEL_ID} for weight bundling...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        cache_dir=CACHE_PATH
    )
    model.eval()
    config = model.config
    hidden_size = config.hidden_size
    ffn_dim = config.intermediate_size
    num_layers = config.num_hidden_layers

    print(f"Detected Arch: Hidden={hidden_size}, FFN={ffn_dim}, Layers={num_layers}")
    print(f"Bundling Llama3 weights to {OUT_PATH} (using Identity Hack)...")
    
    with open(OUT_PATH, "wb") as f:
        for i in tqdm(range(num_layers), desc="Layers"):
            layer = model.model.layers[i]
            
            with torch.no_grad():
                # Step 1: gate_proj and up_proj
                eye_h = torch.eye(hidden_size, dtype=torch.float16).cuda()
                gate_w = layer.mlp.gate_proj(eye_h).T.cpu() # [ffn_dim, hidden_size]
                up_w = layer.mlp.up_proj(eye_h).T.cpu()     # [ffn_dim, hidden_size]
                del eye_h
                
                # Step 2: down_proj
                eye_f = torch.eye(ffn_dim, dtype=torch.float16).cuda()
                down_w = layer.mlp.down_proj(eye_f).cpu()   # [ffn_dim, hidden_size]
                del eye_f
                
                # Step 3: Interleave for each neuron: [gate_row][up_row][down_row]
                # Each row is hidden_size (4096) floats.
                for n in range(ffn_dim):
                    f.write(gate_w[n, :].numpy().tobytes())
                    f.write(up_w[n, :].numpy().tobytes())
                    f.write(down_w[n, :].numpy().tobytes())
                
                del gate_w, up_w, down_w
                torch.cuda.empty_cache()
                
    print("\nLlama3 Bundling Complete!")

if __name__ == "__main__":
    bundle()
