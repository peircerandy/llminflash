import torch
import os
import glob
import json
from safetensors.torch import load_file
import gc

# Configuration
CACHE_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache"
OUT_DIR = "/mnt/wsl/PHYSICALDRIVE0p3/llama3_8b_layers"
os.makedirs(OUT_DIR, exist_ok=True)

def extract_llama_layers():
    print(f"Starting Llama3 Layer extraction to {OUT_DIR}...")
    
    snap_dir = sorted(glob.glob(os.path.join(CACHE_PATH, "models--unsloth--llama-3-8b/snapshots/*")))[-1]
    shards = sorted(glob.glob(os.path.join(snap_dir, "*.safetensors")))
    
    globals_sd = {}
    
    for shard in shards:
        print(f"Processing shard {os.path.basename(shard)}...")
        sd = load_file(shard)
        
        for key in list(sd.keys()):
            # Skip FFN weights (they are streamed)
            if "mlp.gate_proj.weight" in key or "mlp.up_proj.weight" in key or "mlp.down_proj.weight" in key:
                continue
                
            tensor = sd[key]
            
            if "model.layers." in key:
                parts = key.split(".")
                l_idx = parts[2]
                layer_file = os.path.join(OUT_DIR, f"layer_{l_idx}.pt")
                
                if os.path.exists(layer_file):
                    layer_sd = torch.load(layer_file)
                else:
                    layer_sd = {}
                
                layer_sd[key] = tensor
                torch.save(layer_sd, layer_file)
                del layer_sd
            else:
                globals_sd[key] = tensor
        
        del sd; gc.collect()

    print(f"Saving global weights...")
    torch.save(globals_sd, os.path.join(OUT_DIR, "globals.pt"))
    print("Layer extraction complete!")

if __name__ == "__main__":
    extract_llama_layers()
