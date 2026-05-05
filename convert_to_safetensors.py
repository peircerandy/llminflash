import torch
from safetensors.torch import save_file
import os
import glob
import gc
import json

# Configuration
MODEL_ID = "facebook/opt-6.7b"
CACHE_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache"
OUT_FILE = "/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_non_ffn.safetensors"

def convert_surgically():
    print("Starting surgical conversion to Safetensors...")
    
    snap_dir = sorted(glob.glob(os.path.join(CACHE_PATH, "models--facebook--opt-6.7b/snapshots/*")))[-1]
    shards = sorted(glob.glob(os.path.join(snap_dir, "pytorch_model-*.bin")))
    
    non_ffn_state_dict = {}
    
    for shard in shards:
        print(f"Processing {os.path.basename(shard)}...")
        # To avoid OOM, we load the shard and IMMEDIATELY remove massive weights
        sd = torch.load(shard, map_location="cpu", weights_only=True)
        
        keys = list(sd.keys())
        for k in keys:
            # Skip massive FFN weights
            if "fc1.weight" in k or "fc2.weight" in k:
                del sd[k]
                continue
            
            # Keep everything else
            non_ffn_state_dict[k] = sd[k].to(torch.float16).contiguous()
            del sd[k]
            
        del sd
        gc.collect()
        print(f"  Current RAM state dict size: {len(non_ffn_state_dict)} keys")

    print(f"Saving to {OUT_FILE}...")
    save_file(non_ffn_state_dict, OUT_FILE)
    print("Conversion complete!")

if __name__ == "__main__":
    convert_surgically()
