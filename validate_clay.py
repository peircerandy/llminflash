import torch
import os
import glob

# Find the checkpoint
ckpt_path = glob.glob("/mnt/wsl/PHYSICALDRIVE0p3/hf_cache/models--made-with-clay--Clay/snapshots/*/v1.5/clay-v1.5.ckpt")[0]
print(f"Loading Clay v1.5 from {ckpt_path}")

sd = torch.load(ckpt_path, map_location="cpu")
print("Successfully loaded Clay checkpoint!")
print(f"Total keys in State Dict: {len(sd.get('state_dict', sd))}")

# Check for Vision Transformer keys
keys = sd.get('state_dict', sd).keys()
mlp_keys = [k for k in keys if "mlp" in k]
print(f"Found {len(mlp_keys)} MLP related keys. Ready for Flash-ViT integration.")
