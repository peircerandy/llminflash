import torch
from huggingface_hub import snapshot_download
import os

# Clay v1.5
print("Downloading Clay v1.5...")
snapshot_download(repo_id="made-with-clay/Clay", allow_patterns=["v1.5/clay-v1.5.ckpt"], cache_dir="/mnt/wsl/PHYSICALDRIVE0p3/hf_cache")

# Llama 3 8B (Attempt - will fail if not authorized, but let's check if user has access)
try:
    print("Downloading Llama 3 8B...")
    snapshot_download(repo_id="meta-llama/Meta-Llama-3-8B", cache_dir="/mnt/wsl/PHYSICALDRIVE0p3/hf_cache")
except Exception as e:
    print(f"Llama 3 download failed: {e}")
