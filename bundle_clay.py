
import torch
import os
from tqdm import tqdm
from claymodel.module import ClayMAEModule

# Constants for Clay v1.5 (mae_large)
CKPT_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache/models--made-with-clay--Clay/snapshots/70200ebcccdf67bf2a0cb9984c77ddee26c10ed2/v1.5/clay-v1.5.ckpt"
OUT_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/clay_bundled_ffn.bin"
HIDDEN_SIZE = 1024
FFN_DIM = 4096

def bundle():
    if not os.path.exists(CKPT_PATH):
        print(f"Error: Checkpoint not found at {CKPT_PATH}")
        return

    print(f"Loading Clay v1.5 state dict (mmap) from {CKPT_PATH}...")
    try:
        # Use mmap to avoid loading 4.9GB into 4.1GB free RAM
        checkpoint = torch.load(CKPT_PATH, map_location="cpu", mmap=True, weights_only=True)
        sd = checkpoint.get("state_dict", checkpoint)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    print(f"Bundling FFN weights to {OUT_PATH}...")
    with open(OUT_PATH, "wb") as f:
        # Clay mae_large has 24 layers
        for i in tqdm(range(24), desc="Layers"):
            # Construct keys based on inspected structure
            # model.encoder.transformer.layers.i.1.net.1.weight -> FC1 [4096, 1024]
            # model.encoder.transformer.layers.i.1.net.3.weight -> FC2 [1024, 4096]
            fc1_key = f"model.encoder.transformer.layers.{i}.1.net.1.weight"
            fc2_key = f"model.encoder.transformer.layers.{i}.1.net.3.weight"
            
            if fc1_key not in sd:
                # Prefix fallback
                fc1_key = f"encoder.transformer.layers.{i}.1.net.1.weight"
                fc2_key = f"encoder.transformer.layers.{i}.1.net.3.weight"

            if fc1_key not in sd:
                print(f"Warning: Could not find keys for layer {i}. Skipping.")
                continue

            # fc1_w shape [4096, 1024], fc2_w shape [1024, 4096]
            # Convert to half-precision (FP16) to match C++ Engine SSD expectations
            fc1_w = sd[fc1_key].half()
            fc2_w = sd[fc2_key].half()
            
            # Row-Column Bundling for 4096 neurons
            for n in range(FFN_DIM):
                row_bytes = fc1_w[n, :].numpy().tobytes()
                col_bytes = fc2_w[:, n].numpy().tobytes()
                f.write(row_bytes)
                f.write(col_bytes)
                
    print(f"\nClay Matrix Bundling Complete! Saved to {OUT_PATH}")

if __name__ == "__main__":
    bundle()
