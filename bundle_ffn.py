/**
 * bundle_ffn.py: Weight preparation tool for SSD streaming.
 * 
 * This script implements "Row-Column Bundling," the core technique from 
 * Section 3.2 of the paper. It ensures that the input weights (fc1) 
 * and output weights (fc2) for each neuron are stored contiguously on the SSD.
 * 
 * Innovation: The "Identity Matrix Hack"
 * To bypass Accelerate's 'meta-tensor' issues, we pass an Identity matrix 
 * through the FFN. This forces PyTorch to materialize the weights and 
 * perform the multiplication math, which we then capture and save to disk.
 */

import torch
from transformers import OPTForCausalLM
import os
from tqdm import tqdm

# Constants tailored to OPT-6.7B
MODEL_ID = "facebook/opt-6.7b"
CACHE_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache"
OUT_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_bundled_ffn.bin"

def bundle():
    print(f"Loading {MODEL_ID} for weight materialization...")
    
    # Load model with disk offloading enabled
    model = OPTForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        cache_dir=CACHE_PATH
    )
    model.eval()

    print(f"Bundling FFN weights to {OUT_PATH}...")
    with open(OUT_PATH, "wb") as f:
        # Iterate through all 32 decoder layers
        for i in tqdm(range(len(model.model.decoder.layers)), desc="Layers"):
            layer = model.model.decoder.layers[i]
            
            with torch.no_grad():
                # --- Step 1: Materialize FC1 (Up-Projection) ---
                # We pass an identity matrix (4096 x 4096) through FC1.
                # The output is literally the weight matrix itself, extracted
                # via a forward pass to ensure all Accelerate hooks are triggered.
                eye_1 = torch.eye(4096, dtype=torch.float16).cuda()
                # Subtract bias to get clean weights
                zero_1 = torch.zeros(1, 4096, dtype=torch.float16).cuda()
                bias_1 = layer.fc1(zero_1)
                fc1_w = (layer.fc1(eye_1) - bias_1).T.cpu() # Transpose to get [neuron, hidden]
                
                # --- Step 2: Materialize FC2 (Down-Projection) ---
                # We do the same for the second layer (16384 x 16384 identity)
                # This ensures we get exactly what the math expects.
                eye_2 = torch.eye(16384, dtype=torch.float16).cuda()
                zero_2 = torch.zeros(1, 16384, dtype=torch.float16).cuda()
                bias_2 = layer.fc2(zero_2)
                fc2_w = (layer.fc2(eye_2) - bias_2).cpu() # Already in [hidden, neuron]
                
                # --- Step 3: Bundle and Save ---
                # For each of the 16,384 neurons, save [fc1_row][fc2_col] contiguously.
                # This enables the C++ engine to perform a single 16KB read from NVMe
                # to get all weights for an activated neuron.
                for n in range(16384):
                    row_bytes = fc1_w[n, :].numpy().tobytes()
                    col_bytes = fc2_w[:, n].numpy().tobytes()
                    f.write(row_bytes)
                    f.write(col_bytes)
                
                # Aggressive memory cleanup to prevent VRAM overflow
                del eye_1, zero_1, bias_1, fc1_w
                del eye_2, zero_2, bias_2, fc2_w
                torch.cuda.empty_cache()
                
    print("\nSSD Matrix Bundling Complete! The C++ Engine can now read perfect math.")

if __name__ == "__main__":
    bundle()
