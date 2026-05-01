import torch
from transformers import OPTForCausalLM
import os
from tqdm import tqdm

MODEL_ID = "facebook/opt-6.7b"
CACHE_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache"
OUT_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_bundled_ffn.bin"
OFFLOAD_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_offload"

def bundle():
    print("Loading OPT-6.7B Shell...")
    model = OPTForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto",
        cache_dir=CACHE_PATH,
        offload_folder=OFFLOAD_PATH,
        local_files_only=True
    )

    print(f"Bundling FFN matrices into {OUT_PATH}...")
    with open(OUT_PATH, "wb") as f:
        with torch.no_grad():
            for i in tqdm(range(32), desc="Layers"):
                layer = model.model.decoder.layers[i]
                
                # THE IDENTITY MATRIX HACK:
                # Accelerate hides the weights in ghost meta-tensors on disk.
                # Passing an Identity matrix mathematically forces Accelerate to 
                # stream the weights to the GPU, calculate them, and hand them to us!
                
                # Extract FC1: natively (16384, 4096)
                zero_1 = torch.zeros(1, 4096, dtype=model.dtype, device=model.device)
                bias_1 = layer.fc1(zero_1)
                
                eye_1 = torch.eye(4096, dtype=model.dtype, device=model.device)
                fc1_w_t = layer.fc1(eye_1) - bias_1  # Output shape: (4096, 16384)
                fc1_w = fc1_w_t.t().cpu().contiguous() # Transpose back to: (16384, 4096)
                
                # Extract FC2: natively (4096, 16384) -> We WANT it transposed to (16384, 4096)!
                zero_2 = torch.zeros(1, 16384, dtype=model.dtype, device=model.device)
                bias_2 = layer.fc2(zero_2)
                
                eye_2 = torch.eye(16384, dtype=model.dtype, device=model.device)
                # fc2(eye) natively returns shape (16384, 4096). This perfectly transposes it for C++!
                fc2_w = (layer.fc2(eye_2) - bias_2).cpu().contiguous() 
                
                # Concatenate them side-by-side to create the 8192-element Neuron Bundle
                bundled = torch.cat([fc1_w, fc2_w], dim=1)
                
                # Write all 16,384 bundles for this layer instantly to the SSD
                f.write(bundled.numpy().tobytes())
                
                # Clean up GPU memory so we don't hit an OOM crash
                del zero_1, bias_1, eye_1, fc1_w_t, fc1_w
                del zero_2, bias_2, eye_2, fc2_w, bundled
                torch.cuda.empty_cache()
                
    print("\nSSD Matrix Bundling Complete! The C++ Engine can now read perfect math.")

if __name__ == "__main__":
    bundle()