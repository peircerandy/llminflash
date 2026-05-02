import torch
import os

SAVE_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_predictors.bin"
# FIX: Use OPT weights, not Llama weights!
PREDICTOR_DIR = "LLM_Project/opt_6.7b/predictor_weights"

def convert():
    print(f"Converting OPT-6.7B expert predictors...")
    with open(SAVE_PATH, "wb") as f:
        for i in range(32):
            path = f"{PREDICTOR_DIR}/layer_{i}.pt"
            sd = torch.load(path, map_location="cpu")
            # All partner OPT predictors use rank 128 or 1024
            # We already handled the variable rank in C++
            down = sd["down.weight"].float().numpy().tobytes()
            up = sd["up.weight"].float().numpy().tobytes()
            f.write(down)
            f.write(up)
    print(f"Conversion complete!")

if __name__ == "__main__":
    convert()
