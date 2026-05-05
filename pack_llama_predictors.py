import torch
import os

# Config
PRED_DIR = "LLM_Project/llama3_8b/predictor_weights"
OUT_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/llama3_predictors.bin"
NUM_LAYERS = 32

def pack_predictors():
    print(f"Packing Llama3 predictors from {PRED_DIR}...")
    with open(OUT_PATH, "wb") as f:
        for i in range(NUM_LAYERS):
            path = os.path.join(PRED_DIR, f"layer_{i}.pt")
            sd = torch.load(path, map_location="cpu")
            
            # Extract weights as float32
            # net.0.weight: [rank, hidden]
            # net.2.weight: [ffn_dim, rank]
            down = sd["net.0.weight"].float().numpy().tobytes()
            up = sd["net.2.weight"].float().numpy().tobytes()
            
            f.write(down)
            f.write(up)
            
    print(f"Done! Llama3 predictors packed to {OUT_PATH}")

if __name__ == "__main__":
    pack_predictors()
