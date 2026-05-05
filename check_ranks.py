import torch
import os

path = "LLM_Project/opt_6.7b/predictor_weights"
layers = sorted([f for f in os.listdir(path) if f.endswith(".pt")], key=lambda x: int(x.split("_")[1].split(".")[0]))

for f in layers:
    try:
        sd = torch.load(os.path.join(path, f), map_location="cpu", weights_only=True)
        print(f"{f}: {sd['down.weight'].shape}")
    except Exception as e:
        print(f"{f}: Error {e}")
