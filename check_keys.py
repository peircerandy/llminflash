import torch
import os

path = "LLM_Project/opt_6.7b/predictor_weights/layer_0.pt"
sd = torch.load(path, map_location="cpu", weights_only=True)
print(f"Keys in layer_0.pt: {sd.keys()}")
