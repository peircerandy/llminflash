import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights

config = AutoConfig.from_pretrained("./edge_deployment/hf_cache", local_files_only=True)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
    print("Pre-swap params:", sum(p.numel() for p in model.parameters()))
    for i in range(32):
        layer = model.model.decoder.layers[i]
        layer.fc1 = nn.Identity(); layer.fc2 = nn.Identity(); layer.activation_fn = nn.Identity()
    print("Post-swap params:", sum(p.numel() for p in model.parameters()))

print("Materializing...")
model.to_empty(device="cpu")
print("Success!")
