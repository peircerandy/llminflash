import os
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights

MODEL_ID = "facebook/opt-6.7b"
CACHE_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache"
LAYERS_DIR = "/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_layers"

config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=CACHE_PATH, local_files_only=True)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

model.to_empty(device="cpu")

def suffix_load(target_module, state_dict, name_hint=""):
    target_sd = target_module.state_dict()
    clean_sd = {}
    loaded_count = 0
    suffix_map = {}
    for k in target_sd.keys():
        suffix = ".".join(k.split('.')[-2:])
        if suffix not in suffix_map: suffix_map[suffix] = []
        suffix_map[suffix].append(k)
        
    for k_ckpt, v in state_dict.items():
        suffix_ckpt = ".".join(k_ckpt.split('.')[-2:])
        if suffix_ckpt in suffix_map:
            for k_target in suffix_map[suffix_ckpt]:
                if target_sd[k_target].shape == v.shape:
                    clean_sd[k_target] = v.half()
                    loaded_count += 1
                    break
    
    msg = target_module.load_state_dict(clean_sd, strict=False)
    print(f"-> {name_hint}: Loaded {loaded_count} layers. Missing: {len(msg.missing_keys)}")

globals_sd = torch.load(os.path.join(LAYERS_DIR, "globals.pt"), map_location="cpu")
suffix_load(model, globals_sd, "Globals")

print("Embed tokens mean:", model.model.decoder.embed_tokens.weight.mean().item())
if torch.isnan(model.model.decoder.embed_tokens.weight).any():
    print("WARNING: Embed tokens contain NaNs!")

layer_sd = torch.load(os.path.join(LAYERS_DIR, "layer_0.pt"), map_location="cpu")
layer = model.model.decoder.layers[0]
suffix_load(layer, layer_sd, "Block 0")
print("Self attn k_proj mean:", layer.self_attn.k_proj.weight.mean().item())
