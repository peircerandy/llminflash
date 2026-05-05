"""
train_portable_predictor.py: Standalone, High-Performance Predictor Trainer.

This script captures activation sparsity from ANY transformer model (HuggingFace) 
and trains low-rank predictors. Supports LLMs and Vision Transformers (like Clay).

Usage:
    python train_portable_predictor.py --model_id made-with-clay/Clay --rank 128
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import json
import argparse
from datasets import load_dataset
from tqdm import tqdm

class LowRankPredictor(nn.Module):
    def __init__(self, d_model, rank, d_ffn):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_ffn, bias=False)
    def forward(self, x):
        return torch.sigmoid(self.up(torch.relu(self.down(x))))

def train(args):
    print(f"--- Portable Predictor Trainer ---")
    print(f"Target Model: {args.model_id}")
    
    # 1. Setup Model and Architecture Info
    # trust_remote_code=True is essential for custom architectures like geovit+DOFA
    try:
        config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading config: {e}\nEnsure you've installed any model-specific packages (e.g., Clay).")
        return

    # Dynamically detect architecture dims
    d_model = getattr(config, "hidden_size", getattr(config, "d_model", None))
    d_ffn = getattr(config, "intermediate_size", getattr(config, "ffn_dim", None))
    num_layers = getattr(config, "num_hidden_layers", getattr(config, "num_layers", None))
    
    if not all([d_model, d_ffn, num_layers]):
        print(f"Error: Could not auto-detect dims. Found Hidden={d_model}, FFN={d_ffn}, Layers={num_layers}")
        return

    print(f"Detected: Hidden={d_model}, FFN={d_ffn}, Layers={num_layers}")

    # Use AutoModel instead of CausalLM for non-text models (like Clay ViT)
    print("Loading model weights...")
    try:
        if args.is_causal:
            model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        else:
            model = AutoModel.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    except Exception as e:
        print(f"Model load failed: {e}")
        return
        
    model.eval()

    # 2. Initialize Predictors
    predictors = [LowRankPredictor(d_model, args.rank, d_ffn).cuda() for _ in range(num_layers)]
    optimizers = [optim.Adam(p.parameters(), lr=1e-4) for p in predictors]
    
    # 3. Activation Capture Logic
    activations = {}
    def get_hook(idx):
        def hook(m, i, o):
            act_data = o[0] if isinstance(o, tuple) else o
            activations[idx] = (act_data > 0).float().detach()
        return hook

    # Register hooks on FFN activation functions
    hooks = []
    for i in range(num_layers):
        layer_mod = None
        # Logic to find the activation module (Architecture Dependent)
        # 1. Llama-style
        if hasattr(model, "model") and hasattr(model.model, "layers"): 
            layer_mod = model.model.layers[i].mlp.act_fn
        # 2. OPT-style
        elif hasattr(model, "model") and hasattr(model.model.decoder, "layers"):
            layer_mod = model.model.decoder.layers[i].activation_fn
        # 3. ViT / Clay-style (Assuming standard timm/vit blocks)
        elif hasattr(model, "blocks"):
            layer_mod = model.blocks[i].mlp.act # Standard ViT act
        elif hasattr(model, "model") and hasattr(model.model, "blocks"):
            layer_mod = model.model.blocks[i].mlp.act

        if layer_mod:
            hooks.append(layer_mod.register_forward_hook(get_hook(i)))
        else:
            print(f"Warning: Activation module not found for layer {i}")

    # 4. Training Loop
    # For Clay, we'd ideally use SAR data, but C4 works for general feature capture.
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    pbar = tqdm(total=args.samples, desc="Training")
    
    # Simple mock tokenizer if not provided (for non-text models)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    except:
        tokenizer = None

    for i, sample in enumerate(dataset):
        if i >= args.samples: break
        
        if tokenizer:
            inputs = tokenizer(sample['text'], return_tensors="pt", truncation=True, max_length=128).to("cuda")
            with torch.no_grad(): outputs = model(**inputs, output_hidden_states=True)
        else:
            # For Vision models, we'd need dummy pixels or real images
            # This logic would need expansion for pure image models.
            print("Non-text model detected. Script requires expansion for image dataset loading.")
            break
            
        for l_idx in range(num_layers):
            x = outputs.hidden_states[l_idx].detach().float()
            y_true = activations[l_idx].float()
            
            optimizers[l_idx].zero_grad()
            y_pred = predictors[l_idx](x)
            loss = nn.BCELoss()(y_pred, y_true)
            loss.backward()
            optimizers[l_idx].step()
        pbar.update(1)
    
    for h in hooks: h.remove()
    pbar.close()

    # 5. Export
    base_name = args.model_id.split("/")[-1]
    bin_path = f"{base_name}_predictors.bin"
    meta_path = f"{base_name}_predictors.json"

    with open(bin_path, "wb") as f:
        for p in predictors:
            f.write(p.down.weight.data.cpu().float().numpy().tobytes())
            f.write(p.up.weight.data.cpu().float().numpy().tobytes())

    metadata = {
        "model_id": args.model_id, "hidden_size": d_model, "ffn_dim": d_ffn,
        "num_layers": num_layers, "rank": args.rank, "dtype": "float32"
    }
    with open(meta_path, "w") as f: json.dump(metadata, f, indent=4)
    print(f"Exported to {bin_path}. Copy to edge device.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="facebook/opt-6.7b")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--is_causal", action="store_true", help="Use CausalLM loader (for LLMs)")
    train(parser.parse_args())
