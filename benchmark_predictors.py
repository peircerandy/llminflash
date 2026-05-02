import torch
import torch.nn as nn
from transformers import OPTForCausalLM, AutoTokenizer
import os
from tqdm import tqdm
import numpy as np

# Configuration
MODEL_ID = "facebook/opt-6.7b"
CACHE_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache"
OFFLOAD_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_offload"

# Predictor Paths
MY_DIR = "predictor_weights" # Assuming your weights are here or similar
PARTNER_DIR = "LLM_Project/opt_6.7b/predictor_weights"

class LowRankPredictor(nn.Module):
    def __init__(self, d_model, rank, d_ffn):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_ffn, bias=False)
    def forward(self, x):
        return self.up(self.down(x))

def benchmark_recall():
    print("🚀 Benchmarking Predictor Quality (Recall vs Sparsity)...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_PATH)
    model = OPTForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", 
        cache_dir=CACHE_PATH, offload_folder=OFFLOAD_PATH
    )
    model.eval()

    prompt = "Large Language Models are revolutionizing the way we interact with technology by"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Dictionary to store recall scores
    results = {"Partner": [], "Yours": []}
    
    # We'll test Layer 0 (Normal) and Layer 30 (Sensitive)
    layers_to_test = [0, 30]

    for layer_idx in layers_to_test:
        print(f"\n--- Testing Layer {layer_idx} ---")
        
        # 1. Get Ground Truth activations
        with torch.no_grad():
            layer = model.model.decoder.layers[layer_idx]
            hidden_states = []
            def hook(m, i, o): hidden_states.append(i[0])
            handle = layer.fc1.register_forward_hook(hook)
            outputs = model(**inputs)
            handle.remove()
            
            x = hidden_states[0].float()
            # Actual activations from the real model
            true_act = (layer.fc1(hidden_states[0]) > 0).float()
            num_true_active = true_act.sum().item()
            print(f"  Actual Active Neurons: {num_true_active} ({num_true_active/16384*100:.1f}% sparsity)")

        # 2. Test Partner's Predictor
        p_path = f"{PARTNER_DIR}/layer_{layer_idx}.pt"
        if os.path.exists(p_path):
            rank = 128 if layer_idx < 28 else 1024
            p_model = LowRankPredictor(4096, rank, 16384).cpu()
            p_model.load_state_dict(torch.load(p_path, map_location="cpu"))
            p_model.eval()
            
            with torch.no_grad():
                logits = p_model(x.cpu())
                # Top 1024 (as per our engine config)
                top_k_indices = torch.topk(logits, 1024, dim=-1).indices
                pred_mask = torch.zeros_like(true_act.cpu())
                pred_mask.scatter_(-1, top_k_indices, 1.0)
                
                # Recall: What % of 'true active' did we catch?
                recall = (pred_mask * true_act.cpu()).sum() / (num_true_active + 1e-8)
                results["Partner"].append(recall.item())
                print(f"  [Partner] Recall@1024: {recall.item()*100:.2f}%")

        # 3. Test Your Predictor
        # (Assuming your weights follow a similar naming pattern in MY_DIR)
        # Note: If your weights are in a different format, I'll need to adjust this.
        my_path = f"layer_{layer_idx}.pt" # Placeholder
        if os.path.exists(my_path):
            my_model = LowRankPredictor(4096, 128, 16384).cpu()
            my_model.load_state_dict(torch.load(my_path, map_location="cpu"))
            my_model.eval()
            
            with torch.no_grad():
                logits = my_model(x.cpu())
                top_k_indices = torch.topk(logits, 1024, dim=-1).indices
                pred_mask = torch.zeros_like(true_act.cpu())
                pred_mask.scatter_(-1, top_k_indices, 1.0)
                
                recall = (pred_mask * true_act.cpu()).sum() / (num_true_active + 1e-8)
                results["Yours"].append(recall.item())
                print(f"  [Yours]   Recall@1024: {recall.item()*100:.2f}%")

    print("\n" + "="*40)
    print(" FINAL PREDICTOR VERDICT ")
    print("="*40)
    for k, v in results.items():
        if v:
            print(f"{k:10} | Avg Recall: {np.mean(v)*100:.1f}%")
    print("="*40)

if __name__ == "__main__":
    benchmark_recall()
