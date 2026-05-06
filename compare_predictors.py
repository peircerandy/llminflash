
import torch
import torch.nn as nn
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import shutil
import gc

# --- Config ---
DRIVE = "/mnt/wsl/PHYSICALDRIVE0p3/"
MODELS = {
    "opt": {
        "id": "facebook/opt-6.7b",
        "old": DRIVE + "opt_6_7b_predictors.bin",
        "new": DRIVE + "opt_gcp_predictors.bin",
        "hs": 4096, "fd": 16384, "nl": 32
    }
}

class LowRankPredictor(nn.Module):
    def __init__(self, d_model, rank, d_ffn):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_ffn, bias=False)
    def forward(self, x):
        # Handle both 1D and 2D inputs
        if x.dim() == 1: x = x.unsqueeze(0)
        return torch.sigmoid(self.up(torch.relu(self.down(x))))

def detect_rank(path, hs, fd, nl):
    file_size = os.path.getsize(path)
    rank = int(file_size / (nl * (hs + fd) * 4))
    return rank

def load_predictor(path, hs, fd, nl):
    rank = detect_rank(path, hs, fd, nl)
    print(f"Loading {os.path.basename(path)} (Rank {rank})")
    predictors = [LowRankPredictor(hs, rank, fd) for _ in range(nl)]
    with open(path, "rb") as f:
        for p in predictors:
            down_w = np.frombuffer(f.read(hs * rank * 4), dtype=np.float32).reshape(rank, hs)
            up_w = np.frombuffer(f.read(rank * fd * 4), dtype=np.float32).reshape(fd, rank)
            p.down.weight.data = torch.from_numpy(down_w.copy())
            p.up.weight.data = torch.from_numpy(up_w.copy())
    return predictors

def compare(model_key):
    cfg = MODELS[model_key]
    print(f"\n=== Comparing Predictors for {model_key.upper()} ===")
    
    offload = f"offload_{model_key}"
    os.makedirs(offload, exist_ok=True)
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["id"], torch_dtype=torch.float16, low_cpu_mem_usage=True,
            device_map="auto", offload_folder=offload
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg["id"])
        model.eval()
        
        old_p = load_predictor(cfg["old"], cfg["hs"], cfg["fd"], cfg["nl"])
        new_p = load_predictor(cfg["new"], cfg["hs"], cfg["fd"], cfg["nl"])
        
        captured_acts = {}
        def get_hook(idx):
            def hook(m, i, o):
                acts = o[0] if isinstance(o, tuple) else o
                # OPT activation_fn output can be (seq, ffn_dim) or (batch, seq, ffn_dim)
                if acts.dim() == 3: acts = acts[0]
                # Store only the last token
                captured_acts[idx] = (acts[-1, :] > 0).float().detach().cpu()
            return hook
        
        hooks = []
        for i in range(cfg["nl"]):
            layer = model.model.decoder.layers[i].activation_fn
            hooks.append(layer.register_forward_hook(get_hook(i)))

        text = "Artificial intelligence is a branch of computer science."
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        results = []
        for i in range(cfg["nl"]):
            # hidden_states is (batch, seq, hidden_size)
            x = outputs.hidden_states[i][0, -1, :].float().cpu()
            y_true = captured_acts[i]
            
            with torch.no_grad():
                y_old = (old_p[i](x).squeeze() > 0.5).float()
                y_new = (new_p[i](x).squeeze() > 0.5).float()
            
            recall_old = (y_old * y_true).sum() / (y_true.sum() + 1e-6)
            recall_new = (y_new * y_true).sum() / (y_true.sum() + 1e-6)
            results.append({"layer": i, "old": recall_old.item(), "new": recall_new.item()})
        
        for h in hooks: h.remove()
        df = pd.DataFrame(results)
        print(f"Old Mean Recall: {df['old'].mean():.2%}")
        print(f"New GCP Recall: {df['new'].mean():.2%}")
        print(f"Improvement: {df['new'].mean() - df['old'].mean():+.2%}")
        
    except Exception as e:
        print(f"Comparison failed for {model_key}: {e}")
    finally:
        if 'model' in locals(): del model
        gc.collect()
        if os.path.exists(offload): shutil.rmtree(offload)

if __name__ == "__main__":
    compare("opt")
