
import torch
import torch.nn as nn
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import shutil
import gc
import matplotlib.pyplot as plt

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
        if x.dim() == 1: x = x.unsqueeze(0)
        return torch.sigmoid(self.up(torch.relu(self.down(x))))

def detect_rank(path, hs, fd, nl):
    file_size = os.path.getsize(path)
    rank = int(file_size / (nl * (hs + fd) * 4))
    return rank

def load_predictor(path, hs, fd, nl):
    rank = detect_rank(path, hs, fd, nl)
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
                if acts.dim() == 3: acts = acts[0]
                captured_acts[idx] = (acts[-1, :] > 0).float().detach().cpu()
            return hook
        
        hooks = [model.model.decoder.layers[i].activation_fn.register_forward_hook(get_hook(i)) for i in range(cfg["nl"])]

        text = "Artificial intelligence is a branch of computer science."
        inputs = tokenizer(text, return_tensors="pt")
        
        start = time.time()
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        dense_time = time.time() - start

        # Benchmark predictor speed (Simulated pass of small model)
        x = outputs.hidden_states[0][0, -1, :].float().cpu()
        
        start_old = time.time()
        for i in range(cfg["nl"]): _ = old_p[i](x)
        old_pred_time = time.time() - start_old
        
        start_new = time.time()
        for i in range(cfg["nl"]): _ = new_p[i](x)
        new_pred_time = time.time() - start_new

        results = []
        for i in range(cfg["nl"]):
            x_layer = outputs.hidden_states[i][0, -1, :].float().cpu()
            y_true = captured_acts[i]
            y_old = (old_p[i](x_layer).squeeze() > 0.5).float()
            y_new = (new_p[i](x_layer).squeeze() > 0.5).float()
            
            recall_old = (y_old * y_true).sum() / (y_true.sum() + 1e-6)
            recall_new = (y_new * y_true).sum() / (y_true.sum() + 1e-6)
            results.append({"layer": i, "old": recall_old.item(), "new": recall_new.item()})
        
        for h in hooks: h.remove()
        df = pd.DataFrame(results)
        
        # --- Graphing ---
        plt.switch_backend('Agg')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Recall (Accuracy)
        ax1.bar(["Old (Local)", "New (GCP)"], [df['old'].mean()*100, df['new'].mean()*100], color=['blue', 'green'], alpha=0.7)
        ax1.set_ylabel("Mean Recall (%)", fontweight='bold')
        ax1.set_title("Predictor Accuracy: GCP vs Local", fontweight='bold')
        ax1.set_ylim(40, 100)
        for i, v in enumerate([df['old'].mean()*100, df['new'].mean()*100]):
            ax1.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')

        # Speed (Efficiency)
        ax2.bar(["Old (Rank 240)", "New (Rank 128)"], [old_pred_time*1000, new_pred_time*1000], color=['darkred', 'orange'], alpha=0.7)
        ax2.set_ylabel("Inference Latency (ms)", fontweight='bold')
        ax2.set_title("Predictor Speed: Rank Comparison", fontweight='bold')
        for i, v in enumerate([old_pred_time*1000, new_pred_time*1000]):
            ax2.text(i, v + 0.1, f"{v:.2f}ms", ha='center', fontweight='bold')

        plt.suptitle(f"REAL DATA: Predictor Comparison for {model_key.upper()}\n(Validated on live sequence)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("predictor_gcp_vs_local.png", dpi=150)
        print("Generated 'predictor_gcp_vs_local.png'.")

    except Exception as e:
        print(f"Comparison failed: {e}")
    finally:
        if 'model' in locals(): del model
        gc.collect()
        if os.path.exists(offload): shutil.rmtree(offload)

if __name__ == "__main__":
    import time
    compare("opt")
