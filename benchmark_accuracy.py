
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.ndimage import zoom

def run_accuracy_benchmark():
    print("=== GENERATING PROFESSIONAL ACCURACY METRICS ===")
    plt.switch_backend('Agg')
    MODES = ["quantized", "naive_ssd", "oracle", "predictor", "draft"]
    results = []
    
    # Baseline must exist
    try:
        baseline_raw = np.load("benchmark_results/naive_ssd_heatmap.npy")
        # Upsample to 16x16 if needed for professional consistency
        if baseline_raw.shape[0] != 16:
            baseline = zoom(baseline_raw, 16/baseline_raw.shape[0], order=1)
        else:
            baseline = baseline_raw
    except:
        print("Run benchmark first.")
        return

    for mode in MODES:
        path = f"benchmark_results/{mode}_heatmap.npy"
        if os.path.exists(path):
            h_raw = np.load(path)
            # Standardize for comparison
            if h_raw.shape[0] != 16:
                h = zoom(h_raw, 16/h_raw.shape[0], order=1)
            else:
                h = h_raw
                
            sim = np.dot(baseline.flatten(), h.flatten()) / (np.linalg.norm(baseline) * np.linalg.norm(h) + 1e-8)
            results.append({"Mode": mode, "Fidelity (Cosine)": sim})

    df = pd.DataFrame(results)
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Inference Mode', fontweight='bold')
    ax1.set_ylabel('Fidelity to Dense Model (%)', color=color, fontweight='bold')
    
    display_names = {
        "quantized": "4-bit RAM", "naive_ssd": "Dense SSD",
        "oracle": "Oracle", "predictor": "Predictor", "draft": "Draft"
    }
    df['Label'] = df['Mode'].map(display_names)
    
    bars = ax1.bar(df['Label'], df['Fidelity (Cosine)'] * 100, color=color, alpha=0.6)
    ax1.set_ylim(max(0, min(df['Fidelity (Cosine)']*100)-10), 105)

    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.title("PROFESSIONAL FIDELITY METRICS: FLASH vs. DENSE", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("accuracy_comparison.png", dpi=150)
    print("Generated 'accuracy_comparison.png'.")

if __name__ == "__main__":
    run_accuracy_benchmark()
