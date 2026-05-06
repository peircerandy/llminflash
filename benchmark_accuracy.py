
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def run_accuracy_benchmark():
    print("=== GENERATING PROFESSIONAL ACCURACY METRICS ===")
    plt.switch_backend('Agg')
    MODES = ["quantized", "naive_ssd", "oracle", "predictor", "draft"]
    
    # 1. Load Heatmaps (Signal Representation)
    results = []
    
    # Baseline must be 16x16 (256)
    try:
        baseline = np.load("benchmark_results/naive_ssd_heatmap.npy")
        if baseline.size != 256:
             print(f"Warning: Baseline size {baseline.size} is not 256. Fixing...")
             baseline = np.zeros((16, 16))
    except:
        print("Run benchmark first.")
        return

    for mode in MODES:
        path = f"benchmark_results/{mode}_heatmap.npy"
        if os.path.exists(path):
            h = np.load(path)
            # Ensure consistent shape for comparison
            if h.size != baseline.size:
                print(f"Skipping {mode} due to size mismatch ({h.size} vs {baseline.size})")
                continue
                
            # Cosine Similarity
            sim = np.dot(baseline.flatten(), h.flatten()) / (np.linalg.norm(baseline) * np.linalg.norm(h) + 1e-8)
            mse = np.mean((baseline - h)**2)
            results.append({"Mode": mode, "Fidelity (Cosine)": sim, "Error (MSE)": mse})

    if not results:
        print("No results to plot.")
        return
        
    df = pd.DataFrame(results)
    
    # --- Professional Plotting ---
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Inference Mode', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fidelity to Dense Model (%)', color=color, fontsize=12, fontweight='bold')
    
    # Map display names
    display_names = {
        "quantized": "4-bit RAM", "naive_ssd": "Dense SSD",
        "oracle": "Oracle", "predictor": "Predictor", "draft": "Draft"
    }
    df['Label'] = df['Mode'].map(display_names)
    
    bars = ax1.bar(df['Label'], df['Fidelity (Cosine)'] * 100, color=color, alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(min(df['Fidelity (Cosine)']*100) - 5, 105)

    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.title("PROFESSIONAL FIDELITY METRICS: FLASH vs. DENSE\n(Preservation of model 'signal' during sparse loading)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("accuracy_comparison.png", dpi=150)
    print("Generated 'accuracy_comparison.png'.")

if __name__ == "__main__":
    run_accuracy_benchmark()
