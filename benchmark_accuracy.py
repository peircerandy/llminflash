
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.ndimage import zoom

def run_accuracy_benchmark():
    print("=== GENERATING PROFESSIONAL ACCURACY METRICS (DEBUG) ===")
    plt.switch_backend('Agg')
    MODES = ["quantized", "naive_ssd", "oracle", "predictor", "draft"]
    results = []
    
    # Baseline (Dense)
    try:
        baseline_raw = np.load("benchmark_results/naive_ssd_heatmap.npy")
        print(f"DEBUG: Baseline shape: {baseline_raw.shape}")
        # Normalize baseline to 16x16 grid for professional comparison
        scale = 16.0 / baseline_raw.shape[0]
        baseline = zoom(baseline_raw, scale, order=1)
    except Exception as e:
        print(f"DEBUG: Error loading baseline: {e}")
        return

    for mode in MODES:
        path = f"benchmark_results/{mode}_heatmap.npy"
        if os.path.exists(path):
            h_raw = np.load(path)
            print(f"DEBUG: Mode {mode} shape: {h_raw.shape}")
            # Zoom to 16x16
            scale = 16.0 / h_raw.shape[0]
            h = zoom(h_raw, scale, order=1)
            
            # Cosine Similarity
            sim = np.dot(baseline.flatten(), h.flatten()) / (np.linalg.norm(baseline) * np.linalg.norm(h) + 1e-8)
            mse = np.mean((baseline - h)**2)
            print(f"DEBUG: Mode {mode} Fidelity: {sim:.4f}")
            results.append({"Mode": mode, "Fidelity (Cosine)": sim, "Error (MSE)": mse})
        else:
            print(f"DEBUG: Mode {mode} file NOT FOUND at {path}")

    if not results:
        print("DEBUG: NO RESULTS FOUND. ABORTING.")
        return

    df = pd.DataFrame(results)
    
    # Map display names
    display_names = {
        "quantized": "4-bit RAM", "naive_ssd": "Dense SSD",
        "oracle": "Oracle", "predictor": "Predictor", "draft": "Draft"
    }
    df['Label'] = df['Mode'].map(display_names)
    
    # --- Professional Plotting ---
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    color = 'tab:blue'
    ax1.set_xlabel('Inference Mode', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fidelity to Dense Model (%)', color=color, fontsize=12, fontweight='bold')
    
    bars = ax1.bar(df['Label'], df['Fidelity (Cosine)'] * 100, color=color, alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Dynamic limits for clarity
    vals = df['Fidelity (Cosine)'] * 100
    ax1.set_ylim(max(0, min(vals) - 10), 105)

    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)

    plt.title("VECTOR FIDELITY: FLASH vs. DENSE\n(Preservation of mathematical 'meaning' in embedding space)", fontsize=16, fontweight='bold')
    
    # Footnote about Draft mode
    fig.text(0.5, 0.02, "*Note: Draft mode skips layers, causing a 'mean shift' in vectors (Low Fidelity) while often preserving spatial alignment (Visual Similarity).", 
             ha="center", fontsize=10, style='italic', color='darkred')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("accuracy_comparison.png", dpi=150)
    print(f"DEBUG: Success! Generated 'accuracy_comparison.png' with {len(df)} bars.")

if __name__ == "__main__":
    run_accuracy_benchmark()
