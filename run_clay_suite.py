
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
import torch

# --- Configuration ---
MODES = ["quantized", "naive_ssd", "oracle", "predictor", "draft"]
PYTHON_PATH = "/home/randy/miniconda3/envs/llm-flash-v2/bin/python"

def run_viz():
    print("=== GENERATING MULTI-VIEW VISUAL COMPARISON ===")
    
    # 1. Load the original reference data
    try:
        rgb = np.load("benchmark_results/original_rgb.npy")
        with open("benchmark_results/sample_class.txt", "r") as f:
            s_class = f.read()
    except FileNotFoundError:
        print("Error: Missing benchmark data. Run 'bash flash_clay.sh' first.")
        return

    plt.switch_backend('Agg')
    # 2 Rows: Heatmap vs Overlay
    fig, axes = plt.subplots(2, len(MODES) + 1, figsize=(30, 12))
    
    # Column 0: Original Image
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"INPUT: {s_class}\n(Original RGB)", fontsize=16, fontweight='bold', color='navy')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off') # Spacer column

    results = []
    for i, mode in enumerate(MODES):
        h_path = f"benchmark_results/{mode}_heatmap.npy"
        l_path = f"benchmark_results/{mode}_latency.txt"
        
        if os.path.exists(h_path) and os.path.exists(l_path):
            h_map = np.load(h_path)
            with open(l_path, "r") as f:
                lat = float(f.read())
            
            # --- ROW 1: Raw Heatmap ---
            im = axes[0, i+1].imshow(h_map, cmap='magma')
            mode_labels = {
                "standard": "Standard (Full RAM)",
                "quantized": "4-bit RAM (Baseline)", 
                "naive_ssd": "Naive SSD (No Prediction)",
                "oracle": "Oracle (Perfect Sparsity)", 
                "draft": "Draft Mode (Layer Skip)", 
                "predictor": "Flash Predictor (Real Sparsity)"
            }
            axes[0, i+1].set_title(f"{mode_labels[mode]}\nLatency: {lat:.2f}s", fontsize=14, fontweight='bold')
            axes[0, i+1].axis('off')
            
            # --- ROW 2: Alpha Overlay (Direct Visual Proof) ---
            from scipy.ndimage import zoom
            scale = rgb.shape[0] / h_map.shape[0]
            h_zoomed = zoom(h_map, scale, order=1)
            h_zoomed = (h_zoomed - h_zoomed.min()) / (h_zoomed.max() - h_zoomed.min() + 1e-8)
            
            axes[1, i+1].imshow(rgb)
            axes[1, i+1].imshow(h_zoomed, cmap='magma', alpha=0.6) 
            axes[1, i+1].set_title(f"{mode.upper()} Feature Alignment", fontsize=11, style='italic')
            axes[1, i+1].axis('off')
            
            results.append({"Mode": mode, "Avg Latency (s)": lat})

    # Add Explanatory Legend / Color Key
    plt.subplots_adjust(bottom=0.22)
    msg = (
        "COLOR KEY: [Bright/Yellow] = High Feature Intensity (Identified buildings, roads, vegetation) | "
        "[Dark/Purple] = Low Intensity (Uniform areas like water/flat soil)\n"
        "VISUAL PROOF: Row 2 shows the Heatmap OVERLAID on the original image. Look for the yellow 'glow' aligning with physical structures."
    )
    fig.text(0.5, 0.05, msg, ha="center", fontsize=16, color='white', 
             fontweight='bold', bbox=dict(facecolor='black', alpha=0.85, pad=12))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig("benchmark_visual_comparison.png", dpi=160, facecolor='whitesmoke')
    print("\nVisual Analysis complete! Generated 'benchmark_visual_comparison.png'.")
    
    df = pd.DataFrame(results)
    df.to_csv("clay_benchmark_results.csv", index=False)
    print("\nBenchmark Summary Table:")
    print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--viz_only", action="store_true")
    run_viz()
