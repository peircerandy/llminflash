import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# --- Configuration ---
MODES = ["quantized", "naive_ssd", "oracle", "predictor", "draft"]
CLASS_NAMES = ["AnnualCrop", "Forest", "HerbaceousVeg", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]

def run_viz():
    print("=== GENERATING ENHANCED PPT-READY VISUAL COMPARISON & CLASSIFICATION (V2) ===")
    
    # 1. Load reference data
    try:
        rgb = np.load("benchmark_results/original_rgb.npy")
        y_true = np.load("benchmark_results/ground_truth.npy")
        with open("benchmark_results/sample_class.txt", "r") as f:
            s_class = f.read()
    except FileNotFoundError as e:
        print(f"Error: Missing benchmark data: {e}")
        return

    plt.switch_backend('Agg')
    plt.rcParams.update({'font.size': 20}) # Ultra-large for PPT
    
    # --- Part 1: Heatmap Visualization (The "Industrial" Reference) ---
    fig, axes = plt.subplots(2, len(MODES) + 1, figsize=(42, 18), facecolor='white')
    
    # Input Column
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"INPUT: {s_class.upper()}\n(Original RGB)", fontsize=28, fontweight='bold', color='darkblue', pad=25)
    axes[0, 0].axis('off')
    
    # Row Labels
    fig.text(0.015, 0.75, "RAW 28x28 GRID\n(Model Output)", va='center', ha='center', fontsize=26, fontweight='bold', color='#2c3e50', rotation=90)
    fig.text(0.015, 0.35, "SPATIAL ALIGNMENT\n(Overlay Proof)", va='center', ha='center', fontsize=26, fontweight='bold', color='#2c3e50', rotation=90)
    axes[1, 0].axis('off')

    results = []
    for i, mode in enumerate(MODES):
        h_path = f"benchmark_results/{mode}_heatmap.npy"
        l_path = f"benchmark_results/{mode}_latency.txt"
        p_path = f"benchmark_results/{mode}_predictions.npy"
        c_path = f"benchmark_results/{mode}_confidences.npy"
        
        if os.path.exists(h_path):
            h_map = np.load(h_path)
            lat = 0.0
            if os.path.exists(l_path):
                with open(l_path, "r") as f: lat = float(f.read())
            
            mode_labels = {"quantized": "4-bit RAM\n(Baseline)", "naive_ssd": "Naive SSD\n(Dense)", "oracle": "Oracle\n(Perfect)", "predictor": "Flash Predictor\n(Ours)", "draft": "Draft Mode\n(Skips)"}
            
            # --- ROW 1: Raw Heatmap ---
            h_vis = (h_map - h_map.min()) / (h_map.max() - h_map.min() + 1e-8)
            axes[0, i+1].imshow(h_vis, cmap='magma', interpolation='nearest') 
            axes[0, i+1].set_title(f"{mode_labels[mode]}\n{lat:.2f}s", fontsize=26, fontweight='bold', pad=20)
            axes[0, i+1].axis('off')
            
            # --- ROW 2: Alpha Overlay ---
            scale = rgb.shape[0] / h_map.shape[0]
            h_zoomed = zoom(h_map, scale, order=1)
            h_zoomed = (h_zoomed - h_zoomed.min()) / (h_zoomed.max() - h_zoomed.min() + 1e-8)
            axes[1, i+1].imshow(rgb)
            # Higher transparency (alpha=0.4) for better "passthrough"
            axes[1, i+1].imshow(h_zoomed, cmap='magma', alpha=0.45, interpolation='bilinear') 
            axes[1, i+1].axis('off')

            # --- Stats ---
            acc, conf = 0, 0
            if os.path.exists(p_path) and os.path.exists(c_path):
                y_p = np.load(p_path)
                y_c = np.load(c_path)
                if len(y_p) == len(y_true):
                    acc = accuracy_score(y_true, y_p)
                    conf = np.mean(y_c)
                else:
                    print(f"Warning: Sample mismatch for {mode} ({len(y_p)} vs {len(y_true)})")
            
            results.append({"Mode": mode, "Avg Latency (s)": lat, "Accuracy": acc, "Confidence": conf})
        else:
            axes[0, i+1].text(0.5, 0.5, "PENDING", ha='center', va='center', fontsize=20, color='gray')
            axes[0, i+1].axis('off')
            axes[1, i+1].axis('off')

    # Re-add Color Key (Enhanced)
    fig.text(0.40, 0.04, "COLOR KEY:", fontsize=24, fontweight='bold')
    fig.text(0.50, 0.04, " [ LOW FEATURE INTENSITY ] ", fontsize=22, color='white', fontweight='bold', bbox=dict(facecolor='#3b0042', pad=8))
    fig.text(0.68, 0.04, " [ HIGH FEATURE INTENSITY ] ", fontsize=22, color='black', fontweight='bold', bbox=dict(facecolor='#fde725', pad=8))

    plt.tight_layout(rect=[0.05, 0.08, 1, 0.95])
    plt.savefig("benchmark_visual_comparison.png", dpi=200, facecolor='whitesmoke')

    # --- Part 2: Confusion Matrices (Semantic Labels) ---
    fig_cm, axes_cm = plt.subplots(1, len(MODES), figsize=(48, 10))
    for i, mode in enumerate(MODES):
        p_path = f"benchmark_results/{mode}_predictions.npy"
        if os.path.exists(p_path):
            y_p = np.load(p_path)
            if len(y_p) == len(y_true):
                cm = confusion_matrix(y_true, y_p, labels=range(10))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
                disp.plot(ax=axes_cm[i], cmap='Blues', colorbar=False, xticks_rotation=45)
                axes_cm[i].set_title(f"{mode.upper()}\nAccuracy: {accuracy_score(y_true, y_p):.2f}", fontsize=24, fontweight='bold', pad=20)
            else:
                axes_cm[i].text(0.5, 0.5, f"WAITING FOR\nDATA\n({len(y_p)}/50)", ha='center', va='center', fontsize=20)
        else:
            axes_cm[i].text(0.5, 0.5, "NOT FOUND", ha='center', va='center', fontsize=20)
    
    plt.tight_layout()
    plt.savefig("benchmark_confusion_matrices.png", dpi=150)

    # --- Part 3: Accuracy/Confidence Summary ---
    df = pd.DataFrame(results)
    df.to_csv("clay_benchmark_results.csv", index=False)
    
    fig_acc, ax_acc = plt.subplots(figsize=(16, 9))
    x_pos = np.arange(len(df))
    width = 0.35
    
    bar1 = ax_acc.bar(x_pos - width/2, df['Accuracy'] * 100, width, label='Accuracy (%)', color='#3498db', alpha=0.8)
    bar2 = ax_acc.bar(x_pos + width/2, df['Confidence'] * 100, width, label='Model Confidence (%)', color='#e67e22', alpha=0.6)
    
    # Add values on top
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax_acc.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=16)
    autolabel(bar1); autolabel(bar2)

    ax_acc.set_xticks(x_pos)
    ax_acc.set_xticklabels([m.upper() for m in df['Mode']], fontsize=18, fontweight='bold')
    ax_acc.set_ylabel("Percentage (%)", fontsize=20, fontweight='bold')
    ax_acc.set_ylim(0, 105)
    ax_acc.legend(fontsize=18)
    ax_acc.set_title("Semantic Preservation: Accuracy vs. Confidence", fontsize=26, fontweight='bold', pad=30)
    ax_acc.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("classification_performance.png", dpi=150)

    print("\nVisual Analysis complete! Generated:")
    print("- benchmark_visual_comparison.png")
    print("- benchmark_confusion_matrices.png")
    print("- classification_performance.png")

if __name__ == "__main__":
    run_viz()
