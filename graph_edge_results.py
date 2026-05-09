import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from scipy.ndimage import zoom

def graph_comparison(results_dir="edge_results"):
    print("=== GENERATING MULTI-MODEL EDGE PERFORMANCE SUITE ===")
    
    metrics_files = [f for f in os.listdir(results_dir) if f.endswith("_metrics.json")]
    class_files = [f for f in os.listdir(results_dir) if f.endswith("_classification.json")]
    
    if not metrics_files:
        print(f"No results found in {results_dir}. Copy your JSON files there first!")
        return

    # 1. Parse Data
    all_metrics = []
    for f in metrics_files:
        device_label = f.split("_metrics")[0].upper()
        with open(os.path.join(results_dir, f), "r") as jf:
            data = json.load(jf)
            data["device_label"] = device_label
            # Find matching classification
            matching_class = f.replace("_metrics", "_output_cls_token_classification")
            if matching_class in class_files:
                with open(os.path.join(results_dir, matching_class), "r") as cf:
                    c_data = json.load(cf)
                    data["confidence"] = c_data["confidence"]
                    data["prediction"] = c_data["prediction"]
            else:
                data["confidence"] = 0
                data["prediction"] = "N/A"
            all_metrics.append(data)
            
    df = pd.DataFrame(all_metrics)
    plt.switch_backend('Agg')
    
    # Process each model separately
    for model_name in df["model"].unique():
        m_df = df[df["model"] == model_name]
        print(f"\nProcessing Model: {model_name.upper()}")
        
        # --- CHART 1: LATENCY ---
        fig1, ax1 = plt.subplots(figsize=(14, 8))
        devices = m_df["device_label"].unique()
        modes = ["draft", "predictor", "dense", "naive"] # Preferred order
        actual_modes = [m for m in modes if m in m_df["mode"].unique()]
        
        x = np.arange(len(actual_modes))
        width = 0.8 / len(devices)
        
        for i, dev in enumerate(devices):
            dev_df = m_df[m_df["device_label"] == dev]
            lats = []
            for m in actual_modes:
                val = dev_df[dev_df["mode"] == m]["avg_latency"].values
                lats.append(val[0] if len(val) > 0 else 0)
            
            bars = ax1.bar(x + i*width, lats, width, label=dev)
            for bar in bars:
                h = bar.get_height()
                if h > 0: ax1.text(bar.get_x() + bar.get_width()/2, h + 0.05, f'{h:.2f}s', ha='center', va='bottom', fontsize=12)

        ax1.set_ylabel("Latency (Seconds per inference/token)")
        ax1.set_title(f"{model_name.upper()} PERFORMANCE ACROSS HARDWARE", fontsize=20, pad=20)
        ax1.set_xticks(x + width * (len(devices)-1) / 2)
        ax1.set_xticklabels([m.upper() for m in actual_modes], fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        plt.savefig(f"edge_latency_{model_name}.png", dpi=150)
        print(f"- Generated: edge_latency_{model_name}.png")

        # --- CHART 2: CLAY VISUAL PROOF ---
        if model_name == "clay":
            for i, row in m_df.iterrows():
                dev = row["device_label"]
                mode = row["mode"]
                if mode != "predictor": continue # Show predictor as the proof
                
                # Try to find heatmap and image
                h_path = os.path.join(results_dir, f"{dev.lower()}_output_heatmap.npy")
                # We assume the user copied the source image to edge_results and named it dev_image.png
                # or just use the same name as recorded.
                img_name = row["source_image"]
                img_path = os.path.join(results_dir, img_name)
                # Fallback: check if it's just in the folder with device prefix
                if not os.path.exists(img_path):
                    img_path = os.path.join(results_dir, f"{dev.lower()}_{img_name}")

                if os.path.exists(h_path) and os.path.exists(img_path):
                    print(f"  -> Generating Visual Proof for {dev}...")
                    h_map = np.load(h_path)
                    rgb = Image.open(img_path).convert("RGB")
                    rgb_np = np.array(rgb)
                    
                    fig_v, (ax_img, ax_h, ax_over) = plt.subplots(1, 3, figsize=(24, 8), facecolor='white')
                    
                    # 1. Original
                    ax_img.imshow(rgb)
                    ax_img.set_title(f"INPUT ({dev})\n{row['prediction']}", fontsize=22, fontweight='bold')
                    ax_img.axis('off')
                    
                    # 2. Raw Heatmap
                    h_vis = (h_map - h_map.min()) / (h_map.max() - h_map.min() + 1e-8)
                    ax_h.imshow(h_vis, cmap='magma', interpolation='nearest')
                    ax_h.set_title("MODEL ACTIVATIONS", fontsize=22, fontweight='bold')
                    ax_h.axis('off')
                    
                    # 3. Overlay
                    scale = rgb_np.shape[0] / h_map.shape[0]
                    h_zoomed = zoom(h_map, scale, order=1)
                    h_zoomed = (h_zoomed - h_zoomed.min()) / (h_zoomed.max() - h_zoomed.min() + 1e-8)
                    ax_over.imshow(rgb)
                    ax_over.imshow(h_zoomed, cmap='magma', alpha=0.4, interpolation='bilinear')
                    ax_over.set_title("SPATIAL ALIGNMENT", fontsize=22, fontweight='bold')
                    ax_over.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(f"edge_visual_proof_{dev.lower()}.png", dpi=150)
                    print(f"  - Generated: edge_visual_proof_{dev.lower()}.png")

    print("\nDone! All multi-model edge charts generated.")

if __name__ == "__main__":
    graph_comparison()
