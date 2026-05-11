import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from scipy.ndimage import zoom

def graph_comparison(results_dir="edge_results"):
    print("=== GENERATING PROFESSIONAL MULTI-MODEL EDGE PERFORMANCE SUITE (V2) ===")
    
    metrics_files = [f for f in os.listdir(results_dir) if f.endswith("_metrics.json")]
    class_files = [f for f in os.listdir(results_dir) if f.endswith("_classification.json")]
    
    if not metrics_files:
        print(f"No results found in {results_dir}. Copy your JSON files there first!")
        return

    # 1. Parse Data
    all_metrics = []
    for f in metrics_files:
        # Extract device and mode from filename, e.g., "pi4_metrics_predictor.json"
        # Logic: find last '_' and split
        parts = f.replace(".json", "").split("_")
        device_label = parts[0].upper()
        
        with open(os.path.join(results_dir, f), "r") as jf:
            data = json.load(jf)
            data["device_label"] = device_label
            # Find matching classification (if exists)
            # Filename for classification is usually {device}_output_cls_token_classification.json
            matching_class = f"{parts[0].lower()}_output_cls_token_classification.json"
            if matching_class in class_files:
                with open(os.path.join(results_dir, matching_class), "r") as cf:
                    c_data = json.load(cf)
                    data["confidence"] = c_data["confidence"]
                    data["prediction"] = c_data["prediction"]
            else:
                data.setdefault("confidence", 0)
                data.setdefault("prediction", "N/A")
            all_metrics.append(data)
            
    df = pd.DataFrame(all_metrics)
    plt.switch_backend('Agg')
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold'})

    # Process each model separately
    for model_name in df["model"].unique():
        m_df = df[df["model"] == model_name]
        print(f"\nProcessing Model: {model_name.upper()}")
        
        # --- CHART 1: LATENCY / SPEED ---
        fig1, ax1 = plt.subplots(figsize=(14, 8), facecolor='white')
        devices = sorted(m_df["device_label"].unique())
        modes = ["draft", "predictor", "dense", "naive"] 
        actual_modes = [m for m in modes if m in m_df["mode"].unique()]
        
        x = np.arange(len(actual_modes))
        width = 0.8 / len(devices)
        
        for i, dev in enumerate(devices):
            dev_df = m_df[m_df["device_label"] == dev]
            vals = []
            for m in actual_modes:
                row = dev_df[dev_df["mode"] == m]
                if not row.empty:
                    # For OPT, we might want to show TPS instead of raw latency
                    if model_name == "opt" and "tps" in row:
                        vals.append(row["tps"].values[0])
                    else:
                        vals.append(row["avg_latency"].values[0])
                else:
                    vals.append(0)
            
            label_suffix = " (TPS)" if model_name == "opt" else " (s)"
            bars = ax1.bar(x + i*width, vals, width, label=dev)
            for bar in bars:
                h = bar.get_height()
                if h > 0: ax1.text(bar.get_x() + bar.get_width()/2, h + (h*0.02), f'{h:.2f}', ha='center', va='bottom', fontsize=12)

        ylabel = "Tokens Per Second (Higher is Better)" if model_name == "opt" else "Inference Latency (Seconds - Lower is Better)"
        ax1.set_ylabel(ylabel, fontsize=16)
        ax1.set_title(f"{model_name.upper()} SPEED ACROSS HARDWARE", fontsize=22, pad=25)
        ax1.set_xticks(x + width * (len(devices)-1) / 2)
        ax1.set_xticklabels([m.upper() for m in actual_modes], fontweight='bold')
        ax1.legend(fontsize=14)
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"edge_latency_{model_name}.png", dpi=150)
        print(f"- Generated: edge_latency_{model_name}.png")

        # --- CHART 2: SEMANTIC FIDELITY (CONFIDENCE) ---
        fig2, ax2 = plt.subplots(figsize=(12, 7), facecolor='white')
        # Filter for Predictor mode to show it's accurate everywhere
        perf_df = m_df[m_df["mode"] == "predictor"]
        if not perf_df.empty:
            bars2 = ax2.bar(perf_df["device_label"], perf_df["confidence"] * 100, color='#3498db', alpha=0.7)
            ax2.set_ylabel("Model Confidence (%)", fontsize=16)
            ax2.set_ylim(0, 110)
            ax2.set_title(f"{model_name.upper()} SEMANTIC PRESERVATION\n(Laptop vs. Edge)", fontsize=20, pad=25)
            for bar in bars2:
                h = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, h + 2, f'{h:.1f}%', ha='center', fontweight='bold', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"edge_fidelity_{model_name}.png", dpi=150)
            print(f"- Generated: edge_fidelity_{model_name}.png")

        # --- CHART 3: CLAY VISUAL PROOFS (One per device) ---
        if model_name == "clay":
            for _, row in m_df.iterrows():
                dev = row["device_label"]
                mode = row["mode"]
                if mode != "predictor": continue 
                
                h_path = os.path.join(results_dir, f"{dev.lower()}_output_heatmap.npy")
                img_name = row.get("source_image", "sample_satellite.png")
                img_path = os.path.join(results_dir, img_name)
                if not os.path.exists(img_path):
                    # Try with prefix
                    img_path = os.path.join(results_dir, f"{dev.lower()}_{img_name}")

                if os.path.exists(h_path) and os.path.exists(img_path):
                    print(f"  -> Reconstructing Visual Proof for {dev}...")
                    h_map = np.load(h_path)
                    
                    if img_path.endswith(".npy"):
                        ms_data = np.load(img_path)
                        # Expecting (10, H, W) or (B, 10, H, W)
                        if ms_data.ndim == 4: ms_data = ms_data[0]
                        # Sentinel-2 RGB indices: Red=2, Green=1, Blue=0 in our 10-ch stack
                        rgb_np = ms_data[[2, 1, 0], :, :].transpose(1, 2, 0)
                        # Normalize for display
                        rgb_np = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min() + 1e-8)
                        rgb = Image.fromarray((rgb_np * 255).astype(np.uint8))
                    else:
                        rgb = Image.open(img_path).convert("RGB")
                        rgb_np = np.array(rgb)
                    
                    fig_v, (ax_img, ax_h, ax_over) = plt.subplots(1, 3, figsize=(24, 8), facecolor='white')
                    ax_img.imshow(rgb); ax_img.set_title(f"INPUT ({dev})\nPred: {row.get('prediction', 'N/A')}", fontsize=24, fontweight='bold'); ax_img.axis('off')
                    
                    h_vis = (h_map - h_map.min()) / (h_map.max() - h_map.min() + 1e-8)
                    ax_h.imshow(h_vis, cmap='magma', interpolation='nearest'); ax_h.set_title("MODEL FEATURES (3KB)", fontsize=24, fontweight='bold'); ax_h.axis('off')
                    
                    scale = rgb_np.shape[0] / h_map.shape[0]
                    h_zoomed = zoom(h_map, scale, order=1)
                    h_zoomed = (h_zoomed - h_zoomed.min()) / (h_zoomed.max() - h_zoomed.min() + 1e-8)
                    ax_over.imshow(rgb)
                    ax_over.imshow(h_zoomed, cmap='magma', alpha=0.4, interpolation='bilinear')
                    ax_over.set_title("SPATIAL ALIGNMENT", fontsize=24, fontweight='bold'); ax_over.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(f"edge_visual_proof_{dev.lower()}.png", dpi=150)
                    print(f"    - Generated: edge_visual_proof_{dev.lower()}.png")

        # --- CHART 4: DATA REDUCTION (One per model) ---
        if "input_img_kb" in m_df.columns and "telemetry_kb" in m_df.columns:
            sample = m_df.iloc[0]
            fig4, ax4 = plt.subplots(figsize=(10, 6), facecolor='white')
            sizes = [sample["input_img_kb"], sample["telemetry_kb"]]
            labels = [f"Raw Data ({sample['input_img_kb']:.1f} KB)", f"Telemetry ({sample['telemetry_kb']:.1f} KB)"]
            ax4.pie(sizes, labels=labels, autopct='%1.2f%%', colors=['#e74c3c', '#2ecc71'], startangle=140, explode=(0, 0.3))
            ax4.set_title(f"{model_name.upper()} DATA REDUCTION IMPACT\n({sample.get('data_reduction_ratio', 0):.1f}x smaller)", fontsize=20, pad=20)
            plt.savefig(f"edge_reduction_{model_name}.png", dpi=150)
            print(f"- Generated: edge_reduction_{model_name}.png")

    print("\nDone! Full multi-model edge analysis completed.")

if __name__ == "__main__":
    graph_comparison()
