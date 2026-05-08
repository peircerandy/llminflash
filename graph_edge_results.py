import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def graph_comparison(results_dir="edge_results"):
    all_data = []
    
    # 1. Scan for all JSON files (laptop_metrics.json, pi_metrics.json, etc)
    files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    
    if not files:
        print(f"No results found in {results_dir}. Copy your JSON files there first!")
        return

    for f in files:
        with open(os.path.join(results_dir, f), "r") as jf:
            data = json.load(jf)
            # Use filename as the 'Device Name' if not in JSON
            data["label"] = f.replace("_metrics.json", "").upper()
            all_data.append(data)
            
    df = pd.DataFrame(all_data)
    
    # 2. Plot Latency Comparison
    plt.figure(figsize=(12, 8))
    plt.switch_backend('Agg')
    plt.rcParams.update({'font.size': 16, 'font.weight': 'bold'})
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(df)))
    bars = plt.bar(df["label"], df["avg_latency"], color=colors, alpha=0.8)
    
    plt.ylabel("Inference Latency (seconds)", fontsize=18)
    plt.title("CROSS-DEVICE PERFORMANCE: LAPTOP vs. EDGE", fontsize=22, pad=20)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add labels on top
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.1, f'{h:.2f}s', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig("edge_performance_comparison.png", dpi=150)
    print("\nGenerated 'edge_performance_comparison.png'!")
    print(f"Successfully compared {len(df)} devices.\n")

if __name__ == "__main__":
    graph_comparison()
