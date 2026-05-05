"""
run_all_benchmarks.py: Automation script for full multi-model performance analysis.

This script iterates through all models (OPT-6.7B, Llama3-8B) and all execution 
modes (Quantized, Predictor, Speculative) and generates the final performance 
spreadsheets for your class presentation.
"""

import subprocess
import os
import pandas as pd

MODELS = ["opt-6.7b", "llama3-8b"]

def run():
    print("=== Starting Full LLM-in-a-Flash Benchmark Suite ===")
    
    # 1. Rebuild engine just in case
    print("\n[1/3] Building Optimized C++ Engine...")
    subprocess.run(["make"], check=True)

    # 2. Run benchmarks for each model
    all_results = []
    for model in MODELS:
        print(f"\n[2/3] Benchmarking Model: {model.upper()}")
        # We skip quantized for Llama 3 on 8GB systems to prevent crash, 
        # but you can toggle it if you have more RAM.
        skip_q = "--skip_quantized" if model == "llama3-8b" else ""
        
        cmd = f"conda run -n llm-flash --no-capture-output env PYTHONNOUSERSITE=1 python benchmark_comprehensive.py --model {model} {skip_q}"
        subprocess.run(cmd, shell=True)
        
        # Collect CSV result
        csv_path = f"{model}_comparison.csv"
        if os.path.exists(csv_path):
            all_results.append(pd.read_csv(csv_path))

    # 3. Generate Consolidated Graphs
    if all_results:
        print("\n[3/3] Generating Final Comparison Graphs...")
        subprocess.run(["python3", "generate_graphs.py"], check=True)
        
        final_df = pd.concat(all_results)
        final_df.to_csv("FINAL_PRESENTATION_DATA.csv", index=False)
        print("\n=== Success! ===")
        print("Final charts: performance_opt.png, performance_llama.png, accuracy_comparison.png")
        print("Consolidated Data: FINAL_PRESENTATION_DATA.csv")
    else:
        print("\nError: No results gathered.")

if __name__ == "__main__":
    run()
