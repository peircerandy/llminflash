import subprocess
import os

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr: print(f"ERROR: {result.stderr}")

# 1. OPT-6.7B Predictor
run_cmd("conda run -n llm-flash --no-capture-output env PYTHONNOUSERSITE=1 python benchmark_comprehensive.py --model opt-6.7b --skip_quantized --tokens 5")

# 2. Llama-3 8B Predictor
run_cmd("conda run -n llm-flash --no-capture-output env PYTHONNOUSERSITE=1 python benchmark_comprehensive.py --model llama3-8b --skip_quantized --tokens 5")
