#!/bin/bash
# flash_clay.sh: The Zero-Overhead Orchestrator for Clay Benchmarks.

# Ensure we use the correct python path
PY="/home/randy/miniconda3/envs/llm-flash-v2/bin/python"

echo "=== STARTING CLAY FLASH BENCHMARK SUITE (Zero-RAM Overhead Mode) ==="

# 1. Run each mode in a completely fresh process
# This ensures 100% RAM recovery between modes.
MODES=("quantized" "naive_ssd" "oracle" "predictor" "draft")

for mode in "${MODES[@]}"
do
    echo -e "\n[Mode: ${mode^^}] Launching..."
    $PY benchmark_clay.py --mode $mode
    
    # Tiny pause to let the OS finish reclaiming memory
    sleep 2
done

# 2. Generate the final visual comparison
echo -e "\n=== GENERATING VISUAL PROOF ==="
$PY run_clay_suite.py --viz_only

echo -e "\nDone! Open 'benchmark_visual_comparison.png' for the final results."
