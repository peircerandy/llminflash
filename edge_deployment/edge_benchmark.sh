#!/bin/bash

# --- LLM IN A FLASH: Edge Benchmark Suite ---
IMAGE="sample_satellite.png"
RUN_LLM=false

# Parse flags
for arg in "$@"
do
    case $arg in
        --llm)
        RUN_LLM=true
        shift
        ;;
    esac
done

echo "==============================================="
echo "   STARTING EDGE DEPLOYMENT BENCHMARKS        "
echo "==============================================="

# 1. DRAFT MODE (Fastest)
echo -e "\n[1/3] RUNNING DRAFT MODE (Block Skipping)..."
OPENBLAS_VERBOSE=0 OMP_WAIT_POLICY=PASSIVE python edge_clay.py --mode draft --image $IMAGE

# 2. PREDICTOR MODE (Our Contribution)
echo -e "\n[2/3] RUNNING PREDICTOR MODE (Sparse Streaming)..."
OPENBLAS_VERBOSE=0 OMP_WAIT_POLICY=PASSIVE python edge_clay.py --mode predictor --image $IMAGE

# 3. DENSE MODE (The Baseline / Likely to fail)
echo -e "\n[3/3] RUNNING DENSE MODE (Standard PyTorch Baseline)..."
echo "Note: This mode is expected to trigger an OOM (Out of Memory) crash on Pi 4B."
OPENBLAS_VERBOSE=0 OMP_WAIT_POLICY=PASSIVE python edge_clay.py --mode dense --image $IMAGE

# Optional LLM Benchmark
if [ "$RUN_LLM" = true ] ; then
    echo -e "\n[OPTIONAL] RUNNING LLM BENCHMARKS (OPT-6.7b)..."
    
    echo -e "\n[1/3] LLM DRAFT MODE (Fastest)..."
    echo "What is the capital of France?" | OPENBLAS_VERBOSE=0 OMP_WAIT_POLICY=PASSIVE python chat.py --mode draft
    
    echo -e "\n[2/3] LLM PREDICTOR MODE (Our Contribution)..."
    echo "What is the capital of France?" | OPENBLAS_VERBOSE=0 OMP_WAIT_POLICY=PASSIVE python chat.py --mode predictor
    
    echo -e "\n[3/3] LLM NAIVE MODE (Dense Baseline)..."
    echo "What is the capital of France?" | OPENBLAS_VERBOSE=0 OMP_WAIT_POLICY=PASSIVE python chat.py --mode naive
fi

echo -e "\n==============================================="
echo "   BENCHMARKS COMPLETE                        "
echo "==============================================="
echo "Next: Use 'scp edge_metrics_*.json' back to your laptop to generate graphs."
