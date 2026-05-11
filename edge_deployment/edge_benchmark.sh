#!/bin/bash

# --- LLM IN A FLASH: Edge Benchmark Suite ---
IMAGE="sample_satellite.npy"
RUN_LLM=false
RUN_CLAY=true
USE_GCP=false

# Parse flags
for arg in "$@"
do
    case $arg in
        --llm)
        RUN_LLM=true
        RUN_CLAY=false
        shift
        ;;
        --all)
        RUN_LLM=true
        RUN_CLAY=true
        shift
        ;;
        --gcp)
        USE_GCP=true
        shift
        ;;
    esac
done

echo "==============================================="
echo "   STARTING EDGE DEPLOYMENT BENCHMARKS        "
echo "==============================================="

# --- 0. PREPARE DATA ---
if [ ! -f "$IMAGE" ]; then
    echo "Generating multispectral sample data..."
    python3 generate_sample_ms.py
fi

# --- 1. CLAY BENCHMARKS ---
if [ "$RUN_CLAY" = true ] ; then
    echo -e "\n>>> RUNNING CLAY VISION TRANSFORMER BENCHMARKS <<<"
    
    echo -e "\n[1/3] CLAY DRAFT MODE (Block Skipping)..."
    OPENBLAS_VERBOSE=0 OMP_WAIT_POLICY=PASSIVE python edge_clay.py --mode draft --image $IMAGE

    echo -e "\n[2/3] CLAY PREDICTOR MODE (Sparse Streaming)..."
    OPENBLAS_VERBOSE=0 OMP_WAIT_POLICY=PASSIVE python edge_clay.py --mode predictor --image $IMAGE

    echo -e "\n[3/3] CLAY DENSE MODE (Standard PyTorch Baseline)..."
    echo "Note: This mode is expected to trigger an OOM (Out of Memory) crash on Pi 4B."
    OPENBLAS_VERBOSE=0 OMP_WAIT_POLICY=PASSIVE python edge_clay.py --mode dense --image $IMAGE
fi

# --- 2. LLM BENCHMARKS ---
if [ "$RUN_LLM" = true ] ; then
    echo -e "\n>>> RUNNING LLM BENCHMARKS (OPT-6.7b) <<<"
    
    # Ensure we use the bundled local cache for offline stability
    LLM_ARGS="--cache ./hf_cache"
    
    if [ "$USE_GCP" = true ] ; then
        echo "Using GCP Trained Predictors..."
        LLM_ARGS="$LLM_ARGS --predictor opt_gcp_predictors.bin --layers ./opt_6_7b_layers --ffn_bin opt_6_7b_bundled_ffn.bin"
    fi

    echo -e "\n[1/3] LLM DRAFT MODE (Fastest)..."
    echo "What is the capital of France?" | OPENBLAS_VERBOSE=0 OMP_WAIT_POLICY=PASSIVE python chat.py --mode draft $LLM_ARGS
    
    echo -e "\n[2/3] LLM PREDICTOR MODE (Our Contribution)..."
    echo "What is the capital of France?" | OPENBLAS_VERBOSE=0 OMP_WAIT_POLICY=PASSIVE python chat.py --mode predictor $LLM_ARGS
    
    echo -e "\n[3/3] LLM NAIVE MODE (Dense Baseline)..."
    echo "What is the capital of France?" | OPENBLAS_VERBOSE=0 OMP_WAIT_POLICY=PASSIVE python chat.py --mode naive $LLM_ARGS
fi

echo -e "\n==============================================="
echo "   BENCHMARKS COMPLETE                        "
echo "==============================================="
echo "Next: Use 'bash rename_results.sh <device>' then 'scp' back to your laptop."
