#!/bin/bash

# --- Rename Edge Results Utility ---
# Usage: bash rename_results.sh <device_name>
# Example: bash rename_results.sh pi4

DEVICE=$1

if [ -z "$DEVICE" ]; then
    echo "Usage: bash rename_results.sh <device_name>"
    echo "Example: bash rename_results.sh pi4"
    exit 1
fi

echo "Renaming result files with prefix: $DEVICE..."

# Target only specific result extensions to protect .sh and .py scripts
for f in edge_metrics_*.json edge_output_*.npy; do
    if [ -e "$f" ]; then
        # Rename "edge_metrics_predictor.json" -> "pi4_metrics_predictor.json"
        NEW_NAME="${DEVICE}_${f#edge_}"
        mv "$f" "$NEW_NAME"
        echo "  -> $f  =>  $NEW_NAME"
    fi
done

echo "Done! You can now scp these files to your laptop's 'edge_results/' folder."
