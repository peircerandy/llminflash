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

echo "Renaming output files with prefix: $DEVICE..."

for f in edge_*; do
    if [ -e "$f" ]; then
        NEW_NAME="${DEVICE}_${f#edge_}"
        mv "$f" "$NEW_NAME"
        echo "  -> $f  =>  $NEW_NAME"
    fi
done

echo "Done! You can now scp these files to your laptop's 'edge_results/' folder."
