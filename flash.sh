#!/bin/bash
# flash.sh: A simple wrapper to handle environment isolation and library paths.

# 1. Define paths
CONDA_ENV="llm-flash"
CONDA_BASE=$(conda info --base)
ENV_PATH="$CONDA_BASE/envs/$CONDA_ENV"

# 2. Set Environmental Variables for Isolation
# - PYTHONNOUSERSITE: Ignores ~/.local libraries to prevent version conflicts
# - LD_LIBRARY_PATH: Forces the use of newer Conda C++ libraries over old system ones
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$ENV_PATH/lib:$LD_LIBRARY_PATH"

# 3. Execute the command inside the conda environment
if [ $# -eq 0 ]; then
    echo "Usage: ./flash.sh <script.py> [args]"
    echo "Example: ./flash.sh chat.py --mode predictor"
    exit 1
fi

conda run -n "$CONDA_ENV" --no-capture-output python "$@"
