# LLM in a Flash (Implementation)

This repository contains an implementation of the "LLM in a Flash: Efficient Out-of-Memory Inference" paper, which enables running large language models (LLMs) on devices with limited memory by efficiently offloading weights and using sparse activation prediction.

## Project Overview

The project uses a hybrid C++/Python approach:
- **C++ Engine:** Handles high-performance FFN (Feed-Forward Network) execution and efficient memory management.
- **Python Layer:** Manages model loading (using HuggingFace Transformers), predictor training, and high-level inference logic.
- **Sparse Predictor:** A low-rank neural network that predicts which neurons in the FFN will activate, reducing the amount of data that needs to be loaded from storage.

## Key Components

- `engine.cpp`: The core C++ engine for executing FFN layers.
- `chat.py`: The main entry point for interacting with the model.
- `train_predictor.py`: Script to train the low-rank activation predictors.
- `bundle_ffn.py`: Utility to prepare and bundle FFN weights for the C++ engine.
- `preprocessing.py`: Data preprocessing for training.
- `test_inference.py`: Verification script for model output.
- `Makefile`: Build configuration for the C++ library.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- HuggingFace Transformers & Accelerate
- G++ with OpenMP support

### Building the C++ Engine

```bash
make
```
This will generate `libengine.so`, which is used by the Python layer via `ctypes`.

### Usage

1. **Prepare the model:** Bundle the FFN weights.
2. **Train the predictor:** Use `train_predictor.py` to create the sparse activation models.
3. **Run inference:** Use `chat.py` or `test_inference.py`.

## References

- [LLM in a Flash: Efficient Out-of-Memory Inference](LLMinaFlash.pdf) (Original Paper)

## License

*Specify your license here (e.g., MIT, Apache 2.0).*
