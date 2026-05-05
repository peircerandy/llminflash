# LLM-in-a-Flash Project Guide

This guide explains the purpose of each file in the repository to help you understand the architecture and determine which files are necessary for your final presentation.

## Core Implementation Files (KEEP)
These files form the heart of the "LLM in a Flash" replication.

*   **`engine.cpp`**: The high-performance C++ core. It handles the memory-mapped I/O from the SSD, the Sliding Window Cache (k=5), the parallel FFN math via OpenMP, and the partner's trained sparsity predictors.
*   **`chat.py`**: The main user interface. It orchestrates the hybrid CPU/GPU/NVMe pipeline, performs the monkey-patching of the OPT model, and provides the interactive terminal chat experience.
*   **`Makefile`**: The build script for `libengine.so`. It ensures the C++ code is compiled with optimal flags (`-O3`, `-march=native`, `-fopenmp`).
*   **`bundle_ffn.py`**: The most advanced weight preparation script. It uses the "Identity Matrix Hack" to force the OPT model to export its weights exactly as the math expects them, pre-formatted for SSD streaming.
*   **`train_predictor.py`**: The script used to train the Low-Rank Predictors (N -> 128/1024 -> M) on the C4 dataset, as described in Section 3.1 of the paper.

## Analysis & Benchmarking (KEEP)
*   **`benchmark_accuracy.py`**: The automated testing suite. It runs the model in different modes (Quantized, Naive, Oracle, Predictor, Speculative) and generates the comparison graphs for your presentation.
*   **`benchmark_predictors.py`**: A specialized tool to measure the "Recall" of your predictors compared to your partner's, ensuring the sparsity logic is catching the right neurons.
*   **`speed_test.py`**: A simple utility to measure exactly how many seconds it takes to generate a token in the current mode.

## Utilities & Preparation (KEEP)
*   **`layer_extractor.py`**: A memory-efficient tool for 8GB systems. It surgically extracts Attention/Norm layers into small files, allowing the model to load sequentially without hitting OOM limits.
*   **`zip_surgery.py`**: A low-level utility that explores the ZIP-archive structure of PyTorch `.bin` files to selectively extract tensors.
*   **`convert_partner_predictors.py`**: A bridge script that takes your partner's PyTorch `.pt` files and packs them into a single binary file for the C++ engine.
*   **`verify_fix.py`**: A quick-start test script to verify model coherence and speed after making engine changes.
*   **`requirements.txt`**: Lists the Python libraries (torch, transformers, etc.) needed to run the project.

## Experimental / Potentially Redundant (CLEANUP CANDIDATES)
*   **`magic_loader.py`**: An experimental script attempting to use custom unpicklers to skip massive tensors during `torch.load`.
*   **`preprocessing.py`**: Our early attempt at weight extraction. It has been largely superseded by the more robust `bundle_ffn.py`.
*   **`final_engine.py`**: A manual, Python-only implementation of the transformer loop used during debugging to fix the `<s>` repetition issues.
*   **`test_coherence.py`**: A small test script used to verify if the model was outputting real words during development.
*   **`test_inference.py`**: Another development test script used to debug the "stuck shell" issue with progress bars.
*   **`manual_test.py`**: An early experiment in manual forward-pass patching that didn't use the final `FlashFFN` class.
*   **`speed_test copy.py`**: Likely an accidental duplicate.

