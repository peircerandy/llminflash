# LLM in a Flash (Implementation)

An implementation of the "LLM in a Flash: Efficient Out-of-Memory Inference" research paper by Apple researchers. This project enables running Large Language Models (LLMs) on devices with memory constraints by efficiently offloading weights and using sparse activation prediction.

## 🚀 Overview

The "LLM in a Flash" approach addresses the bottleneck of limited DRAM when running large models. It treats flash memory as the primary storage and uses two key techniques:
1.  **Windowing:** Reducing data transfer by reusing previously activated neurons.
2.  **Sparse Activation Prediction:** Using a low-rank predictor to guess which neurons in the Feed-Forward Network (FFN) will activate, loading only the necessary weights.

This repository provides a hybrid C++/Python implementation designed for the OPT-6.7B model architecture.

## 🛠 Architecture

### C++ Engine (`engine.cpp`)
A high-performance backend responsible for:
- Efficient memory-mapped I/O for weight loading.
- Parallelized FFN execution using OpenMP.
- Optimized matrix operations for sparse activations.

### Python Layer
- **`chat.py`**: Interactive CLI for model inference.
- **`train_predictor.py`**: Trains the `LowRankPredictor` (the "sparse predictor") using a bottleneck architecture (4096 -> 128 -> 16384).
- **`bundle_ffn.py`**: Pre-processes and bundles FFN weights into a custom binary format for the C++ engine.

## 📦 Getting Started

### Prerequisites
- **Compiler:** `g++` with OpenMP support.
- **Python:** 3.8+
- **Hardware:** Works best on systems where model weights exceed available GPU/System RAM.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:peircerandy/llminflash.git
    cd llminflash
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Build the C++ Engine:**
    ```bash
    make
    ```

## 📖 Usage

### 1. Data Preprocessing
Prepare your training data for the sparse predictor:
```bash
python preprocessing.py
```

### 2. Training the Predictors
Train the low-rank neural networks that predict neuron activation:
```bash
python train_predictor.py
```

### 3. Bundling Weights
Bundle the OPT-6.7B FFN weights for use with the flash engine:
```bash
python bundle_ffn.py
```

### 4. Running Inference
Run the model in an interactive session:
```bash
python chat.py
```

## 📊 Performance & Accuracy
- Performance metrics can be generated using `speed_test.py`.
- Accuracy verification is handled by `benchmark_accuracy.py`.

## 📚 References
- **Original Paper:** [LLM in a Flash: Efficient Out-of-Memory Inference](LLMinaFlash.pdf) (Alizadeh et al., 2023)
- **Model Architecture:** Facebook/Meta OPT-6.7B

## ⚖️ License
This project is licensed under the [MIT License](LICENSE).
