# LLM in a Flash (Implementation)

An implementation of the "LLM in a Flash: Efficient Out-of-Memory Inference" research paper by Apple researchers. This project enables running Large Language Models (LLMs) on devices with memory constraints by efficiently offloading weights and using sparse activation prediction.

## 🚀 New Features (v2.0)

### 1. Multi-Model Support
*   **Llama 3 8B**: Full support for Llama 3's SwiGLU architecture and Gated MLPs.
*   **Clay v1.5 (Radar/SAR)**: Added proof-of-concept integration for the Clay Earth Observation model, allowing massive radar foundation models to run on field devices.

### 2. Speculative Decoding (+Draft Models)
*   Boost inference speeds by ~40% using small draft models (e.g., TinyLlama-1.1B for Llama 3 or OPT-125M for OPT-6.7B).
*   Toggle via `--mode speculative` in the benchmark or chat scripts.

### 3. Portable Predictors (Model Agnostic)
*   **`train_portable_predictor.py`**: A new, high-performance training script that works with any HuggingFace transformer.
*   **Zero-Recompile**: Predictors are exported with JSON metadata, allowing the C++ engine to reconfigure itself for different models without being rebuilt.

### 4. Hardware Optimization (ARM NEON)
*   The C++ core now uses **ARM NEON SIMD intrinsics**, providing significant speedups on Raspberry Pi 4/5 and Samsung S24 Ultra (Termux).

## 📊 Performance & Accuracy
*   **`performance_opt.png`**: Speed benchmarks for the OPT architecture.
*   **`performance_llama.png`**: Demonstrates how Llama-3 8B bypasses the "Memory Wall" (8GB OOM) using SSD streaming.
*   **`performance_clay_radar.png`**: Performance projections for the Clay v1.5 model.
*   **`accuracy_comparison.png`**: Shows why Sparse Prediction is superior to INT4 Quantization for retaining factual coherence.
## 🛠 Architecture

### The "Memory Wall" & Edge Performance
This project was specifically designed to bypass the memory limitations of edge hardware. 
*   **Traditional Approach:** A standard PyTorch implementation of the Clay v1.5 model (5GB) requires ~8GB of free RAM to load and run inference. On a Raspberry Pi 4B (8GB) or a mobile phone, this often triggers an OOM (Out of Memory) crash before the first prediction is even made.
*   **Flash Engine Approach:** By offloading the heavy Feed-Forward (FFN) weights to SSD and using a **Low-Rank Predictor**, we reduce the active RAM footprint by **~80%**. 
*   **Result:** You can run full-precision foundation models on hardware where they previously could not even be loaded.

### C++ Engine (`engine.cpp`)
...

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

### 3. Bundling & Extraction
Bundle the OPT-6.7B FFN weights and extract the resident layers (Attention/Norms) for memory efficiency:
```bash
python bundle_ffn.py
python layer_extractor.py
```

### 4. Running Inference
Run the model in an interactive session:
```bash
python chat.py --mode predictor
```
*Note: Use `--top_k` and `--threshold` to tune performance vs accuracy.*


## 📊 Performance & Accuracy
- Performance metrics can be generated using `speed_test.py`.
- Accuracy verification is handled by `benchmark_accuracy.py`.

## 📚 References
- **Original Paper:** [LLM in a Flash: Efficient Out-of-Memory Inference](LLMinaFlash.pdf) (Alizadeh et al., 2023)
- **Model Architecture:** Facebook/Meta OPT-6.7B

## ⚖️ License
This project is licensed under the [MIT License](LICENSE).
