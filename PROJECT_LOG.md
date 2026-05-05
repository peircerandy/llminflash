# LLM-in-a-Flash Project Log

This log documents the major implementation choices, architectural changes, and bug fixes made during the replication of the Apple "LLM in a Flash" paper.

## Phase 1: Core Scaffolding
*   **Choice:** Chose `mmap()` for SSD weight streaming.
*   **Reason:** Allows the OS to handle the heavy lifting of page faults and data fetching, enabling zero-copy access to the 14GB weight file without exceeding 16GB of DRAM.
*   **Architecture:** Implemented "Row-Column Bundling." Interleaved `fc1` and `fc2` weights in a single binary file to ensure that when a neuron is activated, its input row and output column are read sequentially from the SSD, maximizing PCIe Gen 4 throughput.

## Phase 2: Hybrid Compute & Monkey Patching
*   **Choice:** Used PyTorch for Attention/Norms and C++ for FFN.
*   **Reason:** The Attention and LayerNorm layers are small enough to fit in the 6GB VRAM of the RTX 3060. Offloading the massive FFN layers to C++ allows us to bypass the 16GB system RAM bottleneck using NVMe streaming.
*   **Choice:** "Monkey-patching" the `OPTForCausalLM` module.
*   **Reason:** Allows us to reuse Hugging Face's high-quality Tokenizer and generation infrastructure while swapping out the specific bottlenecks for our custom engine.

## Phase 3: Sparsity & Sliding Window
*   **Choice:** Re-implemented the Sliding Window Cache ($k=5$).
*   **Reason:** As per Section 3.1 of the paper, keeping recently activated neurons in DRAM reduces SSD reads by ~90% for subsequent tokens, dramatically speeding up generation.
*   **Major Bug Fix (The `<s>` Loop):** Discovered that Python was garbage collecting the bias tensors.
*   **Fix:** Added a strong reference to `self.fc1_bias` in the `FlashFFN` module. This prevented C++ from reading random corrupted memory and fixed the infinite BOS repetition loop.

## Phase 4: Expert Predictors & Speculative Decoding
*   **Choice:** Integrated partner's trained predictors with variable ranks (128 and 1024).
*   **Reason:** Final layers in OPT are "sensitive" and require higher rank (1024) for accurate activation prediction, as detailed in the paper's Appendix B.
*   **Major Bug Fix (Benchmark Hang):** Fixed cache thrashing in Oracle mode.
*   **Fix:** Bypassed the Sliding Window for modes that require 100% neuron activation (Naive and Oracle). Caching all 16,384 neurons was causing the cache to overflow and triggering thousands of random random SSD reads per token.

## Phase 5: Final Optimization & Coherence
*   **Choice:** Switched to deterministic (greedy) decoding for verification, then re-enabled Top-P sampling.
*   **Reason:** Deterministic decoding helped prove the math was correct. Sampling was then added back to ensure a natural, non-repetitive "chatbox" experience.

## Phase 6: Memory Portability & Fine-Grained Parallelism
*   **Choice:** Implemented "Surgical Layer Extraction" via `layer_extractor.py`.
*   **Reason:** Standard `torch.load` attempts to pull entire 7GB shards into RAM, which is fatal on systems with 8GB total RAM (like the Raspberry Pi 4B target). By splitting the model into 32 small per-layer files (~150MB each) and loading them sequentially, we keep peak RAM usage well below the 8GB limit.
*   **Optimization:** Parallelized the predictor scoring loop in `engine.cpp` using OpenMP.
*   **Reason:** The previous sequential scoring was the primary generation bottleneck. Parallelizing across 16,384 neurons even for single tokens significantly reduced latency.
*   **Optimization:** Implemented "Ordered SSD Loading."
*   **Reason:** By sorting active neuron indices before reading from the memory-mapped FFN file, we ensure that SSD/NVMe access is mostly sequential, minimizing controller overhead and maximizing throughput.
*   **Feature:** Added a configurable `threshold` parameter to the predictor.
*   **Reason:** Allows including "on-the-fence" neurons that `Top-K` might miss, significantly improving model coherence and eliminating the accuracy drop compared to quantized baselines.

