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

## Phase 7: Portable Predictors & Agnostic Training
*   **Choice:** Standardized the predictor format with a paired JSON metadata file.
*   **Reason:** Allows predictors to be trained on powerful multi-GPU machines and then copied to edge devices (Pi, S24) without recompiling the C++ engine for each specific architecture.
*   **Architecture:** Implemented `train_portable_predictor.py`. It uses dynamic HuggingFace config detection to support any Transformer architecture (OPT, Llama, Falcon, etc.) and exports weights as raw float32 buffers.
*   **Engine Update:** Added `set_predictor_layer_info` to `engine.cpp`, enabling the Python layer to "tell" the C++ core exactly where each layer's weights are and what rank they use, based on the JSON metadata.

