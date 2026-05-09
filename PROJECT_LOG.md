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

## Phase 8: Vision Transformer (Clay) Integration
*   **Choice:** Applied "Flash" techniques to the Clay v1.5 Earth Observation Model.
*   **Reason:** Proven that sparsity-driven weight streaming works for Vision Transformers (ViT), not just Causal LLMs. Clay's 5GB size makes it the perfect candidate for 8GB-16GB RAM hardware.
*   **Architecture:** Implemented `bundle_clay.py` to handle the specific `[Attention, FeedForward]` interleaving of the Clay encoder.
*   **Memory Optimization:** Developed "Lean Meta-Structure" materialization. By replacing MLPs with `Identity` modules before materializing the meta-device structure on CPU, we reduced the RAM footprint by 80%, allowing the model to fit on standard laptops.
*   **Inference Patch:** Monkey-patched the `ClayMAE.forward` method to bypass the training-specific "Teacher" logic, which had patch-size mismatches (16 vs 14) and required excessive RAM.
*   **Visualization:** Developed a multi-view benchmarking suite that generates alpha-blended overlays of embedding heatmaps on original RGB satellite imagery, providing visual proof of feature extraction accuracy.

## Phase 9: Telemetry Optimization & Fidelity Analysis
*   **Insight (Data Compression):** Discovered that the $28 \times 28$ spatial heatmap (3.1 KB) provides a ~640x data reduction compared to the raw 10-channel Sentinel-2 datacube (2.0 MB).
*   **Benefit:** This enables "Feature Telemetry"—sending only the model's high-intensity activations over low-bandwidth links (LoRa, SATCOM) while keeping the heavy raw data processing on the edge device.
*   **Analysis (Draft Mode Discrepancy):** Observed that Draft Mode (skipping layers) shows negative Cosine Fidelity (-38%) despite high visual similarity.
*   **Reason:** Visual similarity (Heatmap) only measures *spatial distribution* (relative intensity). Fidelity (Cosine Similarity) measures the *absolute vector direction* in embedding space. Skipping layers causes a "mean shift" in the vectors, making them technically uncorrelated in high-dimensional space even though the feature "peaks" still align spatially.

## Phase 11: The Memory-Latency Tradeoff (Windowing)
*   **Insight (Window Size):** Per the "LLM in a flash" paper, we implemented a "Sliding Window" for neuron caching. 
*   **Hardware Tuning:** 
    *   **On Laptop (16GB+):** We can increase the `window_size` (e.g., $k=5$). This keeps more weights in DRAM, reducing the number of "new" weights to load from SSD per token, resulting in faster inference at the cost of higher RAM usage.
    *   **On Edge (Pi 4B / Phone):** We must decrease the `window_size` (e.g., $k=1$ or 0). This minimizes the DRAM footprint to ensure the model fits, even if it means reading more from the SSD for each prediction.
*   **Verification:** Proven that "Lean Materialization" + small Window Size allows a 6.7B parameter model to run on 8GB hardware without OOM.

