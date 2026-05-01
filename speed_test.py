import torch
import torch.nn as nn
from transformers import OPTForCausalLM, AutoTokenizer
from accelerate.hooks import remove_hook_from_module
import ctypes
import os
import time
import gc
import matplotlib.pyplot as plt

# --- Force HuggingFace Offline ---
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# --- Configuration ---
MODEL_ID = "facebook/opt-6.7b"
CACHE_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache"
OFFLOAD_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_offload"
FFN_BIN_PATH = b"/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_bundled_ffn.bin"
HIDDEN_SIZE = 4096

# --- C++ Engine Bindings ---
lib = ctypes.CDLL(os.path.abspath("./libengine.so"))
lib.init_engine.argtypes = [ctypes.c_char_p]
lib.init_engine.restype = ctypes.c_void_p
lib.execute_ffn_layer.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int
]
lib.destroy_engine.argtypes = [ctypes.c_void_p]

class FlashFFN(nn.Module):
    def __init__(self, layer_idx, engine_ptr, fc1_bias, fc2_bias, mode_int):
        super().__init__()
        self.layer_idx = layer_idx
        self.engine_ptr = engine_ptr
        self.mode_int = mode_int
        self.fc1_bias = fc1_bias
        self.fc1_bias_c = ctypes.cast(self.fc1_bias.data_ptr(), ctypes.POINTER(ctypes.c_float)) if fc1_bias is not None else None
        self.fc2_bias = fc2_bias

    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        device = hidden_states.device
        hidden_flat = hidden_states.view(-1, HIDDEN_SIZE).float().cpu().contiguous()
        num_tokens = hidden_flat.shape[0]
        out_flat = torch.zeros_like(hidden_flat)

        in_c = ctypes.cast(hidden_flat.data_ptr(), ctypes.POINTER(ctypes.c_float))
        out_c = ctypes.cast(out_flat.data_ptr(), ctypes.POINTER(ctypes.c_float))

        lib.execute_ffn_layer(self.engine_ptr, self.layer_idx, in_c, out_c, num_tokens, self.fc1_bias_c, self.mode_int)

        out_tensor = out_flat.to(device).to(hidden_states.dtype).view(orig_shape)
        if self.fc2_bias is not None:
            out_tensor = out_tensor + self.fc2_bias
        return out_tensor

def run_benchmark(mode_name, num_tokens=5):
    print(f"\n[{mode_name.upper()}] Initializing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_PATH, local_files_only=True)
    engine_ptr = None

    if mode_name == "quantized":
        # The GPT4All Method: Shrink the weights to 4-bit to fit perfectly in RAM
        print("  Loading 4-bit Quantized Model (BitsAndBytes)...")
        model = OPTForCausalLM.from_pretrained(
            MODEL_ID, 
            load_in_4bit=True, # <--- 4-Bit Compression Magic
            device_map="auto",
            cache_dir=CACHE_PATH,
            local_files_only=True
        )
    elif mode_name == "standard":
        # The Standard HF Method: Let it swap to the OS hard drive
        print("  Loading Standard Model (Warning: Heavy OS Disk Swapping)...")
        model = OPTForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, device_map="auto",
            cache_dir=CACHE_PATH, offload_folder=OFFLOAD_PATH, local_files_only=True
        )
    else:
        # The Apple Method: SSD Flash Streaming via C++
        print(f"  Loading Hardware Architecture ({mode_name.upper()})...")
        engine_ptr = lib.init_engine(FFN_BIN_PATH)
        mode_map = {"predictor": 0, "oracle": 1, "naive": 2}
        mode_int = mode_map[mode_name]

        custom_device_map = {"lm_head": "cuda:0", "model.decoder.embed_tokens": "cuda:0", "model.decoder.embed_positions": "cuda:0", "model.decoder.final_layer_norm": "cuda:0"}
        for i in range(32):
            prefix = f"model.decoder.layers.{i}"
            custom_device_map[f"{prefix}.self_attn"] = "cuda:0"
            custom_device_map[f"{prefix}.self_attn_layer_norm"] = "cuda:0"
            custom_device_map[f"{prefix}.final_layer_norm"] = "cuda:0"
            custom_device_map[f"{prefix}.fc1"] = "disk"
            custom_device_map[f"{prefix}.fc2"] = "disk"
        
        model = OPTForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map=custom_device_map, cache_dir=CACHE_PATH, offload_folder=OFFLOAD_PATH, local_files_only=True)

        with torch.no_grad():
            for i, layer in enumerate(model.model.decoder.layers):
                zero_fc1 = torch.zeros(1, HIDDEN_SIZE, dtype=model.dtype, device=model.device)
                fc1_bias = layer.fc1(zero_fc1).squeeze().float().cpu().contiguous()
                zero_fc2 = torch.zeros(1, 16384, dtype=model.dtype, device=model.device)
                fc2_bias = layer.fc2(zero_fc2).squeeze().to(model.device)
                
                layer.fc1 = FlashFFN(i, engine_ptr, fc1_bias, fc2_bias, mode_int)
                layer.activation_fn = nn.Identity()
                layer.fc2 = nn.Identity()
                remove_hook_from_module(layer, recurse=True)

    gc.collect()
    torch.cuda.empty_cache()

    print("  Warming up model...")
    inputs = tokenizer("The theory of relativity", return_tensors="pt").to(model.device)
    
    # Run once to initialize caches (not timed)
    _ = model.generate(**inputs, max_new_tokens=1)

    print(f"  Benchmarking {num_tokens} tokens...")
    start_time = time.time()
    _ = model.generate(**inputs, max_new_tokens=num_tokens, min_new_tokens=num_tokens)
    end_time = time.time()

    elapsed = end_time - start_time
    tps = num_tokens / elapsed
    print(f"  Result: {tps:.2f} Tokens/Second ({elapsed:.1f}s total)")

    # Clean up to prevent OOM on the next loop
    del model
    if engine_ptr:
        lib.destroy_engine(engine_ptr)
    gc.collect()
    torch.cuda.empty_cache()

    return tps

def generate_graph(results):
    print("\nGenerating Benchmark Graph (benchmark_chart.png)...")
    modes = list(results.keys())
    speeds = list(results.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(modes, speeds, color=['#ff9999','#ffcc99','#99ccff','#99ff99','#c2c2f0'])
    
    plt.title('LLM Inference Speeds by Hardware Architecture (OPT-6.7B on 16GB RAM)', fontsize=14)
    plt.ylabel('Tokens Per Second (Higher is Better)', fontsize=12)
    plt.xlabel('Execution Mode', fontsize=12)
    
    # Add the text values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')

    # Add a descriptive legend/text box
    desc = (
        "Standard: Python Disk Swapping (Baseline)\n"
        "Naive: C++ Reading 100% of SSD\n"
        "Oracle: C++ Sparse Streaming (Exact Math)\n"
        "Predictor: Apple's ML Sparse Streaming\n"
        "Quantized: 4-Bit RAM Compression (GPT4All)"
    )
    plt.figtext(0.15, 0.70, desc, fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})

    plt.tight_layout()
    plt.savefig('benchmark_chart.png', dpi=300)
    print("Graph saved successfully!")

if __name__ == "__main__":
    import pip
    try:
        import matplotlib
    except ImportError:
        print("Installing matplotlib for graphing...")
        pip.main(['install', 'matplotlib'])
        
    print("Starting LLM Hardware Benchmark Suite...")
    
    results = {}
    
    # Note: We skip 'standard' by default because 5 tokens would take ~30+ minutes.
    # We will manually assign it an estimated value based on your earlier 419s / 50 token run.
    results["Standard"] = 0.11 # (50 tokens / 419 seconds)
    
    # Test the real architectures
    for mode in ["naive", "oracle", "predictor", "quantized"]:
        try:
            tps = run_benchmark(mode, num_tokens=5)
            results[mode.capitalize()] = tps
        except Exception as e:
            print(f"Error benchmarking {mode}: {e}")
            results[mode.capitalize()] = 0.0

    generate_graph(results)