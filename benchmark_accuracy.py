import torch
import torch.nn as nn
import transformers
from transformers import OPTForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate.hooks import remove_hook_from_module
import ctypes
import os
import time
import gc
import logging

# --- Suppress HF Warnings safely ---
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
transformers.logging.set_verbosity_error() 
logging.getLogger("huggingface_hub").setLevel(logging.ERROR) # Mutes the unauthenticated warnings without breaking the cache!

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

# --- The Test Suite ---
TEST_QUESTIONS = [
    {"prompt": "Question: What is the capital of France?\nAnswer:", "expected": "Paris", "desc": "Basic Knowledge"},
    {"prompt": "Beginners BBQ Class Taking Place in Missoula! Do you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 22nd join World Class BBQ Champion, Tony Balay from Lonestar Smoke Rangers.", "expected": "teach", "desc": "Overfit Recall"},
]

def generate_graph(results):
    print("\nGenerating Benchmark Graph (presentation_benchmark.png)...")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        import pip
        print("Installing matplotlib for graphing...")
        pip.main(['install', 'matplotlib'])
        import matplotlib.pyplot as plt

    modes = list(results.keys())
    speeds = [results[m]["avg_tps"] for m in modes]
    accuracies = [results[m]["accuracy_pct"] for m in modes]
    times = [results[m]["avg_time"] for m in modes]

    plt.figure(figsize=(11, 7))
    
    # Use distinct colors for each architecture
    colors = ['#ffcc99', '#99ccff', '#ff9999', '#c2c2f0']
    bars = plt.bar([m.upper() for m in modes], speeds, color=colors[:len(modes)])
    
    plt.title('Hardware Architecture Performance vs. Accuracy (OPT-6.7B)', fontsize=15, fontweight='bold', pad=20)
    plt.ylabel('Tokens Per Second (Higher is Better)', fontsize=12)
    
    # Dynamically increase the Y-axis ceiling so our new, taller text blocks don't get cut off
    plt.ylim(0, max(speeds) * 1.35)
    
    # Get the baseline speed (Naive mode) to calculate the speedup multipliers
    baseline_speed = results.get("naive", {}).get("avg_tps", 0)

    # Annotate speed, accuracy, process time, AND relative speedup on top of the bars
    for bar, acc, t, m in zip(bars, accuracies, times, modes):
        yval = bar.get_height()
        offset = max(speeds) * 0.03
        
        # Calculate the relative speedup compared to Naive (Baseline)
        speedup_str = ""
        if baseline_speed > 0:
            multiplier = results[m]["avg_tps"] / baseline_speed
            if m == "naive":
                speedup_str = "\n(Baseline)"
            else:
                speedup_str = f"\n({multiplier:.2f}x Speedup)"

        # EXPLICIT LABELING: Now includes the mathematical multiplier!
        label = f"{yval:.2f} Tokens/Sec{speedup_str}\nTime: {t:.1f}s\nAcc: {acc}%"
        
        plt.text(bar.get_x() + bar.get_width()/2, yval + offset, label, ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig('presentation_benchmark.png', dpi=300)
    
    # Save raw stats to CSV for presentation backup
    import csv
    with open('presentation_stats.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        # EXPLICIT LABELING: Spelled out Tokens/Sec in the spreadsheet header
        writer.writerow(['Mode', 'Avg Tokens/Sec', 'Avg Time (s)', 'Accuracy %'])
        for m in modes:
            writer.writerow([m.upper(), f"{results[m]['avg_tps']:.2f}", f"{results[m]['avg_time']:.2f}", results[m]['accuracy_pct']])
            
    print("Graph saved successfully as 'presentation_benchmark.png'!")
    print("Raw statistics saved to 'presentation_stats.csv'!")

def run_suite():
    print("==================================================")
    print(" LLM IN A FLASH - AUTOMATED ACCURACY BENCHMARK ")
    print("==================================================\n")
    
    # We skip "standard" because OS disk swapping will likely crash your laptop.
    # But we added "naive" back in to show Apple's baseline vs the optimizations!
    modes_to_test = ["quantized", "naive", "oracle", "predictor"]
    
    # Dictionary to store results for graphing
    benchmark_results = {}
    # NEW: Store the table rows to print cleanly at the very end
    summary_table_rows = []

    for mode_name in modes_to_test:
        print("\n" + "="*50)
        print(f" LOADING {mode_name.upper()} MODE")
        print("="*50)
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_PATH, local_files_only=True)
        engine_ptr = None

        if mode_name == "quantized":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16 # CRITICAL FIX: Stabilizes 4-bit CUDA math
            )
            model = OPTForCausalLM.from_pretrained(
                MODEL_ID, quantization_config=bnb_config, device_map="auto",
                cache_dir=CACHE_PATH, local_files_only=True
            )
        else:
            engine_ptr = lib.init_engine(FFN_BIN_PATH)
            # CRITICAL FIX: Add the 'naive' mode integer mapping so the C++ engine knows what to do
            mode_int = {"predictor": 0, "oracle": 1, "naive": 2}[mode_name]

            custom_map = {"model.decoder.embed_tokens": "cuda:0", "model.decoder.embed_positions": "cuda:0", "model.decoder.final_layer_norm": "cuda:0", "lm_head": "cuda:0"}
            for i in range(32):
                p = f"model.decoder.layers.{i}"
                custom_map[f"{p}.self_attn"] = "cuda:0"
                custom_map[f"{p}.self_attn_layer_norm"] = "cuda:0"
                custom_map[f"{p}.final_layer_norm"] = "cuda:0"
                custom_map[f"{p}.fc1"] = "disk"
                custom_map[f"{p}.fc2"] = "disk"
            
            model = OPTForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map=custom_map, cache_dir=CACHE_PATH, offload_folder=OFFLOAD_PATH, local_files_only=True)

            with torch.no_grad():
                for i, layer in enumerate(model.model.decoder.layers):
                    z1 = torch.zeros(1, HIDDEN_SIZE, dtype=model.dtype, device=model.device)
                    fc1_b = layer.fc1(z1).squeeze().float().cpu().contiguous()
                    z2 = torch.zeros(1, 16384, dtype=model.dtype, device=model.device)
                    fc2_b = layer.fc2(z2).squeeze().to(model.device)
                    
                    layer.fc1 = FlashFFN(i, engine_ptr, fc1_b, fc2_b, mode_int)
                    layer.activation_fn = nn.Identity()
                    layer.fc2 = nn.Identity()
                    remove_hook_from_module(layer, recurse=True)

        mode_total_tps = 0
        mode_passes = 0
        mode_total_time = 0

        # Run the questions
        for q in TEST_QUESTIONS:
            print(f"\n--- Testing: {q['desc']} ---")
            print(f"You: {q['prompt']}")
            
            inputs = tokenizer(q["prompt"], return_tensors="pt").to(model.device)
            
            start_time = time.time()
            # Force exact 10 token limit and deterministic greedy search for standardized benchmarking
            outputs = model.generate(
                **inputs, 
                max_new_tokens=10, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False # CRITICAL FIX: Forces deterministic output and prevents sampling crashes
            )
            elapsed = time.time() - start_time
            
            # Extract only the newly generated text
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # NEW: Print the exact text output like chat.py!
            print(f"\nOPT [{mode_name.upper()}]: {generated_text}")
            
            tps = 10 / elapsed
            mode_total_tps += tps
            mode_total_time += elapsed
            
            # Check accuracy
            is_accurate = "PASS" if q["expected"].lower() in generated_text.lower() else "FAIL"
            if is_accurate == "PASS":
                mode_passes += 1
            
            # Save the result to print at the end of the script
            row = f"| {mode_name.upper():<12} | {q['desc']:<16} | {is_accurate:<10} | {tps:<10.2f} | {elapsed:<10.2f} |"
            summary_table_rows.append(row)

        # Save aggregate results for this mode
        avg_tps = mode_total_tps / len(TEST_QUESTIONS)
        avg_time = mode_total_time / len(TEST_QUESTIONS)
        accuracy_pct = int((mode_passes / len(TEST_QUESTIONS)) * 100)
        benchmark_results[mode_name] = {"avg_tps": avg_tps, "accuracy_pct": accuracy_pct, "avg_time": avg_time}

        # Memory cleanup between mode switches
        del model
        if engine_ptr:
            lib.destroy_engine(engine_ptr)
        gc.collect()
        torch.cuda.empty_cache()

    # NEW: Print the final beautiful summary table once all tests are completely finished!
    print("\n\n==================================================")
    print(" FINAL BENCHMARK SUMMARY TABLE ")
    print("==================================================")
    print(f"| {'Mode':<12} | {'Test Type':<16} | {'Accuracy':<10} | {'Tokens/Sec':<10} | {'Time (s)':<10} |")
    print("-" * 75)
    for row in summary_table_rows:
        print(row)
    print("-" * 75)

    # After testing all modes, generate the graph and CSV!
    generate_graph(benchmark_results)

if __name__ == "__main__":
    run_suite()